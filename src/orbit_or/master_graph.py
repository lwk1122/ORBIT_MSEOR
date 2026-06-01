import asyncio
import json
import logging
import sqlite3
from typing import Any, Dict, List, TypedDict as TypingTypedDict

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from . import analytics
from . import api
from . import topic_config
from .agents import SKYNET, get_agent, voting_agents
from .json_utils import extract_json_object
from .broker import (
    DEFAULT_MAX_TOKENS,
    PROFILE_MINIMAX,
    llm_call_with_web,
)
from .db import MAX_CONTENT_LEN
from .embedding import aget_embedding
from .server import run_subtopic_graph
from .structured_retry import retry_structured_output, usable_text_output

logger = logging.getLogger(__name__)

# Prevent background tasks from being garbage-collected before completion
_background_tasks: set[asyncio.Task] = set()

SUBTOPIC_CANDIDATE_COUNT = 3
SUBTOPIC_VOTE_CYCLE_LIMIT = 3
DECISION_PASS_RATIO = 2 / 3


def _topic_candidate_count(topic_id: int) -> int:
    return topic_config.get_int(topic_id, "subtopic_candidate_count")


def _topic_pass_ratio(topic_id: int) -> float:
    return topic_config.get_float(topic_id, "decision_pass_ratio")


def _topic_provider(topic_id: int, key: str, fallback_key: str = "") -> str:
    if not topic_id:
        return "minimax"
    try:
        return topic_config.get_provider_profile_for(topic_id, key, fallback_key)
    except sqlite3.OperationalError:
        raise
    except Exception as exc:
        logger.debug("[provider] Topic provider lookup failed for %s: %s", key, exc)
        return "minimax"


def _is_usable_json_text(text: str) -> bool:
    if not usable_text_output(text):
        return False
    try:
        return isinstance(_parse_json_object(text), dict)
    except Exception:
        return False


class TopicState(TypedDict, total=False):
    topic_id: int
    plan_id: int
    current_subtopic_id: int
    next_action: str
    topic_complete: bool
    deferred: bool


class VoteTally(TypingTypedDict):
    yes_votes: int
    successful_votes: int
    failed_votes: int


def _parse_json_object(output: str) -> Dict[str, Any]:
    result = extract_json_object(output)
    if isinstance(result, dict):
        return result
    return json.loads(output)


def _parse_plan_content(plan: Dict[str, Any] | None) -> List[Dict[str, str]]:
    if not plan:
        return []
    try:
        content = json.loads(plan["content"])
        if isinstance(content, list):
            return [item for item in content if isinstance(item, dict)]
    except Exception:
        pass
    return []


def _sanitize_subtopics(raw_subtopics: Any, limit: int) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    if not isinstance(raw_subtopics, list):
        return cleaned

    for item in raw_subtopics:
        if not isinstance(item, dict):
            continue
        summary = item.get("summary")
        detail = item.get("detail")
        if not isinstance(summary, str) or not summary.strip():
            continue
        if not isinstance(detail, str) or not detail.strip():
            continue
        cleaned.append(
            {"summary": summary.strip()[:200], "detail": detail.strip()[:500]}
        )
        if len(cleaned) >= limit:
            break
    return cleaned


async def ask_control_model(
    system_prompt: str,
    context: str,
    role: str,
    model: str = "",
    *,
    topic_id: int = 0,
) -> Dict[str, Any]:
    prompt = f"{system_prompt}\n\nHere is the context of the ORBIT workspace:\n{context}"
    provider_profile = (
        topic_config.get_provider_profile_for(
            topic_id, "web_provider", fallback_key="control_provider"
        )
        if topic_id
        else PROFILE_MINIMAX
    )
    logger.info(
        "[%s] Starting orchestration call profile=%s model=%s allow_web=%s prompt_chars=%s context_chars=%s",
        role,
        provider_profile,
        model or "(default)",
        True,
        len(prompt),
        len(context),
    )

    try:
        result = await retry_structured_output(
            stage_name=f"{role} orchestration",
            logger=logger,
            is_usable=lambda item: _is_usable_json_text(item.text),
            invoke=lambda: llm_call_with_web(
                prompt,
                provider_profile=provider_profile,
                role=role,
                require_json=True,
                model=model,
                search_budget=2,
                temperature=0.7,
                system_prompt=system_prompt,
                topic_id=topic_id,
            ),
        )
        if result is None:
            raise RuntimeError("orchestration structured retry exhausted")
        output = result.text
        logger.info(
            "[%s] Orchestration broker call succeeded provider_used=%s fallback_used=%s search_used=%s text_chars=%s; attempting JSON parse.",
            role,
            result.provider_used,
            result.fallback_used,
            result.search_used,
            len(output or ""),
        )
        parsed = _parse_json_object(output)
        logger.info(
            "[%s] Orchestration JSON parse succeeded keys=%s",
            role,
            (
                sorted(parsed.keys())
                if isinstance(parsed, dict)
                else type(parsed).__name__
            ),
        )
        return parsed
    except Exception as e:
        logger.error("[%s] Orchestration call failed: %s", role, e)
        return {"error": str(e)}


def _decision_passes(
    yes_votes: int, total_votes: int, pass_ratio: float = DECISION_PASS_RATIO
) -> bool:
    if total_votes <= 0:
        return False
    return (yes_votes / total_votes) > pass_ratio


def _build_vote_prompt(
    *,
    topic: dict,
    question: str,
    candidate_summary: str = "",
    candidate_detail: str = "",
    selected: list[str] | None = None,
    rejected: list[str] | None = None,
) -> str:
    selected_block = ", ".join(selected or []) or "none"
    rejected_block = ", ".join(rejected or []) or "none"
    lines = [
        f"Topic: {topic['summary']}",
        f"Topic Detail: {topic['detail']}",
    ]
    if candidate_summary:
        lines.append(f"Candidate Subtopic: {candidate_summary}")
    if candidate_detail:
        lines.append(f"Candidate Detail: {candidate_detail}")
    lines.extend(
        [
            f"Already selected candidates: {selected_block}",
            f"Already rejected candidates: {rejected_block}",
            "",
            question,
            'Reply with strict JSON: {"vote":"yes|no","reason":"short sentence"}.',
        ]
    )
    return "\n".join(lines)


async def _collect_votes(
    prompt: str,
    *,
    topic_id: int,
    subtopic_id: int | None,
    round_number: int | None,
    vote_kind: str,
    subject: str,
) -> VoteTally:
    """Collect votes from all voting agents in parallel."""
    vote_provider = _topic_provider(topic_id, "vote_provider")

    async def _vote_one(voter: str):
        agent = get_agent(voter)
        try:
            return (
                voter,
                await agent.vote_detail(
                    prompt, allow_web=False, provider_profile=vote_provider
                ),
                None,
            )
        except Exception as exc:
            return voter, None, exc

    results = await asyncio.gather(
        *[_vote_one(v) for v in voting_agents()], return_exceptions=True
    )

    yes_votes = 0
    successful_votes = 0
    failed_votes = 0
    for result in results:
        if isinstance(result, Exception):
            failed_votes += 1
            continue
        voter, payload, exc = result
        if exc is not None:
            logger.warning("[skynet] Vote execution failed for %s: %s", voter, exc)
            api.insert_vote_record(
                topic_id,
                subtopic_id,
                round_number,
                vote_kind,
                subject,
                prompt,
                voter,
                False,
                None,
                None,
                "",
                metadata_json=json.dumps(
                    {"invalid_reason": f"exception:{type(exc).__name__}"}
                ),
            )
            failed_votes += 1
        elif payload is None:
            logger.warning("[skynet] Vote from %s was invalid or malformed.", voter)
            api.insert_vote_record(
                topic_id,
                subtopic_id,
                round_number,
                vote_kind,
                subject,
                prompt,
                voter,
                False,
                None,
                None,
                "",
                metadata_json=json.dumps({"invalid_reason": "invalid_or_malformed"}),
            )
            failed_votes += 1
        else:
            successful_votes += 1
            decision = payload["decision"]
            yes_votes += int(decision)
            api.insert_vote_record(
                topic_id,
                subtopic_id,
                round_number,
                vote_kind,
                subject,
                prompt,
                voter,
                True,
                payload["decision_label"],
                payload["reason"],
                payload["raw_response"],
            )
    return {
        "yes_votes": yes_votes,
        "successful_votes": successful_votes,
        "failed_votes": failed_votes,
    }


async def _propose_subtopics(
    topic: dict,
    selected: list[str],
    rejected: list[str],
    unavailable: list[str] | None = None,
    candidate_limit: int = SUBTOPIC_CANDIDATE_COUNT,
) -> dict[str, Any]:
    system_prompt = (
        f"You are Skynet. Propose exactly {candidate_limit} candidate subtopics for this topic. "
        "Avoid duplicating already selected, rejected, or completed subtopics. "
        "All JSON string values must be written in English only. "
        'Output strictly JSON using this schema: {"action":"create_plan","subtopics":[{"summary":"...","detail":"..."}]}.'
    )
    completed = unavailable or []
    context = (
        f"Topic: {topic['summary']}\n"
        f"Topic Detail: {topic['detail']}\n"
        f"Already selected candidates: {', '.join(selected) or 'none'}\n"
        f"Already rejected candidates: {', '.join(rejected) or 'none'}\n"
        f"Already completed subtopics: {', '.join(completed) or 'none'}"
    )
    data = await ask_control_model(system_prompt, context, SKYNET, topic_id=topic["id"])
    if isinstance(data, dict) and data.get("error"):
        return {"candidates": [], "error": str(data["error"])}
    candidates = _sanitize_subtopics(
        data.get("subtopics", []) if isinstance(data, dict) else [],
        limit=candidate_limit,
    )
    seen = set(selected) | set(rejected) | set(completed)
    deduped: list[dict[str, str]] = []
    for candidate in candidates:
        summary = candidate["summary"]
        if summary in seen:
            continue
        seen.add(summary)
        deduped.append(candidate)
        if len(deduped) >= candidate_limit:
            break
    return {"candidates": deduped, "error": None}


def node_inspect_topic_state(state: TopicState) -> TopicState:
    topic_id = state["topic_id"]
    active_plan = api.get_active_plan(topic_id)
    open_subtopic = api.get_open_subtopic(topic_id)

    if open_subtopic:
        return {
            "plan_id": active_plan["id"] if active_plan else 0,
            "current_subtopic_id": open_subtopic["id"],
            "next_action": "run_subtopic",
        }

    if active_plan:
        planned_subtopics = _parse_plan_content(active_plan)
        if active_plan.get("current_index", 0) < len(planned_subtopics):
            return {
                "plan_id": active_plan["id"],
                "next_action": "open_next_subtopic",
            }

    if api.get_current_subtopics(topic_id):
        return {"next_action": "replan_or_close"}

    return {"next_action": "generate_plan"}


def route_from_inspect(state: TopicState) -> str:
    return state.get("next_action", "generate_plan")


async def node_plan_generation(state: TopicState) -> TopicState:
    topic = api.get_topic(state["topic_id"])
    if not topic:
        return {"topic_complete": True, "next_action": "close_topic"}
    selected: list[dict[str, str]] = []
    rejected: list[str] = []
    candidate_count = _topic_candidate_count(state["topic_id"])
    pass_ratio = _topic_pass_ratio(state["topic_id"])

    for cycle in range(1, SUBTOPIC_VOTE_CYCLE_LIMIT + 1):
        proposal = await _propose_subtopics(
            topic,
            [item["summary"] for item in selected],
            rejected,
            candidate_limit=candidate_count,
        )
        if proposal.get("error"):
            logger.warning(
                "[skynet] Deferring topic after proposal-generation failure: %s",
                proposal["error"],
            )
            return {
                "deferred": True,
                "topic_complete": False,
                "next_action": "defer_topic",
            }
        candidates = proposal["candidates"]
        if not candidates:
            continue
        for candidate in candidates:
            prompt = _build_vote_prompt(
                topic=topic,
                question=(
                    "Should this subtopic be admitted to the discussion plan? "
                    "Vote YES only if it materially helps resolve the topic and is not redundant with already selected subtopics."
                ),
                candidate_summary=candidate["summary"],
                candidate_detail=candidate["detail"],
                selected=[item["summary"] for item in selected],
                rejected=rejected,
            )
            tally = await _collect_votes(
                prompt,
                topic_id=topic["id"],
                subtopic_id=None,
                round_number=None,
                vote_kind="candidate_admission",
                subject=candidate["summary"],
            )
            if tally["failed_votes"] > 2 or tally["successful_votes"] == 0:
                logger.warning(
                    "[skynet] Deferring topic after vote execution failures during plan generation."
                )
                return {
                    "deferred": True,
                    "topic_complete": False,
                    "next_action": "defer_topic",
                }
            if _decision_passes(
                tally["yes_votes"], tally["successful_votes"], pass_ratio
            ):
                selected.append(candidate)
                if len(selected) >= candidate_count:
                    break
            else:
                rejected.append(candidate["summary"])
        if len(selected) >= candidate_count:
            break

    if not selected:
        final_summary = (
            f"Topic '{topic['summary']}' is closed because the room could not reach basic consensus on any discussable subtopic after "
            f"{SUBTOPIC_VOTE_CYCLE_LIMIT} proposal cycles. Please restate or narrow the topic."
        )
        emb = await aget_embedding(final_summary)
        if emb:
            api.insert_message_with_embedding(
                state["topic_id"],
                None,
                SKYNET,
                final_summary,
                msg_type="summary",
                embedding=emb,
            )
        else:
            api.post_message(
                state["topic_id"], None, SKYNET, final_summary, msg_type="summary"
            )
        api.set_topic_status(state["topic_id"], "Closed")
        return {"topic_complete": True, "next_action": "close_topic"}

    plan_id = api.create_plan(
        state["topic_id"],
        json.dumps(selected[:candidate_count]),
        current_index=0,
    )
    analytics.capture(
        f"topic_{state['topic_id']}",
        "plan_generated",
        {"subtopic_count": len(selected), "rejected_count": len(rejected)},
    )
    return {"plan_id": plan_id, "next_action": "open_next_subtopic"}


async def node_open_next_subtopic(state: TopicState) -> TopicState:
    topic = api.get_topic(state["topic_id"])
    plan = api.get_active_plan(state["topic_id"])
    if not topic or not plan:
        return {"next_action": "replan_or_close"}

    subtopics = _parse_plan_content(plan)
    plan_index = plan.get("current_index", 0)
    if plan_index >= len(subtopics):
        return {"next_action": "replan_or_close"}

    next_subtopic = subtopics[plan_index]
    system_prompt = (
        "You are Skynet. Create a detailed grounding brief for the next subtopic. "
        "All JSON string values must be written in English only. "
        "Output strictly JSON using this schema: "
        '{"action":"post_message","content":"grounding brief text"}'
    )
    context = (
        f"Topic: {topic['summary']}\n"
        f"Topic Detail: {topic['detail']}\n"
        f"Subtopic: {next_subtopic['summary']}\n"
        f"Subtopic Detail: {next_subtopic['detail']}"
    )
    data = await ask_control_model(
        system_prompt, context, SKYNET, topic_id=state["topic_id"]
    )
    brief_content = data.get("content") if isinstance(data, dict) else None
    if not brief_content:
        brief_content = f"Grounding Brief: {next_subtopic['detail']}"

    subtopic_id = api.create_subtopic(
        state["topic_id"], next_subtopic["summary"], next_subtopic["detail"]
    )
    start_msg_id = await api.persist_message(
        state["topic_id"], subtopic_id, SKYNET, brief_content
    )
    api.update_subtopic_start_msg(subtopic_id, start_msg_id)
    api.advance_plan_cursor(plan["id"])
    api.set_topic_status(state["topic_id"], "Running")
    analytics.capture(
        f"topic_{state['topic_id']}",
        "subtopic_opened",
        {"subtopic_id": subtopic_id, "plan_index": plan_index},
    )
    if plan_index == 0:
        analytics.capture(f"topic_{state['topic_id']}", "topic_started", {})

    # Generate locked_scope for scope integrity enforcement
    try:
        from .broker import call_text

        scope_prompt = (
            f"Extract the evaluation scope for this topic segment.\n"
            f"Topic: {topic['summary']}\nSubtopic: {next_subtopic['detail']}\n\n"
            f'Output JSON: {{"target_metric": "...", "entity_boundaries": "...", '
            f'"metric_definition": "..."}}\n'
            f"If the subtopic is open-ended with no quantifiable metric, output: "
            f'{{"target_metric": null, "entity_boundaries": "...", "metric_definition": null}}'
        )
        scope_resp = await call_text(
            scope_prompt,
            provider=_topic_provider(state["topic_id"], "control_provider"),
            strategy="direct",
            temperature=0.2,
            max_tokens=DEFAULT_MAX_TOKENS,
            require_json=True,
            fallback_role="skynet",
        )
        if scope_resp and scope_resp.strip():
            scope_json = extract_json_object(scope_resp)
            if isinstance(scope_json, dict) and scope_json.get("entity_boundaries"):
                # Ensure string values for prompt injection safety
                for key in ("target_metric", "entity_boundaries", "metric_definition"):
                    val = scope_json.get(key)
                    if val is not None and not isinstance(val, str):
                        scope_json[key] = str(val)[:500]
                api.update_subtopic_locked_scope(
                    subtopic_id, json.dumps(scope_json, ensure_ascii=False)
                )
                logger.info(
                    "[skynet] Locked scope for subtopic %s: %s", subtopic_id, scope_json
                )
    except Exception as exc:
        logger.warning(
            "[skynet] Scope lock generation failed for subtopic %s: %s",
            subtopic_id,
            exc,
        )

    try:
        from .ledger import seed_ledger_from_topic

        await seed_ledger_from_topic(
            topic_id=state["topic_id"],
            subtopic_id=subtopic_id,
            topic_summary=topic["summary"],
            topic_detail=topic["detail"],
            subtopic_summary=next_subtopic["summary"],
            subtopic_detail=next_subtopic["detail"],
        )
    except Exception as exc:
        logger.warning("Ledger seeding failed for subtopic %s: %s", subtopic_id, exc)

    return {
        "plan_id": plan["id"],
        "current_subtopic_id": subtopic_id,
        "next_action": "run_subtopic",
    }


async def node_run_subtopic(state: TopicState) -> TopicState:
    await run_subtopic_graph(
        state["topic_id"],
        state["current_subtopic_id"],
        plan_id=state.get("plan_id", 0),
    )
    return {}


async def node_conclude_subtopic(state: TopicState) -> TopicState:
    topic = api.get_topic(state["topic_id"])
    subtopic = api.get_subtopic(state["current_subtopic_id"])
    messages = api.get_messages(
        state["topic_id"], subtopic_id=state["current_subtopic_id"], limit=40
    )

    if not topic or not subtopic:
        return {}

    ctx = f"Topic: {topic['summary']}\nSubtopic: {subtopic['summary']}\n"
    for message in messages:
        ctx += f"[{message['sender']}]: {message['content']}\n"

    system_prompt = (
        "You are Skynet. Write the final conclusion for this completed subtopic. "
        "All JSON string values must be written in English only. "
        "Output strictly JSON using this schema: "
        '{"action":"close_subtopic","content":"final conclusion"}'
    )
    data = await ask_control_model(
        system_prompt,
        ctx,
        SKYNET,
        topic_id=state["topic_id"],
    )
    conclusion = data.get("content") if isinstance(data, dict) else None
    if not conclusion:
        conclusion = f"Subtopic '{subtopic['summary']}' exhausted."

    emb = await aget_embedding(conclusion)
    if emb:
        api.insert_message_with_embedding(
            state["topic_id"],
            state["current_subtopic_id"],
            SKYNET,
            conclusion,
            msg_type="summary",
            embedding=emb,
        )
    else:
        api.post_message(
            state["topic_id"],
            state["current_subtopic_id"],
            SKYNET,
            conclusion,
            msg_type="summary",
        )

    api.close_subtopic(state["current_subtopic_id"], conclusion)
    analytics.capture(
        f"topic_{state['topic_id']}",
        "subtopic_closed",
        {"subtopic_id": state["current_subtopic_id"]},
    )

    # VIKI watchdog: scan every subtopic close, trigger processing on threshold
    try:
        from . import viki
        import asyncio

        # Always scan (cheap SQL)
        issues = viki.run_watchdog(state["topic_id"])
        # Trigger VIKI processing if: enough issues OR every 2 rounds
        open_count = viki.get_open_issue_count(state["topic_id"])
        current_round = (
            api.get_max_round_number(state["topic_id"], state["current_subtopic_id"])
            or 0
        )
        should_trigger = open_count >= viki.TRIGGER_THRESHOLD or (
            current_round and current_round % 2 == 0
        )
        if issues and should_trigger:
            viki_task = asyncio.create_task(
                viki.process_issues_background(state["topic_id"], issues)
            )
            _background_tasks.add(viki_task)
            viki_task.add_done_callback(_background_tasks.discard)
            viki_task.add_done_callback(_report_task_done)
            logger.info(
                "[VIKI] Processing %d issues (open=%d, round=%d)",
                len(issues),
                open_count,
                current_round,
            )
    except Exception as exc:
        logger.warning("[VIKI] Watchdog failed: %s", exc)

    return {"current_subtopic_id": 0}


async def node_topic_replan_or_close(state: TopicState) -> TopicState:
    topic = api.get_topic(state["topic_id"])
    subtopics = api.get_current_subtopics(state["topic_id"])
    if not topic:
        return {"topic_complete": True}

    # Check replan limit — count how many replan cycles have already happened
    # Initial plan creates the first batch of subtopics; each replan adds more.
    # closed_count tracks completed subtopics; initial plan creates ~candidate_count.
    max_replan = topic_config.get_int(state["topic_id"], "max_replan_rounds")
    initial_batch = topic_config.get_int(state["topic_id"], "subtopic_candidate_count")
    closed_count = sum(1 for s in subtopics if s.get("status") == "Closed")
    replan_rounds_used = max(
        0, (closed_count - initial_batch + initial_batch - 1) // initial_batch
    )
    if replan_rounds_used >= max_replan:
        logger.info(
            "[skynet] Replan limit reached (%d/%d). Force closing topic %d.",
            replan_rounds_used,
            max_replan,
            state["topic_id"],
        )
        api.set_topic_status(state["topic_id"], "Closed")
        return {"topic_complete": True, "next_action": "close_topic"}

    candidate_count = _topic_candidate_count(state["topic_id"])
    pass_ratio = _topic_pass_ratio(state["topic_id"])
    ctx = f"Topic: {topic['summary']}\nDetail: {topic['detail']}\n"
    for subtopic in subtopics:
        conclusion = subtopic.get("conclusion") or "(No conclusion recorded)"
        ctx += f"Subtopic: {subtopic['summary']}\nConclusion: {conclusion}\n"

    replan_vote_prompt = _build_vote_prompt(
        topic=topic,
        question=(
            "Should the room open additional subtopics for this topic? "
            "Vote YES only if the completed subtopics are still insufficient to support a final answer."
        ),
        candidate_summary="",
        candidate_detail=ctx,
    )
    tally = await _collect_votes(
        replan_vote_prompt,
        topic_id=topic["id"],
        subtopic_id=None,
        round_number=None,
        vote_kind="replan_gate",
        subject="Should the topic replan?",
    )
    if tally["failed_votes"] > 2 or tally["successful_votes"] == 0:
        logger.warning(
            "[skynet] Deferring topic after vote execution failures during replan gate."
        )
        return {"deferred": True, "topic_complete": False, "next_action": "defer_topic"}
    if not _decision_passes(tally["yes_votes"], tally["successful_votes"], pass_ratio):
        final_summary = f"Topic '{topic['summary']}' is complete."
        emb = await aget_embedding(final_summary)
        if emb:
            api.insert_message_with_embedding(
                state["topic_id"],
                None,
                SKYNET,
                final_summary,
                msg_type="summary",
                embedding=emb,
            )
        else:
            api.post_message(
                state["topic_id"], None, SKYNET, final_summary, msg_type="summary"
            )
        api.set_topic_status(state["topic_id"], "Closed")
        return {"topic_complete": True, "next_action": "close_topic"}

    selected: list[dict[str, str]] = []
    rejected: list[str] = []
    completed_summaries = [
        item["summary"] for item in subtopics if isinstance(item.get("summary"), str)
    ]
    for _cycle in range(1, SUBTOPIC_VOTE_CYCLE_LIMIT + 1):
        proposal = await _propose_subtopics(
            topic,
            [item["summary"] for item in selected],
            rejected,
            completed_summaries,
            candidate_limit=candidate_count,
        )
        if proposal.get("error"):
            logger.warning(
                "[skynet] Deferring topic after replanning proposal-generation failure: %s",
                proposal["error"],
            )
            return {
                "deferred": True,
                "topic_complete": False,
                "next_action": "defer_topic",
            }
        candidates = proposal["candidates"]
        if not candidates:
            continue
        for candidate in candidates:
            prompt = _build_vote_prompt(
                topic=topic,
                question=(
                    "Should this newly proposed subtopic be admitted during replanning? "
                    "Vote YES only if it adds needed coverage beyond what has already been completed."
                ),
                candidate_summary=candidate["summary"],
                candidate_detail=candidate["detail"],
                selected=[item["summary"] for item in selected],
                rejected=rejected,
            )
            tally = await _collect_votes(
                prompt,
                topic_id=topic["id"],
                subtopic_id=None,
                round_number=None,
                vote_kind="replan_admission",
                subject=candidate["summary"],
            )
            if tally["failed_votes"] > 2 or tally["successful_votes"] == 0:
                logger.warning(
                    "[skynet] Deferring topic after vote execution failures during replanning."
                )
                return {
                    "deferred": True,
                    "topic_complete": False,
                    "next_action": "defer_topic",
                }
            if _decision_passes(
                tally["yes_votes"], tally["successful_votes"], pass_ratio
            ):
                selected.append(candidate)
                if len(selected) >= candidate_count:
                    break
            else:
                rejected.append(candidate["summary"])
        if len(selected) >= candidate_count:
            break

    if not selected:
        final_summary = f"Topic '{topic['summary']}' is complete."
        emb = await aget_embedding(final_summary)
        if emb:
            api.insert_message_with_embedding(
                state["topic_id"],
                None,
                SKYNET,
                final_summary,
                msg_type="summary",
                embedding=emb,
            )
        else:
            api.post_message(
                state["topic_id"], None, SKYNET, final_summary, msg_type="summary"
            )
        api.set_topic_status(state["topic_id"], "Closed")
        return {"topic_complete": True, "next_action": "close_topic"}

    plan_id = api.create_plan(
        state["topic_id"],
        json.dumps(selected[:candidate_count]),
        current_index=0,
    )
    return {
        "plan_id": plan_id,
        "topic_complete": False,
        "next_action": "open_next_subtopic",
    }


def route_after_replan(state: TopicState) -> str:
    if state.get("deferred"):
        return "defer_topic"
    if state.get("topic_complete"):
        return "close_topic"
    return "open_next_subtopic"


def route_after_generate_plan(state: TopicState) -> str:
    if state.get("deferred"):
        return "defer_topic"
    if state.get("topic_complete"):
        return "close_topic"
    return "open_next_subtopic"


def route_after_open_next_subtopic(state: TopicState) -> str:
    if state.get("next_action") == "replan_or_close" or not state.get(
        "current_subtopic_id"
    ):
        return "replan_or_close"
    return "run_subtopic"


async def node_topic_conclusion(state: TopicState) -> TopicState:
    logger.info("[skynet] Generating final topic conclusion...")
    topic = api.get_topic(state["topic_id"])
    subtopics = api.get_current_subtopics(state["topic_id"])
    if not topic:
        return {"topic_complete": True, "next_action": "close_topic"}

    ctx = f"Topic: {topic['summary']}\nDetail: {topic['detail']}\n\n"
    for st in subtopics:
        conclusion = st.get("conclusion") or "(No conclusion recorded)"
        ctx += f"Subtopic: {st['summary']}\nConclusion:\n{conclusion}\n\n"

    system_prompt = (
        "You are Skynet. Synthesize the conclusions of all subtopics into a single, "
        "comprehensive, final conclusion for the entire topic. Address the original topic details. "
        "All JSON string values must be written in English only. "
        "Output strictly JSON using this schema: "
        '{"action":"conclude_topic","content":"your comprehensive final conclusion"}'
    )
    data = await ask_control_model(
        system_prompt,
        ctx,
        SKYNET,
        topic_id=state["topic_id"],
    )
    if isinstance(data, dict) and data.get("error"):
        logger.warning(
            "[skynet] Topic conclusion LLM call failed: %s; using fallback",
            data["error"],
        )
    conclusion_text = data.get("content") if isinstance(data, dict) else None
    if (
        not conclusion_text
        or not isinstance(conclusion_text, str)
        or not conclusion_text.strip()
    ):
        # Fallback: synthesize from subtopic conclusions deterministically
        parts = []
        for st in subtopics:
            c = st.get("conclusion")
            if c and isinstance(c, str) and c.strip():
                parts.append(f"- {st['summary']}: {c.strip()}")
        conclusion_text = f"Topic '{topic['summary']}' concluded.\n" + (
            "\n".join(parts) if parts else "No subtopic conclusions were recorded."
        )
    else:
        conclusion_text = conclusion_text.strip()

    conclusion_text = conclusion_text[:MAX_CONTENT_LEN]

    # Store as Message with embedding (matching subtopic pattern)
    emb = await aget_embedding(conclusion_text)
    if emb:
        api.insert_message_with_embedding(
            state["topic_id"],
            None,
            SKYNET,
            conclusion_text,
            msg_type="summary",
            embedding=emb,
        )
    else:
        api.post_message(
            state["topic_id"],
            None,
            SKYNET,
            conclusion_text,
            msg_type="summary",
        )

    # Also store in Topic.conclusion column
    api.update_topic_conclusion(state["topic_id"], conclusion_text)

    return {"topic_complete": True, "next_action": "close_topic"}


def _report_task_done(task):
    """Log exceptions from fire-and-forget report generation."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.warning("[skynet] Background report generation failed: %s", exc)


async def close_topic_node(state: TopicState) -> TopicState:
    logger.info("[skynet] Topic complete.")
    analytics.capture(f"topic_{state['topic_id']}", "topic_completed", {})
    try:
        from . import report

        task = asyncio.create_task(report.generate_report(state["topic_id"]))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
        task.add_done_callback(_report_task_done)
        logger.info(
            "[skynet] Report generation scheduled for topic %d", state["topic_id"]
        )
    except Exception as exc:
        logger.warning("[skynet] Could not schedule report generation: %s", exc)
    return {"topic_complete": True}


def defer_topic_node(state: TopicState) -> TopicState:
    logger.info("[skynet] Topic deferred after transient orchestration failure.")
    return {"deferred": True, "topic_complete": False}


# ---------------------------------------------------------------------------
# HITL pause check nodes (Phase F.2)
# ---------------------------------------------------------------------------


def node_hitl_plan_check(state: TopicState) -> TopicState:
    topic_id = state["topic_id"]
    if topic_config.get_bool(topic_id, "hitl_plan_approval"):
        api.pause_topic(topic_id, "plan_approval")
        return {"deferred": True}
    return {"deferred": False}


def node_hitl_subtopic_check(state: TopicState) -> TopicState:
    topic_id = state["topic_id"]
    if topic_config.get_bool(topic_id, "hitl_subtopic_review"):
        api.pause_topic(topic_id, "subtopic_review")
        return {"deferred": True}
    return {"deferred": False}


def node_hitl_final_check(state: TopicState) -> TopicState:
    topic_id = state["topic_id"]
    if topic_config.get_bool(topic_id, "hitl_final_review"):
        api.pause_topic(topic_id, "final_review")
        return {"deferred": True}
    return {"deferred": False}


def node_hitl_replan_check(state: TopicState) -> TopicState:
    topic_id = state["topic_id"]
    if topic_config.get_bool(topic_id, "hitl_replan_pause"):
        api.pause_topic(topic_id, "replan")
        return {"deferred": True}
    return {"deferred": False}


def route_after_hitl(state: TopicState) -> str:
    if state.get("deferred"):
        return "defer_topic"
    return "continue"


def build_master_graph():
    builder = StateGraph(TopicState)

    builder.add_node("inspect_topic_state", node_inspect_topic_state)
    builder.add_node("generate_plan", node_plan_generation)
    builder.add_node("hitl_plan_check", node_hitl_plan_check)
    builder.add_node("open_next_subtopic", node_open_next_subtopic)
    builder.add_node("run_subtopic", node_run_subtopic)
    builder.add_node("conclude_subtopic", node_conclude_subtopic)
    builder.add_node("hitl_subtopic_check", node_hitl_subtopic_check)
    builder.add_node("replan_or_close", node_topic_replan_or_close)
    builder.add_node("hitl_replan_check", node_hitl_replan_check)
    builder.add_node("defer_topic", defer_topic_node)
    builder.add_node("close_topic", close_topic_node)
    builder.add_node("conclude_topic", node_topic_conclusion)
    builder.add_node("hitl_final_check", node_hitl_final_check)

    builder.add_edge(START, "inspect_topic_state")
    builder.add_conditional_edges(
        "inspect_topic_state",
        route_from_inspect,
        {
            "generate_plan": "generate_plan",
            "open_next_subtopic": "open_next_subtopic",
            "run_subtopic": "run_subtopic",
            "replan_or_close": "replan_or_close",
        },
    )
    # generate_plan -> hitl_plan_check -> open_next_subtopic | defer_topic
    builder.add_conditional_edges(
        "generate_plan",
        route_after_generate_plan,
        {
            "open_next_subtopic": "hitl_plan_check",
            "defer_topic": "defer_topic",
            "close_topic": "conclude_topic",
        },
    )
    builder.add_conditional_edges(
        "hitl_plan_check",
        route_after_hitl,
        {
            "defer_topic": "defer_topic",
            "continue": "open_next_subtopic",
        },
    )
    builder.add_conditional_edges(
        "open_next_subtopic",
        route_after_open_next_subtopic,
        {
            "run_subtopic": "run_subtopic",
            "replan_or_close": "replan_or_close",
        },
    )
    builder.add_edge("run_subtopic", "conclude_subtopic")
    # conclude_subtopic -> hitl_subtopic_check -> inspect_topic_state | defer_topic
    builder.add_edge("conclude_subtopic", "hitl_subtopic_check")
    builder.add_conditional_edges(
        "hitl_subtopic_check",
        route_after_hitl,
        {
            "defer_topic": "defer_topic",
            "continue": "inspect_topic_state",
        },
    )
    # replan_or_close -> hitl_replan_check -> open_next_subtopic | defer_topic
    builder.add_conditional_edges(
        "replan_or_close",
        route_after_replan,
        {
            "open_next_subtopic": "hitl_replan_check",
            "defer_topic": "defer_topic",
            "close_topic": "conclude_topic",
        },
    )
    builder.add_conditional_edges(
        "hitl_replan_check",
        route_after_hitl,
        {
            "defer_topic": "defer_topic",
            "continue": "open_next_subtopic",
        },
    )
    # conclude_topic -> hitl_final_check -> close_topic | defer_topic
    builder.add_edge("conclude_topic", "hitl_final_check")
    builder.add_conditional_edges(
        "hitl_final_check",
        route_after_hitl,
        {
            "defer_topic": "defer_topic",
            "continue": "close_topic",
        },
    )
    builder.add_edge("defer_topic", END)
    builder.add_edge("close_topic", END)

    return builder.compile()
