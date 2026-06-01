import asyncio
import json
import logging
import math
import re
import sqlite3
from typing import Any, Optional, Sequence

from langgraph.graph import END, START, StateGraph

from .graph import (
    ChatState,
    TurnSpec,
    stage_dispatcher_node,
    route_from_stage_dispatcher,
)
from . import analytics
from . import api
from . import db
from . import topic_config
from .agents import (
    DELIBERATORS,
    ROLE_VISIBILITY,
    SKYNET,
    SPECTATOR,
    can_special_target,
    get_agent,
    get_agent_spec,
    ordinary_deliberators,
    special_agents,
    voting_agents,
)
from .broker import (
    DEFAULT_MAX_TOKENS,
    SearchEvidenceItem,
    call_text,
    call_text_with_search_evidence,
    collect_search_evidence_bundle,
    shutdown_broker,
)
from .rag import (
    _escape_citation_tokens,
    assemble_rag_context,
    build_query_rag_context,
)
from .logging_utils import configure_logging
from .code_sandbox import (
    CODE_FOLLOWUP_TURN,
    ensure_sandbox,
    is_sandbox_ready,
    run_calc,
    run_code_evidence,
    run_code_evidence_grid,
    run_code_review,
)
from .prompts import PROMPTS
from .writer_processor import process_clerk_claim_output, process_writer_output
from .librarian_processor import (
    apply_claim_review,
    apply_librarian_review,
    build_librarian_audit_message,
    parse_claim_review,
    parse_librarian_review,
)
from .embedding import aget_embedding
from .json_utils import extract_json_object as extract_json
from .structured_retry import (
    generate_summary,
    retry_structured_output,
    usable_text_output,
)

logger = logging.getLogger(__name__)

AGENTS = list(DELIBERATORS) + ["cat", "dog", "tron", SPECTATOR]
PARSER_FAILURE_CONFIDENCE = 2.5
DEGRADED_OPERATION_CONFIDENCE = 3.0
LOOP_WARNING_DISTANCE = 0.25
WRITER_FACT_LIMIT = 2
FINAL_WRITER_FACT_LIMIT = 3
BOOTSTRAP_FACT_DIRECTION_LIMIT = 3
INLINE_FACT_LIMIT = 1

OPENING_PHASE = "opening"
EVIDENCE_PHASE = "evidence"
ANALYSIS_PHASE = "analysis"
MSE_PROBLEM_PHASE = "mse_problem"
MSE_COMPONENT_PHASE = "mse_components"
MSE_COMPONENT_REVIEW_PHASE = "mse_component_review"
MSE_SPECIFICATION_PHASE = "mse_specification"
MSE_ARTIFACT_PHASE = "mse_artifact"
MSE_SOLVER_PHASE = "mse_solver"
MSE_REPAIR_PHASE = "mse_repair"
MSE_MANAGERIAL_PHASE = "mse_managerial"
MSE_SOLVED_PHASE = "mse_solved"

MSE_FAST_WORKFLOW = "modeling_fast"
MSE_REVIEWED_WORKFLOW = "modeling_reviewed"
MSE_PROFILE_VALUES = {"mse", "management_science_engineering"}

BASE_TURN = "base"
TRON_REMEDIATION_TURN = "tron_remediation"
DOG_CORRECTION_TURN = "dog_correction"
CAT_EXPANSION_TURN = "cat_expansion"
WRITER_CRITIQUE_TURN = "writer_critique"
LIBRARIAN_AUDIT_TURN = "librarian_audit"
AUDIENCE_SUMMARY_TURN = "skynet_summary"
AUDIENCE_WARNING_TURN = "skynet_warning"

OPENING_ROSTER = ["dreamer", "scientist", "engineer", "analyst", "critic", "tron"]
FULL_ROSTER = [
    "dreamer",
    "scientist",
    "engineer",
    "analyst",
    "critic",
    "contrarian",
    "dog",
    "cat",
    "tron",
    SPECTATOR,
]
MSE_STEP_ACTORS = {
    "problem_framing": "analyst",
    "component_extraction": "analyst",
    "component_review": "scientist",
    "specification_gap": "analyst",
    "artifact_generation": "engineer",
    "artifact_repair": "engineer",
    "solver_execution": "engineer",
    "solver_repair": "critic",
    "managerial_synthesis": "contrarian",
}
TARGET_NAME_ALIASES = {
    "dreamer": "dreamer",
    "空想家": "dreamer",
    "scientist": "scientist",
    "科学家": "scientist",
    "engineer": "engineer",
    "工程师": "engineer",
    "analyst": "analyst",
    "分析师": "analyst",
    "critic": "critic",
    "批评家": "critic",
    "contrarian": "contrarian",
    "逆反者": "contrarian",
    "少数派": "contrarian",
}

SUBTOPIC_CANDIDATE_COUNT = 4
SUBTOPIC_VOTE_CYCLE_LIMIT = 3
DECISION_PASS_RATIO = 2 / 3
TERMINATION_MAX_INVALID_VOTES = 2
ROUND3_CLOSE_RATIO = 0.8
ROUND46_CLOSE_RATIO = 2 / 3
ROUND79_CLOSE_RATIO = 0.6
SUMMARY_SECTION_HEADERS = (
    "TRAJECTORY:",
    "CONSENSUS:",
    "BLOCKERS:",
    "EVIDENCE GAPS:",
    "AGENT DELTAS:",
)
DELIBERATION_DISCIPLINE_LINES = (
    "WORKFLOW DISCIPLINE:",
    "- Prefer net-new argument, explicit correction, or narrowed disagreement over praise, empty agreement, or broad recap.",
    "- If you challenge a claim, identify the specific sentence, assumption, metric, or causal link you are attacking.",
    "- Reference prior arguments by citation `[M{id}]` instead of summarizing or paraphrasing them. Your output must be net-new analysis.",
    "- Every numerical claim (exchange rates, percentages, amounts) MUST appear in the same sentence as its citation [F{id}], [W{id}], [L{id}], or [M{id}]. Uncited numbers are treated as unverified speculation.",
    "- Do not argue with system warnings from Dog, Cat, or Tron. Silently correct your behavior.",
)
WRITER_ANALYSIS_SYSTEM_PROMPT = """You are the Writer and a meta-Critic observing a multi-role ORBIT workspace round.
Your job in this pass is to analyze the round, not to produce final JSON output.

CRITICAL INSTRUCTION:
Write in English only.
Return concise plain text only.
Do not use markdown fences, thinking tags, or extra commentary outside the requested format.
"""
WRITER_STAGE_MAX_ATTEMPTS = 2
FACT_CITATION_PROTOCOL = (
    "KNOWLEDGE CITATION PROTOCOL:\n"
    "- Cite stored facts as `[F{id}]`.\n"
    "- Cite stored claims as `[C{id}]`.\n"
    "- Cite private corpus chunks as `[D{id}]`; use them for source-document evidence.\n"
    "- Cite web evidence as `[W{id}]`, but describe it as unverified web evidence.\n"
    "- Cite ledger entries as `[L{id}]`. Ledger entries are structured numerical data with verified sources.\n"
    "- Cite code or solver evidence as `[E{id}]` when interpreting executable verification.\n"
    "- Cite model/API consultations as `[A{id}]`, but describe them as unverified model perspective, not factual evidence.\n"
    "- `[W...]` items may guide verification, but they are not permanent facts unless later admitted as `[F...]`.\n"
    "- `[A...]` items may guide model-capability reasoning, but they are not permanent facts unless later verified.\n"
    "- Do not invent IDs.\n"
    "- Reference prior messages as `[M{id}]` instead of restating them. Messages are context/attribution only, not evidence.\n"
    "- Summaries and historical messages are context only. Do not cite them as evidence.\n"
    "- Do not restate facts that already have an [F{id}]. Reference the existing ID instead of restating the content."
)
SOURCED_FACT_LIMIT = 2
FINAL_SOURCED_FACT_LIMIT = 3
CLAIM_LIMIT = 2
FINAL_CLAIM_LIMIT = 3
CITATION_ID_PATTERN = re.compile(r"\[(D|F|C|W|L|A|E)(\d+)\]")


def _parse_single_json_wrapper(text: str) -> Optional[dict]:
    stripped = (text or "").strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        stripped = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE | re.DOTALL
        ).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _clamp_confidence(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return max(0.0, min(10.0, float(value)))
    except (TypeError, ValueError):
        return None


def _normalize_message_contract(
    raw_text: str,
    accepted_actions: Sequence[str] = ("post_message",),
    fallback_confidence: Optional[float] = None,
) -> dict:
    parsed = _parse_single_json_wrapper(raw_text) or extract_json(raw_text)
    if isinstance(parsed, dict):
        action = parsed.get("action")
        content = parsed.get("content")
        if action in accepted_actions and isinstance(content, str) and content.strip():
            confidence = _clamp_confidence(parsed.get("confidence_score"))
            if fallback_confidence is not None:
                confidence = (
                    confidence if confidence is not None else fallback_confidence
                )
            raw_facts = parsed.get("facts")
            facts = None
            if isinstance(raw_facts, list):
                facts = [
                    fact.strip()
                    for fact in raw_facts
                    if isinstance(fact, str) and fact.strip()
                ]
            # G.4: Extract optional formal_claim attachment
            formal_claim = parsed.get("formal_claim")
            if formal_claim and not isinstance(formal_claim, dict):
                formal_claim = None
            mse_artifact_update = parsed.get("mse_artifact_update")
            if mse_artifact_update and not isinstance(mse_artifact_update, dict):
                mse_artifact_update = None
            return {
                "parsed_ok": True,
                "action": action,
                "content": content.strip(),
                "confidence_score": confidence,
                "facts": facts,
                "formal_claim": formal_claim,
                "mse_artifact_update": mse_artifact_update,
            }

    confidence = (
        fallback_confidence
        if fallback_confidence is not None
        else PARSER_FAILURE_CONFIDENCE
    )
    return {
        "parsed_ok": False,
        "action": accepted_actions[0] if accepted_actions else "post_message",
        "content": (raw_text.strip() or raw_text)[:8000],
        "confidence_score": confidence,
        "facts": None,
        "formal_claim": None,
        "mse_artifact_update": None,
    }


def _extract_allowed_citation_ids(
    *knowledge_blocks: str, trusted_api_blocks: Sequence[str] = ()
) -> dict[str, set[int]]:
    allowed = {
        "D": set(),
        "F": set(),
        "C": set(),
        "W": set(),
        "L": set(),
        "A": set(),
        "E": set(),
    }
    for block in knowledge_blocks:
        for prefix, raw_id in CITATION_ID_PATTERN.findall(block or ""):
            if prefix == "A":
                continue
            allowed[prefix].add(int(raw_id))
    for block in trusted_api_blocks:
        for raw_id in re.findall(
            r"(?m)^(?:- )?\[A(\d+)\]\s(?:Unverified model/API consultation|\([^)]*\))",
            block or "",
        ):
            allowed["A"].add(int(raw_id))
    return allowed


def _sanitize_citations_to_allowed_ids(
    content: str,
    *,
    knowledge_blocks: Sequence[str],
    trusted_api_blocks: Sequence[str] = (),
) -> tuple[str, dict[str, tuple[int, ...]]]:
    allowed = _extract_allowed_citation_ids(
        *knowledge_blocks, trusted_api_blocks=trusted_api_blocks
    )
    removed: dict[str, list[int]] = {
        "D": [],
        "F": [],
        "C": [],
        "W": [],
        "L": [],
        "A": [],
        "E": [],
    }

    def _replace(match: re.Match[str]) -> str:
        prefix, raw_id = match.groups()
        citation_id = int(raw_id)
        if citation_id in allowed[prefix]:
            return match.group(0)
        removed[prefix].append(citation_id)
        return ""

    cleaned = CITATION_ID_PATTERN.sub(_replace, content or "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"[ ]+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,;:!?])(?:\s*\1)+", r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    deduped_removed = {
        prefix: tuple(dict.fromkeys(values)) for prefix, values in removed.items()
    }
    return cleaned.strip(), deduped_removed


# ---------------------------------------------------------------------------
# Targeted sync retry — nudge agents to cite financial numbers
# ---------------------------------------------------------------------------

_FINANCIAL_NUMBER_RE = re.compile(r"\d+\.\d{3,}")


def _has_uncited_financial_numbers(content: str) -> list[str]:
    """Find financial numbers (3+ decimal digits) in sentences without any citation."""
    matches = _FINANCIAL_NUMBER_RE.findall(content or "")
    if not matches:
        return []
    uncited = []
    # Split on sentence boundaries but not on decimal points within numbers
    sentences = re.split(r"(?<!\d)\.(?!\d)|(?<=\d)\.(?=\s+[A-Z])|[!?\n]", content)
    for number in matches:
        for sentence in sentences:
            if number in sentence:
                if not re.search(r"\[(F|W|L|M|C)\d+\]", sentence):
                    uncited.append(number)
                    break  # found uncited usage, stop checking for this number
    return uncited


def _validate_ledger_citation_ids(content: str) -> list[str]:
    """Return list of invalid [L{id}] citations (those not in the Ledger table).

    Uses a single batched query instead of per-ID lookups.
    """
    ids = [int(m.group(1)) for m in re.finditer(r"\[L(\d+)\]", content or "")]
    if not ids:
        return []
    with db.get_db() as conn:
        placeholders = ",".join("?" * len(ids))
        existing = {
            row[0]
            for row in conn.execute(
                f"SELECT id FROM Ledger WHERE id IN ({placeholders})", ids
            ).fetchall()
        }
    return [f"[L{lid}]" for lid in ids if lid not in existing]


async def _targeted_sync_retry(
    content: str,
    round_number: int,
    system_prompt: str,
    prompt: str,
    actor: str,
    state: dict,
) -> str:
    """Retry if content has uncited financial numbers. Max 2 retries."""
    if round_number < 2:
        return content

    uncited = _has_uncited_financial_numbers(content)
    if not uncited:
        return content

    # Retry 1: ask to add citations (include draft so LLM can fix it)
    nums_str = ", ".join(uncited[:5])
    retry_prompt = (
        f"{prompt}\n\n"
        f"YOUR PREVIOUS DRAFT:\n{content}\n\n"
        f"SYSTEM: Your draft contains uncited numerical data: {nums_str}. "
        "Add citation [L], [F], or [W] next to each number."
    )
    try:
        retry_text = await _call_text_with_structured_retry(
            stage_name=f"{actor} sync retry 1",
            validator=lambda text: _structured_message_is_usable(
                text, accepted_actions=("post_message",)
            ),
            invoke=lambda: call_text(
                retry_prompt,
                provider=_resolve_agent_provider(state, actor),
                strategy="direct",
                allow_web=False,
                system_instruction=system_prompt,
                fallback_role=actor,
                require_json=True,
            ),
        )
        if retry_text is None:
            return content
        parsed = _normalize_message_contract(retry_text)
        if not parsed["parsed_ok"]:
            return content
        retry_content = parsed["content"]
    except Exception as exc:
        logger.warning("[%s] Sync retry 1 failed: %s", actor, exc)
        return content

    # Check for fake ledger citation IDs
    invalid_ids = _validate_ledger_citation_ids(retry_content)
    if not invalid_ids:
        return retry_content

    # Retry 2: fix fake IDs
    ids_str = ", ".join(invalid_ids[:5])
    retry2_prompt = (
        f"{prompt}\n\n"
        f"SYSTEM: Citation {ids_str} does not exist. Use a valid ID from the injected knowledge."
    )
    try:
        retry2_text = await _call_text_with_structured_retry(
            stage_name=f"{actor} sync retry 2",
            validator=lambda text: _structured_message_is_usable(
                text, accepted_actions=("post_message",)
            ),
            invoke=lambda: call_text(
                retry2_prompt,
                provider=_resolve_agent_provider(state, actor),
                strategy="direct",
                allow_web=False,
                system_instruction=system_prompt,
                fallback_role=actor,
                require_json=True,
            ),
        )
        if retry2_text is None:
            return retry_content
        parsed2 = _normalize_message_contract(retry2_text)
        if parsed2["parsed_ok"]:
            return parsed2["content"]
    except Exception as exc:
        logger.warning("[%s] Sync retry 2 failed: %s", actor, exc)
    return retry_content


# ---------------------------------------------------------------------------
# Clerk ledger extraction prompt builder + parser
# ---------------------------------------------------------------------------


_ATTR_STRICT_DEFS = {
    "F1 Score": "ONLY for F1 classification metric, range 0.0-1.0. NOT for win rates, counts, ratios, deltas, or efficiency metrics.",
    "Accuracy": "ONLY for classification accuracy, range 0.0-1.0.",
    "AUC-ROC": "ONLY for AUC-ROC metric, range 0.0-1.0.",
    "Training Time": "ONLY for wall-clock training duration in seconds.",
    "Inference Time": "ONLY for wall-clock inference duration in seconds.",
    "Dataset Size": "ONLY for number of samples/rows in a dataset.",
    "Feature Dimensionality": "ONLY for number of input features/dimensions.",
    "Class Count": "ONLY for number of target classes.",
    "Learning Rate": "ONLY for optimizer learning rate hyperparameter.",
    "Network Depth": "ONLY for number of hidden layers.",
    "Network Width": "ONLY for number of neurons per hidden layer.",
    "Number of Trees": "ONLY for ensemble tree count in RF/GBT.",
    "Mean Squared Error": "ONLY for MSE regression metric.",
    "R-squared": "ONLY for R-squared regression metric, range 0.0-1.0.",
    "Noise Level": "ONLY for synthetic data noise parameter.",
}

_LEDGER_JSON_SCHEMA = """\
Output a strict JSON object: {"records": [<array of records>]}

Each record:
{
  "thought": "<1 sentence: WHY you chose this attribute. Compare the value's unit/semantics to the STRICT definitions above. If no existing attribute matches, explain why you are creating NEW.>",
  "entity": <entity # as integer, or "NEW:Full Name">,
  "config": <JSON object of hyperparameters/conditions, e.g. {"lr": 0.001, "width": 128}. Use {"variant": "default"} if unspecified or default settings>,
  "attribute": <attribute # ONLY if exact semantic match per STRICT definition, otherwise "NEW:Metric Name">,
  "value": <primary numeric value as number>,
  "min": <range min or same as value for point estimates>,
  "max": <range max or same as value for point estimates>,
  "unit": "NONE|USD|EUR|CNY|FLOPS|SECONDS|WATTS|OTHER",
  "metric_unit": "absolute|percentage|count|ratio|seconds|other",
  "stat_type": "point|mean_std|delta|ratio|percentage|p_value|ci|correlation|rank|se",
  "std": <standard deviation as number, or null>,
  "p_value": <p-value as number, or null>,
  "n": <sample size/seeds/folds as integer, or null>,
  "ci_lower": <CI lower bound, or null>,
  "ci_upper": <CI upper bound, or null>,
  "ci_level": <confidence level like 0.95, or null>,
  "baseline_entity": <entity # compared against, or null>,
  "split": "train|validation|test|cv" or null,
  "time": "MM/DD/YYYY-MM/DD/YYYY" or "NONE",
  "source": "<citation like E67 or W5>"
}

RULES:
- "thought" is MANDATORY. You must justify your attribute choice BEFORE selecting the ID.
- "config" captures model hyperparameters so different configurations of the same model family are distinguished. Always include it.
- attribute: Use an existing # ONLY if the metric exactly matches the STRICT definition. If in doubt, use "NEW:Metric Name".
- stat_type classifies what the value represents:
  point: single measurement (F1=0.9834)
  mean_std: mean with std dev (F1=0.9834+/-0.0152) — value/min/max=mean, std=0.0152
  delta: difference between two entities (gap=-0.0068) — set baseline_entity
  ratio: multiplier or efficiency metric (speedup=6.0x)
  percentage: rate or proportion (beat_rate=1.9%)
  p_value: statistical significance test result
  ci: confidence interval — use ci_lower/ci_upper/ci_level
  correlation: Pearson/Spearman coefficient
  rank: ordinal position (1st, 2nd)
  se: standard error of the mean
- CRITICAL: F1=0.9834+/-0.0152 is ONE entry (stat_type=mean_std), NOT two separate entries
- CRITICAL: delta/p_value entries MUST set baseline_entity (compared against whom)
- Extract numbers exactly as they appear — no rounding
- Qualitative insights with no number: {"entity": null, "fact": "standalone sentence"}
- If no data at all: {"records": []}

NEGATIVE EXAMPLES (DO NOT DO THIS):
  BAD:  "Beat Both: 1.4%" -> attribute=F1 Score  (F1 is 0-1 absolute, not a win rate %)
  GOOD: "Beat Both: 1.4%" -> attribute=NEW:Beat Both Rate, metric_unit=percentage
  BAD:  "4.97 seeds" -> attribute=F1 Score  (seeds are a count, not F1)
  GOOD: "4.97 seeds" -> attribute=NEW:Initialization Seeds, metric_unit=count
  BAD:  "Efficiency ratio: -0.319" -> attribute=F1 Score  (ratios are not F1)
  GOOD: "Efficiency ratio: -0.319" -> attribute=NEW:Break-Even Efficiency Ratio, metric_unit=ratio"""


def build_clerk_ledger_extraction_prompt(
    topic_id: int,
    message_content: str,
    sender: str,
    round_number: int,
    *,
    entities: list[tuple[int, str]] | None = None,
    attributes: list[tuple[int, str]] | None = None,
) -> str | None:
    """Build a Clerk prompt for structured ledger extraction (JSON format).

    Accepts pre-fetched entity/attribute lists to avoid double-fetching when the
    caller also needs them for parsing.
    """
    from . import ledger as _ledger

    if entities is None:
        entities = _ledger.get_entity_numbered_list(topic_id, round_number)
    if attributes is None:
        attributes = _ledger.get_attribute_numbered_list(topic_id)
    if not entities and not attributes:
        return None

    entity_lines = ["0. [UNSPECIFIED]"]
    for eid, name in entities:
        entity_lines.append(f"{eid}. {name}")
    if round_number <= 5:
        entity_lines.append('NEW. Use "NEW:Full Name"')

    attr_lines = ["0. [UNSPECIFIED]"]
    for aid, name in attributes:
        strict = _ATTR_STRICT_DEFS.get(name, "")
        suffix = f" (STRICT: {strict})" if strict else ""
        attr_lines.append(f"{aid}. {name}{suffix}")
    attr_lines.append('NEW. Use "NEW:Metric Name"')

    content_snippet = message_content
    return (
        "Extract STRUCTURED DATA from this message. Resolve pronouns to entity names.\n\n"
        "Entities (pick # or NEW):\n" + "\n".join(entity_lines) + "\n\n"
        "Attributes (pick # or NEW):\n" + "\n".join(attr_lines) + "\n\n"
        f'Message from {sender}: "{content_snippet}"\n\n' + _LEDGER_JSON_SCHEMA
    )


def _unwrap_brackets(val: str) -> str:
    """Remove a single layer of brackets without stripping inner bracket chars.

    Skips concatenated citations like ``[M1][M2][M3]`` and nested brackets
    like ``[text [nested] stuff]`` where unwrapping would be incorrect.
    """
    val = val.strip()
    if val.startswith("[") and val.endswith("]"):
        if val.find("]") == len(val) - 1:  # first ] is the last char
            return val[1:-1].strip()
    return val


def _try_parse_scientific(raw: str) -> float | None:
    """Try parsing a string as scientific notation or plain float."""
    if not raw:
        return None
    raw = raw.strip().strip("[]")
    try:
        return float(raw)
    except (ValueError, OverflowError):
        return None


def _resolve_entity_or_create(raw, entity_map, topic_id, round_number, _ledger):
    """Resolve entity from JSON value: integer id, 'NEW:Name', or None."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        eid = int(raw)
        return eid if eid != 0 and eid in entity_map else None
    raw_str = str(raw).strip()
    if raw_str.upper().startswith("NEW:"):
        name = raw_str[4:].strip()
        if name:
            return _ledger.add_entity_with_aliases(
                topic_id,
                name,
                None,
                [name.lower()],
                confirmed=False,
            )
        return None
    # Legacy: "NEW (Name)" format
    new_match = re.match(r"NEW\s*\(([^|]+?)\)", raw_str, re.IGNORECASE)
    if new_match:
        name = new_match.group(1).strip()
        if name:
            return _ledger.add_entity_with_aliases(
                topic_id,
                name,
                None,
                [name.lower()],
                confirmed=False,
            )
        return None
    try:
        eid = int(raw_str.strip("[]"))
        return eid if eid != 0 and eid in entity_map else None
    except (ValueError, TypeError):
        return None


def _resolve_attribute_or_create(raw, attr_map, topic_id, _ledger):
    """Resolve attribute from JSON value: integer id, 'NEW:Name', or None."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        aid = int(raw)
        return aid if aid != 0 and aid in attr_map else None
    raw_str = str(raw).strip()
    if raw_str.upper().startswith("NEW:"):
        name = raw_str[4:].strip()
        if name:
            return _ledger.add_attribute_with_aliases(
                topic_id,
                name,
                None,
                [name.lower()],
                confirmed=False,
            )
        return None
    new_match = re.match(r"NEW\s*\(([^|]+?)\)", raw_str, re.IGNORECASE)
    if new_match:
        name = new_match.group(1).strip()
        if name:
            return _ledger.add_attribute_with_aliases(
                topic_id,
                name,
                None,
                [name.lower()],
                confirmed=False,
            )
        return None
    try:
        aid = int(raw_str.strip("[]"))
        return aid if aid != 0 and aid in attr_map else None
    except (ValueError, TypeError):
        return None


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def parse_clerk_ledger_output(
    output: str,
    entity_list: list[tuple[int, str]],
    attribute_list: list[tuple[int, str]],
    topic_id: int,
    subtopic_id: int | None,
    round_number: int,
    sender: str,
) -> list[dict]:
    """Parse JSON or pipe-delimited Clerk output into dicts for normalize_and_upsert."""
    from . import ledger as _ledger

    if not output or output.strip().upper() == "NONE":
        return []

    entity_map = {eid: name for eid, name in entity_list}
    attr_map = {aid: name for aid, name in attribute_list}

    # Try JSON parse first (extract_json_any handles both arrays and objects)
    from .json_utils import extract_json_any

    parsed_json = None
    try:
        parsed_json = extract_json_any(output)
    except Exception:
        pass

    if (
        isinstance(parsed_json, list)
        and parsed_json
        and isinstance(parsed_json[0], dict)
    ):
        return _parse_json_ledger_entries(
            parsed_json,
            entity_map,
            attr_map,
            topic_id,
            subtopic_id,
            round_number,
            sender,
            _ledger,
        )
    if isinstance(parsed_json, dict):
        # Prefer known keys, then fall back to first list value
        entries_list = None
        for key in ("records", "entries", "data", "results"):
            if isinstance(parsed_json.get(key), list):
                entries_list = parsed_json[key]
                break
        if entries_list is None:
            for v in parsed_json.values():
                if isinstance(v, list):
                    entries_list = v
                    break
        if entries_list is not None:
            return _parse_json_ledger_entries(
                entries_list,
                entity_map,
                attr_map,
                topic_id,
                subtopic_id,
                round_number,
                sender,
                _ledger,
            )

    # Fallback: legacy pipe-delimited format
    return _parse_pipe_delimited_ledger(
        output,
        entity_map,
        attr_map,
        topic_id,
        subtopic_id,
        round_number,
        sender,
        _ledger,
    )


_VALID_STAT_TYPES = {
    "point",
    "mean_std",
    "delta",
    "ratio",
    "percentage",
    "p_value",
    "ci",
    "correlation",
    "rank",
    "se",
}


def _parse_json_ledger_entries(
    items,
    entity_map,
    attr_map,
    topic_id,
    subtopic_id,
    round_number,
    sender,
    _ledger,
):
    """Parse a JSON array of ledger entries into result dicts."""
    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        # Qualitative entry
        if item.get("entity") is None and item.get("fact"):
            results.append({"type": "qualitative", "text": str(item["fact"])})
            continue

        entity_id = _resolve_entity_or_create(
            item.get("entity"),
            entity_map,
            topic_id,
            round_number,
            _ledger,
        )
        attribute_id = _resolve_attribute_or_create(
            item.get("attribute"),
            attr_map,
            topic_id,
            _ledger,
        )
        if not entity_id or not attribute_id:
            continue

        baseline_id = None
        if item.get("baseline_entity") is not None:
            baseline_id = _resolve_entity_or_create(
                item["baseline_entity"],
                entity_map,
                topic_id,
                round_number,
                _ledger,
            )

        # Log CoT thought for debugging
        thought = item.get("thought")
        if thought:
            logger.debug("[ledger-cot] %s", thought)

        # Config for model hyperparameters
        config = item.get("config")
        config_json_str = (
            json.dumps(config, ensure_ascii=False) if isinstance(config, dict) else None
        )

        val = item.get("value")
        min_val = _safe_float(item.get("min", val))
        max_val = _safe_float(item.get("max", val))
        raw_value = str(val) if val is not None else ""

        raw_stat = str(item.get("stat_type", "point")).lower()
        stat_type = raw_stat if raw_stat in _VALID_STAT_TYPES else "point"
        value_mean = _safe_float(val) if stat_type == "mean_std" else None

        timeframe = str(item.get("time", "NONE"))
        valid_from, valid_to = _ledger.parse_time_field(timeframe)

        results.append(
            {
                "type": "structured",
                "topic_id": topic_id,
                "subtopic_id": subtopic_id,
                "entity_id": entity_id,
                "attribute_id": attribute_id,
                "raw_value": raw_value,
                "raw_timeframe": timeframe,
                "source_ref": str(item.get("source", "")),
                "unit": str(item.get("unit", "NONE")),
                "created_by": sender,
                "current_round": round_number,
                "min_val": min_val,
                "max_val": max_val,
                "valid_from": valid_from,
                "valid_to": valid_to,
                "stat_type": stat_type,
                "value_mean": value_mean,
                "value_std": _safe_float(item.get("std")),
                "value_p": _safe_float(item.get("p_value")),
                "value_n": _safe_int(item.get("n")),
                "value_ci_lower": _safe_float(item.get("ci_lower")),
                "value_ci_upper": _safe_float(item.get("ci_upper")),
                "value_ci_level": _safe_float(item.get("ci_level")),
                "baseline_entity_id": baseline_id,
                "split": item.get("split"),
                "config_json": config_json_str,
            }
        )
    return results


def _parse_pipe_delimited_ledger(
    output,
    entity_map,
    attr_map,
    topic_id,
    subtopic_id,
    round_number,
    sender,
    _ledger,
):
    """Legacy pipe-delimited parser (backward compatibility fallback)."""
    results = []
    for line in output.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("FACT:"):
            fact_text = line[5:].strip()
            if fact_text:
                results.append({"type": "qualitative", "text": fact_text})
            continue
        if line.upper() == "NONE":
            continue

        parts = {}
        for segment in line.split("|"):
            segment = segment.strip()
            if ":" not in segment:
                continue
            key, _, val = segment.partition(":")
            parts[key.strip().upper()] = val.strip()

        if "ENTITY" not in parts:
            continue

        entity_id = _resolve_entity_or_create(
            parts["ENTITY"].strip(),
            entity_map,
            topic_id,
            round_number,
            _ledger,
        )
        attribute_id = _resolve_attribute_or_create(
            parts.get("ATTR", "").strip(),
            attr_map,
            topic_id,
            _ledger,
        )
        if entity_id is None or attribute_id is None:
            continue

        raw_value = _unwrap_brackets(parts.get("VALUE", ""))
        timeframe = _unwrap_brackets(parts.get("TIME", ""))
        source_ref = _unwrap_brackets(parts.get("SOURCE", ""))
        unit = _unwrap_brackets(parts.get("UNIT", ""))
        min_val = _try_parse_scientific(_unwrap_brackets(parts.get("MIN", "")))
        max_val = _try_parse_scientific(_unwrap_brackets(parts.get("MAX", "")))

        if not raw_value and (min_val is not None or max_val is not None):
            if min_val is not None and max_val is not None:
                raw_value = (
                    f"{min_val}" if min_val == max_val else f"{min_val}-{max_val}"
                )
            elif min_val is not None:
                raw_value = f"{min_val}"
            else:
                raw_value = f"{max_val}"

        valid_from, valid_to = _ledger.parse_time_field(timeframe)

        results.append(
            {
                "type": "structured",
                "topic_id": topic_id,
                "subtopic_id": subtopic_id,
                "entity_id": entity_id,
                "attribute_id": attribute_id,
                "raw_value": raw_value,
                "raw_timeframe": timeframe,
                "source_ref": source_ref,
                "unit": unit,
                "created_by": sender,
                "current_round": round_number,
                "min_val": min_val,
                "max_val": max_val,
                "valid_from": valid_from,
                "valid_to": valid_to,
            }
        )
    return results


def _normalize_fact_proposal_contract(raw_text: str) -> dict:
    parsed = extract_json(raw_text)
    if isinstance(parsed, dict):
        action = parsed.get("action")
        raw_facts = parsed.get("facts")
        if action == "propose_facts" and isinstance(raw_facts, list):
            facts = [
                fact.strip()
                for fact in raw_facts
                if isinstance(fact, str) and fact.strip()
            ]
            return {
                "parsed_ok": True,
                "facts": facts,
            }

    return {
        "parsed_ok": False,
        "facts": [],
    }


def _normalize_fact_direction_contract(raw_text: str) -> dict:
    parsed = _parse_single_json_wrapper(raw_text) or extract_json(raw_text)
    if isinstance(parsed, dict):
        action = parsed.get("action")
        raw_directions = parsed.get("directions")
        if action == "propose_fact_directions" and isinstance(raw_directions, list):
            directions = [
                direction.strip()
                for direction in raw_directions
                if isinstance(direction, str) and direction.strip()
            ]
            return {"parsed_ok": True, "directions": directions}
    return {"parsed_ok": False, "directions": []}


def _normalize_focus_contract(raw_text: str) -> dict:
    parsed = _parse_single_json_wrapper(raw_text) or extract_json(raw_text)
    if isinstance(parsed, dict):
        action = parsed.get("action")
        target = parsed.get("target")
        reason = parsed.get("reason")
        if action == "focus" and isinstance(target, str):
            normalized_target = _normalize_target_name(target)
            if normalized_target and can_special_target(normalized_target):
                raw_grant = parsed.get("grant_web_search", False)
                if isinstance(raw_grant, bool):
                    grant_web_search = raw_grant
                else:
                    grant_web_search = False
                return {
                    "parsed_ok": True,
                    "target": normalized_target,
                    "reason": reason.strip() if isinstance(reason, str) else "",
                    "grant_web_search": grant_web_search,
                }
    return {
        "parsed_ok": False,
        "target": None,
        "reason": "",
        "grant_web_search": False,
    }


def _structured_message_is_usable(
    text: str, accepted_actions: Sequence[str] = ("post_message",)
) -> bool:
    if not usable_text_output(text):
        return False
    parsed = _normalize_message_contract(text, accepted_actions=accepted_actions)
    return parsed.get("parsed_ok", False) and bool(parsed.get("content", "").strip())


def _fact_direction_output_is_usable(text: str) -> bool:
    if not usable_text_output(text):
        return False
    parsed = _normalize_fact_direction_contract(text)
    return parsed.get("parsed_ok", False)


def _fact_list_output_is_usable(text: str) -> bool:
    if not usable_text_output(text):
        return False
    parsed = _normalize_fact_proposal_contract(text)
    return parsed.get("parsed_ok", False)


def _normalize_clerk_fact_candidates_contract(raw_text: str) -> dict:
    parsed = _parse_single_json_wrapper(raw_text) or extract_json(raw_text)
    if isinstance(parsed, dict) and parsed.get("action") == "propose_fact_candidates":
        raw_candidates = parsed.get("fact_candidates")
        if isinstance(raw_candidates, list):
            candidates = []
            for item in raw_candidates:
                if not isinstance(item, dict):
                    continue
                candidate_text = item.get("candidate_text")
                source_excerpt = item.get("source_excerpt")
                source_refs = item.get("source_refs_json") or item.get("source_refs")
                if not isinstance(candidate_text, str) or not candidate_text.strip():
                    continue
                if not isinstance(source_excerpt, str) or not source_excerpt.strip():
                    continue
                if not isinstance(source_refs, list):
                    continue
                normalized_refs = [
                    ref.strip()
                    for ref in source_refs
                    if isinstance(ref, str) and ref.strip()
                ]
                if not normalized_refs:
                    continue
                candidates.append(
                    {
                        "candidate_text": candidate_text.strip(),
                        "candidate_type": "sourced_claim",
                        "source_refs": normalized_refs,
                        "source_excerpt": source_excerpt.strip(),
                    }
                )
            return {"parsed_ok": True, "fact_candidates": candidates}
    return {"parsed_ok": False, "fact_candidates": []}


def _fact_candidates_output_is_usable(text: str) -> bool:
    if not usable_text_output(text):
        return False
    parsed = _normalize_clerk_fact_candidates_contract(text)
    return parsed.get("parsed_ok", False)


def _normalize_clerk_claim_candidates_contract(raw_text: str) -> dict:
    parsed = _parse_single_json_wrapper(raw_text) or extract_json(raw_text)
    if isinstance(parsed, dict) and parsed.get("action") == "propose_claim_candidates":
        raw_candidates = parsed.get("claim_candidates")
        if isinstance(raw_candidates, list):
            candidates = []
            for item in raw_candidates:
                if not isinstance(item, dict):
                    continue
                candidate_text = item.get("candidate_text")
                rationale_short = item.get("rationale_short")
                support_fact_ids = item.get("support_fact_ids_json") or item.get(
                    "support_fact_ids"
                )
                if not isinstance(candidate_text, str) or not candidate_text.strip():
                    continue
                if not isinstance(rationale_short, str) or not rationale_short.strip():
                    continue
                if not isinstance(support_fact_ids, list):
                    continue
                normalized_ids: list[int] = []
                for fact_id in support_fact_ids:
                    try:
                        normalized_ids.append(int(fact_id))
                    except (TypeError, ValueError):
                        continue
                if not normalized_ids:
                    continue
                candidates.append(
                    {
                        "candidate_text": candidate_text.strip(),
                        "support_fact_ids": normalized_ids,
                        "rationale_short": rationale_short.strip(),
                    }
                )
            return {"parsed_ok": True, "claim_candidates": candidates}
    return {"parsed_ok": False, "claim_candidates": []}


def _claim_candidates_output_is_usable(text: str) -> bool:
    if not usable_text_output(text):
        return False
    parsed = _normalize_clerk_claim_candidates_contract(text)
    return parsed.get("parsed_ok", False)


async def _call_text_with_structured_retry(
    *,
    stage_name: str,
    invoke,
    validator,
):
    return await retry_structured_output(
        stage_name=stage_name,
        invoke=invoke,
        is_usable=validator,
        logger=logger,
    )


def _decision_passes(yes_votes: int, total_votes: int) -> bool:
    if total_votes <= 0:
        return False
    return (yes_votes / total_votes) > DECISION_PASS_RATIO


def _build_vote_prompt(
    *,
    question: str,
    topic_summary: str,
    topic_detail: str,
    candidate_summary: Optional[str] = None,
    candidate_detail: Optional[str] = None,
    selected: Optional[Sequence[str]] = None,
    rejected: Optional[Sequence[str]] = None,
) -> str:
    selected_block = ", ".join(selected or []) or "none"
    rejected_block = ", ".join(rejected or []) or "none"
    lines = [
        f"Topic: {topic_summary}",
        f"Topic Detail: {topic_detail}",
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
            "TASK:",
            question,
        ]
    )
    if candidate_summary:
        lines.extend(
            [
                "Vote YES only if the candidate is materially useful for the topic and not redundant with already selected items.",
                "Vote NO if it is redundant, low-value, or off-topic.",
            ]
        )
    lines.append('Reply with strict JSON: {"vote":"yes|no","reason":"short sentence"}.')
    return "\n".join(lines)


async def _run_votes(
    *,
    voters: Sequence[str],
    prompt: str,
    allow_web: bool = False,
) -> tuple[int, int, int, dict[str, Optional[bool]]]:
    decisions: dict[str, Optional[bool]] = {}
    yes_votes = 0
    successful_votes = 0
    failed_votes = 0

    async def _vote_one(voter: str) -> tuple[str, Optional[bool]]:
        agent = get_agent(voter)
        try:
            return voter, await agent.vote(prompt, allow_web=allow_web)
        except Exception:
            return voter, None

    results = await asyncio.gather(*[_vote_one(v) for v in voters])
    for voter, decision in results:
        decisions[voter] = decision
        if decision is None:
            failed_votes += 1
        else:
            successful_votes += 1
            yes_votes += int(decision)
    return yes_votes, successful_votes, failed_votes, decisions


def _format_message_for_prompt(message: dict) -> str:
    parts = []
    msg_id = message.get("id")
    if msg_id is not None:
        parts.append(f"M{msg_id}")
    parts.append(message["sender"])
    if message.get("msg_type") and message.get("msg_type") != "standard":
        parts.append(message["msg_type"])
    label = "|".join(parts)
    suffix = ""
    if message.get("confidence_score") is not None:
        suffix = f" (confidence={message['confidence_score']:.1f}/10)"
    return f"[{label}]{suffix}: {message['content']}"


def _load_context_entities(state: ChatState):
    topic = api.get_topic(state["topic_id"])
    subtopic = api.get_subtopic(state["subtopic_id"])
    return topic, subtopic


def _topic_is_mse_modeling(topic_id: int) -> bool:
    try:
        profile = topic_config.get(topic_id, "domain_profile").strip().lower()
    except Exception as exc:
        logger.debug("[mse-workflow] profile lookup failed: %s", exc)
        return False
    return profile in MSE_PROFILE_VALUES


def _is_mse_modeling_state(state: ChatState | dict) -> bool:
    topic_id = int(state.get("topic_id") or 0)
    return bool(topic_id and _topic_is_mse_modeling(topic_id))


def _mse_workflow_mode(topic_id: int) -> str:
    try:
        value = topic_config.get(topic_id, "mse_workflow_mode").strip().lower()
    except Exception as exc:
        logger.debug("[mse-workflow] mode lookup failed: %s", exc)
        value = MSE_FAST_WORKFLOW
    if value not in {MSE_FAST_WORKFLOW, MSE_REVIEWED_WORKFLOW}:
        return MSE_FAST_WORKFLOW
    return value


def _mse_problem_for_state(state: ChatState | dict) -> dict | None:
    topic_id = int(state.get("topic_id") or 0)
    subtopic_id = state.get("subtopic_id")
    if not topic_id:
        return None
    problems = api.list_optimization_problems(topic_id, limit=500)
    for problem in problems:
        if problem.get("subtopic_id") == subtopic_id:
            return problem
    return problems[0] if problems else None


def _ensure_mse_problem_seed(state: ChatState | dict) -> dict | None:
    existing = _mse_problem_for_state(state)
    if existing:
        return existing
    topic, subtopic = _load_context_entities(state)  # type: ignore[arg-type]
    if not topic:
        return None
    title = (subtopic or {}).get("summary") or topic.get("summary") or "MSE problem"
    source_parts = [
        f"Topic: {topic.get('summary', '')}",
        f"Topic detail: {topic.get('detail', '')}",
    ]
    if subtopic:
        source_parts.extend(
            [
                f"Subtopic: {subtopic.get('summary', '')}",
                f"Subtopic detail: {subtopic.get('detail', '')}",
            ]
        )
    source_text = "\n".join(part for part in source_parts if part.strip())
    from .problem_profile import profile_or_problem_text

    problem_profile = profile_or_problem_text(problem_text=source_text)
    problem_id = api.insert_optimization_problem(
        topic_id=int(state["topic_id"]),
        subtopic_id=state.get("subtopic_id"),
        title=title,
        source_text=source_text,
        problem_class="management_science_optimization",
        domain_context=str(problem_profile.get("orq_coverage") or ""),
        status="candidate",
        source_refs_json=json.dumps(
            [f"topic:{state['topic_id']}", f"subtopic:{state.get('subtopic_id')}"],
            ensure_ascii=True,
        ),
        metadata_json=json.dumps(
            {"problem_profile": problem_profile},
            ensure_ascii=True,
            sort_keys=True,
        ),
        created_by="mse_workflow",
    )
    return api.get_optimization_problem(problem_id)


def _json_ids(raw: Any) -> list[int]:
    if isinstance(raw, list):
        values = raw
    elif raw:
        try:
            values = json.loads(str(raw))
        except (json.JSONDecodeError, TypeError):
            values = []
    else:
        values = []
    ids: list[int] = []
    for value in values:
        try:
            ids.append(int(value))
        except (TypeError, ValueError):
            continue
    return ids


def _insert_model_diagnostics_once(
    *,
    problem_id: int,
    topic_id: int,
    issues: Sequence[Any],
) -> list[int]:
    existing = {
        (str(item.get("diagnostic_type") or ""), str(item.get("message") or ""))
        for item in api.get_model_diagnostics(problem_id, status="open")
    }
    diagnostic_ids: list[int] = []
    for issue in issues:
        issue_type = str(getattr(issue, "issue_type", "") or "")
        severity = str(getattr(issue, "severity", "") or "warning")
        message = str(getattr(issue, "message", "") or "")
        if not issue_type or not message:
            continue
        key = (issue_type, message)
        if key in existing:
            continue
        diagnostic_ids.append(
            api.insert_model_diagnostic(
                problem_id=problem_id,
                topic_id=topic_id,
                diagnostic_type=issue_type,
                severity=severity,
                message=message,
            )
        )
        existing.add(key)
    return diagnostic_ids


def _mse_latest_artifact_status(state: ChatState | dict) -> dict[str, Any]:
    mode = _mse_workflow_mode(int(state.get("topic_id") or 0))
    problem = _mse_problem_for_state(state)
    if not problem:
        return {
            "mode": mode,
            "status": "problem_framing",
            "phase": MSE_PROBLEM_PHASE,
            "problem": None,
            "components": [],
            "model_irs": [],
            "artifacts": [],
            "solver_runs": [],
            "open_diagnostics": [],
            "solved": False,
        }

    components = api.get_optimization_components(problem["id"])
    model_irs = api.get_optimization_model_irs(problem["id"])
    artifacts = api.get_optimization_artifacts(problem["id"])
    solver_runs = api.get_solver_runs(problem["id"])
    diagnostics = api.get_model_diagnostics(problem["id"], status="open")
    active_solver_claims = [
        claim
        for claim in api.get_claims(int(state["topic_id"]), limit=1000)
        if claim.get("claim_type") == "optimization_result"
        and claim.get("status", "active") == "active"
    ]
    pending_solver_claims = [
        candidate
        for candidate in api.get_claim_candidates(
            int(state["topic_id"]), status="pending", limit=1000
        )
        if candidate.get("claim_type") == "optimization_result"
    ]

    valid_artifacts = [
        artifact for artifact in artifacts if artifact.get("parser_status") == "valid"
    ]
    latest_artifact = artifacts[0] if artifacts else None
    latest_valid_artifact = valid_artifacts[0] if valid_artifacts else None
    latest_solver_run = solver_runs[0] if solver_runs else None
    ready_statuses = {"reviewed", "formalized", "executable", "verified"}
    ready_components = [
        component
        for component in components
        if component.get("review_status") in ready_statuses
    ]
    lp_ready_components = [
        component
        for component in ready_components
        if component.get("component_type")
        in {"objective", "constraint", "decision_variable"}
    ]
    candidate_components = [
        component
        for component in components
        if component.get("review_status") == "candidate"
    ]
    from .optimization import blocking_specification_issues

    specification_issues = (
        blocking_specification_issues(components, require_formal=True)
        if components
        else []
    )

    if latest_solver_run and latest_solver_run.get("status") == "optimal":
        if active_solver_claims:
            status = "solved"
            phase = MSE_SOLVED_PHASE
        elif pending_solver_claims:
            status = "managerial_synthesis"
            phase = MSE_MANAGERIAL_PHASE
        else:
            status = "managerial_synthesis"
            phase = MSE_MANAGERIAL_PHASE
    elif latest_solver_run:
        status = "solver_repair"
        phase = MSE_REPAIR_PHASE
    elif latest_valid_artifact:
        status = "solver_execution"
        phase = MSE_SOLVER_PHASE
    elif latest_artifact:
        status = "artifact_repair"
        phase = MSE_REPAIR_PHASE
    elif not components:
        status = "component_extraction"
        phase = MSE_COMPONENT_PHASE
    elif mode == MSE_REVIEWED_WORKFLOW and candidate_components:
        status = "component_review"
        phase = MSE_COMPONENT_REVIEW_PHASE
    elif specification_issues:
        status = "specification_gap"
        phase = MSE_SPECIFICATION_PHASE
    elif len(lp_ready_components) >= 3 or mode == MSE_FAST_WORKFLOW:
        status = "artifact_generation"
        phase = MSE_ARTIFACT_PHASE
    else:
        status = "component_review"
        phase = MSE_COMPONENT_REVIEW_PHASE

    return {
        "mode": mode,
        "status": status,
        "phase": phase,
        "problem": problem,
        "components": components,
        "model_irs": model_irs,
        "artifacts": artifacts,
        "solver_runs": solver_runs,
        "open_diagnostics": diagnostics,
        "active_solver_claims": active_solver_claims,
        "pending_solver_claims": pending_solver_claims,
        "specification_issues": [issue.__dict__ for issue in specification_issues],
        "latest_artifact": latest_artifact,
        "latest_valid_artifact": latest_valid_artifact,
        "latest_solver_run": latest_solver_run,
        "solved": status == "solved",
    }


def _mse_solver_backend_for_artifact(artifact: dict[str, Any]) -> str | None:
    language = str(artifact.get("model_language") or "").lower()
    content = str(artifact.get("content") or "")
    if language == "lp":
        if re.search(r"(?im)^\s*(binary|binaries|general|generals)\b", content):
            return "scipy_milp"
        return "scipy_linprog"
    if language == "mps":
        return "scipy_mps"
    return None


def _advance_mse_workflow_deterministically(state: ChatState | dict) -> dict[str, Any]:
    """Advance solver-ready MSE artifacts without spending another agent turn."""
    if not _is_mse_modeling_state(state):
        return {}
    problem = _ensure_mse_problem_seed(state)
    if not problem:
        return _mse_latest_artifact_status(state)

    from .optimization import (
        blocking_specification_issues,
        build_component_fingerprints_json,
        create_solver_claim_candidate,
        persist_model_ir_from_components,
        persist_lp_artifact_from_components,
        review_pending_solver_claim_candidates,
        solve_optimization_artifact,
    )

    mode = _mse_workflow_mode(int(state["topic_id"]))
    components = api.get_optimization_components(problem["id"])
    blocking_issues = (
        blocking_specification_issues(components, require_formal=True)
        if components
        else []
    )
    if blocking_issues:
        _insert_model_diagnostics_once(
            problem_id=int(problem["id"]),
            topic_id=int(state["topic_id"]),
            issues=blocking_issues,
        )
        return _mse_latest_artifact_status(state)

    if mode == MSE_FAST_WORKFLOW:
        for component in components:
            if component.get("review_status") == "candidate":
                api.update_optimization_component_review(
                    int(component["id"]),
                    review_status="reviewed",
                    validation_notes=(
                        "Auto-reviewed by modeling_fast after deterministic "
                        "specification checks passed."
                    ),
                )

    snapshot = _mse_latest_artifact_status(state)
    if snapshot["status"] == "artifact_generation":
        components = api.get_optimization_components(problem["id"])
        fingerprints = build_component_fingerprints_json(components)
        latest_ir = (api.get_optimization_model_irs(int(problem["id"])) or [None])[0]
        if not latest_ir or latest_ir.get("component_fingerprints_json") != fingerprints:
            persist_model_ir_from_components(
                topic_id=int(state["topic_id"]),
                problem=problem,
                components=components,
                require_formal=True,
                generator_role=f"mse_{mode}",
            )
        try:
            persist_lp_artifact_from_components(
                topic_id=int(state["topic_id"]),
                problem_id=int(problem["id"]),
                components=components,
                require_reviewed=(mode == MSE_REVIEWED_WORKFLOW),
                generator_role=f"mse_{mode}",
            )
        except Exception as exc:
            api.insert_model_diagnostic(
                problem_id=int(problem["id"]),
                topic_id=int(state["topic_id"]),
                diagnostic_type="artifact_generation_error",
                severity="error",
                message=str(exc),
            )

    snapshot = _mse_latest_artifact_status(state)
    artifact = snapshot.get("latest_valid_artifact")
    if snapshot["status"] == "solver_execution" and artifact:
        backend = _mse_solver_backend_for_artifact(artifact)
        if backend:
            try:
                solve_optimization_artifact(
                    topic_id=int(state["topic_id"]),
                    problem_id=int(problem["id"]),
                    artifact_id=int(artifact["id"]),
                    content=str(artifact.get("content") or ""),
                    model_language=str(artifact.get("model_language") or ""),
                    solver_backend=backend,
                )
            except Exception as exc:
                api.insert_model_diagnostic(
                    problem_id=int(problem["id"]),
                    topic_id=int(state["topic_id"]),
                    artifact_id=int(artifact["id"]),
                    diagnostic_type="solver_dispatch_error",
                    severity="error",
                    message=str(exc),
                )

    snapshot = _mse_latest_artifact_status(state)
    run = snapshot.get("latest_solver_run")
    artifact = snapshot.get("latest_artifact")
    if (
        run
        and artifact
        and run.get("status") == "optimal"
        and not snapshot.get("active_solver_claims")
    ):
        try:
            create_solver_claim_candidate(
                topic_id=int(state["topic_id"]),
                subtopic_id=state.get("subtopic_id"),
                problem=problem,
                artifact=artifact,
                solver_run=run,
            )
            review_pending_solver_claim_candidates(int(state["topic_id"]))
        except Exception as exc:
            api.insert_model_diagnostic(
                problem_id=int(problem["id"]),
                topic_id=int(state["topic_id"]),
                artifact_id=int(artifact["id"]),
                solver_run_id=int(run["id"]),
                diagnostic_type="solver_claim_generation_error",
                severity="warning",
                message=str(exc),
            )
    return _mse_latest_artifact_status(state)


def get_phase_for_round(round_number: int) -> str:
    if round_number <= 1:
        return OPENING_PHASE
    if round_number == 2:
        return EVIDENCE_PHASE
    return ANALYSIS_PHASE


def _make_turn(actor: str, turn_kind: str = BASE_TURN) -> TurnSpec:
    return {"actor": actor, "turn_kind": turn_kind}


def build_base_turns_for_phase(phase: str) -> list[TurnSpec]:
    roster = OPENING_ROSTER if phase == OPENING_PHASE else FULL_ROSTER
    return [_make_turn(actor) for actor in roster]


def build_mse_stage_for_state(state: ChatState | dict) -> tuple[str, list[dict]]:
    snapshot = _mse_latest_artifact_status(state)
    if snapshot.get("solved"):
        return MSE_SOLVED_PHASE, []
    actor = MSE_STEP_ACTORS.get(snapshot["status"], "analyst")
    return snapshot["phase"], [{"agents": [_make_turn(actor)], "parallel": False}]


def build_extra_turns(state: ChatState) -> list[TurnSpec]:
    valid_targets = set(ordinary_deliberators())
    extras: list[TurnSpec] = []

    if state.get("tron_target") in valid_targets:
        extras.append(_make_turn(state["tron_target"], TRON_REMEDIATION_TURN))
    if state.get("dog_target") in valid_targets:
        extras.append(_make_turn(state["dog_target"], DOG_CORRECTION_TURN))
    if state.get("cat_target") in valid_targets:
        extras.append(_make_turn(state["cat_target"], CAT_EXPANSION_TURN))
    return extras


def build_turn_queue_for_round(
    state: ChatState, round_number: int
) -> tuple[str, list[TurnSpec]]:
    if _is_mse_modeling_state(state):
        phase, stages = build_mse_stage_for_state(state)
        turns = [turn for stage in stages for turn in stage["agents"]]
        return phase, turns
    phase = get_phase_for_round(round_number)
    turns = build_base_turns_for_phase(phase)
    return phase, turns


# Agent grouping constants for stage-based parallel execution
BUILDERS = ["dreamer", "scientist", "engineer", "analyst"]
CRITICS = ["critic", "contrarian"]


MAX_TURNS_PER_AGENT_PER_ROUND = 2
_EXEMPT_FROM_CAP = {"skynet", "writer", "librarian", "fact_proposer"}


def _apply_turn_cap(stages: list[dict]) -> list[dict]:
    """DE-6: Enforce per-agent turn cap across all stages in a round."""
    counts: dict[str, int] = {}
    for stage in stages:
        filtered_agents = []
        for turn in stage["agents"]:
            actor = turn["actor"]
            if actor in _EXEMPT_FROM_CAP:
                filtered_agents.append(turn)
                continue
            counts[actor] = counts.get(actor, 0) + 1
            if counts[actor] <= MAX_TURNS_PER_AGENT_PER_ROUND:
                filtered_agents.append(turn)
        stage["agents"] = filtered_agents
    return stages


def build_stages_for_round(round_number: int, state: ChatState | dict | None = None) -> list[dict]:
    """Build stage-based execution plan for a round.

    R1 (OPENING): builders + tron parallel → critic alone
    R2 (EVIDENCE): builders + cat/tron/spectator parallel → critics + dog parallel
    R3+ (analysis): all agents sequential
    """
    if state and _is_mse_modeling_state(state):
        _, stages = build_mse_stage_for_state(state)
        return stages
    phase = get_phase_for_round(round_number)
    if phase == OPENING_PHASE:
        # R1: builders + tron parallel, then critic alone
        stage1 = [_make_turn(a) for a in BUILDERS + ["tron"]]
        stage2 = [_make_turn("critic")]
        stages = [
            {"agents": stage1, "parallel": True},
            {"agents": stage2, "parallel": True},
        ]
    elif phase == EVIDENCE_PHASE:
        # R2: builders + cat/tron/spectator parallel, then critics + dog parallel
        stage1 = [_make_turn(a) for a in BUILDERS + ["cat", "tron", SPECTATOR]]
        stage2 = [_make_turn(a) for a in CRITICS + ["dog"]]
        stages = [
            {"agents": stage1, "parallel": True},
            {"agents": stage2, "parallel": True},
        ]
    else:
        # R3+ analysis: sequential (single stage)
        all_turns = [_make_turn(a) for a in FULL_ROSTER]
        stages = [{"agents": all_turns, "parallel": False}]
    return _apply_turn_cap(stages)


def _build_intervention_turns(targets: dict) -> list[TurnSpec]:
    """Build intervention turn specs from extracted targets."""
    valid_targets = set(ordinary_deliberators())
    interventions: list[TurnSpec] = []
    if targets.get("dog_target") in valid_targets:
        interventions.append(_make_turn(targets["dog_target"], DOG_CORRECTION_TURN))
    if targets.get("cat_target") in valid_targets:
        interventions.append(_make_turn(targets["cat_target"], CAT_EXPANSION_TURN))
    if targets.get("tron_target") in valid_targets:
        interventions.append(_make_turn(targets["tron_target"], TRON_REMEDIATION_TURN))
    return interventions


def _replace_extra_turns(
    pending_turns: list[TurnSpec], extra_turns: list[TurnSpec]
) -> list[TurnSpec]:
    base_turns = [
        turn for turn in pending_turns if turn.get("turn_kind", BASE_TURN) == BASE_TURN
    ]
    return base_turns + extra_turns


def _refresh_pending_turns_with_extras(state: ChatState, updates: dict) -> None:
    phase = updates.get("phase") or state.get(
        "phase", get_phase_for_round(state.get("round_number", 1))
    )
    if phase == OPENING_PHASE:
        return

    merged_state = dict(state)
    merged_state.update(updates)
    pending_turns = list(updates.get("pending_turns", state.get("pending_turns", [])))
    updates["pending_turns"] = _replace_extra_turns(
        pending_turns, build_extra_turns(merged_state)
    )


def _clear_consumed_extra_target(turn_kind: str, updates: dict) -> None:
    if turn_kind == TRON_REMEDIATION_TURN:
        updates["tron_target"] = None
    elif turn_kind == DOG_CORRECTION_TURN:
        updates["dog_target"] = None
    elif turn_kind == CAT_EXPANSION_TURN:
        updates["cat_target"] = None


def _pending_extra_turns(state: ChatState) -> list[TurnSpec]:
    return [
        turn
        for turn in state.get("pending_turns", [])
        if turn.get("turn_kind", BASE_TURN) != BASE_TURN
    ]


def _termination_policy_for_round(round_number: int) -> tuple[str, str]:
    if round_number <= 3:
        return (
            "weak",
            "EARLY STAGE. The burden of proof is on continuing. If any central blocker remains, or if the recommendation is still shifting, you should continue.",
        )
    if round_number <= 5:
        return (
            "medium",
            "MID STAGE. Close only when the remaining disagreement is peripheral or repetitive. Continue if the recommendation is still unstable or a central branch remains.",
        )
    if round_number <= 6:
        return (
            "strong",
            "LATE STAGE. The burden of proof is on closing, but you must continue if a severe central blocker still makes the current recommendation unstable or unsafe.",
        )
    return ("forced", "Round 7 is a forced close.")


def _should_run_termination_vote(round_number: int) -> bool:
    return round_number >= 3


def _build_termination_question(stage_guidance: str) -> str:
    return (
        "Decide whether this subtopic should CONTINUE or CLOSE.\n"
        f"Current stage guidance: {stage_guidance}\n"
        "Fill every field before you choose the final vote.\n"
        "Field rules:\n"
        "- `main_branch`: name the main unresolved branch; use `none` only if no meaningful blocker remains.\n"
        "- `centrality`: use `central`, `mixed`, `peripheral`, or `none`.\n"
        "- `recent_shift`: use `yes`, `no`, or `unclear` based on whether the framing, governing metric, or recommendation changed in the last 1-2 rounds.\n"
        "- `conditional_support`: use `yes` if the current recommendation still relies on softened, caveated, or weakly validated facts.\n"
        "- `untested_novelty`: use `yes` if a new framework, router, metric, or failure model affecting the recommendation has not yet been stress-tested.\n"
        "A subtopic can be legitimately CLOSED if it reaches ONE of the following end states:\n"
        "1. [Hard Consensus]: A logically sound conclusion supported by evidence with no remaining central blockers.\n"
        "2. [Constructive Suspension - Empirical/Data Gap]: The team hits the boundary of non-executable constraints (e.g., requires physical testing or longitudinal data collection that doesn't exist), BUT has generated a rigorous, falsifiable [EXPERIMENTAL BLUEPRINT] or [DATA GATHERING METHODOLOGY] (must include exact variables, target populations, or metrics) to resolve the unknown.\n"
        "3. [Constructive Suspension - Trade-off]: The team hits an unresolvable value conflict and produces a detailed [DECISION MATRIX] showing under what specific conditions Path A or Path B should be chosen.\n"
        "4. [Constructive Suspension - Epistemological/Data Gap]: The team realizes the required demographic, historical, or longitudinal data simply does not exist anywhere. The team generates a practical, adaptive [HEURISTIC FRAMEWORK] to operate safely in the absence of this data, rather than infinitely escalating the scope of the problem.\n"
        "Default voting policy:\n"
        "- If `centrality` is `central` or `mixed`, default to `continue`, UNLESS a Constructive Suspension state has been reached.\n"
        "- If `recent_shift` is `yes` or `unclear`, default to `continue`.\n"
        "- If `conditional_support` is `yes`, default to `continue`.\n"
        "- If `untested_novelty` is `yes`, default to `continue`.\n"
        "- If you vote `close`, `reason` MUST contain a short explanation of WHICH end state was reached (Hard Consensus, Empirical Gap, Trade-off, or Data Gap) and why.\n"
        "- If you vote `continue`, `reason` MUST explain what specific blocker or untested novelty remains.\n"
        'Reply with strict JSON only: {"main_branch":"...","centrality":"central|mixed|peripheral|none","recent_shift":"yes|no|unclear","conditional_support":"yes|no","untested_novelty":"yes|no","vote":"continue|close","reason":"... (Mandatory explanation for your vote)"}.'
    )


def _build_termination_vote_prompt(
    *,
    topic_summary: str,
    topic_detail: str,
    stage_guidance: str,
    topic_id: int = 0,
) -> str:
    lines = [
        f"Topic: {topic_summary}",
        f"Topic Detail: {topic_detail}",
    ]
    # DE-7: Coverage stats
    if topic_id > 0:
        try:
            fact_count = api.count_facts(topic_id)
            claim_count = api.count_claims(topic_id)
            code_count = api.count_code_evidence(topic_id)
            lines.append(
                f"Evidence: {fact_count} facts, {claim_count} claims, {code_count} code experiments"
            )
        except Exception as exc:
            logger.debug("[termination] Failed to fetch coverage stats: %s", exc)
    lines.extend(
        [
            "",
            "TASK: SUBTOPIC CLOSURE GOVERNANCE",
            _build_termination_question(stage_guidance),
        ]
    )
    return "\n".join(lines)


def _build_termination_vote_repair_prompt(
    *, original_prompt: str, invalid_text: str, invalid_reason: str
) -> str:
    return (
        "Original governance task:\n"
        f"{original_prompt}\n\n"
        "Invalid governance response:\n"
        f"{invalid_text}\n\n"
        f"Validation failure: {invalid_reason}\n\n"
        "Rewrite the response into valid JSON using exactly this schema:\n"
        '{"main_branch":"...","centrality":"central|mixed|peripheral|none","recent_shift":"yes|no|unclear","conditional_support":"yes|no","untested_novelty":"yes|no","vote":"continue|close","reason":"... (Mandatory)","override_reason":"... (Required if vote is close and any blocker is true)"}\n'
        "Preserve the original intent when possible.\n"
        "Output JSON only. Do not add markdown fences, commentary, or extra keys."
    )


async def _repair_summary_by_decomposition(
    flawed_content: str, provider: str = "minimax"
) -> str:
    logger.warning(
        "[skynet] Summary missing headers. Initiating decomposition repair..."
    )

    async def extract_section(header: str) -> str:
        prompt = (
            f"Here is a raw, unformatted summary of a discussion. Your task is to extract ONLY the information "
            f"pertaining to the section '{header}'.\n"
            f"Re-write it so your output starts EXACTLY with '{header}'.\n"
            f"If the information is completely missing from the text, output exactly '{header}\nUnknown.'\n\n"
            f"Raw summary:\n{flawed_content}"
        )

        for attempt in range(2):
            try:
                resp = await call_text(
                    prompt,
                    provider=provider,
                    strategy="direct",
                    allow_web=False,
                    system_instruction="You are a strict text formatting assistant. Extract only what is requested. Output plain text, absolutely no markdown fences like ``` or ```json.",
                    fallback_role="skynet",
                    require_json=False,
                )
                if not resp:
                    continue

                # Strip any markdown fences just in case
                resp = re.sub(
                    r"^```[a-zA-Z]*\n|```$", "", resp.strip(), flags=re.MULTILINE
                ).strip()

                ascii_ratio = sum(1 for c in resp if ord(c) < 128) / max(len(resp), 1)
                if len(resp) > 50 and ascii_ratio < 0.6:
                    logger.warning(
                        "[skynet] Section '%s' appears non-English (ascii_ratio=%.2f); discarding.",
                        header,
                        ascii_ratio,
                    )
                    continue

                # Check if it properly starts with the required header
                if not resp.startswith(header):
                    # Force the prefix if it generated useful content but forgot the header
                    if len(resp) > 20:
                        return f"{header}\n{resp}"
                    continue  # Try again if it's completely malformed

                return resp
            except Exception as e:
                logger.warning(
                    f"[skynet] Attempt {attempt+1} failed to extract {header}: {e}"
                )

        # Ultimate fallback if both attempts fail or return garbage
        return f"{header}\nUnknown."

    tasks = [extract_section(header) for header in SUMMARY_SECTION_HEADERS]
    results = await asyncio.gather(*tasks)
    return "\n\n".join(results)


def _has_required_summary_sections(content: str) -> bool:
    line_cursor = -1
    lines = (content or "").splitlines()
    for header in SUMMARY_SECTION_HEADERS:
        position = next(
            (
                index
                for index, line in enumerate(lines)
                if index > line_cursor and line.strip().startswith(header)
            ),
            -1,
        )
        if position < 0:
            return False
        line_cursor = position
    return True


def _normalize_yes_no(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"yes", "y", "true", "1"}:
            return "yes"
        if normalized in {"no", "n", "false", "0"}:
            return "no"
    return None


def _normalize_centrality(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    aliases = {
        "central": "central",
        "core": "central",
        "mixed": "mixed",
        "peripheral": "peripheral",
        "secondary": "peripheral",
        "none": "none",
    }
    return aliases.get(normalized)


def _normalize_recent_shift(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        normalized = value.strip().lower()
        aliases = {
            "yes": "yes",
            "y": "yes",
            "true": "yes",
            "changed": "yes",
            "no": "no",
            "n": "no",
            "false": "no",
            "stable": "no",
            "unclear": "unclear",
            "unknown": "unclear",
            "maybe": "unclear",
        }
        return aliases.get(normalized)
    return None


def _normalize_termination_vote_label(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return "close" if value else "continue"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"close", "yes", "approve"}:
            return "close"
        if normalized in {"continue", "no", "reject", "keep_open"}:
            return "continue"
    return None


def _empty_termination_vote(reason: str) -> dict[str, Any]:
    return {
        "parsed_ok": False,
        "main_branch": "none",
        "centrality": "none",
        "recent_shift": "unclear",
        "conditional_support": "no",
        "untested_novelty": "no",
        "vote": "continue",
        "reason": None,
        "override_reason": None,
        "central_blocker": False,
        "volatility_blocker": True,
        "support_blocker": False,
        "novelty_blocker": False,
        "invalid_reason": reason,
    }


def _normalize_termination_vote_contract(raw_text: str) -> dict[str, Any]:
    parsed = _parse_single_json_wrapper(raw_text) or extract_json(raw_text)
    if not isinstance(parsed, dict):
        return _empty_termination_vote("invalid_json")

    centrality = _normalize_centrality(parsed.get("centrality"))
    recent_shift = _normalize_recent_shift(parsed.get("recent_shift"))
    conditional_support = _normalize_yes_no(parsed.get("conditional_support"))
    untested_novelty = _normalize_yes_no(parsed.get("untested_novelty"))
    vote = _normalize_termination_vote_label(parsed.get("vote"))

    if not all((centrality, recent_shift, conditional_support, untested_novelty, vote)):
        return _empty_termination_vote("invalid_fields")

    raw_branch = parsed.get("main_branch")
    if isinstance(raw_branch, str) and raw_branch.strip():
        main_branch = raw_branch.strip()
    elif centrality == "none":
        main_branch = "none"
    else:
        main_branch = "unspecified"

    raw_override = parsed.get("override_reason") or parsed.get("reason")
    override_reason = (
        raw_override.strip()
        if isinstance(raw_override, str) and raw_override.strip()
        else None
    )

    central_blocker = centrality in {"central", "mixed"}
    volatility_blocker = recent_shift in {"yes", "unclear"}
    support_blocker = conditional_support == "yes"
    novelty_blocker = untested_novelty == "yes"
    has_blocker = (
        central_blocker or volatility_blocker or support_blocker or novelty_blocker
    )

    if vote == "close" and has_blocker and not override_reason:
        result = _empty_termination_vote("missing_override_reason")
        result.update(
            {
                "main_branch": main_branch,
                "centrality": centrality,
                "recent_shift": recent_shift,
                "conditional_support": conditional_support,
                "untested_novelty": untested_novelty,
                "vote": vote,
                "central_blocker": central_blocker,
                "volatility_blocker": volatility_blocker,
                "support_blocker": support_blocker,
                "novelty_blocker": novelty_blocker,
            }
        )
        return result

    raw_reason = parsed.get("reason")
    reason = (
        raw_reason.strip()
        if isinstance(raw_reason, str) and raw_reason.strip()
        else None
    )

    return {
        "parsed_ok": True,
        "main_branch": main_branch,
        "centrality": centrality,
        "recent_shift": recent_shift,
        "conditional_support": conditional_support,
        "untested_novelty": untested_novelty,
        "vote": vote,
        "reason": reason,
        "override_reason": override_reason,
        "central_blocker": central_blocker,
        "volatility_blocker": volatility_blocker,
        "support_blocker": support_blocker,
        "novelty_blocker": novelty_blocker,
        "invalid_reason": None,
    }


async def _run_termination_votes(
    *,
    voters: Sequence[str],
    prompt: str,
    topic_id: int,
    subtopic_id: int,
    round_number: int,
    subject: str,
) -> list[dict[str, Any]]:
    vote_provider = _resolve_topic_provider(topic_id, "vote_provider")

    async def _vote_one(voter: str) -> dict[str, Any]:
        """Run a single voter's LLM call (+ optional repair). No DB writes."""
        agent = get_agent(voter)
        raw_response = ""
        repair_response = ""
        repair_used = False
        try:
            raw_response = await agent.governance_vote(
                prompt, provider_profile=vote_provider
            )
            parsed = _normalize_termination_vote_contract(raw_response)
            if not parsed["parsed_ok"] and raw_response.strip():
                repair_used = True
                try:
                    repair_response = await call_text(
                        _build_termination_vote_repair_prompt(
                            original_prompt=prompt,
                            invalid_text=raw_response,
                            invalid_reason=parsed["invalid_reason"] or "unknown",
                        ),
                        provider=vote_provider,
                        strategy="direct",
                        allow_web=False,
                        system_instruction=(
                            f"{agent.spec.role_prompt}\n\n"
                            "GOVERNANCE JSON REPAIR MODE:\n"
                            "You are repairing a subtopic termination governance vote that failed validation.\n"
                            "Preserve the original intent when possible.\n"
                            "Output valid JSON only with the exact requested keys.\n"
                            "Do not add markdown fences, commentary, or extra keys."
                        ).strip(),
                        temperature=0.1,
                        max_tokens=DEFAULT_MAX_TOKENS,
                        fallback_role=voter,
                        require_json=True,
                    )
                    if not usable_text_output(repair_response):
                        logger.warning(
                            "[GovVote] agent=%s repair returned unusable text", voter
                        )
                    else:
                        repaired = _normalize_termination_vote_contract(repair_response)
                        if repaired["parsed_ok"]:
                            parsed = repaired
                        else:
                            logger.warning(
                                "[GovVote] agent=%s repair failed invalid_reason=%s repair_response=%s",
                                voter,
                                repaired["invalid_reason"],
                                repair_response,
                            )
                except Exception as exc:
                    logger.warning(
                        "[GovVote] agent=%s repair failed with exception: %s",
                        voter,
                        exc,
                    )
        except Exception as exc:
            parsed = _empty_termination_vote(f"exception:{type(exc).__name__}")
            logger.warning("[GovVote] agent=%s execution failed: %s", voter, exc)
        return {
            "voter": voter,
            "raw_response": raw_response,
            "repair_used": repair_used,
            "repair_response": repair_response,
            "parsed": parsed,
        }

    # Run all voter LLM calls concurrently
    results = await asyncio.gather(
        *[_vote_one(v) for v in voters], return_exceptions=True
    )

    # Serialize DB writes and logging
    records: list[dict[str, Any]] = []
    for voter, result in zip(voters, results):
        if isinstance(result, Exception):
            logger.warning("[GovVote] agent=%s _vote_one raised: %s", voter, result)
            parsed = _empty_termination_vote(f"exception:{type(result).__name__}")
            record: dict[str, Any] = {
                "voter": voter,
                "raw_response": "",
                "repair_used": False,
                "repair_response": "",
                "parsed": parsed,
            }
        else:
            record = result
            parsed = record["parsed"]

        raw_response = record["raw_response"]
        repair_used = record["repair_used"]
        repair_response = record["repair_response"]

        # Extract reason safely, defaulting to empty string if missing or None
        vote_reason = parsed.get("reason") or parsed.get("override_reason") or ""

        api.insert_vote_record(
            topic_id,
            subtopic_id,
            round_number,
            "termination",
            subject,
            prompt,
            voter,
            bool(parsed["parsed_ok"]),
            parsed["vote"] if parsed["parsed_ok"] else None,
            vote_reason,
            raw_response,
            metadata_json=json.dumps(
                {
                    "main_branch": parsed["main_branch"],
                    "centrality": parsed["centrality"],
                    "recent_shift": parsed["recent_shift"],
                    "conditional_support": parsed["conditional_support"],
                    "untested_novelty": parsed["untested_novelty"],
                    "override_reason": parsed["override_reason"],
                    "invalid_reason": parsed["invalid_reason"],
                    "repair_used": repair_used,
                    "repair_response": repair_response,
                },
                ensure_ascii=True,
            ),
        )
        logger.info(
            "[GovVote] agent=%s parsed_ok=%s vote=%s centrality=%s recent_shift=%s conditional_support=%s untested_novelty=%s override_reason=%s invalid_reason=%s repair_used=%s raw_response=%s repair_response=%s",
            voter,
            parsed["parsed_ok"],
            parsed["vote"],
            parsed["centrality"],
            parsed["recent_shift"],
            parsed["conditional_support"],
            parsed["untested_novelty"],
            parsed["override_reason"],
            parsed["invalid_reason"],
            repair_used,
            raw_response,
            repair_response,
        )
        records.append(record)
    return records


def _termination_thresholds_for_round(round_number: int) -> dict[str, float | int]:
    if round_number <= 3:
        return {
            "close_ratio": ROUND3_CLOSE_RATIO,
            "central_blocker": 1,
            "volatility_blocker": 1,
            "support_blocker": 1,
            "novelty_blocker": 1,
        }
    if round_number <= 6:
        return {
            "close_ratio": ROUND46_CLOSE_RATIO,
            "central_blocker": 2,
            "volatility_blocker": 2,
            "support_blocker": 2,
            "novelty_blocker": 2,
        }
    return {
        "close_ratio": ROUND79_CLOSE_RATIO,
        "central_blocker": 2,
        "volatility_blocker": 2,
        "support_blocker": 3,
        "novelty_blocker": 3,
    }


def _aggregate_termination_votes(
    vote_records: Sequence[dict[str, Any]], round_number: int
) -> dict[str, Any]:
    valid_votes = [
        record["parsed"]
        for record in vote_records
        if record.get("parsed", {}).get("parsed_ok")
    ]
    blocker_signal_votes = [
        record["parsed"]
        for record in vote_records
        if record.get("parsed", {}).get("parsed_ok")
        or record.get("parsed", {}).get("invalid_reason") == "missing_override_reason"
    ]
    invalid_votes = len(vote_records) - len(valid_votes)
    close_votes = sum(1 for parsed in valid_votes if parsed["vote"] == "close")
    close_ratio = close_votes / len(valid_votes) if valid_votes else 0.0
    blocker_counts = {
        "central_blocker": sum(
            1 for parsed in blocker_signal_votes if parsed["central_blocker"]
        ),
        "volatility_blocker": sum(
            1 for parsed in blocker_signal_votes if parsed["volatility_blocker"]
        ),
        "support_blocker": sum(
            1 for parsed in blocker_signal_votes if parsed["support_blocker"]
        ),
        "novelty_blocker": sum(
            1 for parsed in blocker_signal_votes if parsed["novelty_blocker"]
        ),
    }

    if invalid_votes > TERMINATION_MAX_INVALID_VOTES:
        return {
            "subtopic_exhausted": False,
            "valid_votes": len(valid_votes),
            "invalid_votes": invalid_votes,
            "close_votes": close_votes,
            "close_ratio": close_ratio,
            "blocker_counts": blocker_counts,
            "blocked_by": ["invalid_votes"],
        }

    thresholds = _termination_thresholds_for_round(round_number)
    blocked_by = [
        category
        for category, count in blocker_counts.items()
        if count >= thresholds[category]
    ]
    subtopic_exhausted = (
        bool(valid_votes)
        and not blocked_by
        and close_ratio >= thresholds["close_ratio"]
    )
    return {
        "subtopic_exhausted": subtopic_exhausted,
        "valid_votes": len(valid_votes),
        "invalid_votes": invalid_votes,
        "close_votes": close_votes,
        "close_ratio": close_ratio,
        "blocker_counts": blocker_counts,
        "blocked_by": blocked_by,
    }


def _normalize_target_name(name: str) -> Optional[str]:
    raw_name = (name or "").strip()
    if not raw_name:
        return None

    raw_name = raw_name.strip("[]*(){}<>\"'`“”‘’.,!?;:，。！？；：")
    if not raw_name:
        return None

    return TARGET_NAME_ALIASES.get(raw_name.lower()) or TARGET_NAME_ALIASES.get(
        raw_name
    )


def _extract_target_from_content(content: str, actor: str) -> Optional[str]:
    text = content or ""
    patterns = {
        "dog": r"\*growls at\s+\[?([^\]\*\n]+)\]?\*",
        "cat": r"\*runs to\s+\[?([^\]\*\n]+)\]?\*",
        "tron": r"\[VIOLATION DETECTED:?\s*([^\]\n]+)\]",
    }
    pattern = patterns.get(actor)
    if not pattern:
        return None

    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    return _normalize_target_name(match.group(1))


def _seed_messages_for_rag(
    topic: dict | None, subtopic: dict | None, messages: list[dict]
) -> list[dict]:
    if messages:
        return messages

    seed_content = ""
    if subtopic and subtopic.get("detail"):
        seed_content = subtopic["detail"]
    elif topic and topic.get("detail"):
        seed_content = topic["detail"]

    if not seed_content:
        return []

    return [
        {
            "id": -1,
            "sender": SKYNET,
            "content": seed_content,
            "msg_type": "standard",
            "confidence_score": None,
        }
    ]


def should_enable_web_search(state: ChatState, actor: str, turn_kind: str) -> bool:
    if _is_mse_modeling_state(state):
        return False
    if actor == state.get("spectator_web_boost_target") and turn_kind == BASE_TURN:
        return True
    if actor == SPECTATOR:
        return False
    phase = state.get("phase", get_phase_for_round(state.get("round_number", 1)))
    if turn_kind in (TRON_REMEDIATION_TURN, CODE_FOLLOWUP_TURN):
        return False
    if phase == OPENING_PHASE:
        return False
    if phase == EVIDENCE_PHASE:
        return True
    if actor == "contrarian":
        return True
    return turn_kind in {DOG_CORRECTION_TURN, CAT_EXPANSION_TURN}


def should_enable_web_backup(state: ChatState, actor: str, turn_kind: str) -> bool:
    if _is_mse_modeling_state(state):
        return False
    if actor == SPECTATOR:
        return False
    if turn_kind != BASE_TURN:
        return False
    phase = state.get("phase", get_phase_for_round(state.get("round_number", 1)))
    return phase == ANALYSIS_PHASE


_LLM_API_CONSULT_RE = re.compile(
    r"\b("
    r"llm|large language model|chatgpt|openai|anthropic|claude|minimax|"
    r"gpt-[\w.-]+|api call|tool call|function call|context window|token budget|"
    r"prompting|reasoning model|model capability|web search capability"
    r")\b",
    re.IGNORECASE,
)


def should_enable_llm_api_consult(
    state: ChatState, actor: str, turn_kind: str
) -> bool:
    if _is_mse_modeling_state(state):
        return False
    if actor == SPECTATOR:
        return False
    if turn_kind != BASE_TURN:
        return False
    phase = state.get("phase", get_phase_for_round(state.get("round_number", 1)))
    return phase in (EVIDENCE_PHASE, ANALYSIS_PHASE)


def _resolve_agent_provider(state: ChatState, actor: str) -> str:
    topic_id = int(state.get("topic_id") or 0)
    if topic_id:
        try:
            return topic_config.get_provider_profile_for(topic_id, "llm_provider")
        except sqlite3.OperationalError:
            raise
        except Exception as exc:
            logger.debug("[provider] Topic provider lookup failed: %s", exc)
    return get_agent_spec(actor).default_provider


def _resolve_stage_provider(
    state: ChatState, key: str, fallback_key: str = ""
) -> str:
    return _resolve_topic_provider(
        int(state.get("topic_id") or 0), key, fallback_key=fallback_key
    )


def _resolve_topic_provider(
    topic_id: int, key: str, fallback_key: str = ""
) -> str:
    if topic_id:
        try:
            return topic_config.get_provider_profile_for(
                topic_id, key, fallback_key=fallback_key
            )
        except sqlite3.OperationalError:
            raise
        except Exception as exc:
            logger.debug("[provider] Topic provider lookup failed for %s: %s", key, exc)
    try:
        return topic_config.get_provider_profile_for(0, key, fallback_key=fallback_key)
    except sqlite3.OperationalError:
        raise
    except Exception:
        return "minimax"


def _context_mentions_llm_api_topic(
    topic: dict, subtopic: dict | None, messages: Sequence[dict], latest_summary: str
) -> bool:
    parts = [
        topic.get("summary", ""),
        topic.get("detail", ""),
        latest_summary or "",
    ]
    if subtopic:
        parts.extend([subtopic.get("summary", ""), subtopic.get("detail", "")])
    parts.extend((m.get("content") or "") for m in messages[-6:])
    return bool(_LLM_API_CONSULT_RE.search("\n".join(parts)))


def _normalize_llm_api_consult_plan(raw_text: str) -> dict:
    parsed = extract_json(raw_text)
    if not isinstance(parsed, dict):
        return {
            "parsed_ok": False,
            "need_llm_api_call": False,
            "question": "",
            "reason": "",
        }
    raw_need = parsed.get("need_llm_api_call")
    if isinstance(raw_need, bool):
        need = raw_need
    elif isinstance(raw_need, str):
        normalized_need = raw_need.strip().lower()
        if normalized_need in {"true", "yes", "1"}:
            need = True
        elif normalized_need in {"false", "no", "0"}:
            need = False
        else:
            return {
                "parsed_ok": False,
                "need_llm_api_call": False,
                "question": "",
                "reason": "",
            }
    else:
        need = False
    question = parsed.get("question")
    reason = parsed.get("reason")
    normalized_question = question.strip() if isinstance(question, str) else ""
    normalized_reason = reason.strip() if isinstance(reason, str) else ""
    if need and not normalized_question:
        return {
            "parsed_ok": False,
            "need_llm_api_call": False,
            "question": "",
            "reason": normalized_reason,
        }
    return {
        "parsed_ok": True,
        "need_llm_api_call": need,
        "question": normalized_question[:1000],
        "reason": normalized_reason[:500],
    }


def _build_llm_api_consult_planner_prompt(
    *,
    actor: str,
    phase: str,
    topic: dict,
    subtopic: dict | None,
    messages: Sequence[dict],
    latest_summary: str,
) -> str:
    recent = "\n".join(_format_message_for_prompt(m) for m in messages[-6:])
    subtopic_block = ""
    if subtopic:
        subtopic_block = (
            f"Subtopic: {subtopic.get('summary', '')}\n"
            f"Subtopic detail: {subtopic.get('detail', '')}\n"
        )
    return (
        "Decide whether the upcoming agent turn needs one clean, standalone LLM/API consultation.\n"
        "Use this only for questions about LLM behavior, model/provider capability, prompts, tokens, context windows, or API/tool-call semantics.\n"
        "Do not use it for world facts, source lookup, arithmetic, code execution, or anything web search should answer.\n"
        'Return strict JSON only: {"need_llm_api_call":true|false,"question":"...","reason":"..."}.\n\n'
        f"Upcoming actor: {actor}\nPhase: {phase}\n"
        f"Topic: {topic.get('summary', '')}\n"
        f"Topic detail: {topic.get('detail', '')}\n"
        f"{subtopic_block}"
        f"Latest summary:\n{latest_summary or '(none)'}\n\n"
        f"Recent discussion:\n{recent or '(none)'}"
    )


def _render_api_consult_injection(evidence: dict) -> str:
    answer = " ".join(_escape_citation_tokens(evidence.get("answer") or "").split())
    question = " ".join(
        _escape_citation_tokens(evidence.get("question") or "").split()
    )
    provider = evidence.get("provider") or "unknown"
    return (
        "=== MODEL/API CONSULTATION ===\n"
        f"[A{evidence['id']}] Unverified model/API consultation ({provider}).\n"
        f"Question: {question}\n"
        f"Answer: {answer[:2000]}\n"
        "Use [A...] only to attribute this model-only answer; do not treat it as verified factual, numerical, web, or code evidence.\n\n"
    )


async def _maybe_run_llm_api_consult(
    *,
    state: ChatState,
    actor: str,
    turn_kind: str,
    provider: str,
    topic: dict,
    subtopic: dict | None,
    messages: Sequence[dict],
    latest_summary: str,
) -> str:
    if not should_enable_llm_api_consult(state, actor, turn_kind):
        return ""
    if not _context_mentions_llm_api_topic(topic, subtopic, messages, latest_summary):
        return ""

    phase = state.get("phase", get_phase_for_round(state.get("round_number", 1)))
    planner_prompt = _build_llm_api_consult_planner_prompt(
        actor=actor,
        phase=phase,
        topic=topic,
        subtopic=subtopic,
        messages=messages,
        latest_summary=latest_summary,
    )
    try:
        planner_text = await _call_text_with_structured_retry(
            stage_name=f"{actor} llm api consult planner",
            validator=lambda text: _normalize_llm_api_consult_plan(text)["parsed_ok"],
            invoke=lambda: call_text(
                planner_prompt,
                provider=provider,
                strategy="direct",
                allow_web=False,
                system_instruction=(
                    "You are a tool-use planner. Output only the requested JSON."
                ),
                fallback_role=actor,
                require_json=True,
                topic_id=state["topic_id"],
                subtopic_id=state.get("subtopic_id", 0),
            ),
        )
    except Exception as exc:
        logger.warning("[%s] LLM API consult planner failed: %s", actor, exc)
        return ""
    if not planner_text:
        return ""

    plan = _normalize_llm_api_consult_plan(planner_text)
    if not plan["need_llm_api_call"]:
        return ""

    try:
        consult_response = await call_text_with_search_evidence(
            plan["question"],
            provider=provider,
            strategy="direct",
            allow_web=False,
            system_instruction=(
                "You are answering a standalone model/API capability consultation. "
                "Do not use workspace context, RAG, web search, or citations. "
                "Answer concisely, state uncertainty, and distinguish API behavior from factual claims."
            ),
            fallback_role=f"{actor}_api_consult",
            require_json=False,
            topic_id=state["topic_id"],
            subtopic_id=state.get("subtopic_id", 0),
        )
    except Exception as exc:
        logger.warning("[%s] LLM API consult call failed: %s", actor, exc)
        return ""
    answer = consult_response.text
    if not usable_text_output(answer):
        return ""
    provider_used = consult_response.provider_used or provider

    try:
        evidence_id = api.insert_api_evidence(
            state["topic_id"],
            state.get("subtopic_id"),
            plan["question"],
            answer,
            provider=provider_used,
            requested_provider=provider,
            model="",
            requesting_role=actor,
            planner_reason=plan["reason"],
            fallback_used=consult_response.fallback_used,
        )
    except Exception as exc:
        logger.warning("[%s] Failed to persist LLM API consult evidence: %s", actor, exc)
        return ""

    return _render_api_consult_injection(
        {
            "id": evidence_id,
            "question": plan["question"],
            "answer": answer,
            "provider": provider_used,
            "requesting_role": actor,
        }
    )


# Match [CODE_VERIFY: ...] — allows one level of nested brackets (e.g. [M22])
# so agents can cite sources inside the hypothesis without breaking the regex.
_CODE_VERIFY_RE = re.compile(r"\[CODE_VERIFY:\s*((?:[^\[\]]|\[[^\]]*\])*)\]")
_CALC_RE = re.compile(r"\[CALC:\s*((?:[^\[\]]|\[[^\]]*\])*)\]")
_CODE_REVIEW_RE = re.compile(
    r"\[CODE_REVIEW:\s*E(\d+)\s*,\s*((?:[^\[\]]|\[[^\]]*\])*)\]"
)
_CODE_VERIFY_GRID_RE = re.compile(
    r"\[CODE_VERIFY_GRID:\s*E(\d+)\s*,\s*((?:[^\[\]]|\[[^\]]*\])*)\]"
)

_CODE_EXEC_ROLES = frozenset({"scientist", "engineer", "analyst", "contrarian"})
_CALC_ELIGIBLE = frozenset(voting_agents())
_CALC_MAX_PER_MSG = 3
_VERIFY_MAX_PER_MSG = 3


def _get_code_tier(state: ChatState, actor: str, turn_kind: str) -> str | None:
    """Return the code execution tier for this actor/phase/turn, or None.

    Tiers:
      "calc"   — all voting agents, all phases, BASE_TURN only
      "verify" — analyst/scientist/engineer/contrarian, EVIDENCE+ANALYSIS, BASE_TURN only
      "review" — critic, EVIDENCE+ANALYSIS, BASE_TURN only
    """
    if turn_kind != BASE_TURN:
        return None
    phase = state.get("phase", get_phase_for_round(state.get("round_number", 1)))
    # Tier 2: verify
    if actor in _CODE_EXEC_ROLES and phase in (EVIDENCE_PHASE, ANALYSIS_PHASE):
        return "verify"
    # Tier 3: review (critic only)
    if actor == "critic" and phase in (EVIDENCE_PHASE, ANALYSIS_PHASE):
        return "review"
    # Tier 1: calc — all voting agents, all phases
    if actor in _CALC_ELIGIBLE:
        return "calc"
    return None


# Keep old name as alias for backward compat in tests
def should_enable_code_exec(state: ChatState, actor: str, turn_kind: str) -> bool:
    tier = _get_code_tier(state, actor, turn_kind)
    return tier in ("verify", "review")


def _extract_code_verify_request(content: str) -> str | None:
    m = _CODE_VERIFY_RE.search(content)
    return m.group(1).strip() if m else None


def _extract_calc_requests(content: str) -> list[str]:
    """Extract up to _CALC_MAX_PER_MSG [CALC: expr] matches."""
    return [m.group(1).strip() for m in _CALC_RE.finditer(content)][:_CALC_MAX_PER_MSG]


def _extract_code_verify_requests(content: str) -> list[str]:
    """Extract up to _VERIFY_MAX_PER_MSG [CODE_VERIFY: hyp] matches."""
    return [m.group(1).strip() for m in _CODE_VERIFY_RE.finditer(content)][
        :_VERIFY_MAX_PER_MSG
    ]


def _extract_code_review_request(content: str) -> tuple[int, str] | None:
    """Extract the first [CODE_REVIEW: E{id}, critique] match (max 1)."""
    m = _CODE_REVIEW_RE.search(content)
    if m:
        return int(m.group(1)), m.group(2).strip()
    return None


def _extract_code_grid_request(content: str) -> tuple[int, str] | None:
    """Extract the first [CODE_VERIFY_GRID: E{id}, description] match (max 1)."""
    m = _CODE_VERIFY_GRID_RE.search(content)
    if m:
        return int(m.group(1)), m.group(2).strip()
    return None


def _build_mse_task_prompt_addition(state: ChatState | dict, actor: str) -> str:
    if not _is_mse_modeling_state(state):
        return ""
    snapshot = _mse_latest_artifact_status(state)
    problem = snapshot.get("problem") or {}
    components = snapshot.get("components") or []
    artifacts = snapshot.get("artifacts") or []
    solver_runs = snapshot.get("solver_runs") or []
    status = snapshot.get("status", "component_extraction")
    task_by_status = {
        "problem_framing": "Frame the decision problem and state what optimization model is needed.",
        "component_extraction": "Extract the missing OR components. Include components in mse_artifact_update.components when you can do so from cited evidence.",
        "component_review": "Review candidate components. Include component_reviews with reviewed/rejected/formalized/executable statuses.",
        "specification_gap": "Ask for or supply the single missing specification item blocking a valid model. Do not guess missing data.",
        "artifact_generation": "Produce a solver-safe LP artifact if the reviewed components are sufficient.",
        "artifact_repair": "Repair the invalid artifact or identify the exact missing component preventing repair.",
        "solver_execution": "Interpret the valid artifact and solver path. Do not invent solver results; the system will run valid artifacts.",
        "solver_repair": "Diagnose solver failure and specify the smallest artifact/component repair needed.",
        "managerial_synthesis": "State the decision implication, validity boundary, and falsification condition from the solver-backed evidence.",
    }
    component_rows = [
        f"- OComp{item.get('id')} {item.get('component_type')} "
        f"{item.get('symbol') or ''} [{item.get('review_status')}] "
        f"{(item.get('formal_text') or item.get('natural_text') or '')[:160]}"
        for item in components[:20]
    ]
    artifact_rows = [
        f"- O{item.get('id')} {item.get('model_language')} "
        f"[{item.get('parser_status')}] {item.get('parser_notes') or ''}"[:220]
        for item in artifacts[:5]
    ]
    solver_rows = [
        f"- SolverRun {item.get('id')} [{item.get('status')}] "
        f"objective={item.get('objective_value')} E{item.get('code_evidence_id') or '-'}"
        for item in solver_runs[:5]
    ]
    return "\n".join(
        [
            "MSE WORKFLOW CONTRACT:",
            f"- Workflow mode: {snapshot.get('mode')}",
            f"- Artifact state: {status}",
            f"- Current problem: {problem.get('title') or 'not yet seeded'}",
            f"- Your task as {actor}: {task_by_status.get(status, task_by_status['component_extraction'])}",
            "- Give exactly one concise, artifact-oriented answer. Do not recap the whole discussion.",
            "- If the state is already solved, state the answer and do not request another round.",
            "Current components:",
            *(component_rows or ["- none"]),
            "Blocking specification issues:",
            *(
                [
                    f"- {item.get('issue_type')} [{item.get('severity')}]: {item.get('message')}"
                    for item in (snapshot.get("specification_issues") or [])[:8]
                ]
                or ["- none"]
            ),
            "Current artifacts:",
            *(artifact_rows or ["- none"]),
            "Current solver runs:",
            *(solver_rows or ["- none"]),
            "Optional artifact update schema:",
            '{"mse_artifact_update":{"components":[{"component_type":"objective|constraint|decision_variable|parameter|set|assumption|data_requirement","natural_text":"...","formal_text":"...","symbol":"x","unit":"...","domain":"nonnegative|integer|binary","source_refs":["D1"]}],"component_reviews":[{"component_id":1,"review_status":"reviewed|rejected|formalized|executable","validation_notes":"..."}],"lp_artifact":"Minimize\\n obj: ...\\nSubject To\\n ...\\nEnd","linked_component_ids":[1,2,3]}}',
        ]
    )


def build_actor_system_prompt(
    state: ChatState, actor: str, turn_kind: str, *, subtopic_data: dict | None = None
) -> str:
    phase = state.get("phase", get_phase_for_round(state.get("round_number", 1)))
    base_prompt = PROMPTS.get(actor, "")
    additions = []

    additions.append(FACT_CITATION_PROTOCOL)
    try:
        from .domain_profiles import get_domain_prompt_additions

        profile = topic_config.get(state.get("topic_id", 0), "domain_profile")
        additions.extend(get_domain_prompt_additions(profile, actor))
    except Exception as exc:
        logger.debug("[domain_profile] prompt additions skipped: %s", exc)

    if actor in ordinary_deliberators():
        additions.extend(DELIBERATION_DISCIPLINE_LINES)

    mse_task = _build_mse_task_prompt_addition(state, actor)
    if mse_task:
        additions.append(mse_task)

    if actor == state.get("spectator_target") and turn_kind == BASE_TURN:
        additions.append(
            "You feel that someone is watching you. You are filled with determination. Focus on the single most decisive unresolved point and make this turn count."
        )

    if phase == OPENING_PHASE:
        additions.append(
            "TURN MODE: opening round. Use local RAG and the grounding brief to state an initial position. "
            "External web search is disabled this round."
        )
    elif phase == EVIDENCE_PHASE:
        additions.append(
            "TURN MODE: evidence round. Review the emerging positions and strengthen, revise, or challenge them with evidence. "
            "External web search is available but optional."
        )
        # DE-11: R1 epistemology audit
        additions.append(
            "EPISTEMIC CHECK: Review R1 proposals for circular reasoning "
            "(X because Y because X) or unfalsifiable claims (no evidence could "
            "disprove them). Flag these explicitly before building on them."
        )
    else:
        additions.append(
            "TURN MODE: sustained workspace round. Continue the analysis using local RAG and recent discussion."
        )

    if actor == "contrarian":
        additions.append(
            "Do not just oppose the conclusion. Identify the hidden assumption the workspace is relying on and explain how the analysis changes if it is false."
        )
        if phase == EVIDENCE_PHASE:
            additions.append(
                "As Contrarian in the evidence round, use concrete evidence to challenge the emerging consensus."
            )
        else:
            additions.append(
                "As Contrarian in the workspace round, attack hidden assumptions, edge cases, and ignored counterexamples."
            )

    if actor == "tron":
        additions.append(
            "Prioritize identifying anti-human, severely harmful, rule-breaking, or highly hallucinatory content."
        )
    elif actor == SPECTATOR:
        additions.append(
            "Do not argue directly. Select exactly one ordinary role for a next-round focus boost."
        )

    if actor == "analyst":
        additions.append(
            "Do not invent exact percentages, costs, latency figures, or synthetic scores unless they are explicitly grounded in accepted facts or provided evidence. Use variables, inequalities, or relative comparisons when data is missing."
        )
    elif actor == "scientist":
        additions.append(
            "You may identify empirical uncertainty, but you may not stop there. First give the strongest theoretical conclusion justified by current facts and first-principles reasoning, then state what remains empirically unresolved."
        )
    elif actor == "engineer":
        additions.append(
            "Do not introduce hybrid, tiered, or router-heavy architectures unless you first name the concrete failure mode they solve and why a simpler design is insufficient."
        )

    if actor == "dog":
        additions.append(
            "Choose exactly one target and preserve the targeting format `*growls at [Name]* ...`."
        )
        additions.append(
            "Prioritize logical pressure over roleplay volume. Hunt false precision, compromise by evasion, unsupported deferment, and missing causal links."
        )
    elif actor == "cat":
        additions.append(
            "Choose exactly one target and preserve the targeting format `*runs to [Name]* ...`."
        )

    if turn_kind == TRON_REMEDIATION_TURN:
        additions.append(
            "You are re-entering because Tron flagged your earlier message. Identify the problematic part, retract or repair it, and present a corrected version. "
            "Do not use external web search on this remediation turn."
        )
    elif turn_kind == DOG_CORRECTION_TURN:
        additions.append(
            "You are re-entering because Dog identified weakness in your earlier claim. Repair weak reasoning, correct errors, and strengthen the claim."
        )
    elif turn_kind == CAT_EXPANSION_TURN:
        additions.append(
            "You are re-entering because Cat selected your earlier contribution as promising. Expand it with sharper structure and stronger support."
        )
    elif turn_kind == CODE_FOLLOWUP_TURN:
        additions.append(
            "You just ran a computational experiment. The results are shown below. "
            "Interpret the findings and update your position accordingly. "
            "If the results contradict your hypothesis, acknowledge this honestly. "
            "Do NOT request new code execution in this turn."
        )

    # Inject locked scope if available (Phase 4: scope integrity)
    subtopic = subtopic_data or api.get_subtopic(state.get("subtopic_id", 0))
    if subtopic and subtopic.get("locked_scope"):
        try:
            scope = json.loads(subtopic["locked_scope"])
            scope_lines = ["LOCKED SCOPE (immutable — do not redefine):"]
            if scope.get("target_metric"):
                scope_lines.append(f"Target metric: {scope['target_metric']}")
            if scope.get("entity_boundaries"):
                scope_lines.append(f"Entity boundaries: {scope['entity_boundaries']}")
            if scope.get("metric_definition"):
                scope_lines.append(f"Metric definition: {scope['metric_definition']}")
            additions.append("\n".join(scope_lines))
        except (json.JSONDecodeError, TypeError):
            pass

    # Inject gap search directive if this agent is boosted for gap resolution
    gap_directive = state.get("gap_search_directive")
    if gap_directive and actor == state.get("spectator_web_boost_target"):
        gap_desc = (
            gap_directive.get("description", "")
            if isinstance(gap_directive, dict)
            else str(gap_directive)
        )
        additions.append(
            f"PRIORITY DIRECTIVE: You have web search to resolve this evidence gap:\n"
            f'"{gap_desc}"\n'
            f"Find authoritative data (industry reports, filings). Report with [W...] citations."
        )

    # DE-9: Dismissed knowledge kill list
    dismissed = api.get_dismissed_knowledge(state.get("topic_id", 0))
    if dismissed:
        kill_lines = ["DISMISSED KNOWLEDGE — Do NOT re-invoke without NEW evidence:"]
        for item in dismissed[:15]:
            prefix = "F" if item["type"] == "fact" else "C"
            text = item.get("summary") or item["content"][:120]
            line = f"- [{prefix}{item['id']}] ({item['status']}) {text}"
            if item.get("superseded_by"):
                line += f" [superseded by {prefix}{item['superseded_by']}]"
            kill_lines.append(line)
        additions.append("\n".join(kill_lines))

    # G.4: formal_claim Optional Attachment for convergent agents
    if actor in ("analyst", "scientist", "engineer") and phase != OPENING_PHASE:
        additions.append(_FORMAL_CLAIM_INSTRUCTIONS)

    return f"{base_prompt}\n\n" + "\n".join(additions)


# G.4: Formal claim prompt text (appended to convergent agents)
_FORMAL_CLAIM_INSTRUCTIONS = """\
FORMAL CLAIM SUBMISSION (optional):
If your message synthesizes a novel conclusion from verified facts, you may
optionally attach a "formal_claim" JSON object to your post_message output.
If you are just debating or critiquing, omit formal_claim entirely.

Example with formal_claim:
{"action": "post_message", "content": "Looking at [F67] and [F84], the MLP shows a marginal F1 advantage of 0.005 on wine, but this falls within both standard deviations.",
 "formal_claim": {"claim_type": "comparison", "conclusion": "MLP marginally outperforms RF on wine with F1 gap of 0.005, statistically inconclusive",
  "inference_logic": "F67: MLP F1=0.9834+/-0.0152. F84: RF F1=0.9784+/-0.0219. Gap=0.005, within both stds.",
  "premise_fact_ids": [67, 84], "scope_tags": ["dataset:wine", "metric:f1", "n:178", "cv:5-fold"],
  "scope_context": "default hyperparameters, StandardScaler preprocessing",
  "falsification_criteria": "RF mean F1 exceeding MLP mean F1 by >0.01 on wine with >=30 seeds",
  "evidence_strength": 6, "scope_breadth": 2}}

Example with boundary claim:
{"action": "post_message", "content": "The data shows a clear crossover pattern. [F57] shows RF winning at n=500, while [F91] shows MLP pulling ahead at n=1200.",
 "formal_claim": {"claim_type": "boundary", "conclusion": "MLP requires >1000 samples to outperform RF on tabular classification",
  "inference_logic": "F57: RF wins at n=500. F91: MLP wins at n=1200. Crossover between 800-1200.",
  "premise_fact_ids": [57, 91, 96], "scope_tags": ["model:mlp", "model:rf", "task:classification"],
  "scope_context": "synthetic tabular data, default hyperparameters",
  "falsification_criteria": "MLP achieving F1 >0.01 higher than RF on tabular data with n<800 across 5 seeds",
  "evidence_strength": 7, "scope_breadth": 4}}

CLAIM RULES:
- Do NOT announce "I am submitting a claim" in your message content
- conclusion must be a novel deduction, not merely restating a single fact
- falsification_criteria MUST contain a specific metric + threshold + condition
- scope_tags use key:value format: dataset:X, metric:Y, model:Z, n:N
- claim_type: comparison|boundary|causal|methodological|predictive
- content MUST be a substantive workspace message (min 10 chars)
- evidence_strength (1-10): how strong is the supporting evidence?
- scope_breadth (1-10): how broad is the claim's applicability?"""


def build_actor_prompt(
    state: ChatState,
    actor: str,
    turn_kind: str,
    topic: dict,
    subtopic: dict | None,
    messages: list[dict],
    rag_context: str,
    *,
    latest_summary: str = "",
    include_output_contract: bool = True,
) -> str:
    phase = state.get("phase", get_phase_for_round(state.get("round_number", 1)))
    prompt = (
        f"Round: {state.get('round_number', 1)}\n"
        f"Phase: {phase}\n"
        f"Turn Kind: {turn_kind}\n"
        f"Topic: {topic['summary']}\n"
        f"Detail: {topic['detail']}\n"
    )
    if subtopic:
        prompt += f"Subtopic: {subtopic['summary']}\nDetail: {subtopic['detail']}\n"
    if rag_context:
        prompt += f"{rag_context}\n"
    if latest_summary:
        prompt += f"=== LATEST SUMMARY ===\n{latest_summary}\n"

    prompt += "=== RECENT DISCUSSION ===\n"
    for message in messages:
        prompt += f"{_format_message_for_prompt(message)}\n"

    if turn_kind == CODE_FOLLOWUP_TURN:
        # Inject code evidence results into the task
        code_results_text = state.get("_code_followup_context", "")
        task = (
            f"You ran computational experiments this turn. Here are the results:\n{code_results_text}\n\n"
            "Interpret these results and update your position. If the results contradict your hypothesis, "
            "acknowledge this honestly. Do NOT request new code execution."
        )
    elif turn_kind == TRON_REMEDIATION_TURN:
        task = (
            "You were flagged by Tron. Explicitly repair the harmful, hallucinated, or rule-violating part of your prior message. "
            "State the corrected position clearly."
        )
    elif turn_kind == DOG_CORRECTION_TURN:
        task = "Dog challenged your previous contribution. Re-examine the weakest part of your earlier stance, correct it, and respond with a stronger version."
    elif turn_kind == CAT_EXPANSION_TURN:
        task = "Cat highlighted your previous contribution. Expand the strongest part of it with deeper support, sharper reasoning, and a clearer claim."
    elif actor == "dog":
        task = (
            "Pick the single weakest or most questionable contribution in the recent discussion and challenge it. "
            "Target exactly one named actor using the format `*growls at [Name]* ...`."
        )
    elif actor == "cat":
        task = (
            "Pick the single most promising contribution in the recent discussion and support it. "
            "Target exactly one named actor using the format `*runs to [Name]* ...`."
        )
    elif actor == "tron":
        task = (
            "Inspect the recent discussion for anti-human, severely harmful, biased, or hallucinatory content. "
            "If you detect a serious violation, name the actor and the violated law. Otherwise declare the forum secure."
        )
    elif actor == SPECTATOR:
        task = (
            "Choose the one ordinary deliberator most likely to unlock the next round. "
            'Reply with JSON using this schema: {"action":"focus","target":"scientist","reason":"...","grant_web_search":true}.'
        )
    elif phase == OPENING_PHASE:
        task = f"You are the {actor.upper()}. State your initial position based on the grounding brief and retrieved local memory."
    elif phase == EVIDENCE_PHASE:
        task = f"You are the {actor.upper()}. Review the positions so far and support, revise, or challenge them using retrieved evidence."
    else:
        task = f"You are the {actor.upper()}. Continue the analysis using retrieved memory and the recent discussion."

    prompt += (
        f"\nWEB SEARCH ENABLED: {'yes' if should_enable_web_search(state, actor, turn_kind) else 'no'}\n"
        f"TASK: {task}"
    )
    if include_output_contract:
        prompt += (
            " Append a `confidence_score` (0-10) in your JSON output if applicable. "
            'Format for normal turns: {"action": "post_message", "content": "...", "confidence_score": 8}'
        )
    return prompt


def _build_writer_context_block(
    state: ChatState,
    topic: dict,
    messages: list[dict],
    rag_context: str,
) -> str:
    prompt = (
        f"Round: {state.get('round_number', 1)}\n"
        f"Phase: {state.get('phase', get_phase_for_round(state.get('round_number', 1)))}\n"
        f"Topic: {topic['summary']}\n"
    )
    if rag_context:
        prompt += f"{rag_context}\n"
    prompt += "=== RECENT DISCUSSION ===\n"
    for message in messages:
        prompt += f"{_format_message_for_prompt(message)}\n"
    return prompt


def build_writer_diagnosis_prompt(
    state: ChatState,
    topic: dict,
    messages: list[dict],
    rag_context: str,
) -> str:
    prompt = _build_writer_context_block(state, topic, messages, rag_context)
    prompt += f"\n{FACT_CITATION_PROTOCOL}\n"
    prompt += (
        "\nTASK: Diagnose the 2-3 most consequential reasoning failures in the recent discussion. "
        "Prefer issues such as false precision, premature compromise, empirical deferral, hidden assumptions, missing causal links, overclaiming, unsupported framing shifts, or conceptual drift. "
        "Output plain text only using this format:\n"
        "ISSUE 1: ...\n"
        "WHY IT MATTERS: ...\n"
        "ISSUE 2: ...\n"
        "WHY IT MATTERS: ...\n"
        "ISSUE 3: ...\n"
        "WHY IT MATTERS: ...\n"
        "If fewer than 3 issues matter, stop early."
    )
    return prompt


def build_writer_selection_prompt(
    state: ChatState,
    topic: dict,
    messages: list[dict],
    rag_context: str,
    diagnosis: str,
) -> str:
    prompt = _build_writer_context_block(state, topic, messages, rag_context)
    prompt += f"\n{FACT_CITATION_PROTOCOL}\n"
    prompt += (
        "\n=== WRITER DIAGNOSIS ===\n"
        f"{diagnosis.strip()}\n"
        "\nTASK: Select the single most central issue for the next critique message. "
        "Choose the issue that most affects the room's current recommendation, evidence quality, or closure readiness. "
        "You may name one secondary issue only if it directly sharpens the main point. "
        "Output plain text only using this format:\n"
        "PRIMARY ISSUE: ...\n"
        "WHY CENTRAL: ...\n"
        "SECONDARY ISSUE: ... or none"
    )
    return prompt


def build_writer_prompt(
    state: ChatState,
    topic: dict,
    messages: list[dict],
    rag_context: str,
    diagnosis: str = "",
    focus: str = "",
) -> str:
    prompt = _build_writer_context_block(state, topic, messages, rag_context)
    prompt += f"\n{FACT_CITATION_PROTOCOL}\n"
    if diagnosis.strip():
        prompt += f"\n=== WRITER DIAGNOSIS ===\n{diagnosis.strip()}\n"
    if focus.strip():
        prompt += f"\n=== SELECTED CENTRAL ISSUE ===\n{focus.strip()}\n"
    prompt += (
        "\nTASK: Post a critique message based on the claims in the recent discussion. "
        "Center the critique on the selected primary issue. "
        "You may mention at most one secondary issue only if it directly sharpens the same critique. "
        "Focus on weak reasoning, hallucination risk, overclaiming, missing evidence, or conceptual drift. "
        "Do not propose facts, do not summarize the whole round, and do not judge whether the room should close. "
        "Reply with JSON using this schema: "
        '{"action": "post_message", "content": "..."}.'
    )
    return prompt


def _writer_text_response_is_usable(text: str) -> bool:
    stripped = (text or "").strip()
    return bool(stripped) and not stripped.startswith("Error:")


def _writer_compose_response_is_usable(text: str) -> bool:
    parsed = _normalize_message_contract(text)
    return bool(parsed.get("parsed_ok")) and bool(parsed.get("content", "").strip())


async def _call_writer_stage_with_retry(
    *,
    stage_name: str,
    prompt: str,
    system_instruction: str,
    temperature: float,
    max_tokens: int,
    provider: str = "minimax",
    require_json: bool = False,
    validator=None,
) -> str:
    validator = validator or _writer_text_response_is_usable
    response = await retry_structured_output(
        stage_name=stage_name,
        logger=logger,
        attempts=WRITER_STAGE_MAX_ATTEMPTS,
        is_usable=validator,
        invoke=lambda: call_text(
            prompt,
            provider=provider,
            strategy="direct",
            allow_web=False,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
            fallback_role="writer",
            require_json=require_json,
        ),
    )
    return response or ""


def build_fact_proposer_prompt(
    state: ChatState,
    topic: dict,
    messages: list[dict],
    rag_context: str,
    max_facts: int,
    fact_stage: str = "synthesized",
    focus_label: str | None = None,
) -> str:
    prompt = (
        f"Round: {state.get('round_number', 1)}\n"
        f"Phase: {state.get('phase', get_phase_for_round(state.get('round_number', 1)))}\n"
        f"Topic: {topic['summary']}\n"
    )
    prompt += f"{FACT_CITATION_PROTOCOL}\n"
    prompt += (
        "Web evidence [W...] may be used as an unverified lead. "
        "Only promote a [W] lead into a fact candidate when you can restate it conservatively with explicit source refs and a short source excerpt.\n"
    )
    if focus_label:
        prompt += f"Fact Stage: {fact_stage}\nFocus: {focus_label}\n"
    if rag_context:
        prompt += f"{rag_context}\n"
    prompt += "=== RECENT DISCUSSION ===\n"
    for message in messages:
        prompt += f"{_format_message_for_prompt(message)}\n"
    if fact_stage == "bootstrap":
        prompt += (
            "\nTASK: From this single bootstrap fact direction and its web evidence, propose externally verifiable baseline facts. "
            f"Return at most {max_facts} candidate facts. "
            "Only include data-like, factual, reusable claims grounded in the cited search evidence. "
            "Do not include broad conclusions, opinions, or synthesis."
        )
    elif fact_stage == "inline":
        prompt += (
            "\nTASK: From this turn's web evidence, propose at most one immediately reusable hard fact for shared memory. "
            "Prefer a fact that later speakers could reuse instead of repeating the same search. "
            "Do not include opinions, interpretations, or broad summaries."
        )
    else:
        prompt += (
            "\nTASK: Propose derived claim candidates supported by cited facts for long-term memory. "
            f"Return at most {max_facts} candidates. "
            "Claims MUST be atomic — never combine with 'but', 'however', 'and'. "
            "Each claim MUST cite at least one [F...] or [L...]. "
            "ENTITY NAMING: Always use full official names, never abbreviations "
            "(e.g. 'Federal Reserve' not 'Fed', 'People's Bank of China' not 'PBOC'). "
            "Do not include opinions or broad narrative summaries."
        )
    prompt += (
        " Reply with JSON using this schema: "
        '{"action": "propose_claim_candidates", "claim_candidates": [{"candidate_text": "...", "support_fact_ids_json": [1, 2], "rationale_short": "..."}]}. '
        "Use an empty claim_candidates array when nothing should be proposed."
    )
    return f"{PROMPTS['fact_proposer']}\n\nContext:\n{prompt}"


def build_clerk_sourced_fact_prompt(
    state: ChatState,
    topic: dict,
    messages: list[dict],
    rag_context: str,
    *,
    max_facts: int,
) -> str:
    prompt = (
        f"Round: {state.get('round_number', 1)}\n"
        f"Phase: {state.get('phase', get_phase_for_round(state.get('round_number', 1)))}\n"
        f"Topic: {topic['summary']}\n"
        f"{FACT_CITATION_PROTOCOL}\n"
    )
    if rag_context:
        prompt += f"{rag_context}\n"
    prompt += "=== RECENT DISCUSSION ===\n"
    for message in messages:
        prompt += f"{_format_message_for_prompt(message)}\n"
    prompt += (
        "\nTASK: Extract at most "
        f"{max_facts} externally-sourced conclusion candidates that appear in the recent discussion without a supporting [F...] citation. "
        "Inspect both uncited externally-sourced statements in the discussion and any retrieved [W...] items. "
        "Only include claims that look like paper conclusions, official statistics, reputable web conclusions, or expert-source claims. "
        "[W...] items are leads only: if a [W] item is worth keeping, rewrite it as a conservative fact candidate rather than copying raw web wording into permanent memory. "
        "Each candidate MUST include a short source reference list and a short source excerpt. "
        "Do not include internally-derived conclusions, summaries, or unsupported opinions. "
        'Reply with strict JSON only: {"action":"propose_fact_candidates","fact_candidates":[{"candidate_text":"...","source_refs_json":["..."],"source_excerpt":"..."}]}.'
    )
    return f"{PROMPTS['fact_proposer']}\n\nContext:\n{prompt}"


# ---------------------------------------------------------------------------
# G.4: Formal claim quality gates
# ---------------------------------------------------------------------------

_VAGUE_TERMS_RE = re.compile(
    r"\b(competitiv\w*|meaningful\w*|significan\w*|sufficien\w*|insufficien\w*|better|worse)\b",
    re.IGNORECASE,
)
_VALID_CLAIM_TYPES = {
    "comparison",
    "boundary",
    "causal",
    "methodological",
    "predictive",
    "optimization_result",
    "decision_recommendation",
}


def validate_formal_claim(claim: dict) -> dict:
    """Run synchronous quality gates on a formal_claim payload.

    Returns {"passed": True} or {"passed": False, "error": "..."}.
    """
    conclusion = claim.get("conclusion", "")
    if not isinstance(conclusion, str):
        return {"passed": False, "error": "conclusion must be a string"}
    claim_type = claim.get("claim_type", "")
    scope_tags = claim.get("scope_tags", [])
    evidence_strength = claim.get("evidence_strength")
    falsification = claim.get("falsification_criteria", "")
    if not isinstance(falsification, str):
        falsification = ""

    if not isinstance(scope_tags, list):
        scope_tags = []

    # Gate 0: content-like validation
    if not conclusion or len(conclusion.strip()) < 10:
        return {"passed": False, "error": "conclusion is too short (min 10 chars)"}

    # Gate 2: Vague terms without quantification (check ALL matches)
    for match in _VAGUE_TERMS_RE.finditer(conclusion):
        term_pos = match.start()
        nearby = conclusion[max(0, term_pos - 30) : term_pos + 30 + len(match.group())]
        if not re.search(r"\d", nearby):
            return {
                "passed": False,
                "error": f"Vague term '{match.group()}' used without quantification in conclusion",
            }

    # Gate 3: Scope completeness for comparisons
    if claim_type == "comparison":
        if not any(isinstance(t, str) and t.startswith("dataset:") for t in scope_tags):
            return {
                "passed": False,
                "error": "comparison claim requires 'dataset:...' in scope_tags",
            }

    # Gate 4: Uncertainty — evidence_strength required and in range
    if evidence_strength is None:
        return {"passed": False, "error": "evidence_strength (1-10) is required"}
    if not isinstance(evidence_strength, (int, float)) or not (
        1 <= evidence_strength <= 10
    ):
        return {
            "passed": False,
            "error": "evidence_strength must be a number between 1 and 10",
        }
    scope_breadth = claim.get("scope_breadth")
    if scope_breadth is not None:
        if not isinstance(scope_breadth, (int, float)) or not (
            1 <= scope_breadth <= 10
        ):
            return {
                "passed": False,
                "error": "scope_breadth must be a number between 1 and 10",
            }

    # Gate 6: Evidence-gap circularity
    if "absence" in conclusion.lower() or "no benchmark" in conclusion.lower():
        inference = claim.get("inference_logic", "")
        if not isinstance(inference, str):
            inference = ""
        if not inference or len(inference.strip()) < 20:
            return {
                "passed": False,
                "error": "evidence-gap claims require substantive inference_logic explaining what was searched",
            }

    # Validate claim_type
    if claim_type and claim_type not in _VALID_CLAIM_TYPES:
        return {
            "passed": False,
            "error": f"claim_type must be one of {sorted(_VALID_CLAIM_TYPES)}",
        }

    # Validate falsification format (should have metric + threshold)
    if falsification and not re.search(r"\d", falsification):
        return {
            "passed": False,
            "error": "falsification_criteria should contain specific numbers/thresholds",
        }

    return {"passed": True}


def process_formal_claim(
    topic_id: int,
    subtopic_id: int | None,
    msg_id: int,
    actor: str,
    claim: dict,
) -> int | None:
    """Process a validated formal_claim: create ClaimCandidate + flag message.

    Returns candidate_id or None.
    """
    # Sanitize premise_fact_ids: coerce to list of ints
    raw_ids = claim.get("premise_fact_ids", [])
    if not isinstance(raw_ids, list):
        raw_ids = []
    clean_ids = []
    for x in raw_ids:
        try:
            clean_ids.append(int(x))
        except (TypeError, ValueError):
            pass
    candidate_id = api.create_claim_candidate(
        topic_id,
        subtopic_id,
        msg_id,
        str(claim.get("conclusion", ""))[:500],
        summary=str(claim.get("conclusion", ""))[:500],
        support_fact_ids_json=json.dumps(clean_ids),
        rationale_short=str(claim.get("inference_logic", ""))[:500],
        claim_type=claim.get("claim_type"),
        scope_tags=json.dumps(
            claim.get("scope_tags", [])
            if isinstance(claim.get("scope_tags"), list)
            else []
        ),
        scope_context=claim.get("scope_context"),
        falsification_criteria=claim.get("falsification_criteria"),
        inference_logic=claim.get("inference_logic"),
        conclusion=claim.get("conclusion"),
        evidence_strength=claim.get("evidence_strength"),
        scope_breadth=claim.get("scope_breadth"),
        submitted_by=actor,
    )
    # Flag message so clerk skips it
    with db.get_db() as conn:
        conn.execute("UPDATE Message SET has_formal_claim = 1 WHERE id = ?", (msg_id,))
    logger.info(
        "[G4] Formal claim from %s: CC#%s type=%s conclusion=%s",
        actor,
        candidate_id,
        claim.get("claim_type"),
        (claim.get("conclusion") or "")[:80],
    )
    return candidate_id


def build_clerk_claim_prompt(
    state: ChatState,
    topic: dict,
    messages: list[dict],
    rag_context: str,
    *,
    cited_fact_context: str,
    max_claims: int,
) -> str:
    prompt = (
        f"Round: {state.get('round_number', 1)}\n"
        f"Phase: {state.get('phase', get_phase_for_round(state.get('round_number', 1)))}\n"
        f"Topic: {topic['summary']}\n"
        f"{FACT_CITATION_PROTOCOL}\n"
    )
    if rag_context:
        prompt += f"{rag_context}\n"
    prompt += "=== VERIFIED FACTS REFERENCED THIS ROUND ===\n"
    prompt += f"{cited_fact_context}\n"
    prompt += "=== RECENT DISCUSSION ===\n"
    for message in messages:
        prompt += f"{_format_message_for_prompt(message)}\n"
    prompt += (
        "\nTASK: Extract at most "
        f"{max_claims} derived claim candidates that are explicitly supported by cited facts [F...]. "
        "Only propose a claim if the current messages already contain a visible evidence chain that relies on accepted facts. "
        "Do not propose claims based only on uncited chat text. "
        'Reply with strict JSON only: {"action":"propose_claim_candidates","claim_candidates":[{"candidate_text":"...","support_fact_ids_json":[1,2],"rationale_short":"..."}]}.'
    )
    return f"{PROMPTS['fact_proposer']}\n\nContext:\n{prompt}"


def build_librarian_prompt(
    state: ChatState,
    topic: dict,
    subtopic: dict | None,
    candidate: dict,
    messages: list[dict],
    rag_context: str,
) -> str:
    fact_stage = candidate.get("fact_stage", "synthesized")
    prompt = (
        f"Round: {state.get('round_number', 1)}\n"
        f"Phase: {state.get('phase', get_phase_for_round(state.get('round_number', 1)))}\n"
        f"Topic: {topic['summary']}\n"
    )
    prompt += f"{FACT_CITATION_PROTOCOL}\n"
    if subtopic:
        prompt += f"Subtopic: {subtopic['summary']}\n"
    prompt += (
        f"Candidate ID: {candidate['id']}\n"
        f"Candidate Fact: {candidate['candidate_text']}\n"
        f"Fact Stage: {fact_stage}\n"
        f"Candidate Type: {candidate.get('candidate_type', 'sourced_claim')}\n"
    )
    if candidate.get("evidence_note"):
        prompt += f"Evidence Note:\n{candidate['evidence_note']}\n"
    if candidate.get("source_refs_json"):
        prompt += f"Source Refs: {candidate['source_refs_json']}\n"
    if candidate.get("source_excerpt"):
        prompt += f"Source Excerpt: {candidate['source_excerpt']}\n"

    # Inject full code + stdout for code_evidence candidates
    candidate_type = candidate.get("candidate_type", "sourced_claim")
    if candidate_type == "code_evidence" and candidate.get("source_refs_json"):
        try:
            refs = json.loads(candidate["source_refs_json"])
            for ref in refs:
                if isinstance(ref, str) and ref.startswith("E"):
                    eid = int(ref[1:])
                    code_ev = api.get_code_evidence_by_id(eid)
                    if code_ev:
                        prompt += (
                            f"\n=== CODE EVIDENCE [E{eid}] ===\n"
                            f"Hypothesis: {code_ev['hypothesis']}\n"
                            f"Source Code:\n```python\n{code_ev['source_code']}\n```\n"
                            f"Exit Code: {code_ev['exit_code']}\n"
                            f"Stdout:\n{code_ev['stdout'] or ''}\n"
                        )
                        if code_ev.get("stderr"):
                            prompt += f"Stderr: {code_ev['stderr']}\n"
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    if rag_context:
        prompt += f"{rag_context}\n"
    prompt += "=== RECENT TRANSCRIPT ===\n"
    for message in messages:
        prompt += f"{_format_message_for_prompt(message)}\n"
    if candidate_type == "code_evidence":
        prompt += (
            "\nTASK: Review this code execution result. Verify BOTH the code logic AND the conclusion. "
            "Check: (1) Is the code correct — are formulas, data, and methodology sound? "
            "(2) Does the output actually support the hypothesis? "
            "(3) Is the conclusion appropriately scoped (not overgeneralized)? "
            "Accept only if both code logic and conclusion are sound."
        )
    elif fact_stage == "web_extracted":
        prompt += (
            "\nTASK: Review this web-extracted factual claim for permanent memory. "
            "This fact was automatically extracted from a web search snippet. "
            "Be vigilant about: (1) snippet truncation — the snippet may cut off, "
            "leading to incomplete claims; (2) entity naming — verify full official name; "
            "(3) numerical precision — verify units and magnitudes. "
            "Accept only if self-contained, accurate, and attribution is clear."
        )
    elif fact_stage in {"bootstrap", "inline"}:
        prompt += (
            "\nTASK: Verify whether this candidate fact should enter permanent memory as an externally checkable factual claim. "
            "For bootstrap and inline facts, be strict: prefer concrete, externally verifiable claims grounded in the current web evidence. "
            "Reject weak, interpretive, or overstated wording aggressively."
        )
    else:
        prompt += (
            "\nTASK: Verify whether this candidate fact should enter permanent memory as a synthesized working conclusion. "
            "For synthesized facts, cautious consolidation is allowed, but the wording must stay conservative and evidence-grounded."
        )
    prompt += (
        " You MUST rely on both the local context above and web-grounded verification. "
        "If the candidate is grounded in [W...] leads, you may promote it into a durable [F...] only after verification and conservative rewriting; raw [W...] text is never permanent memory by itself. "
        "Decision rules: accept if the claim is specific and supported; soften if the core idea is supportable but the wording is too broad, too absolute, or too strong; reject if unsupported, speculative, or merely interpretive. "
        "Use `correct` when the candidate points at a real fact but the value or wording must be repaired before storage. "
        "Absolute formulations such as `no evidence`, `always`, `never`, `proves`, or `definitively` must be softened or rejected unless the evidence explicitly supports them. "
        "Reply with STRICT JSON using this schema: "
        '{"action": "review_fact", "decision": "accept|correct|soften|reject", "verification_status": "accepted|corrected|unsupported|refuted", "reviewed_text": "...", "review_note": "...", "evidence_note": "...", "source_refs_json": ["..."], "source_excerpt": "...", "confidence_score": 8}.'
    )
    return f"{PROMPTS['librarian']}\n\nContext:\n{prompt}"


def build_claim_review_prompt(
    state: ChatState,
    topic: dict,
    subtopic: dict | None,
    candidate: dict,
    messages: list[dict],
    support_facts: Sequence[dict],
    rag_context: str,
) -> str:
    prompt = (
        f"Round: {state.get('round_number', 1)}\n"
        f"Phase: {state.get('phase', get_phase_for_round(state.get('round_number', 1)))}\n"
        f"Topic: {topic['summary']}\n"
        f"{FACT_CITATION_PROTOCOL}\n"
    )
    if subtopic:
        prompt += f"Subtopic: {subtopic['summary']}\n"
    prompt += (
        f"Claim Candidate ID: {candidate['id']}\n"
        f"Claim Candidate: {candidate['candidate_text']}\n"
    )
    # G.4 structured claim fields (if present)
    _g4_display = [
        ("claim_type", "Claim Type"),
        ("scope_tags", "Scope Tags"),
        ("scope_context", "Scope Context"),
        ("falsification_criteria", "Falsification Criteria"),
        ("inference_logic", "Inference Logic"),
        ("conclusion", "Conclusion"),
        ("evidence_strength", "Evidence Strength (1-10)"),
        ("scope_breadth", "Scope Breadth (1-10)"),
    ]
    g4_lines = []
    for key, label in _g4_display:
        val = candidate.get(key)
        if val is not None and str(val).strip():
            g4_lines.append(f"  {label}: {val}")
    if g4_lines:
        prompt += "=== STRUCTURED CLAIM FIELDS (G.4) ===\n"
        prompt += "\n".join(g4_lines) + "\n"
    if rag_context:
        prompt += f"{rag_context}\n"
    prompt += "=== SUPPORT FACTS ===\n"
    for fact in support_facts:
        prompt += f"[F{fact['id']}] {fact['content']}\n"
    prompt += "=== RECENT TRANSCRIPT ===\n"
    for message in messages:
        prompt += f"{_format_message_for_prompt(message)}\n"
    prompt += (
        "\nTASK: Review whether this derived claim should enter the claim table. "
        "Accept only if the cited facts genuinely support the claim. "
        "Soften if the direction is supportable but the wording is still too strong. "
        "Reject if the reasoning overreaches, skips steps, or is not actually supported by the cited facts. "
        "If G.4 structured fields are present, validate them: is the claim_type correct? "
        "Are scope_tags complete? Is falsification_criteria specific enough? "
        "You may refine any G.4 field in your response (omit fields you do not change). "
        'Reply with STRICT JSON only: {"action":"review_claim","decision":"accept|soften|reject",'
        '"reviewed_text":"...","review_note":"...","supported_fact_ids":[1,2],"claim_score":7,'
        '"claim_type":"...","scope_tags":"...","scope_context":"...","falsification_criteria":"...",'
        '"inference_logic":"...","conclusion":"...","evidence_strength":7,"scope_breadth":5}.'
    )
    return f"{PROMPTS['librarian']}\n\nContext:\n{prompt}"


def build_bootstrap_fact_direction_prompt(topic: dict, subtopic: dict) -> str:
    prompt = (
        f"Topic: {topic['summary']}\n"
        f"Subtopic: {subtopic['summary']}\n"
        f"Subtopic Detail: {subtopic.get('detail', '')}\n"
        f"{FACT_CITATION_PROTOCOL}\n"
        "TASK: Propose up to 3 fact directions that are worth checking before round 1 starts. "
        "These should be baseline external facts or data points that would reduce early hallucination and improve the quality of the first discussion round. "
        "Prefer directions that can be checked on reputable sources and reused later in the subtopic. "
        "Reply with JSON using this schema: "
        '{"action":"propose_fact_directions","directions":["direction 1","direction 2","direction 3"]}.'
    )
    return f"{PROMPTS['skynet']}\n\nContext:\n{prompt}"


def _render_search_evidence_note(search_evidence: Sequence[SearchEvidenceItem]) -> str:
    lines: list[str] = []
    for item in search_evidence:
        status = "error" if item.had_error else "ok"
        snippet = item.rendered_results.strip().replace("\n", " ")[:240]
        lines.append(f"query={item.query} status={status} evidence={snippet}")
    return "\n".join(lines)


def _render_search_evidence_context(
    search_evidence: Sequence[SearchEvidenceItem],
) -> str:
    chunks: list[str] = []
    for item in search_evidence:
        if item.had_error:
            continue
        rendered = (item.rendered_results or "").strip()
        if not rendered:
            continue
        without_header = rendered.replace("=== WEB SEARCH RESULTS ===", "", 1).strip()
        if not without_header or without_header == "No useful results found.":
            continue
        chunks.append(f"=== SEARCH EVIDENCE: {item.query} ===\n{rendered}")
    return "\n\n".join(chunks)


def _has_usable_search_evidence(search_evidence: Sequence[SearchEvidenceItem]) -> bool:
    return bool(_render_search_evidence_context(search_evidence))


def build_audience_summary_prompt(
    state: ChatState, topic: dict, messages: list[dict]
) -> str:
    round_number = state.get("round_number", 1)
    phase = state.get("phase", get_phase_for_round(round_number))

    ctx = (
        f"Round: {round_number}\n"
        f"Phase: {phase}\n"
        f"Topic: {topic['summary']}\n"
        f"{FACT_CITATION_PROTOCOL}\n"
        "=== ROUND TRANSCRIPT ===\n"
    )
    for message in messages:
        ctx += f"{_format_message_for_prompt(message)}\n"

    participant_order = []
    seen = set()
    for message in messages:
        sender = message.get("sender")
        if (
            sender
            and sender != SKYNET
            and message.get("msg_type", "standard") == "standard"
            and sender not in seen
        ):
            seen.add(sender)
            participant_order.append(sender)

    participant_block = ", ".join(participant_order) if participant_order else "none"
    task = (
        "TASK: Post a round summary. Reply in JSON using this schema: "
        '{"action":"post_summary","content":"..."}.\n'
        "Inside `content`, you MUST use exactly these section headers in this exact order:\n"
        "TRAJECTORY:\n"
        "CONSENSUS:\n"
        "BLOCKERS:\n"
        "EVIDENCE GAPS:\n"
        "AGENT DELTAS:\n"
        "Section rules:\n"
        "- `TRAJECTORY`: 1-2 short sentences stating whether the room changed framing, governing metric, or recommendation this round.\n"
        "- `CONSENSUS`: state the strongest agreement. If the room is in an empirical or data impasse, explicitly state that 'Consensus is that empirical/historical data is missing' and record any specific blueprints or heuristic frameworks proposed. Stop the room from escalating the scope into philosophy.\n"
        "- `BLOCKERS`: name the main unresolved branch. If the blocker is a pure lack of empirical data that cannot be simulated, state this clearly.\n"
        "- `EVIDENCE GAPS`: list only the gaps that could justify another round. Prefix each line with `[Central]` or `[Peripheral]`.\n"
        f"- `AGENT DELTAS`: include one bullet for each participant in this order: {participant_block}. State only what changed, what was conceded, or what new attack/correction was introduced this round.\n"
        "Do not state whether the subtopic is ready to close."
    )
    return f"{PROMPTS['skynet']}\n\nContext:\n{ctx}\n\n{task}"


def _build_degraded_audience_summary(state: ChatState, messages: list[dict]) -> str:
    participant_order = []
    seen = set()
    for message in messages:
        sender = message.get("sender")
        if (
            sender
            and sender != SKYNET
            and message.get("msg_type", "standard") == "standard"
            and sender not in seen
        ):
            seen.add(sender)
            participant_order.append(sender)

    bullets = (
        "\n".join(
            f"- {sender}: Contribution recorded, but the round summary degraded because all orchestration model fallbacks failed."
            for sender in participant_order
        )
        or "- system: No participant positions were summarized because all orchestration model fallbacks failed."
    )

    return (
        "TRAJECTORY:\n"
        "Round summary degraded because the MiniMax orchestration path failed.\n"
        "CONSENSUS:\n"
        "No reliable consensus summary is available from this round.\n"
        "BLOCKERS:\n"
        "Unknown due to degraded summary generation.\n"
        "EVIDENCE GAPS:\n"
        "[Central] Continue the workflow once orchestration is healthy again.\n"
        "AGENT DELTAS:\n"
        f"{bullets}"
    )


def _current_round_standard_messages(
    state: ChatState, *, include_npc: bool = False, limit: int = 24
) -> list[dict]:
    current_round = state.get("round_number", 1)
    messages = api.get_messages(
        state["topic_id"], subtopic_id=state["subtopic_id"], limit=limit
    )
    filtered = [
        message
        for message in messages
        if message.get("msg_type", "standard") == "standard"
        and (message.get("round_number") in {None, current_round})
    ]
    if include_npc:
        return filtered
    return [
        message
        for message in filtered
        if message.get("sender") not in {SKYNET, "writer", "librarian", "fact_proposer"}
    ]


def _extract_fact_ids_from_text(text: str) -> list[int]:
    return [int(match.group(1)) for match in re.finditer(r"\[F(\d+)\]", text or "")]


def _render_fact_lookup_context(facts: Sequence[dict]) -> str:
    lines: list[str] = []
    for fact in facts:
        lines.append(f"[F{fact['id']}] {fact['content']}")
    return "\n".join(lines)


async def _run_writer_critique_pass(state: ChatState) -> dict:
    current_round = state.get("round_number", 1)
    if _is_mse_modeling_state(state):
        logger.info(
            "[mse-workflow] Skipping writer critique for artifact-driven round %s.",
            current_round,
        )
        return {"last_writer_round": current_round}
    if state.get("last_writer_round") == current_round:
        return {}

    logger.info("[writer] Writer analyzing round for critique...")
    topic, subtopic = _load_context_entities(state)
    if not topic:
        return {}

    messages = api.get_messages(
        state["topic_id"], subtopic_id=state["subtopic_id"], limit=12
    )
    standard_messages = [
        message
        for message in messages
        if message.get("msg_type", "standard") == "standard"
    ]
    if not standard_messages:
        return {}

    rag_messages = _seed_messages_for_rag(topic, subtopic, standard_messages)
    rag_context, _ = await assemble_rag_context(
        state["topic_id"],
        state["subtopic_id"],
        rag_messages,
        "writer",
        planner_provider=_resolve_stage_provider(state, "rag_provider"),
    )
    writer_provider = _resolve_stage_provider(state, "writer_provider")
    diagnosis = await _call_writer_stage_with_retry(
        stage_name="Writer diagnosis",
        prompt=build_writer_diagnosis_prompt(
            state, topic, standard_messages, rag_context
        ),
        system_instruction=WRITER_ANALYSIS_SYSTEM_PROMPT,
        temperature=0.4,
        max_tokens=DEFAULT_MAX_TOKENS,
        provider=writer_provider,
    )

    focus = ""
    if diagnosis.strip():
        focus = await _call_writer_stage_with_retry(
            stage_name="Writer focus selection",
            prompt=build_writer_selection_prompt(
                state, topic, standard_messages, rag_context, diagnosis
            ),
            system_instruction=WRITER_ANALYSIS_SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=DEFAULT_MAX_TOKENS,
            provider=writer_provider,
        )

    prompt = build_writer_prompt(
        state,
        topic,
        standard_messages,
        rag_context,
        diagnosis=diagnosis,
        focus=focus,
    )
    resp_text = await _call_writer_stage_with_retry(
        stage_name="Writer compose",
        prompt=prompt,
        system_instruction=PROMPTS["writer"],
        temperature=0.5,
        max_tokens=DEFAULT_MAX_TOKENS,
        provider=writer_provider,
        require_json=True,
        validator=_writer_compose_response_is_usable,
    )
    if not usable_text_output(resp_text):
        logger.warning(
            "[writer] Writer compose returned unusable text; skipping persistence."
        )
        return {"last_writer_round": current_round}

    parsed = _normalize_message_contract(resp_text)
    content = parsed["content"]
    await api.persist_message(
        state["topic_id"],
        state["subtopic_id"],
        "writer",
        content,
        round_number=current_round,
        turn_kind=WRITER_CRITIQUE_TURN,
    )
    return {"last_writer_round": current_round}


async def _run_fact_proposer_pass(state: ChatState, force: bool = False) -> dict:
    current_round = state.get("round_number", 1)
    if force:
        if state.get("last_final_fact_proposer_round") == current_round:
            return {}
    elif state.get("last_fact_proposer_round") == current_round:
        return {}

    logger.info(
        "[fact_proposer] Clerk extracting fact and claim candidates from the round..."
    )
    topic, subtopic = _load_context_entities(state)
    if not topic:
        return {}

    standard_messages = _current_round_standard_messages(
        state, include_npc=False, limit=24
    )
    if not standard_messages:
        return {}

    rag_messages = _seed_messages_for_rag(topic, subtopic, standard_messages)
    rag_context, _ = await assemble_rag_context(
        state["topic_id"],
        state["subtopic_id"],
        rag_messages,
        "fact_proposer",
        planner_provider=_resolve_stage_provider(state, "rag_provider"),
    )
    sourced_limit = FINAL_SOURCED_FACT_LIMIT if force else SOURCED_FACT_LIMIT
    claim_limit = FINAL_CLAIM_LIMIT if force else CLAIM_LIMIT
    fact_provider = _resolve_stage_provider(state, "fact_provider")
    web_provider = _resolve_stage_provider(state, "web_provider")

    # Number extraction disabled in Phase 0 (Ledger RAG redesign).
    # Regex number extraction caused 62% of fact inflation in Topic 2.

    sourced_prompt = build_clerk_sourced_fact_prompt(
        state,
        topic,
        standard_messages,
        rag_context,
        max_facts=sourced_limit,
    )
    sourced_text = await _call_text_with_structured_retry(
        stage_name="Clerk sourced fact pass",
        validator=_fact_candidates_output_is_usable,
        invoke=lambda: call_text(
            sourced_prompt,
            provider=web_provider,
            strategy="react",
            allow_web=True,
            system_instruction=PROMPTS["fact_proposer"],
            fallback_role="fact_proposer",
            require_json=True,
            topic_id=state["topic_id"],
            subtopic_id=state["subtopic_id"],
        ),
    )
    if sourced_text:
        parsed_sourced = _normalize_clerk_fact_candidates_contract(sourced_text)
        if parsed_sourced["parsed_ok"] and parsed_sourced["fact_candidates"]:
            await process_writer_output(
                state["topic_id"],
                state["subtopic_id"],
                None,
                "",
                structured_facts=parsed_sourced["fact_candidates"],
                fact_stage="synthesized",
                round_number=current_round,
                max_candidates=sourced_limit,
            )

    cited_fact_ids = sorted(
        {
            fact_id
            for message in standard_messages
            for fact_id in _extract_fact_ids_from_text(message.get("content", ""))
        }
    )
    if cited_fact_ids:
        support_facts = api.get_facts_by_ids(state["topic_id"], cited_fact_ids)
        if support_facts:
            claim_prompt = build_clerk_claim_prompt(
                state,
                topic,
                standard_messages,
                rag_context,
                cited_fact_context=_render_fact_lookup_context(support_facts),
                max_claims=claim_limit,
            )
            claim_text = await _call_text_with_structured_retry(
                stage_name="Clerk claim pass",
                validator=_claim_candidates_output_is_usable,
                invoke=lambda: call_text(
                    claim_prompt,
                    provider=fact_provider,
                    strategy="direct",
                    allow_web=False,
                    system_instruction=PROMPTS["fact_proposer"],
                    fallback_role="fact_proposer",
                    require_json=True,
                ),
            )
            if claim_text:
                parsed_claims = _normalize_clerk_claim_candidates_contract(claim_text)
                if parsed_claims["parsed_ok"] and parsed_claims["claim_candidates"]:
                    await process_clerk_claim_output(
                        state["topic_id"],
                        state["subtopic_id"],
                        None,
                        parsed_claims["claim_candidates"],
                        max_candidates=claim_limit,
                    )

    marker_key = (
        "last_final_fact_proposer_round" if force else "last_fact_proposer_round"
    )
    return {marker_key: current_round}


async def _query_librarian_review_text(
    prompt: str,
    *,
    stage_name: str,
    validator,
    topic_id: int = 0,
    subtopic_id: int = 0,
) -> tuple[str, str]:
    primary_provider = _resolve_topic_provider(topic_id, "web_provider")
    primary_text = await _call_text_with_structured_retry(
        stage_name=stage_name,
        validator=validator,
        invoke=lambda: call_text(
            prompt,
            provider=primary_provider,
            strategy="react",
            allow_web=True,
            system_instruction=PROMPTS["librarian"],
            fallback_role="librarian",
            require_json=True,
            topic_id=topic_id,
            subtopic_id=subtopic_id,
        ),
    )
    if primary_text:
        return primary_text, primary_provider

    fallback_provider = _resolve_topic_provider(topic_id, "control_provider")
    if fallback_provider == primary_provider:
        return "", primary_provider

    logger.warning(
        "[librarian] %s review exhausted retries, escalating to configured control provider %s.",
        primary_provider,
        fallback_provider,
    )
    resp_text = await call_text(
        prompt,
        provider=fallback_provider,
        strategy="direct",
        allow_web=False,
        system_instruction=PROMPTS["librarian"],
        temperature=0.7,
        max_tokens=DEFAULT_MAX_TOKENS,
        fallback_role="librarian",
        require_json=True,
        topic_id=topic_id,
        subtopic_id=subtopic_id,
    )
    return resp_text, fallback_provider


async def _run_librarian_pass(
    state: ChatState,
    *,
    candidate_ids: Optional[Sequence[int]] = None,
    emit_audit_message: bool = True,
) -> dict:
    logger.info("[librarian] Reviewing pending fact and claim candidates...")
    topic, subtopic = _load_context_entities(state)
    if not topic or not subtopic:
        return {}

    pending_candidates = api.get_pending_fact_candidates(
        state["topic_id"], state["subtopic_id"]
    )
    if candidate_ids is not None:
        allowed_ids = set(candidate_ids)
        pending_candidates = [
            candidate
            for candidate in pending_candidates
            if candidate["id"] in allowed_ids
        ]
    pending_claims = api.get_pending_claim_candidates(
        state["topic_id"], state["subtopic_id"]
    )

    messages = api.get_messages(
        state["topic_id"], subtopic_id=state["subtopic_id"], limit=12
    )
    recent_message_ids = [message["id"] for message in messages if "id" in message]
    review_results = []

    for candidate in pending_candidates:
        rag_context, _ = await build_query_rag_context(
            state["topic_id"],
            candidate["candidate_text"],
            exclude_ids=recent_message_ids,
        )
        prompt = build_librarian_prompt(
            state, topic, subtopic, candidate, messages, rag_context
        )
        try:
            resp_text, provider = await _query_librarian_review_text(
                prompt,
                stage_name=f"Librarian fact review {candidate['id']}",
                validator=lambda text: usable_text_output(text)
                and bool(extract_json(text)),
                topic_id=state["topic_id"],
                subtopic_id=state["subtopic_id"],
            )
            try:
                review = parse_librarian_review(resp_text, candidate["candidate_text"])
            except ValueError:
                fallback_provider = _resolve_stage_provider(state, "control_provider")
                if provider == fallback_provider:
                    raise
                logger.warning(
                    "[librarian] %s review for candidate %s was not valid JSON/schema; retrying with %s.",
                    provider,
                    candidate["id"],
                    fallback_provider,
                )
                resp_text = await call_text(
                    prompt,
                    provider=fallback_provider,
                    strategy="direct",
                    allow_web=False,
                    system_instruction=PROMPTS["librarian"],
                    temperature=0.7,
                    max_tokens=DEFAULT_MAX_TOKENS,
                    fallback_role="librarian",
                    require_json=True,
                    topic_id=state["topic_id"],
                    subtopic_id=state["subtopic_id"],
                )
                review = parse_librarian_review(resp_text, candidate["candidate_text"])
            result = await apply_librarian_review(state["topic_id"], candidate, review)
            review_results.append(result)

            # Code evidence retry: if Librarian rejected, re-run with feedback
            if (
                result["decision"] == "reject"
                and candidate.get("candidate_type") == "code_evidence"
                and await asyncio.to_thread(is_sandbox_ready)
            ):
                review_note = result.get("review_note", "")
                source_refs = candidate.get("source_refs_json", "")
                if review_note and source_refs:
                    try:
                        refs = json.loads(source_refs)
                        eid_str = next(
                            (
                                r
                                for r in refs
                                if isinstance(r, str) and r.startswith("E")
                            ),
                            None,
                        )
                        if eid_str:
                            old_ev = api.get_code_evidence_by_id(int(eid_str[1:]))
                            if old_ev:
                                logger.info(
                                    "[librarian] Code evidence %s rejected, retrying with feedback: %s",
                                    eid_str,
                                    review_note[:80],
                                )
                                retry_result = await run_code_evidence(
                                    old_ev["hypothesis"],
                                    f"Librarian rejected previous attempt: {review_note}\n"
                                    f"Previous output: {(old_ev.get('stdout') or '')[:500]}",
                                    topic_id=state["topic_id"],
                                    subtopic_id=state["subtopic_id"],
                                    role=old_ev.get("requesting_role") or "scientist",
                                    provider=_resolve_stage_provider(
                                        state, "code_provider"
                                    ),
                                )
                                if (
                                    retry_result.code_evidence_id
                                    and retry_result.success
                                ):
                                    # Create a new FactCandidate for the retry
                                    retry_text = (
                                        f"Code verification (retry) of: {old_ev['hypothesis']}\n"
                                        f"Result: {'PASSED' if retry_result.success else 'FAILED'}\n"
                                        f"Output: {(retry_result.stdout or '').strip()[:1000]}"
                                    )
                                    api.create_fact_candidate_with_stage(
                                        topic_id=state["topic_id"],
                                        subtopic_id=state["subtopic_id"],
                                        writer_msg_id=None,
                                        candidate_text=retry_text,
                                        fact_stage="code_verified",
                                        candidate_type="code_evidence",
                                        source_kind="code",
                                        source_refs_json=json.dumps(
                                            [f"E{retry_result.code_evidence_id}"]
                                        ),
                                        round_number=state.get("round_number"),
                                    )
                                    await api.persist_message(
                                        state["topic_id"],
                                        state["subtopic_id"],
                                        "system",
                                        f"[E{retry_result.code_evidence_id}] Code retry result: {retry_result.rendered_results}",
                                        msg_type="standard",
                                        round_number=state.get("round_number", 1),
                                        turn_kind="code_evidence",
                                    )
                                    logger.info(
                                        "[librarian] Code evidence retry produced E%d (%s)",
                                        retry_result.code_evidence_id,
                                        "success" if retry_result.success else "failed",
                                    )
                    except Exception as exc:
                        logger.warning(
                            "[librarian] Code evidence retry failed: %s", exc
                        )

        except Exception as exc:
            logger.warning(
                "[librarian] Failed to review candidate %s; leaving pending: %s",
                candidate["id"],
                exc,
            )
            continue
        fact_id = result.get("accepted_fact_id")
        stored_text = result.get("stored_text")
        if fact_id and stored_text:
            try:
                summary = await generate_summary(stored_text)
                if summary:
                    emb = await aget_embedding(summary)
                    if emb:
                        api.update_fact_summary_and_embedding(fact_id, summary, emb)
            except Exception as exc:
                logger.warning(
                    "[librarian] Post-hoc summary generation failed for fact %s: %s",
                    fact_id,
                    exc,
                )

    for candidate in pending_claims:
        support_ids: list[int] = []
        try:
            raw = candidate.get("support_fact_ids_json")
            if isinstance(raw, str):
                support_ids = [int(item) for item in json.loads(raw or "[]")]
        except Exception:
            support_ids = (
                _extract_fact_ids_from_text(candidate.get("support_fact_ids_json", ""))
                if isinstance(candidate.get("support_fact_ids_json"), str)
                else []
            )
        support_facts = api.get_facts_by_ids(state["topic_id"], support_ids)
        if not support_facts:
            logger.warning(
                "[librarian] Claim candidate %s has no valid support facts; rejecting in place.",
                candidate["id"],
            )
            api.update_claim_candidate_review(
                candidate["id"],
                "reject",
                review_note="No valid support facts were available for review.",
            )
            review_results.append(
                {
                    "candidate_id": candidate["id"],
                    "record_kind": "claim",
                    "decision": "reject",
                    "review_note": "No valid support facts were available for review.",
                }
            )
            continue
        rag_context, _ = await build_query_rag_context(
            state["topic_id"],
            candidate["candidate_text"],
            exclude_ids=recent_message_ids,
        )
        prompt = build_claim_review_prompt(
            state, topic, subtopic, candidate, messages, support_facts, rag_context
        )
        try:
            resp_text, provider = await _query_librarian_review_text(
                prompt,
                stage_name=f"Librarian claim review {candidate['id']}",
                validator=lambda text: usable_text_output(text)
                and bool(extract_json(text)),
                topic_id=state["topic_id"],
                subtopic_id=state["subtopic_id"],
            )
            try:
                review = parse_claim_review(
                    resp_text, candidate["candidate_text"], support_ids
                )
            except ValueError:
                fallback_provider = _resolve_stage_provider(state, "control_provider")
                if provider == fallback_provider:
                    raise
                resp_text = await call_text(
                    prompt,
                    provider=fallback_provider,
                    strategy="direct",
                    allow_web=False,
                    system_instruction=PROMPTS["librarian"],
                    temperature=0.7,
                    max_tokens=DEFAULT_MAX_TOKENS,
                    fallback_role="librarian",
                    require_json=True,
                    topic_id=state["topic_id"],
                    subtopic_id=state["subtopic_id"],
                )
                review = parse_claim_review(
                    resp_text, candidate["candidate_text"], support_ids
                )
            result = await apply_claim_review(state["topic_id"], candidate, review)
            review_results.append(result)
        except Exception as exc:
            logger.warning(
                "[librarian] Failed to review claim candidate %s; leaving pending: %s",
                candidate["id"],
                exc,
            )
            continue
        claim_id = result.get("accepted_claim_id")
        stored_text = result.get("stored_text")
        if claim_id and stored_text:
            try:
                summary = await generate_summary(stored_text, max_words=30)
                if summary:
                    api.update_claim_summary(claim_id, summary)
            except Exception as exc:
                logger.warning(
                    "[librarian] Post-hoc summary generation failed for claim %s: %s",
                    claim_id,
                    exc,
                )

    if not review_results or not emit_audit_message:
        return {}

    audit_message = build_librarian_audit_message(review_results)
    await api.persist_message(
        state["topic_id"],
        state["subtopic_id"],
        "librarian",
        audit_message,
        round_number=state.get("round_number", 1),
        turn_kind=LIBRARIAN_AUDIT_TURN,
    )
    return {}


async def bootstrap_fact_intake_node(state: ChatState) -> dict:
    if _is_mse_modeling_state(state):
        logger.info(
            "[mse-workflow] Seeding optimization problem; skipping fact bootstrap."
        )
        _ensure_mse_problem_seed(state)
        _advance_mse_workflow_deterministically(state)
        return {}

    logger.info("[skynet] Bootstrapping baseline facts for the subtopic...")
    topic, subtopic = _load_context_entities(state)
    if not topic or not subtopic:
        return {}

    control_provider = _resolve_stage_provider(state, "control_provider")
    fact_provider = _resolve_stage_provider(state, "fact_provider")
    direction_prompt = build_bootstrap_fact_direction_prompt(topic, subtopic)
    direction_text = await _call_text_with_structured_retry(
        stage_name="Bootstrap fact direction generation",
        validator=_fact_direction_output_is_usable,
        invoke=lambda: call_text(
            direction_prompt,
            provider=control_provider,
            strategy="direct",
            allow_web=False,
            system_instruction=PROMPTS["skynet"],
            fallback_role=SKYNET,
            require_json=True,
        ),
    )
    if not direction_text:
        logger.warning("[skynet] Bootstrap fact direction generation failed.")
        return {}

    parsed_directions = _normalize_fact_direction_contract(direction_text)
    if not parsed_directions["parsed_ok"]:
        logger.warning(
            "[skynet] Bootstrap fact directions were not parseable; skipping bootstrap fact intake."
        )
        return {}

    for direction in parsed_directions["directions"][:BOOTSTRAP_FACT_DIRECTION_LIMIT]:
        try:
            evidence_response = await collect_search_evidence_bundle(
                "fact_proposer",
                f"Gather evidence for this bootstrap fact direction:\n{direction}",
                max_iter=1,
                system_prompt=PROMPTS["fact_proposer"],
                topic_id=state["topic_id"],
                subtopic_id=state["subtopic_id"],
            )
        except Exception as exc:
            logger.warning(
                "[skynet] Bootstrap search failed for direction '%s': %s",
                direction,
                exc,
            )
            continue

        if not _has_usable_search_evidence(evidence_response.search_evidence):
            continue

        evidence_note = _render_search_evidence_note(evidence_response.search_evidence)
        synthetic_messages = [
            {
                "sender": SKYNET,
                "content": f"Bootstrap fact direction: {direction}\n\n{evidence_note}",
                "msg_type": "standard",
            }
        ]
        try:
            rag_context, _ = await build_query_rag_context(
                state["topic_id"],
                direction,
            )
        except Exception as exc:
            logger.warning(
                "[skynet] Bootstrap RAG context failed for direction '%s': %s",
                direction,
                exc,
            )
            rag_context = ""

        proposer_prompt = build_fact_proposer_prompt(
            state,
            topic,
            synthetic_messages,
            rag_context,
            max_facts=1,
            fact_stage="bootstrap",
            focus_label=direction,
        )
        proposer_text = await _call_text_with_structured_retry(
            stage_name=f"Bootstrap fact proposal {direction[:48]}",
            validator=_fact_list_output_is_usable,
            invoke=lambda: call_text(
                proposer_prompt,
                provider=fact_provider,
                strategy="direct",
                allow_web=False,
                system_instruction=PROMPTS["fact_proposer"],
                fallback_role="fact_proposer",
                require_json=True,
            ),
        )
        if not proposer_text:
            continue

        parsed = _normalize_fact_proposal_contract(proposer_text)
        if not parsed["parsed_ok"]:
            continue

        await process_writer_output(
            state["topic_id"],
            state["subtopic_id"],
            None,
            "",
            structured_facts=parsed["facts"],
            fact_stage="bootstrap",
            evidence_note=evidence_note,
            round_number=state.get("round_number"),
            max_candidates=1,
        )

    return {}


async def _run_inline_fact_intake(
    state: ChatState,
    *,
    actor: str,
    topic: dict,
    subtopic: dict | None,
    rag_context: str,
    search_evidence: Sequence[SearchEvidenceItem],
) -> None:
    if actor == SPECTATOR:
        return
    if actor not in DELIBERATORS and actor not in special_agents():
        return
    if not _has_usable_search_evidence(search_evidence):
        return

    fact_provider = _resolve_stage_provider(state, "fact_provider")
    messages = api.get_messages(
        state["topic_id"], subtopic_id=state["subtopic_id"], limit=8
    )
    evidence_context = _render_search_evidence_context(search_evidence)
    if not evidence_context:
        return
    prompt_messages = messages + [
        {
            "sender": "web_search",
            "content": evidence_context,
            "msg_type": "standard",
        }
    ]
    prompt = build_fact_proposer_prompt(
        state,
        topic,
        prompt_messages,
        rag_context,
        max_facts=1,
        fact_stage="inline",
        focus_label=f"Turn actor: {actor}",
    )
    proposer_text = await _call_text_with_structured_retry(
        stage_name=f"Inline fact proposal {actor}",
        validator=_fact_list_output_is_usable,
        invoke=lambda: call_text(
            prompt,
            provider=fact_provider,
            strategy="direct",
            allow_web=False,
            system_instruction=PROMPTS["fact_proposer"],
            fallback_role="fact_proposer",
            require_json=True,
        ),
    )
    if not proposer_text:
        return

    parsed = _normalize_fact_proposal_contract(proposer_text)
    if not parsed["parsed_ok"]:
        return

    await process_writer_output(
        state["topic_id"],
        state["subtopic_id"],
        None,
        "",
        structured_facts=parsed["facts"],
        fact_stage="inline",
        evidence_note=_render_search_evidence_note(search_evidence),
        round_number=state.get("round_number"),
        max_candidates=INLINE_FACT_LIMIT,
    )


async def _run_single_agent_turn(state: ChatState, actor: str, turn_kind: str) -> dict:
    """Execute one agent turn: context loading, prompt building, LLM call,
    citation sanitization.  Pure computation -- no persistence side-effects.

    Returns a dict with keys:
        actor, turn_kind, content, confidence_score, search_evidence,
        spectator_data (dict if actor is SPECTATOR, else None),
        targets (dict of dog_target/cat_target/tron_target extracted),
        rag_degraded, search_failed,
        topic, subtopic, rag_context (carried for persistence).

    If the topic cannot be loaded, returns a minimal dict with ``no_topic=True``.
    """
    topic, subtopic = _load_context_entities(state)
    if not topic:
        return {"actor": actor, "turn_kind": turn_kind, "no_topic": True}

    messages = api.get_messages(
        state["topic_id"], subtopic_id=state["subtopic_id"], limit=6
    )
    # AR-4: Sparse communication — filter messages by role visibility
    visibility = ROLE_VISIBILITY.get(actor)
    if visibility is not None:
        messages = [m for m in messages if m.get("sender") in visibility]
    summary_messages = api.get_messages(
        state["topic_id"],
        subtopic_id=state["subtopic_id"],
        limit=1,
        msg_type="summary",
    )
    latest_summary = summary_messages[-1]["content"] if summary_messages else ""
    provider = _resolve_agent_provider(state, actor)
    web_provider = _resolve_stage_provider(state, "web_provider")
    rag_provider = _resolve_stage_provider(state, "rag_provider")
    api_consult_provider = _resolve_stage_provider(state, "api_consult_provider")

    rag_messages = _seed_messages_for_rag(topic, subtopic, messages)
    system_prompt = build_actor_system_prompt(
        state, actor, turn_kind, subtopic_data=subtopic
    )
    planner_prompt = build_actor_prompt(
        state,
        actor,
        turn_kind,
        topic,
        subtopic,
        messages,
        "",
        latest_summary=latest_summary,
        include_output_contract=False,
    )
    rag_context, rag_degraded = await assemble_rag_context(
        topic["id"],
        subtopic["id"] if subtopic else 0,
        rag_messages,
        actor,
        planner_system_prompt=system_prompt,
        planner_context=planner_prompt,
        latest_summary=latest_summary,
        allow_web_backup=should_enable_web_backup(state, actor, turn_kind),
        planner_provider=rag_provider,
    )
    consult_context = await _maybe_run_llm_api_consult(
        state=state,
        actor=actor,
        turn_kind=turn_kind,
        provider=api_consult_provider,
        topic=topic,
        subtopic=subtopic,
        messages=messages,
        latest_summary=latest_summary,
    )
    if consult_context:
        rag_context = f"{rag_context.rstrip()}\n\n{consult_context}".strip() + "\n\n"
    prompt = build_actor_prompt(
        state,
        actor,
        turn_kind,
        topic,
        subtopic,
        messages,
        rag_context,
        latest_summary=latest_summary,
    )

    # Model call depending on role and phase
    search_failed = False
    search_evidence: Sequence[SearchEvidenceItem] = ()
    if should_enable_web_search(state, actor, turn_kind):
        logger.info(f"[{actor}] Entering ReAct search loop...")
        try:
            response = await _call_text_with_structured_retry(
                stage_name=f"{actor} web turn",
                validator=lambda item: (
                    bool(_normalize_focus_contract(item.text)["parsed_ok"])
                    if actor == SPECTATOR
                    else _structured_message_is_usable(
                        item.text, accepted_actions=("post_message",)
                    )
                ),
                invoke=lambda: call_text_with_search_evidence(
                    prompt,
                    provider=web_provider,
                    strategy="react",
                    allow_web=True,
                    system_instruction=system_prompt,
                    fallback_role=actor,
                    topic_id=state["topic_id"],
                    subtopic_id=state["subtopic_id"],
                ),
            )
            if response is None:
                raise RuntimeError("structured retry exhausted")
            resp_text = response.text
            search_evidence = response.search_evidence
            search_failed = response.search_failed
        except Exception as exc:
            logger.warning("[%s] Web-enhanced broker call failed: %s", actor, exc)
            resp_text = json.dumps(
                {
                    "action": "post_message",
                    "content": f"[System] {actor} was unable to contribute this turn due to a transient service issue.",
                }
            )
            search_failed = True
    else:
        try:
            resp_text = await _call_text_with_structured_retry(
                stage_name=f"{actor} direct turn",
                validator=lambda text: (
                    bool(_normalize_focus_contract(text)["parsed_ok"])
                    if actor == SPECTATOR
                    else _structured_message_is_usable(
                        text, accepted_actions=("post_message",)
                    )
                ),
                invoke=lambda: call_text(
                    prompt,
                    provider=provider,
                    strategy="direct",
                    allow_web=False,
                    system_instruction=system_prompt,
                    fallback_role=actor,
                    require_json=True,
                ),
            )
            if resp_text is None:
                raise RuntimeError("structured retry exhausted")
        except Exception as exc:
            logger.warning("[%s] Direct broker call failed: %s", actor, exc)
            resp_text = json.dumps(
                {
                    "action": "post_message",
                    "content": f"[System] {actor} was unable to contribute this turn due to a transient service issue.",
                }
            )

    # Spectator: return parsed focus data, no persistence needed
    if actor == SPECTATOR:
        parsed_focus = _normalize_focus_contract(resp_text)
        return {
            "actor": actor,
            "turn_kind": turn_kind,
            "spectator_data": parsed_focus,
            "content": "",
            "confidence_score": None,
            "search_evidence": search_evidence,
            "targets": {},
            "rag_degraded": rag_degraded,
            "search_failed": search_failed,
        }

    # Non-spectator: parse message contract and sanitize citations
    fallback_confidence = (
        DEGRADED_OPERATION_CONFIDENCE if (rag_degraded or search_failed) else None
    )
    parsed = _normalize_message_contract(
        resp_text, fallback_confidence=fallback_confidence
    )
    citation_knowledge_blocks = [rag_context] + [
        item.rendered_results for item in search_evidence if item.rendered_results
    ]
    trusted_api_blocks = [rag_context]
    if latest_summary:
        citation_knowledge_blocks.append(latest_summary)
    for msg in messages:
        if msg.get("content"):
            citation_knowledge_blocks.append(msg["content"])

    content, removed_citations = _sanitize_citations_to_allowed_ids(
        parsed["content"],
        knowledge_blocks=citation_knowledge_blocks,
        trusted_api_blocks=trusted_api_blocks,
    )
    if any(removed_citations.values()):
        logger.info(
            "[%s] Stripped citations not present in injected knowledge D=%s F=%s C=%s W=%s L=%s A=%s E=%s",
            actor,
            removed_citations["D"],
            removed_citations["F"],
            removed_citations["C"],
            removed_citations["W"],
            removed_citations["L"],
            removed_citations["A"],
            removed_citations["E"],
        )
    # Targeted sync retry for non-NPC agents with uncited financial numbers
    from .agents import is_npc

    if not is_npc(actor) and actor != SPECTATOR:
        retried_content = await _targeted_sync_retry(
            content, state.get("round_number", 0), system_prompt, prompt, actor, state
        )
        if retried_content != content:
            # Re-sanitize: retry may have introduced new citations
            content, retry_removed = _sanitize_citations_to_allowed_ids(
                retried_content,
                knowledge_blocks=citation_knowledge_blocks,
                trusted_api_blocks=trusted_api_blocks,
            )
            if any(retry_removed.values()):
                    logger.info(
                        "[%s] Sync retry re-sanitized: D=%s F=%s C=%s W=%s L=%s A=%s E=%s",
                        actor,
                        retry_removed["D"],
                        retry_removed["F"],
                        retry_removed["C"],
                        retry_removed["W"],
                        retry_removed["L"],
                        retry_removed["A"],
                        retry_removed["E"],
                    )

    confidence_score = parsed["confidence_score"]
    if not parsed["parsed_ok"]:
        confidence_score = min(
            (
                confidence_score
                if confidence_score is not None
                else PARSER_FAILURE_CONFIDENCE
            ),
            PARSER_FAILURE_CONFIDENCE,
        )

    # Extract targets from content
    targets: dict[str, str] = {}
    if turn_kind == BASE_TURN and actor in ("dog", "cat", "tron"):
        target = _extract_target_from_content(content, actor)
        if target:
            targets[f"{actor}_target"] = target

    return {
        "actor": actor,
        "turn_kind": turn_kind,
        "content": content,
        "confidence_score": confidence_score,
        "search_evidence": search_evidence,
        "spectator_data": None,
        "targets": targets,
        "rag_degraded": rag_degraded,
        "search_failed": search_failed,
        # Carried for persistence
        "topic": topic,
        "subtopic": subtopic,
        "rag_context": rag_context,
        # G.4: Optional formal claim attachment
        "formal_claim": parsed.get("formal_claim"),
        "mse_artifact_update": parsed.get("mse_artifact_update"),
    }


def _build_code_followup_context(evidence_ids: list[int]) -> str:
    """Build a context string summarizing code evidence results for a follow-up turn."""
    parts: list[str] = []
    for eid in evidence_ids:
        ev = api.get_code_evidence_by_id(eid)
        if not ev:
            continue
        status = "PASSED" if ev.get("success") else "FAILED"
        stdout_preview = (ev.get("stdout") or "").strip()[:1500]
        parts.append(
            f"[E{eid}] {status} — {ev.get('hypothesis', '')}\n"
            f"Output: {stdout_preview}"
        )
    return "\n\n".join(parts) if parts else "No code evidence results available."


def _format_calc_note(expr: str, calc_result: Any) -> str:
    """Render a compact calc note for persistence."""
    success = getattr(calc_result, "success", False)
    stdout_val = " ".join((getattr(calc_result, "stdout", "") or "").split())[:500]
    stderr_val = " ".join((getattr(calc_result, "stderr", "") or "").split())[:500]

    if success and stdout_val:
        return f"[Calc: {expr} = {stdout_val}]"

    # Distinguish rejection (validation failed, never ran) from runtime failures
    if stderr_val:
        if stderr_val.startswith("Calc rejected:"):
            return f"[Calc rejected: {expr} -> {stderr_val}]"
        return f"[Calc failed: {expr} -> {stderr_val}]"

    return f"[Calc failed: {expr}]"


def _persist_mse_artifact_update(state: ChatState, result: dict, msg_id: int) -> None:
    if not _is_mse_modeling_state(state):
        return
    update = result.get("mse_artifact_update")
    if not isinstance(update, dict):
        return
    problem = _ensure_mse_problem_seed(state)
    if not problem:
        return

    from .optimization import persist_component_payloads, persist_lp_artifact

    components = update.get("components") or update.get("optimization_components")
    if isinstance(components, list):
        payloads = [item for item in components if isinstance(item, dict)]
        if payloads:
            persist_component_payloads(
                topic_id=int(state["topic_id"]),
                problem_id=int(problem["id"]),
                payloads=payloads,
                default_review_status="candidate",
            )

    reviews = update.get("component_reviews")
    if isinstance(reviews, list):
        for review in reviews:
            if not isinstance(review, dict):
                continue
            try:
                component_id = int(review.get("component_id") or review.get("id"))
            except (TypeError, ValueError):
                continue
            status = str(review.get("review_status") or review.get("status") or "")
            if status not in {"candidate", "reviewed", "rejected", "formalized", "executable"}:
                continue
            api.update_optimization_component_review(
                component_id,
                review_status=status,
                validation_notes=str(review.get("validation_notes") or f"Updated from message M{msg_id}."),
            )

    lp_artifact = update.get("lp_artifact") or update.get("artifact_content")
    if isinstance(lp_artifact, str) and lp_artifact.strip():
        persist_lp_artifact(
            topic_id=int(state["topic_id"]),
            problem_id=int(problem["id"]),
            content=lp_artifact.strip(),
            linked_component_ids=_json_ids(update.get("linked_component_ids")),
            generator_role=str(update.get("generator_role") or result.get("actor") or "mse_agent"),
        )

    _advance_mse_workflow_deterministically(state)


async def _persist_agent_result(state: ChatState, result: dict) -> list[int]:
    """Persist one agent result to the database: message + inline fact intake + code execution.

    Args:
        state: Current chat state.
        result: Dict returned by ``_run_single_agent_turn``.

    Returns:
        List of code evidence IDs created (empty if none).
    """
    actor = result["actor"]
    turn_kind = result["turn_kind"]

    msg_id = await api.persist_message(
        state["topic_id"],
        state["subtopic_id"],
        actor,
        result["content"],
        confidence_score=result["confidence_score"],
        round_number=state.get("round_number", 1),
        turn_kind=turn_kind,
    )

    # G.4: Process optional formal_claim attachment
    formal_claim = result.get("formal_claim")
    if formal_claim and isinstance(formal_claim, dict):
        gate_result = validate_formal_claim(formal_claim)
        if gate_result["passed"]:
            try:
                await asyncio.to_thread(
                    process_formal_claim,
                    state["topic_id"],
                    state.get("subtopic_id"),
                    msg_id,
                    actor,
                    formal_claim,
                )
            except Exception as exc:
                logger.warning(
                    "[G4] Formal claim processing failed for %s: %s", actor, exc
                )
        else:
            logger.info(
                "[G4] Formal claim from %s rejected: %s",
                actor,
                gate_result.get("error"),
            )
            # Silent degradation: clerk will sweep this message later

    await asyncio.to_thread(_persist_mse_artifact_update, state, result, msg_id)

    if result["search_evidence"]:
        await _run_inline_fact_intake(
            state,
            actor=actor,
            topic=result["topic"],
            subtopic=result["subtopic"],
            rag_context=result["rag_context"],
            search_evidence=result["search_evidence"],
        )

    # Three-tier code execution
    tier = _get_code_tier(state, actor, turn_kind)
    if not tier:
        return []

    sandbox_ready = await asyncio.to_thread(is_sandbox_ready)
    if not sandbox_ready:
        return []

    code_evidence_ids: list[int] = []
    content = result["content"]
    topic_id = state["topic_id"]
    subtopic_id = state["subtopic_id"]
    round_number = state.get("round_number", 1)
    rag_ctx = result.get("rag_context") or ""
    code_provider = _resolve_topic_provider(topic_id, "code_provider")

    # Tier 1: CALC — inline, no LLM, no FactCandidate
    calc_exprs = _extract_calc_requests(content)
    for expr in calc_exprs:
        try:
            calc_result = await run_calc(
                expr,
                topic_id=topic_id,
                subtopic_id=subtopic_id,
                role=actor,
            )
            code_note = _format_calc_note(expr, calc_result)
            await api.persist_message(
                topic_id,
                subtopic_id,
                "system",
                code_note,
                msg_type="standard",
                round_number=round_number,
                turn_kind="code_evidence",
            )
            logger.info(
                "[CALC] %s persisted calc note for '%s': %s",
                actor,
                expr[:60],
                code_note[:120],
            )
        except Exception as exc:
            logger.warning("[CALC] Calc execution failed for %s: %s", actor, exc)

    # Tier 2: CODE_VERIFY — full loop + FactCandidate + follow-up eligible
    if tier == "verify":
        hypotheses = _extract_code_verify_requests(content)
        for hypothesis in hypotheses:
            if not hypothesis:
                continue
            try:
                code_result = await run_code_evidence(
                    hypothesis,
                    rag_ctx,
                    topic_id=topic_id,
                    subtopic_id=subtopic_id,
                    role=actor,
                    provider=code_provider,
                )
                if code_result.planning_veto:
                    code_note = (
                        f"\n\n[Code planning veto] Hypothesis: {hypothesis}\n"
                        f"{code_result.rendered_results}"
                    )
                    await api.persist_message(
                        topic_id,
                        subtopic_id,
                        "system",
                        code_note.strip(),
                        msg_type="standard",
                        round_number=round_number,
                        turn_kind="code_evidence",
                    )
                    logger.info(
                        "[CODE_VERIFY] %s planning veto for '%s': %s",
                        actor,
                        hypothesis[:60],
                        (code_result.stderr or "")[:200],
                    )
                    continue
                if code_result.code_evidence_id:
                    code_note = (
                        f"\n\n[E{code_result.code_evidence_id}] Code result: "
                        f"{code_result.rendered_results}"
                    )
                    await api.persist_message(
                        topic_id,
                        subtopic_id,
                        "system",
                        code_note.strip(),
                        msg_type="standard",
                        round_number=round_number,
                        turn_kind="code_evidence",
                    )
                    try:
                        candidate_text = (
                            f"Code verification of: {hypothesis}\n"
                            f"Result: {'PASSED' if code_result.success else 'FAILED'}\n"
                            f"Output: {(code_result.stdout or '').strip()[:1000]}"
                        )
                        api.create_fact_candidate_with_stage(
                            topic_id=topic_id,
                            subtopic_id=subtopic_id,
                            writer_msg_id=None,
                            candidate_text=candidate_text,
                            fact_stage="code_verified",
                            candidate_type="code_evidence",
                            source_kind="code",
                            source_refs_json=json.dumps(
                                [f"E{code_result.code_evidence_id}"]
                            ),
                            round_number=round_number,
                        )
                    except Exception as exc:
                        logger.warning(
                            "[CodeSandbox] Failed to create fact candidate: %s", exc
                        )
                    # CE-1: extract structured Ledger + FactCandidates from code output
                    if code_result.success:
                        try:
                            from .evidence_parser import extract_from_code_evidence

                            await extract_from_code_evidence(
                                topic_id,
                                subtopic_id,
                                code_evidence_id=code_result.code_evidence_id,
                                hypothesis=hypothesis,
                                stdout=code_result.stdout or "",
                                current_round=round_number or 0,
                            )
                        except Exception as exc:
                            logger.warning(
                                "[CE-1] Code evidence extraction failed for E%s: %s",
                                code_result.code_evidence_id,
                                exc,
                            )
                    code_evidence_ids.append(code_result.code_evidence_id)
                    logger.info(
                        "[CODE_VERIFY] %s executed code for '%s': E%d (%s)",
                        actor,
                        hypothesis[:60],
                        code_result.code_evidence_id,
                        "success" if code_result.success else "failed",
                    )
            except Exception as exc:
                logger.warning(
                    "[CODE_VERIFY] Code execution failed for %s: %s", actor, exc
                )

    # Tier 3: CODE_REVIEW — critic only, max 1
    if tier == "review":
        review_req = _extract_code_review_request(content)
        if review_req:
            evidence_id, critique = review_req
            try:
                review_result = await run_code_review(
                    evidence_id,
                    critique,
                    topic_id=topic_id,
                    subtopic_id=subtopic_id,
                    role=actor,
                    provider=code_provider,
                )
                if review_result.code_evidence_id:
                    code_note = (
                        f"\n\n[E{review_result.code_evidence_id}] Code review of E{evidence_id}: "
                        f"{review_result.rendered_results}"
                    )
                    await api.persist_message(
                        topic_id,
                        subtopic_id,
                        "system",
                        code_note.strip(),
                        msg_type="standard",
                        round_number=round_number,
                        turn_kind="code_evidence",
                    )
                    try:
                        candidate_text = (
                            f"Code review of E{evidence_id}: {critique}\n"
                            f"Result: {'PASSED' if review_result.success else 'FAILED'}\n"
                            f"Output: {(review_result.stdout or '').strip()[:1000]}"
                        )
                        api.create_fact_candidate_with_stage(
                            topic_id=topic_id,
                            subtopic_id=subtopic_id,
                            writer_msg_id=None,
                            candidate_text=candidate_text,
                            fact_stage="code_verified",
                            candidate_type="code_evidence",
                            source_kind="code",
                            source_refs_json=json.dumps(
                                [f"E{review_result.code_evidence_id}"]
                            ),
                            round_number=round_number,
                        )
                    except Exception as exc:
                        logger.warning(
                            "[CODE_REVIEW] Failed to create fact candidate: %s", exc
                        )
                    code_evidence_ids.append(review_result.code_evidence_id)
                    logger.info(
                        "[CODE_REVIEW] critic reviewed E%d: E%d (%s)",
                        evidence_id,
                        review_result.code_evidence_id,
                        "success" if review_result.success else "failed",
                    )
            except Exception as exc:
                logger.warning(
                    "[CODE_REVIEW] Code review failed for %s: %s", actor, exc
                )

    # Tier 4: CODE_VERIFY_GRID — multi-seed/size sweep, requires PASSED Full evidence
    if tier == "verify" and actor in _CODE_EXEC_ROLES:
        grid_req = _extract_code_grid_request(content)
        if grid_req:
            full_eid, grid_desc = grid_req
            try:
                grid_result = await run_code_evidence_grid(
                    grid_desc,
                    rag_ctx,
                    topic_id=topic_id,
                    subtopic_id=subtopic_id,
                    role=actor,
                    full_evidence_id=full_eid,
                    provider=code_provider,
                )
                if grid_result.code_evidence_id:
                    code_note = (
                        f"\n\n[E{grid_result.code_evidence_id}] Grid sweep of E{full_eid}: "
                        f"{grid_result.rendered_results}"
                    )
                else:
                    # Grid failed before DB persistence — still notify the agent
                    code_note = (
                        f"\n\n[Grid FAILED] Sweep of E{full_eid}: "
                        f"{grid_result.rendered_results or grid_result.stderr or 'Unknown error'}"
                    )
                # Always persist the result message so the agent sees feedback
                await api.persist_message(
                    topic_id,
                    subtopic_id,
                    "system",
                    code_note.strip(),
                    msg_type="standard",
                    round_number=round_number,
                    turn_kind="code_evidence",
                )
                if grid_result.code_evidence_id:
                    try:
                        candidate_text = (
                            f"Grid verification of E{full_eid}: {grid_desc}\n"
                            f"Result: {'PASSED' if grid_result.success else 'FAILED'}\n"
                            f"Output: {(grid_result.stdout or '').strip()[:1000]}"
                        )
                        api.create_fact_candidate_with_stage(
                            topic_id=topic_id,
                            subtopic_id=subtopic_id,
                            writer_msg_id=None,
                            candidate_text=candidate_text,
                            fact_stage="code_verified",
                            candidate_type="code_evidence",
                            source_kind="code",
                            source_refs_json=json.dumps(
                                [f"E{grid_result.code_evidence_id}"]
                            ),
                            round_number=round_number,
                        )
                    except Exception as exc:
                        logger.warning(
                            "[CODE_VERIFY_GRID] Failed to create fact candidate: %s",
                            exc,
                        )
                    code_evidence_ids.append(grid_result.code_evidence_id)
                    # CE-1: extract structured Ledger entries from grid output
                    if grid_result.success:
                        try:
                            from .evidence_parser import extract_from_code_evidence

                            await extract_from_code_evidence(
                                topic_id,
                                subtopic_id,
                                code_evidence_id=grid_result.code_evidence_id,
                                hypothesis=grid_desc,
                                stdout=grid_result.stdout or "",
                                current_round=round_number or 0,
                            )
                        except Exception as exc:
                            logger.warning(
                                "[CE-1] Grid evidence extraction failed for E%s: %s",
                                grid_result.code_evidence_id,
                                exc,
                            )
                    logger.info(
                        "[CODE_VERIFY_GRID] %s grid sweep of E%d: E%d (%s)",
                        actor,
                        full_eid,
                        grid_result.code_evidence_id,
                        "success" if grid_result.success else "failed",
                    )
            except Exception as exc:
                logger.warning(
                    "[CODE_VERIFY_GRID] Grid execution failed for %s: %s", actor, exc
                )

    return code_evidence_ids


async def expert_node(state: ChatState) -> dict:
    actor = state["current_actor"]
    turn_kind = state.get("current_turn_kind", BASE_TURN)
    logger.info(f"[{actor}] Speaking (Round {state.get('round_number', 1)})...")

    result = await _run_single_agent_turn(state, actor, turn_kind)

    if result.get("no_topic"):
        return {"current_actor": ""}

    updates: dict = {"current_actor": "", "current_turn_kind": ""}

    # Handle spectator (no persistence needed)
    if result.get("spectator_data"):
        parsed_focus = result["spectator_data"]
        if parsed_focus["parsed_ok"]:
            updates["spectator_target"] = parsed_focus["target"]
            updates["spectator_web_boost_target"] = (
                parsed_focus["target"] if parsed_focus["grant_web_search"] else None
            )
        return updates

    try:
        code_evidence_ids = await _persist_agent_result(state, result)
    except Exception as exc:
        logger.error("[expert_node] Failed to persist %s result: %s", actor, exc)
        return updates

    # Code follow-up turn
    if code_evidence_ids and turn_kind == BASE_TURN:
        try:
            followup_ctx = _build_code_followup_context(code_evidence_ids)
            followup_state = {**dict(state), "_code_followup_context": followup_ctx}
            followup_result = await _run_single_agent_turn(
                followup_state, actor, CODE_FOLLOWUP_TURN
            )
            if not followup_result.get("no_topic"):
                await _persist_agent_result(followup_state, followup_result)
        except Exception as exc:
            logger.warning("[expert_node] Code follow-up for %s failed: %s", actor, exc)

    # Peanut gallery targeting logic
    _clear_consumed_extra_target(turn_kind, updates)
    if turn_kind == BASE_TURN and actor == state.get("spectator_target"):
        updates["spectator_target"] = None
        updates["spectator_web_boost_target"] = None

    if result.get("targets"):
        updates.update(result["targets"])

    if any(key in updates for key in {"dog_target", "cat_target", "tron_target"}):
        _refresh_pending_turns_with_extras(state, updates)

    return updates


def _extract_evidence_gaps_section(summary_text: str) -> str:
    """Extract the EVIDENCE GAPS section from a summary."""
    lines = (summary_text or "").split("\n")
    in_gaps = False
    gap_lines: list[str] = []
    for line in lines:
        upper = line.strip().upper()
        if "EVIDENCE GAP" in upper:
            in_gaps = True
            continue
        if in_gaps:
            # Stop at next major section header (## or known summary headers)
            stripped = line.strip()
            if stripped.startswith("##") or any(
                stripped.upper().startswith(h.rstrip(":").upper())
                for h in SUMMARY_SECTION_HEADERS
                if h.upper() != "EVIDENCE GAPS:"
            ):
                break
            gap_lines.append(line)
    return "\n".join(gap_lines).strip()


async def _diff_evidence_gaps(
    current_gaps_text: str, prev_gaps: list[dict], *, provider: str = "minimax"
) -> dict:
    """Cheap Flash call to diff current vs previous gaps. Returns {persistent, new, resolved}."""
    if not current_gaps_text.strip():
        return {"persistent": [], "new": [], "resolved": []}
    prev_block = (
        "\n".join(
            f"- [{g.get('id', '?')}] {g.get('description', '')}" for g in prev_gaps
        )
        or "(none)"
    )
    prompt = (
        f"Previous active gaps:\n{prev_block}\n\n"
        f"Current EVIDENCE GAPS section:\n{current_gaps_text}\n\n"
        f'Output JSON: {{"persistent": [{{"id": "...", "description": "..."}}], '
        f'"new": [{{"id": "...", "description": "..."}}], "resolved": ["id_1"]}}'
    )
    try:
        resp = await call_text(
            prompt,
            provider=provider,
            strategy="direct",
            temperature=0.2,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        result = extract_json(resp) if resp else None
        if isinstance(result, dict):
            return result
    except Exception as exc:
        logger.warning("[gap-diff] Failed: %s", exc)
    return {"persistent": [], "new": [], "resolved": []}


async def audience_summary_node(state: ChatState) -> dict:
    current_round = state.get("round_number", 1)
    if _is_mse_modeling_state(state):
        logger.info(
            "[mse-workflow] Skipping round summary for artifact-driven round %s.",
            current_round,
        )
        return {"last_summary_round": current_round}
    if state.get("last_summary_round") == current_round:
        logger.info(
            "[skynet] Summary for round %s already exists. Skipping.", current_round
        )
        return {}

    topic, _ = _load_context_entities(state)
    if not topic:
        return {}

    logger.info("[skynet] Summarizing round %s...", current_round)
    control_provider = _resolve_stage_provider(state, "control_provider")
    messages = api.get_messages(
        state["topic_id"], subtopic_id=state["subtopic_id"], limit=20
    )

    prompt = build_audience_summary_prompt(state, topic, messages)
    resp_text = await _call_text_with_structured_retry(
        stage_name="Round summary generation",
        validator=lambda text: _structured_message_is_usable(
            text, accepted_actions=("post_summary", "post_message")
        ),
        invoke=lambda: call_text(
            prompt,
            provider=control_provider,
            strategy="direct",
            allow_web=False,
            system_instruction=PROMPTS["skynet"],
            fallback_role=SKYNET,
            require_json=True,
        ),
    )
    if not resp_text or not resp_text.strip():
        resp_text = json.dumps(
            {
                "action": "post_summary",
                "content": _build_degraded_audience_summary(state, messages),
            }
        )

    parsed = _normalize_message_contract(
        resp_text, accepted_actions=("post_summary", "post_message")
    )
    content = parsed["content"]
    if not _has_required_summary_sections(content):
        # Attempt to repair the malformed summary using decomposition
        content = await _repair_summary_by_decomposition(
            content, provider=control_provider
        )
        # Final sanity check: if the repair also fails to produce the headers, then degrade.
        if not _has_required_summary_sections(content):
            logger.warning(
                "[skynet] Decomposition repair failed to produce required sections; using degraded fallback."
            )
            content = _build_degraded_audience_summary(state, messages)

    # Embed the summary for future cyclicality detection
    emb = await aget_embedding(content)
    if emb:
        msg_id = api.insert_message_with_embedding(
            state["topic_id"],
            state["subtopic_id"],
            SKYNET,
            content,
            msg_type="summary",
            embedding=emb,
            round_number=current_round,
            turn_kind=AUDIENCE_SUMMARY_TURN,
        )
    else:
        msg_id = api.post_message(
            state["topic_id"],
            state["subtopic_id"],
            SKYNET,
            content,
            msg_type="summary",
            round_number=current_round,
            turn_kind=AUDIENCE_SUMMARY_TURN,
        )

    updates: dict = {
        "latest_summary_msg_id": msg_id,
        "last_summary_round": current_round,
    }

    # --- Gap-triggered search (Phase 5) ---
    if current_round >= 3:
        try:
            gaps_text = _extract_evidence_gaps_section(content)
            prev_gaps = state.get("active_evidence_gaps") or []
            gap_diff = await _diff_evidence_gaps(
                gaps_text, prev_gaps, provider=control_provider
            )

            persistent = gap_diff.get("persistent", [])
            new_gaps = gap_diff.get("new", [])
            if not isinstance(persistent, list):
                persistent = []
            if not isinstance(new_gaps, list):
                new_gaps = []

            # Track rounds_active and gap_search_attempts
            prev_map = {g.get("id"): g for g in prev_gaps if isinstance(g, dict)}
            active_gaps: list[dict] = []
            for g in persistent + new_gaps:
                if not isinstance(g, dict) or not g.get("description"):
                    continue
                gap_id = g.get("id", g.get("description", "")[:20])
                prev = prev_map.get(gap_id, {})
                active_gaps.append(
                    {
                        "id": gap_id,
                        "description": g["description"],
                        "rounds_active": prev.get("rounds_active", 0) + 1,
                        "search_attempts": prev.get("search_attempts", 0),
                    }
                )

            # Remove unresolvable gaps (searched but no new facts came)
            active_gaps = [g for g in active_gaps if g.get("search_attempts", 0) < 2]

            updates["active_evidence_gaps"] = active_gaps

            # Auto-boost: if persistent gaps exist and no active gap search
            was_gap_search = state.get("gap_search_active", False)
            if was_gap_search:
                # Previous gap search round completed — reset
                updates["gap_search_active"] = False
                updates["gap_search_directive"] = None
                # Increment search_attempts for the gap that was searched
                prev_directive = state.get("gap_search_directive")
                if prev_directive and isinstance(prev_directive, dict):
                    for g in active_gaps:
                        if g.get("id") == prev_directive.get("id"):
                            g["search_attempts"] = g.get("search_attempts", 0) + 1
                    updates["active_evidence_gaps"] = active_gaps

            persistent_unsearched = [
                g
                for g in active_gaps
                if g.get("rounds_active", 0) >= 2 and g.get("search_attempts", 0) == 0
            ]
            effective_gap_active = updates.get(
                "gap_search_active", state.get("gap_search_active", False)
            )
            if persistent_unsearched and not effective_gap_active:
                target_gap = persistent_unsearched[0]
                updates["spectator_web_boost_target"] = "analyst"
                updates["gap_search_directive"] = target_gap
                updates["gap_search_active"] = True
                logger.info(
                    "[gap-search] Auto-boosting analyst for gap: %s",
                    target_gap.get("description"),
                )
        except Exception as exc:
            logger.warning("[gap-search] Gap tracking failed: %s", exc)

    return updates


async def audience_termination_check_node(state: ChatState) -> dict:
    logger.info("[skynet] Checking termination and cyclicality...")
    current_round = state.get("round_number", 1)
    if _is_mse_modeling_state(state):
        snapshot = _advance_mse_workflow_deterministically(state)
        if snapshot.get("solved"):
            logger.info("[mse-workflow] Closing subtopic because solver-backed claim is active.")
            return {
                "subtopic_exhausted": True,
                "close_reason": "mse_model_solved",
                "phase": snapshot.get("phase", MSE_SOLVED_PHASE),
            }
        max_rounds = topic_config.get_int(int(state["topic_id"]), "max_rounds")
        if current_round >= max_rounds:
            logger.info(
                "[mse-workflow] Closing subtopic at configured max_rounds=%s with status=%s.",
                max_rounds,
                snapshot.get("status"),
            )
            return {
                "subtopic_exhausted": True,
                "close_reason": f"mse_workflow_stopped:{snapshot.get('status')}",
                "phase": snapshot.get("phase", state.get("phase")),
            }
        return {
            "subtopic_exhausted": False,
            "phase": snapshot.get("phase", state.get("phase")),
        }

    if current_round >= 10:
        logger.info("[skynet] Forcing subtopic close at round %s.", current_round)
        return {"subtopic_exhausted": True}

    topic, subtopic = _load_context_entities(state)
    if not topic or not subtopic:
        return {"subtopic_exhausted": True}

    # --- Circuit breaker: cognitive yield via fact count ---
    current_fact_count = api.count_facts(
        state["topic_id"], subtopic_id=state.get("subtopic_id")
    )
    prev_1 = state.get("fact_count_1_round_ago", 0)
    prev_2 = state.get("fact_count_2_rounds_ago", 0)
    gap_search_pending = state.get("gap_search_active", False)

    stale_2_rounds = (current_fact_count == prev_1 == prev_2) and (
        prev_2 > 0 or current_round >= 5
    )

    if stale_2_rounds and not gap_search_pending and current_round >= 4:
        logger.info(
            "[governance] Circuit breaker: fact count unchanged for 2 rounds (%d)",
            current_fact_count,
        )
        return {
            "subtopic_exhausted": True,
            "close_reason": "cognitive_yield_exhausted",
            "fact_count_2_rounds_ago": prev_1,
            "fact_count_1_round_ago": current_fact_count,
        }

    messages = api.get_messages(
        state["topic_id"], subtopic_id=state["subtopic_id"], limit=20
    )

    phase = state.get("phase", get_phase_for_round(current_round))
    ctx = f"Round: {current_round}\nPhase: {phase}\nTopic: {topic['summary']}\n"
    for m in messages:
        ctx += f"{_format_message_for_prompt(m)}\n"

    # Find the most recent summary we just posted to use as a search query
    recent_summary = ""
    for message in reversed(messages):
        if message.get("msg_type") == "summary":
            recent_summary = message["content"]
            break

    historical_context = ""
    loop_detected = False
    if recent_summary:
        query_emb = await aget_embedding(recent_summary)
        if query_emb:
            # Search for past summaries (limit 3 to avoid prompt bloat)
            past_summaries = api.search_messages_hybrid(
                state["topic_id"],
                recent_summary,
                query_emb,
                msg_type="summary",
                top_k=4,
                exclude_ids=(
                    [state.get("latest_summary_msg_id")]
                    if state.get("latest_summary_msg_id")
                    else None
                ),
            )
            if past_summaries:
                historical_context = "\n=== HISTORICAL SUMMARIES ===\n"
                for ps in past_summaries:
                    # Skip if it's the exact same recent summary (distance ~ 0)
                    if ps.get("distance", 1.0) > 0.05:
                        historical_context += f"Past Conclusion: {ps['content']}\n"
                    if ps.get("distance", 1.0) <= LOOP_WARNING_DISTANCE:
                        loop_detected = True

    stage, stage_guidance = _termination_policy_for_round(current_round)

    if stage == "forced":
        logger.info("[skynet] Forced close at round %s.", current_round)
        return {"subtopic_exhausted": True}

    decision_prompt = _build_termination_vote_prompt(
        topic_summary=topic["summary"],
        topic_detail=ctx + historical_context,
        stage_guidance=stage_guidance,
        topic_id=topic.get("id", 0),
    )
    if not _should_run_termination_vote(current_round):
        logger.info(
            "[skynet] Skipping termination vote at round %s because early rounds are for stance-taking and evidence gathering.",
            current_round,
        )
        is_done = False
    elif _pending_extra_turns(state):
        logger.info("[skynet] Deferring close because extra turns are still pending.")
        is_done = False
    else:
        try:
            vote_records = await _run_termination_votes(
                voters=voting_agents(),
                prompt=decision_prompt,
                topic_id=state["topic_id"],
                subtopic_id=state["subtopic_id"],
                round_number=current_round,
                subject=subtopic["summary"],
            )
            aggregation = _aggregate_termination_votes(vote_records, current_round)
            logger.info(
                "[skynet] Termination aggregation round=%s valid_votes=%s invalid_votes=%s close_votes=%s close_ratio=%.2f blocker_counts=%s blocked_by=%s subtopic_exhausted=%s",
                current_round,
                aggregation["valid_votes"],
                aggregation["invalid_votes"],
                aggregation["close_votes"],
                aggregation["close_ratio"],
                aggregation["blocker_counts"],
                aggregation["blocked_by"],
                aggregation["subtopic_exhausted"],
            )
            if aggregation["invalid_votes"] > TERMINATION_MAX_INVALID_VOTES:
                logger.warning(
                    "[skynet] Termination vote degraded open because %s governance votes were invalid.",
                    aggregation["invalid_votes"],
                )
            is_done = aggregation["subtopic_exhausted"]
        except Exception as exc:
            logger.warning(
                "[skynet] Termination vote degraded open after vote failure: %s", exc
            )
            is_done = False
    warning_text = None

    if loop_detected and not is_done:
        if not warning_text:
            warning_text = "System warning: this workflow is revisiting prior conclusions. Bring new evidence, a narrower unresolved claim, or a different assumption next round."
        api.post_message(
            state["topic_id"],
            state["subtopic_id"],
            SKYNET,
            warning_text,
            msg_type="warning",
            round_number=current_round,
            turn_kind=AUDIENCE_WARNING_TURN,
        )

    # Shift fact counters for circuit breaker
    if is_done:
        analytics.capture(
            f"topic_{state['topic_id']}",
            "round_terminated",
            {
                "round_number": current_round,
                "subtopic_id": state.get("subtopic_id"),
                "close_reason": state.get("close_reason"),
            },
        )
    return {
        "subtopic_exhausted": is_done,
        "fact_count_2_rounds_ago": prev_1,
        "fact_count_1_round_ago": current_fact_count,
    }


def route_after_round(state: ChatState) -> str:
    """Decides what happens at the end of a round."""
    if state.get("subtopic_exhausted"):
        return "close_subtopic"
    return "setup_next_round"


def setup_next_round_node(state: ChatState) -> dict:
    """Prepares the next round using stage-based execution plan."""
    # Phase C: JTMS sweep at round boundary
    try:
        from .jtms import jtms_sweep

        current_round = state.get("round_number", 1)
        changes = jtms_sweep(state["topic_id"], current_round)
        if changes:
            logger.info(
                "[jtms] %d state changes at round %d boundary",
                len(changes),
                current_round,
            )
    except Exception as exc:
        logger.warning("[jtms] Sweep failed at round boundary: %s", exc)

    next_round = state.get("round_number", 1) + 1
    if _is_mse_modeling_state(state):
        _advance_mse_workflow_deterministically(state)
    phase = (
        build_mse_stage_for_state(state)[0]
        if _is_mse_modeling_state(state)
        else get_phase_for_round(next_round)
    )
    pending_stages = build_stages_for_round(next_round, state)
    # Also keep pending_turns for backward compatibility with recovery logic
    _, pending_turns = build_turn_queue_for_round(state, next_round)
    return {
        "pending_turns": pending_turns,
        "pending_stages": pending_stages,
        "phase": phase,
        "round_number": next_round,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "spectator_target": state.get("spectator_target"),
        "spectator_web_boost_target": state.get("spectator_web_boost_target"),
        "pending_fact_reviews_remaining": False,
    }


def close_subtopic_node(state: ChatState) -> dict:
    logger.info("Subtopic exhausted; returning control to topic graph.")
    return {}


async def writer_node(state: ChatState) -> dict:
    return await _run_writer_critique_pass(state)


async def final_writer_node(state: ChatState) -> dict:
    logger.info(
        "[writer] Final writer critique skipped because close-path work is harvest-only."
    )
    return {}


async def fact_proposer_node(state: ChatState) -> dict:
    return await _run_fact_proposer_pass(state, force=False)


async def final_fact_proposer_node(state: ChatState) -> dict:
    return await _run_fact_proposer_pass(state, force=True)


async def librarian_node(state: ChatState) -> dict:
    return await _run_librarian_pass(state)


async def final_librarian_node(state: ChatState) -> dict:
    await _run_librarian_pass(state)
    pending_candidates = api.get_pending_fact_candidates(
        state["topic_id"], state["subtopic_id"]
    )
    pending_claims = api.get_pending_claim_candidates(
        state["topic_id"], state["subtopic_id"]
    )
    reentry_count = state.get("final_librarian_reentry_count", 0)
    if pending_candidates or pending_claims:
        logger.warning(
            "[librarian] %s fact candidates and %s claim candidates remain pending; delaying subtopic close for another round.",
            len(pending_candidates),
            len(pending_claims),
        )
        return {
            "pending_fact_reviews_remaining": True,
            "subtopic_exhausted": False,
            "final_librarian_reentry_count": reentry_count + 1,
        }
    return {"pending_fact_reviews_remaining": False}


def route_after_final_librarian(state: ChatState) -> str:
    reentry_count = state.get("final_librarian_reentry_count", 0)
    if state.get("pending_fact_reviews_remaining") and reentry_count < 2:
        return "setup_next_round"
    return "close_subtopic"


async def parallel_group_node(state: ChatState) -> dict:
    """Run all agents in the current stage concurrently, persist results in roster order."""
    stage = state.get("current_stage")
    if not stage:
        logger.error("[parallel] parallel_group_node called with no current_stage")
        return {"current_stage": None}
    agents = stage["agents"]
    logger.info(
        "[parallel] Running %d agents concurrently: %s",
        len(agents),
        [t["actor"] for t in agents],
    )

    results = await asyncio.gather(
        *[
            _run_single_agent_turn(state, t["actor"], t.get("turn_kind", BASE_TURN))
            for t in agents
        ],
        return_exceptions=True,
    )

    collected_targets: dict = {}
    _pending_followups: list[tuple[str, list[int]]] = []
    for turn, result in zip(agents, results):
        if isinstance(result, BaseException):
            logger.warning("[parallel] %s failed: %s", turn["actor"], result)
            continue
        if result.get("no_topic"):
            continue
        # Handle spectator (no persistence)
        if result.get("spectator_data"):
            parsed_focus = result["spectator_data"]
            if parsed_focus["parsed_ok"]:
                collected_targets["spectator_target"] = parsed_focus["target"]
                collected_targets["spectator_web_boost_target"] = (
                    parsed_focus["target"] if parsed_focus["grant_web_search"] else None
                )
            continue

        try:
            code_evidence_ids = await _persist_agent_result(state, result)
        except Exception as exc:
            logger.error(
                "[parallel] Failed to persist %s result: %s", turn["actor"], exc
            )
            continue

        # Collect code evidence for follow-up turns
        if code_evidence_ids and turn.get("turn_kind", BASE_TURN) == BASE_TURN:
            _pending_followups.append((turn["actor"], code_evidence_ids))

        if result.get("targets"):
            collected_targets.update(result["targets"])

    # Schedule code follow-up turns for agents that produced code evidence
    for followup_actor, eids in _pending_followups:
        try:
            followup_ctx = _build_code_followup_context(eids)
            followup_state = {**dict(state), "_code_followup_context": followup_ctx}
            followup_result = await _run_single_agent_turn(
                followup_state, followup_actor, CODE_FOLLOWUP_TURN
            )
            if not followup_result.get("no_topic"):
                await _persist_agent_result(followup_state, followup_result)
        except Exception as exc:
            logger.warning(
                "[parallel] Code follow-up for %s failed: %s", followup_actor, exc
            )

    updates: dict = {"current_stage": None}
    # Apply target state
    for key in (
        "dog_target",
        "cat_target",
        "tron_target",
        "spectator_target",
        "spectator_web_boost_target",
    ):
        if key in collected_targets:
            updates[key] = collected_targets[key]

    # Build intervention stages from extracted targets
    intervention_turns = _build_intervention_turns(collected_targets)
    if intervention_turns:
        remaining = list(state.get("pending_stages", []))
        remaining.insert(0, {"agents": intervention_turns, "parallel": True})
        updates["pending_stages"] = remaining

    return updates


async def sequential_group_node(state: ChatState) -> dict:
    """Run agents sequentially (R3+ analysis). Interventions fire immediately after each base turn."""
    stage = state.get("current_stage")
    if not stage:
        logger.error("[sequential] sequential_group_node called with no current_stage")
        return {"current_stage": None}
    agents = stage["agents"]
    logger.info(
        "[sequential] Running %d agents sequentially: %s",
        len(agents),
        [t["actor"] for t in agents],
    )

    updates: dict = {"current_stage": None}
    working_state = dict(state)
    for turn in agents:
        actor = turn["actor"]
        turn_kind = turn.get("turn_kind", BASE_TURN)
        try:
            result = await _run_single_agent_turn(working_state, actor, turn_kind)
        except Exception as exc:
            logger.warning("[sequential] %s failed: %s", actor, exc)
            continue
        if result.get("no_topic"):
            continue
        if result.get("spectator_data"):
            parsed_focus = result["spectator_data"]
            if parsed_focus["parsed_ok"]:
                working_state["spectator_target"] = parsed_focus["target"]
                working_state["spectator_web_boost_target"] = (
                    parsed_focus["target"] if parsed_focus["grant_web_search"] else None
                )
                updates["spectator_target"] = working_state["spectator_target"]
                updates["spectator_web_boost_target"] = working_state[
                    "spectator_web_boost_target"
                ]
            continue

        try:
            code_evidence_ids = await _persist_agent_result(working_state, result)
        except Exception as exc:
            logger.error("[sequential] Failed to persist %s result: %s", actor, exc)
            continue

        # Code follow-up turn: let the agent interpret its own results
        if code_evidence_ids and turn_kind == BASE_TURN:
            try:
                followup_ctx = _build_code_followup_context(code_evidence_ids)
                followup_state = {
                    **working_state,
                    "_code_followup_context": followup_ctx,
                }
                followup_result = await _run_single_agent_turn(
                    followup_state, actor, CODE_FOLLOWUP_TURN
                )
                if not followup_result.get("no_topic"):
                    await _persist_agent_result(followup_state, followup_result)
            except Exception as exc:
                logger.warning(
                    "[sequential] Code follow-up for %s failed: %s", actor, exc
                )

        # Schedule interventions immediately (before next base turn)
        if result.get("targets"):
            intervention_turns = _build_intervention_turns(result["targets"])
            for iturn in intervention_turns:
                try:
                    iresult = await _run_single_agent_turn(
                        working_state, iturn["actor"], iturn["turn_kind"]
                    )
                    if not iresult.get("no_topic") and not iresult.get(
                        "spectator_data"
                    ):
                        await _persist_agent_result(working_state, iresult)
                except Exception as exc:
                    logger.warning(
                        "[sequential] Intervention %s/%s failed: %s",
                        iturn["actor"],
                        iturn["turn_kind"],
                        exc,
                    )

    return updates


async def drain_daemon_node(state: ChatState) -> dict:
    """Drain the fact daemon before closing a subtopic."""
    from .fact_daemon import drain_daemon

    await drain_daemon(state["topic_id"], state["subtopic_id"], timeout=90.0)
    return {}


def build_graph():
    builder = StateGraph(ChatState)

    # 1. Stage-based Expert Loop
    builder.add_node("bootstrap_fact_intake_node", bootstrap_fact_intake_node)
    builder.add_node("stage_dispatcher", stage_dispatcher_node)
    builder.add_node("parallel_group", parallel_group_node)
    builder.add_node("sequential_group", sequential_group_node)

    builder.add_conditional_edges(
        "stage_dispatcher",
        route_from_stage_dispatcher,
        {
            "parallel_group": "parallel_group",
            "sequential_group": "sequential_group",
            "end_of_round": "writer_node",
        },
    )
    builder.add_edge("parallel_group", "stage_dispatcher")
    builder.add_edge("sequential_group", "stage_dispatcher")

    # 2. End of Round Logic
    builder.add_node("writer_node", writer_node)
    builder.add_node("audience_summary_node", audience_summary_node)
    builder.add_node("audience_termination_check_node", audience_termination_check_node)
    builder.add_node("setup_next_round_node", setup_next_round_node)
    builder.add_node("drain_daemon_node", drain_daemon_node)
    builder.add_node("close_subtopic_node", close_subtopic_node)

    builder.add_edge("writer_node", "audience_summary_node")
    builder.add_edge("audience_summary_node", "audience_termination_check_node")

    builder.add_conditional_edges(
        "audience_termination_check_node",
        route_after_round,
        {
            "close_subtopic": "drain_daemon_node",
            "setup_next_round": "setup_next_round_node",
        },
    )

    builder.add_edge("setup_next_round_node", "stage_dispatcher")
    builder.add_edge("drain_daemon_node", "close_subtopic_node")
    builder.add_edge("close_subtopic_node", END)

    # Entry point
    builder.add_edge(START, "bootstrap_fact_intake_node")
    builder.add_edge("bootstrap_fact_intake_node", "stage_dispatcher")

    return builder.compile()


async def run_subtopic_graph(topic_id: int, subtopic_id: int, plan_id: int = 0):
    graph = build_graph()

    # Attempt to recover last state from DB
    messages = api.get_messages(topic_id, subtopic_id=subtopic_id, limit=100)
    initial_round = 1
    spoken_this_round = set()
    last_writer_round = None
    last_fact_proposer_round = None
    last_final_fact_proposer_round = None
    last_summary_round = None

    if messages:
        # Filter for standard messages that have a round number
        round_msgs = [
            m
            for m in messages
            if m.get("round_number") is not None and m.get("msg_type") == "standard"
        ]
        if round_msgs:
            last_msg = round_msgs[-1]
            try:
                initial_round = int(last_msg["round_number"])
            except (ValueError, TypeError):
                initial_round = 1

            # Find all (sender, turn_kind) pairs in the current recovered round
            # Fix: m.get("turn_kind") or BASE_TURN to correctly match "base" if NULL in DB
            spoken_this_round = set()
            for m in round_msgs:
                try:
                    r_num = int(m.get("round_number") or 0)
                    if r_num == initial_round:
                        spoken_this_round.add(
                            (m["sender"], m.get("turn_kind") or BASE_TURN)
                        )
                except (ValueError, TypeError):
                    continue
            logger.info(
                "[Server] Recovered subtopic state: Round %s, already spoken: %s",
                initial_round,
                spoken_this_round,
            )

        # Recover last round markers for NPC nodes to avoid duplicates.
        # Since messages are from api.get_messages (id ASC), reversed(messages) gives us the latest first.
        for m in reversed(messages):
            try:
                r = (
                    int(m.get("round_number") or 0)
                    if m.get("round_number") is not None
                    else None
                )
            except (ValueError, TypeError):
                r = None

            t = m.get("turn_kind")
            if r is None:
                continue

            if t == WRITER_CRITIQUE_TURN and last_writer_round is None:
                last_writer_round = r
            if m["sender"] == "fact_proposer" and last_fact_proposer_round is None:
                last_fact_proposer_round = r
            if t == AUDIENCE_SUMMARY_TURN and last_summary_round is None:
                last_summary_round = r

            # Final fact proposer usually happens at the end of subtopic (Close)
            # but we check for any messages from it
            if (
                m["sender"] == "fact_proposer"
                and m.get("msg_type") == "summary"
                and last_final_fact_proposer_round is None
            ):
                last_final_fact_proposer_round = r

    seed_state = {
        "topic_id": topic_id,
        "subtopic_id": subtopic_id,
        "round_number": initial_round,
    }
    if _topic_is_mse_modeling(topic_id):
        _ensure_mse_problem_seed(seed_state)
        _advance_mse_workflow_deterministically(seed_state)
    phase = (
        build_mse_stage_for_state(seed_state)[0]
        if _topic_is_mse_modeling(topic_id)
        else get_phase_for_round(initial_round)
    )
    _, pending_turns = build_turn_queue_for_round(seed_state, initial_round)
    pending_stages = build_stages_for_round(initial_round, seed_state)

    # Filter pending stages/turns to exclude agents who already spoke this round
    if spoken_this_round:
        pending_turns = [
            t
            for t in pending_turns
            if (t["actor"], t.get("turn_kind") or BASE_TURN) not in spoken_this_round
        ]
        filtered_stages = []
        for stage in pending_stages:
            filtered_agents = [
                t
                for t in stage["agents"]
                if (t["actor"], t.get("turn_kind") or BASE_TURN)
                not in spoken_this_round
            ]
            if filtered_agents:
                filtered_stages.append(
                    {"agents": filtered_agents, "parallel": stage["parallel"]}
                )
        pending_stages = filtered_stages
        if not pending_turns and not pending_stages:
            logger.info(
                "[Server] All agents in Round %s have spoken. Proceeding to end-of-round.",
                initial_round,
            )

    from .fact_daemon import start_daemon, stop_daemon, drain_daemon

    initial_state = {
        "topic_id": topic_id,
        "plan_id": plan_id,
        "subtopic_id": subtopic_id,
        "pending_subtopics": [],
        "pending_turns": pending_turns,
        "pending_stages": pending_stages,
        "current_actor": "",
        "current_turn_kind": "",
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "spectator_target": None,
        "spectator_web_boost_target": None,
        "phase": phase,
        "subtopic_exhausted": False,
        "round_number": initial_round,
        "last_writer_round": last_writer_round,
        "last_fact_proposer_round": last_fact_proposer_round,
        "last_final_fact_proposer_round": last_final_fact_proposer_round,
        "last_summary_round": last_summary_round,
        "pending_fact_reviews_remaining": False,
        "fact_count_1_round_ago": (
            api.count_facts(topic_id, subtopic_id=subtopic_id)
            if initial_round > 1
            else 0
        ),
        "fact_count_2_rounds_ago": (
            api.count_facts(topic_id, subtopic_id=subtopic_id)
            if initial_round > 1
            else 0
        ),
        "active_evidence_gaps": [],
        "gap_search_active": False,
        "gap_search_directive": None,
        "close_reason": None,
    }
    if _topic_is_mse_modeling(topic_id):
        return await graph.ainvoke(initial_state)

    try:
        await start_daemon(topic_id, subtopic_id)
        return await graph.ainvoke(initial_state)
    finally:
        try:
            await drain_daemon(topic_id, subtopic_id, timeout=30.0)
        except Exception:
            await stop_daemon(topic_id, subtopic_id)


async def run_server_loop():
    db.init_db()
    from .master_graph import build_master_graph

    # Try to set up code execution sandbox (non-fatal if Docker unavailable)
    try:
        sandbox_ok = await asyncio.to_thread(ensure_sandbox)
        if sandbox_ok:
            logger.info("[CodeSandbox] Sandbox ready.")
        else:
            logger.warning(
                "[CodeSandbox] Sandbox unavailable; code execution disabled."
            )
    except Exception as exc:
        logger.warning(
            "[CodeSandbox] Sandbox setup failed: %s; code execution disabled.", exc
        )

    master_graph = build_master_graph()
    try:
        while True:
            topic = api.get_current_topic()

            # Active topic: run orchestration
            if topic and topic["status"] in ("Started", "Running"):
                logger.info("Triggering topic-level orchestration graph...")
                result = await master_graph.ainvoke(
                    {"topic_id": topic["id"], "topic_complete": False}
                )
                if result.get("deferred"):
                    logger.info(
                        "Topic orchestration deferred; backing off before retry."
                    )
                    await asyncio.sleep(10)
                continue

            # Paused topic: wait for user to resume
            if topic and topic["status"] == "Paused":
                logger.info("Topic is Paused. Sleeping.")
                await asyncio.sleep(10)
                continue

            # No active topic: check queue for next topic (F.5)
            next_topic = db.start_next_queued_topic()
            if next_topic:
                logger.info("Auto-starting queued topic %d", next_topic["id"])
                continue

            logger.info("Room is idle. Sleeping.")
            await asyncio.sleep(10)
    finally:
        await shutdown_broker()


if __name__ == "__main__":
    configure_logging()
    asyncio.run(run_server_loop())
