"""Background fact extraction and librarian review daemon."""

import asyncio
import json
import logging
import re
import sqlite3
from typing import Optional

from . import api
from . import topic_config
from .agents import SKYNET, SPECTATOR
from .broker import DEFAULT_MAX_TOKENS, call_text
from .embedding import aget_embedding
from .librarian_processor import (
    apply_claim_review,
    apply_librarian_review,
    parse_claim_review,
    parse_librarian_review,
)
from .prompts import PROMPTS
from .rag import build_query_rag_context
from .structured_retry import generate_summary, usable_text_output
from .writer_processor import process_clerk_claim_output

logger = logging.getLogger(__name__)

_NUMERICAL_CONTENT_RE = re.compile(r"\d+\.\d+|\d{1,3}(?:,\d{3})+")


def _message_has_numerical_content(content: str) -> bool:
    """Fast regex gate: True if message contains any number."""
    return bool(_NUMERICAL_CONTENT_RE.search(content or ""))


# Re-use constants from server module -- import lazily to avoid circular deps
_INLINE_FACT_LIMIT = 1
_CLAIM_LIMIT = 2


def _topic_provider(
    topic_id: int, key: str, fallback_key: str = ""
) -> str:
    try:
        return topic_config.get_provider_profile_for(
            topic_id, key, fallback_key=fallback_key
        )
    except sqlite3.OperationalError:
        raise
    except Exception as exc:
        logger.debug("[daemon] Provider lookup failed for %s: %s", key, exc)
        return "minimax"


class FactDaemon:
    """Continuously polls for new messages and pending candidates,
    extracting and reviewing facts in the background."""

    def __init__(self, topic_id: int, subtopic_id: int):
        self.topic_id = topic_id
        self.subtopic_id = subtopic_id
        self.current_round: int = 0
        self._shutdown = asyncio.Event()
        self._drain = asyncio.Event()
        self._drain_complete = asyncio.Event()
        self._clerk_done = asyncio.Event()
        self._evidence_done = asyncio.Event()
        self._clerk_task: Optional[asyncio.Task] = None
        self._librarian_task: Optional[asyncio.Task] = None
        self._evidence_task: Optional[asyncio.Task] = None

    async def start(self):
        self._clerk_task = asyncio.create_task(
            self._clerk_loop(),
            name=f"fact-clerk-{self.topic_id}-{self.subtopic_id}",
        )
        self._librarian_task = asyncio.create_task(
            self._librarian_loop(),
            name=f"fact-librarian-{self.topic_id}-{self.subtopic_id}",
        )
        self._evidence_task = asyncio.create_task(
            self._evidence_loop(),
            name=f"fact-evidence-{self.topic_id}-{self.subtopic_id}",
        )
        logger.info(
            "[daemon] Started fact daemon for topic=%s subtopic=%s",
            self.topic_id,
            self.subtopic_id,
        )

    async def _clerk_loop(self):
        """Poll for new standard messages, extract FactCandidates + ClaimCandidates."""
        from .minimax_client import is_daemon_channel

        is_daemon_channel.set(True)
        last_processed_id = 0
        while not self._shutdown.is_set():
            try:
                new_msgs = api.get_messages_since(
                    self.topic_id,
                    self.subtopic_id,
                    last_processed_id,
                    "standard",
                )
                for msg in new_msgs:
                    self.current_round = max(
                        self.current_round, msg.get("round_number") or 0
                    )
                    # G.4: Skip messages that already have a formal claim
                    if msg.get("has_formal_claim"):
                        last_processed_id = max(last_processed_id, msg["id"])
                        continue
                    try:
                        await self._extract_candidates(msg)
                        last_processed_id = max(last_processed_id, msg["id"])
                    except Exception as exc:
                        logger.warning(
                            "[daemon] Extraction failed for msg %d, will retry: %s",
                            msg["id"],
                            exc,
                        )
                if self._drain.is_set() and not new_msgs:
                    break
            except Exception as exc:
                logger.warning("[daemon] Clerk loop error: %s", exc)
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=5.0)
                break  # shutdown was set
            except asyncio.TimeoutError:
                pass  # just a sleep substitute
        self._clerk_done.set()

    async def _librarian_loop(self):
        """Poll for pending candidates, review them."""
        from .minimax_client import is_daemon_channel

        is_daemon_channel.set(True)
        while not self._shutdown.is_set():
            try:
                pending = api.get_pending_fact_candidates(
                    self.topic_id, self.subtopic_id
                )
                pending_claims = api.get_pending_claim_candidates(
                    self.topic_id, self.subtopic_id
                )
                if pending or pending_claims:
                    await self._review_batch(pending, pending_claims)
                elif self._drain.is_set() and self._clerk_done.is_set():
                    # Final sweep after clerk is done to catch any late candidates
                    pending = api.get_pending_fact_candidates(
                        self.topic_id, self.subtopic_id
                    )
                    pending_claims = api.get_pending_claim_candidates(
                        self.topic_id, self.subtopic_id
                    )
                    if pending or pending_claims:
                        try:
                            await self._review_batch(pending, pending_claims)
                        except Exception as exc:
                            logger.warning("[daemon] Final drain sweep failed: %s", exc)
                    self._drain_complete.set()
                    break
            except Exception as exc:
                logger.warning("[daemon] Librarian loop error: %s", exc)
                if self._drain.is_set() and self._clerk_done.is_set():
                    logger.warning(
                        "[daemon] Review failed during drain; completing drain anyway"
                    )
                    self._drain_complete.set()
                    break
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=8.0)
                break
            except asyncio.TimeoutError:
                pass

    async def _evidence_loop(self):
        """Poll for unprocessed web evidence and extract into Ledger + Facts."""
        from .minimax_client import is_daemon_channel
        from .evidence_parser import (
            extract_all_from_evidence,
            try_promote_pending_entries,
            backfill_incomplete_ledger_entries,
        )

        is_daemon_channel.set(True)
        while not self._shutdown.is_set():
            try:
                batch = api.get_unprocessed_web_evidence(self.topic_id, limit=5)
                for row in batch:
                    # Single unified pass: Ledger + FactCandidates
                    try:
                        await extract_all_from_evidence(
                            self.topic_id,
                            row.get("origin_subtopic_id") or self.subtopic_id,
                            row,
                            current_round=self.current_round,
                        )
                    except Exception as exc:
                        logger.warning(
                            "[daemon] Unified extraction from W%s failed: %s",
                            row.get("id"),
                            exc,
                        )
                if batch:
                    try_promote_pending_entries(
                        self.topic_id, current_round=self.current_round
                    )
                    # Backfill incomplete Ledger entries (missing unit/time)
                    try:
                        await backfill_incomplete_ledger_entries(
                            self.topic_id, self.current_round
                        )
                    except Exception as exc:
                        logger.debug("[daemon] Ledger backfill error: %s", exc)
                if self._drain.is_set() and not batch:
                    break
            except Exception as exc:
                logger.warning("[daemon] Evidence loop error: %s", exc)
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=8.0)
                break
            except asyncio.TimeoutError:
                pass
        # Evidence sentinel: warn if no web evidence exists for this topic
        try:
            from . import db as _db

            count = _db.get_web_evidence_count(self.topic_id)
            if count == 0:
                logger.error(
                    "[daemon] Evidence sentinel: topic=%s has ZERO web evidence rows at drain",
                    self.topic_id,
                )
        except Exception:
            pass
        self._evidence_done.set()

    async def drain_and_stop(self, timeout: float = 90.0):
        """Drain all pending work, then stop."""
        self._drain.set()
        try:
            await asyncio.wait_for(
                asyncio.gather(self._drain_complete.wait(), self._evidence_done.wait()),
                timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("[daemon] Drain timed out after %.0fs", timeout)
        self._shutdown.set()
        # Wait for tasks to finish
        tasks = [
            t
            for t in (self._clerk_task, self._librarian_task, self._evidence_task)
            if t
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(
            "[daemon] Stopped fact daemon for topic=%s subtopic=%s",
            self.topic_id,
            self.subtopic_id,
        )

    async def stop(self):
        """Immediately stop without draining."""
        self._shutdown.set()
        tasks = [
            t
            for t in (self._clerk_task, self._librarian_task, self._evidence_task)
            if t
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _extract_ledger_entries(
        self, msg: dict, entry_type: str = "web_evidence"
    ):
        """Clerk structured extraction: parse message into Ledger entries."""
        from .server import (
            build_clerk_ledger_extraction_prompt,
            parse_clerk_ledger_output,
        )
        from . import ledger as _ledger

        sender = msg.get("sender", "")
        content = msg.get("content", "")
        round_number = msg.get("round_number") or 0

        # Fetch entity/attribute lists once — used for both prompt and parsing
        entities = _ledger.get_entity_numbered_list(self.topic_id, round_number)
        attributes = _ledger.get_attribute_numbered_list(self.topic_id)

        prompt = build_clerk_ledger_extraction_prompt(
            self.topic_id,
            content,
            sender,
            round_number,
            entities=entities,
            attributes=attributes,
        )
        if prompt is None:
            return

        try:
            resp = await call_text(
                prompt,
                provider=_topic_provider(self.topic_id, "fact_provider"),
                strategy="direct",
                allow_web=False,
                system_instruction="You are a data extraction clerk. Extract structured data points from messages.",
                fallback_role="fact_proposer",
            )
        except Exception as exc:
            logger.warning("[daemon] Clerk ledger extraction LLM call failed: %s", exc)
            return

        if not resp:
            return

        parsed = parse_clerk_ledger_output(
            resp,
            entities,
            attributes,
            self.topic_id,
            self.subtopic_id,
            round_number,
            sender,
        )

        for entry in parsed:
            if entry.get("type") == "qualitative":
                # Feed qualitative entries into the existing fact pipeline
                logger.info(
                    "[daemon] Clerk qualitative fact: %s", entry.get("text", "")[:80]
                )
                continue
            if entry.get("type") == "structured":
                try:
                    _ledger.normalize_and_upsert(
                        topic_id=entry["topic_id"],
                        subtopic_id=entry.get("subtopic_id"),
                        entity_id=entry.get("entity_id"),
                        attribute_id=entry.get("attribute_id"),
                        raw_value=entry.get("raw_value", ""),
                        raw_timeframe=entry.get("raw_timeframe"),
                        entry_type=entry_type,
                        source_ref=entry.get("source_ref", ""),
                        created_by=entry.get("created_by"),
                        current_round=entry.get("current_round"),
                        valid_from=entry.get("valid_from"),
                        valid_to=entry.get("valid_to"),
                        min_val=entry.get("min_val"),
                        max_val=entry.get("max_val"),
                        unit=entry.get("unit"),
                        stat_type=entry.get("stat_type"),
                        value_mean=entry.get("value_mean"),
                        value_std=entry.get("value_std"),
                        value_p=entry.get("value_p"),
                        value_n=entry.get("value_n"),
                        value_ci_lower=entry.get("value_ci_lower"),
                        value_ci_upper=entry.get("value_ci_upper"),
                        value_ci_level=entry.get("value_ci_level"),
                        baseline_entity_id=entry.get("baseline_entity_id"),
                        split=entry.get("split"),
                        config_json=entry.get("config_json"),
                    )
                except Exception as exc:
                    logger.warning("[daemon] Clerk ledger upsert failed: %s", exc)

        logger.info(
            "[daemon] Clerk extracted %d entries from msg=%s sender=%s",
            len(parsed),
            msg.get("id"),
            sender,
        )

    async def _extract_candidates(self, msg: dict):
        """Extract fact and claim candidates from a single message."""
        sender = msg.get("sender", "")
        content = msg.get("content", "")

        # Synthesized conclusions path: skynet/writer summaries get Clerk extraction
        if sender in {SKYNET, "writer"} and msg.get("msg_type") == "summary":
            return
        if not content or sender in {SKYNET, "writer", "librarian", "fact_proposer"}:
            return
        if sender == SPECTATOR:
            return

        # Agent claims go to Fact/Claim tables, not Ledger (entry-type gate also enforces this)

        # Import server functions lazily to avoid circular imports
        from .server import (
            _normalize_clerk_claim_candidates_contract,
            _claim_candidates_output_is_usable,
            _extract_fact_ids_from_text,
            _render_fact_lookup_context,
            _call_text_with_structured_retry,
            FACT_CITATION_PROTOCOL,
        )

        # Claim candidates from cited facts
        cited_fact_ids = sorted(set(_extract_fact_ids_from_text(content)))
        if cited_fact_ids:
            support_facts = api.get_facts_by_ids(self.topic_id, cited_fact_ids)
            if support_facts:
                claim_prompt = (
                    f"Topic context:\n{FACT_CITATION_PROTOCOL}\n"
                    f"Message from {sender}:\n{content}\n"
                    f"=== VERIFIED FACTS REFERENCED ===\n"
                    f"{_render_fact_lookup_context(support_facts)}\n"
                    f"\nTASK: Extract at most {_CLAIM_LIMIT} atomic derived claim "
                    "candidates that are explicitly supported by cited "
                    "facts [F...]. "
                    "CRITICAL: Claims MUST be atomic. Never combine with 'but', 'however', 'and'. "
                    "Each claim MUST cite at least one [F...] or [L...]. "
                    "ENTITY NAMING: Always use full official names, never abbreviations "
                    "(e.g. 'Federal Reserve' not 'Fed', 'People's Bank of China' not 'PBOC'). "
                    "Reply with strict JSON only: "
                    '{"action":"propose_claim_candidates",'
                    '"claim_candidates":[{"candidate_text":"...",'
                    '"support_fact_ids_json":[1,2],'
                    '"rationale_short":"..."}]}.'
                )
                claim_text = await _call_text_with_structured_retry(
                    stage_name=f"Daemon claim pass msg={msg['id']}",
                    validator=_claim_candidates_output_is_usable,
                    invoke=lambda _p=claim_prompt: call_text(
                        _p,
                        provider=_topic_provider(self.topic_id, "fact_provider"),
                        strategy="direct",
                        allow_web=False,
                        system_instruction=PROMPTS["fact_proposer"],
                        fallback_role="fact_proposer",
                        require_json=True,
                    ),
                )
                if claim_text:
                    parsed_claims = _normalize_clerk_claim_candidates_contract(
                        claim_text
                    )
                    if parsed_claims["parsed_ok"] and parsed_claims["claim_candidates"]:
                        await process_clerk_claim_output(
                            self.topic_id,
                            self.subtopic_id,
                            None,
                            parsed_claims["claim_candidates"],
                            max_candidates=_CLAIM_LIMIT,
                        )

    async def _review_batch(self, facts: list, claims: list):
        """Review pending fact and claim candidates."""
        from .server import (
            _query_librarian_review_text,
            build_librarian_prompt,
            build_claim_review_prompt,
            _extract_fact_ids_from_text,
        )

        topic = api.get_topic(self.topic_id)
        subtopic = api.get_subtopic(self.subtopic_id)
        if not topic or not subtopic:
            return

        messages = api.get_messages(
            self.topic_id, subtopic_id=self.subtopic_id, limit=12
        )
        recent_message_ids = [m["id"] for m in messages if "id" in m]

        for candidate in facts:
            try:
                rag_context, _ = await build_query_rag_context(
                    self.topic_id,
                    candidate["candidate_text"],
                    exclude_ids=recent_message_ids,
                )
                # Create a minimal state-like dict for the prompt builder
                fake_state = {
                    "round_number": 0,
                    "phase": "analysis",
                }
                prompt = build_librarian_prompt(
                    fake_state,
                    topic,
                    subtopic,
                    candidate,
                    messages,
                    rag_context,
                )

                resp_text, provider = await _query_librarian_review_text(
                    prompt,
                    stage_name=f"Daemon fact review {candidate['id']}",
                    validator=_librarian_response_is_usable,
                    topic_id=self.topic_id,
                    subtopic_id=self.subtopic_id,
                )
                try:
                    review = parse_librarian_review(
                        resp_text, candidate["candidate_text"]
                    )
                except ValueError:
                    fallback_provider = _topic_provider(
                        self.topic_id, "control_provider"
                    )
                    if provider == fallback_provider:
                        raise
                    logger.warning(
                        "[daemon] %s review for candidate %s failed; retrying with %s.",
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
                        topic_id=self.topic_id,
                        subtopic_id=self.subtopic_id,
                    )
                    review = parse_librarian_review(
                        resp_text, candidate["candidate_text"]
                    )
                result = await apply_librarian_review(self.topic_id, candidate, review)
                fact_id = result.get("accepted_fact_id")
                stored_text = result.get("stored_text")
                if fact_id and stored_text:
                    try:
                        summary = await generate_summary(stored_text)
                        if summary:
                            emb = await aget_embedding(summary)
                            if emb:
                                api.update_fact_summary_and_embedding(
                                    fact_id, summary, emb
                                )
                    except Exception as exc:
                        logger.warning(
                            "[daemon] Summary generation failed for " "fact %s: %s",
                            fact_id,
                            exc,
                        )
            except Exception as exc:
                logger.warning(
                    "[daemon] Failed to review candidate %s: %s",
                    candidate["id"],
                    exc,
                )

        for candidate in claims:
            try:
                support_ids = []
                try:
                    if isinstance(candidate.get("support_fact_ids_json"), str):
                        support_ids = [
                            int(item)
                            for item in json.loads(
                                candidate["support_fact_ids_json"] or "[]"
                            )
                        ]
                except Exception:
                    support_ids = (
                        _extract_fact_ids_from_text(
                            candidate.get("support_fact_ids_json", "")
                        )
                        if isinstance(candidate.get("support_fact_ids_json"), str)
                        else []
                    )

                support_facts = api.get_facts_by_ids(self.topic_id, support_ids)
                if not support_facts:
                    api.update_claim_candidate_review(
                        candidate["id"],
                        "reject",
                        review_note="No valid support facts were available "
                        "for review.",
                    )
                    continue

                rag_context, _ = await build_query_rag_context(
                    self.topic_id,
                    candidate["candidate_text"],
                    exclude_ids=recent_message_ids,
                )
                fake_state = {"round_number": 0, "phase": "analysis"}
                prompt = build_claim_review_prompt(
                    fake_state,
                    topic,
                    subtopic,
                    candidate,
                    messages,
                    support_facts,
                    rag_context,
                )

                resp_text, provider = await _query_librarian_review_text(
                    prompt,
                    stage_name=f"Daemon claim review {candidate['id']}",
                    validator=_librarian_response_is_usable,
                    topic_id=self.topic_id,
                    subtopic_id=self.subtopic_id,
                )
                try:
                    review = parse_claim_review(
                        resp_text,
                        candidate["candidate_text"],
                        support_ids,
                    )
                except ValueError:
                    fallback_provider = _topic_provider(
                        self.topic_id, "control_provider"
                    )
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
                        topic_id=self.topic_id,
                        subtopic_id=self.subtopic_id,
                    )
                    review = parse_claim_review(
                        resp_text,
                        candidate["candidate_text"],
                        support_ids,
                    )
                result = await apply_claim_review(self.topic_id, candidate, review)
                claim_id = result.get("accepted_claim_id")
                stored_text = result.get("stored_text")
                if claim_id and stored_text:
                    try:
                        summary = await generate_summary(stored_text, max_words=30)
                        if summary:
                            api.update_claim_summary(claim_id, summary)
                    except Exception as exc:
                        logger.warning(
                            "[daemon] Summary generation failed for " "claim %s: %s",
                            claim_id,
                            exc,
                        )
            except Exception as exc:
                logger.warning(
                    "[daemon] Failed to review claim candidate %s: %s",
                    candidate["id"],
                    exc,
                )

        # Phase C: JTMS sweep after librarian batch
        try:
            from .jtms import jtms_sweep

            changes = jtms_sweep(self.topic_id, self.current_round)
            if changes:
                logger.info(
                    "[jtms] %d state changes after librarian batch", len(changes)
                )
        except Exception as exc:
            logger.warning("[jtms] Sweep failed after librarian batch: %s", exc)


def _extract_json(text: str):
    """Lightweight JSON extraction for validation."""
    from .json_utils import extract_json_object

    return extract_json_object(text)


def _librarian_response_is_usable(text: str) -> bool:
    return usable_text_output(text) and bool(_extract_json(text))


# ---------------------------------------------------------------------------
# Module-level daemon registry
# ---------------------------------------------------------------------------

_active_daemons: dict[tuple[int, int], FactDaemon] = {}


def get_active_daemon(topic_id: int, subtopic_id: int) -> Optional[FactDaemon]:
    """Return the running daemon for a given topic/subtopic pair, if any."""
    return _active_daemons.get((topic_id, subtopic_id))


async def start_daemon(topic_id: int, subtopic_id: int) -> FactDaemon:
    """Create and start a new FactDaemon, stopping any existing one first."""
    key = (topic_id, subtopic_id)
    if key in _active_daemons:
        await _active_daemons[key].stop()
    daemon = FactDaemon(topic_id, subtopic_id)
    # Initialize round from most recent message
    daemon.current_round = api.get_max_round_number(topic_id, subtopic_id)
    _active_daemons[key] = daemon
    await daemon.start()
    return daemon


async def stop_daemon(topic_id: int, subtopic_id: int):
    """Immediately stop a running daemon."""
    key = (topic_id, subtopic_id)
    daemon = _active_daemons.pop(key, None)
    if daemon:
        await daemon.stop()


async def drain_daemon(topic_id: int, subtopic_id: int, timeout: float = 90.0):
    """Drain pending work and then stop the daemon."""
    key = (topic_id, subtopic_id)
    daemon = _active_daemons.get(key)
    if daemon:
        try:
            await daemon.drain_and_stop(timeout)
        finally:
            _active_daemons.pop(key, None)
