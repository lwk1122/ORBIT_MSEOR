"""Semantic deduplication for facts and claims.

Uses embedding recall (cosine >= threshold, top-K) + LLM judge to detect
same-finding-different-wording duplicates that slip past exact text matching.
"""

import json
import logging
import re
import sqlite3
from typing import Optional

from . import db as _db
from . import topic_config
from .broker import DEFAULT_MAX_TOKENS, call_text as _call_text
from .json_utils import extract_json_object as _extract_json

logger = logging.getLogger(__name__)


def _fact_provider(topic_id: int) -> str:
    try:
        return topic_config.get_provider_profile_for(topic_id, "fact_provider")
    except sqlite3.OperationalError:
        raise
    except Exception as exc:
        logger.debug("[fact-dedup] Fact provider lookup failed: %s", exc)
        return "minimax"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _l2_to_cosine(l2_dist: float) -> float:
    """Convert L2 distance (from sqlite-vec on unit-norm embeddings) to cosine similarity.

    For unit-norm vectors: L2^2 = 2 - 2*cos(theta)  =>  cos = 1 - L2^2/2
    """
    return max(-1.0, min(1.0, 1.0 - (l2_dist**2) / 2.0))


_FACT_JUDGE_SYSTEM = """\
You are a Fact Deduplication Judge.

Given a set of existing facts and one new fact, decide whether the new fact
is a DUPLICATE of any existing fact.

DUPLICATE CRITERIA (mark as duplicate):
- Same core information, even with different wording
- Subset or superset of an existing fact (mark the LESS informative one)
- Same statistic rephrased

NOT DUPLICATES (keep both):
- Different numbers, dates, or quantities
- Different entities or subjects
- Complementary facts that add distinct information

OUTPUT FORMAT:
- If duplicate: ["Mk"]
- If the new fact UPDATES/SUPERSEDES an existing fact (temporal replacement,
  e.g., old data revised with newer data): {"action":"UPDATE","key":"Mk"}
- If the new fact CONTRADICTS an existing fact (same time period,
  mutually exclusive claims): {"action":"CONTRADICTION","key":"Mk"}
- If not duplicate/update/contradiction: []
Return ONLY the JSON, nothing else."""

_CLAIM_JUDGE_SYSTEM = """\
You are a Claim Deduplication Judge.

Given a set of existing claims and one new claim, decide whether the new claim
is a DUPLICATE of or should be MERGED with any existing claim.

DUPLICATE: same claim, same supporting evidence -> skip insertion
MERGE: same claim assertion, but cites different supporting facts -> merge fact lists
CONTRADICTION: mutually exclusive claims about the same subject -> flag conflict
RESTATEMENT: claim merely restates or paraphrases its own support facts without adding \
new argument, synthesis, comparison, or inference -> reject as non-argumentative
DIFFERENT: genuinely different claims -> insert new

OUTPUT FORMAT (JSON object):
- Duplicate: {"action":"DUPLICATE","key":"Mk"}
- Merge: {"action":"MERGE","key":"Mk"}
- Contradiction: {"action":"CONTRADICTION","key":"Mk"}
- Restatement: {"action":"RESTATEMENT"}
- Different: {"action":"INSERT"}
Return ONLY the JSON object, nothing else."""


# ---------------------------------------------------------------------------
# Fact dedup
# ---------------------------------------------------------------------------


async def check_fact_duplicate(
    topic_id: int,
    new_fact_text: str,
    new_fact_embedding: list[float],
    top_k: int = 20,
    cosine_threshold: float = 0.80,
) -> tuple[str, Optional[int]]:
    """Check if *new_fact_text* is a semantic duplicate of an existing fact.

    Returns
    -------
    ("INSERT", None)              – no duplicate, proceed with insert
    ("DUPLICATE", fact_id)        – duplicate found, skip insert
    ("UPDATE", fact_id)           – temporal supersession detected
    ("CONTRADICTION", fact_id)    – same-period conflict detected
    """
    candidates = _db.search_facts(topic_id, new_fact_embedding, top_k=top_k)

    # Filter by cosine threshold
    similar: list[dict] = []
    for c in candidates:
        dist = c.get("distance")
        if dist is None:
            continue
        if _l2_to_cosine(dist) >= cosine_threshold:
            similar.append(c)

    if not similar:
        return ("INSERT", None)

    # Build lookup by ID
    similar_by_id = {c["id"]: c for c in similar}

    # Build dict for LLM judge
    existing_map = {f"M{c['id']}": c["content"] for c in similar}
    prompt = (
        f"existing = {json.dumps(existing_map, ensure_ascii=False)}\n"
        f"new_fact = {json.dumps(new_fact_text, ensure_ascii=False)}\n"
        f"Output:"
    )

    provider = _fact_provider(topic_id)
    try:
        resp = await _call_text(
            prompt,
            system_instruction=_FACT_JUDGE_SYSTEM,
            provider=provider,
            temperature=0.0,
            max_tokens=DEFAULT_MAX_TOKENS,
            fallback_role="librarian",
        )
    except Exception:
        logger.warning("[fact-dedup] LLM judge call failed; defaulting to INSERT")
        return ("INSERT", None)

    resp = resp.strip()

    # --- Try parsing as JSON ---
    parsed = None
    try:
        parsed = json.loads(resp)
    except json.JSONDecodeError:
        # Try to extract JSON array from response
        match = re.search(r"\[.*?\]", resp)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                pass
        if parsed is None:
            parsed = _extract_json(resp)

    # Handle dict response (UPDATE/CONTRADICTION)
    if isinstance(parsed, dict):
        action = str(parsed.get("action", "")).upper()
        key = str(parsed.get("key", ""))
        id_match = re.match(r"M(\d+)", key)
        if id_match and action in ("UPDATE", "CONTRADICTION"):
            matched_id = int(id_match.group(1))
            if matched_id in similar_by_id:
                return (action, matched_id)
        return ("INSERT", None)

    # Handle list response (DUPLICATE or INSERT)
    if isinstance(parsed, list):
        if not parsed:
            return ("INSERT", None)
        key = str(parsed[0])
        id_match = re.match(r"M(\d+)", key)
        if not id_match:
            return ("INSERT", None)
        matched_id = int(id_match.group(1))
        if matched_id not in similar_by_id:
            logger.debug(
                "[fact-dedup] LLM returned ID %d not in candidates", matched_id
            )
            return ("INSERT", None)
        return ("DUPLICATE", matched_id)

    return ("INSERT", None)


# ---------------------------------------------------------------------------
# Claim dedup
# ---------------------------------------------------------------------------


async def check_claim_duplicate(
    topic_id: int,
    new_claim_text: str,
    new_support_fact_ids: list[int],
    top_k: int = 20,
    support_fact_texts: list[str] | None = None,
) -> tuple[str, Optional[int]]:
    """Check if *new_claim_text* is a semantic duplicate/mergeable with existing claims.

    Returns
    -------
    ("INSERT", None)        – no duplicate
    ("DUPLICATE", claim_id) – same claim + same facts -> skip
    ("MERGE", claim_id)     – same claim + different facts -> merge support_fact_ids
    ("RESTATEMENT", None)   – claim merely restates support facts
    ("CONTRADICTION", claim_id) – mutually exclusive claims
    """
    # Fetch support fact texts for restatement detection
    if support_fact_texts is None and new_support_fact_ids:
        try:
            fact_rows = _db.get_facts_by_ids(topic_id, new_support_fact_ids[:5])
            support_fact_texts = [r["content"] for r in fact_rows if r.get("content")]
        except Exception:
            support_fact_texts = []

    # Use lexical search to find candidate claims (claims lack a vec table)
    claim_candidates = _db.search_claims_lexical(topic_id, new_claim_text, top_k=top_k)

    if not claim_candidates:
        # Even without existing claims, check for restatement
        if support_fact_texts:
            return await _check_restatement_only(
                topic_id, new_claim_text, support_fact_texts
            )
        return ("INSERT", None)

    # Build dict for LLM judge
    existing_map = {}
    for c in claim_candidates:
        try:
            support_ids = json.loads(c.get("support_fact_ids_json") or "[]")
        except (json.JSONDecodeError, TypeError):
            support_ids = []
        existing_map[f"M{c['id']}"] = {
            "claim": c["content"],
            "support_fact_ids": support_ids,
        }
    new_facts_str = json.dumps(sorted(new_support_fact_ids))
    fact_texts_block = ""
    if support_fact_texts:
        fact_texts_block = f"support_fact_texts = {json.dumps(support_fact_texts[:5], ensure_ascii=False)}\n"
    prompt = (
        f"existing_claims = {json.dumps(existing_map, ensure_ascii=False)}\n"
        f"new_claim = {json.dumps(new_claim_text, ensure_ascii=False)}\n"
        f"new_claim_support_fact_ids = {new_facts_str}\n"
        f"{fact_texts_block}"
        f"Output:"
    )

    provider = _fact_provider(topic_id)
    try:
        resp = await _call_text(
            prompt,
            system_instruction=_CLAIM_JUDGE_SYSTEM,
            provider=provider,
            temperature=0.0,
            max_tokens=DEFAULT_MAX_TOKENS,
            fallback_role="librarian",
        )
    except Exception:
        logger.warning("[claim-dedup] LLM judge call failed; defaulting to INSERT")
        return ("INSERT", None)

    parsed = _extract_json(resp.strip())
    if not isinstance(parsed, dict):
        return ("INSERT", None)

    action = str(parsed.get("action", "")).upper()
    if action == "RESTATEMENT":
        return ("RESTATEMENT", None)

    key = str(parsed.get("key", ""))
    id_match = re.match(r"M(\d+)", key)
    if not id_match:
        return ("INSERT", None)

    matched_id = int(id_match.group(1))
    candidate_ids = {c["id"] for c in claim_candidates}
    if matched_id not in candidate_ids:
        logger.debug("[claim-dedup] LLM returned ID %d not in candidates", matched_id)
        return ("INSERT", None)

    if action == "DUPLICATE":
        return ("DUPLICATE", matched_id)
    elif action == "MERGE":
        return ("MERGE", matched_id)
    elif action == "CONTRADICTION":
        return ("CONTRADICTION", matched_id)
    else:
        return ("INSERT", None)


async def _check_restatement_only(
    topic_id: int, claim_text: str, support_fact_texts: list[str]
) -> tuple[str, Optional[int]]:
    """Lightweight check: does the claim merely restate its support facts?"""
    prompt = (
        f"claim = {json.dumps(claim_text, ensure_ascii=False)}\n"
        f"support_fact_texts = {json.dumps(support_fact_texts[:5], ensure_ascii=False)}\n"
        "Does this claim merely restate/paraphrase its support facts without adding "
        "new argument, synthesis, comparison, or inference?\n"
        'Reply: {{"action":"RESTATEMENT"}} or {{"action":"INSERT"}}'
    )
    provider = _fact_provider(topic_id)
    try:
        resp = await _call_text(
            prompt,
            system_instruction=_CLAIM_JUDGE_SYSTEM,
            provider=provider,
            temperature=0.0,
            max_tokens=DEFAULT_MAX_TOKENS,
            fallback_role="librarian",
        )
    except Exception:
        return ("INSERT", None)
    parsed = _extract_json(resp.strip())
    if (
        isinstance(parsed, dict)
        and str(parsed.get("action", "")).upper() == "RESTATEMENT"
    ):
        return ("RESTATEMENT", None)
    return ("INSERT", None)
