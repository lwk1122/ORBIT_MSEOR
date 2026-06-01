import logging
import json
import re
import unicodedata
from typing import Iterable, List, Optional

from . import api, db

logger = logging.getLogger(__name__)


def _normalize_fact_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return " ".join(text.split())


def _extract_fact_lines(writer_text: str) -> List[str]:
    fact_lines = []
    for line in writer_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("VERIFIED:"):
            fact_lines.append(stripped[9:].strip())
        elif stripped.startswith("FACT:"):
            fact_lines.append(stripped[5:].strip())
    return fact_lines


async def _store_fact_candidates(
    topic_id: int,
    subtopic_id: int,
    writer_msg_id: Optional[int],
    facts: Iterable[str | dict],
    fact_stage: str = "synthesized",
    evidence_note: Optional[str] = None,
    round_number: int | None = None,
    max_candidates: int | None = None,
) -> list[int]:
    seen: set[str] = set()
    created_ids: list[int] = []
    for fact_item in facts:
        if isinstance(fact_item, dict):
            fact_content = (
                fact_item.get("candidate_text") or fact_item.get("text") or ""
            )
            summary = fact_item.get("summary")
            candidate_type = fact_item.get("candidate_type", "sourced_claim")
            source_refs_json = json.dumps(
                fact_item.get("source_refs", []), ensure_ascii=False
            )
            source_excerpt = fact_item.get("source_excerpt")
            verification_status = fact_item.get("verification_status")
            # Structured columns (Wikidata-style S/P/O)
            subject = fact_item.get("subject")
            predicate = fact_item.get("predicate")
            object_json = fact_item.get("object_json")
            qualifiers_json = fact_item.get("qualifiers_json")
            attribution_json = fact_item.get("attribution_json")
        else:
            fact_content = fact_item
            summary = None
            candidate_type = "sourced_claim"
            source_refs_json = None
            source_excerpt = None
            verification_status = None
            subject = None
            predicate = None
            object_json = None
            qualifiers_json = None
            attribution_json = None

        if not isinstance(fact_content, str):
            continue
        normalized = _normalize_fact_text(fact_content)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        if api.fact_exists(topic_id, normalized):
            logger.info(
                "[Writer Processor] Skipping final fact duplicate: %s...",
                normalized[:50],
            )
            continue
        if api.fact_candidate_exists(topic_id, normalized, statuses=("pending",)):
            logger.info(
                "[Writer Processor] Skipping pending fact candidate duplicate: %s...",
                normalized[:50],
            )
            continue
        # Number-based dedup: skip if a key number already exists in the Fact table
        numbers_in_candidate = re.findall(r"(?<!\d)\d+(?:\.\d+)?(?!\d)", normalized)
        skip = False
        if numbers_in_candidate:
            with db.get_db() as conn:
                for num in numbers_in_candidate[:3]:
                    rows = conn.execute(
                        "SELECT content FROM Fact WHERE topic_id = ? AND content LIKE ? LIMIT 3",
                        (topic_id, f"%{num}%"),
                    ).fetchall()
                    for (fact_content,) in rows:
                        if re.search(
                            r"(?<!\d)" + re.escape(num) + r"(?!\d)", fact_content
                        ):
                            logger.info(
                                "[Writer Processor] Skipping number-duplicate fact (number %s already in Fact table): %s...",
                                num,
                                normalized[:50],
                            )
                            skip = True
                            break
                    if skip:
                        break
        if skip:
            continue
        # FA-3: Writer citation pre-flight — strip invalid [W] references
        if source_refs_json:
            w_ids_raw = re.findall(r"W(\d+)", source_refs_json)
            if w_ids_raw:
                w_ids = [int(x) for x in w_ids_raw]
                valid_w_ids = db.web_evidence_ids_exist(topic_id, w_ids)
                if valid_w_ids != set(w_ids):
                    invalid = set(w_ids) - valid_w_ids
                    logger.info(
                        "[Writer Processor] Stripping invalid W-refs %s from candidate: %s...",
                        invalid,
                        normalized[:50],
                    )
                    try:
                        refs_list = json.loads(source_refs_json)
                        refs_list = [
                            r
                            for r in refs_list
                            if not (
                                isinstance(r, str)
                                and re.match(r"W\d+$", r)
                                and int(r[1:]) in invalid
                            )
                        ]
                        source_refs_json = json.dumps(refs_list, ensure_ascii=False)
                    except (json.JSONDecodeError, TypeError):
                        pass
        candidate_id = api.create_fact_candidate_with_stage(
            topic_id,
            subtopic_id,
            writer_msg_id,
            normalized,
            summary=summary,
            fact_stage=fact_stage,
            candidate_type=candidate_type,
            source_kind="agent",
            evidence_note=evidence_note,
            source_refs_json=source_refs_json,
            source_excerpt=source_excerpt,
            verification_status=verification_status,
            round_number=round_number,
            subject=subject,
            predicate=predicate,
            object_json=object_json,
            qualifiers_json=qualifiers_json,
            attribution_json=attribution_json,
        )
        created_ids.append(candidate_id)
        logger.info("[Writer Processor] Created FactCandidate ID: %s", candidate_id)
        if max_candidates is not None and len(created_ids) >= max_candidates:
            break
    return created_ids


async def process_writer_output(
    topic_id: int,
    subtopic_id: int,
    writer_msg_id: Optional[int],
    writer_text: str,
    structured_facts: List[str] | None = None,
    fact_stage: str = "synthesized",
    evidence_note: Optional[str] = None,
    round_number: int | None = None,
    max_candidates: int | None = None,
) -> list[int]:
    """
    Stores structured candidate facts in FactCandidate after normalization and deduplication.
    Permanent Fact insertion is delegated to the Librarian review stage.
    """
    facts = (
        structured_facts
        if structured_facts is not None
        else _extract_fact_lines(writer_text)
    )
    return await _store_fact_candidates(
        topic_id,
        subtopic_id,
        writer_msg_id,
        facts,
        fact_stage=fact_stage,
        evidence_note=evidence_note,
        round_number=round_number,
        max_candidates=max_candidates,
    )


async def process_clerk_claim_output(
    topic_id: int,
    subtopic_id: int,
    clerk_msg_id: Optional[int],
    claim_candidates: Iterable[dict],
    *,
    max_candidates: int | None = None,
) -> list[int]:
    seen: set[str] = set()
    created_ids: list[int] = []
    for claim in claim_candidates:
        if not isinstance(claim, dict):
            continue
        candidate_text = claim.get("candidate_text") or ""
        if not isinstance(candidate_text, str):
            continue
        normalized = _normalize_fact_text(candidate_text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        if api.claim_candidate_exists(
            topic_id, normalized, statuses=("pending", "accept", "soften")
        ):
            logger.info(
                "[Writer Processor] Skipping claim duplicate: %s...", normalized[:50]
            )
            continue
        support_fact_ids = claim.get("support_fact_ids") or []
        if not isinstance(support_fact_ids, list) or not support_fact_ids:
            continue
        rationale_short = claim.get("rationale_short")
        summary = claim.get("summary", "").strip()
        created_id = api.create_claim_candidate(
            topic_id,
            subtopic_id,
            clerk_msg_id,
            normalized,
            summary=summary,
            support_fact_ids_json=json.dumps(support_fact_ids, ensure_ascii=True),
            rationale_short=(
                rationale_short if isinstance(rationale_short, str) else None
            ),
        )
        created_ids.append(created_id)
        logger.info("[Writer Processor] Created ClaimCandidate ID: %s", created_id)
        if max_candidates is not None and len(created_ids) >= max_candidates:
            break
    return created_ids
