import json
import logging
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence

from . import analytics
from . import api
from .embedding import aget_embedding
from .fact_dedup import check_claim_duplicate, check_fact_duplicate
from .json_utils import extract_json_object as _extract_json
from .structured_retry import judge_claim_quality

logger = logging.getLogger(__name__)

VALID_FACT_DECISIONS = {"accept", "correct", "soften", "reject"}
VALID_CLAIM_DECISIONS = {"accept", "soften", "reject"}
MAX_CLAIMS_PER_FACT_SET = 3


def _normalize_fact_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return " ".join(text.split())


def _clamp_confidence(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return max(0.0, min(10.0, float(value)))
    except (TypeError, ValueError):
        return None


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _normalize_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    ids: list[int] = []
    for item in value:
        try:
            ids.append(int(item))
        except (TypeError, ValueError):
            continue
    return ids


def parse_librarian_review(raw_text: str, candidate_text: str) -> Dict[str, Any]:
    parsed = _extract_json(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError("Librarian review did not return valid JSON.")

    decision = str(parsed.get("decision", "")).strip().lower()
    if decision not in VALID_FACT_DECISIONS:
        raise ValueError(f"Invalid librarian decision: {decision}")

    verification_status = str(parsed.get("verification_status", "")).strip().lower()
    if verification_status not in {"accepted", "corrected", "unsupported", "refuted"}:
        verification_status = {
            "accept": "accepted",
            "correct": "corrected",
            "soften": "accepted",
            "reject": "unsupported",
        }[decision]

    reviewed_text = parsed.get("reviewed_text")
    summary = parsed.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        summary = None
    else:
        summary = summary.strip()
        if len(summary) > 500:
            summary = summary[:500]
    if decision in {"accept", "correct"}:
        if not isinstance(reviewed_text, str) or not reviewed_text.strip():
            reviewed_text = candidate_text
        reviewed_text = _normalize_fact_text(reviewed_text)
        if not reviewed_text:
            raise ValueError("Accepted fact must include reviewed text.")
    elif decision == "soften":
        if not isinstance(reviewed_text, str) or not reviewed_text.strip():
            raise ValueError("Softened fact must include rewritten reviewed text.")
        reviewed_text = _normalize_fact_text(reviewed_text)
        if not reviewed_text:
            raise ValueError("Softened fact must include rewritten reviewed text.")
    else:
        reviewed_text = None

    review_note = parsed.get("review_note")
    if not isinstance(review_note, str):
        review_note = ""

    evidence_note = parsed.get("evidence_note")
    if not isinstance(evidence_note, str):
        evidence_note = ""

    return {
        "decision": decision,
        "verification_status": verification_status,
        "reviewed_text": reviewed_text,
        "summary": summary,
        "review_note": review_note.strip(),
        "evidence_note": evidence_note.strip(),
        "source_refs": _normalize_string_list(
            parsed.get("source_refs_json") or parsed.get("source_refs")
        ),
        "source_excerpt": (
            parsed.get("source_excerpt", "").strip()
            if isinstance(parsed.get("source_excerpt"), str)
            else ""
        ),
        "confidence_score": _clamp_confidence(parsed.get("confidence_score")),
    }


def _build_fact_insert_kwargs(
    candidate,
    verification_status,
    source_refs,
    source_excerpt,
    candidate_id,
    decision,
    evidence_note,
    confidence_score,
) -> dict:
    """Build kwargs dict for api.insert_fact() from librarian review context."""
    return {
        "subtopic_id": candidate.get("subtopic_id"),
        "fact_stage": candidate.get("fact_stage", "synthesized"),
        "fact_type": candidate.get("candidate_type", "sourced_claim"),
        "verification_status": verification_status,
        "source_kind": candidate.get("source_kind")
        or (
            "web"
            if candidate.get("fact_stage") in ("web_extracted", "inline", "bootstrap")
            else "code" if candidate.get("fact_stage") == "code_verified" else "agent"
        ),
        "source_refs_json": (
            json.dumps(source_refs, ensure_ascii=True)
            if source_refs
            else candidate.get("source_refs_json")
        ),
        "source_excerpt": source_excerpt or candidate.get("source_excerpt"),
        "candidate_id": candidate_id,
        "review_status": decision,
        "evidence_note": evidence_note or None,
        "confidence_score": confidence_score,
    }


async def apply_librarian_review(
    topic_id: int,
    candidate: Dict[str, Any],
    review: Dict[str, Any],
) -> Dict[str, Any]:
    candidate_id = candidate["id"]
    decision = review["decision"]
    verification_status = review.get("verification_status")
    reviewed_text = review.get("reviewed_text")
    review_note = review.get("review_note")
    evidence_note = review.get("evidence_note")
    source_refs = review.get("source_refs") or []
    source_excerpt = review.get("source_excerpt") or ""
    confidence_score = review.get("confidence_score")

    accepted_fact_id = None
    stored_text = None
    is_new_fact = False

    # Compute external-source flag once, used by multiple fact gates below
    candidate_source_refs = candidate.get("source_refs_json") or ""
    has_external_source = bool(
        re.search(r"D\d+", candidate_source_refs)
        or re.search(r"W\d+", candidate_source_refs)
        or re.search(r"E\d+", candidate_source_refs)
    )

    if decision in {"accept", "correct", "soften"}:
        # --- Fact Gate: reject candidates without Layer 0 provenance ---
        candidate_type = candidate.get("candidate_type", "sourced_claim")
        if candidate_type == "number" and not has_external_source:
            logger.info(
                "[Librarian] Fact gate: rejecting candidate %s (number without [D]/[W]/[E] source)",
                candidate_id,
            )
            decision = "reject"
            verification_status = "unsupported"
            review_note = (
                review_note or ""
            ) + " [Fact gate: number without external source]"
            review["decision"] = "reject"
            review["verification_status"] = verification_status
            review["review_note"] = review_note
            stored_text = None
            # Fall through to update_fact_candidate_review below

    if decision in {"accept", "correct", "soften"}:
        # --- FA-1: Hard boundary — agent-sourced without external evidence ---
        inferred_source_kind = candidate.get("source_kind") or (
            "web"
            if candidate.get("fact_stage") in ("web_extracted", "inline", "bootstrap")
            else "code" if candidate.get("fact_stage") == "code_verified" else "agent"
        )
        if inferred_source_kind == "agent" and not has_external_source:
            logger.info(
                "[Librarian] Fact gate: rejecting candidate %s (agent source without [D]/[W]/[E])",
                candidate_id,
            )
            decision = "reject"
            verification_status = "unsupported"
            review_note = (
                review_note or ""
            ) + " [Fact gate: agent opinion without external evidence]"
            review["decision"] = "reject"
            review["verification_status"] = verification_status
            review["review_note"] = review_note
            stored_text = None

    if decision in {"accept", "correct", "soften"}:
        stored_text = _normalize_fact_text(reviewed_text or candidate["candidate_text"])
        existing_fact = api.get_fact_by_content(topic_id, stored_text)

        if existing_fact:
            accepted_fact_id = existing_fact["id"]
            logger.info(
                "[Librarian] Reusing existing fact %s for candidate %s",
                accepted_fact_id,
                candidate_id,
            )
        else:
            # Phase B.1: Semantic dedup before insert
            emb = await aget_embedding(stored_text)
            if emb is not None:
                dedup_action, matched_id = await check_fact_duplicate(
                    topic_id, stored_text, emb
                )
                if dedup_action == "DUPLICATE" and matched_id is not None:
                    accepted_fact_id = matched_id
                    new_refs = review.get("source_refs") or []
                    if new_refs:
                        api.merge_fact_source_ref(matched_id, new_refs)
                    logger.info(
                        "[fact-dedup] Semantic duplicate of F%s for candidate %s",
                        matched_id,
                        candidate_id,
                    )
                elif dedup_action == "UPDATE" and matched_id is not None:
                    # Phase C: temporal supersession — insert new, supersede old
                    insert_kwargs = _build_fact_insert_kwargs(
                        candidate,
                        verification_status,
                        source_refs,
                        source_excerpt,
                        candidate_id,
                        decision,
                        evidence_note,
                        confidence_score,
                    )
                    accepted_fact_id = api.insert_fact(
                        topic_id,
                        stored_text,
                        source="Librarian",
                        **insert_kwargs,
                    )
                    is_new_fact = True
                    api.supersede_fact(matched_id, accepted_fact_id)
                    api.insert_knowledge_edge(
                        topic_id,
                        accepted_fact_id,
                        "fact",
                        matched_id,
                        "fact",
                        "supersedes",
                        created_by="librarian",
                    )
                    logger.info(
                        "[fact-dedup] F%s supersedes F%s for candidate %s",
                        accepted_fact_id,
                        matched_id,
                        candidate_id,
                    )
                elif dedup_action == "CONTRADICTION" and matched_id is not None:
                    # Phase C: same-period conflict — insert new, create conflict edge
                    insert_kwargs = _build_fact_insert_kwargs(
                        candidate,
                        verification_status,
                        source_refs,
                        source_excerpt,
                        candidate_id,
                        decision,
                        evidence_note,
                        confidence_score,
                    )
                    accepted_fact_id = api.insert_fact(
                        topic_id,
                        stored_text,
                        source="Librarian",
                        **insert_kwargs,
                    )
                    is_new_fact = True
                    api.insert_knowledge_edge(
                        topic_id,
                        accepted_fact_id,
                        "fact",
                        matched_id,
                        "fact",
                        "conflicts_with",
                        created_by="librarian",
                    )
                    logger.info(
                        "[fact-dedup] F%s conflicts_with F%s for candidate %s",
                        accepted_fact_id,
                        matched_id,
                        candidate_id,
                    )

            if accepted_fact_id is None:
                insert_kwargs = _build_fact_insert_kwargs(
                    candidate,
                    verification_status,
                    source_refs,
                    source_excerpt,
                    candidate_id,
                    decision,
                    evidence_note,
                    confidence_score,
                )
                # Insert without embedding; post-hoc summary generation in server.py
                # will compute the summary-based embedding and update it.
                accepted_fact_id = api.insert_fact(
                    topic_id,
                    stored_text,
                    source="Librarian",
                    **insert_kwargs,
                )
                is_new_fact = True

        # Phase C: create derived_from edges from source_refs for newly inserted facts
        if is_new_fact and accepted_fact_id and source_refs:
            for ref in source_refs:
                m = re.match(r"W(\d+)", str(ref))
                if m:
                    api.insert_knowledge_edge(
                        topic_id,
                        int(m.group(1)),
                        "web_evidence",
                        accepted_fact_id,
                        "fact",
                        "derived_from",
                        created_by="librarian",
                    )
                m_e = re.match(r"\[?E(\d+)\]?", str(ref))
                if m_e:
                    api.insert_knowledge_edge(
                        topic_id,
                        int(m_e.group(1)),
                        "code_evidence",
                        accepted_fact_id,
                        "fact",
                        "derived_from",
                        created_by="librarian",
                    )
                m_f = re.match(r"\[?F(\d+)\]?", str(ref))
                if m_f:
                    src_fid = int(m_f.group(1))
                    if src_fid != accepted_fact_id:
                        api.insert_knowledge_edge(
                            topic_id,
                            src_fid,
                            "fact",
                            accepted_fact_id,
                            "fact",
                            "derived_from",
                            created_by="librarian",
                        )

    # Propagate structured columns from candidate to Fact (only for newly inserted facts)
    if accepted_fact_id and is_new_fact and candidate.get("subject"):
        try:
            api.update_fact_structured_columns(
                accepted_fact_id,
                subject=candidate.get("subject"),
                predicate=candidate.get("predicate"),
                object_json=candidate.get("object_json"),
                qualifiers_json=candidate.get("qualifiers_json"),
                attribution_json=candidate.get("attribution_json"),
            )
        except Exception as exc:
            logger.warning(
                "[Librarian] Failed to propagate structured columns to fact %s: %s",
                accepted_fact_id,
                exc,
            )

    api.update_fact_candidate_review(
        candidate_id,
        decision,
        reviewed_text=stored_text,
        review_note=review_note or None,
        evidence_note=evidence_note or None,
        confidence_score=confidence_score,
        reviewer="Librarian",
        accepted_fact_id=accepted_fact_id,
    )

    if decision in {"accept", "correct", "soften"} and accepted_fact_id:
        analytics.capture(
            f"topic_{topic_id}",
            "fact_accepted",
            {
                "fact_id": accepted_fact_id,
                "candidate_id": candidate_id,
                "decision": decision,
                "is_new_fact": is_new_fact,
                "fact_stage": candidate.get("fact_stage"),
            },
        )

    return {
        "candidate_id": candidate_id,
        "record_kind": "fact",
        "candidate_text": candidate["candidate_text"],
        "decision": decision,
        "verification_status": verification_status,
        "reviewed_text": stored_text,
        "stored_text": stored_text,
        "review_note": review_note or "",
        "evidence_note": evidence_note or "",
        "confidence_score": confidence_score,
        "accepted_fact_id": accepted_fact_id,
    }


def parse_claim_review(
    raw_text: str, candidate_text: str, fallback_support_ids: list[int]
) -> Dict[str, Any]:
    parsed = _extract_json(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError("Claim review did not return valid JSON.")

    decision = str(parsed.get("decision", "")).strip().lower()
    if decision not in VALID_CLAIM_DECISIONS:
        raise ValueError(f"Invalid claim review decision: {decision}")

    reviewed_text = parsed.get("reviewed_text")
    summary = parsed.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        summary = None
    else:
        summary = summary.strip()
        if len(summary) > 500:
            summary = summary[:500]
    if decision == "accept":
        if not isinstance(reviewed_text, str) or not reviewed_text.strip():
            reviewed_text = candidate_text
        reviewed_text = _normalize_fact_text(reviewed_text)
    elif decision == "soften":
        if not isinstance(reviewed_text, str) or not reviewed_text.strip():
            raise ValueError("Softened claim must include reviewed text.")
        reviewed_text = _normalize_fact_text(reviewed_text)
    else:
        reviewed_text = None

    review_note = parsed.get("review_note")
    if not isinstance(review_note, str):
        review_note = ""

    supported_fact_ids = _normalize_int_list(
        parsed.get("supported_fact_ids") or parsed.get("support_fact_ids")
    )
    if not supported_fact_ids:
        supported_fact_ids = list(fallback_support_ids)

    # Extract librarian-refined G.4 fields (optional — omitted fields = no change)
    g4_fields = {}
    for key in (
        "claim_type",
        "scope_tags",
        "scope_context",
        "falsification_criteria",
        "inference_logic",
        "conclusion",
    ):
        val = parsed.get(key)
        if isinstance(val, str) and val.strip():
            g4_fields[key] = val.strip()
    for key in ("evidence_strength", "scope_breadth"):
        val = parsed.get(key)
        if val is not None:
            try:
                g4_fields[key] = float(val)
            except (TypeError, ValueError):
                pass

    return {
        "decision": decision,
        "reviewed_text": reviewed_text,
        "summary": summary,
        "review_note": review_note.strip(),
        "supported_fact_ids": supported_fact_ids,
        "claim_score": _clamp_confidence(parsed.get("claim_score")),
        "g4_fields": g4_fields,
    }


def _filter_valid_supported_fact_ids(
    topic_id: int, fact_ids: Sequence[int]
) -> list[int]:
    if not fact_ids:
        return []
    facts = api.get_facts_by_ids(topic_id, list(fact_ids))
    valid_ids = {fact["id"] for fact in facts}
    invalid_ids = [fid for fid in fact_ids if fid not in valid_ids]
    if invalid_ids:
        logger.warning(
            "[claim] Dropping unsupported fact_ids=%s for topic=%s",
            invalid_ids,
            topic_id,
        )
    return [fid for fid in fact_ids if fid in valid_ids]


def _canonical_fact_ids_json(fact_ids: list[int]) -> str:
    return json.dumps(sorted(fact_ids), ensure_ascii=True)


def _insert_new_claim(
    topic_id,
    candidate,
    stored_text,
    summary,
    canonical_ids_json,
    claim_score,
    decision,
    candidate_id,
    supported_fact_ids=None,
):
    claim_id = api.insert_claim(
        topic_id,
        candidate.get("subtopic_id"),
        stored_text,
        summary=summary or candidate.get("summary"),
        support_fact_ids_json=canonical_ids_json,
        rationale_short=candidate.get("rationale_short"),
        claim_score=claim_score,
        status="active" if decision == "accept" else "contested",
        candidate_id=candidate_id,
        claim_type=candidate.get("claim_type"),
        scope_tags=candidate.get("scope_tags"),
        scope_context=candidate.get("scope_context"),
        falsification_criteria=candidate.get("falsification_criteria"),
        inference_logic=candidate.get("inference_logic"),
        conclusion=candidate.get("conclusion"),
        evidence_strength=candidate.get("evidence_strength"),
        scope_breadth=candidate.get("scope_breadth"),
        submitted_by=candidate.get("submitted_by"),
    )

    # Phase C: create supports edges (fact→claim)
    if claim_id and supported_fact_ids:
        group_id = f"g_{claim_id}"
        for fid in supported_fact_ids:
            api.insert_knowledge_edge(
                topic_id,
                fid,
                "fact",
                claim_id,
                "claim",
                "supports",
                justification_group=group_id,
                created_by="librarian",
            )

    return claim_id


async def apply_claim_review(
    topic_id: int,
    candidate: Dict[str, Any],
    review: Dict[str, Any],
) -> Dict[str, Any]:
    candidate_id = candidate["id"]
    decision = review["decision"]
    reviewed_text = review.get("reviewed_text")
    summary = review.get("summary")
    review_note = review.get("review_note")
    claim_score = review.get("claim_score")
    supported_fact_ids = review.get("supported_fact_ids") or []
    supported_fact_ids = _filter_valid_supported_fact_ids(topic_id, supported_fact_ids)

    # Apply librarian G.4 refinements to candidate (override only non-empty fields)
    g4_overrides = review.get("g4_fields") or {}
    if g4_overrides:
        for key, val in g4_overrides.items():
            candidate[key] = val

    accepted_claim_id = None
    stored_text = None
    if decision in {"accept", "soften"} and not supported_fact_ids:
        logger.warning(
            "[claim-dedup] Candidate %s accepted/softened but all support facts invalid; skipping claim creation",
            candidate_id,
        )
    if decision in {"accept", "soften"} and supported_fact_ids:
        stored_text = _normalize_fact_text(reviewed_text or candidate["candidate_text"])
        canonical_ids_json = _canonical_fact_ids_json(supported_fact_ids)

        # --- Level 0 (Phase B.1): Semantic dedup via lexical + LLM ---
        sem_action, sem_matched_id = await check_claim_duplicate(
            topic_id, stored_text, supported_fact_ids
        )
        if sem_action == "DUPLICATE" and sem_matched_id is not None:
            accepted_claim_id = sem_matched_id
            logger.info(
                "[claim-dedup] L0 semantic duplicate of claim %s",
                accepted_claim_id,
            )
        elif sem_action == "MERGE" and sem_matched_id is not None:
            accepted_claim_id = sem_matched_id
            api.merge_claim_support_facts(sem_matched_id, supported_fact_ids)
            logger.info(
                "[claim-dedup] L0 semantic merge into claim %s",
                accepted_claim_id,
            )
        elif sem_action == "RESTATEMENT":
            logger.info(
                "[claim-dedup] L0 restatement: claim candidate %s", candidate_id
            )
            decision = "reject"
            review_note = (review_note or "") + " [Fact restatement: no new argument]"
        elif sem_action == "CONTRADICTION" and sem_matched_id is not None:
            # Phase C: insert new claim, create conflicts_with edge
            accepted_claim_id = _insert_new_claim(
                topic_id,
                candidate,
                stored_text,
                summary,
                canonical_ids_json,
                claim_score,
                decision,
                candidate_id,
                supported_fact_ids=supported_fact_ids,
            )
            if accepted_claim_id:
                api.insert_knowledge_edge(
                    topic_id,
                    accepted_claim_id,
                    "claim",
                    sem_matched_id,
                    "claim",
                    "conflicts_with",
                    created_by="librarian",
                )
                logger.info(
                    "[claim-dedup] L0 contradiction: claim %s conflicts_with %s",
                    accepted_claim_id,
                    sem_matched_id,
                )

        # --- Level 1: Exact text match + same facts → reuse ---
        if accepted_claim_id is None:
            existing_exact = api.get_claim_by_content(topic_id, stored_text)
            if (
                existing_exact
                and existing_exact.get("support_fact_ids_json") == canonical_ids_json
            ):
                accepted_claim_id = existing_exact["id"]
                logger.info("[claim-dedup] L1 exact: reuse claim %s", accepted_claim_id)
            else:
                # --- Level 2: Same support facts → 1vN tournament ---
                overlapping = api.get_claims_by_support_fact_set(
                    topic_id, canonical_ids_json
                )
                if overlapping:
                    if len(overlapping) >= MAX_CLAIMS_PER_FACT_SET:
                        # Hard cap — reuse champion
                        accepted_claim_id = overlapping[0]["id"]
                        logger.info(
                            "[claim-dedup] Cap reached (%d), reuse %s",
                            len(overlapping),
                            accepted_claim_id,
                        )
                    else:
                        # 1vN tournament
                        existing_for_judge = [
                            {"id": c["id"], "content": c["content"]}
                            for c in overlapping
                        ]
                        fact_texts = [
                            f.get("content", "")
                            for f in api.get_facts_by_ids(topic_id, supported_fact_ids)
                        ][:5]
                        verdict = await judge_claim_quality(
                            stored_text, existing_for_judge, fact_texts
                        )

                        match = re.match(r"DUPLICATE_OF:(\d+)", verdict)
                        if match:
                            idx = int(match.group(1)) - 1
                            if 0 <= idx < len(overlapping):
                                accepted_claim_id = overlapping[idx]["id"]
                                logger.info(
                                    "[claim-dedup] L3 duplicate of claim %s",
                                    accepted_claim_id,
                                )
                            else:
                                accepted_claim_id = overlapping[0]["id"]
                                logger.info(
                                    "[claim-dedup] L3 bad index, reuse champion %s",
                                    accepted_claim_id,
                                )
                        else:
                            accepted_claim_id = _insert_new_claim(
                                topic_id,
                                candidate,
                                stored_text,
                                summary,
                                canonical_ids_json,
                                claim_score,
                                decision,
                                candidate_id,
                                supported_fact_ids=supported_fact_ids,
                            )
                            logger.info(
                                "[claim-dedup] L3 different angle: %s",
                                accepted_claim_id,
                            )
                else:
                    # No overlap — insert normally
                    accepted_claim_id = _insert_new_claim(
                        topic_id,
                        candidate,
                        stored_text,
                        summary,
                        canonical_ids_json,
                        claim_score,
                        decision,
                        candidate_id,
                        supported_fact_ids=supported_fact_ids,
                    )

    api.update_claim_candidate_review(
        candidate_id,
        decision,
        reviewed_text=stored_text,
        review_note=review_note or None,
        claim_score=claim_score,
        accepted_claim_id=accepted_claim_id,
    )

    if decision in {"accept", "soften"} and accepted_claim_id:
        analytics.capture(
            f"topic_{topic_id}",
            "claim_accepted",
            {
                "claim_id": accepted_claim_id,
                "candidate_id": candidate_id,
                "decision": decision,
                "supported_fact_count": len(supported_fact_ids),
            },
        )

    return {
        "candidate_id": candidate_id,
        "record_kind": "claim",
        "candidate_text": candidate["candidate_text"],
        "decision": decision,
        "reviewed_text": stored_text,
        "stored_text": stored_text,
        "review_note": review_note or "",
        "claim_score": claim_score,
        "accepted_claim_id": accepted_claim_id,
    }


def build_librarian_audit_message(results: Iterable[Dict[str, Any]]) -> str:
    fact_accepted: List[str] = []
    fact_corrected: List[str] = []
    fact_softened: List[str] = []
    fact_rejected: List[str] = []
    claim_accepted: List[str] = []
    claim_softened: List[str] = []
    claim_rejected: List[str] = []

    for result in results:
        decision = result["decision"]
        record_kind = result.get("record_kind", "fact")
        if record_kind == "claim":
            if decision == "accept":
                claim_accepted.append(
                    f"- Claim Candidate {result['candidate_id']}: {result['reviewed_text']}"
                )
            elif decision == "soften":
                claim_softened.append(
                    f"- Claim Candidate {result['candidate_id']}: softened to `{result['reviewed_text']}`"
                )
            else:
                note = result.get("review_note") or "unsupported derivation"
                claim_rejected.append(
                    f"- Claim Candidate {result['candidate_id']}: {note}"
                )
            continue

        if decision == "accept":
            fact_accepted.append(
                f"- Candidate {result['candidate_id']}: {result['reviewed_text']}"
            )
        elif decision == "correct":
            fact_corrected.append(
                f"- Candidate {result['candidate_id']}: corrected to `{result['reviewed_text']}`"
            )
        elif decision == "soften":
            fact_softened.append(
                f"- Candidate {result['candidate_id']}: softened to `{result['reviewed_text']}`"
            )
        else:
            note = result.get("review_note") or "unsupported or too interpretive"
            fact_rejected.append(f"- Candidate {result['candidate_id']}: {note}")

    sections = ["LIBRARIAN AUDIT:"]
    if fact_accepted:
        sections.append("FACT ACCEPTED:")
        sections.extend(fact_accepted)
    if fact_corrected:
        sections.append("FACT CORRECTED:")
        sections.extend(fact_corrected)
    if fact_softened:
        sections.append("FACT SOFTENED:")
        sections.extend(fact_softened)
    if fact_rejected:
        sections.append("FACT REJECTED:")
        sections.extend(fact_rejected)
    if claim_accepted:
        sections.append("CLAIM ACCEPTED:")
        sections.extend(claim_accepted)
    if claim_softened:
        sections.append("CLAIM SOFTENED:")
        sections.extend(claim_softened)
    if claim_rejected:
        sections.append("CLAIM REJECTED:")
        sections.extend(claim_rejected)
    return "\n".join(sections)
