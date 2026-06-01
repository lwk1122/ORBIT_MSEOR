"""Deterministic review helpers for non-solver formal ClaimCandidates."""

from __future__ import annotations

import json
import re
from typing import Any


_VAGUE_TERMS_RE = re.compile(
    r"\b(significant|competitive|robust|better|worse|substantial|strong|weak)\b",
    flags=re.IGNORECASE,
)


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _int_list(value: Any) -> list[int]:
    ids: list[int] = []
    for item in _json_list(value):
        try:
            ids.append(int(item))
        except (TypeError, ValueError):
            continue
    return list(dict.fromkeys(ids))


def _candidate_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim_type": candidate.get("claim_type"),
        "conclusion": candidate.get("conclusion") or candidate.get("candidate_text"),
        "scope_tags": _json_list(candidate.get("scope_tags")),
        "scope_context": candidate.get("scope_context"),
        "falsification_criteria": candidate.get("falsification_criteria"),
        "inference_logic": candidate.get("inference_logic")
        or candidate.get("rationale_short"),
        "evidence_strength": candidate.get("evidence_strength"),
        "scope_breadth": candidate.get("scope_breadth"),
        "support_fact_ids": _int_list(candidate.get("support_fact_ids_json")),
        "candidate_text": candidate.get("candidate_text"),
        "rationale_short": candidate.get("rationale_short"),
        "submitted_by": candidate.get("submitted_by") or "formal_claim_review",
    }


def validate_formal_claim_candidate_payload(payload: dict[str, Any]) -> list[str]:
    """Return deterministic rejection reasons for a non-solver formal claim."""
    claim_type = str(payload.get("claim_type") or "").strip()
    conclusion = str(payload.get("conclusion") or "").strip()
    scope_tags = payload.get("scope_tags") or []
    falsification = str(payload.get("falsification_criteria") or "").strip()
    inference = str(payload.get("inference_logic") or "").strip()
    evidence_strength = payload.get("evidence_strength")
    support_fact_ids = payload.get("support_fact_ids") or []

    issues: list[str] = []
    if claim_type == "optimization_result":
        issues.append("solver_claim_not_supported_here")
    if len(conclusion) < 20:
        issues.append("missing_concrete_conclusion")
    for match in _VAGUE_TERMS_RE.finditer(conclusion):
        nearby = conclusion[max(0, match.start() - 30) : match.end() + 30]
        if not re.search(r"\d", nearby):
            issues.append("vague_unquantified_conclusion")
            break
    if not isinstance(scope_tags, list) or not scope_tags:
        issues.append("missing_scope_tags")
    if not falsification or not re.search(r"\d", falsification):
        issues.append("missing_falsification_threshold")
    if not isinstance(evidence_strength, (int, float)) or not (
        1 <= evidence_strength <= 10
    ):
        issues.append("invalid_evidence_strength")
    if len(inference) < 20:
        issues.append("missing_inference_logic")
    if not support_fact_ids:
        issues.append("missing_support_facts")
    if claim_type == "comparison" and not any(
        isinstance(tag, str) and tag.startswith("dataset:") for tag in scope_tags
    ):
        issues.append("comparison_missing_dataset_scope")
    return issues


def review_formal_claim_candidate(
    topic_id: int,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Accept/reject one non-solver formal ClaimCandidate without provider calls."""
    from . import api

    payload = _candidate_payload(candidate)
    if payload.get("claim_type") == "optimization_result":
        return {"accepted": False, "claim_id": None, "reason": "solver_claim"}

    issues = validate_formal_claim_candidate_payload(payload)
    support_fact_ids = payload["support_fact_ids"]
    existing_support = api.get_facts_by_ids(topic_id, support_fact_ids)
    existing_support_ids = {int(row["id"]) for row in existing_support}
    missing_support_ids = [
        fact_id for fact_id in support_fact_ids if fact_id not in existing_support_ids
    ]
    if missing_support_ids:
        issues.append("orphan_support_facts")

    if issues:
        note = "; ".join(dict.fromkeys(issues))
        api.update_claim_candidate_review(
            int(candidate["id"]),
            "rejected",
            reviewed_text=candidate.get("candidate_text"),
            review_note=note,
            claim_score=payload.get("evidence_strength"),
        )
        return {
            "accepted": False,
            "claim_id": None,
            "reason": "validation_failed",
            "issues": list(dict.fromkeys(issues)),
        }

    conclusion = str(payload["conclusion"]).strip()
    scope_tags_json = json.dumps(payload["scope_tags"], ensure_ascii=True)
    for claim in api.get_claims(topic_id, limit=1000, include_superseded=False):
        if claim.get("claim_type") == "optimization_result":
            continue
        same_conclusion = conclusion == (claim.get("conclusion") or claim.get("content"))
        same_scope = scope_tags_json == (claim.get("scope_tags") or "")
        if same_conclusion and same_scope:
            api.update_claim_candidate_review(
                int(candidate["id"]),
                "rejected",
                reviewed_text=candidate.get("candidate_text"),
                review_note=f"Duplicate of active formal claim C{claim['id']}.",
                claim_score=payload.get("evidence_strength"),
                accepted_claim_id=int(claim["id"]),
            )
            return {
                "accepted": False,
                "claim_id": int(claim["id"]),
                "reason": "duplicate",
            }

    claim_id = api.insert_claim(
        topic_id,
        candidate.get("subtopic_id"),
        candidate.get("candidate_text") or conclusion,
        summary=conclusion,
        support_fact_ids_json=json.dumps(support_fact_ids, ensure_ascii=True),
        rationale_short=payload.get("rationale_short"),
        claim_score=payload.get("evidence_strength"),
        status="active",
        candidate_id=int(candidate["id"]),
        claim_type=payload.get("claim_type"),
        scope_tags=scope_tags_json,
        scope_context=payload.get("scope_context"),
        falsification_criteria=payload.get("falsification_criteria"),
        inference_logic=payload.get("inference_logic"),
        conclusion=conclusion,
        evidence_strength=payload.get("evidence_strength"),
        scope_breadth=payload.get("scope_breadth"),
        submitted_by=payload.get("submitted_by"),
    )
    for fact_id in support_fact_ids:
        api.insert_knowledge_edge(
            topic_id,
            fact_id,
            "fact",
            claim_id,
            "claim",
            "supports",
            justification_group="formal_claim_evidence",
            confidence=min(1.0, float(payload.get("evidence_strength") or 0.0) / 10.0),
            created_by="formal_claim_review",
        )
    api.update_claim_candidate_review(
        int(candidate["id"]),
        "accepted",
        reviewed_text=candidate.get("candidate_text"),
        review_note="Accepted by deterministic formal-claim review.",
        claim_score=payload.get("evidence_strength"),
        accepted_claim_id=claim_id,
    )
    return {"accepted": True, "claim_id": claim_id, "reason": "accepted"}


def review_pending_formal_claim_candidates(topic_id: int) -> list[dict[str, Any]]:
    """Review all pending non-solver structured ClaimCandidates for a topic."""
    from . import api

    results: list[dict[str, Any]] = []
    for candidate in api.get_claim_candidates(topic_id, status="pending", limit=10000):
        if candidate.get("claim_type") and candidate.get("claim_type") != "optimization_result":
            results.append(review_formal_claim_candidate(topic_id, candidate))
    return results
