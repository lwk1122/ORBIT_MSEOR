"""Canonical text builder, Pydantic models, and entity resolution for structured facts/claims."""

import json
import logging
from typing import Literal, Optional

from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models — Wikidata-style structured schema
# ---------------------------------------------------------------------------


class Proposition(BaseModel):
    subject_entity: str
    predicate: str
    object: dict = {}

    @field_validator("subject_entity", "predicate")
    @classmethod
    def must_be_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must be a non-empty string")
        return v.strip()


class Qualifier(BaseModel):
    key: str
    value: str


class Attribution(BaseModel):
    claimed_by: str = ""
    claim_act: str = ""


class StructuredFact(BaseModel):
    """Schema for facts extracted from WebEvidence (Layer 0 → Layer 1)."""

    proposition: Proposition
    qualifiers: list[Qualifier] = []
    attribution: Attribution = Attribution()
    source_refs: list[str] = []
    raw_text: str = ""


class StructuredClaim(BaseModel):
    """Schema for claims derived by agents (Layer 2)."""

    proposition: Proposition
    qualifiers: list[Qualifier] = []
    polarity: Literal["positive", "negative", "neutral"] = "positive"
    support_fact_ids_json: list[int] = []
    rationale_short: str = ""


# ---------------------------------------------------------------------------
# Canonical text rendering
# ---------------------------------------------------------------------------


def build_canonical_text(
    subject: str,
    predicate: str,
    object_json: Optional[str] = None,
    qualifiers_json: Optional[str] = None,
    attribution_json: Optional[str] = None,
    source_refs: Optional[list[str]] = None,
) -> str:
    """Unified template for Fact and Claim display text.

    Produces a readable one-liner from structured fields.
    """
    text = f"{subject} {predicate}"

    # Object
    obj: dict = {}
    if object_json:
        try:
            obj = (
                json.loads(object_json) if isinstance(object_json, str) else object_json
            )
        except (json.JSONDecodeError, TypeError):
            pass
    if obj.get("value") is not None:
        text += f" {obj['value']}"
        if obj.get("unit"):
            text += f" {obj['unit']}"

    # Qualifiers
    quals: list[dict] = []
    if qualifiers_json:
        try:
            quals = (
                json.loads(qualifiers_json)
                if isinstance(qualifiers_json, str)
                else qualifiers_json
            )
        except (json.JSONDecodeError, TypeError):
            pass
    if quals:
        qual_strs = [
            f"{q.get('key', '')}={q.get('value', '')}" for q in quals if q.get("key")
        ]
        if qual_strs:
            text += f" ({', '.join(qual_strs)})"

    # Attribution
    attr: dict = {}
    if attribution_json:
        try:
            attr = (
                json.loads(attribution_json)
                if isinstance(attribution_json, str)
                else attribution_json
            )
        except (json.JSONDecodeError, TypeError):
            pass
    if attr.get("claimed_by"):
        text += f" — {attr['claimed_by']}"

    # Source refs
    if source_refs:
        text += " " + " ".join(f"[{r}]" for r in source_refs)

    return text


# ---------------------------------------------------------------------------
# Subject snapping — inline cosine-based entity resolution
# ---------------------------------------------------------------------------


def snap_subject(
    subject: str,
    topic_id: int,
    threshold: float = 0.92,
) -> str:
    """Snap a subject entity to an existing canonical name if cosine > threshold.

    Falls back to returning the original subject if no close match or if
    embedding infrastructure is unavailable.
    """
    if not subject or not subject.strip():
        return subject

    from .embedding import get_embedding

    subject_emb = get_embedding(subject)
    if subject_emb is None:
        return subject

    existing = _get_existing_subjects(topic_id)
    if not existing:
        return subject

    best_match: Optional[str] = None
    best_score: float = -1.0

    for canonical, emb in existing:
        score = _cosine_similarity(subject_emb, emb)
        if score > best_score:
            best_score = score
            best_match = canonical

    if best_score >= threshold and best_match is not None:
        logger.info(
            "[canonical] Snapped '%s' → '%s' (cosine=%.4f)",
            subject,
            best_match,
            best_score,
        )
        return best_match

    return subject


def _get_existing_subjects(topic_id: int) -> list[tuple[str, list[float]]]:
    """Fetch unique subject+embedding pairs from Fact and Claim tables for a topic."""
    from . import db as _db
    from .embedding import get_embeddings_batch

    unique_subjects: list[str] = []
    seen: set[str] = set()

    with _db.get_db() as conn:
        for table in ("Fact", "Claim"):
            try:
                rows = conn.execute(
                    f"SELECT DISTINCT subject FROM {table} WHERE topic_id = ? AND subject IS NOT NULL AND subject != ''",
                    (topic_id,),
                ).fetchall()
                for row in rows:
                    subj = row[0]
                    if subj not in seen:
                        seen.add(subj)
                        unique_subjects.append(subj)
            except Exception:
                # Column may not exist yet
                continue

    if not unique_subjects:
        return []

    embeddings = get_embeddings_batch(unique_subjects)
    if embeddings is None or len(embeddings) != len(unique_subjects):
        return []

    return [(subj, emb) for subj, emb in zip(unique_subjects, embeddings) if emb]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Pydantic validation + bounce-back helpers
# ---------------------------------------------------------------------------


def validate_structured_fact(data: dict) -> tuple[Optional[StructuredFact], str]:
    """Validate a dict as StructuredFact. Returns (model, error_string)."""
    try:
        return StructuredFact(**data), ""
    except Exception as exc:
        return None, str(exc)


def validate_structured_claim(data: dict) -> tuple[Optional[StructuredClaim], str]:
    """Validate a dict as StructuredClaim. Returns (model, error_string)."""
    try:
        return StructuredClaim(**data), ""
    except Exception as exc:
        return None, str(exc)


def structured_fact_to_columns(fact: StructuredFact) -> dict:
    """Convert a validated StructuredFact to DB column values."""
    return {
        "subject": fact.proposition.subject_entity,
        "predicate": fact.proposition.predicate,
        "object_json": (
            json.dumps(fact.proposition.object, ensure_ascii=False)
            if fact.proposition.object
            else None
        ),
        "qualifiers_json": (
            json.dumps([q.model_dump() for q in fact.qualifiers], ensure_ascii=False)
            if fact.qualifiers
            else None
        ),
        "attribution_json": (
            json.dumps(fact.attribution.model_dump(), ensure_ascii=False)
            if fact.attribution.claimed_by
            else None
        ),
    }


def structured_claim_to_columns(claim: StructuredClaim) -> dict:
    """Convert a validated StructuredClaim to DB column values."""
    return {
        "subject": claim.proposition.subject_entity,
        "predicate": claim.proposition.predicate,
        "object_json": (
            json.dumps(claim.proposition.object, ensure_ascii=False)
            if claim.proposition.object
            else None
        ),
        "qualifiers_json": (
            json.dumps([q.model_dump() for q in claim.qualifiers], ensure_ascii=False)
            if claim.qualifiers
            else None
        ),
        "polarity": claim.polarity,
    }
