"""Tests for Claim Tournament v2: judge_claim_quality and apply_claim_review."""

import os
from unittest.mock import AsyncMock, patch

import pytest

os.environ["TESTING"] = "1"

from orbit_or.librarian_processor import MAX_CLAIMS_PER_FACT_SET, apply_claim_review
from orbit_or.structured_retry import judge_claim_quality

EXISTING_CLAIMS = [
    {"id": 10, "content": "Claim A"},
    {"id": 20, "content": "Claim B"},
]
SUPPORT_FACTS = ["Fact 1", "Fact 2"]


# ---------------------------------------------------------------------------
# judge_claim_quality
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "llm_response,expected",
    [
        ("DUPLICATE_OF: 2 ", "DUPLICATE_OF:2"),
        (" duplicate_of:3", "DUPLICATE_OF:3"),
        ("DUPLICATE_OF:  1 ", "DUPLICATE_OF:1"),
    ],
)
async def test_judge_claim_quality_normalizes_whitespace(llm_response, expected):
    with patch("orbit_or.broker.call_text", new=AsyncMock(return_value=llm_response)):
        result = await judge_claim_quality("New claim", EXISTING_CLAIMS, SUPPORT_FACTS)
    assert result == expected


@pytest.mark.asyncio
async def test_judge_claim_quality_non_digit_index():
    with patch(
        "orbit_or.broker.call_text", new=AsyncMock(return_value="DUPLICATE_OF:abc")
    ):
        result = await judge_claim_quality("New claim", EXISTING_CLAIMS, SUPPORT_FACTS)
    assert result == "different_angle"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "new_claim,existing",
    [
        ("", EXISTING_CLAIMS),
        ("New claim", []),
    ],
)
async def test_judge_claim_quality_empty_inputs(new_claim, existing):
    mock_call = AsyncMock()
    with patch("orbit_or.broker.call_text", new=mock_call):
        result = await judge_claim_quality(new_claim, existing, SUPPORT_FACTS)
    assert result == "different_angle"
    mock_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_judge_claim_quality_llm_exception():
    with patch(
        "orbit_or.broker.call_text", new=AsyncMock(side_effect=RuntimeError("boom"))
    ):
        result = await judge_claim_quality("New claim", EXISTING_CLAIMS, SUPPORT_FACTS)
    assert result == "different_angle"


@pytest.mark.asyncio
async def test_judge_claim_quality_llm_returns_none():
    with patch("orbit_or.broker.call_text", new=AsyncMock(return_value=None)):
        result = await judge_claim_quality("New claim", EXISTING_CLAIMS, SUPPORT_FACTS)
    assert result == "different_angle"


# ---------------------------------------------------------------------------
# apply_claim_review — helpers
# ---------------------------------------------------------------------------

TOPIC_ID = 1

CANDIDATE = {
    "id": 100,
    "candidate_text": "New claim text",
    "subtopic_id": 5,
    "summary": "short summary",
    "rationale_short": "reason",
}

ACCEPT_REVIEW = {
    "decision": "accept",
    "reviewed_text": "New claim text",
    "summary": None,
    "review_note": "",
    "supported_fact_ids": [1, 2],
    "claim_score": 7.5,
}


def _make_overlapping(n: int) -> list[dict]:
    return [{"id": 200 + i, "content": f"Existing claim {i}"} for i in range(n)]


def _make_facts(fact_ids: list[int]) -> list[dict]:
    return [{"id": fid, "content": f"Fact {fid}"} for fid in fact_ids]


def _patch_apply(
    *,
    facts_by_ids=None,
    claim_by_content=None,
    claims_by_fact_set=None,
    insert_claim_rv=999,
    judge_rv="different_angle",
):
    """Return a stack of patches for apply_claim_review dependencies."""
    if facts_by_ids is None:
        facts_by_ids = _make_facts([1, 2])
    if claims_by_fact_set is None:
        claims_by_fact_set = []

    patches = {
        "aget_embedding": patch(
            "orbit_or.librarian_processor.aget_embedding",
            new=AsyncMock(return_value=None),
        ),
        "check_claim_dup": patch(
            "orbit_or.librarian_processor.check_claim_duplicate",
            new=AsyncMock(return_value=("INSERT", None)),
        ),
        "get_facts_by_ids": patch(
            "orbit_or.librarian_processor.api.get_facts_by_ids",
            return_value=facts_by_ids,
        ),
        "get_claim_by_content": patch(
            "orbit_or.librarian_processor.api.get_claim_by_content",
            return_value=claim_by_content,
        ),
        "get_claims_by_support_fact_set": patch(
            "orbit_or.librarian_processor.api.get_claims_by_support_fact_set",
            return_value=claims_by_fact_set,
        ),
        "insert_claim": patch(
            "orbit_or.librarian_processor.api.insert_claim",
            return_value=insert_claim_rv,
        ),
        "update_review": patch(
            "orbit_or.librarian_processor.api.update_claim_candidate_review",
        ),
        "judge": patch(
            "orbit_or.librarian_processor.judge_claim_quality",
            new=AsyncMock(return_value=judge_rv),
        ),
        "insert_knowledge_edge": patch(
            "orbit_or.librarian_processor.api.insert_knowledge_edge",
            return_value=None,
        ),
    }
    return patches


# ---------------------------------------------------------------------------
# apply_claim_review tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_claim_review_cap_reached():
    overlapping = _make_overlapping(MAX_CLAIMS_PER_FACT_SET)
    p = _patch_apply(claims_by_fact_set=overlapping)
    with (
        p["aget_embedding"],
        p["check_claim_dup"],
        p["get_facts_by_ids"],
        p["get_claim_by_content"],
        p["get_claims_by_support_fact_set"],
        p["insert_claim"],
        p["update_review"],
        p["judge"] as mock_judge,
    ):
        result = await apply_claim_review(TOPIC_ID, CANDIDATE, ACCEPT_REVIEW)

    assert result["accepted_claim_id"] == overlapping[0]["id"]
    mock_judge.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_claim_review_tournament_valid_index():
    overlapping = _make_overlapping(2)
    p = _patch_apply(claims_by_fact_set=overlapping, judge_rv="DUPLICATE_OF:2")
    with (
        p["aget_embedding"],
        p["check_claim_dup"],
        p["get_facts_by_ids"],
        p["get_claim_by_content"],
        p["get_claims_by_support_fact_set"],
        p["insert_claim"],
        p["update_review"],
        p["judge"],
    ):
        result = await apply_claim_review(TOPIC_ID, CANDIDATE, ACCEPT_REVIEW)

    assert result["accepted_claim_id"] == overlapping[1]["id"]


@pytest.mark.asyncio
async def test_apply_claim_review_tournament_bad_index():
    overlapping = _make_overlapping(2)
    p = _patch_apply(claims_by_fact_set=overlapping, judge_rv="DUPLICATE_OF:99")
    with (
        p["aget_embedding"],
        p["check_claim_dup"],
        p["get_facts_by_ids"],
        p["get_claim_by_content"],
        p["get_claims_by_support_fact_set"],
        p["insert_claim"],
        p["update_review"],
        p["judge"],
    ):
        result = await apply_claim_review(TOPIC_ID, CANDIDATE, ACCEPT_REVIEW)

    assert result["accepted_claim_id"] == overlapping[0]["id"]


@pytest.mark.asyncio
async def test_apply_claim_review_different_angle_inserts():
    overlapping = _make_overlapping(2)
    p = _patch_apply(claims_by_fact_set=overlapping, insert_claim_rv=777)
    with (
        p["aget_embedding"],
        p["check_claim_dup"],
        p["get_facts_by_ids"],
        p["get_claim_by_content"],
        p["get_claims_by_support_fact_set"],
        p["insert_claim"] as mock_insert,
        p["update_review"],
        p["judge"],
        p["insert_knowledge_edge"],
    ):
        result = await apply_claim_review(TOPIC_ID, CANDIDATE, ACCEPT_REVIEW)

    mock_insert.assert_called_once()
    assert result["accepted_claim_id"] == 777


@pytest.mark.asyncio
async def test_apply_claim_review_all_facts_invalid():
    p = _patch_apply(facts_by_ids=[])
    with (
        p["aget_embedding"],
        p["check_claim_dup"],
        p["get_facts_by_ids"],
        p["get_claim_by_content"],
        p["get_claims_by_support_fact_set"],
        p["insert_claim"] as mock_insert,
        p["update_review"],
        p["judge"],
    ):
        result = await apply_claim_review(TOPIC_ID, CANDIDATE, ACCEPT_REVIEW)

    assert result["accepted_claim_id"] is None
    mock_insert.assert_not_called()
