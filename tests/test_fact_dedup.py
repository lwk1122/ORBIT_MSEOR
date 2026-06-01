"""Tests for fact/claim semantic deduplication (Phase B.1)."""

import os
import sqlite3
from unittest.mock import AsyncMock, patch

import pytest

os.environ["TESTING"] = "1"

from orbit_or.fact_dedup import check_claim_duplicate, check_fact_duplicate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fact_row(fact_id: int, content: str, distance: float) -> dict:
    return {
        "id": fact_id,
        "content": content,
        "distance": distance,
        "topic_id": 1,
        "source": "Librarian",
        "review_status": "accept",
    }


def _make_claim_row(claim_id: int, content: str) -> dict:
    return {
        "id": claim_id,
        "content": content,
        "topic_id": 1,
        "support_fact_ids_json": "[1, 2]",
        "superseded_by": None,
    }


DUMMY_EMBEDDING = [0.1] * 768


# ---------------------------------------------------------------------------
# check_fact_duplicate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_candidates_returns_insert():
    """No similar facts in DB -> INSERT."""
    with patch("orbit_or.fact_dedup._db") as mock_db:
        mock_db.search_facts.return_value = []
        action, matched_id = await check_fact_duplicate(
            1, "Brand new fact", DUMMY_EMBEDDING
        )
    assert action == "INSERT"
    assert matched_id is None


@pytest.mark.asyncio
async def test_candidates_below_threshold_returns_insert():
    """Candidates exist but all below cosine threshold -> INSERT."""
    # L2 distance of 1.0 -> cosine = 1 - 1.0/2 = 0.5 (below 0.80)
    far_fact = _make_fact_row(10, "Some distant fact", distance=1.0)
    with patch("orbit_or.fact_dedup._db") as mock_db:
        mock_db.search_facts.return_value = [far_fact]
        action, matched_id = await check_fact_duplicate(
            1, "Brand new fact", DUMMY_EMBEDDING
        )
    assert action == "INSERT"
    assert matched_id is None


@pytest.mark.asyncio
async def test_duplicate_detected():
    """LLM returns ["M42"] -> DUPLICATE."""
    # L2 distance of 0.2 -> cosine = 1 - 0.04/2 = 0.98
    close_fact = _make_fact_row(42, "World has 97M new AI roles by 2025", distance=0.2)
    with patch("orbit_or.fact_dedup._db") as mock_db:
        mock_db.search_facts.return_value = [close_fact]
        with patch(
            "orbit_or.fact_dedup._call_text",
            new=AsyncMock(return_value='["M42"]'),
        ):
            action, matched_id = await check_fact_duplicate(
                1, "97 million new AI roles expected by 2025", DUMMY_EMBEDDING
            )
    assert action == "DUPLICATE"
    assert matched_id == 42


@pytest.mark.asyncio
async def test_different_numbers_not_duplicate():
    """LLM returns [] for different numbers -> INSERT."""
    close_fact = _make_fact_row(42, "Revenue was $5.2B in Q3", distance=0.2)
    with patch("orbit_or.fact_dedup._db") as mock_db:
        mock_db.search_facts.return_value = [close_fact]
        with patch(
            "orbit_or.fact_dedup._call_text",
            new=AsyncMock(return_value="[]"),
        ):
            action, matched_id = await check_fact_duplicate(
                1, "Revenue was $7.1B in Q4", DUMMY_EMBEDDING
            )
    assert action == "INSERT"
    assert matched_id is None


@pytest.mark.asyncio
async def test_llm_returns_invalid_id():
    """LLM returns ID not in candidates -> INSERT (safety)."""
    close_fact = _make_fact_row(42, "Some fact", distance=0.2)
    with patch("orbit_or.fact_dedup._db") as mock_db:
        mock_db.search_facts.return_value = [close_fact]
        with patch(
            "orbit_or.fact_dedup._call_text",
            new=AsyncMock(return_value='["M999"]'),
        ):
            action, matched_id = await check_fact_duplicate(
                1, "Some similar fact", DUMMY_EMBEDDING
            )
    assert action == "INSERT"
    assert matched_id is None


@pytest.mark.asyncio
async def test_llm_call_fails_returns_insert():
    """LLM call raises -> graceful fallback to INSERT."""
    close_fact = _make_fact_row(42, "Some fact", distance=0.2)
    with patch("orbit_or.fact_dedup._db") as mock_db:
        mock_db.search_facts.return_value = [close_fact]
        with patch(
            "orbit_or.fact_dedup._call_text",
            new=AsyncMock(side_effect=RuntimeError("API down")),
        ):
            action, matched_id = await check_fact_duplicate(
                1, "Some similar fact", DUMMY_EMBEDDING
            )
    assert action == "INSERT"
    assert matched_id is None


@pytest.mark.asyncio
async def test_fact_provider_operational_error_reraises():
    close_fact = _make_fact_row(42, "Some fact", distance=0.2)
    with patch("orbit_or.fact_dedup._db") as mock_db:
        mock_db.search_facts.return_value = [close_fact]
        with patch(
            "orbit_or.fact_dedup.topic_config.get_provider_profile_for",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            with pytest.raises(sqlite3.OperationalError):
                await check_fact_duplicate(1, "Some similar fact", DUMMY_EMBEDDING)


# ---------------------------------------------------------------------------
# check_claim_duplicate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_claim_no_candidates_returns_insert():
    """No lexical matches -> INSERT."""
    with patch("orbit_or.fact_dedup._db") as mock_db:
        mock_db.search_claims_lexical.return_value = []
        action, matched_id = await check_claim_duplicate(1, "Brand new claim", [1, 2])
    assert action == "INSERT"
    assert matched_id is None


@pytest.mark.asyncio
async def test_claim_restatement_provider_operational_error_reraises():
    with patch("orbit_or.fact_dedup._db") as mock_db:
        mock_db.search_claims_lexical.return_value = []
        with patch(
            "orbit_or.fact_dedup.topic_config.get_provider_profile_for",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            with pytest.raises(sqlite3.OperationalError):
                await check_claim_duplicate(
                    1,
                    "Claim restates support",
                    [1],
                    support_fact_texts=["Support fact"],
                )


@pytest.mark.asyncio
async def test_claim_provider_operational_error_reraises():
    existing = _make_claim_row(10, "AI will transform healthcare")
    with patch("orbit_or.fact_dedup._db") as mock_db:
        mock_db.search_claims_lexical.return_value = [existing]
        with patch(
            "orbit_or.fact_dedup.topic_config.get_provider_profile_for",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            with pytest.raises(sqlite3.OperationalError):
                await check_claim_duplicate(1, "AI transforms healthcare", [1, 2])


@pytest.mark.asyncio
async def test_claim_duplicate_same_facts():
    """Same claim, same facts -> DUPLICATE."""
    existing = _make_claim_row(10, "AI will transform healthcare")
    with patch("orbit_or.fact_dedup._db") as mock_db:
        mock_db.search_claims_lexical.return_value = [existing]
        with patch(
            "orbit_or.fact_dedup._call_text",
            new=AsyncMock(return_value='{"action":"DUPLICATE","key":"M10"}'),
        ):
            action, matched_id = await check_claim_duplicate(
                1, "AI is going to transform healthcare", [1, 2]
            )
    assert action == "DUPLICATE"
    assert matched_id == 10


@pytest.mark.asyncio
async def test_claim_merge_different_facts():
    """Same claim, different facts -> MERGE."""
    existing = _make_claim_row(10, "AI will transform healthcare")
    with patch("orbit_or.fact_dedup._db") as mock_db:
        mock_db.search_claims_lexical.return_value = [existing]
        with patch(
            "orbit_or.fact_dedup._call_text",
            new=AsyncMock(return_value='{"action":"MERGE","key":"M10"}'),
        ):
            action, matched_id = await check_claim_duplicate(
                1, "AI is going to transform healthcare", [3, 4]
            )
    assert action == "MERGE"
    assert matched_id == 10


@pytest.mark.asyncio
async def test_claim_insert_when_different():
    """Genuinely different claim -> INSERT."""
    existing = _make_claim_row(10, "AI will transform healthcare")
    with patch("orbit_or.fact_dedup._db") as mock_db:
        mock_db.search_claims_lexical.return_value = [existing]
        with patch(
            "orbit_or.fact_dedup._call_text",
            new=AsyncMock(return_value='{"action":"INSERT"}'),
        ):
            action, matched_id = await check_claim_duplicate(
                1, "Quantum computing will not replace classical", [5]
            )
    assert action == "INSERT"
    assert matched_id is None


# ---------------------------------------------------------------------------
# Integration: librarian_processor dedup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_integration_librarian_fact_dedup():
    """End-to-end: librarian accept + semantic dedup skips insert."""
    candidate = {
        "id": 5,
        "candidate_text": "97M new AI roles by 2025",
        "fact_stage": "synthesized",
        "candidate_type": "sourced_claim",
        "source_refs_json": '["W1"]',
    }
    review = {
        "decision": "accept",
        "reviewed_text": "97 million new AI roles by 2025",
        "review_note": "Supported",
        "evidence_note": "",
        "source_refs": ["W1", "W2"],
        "confidence_score": 9.0,
    }

    with (
        patch(
            "orbit_or.librarian_processor.api.get_fact_by_content", return_value=None
        ),
        patch(
            "orbit_or.librarian_processor.aget_embedding",
            new=AsyncMock(return_value=DUMMY_EMBEDDING),
        ),
        patch(
            "orbit_or.librarian_processor.check_fact_duplicate",
            new=AsyncMock(return_value=("DUPLICATE", 42)),
        ),
        patch("orbit_or.librarian_processor.api.merge_fact_source_ref") as mock_merge,
        patch("orbit_or.librarian_processor.api.insert_fact") as mock_insert,
        patch("orbit_or.librarian_processor.api.update_fact_candidate_review"),
    ):
        from orbit_or.librarian_processor import apply_librarian_review

        result = await apply_librarian_review(1, candidate, review)

    assert result["accepted_fact_id"] == 42
    mock_insert.assert_not_called()
    mock_merge.assert_called_once_with(42, ["W1", "W2"])


# ---------------------------------------------------------------------------
# Phase C: UPDATE / CONTRADICTION
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contradiction_detected():
    """LLM returns CONTRADICTION -> ("CONTRADICTION", fact_id)."""
    close_fact = _make_fact_row(42, "GDP grew 3.2% in Q1 2025", distance=0.2)
    with patch("orbit_or.fact_dedup._db") as mock_db:
        mock_db.search_facts.return_value = [close_fact]
        with patch(
            "orbit_or.fact_dedup._call_text",
            new=AsyncMock(return_value='{"action":"CONTRADICTION","key":"M42"}'),
        ):
            action, matched_id = await check_fact_duplicate(
                1, "GDP contracted 0.5% in Q1 2025", DUMMY_EMBEDDING
            )
    assert action == "CONTRADICTION"
    assert matched_id == 42


@pytest.mark.asyncio
async def test_update_detected():
    """LLM returns UPDATE -> ("UPDATE", fact_id)."""
    close_fact = _make_fact_row(42, "Revenue was $5.2B in Q3 2024", distance=0.2)
    with patch("orbit_or.fact_dedup._db") as mock_db:
        mock_db.search_facts.return_value = [close_fact]
        with patch(
            "orbit_or.fact_dedup._call_text",
            new=AsyncMock(return_value='{"action":"UPDATE","key":"M42"}'),
        ):
            action, matched_id = await check_fact_duplicate(
                1, "Revenue was revised to $5.5B in Q3 2024", DUMMY_EMBEDDING
            )
    assert action == "UPDATE"
    assert matched_id == 42


@pytest.mark.asyncio
async def test_claim_contradiction_detected():
    """Claim LLM returns CONTRADICTION -> ("CONTRADICTION", claim_id)."""
    existing = _make_claim_row(10, "AI will reduce jobs in healthcare")
    with patch("orbit_or.fact_dedup._db") as mock_db:
        mock_db.search_claims_lexical.return_value = [existing]
        with patch(
            "orbit_or.fact_dedup._call_text",
            new=AsyncMock(return_value='{"action":"CONTRADICTION","key":"M10"}'),
        ):
            action, matched_id = await check_claim_duplicate(
                1, "AI will create net new jobs in healthcare", [3, 4]
            )
    assert action == "CONTRADICTION"
    assert matched_id == 10


@pytest.mark.asyncio
async def test_malformed_json_falls_back_to_insert():
    """LLM returns garbled text -> fallback to INSERT."""
    close_fact = _make_fact_row(42, "Some fact", distance=0.2)
    with patch("orbit_or.fact_dedup._db") as mock_db:
        mock_db.search_facts.return_value = [close_fact]
        with patch(
            "orbit_or.fact_dedup._call_text",
            new=AsyncMock(return_value="I think this updates M42 somehow"),
        ):
            action, matched_id = await check_fact_duplicate(
                1, "Some similar fact", DUMMY_EMBEDDING
            )
    assert action == "INSERT"
    assert matched_id is None


@pytest.mark.asyncio
async def test_update_invalid_key_falls_back_to_insert():
    """LLM returns UPDATE with unknown ID -> INSERT."""
    close_fact = _make_fact_row(42, "Some fact", distance=0.2)
    with patch("orbit_or.fact_dedup._db") as mock_db:
        mock_db.search_facts.return_value = [close_fact]
        with patch(
            "orbit_or.fact_dedup._call_text",
            new=AsyncMock(return_value='{"action":"UPDATE","key":"M999"}'),
        ):
            action, matched_id = await check_fact_duplicate(
                1, "Some similar fact", DUMMY_EMBEDDING
            )
    assert action == "INSERT"
    assert matched_id is None
