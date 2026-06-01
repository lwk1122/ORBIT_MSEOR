"""Tests for the circuit breaker in audience_termination_check_node and count_facts."""

import os

os.environ["TESTING"] = "1"

import pytest
from unittest.mock import AsyncMock, patch

from orbit_or import api, db
from orbit_or.server import (
    ANALYSIS_PHASE,
    audience_termination_check_node,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fresh_db(tmp_path, monkeypatch):
    """Provide a clean database for count_facts tests."""
    test_db = str(tmp_path / "test_cb.db")
    monkeypatch.setattr(db, "get_db_path", lambda: test_db)
    db.init_db()
    yield


def _base_state(
    *,
    round_number: int = 5,
    prev_1: int = 10,
    prev_2: int = 10,
    gap_search_active: bool = False,
) -> dict:
    """Return a minimal ChatState dict for circuit breaker tests."""
    return {
        "topic_id": 1,
        "plan_id": 1,
        "subtopic_id": 1,
        "pending_subtopics": [],
        "pending_turns": [],
        "current_actor": "",
        "current_turn_kind": "",
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "phase": ANALYSIS_PHASE,
        "subtopic_exhausted": False,
        "round_number": round_number,
        "fact_count_1_round_ago": prev_1,
        "fact_count_2_rounds_ago": prev_2,
        "gap_search_active": gap_search_active,
    }


def _patch_context_entities():
    """Mock _load_context_entities to pass the guard."""
    return patch(
        "orbit_or.server._load_context_entities",
        return_value=({"summary": "test"}, {"summary": "sub"}),
    )


def _patch_past_circuit_breaker():
    """Stack of mocks needed to let the function proceed past the circuit breaker
    through the termination vote path without hitting real I/O."""
    return [
        patch("orbit_or.server.api.get_messages", return_value=[]),
        patch(
            "orbit_or.server.aget_embedding", new=AsyncMock(return_value=[0.1] * 768)
        ),
        patch("orbit_or.server.api.search_messages_hybrid", return_value=[]),
        patch(
            "orbit_or.server._run_termination_votes", new=AsyncMock(return_value=[])
        ),
        patch("orbit_or.server.api.post_message"),
    ]


# ---------------------------------------------------------------------------
# 1. count_facts — subtopic scope (real DB)
# ---------------------------------------------------------------------------


def test_count_facts_subtopic_scope(fresh_db):
    tid = api.create_topic("CB topic", "detail")
    sid_a = api.create_subtopic(tid, "Sub A", "detail A")
    sid_b = api.create_subtopic(tid, "Sub B", "detail B")

    for i in range(3):
        api.insert_fact(tid, f"Fact A-{i}", "test", subtopic_id=sid_a)
    for i in range(2):
        api.insert_fact(tid, f"Fact B-{i}", "test", subtopic_id=sid_b)

    assert api.count_facts(tid, subtopic_id=sid_a) == 3
    assert api.count_facts(tid, subtopic_id=sid_b) == 2
    assert api.count_facts(tid) == 5


# ---------------------------------------------------------------------------
# 2. Circuit breaker fires when stale for 2 rounds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_circuit_breaker_fires_stale_2_rounds():
    state = _base_state(round_number=5, prev_1=10, prev_2=10)

    with _patch_context_entities():
        with patch("orbit_or.server.api.count_facts", return_value=10):
            result = await audience_termination_check_node(state)

    assert result["subtopic_exhausted"] is True
    assert result["close_reason"] == "cognitive_yield_exhausted"


# ---------------------------------------------------------------------------
# 3. Circuit breaker yields to gap search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_circuit_breaker_yields_to_gap_search():
    state = _base_state(round_number=5, prev_1=10, prev_2=10, gap_search_active=True)

    with _patch_context_entities():
        with patch("orbit_or.server.api.count_facts", return_value=10):
            patches = _patch_past_circuit_breaker()
            for p in patches:
                p.start()
            try:
                result = await audience_termination_check_node(state)
            finally:
                for p in reversed(patches):
                    p.stop()

    assert result.get("close_reason") != "cognitive_yield_exhausted"


# ---------------------------------------------------------------------------
# 4. Circuit breaker does not fire before round 4
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_circuit_breaker_not_before_round_4():
    state = _base_state(round_number=3, prev_1=10, prev_2=10)

    with _patch_context_entities():
        with patch("orbit_or.server.api.count_facts", return_value=10):
            patches = _patch_past_circuit_breaker()
            for p in patches:
                p.start()
            try:
                result = await audience_termination_check_node(state)
            finally:
                for p in reversed(patches):
                    p.stop()

    assert result.get("close_reason") != "cognitive_yield_exhausted"


# ---------------------------------------------------------------------------
# 5. Circuit breaker does not fire when facts are growing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_circuit_breaker_not_when_facts_growing():
    state = _base_state(round_number=5, prev_1=8, prev_2=5)

    with _patch_context_entities():
        with patch("orbit_or.server.api.count_facts", return_value=10):
            patches = _patch_past_circuit_breaker()
            for p in patches:
                p.start()
            try:
                result = await audience_termination_check_node(state)
            finally:
                for p in reversed(patches):
                    p.stop()

    assert result.get("close_reason") != "cognitive_yield_exhausted"


# ---------------------------------------------------------------------------
# 6. Circuit breaker fires with zero facts at round 5
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_circuit_breaker_zero_facts_round_5():
    state = _base_state(round_number=5, prev_1=0, prev_2=0)

    with _patch_context_entities():
        with patch("orbit_or.server.api.count_facts", return_value=0):
            result = await audience_termination_check_node(state)

    assert result["subtopic_exhausted"] is True
    assert result["close_reason"] == "cognitive_yield_exhausted"


# ---------------------------------------------------------------------------
# 7. Fact counter shift — counters slide correctly in return dict
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fact_counter_shift():
    old_prev_1 = 8
    current_count = 12
    state = _base_state(round_number=5, prev_1=old_prev_1, prev_2=5)

    with _patch_context_entities():
        with patch("orbit_or.server.api.count_facts", return_value=current_count):
            patches = _patch_past_circuit_breaker()
            for p in patches:
                p.start()
            try:
                result = await audience_termination_check_node(state)
            finally:
                for p in reversed(patches):
                    p.stop()

    assert result["fact_count_2_rounds_ago"] == old_prev_1
    assert result["fact_count_1_round_ago"] == current_count
