"""Tests for Phase C: KnowledgeEdge table, JTMS sweep, and contradiction detection."""

import json
import os

import pytest

os.environ["TESTING"] = "1"

from orbit_or import db
from orbit_or.jtms import jtms_sweep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """Create a fresh in-memory-like DB for each test."""
    db_path = str(tmp_path / "test_jtms.db")
    monkeypatch.setattr(db, "get_db_path", lambda: db_path)
    db.init_db()
    return db_path


def _make_topic(conn) -> int:
    cursor = conn.execute(
        "INSERT INTO Topic (summary, detail, status) VALUES ('t', 't', 'Running')"
    )
    return cursor.lastrowid


def _make_fact(conn, topic_id, content="test fact", review_status=None) -> int:
    cursor = conn.execute(
        "INSERT INTO Fact (topic_id, content, source, review_status) VALUES (?, ?, 'test', ?)",
        (topic_id, content, review_status),
    )
    return cursor.lastrowid


def _make_claim(
    conn,
    topic_id,
    content="test claim",
    status="active",
    support_fact_ids_json=None,
    contested_since_round=None,
) -> int:
    cursor = conn.execute(
        "INSERT INTO Claim (topic_id, content, status, support_fact_ids_json, contested_since_round) VALUES (?, ?, ?, ?, ?)",
        (topic_id, content, status, support_fact_ids_json, contested_since_round),
    )
    return cursor.lastrowid


def _make_web_evidence(
    conn, topic_id, url="http://example.com", source_domain="example.com"
) -> int:
    cursor = conn.execute(
        """INSERT INTO WebEvidence (origin_topic_id, query_text, title, snippet, url,
           source_domain, result_rank, search_provider, search_role)
           VALUES (?, 'q', 't', 's', ?, ?, 1, 'test', 'test')""",
        (topic_id, url, source_domain),
    )
    return cursor.lastrowid


def _make_code_evidence(conn, topic_id, success=True) -> int:
    cursor = conn.execute(
        """INSERT INTO CodeEvidence (
            origin_topic_id, hypothesis, source_code, stdout, stderr, exit_code,
            execution_time_s, iterations, success, requesting_role, summary
        )
        VALUES (?, 'solver run', 'lp artifact', '', '', ?, 0.01, 1, ?, 'or_solver', 'solver result')""",
        (topic_id, 0 if success else 1, int(success)),
    )
    return cursor.lastrowid


def _make_ledger(
    conn,
    topic_id,
    entity_id=1,
    attribute_id=1,
    domain_score=0.8,
    review_status=None,
    contested_since_round=None,
    source_ref="[W1]",
) -> int:
    # Ensure entity and attribute exist (use unique names per id)
    conn.execute(
        "INSERT OR IGNORE INTO LedgerEntity (id, topic_id, canonical_name) VALUES (?, ?, ?)",
        (entity_id, topic_id, f"Entity_{entity_id}"),
    )
    conn.execute(
        "INSERT OR IGNORE INTO LedgerAttribute (id, topic_id, canonical_name) VALUES (?, ?, ?)",
        (attribute_id, topic_id, f"Attr_{attribute_id}"),
    )
    cursor = conn.execute(
        """INSERT INTO Ledger (topic_id, entity_id, attribute_id, value,
           value_numeric_min, value_numeric_max,
           unit, normalized_timeframe, entry_type, source_ref, source_domain,
           domain_score, review_status, contested_since_round)
           VALUES (?, ?, ?, '1.0', 1.0, 1.0, 'USD', 'NONE', 'web_evidence', ?, 'example.com', ?, ?, ?)""",
        (
            topic_id,
            entity_id,
            attribute_id,
            source_ref,
            domain_score,
            review_status,
            contested_since_round,
        ),
    )
    return cursor.lastrowid


def _make_edge(
    conn,
    topic_id,
    source_id,
    source_type,
    target_id,
    target_type,
    relation,
    justification_group="default",
) -> int:
    cursor = conn.execute(
        """INSERT INTO KnowledgeEdge
           (topic_id, source_id, source_type, target_id, target_type, relation, justification_group)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            topic_id,
            source_id,
            source_type,
            target_id,
            target_type,
            relation,
            justification_group,
        ),
    )
    return cursor.lastrowid


# ---------------------------------------------------------------------------
# Test 1: KnowledgeEdge CRUD
# ---------------------------------------------------------------------------


def test_knowledge_edge_insert_and_query():
    """Basic insert + query via db functions."""
    with db.get_db() as conn:
        tid = _make_topic(conn)

    edge_id = db.insert_knowledge_edge(
        tid,
        1,
        "fact",
        2,
        "claim",
        "supports",
        justification_group="g1",
        confidence=0.9,
        created_by="test",
    )
    assert edge_id is not None

    edges = db.get_knowledge_edges(tid, source_id=1, source_type="fact")
    assert len(edges) == 1
    assert edges[0]["relation"] == "supports"
    assert edges[0]["justification_group"] == "g1"
    assert edges[0]["confidence"] == 0.9
    assert edges[0]["is_active"] == 1

    db.deactivate_knowledge_edge(edge_id)
    edges_active = db.get_knowledge_edges(tid, source_id=1, active_only=True)
    assert len(edges_active) == 0
    edges_all = db.get_knowledge_edges(tid, source_id=1, active_only=False)
    assert len(edges_all) == 1


# ---------------------------------------------------------------------------
# Test 2: Unique constraint
# ---------------------------------------------------------------------------


def test_knowledge_edge_unique_constraint():
    """Duplicate rejected, different justification_group allowed."""
    with db.get_db() as conn:
        tid = _make_topic(conn)

    eid1 = db.insert_knowledge_edge(tid, 1, "fact", 2, "claim", "supports", "g1")
    assert eid1 is not None

    # Same edge -> None (duplicate)
    eid2 = db.insert_knowledge_edge(tid, 1, "fact", 2, "claim", "supports", "g1")
    assert eid2 is None

    # Different justification_group -> OK
    eid3 = db.insert_knowledge_edge(tid, 1, "fact", 2, "claim", "supports", "g2")
    assert eid3 is not None


# ---------------------------------------------------------------------------
# Test 3: Backfill from support_fact_ids_json
# ---------------------------------------------------------------------------


def test_backfill_from_support_fact_ids_json():
    """Claims with support_fact_ids_json get supports edges on backfill."""
    with db.get_db() as conn:
        tid = _make_topic(conn)
        f1 = _make_fact(conn, tid, "fact1")
        f2 = _make_fact(conn, tid, "fact2")
        _make_claim(conn, tid, "claim1", support_fact_ids_json=json.dumps([f1, f2]))

    # Backfill runs during init_db() — re-run manually
    with db.get_db() as conn:
        db._backfill_knowledge_edges(conn)

    edges = db.get_knowledge_edges(tid, target_type="claim", relation="supports")
    source_ids = {e["source_id"] for e in edges}
    assert f1 in source_ids
    assert f2 in source_ids


# ---------------------------------------------------------------------------
# Test 4: Backfill from source_refs_json
# ---------------------------------------------------------------------------


def test_backfill_from_source_refs_json():
    """Facts with source_refs_json get derived_from edges on backfill."""
    with db.get_db() as conn:
        tid = _make_topic(conn)
        w1 = _make_web_evidence(conn, tid)
        conn.execute(
            "INSERT INTO Fact (topic_id, content, source, source_refs_json) VALUES (?, ?, 'test', ?)",
            (tid, "fact with W ref", json.dumps([f"W{w1}"])),
        )

    with db.get_db() as conn:
        db._backfill_knowledge_edges(conn)

    edges = db.get_knowledge_edges(
        tid, source_type="web_evidence", relation="derived_from"
    )
    assert len(edges) >= 1
    assert edges[0]["source_id"] == w1


# ---------------------------------------------------------------------------
# Test 5: Fact contradiction creates edge (unit test for dedup integration)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fact_contradiction_creates_edge():
    """check_fact_duplicate returns CONTRADICTION -> librarian creates conflict edge."""
    from unittest.mock import AsyncMock, patch

    with db.get_db() as conn:
        tid = _make_topic(conn)
        old_fact = _make_fact(conn, tid, "GDP grew 3.2%")

    candidate = {
        "id": 1,
        "candidate_text": "GDP contracted",
        "fact_stage": "synthesized",
        "candidate_type": "sourced_claim",
        "source_refs_json": '["W1"]',
    }
    review = {
        "decision": "accept",
        "reviewed_text": "GDP contracted 0.5%",
        "review_note": "",
        "evidence_note": "",
        "source_refs": ["W1"],
        "confidence_score": 8.0,
    }

    with (
        patch(
            "orbit_or.librarian_processor.api.get_fact_by_content", return_value=None
        ),
        patch(
            "orbit_or.librarian_processor.aget_embedding",
            new=AsyncMock(return_value=[0.1] * 768),
        ),
        patch(
            "orbit_or.librarian_processor.check_fact_duplicate",
            new=AsyncMock(return_value=("CONTRADICTION", old_fact)),
        ),
        patch(
            "orbit_or.librarian_processor.api.insert_fact", return_value=old_fact + 1
        ),
        patch("orbit_or.librarian_processor.api.insert_knowledge_edge") as mock_edge,
        patch("orbit_or.librarian_processor.api.update_fact_candidate_review"),
        patch("orbit_or.librarian_processor.api.update_fact_structured_columns"),
    ):
        from orbit_or.librarian_processor import apply_librarian_review

        result = await apply_librarian_review(tid, candidate, review)

    assert result["accepted_fact_id"] == old_fact + 1
    # Should have created conflicts_with + derived_from edges
    assert any("conflicts_with" in str(c) for c in mock_edge.call_args_list)


# ---------------------------------------------------------------------------
# Test 6: Fact update creates supersedes edge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fact_update_creates_supersedes_edge():
    """UPDATE -> supersedes edge + old fact superseded."""
    from unittest.mock import AsyncMock, patch

    with db.get_db() as conn:
        tid = _make_topic(conn)
        old_fact = _make_fact(conn, tid, "Revenue was $5.2B")

    candidate = {
        "id": 1,
        "candidate_text": "Revenue revised",
        "fact_stage": "synthesized",
        "candidate_type": "sourced_claim",
        "source_refs_json": '["W1"]',
    }
    review = {
        "decision": "accept",
        "reviewed_text": "Revenue was revised to $5.5B",
        "review_note": "",
        "evidence_note": "",
        "source_refs": ["W1"],
        "confidence_score": 9.0,
    }

    with (
        patch(
            "orbit_or.librarian_processor.api.get_fact_by_content", return_value=None
        ),
        patch(
            "orbit_or.librarian_processor.aget_embedding",
            new=AsyncMock(return_value=[0.1] * 768),
        ),
        patch(
            "orbit_or.librarian_processor.check_fact_duplicate",
            new=AsyncMock(return_value=("UPDATE", old_fact)),
        ),
        patch(
            "orbit_or.librarian_processor.api.insert_fact", return_value=old_fact + 1
        ),
        patch("orbit_or.librarian_processor.api.supersede_fact") as mock_supersede,
        patch("orbit_or.librarian_processor.api.insert_knowledge_edge") as mock_edge,
        patch("orbit_or.librarian_processor.api.update_fact_candidate_review"),
        patch("orbit_or.librarian_processor.api.update_fact_structured_columns"),
    ):
        from orbit_or.librarian_processor import apply_librarian_review

        result = await apply_librarian_review(tid, candidate, review)

    assert result["accepted_fact_id"] == old_fact + 1
    mock_supersede.assert_called_once_with(old_fact, old_fact + 1)
    assert any("supersedes" in str(c) for c in mock_edge.call_args_list)


# ---------------------------------------------------------------------------
# Test 7: JTMS sweep contests unsupported claim
# ---------------------------------------------------------------------------


def test_jtms_sweep_contests_unsupported_claim():
    """Claim with all support facts retired -> contested."""
    with db.get_db() as conn:
        tid = _make_topic(conn)
        f1 = _make_fact(conn, tid, "fact1", review_status="retired")
        c1 = _make_claim(conn, tid, "claim1", status="active")
        _make_edge(conn, tid, f1, "fact", c1, "claim", "supports", "g1")

    changes = jtms_sweep(tid, current_round=5)
    assert any(
        ch["type"] == "claim" and ch["id"] == c1 and ch["new_status"] == "contested"
        for ch in changes
    )


# ---------------------------------------------------------------------------
# Test 8: JTMS sweep recovers claim
# ---------------------------------------------------------------------------


def test_jtms_sweep_recovers_claim():
    """Contested claim gets new valid support -> back to active."""
    with db.get_db() as conn:
        tid = _make_topic(conn)
        f1 = _make_fact(conn, tid, "active fact", review_status=None)  # active
        c1 = _make_claim(
            conn, tid, "claim1", status="contested", contested_since_round=4
        )
        _make_edge(conn, tid, f1, "fact", c1, "claim", "supports", "g_new")

    # contested_since_round=4, current_round=5 -> only 1 round, not stale yet
    changes = jtms_sweep(tid, current_round=5)
    assert any(
        ch["type"] == "claim" and ch["id"] == c1 and ch["new_status"] == "active"
        for ch in changes
    )


def test_jtms_sweep_contests_claim_with_failed_code_evidence():
    """Failed solver/code evidence invalidates the justification group."""
    with db.get_db() as conn:
        tid = _make_topic(conn)
        ev1 = _make_code_evidence(conn, tid, success=False)
        c1 = _make_claim(conn, tid, "solver-backed claim", status="active")
        _make_edge(conn, tid, ev1, "code_evidence", c1, "claim", "supports", "solver")

    changes = jtms_sweep(tid, current_round=5)
    assert any(
        ch["type"] == "claim"
        and ch["id"] == c1
        and ch["new_status"] == "contested"
        for ch in changes
    )


def test_jtms_sweep_recovers_claim_with_successful_code_evidence():
    """Successful solver/code evidence can recover a contested solver claim."""
    with db.get_db() as conn:
        tid = _make_topic(conn)
        ev1 = _make_code_evidence(conn, tid, success=True)
        c1 = _make_claim(
            conn,
            tid,
            "solver-backed claim",
            status="contested",
            contested_since_round=4,
        )
        _make_edge(conn, tid, ev1, "code_evidence", c1, "claim", "supports", "solver")

    changes = jtms_sweep(tid, current_round=5)
    assert any(
        ch["type"] == "claim" and ch["id"] == c1 and ch["new_status"] == "active"
        for ch in changes
    )


# ---------------------------------------------------------------------------
# Test 9: JTMS sweep retires stale contested
# ---------------------------------------------------------------------------


def test_jtms_sweep_retires_stale_contested():
    """3 rounds contested -> retired."""
    with db.get_db() as conn:
        tid = _make_topic(conn)
        f1 = _make_fact(conn, tid, "contested fact", review_status="contested")
        # contested_since_round=2, current_round=5, diff=3 >= CONTESTED_ROUNDS_TO_RETIRE
        conn.execute("UPDATE Fact SET contested_since_round = 2 WHERE id = ?", (f1,))

    changes = jtms_sweep(tid, current_round=5)
    assert any(
        ch["type"] == "fact" and ch["id"] == f1 and ch["new_status"] == "retired"
        for ch in changes
    )


# ---------------------------------------------------------------------------
# Test 10: JTMS justification groups (OR-of-AND)
# ---------------------------------------------------------------------------


def test_jtms_sweep_justification_groups():
    """OR-of-AND: one group fails, other holds -> claim stays active."""
    with db.get_db() as conn:
        tid = _make_topic(conn)
        f_retired = _make_fact(conn, tid, "retired fact", review_status="retired")
        f_active = _make_fact(conn, tid, "active fact", review_status=None)
        c1 = _make_claim(conn, tid, "claim with 2 groups", status="active")
        # Group 1: retired fact (fails)
        _make_edge(conn, tid, f_retired, "fact", c1, "claim", "supports", "g1")
        # Group 2: active fact (holds)
        _make_edge(conn, tid, f_active, "fact", c1, "claim", "supports", "g2")

    changes = jtms_sweep(tid, current_round=5)
    # Claim should NOT be contested because g2 holds
    contested_claims = [
        ch
        for ch in changes
        if ch["type"] == "claim" and ch["new_status"] == "contested"
    ]
    assert len(contested_claims) == 0


# ---------------------------------------------------------------------------
# Test 11: Anti-resurrection (via dedup returning DUPLICATE for retired fact)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anti_resurrection_rejects_without_new_evidence():
    """Retired fact re-proposed without new evidence -> DUPLICATE (skip)."""
    from unittest.mock import AsyncMock, patch

    retired_fact = {
        "id": 42,
        "content": "Old retired fact",
        "distance": 0.2,
        "topic_id": 1,
        "source": "Librarian",
        "review_status": "retired",
        "source_refs_json": '["W1"]',
    }
    with patch("orbit_or.fact_dedup._db") as mock_db:
        mock_db.search_facts.return_value = [retired_fact]
        with patch(
            "orbit_or.fact_dedup._call_text",
            new=AsyncMock(return_value='["M42"]'),
        ):
            from orbit_or.fact_dedup import check_fact_duplicate

            action, matched_id = await check_fact_duplicate(
                1, "Old retired fact restated", [0.1] * 768
            )
    # LLM says duplicate -> DUPLICATE
    assert action == "DUPLICATE"
    assert matched_id == 42


# ---------------------------------------------------------------------------
# Test 12: Anti-resurrection allows with new evidence (UPDATE)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anti_resurrection_allows_with_new_evidence():
    """Retired fact re-proposed with new W evidence -> UPDATE (allowed)."""
    from unittest.mock import AsyncMock, patch

    retired_fact = {
        "id": 42,
        "content": "Old data point",
        "distance": 0.2,
        "topic_id": 1,
        "source": "Librarian",
        "review_status": "retired",
        "source_refs_json": '["W1"]',
    }
    with patch("orbit_or.fact_dedup._db") as mock_db:
        mock_db.search_facts.return_value = [retired_fact]
        with patch(
            "orbit_or.fact_dedup._call_text",
            new=AsyncMock(return_value='{"action":"UPDATE","key":"M42"}'),
        ):
            from orbit_or.fact_dedup import check_fact_duplicate

            action, matched_id = await check_fact_duplicate(
                1, "Revised data point from new source", [0.1] * 768
            )
    assert action == "UPDATE"
    assert matched_id == 42


# ---------------------------------------------------------------------------
# Test 13: Ledger indices creates supports edges
# ---------------------------------------------------------------------------


def test_ledger_indices_creates_supports_edges():
    """Unified extraction with ledger_indices -> L1->L2 edges."""
    # Test the db-level edge creation that would happen from evidence_parser
    with db.get_db() as conn:
        tid = _make_topic(conn)
        w1 = _make_web_evidence(conn, tid)
        led1 = _make_ledger(conn, tid, source_ref=f"[W{w1}]")

    # Simulate what evidence_parser does:
    # Create a fact candidate and then a supports edge from ledger to fact
    fact_cand_id = 999  # simulated
    edge_id = db.insert_knowledge_edge(
        tid,
        led1,
        "ledger",
        fact_cand_id,
        "fact",
        "supports",
        created_by="extract_all_from_evidence",
    )
    assert edge_id is not None

    edges = db.get_knowledge_edges(
        tid, source_id=led1, source_type="ledger", relation="supports"
    )
    assert len(edges) == 1
    assert edges[0]["target_id"] == fact_cand_id

    # Also test derived_from
    edge_id2 = db.insert_knowledge_edge(
        tid,
        w1,
        "web_evidence",
        led1,
        "ledger",
        "derived_from",
        created_by="extract_all_from_evidence",
    )
    assert edge_id2 is not None

    derived = db.get_knowledge_edges(
        tid, source_type="web_evidence", relation="derived_from"
    )
    assert len(derived) >= 1


# ---------------------------------------------------------------------------
# Test: Ledger conflict resolution
# ---------------------------------------------------------------------------


def test_jtms_sweep_ledger_conflicts():
    """Ledger conflicts_with -> lower domain_score becomes contested."""
    with db.get_db() as conn:
        tid = _make_topic(conn)
        led_high = _make_ledger(
            conn, tid, entity_id=1, attribute_id=1, domain_score=0.9
        )
        led_low = _make_ledger(conn, tid, entity_id=2, attribute_id=2, domain_score=0.5)
        _make_edge(conn, tid, led_high, "ledger", led_low, "ledger", "conflicts_with")

    changes = jtms_sweep(tid, current_round=1)
    assert any(
        ch["type"] == "ledger"
        and ch["id"] == led_low
        and ch["new_status"] == "contested"
        for ch in changes
    )


# ---------------------------------------------------------------------------
# Test: get_active_conflicts and get_claim_justification_groups
# ---------------------------------------------------------------------------


def test_get_active_conflicts():
    with db.get_db() as conn:
        tid = _make_topic(conn)
        _make_edge(conn, tid, 1, "fact", 2, "fact", "conflicts_with")
        _make_edge(conn, tid, 3, "ledger", 4, "ledger", "conflicts_with")

    fact_conflicts = db.get_active_conflicts(tid, "fact")
    assert len(fact_conflicts) == 1

    ledger_conflicts = db.get_active_conflicts(tid, "ledger")
    assert len(ledger_conflicts) == 1


def test_get_claim_justification_groups():
    with db.get_db() as conn:
        tid = _make_topic(conn)
        _make_edge(conn, tid, 10, "fact", 100, "claim", "supports", "g1")
        _make_edge(conn, tid, 11, "fact", 100, "claim", "supports", "g1")
        _make_edge(conn, tid, 20, "fact", 100, "claim", "supports", "g2")

    groups = db.get_claim_justification_groups(tid, 100)
    assert "g1" in groups
    assert "g2" in groups
    assert len(groups["g1"]) == 2
    assert len(groups["g2"]) == 1


# ---------------------------------------------------------------------------
# Round 2 tests: Coverage gaps
# ---------------------------------------------------------------------------


def test_jtms_sweep_empty_topic():
    """Sweep on empty topic returns no changes and does not crash."""
    with db.get_db() as conn:
        tid = _make_topic(conn)
    changes = jtms_sweep(tid, current_round=1)
    assert changes == []


def test_jtms_sweep_idempotent():
    """Running sweep twice produces no changes on the second run."""
    with db.get_db() as conn:
        tid = _make_topic(conn)
        f1 = _make_fact(conn, tid, "fact1", review_status="retired")
        c1 = _make_claim(conn, tid, "claim1", status="active")
        _make_edge(conn, tid, f1, "fact", c1, "claim", "supports", "g1")

    changes1 = jtms_sweep(tid, current_round=5)
    assert len(changes1) > 0

    changes2 = jtms_sweep(tid, current_round=5)
    assert changes2 == []


def test_jtms_sweep_domain_score_tie():
    """Tie in domain_score (diff < 0.1) -> no changes."""
    with db.get_db() as conn:
        tid = _make_topic(conn)
        led_a = _make_ledger(conn, tid, entity_id=1, attribute_id=1, domain_score=0.75)
        led_b = _make_ledger(conn, tid, entity_id=2, attribute_id=2, domain_score=0.80)
        _make_edge(conn, tid, led_a, "ledger", led_b, "ledger", "conflicts_with")

    changes = jtms_sweep(tid, current_round=1)
    ledger_changes = [ch for ch in changes if ch["type"] == "ledger"]
    assert len(ledger_changes) == 0


def test_jtms_sweep_all_groups_fail():
    """Claim with ALL justification groups failing -> contested."""
    with db.get_db() as conn:
        tid = _make_topic(conn)
        f1 = _make_fact(conn, tid, "retired1", review_status="retired")
        f2 = _make_fact(conn, tid, "retired2", review_status="retired")
        c1 = _make_claim(conn, tid, "claim with 2 groups", status="active")
        _make_edge(conn, tid, f1, "fact", c1, "claim", "supports", "g1")
        _make_edge(conn, tid, f2, "fact", c1, "claim", "supports", "g2")

    changes = jtms_sweep(tid, current_round=5)
    assert any(
        ch["type"] == "claim" and ch["id"] == c1 and ch["new_status"] == "contested"
        for ch in changes
    )


def test_jtms_sweep_stale_claim_retirement():
    """Claim contested for 3+ rounds -> retired."""
    with db.get_db() as conn:
        tid = _make_topic(conn)
        c1 = _make_claim(
            conn, tid, "stale claim", status="contested", contested_since_round=2
        )

    changes = jtms_sweep(tid, current_round=5)
    assert any(
        ch["type"] == "claim" and ch["id"] == c1 and ch["new_status"] == "retired"
        for ch in changes
    )


def test_jtms_sweep_stale_ledger_retirement():
    """Ledger contested for 3+ rounds with active conflict -> retired."""
    with db.get_db() as conn:
        tid = _make_topic(conn)
        led = _make_ledger(
            conn,
            tid,
            entity_id=1,
            attribute_id=1,
            domain_score=0.3,
            review_status="contested",
            contested_since_round=2,
        )
        # Active high-score conflict partner keeps this one contested
        led_winner = _make_ledger(
            conn,
            tid,
            entity_id=2,
            attribute_id=2,
            domain_score=0.9,
        )
        _make_edge(conn, tid, led_winner, "ledger", led, "ledger", "conflicts_with")

    changes = jtms_sweep(tid, current_round=5)
    assert any(
        ch["type"] == "ledger" and ch["id"] == led and ch["new_status"] == "retired"
        for ch in changes
    )


def test_jtms_sweep_multiple_conflicts_per_node():
    """Fact A (high score) conflicts with B and C (low score) -> both contested."""
    with db.get_db() as conn:
        tid = _make_topic(conn)
        # Create web evidence for domain score resolution
        w_high = _make_web_evidence(conn, tid, source_domain="reuters.com")
        w_low1 = _make_web_evidence(
            conn, tid, url="http://reddit.com/1", source_domain="reddit.com"
        )
        w_low2 = _make_web_evidence(
            conn, tid, url="http://reddit.com/2", source_domain="reddit.com"
        )
        f_a = _make_fact(conn, tid, "high-cred fact")
        f_b = _make_fact(conn, tid, "low-cred fact 1")
        f_c = _make_fact(conn, tid, "low-cred fact 2")
        # derived_from edges for score resolution
        _make_edge(conn, tid, w_high, "web_evidence", f_a, "fact", "derived_from")
        _make_edge(conn, tid, w_low1, "web_evidence", f_b, "fact", "derived_from")
        _make_edge(conn, tid, w_low2, "web_evidence", f_c, "fact", "derived_from")
        # conflict edges
        _make_edge(conn, tid, f_a, "fact", f_b, "fact", "conflicts_with")
        _make_edge(conn, tid, f_a, "fact", f_c, "fact", "conflicts_with")

    changes = jtms_sweep(tid, current_round=1)
    contested_ids = {
        ch["id"]
        for ch in changes
        if ch["type"] == "fact" and ch["new_status"] == "contested"
    }
    assert f_b in contested_ids
    assert f_c in contested_ids
    assert f_a not in contested_ids


def test_supersede_fact_db():
    """Direct DB test for supersede_fact()."""
    with db.get_db() as conn:
        tid = _make_topic(conn)
        old_id = _make_fact(conn, tid, "old fact")
        new_id = _make_fact(conn, tid, "new fact")

    db.supersede_fact(old_id, new_id)

    with db.get_db() as conn:
        row = conn.execute(
            "SELECT review_status, superseded_by FROM Fact WHERE id = ?", (old_id,)
        ).fetchone()
    assert row["review_status"] == "superseded"
    assert row["superseded_by"] == new_id


def test_backfill_idempotent():
    """Calling _backfill_knowledge_edges twice creates no duplicates."""
    with db.get_db() as conn:
        tid = _make_topic(conn)
        f1 = _make_fact(conn, tid, "fact1")
        _make_claim(conn, tid, "claim1", support_fact_ids_json=json.dumps([f1]))
        db._backfill_knowledge_edges(conn)

    count1 = len(db.get_knowledge_edges(tid, active_only=False))

    with db.get_db() as conn:
        db._backfill_knowledge_edges(conn)

    count2 = len(db.get_knowledge_edges(tid, active_only=False))
    assert count1 == count2


def test_jtms_sweep_fact_recovery():
    """Contested fact recovers when conflict counterpart is retired."""
    with db.get_db() as conn:
        tid = _make_topic(conn)
        # Fact A is contested, fact B (its conflict winner) is now retired
        f_a = _make_fact(conn, tid, "contested fact", review_status="contested")
        conn.execute("UPDATE Fact SET contested_since_round = 3 WHERE id = ?", (f_a,))
        f_b = _make_fact(conn, tid, "retired winner", review_status="retired")
        _make_edge(conn, tid, f_a, "fact", f_b, "fact", "conflicts_with")
        # Give f_a some active support
        w1 = _make_web_evidence(conn, tid)
        _make_edge(conn, tid, w1, "web_evidence", f_a, "fact", "derived_from")

    changes = jtms_sweep(tid, current_round=5)
    assert any(
        ch["type"] == "fact" and ch["id"] == f_a and ch["new_status"] == "active"
        for ch in changes
    )


def test_jtms_recovery_respects_score_order():
    """When both sides of a conflict are contested, only the higher-scored one recovers."""
    with db.get_db() as conn:
        tid = _make_topic(conn)
        # Both facts are contested; high_score should recover, low_score should not
        w_high = _make_web_evidence(conn, tid, source_domain="reuters.com")
        w_low = _make_web_evidence(
            conn, tid, url="http://reddit.com/x", source_domain="reddit.com"
        )
        f_high = _make_fact(conn, tid, "high score fact", review_status="contested")
        conn.execute(
            "UPDATE Fact SET contested_since_round = 4 WHERE id = ?", (f_high,)
        )
        f_low = _make_fact(conn, tid, "low score fact", review_status="contested")
        conn.execute("UPDATE Fact SET contested_since_round = 4 WHERE id = ?", (f_low,))
        # derived_from edges for score
        _make_edge(conn, tid, w_high, "web_evidence", f_high, "fact", "derived_from")
        _make_edge(conn, tid, w_low, "web_evidence", f_low, "fact", "derived_from")
        # conflict between them
        _make_edge(conn, tid, f_high, "fact", f_low, "fact", "conflicts_with")

    changes = jtms_sweep(tid, current_round=5)
    recovered = {
        ch["id"]
        for ch in changes
        if ch["type"] == "fact" and ch["new_status"] == "active"
    }
    # High score recovers
    assert f_high in recovered
    # Low score does NOT recover (blocked by recovered high)
    assert f_low not in recovered


def test_jtms_ledger_recovery_respects_score_order():
    """When both sides of a ledger conflict are contested, only higher-scored one recovers."""
    with db.get_db() as conn:
        tid = _make_topic(conn)
        led_high = _make_ledger(
            conn,
            tid,
            entity_id=1,
            attribute_id=1,
            domain_score=0.9,
            review_status="contested",
            contested_since_round=4,
        )
        led_low = _make_ledger(
            conn,
            tid,
            entity_id=2,
            attribute_id=2,
            domain_score=0.3,
            review_status="contested",
            contested_since_round=4,
        )
        _make_edge(conn, tid, led_high, "ledger", led_low, "ledger", "conflicts_with")

    changes = jtms_sweep(tid, current_round=5)
    recovered = {
        ch["id"]
        for ch in changes
        if ch["type"] == "ledger" and ch["new_status"] == "active"
    }
    assert led_high in recovered
    assert led_low not in recovered
