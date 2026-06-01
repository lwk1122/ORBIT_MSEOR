"""Tests for the Ledger schema, normalization, CRUD, and seeding."""

import os
from unittest.mock import AsyncMock, patch

import pytest

os.environ["TESTING"] = "1"

from orbit_or import db
from orbit_or.ledger import (
    ALIAS_CONFIRMATION_THRESHOLD,
    auto_generate_conflict_edges,
    normalize_timeframe,
    normalize_value,
    resolve_entity,
    add_entity_with_aliases,
    decontextualize_ledger_entry,
    expand_abbreviations,
    seed_ledger_from_topic,
    timeframe_to_interval,
    intervals_overlap,
    parse_time_field,
    normalize_and_upsert,
)


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """Each test gets a fresh database."""
    test_db = str(tmp_path / "test_ledger.db")
    monkeypatch.setattr(db, "get_db_path", lambda: test_db)
    db.init_db()
    yield


def _make_topic(summary="Test topic", detail="Detail") -> int:
    with db.get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO Topic (summary, detail) VALUES (?, ?)",
            (summary, detail),
        )
        return cursor.lastrowid


def _make_subtopic(topic_id: int, summary="Sub") -> int:
    with db.get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO Subtopic (topic_id, summary, detail) VALUES (?, ?, ?)",
            (topic_id, summary, "detail"),
        )
        return cursor.lastrowid


# ---------------------------------------------------------------------------
# 1. Schema
# ---------------------------------------------------------------------------


def test_ledger_tables_exist():
    expected = {
        "LedgerEntity",
        "LedgerEntityAlias",
        "LedgerAttribute",
        "LedgerAttributeAlias",
        "Ledger",
        "LedgerPending",
    }
    with db.get_db() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {r["name"] for r in rows}
    assert expected.issubset(names)


# ---------------------------------------------------------------------------
# 2. Entity CRUD + UNIQUE
# ---------------------------------------------------------------------------


def test_entity_create_and_unique():
    tid = _make_topic()
    eid1 = db.create_ledger_entity(tid, "MUFG", "institution")
    eid2 = db.create_ledger_entity(tid, "MUFG", "institution")
    assert eid1 == eid2  # INSERT OR IGNORE returns existing id

    entity = db.get_ledger_entity(eid1)
    assert entity["canonical_name"] == "MUFG"
    assert entity["entity_type"] == "institution"

    entities = db.get_ledger_entities(tid)
    assert len(entities) == 1


# ---------------------------------------------------------------------------
# 3. Entity alias confirmed lookup
# ---------------------------------------------------------------------------


def test_entity_alias_confirmed_lookup():
    tid = _make_topic()
    eid = add_entity_with_aliases(
        tid, "MUFG", "institution", ["mufg bank"], confirmed=True
    )
    resolved = resolve_entity("mufg bank", tid)
    assert resolved == eid


# ---------------------------------------------------------------------------
# 4. Entity alias promotion
# ---------------------------------------------------------------------------


def test_entity_alias_promotion():
    tid = _make_topic()
    eid = db.create_ledger_entity(tid, "PBOC", "institution")
    db.create_ledger_entity_alias(
        eid, "people's bank of china", confirmed=False, match_count=1
    )

    # First resolution: not yet confirmed
    result = db.lookup_entity_alias("people's bank of china", tid)
    assert result["confirmed"] == 0

    # Simulate match_count reaching threshold
    for _ in range(ALIAS_CONFIRMATION_THRESHOLD - 1):
        resolve_entity("people's bank of china", tid)

    result = db.lookup_entity_alias("people's bank of china", tid)
    assert result["confirmed"] == 1


# ---------------------------------------------------------------------------
# 5. Timeframe normalizer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Q4 2026", "2026-Q4"),
        ("2026 Q4", "2026-Q4"),
        ("q1 2027", "2027-Q1"),
        ("H2 2026", "2026-H2"),
        ("2026 H1", "2026-H1"),
        ("May 2027", "2027-05"),
        ("January 2026", "2026-01"),
        ("end of 2026", "2026-Q4"),
        ("late 2027", "2027-Q4"),
        ("early 2026", "2026-Q1"),
        ("start of 2027", "2027-Q1"),
        ("mid-2027", "2027-Q2"),
        ("mid 2026", "2026-Q2"),
        ("2027", "2027"),
        ("", ""),
        ("no match here", ""),
    ],
)
def test_normalize_timeframe(raw, expected):
    assert normalize_timeframe(raw) == expected


# ---------------------------------------------------------------------------
# 6. Value normalizer — scale invariance
# ---------------------------------------------------------------------------


def test_normalize_value_scale_invariance():
    r1 = normalize_value("1.5 Trillion USD")
    r2 = normalize_value("1500 Billion USD")
    assert r1[0] == pytest.approx(1.5e12)
    assert r1[1] == pytest.approx(1.5e12)
    assert r1[2] == "USD"
    assert r1[0] == pytest.approx(r2[0])
    assert r1[1] == pytest.approx(r2[1])
    assert r1[2] == r2[2]


# ---------------------------------------------------------------------------
# 7. Value normalizer — ranges
# ---------------------------------------------------------------------------


def test_normalize_value_ranges():
    lo, hi, unit = normalize_value("2.5%-3.0%")
    assert lo == pytest.approx(2.5)
    assert hi == pytest.approx(3.0)
    assert unit == "%"

    # One-sided range
    lo2, hi2, unit2 = normalize_value("up to 50 billion")
    assert lo2 is None
    assert hi2 == pytest.approx(5e10)

    # Simple number
    lo3, hi3, unit3 = normalize_value("6.8000")
    assert lo3 == pytest.approx(6.8)
    assert hi3 == pytest.approx(6.8)
    assert unit3 is None


# ---------------------------------------------------------------------------
# 8. Ledger upsert — same source updates
# ---------------------------------------------------------------------------


def test_ledger_upsert_same_source_updates():
    tid = _make_topic()
    sid = _make_subtopic(tid)
    eid = db.create_ledger_entity(tid, "MUFG", "institution")
    aid = db.create_ledger_attribute(tid, "Forecast", "numeric")

    lid1, inserted1 = db.upsert_ledger_entry(
        tid,
        sid,
        eid,
        aid,
        "6.8000",
        6.8,
        6.8,
        "USD/CNY",
        "2026-Q4",
        "web_evidence",
        "[W12]",
    )
    assert inserted1 is True

    # Same UNIQUE key, different value → UPDATE
    lid2, inserted2 = db.upsert_ledger_entry(
        tid,
        sid,
        eid,
        aid,
        "6.9000",
        6.9,
        6.9,
        "USD/CNY",
        "2026-Q4",
        "web_evidence",
        "[W12]",
    )
    assert inserted2 is False
    assert lid1 == lid2

    entries = db.get_ledger_entries(tid)
    assert len(entries) == 1
    assert entries[0]["value"] == "6.9000"


# ---------------------------------------------------------------------------
# 9. Ledger upsert — multi-source
# ---------------------------------------------------------------------------


def test_ledger_upsert_multi_source():
    tid = _make_topic()
    eid = db.create_ledger_entity(tid, "MUFG", "institution")
    aid = db.create_ledger_attribute(tid, "Forecast", "numeric")

    db.upsert_ledger_entry(
        tid,
        None,
        eid,
        aid,
        "6.8000",
        6.8,
        6.8,
        "USD/CNY",
        "2026-Q4",
        "web_evidence",
        "[W12]",
    )
    db.upsert_ledger_entry(
        tid,
        None,
        eid,
        aid,
        "7.1000",
        7.1,
        7.1,
        "USD/CNY",
        "2026-Q4",
        "web_evidence",
        "[W15]",
    )

    entries = db.get_ledger_entries(tid)
    assert len(entries) == 2


# ---------------------------------------------------------------------------
# 10. Decontextualize
# ---------------------------------------------------------------------------


def test_decontextualize():
    result = decontextualize_ledger_entry(
        "MUFG", "Forecast", "6.8000 USD/CNY", "2026-Q4", "[W12]", "reuters.com"
    )
    assert "MUFG" in result
    assert "Forecast" in result
    assert "6.8000 USD/CNY" in result
    assert "2026-Q4" in result
    assert "[W12]" in result
    assert "reuters.com" in result


# ---------------------------------------------------------------------------
# 11. Abbreviation expansion
# ---------------------------------------------------------------------------


def test_expand_abbreviations():
    tid = _make_topic()
    add_entity_with_aliases(
        tid, "People's Bank of China", "institution", ["PBOC", "pboc"], confirmed=True
    )

    result = expand_abbreviations("PBOC intervention in forex markets", tid)
    assert "People's Bank of China" in result
    assert "PBOC" not in result


# ---------------------------------------------------------------------------
# 12. Pending TTL
# ---------------------------------------------------------------------------


def test_pending_ttl():
    tid = _make_topic()
    pid = db.create_ledger_pending(
        tid, None, "some raw text", "[W5]", "[6.8]", "entity", 3, 5
    )
    assert pid > 0

    # Active at round 4
    active = db.get_active_ledger_pending(tid, 4)
    assert len(active) == 1
    assert active[0]["raw_text"] == "some raw text"

    # Active at round 5 (boundary)
    active = db.get_active_ledger_pending(tid, 5)
    assert len(active) == 1

    # Expired at round 6
    expired_count = db.expire_ledger_pending(6)
    assert expired_count == 1

    active = db.get_active_ledger_pending(tid, 6)
    assert len(active) == 0


# ---------------------------------------------------------------------------
# 13. Seed ledger (mock MiniMax)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_seed_ledger():
    tid = _make_topic("USD/CNY Exchange Rate", "Forecast discussion")
    sid = _make_subtopic(tid)

    mock_response = {
        "entities": [
            {
                "name": "MUFG",
                "type": "institution",
                "aliases": ["mufg bank", "mitsubishi ufj"],
            },
            {
                "name": "PBOC",
                "type": "institution",
                "aliases": ["people's bank of china"],
            },
        ],
        "attributes": [
            {
                "name": "Forecast",
                "value_type": "numeric",
                "aliases": ["forecast target", "prediction"],
            },
        ],
    }

    with patch(
        "orbit_or.master_graph.ask_control_model",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        await seed_ledger_from_topic(
            tid, sid, "USD/CNY", "Detail", "Sub summary", "Sub detail"
        )

    entities = db.get_ledger_entities(tid)
    assert len(entities) == 2
    names = {e["canonical_name"] for e in entities}
    assert "MUFG" in names
    assert "PBOC" in names

    # Check aliases are confirmed
    alias = db.lookup_entity_alias("mufg bank", tid)
    assert alias is not None
    assert alias["confirmed"] == 1

    attrs = db.get_ledger_attributes(tid)
    assert len(attrs) == 1
    assert attrs[0]["canonical_name"] == "Forecast"


# ---------------------------------------------------------------------------
# 14. Entities are topic-scoped (shared across subtopics)
# ---------------------------------------------------------------------------


def test_entities_topic_scoped():
    tid = _make_topic()
    sid1 = _make_subtopic(tid, "Sub 1")
    sid2 = _make_subtopic(tid, "Sub 2")

    eid = db.create_ledger_entity(tid, "MUFG", "institution")
    aid = db.create_ledger_attribute(tid, "Forecast", "numeric")

    # Both subtopics can use the same entity/attribute
    db.upsert_ledger_entry(
        tid,
        sid1,
        eid,
        aid,
        "6.8",
        6.8,
        6.8,
        "USD/CNY",
        "2026-Q4",
        "agent_claim",
        "agent:dreamer",
    )
    db.upsert_ledger_entry(
        tid,
        sid2,
        eid,
        aid,
        "7.0",
        7.0,
        7.0,
        "USD/CNY",
        "2027-Q1",
        "agent_claim",
        "agent:scientist",
    )

    all_entries = db.get_ledger_entries(tid)
    assert len(all_entries) == 2

    s1_entries = db.get_ledger_entries(tid, subtopic_id=sid1)
    assert len(s1_entries) == 1
    assert s1_entries[0]["value"] == "6.8"


# ---------------------------------------------------------------------------
# 15. LedgerEdge — schema
# ---------------------------------------------------------------------------


def test_edge_table_exists():
    with db.get_db() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='LedgerEdge'"
        ).fetchall()
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# 16. LedgerEdge — create
# ---------------------------------------------------------------------------


def test_create_edge():
    tid = _make_topic()
    eid = db.create_ledger_entity(tid, "MUFG", "institution")
    aid = db.create_ledger_attribute(tid, "Forecast", "numeric")
    lid1, _ = db.upsert_ledger_entry(
        tid,
        None,
        eid,
        aid,
        "6.8",
        6.8,
        6.8,
        "USD/CNY",
        "2026-Q4",
        "web_evidence",
        "[W12]",
    )
    lid2, _ = db.upsert_ledger_entry(
        tid,
        None,
        eid,
        aid,
        "7.1",
        7.1,
        7.1,
        "USD/CNY",
        "2026-Q4",
        "web_evidence",
        "[W15]",
    )
    edge_id = db.create_ledger_edge(
        tid, lid1, lid2, "conflicts_with", created_by="auto"
    )
    assert edge_id is not None
    assert isinstance(edge_id, int)


# ---------------------------------------------------------------------------
# 17. LedgerEdge — unique constraint
# ---------------------------------------------------------------------------


def test_edge_unique_constraint():
    tid = _make_topic()
    eid = db.create_ledger_entity(tid, "MUFG", "institution")
    aid = db.create_ledger_attribute(tid, "Forecast", "numeric")
    lid1, _ = db.upsert_ledger_entry(
        tid,
        None,
        eid,
        aid,
        "6.8",
        6.8,
        6.8,
        "USD/CNY",
        "2026-Q4",
        "web_evidence",
        "[W12]",
    )
    lid2, _ = db.upsert_ledger_entry(
        tid,
        None,
        eid,
        aid,
        "7.1",
        7.1,
        7.1,
        "USD/CNY",
        "2026-Q4",
        "web_evidence",
        "[W15]",
    )
    first = db.create_ledger_edge(tid, lid1, lid2, "conflicts_with")
    assert first is not None
    second = db.create_ledger_edge(tid, lid1, lid2, "conflicts_with")
    assert second is None


# ---------------------------------------------------------------------------
# 18. LedgerEdge — get by entry
# ---------------------------------------------------------------------------


def test_get_edges_by_entry():
    tid = _make_topic()
    eid = db.create_ledger_entity(tid, "MUFG", "institution")
    aid = db.create_ledger_attribute(tid, "Forecast", "numeric")
    lid1, _ = db.upsert_ledger_entry(
        tid,
        None,
        eid,
        aid,
        "6.8",
        6.8,
        6.8,
        "USD/CNY",
        "2026-Q4",
        "web_evidence",
        "[W12]",
    )
    lid2, _ = db.upsert_ledger_entry(
        tid,
        None,
        eid,
        aid,
        "7.1",
        7.1,
        7.1,
        "USD/CNY",
        "2026-Q4",
        "web_evidence",
        "[W15]",
    )
    db.create_ledger_edge(tid, lid1, lid2, "conflicts_with")

    # Query from either direction
    edges = db.get_ledger_edges(tid, entry_id=lid1)
    assert len(edges) == 1
    edges = db.get_ledger_edges(tid, entry_id=lid2)
    assert len(edges) == 1


# ---------------------------------------------------------------------------
# 19. LedgerEdge — get by type
# ---------------------------------------------------------------------------


def test_get_edges_by_type():
    tid = _make_topic()
    eid = db.create_ledger_entity(tid, "MUFG", "institution")
    aid = db.create_ledger_attribute(tid, "Forecast", "numeric")
    lid1, _ = db.upsert_ledger_entry(
        tid,
        None,
        eid,
        aid,
        "6.8",
        6.8,
        6.8,
        "USD/CNY",
        "2026-Q4",
        "web_evidence",
        "[W12]",
    )
    lid2, _ = db.upsert_ledger_entry(
        tid,
        None,
        eid,
        aid,
        "7.1",
        7.1,
        7.1,
        "USD/CNY",
        "2026-Q4",
        "web_evidence",
        "[W15]",
    )
    db.create_ledger_edge(tid, lid1, lid2, "conflicts_with")
    db.create_ledger_edge(tid, lid1, lid2, "supports")

    conflicts = db.get_ledger_edges(tid, edge_type="conflicts_with")
    assert len(conflicts) == 1
    supports = db.get_ledger_edges(tid, edge_type="supports")
    assert len(supports) == 1
    all_edges = db.get_ledger_edges(tid)
    assert len(all_edges) == 2


# ---------------------------------------------------------------------------
# 20. auto_generate_conflict_edges
# ---------------------------------------------------------------------------


def test_auto_generate_conflict_edges():
    tid = _make_topic()
    eid = db.create_ledger_entity(tid, "MUFG", "institution")
    aid = db.create_ledger_attribute(tid, "Forecast", "numeric")

    # Two entries with same entity/attribute/timeframe but different values → contested
    db.upsert_ledger_entry(
        tid,
        None,
        eid,
        aid,
        "6.8",
        6.8,
        6.8,
        "USD/CNY",
        "2026-Q4",
        "web_evidence",
        "[W12]",
    )
    db.upsert_ledger_entry(
        tid,
        None,
        eid,
        aid,
        "7.1",
        7.1,
        7.1,
        "USD/CNY",
        "2026-Q4",
        "web_evidence",
        "[W15]",
    )

    new_count = auto_generate_conflict_edges(tid)
    assert new_count == 1

    edges = db.get_ledger_edges(tid, edge_type="conflicts_with")
    assert len(edges) == 1

    # Idempotent: running again should create 0 new edges
    assert auto_generate_conflict_edges(tid) == 0


# ---------------------------------------------------------------------------
# 21. P3-7: delete_ledger_pending returns rowcount
# ---------------------------------------------------------------------------


def test_delete_ledger_pending_returns_rowcount():
    tid = _make_topic()
    pid = db.create_ledger_pending(tid, None, "raw", "[W1]", None, "entity", 0, 5)
    assert db.delete_ledger_pending(pid) == 1
    # Already deleted — returns 0
    assert db.delete_ledger_pending(pid) == 0


# ---------------------------------------------------------------------------
# 22. QA-P5-4: delete_ledger_edge requires topic_id
# ---------------------------------------------------------------------------


def test_delete_edge_requires_topic_id():
    tid1 = _make_topic("Topic 1")
    tid2 = _make_topic("Topic 2")
    eid = db.create_ledger_entity(tid1, "MUFG", "institution")
    aid = db.create_ledger_attribute(tid1, "Forecast", "numeric")
    lid1, _ = db.upsert_ledger_entry(
        tid1,
        None,
        eid,
        aid,
        "6.8",
        6.8,
        6.8,
        "USD/CNY",
        "2026-Q4",
        "web_evidence",
        "[W12]",
    )
    lid2, _ = db.upsert_ledger_entry(
        tid1,
        None,
        eid,
        aid,
        "7.1",
        7.1,
        7.1,
        "USD/CNY",
        "2026-Q4",
        "web_evidence",
        "[W15]",
    )
    edge_id = db.create_ledger_edge(tid1, lid1, lid2, "conflicts_with")
    assert edge_id is not None
    # Wrong topic_id → should not delete
    assert db.delete_ledger_edge(edge_id, tid2) is False
    # Correct topic_id → should delete
    assert db.delete_ledger_edge(edge_id, tid1) is True


# ---------------------------------------------------------------------------
# 23. QA-P5-5: contested pairs include NULL timeframes
# ---------------------------------------------------------------------------


def test_contested_excludes_empty_timeframe():
    """Entries with empty timeframes should not be grouped as contested."""
    tid = _make_topic()
    eid = db.create_ledger_entity(tid, "MUFG", "institution")
    aid = db.create_ledger_attribute(tid, "Forecast", "numeric")
    # Insert entries with empty timeframe — not meaningful for conflict detection
    db.upsert_ledger_entry(
        tid,
        None,
        eid,
        aid,
        "6.8",
        6.8,
        6.8,
        "USD/CNY",
        "",
        "web_evidence",
        "[W1]",
    )
    db.upsert_ledger_entry(
        tid,
        None,
        eid,
        aid,
        "7.1",
        7.1,
        7.1,
        "USD/CNY",
        "",
        "web_evidence",
        "[W2]",
    )
    # Empty timeframes are excluded from conflict detection (COALESCE handles NULL defensively)
    contested = db.get_contested_ledger_pairs(tid)
    assert len(contested) == 0

    # But entries WITH timeframes are still detected
    db.upsert_ledger_entry(
        tid,
        None,
        eid,
        aid,
        "6.8",
        6.8,
        6.8,
        "USD/CNY",
        "2026-Q4",
        "web_evidence",
        "[W3]",
    )
    db.upsert_ledger_entry(
        tid,
        None,
        eid,
        aid,
        "7.1",
        7.1,
        7.1,
        "USD/CNY",
        "2026-Q4",
        "web_evidence",
        "[W4]",
    )
    contested = db.get_contested_ledger_pairs(tid)
    assert len(contested) == 1


# ---------------------------------------------------------------------------
# 24. QA-R2-3: expand_abbreviations with symbol aliases
# ---------------------------------------------------------------------------


def test_expand_abbreviations_symbol_alias():
    tid = _make_topic()
    add_entity_with_aliases(tid, "Apple Inc", "company", ["$AAPL"], confirmed=True)
    result = expand_abbreviations("The $AAPL stock surged today", tid)
    assert "Apple Inc" in result
    assert "$AAPL" not in result


# ---------------------------------------------------------------------------
# 25. P5-1: bulk_create_ledger_edges
# ---------------------------------------------------------------------------


def test_bulk_create_edges():
    tid = _make_topic()
    eid = db.create_ledger_entity(tid, "MUFG", "institution")
    aid = db.create_ledger_attribute(tid, "Forecast", "numeric")
    lid1, _ = db.upsert_ledger_entry(
        tid,
        None,
        eid,
        aid,
        "6.8",
        6.8,
        6.8,
        "USD/CNY",
        "2026-Q4",
        "web_evidence",
        "[W12]",
    )
    lid2, _ = db.upsert_ledger_entry(
        tid,
        None,
        eid,
        aid,
        "7.1",
        7.1,
        7.1,
        "USD/CNY",
        "2026-Q4",
        "web_evidence",
        "[W15]",
    )
    lid3, _ = db.upsert_ledger_entry(
        tid,
        None,
        eid,
        aid,
        "7.3",
        7.3,
        7.3,
        "USD/CNY",
        "2026-Q4",
        "web_evidence",
        "[W18]",
    )
    count = db.bulk_create_ledger_edges(
        tid,
        [(lid1, lid2), (lid1, lid3), (lid2, lid3)],
        "conflicts_with",
        created_by="auto",
    )
    assert count == 3
    edges = db.get_ledger_edges(tid, edge_type="conflicts_with")
    assert len(edges) == 3
    # Idempotent
    assert (
        db.bulk_create_ledger_edges(
            tid, [(lid1, lid2)], "conflicts_with", created_by="auto"
        )
        == 0
    )


# ---------------------------------------------------------------------------
# 26. PY-5: normalize_and_upsert single connection (functional test)
# ---------------------------------------------------------------------------


def test_normalize_and_upsert_single_connection():
    """Verify normalize_and_upsert still produces correct results with single-conn refactor."""
    from orbit_or.ledger import normalize_and_upsert

    tid = _make_topic()
    add_entity_with_aliases(
        tid, "MUFG", "institution", ["mufg", "mufg bank"], confirmed=True
    )
    from orbit_or.ledger import add_attribute_with_aliases

    add_attribute_with_aliases(tid, "Forecast", "numeric", ["forecast"], confirmed=True)

    lid, status = normalize_and_upsert(
        topic_id=tid,
        raw_entity="mufg",
        raw_attribute="forecast",
        raw_value="6.8000",
        raw_timeframe="Q4 2026",
        entry_type="web_evidence",
        source_ref="[W12]",
        source_domain="reuters.com",
        created_by="test",
    )
    assert lid is not None
    assert status == "inserted"

    # Verify data
    entries = db.get_ledger_entries(tid)
    assert len(entries) == 1
    assert entries[0]["value"] == "6.8000"
    assert entries[0]["normalized_timeframe"] == "2026-Q4"


# ---------------------------------------------------------------------------
# 27. P2-2: _has_uncited_financial_numbers multi-sentence
# ---------------------------------------------------------------------------


def test_uncited_numbers_multi_sentence():
    from orbit_or.server import _has_uncited_financial_numbers

    # Number in first sentence is cited, but second is not → should detect
    content = "The rate is 6.8000 [F1]. Another source says 6.8000 without citation."
    uncited = _has_uncited_financial_numbers(content)
    assert "6.8000" in uncited

    # Number only in cited sentence → should NOT be uncited
    content2 = "The rate is 6.8000 [F1]."
    assert _has_uncited_financial_numbers(content2) == []


# ---------------------------------------------------------------------------
# 28. P2-3: _unwrap_brackets preserves source brackets
# ---------------------------------------------------------------------------


def test_clerk_parser_preserves_source_brackets():
    from orbit_or.server import _unwrap_brackets

    # Single bracket layer is removed
    assert _unwrap_brackets("[W12]") == "W12"
    assert _unwrap_brackets("[1.5]") == "1.5"
    # No brackets → unchanged
    assert _unwrap_brackets("plain") == "plain"
    # Nested brackets preserved (not unwrapped since inner ] found before end)
    assert _unwrap_brackets("[[nested]]") == "[[nested]]"
    # Whitespace handling
    assert _unwrap_brackets("  [  spaced  ]  ") == "spaced"
    # Only opening bracket → no removal
    assert _unwrap_brackets("[partial") == "[partial"


# ---------------------------------------------------------------------------
# 29. P3-2: daemon tracks round number
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_daemon_tracks_round():
    from orbit_or.fact_daemon import FactDaemon

    daemon = FactDaemon(topic_id=1, subtopic_id=1)
    assert daemon.current_round == 0

    # Simulate clerk loop updating round
    daemon.current_round = max(daemon.current_round, 3)
    assert daemon.current_round == 3

    # Lower round should not decrease
    daemon.current_round = max(daemon.current_round, 1)
    assert daemon.current_round == 3


# ---------------------------------------------------------------------------
# 30. timeframe_to_interval parametrized
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,exp_from,exp_to",
    [
        ("Q1 2024", "2024-01-01", "2024-03-31"),
        ("Q3 2025", "2025-07-01", "2025-09-30"),
        ("Q4 2026", "2026-10-01", "2026-12-31"),
        ("q2 2023", "2023-04-01", "2023-06-30"),
        ("2025 Q1", "2025-01-01", "2025-03-31"),
        ("2027-Q4", "2027-10-01", "2027-12-31"),
        ("2026-Q4", "2026-10-01", "2026-12-31"),
        ("2024-Q2", "2024-04-01", "2024-06-30"),
        ("H1 2027", "2027-01-01", "2027-06-30"),
        ("H2 2026", "2026-07-01", "2026-12-31"),
        ("2027 H1", "2027-01-01", "2027-06-30"),
        ("2027-H1", "2027-01-01", "2027-06-30"),
        ("March 2025", "2025-03-01", "2025-03-31"),
        ("January 2024", "2024-01-01", "2024-01-31"),
        ("Sep 2026", "2026-09-01", "2026-09-30"),
        ("December 2023", "2023-12-01", "2023-12-31"),
        ("Jan-Mar 2024", "2024-01-01", "2024-03-31"),
        ("Jul-Sep 2025", "2025-07-01", "2025-09-30"),
        ("early 2024", "2024-01-01", "2024-03-31"),
        ("late 2024", "2024-10-01", "2024-12-31"),
        ("end of 2025", "2025-10-01", "2025-12-31"),
        ("mid-2024", "2024-04-01", "2024-09-30"),
        ("mid 2025", "2025-04-01", "2025-09-30"),
        ("by 2027", "2027-01-01", "2027-12-31"),
        ("through 2027", "2027-01-01", "2027-12-31"),
        ("after 2025", "2026-01-01", "2030-12-31"),
        ("since 2020", "2021-01-01", "2025-12-31"),
        ("2024-07-01", "2024-07-01", "2024-07-01"),
        ("2025-03", "2025-03-01", "2025-03-31"),
        ("2024-2026", "2024-01-01", "2026-12-31"),
        ("2025 to 2027", "2025-01-01", "2027-12-31"),
        ("2020s", "2020-01-01", "2029-12-31"),
        ("FY2025", "2025-01-01", "2025-12-31"),
        ("FY 2025", "2025-01-01", "2025-12-31"),
        ("2024", "2024-01-01", "2024-12-31"),
        ("2027", "2027-01-01", "2027-12-31"),
        ("", None, None),
        ("unknown", None, None),
        ("recent", None, None),
        ("ongoing", None, None),
        ("UNSPECIFIED", None, None),
    ],
)
def test_timeframe_to_interval(raw, exp_from, exp_to):
    got_from, got_to = timeframe_to_interval(raw)
    assert got_from == exp_from
    assert got_to == exp_to


# ---------------------------------------------------------------------------
# 31. intervals_overlap parametrized
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_a,raw_b,expected",
    [
        ("2027", "Q1 2027", True),
        ("2027", "H1 2027", True),
        ("2027", "Q4 2027", True),
        ("Q1 2027", "H1 2027", True),
        ("Q4 2027", "H2 2027", True),
        ("Q1 2027", "Q4 2027", False),
        ("H1 2027", "H2 2027", False),
        ("2026", "2027", False),
        ("Q4 2026", "Q1 2027", False),
        ("Q4 2026", "2027", False),
        ("2024-2026", "Q3 2025", True),
        ("2024-2026", "2027", False),
        ("2024-2026", "2026", True),
        ("2024-2026", "late 2026", True),
        ("2020s", "Q3 2025", True),
        ("2020s", "2030", False),
        ("after 2025", "2027", True),
        ("after 2025", "2024", False),
        ("", "", True),
        ("", "2027", True),
        ("2024", "", True),
        ("March 2025", "Q1 2025", True),
        ("March 2025", "Q2 2025", False),
        ("January 2025", "H1 2025", True),
        ("Jan-Mar 2024", "Q1 2024", True),
        ("Jul-Sep 2025", "Q3 2025", True),
        ("Jul-Sep 2025", "Q1 2025", False),
        ("2027", "2027", True),
        ("Q3 2025", "Q3 2025", True),
        ("2027-Q4", "2027", True),
        ("2027-H1", "Q1 2027", True),
        ("2026-Q4", "2027-Q1", False),
    ],
)
def test_intervals_overlap(raw_a, raw_b, expected):
    int_a = timeframe_to_interval(raw_a)
    int_b = timeframe_to_interval(raw_b)
    assert intervals_overlap(int_a, int_b) == expected


# ---------------------------------------------------------------------------
# 32. parse_time_field MM/DD/YYYY format
# ---------------------------------------------------------------------------


def test_parse_time_field_mmddyyyy():
    vf, vt = parse_time_field("01/01/2024-12/31/2024")
    assert vf == "2024-01-01"
    assert vt == "2024-12-31"

    vf2, vt2 = parse_time_field("NONE")
    assert vf2 is None
    assert vt2 is None

    vf3, vt3 = parse_time_field("Q3 2025")
    assert vf3 == "2025-07-01"
    assert vt3 == "2025-09-30"


# ---------------------------------------------------------------------------
# 33. upsert rejects agent_claim
# ---------------------------------------------------------------------------


def test_upsert_rejects_agent_claim():
    lid, status = normalize_and_upsert(
        topic_id=1,
        raw_entity="Test",
        raw_attribute="Attr",
        raw_value="100",
        entry_type="agent_claim",
        source_ref="M1",
    )
    assert lid is None
    assert status == "rejected"


# ---------------------------------------------------------------------------
# 34. upsert rejects low domain score
# ---------------------------------------------------------------------------


def test_upsert_rejects_low_domain_score():
    lid, status = normalize_and_upsert(
        topic_id=1,
        raw_entity="Test",
        raw_attribute="Attr",
        raw_value="100",
        entry_type="web_evidence",
        source_ref="W1",
        domain_score=0.3,
    )
    assert lid is None
    assert status == "rejected"


# ---------------------------------------------------------------------------
# 35. credibility: replace higher domain score
# ---------------------------------------------------------------------------


def test_credibility_replace():
    from orbit_or.ledger import add_attribute_with_aliases

    tid = _make_topic()
    add_entity_with_aliases(tid, "NVIDIA", "company", ["nvidia"], confirmed=True)
    add_attribute_with_aliases(tid, "Revenue", "numeric", ["revenue"], confirmed=True)

    # Insert base entry
    lid1, status1 = normalize_and_upsert(
        topic_id=tid,
        raw_entity="nvidia",
        raw_attribute="revenue",
        raw_value="2.697e10",
        raw_timeframe="2023",
        entry_type="web_evidence",
        source_ref="[W101]",
        domain_score=0.80,
        source_domain="wsj.com",
    )
    assert status1 == "inserted"

    # Higher credibility source with different value → REPLACE
    lid2, status2 = normalize_and_upsert(
        topic_id=tid,
        raw_entity="nvidia",
        raw_attribute="revenue",
        raw_value="2.698e10",
        raw_timeframe="2023",
        entry_type="web_evidence",
        source_ref="[W103]",
        domain_score=0.95,
        source_domain="sec.gov",
    )
    assert status2 == "replaced"
    assert lid2 == lid1  # same id, updated in place


# ---------------------------------------------------------------------------
# 36. credibility: discard lower domain score
# ---------------------------------------------------------------------------


def test_credibility_discard():
    from orbit_or.ledger import add_attribute_with_aliases

    tid = _make_topic()
    add_entity_with_aliases(tid, "NVIDIA", "company", ["nvidia"], confirmed=True)
    add_attribute_with_aliases(tid, "Revenue", "numeric", ["revenue"], confirmed=True)

    # Insert high-credibility entry
    normalize_and_upsert(
        topic_id=tid,
        raw_entity="nvidia",
        raw_attribute="revenue",
        raw_value="2.697e10",
        raw_timeframe="2023",
        entry_type="web_evidence",
        source_ref="[W101]",
        domain_score=0.95,
        source_domain="sec.gov",
    )

    # Lower credibility → DISCARD
    lid, status = normalize_and_upsert(
        topic_id=tid,
        raw_entity="nvidia",
        raw_attribute="revenue",
        raw_value="2.698e10",
        raw_timeframe="2023",
        entry_type="web_evidence",
        source_ref="[W104]",
        domain_score=0.70,
        source_domain="randomblog.com",
    )
    assert lid is None
    assert status == "discarded"


# ---------------------------------------------------------------------------
# 37. credibility: conflict similar scores different values
# ---------------------------------------------------------------------------


def test_credibility_conflict():
    from orbit_or.ledger import add_attribute_with_aliases

    tid = _make_topic()
    add_entity_with_aliases(tid, "NVIDIA", "company", ["nvidia"], confirmed=True)
    add_attribute_with_aliases(tid, "Revenue", "numeric", ["revenue"], confirmed=True)

    normalize_and_upsert(
        topic_id=tid,
        raw_entity="nvidia",
        raw_attribute="revenue",
        raw_value="2.697e10",
        raw_timeframe="2023",
        entry_type="web_evidence",
        source_ref="[W101]",
        domain_score=0.90,
        source_domain="sec.gov",
    )

    # Similar credibility + different value → CONFLICT
    lid, status = normalize_and_upsert(
        topic_id=tid,
        raw_entity="nvidia",
        raw_attribute="revenue",
        raw_value="2.700e10",
        raw_timeframe="2023",
        entry_type="web_evidence",
        source_ref="[W105]",
        domain_score=0.92,
        source_domain="ft.com",
    )
    assert status == "conflict"
    assert lid is not None

    # Verify conflict edge was created
    edges = db.get_ledger_edges(tid, edge_type="conflicts_with")
    assert len(edges) >= 1


# ---------------------------------------------------------------------------
# 38. credibility: duplicate same value same time
# ---------------------------------------------------------------------------


def test_credibility_duplicate():
    from orbit_or.ledger import add_attribute_with_aliases

    tid = _make_topic()
    add_entity_with_aliases(tid, "NVIDIA", "company", ["nvidia"], confirmed=True)
    add_attribute_with_aliases(tid, "Revenue", "numeric", ["revenue"], confirmed=True)

    lid1, _ = normalize_and_upsert(
        topic_id=tid,
        raw_entity="nvidia",
        raw_attribute="revenue",
        raw_value="2.697e10",
        raw_timeframe="2023",
        entry_type="web_evidence",
        source_ref="[W101]",
        domain_score=0.80,
        source_domain="wsj.com",
    )

    # Same value, same time → DUPLICATE
    lid2, status2 = normalize_and_upsert(
        topic_id=tid,
        raw_entity="nvidia",
        raw_attribute="revenue",
        raw_value="2.697e10",
        raw_timeframe="2023",
        entry_type="web_evidence",
        source_ref="[W102]",
        domain_score=0.80,
        source_domain="bloomberg.com",
    )
    assert status2 == "deduplicated"
    assert lid2 == lid1


# ---------------------------------------------------------------------------
# 39. credibility: provenance upgrade (same value, higher score)
# ---------------------------------------------------------------------------


def test_credibility_provenance_upgrade():
    from orbit_or.ledger import add_attribute_with_aliases

    tid = _make_topic()
    add_entity_with_aliases(tid, "Market", "concept", ["market"], confirmed=True)
    add_attribute_with_aliases(tid, "Revenue", "numeric", ["revenue"], confirmed=True)

    # arxiv preprint
    lid1, _ = normalize_and_upsert(
        topic_id=tid,
        raw_entity="market",
        raw_attribute="revenue",
        raw_value="9.0e10",
        raw_timeframe="2025",
        entry_type="web_evidence",
        source_ref="[W111]",
        domain_score=0.65,
        source_domain="arxiv.org",
    )

    # Peer-reviewed confirms same data → upgrades provenance
    lid2, status2 = normalize_and_upsert(
        topic_id=tid,
        raw_entity="market",
        raw_attribute="revenue",
        raw_value="9.0e10",
        raw_timeframe="2025",
        entry_type="web_evidence",
        source_ref="[W112]",
        domain_score=0.90,
        source_domain="ieee.org",
    )
    assert status2 == "replaced"
    assert lid2 == lid1
