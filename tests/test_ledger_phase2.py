"""Tests for Ledger RAG Redesign Phase 2: agent integration."""

import os

os.environ["TESTING"] = "1"

import pytest

from orbit_or import db
from orbit_or.rag import (
    _render_contested_block,
    _render_knowledge_guide,
    _render_ledger_section,
)
from orbit_or.server import (
    CITATION_ID_PATTERN,
    _has_uncited_financial_numbers,
    _sanitize_citations_to_allowed_ids,
    _validate_ledger_citation_ids,
    build_clerk_ledger_extraction_prompt,
    parse_clerk_ledger_output,
)
from orbit_or.fact_daemon import _message_has_numerical_content
from orbit_or import ledger as _ledger


@pytest.fixture(autouse=True)
def _setup_db():
    """Ensure a fresh test database with schema."""
    db.init_db()
    yield


@pytest.fixture
def topic_id():
    with db.get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO Topic (summary, detail) VALUES (?, ?)",
            ("Test Topic", "Detail"),
        )
        return cursor.lastrowid


@pytest.fixture
def entity_and_attr(topic_id):
    eid = db.create_entity_with_aliases_batch(
        topic_id, "MUFG", "bank", ["mufg"], confirmed=True
    )
    aid = db.create_attribute_with_aliases_batch(
        topic_id, "Forecast", "numeric", ["forecast"], confirmed=True
    )
    return eid, aid


# --- 1. Ledger rendering ---


def test_render_ledger_section_groups_by_entity():
    entries = [
        {
            "id": 1,
            "entity_name": "MUFG",
            "attribute_name": "Forecast",
            "value": "6.8000",
            "unit": "USD/CNY",
            "normalized_timeframe": "2026-Q4",
            "source_ref": "[W12]",
            "source_domain": "reuters.com",
            "entry_type": "evidence",
            "status": "accepted",
        },
        {
            "id": 2,
            "entity_name": "MUFG",
            "attribute_name": "Rate",
            "value": "7.0",
            "unit": "%",
            "normalized_timeframe": "2026-Q1",
            "source_ref": "[W13]",
            "source_domain": None,
            "entry_type": "agent_claim",
            "status": "accepted",
        },
        {
            "id": 3,
            "entity_name": "PBOC",
            "attribute_name": "Rate",
            "value": "3.5",
            "unit": "%",
            "normalized_timeframe": "2026-Q2",
            "source_ref": "[F5]",
            "source_domain": None,
            "entry_type": "evidence",
            "status": "accepted",
        },
    ]
    result = _render_ledger_section(entries)
    assert "### MUFG" in result
    assert "### PBOC" in result
    assert "L1" in result
    assert "L2" in result
    assert "L3" in result
    assert "6.8000 USD/CNY" in result


def test_render_ledger_section_empty():
    assert _render_ledger_section([]) == ""


def test_render_contested_block():
    contested = [
        {
            "entity_name": "MUFG",
            "attribute_name": "Forecast",
            "timeframe": "2026-Q4",
            "entries": [
                {"id": 1, "value": "6.8000", "source_ref": "[W12]"},
                {"id": 2, "value": "7.1000", "source_ref": "[W15]"},
            ],
        }
    ]
    result = _render_contested_block(contested)
    assert "CONTESTED DATA" in result
    assert "L1 vs L2" in result
    assert "6.8000" in result
    assert "7.1000" in result


def test_knowledge_guide_includes_ledger():
    guide = _render_knowledge_guide(include_web=False, include_ledger=True)
    assert "[L...]" in guide
    assert "numerical" in guide.lower()


def test_knowledge_guide_excludes_ledger():
    guide = _render_knowledge_guide(include_web=False, include_ledger=False)
    assert "[L...]" not in guide


# --- 5. Citation pattern ---


def test_citation_pattern_matches_L():
    match = CITATION_ID_PATTERN.search("[L42]")
    assert match is not None
    assert match.group(1) == "L"
    assert match.group(2) == "42"


def test_citation_pattern_matches_all_types():
    text = "[F1] [C2] [W3] [L4]"
    matches = CITATION_ID_PATTERN.findall(text)
    assert len(matches) == 4
    assert ("L", "4") in matches


# --- 6-7. Citation sanitization ---


def test_sanitize_citations_removes_invalid_L():
    content = "Rate is 6.8000 [L999] according to sources."
    knowledge = "[F1] fact [W2] web"
    cleaned, removed = _sanitize_citations_to_allowed_ids(
        content, knowledge_blocks=[knowledge]
    )
    assert "[L999]" not in cleaned
    assert 999 in removed["L"]


def test_sanitize_citations_keeps_valid_L():
    content = "Rate is 6.8000 [L42] according to sources."
    knowledge = "[L42] MUFG Forecast"
    cleaned, removed = _sanitize_citations_to_allowed_ids(
        content, knowledge_blocks=[knowledge]
    )
    assert "[L42]" in cleaned
    assert removed["L"] == ()


# --- 8-10. Uncited financial numbers ---


def test_has_uncited_financial_numbers_detects():
    result = _has_uncited_financial_numbers("The rate is 6.8000 per the latest data.")
    assert "6.8000" in result


def test_has_uncited_financial_numbers_cited():
    result = _has_uncited_financial_numbers(
        "The rate is 6.8000 [L1] per the latest data."
    )
    assert result == []


def test_has_uncited_financial_numbers_no_match():
    result = _has_uncited_financial_numbers("In the year 2026, we expect changes.")
    assert result == []


# --- 11-15. Clerk prompt parser ---


def test_parse_clerk_output_basic(topic_id, entity_and_attr):
    eid, aid = entity_and_attr
    entities = [(eid, "MUFG")]
    attrs = [(aid, "Forecast")]
    output = f"ENTITY: [{eid}] | ATTR: [{aid}] | VALUE: [6.8000] | MIN: [6.8] | MAX: [6.8] | UNIT: [USD/CNY] | TIME: [2026-Q4] | SOURCE: [[W12]]"
    results = parse_clerk_ledger_output(
        output, entities, attrs, topic_id, None, 3, "scientist"
    )
    assert len(results) == 1
    assert results[0]["type"] == "structured"
    assert results[0]["entity_id"] == eid
    assert results[0]["attribute_id"] == aid


def test_parse_clerk_output_new_entity(topic_id, entity_and_attr):
    eid, aid = entity_and_attr
    entities = [(eid, "MUFG")]
    attrs = [(aid, "Forecast")]
    output = f"ENTITY: NEW (Bank of Japan) | ATTR: [{aid}] | VALUE: [1.2] | UNIT: [%] | TIME: [2026] | SOURCE: [web]"
    results = parse_clerk_ledger_output(
        output, entities, attrs, topic_id, None, 2, "analyst"
    )
    assert len(results) == 1
    assert results[0]["type"] == "structured"
    # New entity should have been created
    assert results[0]["entity_id"] is not None


def test_parse_clerk_output_unspecified(topic_id, entity_and_attr):
    eid, aid = entity_and_attr
    entities = [(eid, "MUFG")]
    attrs = [(aid, "Forecast")]
    output = (
        f"ENTITY: [0] | ATTR: [{aid}] | VALUE: [5.0] | TIME: [2026] | SOURCE: [src]"
    )
    results = parse_clerk_ledger_output(
        output, entities, attrs, topic_id, None, 3, "dreamer"
    )
    assert len(results) == 0  # Entity 0 = UNSPECIFIED → skipped


def test_parse_clerk_output_none():
    results = parse_clerk_ledger_output("NONE", [], [], 1, None, 1, "test")
    assert results == []


def test_parse_clerk_output_fact_line():
    output = "FACT: PBOC is likely to maintain current policy stance through 2026."
    results = parse_clerk_ledger_output(output, [], [], 1, None, 1, "test")
    assert len(results) == 1
    assert results[0]["type"] == "qualitative"
    assert "PBOC" in results[0]["text"]


# --- 16. Entity list freeze ---


def test_entity_list_r5_freeze(topic_id, entity_and_attr):
    prompt_r3 = build_clerk_ledger_extraction_prompt(topic_id, "test 6.8", "sci", 3)
    assert prompt_r3 is not None
    assert "NEW" in prompt_r3

    prompt_r6 = build_clerk_ledger_extraction_prompt(topic_id, "test 6.8", "sci", 6)
    assert prompt_r6 is not None
    # After R5, entity NEW option should not appear but attribute NEW still does
    # Split on "Attributes" to only check entity section
    entity_section = prompt_r6.split("Attributes")[0]
    assert "NEW:" not in entity_section and "NEW. Use" not in entity_section


# --- 17. Numerical content detection ---


def test_message_has_numerical_content():
    assert _message_has_numerical_content("The rate is 6.8 percent.") is True
    assert _message_has_numerical_content("We should analysis this topic.") is False
    assert _message_has_numerical_content("In Q1 2026, rate was 3.5.") is True


# --- 18. Clerk prompt includes entities ---


def test_build_clerk_prompt_includes_entities(topic_id, entity_and_attr):
    prompt = build_clerk_ledger_extraction_prompt(
        topic_id, "The rate is 6.8", "scientist", 3
    )
    assert prompt is not None
    assert "MUFG" in prompt
    assert "Forecast" in prompt
    assert '"entity"' in prompt  # JSON format


def test_build_clerk_prompt_returns_none_empty(topic_id):
    """No entities or attributes → returns None."""
    prompt = build_clerk_ledger_extraction_prompt(topic_id, "test", "sci", 1)
    assert prompt is None


# --- 19. Validate ledger citation IDs ---


def test_validate_ledger_citation_ids(topic_id, entity_and_attr):
    """Fake [L999] detected as invalid."""
    invalid = _validate_ledger_citation_ids("Rate is 6.8 [L999] and 7.0 [L998]")
    assert "[L999]" in invalid
    assert "[L998]" in invalid


# --- 20. Ledger entry exists ---


def test_ledger_entry_exists(topic_id, entity_and_attr):
    eid, aid = entity_and_attr
    lid, _ = db.upsert_ledger_entry(
        topic_id=topic_id,
        subtopic_id=None,
        entity_id=eid,
        attribute_id=aid,
        value="6.8",
        value_numeric_min=6.8,
        value_numeric_max=6.8,
        unit="USD/CNY",
        normalized_timeframe="2026-Q4",
        entry_type="evidence",
        source_ref="[W12]",
    )
    assert db.ledger_entry_exists(lid) is True
    assert db.ledger_entry_exists(99999) is False


# --- DB query functions ---


def test_get_ledger_entries_with_names(topic_id, entity_and_attr):
    eid, aid = entity_and_attr
    db.upsert_ledger_entry(
        topic_id=topic_id,
        subtopic_id=None,
        entity_id=eid,
        attribute_id=aid,
        value="6.8",
        value_numeric_min=6.8,
        value_numeric_max=6.8,
        unit="USD/CNY",
        normalized_timeframe="2026-Q4",
        entry_type="evidence",
        source_ref="[W12]",
    )
    entries = db.get_ledger_entries_with_names(topic_id)
    assert len(entries) >= 1
    assert entries[0]["entity_name"] == "MUFG"
    assert entries[0]["attribute_name"] == "Forecast"


def test_get_contested_ledger_pairs(topic_id, entity_and_attr):
    eid, aid = entity_and_attr
    db.upsert_ledger_entry(
        topic_id=topic_id,
        subtopic_id=None,
        entity_id=eid,
        attribute_id=aid,
        value="6.8",
        value_numeric_min=6.8,
        value_numeric_max=6.8,
        unit="USD/CNY",
        normalized_timeframe="2026-Q4",
        entry_type="evidence",
        source_ref="[W12]",
    )
    db.upsert_ledger_entry(
        topic_id=topic_id,
        subtopic_id=None,
        entity_id=eid,
        attribute_id=aid,
        value="7.1",
        value_numeric_min=7.1,
        value_numeric_max=7.1,
        unit="USD/CNY",
        normalized_timeframe="2026-Q4",
        entry_type="evidence",
        source_ref="[W15]",
    )
    contested = db.get_contested_ledger_pairs(topic_id)
    assert len(contested) == 1
    assert contested[0]["entity_name"] == "MUFG"
    assert len(contested[0]["entries"]) == 2


def test_entity_numbered_list(topic_id, entity_and_attr):
    entities = _ledger.get_entity_numbered_list(topic_id, 1)
    assert len(entities) >= 1
    assert any(name == "MUFG" for _, name in entities)


def test_attribute_numbered_list(topic_id, entity_and_attr):
    attrs = _ledger.get_attribute_numbered_list(topic_id)
    assert len(attrs) >= 1
    assert any(name == "Forecast" for _, name in attrs)
