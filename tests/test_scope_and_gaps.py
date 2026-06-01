"""Tests for Phase 4 (Scope Lock) and Phase 5 (Gap Search) features."""

import os

os.environ["TESTING"] = "1"

import pytest
from unittest.mock import AsyncMock, patch

from orbit_or.server import (
    ANALYSIS_PHASE,
    _extract_evidence_gaps_section,
    _diff_evidence_gaps,
    build_actor_system_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_state(**overrides) -> dict:
    """Return the smallest ChatState-compatible dict for build_actor_system_prompt."""
    base = {
        "phase": ANALYSIS_PHASE,
        "round_number": 3,
        "subtopic_id": 0,
    }
    base.update(overrides)
    return base


# ===========================================================================
# 1-6: _extract_evidence_gaps_section (pure function)
# ===========================================================================


def test_extract_gaps_standard_format():
    text = (
        "TRAJECTORY:\ntext\n"
        "EVIDENCE GAPS:\n- Gap A\n- Gap B\n"
        "AGENT DELTAS:\ntext"
    )
    assert _extract_evidence_gaps_section(text) == "- Gap A\n- Gap B"


def test_extract_gaps_no_header():
    text = "TRAJECTORY:\ntext\nCONSENSUS:\ntext"
    assert _extract_evidence_gaps_section(text) == ""


def test_extract_gaps_at_end_of_text():
    text = "CONSENSUS:\ntext\nEVIDENCE GAPS:\n- Gap 1\n- Gap 2"
    assert _extract_evidence_gaps_section(text) == "- Gap 1\n- Gap 2"


def test_extract_gaps_markdown_header():
    text = "## Evidence Gaps\n- gap item\nBLOCKERS:\ntext"
    assert _extract_evidence_gaps_section(text) == "- gap item"


@pytest.mark.parametrize(
    "input_text,expected",
    [
        (None, ""),
        ("", ""),
    ],
)
def test_extract_gaps_empty_and_none(input_text, expected):
    assert _extract_evidence_gaps_section(input_text) == expected


def test_extract_gaps_empty_section():
    text = "EVIDENCE GAPS:\nAGENT DELTAS:\ntext"
    assert _extract_evidence_gaps_section(text) == ""


# ===========================================================================
# 7-8: _diff_evidence_gaps (async, needs mock for call_text)
# ===========================================================================


@pytest.mark.asyncio
async def test_diff_gaps_blank_input():
    prev_gaps = [{"id": "g1", "description": "x"}]
    with patch("orbit_or.server.call_text", new_callable=AsyncMock) as mock_call:
        result = await _diff_evidence_gaps("", prev_gaps)
    assert result == {"persistent": [], "new": [], "resolved": []}
    mock_call.assert_not_called()


@pytest.mark.asyncio
async def test_diff_gaps_llm_failure():
    with patch(
        "orbit_or.server.call_text",
        new_callable=AsyncMock,
        side_effect=RuntimeError("LLM down"),
    ):
        result = await _diff_evidence_gaps("- Some gap text", [])
    assert result == {"persistent": [], "new": [], "resolved": []}


# ===========================================================================
# 9-11: build_actor_system_prompt — locked scope injection
# ===========================================================================


def test_build_prompt_injects_locked_scope():
    scope_json = (
        '{"target_metric":"USD/CNY",'
        '"entity_boundaries":"MUFG, PBOC",'
        '"metric_definition":"spot rate"}'
    )
    state = _minimal_state()
    prompt = build_actor_system_prompt(
        state, "dreamer", "base",
        subtopic_data={"locked_scope": scope_json},
    )
    assert "LOCKED SCOPE (immutable" in prompt
    assert "Target metric: USD/CNY" in prompt
    assert "Entity boundaries: MUFG, PBOC" in prompt
    assert "Metric definition: spot rate" in prompt


def test_build_prompt_invalid_scope_json():
    state = _minimal_state()
    prompt = build_actor_system_prompt(
        state, "dreamer", "base",
        subtopic_data={"locked_scope": "not json{"},
    )
    assert "LOCKED SCOPE" not in prompt


def test_build_prompt_scope_null_optional_keys():
    scope_json = (
        '{"target_metric":null,'
        '"entity_boundaries":"China macro",'
        '"metric_definition":null}'
    )
    state = _minimal_state()
    prompt = build_actor_system_prompt(
        state, "dreamer", "base",
        subtopic_data={"locked_scope": scope_json},
    )
    assert "Entity boundaries: China macro" in prompt
    assert "Target metric:" not in prompt


# ===========================================================================
# 12-14: build_actor_system_prompt — gap directive injection
# ===========================================================================


def test_build_prompt_injects_gap_directive_for_target():
    state = _minimal_state(
        gap_search_directive={"id": "g1", "description": "Missing GDP data"},
        spectator_web_boost_target="analyst",
    )
    prompt = build_actor_system_prompt(
        state, "analyst", "base",
        subtopic_data={"id": 0},
    )
    assert "PRIORITY DIRECTIVE" in prompt
    assert "Missing GDP data" in prompt


def test_build_prompt_no_gap_directive_for_non_target():
    state = _minimal_state(
        gap_search_directive={"id": "g1", "description": "Missing GDP data"},
        spectator_web_boost_target="analyst",
    )
    prompt = build_actor_system_prompt(
        state, "scientist", "base",
        subtopic_data={"id": 0},
    )
    assert "PRIORITY DIRECTIVE" not in prompt


def test_build_prompt_gap_directive_string_fallback():
    state = _minimal_state(
        gap_search_directive="Missing data",
        spectator_web_boost_target="analyst",
    )
    prompt = build_actor_system_prompt(
        state, "analyst", "base",
        subtopic_data={"id": 0},
    )
    assert "PRIORITY DIRECTIVE" in prompt
    assert "Missing data" in prompt
