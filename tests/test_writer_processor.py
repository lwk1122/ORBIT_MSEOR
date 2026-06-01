import pytest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from orbit_or.writer_processor import process_clerk_claim_output, process_writer_output


@contextmanager
def _mock_empty_db():
    """Mock db.get_db() returning a connection with no matching rows."""
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = []
    mock_conn.execute.return_value.fetchone.return_value = None
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_conn)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    with patch("orbit_or.writer_processor.db.get_db", return_value=mock_ctx):
        yield mock_conn


@pytest.mark.asyncio
async def test_process_writer_output_respects_explicit_empty_facts():
    with patch("orbit_or.writer_processor.api.fact_exists", return_value=False):
        with patch("orbit_or.writer_processor.api.fact_candidate_exists", return_value=False):
            with patch("orbit_or.writer_processor.api.create_fact_candidate_with_stage") as create_candidate:
                created = await process_writer_output(
                    1,
                    2,
                    3,
                    "Writer quoted earlier notes:\nFACT: old quoted line",
                    structured_facts=[],
                )

    assert created == []
    create_candidate.assert_not_called()


@pytest.mark.asyncio
async def test_process_writer_output_deduplicates_duplicate_structured_facts():
    with patch("orbit_or.writer_processor.api.fact_exists", return_value=False):
        with patch("orbit_or.writer_processor.api.fact_candidate_exists", return_value=False):
            with patch(
                "orbit_or.writer_processor.api.create_fact_candidate_with_stage",
                side_effect=[11],
            ) as create_candidate:
                created = await process_writer_output(
                    1,
                    2,
                    3,
                    "unused",
                    structured_facts=["A   verified fact", "A verified fact"],
                )

    assert created == [11]
    assert create_candidate.call_args.args == (1, 2, 3, "A verified fact")
    assert create_candidate.call_args.kwargs["fact_stage"] == "synthesized"
    assert create_candidate.call_args.kwargs["candidate_type"] == "sourced_claim"
    assert create_candidate.call_args.kwargs["evidence_note"] is None


@pytest.mark.asyncio
async def test_process_writer_output_skips_existing_final_fact():
    with patch("orbit_or.writer_processor.api.fact_exists", return_value=True):
        with patch("orbit_or.writer_processor.api.fact_candidate_exists", return_value=False):
            with patch("orbit_or.writer_processor.api.create_fact_candidate_with_stage") as create_candidate:
                created = await process_writer_output(
                    1,
                    2,
                    3,
                    "unused",
                    structured_facts=["Already stored fact"],
                )

    assert created == []
    create_candidate.assert_not_called()


@pytest.mark.asyncio
async def test_process_writer_output_ignores_non_string_fact_entries():
    with patch("orbit_or.writer_processor.api.fact_exists", return_value=False):
        with patch("orbit_or.writer_processor.api.fact_candidate_exists", return_value=False):
            with patch(
                "orbit_or.writer_processor.api.create_fact_candidate_with_stage",
                side_effect=[17],
            ) as create_candidate:
                created = await process_writer_output(
                    1,
                    2,
                    3,
                    "unused",
                    structured_facts=[{"bad": 1}, 99, "Valid fact"],
                )

    assert created == [17]
    assert create_candidate.call_args.args == (1, 2, 3, "Valid fact")
    assert create_candidate.call_args.kwargs["fact_stage"] == "synthesized"
    assert create_candidate.call_args.kwargs["candidate_type"] == "sourced_claim"
    assert create_candidate.call_args.kwargs["evidence_note"] is None


@pytest.mark.asyncio
async def test_process_writer_output_caps_candidates():
    with patch("orbit_or.writer_processor.api.fact_exists", return_value=False):
        with patch("orbit_or.writer_processor.api.fact_candidate_exists", return_value=False):
            with patch(
                "orbit_or.writer_processor.api.create_fact_candidate_with_stage",
                side_effect=[1, 2],
            ) as create_candidate:
                created = await process_writer_output(
                    1,
                    2,
                    3,
                    "unused",
                    structured_facts=["Fact one", "Fact two", "Fact three"],
                    max_candidates=2,
                )

    assert created == [1, 2]
    assert create_candidate.call_count == 2


@pytest.mark.asyncio
async def test_process_writer_output_caps_after_filtering_duplicates():
    with patch("orbit_or.writer_processor.api.fact_exists", return_value=False):
        with patch("orbit_or.writer_processor.api.fact_candidate_exists", return_value=False):
            with patch(
                "orbit_or.writer_processor.api.create_fact_candidate_with_stage",
                side_effect=[1, 2],
            ) as create_candidate:
                created = await process_writer_output(
                    1,
                    2,
                    None,
                    "unused",
                    structured_facts=["Fact one", "Fact one", "Fact two"],
                    max_candidates=2,
                )

    assert created == [1, 2]
    assert create_candidate.call_count == 2


@pytest.mark.asyncio
async def test_process_writer_output_caps_after_filtering_blank_entries():
    with patch("orbit_or.writer_processor.api.fact_exists", return_value=False):
        with patch("orbit_or.writer_processor.api.fact_candidate_exists", return_value=False):
            with patch(
                "orbit_or.writer_processor.api.create_fact_candidate_with_stage",
                side_effect=[5],
            ) as create_candidate:
                created = await process_writer_output(
                    1,
                    2,
                    None,
                    "unused",
                    structured_facts=["  ", "Fact two"],
                    max_candidates=2,
                )

    assert created == [5]
    assert create_candidate.call_args.args == (1, 2, None, "Fact two")
    assert create_candidate.call_args.kwargs["fact_stage"] == "synthesized"
    assert create_candidate.call_args.kwargs["candidate_type"] == "sourced_claim"
    assert create_candidate.call_args.kwargs["evidence_note"] is None


@pytest.mark.asyncio
async def test_process_writer_output_preserves_structured_fact_metadata():
    structured_fact = {
        "candidate_text": "The benchmark reported a 12% latency increase.",
        "candidate_type": "number",
        "source_refs": ["paper:smith2024"],
        "source_excerpt": "Table 2 reports a 12% latency increase.",
        "verification_status": "accepted",
    }

    with _mock_empty_db():
        with patch("orbit_or.writer_processor.api.fact_exists", return_value=False):
            with patch("orbit_or.writer_processor.api.fact_candidate_exists", return_value=False):
                with patch("orbit_or.writer_processor.api.create_fact_candidate_with_stage", side_effect=[9]) as create_candidate:
                    created = await process_writer_output(1, 2, None, "unused", structured_facts=[structured_fact], round_number=3)

    assert created == [9]
    assert create_candidate.call_args.args == (1, 2, None, "The benchmark reported a 12% latency increase.")
    assert create_candidate.call_args.kwargs["candidate_type"] == "number"
    assert create_candidate.call_args.kwargs["source_excerpt"] == "Table 2 reports a 12% latency increase."
    assert create_candidate.call_args.kwargs["verification_status"] == "accepted"
    assert create_candidate.call_args.kwargs["round_number"] == 3


@pytest.mark.asyncio
async def test_process_clerk_claim_output_requires_support_fact_ids():
    with patch("orbit_or.writer_processor.api.claim_candidate_exists", return_value=False):
        with patch("orbit_or.writer_processor.api.create_claim_candidate", side_effect=[31]) as create_candidate:
            created = await process_clerk_claim_output(
                1,
                2,
                None,
                [
                    {
                        "candidate_text": "Claim with support",
                        "support_fact_ids": [1, 2],
                        "rationale_short": "Both facts support the same direction.",
                    },
                    {
                        "candidate_text": "Claim without support",
                        "support_fact_ids": [],
                        "rationale_short": "Should be dropped.",
                    },
                ],
            )

    assert created == [31]
    create_candidate.assert_called_once()
