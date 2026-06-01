import os

os.environ.setdefault("TESTING", "1")

import pytest
from unittest.mock import patch, AsyncMock

from orbit_or.broker import BrokerResponse, SearchEvidenceItem
from orbit_or.prompts import PROMPTS
from orbit_or.rag import (
    _collect_local_rag_records,
    _compress_rag_context,
    _normalize_query_planner_contract,
    _render_retrieval_notices,
    _strip_agent_format_instructions,
    assemble_rag_context,
    build_query_rag_context,
)

LEDGER_MOCKS = {
    "orbit_or.rag.api.get_ledger_entries_with_names": [],
    "orbit_or.rag.api.get_contested_ledger_pairs": [],
}


def _patch_ledger():
    """Context manager stack that patches both ledger API calls to return []."""
    import contextlib

    return contextlib.ExitStack()


def _apply_ledger_patches(stack):
    for target, rv in LEDGER_MOCKS.items():
        stack.enter_context(patch(target, return_value=rv))


@pytest.mark.asyncio
async def test_assemble_rag_context_empty():
    res, degraded = await assemble_rag_context(1, 1, [], "dreamer")
    assert res == ""
    assert degraded is False


def test_retrieval_notices_surface_stale_versions_and_conflicts():
    notices = _render_retrieval_notices(
        corpus_chunks=[
            {
                "id": 10,
                "document_id": 1,
                "document_title": "Capacity memo",
                "freshness_timestamp": "2020-01-01T00:00:00+00:00",
            },
            {
                "id": 11,
                "document_id": 2,
                "document_title": "Capacity memo",
                "freshness_timestamp": "2026-01-01T00:00:00+00:00",
            },
        ],
        facts=[{"id": 3, "content": "Fact A"}],
        claims=[{"id": 5, "content": "Claim A"}],
        fact_conflicts=[{"source_id": 3, "target_id": 4}],
        claim_conflicts=[{"source_id": 5, "target_id": 6}],
    )

    assert "Retrieval Notices" in notices
    assert "[D10] may be stale" in notices
    assert "Potential corpus version conflict" in notices
    assert "[F3] conflicts with [F4]" in notices
    assert "[C5] conflicts with [C6]" in notices


def test_compress_rag_context_adds_budget_notice():
    text = "=== RAG KNOWLEDGE INJECTION ===\n" + "\n".join(
        f"- [F{i}] fact text {i}" for i in range(50)
    )

    compressed = _compress_rag_context(text, max_chars=260)

    assert len(compressed) < len(text)
    assert "Context Compression Notice" in compressed
    assert "Omitted" in compressed


@pytest.mark.asyncio
async def test_collect_local_rag_records_confidence_gate_marks_degraded():
    with patch("orbit_or.rag.aget_embedding", new=AsyncMock(return_value=[0.1] * 3)):
        with patch(
            "orbit_or.rag.api.search_facts_hybrid",
            return_value=[{"id": 1, "content": "Weak fact"}],
        ):
            with patch("orbit_or.rag.api.search_claims_hybrid", return_value=[]):
                with patch("orbit_or.rag.api.search_corpus_chunks_hybrid", return_value=[]):
                    with patch("orbit_or.rag.api.search_messages_hybrid", return_value=[]):
                        with patch(
                            "orbit_or.rag.arerank",
                            new=AsyncMock(return_value=[(0, 0.1)]),
                        ):
                            records, degraded = await _collect_local_rag_records(
                                1, "weak query"
                            )

    assert degraded is True
    assert records["facts"] == ()


@pytest.mark.asyncio
async def test_assemble_rag_context():
    recent_messages = [
        {"id": 9, "content": "This is a test message", "sender": "critic"}
    ]

    mock_llm_call = AsyncMock(
        return_value=BrokerResponse(
            text='{"query":"test query"}', provider_used="minimax"
        )
    )
    mock_embedding = AsyncMock(return_value=[0.1] * 768)

    mock_facts = [
        {"id": 1, "content": "Fact 1", "source": "Writer"},
        {"id": 2, "content": "Fact 2", "source": "Writer"},
    ]
    mock_claims = [
        {"id": 7, "content": "Claim 1", "source": "Librarian"},
        {"id": 8, "content": "Claim 2", "source": "Librarian"},
    ]
    mock_summaries = [
        {"id": 3, "content": "Summary 1", "source": "Skynet"},
        {"id": 4, "content": "Summary 2", "source": "Skynet"},
    ]
    mock_messages = [
        {"id": 5, "content": "Historical message 1", "source": "engineer"},
        {"id": 6, "content": "Historical message 2", "source": "critic"},
    ]

    # Reranker returns indices and scores
    mock_rerank = AsyncMock(
        side_effect=[
            [(0, 0.9), (1, 0.1)],
            [(0, 0.8), (1, 0.1)],
            [(0, 0.8), (1, 0.1)],
            [(0, 0.7), (1, 0.1)],
        ]
    )

    with patch("orbit_or.rag.llm_call", new=mock_llm_call):
        with patch("orbit_or.rag.aget_embedding", new=mock_embedding):
            with patch(
                "orbit_or.rag.api.search_facts_hybrid", return_value=mock_facts
            ):
                with patch(
                    "orbit_or.rag.api.search_claims_hybrid", return_value=mock_claims
                ):
                    with patch(
                        "orbit_or.rag.api.search_messages_hybrid",
                        side_effect=[mock_summaries, mock_messages],
                    ):
                        with patch("orbit_or.rag.arerank", new=mock_rerank):
                            with patch(
                                "orbit_or.rag.api.get_ledger_entries_with_names",
                                return_value=[],
                            ):
                                with patch(
                                    "orbit_or.rag.api.get_contested_ledger_pairs",
                                    return_value=[],
                                ):
                                    with patch(
                                        "orbit_or.rag.api.get_code_evidence_for_topic",
                                        return_value=[],
                                    ):
                                        with patch(
                                            "orbit_or.rag.api.get_api_evidence_for_topic",
                                            return_value=[],
                                        ):
                                            res, degraded = await assemble_rag_context(
                                                1, 1, recent_messages, "dreamer"
                                            )

                                    assert "RAG KNOWLEDGE INJECTION" in res
                                    assert degraded is False
                                    assert "[Related Claims]" in res
                                    assert "[F1]" in res
                                    assert "[C7]" in res
                                    assert "Fact 1" in res
                                    assert "[M3]" in res
                                    assert "Summary 1" in res
                                    assert "[M5]" in res
                                    assert "Historical message 1" in res
                                    # Fact 2 is dropped because score 0.1 < 0.3
                                    assert "[F2]" not in res


@pytest.mark.asyncio
async def test_assemble_rag_context_falls_back_to_last_message_when_query_distillation_raises():
    recent_messages = [{"id": 9, "content": "Fallback me", "sender": "critic"}]

    with patch(
        "orbit_or.rag.retry_structured_output", new=AsyncMock(return_value=None)
    ):
        with patch(
            "orbit_or.rag._collect_local_rag_records",
            new=AsyncMock(
                return_value=(
                    {"facts": (), "claims": (), "summaries": (), "messages": ()},
                    False,
                )
            ),
        ) as collect_rag:
            with patch(
                "orbit_or.rag.api.get_ledger_entries_with_names", return_value=[]
            ):
                with patch(
                    "orbit_or.rag.api.get_contested_ledger_pairs", return_value=[]
                ):
                    with patch(
                        "orbit_or.rag.api.get_code_evidence_for_topic",
                        return_value=[],
                    ):
                        with patch(
                            "orbit_or.rag.api.get_api_evidence_for_topic",
                            return_value=[],
                        ):
                            res, degraded = await assemble_rag_context(
                                1, 1, recent_messages, "dreamer"
                            )

    assert res == ""
    assert degraded is True
    assert collect_rag.await_args.args[1] == "Fallback me"


@pytest.mark.asyncio
async def test_assemble_rag_context_uses_web_backup_when_local_memory_is_empty():
    recent_messages = [{"id": 9, "content": "Fallback me", "sender": "critic"}]

    with patch(
        "orbit_or.rag.retry_structured_output",
        new=AsyncMock(return_value=BrokerResponse(text='{"query":"backup query"}')),
    ):
        with patch(
            "orbit_or.rag._collect_local_rag_records",
            new=AsyncMock(
                return_value=(
                    {"facts": (), "claims": (), "summaries": (), "messages": ()},
                    False,
                )
            ),
        ):
            with patch(
                "orbit_or.rag.get_or_collect_search_evidence_item",
                new=AsyncMock(
                    return_value=SearchEvidenceItem(
                        query="backup query",
                        rendered_results="=== WEB SEARCH RESULTS ===\n[W11] Title: T\nSource: example.com\nSnippet: S\n\n",
                        had_error=False,
                        web_ids=(11,),
                    )
                ),
            ):
                with patch(
                    "orbit_or.rag.api.get_ledger_entries_with_names", return_value=[]
                ):
                    with patch(
                        "orbit_or.rag.api.get_contested_ledger_pairs", return_value=[]
                    ):
                        with patch(
                            "orbit_or.rag.api.get_code_evidence_for_topic",
                            return_value=[],
                        ):
                            with patch(
                                "orbit_or.rag.api.get_api_evidence_for_topic",
                                return_value=[],
                            ):
                                res, degraded = await assemble_rag_context(
                                    1,
                                    1,
                                    recent_messages,
                                    "dreamer",
                                    allow_web_backup=True,
                                )

    assert degraded is False
    assert "[Related Web Evidence]" in res
    assert "[W11]" in res


@pytest.mark.asyncio
async def test_assemble_rag_context_includes_api_evidence():
    recent_messages = [{"id": 9, "content": "LLM capability question", "sender": "critic"}]

    with patch(
        "orbit_or.rag.retry_structured_output",
        new=AsyncMock(return_value=BrokerResponse(text='{"query":"LLM API"}')),
    ):
        with patch(
            "orbit_or.rag._collect_local_rag_records",
            new=AsyncMock(
                return_value=(
                    {"facts": (), "claims": (), "summaries": (), "messages": ()},
                    False,
                )
            ),
        ):
            with patch(
                "orbit_or.rag.api.get_ledger_entries_with_names", return_value=[]
            ):
                with patch(
                    "orbit_or.rag.api.get_contested_ledger_pairs", return_value=[]
                ):
                    with patch(
                        "orbit_or.rag.api.get_code_evidence_for_topic",
                        return_value=[],
                    ):
                        with patch(
                            "orbit_or.rag.api.get_api_evidence_for_topic",
                            return_value=[
                                {
                                    "id": 9,
                                    "question": "Can a clean API call help?",
                                    "answer": "Yes, as model perspective only.",
                                    "provider": "minimax",
                                    "requesting_role": "scientist",
                                }
                            ],
                        ):
                            res, degraded = await assemble_rag_context(
                                1, 1, recent_messages, "scientist"
                            )

    assert degraded is False
    assert "[A9]" in res
    assert "unverified model/API consultation" in res


@pytest.mark.asyncio
async def test_api_only_rag_does_not_suppress_web_backup():
    recent_messages = [{"id": 9, "content": "Need factual evidence", "sender": "critic"}]

    with patch(
        "orbit_or.rag.retry_structured_output",
        new=AsyncMock(return_value=BrokerResponse(text='{"query":"factual evidence"}')),
    ):
        with patch(
            "orbit_or.rag._collect_local_rag_records",
            new=AsyncMock(
                return_value=(
                    {"facts": (), "claims": (), "summaries": (), "messages": ()},
                    False,
                )
            ),
        ):
            with patch(
                "orbit_or.rag.api.get_ledger_entries_with_names", return_value=[]
            ):
                with patch(
                    "orbit_or.rag.api.get_contested_ledger_pairs", return_value=[]
                ):
                    with patch(
                        "orbit_or.rag.api.get_code_evidence_for_topic",
                        return_value=[],
                    ):
                        with patch(
                            "orbit_or.rag.api.get_api_evidence_for_topic",
                            return_value=[
                                {
                                    "id": 5,
                                    "question": "Model-only question",
                                    "answer": "Model-only answer",
                                    "provider": "minimax",
                                    "requesting_role": "scientist",
                                }
                            ],
                        ):
                            with patch(
                                "orbit_or.rag._check_rag_sufficiency",
                                new=AsyncMock(return_value=True),
                            ) as sufficiency:
                                with patch(
                                    "orbit_or.rag.get_or_collect_search_evidence_item",
                                    new=AsyncMock(
                                        return_value=SearchEvidenceItem(
                                            query="factual evidence",
                                            rendered_results="=== WEB SEARCH RESULTS ===\n[W11] Title: T\nSnippet: S\n",
                                            had_error=False,
                                            web_ids=(11,),
                                        )
                                    ),
                                ) as collect_web:
                                    res, degraded = await assemble_rag_context(
                                        1,
                                        1,
                                        recent_messages,
                                        "scientist",
                                        allow_web_backup=True,
                                    )

    assert degraded is False
    assert "[A5]" in res
    assert "[W11]" in res
    sufficiency.assert_not_awaited()
    collect_web.assert_awaited_once()


@pytest.mark.asyncio
async def test_build_query_rag_context_excludes_api_evidence_by_default():
    with patch(
        "orbit_or.rag._collect_local_rag_records",
        new=AsyncMock(
            return_value=(
                {"facts": (), "claims": (), "summaries": (), "messages": ()},
                False,
            )
        ),
    ):
        with patch("orbit_or.rag.api.get_ledger_entries_with_names", return_value=[]):
            with patch("orbit_or.rag.api.get_contested_ledger_pairs", return_value=[]):
                with patch(
                    "orbit_or.rag.api.get_code_evidence_for_topic", return_value=[]
                ):
                    with patch(
                        "orbit_or.rag.api.get_api_evidence_for_topic",
                        return_value=[
                            {
                                "id": 8,
                                "question": "Unverified model question",
                                "answer": "Unverified model answer",
                            }
                        ],
                    ) as get_api_evidence:
                        res, degraded = await build_query_rag_context(
                            1, "verification task"
                        )

    assert res == ""
    assert degraded is False
    get_api_evidence.assert_not_called()


# --- Tests for _normalize_query_planner_contract fallbacks ---


class TestNormalizeQueryPlannerContract:
    def test_primary_query_key(self):
        result = _normalize_query_planner_contract('{"query": "test query"}')
        assert result == {"parsed_ok": True, "query": "test query"}

    def test_empty_input(self):
        result = _normalize_query_planner_contract("")
        assert result == {"parsed_ok": False, "query": ""}

    def test_non_json(self):
        result = _normalize_query_planner_contract("just some text")
        assert result == {"parsed_ok": False, "query": ""}

    def test_fallback_content_key(self):
        """Agent format with 'content' key should be salvaged."""
        result = _normalize_query_planner_contract(
            '{"action": "post_message", "content": "economic impact of AI"}'
        )
        assert result["parsed_ok"] is True
        assert result["query"] == "economic impact of AI"

    def test_fallback_reason_key(self):
        """Spectator focus format with 'reason' key should be salvaged."""
        result = _normalize_query_planner_contract(
            '{"action": "focus", "reason": "scientist has the strongest data", "target": "scientist"}'
        )
        assert result["parsed_ok"] is True
        # 'reason' is checked before 'target'
        assert result["query"] == "scientist has the strongest data"

    def test_fallback_target_key(self):
        """If only 'target' key is usable, salvage it."""
        result = _normalize_query_planner_contract(
            '{"action": "focus", "target": "scientist"}'
        )
        assert result["parsed_ok"] is True
        assert result["query"] == "scientist"

    def test_empty_query_with_content_fallback(self):
        """Empty 'query' should trigger fallback to 'content'."""
        result = _normalize_query_planner_contract(
            '{"query": "", "content": "actual useful text"}'
        )
        assert result["parsed_ok"] is True
        assert result["query"] == "actual useful text"

    def test_all_keys_empty(self):
        result = _normalize_query_planner_contract(
            '{"action": "post_message", "content": "", "reason": ""}'
        )
        assert result == {"parsed_ok": False, "query": ""}


# --- Tests for _strip_agent_format_instructions ---


class TestStripAgentFormatInstructions:
    def test_strips_mandatory_reasoning_drafting(self):
        """Deliberator prompts: everything from 【MANDATORY REASONING DRAFTING】 onward is removed."""
        result = _strip_agent_format_instructions(PROMPTS["dreamer"])
        assert "【MANDATORY REASONING DRAFTING】" not in result
        assert '"action": "post_message"' not in result
        assert "internal_citation_mapping" not in result
        # Identity preserved
        assert "You are the Dreamer" in result

    def test_strips_skynet_format_blocks(self):
        result = _strip_agent_format_instructions(PROMPTS["skynet"])
        assert "Depending on the TASK" not in result
        assert '"action": "create_plan"' not in result
        assert '"action": "post_summary"' not in result
        # Identity preserved
        assert "You are Skynet" in result

    def test_strips_writer_format(self):
        result = _strip_agent_format_instructions(PROMPTS["writer"])
        assert 'Format: {"action"' not in result
        assert "You are the Writer" in result

    def test_strips_cat_format(self):
        result = _strip_agent_format_instructions(PROMPTS["cat"])
        assert 'Format: {"action"' not in result
        assert "You are the Mascot" in result

    def test_strips_dog_format(self):
        result = _strip_agent_format_instructions(PROMPTS["dog"])
        assert 'Format: {"action"' not in result
        assert "You are the Guard Dog" in result

    def test_strips_tron_format(self):
        result = _strip_agent_format_instructions(PROMPTS["tron"])
        assert "Format if violation" not in result
        assert "Format if safe" not in result
        assert "You are Tron" in result

    def test_strips_spectator_format(self):
        result = _strip_agent_format_instructions(PROMPTS["spectator"])
        assert "use this Format" not in result
        assert "If you are voting in a governance round" not in result
        assert "You are Spectator" in result

    def test_preserves_identity_for_all_prompts(self):
        """Every prompt should retain its core identity line after stripping."""
        identity_markers = {
            "skynet": "You are Skynet",
            "writer": "You are the Writer",
            "fact_proposer": "You are the hidden Clerk",
            "librarian": "You are the Librarian",
            "dreamer": "You are the Dreamer",
            "scientist": "You are the Scientist",
            "engineer": "You are the Engineer",
            "analyst": "You are the Data Analyst",
            "critic": "You are the Critic",
            "cat": "You are the Mascot",
            "dog": "You are the Guard Dog",
            "contrarian": "You are the Contrarian",
            "tron": "You are Tron",
            "spectator": "You are Spectator",
        }
        for agent_name, marker in identity_markers.items():
            result = _strip_agent_format_instructions(PROMPTS[agent_name])
            assert marker in result, f"{agent_name} lost identity marker: {marker}"

    def test_all_deliberator_prompts_stripped(self):
        """All deliberators should have their format blocks fully removed."""
        deliberators = [
            "dreamer",
            "scientist",
            "engineer",
            "analyst",
            "critic",
            "contrarian",
        ]
        for name in deliberators:
            result = _strip_agent_format_instructions(PROMPTS[name])
            assert (
                "【MANDATORY REASONING DRAFTING】" not in result
            ), f"{name} still has MANDATORY block"
            assert (
                "internal_citation_mapping" not in result
            ), f"{name} still has citation mapping"
            assert (
                "confidence_score" not in result
            ), f"{name} still has confidence_score"
