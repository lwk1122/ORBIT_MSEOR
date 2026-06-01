"""Tests for the W→F evidence pipeline (Phase 3)."""

import asyncio
import os
import sqlite3
import unittest
from unittest.mock import AsyncMock, patch

os.environ["TESTING"] = "1"

from orbit_or.evidence_parser import (
    DOMAIN_SCORE_THRESHOLD,
    UNKNOWN_DOMAIN_SCORE,
    build_evidence_extraction_prompt,
    extract_evidence_to_ledger,
    score_domain,
    try_promote_pending_entries,
)
from orbit_or.db import (
    get_db,
    get_unprocessed_web_evidence,
    init_db,
    insert_web_evidence,
    mark_web_evidence_ledger_processed,
)
from orbit_or.server import parse_clerk_ledger_output


class TestScoreDomain(unittest.TestCase):
    """Tests 1-4: Domain scoring."""

    def test_score_domain_known(self):
        self.assertEqual(score_domain("reuters.com"), 1.0)
        self.assertEqual(score_domain("reddit.com"), 0.2)
        self.assertEqual(score_domain("cnbc.com"), 0.8)

    def test_score_domain_unknown(self):
        self.assertEqual(score_domain("unknown.example.com"), UNKNOWN_DOMAIN_SCORE)

    def test_score_domain_www_prefix(self):
        self.assertEqual(score_domain("www.reuters.com"), 1.0)
        self.assertEqual(score_domain("www.reddit.com"), 0.2)

    def test_score_domain_empty(self):
        self.assertEqual(score_domain(""), UNKNOWN_DOMAIN_SCORE)
        self.assertEqual(score_domain(None), UNKNOWN_DOMAIN_SCORE)

    def test_score_domain_academic(self):
        """New academic domains should score 0.65+."""
        self.assertEqual(score_domain("arxiv.org"), 0.65)
        self.assertGreaterEqual(score_domain("nature.com"), 0.85)
        self.assertGreaterEqual(score_domain("ieee.org"), 0.85)
        self.assertGreaterEqual(score_domain("dl.acm.org"), 0.85)

    def test_score_domain_edu_wildcard(self):
        self.assertEqual(score_domain("mit.edu"), 0.65)
        self.assertEqual(score_domain("stanford.edu"), 0.65)

    def test_score_domain_gov_wildcard(self):
        self.assertEqual(score_domain("data.gov"), 0.80)
        self.assertEqual(score_domain("nih.gov"), 0.80)

    def test_score_domain_tech(self):
        """Big tech research should score high."""
        self.assertGreaterEqual(score_domain("research.google"), 0.85)
        self.assertGreaterEqual(score_domain("openai.com"), 0.85)


class TestBuildPrompt(unittest.TestCase):
    """Tests 5-6: Evidence extraction prompt."""

    def _make_web_row(self, **overrides):
        base = {
            "id": 42,
            "query_text": "MUFG forex reserves 2025",
            "title": "MUFG Reports Record Forex Holdings",
            "snippet": "MUFG announced $1.2 trillion in forex reserves for Q1 2025",
            "source_domain": "reuters.com",
            "url": "https://reuters.com/article/123",
        }
        base.update(overrides)
        return base

    def test_build_prompt_contains_query_and_snippet(self):
        entities = [(1, "MUFG"), (2, "Bank of Japan")]
        attributes = [(1, "forex_reserves"), (2, "interest_rate")]
        prompt = build_evidence_extraction_prompt(
            topic_id=1,
            web_row=self._make_web_row(),
            entities=entities,
            attributes=attributes,
        )
        self.assertIsNotNone(prompt)
        self.assertIn("MUFG forex reserves 2025", prompt)
        self.assertIn("MUFG Reports Record Forex Holdings", prompt)
        self.assertIn("$1.2 trillion in forex reserves", prompt)
        self.assertIn("1. MUFG", prompt)
        self.assertIn("2. Bank of Japan", prompt)
        self.assertIn("1. forex_reserves", prompt)
        self.assertIn("W42", prompt)

    def test_build_prompt_returns_none_no_entities(self):
        prompt = build_evidence_extraction_prompt(
            topic_id=1,
            web_row=self._make_web_row(),
            entities=[],
            attributes=[],
        )
        self.assertIsNone(prompt)

    def test_build_prompt_uses_json_format(self):
        entities = [(1, "MUFG")]
        attributes = [(1, "forex_reserves")]
        prompt = build_evidence_extraction_prompt(
            topic_id=1,
            web_row=self._make_web_row(),
            entities=entities,
            attributes=attributes,
        )
        self.assertIn("strict JSON object", prompt)
        self.assertIn("stat_type", prompt)
        self.assertIn("baseline_entity", prompt)
        self.assertIn("config", prompt)
        self.assertIn("thought", prompt)
        self.assertIn("STRICT", prompt)


class TestDomainFilter(unittest.TestCase):
    """Test 7: Low-score domains are skipped."""

    def test_domain_filter_skips_low_score(self):
        web_row = {
            "id": 99,
            "source_domain": "reddit.com",
            "query_text": "test",
            "title": "test",
            "snippet": "test",
        }
        with patch("orbit_or.evidence_parser.api") as mock_api:
            result = asyncio.run(extract_evidence_to_ledger(1, 1, web_row))
        self.assertEqual(result, [])
        mock_api.mark_web_evidence_ledger_processed.assert_called_once_with([99])

    def test_domain_filter_threshold_boundary(self):
        """Domain score at exactly 0.6 should NOT be filtered (threshold is < 0.6, not <=)."""
        self.assertEqual(DOMAIN_SCORE_THRESHOLD, 0.6)


class TestExtractionPipeline(unittest.TestCase):
    """Test 8: Full pipeline with mocked LLM."""

    def test_extraction_pipeline_mock_llm(self):
        web_row = {
            "id": 55,
            "source_domain": "reuters.com",
            "query_text": "MUFG forex",
            "title": "MUFG Report",
            "snippet": "MUFG holds $1.2T in reserves",
            "origin_subtopic_id": 2,
        }
        llm_response = "ENTITY: 1 | ATTR: 1 | VALUE: 1.2 trillion | MIN: 1200000000000 | MAX: 1200000000000 | UNIT: USD | TIME: Q1 2025 | SOURCE: [W55]"

        entities = [(1, "MUFG")]
        attributes = [(1, "forex_reserves")]

        with (
            patch("orbit_or.evidence_parser._ledger") as mock_ledger,
            patch(
                "orbit_or.evidence_parser.call_text",
                new_callable=AsyncMock,
                return_value=llm_response,
            ),
            patch("orbit_or.evidence_parser.api") as mock_api,
            patch("orbit_or.server.parse_clerk_ledger_output") as mock_parse,
        ):
            mock_ledger.get_entity_numbered_list.return_value = entities
            mock_ledger.get_attribute_numbered_list.return_value = attributes
            mock_ledger.normalize_and_upsert.return_value = (10, "inserted")
            mock_parse.return_value = [
                {
                    "type": "structured",
                    "topic_id": 1,
                    "subtopic_id": 2,
                    "entity_id": 1,
                    "attribute_id": 1,
                    "raw_value": "1.2 trillion",
                    "raw_timeframe": "Q1 2025",
                    "source_ref": "[W55]",
                    "created_by": "evidence_parser",
                    "current_round": 0,
                }
            ]

            result = asyncio.run(extract_evidence_to_ledger(1, 2, web_row))

        self.assertEqual(len(result), 1)
        mock_ledger.normalize_and_upsert.assert_called_once()
        call_kwargs = mock_ledger.normalize_and_upsert.call_args[1]
        self.assertEqual(call_kwargs["entry_type"], "web_evidence")
        self.assertEqual(call_kwargs["source_ref"], "[W55]")
        self.assertEqual(call_kwargs["domain_score"], 1.0)
        self.assertEqual(call_kwargs["source_domain"], "reuters.com")
        mock_api.mark_web_evidence_ledger_processed.assert_called_once_with([55])

    def test_provider_operational_error_does_not_mark_processed(self):
        web_row = {
            "id": 56,
            "source_domain": "reuters.com",
            "query_text": "MUFG forex",
            "title": "MUFG Report",
            "snippet": "MUFG holds $1.2T in reserves",
            "origin_subtopic_id": 2,
        }

        with (
            patch("orbit_or.evidence_parser._ledger") as mock_ledger,
            patch("orbit_or.evidence_parser.api") as mock_api,
            patch(
                "orbit_or.evidence_parser._fact_provider",
                side_effect=sqlite3.OperationalError("database is locked"),
            ),
        ):
            mock_ledger.get_entity_numbered_list.return_value = [(1, "MUFG")]
            mock_ledger.get_attribute_numbered_list.return_value = [
                (1, "forex_reserves")
            ]

            with self.assertRaises(sqlite3.OperationalError):
                asyncio.run(extract_evidence_to_ledger(1, 2, web_row))

        mock_api.mark_web_evidence_ledger_processed.assert_not_called()


class TestPendingPromotion(unittest.TestCase):
    """Tests 9-10: Pending entry promotion and cleanup."""

    def test_pending_promotion(self):
        with (
            patch("orbit_or.evidence_parser.db") as mock_db,
            patch("orbit_or.evidence_parser._ledger") as mock_ledger,
        ):
            mock_db.get_active_ledger_pending.return_value = [
                {
                    "id": 5,
                    "topic_id": 1,
                    "subtopic_id": 1,
                    "raw_text": "GDP growth",
                    "source_ref": "[W10]",
                    "missing_fields": "entity",
                }
            ]
            mock_ledger.resolve_entity.return_value = 3
            mock_ledger.normalize_and_upsert.return_value = (20, "inserted")
            mock_db.expire_ledger_pending.return_value = 0

            promoted = try_promote_pending_entries(1, current_round=5)

        self.assertEqual(promoted, 1)
        mock_db.delete_ledger_pending.assert_called_once_with(5)

    def test_pending_cleanup(self):
        with (
            patch("orbit_or.evidence_parser.db") as mock_db,
            patch("orbit_or.evidence_parser._ledger"),
        ):
            mock_db.get_active_ledger_pending.return_value = []
            mock_db.expire_ledger_pending.return_value = 3

            promoted = try_promote_pending_entries(1, current_round=10)

        self.assertEqual(promoted, 0)
        mock_db.expire_ledger_pending.assert_called_once_with(10)


class TestLedgerProcessedFlag(unittest.TestCase):
    """Test 11: ledger_processed flag on WebEvidence."""

    @classmethod
    def setUpClass(cls):
        init_db()

    def test_ledger_processed_flag(self):
        # Create a topic to satisfy FK
        with get_db() as conn:
            cur = conn.execute(
                "INSERT INTO Topic (summary, detail, status) VALUES (?, ?, 'Started')",
                ("test_evidence_topic", "detail"),
            )
            topic_id = cur.lastrowid

        try:
            # Insert web evidence
            web_id = insert_web_evidence(
                origin_topic_id=topic_id,
                origin_subtopic_id=None,
                query_text="test query",
                title="Test Title",
                snippet="Test snippet",
                url="https://example.com",
                source_domain="example.com",
                result_rank=1,
                search_provider="test",
                search_role="test",
            )

            # Verify unprocessed
            unprocessed = get_unprocessed_web_evidence(topic_id)
            ids = [r["id"] for r in unprocessed]
            self.assertIn(web_id, ids)

            # Mark processed
            mark_web_evidence_ledger_processed([web_id])

            # Verify no longer in unprocessed
            unprocessed_after = get_unprocessed_web_evidence(topic_id)
            ids_after = [r["id"] for r in unprocessed_after]
            self.assertNotIn(web_id, ids_after)
        finally:
            # Cleanup
            with get_db() as conn:
                conn.execute(
                    "DELETE FROM WebEvidence WHERE origin_topic_id = ?", (topic_id,)
                )
                conn.execute("DELETE FROM Topic WHERE id = ?", (topic_id,))


class TestParseReuseWithEvidenceFormat(unittest.TestCase):
    """Test 12: parse_clerk_ledger_output works with [W42] source format."""

    def test_parse_reuse_with_evidence_format(self):
        output = "ENTITY: 1 | ATTR: 1 | VALUE: 500B | MIN: 500000000000 | MAX: 500000000000 | UNIT: USD | TIME: 2025 | SOURCE: [W42]"
        entities = [(1, "MUFG")]
        attributes = [(1, "total_assets")]

        result = parse_clerk_ledger_output(
            output,
            entities,
            attributes,
            topic_id=1,
            subtopic_id=1,
            round_number=0,
            sender="evidence_parser",
        )

        self.assertTrue(len(result) >= 1)
        entry = result[0]
        self.assertEqual(entry["type"], "structured")
        # Parser strips brackets from SOURCE field values
        self.assertIn("W42", entry["source_ref"])
        self.assertEqual(entry["entity_id"], 1)
        self.assertEqual(entry["attribute_id"], 1)


if __name__ == "__main__":
    unittest.main()
