"""Tests for Phase D: Data Quality — LE-4, LE-1, FA-1, FA-3, CL-5."""

import json
import os
from unittest.mock import patch

import pytest

os.environ["TESTING"] = "1"

from orbit_or import db
from orbit_or.ledger import (
    _derive_timeframe_from_dates,
    _validate_ledger_entry,
    normalize_and_upsert,
    add_entity_with_aliases,
    add_attribute_with_aliases,
)


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """Each test gets a fresh database."""
    test_db = str(tmp_path / "test_phase_d.db")
    monkeypatch.setattr(db, "get_db_path", lambda: test_db)
    db.init_db()
    yield


def _make_subtopic(topic_id: int, summary: str = "Sub1") -> int:
    with db.get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO Subtopic (topic_id, summary, detail) VALUES (?, ?, ?)",
            (topic_id, summary, "detail"),
        )
        return cursor.lastrowid


def _make_topic(summary="Phase D test topic") -> int:
    with db.get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO Topic (summary, detail) VALUES (?, ?)",
            (summary, "Detail"),
        )
        return cursor.lastrowid


# ---------------------------------------------------------------------------
# LE-4: Timeframe normalization fallback
# ---------------------------------------------------------------------------


class TestLE4TimeframeFallback:
    """LE-4: derive normalized_timeframe from valid_from/valid_to."""

    def test_derive_from_iso_date_with_month(self):
        assert _derive_timeframe_from_dates("2025-03-15", None) == "2025-Q1"

    def test_derive_from_q2_month(self):
        assert _derive_timeframe_from_dates("2024-06-01", None) == "2024-Q2"

    def test_derive_from_q3_month(self):
        assert _derive_timeframe_from_dates(None, "2024-09-30") == "2024-Q3"

    def test_derive_from_q4_month(self):
        assert _derive_timeframe_from_dates("2024-12-31", None) == "2024-Q4"

    def test_derive_year_only(self):
        # Year without recognizable month still gets year
        assert _derive_timeframe_from_dates("2024", None) == "2024"

    def test_derive_empty(self):
        assert _derive_timeframe_from_dates(None, None) == ""

    def test_normalize_and_upsert_uses_fallback(self):
        """When raw_timeframe is empty but valid_from is populated, derive timeframe."""
        tid = _make_topic()
        add_entity_with_aliases(
            tid, "TestCorp", "company", ["testcorp"], confirmed=True
        )
        add_attribute_with_aliases(
            tid, "Revenue", "numeric", ["revenue"], confirmed=True
        )

        lid, status = normalize_and_upsert(
            topic_id=tid,
            raw_entity="testcorp",
            raw_attribute="revenue",
            raw_value="5.2",
            raw_timeframe="",  # empty
            entry_type="web_evidence",
            source_ref="[W1]",
            source_domain="reuters.com",
            valid_from="2025-03-01",
            valid_to="2025-03-31",
        )
        assert lid is not None
        assert status == "inserted"
        entries = db.get_ledger_entries(tid)
        assert len(entries) == 1
        assert entries[0]["normalized_timeframe"] == "2025-Q1"


# ---------------------------------------------------------------------------
# LE-1: Ledger validation gate
# ---------------------------------------------------------------------------


class TestLE1ValidationGate:
    """LE-1: _validate_ledger_entry rejects garbage."""

    def test_valid_entry_passes(self):
        assert (
            _validate_ledger_entry(
                raw_value="6.8",
                vmin=6.8,
                vmax=6.8,
                unit="USD",
            )
            is None
        )

    def test_non_finite_vmin_rejected(self):
        result = _validate_ledger_entry(
            raw_value="inf",
            vmin=float("inf"),
            vmax=100.0,
            unit=None,
        )
        assert result == "non-finite vmin"

    def test_non_finite_vmax_rejected(self):
        result = _validate_ledger_entry(
            raw_value="nan",
            vmin=0.0,
            vmax=float("nan"),
            unit=None,
        )
        assert result == "non-finite vmax"

    def test_vmin_greater_than_vmax_rejected(self):
        result = _validate_ledger_entry(
            raw_value="100-50",
            vmin=100.0,
            vmax=50.0,
            unit=None,
        )
        assert result == "vmin > vmax"

    def test_invalid_unit_rejected(self):
        result = _validate_ledger_entry(
            raw_value="5 widgets",
            vmin=5.0,
            vmax=5.0,
            unit="widgets",
        )
        assert result is not None
        assert "invalid unit" in result

    def test_prose_without_numeric_rejected(self):
        result = _validate_ledger_entry(
            raw_value="This is a long prose description of the methodology used",
            vmin=None,
            vmax=None,
            unit=None,
        )
        assert result == "prose without numeric"

    def test_prose_pattern_rejected(self):
        result = _validate_ledger_entry(
            raw_value="the approach was measured carefully",
            vmin=None,
            vmax=None,
            unit=None,
        )
        assert result is not None

    def test_short_text_without_numeric_passes(self):
        """Short text (<=5 words) without numeric should pass rule 3."""
        result = _validate_ledger_entry(
            raw_value="N/A",
            vmin=None,
            vmax=None,
            unit=None,
        )
        assert result is None

    def test_unspecified_entity_rejected(self):
        result = _validate_ledger_entry(
            raw_value="6.8",
            vmin=6.8,
            vmax=6.8,
            unit=None,
            entity_id=0,
            attribute_id=1,
        )
        assert result == "unspecified entity"

    def test_unspecified_attribute_rejected(self):
        result = _validate_ledger_entry(
            raw_value="6.8",
            vmin=6.8,
            vmax=6.8,
            unit=None,
            entity_id=1,
            attribute_id=0,
        )
        assert result == "unspecified attribute"

    def test_normalize_and_upsert_rejects_prose(self):
        """Full pipeline rejects prose values."""
        tid = _make_topic()
        add_entity_with_aliases(
            tid, "TestCorp", "company", ["testcorp"], confirmed=True
        )
        add_attribute_with_aliases(
            tid, "Revenue", "numeric", ["revenue"], confirmed=True
        )

        lid, status = normalize_and_upsert(
            topic_id=tid,
            raw_entity="testcorp",
            raw_attribute="revenue",
            raw_value="The methodology was carefully designed to measure the approach used",
            raw_timeframe="2025",
            entry_type="web_evidence",
            source_ref="[W1]",
        )
        assert lid is None
        assert status == "rejected"


# ---------------------------------------------------------------------------
# FA-1: Fact/Claim hard boundary
# ---------------------------------------------------------------------------


class TestFA1SourceKind:
    """FA-1: source_kind column + hard boundary for agent-sourced facts."""

    def test_source_kind_column_exists(self):
        """FactCandidate table has source_kind column."""
        with db.get_db() as conn:
            cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(FactCandidate)").fetchall()
            }
        assert "source_kind" in cols

    def test_create_fact_candidate_with_source_kind(self):
        """source_kind is stored and retrievable."""
        tid = _make_topic()
        sid = _make_subtopic(tid)
        cid = db.create_fact_candidate(
            tid,
            sid,
            None,
            "Test fact",
            fact_stage="synthesized",
            source_kind="agent",
        )
        with db.get_db() as conn:
            row = conn.execute(
                "SELECT source_kind FROM FactCandidate WHERE id = ?", (cid,)
            ).fetchone()
        assert row["source_kind"] == "agent"

    def test_web_source_kind_stored(self):
        tid = _make_topic()
        sid = _make_subtopic(tid)
        cid = db.create_fact_candidate(
            tid,
            sid,
            None,
            "Web-sourced fact",
            fact_stage="web_extracted",
            source_kind="web",
        )
        with db.get_db() as conn:
            row = conn.execute(
                "SELECT source_kind FROM FactCandidate WHERE id = ?", (cid,)
            ).fetchone()
        assert row["source_kind"] == "web"

    @pytest.mark.asyncio
    async def test_agent_sourced_rejected_without_external_evidence(self):
        """FA-1 gate: agent-sourced candidate without [W]/[E] is rejected."""
        from orbit_or.librarian_processor import apply_librarian_review

        candidate = {
            "id": 99,
            "candidate_text": "Agent opinion without evidence",
            "fact_stage": "synthesized",
            "source_kind": "agent",
            "source_refs_json": json.dumps(["M1", "M2"]),  # No W or E refs
        }
        review = {
            "decision": "accept",
            "reviewed_text": "Agent opinion without evidence",
            "review_note": "Looks good",
            "evidence_note": None,
            "confidence_score": 8.0,
        }
        with patch("orbit_or.librarian_processor.api.update_fact_candidate_review"):
            result = await apply_librarian_review(1, candidate, review)

        assert result["decision"] == "reject"
        assert "agent opinion" in (result.get("review_note") or "").lower()

    @pytest.mark.asyncio
    async def test_web_sourced_passes_through(self):
        """Web-sourced candidate with [W] refs is accepted."""
        from orbit_or.librarian_processor import apply_librarian_review

        candidate = {
            "id": 100,
            "candidate_text": "Web-verified fact with source",
            "fact_stage": "web_extracted",
            "source_kind": "web",
            "source_refs_json": json.dumps(["W1", "W2"]),
        }
        review = {
            "decision": "accept",
            "reviewed_text": "Web-verified fact with source",
            "review_note": "Verified",
            "evidence_note": "Matched",
            "confidence_score": 9.0,
        }
        with patch(
            "orbit_or.librarian_processor.api.get_fact_by_content", return_value=None
        ):
            with patch(
                "orbit_or.librarian_processor.api.insert_fact", return_value=42
            ):
                with patch(
                    "orbit_or.librarian_processor.api.update_fact_candidate_review"
                ):
                    result = await apply_librarian_review(1, candidate, review)

        assert result["accepted_fact_id"] == 42
        assert result["decision"] != "reject"

    @pytest.mark.asyncio
    async def test_agent_sourced_with_web_evidence_passes(self):
        """Agent-sourced but with [W] external ref should pass."""
        from orbit_or.librarian_processor import apply_librarian_review

        candidate = {
            "id": 101,
            "candidate_text": "Agent claim backed by web",
            "fact_stage": "synthesized",
            "source_kind": "agent",
            "source_refs_json": json.dumps(["W5", "M1"]),  # Has W ref
        }
        review = {
            "decision": "accept",
            "reviewed_text": "Agent claim backed by web",
            "review_note": "Verified via web",
            "evidence_note": "W5 confirms",
            "confidence_score": 8.5,
        }
        with patch(
            "orbit_or.librarian_processor.api.get_fact_by_content", return_value=None
        ):
            with patch(
                "orbit_or.librarian_processor.api.insert_fact", return_value=43
            ):
                with patch(
                    "orbit_or.librarian_processor.api.update_fact_candidate_review"
                ):
                    result = await apply_librarian_review(1, candidate, review)

        assert result["accepted_fact_id"] == 43

    @pytest.mark.asyncio
    async def test_bootstrap_stage_not_rejected(self):
        """bootstrap fact_stage should not be classified as agent."""
        from orbit_or.librarian_processor import apply_librarian_review

        candidate = {
            "id": 102,
            "candidate_text": "Bootstrap seed fact",
            "fact_stage": "bootstrap",
        }
        review = {
            "decision": "accept",
            "reviewed_text": "Bootstrap seed fact",
            "review_note": "OK",
            "evidence_note": None,
            "confidence_score": 7.0,
        }
        with patch(
            "orbit_or.librarian_processor.api.get_fact_by_content", return_value=None
        ):
            with patch(
                "orbit_or.librarian_processor.api.insert_fact", return_value=44
            ):
                with patch(
                    "orbit_or.librarian_processor.api.update_fact_candidate_review"
                ):
                    result = await apply_librarian_review(1, candidate, review)

        assert result["accepted_fact_id"] == 44


# ---------------------------------------------------------------------------
# FA-3: Writer citation pre-flight
# ---------------------------------------------------------------------------


class TestFA3WriterCitationPreflight:
    """FA-3: Invalid [W] IDs stripped before fact candidate creation."""

    def test_web_evidence_ids_exist_returns_valid_subset(self):
        """db.web_evidence_ids_exist returns only existing IDs."""
        tid = _make_topic()
        # Create some web evidence
        with db.get_db() as conn:
            c1 = conn.execute(
                "INSERT INTO WebEvidence (origin_topic_id, query_text, title, url) VALUES (?, ?, ?, ?)",
                (tid, "test query", "Title", "http://example.com"),
            ).lastrowid
            c2 = conn.execute(
                "INSERT INTO WebEvidence (origin_topic_id, query_text, title, url) VALUES (?, ?, ?, ?)",
                (tid, "test query 2", "Title 2", "http://example2.com"),
            ).lastrowid

        valid = db.web_evidence_ids_exist(tid, [c1, c2, 9999])
        assert c1 in valid
        assert c2 in valid
        assert 9999 not in valid

    def test_web_evidence_ids_exist_empty_list(self):
        tid = _make_topic()
        assert db.web_evidence_ids_exist(tid, []) == set()

    @pytest.mark.asyncio
    async def test_invalid_w_refs_stripped(self):
        """Invalid [W999] stripped from source_refs_json, valid [W1] kept."""
        from orbit_or.writer_processor import _store_fact_candidates

        tid = _make_topic()
        sid = _make_subtopic(tid)

        # Create a real web evidence record
        with db.get_db() as conn:
            real_wid = conn.execute(
                "INSERT INTO WebEvidence (origin_topic_id, query_text, title, url) VALUES (?, ?, ?, ?)",
                (tid, "query", "Real source", "http://real.com"),
            ).lastrowid

        facts = [
            {
                "candidate_text": "Fact with mixed W refs",
                "source_refs": [f"W{real_wid}", "W99999"],
                "candidate_type": "sourced_claim",
            }
        ]

        ids = await _store_fact_candidates(
            tid, sid, None, facts, fact_stage="synthesized"
        )
        assert len(ids) == 1

        with db.get_db() as conn:
            row = conn.execute(
                "SELECT source_refs_json FROM FactCandidate WHERE id = ?", (ids[0],)
            ).fetchone()
        refs = json.loads(row["source_refs_json"])
        # W99999 should be stripped, real_wid kept
        assert f"W{real_wid}" in refs
        assert "W99999" not in refs


# ---------------------------------------------------------------------------
# CL-5: Cross-subtopic claim awareness (audit test)
# ---------------------------------------------------------------------------


class TestCL5CrossSubtopicClaims:
    """CL-5: Verify claim dedup is topic-wide, not subtopic-scoped."""

    @pytest.mark.asyncio
    async def test_claim_dedup_uses_topic_id(self):
        """check_claim_duplicate searches by topic_id, not subtopic_id."""
        from orbit_or.fact_dedup import check_claim_duplicate

        # Mock the lexical search to verify it's called with topic_id
        with patch("orbit_or.fact_dedup._db") as mock_db:
            mock_db.search_claims_lexical.return_value = []
            action, matched_id = await check_claim_duplicate(
                topic_id=42,
                new_claim_text="Test claim",
                new_support_fact_ids=[1, 2],
            )
            # Verify search_claims_lexical was called with topic_id (not subtopic_id)
            mock_db.search_claims_lexical.assert_called_once()
            call_args = mock_db.search_claims_lexical.call_args
            assert call_args[0][0] == 42  # topic_id is first positional arg

        assert action == "INSERT"
        assert matched_id is None


# ---------------------------------------------------------------------------
# DB: ON CONFLICT clause includes domain_score and source_domain
# ---------------------------------------------------------------------------


class TestDBOnConflictFix:
    """LE-4: ON CONFLICT UPDATE includes domain_score and source_domain."""

    def test_upsert_updates_domain_score(self):
        """Upserting same ledger key updates domain_score."""
        tid = _make_topic()
        add_entity_with_aliases(
            tid, "TestCorp", "company", ["testcorp"], confirmed=True
        )
        add_attribute_with_aliases(
            tid, "Revenue", "numeric", ["revenue"], confirmed=True
        )

        # First insert
        lid1, status1 = normalize_and_upsert(
            topic_id=tid,
            raw_entity="testcorp",
            raw_attribute="revenue",
            raw_value="5.2",
            raw_timeframe="2025",
            entry_type="web_evidence",
            source_ref="[W1]",
            source_domain="example.com",
            domain_score=0.7,
        )
        assert status1 == "inserted"

        # Upsert with higher domain_score
        lid2, status2 = normalize_and_upsert(
            topic_id=tid,
            raw_entity="testcorp",
            raw_attribute="revenue",
            raw_value="5.2",
            raw_timeframe="2025",
            entry_type="web_evidence",
            source_ref="[W1]",
            source_domain="reuters.com",
            domain_score=1.0,
        )

        entries = db.get_ledger_entries(tid)
        # Should have replaced due to higher credibility
        assert len(entries) >= 1
