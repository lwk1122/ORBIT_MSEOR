"""Tests for ledger dedup, source_ref normalization, and empty-value rejection."""

import os

os.environ.setdefault("TESTING", "1")

from orbit_or.ledger import normalize_source_ref, normalize_and_upsert
from orbit_or.rag import _format_source_for_display
from orbit_or.server import _unwrap_brackets


# ---------------------------------------------------------------------------
# normalize_source_ref
# ---------------------------------------------------------------------------


class TestNormalizeSourceRef:
    def test_single_bare_marker(self):
        assert normalize_source_ref("M415") == "M415"

    def test_single_bracketed_marker(self):
        assert normalize_source_ref("[L9]") == "L9"

    def test_multiple_bare_markers(self):
        assert normalize_source_ref("M418 M422 M423") == "M418 M422 M423"

    def test_concatenated_bracketed_markers(self):
        assert normalize_source_ref("[M426][M427][M429]") == "M426 M427 M429"

    def test_marker_with_surrounding_text(self):
        assert normalize_source_ref("dreamer M415 projection") == "M415"

    def test_marker_with_quote(self):
        result = normalize_source_ref(
            '[L9] "the mathematical floor for NVIDIA is 47-65%"'
        )
        assert result == "L9"

    def test_web_evidence_marker(self):
        assert normalize_source_ref("[W11]") == "W11"

    def test_dedup_repeated_markers(self):
        assert normalize_source_ref("[M1][M1][M2]") == "M1 M2"

    def test_empty_string(self):
        assert normalize_source_ref("") == ""

    def test_no_markers_fallback(self):
        """When no citation markers found, return truncated raw text."""
        result = normalize_source_ref("some random source text")
        assert result == "some random source text"

    def test_no_markers_long_fallback_truncated(self):
        long_text = "x" * 200
        result = normalize_source_ref(long_text)
        assert len(result) <= 100

    def test_fact_marker(self):
        assert normalize_source_ref("[F42]") == "F42"

    def test_claim_marker(self):
        assert normalize_source_ref("[C5]") == "C5"

    def test_mixed_marker_types(self):
        assert normalize_source_ref("[W10] confirms [F3]") == "W10 F3"


# ---------------------------------------------------------------------------
# _unwrap_brackets fix
# ---------------------------------------------------------------------------


class TestUnwrapBrackets:
    def test_single_bracket_pair(self):
        assert _unwrap_brackets("[hello]") == "hello"

    def test_no_brackets(self):
        assert _unwrap_brackets("hello") == "hello"

    def test_concatenated_citations_preserved(self):
        """Concatenated citations should NOT be unwrapped."""
        assert _unwrap_brackets("[M426][M427][M429]") == "[M426][M427][M429]"

    def test_two_concatenated_citations(self):
        assert _unwrap_brackets("[M429][M427]") == "[M429][M427]"

    def test_empty(self):
        assert _unwrap_brackets("") == ""

    def test_single_value(self):
        assert _unwrap_brackets("[47-65%]") == "47-65%"


# ---------------------------------------------------------------------------
# Empty value rejection in normalize_and_upsert
# ---------------------------------------------------------------------------


class TestEmptyValueRejection:
    def test_empty_string_rejected(self):
        lid, status = normalize_and_upsert(
            topic_id=1,
            raw_entity="Test",
            raw_attribute="Attr",
            raw_value="",
            entry_type="test",
            source_ref="M1",
        )
        assert lid is None
        assert status == "skipped"

    def test_whitespace_only_rejected(self):
        lid, status = normalize_and_upsert(
            topic_id=1,
            raw_entity="Test",
            raw_attribute="Attr",
            raw_value="   ",
            entry_type="test",
            source_ref="M1",
        )
        assert lid is None
        assert status == "skipped"


# ---------------------------------------------------------------------------
# Value-based dedup (integration test with real DB)
# ---------------------------------------------------------------------------


class TestValueBasedDedup:
    """These tests use the TESTING=1 test database."""

    @classmethod
    def setup_class(cls):
        from orbit_or.db import init_db

        init_db()

    def _seed_entity_and_attribute(self, topic_id: int):
        """Create a test topic, entity, and attribute. Return (entity_id, attribute_id)."""
        from orbit_or import db, ledger as _ledger

        # Ensure a Topic row exists for FK
        with db.get_db() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO Topic (id, summary, detail, status) VALUES (?, ?, ?, ?)",
                (topic_id, "test topic", "test detail", "Started"),
            )
        eid = _ledger.add_entity_with_aliases(
            topic_id, "TestCorp", "company", ["testcorp"], confirmed=True
        )
        aid = _ledger.add_attribute_with_aliases(
            topic_id, "Revenue", "numeric", ["revenue"], confirmed=True
        )
        return eid, aid

    def test_duplicate_value_is_deduplicated(self):
        """Inserting same numeric value twice should merge, not create duplicate."""
        from orbit_or import db

        # Use a unique topic_id to isolate test
        topic_id = 99990
        # Pre-cleanup in case of leftover data from a prior failed run
        with db.get_db() as conn:
            conn.execute("DELETE FROM LedgerEdge WHERE topic_id = ?", (topic_id,))
            conn.execute("DELETE FROM Ledger WHERE topic_id = ?", (topic_id,))
        eid, aid = self._seed_entity_and_attribute(topic_id)

        # First insert
        lid1, status1 = normalize_and_upsert(
            topic_id=topic_id,
            entity_id=eid,
            attribute_id=aid,
            raw_value="47-65%",
            raw_timeframe="2027",
            entry_type="web_evidence",
            source_ref="[M415]",
            domain_score=0.8,
        )
        assert status1 == "inserted"
        assert lid1 is not None

        # Second insert with same value, different source
        lid2, status2 = normalize_and_upsert(
            topic_id=topic_id,
            entity_id=eid,
            attribute_id=aid,
            raw_value="47-65%",
            raw_timeframe="2027",
            entry_type="web_evidence",
            source_ref="[L9]",
            domain_score=0.8,
        )
        assert status2 == "deduplicated"
        assert lid2 == lid1

        # Verify source_ref was merged
        with db.get_db() as conn:
            row = conn.execute(
                "SELECT source_ref FROM Ledger WHERE id = ?", (lid1,)
            ).fetchone()
        assert "M415" in row["source_ref"]
        assert "L9" in row["source_ref"]

        # Cleanup
        with db.get_db() as conn:
            conn.execute("DELETE FROM Ledger WHERE topic_id = ?", (topic_id,))
            conn.execute(
                "DELETE FROM LedgerEntityAlias WHERE entity_id IN (SELECT id FROM LedgerEntity WHERE topic_id = ?)",
                (topic_id,),
            )
            conn.execute(
                "DELETE FROM LedgerAttributeAlias WHERE attribute_id IN (SELECT id FROM LedgerAttribute WHERE topic_id = ?)",
                (topic_id,),
            )
            conn.execute("DELETE FROM LedgerEntity WHERE topic_id = ?", (topic_id,))
            conn.execute("DELETE FROM LedgerAttribute WHERE topic_id = ?", (topic_id,))
            conn.execute("DELETE FROM Topic WHERE id = ?", (topic_id,))

    def test_different_values_not_deduplicated(self):
        """Different numeric values should create separate entries (as conflict)."""
        from orbit_or import db

        topic_id = 99991
        # Pre-cleanup in case of leftover data from a prior failed run
        with db.get_db() as conn:
            conn.execute("DELETE FROM LedgerEdge WHERE topic_id = ?", (topic_id,))
            conn.execute("DELETE FROM Ledger WHERE topic_id = ?", (topic_id,))
        eid, aid = self._seed_entity_and_attribute(topic_id)

        lid1, status1 = normalize_and_upsert(
            topic_id=topic_id,
            entity_id=eid,
            attribute_id=aid,
            raw_value="47-65%",
            raw_timeframe="2027",
            entry_type="web_evidence",
            source_ref="[M415]",
            domain_score=0.8,
        )
        assert status1 == "inserted"

        lid2, status2 = normalize_and_upsert(
            topic_id=topic_id,
            entity_id=eid,
            attribute_id=aid,
            raw_value="70%",
            raw_timeframe="2027",
            entry_type="web_evidence",
            source_ref="[M418]",
            domain_score=0.8,
        )
        # Same entity/attr/time, same domain_score, different values → conflict
        assert status2 == "conflict"
        assert lid2 != lid1

        # Cleanup
        with db.get_db() as conn:
            conn.execute("DELETE FROM LedgerEdge WHERE topic_id = ?", (topic_id,))
            conn.execute("DELETE FROM Ledger WHERE topic_id = ?", (topic_id,))
            conn.execute(
                "DELETE FROM LedgerEntityAlias WHERE entity_id IN (SELECT id FROM LedgerEntity WHERE topic_id = ?)",
                (topic_id,),
            )
            conn.execute(
                "DELETE FROM LedgerAttributeAlias WHERE attribute_id IN (SELECT id FROM LedgerAttribute WHERE topic_id = ?)",
                (topic_id,),
            )
            conn.execute("DELETE FROM LedgerEntity WHERE topic_id = ?", (topic_id,))
            conn.execute("DELETE FROM LedgerAttribute WHERE topic_id = ?", (topic_id,))
            conn.execute("DELETE FROM Topic WHERE id = ?", (topic_id,))


# ---------------------------------------------------------------------------
# Double-unit rendering fix
# ---------------------------------------------------------------------------


class TestDoubleUnitRendering:
    def test_no_double_percent(self):
        from orbit_or.rag import _render_ledger_section

        entries = [
            {
                "id": 9,
                "entity_name": "NVIDIA",
                "attribute_name": "Market Share",
                "value": "47-65%",
                "unit": "%",
                "normalized_timeframe": "2027",
                "source_ref": "M415",
                "source_domain": "",
                "entry_type": "agent_claim",
                "status": "accepted",
            }
        ]
        result = _render_ledger_section(entries)
        assert "47-65% %" not in result
        assert "47-65%" in result

    def test_unit_appended_when_not_present(self):
        from orbit_or.rag import _render_ledger_section

        entries = [
            {
                "id": 10,
                "entity_name": "NVIDIA",
                "attribute_name": "Revenue",
                "value": "1.5 trillion",
                "unit": "USD",
                "normalized_timeframe": "2027",
                "source_ref": "W10",
                "source_domain": "",
                "entry_type": "web_evidence",
                "status": "accepted",
            }
        ]
        result = _render_ledger_section(entries)
        assert "1.5 trillion USD" in result


# ---------------------------------------------------------------------------
# _unwrap_brackets improved (nested bracket edge case)
# ---------------------------------------------------------------------------


class TestUnwrapBracketsImproved:
    def test_nested_brackets_preserved(self):
        """Nested brackets like [text [nested] stuff] should NOT be unwrapped."""
        assert _unwrap_brackets("[text [nested] stuff]") == "[text [nested] stuff]"

    def test_simple_still_unwraps(self):
        assert _unwrap_brackets("[hello]") == "hello"

    def test_concatenated_still_preserved(self):
        assert _unwrap_brackets("[M1][M2]") == "[M1][M2]"

    def test_single_value_with_special_chars(self):
        assert _unwrap_brackets("[47-65%]") == "47-65%"

    def test_empty_brackets(self):
        assert _unwrap_brackets("[]") == ""


# ---------------------------------------------------------------------------
# _format_source_for_display
# ---------------------------------------------------------------------------


class TestFormatSourceForDisplay:
    def test_bare_markers(self):
        assert _format_source_for_display("M418 M422 M423") == "[M418] [M422] [M423]"

    def test_bracketed_markers(self):
        assert (
            _format_source_for_display("[M426][M427][M429]") == "[M426] [M427] [M429]"
        )

    def test_malformed_concat(self):
        """M426][M427 style input gets properly bracketed."""
        assert _format_source_for_display("M426][M427][M429") == "[M426] [M427] [M429]"

    def test_empty_string(self):
        assert _format_source_for_display("") == ""

    def test_none_input(self):
        assert _format_source_for_display(None) == ""

    def test_whitespace_only(self):
        assert _format_source_for_display("   ") == ""

    def test_no_markers_fallback(self):
        result = _format_source_for_display("some random text")
        assert result == "some random text"

    def test_long_no_marker_truncated(self):
        result = _format_source_for_display("x" * 200)
        assert len(result) <= 60

    def test_dedup_repeated_markers(self):
        assert _format_source_for_display("[M1][M1][M2]") == "[M1] [M2]"

    def test_mixed_types(self):
        assert _format_source_for_display("[W10] confirms [F3]") == "[W10] [F3]"

    def test_single_marker(self):
        assert _format_source_for_display("L9") == "[L9]"

    def test_web_and_ledger(self):
        assert _format_source_for_display("W11 L5") == "[W11] [L5]"


# ---------------------------------------------------------------------------
# Contested detection (hybrid fingerprint)
# ---------------------------------------------------------------------------


class TestContestedDetection:
    """Integration tests for contested ledger detection with hybrid fingerprint."""

    @classmethod
    def setup_class(cls):
        from orbit_or.db import init_db

        init_db()

    def _seed(self, topic_id: int):
        from orbit_or import db, ledger as _ledger

        with db.get_db() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO Topic (id, summary, detail, status) VALUES (?, ?, ?, ?)",
                (topic_id, "contested test", "contested test detail", "Started"),
            )
        eid = _ledger.add_entity_with_aliases(
            topic_id, "TestEntity", "company", ["testentity"], confirmed=True
        )
        aid = _ledger.add_attribute_with_aliases(
            topic_id, "Metric", "numeric", ["metric"], confirmed=True
        )
        return eid, aid

    def _cleanup(self, topic_id: int):
        from orbit_or import db

        with db.get_db() as conn:
            conn.execute("DELETE FROM Ledger WHERE topic_id = ?", (topic_id,))
            conn.execute(
                "DELETE FROM LedgerEntityAlias WHERE entity_id IN (SELECT id FROM LedgerEntity WHERE topic_id = ?)",
                (topic_id,),
            )
            conn.execute(
                "DELETE FROM LedgerAttributeAlias WHERE attribute_id IN (SELECT id FROM LedgerAttribute WHERE topic_id = ?)",
                (topic_id,),
            )
            conn.execute("DELETE FROM LedgerEntity WHERE topic_id = ?", (topic_id,))
            conn.execute("DELETE FROM LedgerAttribute WHERE topic_id = ?", (topic_id,))
            conn.execute("DELETE FROM Topic WHERE id = ?", (topic_id,))

    def _direct_insert(
        self,
        conn,
        topic_id,
        eid,
        aid,
        value,
        timeframe,
        source_ref,
        value_numeric_min=None,
        value_numeric_max=None,
        unit=None,
    ):
        """Insert directly into Ledger to control numeric fields."""
        conn.execute(
            """INSERT INTO Ledger (topic_id, entity_id, attribute_id, value,
               normalized_timeframe, entry_type, source_ref, status,
               value_numeric_min, value_numeric_max, unit)
               VALUES (?, ?, ?, ?, ?, 'agent_claim', ?, 'accepted', ?, ?, ?)""",
            (
                topic_id,
                eid,
                aid,
                value,
                timeframe,
                source_ref,
                value_numeric_min,
                value_numeric_max,
                unit,
            ),
        )

    def test_same_number_different_text_not_contested(self):
        """2.6 POPs vs '2.6 POPs peak (5.22 with sparsity)' should NOT be contested
        when they share the same numeric min/max."""
        from orbit_or import db

        topic_id = 99980
        eid, aid = self._seed(topic_id)
        try:
            with db.get_db() as conn:
                self._direct_insert(
                    conn,
                    topic_id,
                    eid,
                    aid,
                    "2.6 POPs",
                    "2027",
                    "M1",
                    value_numeric_min=2.6,
                    value_numeric_max=2.6,
                )
                self._direct_insert(
                    conn,
                    topic_id,
                    eid,
                    aid,
                    "2.6 POPs peak (5.22 with sparsity)",
                    "2027",
                    "M2",
                    value_numeric_min=2.6,
                    value_numeric_max=2.6,
                )
            contested = db.get_contested_ledger_pairs(topic_id)
            assert (
                len(contested) == 0
            ), f"Expected no contested pairs, got {len(contested)}"
        finally:
            self._cleanup(topic_id)

    def test_different_numbers_are_contested(self):
        """Genuinely different numeric values should be detected as contested."""
        from orbit_or import db

        topic_id = 99981
        eid, aid = self._seed(topic_id)
        try:
            with db.get_db() as conn:
                self._direct_insert(
                    conn,
                    topic_id,
                    eid,
                    aid,
                    "47%",
                    "2027",
                    "M1",
                    value_numeric_min=47.0,
                    value_numeric_max=47.0,
                    unit="%",
                )
                self._direct_insert(
                    conn,
                    topic_id,
                    eid,
                    aid,
                    "65%",
                    "2027",
                    "M2",
                    value_numeric_min=65.0,
                    value_numeric_max=65.0,
                    unit="%",
                )
            contested = db.get_contested_ledger_pairs(topic_id)
            assert (
                len(contested) == 1
            ), f"Expected 1 contested pair, got {len(contested)}"
            assert len(contested[0]["entries"]) == 2
        finally:
            self._cleanup(topic_id)

    def test_empty_values_excluded_from_contested(self):
        """Empty value rows should not appear in contested results."""
        from orbit_or import db

        topic_id = 99982
        eid, aid = self._seed(topic_id)
        try:
            # Direct insert to bypass empty-value rejection
            with db.get_db() as conn:
                conn.execute(
                    """INSERT INTO Ledger (topic_id, entity_id, attribute_id, value,
                       normalized_timeframe, entry_type, source_ref, status)
                       VALUES (?, ?, ?, '', '2027', 'test', 'M1', 'accepted')""",
                    (topic_id, eid, aid),
                )
                conn.execute(
                    """INSERT INTO Ledger (topic_id, entity_id, attribute_id, value,
                       normalized_timeframe, entry_type, source_ref, status)
                       VALUES (?, ?, ?, '50%', '2027', 'test', 'M2', 'accepted')""",
                    (topic_id, eid, aid),
                )
            contested = db.get_contested_ledger_pairs(topic_id)
            # Should not flag empty-vs-nonempty as contested
            assert len(contested) == 0
        finally:
            self._cleanup(topic_id)
