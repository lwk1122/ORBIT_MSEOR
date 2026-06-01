"""Tests for G.4 Claim V2: structured claims, quality gates, formal_claim processing."""

import json
import os
import pytest

from orbit_or.db import get_db, get_db_path, init_db
from orbit_or import api
from orbit_or.formal_claims import (
    review_formal_claim_candidate,
    review_pending_formal_claim_candidates,
    validate_formal_claim_candidate_payload,
)
from orbit_or.server import validate_formal_claim, _normalize_message_contract


@pytest.fixture(autouse=True)
def setup_teardown():
    os.environ["TESTING"] = "1"
    db_path = get_db_path()
    if os.path.exists(db_path):
        os.remove(db_path)
    init_db()
    yield
    if os.path.exists(db_path):
        os.remove(db_path)


# --- Schema tests ---


def test_claim_new_columns_exist():
    with get_db() as conn:
        cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(Claim)").fetchall()
        }
        for col in [
            "claim_type",
            "scope_tags",
            "scope_context",
            "falsification_criteria",
            "inference_logic",
            "conclusion",
            "evidence_strength",
            "scope_breadth",
            "submitted_by",
        ]:
            assert col in cols, f"Missing column: {col}"


def test_claim_candidate_new_columns_exist():
    with get_db() as conn:
        cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(ClaimCandidate)").fetchall()
        }
        for col in [
            "claim_type",
            "scope_tags",
            "scope_context",
            "falsification_criteria",
            "inference_logic",
            "conclusion",
            "evidence_strength",
            "scope_breadth",
            "submitted_by",
        ]:
            assert col in cols, f"Missing column: {col}"


def test_message_has_formal_claim_column():
    with get_db() as conn:
        cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(Message)").fetchall()
        }
        assert "has_formal_claim" in cols


def test_knowledge_edge_allows_refutes():
    """KnowledgeEdge CHECK constraint should now allow 'refutes' and 'qualifies'."""
    topic_id = api.create_topic("Test", "Detail")
    # This should NOT raise IntegrityError after migration
    edge_id = api.insert_knowledge_edge(
        topic_id, 1, "fact", 2, "claim", "refutes", created_by="test"
    )
    assert edge_id is not None


def test_knowledge_edge_allows_qualifies():
    topic_id = api.create_topic("Test", "Detail")
    edge_id = api.insert_knowledge_edge(
        topic_id, 1, "claim", 2, "claim", "qualifies", created_by="test"
    )
    assert edge_id is not None


# --- insert_claim with new fields ---


def test_insert_claim_with_v2_fields():
    topic_id = api.create_topic("Test", "Detail")
    claim_id = api.insert_claim(
        topic_id,
        None,
        "MLP outperforms RF on wine",
        claim_type="comparison",
        scope_tags='["dataset:wine", "metric:f1"]',
        scope_context="default hyperparameters",
        falsification_criteria="RF F1 > MLP F1 + 0.01 on wine with 30 seeds",
        inference_logic="F67 shows MLP=0.9834, F84 shows RF=0.9784",
        conclusion="MLP marginally outperforms RF",
        evidence_strength=6.0,
        scope_breadth=2.0,
        submitted_by="analyst",
    )
    with get_db() as conn:
        row = conn.execute("SELECT * FROM Claim WHERE id = ?", (claim_id,)).fetchone()
    assert row["claim_type"] == "comparison"
    assert row["scope_tags"] == '["dataset:wine", "metric:f1"]'
    assert row["submitted_by"] == "analyst"
    assert row["evidence_strength"] == 6.0


# --- Quality gates ---


def test_gate_passes_valid_claim():
    result = validate_formal_claim(
        {
            "claim_type": "comparison",
            "conclusion": "MLP outperforms RF by 0.005 F1 on wine",
            "scope_tags": ["dataset:wine", "metric:f1"],
            "evidence_strength": 7,
            "falsification_criteria": "RF F1 > MLP F1 + 0.01 on wine with 30 seeds",
        }
    )
    assert result["passed"] is True


def test_gate_passes_optimization_result_claim_type():
    result = validate_formal_claim(
        {
            "claim_type": "optimization_result",
            "conclusion": "Solver run 7 found objective value 12.5 for model O3",
            "scope_tags": ["problem:2", "artifact:3", "solver:scipy_milp"],
            "scope_context": "Problem 2, artifact O3, solver scipy_milp.",
            "evidence_strength": 8,
            "falsification_criteria": "independent solver objective differs by more than 1e-6",
        }
    )
    assert result["passed"] is True


def test_gate_rejects_short_conclusion():
    result = validate_formal_claim(
        {
            "conclusion": "yes",
            "evidence_strength": 5,
        }
    )
    assert result["passed"] is False
    assert "too short" in result["error"]


def test_gate_rejects_vague_terms():
    result = validate_formal_claim(
        {
            "conclusion": "MLP shows competitive performance against RF",
            "evidence_strength": 5,
        }
    )
    assert result["passed"] is False
    assert "competitive" in result["error"].lower()


def test_gate_allows_vague_with_numbers():
    result = validate_formal_claim(
        {
            "conclusion": "MLP shows competitive performance (within 0.02 F1) against RF",
            "evidence_strength": 5,
        }
    )
    # "competitive" is near numbers, so it should pass
    assert result["passed"] is True


def test_gate_rejects_comparison_without_dataset():
    result = validate_formal_claim(
        {
            "claim_type": "comparison",
            "conclusion": "MLP outperforms RF by a significant 0.05 margin",
            "scope_tags": ["metric:f1"],  # no dataset: tag
            "evidence_strength": 7,
        }
    )
    assert result["passed"] is False
    assert "dataset" in result["error"]


def test_gate_rejects_missing_evidence_strength():
    result = validate_formal_claim(
        {
            "conclusion": "MLP needs >1000 samples to beat RF on tabular data",
        }
    )
    assert result["passed"] is False
    assert "evidence_strength" in result["error"]


def test_gate_rejects_absence_without_inference():
    result = validate_formal_claim(
        {
            "conclusion": "There is an absence of benchmark data for this model",
            "evidence_strength": 3,
        }
    )
    assert result["passed"] is False
    assert "evidence-gap" in result["error"]


def test_deterministic_formal_claim_review_accepts_supported_candidate():
    topic_id = api.create_topic("Formal review", "Review claims.")
    fact_id = api.insert_fact(
        topic_id,
        "Dataset A shows MLP F1 is 0.91 and RF F1 is 0.86.",
        "Librarian",
        review_status="accepted",
    )
    candidate_id = api.create_claim_candidate(
        topic_id,
        None,
        None,
        "MLP outperforms RF by 0.05 F1 on dataset A.",
        support_fact_ids_json=json.dumps([fact_id]),
        rationale_short="F1 values in the accepted fact support the comparison.",
        claim_type="comparison",
        scope_tags=json.dumps(["dataset:A", "metric:f1"]),
        scope_context="Dataset A, F1 metric.",
        falsification_criteria="RF F1 exceeds MLP F1 by more than 0.01.",
        inference_logic="Accepted fact reports MLP F1 0.91 and RF F1 0.86.",
        conclusion="MLP outperforms RF by 0.05 F1 on dataset A.",
        evidence_strength=7,
        scope_breadth=2,
        submitted_by="analyst",
    )
    candidate = api.get_claim_candidates(topic_id)[0]

    result = review_formal_claim_candidate(topic_id, candidate)

    assert result["accepted"] is True
    reviewed = api.get_claim_candidates(topic_id)[0]
    assert reviewed["status"] == "accepted"
    claim = api.get_claims(topic_id)[0]
    assert claim["claim_type"] == "comparison"
    edges = api.get_knowledge_edges(
        topic_id,
        source_type="fact",
        target_type="claim",
        relation="supports",
    )
    assert edges[0]["source_id"] == fact_id
    assert edges[0]["target_id"] == result["claim_id"]
    assert reviewed["accepted_claim_id"] == result["claim_id"]
    assert reviewed["id"] == candidate_id


def test_deterministic_formal_claim_review_rejects_missing_scope_and_support():
    payload = {
        "claim_type": "boundary",
        "conclusion": "This method is robust for broad use.",
        "evidence_strength": 5,
        "falsification_criteria": "failure rate exceeds 10%",
        "inference_logic": "The candidate lacks cited support facts.",
        "support_fact_ids": [],
    }

    issues = validate_formal_claim_candidate_payload(payload)

    assert "vague_unquantified_conclusion" in issues
    assert "missing_scope_tags" in issues
    assert "missing_support_facts" in issues


def test_review_pending_formal_claim_candidates_deduplicates():
    topic_id = api.create_topic("Formal review", "Review claims.")
    fact_id = api.insert_fact(
        topic_id,
        "Dataset A shows MLP F1 is 0.91 and RF F1 is 0.86.",
        "Librarian",
        review_status="accepted",
    )
    for _ in range(2):
        api.create_claim_candidate(
            topic_id,
            None,
            None,
            "MLP outperforms RF by 0.05 F1 on dataset A.",
            support_fact_ids_json=json.dumps([fact_id]),
            rationale_short="F1 values in the accepted fact support the comparison.",
            claim_type="comparison",
            scope_tags=json.dumps(["dataset:A", "metric:f1"]),
            scope_context="Dataset A, F1 metric.",
            falsification_criteria="RF F1 exceeds MLP F1 by more than 0.01.",
            inference_logic="Accepted fact reports MLP F1 0.91 and RF F1 0.86.",
            conclusion="MLP outperforms RF by 0.05 F1 on dataset A.",
            evidence_strength=7,
            scope_breadth=2,
            submitted_by="analyst",
        )

    results = review_pending_formal_claim_candidates(topic_id)

    assert [result["reason"] for result in results] == ["accepted", "duplicate"]
    assert len(api.get_claims(topic_id)) == 1


# --- _normalize_message_contract extracts formal_claim ---


def test_normalize_extracts_formal_claim():
    raw = json.dumps(
        {
            "action": "post_message",
            "content": "Based on F67 and F84, the gap is marginal.",
            "formal_claim": {
                "claim_type": "comparison",
                "conclusion": "MLP marginally outperforms RF",
            },
        }
    )
    parsed = _normalize_message_contract(raw)
    assert parsed["parsed_ok"] is True
    assert parsed["formal_claim"] is not None
    assert parsed["formal_claim"]["claim_type"] == "comparison"


def test_normalize_no_formal_claim():
    raw = json.dumps(
        {
            "action": "post_message",
            "content": "I disagree with the analyst's position.",
        }
    )
    parsed = _normalize_message_contract(raw)
    assert parsed["parsed_ok"] is True
    assert parsed.get("formal_claim") is None


def test_normalize_backward_compat():
    """Old-format messages without formal_claim should parse fine."""
    raw = json.dumps(
        {
            "action": "post_message",
            "content": "Standard workspace message.",
            "confidence_score": 7.5,
        }
    )
    parsed = _normalize_message_contract(raw)
    assert parsed["parsed_ok"] is True
    assert parsed["content"] == "Standard workspace message."
    assert parsed.get("formal_claim") is None
