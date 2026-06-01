import os
import pytest

from orbit_or.db import get_db, get_db_path, init_db
from orbit_or import api
from orbit_or.viki import (
    run_watchdog,
    _check_claim_conflicts,
    _check_missing_spo,
    _check_solver_claim_evidence,
    _filter_new_issues,
    _handle_solver_claim_stale_evidence,
    _handle_solver_claim_status_mismatch,
)
from orbit_or.db import (
    upsert_viki_issue,
    increment_viki_check,
    mark_viki_resolved,
    mark_viki_wont_fix,
    get_open_viki_issues,
)


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


@pytest.fixture
def topic_id():
    return api.create_topic("Test Topic", "Detail")


# --- VikiTracker CRUD ---


def test_viki_tracker_table_exists():
    with get_db() as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='VikiTracker'"
        ).fetchone()
        assert row is not None


def test_upsert_viki_issue():
    issue = upsert_viki_issue("Fact", 1, "missing_spo")
    assert issue["target_table"] == "Fact"
    assert issue["target_id"] == 1
    assert issue["status"] == "open"
    assert issue["check_count"] == 0


def test_upsert_viki_issue_idempotent():
    issue1 = upsert_viki_issue("Fact", 1, "missing_spo")
    issue2 = upsert_viki_issue("Fact", 1, "missing_spo")
    assert issue1["id"] == issue2["id"]


def test_increment_viki_check():
    issue = upsert_viki_issue("Fact", 1, "missing_spo")
    increment_viki_check(issue["id"])
    with get_db() as conn:
        row = conn.execute(
            "SELECT check_count FROM VikiTracker WHERE id = ?", (issue["id"],)
        ).fetchone()
        assert row["check_count"] == 1


def test_mark_resolved():
    issue = upsert_viki_issue("Fact", 1, "missing_spo")
    mark_viki_resolved(issue["id"], "backfilled")
    with get_db() as conn:
        row = conn.execute(
            "SELECT status, resolution FROM VikiTracker WHERE id = ?", (issue["id"],)
        ).fetchone()
        assert row["status"] == "resolved"
        assert row["resolution"] == "backfilled"


def test_mark_wont_fix():
    issue = upsert_viki_issue("Fact", 1, "missing_spo")
    mark_viki_wont_fix(issue["id"], "max checks reached")
    with get_db() as conn:
        row = conn.execute(
            "SELECT status FROM VikiTracker WHERE id = ?", (issue["id"],)
        ).fetchone()
        assert row["status"] == "wont_fix"


def test_get_open_issues_excludes_resolved():
    upsert_viki_issue("Fact", 1, "missing_spo")
    issue2 = upsert_viki_issue("Fact", 2, "missing_spo")
    mark_viki_resolved(issue2["id"], "done")
    open_issues = get_open_viki_issues("missing_spo")
    assert len(open_issues) == 1
    assert open_issues[0]["target_id"] == 1


def test_get_open_issues_excludes_maxed():
    issue = upsert_viki_issue("Fact", 1, "missing_spo")
    for _ in range(3):
        increment_viki_check(issue["id"])
    open_issues = get_open_viki_issues("missing_spo")
    assert len(open_issues) == 0


# --- Watchdog Checks ---


def test_check_missing_spo(topic_id):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO Fact (topic_id, content, source) VALUES (?, ?, ?)",
            (topic_id, "Test fact with no SPO", "test"),
        )
    issues = _check_missing_spo(topic_id)
    assert len(issues) >= 1
    assert issues[0]["issue_type"] == "missing_spo"


def test_check_missing_spo_none_when_filled(topic_id):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO Fact (topic_id, content, source, subject, predicate) VALUES (?, ?, ?, ?, ?)",
            (topic_id, "Test fact", "test", "MLP", "achieves"),
        )
    issues = _check_missing_spo(topic_id)
    assert len(issues) == 0


def test_check_claim_conflicts(topic_id):
    # Create two claims with a conflicts_with edge
    claim1 = api.insert_claim(topic_id, None, "Claim A", claim_score=8.0)
    claim2 = api.insert_claim(topic_id, None, "Claim B", claim_score=5.0)
    api.insert_knowledge_edge(
        topic_id, claim1, "claim", claim2, "claim", "conflicts_with", created_by="test"
    )
    issues = _check_claim_conflicts(topic_id)
    assert len(issues) == 1
    assert issues[0]["issue_type"] == "claim_conflict"
    assert issues[0]["source_id"] == claim1
    assert issues[0]["target_claim_id"] == claim2


def test_check_solver_claim_missing_evidence(topic_id):
    api.create_claim_candidate(
        topic_id,
        None,
        None,
        "Solver found optimal objective value 1.0 for O1",
        claim_type="optimization_result",
        conclusion="Solver found optimal objective value 1.0 for O1",
        inference_logic="Derived from an unstated solver run.",
        evidence_strength=8,
    )

    issues = _check_solver_claim_evidence(topic_id)

    assert len(issues) == 1
    assert issues[0]["target_table"] == "ClaimCandidate"
    assert issues[0]["issue_type"] == "solver_claim_missing_evidence"


def test_check_solver_claim_status_mismatch(topic_id):
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="LP",
        source_text="Minimize x subject to x >= 1.",
    )
    artifact_id = api.insert_optimization_artifact(
        problem_id=problem_id,
        topic_id=topic_id,
        artifact_type="lp_model",
        model_language="lp",
        content="Minimize\n obj: x\nSubject To\n c1: x >= 1\nEnd",
    )
    run_id = api.insert_solver_run(
        artifact_id=artifact_id,
        problem_id=problem_id,
        topic_id=topic_id,
        solver_backend="stub",
        status="solver_infeasible",
    )
    api.create_claim_candidate(
        topic_id,
        None,
        None,
        f"SolverRun {run_id} found optimal objective value 1.0 for O{artifact_id}",
        claim_type="optimization_result",
        conclusion=f"SolverRun {run_id} found optimal objective value 1.0",
        inference_logic=f"Derived from SolverRun {run_id}.",
        evidence_strength=8,
    )

    issues = _check_solver_claim_evidence(topic_id)

    assert len(issues) == 1
    assert issues[0]["issue_type"] == "solver_claim_status_mismatch"


def test_solver_claim_status_mismatch_handler_rejects_candidate(topic_id):
    candidate_id = api.create_claim_candidate(
        topic_id,
        None,
        None,
        "SolverRun 999 found optimal objective value 1.0 for O1",
        claim_type="optimization_result",
        conclusion="SolverRun 999 found optimal objective value 1.0",
        inference_logic="Derived from SolverRun 999.",
        evidence_strength=8,
    )

    resolution = _handle_solver_claim_status_mismatch(
        {
            "target_table": "ClaimCandidate",
            "target_id": candidate_id,
            "issue_type": "solver_claim_status_mismatch",
        }
    )

    assert resolution == "candidate_rejected_solver_claim_status_mismatch"
    candidate = api.get_claim_candidates(topic_id)[0]
    assert candidate["status"] == "rejected"
    assert "solver_claim_status_mismatch" in candidate["review_note"]


def test_check_solver_claim_stale_evidence_and_handler_contests_claim(topic_id):
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="LP",
        source_text="Minimize x subject to x >= 1.",
    )
    artifact_id = api.insert_optimization_artifact(
        problem_id=problem_id,
        topic_id=topic_id,
        artifact_type="lp_model",
        model_language="lp",
        content="Minimize\n obj: x\nSubject To\n c1: x >= 1\nEnd",
    )
    code_evidence_id = api.insert_code_evidence(
        topic_id,
        None,
        hypothesis="Solve LP",
        source_code="lp",
        stdout="optimal",
        stderr="",
        exit_code=0,
        execution_time_s=0.01,
        iterations=1,
        success=True,
        requesting_role="or_solver",
        summary="optimal",
    )
    run_id = api.insert_solver_run(
        artifact_id=artifact_id,
        problem_id=problem_id,
        topic_id=topic_id,
        solver_backend="stub",
        status="optimal",
        code_evidence_id=code_evidence_id,
    )
    claim_id = api.insert_claim(
        topic_id,
        None,
        f"SolverRun {run_id} found optimal objective value 1.0 [E{code_evidence_id}]",
        claim_type="optimization_result",
        conclusion=f"SolverRun {run_id} found optimal objective value 1.0",
        inference_logic=f"Derived from SolverRun {run_id}.",
        evidence_strength=8,
    )
    api.insert_model_diagnostic(
        problem_id=problem_id,
        topic_id=topic_id,
        artifact_id=artifact_id,
        solver_run_id=run_id,
        diagnostic_type="linked_component_inactive",
        severity="error",
        message="linked component rejected",
    )

    issues = _check_solver_claim_evidence(topic_id)

    assert len(issues) == 1
    assert issues[0]["issue_type"] == "solver_claim_stale_evidence"
    resolution = _handle_solver_claim_stale_evidence(issues[0])
    assert resolution == "claim_contested_solver_claim_stale_evidence"
    claim = api.get_claims(topic_id)[0]
    assert claim["id"] == claim_id
    assert claim["status"] == "contested"


def test_filter_skips_resolved(topic_id):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO Fact (topic_id, content, source) VALUES (?, ?, ?)",
            (topic_id, "No SPO", "test"),
        )
    issues = _check_missing_spo(topic_id)
    assert len(issues) >= 1

    # Filter once — should pass through
    filtered = _filter_new_issues(issues)
    assert len(filtered) >= 1

    # Mark as resolved
    mark_viki_resolved(filtered[0]["tracker_id"], "done")

    # Filter again — should be empty
    issues2 = _check_missing_spo(topic_id)
    filtered2 = _filter_new_issues(issues2)
    assert len(filtered2) == 0


def test_filter_skips_maxed_out(topic_id):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO Fact (topic_id, content, source) VALUES (?, ?, ?)",
            (topic_id, "No SPO", "test"),
        )
    issues = _check_missing_spo(topic_id)
    filtered = _filter_new_issues(issues)
    assert len(filtered) >= 1

    # Max out the check count
    tracker_id = filtered[0]["tracker_id"]
    for _ in range(3):
        increment_viki_check(tracker_id)

    issues2 = _check_missing_spo(topic_id)
    filtered2 = _filter_new_issues(issues2)
    assert len(filtered2) == 0


def test_run_watchdog_integration(topic_id):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO Fact (topic_id, content, source) VALUES (?, ?, ?)",
            (topic_id, "Fact without SPO", "test"),
        )
    issues = run_watchdog(topic_id)
    types = {i["issue_type"] for i in issues}
    assert "missing_spo" in types
