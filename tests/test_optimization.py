import json
import os

import pytest
from unittest.mock import AsyncMock, patch

from orbit_or import api, topic_config
from orbit_or.broker import BrokerResponse
from orbit_or.db import get_db_path, init_db
from orbit_or.optimization import (
    apply_lp_repair_candidate,
    apply_mps_repair_candidate,
    available_solver_backends,
    propagate_component_status_to_solver_evidence,
    build_component_extraction_prompt,
    build_model_ir_from_components,
    build_lp_repair_prompt,
    build_lp_artifact_from_components,
    build_mps_repair_prompt,
    build_mps_artifact_from_lp,
    build_semantic_repair_prompt,
    build_solver_claim_payload,
    classify_solver_failure,
    create_solver_claim_candidate,
    extract_component_candidate_tournament,
    extract_and_persist_components,
    normalize_lp_repair_candidate,
    normalize_mps_repair_candidate,
    parse_lp_artifact,
    parse_mps_artifact,
    persist_component_payloads,
    persist_model_ir_from_components,
    persist_lp_artifact_from_components,
    persist_lp_artifact,
    persist_mps_artifact_from_lp,
    repair_artifact_semantics_with_llm,
    repair_lp_artifact_with_llm,
    repair_mps_artifact_with_llm,
    classify_modeling_error,
    rank_modeling_candidates,
    review_pending_solver_claim_candidates,
    review_solver_claim_candidate,
    select_best_modeling_candidate,
    solve_mps_artifact,
    solve_optimization_artifact,
    solve_lp_artifact,
    validate_model_specification,
    validate_modeling_techniques,
    validate_component_payload,
    validate_artifact_component_semantics,
    validate_lp_artifact,
    validate_mps_artifact,
    validate_solver_claim_payload,
)
from orbit_or.jtms import jtms_sweep
from orbit_or.server import _advance_mse_workflow_deterministically


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


def test_optimization_problem_component_artifact_lifecycle():
    topic_id = api.create_topic("Facility location", "Choose warehouses.")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="Warehouse location",
        source_text="Choose warehouses to minimize cost under capacity constraints.",
        problem_class="facility_location",
        created_by="analyst",
    )
    component_id = api.insert_optimization_component(
        problem_id=problem_id,
        topic_id=topic_id,
        component_type="decision_variable",
        natural_text="x_i indicates whether warehouse i is opened.",
        formal_text="x_i in {0,1}",
        symbol="x_i",
    )
    lp = "Minimize\n obj: x\nSubject To\n c1: x >= 1\nBinary\n x\nEnd"
    artifact_id = api.insert_optimization_artifact(
        problem_id=problem_id,
        topic_id=topic_id,
        artifact_type="lp_model",
        model_language="lp",
        content=lp,
        parser_status="valid",
        linked_component_ids_json=f"[{component_id}]",
    )
    run_id = api.insert_solver_run(
        artifact_id=artifact_id,
        problem_id=problem_id,
        topic_id=topic_id,
        solver_backend="stub",
        status="optimal",
        objective_value=1.0,
    )

    assert api.get_optimization_problem(problem_id)["problem_class"] == "facility_location"
    assert api.get_optimization_components(problem_id)[0]["id"] == component_id
    assert api.update_optimization_component_review(
        component_id, review_status="reviewed"
    )
    assert api.get_optimization_components(problem_id)[0]["review_status"] == "reviewed"
    assert api.get_optimization_artifacts(problem_id)[0]["id"] == artifact_id
    assert api.get_solver_runs(problem_id)[0]["id"] == run_id

    snapshot = api.get_mse_review_snapshot(topic_id)
    assert snapshot["review_counts"]["problems"] == 1
    assert snapshot["review_counts"]["components"] == 1
    assert snapshot["problems"][0]["solver_runs"][0]["id"] == run_id


def test_mse_fast_workflow_advances_lp_artifacts_to_solver_claim():
    topic_id = api.create_topic("MSE LP", "Minimize x subject to x >= 1.")
    subtopic_id = api.create_subtopic(topic_id, "LP", "Minimize x subject to x >= 1.")
    topic_config.set_config(topic_id, "domain_profile", "mse")
    topic_config.set_config(topic_id, "mse_workflow_mode", "modeling_fast")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        subtopic_id=subtopic_id,
        title="LP",
        source_text="Minimize x subject to x >= 1.",
    )
    refs = '["D1"]'
    api.insert_optimization_component(
        problem_id=problem_id,
        topic_id=topic_id,
        component_type="decision_variable",
        natural_text="x is the decision variable.",
        formal_text="x >= 0",
        symbol="x",
        source_refs_json=refs,
        review_status="candidate",
    )
    api.insert_optimization_component(
        problem_id=problem_id,
        topic_id=topic_id,
        component_type="objective",
        natural_text="Minimize x.",
        formal_text="obj: x",
        symbol="obj",
        source_refs_json=refs,
        review_status="candidate",
    )
    api.insert_optimization_component(
        problem_id=problem_id,
        topic_id=topic_id,
        component_type="constraint",
        natural_text="x must be at least 1.",
        formal_text="c1: x >= 1",
        symbol="c1",
        source_refs_json=refs,
        review_status="candidate",
    )

    snapshot = _advance_mse_workflow_deterministically(
        {"topic_id": topic_id, "subtopic_id": subtopic_id}
    )

    assert snapshot["solved"] is True
    assert api.get_optimization_model_irs(problem_id)[0]["status"] == "valid"
    assert api.get_optimization_artifacts(problem_id)[0]["parser_status"] == "valid"
    assert api.get_solver_runs(problem_id)[0]["status"] == "optimal"
    assert any(
        claim.get("claim_type") == "optimization_result"
        for claim in api.get_claims(topic_id)
    )


def test_mse_fast_workflow_blocks_incomplete_specification():
    topic_id = api.create_topic("Incomplete MSE LP", "Minimize x.")
    subtopic_id = api.create_subtopic(topic_id, "LP", "Minimize x.")
    topic_config.set_config(topic_id, "domain_profile", "mse")
    topic_config.set_config(topic_id, "mse_workflow_mode", "modeling_fast")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        subtopic_id=subtopic_id,
        title="Incomplete LP",
        source_text="Minimize x.",
    )
    api.insert_optimization_component(
        problem_id=problem_id,
        topic_id=topic_id,
        component_type="objective",
        natural_text="Minimize x.",
        formal_text="obj: x",
        symbol="obj",
        source_refs_json='["D1"]',
        review_status="candidate",
    )

    snapshot = _advance_mse_workflow_deterministically(
        {"topic_id": topic_id, "subtopic_id": subtopic_id}
    )
    diagnostics = api.get_model_diagnostics(problem_id, status="open")

    assert snapshot["status"] == "specification_gap"
    assert api.get_optimization_artifacts(problem_id) == []
    assert {item["diagnostic_type"] for item in diagnostics} >= {
        "missing_constraints",
        "missing_decision_variables",
    }


def test_mse_provenance_report_exports_claim_solver_links():
    topic_id = api.create_topic("MSE report", "Review provenance.")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="LP",
        source_text="Minimize x subject to x >= 1.",
    )
    component_id = api.insert_optimization_component(
        problem_id=problem_id,
        topic_id=topic_id,
        component_type="decision_variable",
        natural_text="x is quantity.",
        symbol="x",
        source_refs_json='["D1"]',
        review_status="reviewed",
    )
    artifact_id = api.insert_optimization_artifact(
        problem_id=problem_id,
        topic_id=topic_id,
        artifact_type="lp_model",
        model_language="lp",
        content="Minimize\n obj: x\nSubject To\n c1: x >= 1\nEnd",
        linked_component_ids_json=json.dumps([component_id]),
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
        objective_value=1.0,
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
    api.insert_knowledge_edge(
        topic_id,
        code_evidence_id,
        "code_evidence",
        claim_id,
        "claim",
        "supports",
        created_by="test",
    )

    report = api.get_mse_provenance_report(topic_id)
    markdown = api.render_mse_provenance_markdown(report)

    assert report["review_counts"]["problems"] == 1
    assert report["problems"][0]["components"][0]["source_refs"] == ["D1"]
    assert (
        report["problems"][0]["solver_runs"][0]["code_evidence_id"]
        == code_evidence_id
    )
    assert report["solver_claims"][0]["solver_run_ids"] == [run_id]
    assert report["solver_claims"][0]["support_edges"][0]["source_id"] == code_evidence_id
    assert "MSE Provenance Report" in markdown
    assert f"SolverRun {run_id}" in markdown


def test_validate_component_payload_and_lp_artifact():
    issues = validate_component_payload(
        {
            "component_type": "parameter",
            "natural_text": "capacity is 10",
            "symbol": "cap_i",
        }
    )
    assert [issue.issue_type for issue in issues] == ["missing_unit"]

    valid = validate_lp_artifact(
        "Maximize\n obj: 3 x\nSubject To\n c1: x <= 4\nBounds\n x >= 0\nEnd"
    )
    assert valid["status"] == "valid"

    invalid = validate_lp_artifact("x <= 4")
    assert invalid["status"] == "invalid"
    assert "missing_objective_section" in {
        issue["issue_type"] for issue in invalid["issues"]
    }
    repair_prompt = build_lp_repair_prompt("x <= 4", invalid)
    assert "Repair this LP artifact" in repair_prompt
    assert "missing_objective_section" in repair_prompt

    invalid_mps = validate_mps_artifact("This model is:\nNAME BAD\nROWS\n N OBJ\nENDATA")
    assert invalid_mps["status"] == "invalid"
    assert "non_model_preamble" in {
        issue["issue_type"] for issue in invalid_mps["issues"]
    }
    mps_repair_prompt = build_mps_repair_prompt("ROWS\n N OBJ", invalid_mps)
    assert "Repair this MPS artifact" in mps_repair_prompt
    assert "non_model_preamble" in mps_repair_prompt


def test_modeling_error_taxonomy_and_technique_validation():
    issues = [
        {
            "issue_type": "linked_constraint_missing",
            "severity": "error",
            "message": "Demand constraint missing.",
        }
    ]
    assert classify_modeling_error(issues=issues) == "low_model_completeness"
    assert (
        classify_modeling_error(
            issues=[
                {
                    "issue_type": "objective_direction_mismatch",
                    "severity": "error",
                    "message": "Expected maximize.",
                }
            ]
        )
        == "objective_constraint_translation_error"
    )

    technique_issues = validate_modeling_techniques(
        [
            {
                "component_type": "constraint",
                "natural_text": "If a truck is selected, train cannot be selected.",
                "formal_text": "x_truck + x_train <= 1",
            }
        ],
        content="x_truck + x_train <= 1 using Big M",
    )
    issue_types = {issue.issue_type for issue in technique_issues}

    assert "big_m_without_binary_indicator" in issue_types
    assert "logical_constraint_without_indicator" in issue_types


def test_modeling_technique_validation_does_not_flag_set_symbol_m_as_big_m():
    issues = validate_modeling_techniques(
        [
            {
                "component_type": "set",
                "symbol": "M",
                "natural_text": "Transportation modes",
                "formal_text": "M = {boat, canoe}",
            },
            {
                "component_type": "constraint",
                "natural_text": "At least 60% of trips are canoes.",
                "formal_text": "2 c >= 3 b",
            },
        ]
    )

    assert "big_m_without_binary_indicator" not in {issue.issue_type for issue in issues}


def test_modeling_technique_validation_does_not_flag_simple_or_enumeration():
    issues = validate_modeling_techniques(
        [
            {
                "component_type": "index",
                "natural_text": "i denotes container type, i = small or large.",
            },
            {
                "component_type": "constraint",
                "natural_text": "Water usage must not exceed 500 units.",
                "formal_text": "10 x_s + 20 x_l <= 500",
            },
        ]
    )

    assert "logical_constraint_without_indicator" not in {
        issue.issue_type for issue in issues
    }


def test_model_ir_and_specification_validation_from_components():
    topic_id = api.create_topic("IR topic", "Minimize x subject to x >= 1.")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="IR LP",
        source_text="Minimize x subject to x >= 1.",
    )
    refs = '["D1"]'
    components = [
        {
            "id": api.insert_optimization_component(
                problem_id=problem_id,
                topic_id=topic_id,
                component_type="decision_variable",
                natural_text="x is the quantity.",
                formal_text="x >= 0",
                symbol="x",
                source_refs_json=refs,
                review_status="reviewed",
            ),
            "component_type": "decision_variable",
            "natural_text": "x is the quantity.",
            "formal_text": "x >= 0",
            "symbol": "x",
            "source_refs_json": refs,
            "review_status": "reviewed",
        },
        {
            "id": api.insert_optimization_component(
                problem_id=problem_id,
                topic_id=topic_id,
                component_type="objective",
                natural_text="Minimize x.",
                formal_text="obj: x",
                symbol="obj",
                source_refs_json=refs,
                review_status="reviewed",
            ),
            "component_type": "objective",
            "natural_text": "Minimize x.",
            "formal_text": "obj: x",
            "symbol": "obj",
            "source_refs_json": refs,
            "review_status": "reviewed",
        },
        {
            "id": api.insert_optimization_component(
                problem_id=problem_id,
                topic_id=topic_id,
                component_type="constraint",
                natural_text="x must be at least 1.",
                formal_text="c1: x >= 1",
                symbol="c1",
                source_refs_json=refs,
                review_status="reviewed",
            ),
            "component_type": "constraint",
            "natural_text": "x must be at least 1.",
            "formal_text": "c1: x >= 1",
            "symbol": "c1",
            "source_refs_json": refs,
            "review_status": "reviewed",
        },
    ]
    problem = api.get_optimization_problem(problem_id)

    assert validate_model_specification(components) == []
    built = build_model_ir_from_components(problem=problem, components=components)
    persisted = persist_model_ir_from_components(
        topic_id=topic_id,
        problem=problem,
        components=components,
    )

    assert built["accepted"] is True
    assert built["ir"]["schema"] == "orbit_model_ir.v1"
    assert len(built["ir"]["model"]["decision_variables"]) == 1
    assert api.get_optimization_model_irs(problem_id)[0]["id"] == persisted["ir_id"]


def test_rank_modeling_candidates_prefers_valid_solved_complete_candidate():
    invalid = {
        "content": "x <= 1",
        "model_language": "lp",
        "components": [
            {
                "component_type": "constraint",
                "natural_text": "capacity",
            }
        ],
    }
    valid = {
        "content": "Minimize\n obj: x\nSubject To\n c1: x >= 1\nBounds\n x >= 0\nEnd",
        "model_language": "lp",
        "components": [
            {
                "component_type": "decision_variable",
                "symbol": "x",
                "source_refs_json": '["D1"]',
            },
            {
                "component_type": "objective",
                "formal_text": "minimize x",
                "source_refs_json": '["D1"]',
            },
            {
                "component_type": "constraint",
                "formal_text": "x >= 1",
                "source_refs_json": '["D1"]',
            },
        ],
        "solver_result": {"status": "optimal", "objective_value": 1.0},
    }

    ranked = rank_modeling_candidates([invalid, valid])

    assert ranked[0]["content"] == valid["content"]
    assert ranked[0]["rank"] == 1
    assert ranked[0]["modeling_error"] == "none"
    assert select_best_modeling_candidate([invalid, valid])["score"] > ranked[1]["score"]


def test_rank_modeling_candidates_uses_component_detail_tiebreaker():
    content = "Maximize\n obj: 2 x\nSubject To\n c1: x <= 10\nBounds\n x >= 0\nGeneral\n x\nEnd"
    minimal = {
        "content": content,
        "model_language": "lp",
        "components": [
            {
                "component_type": "decision_variable",
                "symbol": "x",
                "natural_text": "x quantity.",
                "source_refs_json": '["D1"]',
            },
            {
                "component_type": "objective",
                "formal_text": "maximize 2 x",
                "natural_text": "Maximize return.",
                "source_refs_json": '["D1"]',
            },
            {
                "component_type": "constraint",
                "formal_text": "x <= 10",
                "natural_text": "Capacity.",
                "source_refs_json": '["D1"]',
            },
        ],
    }
    detailed = {
        **minimal,
        "components": minimal["components"]
        + [
            {
                "component_type": "parameter",
                "formal_text": "profit = 2",
                "natural_text": "Profit per unit.",
                "source_refs_json": '["D1"]',
            },
            {
                "component_type": "parameter",
                "formal_text": "capacity = 10",
                "natural_text": "Capacity.",
                "source_refs_json": '["D1"]',
            },
        ],
    }

    ranked = rank_modeling_candidates([minimal, detailed])

    assert ranked[0]["components"] == detailed["components"]
    assert any(
        str(reason).startswith("component_detail_bonus=")
        for reason in ranked[0]["reasons"]
    )


def test_parse_and_persist_lp_artifact():
    topic_id = api.create_topic("LP topic", "Solve LP.")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="LP",
        source_text="Minimize x subject to x >= 1.",
    )
    lp = "Minimize\n obj: x\nSubject To\n c1: x >= 1\nBounds\n x >= 0\nEnd"

    parsed = parse_lp_artifact(lp)
    assert parsed.variables == ("x",)
    assert parsed.objective == (1.0,)
    assert parsed.constraints[0]["operator"] == ">="

    result = persist_lp_artifact(topic_id=topic_id, problem_id=problem_id, content=lp)
    assert result["validation"]["status"] == "valid"
    assert result["diagnostic_ids"] == []
    assert api.get_optimization_artifacts(problem_id)[0]["id"] == result["artifact_id"]


def test_lp_builder_infers_integer_domain_for_discrete_count_variables():
    generated = build_lp_artifact_from_components(
        [
            {
                "id": 1,
                "component_type": "decision_variable",
                "natural_text": "Number of Zodiac pills purchased.",
                "formal_text": "z",
                "symbol": "z",
                "domain": "nonnegative real",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
            {
                "id": 2,
                "component_type": "decision_variable",
                "natural_text": "Number of Sunny pills purchased.",
                "formal_text": "s",
                "symbol": "s",
                "domain": "nonnegative real",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
            {
                "id": 3,
                "component_type": "objective",
                "formal_text": "minimize 1 z + 3 s",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
            {
                "id": 4,
                "component_type": "constraint",
                "formal_text": "1.3 z + 1.2 s >= 5",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
            {
                "id": 5,
                "component_type": "constraint",
                "formal_text": "1.5 z + 5 s >= 10",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
        ]
    )

    assert generated["accepted"] is True
    assert "General\n z s" in generated["content"]


def test_lp_builder_parses_implicit_negative_variable_terms():
    generated = build_lp_artifact_from_components(
        [
            {
                "id": 1,
                "component_type": "decision_variable",
                "natural_text": "Number of sled dog trips.",
                "symbol": "d",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
            {
                "id": 2,
                "component_type": "decision_variable",
                "natural_text": "Number of truck trips.",
                "symbol": "t",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
            {
                "id": 3,
                "component_type": "objective",
                "formal_text": "maximize 100 d + 300 t",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
            {
                "id": 4,
                "component_type": "constraint",
                "formal_text": "t - d >= 1",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
        ]
    )

    assert generated["accepted"] is True
    assert "c1: -d + t >= 1" in generated["content"]


def test_lp_builder_promotes_single_objective_like_decision_problem():
    components = [
        {
            "id": 1,
            "component_type": "decision_problem",
            "natural_text": "Maximize total crop revenue.",
            "formal_text": "maximize 300 T + 450 P",
            "source_refs_json": '["D1"]',
            "review_status": "reviewed",
        },
        {
            "id": 2,
            "component_type": "decision_variable",
            "natural_text": "Acres of turnips.",
            "symbol": "T",
            "domain": "nonnegative real",
            "source_refs_json": '["D1"]',
            "review_status": "reviewed",
        },
        {
            "id": 3,
            "component_type": "decision_variable",
            "natural_text": "Acres of pumpkins.",
            "symbol": "P",
            "domain": "nonnegative real",
            "source_refs_json": '["D1"]',
            "review_status": "reviewed",
        },
        {
            "id": 4,
            "component_type": "constraint",
            "natural_text": "Available land.",
            "formal_text": "T + P <= 500",
            "source_refs_json": '["D1"]',
            "review_status": "reviewed",
        },
    ]

    generated = build_lp_artifact_from_components(components)
    issue_types = {
        issue.issue_type for issue in validate_model_specification(components)
    }

    assert generated["accepted"] is True
    assert "Maximize\n obj: 450 P + 300 T" in generated["content"]
    assert "invalid_objective_count" not in issue_types
    assert generated["component_ids"] == [2, 3, 4, 1]


def test_lp_builder_keeps_substantive_constraints_with_domain_metadata():
    generated = build_lp_artifact_from_components(
        [
            {
                "id": 1,
                "component_type": "decision_variable",
                "natural_text": "Number of bus trips.",
                "symbol": "b",
                "domain": "integer, non-negative",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
            {
                "id": 2,
                "component_type": "decision_variable",
                "natural_text": "Number of car trips.",
                "symbol": "c",
                "domain": "integer, non-negative",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
            {
                "id": 3,
                "component_type": "objective",
                "formal_text": "minimize 2 b + 1.5 c",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
            {
                "id": 4,
                "component_type": "constraint",
                "formal_text": "100 b + 40 c >= 1200",
                "domain": "b, c integer",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
        ]
    )

    assert generated["accepted"] is True
    assert "c1: 100 b + 40 c >= 1200" in generated["content"]


def test_lp_builder_expands_derived_variables_before_solving_constraints():
    generated = build_lp_artifact_from_components(
        [
            {
                "id": 1,
                "component_type": "decision_variable",
                "natural_text": "Number of camels.",
                "symbol": "c",
                "domain": "integer",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
            {
                "id": 2,
                "component_type": "decision_variable",
                "natural_text": "Number of horses.",
                "symbol": "h",
                "domain": "integer",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
            {
                "id": 3,
                "component_type": "derived_variable",
                "formal_text": "P = 50 c + 60 h",
                "symbol": "P",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
            {
                "id": 4,
                "component_type": "derived_variable",
                "formal_text": "F = 20 c + 30 h",
                "symbol": "F",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
            {
                "id": 5,
                "component_type": "objective",
                "formal_text": "minimize c + h",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
            {
                "id": 6,
                "component_type": "constraint",
                "formal_text": "P >= 1000",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
            {
                "id": 7,
                "component_type": "constraint",
                "formal_text": "F <= 450",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
            {
                "id": 8,
                "component_type": "constraint",
                "formal_text": "h <= c",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
        ]
    )

    assert generated["accepted"] is True
    assert set(generated["component_ids"]) == set(range(1, 9))
    assert "c1: 50 c + 60 h >= 1000" in generated["content"]
    assert "c2: 20 c + 30 h <= 450" in generated["content"]


def test_lp_builder_requires_reviewed_derived_variable_definitions():
    generated = build_lp_artifact_from_components(
        [
            {
                "id": 1,
                "component_type": "decision_variable",
                "natural_text": "Number of camels.",
                "symbol": "c",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
            {
                "id": 2,
                "component_type": "derived_variable",
                "formal_text": "P = 50 c",
                "symbol": "P",
                "source_refs_json": '["D1"]',
                "review_status": "candidate",
            },
            {
                "id": 3,
                "component_type": "objective",
                "formal_text": "minimize c",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
            {
                "id": 4,
                "component_type": "constraint",
                "formal_text": "P >= 1000",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
        ]
    )

    assert generated["accepted"] is False
    assert "unreviewed_component" in {
        issue["issue_type"] for issue in generated["issues"]
    }


def test_component_extraction_prompt_disambiguates_either_type_limits():
    prompt = build_component_extraction_prompt(
        "A company can sell at most 475 handbags of either type."
    )

    assert "`x_A + x_B <= N`" in prompt
    assert "Use separate per-type limits only" in prompt


def test_component_extraction_prompt_disambiguates_share_direction():
    prompt = build_component_extraction_prompt(
        "At most 40% of the concerts can be R&B."
    )

    assert "Y <= p(X + Y)" in prompt
    assert "Y >= p(X + Y)" in prompt


def test_component_extraction_prompt_requires_expanded_variable_consistency():
    prompt = build_component_extraction_prompt(
        "A patient can take pain killer 1 or pain killer 2."
    )

    assert "symbols `x1` and `x2`" in prompt
    assert "do not return only an indexed symbol such as `x_i`" in prompt


def test_lp_builder_normalizes_minimax_style_linear_formal_text():
    components = [
        {
            "id": 1,
            "component_type": "decision_variable",
            "natural_text": "Number of boat trips",
            "symbol": "x",
            "source_refs_json": '["D1"]',
            "review_status": "reviewed",
        },
        {
            "id": 2,
            "component_type": "decision_variable",
            "natural_text": "Number of canoe trips",
            "symbol": "y",
            "source_refs_json": '["D1"]',
            "review_status": "reviewed",
        },
        {
            "id": 3,
            "component_type": "objective",
            "natural_text": "Minimize total transportation time",
            "formal_text": "Minimize Z = 20x + 40y",
            "symbol": "Z",
            "source_refs_json": '["D1"]',
            "review_status": "reviewed",
        },
        {
            "id": 4,
            "component_type": "constraint",
            "natural_text": "At least 300 ducks are transported",
            "formal_text": "10x + 8y >= 300",
            "symbol": "duck_capacity",
            "source_refs_json": '["D1"]',
            "review_status": "reviewed",
        },
        {
            "id": 5,
            "component_type": "constraint",
            "natural_text": "At most 12 boat trips",
            "formal_text": "x <= 12",
            "symbol": "boat_limit",
            "source_refs_json": '["D1"]',
            "review_status": "reviewed",
        },
        {
            "id": 6,
            "component_type": "constraint",
            "natural_text": "At least 60% of trips should be by canoe",
            "formal_text": "y >= 0.60(x + y)",
            "symbol": "canoe_share",
            "source_refs_json": '["D1"]',
            "review_status": "reviewed",
        },
        {
            "id": 7,
            "component_type": "constraint",
            "natural_text": "Non-negativity and integer requirements for trips",
            "formal_text": "x >= 0, y >= 0, x, y integer",
            "symbol": "domains",
            "source_refs_json": '["D1"]',
            "review_status": "reviewed",
        },
    ]

    generated = build_lp_artifact_from_components(components)
    parsed = parse_lp_artifact(generated["content"])

    assert generated["accepted"] is True
    assert "obj: 20 x + 40 y" in generated["content"]
    assert "canoe_share: -0.6 x + 0.4 y >= 0" in generated["content"]
    assert "General\n x y" in generated["content"]
    assert "x, y integer" not in generated["content"]
    assert parsed.integrality == (1, 1)
    assert parsed.objective == (20.0, 40.0)


def test_strict_lp_validation_rejects_solver_parse_errors():
    invalid = validate_lp_artifact(
        "Minimize\n"
        " obj: Z = 20x + 40y\n"
        "Subject To\n"
        " c1: 10x + 8y >= 300\n"
        " c2: x >= 0, y >= 0, x, y integer\n"
        "End"
    )

    assert invalid["status"] == "invalid"
    assert "invalid_artifact_syntax" in {
        issue["issue_type"] for issue in invalid["issues"]
    }


def test_artifact_component_semantics_gate_repair_candidates():
    topic_id = api.create_topic("Semantic repair", "Reject misaligned repairs.")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="Production",
        source_text="Maximize profit with binary launch decision and demand.",
    )
    component_ids = [
        api.insert_optimization_component(
            problem_id=problem_id,
            topic_id=topic_id,
            component_type="decision_variable",
            natural_text="x indicates whether the launch is selected.",
            formal_text="x in {0,1}",
            symbol="x",
            domain="binary",
            review_status="reviewed",
        ),
        api.insert_optimization_component(
            problem_id=problem_id,
            topic_id=topic_id,
            component_type="objective",
            natural_text="maximize profit.",
            formal_text="maximize 3 x",
            symbol="obj",
            review_status="reviewed",
        ),
        api.insert_optimization_component(
            problem_id=problem_id,
            topic_id=topic_id,
            component_type="constraint",
            natural_text="demand requires at least one selected launch.",
            formal_text="demand: x >= 1",
            symbol="demand",
            review_status="reviewed",
        ),
        api.insert_optimization_component(
            problem_id=problem_id,
            topic_id=topic_id,
            component_type="parameter",
            natural_text="Demand is 1 launch.",
            symbol="d",
            unit="launches",
            review_status="reviewed",
        ),
        api.insert_optimization_component(
            problem_id=problem_id,
            topic_id=topic_id,
            component_type="parameter",
            natural_text="Demand is measured in hours in a bad extraction.",
            symbol="d",
            unit="hours",
            review_status="reviewed",
        ),
    ]
    candidate = (
        "Minimize\n"
        " obj: 3 x\n"
        "Subject To\n"
        " wrong: x <= 1\n"
        "Bounds\n"
        " x >= 0\n"
        "End"
    )

    semantic = validate_artifact_component_semantics(
        problem_id=problem_id,
        content=candidate,
        model_language="lp",
        linked_component_ids=component_ids,
    )

    issue_types = {issue["issue_type"] for issue in semantic["issues"]}
    assert semantic["status"] == "invalid"
    assert "objective_direction_mismatch" in issue_types
    assert "linked_variable_domain_mismatch" in issue_types
    assert "linked_constraint_missing" in issue_types
    assert "linked_parameter_unit_conflict" in issue_types

    source = "x <= 1"
    source_id = persist_lp_artifact(
        topic_id=topic_id,
        problem_id=problem_id,
        content=source,
    )["artifact_id"]
    rejected = apply_lp_repair_candidate(
        topic_id=topic_id,
        problem_id=problem_id,
        source_artifact_id=source_id,
        source_content=source,
        repaired_content=candidate,
        linked_component_ids=component_ids,
    )

    assert rejected["accepted"] is False
    diagnostics = api.get_model_diagnostics(problem_id)
    assert "objective_direction_mismatch" in {
        diagnostic["diagnostic_type"] for diagnostic in diagnostics
    }
    artifact = next(
        item
        for item in api.get_optimization_artifacts(problem_id)
        if item["id"] == rejected["artifact_id"]
    )
    assert artifact["parser_status"] == "valid"
    assert artifact["repair_status"] == "rejected"


@pytest.mark.asyncio
async def test_repair_artifact_semantics_with_mocked_llm():
    topic_id = api.create_topic("Semantic repair", "Repair misaligned LP.")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="Production",
        source_text="Maximize profit with a binary launch decision.",
    )
    component_ids = [
        api.insert_optimization_component(
            problem_id=problem_id,
            topic_id=topic_id,
            component_type="decision_variable",
            natural_text="x indicates whether the launch is selected.",
            formal_text="x in {0,1}",
            symbol="x",
            domain="binary",
            review_status="reviewed",
        ),
        api.insert_optimization_component(
            problem_id=problem_id,
            topic_id=topic_id,
            component_type="objective",
            natural_text="maximize profit.",
            formal_text="maximize 3 x",
            symbol="obj",
            review_status="reviewed",
        ),
        api.insert_optimization_component(
            problem_id=problem_id,
            topic_id=topic_id,
            component_type="constraint",
            natural_text="demand requires at least one selected launch.",
            formal_text="demand: x >= 1",
            symbol="demand",
            review_status="reviewed",
        ),
    ]
    source = (
        "Minimize\n"
        " obj: 3 x\n"
        "Subject To\n"
        " wrong: x <= 1\n"
        "Bounds\n"
        " x >= 0\n"
        "End"
    )
    source_id = persist_lp_artifact(
        topic_id=topic_id,
        problem_id=problem_id,
        content=source,
    )["artifact_id"]
    semantic = validate_artifact_component_semantics(
        problem_id=problem_id,
        content=source,
        model_language="lp",
        linked_component_ids=component_ids,
    )
    prompt = build_semantic_repair_prompt(
        content=source,
        model_language="lp",
        semantic_validation=semantic,
        linked_components=api.get_optimization_components(problem_id),
    )
    assert "objective_direction_mismatch" in prompt
    response = BrokerResponse(
        text=(
            "Maximize\n"
            " obj: 3 x\n"
            "Subject To\n"
            " demand: x >= 1\n"
            "Binary\n"
            " x\n"
            "End"
        ),
        provider_used="mock",
    )

    with patch("orbit_or.broker.llm_call", new=AsyncMock(return_value=response)):
        result = await repair_artifact_semantics_with_llm(
            topic_id=topic_id,
            problem_id=problem_id,
            source_artifact_id=source_id,
            source_content=source,
            model_language="lp",
            linked_component_ids=component_ids,
        )

    assert result["accepted"] is True
    assert result["source_semantic_validation"]["status"] == "invalid"
    assert normalize_lp_repair_candidate(result["raw_text"]).startswith("Maximize")


def test_generate_lp_artifact_from_reviewed_components():
    topic_id = api.create_topic("Component LP", "Build model from components.")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="Production",
        source_text="Maximize profit with capacity.",
    )
    source_refs_json = '["D1"]'
    for payload in [
        {
            "component_type": "decision_variable",
            "natural_text": "x is units produced.",
            "symbol": "x",
            "domain": "nonnegative",
        },
        {
            "component_type": "objective",
            "natural_text": "maximize 3 profit per unit",
            "formal_text": "maximize 3 x",
            "symbol": "obj",
        },
        {
            "component_type": "constraint",
            "natural_text": "capacity at most 4 units",
            "formal_text": "x <= 4",
            "symbol": "capacity",
        },
    ]:
        api.insert_optimization_component(
            problem_id=problem_id,
            topic_id=topic_id,
            review_status="reviewed",
            source_refs_json=source_refs_json,
            **payload,
        )
    components = api.get_optimization_components(problem_id)

    generated = build_lp_artifact_from_components(components)

    assert generated["accepted"] is True
    assert "Maximize" in generated["content"]
    assert "capacity: x <= 4" in generated["content"]
    assert "Bounds\n x >= 0" in generated["content"]

    persisted = persist_lp_artifact_from_components(
        topic_id=topic_id,
        problem_id=problem_id,
        components=components,
    )
    assert persisted["artifact_id"]
    artifact = api.get_optimization_artifacts(problem_id)[0]
    assert artifact["generator_role"] == "component_lp_generator"
    assert json.loads(artifact["linked_component_ids_json"]) == generated["component_ids"]
    assert artifact["component_fingerprints_json"]


def test_generate_lp_artifact_rejects_unreviewed_components():
    topic_id = api.create_topic("Component LP", "Build model from components.")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="Production",
        source_text="Minimize cost with demand.",
    )
    api.insert_optimization_component(
        problem_id=problem_id,
        topic_id=topic_id,
        component_type="decision_variable",
        natural_text="x is units produced.",
        symbol="x",
        review_status="candidate",
        source_refs_json='["D1"]',
    )
    api.insert_optimization_component(
        problem_id=problem_id,
        topic_id=topic_id,
        component_type="objective",
        natural_text="minimize x",
        formal_text="minimize x",
        review_status="reviewed",
        source_refs_json='["D1"]',
    )
    api.insert_optimization_component(
        problem_id=problem_id,
        topic_id=topic_id,
        component_type="constraint",
        natural_text="x at least 1",
        formal_text="x >= 1",
        review_status="reviewed",
        source_refs_json='["D1"]',
    )

    result = persist_lp_artifact_from_components(
        topic_id=topic_id,
        problem_id=problem_id,
        components=api.get_optimization_components(problem_id),
    )

    assert result["artifact_id"] is None
    assert "unreviewed_component" in {issue["issue_type"] for issue in result["issues"]}
    diagnostics = api.get_model_diagnostics(problem_id)
    assert diagnostics[0]["diagnostic_type"] == "unreviewed_component"


def test_build_and_persist_mps_artifact_from_lp():
    topic_id = api.create_topic("MPS topic", "Build MPS.")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="Binary project selection",
        source_text="Maximize benefit with a resource limit.",
    )
    lp = (
        "Maximize\n"
        " obj: 3 x + 2 y\n"
        "Subject To\n"
        " capacity: 2 x + y <= 2\n"
        "Binary\n"
        " x y\n"
        "End"
    )

    generated = build_mps_artifact_from_lp(lp, name="project")

    assert generated["accepted"] is True
    assert "OBJSENSE\n MAX" in generated["content"]
    assert "ROWS" in generated["content"]
    assert "BV BND       x" in generated["content"]
    assert validate_mps_artifact(generated["content"])["status"] == "valid"
    parsed_mps = parse_mps_artifact(generated["content"])
    assert parsed_mps.direction == "maximize"
    assert parsed_mps.integrality == (1, 1)

    component_ids = [
        api.insert_optimization_component(
            problem_id=problem_id,
            topic_id=topic_id,
            component_type="decision_variable",
            natural_text="x is a binary project decision.",
            symbol="x",
            domain="binary",
            review_status="reviewed",
        ),
        api.insert_optimization_component(
            problem_id=problem_id,
            topic_id=topic_id,
            component_type="decision_variable",
            natural_text="y is a binary project decision.",
            symbol="y",
            domain="binary",
            review_status="reviewed",
        ),
        api.insert_optimization_component(
            problem_id=problem_id,
            topic_id=topic_id,
            component_type="objective",
            natural_text="maximize project benefit.",
            formal_text="maximize 3 x + 2 y",
            symbol="obj",
            review_status="reviewed",
        ),
        api.insert_optimization_component(
            problem_id=problem_id,
            topic_id=topic_id,
            component_type="constraint",
            natural_text="capacity limits selected projects.",
            formal_text="capacity: 2 x + y <= 2",
            symbol="capacity",
            review_status="reviewed",
        ),
    ]
    semantic = validate_artifact_component_semantics(
        problem_id=problem_id,
        content=generated["content"],
        model_language="mps",
        linked_component_ids=component_ids,
    )
    assert semantic == {"status": "valid", "issues": []}

    persisted = persist_mps_artifact_from_lp(
        topic_id=topic_id,
        problem_id=problem_id,
        lp_content=lp,
        linked_component_ids=[1, 2],
    )

    assert persisted["artifact_id"]
    artifact = api.get_optimization_artifacts(problem_id)[0]
    assert artifact["model_language"] == "mps"
    assert artifact["parser_status"] == "valid"
    assert json.loads(artifact["linked_component_ids_json"]) == [1, 2]


def test_solver_backend_registry_and_mps_diagnostic_path():
    topic_id = api.create_topic("MPS solve", "Solve MPS backend path.")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="MPS",
        source_text="Maximize x subject to x <= 1.",
    )
    lp = "Maximize\n obj: x\nSubject To\n c1: x <= 1\nBinary\n x\nEnd"
    persisted = persist_mps_artifact_from_lp(
        topic_id=topic_id,
        problem_id=problem_id,
        lp_content=lp,
    )
    artifact = api.get_optimization_artifacts(problem_id)[0]

    backends = available_solver_backends()
    assert backends["scipy_mps"]["model_languages"] == ["mps"]
    assert backends["scipy_mps"]["executes"] is True
    assert backends["mps_validate"]["model_languages"] == ["mps"]
    assert backends["mps_validate"]["executes"] is False

    result = solve_mps_artifact(
        topic_id=topic_id,
        problem_id=problem_id,
        artifact_id=persisted["artifact_id"],
        content=artifact["content"],
    )

    assert result["status"] in {"optimal", "solver_unavailable"}
    if result["status"] == "optimal":
        assert result["objective_value"] == pytest.approx(1.0)
        assert result["variable_values"]["x"] == pytest.approx(1.0)
        assert api.get_model_diagnostics(problem_id) == []
        code_evidence = api.get_code_evidence_by_id(result["code_evidence_id"])
        assert code_evidence["success"] == 1

    validate_only = solve_mps_artifact(
        topic_id=topic_id,
        problem_id=problem_id,
        artifact_id=persisted["artifact_id"],
        content=artifact["content"],
        solver_backend="mps_validate",
    )
    assert validate_only["status"] == "unsupported_model_class"
    diagnostics = api.get_model_diagnostics(problem_id)
    assert diagnostics[-1]["diagnostic_type"] == "unsupported_model_class"
    code_evidence = api.get_code_evidence_by_id(validate_only["code_evidence_id"])
    assert code_evidence["success"] == 0


def test_solve_optimization_artifact_dispatches_by_language():
    topic_id = api.create_topic("LP dispatch", "Solve LP through dispatcher.")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="LP",
        source_text="Minimize x subject to x >= 1.",
    )
    lp = "Minimize\n obj: x\nSubject To\n c1: x >= 1\nBounds\n x >= 0\nEnd"
    artifact_id = persist_lp_artifact(
        topic_id=topic_id,
        problem_id=problem_id,
        content=lp,
    )["artifact_id"]

    result = solve_optimization_artifact(
        topic_id=topic_id,
        problem_id=problem_id,
        artifact_id=artifact_id,
        content=lp,
        model_language="lp",
    )

    assert result["status"] in {"optimal", "solver_unavailable"}


def test_apply_lp_repair_candidate_is_gated_by_validator():
    topic_id = api.create_topic("LP repair", "Repair invalid LP.")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="LP",
        source_text="Minimize x subject to x >= 1.",
    )
    source = "x >= 1"
    source_id = persist_lp_artifact(
        topic_id=topic_id,
        problem_id=problem_id,
        content=source,
    )["artifact_id"]
    repaired = "```lp\nMinimize\n obj: x\nSubject To\n c1: x >= 1\nBounds\n x >= 0\nEnd\n```"

    accepted = apply_lp_repair_candidate(
        topic_id=topic_id,
        problem_id=problem_id,
        source_artifact_id=source_id,
        source_content=source,
        repaired_content=repaired,
    )

    assert accepted["accepted"] is True
    assert accepted["diagnostic_ids"] == []
    artifacts = api.get_optimization_artifacts(problem_id)
    assert artifacts[0]["source_artifact_id"] == source_id
    assert artifacts[0]["repair_status"] == "accepted"

    rejected = apply_lp_repair_candidate(
        topic_id=topic_id,
        problem_id=problem_id,
        source_artifact_id=source_id,
        source_content=source,
        repaired_content="This model is:\nMinimize\n obj: x\nSubject To\n c1: x >= 1\nEnd",
    )
    assert rejected["accepted"] is False
    assert "non_model_preamble" in {
        issue["issue_type"] for issue in rejected["validation"]["issues"]
    }


def test_apply_mps_repair_candidate_is_gated_by_validator():
    topic_id = api.create_topic("MPS repair", "Repair invalid MPS.")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="MPS",
        source_text="Maximize x subject to x <= 1.",
    )
    source = "This model is:\nNAME BAD\nROWS\n N OBJ\nENDATA"
    source_validation = validate_mps_artifact(source)
    source_id = api.insert_optimization_artifact(
        topic_id=topic_id,
        problem_id=problem_id,
        artifact_type="mps_model",
        model_language="mps",
        content=source,
        parser_status=source_validation["status"],
        parser_notes=json.dumps(source_validation),
    )
    repaired_content = build_mps_artifact_from_lp(
        "Maximize\n obj: x\nSubject To\n c1: x <= 1\nBinary\n x\nEnd",
        name="repair",
    )["content"]

    accepted = apply_mps_repair_candidate(
        topic_id=topic_id,
        problem_id=problem_id,
        source_artifact_id=source_id,
        source_content=source,
        repaired_content=f"```mps\n{repaired_content}\n```",
    )

    assert accepted["accepted"] is True
    assert accepted["diagnostic_ids"] == []
    assert normalize_mps_repair_candidate(accepted["content"]).startswith("OBJSENSE")
    artifacts = {
        artifact["id"]: artifact for artifact in api.get_optimization_artifacts(problem_id)
    }
    assert artifacts[accepted["artifact_id"]]["source_artifact_id"] == source_id
    assert artifacts[accepted["artifact_id"]]["repair_status"] == "accepted"

    rejected = apply_mps_repair_candidate(
        topic_id=topic_id,
        problem_id=problem_id,
        source_artifact_id=source_id,
        source_content=source,
        repaired_content="NAME BAD\nROWS\n N OBJ\nENDATA",
    )
    assert rejected["accepted"] is False
    assert "missing_columns_section" in {
        issue["issue_type"] for issue in rejected["validation"]["issues"]
    }


@pytest.mark.asyncio
async def test_repair_lp_artifact_with_mocked_llm():
    topic_id = api.create_topic("LP repair", "Repair invalid LP.")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="LP",
        source_text="Minimize x subject to x >= 1.",
    )
    source = "x >= 1"
    source_id = persist_lp_artifact(
        topic_id=topic_id,
        problem_id=problem_id,
        content=source,
    )["artifact_id"]
    response = BrokerResponse(
        text="Minimize\n obj: x\nSubject To\n c1: x >= 1\nBounds\n x >= 0\nEnd",
        provider_used="mock",
    )

    with patch("orbit_or.broker.llm_call", new=AsyncMock(return_value=response)):
        result = await repair_lp_artifact_with_llm(
            topic_id=topic_id,
            problem_id=problem_id,
            source_artifact_id=source_id,
            source_content=source,
        )

    assert result["accepted"] is True
    assert normalize_lp_repair_candidate(result["raw_text"]).startswith("Minimize")


@pytest.mark.asyncio
async def test_repair_mps_artifact_with_mocked_llm():
    topic_id = api.create_topic("MPS repair", "Repair invalid MPS.")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="MPS",
        source_text="Maximize x subject to x <= 1.",
    )
    source = "NAME BAD\nROWS\n N OBJ\nENDATA"
    source_validation = validate_mps_artifact(source)
    source_id = api.insert_optimization_artifact(
        topic_id=topic_id,
        problem_id=problem_id,
        artifact_type="mps_model",
        model_language="mps",
        content=source,
        parser_status=source_validation["status"],
        parser_notes=json.dumps(source_validation),
    )
    repaired_content = build_mps_artifact_from_lp(
        "Maximize\n obj: x\nSubject To\n c1: x <= 1\nBinary\n x\nEnd",
        name="repair",
    )["content"]
    response = BrokerResponse(text=repaired_content, provider_used="mock")

    with patch("orbit_or.broker.llm_call", new=AsyncMock(return_value=response)):
        result = await repair_mps_artifact_with_llm(
            topic_id=topic_id,
            problem_id=problem_id,
            source_artifact_id=source_id,
            source_content=source,
        )

    assert result["accepted"] is True
    assert normalize_mps_repair_candidate(result["raw_text"]).startswith("OBJSENSE")


def test_solve_lp_artifact_records_solver_run_and_evidence():
    topic_id = api.create_topic("LP topic", "Solve LP.")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="LP",
        source_text="Minimize x subject to x >= 1.",
    )
    lp = "Minimize\n obj: x\nSubject To\n c1: x >= 1\nBounds\n x >= 0\nEnd"
    artifact_id = persist_lp_artifact(
        topic_id=topic_id, problem_id=problem_id, content=lp
    )["artifact_id"]

    result = solve_lp_artifact(
        topic_id=topic_id,
        problem_id=problem_id,
        artifact_id=artifact_id,
        content=lp,
    )

    assert result["solver_run_id"]
    assert result["code_evidence_id"]
    assert result["status"] in {"optimal", "solver_unavailable"}
    if result["status"] == "optimal":
        assert result["objective_value"] == pytest.approx(1.0)
        assert result["variable_values"]["x"] == pytest.approx(1.0)
    else:
        assert result["diagnostic_id"]


def test_solve_binary_lp_with_optional_milp_backend():
    topic_id = api.create_topic("MILP topic", "Choose projects.")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="Binary project selection",
        source_text="Maximize benefit with a resource limit.",
    )
    lp = (
        "Maximize\n"
        " obj: 3 x + 2 y\n"
        "Subject To\n"
        " c1: 2 x + y <= 2\n"
        "Binary\n"
        " x y\n"
        "End"
    )
    parsed = parse_lp_artifact(lp)
    assert parsed.integrality == (1, 1)
    artifact_id = persist_lp_artifact(
        topic_id=topic_id, problem_id=problem_id, content=lp
    )["artifact_id"]

    unsupported = solve_lp_artifact(
        topic_id=topic_id,
        problem_id=problem_id,
        artifact_id=artifact_id,
        content=lp,
        solver_backend="scipy_linprog",
        persist_code_evidence=False,
    )
    assert unsupported["status"] == "unsupported_model_class"

    result = solve_lp_artifact(
        topic_id=topic_id,
        problem_id=problem_id,
        artifact_id=artifact_id,
        content=lp,
        solver_backend="scipy_milp",
        persist_code_evidence=False,
    )

    assert result["status"] in {"optimal", "solver_unavailable"}
    if result["status"] == "optimal":
        assert result["objective_value"] == pytest.approx(3.0)
        assert result["variable_values"]["x"] == pytest.approx(1.0)
        assert result["variable_values"]["y"] == pytest.approx(0.0)


def test_solver_run_can_create_formal_claim_candidate():
    topic_id = api.create_topic("LP topic", "Solve LP.")
    subtopic_id = api.create_subtopic(topic_id, "LP subtopic", "Solve LP.")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="LP",
        source_text="Minimize x subject to x >= 1.",
    )
    lp = "Minimize\n obj: x\nSubject To\n c1: x >= 1\nBounds\n x >= 0\nEnd"
    artifact_id = persist_lp_artifact(
        topic_id=topic_id, problem_id=problem_id, content=lp
    )["artifact_id"]
    solve_lp_artifact(
        topic_id=topic_id,
        problem_id=problem_id,
        artifact_id=artifact_id,
        content=lp,
    )
    problem = api.get_optimization_problem(problem_id)
    artifact = api.get_optimization_artifacts(problem_id)[0]
    solver_run = api.get_solver_runs(problem_id)[0]

    payload = build_solver_claim_payload(
        problem=problem, artifact=artifact, solver_run=solver_run
    )
    assert validate_solver_claim_payload(payload) == []

    result = create_solver_claim_candidate(
        topic_id=topic_id,
        subtopic_id=subtopic_id,
        problem=problem,
        artifact=artifact,
        solver_run=solver_run,
    )

    assert result["candidate_id"]
    candidate = api.get_claim_candidates(topic_id, subtopic_id=subtopic_id)[0]
    assert candidate["claim_type"] == "optimization_result"
    assert "objective value" in candidate["candidate_text"]
    assert "solver:scipy_linprog" in candidate["scope_tags"]
    assert "[E" in candidate["candidate_text"]


def test_review_solver_claim_candidate_accepts_and_deduplicates():
    topic_id = api.create_topic("LP topic", "Solve LP.")
    subtopic_id = api.create_subtopic(topic_id, "LP subtopic", "Solve LP.")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="LP",
        source_text="Minimize x subject to x >= 1.",
    )
    lp = "Minimize\n obj: x\nSubject To\n c1: x >= 1\nBounds\n x >= 0\nEnd"
    artifact_id = persist_lp_artifact(
        topic_id=topic_id, problem_id=problem_id, content=lp
    )["artifact_id"]
    solve_lp_artifact(
        topic_id=topic_id,
        problem_id=problem_id,
        artifact_id=artifact_id,
        content=lp,
    )
    problem = api.get_optimization_problem(problem_id)
    artifact = api.get_optimization_artifacts(problem_id)[0]
    solver_run = api.get_solver_runs(problem_id)[0]

    create_solver_claim_candidate(
        topic_id=topic_id,
        subtopic_id=subtopic_id,
        problem=problem,
        artifact=artifact,
        solver_run=solver_run,
    )
    candidate = api.get_claim_candidates(topic_id, subtopic_id=subtopic_id)[0]
    accepted = review_solver_claim_candidate(topic_id, candidate)

    assert accepted["accepted"] is True
    claim_id = accepted["claim_id"]
    reviewed = api.get_claim_candidates(topic_id, subtopic_id=subtopic_id)[0]
    assert reviewed["status"] == "accepted"
    assert reviewed["accepted_claim_id"] == claim_id
    claim = api.get_claims(topic_id)[0]
    assert claim["claim_type"] == "optimization_result"
    edges = api.get_knowledge_edges(
        topic_id,
        source_type="code_evidence",
        target_type="claim",
        relation="supports",
    )
    assert edges[0]["target_id"] == claim_id

    create_solver_claim_candidate(
        topic_id=topic_id,
        subtopic_id=subtopic_id,
        problem=problem,
        artifact=artifact,
        solver_run=solver_run,
    )
    results = review_pending_solver_claim_candidates(topic_id)

    assert results[0]["accepted"] is False
    assert results[0]["reason"] == "duplicate"
    assert len(api.get_claims(topic_id)) == 1


def test_component_status_propagates_to_solver_claim_jtms():
    topic_id = api.create_topic("Component LP", "Build model from components.")
    subtopic_id = api.create_subtopic(topic_id, "LP subtopic", "Solve LP.")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="Production",
        source_text="Minimize x subject to x >= 1.",
    )
    source_refs_json = '["D1"]'
    for payload in [
        {
            "component_type": "decision_variable",
            "natural_text": "x is production quantity.",
            "symbol": "x",
            "domain": "nonnegative",
        },
        {
            "component_type": "objective",
            "natural_text": "minimize production quantity",
            "formal_text": "minimize x",
            "symbol": "obj",
        },
        {
            "component_type": "constraint",
            "natural_text": "x must cover demand of one unit",
            "formal_text": "x >= 1",
            "symbol": "demand",
        },
    ]:
        api.insert_optimization_component(
            problem_id=problem_id,
            topic_id=topic_id,
            review_status="reviewed",
            source_refs_json=source_refs_json,
            **payload,
        )
    components = api.get_optimization_components(problem_id)
    artifact_result = persist_lp_artifact_from_components(
        topic_id=topic_id,
        problem_id=problem_id,
        components=components,
    )
    artifact_id = artifact_result["artifact_id"]
    artifact = api.get_optimization_artifacts(problem_id)[0]
    solve_lp_artifact(
        topic_id=topic_id,
        problem_id=problem_id,
        artifact_id=artifact_id,
        content=artifact["content"],
    )
    solver_run = api.get_solver_runs(problem_id)[0]
    create_solver_claim_candidate(
        topic_id=topic_id,
        subtopic_id=subtopic_id,
        problem=api.get_optimization_problem(problem_id),
        artifact=artifact,
        solver_run=solver_run,
    )
    candidate = api.get_claim_candidates(topic_id, subtopic_id=subtopic_id)[0]
    accepted = review_solver_claim_candidate(topic_id, candidate)
    claim_id = accepted["claim_id"]

    decision_variable = next(
        component
        for component in components
        if component["component_type"] == "decision_variable"
    )
    assert api.update_optimization_component_review(
        decision_variable["id"],
        review_status="rejected",
        validation_notes="Variable scope is unsupported.",
    )
    propagation = propagate_component_status_to_solver_evidence(
        topic_id, problem_id=problem_id
    )
    assert artifact_id in propagation["affected_artifact_ids"]
    diagnostics = api.get_model_diagnostics(problem_id, status="open")
    assert {
        diagnostic["diagnostic_type"] for diagnostic in diagnostics
    } >= {"linked_component_inactive"}

    changes = jtms_sweep(topic_id, current_round=3)
    assert any(
        change["type"] == "claim"
        and change["id"] == claim_id
        and change["new_status"] == "contested"
        for change in changes
    )


def test_solver_claim_validation_requires_scope_and_evidence():
    issues = validate_solver_claim_payload(
        {
            "conclusion": "Solver result exists",
            "evidence_strength": 5,
            "falsification_criteria": "different by more than 1e-6",
        }
    )
    issue_types = {issue.issue_type for issue in issues}
    assert "missing_scope" in issue_types
    assert "missing_solver_evidence" in issue_types


def test_component_extraction_prompt_and_persistence():
    topic_id = api.create_topic("Production planning", "Plan production.")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="Production",
        source_text="Minimize cost while meeting demand.",
    )

    prompt = build_component_extraction_prompt("Minimize cost.")
    assert '"components"' in prompt
    assert "decision_variable" in prompt

    result = persist_component_payloads(
        topic_id=topic_id,
        problem_id=problem_id,
        payloads=[
            {
                "component_type": "parameter",
                "natural_text": "Demand is 10 units.",
                "symbol": "d",
                "source_refs": ["D1"],
            }
        ],
    )

    assert len(result["component_ids"]) == 1
    assert len(result["diagnostic_ids"]) == 1
    diagnostic = api.get_model_diagnostics(problem_id)[0]
    assert diagnostic["diagnostic_type"] == "missing_unit"
    assert api.update_model_diagnostic_status(
        diagnostic["id"], status="resolved", resolution="unit reviewed"
    )
    assert api.get_model_diagnostics(problem_id)[0]["status"] == "resolved"


def test_component_persistence_normalizes_list_scalar_fields():
    topic_id = api.create_topic("Component coercion", "Persist noisy payloads.")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="Payload",
        source_text="Maximize output.",
    )

    result = persist_component_payloads(
        topic_id=topic_id,
        problem_id=problem_id,
        payloads=[
            {
                "component_type": "decision_variable",
                "natural_text": "Number of process J runs.",
                "symbol": ["x_J"],
                "unit": ["runs"],
                "domain": ["integer", "nonnegative"],
                "source_refs": ["D1"],
            }
        ],
    )

    component = api.get_optimization_components(problem_id)[0]

    assert len(result["component_ids"]) == 1
    assert component["symbol"] == "x_J"
    assert component["unit"] == "runs"
    assert component["domain"] == "integer, nonnegative"


@pytest.mark.asyncio
async def test_extract_and_persist_components_with_mocked_llm():
    topic_id = api.create_topic("Production planning", "Plan production.")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="Production",
        source_text="Minimize cost while meeting demand.",
    )
    response = BrokerResponse(
        text=(
            '{"components":[{"component_type":"objective",'
            '"natural_text":"Minimize cost","formal_text":"min c^T x",'
            '"source_refs":["D1"]}]}'
        ),
        provider_used="mock",
    )

    with patch("orbit_or.broker.llm_call", new=AsyncMock(return_value=response)):
        result = await extract_and_persist_components(
            topic_id=topic_id,
            problem_id=problem_id,
            source_text="Minimize cost while meeting demand.",
        )

    assert len(result["component_ids"]) == 1
    component = api.get_optimization_components(problem_id)[0]
    assert component["component_type"] == "objective"
    assert component["formal_text"] == "min c^T x"


@pytest.mark.asyncio
async def test_extract_component_candidate_tournament_persists_best_candidate():
    topic_id = api.create_topic("Tournament", "Minimize x subject to x >= 1.")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="Tournament LP",
        source_text="Minimize x subject to x >= 1.",
    )
    invalid = BrokerResponse(
        text=(
            '{"components":[{"component_type":"objective",'
            '"natural_text":"Minimize x","formal_text":"obj: x",'
            '"symbol":"obj","source_refs":["D1"]}]}'
        ),
        provider_used="mock",
    )
    valid = BrokerResponse(
        text=(
            '{"components":['
            '{"component_type":"decision_variable","natural_text":"x is nonnegative",'
            '"formal_text":"x >= 0","symbol":"x","source_refs":["D1"]},'
            '{"component_type":"objective","natural_text":"Minimize x",'
            '"formal_text":"obj: x","symbol":"obj","source_refs":["D1"]},'
            '{"component_type":"constraint","natural_text":"x must be at least 1",'
            '"formal_text":"c1: x >= 1","symbol":"c1","source_refs":["D1"]}'
            ']}'
        ),
        provider_used="mock",
    )

    with patch("orbit_or.broker.llm_call", new=AsyncMock(side_effect=[invalid, valid])):
        result = await extract_component_candidate_tournament(
            topic_id=topic_id,
            problem_id=problem_id,
            source_text="Minimize x subject to x >= 1.",
            candidate_count=2,
        )

    assert result["best"]["candidate_index"] == 1
    assert result["best"]["modeling_error"] == "none"
    assert len(result["component_ids"]) == 3
    assert {
        component["component_type"]
        for component in api.get_optimization_components(problem_id)
    } == {"decision_variable", "objective", "constraint"}


<<<<<<< Updated upstream
=======
@pytest.mark.asyncio
async def test_extract_component_candidate_tournament_single_call_persists_best_candidate():
    topic_id = api.create_topic("Tournament single call", "Minimize x subject to x >= 1.")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="Tournament LP",
        source_text="Minimize x subject to x >= 1.",
    )
    response = BrokerResponse(
        text=json.dumps(
            {
                "candidates": [
                    {
                        "components": [
                            {
                                "component_type": "objective",
                                "natural_text": "Minimize x",
                                "formal_text": "obj: x",
                                "symbol": "obj",
                                "source_refs": ["D1"],
                            }
                        ]
                    },
                    {
                        "components": [
                            {
                                "component_type": "decision_variable",
                                "natural_text": "x is nonnegative",
                                "formal_text": "x >= 0",
                                "symbol": "x",
                                "source_refs": ["D1"],
                            },
                            {
                                "component_type": "objective",
                                "natural_text": "Minimize x",
                                "formal_text": "obj: x",
                                "symbol": "obj",
                                "source_refs": ["D1"],
                            },
                            {
                                "component_type": "constraint",
                                "natural_text": "x must be at least 1",
                                "formal_text": "c1: x >= 1",
                                "symbol": "c1",
                                "source_refs": ["D1"],
                            },
                        ]
                    },
                ]
            }
        ),
        provider_used="mock",
    )

    mock_call = AsyncMock(return_value=response)
    with patch("orbit_or.broker.llm_call", new=mock_call):
        result = await extract_component_candidate_tournament(
            topic_id=topic_id,
            problem_id=problem_id,
            source_text="Minimize x subject to x >= 1.",
            model="MiniMax-M2.7",
            candidate_count=2,
            single_call=True,
        )

    assert mock_call.await_count == 1
    assert mock_call.await_args.kwargs["model"] == "MiniMax-M2.7"
    assert result["best"]["candidate_index"] == 1
    assert result["best"]["modeling_error"] == "none"
    assert len(result["ranked"]) == 2
    assert {
        component["component_type"]
        for component in api.get_optimization_components(problem_id)
    } == {"decision_variable", "objective", "constraint"}


>>>>>>> Stashed changes
def test_classify_solver_failure():
    assert (
        classify_solver_failure("failed", stderr="model is infeasible")
        == "solver_infeasible"
    )
    assert classify_solver_failure("success", stdout="Optimal objective 12") == "none"
    assert classify_solver_failure("failed", stderr="missing data for demand") == "missing_data"
    assert (
        classify_solver_failure("failed", stderr="ambiguous variable x")
        == "ambiguous_variable"
    )
    assert classify_solver_failure("failed", stderr="unit mismatch: kg vs tons") == "unit_mismatch"
    assert (
        classify_solver_failure("failed", stderr="optimal value differs from gold")
        == "wrong_optimal_value"
    )
    assert (
        classify_solver_failure("failed", stderr="no-solution calibration error")
        == "no_solution_calibration_error"
    )
