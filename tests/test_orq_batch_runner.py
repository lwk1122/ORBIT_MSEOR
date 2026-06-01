import importlib.util
from pathlib import Path

from orbit_or.optimization import build_tsp_artifact_from_source


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_orq_minimax_batch.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("orq_batch_runner", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_orq_batch_runner_selects_dataset_and_split(tmp_path):
    runner = _load_runner()
    root = tmp_path / "ORQ_Dataset"
    (root / "MAMO").mkdir(parents=True)
    (root / "IndustryOR").mkdir(parents=True)
    (root / "NL4OPT").mkdir(parents=True)
    (root / "MAMO" / "MAMO_EasyLP.json").write_text(
        '{"en_question": "Easy one.", "en_answer": "1"}\n',
        encoding="utf-8",
    )
    (root / "MAMO" / "MAMO_ComplexLP.json").write_text(
        '{"en_question": "Complex one.", "en_answer": "2"}\n',
        encoding="utf-8",
    )
    (root / "IndustryOR" / "IndustryOR.json").write_text(
        '[{"id": 7, "en_question": "Industry one.", "en_answer": "3"}]',
        encoding="utf-8",
    )
    (root / "NL4OPT" / "NL4OPT_with_optimal_solution.json").write_text(
        '{"en_question": "NL4OPT one.", "en_answer": "4"}\n',
        encoding="utf-8",
    )

    selected = runner._select_cases(
        root=root,
        dataset="MAMO",
        split="complex_lp",
        start=1,
        limit=None,
        case_ids=[],
    )

    assert [case["id"] for _, case in selected] == ["MAMO:complex_lp:1"]


def test_orq_batch_runner_selects_explicit_case_id(tmp_path):
    runner = _load_runner()
    root = tmp_path / "ORQ_Dataset"
    (root / "MAMO").mkdir(parents=True)
    (root / "MAMO" / "MAMO_EasyLP.json").write_text(
        '{"en_question": "Easy one.", "en_answer": "1"}\n',
        encoding="utf-8",
    )

    selected = runner._select_cases(
        root=root,
        dataset="MAMO",
        split="easy_lp",
        start=1,
        limit=None,
        case_ids=["MAMO:easy_lp:1"],
    )

    assert len(selected) == 1
    assert selected[0][1]["problem_text"] == "Easy one."


def test_orq_batch_runner_selects_case_ids_across_datasets(tmp_path):
    runner = _load_runner()
    root = tmp_path / "ORQ_Dataset"
    (root / "MAMO").mkdir(parents=True)
    (root / "IndustryOR").mkdir(parents=True)
    (root / "MAMO" / "MAMO_EasyLP.json").write_text(
        '{"en_question": "Easy one.", "en_answer": "1"}\n',
        encoding="utf-8",
    )
    (root / "IndustryOR" / "IndustryOR.json").write_text(
        '[{"id": 7, "en_question": "Industry one.", "en_answer": "3"}]',
        encoding="utf-8",
    )

    selected = runner._select_cases(
        root=root,
        dataset="MAMO",
        split="easy_lp",
        start=1,
        limit=None,
        case_ids=["MAMO:easy_lp:1", "IndustryOR:test:7"],
    )

    assert [case["id"] for _, case in selected] == [
        "IndustryOR:test:7",
        "MAMO:easy_lp:1",
    ]


def test_orq_batch_runner_loads_case_set_ids(tmp_path):
    runner = _load_runner()
    case_set = tmp_path / "cases.json"
    case_set.write_text(
        '{"cases": [{"case_id": "A:one:1"}, {"id": "B:two:2"}, "C:three:3"]}',
        encoding="utf-8",
    )

    assert runner._load_case_ids(case_set) == ["A:one:1", "B:two:2", "C:three:3"]


def test_nl4opt_compatibility_entrypoint_imports_orq_main():
    wrapper_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_nl4opt_minimax_batch.py"
    )
    spec = importlib.util.spec_from_file_location("nl4opt_batch_wrapper", wrapper_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    assert callable(module.main)
    assert module.main.__module__ == "run_orq_minimax_batch"


def test_orq_batch_runner_parser_accepts_case_timeout():
    runner = _load_runner()

    args = runner.build_parser().parse_args(
        [
            "--case-timeout-s",
            "120",
            "--case-set",
            "cases.json",
            "--model",
            "MiniMax-M2.7",
            "--modeling-brief-mode",
            "always",
            "--deterministic-only",
            "--single-call-candidates",
            "--direct-fallback-predictions",
            "predictions.json",
            "--disable-reviewed-promotion",
            "--disable-solver-validation",
            "--disable-deterministic-adapters",
            "--disable-direct-fallback",
            "--disable-repair-loop",
        ]
    )

    assert args.case_timeout_s == 120
    assert args.case_set == "cases.json"
    assert args.model == "MiniMax-M2.7"
    assert args.modeling_brief_mode == "always"
    assert args.deterministic_only is True
    assert args.single_call_candidates is True
    assert args.direct_fallback_predictions == "predictions.json"
    assert args.disable_reviewed_promotion is True
    assert args.disable_solver_validation is True
    assert args.disable_deterministic_adapters is True
    assert args.disable_direct_fallback is True
    assert args.disable_repair_loop is True


def test_orq_batch_runner_unknown_evaluation_for_deterministic_only():
    runner = _load_runner()

    evaluation = runner._unknown_evaluation(
        {"gold_solver": {"status": "optimal", "objective_value": 42.0}},
        reason="deterministic_uncovered",
    )

    assert evaluation["prediction"]["status"] == "unknown"
    assert evaluation["prediction"]["reason"] == "deterministic_uncovered"
    assert evaluation["metrics"]["correct"] is None


def test_orq_batch_runner_applies_recorded_direct_fallback():
    runner = _load_runner()
    case = {
        "id": "MAMO:easy_lp:1",
        "gold_solver": {"status": "optimal", "objective_value": 5.0},
    }
    archive = {
        "case_id": "MAMO:easy_lp:1",
        "question": "Minimize cost.",
        "evaluation": runner._unknown_evaluation(
            case,
            reason="deterministic_uncovered",
        ),
        "workflow_snapshots": [],
        "diagnostics": [],
    }

    applied = runner._apply_direct_fallback(
        archive,
        case,
        {"MAMO:easy_lp:1": {"status": "optimal", "objective_value": 5.0}},
    )

    assert applied is True
    assert archive["evaluation"]["metrics"]["correct"] is True
    assert archive["evaluation"]["prediction"]["direct_fallback_used"] is True
    assert archive["direct_fallback"]["used"] is True


def test_orq_batch_runner_direct_fallback_requires_uncovered_reason():
    runner = _load_runner()
    case = {
        "id": "MAMO:easy_lp:1",
        "gold_solver": {"status": "optimal", "objective_value": 5.0},
    }
    archive = {
        "case_id": "MAMO:easy_lp:1",
        "evaluation": runner._unknown_evaluation(
            case,
            reason="deterministic_source_blocked",
        ),
    }

    applied = runner._apply_direct_fallback(
        archive,
        case,
        {"MAMO:easy_lp:1": {"status": "optimal", "objective_value": 5.0}},
    )

    assert applied is False
    assert archive["evaluation"]["prediction"]["status"] == "unknown"
    assert archive["direct_fallback"]["reason"] == "primary_reason_not_eligible"


def test_orq_batch_runner_clear_direct_fallback_restores_unknown():
    runner = _load_runner()
    case = {
        "id": "MAMO:easy_lp:1",
        "gold_solver": {"status": "optimal", "objective_value": 5.0},
    }
    archive = {
        "case_id": "MAMO:easy_lp:1",
        "evaluation": {
            "prediction": {
                "status": "optimal",
                "objective_value": 5.0,
                "direct_fallback_used": True,
            },
            "gold": case["gold_solver"],
            "metrics": {"correct": True},
        },
        "direct_fallback": {"used": True, "reason": "used"},
        "solver_runs": [],
    }

    cleared = runner._clear_direct_fallback(archive, case)

    assert cleared is True
    assert archive["evaluation"]["prediction"]["status"] == "unknown"
    assert archive["evaluation"]["prediction"]["reason"] == "deterministic_uncovered"
    assert archive["direct_fallback"]["reason"] == "direct_fallback_disabled"


def test_orq_batch_runner_workflow_evaluation_can_disable_solver_validation():
    runner = _load_runner()
    case = {
        "gold_solver": {"status": "optimal", "objective_value": 5.0},
    }

    evaluation = runner._workflow_evaluation(
        case,
        latest_solver_run={"status": "optimal", "objective_value": 5.0},
        deterministic_uncovered=False,
        blocking_issues=[],
        solver_validation_disabled=True,
    )

    assert evaluation["prediction"]["status"] == "unknown"
    assert evaluation["prediction"]["reason"] == "solver_validation_disabled"
    assert evaluation["metrics"]["correct"] is None


def test_orq_batch_runner_promotes_clean_reviewed_tournament(monkeypatch):
    runner = _load_runner()
    calls = []

    monkeypatch.setattr(
        runner.api,
        "update_optimization_component_review",
        lambda component_id, **kwargs: calls.append((component_id, kwargs)) or True,
    )

    result = runner._promote_reviewed_tournament_components(
        workflow_mode="modeling_reviewed",
        tournament={
            "component_ids": [1, 2, 3],
            "best": {
                "modeling_error": "none",
                "validation": {"status": "valid"},
                "diagnostics": [],
            },
        },
    )

    assert result == {
        "promoted_component_ids": [1, 2, 3],
        "reason": "clean_reviewed_candidate",
    }
    assert [call[0] for call in calls] == [1, 2, 3]
    assert {call[1]["review_status"] for call in calls} == {"executable"}


def test_orq_batch_runner_can_disable_reviewed_promotion(monkeypatch):
    runner = _load_runner()
    calls = []

    monkeypatch.setattr(
        runner.api,
        "update_optimization_component_review",
        lambda component_id, **kwargs: calls.append((component_id, kwargs)) or True,
    )

    result = runner._promote_reviewed_tournament_components(
        workflow_mode="modeling_reviewed",
        disabled=True,
        tournament={
            "component_ids": [1],
            "best": {
                "modeling_error": "none",
                "validation": {"status": "valid"},
                "diagnostics": [],
            },
        },
    )

    assert result == {
        "promoted_component_ids": [],
        "reason": "reviewed_promotion_disabled",
    }
    assert calls == []


def test_orq_batch_runner_does_not_promote_diagnostic_reviewed_tournament(monkeypatch):
    runner = _load_runner()
    calls = []

    monkeypatch.setattr(
        runner.api,
        "update_optimization_component_review",
        lambda component_id, **kwargs: calls.append((component_id, kwargs)) or True,
    )

    result = runner._promote_reviewed_tournament_components(
        workflow_mode="modeling_reviewed",
        tournament={
            "component_ids": [1, 2, 3],
            "best": {
                "modeling_error": "missing_constraint",
                "validation": {"status": "valid"},
                "diagnostics": [
                    {
                        "issue_type": "missing_constraint",
                        "severity": "warning",
                    }
                ],
            },
        },
    )

    assert result["promoted_component_ids"] == []
    assert result["reason"] == "best_candidate_has_modeling_error"
    assert calls == []


def test_orq_batch_runner_treats_timeout_as_retryable_status():
    runner = _load_runner()

    assert "timeout" in runner.RETRYABLE_ARCHIVE_STATUSES
    assert {"error", "failed"}.issubset(runner.RETRYABLE_ARCHIVE_STATUSES)


def test_orq_batch_runner_does_not_use_deterministic_tsp_for_incomplete_matrix():
    runner = _load_runner()
    source = (
        "The shipment must visit each city exactly once before returning to the "
        "city of origin while minimizing total cost. The cities are 1 to 6. "
        "The cost to move the shipment from City 1 to City 4 is 38 units. "
        "From City 2, it costs 13 units to deliver to City 4 but 93 units to "
        "deliver to City 3."
    )

    built = build_tsp_artifact_from_source(source)
    payloads = runner._deterministic_component_payloads("case", source)

    assert built["accepted"] is False
    assert payloads == []


def test_orq_batch_runner_uses_simple_allocation_adapter():
    runner = _load_runner()
    source = (
        "The total budget for both channels combined is constrained to a maximum "
        "of $5000. The combined effort must yield a minimum effectiveness score, "
        "calculated as 3 times the budget for channel X plus 4 times the budget "
        "for channel Y, of at least 12000 points. At the same time, 5 times the "
        "budget allocated to channel X minus twice that allocated to channel Y "
        "should not exceed 10000 points. Given that cost per unit of effectiveness "
        "for channel $X$ is $200 and for channel $Y$ is $150, budgets are integers."
    )

    payloads = runner._deterministic_component_payloads("case", source)

    assert any(payload["component_type"] == "objective" for payload in payloads)
    assert any(payload.get("formal_text") == "3 x + 4 y >= 12000" for payload in payloads)


def test_orq_batch_runner_builds_industry_modeling_brief():
    runner = _load_runner()
    text = (
        "Workers train trainees over 8 weeks and overtime is available.\n\n"
        "| Week | 1 | 2 |\n"
        "|------|---|---|\n"
        "| I | 10 | 20 |"
    )

    modeling_text, brief = runner._modeling_source_text(
        {"dataset": "IndustryOR"},
        text,
        modeling_brief_mode="auto",
    )

    assert "ORBIT modeling brief" in brief
    assert "time indices" in brief
    assert "markdown tables" in brief
    assert modeling_text.endswith(text)


def test_orq_batch_runner_builds_easy_lp_translation_brief():
    runner = _load_runner()
    text = (
        "The total budget for both channels combined is constrained to a maximum of $2000. "
        "The effort from channel X should be three times greater than twice the effort "
        "from channel Y, yielding a difference of at least 500 points. "
        "The cost per unit for X is $50 and for Y is $100."
    )

    modeling_text, brief = runner._modeling_source_text(
        {"dataset": "MAMO"},
        text,
        modeling_brief_mode="auto",
    )

    assert "combined decision variables directly" in brief
    assert "3 X - 2 Y >= D" in brief
    assert "direct per-unit costs" in brief
    assert modeling_text.endswith(text)


def test_orq_batch_runner_summarizes_problem_spec_for_manifest():
    runner = _load_runner()
    archive = {
        "archive_status": "complete",
        "case_index": 1,
        "case_id": "MAMO:easy_lp:1",
        "dataset": "MAMO",
        "split": "easy_lp",
        "workflow_snapshots": [
            {
                "problem_spec": {
                    "schema": "orbit_problem_spec.v1",
                    "classification": {
                        "model_class": "scheduling",
                        "confidence": "medium",
                    },
                    "solver_plan": {
                        "route": "lp_or_mps",
                        "preferred_artifact": "lp",
                        "candidate_backends": ["scipy_linprog"],
                        "requires_review": True,
                    },
                    "validation": {
                        "status": "invalid",
                        "issues": [
                            {
                                "issue_type": "unbound_formal_symbol",
                                "severity": "error",
                                "message": "d1 is not bound.",
                            }
                        ],
                    },
                }
            }
        ],
        "diagnostics": [
            {
                "diagnostic_type": "solver_route_blocked",
                "severity": "warning",
            }
        ],
        "evaluation": {
            "prediction": {"status": "no_solver_run", "objective_value": None},
            "gold": {"status": "optimal", "objective_value": 1.0},
            "metrics": {"correct": False},
        },
    }

    archive["problem_spec_summary"] = runner._problem_spec_summary(archive)
    archive["failure_signature"] = runner._failure_signature(archive)
    summary = runner._manifest_summary(archive)

    assert archive["problem_spec_summary"]["model_class"] == "scheduling"
    assert archive["problem_spec_summary"]["solver_route"] == "lp_or_mps"
    assert archive["problem_spec_summary"]["validation_issue_types"] == [
        "unbound_formal_symbol"
    ]
    assert summary["model_class"] == "scheduling"
    assert summary["solver_route"] == "lp_or_mps"
    assert summary["dataset"] == "MAMO"
    assert summary["split"] == "easy_lp"
    assert summary["diagnostic_types"] == ["solver_route_blocked"]
    assert summary["failure_signature"] == "problem_spec:unbound_formal_symbol"


def test_orq_batch_runner_adjudicates_tsp_assignment_relaxation_gold():
    runner = _load_runner()
    source = (
        "In a scenario involving a traveling salesperson, there are six cities "
        "labeled 1 through 6. The salesperson needs to visit each city exactly "
        "once, starting and ending at the same city, with the objective to "
        "minimize the total travel cost. The costs are provided in a cost "
        "matrix. The cost from city 1 to city 2 is 86, to city 3 is 81, to city "
        "4 is 64, to city 5 is 65, and to city 6 is 24. From city 2, the travel "
        "costs are 86 to city 1, 44 to city 3, 80 to city 4, 91 to city 5, and "
        "23 to city 6. Traveling from city 3, the costs are 81 to city 1, 44 to "
        "city 2, 15 to city 4, 25 to city 5, and 89 to city 6. From city 4, it "
        "costs 64 to travel to city 1, 80 to city 2, 15 to city 3, 89 to city "
        "5, and 41 to city 6. From city 5, the travel costs are 65 to city 1, "
        "91 to city 2, 25 to city 3, 89 to city 4, and 29 to city 6. Lastly, "
        "from city 6, the travel costs are 24 to city 1, 23 to city 2, 89 to "
        "city 3, 41 to city 4, and 29 to city 5."
    )
    built = build_tsp_artifact_from_source(source)
    archive = {
        "archive_status": "complete",
        "case_id": "MAMO:complex_lp:66",
        "question": source,
        "artifacts": [
            {
                "model_language": "tsp_json",
                "content": built["content"],
                "parser_status": "valid",
            }
        ],
        "solver_runs": [
            {
                "id": 1,
                "status": "optimal",
                "solver_backend": "exact_tsp_enumeration",
                "objective_value": 232.0,
            }
        ],
        "evaluation": {
            "prediction": {"status": "optimal", "objective_value": 232.0},
            "gold": {
                "status": "optimal",
                "objective_value": 206.0,
                "abs_tol": 1e-4,
                "rel_tol": 0.05,
            },
            "metrics": {"correct": False},
        },
        "workflow_snapshots": [],
        "diagnostics": [],
    }

    adjudication = runner._gold_adjudication(archive)
    archive["evaluation"]["gold_adjudication"] = adjudication
    archive["problem_spec_summary"] = runner._problem_spec_summary(archive)
    archive["failure_signature"] = runner._failure_signature(archive)
    summary = runner._manifest_summary(archive)

    assert adjudication["status"] == "gold_matches_assignment_relaxation"
    assert adjudication["adjudicated_correct"] is True
    assert len(adjudication["assignment_cycles"]) == 3
    assert archive["failure_signature"] == (
        "gold_adjudication:gold_matches_assignment_relaxation"
    )
    assert summary["correct"] is False
    assert summary["adjudicated_correct"] is True


def test_orq_batch_runner_adjudicates_incomplete_tsp_cost_matrix():
    runner = _load_runner()
    source = (
        "The shipment must visit each city exactly once before returning to the "
        "city of origin while minimizing total cost. The cities are 1 to 6. "
        "The cost to move the shipment from City 1 to City 4 is 38 units. "
        "From City 2, it costs 13 units to deliver to City 4 but 93 units to "
        "deliver to City 3."
    )
    archive = {
        "archive_status": "complete",
        "case_id": "MAMO:complex_lp:70",
        "question": source,
        "artifacts": [],
        "solver_runs": [],
        "evaluation": {
            "prediction": {"status": "no_solver_run", "objective_value": None},
            "gold": {
                "status": "optimal",
                "objective_value": 162.0,
                "abs_tol": 1e-4,
                "rel_tol": 0.05,
            },
            "metrics": {"correct": False},
        },
        "workflow_snapshots": [],
        "diagnostics": [
            {
                "diagnostic_type": "missing_tsp_arcs",
                "severity": "error",
            }
        ],
    }

    adjudication = runner._gold_adjudication(archive)

    assert adjudication["status"] == "gold_requires_incomplete_tsp_cost_matrix"
    assert adjudication["adjudicated_correct"] is True


def test_orq_batch_runner_adjudicates_unknown_incomplete_tsp_cost_matrix():
    runner = _load_runner()
    source = (
        "A traveling salesman must visit 7 customers at 7 different locations, "
        "with a symmetric distance matrix, but the table omits some distances."
    )
    archive = {
        "archive_status": "complete",
        "case_id": "IndustryOR:test:60",
        "question": source,
        "artifacts": [],
        "solver_runs": [],
        "evaluation": {
            "prediction": {
                "status": "unknown",
                "objective_value": None,
                "reason": "deterministic_source_blocked",
            },
            "gold": {
                "status": "optimal",
                "objective_value": 153.0,
                "abs_tol": 1e-4,
                "rel_tol": 0.05,
            },
            "metrics": {"correct": None},
        },
        "workflow_snapshots": [],
        "diagnostics": [
            {
                "diagnostic_type": "insufficient_tsp_costs",
                "severity": "error",
            }
        ],
    }

    adjudication = runner._gold_adjudication(archive)

    assert adjudication["status"] == "gold_requires_incomplete_tsp_cost_matrix"
    assert adjudication["adjudicated_correct"] is True


def test_orq_batch_runner_adjudicates_summarized_max_flow_capacity_rows():
    runner = _load_runner()
    source = (
        "Find the maximum flow from Data Center 0 to Data Center 2 under capacity "
        "limits. From Data Center 0 (Source): Can transmit information to Center "
        "1 (5 TB), and Center 2 (3 TB). From Data Center 1: Can transmit "
        "information to Centers ranging from 1 to 2, with capacities varying "
        "between 1 TB to 20 TB. From Data Center 2 (Destination): Can receive "
        "information."
    )
    archive = {
        "archive_status": "complete",
        "case_id": "MAMO:complex_lp:117",
        "question": source,
        "artifacts": [],
        "solver_runs": [],
        "evaluation": {
            "prediction": {"status": "no_solver_run", "objective_value": None},
            "gold": {
                "status": "optimal",
                "objective_value": 81.0,
                "abs_tol": 1e-4,
                "rel_tol": 0.05,
            },
            "metrics": {"correct": False},
        },
        "workflow_snapshots": [],
        "diagnostics": [],
    }

    adjudication = runner._gold_adjudication(archive)

    assert (
        adjudication["status"]
        == "gold_requires_summarized_max_flow_capacity_rows"
    )
    assert adjudication["adjudicated_correct"] is True


def test_orq_batch_runner_adjudicates_summarized_facility_location_data():
    runner = _load_runner()
    source = (
        "A company must choose distribution centers to supply retail stores while "
        "minimizing opening and transportation costs. Number of Potential "
        "Distribution Centers: 2. Number of Retail Stores Needing Supply: 3. "
        "Opening Costs for Each Distribution Center: Center 1: $100, Center 2: $80. "
        "Transportation Cost Per Unit from Each Distribution Center to Retail Stores: "
        "From Center 1: $1 to Store 1, and so on, up to $5 to Store 3. "
        "From Center 2: $4 to Store 1, etc. Demand of Each Retail Store: "
        "Store 1: 10, Store 2: 10, Store 3: 10. Supply Capacity of Each "
        "Distribution Center: Center 1: 20, Center 2: 20."
    )
    archive = {
        "archive_status": "complete",
        "case_id": "MAMO:complex_lp:150",
        "question": source,
        "artifacts": [],
        "solver_runs": [],
        "evaluation": {
            "prediction": {"status": "no_solver_run", "objective_value": None},
            "gold": {
                "status": "optimal",
                "objective_value": 607479.0,
                "abs_tol": 1e-4,
                "rel_tol": 0.05,
            },
            "metrics": {"correct": False},
        },
        "workflow_snapshots": [],
        "diagnostics": [
            {
                "diagnostic_type": "insufficient_facility_location_data",
                "severity": "error",
            }
        ],
    }

    adjudication = runner._gold_adjudication(archive)

    assert (
        adjudication["status"]
        == "gold_requires_summarized_facility_location_data"
    )
    assert adjudication["adjudicated_correct"] is True


def test_orq_batch_runner_prefers_facility_adjudication_when_text_mentions_network():
    runner = _load_runner()
    source = (
        "A company must supply nine retail stores with products from a network "
        "of seven potential distribution centers. Opening Costs for Each "
        "Distribution Center: Center 1: 151,000, Center 2: 192,000. "
        "Transportation Cost Per Unit from Each Distribution Center to Retail "
        "Stores: Costs vary for each distribution center and are specific to "
        "the route to each retail store, ranging from $1 to $5 per unit. "
        "Demand of Each Retail Store: Store demands range from 440 to 892 "
        "units. Supply Capacity of Each Distribution Center: capacities range "
        "from 814 to 1962 units. The goal is minimizing the total cost."
    )
    archive = {
        "archive_status": "complete",
        "case_id": "MAMO:complex_lp:151",
        "question": source,
        "artifacts": [],
        "solver_runs": [],
        "evaluation": {
            "prediction": {"status": "no_solver_run", "objective_value": None},
            "gold": {"status": "optimal", "objective_value": 607479.0},
            "metrics": {"correct": False},
        },
        "workflow_snapshots": [],
        "diagnostics": [
            {
                "diagnostic_type": "insufficient_facility_location_data",
                "severity": "error",
            }
        ],
    }

    adjudication = runner._gold_adjudication(archive)

    assert (
        adjudication["status"]
        == "gold_requires_summarized_facility_location_data"
    )


def test_orq_batch_runner_adjudicates_cash_product_mix_impossible_gold():
    runner = _load_runner()
    source = (
        "A company produces and sells two different products. Each unit of the "
        "first and second product requires 3 and 4 machine hours, respectively. "
        "There are 20,000 machine hours available in the current production "
        "period. The production costs are $3 and $2 per unit of the first and "
        "second product, respectively. The selling prices of the first and "
        "second product are $6 and $5.40 per unit, respectively. The available "
        "cash is $4,000; furthermore, 45% of the sales revenues from the first "
        "product and 30% of the sales revenues from the product will be made "
        "available to finance operations during the current period. Suppose "
        "that the company could increase its available machine hours by 2,000, "
        "after spending $400 for certain repairs. Find the maximum net income "
        "subject to the cash availability and machine capacity limitations."
    )
    archive = {
        "archive_status": "complete",
        "case_id": "MAMO:complex_lp:178",
        "question": source,
        "artifacts": [],
        "solver_runs": [],
        "evaluation": {
            "prediction": {"status": "optimal", "objective_value": 21600.0},
            "gold": {"status": "optimal", "objective_value": 242000.0},
            "metrics": {"correct": False},
        },
        "workflow_snapshots": [],
        "diagnostics": [],
    }

    adjudication = runner._gold_adjudication(archive)

    assert (
        adjudication["status"]
        == "gold_exceeds_cash_machine_revenue_upper_bound"
    )
    assert adjudication["adjudicated_correct"] is True


def test_orq_batch_runner_adjudicates_shortest_path_gold_mismatch():
    runner = _load_runner()
    source = (
        "There are two special nodes marked as S and T. Node S is connected to "
        "nodes 2 and 3 with edge weights of 5 and 4, respectively. Node 3 is "
        "connected to node S with a weight of 4, to node 5 with a weight of 1. "
        "Node 5 is connected to node 3 with a weight of 1 and to node T with "
        "a weight of 5. Considering the weight as distance, find the shortest "
        "distance from S to T."
    )
    archive = {
        "archive_status": "complete",
        "case_id": "MAMO:complex_lp:186",
        "question": source,
        "artifacts": [],
        "solver_runs": [],
        "evaluation": {
            "prediction": {"status": "optimal", "objective_value": 10.0},
            "gold": {"status": "optimal", "objective_value": 8.0},
            "metrics": {"correct": False},
        },
        "workflow_snapshots": [],
        "diagnostics": [],
    }

    adjudication = runner._gold_adjudication(archive)

    assert adjudication["status"] == "gold_mismatches_shortest_path_distance"
    assert adjudication["adjudicated_correct"] is True


def test_orq_batch_runner_adjudicates_freight_relocation_gold_mismatch():
    runner = _load_runner()
    source = (
        "The China Railroad Ministry is in the process of planning relocations "
        "of freight cars among 5 regions of the country to get ready for the "
        "fall harvest. location 1 to 3 costs 11 units. Moving a car from "
        "location 1 to 4 costs 13 units. Moving a car from location 1 to 5 "
        "costs 28 units. Moving a car from location 2 to 3 costs 18 units. "
        "Moving a car from location 2 to 4 costs 8 units. Moving a car from "
        "location 2 to 5 costs 46 units. Moving a car from location 3 to 4 "
        "costs 9 units. Moving a car from location 3 to 5 costs 27 units. "
        "Moving a car from location 4 to 5 costs 20 units. At location 1, "
        "there are currently 120 cars present, but 150 cars are needed. "
        "At location 2, there are currently 330 cars present, but 200 cars "
        "are needed. At location 3, there are currently 400 cars present, but "
        "600 cars are needed. At location 4, there are currently 400 cars "
        "present, but 200 cars are needed. At location 5, there are currently "
        "600 cars present, but 400 cars are needed. Write down a linear "
        "optimization to compute the least costly way to move the cars such us "
        "the need is met."
    )
    archive = {
        "archive_status": "complete",
        "case_id": "MAMO:complex_lp:185",
        "question": source,
        "artifacts": [],
        "solver_runs": [],
        "evaluation": {
            "prediction": {"status": "optimal", "objective_value": 2430.0},
            "gold": {"status": "optimal", "objective_value": 2400.0},
            "metrics": {"correct": False},
        },
        "workflow_snapshots": [],
        "diagnostics": [],
    }

    adjudication = runner._gold_adjudication(archive)

    assert (
        adjudication["status"]
        == "gold_mismatches_freight_relocation_shortest_path_cost"
    )
    assert adjudication["adjudicated_correct"] is True


def test_orq_batch_runner_adjudicates_vertex_cover_gold_mismatch():
    runner = _load_runner()
    source = (
        "The vertex cover problem is a classical optimization problem where the "
        "goal is to find the smallest set of vertices such that every edge in "
        "the graph is incident to at least one vertex in the set. Vertex 'a' "
        "connects to vertices 'f', 'e', and 'b'. Vertex 'b' connect to vertices "
        "'a', 'g', and 'c'. Vertex 'c' connects to vertices 'b', 'h', and 'd'. "
        "Vertex 'd' connects to vertices 'c', 'i', and 'e'. Vertex 'e' connects "
        "to vertices 'd', 'j', and 'a'. The vertices inside the pentagon "
        "('f', 'g', 'h', 'i', 'j') are all interconnected. Find the minimum "
        "vertex cover."
    )
    archive = {
        "archive_status": "complete",
        "case_id": "MAMO:complex_lp:200",
        "question": source,
        "artifacts": [],
        "solver_runs": [],
        "evaluation": {
            "prediction": {"status": "optimal", "objective_value": 7.0},
            "gold": {"status": "optimal", "objective_value": 5.0},
            "metrics": {"correct": False},
        },
        "workflow_snapshots": [],
        "diagnostics": [],
    }

    adjudication = runner._gold_adjudication(archive)

    assert adjudication["status"] == "gold_mismatches_vertex_cover_optimum"
    assert adjudication["adjudicated_correct"] is True


def test_orq_batch_runner_adjudicates_production_inventory_gold_below_lower_bound():
    runner = _load_runner()
    source = (
        "Basel Tool and Die Company makes pipe wrenches. The marketing department "
        "estimates demand during the next 6 months to be: In January, the demand "
        "is 430, in February, the demand is 430, in March, the demand is 380, "
        "in April, the demand is 450, in May, the demand is 520, in June, the "
        "demand is 440. With the current labor force, BTD can make approximately "
        "420 pipe wrenches per month at a cost of $40 per wrench using "
        "regular-time production. An additional 80 wrenches per month can be "
        "made using overtime production at a cost per wrench of $45. Wrenches "
        "can be held in inventory at a cost of $3 per month per wrench. At the "
        "beginning of January BTD has 10 wrenches in inventory. BTD wants to "
        "plan production and inventory while minimizing the total costs."
    )
    archive = {
        "archive_status": "complete",
        "case_id": "MAMO:complex_lp:201",
        "question": source,
        "artifacts": [],
        "solver_runs": [],
        "evaluation": {
            "prediction": {"status": "optimal", "objective_value": 106380.0},
            "gold": {"status": "optimal", "objective_value": 103960.0},
            "metrics": {"correct": False},
        },
        "workflow_snapshots": [],
        "diagnostics": [],
    }

    adjudication = runner._gold_adjudication(archive)

    assert (
        adjudication["status"]
        == "gold_below_production_inventory_cost_lower_bound"
    )
    assert adjudication["adjudicated_correct"] is True


def test_orq_batch_runner_adjudicates_simple_allocation_gold_below_explicit_lower_bound():
    runner = _load_runner()
    source = (
        "An agency allocates resources across three projects X, Y, and Z. "
        "The total investment across all three projects cannot exceed 1000. "
        "Project X must be at least twice that of project Y plus 200 units. "
        "Project Z requires a minimum of 100 more units than project Y. "
        "Each unit of investment in projects X, Y, and Z incurs costs quantified "
        "as 10, 15, and 20 units respectively. Allocations are integers."
    )
    archive = {
        "archive_status": "complete",
        "case_id": "MAMO:easy_lp:308",
        "question": source,
        "artifacts": [],
        "solver_runs": [],
        "evaluation": {
            "prediction": {"status": "optimal", "objective_value": 4000.0},
            "gold": {"status": "optimal", "objective_value": 2000.0},
            "metrics": {"correct": False},
        },
        "workflow_snapshots": [],
        "diagnostics": [],
    }

    adjudication = runner._gold_adjudication(archive)

    assert (
        adjudication["status"]
        == "gold_below_explicit_simple_allocation_cost_lower_bound"
    )
    assert adjudication["adjudicated_correct"] is True
    assert adjudication["cost_lower_bound"] == 4000.0
