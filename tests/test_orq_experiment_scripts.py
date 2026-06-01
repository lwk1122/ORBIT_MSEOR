import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _write_small_orq_dataset(root: Path) -> None:
    (root / "NL4OPT").mkdir(parents=True)
    (root / "MAMO").mkdir(parents=True)
    (root / "IndustryOR").mkdir(parents=True)
    (root / "NL4OPT" / "NL4OPT_with_optimal_solution.json").write_text(
        '{"en_question": "Maximize x.", "en_answer": "10"}\n',
        encoding="utf-8",
    )
    (root / "MAMO" / "MAMO_EasyLP.json").write_text(
        '{"en_question": "Minimize y.", "en_answer": "5"}\n',
        encoding="utf-8",
    )
    (root / "MAMO" / "MAMO_ComplexLP.json").write_text("", encoding="utf-8")
    (root / "IndustryOR" / "IndustryOR.json").write_text("[]", encoding="utf-8")


def test_normalize_direct_baseline_writes_predictions_and_summary(tmp_path):
    script = _load_script("normalize_direct_baseline.py")
    dataset_root = tmp_path / "ORQ_Dataset"
    _write_small_orq_dataset(dataset_root)
    source = tmp_path / "direct.json"
    source.write_text(
        json.dumps(
            {
                "NL4OPT": {
                    "dataset": "NL4OPT",
                    "mode": "direct",
                    "model": "MiniMax-M2.7",
                    "pass_at_1": 100.0,
                    "correct": 1,
                    "total": 1,
                    "results": [
                        {
                            "idx": 0,
                            "question": "Maximize x.",
                            "gold_answer": "10",
                            "predicted": 10,
                            "status": "optimal",
                            "correct": True,
                            "time": 1.2,
                            "has_code": True,
                        }
                    ],
                },
                "MAMO_EasyLP": {
                    "dataset": "MAMO_EasyLP",
                    "mode": "direct",
                    "model": "MiniMax-M2.7",
                    "pass_at_1": 0.0,
                    "correct": 0,
                    "total": 1,
                    "results": [
                        {
                            "idx": 0,
                            "question": "Minimize y.",
                            "gold_answer": "5",
                            "predicted": 8,
                            "status": "optimal",
                            "correct": False,
                            "time": 2.0,
                            "has_code": True,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "baseline"

    assert script.main(
        ["--input", str(source), "--root", str(dataset_root), "--output-dir", str(output_dir)]
    ) == 0

    predictions = json.loads((output_dir / "predictions.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert predictions["predictions"]["NL4OPT:test:1"]["objective_value"] == 10.0
    assert predictions["predictions"]["NL4OPT:test:1"]["source"] == "direct_minimax_recorded"
    assert predictions["predictions"]["MAMO:easy_lp:1"]["objective_value"] == 8.0
    assert summary["evaluation"]["evaluated_count"] == 2
    assert summary["evaluation"]["pass@1"] == 0.5


def test_aggregate_orq_experiments_writes_main_table(tmp_path):
    script = _load_script("aggregate_orq_experiments.py")
    dataset_root = tmp_path / "ORQ_Dataset"
    _write_small_orq_dataset(dataset_root)
    method_a = tmp_path / "method_a"
    method_b = tmp_path / "method_b"
    method_a.mkdir()
    method_b.mkdir()
    (method_a / "predictions.json").write_text(
        json.dumps(
            {
                "predictions": {
                    "NL4OPT:test:1": {"status": "optimal", "objective_value": 10},
                    "MAMO:easy_lp:1": {"status": "optimal", "objective_value": 5},
                }
            }
        ),
        encoding="utf-8",
    )
    (method_b / "predictions.json").write_text(
        json.dumps(
            {
                "predictions": {
                    "NL4OPT:test:1": {"status": "optimal", "objective_value": 7},
                    "MAMO:easy_lp:1": {"status": "optimal", "objective_value": 5},
                }
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "summary"

    assert script.main(
        [
            "--root",
            str(dataset_root),
            "--method",
            f"a={method_a}",
            "--method",
            f"b={method_b}",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    rows = list(csv.DictReader((output_dir / "main_accuracy.csv").open()))
    by_method = {row["method"]: row for row in rows if row["dataset"] == "<all>"}

    assert by_method["a"]["evaluated_count"] == "2"
    assert float(by_method["a"]["pass@1"]) == 1.0
    assert float(by_method["b"]["pass@1"]) == 0.5


def test_aggregate_orq_experiments_accepts_case_set(tmp_path):
    script = _load_script("aggregate_orq_experiments.py")
    dataset_root = tmp_path / "ORQ_Dataset"
    _write_small_orq_dataset(dataset_root)
    method = tmp_path / "method"
    method.mkdir()
    (method / "predictions.json").write_text(
        json.dumps(
            {
                "predictions": {
                    "NL4OPT:test:1": {"status": "optimal", "objective_value": 10},
                    "MAMO:easy_lp:1": {"status": "optimal", "objective_value": 5},
                }
            }
        ),
        encoding="utf-8",
    )
    case_set = tmp_path / "cases.json"
    case_set.write_text(
        json.dumps({"cases": [{"case_id": "MAMO:easy_lp:1"}]}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "summary"

    assert script.main(
        [
            "--root",
            str(dataset_root),
            "--case-set",
            str(case_set),
            "--method",
            f"a={method}",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    summary = json.loads((output_dir / "main_accuracy.json").read_text(encoding="utf-8"))

    assert summary["a"]["evaluation"]["case_count"] == 1
    assert summary["a"]["evaluation"]["pass@1"] == 1.0


def test_factorial_ablation_builds_binary_configs():
    script = _load_script("run_orq_factorial_ablation.py")

    configs = script._factorial_configs()

    assert len(configs) == 32
    assert configs[0] == {
        "review": True,
        "solver_validation": True,
        "adapters": True,
        "repair": True,
        "fallback": True,
    }
    assert configs[-1] == {
        "review": False,
        "solver_validation": False,
        "adapters": False,
        "repair": False,
        "fallback": False,
    }
    assert script._method_name(configs[0]) == (
        "review1_solver_validation1_adapters1_repair1_fallback1"
    )


def test_factorial_ablation_loads_selected_cases(tmp_path):
    script = _load_script("run_orq_factorial_ablation.py")
    dataset_root = tmp_path / "ORQ_Dataset"
    _write_small_orq_dataset(dataset_root)
    case_set = tmp_path / "cases.json"
    case_set.write_text(
        json.dumps({"cases": ["MAMO:easy_lp:1"]}),
        encoding="utf-8",
    )

    selected = script._selected_cases(dataset_root, case_set)

    assert [case["id"] for case in selected] == ["MAMO:easy_lp:1"]


def test_factorial_ablation_row_counts_only_selected_predictions(tmp_path):
    script = _load_script("run_orq_factorial_ablation.py")
    dataset_root = tmp_path / "ORQ_Dataset"
    _write_small_orq_dataset(dataset_root)
    all_cases = script.load_orq_dataset(dataset_root)
    nl4opt_cases = [case for case in all_cases if case["id"] == "NL4OPT:test:1"]
    predictions = {
        "NL4OPT:test:1": {"status": "optimal", "objective_value": 10},
        "MAMO:easy_lp:1": {
            "status": "unknown",
            "objective_value": None,
            "direct_fallback_used": True,
        },
    }

    row = script._row_for_method(
        method="m",
        config={factor: True for factor in script.FACTORS},
        cases=nl4opt_cases,
        predictions=predictions,
    )

    assert row["case_count"] == 1
    assert row["correct"] == 1
    assert row["fallback_used_count"] == 0
    assert row["unknown_count"] == 0


def test_factorial_ablation_main_effect_rows_support_groups():
    script = _load_script("run_orq_factorial_ablation.py")

    def row(dataset: str, review: int, score: float) -> dict:
        payload = {
            "dataset": dataset,
            "split": "test",
            "pass@1": score,
        }
        payload.update({factor: 1 for factor in script.FACTORS})
        payload["review"] = review
        return payload

    rows = [
        row("A", 1, 0.75),
        row("A", 0, 0.25),
        row("B", 1, 0.50),
        row("B", 0, 0.25),
    ]

    effects = script._main_effect_rows(rows, group_fields=("dataset", "split"))
    review_effects = {
        effect["dataset"]: effect
        for effect in effects
        if effect["factor"] == "review"
    }

    assert review_effects["A"]["enabled_mean"] == 0.75
    assert review_effects["A"]["disabled_mean"] == 0.25
    assert review_effects["A"]["delta"] == 0.5
    assert review_effects["B"]["delta"] == 0.25


def test_compare_orq_methods_writes_paired_statistics(tmp_path):
    script = _load_script("compare_orq_methods.py")
    per_case = tmp_path / "per_case.json"
    per_case.write_text(
        json.dumps(
            {
                "case-1": {
                    "base": {"dataset": "NL4OPT", "adjusted_pass@1": True},
                    "orbit": {"dataset": "NL4OPT", "adjusted_pass@1": True},
                },
                "case-2": {
                    "base": {"dataset": "NL4OPT", "adjusted_pass@1": False},
                    "orbit": {"dataset": "NL4OPT", "adjusted_pass@1": True},
                },
                "case-3": {
                    "base": {"dataset": "NL4OPT", "adjusted_pass@1": True},
                    "orbit": {"dataset": "NL4OPT", "adjusted_pass@1": False},
                },
                "case-4": {
                    "base": {"dataset": "MAMO", "adjusted_pass@1": False},
                    "orbit": {"dataset": "MAMO", "adjusted_pass@1": True},
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "stats.json"

    assert script.main(
        [
            "--per-case",
            str(per_case),
            "--baseline",
            "base",
            "--challenger",
            "orbit",
            "--dataset",
            "NL4OPT",
            "--bootstrap-samples",
            "20",
            "--output",
            str(output),
        ]
    ) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    comparison = payload["comparisons"][0]

    assert comparison["paired_case_count"] == 3
    assert comparison["both_correct"] == 1
    assert comparison["baseline_only"] == 1
    assert comparison["challenger_only"] == 1
    assert comparison["baseline_accuracy"]["mean"] == 2 / 3
    assert comparison["challenger_accuracy"]["mean"] == 2 / 3
    assert (tmp_path / "stats.csv").exists()


def test_build_orq_fallback_predictions_replaces_uncovered_cases(tmp_path):
    script = _load_script("build_orq_fallback_predictions.py")
    dataset_root = tmp_path / "ORQ_Dataset"
    _write_small_orq_dataset(dataset_root)
    primary = tmp_path / "primary.json"
    fallback = tmp_path / "fallback.json"
    primary.write_text(
        json.dumps(
            {
                "predictions": {
                    "NL4OPT:test:1": {"status": "optimal", "objective_value": 10},
                    "MAMO:easy_lp:1": {
                        "status": "unknown",
                        "objective_value": None,
                        "reason": "deterministic_uncovered",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    fallback.write_text(
        json.dumps(
            {
                "predictions": {
                    "MAMO:easy_lp:1": {"status": "optimal", "objective_value": 5}
                }
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "hybrid"

    assert script.main(
        [
            "--root",
            str(dataset_root),
            "--primary",
            str(primary),
            "--fallback",
            str(fallback),
            "--primary-name",
            "orbit",
            "--fallback-name",
            "direct",
            "--dataset",
            "MAMO",
            "--split",
            "easy_lp",
            "--fallback-reason",
            "deterministic_uncovered",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    payload = json.loads((output_dir / "predictions.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    replacement = payload["predictions"]["MAMO:easy_lp:1"]

    assert replacement["objective_value"] == 5
    assert replacement["fallback_used"] is True
    assert summary["fallback_summary"]["fallback_applied_count"] == 1
    assert summary["evaluation"]["pass@1"] == 1.0


def test_audit_adjusted_correct_reports_raw_incorrect_cases(tmp_path):
    script = _load_script("audit_adjusted_correct.py")
    dataset_root = tmp_path / "ORQ_Dataset"
    _write_small_orq_dataset(dataset_root)
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps(
            {
                "predictions": {
                    "NL4OPT:test:1": {
                        "status": "optimal",
                        "objective_value": 9,
                        "adjudicated_correct": True,
                        "gold_adjudication_status": "gold_below_test_lower_bound",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "audit"

    assert script.main(
        [
            "--root",
            str(dataset_root),
            "--predictions",
            str(predictions),
            "--method-name",
            "method",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    audit = json.loads((output_dir / "adjusted_correct_audit.json").read_text())

    assert audit["raw_incorrect_adjusted_correct_count"] == 1
    assert audit["by_category"] == {"gold_bound_violation": 1}
    assert audit["policy"]["primary_metric"] == "pass@1"
    assert (output_dir / "adjusted_correct_audit.csv").exists()
    assert (output_dir / "adjusted_correct_audit.md").exists()


def test_build_orq_small_ablation_writes_dataset_and_tables(tmp_path):
    script = _load_script("build_orq_small_ablation.py")
    dataset_root = tmp_path / "ORQ_Dataset"
    _write_small_orq_dataset(dataset_root)
    direct = tmp_path / "direct.json"
    native = tmp_path / "native.json"
    full = tmp_path / "full.json"
    direct.write_text(
        json.dumps(
            {
                "predictions": {
                    "NL4OPT:test:1": {"status": "optimal", "objective_value": 10},
                    "MAMO:easy_lp:1": {"status": "optimal", "objective_value": 4},
                }
            }
        ),
        encoding="utf-8",
    )
    native.write_text(
        json.dumps(
            {
                "predictions": {
                    "NL4OPT:test:1": {"status": "optimal", "objective_value": 8},
                    "MAMO:easy_lp:1": {"status": "unknown", "objective_value": None},
                }
            }
        ),
        encoding="utf-8",
    )
    full.write_text(
        json.dumps(
            {
                "predictions": {
                    "NL4OPT:test:1": {"status": "optimal", "objective_value": 10},
                    "MAMO:easy_lp:1": {
                        "status": "optimal",
                        "objective_value": 5,
                        "fallback_used": True,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "small_ablation"

    assert script.main(
        [
            "--root",
            str(dataset_root),
            "--output-dir",
            str(output_dir),
            "--per-split",
            "1",
            "--method",
            f"direct_minimax_recorded={direct}",
            "--method",
            f"orbit_full_no_direct_fallback={native}",
            "--method",
            f"orbit_full_direct_fallback={full}",
        ]
    ) == 0

    cases = json.loads((output_dir / "cases.json").read_text(encoding="utf-8"))
    main_rows = list(csv.DictReader((output_dir / "summaries" / "main_accuracy.csv").open()))

    assert len(cases["cases"]) == 2
    assert {row["method"] for row in main_rows} == {
        "direct_minimax_recorded",
        "orbit_full_no_direct_fallback",
        "orbit_full_direct_fallback",
        "orbit_no_adapters_proxy",
    }
    assert (output_dir / "summaries" / "paired_ablation.csv").exists()
    assert (output_dir / "summaries" / "small_ablation_report.md").exists()


def test_build_orq_formal_results_bundle_writes_tables(tmp_path):
    script = _load_script("build_orq_formal_results_bundle.py")
    dataset_root = tmp_path / "ORQ_Dataset"
    _write_small_orq_dataset(dataset_root)
    base = tmp_path / "base.json"
    orbit = tmp_path / "orbit.json"
    base.write_text(
        json.dumps(
            {
                "predictions": {
                    "NL4OPT:test:1": {"status": "optimal", "objective_value": 10},
                    "MAMO:easy_lp:1": {"status": "optimal", "objective_value": 8},
                }
            }
        ),
        encoding="utf-8",
    )
    orbit.write_text(
        json.dumps(
            {
                "predictions": {
                    "NL4OPT:test:1": {"status": "optimal", "objective_value": 10},
                    "MAMO:easy_lp:1": {
                        "status": "optimal",
                        "objective_value": 5,
                        "fallback_used": True,
                        "fallback_method": "base",
                        "primary_status": "unknown",
                        "primary_reason": "deterministic_uncovered",
                        "runtime_s": 1.5,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "bundle"

    assert script.main(
        [
            "--root",
            str(dataset_root),
            "--method",
            f"base={base}",
            "--method",
            f"orbit={orbit}",
            "--baseline",
            "base",
            "--challenger",
            "orbit",
            "--bootstrap-samples",
            "20",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    main_rows = list(csv.DictReader((output_dir / "main_accuracy.csv").open()))
    fallback_rows = list(csv.DictReader((output_dir / "fallback_summary.csv").open()))
    stats_rows = list(csv.DictReader((output_dir / "statistical_tests_raw.csv").open()))

    overall = {
        row["method"]: row
        for row in main_rows
        if row["dataset"] == "<all>" and row["split"] == "<all>"
    }
    assert float(overall["base"]["pass@1"]) == 0.5
    assert float(overall["orbit"]["pass@1"]) == 1.0
    assert any(row["method"] == "orbit" and row["case_count"] == "1" for row in fallback_rows)
    assert stats_rows[0]["baseline"] == "base"
    assert (output_dir / "formal_results_report.md").exists()
    assert (output_dir / "prediction_record_schema.csv").exists()


def test_evaluate_orq_candidate_tournaments_writes_passk_outputs(tmp_path):
    script = _load_script("evaluate_orq_candidate_tournaments.py")
    dataset_root = tmp_path / "ORQ_Dataset"
    _write_small_orq_dataset(dataset_root)
    archive = tmp_path / "archive.json"
    archive.write_text(
        json.dumps(
            {
                "case_id": "NL4OPT:test:1",
                "tournament": {
                    "ranked": [
                        {
                            "candidate_index": 0,
                            "score": 50,
                            "component_count": 3,
                            "content": (
                                "Maximize\n"
                                " obj: 5 x\n"
                                "Subject To\n"
                                " c1: x <= 1\n"
                                "Bounds\n"
                                " x >= 0\n"
                                "End"
                            ),
                            "validation": {"status": "valid"},
                        },
                        {
                            "candidate_index": 1,
                            "score": 80,
                            "component_count": 3,
                            "content": (
                                "Maximize\n"
                                " obj: 10 x\n"
                                "Subject To\n"
                                " c1: x <= 1\n"
                                "Bounds\n"
                                " x >= 0\n"
                                "End"
                            ),
                            "validation": {"status": "valid"},
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "passk"

    assert script.main(
        [
            "--root",
            str(dataset_root),
            "--archive",
            str(archive),
            "--k",
            "2",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    summary = json.loads(
        (output_dir / "candidate_passk_summary.json").read_text(encoding="utf-8")
    )
    rows = list(csv.DictReader((output_dir / "candidate_rows.csv").open()))

    assert summary["evaluation"]["pass@1"] == 0.0
    assert summary["evaluation"]["pass@2"] == 1.0
    assert len(rows) == 2
    assert rows[1]["correct"] == "True"


def test_analyze_orq_failures_writes_taxonomy_outputs(tmp_path):
    script = _load_script("analyze_orq_failures.py")
    archive = tmp_path / "archive.json"
    archive.write_text(
        json.dumps(
            {
                "question": "Uncovered small LP.",
                "evaluation": {
                    "prediction": {
                        "status": "unknown",
                        "objective_value": None,
                        "reason": "deterministic_uncovered",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "case_id": "MAMO:easy_lp:1",
                        "dataset": "MAMO",
                        "split": "easy_lp",
                        "correct": True,
                        "predicted_status": "optimal",
                        "predicted_objective": 10,
                        "gold_status": "optimal",
                        "gold_objective": 10,
                        "failure_signature": "correct",
                    }
                ),
                json.dumps(
                    {
                        "case_id": "MAMO:easy_lp:2",
                        "dataset": "MAMO",
                        "split": "easy_lp",
                        "correct": None,
                        "predicted_status": "unknown",
                        "predicted_objective": None,
                        "gold_status": "optimal",
                        "gold_objective": 168,
                        "model_class": None,
                        "solver_route": None,
                        "failure_signature": "prediction:unknown",
                        "gold_adjudication_reason": None,
                        "archive_path": str(archive),
                    }
                ),
                json.dumps(
                    {
                        "case_id": "MAMO:easy_lp:3",
                        "dataset": "MAMO",
                        "split": "easy_lp",
                        "correct": False,
                        "predicted_status": "optimal",
                        "predicted_objective": 12,
                        "gold_status": "optimal",
                        "gold_objective": 10,
                        "objective_match": False,
                        "failure_signature": "objective_mismatch",
                    }
                ),
                json.dumps(
                    {
                        "case_id": "MAMO:easy_lp:4",
                        "dataset": "MAMO",
                        "split": "easy_lp",
                        "correct": None,
                        "predicted_status": "error",
                        "failure_signature": "case timed out after 30s",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "failures"

    assert script.main([str(manifest), "--output-dir", str(output_dir)]) == 0

    taxonomy = json.loads((output_dir / "failure_taxonomy.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((output_dir / "failure_taxonomy_by_dataset.csv").open()))
    examples = (output_dir / "failure_examples.md").read_text(encoding="utf-8")

    assert taxonomy["failure_label_counts"]["correct"] == 1
    assert taxonomy["non_correct_label_counts"]["unsupported_model_class"] == 1
    assert taxonomy["non_correct_label_counts"]["wrong_objective_expression"] == 1
    assert taxonomy["non_correct_label_counts"]["timeout"] == 1
    assert {row["failure_label"] for row in rows} >= {
        "unsupported_model_class",
        "wrong_objective_expression",
        "timeout",
    }
    assert "MAMO:easy_lp:2" in examples


def test_evaluate_orq_dataset_filters_selected_dataset(tmp_path, capsys):
    script = _load_script("evaluate_orq_dataset.py")
    dataset_root = tmp_path / "ORQ_Dataset"
    _write_small_orq_dataset(dataset_root)
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps(
            {
                "predictions": {
                    "NL4OPT:test:1": {"status": "optimal", "objective_value": 10},
                }
            }
        ),
        encoding="utf-8",
    )

    assert script.main(
        [
            "--root",
            str(dataset_root),
            "--dataset",
            "NL4OPT",
            "--predictions",
            str(predictions),
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["case_count"] == 1
    assert payload["evaluated_count"] == 1
    assert payload["pass@1"] == 1.0
