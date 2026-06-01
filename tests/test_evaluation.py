from pathlib import Path

from orbit_or.evaluation import (
    evaluate_answer_faithfulness,
    evaluate_chunk_boundary_quality,
    evaluate_citation_accuracy,
    evaluate_component_extraction,
    evaluate_contradiction_detection,
    evaluate_gold_set_predictions,
    evaluate_ledger_extraction,
    evaluate_no_answer_calibration,
    evaluate_retrieval,
    evaluate_reranker_lift,
    evaluate_solver_result,
    evaluate_table_fidelity,
    load_or_mse_gold_set,
    load_orq_dataset,
    profile_or_problem_text,
    summarize_orq_cases,
    evaluate_orq_predictions,
    summarize_claim_review_outcomes,
    summarize_stage_latency,
)


def test_evaluate_component_extraction_scores_by_type_and_symbol():
    predicted = [
        {"component_type": "decision_variable", "symbol": "x_i"},
        {"component_type": "constraint", "natural_text": "capacity <= 10"},
    ]
    gold = [
        {"component_type": "decision_variable", "symbol": "x_i"},
        {"component_type": "objective", "natural_text": "minimize cost"},
    ]

    result = evaluate_component_extraction(predicted, gold)

    assert result["overall"]["true_positive"] == 1
    assert result["overall"]["precision"] == 0.5
    assert result["overall"]["recall"] == 0.5
    assert result["by_type"]["decision_variable"]["f1"] == 1.0


def test_evaluate_chunk_boundary_and_table_fidelity():
    chunk_result = evaluate_chunk_boundary_quality(
        [
            {"position_start": 0, "position_end": 10, "section_path": "Intro"},
            {"position_start": 10, "position_end": 20, "section_path": "Model"},
        ],
        [
            {"position_start": 0, "position_end": 10, "section_path": "Intro"},
            {"position_start": 12, "position_end": 22, "section_path": "Model"},
        ],
    )
    assert chunk_result["precision"] == 0.5
    assert chunk_result["recall"] == 0.5

    table_result = evaluate_table_fidelity(
        "| Site | Capacity |\n| --- | --- |\n| A | 10 |",
        "| Site | Capacity |\n| --- | --- |\n| A | 10 |",
    )
    assert table_result["exact_match"] is True
    assert table_result["f1"] == 1.0


def test_evaluate_retrieval_computes_recall_mrr_ndcg():
    result = evaluate_retrieval([9, 2, 3, 4], {2, 4, 8}, k=4)

    assert result["recall@4"] == 2 / 3
    assert result["mrr@4"] == 0.5
    assert 0 < result["ndcg@4"] <= 1


def test_evaluate_reranker_lift_and_citation_accuracy():
    lift = evaluate_reranker_lift([9, 8, 2], [2, 9, 8], {2}, k=3)

    assert lift["mrr_lift@3"] > 0
    assert lift["ndcg_lift@3"] > 0

    citations = evaluate_citation_accuracy(
        "Use [D1] and [F2], not [F999].",
        {"D1", "F2"},
        required_citations={"D1"},
    )
    assert citations["hallucinated"] == ["F999"]
    assert citations["required_recall"] == 1.0
    assert citations["passed"] is False


def test_evaluate_answer_faithfulness_and_no_answer_calibration():
    faithfulness = evaluate_answer_faithfulness(
        "Warehouse capacity is 10 pallets [D1]. Unsupported sentence [F2].",
        {"D1": "The warehouse capacity is 10 pallets per day.", "F2": "Demand is 8."},
        min_token_overlap=0.4,
    )

    assert faithfulness["checked_sentence_count"] == 2
    assert faithfulness["supported_sentence_count"] == 1
    assert faithfulness["passed"] is False

    calibration = evaluate_no_answer_calibration(
        [
            {"should_answer": True, "no_answer": False},
            {"should_answer": False, "no_answer": False},
            {"should_answer": True, "no_answer": True},
        ]
    )
    assert calibration["calibration_accuracy"] == 1 / 3
    assert calibration["false_answer_rate"] == 1 / 3
    assert calibration["false_abstain_rate"] == 1 / 3


def test_evaluate_contradictions_claim_reviews_ledger_and_latency():
    contradictions = evaluate_contradiction_detection(
        [(1, 2), (3, 4)],
        [(2, 1), (5, 6)],
    )
    assert contradictions["precision"] == 0.5
    assert contradictions["recall"] == 0.5

    reviews = summarize_claim_review_outcomes(
        [
            {"review_status": "accepted", "review_note": "ok"},
            {"review_status": "rejected", "review_note": "missing_scope; vague"},
            {"review_status": "rejected", "review_note": "missing_scope"},
        ]
    )
    assert reviews["acceptance_rate"] == 1 / 3
    assert reviews["reason_counts"]["missing_scope"] == 2

    ledger = evaluate_ledger_extraction(
        [{"entity": "Site A", "attribute": "capacity", "value": "10", "timeframe": "2025"}],
        [{"entity_name": "Site A", "attribute_name": "capacity", "value": "10", "normalized_timeframe": "2025"}],
    )
    assert ledger["f1"] == 1.0

    latency = summarize_stage_latency(
        [
            {"stage": "retrieval", "elapsed_time_s": 0.1},
            {"stage": "retrieval", "elapsed_time_s": 0.3},
            {"stage": "solver", "duration_s": 1.0},
        ]
    )
    assert latency["retrieval"]["count"] == 2
    assert latency["solver"]["max_s"] == 1.0


def test_evaluate_solver_result_handles_relative_tolerance_and_status():
    result = evaluate_solver_result(
        predicted_status="optimal",
        gold_status="optimal",
        predicted_objective=104.0,
        gold_objective=100.0,
        rel_tol=0.05,
    )

    assert result["correct"] is True
    assert result["objective_match"] is True

    failed = evaluate_solver_result(
        predicted_status="infeasible",
        gold_status="optimal",
        predicted_objective=None,
        gold_objective=100.0,
    )
    assert failed["correct"] is False

    no_finite_optimum = evaluate_solver_result(
        predicted_status="solver_infeasible",
        gold_status="no_solution_reported",
        predicted_objective=None,
        gold_objective=None,
    )
    assert no_finite_optimum["correct"] is True
    assert no_finite_optimum["status_match"] is True


def test_load_and_evaluate_or_mse_gold_fixture():
    fixture = Path(__file__).parent / "fixtures" / "or_mse_gold.json"
    cases = load_or_mse_gold_set(fixture)

    assert {case["task_type"] for case in cases} == {
        "component_extraction",
        "solver",
        "retrieval",
    }

    predictions = {
        "nl4opt_component_capacity_001": {
            "components": [
                {"component_type": "decision_variable", "symbol": "x"},
                {"component_type": "objective", "symbol": "obj"},
                {"component_type": "constraint", "symbol": "demand"},
                {"component_type": "constraint", "symbol": "capacity"},
            ]
        },
        "mamo_solver_lp_001": {
            "status": "optimal",
            "objective_value": 1.0,
        },
        "nl4opt_component_transport_002": {
            "components": [
                {"component_type": "set", "symbol": "i"},
                {"component_type": "set", "symbol": "j"},
                {"component_type": "decision_variable", "symbol": "x_ij"},
                {"component_type": "parameter", "symbol": "c_ij"},
                {"component_type": "objective", "symbol": "obj"},
                {"component_type": "constraint", "symbol": "supply"},
                {"component_type": "constraint", "symbol": "demand"},
            ]
        },
        "mamo_solver_infeasible_002": {
            "status": "infeasible",
            "objective_value": None,
        },
        "mamo_solver_binary_003": {
            "status": "optimal",
            "objective_value": 5.0,
        },
        "orqa_retrieval_components_001": {
            "ranked_ids": [101, 200, 103],
        },
        "orqa_retrieval_solver_status_002": {
            "ranked_ids": [300, 201, 204, 500],
        },
    }

    result = evaluate_gold_set_predictions(cases, predictions)

    assert result["case_count"] == 7
    assert result["evaluated_count"] == 7
    assert result["solver_accuracy"] == 1.0
    assert result["mean_component_f1"] == 1.0
    assert result["mean_retrieval_recall"] == 1.0


def test_load_and_evaluate_orq_dataset(tmp_path):
    root = tmp_path / "ORQ_Dataset"
    (root / "IndustryOR").mkdir(parents=True)
    (root / "MAMO").mkdir(parents=True)
    (root / "NL4OPT").mkdir(parents=True)
    (root / "IndustryOR" / "IndustryOR.json").write_text(
        '[{"id": 7, "difficulty": "Hard", "en_question": "Maximize x.", "en_answer": "3"}]',
        encoding="utf-8",
    )
    (root / "MAMO" / "MAMO_EasyLP.json").write_text(
        '{"en_question": "Minimize x.", "en_answer": "1.0"}\n',
        encoding="utf-8",
    )
    (root / "MAMO" / "MAMO_ComplexLP.json").write_text(
        '{"en_question": "No feasible model.", "en_answer": "No Best Solution"}\n',
        encoding="utf-8",
    )
    (root / "NL4OPT" / "NL4OPT_with_optimal_solution.json").write_text(
        '{"en_question": "Minimize y.", "en_answer": "2"}\n',
        encoding="utf-8",
    )

    cases = load_orq_dataset(root)
    summary = summarize_orq_cases(cases)

    assert summary["case_count"] == 4
    assert summary["datasets"] == {"IndustryOR": 1, "MAMO": 2, "NL4OPT": 1}
    assert summary["gold_statuses"]["no_solution_reported"] == 1
    assert summary["workflow_hints"]["solver_modeling"] == 3
    assert cases[0]["answer_kind"] == "numeric_objective"

    predictions = {
        "IndustryOR:test:7": {"status": "optimal", "objective_value": 3.0},
        "MAMO:easy_lp:1": [
            {"status": "optimal", "objective_value": 9.0},
            {"status": "optimal", "objective_value": 1.0},
        ],
        "MAMO:complex_lp:1": {"status": "no_solution_reported"},
        "NL4OPT:test:1": {"status": "optimal", "objective_value": 2.0},
    }
    result = evaluate_orq_predictions(cases, predictions, k=2)

    assert result["evaluated_count"] == 4
    assert result["pass@1"] == 0.75
    assert result["pass@2"] == 1.0


def test_load_orq_dataset_treats_big_m_sentinel_as_no_solution(tmp_path):
    root = tmp_path / "ORQ_Dataset"
    (root / "IndustryOR").mkdir(parents=True)
    (root / "MAMO").mkdir(parents=True)
    (root / "NL4OPT").mkdir(parents=True)
    (root / "MAMO" / "MAMO_EasyLP.json").write_text(
        '{"en_question": "Infeasible integer allocation.", "en_answer": "1000000000000"}\n',
        encoding="utf-8",
    )

    cases = load_orq_dataset(root)

    assert cases[0]["gold_solver"]["status"] == "no_solution_reported"
    assert cases[0]["gold_solver"]["objective_value"] is None


def test_orq_problem_profile_uses_text_not_dataset_name(tmp_path):
    profile = profile_or_problem_text(
        problem_text="How many boxes are left over after these sales?"
    )
    assert profile["workflow_hint"] == "direct_calculation"

    industrial_direct = profile_or_problem_text(
        problem_text=(
            "Annual production is 10 jets in year 1 and 15 in year 2. "
            "Each training jet can train 5 pilots per year. How many trained "
            "pilots are available by the end of year 2?"
        )
    )
    assert industrial_direct["workflow_hint"] == "direct_calculation"
    assert industrial_direct["industrial_realism"] == "medium"

    goal_profile = profile_or_problem_text(
        problem_text=(
            "The company considers priorities p_1, p_2, and p_3 in a goal "
            "programming model. How many workers miss the P3 target?"
        )
    )
    assert goal_profile["problem_family_hint"] == "goal_programming"
    assert goal_profile["workflow_hint"] != "direct_calculation"

    scheduling_profile = profile_or_problem_text(
        problem_text=(
            "Three jobs must be processed on machine 1, then sequentially on "
            "machines 2 and 3. Minimize the completion time."
        )
    )
    assert scheduling_profile["problem_family_hint"] == "flow_shop_scheduling"

    workforce_profile = profile_or_problem_text(
        problem_text=(
            "A hospital schedules nurses. d1 = 2, d2 = 4, d3 = 4, d4 = 3, "
            "d5 = 1, d6 = 2, d7 = 3. Every nurse works 5 days in a row. "
            "Minimize the total number of nurses."
        )
    )
    assert workforce_profile["problem_family_hint"] == "workforce_scheduling"

    duty_profile = profile_or_problem_text(
        problem_text=(
            "A lab hires students for duty shifts. Each student has a wage "
            "and can be scheduled no more than two shifts. Minimize gross pay."
        )
    )
    assert duty_profile["problem_family_hint"] == "workforce_scheduling"

    shift_coverage_profile = profile_or_problem_text(
        problem_text=(
            "A restaurant operates around the clock. Waiters work continuously "
            "for 8 hours, and the goal is to find the minimum number of waiters "
            "needed across the 24-hour demand periods."
        )
    )
    assert shift_coverage_profile["problem_family_hint"] == "workforce_scheduling"

    network_profile = profile_or_problem_text(
        problem_text=(
            "There is an edge from node 1 to node 2 with capacity 8. "
            "Find the maximum flow from the source to the terminal."
        )
    )
    assert network_profile["problem_family_hint"] == "network_flow"

    overtime_mix_profile = profile_or_problem_text(
        problem_text=(
            "Two products use assembly labor, testing, and raw materials. "
            "Overtime assembly labor is available at a cost per hour. "
            "Maximize profit."
        )
    )
    assert overtime_mix_profile["problem_family_hint"] == "overtime_product_mix"

    subset_profile = profile_or_problem_text(
        problem_text=(
            "A company must choose candidates to hire, with a budget, at least "
            "two hires, and at most one from two equivalent candidates. "
            "Minimize total salary."
        )
    )
    assert subset_profile["problem_family_hint"] == "binary_subset_selection"

    cutting_profile = profile_or_problem_text(
        problem_text=(
            "A workshop cuts raw steel bars into ordered pieces and wants to "
            "minimize total cutting waste."
        )
    )
    assert cutting_profile["problem_family_hint"] == "cutting_stock"

    contract_profile = profile_or_problem_text(
        problem_text=(
            "A factory must rent warehouse space for several months using "
            "contracts of different lengths to cover required area. Minimize "
            "total rental cost."
        )
    )
    assert contract_profile["problem_family_hint"] == "interval_contract_covering"

    resource_mix_profile = profile_or_problem_text(
        problem_text=(
            "A carrier compares transportation methods, each with trip "
            "capacity and pollution, to transport demand at minimum pollution."
        )
    )
    assert resource_mix_profile["problem_family_hint"] == "integer_resource_mix"

    procurement_profile = profile_or_problem_text(
        problem_text=(
            "A factory orders parts from suppliers with fixed lot sizes, "
            "minimum required demand, and per-lot freight cost. Minimize cost."
        )
    )
    assert procurement_profile["problem_family_hint"] == "procurement_lot_mix"

    nutrition_profile = profile_or_problem_text(
        problem_text=(
            "A feed mix table lists protein, minerals, vitamins, and price. "
            "Meet at least the required nutrition while minimizing cost."
        )
    )
    assert nutrition_profile["problem_family_hint"] == "nutrition_mix"

    narrative_transport_profile = profile_or_problem_text(
        problem_text=(
            "Two coal yards supply three residential areas. Distances in "
            "kilometers are listed for each yard, and the goal is to minimize "
            "ton-kilometers of transportation."
        )
    )
    assert narrative_transport_profile["problem_family_hint"] == "transportation"

    fixed_transshipment_profile = profile_or_problem_text(
        problem_text=(
            "Production points ship through intermediate marshaling stations "
            "with fixed cost and transshipment capacity before reaching demand "
            "points. Minimize total cost."
        )
    )
    assert fixed_transshipment_profile["problem_family_hint"] == "fixed_charge_transshipment"

    package_mix_profile = profile_or_problem_text(
        problem_text=(
            "A store creates promotional packages from shirts and pants, with "
            "package prices and warehouse stock limits. Maximize revenue."
        )
    )
    assert package_mix_profile["problem_family_hint"] == "product_mix"

    root = tmp_path / "ORQ_Dataset"
    (root / "IndustryOR").mkdir(parents=True)
    (root / "NL4OPT").mkdir(parents=True)
    (root / "IndustryOR" / "IndustryOR.json").write_text(
        '[{"id": 1, "en_question": "Minimize production cost x.", "en_answer": "1"}]',
        encoding="utf-8",
    )
    (root / "NL4OPT" / "NL4OPT_with_optimal_solution.json").write_text(
        '{"en_question": "How many boxes are left over after these sales?", "en_answer": "4"}\n',
        encoding="utf-8",
    )

    cases = {case["id"]: case for case in load_orq_dataset(root)}

    assert cases["IndustryOR:test:1"]["workflow_hint"] == "solver_modeling"
    assert cases["NL4OPT:test:1"]["workflow_hint"] == "direct_calculation"
