import importlib.util
from pathlib import Path


def _load_batch_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_nl4opt_minimax_batch.py"
    )
    spec = importlib.util.spec_from_file_location("run_nl4opt_minimax_batch", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batch_routes_direct_calculation_to_direct_answer_when_enabled():
    module = _load_batch_module()
    decision = module._case_routing_decision(
        {
            "problem_text": (
                "Annual production is 10 jets in year 1 and 15 in year 2. "
                "Each training jet can train 5 pilots per year. How many "
                "trained pilots are available by the end of year 2?"
            ),
            "gold_solver": {"status": "optimal", "objective_value": 4.0},
        },
        workflow_mode="modeling_fast",
        solver_only_components=True,
        direct_answer_fallback=True,
        auto_route=True,
    )

    assert decision["problem_profile"]["workflow_hint"] == "direct_calculation"
    assert decision["route_strategy"] == "direct_answer"
    assert decision["solver_only_components"] is False


def test_batch_keeps_industrial_modeling_in_model_route_without_solver_only_prompt():
    module = _load_batch_module()
    decision = module._case_routing_decision(
        {
            "problem_text": (
                "Minimize weekly production cost.\n\n"
                "| Week | Demand |\n| --- | --- |\n| 1 | 10 |"
            ),
            "gold_solver": {"status": "optimal", "objective_value": 12.0},
        },
        workflow_mode="modeling_fast",
        solver_only_components=True,
        direct_answer_fallback=True,
        auto_route=True,
    )

    assert decision["problem_profile"]["workflow_hint"] == "industrial_modeling_reviewed"
    assert decision["route_strategy"] == "optimization_model"
    assert decision["workflow_mode"] == "modeling_fast"
    assert decision["solver_only_components"] is False


def test_batch_keeps_known_or_family_out_of_direct_answer_route():
    module = _load_batch_module()
    decision = module._case_routing_decision(
        {
            "problem_text": (
                "The company considers priorities p_1, p_2, and p_3 in a "
                "goal programming model. How many workers miss the P3 target?"
            ),
            "gold_solver": {"status": "optimal", "objective_value": 2.0},
        },
        workflow_mode="modeling_fast",
        solver_only_components=True,
        direct_answer_fallback=True,
        auto_route=True,
    )

    assert decision["problem_profile"]["problem_family_hint"] == "goal_programming"
    assert decision["problem_profile"]["workflow_hint"] != "direct_calculation"
    assert decision["route_strategy"] == "optimization_model"


def test_batch_routes_quality_mixing_as_blending_not_generic_product_mix():
    module = _load_batch_module()
    decision = module._case_routing_decision(
        {
            "problem_text": (
                "A chemical plant mixes liquid raw materials A, B, and C with "
                "different sulfur contents to produce two products. Product "
                "sulfur content must not exceed stated limits, and the plant "
                "should maximize profit subject to market demand."
            ),
            "gold_solver": {"status": "optimal", "objective_value": 115.0},
        },
        workflow_mode="modeling_fast",
        solver_only_components=True,
        direct_answer_fallback=True,
        auto_route=True,
    )

    assert decision["problem_profile"]["problem_family_hint"] == "blending"
    assert decision["route_strategy"] == "optimization_model"


def test_batch_routes_uncertain_coefficients_as_robust_optimization():
    module = _load_batch_module()
    decision = module._case_routing_decision(
        {
            "problem_text": (
                "A plant produces products A and B with unit profits. The "
                "resource coefficients are uncertain under budget uncertainty "
                "with Gamma = 1. Maximize profit subject to resource capacity."
            ),
            "gold_solver": {"status": "optimal", "objective_value": 12.0},
        },
        workflow_mode="modeling_fast",
        solver_only_components=True,
        direct_answer_fallback=True,
        auto_route=True,
    )

    assert decision["problem_profile"]["problem_family_hint"] == "robust_optimization"
    assert decision["route_strategy"] == "optimization_model"


def test_batch_routes_table_capacity_space_mix_as_known_family():
    module = _load_batch_module()
    decision = module._case_routing_decision(
        {
            "problem_text": (
                "At circular tables, poster boards and participants can fit to "
                "cater guests. Rectangular tables use space differently. The "
                "fair must fit at least 500 participants and maximize catered "
                "guests under available space."
            ),
            "gold_solver": {"status": "optimal", "objective_value": 1080.0},
        },
        workflow_mode="modeling_fast",
        solver_only_components=True,
        direct_answer_fallback=True,
        auto_route=True,
    )

    assert decision["problem_profile"]["problem_family_hint"] == "table_capacity_space_mix"
    assert decision["route_strategy"] == "optimization_model"


def test_batch_routes_fixed_charge_machine_assignment_as_known_family():
    module = _load_batch_module()
    decision = module._case_routing_decision(
        {
            "problem_text": (
                "Several parts can be processed on machines A, B, or C with a "
                "unit processing cost table and a setup cost for every machine "
                "used. Assign each part to a machine to minimize total cost."
            ),
            "gold_solver": {"status": "optimal", "objective_value": 1005.0},
        },
        workflow_mode="modeling_fast",
        solver_only_components=True,
        direct_answer_fallback=True,
        auto_route=True,
    )

    assert (
        decision["problem_profile"]["problem_family_hint"]
        == "fixed_charge_machine_assignment"
    )
    assert decision["route_strategy"] == "optimization_model"


def test_batch_routes_multi_period_workforce_production_as_known_family():
    module = _load_batch_module()
    decision = module._case_routing_decision(
        {
            "problem_text": (
                "A company builds a six-month production plan with workforce "
                "hiring and firing, outsourcing, overtime labor, inventory, "
                "backorders, and terminal inventory requirements to maximize "
                "net profit."
            ),
            "gold_solver": {"status": "optimal", "objective_value": 10349920.0},
        },
        workflow_mode="modeling_fast",
        solver_only_components=True,
        direct_answer_fallback=True,
        auto_route=True,
    )

    assert (
        decision["problem_profile"]["problem_family_hint"]
        == "multi_period_workforce_production"
    )
    assert decision["route_strategy"] == "optimization_model"
