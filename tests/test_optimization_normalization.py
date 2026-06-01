import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from orbit_or import api
from orbit_or.db import init_db
from orbit_or.optimization import (
    build_indexed_model_ir_prompt,
    build_lp_artifact_from_components,
    build_lp_artifact_from_indexed_ir,
    extract_indexed_ir_candidate_tournament,
    parse_lp_artifact,
)


def test_lp_builder_normalizes_british_objective_and_ellipsis_ranges():
    components = [
        {
            "id": 1,
            "component_type": "decision_variable",
            "natural_text": "Quantity in period 1.",
            "symbol": "x_1",
            "source_refs_json": '["D1"]',
            "review_status": "reviewed",
        },
        {
            "id": 2,
            "component_type": "decision_variable",
            "natural_text": "Quantity in period 2.",
            "symbol": "x_2",
            "source_refs_json": '["D1"]',
            "review_status": "reviewed",
        },
        {
            "id": 3,
            "component_type": "decision_variable",
            "natural_text": "Quantity in period 3.",
            "symbol": "x_3",
            "source_refs_json": '["D1"]',
            "review_status": "reviewed",
        },
        {
            "id": 4,
            "component_type": "objective",
            "formal_text": "minimise 2 x_1 + ... + 2 x_3",
            "source_refs_json": '["D1"]',
            "review_status": "reviewed",
        },
        {
            "id": 5,
            "component_type": "constraint",
            "formal_text": "x_1 + \u2026 + x_3 >= 3",
            "source_refs_json": '["D1"]',
            "review_status": "reviewed",
        },
    ]

    generated = build_lp_artifact_from_components(components)
    parsed = parse_lp_artifact(generated["content"])

    assert generated["accepted"] is True
    assert "obj: 2 x_1 + 2 x_2 + 2 x_3" in generated["content"]
    assert "c1: x_1 + x_2 + x_3 >= 3" in generated["content"]
    assert parsed.direction == "minimize"
    assert parsed.variables == ("x_1", "x_2", "x_3")


def test_lp_builder_normalizes_unicode_linear_symbols():
    generated = build_lp_artifact_from_components(
        [
            {
                "id": 1,
                "component_type": "decision_variable",
                "natural_text": "Quantity.",
                "symbol": "x",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
            {
                "id": 2,
                "component_type": "objective",
                "formal_text": "minimise 2·x",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
            {
                "id": 3,
                "component_type": "constraint",
                "formal_text": "x ≤ 3",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
        ]
    )

    assert generated["accepted"] is True
    assert "obj: 2 x" in generated["content"]
    assert "c1: x <= 3" in generated["content"]


def test_lp_builder_reports_indexed_notation_before_lp_parse():
    generated = build_lp_artifact_from_components(
        [
            {
                "id": 1,
                "component_type": "decision_variable",
                "natural_text": "Quantity by week.",
                "symbol": "x_t",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
            {
                "id": 2,
                "component_type": "objective",
                "formal_text": "minimize Σ_{t∈T} x_t",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
            {
                "id": 3,
                "component_type": "constraint",
                "formal_text": "x_t >= 0",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
        ]
    )

    assert generated["accepted"] is False
    assert generated["issues"][0]["issue_type"] == (
        "indexed_formal_text_requires_scalarization"
    )


def test_lp_builder_rejects_unlinked_objective_variable():
    generated = build_lp_artifact_from_components(
        [
            {
                "id": 1,
                "component_type": "decision_variable",
                "natural_text": "Production quantity.",
                "symbol": "x",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
            {
                "id": 2,
                "component_type": "objective",
                "formal_text": "minimize total_cost",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
            {
                "id": 3,
                "component_type": "constraint",
                "formal_text": "x >= 10",
                "source_refs_json": '["D1"]',
                "review_status": "reviewed",
            },
        ]
    )

    assert generated["accepted"] is False
    assert generated["issues"][0]["issue_type"] == "objective_variable_not_linked"


def test_indexed_model_ir_prompt_requires_finite_structured_ir():
    prompt = build_indexed_model_ir_prompt(
        "Minimize production cost over three periods."
    )

    assert "orbit_indexed_model_ir.v1" in prompt
    assert "Use finite set values only" in prompt
    assert "Do not return LP text" in prompt


def test_indexed_ir_scalarizes_finite_horizon_lp():
    generated = build_lp_artifact_from_indexed_ir(
        {
            "sets": {"T": [1, 2, 3]},
            "parameters": {
                "d": {"values": {"1": 10, "2": 20, "3": 30}},
            },
            "variables": [
                {"symbol": "x", "indices": ["T"], "domain": "integer"},
            ],
            "objective": {
                "sense": "minimize",
                "terms": [
                    {
                        "coefficient": 2,
                        "variable": "x",
                        "subscripts": ["t"],
                        "sum_over": {"t": "T"},
                    }
                ],
            },
            "constraints": [
                {
                    "name": "demand",
                    "for_each": {"t": "T"},
                    "terms": [
                        {"coefficient": 1, "variable": "x", "subscripts": ["t"]}
                    ],
                    "sense": ">=",
                    "rhs": {"parameter": "d", "subscripts": ["t"]},
                }
            ],
        }
    )

    parsed = parse_lp_artifact(generated["content"])

    assert generated["accepted"] is True
    assert "obj: 2 x_1 + 2 x_2 + 2 x_3" in generated["content"]
    assert "demand_2: x_2 >= 20" in generated["content"]
    assert "General\n x_1 x_2 x_3" in generated["content"]
    assert parsed.variables == ("x_1", "x_2", "x_3")


def test_indexed_ir_scalarizes_prefix_sum_constraint():
    generated = build_lp_artifact_from_indexed_ir(
        {
            "sets": {"T": [1, 2, 3]},
            "variables": [
                {"symbol": "new", "indices": ["T"], "domain": "integer"},
            ],
            "objective": {
                "sense": "minimize",
                "terms": [
                    {
                        "coefficient": 1,
                        "variable": "new",
                        "subscripts": ["t"],
                        "sum_over": {"t": "T"},
                    }
                ],
            },
            "constraints": [
                {
                    "name": "prefix",
                    "for_each": {"t": "T"},
                    "terms": [
                        {
                            "coefficient": 1,
                            "variable": "new",
                            "subscripts": ["tau"],
                            "sum_over": {"tau": "T"},
                            "where": {"tau": {"<=": "t"}},
                        }
                    ],
                    "sense": ">=",
                    "rhs": "t",
                }
            ],
        }
    )

    assert generated["accepted"] is True
    assert "prefix_1: new_1 >= 1" in generated["content"]
    assert "prefix_2: new_1 + new_2 >= 2" in generated["content"]
    assert "prefix_3: new_1 + new_2 + new_3 >= 3" in generated["content"]


def test_indexed_ir_rejects_insufficient_data_status():
    generated = build_lp_artifact_from_indexed_ir(
        {
            "schema": "orbit_indexed_model_ir.v1",
            "status": "insufficient_data",
            "issues": [{"issue_type": "missing_data", "message": "Demand is absent."}],
        }
    )

    assert generated["accepted"] is False
    assert generated["issues"][0]["issue_type"] == "indexed_ir_incomplete"


def test_indexed_ir_scalarizes_nested_parameter_tables():
    generated = build_lp_artifact_from_indexed_ir(
        {
            "sets": {"T": [1, 2], "F": ["I", "II"]},
            "parameters": {
                "demand": {
                    "values": {
                        "1": {"I": 10, "II": 20},
                        "2": {"I": 30, "II": 40},
                    }
                },
                "capacity": {"max": 3},
            },
            "variables": [
                {"symbol": "x", "indices": ["T", "F"], "domain": "continuous"},
            ],
            "objective": {
                "sense": "minimize",
                "terms": [
                    {
                        "coefficient": 1,
                        "variable": "x",
                        "subscripts": ["t", "f"],
                        "sum_over": {"t": "T", "f": "F"},
                    }
                ],
            },
            "constraints": [
                {
                    "name": "demand",
                    "for_each": {"t": "T", "f": "F"},
                    "terms": [
                        {"coefficient": 1, "variable": "x", "subscripts": ["t", "f"]}
                    ],
                    "sense": ">=",
                    "rhs": {"parameter": "demand", "subscripts": ["t", "f"]},
                },
                {
                    "name": "capacity",
                    "terms": [
                        {"coefficient": 1, "variable": "x", "subscripts": [1, "I"]}
                    ],
                    "sense": "<=",
                    "rhs": {"parameter": "capacity", "subscripts": ["max"]},
                },
            ],
        }
    )

    assert generated["accepted"] is True
    assert "demand_2_II: x_2_II >= 40" in generated["content"]
    assert "capacity: x_1_I <= 3" in generated["content"]


def test_indexed_ir_rejects_unused_critical_parameters():
    generated = build_lp_artifact_from_indexed_ir(
        {
            "sets": {"T": [1, 2]},
            "parameters": {"new_workers_target": 50},
            "variables": [{"symbol": "x", "indices": ["T"], "domain": "integer"}],
            "objective": {
                "sense": "minimize",
                "terms": [
                    {
                        "coefficient": 1,
                        "variable": "x",
                        "subscripts": ["t"],
                        "sum_over": {"t": "T"},
                    }
                ],
            },
            "constraints": [
                {
                    "name": "nonzero",
                    "for_each": {"t": "T"},
                    "terms": [
                        {"coefficient": 1, "variable": "x", "subscripts": ["t"]}
                    ],
                    "sense": ">=",
                    "rhs": 1,
                }
            ],
        }
    )

    assert generated["accepted"] is False
    assert generated["issues"][0]["issue_type"] == "indexed_scalarization_error"
    assert "new_workers_target" in generated["issues"][0]["message"]


def test_extract_indexed_ir_candidate_tournament_persists_scalarized_lp(tmp_path):
    os.environ["ORBIT_DB_PATH"] = str(tmp_path / "orbit.db")
    init_db()
    topic_id = api.create_topic("Indexed", "Minimize period cost.")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="Indexed LP",
        source_text="Minimize cost over three periods while meeting demand.",
    )
    response = SimpleNamespace(
        text=json.dumps(
            {
                "schema": "orbit_indexed_model_ir.v1",
                "status": "complete",
                "sets": {"T": [1, 2, 3]},
                "parameters": {
                    "d": {"values": {"1": 10, "2": 20, "3": 30}},
                },
                "variables": [
                    {"symbol": "x", "indices": ["T"], "domain": "integer"},
                ],
                "objective": {
                    "sense": "minimize",
                    "terms": [
                        {
                            "coefficient": 2,
                            "variable": "x",
                            "subscripts": ["t"],
                            "sum_over": {"t": "T"},
                        }
                    ],
                },
                "constraints": [
                    {
                        "name": "demand",
                        "for_each": {"t": "T"},
                        "terms": [
                            {
                                "coefficient": 1,
                                "variable": "x",
                                "subscripts": ["t"],
                            }
                        ],
                        "sense": ">=",
                        "rhs": {"parameter": "d", "subscripts": ["t"]},
                    }
                ],
            }
        )
    )

    with patch("orbit_or.broker.llm_call", new=AsyncMock(return_value=response)):
        result = asyncio.run(
            extract_indexed_ir_candidate_tournament(
                topic_id=topic_id,
                problem_id=problem_id,
                source_text="Minimize cost over three periods while meeting demand.",
            )
        )

    artifacts = api.get_optimization_artifacts(problem_id)

    assert result["artifact_id"] == artifacts[0]["id"]
    assert result["best"]["modeling_error"] == "none"
    assert artifacts[0]["parser_status"] == "valid"
    assert "demand_2: x_2 >= 20" in artifacts[0]["content"]


def test_extract_indexed_ir_candidate_tournament_repairs_invalid_ir(tmp_path):
    os.environ["ORBIT_DB_PATH"] = str(tmp_path / "orbit.db")
    init_db()
    topic_id = api.create_topic("Repair indexed", "Meet target.")
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        title="Repair indexed LP",
        source_text="Minimize hires while meeting a target of 2 workers.",
    )
    invalid_ir = {
        "schema": "orbit_indexed_model_ir.v1",
        "status": "complete",
        "sets": {"T": [1]},
        "parameters": {"new_workers_target": 2},
        "variables": [{"symbol": "x", "indices": ["T"], "domain": "integer"}],
        "objective": {
            "sense": "minimize",
            "terms": [
                {
                    "coefficient": 1,
                    "variable": "x",
                    "subscripts": ["t"],
                    "sum_over": {"t": "T"},
                }
            ],
        },
        "constraints": [
            {
                "name": "nonzero",
                "for_each": {"t": "T"},
                "terms": [{"coefficient": 1, "variable": "x", "subscripts": ["t"]}],
                "sense": ">=",
                "rhs": 1,
            }
        ],
    }
    repaired_ir = {
        **invalid_ir,
        "constraints": [
            {
                "name": "target",
                "terms": [
                    {
                        "coefficient": 1,
                        "variable": "x",
                        "subscripts": ["t"],
                        "sum_over": {"t": "T"},
                    }
                ],
                "sense": ">=",
                "rhs": {"parameter": "new_workers_target", "subscripts": []},
            }
        ],
    }
    responses = [
        SimpleNamespace(text=json.dumps(invalid_ir)),
        SimpleNamespace(text=json.dumps(repaired_ir)),
    ]

    with patch("orbit_or.broker.llm_call", new=AsyncMock(side_effect=responses)) as call:
        result = asyncio.run(
            extract_indexed_ir_candidate_tournament(
                topic_id=topic_id,
                problem_id=problem_id,
                source_text="Minimize hires while meeting a target of 2 workers.",
                candidate_count=1,
            )
        )

    artifacts = api.get_optimization_artifacts(problem_id)

    assert call.await_count == 2
    assert result["best"]["repair_attempt"] is True
    assert artifacts[0]["parser_status"] == "valid"
    assert "target: x_1 >= 2" in artifacts[0]["content"]
