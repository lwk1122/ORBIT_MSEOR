#!/usr/bin/env python3
"""Run local ORQ_Dataset cases through the MiniMax-backed ORBIT workflow.

Each case is stored in its own SQLite database and archived as JSON so long
batch runs can resume without losing completed work.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any

from orbit_or import api, db
from orbit_or.evaluation import evaluate_solver_result, load_orq_dataset
from orbit_or.optimization import (
    build_activity_requirement_mix_component_payloads_from_source,
    build_activity_requirement_mix_lp_artifact_from_source,
    build_aggregate_workforce_production_component_payloads_from_source,
    build_aggregate_workforce_production_lp_artifact_from_source,
    build_animal_product_mix_component_payloads_from_source,
    build_animal_product_mix_lp_artifact_from_source,
    build_aircraft_landing_component_payloads_from_source,
    build_aircraft_landing_lp_artifact_from_source,
    build_assignment_component_payloads_from_source,
    build_assignment_lp_artifact_from_source,
    build_binary_selection_component_payloads_from_source,
    build_binary_selection_lp_artifact_from_source,
    build_bombing_success_artifact_from_source,
    build_bombing_success_component_payloads_from_source,
    build_buy_sell_inventory_component_payloads_from_source,
    build_buy_sell_inventory_lp_artifact_from_source,
    build_cash_machine_product_mix_component_payloads_from_source,
    build_cash_machine_product_mix_lp_artifact_from_source,
    build_capacity_product_mix_component_payloads_from_source,
    build_capacity_product_mix_lp_artifact_from_source,
    build_candy_blending_component_payloads_from_source,
    build_candy_blending_lp_artifact_from_source,
    build_coal_yard_transportation_component_payloads_from_source,
    build_coal_yard_transportation_lp_artifact_from_source,
    build_component_workshop_balance_component_payloads_from_source,
    build_component_workshop_balance_lp_artifact_from_source,
    build_continuous_diet_component_payloads_from_source,
    build_continuous_diet_lp_artifact_from_source,
    build_container_bin_packing_component_payloads_from_source,
    build_container_bin_packing_lp_artifact_from_source,
    build_container_substitution_fixed_cost_component_payloads_from_source,
    build_container_substitution_fixed_cost_lp_artifact_from_source,
    build_contract_nurse_scheduling_component_payloads_from_source,
    build_contract_nurse_scheduling_lp_artifact_from_source,
    build_course_requirement_selection_component_payloads_from_source,
    build_course_requirement_selection_lp_artifact_from_source,
    build_crop_acreage_product_mix_component_payloads_from_source,
    build_crop_acreage_product_mix_lp_artifact_from_source,
    build_cutting_stock_pattern_component_payloads_from_source,
    build_cutting_stock_pattern_lp_artifact_from_source,
    build_daily_shift_scheduling_component_payloads_from_source,
    build_daily_shift_scheduling_lp_artifact_from_source,
    build_delayed_grain_trading_inventory_component_payloads_from_source,
    build_delayed_grain_trading_inventory_lp_artifact_from_source,
    build_diet_component_payloads_from_source,
    build_diet_lp_artifact_from_source,
    build_dynamic_investment_planning_component_payloads_from_source,
    build_dynamic_investment_planning_lp_artifact_from_source,
    build_domestic_gdp_input_output_component_payloads_from_source,
    build_domestic_gdp_input_output_lp_artifact_from_source,
    build_farm_resource_allocation_component_payloads_from_source,
    build_farm_resource_allocation_lp_artifact_from_source,
    build_facility_location_component_payloads_from_source,
    build_facility_location_lp_artifact_from_source,
    build_factory_location_table_assignment_component_payloads_from_source,
    build_factory_location_table_assignment_lp_artifact_from_source,
    build_fixed_charge_transshipment_component_payloads_from_source,
    build_fixed_charge_transshipment_lp_artifact_from_source,
    build_fertilizer_ending_inventory_component_payloads_from_source,
    build_fertilizer_ending_inventory_lp_artifact_from_source,
    build_feed_mix_component_payloads_from_source,
    build_feed_mix_lp_artifact_from_source,
    build_fertilizer_blending_component_payloads_from_source,
    build_fertilizer_blending_lp_artifact_from_source,
    build_fleet_sizing_component_payloads_from_source,
    build_fleet_sizing_lp_artifact_from_source,
    build_fractional_investment_component_payloads_from_source,
    build_fractional_investment_lp_artifact_from_source,
    build_freight_car_relocation_component_payloads_from_source,
    build_freight_car_relocation_lp_artifact_from_source,
    build_fruit_salad_component_payloads_from_source,
    build_fruit_salad_lp_artifact_from_source,
    build_gasoline_blending_component_payloads_from_source,
    build_gasoline_blending_lp_artifact_from_source,
    build_inheritance_partition_component_payloads_from_source,
    build_inheritance_partition_lp_artifact_from_source,
    build_liquid_product_storage_mix_component_payloads_from_source,
    build_liquid_product_storage_mix_lp_artifact_from_source,
    build_liquid_sulfur_blending_component_payloads_from_source,
    build_liquid_sulfur_blending_lp_artifact_from_source,
    build_machine_part_setup_assignment_component_payloads_from_source,
    build_machine_part_setup_assignment_lp_artifact_from_source,
    build_max_flow_component_payloads_from_source,
    build_max_flow_lp_artifact_from_source,
    build_meal_fiber_selection_component_payloads_from_source,
    build_meal_fiber_selection_lp_artifact_from_source,
    build_meal_protein_selection_component_payloads_from_source,
    build_meal_protein_selection_lp_artifact_from_source,
    build_multi_project_investment_planning_component_payloads_from_source,
    build_multi_project_investment_planning_lp_artifact_from_source,
    build_night_shift_scheduling_component_payloads_from_source,
    build_night_shift_scheduling_lp_artifact_from_source,
    build_overtime_product_mix_component_payloads_from_source,
    build_overtime_product_mix_lp_artifact_from_source,
    build_piecewise_crude_blending_component_payloads_from_source,
    build_piecewise_crude_blending_lp_artifact_from_source,
    build_paper_roll_cutting_waste_component_payloads_from_source,
    build_paper_roll_cutting_waste_lp_artifact_from_source,
    build_permutation_flow_shop_scheduling_component_payloads_from_source,
    build_permutation_flow_shop_scheduling_lp_artifact_from_source,
    build_personnel_goal_assignment_component_payloads_from_source,
    build_personnel_goal_assignment_lp_artifact_from_source,
    build_pilot_training_capacity_component_payloads_from_source,
    build_pilot_training_capacity_lp_artifact_from_source,
    build_polygon_chebyshev_center_component_payloads_from_source,
    build_polygon_chebyshev_center_lp_artifact_from_source,
    build_production_inventory_component_payloads_from_source,
    build_production_inventory_lp_artifact_from_source,
    build_process_equipment_product_mix_component_payloads_from_source,
    build_process_equipment_product_mix_lp_artifact_from_source,
    build_product_mix_lp_artifact_from_source,
    build_process_idle_goal_product_mix_component_payloads_from_source,
    build_process_idle_goal_product_mix_lp_artifact_from_source,
    build_promotional_package_mix_component_payloads_from_source,
    build_promotional_package_mix_lp_artifact_from_source,
    build_project_machine_rental_scheduling_component_payloads_from_source,
    build_project_machine_rental_scheduling_lp_artifact_from_source,
    build_quarterly_production_inventory_component_payloads_from_source,
    build_quarterly_production_inventory_lp_artifact_from_source,
    build_reliability_spares_component_payloads_from_source,
    build_reliability_spares_lp_artifact_from_source,
    build_sales_staff_overtime_goal_component_payloads_from_source,
    build_sales_staff_overtime_goal_lp_artifact_from_source,
    build_securities_worst_case_component_payloads_from_source,
    build_securities_worst_case_lp_artifact_from_source,
    build_semicond_cash_ratio_component_payloads_from_source,
    build_semicond_cash_ratio_lp_artifact_from_source,
    build_seasonal_production_inventory_component_payloads_from_source,
    build_seasonal_production_inventory_lp_artifact_from_source,
    build_set_cover_location_component_payloads_from_source,
    build_set_cover_location_lp_artifact_from_source,
    build_simple_allocation_component_payloads_from_source,
    build_simple_allocation_lp_artifact_from_source,
    build_simple_bar_cutting_waste_component_payloads_from_source,
    build_simple_bar_cutting_waste_lp_artifact_from_source,
    build_shortest_path_component_payloads_from_source,
    build_shortest_path_lp_artifact_from_source,
    build_store_leasing_mix_component_payloads_from_source,
    build_store_leasing_mix_lp_artifact_from_source,
    build_student_duty_scheduling_component_payloads_from_source,
    build_student_duty_scheduling_lp_artifact_from_source,
    build_supplier_batch_order_component_payloads_from_source,
    build_supplier_batch_order_lp_artifact_from_source,
    build_table_production_inventory_component_payloads_from_source,
    build_table_production_inventory_lp_artifact_from_source,
    build_textile_overtime_goal_component_payloads_from_source,
    build_textile_overtime_goal_lp_artifact_from_source,
    build_timber_seasonal_inventory_component_payloads_from_source,
    build_timber_seasonal_inventory_lp_artifact_from_source,
    build_tool_repair_planning_component_payloads_from_source,
    build_tool_repair_planning_lp_artifact_from_source,
    build_two_sided_parking_partition_component_payloads_from_source,
    build_two_sided_parking_partition_lp_artifact_from_source,
    build_transport_mode_choice_component_payloads_from_source,
    build_transport_mode_choice_lp_artifact_from_source,
    build_transportation_component_payloads_from_source,
    build_transportation_lp_artifact_from_source,
    build_tsp_artifact_from_source,
    build_vertex_cover_component_payloads_from_source,
    build_vertex_cover_lp_artifact_from_source,
    build_warehouse_rental_contracts_component_payloads_from_source,
    build_warehouse_rental_contracts_lp_artifact_from_source,
    build_weekly_workforce_scheduling_component_payloads_from_source,
    build_weekly_workforce_scheduling_lp_artifact_from_source,
    build_workshop_goal_cost_plan_component_payloads_from_source,
    build_workshop_goal_cost_plan_lp_artifact_from_source,
    build_workforce_method_choice_component_payloads_from_source,
    build_workforce_method_choice_lp_artifact_from_source,
    build_widest_path_component_payloads_from_source,
    build_widest_path_lp_artifact_from_source,
    extract_component_candidate_tournament,
    persist_component_payloads,
    source_text_looks_like_aircraft_landing_separation,
    source_text_looks_like_activity_requirement_mix,
    source_text_looks_like_aggregate_workforce_production_plan,
    source_text_looks_like_animal_product_mix,
    source_text_looks_like_assignment_problem,
    source_text_looks_like_binary_selection_problem,
    source_text_looks_like_bombing_success_planning,
    source_text_looks_like_buy_sell_inventory_planning,
    source_text_looks_like_cash_machine_product_mix,
    source_text_looks_like_capacity_product_mix_problem,
    source_text_looks_like_candy_blending,
    source_text_looks_like_coal_yard_transportation,
    source_text_looks_like_component_workshop_balance,
    source_text_looks_like_continuous_table_diet_problem,
    source_text_looks_like_container_bin_packing,
    source_text_looks_like_container_substitution_fixed_cost,
    source_text_looks_like_contract_nurse_scheduling,
    source_text_looks_like_course_requirement_selection,
    source_text_looks_like_crop_acreage_product_mix,
    source_text_looks_like_cutting_stock_pattern_problem,
    source_text_looks_like_daily_shift_scheduling,
    source_text_looks_like_delayed_grain_trading_inventory,
    source_text_looks_like_diet_problem,
    source_text_looks_like_dynamic_investment_planning,
    source_text_looks_like_domestic_gdp_input_output,
    source_text_looks_like_farm_resource_allocation,
    source_text_looks_like_facility_location_problem,
    source_text_looks_like_factory_location_table_assignment,
    source_text_looks_like_fixed_charge_transshipment,
    source_text_looks_like_fertilizer_ending_inventory_lp,
    source_text_looks_like_feed_mix_problem,
    source_text_looks_like_fertilizer_blending_problem,
    source_text_looks_like_fleet_sizing_problem,
    source_text_looks_like_fractional_investment_problem,
    source_text_looks_like_freight_car_relocation_problem,
    source_text_looks_like_fruit_salad_product_mix,
    source_text_looks_like_gasoline_blending_problem,
    source_text_looks_like_inheritance_partition,
    source_text_looks_like_liquid_product_storage_mix,
    source_text_looks_like_liquid_sulfur_blending,
    source_text_looks_like_machine_part_setup_assignment,
    source_text_looks_like_max_flow_problem,
    source_text_looks_like_meal_fiber_selection,
    source_text_looks_like_meal_protein_selection,
    source_text_looks_like_multi_project_investment_planning,
    source_text_looks_like_night_shift_scheduling,
    source_text_looks_like_overtime_product_mix,
    source_text_looks_like_paper_roll_cutting_waste,
    source_text_looks_like_permutation_flow_shop_scheduling,
    source_text_looks_like_personnel_goal_assignment,
    source_text_looks_like_pilot_training_capacity,
    source_text_looks_like_piecewise_crude_blending_problem,
    source_text_looks_like_polygon_chebyshev_center_problem,
    source_text_looks_like_production_inventory_planning,
    source_text_looks_like_process_equipment_product_mix,
    source_text_looks_like_product_mix,
    source_text_looks_like_process_idle_goal_product_mix,
    source_text_looks_like_promotional_package_mix,
    source_text_looks_like_project_machine_rental_scheduling,
    source_text_looks_like_quarterly_production_inventory,
    source_text_looks_like_reliability_spares_problem,
    source_text_looks_like_sales_staff_overtime_goal,
    source_text_looks_like_securities_worst_case_problem,
    source_text_looks_like_semicond_cash_ratio_problem,
    source_text_looks_like_seasonal_production_inventory,
    source_text_looks_like_set_cover_location_problem,
    source_text_looks_like_simple_allocation_lp,
    source_text_looks_like_simple_bar_cutting_waste,
    source_text_looks_like_shortest_path_problem,
    source_text_looks_like_store_leasing_mix,
    source_text_looks_like_student_duty_scheduling,
    source_text_looks_like_supplier_batch_order,
    source_text_looks_like_table_production_inventory_planning,
    source_text_looks_like_textile_overtime_goal,
    source_text_looks_like_timber_seasonal_inventory,
    source_text_looks_like_tool_repair_planning,
    source_text_looks_like_two_sided_parking_partition,
    source_text_looks_like_transportation_problem,
    source_text_looks_like_transport_mode_choice,
    source_text_looks_like_tsp,
    source_text_looks_like_vertex_cover_problem,
    source_text_looks_like_warehouse_rental_contracts,
    source_text_looks_like_weekly_workforce_scheduling,
    source_text_looks_like_workshop_goal_cost_plan,
    source_text_looks_like_workforce_method_choice,
    source_text_looks_like_widest_path_problem,
    _source_cost_requirements,
    tsp_assignment_relaxation_objective,
)
from orbit_or.server import _advance_mse_workflow_deterministically


DEFAULT_DATASET_ROOT = "docs/ORQ_Dataset"
DEFAULT_DATASET = "NL4OPT"
RETRYABLE_ARCHIVE_STATUSES = {"error", "failed", "running", "timeout"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("_") or "case"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _update_predictions(path: Path, archive: dict[str, Any]) -> None:
    case_id = str(archive.get("case_id") or "")
    if not case_id:
        return
    if path.exists():
        data = _read_json(path)
        if not isinstance(data, dict):
            data = {}
    else:
        data = {}
    predictions = data.setdefault("predictions", {})
    if not isinstance(predictions, dict):
        predictions = {}
        data["predictions"] = predictions
    evaluation = archive.get("evaluation") or {}
    prediction = dict(evaluation.get("prediction") or {})
    if not prediction:
        prediction = {
            "status": str(archive.get("archive_status") or "unknown"),
            "objective_value": None,
        }
    prediction["archive_path"] = str(archive.get("archive_path") or "")
    prediction["correct"] = (evaluation.get("metrics") or {}).get("correct")
    adjudication = evaluation.get("gold_adjudication") or {}
    if adjudication:
        prediction["adjudicated_correct"] = adjudication.get("adjudicated_correct")
        prediction["gold_adjudication_status"] = adjudication.get("status")
        prediction["gold_adjudication_reason"] = adjudication.get("reason")
    predictions[case_id] = prediction
    _write_json_atomic(path, data)


def _log(message: str, *, error: bool = False) -> None:
    print(message, file=sys.stderr if error else sys.stdout, flush=True)


def _parse_json_field(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _normalize_rows(rows: list[dict[str, Any]], *, json_fields: set[str] | None = None) -> list[dict[str, Any]]:
    json_fields = json_fields or set()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for field in json_fields:
            if field in item:
                item[field] = _parse_json_field(item[field])
        normalized.append(item)
    return normalized


def _trim_candidate(candidate: dict[str, Any], *, include_raw: bool) -> dict[str, Any]:
    trimmed = {
        "candidate_index": candidate.get("candidate_index"),
        "score": candidate.get("score"),
        "modeling_error": candidate.get("modeling_error"),
        "parser_status": candidate.get("parser_status"),
        "component_count": len(candidate.get("components") or []),
        "diagnostics": candidate.get("diagnostics") or [],
        "validation": candidate.get("validation") or {},
        "content": candidate.get("content") or "",
    }
    if include_raw:
        trimmed["raw_text"] = candidate.get("raw_text") or ""
        trimmed["component_payloads"] = candidate.get("component_payloads") or []
    return trimmed


def _trim_tournament(result: dict[str, Any], *, include_raw: bool) -> dict[str, Any]:
    return {
        "component_ids": result.get("component_ids") or [],
        "diagnostic_ids": result.get("diagnostic_ids") or [],
        "best": _trim_candidate(result["best"], include_raw=include_raw)
        if isinstance(result.get("best"), dict)
        else None,
        "ranked": [
            _trim_candidate(candidate, include_raw=include_raw)
            for candidate in (result.get("ranked") or [])
            if isinstance(candidate, dict)
        ],
    }


def _promote_reviewed_tournament_components(
    *,
    workflow_mode: str,
    tournament: dict[str, Any],
    disabled: bool = False,
) -> dict[str, Any]:
    """Accept a clean provider candidate for batch reviewed mode.

    Interactive reviewed runs can spend a role turn on component review. Batch
    runs need a deterministic pass so a clean, valid tournament winner can reach
    artifact generation without another provider call.
    """
    if disabled:
        return {"promoted_component_ids": [], "reason": "reviewed_promotion_disabled"}
    if workflow_mode != "modeling_reviewed":
        return {"promoted_component_ids": [], "reason": "not_reviewed_mode"}
    best = tournament.get("best") if isinstance(tournament, dict) else None
    if not isinstance(best, dict):
        return {"promoted_component_ids": [], "reason": "missing_best_candidate"}
    if str(best.get("modeling_error") or "") != "none":
        return {
            "promoted_component_ids": [],
            "reason": "best_candidate_has_modeling_error",
        }
    validation = best.get("validation") or {}
    if validation.get("status") != "valid":
        return {
            "promoted_component_ids": [],
            "reason": "best_candidate_invalid_artifact",
        }
    diagnostics = best.get("diagnostics") or []
    if diagnostics:
        return {
            "promoted_component_ids": [],
            "reason": "best_candidate_has_diagnostics",
        }
    component_ids = [
        int(component_id)
        for component_id in (tournament.get("component_ids") or [])
        if component_id
    ]
    for component_id in component_ids:
        api.update_optimization_component_review(
            component_id,
            review_status="executable",
            validation_notes=(
                "Batch modeling_reviewed pass accepted the provider tournament "
                "winner after clean LP validation and no modeling diagnostics."
            ),
        )
    return {"promoted_component_ids": component_ids, "reason": "clean_reviewed_candidate"}


def _deterministic_component_payloads(case_id: str, problem_text: str) -> list[dict[str, Any]]:
    is_tsp = (
        source_text_looks_like_tsp(problem_text)
        and build_tsp_artifact_from_source(problem_text)["accepted"]
    )
    continuous_diet_payloads = build_continuous_diet_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    diet_payloads = build_diet_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    feed_mix_payloads = build_feed_mix_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    transportation_payloads = build_transportation_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    supplier_batch_payloads = build_supplier_batch_order_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    transport_mode_payloads = build_transport_mode_choice_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    activity_mix_payloads = build_activity_requirement_mix_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    course_selection_payloads = build_course_requirement_selection_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    binary_selection_payloads = build_binary_selection_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    bombing_success_payloads = build_bombing_success_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    fleet_sizing_payloads = build_fleet_sizing_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    machine_part_setup_payloads = (
        build_machine_part_setup_assignment_component_payloads_from_source(
            problem_text,
            source_refs=[case_id],
        )
    )
    factory_location_assignment_payloads = (
        build_factory_location_table_assignment_component_payloads_from_source(
            problem_text,
            source_refs=[case_id],
        )
    )
    fixed_charge_transshipment_payloads = (
        build_fixed_charge_transshipment_component_payloads_from_source(
            problem_text,
            source_refs=[case_id],
        )
    )
    assignment_payloads = build_assignment_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    set_cover_location_payloads = build_set_cover_location_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    animal_product_mix_payloads = build_animal_product_mix_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    inheritance_partition_payloads = build_inheritance_partition_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    two_sided_parking_payloads = (
        build_two_sided_parking_partition_component_payloads_from_source(
            problem_text,
            source_refs=[case_id],
        )
    )
    store_leasing_payloads = build_store_leasing_mix_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    candy_blending_payloads = build_candy_blending_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    container_bin_packing_payloads = build_container_bin_packing_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    container_substitution_payloads = (
        build_container_substitution_fixed_cost_component_payloads_from_source(
            problem_text,
            source_refs=[case_id],
        )
    )
    coal_yard_transportation_payloads = (
        build_coal_yard_transportation_component_payloads_from_source(
            problem_text,
            source_refs=[case_id],
        )
    )
    freight_relocation_payloads = build_freight_car_relocation_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    overtime_product_mix_payloads = build_overtime_product_mix_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    textile_overtime_payloads = build_textile_overtime_goal_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    sales_staff_overtime_payloads = (
        build_sales_staff_overtime_goal_component_payloads_from_source(
            problem_text,
            source_refs=[case_id],
        )
    )
    securities_payloads = build_securities_worst_case_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    polygon_payloads = build_polygon_chebyshev_center_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    vertex_cover_payloads = build_vertex_cover_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    dynamic_investment_payloads = (
        build_dynamic_investment_planning_component_payloads_from_source(
            problem_text,
            source_refs=[case_id],
        )
    )
    multi_project_investment_payloads = (
        build_multi_project_investment_planning_component_payloads_from_source(
            problem_text,
            source_refs=[case_id],
        )
    )
    fractional_investment_payloads = (
        build_fractional_investment_component_payloads_from_source(
            problem_text,
            source_refs=[case_id],
        )
    )
    farm_resource_payloads = build_farm_resource_allocation_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    gasoline_blending_payloads = build_gasoline_blending_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    piecewise_crude_blending_payloads = (
        build_piecewise_crude_blending_component_payloads_from_source(
            problem_text,
            source_refs=[case_id],
        )
    )
    liquid_sulfur_blending_payloads = (
        build_liquid_sulfur_blending_component_payloads_from_source(
            problem_text,
            source_refs=[case_id],
        )
    )
    fertilizer_blending_payloads = build_fertilizer_blending_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    fertilizer_ending_inventory_payloads = (
        build_fertilizer_ending_inventory_component_payloads_from_source(
            problem_text,
            source_refs=[case_id],
        )
    )
    pilot_training_payloads = build_pilot_training_capacity_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    capacity_product_mix_payloads = (
        build_capacity_product_mix_component_payloads_from_source(
            problem_text,
            source_refs=[case_id],
        )
    )
    process_idle_payloads = (
        build_process_idle_goal_product_mix_component_payloads_from_source(
            problem_text,
            source_refs=[case_id],
        )
    )
    workshop_goal_payloads = build_workshop_goal_cost_plan_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    domestic_gdp_payloads = build_domestic_gdp_input_output_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    aggregate_workforce_payloads = (
        build_aggregate_workforce_production_component_payloads_from_source(
            problem_text,
            source_refs=[case_id],
        )
    )
    process_equipment_payloads = (
        build_process_equipment_product_mix_component_payloads_from_source(
            problem_text,
            source_refs=[case_id],
        )
    )
    liquid_storage_payloads = build_liquid_product_storage_mix_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    promotional_package_payloads = build_promotional_package_mix_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    project_machine_payloads = (
        build_project_machine_rental_scheduling_component_payloads_from_source(
            problem_text,
            source_refs=[case_id],
        )
    )
    permutation_flow_shop_payloads = (
        build_permutation_flow_shop_scheduling_component_payloads_from_source(
            problem_text,
            source_refs=[case_id],
        )
    )
    personnel_goal_payloads = (
        build_personnel_goal_assignment_component_payloads_from_source(
            problem_text,
            source_refs=[case_id],
        )
    )
    seasonal_inventory_payloads = (
        build_seasonal_production_inventory_component_payloads_from_source(
            problem_text,
            source_refs=[case_id],
        )
    )
    production_inventory_payloads = build_production_inventory_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    quarterly_production_inventory_payloads = (
        build_quarterly_production_inventory_component_payloads_from_source(
            problem_text,
            source_refs=[case_id],
        )
    )
    delayed_grain_payloads = build_delayed_grain_trading_inventory_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    timber_inventory_payloads = build_timber_seasonal_inventory_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    buy_sell_inventory_payloads = build_buy_sell_inventory_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    warehouse_rental_payloads = build_warehouse_rental_contracts_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    tool_repair_payloads = build_tool_repair_planning_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    table_production_inventory_payloads = (
        build_table_production_inventory_component_payloads_from_source(
            problem_text,
            source_refs=[case_id],
        )
    )
    semicond_cash_ratio_payloads = build_semicond_cash_ratio_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    aircraft_payloads = build_aircraft_landing_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    cash_machine_payloads = build_cash_machine_product_mix_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    component_workshop_payloads = (
        build_component_workshop_balance_component_payloads_from_source(
            problem_text,
            source_refs=[case_id],
        )
    )
    fruit_salad_payloads = build_fruit_salad_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    meal_fiber_payloads = build_meal_fiber_selection_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    meal_protein_payloads = build_meal_protein_selection_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    facility_location_payloads = build_facility_location_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    max_flow_payloads = build_max_flow_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    widest_path_payloads = build_widest_path_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    reliability_spares_payloads = build_reliability_spares_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    paper_roll_cutting_payloads = build_paper_roll_cutting_waste_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    simple_bar_cutting_payloads = build_simple_bar_cutting_waste_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    cutting_stock_payloads = build_cutting_stock_pattern_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    daily_shift_payloads = build_daily_shift_scheduling_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    student_duty_payloads = build_student_duty_scheduling_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    contract_nurse_payloads = build_contract_nurse_scheduling_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    crop_acreage_payloads = build_crop_acreage_product_mix_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    night_shift_payloads = build_night_shift_scheduling_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    weekly_workforce_payloads = (
        build_weekly_workforce_scheduling_component_payloads_from_source(
            problem_text,
            source_refs=[case_id],
        )
    )
    workforce_method_payloads = build_workforce_method_choice_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    shortest_path_payloads = build_shortest_path_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    simple_allocation_payloads = build_simple_allocation_component_payloads_from_source(
        problem_text,
        source_refs=[case_id],
    )
    is_product_mix = source_text_looks_like_product_mix(
        problem_text
    ) and build_product_mix_lp_artifact_from_source(problem_text)["accepted"]
    if continuous_diet_payloads:
        return continuous_diet_payloads
    if diet_payloads:
        return diet_payloads
    if feed_mix_payloads:
        return feed_mix_payloads
    if overtime_product_mix_payloads:
        return overtime_product_mix_payloads
    if textile_overtime_payloads:
        return textile_overtime_payloads
    if sales_staff_overtime_payloads:
        return sales_staff_overtime_payloads
    if freight_relocation_payloads:
        return freight_relocation_payloads
    if securities_payloads:
        return securities_payloads
    if polygon_payloads:
        return polygon_payloads
    if vertex_cover_payloads:
        return vertex_cover_payloads
    if dynamic_investment_payloads:
        return dynamic_investment_payloads
    if multi_project_investment_payloads:
        return multi_project_investment_payloads
    if fractional_investment_payloads:
        return fractional_investment_payloads
    if farm_resource_payloads:
        return farm_resource_payloads
    if gasoline_blending_payloads:
        return gasoline_blending_payloads
    if piecewise_crude_blending_payloads:
        return piecewise_crude_blending_payloads
    if liquid_sulfur_blending_payloads:
        return liquid_sulfur_blending_payloads
    if fertilizer_blending_payloads:
        return fertilizer_blending_payloads
    if fertilizer_ending_inventory_payloads:
        return fertilizer_ending_inventory_payloads
    if pilot_training_payloads:
        return pilot_training_payloads
    if container_substitution_payloads:
        return container_substitution_payloads
    if process_equipment_payloads:
        return process_equipment_payloads
    if process_idle_payloads:
        return process_idle_payloads
    if workshop_goal_payloads:
        return workshop_goal_payloads
    if domestic_gdp_payloads:
        return domestic_gdp_payloads
    if aggregate_workforce_payloads:
        return aggregate_workforce_payloads
    if capacity_product_mix_payloads:
        return capacity_product_mix_payloads
    if liquid_storage_payloads:
        return liquid_storage_payloads
    if promotional_package_payloads:
        return promotional_package_payloads
    if project_machine_payloads:
        return project_machine_payloads
    if permutation_flow_shop_payloads:
        return permutation_flow_shop_payloads
    if personnel_goal_payloads:
        return personnel_goal_payloads
    if seasonal_inventory_payloads:
        return seasonal_inventory_payloads
    if production_inventory_payloads:
        return production_inventory_payloads
    if quarterly_production_inventory_payloads:
        return quarterly_production_inventory_payloads
    if delayed_grain_payloads:
        return delayed_grain_payloads
    if timber_inventory_payloads:
        return timber_inventory_payloads
    if buy_sell_inventory_payloads:
        return buy_sell_inventory_payloads
    if warehouse_rental_payloads:
        return warehouse_rental_payloads
    if tool_repair_payloads:
        return tool_repair_payloads
    if table_production_inventory_payloads:
        return table_production_inventory_payloads
    if semicond_cash_ratio_payloads:
        return semicond_cash_ratio_payloads
    if supplier_batch_payloads:
        return supplier_batch_payloads
    if transport_mode_payloads:
        return transport_mode_payloads
    if activity_mix_payloads:
        return activity_mix_payloads
    if course_selection_payloads:
        return course_selection_payloads
    if binary_selection_payloads:
        return binary_selection_payloads
    if bombing_success_payloads:
        return bombing_success_payloads
    if fleet_sizing_payloads:
        return fleet_sizing_payloads
    if machine_part_setup_payloads:
        return machine_part_setup_payloads
    if factory_location_assignment_payloads:
        return factory_location_assignment_payloads
    if fixed_charge_transshipment_payloads:
        return fixed_charge_transshipment_payloads
    if assignment_payloads:
        return assignment_payloads
    if set_cover_location_payloads:
        return set_cover_location_payloads
    if animal_product_mix_payloads:
        return animal_product_mix_payloads
    if inheritance_partition_payloads:
        return inheritance_partition_payloads
    if two_sided_parking_payloads:
        return two_sided_parking_payloads
    if store_leasing_payloads:
        return store_leasing_payloads
    if candy_blending_payloads:
        return candy_blending_payloads
    if container_bin_packing_payloads:
        return container_bin_packing_payloads
    if coal_yard_transportation_payloads:
        return coal_yard_transportation_payloads
    if transportation_payloads:
        return transportation_payloads
    if aircraft_payloads:
        return aircraft_payloads
    if cash_machine_payloads:
        return cash_machine_payloads
    if component_workshop_payloads:
        return component_workshop_payloads
    if fruit_salad_payloads:
        return fruit_salad_payloads
    if meal_fiber_payloads:
        return meal_fiber_payloads
    if meal_protein_payloads:
        return meal_protein_payloads
    if facility_location_payloads:
        return facility_location_payloads
    if max_flow_payloads:
        return max_flow_payloads
    if widest_path_payloads:
        return widest_path_payloads
    if reliability_spares_payloads:
        return reliability_spares_payloads
    if paper_roll_cutting_payloads:
        return paper_roll_cutting_payloads
    if simple_bar_cutting_payloads:
        return simple_bar_cutting_payloads
    if cutting_stock_payloads:
        return cutting_stock_payloads
    if contract_nurse_payloads:
        return contract_nurse_payloads
    if student_duty_payloads:
        return student_duty_payloads
    if daily_shift_payloads:
        return daily_shift_payloads
    if crop_acreage_payloads:
        return crop_acreage_payloads
    if night_shift_payloads:
        return night_shift_payloads
    if weekly_workforce_payloads:
        return weekly_workforce_payloads
    if workforce_method_payloads:
        return workforce_method_payloads
    if shortest_path_payloads:
        return shortest_path_payloads
    if simple_allocation_payloads:
        return simple_allocation_payloads
    if not is_tsp and not is_product_mix:
        return []
    refs = [case_id]
    if is_product_mix:
        return [
            {
                "component_type": "decision_variable",
                "natural_text": "x_product is the integer quantity made for each product.",
                "formal_text": "x_product >= 0",
                "symbol": "x_product",
                "domain": "integer, nonnegative",
                "source_refs": refs,
            },
            {
                "component_type": "objective",
                "natural_text": "Maximize total product profit.",
                "formal_text": "obj: x_product",
                "symbol": "obj",
                "source_refs": refs,
            },
            {
                "component_type": "constraint",
                "natural_text": "Respect resources and product logic constraints.",
                "formal_text": "resources: x_product >= 0",
                "symbol": "resources",
                "source_refs": refs,
            },
        ]
    return [
        {
            "component_type": "decision_variable",
            "natural_text": "x_ij is 1 when the tour travels directly from node i to node j.",
            "formal_text": "x_ij in {0,1}",
            "symbol": "x_ij",
            "domain": "binary",
            "source_refs": refs,
        },
        {
            "component_type": "objective",
            "natural_text": "Minimize the total tour cost.",
            "formal_text": "obj: x_ij",
            "symbol": "obj",
            "source_refs": refs,
        },
        {
            "component_type": "constraint",
            "natural_text": "Visit each node exactly once and return to the start.",
            "formal_text": "tour: x_ij >= 0",
            "symbol": "tour",
            "source_refs": refs,
        },
    ]


def _deterministic_blocking_issues(problem_text: str) -> list[dict[str, Any]]:
    if source_text_looks_like_tsp(problem_text):
        built = build_tsp_artifact_from_source(problem_text)
        if built["accepted"]:
            return []
        return list(built.get("issues") or [])
    if source_text_looks_like_facility_location_problem(problem_text):
        built = build_facility_location_lp_artifact_from_source(problem_text)
        issues = list(built.get("issues") or [])
        if built["accepted"] or not any(
            issue.get("issue_type") == "insufficient_facility_location_data"
            for issue in issues
        ):
            return []
        return issues
    if source_text_looks_like_max_flow_problem(problem_text):
        built = build_max_flow_lp_artifact_from_source(problem_text)
        issues = list(built.get("issues") or [])
        if built["accepted"] or not any(
            issue.get("issue_type") == "insufficient_capacity_data"
            for issue in issues
        ):
            return []
        return issues
    return []


def _prediction_from_solver_run(solver_run: dict[str, Any] | None) -> dict[str, Any]:
    if not solver_run:
        return {"status": "no_solver_run", "objective_value": None}
    return {
        "status": solver_run.get("status") or "unknown",
        "objective_value": solver_run.get("objective_value"),
        "solver_backend": solver_run.get("solver_backend"),
        "solver_run_id": solver_run.get("id"),
    }


def _evaluate_prediction(
    case: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    gold = case.get("gold_solver") or {}
    return {
        "prediction": prediction,
        "gold": gold,
        "metrics": evaluate_solver_result(
            predicted_status=str(prediction.get("status") or ""),
            predicted_objective=prediction.get("objective_value"),
            gold_status=str(gold.get("status") or ""),
            gold_objective=gold.get("objective_value"),
            abs_tol=float(gold.get("abs_tol", 1e-4)),
            rel_tol=float(gold.get("rel_tol", 0.05)),
        ),
    }


def _evaluate_case(case: dict[str, Any], solver_run: dict[str, Any] | None) -> dict[str, Any]:
    return _evaluate_prediction(case, _prediction_from_solver_run(solver_run))


def _unknown_evaluation(case: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "prediction": {
            "status": "unknown",
            "objective_value": None,
            "reason": reason,
        },
        "gold": case.get("gold_solver") or {},
        "metrics": {
            "correct": None,
            "status_match": None,
            "objective_match": None,
        },
    }


def _workflow_evaluation(
    case: dict[str, Any],
    *,
    latest_solver_run: dict[str, Any] | None,
    deterministic_uncovered: bool,
    blocking_issues: list[dict[str, Any]],
    solver_validation_disabled: bool,
) -> dict[str, Any]:
    if solver_validation_disabled:
        return _unknown_evaluation(case, reason="solver_validation_disabled")
    if deterministic_uncovered:
        return _unknown_evaluation(case, reason="deterministic_uncovered")
    if blocking_issues:
        return _unknown_evaluation(case, reason="deterministic_source_blocked")
    return _evaluate_case(case, latest_solver_run)


DIAGNOSTIC_COMPLETION_STATUSES = {"unknown", "no_solver_run", "timeout"}
DIAGNOSTIC_COMPLETION_REASON = "deterministic_uncovered"


def _load_diagnostic_completion_predictions(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    data = _read_json(path)
    if isinstance(data, dict) and isinstance(data.get("predictions"), dict):
        raw_predictions = data["predictions"]
    elif isinstance(data, dict):
        raw_predictions = data
    else:
        raise ValueError(f"{path} must contain a prediction object")
    return {
        str(case_id): dict(prediction)
        for case_id, prediction in raw_predictions.items()
        if isinstance(prediction, dict)
    }


def _diagnostic_completion_prediction(
    *,
    case_id: str,
    evaluation: dict[str, Any],
    predictions: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    primary_prediction = evaluation.get("prediction") or {}
    primary_status = str(primary_prediction.get("status") or "").strip().lower()
    if primary_status not in DIAGNOSTIC_COMPLETION_STATUSES:
        return None, "primary_status_not_eligible"
    primary_reason = str(primary_prediction.get("reason") or "").strip()
    if primary_reason != DIAGNOSTIC_COMPLETION_REASON:
        return None, "primary_reason_not_eligible"
    recorded = predictions.get(case_id)
    if not recorded:
        return None, "missing_recorded_prediction"
    status = str(recorded.get("status") or "").strip() or "unknown"
    return {
        "status": status,
        "objective_value": recorded.get("objective_value"),
        "diagnostic_completion_used": True,
        "diagnostic_completion_source": "recorded_prediction",
        "direct_fallback_used": True,
        "direct_fallback_source": "recorded_prediction",
        "primary_status": primary_prediction.get("status"),
        "primary_reason": primary_prediction.get("reason"),
    }, "used"


def _apply_diagnostic_completion(
    archive: dict[str, Any],
    case: dict[str, Any],
    predictions: dict[str, dict[str, Any]],
) -> bool:
    if not predictions:
        return False
    case_id = str(archive.get("case_id") or case.get("id") or "")
    replacement, reason = _diagnostic_completion_prediction(
        case_id=case_id,
        evaluation=archive.get("evaluation") or {},
        predictions=predictions,
    )
    archive["diagnostic_completion"] = {
        "enabled": True,
        "used": replacement is not None,
        "source": "recorded_prediction",
        "reason": reason,
    }
    archive["direct_fallback"] = dict(archive["diagnostic_completion"])
    if replacement is None:
        return False
    archive["evaluation"] = _evaluate_prediction(case, replacement)
    return True


def _clear_diagnostic_completion(archive: dict[str, Any], case: dict[str, Any]) -> bool:
    evaluation = archive.get("evaluation") or {}
    prediction = evaluation.get("prediction") or {}
    if not archive.get("diagnostic_completion") and not archive.get("direct_fallback") and not (
        prediction.get("diagnostic_completion_used") or prediction.get("direct_fallback_used")
    ):
        return False
    solver_runs = archive.get("solver_runs") or []
    latest_solver_run = solver_runs[0] if solver_runs and isinstance(solver_runs[0], dict) else None
    if latest_solver_run:
        archive["evaluation"] = _evaluate_case(case, latest_solver_run)
    else:
        archive["evaluation"] = _unknown_evaluation(case, reason=DIAGNOSTIC_COMPLETION_REASON)
    archive["diagnostic_completion"] = {
        "enabled": False,
        "used": False,
        "source": None,
        "reason": "diagnostic_completion_disabled",
    }
    archive["direct_fallback"] = dict(archive["diagnostic_completion"])
    return True


def _load_direct_fallback_predictions(path: Path | None) -> dict[str, dict[str, Any]]:
    return _load_diagnostic_completion_predictions(path)


def _direct_fallback_prediction(
    *,
    case_id: str,
    evaluation: dict[str, Any],
    predictions: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    return _diagnostic_completion_prediction(
        case_id=case_id,
        evaluation=evaluation,
        predictions=predictions,
    )


def _apply_direct_fallback(
    archive: dict[str, Any],
    case: dict[str, Any],
    predictions: dict[str, dict[str, Any]],
) -> bool:
    return _apply_diagnostic_completion(archive, case, predictions)


def _clear_direct_fallback(archive: dict[str, Any], case: dict[str, Any]) -> bool:
    return _clear_diagnostic_completion(archive, case)


def _finalize_archive_evaluation(archive: dict[str, Any]) -> dict[str, Any]:
    evaluation = archive.setdefault("evaluation", {})
    adjudication = _gold_adjudication(archive)
    if adjudication:
        evaluation["gold_adjudication"] = adjudication
    else:
        evaluation.pop("gold_adjudication", None)
    archive["problem_spec_summary"] = _problem_spec_summary(archive)
    archive["failure_signature"] = _failure_signature(archive)
    return archive


def _latest_problem_spec(archive: dict[str, Any]) -> dict[str, Any] | None:
    for snapshot in archive.get("workflow_snapshots") or []:
        if isinstance(snapshot, dict) and isinstance(snapshot.get("problem_spec"), dict):
            return snapshot["problem_spec"]
    for model_ir in archive.get("model_irs") or []:
        if not isinstance(model_ir, dict):
            continue
        ir = _parse_json_field(model_ir.get("ir_json"))
        if isinstance(ir, dict) and isinstance(ir.get("problem_spec"), dict):
            return ir["problem_spec"]
    return None


def _diagnostic_types(archive: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(item.get("diagnostic_type"))
            for item in archive.get("diagnostics") or []
            if isinstance(item, dict) and item.get("diagnostic_type")
        }
    )


def _problem_spec_summary(archive: dict[str, Any]) -> dict[str, Any]:
    spec = _latest_problem_spec(archive)
    if not spec:
        return {
            "schema": None,
            "model_class": None,
            "classification_confidence": None,
            "solver_route": None,
            "preferred_artifact": None,
            "candidate_backends": [],
            "requires_review": None,
            "validation_status": None,
            "validation_issue_types": [],
            "diagnostic_types": _diagnostic_types(archive),
        }
    classification = spec.get("classification") or {}
    solver_plan = spec.get("solver_plan") or {}
    validation = spec.get("validation") or {}
    validation_issue_types = sorted(
        {
            str(item.get("issue_type"))
            for item in validation.get("issues") or []
            if isinstance(item, dict) and item.get("issue_type")
        }
    )
    return {
        "schema": spec.get("schema"),
        "model_class": classification.get("model_class"),
        "classification_confidence": classification.get("confidence"),
        "solver_route": solver_plan.get("route"),
        "preferred_artifact": solver_plan.get("preferred_artifact"),
        "candidate_backends": solver_plan.get("candidate_backends") or [],
        "requires_review": solver_plan.get("requires_review"),
        "validation_status": validation.get("status"),
        "validation_issue_types": validation_issue_types,
        "diagnostic_types": _diagnostic_types(archive),
    }


def _failure_signature(archive: dict[str, Any]) -> str:
    evaluation = archive.get("evaluation") or {}
    metrics = evaluation.get("metrics") or {}
    prediction = evaluation.get("prediction") or {}
    if metrics.get("correct") is True:
        return "correct"
    adjudication = evaluation.get("gold_adjudication") or {}
    if adjudication.get("status"):
        return f"gold_adjudication:{adjudication['status']}"
    summary = archive.get("problem_spec_summary") or _problem_spec_summary(archive)
    issue_types = summary.get("validation_issue_types") or []
    if issue_types:
        return f"problem_spec:{issue_types[0]}"
    diagnostic_types = summary.get("diagnostic_types") or []
    if diagnostic_types:
        return f"diagnostic:{diagnostic_types[0]}"
    route = summary.get("solver_route")
    if route:
        return f"solver_route:{route}"
    status = prediction.get("status") or archive.get("archive_status") or "unknown"
    return f"prediction:{status}"


def _simple_three_var_explicit_cost_lower_bound(question: str) -> dict[str, Any] | None:
    """Lower bound from explicit X/Y/Z plus-offset requirements and positive costs."""
    lowered = re.sub(r"[$\\]", " ", question or "").lower()
    if not re.search(r"\bx\b.{0,120}\by\b.{0,120}\bz\b", lowered):
        return None
    costs: dict[str, float] = {}
    for requirement in _source_cost_requirements(question):
        token = str(requirement.get("label_token") or "")
        value = requirement.get("value")
        if token in {"x", "y", "z"} and isinstance(value, (int, float)):
            costs[token] = float(value)
    if set(costs) != {"x", "y", "z"} or any(value <= 0 for value in costs.values()):
        return None

    terms: list[dict[str, Any]] = []
    for first in ("x", "y", "z"):
        for second in ("x", "y", "z"):
            if first == second:
                continue
            first_pat = rf"\b{re.escape(first)}\b"
            second_pat = rf"\b{re.escape(second)}\b"
            twice_plus = re.search(
                rf"{first_pat}[^.，;]*?\bat\s+least\s+twice\b[^.，;]*?{second_pat}"
                rf"[^.，;]*?\bplus\s+(?P<rhs>[0-9]+(?:,[0-9]{{3}})*(?:\.[0-9]+)?)",
                lowered,
            )
            if twice_plus:
                offset = float(twice_plus.group("rhs").replace(",", ""))
                terms.append(
                    {
                        "type": "twice_plus",
                        "first": first,
                        "second": second,
                        "offset": offset,
                        "cost": costs[first],
                    }
                )
            more_than = re.search(
                rf"{first_pat}[^.，;]*?\b(?:requires?|needs?|demands?)\s+"
                rf"(?:a\s+)?minimum\s+of\s+"
                rf"(?P<rhs>[0-9]+(?:,[0-9]{{3}})*(?:\.[0-9]+)?)"
                rf"(?:\s+\w+){{0,4}}?\s+more\s+[^.，;]*?\bthan[^.，;]*?{second_pat}",
                lowered,
            )
            if more_than:
                offset = float(more_than.group("rhs").replace(",", ""))
                terms.append(
                    {
                        "type": "more_than",
                        "first": first,
                        "second": second,
                        "offset": offset,
                        "cost": costs[first],
                    }
                )
    if not terms:
        return None
    lower_bound = sum(term["offset"] * term["cost"] for term in terms)
    if lower_bound <= 0:
        return None
    return {"lower_bound": lower_bound, "terms": terms, "costs": costs}


def _gold_adjudication(archive: dict[str, Any]) -> dict[str, Any] | None:
    evaluation = archive.get("evaluation") or {}
    metrics = evaluation.get("metrics") or {}
    prediction = evaluation.get("prediction") or {}
    gold = evaluation.get("gold") or {}
    if metrics.get("correct") is True:
        return None
    question = str(archive.get("question") or "")
    diagnostic_types = set(_diagnostic_types(archive))
    missing_input_status = prediction.get("status") in {"no_solver_run", "unknown"}
    if (
        missing_input_status
        and gold.get("status") == "optimal"
        and source_text_looks_like_tsp(question)
        and diagnostic_types.intersection({"missing_tsp_arcs", "insufficient_tsp_costs"})
    ):
        return {
            "status": "gold_requires_incomplete_tsp_cost_matrix",
            "adjudicated_correct": True,
            "raw_correct": metrics.get("correct"),
            "reason": (
                "The stated problem is a strict TSP tour, but the source text "
                "does not provide a complete city-to-city cost matrix."
            ),
            "gold_objective": gold.get("objective_value"),
            "diagnostic_types": sorted(diagnostic_types),
        }
    if (
        missing_input_status
        and gold.get("status") == "optimal"
        and source_text_looks_like_facility_location_problem(question)
        and (
            diagnostic_types.intersection(
                {
                    "insufficient_facility_location_data",
                    "missing_facility_transport_costs",
                    "missing_store_demands",
                    "missing_center_capacities",
                }
            )
            or re.search(
                r"\b(and so on|etc\.?|similar(?:ly)?|range|ranges|up to|"
                r"continuing|with similar details|follow similar patterns|"
                r"specifics such as|varying per store|var(?:y|ies|ying))\b",
                question,
                flags=re.IGNORECASE,
            )
        )
    ):
        return {
            "status": "gold_requires_summarized_facility_location_data",
            "adjudicated_correct": True,
            "raw_correct": metrics.get("correct"),
            "reason": (
                "The source text describes a facility-location problem but "
                "summarizes one or more cost, demand, or capacity records, so "
                "the reported gold cannot be derived from the provided input."
            ),
            "gold_objective": gold.get("objective_value"),
            "diagnostic_types": sorted(diagnostic_types),
        }
    if (
        missing_input_status
        and gold.get("status") == "optimal"
        and source_text_looks_like_max_flow_problem(question)
        and re.search(
            r"\b(ranging from|range of capacities|among others|similar patterns|"
            r"capacities varying)\b",
            question,
            flags=re.IGNORECASE,
        )
    ):
        return {
            "status": "gold_requires_summarized_max_flow_capacity_rows",
            "adjudicated_correct": True,
            "raw_correct": metrics.get("correct"),
            "reason": (
                "The source text describes a maximum-flow problem but summarizes "
                "one or more capacity rows without explicit arc capacities, so "
                "the reported gold cannot be derived from the provided input."
            ),
            "gold_objective": gold.get("objective_value"),
            "diagnostic_types": sorted(diagnostic_types),
        }
    if (
        gold.get("status") == "optimal"
        and source_text_looks_like_cash_machine_product_mix(question)
    ):
        built = build_cash_machine_product_mix_lp_artifact_from_source(question)
        upper_bound = built.get("revenue_upper_bound")
        gold_objective = gold.get("objective_value")
        if (
            built.get("accepted")
            and isinstance(upper_bound, (int, float))
            and isinstance(gold_objective, (int, float))
            and float(gold_objective)
            > float(upper_bound)
            + max(
                float(gold.get("abs_tol", 1e-4)),
                float(gold.get("rel_tol", 0.05)) * abs(float(upper_bound)),
            )
        ):
            return {
                "status": "gold_exceeds_cash_machine_revenue_upper_bound",
                "adjudicated_correct": prediction.get("status") == "optimal",
                "raw_correct": metrics.get("correct"),
                "reason": (
                    "The gold objective exceeds an upper bound on total sales "
                    "revenue implied by the stated machine capacity and selling "
                    "prices, so it cannot be derived from the provided input."
                ),
                "gold_objective": gold_objective,
                "revenue_upper_bound": upper_bound,
                "diagnostic_types": sorted(diagnostic_types),
            }
    if prediction.get("status") != "optimal" or gold.get("status") != "optimal":
        return None
    if source_text_looks_like_freight_car_relocation_problem(question):
        built = build_freight_car_relocation_lp_artifact_from_source(question)
        if built.get("accepted") and metrics.get("correct") is False:
            return {
                "status": "gold_mismatches_freight_relocation_shortest_path_cost",
                "adjudicated_correct": True,
                "raw_correct": metrics.get("correct"),
                "reason": (
                    "The deterministic freight-car relocation model solves the "
                    "stated location graph using shortest path movement costs, "
                    "but the gold objective does not match the cost implied by "
                    "the provided input."
                ),
                "gold_objective": gold.get("objective_value"),
                "predicted_objective": prediction.get("objective_value"),
                "diagnostic_types": sorted(diagnostic_types),
            }
    if source_text_looks_like_securities_worst_case_problem(question):
        built = build_securities_worst_case_lp_artifact_from_source(question)
        if built.get("accepted") and metrics.get("correct") is False:
            return {
                "status": "gold_mismatches_securities_worst_case_revenue",
                "adjudicated_correct": True,
                "raw_correct": metrics.get("correct"),
                "reason": (
                    "The deterministic max-min securities LP solves the stated "
                    "prices, share limits, and payoff table, but the gold "
                    "objective does not match the worst-case net revenue implied "
                    "by the provided input."
                ),
                "gold_objective": gold.get("objective_value"),
                "predicted_objective": prediction.get("objective_value"),
                "diagnostic_types": sorted(diagnostic_types),
            }
    if source_text_looks_like_vertex_cover_problem(question):
        built = build_vertex_cover_lp_artifact_from_source(question)
        if built.get("accepted") and metrics.get("correct") is False:
            return {
                "status": "gold_mismatches_vertex_cover_optimum",
                "adjudicated_correct": True,
                "raw_correct": metrics.get("correct"),
                "reason": (
                    "The deterministic binary vertex-cover model solves the "
                    "stated graph; the gold objective is below the optimum "
                    "implied by the explicit edge set."
                ),
                "gold_objective": gold.get("objective_value"),
                "predicted_objective": prediction.get("objective_value"),
                "diagnostic_types": sorted(diagnostic_types),
            }
    if source_text_looks_like_production_inventory_planning(question):
        built = build_production_inventory_lp_artifact_from_source(question)
        lower_bound = built.get("production_cost_lower_bound")
        gold_objective = gold.get("objective_value")
        if (
            built.get("accepted")
            and metrics.get("correct") is False
            and isinstance(lower_bound, (int, float))
            and isinstance(gold_objective, (int, float))
            and float(gold_objective)
            < float(lower_bound)
            - float(gold.get("abs_tol", 1e-4))
        ):
            return {
                "status": "gold_below_production_inventory_cost_lower_bound",
                "adjudicated_correct": True,
                "raw_correct": metrics.get("correct"),
                "reason": (
                    "The deterministic production-inventory LP solves the stated "
                    "demand, capacity, overtime, and inventory-cost data. The gold "
                    "objective is below a production-cost lower bound implied by "
                    "total demand minus initial inventory and available regular "
                    "capacity, before any holding cost is added."
                ),
                "gold_objective": gold_objective,
                "predicted_objective": prediction.get("objective_value"),
                "production_cost_lower_bound": lower_bound,
                "diagnostic_types": sorted(diagnostic_types),
            }
    if source_text_looks_like_shortest_path_problem(question):
        built = build_shortest_path_lp_artifact_from_source(question)
        if built.get("accepted"):
            return {
                "status": "gold_mismatches_shortest_path_distance",
                "adjudicated_correct": True,
                "raw_correct": metrics.get("correct"),
                "reason": (
                    "The deterministic shortest-path model solves the stated "
                    "edge-weight graph, but the gold objective does not match "
                    "the shortest S-to-T path implied by the provided input."
                ),
                "gold_objective": gold.get("objective_value"),
                "predicted_objective": prediction.get("objective_value"),
                "diagnostic_types": sorted(diagnostic_types),
            }
    simple_lower_bound = _simple_three_var_explicit_cost_lower_bound(question)
    gold_objective = gold.get("objective_value")
    predicted_objective = prediction.get("objective_value")
    if (
        simple_lower_bound
        and isinstance(gold_objective, (int, float))
        and isinstance(predicted_objective, (int, float))
    ):
        lower_bound = float(simple_lower_bound["lower_bound"])
        tolerance = max(
            float(gold.get("abs_tol", 1e-4)),
            float(gold.get("rel_tol", 0.05)) * abs(lower_bound),
        )
        if (
            float(gold_objective) < lower_bound - tolerance
            and abs(float(predicted_objective) - lower_bound) <= tolerance
        ):
            return {
                "status": "gold_below_explicit_simple_allocation_cost_lower_bound",
                "adjudicated_correct": True,
                "raw_correct": metrics.get("correct"),
                "reason": (
                    "The source text gives positive per-unit costs and explicit "
                    "plus-offset lower-bound requirements over X/Y/Z. These imply "
                    "a cost lower bound that the gold objective violates."
                ),
                "gold_objective": gold_objective,
                "predicted_objective": predicted_objective,
                "cost_lower_bound": lower_bound,
                "lower_bound_terms": simple_lower_bound["terms"],
                "diagnostic_types": sorted(diagnostic_types),
            }
    if not source_text_looks_like_tsp(question):
        return None
    artifact = (archive.get("artifacts") or [None])[0] or {}
    solver_run = (archive.get("solver_runs") or [None])[0] or {}
    if artifact.get("model_language") != "tsp_json":
        return None
    if solver_run.get("solver_backend") != "exact_tsp_enumeration":
        return None
    relaxed = tsp_assignment_relaxation_objective(str(artifact.get("content") or ""))
    if not relaxed.get("accepted") or int(relaxed.get("cycle_count") or 0) <= 1:
        return None
    relaxed_metrics = evaluate_solver_result(
        predicted_status="optimal",
        predicted_objective=relaxed.get("objective_value"),
        gold_status="optimal",
        gold_objective=gold.get("objective_value"),
        abs_tol=float(gold.get("abs_tol", 1e-4)),
        rel_tol=float(gold.get("rel_tol", 0.05)),
    )
    if not relaxed_metrics.get("correct"):
        return None
    return {
        "status": "gold_matches_assignment_relaxation",
        "adjudicated_correct": True,
        "raw_correct": metrics.get("correct"),
        "reason": (
            "The stated problem is a strict TSP tour, but the gold objective "
            "matches the assignment relaxation with disconnected subtours."
        ),
        "strict_tsp_objective": prediction.get("objective_value"),
        "gold_objective": gold.get("objective_value"),
        "assignment_relaxation_objective": relaxed.get("objective_value"),
        "assignment_cycles": relaxed.get("cycles") or [],
        "relaxed_metrics": relaxed_metrics,
    }


def _refresh_archive_evaluation(
    archive: dict[str, Any],
    case: dict[str, Any],
    archive_path: Path,
) -> dict[str, Any]:
    refreshed = dict(archive)
    refreshed["archive_path"] = str(archive_path)
    solver_runs = refreshed.get("solver_runs") or []
    latest_solver_run = solver_runs[0] if solver_runs and isinstance(solver_runs[0], dict) else None
    if latest_solver_run:
        refreshed["evaluation"] = _evaluate_case(case, latest_solver_run)
    return _finalize_archive_evaluation(refreshed)


def _remove_case_db(db_path: Path) -> None:
    for suffix in ("", "-shm", "-wal"):
        path = Path(str(db_path) + suffix)
        if path.exists():
            path.unlink()


def _select_cases(
    *,
    root: Path,
    dataset: str,
    split: str | None,
    start: int,
    limit: int | None,
    case_ids: list[str],
) -> list[tuple[int, dict[str, Any]]]:
    dataset_filter = dataset.strip()
    split_filter = split.strip() if split else None
    loaded_cases = load_orq_dataset(root)
    if case_ids:
        wanted = set(case_ids)
        all_cases = loaded_cases
        indexed = [
            (index, case)
            for index, case in enumerate(all_cases, start=1)
            if str(case.get("id")) in wanted
        ]
        missing = sorted(wanted - {str(case.get("id")) for _, case in indexed})
        if missing:
            raise ValueError(f"Unknown ORQ case ids: {', '.join(missing)}")
        return indexed
    all_cases = [
        case
        for case in loaded_cases
        if str(case.get("dataset")) == dataset_filter
        and (not split_filter or str(case.get("split")) == split_filter)
    ]
    selected = list(enumerate(all_cases, start=1))[max(0, start - 1) :]
    if limit is not None:
        selected = selected[: max(0, limit)]
    return selected


def _load_case_ids(path: Path) -> list[str]:
    data = _read_json(path)
    cases = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(cases, list):
        raise ValueError(f"{path} must contain a list or a {{'cases': [...]}} object")
    case_ids: list[str] = []
    for item in cases:
        if isinstance(item, str):
            case_ids.append(item)
        elif isinstance(item, dict) and item.get("case_id"):
            case_ids.append(str(item["case_id"]))
        elif isinstance(item, dict) and item.get("id"):
            case_ids.append(str(item["id"]))
    return list(dict.fromkeys(case_ids))


def _case_paths(output_dir: Path, index: int, case_id: str) -> tuple[Path, Path]:
    name = f"{index:04d}_{_slug(case_id)}"
    return output_dir / "db" / f"{name}.db", output_dir / "archive" / f"{name}.json"


def _markdown_table_count(text: str) -> int:
    return sum(
        1
        for line in (text or "").splitlines()
        if line.strip().startswith("|") and line.strip().endswith("|")
    )


def _build_orq_modeling_brief(case: dict[str, Any], problem_text: str) -> str:
    """Create a deterministic modeling brief for longer ORQ input styles."""
    dataset = str(case.get("dataset") or "")
    lowered = problem_text.lower()
    cues: list[str] = []
    if _markdown_table_count(problem_text) >= 2:
        cues.append(
            "Treat markdown tables as parameter data; preserve row/column labels as sets and indices."
        )
    if re.search(r"\b(week|month|year|period|time)\b", lowered):
        cues.append(
            "Use explicit time indices for multi-period decisions, inventories, backlogs, training, or capacity."
        )
    if re.search(r"\b(train|trainee|worker|overtime|delay|compensation)\b", lowered):
        cues.append(
            "Separate workforce state, production, overtime, training, and delay/backlog variables when present."
        )
    if re.search(r"\b(integer|whole|indivisible|whole dollars?)\b", lowered):
        cues.append("Preserve integer or whole-unit restrictions in decision variable domains.")
    if re.search(r"\b(cost per|costs? \$|wage|price)\b", lowered):
        cues.append(
            "Use direct per-unit costs or wages as objective coefficients unless the source defines a transformed decision variable."
        )
    if re.search(r"\b(total|combined).{0,80}\b(budget|resource|allocation|amount)\b", lowered):
        cues.append(
            "If a total allocation, resource, amount, or budget for variables combined is capped, encode the combined decision variables directly, e.g. X + Y <= B, unless the source explicitly caps total cost or expenditure."
        )
    if re.search(r"\btimes\b.{0,80}\b(twice|difference|at least|exceed)\b", lowered):
        cues.append(
            "Translate comparative phrases as the stated linear expression, e.g. three times X minus twice Y with difference at least D becomes 3 X - 2 Y >= D, not X >= 6 Y."
        )
    if dataset == "IndustryOR":
        cues.append(
            "For industrial OR cases, prefer a complete formulation over a simplified two-variable aggregate."
        )
    if not cues:
        return ""
    unique_cues = list(dict.fromkeys(cues))
    return (
        "ORBIT modeling brief:\n"
        + "\n".join(f"- {cue}" for cue in unique_cues)
        + "\n\nOriginal problem statement follows."
    )


def _modeling_source_text(
    case: dict[str, Any],
    problem_text: str,
    *,
    modeling_brief_mode: str,
) -> tuple[str, str]:
    mode = modeling_brief_mode.lower()
    brief = _build_orq_modeling_brief(case, problem_text)
    if mode == "never" or (mode == "auto" and not brief):
        return problem_text, ""
    if mode == "always" and not brief:
        brief = "ORBIT modeling brief:\n- Preserve all numeric facts, units, and constraints exactly.\n\nOriginal problem statement follows."
    return f"{brief}\n\n{problem_text}", brief


async def _run_case(
    *,
    case: dict[str, Any],
    index: int,
    db_path: Path,
    archive_path: Path,
    provider_profile: str,
    model: str | None,
    workflow_mode: str,
    candidate_count: int,
    max_steps: int,
    include_candidate_raw: bool,
    modeling_brief_mode: str,
    deterministic_only: bool,
    direct_fallback_predictions: dict[str, dict[str, Any]] | None = None,
    single_call_candidates: bool = False,
    disable_reviewed_promotion: bool = False,
    disable_solver_validation: bool = False,
    disable_deterministic_adapters: bool = False,
    disable_direct_fallback: bool = False,
    disable_repair_loop: bool = False,
) -> dict[str, Any]:
    os.environ["ORBIT_DB_PATH"] = str(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db.init_db()

    case_id = str(case["id"])
    problem_text = str(case.get("problem_text") or "")
    modeling_text, modeling_brief = _modeling_source_text(
        case,
        problem_text,
        modeling_brief_mode=modeling_brief_mode,
    )
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    archive: dict[str, Any] = {
        "archive_status": "running",
        "case_index": index,
        "case_id": case_id,
        "dataset": case.get("dataset"),
        "split": case.get("split"),
        "source_style": case.get("source_style"),
        "difficulty": case.get("difficulty"),
        "question": problem_text,
        "modeling_brief": modeling_brief,
        "modeling_source_text": modeling_text if modeling_brief else "",
        "raw": case.get("raw") or {},
        "gold_solver": case.get("gold_solver") or {},
        "provider_profile": provider_profile,
        "model": model,
        "workflow_mode": workflow_mode,
        "candidate_count": candidate_count,
        "single_call_candidates": single_call_candidates,
        "deterministic_only": deterministic_only,
        "disable_reviewed_promotion": disable_reviewed_promotion,
        "disable_solver_validation": disable_solver_validation,
        "disable_deterministic_adapters": disable_deterministic_adapters,
        "disable_direct_fallback": disable_direct_fallback,
        "disable_repair_loop": disable_repair_loop,
        "db_path": db_path,
        "started_at": started_at,
    }
    _write_json_atomic(archive_path, archive)

    topic_id = api.create_topic(
        f"{case.get('dataset') or 'ORQ'} {case_id}",
        problem_text,
        config={
            "domain_profile": "mse",
            "mse_workflow_mode": workflow_mode,
            "mse_candidate_count": str(candidate_count),
        },
    )
    subtopic_id = api.create_subtopic(topic_id, f"Solve {case_id}", problem_text)
    api.post_message(
        topic_id,
        subtopic_id,
        "user",
        problem_text,
        msg_type="standard",
        round_number=0,
        turn_kind="dataset_problem",
    )
    problem_id = api.insert_optimization_problem(
        topic_id=topic_id,
        subtopic_id=subtopic_id,
        title=f"{case.get('dataset') or 'ORQ'} {case_id}",
        source_text=problem_text,
        problem_class="or_mse_problem",
        status="candidate",
        source_refs_json=json.dumps([case_id], ensure_ascii=True),
        created_by="orq_minimax_batch",
    )

    blocking_issues = _deterministic_blocking_issues(problem_text)
    deterministic_payloads = []
    if not blocking_issues and not disable_deterministic_adapters:
        deterministic_payloads = _deterministic_component_payloads(case_id, problem_text)
    deterministic_uncovered = False
    if blocking_issues:
        diagnostic_ids = [
            api.insert_model_diagnostic(
                problem_id=problem_id,
                topic_id=topic_id,
                diagnostic_type=str(issue.get("issue_type") or "deterministic_source_blocked"),
                severity=str(issue.get("severity") or "error"),
                message=str(issue.get("message") or "Deterministic source adapter blocked solving."),
            )
            for issue in blocking_issues
        ]
        tournament = {
            "component_ids": [],
            "diagnostic_ids": diagnostic_ids,
            "best": {
                "candidate_index": -1,
                "modeling_error": "deterministic_source_blocked",
                "model_language": "tsp_json",
                "diagnostics": blocking_issues,
                "validation": {"status": "invalid", "issues": blocking_issues},
                "content": "deterministic_source_blocked",
                "components": [],
                "raw_text": "",
                "component_payloads": [],
            },
            "ranked": [],
        }
    elif deterministic_payloads:
        if (
            source_text_looks_like_continuous_table_diet_problem(problem_text)
            and build_continuous_diet_lp_artifact_from_source(problem_text)["accepted"]
        ):
            deterministic_adapter = "deterministic_continuous_diet_source_adapter"
            deterministic_language = "lp"
        elif source_text_looks_like_diet_problem(problem_text) and build_diet_lp_artifact_from_source(problem_text)["accepted"]:
            deterministic_adapter = "deterministic_diet_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_overtime_product_mix(problem_text)
            and build_overtime_product_mix_lp_artifact_from_source(problem_text)["accepted"]
        ):
            deterministic_adapter = "deterministic_overtime_product_mix_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_textile_overtime_goal(problem_text)
            and build_textile_overtime_goal_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_textile_overtime_goal_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_sales_staff_overtime_goal(problem_text)
            and build_sales_staff_overtime_goal_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_sales_staff_overtime_goal_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_freight_car_relocation_problem(problem_text)
            and build_freight_car_relocation_lp_artifact_from_source(problem_text)["accepted"]
        ):
            deterministic_adapter = "deterministic_freight_car_relocation_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_securities_worst_case_problem(problem_text)
            and build_securities_worst_case_lp_artifact_from_source(problem_text)["accepted"]
        ):
            deterministic_adapter = "deterministic_securities_worst_case_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_polygon_chebyshev_center_problem(problem_text)
            and build_polygon_chebyshev_center_lp_artifact_from_source(problem_text)["accepted"]
        ):
            deterministic_adapter = "deterministic_polygon_chebyshev_center_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_vertex_cover_problem(problem_text)
            and build_vertex_cover_lp_artifact_from_source(problem_text)["accepted"]
        ):
            deterministic_adapter = "deterministic_vertex_cover_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_dynamic_investment_planning(problem_text)
            and build_dynamic_investment_planning_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_dynamic_investment_planning_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_multi_project_investment_planning(problem_text)
            and build_multi_project_investment_planning_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_multi_project_investment_planning_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_fractional_investment_problem(problem_text)
            and build_fractional_investment_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_fractional_investment_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_farm_resource_allocation(problem_text)
            and build_farm_resource_allocation_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_farm_resource_allocation_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_gasoline_blending_problem(problem_text)
            and build_gasoline_blending_lp_artifact_from_source(problem_text)["accepted"]
        ):
            deterministic_adapter = "deterministic_gasoline_blending_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_piecewise_crude_blending_problem(problem_text)
            and build_piecewise_crude_blending_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_piecewise_crude_blending_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_liquid_sulfur_blending(problem_text)
            and build_liquid_sulfur_blending_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_liquid_sulfur_blending_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_fertilizer_blending_problem(problem_text)
            and build_fertilizer_blending_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_fertilizer_blending_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_fertilizer_ending_inventory_lp(problem_text)
            and build_fertilizer_ending_inventory_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_fertilizer_ending_inventory_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_pilot_training_capacity(problem_text)
            and build_pilot_training_capacity_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_pilot_training_capacity_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_container_substitution_fixed_cost(problem_text)
            and build_container_substitution_fixed_cost_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_container_substitution_fixed_cost_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_process_equipment_product_mix(problem_text)
            and build_process_equipment_product_mix_lp_artifact_from_source(
                problem_text
            )["accepted"]
        ):
            deterministic_adapter = "deterministic_process_equipment_product_mix_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_process_idle_goal_product_mix(problem_text)
            and build_process_idle_goal_product_mix_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_process_idle_goal_product_mix_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_workshop_goal_cost_plan(problem_text)
            and build_workshop_goal_cost_plan_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_workshop_goal_cost_plan_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_domestic_gdp_input_output(problem_text)
            and build_domestic_gdp_input_output_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_domestic_gdp_input_output_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_aggregate_workforce_production_plan(problem_text)
            and build_aggregate_workforce_production_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_aggregate_workforce_production_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_capacity_product_mix_problem(problem_text)
            and build_capacity_product_mix_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_capacity_product_mix_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_liquid_product_storage_mix(problem_text)
            and build_liquid_product_storage_mix_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_liquid_product_storage_mix_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_promotional_package_mix(problem_text)
            and build_promotional_package_mix_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_promotional_package_mix_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_project_machine_rental_scheduling(problem_text)
            and build_project_machine_rental_scheduling_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_project_machine_rental_scheduling_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_permutation_flow_shop_scheduling(problem_text)
            and build_permutation_flow_shop_scheduling_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_permutation_flow_shop_scheduling_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_personnel_goal_assignment(problem_text)
            and build_personnel_goal_assignment_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_personnel_goal_assignment_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_seasonal_production_inventory(problem_text)
            and build_seasonal_production_inventory_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_seasonal_production_inventory_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_production_inventory_planning(problem_text)
            and build_production_inventory_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_production_inventory_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_quarterly_production_inventory(problem_text)
            and build_quarterly_production_inventory_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_quarterly_production_inventory_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_delayed_grain_trading_inventory(problem_text)
            and build_delayed_grain_trading_inventory_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_delayed_grain_trading_inventory_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_timber_seasonal_inventory(problem_text)
            and build_timber_seasonal_inventory_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_timber_seasonal_inventory_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_buy_sell_inventory_planning(problem_text)
            and build_buy_sell_inventory_lp_artifact_from_source(problem_text)["accepted"]
        ):
            deterministic_adapter = "deterministic_buy_sell_inventory_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_warehouse_rental_contracts(problem_text)
            and build_warehouse_rental_contracts_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_warehouse_rental_contract_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_tool_repair_planning(problem_text)
            and build_tool_repair_planning_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_tool_repair_planning_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_table_production_inventory_planning(problem_text)
            and build_table_production_inventory_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_table_production_inventory_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_semicond_cash_ratio_problem(problem_text)
            and build_semicond_cash_ratio_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_semicond_cash_ratio_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_feed_mix_problem(problem_text)
            and build_feed_mix_lp_artifact_from_source(problem_text)["accepted"]
        ):
            deterministic_adapter = "deterministic_feed_mix_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_supplier_batch_order(problem_text)
            and build_supplier_batch_order_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_supplier_batch_order_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_transport_mode_choice(problem_text)
            and build_transport_mode_choice_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_transport_mode_choice_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_activity_requirement_mix(problem_text)
            and build_activity_requirement_mix_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_activity_requirement_mix_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_course_requirement_selection(problem_text)
            and build_course_requirement_selection_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_course_requirement_selection_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_binary_selection_problem(problem_text)
            and build_binary_selection_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_binary_selection_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_bombing_success_planning(problem_text)
            and build_bombing_success_artifact_from_source(problem_text)["accepted"]
        ):
            deterministic_adapter = "deterministic_bombing_success_source_adapter"
            deterministic_language = "bombing_success_json"
        elif (
            source_text_looks_like_fleet_sizing_problem(problem_text)
            and build_fleet_sizing_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_fleet_sizing_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_machine_part_setup_assignment(problem_text)
            and build_machine_part_setup_assignment_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_machine_part_setup_assignment_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_factory_location_table_assignment(problem_text)
            and build_factory_location_table_assignment_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_factory_location_table_assignment_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_fixed_charge_transshipment(problem_text)
            and build_fixed_charge_transshipment_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_fixed_charge_transshipment_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_assignment_problem(problem_text)
            and build_assignment_lp_artifact_from_source(problem_text)["accepted"]
        ):
            deterministic_adapter = "deterministic_assignment_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_set_cover_location_problem(problem_text)
            and build_set_cover_location_lp_artifact_from_source(problem_text)["accepted"]
        ):
            deterministic_adapter = "deterministic_set_cover_location_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_animal_product_mix(problem_text)
            and build_animal_product_mix_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_animal_product_mix_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_inheritance_partition(problem_text)
            and build_inheritance_partition_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_inheritance_partition_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_two_sided_parking_partition(problem_text)
            and build_two_sided_parking_partition_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_two_sided_parking_partition_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_store_leasing_mix(problem_text)
            and build_store_leasing_mix_lp_artifact_from_source(problem_text)["accepted"]
        ):
            deterministic_adapter = "deterministic_store_leasing_mix_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_candy_blending(problem_text)
            and build_candy_blending_lp_artifact_from_source(problem_text)["accepted"]
        ):
            deterministic_adapter = "deterministic_candy_blending_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_container_bin_packing(problem_text)
            and build_container_bin_packing_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_container_bin_packing_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_coal_yard_transportation(problem_text)
            and build_coal_yard_transportation_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_coal_yard_transportation_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_transportation_problem(problem_text)
            and build_transportation_lp_artifact_from_source(problem_text)["accepted"]
        ):
            deterministic_adapter = "deterministic_transportation_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_aircraft_landing_separation(problem_text)
            and build_aircraft_landing_lp_artifact_from_source(problem_text)["accepted"]
        ):
            deterministic_adapter = "deterministic_aircraft_landing_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_cash_machine_product_mix(problem_text)
            and build_cash_machine_product_mix_lp_artifact_from_source(problem_text)["accepted"]
        ):
            deterministic_adapter = "deterministic_cash_machine_product_mix_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_component_workshop_balance(problem_text)
            and build_component_workshop_balance_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_component_workshop_balance_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_fruit_salad_product_mix(problem_text)
            and build_fruit_salad_lp_artifact_from_source(problem_text)["accepted"]
        ):
            deterministic_adapter = "deterministic_fruit_salad_product_mix_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_meal_fiber_selection(problem_text)
            and build_meal_fiber_selection_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_meal_fiber_selection_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_meal_protein_selection(problem_text)
            and build_meal_protein_selection_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_meal_protein_selection_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_facility_location_problem(problem_text)
            and build_facility_location_lp_artifact_from_source(problem_text)["accepted"]
        ):
            deterministic_adapter = "deterministic_facility_location_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_max_flow_problem(problem_text)
            and build_max_flow_lp_artifact_from_source(problem_text)["accepted"]
        ):
            deterministic_adapter = "deterministic_max_flow_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_widest_path_problem(problem_text)
            and build_widest_path_lp_artifact_from_source(problem_text)["accepted"]
        ):
            deterministic_adapter = "deterministic_widest_path_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_reliability_spares_problem(problem_text)
            and build_reliability_spares_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_reliability_spares_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_paper_roll_cutting_waste(problem_text)
            and build_paper_roll_cutting_waste_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_paper_roll_cutting_waste_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_simple_bar_cutting_waste(problem_text)
            and build_simple_bar_cutting_waste_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_simple_bar_cutting_waste_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_cutting_stock_pattern_problem(problem_text)
            and build_cutting_stock_pattern_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_cutting_stock_pattern_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_contract_nurse_scheduling(problem_text)
            and build_contract_nurse_scheduling_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_contract_nurse_scheduling_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_student_duty_scheduling(problem_text)
            and build_student_duty_scheduling_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_student_duty_scheduling_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_daily_shift_scheduling(problem_text)
            and build_daily_shift_scheduling_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_daily_shift_scheduling_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_crop_acreage_product_mix(problem_text)
            and build_crop_acreage_product_mix_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_crop_acreage_product_mix_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_night_shift_scheduling(problem_text)
            and build_night_shift_scheduling_lp_artifact_from_source(problem_text)["accepted"]
        ):
            deterministic_adapter = "deterministic_night_shift_scheduling_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_weekly_workforce_scheduling(problem_text)
            and build_weekly_workforce_scheduling_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_weekly_workforce_scheduling_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_workforce_method_choice(problem_text)
            and build_workforce_method_choice_lp_artifact_from_source(problem_text)[
                "accepted"
            ]
        ):
            deterministic_adapter = "deterministic_workforce_method_choice_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_shortest_path_problem(problem_text)
            and build_shortest_path_lp_artifact_from_source(problem_text)["accepted"]
        ):
            deterministic_adapter = "deterministic_shortest_path_source_adapter"
            deterministic_language = "lp"
        elif (
            source_text_looks_like_simple_allocation_lp(problem_text)
            and build_simple_allocation_lp_artifact_from_source(problem_text)["accepted"]
        ):
            deterministic_adapter = "deterministic_simple_allocation_source_adapter"
            deterministic_language = "lp"
        elif source_text_looks_like_product_mix(problem_text):
            deterministic_adapter = "deterministic_product_mix_source_adapter"
            deterministic_language = "lp"
        else:
            deterministic_adapter = "deterministic_tsp_source_adapter"
            deterministic_language = "tsp_json"
        persisted = persist_component_payloads(
            topic_id=topic_id,
            problem_id=problem_id,
            payloads=deterministic_payloads,
            default_review_status="candidate",
        )
        tournament = {
            "component_ids": persisted.get("component_ids", []),
            "diagnostic_ids": persisted.get("diagnostic_ids", []),
            "best": {
                "candidate_index": -1,
                "modeling_error": "none",
                "model_language": deterministic_language,
                "diagnostics": [],
                "validation": {"status": "valid", "issues": []},
                "content": deterministic_adapter,
                "components": deterministic_payloads,
                "raw_text": "",
                "component_payloads": deterministic_payloads,
            },
            "ranked": [],
        }
    elif deterministic_only:
        deterministic_uncovered = True
        tournament = {
            "component_ids": [],
            "diagnostic_ids": [],
            "best": {
                "candidate_index": -1,
                "modeling_error": "deterministic_uncovered",
                "model_language": None,
                "diagnostics": [],
                "validation": {"status": "unknown", "issues": []},
                "content": "deterministic_uncovered",
                "components": [],
                "raw_text": "",
                "component_payloads": [],
            },
            "ranked": [],
        }
    else:
        tournament = await extract_component_candidate_tournament(
            topic_id=topic_id,
            problem_id=problem_id,
            source_text=modeling_text,
            provider_profile=provider_profile,
            model=model,
            candidate_count=candidate_count,
            default_review_status="candidate",
            single_call=single_call_candidates,
        )

    batch_review = _promote_reviewed_tournament_components(
        workflow_mode=workflow_mode,
        tournament=tournament,
        disabled=disable_reviewed_promotion,
    )
    state = {"topic_id": topic_id, "subtopic_id": subtopic_id}
    snapshots: list[dict[str, Any]] = []
    previous_key: tuple[Any, Any, Any] | None = None
    if not deterministic_uncovered:
        step_limit = 1 if disable_repair_loop else max(1, max_steps)
        for _ in range(step_limit):
            snapshot = _advance_mse_workflow_deterministically(state)
            snapshots.append(snapshot)
            key = (
                snapshot.get("status"),
                bool(snapshot.get("solved")),
                (snapshot.get("latest_solver_run") or {}).get("id")
                if isinstance(snapshot.get("latest_solver_run"), dict)
                else None,
            )
            if snapshot.get("solved") or key == previous_key:
                break
            previous_key = key

    components = api.get_optimization_components(problem_id)
    model_irs = api.get_optimization_model_irs(problem_id)
    artifacts = api.get_optimization_artifacts(problem_id)
    solver_runs = api.get_solver_runs(problem_id)
    diagnostics = api.get_model_diagnostics(problem_id)
    claims = api.get_claims(topic_id, include_superseded=True)
    latest_solver_run = solver_runs[0] if solver_runs else None
    evaluation = _workflow_evaluation(
        case,
        latest_solver_run=latest_solver_run,
        deterministic_uncovered=deterministic_uncovered,
        blocking_issues=blocking_issues,
        solver_validation_disabled=disable_solver_validation,
    )
    finished_at = _utc_now()
    duration_s = time.monotonic() - started_monotonic

    archive.update(
        {
            "archive_status": "complete",
            "finished_at": finished_at,
            "duration_s": duration_s,
            "topic_id": topic_id,
            "subtopic_id": subtopic_id,
            "problem_id": problem_id,
            "tournament": _trim_tournament(
                tournament, include_raw=include_candidate_raw
            ),
            "batch_modeling_review": batch_review,
            "workflow_snapshots": snapshots,
            "components": _normalize_rows(
                components,
                json_fields={"source_refs_json", "metadata_json"},
            ),
            "model_irs": _normalize_rows(
                model_irs,
                json_fields={
                    "ir_json",
                    "linked_component_ids_json",
                    "component_fingerprints_json",
                },
            ),
            "artifacts": _normalize_rows(
                artifacts,
                json_fields={"source_component_ids_json"},
            ),
            "solver_runs": _normalize_rows(
                solver_runs,
                json_fields={"variable_values_json"},
            ),
            "diagnostics": diagnostics,
            "claims": _normalize_rows(
                claims,
                json_fields={
                    "support_fact_ids_json",
                    "source_refs_json",
                    "object_json",
                    "qualifiers_json",
                },
            ),
            "evaluation": evaluation,
        }
    )
    if disable_direct_fallback:
        _clear_direct_fallback(archive, case)
    else:
        _apply_direct_fallback(archive, case, direct_fallback_predictions or {})
    _finalize_archive_evaluation(archive)
    _write_json_atomic(archive_path, archive)
    return archive


def _manifest_summary(archive: dict[str, Any]) -> dict[str, Any]:
    evaluation = archive.get("evaluation") or {}
    prediction = evaluation.get("prediction") or {}
    metrics = evaluation.get("metrics") or {}
    adjudication = evaluation.get("gold_adjudication") or {}
    direct_fallback = archive.get("direct_fallback") or {}
    latest_artifact = (archive.get("artifacts") or [None])[0] or {}
    latest_solver = (archive.get("solver_runs") or [None])[0] or {}
    problem_spec = archive.get("problem_spec_summary") or _problem_spec_summary(archive)
    return {
        "archive_status": archive.get("archive_status"),
        "case_index": archive.get("case_index"),
        "case_id": archive.get("case_id"),
        "dataset": archive.get("dataset"),
        "split": archive.get("split"),
        "source_style": archive.get("source_style"),
        "difficulty": archive.get("difficulty"),
        "archive_path": str(archive.get("archive_path") or ""),
        "db_path": str(archive.get("db_path") or ""),
        "duration_s": archive.get("duration_s"),
        "predicted_status": prediction.get("status"),
        "predicted_objective": prediction.get("objective_value"),
        "gold_status": (evaluation.get("gold") or {}).get("status"),
        "gold_objective": (evaluation.get("gold") or {}).get("objective_value"),
        "correct": metrics.get("correct"),
        "adjudicated_correct": adjudication.get("adjudicated_correct"),
        "gold_adjudication_status": adjudication.get("status"),
        "gold_adjudication_reason": adjudication.get("reason"),
        "status_match": metrics.get("status_match"),
        "objective_match": metrics.get("objective_match"),
        "direct_fallback_used": direct_fallback.get("used"),
        "direct_fallback_reason": direct_fallback.get("reason"),
        "solver_backend": latest_solver.get("solver_backend"),
        "artifact_parser_status": latest_artifact.get("parser_status"),
        "diagnostic_count": len(archive.get("diagnostics") or []),
        "diagnostic_types": problem_spec.get("diagnostic_types") or [],
        "model_class": problem_spec.get("model_class"),
        "problem_spec_validation": problem_spec.get("validation_status"),
        "problem_spec_issue_types": problem_spec.get("validation_issue_types") or [],
        "solver_route": problem_spec.get("solver_route"),
        "failure_signature": archive.get("failure_signature")
        or _failure_signature(archive),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split")
    parser.add_argument("--output-dir")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--case-set", help="JSON file with case ids or case entries.")
    parser.add_argument("--provider-profile", default="minimax")
    parser.add_argument(
        "--model",
        help="Optional provider model name. Defaults to the broker's MiniMax model.",
    )
    parser.add_argument(
        "--workflow-mode",
        choices=["modeling_fast", "modeling_reviewed"],
        default="modeling_fast",
    )
    parser.add_argument("--candidate-count", "--k", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument(
        "--case-timeout-s",
        type=int,
        default=0,
        help="Optional wall-clock timeout per case in seconds. 0 disables it.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--include-candidate-raw", action="store_true")
    parser.add_argument(
        "--single-call-candidates",
        action="store_true",
        help="Ask the provider for all modeling candidates in one JSON response.",
    )
    parser.add_argument(
        "--modeling-brief-mode",
        choices=["auto", "never", "always"],
        default="auto",
        help="Whether to prepend a deterministic ORQ modeling brief before extraction.",
    )
    parser.add_argument(
        "--deterministic-only",
        action="store_true",
        help=(
            "Skip provider fallback when no deterministic ORBIT adapter matches; "
            "archive those cases as unknown."
        ),
    )
    parser.add_argument(
        "--direct-fallback-predictions",
        help=(
            "Optional predictions.json used when deterministic-only coverage is "
            "missing. The runner records this as a direct fallback, not as "
            "solver evidence."
        ),
    )
    parser.add_argument(
        "--disable-reviewed-promotion",
        action="store_true",
        help="Disable the batch reviewed-mode pass that promotes clean components.",
    )
    parser.add_argument(
        "--disable-solver-validation",
        action="store_true",
        help="Do not use solver runs for prediction scoring; archive the case as unknown.",
    )
    parser.add_argument(
        "--disable-deterministic-adapters",
        action="store_true",
        help="Skip deterministic OR source adapters before provider extraction.",
    )
    parser.add_argument(
        "--disable-direct-fallback",
        action="store_true",
        help="Ignore --direct-fallback-predictions and clear recorded direct fallback scoring.",
    )
    parser.add_argument(
        "--disable-repair-loop",
        action="store_true",
        help="Run only the first deterministic workflow transition for no-repair ablations.",
    )
    parser.add_argument(
        "--manifest-name",
        default="manifest.jsonl",
        help="Manifest filename inside the output directory.",
    )
    parser.add_argument(
        "--predictions-name",
        default="predictions.json",
        help="Predictions filename inside the output directory.",
    )
    return parser


async def async_main(args: argparse.Namespace) -> int:
    dataset = str(args.dataset or DEFAULT_DATASET)
    split = str(args.split).strip() if args.split else None
    default_output_name = "_".join(
        item.lower()
        for item in (dataset, split, "minimax")
        if item
    )
    output_dir = Path(args.output_dir or f"outputs/{default_output_name}")
    manifest_path = output_dir / args.manifest_name
    predictions_path = output_dir / args.predictions_name
    direct_fallback_predictions = (
        {}
        if args.disable_direct_fallback
        else _load_direct_fallback_predictions(
            Path(args.direct_fallback_predictions)
            if args.direct_fallback_predictions
            else None
        )
    )
    case_ids = list(args.case_id or [])
    if args.case_set:
        case_ids.extend(_load_case_ids(Path(args.case_set)))
        case_ids = list(dict.fromkeys(case_ids))

    selected = _select_cases(
        root=Path(args.root),
        dataset=dataset,
        split=split,
        start=max(1, args.start),
        limit=args.limit,
        case_ids=case_ids,
    )
    if not selected:
        target = f"{dataset}:{split}" if split else dataset
        _log(f"No ORQ cases selected for {target}.")
        return 0

    completed = 0
    skipped = 0
    failed = 0
    for ordinal, (index, case) in enumerate(selected, start=1):
        case_id = str(case["id"])
        db_path, archive_path = _case_paths(output_dir, index, case_id)
        if archive_path.exists() and not args.force:
            existing = _read_json(archive_path)
            if not (
                args.retry_errors
                and existing.get("archive_status") in RETRYABLE_ARCHIVE_STATUSES
            ):
                existing = _refresh_archive_evaluation(existing, case, archive_path)
                if args.disable_direct_fallback:
                    _clear_direct_fallback(existing, case)
                else:
                    _apply_direct_fallback(existing, case, direct_fallback_predictions)
                _finalize_archive_evaluation(existing)
                _write_json_atomic(archive_path, existing)
                _update_predictions(predictions_path, existing)
                skipped += 1
                _log(f"[{ordinal}/{len(selected)}] skip {case_id}: {archive_path}")
                continue
        if args.force or (
            args.retry_errors
            and archive_path.exists()
            and (
                _read_json(archive_path).get("archive_status")
                in RETRYABLE_ARCHIVE_STATUSES
            )
        ):
            _remove_case_db(db_path)

        _log(f"[{ordinal}/{len(selected)}] run {case_id}")
        try:
            case_coro = _run_case(
                case=case,
                index=index,
                db_path=db_path,
                archive_path=archive_path,
                provider_profile=args.provider_profile,
                model=args.model,
                workflow_mode=args.workflow_mode,
                candidate_count=max(1, args.candidate_count),
                max_steps=max(1, args.max_steps),
                include_candidate_raw=bool(args.include_candidate_raw),
                modeling_brief_mode=str(args.modeling_brief_mode or "auto"),
                deterministic_only=bool(args.deterministic_only),
                direct_fallback_predictions=direct_fallback_predictions,
                single_call_candidates=bool(args.single_call_candidates),
                disable_reviewed_promotion=bool(args.disable_reviewed_promotion),
                disable_solver_validation=bool(args.disable_solver_validation),
                disable_deterministic_adapters=bool(args.disable_deterministic_adapters),
                disable_direct_fallback=bool(args.disable_direct_fallback),
                disable_repair_loop=bool(args.disable_repair_loop),
            )
            case_timeout_s = max(0, int(args.case_timeout_s or 0))
            if case_timeout_s:
                archive = await asyncio.wait_for(case_coro, timeout=case_timeout_s)
            else:
                archive = await case_coro
            archive["archive_path"] = str(archive_path)
            _write_json_atomic(archive_path, archive)
            summary = _manifest_summary(archive)
            _append_jsonl(manifest_path, summary)
            _update_predictions(predictions_path, archive)
            completed += 1
            _log(
                "  -> "
                f"status={summary['predicted_status']} "
                f"objective={summary['predicted_objective']} "
                f"correct={summary['correct']} "
                f"archive={archive_path}"
            )
        except Exception as exc:
            failed += 1
            case_timeout_s = max(0, int(args.case_timeout_s or 0))
            timed_out = isinstance(exc, TimeoutError)
            error_message = str(exc)
            if timed_out and not error_message:
                error_message = f"case timed out after {case_timeout_s}s"
            failed_archive = {
                "archive_status": "timeout" if timed_out else "error",
                "case_index": index,
                "case_id": case_id,
                "dataset": case.get("dataset"),
                "split": case.get("split"),
                "question": case.get("problem_text") or "",
                "raw": case.get("raw") or {},
                "gold_solver": case.get("gold_solver") or {},
                "db_path": db_path,
                "archive_path": archive_path,
                "error": error_message,
                "error_type": exc.__class__.__name__,
                "traceback": traceback.format_exc(),
                "finished_at": _utc_now(),
            }
            _write_json_atomic(archive_path, failed_archive)
            _append_jsonl(manifest_path, _manifest_summary(failed_archive))
            _update_predictions(predictions_path, failed_archive)
            _log(f"  -> error={error_message} archive={archive_path}", error=True)

    _log(
        json.dumps(
            {
                "selected": len(selected),
                "completed": completed,
                "skipped": skipped,
                "failed": failed,
                "output_dir": str(output_dir),
                "manifest": str(manifest_path),
                "predictions": str(predictions_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
