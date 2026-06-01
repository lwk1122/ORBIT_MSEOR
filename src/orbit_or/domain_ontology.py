"""Domain ontology profiles for structured extraction and review."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any


BASE_ONTOLOGY: dict[str, Any] = {
    "profile": "base",
    "entity_types": {
        "fact": "Externally checkable factual statement.",
        "claim": "Scoped derived conclusion supported by evidence.",
        "evidence": "Source material, computation, or observation.",
    },
    "required_fields": ["entity_type", "natural_text"],
    "source_ref_required": False,
}

MSE_ONTOLOGY: dict[str, Any] = {
    "profile": "mse",
    "entity_types": {
        "decision_problem": "Decision context, owner, horizon, and action point.",
        "alternative": "Feasible option or policy under consideration.",
        "objective": "Optimization or decision criterion to minimize or maximize.",
        "kpi": "Managerial metric used for evaluation or monitoring.",
        "constraint": "Hard feasibility limit, policy, budget, or service bound.",
        "stakeholder": "Actor affected by or responsible for the decision.",
        "assumption": "Explicit modeling or data assumption.",
        "intervention": "Operational, policy, or managerial action.",
        "dataset": "Named data source, sample frame, or measurement table.",
        "sample": "Population, sample size, period, or observational unit.",
        "method": "Optimization, simulation, forecasting, or causal method.",
        "effect_size": "Quantified effect, lift, loss, or uncertainty interval.",
        "uncertainty": "Variance, confidence, sensitivity, or unresolved ambiguity.",
        "boundary_condition": "Scope limit, validity condition, or failure boundary.",
        "managerial_implication": "Decision-relevant implication or trade-off.",
        "set": "Optimization set or index domain.",
        "index": "Optimization index variable.",
        "parameter": "Numerical model input with unit and provenance.",
        "decision_variable": "Variable controlled by the decision maker.",
        "derived_variable": "Variable computed from decisions or parameters.",
        "data_requirement": "Missing data needed before execution or review.",
    },
    "component_type_aliases": {
        "optimization_method": "method",
        "forecasting_method": "method",
        "causal_method": "method",
        "simulation_method": "method",
        "dataset_field": "dataset",
        "managerial_boundary": "boundary_condition",
    },
    "required_fields": ["entity_type", "natural_text"],
    "source_ref_required": True,
}

ONTOLOGY_PROFILES = {
    "base": BASE_ONTOLOGY,
    "mse": MSE_ONTOLOGY,
    "management_science_engineering": MSE_ONTOLOGY,
}


def get_domain_ontology(profile: str | None = None) -> dict[str, Any]:
    normalized = (profile or "base").strip().lower()
    return deepcopy(ONTOLOGY_PROFILES.get(normalized, BASE_ONTOLOGY))


def normalize_ontology_entity_type(entity_type: str, profile: str | None = None) -> str:
    ontology = get_domain_ontology(profile)
    normalized = (entity_type or "").strip().lower()
    aliases = ontology.get("component_type_aliases") or {}
    return str(aliases.get(normalized, normalized))


def validate_ontology_payload(
    payload: dict[str, Any],
    *,
    profile: str | None = None,
) -> list[str]:
    ontology = get_domain_ontology(profile)
    required = ontology.get("required_fields") or []
    issues: list[str] = []
    for field in required:
        if not str(payload.get(field) or "").strip():
            issues.append(f"missing_{field}")

    entity_type = normalize_ontology_entity_type(
        str(payload.get("entity_type") or payload.get("component_type") or ""),
        profile,
    )
    if entity_type not in set(ontology.get("entity_types") or {}):
        issues.append("unsupported_entity_type")

    if ontology.get("source_ref_required"):
        source_refs = payload.get("source_refs") or payload.get("source_refs_json")
        has_source = False
        if isinstance(source_refs, list):
            has_source = bool(source_refs)
        elif source_refs:
            try:
                parsed = json.loads(str(source_refs))
                has_source = bool(parsed)
            except json.JSONDecodeError:
                has_source = bool(str(source_refs).strip())
        if not has_source:
            issues.append("missing_source_refs")
    return issues


def build_domain_ontology_prompt(profile: str | None = None) -> str:
    ontology = get_domain_ontology(profile)
    rows = [
        f"- {name}: {description}"
        for name, description in sorted((ontology.get("entity_types") or {}).items())
    ]
    return (
        "DOMAIN ONTOLOGY:\n"
        "Use these entity/component types when extracting structured domain data.\n"
        + "\n".join(rows)
    )
