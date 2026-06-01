from orbit_or.domain_ontology import (
    build_domain_ontology_prompt,
    get_domain_ontology,
    normalize_ontology_entity_type,
    validate_ontology_payload,
)
from orbit_or.domain_profiles import get_domain_prompt_additions
from orbit_or.optimization import validate_component_payload


def test_mse_domain_ontology_exposes_required_entity_types():
    ontology = get_domain_ontology("mse")

    for entity_type in [
        "decision_problem",
        "alternative",
        "objective",
        "kpi",
        "constraint",
        "stakeholder",
        "assumption",
        "intervention",
        "dataset",
        "sample",
        "method",
        "effect_size",
        "uncertainty",
        "boundary_condition",
        "managerial_implication",
    ]:
        assert entity_type in ontology["entity_types"]

    assert normalize_ontology_entity_type("forecasting_method", "mse") == "method"


def test_mse_ontology_validation_requires_source_refs():
    issues = validate_ontology_payload(
        {
            "entity_type": "dataset",
            "natural_text": "Demand observations for 2024.",
        },
        profile="mse",
    )

    assert "missing_source_refs" in issues
    assert validate_ontology_payload(
        {
            "entity_type": "dataset",
            "natural_text": "Demand observations for 2024.",
            "source_refs": ["D1"],
        },
        profile="mse",
    ) == []


def test_domain_profile_includes_ontology_prompt():
    additions = get_domain_prompt_additions("mse", "analyst")
    joined = "\n".join(additions)

    assert "DOMAIN ONTOLOGY" in joined
    assert "decision_problem" in joined
    assert "MSE ROLE: OR Modeler" in joined
    assert "managerial_implication" in build_domain_ontology_prompt("mse")


def test_optimization_component_validation_accepts_mse_ontology_types():
    issues = validate_component_payload(
        {
            "component_type": "dataset",
            "natural_text": "Demand observations for 2024.",
            "source_refs_json": '["D1"]',
        }
    )

    assert issues == []
