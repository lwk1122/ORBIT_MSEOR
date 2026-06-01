"""Management science and OR modeling primitives.

The functions here are deliberately deterministic. LLM prompts can call into
them later, but tests and VIKI checks should be able to validate artifacts
without provider access.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import math
import re
from typing import Any


COMPONENT_TYPES = frozenset(
    {
        "set",
        "index",
        "decision_problem",
        "alternative",
        "parameter",
        "decision_variable",
        "derived_variable",
        "objective",
        "constraint",
        "assumption",
        "intervention",
        "dataset",
        "sample",
        "method",
        "effect_size",
        "uncertainty",
        "boundary_condition",
        "managerial_implication",
        "data_requirement",
        "kpi",
        "stakeholder",
    }
)
LP_READY_REVIEW_STATUSES = frozenset({"reviewed", "formalized", "executable"})
COMPONENT_STALE_DIAGNOSTICS = frozenset(
    {
        "linked_component_inactive",
        "linked_component_changed",
        "linked_component_missing",
    }
)
MODELING_ERROR_TYPES = frozenset(
    {
        "semantic_misunderstanding",
        "objective_constraint_translation_error",
        "low_model_completeness",
        "none",
        "unknown_modeling_error",
    }
)

LP_SECTION_ALIASES = {
    "objective": {
        "minimize",
        "minimise",
        "minimization",
        "minimisation",
        "minimum",
        "min",
        "maximize",
        "maximise",
        "maximization",
        "maximisation",
        "maximum",
        "max",
    },
    "constraints": {"subject to", "such that", "s.t.", "st", "constraints"},
    "bounds": {"bounds", "bound"},
    "binary": {"binary", "binaries", "bin"},
    "general": {"general", "generals", "integer", "integers"},
    "end": {"end"},
}
LP_MAXIMIZE_ALIASES = {
    "maximize",
    "maximise",
    "maximization",
    "maximisation",
    "maximum",
    "max",
}


@dataclass(frozen=True)
class ValidationIssue:
    issue_type: str
    severity: str
    message: str


@dataclass(frozen=True)
class ParsedLinearProgram:
    direction: str
    variables: tuple[str, ...]
    objective: tuple[float, ...]
    constraints: tuple[dict[str, Any], ...]
    bounds: tuple[tuple[float | None, float | None], ...]
    integrality: tuple[int, ...]


def _parse_component_ids(raw: str | None) -> list[int]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    ids: list[int] = []
    for item in parsed:
        try:
            ids.append(int(item))
        except (TypeError, ValueError):
            continue
    return ids


def build_component_fingerprints_json(components: list[dict[str, Any]]) -> str:
    """Build a stable snapshot for detecting stale solver artifacts."""
    fields = (
        "id",
        "component_type",
        "natural_text",
        "formal_text",
        "symbol",
        "unit",
        "domain",
        "source_refs_json",
        "review_status",
    )
    fingerprints = []
    for component in sorted(components, key=lambda item: int(item.get("id") or 0)):
        fingerprints.append(
            {
                field: str(component.get(field) or "").strip()
                for field in fields
            }
        )
    return json.dumps(fingerprints, ensure_ascii=True, sort_keys=True)


def _component_fingerprint_index(raw: str | None) -> dict[int, dict[str, str]]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, list):
        return {}
    index: dict[int, dict[str, str]] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        try:
            component_id = int(item.get("id") or 0)
        except (TypeError, ValueError):
            continue
        if component_id:
            index[component_id] = {
                str(key): str(value or "").strip() for key, value in item.items()
            }
    return index


def _component_fingerprints_for_ids(
    problem_id: int, component_ids: list[int] | None
) -> str | None:
    if not component_ids:
        return None
    from . import api

    wanted = {int(component_id) for component_id in component_ids}
    components = [
        component
        for component in api.get_optimization_components(problem_id)
        if int(component.get("id") or 0) in wanted
    ]
    if not components:
        return None
    return build_component_fingerprints_json(components)


def _open_diagnostic_id(
    conn,
    *,
    problem_id: int,
    topic_id: int,
    diagnostic_type: str,
    component_id: int | None = None,
    artifact_id: int | None = None,
    solver_run_id: int | None = None,
) -> int | None:
    row = conn.execute(
        """
        SELECT id
        FROM ModelDiagnostic
        WHERE problem_id = ?
          AND topic_id = ?
          AND diagnostic_type = ?
          AND status = 'open'
          AND COALESCE(component_id, -1) = COALESCE(?, -1)
          AND COALESCE(artifact_id, -1) = COALESCE(?, -1)
          AND COALESCE(solver_run_id, -1) = COALESCE(?, -1)
        LIMIT 1
        """,
        (
            problem_id,
            topic_id,
            diagnostic_type,
            component_id,
            artifact_id,
            solver_run_id,
        ),
    ).fetchone()
    return int(row["id"]) if row else None


def _ensure_stale_diagnostic(
    conn,
    *,
    problem_id: int,
    topic_id: int,
    diagnostic_type: str,
    message: str,
    component_id: int | None = None,
    artifact_id: int | None = None,
    solver_run_id: int | None = None,
) -> int | None:
    existing_id = _open_diagnostic_id(
        conn,
        problem_id=problem_id,
        topic_id=topic_id,
        diagnostic_type=diagnostic_type,
        component_id=component_id,
        artifact_id=artifact_id,
        solver_run_id=solver_run_id,
    )
    if existing_id is not None:
        return None
    cursor = conn.execute(
        """
        INSERT INTO ModelDiagnostic (
            problem_id, topic_id, component_id, artifact_id, solver_run_id,
            diagnostic_type, severity, message, status
        )
        VALUES (?, ?, ?, ?, ?, ?, 'error', ?, 'open')
        """,
        (
            problem_id,
            topic_id,
            component_id,
            artifact_id,
            solver_run_id,
            diagnostic_type,
            message,
        ),
    )
    return int(cursor.lastrowid)


def validate_component_payload(payload: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    component_type = str(payload.get("component_type") or "").strip()
    natural_text = str(payload.get("natural_text") or "").strip()
    formal_text = str(payload.get("formal_text") or "").strip()
    symbol = str(payload.get("symbol") or "").strip()
    unit = str(payload.get("unit") or "").strip()

    if component_type not in COMPONENT_TYPES:
        issues.append(
            ValidationIssue(
                "unsupported_component_type",
                "error",
                f"Unsupported component_type: {component_type or '<missing>'}",
            )
        )
    if not natural_text:
        issues.append(
            ValidationIssue(
                "missing_natural_text",
                "error",
                "Component must preserve the source natural-language text.",
            )
        )
    if component_type in {"parameter", "decision_variable"} and not symbol:
        issues.append(
            ValidationIssue(
                "missing_symbol",
                "warning",
                f"{component_type} should have a stable symbol.",
            )
        )
    if component_type == "parameter" and re.search(r"\d", natural_text) and not unit:
        issues.append(
            ValidationIssue(
                "missing_unit",
                "warning",
                "Numeric parameters should carry an explicit unit when available.",
            )
        )
    if component_type in {"objective", "constraint"} and not formal_text:
        issues.append(
            ValidationIssue(
                "missing_formal_text",
                "warning",
                f"{component_type} is not yet formalized.",
            )
        )
    return issues


def _component_text_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        if not value:
            return None
        if all(not isinstance(item, (dict, list, tuple)) for item in value):
            text = ", ".join(str(item).strip() for item in value if str(item).strip())
            return text or None
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    except TypeError:
        text = str(value).strip()
        return text or None


def _has_unsupported_indexed_lp_notation(text: str) -> bool:
    return bool(
        re.search(
            r"(\u03a3|\u2211|\u2200|\u2208|\bforall\b|\bsum\s*[_({]|\bΣ\b|_\{[^}]*[,≤<∈])",
            text or "",
            flags=re.IGNORECASE,
        )
    )


def _has_source_refs(component: dict[str, Any]) -> bool:
    raw = component.get("source_refs_json")
    if raw is None:
        raw = component.get("source_refs")
    if isinstance(raw, list):
        return bool(raw)
    if not raw:
        return False
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return bool(str(raw).strip())
    return bool(parsed)


def _payload_symbol(payload: dict[str, Any]) -> str | None:
    symbol = _component_text_value(payload.get("symbol"))
    if symbol:
        return symbol
    if str(payload.get("component_type") or "") != "decision_variable":
        return None
    formal_text = _component_text_value(payload.get("formal_text"))
    if formal_text and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", formal_text):
        return formal_text
    return None


_OBJECTIVE_CONTEXT_COMPONENT_TYPES = frozenset(
    {"decision_problem", "kpi", "managerial_implication"}
)


def _objective_like_context_components(
    components: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for component in components:
        if component.get("component_type") not in _OBJECTIVE_CONTEXT_COMPONENT_TYPES:
            continue
        if not str(component.get("formal_text") or "").strip():
            continue
        if not _objective_direction_hint(component):
            continue
        try:
            _objective_line(component)
        except ValueError:
            continue
        candidates.append(component)
    return candidates


def _lp_relevant_components(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return solver-relevant components, promoting one misplaced objective if clear."""
    relevant = [
        component
        for component in components
        if component.get("component_type")
        in {"objective", "constraint", "decision_variable", "derived_variable"}
    ]
    if any(component.get("component_type") == "objective" for component in relevant):
        return relevant

    context_objectives = _objective_like_context_components(components)
    if len(context_objectives) != 1:
        return relevant

    promoted = dict(context_objectives[0])
    promoted["component_type"] = "objective"
    relevant.append(promoted)
    return relevant


def validate_lp_component_set(
    components: list[dict[str, Any]],
    *,
    require_reviewed: bool = True,
) -> list[ValidationIssue]:
    """Validate components before generating an LP artifact."""
    issues: list[ValidationIssue] = []
    objectives = [c for c in components if c.get("component_type") == "objective"]
    constraints = [c for c in components if c.get("component_type") == "constraint"]
    variables = [c for c in components if c.get("component_type") == "decision_variable"]

    if len(objectives) != 1:
        issues.append(
            ValidationIssue(
                "invalid_objective_count",
                "error",
                "LP generation requires exactly one objective component.",
            )
        )
    if not constraints:
        issues.append(
            ValidationIssue(
                "missing_constraints",
                "error",
                "LP generation requires at least one constraint component.",
            )
        )
    if not variables:
        issues.append(
            ValidationIssue(
                "missing_decision_variables",
                "error",
                "LP generation requires at least one decision variable component.",
            )
        )

    for component in components:
        component_type = component.get("component_type")
        if component_type not in {
            "objective",
            "constraint",
            "decision_variable",
            "derived_variable",
        }:
            continue
        if require_reviewed and component.get("review_status") not in LP_READY_REVIEW_STATUSES:
            issues.append(
                ValidationIssue(
                    "unreviewed_component",
                    "error",
                    f"Component {component.get('id')} is not reviewed for LP generation.",
                )
            )
        if component_type in {"objective", "constraint"} and not str(
            component.get("formal_text") or ""
        ).strip():
            issues.append(
                ValidationIssue(
                    "missing_formal_text",
                    "error",
                    f"{component_type} component {component.get('id')} lacks formal_text.",
                )
            )
        if component_type in {"objective", "constraint", "derived_variable"}:
            formal_text = str(component.get("formal_text") or "")
            if _has_unsupported_indexed_lp_notation(formal_text):
                issues.append(
                    ValidationIssue(
                        "indexed_formal_text_requires_scalarization",
                        "error",
                        (
                            f"{component_type} component {component.get('id')} uses "
                            "indexed, summation, or forall notation that must be "
                            "expanded to scalar LP/MILP algebra or handled by an "
                            "indexed model IR before LP generation."
                        ),
                    )
                )
        if component_type == "derived_variable" and not str(
            component.get("formal_text") or ""
        ).strip():
            issues.append(
                ValidationIssue(
                    "missing_formal_text",
                    "error",
                    f"derived_variable component {component.get('id')} lacks formal_text.",
                )
            )
        if component_type == "decision_variable" and not str(
            component.get("symbol") or ""
        ).strip():
            issues.append(
                ValidationIssue(
                    "missing_symbol",
                    "error",
                    f"decision_variable component {component.get('id')} lacks symbol.",
                )
            )
        if not _has_source_refs(component):
            issues.append(
                ValidationIssue(
                    "missing_source_refs",
                    "error",
                    f"Component {component.get('id')} lacks source_refs_json.",
                )
            )
    return issues


def validate_model_specification(
    components: list[dict[str, Any]],
    *,
    require_formal: bool = True,
) -> list[ValidationIssue]:
    """Check whether components are complete enough for solver-backed modeling."""
    relevant = _lp_relevant_components(components)
    issues = validate_lp_component_set(relevant, require_reviewed=False)
    if not require_formal:
        issues = [
            issue
            for issue in issues
            if issue.issue_type not in {"missing_formal_text", "missing_symbol"}
        ]
    issues.extend(validate_modeling_techniques(components))
    return issues


def blocking_specification_issues(
    components: list[dict[str, Any]],
    *,
    require_formal: bool = True,
) -> list[ValidationIssue]:
    """Return only errors that should stop artifact generation."""
    return [
        issue
        for issue in validate_model_specification(
            components,
            require_formal=require_formal,
        )
        if issue.severity == "error"
    ]


def _source_refs_list(component: dict[str, Any]) -> list[str]:
    raw = component.get("source_refs_json")
    if raw is None:
        raw = component.get("source_refs")
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item).strip()]
    if not raw:
        return []
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return [str(raw).strip()] if str(raw).strip() else []
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item).strip()]
    return []


def _component_ir_payload(component: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "id": component.get("id"),
        "component_type": component.get("component_type"),
        "symbol": component.get("symbol"),
        "natural_text": component.get("natural_text"),
        "formal_text": component.get("formal_text"),
        "unit": component.get("unit"),
        "domain": component.get("domain"),
        "review_status": component.get("review_status"),
        "source_refs": _source_refs_list(component),
    }
    return {key: value for key, value in payload.items() if value not in (None, "", [])}


def build_model_ir_from_components(
    *,
    problem: dict[str, Any] | None,
    components: list[dict[str, Any]],
    require_formal: bool = True,
) -> dict[str, Any]:
    """Build ORBIT's solver-agnostic model IR from reviewed component records."""
    issues = validate_model_specification(components, require_formal=require_formal)
    buckets = {
        "sets": [],
        "indices": [],
        "parameters": [],
        "decision_variables": [],
        "derived_variables": [],
        "objectives": [],
        "constraints": [],
        "assumptions": [],
        "data_requirements": [],
        "managerial_context": [],
    }
    bucket_by_type = {
        "set": "sets",
        "index": "indices",
        "parameter": "parameters",
        "decision_variable": "decision_variables",
        "derived_variable": "derived_variables",
        "objective": "objectives",
        "constraint": "constraints",
        "assumption": "assumptions",
        "data_requirement": "data_requirements",
        "decision_problem": "managerial_context",
        "alternative": "managerial_context",
        "kpi": "managerial_context",
        "stakeholder": "managerial_context",
        "intervention": "managerial_context",
        "dataset": "managerial_context",
        "sample": "managerial_context",
        "method": "managerial_context",
        "effect_size": "managerial_context",
        "uncertainty": "managerial_context",
        "boundary_condition": "managerial_context",
        "managerial_implication": "managerial_context",
    }
    linked_ids: list[int] = []
    for component in components:
        component_type = str(component.get("component_type") or "")
        bucket = bucket_by_type.get(component_type)
        if not bucket:
            continue
        buckets[bucket].append(_component_ir_payload(component))
        if component.get("id"):
            linked_ids.append(int(component["id"]))

    ir = {
        "schema": "orbit_model_ir.v1",
        "problem": {
            "id": (problem or {}).get("id"),
            "title": (problem or {}).get("title"),
            "problem_class": (problem or {}).get("problem_class"),
            "domain_context": (problem or {}).get("domain_context"),
            "stakeholder": (problem or {}).get("stakeholder"),
            "time_horizon": (problem or {}).get("time_horizon"),
            "source_refs": _source_refs_list(problem or {}),
        },
        "model": buckets,
        "validation": {
            "status": (
                "invalid"
                if any(issue.severity == "error" for issue in issues)
                else "valid"
            ),
            "issues": [issue.__dict__ for issue in issues],
        },
    }
    return {
        "accepted": ir["validation"]["status"] == "valid",
        "ir": ir,
        "linked_component_ids": linked_ids,
        "issues": ir["validation"]["issues"],
    }


def persist_model_ir_from_components(
    *,
    topic_id: int,
    problem: dict[str, Any],
    components: list[dict[str, Any]],
    require_formal: bool = True,
    generator_role: str = "component_ir_builder",
) -> dict[str, Any]:
    """Persist a solver-agnostic IR snapshot without turning it into a solver artifact."""
    from . import api

    built = build_model_ir_from_components(
        problem=problem,
        components=components,
        require_formal=require_formal,
    )
    ir_json = json.dumps(built["ir"], ensure_ascii=True, sort_keys=True)
    ir_id = api.insert_optimization_model_ir(
        problem_id=int(problem["id"]),
        topic_id=topic_id,
        ir_json=ir_json,
        status="valid" if built["accepted"] else "invalid",
        validation_notes="; ".join(
            str(issue.get("message") or "")
            for issue in built["issues"]
            if issue.get("severity") == "error"
        )
        or None,
        linked_component_ids_json=json.dumps(
            built["linked_component_ids"],
            ensure_ascii=True,
        ),
        component_fingerprints_json=build_component_fingerprints_json(components),
        generator_role=generator_role,
    )
    return {**built, "ir_id": ir_id}


def _clean_lp_line(text: str) -> str:
    cleaned = re.sub(r"```.*?```", "", text or "", flags=re.DOTALL)
    cleaned = (
        cleaned.replace("\r", "\n")
        .replace("\u00a0", " ")
        .replace("\u202f", " ")
        .replace("\u2003", " ")
        .replace("\u2264", "<=")
        .replace("\u2265", ">=")
        .replace("\u2212", "-")
        .replace("\u00b7", "*")
        .replace("\u00d7", "*")
        .strip()
    )
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[0]


def _format_number(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-10):
        return str(int(round(value)))
    return f"{value:.12g}"


def _format_linear_expression(coeffs: dict[str, float], constant: float = 0.0) -> str:
    terms: list[str] = []
    for var in sorted(coeffs):
        coeff = coeffs[var]
        if math.isclose(coeff, 0.0, abs_tol=1e-12):
            continue
        abs_coeff = abs(coeff)
        if math.isclose(abs_coeff, 1.0, abs_tol=1e-12):
            body = var
        else:
            body = f"{_format_number(abs_coeff)} {var}"
        if not terms:
            terms.append(f"-{body}" if coeff < 0 else body)
        else:
            sign = "-" if coeff < 0 else "+"
            terms.append(f"{sign} {body}")
    if not math.isclose(constant, 0.0, abs_tol=1e-12):
        body = _format_number(abs(constant))
        if not terms:
            terms.append(f"-{body}" if constant < 0 else body)
        else:
            sign = "-" if constant < 0 else "+"
            terms.append(f"{sign} {body}")
    return " ".join(terms) if terms else "0"


def _strip_objective_lhs(expr: str) -> str:
    """Turn `Z = 20 x` into `20 x` for LP objective expressions."""
    if "=" not in expr:
        return expr
    left, right = expr.split("=", 1)
    left = left.strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", left):
        return right.strip()
    return expr


def _parse_linear_expression_resolving_derived(
    expr: str,
    derived_variables: dict[str, tuple[dict[str, float], float]] | None = None,
) -> tuple[dict[str, float], float]:
    coeffs, constant = _parse_linear_expression(expr)
    if not derived_variables:
        return coeffs, constant

    resolved: dict[str, float] = {}
    resolved_constant = constant
    for var, coeff in coeffs.items():
        derived = derived_variables.get(var)
        if derived is None:
            resolved[var] = resolved.get(var, 0.0) + coeff
            continue
        derived_coeffs, derived_constant = derived
        resolved_constant += coeff * derived_constant
        for derived_var, derived_coeff in derived_coeffs.items():
            resolved[derived_var] = (
                resolved.get(derived_var, 0.0) + coeff * derived_coeff
            )
    return resolved, resolved_constant


def _derived_variable_expression_map(
    components: list[dict[str, Any]],
) -> dict[str, tuple[dict[str, float], float]]:
    decision_symbols = {
        str(component.get("symbol") or "").strip()
        for component in components
        if component.get("component_type") == "decision_variable"
        and str(component.get("symbol") or "").strip()
    }
    derived: dict[str, tuple[dict[str, float], float]] = {}
    for component in components:
        if component.get("component_type") != "derived_variable":
            continue
        text = _clean_lp_line(str(component.get("formal_text") or ""))
        symbol = str(component.get("symbol") or "").strip()
        name = symbol
        expr = text
        if "=" in text:
            left, right = text.split("=", 1)
            left = left.strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", left):
                name = left
                expr = right.strip()
        if not name or name in decision_symbols:
            continue
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            continue
        try:
            derived[name] = _parse_linear_expression(expr)
        except ValueError:
            continue
    return derived


def _normalize_objective_expression(
    expr: str,
    *,
    derived_variables: dict[str, tuple[dict[str, float], float]] | None = None,
) -> str:
    expr = _strip_objective_lhs(expr.strip())
    coeffs, constant = _parse_linear_expression_resolving_derived(
        expr, derived_variables
    )
    if not math.isclose(constant, 0.0, abs_tol=1e-12):
        raise ValueError("Objective constants are not supported")
    return _format_linear_expression(coeffs)


def _normalize_constraint_body(
    body: str,
    *,
    derived_variables: dict[str, tuple[dict[str, float], float]] | None = None,
) -> str:
    match = re.search(r"(<=|>=|=)", body)
    if not match:
        raise ValueError("Constraint formal_text lacks <=, >=, or =")
    op = match.group(1)
    left = body[: match.start()].strip()
    right = body[match.end() :].strip()
    left_coeffs, left_constant = _parse_linear_expression_resolving_derived(
        left, derived_variables
    )
    if _is_number(right):
        rhs = _parse_float(right) - left_constant
        return f"{_format_linear_expression(left_coeffs)} {op} {_format_number(rhs)}"

    right_coeffs, right_constant = _parse_linear_expression_resolving_derived(
        right, derived_variables
    )
    coeffs = dict(left_coeffs)
    for var, coeff in right_coeffs.items():
        coeffs[var] = coeffs.get(var, 0.0) - coeff
    rhs = right_constant - left_constant
    return f"{_format_linear_expression(coeffs)} {op} {_format_number(rhs)}"


def _objective_line(
    component: dict[str, Any],
    *,
    derived_variables: dict[str, tuple[dict[str, float], float]] | None = None,
) -> tuple[str, str]:
    text = _clean_lp_line(str(component.get("formal_text") or ""))
    lowered = _canonical_line(text)
    direction = "minimize"
    expr = text
    for alias in LP_SECTION_ALIASES["objective"]:
        if lowered == alias or lowered.startswith(alias + " "):
            direction = "maximize" if alias in LP_MAXIMIZE_ALIASES else "minimize"
            expr = text[len(alias) :].strip()
            break
    if ":" in expr:
        expr = expr.split(":", 1)[-1].strip()
    if not expr:
        raise ValueError("Objective formal_text does not contain an expression")
    normalized = _normalize_objective_expression(
        expr, derived_variables=derived_variables
    )
    return direction, f" obj: {normalized}"


def _constraint_line(
    component: dict[str, Any],
    fallback_index: int,
    *,
    derived_variables: dict[str, tuple[dict[str, float], float]] | None = None,
) -> str:
    text = _clean_lp_line(str(component.get("formal_text") or ""))
    label = str(component.get("symbol") or f"c{fallback_index}").strip()
    if ":" in text:
        raw_label, body = text.split(":", 1)
        safe_label = (
            re.sub(r"[^A-Za-z0-9_]", "_", raw_label.strip())
            or f"c{fallback_index}"
        )
        normalized = _normalize_constraint_body(
            body.strip(), derived_variables=derived_variables
        )
        return f" {safe_label}: {normalized}"
    safe_label = re.sub(r"[^A-Za-z0-9_]", "_", label) or f"c{fallback_index}"
    normalized = _normalize_constraint_body(text, derived_variables=derived_variables)
    return f" {safe_label}: {normalized}"


_DOMAIN_WORDS = {
    "integer",
    "integers",
    "general",
    "binary",
    "binaries",
    "nonnegative",
    "non-negativity",
    "nonnegativity",
}


def _domain_overrides_from_constraints(
    constraints: list[dict[str, Any]],
) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for component in constraints:
        text = _canonical_line(
            " ".join(
                str(component.get(field) or "")
                for field in ("natural_text", "formal_text", "domain")
            )
        )
        if not any(word in text for word in _DOMAIN_WORDS):
            continue
        formal = str(component.get("formal_text") or "")
        variables = [
            token
            for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", formal)
            if token.lower()
            not in {
                "integer",
                "integers",
                "general",
                "binary",
                "binaries",
                "nonnegative",
                "non",
                "negativity",
            }
        ]
        if any(word in text for word in ("binary", "binaries", "{0,1}", "0/1")):
            for var in variables:
                overrides[var] = "binary"
            continue
        if any(word in text for word in ("integer", "integers", "general")):
            for var in variables:
                overrides.setdefault(var, "integer")
    return overrides


def _is_domain_only_constraint(component: dict[str, Any]) -> bool:
    text = _canonical_line(
        " ".join(
            str(component.get(field) or "")
            for field in ("natural_text", "formal_text", "domain")
        )
    )
    if not any(word in text for word in _DOMAIN_WORDS):
        return False
    formal = str(component.get("formal_text") or "")
    formal_text = _canonical_line(formal)
    if (
        re.search(r"\b(integer|integers|general|binary|binaries)\b", formal_text)
        and not re.search(r"(<=|>=|=)", formal_text)
    ):
        return True
    clauses = [clause.strip() for clause in re.split(r"[,;]", formal) if clause.strip()]
    if clauses:
        domain_clause_count = sum(
            1
            for clause in clauses
            if re.search(r"\b(integer|integers|general|binary|binaries)\b", clause, re.I)
        )
        if all(
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\s*>=\s*0", clause)
            or re.search(r"\b(integer|integers|general|binary|binaries)\b", clause, re.I)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", clause)
            for clause in clauses
        ) and domain_clause_count:
            return True
        if all(
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\s*>=\s*0", clause)
            for clause in clauses
        ):
            return True
    return False


_DISCRETE_COUNT_NOUN_RE = re.compile(
    r"\b("
    r"pill|pills|capsule|capsules|tablet|tablets|"
    r"appliance|appliances|machine|machines|freezer|freezers|"
    r"vehicle|vehicles|ambulance|ambulances|van|vans|truck|trucks|"
    r"shift|shifts|trip|trips|container|containers|"
    r"boat|boats|canoe|canoes|worker|workers|employee|employees|"
    r"item|items|product|products"
    r")\b",
    flags=re.IGNORECASE,
)


def _looks_like_discrete_count_variable(component: dict[str, Any]) -> bool:
    text = " ".join(
        str(component.get(field) or "")
        for field in ("natural_text", "unit", "symbol")
    )
    lowered = text.lower()
    if not re.search(r"\b(number|count|integer|whole)\b", lowered):
        return False
    if re.search(r"\b(fractional|continuous|relaxation|proportion|percentage|rate)\b", lowered):
        return False
    return bool(_DISCRETE_COUNT_NOUN_RE.search(text))


def _variable_domain_sections(
    components: list[dict[str, Any]],
    *,
    domain_overrides: dict[str, str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    bounds: list[str] = []
    binary: list[str] = []
    general: list[str] = []
    overrides = domain_overrides or {}
    for component in components:
        if component.get("component_type") != "decision_variable":
            continue
        symbol = str(component.get("symbol") or "").strip()
        domain = str(component.get("domain") or component.get("formal_text") or "").lower()
        if not symbol:
            continue
        if _looks_like_discrete_count_variable(component):
            domain = f"{domain} integer".strip()
        domain = f"{domain} {overrides.get(symbol, '')}".strip()
        if any(token in domain for token in ("binary", "{0,1}", "0/1", "bool")):
            binary.append(symbol)
            continue
        if any(token in domain for token in ("integer", "general")):
            general.append(symbol)
            bounds.append(f" {symbol} >= 0")
            continue
        if "free" in domain:
            bounds.append(f" {symbol} free")
        else:
            bounds.append(f" {symbol} >= 0")
    return bounds, binary, general


def _indexed_issue(issue_type: str, message: str, severity: str = "error") -> dict[str, str]:
    return {"issue_type": issue_type, "severity": severity, "message": message}


def _indexed_set_values(raw: Any) -> list[Any]:
    if isinstance(raw, dict):
        if "values" in raw:
            return _indexed_set_values(raw.get("values"))
        if {"start", "end"} <= set(raw):
            start = int(raw["start"])
            end = int(raw["end"])
            step = 1 if end >= start else -1
            return list(range(start, end + step, step))
    if isinstance(raw, list):
        return list(raw)
    return []


def _indexed_index_set_names(raw: Any) -> list[str]:
    if isinstance(raw, dict):
        return [str(value) for value in raw.values()]
    if isinstance(raw, list):
        return [str(value) for value in raw]
    return []


def _indexed_scalar_suffix(values: list[Any]) -> str:
    return "_".join(re.sub(r"[^A-Za-z0-9_]", "_", str(value)).strip("_") for value in values)


def _indexed_scalar_name(symbol: str, subscript_values: list[Any]) -> str:
    suffix = _indexed_scalar_suffix(subscript_values)
    base = re.sub(r"[^A-Za-z0-9_]", "_", str(symbol or "")).strip("_")
    return f"{base}_{suffix}" if suffix else base


def _indexed_param_key(values: list[Any]) -> str:
    return ",".join(str(value) for value in values)


def _indexed_parameter_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        parameter_name = value.get("parameter") or value.get("param")
        if parameter_name:
            refs.add(str(parameter_name))
        for item in value.values():
            refs.update(_indexed_parameter_refs(item))
    elif isinstance(value, list):
        for item in value:
            refs.update(_indexed_parameter_refs(item))
    return refs


_CRITICAL_INDEXED_PARAMETER_RE = re.compile(
    r"("
    r"target|required|requirement|minimum|maximum|min|max|"
    r"demand|capacity|limit|duration|deadline|initial|available"
    r")",
    flags=re.IGNORECASE,
)


def _indexed_nested_value(raw_values: Any, subscript_values: list[Any]) -> float | None:
    current = raw_values
    for value in subscript_values:
        if not isinstance(current, dict):
            return None
        key = str(value)
        if key in current:
            current = current[key]
        elif value in current:
            current = current[value]
        else:
            return None
    if isinstance(current, (int, float)):
        return float(current)
    return None


def _indexed_numeric_ref(
    value: Any,
    *,
    parameters: dict[str, Any],
    context: dict[str, Any],
) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if value in context:
            return float(context[value])
        if _is_number(value):
            return _parse_float(value)
        parameter = parameters.get(value)
        if isinstance(parameter, (int, float)):
            return float(parameter)
        if isinstance(parameter, dict) and isinstance(parameter.get("value"), (int, float)):
            return float(parameter["value"])
    if isinstance(value, dict):
        if "value" in value and isinstance(value["value"], (int, float)):
            return float(value["value"])
        parameter_name = value.get("parameter") or value.get("param")
        if parameter_name:
            parameter = parameters.get(str(parameter_name))
            if isinstance(parameter, (int, float)):
                return float(parameter)
            if not isinstance(parameter, dict):
                raise ValueError(f"Unknown indexed parameter: {parameter_name}")
            raw_values = parameter.get("values", parameter.get("value"))
            if raw_values is None:
                raw_values = {
                    key: item
                    for key, item in parameter.items()
                    if key not in {"description", "unit", "source_refs"}
                }
            if isinstance(raw_values, (int, float)):
                return float(raw_values)
            subscripts = value.get("subscripts") or value.get("indices") or []
            subscript_values = [context.get(str(item), item) for item in subscripts]
            if isinstance(raw_values, list) and len(subscript_values) == 1:
                index = int(subscript_values[0]) - 1
                return float(raw_values[index])
            if isinstance(raw_values, dict):
                key = _indexed_param_key(subscript_values)
                if key in raw_values:
                    return float(raw_values[key])
                if len(subscript_values) == 1 and str(subscript_values[0]) in raw_values:
                    return float(raw_values[str(subscript_values[0])])
                nested_value = _indexed_nested_value(raw_values, subscript_values)
                if nested_value is not None:
                    return nested_value
            raise ValueError(f"Missing indexed parameter value: {parameter_name}{subscript_values}")
    raise ValueError(f"Unsupported indexed numeric reference: {value!r}")


def _indexed_iter_contexts(
    sum_over: dict[str, str],
    *,
    sets: dict[str, list[Any]],
    base_context: dict[str, Any],
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    contexts = [dict(base_context)]
    for index_name, set_name in sum_over.items():
        values = sets.get(str(set_name))
        if values is None:
            raise ValueError(f"Unknown indexed set: {set_name}")
        contexts = [
            {**context, str(index_name): value}
            for context in contexts
            for value in values
        ]

    def allowed(context: dict[str, Any]) -> bool:
        for left, comparisons in (where or {}).items():
            left_value = context.get(str(left), left)
            if not isinstance(comparisons, dict):
                continue
            for op, raw_right in comparisons.items():
                right_value = context.get(str(raw_right), raw_right)
                left_num = float(left_value)
                right_num = float(right_value)
                if op == "<=" and not left_num <= right_num:
                    return False
                if op == "<" and not left_num < right_num:
                    return False
                if op == ">=" and not left_num >= right_num:
                    return False
                if op == ">" and not left_num > right_num:
                    return False
                if op in {"=", "=="} and not math.isclose(left_num, right_num):
                    return False
        return True

    return [context for context in contexts if allowed(context)]


def _indexed_add_coeffs(target: dict[str, float], source: dict[str, float]) -> None:
    for variable, coefficient in source.items():
        target[variable] = target.get(variable, 0.0) + coefficient


def _indexed_expression(
    expression: Any,
    *,
    sets: dict[str, list[Any]],
    parameters: dict[str, Any],
    variables: dict[str, dict[str, Any]],
    context: dict[str, Any],
) -> tuple[dict[str, float], float]:
    if isinstance(expression, (int, float, str)) or (
        isinstance(expression, dict) and ("parameter" in expression or "param" in expression)
    ):
        return {}, _indexed_numeric_ref(expression, parameters=parameters, context=context)

    if not isinstance(expression, dict):
        raise ValueError(f"Unsupported indexed expression: {expression!r}")
    if isinstance(expression.get("formula"), dict):
        return _indexed_expression(
            expression["formula"],
            sets=sets,
            parameters=parameters,
            variables=variables,
            context=context,
        )

    coeffs: dict[str, float] = {}
    terms = expression.get("terms") or []
    raw_constant = expression.get("constant")
    if raw_constant is None and not terms:
        raw_constant = expression.get("expression")
    constant = float(raw_constant or 0.0)
    for term in terms:
        if not isinstance(term, dict):
            raise ValueError(f"Unsupported indexed term: {term!r}")
        sum_over = term.get("sum_over") or term.get("sum") or {}
        if sum_over:
            for local_context in _indexed_iter_contexts(
                {str(key): str(value) for key, value in dict(sum_over).items()},
                sets=sets,
                base_context=context,
                where=term.get("where"),
            ):
                inner = dict(term)
                inner.pop("sum_over", None)
                inner.pop("sum", None)
                inner.pop("where", None)
                inner_coeffs, inner_constant = _indexed_expression(
                    {"terms": [inner]},
                    sets=sets,
                    parameters=parameters,
                    variables=variables,
                    context=local_context,
                )
                _indexed_add_coeffs(coeffs, inner_coeffs)
                constant += inner_constant
            continue

        coefficient = _indexed_numeric_ref(
            term.get("coefficient", term.get("coef", 1.0)),
            parameters=parameters,
            context=context,
        )
        variable = term.get("variable") or term.get("var")
        if not variable:
            constant += coefficient
            continue
        symbol = str(variable)
        declaration = variables.get(symbol)
        subscripts = term.get("subscripts") or term.get("indices") or []
        if declaration is None and not subscripts:
            scalar_name = _indexed_scalar_name(symbol, [])
        else:
            subscript_values = [context.get(str(item), item) for item in subscripts]
            scalar_name = _indexed_scalar_name(symbol, subscript_values)
        coeffs[scalar_name] = coeffs.get(scalar_name, 0.0) + coefficient
    return coeffs, constant


def _indexed_declared_variables(
    variables: dict[str, dict[str, Any]],
    sets: dict[str, list[Any]],
) -> tuple[list[str], list[str], list[str]]:
    bounds: list[str] = []
    binary: list[str] = []
    general: list[str] = []
    for symbol, declaration in variables.items():
        index_sets = _indexed_index_set_names(
            declaration.get("indices") or declaration.get("indexes") or []
        )
        contexts = [{}]
        for set_name in index_sets:
            values = sets.get(str(set_name))
            if values is None:
                raise ValueError(f"Unknown indexed set: {set_name}")
            contexts = [
                {**context, str(set_name): value}
                for context in contexts
                for value in values
            ]
        if not index_sets:
            contexts = [{}]
        domain = str(declaration.get("domain") or "").lower()
        for context in contexts:
            values = [context[str(set_name)] for set_name in index_sets]
            scalar = _indexed_scalar_name(symbol, values)
            if any(token in domain for token in ("binary", "{0,1}", "0/1", "bool")):
                binary.append(scalar)
                continue
            bounds.append(f" {scalar} >= 0")
            if any(token in domain for token in ("integer", "general")):
                general.append(scalar)
    return bounds, binary, general


def build_lp_artifact_from_indexed_ir(indexed_ir: dict[str, Any]) -> dict[str, Any]:
    """Scalarize a finite indexed LP/MILP IR into an LP artifact."""
    issues: list[dict[str, str]] = []
    status = str(indexed_ir.get("status") or "complete").lower()
    if status not in {"complete", "ready", "valid"}:
        issues.append(
            _indexed_issue(
                "indexed_ir_incomplete",
                "Indexed model IR is marked incomplete or insufficient.",
            )
        )
        return {
            "accepted": False,
            "content": "",
            "issues": issues,
            "validation": {"status": "invalid", "issues": issues},
        }
    sets = {
        str(name): _indexed_set_values(raw)
        for name, raw in (indexed_ir.get("sets") or {}).items()
    }
    parameters = dict(indexed_ir.get("parameters") or {})
    variable_declarations = {
        str(item.get("symbol") or item.get("name")): item
        for item in (indexed_ir.get("variables") or [])
        if isinstance(item, dict) and (item.get("symbol") or item.get("name"))
    }
    try:
        empty_sets = [name for name, values in sets.items() if not values]
        if empty_sets:
            raise ValueError(f"Indexed sets must be finite and nonempty: {empty_sets}")
        if not variable_declarations:
            raise ValueError("Indexed model IR has no decision variables")
        objective = indexed_ir.get("objective") or {}
        if not objective.get("terms"):
            raise ValueError("Indexed model IR has no objective terms")
        if not indexed_ir.get("constraints"):
            raise ValueError("Indexed model IR has no constraints")
        parameter_refs = _indexed_parameter_refs(
            {
                "objective": objective,
                "constraints": indexed_ir.get("constraints") or [],
            }
        )
        unused_critical = [
            name
            for name in parameters
            if name not in parameter_refs and _CRITICAL_INDEXED_PARAMETER_RE.search(name)
        ]
        if unused_critical:
            raise ValueError(
                "Critical indexed parameters are not used in the model: "
                + ", ".join(sorted(unused_critical))
            )
        direction = str(objective.get("sense") or objective.get("direction") or "minimize").lower()
        objective_coeffs, objective_constant = _indexed_expression(
            objective,
            sets=sets,
            parameters=parameters,
            variables=variable_declarations,
            context={},
        )
        if not math.isclose(objective_constant, 0.0, abs_tol=1e-12):
            raise ValueError("Indexed LP objective constants are not supported")

        constraint_lines: list[str] = []
        for constraint in indexed_ir.get("constraints") or []:
            if not isinstance(constraint, dict):
                raise ValueError(f"Unsupported indexed constraint: {constraint!r}")
            for_each = {
                str(key): str(value)
                for key, value in dict(constraint.get("for_each") or {}).items()
            }
            contexts = _indexed_iter_contexts(
                for_each,
                sets=sets,
                base_context={},
                where=constraint.get("where"),
            ) if for_each else [{}]
            for context in contexts:
                raw_lhs = constraint.get("lhs")
                if constraint.get("terms") is not None:
                    lhs_expression = {"terms": constraint.get("terms") or []}
                elif isinstance(raw_lhs, dict):
                    lhs_expression = raw_lhs
                else:
                    lhs_expression = {"terms": raw_lhs or []}
                left_coeffs, left_constant = _indexed_expression(
                    lhs_expression,
                    sets=sets,
                    parameters=parameters,
                    variables=variable_declarations,
                    context=context,
                )
                right_coeffs, right_constant = _indexed_expression(
                    constraint.get("rhs", 0.0),
                    sets=sets,
                    parameters=parameters,
                    variables=variable_declarations,
                    context=context,
                )
                coeffs = dict(left_coeffs)
                for variable, coefficient in right_coeffs.items():
                    coeffs[variable] = coeffs.get(variable, 0.0) - coefficient
                rhs = right_constant - left_constant
                name_values = [context[key] for key in for_each if key in context]
                label = _indexed_scalar_name(
                    str(constraint.get("name") or "c"),
                    name_values,
                )
                sense = str(constraint.get("sense") or constraint.get("operator") or "<=")
                constraint_lines.append(
                    f" {label}: {_format_linear_expression(coeffs)} {sense} {_format_number(rhs)}"
                )
        bounds, binary, general = _indexed_declared_variables(variable_declarations, sets)
    except Exception as exc:
        issues.append(_indexed_issue("indexed_scalarization_error", str(exc)))
        return {
            "accepted": False,
            "content": "",
            "issues": issues,
            "validation": {"status": "invalid", "issues": issues},
        }

    lines = [
        "Maximize" if direction.startswith("max") else "Minimize",
        f" obj: {_format_linear_expression(objective_coeffs)}",
        "Subject To",
        *constraint_lines,
    ]
    if bounds:
        lines.extend(["Bounds", *bounds])
    if binary:
        lines.extend(["Binary", " " + " ".join(binary)])
    if general:
        lines.extend(["General", " " + " ".join(general)])
    lines.append("End")
    content = "\n".join(lines)
    validation = validate_lp_artifact(content)
    accepted = validation["status"] == "valid"
    return {
        "accepted": accepted,
        "content": content,
        "issues": [] if accepted else validation.get("issues", []),
        "validation": validation,
    }


def persist_lp_artifact_from_indexed_ir(
    *,
    topic_id: int,
    problem_id: int,
    indexed_ir: dict[str, Any],
    generator_role: str = "indexed_ir_scalarizer",
) -> dict[str, Any]:
    """Scalarize and persist an LP artifact from a finite indexed IR."""
    from . import api

    generated = build_lp_artifact_from_indexed_ir(indexed_ir)
    diagnostic_ids: list[int] = []
    if not generated["accepted"]:
        for issue in generated.get("issues", []):
            diagnostic_ids.append(
                api.insert_model_diagnostic(
                    problem_id=problem_id,
                    topic_id=topic_id,
                    diagnostic_type=str(issue.get("issue_type") or "indexed_scalarization_error"),
                    severity=str(issue.get("severity") or "error"),
                    message=str(issue.get("message") or issue),
                )
            )
        return {
            **generated,
            "artifact_id": None,
            "diagnostic_ids": diagnostic_ids,
        }

    persisted = persist_lp_artifact(
        topic_id=topic_id,
        problem_id=problem_id,
        content=generated["content"],
        linked_component_ids=[],
        generator_role=generator_role,
    )
    return {
        **generated,
        "artifact_id": persisted["artifact_id"],
        "diagnostic_ids": persisted["diagnostic_ids"],
    }


def build_lp_artifact_from_components(
    components: list[dict[str, Any]],
    *,
    require_reviewed: bool = True,
) -> dict[str, Any]:
    """Generate LP text from reviewed objective/constraint/variable components."""
    relevant = _lp_relevant_components(components)
    issues = validate_lp_component_set(relevant, require_reviewed=require_reviewed)
    if any(issue.severity == "error" for issue in issues):
        return {
            "accepted": False,
            "content": "",
            "component_ids": [
                int(component["id"]) for component in relevant if component.get("id")
            ],
            "issues": [issue.__dict__ for issue in issues],
            "validation": {"status": "invalid", "issues": []},
        }

    objectives = [c for c in relevant if c.get("component_type") == "objective"]
    raw_constraints = [c for c in relevant if c.get("component_type") == "constraint"]
    constraints = [
        component
        for component in raw_constraints
        if not _is_domain_only_constraint(component)
    ]
    variables = [c for c in relevant if c.get("component_type") == "decision_variable"]
    derived_variables = _derived_variable_expression_map(components)

    try:
        direction, objective = _objective_line(
            objectives[0], derived_variables=derived_variables
        )
        constraint_lines = [
            _constraint_line(
                component,
                index,
                derived_variables=derived_variables,
            )
            for index, component in enumerate(constraints, start=1)
        ]
    except ValueError as exc:
        issue = ValidationIssue("invalid_formal_text", "error", str(exc))
        return {
            "accepted": False,
            "content": "",
            "component_ids": [
                int(component["id"]) for component in relevant if component.get("id")
            ],
            "issues": [issue.__dict__],
            "validation": {"status": "invalid", "issues": []},
        }

    bounds, binary, general = _variable_domain_sections(
        variables,
        domain_overrides=_domain_overrides_from_constraints(raw_constraints),
    )
    lines = [
        "Maximize" if direction == "maximize" else "Minimize",
        objective,
        "Subject To",
        *constraint_lines,
    ]
    if bounds:
        lines.extend(["Bounds", *bounds])
    if binary:
        lines.extend(["Binary", " " + " ".join(binary)])
    if general:
        lines.extend(["General", " " + " ".join(general)])
    lines.append("End")
    content = "\n".join(lines)
    validation = validate_lp_artifact(content)
    technique_issues = (
        validate_modeling_techniques(relevant, content=content)
        if validation["status"] == "valid"
        else []
    )
    if any(issue.severity == "error" for issue in technique_issues):
        validation = {
            "status": "invalid",
            "issues": [
                *validation.get("issues", []),
                *(issue.__dict__ for issue in technique_issues),
            ],
        }
    accepted = validation["status"] == "valid"
    return {
        "accepted": accepted,
        "content": content,
        "component_ids": [
            int(component["id"]) for component in relevant if component.get("id")
        ],
        "issues": [] if accepted else validation.get("issues", []),
        "validation": validation,
    }


def persist_lp_artifact_from_components(
    *,
    topic_id: int,
    problem_id: int,
    components: list[dict[str, Any]],
    require_reviewed: bool = True,
    generator_role: str = "component_lp_generator",
) -> dict[str, Any]:
    """Generate and persist an LP artifact from reviewed components."""
    from . import api

    generated = build_lp_artifact_from_components(
        components, require_reviewed=require_reviewed
    )
    diagnostic_ids: list[int] = []
    if not generated["accepted"]:
        for issue in generated.get("issues", []):
            component_id = None
            match = re.search(r"Component\s+(\d+)", str(issue.get("message") or ""))
            if match:
                component_id = int(match.group(1))
            diagnostic_ids.append(
                api.insert_model_diagnostic(
                    problem_id=problem_id,
                    topic_id=topic_id,
                    component_id=component_id,
                    diagnostic_type=str(issue.get("issue_type") or "lp_generation_error"),
                    severity=str(issue.get("severity") or "error"),
                    message=str(issue.get("message") or issue),
                )
            )
        return {
            **generated,
            "artifact_id": None,
            "diagnostic_ids": diagnostic_ids,
        }

    persisted = persist_lp_artifact(
        topic_id=topic_id,
        problem_id=problem_id,
        content=generated["content"],
        linked_component_ids=generated["component_ids"],
        generator_role=generator_role,
    )
    return {
        **generated,
        "artifact_id": persisted["artifact_id"],
        "diagnostic_ids": persisted["diagnostic_ids"],
    }


def _mps_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", value or "").strip("_").upper()
    if not cleaned:
        cleaned = fallback
    return cleaned[:16]


def _constraint_mps_name(constraint: dict[str, Any], index: int) -> str:
    raw = str(constraint.get("raw") or "")
    if ":" in raw:
        return _mps_name(raw.split(":", 1)[0], f"C{index}")
    return f"C{index}"


def build_mps_artifact_from_lp(content: str, *, name: str = "ORBITMODEL") -> dict[str, Any]:
    """Convert a valid LP artifact into a conservative fixed-free MPS artifact."""
    try:
        parsed = parse_lp_artifact(content)
    except Exception as exc:
        issue = {"issue_type": "invalid_lp_source", "severity": "error", "message": str(exc)}
        return {
            "accepted": False,
            "content": "",
            "validation": {"status": "invalid", "issues": [issue]},
            "issues": [issue],
        }

    model_name = _mps_name(name, "ORBITMODEL")
    constraint_names = [
        _constraint_mps_name(constraint, index)
        for index, constraint in enumerate(parsed.constraints, start=1)
    ]
    row_codes = {"<=": "L", ">=": "G", "=": "E"}
    lines = [
        "OBJSENSE",
        " MAX" if parsed.direction == "maximize" else " MIN",
        f"NAME          {model_name}",
        "ROWS",
        " N  OBJ",
    ]
    for constraint, cname in zip(parsed.constraints, constraint_names):
        lines.append(f" {row_codes[constraint['operator']]}  {cname}")

    lines.append("COLUMNS")
    for var_index, var in enumerate(parsed.variables):
        if parsed.integrality[var_index]:
            lines.append(f"    MARK{var_index:04d}  'MARKER'                 'INTORG'")
        coeff = parsed.objective[var_index]
        if not math.isclose(coeff, 0.0):
            lines.append(f"    {var:<16}  OBJ       {coeff:.12g}")
        for constraint, cname in zip(parsed.constraints, constraint_names):
            value = constraint["coeffs"].get(var, 0.0)
            if not math.isclose(value, 0.0):
                lines.append(f"    {var:<16}  {cname:<8}  {value:.12g}")
        if parsed.integrality[var_index]:
            lines.append(f"    MARK{var_index:04d}  'MARKER'                 'INTEND'")

    lines.append("RHS")
    for constraint, cname in zip(parsed.constraints, constraint_names):
        lines.append(f"    RHS1              {cname:<8}  {constraint['rhs']:.12g}")

    lines.append("BOUNDS")
    for var_index, (var, (low, high)) in enumerate(zip(parsed.variables, parsed.bounds)):
        is_integer = parsed.integrality[var_index] == 1
        if is_integer and low == 0.0 and high == 1.0:
            lines.append(f" BV BND       {var}")
            continue
        if low is None and high is None:
            lines.append(f" FR BND       {var}")
            continue
        if low is not None:
            lines.append(f" {'LI' if is_integer else 'LO'} BND       {var:<16}  {low:.12g}")
        if high is not None:
            lines.append(f" {'UI' if is_integer else 'UP'} BND       {var:<16}  {high:.12g}")
    lines.append("ENDATA")

    mps = "\n".join(lines)
    validation = validate_mps_artifact(mps)
    return {
        "accepted": validation["status"] == "valid",
        "content": mps,
        "validation": validation,
        "issues": validation.get("issues", []),
    }


def validate_mps_artifact(content: str) -> dict[str, Any]:
    """Validate minimal structural requirements for an MPS artifact."""
    text = content or ""
    lines = [
        _canonical_line(line)
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("*")
    ]
    section_headers = {"objsense", "name", "rows", "columns", "rhs", "bounds", "endata"}
    sections = {
        line.split()[0]
        for line in lines
        if line.split()[0] in section_headers
    }
    issues: list[dict[str, str]] = []
    if lines and lines[0].split()[0] not in section_headers:
        issues.append(
            {
                "issue_type": "non_model_preamble",
                "severity": "error",
                "message": "MPS artifact must start with a section header, not prose.",
            }
        )
    required = {"name", "rows", "columns", "rhs", "endata"}
    for section in sorted(required - sections):
        issues.append(
            {
                "issue_type": f"missing_{section}_section",
                "severity": "error",
                "message": f"MPS artifact needs {section.upper()} section.",
            }
        )
    if "endata" in sections and lines[-1] != "endata":
        issues.append(
            {
                "issue_type": "non_model_trailer",
                "severity": "error",
                "message": "MPS artifact must end with ENDATA.",
            }
        )
    status = "valid" if not any(issue["severity"] == "error" for issue in issues) else "invalid"
    return {"status": status, "issues": issues, "sections": sorted(sections)}


def _mps_sections(content: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    current = ""
    section_names = {"OBJSENSE", "NAME", "ROWS", "COLUMNS", "RHS", "BOUNDS", "ENDATA"}
    for raw_line in (content or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("*"):
            continue
        first = line.split()[0].upper()
        if first in section_names:
            current = first
            sections.append((current, []))
            tail = line[len(first) :].strip()
            if tail:
                sections[-1][1].append(tail)
            continue
        if current:
            sections[-1][1].append(line)
    return sections


def parse_mps_artifact(content: str) -> ParsedLinearProgram:
    """Parse the deterministic MPS subset emitted by `build_mps_artifact_from_lp`."""
    validation = validate_mps_artifact(content)
    if validation["status"] != "valid":
        raise ValueError("MPS artifact is structurally invalid")

    direction = "minimize"
    row_types: dict[str, str] = {}
    row_order: list[str] = []
    objective_row = "OBJ"
    coeffs_by_row: dict[str, dict[str, float]] = {}
    rhs_by_row: dict[str, float] = {}
    bounds_by_var: dict[str, tuple[float | None, float | None]] = {}
    variables: set[str] = set()
    integer_vars: set[str] = set()
    in_integer_section = False

    for section, lines in _mps_sections(content):
        if section == "OBJSENSE":
            for line in lines:
                token = line.split()[0].upper()
                if token in {"MAX", "MAXIMIZE"}:
                    direction = "maximize"
                elif token in {"MIN", "MINIMIZE"}:
                    direction = "minimize"
        elif section == "ROWS":
            for line in lines:
                tokens = line.split()
                if len(tokens) < 2:
                    continue
                code, name = tokens[0].upper(), tokens[1]
                if code == "N":
                    objective_row = name
                    continue
                if code not in {"L", "G", "E"}:
                    raise ValueError(f"Unsupported MPS row type: {code}")
                row_types[name] = code
                row_order.append(name)
        elif section == "COLUMNS":
            for line in lines:
                tokens = line.split()
                if len(tokens) >= 3 and tokens[1].strip("'").upper() == "MARKER":
                    marker = tokens[2].strip("'").upper()
                    if marker == "INTORG":
                        in_integer_section = True
                    elif marker == "INTEND":
                        in_integer_section = False
                    continue
                if len(tokens) < 3:
                    continue
                var = tokens[0]
                variables.add(var)
                if in_integer_section:
                    integer_vars.add(var)
                for row_name, value in zip(tokens[1::2], tokens[2::2]):
                    try:
                        numeric_value = _parse_float(value)
                    except ValueError as exc:
                        raise ValueError(f"Invalid MPS coefficient: {value!r}") from exc
                    coeffs_by_row.setdefault(row_name, {})[var] = (
                        coeffs_by_row.setdefault(row_name, {}).get(var, 0.0)
                        + numeric_value
                    )
        elif section == "RHS":
            for line in lines:
                tokens = line.split()
                if len(tokens) < 3:
                    continue
                for row_name, value in zip(tokens[1::2], tokens[2::2]):
                    try:
                        rhs_by_row[row_name] = _parse_float(value)
                    except ValueError as exc:
                        raise ValueError(f"Invalid MPS RHS: {value!r}") from exc
        elif section == "BOUNDS":
            for line in lines:
                tokens = line.split()
                if len(tokens) < 3:
                    continue
                bound_type = tokens[0].upper()
                var = tokens[2]
                variables.add(var)
                low, high = bounds_by_var.get(var, (0.0, None))
                if bound_type == "BV":
                    bounds_by_var[var] = (0.0, 1.0)
                    integer_vars.add(var)
                    continue
                if bound_type == "FR":
                    bounds_by_var[var] = (None, None)
                    continue
                value = _parse_float(tokens[3]) if len(tokens) >= 4 else 0.0
                if bound_type in {"LI", "LO"}:
                    low = value
                elif bound_type in {"UI", "UP"}:
                    high = value
                else:
                    raise ValueError(f"Unsupported MPS bound type: {bound_type}")
                if bound_type in {"LI", "UI"}:
                    integer_vars.add(var)
                bounds_by_var[var] = (low, high)

    variable_names = tuple(sorted(variables))
    objective_coeffs = coeffs_by_row.get(objective_row, {})
    row_code_to_operator = {"L": "<=", "G": ">=", "E": "="}
    constraints = tuple(
        {
            "coeffs": coeffs_by_row.get(row_name, {}),
            "operator": row_code_to_operator[row_types[row_name]],
            "rhs": rhs_by_row.get(row_name, 0.0),
            "raw": row_name,
        }
        for row_name in row_order
    )
    bounds = tuple(bounds_by_var.get(var, (0.0, None)) for var in variable_names)
    integrality = tuple(1 if var in integer_vars else 0 for var in variable_names)
    return ParsedLinearProgram(
        direction=direction,
        variables=variable_names,
        objective=tuple(objective_coeffs.get(var, 0.0) for var in variable_names),
        constraints=constraints,
        bounds=bounds,
        integrality=integrality,
    )


def persist_mps_artifact_from_lp(
    *,
    topic_id: int,
    problem_id: int,
    lp_content: str,
    linked_component_ids: list[int] | None = None,
    generator_role: str = "lp_to_mps_generator",
    name: str = "ORBITMODEL",
) -> dict[str, Any]:
    """Persist an MPS artifact generated from LP text."""
    from . import api

    generated = build_mps_artifact_from_lp(lp_content, name=name)
    artifact_id = api.insert_optimization_artifact(
        problem_id=problem_id,
        topic_id=topic_id,
        artifact_type="mps_model",
        model_language="mps",
        content=generated["content"],
        parser_status=generated["validation"]["status"],
        parser_notes=json.dumps(generated["validation"], ensure_ascii=True, sort_keys=True),
        linked_component_ids_json=json.dumps(linked_component_ids or [], ensure_ascii=True),
        component_fingerprints_json=_component_fingerprints_for_ids(
            problem_id, linked_component_ids
        ),
        generator_role=generator_role,
    )
    diagnostic_ids: list[int] = []
    for issue in generated["validation"].get("issues", []):
        diagnostic_ids.append(
            api.insert_model_diagnostic(
                problem_id=problem_id,
                topic_id=topic_id,
                artifact_id=artifact_id,
                diagnostic_type=issue["issue_type"],
                severity=issue["severity"],
                message=issue["message"],
            )
        )
    return {
        **generated,
        "artifact_id": artifact_id,
        "diagnostic_ids": diagnostic_ids,
    }


def validate_solver_claim_payload(payload: dict[str, Any]) -> list[ValidationIssue]:
    """Validate a solver-backed formal claim before creating a candidate."""
    issues: list[ValidationIssue] = []
    conclusion = str(payload.get("conclusion") or "").strip()
    scope_context = str(payload.get("scope_context") or "").strip()
    falsification = str(payload.get("falsification_criteria") or "").strip()
    inference = str(payload.get("inference_logic") or "").strip()
    evidence_strength = payload.get("evidence_strength")
    scope_tags = payload.get("scope_tags") or []

    if len(conclusion) < 20:
        issues.append(
            ValidationIssue(
                "missing_conclusion",
                "error",
                "Solver-backed claims need a concrete conclusion.",
            )
        )
    if not scope_context:
        issues.append(
            ValidationIssue(
                "missing_scope",
                "error",
                "Solver-backed claims need problem, artifact, and backend scope.",
            )
        )
    if not falsification or not re.search(r"\d", falsification):
        issues.append(
            ValidationIssue(
                "missing_falsification_threshold",
                "error",
                "Falsification criteria must include a numeric tolerance or threshold.",
            )
        )
    if "[E" not in inference and "SolverRun" not in inference:
        issues.append(
            ValidationIssue(
                "missing_solver_evidence",
                "error",
                "Inference logic must point to solver/code evidence.",
            )
        )
    if not isinstance(scope_tags, list) or not any(
        str(tag).startswith("solver:") for tag in scope_tags
    ):
        issues.append(
            ValidationIssue(
                "missing_solver_scope_tag",
                "warning",
                "Scope tags should include solver:<backend>.",
            )
        )
    if not isinstance(evidence_strength, (int, float)) or not (
        1 <= evidence_strength <= 10
    ):
        issues.append(
            ValidationIssue(
                "invalid_evidence_strength",
                "error",
                "Evidence strength must be a number from 1 to 10.",
            )
        )
    return issues


def build_solver_claim_payload(
    *,
    problem: dict[str, Any],
    artifact: dict[str, Any],
    solver_run: dict[str, Any],
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Build a scoped formal claim payload from persisted solver evidence."""
    problem_title = problem.get("title") or f"optimization problem {problem.get('id')}"
    artifact_id = artifact.get("id")
    run_id = solver_run.get("id")
    backend = solver_run.get("solver_backend") or "unknown_solver"
    status = solver_run.get("status") or "unknown"
    objective = solver_run.get("objective_value")
    evidence_ref = (
        f"[E{solver_run['code_evidence_id']}]" if solver_run.get("code_evidence_id") else ""
    )
    if status == "optimal" and objective is not None:
        conclusion = (
            f"Solver run {run_id} found an optimal objective value of "
            f"{float(objective):.6g} for {problem_title} using artifact O{artifact_id}."
        )
        evidence_strength = 8.0
    else:
        conclusion = (
            f"Solver run {run_id} returned status {status} for {problem_title} "
            f"using artifact O{artifact_id}."
        )
        evidence_strength = 4.0
    scope_tags = [
        f"problem:{problem.get('id')}",
        f"artifact:{artifact_id}",
        f"solver:{backend}",
        f"status:{status}",
    ]
    scope_context = (
        f"Problem {problem.get('id')} ({problem_title}); artifact O{artifact_id}; "
        f"solver backend {backend}; run {run_id}; parser_status "
        f"{artifact.get('parser_status') or 'unknown'}."
    )
    falsification = (
        f"Re-solving artifact O{artifact_id} with the same data and an independent "
        f"compatible solver returns a different feasibility status or objective "
        f"value differing by more than {tolerance:g}."
    )
    inference = (
        f"Derived from SolverRun {run_id}"
        + (f" and code evidence {evidence_ref}" if evidence_ref else "")
        + f"; solver status={status}; objective={objective}."
    )
    return {
        "claim_type": "optimization_result",
        "conclusion": conclusion,
        "scope_tags": scope_tags,
        "scope_context": scope_context,
        "falsification_criteria": falsification,
        "inference_logic": inference,
        "evidence_strength": evidence_strength,
        "scope_breadth": 2.0,
        "submitted_by": "or_solver",
        "candidate_text": (conclusion + (f" {evidence_ref}" if evidence_ref else "")).strip(),
        "rationale_short": inference,
    }


def create_solver_claim_candidate(
    *,
    topic_id: int,
    subtopic_id: int | None,
    problem: dict[str, Any],
    artifact: dict[str, Any],
    solver_run: dict[str, Any],
    clerk_msg_id: int | None = None,
) -> dict[str, Any]:
    """Create a pending ClaimCandidate from solver evidence if gates pass."""
    from . import api

    payload = build_solver_claim_payload(
        problem=problem,
        artifact=artifact,
        solver_run=solver_run,
    )
    issues = validate_solver_claim_payload(payload)
    if any(issue.severity == "error" for issue in issues):
        return {
            "candidate_id": None,
            "payload": payload,
            "issues": [issue.__dict__ for issue in issues],
        }
    candidate_id = api.create_claim_candidate(
        topic_id,
        subtopic_id,
        clerk_msg_id,
        payload["candidate_text"],
        summary=payload["conclusion"],
        support_fact_ids_json="[]",
        rationale_short=payload["rationale_short"],
        claim_type=payload["claim_type"],
        scope_tags=json.dumps(payload["scope_tags"], ensure_ascii=True),
        scope_context=payload["scope_context"],
        falsification_criteria=payload["falsification_criteria"],
        inference_logic=payload["inference_logic"],
        conclusion=payload["conclusion"],
        evidence_strength=payload["evidence_strength"],
        scope_breadth=payload["scope_breadth"],
        submitted_by=payload["submitted_by"],
    )
    return {
        "candidate_id": candidate_id,
        "payload": payload,
        "issues": [issue.__dict__ for issue in issues],
    }


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _solver_claim_payload_from_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim_type": candidate.get("claim_type"),
        "conclusion": candidate.get("conclusion") or candidate.get("candidate_text"),
        "scope_tags": _json_list(candidate.get("scope_tags")),
        "scope_context": candidate.get("scope_context"),
        "falsification_criteria": candidate.get("falsification_criteria"),
        "inference_logic": candidate.get("inference_logic") or candidate.get("rationale_short"),
        "evidence_strength": candidate.get("evidence_strength"),
        "scope_breadth": candidate.get("scope_breadth"),
        "candidate_text": candidate.get("candidate_text"),
        "rationale_short": candidate.get("rationale_short"),
        "submitted_by": candidate.get("submitted_by") or "or_solver",
    }


def _evidence_refs_from_text(text: str) -> list[int]:
    refs: list[int] = []
    for match in re.finditer(r"\[E(\d+)\]", text or ""):
        refs.append(int(match.group(1)))
    return refs


def review_solver_claim_candidate(
    topic_id: int,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Deterministically accept/reject a pending solver-backed claim candidate."""
    from . import api

    if candidate.get("claim_type") != "optimization_result":
        return {"accepted": False, "claim_id": None, "reason": "not_solver_claim"}
    payload = _solver_claim_payload_from_candidate(candidate)
    issues = validate_solver_claim_payload(payload)
    error_issues = [issue for issue in issues if issue.severity == "error"]
    if error_issues:
        note = "; ".join(issue.message for issue in error_issues)
        api.update_claim_candidate_review(
            int(candidate["id"]),
            "rejected",
            reviewed_text=candidate.get("candidate_text"),
            review_note=note,
            claim_score=payload.get("evidence_strength"),
        )
        return {
            "accepted": False,
            "claim_id": None,
            "reason": "validation_failed",
            "issues": [issue.__dict__ for issue in issues],
        }

    conclusion = str(payload.get("conclusion") or "").strip()
    scope_context = str(payload.get("scope_context") or "").strip()
    for claim in api.get_claims(topic_id, limit=1000, include_superseded=False):
        if claim.get("claim_type") != "optimization_result":
            continue
        same_conclusion = conclusion and conclusion == (claim.get("conclusion") or claim.get("content"))
        same_scope = scope_context and scope_context == (claim.get("scope_context") or "")
        if same_conclusion or same_scope:
            api.update_claim_candidate_review(
                int(candidate["id"]),
                "rejected",
                reviewed_text=candidate.get("candidate_text"),
                review_note=f"Duplicate of active optimization_result claim C{claim['id']}.",
                claim_score=payload.get("evidence_strength"),
                accepted_claim_id=int(claim["id"]),
            )
            return {
                "accepted": False,
                "claim_id": int(claim["id"]),
                "reason": "duplicate",
            }

    claim_id = api.insert_claim(
        topic_id,
        candidate.get("subtopic_id"),
        candidate.get("candidate_text") or conclusion,
        summary=conclusion,
        support_fact_ids_json=candidate.get("support_fact_ids_json") or "[]",
        rationale_short=payload.get("rationale_short"),
        claim_score=payload.get("evidence_strength"),
        status="active",
        candidate_id=int(candidate["id"]),
        claim_type="optimization_result",
        scope_tags=json.dumps(payload.get("scope_tags") or [], ensure_ascii=True),
        scope_context=scope_context,
        falsification_criteria=payload.get("falsification_criteria"),
        inference_logic=payload.get("inference_logic"),
        conclusion=conclusion,
        evidence_strength=payload.get("evidence_strength"),
        scope_breadth=payload.get("scope_breadth"),
        submitted_by=payload.get("submitted_by"),
    )
    api.update_claim_candidate_review(
        int(candidate["id"]),
        "accepted",
        reviewed_text=candidate.get("candidate_text"),
        review_note="Accepted by deterministic solver-claim review.",
        claim_score=payload.get("evidence_strength"),
        accepted_claim_id=claim_id,
    )
    evidence_text = " ".join(
        str(payload.get(key) or "") for key in ("candidate_text", "inference_logic")
    )
    for evidence_id in _evidence_refs_from_text(evidence_text):
        api.insert_knowledge_edge(
            topic_id,
            evidence_id,
            "code_evidence",
            claim_id,
            "claim",
            "supports",
            justification_group="solver_evidence",
            confidence=min(1.0, float(payload.get("evidence_strength") or 0.0) / 10.0),
            created_by="solver_claim_review",
        )
    return {"accepted": True, "claim_id": claim_id, "reason": "accepted"}


def review_pending_solver_claim_candidates(topic_id: int) -> list[dict[str, Any]]:
    """Review all pending solver-backed claim candidates for a topic."""
    from . import api

    results: list[dict[str, Any]] = []
    for candidate in api.get_claim_candidates(topic_id, status="pending", limit=10000):
        if candidate.get("claim_type") == "optimization_result":
            results.append(review_solver_claim_candidate(topic_id, candidate))
    return results


def build_component_extraction_prompt(source_text: str) -> str:
    return f"""\
Extract management science / operations research model components from the
problem statement.

Return strict JSON only:
{{
  "components": [
    {{
      "component_type": "decision_problem|alternative|objective|kpi|constraint|stakeholder|assumption|intervention|dataset|sample|method|effect_size|uncertainty|boundary_condition|managerial_implication|set|index|parameter|decision_variable|derived_variable|data_requirement",
      "natural_text": "verbatim or conservative source-backed text",
      "formal_text": "optional algebra, inequality, or definition",
      "symbol": "optional stable symbol",
      "unit": "optional unit",
      "domain": "optional domain such as binary/nonnegative/integer",
      "source_refs": ["D1"]
    }}
  ]
}}

Rules:
- Preserve uncertainty and missing data as assumptions or data_requirements.
- Do not invent numbers, units, symbols, or constraints not present in source.
- Use objective for maximize/minimize criteria and kpi for managerial metrics.
- Use constraint for hard feasibility limits, budgets, capacities, policies, or service requirements.
- For objective formal_text, write only a parseable linear expression such as `minimize 20 b + 40 c`; do not write `Z = ...`.
- For constraint formal_text, write one linear equality/inequality per component. Expand percentages or shares into linear form, e.g. `-0.6 b + 0.4 c >= 0`, not `c >= 0.6(b + c)`.
- For "at most p% of the total can be Y", encode `Y <= p(X + Y)`, e.g. `0.6 Y - 0.4 X <= 0`.
- For "at least p% of the total must be Y", encode `Y >= p(X + Y)`, e.g. `0.4 Y - 0.6 X >= 0`.
- Keep decision_variable symbols consistent with objective and constraint algebra. If formal_text uses expanded variables such as `x1` and `x2`, return separate decision_variable components with symbols `x1` and `x2`; do not return only an indexed symbol such as `x_i`.
- For finite tables or short horizons, prefer scalar ASCII LP-ready algebra over indexed notation. For example use `x_1 + x_2 + x_3`, not `sum_t x_t`, `Σ`, `∀`, `∈`, or `x_{{i,t}}`.
- If a formulation cannot be safely expanded to scalar LP/MILP terms, keep the indexed expression in natural_text and return a `data_requirement` or `assumption`; do not put unsupported indexed notation in formal_text.
- Use ASCII symbols in formal_text. Avoid Greek letters, Unicode inequality signs, and multiplication dots; write `alpha_I`, `<=`, `>=`, and `2 x`.
- Keep named resource pools separate. If inspection time and fixing/schedule time are stated separately, do not add inspection time into the fixing/schedule capacity unless the source explicitly says the capacity is total time including both.
- Interpret phrases like "at most N items of either type", "at most N of either product", or "can sell at most N items of either kind" as a combined total limit across the listed decision variables, e.g. `x_A + x_B <= N`. Use separate per-type limits only when the source says "each", "per type", or gives one limit for each type.
- Treat counts of discrete items (pills, appliances, vehicles, shifts, trips, containers, workers, products) as integer decision variables unless the source explicitly allows fractional/continuous quantities.
- Put nonnegative/integer/binary restrictions on decision_variable `domain`; do not return them as a separate constraint.
- Use explicit spaces between coefficients and variables, e.g. `20 b`, not `20b`.

Problem statement:
{source_text}
"""


def build_solver_component_extraction_prompt(source_text: str) -> str:
    return f"""\
Extract only solver-relevant operations research model components from the
problem statement. The output will be compiled into an LP/MILP artifact, so do
not include broad business-context components.

Return strict JSON only:
{{
  "components": [
    {{
      "component_type": "set|index|parameter|decision_variable|derived_variable|objective|constraint|assumption|data_requirement",
      "natural_text": "verbatim or conservative source-backed text",
      "formal_text": "optional parseable algebra, inequality, or definition",
      "symbol": "optional stable symbol",
      "unit": "optional unit",
      "domain": "optional domain such as nonnegative/integer/binary",
      "source_refs": ["D1"]
    }}
  ]
}}

Rules:
- Every component must include `"source_refs": ["D1"]`.
- Do not return decision_problem, stakeholder, kpi, dataset, method,
  managerial_implication, alternative, intervention, sample, effect_size,
  uncertainty, or boundary_condition components.
- Include at least one objective, at least one decision_variable, and the hard
  constraints needed for a solver when the problem statement provides enough
  information.
- For objective formal_text, write only a parseable linear expression such as
  `minimize 20 x + 40 y`; do not write `Z = ...`.
- For constraint formal_text, write one linear equality/inequality per
  component. Use explicit spaces between coefficients and variables, e.g.
  `20 x`, not `20x`.
- If indexed or summation notation is needed, prefer simple scalar expansions
  over expressions containing `sum`, `forall`, set-builder syntax, or
  parentheses around indexed terms.
- For finite tables or short horizons, expand variables and constraints to
  ASCII scalar form such as `x_1 + x_2 + x_3`; do not use `Σ`, `∀`, `∈`, or
  `x_{{i,t}}` in formal_text.
- If the safe scalar expansion is too large or ambiguous, preserve the indexed
  expression in natural_text and return a data_requirement instead of
  unsupported formal_text.
- Put nonnegative/integer/binary restrictions on decision_variable `domain`;
  do not return them as separate constraints.
- Treat counts of discrete items (workers, vehicles, shifts, trips, products,
  containers, facilities) as integer unless the source explicitly permits
  fractional quantities.
- Preserve missing information as data_requirement rather than inventing data.

Problem statement:
{source_text}
"""


def build_indexed_model_ir_prompt(source_text: str) -> str:
    return f"""\
Extract a finite indexed LP/MILP representation for this operations research
problem. This is an intermediate representation; deterministic code will
scalarize it into LP text.

Return strict JSON only:
{{
  "schema": "orbit_indexed_model_ir.v1",
  "status": "complete",
  "sets": {{"T": [1, 2, 3]}},
  "parameters": {{"d": {{"values": {{"1": 10, "2": 20, "3": 30}}}}}},
  "variables": [
    {{"symbol": "x", "indices": ["T"], "domain": "integer"}}
  ],
  "objective": {{
    "sense": "minimize",
    "terms": [
      {{
        "coefficient": 2,
        "variable": "x",
        "subscripts": ["t"],
        "sum_over": {{"t": "T"}}
      }}
    ]
  }},
  "constraints": [
    {{
      "name": "demand",
      "for_each": {{"t": "T"}},
      "terms": [
        {{"coefficient": 1, "variable": "x", "subscripts": ["t"]}}
      ],
      "sense": ">=",
      "rhs": {{"parameter": "d", "subscripts": ["t"]}}
    }}
  ]
}}

Expression rules:
- A term may contain coefficient, variable, subscripts, sum_over, and where.
- Use finite set values only; never use symbolic ranges without listing values.
- Use parameter references as {{"parameter": "name", "subscripts": ["t"]}}.
- Use prefix or triangular sums with where, e.g. {{"tau": {{"<=": "t"}}}}.
- Put integer or binary domains on variables, not as constraints.
- Preserve every numeric value from the source; do not invent missing data.
- Every target, demand, capacity, deadline, duration, initial state, and limit
  parameter must appear in at least one objective or constraint. Return
  insufficient_data rather than omitting a critical parameter.
- If the source lacks enough numeric data or the model is nonlinear, return
  {{"schema":"orbit_indexed_model_ir.v1","status":"insufficient_data","issues":[{{"issue_type":"missing_data","message":"..."}}]}}.
- Do not return LP text, Python code, markdown, or prose.

Problem statement:
{source_text}
"""


def build_indexed_model_ir_repair_prompt(
    *,
    source_text: str,
    indexed_ir: dict[str, Any],
    validation: dict[str, Any],
) -> str:
    return f"""\
Repair this finite indexed LP/MILP IR so deterministic scalarization can
produce a complete LP artifact. Return the full corrected
orbit_indexed_model_ir.v1 JSON only.

Rules:
- Preserve source-backed numbers and finite sets from the problem statement.
- Fix every validation issue listed below.
- Every target, demand, capacity, deadline, duration, initial state, and limit
  parameter must appear in at least one objective or constraint.
- Do not return patches, prose, LP text, Python code, or markdown.
- If the model cannot be completed from the source, return
  {{"schema":"orbit_indexed_model_ir.v1","status":"insufficient_data","issues":[...]}}.

Validation issues:
{json.dumps(validation.get("issues") or [], ensure_ascii=False, indent=2)}

Current indexed IR:
{json.dumps(indexed_ir, ensure_ascii=False, indent=2)}

Problem statement:
{source_text}
"""


def persist_component_payloads(
    *,
    topic_id: int,
    problem_id: int,
    payloads: list[dict[str, Any]],
    default_review_status: str = "candidate",
    default_source_refs: list[str] | None = None,
) -> dict[str, Any]:
    """Persist extracted components and diagnostics from structured payloads."""
    from . import api

    component_ids: list[int] = []
    diagnostic_ids: list[int] = []
    for payload in payloads:
        normalized_payload = dict(payload)
        inferred_symbol = _payload_symbol(normalized_payload)
        if inferred_symbol and not _component_text_value(normalized_payload.get("symbol")):
            normalized_payload["symbol"] = inferred_symbol
            normalized_payload.pop("formal_text", None)
        payload = normalized_payload
        issues = validate_component_payload(payload)
        source_refs = payload.get("source_refs") or default_source_refs or []
        source_refs_json = json.dumps(source_refs, ensure_ascii=True)
        component_id = api.insert_optimization_component(
            problem_id=problem_id,
            topic_id=topic_id,
            component_type=str(payload.get("component_type") or ""),
            natural_text=str(payload.get("natural_text") or ""),
            formal_text=_component_text_value(payload.get("formal_text")),
            symbol=_component_text_value(payload.get("symbol")),
            unit=_component_text_value(payload.get("unit")),
            domain=_component_text_value(payload.get("domain")),
            source_refs_json=source_refs_json,
            review_status=default_review_status,
            validation_notes="; ".join(issue.message for issue in issues) or None,
            metadata_json=json.dumps(
                {"raw_payload": payload}, ensure_ascii=True, sort_keys=True
            ),
        )
        component_ids.append(component_id)
        for issue in issues:
            diagnostic_ids.append(
                api.insert_model_diagnostic(
                    problem_id=problem_id,
                    topic_id=topic_id,
                    component_id=component_id,
                    diagnostic_type=issue.issue_type,
                    severity=issue.severity,
                    message=issue.message,
                    source_refs_json=source_refs_json,
                )
            )
    return {"component_ids": component_ids, "diagnostic_ids": diagnostic_ids}


async def extract_and_persist_components(
    *,
    topic_id: int,
    problem_id: int,
    source_text: str,
    provider_profile: str = "minimax",
) -> dict[str, Any]:
    """Call an LLM extractor and persist returned OR/MSE components."""
    from .broker import DEFAULT_MAX_TOKENS, llm_call
    from .json_utils import extract_json_object

    prompt = build_component_extraction_prompt(source_text)
    response = await llm_call(
        prompt,
        system_prompt=(
            "You extract optimization model components. Return strict JSON only."
        ),
        provider_profile=provider_profile,
        role="or_component_extractor",
        max_tokens=DEFAULT_MAX_TOKENS,
        require_json=True,
    )
    parsed = extract_json_object(response.text)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("components"), list):
        from . import api

        diagnostic_id = api.insert_model_diagnostic(
            problem_id=problem_id,
            topic_id=topic_id,
            diagnostic_type="component_extraction_parse_error",
            severity="error",
            message="Component extractor did not return {'components': [...]} JSON.",
        )
        return {"component_ids": [], "diagnostic_ids": [diagnostic_id], "raw_text": response.text}
    persisted = persist_component_payloads(
        topic_id=topic_id,
        problem_id=problem_id,
        payloads=[item for item in parsed["components"] if isinstance(item, dict)],
    )
    return {**persisted, "raw_text": response.text}


def _indexed_ir_from_parsed(parsed: Any) -> dict[str, Any] | None:
    if not isinstance(parsed, dict):
        return None
    wrapped = parsed.get("indexed_model_ir")
    if isinstance(wrapped, dict):
        return wrapped
    if parsed.get("schema") == "orbit_indexed_model_ir.v1":
        return parsed
    if any(key in parsed for key in ("sets", "variables", "objective", "constraints")):
        return parsed
    return None


async def extract_indexed_ir_candidate_tournament(
    *,
    topic_id: int,
    problem_id: int,
    source_text: str,
    provider_profile: str = "minimax",
    model: str = "",
    temperature: float = 0.4,
    candidate_count: int = 2,
    generator_role: str = "or_indexed_ir_extractor",
) -> dict[str, Any]:
    """Generate indexed IR candidates, scalarize them, and persist the best LP."""
    from . import api
    from .broker import DEFAULT_MAX_TOKENS, llm_call
    from .json_utils import extract_json_object

    prompt = build_indexed_model_ir_prompt(source_text)
    candidates: list[dict[str, Any]] = []
    for index in range(max(1, min(candidate_count, 4))):
        response = await llm_call(
            f"{prompt}\n\nGenerate candidate {index + 1} independently.",
            system_prompt=(
                "You extract finite indexed LP/MILP model IR. "
                "Return strict JSON only."
            ),
            provider_profile=provider_profile,
            model=model,
            temperature=temperature,
            role="or_indexed_ir_extractor",
            max_tokens=DEFAULT_MAX_TOKENS,
            require_json=True,
        )
        parsed = extract_json_object(response.text)
        indexed_ir = _indexed_ir_from_parsed(parsed)
        diagnostics: list[dict[str, Any]] = []
        if indexed_ir is None:
            generated = {
                "accepted": False,
                "content": "",
                "validation": {
                    "status": "invalid",
                    "issues": [
                        {
                            "issue_type": "indexed_ir_parse_error",
                            "severity": "error",
                            "message": (
                                "Indexed IR extractor did not return "
                                "orbit_indexed_model_ir.v1 JSON."
                            ),
                        }
                    ],
                },
                "issues": [],
            }
        else:
            generated = build_lp_artifact_from_indexed_ir(indexed_ir)
        diagnostics.extend(generated.get("issues") or [])
        candidates.append(
            {
                "candidate_index": index,
                "repair_attempt": False,
                "indexed_ir": indexed_ir,
                "content": generated.get("content") or "",
                "model_language": "lp",
                "validation": generated.get("validation") or {},
                "diagnostics": diagnostics,
                "raw_text": response.text,
            }
        )
        if indexed_ir is not None and not generated.get("accepted"):
            repair_response = await llm_call(
                build_indexed_model_ir_repair_prompt(
                    source_text=source_text,
                    indexed_ir=indexed_ir,
                    validation=generated.get("validation") or {},
                ),
                system_prompt=(
                    "You repair finite indexed LP/MILP model IR. "
                    "Return strict JSON only."
                ),
                provider_profile=provider_profile,
                model=model,
                temperature=min(temperature, 0.2),
                role="or_indexed_ir_repair",
                max_tokens=DEFAULT_MAX_TOKENS,
                require_json=True,
            )
            repair_parsed = extract_json_object(repair_response.text)
            repair_ir = _indexed_ir_from_parsed(repair_parsed)
            if repair_ir is None:
                repair_generated = {
                    "accepted": False,
                    "content": "",
                    "validation": {
                        "status": "invalid",
                        "issues": [
                            {
                                "issue_type": "indexed_ir_parse_error",
                                "severity": "error",
                                "message": (
                                    "Indexed IR repair did not return "
                                    "orbit_indexed_model_ir.v1 JSON."
                                ),
                            }
                        ],
                    },
                    "issues": [],
                }
            else:
                repair_generated = build_lp_artifact_from_indexed_ir(repair_ir)
            candidates.append(
                {
                    "candidate_index": index,
                    "repair_attempt": True,
                    "indexed_ir": repair_ir,
                    "content": repair_generated.get("content") or "",
                    "model_language": "lp",
                    "validation": repair_generated.get("validation") or {},
                    "diagnostics": repair_generated.get("issues") or [],
                    "raw_text": repair_response.text,
                }
            )

    ranked = rank_modeling_candidates(candidates)
    best = ranked[0] if ranked else None
    persisted: dict[str, Any] = {"artifact_id": None, "diagnostic_ids": []}
    if best and isinstance(best.get("indexed_ir"), dict):
        persisted = persist_lp_artifact_from_indexed_ir(
            topic_id=topic_id,
            problem_id=problem_id,
            indexed_ir=best["indexed_ir"],
            generator_role=generator_role,
        )
    elif best:
        persisted["diagnostic_ids"].append(
            api.insert_model_diagnostic(
                problem_id=problem_id,
                topic_id=topic_id,
                diagnostic_type="indexed_ir_parse_error",
                severity="error",
                message="Indexed IR extractor did not return parseable JSON.",
            )
        )

    return {
        "best": best,
        "ranked": ranked,
        "artifact_id": persisted.get("artifact_id"),
        "diagnostic_ids": persisted.get("diagnostic_ids", []),
    }


def _candidate_components_from_payloads(
    payloads: list[dict[str, Any]],
    *,
    review_status: str = "candidate",
    default_source_refs: list[str] | None = None,
) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    for index, payload in enumerate(payloads, start=1):
        source_refs = (
            payload.get("source_refs")
            or payload.get("source_refs_json")
            or default_source_refs
            or []
        )
        if isinstance(source_refs, str):
            try:
                parsed_refs = json.loads(source_refs)
            except json.JSONDecodeError:
                parsed_refs = [source_refs] if source_refs.strip() else []
            source_refs = parsed_refs
        components.append(
            {
                "id": index,
                "component_type": str(payload.get("component_type") or ""),
                "natural_text": str(payload.get("natural_text") or ""),
                "formal_text": _component_text_value(payload.get("formal_text")),
                "symbol": _component_text_value(payload.get("symbol")),
                "unit": _component_text_value(payload.get("unit")),
                "domain": _component_text_value(payload.get("domain")),
                "source_refs_json": json.dumps(source_refs, ensure_ascii=True),
                "review_status": review_status,
            }
        )
    return components


async def extract_component_candidate_tournament(
    *,
    topic_id: int,
    problem_id: int,
    source_text: str,
    provider_profile: str = "minimax",
<<<<<<< Updated upstream
    model: str = "",
    temperature: float = 0.7,
    solver_only: bool = False,
=======
    model: str | None = None,
>>>>>>> Stashed changes
    candidate_count: int = 3,
    default_review_status: str = "candidate",
    default_source_refs: list[str] | None = None,
) -> dict[str, Any]:
    """Generate multiple component candidates, rank them, and persist the best one.

    This is the MiniMax-facing entrypoint for pass@k-style modeling without
    exposing multiple answers to the user.
    """
    from .broker import DEFAULT_MAX_TOKENS, llm_call
    from .json_utils import extract_json_object

    prompt = (
        build_solver_component_extraction_prompt(source_text)
        if solver_only
        else build_component_extraction_prompt(source_text)
    )
    candidates: list[dict[str, Any]] = []
    for index in range(max(1, min(candidate_count, 8))):
        response = await llm_call(
            f"{prompt}\n\nGenerate candidate {index + 1} with independent wording.",
            system_prompt=(
                "You extract optimization model components. Return strict JSON only."
            ),
            provider_profile=provider_profile,
            model=model,
            temperature=temperature,
            role="or_component_extractor",
            max_tokens=DEFAULT_MAX_TOKENS,
            require_json=True,
        )
        parsed = extract_json_object(response.text)
        diagnostics: list[dict[str, Any]] = []
        payloads: list[dict[str, Any]] = []
        if isinstance(parsed, dict) and isinstance(parsed.get("components"), list):
            payloads = [item for item in parsed["components"] if isinstance(item, dict)]
            for payload in payloads:
                diagnostics.extend(issue.__dict__ for issue in validate_component_payload(payload))
        else:
            diagnostics.append(
                {
                    "issue_type": "component_extraction_parse_error",
                    "severity": "error",
                    "message": "Component extractor did not return {'components': [...]} JSON.",
                }
            )
        components = _candidate_components_from_payloads(
            payloads,
            review_status="reviewed",
            default_source_refs=default_source_refs,
        )
        generated = build_lp_artifact_from_components(
            components,
            require_reviewed=False,
        )
        candidates.append(
            {
                "candidate_index": index,
                "components": components,
                "component_payloads": payloads,
                "content": generated.get("content") or "",
                "model_language": "lp",
                "validation": generated.get("validation") or {},
                "diagnostics": diagnostics + list(generated.get("issues") or []),
                "raw_text": response.text,
            }
        )

<<<<<<< Updated upstream
=======
    if single_call and bounded_count > 1:
        response = await llm_call(
            (
                f"{prompt}\n\nGenerate {bounded_count} independent candidate "
                "formulations. Return strict JSON only in this shape: "
                '{"candidates":[{"components":[...]},{"components":[...]}]}.'
            ),
            system_prompt=(
                "You extract multiple alternative optimization model component "
                "sets. Return strict JSON only."
            ),
            provider_profile=provider_profile,
            model=model,
            role="or_component_extractor",
            max_tokens=COMPONENT_EXTRACTOR_MAX_TOKENS,
            require_json=True,
        )
        parsed = extract_json_object(response.text)
        raw_candidates = []
        if isinstance(parsed, dict) and isinstance(parsed.get("candidates"), list):
            raw_candidates = [
                item for item in parsed["candidates"][:bounded_count] if isinstance(item, dict)
            ]
        elif isinstance(parsed, dict) and isinstance(parsed.get("components"), list):
            raw_candidates = [parsed]
        for index in range(bounded_count):
            item = raw_candidates[index] if index < len(raw_candidates) else {}
            diagnostics: list[dict[str, Any]] = []
            payloads: list[dict[str, Any]] = []
            if isinstance(item, dict) and isinstance(item.get("components"), list):
                payloads = [entry for entry in item["components"] if isinstance(entry, dict)]
                for payload in payloads:
                    diagnostics.extend(
                        issue.__dict__ for issue in validate_component_payload(payload)
                    )
            else:
                diagnostics.append(
                    {
                        "issue_type": "component_extraction_parse_error",
                        "severity": "error",
                        "message": "Component extractor did not return a candidate components list.",
                    }
                )
            append_candidate(
                index=index,
                payloads=payloads,
                diagnostics=diagnostics,
                raw_text=response.text,
            )
    else:
        for index in range(bounded_count):
            response = await llm_call(
                f"{prompt}\n\nGenerate candidate {index + 1} with independent wording.",
                system_prompt=(
                    "You extract optimization model components. Return strict JSON only."
                ),
                provider_profile=provider_profile,
                model=model,
                role="or_component_extractor",
                max_tokens=COMPONENT_EXTRACTOR_MAX_TOKENS,
                require_json=True,
            )
            parsed = extract_json_object(response.text)
            diagnostics: list[dict[str, Any]] = []
            payloads: list[dict[str, Any]] = []
            if isinstance(parsed, dict) and isinstance(parsed.get("components"), list):
                payloads = [item for item in parsed["components"] if isinstance(item, dict)]
                for payload in payloads:
                    diagnostics.extend(
                        issue.__dict__ for issue in validate_component_payload(payload)
                    )
            else:
                diagnostics.append(
                    {
                        "issue_type": "component_extraction_parse_error",
                        "severity": "error",
                        "message": "Component extractor did not return {'components': [...]} JSON.",
                    }
                )
            append_candidate(
                index=index,
                payloads=payloads,
                diagnostics=diagnostics,
                raw_text=response.text,
            )

>>>>>>> Stashed changes
    ranked = rank_modeling_candidates(candidates)
    best = ranked[0] if ranked else None
    persisted: dict[str, Any] = {"component_ids": [], "diagnostic_ids": []}
    if best and best.get("component_payloads"):
        persisted = persist_component_payloads(
            topic_id=topic_id,
            problem_id=problem_id,
            payloads=best["component_payloads"],
            default_review_status=default_review_status,
            default_source_refs=default_source_refs,
        )
    if best and best.get("modeling_error") != "none":
        from . import api

        persisted.setdefault("diagnostic_ids", []).append(
            api.insert_model_diagnostic(
                problem_id=problem_id,
                topic_id=topic_id,
                diagnostic_type=str(best.get("modeling_error") or "modeling_error"),
                severity="warning",
                message=(
                    "Best component tournament candidate still has modeling "
                    f"diagnostics: {best.get('modeling_error')}"
                ),
            )
        )
    return {
        "best": best,
        "ranked": ranked,
        "component_ids": persisted.get("component_ids", []),
        "diagnostic_ids": persisted.get("diagnostic_ids", []),
    }


def classify_solver_failure(status: str, stdout: str = "", stderr: str = "") -> str:
    text = " ".join([status or "", stdout or "", stderr or ""]).lower()
    if "missing data" in text or "data requirement" in text:
        return "missing_data"
    if "ambiguous variable" in text or "ambiguous symbol" in text:
        return "ambiguous_variable"
    if "unit mismatch" in text or "unit conflict" in text:
        return "unit_mismatch"
    if (
        "wrong optimal value" in text
        or "objective mismatch" in text
        or "optimal value differs" in text
    ):
        return "wrong_optimal_value"
    if "no-solution calibration" in text or "no solution calibration" in text:
        return "no_solution_calibration_error"
    if any(token in text for token in ("infeasible", "no feasible")):
        return "solver_infeasible"
    if "unbounded" in text:
        return "solver_unbounded"
    if "timeout" in text or "timed out" in text:
        return "solver_timeout"
    if "syntax" in text or "parse" in text or "invalid lp" in text:
        return "invalid_artifact_syntax"
    if "no best solution" in text or "no available solution" in text:
        return "no_solution_reported"
    if "unsupported" in text or "requires a milp solver" in text:
        return "unsupported_model_class"
    if "solver_unavailable" in text or "no module named" in text:
        return "solver_unavailable"
    if "traceback" in text or "runtimeerror" in text or "exception" in text:
        return "generated_code_runtime_error"
    if status.lower() in {"optimal", "solved", "success"}:
        return "none"
    return "unknown_solver_failure"


def classify_modeling_error(
    text: str = "", issues: list[dict[str, Any]] | None = None
) -> str:
    """Classify model-construction errors using the ORLM-style taxonomy."""
    issue_types = {
        str(issue.get("issue_type") or "").lower()
        for issue in (issues or [])
        if isinstance(issue, dict)
    }
    joined = " ".join(
        [
            text or "",
            " ".join(issue_types),
            " ".join(
                str(issue.get("message") or "")
                for issue in (issues or [])
                if isinstance(issue, dict)
            ),
        ]
    ).lower()
    if not joined.strip():
        return "none"
    if any(
        token in joined
        for token in (
            "wrong problem",
            "misunderstand",
            "semantic misunderstanding",
            "wrong objective context",
            "irrelevant model",
        )
    ):
        return "semantic_misunderstanding"
    if any(
        token in issue_types
        for token in (
            "objective_direction_mismatch",
            "linked_variable_domain_mismatch",
            "linked_parameter_unit_conflict",
            "big_m_without_binary_indicator",
            "logical_constraint_without_indicator",
        )
    ) or any(
        token in joined
        for token in (
            "objective mismatch",
            "constraint translation",
            "direction mismatch",
            "unit mismatch",
            "logical relationship",
            "nonlinear term",
        )
    ):
        return "objective_constraint_translation_error"
    if any(
        token in issue_types
        for token in (
            "linked_constraint_missing",
            "linked_decision_variable_missing",
            "missing_constraints",
            "missing_decision_variables",
            "missing_objective_section",
            "missing_constraints_section",
            "missing_auxiliary_variable",
            "multi_objective_without_priority",
        )
    ) or any(
        token in joined
        for token in (
            "low model completeness",
            "missing constraint",
            "missing variable",
            "omitted",
            "incomplete",
            "ignore",
        )
    ):
        return "low_model_completeness"
    if any(token.endswith("_mismatch") or token.startswith("linked_") for token in issue_types):
        return "objective_constraint_translation_error"
    if any("missing" in token for token in issue_types):
        return "low_model_completeness"
    return "unknown_modeling_error"


def validate_modeling_techniques(
    components: list[dict[str, Any]],
    *,
    content: str = "",
) -> list[ValidationIssue]:
    """Detect common OR modeling technique gaps before solver execution."""
    issues: list[ValidationIssue] = []
    component_text = "\n".join(
        " ".join(
            str(component.get(field) or "")
            for field in ("component_type", "natural_text", "formal_text", "symbol", "domain")
        )
        for component in components
    )
    text = f"{component_text}\n{content or ''}"
    lowered = text.lower()
    variable_components = [
        component
        for component in components
        if component.get("component_type") in {"decision_variable", "derived_variable"}
    ]
    has_binary = any(_domain_hint(component) == "binary" for component in variable_components)
    has_derived = any(
        component.get("component_type") == "derived_variable"
        for component in variable_components
    )
    objective_components = [
        component for component in components if component.get("component_type") == "objective"
    ]

    if re.search(r"\b(big[-\s]?m|large\s+m)\b", lowered) and not has_binary:
        issues.append(
            ValidationIssue(
                "big_m_without_binary_indicator",
                "error",
                "Big-M formulations should include an explicit binary indicator variable.",
            )
        )
    logical_text = "\n".join(
        " ".join(
            str(component.get(field) or "")
            for field in ("natural_text", "formal_text", "symbol")
        )
        for component in components
        if component.get("component_type") in {"constraint", "objective"}
    )
    logical_lowered = f"{logical_text}\n{content or ''}".lower()
    if (
        re.search(
            (
                r"\b(only if|implies|mutual exclusion|cannot\b.{0,80}\btogether)\b"
                r"|\bif\b.{0,160}\bthen\b"
                r"|\bif\b.{0,160}\b(cannot|requires?|selected|chosen)\b"
                r"|\bwhen\b.{0,160}\bthen\b"
                r"|\beither\b.{0,160}\bor\b"
            ),
            logical_lowered,
        )
        and not has_binary
    ):
        issues.append(
            ValidationIssue(
                "logical_constraint_without_indicator",
                "error",
                "Logical constraints should be represented with binary or indicator variables.",
            )
        )
    if re.search(r"\b(auxiliary|lineariz|piecewise|absolute value|max\(|min\()\b", lowered) and not has_derived:
        issues.append(
            ValidationIssue(
                "missing_auxiliary_variable",
                "warning",
                "Linearization or auxiliary-variable language appears without a derived variable component.",
            )
        )
    if len(objective_components) > 1:
        objective_text = " ".join(
            str(component.get("formal_text") or component.get("natural_text") or "")
            for component in objective_components
        ).lower()
        if not re.search(r"\b(weight|priority|lexicographic|goal|penalty)\b", objective_text):
            issues.append(
                ValidationIssue(
                    "multi_objective_without_priority",
                    "warning",
                    "Multiple objectives should declare weights, priorities, or a goal-programming scheme.",
                )
            )
    if content.strip():
        try:
            parsed = parse_lp_artifact(content)
        except Exception:
            parsed = None
        if parsed is not None:
            constrained_variables = {
                variable
                for constraint in parsed.constraints
                for variable, coefficient in (constraint.get("coeffs") or {}).items()
                if not math.isclose(float(coefficient), 0.0, abs_tol=1e-12)
            }
            for index, variable in enumerate(parsed.variables):
                coefficient = parsed.objective[index]
                if math.isclose(coefficient, 0.0, abs_tol=1e-12):
                    continue
                if variable in constrained_variables:
                    continue
                lower, upper = parsed.bounds[index]
                has_effective_bound = (
                    lower is not None
                    and not math.isclose(lower, 0.0, abs_tol=1e-12)
                ) or upper is not None
                if has_effective_bound:
                    continue
                issues.append(
                    ValidationIssue(
                        "objective_variable_not_linked",
                        "error",
                        (
                            f"Objective variable {variable!r} is not linked to "
                            "any constraint or effective bound; define it as a "
                            "linear expression or add a derived-variable equality."
                        ),
                    )
                )
    return issues


def _candidate_components(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    components = candidate.get("components") or []
    return [component for component in components if isinstance(component, dict)]


def _candidate_validation(candidate: dict[str, Any]) -> dict[str, Any]:
    validation = candidate.get("validation")
    if isinstance(validation, dict):
        return validation
    content = str(candidate.get("content") or "")
    language = str(candidate.get("model_language") or "lp").lower()
    if not content:
        return {"status": "missing", "issues": []}
    if language == "mps":
        return validate_mps_artifact(content)
    return validate_lp_artifact(content)


def _candidate_diagnostics(
    candidate: dict[str, Any],
    validation: dict[str, Any],
    technique_issues: list[ValidationIssue],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for item in candidate.get("diagnostics") or []:
        if isinstance(item, dict):
            diagnostics.append(item)
    diagnostics.extend(
        issue for issue in validation.get("issues", []) if isinstance(issue, dict)
    )
    diagnostics.extend(issue.__dict__ for issue in technique_issues)
    return diagnostics


def score_modeling_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Score a model/artifact candidate before choosing one answer to expose."""
    components = _candidate_components(candidate)
    relevant_components = _lp_relevant_components(components)
    content = str(candidate.get("content") or "")
    validation = _candidate_validation(candidate)
    technique_issues = validate_modeling_techniques(components, content=content)
    diagnostics = _candidate_diagnostics(candidate, validation, technique_issues)
    score = 0.0
    reasons: list[str] = []

    status = str(validation.get("status") or "").lower()
    if status == "valid":
        score += 40.0
        reasons.append("valid_artifact")
    elif status == "missing":
        score -= 10.0
        reasons.append("missing_artifact")
    else:
        score -= 25.0
        reasons.append("invalid_artifact")

    component_types = {
        str(component.get("component_type") or "") for component in relevant_components
    }
    for required in ("decision_variable", "objective", "constraint"):
        if required in component_types:
            score += 6.0
            reasons.append(f"has_{required}")
        else:
            score -= 8.0
            reasons.append(f"missing_{required}")
    if {"decision_variable", "objective", "constraint"} <= component_types:
        score += 8.0
        reasons.append("complete_core_components")

    sourced_components = sum(1 for component in components if _has_source_refs(component))
    if components:
        source_ratio = sourced_components / len(components)
        score += 12.0 * source_ratio
        reasons.append(f"source_ref_coverage={source_ratio:.2f}")
    detailed_components = sum(
        1
        for component in components
        if component.get("component_type")
        in {"parameter", "decision_variable", "objective", "constraint"}
        and _has_source_refs(component)
        and str(component.get("natural_text") or "").strip()
    )
    if detailed_components:
        detail_bonus = min(6.0, 0.25 * detailed_components)
        score += detail_bonus
        reasons.append(f"component_detail_bonus={detail_bonus:.2f}")

    solver = candidate.get("solver_result") or candidate.get("solver") or {}
    if isinstance(solver, dict):
        solver_status = str(solver.get("status") or "").lower()
        if solver_status == "optimal":
            score += 45.0
            reasons.append("solver_optimal")
        elif solver_status in {"solver_infeasible", "infeasible"}:
            score += 8.0
            reasons.append("solver_infeasible")
        elif solver_status in {"solver_unbounded", "unbounded"}:
            score += 4.0
            reasons.append("solver_unbounded")
        elif solver_status:
            score -= 12.0
            reasons.append(f"solver_{solver_status}")

    error_count = sum(1 for item in diagnostics if item.get("severity") == "error")
    warning_count = sum(1 for item in diagnostics if item.get("severity") == "warning")
    score -= 15.0 * error_count
    score -= 4.0 * warning_count
    if error_count:
        reasons.append(f"errors={error_count}")
    if warning_count:
        reasons.append(f"warnings={warning_count}")

    modeling_error = classify_modeling_error(issues=diagnostics)
    if modeling_error == "none":
        score += 5.0
    elif modeling_error == "low_model_completeness":
        score -= 8.0
    elif modeling_error == "objective_constraint_translation_error":
        score -= 12.0
    elif modeling_error == "semantic_misunderstanding":
        score -= 20.0

    return {
        "score": round(score, 3),
        "reasons": reasons,
        "validation": validation,
        "diagnostics": diagnostics,
        "modeling_error": modeling_error,
    }


def rank_modeling_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank candidate artifacts so each workflow round can emit one best answer."""
    scored = []
    for index, candidate in enumerate(candidates):
        score = score_modeling_candidate(candidate)
        scored.append({**candidate, **score, "input_index": index})
    scored.sort(key=lambda item: (float(item["score"]), -int(item["input_index"])), reverse=True)
    for rank, candidate in enumerate(scored, start=1):
        candidate["rank"] = rank
    return scored


def select_best_modeling_candidate(
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    ranked = rank_modeling_candidates(candidates)
    return ranked[0] if ranked else None


def available_solver_backends() -> dict[str, dict[str, Any]]:
    """Return solver backend capabilities without importing heavy modules."""
    scipy_available = importlib.util.find_spec("scipy") is not None
    return {
        "scipy_linprog": {
            "model_languages": ["lp"],
            "model_classes": ["continuous_linear_program"],
            "available": scipy_available,
            "executes": True,
        },
        "scipy_milp": {
            "model_languages": ["lp"],
            "model_classes": ["linear_or_mixed_integer_program"],
            "available": scipy_available,
            "executes": True,
        },
        "scipy_mps": {
            "model_languages": ["mps"],
            "model_classes": ["linear_or_mixed_integer_program"],
            "available": scipy_available,
            "executes": True,
        },
        "mps_validate": {
            "model_languages": ["mps"],
            "model_classes": ["linear_or_mixed_integer_program"],
            "available": True,
            "executes": False,
        },
    }


def _semantic_validation_result(issues: list[dict[str, Any]]) -> dict[str, Any]:
    status = (
        "invalid"
        if any(issue["severity"] == "error" for issue in issues)
        else "valid"
    )
    return {"status": status, "issues": issues}


def _objective_direction_hint(component: dict[str, Any]) -> str | None:
    text = _canonical_line(
        " ".join(
            str(component.get(field) or "")
            for field in ("formal_text", "natural_text", "metadata_json")
        )
    )
    maximize = re.search(r"\b(max|maximize|maximizes|maximization|maximum)\b", text)
    minimize = re.search(r"\b(min|minimize|minimizes|minimization|minimum)\b", text)
    if maximize and not minimize:
        return "maximize"
    if minimize and not maximize:
        return "minimize"
    return None


def _domain_hint(component: dict[str, Any]) -> str | None:
    text = _canonical_line(
        " ".join(
            str(component.get(field) or "")
            for field in ("domain", "formal_text", "natural_text")
        )
    )
    if any(token in text for token in ("binary", "{0,1}", "0/1", "boolean", "bool")):
        return "binary"
    if any(token in text for token in ("integer", "integers", "general")):
        return "integer"
    return None


def _constraint_component_present(
    component: dict[str, Any], parsed: ParsedLinearProgram
) -> bool:
    formal = _clean_lp_line(str(component.get("formal_text") or ""))
    symbol = str(component.get("symbol") or "").strip()
    body = formal.split(":", 1)[-1].strip() if formal else ""
    label = formal.split(":", 1)[0].strip() if ":" in formal else symbol
    candidates = {
        _canonical_line(item)
        for item in (formal, body, label, symbol)
        if str(item or "").strip()
    }
    for constraint in parsed.constraints:
        raw = _canonical_line(str(constraint.get("raw") or ""))
        raw_body = _canonical_line(str(constraint.get("raw") or "").split(":", 1)[-1])
        if any(
            candidate == raw
            or candidate == raw_body
            or (candidate and candidate in raw)
            or (raw and raw in candidate)
            for candidate in candidates
        ):
            return True
    return False


def validate_artifact_component_semantics(
    *,
    problem_id: int,
    content: str,
    model_language: str,
    linked_component_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Check an LP/MPS artifact against linked reviewed OR/MSE components."""
    if not linked_component_ids:
        return {"status": "valid", "issues": []}

    from . import api

    components = {
        int(component["id"]): component
        for component in api.get_optimization_components(problem_id)
        if component.get("id") is not None
    }
    linked_components = [
        components[component_id]
        for component_id in linked_component_ids
        if component_id in components
    ]
    if not linked_components:
        return {"status": "valid", "issues": []}

    try:
        parsed = (
            parse_mps_artifact(content)
            if model_language.lower() == "mps"
            else parse_lp_artifact(content)
        )
    except Exception as exc:
        return _semantic_validation_result(
            [
                {
                    "issue_type": "semantic_parse_error",
                    "severity": "error",
                    "message": (
                        "Artifact could not be parsed for semantic checks: "
                        f"{exc}"
                    ),
                }
            ]
        )

    issues: list[dict[str, Any]] = []
    variable_index = {variable: index for index, variable in enumerate(parsed.variables)}
    units_by_symbol: dict[str, dict[str, Any]] = {}

    for component in linked_components:
        component_id = int(component["id"])
        component_type = component.get("component_type")
        if component_type == "objective":
            expected_direction = _objective_direction_hint(component)
            if expected_direction and expected_direction != parsed.direction:
                issues.append(
                    {
                        "issue_type": "objective_direction_mismatch",
                        "severity": "error",
                        "component_id": component_id,
                        "message": (
                            f"Objective component {component_id} expects "
                            f"{expected_direction}, but artifact is "
                            f"{parsed.direction}."
                        ),
                    }
                )
        elif component_type == "decision_variable":
            symbol = str(component.get("symbol") or "").strip()
            if not symbol:
                continue
            index = variable_index.get(symbol)
            if index is None:
                issues.append(
                    {
                        "issue_type": "linked_decision_variable_missing",
                        "severity": "error",
                        "component_id": component_id,
                        "message": (
                            f"Decision variable component {component_id} symbol "
                            f"{symbol!r} is absent from the artifact."
                        ),
                    }
                )
                continue
            hint = _domain_hint(component)
            low, high = parsed.bounds[index]
            if hint == "binary" and (
                parsed.integrality[index] != 1
                or not math.isclose(low or 0.0, 0.0)
                or not math.isclose(high or 0.0, 1.0)
            ):
                issues.append(
                    {
                        "issue_type": "linked_variable_domain_mismatch",
                        "severity": "error",
                        "component_id": component_id,
                        "message": (
                            f"Decision variable component {component_id} "
                            "expects binary domain, but artifact "
                            "bounds/integrality differ."
                        ),
                    }
                )
            elif hint == "integer" and parsed.integrality[index] != 1:
                issues.append(
                    {
                        "issue_type": "linked_variable_domain_mismatch",
                        "severity": "error",
                        "component_id": component_id,
                        "message": (
                            f"Decision variable component {component_id} "
                            "expects integer domain, but artifact is continuous."
                        ),
                    }
                )
        elif component_type == "constraint" and not _constraint_component_present(
            component, parsed
        ):
            issues.append(
                {
                    "issue_type": "linked_constraint_missing",
                    "severity": "error",
                    "component_id": component_id,
                    "message": (
                        f"Constraint component {component_id} is not preserved "
                        "in the artifact."
                    ),
                }
            )
        elif component_type == "parameter":
            symbol = str(component.get("symbol") or "").strip()
            unit = str(component.get("unit") or "").strip()
            if symbol and unit:
                prior = units_by_symbol.setdefault(symbol, {"unit": unit, "ids": []})
                prior["ids"].append(component_id)
                if prior["unit"] != unit:
                    issues.append(
                        {
                            "issue_type": "linked_parameter_unit_conflict",
                            "severity": "error",
                            "component_id": component_id,
                            "message": (
                                f"Linked parameter symbol {symbol!r} has "
                                f"conflicting units: {prior['unit']!r} and "
                                f"{unit!r}."
                            ),
                        }
                    )

    return _semantic_validation_result(issues)


def build_lp_repair_prompt(content: str, validation: dict[str, Any]) -> str:
    issues = validation.get("issues") or []
    issue_lines = "\n".join(
        f"- {item.get('issue_type')}: {item.get('message')}" for item in issues
    )
    return f"""\
Repair this LP artifact. Return LP model text only, no markdown fences and no
explanation.

Required sections:
- Minimize or Maximize
- Subject To
- optional Bounds
- optional Binary or General
- End

Validation issues:
{issue_lines or "- none"}

LP artifact:
{content}
"""


def build_mps_repair_prompt(content: str, validation: dict[str, Any]) -> str:
    issues = validation.get("issues") or []
    issue_lines = "\n".join(
        f"- {item.get('issue_type')}: {item.get('message')}" for item in issues
    )
    return f"""\
Repair this MPS artifact. Return MPS model text only, no markdown fences and no
explanation. Preserve rows, columns, RHS, bounds, objective sense, and variable
names unless the validation issue requires a minimal structural fix.

Validation issues:
{issue_lines or "- none"}

MPS artifact:
{content}
"""


def build_semantic_repair_prompt(
    *,
    content: str,
    model_language: str,
    semantic_validation: dict[str, Any],
    linked_components: list[dict[str, Any]],
) -> str:
    issues = semantic_validation.get("issues") or []
    issue_lines = "\n".join(
        f"- {item.get('issue_type')}: {item.get('message')}" for item in issues
    )
    component_rows = [
        {
            "id": component.get("id"),
            "component_type": component.get("component_type"),
            "natural_text": component.get("natural_text"),
            "formal_text": component.get("formal_text"),
            "symbol": component.get("symbol"),
            "unit": component.get("unit"),
            "domain": component.get("domain"),
            "review_status": component.get("review_status"),
        }
        for component in linked_components
    ]
    return f"""\
Repair this {model_language.upper()} optimization artifact so it stays aligned
with the reviewed components. Return {model_language.upper()} model text only,
no markdown fences and no explanation.

Semantic issues:
{issue_lines or "- none"}

Reviewed linked components:
{json.dumps(component_rows, ensure_ascii=True, sort_keys=True)}

Artifact:
{content}
"""


def normalize_lp_repair_candidate(text: str) -> str:
    """Extract bare LP text from a repair response without trusting prose."""
    candidate = (text or "").strip()
    fence = re.fullmatch(
        r"```(?:lp|mps|text)?\s*(.*?)\s*```",
        candidate,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fence:
        candidate = fence.group(1).strip()
    return candidate


def normalize_mps_repair_candidate(text: str) -> str:
    """Extract bare MPS text from a repair response without trusting prose."""
    candidate = (text or "").strip()
    fence = re.fullmatch(
        r"```(?:mps|text)?\s*(.*?)\s*```",
        candidate,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fence:
        candidate = fence.group(1).strip()
    return candidate


def apply_lp_repair_candidate(
    *,
    topic_id: int,
    problem_id: int,
    source_artifact_id: int,
    source_content: str,
    repaired_content: str,
    linked_component_ids: list[int] | None = None,
    generator_role: str = "lp_repair",
) -> dict[str, Any]:
    """Persist a repaired LP artifact only through the LP validator gate.

    The original artifact is never overwritten. A candidate repair becomes
    accepted only when it is valid LP model text with no explanatory wrapper.
    """
    from . import api

    source_validation = validate_lp_artifact(source_content)
    candidate = normalize_lp_repair_candidate(repaired_content)
    validation = validate_lp_artifact(candidate)
    semantic_validation = {"status": "valid", "issues": []}
    if validation["status"] == "valid":
        semantic_validation = validate_artifact_component_semantics(
            problem_id=problem_id,
            content=candidate,
            model_language="lp",
            linked_component_ids=linked_component_ids,
        )
    accepted = (
        validation["status"] == "valid" and semantic_validation["status"] == "valid"
    )
    artifact_id = api.insert_optimization_artifact(
        problem_id=problem_id,
        topic_id=topic_id,
        artifact_type="lp_model",
        model_language="lp",
        content=candidate,
        parser_status=validation["status"],
        parser_notes=json.dumps(
            {
                "source_validation": source_validation,
                "repair_validation": validation,
                "semantic_validation": semantic_validation,
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        linked_component_ids_json=json.dumps(
            linked_component_ids or [], ensure_ascii=True
        ),
        component_fingerprints_json=_component_fingerprints_for_ids(
            problem_id, linked_component_ids
        ),
        generator_role=generator_role,
        source_artifact_id=source_artifact_id,
        repair_status="accepted" if accepted else "rejected",
    )
    diagnostic_ids: list[int] = []
    for issue in [*validation.get("issues", []), *semantic_validation.get("issues", [])]:
        diagnostic_ids.append(
            api.insert_model_diagnostic(
                problem_id=problem_id,
                topic_id=topic_id,
                component_id=issue.get("component_id"),
                artifact_id=artifact_id,
                diagnostic_type=issue["issue_type"],
                severity=issue["severity"],
                message=issue["message"],
            )
        )
    return {
        "artifact_id": artifact_id,
        "accepted": accepted,
        "validation": validation,
        "diagnostic_ids": diagnostic_ids,
        "content": candidate,
    }


def apply_mps_repair_candidate(
    *,
    topic_id: int,
    problem_id: int,
    source_artifact_id: int,
    source_content: str,
    repaired_content: str,
    linked_component_ids: list[int] | None = None,
    generator_role: str = "mps_repair",
) -> dict[str, Any]:
    """Persist a repaired MPS artifact only through the MPS validator gate."""
    from . import api

    source_validation = validate_mps_artifact(source_content)
    candidate = normalize_mps_repair_candidate(repaired_content)
    validation = validate_mps_artifact(candidate)
    semantic_validation = {"status": "valid", "issues": []}
    if validation["status"] == "valid":
        semantic_validation = validate_artifact_component_semantics(
            problem_id=problem_id,
            content=candidate,
            model_language="mps",
            linked_component_ids=linked_component_ids,
        )
    accepted = (
        validation["status"] == "valid" and semantic_validation["status"] == "valid"
    )
    artifact_id = api.insert_optimization_artifact(
        problem_id=problem_id,
        topic_id=topic_id,
        artifact_type="mps_model",
        model_language="mps",
        content=candidate,
        parser_status=validation["status"],
        parser_notes=json.dumps(
            {
                "source_validation": source_validation,
                "repair_validation": validation,
                "semantic_validation": semantic_validation,
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        linked_component_ids_json=json.dumps(
            linked_component_ids or [], ensure_ascii=True
        ),
        component_fingerprints_json=_component_fingerprints_for_ids(
            problem_id, linked_component_ids
        ),
        generator_role=generator_role,
        source_artifact_id=source_artifact_id,
        repair_status="accepted" if accepted else "rejected",
    )
    diagnostic_ids: list[int] = []
    for issue in [*validation.get("issues", []), *semantic_validation.get("issues", [])]:
        diagnostic_ids.append(
            api.insert_model_diagnostic(
                problem_id=problem_id,
                topic_id=topic_id,
                component_id=issue.get("component_id"),
                artifact_id=artifact_id,
                diagnostic_type=issue["issue_type"],
                severity=issue["severity"],
                message=issue["message"],
            )
        )
    return {
        "artifact_id": artifact_id,
        "accepted": accepted,
        "validation": validation,
        "diagnostic_ids": diagnostic_ids,
        "content": candidate,
    }


async def repair_lp_artifact_with_llm(
    *,
    topic_id: int,
    problem_id: int,
    source_artifact_id: int,
    source_content: str,
    linked_component_ids: list[int] | None = None,
    provider_profile: str = "minimax",
) -> dict[str, Any]:
    """Request a constrained LP syntax repair and persist the gated candidate."""
    from .broker import DEFAULT_MAX_TOKENS, llm_call

    validation = validate_lp_artifact(source_content)
    prompt = build_lp_repair_prompt(source_content, validation)
    response = await llm_call(
        prompt,
        system_prompt="You repair LP artifacts. Return model text only.",
        provider_profile=provider_profile,
        role="or_lp_repair",
        max_tokens=DEFAULT_MAX_TOKENS,
        require_json=False,
    )
    result = apply_lp_repair_candidate(
        topic_id=topic_id,
        problem_id=problem_id,
        source_artifact_id=source_artifact_id,
        source_content=source_content,
        repaired_content=response.text,
        linked_component_ids=linked_component_ids,
        generator_role="or_lp_repair",
    )
    return {**result, "raw_text": response.text}


async def repair_mps_artifact_with_llm(
    *,
    topic_id: int,
    problem_id: int,
    source_artifact_id: int,
    source_content: str,
    linked_component_ids: list[int] | None = None,
    provider_profile: str = "minimax",
) -> dict[str, Any]:
    """Request a constrained MPS syntax repair and persist the gated candidate."""
    from .broker import DEFAULT_MAX_TOKENS, llm_call

    validation = validate_mps_artifact(source_content)
    prompt = build_mps_repair_prompt(source_content, validation)
    response = await llm_call(
        prompt,
        system_prompt="You repair MPS artifacts. Return model text only.",
        provider_profile=provider_profile,
        role="or_mps_repair",
        max_tokens=DEFAULT_MAX_TOKENS,
        require_json=False,
    )
    result = apply_mps_repair_candidate(
        topic_id=topic_id,
        problem_id=problem_id,
        source_artifact_id=source_artifact_id,
        source_content=source_content,
        repaired_content=response.text,
        linked_component_ids=linked_component_ids,
        generator_role="or_mps_repair",
    )
    return {**result, "raw_text": response.text}


async def repair_artifact_semantics_with_llm(
    *,
    topic_id: int,
    problem_id: int,
    source_artifact_id: int,
    source_content: str,
    model_language: str,
    linked_component_ids: list[int],
    provider_profile: str = "minimax",
) -> dict[str, Any]:
    """Request an evidence-aware semantic repair and re-apply deterministic gates."""
    from . import api
    from .broker import DEFAULT_MAX_TOKENS, llm_call

    language = model_language.lower()
    if language not in {"lp", "mps"}:
        raise ValueError(f"Unsupported model_language for semantic repair: {model_language}")

    components_by_id = {
        int(component["id"]): component
        for component in api.get_optimization_components(problem_id)
        if component.get("id") is not None
    }
    linked_components = [
        components_by_id[component_id]
        for component_id in linked_component_ids
        if component_id in components_by_id
    ]
    semantic_validation = validate_artifact_component_semantics(
        problem_id=problem_id,
        content=source_content,
        model_language=language,
        linked_component_ids=linked_component_ids,
    )
    prompt = build_semantic_repair_prompt(
        content=source_content,
        model_language=language,
        semantic_validation=semantic_validation,
        linked_components=linked_components,
    )
    response = await llm_call(
        prompt,
        system_prompt=(
            "You repair optimization model semantics. Return model text only."
        ),
        provider_profile=provider_profile,
        role=f"or_{language}_semantic_repair",
        max_tokens=DEFAULT_MAX_TOKENS,
        require_json=False,
    )
    apply_fn = apply_mps_repair_candidate if language == "mps" else apply_lp_repair_candidate
    result = apply_fn(
        topic_id=topic_id,
        problem_id=problem_id,
        source_artifact_id=source_artifact_id,
        source_content=source_content,
        repaired_content=response.text,
        linked_component_ids=linked_component_ids,
        generator_role=f"or_{language}_semantic_repair",
    )
    return {
        **result,
        "raw_text": response.text,
        "source_semantic_validation": semantic_validation,
    }


def persist_lp_artifact(
    *,
    topic_id: int,
    problem_id: int,
    content: str,
    linked_component_ids: list[int] | None = None,
    generator_role: str | None = None,
    component_fingerprints_json: str | None = None,
) -> dict[str, Any]:
    from . import api

    validation = validate_lp_artifact(content)
    artifact_id = api.insert_optimization_artifact(
        problem_id=problem_id,
        topic_id=topic_id,
        artifact_type="lp_model",
        model_language="lp",
        content=content,
        parser_status=validation["status"],
        parser_notes=json.dumps(validation, ensure_ascii=True, sort_keys=True),
        linked_component_ids_json=json.dumps(
            linked_component_ids or [], ensure_ascii=True
        ),
        component_fingerprints_json=component_fingerprints_json
        or _component_fingerprints_for_ids(problem_id, linked_component_ids),
        generator_role=generator_role,
    )
    diagnostic_ids: list[int] = []
    for issue in validation.get("issues", []):
        diagnostic_ids.append(
            api.insert_model_diagnostic(
                problem_id=problem_id,
                topic_id=topic_id,
                artifact_id=artifact_id,
                diagnostic_type=issue["issue_type"],
                severity=issue["severity"],
                message=issue["message"],
            )
        )
    return {
        "artifact_id": artifact_id,
        "validation": validation,
        "diagnostic_ids": diagnostic_ids,
    }


def propagate_component_status_to_solver_evidence(
    topic_id: int, *, problem_id: int | None = None
) -> dict[str, Any]:
    """Flag artifacts and solver runs whose linked components are no longer valid.

    Historical solver runs are preserved. Open error diagnostics created here
    make their `CodeEvidence` inactive for JTMS support checks.
    """
    from . import db

    artifact_params: list[Any] = [topic_id]
    artifact_where = ["topic_id = ?"]
    if problem_id is not None:
        artifact_where.append("problem_id = ?")
        artifact_params.append(problem_id)

    diagnostics_created = 0
    diagnostics_resolved = 0
    affected_artifact_ids: set[int] = set()
    affected_solver_run_ids: set[int] = set()
    artifacts_checked = 0

    with db.get_db() as conn:
        artifacts = conn.execute(
            f"""
            SELECT *
            FROM OptimizationArtifact
            WHERE {' AND '.join(artifact_where)}
            ORDER BY id ASC
            """,
            artifact_params,
        ).fetchall()

        for artifact in artifacts:
            artifacts_checked += 1
            artifact_id = int(artifact["id"])
            linked_component_ids = _parse_component_ids(
                artifact["linked_component_ids_json"]
            )
            if not linked_component_ids:
                continue

            placeholders = ",".join("?" for _ in linked_component_ids)
            components = conn.execute(
                f"""
                SELECT *
                FROM OptimizationComponent
                WHERE id IN ({placeholders})
                """,
                linked_component_ids,
            ).fetchall()
            components_by_id = {
                int(component["id"]): dict(component) for component in components
            }
            stored_fingerprints = _component_fingerprint_index(
                artifact["component_fingerprints_json"]
            )
            solver_runs = conn.execute(
                "SELECT id FROM SolverRun WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchall()

            current_issue_keys: set[tuple[str, int | None]] = set()
            for component_id in linked_component_ids:
                component = components_by_id.get(component_id)
                issue_type: str | None = None
                issue_component_id: int | None = component_id
                if component is None:
                    issue_type = "linked_component_missing"
                    issue_component_id = None
                    message = (
                        f"Artifact O{artifact_id} links to missing component "
                        f"{component_id}."
                    )
                elif component.get("review_status") not in LP_READY_REVIEW_STATUSES:
                    issue_type = "linked_component_inactive"
                    message = (
                        f"Artifact O{artifact_id} depends on component "
                        f"{component_id} with review_status="
                        f"{component.get('review_status') or '<missing>'}."
                    )
                elif stored_fingerprints:
                    current_fingerprint = _component_fingerprint_index(
                        build_component_fingerprints_json([component])
                    ).get(component_id)
                    if (
                        current_fingerprint
                        and stored_fingerprints.get(component_id) != current_fingerprint
                    ):
                        issue_type = "linked_component_changed"
                        message = (
                            f"Artifact O{artifact_id} depends on component "
                            f"{component_id}, which changed after artifact generation."
                        )
                if issue_type is None:
                    continue

                current_issue_keys.add((issue_type, issue_component_id))
                affected_artifact_ids.add(artifact_id)
                created = _ensure_stale_diagnostic(
                    conn,
                    problem_id=int(artifact["problem_id"]),
                    topic_id=topic_id,
                    diagnostic_type=issue_type,
                    message=message,
                    component_id=issue_component_id,
                    artifact_id=artifact_id,
                )
                if created is not None:
                    diagnostics_created += 1

                for solver_run in solver_runs:
                    solver_run_id = int(solver_run["id"])
                    affected_solver_run_ids.add(solver_run_id)
                    created = _ensure_stale_diagnostic(
                        conn,
                        problem_id=int(artifact["problem_id"]),
                        topic_id=topic_id,
                        diagnostic_type=issue_type,
                        message=message,
                        component_id=issue_component_id,
                        artifact_id=artifact_id,
                        solver_run_id=solver_run_id,
                    )
                    if created is not None:
                        diagnostics_created += 1

            stale_types = tuple(sorted(COMPONENT_STALE_DIAGNOSTICS))
            type_placeholders = ",".join("?" for _ in stale_types)
            open_rows = conn.execute(
                f"""
                SELECT id, diagnostic_type, component_id
                FROM ModelDiagnostic
                WHERE topic_id = ?
                  AND artifact_id = ?
                  AND diagnostic_type IN ({type_placeholders})
                  AND status = 'open'
                """,
                (topic_id, artifact_id, *stale_types),
            ).fetchall()
            for row in open_rows:
                key = (row["diagnostic_type"], row["component_id"])
                if key in current_issue_keys:
                    continue
                cursor = conn.execute(
                    """
                    UPDATE ModelDiagnostic
                    SET status = 'resolved',
                        resolved_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (row["id"],),
                )
                diagnostics_resolved += cursor.rowcount

    return {
        "artifacts_checked": artifacts_checked,
        "diagnostics_created": diagnostics_created,
        "diagnostics_resolved": diagnostics_resolved,
        "affected_artifact_ids": sorted(affected_artifact_ids),
        "affected_solver_run_ids": sorted(affected_solver_run_ids),
    }


def _canonical_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip().lower())


def _parse_float(value: str) -> float:
    return float(value.replace(",", ""))


def _is_number(value: str) -> bool:
    try:
        _parse_float(value)
        return True
    except ValueError:
        return False


def _normalize_linear_expression_text(expr: str) -> str:
    return (
        str(expr or "")
        .replace("\u00a0", " ")
        .replace("\u202f", " ")
        .replace("\u2212", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2026", "...")
        .replace("\u22ef", "...")
    )


def _expand_ellipsis_ranges(expr: str) -> str:
    """Expand finite variable ranges such as `x_1 + ... + x_3`."""

    def values(start_text: str, end_text: str) -> list[int] | None:
        start = int(start_text)
        end = int(end_text)
        count = abs(end - start) + 1
        if count > 200:
            return None
        step = 1 if end >= start else -1
        return list(range(start, end + step, step))

    coeff_range = re.compile(
        r"(?<![A-Za-z0-9_])"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s+"
        r"([A-Za-z_][A-Za-z0-9_]*)_(\d+)"
        r"\s*\+\s*\.{3}\s*\+\s*"
        r"\1\s+\2_(\d+)"
        r"(?![A-Za-z0-9_])"
    )
    underscore_range = re.compile(
        r"(?<![A-Za-z0-9_])"
        r"([A-Za-z_][A-Za-z0-9_]*)_(\d+)"
        r"\s*\+\s*\.{3}\s*\+\s*"
        r"\1_(\d+)"
        r"(?![A-Za-z0-9_])"
    )
    compact_range = re.compile(
        r"(?<![A-Za-z0-9_])"
        r"([A-Za-z_]+)(\d+)"
        r"\s*\+\s*\.{3}\s*\+\s*"
        r"\1(\d+)"
        r"(?![A-Za-z0-9_])"
    )

    def coeff_repl(match: re.Match[str]) -> str:
        expanded = values(match.group(3), match.group(4))
        if expanded is None:
            return match.group(0)
        coeff = match.group(1)
        prefix = match.group(2)
        return " + ".join(f"{coeff} {prefix}_{index}" for index in expanded)

    def underscore_repl(match: re.Match[str]) -> str:
        expanded = values(match.group(2), match.group(3))
        if expanded is None:
            return match.group(0)
        prefix = match.group(1)
        return " + ".join(f"{prefix}_{index}" for index in expanded)

    def compact_repl(match: re.Match[str]) -> str:
        expanded = values(match.group(2), match.group(3))
        if expanded is None:
            return match.group(0)
        prefix = match.group(1)
        return " + ".join(f"{prefix}{index}" for index in expanded)

    previous = ""
    expanded = _normalize_linear_expression_text(expr)
    while previous != expanded:
        previous = expanded
        expanded = coeff_range.sub(coeff_repl, expanded)
        expanded = underscore_range.sub(underscore_repl, expanded)
        expanded = compact_range.sub(compact_repl, expanded)
    return expanded


def _expand_scalar_parentheses(expr: str) -> str:
    """Expand simple linear forms such as `0.6(x + y)` before parsing."""
    pattern = re.compile(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\(\s*([^()]+?)\s*\)"
    )

    def repl(match: re.Match[str]) -> str:
        scalar = _parse_float(match.group(1))
        inner = match.group(2)
        coeffs, constant = _parse_linear_expression(inner)
        scaled = {var: scalar * coeff for var, coeff in coeffs.items()}
        return _format_linear_expression(scaled, scalar * constant)

    previous = ""
    expanded = expr
    while previous != expanded:
        previous = expanded
        expanded = pattern.sub(repl, expanded)
    return expanded


def _parse_linear_expression(expr: str) -> tuple[dict[str, float], float]:
    normalized = (
        _expand_scalar_parentheses(_expand_ellipsis_ranges(expr))
        .replace("*", " ")
        .replace("-", "+-")
        .replace("++", "+")
        .strip()
    )
    coeffs: dict[str, float] = {}
    constant = 0.0
    for raw_part in normalized.split("+"):
        part = raw_part.strip()
        if not part:
            continue
        compact = part.replace(" ", "")
        compact_match = re.fullmatch(
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))([A-Za-z_][A-Za-z0-9_]*)",
            compact,
        )
        if compact_match:
            coeff = _parse_float(compact_match.group(1))
            var = compact_match.group(2)
        else:
            pieces = part.split()
            if len(pieces) == 1:
                token = pieces[0]
                if _is_number(token):
                    constant += _parse_float(token)
                    continue
                coeff = -1.0 if token.startswith("-") else 1.0
                var = token[1:] if token.startswith("-") else token
            elif len(pieces) == 2 and _is_number(pieces[0]):
                coeff = _parse_float(pieces[0])
                var = pieces[1]
            elif len(pieces) == 2 and pieces[0] in {"+", "-"}:
                coeff = -1.0 if pieces[0] == "-" else 1.0
                var = pieces[1]
            else:
                raise ValueError(f"Unsupported linear term: {part!r}")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", var):
            raise ValueError(f"Unsupported variable name: {var!r}")
        coeffs[var] = coeffs.get(var, 0.0) + coeff
    return coeffs, constant


def _split_lp_sections(content: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = ""
    for raw_line in (content or "").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("\\"):
            continue
        canonical = _canonical_line(stripped)
        matched_section = ""
        section_tail = ""
        for section, aliases in LP_SECTION_ALIASES.items():
            for alias in sorted(aliases, key=len, reverse=True):
                if canonical == alias or canonical.startswith(alias + " "):
                    matched_section = section
                    section_tail = stripped[len(alias) :].strip()
                    break
            if matched_section:
                break
        if matched_section:
            current = matched_section
            sections.setdefault(current, [])
            if section_tail:
                sections[current].append(section_tail)
            continue
        if current:
            sections.setdefault(current, []).append(stripped)
    return sections


def _section_variables(lines: list[str]) -> set[str]:
    variables: set[str] = set()
    for line in lines:
        for token in re.split(r"[\s,]+", line.strip()):
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
                variables.add(token)
    return variables


def parse_lp_artifact(content: str) -> ParsedLinearProgram:
    validation = validate_lp_artifact(content, strict=False)
    if validation["status"] != "valid":
        raise ValueError("LP artifact is structurally invalid")
    sections = _split_lp_sections(content)

    direction_line = next(
        line
        for line in (content or "").splitlines()
        if any(
            _canonical_line(line) == alias
            or _canonical_line(line).startswith(alias + " ")
            for alias in LP_SECTION_ALIASES["objective"]
        )
    )
    direction_canonical = _canonical_line(direction_line)
    direction = (
        "maximize"
        if any(
            direction_canonical == alias or direction_canonical.startswith(alias + " ")
            for alias in LP_MAXIMIZE_ALIASES
        )
        else "minimize"
    )

    objective_lines = sections.get("objective") or []
    if not objective_lines:
        raise ValueError("LP objective section is empty")
    objective_expr = _strip_objective_lhs(objective_lines[0].split(":", 1)[-1])
    objective_coeffs, objective_constant = _parse_linear_expression(objective_expr)
    if not math.isclose(objective_constant, 0.0):
        raise ValueError("Objective constants are not supported")

    constraint_rows: list[dict[str, Any]] = []
    variables = set(objective_coeffs)
    for raw_constraint in sections.get("constraints") or []:
        body = raw_constraint.split(":", 1)[-1].strip()
        match = re.search(r"(<=|>=|=)", body)
        if not match:
            raise ValueError(f"Constraint lacks an operator: {raw_constraint!r}")
        op = match.group(1)
        left = body[: match.start()].strip()
        right = body[match.end() :].strip()
        left_coeffs, left_constant = _parse_linear_expression(left)
        if _is_number(right):
            coeffs = left_coeffs
            rhs = _parse_float(right) - left_constant
        else:
            right_coeffs, right_constant = _parse_linear_expression(right)
            coeffs = dict(left_coeffs)
            for var, coeff in right_coeffs.items():
                coeffs[var] = coeffs.get(var, 0.0) - coeff
            rhs = right_constant - left_constant
        variables.update(coeffs)
        constraint_rows.append(
            {
                "coeffs": coeffs,
                "operator": op,
                "rhs": rhs,
                "raw": raw_constraint,
            }
        )

    variable_names = tuple(sorted(variables))
    bounds_by_var: dict[str, tuple[float | None, float | None]] = {
        var: (0.0, None) for var in variable_names
    }
    for raw_bound in sections.get("bounds") or []:
        line = raw_bound.strip()
        if line.lower().endswith(" free"):
            var = line.split()[0]
            bounds_by_var[var] = (None, None)
            continue
        two_sided = re.fullmatch(
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*<=\s*([A-Za-z_][A-Za-z0-9_]*)\s*<=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
            line,
        )
        if two_sided:
            bounds_by_var[two_sided.group(2)] = (
                _parse_float(two_sided.group(1)),
                _parse_float(two_sided.group(3)),
            )
            continue
        one_sided = re.fullmatch(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*(<=|>=)\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
            line,
        )
        if not one_sided:
            raise ValueError(f"Unsupported bound syntax: {raw_bound!r}")
        var, op, value = one_sided.groups()
        low, high = bounds_by_var.get(var, (0.0, None))
        if op == ">=":
            low = _parse_float(value)
        else:
            high = _parse_float(value)
        bounds_by_var[var] = (low, high)

    binary_vars = _section_variables(sections.get("binary") or [])
    general_vars = _section_variables(sections.get("general") or [])
    variables.update(binary_vars)
    variables.update(general_vars)
    variable_names = tuple(sorted(variables))
    for var in binary_vars:
        low, high = bounds_by_var.get(var, (0.0, None))
        low = max(low if low is not None else 0.0, 0.0)
        high = min(high if high is not None else 1.0, 1.0)
        bounds_by_var[var] = (low, high)

    objective = tuple(objective_coeffs.get(var, 0.0) for var in variable_names)
    constraints = tuple(constraint_rows)
    bounds = tuple(bounds_by_var.get(var, (0.0, None)) for var in variable_names)
    integrality = tuple(
        1 if var in binary_vars or var in general_vars else 0 for var in variable_names
    )
    return ParsedLinearProgram(
        direction,
        variable_names,
        objective,
        constraints,
        bounds,
        integrality,
    )


def _linprog_solver_payload(parsed: ParsedLinearProgram) -> dict[str, Any]:
    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return {
            "status": "solver_unavailable",
            "stdout": "",
            "stderr": str(exc),
            "objective_value": None,
            "variable_values": {},
        }

    c = list(parsed.objective)
    if parsed.direction == "maximize":
        c = [-value for value in c]
    a_ub: list[list[float]] = []
    b_ub: list[float] = []
    a_eq: list[list[float]] = []
    b_eq: list[float] = []
    for constraint in parsed.constraints:
        row = [constraint["coeffs"].get(var, 0.0) for var in parsed.variables]
        if constraint["operator"] == "<=":
            a_ub.append(row)
            b_ub.append(constraint["rhs"])
        elif constraint["operator"] == ">=":
            a_ub.append([-value for value in row])
            b_ub.append(-constraint["rhs"])
        else:
            a_eq.append(row)
            b_eq.append(constraint["rhs"])
    result = linprog(
        c,
        A_ub=a_ub or None,
        b_ub=b_ub or None,
        A_eq=a_eq or None,
        b_eq=b_eq or None,
        bounds=list(parsed.bounds),
        method="highs",
    )
    status_map = {
        0: "optimal",
        1: "solver_timeout",
        2: "solver_infeasible",
        3: "solver_unbounded",
    }
    status = status_map.get(int(result.status), "solver_failed")
    objective_value = None
    variable_values: dict[str, float] = {}
    if result.success:
        objective_value = float(result.fun)
        if parsed.direction == "maximize":
            objective_value = -objective_value
        variable_values = {
            var: float(value) for var, value in zip(parsed.variables, result.x)
        }
    return {
        "status": status,
        "stdout": str(result.message),
        "stderr": "",
        "objective_value": objective_value,
        "variable_values": variable_values,
    }


def _milp_solver_payload(parsed: ParsedLinearProgram) -> dict[str, Any]:
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError as exc:
        return {
            "status": "solver_unavailable",
            "stdout": "",
            "stderr": str(exc),
            "objective_value": None,
            "variable_values": {},
        }

    c = list(parsed.objective)
    if parsed.direction == "maximize":
        c = [-value for value in c]
    rows: list[list[float]] = []
    lower: list[float] = []
    upper: list[float] = []
    for constraint in parsed.constraints:
        row = [constraint["coeffs"].get(var, 0.0) for var in parsed.variables]
        rows.append(row)
        if constraint["operator"] == "<=":
            lower.append(-math.inf)
            upper.append(constraint["rhs"])
        elif constraint["operator"] == ">=":
            lower.append(constraint["rhs"])
            upper.append(math.inf)
        else:
            lower.append(constraint["rhs"])
            upper.append(constraint["rhs"])
    bound_lows = [low if low is not None else -math.inf for low, _ in parsed.bounds]
    bound_highs = [high if high is not None else math.inf for _, high in parsed.bounds]
    constraints = (
        LinearConstraint(rows, lower, upper)
        if rows
        else None
    )
    result = milp(
        c,
        integrality=list(parsed.integrality),
        bounds=Bounds(bound_lows, bound_highs),
        constraints=constraints,
    )
    status_map = {
        0: "optimal",
        1: "solver_timeout",
        2: "solver_infeasible",
        3: "solver_unbounded",
    }
    status = status_map.get(int(result.status), "solver_failed")
    objective_value = None
    variable_values: dict[str, float] = {}
    if result.success:
        objective_value = float(result.fun)
        if parsed.direction == "maximize":
            objective_value = -objective_value
        variable_values = {
            var: float(value) for var, value in zip(parsed.variables, result.x)
        }
    return {
        "status": status,
        "stdout": str(result.message),
        "stderr": "",
        "objective_value": objective_value,
        "variable_values": variable_values,
    }


def solve_lp_artifact(
    *,
    topic_id: int,
    problem_id: int,
    artifact_id: int,
    content: str,
    solver_backend: str = "scipy_linprog",
    persist_code_evidence: bool = True,
) -> dict[str, Any]:
    from . import api

    validation = validate_lp_artifact(content)
    code_evidence_id = None
    if validation["status"] != "valid":
        status = "invalid_artifact_syntax"
        stdout = ""
        stderr = json.dumps(validation, ensure_ascii=True, sort_keys=True)
        objective_value = None
        variable_values: dict[str, float] = {}
    else:
        try:
            parsed = parse_lp_artifact(content)
            if solver_backend == "scipy_linprog":
                if any(parsed.integrality):
                    raise NotImplementedError(
                        "Integer and binary LP artifacts require a MILP solver"
                    )
                payload = _linprog_solver_payload(parsed)
            elif solver_backend == "scipy_milp":
                payload = _milp_solver_payload(parsed)
            else:
                raise NotImplementedError(f"Unsupported solver backend: {solver_backend}")
            status = payload["status"]
            stdout = payload["stdout"]
            stderr = payload["stderr"]
            objective_value = payload["objective_value"]
            variable_values = payload["variable_values"]
        except NotImplementedError as exc:
            status = "unsupported_model_class"
            stdout = ""
            stderr = str(exc)
            objective_value = None
            variable_values = {}
        except Exception as exc:
            status = "solver_parse_error"
            stdout = ""
            stderr = str(exc)
            objective_value = None
            variable_values = {}

    if persist_code_evidence:
        code_evidence_id = api.insert_code_evidence(
            topic_id,
            None,
            hypothesis=f"Solve LP artifact O{artifact_id} with {solver_backend}",
            source_code=content,
            stdout=stdout,
            stderr=stderr,
            exit_code=0 if status == "optimal" else 1,
            execution_time_s=None,
            iterations=1,
            success=status == "optimal",
            requesting_role="or_solver",
            summary=f"LP solver status: {status}",
        )
    solver_run_id = api.insert_solver_run(
        artifact_id=artifact_id,
        problem_id=problem_id,
        topic_id=topic_id,
        solver_backend=solver_backend,
        status=status,
        objective_value=objective_value,
        variable_values_json=json.dumps(variable_values, ensure_ascii=True, sort_keys=True),
        stdout=stdout,
        stderr=stderr,
        error_trace=stderr if status != "optimal" else None,
        code_evidence_id=code_evidence_id,
    )
    diagnostic_id = None
    diagnostic_type = classify_solver_failure(status, stdout=stdout, stderr=stderr)
    if diagnostic_type != "none":
        diagnostic_id = api.insert_model_diagnostic(
            problem_id=problem_id,
            topic_id=topic_id,
            artifact_id=artifact_id,
            solver_run_id=solver_run_id,
            diagnostic_type=diagnostic_type,
            severity="error" if status != "solver_unavailable" else "warning",
            message=stderr or stdout or status,
        )
    return {
        "solver_run_id": solver_run_id,
        "code_evidence_id": code_evidence_id,
        "diagnostic_id": diagnostic_id,
        "status": status,
        "objective_value": objective_value,
        "variable_values": variable_values,
    }


def solve_mps_artifact(
    *,
    topic_id: int,
    problem_id: int,
    artifact_id: int,
    content: str,
    solver_backend: str = "scipy_mps",
    persist_code_evidence: bool = True,
) -> dict[str, Any]:
    """Solve or validate an MPS artifact through a registered backend."""
    from . import api

    validation = validate_mps_artifact(content)
    code_evidence_id = None
    objective_value = None
    variable_values: dict[str, float] = {}
    stdout = ""
    if validation["status"] != "valid":
        status = "invalid_artifact_syntax"
        stderr = json.dumps(validation, ensure_ascii=True, sort_keys=True)
    elif solver_backend == "mps_validate":
        status = "unsupported_model_class"
        stderr = (
            "MPS artifact is structurally valid; mps_validate does not execute "
            "optimization."
        )
    elif solver_backend == "scipy_mps":
        try:
            parsed = parse_mps_artifact(content)
            if any(parsed.integrality):
                payload = _milp_solver_payload(parsed)
            else:
                payload = _linprog_solver_payload(parsed)
            status = payload["status"]
            stdout = payload["stdout"]
            stderr = payload["stderr"]
            objective_value = payload["objective_value"]
            variable_values = payload["variable_values"]
        except Exception as exc:
            status = "solver_parse_error"
            stdout = ""
            stderr = str(exc)
            objective_value = None
            variable_values = {}
    else:
        status = "unsupported_model_class"
        stderr = f"Unsupported MPS solver backend: {solver_backend}"

    if persist_code_evidence:
        code_evidence_id = api.insert_code_evidence(
            topic_id,
            None,
            hypothesis=f"Validate MPS artifact O{artifact_id} with {solver_backend}",
            source_code=content,
            stdout=stdout,
            stderr=stderr,
            exit_code=0 if status == "optimal" else 1,
            execution_time_s=None,
            iterations=1,
            success=status == "optimal",
            requesting_role="or_solver",
            summary=f"MPS solver status: {status}",
        )
    solver_run_id = api.insert_solver_run(
        artifact_id=artifact_id,
        problem_id=problem_id,
        topic_id=topic_id,
        solver_backend=solver_backend,
        status=status,
        objective_value=objective_value,
        variable_values_json=json.dumps(variable_values, ensure_ascii=True, sort_keys=True),
        stdout=stdout,
        stderr=stderr,
        error_trace=stderr,
        code_evidence_id=code_evidence_id,
    )
    diagnostic_id = None
    diagnostic_type = classify_solver_failure(status, stdout=stdout, stderr=stderr)
    if diagnostic_type != "none":
        diagnostic_id = api.insert_model_diagnostic(
            problem_id=problem_id,
            topic_id=topic_id,
            artifact_id=artifact_id,
            solver_run_id=solver_run_id,
            diagnostic_type=diagnostic_type,
            severity="error" if status != "solver_unavailable" else "warning",
            message=stderr or stdout or status,
        )
    return {
        "solver_run_id": solver_run_id,
        "code_evidence_id": code_evidence_id,
        "diagnostic_id": diagnostic_id,
        "status": status,
        "objective_value": objective_value,
        "variable_values": variable_values,
    }


def solve_optimization_artifact(
    *,
    topic_id: int,
    problem_id: int,
    artifact_id: int,
    content: str,
    model_language: str,
    solver_backend: str | None = None,
    persist_code_evidence: bool = True,
) -> dict[str, Any]:
    """Dispatch solver execution by artifact model language."""
    language = (model_language or "").strip().lower()
    if language == "lp":
        return solve_lp_artifact(
            topic_id=topic_id,
            problem_id=problem_id,
            artifact_id=artifact_id,
            content=content,
            solver_backend=solver_backend or "scipy_linprog",
            persist_code_evidence=persist_code_evidence,
        )
    if language == "mps":
        return solve_mps_artifact(
            topic_id=topic_id,
            problem_id=problem_id,
            artifact_id=artifact_id,
            content=content,
            solver_backend=solver_backend or "scipy_mps",
            persist_code_evidence=persist_code_evidence,
        )
    raise ValueError(f"Unsupported model_language: {model_language}")


def validate_lp_artifact(content: str, *, strict: bool = True) -> dict[str, Any]:
    """Validate an LP-format artifact with conservative structural checks."""
    text = content or ""
    lines = [_canonical_line(line) for line in text.splitlines()]
    lines = [line for line in lines if line and not line.startswith("\\")]
    seen: dict[str, int] = {}

    for index, line in enumerate(lines):
        for section, aliases in LP_SECTION_ALIASES.items():
            if line in aliases or any(line.startswith(alias + " ") for alias in aliases):
                seen.setdefault(section, index)

    issues: list[dict[str, str]] = []
    if lines and seen and min(seen.values()) != 0:
        issues.append(
            {
                "issue_type": "non_model_preamble",
                "severity": "error",
                "message": "LP artifact must start with a model section, not prose.",
            }
        )
    if "objective" not in seen:
        issues.append(
            {
                "issue_type": "missing_objective_section",
                "severity": "error",
                "message": "LP artifact needs Minimize or Maximize.",
            }
        )
    if "constraints" not in seen:
        issues.append(
            {
                "issue_type": "missing_constraints_section",
                "severity": "error",
                "message": "LP artifact needs Subject To constraints.",
            }
        )
    if "end" not in seen:
        issues.append(
            {
                "issue_type": "missing_end_section",
                "severity": "error",
                "message": "LP artifact must end with End.",
            }
        )
    if "objective" in seen and "constraints" in seen and seen["objective"] > seen["constraints"]:
        issues.append(
            {
                "issue_type": "section_order_error",
                "severity": "error",
                "message": "Objective section should appear before constraints.",
            }
        )
    if "end" in seen and any(line for line in lines[seen["end"] + 1 :]):
        issues.append(
            {
                "issue_type": "non_model_trailer",
                "severity": "error",
                "message": "LP artifact must not contain prose or text after End.",
            }
        )
    if re.search(r"```|here('| i)s|the following", text, flags=re.IGNORECASE):
        issues.append(
            {
                "issue_type": "extraneous_explanation",
                "severity": "error",
                "message": "LP artifact should contain model text only.",
            }
        )

    if strict and not any(i["severity"] == "error" for i in issues):
        try:
            parse_lp_artifact(content)
        except Exception as exc:
            issues.append(
                {
                    "issue_type": "invalid_artifact_syntax",
                    "severity": "error",
                    "message": str(exc),
                }
            )

    status = "valid" if not any(i["severity"] == "error" for i in issues) else "invalid"
    return {
        "status": status,
        "issues": issues,
        "sections": sorted(seen),
    }
