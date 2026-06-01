"""Provider-free OR problem profiling for routing and evaluation metadata."""

from __future__ import annotations

import re
from typing import Any


_OBJECTIVE_KEYWORD_RE = re.compile(
    r"\b("
    r"maximi[sz]e|maximi[sz]ation|maximum|max|minimi[sz]e|minimi[sz]ation|"
    r"minimum|min|optimal|optimi[sz]e|objective|profit|cost"
    r")\b",
    flags=re.IGNORECASE,
)
_OPTIMIZATION_KEYWORD_RE = re.compile(
    r"\b("
    r"maximi[sz]e|maximi[sz]ation|maximum|max|minimi[sz]e|minimi[sz]ation|"
    r"minimum|min|optimal|optimi[sz]e|objective|constraint|profit|cost|"
    r"capacity|demand|resource|schedule|allocation|allocate|inventory|"
    r"shipment|transport|facility|assignment"
    r")\b",
    flags=re.IGNORECASE,
)
_DIRECT_CALCULATION_RE = re.compile(
    r"\b("
    r"calculate|compute|determine|what is|how many|how much|find the|"
    r"available in total|left over|remaining|by the end|after these"
    r")\b",
    flags=re.IGNORECASE,
)
_INDUSTRIAL_STRUCTURE_RE = re.compile(
    r"\b("
    r"week|month|year|period|shift|plant|factory|warehouse|supplier|"
    r"customer|machine|line|vehicle|route|project|scenario|table|"
    r"integer|binary|nonlinear|mixed[- ]integer|mip|lp|nlp"
    r")s?\b",
    flags=re.IGNORECASE,
)
_GOAL_PROGRAMMING_RE = re.compile(
    r"\b(goal\s+(?:programming|planning)|priorit(?:y|ies)|p\s*_?\s*\{?\s*[0-9]+\s*\}?\s*:)",
    flags=re.IGNORECASE,
)
_FLOW_SHOP_RE = re.compile(
    r"\b(flow[- ]shop|machine\s+[0-9]|machines?|vats?|jobs?|batches?|products?)\b"
    r".{0,240}\b(sequence|sequential|same\s+order|completion\s+time|makespan|processing\s+cycle)\b",
    flags=re.IGNORECASE | re.DOTALL,
)
_WORKFORCE_SCHEDULING_RE = re.compile(
    r"\b(nurse|nurses|worker|workers|employee|employees|staff|clerk|clerks|"
    r"waiter|waiters|student|students|salesperson|salespeople|driver|drivers|crew|members?)\b"
    r".{0,260}\b(schedule|scheduled|shift|duty|consecutive\s+days|days\s+in\s+a\s+row|demand|gross\s+pay)\b",
    flags=re.IGNORECASE | re.DOTALL,
)
_SHIFT_COVERAGE_RE = re.compile(
    r"(?=.*\b(?:24[-\s]?hour|24\s+hours|over\s+24\s+hours|around\s+the\s+clock)\b)"
    r"(?=.*\b(?:shift|shifts|work(?:s|ing)?\s+continuously)\b)"
    r"(?=.*\b(?:nurses?|workers?|employees?|staff|clerks?|waiters?|students?|"
    r"salespeople|salespersons?|drivers?|crew|members?)\b)",
    flags=re.IGNORECASE | re.DOTALL,
)
_MAX_FLOW_RE = re.compile(
    r"\b(maximum\s+flow|max\s+flow|flow\s+from|source\s+to\s+sink)\b",
    flags=re.IGNORECASE,
)
_TSP_RE = re.compile(
    r"\b(travel(?:ing|ling)\s+salesman|visit(?:ing)?\s+order|starting\s+and\s+ending|return(?:ing)?\s+to\s+the\s+start)\b",
    flags=re.IGNORECASE,
)
_FACILITY_LOCATION_RE = re.compile(
    r"\b(distribution\s+centers?|facility\s+location|opening\s+costs?|open\s+(?:the\s+)?(?:centers?|facilities?))\b",
    flags=re.IGNORECASE,
)
_FIXED_CHARGE_TRANSSHIPMENT_RE = re.compile(
    r"\b(?:intermediate\s+(?:marshaling\s+)?stations?|transshipment|transshipment\s+capacity)\b"
    r".{0,260}\b(?:fixed\s+cost|capacity|production\s+points?|demand\s+points?)\b",
    flags=re.IGNORECASE | re.DOTALL,
)
_TRANSPORTATION_RE = re.compile(
    r"\b(transportation|transport|shipment|ship|warehouse|supply)\b.{0,180}\b(demand|destination|sales\s+points?|stores?)\b",
    flags=re.IGNORECASE | re.DOTALL,
)
_NARRATIVE_TRANSPORTATION_RE = re.compile(
    r"\b(?:coal\s+yards?|warehouses?|plants?|factories?|supply\s+points?)\b"
    r".{0,320}\b(?:residential\s+areas?|customers?|demand\s+points?|destinations?)\b"
    r".{0,320}\b(?:kilometers?|ton[-\s]?kilometers?|transportation)\b",
    flags=re.IGNORECASE | re.DOTALL,
)
_INTERVAL_CONTRACT_RE = re.compile(
    r"\b(?:rent|rental|lease|contract)\b.{0,260}\b(?:months?|periods?|consecutive|lengths?)\b"
    r".{0,260}\b(?:required|demand|cover|area|capacity)\b",
    flags=re.IGNORECASE | re.DOTALL,
)
_INTEGER_RESOURCE_MIX_RE = re.compile(
    r"\b(?:transportation\s+(?:options?|methods?)|types?\s+of\s+trucks?|"
    r"buses?|minibuses?|trips?)\b"
    r".{0,300}\b(?:capacity|seats?|pollution|rental\s+cost|drivers?|transport)\b",
    flags=re.IGNORECASE | re.DOTALL,
)
_TABLE_CAPACITY_SPACE_MIX_RE = re.compile(
    r"\btables?\b.{0,320}\b(?:participants?|poster\s+boards?|guests?)\b"
    r".{0,320}\b(?:space|maximize|cater)\b",
    flags=re.IGNORECASE | re.DOTALL,
)
_FIXED_CHARGE_MACHINE_ASSIGNMENT_RE = re.compile(
    r"(?=.*\bmachines?\b)(?=.*\bparts?\b)(?=.*\bsetup\s+cost\b)",
    flags=re.IGNORECASE | re.DOTALL,
)
_PROCUREMENT_LOT_MIX_RE = re.compile(
    r"\b(?:suppliers?|manufacturers?|warehouses?)\b"
    r".{0,260}\b(?:orders?|trucks?|raw\s+materials?|cost|freight)\b"
    r".{0,260}\b(?:minimi[sz]e|minimum|at\s+least|required)\b",
    flags=re.IGNORECASE | re.DOTALL,
)
_INVENTORY_PRODUCTION_RE = re.compile(
    r"\b(inventory|holding\s+cost|backorder|quarters?|periods?|production\s+schedule|regular[- ]time|overtime\s+labor)\b",
    flags=re.IGNORECASE,
)
_MULTI_PERIOD_WORKFORCE_PRODUCTION_RE = re.compile(
    r"(?=.*\bworkforce\b)(?=.*\boutsourcing\b)(?=.*\bbackorders?\b)"
    r"(?=.*\binventory\b)(?=.*\b(?:hire|hiring|fire|firing)\b)"
    r"(?=.*\bovertime\b)",
    flags=re.IGNORECASE | re.DOTALL,
)
_OVERTIME_PRODUCT_MIX_RE = re.compile(
    r"\b(overtime\s+(?:assembly\s+)?labor|overtime\s+pay)\b"
    r".{0,260}\b(products?|raw\s+materials?|market\s+value|profit)\b",
    flags=re.IGNORECASE | re.DOTALL,
)
_RESOURCE_PRODUCT_MIX_RE = re.compile(
    r"\b(products?|produce|packages?|batches?)\b.{0,260}\b("
    r"raw\s+materials?|labor|assembly|testing|steel|aluminum|profit|"
    r"revenue|warehouse\s+space|space|shirts?|pants?"
    r")\b",
    flags=re.IGNORECASE,
)
_QUALITY_BLENDING_RE = re.compile(
    r"\b(?:mix(?:es|ed|ing)?|blend(?:s|ed|ing)?)\b"
    r".{0,320}\b(?:raw\s+materials?|inputs?|ingredients?)\b"
    r".{0,320}\b(?:sulfur|sulphur|impurit(?:y|ies)|quality|content)\b",
    flags=re.IGNORECASE | re.DOTALL,
)
_BLENDING_RE = re.compile(
    r"\b(blending|blend|octane|raw\s+gasoline|mixture)\b",
    flags=re.IGNORECASE,
)
_NUTRITION_MIX_RE = re.compile(
    r"\b(?:feed|foods?|diet|meal|nutrition|nutritional|protein|vitamins?|minerals?)\b"
    r".{0,260}\b(?:minimi[sz]e|minimum|least|cost|price|at\s+least)\b",
    flags=re.IGNORECASE | re.DOTALL,
)
_CUTTING_STOCK_RE = re.compile(
    r"\b(?:cut|cutting|cut)\b.{0,220}\b(?:rolls?|bars?|pipes?|raw\s+material|waste|patterns?)\b",
    flags=re.IGNORECASE | re.DOTALL,
)
_CAPITAL_BUDGETING_RE = re.compile(
    r"\b(investment\s+opportunities?|capital\s+budget|cash\s+outflows?|npv|net\s+present\s+value|principal\s+plus\s+interest)\b",
    flags=re.IGNORECASE,
)
_ASSIGNMENT_RE = re.compile(
    r"\b(assignment|assign|assigned)\b.{0,160}\b(worker|task|job|specialty|city)\b",
    flags=re.IGNORECASE | re.DOTALL,
)
_BINARY_SUBSET_SELECTION_RE = re.compile(
    r"\b(?:hire|select|choose|take|decide\s+whether)\b"
    r".{0,260}\b(?:at\s+least|at\s+most|up\s+to|maximum|budget|must|cannot|can\s+not)\b"
    r".{0,260}\b(?:candidate|candidates|children|employees?)\b",
    flags=re.IGNORECASE | re.DOTALL,
)
_COVERING_RE = re.compile(
    r"\b(set\s+cover|covering|vertex\s+cover|minimum\s+number)\b",
    flags=re.IGNORECASE,
)
_BLENDED_SECURITY_RE = re.compile(
    r"\b(securit(?:y|ies)|scenario|payoff|worst[- ]case|maximin)\b",
    flags=re.IGNORECASE,
)
_ROBUST_OPTIMIZATION_RE = re.compile(
    r"\b(?:robust\s+optimization|robust\s+counterpart|uncertain\s+(?:coefficients?|parameters?|demand|capacity)|"
    r"box\s+uncertainty|budget\s+uncertainty|polyhedral\s+uncertainty|uncertainty\s+set)\b",
    flags=re.IGNORECASE,
)


def _problem_family_hint(problem_text: str) -> str:
    if _GOAL_PROGRAMMING_RE.search(problem_text):
        return "goal_programming"
    if _MAX_FLOW_RE.search(problem_text):
        return "network_flow"
    if _TSP_RE.search(problem_text):
        return "routing_tsp"
    if _FLOW_SHOP_RE.search(problem_text):
        return "flow_shop_scheduling"
    if _SHIFT_COVERAGE_RE.search(problem_text):
        return "workforce_scheduling"
    if _MULTI_PERIOD_WORKFORCE_PRODUCTION_RE.search(problem_text):
        return "multi_period_workforce_production"
    if _WORKFORCE_SCHEDULING_RE.search(problem_text):
        return "workforce_scheduling"
    if _CAPITAL_BUDGETING_RE.search(problem_text):
        return "capital_budgeting"
    if _FACILITY_LOCATION_RE.search(problem_text):
        return "facility_location"
    if _FIXED_CHARGE_TRANSSHIPMENT_RE.search(problem_text):
        return "fixed_charge_transshipment"
    if _INTERVAL_CONTRACT_RE.search(problem_text):
        return "interval_contract_covering"
    if _INTEGER_RESOURCE_MIX_RE.search(problem_text):
        return "integer_resource_mix"
    if _TABLE_CAPACITY_SPACE_MIX_RE.search(problem_text):
        return "table_capacity_space_mix"
    if _FIXED_CHARGE_MACHINE_ASSIGNMENT_RE.search(problem_text):
        return "fixed_charge_machine_assignment"
    if _PROCUREMENT_LOT_MIX_RE.search(problem_text):
        return "procurement_lot_mix"
    if _NARRATIVE_TRANSPORTATION_RE.search(problem_text):
        return "transportation"
    if _TRANSPORTATION_RE.search(problem_text):
        return "transportation"
    if _OVERTIME_PRODUCT_MIX_RE.search(problem_text):
        return "overtime_product_mix"
    if _INVENTORY_PRODUCTION_RE.search(problem_text):
        return "production_inventory"
    if _ROBUST_OPTIMIZATION_RE.search(problem_text):
        return "robust_optimization"
    if _QUALITY_BLENDING_RE.search(problem_text):
        return "blending"
    if _RESOURCE_PRODUCT_MIX_RE.search(problem_text):
        return "product_mix"
    if _BLENDING_RE.search(problem_text):
        return "blending"
    if _NUTRITION_MIX_RE.search(problem_text):
        return "nutrition_mix"
    if _CUTTING_STOCK_RE.search(problem_text):
        return "cutting_stock"
    if _ASSIGNMENT_RE.search(problem_text):
        return "assignment"
    if _BINARY_SUBSET_SELECTION_RE.search(problem_text):
        return "binary_subset_selection"
    if _COVERING_RE.search(problem_text):
        return "covering"
    if _BLENDED_SECURITY_RE.search(problem_text):
        return "robust_security"
    return "unknown"


def _problem_text_features(problem_text: str) -> dict[str, Any]:
    text = str(problem_text or "")
    return {
        "question_length": len(text),
        "has_table": "|" in text and re.search(r"^\s*\|.+\|\s*$", text, re.MULTILINE)
        is not None,
        "has_objective_language": _OBJECTIVE_KEYWORD_RE.search(text) is not None,
        "has_optimization_language": _OPTIMIZATION_KEYWORD_RE.search(text) is not None,
        "has_direct_calculation_language": _DIRECT_CALCULATION_RE.search(text)
        is not None,
        "has_industrial_structure": _INDUSTRIAL_STRUCTURE_RE.search(text) is not None,
        "problem_family_hint": _problem_family_hint(text),
    }


def profile_or_problem_text(
    *,
    problem_text: str,
    gold_solver: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Profile an OR problem statement for routing and evaluation metadata."""
    features = _problem_text_features(problem_text)
    gold_solver = gold_solver or {}
    answer_kind = (
        "no_finite_optimum"
        if gold_solver.get("status") == "no_solution_reported"
        else "numeric_objective"
        if gold_solver.get("objective_value") is not None
        else "unknown"
    )

    known_family = features["problem_family_hint"] != "unknown"
    if (
        features["has_direct_calculation_language"]
        and not features["has_objective_language"]
        and not known_family
    ):
        workflow_hint = "direct_calculation"
    elif features["has_table"] or features["has_industrial_structure"]:
        workflow_hint = "industrial_modeling_reviewed"
    elif features["has_optimization_language"]:
        workflow_hint = "solver_modeling"
    else:
        workflow_hint = "quantitative_reasoning"

    if features["has_table"] or features["question_length"] >= 1200:
        coverage = "structured_industrial_problem"
        industrial_realism = "high"
    elif features["has_industrial_structure"]:
        coverage = "industrial_decision_problem"
        industrial_realism = "medium"
    elif features["has_optimization_language"]:
        coverage = "optimization_word_problem"
        industrial_realism = "low"
    else:
        coverage = "quantitative_word_problem"
        industrial_realism = "low"

    return {
        **features,
        "workflow_hint": workflow_hint,
        "orq_coverage": coverage,
        "industrial_realism": industrial_realism,
        "answer_kind": answer_kind,
    }
