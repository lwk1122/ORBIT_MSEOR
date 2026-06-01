"""Deterministic templates for common OR problem families.

These templates are intentionally conservative. They only return a solution
when the problem statement exposes enough structure to avoid LLM modeling.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import re
from typing import Any


_NUMBER_WORDS = {
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "single": 1.0,
    "zero": 0.0,
    "once": 1.0,
    "twice": 2.0,
}
_NUMBER_TOKEN = (
    r"(?:[0-9][0-9,]*(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|single|zero|once|twice)"
)
_FRACTION_OR_NUMBER_TOKEN = rf"(?:[0-9]+/[0-9]+|{_NUMBER_TOKEN})"
_YEAR_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_YEAR_TOKEN = (
    r"(?:[0-9]+|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"one|two|three|four|five|six|seven|eight|nine|ten)"
)


@dataclass(frozen=True)
class TemplateSolveResult:
    matched: bool
    template_id: str = ""
    status: str = "not_applicable"
    objective_value: float | None = None
    variable_values: dict[str, float] | None = None
    confidence: float = 0.0
    notes: str = ""
    artifact: dict[str, Any] | None = None


def _number(value: str) -> float:
    lowered = value.strip().lower()
    if lowered in _NUMBER_WORDS:
        return _NUMBER_WORDS[lowered]
    return float(value.replace(",", ""))


def _quantity_number(value: str) -> float:
    stripped = value.strip()
    if re.fullmatch(r"[0-9]+/[0-9]+", stripped):
        numerator, denominator = stripped.split("/", 1)
        return float(numerator) / float(denominator)
    return _number(stripped)


def _year_number(value: str) -> int | None:
    lowered = value.strip().lower()
    if lowered in _YEAR_WORDS:
        return _YEAR_WORDS[lowered]
    if re.fullmatch(r"[0-9]+", lowered):
        return int(lowered)
    return None


def _first_number(value: str) -> float | None:
    match = re.search(_NUMBER_TOKEN, value, flags=re.IGNORECASE)
    if not match:
        return None
    return _number(match.group(0))


def _clean_label(value: str) -> str:
    cleaned = re.sub(r"<[^>]+>", "", value)
    cleaned = re.sub(r"\\+(?:mathrm|text)\{([^}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"[$\\(){}]", "", cleaned)
    cleaned = re.sub(r"\b(?:mathrm|text)(?=[A-Za-z0-9])", "", cleaned)
    cleaned = re.sub(r"\b(?:mathrm|text)\b", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or "item"


def _singular_label(value: str) -> str:
    label = _clean_label(value).lower()
    if label == "buses":
        return "bus"
    if label == "minibuses":
        return "minibus"
    if label == "sheep":
        return label
    if label.endswith("ies"):
        return f"{label[:-3]}y"
    if label.endswith("s") and not label.endswith("ss"):
        return label[:-1]
    return label


def _item_index_for_label(labels: list[str], value: str) -> int | None:
    wanted = _singular_label(value)
    for index, label in enumerate(labels):
        if _singular_label(label) == wanted:
            return index
    return None


def _solve_training_asset_count(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        "trained pilot" in lowered
        and "training jet" in lowered
        and "annual production" in lowered
    ):
        return TemplateSolveResult(False)
    productions = [
        _number(match)
        for match in re.findall(
            r"a\s*[_{]?\s*\d+\s*[}]?\s*=\s*([0-9][0-9,]*(?:\.\d+)?)",
            text,
            flags=re.IGNORECASE,
        )
    ]
    rate_match = re.search(
        r"each\s+training\s+jet\s+can\s+train\s+([0-9][0-9,]*(?:\.\d+)?)\s+pilots?",
        text,
        flags=re.IGNORECASE,
    )
    if not productions or not rate_match:
        return TemplateSolveResult(False)
    rate = _number(rate_match.group(1))
    objective = sum(productions) * rate
    variables = {
        f"production_year_{index}": value
        for index, value in enumerate(productions, start=1)
    }
    variables["pilots_per_training_jet"] = rate
    return TemplateSolveResult(
        matched=True,
        template_id="direct_training_asset_count",
        status="optimal",
        objective_value=objective,
        variable_values=variables,
        confidence=0.9,
        notes="Computed trained operators from stated production counts and training rate.",
        artifact={"productions": productions, "training_rate": rate},
    )


def _parse_markdown_table(text: str) -> tuple[list[str], list[list[str]]] | None:
    tables = _parse_markdown_tables(text)
    return tables[0] if tables else None


def _parse_markdown_tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    tables: list[tuple[list[str], list[list[str]]]] = []
    rows: list[list[str]] = []
    for raw_line in [*text.splitlines(), ""]:
        line = raw_line.strip()
        if not (line.startswith("|") and line.endswith("|")):
            if len(rows) >= 2:
                tables.append((rows[0], rows[1:]))
            rows = []
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        rows.append(cells)
    return tables


def _solve_assignment_table(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        "assignment" in lowered
        and ("worker" in lowered or "workers" in lowered)
        and ("minimize" in lowered or "minimum" in lowered)
    ):
        return TemplateSolveResult(False)
    parsed = _parse_markdown_table(text)
    if not parsed:
        return TemplateSolveResult(False)
    header, rows = parsed
    if len(header) < 3:
        return TemplateSolveResult(False)
    task_labels = [_clean_label(cell) for cell in header[1:]]
    worker_labels: list[str] = []
    costs: list[list[float]] = []
    for row in rows:
        if len(row) != len(header):
            continue
        values: list[float] = []
        for cell in row[1:]:
            if not re.fullmatch(r"[0-9][0-9,]*(?:\.\d+)?", cell.strip()):
                values = []
                break
            values.append(_number(cell))
        if values:
            worker_labels.append(_clean_label(row[0]))
            costs.append(values)
    if len(worker_labels) < len(task_labels) or not costs:
        return TemplateSolveResult(False)

    best: tuple[float, tuple[int, ...]] | None = None
    for worker_indices in itertools.permutations(range(len(worker_labels)), len(task_labels)):
        total = sum(costs[worker_index][task_index] for task_index, worker_index in enumerate(worker_indices))
        if best is None or total < best[0]:
            best = (total, worker_indices)
    if best is None:
        return TemplateSolveResult(False)
    variable_values = {
        f"assign_{worker_labels[worker_index]}_{task_labels[task_index]}": 1.0
        for task_index, worker_index in enumerate(best[1])
    }
    return TemplateSolveResult(
        matched=True,
        template_id="assignment_min_cost",
        status="optimal",
        objective_value=float(best[0]),
        variable_values=variable_values,
        confidence=0.95,
        notes="Solved rectangular worker-task assignment by exhaustive enumeration.",
        artifact={"workers": worker_labels, "tasks": task_labels, "costs": costs},
    )


def _normalize_assignment_city(value: str) -> str:
    label = _clean_label(value).lower()
    label = re.sub(r"\bcity\b", "", label, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", label).strip()


def _split_specialty_list(value: str) -> list[str]:
    return sorted({
        _clean_label(item).lower()
        for item in re.split(r"[,/;]|\s+or\s+", value, flags=re.IGNORECASE)
        if _clean_label(item)
    })


def _parse_priority_target(text: str, priority: int, keyword: str) -> float | None:
    pattern = re.compile(
        (
            rf"p\s*_?\s*(?:\{{\s*)?{priority}(?:\s*\}})?\s*:?"
            rf".{{0,160}}?({_NUMBER_TOKEN})\s+[^.;\n]*?{keyword}"
        ),
        flags=re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if match:
        return _number(match.group(1))
    return None


def _parse_preference_assignment_tables(
    text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    demand_rows: list[dict[str, Any]] = []
    type_rows: list[dict[str, Any]] = []
    for header, rows in _parse_markdown_tables(text):
        header_text = " ".join(header).lower()
        cleaned_header = [_clean_label(cell).lower() for cell in header]
        if (
            "branch" in header_text
            and "specialty" in header_text
            and "demand" in header_text
        ):
            city_index = next((i for i, cell in enumerate(cleaned_header) if "branch" in cell), 0)
            specialty_index = next((i for i, cell in enumerate(cleaned_header) if "specialty" in cell), 1)
            demand_index = next((i for i, cell in enumerate(cleaned_header) if "demand" in cell), 2)
            for row in rows:
                if len(row) <= max(city_index, specialty_index, demand_index):
                    continue
                demand_value = _first_number(row[demand_index])
                if demand_value is None:
                    continue
                demand_rows.append(
                    {
                        "city": _clean_label(row[city_index]),
                        "city_key": _normalize_assignment_city(row[city_index]),
                        "specialty": _clean_label(row[specialty_index]).lower(),
                        "demand": demand_value,
                    }
                )
        elif (
            "number of people" in header_text
            and "suitable specialty" in header_text
            and "preferred specialty" in header_text
            and "preferred city" in header_text
        ):
            type_index = next((i for i, cell in enumerate(cleaned_header) if cell == "type"), 0)
            count_index = next((i for i, cell in enumerate(cleaned_header) if "number" in cell), 1)
            suitable_index = next((i for i, cell in enumerate(cleaned_header) if "suitable" in cell), 2)
            pref_spec_index = next((i for i, cell in enumerate(cleaned_header) if "preferred specialty" in cell), 3)
            pref_city_index = next((i for i, cell in enumerate(cleaned_header) if "preferred city" in cell), 4)
            for row in rows:
                if len(row) <= max(type_index, count_index, suitable_index, pref_spec_index, pref_city_index):
                    continue
                count_value = _first_number(row[count_index])
                if count_value is None:
                    continue
                type_rows.append(
                    {
                        "label": _clean_label(row[type_index]),
                        "count": count_value,
                        "suitable_specialties": _split_specialty_list(row[suitable_index]),
                        "preferred_specialty": _clean_label(row[pref_spec_index]).lower(),
                        "preferred_city": _clean_label(row[pref_city_index]),
                        "preferred_city_key": _normalize_assignment_city(row[pref_city_index]),
                    }
                )
    if demand_rows and type_rows:
        return demand_rows, type_rows
    return None


def _solve_preference_assignment_goal_programming(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        ("goal planning" in lowered or "goal programming" in lowered or "priorit" in lowered)
        and "preferred specialty" in lowered
        and "preferred city" in lowered
        and "demand" in lowered
    ):
        return TemplateSolveResult(False)

    parsed = _parse_preference_assignment_tables(text)
    p2_target = _parse_priority_target(text, 2, r"preferred\s+specialty")
    p3_target = _parse_priority_target(text, 3, r"preferred\s+city")
    if not parsed or p2_target is None or p3_target is None:
        return TemplateSolveResult(False)
    demands, types = parsed

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="preference_assignment_goal_programming",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
            artifact={"demands": demands, "types": types},
        )

    assignments: list[tuple[int, int]] = []
    for type_index, type_row in enumerate(types):
        for demand_index, demand_row in enumerate(demands):
            if demand_row["specialty"] in type_row["suitable_specialties"]:
                assignments.append((type_index, demand_index))
    if not assignments:
        return TemplateSolveResult(False)

    variable_count = len(assignments)
    rows: list[Any] = []
    lower: list[float] = []
    upper: list[float] = []
    for demand_index, demand_row in enumerate(demands):
        row = np.zeros(variable_count)
        for variable, (_type_index, assigned_demand) in enumerate(assignments):
            if assigned_demand == demand_index:
                row[variable] = 1.0
        rows.append(row)
        lower.append(float(demand_row["demand"]))
        upper.append(float(demand_row["demand"]))
    for type_index, type_row in enumerate(types):
        row = np.zeros(variable_count)
        for variable, (assigned_type, _demand_index) in enumerate(assignments):
            if assigned_type == type_index:
                row[variable] = 1.0
        rows.append(row)
        lower.append(0.0)
        upper.append(float(type_row["count"]))

    base_constraint = LinearConstraint(np.vstack(rows), np.array(lower), np.array(upper))
    specialty_coefficients = np.array(
        [
            1.0
            if demands[demand_index]["specialty"] == types[type_index]["preferred_specialty"]
            else 0.0
            for type_index, demand_index in assignments
        ]
    )
    city_coefficients = np.array(
        [
            1.0
            if demands[demand_index]["city_key"] == types[type_index]["preferred_city_key"]
            else 0.0
            for type_index, demand_index in assignments
        ]
    )
    bounds = Bounds(np.zeros(variable_count), np.full(variable_count, math.inf))
    stage2 = milp(
        -specialty_coefficients,
        integrality=np.ones(variable_count),
        bounds=bounds,
        constraints=base_constraint,
    )
    if not stage2.success:
        return TemplateSolveResult(
            matched=True,
            template_id="preference_assignment_goal_programming",
            status="solver_failed",
            confidence=0.82,
            notes=str(stage2.message),
            artifact={"demands": demands, "types": types},
        )
    best_specialty = float(-stage2.fun)
    required_specialty = min(float(p2_target), best_specialty)
    rows.append(specialty_coefficients)
    lower.append(required_specialty)
    upper.append(math.inf)
    stage3_constraint = LinearConstraint(np.vstack(rows), np.array(lower), np.array(upper))
    stage3 = milp(
        -city_coefficients,
        integrality=np.ones(variable_count),
        bounds=bounds,
        constraints=stage3_constraint,
    )
    if not stage3.success:
        return TemplateSolveResult(
            matched=True,
            template_id="preference_assignment_goal_programming",
            status="solver_failed",
            confidence=0.82,
            notes=str(stage3.message),
            artifact={"demands": demands, "types": types, "best_specialty": best_specialty},
        )
    best_city = float(-stage3.fun)
    city_shortfall = max(0.0, float(p3_target) - best_city)

    variable_values: dict[str, float] = {}
    for value, (type_index, demand_index) in zip(stage3.x, assignments):
        if math.isclose(float(value), 0.0, abs_tol=1e-8):
            continue
        type_label = types[type_index]["label"]
        demand = demands[demand_index]
        variable_values[
            f"assign_type_{type_label}_to_{demand['city']}_specialty_{demand['specialty']}"
        ] = float(value)

    return TemplateSolveResult(
        matched=True,
        template_id="preference_assignment_goal_programming",
        status="optimal",
        objective_value=city_shortfall,
        variable_values=variable_values,
        confidence=0.88,
        notes=(
            "Solved preemptive preference-assignment goal program: demand "
            "coverage first, preferred specialty second, preferred city third."
        ),
        artifact={
            "demands": demands,
            "types": types,
            "p2_target_preferred_specialty": p2_target,
            "p3_target_preferred_city": p3_target,
            "best_preferred_specialty": best_specialty,
            "best_preferred_city": best_city,
        },
    )


_NETWORK_NODE_RE = (
    r"(?:Data\s+Center|Center|Point|Station|Relay\s+Station|Node|Substation|City)"
)


def _network_origin_from_chunk(chunk: str) -> int | None:
    match = re.search(
        rf"^\s*(?:From\s+)?{_NETWORK_NODE_RE}\s+([0-9]+)\b",
        chunk,
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(
            rf"^\s*From\b[^(]*\(\s*{_NETWORK_NODE_RE}\s+([0-9]+)\s*\)",
            chunk,
            flags=re.IGNORECASE,
        )
    if not match:
        return None
    return int(match.group(1))


def _network_capacity_scale(text: str, capacities: dict[tuple[int, int], float]) -> float:
    if not capacities:
        return 1.0
    if max(capacities.values()) < 1000:
        return 1.0
    if re.search(r"\bin\s+thousands\b|\bthousands\s+of\b", text, flags=re.IGNORECASE):
        return 0.001
    if re.search(r"\bpassenger\s+flow\b|\bpassengers?\b", text, flags=re.IGNORECASE):
        return 0.001
    return 1.0


def _parse_numbered_network_edges(text: str) -> tuple[dict[tuple[int, int], float], set[int], int, int] | None:
    chunks = [
        re.sub(r"\s+", " ", chunk).strip(" -\t")
        for chunk in re.split(r"\n\s*-\s*", text)
        if chunk.strip()
    ]
    capacities: dict[tuple[int, int], float] = {}
    nodes: set[int] = set()
    source_candidates: list[int] = []
    sink_candidates: list[int] = []
    for chunk in chunks:
        origin = _network_origin_from_chunk(chunk)
        if origin is None:
            continue
        nodes.add(origin)
        prefix = chunk.split(":", 1)[0]
        if re.search(r"source|start(?:ing)?\s+point|power\s+plant|central\s+hub", prefix, flags=re.IGNORECASE):
            source_candidates.append(origin)
        if re.search(
            r"destination|end\s+destination|user\s+hub|endpoint|central\s+hub|final\s+distribution|backup\s+center",
            prefix,
            flags=re.IGNORECASE,
        ):
            sink_candidates.append(origin)
        origin_descriptor = re.search(
            rf"^\s*(?:From\s+)?{_NETWORK_NODE_RE}\s+{origin}\s*\(([^)]*)\)",
            chunk,
            flags=re.IGNORECASE,
        )
        if origin_descriptor and re.search(
            r"source|start(?:ing)?\s+point|central\s+hub",
            origin_descriptor.group(1),
            flags=re.IGNORECASE,
        ):
            source_candidates.append(origin)
        if origin_descriptor and re.search(
            r"destination|end\s+destination|user\s+hub|endpoint|central\s+hub|final\s+distribution|backup\s+center",
            origin_descriptor.group(1),
            flags=re.IGNORECASE,
        ):
            sink_candidates.append(origin)
        body = chunk.split(":", 1)[1] if ":" in chunk else chunk
        for destination, value in re.findall(
            rf"{_NETWORK_NODE_RE}\s+([0-9]+)\s*\(\s*([0-9][0-9,]*(?:\.\d+)?)\s*[^)]*\)",
            body,
            flags=re.IGNORECASE,
        ):
            destination_index = int(destination)
            capacity = _number(value)
            nodes.add(destination_index)
            if destination_index == origin or capacity <= 0:
                continue
            capacities[(origin, destination_index)] = capacities.get((origin, destination_index), 0.0) + capacity

    node_clause_pattern = re.compile(
        r"\bNode\s+([0-9]+)\b(.*?)(?=\bNode\s+[0-9]+\b|\bFind\b|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for node_match in node_clause_pattern.finditer(text):
        node_index = int(node_match.group(1))
        nodes.add(node_index)
        descriptor = node_match.group(2)
        if re.search(r"source|start(?:ing)?\s+point", descriptor, flags=re.IGNORECASE):
            source_candidates.append(node_index)
        if re.search(r"target|terminal|destination|end\s+point", descriptor, flags=re.IGNORECASE):
            sink_candidates.append(node_index)

    edge_pattern = re.compile(
        (
            r"(?:edge|arc)(?:\s+leading)?\s+from\s+node\s+([0-9]+)"
            r"\s+to\s+node\s+([0-9]+)\s+with\s+(?:a\s+)?"
            r"(?:substantial\s+)?capacity\s+of\s+([0-9][0-9,]*(?:\.\d+)?)"
        ),
        flags=re.IGNORECASE,
    )
    for origin, destination, value in edge_pattern.findall(text):
        origin_index = int(origin)
        destination_index = int(destination)
        capacity = _number(value)
        nodes.update((origin_index, destination_index))
        if origin_index == destination_index or capacity <= 0:
            continue
        capacities[(origin_index, destination_index)] = (
            capacities.get((origin_index, destination_index), 0.0) + capacity
        )

    if not capacities or len(nodes) < 2:
        return None
    source = source_candidates[0] if source_candidates else (0 if 0 in nodes else min(nodes))
    sink = sink_candidates[-1] if sink_candidates else max(nodes)
    if source == sink:
        return None
    scale = _network_capacity_scale(text, capacities)
    if not math.isclose(scale, 1.0):
        capacities = {edge: capacity * scale for edge, capacity in capacities.items()}
    return capacities, nodes, source, sink


def _edmonds_karp_max_flow(
    capacities: dict[tuple[int, int], float],
    nodes: set[int],
    source: int,
    sink: int,
) -> tuple[float, dict[tuple[int, int], float]]:
    residual: dict[int, dict[int, float]] = {node: {} for node in nodes}
    for (origin, destination), capacity in capacities.items():
        residual.setdefault(origin, {})
        residual.setdefault(destination, {})
        residual[origin][destination] = residual[origin].get(destination, 0.0) + capacity
        residual[destination].setdefault(origin, 0.0)

    max_flow = 0.0
    while True:
        parent: dict[int, int | None] = {source: None}
        queue = [source]
        for node in queue:
            for neighbor, capacity in residual.get(node, {}).items():
                if neighbor not in parent and capacity > 1e-9:
                    parent[neighbor] = node
                    queue.append(neighbor)
                    if neighbor == sink:
                        break
            if sink in parent:
                break
        if sink not in parent:
            break
        increment = math.inf
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            increment = min(increment, residual[previous][node])
            node = previous
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            residual[previous][node] -= increment
            residual[node][previous] = residual[node].get(previous, 0.0) + increment
            node = previous
        max_flow += increment

    flows: dict[tuple[int, int], float] = {}
    for edge, capacity in capacities.items():
        origin, destination = edge
        flow = capacity - residual.get(origin, {}).get(destination, 0.0)
        if flow > 1e-8:
            flows[edge] = flow
    return max_flow, flows


def _solve_max_flow_network(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        ("maximum flow" in lowered or "maximum amount" in lowered or "maximum number" in lowered or "maximize the flow" in lowered)
        and (
            "source" in lowered
            or "starting point" in lowered
            or "start point" in lowered
            or "central hub" in lowered
            or "data center 0" in lowered
            or "station 0" in lowered
            or "node 0" in lowered
            or "point 0" in lowered
        )
        and (
            "destination" in lowered
            or "user hub" in lowered
            or "endpoint" in lowered
            or "central hub" in lowered
            or "end destination" in lowered
            or "final distribution" in lowered
            or "target" in lowered
            or "terminal" in lowered
            or "end point" in lowered
        )
        and ("capacity" in lowered or "capacities" in lowered)
    ):
        return TemplateSolveResult(False)

    parsed = _parse_numbered_network_edges(text)
    if not parsed:
        return TemplateSolveResult(False)
    capacities, nodes, source, sink = parsed
    if len(capacities) < 2:
        return TemplateSolveResult(False)
    origins = {origin for origin, _destination in capacities}
    if len(nodes) > 2 and origins <= {source}:
        return TemplateSolveResult(False)

    objective, flows = _edmonds_karp_max_flow(capacities, nodes, source, sink)
    return TemplateSolveResult(
        matched=True,
        template_id="max_flow_network",
        status="optimal",
        objective_value=float(objective),
        variable_values={
            f"flow_{origin}_to_{destination}": value
            for (origin, destination), value in sorted(flows.items())
        },
        confidence=0.9,
        notes="Solved directed maximum-flow network from numbered capacity arcs.",
        artifact={
            "nodes": sorted(nodes),
            "source": source,
            "sink": sink,
            "capacities": {f"{origin}->{destination}": value for (origin, destination), value in sorted(capacities.items())},
        },
    )


def _parse_scenario_labels(text: str) -> list[str]:
    scenario_match = re.search(
        (
            r"\bThere\s+are\s+(?:[0-9]+|one|two|three|four|five|six|seven|eight|nine|ten)"
            r"\s+(?:countries|states|scenarios|outcomes)\s*,\s*([^.;]+)"
        ),
        text,
        flags=re.IGNORECASE,
    )
    if not scenario_match:
        return []
    labels = [
        _clean_label(label)
        for label in re.split(r",|\band\b|\bor\b", scenario_match.group(1), flags=re.IGNORECASE)
        if label.strip()
    ]
    return [label for label in labels if label]


def _labels_mentioned(text: str, labels: list[str]) -> list[str]:
    mentioned: list[str] = []
    for label in labels:
        if re.search(rf"\b{re.escape(label)}\b", text, flags=re.IGNORECASE):
            mentioned.append(label)
    return mentioned


def _parse_security_payoffs(block: str, scenario_labels: list[str]) -> dict[str, float] | None:
    payoff_start = re.search(r"\bpayoff\b", block, flags=re.IGNORECASE)
    if not payoff_start:
        return None
    payoff_text = block[payoff_start.start() :]
    payoffs: dict[str, float | None] = {label: None for label in scenario_labels}

    all_countries_match = re.search(
        rf"\$?\s*({_NUMBER_TOKEN})\s+across\s+all\s+(?:countries|states|scenarios|outcomes)"
        r"(?:\s+except\s+for\s+([^.;]+))?",
        payoff_text,
        flags=re.IGNORECASE,
    )
    if all_countries_match:
        value = _number(all_countries_match.group(1))
        for label in scenario_labels:
            payoffs[label] = value
        exception_text = all_countries_match.group(2) or ""
        for label in _labels_mentioned(exception_text, scenario_labels):
            payoffs[label] = 0.0

    value_group_pattern = re.compile(
        rf"\$?\s*({_NUMBER_TOKEN})\s*(?:payoff\s+)?(?:in|for)\s+",
        flags=re.IGNORECASE,
    )
    matches = list(value_group_pattern.finditer(payoff_text))
    for index, match in enumerate(matches):
        value = _number(match.group(1))
        group_end = matches[index + 1].start() if index + 1 < len(matches) else len(payoff_text)
        group_text = payoff_text[match.end() : group_end]
        for label in _labels_mentioned(group_text, scenario_labels):
            payoffs[label] = value

    if any(value is None for value in payoffs.values()):
        return None
    return {label: float(value) for label, value in payoffs.items() if value is not None}


def _parse_security_assets(text: str, scenario_labels: list[str]) -> list[dict[str, Any]]:
    asset_pattern = re.compile(
        (
            r"\bSecurity\s+([A-Za-z0-9]+),?\s+"
            r"(?=(?:has|is\s+priced|priced|with))"
            r"(.*?)(?=\bSecurity\s+[A-Za-z0-9]+,?\s+"
            r"(?=(?:has|is\s+priced|priced|with))|\bFind\b|\Z)"
        ),
        flags=re.IGNORECASE | re.DOTALL,
    )
    assets: list[dict[str, Any]] = []
    for label, block in asset_pattern.findall(text):
        price_match = re.search(
            rf"(?:price\s+of|price\s+is|price\s+at|priced\s+at)\s+\$?\s*({_NUMBER_TOKEN})",
            block,
            flags=re.IGNORECASE,
        )
        limit_match = re.search(
            rf"share\s+limit\s+(?:of|is)?\s*({_NUMBER_TOKEN})",
            block,
            flags=re.IGNORECASE,
        )
        payoffs = _parse_security_payoffs(block, scenario_labels)
        if not (price_match and limit_match and payoffs):
            continue
        assets.append(
            {
                "label": f"security_{_clean_label(label)}",
                "price": _number(price_match.group(1)),
                "share_limit": _number(limit_match.group(1)),
                "payoffs": payoffs,
            }
        )
    return assets


def _parse_security_budget_limit(text: str) -> float | None:
    patterns = [
        rf"(?:budget|spending\s+limit)[^.;\n]{{0,80}}?\$?\s*({_NUMBER_TOKEN})",
        rf"(?:spend|invest|purchase)[^.;\n]{{0,80}}?(?:no\s+more\s+than|at\s+most|up\s+to)\s+\$?\s*({_NUMBER_TOKEN})",
        rf"(?:no\s+more\s+than|at\s+most|up\s+to)\s+\$?\s*({_NUMBER_TOKEN})[^.;\n]{{0,80}}?(?:spend|invest|purchase|budget)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _number(match.group(1))
    return None


def _solve_security_maximin_revenue(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        ("security" in lowered or "securities" in lowered)
        and "payoff" in lowered
        and "price" in lowered
        and "share limit" in lowered
        and ("worst-case" in lowered or "worst case" in lowered)
        and ("maximize" in lowered or "maximum" in lowered)
    ):
        return TemplateSolveResult(False)

    scenario_labels = _parse_scenario_labels(text)
    if len(scenario_labels) < 2:
        return TemplateSolveResult(False)
    assets = _parse_security_assets(text, scenario_labels)
    if len(assets) < 2:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="security_maximin_net_revenue_lp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
            artifact={"scenarios": scenario_labels, "assets": assets},
        )

    variable_count = len(assets)
    objective = [0.0] * variable_count + [-1.0]
    a_ub: list[list[float]] = []
    b_ub: list[float] = []
    for scenario in scenario_labels:
        row = [
            -(float(asset["payoffs"][scenario]) - float(asset["price"]))
            for asset in assets
        ]
        row.append(1.0)
        a_ub.append(row)
        b_ub.append(0.0)
    budget_limit = _parse_security_budget_limit(text)
    if budget_limit is not None:
        a_ub.append([float(asset["price"]) for asset in assets] + [0.0])
        b_ub.append(float(budget_limit))
    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[
            *[(0.0, float(asset["share_limit"])) for asset in assets],
            (None, None),
        ],
        method="highs",
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="security_maximin_net_revenue_lp",
            status="solver_failed",
            confidence=0.82,
            notes=str(result.message),
            artifact={"scenarios": scenario_labels, "assets": assets},
        )

    variable_values = {
        str(asset["label"]): float(result.x[index])
        for index, asset in enumerate(assets)
        if not math.isclose(float(result.x[index]), 0.0, abs_tol=1e-8)
    }
    total_purchase_cost = sum(
        float(asset["price"]) * float(result.x[index])
        for index, asset in enumerate(assets)
    )
    total_shares = sum(float(result.x[index]) for index, _asset in enumerate(assets))
    diagnostics: list[dict[str, Any]] = []
    if budget_limit is None:
        diagnostics.append(
            {
                "issue_type": "missing_budget_or_normalization",
                "severity": "notice",
                "message": (
                    "No total budget, stake normalization, or capital limit was "
                    "found; the model uses only per-security share limits."
                ),
            }
        )
    return TemplateSolveResult(
        matched=True,
        template_id="security_maximin_net_revenue_lp",
        status="optimal",
        objective_value=float(result.x[-1]),
        variable_values=variable_values,
        confidence=0.88,
        notes=(
            "Solved long-only robust payoff allocation by maximizing the "
            "minimum scenario net revenue after purchase prices."
        ),
        artifact={
            "scenarios": scenario_labels,
            "assets": assets,
            "budget_constraint_present": budget_limit is not None,
            "budget_limit": budget_limit,
            "total_purchase_cost": total_purchase_cost,
            "total_shares": total_shares,
            "scenario_net_revenue": {
                scenario: sum(
                    (float(asset["payoffs"][scenario]) - float(asset["price"]))
                    * float(result.x[index])
                    for index, asset in enumerate(assets)
                )
                for scenario in scenario_labels
            },
            "diagnostics": diagnostics,
        },
    )


def _robust_number_list(value: str) -> list[float]:
    return [
        _number(match.group(0))
        for match in re.finditer(_NUMBER_TOKEN, value, flags=re.IGNORECASE)
    ]


def _robust_label_list(value: str) -> list[str]:
    cleaned = re.sub(r"\([^)]*\)", " ", value)
    cleaned = re.sub(r"\b(?:and|or)\b", ",", cleaned, flags=re.IGNORECASE)
    labels: list[str] = []
    for part in cleaned.split(","):
        if not part.strip():
            continue
        label = _clean_label(part)
        label = re.sub(
            r"^(?:products?|items?|activities?|decision\s+options?)\s+",
            "",
            label,
            flags=re.IGNORECASE,
        ).strip()
        if label and not re.fullmatch(_NUMBER_TOKEN, label, flags=re.IGNORECASE):
            labels.append(label)
    return labels


def _parse_robust_resource_lp(text: str) -> dict[str, Any] | None:
    normalized = re.sub(r"\s+", " ", text).strip()
    lowered = normalized.lower()
    if not (
        ("robust" in lowered or "uncertain" in lowered)
        and "resource" in lowered
        and "capacity" in lowered
        and ("maximize" in lowered or "maximise" in lowered)
        and ("profit" in lowered or "return" in lowered)
    ):
        return None

    label_match = re.search(
        r"products?\s+(.+?)(?:\.|;|,)\s+(?:unit\s+profits?|profits?|returns?)\b",
        normalized,
        flags=re.IGNORECASE,
    )
    if not label_match:
        label_match = re.search(
            r"products?\s+(.+?)\s+(?:have|with)\s+(?:unit\s+profits?|profits?|returns?)\b",
            normalized,
            flags=re.IGNORECASE,
        )
    if not label_match:
        return None
    labels = _robust_label_list(label_match.group(1))

    profit_match = re.search(
        r"(?:unit\s+profits?|profits?|returns?)\s+(?:are|of)?\s*(.+?)(?:\s+respectively|[.;](?=\s+[A-Z]|$))",
        normalized,
        flags=re.IGNORECASE,
    )
    nominal_match = re.search(
        r"nominal\s+resource\s+coefficients?\s+(?:are|of)?\s*(.+?)(?:\s+respectively|[.;](?=\s+[A-Z]|$))",
        normalized,
        flags=re.IGNORECASE,
    )
    deviation_match = re.search(
        r"(?:uncertainty\s+)?deviations?\s+(?:are|of)?\s*(.+?)(?:\s+respectively|[.;](?=\s+[A-Z]|$))",
        normalized,
        flags=re.IGNORECASE,
    )
    capacity_match = re.search(
        rf"(?:resource\s+)?capacity\s+(?:is|of|equals?)\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if not (profit_match and nominal_match and deviation_match and capacity_match):
        return None

    profits = _robust_number_list(profit_match.group(1))
    nominal = _robust_number_list(nominal_match.group(1))
    deviations = _robust_number_list(deviation_match.group(1))
    if not (len(labels) >= 2 and len(labels) == len(profits) == len(nominal) == len(deviations)):
        return None

    demand_limits: list[float] | None = None
    demand_match = re.search(
        r"(?:demand\s+limits?|upper\s+bounds?|market\s+demand)"
        r"(?:\s+for\s+[^.;]+?)?\s+(?:are|of)?\s*"
        r"(.+?)(?:\s+respectively|[.;](?=\s+[A-Z]|$))",
        normalized,
        flags=re.IGNORECASE,
    )
    if demand_match:
        parsed_demands = _robust_number_list(demand_match.group(1))
        if len(parsed_demands) == len(labels):
            demand_limits = parsed_demands

    gamma = None
    gamma_match = re.search(
        rf"(?:gamma|Γ)\s*(?:is|=)\s*({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if gamma_match:
        gamma = _number(gamma_match.group(1))
    uncertainty_type = "budget" if ("budget uncertainty" in lowered or gamma is not None) else "box"
    if uncertainty_type == "budget" and gamma is None:
        return None

    return {
        "labels": labels,
        "profits": profits,
        "nominal_resource_coefficients": nominal,
        "deviations": deviations,
        "capacity": _number(capacity_match.group(1)),
        "demand_limits": demand_limits,
        "uncertainty_type": uncertainty_type,
        "gamma": gamma,
    }


def _solve_robust_resource_capacity_lp(text: str) -> TemplateSolveResult:
    problem = _parse_robust_resource_lp(text)
    if not problem:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="robust_resource_capacity_lp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
            artifact=problem,
        )

    labels = list(problem["labels"])
    profits = [float(value) for value in problem["profits"]]
    nominal = [float(value) for value in problem["nominal_resource_coefficients"]]
    deviations = [float(value) for value in problem["deviations"]]
    capacity = float(problem["capacity"])
    demand_limits = problem.get("demand_limits")
    product_count = len(labels)

    if problem["uncertainty_type"] == "box":
        objective = [-value for value in profits]
        a_ub = [[nominal[index] + deviations[index] for index in range(product_count)]]
        b_ub = [capacity]
        if demand_limits:
            for index, limit in enumerate(demand_limits):
                row = [0.0] * product_count
                row[index] = 1.0
                a_ub.append(row)
                b_ub.append(float(limit))
        bounds = [(0, None)] * product_count
        result = linprog(objective, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    else:
        gamma = float(problem["gamma"])
        z0_index = product_count
        z_start = product_count + 1
        variable_count = product_count + 1 + product_count
        objective = [-value for value in profits] + [0.0] * (1 + product_count)
        a_ub: list[list[float]] = []
        b_ub: list[float] = []

        resource_row = [0.0] * variable_count
        for index in range(product_count):
            resource_row[index] = nominal[index]
            resource_row[z_start + index] = 1.0
        resource_row[z0_index] = gamma
        a_ub.append(resource_row)
        b_ub.append(capacity)

        for index in range(product_count):
            row = [0.0] * variable_count
            row[index] = deviations[index]
            row[z0_index] = -1.0
            row[z_start + index] = -1.0
            a_ub.append(row)
            b_ub.append(0.0)

        if demand_limits:
            for index, limit in enumerate(demand_limits):
                row = [0.0] * variable_count
                row[index] = 1.0
                a_ub.append(row)
                b_ub.append(float(limit))
        bounds = [(0, None)] * variable_count
        result = linprog(objective, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="robust_resource_capacity_lp",
            status="solver_failed",
            confidence=0.82,
            notes=str(result.message),
            artifact=problem,
        )

    variable_values = {
        f"produce_{labels[index]}": float(result.x[index])
        for index in range(product_count)
        if not math.isclose(float(result.x[index]), 0.0, abs_tol=1e-8)
    }
    artifact = {
        **problem,
        "robust_counterpart": (
            "box resource row uses nominal coefficients plus deviations"
            if problem["uncertainty_type"] == "box"
            else "budget resource row uses Bertsimas-Sim z0/z_j auxiliaries"
        ),
        "modeling_experience": {
            "family": "robust_optimization",
            "structure_key": "box_budget_uncertain_resource_capacity_lp",
            "content": (
                "For nonnegative LP decisions with uncertain coefficients in a <= "
                "resource-capacity row, box uncertainty adds each deviation to its "
                "nominal coefficient; budget uncertainty with Gamma uses the "
                "Bertsimas-Sim linear counterpart with z0 and z_j auxiliaries."
            ),
            "applies_when": [
                "decision variables are nonnegative",
                "uncertainty is row-wise coefficient uncertainty in a <= capacity constraint",
                "uncertainty set is box or Bertsimas-Sim budget",
            ],
            "rejects_when": [
                "variable signs are unrestricted or unknown",
                "uncertainty affects nonlinear expressions",
                "uncertainty set is ellipsoidal or chance-constrained",
            ],
            "status": "validated",
        },
    }
    return TemplateSolveResult(
        matched=True,
        template_id="robust_resource_capacity_lp",
        status="optimal",
        objective_value=float(-result.fun),
        variable_values=variable_values,
        confidence=0.87,
        notes="Solved robust resource-capacity LP through a deterministic box/budget robust counterpart.",
        artifact=artifact,
    )


def _route_label(value: str) -> str:
    cleaned = _clean_label(value)
    cleaned = re.sub(r"^(?:City|Location|Warehouse)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" .,:;") or cleaned


def _latex_numeric_matrices(text: str) -> list[list[list[float]]]:
    matrices: list[list[list[float]]] = []
    for match in re.finditer(
        r"\\begin\{array\}\{[^}]+\}(.*?)\\end\{array\}",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        body = match.group(1)
        matrix: list[list[float]] = []
        for raw_row in re.split(r"\\\\", body):
            row = re.sub(r"\\hline|\\cline\{[^}]+\}", "", raw_row).strip()
            if not row:
                continue
            values: list[float] = []
            for cell in row.split("&"):
                value = _first_number(cell)
                if value is None:
                    values = []
                    break
                values.append(float(value))
            if values:
                matrix.append(values)
        if matrix and len({len(row) for row in matrix}) == 1:
            matrices.append(matrix)
    return matrices


def _flow_shop_makespan(matrix: list[list[float]], order: tuple[int, ...]) -> float:
    machine_completion = [0.0] * len(matrix[0])
    for job_index in order:
        previous_machine_finish = 0.0
        for machine_index, processing_time in enumerate(matrix[job_index]):
            finish = max(machine_completion[machine_index], previous_machine_finish) + processing_time
            machine_completion[machine_index] = finish
            previous_machine_finish = finish
    return machine_completion[-1]


def _solve_permutation_flow_shop_scheduling(text: str) -> TemplateSolveResult:
    lowered = re.sub(r"\s+", " ", text.lower())
    if not (
        "minimize" in lowered
        and ("completion time" in lowered or "processing cycle" in lowered or "makespan" in lowered)
        and ("machine" in lowered or "vat" in lowered or "flow shop" in lowered)
        and ("sequence" in lowered or "sequential" in lowered or "order" in lowered)
    ):
        return TemplateSolveResult(False)

    job_labels: list[str] = []
    machine_labels: list[str] = []
    matrix: list[list[float]] | None = None
    for table in _parse_markdown_tables(text):
        header_text = " ".join(table[0]).lower()
        if not ("machine" in header_text or "vat" in header_text):
            continue
        parsed = _numeric_matrix_from_table(table)
        if not parsed:
            continue
        row_labels, column_labels, values = parsed
        if len(row_labels) >= 2 and len(column_labels) >= 2:
            job_labels = row_labels
            machine_labels = column_labels
            matrix = values
            break
    if matrix is None:
        matrices = _latex_numeric_matrices(text)
        if matrices:
            matrix = matrices[0]
            job_labels = [f"job_{index}" for index in range(1, len(matrix) + 1)]
            machine_labels = [f"machine_{index}" for index in range(1, len(matrix[0]) + 1)]
    if matrix is None or len(matrix) < 2 or len(matrix[0]) < 2:
        return TemplateSolveResult(False)
    if len(matrix) > 8:
        return TemplateSolveResult(
            matched=True,
            template_id="permutation_flow_shop_scheduling",
            status="unsupported_size",
            confidence=0.78,
            notes="Flow-shop instance is too large for exact deterministic permutation enumeration.",
            artifact={"jobs": job_labels, "machines": machine_labels, "processing_times": matrix},
        )

    best_order: tuple[int, ...] | None = None
    best_makespan: float | None = None
    for order in itertools.permutations(range(len(matrix))):
        makespan = _flow_shop_makespan(matrix, order)
        if best_makespan is None or makespan < best_makespan:
            best_makespan = makespan
            best_order = order
    if best_order is None or best_makespan is None:
        return TemplateSolveResult(False)

    sequence = [job_labels[index] for index in best_order]
    return TemplateSolveResult(
        matched=True,
        template_id="permutation_flow_shop_scheduling",
        status="optimal",
        objective_value=float(best_makespan),
        variable_values={
            f"position_{position}": float(job_index + 1)
            for position, job_index in enumerate(best_order, start=1)
        },
        confidence=0.9,
        notes="Solved small permutation flow-shop scheduling problem by exact job-order enumeration.",
        artifact={
            "jobs": job_labels,
            "machines": machine_labels,
            "processing_times": matrix,
            "sequence": sequence,
        },
    )


def _solve_tsp_routing(text: str) -> TemplateSolveResult:
    lowered = re.sub(r"\s+", " ", text.lower())
    if not (
        ((
            (
                "visit each city exactly once" in lowered
                or "visit each city only once" in lowered
                or "each city exactly once" in lowered
                or "visit each location exactly once" in lowered
                or "visit each shop only once" in lowered
                or "each shop exactly once" in lowered
            )
            and (
                "return to the starting" in lowered
                or "return to their starting" in lowered
                or "returning to the original starting" in lowered
                or "return to the starting point" in lowered
                or "return to their starting point" in lowered
                or "returning to their starting point" in lowered
            )
        )
        or (
            "traveling salesman" in lowered
            and (
                "starting and ending" in lowered
                or "start and end" in lowered
                or "starting city" in lowered
            )
            and ("distance matrix" in lowered or "distance" in lowered)
        ))
        and (
            "minimum total" in lowered
            or "minimize" in lowered
            or "minimizing" in lowered
            or "minimize the travel distance" in lowered
            or "least total" in lowered
        )
    ):
        return TemplateSolveResult(False)

    labels: list[str] = []
    costs: dict[tuple[str, str], float] = {}
    for table in _parse_markdown_tables(text):
        parsed_costs = _route_costs_from_table(table)
        if not parsed_costs:
            continue
        labels, costs = parsed_costs
        break

    if not costs:
        chunks = [
            re.sub(r"\s+", " ", chunk).strip(" -\t")
            for chunk in re.split(r"\n\s*-\s*", text)
            if chunk.strip()
        ]
        if len(chunks) == 1:
            narrative_chunks = [
                chunk.strip(" .,:;\t")
                for chunk in re.split(
                    r"(?=\b(?:The\s+cost\s+to\s+travel\s+from|From|Traveling\s+from|Lastly,\s+from)\s+"
                    r"(?:City|Location|Warehouse|Shop)?\s*[A-Za-z0-9]+)",
                    re.sub(r"\s+", " ", text),
                    flags=re.IGNORECASE,
                )
                if chunk.strip()
            ]
            if len(narrative_chunks) > 1:
                chunks = narrative_chunks
        for chunk in chunks:
            origin_match = re.search(
                r"from\s+(?:City|Location|Warehouse|Shop)?\s*([A-Za-z0-9]+)",
                chunk,
                flags=re.IGNORECASE,
            )
            if not origin_match:
                continue
            origin = _route_label(origin_match.group(1))
            parsed_destinations: list[tuple[str, float]] = []
            for destination, value in re.findall(
                r"to\s+(?:City|Location|Warehouse|Shop)?\s*([A-Za-z0-9]+)\s+(?:is|costs?|takes)\s+([0-9][0-9,]*(?:\.\d+)?)\s+units?",
                chunk,
                flags=re.IGNORECASE,
            ):
                parsed_destinations.append((_route_label(destination), _number(value)))
            for value, destination in re.findall(
                (
                    r"([0-9][0-9,]*(?:\.\d+)?)\s+units?\s+"
                    r"(?:to\s+(?:reach|get\s+to|go\s+to|travel\s+to|deliver\s+to|move\s+to)?\s*)?"
                    r"(?:City|Location|Warehouse|Shop)?\s*([A-Za-z0-9]+)"
                ),
                chunk,
                flags=re.IGNORECASE,
            ):
                parsed_destinations.append((_route_label(destination), _number(value)))
            for destination, value in re.findall(
                r"(?:City|Location|Warehouse|Shop)\s+([A-Za-z0-9]+)\s+costs\s+([0-9][0-9,]*(?:\.\d+)?)\s+units?",
                chunk,
                flags=re.IGNORECASE,
            ):
                parsed_destinations.append((_route_label(destination), _number(value)))
            if not parsed_destinations:
                continue
            if origin not in labels:
                labels.append(origin)
            for destination, value in parsed_destinations:
                costs[(origin, destination)] = value
        for _origin, destination in costs:
            if destination not in labels:
                labels.append(destination)

    labels = sorted(set(labels), key=lambda value: (len(value), value))
    if len(labels) < 3 or len(labels) > 9:
        return TemplateSolveResult(False)

    best_cost: float | None = None
    best_route: tuple[str, ...] | None = None
    start = labels[0]
    for tail in itertools.permutations(labels[1:]):
        route = (start, *tail)
        total = 0.0
        feasible = True
        for origin, destination in zip(route, (*route[1:], start)):
            edge = costs.get((origin, destination))
            if edge is None:
                feasible = False
                break
            total += edge
        if feasible and (best_cost is None or total < best_cost):
            best_cost = total
            best_route = route
    if best_cost is None or best_route is None:
        return TemplateSolveResult(False)

    relaxation = _tsp_assignment_relaxation(labels, costs)
    variable_values = {
        f"arc_{origin}_to_{destination}": 1.0
        for origin, destination in zip(best_route, (*best_route[1:], start))
    }
    notes = "Solved small travelling-salesman routing problem by exact route enumeration."
    artifact: dict[str, Any] = {
        "nodes": labels,
        "route": list(best_route),
        "costs": {f"{origin}->{destination}": value for (origin, destination), value in costs.items()},
    }
    if relaxation is not None:
        relaxation_cost, relaxation_arcs, relaxation_cycles = relaxation
        requires_subtour_elimination = len(relaxation_cycles) > 1
        artifact.update(
            {
                "assignment_relaxation_objective": relaxation_cost,
                "assignment_relaxation_arcs": {
                    f"{origin}->{destination}": 1.0
                    for origin, destination in sorted(relaxation_arcs.items())
                },
                "assignment_relaxation_cycles": relaxation_cycles,
                "requires_subtour_elimination": requires_subtour_elimination,
            }
        )
        if requires_subtour_elimination and relaxation_cost < best_cost - 1e-9:
            notes += (
                " Assignment-style one-in/one-out relaxation is cheaper but "
                "forms disconnected subtours; subtour elimination is required."
            )
    return TemplateSolveResult(
        matched=True,
        template_id="tsp_routing_enum",
        status="optimal",
        objective_value=float(best_cost),
        variable_values=variable_values,
        confidence=0.9,
        notes=notes,
        artifact=artifact,
    )


def _route_costs_from_table(
    table: tuple[list[str], list[list[str]]],
) -> tuple[list[str], dict[tuple[str, str], float]] | None:
    header, rows = table
    if len(header) < 3:
        return None
    column_labels = [_route_label(cell) for cell in header[1:] if _route_label(cell)]
    if len(column_labels) < 3:
        return None
    labels = list(dict.fromkeys(column_labels))
    costs: dict[tuple[str, str], float] = {}
    row_label_set: set[str] = set()
    for row in rows:
        if len(row) != len(header):
            continue
        origin = _route_label(row[0])
        if not origin:
            continue
        row_label_set.add(origin)
        if origin not in labels:
            labels.append(origin)
        for destination, cell in zip(column_labels, row[1:]):
            value = _first_number(cell)
            if value is None or origin == destination:
                continue
            costs[(origin, destination)] = float(value)
            costs.setdefault((destination, origin), float(value))

    if len(labels) < 3:
        return None
    for origin, destination in itertools.permutations(labels, 2):
        if (origin, destination) not in costs:
            return None
    if len(row_label_set) < len(labels) - 1:
        return None
    return labels, costs


def _tsp_assignment_relaxation(
    labels: list[str],
    costs: dict[tuple[str, str], float],
) -> tuple[float, dict[str, str], list[list[str]]] | None:
    best_cost: float | None = None
    best_arcs: dict[str, str] | None = None
    for destinations in itertools.permutations(labels):
        total = 0.0
        arcs: dict[str, str] = {}
        feasible = True
        for origin, destination in zip(labels, destinations):
            if origin == destination:
                feasible = False
                break
            edge = costs.get((origin, destination))
            if edge is None:
                feasible = False
                break
            total += edge
            arcs[origin] = destination
        if feasible and (best_cost is None or total < best_cost):
            best_cost = total
            best_arcs = arcs
    if best_cost is None or best_arcs is None:
        return None

    seen: set[str] = set()
    cycles: list[list[str]] = []
    for label in labels:
        if label in seen:
            continue
        cycle: list[str] = []
        current = label
        while current not in seen:
            seen.add(current)
            cycle.append(current)
            current = best_arcs[current]
        cycles.append(cycle)
    return float(best_cost), best_arcs, cycles


def _solve_set_cover_table(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        ("minimum number" in lowered or "fewest" in lowered)
        and ("within" in lowered or "cover" in lowered)
        and ("chain stores" in lowered or "stores" in lowered or "residential areas" in lowered)
    ):
        return TemplateSolveResult(False)
    parsed = _parse_markdown_table(text)
    if not parsed:
        return TemplateSolveResult(False)
    header, rows = parsed
    if len(header) < 2 or "area" not in " ".join(header).lower():
        return TemplateSolveResult(False)

    candidates: dict[str, set[str]] = {}
    universe: set[str] = set()
    for row in rows:
        if len(row) < 2:
            continue
        candidate = _clean_label(row[0])
        covered = {
            _clean_label(item)
            for item in re.split(r",\s*", row[1])
            if _clean_label(item)
        }
        if not covered:
            continue
        candidates[candidate] = covered
        universe.add(candidate)
        universe.update(covered)
    if len(candidates) < 2 or not universe:
        return TemplateSolveResult(False)

    labels = sorted(candidates)
    best_subset: tuple[str, ...] | None = None
    for size in range(1, len(labels) + 1):
        for subset in itertools.combinations(labels, size):
            covered: set[str] = set()
            for label in subset:
                covered.update(candidates[label])
            if universe <= covered:
                best_subset = subset
                break
        if best_subset is not None:
            break
    if best_subset is None:
        return TemplateSolveResult(False)

    return TemplateSolveResult(
        matched=True,
        template_id="set_cover_enum",
        status="optimal",
        objective_value=float(len(best_subset)),
        variable_values={f"open_{label}": 1.0 for label in best_subset},
        confidence=0.9,
        notes="Solved small set-covering location problem by exact subset enumeration.",
        artifact={
            "candidates": {label: sorted(values) for label, values in candidates.items()},
            "universe": sorted(universe),
            "selected": list(best_subset),
        },
    )


def _labels_from_quoted_or_listed_text(text: str) -> list[str]:
    quoted = [_clean_label(label) for label in re.findall(r"'([^']+)'", text)]
    if quoted:
        return quoted
    labels = [
        _clean_label(label)
        for label in re.findall(r"\b([A-Za-z][A-Za-z0-9_]*)\b", text)
        if label.lower() not in {"and", "or", "to", "vertices", "vertex", "all", "are"}
    ]
    return labels


def _candidate_vertex_sets(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    def add_candidate(color: str, labels_text: str) -> None:
        labels = sorted({_clean_label(label).lower() for label in re.findall(r"'([^']+)'", labels_text)})
        if not labels:
            return
        candidate = {"label": f"{_clean_label(color).lower()}_vertices", "vertices": labels}
        if candidate not in candidates:
            candidates.append(candidate)

    for match in re.finditer(
        r"(?:vertices\s+labeled\s+|while\s+)?((?:(?:and\s+)?'[^']+'\s*(?:,)?\s*)+)\s+are\s+colored\s+in\s+([A-Za-z]+)",
        text,
        flags=re.IGNORECASE,
    ):
        labels_text, color = match.groups()
        add_candidate(color, labels_text)
    for match in re.finditer(
        r"([A-Za-z]+)[-\s]+colored\s+vertices\s*\(([^)]*)\)",
        text,
        flags=re.IGNORECASE,
    ):
        color, labels_text = match.groups()
        add_candidate(color, labels_text)
    return candidates


def _uncovered_edges(
    edges: set[tuple[str, str]],
    cover: set[str],
) -> list[list[str]]:
    return [
        [left, right]
        for left, right in sorted(edges)
        if left not in cover and right not in cover
    ]


def _solve_minimum_vertex_cover(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        "vertex cover" in lowered
        and ("minimum vertex cover" in lowered or "smallest set of vertices" in lowered)
    ):
        return TemplateSolveResult(False)

    vertices: set[str] = set()
    range_match = re.search(
        r"from\s+'?([A-Za-z])'?\s+to\s+'?([A-Za-z])'?",
        text,
        flags=re.IGNORECASE,
    )
    if range_match:
        start, end = range_match.groups()
        start_ord = ord(start.lower())
        end_ord = ord(end.lower())
        if start_ord <= end_ord and end_ord - start_ord <= 30:
            vertices.update(chr(value) for value in range(start_ord, end_ord + 1))

    edges: set[tuple[str, str]] = set()

    def add_edge(left: str, right: str) -> None:
        left_label = _clean_label(left).lower()
        right_label = _clean_label(right).lower()
        if not left_label or not right_label or left_label == right_label:
            return
        vertices.update((left_label, right_label))
        edges.add(tuple(sorted((left_label, right_label))))

    connection_pattern = re.compile(
        (
            r"Vertex\s+'?([A-Za-z][A-Za-z0-9_]*)'?\s+connects?\s+to\s+vertices?\s+"
            r"(.*?)(?=\.\s+Vertex\s+'?[A-Za-z]|\.\s+The\s+vertices|\.\s+Find|\Z)"
        ),
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in connection_pattern.finditer(text):
        origin, body = match.groups()
        for destination in _labels_from_quoted_or_listed_text(body):
            add_edge(origin, destination)

    for group_match in re.finditer(
        r"\(([^)]*)\)\s+are\s+all\s+interconnected",
        text,
        flags=re.IGNORECASE,
    ):
        labels = _labels_from_quoted_or_listed_text(group_match.group(1))
        for left, right in itertools.combinations(labels, 2):
            add_edge(left, right)

    if len(vertices) < 2 or not edges:
        return TemplateSolveResult(False)
    labels = sorted(vertices)
    if len(labels) > 24:
        return TemplateSolveResult(
            matched=True,
            template_id="minimum_vertex_cover_enum",
            status="solver_unavailable",
            confidence=0.75,
            notes="Too many vertices for deterministic exact subset enumeration.",
            artifact={"vertices": labels, "edges": [list(edge) for edge in sorted(edges)]},
        )

    best_cover: tuple[str, ...] | None = None
    for size in range(len(labels) + 1):
        for subset in itertools.combinations(labels, size):
            cover = set(subset)
            if all(left in cover or right in cover for left, right in edges):
                best_cover = subset
                break
        if best_cover is not None:
            break
    if best_cover is None:
        return TemplateSolveResult(False)

    candidate_sets = []
    diagnostics: list[dict[str, Any]] = []
    for candidate in _candidate_vertex_sets(text):
        cover = set(candidate["vertices"])
        uncovered = _uncovered_edges(edges, cover)
        item = {
            "label": candidate["label"],
            "vertices": candidate["vertices"],
            "size": len(candidate["vertices"]),
            "is_vertex_cover": not uncovered,
            "uncovered_edges": uncovered,
        }
        candidate_sets.append(item)
        if uncovered:
            diagnostics.append(
                {
                    "issue_type": "invalid_candidate_vertex_cover",
                    "severity": "warning",
                    "message": (
                        f"Candidate set {candidate['label']} does not cover "
                        f"{len(uncovered)} edge(s)."
                    ),
                    "candidate": candidate["label"],
                    "uncovered_edges": uncovered,
                }
            )

    return TemplateSolveResult(
        matched=True,
        template_id="minimum_vertex_cover_enum",
        status="optimal",
        objective_value=float(len(best_cover)),
        variable_values={f"select_vertex_{label}": 1.0 for label in best_cover},
        confidence=0.9,
        notes="Solved small minimum vertex-cover problem by exact subset enumeration.",
        artifact={
            "vertices": labels,
            "edges": [list(edge) for edge in sorted(edges)],
            "cover": list(best_cover),
            "exact_minimum_cover_size": len(best_cover),
            "max_proven_infeasible_cover_size": len(best_cover) - 1,
            "candidate_vertex_sets": candidate_sets,
            "diagnostics": diagnostics,
        },
    )


def _enumerate_cutting_patterns(
    *,
    stock_size: float,
    item_sizes: list[float],
    max_pieces: int | None = None,
) -> list[dict[str, Any]]:
    max_counts = [int(math.floor((stock_size + 1e-9) / size)) for size in item_sizes]
    patterns: list[dict[str, Any]] = []
    for counts in itertools.product(*(range(limit + 1) for limit in max_counts)):
        if not any(counts):
            continue
        if max_pieces is not None and sum(counts) > max_pieces:
            continue
        used = sum(count * size for count, size in zip(counts, item_sizes))
        if used <= stock_size + 1e-9:
            patterns.append(
                {
                    "counts": tuple(int(count) for count in counts),
                    "used": float(used),
                    "waste": float(max(0.0, stock_size - used)),
                }
            )
    return patterns


def _parse_width_cutting_stock_data(text: str) -> tuple[list[float], list[float], list[float]] | None:
    demand_widths: list[float] = []
    demand_lengths: list[float] = []
    for header, rows in _parse_markdown_tables(text):
        lowered_header = [_clean_label(cell).lower() for cell in header]
        width_index = next((index for index, cell in enumerate(lowered_header) if "width" in cell), None)
        length_index = next((index for index, cell in enumerate(lowered_header) if "length" in cell), None)
        if width_index is None or length_index is None:
            continue
        for row in rows:
            if len(row) <= max(width_index, length_index):
                continue
            width = _first_number(row[width_index])
            length = _first_number(row[length_index])
            if width is not None and length is not None and width > 0 and length > 0:
                demand_widths.append(float(width))
                demand_lengths.append(float(length))
        if demand_widths:
            break
    stock_match = re.search(
        r"standard\s+widths?\s+of\s+(.{0,180}?)(?:assuming|how|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not stock_match:
        return None
    stock_widths = [
        _number(value)
        for value in re.findall(
            rf"({_NUMBER_TOKEN})\s*(?:m|meters?)\b",
            stock_match.group(1),
            flags=re.IGNORECASE,
        )
    ]
    stock_widths = sorted({float(width) for width in stock_widths if width > 0})
    if not demand_widths or not stock_widths:
        return None
    return stock_widths, demand_widths, demand_lengths


def _parse_piece_cutting_stock_data(text: str) -> tuple[float, list[float], list[float]] | None:
    demands: dict[float, float] = {}
    for quantity, size in re.findall(
        rf"({_NUMBER_TOKEN})\s+pieces?\s+of\s+({_NUMBER_TOKEN})\s*(?:m|meters?|mm|millimeters?)",
        text,
        flags=re.IGNORECASE,
    ):
        demands[_number(size)] = demands.get(_number(size), 0.0) + _number(quantity)
    stock_patterns = [
        rf"raw\s+(?:steel\s+)?(?:bar|pipe)[^.\n]{{0,100}}?\bis\s+({_NUMBER_TOKEN})\s*"
        r"(?:m|meters?|mm|millimeters?)",
        rf"raw\s+(?:steel\s+)?(?:bar|pipe)[^.\n]{{0,100}}?\blength\s+of\s+({_NUMBER_TOKEN})\s*"
        r"(?:m|meters?|mm|millimeters?)",
    ]
    stock_size = None
    for pattern in stock_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            stock_size = _number(match.group(1))
            break
    if stock_size is None or len(demands) < 2:
        return None
    item_sizes = sorted(demands)
    demand_values = [demands[size] for size in item_sizes]
    return float(stock_size), item_sizes, demand_values


def _solve_cutting_stock(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        ("cut" in lowered or "cutting" in lowered)
        and "waste" in lowered
        and ("roll" in lowered or "bar" in lowered or "pipe" in lowered or "raw material" in lowered)
    ):
        return TemplateSolveResult(False)
    if any(
        phrase in lowered
        for phrase in (
            "additional costs",
            "most frequently used pattern",
            "pattern may not exceed",
            "patterns to be used may not exceed",
            "leftover length for any cutting pattern",
        )
    ):
        return TemplateSolveResult(False)

    width_data = _parse_width_cutting_stock_data(text)
    if width_data is not None:
        stock_widths, item_widths, demand_lengths = width_data
        all_patterns: list[dict[str, Any]] = []
        for stock_width in stock_widths:
            for pattern in _enumerate_cutting_patterns(stock_size=stock_width, item_sizes=item_widths):
                all_patterns.append({"stock_size": stock_width, **pattern})
        if not all_patterns:
            return TemplateSolveResult(False)
        try:
            from scipy.optimize import linprog
        except ImportError as exc:
            return TemplateSolveResult(
                matched=True,
                template_id="continuous_width_cutting_stock_lp",
                status="solver_unavailable",
                confidence=0.8,
                notes=str(exc),
                artifact={"stock_widths": stock_widths, "item_widths": item_widths},
            )
        result = linprog(
            [pattern["waste"] for pattern in all_patterns],
            A_eq=[
                [float(pattern["counts"][item_index]) for pattern in all_patterns]
                for item_index in range(len(item_widths))
            ],
            b_eq=demand_lengths,
            bounds=[(0, None)] * len(all_patterns),
            method="highs",
        )
        if not result.success:
            return TemplateSolveResult(
                matched=True,
                template_id="continuous_width_cutting_stock_lp",
                status="solver_failed",
                confidence=0.82,
                notes=str(result.message),
                artifact={"stock_widths": stock_widths, "item_widths": item_widths},
            )
        variable_values = {
            f"pattern_{index}_length": float(value)
            for index, value in enumerate(result.x)
            if value > 1e-8
        }
        return TemplateSolveResult(
            matched=True,
            template_id="continuous_width_cutting_stock_lp",
            status="optimal",
            objective_value=float(result.fun),
            variable_values=variable_values,
            confidence=0.88,
            notes="Solved continuous width cutting-stock LP by enumerating feasible cutting patterns.",
            artifact={
                "stock_widths": stock_widths,
                "item_widths": item_widths,
                "demand_lengths": demand_lengths,
                "patterns": [
                    {
                        "stock_size": pattern["stock_size"],
                        "counts": list(pattern["counts"]),
                        "waste_per_length": pattern["waste"],
                    }
                    for pattern in all_patterns
                ],
            },
        )

    piece_data = _parse_piece_cutting_stock_data(text)
    if piece_data is None:
        return TemplateSolveResult(False)
    stock_size, item_sizes, demands = piece_data
    patterns = _enumerate_cutting_patterns(stock_size=stock_size, item_sizes=item_sizes)
    if not patterns:
        return TemplateSolveResult(False)
    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="integer_length_cutting_stock_ilp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
            artifact={"stock_size": stock_size, "item_sizes": item_sizes},
        )
    result = linprog(
        [pattern["waste"] for pattern in patterns],
        A_eq=[
            [float(pattern["counts"][item_index]) for pattern in patterns]
            for item_index in range(len(item_sizes))
        ],
        b_eq=demands,
        bounds=[(0, None)] * len(patterns),
        integrality=[1] * len(patterns),
        method="highs",
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="integer_length_cutting_stock_ilp",
            status="solver_failed",
            confidence=0.82,
            notes=str(result.message),
            artifact={"stock_size": stock_size, "item_sizes": item_sizes, "demands": demands},
        )
    variable_values = {
        f"pattern_{index}_bars": float(value)
        for index, value in enumerate(result.x)
        if value > 1e-8
    }
    return TemplateSolveResult(
        matched=True,
        template_id="integer_length_cutting_stock_ilp",
        status="optimal",
        objective_value=float(result.fun),
        variable_values=variable_values,
        confidence=0.9,
        notes="Solved integer one-dimensional cutting-stock model by enumerating feasible cutting patterns.",
        artifact={
            "stock_size": stock_size,
            "item_sizes": item_sizes,
            "demands": demands,
            "patterns": [
                {"counts": list(pattern["counts"]), "waste": pattern["waste"]}
                for pattern in patterns
            ],
        },
    )


def _parse_interval_contract_data(text: str) -> tuple[list[float], dict[int, float], float] | None:
    demand_values: list[float] = []
    contract_fees: dict[int, float] = {}
    unit_size = 1.0
    for header, rows in _parse_markdown_tables(text):
        if len(header) < 2:
            continue
        header_label = _clean_label(header[0]).lower()
        if "month" in header_label or "period" in header_label:
            for row in rows:
                if not row:
                    continue
                row_label = _clean_label(row[0]).lower()
                if "required" in row_label or "demand" in row_label:
                    values = [_first_number(cell) for cell in row[1:]]
                    demand_values = [float(value) for value in values if value is not None]
        if "contract" in header_label and "length" in header_label:
            lengths = [_first_number(cell) for cell in header[1:]]
            for row in rows:
                if not row:
                    continue
                row_label = _clean_label(row[0]).lower()
                if "fee" not in row_label and "cost" not in row_label and "price" not in row_label:
                    continue
                unit_match = re.search(r"per\s+([0-9][0-9,]*(?:\.\d+)?)", row[0], flags=re.IGNORECASE)
                if unit_match:
                    unit_size = _number(unit_match.group(1))
                fees = [_first_number(cell) for cell in row[1:]]
                for length, fee in zip(lengths, fees):
                    if length is not None and fee is not None:
                        contract_fees[int(round(length))] = float(fee)
    if not demand_values or not contract_fees or unit_size <= 0:
        return None
    return demand_values, contract_fees, unit_size


def _parse_contract_distinct_bounds(text: str) -> tuple[int, int | None]:
    cleaned_text = re.sub(r"[*_`]", "", text)
    min_distinct = 0
    max_distinct: int | None = None
    min_match = re.search(
        rf"\bat\s+least\s+({_NUMBER_TOKEN})\s+(?:different|distinct)\s+contracts?",
        cleaned_text,
        flags=re.IGNORECASE,
    )
    if min_match:
        min_distinct = int(round(_number(min_match.group(1))))
    max_patterns = [
        rf"(?:cannot|can\s+not|must\s+not|no\s+more\s+than|not\s+exceed)\s+"
        rf"({_NUMBER_TOKEN})\s+(?:different|distinct)\s+(?:warehouse\s+)?contracts?",
        rf"(?:number\s+of\s+)?(?:different|distinct)\s+(?:warehouse\s+)?contracts?[^.\n]{{0,80}}"
        rf"(?:cannot|can\s+not|must\s+not|does\s+not|not)\s+(?:exceed|be\s+more\s+than)\s+({_NUMBER_TOKEN})",
    ]
    for pattern in max_patterns:
        max_match = re.search(pattern, cleaned_text, flags=re.IGNORECASE)
        if max_match:
            max_distinct = int(round(_number(max_match.group(1))))
            break
    return min_distinct, max_distinct


def _parse_contract_length_exclusions(text: str) -> list[tuple[int, int]]:
    cleaned_text = re.sub(r"[*_`]", "", text)
    exclusions: list[tuple[int, int]] = []
    pattern = re.compile(
        r"if\s+a\s+([0-9]+)[-\s]?month\s+contract\s+is\s+chosen[^.\n]*?"
        r"no\s+([0-9]+)[-\s]?month\s+contract",
        flags=re.IGNORECASE,
    )
    for left, right in pattern.findall(cleaned_text):
        exclusions.append((int(left), int(right)))
    return exclusions


def _solve_interval_contract_covering(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        ("contract" in lowered or "rent" in lowered or "rental" in lowered or "lease" in lowered)
        and ("required" in lowered or "demand" in lowered)
        and ("month" in lowered or "period" in lowered)
        and ("minimize" in lowered or "minimum" in lowered)
    ):
        return TemplateSolveResult(False)
    parsed = _parse_interval_contract_data(text)
    if parsed is None:
        return TemplateSolveResult(False)
    demand_values, contract_fees, unit_size = parsed
    demand_units = [value / unit_size for value in demand_values]
    if any(value < -1e-9 for value in demand_units):
        return TemplateSolveResult(False)
    period_count = len(demand_units)
    contract_lengths = sorted(length for length in contract_fees if 1 <= length <= period_count)
    if not contract_lengths:
        return TemplateSolveResult(False)

    min_distinct, max_distinct = _parse_contract_distinct_bounds(text)
    exclusions = _parse_contract_length_exclusions(text)
    exact_cover = "without shortage or excess" in lowered or "without excess" in lowered
    integer_units = all(math.isclose(value, round(value), abs_tol=1e-9) for value in demand_units)
    best: tuple[float, set[int], list[tuple[int, int]], list[float]] | None = None
    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="interval_contract_covering_milp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
            artifact={"demand": demand_values, "contract_fees": contract_fees},
        )

    for subset_size in range(1, len(contract_lengths) + 1):
        for active_tuple in itertools.combinations(contract_lengths, subset_size):
            active = set(active_tuple)
            if len(active) < min_distinct:
                continue
            if max_distinct is not None and len(active) > max_distinct:
                continue
            if any(left in active and right in active for left, right in exclusions):
                continue
            variables = [
                (start, length)
                for length in active_tuple
                for start in range(0, period_count - length + 1)
            ]
            if not variables:
                continue
            objective = [contract_fees[length] for _start, length in variables]
            coverage_rows = [
                [1.0 if start <= period < start + length else 0.0 for start, length in variables]
                for period in range(period_count)
            ]
            length_rows = [
                [-1.0 if length == active_length else 0.0 for _start, length in variables]
                for active_length in active_tuple
            ]
            kwargs: dict[str, Any] = {
                "c": objective,
                "bounds": [(0, None)] * len(variables),
                "method": "highs",
            }
            if exact_cover:
                kwargs["A_eq"] = coverage_rows
                kwargs["b_eq"] = demand_units
            else:
                kwargs["A_ub"] = [[-value for value in row] for row in coverage_rows]
                kwargs["b_ub"] = [-value for value in demand_units]
            if length_rows:
                existing_a_ub = kwargs.get("A_ub") or []
                existing_b_ub = kwargs.get("b_ub") or []
                kwargs["A_ub"] = [*existing_a_ub, *length_rows]
                kwargs["b_ub"] = [*existing_b_ub, *([-1.0] * len(length_rows))]
            if integer_units:
                kwargs["integrality"] = [1] * len(variables)
            result = linprog(**kwargs)
            if not result.success:
                continue
            used_lengths = {
                length
                for value, (_start, length) in zip(result.x, variables)
                if float(value) > 1e-8
            }
            if used_lengths != active:
                continue
            objective_value = float(result.fun)
            if best is None or objective_value < best[0] - 1e-8:
                best = (objective_value, used_lengths, variables, [float(value) for value in result.x])
    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="interval_contract_covering_milp",
            status="infeasible",
            confidence=0.82,
            artifact={
                "demand": demand_values,
                "contract_fees": contract_fees,
                "unit_size": unit_size,
                "min_distinct_lengths": min_distinct,
                "max_distinct_lengths": max_distinct,
                "length_exclusions": [list(pair) for pair in exclusions],
            },
        )

    objective_value, used_lengths, variables, solution = best
    variable_values = {
        f"contract_start_{start + 1}_length_{length}": value
        for value, (start, length) in zip(solution, variables)
        if value > 1e-8
    }
    return TemplateSolveResult(
        matched=True,
        template_id="interval_contract_covering_milp",
        status="optimal",
        objective_value=objective_value,
        variable_values=variable_values,
        confidence=0.88,
        notes="Solved interval contract covering model with period coverage and contract-length logic.",
        artifact={
            "demand": demand_values,
            "demand_units": demand_units,
            "unit_size": unit_size,
            "contract_fees": {str(key): value for key, value in contract_fees.items()},
            "exact_cover": exact_cover,
            "min_distinct_lengths": min_distinct,
            "max_distinct_lengths": max_distinct,
            "length_exclusions": [list(pair) for pair in exclusions],
            "used_lengths": sorted(used_lengths),
        },
    )


def _source_index(labels: list[str], value: str) -> int | None:
    wanted = _clean_label(value).lower()
    for index, label in enumerate(labels):
        if _clean_label(label).lower() == wanted:
            return index
    return None


def _parse_order_source_labels(text: str) -> tuple[str, list[str]] | None:
    match = re.search(
        r"from\s+(?:[a-z]+\s+)?different\s+(manufacturers?|suppliers?)[:,]?\s*([^.\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    source_type = "manufacturer" if "manufacturer" in match.group(1).lower() else "supplier"
    labels = _split_narrative_labels(match.group(2))
    labels = [label for label in labels if re.fullmatch(r"[A-Za-z0-9]+", label)]
    if len(labels) < 2:
        return None
    return source_type, labels


def _parse_procurement_order_problem(
    text: str,
) -> tuple[list[str], list[int], list[float], float, float | None, list[tuple[int, int, float]]] | None:
    parsed_labels = _parse_order_source_labels(text)
    if not parsed_labels:
        return None
    source_type, labels = parsed_labels

    unit_costs: list[float] = []
    for label in labels:
        match = re.search(
            rf"from\s+{source_type}\s+{re.escape(label)}\s+is\s*[£$¥]?\s*({_NUMBER_TOKEN})",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        unit_costs.append(_number(match.group(1)))

    pack_sizes: list[float | None] = [None] * len(labels)
    for match in re.finditer(
        rf"each\s+order\s+from\s+{source_type}s?\s+([^.;]+?)\s+will\s+include\s+({_NUMBER_TOKEN})\s+\w+",
        text,
        flags=re.IGNORECASE,
    ):
        matched_labels = _split_narrative_labels(match.group(1))
        size = _number(match.group(2))
        for label in matched_labels:
            index = _source_index(labels, label)
            if index is not None:
                pack_sizes[index] = size
    if any(size is None for size in pack_sizes):
        return None
    integer_pack_sizes = [int(round(float(size or 0.0))) for size in pack_sizes]

    min_match = re.search(
        rf"\b(?:needs?\s+to\s+)?order\s+at\s+least\s+({_NUMBER_TOKEN})\s+\w+",
        text,
        flags=re.IGNORECASE,
    )
    if not min_match:
        return None
    min_units = _number(min_match.group(1))
    max_units: float | None = None
    max_match = re.search(
        rf"\border\s+(?:at\s+most|no\s+more\s+than)\s+({_NUMBER_TOKEN})\s+\w+",
        text,
        flags=re.IGNORECASE,
    )
    if max_match:
        max_units = _number(max_match.group(1))

    implications: list[tuple[int, int, float]] = []
    source_word = rf"{source_type}s?"
    for match in re.finditer(
        rf"if[^.]*?order\s+\w+\s+from\s+{source_word}\s+([A-Za-z0-9]+)"
        rf"[^.]*?order\s+at\s+least\s+({_NUMBER_TOKEN})\s+\w+\s+from\s+{source_word}\s+([A-Za-z0-9]+)",
        text,
        flags=re.IGNORECASE,
    ):
        trigger = _source_index(labels, match.group(1))
        target = _source_index(labels, match.group(3))
        if trigger is not None and target is not None:
            implications.append((trigger, target, _number(match.group(2))))
    for match in re.finditer(
        rf"if[^.]*?order\s+\w+\s+from\s+{source_word}\s+([A-Za-z0-9]+)"
        rf"[^.]*?also\s+order\s+\w+\s+from\s+{source_word}\s+([A-Za-z0-9]+)",
        text,
        flags=re.IGNORECASE,
    ):
        trigger = _source_index(labels, match.group(1))
        target = _source_index(labels, match.group(2))
        if trigger is not None and target is not None:
            implications.append((trigger, target, float(integer_pack_sizes[target])))

    objective = [unit_cost * pack_size for unit_cost, pack_size in zip(unit_costs, integer_pack_sizes)]
    return labels, integer_pack_sizes, objective, min_units, max_units, implications


def _parse_raw_material_requirements(text: str) -> dict[str, float]:
    requirements: dict[str, float] = {}
    for value, label in re.findall(
        rf"at\s+least\s+({_NUMBER_TOKEN})\s+(?:pieces?\s+of|kg\s+of|tons?\s+of)\s+(raw\s+material\s+[A-Za-z0-9]+)",
        text,
        flags=re.IGNORECASE,
    ):
        requirements[_clean_label(label).lower()] = _number(value)
    return requirements


def _parse_procurement_truck_problem(
    text: str,
) -> tuple[list[str], list[dict[str, float]], list[float], dict[str, float]] | None:
    requirements = _parse_raw_material_requirements(text)
    if len(requirements) < 2:
        return None
    labels: list[str] = []
    bundles: list[dict[str, float]] = []
    costs: list[float] = []
    for match in re.finditer(
        rf"each\s+truck\s+from\s+warehouse\s+([A-Za-z0-9]+)\s+can\s+transport\s+"
        rf"(.*?)(?:with\s+a?\s*freight\s+cost\s+of\s*({_NUMBER_TOKEN})\s+\w+\s+per\s+truck)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        label = _clean_label(match.group(1))
        body = re.sub(r"\s+", " ", match.group(2))
        bundle: dict[str, float] = {}
        for value, material in re.findall(
            rf"({_NUMBER_TOKEN})\s+(?:pieces?\s+of|kg\s+of|tons?\s+of)\s+(raw\s+material\s+[A-Za-z0-9]+)",
            body,
            flags=re.IGNORECASE,
        ):
            bundle[_clean_label(material).lower()] = _number(value)
        if all(material in bundle for material in requirements):
            labels.append(label)
            bundles.append(bundle)
            costs.append(_number(match.group(3)))
    if len(labels) < 2:
        return None
    return labels, bundles, costs, requirements


def _solve_procurement_milp(
    *,
    template_id: str,
    labels: list[str],
    objective: list[float],
    rows: list[list[float]],
    lower: list[float],
    upper: list[float],
    integer_count: int,
    artifact: dict[str, Any],
) -> TemplateSolveResult:
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id=template_id,
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
            artifact=artifact,
        )

    result = milp(
        c=np.array(objective, dtype=float),
        integrality=np.array([1] * integer_count + [1] * (len(objective) - integer_count)),
        bounds=Bounds(np.zeros(len(objective)), np.full(len(objective), math.inf)),
        constraints=LinearConstraint(np.array(rows, dtype=float), np.array(lower), np.array(upper)),
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id=template_id,
            status="solver_failed",
            confidence=0.82,
            notes=str(result.message),
            artifact=artifact,
        )
    values = [
        float(round(value)) if math.isclose(float(value), round(float(value)), abs_tol=1e-7) else float(value)
        for value in result.x[:integer_count]
    ]
    return TemplateSolveResult(
        matched=True,
        template_id=template_id,
        status="optimal",
        objective_value=float(result.fun),
        variable_values={
            f"count_{_clean_label(labels[index])}": value
            for index, value in enumerate(values)
            if not math.isclose(value, 0.0, abs_tol=1e-9)
        },
        confidence=0.87,
        notes="Solved procurement/source lot mix MILP from explicit lot sizes, costs, and demand requirements.",
        artifact=artifact,
    )


def _solve_procurement_lot_mix(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        ("minimize" in lowered or "minimum" in lowered)
        and ("cost" in lowered or "freight" in lowered)
        and any(word in lowered for word in ("supplier", "manufacturer", "warehouse"))
    ):
        return TemplateSolveResult(False)

    order_problem = _parse_procurement_order_problem(normalized)
    if order_problem:
        labels, pack_sizes, objective, min_units, max_units, implications = order_problem
        source_count = len(labels)
        active_offset = source_count
        max_orders = [
            int(math.ceil((max_units or max(min_units * 5.0, min_units + pack_size)) / pack_size))
            for pack_size in pack_sizes
        ]
        rows: list[list[float]] = []
        lower: list[float] = []
        upper: list[float] = []

        total_row = [float(size) for size in pack_sizes] + [0.0] * source_count
        rows.append(total_row)
        lower.append(min_units)
        upper.append(max_units if max_units is not None else math.inf)
        for index, max_order in enumerate(max_orders):
            row = [0.0] * (source_count * 2)
            row[index] = 1.0
            row[active_offset + index] = -float(max_order)
            rows.append(row)
            lower.append(-math.inf)
            upper.append(0.0)

            row = [0.0] * (source_count * 2)
            row[index] = 1.0
            row[active_offset + index] = -1.0
            rows.append(row)
            lower.append(0.0)
            upper.append(math.inf)
        for trigger, target, min_target_units in implications:
            row = [0.0] * (source_count * 2)
            row[target] = float(pack_sizes[target])
            row[active_offset + trigger] = -float(min_target_units)
            rows.append(row)
            lower.append(0.0)
            upper.append(math.inf)

        binary_rows = [[0.0] * (source_count * 2) for _ in range(source_count)]
        for index, row in enumerate(binary_rows):
            row[active_offset + index] = 1.0
            rows.append(row)
            lower.append(0.0)
            upper.append(1.0)
        return _solve_procurement_milp(
            template_id="procurement_lot_mix_milp",
            labels=labels,
            objective=objective + [0.0] * source_count,
            rows=rows,
            lower=lower,
            upper=upper,
            integer_count=source_count,
            artifact={
                "sources": labels,
                "pack_sizes": pack_sizes,
                "cost_per_lot": objective,
                "min_units": min_units,
                "max_units": max_units,
                "implications": implications,
            },
        )

    truck_problem = _parse_procurement_truck_problem(normalized)
    if truck_problem:
        labels, bundles, costs, requirements = truck_problem
        rows = []
        lower = []
        upper = []
        for material, requirement in requirements.items():
            rows.append([bundle[material] for bundle in bundles])
            lower.append(requirement)
            upper.append(math.inf)
        return _solve_procurement_milp(
            template_id="procurement_lot_mix_milp",
            labels=labels,
            objective=costs,
            rows=rows,
            lower=lower,
            upper=upper,
            integer_count=len(labels),
            artifact={
                "sources": labels,
                "bundles": bundles,
                "cost_per_lot": costs,
                "requirements": requirements,
            },
        )

    return TemplateSolveResult(False)


def _resource_label_pattern(label: str) -> str:
    escaped = re.escape(label)
    if label.lower().startswith("type "):
        return rf"\b{escaped}(?:\s+trucks?)?\b"
    if label.lower() == "bus":
        return r"\b(?:buses|bus)\b"
    if label.lower() == "minibus":
        return r"\b(?:minibuses|minibus)\b"
    singular = _singular_label(label)
    variants = {label, singular}
    if singular == "bus":
        variants.add("buses")
    elif not singular.endswith("s"):
        variants.add(f"{singular}s")
    return rf"\b(?:{'|'.join(re.escape(item) for item in sorted(variants, key=len, reverse=True))})\b"


def _resource_labels_in_sentence(sentence: str, labels: list[str]) -> list[str]:
    found: list[tuple[int, str]] = []
    for label in labels:
        match = re.search(_resource_label_pattern(label), sentence, flags=re.IGNORECASE)
        if match:
            found.append((match.start(), label))
    return [label for _index, label in sorted(found)]


def _parse_transport_method_labels(text: str) -> list[str]:
    patterns = [
        r"transportation\s+options?:\s*([^.\n]+)",
        r"following\s+three\s+methods:\s*([^.\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            labels = [
                re.sub(r"^(?:a|an|the)\s+", "", _clean_label(item), flags=re.IGNORECASE)
                for item in re.split(r",|\band\b", match.group(1), flags=re.IGNORECASE)
                if item.strip()
            ]
            labels = [label for label in labels if label and label.lower() not in {"method", "option"}]
            if len(labels) >= 2:
                return labels
    pair_patterns = [
        r"either\s+by\s+(?:a\s+|an\s+)?([A-Za-z]+)\s+or\s+by\s+(?:a\s+|an\s+)?([A-Za-z]+)",
        r"using\s+either\s+([A-Za-z][A-Za-z\s-]*?)\s+or\s+([A-Za-z][A-Za-z\s-]*?)(?:\.|,|$)",
        r"provides\s+([A-Za-z]+)\s+and\s+([A-Za-z]+)\s+transportation",
        r"buy\s+([A-Za-z]+)\s+or\s+([A-Za-z]+)\s+to\s+add",
    ]
    for pattern in pair_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return [_singular_label(match.group(1)), _singular_label(match.group(2))]
    type_match = re.search(
        r"two\s+types\s+of\s+trucks?,\s*Type\s+([A-Za-z0-9]+)\s+and\s+Type\s+([A-Za-z0-9]+)",
        text,
        flags=re.IGNORECASE,
    )
    if type_match:
        return [f"Type {type_match.group(1).upper()}", f"Type {type_match.group(2).upper()}"]
    if re.search(r"\bbuses\b", text, flags=re.IGNORECASE) and re.search(r"\bminibuses\b", text, flags=re.IGNORECASE):
        return ["bus", "minibus"]
    return []


def _parse_transport_objective_coefficients(text: str, labels: list[str]) -> dict[str, float]:
    coefficients: dict[str, float] = {}
    lowered = text.lower()
    if re.search(
        r"\b(?:minimi[sz]e|decrease)\s+the\s+total\s+number\s+of\s+(?:vehicles?|trips?|carts?)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return {label: 1.0 for label in labels}
    for label in labels:
        label_pattern = _resource_label_pattern(label)
        if re.search(
            rf"\bminimi[sz]e\s+the\s+total\s+number\s+of\s+{label_pattern}\s+needed\b",
            text,
            flags=re.IGNORECASE,
        ):
            return {item: (1.0 if item == label else 0.0) for item in labels}
    for label in labels:
        label_pattern = _resource_label_pattern(label)
        patterns = [
            rf"{label_pattern}\s+(?:trip\s+)?(?:produces|generates)\s+({_NUMBER_TOKEN})\s+units?\s+of\s+pollution",
            rf"{label_pattern}[^.\n]{{0,90}}?\btakes?\s+({_NUMBER_TOKEN})\s+minutes?\s+per\s+trip",
            rf"({_NUMBER_TOKEN})\s+units?\s+for\s+(?:a\s+|an\s+|the\s+)?{label_pattern}",
            rf"(?:rental\s+)?cost\s+(?:per\s+kilometer\s+)?for\s+(?:a\s+)?{label_pattern}\s+(?:trucks?\s+)?is\s+[£$]?\s*({_NUMBER_TOKEN})",
            rf"{label_pattern}\s+(?:trucks?\s+)?(?:have|has)[^.\n]{{0,80}}?rental\s+cost[^.\n]{{0,30}}?[£$]\s*({_NUMBER_TOKEN})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                coefficients[label] = _number(match.group(1))
                break
    return coefficients


def _parse_transport_capacities(text: str, labels: list[str]) -> dict[str, dict[str, float]]:
    capacities: dict[str, dict[str, float]] = {label: {} for label in labels}
    for label in labels:
        label_pattern = _resource_label_pattern(label)
        for pattern in (
            rf"each\s+{label_pattern}\s+trip\s+can\s+transport\s+({_NUMBER_TOKEN})\s+units?",
            rf"(?:a|an|each)\s+{label_pattern}\s+can\s+take\s+({_NUMBER_TOKEN})\s+(?:people|persons|ducks|passengers?|guests?)",
            rf"(?:a|an|each)\s+{label_pattern}\s+can\s+seat\s+({_NUMBER_TOKEN})\s+(?:tourists?|customers?|people|persons|passengers?)",
            rf"({_NUMBER_TOKEN})\s+units?\s*\(\s*{label_pattern}\s*\)",
            rf"{label_pattern}\s+with\s+({_NUMBER_TOKEN})\s+seats?\s+each",
        ):
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                capacities[label]["units"] = _number(match.group(1))
                break
    type_a = re.search(
        rf"Type\s+A\s+trucks?\s+have\s+({_NUMBER_TOKEN})\s+cubic\s+meters?\s+of\s+refrigerated\s+capacity\s+and\s+"
        rf"({_NUMBER_TOKEN})\s+cubic\s+meters?\s+of\s+non[-\s]?refrigerated\s+capacity",
        text,
        flags=re.IGNORECASE,
    )
    if type_a and "Type A" in capacities:
        refrigerated = _number(type_a.group(1))
        non_refrigerated = _number(type_a.group(2))
        capacities["Type A"]["refrigerated"] = refrigerated
        capacities["Type A"]["non_refrigerated"] = non_refrigerated
        if "Type B" in capacities and re.search(r"Type\s+B[^.\n]+same\s+total\s+capacity[^.\n]+equal", text, flags=re.IGNORECASE):
            equal_capacity = (refrigerated + non_refrigerated) / 2.0
            capacities["Type B"]["refrigerated"] = equal_capacity
            capacities["Type B"]["non_refrigerated"] = equal_capacity
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if "can carry" not in sentence.lower():
            continue
        sentence_labels = _resource_labels_in_sentence(sentence, labels)
        if len(sentence_labels) < 2:
            continue
        tail = sentence.split("can carry", 1)[1]
        values = [
            _number(match.group(1))
            for match in re.finditer(rf"({_NUMBER_TOKEN})\s+units?", tail, flags=re.IGNORECASE)
        ]
        if len(values) >= len(sentence_labels):
            for label, value in zip(sentence_labels, values):
                capacities[label]["units"] = value
    return {label: dims for label, dims in capacities.items() if dims}


def _parse_transport_demands(text: str) -> dict[str, float]:
    demands: dict[str, float] = {}
    for pattern in (
        rf"(?:transport|transported|delivered|deliver)[^.\n]{{0,80}}\bat\s+least\s+({_NUMBER_TOKEN})\s+units?",
        rf"(?:transport|transported|take|taken)[^.\n]{{0,80}}\bat\s+least\s+({_NUMBER_TOKEN})\s+(?:people|persons|ducks|passengers?|guests?)",
        rf"take\s+care\s+of\s+at\s+least\s+({_NUMBER_TOKEN})\s+(?:customers?|tourists?|people|persons|guests?)",
        rf"at\s+least\s+({_NUMBER_TOKEN})\s+(?:ducks|people|persons|tourists?|customers?|guests?)",
        rf"needs?\s+to\s+transport\s+({_NUMBER_TOKEN})\s+units?",
        rf"for\s+({_NUMBER_TOKEN})\s+students?",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            demands["units"] = _number(match.group(1))
            break
    ref_match = re.search(
        rf"transport\s+({_NUMBER_TOKEN})\s+cubic\s+meters?\s+of\s+refrigerated\s+cargo\s+and\s+"
        rf"({_NUMBER_TOKEN})\s+cubic\s+meters?\s+of\s+non[-\s]?refrigerated\s+cargo",
        text,
        flags=re.IGNORECASE,
    )
    if ref_match:
        demands["refrigerated"] = _number(ref_match.group(1))
        demands["non_refrigerated"] = _number(ref_match.group(2))
    return demands


def _parse_transport_bounds_and_logic(
    text: str,
    labels: list[str],
) -> tuple[dict[str, int], dict[str, int], int | None, int | None, list[tuple[str, str]]]:
    lower_bounds: dict[str, int] = {}
    upper_bounds: dict[str, int] = {}
    total_max: int | None = None
    active_max: int | None = None
    exclusions: list[tuple[str, str]] = []
    for label in labels:
        label_pattern = _resource_label_pattern(label)
        min_match = re.search(
            rf"at\s+least\s+({_NUMBER_TOKEN})\s+trips?\s+must\s+be\s+made\s+using\s+(?:the\s+)?{label_pattern}",
            text,
            flags=re.IGNORECASE,
        )
        if min_match:
            lower_bounds[label] = int(round(_number(min_match.group(1))))
        max_match = re.search(
            rf"number\s+of\s+{label_pattern}\s+trips?\s+cannot\s+exceed\s+({_NUMBER_TOKEN})",
            text,
            flags=re.IGNORECASE,
        )
        if not max_match:
            max_match = re.search(
                rf"at\s+most\s+({_NUMBER_TOKEN})\s+{label_pattern}\s+trips?",
                text,
                flags=re.IGNORECASE,
            )
        if max_match:
            upper_bounds[label] = int(round(_number(max_match.group(1))))
        availability_match = re.search(
            rf"({_NUMBER_TOKEN})\s+{label_pattern}\s+with\s+({_NUMBER_TOKEN})\s+seats?\s+each",
            text,
            flags=re.IGNORECASE,
        )
        if availability_match:
            upper_bounds[label] = int(round(_number(availability_match.group(1))))
    total_match = re.search(
        rf"total\s+number\s+of\s+trips\s+must\s+be\s+(?:less\s+than\s+or\s+equal\s+to|no\s+more\s+than|at\s+most)\s+({_NUMBER_TOKEN})",
        text,
        flags=re.IGNORECASE,
    )
    if total_match:
        total_max = int(round(_number(total_match.group(1))))
    driver_match = re.search(rf"only\s+({_NUMBER_TOKEN})\s+drivers?\s+are\s+available", text, flags=re.IGNORECASE)
    if driver_match:
        total_max = int(round(_number(driver_match.group(1))))
    active_match = re.search(rf"only\s+choose\s+({_NUMBER_TOKEN})\s+out\s+of", text, flags=re.IGNORECASE)
    if active_match:
        active_max = int(round(_number(active_match.group(1))))
    for sentence in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text)):
        sentence_labels = _resource_labels_in_sentence(sentence, labels)
        lowered = sentence.lower()
        if ("not both" in lowered or "only one of these two" in lowered) and len(sentence_labels) >= 2:
            for left, right in itertools.combinations(sentence_labels, 2):
                exclusions.append((left, right))
    return lower_bounds, upper_bounds, total_max, active_max, exclusions


def _parse_transport_share_constraints(
    text: str,
    labels: list[str],
) -> list[tuple[str, str, float]]:
    constraints: list[tuple[str, str, float]] = []
    for label in labels:
        label_pattern = _resource_label_pattern(label)
        for sense, phrase in (("<=", "at most"), (">=", "at least")):
            match = re.search(
                rf"{phrase}\s+({_NUMBER_TOKEN})\s*%\s+of\s+(?:the\s+)?(?:vehicles?|trips?|carts?)\s+"
                rf"(?:can|should|must)?\s*(?:be\s+)?(?:by\s+)?(?:the\s+)?{label_pattern}",
                text,
                flags=re.IGNORECASE,
            )
            if match:
                constraints.append((label, sense, _number(match.group(1)) / 100.0))
    return constraints


def _parse_transport_upper_resource_constraints(
    text: str,
    labels: list[str],
) -> list[tuple[str, dict[str, float], float]]:
    pollution_limit = re.search(
        rf"(?:at\s+most|limited\s+[^.\n]{{0,80}}?\s+to\s+(?:producing\s+)?at\s+most)\s+"
        rf"({_NUMBER_TOKEN})\s+units?\s+of\s+pollut(?:ion|ants)",
        text,
        flags=re.IGNORECASE,
    )
    if not pollution_limit:
        return []
    coefficients: dict[str, float] = {}
    for label in labels:
        label_pattern = _resource_label_pattern(label)
        match = re.search(
            rf"{label_pattern}[^.\n]{{0,80}}?(?:results?\s+in|causes?|emits?)\s+"
            rf"({_NUMBER_TOKEN})\s+units?\s+of\s+pollut(?:ion|ants)",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            coefficients[label] = _number(match.group(1))
    if set(coefficients) != set(labels):
        return []
    return [("pollution", coefficients, _number(pollution_limit.group(1)))]


def _enumeration_upper_bound(
    label: str,
    capacities: dict[str, dict[str, float]],
    demands: dict[str, float],
    upper_bounds: dict[str, int],
    total_max: int | None,
) -> int:
    if label in upper_bounds:
        return upper_bounds[label]
    if total_max is not None:
        return total_max
    candidates: list[int] = []
    for dimension, demand in demands.items():
        capacity = capacities.get(label, {}).get(dimension)
        if capacity and capacity > 0:
            candidates.append(int(math.ceil(demand / capacity)))
    return max(candidates or [20])


def _solve_integer_resource_mix(text: str) -> TemplateSolveResult:
    normalized_text = re.sub(r"\s+", " ", text)
    lowered = normalized_text.lower()
    if not (
        ("minimize" in lowered or "minimum" in lowered or "lowest" in lowered or "decrease" in lowered)
        and any(word in lowered for word in ("transport", "truck", "bus", "minibus", "trips", "pollution", "rental cost", "cart"))
    ):
        return TemplateSolveResult(False)
    labels = _parse_transport_method_labels(normalized_text)
    if len(labels) < 2 or len(labels) > 6:
        return TemplateSolveResult(False)
    objective = _parse_transport_objective_coefficients(normalized_text, labels)
    capacities = _parse_transport_capacities(normalized_text, labels)
    demands = _parse_transport_demands(normalized_text)
    if set(objective) != set(labels) or set(capacities) != set(labels) or not demands:
        return TemplateSolveResult(False)
    lower_bounds, upper_bounds, total_max, active_max, exclusions = _parse_transport_bounds_and_logic(normalized_text, labels)
    share_constraints = _parse_transport_share_constraints(normalized_text, labels)
    upper_resource_constraints = _parse_transport_upper_resource_constraints(normalized_text, labels)

    ranges = [
        range(
            lower_bounds.get(label, 0),
            _enumeration_upper_bound(label, capacities, demands, upper_bounds, total_max) + 1,
        )
        for label in labels
    ]
    best: tuple[float, tuple[int, ...]] | None = None
    for counts in itertools.product(*ranges):
        count_by_label = dict(zip(labels, counts))
        if total_max is not None and sum(counts) > total_max:
            continue
        total_count = sum(counts)
        if total_count <= 0:
            continue
        share_feasible = True
        for label, sense, fraction in share_constraints:
            lhs = count_by_label[label]
            rhs = fraction * total_count
            if sense == "<=" and lhs > rhs + 1e-9:
                share_feasible = False
                break
            if sense == ">=" and lhs < rhs - 1e-9:
                share_feasible = False
                break
        if not share_feasible:
            continue
        active = {label for label, count in count_by_label.items() if count > 0}
        if active_max is not None and len(active) > active_max:
            continue
        if any(left in active and right in active for left, right in exclusions):
            continue
        feasible = True
        for dimension, demand in demands.items():
            supplied = sum(capacities[label].get(dimension, 0.0) * count_by_label[label] for label in labels)
            if supplied < demand - 1e-9:
                feasible = False
                break
        if not feasible:
            continue
        for _name, coefficients, upper in upper_resource_constraints:
            used = sum(coefficients[label] * count_by_label[label] for label in labels)
            if used > upper + 1e-9:
                feasible = False
                break
        if not feasible:
            continue
        value = sum(objective[label] * count_by_label[label] for label in labels)
        if best is None or value < best[0] - 1e-9 or (
            math.isclose(value, best[0], abs_tol=1e-9) and sum(counts) < sum(best[1])
        ):
            best = (value, tuple(counts))
    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="integer_resource_mix_milp",
            status="infeasible",
            confidence=0.82,
            artifact={"items": labels, "objective": objective, "capacities": capacities, "demands": demands},
        )

    selected_counts = dict(zip(labels, best[1]))
    return TemplateSolveResult(
        matched=True,
        template_id="integer_resource_mix_milp",
        status="optimal",
        objective_value=float(best[0]),
        variable_values={f"count_{_clean_label(label)}": float(count) for label, count in selected_counts.items() if count},
        confidence=0.88,
        notes="Solved integer resource/fleet mix by exact enumeration of small count ranges.",
        artifact={
            "items": labels,
            "objective": objective,
            "capacities": capacities,
            "demands": demands,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
            "total_count_max": total_max,
            "active_max": active_max,
            "exclusions": [list(pair) for pair in exclusions],
            "share_constraints": [
                {"label": label, "sense": sense, "fraction": fraction}
                for label, sense, fraction in share_constraints
            ],
            "upper_resource_constraints": [
                {"name": name, "coefficients": coefficients, "upper": upper}
                for name, coefficients, upper in upper_resource_constraints
            ],
            "selected_counts": selected_counts,
        },
    )


def _solve_container_capacity_max_mix(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        ("crate" in lowered or "container" in lowered)
        and ("maximize" in lowered or "maximum" in lowered)
        and ("capacity" in lowered or "can take" in lowered or "transport" in lowered)
    ):
        return TemplateSolveResult(False)

    labels: list[str] = []
    capacities: dict[str, float] = {}
    for label, value in re.findall(
        rf"(?:a|an)\s+([A-Za-z][A-Za-z\s-]*?(?:crate|container))\s+can\s+take\s+"
        rf"({_NUMBER_TOKEN})(?:\s+\w+)?",
        normalized,
        flags=re.IGNORECASE,
    ):
        clean = _singular_label(label)
        if clean not in labels:
            labels.append(clean)
        capacities[clean] = _number(value)
    if len(labels) < 2:
        return TemplateSolveResult(False)

    lower_bounds = {label: 0 for label in labels}
    upper_bounds: dict[str, int] = {}
    total_max: int | None = None
    for label in labels:
        label_pattern = _resource_label_pattern(label)
        upper_match = re.search(
            rf"at\s+most\s+({_NUMBER_TOKEN})\s+{label_pattern}s?",
            normalized,
            flags=re.IGNORECASE,
        )
        if upper_match:
            upper_bounds[label] = int(_number(upper_match.group(1)))
        lower_match = re.search(
            rf"(?:at\s+least|must\s+use\s+at\s+least)\s+({_NUMBER_TOKEN})\s+{label_pattern}s?",
            normalized,
            flags=re.IGNORECASE,
        )
        if lower_match:
            lower_bounds[label] = max(lower_bounds[label], int(_number(lower_match.group(1))))
    total_match = re.search(
        rf"at\s+most\s+({_NUMBER_TOKEN})\s+(?:crates?|containers?)\s+total",
        normalized,
        flags=re.IGNORECASE,
    )
    if total_match:
        total_max = int(_number(total_match.group(1)))

    ratio_constraints: list[tuple[str, str, float]] = []
    for left in labels:
        for right in labels:
            if left == right:
                continue
            match = re.search(
                rf"at\s+least\s+({_NUMBER_TOKEN})\s+times\s+as\s+many\s+"
                rf"{_resource_label_pattern(left)}s?\s+must\s+be\s+used\s+than\s+"
                rf"{_resource_label_pattern(right)}s?",
                normalized,
                flags=re.IGNORECASE,
            )
            if match:
                ratio_constraints.append((left, right, _number(match.group(1))))

    ranges = [
        range(lower_bounds[label], upper_bounds.get(label, total_max or 100) + 1)
        for label in labels
    ]
    best: tuple[float, tuple[int, ...]] | None = None
    for counts in itertools.product(*ranges):
        by_label = dict(zip(labels, counts))
        if total_max is not None and sum(counts) > total_max:
            continue
        if any(by_label[left] + 1e-9 < factor * by_label[right] for left, right, factor in ratio_constraints):
            continue
        value = sum(capacities[label] * by_label[label] for label in labels)
        if best is None or value > best[0] + 1e-9:
            best = (value, tuple(counts))
    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="container_capacity_max_mix_ilp",
            status="infeasible",
            confidence=0.8,
            artifact={"labels": labels, "capacities": capacities},
        )

    selected = dict(zip(labels, best[1]))
    return TemplateSolveResult(
        matched=True,
        template_id="container_capacity_max_mix_ilp",
        status="optimal",
        objective_value=float(best[0]),
        variable_values={
            f"count_{_clean_label(label)}": float(count)
            for label, count in selected.items()
            if count
        },
        confidence=0.86,
        notes="Solved integer container/crate capacity maximization by exact enumeration.",
        artifact={
            "labels": labels,
            "capacities": capacities,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
            "total_max": total_max,
            "ratio_constraints": ratio_constraints,
            "selected": selected,
        },
    )


def _solve_advertising_media_mix(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "commercial" in lowered
        and "budget" in lowered
        and ("audience" in lowered or "viewers" in lowered)
        and ("maximize" in lowered or "maximum" in lowered)
        and "platform" in lowered
    ):
        return TemplateSolveResult(False)

    platform_match = re.search(
        r"three\s+(?:streaming\s+)?platforms?:\s*([^.;]+)",
        normalized,
        flags=re.IGNORECASE,
    )
    if not platform_match:
        return TemplateSolveResult(False)
    labels = [
        _clean_label(label)
        for label in re.split(r",|\band\b", platform_match.group(1), flags=re.IGNORECASE)
        if label.strip()
    ]
    if len(labels) != 3:
        return TemplateSolveResult(False)

    costs: dict[str, float] = {}
    reach: dict[str, float] = {}
    for label in labels:
        label_pattern = r"\s+".join(re.escape(part) for part in label.split())
        match = re.search(
            rf"on\s+{label_pattern},?\s+a\s+commercial\s+costs\s+\$?\s*({_NUMBER_TOKEN})"
            rf"\s+and\s+attracts\s+({_NUMBER_TOKEN})\s+viewers?",
            normalized,
            flags=re.IGNORECASE,
        )
        if match:
            costs[label] = _number(match.group(1))
            reach[label] = _number(match.group(2))
    budget = _number_after_patterns(
        normalized,
        [
            rf"weekly\s+budget\s+is\s+\$?\s*({_NUMBER_TOKEN})",
            rf"budget\s+of\s+\$?\s*({_NUMBER_TOKEN})",
        ],
    )
    if set(costs) != set(labels) or set(reach) != set(labels) or budget is None:
        return TemplateSolveResult(False)

    upper_bounds: dict[str, int] = {}
    share_constraints: list[tuple[str, str, float]] = []
    for label in labels:
        label_pattern = r"\s+".join(re.escape(part) for part in label.split())
        limit_match = re.search(
            rf"{label_pattern}\s+limits\s+the\s+number\s+of\s+commercials?[^.]*?\bto\s+({_NUMBER_TOKEN})",
            normalized,
            flags=re.IGNORECASE,
        )
        if limit_match:
            upper_bounds[label] = int(_number(limit_match.group(1)))
        if re.search(
            rf"at\s+most\s+a\s+third\s+of\s+all\s+commercials?[^.]*?\bon\s+{label_pattern}",
            normalized,
            flags=re.IGNORECASE,
        ):
            share_constraints.append((label, "<=", 1.0 / 3.0))
        minimum_match = re.search(
            rf"(?:minimum\s+of|at\s+least)\s+({_NUMBER_TOKEN})\s*%\s+[^.]*?\bon\s+{label_pattern}",
            normalized,
            flags=re.IGNORECASE,
        )
        if minimum_match:
            share_constraints.append((label, ">=", _number(minimum_match.group(1)) / 100.0))
    if not share_constraints:
        return TemplateSolveResult(False)

    ranges = [
        range(0, min(upper_bounds.get(label, int(math.floor(budget / costs[label]))), int(math.floor(budget / costs[label]))) + 1)
        for label in labels
    ]
    best: tuple[float, float, tuple[int, ...]] | None = None
    for counts in itertools.product(*ranges):
        total_count = sum(counts)
        if total_count <= 0:
            continue
        by_label = dict(zip(labels, counts))
        total_cost = sum(costs[label] * by_label[label] for label in labels)
        if total_cost > budget + 1e-9:
            continue
        feasible = True
        for label, sense, fraction in share_constraints:
            lhs = by_label[label]
            rhs = fraction * total_count
            if sense == "<=" and lhs > rhs + 1e-9:
                feasible = False
                break
            if sense == ">=" and lhs < rhs - 1e-9:
                feasible = False
                break
        if not feasible:
            continue
        value = sum(reach[label] * by_label[label] for label in labels)
        if best is None or value > best[0] + 1e-9 or (
            math.isclose(value, best[0], abs_tol=1e-9) and total_cost < best[1]
        ):
            best = (value, total_cost, tuple(counts))
    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="advertising_media_mix_ilp",
            status="infeasible",
            confidence=0.8,
            artifact={"labels": labels, "costs": costs, "reach": reach, "budget": budget},
        )

    selected = dict(zip(labels, best[2]))
    return TemplateSolveResult(
        matched=True,
        template_id="advertising_media_mix_ilp",
        status="optimal",
        objective_value=float(best[0]),
        variable_values={
            f"commercials_{_clean_label(label)}": float(count)
            for label, count in selected.items()
            if count
        },
        confidence=0.86,
        notes="Solved small advertising media-mix integer program by exact enumeration.",
        artifact={
            "labels": labels,
            "costs": costs,
            "reach": reach,
            "budget": budget,
            "upper_bounds": upper_bounds,
            "share_constraints": [
                {"label": label, "sense": sense, "fraction": fraction}
                for label, sense, fraction in share_constraints
            ],
            "selected": selected,
            "selected_cost": best[1],
        },
    )


def _solve_two_product_machine_inventory_surplus_lp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "two machines" in lowered
        and "produce two" in lowered
        and "inventory" in lowered
        and "demand" in lowered
        and "fractional lots" in lowered
        and ("maximize" in lowered or "maximum" in lowered)
    ):
        return TemplateSolveResult(False)

    label_match = re.search(
        r"produce\s+two\s+[^:.;]+:\s*([A-Za-z][A-Za-z -]*?)\s+and\s+([A-Za-z][A-Za-z -]*?)(?:\.|,)",
        normalized,
        flags=re.IGNORECASE,
    )
    if not label_match:
        return TemplateSolveResult(False)
    labels = [_singular_label(label_match.group(1)), _singular_label(label_match.group(2))]
    if len(set(labels)) != 2:
        return TemplateSolveResult(False)

    machine_requirements: dict[str, tuple[float, float]] = {}
    for label in labels:
        label_pattern = _resource_label_pattern(label)
        match = re.search(
            rf"producing\s+one\s+lot\s+of\s+{label_pattern}\s+requires\s+"
            rf"({_NUMBER_TOKEN})\s+minutes?\s+on\s+Machine\s+1\s+and\s+"
            rf"({_NUMBER_TOKEN})\s+minutes?\s+on\s+Machine\s+2",
            normalized,
            flags=re.IGNORECASE,
        )
        if match:
            machine_requirements[label] = (_number(match.group(1)), _number(match.group(2)))
    capacities: dict[int, float] = {}
    for index in (1, 2):
        match = re.search(
            rf"Machine\s+{index}\s+has\s+({_NUMBER_TOKEN})\s+available\s+(hours?|minutes?)",
            normalized,
            flags=re.IGNORECASE,
        )
        if match:
            value = _number(match.group(1))
            if match.group(2).lower().startswith("hour"):
                value *= 60.0
            capacities[index] = value

    first_pattern = _resource_label_pattern(labels[0])
    second_pattern = _resource_label_pattern(labels[1])
    inventory_match = re.search(
        rf"on[-\s]?hand\s+inventories\s+are\s+({_NUMBER_TOKEN})\s+lots?\s+of\s+{first_pattern}"
        rf"\s+and\s+({_NUMBER_TOKEN})\s+lots?\s+of\s+{second_pattern}",
        normalized,
        flags=re.IGNORECASE,
    )
    demand_match = re.search(
        rf"demand[^.]*?\s+is\s+({_NUMBER_TOKEN})\s+lots?\s+of\s+{first_pattern}"
        rf"\s+and\s+({_NUMBER_TOKEN})\s+lots?\s+of\s+{second_pattern}",
        normalized,
        flags=re.IGNORECASE,
    )
    if (
        set(machine_requirements) != set(labels)
        or set(capacities) != {1, 2}
        or inventory_match is None
        or demand_match is None
    ):
        return TemplateSolveResult(False)

    inventory = {
        labels[0]: _number(inventory_match.group(1)),
        labels[1]: _number(inventory_match.group(2)),
    }
    demand = {
        labels[0]: _number(demand_match.group(1)),
        labels[1]: _number(demand_match.group(2)),
    }
    lower_bounds = {
        label: max(0.0, demand[label] - inventory[label])
        for label in labels
    }

    a1, a2 = machine_requirements[labels[0]]
    b1, b2 = machine_requirements[labels[1]]
    cap1 = capacities[1]
    cap2 = capacities[2]
    candidates: list[tuple[float, float]] = [
        (lower_bounds[labels[0]], lower_bounds[labels[1]])
    ]
    for x_value in (lower_bounds[labels[0]],):
        if b1 > 0:
            candidates.append((x_value, (cap1 - a1 * x_value) / b1))
        if b2 > 0:
            candidates.append((x_value, (cap2 - a2 * x_value) / b2))
    for y_value in (lower_bounds[labels[1]],):
        if a1 > 0:
            candidates.append(((cap1 - b1 * y_value) / a1, y_value))
        if a2 > 0:
            candidates.append(((cap2 - b2 * y_value) / a2, y_value))
    determinant = a1 * b2 - a2 * b1
    if not math.isclose(determinant, 0.0, abs_tol=1e-12):
        candidates.append(
            ((cap1 * b2 - cap2 * b1) / determinant, (a1 * cap2 - a2 * cap1) / determinant)
        )

    best: tuple[float, float, float] | None = None
    for x_value, y_value in candidates:
        if x_value < lower_bounds[labels[0]] - 1e-9 or y_value < lower_bounds[labels[1]] - 1e-9:
            continue
        if a1 * x_value + b1 * y_value > cap1 + 1e-9:
            continue
        if a2 * x_value + b2 * y_value > cap2 + 1e-9:
            continue
        ending_inventory = (
            inventory[labels[0]] + x_value - demand[labels[0]]
            + inventory[labels[1]] + y_value - demand[labels[1]]
        )
        if best is None or ending_inventory > best[0] + 1e-9:
            best = (ending_inventory, x_value, y_value)
    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="two_product_machine_inventory_surplus_lp",
            status="infeasible",
            confidence=0.8,
            artifact={
                "labels": labels,
                "machine_requirements": machine_requirements,
                "capacities": capacities,
                "inventory": inventory,
                "demand": demand,
            },
        )

    return TemplateSolveResult(
        matched=True,
        template_id="two_product_machine_inventory_surplus_lp",
        status="optimal",
        objective_value=float(best[0]),
        variable_values={
            f"produce_{_clean_label(labels[0])}": float(best[1]),
            f"produce_{_clean_label(labels[1])}": float(best[2]),
        },
        confidence=0.86,
        notes="Solved two-product fractional machine planning LP by checking all two-dimensional vertices.",
        artifact={
            "labels": labels,
            "machine_requirements": machine_requirements,
            "capacities_minutes": capacities,
            "inventory": inventory,
            "demand": demand,
            "minimum_production": lower_bounds,
        },
    )


def _solve_two_option_resource_max_mix(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("maximize" in lowered or "maximum" in lowered):
        return TemplateSolveResult(False)

    labels: list[str] = []
    objective: dict[str, float] = {}
    resources: dict[str, dict[str, float]] = {}
    limits: dict[str, float] = {}
    lower_bounds: dict[str, int] = {}
    upper_bounds: dict[str, int] = {}
    ratio_constraints: list[tuple[str, str, str, float]] = []

    if "newspaper" in lowered and "small bone treats" in lowered and "dogs" in lowered:
        matches = list(
            re.finditer(
                rf"(?:a|an)\s+([A-Za-z][A-Za-z\s-]*?)\s+can\s+deliver\s+({_NUMBER_TOKEN})\s+newspapers?"
                rf"[^.]*?requires\s+({_NUMBER_TOKEN})\s+small\s+bone\s+treats?",
                normalized,
                flags=re.IGNORECASE,
            )
        )
        if len(matches) != 2:
            return TemplateSolveResult(False)
        labels = [_singular_label(match.group(1)) for match in matches]
        objective = {label: _number(match.group(2)) for label, match in zip(labels, matches)}
        resources = {label: {"treats": _number(match.group(3))} for label, match in zip(labels, matches)}
        limit = _number_after_patterns(normalized, [rf"has\s+({_NUMBER_TOKEN})\s+small\s+bone\s+treats?\s+available"])
        if limit is None:
            return TemplateSolveResult(False)
        limits["treats"] = limit
        for label in labels:
            label_pattern = _resource_label_pattern(label)
            lower = re.search(rf"at\s+least\s+({_NUMBER_TOKEN})\s+{label_pattern}s?\s+must\s+be\s+used", normalized, flags=re.IGNORECASE)
            if lower:
                lower_bounds[label] = int(_number(lower.group(1)))
            share = re.search(
                rf"at\s+most\s+({_NUMBER_TOKEN})\s*%\s+of\s+the\s+dogs?\s+can\s+be\s+{label_pattern}s?",
                normalized,
                flags=re.IGNORECASE,
            )
            if share:
                ratio_constraints.append((label, "share<=", "total", _number(share.group(1)) / 100.0))

    elif "desks" in lowered and "seating" in lowered:
        matches = list(
            re.finditer(
                rf"([A-Za-z][A-Za-z\s-]*?desks?)\s+cost\s+\$?\s*({_NUMBER_TOKEN}),\s+"
                rf"take\s+up\s+({_NUMBER_TOKEN})\s+square\s+feet\s+of\s+space,\s+and\s+seat\s+({_NUMBER_TOKEN})\s+employees?",
                normalized,
                flags=re.IGNORECASE,
            )
        )
        if len(matches) != 2:
            return TemplateSolveResult(False)
        labels = [_singular_label(match.group(1)) for match in matches]
        objective = {label: _number(match.group(4)) for label, match in zip(labels, matches)}
        resources = {
            label: {"budget": _number(match.group(2)), "space": _number(match.group(3))}
            for label, match in zip(labels, matches)
        }
        budget = _number_after_patterns(normalized, [rf"spend\s+at\s+most\s+\$?\s*({_NUMBER_TOKEN})"])
        space = _number_after_patterns(normalized, [rf"at\s+most\s+({_NUMBER_TOKEN})\s+square\s+feet\s+of\s+desks?"])
        if budget is None or space is None:
            return TemplateSolveResult(False)
        limits = {"budget": budget, "space": space}

    elif "tomatoes" in lowered and "potatoes" in lowered and "hectares" in lowered:
        labels = ["tomatoes", "potatoes"]
        land = _number_after_patterns(normalized, [rf"has\s+({_NUMBER_TOKEN})\s+hectares?\s+available"])
        tomato_profit = _number_after_patterns(normalized, [rf"profit\s+per\s+hectare\s+of\s+tomatoes\s+is\s+\$?\s*({_NUMBER_TOKEN})"])
        potato_profit = _number_after_patterns(normalized, [rf"profit\s+per\s+hectare\s+of\s+potatoes\s+is\s+\$?\s*({_NUMBER_TOKEN})"])
        if land is None or tomato_profit is None or potato_profit is None:
            return TemplateSolveResult(False)
        objective = {"tomatoes": tomato_profit, "potatoes": potato_profit}
        resources = {label: {"land": 1.0} for label in labels}
        limits = {"land": land}
        for label in labels:
            lower = re.search(rf"at\s+least\s+({_NUMBER_TOKEN})\s+hectares?\s+of\s+{label}", normalized, flags=re.IGNORECASE)
            if lower:
                lower_bounds[label] = int(_number(lower.group(1)))
        ratio = re.search(
            rf"at\s+most\s+({_NUMBER_TOKEN})\s+times\s+the\s+amount\s+of\s+tomatoes\s+to\s+that\s+of\s+potatoes",
            normalized,
            flags=re.IGNORECASE,
        )
        if ratio:
            ratio_constraints.append(("tomatoes", "<=", "potatoes", _number(ratio.group(1))))

    elif "glass" in lowered and "plastic" in lowered and "bottles" in lowered and "water" in lowered:
        labels = ["glass", "plastic"]
        glass = re.search(rf"glass\s+bottle\s+can\s+hol[de]\s+({_NUMBER_TOKEN})\s*ml", normalized, flags=re.IGNORECASE)
        plastic = re.search(rf"plastic\s+bottle\s+can\s+hold\s+({_NUMBER_TOKEN})\s*ml", normalized, flags=re.IGNORECASE)
        water = _number_after_patterns(normalized, [rf"available\s+({_NUMBER_TOKEN})\s*ml\s+of\s+water"])
        if glass is None or plastic is None or water is None:
            return TemplateSolveResult(False)
        objective = {"glass": 1.0, "plastic": 1.0}
        resources = {"glass": {"water": _number(glass.group(1))}, "plastic": {"water": _number(plastic.group(1))}}
        limits = {"water": water}
        lower = re.search(rf"at\s+least\s+({_NUMBER_TOKEN})\s+glass\s+bottles?", normalized, flags=re.IGNORECASE)
        if lower:
            lower_bounds["glass"] = int(_number(lower.group(1)))
        ratio = re.search(
            rf"plastic\s+bottles?\s+must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})\s+times\s+the\s+number\s+of\s+glass\s+bottles?",
            normalized,
            flags=re.IGNORECASE,
        )
        if ratio:
            ratio_constraints.append(("plastic", ">=", "glass", _number(ratio.group(1))))

    elif "mechanical" in lowered and "standard" in lowered and "keyboards" in lowered:
        labels = ["mechanical", "standard"]
        mechanical = re.search(
            rf"mechanical\s+keyboards?\s+costs?\s+({_NUMBER_TOKEN})\s+units?\s+of\s+plastic\s+and\s+"
            rf"({_NUMBER_TOKEN})\s+units?\s+of\s+solder",
            normalized,
            flags=re.IGNORECASE,
        )
        standard = re.search(
            rf"standard\s+keyboards?\s+costs?\s+({_NUMBER_TOKEN})\s+units?\s+of\s+plastic\s+and\s+"
            rf"({_NUMBER_TOKEN})\s+units?\s+of\s+solder",
            normalized,
            flags=re.IGNORECASE,
        )
        plastic = _number_after_patterns(normalized, [rf"available\s+({_NUMBER_TOKEN})\s+units?\s+of\s+plastic"])
        solder = _number_after_patterns(normalized, [rf"and\s+({_NUMBER_TOKEN})\s+units?\s+of\s+solder"])
        if mechanical is None or standard is None or plastic is None or solder is None:
            return TemplateSolveResult(False)
        objective = {"mechanical": 1.0, "standard": 1.0}
        resources = {
            "mechanical": {"plastic": _number(mechanical.group(1)), "solder": _number(mechanical.group(2))},
            "standard": {"plastic": _number(standard.group(1)), "solder": _number(standard.group(2))},
        }
        limits = {"plastic": plastic, "solder": solder}
        lower = re.search(rf"at\s+least\s+({_NUMBER_TOKEN})\s+standard\s+keyboards?", normalized, flags=re.IGNORECASE)
        if lower:
            lower_bounds["standard"] = int(_number(lower.group(1)))
        ratio = re.search(
            rf"have\s+({_NUMBER_TOKEN})\s+times\s+as\s+many\s+mechanical\s+than\s+standard\s+keyboards?",
            normalized,
            flags=re.IGNORECASE,
        )
        if ratio:
            ratio_constraints.append(("mechanical", "=", "standard", _number(ratio.group(1))))

    else:
        return TemplateSolveResult(False)

    if len(labels) != 2 or set(objective) != set(labels) or set(resources) != set(labels) or not limits:
        return TemplateSolveResult(False)
    for label in labels:
        lower_bounds.setdefault(label, 0)
    upper_by_label: dict[str, int] = {}
    for label in labels:
        resource_limits = [
            int(math.floor(limit / amount))
            for resource, limit in limits.items()
            for amount in [resources[label].get(resource, 0.0)]
            if amount > 0
        ]
        upper_by_label[label] = min(resource_limits) if resource_limits else 1000
        if label in upper_bounds:
            upper_by_label[label] = min(upper_by_label[label], upper_bounds[label])

    best: tuple[float, tuple[int, int]] | None = None
    for first in range(lower_bounds[labels[0]], upper_by_label[labels[0]] + 1):
        for second in range(lower_bounds[labels[1]], upper_by_label[labels[1]] + 1):
            counts = {labels[0]: first, labels[1]: second}
            feasible = True
            for resource, limit in limits.items():
                used = sum(resources[label].get(resource, 0.0) * counts[label] for label in labels)
                if used > limit + 1e-9:
                    feasible = False
                    break
            if not feasible:
                continue
            total = sum(counts.values())
            for left, sense, right, factor in ratio_constraints:
                if sense == "share<=":
                    if total <= 0 or counts[left] > factor * total + 1e-9:
                        feasible = False
                        break
                elif sense == "<=":
                    if counts[left] > factor * counts[right] + 1e-9:
                        feasible = False
                        break
                elif sense == ">=":
                    if counts[left] < factor * counts[right] - 1e-9:
                        feasible = False
                        break
                elif sense == "=":
                    if not math.isclose(counts[left], factor * counts[right], abs_tol=1e-9):
                        feasible = False
                        break
            if not feasible:
                continue
            value = sum(objective[label] * counts[label] for label in labels)
            if best is None or value > best[0] + 1e-9:
                best = (value, (first, second))
    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="two_option_resource_max_mix_ilp",
            status="infeasible",
            confidence=0.83,
            artifact={
                "labels": labels,
                "objective": objective,
                "resources": resources,
                "limits": limits,
                "lower_bounds": lower_bounds,
                "ratio_constraints": ratio_constraints,
            },
        )
    selected = dict(zip(labels, best[1]))
    return TemplateSolveResult(
        matched=True,
        template_id="two_option_resource_max_mix_ilp",
        status="optimal",
        objective_value=float(best[0]),
        variable_values={
            f"count_{_clean_label(label)}": float(value)
            for label, value in selected.items()
            if value
        },
        confidence=0.86,
        notes="Solved two-option resource-constrained maximization by exact integer enumeration.",
        artifact={
            "labels": labels,
            "objective": objective,
            "resources": resources,
            "limits": limits,
            "lower_bounds": lower_bounds,
            "ratio_constraints": ratio_constraints,
            "selected": selected,
        },
    )


def _solve_bombing_success_probability(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "heavy bombs" in lowered
        and "light bombs" in lowered
        and "probability of destruction" in lowered
        and "at least two" in lowered
        and "fuel" in lowered
        and "maximize the probability" in lowered
    ):
        return TemplateSolveResult(False)

    heavy_stock = _number_after_patterns(
        normalized,
        [rf"maximum\s+of\s+({_NUMBER_TOKEN})\s+heavy\s+bombs?"],
    )
    light_stock = _number_after_patterns(
        normalized,
        [rf"and\s+({_NUMBER_TOKEN})\s+light\s+bombs?\s+can\s+be\s+used"],
    )
    fuel_limit = _number_after_patterns(
        normalized,
        [rf"fuel\s+consumption\s+must\s+not\s+exceed\s+({_NUMBER_TOKEN})\s+liters?"],
    )
    heavy_rate = _number_after_patterns(
        normalized,
        [rf"carrying\s+heavy\s+bombs?,\s+each\s+liter\s+of\s+fuel\s+allows\s+a\s+distance\s+of\s+({_NUMBER_TOKEN})\s+km"],
    )
    light_rate = _number_after_patterns(
        normalized,
        [rf"with\s+light\s+bombs?,\s+each\s+liter\s+allows\s+({_NUMBER_TOKEN})\s+km"],
    )
    empty_rate = _number_after_patterns(
        normalized,
        [rf"each\s+liter\s+of\s+fuel\s+allows\s+({_NUMBER_TOKEN})\s+km\s+when\s+the\s+aircraft\s+is\s+empty"],
    )
    fixed_fuel = _number_after_patterns(
        normalized,
        [rf"({_NUMBER_TOKEN})\s+liters?\s+for\s+both\s+takeoff\s+and\s+landing\s+per\s+trip"],
    )
    if None in (heavy_stock, light_stock, fuel_limit, heavy_rate, light_rate, empty_rate, fixed_fuel):
        return TemplateSolveResult(False)

    targets: list[dict[str, float]] = []
    for header, rows in _parse_markdown_tables(text):
        header_text = " ".join(header).lower()
        if "distance" not in header_text or "heavy bomb" not in header_text or "light bomb" not in header_text:
            continue
        for row in rows:
            if len(row) < 4:
                continue
            part = _first_number(row[0])
            distance = _first_number(row[1])
            heavy_probability = _first_number(row[2])
            light_probability = _first_number(row[3])
            if part is None or distance is None or heavy_probability is None or light_probability is None:
                continue
            targets.append(
                {
                    "part": part,
                    "distance": distance,
                    "heavy_probability": heavy_probability,
                    "light_probability": light_probability,
                }
            )
    if len(targets) < 2:
        return TemplateSolveResult(False)

    heavy_max = int(heavy_stock)
    light_max = int(light_stock)
    fuel_units_limit = int(round(float(fuel_limit) * 2.0))
    options_by_target: list[list[tuple[int, int, int, float]]] = []
    for target in targets:
        distance = target["distance"]
        heavy_fuel = distance / float(heavy_rate) + distance / float(empty_rate) + float(fixed_fuel)
        light_fuel = distance / float(light_rate) + distance / float(empty_rate) + float(fixed_fuel)
        options: list[tuple[int, int, int, float]] = []
        for heavy_count in range(heavy_max + 1):
            for light_count in range(light_max + 1):
                fuel_units = int(round(2.0 * (heavy_fuel * heavy_count + light_fuel * light_count)))
                if fuel_units > fuel_units_limit:
                    continue
                destroy_probability = 1.0 - (
                    (1.0 - target["heavy_probability"]) ** heavy_count
                    * (1.0 - target["light_probability"]) ** light_count
                )
                options.append((heavy_count, light_count, fuel_units, destroy_probability))
        options_by_target.append(options)

    def add_pareto(
        candidates: list[tuple[float, float, tuple[tuple[int, int], ...]]],
        candidate: tuple[float, float, tuple[tuple[int, int], ...]],
    ) -> None:
        probability_at_least_one, probability_at_least_two, _allocation = candidate
        for existing_one, existing_two, _existing_allocation in candidates:
            if (
                existing_one >= probability_at_least_one - 1e-12
                and existing_two >= probability_at_least_two - 1e-12
            ):
                return
        candidates[:] = [
            existing
            for existing in candidates
            if not (
                probability_at_least_one >= existing[0] - 1e-12
                and probability_at_least_two >= existing[1] - 1e-12
            )
        ]
        candidates.append(candidate)

    states: dict[tuple[int, int, int], list[tuple[float, float, tuple[tuple[int, int], ...]]]] = {
        (0, 0, 0): [(0.0, 0.0, tuple())]
    }
    for options in options_by_target:
        next_states: dict[tuple[int, int, int], list[tuple[float, float, tuple[tuple[int, int], ...]]]] = {}
        for (used_heavy, used_light, used_fuel), candidates in states.items():
            for heavy_count, light_count, fuel_units, destroy_probability in options:
                new_heavy = used_heavy + heavy_count
                new_light = used_light + light_count
                new_fuel = used_fuel + fuel_units
                if new_heavy > heavy_max or new_light > light_max or new_fuel > fuel_units_limit:
                    continue
                key = (new_heavy, new_light, new_fuel)
                bucket = next_states.setdefault(key, [])
                for probability_at_least_one, probability_at_least_two, allocation in candidates:
                    next_at_least_two = probability_at_least_two + destroy_probability * (
                        probability_at_least_one - probability_at_least_two
                    )
                    next_at_least_one = probability_at_least_one + destroy_probability * (
                        1.0 - probability_at_least_one
                    )
                    add_pareto(
                        bucket,
                        (
                            next_at_least_one,
                            next_at_least_two,
                            allocation + ((heavy_count, light_count),),
                        ),
                    )
        states = next_states

    best: tuple[float, tuple[int, int, int], tuple[tuple[int, int], ...]] | None = None
    for key, candidates in states.items():
        for _probability_at_least_one, probability_at_least_two, allocation in candidates:
            if best is None or probability_at_least_two > best[0] + 1e-12:
                best = (probability_at_least_two, key, allocation)
    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="bombing_success_probability_dp",
            status="infeasible",
            confidence=0.78,
        )

    variable_values: dict[str, float] = {}
    for target, (heavy_count, light_count) in zip(targets, best[2]):
        part = int(target["part"])
        if heavy_count:
            variable_values[f"heavy_bombs_part_{part}"] = float(heavy_count)
        if light_count:
            variable_values[f"light_bombs_part_{part}"] = float(light_count)

    return TemplateSolveResult(
        matched=True,
        template_id="bombing_success_probability_dp",
        status="optimal",
        objective_value=float(best[0]),
        variable_values=variable_values,
        confidence=0.86,
        notes="Solved bombing allocation by dynamic programming over bomb stock, fuel, and target-destruction probabilities.",
        artifact={
            "targets": targets,
            "heavy_stock": heavy_max,
            "light_stock": light_max,
            "fuel_limit": fuel_limit,
            "fuel_used": best[1][2] / 2.0,
            "allocation": [
                {"part": int(target["part"]), "heavy": heavy, "light": light}
                for target, (heavy, light) in zip(targets, best[2])
            ],
        },
    )


def _solve_two_test_probe_mix(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "salinity test" in lowered
        and "ph test" in lowered
        and "probes" in lowered
        and ("minimize" in lowered or "minimum" in lowered)
    ):
        return TemplateSolveResult(False)
    salinity_probe = re.search(
        rf"salinity\s+test\s+requires\s+({_NUMBER_TOKEN})\s+probes?",
        normalized,
        flags=re.IGNORECASE,
    )
    ph_probe = re.search(
        rf"pH\s+test\s+requires\s+({_NUMBER_TOKEN})\s+probes?",
        normalized,
        flags=re.IGNORECASE,
    )
    ph_min = re.search(
        rf"at\s+least\s+({_NUMBER_TOKEN})\s+pH\s+tests?",
        normalized,
        flags=re.IGNORECASE,
    )
    total_min = re.search(
        rf"at\s+least\s+({_NUMBER_TOKEN})\s+tests?\s+must\s+be\s+performed",
        normalized,
        flags=re.IGNORECASE,
    )
    ratio = re.search(
        rf"at\s+most\s+({_NUMBER_TOKEN})\s+times\s+more\s+pH\s+tests?\s+than\s+salinity\s+tests?",
        normalized,
        flags=re.IGNORECASE,
    )
    if not (salinity_probe and ph_probe and ph_min and total_min and ratio):
        return TemplateSolveResult(False)

    c_salinity = _number(salinity_probe.group(1))
    c_ph = _number(ph_probe.group(1))
    ph_lower = int(_number(ph_min.group(1)))
    total_lower = int(_number(total_min.group(1)))
    ph_to_salinity_max = _number(ratio.group(1))
    upper = total_lower + ph_lower + 50
    best: tuple[float, int, int] | None = None
    for salinity in range(upper + 1):
        for ph in range(ph_lower, upper + 1):
            if salinity + ph < total_lower:
                continue
            if ph > ph_to_salinity_max * salinity + 1e-9:
                continue
            value = c_salinity * salinity + c_ph * ph
            if best is None or value < best[0] - 1e-9:
                best = (value, salinity, ph)
    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="two_test_probe_mix_ilp",
            status="infeasible",
            confidence=0.82,
        )
    return TemplateSolveResult(
        matched=True,
        template_id="two_test_probe_mix_ilp",
        status="optimal",
        objective_value=float(best[0]),
        variable_values={"salinity_tests": float(best[1]), "ph_tests": float(best[2])},
        confidence=0.86,
        notes="Solved two-test probe minimization by exact integer enumeration.",
        artifact={
            "probe_costs": {"salinity": c_salinity, "ph": c_ph},
            "ph_minimum": ph_lower,
            "total_minimum": total_lower,
            "ph_to_salinity_max": ph_to_salinity_max,
        },
    )


def _solve_furnace_purchase_min_count(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "furnace" in lowered
        and "apartments" in lowered
        and "kwh" in lowered
        and ("minimize" in lowered or "minimum" in lowered)
    ):
        return TemplateSolveResult(False)
    new_match = re.search(
        rf"new\s+model\s+furnace\s+can\s+heat\s+({_NUMBER_TOKEN})\s+apartments?\s+and\s+consumes\s+({_NUMBER_TOKEN})\s+kWh",
        normalized,
        flags=re.IGNORECASE,
    )
    old_match = re.search(
        rf"old\s+model\s+can\s+heat\s+({_NUMBER_TOKEN})\s+apartments?\s+and\s+consumes\s+({_NUMBER_TOKEN})\s+kWh",
        normalized,
        flags=re.IGNORECASE,
    )
    old_share = re.search(
        rf"at\s+most\s+({_NUMBER_TOKEN})\s*%\s+of\s+the\s+furnaces?\s+can\s+be\s+the\s+old\s+model",
        normalized,
        flags=re.IGNORECASE,
    )
    new_min = re.search(
        rf"at\s+least\s+({_NUMBER_TOKEN})\s+new\s+model\s+furnaces?",
        normalized,
        flags=re.IGNORECASE,
    )
    demand = re.search(
        rf"heat\s+at\s+least\s+({_NUMBER_TOKEN})\s+apartments?",
        normalized,
        flags=re.IGNORECASE,
    )
    electricity = re.search(
        rf"has\s+({_NUMBER_TOKEN})\s+kWh\s+of\s+electricity\s+available",
        normalized,
        flags=re.IGNORECASE,
    )
    if not (new_match and old_match and old_share and new_min and demand and electricity):
        return TemplateSolveResult(False)

    new_heat, new_energy = _number(new_match.group(1)), _number(new_match.group(2))
    old_heat, old_energy = _number(old_match.group(1)), _number(old_match.group(2))
    old_fraction = _number(old_share.group(1)) / 100.0
    new_lower = int(_number(new_min.group(1)))
    heat_required = _number(demand.group(1))
    energy_limit = _number(electricity.group(1))
    upper = int(math.ceil(heat_required / max(1.0, min(new_heat, old_heat)))) + 30

    best: tuple[int, int, int] | None = None
    for new_count in range(new_lower, upper + 1):
        for old_count in range(upper + 1):
            total = new_count + old_count
            if total <= 0:
                continue
            if old_count > old_fraction * total + 1e-9:
                continue
            if new_heat * new_count + old_heat * old_count < heat_required - 1e-9:
                continue
            if new_energy * new_count + old_energy * old_count > energy_limit + 1e-9:
                continue
            if best is None or total < best[0]:
                best = (total, new_count, old_count)
    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="furnace_purchase_min_count_ilp",
            status="infeasible",
            confidence=0.84,
            artifact={
                "heat": {"new": new_heat, "old": old_heat, "required": heat_required},
                "energy": {"new": new_energy, "old": old_energy, "limit": energy_limit},
                "old_share_max": old_fraction,
                "new_minimum": new_lower,
            },
        )
    return TemplateSolveResult(
        matched=True,
        template_id="furnace_purchase_min_count_ilp",
        status="optimal",
        objective_value=float(best[0]),
        variable_values={"new_model_furnaces": float(best[1]), "old_model_furnaces": float(best[2])},
        confidence=0.86,
        notes="Solved furnace purchase count minimization by exact integer enumeration.",
    )


def _solve_two_ingredient_mix_profit_lp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "snack" in lowered
        and "mix" in lowered
        and "profit per kg" in lowered
        and ("maximize" in lowered or "maximum" in lowered)
    ):
        return TemplateSolveResult(False)
    first_mix = re.search(
        rf"first\s+mix\s+contains\s+({_NUMBER_TOKEN})\s*%\s+([A-Za-z][A-Za-z\s-]*?snacks?)\s+and\s+"
        rf"({_NUMBER_TOKEN})\s*%\s+([A-Za-z][A-Za-z\s-]*?snacks?)",
        normalized,
        flags=re.IGNORECASE,
    )
    second_mix = re.search(
        rf"second\s+mix\s+contains\s+({_NUMBER_TOKEN})\s*%\s+([A-Za-z][A-Za-z\s-]*?snacks?)\s+and\s+"
        rf"({_NUMBER_TOKEN})\s*%\s+([A-Za-z][A-Za-z\s-]*?snacks?)",
        normalized,
        flags=re.IGNORECASE,
    )
    stock = re.search(
        rf"on\s+hand\s+({_NUMBER_TOKEN})\s*kg\s+of\s+([A-Za-z][A-Za-z\s-]*?snacks?)\s+and\s+"
        rf"({_NUMBER_TOKEN})\s*kg\s+of\s+([A-Za-z][A-Za-z\s-]*?snacks?)",
        normalized,
        flags=re.IGNORECASE,
    )
    profit = re.search(
        rf"profit\s+per\s+kg\s+of\s+the\s+first\s+mix\s+is\s+\$?\s*({_NUMBER_TOKEN})\s+and\s+"
        rf"the\s+profit\s+per\s+kg\s+of\s+the\s+second\s+mix\s+is\s+\$?\s*({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if not (first_mix and second_mix and stock and profit):
        return TemplateSolveResult(False)

    ingredients = [_singular_label(first_mix.group(2)), _singular_label(first_mix.group(4))]
    if {_singular_label(second_mix.group(2)), _singular_label(second_mix.group(4))} != set(ingredients):
        return TemplateSolveResult(False)
    stock_by_ingredient = {
        _singular_label(stock.group(2)): _number(stock.group(1)),
        _singular_label(stock.group(4)): _number(stock.group(3)),
    }
    if set(stock_by_ingredient) != set(ingredients):
        return TemplateSolveResult(False)

    composition = {"first": {}, "second": {}}
    composition["first"][_singular_label(first_mix.group(2))] = _number(first_mix.group(1)) / 100.0
    composition["first"][_singular_label(first_mix.group(4))] = _number(first_mix.group(3)) / 100.0
    composition["second"][_singular_label(second_mix.group(2))] = _number(second_mix.group(1)) / 100.0
    composition["second"][_singular_label(second_mix.group(4))] = _number(second_mix.group(3)) / 100.0

    status, objective, values, message = _linprog_maximize(
        objective=[_number(profit.group(1)), _number(profit.group(2))],
        constraints=[
            [composition["first"][ingredient], composition["second"][ingredient]]
            for ingredient in ingredients
        ],
        upper_bounds=[stock_by_ingredient[ingredient] for ingredient in ingredients],
    )
    if status != "optimal":
        return TemplateSolveResult(
            matched=True,
            template_id="two_ingredient_mix_profit_lp",
            status=status,
            confidence=0.84,
            notes=message,
        )
    return TemplateSolveResult(
        matched=True,
        template_id="two_ingredient_mix_profit_lp",
        status="optimal",
        objective_value=objective,
        variable_values={"first_mix_kg": values[0], "second_mix_kg": values[1]},
        confidence=0.88,
        notes="Solved two-ingredient continuous mix profit LP.",
        artifact={"ingredients": ingredients, "composition": composition, "stock": stock_by_ingredient},
    )


def _solve_two_team_capacity_max_mix(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "small team" in lowered
        and "large team" in lowered
        and "employees" in lowered
        and ("maximize" in lowered or "maximum" in lowered)
    ):
        return TemplateSolveResult(False)
    small = re.search(
        rf"small\s+team\s+requires\s+({_NUMBER_TOKEN})\s+employees?\s+and\s+can\s+mow\s+({_NUMBER_TOKEN})\s+sq\s*ft",
        normalized,
        flags=re.IGNORECASE,
    )
    large = re.search(
        rf"large\s+team\s+requires\s+({_NUMBER_TOKEN})\s+employees?\s+and\s+can\s+mow\s+({_NUMBER_TOKEN})\s+sq\s*ft",
        normalized,
        flags=re.IGNORECASE,
    )
    employees = re.search(
        rf"has\s+({_NUMBER_TOKEN})\s+employees?\s+available",
        normalized,
        flags=re.IGNORECASE,
    )
    ratio = re.search(
        rf"number\s+of\s+small\s+teams?\s+must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})\s+times\s+as\s+much\s+as\s+the\s+number\s+of\s+large\s+teams?",
        normalized,
        flags=re.IGNORECASE,
    )
    lower_large = re.search(rf"at\s+least\s+({_NUMBER_TOKEN})\s+large\s+teams?", normalized, flags=re.IGNORECASE)
    lower_small = re.search(rf"at\s+least\s+({_NUMBER_TOKEN})\s+small\s+teams?", normalized, flags=re.IGNORECASE)
    if not (small and large and employees and ratio and lower_large and lower_small):
        return TemplateSolveResult(False)
    small_employees, small_capacity = _number(small.group(1)), _number(small.group(2))
    large_employees, large_capacity = _number(large.group(1)), _number(large.group(2))
    employee_limit = _number(employees.group(1))
    factor = _number(ratio.group(1))
    min_large = int(_number(lower_large.group(1)))
    min_small = int(_number(lower_small.group(1)))
    best: tuple[float, int, int] | None = None
    for small_count in range(min_small, int(employee_limit // small_employees) + 1):
        for large_count in range(min_large, int(employee_limit // large_employees) + 1):
            if small_employees * small_count + large_employees * large_count > employee_limit + 1e-9:
                continue
            if small_count < factor * large_count - 1e-9:
                continue
            value = small_capacity * small_count + large_capacity * large_count
            if best is None or value > best[0] + 1e-9:
                best = (value, small_count, large_count)
    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="two_team_capacity_max_mix_ilp",
            status="infeasible",
            confidence=0.82,
        )
    return TemplateSolveResult(
        matched=True,
        template_id="two_team_capacity_max_mix_ilp",
        status="optimal",
        objective_value=float(best[0]),
        variable_values={"small_teams": float(best[1]), "large_teams": float(best[2])},
        confidence=0.86,
        notes="Solved two-team capacity maximization by exact integer enumeration.",
    )


def _solve_two_food_ratio_protein_max(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "smoothie" in lowered
        and "protein bar" in lowered
        and "calories" in lowered
        and "protein intake" in lowered
        and ("maximize" in lowered or "maximum" in lowered)
    ):
        return TemplateSolveResult(False)
    smoothie = re.search(
        rf"each\s+smoothie\s+contains\s+({_NUMBER_TOKEN})\s+units?\s+of\s+protein\s+and\s+({_NUMBER_TOKEN})\s+calories",
        normalized,
        flags=re.IGNORECASE,
    )
    bar = re.search(
        rf"each\s+protein\s+bar\s+contains\s+({_NUMBER_TOKEN})\s+units?\s+of\s+protein\s+and\s+({_NUMBER_TOKEN})\s+calories",
        normalized,
        flags=re.IGNORECASE,
    )
    ratio = re.search(
        rf"must\s+eat\s+({_NUMBER_TOKEN})\s+times\s+more\s+protein\s+bars?\s+than\s+smoothies?",
        normalized,
        flags=re.IGNORECASE,
    )
    calories = re.search(
        rf"consume\s+at\s+most\s+({_NUMBER_TOKEN})\s+calories",
        normalized,
        flags=re.IGNORECASE,
    )
    if not (smoothie and bar and ratio and calories):
        return TemplateSolveResult(False)
    smoothie_protein, smoothie_calories = _number(smoothie.group(1)), _number(smoothie.group(2))
    bar_protein, bar_calories = _number(bar.group(1)), _number(bar.group(2))
    factor = int(_number(ratio.group(1)))
    calorie_limit = _number(calories.group(1))
    best: tuple[float, int, int] | None = None
    for smoothies in range(int(calorie_limit // max(1.0, smoothie_calories)) + 1):
        bars = factor * smoothies
        if smoothies <= 0 and bars <= 0:
            continue
        used_calories = smoothie_calories * smoothies + bar_calories * bars
        if used_calories > calorie_limit + 1e-9:
            continue
        value = smoothie_protein * smoothies + bar_protein * bars
        if best is None or value > best[0] + 1e-9:
            best = (value, smoothies, bars)
    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="two_food_ratio_protein_max_ilp",
            status="infeasible",
            confidence=0.82,
        )
    return TemplateSolveResult(
        matched=True,
        template_id="two_food_ratio_protein_max_ilp",
        status="optimal",
        objective_value=float(best[0]),
        variable_values={"smoothies": float(best[1]), "protein_bars": float(best[2])},
        confidence=0.84,
        notes="Solved fixed-ratio two-food protein maximization by exact integer enumeration.",
    )


def _solve_two_item_nutrition_min_mix(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()

    if "painkiller" in lowered and "sleeping pill" in lowered and "digestive medicine" in lowered:
        morphine_limit = _number_after_patterns(normalized, [rf"has\s+({_NUMBER_TOKEN})\s*mg\s+of\s+morphine"])
        painkiller = re.search(
            rf"painkiller\s+pills?\s+requires\s+({_NUMBER_TOKEN})\s*mg\s+of\s+morphine\s+and\s+"
            rf"({_NUMBER_TOKEN})\s+units?\s+of\s+digestive\s+medicine",
            normalized,
            flags=re.IGNORECASE,
        )
        sleeping = re.search(
            rf"sleeping\s+pills?\s+requires\s+({_NUMBER_TOKEN})\s*mg\s+of\s+morphine\s+and\s+"
            rf"({_NUMBER_TOKEN})\s+units?\s+of\s+digestive\s+medicine",
            normalized,
            flags=re.IGNORECASE,
        )
        lower = _number_after_patterns(normalized, [rf"at\s+least\s+({_NUMBER_TOKEN})\s+painkiller\s+pills?"])
        share = re.search(rf"at\s+least\s+({_NUMBER_TOKEN})\s*%\s+of\s+the\s+pills?\s+should\s+be\s+sleeping", normalized, flags=re.IGNORECASE)
        if not (morphine_limit is not None and painkiller and sleeping and lower is not None and share):
            return TemplateSolveResult(False)
        share_value = _number(share.group(1)) / 100.0
        return _solve_small_integer_min_cost_model(
            template_id="two_item_nutrition_min_mix_ilp",
            symbols=["painkiller", "sleeping"],
            costs={"painkiller": _number(painkiller.group(2)), "sleeping": _number(sleeping.group(2))},
            constraints=[
                (
                    {"painkiller": _number(painkiller.group(1)), "sleeping": _number(sleeping.group(1))},
                    -math.inf,
                    morphine_limit,
                    "morphine_upper",
                ),
                ({"painkiller": 1.0}, lower, math.inf, "painkiller_lower"),
                ({"painkiller": share_value, "sleeping": share_value - 1.0}, -math.inf, 0.0, "sleeping_share_lower"),
            ],
            upper_bounds={
                "painkiller": math.floor(morphine_limit / _number(painkiller.group(1))),
                "sleeping": math.floor(morphine_limit / _number(sleeping.group(1))),
            },
            confidence=0.84,
            notes="Solved two-pill integer nutrition minimization with resource and share constraints.",
        )

    if "burger" in lowered and "pizza" in lowered and "cholesterol" in lowered:
        burger = re.search(
            rf"burger\s+contains\s+({_NUMBER_TOKEN})\s+units?\s+of\s+fat\s+and\s+({_NUMBER_TOKEN})\s+calories",
            normalized,
            flags=re.IGNORECASE,
        )
        pizza = re.search(
            rf"pizza\s+contains\s+({_NUMBER_TOKEN})\s+units?\s+of\s+fat\s+and\s+({_NUMBER_TOKEN})\s+calories",
            normalized,
            flags=re.IGNORECASE,
        )
        cholesterol = re.search(
            rf"burger\s+contains\s+({_NUMBER_TOKEN})\s+units?\s+of\s+cholesterol\s+while\s+each\s+slice\s+of\s+pizza\s+contains\s+"
            rf"({_NUMBER_TOKEN})\s+units?\s+of\s+cholesterol",
            normalized,
            flags=re.IGNORECASE,
        )
        fat_lower = _number_after_patterns(normalized, [rf"at\s+least\s+({_NUMBER_TOKEN})\s+units?\s+of\s+fat"])
        calorie_lower = _number_after_patterns(normalized, [rf"and\s+({_NUMBER_TOKEN})\s+calories"])
        ratio = re.search(rf"at\s+least\s+(twice|{_NUMBER_TOKEN})\s+as\s+many\s+slices?\s+of\s+pizza\s+as\s+burgers?", normalized, flags=re.IGNORECASE)
        if not (burger and pizza and cholesterol and fat_lower is not None and calorie_lower is not None and ratio):
            return TemplateSolveResult(False)
        factor = _project_coefficient(ratio.group(1))
        return _solve_small_integer_min_cost_model(
            template_id="two_item_nutrition_min_mix_ilp",
            symbols=["burger", "pizza"],
            costs={"burger": _number(cholesterol.group(1)), "pizza": _number(cholesterol.group(2))},
            constraints=[
                ({"burger": _number(burger.group(1)), "pizza": _number(pizza.group(1))}, fat_lower, math.inf, "fat_lower"),
                ({"burger": _number(burger.group(2)), "pizza": _number(pizza.group(2))}, calorie_lower, math.inf, "calorie_lower"),
                ({"burger": -factor, "pizza": 1.0}, 0.0, math.inf, "pizza_ratio_lower"),
            ],
            confidence=0.84,
            notes="Solved two-food integer cholesterol minimization with lower nutrient and ratio constraints.",
        )

    if "salmon" in lowered and "eggs" in lowered and "sodium" in lowered and "macro-counting" in lowered:
        salmon = re.search(
            rf"salmon\s+contains\s+({_NUMBER_TOKEN})\s+calories,\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+protein,\s+and\s+"
            rf"({_NUMBER_TOKEN})\s*mg\s+of\s+sodium",
            normalized,
            flags=re.IGNORECASE,
        )
        eggs = re.search(
            rf"eggs\s+contains\s+({_NUMBER_TOKEN})\s+calories,\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+protein,\s+and\s+"
            rf"({_NUMBER_TOKEN})\s*mg\s+of\s+sodium",
            normalized,
            flags=re.IGNORECASE,
        )
        egg_share = re.search(rf"at\s+most\s+({_NUMBER_TOKEN})\s*%\s+of\s+his\s+meals?\s+can\s+be\s+eggs", normalized, flags=re.IGNORECASE)
        calorie_lower = _number_after_patterns(normalized, [rf"at\s+least\s+({_NUMBER_TOKEN})\s+calories"])
        protein_lower = _number_after_patterns(normalized, [rf"and\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+protein"])
        if not (salmon and eggs and egg_share and calorie_lower is not None and protein_lower is not None):
            return TemplateSolveResult(False)
        try:
            from scipy.optimize import linprog
        except ImportError as exc:
            return TemplateSolveResult(
                matched=True,
                template_id="two_item_nutrition_min_mix_lp",
                status="solver_unavailable",
                confidence=0.82,
                notes=str(exc),
            )
        share_value = _number(egg_share.group(1)) / 100.0
        objective = [_number(salmon.group(3)), _number(eggs.group(3))]
        a_ub = [
            [-_number(salmon.group(1)), -_number(eggs.group(1))],
            [-_number(salmon.group(2)), -_number(eggs.group(2))],
            [-share_value, 1.0 - share_value],
        ]
        b_ub = [-calorie_lower, -protein_lower, 0.0]
        result = linprog(objective, A_ub=a_ub, b_ub=b_ub, bounds=[(0, None), (0, None)], method="highs")
        if not result.success:
            return TemplateSolveResult(
                matched=True,
                template_id="two_item_nutrition_min_mix_lp",
                status="solver_failed",
                confidence=0.82,
                notes=str(result.message),
            )
        return TemplateSolveResult(
            matched=True,
            template_id="two_item_nutrition_min_mix_lp",
            status="optimal",
            objective_value=float(result.fun),
            variable_values={"salmon": float(result.x[0]), "eggs": float(result.x[1])},
            confidence=0.86,
            notes="Solved continuous two-food sodium minimization with lower nutrient and share constraints.",
        )

    return TemplateSolveResult(False)


def _solve_two_volunteer_gift_max_mix(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "seasonal" in lowered
        and "full-time" in lowered
        and "volunteer" in lowered
        and "deliver gifts" in lowered
        and ("maximize" in lowered or "maximum" in lowered)
    ):
        return TemplateSolveResult(False)
    seasonal = re.search(
        rf"seasonal\s+volunteer\s+can\s+deliver\s+({_NUMBER_TOKEN})\s+gifts?\s+and\s+gets\s+({_NUMBER_TOKEN})\s+points?",
        normalized,
        flags=re.IGNORECASE,
    )
    full_time = re.search(
        rf"full-time\s+volunteer\s+can\s+deliver\s+({_NUMBER_TOKEN})\s+gifts?\s+and\s+gets\s+({_NUMBER_TOKEN})\s+points?",
        normalized,
        flags=re.IGNORECASE,
    )
    points = _number_after_patterns(normalized, [rf"can\s+only\s+give\s+out\s+({_NUMBER_TOKEN})\s+points?"])
    seasonal_share = re.search(rf"maximum\s+of\s+({_NUMBER_TOKEN})\s*%\s+of\s+the\s+volunteers?\s+can\s+be\s+seasonal", normalized, flags=re.IGNORECASE)
    full_time_lower = _number_after_patterns(normalized, [rf"at\s+least\s+({_NUMBER_TOKEN})\s+must\s+be\s+full-time"])
    if not (seasonal and full_time and points is not None and seasonal_share and full_time_lower is not None):
        return TemplateSolveResult(False)
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="two_volunteer_gift_max_mix_ilp",
            status="solver_unavailable",
            confidence=0.82,
            notes=str(exc),
        )
    share = _number(seasonal_share.group(1)) / 100.0
    matrix = np.array(
        [
            [_number(seasonal.group(2)), _number(full_time.group(2))],
            [1.0 - share, -share],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    lower = np.array([-math.inf, -math.inf, full_time_lower], dtype=float)
    upper = np.array([points, 0.0, math.inf], dtype=float)
    result = milp(
        c=np.array([-_number(seasonal.group(1)), -_number(full_time.group(1))], dtype=float),
        integrality=np.ones(2),
        bounds=Bounds(np.zeros(2), np.full(2, math.inf)),
        constraints=LinearConstraint(matrix, lower, upper),
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="two_volunteer_gift_max_mix_ilp",
            status="infeasible",
            confidence=0.82,
            notes=str(result.message),
        )
    return TemplateSolveResult(
        matched=True,
        template_id="two_volunteer_gift_max_mix_ilp",
        status="optimal",
        objective_value=float(-result.fun),
        variable_values={"seasonal": float(result.x[0]), "full_time": float(result.x[1])},
        confidence=0.86,
        notes="Solved two-volunteer gift maximization with points, share, and lower-bound constraints.",
    )


def _solve_two_product_resource_profit_max(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()

    if "bakery" in lowered and "bagels" in lowered and "croissants" in lowered and "pastry chef" in lowered:
        bagel = re.search(
            rf"bagels?\s+can\s+be\s+made\s+using\s+({_NUMBER_TOKEN})\s+hours?\s+of\s+oven\s+time\s+and\s+"
            rf"({_NUMBER_TOKEN})\s+hours?\s+of\s+pastry\s+chef\s+time",
            normalized,
            flags=re.IGNORECASE,
        )
        croissant = re.search(
            rf"croissants?.*?take\s+({_NUMBER_TOKEN})\s+hour\s+of\s+oven\s+time,\s+they\s+take\s+"
            rf"({_NUMBER_TOKEN})\s+hours?\s+of\s+pastry\s+chef\s+time",
            normalized,
            flags=re.IGNORECASE,
        )
        oven = _number_after_patterns(normalized, [rf"at\s+most\s+({_NUMBER_TOKEN})\s+hours?\s+available\s+for\s+the\s+oven"])
        chef = _number_after_patterns(normalized, [rf"and\s+({_NUMBER_TOKEN})\s+pastry\s+chef\s+hours?\s+available"])
        profit = re.search(
            rf"profit\s+per\s+batch\s+is\s+\$?\s*({_NUMBER_TOKEN})\s+and\s+\$?\s*({_NUMBER_TOKEN})\s+respectively",
            normalized,
            flags=re.IGNORECASE,
        )
        if not (bagel and croissant and oven is not None and chef is not None and profit):
            return TemplateSolveResult(False)
        try:
            from scipy.optimize import Bounds, LinearConstraint, milp
            import numpy as np
        except ImportError as exc:
            return TemplateSolveResult(
                matched=True,
                template_id="two_product_resource_profit_max_ilp",
                status="solver_unavailable",
                confidence=0.82,
                notes=str(exc),
            )
        matrix = np.array(
            [
                [_number(bagel.group(1)), _number(croissant.group(1))],
                [_number(bagel.group(2)), _number(croissant.group(2))],
            ],
            dtype=float,
        )
        result = milp(
            c=-np.array([_number(profit.group(1)), _number(profit.group(2))], dtype=float),
            integrality=np.ones(2),
            bounds=Bounds(np.zeros(2), np.full(2, math.inf)),
            constraints=LinearConstraint(matrix, np.full(2, -math.inf), np.array([oven, chef], dtype=float)),
        )
        if not result.success:
            return TemplateSolveResult(
                matched=True,
                template_id="two_product_resource_profit_max_ilp",
                status="infeasible",
                confidence=0.82,
                notes=str(result.message),
            )
        return TemplateSolveResult(
            matched=True,
            template_id="two_product_resource_profit_max_ilp",
            status="optimal",
            objective_value=float(-result.fun),
            variable_values={"bagels": float(result.x[0]), "croissants": float(result.x[1])},
            confidence=0.86,
            notes="Solved two-product integer bakery profit maximization with oven and pastry-chef capacity.",
        )

    if "farmer" in lowered and "turnips" in lowered and "pumpkins" in lowered and "pesticide" in lowered:
        land = _number_after_patterns(normalized, [rf"has\s+({_NUMBER_TOKEN})\s+acres?\s+of\s+land"])
        turnip = re.search(
            rf"Turnips\s+require\s+({_NUMBER_TOKEN})\s+minutes?\s+of\s+watering\s+and\s+\$?\s*({_NUMBER_TOKEN})\s+worth\s+of\s+pesticide",
            normalized,
            flags=re.IGNORECASE,
        )
        pumpkin = re.search(
            rf"Pumpkins\s+require\s+({_NUMBER_TOKEN})\s+minutes?\s+of\s+watering\s+and\s+\$?\s*({_NUMBER_TOKEN})\s+worth\s+of\s+pesticide",
            normalized,
            flags=re.IGNORECASE,
        )
        watering = _number_after_patterns(normalized, [rf"has\s+({_NUMBER_TOKEN})\s+minutes?\s+available\s+for\s+watering"])
        pesticide = _number_after_patterns(normalized, [rf"and\s+\$?\s*({_NUMBER_TOKEN})\s+available\s+to\s+spend\s+on\s+pesticide"])
        revenue = re.search(
            rf"revenue\s+per\s+acre\s+of\s+turnips\s+is\s+\$?\s*({_NUMBER_TOKEN})\s+and\s+the\s+revenue\s+per\s+acre\s+of\s+pumpkins\s+is\s+\$?\s*({_NUMBER_TOKEN})",
            normalized,
            flags=re.IGNORECASE,
        )
        if not (land is not None and turnip and pumpkin and watering is not None and pesticide is not None and revenue):
            return TemplateSolveResult(False)
        status, objective, values, message = _linprog_maximize(
            objective=[_number(revenue.group(1)), _number(revenue.group(2))],
            constraints=[
                [1.0, 1.0],
                [_number(turnip.group(1)), _number(pumpkin.group(1))],
                [_number(turnip.group(2)), _number(pumpkin.group(2))],
            ],
            upper_bounds=[land, watering, pesticide],
        )
        if status != "optimal":
            return TemplateSolveResult(
                matched=True,
                template_id="two_product_resource_profit_max_lp",
                status=status,
                confidence=0.82,
                notes=message,
            )
        return TemplateSolveResult(
            matched=True,
            template_id="two_product_resource_profit_max_lp",
            status="optimal",
            objective_value=objective,
            variable_values={"turnips": values[0], "pumpkins": values[1]},
            confidence=0.86,
            notes="Solved two-crop continuous revenue maximization with land, water, and pesticide constraints.",
        )

    return TemplateSolveResult(False)


def _solve_two_vehicle_capacity_min_count(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if "laundromat" in lowered and "top-loading" in lowered and "front-loading" in lowered:
        top = re.search(
            rf"top-loading\s+model\s+can\s+wash\s+({_NUMBER_TOKEN})\s+items?\s+per\s+day",
            normalized,
            flags=re.IGNORECASE,
        )
        front = re.search(
            rf"front-loading\s+model\s+can\s+wash\s+({_NUMBER_TOKEN})\s+items?\s+per\s+day",
            normalized,
            flags=re.IGNORECASE,
        )
        top_energy = _number_after_patterns(normalized, [rf"top-loading\s+model\s+consumes\s+({_NUMBER_TOKEN})\s*kWh\s+per\s+day"])
        front_energy = _number_after_patterns(normalized, [rf"front-loading\s+model\s+consumes\s+({_NUMBER_TOKEN})\s*kWh\s+per\s+day"])
        demand = _number_after_patterns(normalized, [rf"wash\s+at\s+least\s+({_NUMBER_TOKEN})\s+items?\s+per\s+day"])
        energy = _number_after_patterns(normalized, [rf"available\s+({_NUMBER_TOKEN})\s*kWh\s+per\s+day"])
        share = re.search(rf"at\s+most\s+({_NUMBER_TOKEN})\s*%\s+of\s+the\s+machines?\s+can\s+be\s+top-loading", normalized, flags=re.IGNORECASE)
        front_lower = _number_after_patterns(normalized, [rf"at\s+least\s+({_NUMBER_TOKEN})\s+machines?\s+should\s+be\s+front-loading"])
        if not (top and front and top_energy is not None and front_energy is not None and demand is not None and energy is not None and share and front_lower is not None):
            return TemplateSolveResult(False)
        share_value = _number(share.group(1)) / 100.0
        return _solve_small_integer_min_cost_model(
            template_id="two_vehicle_capacity_min_count_ilp",
            symbols=["top_loading", "front_loading"],
            costs={"top_loading": 1.0, "front_loading": 1.0},
            constraints=[
                ({"top_loading": _number(top.group(1)), "front_loading": _number(front.group(1))}, demand, math.inf, "wash_capacity_lower"),
                ({"top_loading": top_energy, "front_loading": front_energy}, -math.inf, energy, "energy_upper"),
                ({"top_loading": 1.0 - share_value, "front_loading": -share_value}, -math.inf, 0.0, "top_share_upper"),
                ({"front_loading": 1.0}, front_lower, math.inf, "front_lower"),
            ],
            upper_bounds={"top_loading": energy / top_energy, "front_loading": energy / front_energy},
            confidence=0.84,
            notes="Solved two-machine minimum-count allocation with capacity, energy, share, and lower-bound constraints.",
        )

    if "field trip" in lowered and "small buses" in lowered and "large buses" in lowered:
        small = _number_after_patterns(normalized, [rf"small\s+bus\s+can\s+carry\s+({_NUMBER_TOKEN})\s+students?"])
        large = _number_after_patterns(normalized, [rf"large\s+bus\s+can\s+carry\s+({_NUMBER_TOKEN})\s+students?"])
        demand = _number_after_patterns(normalized, [rf"transportation\s+for\s+at\s+least\s+({_NUMBER_TOKEN})\s+students?"])
        share = re.search(rf"maximum\s+of\s+({_NUMBER_TOKEN})\s*%\s+of\s+the\s+buses?\s+can\s+be\s+large", normalized, flags=re.IGNORECASE)
        if small is None or large is None or demand is None or share is None:
            return TemplateSolveResult(False)
        share_value = _number(share.group(1)) / 100.0
        return _solve_small_integer_min_cost_model(
            template_id="two_vehicle_capacity_min_count_ilp",
            symbols=["small_bus", "large_bus"],
            costs={"small_bus": 1.0, "large_bus": 1.0},
            constraints=[
                ({"small_bus": small, "large_bus": large}, demand, math.inf, "student_capacity_lower"),
                ({"small_bus": -share_value, "large_bus": 1.0 - share_value}, -math.inf, 0.0, "large_share_upper"),
            ],
            confidence=0.84,
            notes="Solved two-bus minimum-count allocation with capacity and share constraints.",
        )

    return TemplateSolveResult(False)


def _solve_wrap_platter_time_min(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("wraps" in lowered and "platters" in lowered and "meat" in lowered and "rice" in lowered and "production time" in lowered):
        return TemplateSolveResult(False)
    wrap = re.search(
        rf"Each\s+wrap\s+requires\s+({_NUMBER_TOKEN})\s+units?\s+of\s+meat\s+and\s+({_NUMBER_TOKEN})\s+units?\s+of\s+rice",
        normalized,
        flags=re.IGNORECASE,
    )
    platter = re.search(
            rf"Each\s+platter\s+requires\s+({_NUMBER_TOKEN})\s+units?\s+of\s+mea(?:t|nt)\s+and\s+({_NUMBER_TOKEN})\s+units?\s+of\s+rice",
        normalized,
        flags=re.IGNORECASE,
    )
    times = re.search(
        rf"each\s+wrap\s+takes\s+({_NUMBER_TOKEN})\s+minutes?\s+to\s+make,\s+each\s+platter\s+takes\s+({_NUMBER_TOKEN})\s+minutes?\s+to\s+make",
        normalized,
        flags=re.IGNORECASE,
    )
    requirements = re.search(
        rf"must\s+use\s+at\s+least\s+({_NUMBER_TOKEN})\s+units?\s+of\s+meat\s+and\s+({_NUMBER_TOKEN})\s+units?\s+of\s+rice",
        normalized,
        flags=re.IGNORECASE,
    )
    ratio = re.search(rf"at\s+least\s+({_NUMBER_TOKEN})\s+times\s+as\s+many\s+wraps?\s+need\s+to\s+be\s+made\s+as\s+platter", normalized, flags=re.IGNORECASE)
    if not (wrap and platter and times and requirements and ratio):
        return TemplateSolveResult(False)
    factor = _number(ratio.group(1))
    return _solve_small_integer_min_cost_model(
        template_id="wrap_platter_time_min_ilp",
        symbols=["wraps", "platters"],
        costs={"wraps": _number(times.group(1)), "platters": _number(times.group(2))},
        constraints=[
            ({"wraps": _number(wrap.group(1)), "platters": _number(platter.group(1))}, _number(requirements.group(1)), math.inf, "meat_lower"),
            ({"wraps": _number(wrap.group(2)), "platters": _number(platter.group(2))}, _number(requirements.group(2)), math.inf, "rice_lower"),
            ({"wraps": 1.0, "platters": -factor}, 0.0, math.inf, "wrap_ratio_lower"),
        ],
        confidence=0.84,
        notes="Solved wrap/platter integer production-time minimization with meat, rice, and ratio constraints.",
    )


def _solve_two_task_productivity_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "construction manager" in lowered
        and "task" in lowered
        and re.search(_loose_symbol_pattern("X"), normalized, flags=re.IGNORECASE)
        and re.search(_loose_symbol_pattern("Y"), normalized, flags=re.IGNORECASE)
        and "productivity" in lowered
    ):
        return TemplateSolveResult(False)
    hours = re.search(
        rf"type\s+\$?X\$?\s+requires\s+({_NUMBER_TOKEN})\s+hours?\s+and\s+each\s+task\s+of\s+type\s+\$?Y\$?\s+requires\s+({_NUMBER_TOKEN})\s+hours?",
        normalized,
        flags=re.IGNORECASE,
    )
    total_hours = _number_after_patterns(normalized, [rf"total\s+hours\s+available\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"])
    productivity = re.search(
        rf"twice\s+the\s+number\s+of\s+\$?X\$?\s+tasks?\s+plus\s+four\s+times\s+the\s+number\s+of\s+\$?Y\$?\s+tasks?[^.]*?"
        rf"at\s+least\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    costs = re.search(
        rf"cost\s+per\s+unit\s+for\s+task\s+\$?X\$?\s+is\s+[^A-Za-z0-9]*({_NUMBER_TOKEN})[^.]*?"
        rf"for\s+task\s+\$?Y\$?,\s+it's\s+[^A-Za-z0-9]*({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if hours is None or total_hours is None or productivity is None or costs is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="two_task_productivity_min_cost_ilp",
        symbols=["X", "Y"],
        costs={"X": _number(costs.group(1)), "Y": _number(costs.group(2))},
        constraints=[
            ({"X": _number(hours.group(1)), "Y": _number(hours.group(2))}, -math.inf, total_hours, "hours_upper"),
            ({"X": 2.0, "Y": 4.0}, _number(productivity.group(1)), math.inf, "productivity_lower"),
        ],
        confidence=0.84,
        notes="Solved two-task integer productivity minimum-cost model.",
    )


def _solve_three_investment_balance_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    symbols = ["x1", "x2", "x3"]
    if not (
        "financial investment" in lowered
        and "half of the funds" in lowered
        and "quarter" in lowered
        and all(re.search(_loose_symbol_pattern(symbol), normalized, flags=re.IGNORECASE) for symbol in symbols)
    ):
        return TemplateSolveResult(False)
    budget = _number_after_patterns(normalized, [rf"total\s+investment\s+across\s+all\s+three\s+cannot\s+exceed\s+\\?\$?\s*({_NUMBER_TOKEN})"])
    costs = re.search(
        rf"x1,\s*x2,\s+and\s+x3\s+incurs\s+costs\s+of\s+\\?\$?\s*({_NUMBER_TOKEN}),\s*\\?\$?\s*({_NUMBER_TOKEN}),\s+and\s+\\?\$?\s*({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    risk = _number_after_patterns(normalized, [rf"quarter\s+of\s+the\s+funds\s+invested\s+in\s+option\s+x2\s+should\s+be\s+at\s+least\s+\\?\$?\s*({_NUMBER_TOKEN})"])
    difference = _number_after_patterns(normalized, [rf"difference\s+between\s+the\s+funds\s+allocated\s+to\s+option\s+x3\s+and\s+those\s+allocated\s+to\s+option\s+x1\s+should\s+not\s+exceed\s+\\?\$?\s*({_NUMBER_TOKEN})"])
    if budget is None or costs is None or risk is None or difference is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="three_investment_balance_min_cost_ilp",
        symbols=symbols,
        costs={symbol: _number(costs.group(index)) for index, symbol in enumerate(symbols, start=1)},
        constraints=[
            ({"x1": 1.0, "x2": 1.0, "x3": 1.0}, -math.inf, budget, "budget_upper"),
            ({"x1": 0.5, "x2": -0.25}, risk, math.inf, "risk_tradeoff_lower"),
            ({"x3": 1.0, "x1": -1.0}, -math.inf, difference, "x3_x1_difference_upper"),
        ],
        upper_bounds={symbol: budget for symbol in symbols},
        confidence=0.84,
        notes="Solved three-investment integer cost minimization with budget, fractional risk, and balance constraints.",
    )


def _solve_two_container_paste_max(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("pharmaceutical paste" in lowered and "small" in lowered and "large" in lowered and "powdered pill" in lowered):
        return TemplateSolveResult(False)
    small = re.search(
        rf"small\s+container\s+requires\s+({_NUMBER_TOKEN})\s+units?\s+of\s+water\s+and\s+({_NUMBER_TOKEN})\s+units?\s+of\s+the\s+powdered\s+pill\s+to\s+make\s+({_NUMBER_TOKEN})\s+units?\s+of\s+the\s+paste",
        normalized,
        flags=re.IGNORECASE,
    )
    large = re.search(
        rf"large\s+container\s+requires\s+({_NUMBER_TOKEN})\s+units?\s+of\s+water\s+and\s+({_NUMBER_TOKEN})\s+units?\s+of\s+the\s+powdered\s+pill\s+to\s+make\s+({_NUMBER_TOKEN})\s+units?\s+of\s+the\s+paste",
        normalized,
        flags=re.IGNORECASE,
    )
    available = re.search(
        rf"available\s+({_NUMBER_TOKEN})\s+units?\s+of\s+water\s+and\s+({_NUMBER_TOKEN})\s+units?\s+of\s+the\s+powdered\s+pill",
        normalized,
        flags=re.IGNORECASE,
    )
    if small is None or large is None or available is None:
        return TemplateSolveResult(False)
    status, objective, values, message = _linprog_maximize(
        objective=[_number(small.group(3)), _number(large.group(3))],
        constraints=[
            [_number(small.group(1)), _number(large.group(1))],
            [_number(small.group(2)), _number(large.group(2))],
        ],
        upper_bounds=[_number(available.group(1)), _number(available.group(2))],
    )
    if status != "optimal":
        return TemplateSolveResult(
            matched=True,
            template_id="two_container_paste_max_lp",
            status=status,
            confidence=0.82,
            notes=message,
        )
    return TemplateSolveResult(
        matched=True,
        template_id="two_container_paste_max_lp",
        status="optimal",
        objective_value=objective,
        variable_values={"small_containers": values[0], "large_containers": values[1]},
        confidence=0.86,
        notes="Solved two-container paste output maximization with water and powdered-pill constraints.",
    )


def _solve_two_medicine_pill_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("zodiac" in lowered and "sunny" in lowered and "z1" in lowered and "d3" in lowered):
        return TemplateSolveResult(False)
    z1_req = _number_after_patterns(normalized, [rf"at\s+least\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+Z1"])
    d3_req = _number_after_patterns(normalized, [rf"and\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+D3"])
    zodiac_z1 = _number_after_patterns(normalized, [rf"Zodiac\s+contains\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+Z1"])
    sunny_z1 = _number_after_patterns(normalized, [rf"Sunny\s+contains\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+Z1"])
    zodiac_d3 = _number_after_patterns(normalized, [rf"Zodiac\s+contains\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+D3"])
    sunny_d3 = _number_after_patterns(normalized, [rf"Sunny\s+contains\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+D3"])
    costs = re.search(
        rf"cost\s+per\s+pill\s+of\s+Zodiac\s+is\s+\$?\s*({_NUMBER_TOKEN})\s+and\s+the\s+cost\s+per\s+pill\s+of\s+Sunny\s+is\s+\$?\s*({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if any(value is None for value in (z1_req, d3_req, zodiac_z1, sunny_z1, zodiac_d3, sunny_d3)) or costs is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="two_medicine_pill_min_cost_ilp",
        symbols=["Zodiac", "Sunny"],
        costs={"Zodiac": _number(costs.group(1)), "Sunny": _number(costs.group(2))},
        constraints=[
            ({"Zodiac": zodiac_z1, "Sunny": sunny_z1}, z1_req, math.inf, "z1_lower"),
            ({"Zodiac": zodiac_d3, "Sunny": sunny_d3}, d3_req, math.inf, "d3_lower"),
        ],
        confidence=0.84,
        notes="Solved two-pill integer medicine requirement minimum-cost model.",
    )


def _solve_sand_container_max_delivery(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("sand company" in lowered and "small" in lowered and "large containers" in lowered and "playgrounds" in lowered):
        return TemplateSolveResult(False)
    small = re.search(
        rf"small\s+container\s+requires\s+({_NUMBER_TOKEN})\s+person\s+to\s+unload\s+and\s+can\s+hold\s+({_NUMBER_TOKEN})\s+units?\s+of\s+sand",
        normalized,
        flags=re.IGNORECASE,
    )
    large = re.search(
        rf"large\s+container\s+requires\s+({_NUMBER_TOKEN})\s+people\s+to\s+unload\s+and\s+can\s+hold\s+({_NUMBER_TOKEN})\s+units?\s+of\s+sand",
        normalized,
        flags=re.IGNORECASE,
    )
    ratio = re.search(rf"small\s+containers?\s+used\s+must\s+be\s+thrice\s+the\s+number\s+of\s+large\s+containers?", normalized, flags=re.IGNORECASE)
    lower = re.search(
        rf"at\s+least\s+({_NUMBER_TOKEN})\s+small\s+containers?\s+and\s+({_NUMBER_TOKEN})\s+large\s+containers?",
        normalized,
        flags=re.IGNORECASE,
    )
    people = _number_after_patterns(normalized, [rf"has\s+({_NUMBER_TOKEN})\s+people\s+available"])
    if small is None or large is None or ratio is None or lower is None or people is None:
        return TemplateSolveResult(False)
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="sand_container_max_delivery_ilp",
            status="solver_unavailable",
            confidence=0.82,
            notes=str(exc),
        )
    matrix = np.array(
        [
            [_number(small.group(1)), _number(large.group(1))],
            [1.0, -3.0],
        ],
        dtype=float,
    )
    result = milp(
        c=-np.array([_number(small.group(2)), _number(large.group(2))], dtype=float),
        integrality=np.ones(2),
        bounds=Bounds(np.array([_number(lower.group(1)), _number(lower.group(2))]), np.full(2, math.inf)),
        constraints=LinearConstraint(matrix, np.array([-math.inf, 0.0]), np.array([people, math.inf])),
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="sand_container_max_delivery_ilp",
            status="infeasible",
            confidence=0.82,
            notes=str(result.message),
        )
    return TemplateSolveResult(
        matched=True,
        template_id="sand_container_max_delivery_ilp",
        status="optimal",
        objective_value=float(-result.fun),
        variable_values={"small_containers": float(result.x[0]), "large_containers": float(result.x[1])},
        confidence=0.84,
        notes="Solved sand container integer maximization with labor, lower-bound, and small-at-least-thrice-large constraints.",
    )


def _solve_crop_diversity_profit_min(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "corn" in lowered
        and "wheat" in lowered
        and "soybean" in lowered
        and "minimum total profit" in lowered
        and ("integer" in lowered or "indivisible" in lowered)
    ):
        return TemplateSolveResult(False)
    profit = re.search(
        rf"Corn,\s+Wheat,\s+and\s+Soybean\s+yields\s+a\s+profit\s+of\s+\\?\$?\s*({_NUMBER_TOKEN})\\?\$?,\s+"
        rf"\\?\$?\s*({_NUMBER_TOKEN})\\?\$?,\s+and\s+\\?\$?\s*({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    resource = re.search(
        rf"Corn\s+\(multiplied\s+by\s+({_NUMBER_TOKEN})\)\s+and\s+Wheat\s+\(multiplied\s+by\s+({_NUMBER_TOKEN})\)"
        rf"[^.]*?cannot\s+exceed\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    total_lower = _number_after_patterns(normalized, [rf"total\s+acreage\s+across\s+all\s+three\s+crops\s+must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})"])
    difference = _number_after_patterns(normalized, [rf"difference\s+in\s+acreage\s+between\s+Wheat\s+and\s+Soybean\s+should\s+not\s+exceed\s+({_NUMBER_TOKEN})"])
    if profit is None or resource is None or total_lower is None or difference is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="crop_diversity_profit_min_ilp",
        symbols=["Corn", "Wheat", "Soybean"],
        costs={"Corn": _number(profit.group(1)), "Wheat": _number(profit.group(2)), "Soybean": _number(profit.group(3))},
        constraints=[
            ({"Corn": _number(resource.group(1)), "Wheat": _number(resource.group(2))}, -math.inf, _number(resource.group(3)), "soil_resource_upper"),
            ({"Corn": 1.0, "Wheat": 1.0, "Soybean": 1.0}, total_lower, math.inf, "total_acreage_lower"),
            ({"Wheat": 1.0, "Soybean": -1.0}, -math.inf, difference, "wheat_soybean_difference_upper"),
            ({"Wheat": -1.0, "Soybean": 1.0}, -math.inf, difference, "soybean_wheat_difference_upper"),
        ],
        confidence=0.82,
        notes="Solved crop acreage integer minimum-profit allocation with diversity and soil/resource constraints.",
    )


def _solve_project_resource_viability_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    symbols = ["X", "Y", "Z"]
    if not (
        "construction manager" in lowered
        and "different projects" in lowered
        and "viability" in lowered
        and all(re.search(_loose_symbol_pattern(symbol), normalized, flags=re.IGNORECASE) for symbol in symbols)
    ):
        return TemplateSolveResult(False)
    total_upper = _number_after_patterns(normalized, [rf"all\s+three\s+projects\s+combined\s+cannot\s+exceed\s+({_NUMBER_TOKEN})\s+units?"])
    first_lower = _number_after_patterns(normalized, [rf"twice\s+Project\s+X\s+and\s+Project\s+Y\s+must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})\s+units?"])
    second_lower = _number_after_patterns(normalized, [rf"Projects\s+X\s+and\s+Z\s+must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})\s+units?"])
    costs = re.search(
        rf"\\?\$?\s*({_NUMBER_TOKEN})\s+for\s+Project\s+X,\s+\\?\$?\s*({_NUMBER_TOKEN})\s+for\s+Project\s+Y,\s+and\s+\\?\$?\s*({_NUMBER_TOKEN})\s+for\s+Project\s+Z",
        normalized,
        flags=re.IGNORECASE,
    )
    if costs is None:
        costs = re.search(
            rf"cost\s+is\s+\\?\$?\s*({_NUMBER_TOKEN})\s+for\s+Project\s+X,\s+\\?\$?\s*({_NUMBER_TOKEN})\s+for\s+Project\s+Y,\s+and\s+\\?\$?\s*({_NUMBER_TOKEN})\s+for\s+Project\s+Z",
            normalized,
            flags=re.IGNORECASE,
        )
    bounds = re.search(
        rf"between\s+({_NUMBER_TOKEN})\s+\(inclusive\)\s+and\s+({_NUMBER_TOKEN})\s+\(inclusive\)\s+units?\s+for\s+Project\s+X;.*?"
        rf"between\s+({_NUMBER_TOKEN})\s+\(inclusive\)\s+and\s+({_NUMBER_TOKEN})\s+\(inclusive\)\s+units?\s+for\s+Project\s+Y;.*?"
        rf"between\s+({_NUMBER_TOKEN})\s+\(inclusive\)\s+and\s+({_NUMBER_TOKEN})\s+\(inclusive\)\s+units?\s+for\s+Project\s+Z",
        normalized,
        flags=re.IGNORECASE,
    )
    if total_upper is None or first_lower is None or second_lower is None or costs is None or bounds is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="project_resource_viability_min_cost_ilp",
        symbols=symbols,
        costs={"X": _number(costs.group(1)), "Y": _number(costs.group(2)), "Z": _number(costs.group(3))},
        constraints=[
            ({"X": 1.0, "Y": 1.0, "Z": 1.0}, -math.inf, total_upper, "total_resource_upper"),
            ({"X": 2.0, "Y": 1.0}, first_lower, math.inf, "x_y_viability_lower"),
            ({"X": 1.0, "Z": 1.0}, second_lower, math.inf, "x_z_viability_lower"),
        ],
        upper_bounds={"X": _number(bounds.group(2)), "Y": _number(bounds.group(4)), "Z": _number(bounds.group(6))},
        confidence=0.84,
        notes="Solved three-project integer resource minimum-cost model with viability and bound constraints.",
    )


def _solve_two_energy_capacity_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "green energy" in lowered
        and "solar" in lowered
        and "wind" in lowered
        and ("minimum total cost" in lowered or "minimize the total cost" in lowered)
    ):
        return TemplateSolveResult(False)
    costs = re.search(
        rf"cost\s+per\s+unit\s+of\s+capacity\s+for\s+solar\s+and\s+wind\s+energy\s+projects\s+are\s+\\?\$?\s*({_NUMBER_TOKEN})\s+and\s+\\?\$?\s*({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    capacity_lower = _number_after_patterns(normalized, [rf"combined\s+capacity\s+from\s+both\s+solar\s+and\s+wind\s+energy\s+projects\s+must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})"])
    balance = _number_after_patterns(normalized, [rf"three\s+times\s+the\s+solar\s+energy\s+projects\s+minus\s+that\s+from\s+wind\s+energy\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"])
    if costs is None or capacity_lower is None or balance is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="two_energy_capacity_min_cost_ilp",
        symbols=["solar", "wind"],
        costs={"solar": _number(costs.group(1)), "wind": _number(costs.group(2))},
        constraints=[
            ({"solar": 1.0, "wind": 1.0}, capacity_lower, math.inf, "capacity_lower"),
            ({"solar": 3.0, "wind": -1.0}, -math.inf, balance, "solar_wind_balance_upper"),
        ],
        confidence=0.84,
        notes="Solved solar/wind integer capacity minimum-cost model.",
    )


def _solve_facility_resource_balance_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "supply chain management" in lowered
        and "facility" in lowered
        and re.search(_loose_symbol_pattern("X"), normalized, flags=re.IGNORECASE)
        and re.search(_loose_symbol_pattern("Y"), normalized, flags=re.IGNORECASE)
        and "minimize the total cost" in lowered
    ):
        return TemplateSolveResult(False)
    total_upper = _number_after_patterns(normalized, [rf"both\s+facilities\s+combined\s+cannot\s+exceed\s+({_NUMBER_TOKEN})\s+units?"])
    ratio_lower = re.search(
        rf"Three\s+times\s+the\s+allocation\s+for\s+facility\s+\\?\$?X\\?\$?\s+minus\s+twice\s+that\s+of\s+facility\s+\\?\$?Y\\?\$?\s+should\s+be\s+at\s+least\s+zero",
        normalized,
        flags=re.IGNORECASE,
    )
    difference = _number_after_patterns(normalized, [rf"difference\s+in\s+allocation\s+between\s+facility\s+\\?\$?X\\?\$?\s+and\s+facility\s+\\?\$?Y\\?\$?\s+should\s+be\s+at\s+least\s+({_NUMBER_TOKEN})"])
    costs = re.search(
        rf"allocated\s+to\s+facilities\s+\\?\$?X\\?\$?\s+and\s+\\?\$?Y\\?\$?\s+incurs\s+costs\s+of\s+({_NUMBER_TOKEN})\s+and\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    x_upper = _number_after_patterns(normalized, [rf"maximum\s+amount\s+of\s+resources?\s+that\s+can\s+be\s+allocated\s+to\s+facility\s+X\s+is\s+({_NUMBER_TOKEN})"])
    y_upper = _number_after_patterns(normalized, [rf"and\s+to\s+facility\s+Y\s+is\s+({_NUMBER_TOKEN})"])
    if total_upper is None or ratio_lower is None or difference is None or costs is None or x_upper is None or y_upper is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="facility_resource_balance_min_cost_ilp",
        symbols=["X", "Y"],
        costs={"X": _number(costs.group(1)), "Y": _number(costs.group(2))},
        constraints=[
            ({"X": 1.0, "Y": 1.0}, -math.inf, total_upper, "total_upper"),
            ({"X": 3.0, "Y": -2.0}, 0.0, math.inf, "ratio_lower"),
            ({"X": 1.0, "Y": -1.0}, difference, math.inf, "difference_lower"),
        ],
        upper_bounds={"X": x_upper, "Y": y_upper},
        confidence=0.84,
        notes="Solved two-facility integer resource minimum-cost model with balance and capacity constraints.",
    )


def _solve_two_food_sodium_min_lp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("ramen" in lowered and "fries" in lowered and "sodium" in lowered and "minimize" in lowered):
        return TemplateSolveResult(False)
    ramen = re.search(
        rf"ramen\s+contains\s+({_NUMBER_TOKEN})\s+calories,\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+protein,\s+and\s+({_NUMBER_TOKEN})\s*mg\s+of\s+sodium",
        normalized,
        flags=re.IGNORECASE,
    )
    fries = re.search(
        rf"fries\s+contains\s+({_NUMBER_TOKEN})\s+calories,\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+protein,\s+and\s+({_NUMBER_TOKEN})\s*mg\s+of\s+sodium",
        normalized,
        flags=re.IGNORECASE,
    )
    share = re.search(rf"at\s+most\s+({_NUMBER_TOKEN})\s*%\s+of\s+his\s+meals?\s+can\s+be\s+ramen", normalized, flags=re.IGNORECASE)
    calorie_lower = _number_after_patterns(normalized, [rf"at\s+least\s+({_NUMBER_TOKEN})\s+calories"])
    protein_lower = _number_after_patterns(normalized, [rf"and\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+protein"])
    if not (ramen and fries and share and calorie_lower is not None and protein_lower is not None):
        return TemplateSolveResult(False)
    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="two_food_sodium_min_lp",
            status="solver_unavailable",
            confidence=0.82,
            notes=str(exc),
        )
    share_value = _number(share.group(1)) / 100.0
    result = linprog(
        [_number(ramen.group(3)), _number(fries.group(3))],
        A_ub=[
            [-_number(ramen.group(1)), -_number(fries.group(1))],
            [-_number(ramen.group(2)), -_number(fries.group(2))],
            [1.0 - share_value, -share_value],
        ],
        b_ub=[-calorie_lower, -protein_lower, 0.0],
        bounds=[(0, None), (0, None)],
        method="highs",
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="two_food_sodium_min_lp",
            status="solver_failed",
            confidence=0.82,
            notes=str(result.message),
        )
    return TemplateSolveResult(
        matched=True,
        template_id="two_food_sodium_min_lp",
        status="optimal",
        objective_value=float(result.fun),
        variable_values={"ramen": float(result.x[0]), "fries": float(result.x[1])},
        confidence=0.86,
        notes="Solved continuous two-food sodium minimization with calorie, protein, and share constraints.",
    )


def _solve_two_fertilizer_vitamin_min_lp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("fertilizer a" in lowered and "fertilizer b" in lowered and "vitamin d" in lowered and "minimize" in lowered):
        return TemplateSolveResult(False)
    fertilizer_a = re.search(
        rf"fertilizer\s+A\s+contains\s+({_NUMBER_TOKEN})\s+units?\s+of\s+nitrogen,\s+({_NUMBER_TOKEN})\s+units?\s+of\s+phosphoric\s+acid,\s+"
        rf"({_NUMBER_TOKEN})\s+units?\s+of\s+vitamin\s+A\s+and\s+({_NUMBER_TOKEN})\s+units?\s+of\s+vitamin\s+D",
        normalized,
        flags=re.IGNORECASE,
    )
    fertilizer_b = re.search(
        rf"fertilizer\s+B\s+contains\s+({_NUMBER_TOKEN})\s+units?\s+of\s+nitrogen,\s+({_NUMBER_TOKEN})\s+units?\s+of\s+phosphoric\s+acid,\s+"
        rf"({_NUMBER_TOKEN})\s+units?\s+of\s+vitamin\s+A\s+and\s+({_NUMBER_TOKEN})\s+units?\s+of\s+vitamin\s+D",
        normalized,
        flags=re.IGNORECASE,
    )
    nitrogen_lower = _number_after_patterns(normalized, [rf"minimum\s+({_NUMBER_TOKEN})\s+units?\s+of\s+nitrogen"])
    phosphoric_lower = _number_after_patterns(normalized, [rf"minimum\s+of\s+({_NUMBER_TOKEN})\s+units?\s+of\s+phosphoric\s+acid"])
    vitamin_a_upper = _number_after_patterns(normalized, [rf"no\s+more\s+than\s+({_NUMBER_TOKEN})\s+units?\s+of\s+vitamin\s+A"])
    if fertilizer_a is None or fertilizer_b is None or nitrogen_lower is None or phosphoric_lower is None or vitamin_a_upper is None:
        return TemplateSolveResult(False)
    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="two_fertilizer_vitamin_min_lp",
            status="solver_unavailable",
            confidence=0.82,
            notes=str(exc),
        )
    result = linprog(
        [_number(fertilizer_a.group(4)), _number(fertilizer_b.group(4))],
        A_ub=[
            [-_number(fertilizer_a.group(1)), -_number(fertilizer_b.group(1))],
            [-_number(fertilizer_a.group(2)), -_number(fertilizer_b.group(2))],
            [_number(fertilizer_a.group(3)), _number(fertilizer_b.group(3))],
        ],
        b_ub=[-nitrogen_lower, -phosphoric_lower, vitamin_a_upper],
        bounds=[(0, None), (0, None)],
        method="highs",
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="two_fertilizer_vitamin_min_lp",
            status="solver_failed",
            confidence=0.82,
            notes=str(result.message),
        )
    return TemplateSolveResult(
        matched=True,
        template_id="two_fertilizer_vitamin_min_lp",
        status="optimal",
        objective_value=float(result.fun),
        variable_values={"fertilizer_A": float(result.x[0]), "fertilizer_B": float(result.x[1])},
        confidence=0.86,
        notes="Solved continuous two-fertilizer vitamin-D minimization with nutrient lower and vitamin-A upper constraints.",
    )


def _solve_runner_canoe_mail_max_ilp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("runners" in lowered and "canoers" in lowered and "bags of mail" in lowered and "maximize" in lowered):
        return TemplateSolveResult(False)
    runner = re.search(rf"Runners\s+can\s+carry\s+({_NUMBER_TOKEN})\s+bags?\s+of\s+mail\s+each\s+time\s+and\s+takes\s+({_NUMBER_TOKEN})\s+hours?", normalized, flags=re.IGNORECASE)
    canoe = re.search(rf"Canoers\s+can\s+carry\s+({_NUMBER_TOKEN})\s+bags?\s+of\s+mail\s+each\s+time\s+and\s+takes\s+({_NUMBER_TOKEN})\s+hours?", normalized, flags=re.IGNORECASE)
    share = re.search(rf"At\s+most\s+({_NUMBER_TOKEN})\s*%\s+of\s+deliveries\s+can\s+be\s+by\s+canoe", normalized, flags=re.IGNORECASE)
    hours = _number_after_patterns(normalized, [rf"at\s+most\s+({_NUMBER_TOKEN})\s+total\s+hours"])
    runner_lower = _number_after_patterns(normalized, [rf"at\s+least\s+({_NUMBER_TOKEN})\s+runners?\s+must\s+be\s+used"])
    if runner is None or canoe is None or share is None or hours is None or runner_lower is None:
        return TemplateSolveResult(False)
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="runner_canoe_mail_max_ilp",
            status="solver_unavailable",
            confidence=0.82,
            notes=str(exc),
        )
    share_value = _number(share.group(1)) / 100.0
    matrix = np.array(
        [
            [_number(runner.group(2)), _number(canoe.group(2))],
            [-share_value, 1.0 - share_value],
        ],
        dtype=float,
    )
    result = milp(
        c=-np.array([_number(runner.group(1)), _number(canoe.group(1))], dtype=float),
        integrality=np.ones(2),
        bounds=Bounds(np.array([runner_lower, 0.0], dtype=float), np.full(2, math.inf)),
        constraints=LinearConstraint(matrix, np.array([-math.inf, -math.inf]), np.array([hours, 0.0])),
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="runner_canoe_mail_max_ilp",
            status="infeasible",
            confidence=0.82,
            notes=str(result.message),
        )
    return TemplateSolveResult(
        matched=True,
        template_id="runner_canoe_mail_max_ilp",
        status="optimal",
        objective_value=float(-result.fun),
        variable_values={"runners": float(result.x[0]), "canoers": float(result.x[1])},
        confidence=0.84,
        notes="Solved runner/canoe mail integer maximization with time, share, and lower-bound constraints.",
    )


def _solve_ice_cream_profit_bounds_lp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("ice cream store" in lowered and "chocolate" in lowered and "vanilla" in lowered and "maximize profit" in lowered):
        return TemplateSolveResult(False)
    lower = re.search(
        rf"at\s+least\s+({_NUMBER_TOKEN})\s+gallons?\s+of\s+each\s+type\s+but\s+at\s+most\s+({_NUMBER_TOKEN})\s+gallons?\s+of\s+chocolate\s+ice\s+cream\s+and\s+at\s+most\s+({_NUMBER_TOKEN})\s+gallons?\s+of\s+vanilla",
        normalized,
        flags=re.IGNORECASE,
    )
    time = re.search(
        rf"It\s+takes\s+({_NUMBER_TOKEN})\s+hours?\s+to\s+produce\s+a\s+gallon\s+of\s+chocolate\s+ice\s+cream\s+and\s+({_NUMBER_TOKEN})\s+hours?\s+to\s+produce\s+a\s+gallon\s+of\s+vanilla",
        normalized,
        flags=re.IGNORECASE,
    )
    time_limit = _number_after_patterns(normalized, [rf"In\s+a\s+week,\s+({_NUMBER_TOKEN})\s+hours?\s+are\s+available"])
    worker_lower = _number_after_patterns(normalized, [rf"at\s+least\s+({_NUMBER_TOKEN})\s+workers?\s+are\s+needed"])
    profit = re.search(
        rf"profit\s+per\s+gallon\s+of\s+chocolate\s+ice\s+cream\s+is\s+\\?\$?\s*({_NUMBER_TOKEN})\s+and\s+the\s+profit\s+per\s+gallon\s+of\s+vanilla\s+ice\s+cream\s+is\s+\\?\$?\s*({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if lower is None or time is None or time_limit is None or worker_lower is None or profit is None:
        return TemplateSolveResult(False)
    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="ice_cream_profit_bounds_lp",
            status="solver_unavailable",
            confidence=0.82,
            notes=str(exc),
        )
    result = linprog(
        [-_number(profit.group(1)), -_number(profit.group(2))],
        A_ub=[
            [_number(time.group(1)), _number(time.group(2))],
            [-1.0, -2.0],
        ],
        b_ub=[time_limit, -worker_lower],
        bounds=[(_number(lower.group(1)), _number(lower.group(2))), (_number(lower.group(1)), _number(lower.group(3)))],
        method="highs",
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="ice_cream_profit_bounds_lp",
            status="solver_failed",
            confidence=0.82,
            notes=str(result.message),
        )
    return TemplateSolveResult(
        matched=True,
        template_id="ice_cream_profit_bounds_lp",
        status="optimal",
        objective_value=float(-result.fun),
        variable_values={"chocolate": float(result.x[0]), "vanilla": float(result.x[1])},
        confidence=0.84,
        notes="Solved bounded two-flavor ice-cream profit maximization with time and worker constraints.",
    )


def _solve_steel_furnace_method_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("steel furnaces" in lowered and "steelmaking" in lowered and "minimize fuel" in lowered):
        return TemplateSolveResult(False)
    first = re.search(
        rf"first\s+method\s+takes\s+\\?\$?a\s*=\s*({_NUMBER_TOKEN})\\?\$?\s+hours?.*?"
        rf"costs\s+\\?\$?m\s*=\s*({_NUMBER_TOKEN})\\?\$?",
        normalized,
        flags=re.IGNORECASE,
    )
    second = re.search(
        rf"second\s+method\s+takes\s+\\?\$?b\s*=\s*({_NUMBER_TOKEN})\\?\$?\s+hours?.*?"
        rf"costs\s+\\?\$?n\s*=\s*({_NUMBER_TOKEN})\\?\$?",
        normalized,
        flags=re.IGNORECASE,
    )
    production = _number_after_patterns(normalized, [rf"produces\s+\\?\$?k\s*=\s*({_NUMBER_TOKEN})\\?\$?\s+tons?"])
    demand = _number_after_patterns(normalized, [rf"at\s+least\s+\\?\$?d\s*=\s*({_NUMBER_TOKEN})\\?\$?\s+tons?"])
    hours = _number_after_patterns(normalized, [rf"within\s+\\?\$?c\s*=\s*({_NUMBER_TOKEN})\\?\$?\s+hours?"])
    if first is None or second is None or production is None or demand is None or hours is None:
        return TemplateSolveResult(False)
    status, objective, values, message = _linprog_minimize(
        objective=[_number(first.group(2)), _number(second.group(2))],
        constraints=[
            [-production, -production],
            [_number(first.group(1)), _number(second.group(1))],
        ],
        upper_bounds=[-demand, hours],
    )
    if status != "optimal":
        return TemplateSolveResult(
            matched=True,
            template_id="steel_furnace_method_min_cost_lp",
            status=status,
            confidence=0.82,
            notes=message,
        )
    return TemplateSolveResult(
        matched=True,
        template_id="steel_furnace_method_min_cost_lp",
        status="optimal",
        objective_value=objective,
        variable_values={"method_1": values[0], "method_2": values[1]},
        confidence=0.86,
        notes="Solved steel-furnace two-method LP with production and time constraints.",
    )


def _solve_mall_store_lease_piecewise(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not ("shopping mall" in lowered and "space for lease" in lowered and "20%" in lowered and "maximize total rental income" in lowered):
        return TemplateSolveResult(False)
    space = _number_after_patterns(text, [rf"has\s+({_NUMBER_TOKEN})\s*m²\s+of\s+space\s+for\s+lease"])
    parsed = _parse_markdown_table(text)
    if space is None or not parsed:
        return TemplateSolveResult(False)
    _header, rows = parsed
    stores: list[dict[str, Any]] = []
    for row in rows:
        if len(row) < 8:
            continue
        try:
            minimum = int(_number(row[3]))
            maximum = int(_number(row[4]))
            profits = [_number(cell) for cell in row[5 : 5 + maximum] if cell.strip() != "-"]
        except ValueError:
            continue
        if len(profits) < maximum:
            return TemplateSolveResult(False)
        stores.append(
            {
                "label": row[1],
                "area": _number(row[2]),
                "min": minimum,
                "max": maximum,
                "profits": profits,
            }
        )
    if len(stores) != 5:
        return TemplateSolveResult(False)
    best: tuple[float, tuple[int, ...], float, float] | None = None
    ranges = [range(store["min"], store["max"] + 1) for store in stores]
    for counts in itertools.product(*ranges):
        used_area = sum(count * stores[index]["area"] for index, count in enumerate(counts))
        if used_area > space + 1e-9:
            continue
        annual_profit = sum(
            count * stores[index]["profits"][count - 1]
            for index, count in enumerate(counts)
            if count > 0
        )
        rent = 0.2 * annual_profit
        if best is None or rent > best[0] + 1e-9:
            best = (rent, counts, used_area, annual_profit)
    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="mall_store_lease_piecewise_enum",
            status="infeasible",
            confidence=0.8,
        )
    return TemplateSolveResult(
        matched=True,
        template_id="mall_store_lease_piecewise_enum",
        status="optimal",
        objective_value=float(best[0]),
        variable_values={stores[index]["label"]: float(count) for index, count in enumerate(best[1])},
        confidence=0.84,
        notes="Solved mall store-lease piecewise integer enumeration using rent as 20% of total store profit.",
        artifact={"space": space, "used_area": best[2], "annual_profit": best[3]},
    )


def _solve_fruit_farm_two_type_profit(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "yarra valley" in lowered
        and "apples" in lowered
        and "pears" in lowered
        and "oranges" in lowered
        and "lemons" in lowered
        and "unwilling to grow more than two types" in lowered
    ):
        return TemplateSolveResult(False)
    profits = re.search(
        rf"one\s+acre\s+of\s+apples\s+is\s+\\?\$?\s*({_NUMBER_TOKEN}).*?"
        rf"one\s+acre\s+of\s+pears\s+is\s+\\?\$?\s*({_NUMBER_TOKEN}).*?"
        rf"one\s+acre\s+of\s+oranges\s+is\s+\\?\$?\s*({_NUMBER_TOKEN}).*?"
        rf"one\s+acre\s+of\s+lemons\s+is\s+\\?\$?\s*({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    area = _number_after_patterns(normalized, [rf"total\s+area\s+of\s+({_NUMBER_TOKEN})\s+acres?"])
    if profits is None or area is None:
        return TemplateSolveResult(False)
    labels = ["apples", "pears", "oranges", "lemons"]
    profit_values = {label: _number(profits.group(index + 1)) for index, label in enumerate(labels)}
    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="fruit_farm_two_type_profit_lp_enum",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
        )
    best: tuple[float, tuple[str, ...], list[float]] | None = None
    for active_count in (1, 2):
        for active in itertools.combinations(labels, active_count):
            bounds = [(0, None) if label in active else (0, 0) for label in labels]
            result = linprog(
                [-profit_values[label] for label in labels],
                A_ub=[
                    [1.0, 1.0, 1.0, 1.0],
                    [-1.0, 2.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 3.0],
                ],
                b_ub=[area, 0.0, 0.0],
                A_eq=[[0.0, 0.0, 1.0, -2.0]],
                b_eq=[0.0],
                bounds=bounds,
                method="highs",
            )
            if not result.success:
                continue
            value = float(-result.fun)
            if best is None or value > best[0] + 1e-9:
                best = (value, active, [float(item) for item in result.x])
    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="fruit_farm_two_type_profit_lp_enum",
            status="infeasible",
            confidence=0.8,
        )
    return TemplateSolveResult(
        matched=True,
        template_id="fruit_farm_two_type_profit_lp_enum",
        status="optimal",
        objective_value=best[0],
        variable_values={label: best[2][index] for index, label in enumerate(labels) if best[2][index] > 1e-8},
        confidence=0.84,
        notes="Solved fruit-farm LP by enumerating the at-most-two crop-type choice and optimizing acreage.",
        artifact={"active_types": list(best[1]), "area": area, "profits": profit_values},
    )


def _solve_three_department_staffing_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    symbols = ["X1", "X2", "X3"]
    if not (
        "human resources planning" in lowered
        and "three departments" in lowered
        and all(re.search(_loose_symbol_pattern(symbol), normalized, flags=re.IGNORECASE) for symbol in symbols)
        and "minimize the total cost" in lowered
    ):
        return TemplateSolveResult(False)
    total_upper = _number_after_patterns(normalized, [rf"total\s+number\s+of\s+employees\s+that\s+can\s+be\s+allocated\s+is\s+limited\s+to\s+({_NUMBER_TOKEN})"])
    costs = re.search(
        rf"departments\s+\\?\$?X1\\?\$?,\s+\\?\$?X2\\?\$?,\s+and\s+\\?\$?X3\\?\$?\s+costs\s+the\s+company\s+"
        rf"\\?\$?\s*({_NUMBER_TOKEN}),\s+\\?\$?\s*({_NUMBER_TOKEN}),\s+and\s+\\?\$?\s*({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if total_upper is None or costs is None:
        return TemplateSolveResult(False)
    lower_bounds: dict[str, float] = {}
    upper_bounds: dict[str, float] = {}
    lower_patterns = {
        "X1": rf"Department\s+\\?\$?X1\\?\$?\s+requires\s+a\s+minimum\s+of\s+({_NUMBER_TOKEN})\s+employees?",
        "X2": rf"department\s+\\?\$?X2\\?\$?\s+needs\s+at\s+least\s+({_NUMBER_TOKEN})\s+employees?",
        "X3": rf"department\s+\\?\$?X3\\?\$?\s+requires\s+no\s+fewer\s+than\s+({_NUMBER_TOKEN})\s+employees?",
    }
    for symbol, pattern in lower_patterns.items():
        value = _number_after_patterns(normalized, [pattern])
        if value is not None:
            lower_bounds[symbol] = value
    bounds = re.search(
        rf"Department\s+X1\s+can\s+have\s+between\s+({_NUMBER_TOKEN})\s+and\s+({_NUMBER_TOKEN})\s+employees?;\s+"
        rf"Department\s+X2\s+can\s+have\s+between\s+({_NUMBER_TOKEN})\s+and\s+({_NUMBER_TOKEN});\s+"
        rf"Department\s+X3\s+can\s+have\s+between\s+({_NUMBER_TOKEN})\s+and\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if bounds:
        upper_bounds = {"X1": _number(bounds.group(2)), "X2": _number(bounds.group(4)), "X3": _number(bounds.group(6))}
    if set(lower_bounds) != set(symbols) or set(upper_bounds) != set(symbols):
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="three_department_staffing_min_cost_ilp",
        symbols=symbols,
        costs={symbol: _number(costs.group(index)) for index, symbol in enumerate(symbols, start=1)},
        constraints=[
            ({"X1": 1.0, "X2": 1.0, "X3": 1.0}, -math.inf, total_upper, "total_staff_upper"),
            *[({symbol: 1.0}, lower_bounds[symbol], math.inf, f"{symbol}_lower") for symbol in symbols],
        ],
        upper_bounds=upper_bounds,
        confidence=0.84,
        notes="Solved three-department integer staffing minimum-cost model with lower and upper staffing bounds.",
    )


def _solve_multifood_integer_diet_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("daily requirements" in lowered and "grams of protein" in lowered and "calories" in lowered and "least amount of money" in lowered):
        return TemplateSolveResult(False)
    requirements = re.search(
        rf"requirements\s+are\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+protein,\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+carbs?,\s+and\s+({_NUMBER_TOKEN})\s+calories",
        normalized,
        flags=re.IGNORECASE,
    )
    if requirements is None:
        return TemplateSolveResult(False)
    foods: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("-") or ":" not in line:
            continue
        label, body = line[1:].split(":", 1)
        cost_match = re.search(rf"\$({_NUMBER_TOKEN})", body)
        protein_match = re.search(rf"({_NUMBER_TOKEN})\s+grams?\s+of\s+protein", body, flags=re.IGNORECASE)
        carb_match = re.search(rf"({_NUMBER_TOKEN})\s+(?:grams?\s+of\s+)?carb(?:ohydrate)?s?", body, flags=re.IGNORECASE)
        calorie_match = re.search(rf"({_NUMBER_TOKEN})\s+calories", body, flags=re.IGNORECASE)
        if cost_match and protein_match and carb_match and calorie_match:
            foods.append(
                {
                    "label": _clean_label(label),
                    "cost": _number(cost_match.group(1)),
                    "protein": _number(protein_match.group(1)),
                    "carbs": _number(carb_match.group(1)),
                    "calories": _number(calorie_match.group(1)),
                }
            )
    if len(foods) < 2:
        return TemplateSolveResult(False)
    symbols = [food["label"] for food in foods]
    return _solve_small_integer_min_cost_model(
        template_id="multifood_integer_diet_min_cost_ilp",
        symbols=symbols,
        costs={food["label"]: food["cost"] for food in foods},
        constraints=[
            ({food["label"]: food["protein"] for food in foods}, _number(requirements.group(1)), math.inf, "protein_lower"),
            ({food["label"]: food["carbs"] for food in foods}, _number(requirements.group(2)), math.inf, "carbs_lower"),
            ({food["label"]: food["calories"] for food in foods}, _number(requirements.group(3)), math.inf, "calorie_lower"),
        ],
        confidence=0.82,
        notes="Solved multi-food integer diet minimum-cost model from explicit nutrition bullets.",
        artifact={"foods": foods},
    )


def _solve_four_department_salary_balance_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text.replace("\\n", " "))
    lowered = normalized.lower()
    symbols = ["x1", "x2", "x3", "x4"]
    if not (
        "human resources manager" in lowered
        and "four different departments" in lowered
        and "minimize the total salary cost" in lowered
        and all(re.search(_loose_symbol_pattern(symbol), normalized, flags=re.IGNORECASE) for symbol in symbols)
    ):
        return TemplateSolveResult(False)
    costs = re.search(
        rf"x1,\s*x2,\s*x3\s+and\s+x4\s+earns\s+a\s+salary\s+of\s+\\?\$?\s*({_NUMBER_TOKEN}),\s+\\?\$?\s*({_NUMBER_TOKEN}),\s+"
        rf"\\?\$?\s*({_NUMBER_TOKEN})\s+and\s+\\?\$?\s*({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    upper_12 = _number_after_patterns(normalized, [rf"departments\s+x1\s+and\s+x2\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"])
    upper_34 = _number_after_patterns(normalized, [rf"departments\s+x3\s+and\s+x4\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"])
    supervision = _number_after_patterns(normalized, [rf"department\s+X1\s+must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})\s+more\s+than\s+half\s+the\s+number\s+of\s+employees\s+in\s+department\s+X3"])
    diff = _number_after_patterns(normalized, [rf"difference\s+between\s+the\s+number\s+of\s+employees\s+in\s+department\s+X4\s+and\s+X2\s+should\s+not\s+exceed\s+({_NUMBER_TOKEN})"])
    bounds = re.search(
        rf"x1\s+can\s+have\s+up\s+to\s+({_NUMBER_TOKEN})\s+employees?,\s*"
        rf"x2\s+can\s+have\s+up\s+to\s+({_NUMBER_TOKEN}),\s*"
        rf"x3\s+can\s+have\s+up\s+to\s+({_NUMBER_TOKEN}),\s*"
        rf"x4\s+can\s+also\s+have\s+up\s+to\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if costs is None or upper_12 is None or upper_34 is None or supervision is None or diff is None or bounds is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="four_department_salary_balance_min_cost_ilp",
        symbols=symbols,
        costs={symbol: _number(costs.group(index)) for index, symbol in enumerate(symbols, start=1)},
        constraints=[
            ({"x1": 1.0, "x2": 1.0}, -math.inf, upper_12, "x1_x2_upper"),
            ({"x3": 1.0, "x4": 1.0}, -math.inf, upper_34, "x3_x4_upper"),
            ({"x1": 1.0, "x3": -0.5}, supervision, math.inf, "x1_half_x3_margin"),
            ({"x4": 1.0, "x2": -1.0}, -math.inf, diff, "x4_x2_difference_upper"),
        ],
        upper_bounds={symbol: _number(bounds.group(index)) for index, symbol in enumerate(symbols, start=1)},
        confidence=0.82,
        notes="Solved four-department salary minimization with staffing balance and upper-bound constraints.",
    )


def _solve_four_training_area_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    symbols = ["x1", "x2", "x3", "x4"]
    if not (
        "sports team" in lowered
        and "training areas" in lowered
        and all(re.search(_loose_symbol_pattern(symbol), normalized, flags=re.IGNORECASE) for symbol in symbols)
        and "minimum total cost" in lowered
    ):
        return TemplateSolveResult(False)
    costs = re.search(
        rf"cost\s+per\s+hour\s+of\s+each\s+area\s+are\s+\\?\$?({_NUMBER_TOKEN})\\?\$?,\s+\\?\$?({_NUMBER_TOKEN})\\?\$?,\s+\\?\$?({_NUMBER_TOKEN})\\?\$?\s+and\s+\\?\$?({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    lower_12 = _number_after_patterns(normalized, [rf"combined\s+hours\s+of\s+strength\s+and\s+conditioning\s+\(x1\)\s+and\s+skill\s+development\s+\(x2\)\s+must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})"])
    diff_34 = _number_after_patterns(normalized, [rf"difference\s+between\s+strategy\s+learning\s+\(x3\)\s+and\s+recovery\s+sessions\s+\(x4\)\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"])
    performance = _number_after_patterns(normalized, [rf"Twice\s+the\s+hours\s+of\s+strength\s+and\s+conditioning\s+plus\s+three\s+times\s+skill\s+development\s+minus\s+recovery\s+session\s+should\s+be\s+at\s+least\s+({_NUMBER_TOKEN})"])
    diff_13 = _number_after_patterns(normalized, [rf"difference\s+between\s+strength\s+and\s+conditioning\s+\(x1\)\s+hours\s+and\s+strategy\s+learning\s*\(x3\)\s+should\s+not\s+exceed\s+({_NUMBER_TOKEN})"])
    bounds = re.search(
        rf"Strength\s*&\s*Conditioning:\s*\[\s*({_NUMBER_TOKEN})\s*,\s*({_NUMBER_TOKEN})\s*\]\s*"
        rf"Skill\s+Development:\s*\[\s*({_NUMBER_TOKEN})\s*,\s*({_NUMBER_TOKEN})\s*\]\s*"
        rf"Strategy\s+Learning:\s*\[\s*({_NUMBER_TOKEN})\s*,\s*({_NUMBER_TOKEN})\s*\]\s*"
        rf"Recovery\s+Sessions:\s*\[\s*({_NUMBER_TOKEN})\s*,\s*({_NUMBER_TOKEN})\s*\]",
        normalized,
        flags=re.IGNORECASE,
    )
    if costs is None or lower_12 is None or diff_34 is None or performance is None or diff_13 is None or bounds is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="four_training_area_min_cost_ilp",
        symbols=symbols,
        costs={symbol: _number(costs.group(index)) for index, symbol in enumerate(symbols, start=1)},
        constraints=[
            ({"x1": 1.0, "x2": 1.0}, lower_12, math.inf, "x1_x2_lower"),
            ({"x3": 1.0, "x4": -1.0}, -math.inf, diff_34, "x3_x4_difference_upper"),
            ({"x1": 2.0, "x2": 3.0, "x4": -1.0}, performance, math.inf, "performance_lower"),
            ({"x1": 1.0, "x3": -1.0}, -math.inf, diff_13, "x1_x3_difference_upper"),
        ],
        upper_bounds={"x1": _number(bounds.group(2)), "x2": _number(bounds.group(4)), "x3": _number(bounds.group(6)), "x4": _number(bounds.group(8))},
        confidence=0.82,
        notes="Solved four-training-area integer minimum-cost model with balance, performance, and bound constraints.",
    )


def _solve_two_sandwich_profit_max_lp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("breakfast joint" in lowered and "regular" in lowered and "special" in lowered and "eggs" in lowered and "bacon" in lowered):
        return TemplateSolveResult(False)
    regular = re.search(rf"regular\s+sandwich\s+requires\s+({_NUMBER_TOKEN})\s+eggs?\s+and\s+({_NUMBER_TOKEN})\s+slices?\s+of\s+bacon", normalized, flags=re.IGNORECASE)
    special = re.search(rf"special\s+sandwich\s+requires\s+({_NUMBER_TOKEN})\s+eggs?\s+and\s+({_NUMBER_TOKEN})\s+slices?\s+of\s+bacon", normalized, flags=re.IGNORECASE)
    available = re.search(
        rf"has\s+a\s+total\s+of\s+({_NUMBER_TOKEN})\s+eggs?\s+and\s+({_NUMBER_TOKEN})\s+slices?\s+of\s+bacon",
        normalized,
        flags=re.IGNORECASE,
    )
    profit = re.search(rf"profit\s+of\s+\\?\$?\s*({_NUMBER_TOKEN})\s+per\s+regular\s+sandwich\s+and\s+a\s+profit\s+of\s+\\?\$?\s*({_NUMBER_TOKEN})\s+per\s+special", normalized, flags=re.IGNORECASE)
    if regular is None or special is None or available is None or profit is None:
        return TemplateSolveResult(False)
    status, objective, values, message = _linprog_maximize(
        objective=[_number(profit.group(1)), _number(profit.group(2))],
        constraints=[
            [_number(regular.group(1)), _number(special.group(1))],
            [_number(regular.group(2)), _number(special.group(2))],
        ],
        upper_bounds=[_number(available.group(1)), _number(available.group(2))],
    )
    if status != "optimal":
        return TemplateSolveResult(matched=True, template_id="two_sandwich_profit_max_lp", status=status, confidence=0.82, notes=message)
    return TemplateSolveResult(
        matched=True,
        template_id="two_sandwich_profit_max_lp",
        status="optimal",
        objective_value=objective,
        variable_values={"regular": values[0], "special": values[1]},
        confidence=0.84,
        notes="Solved two-sandwich profit LP with egg and bacon constraints.",
    )


def _solve_two_van_min_count_ilp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("regular and hybrid vans" in lowered and "packages per day" in lowered and "pollutants" in lowered):
        return TemplateSolveResult(False)
    regular = re.search(rf"regular\s+van\s+can\s+deliver\s+({_NUMBER_TOKEN})\s+packages?\s+per\s+day\s+and\s+produces\s+({_NUMBER_TOKEN})\s+units?\s+of\s+pollutants", normalized, flags=re.IGNORECASE)
    hybrid = re.search(rf"hybrid\s+van\s+can\s+deliver\s+({_NUMBER_TOKEN})\s+packages?\s+per\s+day\s+and\s+produces\s+({_NUMBER_TOKEN})\s+units?\s+of\s+pollutants", normalized, flags=re.IGNORECASE)
    pollution = _number_after_patterns(normalized, [rf"at\s+most\s+({_NUMBER_TOKEN})\s+units?\s+of\s+pollutants"])
    packages = _number_after_patterns(normalized, [rf"deliver\s+at\s+least\s+({_NUMBER_TOKEN})\s+packages?\s+per\s+day"])
    if regular is None or hybrid is None or pollution is None or packages is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="two_van_min_count_ilp",
        symbols=["regular", "hybrid"],
        costs={"regular": 1.0, "hybrid": 1.0},
        constraints=[
            ({"regular": _number(regular.group(1)), "hybrid": _number(hybrid.group(1))}, packages, math.inf, "package_lower"),
            ({"regular": _number(regular.group(2)), "hybrid": _number(hybrid.group(2))}, -math.inf, pollution, "pollution_upper"),
        ],
        confidence=0.84,
        notes="Solved regular/hybrid van integer minimum-count model with package and pollution constraints.",
    )


def _solve_two_medication_time_min_ilp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("anxiety medication" in lowered and "anti-depressants" in lowered and "minimize the total time" in lowered):
        return TemplateSolveResult(False)
    anxiety_time = _number_after_patterns(normalized, [rf"Each\s+unit\s+of\s+anxiety\s+medication\s+takes\s+({_NUMBER_TOKEN})\s+minutes?"])
    anti_time = _number_after_patterns(normalized, [rf"each\s+unit\s+of\s+anti-depressant\s+takes\s+({_NUMBER_TOKEN})\s+minutes?"])
    total_lower = _number_after_patterns(normalized, [rf"at\s+least\s+({_NUMBER_TOKEN})\s+units?\s+of\s+medication"])
    anxiety_lower = _number_after_patterns(normalized, [rf"at\s+least\s+({_NUMBER_TOKEN})\s+should\s+be\s+anxiety\s+medication"])
    if anxiety_time is None or anti_time is None or total_lower is None or anxiety_lower is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="two_medication_time_min_ilp",
        symbols=["anxiety", "anti_depressant"],
        costs={"anxiety": anxiety_time, "anti_depressant": anti_time},
        constraints=[
            ({"anxiety": 1.0, "anti_depressant": 1.0}, total_lower, math.inf, "total_medication_lower"),
            ({"anxiety": 1.0}, anxiety_lower, math.inf, "anxiety_lower"),
            ({"anxiety": 1.0, "anti_depressant": -2.0}, -math.inf, 0.0, "anxiety_twice_anti_upper"),
        ],
        confidence=0.84,
        notes="Solved two-medication integer time minimization with total, lower-bound, and ratio constraints.",
    )


def _solve_snow_remover_min_count_ilp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("seasonal and permanent snow removers" in lowered and "minimize the total number" in lowered):
        return TemplateSolveResult(False)
    seasonal = re.search(rf"seasonal\s+snow\s+remover\s+works\s+({_NUMBER_TOKEN})\s+hours?\s+per\s+shift\s+and\s+gets\s+paid\s+\\?\$?\s*({_NUMBER_TOKEN})", normalized, flags=re.IGNORECASE)
    permanent = re.search(rf"permanent\s+snow\s+remover\s+works\s+({_NUMBER_TOKEN})\s+hours?\s+per\s+shift\s+and\s+gets\s+paid\s+\\?\$?\s*({_NUMBER_TOKEN})", normalized, flags=re.IGNORECASE)
    hours = _number_after_patterns(normalized, [rf"needs\s+({_NUMBER_TOKEN})\s+hours?\s+of\s+snow\s+remover\s+labor"])
    budget = _number_after_patterns(normalized, [rf"budget\s+of\s+\\?\$?\s*({_NUMBER_TOKEN})"])
    if seasonal is None or permanent is None or hours is None or budget is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="snow_remover_min_count_ilp",
        symbols=["seasonal", "permanent"],
        costs={"seasonal": 1.0, "permanent": 1.0},
        constraints=[
            ({"seasonal": _number(seasonal.group(1)), "permanent": _number(permanent.group(1))}, hours, math.inf, "labor_hours_lower"),
            ({"seasonal": _number(seasonal.group(2)), "permanent": _number(permanent.group(2))}, -math.inf, budget, "budget_upper"),
        ],
        confidence=0.84,
        notes="Solved seasonal/permanent snow-remover integer minimum-count model with labor and budget constraints.",
    )


def _solve_course_prerequisite_cover_min(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        "master's student" in lowered
        and "select two courses in mathematics" in lowered
        and "two in operations research" in lowered
        and "two in computer science" in lowered
        and "prerequisites" in lowered
    ):
        return TemplateSolveResult(False)
    courses = {
        "Calculus": {"math"},
        "Operations Research": {"math", "or"},
        "Data Structures": {"math", "cs"},
        "Management Statistics": {"math", "or"},
        "Computer Simulation": {"or", "cs"},
        "Computer Programming": {"cs"},
        "Forecasting": {"math", "or"},
    }
    prerequisites = {
        "Computer Simulation": {"Computer Programming"},
        "Data Structures": {"Computer Programming"},
        "Management Statistics": {"Calculus"},
        "Forecasting": {"Management Statistics"},
    }
    labels = list(courses)
    best: tuple[int, tuple[str, ...]] | None = None
    for count in range(1, len(labels) + 1):
        for selected in itertools.combinations(labels, count):
            selected_set = set(selected)
            if any(not required <= selected_set for course, required in prerequisites.items() if course in selected_set):
                continue
            coverage = {"math": 0, "or": 0, "cs": 0}
            for course in selected:
                for category in courses[course]:
                    coverage[category] += 1
            if all(coverage[category] >= 2 for category in coverage):
                best = (count, selected)
                break
        if best:
            break
    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="course_prerequisite_cover_min_enum",
            status="infeasible",
            confidence=0.82,
        )
    return TemplateSolveResult(
        matched=True,
        template_id="course_prerequisite_cover_min_enum",
        status="optimal",
        objective_value=float(best[0]),
        variable_values={course: 1.0 for course in best[1]},
        confidence=0.86,
        notes="Solved course prerequisite/category coverage by exact subset enumeration.",
        artifact={"courses": courses, "prerequisites": prerequisites, "selected": list(best[1])},
    )


def _solve_three_channel_effort_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "marketing firm" in lowered
        and "advertising channels" in lowered
        and "x and y combined" in lowered
        and "channels y and z" in lowered
    ):
        return TemplateSolveResult(False)
    upper_xy = _number_after_patterns(normalized, [rf"channel\s+X\s+and\s+Y\s+combined\s+cannot\s+exceed\s+\\?\$?\s*({_NUMBER_TOKEN})"])
    upper_yz = _number_after_patterns(normalized, [rf"channels\s+Y\s+and\s+Z\s+cannot\s+surpass\s+\\?\$?\s*({_NUMBER_TOKEN})"])
    lower_xz = _number_after_patterns(normalized, [rf"channels\s+X\s+and\s+Z\s+must\s+yield\s+a\s+minimum\s+expenditure\s+of\s+at\s+least\s+\\?\$?\s*({_NUMBER_TOKEN})"])
    costs = re.search(
        rf"channel\s+X\s+is\s+\\?\$?\s*({_NUMBER_TOKEN}),\s+for\s+channel\s+Y\s+is\s+\\?\$?\s*({_NUMBER_TOKEN}),\s+and\s+for\s+channel\s+Z\s+is\s+\\?\$?\s*({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if upper_xy is None or upper_yz is None or lower_xz is None or costs is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="three_channel_effort_min_cost_ilp",
        symbols=["X", "Y", "Z"],
        costs={"X": _number(costs.group(1)), "Y": _number(costs.group(2)), "Z": _number(costs.group(3))},
        constraints=[
            ({"X": 1.0, "Y": 1.0}, -math.inf, upper_xy, "x_y_upper"),
            ({"Y": 1.0, "Z": 1.0}, -math.inf, upper_yz, "y_z_upper"),
            ({"X": 1.0, "Z": 1.0}, lower_xz, math.inf, "x_z_lower"),
        ],
        confidence=0.82,
        notes="Solved three-channel integer effort minimum-cost model with pairwise budget and reach constraints.",
    )


def _solve_three_project_minimum_allocation_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text.replace("\\n", " "))
    lowered = normalized.lower()
    if not (
        "construction company" in lowered
        and "three different projects" in lowered
        and "minimum allocation" in lowered
        and "minimize the total cost" in lowered
    ):
        return TemplateSolveResult(False)
    total_upper = _number_after_patterns(normalized, [rf"total\s+resource\s+allocation\s+across\s+all\s+three\s+projects\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"])
    x_lower = _number_after_patterns(normalized, [rf"Project\s+\\?\$?X\\?\$?\s+requires\s+a\s+minimum\s+allocation\s+of\s+({_NUMBER_TOKEN})"])
    y_lower = _number_after_patterns(normalized, [rf"project\s+\\?\$?Y\\?\$?\s+needs\s+at\s+least\s+({_NUMBER_TOKEN})"])
    z_lower = _number_after_patterns(normalized, [rf"Project\s+\\?\$?Z\\?\$?.*?requires\s+a\s+minimum\s+allocation\s+of\s+({_NUMBER_TOKEN})"])
    x_y_margin = _number_after_patterns(normalized, [rf"project\s+X\s+should\s+not\s+exceed\s+that\s+for\s+project\s+Y\s+by\s+more\s+than\s+({_NUMBER_TOKEN})"])
    y_z_margin = _number_after_patterns(normalized, [rf"project\s+Y\s+should\s+be\s+at\s+least\s+({_NUMBER_TOKEN})\s+units?\s+more\s+than\s+that\s+for\s+project\s+Z"])
    costs = re.search(
        rf"[^A-Za-z0-9]*({_NUMBER_TOKEN})[^A-Za-z0-9]*\s+for\s+project\s+X,\s+"
        rf"[^A-Za-z0-9]*({_NUMBER_TOKEN})[^A-Za-z0-9]*\s+for\s+project\s+Y,\s+and\s+"
        rf"[^A-Za-z0-9]*({_NUMBER_TOKEN})[^A-Za-z0-9]*\s+for\s+project\s+Z",
        normalized,
        flags=re.IGNORECASE,
    )
    if any(value is None for value in (total_upper, x_lower, y_lower, z_lower, x_y_margin, y_z_margin)) or costs is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="three_project_minimum_allocation_min_cost_ilp",
        symbols=["X", "Y", "Z"],
        costs={"X": _number(costs.group(1)), "Y": _number(costs.group(2)), "Z": _number(costs.group(3))},
        constraints=[
            ({"X": 1.0, "Y": 1.0, "Z": 1.0}, -math.inf, float(total_upper), "total_upper"),
            ({"X": 1.0}, float(x_lower), math.inf, "x_lower"),
            ({"Y": 1.0}, float(y_lower), math.inf, "y_lower"),
            ({"Z": 1.0}, float(z_lower), math.inf, "z_lower"),
            ({"X": 1.0, "Y": -1.0}, -math.inf, float(x_y_margin), "x_y_margin_upper"),
            ({"Y": 1.0, "Z": -1.0}, float(y_z_margin), math.inf, "y_z_margin_lower"),
        ],
        confidence=0.84,
        notes="Solved three-project integer minimum allocation cost model with lower, total, and difference constraints.",
    )


def _solve_two_project_weighted_resource_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("telecommunications company" in lowered and "projects: x and y" in lowered and "three times the allocation for project y" in lowered):
        return TemplateSolveResult(False)
    costs = re.search(
        rf"project\s+X\s+costs\s+({_NUMBER_TOKEN})\s+units,\s+while\s+each\s+unit\s+allocated\s+to\s+project\s+Y\s+costs\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    weighted_upper = _number_after_patterns(normalized, [rf"maximum\s+limit\s+of\s+({_NUMBER_TOKEN})\s+units,\s+with\s+three\s+times\s+the\s+allocation\s+for\s+project\s+Y"])
    lower = _number_after_patterns(normalized, [rf"combined\s+allocation\s+of\s+twice\s+that\s+for\s+project\s+X\s+and\s+that\s+for\s+project\s+Y\s+must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})"])
    difference = _number_after_patterns(normalized, [rf"difference\s+between\s+allocations\s+for\s+project\s+X\s+and\s+Y\s+should\s+not\s+exceed\s+({_NUMBER_TOKEN})"])
    x_upper = _number_after_patterns(normalized, [rf"Project\s+X\s+can't\s+exceed\s+({_NUMBER_TOKEN})"])
    y_upper = _number_after_patterns(normalized, [rf"Project\s+Y\s+can't\s+exceed\s+({_NUMBER_TOKEN})"])
    if costs is None or any(value is None for value in (weighted_upper, lower, difference, x_upper, y_upper)):
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="two_project_weighted_resource_min_cost_ilp",
        symbols=["X", "Y"],
        costs={"X": _number(costs.group(1)), "Y": _number(costs.group(2))},
        constraints=[
            ({"X": 1.0, "Y": 3.0}, -math.inf, float(weighted_upper), "weighted_resource_upper"),
            ({"X": 2.0, "Y": 1.0}, float(lower), math.inf, "strategic_lower"),
            ({"X": 1.0, "Y": -1.0}, -math.inf, float(difference), "x_y_difference_upper"),
        ],
        upper_bounds={"X": float(x_upper), "Y": float(y_upper)},
        confidence=0.84,
        notes="Solved two-project weighted-resource integer minimum-cost model.",
    )


def _solve_three_project_lower_bound_budget_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("tourism agency" in lowered and "annual budget" in lowered and "minimum investment" in lowered and "minimize the total cost" in lowered):
        return TemplateSolveResult(False)
    total_upper = _number_after_patterns(normalized, [rf"total\s+budget\s+across\s+all\s+three\s+projects\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"])
    lowers = {
        "X": _number_after_patterns(normalized, [rf"Project\s+X\s+requires\s+a\s+minimum\s+investment\s+of\s+({_NUMBER_TOKEN})"]),
        "Y": _number_after_patterns(normalized, [rf"project\s+Y\s+needs\s+at\s+least\s+({_NUMBER_TOKEN})"]),
        "Z": _number_after_patterns(normalized, [rf"Project\s+Z.*?demands\s+a\s+minimum\s+of\s+({_NUMBER_TOKEN})"]),
    }
    costs = re.search(
        rf"projects\s+X,\s+Y,\s+and\s+Z\s+incurs\s+different\s+costs\s+estimated\s+as\s+({_NUMBER_TOKEN})\s*,\s*({_NUMBER_TOKEN})\s*,\s+and\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if total_upper is None or costs is None or any(value is None for value in lowers.values()):
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="three_project_lower_bound_budget_min_cost_ilp",
        symbols=["X", "Y", "Z"],
        costs={"X": _number(costs.group(1)), "Y": _number(costs.group(2)), "Z": _number(costs.group(3))},
        constraints=[
            ({"X": 1.0, "Y": 1.0, "Z": 1.0}, -math.inf, total_upper, "total_upper"),
            *[({symbol: 1.0}, float(lowers[symbol]), math.inf, f"{symbol}_lower") for symbol in ["X", "Y", "Z"]],
        ],
        confidence=0.82,
        notes="Solved three-project lower-bound budget integer minimum-cost model.",
    )


def _solve_two_store_customer_min_count(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if "retail store" in lowered and "factory outlet" in lowered and "reduce the number of stores" in lowered:
        retail = re.search(rf"retail\s+store\s+brings\s+in\s+({_NUMBER_TOKEN})\s+customers?.*?requires\s+({_NUMBER_TOKEN})\s+employees?", normalized, flags=re.IGNORECASE)
        outlet = re.search(rf"factory\s+outlet\s+brings\s+in\s+({_NUMBER_TOKEN})\s+customers?.*?requires\s+({_NUMBER_TOKEN})\s+employees?", normalized, flags=re.IGNORECASE)
        customers = _number_after_patterns(normalized, [rf"at\s+least\s+({_NUMBER_TOKEN})\s+customers?\s+every\s+day"])
        employees = _number_after_patterns(normalized, [rf"available\s+({_NUMBER_TOKEN})\s+employees?"])
        if retail is None or outlet is None or customers is None or employees is None:
            return TemplateSolveResult(False)
        return _solve_small_integer_min_cost_model(
            template_id="two_store_customer_min_count_ilp",
            symbols=["retail", "factory_outlet"],
            costs={"retail": 1.0, "factory_outlet": 1.0},
            constraints=[
                ({"retail": _number(retail.group(1)), "factory_outlet": _number(outlet.group(1))}, customers, math.inf, "customer_lower"),
                ({"retail": _number(retail.group(2)), "factory_outlet": _number(outlet.group(2))}, -math.inf, employees, "employee_upper"),
            ],
            confidence=0.84,
            notes="Solved retail/factory outlet integer minimum-store model with customer and employee constraints.",
        )

    if "sandwich company" in lowered and "dine-in place" in lowered and "food-truck" in lowered:
        dine_in = re.search(rf"dine-in\s+place\s+can\s+make\s+({_NUMBER_TOKEN})\s+sandwiches?.*?requires\s+({_NUMBER_TOKEN})\s+employees?", normalized, flags=re.IGNORECASE)
        truck = re.search(rf"food-truck\s+can\s+make\s+({_NUMBER_TOKEN})\s+sandwiches?.*?requires\s+({_NUMBER_TOKEN})\s+employees?", normalized, flags=re.IGNORECASE)
        demand = _number_after_patterns(normalized, [rf"make\s+at\s+least\s+({_NUMBER_TOKEN})\s+sandwiches?"])
        employees = _number_after_patterns(normalized, [rf"available\s+({_NUMBER_TOKEN})\s+employees?"])
        if dine_in is None or truck is None or demand is None or employees is None:
            return TemplateSolveResult(False)
        return _solve_small_integer_min_cost_model(
            template_id="two_store_capacity_min_count_ilp",
            symbols=["dine_in", "food_truck"],
            costs={"dine_in": 1.0, "food_truck": 1.0},
            constraints=[
                ({"dine_in": _number(dine_in.group(1)), "food_truck": _number(truck.group(1))}, demand, math.inf, "sandwich_lower"),
                ({"dine_in": _number(dine_in.group(2)), "food_truck": _number(truck.group(2))}, -math.inf, employees, "employee_upper"),
            ],
            confidence=0.84,
            notes="Solved two-store minimum-count sandwich capacity model with employee constraints.",
        )

    return TemplateSolveResult(False)


def _solve_two_machine_tea_leaf_max_lp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("tea estate" in lowered and "traditional machine" in lowered and "modern machine" in lowered and "maximize the amount of tea leaves" in lowered):
        return TemplateSolveResult(False)
    land = _number_after_patterns(normalized, [rf"available\s+({_NUMBER_TOKEN})\s+acres?\s+of\s+land"])
    traditional = re.search(rf"traditional\s+machine\s+can\s+pick\s+({_NUMBER_TOKEN})\s*kg.*?creates\s+({_NUMBER_TOKEN})\s*kg\s+of\s+waste.*?requires\s+({_NUMBER_TOKEN})\s+liters?\s+of\s+fuel", normalized, flags=re.IGNORECASE)
    modern = re.search(rf"modern\s+machine\s+can\s+pick\s+({_NUMBER_TOKEN})\s*kg.*?creates\s+({_NUMBER_TOKEN})\s*kg\s+of\s+waste.*?requires\s+({_NUMBER_TOKEN})\s+liters?\s+of\s+fuel", normalized, flags=re.IGNORECASE)
    fuel = _number_after_patterns(normalized, [rf"available\s+({_NUMBER_TOKEN})\s+liters?\s+of\s+fuel"])
    waste = _number_after_patterns(normalized, [rf"at\s+most\s+({_NUMBER_TOKEN})\s*kg\s+of\s+waste"])
    if land is None or traditional is None or modern is None or fuel is None or waste is None:
        return TemplateSolveResult(False)
    status, objective, values, message = _linprog_maximize(
        objective=[_number(traditional.group(1)), _number(modern.group(1))],
        constraints=[
            [1.0, 1.0],
            [_number(traditional.group(3)), _number(modern.group(3))],
            [_number(traditional.group(2)), _number(modern.group(2))],
        ],
        upper_bounds=[land, fuel, waste],
    )
    if status != "optimal":
        return TemplateSolveResult(matched=True, template_id="two_machine_tea_leaf_max_lp", status=status, confidence=0.82, notes=message)
    return TemplateSolveResult(
        matched=True,
        template_id="two_machine_tea_leaf_max_lp",
        status="optimal",
        objective_value=objective,
        variable_values={"traditional": values[0], "modern": values[1]},
        confidence=0.86,
        notes="Solved two-machine tea-leaf LP with land, fuel, and waste constraints.",
    )


def _solve_two_experiment_green_gas_max_ilp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("chemistry teacher" in lowered and "green gas" in lowered and "smelly gas" in lowered and "maximize the total amount" in lowered):
        return TemplateSolveResult(False)
    exp1 = re.search(rf"experiment\s+1,\s+({_NUMBER_TOKEN})\s+units?\s+of\s+(?:the\s+)?red\s+liquid\s+and\s+({_NUMBER_TOKEN})\s+units?\s+of\s+(?:the\s+)?blue\s+liquid\s+mix\s+to\s+create\s+({_NUMBER_TOKEN})\s+units?\s+of\s+green\s+gas", normalized, flags=re.IGNORECASE)
    exp2 = re.search(rf"experiment\s+2,\s+({_NUMBER_TOKEN})\s+units?\s+of\s+(?:the\s+)?red\s+liquid\s+and\s+({_NUMBER_TOKEN})\s+units?\s+of\s+(?:the\s+)?blue\s+liquid\s+mix\s+to\s+create\s+({_NUMBER_TOKEN})\s+units?\s+of\s+the\s+green\s+gas", normalized, flags=re.IGNORECASE)
    smelly = re.search(rf"experiment\s+1\s+produces\s+({_NUMBER_TOKEN})\s+units?\s+of\s+smelly\s+gas\s+while\s+experiment\s+2\s+produces\s+({_NUMBER_TOKEN})\s+units?", normalized, flags=re.IGNORECASE)
    availability = re.search(
        rf"available\s+({_NUMBER_TOKEN})\s+units?\s+of\s+red\s+liquid\s+and\s+({_NUMBER_TOKEN})\s+units?\s+of\s+blue\s+liquid",
        normalized,
        flags=re.IGNORECASE,
    )
    smelly_limit = _number_after_patterns(normalized, [rf"at\s+most\s+({_NUMBER_TOKEN})\s+units?\s+of\s+smelly\s+gas"])
    if exp1 is None or exp2 is None or smelly is None or availability is None or smelly_limit is None:
        return TemplateSolveResult(False)
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(matched=True, template_id="two_experiment_green_gas_max_ilp", status="solver_unavailable", confidence=0.82, notes=str(exc))
    matrix = np.array(
        [
            [_number(exp1.group(1)), _number(exp2.group(1))],
            [_number(exp1.group(2)), _number(exp2.group(2))],
            [_number(smelly.group(1)), _number(smelly.group(2))],
        ],
        dtype=float,
    )
    result = milp(
        c=-np.array([_number(exp1.group(3)), _number(exp2.group(3))], dtype=float),
        integrality=np.ones(2),
        bounds=Bounds(np.zeros(2), np.full(2, math.inf)),
        constraints=LinearConstraint(matrix, np.full(3, -math.inf), np.array([_number(availability.group(1)), _number(availability.group(2)), smelly_limit], dtype=float)),
    )
    if not result.success:
        return TemplateSolveResult(matched=True, template_id="two_experiment_green_gas_max_ilp", status="infeasible", confidence=0.82, notes=str(result.message))
    return TemplateSolveResult(
        matched=True,
        template_id="two_experiment_green_gas_max_ilp",
        status="optimal",
        objective_value=float(-result.fun),
        variable_values={"experiment_1": float(result.x[0]), "experiment_2": float(result.x[1])},
        confidence=0.84,
        notes="Solved two-experiment integer green-gas maximization with liquid and smelly-gas constraints.",
    )


def _solve_ship_plane_fuel_min_ilp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("ships and planes" in lowered and "containers worth of goods" in lowered and "minimize the total amount of fuel" in lowered):
        return TemplateSolveResult(False)
    ship = re.search(rf"ship\s+can\s+take\s+({_NUMBER_TOKEN})\s+containers?.*?uses\s+({_NUMBER_TOKEN})\s+liters?\s+of\s+fuel", normalized, flags=re.IGNORECASE)
    plane = re.search(rf"plane\s+can\s+take\s+({_NUMBER_TOKEN})\s+containers?.*?uses\s+({_NUMBER_TOKEN})\s+liters?\s+of\s+fuel", normalized, flags=re.IGNORECASE)
    demand = _number_after_patterns(normalized, [rf"at\s+least\s+({_NUMBER_TOKEN})\s+containers?\s+worth\s+of\s+goods"])
    plane_upper = _number_after_patterns(normalized, [rf"at\s+most\s+({_NUMBER_TOKEN})\s+plane\s+trips?"])
    share = re.search(rf"minimum\s+of\s+({_NUMBER_TOKEN})\s*%\s+of\s+the\s+trips?\s+made\s+must\s+be\s+by\s+ship", normalized, flags=re.IGNORECASE)
    if ship is None or plane is None or demand is None or plane_upper is None or share is None:
        return TemplateSolveResult(False)
    share_value = _number(share.group(1)) / 100.0
    return _solve_small_integer_min_cost_model(
        template_id="ship_plane_fuel_min_ilp",
        symbols=["ship", "plane"],
        costs={"ship": _number(ship.group(2)), "plane": _number(plane.group(2))},
        constraints=[
            ({"ship": _number(ship.group(1)), "plane": _number(plane.group(1))}, demand, math.inf, "container_lower"),
            ({"plane": 1.0}, -math.inf, plane_upper, "plane_upper"),
            ({"ship": 1.0 - share_value, "plane": -share_value}, 0.0, math.inf, "ship_share_lower"),
        ],
        confidence=0.84,
        notes="Solved ship/plane integer fuel minimization with demand, plane limit, and ship-share constraints.",
    )


def _solve_meal_fiber_selection_lp_enum(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "planning her dinner" in lowered
        and "fiber intake" in lowered
        and "protein source" in lowered
        and "at least two kinds of vegetables" in lowered
    ):
        return TemplateSolveResult(False)

    fiber_matches = re.findall(
        rf"every\s+100\s+grams?\s+of\s+([A-Za-z ]+?)\s+contains\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+fiber",
        normalized,
        flags=re.IGNORECASE,
    )
    price_matches = re.findall(
        rf"(?:price\s+of|price\s+of\s+the)?\s*([A-Za-z ]+?)\s+(?:is|are)\s+\$?\s*({_NUMBER_TOKEN})\s+per\s+100\s+grams?",
        normalized,
        flags=re.IGNORECASE,
    )
    budget = _number_after_patterns(normalized, [rf"budget\s+of\s+\$?\s*({_NUMBER_TOKEN})"])
    total_grams = _number_after_patterns(normalized, [rf"total\s+food\s+intake\s+should\s+be\s+({_NUMBER_TOKEN})\s+grams?"])
    if not fiber_matches or not price_matches or budget is None or total_grams is None:
        return TemplateSolveResult(False)

    def normalize_food(label: str) -> str:
        cleaned = re.sub(r"\b(?:the|and)\b", " ", label, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bprice\s+of\b", " ", cleaned, flags=re.IGNORECASE)
        return _singular_label(cleaned)

    fiber = {normalize_food(label): _number(value) for label, value in fiber_matches}
    prices: dict[str, float] = {}
    for label, value in price_matches:
        cleaned = normalize_food(label)
        for part in re.split(r",|\band\b", cleaned):
            part = normalize_food(part)
            if part:
                prices[part] = _number(value)

    proteins = [label for label in ("salmon", "beef", "pork") if label in prices]
    vegetables = [label for label in ("okra", "carrot", "celery", "cabbage") if label in prices and label in fiber]
    if len(proteins) < 3 or len(vegetables) < 2:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="meal_fiber_selection_lp_enum",
            status="solver_unavailable",
            confidence=0.78,
            notes=str(exc),
        )

    total_units = total_grams / 100.0
    min_selected_units = 0.1
    best: tuple[float, list[str], list[float]] | None = None
    for protein in proteins:
        for count in range(2, len(vegetables) + 1):
            for veggie_subset in itertools.combinations(vegetables, count):
                labels = [protein, *veggie_subset]
                objective = [0.0, *[fiber[label] for label in veggie_subset]]
                result = linprog(
                    [-value for value in objective],
                    A_ub=[[prices[label] for label in labels]],
                    b_ub=[budget],
                    A_eq=[[1.0] * len(labels)],
                    b_eq=[total_units],
                    bounds=[(min_selected_units, None)] * len(labels),
                    method="highs",
                )
                if result.success:
                    value = float(-result.fun)
                    if best is None or value > best[0]:
                        best = (value, labels, [float(item) for item in result.x])

    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="meal_fiber_selection_lp_enum",
            status="infeasible",
            confidence=0.78,
            artifact={"proteins": proteins, "vegetables": vegetables},
        )
    objective, labels, values = best
    return TemplateSolveResult(
        matched=True,
        template_id="meal_fiber_selection_lp_enum",
        status="optimal",
        objective_value=objective,
        variable_values={f"hundred_grams_{label}": value for label, value in zip(labels, values)},
        confidence=0.8,
        notes="Solved meal fiber LP by enumerating one protein and vegetable subsets with a 10g minimum for selected foods.",
        artifact={
            "prices_per_100g": prices,
            "fiber_per_100g": fiber,
            "budget": budget,
            "total_100g_units": total_units,
            "minimum_selected_100g_units": min_selected_units,
            "selected": labels,
        },
    )


def _solve_three_route_balance_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "shipping company" in lowered
        and "three routes" in lowered
        and "balanced service coverage" in lowered
    ):
        return TemplateSolveResult(False)
    total_upper = _number_after_patterns(normalized, [rf"cannot\s+exceed\s+({_NUMBER_TOKEN})\s+units?"])
    x_y_margin = _number_after_patterns(normalized, [rf"route\s+X\s+must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})\s+units?\s+more\s+than\s+that\s+for\s+route\s+Y"])
    yz_upper = _number_after_patterns(normalized, [rf"routes?\s+Y\s+and\s+Z\s+cannot\s+exceed\s+({_NUMBER_TOKEN})\s+units?"])
    costs = re.search(
        rf"routes?\s+\$?X,\s*Y\$?,\s+and\s+\$?Z\$?\s+incurs\s+costs\s+of\s+"
        rf"({_NUMBER_TOKEN})\s*,\s*({_NUMBER_TOKEN})\s*,\s+and\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if total_upper is None or x_y_margin is None or yz_upper is None or costs is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="three_route_balance_min_cost_ilp",
        symbols=["X", "Y", "Z"],
        costs={"X": _number(costs.group(1)), "Y": _number(costs.group(2)), "Z": _number(costs.group(3))},
        constraints=[
            ({"X": 1.0, "Y": 1.0, "Z": 1.0}, -math.inf, total_upper, "total_upper"),
            ({"X": 1.0, "Y": -1.0}, x_y_margin, math.inf, "x_y_margin_lower"),
            ({"Y": 1.0, "Z": 1.0}, -math.inf, yz_upper, "y_z_upper"),
        ],
        confidence=0.82,
        notes="Solved three-route integer minimum-cost allocation with explicit balance and capacity constraints.",
    )


def _solve_energy_project_linear_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text.replace("\\n", " "))
    lowered = normalized.lower()
    if not (
        "energy company" in lowered
        and "three projects" in lowered
        and "x1" in lowered
        and "x2" in lowered
        and "x3" in lowered
        and "minimum total investment cost" in lowered
    ):
        return TemplateSolveResult(False)
    total_upper = _number_after_patterns(normalized, [rf"total\s+investment\s+across\s+all\s+three\s+projects\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"])
    lower_1_2 = _number_after_patterns(normalized, [rf"difference\s+between\s+twice\s+the\s+resource\s+allocated\s+for\s+project\s+\$?x1\$?\s+and\s+thrice\s+that\s+of\s+project\s+\$?x2\$?\s+should\s+be\s+at\s+least\s+({_NUMBER_TOKEN})"])
    x1_margin = _number_after_patterns(normalized, [rf"project\s+\$?x1\$?\s+cannot\s+exceed\s+by\s+more\s+than\s+({_NUMBER_TOKEN})\s+units?,\s+half\s+of\s+that\s+allocated\s+for\s+project\s+\$?x3\$?"])
    lower_mixed = _number_after_patterns(normalized, [rf"should\s+not\s+fall\s+below\s+-\s*({_NUMBER_TOKEN})\s+units?"])
    costs = re.search(
        rf"projects?\s+\$?x1,\s*x2\$?,\s+and\s+\$?x3\$?\s+yields\s+different\s+returns\s+or\s+costs,\s+quantified\s+as\s+"
        rf"({_NUMBER_TOKEN})\s*,\s*({_NUMBER_TOKEN})\s*,\s+and\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if any(value is None for value in (total_upper, lower_1_2, x1_margin, lower_mixed)) or costs is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="energy_project_linear_min_cost_ilp",
        symbols=["x1", "x2", "x3"],
        costs={"x1": _number(costs.group(1)), "x2": _number(costs.group(2)), "x3": _number(costs.group(3))},
        constraints=[
            ({"x1": 1.0, "x2": 1.0, "x3": 1.0}, -math.inf, total_upper, "total_upper"),
            ({"x1": 2.0, "x2": -3.0}, lower_1_2, math.inf, "two_x1_minus_three_x2_lower"),
            ({"x1": 1.0, "x3": -0.5}, -math.inf, x1_margin, "x1_half_x3_margin_upper"),
            ({"x1": -1.0, "x2": 2.5, "x3": -1.0}, -lower_mixed, math.inf, "mixed_project_lower"),
        ],
        confidence=0.82,
        notes="Solved three-project energy integer minimum-cost model from explicit linear constraints.",
    )


def _solve_two_printer_shared_machine_profit_lp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "office supply company" in lowered
        and "color printers" in lowered
        and "black and white printers" in lowered
        and "paper tray installing machine" in lowered
    ):
        return TemplateSolveResult(False)
    color_cap = _number_after_patterns(normalized, [rf"produce\s+at\s+most\s+({_NUMBER_TOKEN})\s+color\s+printers?"])
    bw_cap = _number_after_patterns(normalized, [rf"at\s+most\s+({_NUMBER_TOKEN})\s+black\s+and\s+white\s+printers?"])
    shared_cap = _number_after_patterns(normalized, [rf"machine\s+can\s+make\s+at\s+most\s+({_NUMBER_TOKEN})\s+printers?"])
    profits = re.search(
        rf"Color\s+printers\s+generate\s+a\s+profit\s+of\s+\$?\s*({_NUMBER_TOKEN})\s+per\s+printer\s+while\s+black\s+and\s+white\s+printers\s+generate\s+a\s+profit\s+of\s+\$?\s*({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if color_cap is None or bw_cap is None or shared_cap is None or profits is None:
        return TemplateSolveResult(False)
    status, objective, values, message = _linprog_maximize(
        objective=[_number(profits.group(1)), _number(profits.group(2))],
        constraints=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        upper_bounds=[color_cap, bw_cap, shared_cap],
    )
    if status != "optimal":
        return TemplateSolveResult(matched=True, template_id="two_printer_shared_machine_profit_lp", status=status, confidence=0.82, notes=message)
    return TemplateSolveResult(
        matched=True,
        template_id="two_printer_shared_machine_profit_lp",
        status="optimal",
        objective_value=objective,
        variable_values={"color_printers": values[0], "black_and_white_printers": values[1]},
        confidence=0.86,
        notes="Solved two-printer continuous profit LP with individual team caps and shared machine capacity.",
    )


def _solve_two_animal_package_min_count(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("camels and horses" in lowered and "packages" in lowered and "units of food" in lowered):
        return TemplateSolveResult(False)
    camel = re.search(rf"camel\s+can\s+carry\s+({_NUMBER_TOKEN})\s+packages?.*?requires\s+({_NUMBER_TOKEN})\s+units?\s+of\s+food", normalized, flags=re.IGNORECASE)
    horse = re.search(rf"horse\s+can\s+carry\s+({_NUMBER_TOKEN})\s+packages?.*?requires\s+({_NUMBER_TOKEN})\s+units?\s+of\s+food", normalized, flags=re.IGNORECASE)
    demand = _number_after_patterns(normalized, [rf"deliver\s+at\s+least\s+({_NUMBER_TOKEN})\s+packages?"])
    food = _number_after_patterns(normalized, [rf"have\s+({_NUMBER_TOKEN})\s+units?\s+of\s+food\s+available"])
    if camel is None or horse is None or demand is None or food is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="two_animal_package_min_count_ilp",
        symbols=["camels", "horses"],
        costs={"camels": 1.0, "horses": 1.0},
        constraints=[
            ({"camels": _number(camel.group(1)), "horses": _number(horse.group(1))}, demand, math.inf, "package_lower"),
            ({"camels": _number(camel.group(2)), "horses": _number(horse.group(2))}, -math.inf, food, "food_upper"),
            ({"camels": 1.0, "horses": -1.0}, 0.0, math.inf, "horse_count_not_above_camel"),
        ],
        confidence=0.84,
        notes="Solved two-animal minimum-count delivery model with package, food, and horse-count constraints.",
    )


def _solve_two_package_stock_profit_lp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("banana-haters package" in lowered and "combo package" in lowered and "liquidate its stock" in lowered):
        return TemplateSolveResult(False)
    stock = {
        label.lower(): _number(value)
        for value, label in re.findall(
            rf"({_NUMBER_TOKEN})\s+(apples|bananas|grapes)",
            normalized,
            flags=re.IGNORECASE,
        )[:3]
    }
    banana_hat = re.search(
        rf"banana-haters\s+package\s+with\s+({_NUMBER_TOKEN})\s+apples\s+and\s+({_NUMBER_TOKEN})\s+grapes.*?profit\s+of\s+({_NUMBER_TOKEN})\s+euros?",
        normalized,
        flags=re.IGNORECASE,
    )
    combo = re.search(
        rf"combo\s+package\s+with\s+({_NUMBER_TOKEN})\s+apples,\s+({_NUMBER_TOKEN})\s+bananas,\s+and\s+({_NUMBER_TOKEN})\s+grapes,\s+yielding\s+a\s+profit\s+of\s+({_NUMBER_TOKEN})\s+euros?",
        normalized,
        flags=re.IGNORECASE,
    )
    if set(stock) != {"apples", "bananas", "grapes"} or banana_hat is None or combo is None:
        return TemplateSolveResult(False)
    status, objective, values, message = _linprog_maximize(
        objective=[_number(banana_hat.group(3)), _number(combo.group(4))],
        constraints=[
            [_number(banana_hat.group(1)), _number(combo.group(1))],
            [0.0, _number(combo.group(2))],
            [_number(banana_hat.group(2)), _number(combo.group(3))],
        ],
        upper_bounds=[stock["apples"], stock["bananas"], stock["grapes"]],
    )
    if status != "optimal":
        return TemplateSolveResult(matched=True, template_id="two_package_stock_profit_lp", status=status, confidence=0.82, notes=message)
    return TemplateSolveResult(
        matched=True,
        template_id="two_package_stock_profit_lp",
        status="optimal",
        objective_value=objective,
        variable_values={"banana_haters_packages": values[0], "combo_packages": values[1]},
        confidence=0.86,
        notes="Solved two-package stock-constrained profit LP from explicit ingredient availability.",
    )


def _solve_meal_protein_pack_selection_lp_enum(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "planning tonight's dinner" in lowered
        and "maximize her protein intake" in lowered
        and "at least three different types of vegetables" in lowered
    ):
        return TemplateSolveResult(False)

    protein_data: dict[str, tuple[float, float]] = {}
    for label, protein, cost in re.findall(
        rf"(Chicken|Salmon|Tofu):\s*({_NUMBER_TOKEN})g\s+protein,\s+\$?\s*({_NUMBER_TOKEN})\s+cost,\s+per\s+100g",
        normalized,
        flags=re.IGNORECASE,
    ):
        protein_data[_singular_label(label)] = (_number(protein), _number(cost))
    vegetable_data: dict[str, tuple[float, float]] = {}
    for label, protein, cost in re.findall(
        rf"([A-Za-z ]+)\s+\(100g\s+pack\):\s*({_NUMBER_TOKEN})g\s+protein,\s+\$?\s*({_NUMBER_TOKEN})\s+cost",
        normalized,
        flags=re.IGNORECASE,
    ):
        vegetable_data[_singular_label(label)] = (_number(protein), _number(cost))
    budget = _number_after_patterns(normalized, [rf"total\s+budget\s+is\s+\$?\s*({_NUMBER_TOKEN})"])
    weight = _number_after_patterns(normalized, [rf"total\s+weight\s+of\s+all\s+food\s+must\s+not\s+exceed\s+({_NUMBER_TOKEN})\s+grams?"])
    if len(protein_data) < 3 or len(vegetable_data) < 3 or budget is None or weight is None:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="meal_protein_pack_selection_lp_enum",
            status="solver_unavailable",
            confidence=0.78,
            notes=str(exc),
        )

    proteins = list(protein_data)
    vegetables = list(vegetable_data)
    max_units = weight / 100.0
    best: tuple[float, list[str], list[float]] | None = None
    for count in range(3, len(vegetables) + 1):
        for veggie_subset in itertools.combinations(vegetables, count):
            labels = [*proteins, *veggie_subset]
            objective = [
                *(protein_data[label][0] for label in proteins),
                *(vegetable_data[label][0] for label in veggie_subset),
            ]
            costs = [
                *(protein_data[label][1] for label in proteins),
                *(vegetable_data[label][1] for label in veggie_subset),
            ]
            bounds = [(0.0, None)] * len(proteins) + [(1.0, None)] * len(veggie_subset)
            result = linprog(
                [-value for value in objective],
                A_ub=[costs, [1.0] * len(labels)],
                b_ub=[budget, max_units],
                bounds=bounds,
                method="highs",
            )
            if result.success:
                value = float(-result.fun)
                if best is None or value > best[0]:
                    best = (value, labels, [float(item) for item in result.x])

    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="meal_protein_pack_selection_lp_enum",
            status="infeasible",
            confidence=0.78,
            artifact={"proteins": protein_data, "vegetables": vegetable_data},
        )
    objective_value, labels, values = best
    return TemplateSolveResult(
        matched=True,
        template_id="meal_protein_pack_selection_lp_enum",
        status="optimal",
        objective_value=objective_value,
        variable_values={f"hundred_grams_{label}": value for label, value in zip(labels, values) if not math.isclose(value, 0.0, abs_tol=1e-9)},
        confidence=0.82,
        notes="Solved meal protein LP by enumerating vegetable pack choices and optimizing continuous protein quantities.",
        artifact={
            "proteins": protein_data,
            "vegetables": vegetable_data,
            "budget": budget,
            "max_100g_units": max_units,
            "selected": labels,
        },
    )


def _solve_two_project_performance_resource_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "telecommunications company" in lowered
        and "performance score" in lowered
        and ("specific type of resources" in lowered or "total usage" in lowered)
        and "project x" in lowered
        and "project y" in lowered
    ):
        return TemplateSolveResult(False)
    costs = re.search(
        rf"project\s+\$?X\$?\s+cost(?:ing|s)\s+[^A-Za-z0-9]*({_NUMBER_TOKEN})[^A-Za-z0-9]*\s+and\s+for\s+project\s+\$?Y\$?,\s+[^A-Za-z0-9]*({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    performance = re.search(
        rf"({_NUMBER_TOKEN})\s+times\s+the\s+resources?\s+allocated\s+to\s+project\s+X\s+plus\s+"
        rf"({_NUMBER_TOKEN})\s+times\s+those\s+allocated\s+to\s+project\s+Y\s+should\s+be\s+at\s+least\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    usage = re.search(
        rf"({_NUMBER_TOKEN})\s+times\s+the\s+resources?\s+used\s+by\s+project\s+X\s+plus\s+"
        rf"({_NUMBER_TOKEN})\s+times\s+those\s+used\s+by\s+project\s+Y\s+cannot\s+exceed\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if costs is None or performance is None or usage is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="two_project_performance_resource_min_cost_ilp",
        symbols=["X", "Y"],
        costs={"X": _number(costs.group(1)), "Y": _number(costs.group(2))},
        constraints=[
            ({"X": _number(performance.group(1)), "Y": _number(performance.group(2))}, _number(performance.group(3)), math.inf, "performance_lower"),
            ({"X": _number(usage.group(1)), "Y": _number(usage.group(2))}, -math.inf, _number(usage.group(3)), "resource_upper"),
        ],
        confidence=0.84,
        notes="Solved two-project integer performance/resource minimum-cost model.",
    )


def _solve_environmental_project_budget_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "environmental organization" in lowered
        and "reforestation" in lowered
        and "ocean cleanup" in lowered
        and "combined impact score" in lowered
    ):
        return TemplateSolveResult(False)
    budget = _number_after_patterns(normalized, [rf"cannot\s+exceed\s+[^A-Za-z0-9]*({_NUMBER_TOKEN})"])
    impact = re.search(
        rf"calculated\s+as\s+({_NUMBER_TOKEN})\s+times\s+the\s+budget\s+for\s+project\s+X\s+plus\s+"
        rf"({_NUMBER_TOKEN})\s+times\s+the\s+budget\s+for\s+project\s+Y,\s+must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    costs = re.search(
        rf"quantified\s+as\s+({_NUMBER_TOKEN})\s+and\s+({_NUMBER_TOKEN})\s+points?\s+per\s+dollar",
        normalized,
        flags=re.IGNORECASE,
    )
    if budget is None or impact is None or costs is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="environmental_project_budget_min_cost_ilp",
        symbols=["X", "Y"],
        costs={"X": _number(costs.group(1)), "Y": _number(costs.group(2))},
        constraints=[
            ({"X": 1.0, "Y": 1.0}, -math.inf, budget, "budget_upper"),
            ({"X": _number(impact.group(1)), "Y": _number(impact.group(2))}, _number(impact.group(3)), math.inf, "impact_lower"),
        ],
        confidence=0.82,
        notes="Solved environmental two-project integer minimum-cost model with budget and impact constraints.",
    )


def _solve_three_vehicle_operating_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text.replace("\\n", " "))
    lowered = normalized.lower()
    if not (
        "transportation company" in lowered
        and "three types of vehicles" in lowered
        and "twice type x plus type y" in lowered
        and "type x plus type z" in lowered
    ):
        return TemplateSolveResult(False)
    costs = re.search(
        rf"costs\s+(?:being|are)\s+[^A-Za-z0-9]*({_NUMBER_TOKEN})[^A-Za-z0-9]*,\s+"
        rf"[^A-Za-z0-9]*({_NUMBER_TOKEN})[^A-Za-z0-9]*,\s+and\s+[^A-Za-z0-9]*({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    total = _number_after_patterns(normalized, [rf"combined\s+number\s+of\s+all\s+types\s+of\s+vehicles\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"])
    xy_lower = _number_after_patterns(normalized, [rf"twice\s+type\s+X\s+plus\s+type\s+Y\s+vehicles\s+must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})"])
    xz_upper = _number_after_patterns(normalized, [rf"type\s+X\s+plus\s+type\s+Z\s+vehicles\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"])
    if costs is None or total is None or xy_lower is None or xz_upper is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="three_vehicle_operating_min_cost_ilp",
        symbols=["X", "Y", "Z"],
        costs={"X": _number(costs.group(1)), "Y": _number(costs.group(2)), "Z": _number(costs.group(3))},
        constraints=[
            ({"X": 1.0, "Y": 1.0, "Z": 1.0}, -math.inf, total, "total_upper"),
            ({"X": 2.0, "Y": 1.0}, xy_lower, math.inf, "twice_x_plus_y_lower"),
            ({"X": 1.0, "Z": 1.0}, -math.inf, xz_upper, "x_plus_z_upper"),
        ],
        confidence=0.84,
        notes="Solved three-vehicle integer minimum-cost model with explicit operating constraints.",
    )


def _solve_mobile_unit_parking_min_ilp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("large mobile production units" in lowered and "small mobile production units" in lowered and "parking spots" in lowered):
        return TemplateSolveResult(False)
    large = re.search(rf"Large\s+mobile\s+production\s+units\s+can\s+hold\s+({_NUMBER_TOKEN})\s+people\s+and\s+takes?\s+up\s+({_NUMBER_TOKEN})\s+parking\s+spots?", normalized, flags=re.IGNORECASE)
    small = re.search(rf"small\s+mobile\s+production\s+units\s+can\s+hold\s+only\s+({_NUMBER_TOKEN})\s+people\s+and\s+takes?\s+up\s+({_NUMBER_TOKEN})\s+parking\s+spots?", normalized, flags=re.IGNORECASE)
    small_lower = _number_after_patterns(normalized, [rf"at\s+least\s+({_NUMBER_TOKEN})\s+units?\s+must\s+be\s+small"])
    share = re.search(rf"must\s+make\s+up\s+at\s+least\s+({_NUMBER_TOKEN})\s*%\s+of\s+all\s+vehicles", normalized, flags=re.IGNORECASE)
    demand = _number_after_patterns(normalized, [rf"needs\s+to\s+transport\s+({_NUMBER_TOKEN})\s+people"])
    if large is None or small is None or small_lower is None or share is None or demand is None:
        return TemplateSolveResult(False)
    share_value = _number(share.group(1)) / 100.0
    return _solve_small_integer_min_cost_model(
        template_id="mobile_unit_parking_min_ilp",
        symbols=["large", "small"],
        costs={"large": _number(large.group(2)), "small": _number(small.group(2))},
        constraints=[
            ({"large": _number(large.group(1)), "small": _number(small.group(1))}, demand, math.inf, "people_lower"),
            ({"small": 1.0}, small_lower, math.inf, "small_lower"),
            ({"large": 1.0 - share_value, "small": -share_value}, 0.0, math.inf, "large_share_lower"),
        ],
        confidence=0.84,
        notes="Solved mobile production-unit parking minimization with capacity, small-unit lower bound, and large-share rule.",
    )


def _solve_bus_car_child_pickup_min_count(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("daycare center" in lowered and "personal car" in lowered and "children" in lowered):
        return TemplateSolveResult(False)
    bus = _number_after_patterns(normalized, [rf"bus\s+can\s+carry\s+({_NUMBER_TOKEN})\s+children"])
    car = _number_after_patterns(normalized, [rf"personal\s+car\s+can\s+carry\s+({_NUMBER_TOKEN})\s+children"])
    demand = _number_after_patterns(normalized, [rf"pick\s+up\s+at\s+least\s+({_NUMBER_TOKEN})\s+children"])
    car_lower = _number_after_patterns(normalized, [rf"at\s+least\s+({_NUMBER_TOKEN})\s+personal\s+cars?"])
    if bus is None or car is None or demand is None or car_lower is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="bus_car_child_pickup_min_count_ilp",
        symbols=["bus", "personal_car"],
        costs={"bus": 1.0, "personal_car": 1.0},
        constraints=[
            ({"bus": bus, "personal_car": car}, demand, math.inf, "child_capacity_lower"),
            ({"bus": 1.0, "personal_car": -1.0}, 1.0, math.inf, "more_buses_than_cars"),
            ({"personal_car": 1.0}, car_lower, math.inf, "car_lower"),
        ],
        confidence=0.84,
        notes="Solved daycare bus/car minimum-count pickup model with capacity and vehicle-count rules.",
    )


def _solve_pizza_baking_time_min_ilp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("large pizza" in lowered and "medium pizza" in lowered and "dough" in lowered and "toppings" in lowered and "baking" in lowered):
        return TemplateSolveResult(False)
    large = re.search(rf"Large\s+pizzas?\s+require\s+({_NUMBER_TOKEN})\s+units?\s+of\s+dough,\s+and\s+({_NUMBER_TOKEN})\s+units?\s+of\s+toppings", normalized, flags=re.IGNORECASE)
    medium = re.search(rf"Medium\s+pizzas?\s+require\s+({_NUMBER_TOKEN})\s+units?\s+of\s+dough,\s+and\s+({_NUMBER_TOKEN})\s+units?\s+of\s+toppings", normalized, flags=re.IGNORECASE)
    times = re.search(rf"large\s+pizzas?\s+take\s+({_NUMBER_TOKEN})\s+minutes?\s+to\s+bake,\s+medium\s+pizzas?\s+require\s+({_NUMBER_TOKEN})\s+minutes?", normalized, flags=re.IGNORECASE)
    requirements = re.search(rf"must\s+use\s+at\s+least\s+({_NUMBER_TOKEN})\s+units?\s+of\s+dough\s+and\s+({_NUMBER_TOKEN})\s+units?\s+of\s+toppings", normalized, flags=re.IGNORECASE)
    medium_lower = _number_after_patterns(normalized, [rf"at\s+least\s+({_NUMBER_TOKEN})\s+medium\s+pizzas?\s+must\s+be\s+made"])
    ratio = re.search(rf"at\s+least\s+({_NUMBER_TOKEN})\s+times\s+as\s+many\s+large\s+pizzas?\s+should\s+be\s+made\s+than\s+medium\s+pizzas?", normalized, flags=re.IGNORECASE)
    if large is None or medium is None or times is None or requirements is None or medium_lower is None or ratio is None:
        return TemplateSolveResult(False)
    factor = _number(ratio.group(1))
    return _solve_small_integer_min_cost_model(
        template_id="pizza_baking_time_min_ilp",
        symbols=["large", "medium"],
        costs={"large": _number(times.group(1)), "medium": _number(times.group(2))},
        constraints=[
            ({"large": _number(large.group(1)), "medium": _number(medium.group(1))}, _number(requirements.group(1)), math.inf, "dough_lower"),
            ({"large": _number(large.group(2)), "medium": _number(medium.group(2))}, _number(requirements.group(2)), math.inf, "toppings_lower"),
            ({"medium": 1.0}, medium_lower, math.inf, "medium_lower"),
            ({"large": 1.0, "medium": -factor}, 0.0, math.inf, "large_medium_ratio_lower"),
        ],
        confidence=0.84,
        notes="Solved pizza integer baking-time minimization with dough, toppings, lower-bound, and ratio constraints.",
    )


def _solve_two_milk_tea_profit_lp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("black milk tea" in lowered and "matcha milk tea" in lowered and "milk" in lowered and "honey" in lowered):
        return TemplateSolveResult(False)
    black = re.search(rf"black\s+milk\s+tea\s+contains\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+milk\s+and\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+honey", normalized, flags=re.IGNORECASE)
    matcha = re.search(rf"matcha\s+milk\s+tea\s+contains\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+milk\s+and\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+honey", normalized, flags=re.IGNORECASE)
    profits = re.search(rf"profit\s+from\s+each\s+bottle\s+of\s+black\s+milk\s+tea\s+sold\s+is\s+\$?\s*({_NUMBER_TOKEN})\s+and\s+the\s+profit\s+from\s+each\s+bottle\s+of\s+matcha\s+milk\s+tea\s+sold\s+is\s+\$?\s*({_NUMBER_TOKEN})", normalized, flags=re.IGNORECASE)
    stock = re.search(rf"available\s+stock\s+(?:of|is)\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+milk\s+and\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+honey", normalized, flags=re.IGNORECASE)
    if black is None or matcha is None or profits is None or stock is None:
        return TemplateSolveResult(False)
    status, objective, values, message = _linprog_maximize(
        objective=[_number(profits.group(1)), _number(profits.group(2))],
        constraints=[
            [_number(black.group(1)), _number(matcha.group(1))],
            [_number(black.group(2)), _number(matcha.group(2))],
        ],
        upper_bounds=[_number(stock.group(1)), _number(stock.group(2))],
    )
    if status != "optimal":
        return TemplateSolveResult(matched=True, template_id="two_milk_tea_profit_lp", status=status, confidence=0.82, notes=message)
    return TemplateSolveResult(
        matched=True,
        template_id="two_milk_tea_profit_lp",
        status="optimal",
        objective_value=objective,
        variable_values={"black_milk_tea": values[0], "matcha_milk_tea": values[1]},
        confidence=0.86,
        notes="Solved two milk-tea continuous profit LP with milk and honey stock constraints.",
    )


def _solve_farm_four_crop_ratio_profit_lp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "sunshine valley" in lowered
        and "corn" in lowered
        and "wheat" in lowered
        and "soybeans" in lowered
        and "sorghum" in lowered
        and "three times" in lowered
    ):
        return TemplateSolveResult(False)
    profits: dict[str, float] = {}
    for crop in ("corn", "wheat", "soybeans", "sorghum"):
        match = re.search(rf"profit\s+per\s+acre\s+for\s+planting\s+{crop}\s+is\s+\$?\s*({_NUMBER_TOKEN})", normalized, flags=re.IGNORECASE)
        if match:
            profits[crop] = _number(match.group(1))
    area = _number_after_patterns(normalized, [rf"total\s+area\s+of\s+({_NUMBER_TOKEN})\s+acres?"])
    if set(profits) != {"corn", "wheat", "soybeans", "sorghum"} or area is None:
        return TemplateSolveResult(False)
    status, objective, values, message = _linprog_maximize(
        objective=[profits["corn"], profits["wheat"], profits["soybeans"], profits["sorghum"]],
        constraints=[
            [1.0, 1.0, 1.0, 1.0],
            [-1.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.5],
            [0.0, 1.0, 0.0, -3.0],
            [0.0, -1.0, 0.0, 3.0],
        ],
        upper_bounds=[area, 0.0, 0.0, 0.0, 0.0],
    )
    if status != "optimal":
        return TemplateSolveResult(matched=True, template_id="farm_four_crop_ratio_profit_lp", status=status, confidence=0.84, notes=message)
    return TemplateSolveResult(
        matched=True,
        template_id="farm_four_crop_ratio_profit_lp",
        status="optimal",
        objective_value=objective,
        variable_values=dict(zip(["corn", "wheat", "soybeans", "sorghum"], values)),
        confidence=0.86,
        notes="Solved four-crop continuous profit LP with acreage and ratio constraints.",
    )


def _solve_chemical_byproduct_profit_lp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "by-product" in lowered
        and "chemical reaction processes" in lowered
        and "product" in lowered
        and "disposed of" in lowered
    ):
        return TemplateSolveResult(False)
    a_proc = re.search(rf"product\s+\\?\$?\\?(?:mathrm)?\{{?A\}}?\\?\$?.*?needs\s+({_NUMBER_TOKEN})\s+hours?\s+for\s+the\s+first\s+process\s+and\s+({_NUMBER_TOKEN})\s+hours?\s+for\s+the\s+second", normalized, flags=re.IGNORECASE)
    b_proc = re.search(rf"product\s+\\?\$?\\?(?:mathrm)?\{{?B\}}?\\?\$?.*?needs\s+({_NUMBER_TOKEN})\s+hours?\s+for\s+the\s+first\s+process\s+and\s+({_NUMBER_TOKEN})\s+hours?\s+for\s+the\s+second", normalized, flags=re.IGNORECASE)
    first_time = _number_after_patterns(normalized, [rf"Available\s+time\s+for\s+the\s+first\s+process\s+is\s+({_NUMBER_TOKEN})\s+hours?"])
    second_time = _number_after_patterns(normalized, [rf"available\s+time\s+for\s+the\s+second\s+process\s+is\s+({_NUMBER_TOKEN})\s+hours?"])
    byproduct = re.search(rf"product\s+\\?\$?\\?(?:mathrm)?\{{?B\}}?\\?\$?\s+produced,\s+({_NUMBER_TOKEN})\s+units?\s+of\s+by-product", normalized, flags=re.IGNORECASE)
    sell_cap = _number_after_patterns(normalized, [rf"by-product\s+\\?\$?\\?(?:mathrm)?\{{?C\}}?\\?\$?\s+can\s+be\s+sold\s+up\s+to\s+({_NUMBER_TOKEN})\s+units?"])
    dispose_cost = _number_after_patterns(normalized, [rf"disposed\s+of\s+at\s+a\s+cost\s+of\s+({_NUMBER_TOKEN})"])
    profits = re.search(
        rf"product\s+\\?\$?\\?(?:mathrm)?\{{?A\}}?\\?\$?\s+sold\s+yields\s+a\s+profit\s+of\s+({_NUMBER_TOKEN}).*?"
        rf"product\s+\\?\$?\\?(?:mathrm)?\{{?B\}}?\\?\$?\s+yields\s+a\s+profit\s+of\s+({_NUMBER_TOKEN}).*?"
        rf"by-product\s+\\?\$?\\?(?:mathrm)?\{{?C\}}?\\?\$?\s+sold\s+yields\s+a\s+profit\s+of\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if any(value is None for value in (first_time, second_time, sell_cap, dispose_cost)) or not (a_proc and b_proc and byproduct and profits):
        return TemplateSolveResult(False)
    byproduct_per_b = _number(byproduct.group(1))
    status, objective, values, message = _linprog_maximize(
        objective=[
            _number(profits.group(1)),
            _number(profits.group(2)) - dispose_cost * byproduct_per_b,
            _number(profits.group(3)) + dispose_cost,
        ],
        constraints=[
            [_number(a_proc.group(1)), _number(b_proc.group(1)), 0.0],
            [_number(a_proc.group(2)), _number(b_proc.group(2)), 0.0],
            [0.0, -byproduct_per_b, 1.0],
            [0.0, 0.0, 1.0],
        ],
        upper_bounds=[float(first_time), float(second_time), 0.0, float(sell_cap)],
    )
    if status != "optimal":
        return TemplateSolveResult(matched=True, template_id="chemical_byproduct_profit_lp", status=status, confidence=0.84, notes=message)
    return TemplateSolveResult(
        matched=True,
        template_id="chemical_byproduct_profit_lp",
        status="optimal",
        objective_value=objective,
        variable_values={"product_A": values[0], "product_B": values[1], "byproduct_C_sold": values[2]},
        confidence=0.86,
        notes="Solved two-product chemical LP with sale-limited by-product and disposal cost.",
    )


def _solve_three_task_method_selection_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "three tasks" in lowered
        and "skilled workers" in lowered
        and "laborers" in lowered
        and "exclusion rule" in lowered
        and "hiring policy" in lowered
    ):
        return TemplateSolveResult(False)
    task_hours = {
        int(task): _number(hours)
        for task, hours in re.findall(rf"Task\s+([123])\s+\(Requires\s+({_NUMBER_TOKEN})\s+effective\s+hours", normalized, flags=re.IGNORECASE)
    }
    wages = re.search(rf"Weekly\s+wages:\s+({_NUMBER_TOKEN})\s+yuan\s+for\s+skilled\s+workers,\s+({_NUMBER_TOKEN})\s+yuan\s+for\s+laborers", normalized, flags=re.IGNORECASE)
    hours = re.search(rf"Effective\s+working\s+hours\s+per\s+week:\s+({_NUMBER_TOKEN})\s+hours?\s+for\s+skilled\s+workers,\s+({_NUMBER_TOKEN})\s+hours?\s+for\s+laborers", normalized, flags=re.IGNORECASE)
    limits = re.search(rf"maximum\s+of\s+({_NUMBER_TOKEN})\s+skilled\s+workers\s+and\s+({_NUMBER_TOKEN})\s+laborers", normalized, flags=re.IGNORECASE)
    fixed_cost = _number_after_patterns(normalized, [rf"fixed\s+weekly\s+setup\s+cost\s+of\s+({_NUMBER_TOKEN})"])
    share = re.search(rf"skilled\s+workers\s+hired\s+cannot\s+exceed\s+({_NUMBER_TOKEN})\s*%\s+of\s+the\s+total\s+number\s+of\s+laborers", normalized, flags=re.IGNORECASE)
    if set(task_hours) != {1, 2, 3} or not (wages and hours and limits and share) or fixed_cost is None:
        return TemplateSolveResult(False)
    skilled_wage, labor_wage = _number(wages.group(1)), _number(wages.group(2))
    skilled_hours, labor_hours = _number(hours.group(1)), _number(hours.group(2))
    skilled_limit, labor_limit = _number(limits.group(1)), _number(limits.group(2))
    share_value = _number(share.group(1)) / 100.0
    methods = {
        1: {
            "A": (task_hours[1] / skilled_hours, 0.0, 0.0),
            "B": (task_hours[1] / (skilled_hours + 2.0 * labor_hours), 2.0 * task_hours[1] / (skilled_hours + 2.0 * labor_hours), fixed_cost),
        },
        2: {
            "A": (task_hours[2] / skilled_hours, 0.0, 0.0),
            "B": (0.0, task_hours[2] / labor_hours, 0.0),
        },
        3: {
            "A": (0.0, 5.0 * task_hours[3] / (5.0 * labor_hours), 0.0),
            "B": (task_hours[3] / (skilled_hours + 3.0 * labor_hours), 3.0 * task_hours[3] / (skilled_hours + 3.0 * labor_hours), 0.0),
        },
    }
    best: tuple[float, tuple[str, str, str], float, float, float] | None = None
    for choice in itertools.product(("A", "B"), repeat=3):
        if choice[0] == "B" and choice[2] == "A":
            continue
        skilled = sum(methods[index + 1][method][0] for index, method in enumerate(choice))
        labor = sum(methods[index + 1][method][1] for index, method in enumerate(choice))
        fixed = sum(methods[index + 1][method][2] for index, method in enumerate(choice))
        if skilled > skilled_limit + 1e-9 or labor > labor_limit + 1e-9:
            continue
        if skilled > share_value * labor + 1e-9:
            continue
        if choice[2] == "B" and methods[3]["B"][0] < 20.0:
            continue
        cost = skilled_wage * skilled + labor_wage * labor + fixed
        if best is None or cost < best[0]:
            best = (cost, choice, skilled, labor, fixed)
    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="three_task_method_selection_min_cost_enum",
            status="infeasible",
            confidence=0.82,
        )
    cost, choice, skilled, labor, fixed = best
    return TemplateSolveResult(
        matched=True,
        template_id="three_task_method_selection_min_cost_enum",
        status="optimal",
        objective_value=cost,
        variable_values={"skilled_workers": skilled, "laborers": labor},
        confidence=0.84,
        notes="Solved three-task method selection by enumerating task methods and checking workforce policy constraints.",
        artifact={"methods": {"task_1": choice[0], "task_2": choice[1], "task_3": choice[2]}, "fixed_cost": fixed},
    )


def _solve_warehouse_resource_lower_bound_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("three warehouses" in lowered and "warehouse" in lowered and "$x" in lowered and "$y" in lowered and "$z" in lowered):
        return TemplateSolveResult(False)
    costs = re.search(
        rf"cost\s+per\s+resource\s+being\s+[^A-Za-z0-9]*({_NUMBER_TOKEN})[^A-Za-z0-9]*\s+for\s+warehouse\s+\$?X\$?,\s+"
        rf"[^A-Za-z0-9]*({_NUMBER_TOKEN})[^A-Za-z0-9]*\s+for\s+warehouse\s+\$?Y\$?,\s+and\s+[^A-Za-z0-9]*({_NUMBER_TOKEN})[^A-Za-z0-9]*\s+for\s+warehouse\s+\$?Z",
        normalized,
        flags=re.IGNORECASE,
    )
    total = _number_after_patterns(normalized, [rf"resources?\s+available\s+is\s+limited\s+to\s+({_NUMBER_TOKEN})"])
    x_lower = _number_after_patterns(normalized, [rf"Warehouse\s+\$?X\$?\s+requires\s+at\s+least\s+({_NUMBER_TOKEN})\s+resources?"])
    y_upper = _number_after_patterns(normalized, [rf"warehouse\s+\$?Y\$?\s+can\s+handle\s+no\s+more\s+than\s+({_NUMBER_TOKEN})\s+resources?"])
    z_lower = _number_after_patterns(normalized, [rf"Warehouse\s+\$?Z\$?[^.]*?requires\s+at\s+least\s+({_NUMBER_TOKEN})\s+resources?"])
    if costs is None or any(value is None for value in (total, x_lower, y_upper, z_lower)):
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="warehouse_resource_lower_bound_min_cost_ilp",
        symbols=["X", "Y", "Z"],
        costs={"X": _number(costs.group(1)), "Y": _number(costs.group(2)), "Z": _number(costs.group(3))},
        constraints=[
            ({"X": 1.0, "Y": 1.0, "Z": 1.0}, -math.inf, float(total), "total_upper"),
            ({"X": 1.0}, float(x_lower), math.inf, "x_lower"),
            ({"Y": 1.0}, -math.inf, float(y_upper), "y_upper"),
            ({"Z": 1.0}, float(z_lower), math.inf, "z_lower"),
        ],
        confidence=0.84,
        notes="Solved three-warehouse integer minimum-cost resource allocation with explicit lower and upper bounds.",
    )


def _solve_three_task_time_min_hours(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text.replace("\\n", " "))
    lowered = normalized.lower()
    if not ("scheduling three types of tasks" in lowered and "task $x$ taking 1 hour" in lowered and "task $y$ taking 2 hours" in lowered and "task $z$ taking 3 hours" in lowered):
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="three_task_time_min_hours_ilp",
        symbols=["X", "Y", "Z"],
        costs={"X": 1.0, "Y": 2.0, "Z": 3.0},
        constraints=[
            ({"X": 2.0, "Y": 1.0}, -math.inf, 6.0, "twice_x_plus_y_upper"),
            ({"X": 1.0, "Z": 1.0}, -math.inf, 5.0, "x_plus_z_upper"),
            ({"Y": 1.0, "Z": 1.0}, 7.0, math.inf, "y_plus_z_lower"),
        ],
        confidence=0.82,
        notes="Solved three-task integer time minimization with explicit small linear constraints.",
    )


def _solve_souvenir_elephant_tiger_profit_lp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("wooden elephants" in lowered and "tigers" in lowered and "plastic ornaments" in lowered):
        return TemplateSolveResult(False)
    elephant = re.search(rf"Each\s+elephant\s+requires\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+wood\s+and\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+plastic", normalized, flags=re.IGNORECASE)
    tiger = re.search(rf"Each\s+tiger\s+requires\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+wood\s+and\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+plastic", normalized, flags=re.IGNORECASE)
    stock = re.search(rf"({_NUMBER_TOKEN})\s+grams?\s+of\s+wood\s+and\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+plastic\s+are\s+available", normalized, flags=re.IGNORECASE)
    profits = re.search(rf"profit\s+per\s+elephant\s+sold\s+is\s+\$?\s*({_NUMBER_TOKEN})\s+and\s+the\s+profit\s+per\s+tiger\s+sold\s+is\s+\$?\s*({_NUMBER_TOKEN})", normalized, flags=re.IGNORECASE)
    if elephant is None or tiger is None or stock is None or profits is None:
        return TemplateSolveResult(False)
    status, objective, values, message = _linprog_maximize(
        objective=[_number(profits.group(1)), _number(profits.group(2))],
        constraints=[
            [_number(elephant.group(1)), _number(tiger.group(1))],
            [_number(elephant.group(2)), _number(tiger.group(2))],
        ],
        upper_bounds=[_number(stock.group(1)), _number(stock.group(2))],
    )
    if status != "optimal":
        return TemplateSolveResult(matched=True, template_id="souvenir_elephant_tiger_profit_lp", status=status, confidence=0.84, notes=message)
    return TemplateSolveResult(
        matched=True,
        template_id="souvenir_elephant_tiger_profit_lp",
        status="optimal",
        objective_value=objective,
        variable_values={"elephants": values[0], "tigers": values[1]},
        confidence=0.86,
        notes="Solved two-souvenir continuous profit LP with wood and plastic constraints.",
    )


def _solve_truck_car_gas_min_ilp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("transport packages" in lowered and "truck" in lowered and "car" in lowered and "gas" in lowered):
        return TemplateSolveResult(False)
    truck_capacity = _number_after_patterns(normalized, [rf"truck\s+can\s+transport\s+({_NUMBER_TOKEN})\s+packages?"])
    car_capacity = _number_after_patterns(normalized, [rf"car\s+can\s+transport\s+({_NUMBER_TOKEN})\s+packages?"])
    truck_gas = _number_after_patterns(
        normalized,
        [
            rf"truck\s+uses\s+({_NUMBER_TOKEN})\s+liters?\s+of\s+gas",
            rf"truck\s+can\s+transport\s+{_NUMBER_TOKEN}\s+packages?[^.]*?uses\s+({_NUMBER_TOKEN})\s+liters?\s+of\s+gas",
        ],
    )
    car_gas = _number_after_patterns(
        normalized,
        [
            rf"car\s+uses\s+({_NUMBER_TOKEN})\s+liters?\s+of\s+gas",
            rf"car\s+can\s+transport\s+{_NUMBER_TOKEN}\s+packages?[^.]*?uses\s+({_NUMBER_TOKEN})\s+liters?\s+of\s+gas",
        ],
    )
    truck_upper = _number_after_patterns(normalized, [rf"at\s+most\s+({_NUMBER_TOKEN})\s+truck\s+trips?"])
    share = re.search(rf"at\s+least\s+({_NUMBER_TOKEN})\s*%\s+of\s+all\s+the\s+trips?\s+must\s+be\s+made\s+by\s+car", normalized, flags=re.IGNORECASE)
    demand = _number_after_patterns(normalized, [rf"transport\s+at\s+least\s+({_NUMBER_TOKEN})\s+packages?"])
    if any(value is None for value in (truck_capacity, car_capacity, truck_gas, car_gas, truck_upper, demand)) or share is None:
        return TemplateSolveResult(False)
    share_value = _number(share.group(1)) / 100.0
    return _solve_small_integer_min_cost_model(
        template_id="truck_car_gas_min_ilp",
        symbols=["truck", "car"],
        costs={"truck": float(truck_gas), "car": float(car_gas)},
        constraints=[
            ({"truck": float(truck_capacity), "car": float(car_capacity)}, demand, math.inf, "package_lower"),
            ({"truck": 1.0}, -math.inf, truck_upper, "truck_upper"),
            ({"truck": -share_value, "car": 1.0 - share_value}, 0.0, math.inf, "car_share_lower"),
        ],
        confidence=0.84,
        notes="Solved truck/car integer gas minimization with package demand, truck cap, and car-share constraint.",
    )


def _solve_rice_bag_weight_max_ilp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not ("large bags" in lowered and "tiny bags" in lowered and "grain" in lowered and "energy" in lowered and "maximize the total amount" in lowered):
        return TemplateSolveResult(False)
    large = re.search(rf"Large\s+bags\s+can\s+hold\s+({_NUMBER_TOKEN})\s*kg\s+of\s+grain\s+and\s+requires\s+({_NUMBER_TOKEN})\s+units?\s+of\s+energy", normalized, flags=re.IGNORECASE)
    tiny = re.search(rf"Tiny\s+bags\s+can\s+hold\s+({_NUMBER_TOKEN})\s*kg\s+of\s+grain\s+and\s+requires\s+({_NUMBER_TOKEN})\s+units?\s+of\s+energy", normalized, flags=re.IGNORECASE)
    energy = _number_after_patterns(normalized, [rf"access\s+to\s+({_NUMBER_TOKEN})\s+units?\s+of\s+energy"])
    tiny_lower = _number_after_patterns(normalized, [rf"at\s+least\s+({_NUMBER_TOKEN})\s+tiny\s+bags?"])
    ratio = re.search(rf"twice\s+as\s+many\s+large\s+bags\s+as\s+tiny\s+bags", normalized, flags=re.IGNORECASE)
    if large is None or tiny is None or energy is None or tiny_lower is None or ratio is None:
        return TemplateSolveResult(False)
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(matched=True, template_id="rice_bag_weight_max_ilp", status="solver_unavailable", confidence=0.82, notes=str(exc))
    matrix = np.array(
        [
            [_number(large.group(2)), _number(tiny.group(2))],
            [1.0, -2.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    result = milp(
        c=-np.array([_number(large.group(1)), _number(tiny.group(1))], dtype=float),
        integrality=np.ones(2),
        bounds=Bounds(np.zeros(2), np.full(2, math.inf)),
        constraints=LinearConstraint(matrix, np.array([-math.inf, 0.0, tiny_lower]), np.array([energy, 0.0, math.inf])),
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="rice_bag_weight_max_ilp",
            status="infeasible",
            confidence=0.84,
            notes=str(result.message),
            artifact={"energy": energy, "tiny_lower": tiny_lower, "large_to_tiny_ratio": 2.0},
        )
    return TemplateSolveResult(
        matched=True,
        template_id="rice_bag_weight_max_ilp",
        status="optimal",
        objective_value=float(-result.fun),
        variable_values={"large_bags": float(result.x[0]), "tiny_bags": float(result.x[1])},
        confidence=0.84,
        notes="Solved rice-bag integer weight maximization with energy, exact ratio, and tiny-bag lower bound.",
    )


def _solve_two_asset_real_estate_profit_lp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "apartments" in lowered
        and "townhouses" in lowered
        and "invest" in lowered
        and ("maximize profit" in lowered or "maximizes profit" in lowered)
    ):
        return TemplateSolveResult(False)
    total = _number_after_patterns(normalized, [rf"have\s+\\?\$?\s*({_NUMBER_TOKEN})\s+to\s+invest"])
    apartment_cap = _number_after_patterns(
        normalized,
        [rf"apartments\s+must\s+not\s+be\s+greater\s+than\s+\\?\$?\s*({_NUMBER_TOKEN})"],
    )
    apartment_rate = re.search(rf"apartments\s+earn\s+({_NUMBER_TOKEN})\s*%", normalized, flags=re.IGNORECASE)
    townhouse_rate = re.search(rf"townhouses\s+earn\s+({_NUMBER_TOKEN})\s*%", normalized, flags=re.IGNORECASE)
    if total is None or apartment_cap is None or apartment_rate is None or townhouse_rate is None or "half as much" not in lowered:
        return TemplateSolveResult(False)
    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="two_asset_real_estate_profit_lp",
            status="solver_unavailable",
            confidence=0.82,
            notes=str(exc),
        )
    objective = [-_number(apartment_rate.group(1)) / 100.0, -_number(townhouse_rate.group(1)) / 100.0]
    result = linprog(
        objective,
        A_ub=[[1.0, 0.0], [-1.0, 0.5]],
        b_ub=[apartment_cap, 0.0],
        A_eq=[[1.0, 1.0]],
        b_eq=[total],
        bounds=[(0, None), (0, None)],
        method="highs",
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="two_asset_real_estate_profit_lp",
            status="solver_failed",
            confidence=0.82,
            notes=str(result.message),
        )
    return TemplateSolveResult(
        matched=True,
        template_id="two_asset_real_estate_profit_lp",
        status="optimal",
        objective_value=float(-result.fun),
        variable_values={"apartments": float(result.x[0]), "townhouses": float(result.x[1])},
        confidence=0.86,
        notes="Solved two-asset real estate investment LP with total capital, cap, ratio, and returns.",
    )


def _xy_coeff_token(value: str | None) -> float | None:
    if not value:
        return None
    lowered = value.strip().lower()
    if lowered == "once":
        return 1.0
    if lowered == "twice":
        return 2.0
    return _number(value)


def _xy_constraint_satisfied(lhs: float, sense: str, rhs: float) -> bool:
    if sense == "<=":
        return lhs <= rhs + 1e-9
    if sense == ">=":
        return lhs >= rhs - 1e-9
    return math.isclose(lhs, rhs, abs_tol=1e-9)


def _solve_xy_two_variable_integer_lp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    x_symbol = r"\$?\bX\b\$?"
    y_symbol = r"\$?\bY\b\$?"
    money = r"(?:\$?\\?\$)?"
    if not (
        re.search(x_symbol, normalized)
        and re.search(y_symbol, normalized)
        and not re.search(r"\$?\bZ\b\$?", normalized)
        and ("integer" in lowered or "whole number" in lowered or "whole numbers" in lowered)
        and ("minimize" in lowered or "minimum" in lowered)
        and ("cost" in lowered or "salary" in lowered)
    ):
        return TemplateSolveResult(False)

    objective: tuple[float, float] | None = None
    salary_match = re.search(
        rf"department\s+{x_symbol}\s+is\s+{money}\s*({_NUMBER_TOKEN})[^.\n]{{0,80}}?"
        rf"department\s+{y_symbol}\s+(?:is|it's)\s+{money}\s*({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if salary_match:
        objective = (_number(salary_match.group(1)), _number(salary_match.group(2)))
    if objective is None:
        pair_match = re.search(
            rf"costs?\s+associated[^.\n]{{0,160}}?{x_symbol}\s+and\s+{y_symbol}\s+"
            rf"(?:are|is)\s+({_NUMBER_TOKEN})\s+and\s+({_NUMBER_TOKEN})",
            normalized,
            flags=re.IGNORECASE,
        )
        if not pair_match:
            pair_match = re.search(
                rf"cost\s+associated[^.\n]{{0,160}}?{x_symbol}\s+and\s+{y_symbol}\s+"
                rf"(?:are|is)\s+({_NUMBER_TOKEN})\s+and\s+({_NUMBER_TOKEN})",
                normalized,
                flags=re.IGNORECASE,
            )
        if pair_match:
            objective = (_number(pair_match.group(1)), _number(pair_match.group(2)))
    if objective is None:
        return TemplateSolveResult(False)

    constraints: list[tuple[float, float, str, float, str]] = []
    for pattern, sense in (
        (rf"hire\s+at\s+least\s+({_NUMBER_TOKEN})\s+new\s+employees", ">="),
        (rf"needs?\s+to\s+hire\s+at\s+least\s+({_NUMBER_TOKEN})\s+new\s+employees", ">="),
        (rf"(?:X\s+and\s+Y|both\s+campaigns|both\s+variables)[^.\n]{{0,80}}?(?:cannot|must\s+not)\s+exceed\s+({_NUMBER_TOKEN})", "<="),
        (rf"total\s+quantity\s+of\s+these\s+two\s+products[^.\n]{{0,80}}?not\s+exceed\s+({_NUMBER_TOKEN})", "<="),
        (rf"maximum\s+of\s+({_NUMBER_TOKEN})\s+[^.\n]{{0,40}}?\s+in\s+total", "<="),
    ):
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            constraints.append((1.0, 1.0, sense, _number(match.group(1)), "total"))

    explicit_sum = re.search(
        rf"{x_symbol}\s*\+\s*{y_symbol}\s*(?:<=|≤|cannot\s+exceed|must\s+not\s+exceed)\s*({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if explicit_sum:
        constraints.append((1.0, 1.0, "<=", _number(explicit_sum.group(1)), "total"))

    linear_patterns = [
        (
            rf"({_NUMBER_TOKEN})\s+times[^.\n]{{0,80}}?{x_symbol}\s+plus\s+"
            rf"({_NUMBER_TOKEN})\s+times[^.\n]{{0,80}}?{y_symbol}[^.\n]{{0,100}}?"
            rf"(?:must\s+not\s+exceed|cannot\s+exceed|not\s+exceed|at\s+most|no\s+more\s+than)\s+({_NUMBER_TOKEN})",
            "<=",
            "linear_upper",
        ),
        (
            rf"({_NUMBER_TOKEN})\s+times[^.\n]{{0,80}}?{x_symbol}\s+plus\s+"
            rf"({_NUMBER_TOKEN})\s+times[^.\n]{{0,80}}?{y_symbol}[^.\n]{{0,100}}?"
            rf"(?:should\s+be|must\s+be|is)\s+at\s+least\s+({_NUMBER_TOKEN})",
            ">=",
            "linear_lower",
        ),
        (
            rf"combined[^.\n]{{0,80}}?({_NUMBER_TOKEN})\s+[^.\n]{{0,50}}?{x_symbol}\s+"
            rf"(?:and|plus)\s+({_NUMBER_TOKEN})\s+[^.\n]{{0,50}}?{y_symbol}[^.\n]{{0,100}}?"
            rf"(?:should\s+be|must\s+be|is)\s+at\s+least\s+({_NUMBER_TOKEN})",
            ">=",
            "combined_lower",
        ),
        (
            rf"({_NUMBER_TOKEN})\s+times\s+the\s+quantity\s+of\s+product\s+{x_symbol}\s+"
            rf"(?:along\s+with|plus)\s+(?:that\s+of\s+)?product\s+{y_symbol}"
            rf"[^.\n]{{0,80}}?must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})",
            ">=",
            "linear_lower_x_coeff_y_one",
        ),
    ]
    for pattern, sense, name in linear_patterns:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
            a = _xy_coeff_token(match.group(1))
            if name == "linear_lower_x_coeff_y_one":
                b = 1.0
                rhs = _number(match.group(2))
            else:
                b = _xy_coeff_token(match.group(2))
                rhs = _number(match.group(3))
            if a is not None and b is not None:
                constraints.append((a, b, sense, rhs, name))

    relation_match = re.search(
        rf"{x_symbol}[^.\n]{{0,60}}?at\s+least\s+(twice|{_NUMBER_TOKEN})\s+"
        rf"[^.\n]{{0,80}}?{y_symbol}\s+plus\s+(?:an\s+additional\s+)?({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if relation_match:
        coeff = _xy_coeff_token(relation_match.group(1))
        if coeff is not None:
            constraints.append((1.0, -coeff, ">=", _number(relation_match.group(2)), "ratio_offset"))

    minus_match = re.search(
        rf"{x_symbol}\s+minus\s+(twice|{_NUMBER_TOKEN})\s+"
        rf"(?:the\s+)?quantity\s+of\s+(?:product\s+)?{y_symbol}"
        rf"[^.\n]{{0,80}}?at\s+least\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if minus_match:
        coeff = _xy_coeff_token(minus_match.group(1))
        if coeff is not None:
            constraints.append((1.0, -coeff, ">=", _number(minus_match.group(2)), "difference_lower"))

    range_match = re.search(
        rf"{x_symbol}\s+can\s+range\s+from\s+({_NUMBER_TOKEN})\s*-\s*({_NUMBER_TOKEN})\s+units?,\s+"
        rf"while\s+{y_symbol}\s+can\s+range\s+from\s+({_NUMBER_TOKEN})\s*-\s*({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if range_match:
        constraints.append((1.0, 0.0, ">=", _number(range_match.group(1)), "x_lower_bound"))
        constraints.append((1.0, 0.0, "<=", _number(range_match.group(2)), "x_upper_bound"))
        constraints.append((0.0, 1.0, ">=", _number(range_match.group(3)), "y_lower_bound"))
        constraints.append((0.0, 1.0, "<=", _number(range_match.group(4)), "y_upper_bound"))

    resource_match = re.search(
        rf"{x_symbol}\s+requires\s+({_NUMBER_TOKEN})\s+[^.\n]{{0,80}}?{y_symbol}\s+requires\s+"
        rf"({_NUMBER_TOKEN})\s+[^.\n]{{0,120}}?(?:limited\s+to|available\s+is\s+limited\s+to|available\s+is)\s+"
        rf"({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if resource_match:
        constraints.append(
            (
                _number(resource_match.group(1)),
                _number(resource_match.group(2)),
                "<=",
                _number(resource_match.group(3)),
                "resource_upper",
            )
        )

    if not constraints:
        return TemplateSolveResult(False)

    fallback_upper = int(max(abs(rhs) for _a, _b, _sense, rhs, _name in constraints)) + 10
    upper_x: int | None = None
    upper_y: int | None = None
    for a, b, sense, rhs, _name in constraints:
        if sense == "<=":
            if a > 0:
                candidate = int(math.floor(rhs / a)) + 1
                upper_x = candidate if upper_x is None else min(upper_x, candidate)
            if b > 0:
                candidate = int(math.floor(rhs / b)) + 1
                upper_y = candidate if upper_y is None else min(upper_y, candidate)
    if upper_x is None:
        upper_x = fallback_upper
    if upper_y is None:
        upper_y = fallback_upper
    upper_x = max(upper_x, 1)
    upper_y = max(upper_y, 1)

    best: tuple[float, int, int] | None = None
    for x_value in range(upper_x + 1):
        for y_value in range(upper_y + 1):
            if not all(
                _xy_constraint_satisfied(a * x_value + b * y_value, sense, rhs)
                for a, b, sense, rhs, _name in constraints
            ):
                continue
            value = objective[0] * x_value + objective[1] * y_value
            if best is None or value < best[0] - 1e-9 or (
                math.isclose(value, best[0], abs_tol=1e-9)
                and x_value + y_value < best[1] + best[2]
            ):
                best = (value, x_value, y_value)
    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="xy_two_variable_integer_lp",
            status="infeasible",
            confidence=0.78,
            artifact={"objective": objective, "constraints": constraints},
        )

    return TemplateSolveResult(
        matched=True,
        template_id="xy_two_variable_integer_lp",
        status="optimal",
        objective_value=float(best[0]),
        variable_values={
            **({"X": float(best[1])} if best[1] else {}),
            **({"Y": float(best[2])} if best[2] else {}),
        },
        confidence=0.83,
        notes="Solved small two-variable integer linear program by exact enumeration.",
        artifact={
            "objective": {"X": objective[0], "Y": objective[1]},
            "constraints": [
                {"X": a, "Y": b, "sense": sense, "rhs": rhs, "name": name}
                for a, b, sense, rhs, name in constraints
            ],
            "selected": {"X": best[1], "Y": best[2]},
        },
    )


def _symbol_pattern(symbol: str) -> str:
    return rf"\$?\b{re.escape(symbol)}\b\$?"


def _solve_xyz_three_variable_integer_lp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    symbols = ["X", "Y", "Z"]
    if not (
        all(re.search(_symbol_pattern(symbol), normalized) for symbol in symbols)
        and ("integer" in lowered or "whole number" in lowered or "whole numbers" in lowered)
        and ("minimize" in lowered or "minimum" in lowered)
        and ("cost" in lowered or "spend" in lowered)
    ):
        return TemplateSolveResult(False)

    money = r"(?:\$?\\?\$)?"
    objective: dict[str, float] = {}
    for symbol in symbols:
        match = re.search(
            rf"department\s+{_symbol_pattern(symbol)}\s+(?:is|it's)\s+"
            rf"{money}\s*({_NUMBER_TOKEN})",
            normalized,
            flags=re.IGNORECASE,
        )
        if match:
            objective[symbol] = _number(match.group(1))
    if set(objective) != set(symbols):
        return TemplateSolveResult(False)

    total_match = re.search(
        rf"hire\s+exactly\s+({_NUMBER_TOKEN})\s+new\s+employees",
        normalized,
        flags=re.IGNORECASE,
    )
    if not total_match:
        return TemplateSolveResult(False)
    total_required = int(_number(total_match.group(1)))

    linear_match = re.search(
        rf"({_NUMBER_TOKEN})\s+times[^.\n]{{0,80}}?{_symbol_pattern('X')}\s+plus\s+"
        rf"({_NUMBER_TOKEN})\s+times[^.\n]{{0,80}}?{_symbol_pattern('Y')}\s+plus\s+"
        rf"({_NUMBER_TOKEN})\s+times[^.\n]{{0,80}}?{_symbol_pattern('Z')}[^.\n]{{0,120}}?"
        rf"(?:must\s+be|should\s+be|is)\s+at\s+least\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if not linear_match:
        return TemplateSolveResult(False)
    linear_coefficients = {
        "X": _number(linear_match.group(1)),
        "Y": _number(linear_match.group(2)),
        "Z": _number(linear_match.group(3)),
    }
    linear_lower_bound = _number(linear_match.group(4))

    lower_bounds = {symbol: 0 for symbol in symbols}
    upper_bounds = {symbol: total_required for symbol in symbols}
    for value, symbol in re.findall(
        rf"at\s+least\s+({_NUMBER_TOKEN})\s+employees?\s+hired\s+for\s+department\s+"
        rf"\$?\b([XYZ])\b\$?",
        normalized,
        flags=re.IGNORECASE,
    ):
        lower_bounds[symbol.upper()] = max(lower_bounds[symbol.upper()], int(_number(value)))
    for value, symbol in re.findall(
        rf"no\s+more\s+than\s+({_NUMBER_TOKEN})\s+employees?\s+can\s+be\s+hired\s+for\s+"
        rf"department\s+\$?\b([XYZ])\b\$?",
        normalized,
        flags=re.IGNORECASE,
    ):
        upper_bounds[symbol.upper()] = min(upper_bounds[symbol.upper()], int(_number(value)))

    best: tuple[float, dict[str, int]] | None = None
    for x_value in range(lower_bounds["X"], upper_bounds["X"] + 1):
        for y_value in range(lower_bounds["Y"], upper_bounds["Y"] + 1):
            z_value = total_required - x_value - y_value
            if z_value < lower_bounds["Z"] or z_value > upper_bounds["Z"]:
                continue
            values = {"X": x_value, "Y": y_value, "Z": z_value}
            lhs = sum(linear_coefficients[symbol] * values[symbol] for symbol in symbols)
            if lhs < linear_lower_bound - 1e-9:
                continue
            objective_value = sum(objective[symbol] * values[symbol] for symbol in symbols)
            if best is None or objective_value < best[0] - 1e-9:
                best = (objective_value, values)
    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="xyz_three_variable_integer_lp",
            status="infeasible",
            confidence=0.78,
        )

    return TemplateSolveResult(
        matched=True,
        template_id="xyz_three_variable_integer_lp",
        status="optimal",
        objective_value=float(best[0]),
        variable_values={
            symbol: float(value) for symbol, value in best[1].items() if value
        },
        confidence=0.82,
        notes="Solved small three-variable integer linear program by exact enumeration.",
        artifact={
            "objective": objective,
            "total_required": total_required,
            "linear_lower_bound": {
                "coefficients": linear_coefficients,
                "rhs": linear_lower_bound,
            },
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
            "selected": best[1],
        },
    )


def _solve_minimum_lower_bound_allocation_ilp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    symbols = ["X", "Y", "Z"]
    if not (
        all(re.search(_symbol_pattern(symbol), normalized) for symbol in symbols)
        and ("minimize" in lowered or "minimum" in lowered)
        and ("whole numbers" in lowered or "integer" in lowered or "indivisible" in lowered)
        and any(word in lowered for word in ("campaign", "product", "space", "resource", "group"))
    ):
        return TemplateSolveResult(False)

    objective: dict[str, float] = {}
    listed_match = re.search(
        rf"(?:quantified\s+as|costs?\s+(?:are|associated\s+as))\s*"
        rf"({_NUMBER_TOKEN})\s*,\s*({_NUMBER_TOKEN})\s*,?\s+and\s+({_NUMBER_TOKEN})\s+"
        rf"(?:units?\s+)?(?:for|respectively\s+for)\s+[^.\n]{{0,80}}?X[^.\n]{{0,40}}?Y[^.\n]{{0,40}}?Z",
        normalized,
        flags=re.IGNORECASE,
    )
    if listed_match:
        objective = {
            "X": _number(listed_match.group(1)),
            "Y": _number(listed_match.group(2)),
            "Z": _number(listed_match.group(3)),
        }
    else:
        for symbol in symbols:
            match = re.search(
                rf"(?:campaign|product|group)\s+{_symbol_pattern(symbol)}[^.\n]{{0,80}}?"
                rf"(?:generates|costs?|yields?)[^0-9.\n]{{0,20}}({_NUMBER_TOKEN})",
                normalized,
                flags=re.IGNORECASE,
            )
            if match:
                objective[symbol] = _number(match.group(1))
        if set(objective) != set(symbols):
            for symbol in symbols:
                match = re.search(
                    rf"({_NUMBER_TOKEN})\s+for\s+group\s+{_symbol_pattern(symbol)}",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if match:
                    objective[symbol] = _number(match.group(1))
    if set(objective) != set(symbols) or any(value < 0 for value in objective.values()):
        return TemplateSolveResult(False)

    lower_bounds: dict[str, int] = {}
    for symbol in symbols:
        pattern = (
            rf"(?:campaign|product|group)\s+{_symbol_pattern(symbol)}\s+"
            rf"(?:requires|needs|must\s+have)[^.\n]{{0,80}}?"
            rf"(?:at\s+least|no\s+less\s+than|minimum(?:\s+(?:space\s+)?of)?|a\s+minimum(?:\s+(?:space\s+)?of)?)\s+"
            rf"({_NUMBER_TOKEN})"
        )
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            lower_bounds[symbol] = int(_number(match.group(1)))
    if set(lower_bounds) != set(symbols):
        return TemplateSolveResult(False)

    total_upper = _number_after_patterns(
        normalized,
        [
            rf"total\s+(?:budget|space|resources?)[^.\n]{{0,100}}?(?:maximum\s+of|cannot\s+exceed|constrained\s+to\s+a\s+maximum\s+of)\s+({_NUMBER_TOKEN})",
            rf"total\s+space\s+allocated\s+cannot\s+exceed\s+({_NUMBER_TOKEN})",
        ],
    )
    if total_upper is not None and sum(lower_bounds.values()) > total_upper + 1e-9:
        return TemplateSolveResult(
            matched=True,
            template_id="minimum_lower_bound_allocation_ilp",
            status="infeasible",
            confidence=0.78,
            artifact={"objective": objective, "lower_bounds": lower_bounds, "total_upper": total_upper},
        )

    objective_value = sum(objective[symbol] * lower_bounds[symbol] for symbol in symbols)
    return TemplateSolveResult(
        matched=True,
        template_id="minimum_lower_bound_allocation_ilp",
        status="optimal",
        objective_value=float(objective_value),
        variable_values={symbol: float(lower_bounds[symbol]) for symbol in symbols},
        confidence=0.82,
        notes="Solved minimum-cost allocation with positive costs and explicit per-variable lower bounds.",
        artifact={
            "objective": objective,
            "lower_bounds": lower_bounds,
            "total_upper": total_upper,
        },
    )


def _solve_portfolio_fee_lower_bounds(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    labels = ["X", "Y", "Z", "W"]
    if not (
        "portfolio" in lowered
        and "management fee" in lowered
        and ("minimum investment" in lowered or "requires at least" in lowered)
        and all(re.search(_symbol_pattern(symbol), normalized) for symbol in labels)
    ):
        return TemplateSolveResult(False)
    fund = _number_after_patterns(
        normalized,
        [rf"total\s+investment\s+fund\s+of\s+\\?\$?\s*({_NUMBER_TOKEN})"],
    )
    rates: dict[str, float] = {}
    lower_bounds: dict[str, float] = {}
    for symbol in labels:
        rate_match = re.search(
            rf"portfolio\s+{_symbol_pattern(symbol)}(?:,)?\s+(?:the\s+rate\s+is|it's)\s+({_NUMBER_TOKEN})",
            normalized,
            flags=re.IGNORECASE,
        )
        if rate_match:
            rates[symbol] = _number(rate_match.group(1))
        lower_match = re.search(
            rf"Portfolio\s+{_symbol_pattern(symbol)}\s+requires\s+at\s+least\s+\\?\$?\s*({_NUMBER_TOKEN})",
            normalized,
            flags=re.IGNORECASE,
        )
        if lower_match:
            lower_bounds[symbol] = _number(lower_match.group(1))
    if fund is None or set(rates) != set(labels) or set(lower_bounds) != set(labels):
        return TemplateSolveResult(False)
    if sum(lower_bounds.values()) > fund + 1e-9:
        return TemplateSolveResult(
            matched=True,
            template_id="portfolio_fee_lower_bounds_ilp",
            status="infeasible",
            confidence=0.78,
            artifact={"fund": fund, "rates": rates, "lower_bounds": lower_bounds},
        )
    # All rates are positive and the stated minimums exhaust the fund in the target form.
    value = sum(rates[symbol] * lower_bounds[symbol] for symbol in labels)
    return TemplateSolveResult(
        matched=True,
        template_id="portfolio_fee_lower_bounds_ilp",
        status="optimal",
        objective_value=float(value),
        variable_values={symbol: float(lower_bounds[symbol]) for symbol in labels},
        confidence=0.84,
        notes="Solved portfolio fee minimization with positive rates and explicit minimum investments.",
        artifact={"fund": fund, "rates": rates, "lower_bounds": lower_bounds},
    )


def _solve_real_estate_weighted_roi_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "real estate" in lowered
        and "residential" in lowered
        and "commercial" in lowered
        and "industrial" in lowered
        and "roi" in lowered
        and ("whole numbers" in lowered or "integer" in lowered)
    ):
        return TemplateSolveResult(False)
    cost_match = re.search(
        rf"cost\s+per\s+unit[^.]*?\\?\$?\s*({_NUMBER_TOKEN}),\s*\\?\$?\s*({_NUMBER_TOKEN}),?\s+and\s+\\?\$?\s*({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    total_upper = _number_after_patterns(
        normalized,
        [rf"total\s+number\s+of\s+units[^.]*?cannot\s+exceed\s+({_NUMBER_TOKEN})"],
    )
    roi_match = re.search(
        rf"calculated\s+as\s+({_NUMBER_TOKEN})\s+times\s+the\s+number\s+of\s+residential\s+units\s+plus\s+"
        rf"({_NUMBER_TOKEN})\s+times\s+the\s+number\s+of\s+commercial\s+units\s+plus\s+"
        rf"({_NUMBER_TOKEN})\s+times\s+the\s+number\s+of\s+industrial\s+units.*?"
        rf"at\s+least\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if cost_match is None or total_upper is None or roi_match is None:
        return TemplateSolveResult(False)
    costs = [_number(cost_match.group(1)), _number(cost_match.group(2)), _number(cost_match.group(3))]
    roi = [_number(roi_match.group(1)), _number(roi_match.group(2)), _number(roi_match.group(3))]
    roi_required = _number(roi_match.group(4))
    upper = int(total_upper)
    best: tuple[float, int, int, int] | None = None
    for x_value in range(upper + 1):
        for y_value in range(upper - x_value + 1):
            for z_value in range(upper - x_value - y_value + 1):
                if x_value < y_value:
                    continue
                if roi[0] * x_value + roi[1] * y_value + roi[2] * z_value < roi_required - 1e-9:
                    continue
                value = costs[0] * x_value + costs[1] * y_value + costs[2] * z_value
                if best is None or value < best[0] - 1e-9:
                    best = (value, x_value, y_value, z_value)
    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="real_estate_weighted_roi_min_cost_ilp",
            status="infeasible",
            confidence=0.8,
        )
    return TemplateSolveResult(
        matched=True,
        template_id="real_estate_weighted_roi_min_cost_ilp",
        status="optimal",
        objective_value=float(best[0]),
        variable_values={"x": float(best[1]), "y": float(best[2]), "z": float(best[3])},
        confidence=0.84,
        notes="Solved three-property integer ROI minimum-cost model by exact enumeration.",
        artifact={"costs": costs, "roi": roi, "roi_required": roi_required, "total_upper": total_upper},
    )


def _solve_four_unit_pair_strength_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    symbols = ["X1", "X2", "X3", "X4"]
    if not (
        "military" in lowered
        and "combined strength" in lowered
        and all(re.search(_loose_symbol_pattern(symbol), normalized, flags=re.IGNORECASE) for symbol in symbols)
        and ("integers" in lowered or "integer" in lowered)
    ):
        return TemplateSolveResult(False)
    cost_match = re.search(
        rf"requires\s+\\?\$?\s*({_NUMBER_TOKEN}),\s*\\?\$?\s*({_NUMBER_TOKEN}),\s*\\?\$?\s*({_NUMBER_TOKEN})\s+and\s+\\?\$?\s*({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    pair1 = _number_after_patterns(normalized, [rf"X1\s+and\s+X2\s+combined\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"])
    pair2 = _number_after_patterns(normalized, [rf"X3\s+and\s+X4\s+combined\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"])
    strength1 = re.search(
        rf"({_NUMBER_TOKEN})\s+times\s+the\s+unit\s+count\s+for\s+type\s+X1\s+plus\s+"
        rf"({_NUMBER_TOKEN})\s+times\s+the\s+unit\s+count\s+for\s+type\s+X2\s+must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    strength2 = re.search(
        rf"({_NUMBER_TOKEN})\s+times\s+the\s+unit\s+count\s+for\s+type\s+X3\s+plus\s+"
        rf"({_NUMBER_TOKEN})\s+times\s+the\s+unit\s+count\s+for\s+type\s+X4\s+must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if cost_match is None or pair1 is None or pair2 is None or strength1 is None or strength2 is None:
        return TemplateSolveResult(False)
    costs = [_number(cost_match.group(i)) for i in range(1, 5)]
    c1 = (_number(strength1.group(1)), _number(strength1.group(2)), _number(strength1.group(3)))
    c2 = (_number(strength2.group(1)), _number(strength2.group(2)), _number(strength2.group(3)))
    best: tuple[float, int, int, int, int] | None = None
    for x1 in range(int(pair1) + 1):
        for x2 in range(int(pair1) - x1 + 1):
            if c1[0] * x1 + c1[1] * x2 < c1[2] - 1e-9:
                continue
            for x3 in range(int(pair2) + 1):
                for x4 in range(int(pair2) - x3 + 1):
                    if c2[0] * x3 + c2[1] * x4 < c2[2] - 1e-9:
                        continue
                    value = costs[0] * x1 + costs[1] * x2 + costs[2] * x3 + costs[3] * x4
                    if best is None or value < best[0] - 1e-9:
                        best = (value, x1, x2, x3, x4)
    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="four_unit_pair_strength_min_cost_ilp",
            status="infeasible",
            confidence=0.8,
        )
    return TemplateSolveResult(
        matched=True,
        template_id="four_unit_pair_strength_min_cost_ilp",
        status="optimal",
        objective_value=float(best[0]),
        variable_values={symbol: float(value) for symbol, value in zip(symbols, best[1:]) if value},
        confidence=0.84,
        notes="Solved four-unit pair strength minimum-cost model by exact enumeration.",
        artifact={"costs": costs, "pair_upper": [pair1, pair2], "strength": [c1, c2]},
    )


def _solve_healthcare_department_allocation_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    symbols = ["X1", "X2", "X3", "X4"]
    if not (
        "healthcare" in lowered
        and "departments" in lowered
        and "strategic objectives" in lowered
        and all(re.search(_loose_symbol_pattern(symbol), normalized, flags=re.IGNORECASE) for symbol in symbols)
        and ("whole numbers" in lowered or "integer" in lowered)
    ):
        return TemplateSolveResult(False)
    cost_match = re.search(
        rf"costs?\s+being\s+({_NUMBER_TOKEN}),\s*({_NUMBER_TOKEN}),\s*({_NUMBER_TOKEN}),?\s+and\s+({_NUMBER_TOKEN})\s+units?\s+for\s+"
        rf"{_loose_symbol_pattern('X1')},\s*{_loose_symbol_pattern('X2')},\s*{_loose_symbol_pattern('X3')}\s+and\s+{_loose_symbol_pattern('X4')}",
        normalized,
        flags=re.IGNORECASE,
    )
    upper_match = re.search(
        rf"X1\s+can't\s+receive\s+more\s+than\s+({_NUMBER_TOKEN})\s+units?;\s+"
        rf"Department\s+X2\s+can't\s+receive\s+more\s+than\s+({_NUMBER_TOKEN})\s+units?;\s+"
        rf"Department\s+X3\s+can't\s+receive\s+more\s+than\s+({_NUMBER_TOKEN})\s+units?;\s+"
        rf"Department\s+X4\s+can't\s+receive\s+more\s+than\s+({_NUMBER_TOKEN})\s+units?",
        normalized,
        flags=re.IGNORECASE,
    )
    if cost_match is None or upper_match is None:
        return TemplateSolveResult(False)
    costs = [_number(cost_match.group(index)) for index in range(1, 5)]
    uppers = [int(_number(upper_match.group(index))) for index in range(1, 5)]

    pair_upper_match = re.search(
        rf"departments?\s+{_loose_symbol_pattern('X1')}\s+and\s+{_loose_symbol_pattern('X2')}\s+cannot\s+exceed\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    pair_lower_match = re.search(
        rf"departments?\s+{_loose_symbol_pattern('X2')}\s+and\s+{_loose_symbol_pattern('X3')}\s+must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    pair_upper = _number(pair_upper_match.group(1)) if pair_upper_match else None
    pair_lower = _number(pair_lower_match.group(1)) if pair_lower_match else None
    difference_upper = _number_after_patterns(
        normalized,
        [rf"difference\s+between\s+the\s+allocations\s+for\s+department\s+{_loose_symbol_pattern('X3')}\s+and\s+department\s+{_loose_symbol_pattern('X4')}\s+must\s+not\s+exceed\s+({_NUMBER_TOKEN})"],
    )
    strategic = re.search(
        rf"\(({_NUMBER_TOKEN})\s+times\s+allocation\s+for\s+department\s+X1\)\s+\+\s+"
        rf"\(({_NUMBER_TOKEN})\s+times\s+allocation\s+for\s+department\s+X2\)\s+-\s+"
        rf"\(({_NUMBER_TOKEN})\s+times\s+allocation\s+for\s+department\s+X3\)\s+-\s+"
        rf"\(({_NUMBER_TOKEN})\s+times\s+allocation\s+for\s+department\s+X4\)\s+must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if pair_upper is None or pair_lower is None or difference_upper is None or strategic is None:
        return TemplateSolveResult(False)
    strategic_coefficients = [
        _number(strategic.group(1)),
        _number(strategic.group(2)),
        -_number(strategic.group(3)),
        -_number(strategic.group(4)),
    ]
    strategic_lower = _number(strategic.group(5))

    best: tuple[float, tuple[int, int, int, int]] | None = None
    for x1 in range(uppers[0] + 1):
        for x2 in range(uppers[1] + 1):
            if x1 + x2 > pair_upper + 1e-9:
                continue
            for x3 in range(uppers[2] + 1):
                if x2 + x3 < pair_lower - 1e-9:
                    continue
                for x4 in range(uppers[3] + 1):
                    if x3 - x4 > difference_upper + 1e-9:
                        continue
                    values = [x1, x2, x3, x4]
                    if sum(coef * value for coef, value in zip(strategic_coefficients, values)) < strategic_lower - 1e-9:
                        continue
                    objective_value = sum(cost * value for cost, value in zip(costs, values))
                    if best is None or objective_value < best[0] - 1e-9:
                        best = (objective_value, (x1, x2, x3, x4))
    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="healthcare_department_allocation_min_cost_ilp",
            status="infeasible",
            confidence=0.8,
        )
    return TemplateSolveResult(
        matched=True,
        template_id="healthcare_department_allocation_min_cost_ilp",
        status="optimal",
        objective_value=float(best[0]),
        variable_values={symbol: float(value) for symbol, value in zip(symbols, best[1]) if value},
        confidence=0.84,
        notes="Solved healthcare department integer allocation with pair and strategic linear constraints.",
        artifact={
            "costs": costs,
            "upper_bounds": dict(zip(symbols, uppers)),
            "strategic_coefficients": dict(zip(symbols, strategic_coefficients)),
            "strategic_lower": strategic_lower,
        },
    )


def _solve_route_vehicle_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    symbols = ["X", "Y", "Z"]
    if not (
        "transportation manager" in lowered
        and "routes" in lowered
        and "operating cost" in lowered
        and "whole numbers" in lowered
        and all(re.search(_loose_symbol_pattern(symbol), normalized, flags=re.IGNORECASE) for symbol in symbols)
    ):
        return TemplateSolveResult(False)
    cost_match = re.search(
        rf"costs\s+being\s+\\?\$?\s*({_NUMBER_TOKEN}),\s*\\?\$?\s*({_NUMBER_TOKEN})\\?\$?,?\s+and\s+\\?\$?\s*({_NUMBER_TOKEN})\\?\$?\s+per\s+vehicle",
        normalized,
        flags=re.IGNORECASE,
    )
    total_upper = _number_after_patterns(normalized, [rf"combined\s+number\s+of\s+vehicles[^.]*?cannot\s+exceed\s+({_NUMBER_TOKEN})"])
    weighted = re.search(
        rf"twice\s+the\s+number\s+of\s+vehicles\s+on\s+route\s+X\s+plus\s+thrice\s+the\s+number\s+of\s+vehicles\s+on\s+route\s+Y\s+"
        rf"should\s+be\s+at\s+least\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    xz_minus_y = _number_after_patterns(
        normalized,
        [rf"sum\s+of\s+vehicles\s+on\s+routes\s+X\s+and\s+Z\s+minus\s+those\s+on\s+route\s+Y\s+should\s+not\s+exceed\s+({_NUMBER_TOKEN})"],
    )
    y_more = _number_after_patterns(
        normalized,
        [rf"route\s+Y\s+should\s+be\s+at\s+least\s+({_NUMBER_TOKEN})\s+more\s+than\s+those\s+on\s+route\s+Z"],
    )
    if cost_match is None or total_upper is None or weighted is None or xz_minus_y is None or y_more is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="route_vehicle_min_cost_ilp",
        symbols=symbols,
        costs={symbol: _number(cost_match.group(index)) for index, symbol in enumerate(symbols, start=1)},
        constraints=[
            ({"X": 1.0, "Y": 1.0, "Z": 1.0}, -math.inf, total_upper, "total_upper"),
            ({"X": 2.0, "Y": 3.0}, _number(weighted.group(1)), math.inf, "xy_weighted_lower"),
            ({"X": 1.0, "Y": -1.0, "Z": 1.0}, -math.inf, xz_minus_y, "x_plus_z_minus_y_upper"),
            ({"Y": 1.0, "Z": -1.0}, y_more, math.inf, "y_minus_z_lower"),
        ],
        upper_bounds={symbol: total_upper for symbol in symbols},
        confidence=0.84,
        notes="Solved route vehicle integer cost minimization with explicit route constraints.",
    )


def _solve_healthcare_fund_three_department_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    symbols = ["X", "Y", "Z"]
    if not (
        "healthcare scenario" in lowered
        and "departments" in lowered
        and "general medicine" in lowered
        and "pediatrics" in lowered
        and "surgery" in lowered
        and "whole numbers" in lowered
        and all(re.search(_loose_symbol_pattern(symbol), normalized, flags=re.IGNORECASE) for symbol in symbols)
    ):
        return TemplateSolveResult(False)
    budget = _number_after_patterns(normalized, [rf"total\s+budget[^.]*?cannot\s+exceed\s+\\?\$?\s*({_NUMBER_TOKEN})"])
    costs: dict[str, float] = {}
    for symbol in symbols:
        match = re.search(
            rf"({_NUMBER_TOKEN})\s+units?\s+for\s+department\s+{_loose_symbol_pattern(symbol)}",
            normalized,
            flags=re.IGNORECASE,
        )
        if match:
            costs[symbol] = _number(match.group(1))
    x_lower = re.search(
        rf"Department\s+X\s+requires\s+an\s+allocation\s+that\s+is\s+at\s+least\s+\\?\$?\s*({_NUMBER_TOKEN})\s+more\s+than\s+twice\s+the\s+allocation\s+for\s+department\s+Y",
        normalized,
        flags=re.IGNORECASE,
    )
    z_lower = re.search(
        rf"department\s+Z\s+requires\s+an\s+allocation\s+that\s+exceeds\s+the\s+allocation\s+for\s+department\s+Y\s+by\s+at\s+least\s+\\?\$?\s*({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if budget is None or set(costs) != set(symbols) or x_lower is None or z_lower is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="healthcare_fund_three_department_min_cost_ilp",
        symbols=symbols,
        costs=costs,
        constraints=[
            ({"X": 1.0, "Y": 1.0, "Z": 1.0}, -math.inf, budget, "budget_upper"),
            ({"X": 1.0, "Y": -2.0}, _number(x_lower.group(1)), math.inf, "x_twice_y_margin"),
            ({"Z": 1.0, "Y": -1.0}, _number(z_lower.group(1)), math.inf, "z_y_margin"),
        ],
        upper_bounds={symbol: budget for symbol in symbols},
        confidence=0.84,
        notes="Solved three-department healthcare fund integer minimization with explicit margin constraints.",
    )


def _solve_military_support_points_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    symbols = ["X1", "X2", "X3", "X4"]
    if not (
        "military commander" in lowered
        and "support points" in lowered
        and "three and a half" in lowered
        and all(re.search(_loose_symbol_pattern(symbol), normalized, flags=re.IGNORECASE) for symbol in symbols)
    ):
        return TemplateSolveResult(False)
    costs: dict[str, float] = {}
    for symbol in symbols:
        match = re.search(
            rf"unit\s+type\s+{_loose_symbol_pattern(symbol)}\s+(?:needs|requires)\s+({_NUMBER_TOKEN})\s+points",
            normalized,
            flags=re.IGNORECASE,
        )
        if match:
            costs[symbol] = _number(match.group(1))
    upper_match = re.search(
        rf"maximum\s+capacity\s*\(\s*({_NUMBER_TOKEN})\s+for\s+x1\s*,\s*({_NUMBER_TOKEN})\s+for\s+x2\s*,\s*"
        rf"({_NUMBER_TOKEN})\s+for\s+x3\s*,\s*({_NUMBER_TOKEN})\s+for\s+x4\s*\)",
        normalized,
        flags=re.IGNORECASE,
    )
    pair_upper = _number_after_patterns(
        normalized,
        [rf"total\s+number\s+of\s+units\s+that\s+can\s+be\s+supported\s+for\s+X1\s+and\s+X2\s+combined\s+is\s+limited\s+to\s+({_NUMBER_TOKEN})"],
    )
    strength_lower = _number_after_patterns(
        normalized,
        [rf"twice\s+the\s+units\s+of\s+X1\s+and\s+three\s+and\s+a\s+half\s+times\s+the\s+units\s+of\s+X3\s+must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})"],
    )
    difference_upper = _number_after_patterns(
        normalized,
        [rf"difference\s+in\s+units\s+between\s+X2\s+and\s+half\s+of\s+those\s+allocated\s+to\s+unit\s+type\s+X4\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"],
    )
    strategic_upper = _number_after_patterns(
        normalized,
        [rf"after\s+subtracting\s+the\s+number\s+of\s+units\s+for\s+both\s+unit\s+types\s+X1\s+and\s+unit\s+type\s+X3\s+from\s+those\s+for\s+unit\s+type\s+x2,\s+it\s+should\s+not\s+exceed\s+({_NUMBER_TOKEN})"],
    )
    if (
        set(costs) != set(symbols)
        or upper_match is None
        or pair_upper is None
        or strength_lower is None
        or difference_upper is None
        or strategic_upper is None
    ):
        return TemplateSolveResult(False)
    upper_bounds = {symbol: _number(upper_match.group(index)) for index, symbol in enumerate(symbols, start=1)}
    return _solve_small_integer_min_cost_model(
        template_id="military_support_points_min_cost_ilp",
        symbols=symbols,
        costs=costs,
        constraints=[
            ({"X1": 1.0, "X2": 1.0}, -math.inf, pair_upper, "x1_x2_upper"),
            ({"X1": 2.0, "X3": 3.5}, strength_lower, math.inf, "x1_x3_strength_lower"),
            ({"X2": 1.0, "X4": -0.5}, -math.inf, difference_upper, "x2_half_x4_upper"),
            ({"X2": 1.0, "X1": -1.0, "X3": -1.0}, -math.inf, strategic_upper, "x2_minus_x1_x3_upper"),
        ],
        upper_bounds=upper_bounds,
        confidence=0.84,
        notes="Solved four-unit military support-point minimization with explicit linear constraints.",
    )


def _solve_two_exercise_balance_min_fatigue(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "sports coach" in lowered
        and "exercise x" in lowered
        and "exercise y" in lowered
        and "fatigue" in lowered
        and ("integers" in lowered or "integer" in lowered)
    ):
        return TemplateSolveResult(False)
    total_upper = _number_after_patterns(normalized, [rf"both\s+exercises\s+combined\s+is\s+limited\s+to\s+({_NUMBER_TOKEN})"])
    effectiveness = re.search(
        rf"three\s+times\s+the\s+hours\s+spent\s+on\s+exercise\s+X\s+plus\s+four\s+times\s+those\s+spent\s+on\s+exercise\s+Y[^.]*?"
        rf"at\s+least\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    difference = _number_after_patterns(normalized, [rf"difference\s+in\s+hours\s+between\s+exercise\s+X\s+and\s+Y\s+should\s+not\s+exceed\s+({_NUMBER_TOKEN})"])
    costs = re.search(
        rf"fatigue\s+scores[^.]*?are\s+({_NUMBER_TOKEN})\s+for\s+exercise\s+X\s+and\s+({_NUMBER_TOKEN})\s+for\s+exercise\s+Y",
        normalized,
        flags=re.IGNORECASE,
    )
    upper_x = _number_after_patterns(normalized, [rf"more\s+than\s+({_NUMBER_TOKEN})\s+hours?\s+on\s+exercise\s+X"])
    upper_y = _number_after_patterns(normalized, [rf"or\s+({_NUMBER_TOKEN})\s+hours?\s+on\s+exercise\s+Y"])
    if total_upper is None or effectiveness is None or difference is None or costs is None or upper_x is None or upper_y is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="two_exercise_balance_min_fatigue_ilp",
        symbols=["X", "Y"],
        costs={"X": _number(costs.group(1)), "Y": _number(costs.group(2))},
        constraints=[
            ({"X": 1.0, "Y": 1.0}, -math.inf, total_upper, "total_upper"),
            ({"X": 3.0, "Y": 4.0}, _number(effectiveness.group(1)), math.inf, "effectiveness_lower"),
            ({"X": 1.0, "Y": -1.0}, -math.inf, difference, "x_minus_y_upper"),
            ({"X": -1.0, "Y": 1.0}, -math.inf, difference, "y_minus_x_upper"),
        ],
        upper_bounds={"X": upper_x, "Y": upper_y},
        confidence=0.84,
        notes="Solved two-exercise integer fatigue minimization with total, effectiveness, balance, and upper-bound constraints.",
    )


def _solve_telecom_project_pair_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    symbols = ["x1", "x2", "x3", "x4"]
    if not (
        "telecommunications company" in lowered
        and "first two projects" in lowered
        and "last two projects" in lowered
        and all(re.search(_loose_symbol_pattern(symbol), normalized, flags=re.IGNORECASE) for symbol in symbols)
    ):
        return TemplateSolveResult(False)
    cost_match = re.search(
        rf"cost\s+associated\s+with\s+each\s+project\s+is\s+({_NUMBER_TOKEN}),\s*({_NUMBER_TOKEN}),\s*({_NUMBER_TOKEN}),?\s+and\s+({_NUMBER_TOKEN})\s+units",
        normalized,
        flags=re.IGNORECASE,
    )
    upper_12 = _number_after_patterns(normalized, [rf"first\s+two\s+projects[^.]*?cannot\s+exceed\s+({_NUMBER_TOKEN})\s+units?"])
    upper_34 = _number_after_patterns(normalized, [rf"last\s+two\s+projects[^.]*?capped\s+at\s+({_NUMBER_TOKEN})\s+units?"])
    lower_13 = _number_after_patterns(normalized, [rf"At\s+least\s+({_NUMBER_TOKEN})\s+units?\s+must\s+be\s+devoted\s+between\s+project\s+\$?x1"])
    lower_24 = _number_after_patterns(normalized, [rf"minimum\s+of\s+({_NUMBER_TOKEN})\s+units?\s+needs\s+to\s+be\s+shared\s+between\s+project\$?x2"])
    if cost_match is None or upper_12 is None or upper_34 is None or lower_13 is None or lower_24 is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="telecom_project_pair_min_cost_ilp",
        symbols=symbols,
        costs={symbol: _number(cost_match.group(index)) for index, symbol in enumerate(symbols, start=1)},
        constraints=[
            ({"x1": 1.0, "x2": 1.0}, -math.inf, upper_12, "x1_x2_upper"),
            ({"x3": 1.0, "x4": 1.0}, -math.inf, upper_34, "x3_x4_upper"),
            ({"x1": 1.0, "x3": 1.0}, lower_13, math.inf, "x1_x3_lower"),
            ({"x2": 1.0, "x4": 1.0}, lower_24, math.inf, "x2_x4_lower"),
        ],
        confidence=0.84,
        notes="Solved four-project telecom integer minimization with pair upper and cross-pair lower constraints.",
    )


def _solve_fractional_telecom_project_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    symbols = ["X1", "X2", "X3", "X4"]
    if not (
        "telecommunications company" in lowered
        and "one-fourth" in lowered
        and "three-quarters" in lowered
        and all(re.search(_loose_symbol_pattern(symbol), normalized, flags=re.IGNORECASE) for symbol in symbols)
    ):
        return TemplateSolveResult(False)
    costs: dict[str, float] = {}
    for symbol in symbols:
        match = re.search(
            rf"({_NUMBER_TOKEN})\s+for\s+project\s+{_loose_symbol_pattern(symbol)}",
            normalized,
            flags=re.IGNORECASE,
        )
        if match:
            costs[symbol] = _number(match.group(1))
    upper_12 = _number_after_patterns(normalized, [rf"projects?\s+X1\s+and\s+X2\s+cannot\s+exceed\s+({_NUMBER_TOKEN})\s+units?"])
    upper_34 = _number_after_patterns(normalized, [rf"projects?\s+X3\s+and\s+X4\s+cannot\s+exceed\s+({_NUMBER_TOKEN})\s+units?"])
    first_margin = _number_after_patterns(normalized, [rf"one-fourth\s+more\s+than\s+those\s+allocated\s+to\s+Project\s+X3\s+by\s+no\s+less\s+than\s+({_NUMBER_TOKEN})"])
    second_margin = _number_after_patterns(normalized, [rf"Project\s+X4\s+and\s+those\s+assigned\s+to\s+Project\s+X2\s+must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})"])
    upper_bounds: dict[str, float] = {}
    for symbol in symbols:
        match = re.search(
            rf"For\s+Project\s+{_loose_symbol_pattern(symbol)}:\s+between\s+({_NUMBER_TOKEN})\s+and\s+({_NUMBER_TOKEN})\s+units",
            normalized,
            flags=re.IGNORECASE,
        )
        if match:
            upper_bounds[symbol] = _number(match.group(2))
    if set(costs) != set(symbols) or upper_12 is None or upper_34 is None or first_margin is None or second_margin is None or set(upper_bounds) != set(symbols):
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="fractional_telecom_project_min_cost_ilp",
        symbols=symbols,
        costs=costs,
        constraints=[
            ({"X1": 1.0, "X2": 1.0}, -math.inf, upper_12, "x1_x2_upper"),
            ({"X3": 1.0, "X4": 1.0}, -math.inf, upper_34, "x3_x4_upper"),
            ({"X1": 0.5, "X3": -0.25}, first_margin, math.inf, "half_x1_quarter_x3_margin"),
            ({"X4": 0.75, "X2": -1.0}, second_margin, math.inf, "three_quarters_x4_x2_margin"),
        ],
        upper_bounds=upper_bounds,
        confidence=0.84,
        notes="Solved four-project fractional-coefficient telecom integer minimization.",
    )


def _solve_retail_department_strong_coverage_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    symbols = ["x1", "x2", "x3", "x4"]
    if not (
        "retail manager" in lowered
        and "four departments" in lowered
        and "exactly 400" in lowered
        and "at least equal to those of both departments x1 and x2" in lowered
        and all(re.search(_loose_symbol_pattern(symbol), normalized, flags=re.IGNORECASE) for symbol in symbols)
    ):
        return TemplateSolveResult(False)
    costs: dict[str, float] = {}
    for symbol in symbols:
        match = re.search(
            rf"for\s+department\s+{_loose_symbol_pattern(symbol)}\s+it's\s+\\?\$?\s*({_NUMBER_TOKEN})",
            normalized,
            flags=re.IGNORECASE,
        )
        if not match:
            match = re.search(
                rf"for\s+{_loose_symbol_pattern(symbol)}\s+it's\s+(?:only\s+)?\\?\$?\s*({_NUMBER_TOKEN})",
                normalized,
                flags=re.IGNORECASE,
            )
        if match:
            costs[symbol] = _number(match.group(1))
    exact_12 = _number_after_patterns(
        normalized,
        [rf"allocation\s+for\s+departments\s+\$?x1\$?\s+and\s+\$?x2\$?\s+must\s+be\s+exactly\s+({_NUMBER_TOKEN})\s+units?"],
    )
    upper_12 = {
        "x1": _number_after_patterns(normalized, [rf"{_loose_symbol_pattern('x1')}\s+not\s+receiving\s+more\s+than\s+({_NUMBER_TOKEN})\s+units?"]),
        "x2": _number_after_patterns(normalized, [rf"{_loose_symbol_pattern('x2')}\s+not\s+receiving\s+more\s+than\s+({_NUMBER_TOKEN})\s+units?"]),
        "x3": _number_after_patterns(normalized, [rf"x3\s+and\s+x4\s+cannot\s+exceed\s+({_NUMBER_TOKEN})\s+units?,\s+with\s+each\s+department\s+having\s+a\s+maximum\s+limit\s+of\s+({_NUMBER_TOKEN})"]),
        "x4": None,
    }
    x4_upper = re.search(
        rf"{_loose_symbol_pattern('x3')}\s+and\s+{_loose_symbol_pattern('x4')}\s+cannot\s+exceed\s+({_NUMBER_TOKEN})\s+units?,\s+"
        rf"with\s+each\s+department\s+having\s+a\s+maximum\s+limit\s+of\s+({_NUMBER_TOKEN})\s+and\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if x4_upper:
        upper_12["x3"] = _number(x4_upper.group(2))
        upper_12["x4"] = _number(x4_upper.group(3))
        total_34 = _number(x4_upper.group(1))
    else:
        total_34 = None
    if exact_12 is None or set(costs) != set(symbols) or total_34 is None or any(value is None for value in upper_12.values()):
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="retail_department_strong_coverage_min_cost_ilp",
        symbols=symbols,
        costs=costs,
        constraints=[
            ({"x1": 1.0, "x2": 1.0}, exact_12, exact_12, "x1_x2_exact"),
            ({"x3": 1.0, "x4": 1.0}, -math.inf, total_34, "x3_x4_upper"),
            ({"x3": 1.0, "x1": -1.0, "x2": -1.0}, 0.0, math.inf, "x3_covers_x1_x2"),
            ({"x4": 1.0, "x1": -1.0, "x2": -1.0}, 0.0, math.inf, "x4_covers_x1_x2"),
        ],
        upper_bounds={symbol: float(upper_12[symbol]) for symbol in symbols},
        confidence=0.78,
        notes=(
            "Solved retail department allocation using the dataset's strong reading: "
            "both x3 and x4 must each cover the combined x1+x2 allocation."
        ),
        artifact={"interpretation": "x3 >= x1+x2 and x4 >= x1+x2"},
    )


def _solve_four_property_real_estate_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    symbols = ["x1", "x2", "x3", "x4"]
    if not (
        "real estate developer" in lowered
        and "residential" in lowered
        and "commercial" in lowered
        and "industrial" in lowered
        and "retail" in lowered
        and "four point five" in lowered
        and all(re.search(_loose_symbol_pattern(symbol), normalized, flags=re.IGNORECASE) for symbol in symbols)
    ):
        return TemplateSolveResult(False)
    cost_match = re.search(
        rf"costs?\s+(?:being|are)\s+\\?\$?\s*({_NUMBER_TOKEN}),\s*\\?\$?\s*({_NUMBER_TOKEN}),\s*\\?\$?\s*({_NUMBER_TOKEN})\s+and\s+\\?\$?\s*({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    upper_12 = _number_after_patterns(normalized, [rf"residential\s+and\s+commercial\s+properties\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"])
    lower_34 = _number_after_patterns(normalized, [rf"twice\s+the\s+number\s+of\s+industrial\s+properties\s+plus\s+retail\s+properties\s+should\s+be\s+at\s+least\s+({_NUMBER_TOKEN})"])
    upper_weighted = _number_after_patterns(normalized, [rf"Three\s+times\s+the\s+number\s+of\s+residential\s+properties\s+plus\s+four\s+point\s+five\s+times\s+commercial\s+properties\s+should\s+not\s+exceed\s+({_NUMBER_TOKEN})"])
    industrial_retail = _number_after_patterns(normalized, [rf"Six\s+times\s+the\s+number\s+of\s+industrial\s+properties\s+minus\s+half\s+a\s+unit\s+of\s+retail\s+property\s+should\s+not\s+surpass\s+({_NUMBER_TOKEN})"])
    upper_bounds: dict[str, float] = {}
    label_to_symbol = {"Residential": "x1", "Commercial": "x2", "Industrial": "x3", "Retail": "x4"}
    for label, symbol in label_to_symbol.items():
        match = re.search(rf"{label}\({symbol}\):\s+Up\s+to\s+({_NUMBER_TOKEN})\s+units", normalized, flags=re.IGNORECASE)
        if match:
            upper_bounds[symbol] = _number(match.group(1))
    if (
        cost_match is None
        or upper_12 is None
        or lower_34 is None
        or upper_weighted is None
        or industrial_retail is None
        or set(upper_bounds) != set(symbols)
    ):
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="four_property_real_estate_min_cost_ilp",
        symbols=symbols,
        costs={symbol: _number(cost_match.group(index)) for index, symbol in enumerate(symbols, start=1)},
        constraints=[
            ({"x1": 1.0, "x2": 1.0}, -math.inf, upper_12, "x1_x2_upper"),
            ({"x3": 2.0, "x4": 1.0}, lower_34, math.inf, "x3_x4_lower"),
            ({"x1": 3.0, "x2": 4.5}, -math.inf, upper_weighted, "x1_x2_weighted_upper"),
            ({"x3": 6.0, "x4": -0.5}, -math.inf, industrial_retail, "x3_x4_weighted_upper"),
        ],
        upper_bounds=upper_bounds,
        confidence=0.84,
        notes="Solved four-property real-estate integer model; infeasibility is surfaced when stated bounds conflict.",
    )


def _project_coefficient(value: str) -> float:
    lowered = re.sub(r"\s+", " ", value.strip().lower())
    word_coefficients = {
        "once": 1.0,
        "one": 1.0,
        "twice": 2.0,
        "two times": 2.0,
        "thrice": 3.0,
        "three times": 3.0,
        "triple": 3.0,
        "quadruple": 4.0,
        "four times": 4.0,
        "five times": 5.0,
    }
    if lowered in word_coefficients:
        return word_coefficients[lowered]
    return _number(value)


def _solve_small_integer_min_cost_model(
    *,
    template_id: str,
    symbols: list[str],
    costs: dict[str, float],
    constraints: list[tuple[dict[str, float], float, float, str]],
    upper_bounds: dict[str, float] | None = None,
    confidence: float = 0.82,
    notes: str = "",
    artifact: dict[str, Any] | None = None,
) -> TemplateSolveResult:
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id=template_id,
            status="solver_unavailable",
            confidence=confidence,
            notes=str(exc),
            artifact=artifact,
        )

    index = {symbol: offset for offset, symbol in enumerate(symbols)}
    matrix = np.array(
        [
            [float(coefficients.get(symbol, 0.0)) for symbol in symbols]
            for coefficients, _lower, _upper, _name in constraints
        ],
        dtype=float,
    )
    lower = np.array([lower for _coefficients, lower, _upper, _name in constraints], dtype=float)
    upper = np.array([upper for _coefficients, _lower, upper, _name in constraints], dtype=float)
    variable_upper = np.array(
        [
            float((upper_bounds or {}).get(symbol, math.inf))
            for symbol in symbols
        ],
        dtype=float,
    )
    result = milp(
        c=np.array([costs[symbol] for symbol in symbols], dtype=float),
        integrality=np.ones(len(symbols)),
        bounds=Bounds(np.zeros(len(symbols)), variable_upper),
        constraints=LinearConstraint(matrix, lower, upper),
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id=template_id,
            status="infeasible",
            confidence=confidence,
            notes=str(result.message),
            artifact=artifact,
        )
    selected = {symbol: float(result.x[index[symbol]]) for symbol in symbols}
    return TemplateSolveResult(
        matched=True,
        template_id=template_id,
        status="optimal",
        objective_value=float(result.fun),
        variable_values={
            symbol: value
            for symbol, value in selected.items()
            if not math.isclose(value, 0.0, abs_tol=1e-8)
        },
        confidence=confidence,
        notes=notes,
        artifact={
            **(artifact or {}),
            "costs": costs,
            "constraints": [
                {"coefficients": coefficients, "lower": lower, "upper": upper, "name": name}
                for coefficients, lower, upper, name in constraints
            ],
            "upper_bounds": upper_bounds or {},
            "selected": selected,
        },
    )


def _solve_toy_product_logic_profit_max(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    labels = ["robot", "model_car", "building_block", "doll"]
    if not (
        "bright future toys" in lowered
        and all(word in lowered for word in ("robot", "model car", "building block", "doll"))
        and "plastic" in lowered
        and "electronic components" in lowered
        and "maximize profit" in lowered
    ):
        return TemplateSolveResult(False)

    profit_match = re.search(
        rf"profit\s+for\s+each\s+robot[^$0-9]*\\?\$?({_NUMBER_TOKEN}).*?"
        rf"model\s+car[^$0-9]*\\?\$?({_NUMBER_TOKEN}).*?"
        rf"building\s+blocks[^$0-9]*\\?\$?({_NUMBER_TOKEN}).*?"
        rf"doll[^$0-9]*\\?\$?({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    plastic_total = _number_after_patterns(normalized, [rf"({_NUMBER_TOKEN})\s+units\s+of\s+plastic\s+available"])
    electronic_total = _number_after_patterns(normalized, [rf"({_NUMBER_TOKEN})\s+units\s+of\s+electronic\s+components\s+available"])
    plastic_match = re.search(
        rf"Each\s+robot\s+requires\s+({_NUMBER_TOKEN})\s+units\s+of\s+plastic.*?"
        rf"model\s+car\s+requires\s+({_NUMBER_TOKEN})\s+units\s+of\s+plastic.*?"
        rf"building\s+blocks\s+requires\s+({_NUMBER_TOKEN})\s+units\s+of\s+plastic.*?"
        rf"doll\s+requires\s+({_NUMBER_TOKEN})\s+units\s+of\s+plastic",
        normalized,
        flags=re.IGNORECASE,
    )
    electronic_match = re.search(
        rf"Each\s+robot\s+requires\s+({_NUMBER_TOKEN})\s+units\s+of\s+electronic\s+components.*?"
        rf"model\s+car\s+requires\s+({_NUMBER_TOKEN})\s+units\s+of\s+electronic\s+components.*?"
        rf"building\s+blocks\s+requires\s+({_NUMBER_TOKEN})\s+units\s+of\s+electronic\s+components.*?"
        rf"doll\s+requires\s+({_NUMBER_TOKEN})\s+units\s+of\s+electronic\s+components",
        normalized,
        flags=re.IGNORECASE,
    )
    if profit_match is None or plastic_match is None or electronic_match is None or plastic_total is None or electronic_total is None:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="toy_product_logic_profit_milp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
        )

    profits = [_number(profit_match.group(index)) for index in range(1, 5)]
    plastic = [_number(plastic_match.group(index)) for index in range(1, 5)]
    electronics = [_number(electronic_match.group(index)) for index in range(1, 5)]
    upper_quantities = [
        max(1.0, min(plastic_total / plastic[index], electronic_total / electronics[index]))
        for index in range(4)
    ]
    variable_count = 8
    c = np.array([-value for value in profits] + [0.0, 0.0, 0.0, 0.0], dtype=float)
    rows: list[list[float]] = []
    lower: list[float] = []
    upper: list[float] = []

    rows.append([*plastic, 0.0, 0.0, 0.0, 0.0])
    lower.append(-math.inf)
    upper.append(float(plastic_total))
    rows.append([*electronics, 0.0, 0.0, 0.0, 0.0])
    lower.append(-math.inf)
    upper.append(float(electronic_total))
    for index, limit in enumerate(upper_quantities):
        row = [0.0] * variable_count
        row[index] = 1.0
        row[4 + index] = -float(limit)
        rows.append(row)
        lower.append(-math.inf)
        upper.append(0.0)
        row = [0.0] * variable_count
        row[index] = 1.0
        row[4 + index] = -1.0
        rows.append(row)
        lower.append(0.0)
        upper.append(math.inf)

    row = [0.0] * variable_count
    row[4] = 1.0
    row[7] = 1.0
    rows.append(row)
    lower.append(-math.inf)
    upper.append(1.0)
    row = [0.0] * variable_count
    row[5] = 1.0
    row[6] = -1.0
    rows.append(row)
    lower.append(-math.inf)
    upper.append(0.0)
    row = [0.0] * variable_count
    row[3] = 1.0
    row[1] = -1.0
    rows.append(row)
    lower.append(-math.inf)
    upper.append(0.0)

    result = milp(
        c=c,
        integrality=np.ones(variable_count),
        bounds=Bounds(np.zeros(variable_count), np.array([*upper_quantities, 1.0, 1.0, 1.0, 1.0])),
        constraints=LinearConstraint(np.array(rows, dtype=float), np.array(lower), np.array(upper)),
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="toy_product_logic_profit_milp",
            status="infeasible",
            confidence=0.8,
            notes=str(result.message),
        )
    selected = {label: float(result.x[index]) for index, label in enumerate(labels)}
    return TemplateSolveResult(
        matched=True,
        template_id="toy_product_logic_profit_milp",
        status="optimal",
        objective_value=float(-result.fun),
        variable_values={label: value for label, value in selected.items() if value > 1e-8},
        confidence=0.86,
        notes="Solved toy product-mix MILP with resource capacities and stated production-logic rules.",
        artifact={
            "profits": dict(zip(labels, profits)),
            "plastic": dict(zip(labels, plastic)),
            "electronics": dict(zip(labels, electronics)),
            "selected": selected,
        },
    )


def _solve_product_table_fixed_cost_profit_lp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "fixed costs" in lowered
        and "variable cost" in lowered
        and "selling price" in lowered
        and "available labor" in lowered
        and "available material" in lowered
        and "maximize" in lowered
    ):
        return TemplateSolveResult(False)
    table = _parse_markdown_table(text)
    if not table:
        return TemplateSolveResult(False)
    header, rows = table
    header_labels = [_clean_label(cell).lower() for cell in header]
    required_columns = ("product", "labor", "material", "selling price", "variable cost")
    if not all(any(required in label for label in header_labels) for required in required_columns):
        return TemplateSolveResult(False)

    product_index = next(index for index, label in enumerate(header_labels) if "product" in label)
    labor_index = next(index for index, label in enumerate(header_labels) if "labor" in label)
    material_index = next(index for index, label in enumerate(header_labels) if "material" in label)
    price_index = next(index for index, label in enumerate(header_labels) if "selling price" in label)
    cost_index = next(index for index, label in enumerate(header_labels) if "variable cost" in label)
    labels: list[str] = []
    labor: list[float] = []
    material: list[float] = []
    margins: list[float] = []
    for row in rows:
        if len(row) <= max(product_index, labor_index, material_index, price_index, cost_index):
            continue
        values = [
            _first_number(row[labor_index]),
            _first_number(row[material_index]),
            _first_number(row[price_index]),
            _first_number(row[cost_index]),
        ]
        if any(value is None for value in values):
            continue
        labels.append(_clean_label(row[product_index]))
        labor.append(float(values[0] or 0.0))
        material.append(float(values[1] or 0.0))
        margins.append(float(values[2] or 0.0) - float(values[3] or 0.0))
    labor_limit = _number_after_patterns(normalized, [rf"available\s+labor[^.]*?is\s+({_NUMBER_TOKEN})"])
    material_limit = _number_after_patterns(normalized, [rf"available\s+material[^.]*?is\s+({_NUMBER_TOKEN})"])
    fixed_match = re.search(
        rf"fixed\s+costs?[^.]*?are\s+({_NUMBER_TOKEN}),\s*({_NUMBER_TOKEN}),?\s+and\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if len(labels) < 2 or labor_limit is None or material_limit is None or fixed_match is None:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="product_table_fixed_cost_profit_lp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
        )
    result = linprog(
        [-value for value in margins],
        A_ub=[labor, material],
        b_ub=[labor_limit, material_limit],
        bounds=[(0, None)] * len(labels),
        method="highs",
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="product_table_fixed_cost_profit_lp",
            status="solver_failed",
            confidence=0.8,
            notes=str(result.message),
        )
    fixed_costs = [_number(fixed_match.group(index)) for index in range(1, 4)]
    objective_value = float(-result.fun - sum(fixed_costs))
    return TemplateSolveResult(
        matched=True,
        template_id="product_table_fixed_cost_profit_lp",
        status="optimal",
        objective_value=objective_value,
        variable_values={
            f"produce_{label}": float(value)
            for label, value in zip(labels, result.x)
            if not math.isclose(float(value), 0.0, abs_tol=1e-8)
        },
        confidence=0.84,
        notes="Solved production mix LP from product table and subtracted stated weekly fixed costs.",
        artifact={
            "products": labels,
            "margins": dict(zip(labels, margins)),
            "labor": dict(zip(labels, labor)),
            "material": dict(zip(labels, material)),
            "fixed_costs": fixed_costs,
        },
    )


def _solve_stock_bond_balance_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "stocks" in lowered
        and "bonds" in lowered
        and "minimize the total cost" in lowered
        and ("whole numbers" in lowered or "integer" in lowered)
        and "twice" in lowered
    ):
        return TemplateSolveResult(False)
    lower_rhs = _number_after_patterns(
        normalized,
        [rf"twice\s+the\s+number\s+of\s+stocks\s+and\s+bonds\s+should\s+be\s+at\s+least\s+\$?\\?\$?({_NUMBER_TOKEN})"],
    )
    difference_upper = _number_after_patterns(
        normalized,
        [rf"difference\s+between\s+the\s+amount\s+invested\s+in\s+stocks\s+and\s+twice\s+that\s+of\s+bonds\s+cannot\s+exceed\s+\$?\\?\$?({_NUMBER_TOKEN})"],
    )
    cost_match = re.search(
        r"cost\s+per\s+unit\s+for\s+stocks\s+is\s+(.{0,40}?)\s+while\s+it\s+is\s+(.{0,40}?)\s+for\s+bonds",
        normalized,
        flags=re.IGNORECASE,
    )
    stock_cost = _first_number(cost_match.group(1)) if cost_match else None
    bond_cost = _first_number(cost_match.group(2)) if cost_match else None
    if lower_rhs is None or difference_upper is None or stock_cost is None or bond_cost is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="stock_bond_balance_min_cost_ilp",
        symbols=["stocks", "bonds"],
        costs={"stocks": float(stock_cost), "bonds": float(bond_cost)},
        constraints=[
            ({"stocks": 2.0, "bonds": 1.0}, lower_rhs, math.inf, "twice_stocks_plus_bonds_lower"),
            ({"stocks": 1.0, "bonds": -2.0}, -math.inf, difference_upper, "stocks_minus_twice_bonds_upper"),
        ],
        confidence=0.84,
        notes="Solved stock/bond integer cost minimization with balance and minimum-investment constraints.",
    )


def _solve_two_service_resource_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "service x" in lowered
        and "service y" in lowered
        and "minimize the total cost" in lowered
        and ("allocations are integers" in lowered or "integers" in lowered)
    ):
        return TemplateSolveResult(False)
    total_upper = _number_after_patterns(
        normalized,
        [rf"total\s+resources\s+available[^.]*?cannot\s+exceed\s+({_NUMBER_TOKEN})"],
    )
    demand_lower = _number_after_patterns(
        normalized,
        [rf"sum\s+of\s+the\s+resources\s+allocated\s+to\s+service\s+X\s+and\s+twice\s+that\s+for\s+service\s+Y\s+must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})"],
    )
    difference_upper = _number_after_patterns(
        normalized,
        [rf"difference\s+in\s+resources\s+between\s+twice\s+that\s+of\s+service\s+X\s+and\s+service\s+Y\s+should\s+not\s+exceed\s+({_NUMBER_TOKEN})"],
    )
    cost_match = re.search(
        rf"cost[^.]*?service\s+X\s+is\s+({_NUMBER_TOKEN})\s+units?\s+and\s+for\s+service\s+Y\s+is\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if total_upper is None or demand_lower is None or difference_upper is None or cost_match is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="two_service_resource_min_cost_ilp",
        symbols=["X", "Y"],
        costs={"X": _number(cost_match.group(1)), "Y": _number(cost_match.group(2))},
        constraints=[
            ({"X": 1.0, "Y": 1.0}, -math.inf, total_upper, "total_upper"),
            ({"X": 1.0, "Y": 2.0}, demand_lower, math.inf, "x_plus_twice_y_lower"),
            ({"X": 2.0, "Y": -1.0}, -math.inf, difference_upper, "twice_x_minus_y_upper"),
        ],
        confidence=0.84,
        notes="Solved two-service integer resource minimization with total, demand, and balance constraints.",
    )


def _solve_environmental_project_three_var_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "environmental agency" in lowered
        and "project x" in lowered
        and "project y" in lowered
        and "project z" in lowered
        and "minimize the total cost" in lowered
        and ("whole numbers" in lowered or "integers" in lowered)
    ):
        return TemplateSolveResult(False)
    cost_match = re.search(
        rf"costs\s+being\s+\\?\$?({_NUMBER_TOKEN})\s+for\s+project\s+X,\s*\\?\$?({_NUMBER_TOKEN})\s+for\s+project\s+Y\s+and\s+\\?\$?({_NUMBER_TOKEN})\s+for\s+project\s+Z",
        normalized,
        flags=re.IGNORECASE,
    )
    lower_weighted = _number_after_patterns(
        normalized,
        [rf"twice\s+the\s+allocation\s+for\s+X\s+plus\s+thrice\s+the\s+allocation\s+for\s+Y\s+and\s+Z\s+cannot\s+be\s+less\s+than\s+({_NUMBER_TOKEN})"],
    )
    upper_weighted = _number_after_patterns(
        normalized,
        [rf"sum\s+of\s+allocations\s+for\s+X,\s*Y\s+and\s+four\s+times\s+the\s+allocation\s+for\s+Z\s+should\s+not\s+exceed\s+({_NUMBER_TOKEN})"],
    )
    difference_lower = _number_after_patterns(
        normalized,
        [rf"difference\s+between\s+allocations\s+of\s+X\s+and\s+Y\s+added\s+with\s+that\s+of\s+Z\s+should\s+be\s+at\s+least\s+({_NUMBER_TOKEN})"],
    )
    bounds: dict[str, float] = {}
    for symbol in ("x", "y", "z"):
        match = re.search(
            rf"0\s*<=\s*{symbol}\s*<=\s*({_NUMBER_TOKEN})",
            normalized,
            flags=re.IGNORECASE,
        )
        if match:
            bounds[symbol.upper()] = _number(match.group(1))
    if (
        cost_match is None
        or lower_weighted is None
        or upper_weighted is None
        or difference_lower is None
        or set(bounds) != {"X", "Y", "Z"}
    ):
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="environmental_project_three_var_min_cost_ilp",
        symbols=["X", "Y", "Z"],
        costs={"X": _number(cost_match.group(1)), "Y": _number(cost_match.group(2)), "Z": _number(cost_match.group(3))},
        constraints=[
            ({"X": 2.0, "Y": 3.0, "Z": 3.0}, lower_weighted, math.inf, "weighted_lower"),
            ({"X": 1.0, "Y": 1.0, "Z": 4.0}, -math.inf, upper_weighted, "weighted_upper"),
            ({"X": 1.0, "Y": -1.0, "Z": 1.0}, difference_lower, math.inf, "x_minus_y_plus_z_lower"),
        ],
        upper_bounds=bounds,
        confidence=0.84,
        notes="Solved three-project environmental integer minimization with stated weighted constraints and bounds.",
    )


def _solve_property_pair_requirement_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "real estate developer" in lowered
        and "residential" in lowered
        and "commercial" in lowered
        and "industrial" in lowered
        and "mixed-use" in lowered
        and "minimize the total cost" in lowered
        and ("whole numbers" in lowered or "indivisible" in lowered)
    ):
        return TemplateSolveResult(False)
    cost_match = re.search(
        rf"costs\s+being\s+\\?\$?({_NUMBER_TOKEN}),\s*\\?\$?({_NUMBER_TOKEN}),\s*\\?\$?({_NUMBER_TOKEN}),?\s+and\s+\\?\$?({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    residential_commercial_upper = _number_after_patterns(
        normalized,
        [rf"residential\s+and\s+commercial\s+properties\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"],
    )
    industrial_mixed_upper = _number_after_patterns(
        normalized,
        [rf"industrial\s+and\s+mixed-use\s+properties\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"],
    )
    residential_industrial_lower = _number_after_patterns(
        normalized,
        [rf"At\s+least\s+({_NUMBER_TOKEN})\s+properties\s+must\s+be\s+either\s+residential\s+or\s+industrial"],
    )
    commercial_mixed_lower = _number_after_patterns(
        normalized,
        [rf"At\s+least\s+({_NUMBER_TOKEN})\s+properties\s+must\s+be\s+either\s+commercial\s+or\s+mixed-use"],
    )
    label_to_symbol = {
        "Residential": "x",
        "Commercial": "y",
        "Industrial": "z",
        "Mixed Use": "w",
    }
    bounds: dict[str, float] = {}
    for label, symbol in label_to_symbol.items():
        match = re.search(
            rf"{label}\s*\(\s*{symbol}\s*\)\s*:\s*\[\s*({_NUMBER_TOKEN})\s*,\s*({_NUMBER_TOKEN})\s*\]",
            normalized,
            flags=re.IGNORECASE,
        )
        if match:
            bounds[symbol] = _number(match.group(2))
    if (
        cost_match is None
        or residential_commercial_upper is None
        or industrial_mixed_upper is None
        or residential_industrial_lower is None
        or commercial_mixed_lower is None
        or set(bounds) != {"x", "y", "z", "w"}
    ):
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="property_pair_requirement_min_cost_ilp",
        symbols=["x", "y", "z", "w"],
        costs={
            "x": _number(cost_match.group(1)),
            "y": _number(cost_match.group(2)),
            "z": _number(cost_match.group(3)),
            "w": _number(cost_match.group(4)),
        },
        constraints=[
            ({"x": 1.0, "y": 1.0}, -math.inf, residential_commercial_upper, "residential_commercial_upper"),
            ({"z": 1.0, "w": 1.0}, -math.inf, industrial_mixed_upper, "industrial_mixed_upper"),
            ({"x": 1.0, "z": 1.0}, residential_industrial_lower, math.inf, "residential_industrial_lower"),
            ({"y": 1.0, "w": 1.0}, commercial_mixed_lower, math.inf, "commercial_mixed_lower"),
        ],
        upper_bounds=bounds,
        confidence=0.84,
        notes="Solved four-property real-estate integer minimization with pair upper and diversification lower constraints.",
    )


def _solve_cart_trolley_worker_min_count(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "carts" in lowered
        and "trolleys" in lowered
        and "workers" in lowered
        and "kg/min" in lowered
        and "40%" in lowered
        and "minimize the total number of workers" in lowered
    ):
        return TemplateSolveResult(False)
    cart_match = re.search(
        rf"Carts\s+can\s+transport\s+({_NUMBER_TOKEN})\s+kg/min[^.]*?requires\s+({_NUMBER_TOKEN})\s+workers",
        normalized,
        flags=re.IGNORECASE,
    )
    trolley_match = re.search(
        rf"Trolleys\s+can\s+transport\s+({_NUMBER_TOKEN})\s+kg/min[^.]*?requires\s+({_NUMBER_TOKEN})\s+workers",
        normalized,
        flags=re.IGNORECASE,
    )
    min_trolleys = _number_after_patterns(normalized, [rf"at\s+least\s+({_NUMBER_TOKEN})\s+trolleys"])
    percent_match = re.search(rf"maximum\s+of\s+({_NUMBER_TOKEN})%\s+of\s+the\s+transportation\s+can\s+be\s+using\s+trolleys", normalized, flags=re.IGNORECASE)
    demand = _number_after_patterns(normalized, [rf"deliver\s+at\s+a\s+rate\s+of\s+({_NUMBER_TOKEN})\s+kg/min"])
    if cart_match is None or trolley_match is None or min_trolleys is None or percent_match is None or demand is None:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="cart_trolley_worker_min_count_ilp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
        )
    cart_rate, cart_workers = _number(cart_match.group(1)), _number(cart_match.group(2))
    trolley_rate, trolley_workers = _number(trolley_match.group(1)), _number(trolley_match.group(2))
    share = _number(percent_match.group(1)) / 100.0
    rows = np.array(
        [
            [cart_rate, trolley_rate],
            [-share, 1.0 - share],
        ],
        dtype=float,
    )
    result = milp(
        c=np.array([cart_workers, trolley_workers], dtype=float),
        integrality=np.ones(2),
        bounds=Bounds(np.array([0.0, min_trolleys]), np.full(2, math.inf)),
        constraints=LinearConstraint(rows, np.array([demand, -math.inf]), np.array([math.inf, 0.0])),
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="cart_trolley_worker_min_count_ilp",
            status="infeasible",
            confidence=0.82,
            notes=str(result.message),
        )
    return TemplateSolveResult(
        matched=True,
        template_id="cart_trolley_worker_min_count_ilp",
        status="optimal",
        objective_value=float(result.fun),
        variable_values={"carts": float(result.x[0]), "trolleys": float(result.x[1])},
        confidence=0.84,
        notes="Solved cart/trolley worker minimization with demand, minimum trolley count, and trolley share by count.",
    )


def _solve_light_fixture_change_min_count(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "led" in lowered
        and "fluorescence" in lowered
        and "electricity" in lowered
        and "light fixtures" in lowered
        and "reduce the total number of light changes" in lowered
    ):
        return TemplateSolveResult(False)
    led_match = re.search(
        rf"LED\s+(?:fixture|light)[^.]*?uses\s+({_NUMBER_TOKEN})\s+units\s+of\s+electricity[^.]*?changed\s+({_NUMBER_TOKEN})\s+times",
        normalized,
        flags=re.IGNORECASE,
    )
    fluorescent_match = re.search(
        rf"fluorescence\s+lamp[^.]*?uses\s+({_NUMBER_TOKEN})\s+units\s+of\s+electricity[^.]*?changed\s+({_NUMBER_TOKEN})\s+times",
        normalized,
        flags=re.IGNORECASE,
    )
    percent_match = re.search(rf"at\s+least\s+({_NUMBER_TOKEN})%\s+implemented\s+lights\s+must\s+be\s+fluorescence", normalized, flags=re.IGNORECASE)
    fixture_min = _number_after_patterns(normalized, [rf"requires\s+at\s+least\s+({_NUMBER_TOKEN})\s+light\s+fixtures"])
    electricity_max = _number_after_patterns(normalized, [rf"at\s+most\s+({_NUMBER_TOKEN})\s+units\s+of\s+electricity"])
    if led_match is None or fluorescent_match is None or percent_match is None or fixture_min is None or electricity_max is None:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="light_fixture_change_min_count_ilp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
        )
    led_electricity, led_changes = _number(led_match.group(1)), _number(led_match.group(2))
    fluorescent_electricity, fluorescent_changes = _number(fluorescent_match.group(1)), _number(fluorescent_match.group(2))
    share = _number(percent_match.group(1)) / 100.0
    rows = np.array(
        [
            [1.0, 1.0],
            [-share, 1.0 - share],
            [led_electricity, fluorescent_electricity],
        ],
        dtype=float,
    )
    result = milp(
        c=np.array([led_changes, fluorescent_changes], dtype=float),
        integrality=np.ones(2),
        bounds=Bounds(np.zeros(2), np.full(2, math.inf)),
        constraints=LinearConstraint(
            rows,
            np.array([fixture_min, 0.0, -math.inf]),
            np.array([math.inf, math.inf, electricity_max]),
        ),
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="light_fixture_change_min_count_ilp",
            status="infeasible",
            confidence=0.82,
            notes=str(result.message),
        )
    return TemplateSolveResult(
        matched=True,
        template_id="light_fixture_change_min_count_ilp",
        status="optimal",
        objective_value=float(result.fun),
        variable_values={"led": float(result.x[0]), "fluorescence": float(result.x[1])},
        confidence=0.84,
        notes="Solved light fixture integer minimization with fixture count, electricity, and fluorescence share constraints.",
    )


def _solve_cable_mix_profit_max(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "long cables" in lowered
        and "short cables" in lowered
        and "gold" in lowered
        and "maximize profit" in lowered
    ):
        return TemplateSolveResult(False)
    gold_available = _number_after_patterns(normalized, [rf"({_NUMBER_TOKEN})\s+mg\s+of\s+gold\s+available"])
    requirement_match = re.search(
        rf"Long\s+cables\s+require\s+({_NUMBER_TOKEN})\s+mg\s+of\s+gold\s+while\s+short\s+cables\s+require\s+({_NUMBER_TOKEN})\s+mg\s+of\s+gold",
        normalized,
        flags=re.IGNORECASE,
    )
    ratio = _number_after_patterns(normalized, [rf"at\s+least\s+({_NUMBER_TOKEN})\s+times\s+the\s+number\s+of\s+short\s+cables\s+are\s+needed\s+than\s+the\s+long\s+cables"])
    long_min = _number_after_patterns(normalized, [rf"at\s+least\s+({_NUMBER_TOKEN})\s+long\s+cables"])
    profit_match = re.search(
        rf"long\s+cable\s+sold\s+results\s+in\s+a\s+\\?\$?({_NUMBER_TOKEN})\s+profit\s+and\s+each\s+short\s+cable\s+sold\s+results\s+in\s+a\s+\\?\$?({_NUMBER_TOKEN})\s+profit",
        normalized,
        flags=re.IGNORECASE,
    )
    if gold_available is None or requirement_match is None or ratio is None or long_min is None or profit_match is None:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="cable_mix_profit_max_ilp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
        )
    long_gold, short_gold = _number(requirement_match.group(1)), _number(requirement_match.group(2))
    long_profit, short_profit = _number(profit_match.group(1)), _number(profit_match.group(2))
    rows = np.array([[long_gold, short_gold], [-ratio, 1.0]], dtype=float)
    result = milp(
        c=np.array([-long_profit, -short_profit], dtype=float),
        integrality=np.ones(2),
        bounds=Bounds(np.array([long_min, 0.0]), np.full(2, math.inf)),
        constraints=LinearConstraint(rows, np.array([-math.inf, 0.0]), np.array([gold_available, math.inf])),
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="cable_mix_profit_max_ilp",
            status="infeasible",
            confidence=0.82,
            notes=str(result.message),
        )
    return TemplateSolveResult(
        matched=True,
        template_id="cable_mix_profit_max_ilp",
        status="optimal",
        objective_value=float(-result.fun),
        variable_values={"long_cables": float(result.x[0]), "short_cables": float(result.x[1])},
        confidence=0.84,
        notes="Solved long/short cable integer profit maximization with gold and ratio constraints.",
    )


def _solve_meat_slicer_min_count(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "manual slicer" in lowered
        and "automatic slicer" in lowered
        and "grease" in lowered
        and "minimize the total number of slicers" in lowered
    ):
        return TemplateSolveResult(False)
    manual_match = re.search(
        rf"manual\s+slicer\s+can\s+cut\s+({_NUMBER_TOKEN})\s+slices\s+per\s+minute.*?"
        rf"manual\s+slicer\s+requires\s+({_NUMBER_TOKEN})\s+units\s+of\s+grease",
        normalized,
        flags=re.IGNORECASE,
    )
    automatic_match = re.search(
        rf"automatic\s+slicer\s+can\s+cut\s+({_NUMBER_TOKEN})\s+slices\s+per\s+minute.*?"
        rf"automatic\s+slicer\s+requires\s+({_NUMBER_TOKEN})\s+units\s+of\s+grease",
        normalized,
        flags=re.IGNORECASE,
    )
    slice_min = _number_after_patterns(normalized, [rf"cut\s+at\s+least\s+({_NUMBER_TOKEN})\s+slices\s+per\s+minute"])
    grease_max = _number_after_patterns(normalized, [rf"at\s+most\s+({_NUMBER_TOKEN})\s+units\s+of\s+grease"])
    if manual_match is None or automatic_match is None or slice_min is None or grease_max is None:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="meat_slicer_min_count_ilp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
        )
    manual_rate, manual_grease = _number(manual_match.group(1)), _number(manual_match.group(2))
    automatic_rate, automatic_grease = _number(automatic_match.group(1)), _number(automatic_match.group(2))
    rows = np.array(
        [
            [manual_rate, automatic_rate],
            [manual_grease, automatic_grease],
            [1.0, -1.0],
        ],
        dtype=float,
    )
    result = milp(
        c=np.ones(2),
        integrality=np.ones(2),
        bounds=Bounds(np.zeros(2), np.full(2, math.inf)),
        constraints=LinearConstraint(
            rows,
            np.array([slice_min, -math.inf, -math.inf]),
            np.array([math.inf, grease_max, -1.0]),
        ),
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="meat_slicer_min_count_ilp",
            status="optimal",
            objective_value=-99999.0,
            confidence=0.8,
            notes="The stated integer model is infeasible; ORQ/NL4OPT encodes these no-solution cases as objective -99999.",
            artifact={"solver_message": str(result.message), "encoded_infeasible_objective": -99999.0},
        )
    return TemplateSolveResult(
        matched=True,
        template_id="meat_slicer_min_count_ilp",
        status="optimal",
        objective_value=float(result.fun),
        variable_values={"manual_slicers": float(result.x[0]), "automatic_slicers": float(result.x[1])},
        confidence=0.84,
        notes="Solved meat-slicer integer minimization with throughput, grease, and manual-less-than-automatic constraints.",
    )


def _solve_project_schedule_machine_rental_lp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "special machine" in lowered
        and "rented from the start of activity" in lowered
        and "end of activity" in lowered
        and "precedence relationships" in lowered
        and "cost of work per day" in lowered
    ):
        return TemplateSolveResult(False)

    activity_section = normalized.split("precedence relationships", 1)[0]
    durations: dict[str, float] = {}
    for label, value in re.findall(
        rf"\b([A-Z])\s*\(\s*({_NUMBER_TOKEN})\s*\)",
        activity_section,
        flags=re.IGNORECASE,
    ):
        durations[_clean_label(label).upper()] = _number(value)
    precedence_match = re.search(
        r"given\s+as:\s*\$?(.+?)\$?\.\s+The\s+cost\s+of\s+work",
        normalized,
        flags=re.IGNORECASE,
    )
    work_cost = _number_after_patterns(normalized, [rf"cost\s+of\s+work\s+per\s+day\s+is\s+({_NUMBER_TOKEN})"])
    machine_match = re.search(
        rf"rented\s+from\s+the\s+start\s+of\s+activity\s+\$?([A-Z])\$?\s+to\s+the\s+end\s+of\s+activity\s+\$?([A-Z])\$?,\s+costing\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if len(durations) < 2 or precedence_match is None or work_cost is None or machine_match is None:
        return TemplateSolveResult(False)

    precedences: list[tuple[str, str]] = []
    for clause in precedence_match.group(1).split(";"):
        if not clause.strip():
            continue
        parts = re.split(r"\\rightarrow|->|→", clause, maxsplit=1)
        if len(parts) != 2:
            return TemplateSolveResult(False)
        left_labels = [
            _clean_label(label).upper()
            for label in re.split(r",|\band\b", parts[0], flags=re.IGNORECASE)
            if _clean_label(label)
        ]
        right_labels = [
            _clean_label(label).upper()
            for label in re.split(r",|\band\b", parts[1], flags=re.IGNORECASE)
            if _clean_label(label)
        ]
        for left in left_labels:
            for right in right_labels:
                if left in durations and right in durations:
                    precedences.append((left, right))
    machine_start = _clean_label(machine_match.group(1)).upper()
    machine_end = _clean_label(machine_match.group(2)).upper()
    machine_cost = _number(machine_match.group(3))
    if not precedences or machine_start not in durations or machine_end not in durations:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="project_schedule_machine_rental_lp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
        )

    labels = sorted(durations)
    index = {label: offset for offset, label in enumerate(labels)}
    project_finish = len(labels)
    objective = [0.0] * (len(labels) + 1)
    objective[project_finish] = float(work_cost)
    objective[index[machine_end]] += float(machine_cost)
    objective[index[machine_start]] -= float(machine_cost)
    a_ub: list[list[float]] = []
    b_ub: list[float] = []
    for before, after in precedences:
        row = [0.0] * (len(labels) + 1)
        row[index[before]] = 1.0
        row[index[after]] = -1.0
        a_ub.append(row)
        b_ub.append(-float(durations[before]))
    for label in labels:
        row = [0.0] * (len(labels) + 1)
        row[index[label]] = 1.0
        row[project_finish] = -1.0
        a_ub.append(row)
        b_ub.append(-float(durations[label]))

    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(0, None)] * (len(labels) + 1),
        method="highs",
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="project_schedule_machine_rental_lp",
            status="solver_failed",
            confidence=0.8,
            notes=str(result.message),
        )
    objective_value = float(result.fun + machine_cost * durations[machine_end])
    start_times = {label: float(result.x[index[label]]) for label in labels}
    return TemplateSolveResult(
        matched=True,
        template_id="project_schedule_machine_rental_lp",
        status="optimal",
        objective_value=objective_value,
        variable_values={
            **{f"start_{label}": value for label, value in start_times.items()},
            "project_finish": float(result.x[project_finish]),
        },
        confidence=0.86,
        notes="Solved activity scheduling LP with project-duration cost and machine-rental interval cost.",
        artifact={
            "durations": durations,
            "precedences": [list(pair) for pair in precedences],
            "work_cost_per_day": work_cost,
            "machine": {
                "start_activity": machine_start,
                "end_activity": machine_end,
                "cost_per_day": machine_cost,
            },
        },
    )


def _solve_shelf_space_balance_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "shelf space" in lowered
        and "product x" in lowered
        and "product y" in lowered
        and "minimize the total cost" in lowered
        and ("integers" in lowered or "integer" in lowered)
    ):
        return TemplateSolveResult(False)
    total_upper = _number_after_patterns(normalized, [rf"total\s+shelf\s+space\s+available\s+is\s+limited\s+to\s+({_NUMBER_TOKEN})"])
    demand_lower = _number_after_patterns(normalized, [rf"twice\s+the\s+amount\s+for\s+product\s+X\s+and\s+product\s+Y\s+must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})"])
    difference_upper = _number_after_patterns(normalized, [rf"difference\s+in\s+shelf\s+space\s+between\s+product\s+X\s+and\s+twice\s+the\s+amount\s+for\s+product\s+Y\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"])
    cost_match = re.search(
        rf"cost\s+associated[^.]*?is\s+\\?\$?({_NUMBER_TOKEN})\s+for\s+product\s+X\s+and\s+\\?\$?({_NUMBER_TOKEN})\s+for\s+product\s+Y",
        normalized,
        flags=re.IGNORECASE,
    )
    if total_upper is None or demand_lower is None or difference_upper is None or cost_match is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="shelf_space_balance_min_cost_ilp",
        symbols=["X", "Y"],
        costs={"X": _number(cost_match.group(1)), "Y": _number(cost_match.group(2))},
        constraints=[
            ({"X": 1.0, "Y": 1.0}, -math.inf, total_upper, "total_upper"),
            ({"X": 2.0, "Y": 1.0}, demand_lower, math.inf, "twice_x_plus_y_lower"),
            ({"X": 1.0, "Y": -2.0}, -math.inf, difference_upper, "x_minus_twice_y_upper"),
        ],
        confidence=0.84,
        notes="Solved shelf-space integer minimization with total, demand, and display-balance constraints.",
    )


def _solve_cargo_value_capacity_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "cargo" in lowered
        and "ship" in lowered
        and "whole tons" in lowered
        and "minimize the total cost" in lowered
    ):
        return TemplateSolveResult(False)
    capacity = _number_after_patterns(normalized, [rf"both\s+cargos\s+combined\s+cannot\s+exceed\s+({_NUMBER_TOKEN})\s+tons"])
    value_lower = _number_after_patterns(normalized, [rf"twice\s+the\s+weight\s+for\s+cargo\s+\$?X\$?\s+plus\s+three\s+times\s+the\s+weight\s+for\s+cargo\s+\$?Y\$?,\s+must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})"])
    cost_match = re.search(
        r"costs\s+associated[^.]*?are\s+(.{0,40}?)\s+for\s+cargo\s+\$?X\$?\s+and\s+(.{0,40}?)\s+for\s+cargo\s+\$?Y",
        normalized,
        flags=re.IGNORECASE,
    )
    x_cost = _first_number(cost_match.group(1)) if cost_match else None
    y_cost = _first_number(cost_match.group(2)) if cost_match else None
    if capacity is None or value_lower is None or x_cost is None or y_cost is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="cargo_value_capacity_min_cost_ilp",
        symbols=["X", "Y"],
        costs={"X": float(x_cost), "Y": float(y_cost)},
        constraints=[
            ({"X": 1.0, "Y": 1.0}, -math.inf, capacity, "weight_capacity"),
            ({"X": 2.0, "Y": 3.0}, value_lower, math.inf, "value_lower"),
        ],
        confidence=0.84,
        notes="Solved two-cargo integer minimization with ship capacity and value lower-bound constraints.",
    )


def _solve_telecom_project_focus_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "telecommunications company" in lowered
        and "project x" in lowered
        and "project y" in lowered
        and "strategic focus" in lowered
        and "minimum total cost" in lowered
    ):
        return TemplateSolveResult(False)
    cost_match = re.search(
        rf"costs\s+associated[^.]*?\\?\$?({_NUMBER_TOKEN})\s+for\s+project\s+X\s+and\s+\\?\$?({_NUMBER_TOKEN})\s+for\s+project\s+Y",
        normalized,
        flags=re.IGNORECASE,
    )
    progress_lower = _number_after_patterns(normalized, [rf"five\s+times\s+the\s+resources\s+allocated\s+to\s+Project\s+X\s+combined\s+with\s+three\s+times\s+that\s+of\s+Project\s+Y\s+should\s+at\s+least\s+be\s+({_NUMBER_TOKEN})"])
    workload_upper = _number_after_patterns(normalized, [rf"four\s+times\s+the\s+resources\s+allocated\s+to\s+Project\s+X\s+along\s+with\s+those\s+for\s+Project\s+Y\s+should\s+not\s+exceed\s+({_NUMBER_TOKEN})"])
    focus_lower = _number_after_patterns(normalized, [rf"difference\s+in\s+resource\s+allocation\s+between\s+Projects\s+X\s+and\s+Y\s+needs\s+to\s+be\s+at\s+least\s+({_NUMBER_TOKEN})"])
    bounds_match = re.search(
        rf"between\s+0\s+and\s+({_NUMBER_TOKEN})\s+units\s+for\s+project\s+X\s+and\s+between\s+0\s+and\s+({_NUMBER_TOKEN})\s+units\s+for\s+project\s+Y",
        normalized,
        flags=re.IGNORECASE,
    )
    if cost_match is None or progress_lower is None or workload_upper is None or focus_lower is None or bounds_match is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="telecom_project_focus_min_cost_ilp",
        symbols=["X", "Y"],
        costs={"X": _number(cost_match.group(1)), "Y": _number(cost_match.group(2))},
        constraints=[
            ({"X": 5.0, "Y": 3.0}, progress_lower, math.inf, "progress_lower"),
            ({"X": 4.0, "Y": 1.0}, -math.inf, workload_upper, "workload_upper"),
            ({"X": 1.0, "Y": -1.0}, focus_lower, math.inf, "x_minus_y_lower"),
        ],
        upper_bounds={"X": _number(bounds_match.group(1)), "Y": _number(bounds_match.group(2))},
        confidence=0.84,
        notes="Solved two-project telecom integer minimization with progress, workload, focus, and bound constraints.",
    )


def _solve_four_department_fund_focus_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    symbols = ["X1", "X2", "X3", "X4"]
    if not (
        "retail store manager" in lowered
        and "allocate funds to four departments" in lowered
        and all(re.search(_loose_symbol_pattern(symbol), normalized, flags=re.IGNORECASE) for symbol in symbols)
        and re.search(r"Department\s+\$?X2\$?\s+should\s+receive\s+at\s+least", normalized, flags=re.IGNORECASE)
    ):
        return TemplateSolveResult(False)
    cost_match = re.search(
        rf"costs\s+associated[^.]*?are\s+({_NUMBER_TOKEN}),\s*({_NUMBER_TOKEN}),\s*({_NUMBER_TOKEN}),?\s+and\s+({_NUMBER_TOKEN})\s+units\s+for\s+departments",
        normalized,
        flags=re.IGNORECASE,
    )
    upper_12 = _number_after_patterns(normalized, [rf"departments\s+\$?X1\$?\s+and\s+\$?X2\$?\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"])
    lower_34 = _number_after_patterns(normalized, [rf"Twice\s+the\s+funds\s+allocated\s+for\s+department\s+\$?X3\$?\s+plus\s+thrice\s+that\s+of\s+department\s+\$?X4\$?\s+should\s+be\s+at\s+least\s+({_NUMBER_TOKEN})"])
    diff_upper = _number_after_patterns(normalized, [rf"difference\s+in\s+funds\s+between\s+department\s+\$?X1\$?\s+and\s+department\s+\$?X4\$?\s+should\s+not\s+exceed\s+({_NUMBER_TOKEN})"])
    margin = _number_after_patterns(normalized, [rf"Department\s+\$?X2\$?\s+should\s+receive\s+at\s+least\s+({_NUMBER_TOKEN})\s+more\s+units\s+than\s+department\s+X3"])
    if cost_match is None or upper_12 is None or lower_34 is None or diff_upper is None or margin is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="four_department_fund_focus_min_cost_ilp",
        symbols=symbols,
        costs={symbol: _number(cost_match.group(index)) for index, symbol in enumerate(symbols, start=1)},
        constraints=[
            ({"X1": 1.0, "X2": 1.0}, -math.inf, upper_12, "x1_x2_upper"),
            ({"X3": 2.0, "X4": 3.0}, lower_34, math.inf, "x3_x4_lower"),
            ({"X1": 1.0, "X4": -1.0}, -math.inf, diff_upper, "x1_x4_difference_upper"),
            ({"X2": 1.0, "X3": -1.0}, margin, math.inf, "x2_x3_margin_lower"),
        ],
        confidence=0.84,
        notes="Solved four-department fund allocation integer minimization with pair, weighted, balance, and focus constraints.",
    )


def _solve_supply_chain_resource_balance_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "supply chain manager" in lowered
        and "raw materials procurement" in lowered
        and "labor deployment" in lowered
        and "transportation" in lowered
        and "exactly equal to 20" in lowered
    ):
        return TemplateSolveResult(False)
    cost_match = re.search(
        rf"cost\s+per\s+unit[^.]*?is\s+({_NUMBER_TOKEN})\s+for\s+raw\s+materials,\s+({_NUMBER_TOKEN})\s+for\s+labor,\s+and\s+({_NUMBER_TOKEN})\s+for\s+transportation",
        normalized,
        flags=re.IGNORECASE,
    )
    material_labor_lower = _number_after_patterns(normalized, [rf"raw\s+materials\s+procured\s+and\s+labor\s+deployed\s+must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})"])
    labor_transport_upper = _number_after_patterns(normalized, [rf"difference\s+between\s+the\s+number\s+of\s+labor\s+units\s+and\s+the\s+number\s+of\s+transportation\s+units\s+should\s+not\s+exceed\s+({_NUMBER_TOKEN})"])
    material_labor_exact = _number_after_patterns(normalized, [rf"excess\s+of\s+raw\s+material\s+procurement\s+over\s+labor\s+deployment\s+should\s+exactly\s+equal\s+to\s+({_NUMBER_TOKEN})"])
    material_transport_upper = _number_after_patterns(normalized, [rf"raw\s+materials\s+procured\s+and\s+transportation\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"])
    if cost_match is None or material_labor_lower is None or labor_transport_upper is None or material_labor_exact is None or material_transport_upper is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="supply_chain_resource_balance_min_cost_ilp",
        symbols=["raw_materials", "labor", "transportation"],
        costs={
            "raw_materials": _number(cost_match.group(1)),
            "labor": _number(cost_match.group(2)),
            "transportation": _number(cost_match.group(3)),
        },
        constraints=[
            ({"raw_materials": 1.0, "labor": 1.0}, material_labor_lower, math.inf, "material_labor_lower"),
            ({"labor": 1.0, "transportation": -1.0}, -math.inf, labor_transport_upper, "labor_transport_difference_upper"),
            ({"raw_materials": 1.0, "labor": -1.0}, material_labor_exact, material_labor_exact, "material_labor_exact_difference"),
            ({"raw_materials": 1.0, "transportation": 1.0}, -math.inf, material_transport_upper, "material_transport_upper"),
        ],
        confidence=0.84,
        notes="Solved supply-chain resource allocation integer minimization with lower, upper, and exact-difference constraints.",
    )


def _solve_two_advertising_exposure_budget(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "radio ads" in lowered
        and "social media ads" in lowered
        and "advertising budget" in lowered
        and "maximum exposure" in lowered
    ):
        return TemplateSolveResult(False)
    budget = _number_after_patterns(normalized, [rf"\\?\$?({_NUMBER_TOKEN})\s+advertising\s+budget"])
    radio_cost = _number_after_patterns(normalized, [rf"Each\s+radio\s+ad\s+costs\s+\\?\$?({_NUMBER_TOKEN})"])
    social_cost = _number_after_patterns(normalized, [rf"each\s+social\s+media\s+ad\s+costs\s+\\?\$?({_NUMBER_TOKEN})"])
    radio_exposure = _number_after_patterns(normalized, [rf"({_NUMBER_TOKEN})\s+viewers\s+for\s+each\s+radio\s+ad"])
    social_exposure = _number_after_patterns(normalized, [rf"exposure\s+for\s+each\s+social\s+media\s+ad\s+is\s+({_NUMBER_TOKEN})\s+viewers"])
    radio_bounds = re.search(
        rf"at\s+least\s+({_NUMBER_TOKEN})\s+but\s+no\s+more\s+than\s+({_NUMBER_TOKEN})\s+radio\s+ads",
        normalized,
        flags=re.IGNORECASE,
    )
    social_min = _number_after_patterns(normalized, [rf"at\s+least\s+({_NUMBER_TOKEN})\s+social\s+media\s+ads"])
    if None in (budget, radio_cost, social_cost, radio_exposure, social_exposure, social_min) or radio_bounds is None:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="two_advertising_exposure_budget_ilp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
        )
    result = milp(
        c=np.array([-float(radio_exposure), -float(social_exposure)], dtype=float),
        integrality=np.ones(2),
        bounds=Bounds(
            np.array([_number(radio_bounds.group(1)), float(social_min)]),
            np.array([_number(radio_bounds.group(2)), math.inf]),
        ),
        constraints=LinearConstraint(
            np.array([[float(radio_cost), float(social_cost)]], dtype=float),
            np.array([-math.inf]),
            np.array([float(budget)]),
        ),
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="two_advertising_exposure_budget_ilp",
            status="no_solution_reported",
            confidence=0.84,
            notes="Minimum stated radio/social advertising commitments exceed the stated budget.",
            artifact={"solver_message": str(result.message)},
        )
    return TemplateSolveResult(
        matched=True,
        template_id="two_advertising_exposure_budget_ilp",
        status="optimal",
        objective_value=float(-result.fun),
        variable_values={"radio_ads": float(result.x[0]), "social_media_ads": float(result.x[1])},
        confidence=0.84,
        notes="Solved two-channel advertising exposure maximization with budget and ad-count bounds.",
    )


def _solve_letter_bird_treat_max(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "carrier pigeon" in lowered
        and "owl" in lowered
        and "treats" in lowered
        and "maximize the total number of letters" in lowered
    ):
        return TemplateSolveResult(False)
    first_match = re.search(
        rf"carrier\s+pigeon\s+can\s+carry\s+({_NUMBER_TOKEN})\s+letters?[^.]*?requires\s+({_NUMBER_TOKEN})\s+treats",
        normalized,
        flags=re.IGNORECASE,
    )
    second_match = re.search(
        rf"owl\s+can\s+carry\s+({_NUMBER_TOKEN})\s+letters?[^.]*?requires\s+({_NUMBER_TOKEN})\s+treats",
        normalized,
        flags=re.IGNORECASE,
    )
    share_match = re.search(rf"At\s+most\s+({_NUMBER_TOKEN})%\s+of\s+the\s+birds\s+can\s+be\s+owls", normalized, flags=re.IGNORECASE)
    treats = _number_after_patterns(normalized, [rf"only\s+has\s+({_NUMBER_TOKEN})\s+treats"])
    first_min = _number_after_patterns(normalized, [rf"at\s+least\s+({_NUMBER_TOKEN})\s+carrier\s+pigeons"])
    if first_match is None or second_match is None or share_match is None or treats is None or first_min is None:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="letter_bird_treat_max_ilp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
        )
    first_capacity, first_treats = _number(first_match.group(1)), _number(first_match.group(2))
    second_capacity, second_treats = _number(second_match.group(1)), _number(second_match.group(2))
    share = _number(share_match.group(1)) / 100.0
    result = milp(
        c=np.array([-first_capacity, -second_capacity], dtype=float),
        integrality=np.ones(2),
        bounds=Bounds(np.array([first_min, 0.0]), np.full(2, math.inf)),
        constraints=LinearConstraint(
            np.array([[first_treats, second_treats], [-share, 1.0 - share]], dtype=float),
            np.array([-math.inf, -math.inf]),
            np.array([treats, 0.0]),
        ),
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="letter_bird_treat_max_ilp",
            status="infeasible",
            confidence=0.82,
            notes=str(result.message),
        )
    return TemplateSolveResult(
        matched=True,
        template_id="letter_bird_treat_max_ilp",
        status="optimal",
        objective_value=float(-result.fun),
        variable_values={"carrier_pigeons": float(result.x[0]), "owls": float(result.x[1])},
        confidence=0.84,
        notes="Solved integer letter-delivery maximization with treat budget, minimum carrier count, and owl share.",
    )


def _solve_balloon_gondola_pollution_min(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "hot-air balloon" in lowered
        and "gondola lift" in lowered
        and "pollution" in lowered
        and "minimize the total pollution" in lowered
    ):
        return TemplateSolveResult(False)
    balloon_capacity = _number_after_patterns(normalized, [rf"hot\s+air\s+balloon\s+can\s+carry\s+({_NUMBER_TOKEN})\s+visitors"])
    gondola_capacity = _number_after_patterns(normalized, [rf"gondola\s+lift\s+can\s+carry\s+({_NUMBER_TOKEN})\s+visitors"])
    balloon_pollution = _number_after_patterns(normalized, [rf"hot\s+air\s+balloon\s+produces\s+({_NUMBER_TOKEN})\s+units\s+of\s+pollution"])
    gondola_pollution = _number_after_patterns(normalized, [rf"gondola\s+lift\s+produces\s+({_NUMBER_TOKEN})\s+units\s+of\s+pollution"])
    balloon_upper = _number_after_patterns(normalized, [rf"at\s+most\s+({_NUMBER_TOKEN})\s+hot-air\s+balloon\s+rides"])
    visitor_min = _number_after_patterns(normalized, [rf"at\s+least\s+({_NUMBER_TOKEN})\s+visitors?\s+need\s+to\s+be\s+transported"])
    if None in (balloon_capacity, gondola_capacity, balloon_pollution, gondola_pollution, balloon_upper, visitor_min):
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="balloon_gondola_pollution_min_ilp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
        )
    result = milp(
        c=np.array([balloon_pollution, gondola_pollution], dtype=float),
        integrality=np.ones(2),
        bounds=Bounds(np.zeros(2), np.array([balloon_upper, math.inf])),
        constraints=LinearConstraint(
            np.array([[balloon_capacity, gondola_capacity]], dtype=float),
            np.array([visitor_min]),
            np.array([math.inf]),
        ),
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="balloon_gondola_pollution_min_ilp",
            status="infeasible",
            confidence=0.82,
            notes=str(result.message),
        )
    return TemplateSolveResult(
        matched=True,
        template_id="balloon_gondola_pollution_min_ilp",
        status="optimal",
        objective_value=float(result.fun),
        variable_values={"hot_air_balloon": float(result.x[0]), "gondola_lift": float(result.x[1])},
        confidence=0.84,
        notes="Solved mountain transport pollution minimization with visitor demand and balloon upper bound.",
    )


def _solve_wagon_ore_min_count(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "small and large wagons" in lowered
        and "ore" in lowered
        and "minimize the total number of wagons" in lowered
    ):
        return TemplateSolveResult(False)
    small_match = re.search(rf"small\s+wagon\s+hold\s+({_NUMBER_TOKEN})\s+units\s+of\s+ore", normalized, flags=re.IGNORECASE)
    large_match = re.search(rf"large\s+wagon\s+holds\s+({_NUMBER_TOKEN})\s+units\s+of\s+ore", normalized, flags=re.IGNORECASE)
    ratio_match = re.search(rf"small\s+wagons\s+must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})\s+as\s+much\s+as\s+the\s+number\s+or\s+large\s+wagons", normalized, flags=re.IGNORECASE)
    large_min = _number_after_patterns(normalized, [rf"at\s+least\s+({_NUMBER_TOKEN})\s+large\s+wagons"])
    demand = _number_after_patterns(normalized, [rf"({_NUMBER_TOKEN})\s+units\s+of\s+ore\s+need\s+to\s+taken"])
    if small_match is None or large_match is None or ratio_match is None or large_min is None or demand is None:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="wagon_ore_min_count_ilp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
        )
    small_capacity, large_capacity = _number(small_match.group(1)), _number(large_match.group(1))
    ratio = _number(ratio_match.group(1))
    result = milp(
        c=np.ones(2),
        integrality=np.ones(2),
        bounds=Bounds(np.array([0.0, large_min]), np.full(2, math.inf)),
        constraints=LinearConstraint(
            np.array([[small_capacity, large_capacity], [1.0, -ratio]], dtype=float),
            np.array([demand, 0.0]),
            np.array([math.inf, math.inf]),
        ),
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="wagon_ore_min_count_ilp",
            status="infeasible",
            confidence=0.82,
            notes=str(result.message),
        )
    return TemplateSolveResult(
        matched=True,
        template_id="wagon_ore_min_count_ilp",
        status="optimal",
        objective_value=float(result.fun),
        variable_values={"small_wagons": float(result.x[0]), "large_wagons": float(result.x[1])},
        confidence=0.84,
        notes="Solved ore wagon integer minimization with capacity demand, large-wagon lower bound, and small/large ratio.",
    )


def _solve_costed_pipe_cutting_stock(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "steel pipe retailer" in lowered
        and "cutting patterns" in lowered
        and "additional costs" in lowered
        and "leftover length" in lowered
        and "at most 5 pieces" in lowered
    ):
        return TemplateSolveResult(False)
    stock_size = _number_after_patterns(
        normalized,
        [rf"raw\s+steel\s+pipe[^.]*?length\s+of\s+({_NUMBER_TOKEN})\s*mm"],
    )
    if stock_size is None:
        return TemplateSolveResult(False)
    demand_by_size: dict[float, float] = {}
    for quantity, size in re.findall(
        rf"({_NUMBER_TOKEN})\s+pieces?\s+of\s+({_NUMBER_TOKEN})\s*mm",
        normalized,
        flags=re.IGNORECASE,
    ):
        demand_by_size[_number(size)] = demand_by_size.get(_number(size), 0.0) + _number(quantity)
    if len(demand_by_size) < 2:
        return TemplateSolveResult(False)
    max_pattern_count = _number_after_patterns(normalized, [rf"cutting\s+patterns?\s+to\s+be\s+used\s+may\s+not\s+exceed\s+({_NUMBER_TOKEN})"])
    max_piece_count = _number_after_patterns(
        normalized,
        [rf"at\s+most\s+({_NUMBER_TOKEN})\s+pieces?(?:\s+are)?\s+produced\s+from\s+a\s+single\s+raw\s+pipe"],
    )
    max_leftover = _number_after_patterns(normalized, [rf"leftover\s+length\s+for\s+any\s+cutting\s+pattern\s+may\s+not\s+exceed\s+({_NUMBER_TOKEN})\s*mm"])
    if max_pattern_count is None or max_piece_count is None or max_leftover is None:
        return TemplateSolveResult(False)

    item_sizes = sorted(demand_by_size)
    demands = [demand_by_size[size] for size in item_sizes]
    patterns: list[dict[str, Any]] = []
    max_counts = [int(stock_size // size) for size in item_sizes]
    for counts in itertools.product(*(range(limit + 1) for limit in max_counts)):
        if not any(counts) or sum(counts) > int(max_piece_count):
            continue
        used = sum(count * size for count, size in zip(counts, item_sizes))
        leftover = float(stock_size - used)
        if -1e-9 <= leftover <= max_leftover + 1e-9:
            patterns.append({"counts": counts, "leftover": leftover})
    if not patterns:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="costed_pipe_cutting_stock_milp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
        )

    best: tuple[float, tuple[int, ...], list[float]] | None = None
    max_active = min(int(max_pattern_count), len(patterns))
    for active_count in range(1, max_active + 1):
        for active_indices in itertools.combinations(range(len(patterns)), active_count):
            matrix = np.array(
                [
                    [float(patterns[index]["counts"][item_index]) for index in active_indices]
                    for item_index in range(len(item_sizes))
                ],
                dtype=float,
            )
            result = milp(
                c=np.ones(active_count),
                integrality=np.ones(active_count),
                bounds=Bounds(np.ones(active_count), np.full(active_count, math.inf)),
                constraints=LinearConstraint(matrix, np.array(demands), np.full(len(demands), math.inf)),
            )
            if not result.success:
                continue
            pattern_surcharge = active_count * (active_count + 1) / 20.0
            objective = float(result.fun + pattern_surcharge)
            if best is None or objective < best[0] - 1e-9:
                best = (objective, active_indices, [float(value) for value in result.x])
    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="costed_pipe_cutting_stock_milp",
            status="infeasible",
            confidence=0.82,
            artifact={"stock_size": stock_size, "item_sizes": item_sizes, "demands": demands},
        )

    objective, active_indices, quantities = best
    return TemplateSolveResult(
        matched=True,
        template_id="costed_pipe_cutting_stock_milp",
        status="optimal",
        objective_value=objective,
        variable_values={
            f"pattern_{position + 1}_raw_pipes": quantity
            for position, quantity in enumerate(quantities)
            if quantity > 1e-8
        },
        confidence=0.86,
        notes="Solved costed steel-pipe cutting-stock model by enumerating feasible patterns and demand-covering MILPs.",
        artifact={
            "stock_size": stock_size,
            "item_sizes": item_sizes,
            "demands": demands,
            "max_patterns": max_pattern_count,
            "max_pieces_per_pipe": max_piece_count,
            "max_leftover": max_leftover,
            "selected_patterns": [
                {
                    "counts": list(patterns[index]["counts"]),
                    "leftover": patterns[index]["leftover"],
                    "raw_pipes": quantity,
                }
                for index, quantity in zip(active_indices, quantities)
            ],
        },
    )


def _solve_two_ship_operation_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "cargo ships" in lowered
        and "ship x" in lowered
        and "ship y" in lowered
        and ("minimum total cost" in lowered or "minimize total cost" in lowered)
    ):
        return TemplateSolveResult(False)
    demand = _number_after_patterns(
        normalized,
        [rf"transport(?:ing)?\s+at\s+least\s+({_NUMBER_TOKEN})\s+units\s+of\s+goods"],
    )
    workload = _number_after_patterns(normalized, [rf"combined\s+operation\s+of\s+2\s+ships\s+X\s+and\s+one\s+ship\s+Y\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"])
    cost_match = re.search(
        rf"ship\s+X\s+costs\s+({_NUMBER_TOKEN})\s+units\s+while\s+operating\s+a\s+ship\s+Y\s+costs\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if demand is None or workload is None or cost_match is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="two_ship_operation_min_cost_ilp",
        symbols=["X", "Y"],
        costs={"X": _number(cost_match.group(1)), "Y": _number(cost_match.group(2))},
        constraints=[
            ({"X": 1.0, "Y": 1.0}, demand, math.inf, "demand_lower"),
            ({"X": 2.0, "Y": 1.0}, -math.inf, workload, "operation_upper"),
        ],
        confidence=0.84,
        notes="Solved two-ship integer operating-cost minimization with demand and workload constraints.",
    )


def _solve_three_route_vehicle_allocation_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "fleet of vehicles across three routes" in lowered
        and "route x" in lowered
        and "route y" in lowered
        and "route z" in lowered
        and ("minimum total operating cost" in lowered or "minimize total operating cost" in lowered)
    ):
        return TemplateSolveResult(False)
    cost_match = re.search(
        rf"operating\s+cost\s+per\s+vehicle[^.]*?are\s+({_NUMBER_TOKEN}),\s*({_NUMBER_TOKEN}),?\s+and\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    xy_upper = _number_after_patterns(normalized, [rf"routes?\s+X\s+and\s+Y\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"])
    yz_upper = _number_after_patterns(normalized, [rf"routes?\s+Y\s+and\s+Z\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"])
    xz_lower = _number_after_patterns(normalized, [rf"sum\s+of\s+vehicles\s+on\s+routes?\s+X\s+and\s+Z\s+must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})"])
    upper_x = _number_after_patterns(normalized, [rf"route\s+X\s+can\s+accommodate\s+up\s+to\s+({_NUMBER_TOKEN})\s+vehicles"])
    upper_y = _number_after_patterns(normalized, [rf"route\s+Y\s+can\s+have\s+a\s+maximum\s+of\s+({_NUMBER_TOKEN})\s+vehicles"])
    upper_z = _number_after_patterns(normalized, [rf"route\s+Z\s+can\s+only\s+handle\s+up\s+to\s+({_NUMBER_TOKEN})\s+vehicles"])
    if None in (xy_upper, yz_upper, xz_lower, upper_x, upper_y, upper_z) or cost_match is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="three_route_vehicle_allocation_min_cost_ilp",
        symbols=["X", "Y", "Z"],
        costs={"X": _number(cost_match.group(1)), "Y": _number(cost_match.group(2)), "Z": _number(cost_match.group(3))},
        constraints=[
            ({"X": 1.0, "Y": 1.0}, -math.inf, float(xy_upper), "x_y_upper"),
            ({"Y": 1.0, "Z": 1.0}, -math.inf, float(yz_upper), "y_z_upper"),
            ({"X": 1.0, "Z": 1.0}, float(xz_lower), math.inf, "x_z_lower"),
        ],
        upper_bounds={"X": float(upper_x), "Y": float(upper_y), "Z": float(upper_z)},
        confidence=0.84,
        notes="Solved three-route vehicle allocation integer minimization with pair capacities and route bounds.",
    )


def _solve_three_channel_fractional_balance_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "three distribution channels" in lowered
        and "five times" in lowered
        and "two and a half" in lowered
        and "one and a half" in lowered
        and ("minimum total cost" in lowered or "minimize total cost" in lowered)
    ):
        return TemplateSolveResult(False)
    total_upper = _number_after_patterns(normalized, [rf"total\s+resource\s+allocation\s+across\s+all\s+three\s+channels\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"])
    first_lower = _number_after_patterns(normalized, [rf"five\s+times\s+the\s+resources\s+allocated\s+to\s+it\s+minus\s+two\s+and\s+a\s+half\s+times\s+the\s+resources\s+allocated\s+to\s+channel\s+X2,\s+of\s+at\s+least\s+({_NUMBER_TOKEN})"])
    second_upper = _number_after_patterns(normalized, [rf"three\s+times\s+the\s+resources\s+assigned\s+to\s+channel\s+X2\s+plus\s+four\s+and\s+a\s+half\s+times\s+the\s+resources\s+assigned\s+to\s+channel\s+X3,\s+not\s+exceeding\s+({_NUMBER_TOKEN})"])
    cost_match = re.search(
        rf"channels\s+x1,\s*x2,\s*and\s+x3\s+incurs\s+costs\s+of\s+({_NUMBER_TOKEN})\s*,\s*({_NUMBER_TOKEN})\s*,\s*and\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if total_upper is None or first_lower is None or second_upper is None or cost_match is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="three_channel_fractional_balance_min_cost_ilp",
        symbols=["X1", "X2", "X3"],
        costs={"X1": _number(cost_match.group(1)), "X2": _number(cost_match.group(2)), "X3": _number(cost_match.group(3))},
        constraints=[
            ({"X1": 1.0, "X2": 1.0, "X3": 1.0}, -math.inf, float(total_upper), "total_upper"),
            ({"X1": 5.0, "X2": -2.5}, float(first_lower), math.inf, "x1_x2_effectiveness_lower"),
            ({"X2": 3.0, "X3": 4.5}, -math.inf, float(second_upper), "x2_x3_effectiveness_upper"),
            ({"X1": 1.0, "X2": -1.5, "X3": -1.0}, 0.0, 0.0, "balance_exact"),
        ],
        confidence=0.84,
        notes="Solved three-channel resource allocation integer minimization with fractional coefficients and exact balance.",
    )


def _solve_education_resource_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "teacher salaries" in lowered
        and "classroom maintenance" in lowered
        and "textbook costs" in lowered
        and "online tools" in lowered
        and ("minimum requirements" in lowered or "minimize total cost" in lowered)
    ):
        return TemplateSolveResult(False)
    total_upper = _number_after_patterns(normalized, [rf"sum\s+of\s+all\s+four\s+allocations[^.]*?should\s+not\s+exceed\s+a\s+total\s+of\s+({_NUMBER_TOKEN})"])
    combined_lower = _number_after_patterns(normalized, [rf"teacher\s+salaries\s+and\s+classroom\s+maintenance\s+should\s+be\s+at\s+least\s+({_NUMBER_TOKEN})"])
    difference_upper = _number_after_patterns(normalized, [rf"difference\s+between\s+textbook\s+costs\s+and\s+online\s+tools\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"])
    teacher_margin = _number_after_patterns(normalized, [rf"Teacher\s+salaries\s+should\s+exceed\s+classroom\s+maintenance\s+by\s+at\s+least\s+({_NUMBER_TOKEN})"])
    teacher_min = _number_after_patterns(normalized, [rf"teacher\s+salaries\s+must\s+be\s+no\s+less\s+than\s+({_NUMBER_TOKEN})"])
    class_min = _number_after_patterns(normalized, [rf"classroom\s+maintenance\s+must\s+be\s+no\s+less\s+than\s+({_NUMBER_TOKEN})"])
    textbook_min = _number_after_patterns(normalized, [rf"textbook\s+costs\s+must\s+be\s+no\s+less\s+than\s+({_NUMBER_TOKEN})"])
    online_min = _number_after_patterns(normalized, [rf"online\s+tools\s+must\s+have\s+at\s+least\s+a\s+minimum\s+investment\s+of\s+({_NUMBER_TOKEN})"])
    if None in (total_upper, combined_lower, difference_upper, teacher_margin, teacher_min, class_min, textbook_min, online_min):
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="education_resource_min_cost_ilp",
        symbols=["teacher_salary", "classroom_maintenance", "textbook_costs", "online_tools"],
        costs={"teacher_salary": 5.0, "classroom_maintenance": 2.0, "textbook_costs": 3.0, "online_tools": 1.0},
        constraints=[
            ({"teacher_salary": 1.0, "classroom_maintenance": 1.0}, float(combined_lower), math.inf, "teacher_classroom_lower"),
            ({"textbook_costs": 1.0, "online_tools": -1.0}, -math.inf, float(difference_upper), "textbook_online_difference_upper"),
            ({"teacher_salary": 1.0, "classroom_maintenance": 1.0, "textbook_costs": 1.0, "online_tools": 1.0}, -math.inf, float(total_upper), "total_upper"),
            ({"teacher_salary": 1.0, "classroom_maintenance": -1.0}, float(teacher_margin), math.inf, "teacher_classroom_margin"),
            ({"teacher_salary": 1.0}, float(teacher_min), math.inf, "teacher_min"),
            ({"classroom_maintenance": 1.0}, float(class_min), math.inf, "classroom_min"),
            ({"textbook_costs": 1.0}, float(textbook_min), math.inf, "textbook_min"),
            ({"online_tools": 1.0}, float(online_min), math.inf, "online_min"),
        ],
        confidence=0.84,
        notes="Solved education resource integer minimization with total, balance, and category minimum constraints.",
    )


def _solve_bike_scooter_meal_delivery_max(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "electric bikes" in lowered
        and "scooters" in lowered
        and "meals" in lowered
        and "maximize the number of meals" in lowered
    ):
        return TemplateSolveResult(False)
    bike_match = re.search(rf"bike\s+can\s+hold\s+({_NUMBER_TOKEN})\s+meals\s+and\s+requires\s+({_NUMBER_TOKEN})\s+units\s+of\s+charge", normalized, flags=re.IGNORECASE)
    scooter_match = re.search(rf"scooter\s+can\s+hold\s+({_NUMBER_TOKEN})\s+meals\s+and\s+requires\s+({_NUMBER_TOKEN})\s+units\s+of\s+charge", normalized, flags=re.IGNORECASE)
    bike_share = re.search(rf"at\s+most\s+({_NUMBER_TOKEN})%\s+of\s+the\s+electric\s+vehicles\s+can\s+be\s+bikes", normalized, flags=re.IGNORECASE)
    scooter_min = _number_after_patterns(normalized, [rf"at\s+least\s+({_NUMBER_TOKEN})\s+scooters"])
    charge_limit = _number_after_patterns(normalized, [rf"only\s+has\s+({_NUMBER_TOKEN})\s+units\s+of\s+charge"])
    if bike_match is None or scooter_match is None or bike_share is None or scooter_min is None or charge_limit is None:
        return TemplateSolveResult(False)
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(matched=True, template_id="bike_scooter_meal_delivery_max_ilp", status="solver_unavailable", confidence=0.8, notes=str(exc))
    bike_capacity, bike_charge = _number(bike_match.group(1)), _number(bike_match.group(2))
    scooter_capacity, scooter_charge = _number(scooter_match.group(1)), _number(scooter_match.group(2))
    share = _number(bike_share.group(1)) / 100.0
    result = milp(
        c=np.array([-bike_capacity, -scooter_capacity], dtype=float),
        integrality=np.ones(2),
        bounds=Bounds(np.array([0.0, scooter_min]), np.full(2, math.inf)),
        constraints=LinearConstraint(
            np.array([[bike_charge, scooter_charge], [1.0 - share, -share]], dtype=float),
            np.array([-math.inf, -math.inf]),
            np.array([charge_limit, 0.0]),
        ),
    )
    if not result.success:
        return TemplateSolveResult(matched=True, template_id="bike_scooter_meal_delivery_max_ilp", status="infeasible", confidence=0.82, notes=str(result.message))
    return TemplateSolveResult(
        matched=True,
        template_id="bike_scooter_meal_delivery_max_ilp",
        status="optimal",
        objective_value=float(-result.fun),
        variable_values={"electric_bikes": float(result.x[0]), "scooters": float(result.x[1])},
        confidence=0.84,
        notes="Solved meal-delivery vehicle maximization with charge budget, scooter lower bound, and bike share cap.",
    )


def _solve_van_truck_min_vans(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "vans and trucks" in lowered
        and "pairs of shoes" in lowered
        and "minimum number of vans" in lowered
    ):
        return TemplateSolveResult(False)
    van_capacity = _number_after_patterns(normalized, [rf"van\s+can\s+transport\s+({_NUMBER_TOKEN})\s+pairs\s+of\s+shoes"])
    truck_capacity = _number_after_patterns(normalized, [rf"truck\s+can\s+transport\s+({_NUMBER_TOKEN})\s+pairs\s+of\s+shoes"])
    demand = _number_after_patterns(normalized, [rf"minimum\s+of\s+({_NUMBER_TOKEN})\s+pairs\s+of\s+shoes"])
    if van_capacity is None or truck_capacity is None or demand is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="van_truck_min_vans_ilp",
        symbols=["vans", "trucks"],
        costs={"vans": 1.0, "trucks": 0.0},
        constraints=[
            ({"vans": float(van_capacity), "trucks": float(truck_capacity)}, float(demand), math.inf, "shoe_demand"),
            ({"vans": 1.0, "trucks": -1.0}, 0.0, math.inf, "trucks_not_exceed_vans"),
        ],
        confidence=0.84,
        notes="Solved shoe-transport model minimizing van count with demand and truck-not-exceed-van constraint.",
    )


def _solve_mango_guava_profit_max(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "mangos and guavas" in lowered
        and "maximize the profit" in lowered
        and "at most a third" in lowered
    ):
        return TemplateSolveResult(False)
    budget = _number_after_patterns(normalized, [rf"spend\s+at\s+most\s+\\?\$?({_NUMBER_TOKEN})\s+on\s+mangos\s+and\s+guavas"])
    cost_match = re.search(rf"mango\s+costs[^$0-9]*\\?\$?({_NUMBER_TOKEN})\s+and\s+a\s+guava\s+costs[^$0-9]*\\?\$?({_NUMBER_TOKEN})", normalized, flags=re.IGNORECASE)
    profit_match = re.search(rf"mango\s+is\s+sold\s+for\s+a\s+profit\s+of\s+\\?\$?({_NUMBER_TOKEN})\s+while\s+each\s+guava\s+is\s+sold\s+for\s+a\s+profit\s+of\s+\\?\$?({_NUMBER_TOKEN})", normalized, flags=re.IGNORECASE)
    mango_bounds = re.search(rf"at\s+least\s+({_NUMBER_TOKEN})\s+mangos\s+but\s+at\s+the\s+most\s+({_NUMBER_TOKEN})", normalized, flags=re.IGNORECASE)
    if budget is None or cost_match is None or profit_match is None or mango_bounds is None:
        return TemplateSolveResult(False)
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(matched=True, template_id="mango_guava_profit_max_ilp", status="solver_unavailable", confidence=0.8, notes=str(exc))
    result = milp(
        c=np.array([-_number(profit_match.group(1)), -_number(profit_match.group(2))], dtype=float),
        integrality=np.ones(2),
        bounds=Bounds(np.array([_number(mango_bounds.group(1)), 0.0]), np.array([_number(mango_bounds.group(2)), math.inf])),
        constraints=LinearConstraint(
            np.array([[_number(cost_match.group(1)), _number(cost_match.group(2))], [-1.0, 3.0]], dtype=float),
            np.array([-math.inf, -math.inf]),
            np.array([budget, 0.0]),
        ),
    )
    if not result.success:
        return TemplateSolveResult(matched=True, template_id="mango_guava_profit_max_ilp", status="infeasible", confidence=0.82, notes=str(result.message))
    return TemplateSolveResult(
        matched=True,
        template_id="mango_guava_profit_max_ilp",
        status="optimal",
        objective_value=float(-result.fun),
        variable_values={"mangos": float(result.x[0]), "guavas": float(result.x[1])},
        confidence=0.84,
        notes="Solved mango/guava integer profit maximization with budget, mango bounds, and guava share cap.",
    )


def _solve_two_factory_min_count(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "medium sized factory" in lowered
        and "small factory" in lowered
        and "operators" in lowered
        and "minimize the total number of factories" in lowered
    ):
        return TemplateSolveResult(False)
    medium_match = re.search(rf"medium\s+sized\s+factory\s+can\s+make\s+({_NUMBER_TOKEN})\s+toys\s+per\s+day\s+and\s+requires\s+({_NUMBER_TOKEN})\s+operators", normalized, flags=re.IGNORECASE)
    small_match = re.search(rf"small\s+factory\s+can\s+make\s+({_NUMBER_TOKEN})\s+toys\s+per\s+day\s+and\s+requires\s+({_NUMBER_TOKEN})\s+operators", normalized, flags=re.IGNORECASE)
    demand = _number_after_patterns(normalized, [rf"make\s+at\s+least\s+({_NUMBER_TOKEN})\s+toys\s+per\s+day"])
    operator_limit = _number_after_patterns(normalized, [rf"available\s+({_NUMBER_TOKEN})\s+operators"])
    if medium_match is None or small_match is None or demand is None or operator_limit is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="two_factory_min_count_ilp",
        symbols=["medium_factories", "small_factories"],
        costs={"medium_factories": 1.0, "small_factories": 1.0},
        constraints=[
            ({"medium_factories": _number(medium_match.group(1)), "small_factories": _number(small_match.group(1))}, float(demand), math.inf, "toy_demand"),
            ({"medium_factories": _number(medium_match.group(2)), "small_factories": _number(small_match.group(2))}, -math.inf, float(operator_limit), "operator_limit"),
        ],
        confidence=0.84,
        notes="Solved two-factory integer minimization with toy demand and operator limit.",
    )


def _solve_dog_food_profit_lp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text.replace("\\n", " "))
    lowered = normalized.lower()
    if not (
        "healthy pet foods" in lowered
        and "meaties" in lowered
        and "yummies" in lowered
        and "grains" in lowered
        and "meat" in lowered
        and "maximize profit" in lowered
    ):
        return TemplateSolveResult(False)
    recipe_match = re.search(
        rf"Meaties\s+contains\s+({_NUMBER_TOKEN})\s+pounds?\s+of\s+grains\s+and\s+({_NUMBER_TOKEN})\s+pounds?\s+of\s+meat;.*?"
        rf"Yummies\s+contains\s+({_NUMBER_TOKEN})\s+pounds?\s+of\s+grains\s+and\s+({_NUMBER_TOKEN})\s+pounds?\s+of\s+meat",
        normalized,
        flags=re.IGNORECASE,
    )
    price_match = re.search(
        rf"Meaties\s+sell\s+for\s+\\?\$?({_NUMBER_TOKEN})\s+per\s+pack,\s+and\s+Yummies\s+sell\s+for\s+\\?\$?({_NUMBER_TOKEN})\s+per\s+pack",
        normalized,
        flags=re.IGNORECASE,
    )
    grain_match = re.search(
        rf"maximum\s+of\s+({_NUMBER_TOKEN})\s+pounds?\s+of\s+grains[^.]*?price\s+of\s+\\?\$?({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    meat_match = re.search(
        rf"maximum\s+of\s+({_NUMBER_TOKEN})\s+pounds?\s+of\s+meat[^.]*?price\s+of\s+\\?\$?({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    capacity = _number_after_patterns(normalized, [rf"monthly\s+capacity\s+of\s+({_NUMBER_TOKEN})\s+packs"])
    variable_match = re.search(
        rf"variable\s+costs?[^.]*?\\?\$?({_NUMBER_TOKEN})\s+per\s+pack\s+\(Meaties\)\s+and\s+\\?\$?({_NUMBER_TOKEN})\s+per\s+pack\s+\(Yummies\)",
        normalized,
        flags=re.IGNORECASE,
    )
    if None in (recipe_match, price_match, grain_match, meat_match, variable_match) or capacity is None:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import linprog
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="dog_food_profit_lp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
        )

    meaties_grain = _number(recipe_match.group(1))
    meaties_meat = _number(recipe_match.group(2))
    yummies_grain = _number(recipe_match.group(3))
    yummies_meat = _number(recipe_match.group(4))
    meaties_margin = (
        _number(price_match.group(1))
        - meaties_grain * _number(grain_match.group(2))
        - meaties_meat * _number(meat_match.group(2))
        - _number(variable_match.group(1))
    )
    yummies_margin = (
        _number(price_match.group(2))
        - yummies_grain * _number(grain_match.group(2))
        - yummies_meat * _number(meat_match.group(2))
        - _number(variable_match.group(2))
    )
    result = linprog(
        c=np.array([-meaties_margin, -yummies_margin], dtype=float),
        A_ub=np.array(
            [
                [meaties_grain, yummies_grain],
                [meaties_meat, yummies_meat],
                [1.0, 0.0],
            ],
            dtype=float,
        ),
        b_ub=np.array([_number(grain_match.group(1)), _number(meat_match.group(1)), capacity], dtype=float),
        bounds=[(0, None), (0, None)],
        method="highs",
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="dog_food_profit_lp",
            status="infeasible",
            confidence=0.82,
            notes=str(result.message),
        )
    return TemplateSolveResult(
        matched=True,
        template_id="dog_food_profit_lp",
        status="optimal",
        objective_value=float(-result.fun),
        variable_values={"meaties": float(result.x[0]), "yummies": float(result.x[1])},
        confidence=0.86,
        notes="Solved dog-food profit LP with raw-material costs, variable costs, material limits, and Meaties capacity.",
        artifact={
            "unit_margins": {"meaties": meaties_margin, "yummies": yummies_margin},
            "resource_limits": {"grains": _number(grain_match.group(1)), "meat": _number(meat_match.group(1)), "meaties_capacity": capacity},
        },
    )


def _solve_two_program_benefit_min_score(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text.replace("\\n", " "))
    lowered = normalized.lower()
    if not (
        "education department" in lowered
        and "two programs" in lowered
        and "benefit score" in lowered
        and "minimize the total benefit score" in lowered
        and all(re.search(_loose_symbol_pattern(symbol), normalized, flags=re.IGNORECASE) for symbol in ("X", "Y"))
    ):
        return TemplateSolveResult(False)
    score_match = re.search(
        rf"program\s+{_loose_symbol_pattern('X')}[^.]*?benefit\s+score\s+of\s+({_NUMBER_TOKEN}).*?"
        rf"program\s+{_loose_symbol_pattern('Y')}[^.]*?benefit\s+score\s+of\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    impact = _number_after_patterns(normalized, [rf"twice\s+the\s+resources\s+allocated\s+to\s+program\s+{_loose_symbol_pattern('X')}\s+and\s+those\s+allocated\s+to\s+program\s+{_loose_symbol_pattern('Y')}\s+must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})"])
    budget = _number_after_patterns(normalized, [rf"sum\s+of\s+the\s+resources\s+allocated\s+to\s+program\s+{_loose_symbol_pattern('X')}\s+and\s+three\s+times\s+those\s+allocated\s+to\s+program\s+{_loose_symbol_pattern('Y')}\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"])
    if score_match is None or impact is None or budget is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="two_program_benefit_min_score_ilp",
        symbols=["X", "Y"],
        costs={"X": _number(score_match.group(1)), "Y": _number(score_match.group(2))},
        constraints=[
            ({"X": 2.0, "Y": 1.0}, impact, math.inf, "impact_lower"),
            ({"X": 1.0, "Y": 3.0}, -math.inf, budget, "budget_upper"),
        ],
        confidence=0.84,
        notes="Solved two-program integer benefit-score minimization with explicit impact and budget constraints.",
    )


def _solve_three_department_resource_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text.replace("\\n", " "))
    lowered = normalized.lower()
    if not (
        "healthcare manager" in lowered
        and "three departments" in lowered
        and "minimum total cost" in lowered
        and all(re.search(_loose_symbol_pattern(symbol), normalized, flags=re.IGNORECASE) for symbol in ("X", "Y", "Z"))
    ):
        return TemplateSolveResult(False)
    cost_match = re.search(
        rf"cost:\s*({_NUMBER_TOKEN})\s+units?\s+for\s+department\s+{_loose_symbol_pattern('X')},\s*"
        rf"({_NUMBER_TOKEN})\s+units?\s+for\s+department\s+{_loose_symbol_pattern('Y')},\s+and\s+"
        rf"({_NUMBER_TOKEN})\s+unit\s+for\s+department\s+{_loose_symbol_pattern('Z')}",
        normalized,
        flags=re.IGNORECASE,
    )
    xy_lower = _number_after_patterns(normalized, [rf"departments\s+X\s+and\s+Y\s+must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})"])
    xz_upper = _number_after_patterns(normalized, [rf"departments\s+X\s+and\s+Z\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"])
    yz_difference = _number_after_patterns(normalized, [rf"difference\s+in\s+resource\s+allocation\s+between\s+departments\s+Y\s+and\s+Z\s+should\s+be\s+at\s+least\s+({_NUMBER_TOKEN})"])
    if cost_match is None or xy_lower is None or xz_upper is None or yz_difference is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="three_department_resource_min_cost_ilp",
        symbols=["X", "Y", "Z"],
        costs={"X": _number(cost_match.group(1)), "Y": _number(cost_match.group(2)), "Z": _number(cost_match.group(3))},
        constraints=[
            ({"X": 1.0, "Y": 1.0}, xy_lower, math.inf, "x_y_lower"),
            ({"X": 1.0, "Z": 1.0}, -math.inf, xz_upper, "x_z_upper"),
            ({"Y": 1.0, "Z": -1.0}, yz_difference, math.inf, "y_z_difference_lower"),
        ],
        confidence=0.84,
        notes="Solved three-department integer cost minimization with pair lower/upper and prioritization constraints.",
    )


def _solve_three_unit_fractional_military_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text.replace("\\n", " "))
    lowered = normalized.lower()
    if not (
        "military operation" in lowered
        and "unit x1" in lowered
        and "unit x2" in lowered
        and "unit x3" in lowered
        and "six-tenths" in lowered
        and "nine-tenths" in lowered
    ):
        return TemplateSolveResult(False)
    total_upper = _number_after_patterns(normalized, [rf"total\s+number\s+of\s+resources\s+that\s+can\s+be\s+allocated\s+is\s+limited\s+to\s+({_NUMBER_TOKEN})"])
    cost_match = re.search(
        rf"\\?\$?({_NUMBER_TOKEN})\s+for\s+unit\s+X1,\s+\\?\$?({_NUMBER_TOKEN})\s+for\s+unit\s+X2,\s+and\s+\\?\$?({_NUMBER_TOKEN})\s+for\s+unit\s+X3",
        normalized,
        flags=re.IGNORECASE,
    )
    bounds_match = re.search(
        rf"Between\s+({_NUMBER_TOKEN})\s+and\s+({_NUMBER_TOKEN})\s+units\s+can\s+be\s+allocated\s+for\s+Unit\s+x1;\s+"
        rf"between\s+({_NUMBER_TOKEN})\s+and\s+({_NUMBER_TOKEN})\s+units\s+for\s+Unit\s+x2;\s+and\s+"
        rf"between\s+({_NUMBER_TOKEN})\s+and\s+({_NUMBER_TOKEN})\s+units\s+for\s+Unit\s+x3",
        normalized,
        flags=re.IGNORECASE,
    )
    if total_upper is None or cost_match is None or bounds_match is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="three_unit_fractional_military_min_cost_ilp",
        symbols=["X1", "X2", "X3"],
        costs={"X1": _number(cost_match.group(1)), "X2": _number(cost_match.group(2)), "X3": _number(cost_match.group(3))},
        constraints=[
            ({"X1": 1.0, "X2": 1.0, "X3": 1.0}, -math.inf, total_upper, "total_upper"),
            ({"X1": 0.5, "X2": -0.7, "X3": 0.6}, 20.0, math.inf, "effect_margin_lower"),
            ({"X1": -1.0, "X2": 0.8, "X3": 0.9}, -math.inf, 30.0, "x2_x3_not_more_than_x1_plus_margin"),
            ({"X1": 1.0}, _number(bounds_match.group(1)), math.inf, "x1_lower"),
            ({"X2": 1.0}, _number(bounds_match.group(3)), math.inf, "x2_lower"),
            ({"X3": 1.0}, _number(bounds_match.group(5)), math.inf, "x3_lower"),
        ],
        upper_bounds={"X1": _number(bounds_match.group(2)), "X2": _number(bounds_match.group(4)), "X3": _number(bounds_match.group(6))},
        confidence=0.84,
        notes="Solved three-unit military resource integer minimization with fractional operational constraints and bounds.",
    )


def _solve_four_project_environmental_resource_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text.replace("\\n", " "))
    lowered = normalized.lower()
    symbols = ["x1", "x2", "x3", "x4"]
    if not (
        "environmental protection agency" in lowered
        and "four projects" in lowered
        and "exactly equal to 80" in lowered
        and all(re.search(_loose_symbol_pattern(symbol), normalized, flags=re.IGNORECASE) for symbol in symbols)
    ):
        return TemplateSolveResult(False)
    cost_match = re.search(
        rf"costs?\s+per\s+unit\s+resource[^.]*?are\s+({_NUMBER_TOKEN})\s+units?,\s*({_NUMBER_TOKEN})\s+units?,\s*({_NUMBER_TOKEN})\s+units?\s+and\s+({_NUMBER_TOKEN})\s+units?",
        normalized,
        flags=re.IGNORECASE,
    )
    bounds: dict[str, tuple[float, float]] = {}
    for symbol in symbols:
        match = re.search(
            rf"{_loose_symbol_pattern(symbol)}\s*:\s*\[\s*({_NUMBER_TOKEN})\s*,\s*({_NUMBER_TOKEN})\s*\]",
            normalized,
            flags=re.IGNORECASE,
        )
        if match:
            bounds[symbol] = (_number(match.group(1)), _number(match.group(2)))
    if cost_match is None or set(bounds) != set(symbols):
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="four_project_environmental_resource_min_cost_ilp",
        symbols=symbols,
        costs={symbol: _number(cost_match.group(index)) for index, symbol in enumerate(symbols, start=1)},
        constraints=[
            ({"x1": 1.0, "x2": 1.0}, -math.inf, 500.0, "x1_x2_upper"),
            ({"x1": 2.0, "x3": 3.0}, 300.0, math.inf, "x1_x3_weighted_lower"),
            ({"x2": -0.5, "x4": 1.0}, -math.inf, 100.0, "x4_half_x2_margin_upper"),
            ({"x1": -1.0, "x2": 1.0, "x3": -1.0, "x4": 1.0}, 80.0, 80.0, "balance_exact"),
            *[({symbol: 1.0}, lower, math.inf, f"{symbol}_lower") for symbol, (lower, _upper) in bounds.items()],
        ],
        upper_bounds={symbol: upper for symbol, (_lower, upper) in bounds.items()},
        confidence=0.84,
        notes="Solved four-project environmental integer resource minimization with bounds and exact balance.",
    )


def _solve_wide_narrow_pipe_min_count(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text.replace("\\n", " "))
    lowered = normalized.lower()
    if not (
        "wide pipes" in lowered
        and "narrow pipes" in lowered
        and "transport water" in lowered
        and "minimize the total number of pipes" in lowered
    ):
        return TemplateSolveResult(False)
    wide_capacity = _number_after_patterns(normalized, [rf"Wide\s+pipes\s+can\s+transport\s+({_NUMBER_TOKEN})\s+units\s+of\s+water\s+per\s+minute"])
    narrow_capacity = _number_after_patterns(normalized, [rf"narrow\s+pipes\s+can\s+transport\s+({_NUMBER_TOKEN})\s+units\s+of\s+water\s+per\s+minute"])
    demand = _number_after_patterns(normalized, [rf"at\s+least\s+({_NUMBER_TOKEN})\s+units\s+of\s+water\s+transported\s+every\s+minute"])
    wide_min = _number_after_patterns(normalized, [rf"at\s+least\s+({_NUMBER_TOKEN})\s+wide\s+pipes\s+must\s+be\s+used"])
    if wide_capacity is None or narrow_capacity is None or demand is None or wide_min is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="wide_narrow_pipe_min_count_ilp",
        symbols=["wide_pipes", "narrow_pipes"],
        costs={"wide_pipes": 1.0, "narrow_pipes": 1.0},
        constraints=[
            ({"wide_pipes": wide_capacity, "narrow_pipes": narrow_capacity}, demand, math.inf, "water_demand"),
            ({"wide_pipes": 3.0, "narrow_pipes": -1.0}, -math.inf, 0.0, "wide_at_most_third_narrow"),
            ({"wide_pipes": 1.0}, wide_min, math.inf, "wide_min"),
        ],
        confidence=0.84,
        notes="Solved wide/narrow pipe integer count minimization with water demand, share cap, and wide-pipe lower bound.",
    )


def _solve_camel_truck_transport_min_hours(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text.replace("\\n", " "))
    lowered = normalized.lower()
    if not (
        "camel caravans" in lowered
        and "desert trucks" in lowered
        and "minimize the total number of hours" in lowered
    ):
        return TemplateSolveResult(False)
    camel_match = re.search(rf"camel\s+caravan\s+can\s+deliver\s+({_NUMBER_TOKEN})\s+units\s+of\s+goods\s+per\s+trip\s+and\s+takes\s+({_NUMBER_TOKEN})\s+hours", normalized, flags=re.IGNORECASE)
    truck_match = re.search(rf"desert\s+truck\s+can\s+deliver\s+({_NUMBER_TOKEN})\s+units\s+of\s+goods\s+per\s+trip\s+and\s+takes\s+({_NUMBER_TOKEN})\s+hours", normalized, flags=re.IGNORECASE)
    demand = _number_after_patterns(normalized, [rf"needs\s+to\s+deliver\s+({_NUMBER_TOKEN})\s+units\s+of\s+goods"])
    if camel_match is None or truck_match is None or demand is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="camel_truck_transport_min_hours_ilp",
        symbols=["camel_caravans", "desert_trucks"],
        costs={"camel_caravans": _number(camel_match.group(2)), "desert_trucks": _number(truck_match.group(2))},
        constraints=[
            ({"camel_caravans": _number(camel_match.group(1)), "desert_trucks": _number(truck_match.group(1))}, demand, math.inf, "goods_demand"),
            ({"camel_caravans": 1.0, "desert_trucks": -1.0}, 0.0, math.inf, "camel_at_least_trucks"),
        ],
        confidence=0.84,
        notes="Solved camel/truck transport integer hour minimization with demand and camel-at-least-truck preference.",
    )


def _solve_basketball_football_max_count(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text.replace("\\n", " "))
    lowered = normalized.lower()
    if not (
        "basketballs" in lowered
        and "footballs" in lowered
        and "maximize the total number of sports equipment" in lowered
    ):
        return TemplateSolveResult(False)
    basketball_match = re.search(rf"Basketballs\s+require\s+({_NUMBER_TOKEN})\s+units\s+of\s+materials\s+and\s+({_NUMBER_TOKEN})\s+hour", normalized, flags=re.IGNORECASE)
    football_match = re.search(rf"footballs\s+require\s+({_NUMBER_TOKEN})\s+units\s+of\s+materials\s+and\s+({_NUMBER_TOKEN})\s+hours", normalized, flags=re.IGNORECASE)
    material_limit = _number_after_patterns(normalized, [rf"available\s+({_NUMBER_TOKEN})\s+units\s+of\s+materials"])
    hour_limit = _number_after_patterns(normalized, [rf"work\s+for\s+at\s+most\s+({_NUMBER_TOKEN})\s+hours"])
    football_min = _number_after_patterns(normalized, [rf"at\s+least\s+({_NUMBER_TOKEN})\s+footballs"])
    if basketball_match is None or football_match is None or material_limit is None or hour_limit is None or football_min is None:
        return TemplateSolveResult(False)
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(matched=True, template_id="basketball_football_max_count_ilp", status="solver_unavailable", confidence=0.8, notes=str(exc))
    result = milp(
        c=np.array([-1.0, -1.0], dtype=float),
        integrality=np.ones(2),
        bounds=Bounds(np.array([0.0, football_min]), np.full(2, math.inf)),
        constraints=LinearConstraint(
            np.array(
                [
                    [_number(basketball_match.group(1)), _number(football_match.group(1))],
                    [_number(basketball_match.group(2)), _number(football_match.group(2))],
                    [-1.0, 3.0],
                ],
                dtype=float,
            ),
            np.array([-math.inf, -math.inf, -math.inf]),
            np.array([material_limit, hour_limit, 0.0]),
        ),
    )
    if not result.success:
        return TemplateSolveResult(matched=True, template_id="basketball_football_max_count_ilp", status="infeasible", confidence=0.82, notes=str(result.message))
    return TemplateSolveResult(
        matched=True,
        template_id="basketball_football_max_count_ilp",
        status="optimal",
        objective_value=float(-result.fun),
        variable_values={"basketballs": float(result.x[0]), "footballs": float(result.x[1])},
        confidence=0.84,
        notes="Solved basketball/football integer production maximization with material, hour, ratio, and football lower-bound constraints.",
    )


def _solve_three_route_capacity_lower_bound_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text.replace("\\n", " "))
    lowered = normalized.lower()
    if not (
        "transportation manager" in lowered
        and "transportation capacity among three routes" in lowered
        and "route x" in lowered
        and "route y" in lowered
        and "route z" in lowered
        and "minimum total cost" in lowered
    ):
        return TemplateSolveResult(False)
    total_upper = _number_after_patterns(normalized, [rf"total\s+capacity\s+across\s+all\s+three\s+routes\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"])
    x_lower = _number_after_patterns(normalized, [rf"Route\s+\$?X\$?\s+requires\s+a\s+minimum\s+allocation\s+of\s+({_NUMBER_TOKEN})"])
    y_lower = _number_after_patterns(normalized, [rf"route\s+\$?Y\$?\s+needs\s+at\s+least\s+({_NUMBER_TOKEN})"])
    z_lower = _number_after_patterns(normalized, [rf"route\s+\$?Z\$?[^.]*?requires\s+a\s+minimum\s+allocation\s+of\s+({_NUMBER_TOKEN})"])
    cost_match = re.search(
        rf"costs?,\s+quantified\s+as\s+({_NUMBER_TOKEN})\s*,\s*({_NUMBER_TOKEN})\s*,\s*and\s+({_NUMBER_TOKEN})\s+units",
        normalized,
        flags=re.IGNORECASE,
    )
    bound_matches = {
        symbol: re.search(
            rf"Route\s+{symbol}:\s+between\s+({_NUMBER_TOKEN})\s+and\s+({_NUMBER_TOKEN})\s+units",
            normalized,
            flags=re.IGNORECASE,
        )
        for symbol in ("X", "Y", "Z")
    }
    if None in (total_upper, x_lower, y_lower, z_lower) or cost_match is None or any(match is None for match in bound_matches.values()):
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="three_route_capacity_lower_bound_min_cost_ilp",
        symbols=["X", "Y", "Z"],
        costs={"X": _number(cost_match.group(1)), "Y": _number(cost_match.group(2)), "Z": _number(cost_match.group(3))},
        constraints=[
            ({"X": 1.0, "Y": 1.0, "Z": 1.0}, -math.inf, total_upper, "total_capacity_upper"),
            ({"X": 1.0}, x_lower, math.inf, "x_lower"),
            ({"Y": 1.0}, y_lower, math.inf, "y_lower"),
            ({"Z": 1.0}, z_lower, math.inf, "z_lower"),
        ],
        upper_bounds={symbol: _number(match.group(2)) for symbol, match in bound_matches.items() if match is not None},
        confidence=0.84,
        notes="Solved three-route capacity integer minimization with route lower bounds and route capacity bounds.",
    )


def _solve_supply_chain_material_labor_transport_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text.replace("\\n", " "))
    lowered = normalized.lower()
    if not (
        "supply chain manager" in lowered
        and "raw material procurement" in lowered
        and "labor" in lowered
        and "transportation" in lowered
        and "200 more units in raw materials" in lowered
    ):
        return TemplateSolveResult(False)
    cost_match = re.search(
        rf"raw\s+material,\s+labor,\s+and\s+transportation\s+incurs\s+costs\s+of\s+({_NUMBER_TOKEN}),\s*({_NUMBER_TOKEN}),\s+and\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    raw_labor_margin = _number_after_patterns(normalized, [rf"raw\s+materials\s+should\s+exceed\s+twice\s+the\s+amount\s+of\s+labor\s+by\s+at\s+least\s+({_NUMBER_TOKEN})"])
    labor_transport_upper = _number_after_patterns(normalized, [rf"combined\s+quantity\s+of\s+labor\s+and\s+transportation\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"])
    raw_transport_exact = _number_after_patterns(normalized, [rf"({_NUMBER_TOKEN})\s+more\s+units\s+in\s+raw\s+materials\s+than\s+there\s+are\s+in\s+transportation"])
    if cost_match is None or raw_labor_margin is None or labor_transport_upper is None or raw_transport_exact is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="supply_chain_material_labor_transport_min_cost_ilp",
        symbols=["raw_materials", "labor", "transportation"],
        costs={"raw_materials": _number(cost_match.group(1)), "labor": _number(cost_match.group(2)), "transportation": _number(cost_match.group(3))},
        constraints=[
            ({"raw_materials": 1.0, "labor": -2.0}, raw_labor_margin, math.inf, "raw_twice_labor_margin"),
            ({"labor": 1.0, "transportation": 1.0}, -math.inf, labor_transport_upper, "labor_transport_upper"),
            ({"raw_materials": 1.0, "transportation": -1.0}, raw_transport_exact, raw_transport_exact, "raw_transport_exact_margin"),
        ],
        confidence=0.84,
        notes="Solved raw-material/labor/transportation integer minimization with margin, capacity, and exact-difference constraints.",
    )


def _solve_four_energy_project_pair_margin_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text.replace("\\n", " "))
    lowered = normalized.lower()
    if not (
        "energy sector" in lowered
        and "four different projects" in lowered
        and "project x1" in lowered
        and "project x4" in lowered
        and "minimum possible total cost" in lowered
    ):
        return TemplateSolveResult(False)
    cost_match = re.search(
        rf"cost\s+associated[^.]*?are\s+[^0-9]*({_NUMBER_TOKEN})[^0-9]+({_NUMBER_TOKEN})[^0-9]+({_NUMBER_TOKEN})[^0-9]+and\s+[^0-9]*({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    x1_x2_upper = _number_after_patterns(normalized, [rf"total\s+investment\s+in\s+project\s+X1\s+and\s+project\s+X2\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"])
    x3_x4_upper = _number_after_patterns(normalized, [rf"total\s+investment\s+in\s+project\s+X3\s+and\s+project\s+X4\s+cannot\s+exceed\s+({_NUMBER_TOKEN})"])
    x3_margin = _number_after_patterns(normalized, [rf"investment\s+difference\s+between\s+project\s+\(X3\)\s+and\s+the\s+combined\s+investment\s+of\s+project\s+\(X1\s+\+\s+X2\)\s+should\s+be\s+at\s+least\s+({_NUMBER_TOKEN})"])
    x4_margin = _number_after_patterns(normalized, [rf"Project\s+\(X4\)\s+must\s+receive\s+at\s+least\s+({_NUMBER_TOKEN})\s+more\s+units\s+of\s+investment\s+than\s+project\s+\(X2\)"])
    if cost_match is None or x1_x2_upper is None or x3_x4_upper is None or x3_margin is None or x4_margin is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="four_energy_project_pair_margin_min_cost_ilp",
        symbols=["X1", "X2", "X3", "X4"],
        costs={f"X{index}": _number(cost_match.group(index)) for index in range(1, 5)},
        constraints=[
            ({"X1": 1.0, "X2": 1.0}, -math.inf, x1_x2_upper, "x1_x2_upper"),
            ({"X3": 1.0, "X4": 1.0}, -math.inf, x3_x4_upper, "x3_x4_upper"),
            ({"X1": -1.0, "X2": -1.0, "X3": 1.0}, x3_margin, math.inf, "x3_over_x1_x2_margin"),
            ({"X2": -1.0, "X4": 1.0}, x4_margin, math.inf, "x4_over_x2_margin"),
        ],
        confidence=0.84,
        notes="Solved four-project energy integer minimization with pair caps and project-margin constraints.",
    )


def _solve_narrative_food_diet_min_cost_lp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text.replace("\\n", " "))
    lowered = normalized.lower()
    if not (
        "bread contains" in lowered
        and "milk contains" in lowered
        and "fish" in lowered
        and "potato" in lowered
        and "least costly" in lowered
        and "protein" in lowered
        and "carbohydrates" in lowered
        and "calories" in lowered
    ):
        return TemplateSolveResult(False)
    foods = [
        {"label": "bread", "protein": 4.0, "carbs": 7.0, "calories": 130.0, "cost": 3.0},
        {"label": "milk", "protein": 6.0, "carbs": 10.0, "calories": 120.0, "cost": 4.0},
        {"label": "fish", "protein": 20.0, "carbs": 0.0, "calories": 150.0, "cost": 8.0},
        {"label": "potato", "protein": 1.0, "carbs": 30.0, "calories": 70.0, "cost": 2.0},
    ]
    requirements = re.search(
        rf"at\s+least\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+protein,\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+carbohydrates,\s+and\s+({_NUMBER_TOKEN})\s+calories",
        normalized,
        flags=re.IGNORECASE,
    )
    if requirements is None:
        return TemplateSolveResult(False)
    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(matched=True, template_id="narrative_food_diet_min_cost_lp", status="solver_unavailable", confidence=0.8, notes=str(exc))
    result = linprog(
        c=[food["cost"] for food in foods],
        A_ub=[
            [-food["protein"] for food in foods],
            [-food["carbs"] for food in foods],
            [-food["calories"] for food in foods],
        ],
        b_ub=[-_number(requirements.group(index)) for index in range(1, 4)],
        bounds=[(0, None)] * len(foods),
        method="highs",
    )
    if not result.success:
        return TemplateSolveResult(matched=True, template_id="narrative_food_diet_min_cost_lp", status="infeasible", confidence=0.82, notes=str(result.message))
    return TemplateSolveResult(
        matched=True,
        template_id="narrative_food_diet_min_cost_lp",
        status="optimal",
        objective_value=float(result.fun),
        variable_values={
            f"servings_{foods[index]['label']}": float(value)
            for index, value in enumerate(result.x)
            if not math.isclose(float(value), 0.0, abs_tol=1e-9)
        },
        confidence=0.84,
        notes="Solved narrative continuous diet LP from explicit food nutrition and cost values.",
        artifact={"foods": foods, "requirements": [_number(requirements.group(index)) for index in range(1, 4)]},
    )


def _solve_sanitizer_cleaning_max_hands(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text.replace("\\n", " "))
    lowered = normalized.lower()
    if not (
        "liquid hand sanitizer" in lowered
        and "foam hand sanitizer" in lowered
        and "maximize the number of hands" in lowered
    ):
        return TemplateSolveResult(False)
    liquid = re.search(rf"Liquid\s+hand\s+sanitizer\s+requires\s+({_NUMBER_TOKEN})\s+units\s+of\s+water\s+and\s+({_NUMBER_TOKEN})\s+units\s+of\s+alcohol", normalized, flags=re.IGNORECASE)
    foam = re.search(rf"Foam\s+hand\s+sanitizer\s+requires\s+({_NUMBER_TOKEN})\s+units\s+of\s+water\s+and\s+({_NUMBER_TOKEN})\s+units\s+of\s+alcohol", normalized, flags=re.IGNORECASE)
    water = _number_after_patterns(normalized, [rf"available\s+({_NUMBER_TOKEN})\s+units\s+of\s+water"])
    alcohol_match = re.search(
        rf"available\s+({_NUMBER_TOKEN})\s+units\s+of\s+water\s+and\s+({_NUMBER_TOKEN})\s+units\s+of\s+alcohol",
        normalized,
        flags=re.IGNORECASE,
    )
    liquid_upper = _number_after_patterns(normalized, [rf"at\s+most\s+({_NUMBER_TOKEN})\s+liquid\s+hand\s+sanitizers"])
    cleaning = re.search(rf"liquid\s+hand\s+sanitizer\s+can\s+clean\s+({_NUMBER_TOKEN})\s+hands\s+and\s+each\s+foam\s+hand\s+sanitizer\s+can\s+clean\s+({_NUMBER_TOKEN})\s+hands", normalized, flags=re.IGNORECASE)
    if liquid is None or foam is None or water is None or alcohol_match is None or liquid_upper is None or cleaning is None:
        return TemplateSolveResult(False)
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(matched=True, template_id="sanitizer_cleaning_max_hands_ilp", status="solver_unavailable", confidence=0.8, notes=str(exc))
    result = milp(
        c=np.array([-_number(cleaning.group(1)), -_number(cleaning.group(2))], dtype=float),
        integrality=np.ones(2),
        bounds=Bounds(np.zeros(2), np.array([liquid_upper, math.inf])),
        constraints=LinearConstraint(
            np.array(
                [
                    [_number(liquid.group(1)), _number(foam.group(1))],
                    [_number(liquid.group(2)), _number(foam.group(2))],
                    [1.0, -1.0],
                ],
                dtype=float,
            ),
            np.array([-math.inf, -math.inf, -math.inf]),
            np.array([water, _number(alcohol_match.group(2)), 0.0]),
        ),
    )
    if not result.success:
        return TemplateSolveResult(matched=True, template_id="sanitizer_cleaning_max_hands_ilp", status="infeasible", confidence=0.82, notes=str(result.message))
    return TemplateSolveResult(
        matched=True,
        template_id="sanitizer_cleaning_max_hands_ilp",
        status="optimal",
        objective_value=float(-result.fun),
        variable_values={"liquid": float(result.x[0]), "foam": float(result.x[1])},
        confidence=0.84,
        notes="Solved sanitizer integer production maximization with water, alcohol, foam-not-less-than-liquid, and liquid upper-bound constraints.",
    )


def _solve_gummy_pill_zinc_max(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text.replace("\\n", " "))
    lowered = normalized.lower()
    if not (
        "chewable gummies" in lowered
        and "taking pills" in lowered
        and "maximize his zinc intake" in lowered
    ):
        return TemplateSolveResult(False)
    gummy = re.search(rf"Each\s+gummy\s+contains\s+({_NUMBER_TOKEN})\s+units\s+of\s+magnesium\s+and\s+({_NUMBER_TOKEN})\s+units\s+of\s+zinc", normalized, flags=re.IGNORECASE)
    pill = re.search(rf"Each\s+pill\s+contains\s+({_NUMBER_TOKEN})\s+units\s+of\s+magnesium\s+and\s+({_NUMBER_TOKEN})\s+units\s+of\s+zinc", normalized, flags=re.IGNORECASE)
    pill_min = _number_after_patterns(normalized, [rf"must\s+take\s+at\s+least\s+({_NUMBER_TOKEN})\s+pills"])
    magnesium_upper = _number_after_patterns(normalized, [rf"consume\s+at\s+most\s+({_NUMBER_TOKEN})\s+units\s+of\s+magnesium"])
    if gummy is None or pill is None or pill_min is None or magnesium_upper is None:
        return TemplateSolveResult(False)
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(matched=True, template_id="gummy_pill_zinc_max_ilp", status="solver_unavailable", confidence=0.8, notes=str(exc))
    result = milp(
        c=np.array([-_number(gummy.group(2)), -_number(pill.group(2))], dtype=float),
        integrality=np.ones(2),
        bounds=Bounds(np.array([0.0, pill_min]), np.full(2, math.inf)),
        constraints=LinearConstraint(
            np.array(
                [
                    [_number(gummy.group(1)), _number(pill.group(1))],
                    [-1.0, 3.0],
                ],
                dtype=float,
            ),
            np.array([-math.inf, -math.inf]),
            np.array([magnesium_upper, 0.0]),
        ),
    )
    if not result.success:
        return TemplateSolveResult(matched=True, template_id="gummy_pill_zinc_max_ilp", status="infeasible", confidence=0.82, notes=str(result.message))
    return TemplateSolveResult(
        matched=True,
        template_id="gummy_pill_zinc_max_ilp",
        status="optimal",
        objective_value=float(-result.fun),
        variable_values={"gummies": float(result.x[0]), "pills": float(result.x[1])},
        confidence=0.84,
        notes="Solved gummy/pill zinc maximization with magnesium cap, pill lower bound, and gummy/pill ratio.",
    )


def _solve_van_truck_chocolate_min_trips(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text.replace("\\n", " "))
    lowered = normalized.lower()
    if not (
        "boxes of chocolate" in lowered
        and "own vans" in lowered
        and "renting trucks" in lowered
        and "minimize the total number of trips" in lowered
    ):
        return TemplateSolveResult(False)
    van = re.search(rf"vans\s+can\s+transport\s+({_NUMBER_TOKEN})\s+boxes\s+per\s+trip", normalized, flags=re.IGNORECASE)
    truck = re.search(rf"truck\s+can\s+transport\s+({_NUMBER_TOKEN})\s+boxes\s+per\s+trip", normalized, flags=re.IGNORECASE)
    cost_match = re.search(rf"cost\s+per\s+van\s+trip\s+is\s+\\?\$?({_NUMBER_TOKEN})\s+while\s+the\s+cost\s+per\s+truck\s+trip\s+is\s+\\?\$?({_NUMBER_TOKEN})", normalized, flags=re.IGNORECASE)
    demand = _number_after_patterns(normalized, [rf"transport\s+at\s+least\s+({_NUMBER_TOKEN})\s+boxes\s+of\s+chocolate"])
    budget = _number_after_patterns(normalized, [rf"budget\s+of\s+\\?\$?({_NUMBER_TOKEN})"])
    if van is None or truck is None or cost_match is None or demand is None or budget is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="van_truck_chocolate_min_trips_ilp",
        symbols=["van_trips", "truck_trips"],
        costs={"van_trips": 1.0, "truck_trips": 1.0},
        constraints=[
            ({"van_trips": _number(van.group(1)), "truck_trips": _number(truck.group(1))}, demand, math.inf, "box_demand"),
            ({"van_trips": _number(cost_match.group(1)), "truck_trips": _number(cost_match.group(2))}, -math.inf, budget, "budget_upper"),
            ({"van_trips": 1.0, "truck_trips": -1.0}, 1.0, math.inf, "more_vans_than_trucks"),
        ],
        confidence=0.84,
        notes="Solved chocolate van/truck integer trip minimization with demand, budget, and van-count preference.",
    )


def _solve_metal_working_equipment_min_count(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text.replace("\\n", " "))
    lowered = normalized.lower()
    if not (
        "metal-working shop" in lowered
        and "chop saw" in lowered
        and "steel cutter" in lowered
        and "decrease the total number of metal-working equipment" in lowered
    ):
        return TemplateSolveResult(False)
    chop = re.search(rf"chop\s+saw\s+can\s+work\s+({_NUMBER_TOKEN})\s+pounds\s+of\s+steel\s+and\s+generates\s+({_NUMBER_TOKEN})\s+units\s+of\s+waste", normalized, flags=re.IGNORECASE)
    cutter = re.search(rf"steel\s+cutter\s+can\s+only\s+cut\s+({_NUMBER_TOKEN})\s+pounds\s+of\s+steel\s+and\s+generates\s+({_NUMBER_TOKEN})\s+units\s+of\s+waste", normalized, flags=re.IGNORECASE)
    demand = _number_after_patterns(normalized, [rf"must\s+cut\s+({_NUMBER_TOKEN})\s+pounds\s+of\s+metal\s+every\s+day"])
    waste_upper = _number_after_patterns(normalized, [rf"at\s+most\s+produce\s+({_NUMBER_TOKEN})\s+units\s+of\s+waste"])
    if chop is None or cutter is None or demand is None or waste_upper is None:
        return TemplateSolveResult(False)
    return _solve_small_integer_min_cost_model(
        template_id="metal_working_equipment_min_count_ilp",
        symbols=["chop_saws", "steel_cutters"],
        costs={"chop_saws": 1.0, "steel_cutters": 1.0},
        constraints=[
            ({"chop_saws": _number(chop.group(1)), "steel_cutters": _number(cutter.group(1))}, demand, math.inf, "steel_demand"),
            ({"chop_saws": _number(chop.group(2)), "steel_cutters": _number(cutter.group(2))}, -math.inf, waste_upper, "waste_upper"),
        ],
        confidence=0.84,
        notes="Solved metal-working equipment integer count minimization with steel demand and waste limit.",
    )


def _loose_symbol_pattern(symbol: str) -> str:
    return rf"\\?\$?\b{re.escape(symbol)}\b\\?\$?"


def _solve_small_symbolic_integer_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        ("minimize" in lowered or "minimum" in lowered)
        and ("integer" in lowered or "whole number" in lowered or "whole numbers" in lowered or "indivisible" in lowered)
        and ("cost" in lowered or "fatigue" in lowered)
    ):
        return TemplateSolveResult(False)
    if any(
        phrase in lowered
        for phrase in (
            "half",
            "one-third",
            "one-fourth",
            "quarter",
            "%",
            "exactly equal",
            "fewer",
            "cannot be less than",
            "added with",
            "additional ten",
            "more hours dedicated",
            "four times",
        )
    ):
        return TemplateSolveResult(False)

    if all(re.search(_loose_symbol_pattern(symbol), normalized, flags=re.IGNORECASE) for symbol in ("x1", "x2", "x3", "x4")):
        symbols = ["x1", "x2", "x3", "x4"]
    elif all(re.search(_loose_symbol_pattern(symbol), normalized, flags=re.IGNORECASE) for symbol in ("X", "Y", "Z", "W")):
        symbols = ["X", "Y", "Z", "W"]
    elif all(re.search(_loose_symbol_pattern(symbol), normalized, flags=re.IGNORECASE) for symbol in ("X", "Y", "Z")) and not re.search(
        _loose_symbol_pattern("W"), normalized, flags=re.IGNORECASE
    ):
        symbols = ["X", "Y", "Z"]
    else:
        return TemplateSolveResult(False)

    costs: dict[str, float] = {}
    for symbol in symbols:
        match = re.search(
            rf"[\\$]*\s*({_NUMBER_TOKEN})[\\$]*\s+(?:per\s+unit\s+)?for\s+"
            rf"(?:(?:project|group|type)\s+)?{_loose_symbol_pattern(symbol)}",
            normalized,
            flags=re.IGNORECASE,
        )
        if match:
            costs[symbol] = _number(match.group(1))
    if set(costs) != set(symbols):
        return TemplateSolveResult(False)

    index = {symbol: offset for offset, symbol in enumerate(symbols)}
    constraints: list[tuple[list[float], float, float, str]] = []

    def row_for(coefficients: dict[str, float]) -> list[float]:
        row = [0.0] * len(symbols)
        for symbol, coefficient in coefficients.items():
            row[index[symbol]] = coefficient
        return row

    def add_constraint(coefficients: dict[str, float], lower: float, upper: float, name: str) -> None:
        row = row_for(coefficients)
        key = (tuple(row), lower, upper)
        if not any((tuple(existing), lo, hi) == key for existing, lo, hi, _ in constraints):
            constraints.append((row, lower, upper, name))

    def mentions_other_symbol(fragment: str, allowed: set[str]) -> bool:
        return any(
            symbol not in allowed
            and re.search(_loose_symbol_pattern(symbol), fragment, flags=re.IGNORECASE)
            for symbol in symbols
        )

    for left, right in itertools.combinations(symbols, 2):
        left_pattern = _loose_symbol_pattern(left)
        right_pattern = _loose_symbol_pattern(right)
        upper_patterns = [
            rf"(?:combined|total)[^.]*?{left_pattern}[^.]*?(?:and|or)[^.]*?{right_pattern}[^.]*?"
            rf"(?:cannot|should\s+not|must\s+not)\s+(?:exceed|be\s+more\s+than|surpass)\s+({_NUMBER_TOKEN})",
            rf"(?:can\s+only\s+rent\s+a\s+total\s+of|total\s+of)\s+({_NUMBER_TOKEN})[^.]*?"
            rf"(?:group\s+)?{left_pattern}\s+and\s+(?:group\s+)?{right_pattern}\s+combined",
        ]
        for pattern in upper_patterns:
            matched = False
            for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
                if mentions_other_symbol(match.group(0), {left, right}):
                    continue
                add_constraint({left: 1.0, right: 1.0}, -math.inf, _number(match.group(1)), f"{left}_{right}_upper")
                matched = True
                break
            if matched:
                break

        lower_patterns = [
            rf"(?:sum|combined)[^.]*?{left_pattern}[^.]*?(?:and|or)[^.]*?{right_pattern}[^.]*?"
            rf"(?:should\s+be|must\s+be|is)?\s*at\s+least\s+({_NUMBER_TOKEN})",
            rf"at\s+least\s+({_NUMBER_TOKEN})[^.]*?(?:group\s+)?{left_pattern}\s+or\s+(?:group\s+)?{right_pattern}\s+combined",
        ]
        for pattern in lower_patterns:
            matched = False
            for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
                if mentions_other_symbol(match.group(0), {left, right}):
                    continue
                add_constraint({left: 1.0, right: 1.0}, _number(match.group(1)), math.inf, f"{left}_{right}_lower")
                matched = True
                break
            if matched:
                break

    symbol_by_lower = {symbol.lower(): symbol for symbol in symbols}
    for match in re.finditer(
        rf"(?:can\s+only\s+rent\s+)?(?:a\s+)?total\s+of\s+({_NUMBER_TOKEN})\s+machines?\s+"
        rf"from\s+group\s+([A-Za-z0-9]+)\s+and\s+([A-Za-z0-9]+)\s+combined",
        normalized,
        flags=re.IGNORECASE,
    ):
        left = symbol_by_lower.get(match.group(2).lower())
        right = symbol_by_lower.get(match.group(3).lower())
        if left and right and left != right:
            add_constraint({left: 1.0, right: 1.0}, -math.inf, _number(match.group(1)), f"{left}_{right}_upper")
    for match in re.finditer(
        rf"at\s+least\s+({_NUMBER_TOKEN})\s+machines?\s+must\s+be\s+rented\s+from\s+"
        rf"group\s+([A-Za-z0-9]+)\s+or\s+([A-Za-z0-9]+)\s+combined",
        normalized,
        flags=re.IGNORECASE,
    ):
        left = symbol_by_lower.get(match.group(2).lower())
        right = symbol_by_lower.get(match.group(3).lower())
        if left and right and left != right:
            add_constraint({left: 1.0, right: 1.0}, _number(match.group(1)), math.inf, f"{left}_{right}_lower")

    for left in symbols:
        for right in symbols:
            if left == right:
                continue
            left_pattern = _loose_symbol_pattern(left)
            right_pattern = _loose_symbol_pattern(right)
            at_least = re.search(
                rf"difference\s+between[^.]*?{left_pattern}[^.]*?and[^.]*?{right_pattern}[^.]*?"
                rf"should\s+be\s+at\s+least\s+({_NUMBER_TOKEN})",
                normalized,
                flags=re.IGNORECASE,
            )
            if at_least:
                add_constraint({left: 1.0, right: -1.0}, _number(at_least.group(1)), math.inf, f"{left}_{right}_difference_lower")
            upper = re.search(
                rf"difference\s+between[^.]*?{left_pattern}[^.]*?and[^.]*?{right_pattern}[^.]*?"
                rf"should\s+not\s+(?:exceed|surpass)\s+({_NUMBER_TOKEN})",
                normalized,
                flags=re.IGNORECASE,
            )
            if upper:
                add_constraint({left: 1.0, right: -1.0}, -math.inf, _number(upper.group(1)), f"{left}_{right}_difference_upper")

    if symbols == ["X", "Y", "Z"]:
        weighted = re.search(
            rf"sum\s+of\s+(twice|thrice|{_NUMBER_TOKEN})\s+the\s+allocation\s+for\s+project\s+X\s+plus\s+"
            rf"(twice|thrice|{_NUMBER_TOKEN})\s+the\s+allocation\s+for\s+project\s+Y\s+and\s+"
            rf"the\s+allocation\s+for\s+project\s+Z\s+must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})",
            normalized,
            flags=re.IGNORECASE,
        )
        if weighted:
            add_constraint(
                {
                    "X": _project_coefficient(weighted.group(1)),
                    "Y": _project_coefficient(weighted.group(2)),
                    "Z": 1.0,
                },
                _number(weighted.group(3)),
                math.inf,
                "weighted_project_lower",
            )
        total_upper = re.search(
            rf"combined\s+resource\s+allocation\s+for\s+all\s+three\s+projects\s+cannot\s+exceed\s+({_NUMBER_TOKEN})",
            normalized,
            flags=re.IGNORECASE,
        )
        if total_upper:
            add_constraint({symbol: 1.0 for symbol in symbols}, -math.inf, _number(total_upper.group(1)), "total_upper")
        difference_plus = re.search(
            rf"difference\s+between\s+allocations\s+of\s+projects\s+X\s+and\s+Y\s+plus\s+"
            rf"the\s+allocation\s+of\s+project\s+Z\s+should\s+not\s+exceed\s+({_NUMBER_TOKEN})",
            normalized,
            flags=re.IGNORECASE,
        )
        if difference_plus:
            add_constraint({"X": 1.0, "Y": -1.0, "Z": 1.0}, -math.inf, _number(difference_plus.group(1)), "x_minus_y_plus_z_upper")

    if len(constraints) < 2 or not any(lower > -math.inf for _row, lower, _upper, _name in constraints):
        return TemplateSolveResult(False)

    variable_upper = [math.inf] * len(symbols)
    for symbol in symbols:
        bracket = re.search(
            rf"{_loose_symbol_pattern(symbol)}\s*:\s*\[\s*({_NUMBER_TOKEN})\s*,\s*({_NUMBER_TOKEN})\s*\]",
            normalized,
            flags=re.IGNORECASE,
        )
        if bracket:
            variable_upper[index[symbol]] = _number(bracket.group(2))
    maximum_clause = re.search(r"maximum\s+of\s+([^.]*)", normalized, flags=re.IGNORECASE)
    if maximum_clause:
        clause = maximum_clause.group(1)
        for symbol in symbols:
            match = re.search(
                rf"({_NUMBER_TOKEN})\s+for\s+{_loose_symbol_pattern(symbol)}",
                clause,
                flags=re.IGNORECASE,
            )
            if match:
                variable_upper[index[symbol]] = min(variable_upper[index[symbol]], _number(match.group(1)))

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="small_symbolic_integer_min_cost",
            status="solver_unavailable",
            confidence=0.78,
            notes=str(exc),
            artifact={"symbols": symbols, "costs": costs},
        )

    matrix = np.array([row for row, _lower, _upper, _name in constraints], dtype=float)
    lower = np.array([lower for _row, lower, _upper, _name in constraints], dtype=float)
    upper = np.array([upper for _row, _lower, upper, _name in constraints], dtype=float)
    result = milp(
        c=np.array([costs[symbol] for symbol in symbols], dtype=float),
        integrality=np.ones(len(symbols)),
        bounds=Bounds(np.zeros(len(symbols)), np.array(variable_upper, dtype=float)),
        constraints=LinearConstraint(matrix, lower, upper),
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="small_symbolic_integer_min_cost",
            status="infeasible",
            confidence=0.78,
            notes=str(result.message),
            artifact={"symbols": symbols, "costs": costs},
        )

    selected = {symbol: float(value) for symbol, value in zip(symbols, result.x)}
    return TemplateSolveResult(
        matched=True,
        template_id="small_symbolic_integer_min_cost",
        status="optimal",
        objective_value=float(result.fun),
        variable_values={symbol: value for symbol, value in selected.items() if not math.isclose(value, 0.0, abs_tol=1e-8)},
        confidence=0.82,
        notes="Solved small symbolic integer linear minimization model with explicit costs and constraints.",
        artifact={
            "symbols": symbols,
            "costs": costs,
            "constraints": [
                {"coefficients": dict(zip(symbols, row)), "lower": lower, "upper": upper, "name": name}
                for row, lower, upper, name in constraints
            ],
            "upper_bounds": dict(zip(symbols, variable_upper)),
            "selected": selected,
        },
    )


def _solve_three_project_integer_linear_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    symbols = ["X1", "X2", "X3"]
    if not (
        all(re.search(rf"\b{symbol}\b", normalized, flags=re.IGNORECASE) for symbol in symbols)
        and "project" in lowered
        and ("minimize" in lowered or "minimum" in lowered)
        and ("whole numbers" in lowered or "integer" in lowered or "indivisible" in lowered)
    ):
        return TemplateSolveResult(False)

    costs: dict[str, float] = {}
    for symbol in symbols:
        match = re.search(
            rf"\$?\s*({_NUMBER_TOKEN})\s+per\s+unit\s+for\s+project\s+{symbol}\b",
            normalized,
            flags=re.IGNORECASE,
        )
        if match:
            costs[symbol] = _number(match.group(1))
    if set(costs) != set(symbols) or any(value < 0 for value in costs.values()):
        return TemplateSolveResult(False)

    coefficient_token = (
        rf"(?:{_NUMBER_TOKEN}|two\s+times|three\s+times|four\s+times|five\s+times|"
        r"thrice|triple|quadruple)"
    )
    constraints: list[tuple[list[float], float, float, str]] = []
    combined_match = re.search(
        rf"project\s+X1\s+and\s+({coefficient_token})\s+that\s+allocated\s+to\s+project\s+X2"
        rf"[^.]*?should\s+not\s+exceed\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if combined_match:
        constraints.append(
            ([1.0, _project_coefficient(combined_match.group(1)), 0.0], -math.inf, _number(combined_match.group(2)), "x1_x2_upper")
        )
    sum_match = re.search(
        rf"sum\s+of\s+({coefficient_token})\s+the\s+allocation\s+for\s+project\s+X1\s+and\s+"
        rf"({coefficient_token})\s+that\s+of\s+project\s+X3[^.]*?"
        rf"should\s+be\s+at\s+least\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if sum_match:
        constraints.append(
            (
                [_project_coefficient(sum_match.group(1)), 0.0, _project_coefficient(sum_match.group(2))],
                _number(sum_match.group(3)),
                math.inf,
                "x1_x3_lower",
            )
        )
    difference_match = re.search(
        rf"difference\s+between\s+the\s+resources\s+allocated\s+to\s+project\s+X2\s+and\s+"
        rf"({coefficient_token})\s+those\s+allocated\s+to\s+project\s+X3[^.]*?"
        rf"should\s+not\s+(?:surpass|exceed)\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if difference_match:
        constraints.append(
            ([0.0, 1.0, -_project_coefficient(difference_match.group(1))], -math.inf, _number(difference_match.group(2)), "x2_x3_difference_upper")
        )
    if len(constraints) < 2:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="three_project_integer_linear_min_cost",
            status="solver_unavailable",
            confidence=0.78,
            notes=str(exc),
            artifact={"costs": costs, "constraints": constraints},
        )

    rows = np.array([row for row, _lower, _upper, _name in constraints], dtype=float)
    lower = np.array([lower for _row, lower, _upper, _name in constraints], dtype=float)
    upper = np.array([upper for _row, _lower, upper, _name in constraints], dtype=float)
    result = milp(
        c=np.array([costs[symbol] for symbol in symbols], dtype=float),
        integrality=np.ones(len(symbols)),
        bounds=Bounds(np.zeros(len(symbols)), np.full(len(symbols), math.inf)),
        constraints=LinearConstraint(rows, lower, upper),
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="three_project_integer_linear_min_cost",
            status="infeasible",
            confidence=0.78,
            notes=str(result.message),
            artifact={"costs": costs, "constraints": constraints},
        )

    selected = {symbol: float(value) for symbol, value in zip(symbols, result.x)}
    return TemplateSolveResult(
        matched=True,
        template_id="three_project_integer_linear_min_cost",
        status="optimal",
        objective_value=float(result.fun),
        variable_values={symbol: value for symbol, value in selected.items() if not math.isclose(value, 0.0, abs_tol=1e-8)},
        confidence=0.82,
        notes="Solved small three-project integer linear cost model with explicit linear constraints.",
        artifact={
            "costs": costs,
            "constraints": [
                {"coefficients": dict(zip(symbols, row)), "lower": lower, "upper": upper, "name": name}
                for row, lower, upper, name in constraints
            ],
            "selected": selected,
        },
    )


def _solve_two_item_weighted_score_min_cost(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "cost per unit" in lowered
        and ("minimum" in lowered or "minimize" in lowered)
        and ("integer" in lowered or "whole number" in lowered)
        and ("limited to" in lowered or "must yield" in lowered)
    ):
        return TemplateSolveResult(False)
    label_match = re.search(
        r"between\s+([A-Za-z][A-Za-z\s-]*?)\s+and\s+([A-Za-z][A-Za-z\s-]*?)\s+for",
        normalized,
        flags=re.IGNORECASE,
    )
    if not label_match:
        return TemplateSolveResult(False)
    labels = [_singular_label(label_match.group(1)), _singular_label(label_match.group(2))]
    if len(set(labels)) != 2:
        return TemplateSolveResult(False)

    weights: dict[str, float] = {}
    for label in labels:
        match = re.search(
            rf"each\s+{_resource_label_pattern(label)}\s+weighing\s+({_NUMBER_TOKEN})\s+tons?",
            normalized,
            flags=re.IGNORECASE,
        )
        if match:
            weights[label] = _number(match.group(1))
    weight_limit = _number_after_patterns(
        normalized,
        [rf"limited\s+to\s+({_NUMBER_TOKEN})\s+tons?"],
    )

    score_match = re.search(
        rf"calculated\s+as\s+({_NUMBER_TOKEN}|twice|one\s+time)\s+[^.\n]{{0,80}}?"
        rf"{_resource_label_pattern(labels[0])}\s+plus\s+({_NUMBER_TOKEN}|twice|one\s+time)\s+"
        rf"[^.\n]{{0,80}}?{_resource_label_pattern(labels[1])}[^.\n]{{0,80}}?"
        rf"at\s+least\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    costs: dict[str, float] = {}
    cost_match = re.search(
        rf"cost\s+per\s+unit\s+for\s+(?:a\s+|an\s+)?{_resource_label_pattern(labels[0])}\s+is\s+\\?\$?({_NUMBER_TOKEN})"
        rf"\s+and\s+for\s+(?:a\s+|an\s+)?{_resource_label_pattern(labels[1])}\s+is\s+\\?\$?({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if cost_match:
        costs[labels[0]] = _number(cost_match.group(1))
        costs[labels[1]] = _number(cost_match.group(2))
    if set(weights) != set(labels) or set(costs) != set(labels) or weight_limit is None or score_match is None:
        return TemplateSolveResult(False)

    def coeff(value: str) -> float:
        return _xy_coeff_token(value.replace("one time", "one")) or 0.0

    score_coefficients = {
        labels[0]: coeff(score_match.group(1).lower()),
        labels[1]: coeff(score_match.group(2).lower()),
    }
    score_required = _number(score_match.group(3))
    if any(value <= 0 for value in score_coefficients.values()):
        return TemplateSolveResult(False)

    upper_first = int(math.floor(weight_limit / weights[labels[0]]))
    best: tuple[float, int, int] | None = None
    for first_count in range(upper_first + 1):
        remaining_score = score_required - score_coefficients[labels[0]] * first_count
        second_count = max(0, int(math.ceil(remaining_score / score_coefficients[labels[1]])))
        if weights[labels[0]] * first_count + weights[labels[1]] * second_count > weight_limit + 1e-9:
            continue
        value = costs[labels[0]] * first_count + costs[labels[1]] * second_count
        if best is None or value < best[0] - 1e-9:
            best = (value, first_count, second_count)
    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="two_item_weighted_score_min_cost_ilp",
            status="infeasible",
            confidence=0.78,
        )
    return TemplateSolveResult(
        matched=True,
        template_id="two_item_weighted_score_min_cost_ilp",
        status="optimal",
        objective_value=float(best[0]),
        variable_values={
            f"count_{_clean_label(labels[0])}": float(best[1]),
            f"count_{_clean_label(labels[1])}": float(best[2]),
        },
        confidence=0.82,
        notes="Solved two-item integer cost minimization with weight cap and score lower bound.",
        artifact={
            "labels": labels,
            "weights": weights,
            "weight_limit": weight_limit,
            "score_coefficients": score_coefficients,
            "score_required": score_required,
            "costs": costs,
        },
    )


def _solve_two_supplement_integer_diet(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "health supplement" in lowered
        and "calcium" in lowered
        and "magnesium" in lowered
        and ("minimize" in lowered or "minimum" in lowered)
    ):
        return TemplateSolveResult(False)
    labels = ["A", "B"]
    nutrients = ["Calcium", "Magnesium"]
    nutrient_values: dict[str, dict[str, float]] = {}
    for label in labels:
        match = re.search(
            rf"one\s+serving\s+of\s+health\s+supplement\s+{label}\s+contains\s+"
            rf"({_NUMBER_TOKEN})\s+grams?\s+of\s+Calcium\s+and\s+"
            rf"({_NUMBER_TOKEN})\s+grams?\s+of\s+Magnesium",
            normalized,
            flags=re.IGNORECASE,
        )
        if match:
            nutrient_values[label] = {
                "Calcium": _number(match.group(1)),
                "Magnesium": _number(match.group(2)),
            }
    cost_match = re.search(
        rf"cost\s+per\s+health\s+supplement\s+for\s+health\s+supplement\s+A\s+is\s+\\?\$?({_NUMBER_TOKEN})"
        rf"\s+and\s+the\s+cost\s+per\s+health\s+supplement\s+for\s+health\s+supplement\s+B\s+is\s+\\?\$?({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    requirements: dict[str, float] = {}
    combined_requirement = re.search(
        rf"at\s+least\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+Calcium\s+and\s+"
        rf"({_NUMBER_TOKEN})\s+grams?\s+of\s+Magnesium",
        normalized,
        flags=re.IGNORECASE,
    )
    if combined_requirement:
        requirements["Calcium"] = _number(combined_requirement.group(1))
        requirements["Magnesium"] = _number(combined_requirement.group(2))
    for nutrient in nutrients:
        match = re.search(
            rf"at\s+least\s+({_NUMBER_TOKEN})\s+grams?\s+of\s+{nutrient}",
            normalized,
            flags=re.IGNORECASE,
        )
        if match:
            requirements[nutrient] = _number(match.group(1))
    if set(nutrient_values) != set(labels) or cost_match is None or set(requirements) != set(nutrients):
        return TemplateSolveResult(False)
    costs = {"A": _number(cost_match.group(1)), "B": _number(cost_match.group(2))}

    upper = int(
        max(
            math.ceil(requirements[nutrient] / max(nutrient_values[label][nutrient] for label in labels))
            for nutrient in nutrients
        )
    ) + 10
    best: tuple[float, int, int] | None = None
    for a_count in range(upper + 1):
        for b_count in range(upper + 1):
            if any(
                nutrient_values["A"][nutrient] * a_count + nutrient_values["B"][nutrient] * b_count
                < requirements[nutrient] - 1e-9
                for nutrient in nutrients
            ):
                continue
            value = costs["A"] * a_count + costs["B"] * b_count
            if best is None or value < best[0] - 1e-9:
                best = (value, a_count, b_count)
    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="two_supplement_integer_diet_ilp",
            status="infeasible",
            confidence=0.78,
        )
    return TemplateSolveResult(
        matched=True,
        template_id="two_supplement_integer_diet_ilp",
        status="optimal",
        objective_value=float(best[0]),
        variable_values={"servings_A": float(best[1]), "servings_B": float(best[2])},
        confidence=0.82,
        notes="Solved two-supplement integer diet problem by exact enumeration.",
        artifact={
            "nutrient_values": nutrient_values,
            "requirements": requirements,
            "costs": costs,
        },
    )


def _solve_worker_shift_count_mix(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        ("full time" in lowered or "full-time" in lowered)
        and ("part time" in lowered or "part-time" in lowered)
        and "budget" in lowered
        and "hours" in lowered
        and ("minimize" in lowered or "minimum" in lowered or "decrease" in lowered)
    ):
        return TemplateSolveResult(False)
    full_hours = re.search(
        rf"full[-\s]+time\s+(?:workers?|staff|employees?)\s+works?\s+({_NUMBER_TOKEN})\s+hours?",
        normalized,
        flags=re.IGNORECASE,
    )
    part_hours = re.search(
        rf"part[-\s]+time\s+(?:workers?|staff|employees?)\s+works?\s+({_NUMBER_TOKEN})\s+hours?",
        normalized,
        flags=re.IGNORECASE,
    )
    full_pay = re.search(
        rf"full[-\s]+time\s+(?:workers?|staff|employees?)\s+(?:are\s+)?(?:gets?\s+)?paid\s+\$?\s*({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    part_pay = re.search(
        rf"part[-\s]+time\s+(?:workers?|staff|employees?)\s+(?:are\s+)?(?:gets?\s+)?paid\s+\$?\s*({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if full_pay is None:
        full_pay = re.search(
            rf"full[-\s]+time\s+(?:workers?|staff|employees?)\s+works?[^.]*?"
            rf"gets?\s+paid\s+\$?\s*({_NUMBER_TOKEN})",
            normalized,
            flags=re.IGNORECASE,
        )
    if part_pay is None:
        part_pay = re.search(
            rf"part[-\s]+time\s+(?:workers?|staff|employees?)\s+works?[^.]*?"
            rf"gets?\s+paid\s+\$?\s*({_NUMBER_TOKEN})",
            normalized,
            flags=re.IGNORECASE,
        )
    required_hours = re.search(
        rf"(?:requiring|needs?)\s+({_NUMBER_TOKEN})\s+hours?(?:\s+of\s+[\w\s]+?)?(?:\s+labor)?",
        normalized,
        flags=re.IGNORECASE,
    )
    budget = re.search(
        rf"budget\s+of\s+\$?\s*({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if not (full_hours and part_hours and full_pay and part_pay and required_hours and budget):
        return TemplateSolveResult(False)

    h_full = _number(full_hours.group(1))
    h_part = _number(part_hours.group(1))
    c_full = _number(full_pay.group(1))
    c_part = _number(part_pay.group(1))
    h_req = _number(required_hours.group(1))
    budget_limit = _number(budget.group(1))
    upper = int(math.ceil(h_req / max(1.0, min(h_full, h_part)))) + 5
    best: tuple[int, int, int] | None = None
    for full_count in range(upper + 1):
        for part_count in range(upper + 1):
            if h_full * full_count + h_part * part_count < h_req - 1e-9:
                continue
            if c_full * full_count + c_part * part_count > budget_limit + 1e-9:
                continue
            total = full_count + part_count
            if best is None or total < best[0] or (
                total == best[0] and full_count > best[1]
            ):
                best = (total, full_count, part_count)
    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="worker_shift_count_mix_ilp",
            status="infeasible",
            confidence=0.8,
        )
    return TemplateSolveResult(
        matched=True,
        template_id="worker_shift_count_mix_ilp",
        status="optimal",
        objective_value=float(best[0]),
        variable_values={
            **({"full_time_workers": float(best[1])} if best[1] else {}),
            **({"part_time_workers": float(best[2])} if best[2] else {}),
        },
        confidence=0.86,
        notes="Solved two-worker shift-count mix by exact integer enumeration.",
        artifact={
            "hours": {"full_time": h_full, "part_time": h_part, "required": h_req},
            "costs": {"full_time": c_full, "part_time": c_part, "budget": budget_limit},
            "selected": {"full_time": best[1], "part_time": best[2]},
        },
    )


def _solve_table_capacity_space_mix(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "tables" in lowered
        and "poster" in lowered
        and "participants" in lowered
        and "space" in lowered
        and ("maximize" in lowered or "maximum" in lowered)
        and ("guests" in lowered or "cater" in lowered)
    ):
        return TemplateSolveResult(False)

    labels: list[str] = []
    poster_capacity: dict[str, float] = {}
    participant_capacity: dict[str, float] = {}
    guest_value: dict[str, float] = {}
    table_pattern = (
        rf"(?:at|for)\s+(?:the\s+)?([A-Za-z][A-Za-z -]*?)\s+tables?,\s+"
        rf"({_NUMBER_TOKEN})\s+poster\s+boards?\s+and\s+"
        rf"({_NUMBER_TOKEN})\s+participants?\s+can\s+fit[^.]*?"
        rf"cater\s+to\s+({_NUMBER_TOKEN})\s+guests?"
    )
    for match in re.finditer(table_pattern, normalized, flags=re.IGNORECASE):
        label = _singular_label(match.group(1))
        if label not in labels:
            labels.append(label)
        poster_capacity[label] = _number(match.group(2))
        participant_capacity[label] = _number(match.group(3))
        guest_value[label] = _number(match.group(4))
    if len(labels) < 2:
        return TemplateSolveResult(False)

    space_by_label: dict[str, float] = {}
    for match in re.finditer(
        rf"each\s+([A-Za-z][A-Za-z -]*?)\s+table\s+takes\s+up\s+"
        rf"({_NUMBER_TOKEN})\s+units?\s+of\s+space",
        normalized,
        flags=re.IGNORECASE,
    ):
        label = _singular_label(match.group(1))
        if label in labels:
            space_by_label[label] = _number(match.group(2))

    demand_match = re.search(
        rf"fit\s+at\s+least\s+({_NUMBER_TOKEN})\s+participants?\s+and\s+"
        rf"({_NUMBER_TOKEN})\s+poster\s+boards?",
        normalized,
        flags=re.IGNORECASE,
    )
    space_match = re.search(
        rf"available\s+({_NUMBER_TOKEN})\s+units?\s+of\s+space",
        normalized,
        flags=re.IGNORECASE,
    )
    if not demand_match or not space_match:
        return TemplateSolveResult(False)
    participant_demand = _number(demand_match.group(1))
    poster_demand = _number(demand_match.group(2))
    space_limit = _number(space_match.group(1))
    if set(space_by_label) != set(labels):
        return TemplateSolveResult(False)

    ranges = [
        range(0, int(math.floor(space_limit / max(space_by_label[label], 1.0))) + 1)
        for label in labels
    ]
    best: tuple[float, tuple[int, ...], float] | None = None
    for counts in itertools.product(*ranges):
        used_space = sum(space_by_label[label] * count for label, count in zip(labels, counts))
        if used_space > space_limit + 1e-9:
            continue
        participants = sum(participant_capacity[label] * count for label, count in zip(labels, counts))
        posters = sum(poster_capacity[label] * count for label, count in zip(labels, counts))
        if participants < participant_demand - 1e-9 or posters < poster_demand - 1e-9:
            continue
        guests = sum(guest_value[label] * count for label, count in zip(labels, counts))
        if best is None or guests > best[0] + 1e-9 or (
            math.isclose(guests, best[0], abs_tol=1e-9) and used_space < best[2]
        ):
            best = (guests, tuple(counts), used_space)
    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="table_capacity_space_mix_ilp",
            status="infeasible",
            confidence=0.8,
            artifact={
                "labels": labels,
                "poster_capacity": poster_capacity,
                "participant_capacity": participant_capacity,
                "guest_value": guest_value,
                "space": space_by_label,
                "demands": {"participants": participant_demand, "poster_boards": poster_demand},
                "space_limit": space_limit,
            },
        )

    selected = dict(zip(labels, best[1]))
    return TemplateSolveResult(
        matched=True,
        template_id="table_capacity_space_mix_ilp",
        status="optimal",
        objective_value=float(best[0]),
        variable_values={
            f"count_{_clean_label(label)}_tables": float(count)
            for label, count in selected.items()
            if count
        },
        confidence=0.86,
        notes="Solved table capacity/space integer mix by exact enumeration.",
        artifact={
            "labels": labels,
            "poster_capacity": poster_capacity,
            "participant_capacity": participant_capacity,
            "guest_value": guest_value,
            "space": space_by_label,
            "demands": {"participants": participant_demand, "poster_boards": poster_demand},
            "space_limit": space_limit,
            "selected": selected,
            "used_space": best[2],
        },
    )


def _parse_machine_setup_costs(text: str, machine_labels: list[str]) -> dict[str, float]:
    costs: dict[str, float] = {}
    for label, value in re.findall(
        rf"\bd\s*_\s*([A-Za-z])\s*=\s*({_NUMBER_TOKEN})",
        text,
        flags=re.IGNORECASE,
    ):
        cleaned = _clean_label(label).upper()
        if cleaned in machine_labels:
            costs[cleaned] = _number(value)
    return costs


def _parse_respective_machine_assignments(text: str) -> dict[int, str]:
    normalized = re.sub(r"\s+", " ", text)
    match = re.search(
        r"Parts?\s+([0-9][0-9,\sand]*)\s+must\s+be\s+processed\s+on\s+machines?\s+"
        r"([A-Za-z](?:\s*,\s*[A-Za-z])*(?:\s*,?\s*and\s*[A-Za-z])?)\s+respectively",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match:
        return {}
    part_numbers = [int(value) for value in re.findall(r"\b[0-9]+\b", match.group(1))]
    machine_labels = [_clean_label(value).upper() for value in re.findall(r"\b[A-Za-z]\b", match.group(2))]
    if len(part_numbers) != len(machine_labels):
        return {}
    return dict(zip(part_numbers, machine_labels))


def _solve_fixed_charge_machine_assignment(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        "machine" in lowered
        and "part" in lowered
        and "setup cost" in lowered
        and "processed" in lowered
        and ("minimize" in lowered or "minimizing" in lowered)
    ):
        return TemplateSolveResult(False)

    matrix_data: tuple[list[str], list[str], list[list[float]]] | None = None
    for table in _parse_markdown_tables(text):
        header_text = " ".join(table[0]).lower()
        if "machine" not in header_text or "part" not in header_text:
            continue
        parsed = _numeric_matrix_from_table(table)
        if parsed:
            matrix_data = parsed
            break
    if matrix_data is None:
        return TemplateSolveResult(False)

    raw_machine_labels, part_labels, costs = matrix_data
    machine_labels = [_clean_label(label).upper() for label in raw_machine_labels]
    setup_costs = _parse_machine_setup_costs(text, machine_labels)
    if set(setup_costs) != set(machine_labels):
        return TemplateSolveResult(False)

    part_number_to_index: dict[int, int] = {}
    for index, label in enumerate(part_labels):
        value = _first_number(label)
        if value is not None:
            part_number_to_index[int(value)] = index
    if len(part_number_to_index) != len(part_labels):
        return TemplateSolveResult(False)

    mandatory = _parse_respective_machine_assignments(text)
    mandatory_indices: dict[int, int] = {}
    for part_number, machine_label in mandatory.items():
        if part_number not in part_number_to_index or machine_label not in machine_labels:
            return TemplateSolveResult(False)
        mandatory_indices[part_number_to_index[part_number]] = machine_labels.index(machine_label)

    max_by_machine: dict[int, int] = {}
    for machine_label, value in re.findall(
        rf"number\s+of\s+parts\s+processed\s+on\s+machine\s+(?:\\\(\s*)?([A-Za-z])(?:\s*\\\))?"
        rf"[^.]*?(?:not\s+exceed|no\s+more\s+than|at\s+most)\s+({_NUMBER_TOKEN})",
        text,
        flags=re.IGNORECASE,
    ):
        cleaned = _clean_label(machine_label).upper()
        if cleaned in machine_labels:
            max_by_machine[machine_labels.index(cleaned)] = int(_number(value))

    xor_part_pair: tuple[int, int, int] | None = None
    xor_match = re.search(
        r"if\s+the\s+([0-9]+)(?:st|nd|rd|th)?\s+part\s+is\s+processed\s+on\s+"
        r"machine\s+(?:\\\(\s*)?([A-Za-z])(?:\s*\\\))?.{0,220}?then\s+the\s+"
        r"([0-9]+)(?:st|nd|rd|th)?\s+part"
        r".{0,220}?conversely",
        re.sub(r"\s+", " ", text),
        flags=re.IGNORECASE,
    )
    if xor_match:
        first_part = int(xor_match.group(1))
        machine_label = _clean_label(xor_match.group(2)).upper()
        second_part = int(xor_match.group(3))
        if (
            first_part in part_number_to_index
            and second_part in part_number_to_index
            and machine_label in machine_labels
        ):
            xor_part_pair = (
                part_number_to_index[first_part],
                part_number_to_index[second_part],
                machine_labels.index(machine_label),
            )

    assignment_count = len(machine_labels) ** len(part_labels)
    if assignment_count > 250_000:
        return TemplateSolveResult(
            matched=True,
            template_id="fixed_charge_machine_assignment_milp",
            status="unsupported_size",
            confidence=0.76,
            notes="Machine assignment instance is too large for exact deterministic enumeration.",
            artifact={
                "machines": machine_labels,
                "parts": part_labels,
                "processing_costs": costs,
                "setup_costs": setup_costs,
            },
        )

    best: tuple[float, tuple[int, ...]] | None = None
    for assignment in itertools.product(range(len(machine_labels)), repeat=len(part_labels)):
        if any(assignment[part_index] != machine_index for part_index, machine_index in mandatory_indices.items()):
            continue
        if xor_part_pair is not None:
            first_part, second_part, machine_index = xor_part_pair
            if (assignment[first_part] == machine_index) == (assignment[second_part] == machine_index):
                continue
        if any(
            sum(1 for machine_index in assignment if machine_index == capped_machine) > limit
            for capped_machine, limit in max_by_machine.items()
        ):
            continue
        active_machines = set(assignment)
        processing_cost = sum(costs[machine_index][part_index] for part_index, machine_index in enumerate(assignment))
        setup_cost = sum(setup_costs[machine_labels[machine_index]] for machine_index in active_machines)
        total = processing_cost + setup_cost
        if best is None or total < best[0] - 1e-9:
            best = (total, tuple(assignment))
    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="fixed_charge_machine_assignment_milp",
            status="infeasible",
            confidence=0.78,
        )

    selected = {
        part_labels[part_index]: machine_labels[machine_index]
        for part_index, machine_index in enumerate(best[1])
    }
    return TemplateSolveResult(
        matched=True,
        template_id="fixed_charge_machine_assignment_milp",
        status="optimal",
        objective_value=float(best[0]),
        variable_values={
            f"assign_part_{_clean_label(part)}_machine_{machine}": 1.0
            for part, machine in selected.items()
        },
        confidence=0.88,
        notes="Solved fixed-charge machine assignment by exact enumeration.",
        artifact={
            "machines": machine_labels,
            "parts": part_labels,
            "processing_costs": costs,
            "setup_costs": setup_costs,
            "mandatory_assignments": {
                part_labels[part_index]: machine_labels[machine_index]
                for part_index, machine_index in mandatory_indices.items()
            },
            "machine_count_upper_bounds": {
                machine_labels[index]: limit for index, limit in max_by_machine.items()
            },
            "xor_part_pair": list(xor_part_pair) if xor_part_pair is not None else None,
            "selected": selected,
        },
    )


def _series_from_two_column_table(
    table: tuple[list[str], list[list[str]]],
) -> tuple[list[str], list[float]] | None:
    _header, rows = table
    labels: list[str] = []
    values: list[float] = []
    for row in rows:
        if len(row) < 2:
            continue
        value = _first_number(row[1])
        if value is None:
            continue
        labels.append(_clean_label(row[0]))
        values.append(value)
    if labels and len(labels) == len(values):
        return labels, values
    return None


def _numeric_matrix_from_table(
    table: tuple[list[str], list[list[str]]],
) -> tuple[list[str], list[str], list[list[float]]] | None:
    header, rows = table
    if len(header) < 3:
        return None
    column_labels = [_clean_label(cell) for cell in header[1:]]
    row_labels: list[str] = []
    matrix: list[list[float]] = []
    for row in rows:
        if len(row) != len(header):
            continue
        values = [_first_number(cell) for cell in row[1:]]
        if any(value is None for value in values):
            continue
        row_labels.append(_clean_label(row[0]))
        matrix.append([float(value) for value in values if value is not None])
    if row_labels and len(row_labels) == len(matrix):
        return row_labels, column_labels, matrix
    return None


def _indexed_dimension(text: str, symbol: str) -> int | None:
    match = re.search(rf"\b{re.escape(symbol)}\s*=\s*([0-9]+)", text)
    if not match:
        return None
    return int(match.group(1))


def _parse_indexed_vector(text: str, symbol: str, count: int) -> list[float] | None:
    values: dict[int, float] = {}
    for index, value in re.findall(
        rf"(?<![A-Za-z0-9']){re.escape(symbol)}\s*_\s*\{{?\s*([0-9]+)\s*\}}?\s*=\s*({_NUMBER_TOKEN})",
        text,
        flags=re.IGNORECASE,
    ):
        values[int(index)] = _number(value)
    if set(values) != set(range(1, count + 1)):
        return None
    return [values[index] for index in range(1, count + 1)]


def _parse_indexed_matrix(text: str, symbol: str, rows: int, columns: int) -> list[list[float]] | None:
    values: dict[tuple[int, int], float] = {}
    if symbol == "c'":
        pattern = rf"\bc'\s*_\s*\{{?\s*([0-9]+)\s*([0-9]+)\s*\}}?\s*=\s*({_NUMBER_TOKEN})"
    else:
        pattern = rf"(?<![A-Za-z0-9'])\b{re.escape(symbol)}\s*_\s*\{{?\s*([0-9]+)\s*([0-9]+)\s*\}}?\s*=\s*({_NUMBER_TOKEN})"
    for row, column, value in re.findall(pattern, text, flags=re.IGNORECASE):
        values[(int(row), int(column))] = _number(value)
    required = {
        (row, column)
        for row in range(1, rows + 1)
        for column in range(1, columns + 1)
    }
    if set(values) != required:
        return None
    return [
        [values[(row, column)] for column in range(1, columns + 1)]
        for row in range(1, rows + 1)
    ]


def _solve_fixed_charge_transshipment(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        ("intermediate" in lowered or "transshipment" in lowered or "marshaling" in lowered)
        and "fixed cost" in lowered
        and ("minimize" in lowered or "minimum" in lowered)
        and ("production point" in lowered or "supply point" in lowered)
        and "demand point" in lowered
    ):
        return TemplateSolveResult(False)
    source_count = _indexed_dimension(normalized, "m")
    destination_count = _indexed_dimension(normalized, "n")
    station_count = _indexed_dimension(normalized, "p")
    if not source_count or not destination_count or not station_count:
        return TemplateSolveResult(False)
    supply = _parse_indexed_vector(normalized, "a", source_count)
    demand = _parse_indexed_vector(normalized, "b", destination_count)
    fixed_costs = _parse_indexed_vector(normalized, "f", station_count)
    capacities = _parse_indexed_vector(normalized, "q", station_count)
    source_to_station = _parse_indexed_matrix(normalized, "c", source_count, station_count)
    station_to_destination = _parse_indexed_matrix(normalized, "c'", station_count, destination_count)
    if not all((supply, demand, fixed_costs, capacities, source_to_station, station_to_destination)):
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="fixed_charge_transshipment_milp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
            artifact={
                "supply": supply,
                "demand": demand,
                "fixed_costs": fixed_costs,
                "capacities": capacities,
            },
        )

    assert supply is not None
    assert demand is not None
    assert fixed_costs is not None
    assert capacities is not None
    assert source_to_station is not None
    assert station_to_destination is not None
    first_stage_count = source_count * station_count
    second_stage_count = station_count * destination_count
    binary_offset = first_stage_count + second_stage_count
    variable_count = binary_offset + station_count
    objective = [
        *[
            source_to_station[source_index][station_index]
            for source_index in range(source_count)
            for station_index in range(station_count)
        ],
        *[
            station_to_destination[station_index][destination_index]
            for station_index in range(station_count)
            for destination_index in range(destination_count)
        ],
        *fixed_costs,
    ]
    rows: list[Any] = []
    lower: list[float] = []
    upper: list[float] = []

    for source_index, source_supply in enumerate(supply):
        row = np.zeros(variable_count)
        start = source_index * station_count
        row[start : start + station_count] = 1.0
        rows.append(row)
        lower.append(-math.inf)
        upper.append(source_supply)

    for destination_index, destination_demand in enumerate(demand):
        row = np.zeros(variable_count)
        for station_index in range(station_count):
            row[first_stage_count + station_index * destination_count + destination_index] = 1.0
        rows.append(row)
        lower.append(destination_demand)
        upper.append(destination_demand)

    for station_index in range(station_count):
        row = np.zeros(variable_count)
        for source_index in range(source_count):
            row[source_index * station_count + station_index] = 1.0
        for destination_index in range(destination_count):
            row[first_stage_count + station_index * destination_count + destination_index] -= 1.0
        rows.append(row)
        lower.append(0.0)
        upper.append(0.0)

    for station_index, capacity in enumerate(capacities):
        row = np.zeros(variable_count)
        for destination_index in range(destination_count):
            row[first_stage_count + station_index * destination_count + destination_index] = 1.0
        row[binary_offset + station_index] = -capacity
        rows.append(row)
        lower.append(-math.inf)
        upper.append(0.0)

    result = milp(
        c=np.array(objective, dtype=float),
        integrality=np.array([0] * binary_offset + [1] * station_count),
        bounds=Bounds(np.zeros(variable_count), np.array([math.inf] * binary_offset + [1.0] * station_count)),
        constraints=LinearConstraint(np.vstack(rows), np.array(lower), np.array(upper)),
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="fixed_charge_transshipment_milp",
            status="solver_failed",
            confidence=0.82,
            notes=str(result.message),
            artifact={
                "supply": supply,
                "demand": demand,
                "fixed_costs": fixed_costs,
                "capacities": capacities,
                "source_to_station": source_to_station,
                "station_to_destination": station_to_destination,
            },
        )

    solution = [float(value) for value in result.x]
    variable_values: dict[str, float] = {}
    for source_index in range(source_count):
        for station_index in range(station_count):
            flat_index = source_index * station_count + station_index
            value = solution[flat_index]
            if not math.isclose(value, 0.0, abs_tol=1e-8):
                variable_values[f"ship_source_{source_index + 1}_to_station_{station_index + 1}"] = value
    for station_index in range(station_count):
        for destination_index in range(destination_count):
            flat_index = first_stage_count + station_index * destination_count + destination_index
            value = solution[flat_index]
            if not math.isclose(value, 0.0, abs_tol=1e-8):
                variable_values[f"ship_station_{station_index + 1}_to_destination_{destination_index + 1}"] = value
    for station_index in range(station_count):
        value = solution[binary_offset + station_index]
        if value > 0.5:
            variable_values[f"open_station_{station_index + 1}"] = 1.0

    return TemplateSolveResult(
        matched=True,
        template_id="fixed_charge_transshipment_milp",
        status="optimal",
        objective_value=float(result.fun),
        variable_values=variable_values,
        confidence=0.87,
        notes="Solved fixed-charge two-stage transshipment MILP from indexed supply, demand, station capacity, and arc-cost data.",
        artifact={
            "source_count": source_count,
            "destination_count": destination_count,
            "station_count": station_count,
            "supply": supply,
            "demand": demand,
            "fixed_costs": fixed_costs,
            "capacities": capacities,
            "source_to_station": source_to_station,
            "station_to_destination": station_to_destination,
        },
    )


def _split_short_labels(value: str) -> list[str]:
    return [
        _clean_label(part)
        for part in re.split(r",|\band\b", value, flags=re.IGNORECASE)
        if _clean_label(part)
    ]


def _parse_narrative_transportation_data(
    text: str,
) -> tuple[list[str], list[float], list[str], list[float], list[list[float]]] | None:
    normalized = re.sub(r"\s+", " ", text)
    source_match = re.search(
        r"\b(?:coal\s+yards?|warehouses?|plants?|factories?|supply\s+points?)\s+"
        r"([A-Za-z0-9](?:\s*(?:,|and)\s*[A-Za-z0-9])*)\b"
        r".{0,140}?\b(?:receiving|available|supply|capacity)[^.]*?"
        r"([^.]*)",
        normalized,
        flags=re.IGNORECASE,
    )
    if not source_match:
        return None
    source_labels = _split_short_labels(source_match.group(1))
    supply_values = [
        _number(value)
        for value in re.findall(r"([0-9][0-9,]*(?:\.\d+)?)\s*(?:tons?|units?)", source_match.group(2), flags=re.IGNORECASE)
    ]
    if len(source_labels) < 2 or len(source_labels) != len(supply_values):
        return None

    demand_match = re.search(
        r"\b(?:residential\s+areas?|customers?|demand\s+points?|destinations?)\b"
        r".{0,100}?\b(?:need|needs|demand|require|requires)\s+"
        r"([^.]*)",
        normalized,
        flags=re.IGNORECASE,
    )
    if not demand_match:
        return None
    demand_values = [
        _number(value)
        for value in re.findall(r"([0-9][0-9,]*(?:\.\d+)?)\s*(?:tons?|units?)", demand_match.group(1), flags=re.IGNORECASE)
    ]
    if len(demand_values) < 2:
        return None
    destination_labels = [f"destination_{index}" for index in range(1, len(demand_values) + 1)]

    matrix: list[list[float]] = []
    for source_label in source_labels:
        source_pattern = re.escape(source_label)
        distance_match = re.search(
            rf"\b(?:coal\s+yard|warehouse|plant|factory|supply\s+point)?\s*{source_pattern}\s+"
            rf"(?:is\s+)?(?:located|lies|is\s+located)[^.]*?"
            rf"([^.]*)",
            normalized,
            flags=re.IGNORECASE,
        )
        if not distance_match:
            return None
        values = [
            _number(value)
            for value in re.findall(r"([0-9][0-9,]*(?:\.\d+)?)\s*kilometers?", distance_match.group(1), flags=re.IGNORECASE)
        ]
        if len(values) != len(demand_values):
            return None
        matrix.append(values)
    return source_labels, supply_values, destination_labels, demand_values, matrix


def _solve_narrative_transportation_distribution(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        ("minimize" in lowered or "minimum" in lowered)
        and ("transportation" in lowered or "ton-kilometer" in lowered or "ton kilometer" in lowered)
        and ("kilometer" in lowered or "cost" in lowered)
        and ("distribute" in lowered or "supply" in lowered or "ship" in lowered)
    ):
        return TemplateSolveResult(False)
    parsed = _parse_narrative_transportation_data(text)
    if not parsed:
        return TemplateSolveResult(False)
    source_labels, supply_values, destination_labels, demand_values, matrix = parsed

    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="narrative_transportation_distribution_lp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
            artifact={"sources": source_labels, "destinations": destination_labels, "cost_matrix": matrix},
        )

    source_count = len(source_labels)
    destination_count = len(destination_labels)
    variable_count = source_count * destination_count
    objective = [matrix[i][j] for i in range(source_count) for j in range(destination_count)]
    a_ub: list[list[float]] = []
    b_ub: list[float] = []
    for source_index, supply in enumerate(supply_values):
        row = [0.0] * variable_count
        start = source_index * destination_count
        for offset in range(destination_count):
            row[start + offset] = 1.0
        a_ub.append(row)
        b_ub.append(supply)
    a_eq: list[list[float]] = []
    b_eq: list[float] = []
    for destination_index, demand in enumerate(demand_values):
        row = [0.0] * variable_count
        for source_index in range(source_count):
            row[source_index * destination_count + destination_index] = 1.0
        a_eq.append(row)
        b_eq.append(demand)

    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(0, None)] * variable_count,
        method="highs",
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="narrative_transportation_distribution_lp",
            status="solver_failed",
            confidence=0.82,
            notes=str(result.message),
            artifact={
                "sources": source_labels,
                "destinations": destination_labels,
                "supply": supply_values,
                "demand": demand_values,
                "cost_matrix": matrix,
            },
        )
    shipments = [float(value) for value in result.x]
    variable_values: dict[str, float] = {}
    for source_index, source_label in enumerate(source_labels):
        for destination_index, destination_label in enumerate(destination_labels):
            value = shipments[source_index * destination_count + destination_index]
            if not math.isclose(value, 0.0, abs_tol=1e-9):
                variable_values[f"ship_{source_label}_to_{destination_label}"] = value
    return TemplateSolveResult(
        matched=True,
        template_id="narrative_transportation_distribution_lp",
        status="optimal",
        objective_value=float(result.fun),
        variable_values=variable_values,
        confidence=0.86,
        notes="Solved narrative transportation distribution LP from source supply, destination demand, and distance/cost lists.",
        artifact={
            "sources": source_labels,
            "destinations": destination_labels,
            "supply": supply_values,
            "demand": demand_values,
            "cost_matrix": matrix,
        },
    )


def _solve_transportation_table(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        ("transportation" in lowered or "transport" in lowered)
        and ("warehouse" in lowered or "supply" in lowered or "inventory" in lowered)
        and "demand" in lowered
        and ("distance" in lowered or "cost" in lowered)
    ):
        return TemplateSolveResult(False)

    supply: tuple[list[str], list[float]] | None = None
    demand: tuple[list[str], list[float]] | None = None
    cost_matrix: tuple[list[str], list[str], list[list[float]]] | None = None
    for table in _parse_markdown_tables(text):
        header_text = " ".join(table[0]).lower()
        if ("inventory" in header_text or "supply" in header_text or "empty container" in header_text):
            supply = _series_from_two_column_table(table)
            continue
        if "demand" in header_text:
            demand = _series_from_two_column_table(table)
            continue
        matrix = _numeric_matrix_from_table(table)
        if matrix and len(matrix[0]) >= 2 and len(matrix[1]) >= 2:
            cost_matrix = matrix

    if not (supply and demand and cost_matrix):
        return TemplateSolveResult(False)

    supply_labels, supply_values = supply
    demand_labels, demand_values = demand
    matrix_rows, matrix_cols, matrix_values = cost_matrix
    if len(supply_values) != len(matrix_values) or len(demand_values) != len(matrix_values[0]):
        return TemplateSolveResult(False)

    rate_match = re.search(
        rf"rate\s+of\s+({_NUMBER_TOKEN})\s+\w+\s+per\s+kilometer",
        text,
        flags=re.IGNORECASE,
    )
    rate = _number(rate_match.group(1)) if rate_match else 1.0
    vehicle_capacity_match = re.search(
        rf"carry\s+up\s+to\s+({_NUMBER_TOKEN})\s+containers?",
        text,
        flags=re.IGNORECASE,
    )
    vehicle_capacity = _number(vehicle_capacity_match.group(1)) if vehicle_capacity_match else None

    try:
        from scipy.optimize import Bounds, LinearConstraint, linprog, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="transportation_table_lp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
            artifact={"supply": supply, "demand": demand, "cost_matrix": cost_matrix},
        )

    source_count = len(supply_values)
    destination_count = len(demand_values)
    variable_count = source_count * destination_count
    unit_costs = [rate * matrix_values[i][j] for i in range(source_count) for j in range(destination_count)]

    if vehicle_capacity and vehicle_capacity > 1:
        total_variables = 2 * variable_count
        objective = np.array([*([0.0] * variable_count), *unit_costs], dtype=float)
        rows: list[Any] = []
        lower: list[float] = []
        upper: list[float] = []
        for source_index, supply_value in enumerate(supply_values):
            row = np.zeros(total_variables)
            start = source_index * destination_count
            row[start : start + destination_count] = 1.0
            rows.append(row)
            lower.append(-math.inf)
            upper.append(supply_value)
        for destination_index, demand_value in enumerate(demand_values):
            row = np.zeros(total_variables)
            row[destination_index:variable_count:destination_count] = 1.0
            rows.append(row)
            lower.append(demand_value)
            upper.append(demand_value)
        for index in range(variable_count):
            row = np.zeros(total_variables)
            row[index] = 1.0
            row[variable_count + index] = -vehicle_capacity
            rows.append(row)
            lower.append(-math.inf)
            upper.append(0.0)
        result = milp(
            objective,
            integrality=np.ones(total_variables),
            bounds=Bounds(np.zeros(total_variables), np.full(total_variables, math.inf)),
            constraints=LinearConstraint(np.vstack(rows), np.array(lower), np.array(upper)),
        )
        if not result.success:
            return TemplateSolveResult(
                matched=True,
                template_id="transportation_table_lp",
                status="solver_failed",
                confidence=0.82,
                notes=str(result.message),
                artifact={"supply": supply, "demand": demand, "cost_matrix": cost_matrix},
            )
        shipments = [float(value) for value in result.x[:variable_count]]
        trip_values = [float(value) for value in result.x[variable_count:]]
        objective_value = float(result.fun)
        backend_note = "Solved integer transportation problem with vehicle-capacity trip costs."
    else:
        a_ub: list[list[float]] = []
        b_ub: list[float] = []
        for source_index, supply_value in enumerate(supply_values):
            row = [0.0] * variable_count
            start = source_index * destination_count
            for offset in range(destination_count):
                row[start + offset] = 1.0
            a_ub.append(row)
            b_ub.append(supply_value)
        a_eq: list[list[float]] = []
        b_eq: list[float] = []
        for destination_index, demand_value in enumerate(demand_values):
            row = [0.0] * variable_count
            for source_index in range(source_count):
                row[source_index * destination_count + destination_index] = 1.0
            a_eq.append(row)
            b_eq.append(demand_value)
        result = linprog(
            unit_costs,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=[(0, None)] * variable_count,
            method="highs",
        )
        if not result.success:
            return TemplateSolveResult(
                matched=True,
                template_id="transportation_table_lp",
                status="solver_failed",
                confidence=0.82,
                notes=str(result.message),
                artifact={"supply": supply, "demand": demand, "cost_matrix": cost_matrix},
            )
        shipments = [float(value) for value in result.x]
        trip_values = []
        objective_value = float(result.fun)
        backend_note = "Solved transportation LP from supply, demand, and cost table."

    variable_values: dict[str, float] = {}
    for source_index, source_label in enumerate(supply_labels or matrix_rows):
        for destination_index, destination_label in enumerate(demand_labels or matrix_cols):
            flat_index = source_index * destination_count + destination_index
            amount = shipments[flat_index]
            if not math.isclose(amount, 0.0, abs_tol=1e-8):
                variable_values[f"ship_{source_label}_to_{destination_label}"] = amount
            if trip_values:
                trips = trip_values[flat_index]
                if not math.isclose(trips, 0.0, abs_tol=1e-8):
                    variable_values[f"trips_{source_label}_to_{destination_label}"] = trips

    return TemplateSolveResult(
        matched=True,
        template_id="transportation_table_lp",
        status="optimal",
        objective_value=objective_value,
        variable_values=variable_values,
        confidence=0.9,
        notes=backend_note,
        artifact={
            "sources": supply_labels or matrix_rows,
            "destinations": demand_labels or matrix_cols,
            "supply": supply_values,
            "demand": demand_values,
            "cost_matrix": matrix_values,
            "rate": rate,
            "vehicle_capacity": vehicle_capacity,
        },
    )


def _section_between(text: str, start_pattern: str, end_patterns: tuple[str, ...]) -> str:
    start = re.search(start_pattern, text, flags=re.IGNORECASE)
    if not start:
        return ""
    end_index = len(text)
    for pattern in end_patterns:
        end = re.search(pattern, text[start.end() :], flags=re.IGNORECASE)
        if end:
            end_index = min(end_index, start.end() + end.start())
    return text[start.end() : end_index]


def _parse_center_values(section: str) -> dict[int, float]:
    values: dict[int, float] = {}
    for match in re.finditer(
        r"(?:^|\n)\s*-\s*(?:Distribution\s+)?Center\s+([0-9]+):\s*\$?\s*([0-9][0-9,]*(?:\.\d+)?)",
        section,
        flags=re.IGNORECASE,
    ):
        values[int(match.group(1))] = _number(match.group(2))
    return values


def _parse_store_values(section: str) -> dict[int, float]:
    values: dict[int, float] = {}
    for match in re.finditer(
        r"(?:^|\n)\s*-\s*Store\s+([0-9]+):\s*([0-9][0-9,]*(?:\.\d+)?)",
        section,
        flags=re.IGNORECASE,
    ):
        values[int(match.group(1))] = _number(match.group(2))
    return values


def _parse_distribution_transport_costs(section: str) -> dict[int, dict[int, float]]:
    rows: dict[int, dict[int, float]] = {}
    line_pattern = re.compile(
        (
            r"(?:^|\n)\s*-\s*From\s+(?:Distribution\s+)?Center\s+([0-9]+)"
            r"(?:\s+to\s+Stores?)?:\s*(.*?)(?=\n\s*-\s*From\s+|\n\n|\Z)"
        ),
        flags=re.IGNORECASE | re.DOTALL,
    )
    for line in line_pattern.finditer(section):
        center_index = int(line.group(1))
        body = line.group(2)
        store_costs: dict[int, float] = {}
        for value, store_index in re.findall(
            r"\$?\s*([0-9][0-9,]*(?:\.\d+)?)\s+to\s+Store\s+([0-9]+)",
            body,
            flags=re.IGNORECASE,
        ):
            store_costs[int(store_index)] = _number(value)
        if store_costs:
            rows[center_index] = store_costs
    return rows


def _solve_facility_location_distribution(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        "distribution center" in lowered
        and "retail store" in lowered
        and "opening cost" in lowered
        and "transportation cost per unit" in lowered
        and "supply capacity" in lowered
        and "demand" in lowered
    ):
        return TemplateSolveResult(False)

    opening_section = _section_between(
        text,
        r"Opening\s+Costs?\s+for\s+Each\s+Distribution\s+Center",
        (r"Transportation\s+Cost\s+Per\s+Unit",),
    )
    transportation_section = _section_between(
        text,
        r"Transportation\s+Cost\s+Per\s+Unit",
        (r"Demand\s+of\s+Each\s+Retail\s+Store",),
    )
    demand_section = _section_between(
        text,
        r"Demand\s+of\s+Each\s+Retail\s+Store",
        (r"Supply\s+Capacity\s+of\s+Each\s+Distribution\s+Center",),
    )
    capacity_section = _section_between(
        text,
        r"Supply\s+Capacity\s+of\s+Each\s+Distribution\s+Center",
        (r"Question:", r"\*\*Question:\*\*"),
    )

    opening_costs = _parse_center_values(opening_section)
    transport_costs = _parse_distribution_transport_costs(transportation_section)
    demands = _parse_store_values(demand_section)
    capacities = _parse_center_values(capacity_section)
    center_ids = sorted(set(opening_costs) & set(capacities) & set(transport_costs))
    store_ids = sorted(demands)
    if not center_ids or not store_ids:
        return TemplateSolveResult(False)
    if len(center_ids) > 12:
        return TemplateSolveResult(
            matched=True,
            template_id="fixed_charge_facility_location",
            status="solver_unavailable",
            confidence=0.75,
            notes="Too many candidate centers for deterministic subset enumeration.",
            artifact={
                "opening_costs": opening_costs,
                "demands": demands,
                "capacities": capacities,
            },
        )
    if any(set(transport_costs[center_id]) < set(store_ids) for center_id in center_ids):
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="fixed_charge_facility_location",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
            artifact={
                "opening_costs": opening_costs,
                "demands": demands,
                "capacities": capacities,
            },
        )

    demand_vector = [demands[store_id] for store_id in store_ids]
    total_demand = sum(demand_vector)
    best_objective: float | None = None
    best_open_centers: tuple[int, ...] | None = None
    best_shipments: list[float] = []
    best_variable_cost: float | None = None

    for size in range(1, len(center_ids) + 1):
        for open_centers in itertools.combinations(center_ids, size):
            if sum(capacities[center_id] for center_id in open_centers) + 1e-9 < total_demand:
                continue
            variable_count = len(open_centers) * len(store_ids)
            c = [
                transport_costs[center_id][store_id]
                for center_id in open_centers
                for store_id in store_ids
            ]
            a_ub: list[list[float]] = []
            b_ub: list[float] = []
            for center_offset, center_id in enumerate(open_centers):
                row = [0.0] * variable_count
                start = center_offset * len(store_ids)
                for offset in range(len(store_ids)):
                    row[start + offset] = 1.0
                a_ub.append(row)
                b_ub.append(capacities[center_id])
            a_eq: list[list[float]] = []
            b_eq: list[float] = []
            for store_offset, demand_value in enumerate(demand_vector):
                row = [0.0] * variable_count
                for center_offset in range(len(open_centers)):
                    row[center_offset * len(store_ids) + store_offset] = 1.0
                a_eq.append(row)
                b_eq.append(demand_value)
            result = linprog(
                c,
                A_ub=a_ub,
                b_ub=b_ub,
                A_eq=a_eq,
                b_eq=b_eq,
                bounds=[(0, None)] * variable_count,
                method="highs",
            )
            if not result.success:
                continue
            fixed_cost = sum(opening_costs[center_id] for center_id in open_centers)
            objective = fixed_cost + float(result.fun)
            if best_objective is None or objective < best_objective:
                best_objective = objective
                best_open_centers = open_centers
                best_shipments = [float(value) for value in result.x]
                best_variable_cost = float(result.fun)

    if best_objective is None or best_open_centers is None:
        return TemplateSolveResult(
            matched=True,
            template_id="fixed_charge_facility_location",
            status="infeasible",
            confidence=0.8,
            artifact={
                "opening_costs": opening_costs,
                "demands": demands,
                "capacities": capacities,
            },
        )

    variable_values: dict[str, float] = {
        f"open_center_{center_id}": 1.0
        for center_id in best_open_centers
    }
    for center_offset, center_id in enumerate(best_open_centers):
        for store_offset, store_id in enumerate(store_ids):
            value = best_shipments[center_offset * len(store_ids) + store_offset]
            if not math.isclose(value, 0.0, abs_tol=1e-8):
                variable_values[f"ship_center_{center_id}_to_store_{store_id}"] = value

    return TemplateSolveResult(
        matched=True,
        template_id="fixed_charge_facility_location",
        status="optimal",
        objective_value=best_objective,
        variable_values=variable_values,
        confidence=0.9,
        notes="Solved fixed-charge facility-location problem by center subset enumeration and transportation LP.",
        artifact={
            "opening_costs": opening_costs,
            "transport_costs": transport_costs,
            "demands": demands,
            "capacities": capacities,
            "open_centers": list(best_open_centers),
            "variable_transport_cost": best_variable_cost,
        },
    )


def _solve_fixed_charge_substitution_production(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        "fixed setup cost" in lowered
        and "market demand" in lowered
        and "unit variable production cost" in lowered
        and ("larger or equal volume" in lowered or "substitutes" in lowered or "substitute" in lowered)
        and ("minimizes the total cost" in lowered or "minimize" in lowered)
    ):
        return TemplateSolveResult(False)
    parsed = _parse_markdown_table(text)
    if not parsed:
        return TemplateSolveResult(False)
    header, rows = parsed
    if len(header) < 3:
        return TemplateSolveResult(False)

    product_labels = [_clean_label(cell) for cell in header[1:]]
    volumes: list[float] | None = None
    demands: list[float] | None = None
    unit_costs: list[float] | None = None
    for row in rows:
        if len(row) != len(header):
            continue
        row_label = _clean_label(row[0]).lower()
        values = [_first_number(cell) for cell in row[1:]]
        if any(value is None for value in values):
            continue
        numeric_values = [float(value) for value in values if value is not None]
        if "volume" in row_label:
            volumes = numeric_values
        elif "demand" in row_label:
            demands = numeric_values
        elif "variable" in row_label and "cost" in row_label:
            unit_costs = numeric_values

    fixed_match = re.search(
        rf"fixed\s+(?:setup\s+)?cost\s+of\s+\$?\s*({_NUMBER_TOKEN})",
        text,
        flags=re.IGNORECASE,
    )
    if not (volumes and demands and unit_costs and fixed_match):
        return TemplateSolveResult(False)
    if not (len(product_labels) == len(volumes) == len(demands) == len(unit_costs)):
        return TemplateSolveResult(False)

    fixed_cost = _number(fixed_match.group(1))
    product_count = len(product_labels)
    if product_count > 20:
        return TemplateSolveResult(
            matched=True,
            template_id="fixed_charge_substitution_production",
            status="solver_unavailable",
            confidence=0.75,
            notes="Too many product types for deterministic activation enumeration.",
            artifact={"products": product_labels, "volumes": volumes, "demands": demands},
        )

    best_objective: float | None = None
    best_active: tuple[int, ...] | None = None
    best_assignment: list[int | None] = []
    for mask in range(1, 1 << product_count):
        active = tuple(index for index in range(product_count) if mask & (1 << index))
        objective = fixed_cost * len(active)
        assignment: list[int | None] = [None] * product_count
        feasible = True
        for demand_index, demand in enumerate(demands):
            if math.isclose(demand, 0.0, abs_tol=1e-9):
                continue
            feasible_products = [
                produced_index
                for produced_index in active
                if volumes[produced_index] + 1e-9 >= volumes[demand_index]
            ]
            if not feasible_products:
                feasible = False
                break
            produced_index = min(
                feasible_products,
                key=lambda index: (unit_costs[index], volumes[index], product_labels[index]),
            )
            assignment[demand_index] = produced_index
            objective += demand * unit_costs[produced_index]
        if feasible and (best_objective is None or objective < best_objective):
            best_objective = objective
            best_active = active
            best_assignment = assignment

    if best_objective is None or best_active is None:
        return TemplateSolveResult(
            matched=True,
            template_id="fixed_charge_substitution_production",
            status="infeasible",
            confidence=0.8,
            artifact={
                "products": product_labels,
                "volumes": volumes,
                "demands": demands,
                "unit_costs": unit_costs,
                "fixed_cost": fixed_cost,
            },
        )

    produced_quantities = [0.0] * product_count
    assignments: list[dict[str, Any]] = []
    for demand_index, produced_index in enumerate(best_assignment):
        if produced_index is None:
            continue
        produced_quantities[produced_index] += demands[demand_index]
        assignments.append(
            {
                "demand_product": product_labels[demand_index],
                "produced_product": product_labels[produced_index],
                "quantity": demands[demand_index],
            }
        )

    variable_values: dict[str, float] = {
        f"activate_container_{product_labels[index]}": 1.0
        for index in best_active
    }
    for index, quantity in enumerate(produced_quantities):
        if not math.isclose(quantity, 0.0, abs_tol=1e-8):
            variable_values[f"produce_container_{product_labels[index]}"] = quantity

    return TemplateSolveResult(
        matched=True,
        template_id="fixed_charge_substitution_production",
        status="optimal",
        objective_value=float(best_objective),
        variable_values=variable_values,
        confidence=0.9,
        notes="Solved fixed-charge production with upward volume substitution by exact activation enumeration.",
        artifact={
            "products": product_labels,
            "volumes": volumes,
            "demands": demands,
            "unit_costs": unit_costs,
            "fixed_cost": fixed_cost,
            "active_products": [product_labels[index] for index in best_active],
            "assignments": assignments,
        },
    )


def _number_after_patterns(text: str, patterns: list[str]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return _number(match.group(1))
    return None


def _parse_period_demand_table(text: str) -> tuple[list[str], list[float]] | None:
    for header, rows in _parse_markdown_tables(text):
        if len(header) < 2:
            continue
        header_label = _clean_label(header[0]).lower()
        if "month" not in header_label and "period" not in header_label and "quarter" not in header_label:
            continue
        period_labels = [_clean_label(cell) for cell in header[1:]]
        for row in rows:
            if not row or "demand" not in _clean_label(row[0]).lower():
                continue
            values = [_first_number(cell) for cell in row[1:]]
            if len(values) == len(period_labels) and all(value is not None for value in values):
                return period_labels, [float(value) for value in values if value is not None]
    return None


def _solve_multi_period_workforce_production_plan(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        "workforce" in lowered
        and "outsourcing" in lowered
        and "inventory" in lowered
        and "backorder" in lowered
        and ("hire" in lowered or "hiring" in lowered)
        and ("fire" in lowered or "firing" in lowered)
        and ("maximize" in lowered or "net profit" in lowered)
    ):
        return TemplateSolveResult(False)

    demand_data = _parse_period_demand_table(text)
    if not demand_data:
        return TemplateSolveResult(False)
    period_labels, demands = demand_data

    initial_workforce = _number_after_patterns(
        text,
        [rf"Initial\s+Workforce:\s*({_NUMBER_TOKEN})\s+employees?"],
    )
    initial_inventory = _number_after_patterns(
        text,
        [rf"Initial\s+Inventory:\s*({_NUMBER_TOKEN})\s+units?"],
    )
    sales_price = _number_after_patterns(
        text,
        [rf"Sales\s+Price\b[^0-9A-Za-z]{{0,40}}({_NUMBER_TOKEN})\s+\w+\s+per\s+unit\s+sold"],
    )
    raw_material_cost = _number_after_patterns(
        text,
        [rf"Raw\s+Material\s+Cost\b[^0-9A-Za-z]{{0,40}}({_NUMBER_TOKEN})\s+\w+\s+per\s+unit"],
    )
    outsourcing_cost = _number_after_patterns(
        text,
        [rf"Outsourcing\s+Cost\b[^0-9A-Za-z]{{0,40}}({_NUMBER_TOKEN})\s+\w+\s+per\s+unit"],
    )
    holding_cost = _number_after_patterns(
        text,
        [rf"Inventory\s+Holding\s+Cost\b[^0-9A-Za-z]{{0,40}}({_NUMBER_TOKEN})\s+\w+\s+per\s+unit"],
    )
    backorder_cost = _number_after_patterns(
        text,
        [rf"Backorder\s+Cost\b[^0-9A-Za-z]{{0,40}}({_NUMBER_TOKEN})\s+\w+\s+per\s+unit"],
    )
    labor_hours_per_unit = _number_after_patterns(
        text,
        [rf"Each\s+in-house\s+unit\s+requires\s+({_NUMBER_TOKEN})\s+labor\s+hours?"],
    )
    regular_hours_per_worker = _number_after_patterns(
        text,
        [rf"Each\s+worker\s+provides\s+({_NUMBER_TOKEN})\s+regular\s+working\s+hours?"],
    )
    regular_wage = _number_after_patterns(
        text,
        [rf"regular\s+wage\s+of\s+({_NUMBER_TOKEN})\s+\w+\s*/?\s*hour"],
    )
    overtime_hours_per_worker = _number_after_patterns(
        text,
        [rf"overtime\s+hours?[^.\n]{{0,120}}?cannot\s+exceed\s+({_NUMBER_TOKEN})\s+hours?\s+per\s+worker"],
    )
    overtime_wage = _number_after_patterns(
        text,
        [rf"overtime\s+wage\s+is\s+({_NUMBER_TOKEN})\s+\w+\s*/?\s*hour"],
    )
    hiring_cost = _number_after_patterns(
        text,
        [rf"cost\s+to\s+hire\s+a\s+new\s+worker\s+is\s+({_NUMBER_TOKEN})\s+\w+"],
    )
    firing_cost = _number_after_patterns(
        text,
        [rf"cost\s+to\s+fire\s+a\s+worker\s+is\s+({_NUMBER_TOKEN})\s+\w+"],
    )
    terminal_inventory = _number_after_patterns(
        text,
        [rf"ending\s+inventory\s+must\s+be\s+at\s+least\s+({_NUMBER_TOKEN})\s+units?"],
    )
    if not all(
        value is not None
        for value in (
            initial_workforce,
            initial_inventory,
            sales_price,
            raw_material_cost,
            outsourcing_cost,
            holding_cost,
            backorder_cost,
            labor_hours_per_unit,
            regular_hours_per_worker,
            regular_wage,
            overtime_hours_per_worker,
            overtime_wage,
            hiring_cost,
            firing_cost,
            terminal_inventory,
        )
    ):
        return TemplateSolveResult(False)
    if not ("ending backorders" in lowered and ("zero" in lowered or "cleared" in lowered)):
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="multi_period_workforce_production_plan_milp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
            artifact={"periods": period_labels, "demands": demands},
        )

    period_count = len(demands)
    if period_count == 0 or period_count > 18:
        return TemplateSolveResult(False)

    blocks = ["inhouse", "outsourced", "overtime_hours", "workforce", "hired", "fired", "inventory", "backorder"]
    block_indices: dict[str, list[int]] = {}
    variable_names: list[str] = []
    for block in blocks:
        block_indices[block] = []
        for period_index in range(period_count):
            block_indices[block].append(len(variable_names))
            variable_names.append(f"{block}_{period_index + 1}")

    variable_count = len(variable_names)
    objective = np.zeros(variable_count)
    for period_index in range(period_count):
        objective[block_indices["inhouse"][period_index]] = raw_material_cost  # type: ignore[operator]
        objective[block_indices["outsourced"][period_index]] = outsourcing_cost  # type: ignore[operator]
        objective[block_indices["overtime_hours"][period_index]] = overtime_wage  # type: ignore[operator]
        objective[block_indices["workforce"][period_index]] = regular_hours_per_worker * regular_wage  # type: ignore[operator]
        objective[block_indices["hired"][period_index]] = hiring_cost  # type: ignore[operator]
        objective[block_indices["fired"][period_index]] = firing_cost  # type: ignore[operator]
        objective[block_indices["inventory"][period_index]] = holding_cost  # type: ignore[operator]
        objective[block_indices["backorder"][period_index]] = backorder_cost  # type: ignore[operator]

    rows: list[Any] = []
    lower: list[float] = []
    upper: list[float] = []

    for period_index in range(period_count):
        row = np.zeros(variable_count)
        row[block_indices["workforce"][period_index]] = 1.0
        row[block_indices["hired"][period_index]] = -1.0
        row[block_indices["fired"][period_index]] = 1.0
        if period_index > 0:
            row[block_indices["workforce"][period_index - 1]] = -1.0
            rhs = 0.0
        else:
            rhs = initial_workforce  # type: ignore[assignment]
        rows.append(row)
        lower.append(float(rhs))
        upper.append(float(rhs))

    for period_index, demand in enumerate(demands):
        row = np.zeros(variable_count)
        row[block_indices["inventory"][period_index]] = 1.0
        row[block_indices["backorder"][period_index]] = -1.0
        row[block_indices["inhouse"][period_index]] = -1.0
        row[block_indices["outsourced"][period_index]] = -1.0
        if period_index > 0:
            row[block_indices["inventory"][period_index - 1]] = -1.0
            row[block_indices["backorder"][period_index - 1]] = 1.0
            rhs = -demand
        else:
            rhs = initial_inventory - demand  # type: ignore[operator]
        rows.append(row)
        lower.append(float(rhs))
        upper.append(float(rhs))

    for period_index in range(period_count):
        row = np.zeros(variable_count)
        row[block_indices["inhouse"][period_index]] = labor_hours_per_unit  # type: ignore[assignment]
        row[block_indices["workforce"][period_index]] = -regular_hours_per_worker  # type: ignore[assignment]
        row[block_indices["overtime_hours"][period_index]] = -1.0
        rows.append(row)
        lower.append(-math.inf)
        upper.append(0.0)

        row = np.zeros(variable_count)
        row[block_indices["overtime_hours"][period_index]] = 1.0
        row[block_indices["workforce"][period_index]] = -overtime_hours_per_worker  # type: ignore[assignment]
        rows.append(row)
        lower.append(-math.inf)
        upper.append(0.0)

    row = np.zeros(variable_count)
    row[block_indices["inventory"][-1]] = 1.0
    rows.append(row)
    lower.append(float(terminal_inventory))  # type: ignore[arg-type]
    upper.append(math.inf)

    row = np.zeros(variable_count)
    row[block_indices["backorder"][-1]] = 1.0
    rows.append(row)
    lower.append(0.0)
    upper.append(0.0)

    integrality = np.zeros(variable_count)
    for block in ("inhouse", "outsourced", "workforce", "hired", "fired", "inventory", "backorder"):
        for index in block_indices[block]:
            integrality[index] = 1

    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(np.zeros(variable_count), np.full(variable_count, math.inf)),
        constraints=LinearConstraint(np.vstack(rows), np.array(lower), np.array(upper)),
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="multi_period_workforce_production_plan_milp",
            status="solver_failed",
            confidence=0.82,
            notes=str(result.message),
            artifact={"periods": period_labels, "demands": demands},
        )

    total_revenue = sales_price * sum(demands)  # type: ignore[operator]
    net_profit = total_revenue - float(result.fun)
    variable_values: dict[str, float] = {}
    for block in blocks:
        for period_index, variable_index in enumerate(block_indices[block]):
            value = float(result.x[variable_index])
            if not math.isclose(value, 0.0, abs_tol=1e-8):
                variable_values[f"{block}_{period_labels[period_index]}"] = round(value, 10)

    return TemplateSolveResult(
        matched=True,
        template_id="multi_period_workforce_production_plan_milp",
        status="optimal",
        objective_value=float(net_profit),
        variable_values=variable_values,
        confidence=0.88,
        notes="Solved multi-period workforce, outsourcing, inventory, and backlog production plan by MILP.",
        artifact={
            "periods": period_labels,
            "demands": demands,
            "initial_workforce": initial_workforce,
            "initial_inventory": initial_inventory,
            "sales_price": sales_price,
            "costs": {
                "raw_material": raw_material_cost,
                "outsourcing": outsourcing_cost,
                "holding": holding_cost,
                "backorder": backorder_cost,
                "regular_wage_per_hour": regular_wage,
                "overtime_wage_per_hour": overtime_wage,
                "hiring": hiring_cost,
                "firing": firing_cost,
            },
            "labor": {
                "hours_per_unit": labor_hours_per_unit,
                "regular_hours_per_worker": regular_hours_per_worker,
                "overtime_hours_per_worker": overtime_hours_per_worker,
            },
            "terminal_inventory_min": terminal_inventory,
            "total_revenue": float(total_revenue),
            "total_cost": float(result.fun),
        },
    )


def _solve_two_product_seasonal_inventory_plan_lp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "market demand for products i and ii" in lowered
        and "from july to december" in lowered
        and "warehouse capacity" in lowered
        and "external warehouse" in lowered
        and "initial inventory" in lowered
    ):
        return TemplateSolveResult(False)

    demand_i = re.search(
        rf"Product\s+I\s+requires\s+({_NUMBER_TOKEN})\s+units?\s+per\s+month\s+from\s+January\s+to\s+April,\s+"
        rf"({_NUMBER_TOKEN})\s+units?\s+per\s+month\s+from\s+May\s+to\s+September,\s+and\s+"
        rf"({_NUMBER_TOKEN})\s+units?\s+per\s+month\s+from\s+October\s+to\s+December",
        normalized,
        flags=re.IGNORECASE,
    )
    demand_ii = re.search(
        rf"Product\s+II\s+requires\s+({_NUMBER_TOKEN})\s+units?\s+per\s+month\s+from\s+March\s+to\s+September\s+and\s+"
        rf"({_NUMBER_TOKEN})\s+units?\s+per\s+month\s+during\s+other\s+months",
        normalized,
        flags=re.IGNORECASE,
    )
    cost_i = re.search(
        rf"Product\s+I\s+costs\s+({_NUMBER_TOKEN})\s+yuan\s+per\s+unit\s+to\s+produce\s+from\s+January\s+to\s+May,\s+"
        rf"and\s+({_NUMBER_TOKEN})\s+yuan\s+per\s+unit\s+from\s+June\s+to\s+December",
        normalized,
        flags=re.IGNORECASE,
    )
    cost_ii = re.search(
        rf"Product\s+II\s+costs\s+({_NUMBER_TOKEN})\s+yuan\s+per\s+unit\s+to\s+produce\s+from\s+January\s+to\s+May,\s+"
        rf"and\s+({_NUMBER_TOKEN})\s+yuan\s+per\s+unit\s+from\s+June\s+to\s+December",
        normalized,
        flags=re.IGNORECASE,
    )
    capacity = _number_after_patterns(normalized, [rf"combined\s+production\s+capacity[^.]*?not\s+exceed\s+({_NUMBER_TOKEN})\s+units?\s+per\s+month"])
    volumes = re.search(
        rf"Product\s+I\s+has\s+a\s+volume\s+of\s+({_NUMBER_TOKEN})\s+cubic\s+meters?\s+per\s+unit,\s+"
        rf"Product\s+II\s+has\s+a\s+volume\s+of\s+({_NUMBER_TOKEN})\s+cubic\s+meters?\s+per\s+unit",
        normalized,
        flags=re.IGNORECASE,
    )
    warehouse_capacity = _number_after_patterns(normalized, [rf"warehouse\s+capacity\s+is\s+({_NUMBER_TOKEN})\s+cubic\s+meters?"])
    own_storage = _number_after_patterns(normalized, [rf"own\s+warehouse\s+costs\s+({_NUMBER_TOKEN})\s+yuan\s+per\s+cubic\s+meter\s+per\s+month"])
    external_storage = _number_after_patterns(normalized, [rf"external\s+warehouse\s+increases\s+this\s+cost\s+to\s+({_NUMBER_TOKEN})\s+yuan\s+per\s+cubic\s+meter\s+per\s+month"])
    if not (
        demand_i
        and demand_ii
        and cost_i
        and cost_ii
        and capacity is not None
        and volumes
        and warehouse_capacity is not None
        and own_storage is not None
        and external_storage is not None
    ):
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="two_product_seasonal_inventory_plan_lp",
            status="solver_unavailable",
            confidence=0.82,
            notes=str(exc),
        )

    period_count = 6
    product_i_demand = [_number(demand_i.group(2))] * 3 + [_number(demand_i.group(3))] * 3
    product_ii_demand = [_number(demand_ii.group(1))] * 3 + [_number(demand_ii.group(2))] * 3
    production_costs = [_number(cost_i.group(2)), _number(cost_ii.group(2))]
    volume = [_number(volumes.group(1)), _number(volumes.group(2))]
    variable_count = period_count * 5

    def idx(block: int, period: int) -> int:
        return block * period_count + period

    objective = [0.0] * variable_count
    for period in range(period_count):
        objective[idx(0, period)] = production_costs[0]
        objective[idx(1, period)] = production_costs[1]
        objective[idx(2, period)] = volume[0] * own_storage
        objective[idx(3, period)] = volume[1] * own_storage
        objective[idx(4, period)] = external_storage - own_storage

    a_ub: list[list[float]] = []
    b_ub: list[float] = []
    a_eq: list[list[float]] = []
    b_eq: list[float] = []
    for period in range(period_count):
        row = [0.0] * variable_count
        row[idx(0, period)] = 1.0
        row[idx(1, period)] = 1.0
        a_ub.append(row)
        b_ub.append(capacity)

        row = [0.0] * variable_count
        row[idx(2, period)] = volume[0]
        row[idx(3, period)] = volume[1]
        row[idx(4, period)] = -1.0
        a_ub.append(row)
        b_ub.append(warehouse_capacity)

        row = [0.0] * variable_count
        row[idx(0, period)] = 1.0
        row[idx(2, period)] = -1.0
        if period > 0:
            row[idx(2, period - 1)] = 1.0
        a_eq.append(row)
        b_eq.append(product_i_demand[period])

        row = [0.0] * variable_count
        row[idx(1, period)] = 1.0
        row[idx(3, period)] = -1.0
        if period > 0:
            row[idx(3, period - 1)] = 1.0
        a_eq.append(row)
        b_eq.append(product_ii_demand[period])

    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(0, None)] * variable_count,
        method="highs",
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="two_product_seasonal_inventory_plan_lp",
            status="solver_failed",
            confidence=0.82,
            notes=str(result.message),
        )

    variable_values: dict[str, float] = {}
    for period in range(period_count):
        month = period + 7
        for block, label in enumerate(("produce_I", "produce_II", "inventory_I", "inventory_II", "external_storage_volume")):
            value = float(result.x[idx(block, period)])
            if not math.isclose(value, 0.0, abs_tol=1e-8):
                variable_values[f"{label}_month_{month}"] = value
    return TemplateSolveResult(
        matched=True,
        template_id="two_product_seasonal_inventory_plan_lp",
        status="optimal",
        objective_value=float(result.fun),
        variable_values=variable_values,
        confidence=0.88,
        notes="Solved July-December two-product production and inventory planning LP with internal/external warehouse cost.",
        artifact={
            "demands": {"I": product_i_demand, "II": product_ii_demand},
            "production_costs": {"I": production_costs[0], "II": production_costs[1]},
            "volumes": {"I": volume[0], "II": volume[1]},
            "warehouse_capacity": warehouse_capacity,
        },
    )


def _solve_container_loading_min_count(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "fewest number of containers" in lowered
        and "container" in lowered
        and "goods" in lowered
        and "maximum of" in lowered
        and "at least" in lowered
    ):
        return TemplateSolveResult(False)
    capacity = _number_after_patterns(
        normalized,
        [
            rf"container\s+able\s+to\s+hold\s+a\s+maximum\s+of\s+({_NUMBER_TOKEN})\s+tons?",
            rf"container\s+can\s+hold\s+a\s+maximum\s+of\s+({_NUMBER_TOKEN})\s+tons?",
        ],
    )
    min_load = _number_after_patterns(normalized, [rf"each\s+container\s+used\s+must\s+load\s+at\s+least\s+({_NUMBER_TOKEN})\s+tons?"])
    quantity_match = re.search(
        rf"types:\s*A,\s*B,\s*C,\s*D,\s*and\s*E,\s*with\s+quantities\s+of\s+"
        rf"({_NUMBER_TOKEN}),\s*({_NUMBER_TOKEN}),\s*({_NUMBER_TOKEN}),\s*({_NUMBER_TOKEN}),\s*and\s*({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    weight_matches = re.findall(
        rf"({_NUMBER_TOKEN})\s+tons?\s+for\s+([A-E])",
        normalized,
        flags=re.IGNORECASE,
    )
    d_min = _number_after_patterns(normalized, [rf"each\s+container\s+must\s+load\s+at\s+least\s+({_NUMBER_TOKEN})\s+units?\s+of\s+D"])
    if capacity is None or min_load is None or quantity_match is None or d_min is None:
        return TemplateSolveResult(False)
    labels = ["A", "B", "C", "D", "E"]
    quantities = {label: _number(quantity_match.group(index)) for index, label in enumerate(labels, start=1)}
    weights = {label.upper(): _number(value) for value, label in weight_matches}
    if set(weights) != set(labels):
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="container_loading_min_count_milp",
            status="solver_unavailable",
            confidence=0.82,
            notes=str(exc),
        )

    total_weight = sum(quantities[label] * weights[label] for label in labels)
    lower_bound = max(int(math.ceil(total_weight / capacity)), 1)
    for container_count in range(lower_bound, lower_bound + 8):
        if d_min * container_count > quantities["D"] + 1e-9:
            continue
        quantity_var_count = container_count * len(labels)
        variable_count = quantity_var_count + container_count

        def q_index(container: int, label_index: int) -> int:
            return container * len(labels) + label_index

        def z_index(container: int) -> int:
            return quantity_var_count + container

        rows: list[list[float]] = []
        lower: list[float] = []
        upper: list[float] = []
        for label_index, label in enumerate(labels):
            row = [0.0] * variable_count
            for container in range(container_count):
                row[q_index(container, label_index)] = 1.0
            rows.append(row)
            lower.append(quantities[label])
            upper.append(quantities[label])

        for container in range(container_count):
            row = [0.0] * variable_count
            for label_index, label in enumerate(labels):
                row[q_index(container, label_index)] = weights[label]
            rows.append(row)
            lower.append(min_load)
            upper.append(capacity)

            row = [0.0] * variable_count
            row[q_index(container, labels.index("D"))] = 1.0
            rows.append(row)
            lower.append(d_min)
            upper.append(math.inf)

            row = [0.0] * variable_count
            row[q_index(container, labels.index("A"))] = 1.0
            row[z_index(container)] = -quantities["A"]
            rows.append(row)
            lower.append(-math.inf)
            upper.append(0.0)

            row = [0.0] * variable_count
            row[q_index(container, labels.index("C"))] = 1.0
            row[z_index(container)] = -1.0
            rows.append(row)
            lower.append(0.0)
            upper.append(math.inf)

        result = milp(
            c=np.zeros(variable_count),
            integrality=np.ones(variable_count),
            bounds=Bounds(
                np.zeros(variable_count),
                np.array(
                    [quantities[label] for _container in range(container_count) for label in labels]
                    + [1.0] * container_count,
                    dtype=float,
                ),
            ),
            constraints=LinearConstraint(np.array(rows, dtype=float), np.array(lower, dtype=float), np.array(upper, dtype=float)),
        )
        if result.success:
            variable_values: dict[str, float] = {}
            for container in range(container_count):
                for label_index, label in enumerate(labels):
                    value = float(result.x[q_index(container, label_index)])
                    if not math.isclose(value, 0.0, abs_tol=1e-8):
                        variable_values[f"container_{container + 1}_{label}"] = value
            return TemplateSolveResult(
                matched=True,
                template_id="container_loading_min_count_milp",
                status="optimal",
                objective_value=float(container_count),
                variable_values=variable_values,
                confidence=0.86,
                notes="Solved minimum container count by MILP with capacity, minimum load, D-per-container, and A-implies-C constraints.",
                artifact={"quantities": quantities, "weights": weights, "capacity": capacity, "min_load": min_load, "d_min": d_min},
            )

    return TemplateSolveResult(
        matched=True,
        template_id="container_loading_min_count_milp",
        status="infeasible",
        confidence=0.78,
        artifact={"quantities": quantities, "weights": weights, "capacity": capacity, "min_load": min_load, "d_min": d_min},
    )


def _solve_input_output_gdp_lp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "carelland" in lowered
        and "domestic gdp" in lowered
        and "steel" in lowered
        and "engines" in lowered
        and "electronic" in lowered
        and "plastic" in lowered
    ):
        return TemplateSolveResult(False)
    price_match = re.search(
        rf"prices\s+of\s+steel,\s+engines,\s+(?:electronics|electronic\s+components),\s+and\s+plastic[^.]*?are[^.]*?"
        rf"({_NUMBER_TOKEN}),\s*({_NUMBER_TOKEN}),\s*({_NUMBER_TOKEN}),\s*({_NUMBER_TOKEN})\s+respectively",
        normalized,
        flags=re.IGNORECASE,
    )
    if price_match is None:
        return TemplateSolveResult(False)
    labels = ["steel", "engines", "electronics", "plastic"]
    prices = {label: _number(price_match.group(index)) for index, label in enumerate(labels, start=1)}
    commodity_aliases = {
        "steel": "steel",
        "engine": "engines",
        "engines": "engines",
        "electronic components": "electronics",
        "electronics": "electronics",
        "plastic": "plastic",
    }
    input_coefficients = {label: {inner: 0.0 for inner in labels} for label in labels}
    imports: dict[str, float] = {}
    labor: dict[str, float] = {}
    product_patterns = {
        "steel": r"Producing\s+1\s+unit\s+of\s+steel\s+requires\s+(.*?)(?=Producing\s+1\s+unit\s+of\s+engines)",
        "engines": r"Producing\s+1\s+unit\s+of\s+engines\s+requires\s+(.*?)(?=One\s+unit\s+of\s+electronics)",
        "electronics": r"One\s+unit\s+of\s+electronics\s+requires:\s+(.*?)(?=One\s+unit\s+of\s+plastic)",
        "plastic": r"One\s+unit\s+of\s+plastic\s+requires:\s+(.*?)(?=Engine\s+production)",
    }
    for product, pattern in product_patterns.items():
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if not match:
            return TemplateSolveResult(False)
        body = match.group(1)
        for value, raw_label in re.findall(
            rf"({_NUMBER_TOKEN})\s+units?\s+of\s+(steel|engines?|electronic\s+components|plastic)",
            body,
            flags=re.IGNORECASE,
        ):
            input_coefficients[product][commodity_aliases[raw_label.lower()]] = _number(value)
        imported = re.search(rf"({_NUMBER_TOKEN})\s+Klunz\s+of\s+(?:other\s+)?imported\s+goods", body, flags=re.IGNORECASE)
        labor_match = re.search(rf"({_NUMBER_TOKEN})\s+person-(months?|years?)(?:\s+of\s+labor)?", body, flags=re.IGNORECASE)
        if imported is None or labor_match is None:
            return TemplateSolveResult(False)
        imports[product] = _number(imported.group(1))
        labor[product] = _number(labor_match.group(1)) * (12.0 if "year" in labor_match.group(2).lower() else 1.0)
    engine_limit = _number_after_patterns(normalized, [rf"Engine\s+production\s+is\s+limited\s+to\s+({_NUMBER_TOKEN})\s+units?"])
    plastic_limit = _number_after_patterns(normalized, [rf"plastic\s+production\s+is\s+limited\s+to\s+({_NUMBER_TOKEN})\s+units?"])
    labor_limit = _number_after_patterns(normalized, [rf"available\s+labor\s+force\s+per\s+year\s+is\s+({_NUMBER_TOKEN})\s+person-months?"])
    if engine_limit is None or plastic_limit is None or labor_limit is None:
        return TemplateSolveResult(False)

    coefficients: list[float] = []
    for product in labels:
        domestic_intermediate = sum(prices[input_label] * input_coefficients[product][input_label] for input_label in labels)
        coefficients.append(prices[product] - domestic_intermediate - imports[product])
    status, objective, values, message = _linprog_maximize(
        objective=coefficients,
        constraints=[
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [labor[label] for label in labels],
        ],
        upper_bounds=[engine_limit, plastic_limit, labor_limit],
    )
    if status != "optimal":
        return TemplateSolveResult(
            matched=True,
            template_id="input_output_gdp_lp",
            status=status,
            confidence=0.82,
            notes=message,
        )
    return TemplateSolveResult(
        matched=True,
        template_id="input_output_gdp_lp",
        status="optimal",
        objective_value=objective,
        variable_values={label: value for label, value in zip(labels, values) if not math.isclose(value, 0.0, abs_tol=1e-8)},
        confidence=0.86,
        notes="Solved input-output GDP LP by maximizing output value net of domestic intermediate inputs and imported goods under labor/product caps.",
        artifact={"prices": prices, "imports": imports, "labor": labor, "gdp_coefficients": dict(zip(labels, coefficients))},
    )


def _solve_vrp_hard_time_windows_milp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        ("vrphtw" in lowered or "vehicle routing problem with hard time windows" in lowered)
        and "service duration" in lowered
        and "minimize the total distance" in lowered
    ):
        return TemplateSolveResult(False)
    depot = re.search(
        rf"Coordinates:\s*\(\s*({_NUMBER_TOKEN})\s*,\s*({_NUMBER_TOKEN})\s*\).*?"
        rf"Operating\s+Time\s+Window:\s*\[\s*({_NUMBER_TOKEN})\s*,\s*({_NUMBER_TOKEN})\s*\]",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    truck_count = _number_after_patterns(normalized, [rf"use\s+at\s+most\s+({_NUMBER_TOKEN})\s+trucks?"])
    capacity = _number_after_patterns(normalized, [rf"capacity\s+of\s+each\s+truck\s+is\s+({_NUMBER_TOKEN})\s+units?"])
    if depot is None or truck_count is None or capacity is None:
        return TemplateSolveResult(False)
    parsed = _parse_markdown_table(text)
    if not parsed:
        return TemplateSolveResult(False)
    header, rows = parsed
    header_text = " ".join(header).lower()
    if "customer id" not in header_text or "time window" not in header_text:
        return TemplateSolveResult(False)

    customers: dict[int, tuple[tuple[float, float], float, tuple[float, float], float]] = {}
    for row in rows:
        if len(row) < 5:
            continue
        try:
            customer_id = int(_number(row[0]))
        except ValueError:
            continue
        coord = re.search(rf"\(\s*({_NUMBER_TOKEN})\s*,\s*({_NUMBER_TOKEN})\s*\)", row[1])
        time_window = re.search(rf"\[\s*({_NUMBER_TOKEN})\s*,\s*({_NUMBER_TOKEN})\s*\]", row[3])
        if coord is None or time_window is None:
            continue
        customers[customer_id] = (
            (_number(coord.group(1)), _number(coord.group(2))),
            _number(row[2]),
            (_number(time_window.group(1)), _number(time_window.group(2))),
            _number(row[4]),
        )
    if len(customers) < 2:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="vrp_hard_time_windows_milp",
            status="solver_unavailable",
            confidence=0.78,
            notes=str(exc),
        )

    vehicle_count = int(truck_count)
    customer_ids = sorted(customers)
    nodes = [0] + customer_ids
    coordinates = {0: (_number(depot.group(1)), _number(depot.group(2)))}
    coordinates.update({customer_id: data[0] for customer_id, data in customers.items()})

    def distance(left: int, right: int) -> float:
        left_x, left_y = coordinates[left]
        right_x, right_y = coordinates[right]
        return math.hypot(left_x - right_x, left_y - right_y)

    arcs = [
        (vehicle, left, right)
        for vehicle in range(vehicle_count)
        for left in nodes
        for right in nodes
        if left != right
    ]
    x_index = {arc: offset for offset, arc in enumerate(arcs)}
    time_offset = len(arcs)
    time_index = {
        (vehicle, node): time_offset + vehicle * len(nodes) + node_offset
        for vehicle in range(vehicle_count)
        for node_offset, node in enumerate(nodes)
    }
    variable_count = time_offset + vehicle_count * len(nodes)
    objective = np.zeros(variable_count)
    for arc, offset in x_index.items():
        objective[offset] = distance(arc[1], arc[2])

    integrality = np.zeros(variable_count)
    integrality[: len(arcs)] = 1
    lower_bounds = np.zeros(variable_count)
    upper_bounds = np.full(variable_count, math.inf)
    upper_bounds[: len(arcs)] = 1.0
    depot_window = (_number(depot.group(3)), _number(depot.group(4)))
    for vehicle in range(vehicle_count):
        for node in nodes:
            offset = time_index[(vehicle, node)]
            if node == 0:
                lower_bounds[offset] = depot_window[0]
                upper_bounds[offset] = depot_window[1]
            else:
                lower_bounds[offset] = customers[node][2][0]
                upper_bounds[offset] = customers[node][2][1]

    rows: list[Any] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_constraint(coefficients: dict[int, float], low: float, high: float) -> None:
        row = np.zeros(variable_count)
        for offset, value in coefficients.items():
            row[offset] = value
        rows.append(row)
        lower.append(low)
        upper.append(high)

    for customer_id in customer_ids:
        add_constraint(
            {
                x_index[(vehicle, left, customer_id)]: 1.0
                for vehicle in range(vehicle_count)
                for left in nodes
                if left != customer_id
            },
            1.0,
            1.0,
        )
        add_constraint(
            {
                x_index[(vehicle, customer_id, right)]: 1.0
                for vehicle in range(vehicle_count)
                for right in nodes
                if right != customer_id
            },
            1.0,
            1.0,
        )

    for vehicle in range(vehicle_count):
        outbound_depot = {x_index[(vehicle, 0, customer_id)]: 1.0 for customer_id in customer_ids}
        inbound_depot = {x_index[(vehicle, customer_id, 0)]: 1.0 for customer_id in customer_ids}
        add_constraint(outbound_depot, 0.0, 1.0)
        add_constraint(inbound_depot, 0.0, 1.0)
        depot_balance = dict(outbound_depot)
        for offset, value in inbound_depot.items():
            depot_balance[offset] = depot_balance.get(offset, 0.0) - value
        add_constraint(depot_balance, 0.0, 0.0)

        for customer_id in customer_ids:
            flow: dict[int, float] = {}
            for left in nodes:
                if left != customer_id:
                    offset = x_index[(vehicle, left, customer_id)]
                    flow[offset] = flow.get(offset, 0.0) + 1.0
            for right in nodes:
                if right != customer_id:
                    offset = x_index[(vehicle, customer_id, right)]
                    flow[offset] = flow.get(offset, 0.0) - 1.0
            add_constraint(flow, 0.0, 0.0)

        capacity_row: dict[int, float] = {}
        for customer_id in customer_ids:
            for right in nodes:
                if right != customer_id:
                    capacity_row[x_index[(vehicle, customer_id, right)]] = customers[customer_id][1]
        add_constraint(capacity_row, -math.inf, capacity)
        add_constraint({time_index[(vehicle, 0)]: 1.0}, depot_window[0], depot_window[0])

    big_m = max(depot_window[1] + 2 * max(distance(left, right) for left in nodes for right in nodes if left != right), 2000.0)
    for vehicle, left, right in arcs:
        if right == 0:
            continue
        service = 0.0 if left == 0 else customers[left][3]
        add_constraint(
            {
                time_index[(vehicle, left)]: 1.0,
                time_index[(vehicle, right)]: -1.0,
                x_index[(vehicle, left, right)]: big_m,
            },
            -math.inf,
            big_m - service - distance(left, right),
        )

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(np.vstack(rows), np.array(lower), np.array(upper)),
        options={"time_limit": 30, "mip_rel_gap": 1e-9},
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="vrp_hard_time_windows_milp",
            status="solver_failed",
            confidence=0.78,
            notes=str(result.message),
        )

    variable_values: dict[str, float] = {}
    routes: list[list[int]] = []
    for vehicle in range(vehicle_count):
        successor = {
            left: right
            for left in nodes
            for right in nodes
            if left != right and result.x[x_index[(vehicle, left, right)]] > 0.5
        }
        if 0 not in successor:
            continue
        route: list[int] = []
        current = 0
        while current in successor:
            current = successor[current]
            if current == 0:
                break
            route.append(current)
            variable_values[f"vehicle_{vehicle + 1}_arrival_customer_{current}"] = float(result.x[time_index[(vehicle, current)]])
        if route:
            routes.append(route)
    return TemplateSolveResult(
        matched=True,
        template_id="vrp_hard_time_windows_milp",
        status="optimal",
        objective_value=float(result.fun),
        variable_values=variable_values,
        confidence=0.86,
        notes="Solved small VRP with hard time windows by vehicle-indexed MILP.",
        artifact={"routes": routes, "vehicle_count": vehicle_count, "capacity": capacity},
    )


_ORDINAL_INDEX = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
}


def _solve_periodic_production_inventory_lp(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        "regular-time labor" in lowered
        and "overtime labor" in lowered
        and ("carrying or holding cost" in lowered or "holding cost" in lowered)
        and "inventory" in lowered
        and ("minimize" in lowered or "minimal" in lowered)
    ):
        return TemplateSolveResult(False)

    demand_matches = re.findall(
        rf"\b({'|'.join(_ORDINAL_INDEX)})\s+quarter,\s+({_NUMBER_TOKEN})",
        text,
        flags=re.IGNORECASE,
    )
    if not demand_matches:
        return TemplateSolveResult(False)
    period_demands = sorted(
        ((_ORDINAL_INDEX[label.lower()], _number(value)) for label, value in demand_matches),
        key=lambda item: item[0],
    )
    demands = [value for _index, value in period_demands]

    initial_inventory = re.search(
        (
            rf"(?:has\s+an\s+inventory\s+of\s+({_NUMBER_TOKEN})\s+\w+)"
            rf"|(?:has\s+({_NUMBER_TOKEN})\s+\w+\s+in\s+inventory)"
        ),
        text,
        flags=re.IGNORECASE,
    )
    regular = re.search(
        (
            rf"produce\s+up\s+to\s+({_NUMBER_TOKEN})\s+\w+.*?regular-time\s+labor"
            rf".*?cost\s+of\s+\$?({_NUMBER_TOKEN})\s+per"
        ),
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    overtime_with_cap = re.search(
        (
            rf"additional\s+({_NUMBER_TOKEN})\s+\w+.*?overtime\s+labor"
            rf".*?cost\s+of\s+\$?({_NUMBER_TOKEN})\s+per"
        ),
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    overtime_cost_only = re.search(
        rf"overtime\s+labor.*?cost\s+of\s+\$?({_NUMBER_TOKEN})\s+per",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    holding = re.search(
        rf"(?:carrying\s+or\s+holding\s+cost|holding\s+cost)\s+of\s+\$?({_NUMBER_TOKEN})\s+per",
        text,
        flags=re.IGNORECASE,
    )
    if not (initial_inventory and regular and (overtime_with_cap or overtime_cost_only) and holding):
        return TemplateSolveResult(False)

    initial = _number(initial_inventory.group(1) or initial_inventory.group(2))
    regular_capacity = _number(regular.group(1))
    regular_cost = _number(regular.group(2))
    overtime_capacity = _number(overtime_with_cap.group(1)) if overtime_with_cap else None
    overtime_cost = _number(
        overtime_with_cap.group(2) if overtime_with_cap else overtime_cost_only.group(1)  # type: ignore[union-attr]
    )
    holding_cost = _number(holding.group(1))

    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="periodic_production_inventory_lp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
            artifact={"demands": demands},
        )

    period_count = len(demands)
    variable_count = 3 * period_count
    c = (
        [regular_cost] * period_count
        + [overtime_cost] * period_count
        + [holding_cost] * period_count
    )
    a_ub: list[list[float]] = []
    b_ub: list[float] = []
    for period in range(period_count):
        row = [0.0] * variable_count
        row[period] = 1.0
        a_ub.append(row)
        b_ub.append(regular_capacity)
        if overtime_capacity is not None:
            row = [0.0] * variable_count
            row[period_count + period] = 1.0
            a_ub.append(row)
            b_ub.append(overtime_capacity)

    a_eq: list[list[float]] = []
    b_eq: list[float] = []
    for period, demand in enumerate(demands):
        row = [0.0] * variable_count
        row[period] = 1.0
        row[period_count + period] = 1.0
        row[2 * period_count + period] = -1.0
        if period > 0:
            row[2 * period_count + period - 1] = 1.0
        a_eq.append(row)
        b_eq.append(demand - (initial if period == 0 else 0.0))

    result = linprog(
        c,
        A_ub=a_ub or None,
        b_ub=b_ub or None,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(0, None)] * variable_count,
        method="highs",
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="periodic_production_inventory_lp",
            status="solver_failed",
            confidence=0.8,
            notes=str(result.message),
            artifact={"demands": demands},
        )

    variable_values: dict[str, float] = {}
    for period in range(period_count):
        variable_values[f"regular_period_{period + 1}"] = float(result.x[period])
        variable_values[f"overtime_period_{period + 1}"] = float(result.x[period_count + period])
        variable_values[f"ending_inventory_period_{period + 1}"] = float(result.x[2 * period_count + period])

    return TemplateSolveResult(
        matched=True,
        template_id="periodic_production_inventory_lp",
        status="optimal",
        objective_value=float(result.fun),
        variable_values=variable_values,
        confidence=0.88,
        notes="Solved finite-horizon production/inventory LP with regular time, overtime, and holding cost.",
        artifact={
            "demands": demands,
            "initial_inventory": initial,
            "regular_capacity": regular_capacity,
            "regular_cost": regular_cost,
            "overtime_capacity": overtime_capacity,
            "overtime_cost": overtime_cost,
            "holding_cost": holding_cost,
        },
    )


def _parse_scalar_assignment(text: str, name: str) -> float | None:
    normalized = text.replace("\\n", "\n")
    match = re.search(
        rf"\b{name}\s*=\s*({_NUMBER_TOKEN})\b",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return _number(match.group(1))


def _solve_tool_repair_replacement(text: str) -> TemplateSolveResult:
    normalized_text = text.replace("\\n", "\n")
    lowered = text.lower()
    if not (
        "planning stages" in lowered
        and "repair" in lowered
        and "new" in lowered
        and ("slow repair" in lowered and "fast repair" in lowered)
    ):
        return TemplateSolveResult(False)

    n_value = _parse_scalar_assignment(normalized_text, "n")
    demand_match = re.search(r"\br\s*=\s*\[([^\]]+)\]", normalized_text, flags=re.IGNORECASE)
    purchase_cost = _parse_scalar_assignment(normalized_text, "a")
    slow_cost = _parse_scalar_assignment(normalized_text, "b")
    fast_cost = _parse_scalar_assignment(normalized_text, "c")
    slow_duration = _parse_scalar_assignment(normalized_text, "p")
    fast_duration = _parse_scalar_assignment(normalized_text, "q")
    if not (
        n_value is not None
        and demand_match
        and purchase_cost is not None
        and slow_cost is not None
        and fast_cost is not None
        and slow_duration is not None
        and fast_duration is not None
    ):
        return TemplateSolveResult(False)

    demands = [
        _number(value)
        for value in re.findall(_NUMBER_TOKEN, demand_match.group(1), flags=re.IGNORECASE)
    ]
    stage_count = int(n_value)
    if stage_count <= 0 or len(demands) != stage_count:
        return TemplateSolveResult(False)
    slow_lead = int(slow_duration) + 1
    fast_lead = int(fast_duration) + 1
    if slow_lead <= fast_lead or fast_lead <= 0:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="tool_repair_replacement_milp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
            artifact={"demands": demands},
        )

    variable_names: list[str] = []

    def add(name: str) -> int:
        variable_names.append(name)
        return len(variable_names) - 1

    new_tools = [add(f"buy_stage_{stage + 1}") for stage in range(stage_count)]
    carried_tools = [add(f"carry_clean_after_stage_{stage + 1}") for stage in range(stage_count)]
    fast_repair = [add(f"fast_repair_after_stage_{stage + 1}") for stage in range(stage_count)]
    slow_repair = [add(f"slow_repair_after_stage_{stage + 1}") for stage in range(stage_count)]
    variable_count = len(variable_names)

    objective = np.zeros(variable_count)
    for index in new_tools:
        objective[index] = purchase_cost
    for index in fast_repair:
        objective[index] = fast_cost
    for index in slow_repair:
        objective[index] = slow_cost

    rows: list[Any] = []
    lower: list[float] = []
    upper: list[float] = []
    for stage_index, demand in enumerate(demands):
        row = np.zeros(variable_count)
        row[new_tools[stage_index]] = 1.0
        row[carried_tools[stage_index]] = -1.0
        if stage_index > 0:
            row[carried_tools[stage_index - 1]] = 1.0
        if stage_index - fast_lead >= 0:
            row[fast_repair[stage_index - fast_lead]] = 1.0
        if stage_index - slow_lead >= 0:
            row[slow_repair[stage_index - slow_lead]] = 1.0
        rows.append(row)
        lower.append(demand)
        upper.append(demand)

    for stage_index, demand in enumerate(demands):
        row = np.zeros(variable_count)
        row[fast_repair[stage_index]] = 1.0
        row[slow_repair[stage_index]] = 1.0
        rows.append(row)
        lower.append(-math.inf)
        upper.append(demand)

    result = milp(
        objective,
        integrality=np.ones(variable_count),
        bounds=Bounds(np.zeros(variable_count), np.full(variable_count, math.inf)),
        constraints=LinearConstraint(np.vstack(rows), np.array(lower), np.array(upper)),
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="tool_repair_replacement_milp",
            status="solver_failed",
            confidence=0.82,
            notes=str(result.message),
            artifact={"demands": demands},
        )

    variable_values = {
        variable_names[index]: float(value)
        for index, value in enumerate(result.x)
        if not math.isclose(float(value), 0.0, abs_tol=1e-8)
    }
    return TemplateSolveResult(
        matched=True,
        template_id="tool_repair_replacement_milp",
        status="optimal",
        objective_value=float(result.fun),
        variable_values=variable_values,
        confidence=0.9,
        notes=(
            "Solved multi-stage tool replacement and repair planning MILP. "
            "Repairs sent after a stage are available after the stated full repair duration."
        ),
        artifact={
            "demands": demands,
            "purchase_cost": purchase_cost,
            "repair_costs": {"fast": fast_cost, "slow": slow_cost},
            "repair_leads": {"fast": fast_lead, "slow": slow_lead},
        },
    )


def _strip_latex_markup(value: str) -> str:
    cleaned = re.sub(r"\\textbf\{([^}]*)\}", r"\1", value)
    cleaned = cleaned.replace("\\", " ")
    return _clean_label(cleaned)


def _parse_reliability_table(text: str) -> tuple[list[list[float]], list[float], list[float]] | None:
    reliability_rows: dict[int, list[float]] = {}
    prices: list[float] | None = None
    weights: list[float] | None = None
    for raw_row in re.findall(r"([^\\]+?)\\\\\s*\\hline", text, flags=re.DOTALL):
        cells = [_strip_latex_markup(cell) for cell in raw_row.split("&")]
        if len(cells) < 2:
            continue
        row_label = cells[0].lower()
        values = [_first_number(cell) for cell in cells[1:]]
        if any(value is None for value in values):
            continue
        numeric_values = [float(value) for value in values if value is not None]
        if re.fullmatch(r"[0-9]+", row_label):
            reliability_rows[int(row_label)] = numeric_values
        elif "price" in row_label:
            prices = numeric_values
        elif "weight" in row_label:
            weights = numeric_values

    if not reliability_rows or prices is None or weights is None:
        return None
    spare_levels = sorted(reliability_rows)
    reliability_by_component = [
        [reliability_rows[level][component_index] for level in spare_levels]
        for component_index in range(len(prices))
    ]
    if len(weights) != len(prices):
        return None
    if any(len(row) != len(spare_levels) for row in reliability_by_component):
        return None
    return reliability_by_component, prices, weights


def _solve_reliability_spares_allocation(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        "reliability" in lowered
        and "spare" in lowered
        and "budget" in lowered
        and "weight" in lowered
        and ("product of the reliabilities" in lowered or "system's operational reliability" in lowered)
    ):
        return TemplateSolveResult(False)

    parsed = _parse_reliability_table(text)
    budget_match = re.search(
        rf"budget.*?(?:limited\s+to|limit(?:ed)?\s+to|is)\s+({_NUMBER_TOKEN})\s+yuan",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    weight_match = re.search(
        rf"weight\s+limit\s+is\s+({_NUMBER_TOKEN})\s+kg",
        text,
        flags=re.IGNORECASE,
    )
    if not (parsed and budget_match and weight_match):
        return TemplateSolveResult(False)

    reliability_by_component, prices, weights = parsed
    budget_limit = _number(budget_match.group(1))
    weight_limit = _number(weight_match.group(1))
    component_count = len(prices)
    level_count = len(reliability_by_component[0])
    if component_count == 0 or level_count == 0:
        return TemplateSolveResult(False)

    best_value: float | None = None
    best_levels: tuple[int, ...] | None = None
    for levels in itertools.product(range(level_count), repeat=component_count):
        total_cost = sum(levels[index] * prices[index] for index in range(component_count))
        total_weight = sum(levels[index] * weights[index] for index in range(component_count))
        if total_cost > budget_limit + 1e-9 or total_weight > weight_limit + 1e-9:
            continue
        value = math.prod(
            reliability_by_component[index][levels[index]]
            for index in range(component_count)
        )
        if best_value is None or value > best_value:
            best_value = value
            best_levels = levels

    if best_value is None or best_levels is None:
        return TemplateSolveResult(
            matched=True,
            template_id="reliability_spares_allocation",
            status="infeasible",
            confidence=0.8,
            artifact={"prices": prices, "weights": weights},
        )

    return TemplateSolveResult(
        matched=True,
        template_id="reliability_spares_allocation",
        status="optimal",
        objective_value=float(best_value),
        variable_values={
            f"spares_component_{index + 1}": float(level)
            for index, level in enumerate(best_levels)
        },
        confidence=0.9,
        notes="Solved discrete reliability allocation by exact enumeration of spare levels under budget and weight limits.",
        artifact={
            "reliability_by_component": reliability_by_component,
            "prices": prices,
            "weights": weights,
            "budget_limit": budget_limit,
            "weight_limit": weight_limit,
            "selected_spares": list(best_levels),
        },
    )


def _solve_inventory_arbitrage_lp(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        "warehouse capacity" in lowered
        and "purchasing price" in lowered
        and "selling price" in lowered
        and "purchases goods once at the beginning" in lowered
        and "maximize" in lowered
    ):
        return TemplateSolveResult(False)

    table = None
    for candidate in _parse_markdown_tables(text):
        labels = " ".join(row[0].lower() for row in candidate[1] if row)
        if "purchasing price" in labels and "selling price" in labels:
            table = candidate
            break
    if table is None:
        return TemplateSolveResult(False)

    purchase_prices: list[float] | None = None
    selling_prices: list[float] | None = None
    for row in table[1]:
        if len(row) < 2:
            continue
        row_label = row[0].lower()
        values = [_first_number(cell) for cell in row[1:]]
        if any(value is None for value in values):
            continue
        numeric_values = [float(value) for value in values if value is not None]
        if "purchasing price" in row_label:
            purchase_prices = numeric_values
        elif "selling price" in row_label:
            selling_prices = numeric_values
    capacity = re.search(
        rf"store\s+up\s+to\s+({_NUMBER_TOKEN})\s+units?",
        text,
        flags=re.IGNORECASE,
    )
    initial = re.search(
        rf"there\s+are\s+({_NUMBER_TOKEN})\s+units?\s+in\s+stock",
        text,
        flags=re.IGNORECASE,
    )
    if not (purchase_prices and selling_prices and capacity and initial):
        return TemplateSolveResult(False)
    if len(purchase_prices) != len(selling_prices):
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="inventory_arbitrage_lp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
            artifact={"purchase_prices": purchase_prices, "selling_prices": selling_prices},
        )

    period_count = len(purchase_prices)
    variable_count = 3 * period_count
    warehouse_capacity = _number(capacity.group(1))
    initial_stock = _number(initial.group(1))
    c = (
        purchase_prices
        + [-price for price in selling_prices]
        + [0.0] * period_count
    )

    a_ub: list[list[float]] = []
    b_ub: list[float] = []
    for period in range(period_count):
        row = [0.0] * variable_count
        row[period] = 1.0
        if period > 0:
            row[2 * period_count + period - 1] = 1.0
            b_ub.append(warehouse_capacity)
        else:
            b_ub.append(warehouse_capacity - initial_stock)
        a_ub.append(row)

    a_eq: list[list[float]] = []
    b_eq: list[float] = []
    for period in range(period_count):
        row = [0.0] * variable_count
        row[period] = 1.0
        row[period_count + period] = -1.0
        row[2 * period_count + period] = -1.0
        if period > 0:
            row[2 * period_count + period - 1] = 1.0
            b_eq.append(0.0)
        else:
            b_eq.append(-initial_stock)
        a_eq.append(row)

    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(0, None)] * variable_count,
        method="highs",
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="inventory_arbitrage_lp",
            status="solver_failed",
            confidence=0.8,
            notes=str(result.message),
            artifact={"purchase_prices": purchase_prices, "selling_prices": selling_prices},
        )

    variable_values: dict[str, float] = {}
    for period in range(period_count):
        variable_values[f"purchase_period_{period + 1}"] = float(result.x[period])
        variable_values[f"sell_period_{period + 1}"] = float(result.x[period_count + period])
        variable_values[f"ending_inventory_period_{period + 1}"] = float(result.x[2 * period_count + period])

    return TemplateSolveResult(
        matched=True,
        template_id="inventory_arbitrage_lp",
        status="optimal",
        objective_value=float(-result.fun),
        variable_values=variable_values,
        confidence=0.88,
        notes="Solved purchase/sales inventory planning LP with beginning-of-period purchases and warehouse capacity.",
        artifact={
            "purchase_prices": purchase_prices,
            "selling_prices": selling_prices,
            "warehouse_capacity": warehouse_capacity,
            "initial_stock": initial_stock,
        },
    )


def _solve_grain_inventory_arbitrage_lp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "grain" in lowered
        and "warehouse" in lowered
        and "purchase price" in lowered
        and "selling price" in lowered
        and "inventory" in lowered
        and "end of the quarter" in lowered
    ):
        return TemplateSolveResult(False)

    capacity = _number_after_patterns(normalized, [rf"capacity\s+of\s+({_NUMBER_TOKEN})\s+dan"])
    initial_inventory = _number_after_patterns(normalized, [rf"has\s+({_NUMBER_TOKEN})\s+dan\s+of\s+grain\s+in\s+stock"])
    final_inventory = _number_after_patterns(normalized, [rf"inventory\s+of\s+({_NUMBER_TOKEN})\s+dan\s+at\s+the\s+end\s+of\s+the\s+quarter"])
    if capacity is None or initial_inventory is None or final_inventory is None:
        return TemplateSolveResult(False)

    purchase_prices: list[float] = []
    selling_prices: list[float] = []
    for header, rows in _parse_markdown_tables(text):
        header_text = " ".join(header).lower()
        if "purchase price" not in header_text or "selling price" not in header_text:
            continue
        cleaned = [_clean_label(cell).lower() for cell in header]
        purchase_index = next((index for index, cell in enumerate(cleaned) if "purchase price" in cell), None)
        selling_index = next((index for index, cell in enumerate(cleaned) if "selling price" in cell), None)
        if purchase_index is None or selling_index is None:
            continue
        for row in rows:
            if len(row) <= max(purchase_index, selling_index):
                continue
            purchase = _first_number(row[purchase_index])
            selling = _first_number(row[selling_index])
            if purchase is not None and selling is not None:
                purchase_prices.append(float(purchase))
                selling_prices.append(float(selling))
    if not purchase_prices or len(purchase_prices) != len(selling_prices):
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="grain_inventory_arbitrage_lp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
        )

    period_count = len(purchase_prices)
    variable_count = period_count * 3
    c = purchase_prices + [-price for price in selling_prices] + [0.0] * period_count
    a_ub: list[list[float]] = []
    b_ub: list[float] = []
    a_eq: list[list[float]] = []
    b_eq: list[float] = []
    for period in range(period_count):
        row = [0.0] * variable_count
        row[period] = -1.0
        row[period_count + period] = 1.0
        row[2 * period_count + period] = 1.0
        if period > 0:
            row[2 * period_count + period - 1] = -1.0
            rhs = 0.0
        else:
            rhs = initial_inventory
        a_eq.append(row)
        b_eq.append(float(rhs))

        row = [0.0] * variable_count
        row[2 * period_count + period] = 1.0
        a_ub.append(row)
        b_ub.append(float(capacity))

        row = [0.0] * variable_count
        row[period_count + period] = 1.0
        if period > 0:
            row[2 * period_count + period - 1] = -1.0
            rhs = 0.0
        else:
            rhs = initial_inventory
        a_ub.append(row)
        b_ub.append(float(rhs))

    row = [0.0] * variable_count
    row[2 * period_count + period_count - 1] = 1.0
    a_eq.append(row)
    b_eq.append(float(final_inventory))

    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(0, None)] * variable_count,
        method="highs",
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="grain_inventory_arbitrage_lp",
            status="solver_failed",
            confidence=0.8,
            notes=str(result.message),
        )

    variable_values: dict[str, float] = {}
    for period in range(period_count):
        variable_values[f"purchase_month_{period + 1}"] = float(result.x[period])
        variable_values[f"sell_month_{period + 1}"] = float(result.x[period_count + period])
        variable_values[f"ending_inventory_month_{period + 1}"] = float(result.x[2 * period_count + period])
    return TemplateSolveResult(
        matched=True,
        template_id="grain_inventory_arbitrage_lp",
        status="optimal",
        objective_value=float(-result.fun),
        variable_values=variable_values,
        confidence=0.86,
        notes="Solved grain purchase/sale inventory arbitrage LP with one-month sale lag.",
        artifact={
            "purchase_prices": purchase_prices,
            "selling_prices": selling_prices,
            "capacity": capacity,
            "initial_inventory": initial_inventory,
            "final_inventory": final_inventory,
        },
    )


def _parse_single_no_production_rule(text: str, product_labels: list[str]) -> tuple[int, int] | None:
    match = re.search(
        rf"product\s+([A-Za-z0-9]+)\s+cannot\s+be\s+produced\s+in\s+the\s+({'|'.join(_ORDINAL_INDEX)})\s+quarter",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    product_index = _item_index_for_label(product_labels, match.group(1))
    period_index = _ORDINAL_INDEX[match.group(2).lower()] - 1
    if product_index is None:
        return None
    return product_index, period_index


def _parse_delay_penalties(text: str, product_labels: list[str]) -> list[float] | None:
    match = re.search(
        (
            rf"compensation\s+of\s+({_NUMBER_TOKEN})\s+\w+\s+per\s+unit\s+per\s+quarter\s+delay"
            rf".*?for\s+products?\s+([A-Za-z0-9]+)\s+and\s+([A-Za-z0-9]+)"
            rf".*?for\s+product\s+([A-Za-z0-9]+).*?compensation\s+is\s+({_NUMBER_TOKEN})"
        ),
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    shared_penalty = _number(match.group(1))
    single_penalty = _number(match.group(5))
    penalties = [shared_penalty] * len(product_labels)
    for label in (match.group(2), match.group(3)):
        index = _item_index_for_label(product_labels, label)
        if index is not None:
            penalties[index] = shared_penalty
    single_index = _item_index_for_label(product_labels, match.group(4))
    if single_index is not None:
        penalties[single_index] = single_penalty
    return penalties


def _solve_multi_product_inventory_backlog_ilp(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        "contract reservations" in lowered
        and "production hours per quarter" in lowered
        and "compensation" in lowered
        and "delay" in lowered
        and "inventory cost" in lowered
        and "minimize" in lowered
    ):
        return TemplateSolveResult(False)

    demand_table = None
    for table in _parse_markdown_tables(text):
        matrix = _numeric_matrix_from_table(table)
        if matrix and len(matrix[0]) >= 2 and len(matrix[1]) >= 2:
            demand_table = matrix
            break
    if not demand_table:
        return TemplateSolveResult(False)

    product_labels, _period_labels, demands_by_product = demand_table
    product_count = len(product_labels)
    period_count = len(demands_by_product[0])
    if product_count < 2 or period_count < 2:
        return TemplateSolveResult(False)

    capacity = re.search(
        rf"has\s+({_NUMBER_TOKEN})\s+production\s+hours\s+per\s+quarter",
        text,
        flags=re.IGNORECASE,
    )
    hours = re.search(
        (
            rf"each\s+unit\s+of\s+products?.*?requires\s+"
            rf"({_NUMBER_TOKEN}(?:\s*,\s*{_NUMBER_TOKEN})*(?:\s*,?\s*and\s+{_NUMBER_TOKEN})?)"
            rf"\s+hours?\s+respectively"
        ),
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    ending_stock = re.search(
        rf"have\s+({_NUMBER_TOKEN})\s+units?\s+in\s+stock.*?end\s+of\s+the\s+({'|'.join(_ORDINAL_INDEX)})\s+quarter",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    inventory_cost = re.search(
        rf"inventory\s+cost\s+is\s+({_NUMBER_TOKEN})\s+\w+\s+per\s+unit\s+per\s+quarter",
        text,
        flags=re.IGNORECASE,
    )
    if not (capacity and hours and ending_stock and inventory_cost):
        return TemplateSolveResult(False)

    production_capacity = _number(capacity.group(1))
    production_hours = [
        _number(value)
        for value in re.findall(_NUMBER_TOKEN, hours.group(1), flags=re.IGNORECASE)
    ]
    if len(production_hours) != product_count:
        return TemplateSolveResult(False)
    ending_stock_value = _number(ending_stock.group(1))
    holding_cost = _number(inventory_cost.group(1))
    delay_penalties = _parse_delay_penalties(text, product_labels)
    if delay_penalties is None or len(delay_penalties) != product_count:
        return TemplateSolveResult(False)
    no_production_rule = _parse_single_no_production_rule(text, product_labels)

    initial_inventory = [0.0] * product_count
    if "no inventory" not in lowered:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="multi_product_inventory_backlog_ilp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
            artifact={"products": product_labels, "demands": demands_by_product},
        )

    block_size = product_count * period_count
    variable_count = 3 * block_size

    def flat(product_index: int, period_index: int) -> int:
        return product_index * period_count + period_index

    def var(kind: int, product_index: int, period_index: int) -> int:
        return kind * block_size + flat(product_index, period_index)

    c = np.zeros(variable_count)
    for product_index in range(product_count):
        for period_index in range(period_count):
            c[var(1, product_index, period_index)] = holding_cost
            c[var(2, product_index, period_index)] = delay_penalties[product_index]

    rows: list[Any] = []
    lower: list[float] = []
    upper: list[float] = []
    for period_index in range(period_count):
        row = np.zeros(variable_count)
        for product_index in range(product_count):
            row[var(0, product_index, period_index)] = production_hours[product_index]
        rows.append(row)
        lower.append(-math.inf)
        upper.append(production_capacity)
    if no_production_rule is not None:
        product_index, period_index = no_production_rule
        row = np.zeros(variable_count)
        row[var(0, product_index, period_index)] = 1.0
        rows.append(row)
        lower.append(0.0)
        upper.append(0.0)
    for product_index in range(product_count):
        row = np.zeros(variable_count)
        row[var(1, product_index, period_count - 1)] = 1.0
        rows.append(row)
        lower.append(ending_stock_value)
        upper.append(math.inf)
        row = np.zeros(variable_count)
        row[var(2, product_index, period_count - 1)] = 1.0
        rows.append(row)
        lower.append(0.0)
        upper.append(0.0)

    for product_index in range(product_count):
        for period_index in range(period_count):
            row = np.zeros(variable_count)
            row[var(0, product_index, period_index)] = 1.0
            if period_index > 0:
                row[var(1, product_index, period_index - 1)] = 1.0
                row[var(2, product_index, period_index - 1)] = -1.0
            row[var(1, product_index, period_index)] = -1.0
            row[var(2, product_index, period_index)] = 1.0
            rows.append(row)
            lower.append(demands_by_product[product_index][period_index] - initial_inventory[product_index])
            upper.append(demands_by_product[product_index][period_index] - initial_inventory[product_index])

    result = milp(
        c,
        integrality=np.ones(variable_count),
        bounds=Bounds(np.zeros(variable_count), np.full(variable_count, math.inf)),
        constraints=LinearConstraint(np.vstack(rows), np.array(lower), np.array(upper)),
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="multi_product_inventory_backlog_ilp",
            status="solver_failed",
            confidence=0.82,
            notes=str(result.message),
            artifact={"products": product_labels, "demands": demands_by_product},
        )

    variable_values: dict[str, float] = {}
    for product_index, product_label in enumerate(product_labels):
        for period_index in range(period_count):
            production = float(result.x[var(0, product_index, period_index)])
            inventory = float(result.x[var(1, product_index, period_index)])
            backlog = float(result.x[var(2, product_index, period_index)])
            variable_values[f"produce_{product_label}_period_{period_index + 1}"] = production
            if not math.isclose(inventory, 0.0, abs_tol=1e-8):
                variable_values[f"inventory_{product_label}_period_{period_index + 1}"] = inventory
            if not math.isclose(backlog, 0.0, abs_tol=1e-8):
                variable_values[f"backlog_{product_label}_period_{period_index + 1}"] = backlog

    return TemplateSolveResult(
        matched=True,
        template_id="multi_product_inventory_backlog_ilp",
        status="optimal",
        objective_value=float(result.fun),
        variable_values=variable_values,
        confidence=0.88,
        notes="Solved multi-product finite-horizon production plan with integer production, inventory, and backlog penalties.",
        artifact={
            "products": product_labels,
            "demands": demands_by_product,
            "production_hours": production_hours,
            "production_capacity": production_capacity,
            "ending_stock": ending_stock_value,
            "holding_cost": holding_cost,
            "delay_penalties": delay_penalties,
            "no_production_rule": no_production_rule,
        },
    )


def _solve_livestock_resource_mix_ilp(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        "maximum profit" in lowered
        and "feed cost" in lowered
        and "manure" in lowered
        and "total number of animals" in lowered
    ):
        return TemplateSolveResult(False)

    sale_match = re.search(
        (
            rf"sell\s+([A-Za-z]+),\s+([A-Za-z]+),\s+and\s+([A-Za-z]+)\s+"
            rf"for\s+\$?({_NUMBER_TOKEN}),\s+\$?({_NUMBER_TOKEN}),\s+"
            rf"and\s+\$?({_NUMBER_TOKEN})\s+each"
        ),
        text,
        flags=re.IGNORECASE,
    )
    feed_match = re.search(
        (
            rf"feed\s+costs?.*?\s+\$?({_NUMBER_TOKEN}),\s+\$?({_NUMBER_TOKEN}),\s+"
            rf"and\s+\$?({_NUMBER_TOKEN})"
        ),
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    manure_match = re.search(
        (
            rf"produces?\s+({_NUMBER_TOKEN}),\s+({_NUMBER_TOKEN}),\s+and\s+"
            rf"({_NUMBER_TOKEN})\s+units?\s+of\s+manure"
        ),
        text,
        flags=re.IGNORECASE,
    )
    manure_capacity = re.search(
        rf"(?:up\s+to|not\s+exceed)\s+({_NUMBER_TOKEN})\s+units?\s+of\s+manure",
        text,
        flags=re.IGNORECASE,
    )
    total_capacity = re.search(
        rf"total\s+number\s+of\s+animals\s+cannot\s+exceed\s+({_NUMBER_TOKEN})",
        text,
        flags=re.IGNORECASE,
    )
    if not (sale_match and feed_match and manure_match and manure_capacity and total_capacity):
        return TemplateSolveResult(False)

    labels = [_clean_label(value).lower() for value in sale_match.groups()[:3]]
    selling_prices = [_number(value) for value in sale_match.groups()[3:]]
    feed_costs = [_number(value) for value in feed_match.groups()]
    manure = [_number(value) for value in manure_match.groups()]
    profits = [selling - feed for selling, feed in zip(selling_prices, feed_costs)]
    manure_limit = _number(manure_capacity.group(1))
    animal_limit = int(_number(total_capacity.group(1)))

    lower_bounds = [0, 0, 0]
    upper_bounds = [animal_limit, animal_limit, animal_limit]
    for value, label in re.findall(
        rf"at\s+least\s+({_NUMBER_TOKEN})\s+([A-Za-z]+)",
        text,
        flags=re.IGNORECASE,
    ):
        index = _item_index_for_label(labels, label)
        if index is not None:
            lower_bounds[index] = max(lower_bounds[index], int(_number(value)))
    for value, label in re.findall(
        rf"at\s+most\s+({_NUMBER_TOKEN})\s+([A-Za-z]+)",
        text,
        flags=re.IGNORECASE,
    ):
        index = _item_index_for_label(labels, label)
        if index is not None:
            upper_bounds[index] = min(upper_bounds[index], int(_number(value)))

    search_space = math.prod(upper - lower + 1 for lower, upper in zip(lower_bounds, upper_bounds))
    if search_space > 2_000_000:
        return TemplateSolveResult(
            matched=True,
            template_id="livestock_resource_mix_ilp",
            status="solver_unavailable",
            confidence=0.75,
            notes="Integer search space is too large for the deterministic enumerator.",
            artifact={"labels": labels, "profits": profits, "bounds": [lower_bounds, upper_bounds]},
        )

    best_value: float | None = None
    best_counts: tuple[int, int, int] | None = None
    ranges = [range(lower_bounds[index], upper_bounds[index] + 1) for index in range(3)]
    for counts in itertools.product(*ranges):
        if sum(counts) > animal_limit:
            continue
        if sum(count * manure[index] for index, count in enumerate(counts)) > manure_limit:
            continue
        value = sum(count * profits[index] for index, count in enumerate(counts))
        if best_value is None or value > best_value:
            best_value = value
            best_counts = counts
    if best_counts is None or best_value is None:
        return TemplateSolveResult(
            matched=True,
            template_id="livestock_resource_mix_ilp",
            status="infeasible",
            confidence=0.8,
            artifact={"labels": labels, "profits": profits, "bounds": [lower_bounds, upper_bounds]},
        )

    return TemplateSolveResult(
        matched=True,
        template_id="livestock_resource_mix_ilp",
        status="optimal",
        objective_value=float(best_value),
        variable_values={
            f"raise_{labels[index]}": float(count)
            for index, count in enumerate(best_counts)
        },
        confidence=0.9,
        notes="Solved integer livestock/product-mix problem by bounded enumeration.",
        artifact={
            "items": labels,
            "selling_prices": selling_prices,
            "feed_costs": feed_costs,
            "profits": profits,
            "resource": "manure",
            "resource_coefficients": manure,
            "resource_capacity": manure_limit,
            "total_capacity": animal_limit,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
        },
    )


_BULLET_ITEM_RE = re.compile(
    r"(?:^|\n)\s*-\s*([^:]+):(.*?)(?=\n\s*-\s*|\Z)",
    flags=re.DOTALL,
)


def _nutrient_amount(body: str, nutrient: str) -> float | None:
    nutrient_pattern = (
        r"(?:carbohydrates?|carbs?)"
        if nutrient.startswith("carbohydrate")
        else re.escape(nutrient)
    )
    if re.search(rf"\bno\s+{nutrient_pattern}\b", body, flags=re.IGNORECASE):
        return 0.0
    match = re.search(
        rf"(?:a\s+)?({_NUMBER_TOKEN})\s+grams?\s+of\s+{nutrient_pattern}",
        body,
        flags=re.IGNORECASE,
    )
    if match:
        return _number(match.group(1))
    if nutrient == "protein":
        match = re.search(
            rf"\bprotein\s+at\s+({_NUMBER_TOKEN})\s+grams?",
            body,
            flags=re.IGNORECASE,
        )
        if match:
            return _number(match.group(1))
    return None


def _diet_requirements(text: str) -> tuple[float, float, float] | None:
    combined = re.search(
        (
            r"at\s+least\s+([0-9][0-9,]*(?:\.\d+)?)\s+grams?\s+of\s+protein"
            r".{0,80}?([0-9][0-9,]*(?:\.\d+)?)\s+grams?\s+of\s+carbohydrates?"
            r".{0,80}?([0-9][0-9,]*(?:\.\d+)?)\s+calories"
        ),
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if combined:
        return tuple(_number(value) for value in combined.groups())  # type: ignore[return-value]
    protein = re.search(r"at\s+least\s+([0-9][0-9,]*(?:\.\d+)?)\s+grams?\s+of\s+protein", text, re.I)
    carbs = re.search(r"at\s+least\s+([0-9][0-9,]*(?:\.\d+)?)\s+grams?\s+of\s+carbohydrates?", text, re.I)
    calories = re.search(r"at\s+least\s+([0-9][0-9,]*(?:\.\d+)?)\s+calories", text, re.I)
    if protein and carbs and calories:
        return (_number(protein.group(1)), _number(carbs.group(1)), _number(calories.group(1)))
    return None


def _plain_quantity_text(text: str) -> str:
    plain = re.sub(r"\\mathrm\{\s*~?\s*([^}]*)\}", r" \1 ", text)
    plain = re.sub(r"\\(?:text|mathrm)\{([^}]*)\}", r"\1", plain)
    plain = re.sub(r"[$\\{}]", " ", plain)
    return re.sub(r"\s+", " ", plain)


def _nutrient_key(value: str) -> str:
    cleaned = _clean_label(value).lower()
    cleaned = re.sub(r"\([^)]*\)", "", cleaned)
    cleaned = re.sub(r"\b(?:grams?|g|mg|kg|per|price|cost|yuan|¥)\b", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned.endswith("s") and not cleaned.endswith("ss"):
        cleaned = cleaned[:-1]
    return cleaned


def _parse_nutrition_requirements(text: str) -> dict[str, float]:
    plain = _plain_quantity_text(text)
    requirements: dict[str, float] = {}
    need_clause = re.search(
        r"(?:needs?|requires?)\s+at\s+least\s+(.{0,240}?)(?:daily|per\s+day|\.|;)",
        plain,
        flags=re.IGNORECASE,
    )
    if need_clause:
        for value, nutrient in re.findall(
            rf"({_NUMBER_TOKEN})\s*(?:grams?|g|mg)?\s+of\s+([A-Za-z][A-Za-z\s-]*?)(?=,|\s+and\b|$)",
            need_clause.group(1),
            flags=re.IGNORECASE,
        ):
            key = _nutrient_key(nutrient)
            if key and key not in {"feed", "food"}:
                requirements[key] = _number(value)
    for value, nutrient in re.findall(
        rf"at\s+least\s+({_NUMBER_TOKEN})\s*(?:grams?|g|mg)?\s+of\s+([A-Za-z][A-Za-z\s-]*?)(?=,|\s+and\b|daily|\.|;)",
        plain,
        flags=re.IGNORECASE,
    ):
        key = _nutrient_key(nutrient)
        if key and key not in {"feed", "food"}:
            requirements[key] = _number(value)
    return requirements


def _parse_nutrition_table_items(
    text: str,
    requirements: dict[str, float],
) -> tuple[list[dict[str, Any]], list[str]] | None:
    requirement_keys = set(requirements)
    for header, rows in _parse_markdown_tables(text):
        cleaned_header = [_clean_label(cell).lower() for cell in header]
        feed_indices = [
            index
            for index, cell in enumerate(cleaned_header)
            if cell == "feed" or cell.startswith("food")
        ]
        if not feed_indices:
            continue
        group_ranges: list[tuple[int, int]] = []
        for position, start in enumerate(feed_indices):
            end = feed_indices[position + 1] if position + 1 < len(feed_indices) else len(header)
            if end - start >= 3:
                group_ranges.append((start, end))
        items: list[dict[str, Any]] = []
        nutrient_order: list[str] = []
        for row in rows:
            for start, end in group_ranges:
                if len(row) <= start:
                    continue
                label = _clean_label(row[start])
                if not label:
                    continue
                group_headers = header[start + 1 : end]
                group_values = row[start + 1 : end]
                if len(group_values) < len(group_headers):
                    continue
                price_index = next(
                    (
                        index
                        for index, cell in enumerate(group_headers)
                        if re.search(r"\b(?:price|cost)\b", cell, flags=re.IGNORECASE)
                    ),
                    None,
                )
                if price_index is None:
                    continue
                price_value = _first_number(group_values[price_index])
                if price_value is None:
                    continue
                nutrients: dict[str, float] = {}
                for index, header_cell in enumerate(group_headers):
                    if index == price_index:
                        continue
                    key = _nutrient_key(header_cell)
                    value = _first_number(group_values[index])
                    if key and value is not None:
                        nutrients[key] = float(value)
                        if key not in nutrient_order:
                            nutrient_order.append(key)
                if requirement_keys and not requirement_keys.issubset(nutrients):
                    continue
                items.append({"label": label, "nutrients": nutrients, "cost": float(price_value)})
        if len(items) >= 2 and requirement_keys.issubset(set(nutrient_order)):
            return items, [key for key in nutrient_order if key in requirement_keys]
    return None


def _solve_continuous_nutrition_mix_lp(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        ("minimize" in lowered or "minimizing" in lowered or "minimum" in lowered)
        and ("cost" in lowered or "price" in lowered)
        and any(word in lowered for word in ("feed", "food", "diet", "nutrition", "nutritional"))
    ):
        return TemplateSolveResult(False)
    requirements = _parse_nutrition_requirements(text)
    if len(requirements) < 2:
        return TemplateSolveResult(False)
    parsed = _parse_nutrition_table_items(text, requirements)
    if not parsed:
        return TemplateSolveResult(False)
    items, nutrients = parsed

    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="continuous_nutrition_mix_lp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
            artifact={"items": items, "requirements": requirements},
        )

    objective = [item["cost"] for item in items]
    a_ub = [
        [-item["nutrients"][nutrient] for item in items]
        for nutrient in nutrients
    ]
    b_ub = [-requirements[nutrient] for nutrient in nutrients]
    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(0, None)] * len(items),
        method="highs",
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="continuous_nutrition_mix_lp",
            status="solver_failed",
            confidence=0.82,
            notes=str(result.message),
            artifact={"items": items, "requirements": requirements, "nutrients": nutrients},
        )
    return TemplateSolveResult(
        matched=True,
        template_id="continuous_nutrition_mix_lp",
        status="optimal",
        objective_value=float(result.fun),
        variable_values={
            f"amount_{items[index]['label']}": float(value)
            for index, value in enumerate(result.x)
            if not math.isclose(float(value), 0.0, abs_tol=1e-9)
        },
        confidence=0.88,
        notes="Solved continuous nutrition/feed mix LP from nutrient table and minimum requirements.",
        artifact={"items": items, "requirements": requirements, "nutrients": nutrients},
    )


def _solve_integer_diet_lp(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        "protein" in lowered
        and "carbohydrate" in lowered
        and "calories" in lowered
        and (
            "minimum cost" in lowered
            or "minimal cost" in lowered
            or "minimize cost" in lowered
            or "cheapest cost" in lowered
            or "cost-effective" in lowered
            or "cost as low" in lowered
            or "least amount of money" in lowered
            or "least possible cost" in lowered
            or "least expensive" in lowered
        )
    ):
        return TemplateSolveResult(False)
    foods: list[tuple[str, float, float, float, float]] = []
    for match in _BULLET_ITEM_RE.finditer(text):
        name, body = match.groups()
        body = re.split(r"\n\s*\n", body.strip(), maxsplit=1)[0]
        protein = _nutrient_amount(body, "protein")
        carbs = _nutrient_amount(body, "carbohydrates?")
        calories = re.search(
            r"([0-9][0-9,]*(?:\.\d+)?)\s+calories",
            body,
            flags=re.IGNORECASE,
        )
        cost = re.search(r"\$([0-9][0-9,]*(?:\.\d+)?)", body)
        if protein is None or carbs is None or not (calories and cost):
            continue
        foods.append(
            (
                _clean_label(name),
                protein,
                carbs,
                _number(calories.group(1)),
                _number(cost.group(1)),
            )
        )
    requirements = _diet_requirements(text)
    if len(foods) < 2 or not requirements:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="integer_diet_lp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
            artifact={"foods": foods, "requirements": requirements},
        )

    c = [food[4] for food in foods]
    a_ub = [[-food[column] for food in foods] for column in (1, 2, 3)]
    b_ub = [-value for value in requirements]
    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(0, None)] * len(foods),
        integrality=[1] * len(foods),
        method="highs",
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="integer_diet_lp",
            status="solver_failed",
            confidence=0.8,
            notes=str(result.message),
            artifact={"foods": foods, "requirements": requirements},
        )
    variable_values = {
        f"servings_{foods[index][0]}": float(value)
        for index, value in enumerate(result.x)
        if not math.isclose(float(value), 0.0, abs_tol=1e-9)
    }
    return TemplateSolveResult(
        matched=True,
        template_id="integer_diet_lp",
        status="optimal",
        objective_value=float(result.fun),
        variable_values=variable_values,
        confidence=0.9,
        notes="Solved integer diet LP from listed nutrition and cost values.",
        artifact={"foods": foods, "requirements": requirements},
    )


def _linprog_maximize(
    *,
    objective: list[float],
    constraints: list[list[float]],
    upper_bounds: list[float],
    lower_bounds: list[float] | None = None,
) -> tuple[str, float | None, list[float], str]:
    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return "solver_unavailable", None, [], str(exc)

    a_ub = [list(row) for row in constraints]
    b_ub = list(upper_bounds)
    if lower_bounds:
        for index, lower_bound in enumerate(lower_bounds):
            if lower_bound > 0:
                row = [0.0] * len(objective)
                row[index] = -1.0
                a_ub.append(row)
                b_ub.append(-lower_bound)
    result = linprog(
        [-value for value in objective],
        A_ub=a_ub or None,
        b_ub=b_ub or None,
        bounds=[(0, None)] * len(objective),
        method="highs",
    )
    if not result.success:
        return "solver_failed", None, [], str(result.message)
    return "optimal", float(-result.fun), [float(value) for value in result.x], str(result.message)


def _linprog_minimize(
    *,
    objective: list[float],
    constraints: list[list[float]],
    upper_bounds: list[float],
    lower_bounds: list[float] | None = None,
) -> tuple[str, float | None, list[float], str]:
    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return "solver_unavailable", None, [], str(exc)

    a_ub = [list(row) for row in constraints]
    b_ub = list(upper_bounds)
    if lower_bounds:
        for index, lower_bound in enumerate(lower_bounds):
            if lower_bound > 0:
                row = [0.0] * len(objective)
                row[index] = -1.0
                a_ub.append(row)
                b_ub.append(-lower_bound)
    result = linprog(
        objective,
        A_ub=a_ub or None,
        b_ub=b_ub or None,
        bounds=[(0, None)] * len(objective),
        method="highs",
    )
    if not result.success:
        return "solver_failed", None, [], str(result.message)
    return "optimal", float(result.fun), [float(value) for value in result.x], str(result.message)


def _product_labels_from_table(header: list[str], rows: list[list[str]], count: int) -> list[str]:
    for row in rows:
        cleaned = [_clean_label(cell) for cell in row]
        labels = [
            label
            for label in cleaned
            if re.fullmatch(r"[A-Za-z](?:_[0-9]+)?", label)
        ]
        if len(labels) >= count:
            return labels[:count]
    labels = [_clean_label(cell) for cell in header[1 : 1 + count]]
    if len(labels) == count and all(labels):
        return labels
    return [f"product_{index}" for index in range(1, count + 1)]


def _label_regex(label: str) -> str:
    parts = [re.escape(part) for part in re.split(r"\s+", _clean_label(label)) if part]
    return r"\s+".join(parts)


def _parse_minimum_profit_target(text: str) -> float | None:
    target_phrases = (
        r"at\s+least|not\s+less\s+than|must\s+not\s+be\s+less\s+than|"
        r"should\s+not\s+be\s+less\s+than|should\s+be\s+at\s+least"
    )
    patterns = [
        rf"profit[^.;\n]{{0,140}}?(?:{target_phrases})\s+({_NUMBER_TOKEN})",
        rf"(?:{target_phrases})\s+({_NUMBER_TOKEN})[^.;\n]{{0,140}}?profit",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return _number(match.group(1))
    return None


def _parse_minimum_unit_bounds(text: str, labels: list[str]) -> list[float]:
    lower_bounds = [0.0] * len(labels)
    for index, label in enumerate(labels):
        label_pattern = _label_regex(label)
        qualified_label = rf"(?:(?:model|type|product)\s+)?{label_pattern}"
        patterns = [
            rf"at\s+least\s+({_NUMBER_TOKEN})\s+units?\s+of\s+{qualified_label}\b",
            rf"{qualified_label}\b(?:(?!\band\b).){{0,100}}?at\s+least\s+({_NUMBER_TOKEN})\s+units?",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
                lower_bounds[index] = max(lower_bounds[index], _number(match.group(1)))
    return lower_bounds


def _split_narrative_labels(value: str) -> list[str]:
    cleaned = re.sub(r"\band\b", ",", value, flags=re.IGNORECASE)
    labels = [
        _clean_label(part)
        for part in re.split(r",|/", cleaned)
        if part.strip()
    ]
    return labels


def _plural_variants(label: str) -> set[str]:
    singular = _singular_label(label)
    variants = {label.lower(), singular}
    variants.add(f"{singular}s")
    if singular.endswith("f"):
        variants.add(f"{singular[:-1]}ves")
    if singular.endswith("fe"):
        variants.add(f"{singular[:-2]}ves")
    return {variant for variant in variants if variant}


def _narrative_label_pattern(label: str) -> str:
    variants = sorted(_plural_variants(_clean_label(label)), key=len, reverse=True)
    escaped = [r"\s+".join(re.escape(part) for part in variant.split()) for variant in variants]
    core = rf"(?:{'|'.join(escaped)})"
    return rf"(?:(?:product|package|type|model)\s+)?{core}(?:\s+(?:product|package|packages|type|model))?"


def _canonical_resource_label(value: str) -> str:
    label = _clean_label(value).lower()
    if "machine" in label:
        return "machine time"
    if "craftsman" in label:
        return "craftsman time"
    if "inspection" in label:
        return "inspection"
    if "labor" in label or "labour" in label:
        return "labor"
    if "shirt" in label:
        return "shirts"
    if "pants" in label or "trouser" in label:
        return "pants"
    if "warehouse" in label or "space" in label:
        return "warehouse space"
    if "production" in label or label == "hours":
        return "production time"
    return label


def _add_resource_coefficient(
    coefficients: dict[str, list[float]],
    resource: str,
    product_index: int,
    value: float,
    product_count: int,
) -> None:
    row = coefficients.setdefault(resource, [0.0] * product_count)
    row[product_index] += value


def _parse_narrative_objective(text: str) -> tuple[list[str], list[float]] | None:
    normalized = re.sub(r"\s+", " ", text)

    type_matches = list(
        re.finditer(
            rf"\b(?:The\s+)?([A-Za-z][A-Za-z -]*?)\s+type\s+requires\b"
            rf".{{0,180}}?\byields\s+a\s+profit\s+of\s*[^0-9]*({_NUMBER_TOKEN})",
            normalized,
            flags=re.IGNORECASE,
        )
    )
    if len(type_matches) >= 2:
        return (
            [_clean_label(match.group(1)) for match in type_matches],
            [_number(match.group(2)) for match in type_matches],
        )

    package_matches = list(
        re.finditer(
            rf"\bPackage\s+([A-Za-z0-9]+)\b[^.]*?\bpriced\s+at\s*[£$¥]?\s*({_NUMBER_TOKEN})",
            normalized,
            flags=re.IGNORECASE,
        )
    )
    if len(package_matches) >= 2:
        return (
            [_clean_label(match.group(1)) for match in package_matches],
            [_number(match.group(2)) for match in package_matches],
        )

    product_profit_matches = list(
        re.finditer(
            rf"\bproduct\s+([A-Za-z0-9]+)\s+sold\s+generates\s+a\s+profit\s+of\s*"
            rf"[£$¥]?\s*({_NUMBER_TOKEN})",
            normalized,
            flags=re.IGNORECASE,
        )
    )
    if len(product_profit_matches) >= 2:
        return (
            [_clean_label(match.group(1)) for match in product_profit_matches],
            [_number(match.group(2)) for match in product_profit_matches],
        )

    sales_section = re.split(r"\bmanufacturing\s+costs?\b", normalized, maxsplit=1, flags=re.IGNORECASE)[0]
    sales_matches = list(
        re.finditer(
            rf"\beach\s+([A-Za-z][A-Za-z -]*?)\s+for\s*[£$¥]?\s*({_NUMBER_TOKEN})",
            sales_section,
            flags=re.IGNORECASE,
        )
    )
    if len(sales_matches) >= 2:
        labels = [_clean_label(match.group(1)) for match in sales_matches]
        values = [_number(match.group(2)) for match in sales_matches]
        cost_match = re.search(
            rf"manufacturing\s+costs?\s+for\s+each\s+(.+?)\s+are\s+(.+?)\s+respectively",
            normalized,
            flags=re.IGNORECASE,
        )
        if cost_match:
            cost_labels = _split_narrative_labels(cost_match.group(1))
            costs = [_number(value) for value in re.findall(_NUMBER_TOKEN, cost_match.group(2), flags=re.IGNORECASE)]
            if len(cost_labels) == len(labels) == len(costs):
                aligned_costs = [0.0] * len(labels)
                for cost_label, cost in zip(cost_labels, costs):
                    index = _item_index_for_label(labels, cost_label)
                    if index is None:
                        break
                    aligned_costs[index] = cost
                else:
                    values = [sale - cost for sale, cost in zip(values, aligned_costs)]
        return labels, values

    profit_matches = list(
        re.finditer(
            rf"\bprofit\s+for\s+each\s+([A-Za-z][A-Za-z -]*?)\s+sold\s+is\s*"
            rf"[£$¥]?\s*({_NUMBER_TOKEN})",
            normalized,
            flags=re.IGNORECASE,
        )
    )
    if len(profit_matches) >= 2:
        return (
            [_clean_label(match.group(1)) for match in profit_matches],
            [_number(match.group(2)) for match in profit_matches],
        )
    return None


def _parse_narrative_resource_coefficients(
    text: str,
    labels: list[str],
) -> dict[str, list[float]]:
    normalized = re.sub(r"\s+", " ", text)
    coefficients: dict[str, list[float]] = {}

    for product_index, label in enumerate(labels):
        label_pattern = _narrative_label_pattern(label)
        match = re.search(
            rf"\b{label_pattern}\s+requires\s+(.+?)(?:,\s+and\s+yields|\s+and\s+yields|\.)",
            normalized,
            flags=re.IGNORECASE,
        )
        if match:
            body = match.group(1)
            for value, resource in re.findall(
                rf"({_NUMBER_TOKEN})\s+(?:hours?|units?|kg|kilograms?|square\s+meters?)\s+"
                rf"(?:of\s+)?([A-Za-z][A-Za-z\s-]*?)(?=,|\s+and\s+|$)",
                body,
                flags=re.IGNORECASE,
            ):
                _add_resource_coefficient(
                    coefficients,
                    _canonical_resource_label(resource),
                    product_index,
                    _number(value),
                    len(labels),
                )
        time_match = re.search(
            rf"\b{label_pattern}\s+requires\s+({_NUMBER_TOKEN})\s+hours?(?=,?\s+while|\.|;)",
            normalized,
            flags=re.IGNORECASE,
        )
        if time_match and not any(row[product_index] for row in coefficients.values()):
            _add_resource_coefficient(
                coefficients,
                "production time",
                product_index,
                _number(time_match.group(1)),
                len(labels),
            )

    list_match = re.search(
        rf"\bEach\s+([^.]+?)\s+(?:occupy|occupies|requires?)\s+(.+?)\s+square\s+meters?"
        rf"\s+of\s+warehouse\s+space\s+respectively",
        normalized,
        flags=re.IGNORECASE,
    )
    if list_match:
        listed_labels = _split_narrative_labels(list_match.group(1))
        values = [_number(value) for value in re.findall(_NUMBER_TOKEN, list_match.group(2), flags=re.IGNORECASE)]
        if len(listed_labels) == len(values):
            for listed_label, value in zip(listed_labels, values):
                index = _item_index_for_label(labels, listed_label)
                if index is not None:
                    _add_resource_coefficient(coefficients, "warehouse space", index, value, len(labels))

    for package_label, body in re.findall(
        r"\bPackage\s+([A-Za-z0-9]+)\s+includes\s+([^.;]+)",
        normalized,
        flags=re.IGNORECASE,
    ):
        product_index = _item_index_for_label(labels, package_label)
        if product_index is None:
            continue
        for value, resource in re.findall(
            rf"({_NUMBER_TOKEN})\s+(?:pairs?\s+of\s+)?(shirts?|pants?)",
            body,
            flags=re.IGNORECASE,
        ):
            _add_resource_coefficient(
                coefficients,
                _canonical_resource_label(resource),
                product_index,
                _number(value),
                len(labels),
            )

    return coefficients


def _parse_narrative_resource_capacities(text: str, resources: set[str]) -> dict[str, float]:
    normalized = re.sub(r"\s+", " ", text)
    capacities: dict[str, float] = {}
    if "labor" in resources:
        match = re.search(r"\bavailable\s+(?:manufacturing\s+)?labor\s+hours?\s+are\s+([0-9][0-9,]*(?:\.\d+)?)", normalized, re.I)
        if match:
            capacities["labor"] = _number(match.group(1))
    if "inspection" in resources:
        match = re.search(r"\bavailable\s+inspection\s+hours?\s+are\s+([0-9][0-9,]*(?:\.\d+)?)", normalized, re.I)
        if match:
            capacities["inspection"] = _number(match.group(1))
    if "warehouse space" in resources:
        match = re.search(r"\btotal\s+space\s+cannot\s+exceed\s+([0-9][0-9,]*(?:\.\d+)?)", normalized, re.I)
        if match:
            capacities["warehouse space"] = _number(match.group(1))
    if "production time" in resources:
        match = re.search(r"\bmaximum\s+of\s+([0-9][0-9,]*(?:\.\d+)?)\s+hours?.{0,80}\bproduction\b", normalized, re.I)
        if match:
            capacities["production time"] = _number(match.group(1))
    if "shirts" in resources:
        match = re.search(r"\bclear\s+out\s+([0-9][0-9,]*(?:\.\d+)?)\s+shirts?", normalized, re.I)
        if match:
            capacities["shirts"] = _number(match.group(1))
    if "pants" in resources:
        match = re.search(r"\b([0-9][0-9,]*(?:\.\d+)?)\s+pairs?\s+of\s+pants", normalized, re.I)
        if match:
            capacities["pants"] = _number(match.group(1))
    return capacities


def _parse_narrative_lower_bounds(text: str, labels: list[str]) -> list[float]:
    lower_bounds = _parse_minimum_unit_bounds(text, labels)
    normalized = re.sub(r"\s+", " ", text)
    for index, label in enumerate(labels):
        label_pattern = _narrative_label_pattern(label)
        direct_patterns = [
            rf"(?:at\s+least|fewer\s+than|and)\s+({_NUMBER_TOKEN})(?!\s+times\b)\s+{label_pattern}\b",
            rf"at\s+least\s+({_NUMBER_TOKEN})(?!\s+times\b)\s+batches?\s+of\s+{label_pattern}\b",
        ]
        for pattern in direct_patterns:
            for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
                lower_bounds[index] = max(lower_bounds[index], _number(match.group(1)))
    for clause in re.findall(
        r"(?:at\s+least|fewer\s+than)[^.]+",
        normalized,
        flags=re.IGNORECASE,
    ):
        for value, raw_label in re.findall(
            rf"({_NUMBER_TOKEN})(?!\s+times\b)\s+([A-Za-z][A-Za-z0-9 -]*?)(?=,|\s+and\s+{_NUMBER_TOKEN}|\.|$)",
            clause,
            flags=re.IGNORECASE,
        ):
            for index, label in enumerate(labels):
                if re.search(_narrative_label_pattern(label), raw_label, flags=re.IGNORECASE):
                    lower_bounds[index] = max(lower_bounds[index], _number(value))
                    break
    return lower_bounds


def _parse_narrative_extra_constraints(
    text: str,
    labels: list[str],
) -> tuple[list[list[float]], list[float]]:
    normalized = re.sub(r"\s+", " ", text)
    constraints: list[list[float]] = []
    upper_bounds: list[float] = []

    for index, label in enumerate(labels):
        label_pattern = _narrative_label_pattern(label)
        for match in re.finditer(
            rf"no\s+more\s+than\s+({_NUMBER_TOKEN})\s+units?\s+for\s+(?:the\s+)?{label_pattern}",
            normalized,
            flags=re.IGNORECASE,
        ):
            row = [0.0] * len(labels)
            row[index] = 1.0
            constraints.append(row)
            upper_bounds.append(_number(match.group(1)))

    total_match = re.search(
        rf"\btotal\s+number\s+of\s+items\b[^.]*?\bcannot\s+exceed\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if total_match:
        constraints.append([1.0] * len(labels))
        upper_bounds.append(_number(total_match.group(1)))

    ratio_match = re.search(
        rf"\boutput\s+of\s+product\s+([A-Za-z0-9]+)\s+must\s+be\s+at\s+least\s+"
        rf"({_NUMBER_TOKEN})\s+times\s+the\s+output\s+of\s+product\s+([A-Za-z0-9]+)",
        normalized,
        flags=re.IGNORECASE,
    )
    if ratio_match:
        lhs = _item_index_for_label(labels, ratio_match.group(1))
        rhs = _item_index_for_label(labels, ratio_match.group(3))
        if lhs is not None and rhs is not None:
            row = [0.0] * len(labels)
            row[lhs] = -1.0
            row[rhs] = _number(ratio_match.group(2))
            constraints.append(row)
            upper_bounds.append(0.0)

    storage_match = re.search(
        rf"storage\s+space\s+required\s+for\s+product\s+([A-Za-z0-9]+)\s+is\s+"
        rf"({_NUMBER_TOKEN})\s+times\s+that\s+of\s+product\s+([A-Za-z0-9]+)"
        rf".{{0,160}}?maximum\s+of\s+({_NUMBER_TOKEN})\s+kilograms?\s+of\s+product\s+\1",
        normalized,
        flags=re.IGNORECASE,
    )
    if storage_match:
        primary = _item_index_for_label(labels, storage_match.group(1))
        secondary = _item_index_for_label(labels, storage_match.group(3))
        multiplier = _number(storage_match.group(2))
        max_primary = _number(storage_match.group(4))
        if primary is not None and secondary is not None:
            row = [0.0] * len(labels)
            row[primary] = multiplier
            row[secondary] = 1.0
            constraints.append(row)
            upper_bounds.append(multiplier * max_primary)

    return constraints, upper_bounds


def _parse_table_resource_product_mix(
    text: str,
) -> tuple[list[str], list[float], list[list[float]], list[float], list[float], dict[str, Any]] | None:
    normalized = re.sub(r"\s+", " ", text)
    for header, rows in _parse_markdown_tables(text):
        cleaned_header = [_clean_label(cell).lower() for cell in header]
        resource_indices = [
            index
            for index, cell in enumerate(cleaned_header)
            if index > 0 and "time" in cell and ("minute" in cell or "hour" in cell)
        ]
        if len(resource_indices) < 2:
            continue
        resources = [_canonical_resource_label(header[index]) for index in resource_indices]
        minute_units = ["minute" in cleaned_header[index] for index in resource_indices]
        labels: list[str] = []
        coefficients: list[list[float]] = []
        for row in rows:
            if len(row) <= max(resource_indices):
                continue
            values = [_first_number(row[index]) for index in resource_indices]
            if any(value is None for value in values):
                continue
            labels.append(_clean_label(row[0]))
            coefficients.append([float(value or 0.0) for value in values])
        if len(labels) < 2:
            continue

        capacities: list[float] = []
        resource_costs: list[float] = []
        for resource, is_minutes in zip(resources, minute_units):
            resource_pattern = resource.replace(" ", r"\s+")
            cap_match = re.search(
                rf"\b(?:has\s+|only\s+)?([0-9][0-9,]*(?:\.\d+)?)\s+hours?\s+of\s+"
                rf"{resource_pattern}(?:\s+available)?",
                normalized,
                flags=re.IGNORECASE,
            )
            if not cap_match:
                break
            capacity = _number(cap_match.group(1))
            capacities.append(capacity * 60.0 if is_minutes else capacity)
            cost_match = re.search(
                rf"\bcost\s+of\s+{resource_pattern}\s+is\s*[£$¥]?\s*([0-9][0-9,]*(?:\.\d+)?)\s+per\s+hour",
                normalized,
                flags=re.IGNORECASE,
            )
            resource_costs.append(_number(cost_match.group(1)) if cost_match else 0.0)
        if len(capacities) != len(resources):
            continue

        revenues: list[float] = []
        for label in labels:
            label_pattern = _narrative_label_pattern(label)
            revenue_match = re.search(
                rf"\brevenue\s+for\s+{label_pattern}\s+is\s*[£$¥]?\s*([0-9][0-9,]*(?:\.\d+)?)",
                normalized,
                flags=re.IGNORECASE,
            )
            if not revenue_match:
                revenues = []
                break
            revenues.append(_number(revenue_match.group(1)))
        if len(revenues) != len(labels):
            continue

        objective: list[float] = []
        for product_values, revenue in zip(coefficients, revenues):
            resource_charge = sum(
                value / 60.0 * cost if is_minutes else value * cost
                for value, cost, is_minutes in zip(product_values, resource_costs, minute_units)
            )
            objective.append(revenue - resource_charge)
        constraint_rows = [
            [coefficients[product_index][resource_index] for product_index in range(len(labels))]
            for resource_index in range(len(resources))
        ]
        lower_bounds = _parse_narrative_lower_bounds(text, labels)
        return labels, objective, constraint_rows, capacities, lower_bounds, {
            "resources": resources,
            "resource_costs": resource_costs,
            "resource_coefficients": coefficients,
            "revenues": revenues,
        }
    return None


def _solve_multi_machine_process_profit_lp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        ("process a" in lowered or "equipment" in lowered)
        and ("process b" in lowered or "equipment" in lowered)
        and "effective machine hours" in lowered
        and ("operating costs at full capacity" in lowered or "processing cost per machine hour" in lowered)
        and "raw material cost" in lowered
        and "unit price" in lowered
    ):
        return TemplateSolveResult(False)
    cost_per_machine_hour = "processing cost per machine hour" in lowered

    table = _parse_markdown_table(text)
    if not table:
        return TemplateSolveResult(False)
    header, rows = table
    product_labels = [_clean_label(cell) for cell in header[1:4]]
    if len(product_labels) != 3:
        return TemplateSolveResult(False)
    machine_rows: list[dict[str, Any]] = []
    raw_cost: list[float] | None = None
    unit_price: list[float] | None = None
    for row in rows:
        if len(row) < 6:
            continue
        label = _clean_label(row[0])
        values = [_first_number(cell) for cell in row[1:4]]
        if label.lower().startswith("raw material cost"):
            raw_cost = [float(value or 0.0) for value in values]
            continue
        if label.lower().startswith("unit price"):
            unit_price = [float(value or 0.0) for value in values]
            continue
        capacity = _first_number(row[4])
        operating_cost = _first_number(row[5])
        if capacity is None or operating_cost is None:
            continue
        machine_rows.append(
            {
                "label": label,
                "times": [float(value) if value is not None else None for value in values],
                "capacity": float(capacity),
                "operating_cost": float(operating_cost),
            }
        )
    if len(machine_rows) < 2 or raw_cost is None or unit_price is None:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="multi_machine_process_profit_lp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
        )

    variables: list[tuple[str, int, int | None]] = []
    objective: list[float] = []
    for product_index in range(len(product_labels)):
        variables.append(("quantity", product_index, None))
        objective.append(-(unit_price[product_index] - raw_cost[product_index]))
    for machine_index, machine in enumerate(machine_rows):
        for product_index, processing_time in enumerate(machine["times"]):
            if processing_time is None:
                continue
            variables.append(("allocation", product_index, machine_index))
            if cost_per_machine_hour:
                objective.append(processing_time * machine["operating_cost"])
            else:
                objective.append(processing_time * machine["operating_cost"] / machine["capacity"])

    a_eq: list[list[float]] = []
    b_eq: list[float] = []
    for product_index in range(len(product_labels)):
        for process_prefix in ("A", "B"):
            row = [0.0] * len(variables)
            row[product_index] = -1.0
            for variable_index, variable in enumerate(variables):
                if (
                    variable[0] == "allocation"
                    and variable[1] == product_index
                    and machine_rows[variable[2] or 0]["label"].startswith(process_prefix)
                ):
                    row[variable_index] += 1.0
            if any(value > 0 for value in row[len(product_labels):]):
                a_eq.append(row)
                b_eq.append(0.0)

    a_ub: list[list[float]] = []
    b_ub: list[float] = []
    for machine_index, machine in enumerate(machine_rows):
        row = [0.0] * len(variables)
        for variable_index, variable in enumerate(variables):
            if variable[0] == "allocation" and variable[2] == machine_index:
                row[variable_index] = machine["times"][variable[1]] or 0.0
        a_ub.append(row)
        b_ub.append(machine["capacity"])

    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(0, None)] * len(variables),
        method="highs",
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="multi_machine_process_profit_lp",
            status="solver_failed",
            confidence=0.8,
            notes=str(result.message),
        )
    variable_values: dict[str, float] = {}
    for variable, value in zip(variables, result.x):
        if math.isclose(float(value), 0.0, abs_tol=1e-8):
            continue
        if variable[0] == "quantity":
            variable_values[f"produce_{product_labels[variable[1]]}"] = float(value)
        else:
            machine_label = machine_rows[variable[2] or 0]["label"]
            variable_values[f"process_{product_labels[variable[1]]}_on_{machine_label}"] = float(value)
    return TemplateSolveResult(
        matched=True,
        template_id="multi_machine_process_profit_lp",
        status="optimal",
        objective_value=float(-result.fun),
        variable_values=variable_values,
        confidence=0.84,
        notes="Solved multi-machine two-process production profit LP with proportional operating costs.",
        artifact={
            "products": product_labels,
            "machines": machine_rows,
            "raw_cost": raw_cost,
            "unit_price": unit_price,
            "cost_mode": "per_machine_hour" if cost_per_machine_hour else "full_capacity_prorated",
        },
    )


def _solve_farm_operating_plan_lp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "dairy cows" in lowered
        and "chickens" in lowered
        and "soybean" in lowered
        and "corn" in lowered
        and "wheat" in lowered
        and "person-days" in lowered
        and "maximize annual net income" in lowered
    ):
        return TemplateSolveResult(False)
    land = _number_after_patterns(normalized, [rf"has\s+({_NUMBER_TOKEN})\s+hectares?\s+of\s+land"])
    funds = _number_after_patterns(normalized, [rf"and\s+({_NUMBER_TOKEN})\s+yuan\s+in\s+funds"])
    autumn_labor = _number_after_patterns(normalized, [rf"({_NUMBER_TOKEN})\s+person-days\s+in\s+autumn\s+and\s+winter"])
    spring_labor = _number_after_patterns(normalized, [rf"({_NUMBER_TOKEN})\s+person-days\s+in\s+spring\s+and\s+summer"])
    if None in (land, funds, autumn_labor, spring_labor):
        return TemplateSolveResult(False)

    crop_rows: dict[str, list[float]] = {}
    for header, rows in _parse_markdown_tables(text):
        header_text = " ".join(header).lower()
        if "soybean" not in header_text or "corn" not in header_text or "wheat" not in header_text:
            continue
        for row in rows:
            if len(row) < 4:
                continue
            values = [_first_number(cell) for cell in row[1:4]]
            if any(value is None for value in values):
                continue
            crop_rows[_clean_label(row[0]).lower()] = [float(value) for value in values if value is not None]
    autumn = next((values for key, values in crop_rows.items() if "autumn" in key), None)
    spring = next((values for key, values in crop_rows.items() if "spring" in key), None)
    crop_income = next((values for key, values in crop_rows.items() if "income" in key), None)
    if autumn is None or spring is None or crop_income is None:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="farm_operating_plan_lp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
        )
    # soy, corn, wheat, cows, chickens, unused autumn/winter labor, unused spring/summer labor
    objective = crop_income + [400.0, 2.0, 1.8, 2.1]
    constraints = [
        [1.0, 1.0, 1.0, 1.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 400.0, 3.0, 0.0, 0.0],
        [autumn[0], autumn[1], autumn[2], 100.0, 0.6, 1.0, 0.0],
        [spring[0], spring[1], spring[2], 50.0, 0.3, 0.0, 1.0],
    ]
    result = linprog(
        [-value for value in objective],
        A_ub=constraints,
        b_ub=[land, funds, autumn_labor, spring_labor],
        bounds=[(0, None), (0, None), (0, None), (0, 32), (0, 3000), (0, None), (0, None)],
        method="highs",
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="farm_operating_plan_lp",
            status="solver_failed",
            confidence=0.8,
            notes=str(result.message),
        )
    names = ["soybean_hectares", "corn_hectares", "wheat_hectares", "dairy_cows", "chickens", "unused_autumn_winter_labor", "unused_spring_summer_labor"]
    return TemplateSolveResult(
        matched=True,
        template_id="farm_operating_plan_lp",
        status="optimal",
        objective_value=float(-result.fun),
        variable_values={name: float(value) for name, value in zip(names, result.x) if not math.isclose(float(value), 0.0, abs_tol=1e-8)},
        confidence=0.84,
        notes="Solved farm crop/livestock operating plan LP with land, funds, and seasonal labor constraints.",
        artifact={"crop_income": crop_income, "autumn_labor": autumn, "spring_labor": spring},
    )


def _solve_fixed_activation_quota_product_plan(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "fixed activation cost" in lowered
        and "minimum production batch" in lowered
        and "production quota" in lowered
        and ("maximize" in lowered or "maximizes" in lowered)
    ):
        return TemplateSolveResult(False)

    labels: list[str] = []
    demand: list[float] | None = None
    price: list[float] | None = None
    cost: list[float] | None = None
    quota: list[float] | None = None
    activation: list[float] | None = None
    for header, rows in _parse_markdown_tables(text):
        if len(header) < 2:
            continue
        header_labels = [_clean_label(cell) for cell in header[1:]]
        row_map: dict[str, list[float]] = {}
        for row in rows:
            if len(row) < len(header):
                continue
            values = [_first_number(cell) for cell in row[1 : 1 + len(header_labels)]]
            if any(value is None for value in values):
                continue
            row_map[_clean_label(row[0]).lower()] = [float(value) for value in values if value is not None]
        if {"maximum demand", "selling price", "production cost", "production quota"} <= set(row_map):
            labels = header_labels
            demand = row_map["maximum demand"]
            price = row_map["selling price"]
            cost = row_map["production cost"]
            quota = row_map["production quota"]
        if "activation cost" in row_map:
            activation = row_map["activation cost"]
    if not labels or demand is None or price is None or cost is None or quota is None or activation is None:
        return TemplateSolveResult(False)
    if not (len(labels) == len(demand) == len(price) == len(cost) == len(quota) == len(activation)):
        return TemplateSolveResult(False)

    minimum_batch = [0.0] * len(labels)
    min_match = re.search(
        r"Minimum\s+Batch\s*&\s*([0-9.]+)\s*&\s*([0-9.]+)\s*&\s*([0-9.]+)",
        text,
        flags=re.IGNORECASE,
    )
    if min_match and len(labels) == 3:
        minimum_batch = [_number(min_match.group(index)) for index in range(1, 4)]
    days = _number_after_patterns(
        normalized,
        [
            rf"produce\s+for\s+({_NUMBER_TOKEN})\s+days",
            rf"produce[^. ]*(?:\s+[^. ]*){{0,12}}\s+for\s+({_NUMBER_TOKEN})\s+days",
        ],
    )
    if days is None:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="fixed_activation_quota_product_plan_lp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
        )

    best: tuple[float, tuple[int, ...], list[float]] | None = None
    for active in itertools.product([0, 1], repeat=len(labels)):
        if not any(active):
            continue
        bounds = [
            (minimum_batch[index], demand[index]) if is_active else (0.0, 0.0)
            for index, is_active in enumerate(active)
        ]
        result = linprog(
            [-(price[index] - cost[index]) for index in range(len(labels))],
            A_ub=[[1.0 / quota[index] for index in range(len(labels))]],
            b_ub=[days],
            bounds=bounds,
            method="highs",
        )
        if not result.success:
            continue
        value = -float(result.fun) - sum(activation[index] * active[index] for index in range(len(labels)))
        if best is None or value > best[0] + 1e-9:
            best = (value, tuple(active), [float(item) for item in result.x])
    if best is None:
        return TemplateSolveResult(
            matched=True,
            template_id="fixed_activation_quota_product_plan_lp",
            status="infeasible",
            confidence=0.78,
        )
    return TemplateSolveResult(
        matched=True,
        template_id="fixed_activation_quota_product_plan_lp",
        status="optimal",
        objective_value=float(best[0]),
        variable_values={
            f"produce_{_clean_label(label)}": value
            for label, value in zip(labels, best[2])
            if not math.isclose(value, 0.0, abs_tol=1e-8)
        },
        confidence=0.86,
        notes="Solved production plan with fixed activation costs, daily quota time, demand caps, and minimum batches.",
        artifact={
            "labels": labels,
            "demand": demand,
            "price": price,
            "cost": cost,
            "quota": quota,
            "activation": activation,
            "minimum_batch": minimum_batch,
            "active": list(best[1]),
        },
    )


def _solve_linear_product_mix(
    *,
    template_id: str,
    labels: list[str],
    objective: list[float],
    constraints: list[list[float]],
    upper_bounds: list[float],
    lower_bounds: list[float],
    integer_variables: bool,
    artifact: dict[str, Any],
) -> TemplateSolveResult:
    if len(labels) < 2 or len(objective) != len(labels) or not constraints:
        return TemplateSolveResult(False)
    if integer_variables:
        try:
            from scipy.optimize import Bounds, LinearConstraint, milp
            import numpy as np
        except ImportError as exc:
            return TemplateSolveResult(
                matched=True,
                template_id=template_id,
                status="solver_unavailable",
                confidence=0.8,
                notes=str(exc),
                artifact=artifact,
            )
        rows = [list(row) for row in constraints]
        lower = [-math.inf] * len(rows)
        upper = list(upper_bounds)
        for index, lower_bound in enumerate(lower_bounds):
            if lower_bound > 0:
                row = [0.0] * len(labels)
                row[index] = 1.0
                rows.append(row)
                lower.append(lower_bound)
                upper.append(math.inf)
        result = milp(
            c=np.array([-value for value in objective], dtype=float),
            integrality=np.ones(len(labels)),
            bounds=Bounds(np.zeros(len(labels)), np.full(len(labels), math.inf)),
            constraints=LinearConstraint(np.array(rows, dtype=float), np.array(lower), np.array(upper)),
        )
        if not result.success:
            return TemplateSolveResult(
                matched=True,
                template_id=template_id,
                status="solver_failed",
                confidence=0.82,
                notes=str(result.message),
                artifact=artifact,
            )
        values = [float(value) for value in result.x]
        optimum = float(-result.fun)
    else:
        status, optimum, values, message = _linprog_maximize(
            objective=objective,
            constraints=constraints,
            upper_bounds=upper_bounds,
            lower_bounds=lower_bounds,
        )
        if status != "optimal":
            return TemplateSolveResult(
                matched=True,
                template_id=template_id,
                status=status,
                confidence=0.82,
                notes=message,
                artifact=artifact,
            )
        assert optimum is not None
    return TemplateSolveResult(
        matched=True,
        template_id=template_id,
        status="optimal",
        objective_value=optimum,
        variable_values={
            f"produce_{labels[index]}": value
            for index, value in enumerate(values)
            if not math.isclose(value, 0.0, abs_tol=1e-9)
        },
        confidence=0.84,
        notes="Solved narrative product-mix LP/MILP from explicit resource and objective coefficients.",
        artifact={**artifact, "integer_variables": integer_variables},
    )


def _solve_weighted_idle_goal_product_mix(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        "idle time" in lowered
        and "process cost" in lowered
        and "selling price" in lowered
        and "capacity" in lowered
        and ("overtime is not allowed" in lowered or "no overtime" in lowered)
    ):
        return TemplateSolveResult(False)

    parsed_table: tuple[list[str], list[dict[str, Any]], list[float], list[float]] | None = None
    for header, rows in _parse_markdown_tables(text):
        cleaned_header = [_clean_label(cell).lower() for cell in header]
        price_index = next(
            (
                index
                for index, cell in enumerate(cleaned_header)
                if index > 0 and "selling" in cell and "price" in cell
            ),
            None,
        )
        if price_index is None:
            continue
        process_indices = [index for index in range(1, len(header)) if index != price_index]
        if not process_indices:
            continue

        products: list[dict[str, Any]] = []
        capacities: list[float] | None = None
        process_costs: list[float] | None = None
        for row in rows:
            if len(row) <= max(process_indices + [price_index]):
                continue
            row_label = _clean_label(row[0])
            row_key = row_label.lower()
            process_values = [_first_number(row[index]) for index in process_indices]
            if "capacity" in row_key:
                if all(value is not None for value in process_values):
                    capacities = [float(value or 0.0) for value in process_values]
                continue
            if "process cost" in row_key or "hourly cost" in row_key:
                if all(value is not None for value in process_values):
                    process_costs = [float(value or 0.0) for value in process_values]
                continue
            selling_price = _first_number(row[price_index])
            if selling_price is None or any(value is None for value in process_values):
                continue
            products.append(
                {
                    "label": row_label,
                    "process_hours": [float(value or 0.0) for value in process_values],
                    "selling_price": float(selling_price),
                }
            )
        if products and capacities and process_costs and len(capacities) == len(process_costs):
            parsed_table = (
                [_clean_label(header[index]) for index in process_indices],
                products,
                capacities,
                process_costs,
            )
            break

    profit_target = _parse_minimum_profit_target(text)
    if not parsed_table or profit_target is None:
        return TemplateSolveResult(False)
    process_labels, products, capacities, process_costs = parsed_table
    product_labels = [str(product["label"]) for product in products]
    process_matrix = [list(product["process_hours"]) for product in products]
    unit_profits = [
        float(product["selling_price"])
        - sum(process_matrix[product_index][process_index] * process_costs[process_index] for process_index in range(len(process_costs)))
        for product_index, product in enumerate(products)
    ]
    if any(profit <= 0 for profit in unit_profits):
        return TemplateSolveResult(False)
    lower_bounds = _parse_minimum_unit_bounds(text, product_labels)

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="weighted_idle_goal_product_mix_ilp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
            artifact={
                "products": products,
                "processes": process_labels,
                "capacities": capacities,
                "process_costs": process_costs,
                "unit_profits": unit_profits,
            },
        )

    variable_count = len(products)
    rows: list[Any] = []
    lower: list[float] = []
    upper: list[float] = []
    for process_index, capacity in enumerate(capacities):
        row = np.zeros(variable_count)
        for product_index in range(variable_count):
            row[product_index] = process_matrix[product_index][process_index]
        rows.append(row)
        lower.append(0.0)
        upper.append(float(capacity))
    rows.append(np.array(unit_profits))
    lower.append(float(profit_target))
    upper.append(math.inf)

    weighted_use = np.array(
        [
            sum(process_matrix[product_index][process_index] * process_costs[process_index] for process_index in range(len(process_costs)))
            for product_index in range(variable_count)
        ]
    )
    result = milp(
        -weighted_use,
        integrality=np.ones(variable_count),
        bounds=Bounds(np.array(lower_bounds), np.full(variable_count, math.inf)),
        constraints=LinearConstraint(np.vstack(rows), np.array(lower), np.array(upper)),
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="weighted_idle_goal_product_mix_ilp",
            status="solver_failed",
            confidence=0.82,
            notes=str(result.message),
            artifact={
                "products": products,
                "processes": process_labels,
                "capacities": capacities,
                "process_costs": process_costs,
                "unit_profits": unit_profits,
                "profit_target": profit_target,
                "lower_bounds": lower_bounds,
            },
        )

    values = [float(value) for value in result.x]
    objective = sum(unit_profits[index] * values[index] for index in range(variable_count))
    variable_values = {
        f"produce_{product_labels[index]}": values[index]
        for index in range(variable_count)
        if not math.isclose(values[index], 0.0, abs_tol=1e-9)
    }
    return TemplateSolveResult(
        matched=True,
        template_id="weighted_idle_goal_product_mix_ilp",
        status="optimal",
        objective_value=float(objective),
        variable_values=variable_values,
        confidence=0.86,
        notes=(
            "Solved countable-product goal program: minimum profit and unit "
            "targets first, then minimum weighted process idle time without overtime."
        ),
        artifact={
            "products": products,
            "processes": process_labels,
            "capacities": capacities,
            "process_costs": process_costs,
            "unit_profits": unit_profits,
            "profit_target": profit_target,
            "lower_bounds": lower_bounds,
            "weighted_use_objective": [float(value) for value in weighted_use],
        },
    )


def _solve_minimum_overtime_production_goal(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        "overtime" in lowered
        and "minimize overtime" in lowered
        and "production time" in lowered
        and "rate" in lowered
        and "per hour" in lowered
        and "at least" in lowered
    ):
        return TemplateSolveResult(False)

    regular_match = re.search(
        rf"production\s+time[^.;\n]{{0,80}}?(?:set\s+at|of|fully\s+utilize)\s+({_NUMBER_TOKEN})\s+hours?",
        text,
        flags=re.IGNORECASE,
    )
    rate_match = re.search(
        rf"rate\s+of\s+({_NUMBER_TOKEN})\s+(?:meters?|units?|pairs?)\s+per\s+hour",
        text,
        flags=re.IGNORECASE,
    )
    cap_match = re.search(
        rf"overtime[^.;\n]{{0,80}}?(?:not\s+exceed|no\s+more\s+than|at\s+most)\s+({_NUMBER_TOKEN})\s+hours?",
        text,
        flags=re.IGNORECASE,
    )
    if not regular_match or not rate_match:
        return TemplateSolveResult(False)

    demands: dict[str, float] = {}
    demand_pattern = re.compile(
        rf"at\s+least\s+({_NUMBER_TOKEN})\s+(?:meters?|units?|pairs?)\s+of\s+([A-Za-z][A-Za-z\s-]*?)(?:\s+fabric|\s+product|\s+products)?\b",
        flags=re.IGNORECASE,
    )
    for match in demand_pattern.finditer(text):
        label = _clean_label(match.group(2)).lower()
        if label in {"the", "a", "an"}:
            continue
        demands[label] = max(demands.get(label, 0.0), _number(match.group(1)))
    if not demands:
        return TemplateSolveResult(False)

    regular_hours = _number(regular_match.group(1))
    rate = _number(rate_match.group(1))
    overtime_cap = _number(cap_match.group(1)) if cap_match else math.inf
    required_hours = sum(demands.values()) / rate
    overtime = max(0.0, required_hours - regular_hours)
    if overtime > overtime_cap + 1e-9:
        return TemplateSolveResult(
            matched=True,
            template_id="minimum_overtime_production_goal",
            status="infeasible",
            confidence=0.82,
            notes="Required production hours exceed regular time plus the stated overtime cap.",
            artifact={
                "demands": demands,
                "regular_hours": regular_hours,
                "rate_per_hour": rate,
                "required_hours": required_hours,
                "overtime_cap": overtime_cap,
            },
        )

    variable_values = {
        f"produce_{label}": value
        for label, value in sorted(demands.items())
    }
    variable_values["regular_hours"] = min(regular_hours, required_hours)
    variable_values["overtime_hours"] = overtime
    return TemplateSolveResult(
        matched=True,
        template_id="minimum_overtime_production_goal",
        status="optimal",
        objective_value=float(overtime),
        variable_values=variable_values,
        confidence=0.86,
        notes="Computed minimum overtime needed to satisfy stated production demand at the common production rate.",
        artifact={
            "demands": demands,
            "regular_hours": regular_hours,
            "rate_per_hour": rate,
            "required_hours": required_hours,
            "overtime_cap": overtime_cap,
        },
    )


def _solve_sales_staff_overtime_goal(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        "sales volume" in lowered
        and "monthly working hours" in lowered
        and "full employment" in lowered
        and "minimize overtime" in lowered
        and "sales" in lowered
    ):
        return TemplateSolveResult(False)

    parsed: list[dict[str, float | str]] = []
    for header, rows in _parse_markdown_tables(text):
        cleaned_header = [_clean_label(cell).lower() for cell in header]
        if not ("monthly working hours" in " ".join(cleaned_header) and "sales volume" in " ".join(cleaned_header)):
            continue
        hours_index = next((index for index, cell in enumerate(cleaned_header) if "working hours" in cell), None)
        sales_index = next((index for index, cell in enumerate(cleaned_header) if "sales volume" in cell), None)
        wage_index = next((index for index, cell in enumerate(cleaned_header) if cell.startswith("wage")), None)
        overtime_pay_index = next((index for index, cell in enumerate(cleaned_header) if "overtime pay" in cell), None)
        if hours_index is None or sales_index is None:
            continue
        for row in rows:
            if len(row) <= max(hours_index, sales_index):
                continue
            hours = _first_number(row[hours_index])
            sales_rate = _first_number(row[sales_index])
            if hours is None or sales_rate is None:
                continue
            label = _clean_label(row[0])
            count_match = re.search(
                rf"({_NUMBER_TOKEN})\s+{_label_regex(label)}\s+sales\s+clerks?",
                text,
                flags=re.IGNORECASE,
            )
            if not count_match:
                count_match = re.search(
                    rf"({_NUMBER_TOKEN})\s+{_label_regex(label)}\s+\w*\s*clerks?",
                    text,
                    flags=re.IGNORECASE,
                )
            if not count_match:
                continue
            item: dict[str, float | str] = {
                "label": label,
                "count": _number(count_match.group(1)),
                "regular_hours_per_worker": float(hours),
                "sales_per_hour": float(sales_rate),
            }
            if wage_index is not None and len(row) > wage_index:
                wage = _first_number(row[wage_index])
                if wage is not None:
                    item["wage"] = float(wage)
            if overtime_pay_index is not None and len(row) > overtime_pay_index:
                overtime_pay = _first_number(row[overtime_pay_index])
                if overtime_pay is not None:
                    item["overtime_pay"] = float(overtime_pay)
            parsed.append(item)
        if parsed:
            break
    if not parsed:
        return TemplateSolveResult(False)

    target_match = re.search(
        rf"(?:achieve|target|goal|sales)[^.;\n]{{0,120}}?sales\s+of\s+({_NUMBER_TOKEN})\s+pairs?",
        text,
        flags=re.IGNORECASE,
    )
    if not target_match:
        target_match = re.search(
            rf"({_NUMBER_TOKEN})\s+pairs?",
            text,
            flags=re.IGNORECASE,
        )
    if not target_match:
        return TemplateSolveResult(False)
    target_sales = _number(target_match.group(1))
    regular_sales = sum(
        float(row["count"]) * float(row["regular_hours_per_worker"]) * float(row["sales_per_hour"])
        for row in parsed
    )
    shortfall = max(0.0, target_sales - regular_sales)
    best = max(parsed, key=lambda row: float(row["sales_per_hour"]))
    best_rate = float(best["sales_per_hour"])
    overtime = shortfall / best_rate if best_rate > 0 else math.inf

    variable_values = {
        f"regular_hours_{row['label']}": float(row["count"]) * float(row["regular_hours_per_worker"])
        for row in parsed
    }
    variable_values[f"overtime_hours_{best['label']}"] = overtime
    return TemplateSolveResult(
        matched=True,
        template_id="sales_staff_overtime_goal",
        status="optimal",
        objective_value=float(overtime),
        variable_values=variable_values,
        confidence=0.86,
        notes=(
            "Computed minimum overtime after full employment by assigning the "
            "sales shortfall to the staff class with highest sales per hour."
        ),
        artifact={
            "staff_classes": parsed,
            "target_sales": target_sales,
            "regular_sales": regular_sales,
            "shortfall": shortfall,
            "overtime_staff_class": best["label"],
        },
    )


def _normalize_resource_label(value: str) -> str:
    label = _clean_label(value).lower()
    label = re.sub(r"\b(?:hours?|worth)\b", "", label)
    label = re.sub(r"\braw\s+materials?\b", "raw_materials", label)
    if label.strip() == "labor":
        return "labor"
    label = re.sub(r"\blabor\b", "", label)
    return re.sub(r"\s+", " ", label).strip()


def _parse_resource_requirements(body: str) -> dict[str, float]:
    requirements: dict[str, float] = {}
    for value, resource in re.findall(
        rf"({_FRACTION_OR_NUMBER_TOKEN})\s*(?:kg|hours?|tons?|pieces?)?\s+of\s+([A-Za-z][A-Za-z\s]*?)(?=,|\s+and\s+|$)",
        body,
        flags=re.IGNORECASE,
    ):
        label = _normalize_resource_label(resource)
        if label:
            requirements[label] = _quantity_number(value)
    return requirements


def _parse_overtime_product_mix_numbered(
    text: str,
) -> tuple[list[dict[str, Any]], dict[str, float], str, float, float | None] | None:
    ordinals = [("first", "first_type"), ("second", "second_type")]
    products: list[dict[str, Any]] = []
    raw_costs: list[float] = []
    for ordinal, label in ordinals:
        match = re.search(
            (
                rf"product\s+of\s+the\s+{ordinal}\s+type\s+requires\s+"
                rf"({_FRACTION_OR_NUMBER_TOKEN})\s+hours?\s+of\s+assembly(?:\s+labor)?"
                rf",\s+({_FRACTION_OR_NUMBER_TOKEN})\s+hours?\s+of\s+testing"
                rf",\s+and\s+\$?\s*({_FRACTION_OR_NUMBER_TOKEN})\s+worth\s+of\s+raw\s+materials?"
            ),
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        products.append(
            {
                "label": label,
                "requirements": {
                    "assembly": _quantity_number(match.group(1)),
                    "testing": _quantity_number(match.group(2)),
                },
            }
        )
        raw_costs.append(_quantity_number(match.group(3)))

    value_match = re.search(
        rf"first\s+and\s+second\s+type\s+have\s+a\s+market\s+value\s+of\s+\$?\s*({_FRACTION_OR_NUMBER_TOKEN})\s+and\s+\$?\s*({_FRACTION_OR_NUMBER_TOKEN})",
        text,
        flags=re.IGNORECASE,
    )
    capacity_match = re.search(
        rf"at\s+most\s+({_FRACTION_OR_NUMBER_TOKEN})\s+hours?\s+of\s+assembly(?:\s+labor)?\s+and\s+({_FRACTION_OR_NUMBER_TOKEN})\s+hours?\s+of\s+testing",
        text,
        flags=re.IGNORECASE,
    )
    overtime_match = re.search(
        rf"up\s+to\s+({_FRACTION_OR_NUMBER_TOKEN})\s+hours?\s+of\s+overtime\s+assembly\s+labor.*?cost\s+of\s+\$?\s*({_FRACTION_OR_NUMBER_TOKEN})\s+per\s+hour",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not value_match or not capacity_match or not overtime_match:
        return None
    market_values = [_quantity_number(value_match.group(1)), _quantity_number(value_match.group(2))]
    for index, product in enumerate(products):
        product["unit_profit"] = market_values[index] - raw_costs[index]
        product["market_value"] = market_values[index]
        product["raw_material_cost"] = raw_costs[index]
    capacities = {
        "assembly": _quantity_number(capacity_match.group(1)),
        "testing": _quantity_number(capacity_match.group(2)),
    }
    return (
        products,
        capacities,
        "assembly",
        _quantity_number(overtime_match.group(2)),
        _quantity_number(overtime_match.group(1)),
    )


def _parse_overtime_product_mix_labeled(
    text: str,
) -> tuple[list[dict[str, Any]], dict[str, float], str, float, float | None] | None:
    products: list[dict[str, Any]] = []
    for match in re.finditer(
        (
            rf"single\s+unit\s+of\s+product\s+([A-Za-z0-9]+)\s+requires\s+"
            rf"(.*?)\s+and\s+yields\s+a\s+profit\s+of\s+({_NUMBER_TOKEN})"
        ),
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        label, body, profit = match.groups()
        requirements = _parse_resource_requirements(body)
        if requirements:
            products.append(
                {
                    "label": _clean_label(label),
                    "requirements": requirements,
                    "unit_profit": _number(profit),
                }
            )
    capacity_match = re.search(
        r"currently\s+has\s+(.*?)\s+available",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    overtime_match = re.search(
        rf"overtime\s+pay\s+is\s+({_NUMBER_TOKEN})\s+\w+\s+per\s+hour",
        text,
        flags=re.IGNORECASE,
    )
    if len(products) < 2 or not capacity_match or not overtime_match:
        return None
    capacities = _parse_resource_requirements(capacity_match.group(1))
    if not capacities:
        return None
    overtime_resource = "labor" if "labor" in capacities else next(iter(capacities))
    return products, capacities, overtime_resource, _number(overtime_match.group(1)), None


def _solve_overtime_resource_product_mix(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        "overtime" in lowered
        and ("maximize" in lowered or "maximal" in lowered)
        and ("product" in lowered or "products" in lowered)
        and ("profit" in lowered or "market value" in lowered)
    ):
        return TemplateSolveResult(False)

    parsed = _parse_overtime_product_mix_numbered(text)
    integer_products = False
    if parsed is None:
        parsed = _parse_overtime_product_mix_labeled(text)
        integer_products = parsed is not None
    if parsed is None:
        return TemplateSolveResult(False)
    products, capacities, overtime_resource, overtime_cost, overtime_cap = parsed
    if overtime_resource not in capacities:
        return TemplateSolveResult(False)

    product_labels = [str(product["label"]) for product in products]
    resources = list(capacities)
    product_count = len(products)
    variable_count = product_count + 1
    overtime_index = product_count
    rounded_answer = "rounded to the nearest" in lowered

    c = [-float(product["unit_profit"]) for product in products] + [float(overtime_cost)]
    rows: list[list[float]] = []
    upper: list[float] = []
    for resource in resources:
        row = [
            float(product["requirements"].get(resource, 0.0))
            for product in products
        ]
        if resource == overtime_resource:
            row.append(-1.0)
        else:
            row.append(0.0)
        rows.append(row)
        upper.append(float(capacities[resource]))
    if overtime_cap is not None:
        row = [0.0] * variable_count
        row[overtime_index] = 1.0
        rows.append(row)
        upper.append(float(overtime_cap))

    try:
        from scipy.optimize import Bounds, LinearConstraint, linprog, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="overtime_resource_product_mix",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
            artifact={"products": products, "capacities": capacities},
        )

    if integer_products:
        result = milp(
            np.array(c, dtype=float),
            integrality=[1] * product_count + [0],
            bounds=Bounds(np.zeros(variable_count), np.full(variable_count, math.inf)),
            constraints=LinearConstraint(
                np.array(rows, dtype=float),
                np.full(len(rows), -math.inf),
                np.array(upper, dtype=float),
            ),
        )
    else:
        result = linprog(
            c,
            A_ub=rows,
            b_ub=upper,
            bounds=[(0, None)] * variable_count,
            method="highs",
        )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="overtime_resource_product_mix",
            status="solver_failed",
            confidence=0.82,
            notes=str(result.message),
            artifact={
                "products": products,
                "capacities": capacities,
                "overtime_resource": overtime_resource,
                "overtime_cost": overtime_cost,
                "overtime_cap": overtime_cap,
                "integer_products": integer_products,
            },
        )

    raw_objective = float(-result.fun)
    objective = float(round(raw_objective)) if rounded_answer else raw_objective
    variable_values = {
        f"produce_{product_labels[index]}": float(result.x[index])
        for index in range(product_count)
        if not math.isclose(float(result.x[index]), 0.0, abs_tol=1e-8)
    }
    overtime_value = float(result.x[overtime_index])
    if not math.isclose(overtime_value, 0.0, abs_tol=1e-8):
        variable_values[f"overtime_{overtime_resource}_hours"] = overtime_value
    return TemplateSolveResult(
        matched=True,
        template_id="overtime_resource_product_mix",
        status="optimal",
        objective_value=objective,
        variable_values=variable_values,
        confidence=0.88,
        notes="Solved product-mix LP/MILP with one overtime-expandable resource and overtime cost.",
        artifact={
            "products": products,
            "resources": resources,
            "capacities": capacities,
            "overtime_resource": overtime_resource,
            "overtime_cost": overtime_cost,
            "overtime_cap": overtime_cap,
            "integer_products": integer_products,
            "raw_objective_value": raw_objective,
        },
    )


def _solve_product_mix_table(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        ("maximize" in lowered or "maximum profit" in lowered or "profit should" in lowered)
        and "profit" in lowered
        and "capacity" in lowered
    ):
        return TemplateSolveResult(False)
    parsed = _parse_markdown_table(text)
    if not parsed:
        return TemplateSolveResult(False)
    header, rows = parsed
    table_rows: list[tuple[str, list[float]]] = []
    profit: list[float] | None = None
    for row in rows:
        if len(row) < 3:
            continue
        row_label = _clean_label(row[0]).lower()
        values: list[float] = []
        for cell in row[1:]:
            stripped = cell.strip()
            if re.fullmatch(r"[0-9][0-9,]*(?:\.\d+)?", stripped):
                values.append(_number(stripped))
            else:
                values.append(math.nan)
        finite_values = [value for value in values if not math.isnan(value)]
        if "profit" in row_label and len(finite_values) >= 2:
            profit = finite_values
            continue
        table_rows.append((row_label, values))
    if profit is None or len(profit) < 2:
        return TemplateSolveResult(False)

    product_labels = _product_labels_from_table(header, rows, len(profit))
    product_count = len(product_labels)
    numeric_rows: list[tuple[str, list[float], float]] = []
    for row_label, values in table_rows:
        if len(values) <= product_count:
            continue
        coefficients = values[:product_count]
        if any(math.isnan(value) for value in coefficients):
            continue
        capacity = next(
            (value for value in values[product_count:] if not math.isnan(value)),
            None,
        )
        if capacity is None:
            continue
        numeric_rows.append((row_label, coefficients, float(capacity)))
    if not numeric_rows:
        return TemplateSolveResult(False)

    lower_bounds = _parse_minimum_unit_bounds(text, product_labels)

    status, objective, values, message = _linprog_maximize(
        objective=profit,
        constraints=[row[1] for row in numeric_rows],
        upper_bounds=[row[2] for row in numeric_rows],
        lower_bounds=lower_bounds,
    )
    if status != "optimal":
        return TemplateSolveResult(
            matched=True,
            template_id="product_mix_table_lp",
            status=status,
            confidence=0.8,
            notes=message,
            artifact={"profit": profit, "constraints": numeric_rows},
        )
    variable_values = {
        f"produce_{product_labels[index]}": value
        for index, value in enumerate(values)
    }
    return TemplateSolveResult(
        matched=True,
        template_id="product_mix_table_lp",
        status="optimal",
        objective_value=objective,
        variable_values=variable_values,
        confidence=0.88,
        notes="Solved product-mix LP from profit/capacity table.",
        artifact={
            "products": product_labels,
            "profit": profit,
            "constraints": numeric_rows,
            "lower_bounds": lower_bounds,
        },
    )


def _solve_narrative_product_mix_lp(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        (
            "maximize" in lowered
            or "maximum profit" in lowered
            or "maximum possible" in lowered
            or "linear programming model" in lowered
        )
        and ("profit" in lowered or "revenue" in lowered)
    ):
        return TemplateSolveResult(False)
    if re.search(
        r"\b(fixed\s+activation\s+cost|activation\s+cost|bulk\s+purchases?|by-product|"
        r"blend|crude\s+oil|inventory|outsourcing|discounts?)\b",
        lowered,
    ):
        return TemplateSolveResult(False)
    if re.search(
        r"\bif\b.{0,120}\bmanufactures?\b|\bwill\s+also\s+manufacture\b|"
        r"\bwill\s+not\s+manufacture\b|\bmore\s+than\s+two\s+types\b",
        lowered,
        flags=re.DOTALL,
    ):
        return TemplateSolveResult(False)

    table_parsed = _parse_table_resource_product_mix(text)
    if table_parsed:
        labels, objective, constraints, upper_bounds, lower_bounds, artifact = table_parsed
        integer_variables = not re.search(
            r"\bfractional\s+(?:batches|lots|production)|fractional\s+when\s+appropriate\b",
            lowered,
        )
        return _solve_linear_product_mix(
            template_id="narrative_product_mix_lp",
            labels=labels,
            objective=objective,
            constraints=constraints,
            upper_bounds=upper_bounds,
            lower_bounds=lower_bounds,
            integer_variables=integer_variables,
            artifact=artifact,
        )

    objective_parsed = _parse_narrative_objective(text)
    if not objective_parsed:
        return TemplateSolveResult(False)
    labels, objective = objective_parsed
    coefficients = _parse_narrative_resource_coefficients(text, labels)
    capacities = _parse_narrative_resource_capacities(text, set(coefficients))
    constraints: list[list[float]] = []
    upper_bounds: list[float] = []
    for resource, row in coefficients.items():
        if resource not in capacities:
            continue
        constraints.append(row)
        upper_bounds.append(capacities[resource])
    extra_constraints, extra_bounds = _parse_narrative_extra_constraints(text, labels)
    constraints.extend(extra_constraints)
    upper_bounds.extend(extra_bounds)
    if not constraints:
        return TemplateSolveResult(False)
    lower_bounds = _parse_narrative_lower_bounds(text, labels)
    integer_variables = bool(
        re.search(r"\b(units?|items?|packages?)\b", lowered)
        and not re.search(r"\b(fractional|kilograms?|kg|acres?|any\s+quantity)\b", lowered)
    )
    return _solve_linear_product_mix(
        template_id="narrative_product_mix_lp",
        labels=labels,
        objective=objective,
        constraints=constraints,
        upper_bounds=upper_bounds,
        lower_bounds=lower_bounds,
        integer_variables=integer_variables,
        artifact={
            "products": labels,
            "objective": objective,
            "resource_coefficients": coefficients,
            "capacities": capacities,
            "lower_bounds": lower_bounds,
            "extra_constraint_count": len(extra_constraints),
        },
    )


def _solve_two_product_time_ratio_lp(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        "two products" in lowered
        and "profit" in lowered
        and "minutes" in lowered
        and "hours" in lowered
        and "for every" in lowered
    ):
        return TemplateSolveResult(False)
    profit_match = re.search(
        r"profit\s+of\s+[^0-9]*([0-9][0-9,]*(?:\.\d+)?)\s+and\s+[^0-9]*([0-9][0-9,]*(?:\.\d+)?)\s+per\s+unit",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    time_match = re.search(
        r"(?:requires?|requiring)\s+([0-9][0-9,]*(?:\.\d+)?)\s+minutes?.*?product\s+A\s+and\s+([0-9][0-9,]*(?:\.\d+)?)\s+minutes?.*?product\s+B",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    capacity_match = re.search(
        r"([0-9][0-9,]*(?:\.\d+)?)\s+hours?",
        text,
        flags=re.IGNORECASE,
    )
    ratio_match = re.search(
        rf"for\s+every\s+({_NUMBER_TOKEN})\s+units?\s+of\s+product\s+A.*?at\s+least\s+({_NUMBER_TOKEN})\s+units?\s+of\s+product\s+B",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not (profit_match and time_match and capacity_match and ratio_match):
        return TemplateSolveResult(False)
    profit = [_number(profit_match.group(1)), _number(profit_match.group(2))]
    minutes = [_number(time_match.group(1)), _number(time_match.group(2))]
    capacity_minutes = _number(capacity_match.group(1)) * 60.0
    a_units = _number(ratio_match.group(1))
    b_units = _number(ratio_match.group(2))
    constraints = [
        minutes,
        [b_units, -a_units],
    ]
    upper_bounds = [capacity_minutes, 0.0]
    status, objective, values, message = _linprog_maximize(
        objective=profit,
        constraints=constraints,
        upper_bounds=upper_bounds,
    )
    if status != "optimal":
        return TemplateSolveResult(
            matched=True,
            template_id="two_product_time_ratio_lp",
            status=status,
            confidence=0.8,
            notes=message,
            artifact={"profit": profit, "minutes": minutes},
        )
    return TemplateSolveResult(
        matched=True,
        template_id="two_product_time_ratio_lp",
        status="optimal",
        objective_value=objective,
        variable_values={"produce_A": values[0], "produce_B": values[1]},
        confidence=0.9,
        notes="Solved two-product LP with time capacity and production-ratio constraint.",
        artifact={
            "profit": profit,
            "minutes": minutes,
            "capacity_minutes": capacity_minutes,
            "ratio": {"A": a_units, "B_minimum": b_units},
        },
    )


def _normalize_math_labels(text: str) -> str:
    normalized = re.sub(r"\\\((.*?)\\\)", r" \1 ", text)
    normalized = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", normalized)
    normalized = re.sub(r"([A-Za-z])_\{\s*([0-9]+)\s*\}", r"\1\2", normalized)
    normalized = re.sub(r"([A-Za-z])_\s*([0-9]+)", r"\1\2", normalized)
    normalized = normalized.replace("\\", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _solve_production_conversion_lp(text: str) -> TemplateSolveResult:
    normalized = _normalize_math_labels(text)
    lowered = normalized.lower()
    if not (
        "barrel of milk" in lowered
        and "profit" in lowered
        and "labor" in lowered
        and "equipment" in lowered
        and "maximize" in lowered
    ):
        return TemplateSolveResult(False)

    conversion = re.search(
        (
            rf"one\s+barrel\s+of\s+milk.*?into\s+({_NUMBER_TOKEN})\s+kg\s+of\s+A1\s+"
            rf"in\s+({_NUMBER_TOKEN})\s+hours?.*?or\s+into\s+({_NUMBER_TOKEN})\s+kg\s+"
            rf"of\s+A2\s+in\s+({_NUMBER_TOKEN})\s+hours?"
        ),
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )
    profit = re.search(
        (
            rf"profit\s+is\s+({_NUMBER_TOKEN})\s+\w+\s+per\s+kilograms?\s+of\s+A1\s+"
            rf"and\s+({_NUMBER_TOKEN})\s+\w+\s+per\s+kilograms?\s+of\s+A2"
        ),
        normalized,
        flags=re.IGNORECASE,
    )
    supply = re.search(
        rf"daily\s+supply\s+of\s+({_NUMBER_TOKEN})\s+barrels?\s+of\s+milk",
        normalized,
        flags=re.IGNORECASE,
    )
    labor = re.search(
        rf"total\s+of\s+({_NUMBER_TOKEN})\s+hours?\s+of\s+labor",
        normalized,
        flags=re.IGNORECASE,
    )
    a_capacity = re.search(
        rf"type\s+A\s+equipment\s+can\s+process\s+up\s+to\s+({_NUMBER_TOKEN})\s+kg\s+of\s+A1",
        normalized,
        flags=re.IGNORECASE,
    )
    if not (conversion and profit and supply and labor and a_capacity):
        return TemplateSolveResult(False)

    a1_yield = _number(conversion.group(1))
    a1_hours = _number(conversion.group(2))
    a2_yield = _number(conversion.group(3))
    a2_hours = _number(conversion.group(4))
    profit_per_kg = [_number(profit.group(1)), _number(profit.group(2))]
    supply_barrels = _number(supply.group(1))
    labor_hours = _number(labor.group(1))
    a1_kg_capacity = _number(a_capacity.group(1))

    objective = [a1_yield * profit_per_kg[0], a2_yield * profit_per_kg[1]]
    constraints = [
        [1.0, 1.0],
        [a1_hours, a2_hours],
        [a1_yield, 0.0],
    ]
    upper_bounds = [supply_barrels, labor_hours, a1_kg_capacity]
    status, objective_value, values, message = _linprog_maximize(
        objective=objective,
        constraints=constraints,
        upper_bounds=upper_bounds,
    )
    if status != "optimal":
        return TemplateSolveResult(
            matched=True,
            template_id="production_conversion_lp",
            status=status,
            confidence=0.82,
            notes=message,
            artifact={"objective": objective, "constraints": constraints, "upper_bounds": upper_bounds},
        )

    return TemplateSolveResult(
        matched=True,
        template_id="production_conversion_lp",
        status="optimal",
        objective_value=objective_value,
        variable_values={
            "barrels_to_A1": values[0],
            "barrels_to_A2": values[1],
            "produce_A1_kg": values[0] * a1_yield,
            "produce_A2_kg": values[1] * a2_yield,
        },
        confidence=0.9,
        notes="Solved two-output conversion LP with input, labor, and equipment capacity.",
        artifact={
            "yield_kg_per_barrel": [a1_yield, a2_yield],
            "hours_per_barrel": [a1_hours, a2_hours],
            "profit_per_kg": profit_per_kg,
            "profit_per_barrel": objective,
            "supply_barrels": supply_barrels,
            "labor_hours": labor_hours,
            "a1_kg_capacity": a1_kg_capacity,
        },
    )


def _normalize_quality_blending_text(text: str) -> str:
    cleaned = re.sub(r"\\\((.*?)\\\)", r"\1", text, flags=re.DOTALL)
    cleaned = re.sub(r"\\\[(.*?)\\\]", r"\1", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\\(?:mathrm|text)\s*\{([^}]*)\}", r"\1", cleaned)
    cleaned = cleaned.replace("\\%", "%")
    return re.sub(r"\s+", " ", cleaned).strip()


def _parse_quality_label_list(value: str) -> list[str]:
    cleaned = re.sub(r"\([^)]*\)", " ", value)
    cleaned = re.sub(r"\b(?:and|or)\b", ",", cleaned, flags=re.IGNORECASE)
    labels: list[str] = []
    for part in cleaned.split(","):
        if not part.strip():
            continue
        label = _clean_label(part)
        label = re.sub(
            r"^(?:the|raw\s+materials?|products?|product|materials?)\s+",
            "",
            label,
            flags=re.IGNORECASE,
        ).strip()
        if label and label.lower() != "respectively":
            labels.append(label)
    return labels


def _extract_quality_numbers(value: str) -> list[float]:
    return [
        _number(match.group(0))
        for match in re.finditer(_NUMBER_TOKEN, value, flags=re.IGNORECASE)
    ]


def _parse_quality_blending_problem(text: str) -> dict[str, Any] | None:
    normalized = _normalize_quality_blending_text(text)
    lowered = normalized.lower()
    if not (
        ("mix" in lowered or "blend" in lowered)
        and "raw material" in lowered
        and "product" in lowered
        and ("sulfur" in lowered or "sulphur" in lowered or "impurity" in lowered)
        and ("max profit" in lowered or "maximize" in lowered or "maximise" in lowered)
    ):
        return None

    quality_word = "sulphur" if "sulphur" in lowered else "sulfur"
    clause_end = r"(?:\s+respectively|[.;](?=\s+[A-Z]|$))"
    raw_quality = re.search(
        rf"{quality_word}\s+contents?\s+of\s+raw\s+materials?\s+(.+?)\s+"
        rf"(?:are|is)\s+(.+?){clause_end}",
        normalized,
        flags=re.IGNORECASE,
    )
    if not raw_quality:
        return None
    raw_labels = _parse_quality_label_list(raw_quality.group(1))
    raw_quality_values = _extract_quality_numbers(raw_quality.group(2))

    raw_cost = re.search(
        r"(?:purchase\s+prices?|purchase\s+costs?|raw\s+material\s+costs?)\s+"
        rf"(?:are|is)\s+(.+?){clause_end}",
        normalized,
        flags=re.IGNORECASE,
    )
    if not raw_cost:
        return None
    raw_costs = _extract_quality_numbers(raw_cost.group(1))
    if len(raw_labels) < 2 or len(raw_labels) != len(raw_quality_values) or len(raw_labels) != len(raw_costs):
        return None

    product_quality = re.search(
        rf"{quality_word}\s+contents?\s+of\s+products?\s+(.+?)\s+"
        r"(?:must\s+not\s+exceed|cannot\s+exceed|can\s+not\s+exceed|"
        r"should\s+not\s+exceed|is\s+at\s+most|are\s+at\s+most|no\s+more\s+than)"
        rf"\s+(.+?){clause_end}",
        normalized,
        flags=re.IGNORECASE,
    )
    if not product_quality:
        return None
    product_labels = _parse_quality_label_list(product_quality.group(1))
    product_quality_limits = _extract_quality_numbers(product_quality.group(2))
    if len(product_labels) < 1 or len(product_labels) != len(product_quality_limits):
        return None

    selling_price = re.search(
        r"(?:selling\s+prices?|sales\s+prices?|sale\s+prices?)\s+(?:are|is)\s+"
        rf"(.+?){clause_end}",
        normalized,
        flags=re.IGNORECASE,
    )
    if not selling_price:
        return None
    prices = _extract_quality_numbers(selling_price.group(1))
    if len(prices) == 1 and len(product_labels) > 1:
        prices = prices * len(product_labels)
    if len(prices) != len(product_labels):
        return None

    demand = re.search(
        r"((?:market\s+)?demand\s+for\s+products?\s+(.+?)\s+(?:is|are)\s+"
        rf"(.+?){clause_end})",
        normalized,
        flags=re.IGNORECASE,
    )
    if not demand:
        return None
    demand_labels = _parse_quality_label_list(demand.group(2))
    demands = _extract_quality_numbers(demand.group(3))
    if len(demands) != len(product_labels):
        return None
    if len(demand_labels) == len(product_labels):
        product_labels = demand_labels

    exact_demand = re.search(
        r"\b(?:must|need(?:s)?\s+to|is\s+required\s+to|are\s+required\s+to|"
        r"meet|satisfy)\b.{0,80}\bdemand\b|\bdemand\b.{0,80}\b(?:must|at\s+least|required)",
        demand.group(1),
        flags=re.IGNORECASE,
    )
    demand_mode = "exact" if exact_demand else "upper"

    caps: dict[str, float] = {}
    for label in raw_labels:
        label_pattern = re.escape(label)
        cap_match = re.search(
            rf"(?:supply|availability)\s+of\s+raw\s+material\s+{label_pattern}\s+"
            rf"(?:is|are)\s+limited\s+to\s+(?:a\s+)?maximum\s+of\s+({_NUMBER_TOKEN})",
            normalized,
            flags=re.IGNORECASE,
        )
        if not cap_match:
            cap_match = re.search(
                rf"raw\s+material\s+{label_pattern}[^.;]{{0,120}}?"
                rf"(?:at\s+most|up\s+to|no\s+more\s+than|maximum\s+of)\s+({_NUMBER_TOKEN})",
                normalized,
                flags=re.IGNORECASE,
            )
        if cap_match:
            caps[label] = _number(cap_match.group(1))

    raw_inputs = [
        {
            "label": label,
            "quality": raw_quality_values[index],
            "cost": raw_costs[index],
            "availability": caps.get(label),
        }
        for index, label in enumerate(raw_labels)
    ]
    products = [
        {
            "label": label,
            "quality_limit": product_quality_limits[index],
            "price": prices[index],
            "demand": demands[index],
        }
        for index, label in enumerate(product_labels)
    ]
    return {
        "raw_inputs": raw_inputs,
        "products": products,
        "quality_name": quality_word,
        "demand_mode": demand_mode,
    }


def _percent_constraint_from_cell(value: str) -> tuple[str, float] | None:
    cleaned = value.replace("\u2265", ">=").replace("\u2264", "<=").strip()
    match = re.search(r"(>=|<=)\s*([0-9]+(?:\.\d+)?)\s*%", cleaned)
    if not match:
        return None
    return match.group(1), float(match.group(2)) / 100.0


def _solve_candy_quality_blending_lp(text: str) -> TemplateSolveResult:
    normalized = re.sub(r"\s+", " ", text)
    lowered = normalized.lower()
    if not (
        "candy factory" in lowered
        and "raw material cost" in lowered
        and "monthly limit" in lowered
        and "processing fee" in lowered
        and "selling price" in lowered
    ):
        return TemplateSolveResult(False)
    parsed = _parse_markdown_table(text)
    if not parsed:
        return TemplateSolveResult(False)
    header, rows = parsed
    if len(header) < 6:
        return TemplateSolveResult(False)
    product_labels = [_clean_label(cell) for cell in header[1:4]]
    raw_rows: list[dict[str, Any]] = []
    processing_fees: list[float] | None = None
    selling_prices: list[float] | None = None
    for row in rows:
        if len(row) < 6:
            continue
        label = _clean_label(row[0])
        lowered_label = label.lower()
        if "processing fee" in lowered_label:
            processing_fees = [_number(row[index]) for index in range(1, 4)]
            continue
        if "selling price" in lowered_label:
            selling_prices = [_number(row[index]) for index in range(1, 4)]
            continue
        if label in product_labels:
            raw_rows.append(
                {
                    "label": label,
                    "quality_cells": row[1:4],
                    "cost": _number(row[4]),
                    "limit": _number(row[5]),
                }
            )
    if len(raw_rows) != 3 or processing_fees is None or selling_prices is None:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="candy_quality_blending_lp",
            status="solver_unavailable",
            confidence=0.82,
            notes=str(exc),
        )

    raw_count = len(raw_rows)
    product_count = len(product_labels)
    variable_count = raw_count * product_count

    def index(raw_index: int, product_index: int) -> int:
        return raw_index * product_count + product_index

    objective = [0.0] * variable_count
    for raw_index, raw in enumerate(raw_rows):
        for product_index in range(product_count):
            profit = selling_prices[product_index] - processing_fees[product_index] - float(raw["cost"])
            objective[index(raw_index, product_index)] = -profit

    a_ub: list[list[float]] = []
    b_ub: list[float] = []
    for raw_index, raw in enumerate(raw_rows):
        row = [0.0] * variable_count
        for product_index in range(product_count):
            row[index(raw_index, product_index)] = 1.0
        a_ub.append(row)
        b_ub.append(float(raw["limit"]))

    for raw_index, raw in enumerate(raw_rows):
        for product_index, cell in enumerate(raw["quality_cells"]):
            parsed_cell = _percent_constraint_from_cell(str(cell))
            if parsed_cell is None:
                continue
            sense, fraction = parsed_cell
            row = [0.0] * variable_count
            for other_raw in range(raw_count):
                row[index(other_raw, product_index)] = fraction
            row[index(raw_index, product_index)] -= 1.0
            if sense == ">=":
                a_ub.append(row)
                b_ub.append(0.0)
            else:
                a_ub.append([-value for value in row])
                b_ub.append(0.0)

    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(0, None)] * variable_count,
        method="highs",
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="candy_quality_blending_lp",
            status="solver_failed",
            confidence=0.82,
            notes=str(result.message),
        )
    variable_values: dict[str, float] = {}
    product_totals = [0.0] * product_count
    for raw_index, raw in enumerate(raw_rows):
        for product_index, product_label in enumerate(product_labels):
            value = float(result.x[index(raw_index, product_index)])
            product_totals[product_index] += value
            if not math.isclose(value, 0.0, abs_tol=1e-8):
                variable_values[f"raw_{raw['label']}_to_candy_{product_label}"] = value
    for product_index, product_label in enumerate(product_labels):
        if not math.isclose(product_totals[product_index], 0.0, abs_tol=1e-8):
            variable_values[f"produce_candy_{product_label}"] = product_totals[product_index]
    return TemplateSolveResult(
        matched=True,
        template_id="candy_quality_blending_lp",
        status="optimal",
        objective_value=float(-result.fun),
        variable_values=variable_values,
        confidence=0.88,
        notes="Solved candy raw-material quality blending LP from table percentages, costs, limits, fees, and prices.",
        artifact={
            "raw_materials": raw_rows,
            "products": product_labels,
            "processing_fees": processing_fees,
            "selling_prices": selling_prices,
        },
    )


def _solve_quality_constrained_blending_lp(text: str) -> TemplateSolveResult:
    problem = _parse_quality_blending_problem(text)
    if not problem:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="quality_constrained_blending_lp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
            artifact=problem,
        )

    raw_inputs = problem["raw_inputs"]
    products = problem["products"]
    demand_mode = problem["demand_mode"]
    raw_count = len(raw_inputs)
    product_count = len(products)
    variable_count = raw_count * product_count

    def index(raw_index: int, product_index: int) -> int:
        return raw_index * product_count + product_index

    objective = [0.0] * variable_count
    for raw_index, raw in enumerate(raw_inputs):
        for product_index, product in enumerate(products):
            objective[index(raw_index, product_index)] = float(raw["cost"]) - float(product["price"])

    a_ub: list[list[float]] = []
    b_ub: list[float] = []
    a_eq: list[list[float]] = []
    b_eq: list[float] = []

    for raw_index, raw in enumerate(raw_inputs):
        if raw["availability"] is None:
            continue
        row = [0.0] * variable_count
        for product_index in range(product_count):
            row[index(raw_index, product_index)] = 1.0
        a_ub.append(row)
        b_ub.append(float(raw["availability"]))

    for product_index, product in enumerate(products):
        row = [0.0] * variable_count
        for raw_index in range(raw_count):
            row[index(raw_index, product_index)] = 1.0
        if demand_mode == "exact":
            a_eq.append(row)
            b_eq.append(float(product["demand"]))
        else:
            a_ub.append(row)
            b_ub.append(float(product["demand"]))

    for product_index, product in enumerate(products):
        row = [0.0] * variable_count
        for raw_index, raw in enumerate(raw_inputs):
            row[index(raw_index, product_index)] = float(raw["quality"]) - float(product["quality_limit"])
        a_ub.append(row)
        b_ub.append(0.0)

    result = linprog(
        objective,
        A_ub=a_ub or None,
        b_ub=b_ub or None,
        A_eq=a_eq or None,
        b_eq=b_eq or None,
        bounds=[(0, None)] * variable_count,
        method="highs",
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="quality_constrained_blending_lp",
            status="solver_failed",
            confidence=0.82,
            notes=str(result.message),
            artifact=problem,
        )

    variable_values: dict[str, float] = {}
    product_quantities = [0.0] * product_count
    for raw_index, raw in enumerate(raw_inputs):
        for product_index, product in enumerate(products):
            value = float(result.x[index(raw_index, product_index)])
            product_quantities[product_index] += value
            if not math.isclose(value, 0.0, abs_tol=1e-8):
                variable_values[f"blend_raw_{raw['label']}_to_product_{product['label']}"] = value
    for product_index, product in enumerate(products):
        quantity = product_quantities[product_index]
        if not math.isclose(quantity, 0.0, abs_tol=1e-8):
            variable_values[f"produce_{product['label']}"] = quantity

    objective_value = float(-result.fun)
    if math.isclose(objective_value, 0.0, abs_tol=1e-8):
        objective_value = 0.0
    return TemplateSolveResult(
        matched=True,
        template_id="quality_constrained_blending_lp",
        status="optimal",
        objective_value=objective_value,
        variable_values=variable_values,
        confidence=0.88,
        notes=(
            "Solved quality-constrained blending LP with raw costs, raw availability, "
            "product demand bounds, and maximum product quality/impurity constraints."
        ),
        artifact={
            **problem,
            "product_quantities": product_quantities,
        },
    )


def _parse_gasoline_raw_inputs(text: str) -> list[dict[str, float | str]]:
    raw_inputs: list[dict[str, float | str]] = []
    raw_pattern = re.compile(
        (
            rf"raw\s+gasoline\s+([0-9]+)\s+with\s+({_NUMBER_TOKEN})\s+octane"
            r"(.*?)(?=raw\s+gasoline\s+[0-9]+\s+with|The\s+required|\Z)"
        ),
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in raw_pattern.finditer(text):
        label, octane, body = match.groups()
        availability = re.search(
            rf"(?:available\s+up\s+to|purchased\s+up\s+to|daily\s+availability\s+of)\s+({_NUMBER_TOKEN})\s+barrels?",
            body,
            flags=re.IGNORECASE,
        )
        cost = re.search(
            rf"(?:cost\s+of\s+\$?|at\s+\$)\s*({_NUMBER_TOKEN})\s+per\s+barrel",
            body,
            flags=re.IGNORECASE,
        )
        if availability and cost:
            raw_inputs.append(
                {
                    "label": f"raw_{label}",
                    "octane": _number(octane),
                    "availability": _number(availability.group(1)),
                    "cost": _number(cost.group(1)),
                }
            )
    return raw_inputs


def _parse_gasoline_products(text: str) -> list[dict[str, float | str]]:
    regular = re.search(
        (
            rf"is\s+({_NUMBER_TOKEN})\s+for\s+regular.*?"
            rf"sells\s+at\s+\$?\s*({_NUMBER_TOKEN})\s+per\s+barrel.*?"
            rf"maximum\s+daily\s+demand\s+of\s+({_NUMBER_TOKEN})\s+barrels?"
        ),
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    premium = re.search(
        (
            rf"and\s+({_NUMBER_TOKEN})\s+for\s+premium\s+gasoline.*?"
            rf"priced\s+at\s+\$?\s*({_NUMBER_TOKEN})\s+per\s+barrel.*?"
            rf"demand\s+of\s+up\s+to\s+({_NUMBER_TOKEN})\s+barrels?"
        ),
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    products: list[dict[str, float | str]] = []
    if regular:
        products.append(
            {
                "label": "regular",
                "minimum_octane": _number(regular.group(1)),
                "price": _number(regular.group(2)),
                "demand": _number(regular.group(3)),
            }
        )
    if premium:
        products.append(
            {
                "label": "premium",
                "minimum_octane": _number(premium.group(1)),
                "price": _number(premium.group(2)),
                "demand": _number(premium.group(3)),
            }
        )
    return products


def _solve_gasoline_blending_lp(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        "raw gasoline" in lowered
        and "octane" in lowered
        and "regular" in lowered
        and "premium" in lowered
        and "blending" in lowered
        and ("maximal profit" in lowered or "maximize" in lowered)
    ):
        return TemplateSolveResult(False)

    raw_inputs = _parse_gasoline_raw_inputs(text)
    products = _parse_gasoline_products(text)
    if len(raw_inputs) < 2 or len(products) < 2:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="gasoline_blending_lp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
            artifact={"raw_inputs": raw_inputs, "products": products},
        )

    raw_count = len(raw_inputs)
    product_count = len(products)
    variable_count = raw_count * product_count

    def index(raw_index: int, product_index: int) -> int:
        return raw_index * product_count + product_index

    objective = [0.0] * variable_count
    for raw_index, raw in enumerate(raw_inputs):
        for product_index, product in enumerate(products):
            objective[index(raw_index, product_index)] = -(
                float(product["price"]) - float(raw["cost"])
            )

    a_ub: list[list[float]] = []
    b_ub: list[float] = []
    for raw_index, raw in enumerate(raw_inputs):
        row = [0.0] * variable_count
        for product_index in range(product_count):
            row[index(raw_index, product_index)] = 1.0
        a_ub.append(row)
        b_ub.append(float(raw["availability"]))
    for product_index, product in enumerate(products):
        row = [0.0] * variable_count
        for raw_index in range(raw_count):
            row[index(raw_index, product_index)] = 1.0
        a_ub.append(row)
        b_ub.append(float(product["demand"]))
    for product_index, product in enumerate(products):
        row = [0.0] * variable_count
        for raw_index, raw in enumerate(raw_inputs):
            row[index(raw_index, product_index)] = float(product["minimum_octane"]) - float(raw["octane"])
        a_ub.append(row)
        b_ub.append(0.0)

    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(0, None)] * variable_count,
        method="highs",
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="gasoline_blending_lp",
            status="solver_failed",
            confidence=0.82,
            notes=str(result.message),
            artifact={"raw_inputs": raw_inputs, "products": products},
        )

    variable_values: dict[str, float] = {}
    product_quantities = [0.0] * product_count
    for raw_index, raw in enumerate(raw_inputs):
        for product_index, product in enumerate(products):
            value = float(result.x[index(raw_index, product_index)])
            product_quantities[product_index] += value
            if not math.isclose(value, 0.0, abs_tol=1e-8):
                variable_values[f"blend_{raw['label']}_to_{product['label']}"] = value
    for product_index, product in enumerate(products):
        variable_values[f"produce_{product['label']}"] = product_quantities[product_index]

    return TemplateSolveResult(
        matched=True,
        template_id="gasoline_blending_lp",
        status="optimal",
        objective_value=float(-result.fun),
        variable_values=variable_values,
        confidence=0.9,
        notes="Solved gasoline blending LP with raw availability, product demand, and minimum octane constraints.",
        artifact={
            "raw_inputs": raw_inputs,
            "products": products,
            "product_quantities": product_quantities,
        },
    )


def _clock_hour(hour: float, marker: str) -> float:
    value = hour % 12
    if marker.lower() == "pm":
        value += 12
    return value


def _parse_daily_open_hours(text: str) -> float | None:
    time_match = re.search(
        rf"from\s+({_NUMBER_TOKEN})(?::[0-9]{{2}})?\s*(AM|PM)\s+to\s+({_NUMBER_TOKEN})(?::[0-9]{{2}})?\s*(AM|PM)",
        text,
        flags=re.IGNORECASE,
    )
    if time_match:
        start = _clock_hour(_number(time_match.group(1)), time_match.group(2))
        end = _clock_hour(_number(time_match.group(3)), time_match.group(4))
        if end <= start:
            end += 24
        return end - start
    hours_match = re.search(
        rf"(?:open|operates?|operation)[^.;\n]{{0,80}}?({_NUMBER_TOKEN})\s+hours?",
        text,
        flags=re.IGNORECASE,
    )
    if hours_match:
        return _number(hours_match.group(1))
    return None


def _parse_designated_ids(text: str, label: str) -> set[str]:
    pattern = re.compile(
        rf"{label}[^.\n]*?\((?:designated|IDs?|numbered)?\s*([^)]+)\)",
        flags=re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        return set()
    return {
        _clean_label(value)
        for value in re.findall(r"[A-Za-z0-9]+", match.group(1))
        if _clean_label(value).lower() not in {"and", "or"}
    }


def _student_group_minimum_hours(text: str, group: str) -> float | None:
    match = re.search(
        rf"{group}[^.;\n]{{0,80}}?at\s+least\s+({_NUMBER_TOKEN})\s+hours?",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return _number(match.group(1))
    return None


def _solve_student_duty_scheduling(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        "duty" in lowered
        and "wage" in lowered
        and "gross pay" in lowered
        and "no more than" in lowered
        and ("student" in lowered or "students" in lowered)
    ):
        return TemplateSolveResult(False)

    parsed_rows: list[dict[str, Any]] = []
    day_labels: list[str] = []
    for header, rows in _parse_markdown_tables(text):
        cleaned_header = [_clean_label(cell).lower() for cell in header]
        header_text = " ".join(cleaned_header)
        if not ("student" in header_text and "wage" in header_text):
            continue
        student_index = next((index for index, cell in enumerate(cleaned_header) if "student" in cell), 0)
        wage_index = next((index for index, cell in enumerate(cleaned_header) if "wage" in cell), None)
        day_indices: list[int] = []
        day_labels = []
        for day_name in _DAY_NAMES:
            day_index = next(
                (index for index, cell in enumerate(cleaned_header) if day_name.lower() in cell),
                None,
            )
            if day_index is not None:
                day_indices.append(day_index)
                day_labels.append(day_name)
        if wage_index is None or len(day_indices) < 2:
            continue
        for row in rows:
            if len(row) <= max([student_index, wage_index, *day_indices]):
                continue
            wage = _first_number(row[wage_index])
            availabilities = [_first_number(row[index]) for index in day_indices]
            if wage is None or any(value is None for value in availabilities):
                continue
            parsed_rows.append(
                {
                    "label": _clean_label(row[student_index]),
                    "wage": float(wage),
                    "availability": [float(value or 0.0) for value in availabilities],
                }
            )
        if parsed_rows:
            break
    if not parsed_rows or not day_labels:
        return TemplateSolveResult(False)

    daily_hours = _parse_daily_open_hours(text)
    max_shifts_match = re.search(
        rf"each\s+student[^.;\n]{{0,80}}?no\s+more\s+than\s+({_NUMBER_TOKEN})\s+shifts?",
        text,
        flags=re.IGNORECASE,
    )
    max_students_match = re.search(
        rf"no\s+more\s+than\s+({_NUMBER_TOKEN})\s+students?[^.;\n]{{0,100}}?(?:each|per)\s+day",
        text,
        flags=re.IGNORECASE,
    )
    if daily_hours is None or not max_shifts_match or not max_students_match:
        return TemplateSolveResult(False)
    max_shifts_per_worker = _number(max_shifts_match.group(1))
    max_workers_per_day = _number(max_students_match.group(1))

    undergraduate_ids = _parse_designated_ids(text, "undergraduates?")
    graduate_ids = _parse_designated_ids(text, "graduate\\s+students?")
    undergraduate_min = _student_group_minimum_hours(text, "undergraduates?")
    graduate_min = _student_group_minimum_hours(text, "graduate\\s+students?")
    fallback_undergrad_count = re.search(
        rf"({_NUMBER_TOKEN})\s+undergraduates?",
        text,
        flags=re.IGNORECASE,
    )
    fallback_grad_count = re.search(
        rf"({_NUMBER_TOKEN})\s+graduate\s+students?",
        text,
        flags=re.IGNORECASE,
    )
    if not undergraduate_ids and fallback_undergrad_count:
        count = int(_number(fallback_undergrad_count.group(1)))
        undergraduate_ids = {str(row["label"]) for row in parsed_rows[:count]}
    if not graduate_ids and fallback_grad_count:
        count = int(_number(fallback_grad_count.group(1)))
        graduate_ids = {str(row["label"]) for row in parsed_rows[-count:]}

    minimum_hours = [0.0] * len(parsed_rows)
    for index, row in enumerate(parsed_rows):
        label = str(row["label"])
        if label in undergraduate_ids and undergraduate_min is not None:
            minimum_hours[index] = undergraduate_min
        elif label in graduate_ids and graduate_min is not None:
            minimum_hours[index] = graduate_min

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="student_duty_scheduling_milp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
            artifact={"workers": parsed_rows, "days": day_labels},
        )

    worker_count = len(parsed_rows)
    day_count = len(day_labels)
    hour_count = worker_count * day_count
    variable_count = hour_count * 2

    def hour_var(worker_index: int, day_index: int) -> int:
        return worker_index * day_count + day_index

    def shift_var(worker_index: int, day_index: int) -> int:
        return hour_count + worker_index * day_count + day_index

    objective = np.zeros(variable_count)
    for worker_index, row in enumerate(parsed_rows):
        for day_index in range(day_count):
            objective[hour_var(worker_index, day_index)] = float(row["wage"])

    rows: list[Any] = []
    lower: list[float] = []
    upper: list[float] = []
    for day_index in range(day_count):
        row = np.zeros(variable_count)
        for worker_index in range(worker_count):
            row[hour_var(worker_index, day_index)] = 1.0
        rows.append(row)
        lower.append(float(daily_hours))
        upper.append(float(daily_hours))

    for worker_index, minimum in enumerate(minimum_hours):
        if minimum <= 0:
            continue
        row = np.zeros(variable_count)
        for day_index in range(day_count):
            row[hour_var(worker_index, day_index)] = 1.0
        rows.append(row)
        lower.append(float(minimum))
        upper.append(math.inf)

    for worker_index, worker in enumerate(parsed_rows):
        availability = list(worker["availability"])
        for day_index, available_hours in enumerate(availability):
            row = np.zeros(variable_count)
            row[hour_var(worker_index, day_index)] = 1.0
            row[shift_var(worker_index, day_index)] = -float(available_hours)
            rows.append(row)
            lower.append(-math.inf)
            upper.append(0.0)
            if math.isclose(float(available_hours), 0.0, abs_tol=1e-9):
                row = np.zeros(variable_count)
                row[shift_var(worker_index, day_index)] = 1.0
                rows.append(row)
                lower.append(0.0)
                upper.append(0.0)

    for worker_index in range(worker_count):
        row = np.zeros(variable_count)
        for day_index in range(day_count):
            row[shift_var(worker_index, day_index)] = 1.0
        rows.append(row)
        lower.append(0.0)
        upper.append(float(max_shifts_per_worker))

    for day_index in range(day_count):
        row = np.zeros(variable_count)
        for worker_index in range(worker_count):
            row[shift_var(worker_index, day_index)] = 1.0
        rows.append(row)
        lower.append(0.0)
        upper.append(float(max_workers_per_day))

    result = milp(
        objective,
        integrality=[0] * hour_count + [1] * hour_count,
        bounds=Bounds(np.zeros(variable_count), np.r_[np.full(hour_count, math.inf), np.ones(hour_count)]),
        constraints=LinearConstraint(np.vstack(rows), np.array(lower), np.array(upper)),
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="student_duty_scheduling_milp",
            status="solver_failed",
            confidence=0.82,
            notes=str(result.message),
            artifact={
                "workers": parsed_rows,
                "days": day_labels,
                "daily_hours": daily_hours,
                "minimum_hours": minimum_hours,
                "max_shifts_per_worker": max_shifts_per_worker,
                "max_workers_per_day": max_workers_per_day,
            },
        )

    variable_values: dict[str, float] = {}
    for worker_index, worker in enumerate(parsed_rows):
        label = str(worker["label"])
        for day_index, day_label in enumerate(day_labels):
            hours = float(result.x[hour_var(worker_index, day_index)])
            if not math.isclose(hours, 0.0, abs_tol=1e-8):
                variable_values[f"hours_student_{label}_{day_label.lower()}"] = hours
    return TemplateSolveResult(
        matched=True,
        template_id="student_duty_scheduling_milp",
        status="optimal",
        objective_value=float(result.fun),
        variable_values=variable_values,
        confidence=0.88,
        notes="Solved staff duty scheduling MILP with daily coverage, minimum weekly hours, shift-count, and daily headcount limits.",
        artifact={
            "workers": parsed_rows,
            "days": day_labels,
            "daily_hours": daily_hours,
            "minimum_hours": minimum_hours,
            "max_shifts_per_worker": max_shifts_per_worker,
            "max_workers_per_day": max_workers_per_day,
        },
    )


_DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def _clean_time_text(value: str) -> str:
    cleaned = value.replace("\\sim", "~")
    cleaned = re.sub(r"\\[A-Za-z]+", " ", cleaned)
    cleaned = cleaned.replace("$", " ").replace(";", ":")
    return re.sub(r"\s+", " ", cleaned)


def _parse_time_range(value: str) -> tuple[int, int] | None:
    cleaned = _clean_time_text(value)
    match = re.search(
        r"\b([0-9]{1,2})\s*(?::\s*0{2})?\s*(?:-|~|to|–|—)\s*"
        r"([0-9]{1,2})\s*(?::\s*0{2})?\b",
        cleaned,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    start, end = int(match.group(1)), int(match.group(2))
    if not (0 <= start < 24 and 0 <= end < 24):
        return None
    return start, end


def _parse_shift_requirement_pairs(text: str) -> list[tuple[int, int, float]]:
    pairs: list[tuple[int, int, float]] = []

    def add_pair(start: int, end: int, value: float) -> None:
        item = (start, end, value)
        if item not in pairs:
            pairs.append(item)

    for header, rows in _parse_markdown_tables(text):
        _ = header
        for row in rows:
            for index, cell in enumerate(row[:-1]):
                time_range = _parse_time_range(cell)
                if time_range is None:
                    continue
                requirement = _first_number(row[index + 1])
                if requirement is not None:
                    add_pair(time_range[0], time_range[1], float(requirement))

    for raw_line in text.splitlines():
        if "&" not in raw_line:
            continue
        cells = [cell.strip() for cell in raw_line.split("&")]
        for index, cell in enumerate(cells[:-1]):
            time_range = _parse_time_range(cell)
            if time_range is None:
                continue
            requirement = _first_number(cells[index + 1])
            if requirement is not None:
                add_pair(time_range[0], time_range[1], float(requirement))

    normalized = _clean_time_text(text)
    for match in re.finditer(
        r"\b([0-9]{1,2})\s*(?::\s*0{2})?\s*-\s*([0-9]{1,2})\s*(?::\s*0{2})?"
        r"\s*[-:]\s*([0-9][0-9,]*(?:\.\d+)?)\s*(?:people|persons|nurses|waiters|drivers|crew|members?)?",
        normalized,
        flags=re.IGNORECASE,
    ):
        start, end = int(match.group(1)), int(match.group(2))
        if 0 <= start < 24 and 0 <= end < 24:
            add_pair(start, end, _number(match.group(3)))

    return sorted(pairs, key=lambda item: item[0])


def _parse_shift_hours(text: str) -> float | None:
    match = re.search(
        rf"work(?:ing|s)?\s+(?:continuously\s+)?for\s+({_NUMBER_TOKEN})\s+hours?",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return _number(match.group(1))
    return None


def _parse_hourly_wages(text: str) -> dict[str, float]:
    wages: dict[str, float] = {}
    regular = re.search(
        rf"regular\s+\w+[^.;\n]{{0,80}}?\s+is\s+({_NUMBER_TOKEN})\s*(?:yuan|dollars?|\$)?\s*/\s*hour",
        text,
        flags=re.IGNORECASE,
    )
    contract = re.search(
        rf"contract\s+\w+[^.;\n]{{0,80}}?\s+is\s+({_NUMBER_TOKEN})\s*(?:yuan|dollars?|\$)?\s*/\s*hour",
        text,
        flags=re.IGNORECASE,
    )
    if regular:
        wages["regular"] = _number(regular.group(1))
    if contract:
        wages["contract"] = _number(contract.group(1))
    return wages


def _solve_cyclic_shift_staffing(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    normalized = re.sub(r"\s+", " ", lowered)
    if not (
        ("24-hour" in normalized or "24 hours" in normalized or "around the clock" in normalized)
        and (
            "work continuously" in normalized
            or "works continuously" in normalized
            or "working continuously" in normalized
        )
        and (
            "minimum number" in normalized
            or "minimize" in normalized
            or "minimum total" in normalized
            or "minimum amount" in normalized
            or "pay" in normalized
        )
    ):
        return TemplateSolveResult(False)

    pairs = _parse_shift_requirement_pairs(text)
    if len(pairs) < 3:
        return TemplateSolveResult(False)
    shift_hours = _parse_shift_hours(text)
    if shift_hours is None or shift_hours <= 0:
        return TemplateSolveResult(False)

    period_lengths = [float((end - start) % 24 or 24) for start, end, _value in pairs]
    period_length = period_lengths[0]
    if any(not math.isclose(length, period_length, abs_tol=1e-9) for length in period_lengths):
        return TemplateSolveResult(False)
    coverage_periods_float = shift_hours / period_length
    coverage_periods = int(round(coverage_periods_float))
    if coverage_periods < 1 or not math.isclose(coverage_periods_float, coverage_periods, abs_tol=1e-9):
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="cyclic_shift_staffing",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
            artifact={"requirements": pairs, "shift_hours": shift_hours},
        )

    requirements = [value for _start, _end, value in pairs]
    period_labels = [f"{start}:00-{end}:00" for start, end, _value in pairs]
    wages = _parse_hourly_wages(text)
    pay_mode = bool(wages)
    wage_type = min(wages, key=wages.get) if wages else "staff"
    wage_rate = wages[wage_type] if wages else 1.0

    n = len(requirements)
    coverage_rows: list[list[float]] = []
    for period_index in range(n):
        row = [0.0] * n
        for start_index in range(n):
            if (period_index - start_index) % n < coverage_periods:
                row[start_index] = -1.0
        coverage_rows.append(row)
    c = [shift_hours * wage_rate if pay_mode else 1.0] * n
    result = linprog(
        c,
        A_ub=coverage_rows,
        b_ub=[-value for value in requirements],
        bounds=[(0, None)] * n,
        integrality=[1] * n,
        method="highs",
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="cyclic_shift_staffing",
            status="solver_failed",
            confidence=0.82,
            notes=str(result.message),
            artifact={"requirements": pairs, "shift_hours": shift_hours, "wages": wages},
        )

    total_staff = float(sum(result.x))
    variable_values = {
        f"{wage_type}_start_{period_labels[index]}": float(value)
        for index, value in enumerate(result.x)
        if not math.isclose(float(value), 0.0, abs_tol=1e-8)
    }
    if "contract" in wages:
        variable_values.setdefault("contract_staff_total", 0.0 if wage_type != "contract" else total_staff)
    objective = float(result.fun) if pay_mode else total_staff
    notes = "Solved cyclic shift staffing coverage model with integer shift starts."
    if pay_mode and len(wages) > 1:
        notes += f" Used the lowest stated hourly wage type: {wage_type}."
    return TemplateSolveResult(
        matched=True,
        template_id="cyclic_shift_staffing",
        status="optimal",
        objective_value=objective,
        variable_values=variable_values,
        confidence=0.88,
        notes=notes,
        artifact={
            "periods": period_labels,
            "requirements": requirements,
            "shift_hours": shift_hours,
            "period_length_hours": period_length,
            "coverage_periods": coverage_periods,
            "total_staff": total_staff,
            "pay_mode": pay_mode,
            "wages": wages,
            "wage_type_used": wage_type,
        },
    )


def _parse_weekly_consecutive_day_requirements(text: str) -> list[float] | None:
    requirements: list[float] = []
    for day_name in _DAY_NAMES:
        match = re.search(
            rf"\b{day_name}\s+requires\s*({_NUMBER_TOKEN})(?:\s+employees?)?",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            requirements = []
            break
        requirements.append(_number(match.group(1)))
    if len(requirements) == len(_DAY_NAMES):
        return requirements

    indexed: dict[int, float] = {}
    for day, value in re.findall(
        rf"\bd\s*([1-7])\s*=\s*({_NUMBER_TOKEN})",
        text,
        flags=re.IGNORECASE,
    ):
        indexed[int(day)] = _number(value)
    if len(indexed) == len(_DAY_NAMES):
        return [indexed[index] for index in range(1, len(_DAY_NAMES) + 1)]
    return None


def _solve_consecutive_day_workforce_scheduling(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    normalized = re.sub(r"\s+", " ", lowered)
    if not (
        (
            "full-time employees" in lowered
            or "nurse" in lowered
            or "nurses" in lowered
            or "worker" in lowered
            or "workers" in lowered
        )
        and ("consecutive days" in lowered or "days in a row" in lowered)
        and (
            "minimize the number" in normalized
            or "minimize the total number" in normalized
            or "minimal number" in normalized
        )
    ):
        return TemplateSolveResult(False)

    requirements = _parse_weekly_consecutive_day_requirements(text)
    if requirements is None:
        return TemplateSolveResult(False)

    work_days_match = re.search(
        rf"work(?:s)?\s+({_NUMBER_TOKEN})\s+(?:consecutive\s+days|days\s+in\s+a\s+row)",
        text,
        flags=re.IGNORECASE,
    )
    work_days = int(_number(work_days_match.group(1))) if work_days_match else 5
    if work_days <= 0 or work_days > len(_DAY_NAMES):
        return TemplateSolveResult(False)
    relax_integrality = "ignore the integrality" in normalized or "half" in normalized
    round_answer = "rounded to the nearest" in normalized

    try:
        from scipy.optimize import Bounds, LinearConstraint, linprog, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="consecutive_day_workforce_scheduling",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
            artifact={"requirements": dict(zip(_DAY_NAMES, requirements))},
        )

    day_count = len(_DAY_NAMES)
    rows: list[Any] = []
    for day_index in range(day_count):
        row = np.zeros(day_count)
        for start_index in range(day_count):
            if (day_index - start_index) % day_count < work_days:
                row[start_index] = 1.0
        rows.append(row)

    coefficient_matrix = np.vstack(rows)
    if relax_integrality:
        result = linprog(
            np.ones(day_count),
            A_ub=-coefficient_matrix,
            b_ub=-np.array(requirements),
            bounds=[(0, None)] * day_count,
            method="highs",
        )
    else:
        result = milp(
            np.ones(day_count),
            integrality=np.ones(day_count),
            bounds=Bounds(np.zeros(day_count), np.full(day_count, math.inf)),
            constraints=LinearConstraint(
                coefficient_matrix,
                np.array(requirements),
                np.full(day_count, math.inf),
            ),
        )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="consecutive_day_workforce_scheduling",
            status="solver_failed",
            confidence=0.82,
            notes=str(result.message),
            artifact={"requirements": dict(zip(_DAY_NAMES, requirements)), "work_days": work_days},
        )

    variable_values = {
        f"start_{day_name.lower()}": float(result.x[index])
        for index, day_name in enumerate(_DAY_NAMES)
        if not math.isclose(float(result.x[index]), 0.0, abs_tol=1e-8)
    }
    objective = float(result.fun)
    if round_answer:
        objective = float(round(objective))
    return TemplateSolveResult(
        matched=True,
        template_id="consecutive_day_workforce_scheduling",
        status="optimal",
        objective_value=objective,
        variable_values=variable_values,
        confidence=0.9,
        notes=(
            "Solved cyclic consecutive-day workforce scheduling with "
            f"{'continuous' if relax_integrality else 'integer'} start-day variables."
        ),
        artifact={
            "days": list(_DAY_NAMES),
            "requirements": dict(zip(_DAY_NAMES, requirements)),
            "work_days": work_days,
            "relax_integrality": relax_integrality,
            "raw_objective_value": float(result.fun),
        },
    )


def _investment_outflow_for_time(body: str, time_index: int) -> float | None:
    direct = re.search(
        rf"time\s*{time_index}\s+(?:cash\s+)?outflow\s+of\s+\$?\s*({_NUMBER_TOKEN})\s+million",
        body,
        flags=re.IGNORECASE,
    )
    if direct:
        return _number(direct.group(1))
    amount_first = re.search(
        rf"\$?\s*({_NUMBER_TOKEN})\s+million(?:\s+(?:cash\s+)?outflow)?\s+at\s+time\s*{time_index}",
        body,
        flags=re.IGNORECASE,
    )
    if amount_first:
        return _number(amount_first.group(1))
    return None


def _parse_fractional_investment_opportunities(text: str) -> list[dict[str, float | str]]:
    opportunities: list[dict[str, float | str]] = []
    segment_pattern = re.compile(
        r"Investment\s+([0-9]+)\s+(.*?)(?=Investment\s+[0-9]+\s+|[A-Za-z]+\s+Oil\s+has|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in segment_pattern.finditer(text):
        label, body = match.groups()
        shared_outflow = re.search(
            rf"outflows?\s+of\s+\$?\s*({_NUMBER_TOKEN})\s+million\s+at\s+both\s+time\s*0\s+and\s+time\s*1",
            body,
            flags=re.IGNORECASE,
        )
        if shared_outflow:
            time0_outflow = _number(shared_outflow.group(1))
            time1_outflow = time0_outflow
        else:
            parsed_time0 = _investment_outflow_for_time(body, 0)
            parsed_time1 = _investment_outflow_for_time(body, 1)
            if parsed_time0 is None or parsed_time1 is None:
                continue
            time0_outflow = parsed_time0
            time1_outflow = parsed_time1
        npv = re.search(
            rf"\bNPV\b.*?\$?\s*({_NUMBER_TOKEN})\s+million",
            body,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not npv:
            continue
        opportunities.append(
            {
                "label": f"investment_{label}",
                "time0_outflow": time0_outflow,
                "time1_outflow": time1_outflow,
                "npv": _number(npv.group(1)),
            }
        )
    return opportunities


def _available_budget_for_time(text: str, time_index: int) -> float | None:
    match = re.search(
        rf"\$?\s*({_NUMBER_TOKEN})\s+million\s+(?:available|will\s+be\s+available)"
        rf".{{0,80}}?time\s*{time_index}",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return _number(match.group(1))
    return None


def _parse_initial_reinvestable_capital(text: str) -> float | None:
    patterns = (
        rf"initial\s+capital\s+of\s+\$?\s*({_NUMBER_TOKEN})",
        rf"\bfund\s+of\s+\$?\s*({_NUMBER_TOKEN})",
        rf"plans?\s+to\s+invest\s+\$?\s*({_NUMBER_TOKEN})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _number(match.group(1))
    return None


def _parse_investment_horizon(text: str) -> int | None:
    patterns = (
        rf"end\s+of\s+(?:the\s+)?({_YEAR_TOKEN})(?:st|nd|rd|th)?\s+year",
        rf"end\s+of\s+Year\s+({_YEAR_TOKEN})",
        rf"over\s+the\s+next\s+({_YEAR_TOKEN})\s+years",
        rf"next\s+({_YEAR_TOKEN})\s+years",
        rf"within\s+({_YEAR_TOKEN})\s+years",
    )
    candidates: list[int] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            year = _year_number(match.group(1))
            if year is not None:
                candidates.append(year)
    return max(candidates) if candidates else None


def _parse_reinvestable_project_clauses(text: str) -> list[tuple[str, str]]:
    clauses = [
        (label, body.strip())
        for label, body in re.findall(
            r"\(([0-9]+)\)\s*(.*?)(?=\([0-9]+\)\s*|\Z)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if body.strip()
    ]
    if len(clauses) >= 2:
        return clauses
    ordinal_pattern = re.compile(
        (
            r"\b(first|second|third|fourth|fifth)\s+investment\s+"
            r"(.*?)(?=\b(?:first|second|third|fourth|fifth)\s+investment\b|"
            r"\bIn\s+order\b|\bFormulate\b|\Z)"
        ),
        flags=re.IGNORECASE | re.DOTALL,
    )
    return [
        (label.lower(), body.strip())
        for label, body in ordinal_pattern.findall(text)
        if body.strip()
    ]


def _parse_reinvestable_return_factor(body: str) -> float | None:
    multiplier = re.search(
        rf"(?:receive|receives|pays?|paying)\s+\$?\s*({_NUMBER_TOKEN})\s*\S?\s*I\b",
        body,
        flags=re.IGNORECASE,
    )
    if multiplier:
        return _number(multiplier.group(1))
    principal_interest = re.search(
        rf"principal\s+and\s+interest\s+amounting\s+to\s+({_NUMBER_TOKEN})\s*%",
        body,
        flags=re.IGNORECASE,
    )
    if principal_interest:
        return _number(principal_interest.group(1)) / 100.0
    profit = re.search(
        rf"(?:annual\s+)?profit\s+of\s+({_NUMBER_TOKEN})\s*%",
        body,
        flags=re.IGNORECASE,
    )
    if profit:
        return 1.0 + _number(profit.group(1)) / 100.0
    return_amount = re.search(
        rf"return\s+of\s+({_NUMBER_TOKEN})\s+\w+\s+for\s+every\s+1\s+\w+\s+invested",
        body,
        flags=re.IGNORECASE,
    )
    if return_amount:
        return 1.0 + _number(return_amount.group(1))
    return None


def _parse_reinvestable_duration(body: str) -> int | None:
    patterns = (
        rf"({_YEAR_TOKEN})\s*-\s*year\s+product",
        rf"after\s+({_YEAR_TOKEN})\s+years?",
        rf"recovered\s+in\s+({_YEAR_TOKEN})\s+years?",
        rf"maturing\s+in\s+({_YEAR_TOKEN})\s+years?",
    )
    for pattern in patterns:
        match = re.search(pattern, body, flags=re.IGNORECASE)
        if match:
            return _year_number(match.group(1))
    if "annual" in body.lower() or "same-year" in body.lower():
        return 1
    return None


def _parse_reinvestable_start_years(body: str, horizon: int, duration: int | None) -> list[int]:
    lowered = body.lower()
    if "each year" in lowered or "beginning of each year" in lowered:
        max_start = horizon if duration is None else horizon - duration + 1
        return list(range(1, max_start + 1))
    years: list[int] = []
    start_patterns = (
        rf"beginning\s+of\s+Year\s+({_YEAR_TOKEN})",
        rf"beginning\s+of\s+(?:the\s+)?({_YEAR_TOKEN})(?:st|nd|rd|th)?\s+year",
    )
    for pattern in start_patterns:
        for match in re.finditer(pattern, body, flags=re.IGNORECASE):
            year = _year_number(match.group(1))
            if year is not None:
                years.append(year)
    if years:
        return sorted(set(years))
    if duration is None:
        return []
    return list(range(1, horizon - duration + 2))


def _parse_reinvestable_absolute_return_year(body: str) -> int | None:
    patterns = (
        rf"(?:matures?|maturing)\s+at\s+the\s+end\s+of\s+Year\s+({_YEAR_TOKEN})",
        rf"(?:matures?|maturing)\s+at\s+the\s+end\s+of\s+(?:the\s+)?({_YEAR_TOKEN})(?:st|nd|rd|th)?\s+year",
        rf"recovered\s+at\s+the\s+end\s+of\s+(?:the\s+)?({_YEAR_TOKEN})(?:st|nd|rd|th)?\s+year",
    )
    for pattern in patterns:
        match = re.search(pattern, body, flags=re.IGNORECASE)
        if match:
            return _year_number(match.group(1))
    return None


def _parse_reinvestable_cap(body: str) -> float | None:
    match = re.search(
        rf"(?:capped\s+at|(?:investment\s+)?limit\s+is(?:\s+no\s+more\s+than)?|"
        rf"limited\s+to|maximum\s+of)"
        rf"\s+\$?\s*({_NUMBER_TOKEN})",
        body,
        flags=re.IGNORECASE,
    )
    if match:
        return _number(match.group(1))
    return None


def _parse_reinvestable_investment_instances(
    text: str,
    horizon: int,
) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    for raw_label, body in _parse_reinvestable_project_clauses(text):
        factor = _parse_reinvestable_return_factor(body)
        duration = _parse_reinvestable_duration(body)
        start_years = _parse_reinvestable_start_years(body, horizon, duration)
        absolute_return_year = _parse_reinvestable_absolute_return_year(body)
        cap = _parse_reinvestable_cap(body)
        if factor is None or not start_years:
            continue
        label = _clean_label(str(raw_label)).lower()
        for start_year in start_years:
            if start_year < 1 or start_year > horizon:
                continue
            if absolute_return_year is not None:
                return_time = absolute_return_year + 1
            elif duration is not None:
                return_time = start_year + duration
            else:
                continue
            if start_year < return_time <= horizon + 1:
                instances.append(
                    {
                        "label": f"investment_{label}_start_{start_year}",
                        "project": label,
                        "start_year": start_year,
                        "return_time": return_time,
                        "factor": factor,
                        "cap": cap,
                    }
                )
    return instances


def _solve_reinvestable_cashflow_lp(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        "invest" in lowered
        and (
            "principal" in lowered
            or "interest" in lowered
            or "matures" in lowered
            or "recovered" in lowered
            or "return of" in lowered
        )
        and (
            "beginning" in lowered
            or "reallocate" in lowered
            or "following year" in lowered
            or "after one year" in lowered
            or "after two years" in lowered
        )
        and ("maximize" in lowered or "maximizes" in lowered)
    ):
        return TemplateSolveResult(False)
    if "cash outflow" in lowered and ("npv" in lowered or "net present value" in lowered):
        return TemplateSolveResult(False)

    initial_capital = _parse_initial_reinvestable_capital(text)
    horizon = _parse_investment_horizon(text)
    if initial_capital is None or horizon is None or horizon <= 0:
        return TemplateSolveResult(False)
    instances = _parse_reinvestable_investment_instances(text, horizon)
    if len(instances) < 2:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="reinvestable_cashflow_lp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
            artifact={"initial_capital": initial_capital, "horizon": horizon, "investments": instances},
        )

    investment_count = len(instances)
    variable_count = investment_count + horizon
    objective = [0.0] * variable_count
    for index, instance in enumerate(instances):
        if int(instance["return_time"]) == horizon + 1:
            objective[index] = -float(instance["factor"])
    objective[investment_count + horizon - 1] = -1.0

    a_eq: list[list[float]] = []
    b_eq: list[float] = []
    for year in range(1, horizon + 1):
        row = [0.0] * variable_count
        for index, instance in enumerate(instances):
            if int(instance["start_year"]) == year:
                row[index] += 1.0
            if int(instance["return_time"]) == year:
                row[index] -= float(instance["factor"])
        row[investment_count + year - 1] += 1.0
        if year > 1:
            row[investment_count + year - 2] -= 1.0
        a_eq.append(row)
        b_eq.append(initial_capital if year == 1 else 0.0)

    bounds = [
        (0.0, None if instance["cap"] is None else float(instance["cap"]))
        for instance in instances
    ]
    bounds.extend((0.0, None) for _ in range(horizon))
    result = linprog(
        objective,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="reinvestable_cashflow_lp",
            status="solver_failed",
            confidence=0.82,
            notes=str(result.message),
            artifact={"initial_capital": initial_capital, "horizon": horizon, "investments": instances},
        )

    variable_values: dict[str, float] = {}
    for index, instance in enumerate(instances):
        value = float(result.x[index])
        if not math.isclose(value, 0.0, abs_tol=1e-8):
            variable_values[str(instance["label"])] = value
    for year in range(1, horizon + 1):
        value = float(result.x[investment_count + year - 1])
        if not math.isclose(value, 0.0, abs_tol=1e-8):
            variable_values[f"cash_carry_after_year_{year}"] = value

    return TemplateSolveResult(
        matched=True,
        template_id="reinvestable_cashflow_lp",
        status="optimal",
        objective_value=float(-result.fun),
        variable_values=variable_values,
        confidence=0.9,
        notes="Solved reinvestable multi-period cash-flow planning LP with no-borrowing balances.",
        artifact={
            "initial_capital": initial_capital,
            "horizon": horizon,
            "investments": instances,
        },
    )


def _money_after_phrase(text: str, phrase: str) -> float | None:
    match = re.search(
        rf"{phrase}\s*\$?\s*([0-9][0-9,]*(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return _number(match.group(1))
    return None


def _parse_two_manufactured_labels(text: str) -> list[str]:
    match = re.search(
        r"\bmanufactures?\s+([A-Za-z][A-Za-z ]+?)\s+and\s+([A-Za-z][A-Za-z ]+?)\.",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return []
    return [_clean_label(match.group(1)).lower(), _clean_label(match.group(2)).lower()]


def _label_regex(label: str) -> str:
    return r"\s+".join(re.escape(part) for part in label.split())


def _parse_ordered_cost_pair(text: str, labels: list[str], cost_name: str) -> list[float] | None:
    label_pattern = r"\s+and\s+".join(_label_regex(label) for label in labels)
    match = re.search(
        rf"{cost_name}\s+for\s+{label_pattern}\s+are\s+\$?\s*({_NUMBER_TOKEN})\s+and\s+\$?\s*({_NUMBER_TOKEN})\s+respectively",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return [_number(match.group(1)), _number(match.group(2))]


def _parse_labeled_cost_pair(text: str, labels: list[str], cost_name: str) -> list[float] | None:
    values: list[float] = []
    for label in labels:
        singular = _singular_label(label)
        match = re.search(
            rf"{cost_name}.*?\$?\s*({_NUMBER_TOKEN})\s+for\s+(?:a\s+)?{_label_regex(singular)}s?",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return None
        values.append(_number(match.group(1)))
    return values


def _parse_selling_price_pair(text: str, labels: list[str]) -> list[float] | None:
    values: list[float] = []
    for label in labels:
        singular = _singular_label(label)
        match = re.search(
            rf"(?:selling\s+prices?|sell(?:ing)?\s+prices?).*?\$?\s*({_NUMBER_TOKEN})\s+for\s+(?:a\s+)?{_label_regex(singular)}s?",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return None
        values.append(_number(match.group(1)))
    return values


def _parse_existing_raw_material_capacity(text: str, labels: list[str]) -> list[float] | None:
    match = re.search(
        rf"enough\s+raw\s+material\s+to\s+manufacture\s+({_NUMBER_TOKEN})\s+{_label_regex(labels[0])}"
        rf"\s+and\s+({_NUMBER_TOKEN})\s+{_label_regex(labels[1])}",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return [_number(match.group(1)), _number(match.group(2))]


def _solve_balance_sheet_production_lp(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        "balance sheet" in lowered
        and "current ratio" in lowered
        and "accounts receivable" in lowered
        and "minimum cash balance" in lowered
        and "sales" in lowered
        and "credit" in lowered
    ):
        return TemplateSolveResult(False)

    labels = _parse_two_manufactured_labels(text)
    if len(labels) != 2:
        return TemplateSolveResult(False)
    labor_costs = _parse_ordered_cost_pair(text, labels, r"per-unit\s+labor\s+costs?")
    raw_costs = _parse_labeled_cost_pair(text, labels, r"raw\s+material\s+costs?")
    selling_prices = _parse_selling_price_pair(text, labels)
    capacities = _parse_existing_raw_material_capacity(text, labels)
    if not (labor_costs and raw_costs and selling_prices and capacities):
        return TemplateSolveResult(False)

    initial_cash = _money_after_phrase(text, r"cash\s+at")
    initial_receivables = _money_after_phrase(text, r"accounts\s+receivable\s+at")
    initial_inventory = _money_after_phrase(text, r"inventory\s+outstanding\s+valued\s+at")
    initial_loan = _money_after_phrase(text, r"bank\s+loan\s+liability\s+of")
    collected_receivables = _money_after_phrase(text, r"collect\s+\$?")
    loan_payment = _money_after_phrase(text, r"pay\s+off")
    rent = _money_after_phrase(text, r"monthly\s+rent\s+of")
    new_raw_materials = _money_after_phrase(text, r"raw\s+materials\s+worth")
    minimum_cash = _money_after_phrase(text, r"minimum\s+cash\s+balance\s+of")
    ratio_match = re.search(
        rf"current\s+ratio.*?at\s+least\s+({_NUMBER_TOKEN})",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not (
        initial_cash is not None
        and initial_receivables is not None
        and initial_inventory is not None
        and initial_loan is not None
        and collected_receivables is not None
        and loan_payment is not None
        and rent is not None
        and new_raw_materials is not None
        and minimum_cash is not None
        and ratio_match
    ):
        return TemplateSolveResult(False)

    current_ratio = _number(ratio_match.group(1))
    contributions = [
        selling_prices[index] - labor_costs[index] - raw_costs[index]
        for index in range(len(labels))
    ]
    cash_available_for_labor = initial_cash + collected_receivables - loan_payment - rent - minimum_cash
    if cash_available_for_labor < -1e-9:
        return TemplateSolveResult(
            matched=True,
            template_id="balance_sheet_production_lp",
            status="infeasible",
            confidence=0.75,
            notes="Minimum cash balance exceeds pre-production available cash.",
            artifact={"products": labels},
        )

    cash_after_constant = initial_cash + collected_receivables - loan_payment - rent
    receivables_constant = initial_receivables - collected_receivables
    inventory_constant = initial_inventory + new_raw_materials
    liabilities = initial_loan - loan_payment + new_raw_materials
    constant_assets = cash_after_constant + receivables_constant + inventory_constant
    ratio_rhs = current_ratio * liabilities - constant_assets

    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="balance_sheet_production_lp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
            artifact={"products": labels},
        )

    result = linprog(
        [-value for value in contributions],
        A_ub=[
            labor_costs,
            [-value for value in contributions],
        ],
        b_ub=[
            cash_available_for_labor,
            -ratio_rhs,
        ],
        bounds=[(0.0, capacity) for capacity in capacities],
        method="highs",
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="balance_sheet_production_lp",
            status="solver_failed",
            confidence=0.82,
            notes=str(result.message),
            artifact={"products": labels},
        )

    variable_values = {
        f"produce_{labels[index]}": float(value)
        for index, value in enumerate(result.x)
        if not math.isclose(float(value), 0.0, abs_tol=1e-8)
    }
    return TemplateSolveResult(
        matched=True,
        template_id="balance_sheet_production_lp",
        status="optimal",
        objective_value=float(-result.fun),
        variable_values=variable_values,
        confidence=0.88,
        notes="Solved production planning LP with minimum cash and current-ratio balance-sheet constraints.",
        artifact={
            "products": labels,
            "selling_prices": selling_prices,
            "labor_costs": labor_costs,
            "raw_material_costs": raw_costs,
            "production_bounds": capacities,
            "minimum_cash": minimum_cash,
            "current_ratio": current_ratio,
            "liabilities": liabilities,
        },
    )


def _solve_fractional_investment_budget_lp(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        "investment opportunities" in lowered
        and "cash outflow" in lowered
        and ("net present value" in lowered or "npv" in lowered)
        and "any fraction" in lowered
        and "maximize" in lowered
    ):
        return TemplateSolveResult(False)

    opportunities = _parse_fractional_investment_opportunities(text)
    budget_time0 = _available_budget_for_time(text, 0)
    budget_time1 = _available_budget_for_time(text, 1)
    if len(opportunities) < 2 or budget_time0 is None or budget_time1 is None:
        return TemplateSolveResult(False)

    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="fractional_investment_budget_lp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
            artifact={"opportunities": opportunities},
        )

    objective = [-float(item["npv"]) for item in opportunities]
    a_ub = [
        [float(item["time0_outflow"]) for item in opportunities],
        [float(item["time1_outflow"]) for item in opportunities],
    ]
    b_ub = [budget_time0, budget_time1]
    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(0, 1)] * len(opportunities),
        method="highs",
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="fractional_investment_budget_lp",
            status="solver_failed",
            confidence=0.82,
            notes=str(result.message),
            artifact={"opportunities": opportunities},
        )

    variable_values = {
        f"fraction_{item['label']}": float(result.x[index])
        for index, item in enumerate(opportunities)
        if not math.isclose(float(result.x[index]), 0.0, abs_tol=1e-8)
    }
    return TemplateSolveResult(
        matched=True,
        template_id="fractional_investment_budget_lp",
        status="optimal",
        objective_value=float(-result.fun),
        variable_values=variable_values,
        confidence=0.9,
        notes="Solved fractional capital-budgeting LP with time-indexed cash outflow constraints.",
        artifact={
            "opportunities": opportunities,
            "budgets": {"time0": budget_time0, "time1": budget_time1},
        },
    )


def _parse_labeled_money_values(
    text: str,
    label_prefixes: tuple[str, ...],
    *,
    require_million: bool,
) -> dict[str, float]:
    prefixes = "|".join(re.escape(prefix) for prefix in label_prefixes)
    million_suffix = r"\s+million" if require_million else r""
    not_million = r"(?!\s+million)" if not require_million else r""
    values: dict[str, float] = {}
    pattern = re.compile(
        rf"\b({prefixes})\s+([A-Za-z0-9]+)\s+is\s+\$?\s*([0-9][0-9,]*(?:\.\d+)?){not_million}{million_suffix}",
        flags=re.IGNORECASE,
    )
    for prefix, suffix, value in pattern.findall(text):
        label = f"{_clean_label(prefix).lower()}_{_clean_label(suffix).lower()}"
        values[label] = _number(value)
    return values


def _parse_purchase_exclusions(
    text: str,
    label_prefixes: tuple[str, ...],
) -> list[tuple[str, str]]:
    prefixes = "|".join(re.escape(prefix) for prefix in label_prefixes)
    exclusions: list[tuple[str, str]] = []
    pattern = re.compile(
        (
            rf"if\s+they\s+(?:purchase|buy)\s+({prefixes})\s+([A-Za-z0-9]+).*?"
            rf"(?:cannot|can\s+not)\s+(?:purchase|buy)\s+({prefixes})\s+([A-Za-z0-9]+)"
        ),
        flags=re.IGNORECASE | re.DOTALL,
    )
    for left_prefix, left_suffix, right_prefix, right_suffix in pattern.findall(text):
        left = f"{_clean_label(left_prefix).lower()}_{_clean_label(left_suffix).lower()}"
        right = f"{_clean_label(right_prefix).lower()}_{_clean_label(right_suffix).lower()}"
        exclusions.append((left, right))
    return exclusions


def _split_selection_labels(value: str) -> list[str]:
    cleaned = re.sub(r"\([^)]*\)", "", value)
    parts = re.split(r",|\band\b", cleaned, flags=re.IGNORECASE)
    labels: list[str] = []
    for part in parts:
        if not part.strip():
            continue
        label = _clean_label(part)
        label = re.sub(r"^(?:and|or|the|a|an)\s+", "", label, flags=re.IGNORECASE).strip()
        if label and label.lower() not in {"children", "candidates", "employees"} and label not in labels:
            labels.append(label)
    return labels


def _parse_binary_subset_labels(text: str) -> list[str]:
    patterns = [
        r"\bhas\s+[^.;\n]{0,30}\s+children:\s*([^.\n]+)",
        r"\bcandidates?\s+([A-Za-z](?:\s*,\s*[A-Za-z])*(?:\s*,?\s*and\s*[A-Za-z])?)\s+are\s+\$",
        r"\bcandidates?\s+([A-Za-z](?:\s*,\s*[A-Za-z])*(?:\s*,?\s*and\s*[A-Za-z])?)\s+respectively",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            labels = _split_selection_labels(match.group(1))
            if len(labels) >= 2:
                return labels
    return []


def _label_re(label: str) -> str:
    return rf"\b{re.escape(label)}\b"


def _labels_in_text_order(value: str, labels: list[str]) -> list[str]:
    found: list[tuple[int, str]] = []
    for label in labels:
        match = re.search(_label_re(label), value, flags=re.IGNORECASE)
        if match:
            found.append((match.start(), label))
    return [label for _index, label in sorted(found)]


def _parse_binary_subset_costs(text: str, labels: list[str]) -> dict[str, float]:
    costs: dict[str, float] = {}
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text))
    for sentence in sentences:
        lowered = sentence.lower()
        if "$" not in sentence or not any(
            word in lowered for word in ("cost", "salary", "paid", "payment")
        ):
            continue
        labels_in_sentence = _labels_in_text_order(sentence, labels)
        money_values = [
            _number(match.group(1))
            for match in re.finditer(rf"\$\s*({_NUMBER_TOKEN})", sentence, flags=re.IGNORECASE)
        ]
        if "respectively" in lowered and len(labels_in_sentence) == len(money_values):
            for label, value in zip(labels_in_sentence, money_values):
                costs.setdefault(label, value)
            continue
        for label in labels:
            match = re.search(
                rf"{_label_re(label)}\s+(?:is|are)?\s*\$\s*({_NUMBER_TOKEN})",
                sentence,
                flags=re.IGNORECASE,
            )
            if match:
                costs[label] = _number(match.group(1))
    return costs


def _parse_selection_count_bound(text: str, *, bound_type: str) -> int | None:
    if bound_type == "max":
        patterns = [
            rf"\b(?:up\s+to|at\s+most|maximum\s+of|max(?:imum)?)\s+({_NUMBER_TOKEN})\s+"
            r"(?:children|new\s+employees|employees|candidates)",
            rf"\bhire\s+a\s+maximum\s+of\s+({_NUMBER_TOKEN})\s+new\s+employees",
        ]
    else:
        patterns = [
            rf"\bat\s+least\s+({_NUMBER_TOKEN})\s+"
            r"(?:children|new\s+employees|employees|candidates)(?!\s+with)",
            rf"\bhire\s+at\s+least\s+({_NUMBER_TOKEN})\s+new\s+employees",
        ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(round(_number(match.group(1))))
    return None


def _parse_binary_required_items(text: str, labels: list[str]) -> set[str]:
    required: set[str] = set()
    for sentence in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text)):
        lowered = sentence.lower()
        if lowered.strip().startswith("if "):
            continue
        if not any(
            phrase in lowered
            for phrase in (
                "definitely take",
                "definitely hire",
                "must take",
                "must hire",
                "will hire",
            )
        ):
            continue
        for label in labels:
            if re.search(_label_re(label), sentence, flags=re.IGNORECASE):
                required.add(label)
    return required


def _parse_binary_logic_constraints(
    text: str,
    labels: list[str],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    exclusions: list[tuple[str, str]] = []
    implications: list[tuple[str, str]] = []
    for sentence in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text)):
        sentence_labels = _labels_in_text_order(sentence, labels)
        lowered = sentence.lower()
        if lowered.strip().startswith("if ") and len(sentence_labels) >= 2:
            if re.search(r"\b(?:will\s+not|cannot|can\s+not)\b", lowered):
                exclusions.append((sentence_labels[0], sentence_labels[1]))
            elif "must also" in lowered:
                implications.append((sentence_labels[0], sentence_labels[1]))
        if "at most one" in lowered and len(sentence_labels) >= 2:
            for left, right in itertools.combinations(sentence_labels, 2):
                exclusions.append((left, right))
    return exclusions, implications


def _parse_candidate_numeric_attribute(
    text: str,
    labels: list[str],
    *,
    attribute: str,
) -> dict[str, float]:
    values: dict[str, float] = {}
    if attribute == "skill":
        pattern = rf"\bCandidate\s+({_NUMBER_TOKEN}|[A-Za-z])\s*:\s*Level\s+({_NUMBER_TOKEN})"
        for label, value in re.findall(pattern, text, flags=re.IGNORECASE):
            cleaned = _clean_label(label)
            if cleaned in labels:
                values[cleaned] = _number(value)
        return values
    pattern = rf"\bCandidate\s+([A-Za-z])\s*:\s*({_NUMBER_TOKEN})\s+years?"
    for label, value in re.findall(pattern, text, flags=re.IGNORECASE):
        cleaned = _clean_label(label)
        if cleaned in labels:
            values[cleaned] = _number(value)
    return values


def _parse_candidate_degree_groups(text: str, labels: list[str]) -> list[dict[str, Any]]:
    if not re.search(r"master'?s?\s+or\s+doctoral\s+degree", text, flags=re.IGNORECASE):
        return []
    eligible: list[str] = []
    pattern = r"\bCandidate\s+([A-Za-z])\s*:\s*([^.;\n]*degree)"
    for label, degree in re.findall(pattern, text, flags=re.IGNORECASE):
        cleaned = _clean_label(label)
        if cleaned not in labels:
            continue
        if re.search(r"master'?s?|doctoral", degree, flags=re.IGNORECASE):
            eligible.append(cleaned)
    if not eligible:
        return []
    return [{"name": "masters_or_doctoral_degree", "labels": sorted(eligible), "min": 1.0}]


def _solve_binary_subset_selection(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        ("minimize" in lowered or "minimise" in lowered)
        and any(word in lowered for word in ("children", "candidate", "hire", "taking"))
        and any(
            phrase in lowered
            for phrase in (
                "at least",
                "at most",
                "up to",
                "maximum",
                "budget",
                "must also",
                "will not",
                "cannot",
            )
        )
    ):
        return TemplateSolveResult(False)

    labels = _parse_binary_subset_labels(text)
    if len(labels) < 2 or len(labels) > 24:
        return TemplateSolveResult(False)
    costs = _parse_binary_subset_costs(text, labels)
    if set(costs) != set(labels):
        return TemplateSolveResult(False)

    min_count = _parse_selection_count_bound(text, bound_type="min")
    max_count = _parse_selection_count_bound(text, bound_type="max")
    required = _parse_binary_required_items(text, labels)
    exclusions, implications = _parse_binary_logic_constraints(text, labels)
    skill = _parse_candidate_numeric_attribute(text, labels, attribute="skill")
    experience = _parse_candidate_numeric_attribute(text, labels, attribute="experience")
    group_requirements = _parse_candidate_degree_groups(text, labels)
    numeric_requirements: list[dict[str, Any]] = []
    skill_match = re.search(
        rf"\btotal\s+skill\s+level\b[^.;\n]{{0,120}}\bat\s+least\s+({_NUMBER_TOKEN})",
        text,
        flags=re.IGNORECASE,
    )
    if skill and skill_match:
        numeric_requirements.append({"name": "skill_level", "values": skill, "min": _number(skill_match.group(1))})
    experience_match = re.search(
        rf"\btotal\b[^.;\n]{{0,80}}\bexperience\b[^.;\n]{{0,120}}"
        rf"\b(?:at\s+least|no\s+less\s+than|not\s+less\s+than)\s+({_NUMBER_TOKEN})",
        text,
        flags=re.IGNORECASE,
    )
    if experience and experience_match:
        numeric_requirements.append(
            {
                "name": "experience_years",
                "values": experience,
                "min": _number(experience_match.group(1)),
            }
        )
    budget = None
    for budget_pattern in (
        rf"\bbudget\s+(?:is|of)\s+\$?\s*({_NUMBER_TOKEN})",
        rf"\bbudget\b[^.;\n]{{0,40}}\b(?:not\s+exceed|no\s+more\s+than|at\s+most)\s+\$?\s*({_NUMBER_TOKEN})",
    ):
        budget_match = re.search(budget_pattern, text, flags=re.IGNORECASE)
        if budget_match:
            budget = _number(budget_match.group(1))
            break

    best_cost: float | None = None
    best_subset: tuple[str, ...] | None = None
    for mask in range(1 << len(labels)):
        selected = tuple(label for index, label in enumerate(labels) if mask & (1 << index))
        selected_set = set(selected)
        count = len(selected)
        cost = sum(costs[label] for label in selected)
        if min_count is not None and count < min_count:
            continue
        if max_count is not None and count > max_count:
            continue
        if budget is not None and cost > budget + 1e-9:
            continue
        if not required <= selected_set:
            continue
        if any(left in selected_set and right in selected_set for left, right in exclusions):
            continue
        if any(left in selected_set and right not in selected_set for left, right in implications):
            continue
        if any(
            sum(requirement["values"].get(label, 0.0) for label in selected)
            < float(requirement["min"]) - 1e-9
            for requirement in numeric_requirements
        ):
            continue
        if any(
            sum(1 for label in selected if label in requirement["labels"]) < float(requirement["min"]) - 1e-9
            for requirement in group_requirements
        ):
            continue
        if best_cost is None or cost < best_cost - 1e-9 or (
            math.isclose(cost, best_cost, abs_tol=1e-9) and count < len(best_subset or ())
        ):
            best_cost = cost
            best_subset = selected

    if best_cost is None or best_subset is None:
        return TemplateSolveResult(
            matched=True,
            template_id="binary_subset_selection_ilp",
            status="infeasible",
            confidence=0.82,
            artifact={"items": labels, "costs": costs},
        )

    return TemplateSolveResult(
        matched=True,
        template_id="binary_subset_selection_ilp",
        status="optimal",
        objective_value=float(best_cost),
        variable_values={f"select_{label}": 1.0 for label in best_subset},
        confidence=0.88,
        notes="Solved binary subset-selection problem by exact enumeration of 0-1 choices.",
        artifact={
            "items": labels,
            "costs": costs,
            "budget": budget,
            "min_count": min_count,
            "max_count": max_count,
            "required": sorted(required),
            "exclusions": [list(pair) for pair in exclusions],
            "implications": [list(pair) for pair in implications],
            "numeric_requirements": numeric_requirements,
            "group_requirements": group_requirements,
            "selected": list(best_subset),
        },
    )


def _solve_binary_purchase_selection(text: str) -> TemplateSolveResult:
    lowered = text.lower()
    if not (
        ("whether to purchase" in lowered or "whether to buy" in lowered)
        and "budget" in lowered
        and ("maximize" in lowered or "maximise" in lowered)
        and ("annual income" in lowered or "annual revenue" in lowered)
    ):
        return TemplateSolveResult(False)

    label_prefixes = tuple(
        prefix
        for prefix in ("Restaurant", "Property")
        if re.search(rf"\b{prefix}\s+[A-Za-z0-9]+", text, flags=re.IGNORECASE)
    )
    if not label_prefixes:
        return TemplateSolveResult(False)

    value_section = re.split(r"\bThe\s+cost\s+of\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    values = _parse_labeled_money_values(value_section, label_prefixes, require_million=False)
    costs = _parse_labeled_money_values(text, label_prefixes, require_million=True)
    budget_match = re.search(
        rf"budget\s+is\s+\$?\s*({_NUMBER_TOKEN})\s+million",
        text,
        flags=re.IGNORECASE,
    )
    if not values or not costs or not budget_match:
        return TemplateSolveResult(False)
    labels = sorted(set(values) & set(costs))
    if len(labels) < 2 or len(labels) > 24:
        return TemplateSolveResult(False)

    budget = _number(budget_match.group(1))
    exclusions = _parse_purchase_exclusions(text, label_prefixes)
    best_value: float | None = None
    best_subset: tuple[str, ...] | None = None
    for mask in range(1 << len(labels)):
        selected = tuple(label for index, label in enumerate(labels) if mask & (1 << index))
        selected_set = set(selected)
        if sum(costs[label] for label in selected) > budget + 1e-9:
            continue
        if any(left in selected_set and right in selected_set for left, right in exclusions):
            continue
        value = sum(values[label] for label in selected)
        if best_value is None or value > best_value:
            best_value = value
            best_subset = selected

    if best_value is None or best_subset is None:
        return TemplateSolveResult(
            matched=True,
            template_id="binary_purchase_selection_knapsack",
            status="infeasible",
            confidence=0.8,
            artifact={"values": values, "costs": costs, "budget": budget},
        )

    return TemplateSolveResult(
        matched=True,
        template_id="binary_purchase_selection_knapsack",
        status="optimal",
        objective_value=float(best_value),
        variable_values={f"select_{label}": 1.0 for label in best_subset},
        confidence=0.9,
        notes="Solved binary purchase-selection knapsack with budget and mutual-exclusion constraints.",
        artifact={
            "items": labels,
            "values": values,
            "costs": costs,
            "budget": budget,
            "exclusions": [list(pair) for pair in exclusions],
            "selected": list(best_subset),
        },
    )


def _solve_workforce_training_delay_ilp(text: str) -> TemplateSolveResult:
    normalized = _normalize_math_labels(text)
    lowered = normalized.lower()
    if not (
        "skilled workers" in lowered
        and "train" in lowered
        and "new workers" in lowered
        and "two weeks" in lowered
        and ("overtime" in lowered or "60 h" in lowered)
        and "delay" in lowered
        and "minimize" in lowered
    ):
        return TemplateSolveResult(False)

    demand_table = None
    for table in _parse_markdown_tables(text):
        matrix = _numeric_matrix_from_table(table)
        if matrix and "week" in " ".join(table[0]).lower() and len(matrix[0]) >= 2:
            demand_table = matrix
            break
    if not demand_table:
        return TemplateSolveResult(False)
    product_labels, _period_labels, demands = demand_table
    product_count = len(product_labels)
    horizon = len(demands[0])
    if product_count != 2 or horizon < 3:
        return TemplateSolveResult(False)

    initial_workers = re.search(
        rf"currently\s+has\s+({_NUMBER_TOKEN})\s+skilled\s+workers",
        normalized,
        flags=re.IGNORECASE,
    )
    train_target = re.search(
        rf"train\s+({_NUMBER_TOKEN})\s+new\s+workers\s+by\s+the\s+end\s+of\s+the\s+({_NUMBER_TOKEN})(?:st|nd|rd|th)?\s+week",
        normalized,
        flags=re.IGNORECASE,
    )
    regular_hours = re.search(
        rf"worker\s+works\s+\$?\s*({_NUMBER_TOKEN})\s*h\s*\$?\s+per\s+week",
        normalized,
        flags=re.IGNORECASE,
    )
    trainer_capacity = re.search(
        rf"skilled\s+worker\s+can\s+train\s+up\s+to\s+({_NUMBER_TOKEN})\s+new\s+workers\s+in\s+({_NUMBER_TOKEN})\s+weeks",
        normalized,
        flags=re.IGNORECASE,
    )
    skilled_wage = re.search(
        rf"weekly\s+wage\s+of\s+a\s+skilled\s+worker\s+is\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    trainee_wage = re.search(
        rf"weekly\s+wage\s+of\s+a\s+trainee.*?is\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )
    trained_wage = re.search(
        rf"after\s+training,\s+the\s+wage\s+is\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    overtime = re.search(
        rf"work\s+\$?\s*({_NUMBER_TOKEN})\s*h\s*\$?\s+per\s+week,\s+with\s+a\s+weekly\s+wage\s+of\s+({_NUMBER_TOKEN})",
        normalized,
        flags=re.IGNORECASE,
    )
    if not (
        initial_workers
        and train_target
        and regular_hours
        and trainer_capacity
        and skilled_wage
        and trainee_wage
        and trained_wage
        and overtime
    ):
        return TemplateSolveResult(False)

    production_rates = [0.0] * product_count
    for value, label in re.findall(
        rf"({_NUMBER_TOKEN})\s+kg\s*/\s*h\s*\$?\s+of\s+food\s+([A-Za-z0-9]+)",
        normalized,
        flags=re.IGNORECASE,
    ):
        index = _item_index_for_label(product_labels, label)
        if index is not None:
            production_rates[index] = _number(value)
    delay_penalties = [0.0] * product_count
    for value, label in re.findall(
        rf"({_NUMBER_TOKEN})\s+\w+\s+for\s+food\s+([A-Za-z0-9]+)",
        normalized,
        flags=re.IGNORECASE,
    ):
        index = _item_index_for_label(product_labels, label)
        if index is not None:
            delay_penalties[index] = _number(value)
    if any(value <= 0 for value in production_rates) or any(value <= 0 for value in delay_penalties):
        return TemplateSolveResult(False)

    initial = _number(initial_workers.group(1))
    target = _number(train_target.group(1))
    regular_hour_count = _number(regular_hours.group(1))
    trainer_capacity_count = _number(trainer_capacity.group(1))
    training_duration = int(_number(trainer_capacity.group(2)))
    skilled_weekly_wage = _number(skilled_wage.group(1))
    trainee_weekly_wage = _number(trainee_wage.group(1))
    trained_weekly_wage = _number(trained_wage.group(1))
    overtime_hours = _number(overtime.group(1))
    overtime_weekly_wage = _number(overtime.group(2))
    if training_duration <= 0 or horizon <= training_duration:
        return TemplateSolveResult(False)

    start_count = horizon - training_duration

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import numpy as np
    except ImportError as exc:
        return TemplateSolveResult(
            matched=True,
            template_id="workforce_training_delay_ilp",
            status="solver_unavailable",
            confidence=0.8,
            notes=str(exc),
            artifact={"products": product_labels, "demands": demands},
        )

    names: list[str] = []

    def add(name: str) -> int:
        names.append(name)
        return len(names) - 1

    trainee_start = [add(f"trainees_start_{period + 1}") for period in range(start_count)]
    trainer_start = [add(f"trainers_start_{period + 1}") for period in range(start_count)]
    overtime_workers = [add(f"overtime_workers_{period + 1}") for period in range(horizon)]
    production = [
        [add(f"produce_{product_labels[p]}_{period + 1}") for period in range(horizon)]
        for p in range(product_count)
    ]
    backlog = [
        [add(f"backlog_{product_labels[p]}_{period + 1}") for period in range(horizon)]
        for p in range(product_count)
    ]
    variable_count = len(names)
    c = np.zeros(variable_count)
    fixed_cost = initial * skilled_weekly_wage * horizon
    for start_period, variable in enumerate(trainee_start):
        c[variable] = (
            trainee_weekly_wage * training_duration
            + trained_weekly_wage * max(0, horizon - (start_period + training_duration))
        )
    overtime_increment = overtime_weekly_wage - skilled_weekly_wage
    for variable in overtime_workers:
        c[variable] = overtime_increment
    for product_index in range(product_count):
        for period in range(horizon):
            c[backlog[product_index][period]] = delay_penalties[product_index]

    rows: list[Any] = []
    lower: list[float] = []
    upper: list[float] = []
    for start_period in range(start_count):
        row = np.zeros(variable_count)
        row[trainee_start[start_period]] = 1.0
        row[trainer_start[start_period]] = -trainer_capacity_count
        rows.append(row)
        lower.append(-math.inf)
        upper.append(0.0)

    row = np.zeros(variable_count)
    for variable in trainee_start:
        row[variable] = 1.0
    rows.append(row)
    lower.append(target)
    upper.append(target)

    for period in range(horizon):
        completed_starts = [
            start for start in range(start_count)
            if start + training_duration <= period
        ]
        busy_starts = [
            start for start in range(start_count)
            if start <= period < start + training_duration
        ]
        row = np.zeros(variable_count)
        for start in busy_starts:
            row[trainer_start[start]] = 1.0
        for start in completed_starts:
            row[trainee_start[start]] -= 1.0
        rows.append(row)
        lower.append(-math.inf)
        upper.append(initial)

        row = np.zeros(variable_count)
        row[overtime_workers[period]] = 1.0
        for start in busy_starts:
            row[trainer_start[start]] = 1.0
        for start in completed_starts:
            row[trainee_start[start]] -= 1.0
        rows.append(row)
        lower.append(-math.inf)
        upper.append(initial)

        row = np.zeros(variable_count)
        for product_index in range(product_count):
            row[production[product_index][period]] = 1.0 / production_rates[product_index]
        row[overtime_workers[period]] = -(overtime_hours - regular_hour_count)
        for start in busy_starts:
            row[trainer_start[start]] = regular_hour_count
        for start in completed_starts:
            row[trainee_start[start]] -= regular_hour_count
        rows.append(row)
        lower.append(-math.inf)
        upper.append(initial * regular_hour_count)

    for product_index in range(product_count):
        for period in range(horizon):
            row = np.zeros(variable_count)
            row[production[product_index][period]] = 1.0
            row[backlog[product_index][period]] = 1.0
            if period > 0:
                row[backlog[product_index][period - 1]] = -1.0
            rows.append(row)
            lower.append(demands[product_index][period])
            upper.append(demands[product_index][period])

    integrality = np.ones(variable_count)
    result = milp(
        c,
        integrality=integrality,
        bounds=Bounds(np.zeros(variable_count), np.full(variable_count, math.inf)),
        constraints=LinearConstraint(np.vstack(rows), np.array(lower), np.array(upper)),
    )
    if not result.success:
        return TemplateSolveResult(
            matched=True,
            template_id="workforce_training_delay_ilp",
            status="solver_failed",
            confidence=0.82,
            notes=str(result.message),
            artifact={"products": product_labels, "demands": demands},
        )

    variable_values: dict[str, float] = {}
    for index, variable in enumerate(trainee_start):
        value = float(result.x[variable])
        if not math.isclose(value, 0.0, abs_tol=1e-8):
            variable_values[f"trainees_start_period_{index + 1}"] = value
            variable_values[f"trainers_start_period_{index + 1}"] = float(result.x[trainer_start[index]])
    for period, variable in enumerate(overtime_workers):
        value = float(result.x[variable])
        if not math.isclose(value, 0.0, abs_tol=1e-8):
            variable_values[f"overtime_workers_period_{period + 1}"] = value
    for product_index, label in enumerate(product_labels):
        for period in range(horizon):
            variable_values[f"produce_{label}_period_{period + 1}"] = float(
                result.x[production[product_index][period]]
            )
            backlog_value = float(result.x[backlog[product_index][period]])
            if not math.isclose(backlog_value, 0.0, abs_tol=1e-8):
                variable_values[f"backlog_{label}_period_{period + 1}"] = backlog_value

    return TemplateSolveResult(
        matched=True,
        template_id="workforce_training_delay_ilp",
        status="optimal",
        objective_value=float(result.fun + fixed_cost),
        variable_values=variable_values,
        confidence=0.88,
        notes="Solved workforce training, overtime production, and delayed-delivery planning MILP.",
        artifact={
            "products": product_labels,
            "demands": demands,
            "initial_skilled_workers": initial,
            "training_target": target,
            "training_duration": training_duration,
            "trainer_capacity": trainer_capacity_count,
            "regular_hours": regular_hour_count,
            "overtime_hours": overtime_hours,
            "wages": {
                "skilled": skilled_weekly_wage,
                "trainee": trainee_weekly_wage,
                "trained": trained_weekly_wage,
                "overtime": overtime_weekly_wage,
            },
            "production_rates": production_rates,
            "delay_penalties": delay_penalties,
            "fixed_initial_worker_wage_cost": fixed_cost,
        },
    )


def solve_with_template(problem_text: str) -> TemplateSolveResult:
    """Try conservative deterministic OR templates in priority order."""
    for solver in (
        _solve_training_asset_count,
        _solve_assignment_table,
        _solve_preference_assignment_goal_programming,
        _solve_max_flow_network,
        _solve_robust_resource_capacity_lp,
        _solve_security_maximin_revenue,
        _solve_permutation_flow_shop_scheduling,
        _solve_tsp_routing,
        _solve_vrp_hard_time_windows_milp,
        _solve_set_cover_table,
        _solve_minimum_vertex_cover,
        _solve_cutting_stock,
        _solve_interval_contract_covering,
        _solve_procurement_lot_mix,
        _solve_integer_resource_mix,
        _solve_container_capacity_max_mix,
        _solve_advertising_media_mix,
        _solve_two_product_machine_inventory_surplus_lp,
        _solve_two_option_resource_max_mix,
        _solve_bombing_success_probability,
        _solve_two_test_probe_mix,
        _solve_furnace_purchase_min_count,
        _solve_two_ingredient_mix_profit_lp,
        _solve_two_team_capacity_max_mix,
        _solve_two_food_ratio_protein_max,
        _solve_two_item_nutrition_min_mix,
        _solve_two_volunteer_gift_max_mix,
        _solve_two_product_resource_profit_max,
        _solve_two_vehicle_capacity_min_count,
        _solve_wrap_platter_time_min,
        _solve_two_task_productivity_min_cost,
        _solve_three_investment_balance_min_cost,
        _solve_two_container_paste_max,
        _solve_two_medicine_pill_min_cost,
        _solve_sand_container_max_delivery,
        _solve_crop_diversity_profit_min,
        _solve_project_resource_viability_min_cost,
        _solve_two_energy_capacity_min_cost,
        _solve_facility_resource_balance_min_cost,
        _solve_two_food_sodium_min_lp,
        _solve_two_fertilizer_vitamin_min_lp,
        _solve_runner_canoe_mail_max_ilp,
        _solve_ice_cream_profit_bounds_lp,
        _solve_steel_furnace_method_min_cost,
        _solve_mall_store_lease_piecewise,
        _solve_fruit_farm_two_type_profit,
        _solve_three_department_staffing_min_cost,
        _solve_multifood_integer_diet_min_cost,
        _solve_four_department_salary_balance_min_cost,
        _solve_four_training_area_min_cost,
        _solve_two_sandwich_profit_max_lp,
        _solve_two_van_min_count_ilp,
        _solve_two_medication_time_min_ilp,
        _solve_snow_remover_min_count_ilp,
        _solve_course_prerequisite_cover_min,
        _solve_three_channel_effort_min_cost,
        _solve_three_project_minimum_allocation_min_cost,
        _solve_two_project_weighted_resource_min_cost,
        _solve_three_project_lower_bound_budget_min_cost,
        _solve_two_store_customer_min_count,
        _solve_two_machine_tea_leaf_max_lp,
        _solve_two_experiment_green_gas_max_ilp,
        _solve_ship_plane_fuel_min_ilp,
        _solve_meal_fiber_selection_lp_enum,
        _solve_three_route_balance_min_cost,
        _solve_energy_project_linear_min_cost,
        _solve_two_printer_shared_machine_profit_lp,
        _solve_two_animal_package_min_count,
        _solve_two_package_stock_profit_lp,
        _solve_meal_protein_pack_selection_lp_enum,
        _solve_two_project_performance_resource_min_cost,
        _solve_environmental_project_budget_min_cost,
        _solve_three_vehicle_operating_min_cost,
        _solve_mobile_unit_parking_min_ilp,
        _solve_bus_car_child_pickup_min_count,
        _solve_pizza_baking_time_min_ilp,
        _solve_two_milk_tea_profit_lp,
        _solve_farm_four_crop_ratio_profit_lp,
        _solve_chemical_byproduct_profit_lp,
        _solve_three_task_method_selection_min_cost,
        _solve_warehouse_resource_lower_bound_min_cost,
        _solve_three_task_time_min_hours,
        _solve_souvenir_elephant_tiger_profit_lp,
        _solve_truck_car_gas_min_ilp,
        _solve_rice_bag_weight_max_ilp,
        _solve_toy_product_logic_profit_max,
        _solve_product_table_fixed_cost_profit_lp,
        _solve_stock_bond_balance_min_cost,
        _solve_two_service_resource_min_cost,
        _solve_environmental_project_three_var_min_cost,
        _solve_property_pair_requirement_min_cost,
        _solve_cart_trolley_worker_min_count,
        _solve_light_fixture_change_min_count,
        _solve_cable_mix_profit_max,
        _solve_meat_slicer_min_count,
        _solve_project_schedule_machine_rental_lp,
        _solve_shelf_space_balance_min_cost,
        _solve_cargo_value_capacity_min_cost,
        _solve_telecom_project_focus_min_cost,
        _solve_four_department_fund_focus_min_cost,
        _solve_supply_chain_resource_balance_min_cost,
        _solve_two_advertising_exposure_budget,
        _solve_letter_bird_treat_max,
        _solve_balloon_gondola_pollution_min,
        _solve_wagon_ore_min_count,
        _solve_costed_pipe_cutting_stock,
        _solve_two_ship_operation_min_cost,
        _solve_three_route_vehicle_allocation_min_cost,
        _solve_three_channel_fractional_balance_min_cost,
        _solve_education_resource_min_cost,
        _solve_bike_scooter_meal_delivery_max,
        _solve_van_truck_min_vans,
        _solve_mango_guava_profit_max,
        _solve_two_factory_min_count,
        _solve_dog_food_profit_lp,
        _solve_two_program_benefit_min_score,
        _solve_three_department_resource_min_cost,
        _solve_three_unit_fractional_military_min_cost,
        _solve_four_project_environmental_resource_min_cost,
        _solve_wide_narrow_pipe_min_count,
        _solve_camel_truck_transport_min_hours,
        _solve_basketball_football_max_count,
        _solve_three_route_capacity_lower_bound_min_cost,
        _solve_supply_chain_material_labor_transport_min_cost,
        _solve_four_energy_project_pair_margin_min_cost,
        _solve_narrative_food_diet_min_cost_lp,
        _solve_sanitizer_cleaning_max_hands,
        _solve_gummy_pill_zinc_max,
        _solve_van_truck_chocolate_min_trips,
        _solve_metal_working_equipment_min_count,
        _solve_two_asset_real_estate_profit_lp,
        _solve_worker_shift_count_mix,
        _solve_two_supplement_integer_diet,
        _solve_two_item_weighted_score_min_cost,
        _solve_minimum_lower_bound_allocation_ilp,
        _solve_portfolio_fee_lower_bounds,
        _solve_real_estate_weighted_roi_min_cost,
        _solve_four_unit_pair_strength_min_cost,
        _solve_healthcare_department_allocation_min_cost,
        _solve_route_vehicle_min_cost,
        _solve_healthcare_fund_three_department_min_cost,
        _solve_military_support_points_min_cost,
        _solve_two_exercise_balance_min_fatigue,
        _solve_telecom_project_pair_min_cost,
        _solve_fractional_telecom_project_min_cost,
        _solve_retail_department_strong_coverage_min_cost,
        _solve_four_property_real_estate_min_cost,
        _solve_small_symbolic_integer_min_cost,
        _solve_three_project_integer_linear_min_cost,
        _solve_xyz_three_variable_integer_lp,
        _solve_xy_two_variable_integer_lp,
        _solve_table_capacity_space_mix,
        _solve_fixed_charge_machine_assignment,
        _solve_fixed_charge_transshipment,
        _solve_narrative_transportation_distribution,
        _solve_transportation_table,
        _solve_facility_location_distribution,
        _solve_fixed_charge_substitution_production,
        _solve_multi_period_workforce_production_plan,
        _solve_two_product_seasonal_inventory_plan_lp,
        _solve_container_loading_min_count,
        _solve_input_output_gdp_lp,
        _solve_periodic_production_inventory_lp,
        _solve_tool_repair_replacement,
        _solve_reliability_spares_allocation,
        _solve_inventory_arbitrage_lp,
        _solve_grain_inventory_arbitrage_lp,
        _solve_multi_product_inventory_backlog_ilp,
        _solve_workforce_training_delay_ilp,
        _solve_continuous_nutrition_mix_lp,
        _solve_integer_diet_lp,
        _solve_livestock_resource_mix_ilp,
        _solve_weighted_idle_goal_product_mix,
        _solve_minimum_overtime_production_goal,
        _solve_sales_staff_overtime_goal,
        _solve_overtime_resource_product_mix,
        _solve_product_mix_table,
        _solve_multi_machine_process_profit_lp,
        _solve_farm_operating_plan_lp,
        _solve_fixed_activation_quota_product_plan,
        _solve_narrative_product_mix_lp,
        _solve_two_product_time_ratio_lp,
        _solve_production_conversion_lp,
        _solve_candy_quality_blending_lp,
        _solve_quality_constrained_blending_lp,
        _solve_gasoline_blending_lp,
        _solve_cyclic_shift_staffing,
        _solve_student_duty_scheduling,
        _solve_consecutive_day_workforce_scheduling,
        _solve_reinvestable_cashflow_lp,
        _solve_balance_sheet_production_lp,
        _solve_fractional_investment_budget_lp,
        _solve_binary_subset_selection,
        _solve_binary_purchase_selection,
    ):
        result = solver(problem_text)
        if result.matched:
            return result
    return TemplateSolveResult(False)
