"""Offline evaluation helpers for domain-ready ORBIT.

These helpers avoid provider calls and can be used by tests, scripts, or small
gold sets. They are intentionally simple and deterministic.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

from .problem_profile import profile_or_problem_text

_CITATION_RE = re.compile(r"\[(D|F|C|W|L|A|E)(\d+)\]")


def _norm_token(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


_NO_FINITE_OPTIMUM_STATUSES = {
    "no_solution_reported",
    "no best solution",
    "no solution",
    "no optimal solution",
    "solver_infeasible",
    "infeasible",
    "solver_unbounded",
    "unbounded",
}


def _solver_statuses_match(predicted_status: str, gold_status: str) -> bool:
    if predicted_status == gold_status:
        return True
    if gold_status in _NO_FINITE_OPTIMUM_STATUSES:
        return predicted_status in _NO_FINITE_OPTIMUM_STATUSES
    if predicted_status in _NO_FINITE_OPTIMUM_STATUSES:
        return gold_status in _NO_FINITE_OPTIMUM_STATUSES
    return False


def _component_key(item: dict[str, Any]) -> tuple[str, str]:
    component_type = _norm_token(item.get("component_type"))
    symbol = _norm_token(item.get("symbol"))
    natural_text = _norm_token(item.get("natural_text"))
    return component_type, symbol or natural_text


def _chunk_boundary_key(item: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(item.get("position_start") or 0),
        int(item.get("position_end") or 0),
        _norm_token(item.get("section_path")),
    )


def _ledger_key(item: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        _norm_token(item.get("entity") or item.get("entity_name")),
        _norm_token(item.get("attribute") or item.get("attribute_name")),
        _norm_token(item.get("value")),
        _norm_token(item.get("timeframe") or item.get("normalized_timeframe")),
    )


def _unordered_pair_key(item: object) -> tuple[str, str]:
    if isinstance(item, dict):
        left = item.get("source_id") or item.get("left") or item.get("a")
        right = item.get("target_id") or item.get("right") or item.get("b")
    elif isinstance(item, (list, tuple)) and len(item) >= 2:
        left, right = item[0], item[1]
    else:
        left, right = "", ""
    ordered = sorted([_norm_token(left), _norm_token(right)])
    return ordered[0], ordered[1]


def _extract_citations(text: str) -> list[str]:
    return [f"{prefix}{number}" for prefix, number in _CITATION_RE.findall(text or "")]


def _markdown_table_cells(markdown: str) -> list[str]:
    cells: list[str] = []
    for raw_line in (markdown or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or "|" not in line[1:]:
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if parts and all(re.fullmatch(r":?-{3,}:?", part or "") for part in parts):
            continue
        cells.extend(_norm_token(part) for part in parts if _norm_token(part))
    return cells


def precision_recall_f1(predicted: Iterable[object], gold: Iterable[object]) -> dict:
    pred_set = set(predicted)
    gold_set = set(gold)
    true_positive = len(pred_set & gold_set)
    precision = true_positive / len(pred_set) if pred_set else 0.0
    recall = true_positive / len(gold_set) if gold_set else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "predicted_count": len(pred_set),
        "gold_count": len(gold_set),
    }


def evaluate_chunk_boundary_quality(
    predicted_chunks: list[dict[str, Any]],
    gold_chunks: list[dict[str, Any]],
) -> dict:
    """Score chunk boundaries by start/end offsets plus section path."""
    predicted_keys = [_chunk_boundary_key(item) for item in predicted_chunks]
    gold_keys = [_chunk_boundary_key(item) for item in gold_chunks]
    result = precision_recall_f1(predicted_keys, gold_keys)
    length_errors = []
    for predicted, gold in zip(predicted_chunks, gold_chunks):
        pred_len = int(predicted.get("position_end") or 0) - int(
            predicted.get("position_start") or 0
        )
        gold_len = int(gold.get("position_end") or 0) - int(
            gold.get("position_start") or 0
        )
        length_errors.append(abs(pred_len - gold_len))
    result["mean_length_error"] = (
        sum(length_errors) / len(length_errors) if length_errors else 0.0
    )
    return result


def evaluate_table_fidelity(
    predicted_markdown: str,
    gold_markdown: str,
) -> dict:
    """Score table extraction by normalized cell overlap."""
    predicted_cells = _markdown_table_cells(predicted_markdown)
    gold_cells = _markdown_table_cells(gold_markdown)
    result = precision_recall_f1(predicted_cells, gold_cells)
    result["exact_match"] = predicted_cells == gold_cells
    return result


def evaluate_component_extraction(
    predicted_components: list[dict[str, Any]],
    gold_components: list[dict[str, Any]],
) -> dict:
    """Score OR/MSE component extraction by type plus symbol/text key."""
    predicted_keys = [_component_key(item) for item in predicted_components]
    gold_keys = [_component_key(item) for item in gold_components]
    overall = precision_recall_f1(predicted_keys, gold_keys)

    by_type: dict[str, dict] = {}
    component_types = sorted({key[0] for key in predicted_keys + gold_keys if key[0]})
    for component_type in component_types:
        pred = [key for key in predicted_keys if key[0] == component_type]
        gold = [key for key in gold_keys if key[0] == component_type]
        by_type[component_type] = precision_recall_f1(pred, gold)
    return {"overall": overall, "by_type": by_type}


def evaluate_retrieval(
    ranked_ids: list[int],
    relevant_ids: Iterable[int],
    *,
    k: int = 10,
) -> dict:
    relevant = {int(item) for item in relevant_ids}
    ranked = [int(item) for item in ranked_ids[:k]]
    hits = [doc_id for doc_id in ranked if doc_id in relevant]
    recall = len(set(hits)) / len(relevant) if relevant else 0.0
    mrr = 0.0
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            mrr = 1.0 / rank
            break
    dcg = 0.0
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
    return {
        f"recall@{k}": recall,
        f"mrr@{k}": mrr,
        f"ndcg@{k}": ndcg,
        "hit_count": len(set(hits)),
        "relevant_count": len(relevant),
    }


def evaluate_reranker_lift(
    baseline_ranked_ids: list[int],
    reranked_ids: list[int],
    relevant_ids: Iterable[int],
    *,
    k: int = 10,
) -> dict:
    """Compare retrieval quality before and after reranking."""
    baseline = evaluate_retrieval(baseline_ranked_ids, relevant_ids, k=k)
    reranked = evaluate_retrieval(reranked_ids, relevant_ids, k=k)
    return {
        "baseline": baseline,
        "reranked": reranked,
        f"recall_lift@{k}": reranked[f"recall@{k}"] - baseline[f"recall@{k}"],
        f"mrr_lift@{k}": reranked[f"mrr@{k}"] - baseline[f"mrr@{k}"],
        f"ndcg_lift@{k}": reranked[f"ndcg@{k}"] - baseline[f"ndcg@{k}"],
    }


def evaluate_citation_accuracy(
    answer_text: str,
    allowed_citations: Iterable[str],
    *,
    required_citations: Iterable[str] | None = None,
) -> dict:
    """Score whether an answer cites only injected IDs and covers required IDs."""
    cited = list(dict.fromkeys(_extract_citations(answer_text)))
    allowed = {str(item).strip() for item in allowed_citations}
    required = {str(item).strip() for item in (required_citations or [])}
    valid = [item for item in cited if item in allowed]
    hallucinated = [item for item in cited if item not in allowed]
    missing_required = sorted(required - set(cited))
    precision = len(valid) / len(cited) if cited else 1.0
    recall = len(required & set(cited)) / len(required) if required else None
    return {
        "cited": cited,
        "valid": valid,
        "hallucinated": hallucinated,
        "missing_required": missing_required,
        "citation_precision": precision,
        "required_recall": recall,
        "passed": not hallucinated and not missing_required,
    }


def evaluate_answer_faithfulness(
    answer_text: str,
    evidence_texts: dict[str, str],
    *,
    min_token_overlap: float = 0.2,
) -> dict:
    """Provider-free heuristic: cited sentences should overlap cited evidence."""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?。！？])\s+", answer_text or "")
        if sentence.strip()
    ]
    checked = 0
    supported = 0
    unsupported: list[dict[str, Any]] = []
    for sentence in sentences:
        citations = _extract_citations(sentence)
        if not citations:
            continue
        checked += 1
        sentence_tokens = set(re.findall(r"[a-z0-9_]+", sentence.lower()))
        best_overlap = 0.0
        for citation in citations:
            evidence_tokens = set(
                re.findall(r"[a-z0-9_]+", (evidence_texts.get(citation) or "").lower())
            )
            if not sentence_tokens or not evidence_tokens:
                continue
            overlap = len(sentence_tokens & evidence_tokens) / len(sentence_tokens)
            best_overlap = max(best_overlap, overlap)
        if best_overlap >= min_token_overlap:
            supported += 1
        else:
            unsupported.append(
                {
                    "sentence": sentence,
                    "citations": citations,
                    "best_token_overlap": best_overlap,
                }
            )
    return {
        "checked_sentence_count": checked,
        "supported_sentence_count": supported,
        "faithfulness": supported / checked if checked else None,
        "unsupported": unsupported,
        "passed": not unsupported,
    }


def evaluate_no_answer_calibration(predictions: list[dict[str, Any]]) -> dict:
    """Score answer/abstain decisions against whether a question was answerable."""
    total = len(predictions)
    correct = false_answer = false_abstain = 0
    for item in predictions:
        should_answer = bool(item.get("should_answer"))
        abstained = bool(item.get("no_answer") or item.get("abstained"))
        if should_answer and abstained:
            false_abstain += 1
        elif not should_answer and not abstained:
            false_answer += 1
        else:
            correct += 1
    return {
        "count": total,
        "calibration_accuracy": correct / total if total else 0.0,
        "false_answer_rate": false_answer / total if total else 0.0,
        "false_abstain_rate": false_abstain / total if total else 0.0,
    }


def evaluate_contradiction_detection(
    predicted_pairs: Iterable[object],
    gold_pairs: Iterable[object],
) -> dict:
    """Score unordered contradiction/conflict pair detection."""
    return precision_recall_f1(
        [_unordered_pair_key(item) for item in predicted_pairs],
        [_unordered_pair_key(item) for item in gold_pairs],
    )


def summarize_claim_review_outcomes(review_rows: list[dict[str, Any]]) -> dict:
    """Summarize claim acceptance/rejection reasons from review rows."""
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for row in review_rows:
        status = _norm_token(
            row.get("review_status") or row.get("decision") or row.get("status")
        )
        status_counts[status] = status_counts.get(status, 0) + 1
        note = str(row.get("review_note") or row.get("reason") or "").strip()
        for reason in [part.strip() for part in re.split(r";|,", note) if part.strip()]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    total = len(review_rows)
    accepted = status_counts.get("accepted", 0) + status_counts.get("accept", 0)
    rejected = status_counts.get("rejected", 0) + status_counts.get("reject", 0)
    return {
        "count": total,
        "status_counts": status_counts,
        "reason_counts": reason_counts,
        "acceptance_rate": accepted / total if total else 0.0,
        "rejection_rate": rejected / total if total else 0.0,
    }


def evaluate_ledger_extraction(
    predicted_entries: list[dict[str, Any]],
    gold_entries: list[dict[str, Any]],
) -> dict:
    """Score structured ledger extraction by entity, attribute, value, timeframe."""
    return precision_recall_f1(
        [_ledger_key(item) for item in predicted_entries],
        [_ledger_key(item) for item in gold_entries],
    )


def summarize_stage_latency(events: list[dict[str, Any]]) -> dict:
    """Summarize per-stage latency from local event records."""
    by_stage: dict[str, list[float]] = {}
    for event in events:
        stage = str(event.get("stage") or event.get("name") or "unknown")
        duration = event.get("elapsed_time_s", event.get("duration_s"))
        if duration is None:
            continue
        by_stage.setdefault(stage, []).append(float(duration))
    summary: dict[str, dict[str, float]] = {}
    for stage, values in by_stage.items():
        ordered = sorted(values)
        p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
        summary[stage] = {
            "count": len(values),
            "mean_s": sum(values) / len(values),
            "p95_s": ordered[p95_index],
            "max_s": max(values),
        }
    return summary


def evaluate_solver_result(
    *,
    predicted_status: str,
    gold_status: str,
    predicted_objective: float | None = None,
    gold_objective: float | None = None,
    abs_tol: float = 1e-4,
    rel_tol: float = 0.05,
) -> dict:
    pred_status = _norm_token(predicted_status)
    expected_status = _norm_token(gold_status)
    status_match = _solver_statuses_match(pred_status, expected_status)

    objective_match = None
    objective_error = None
    if gold_objective is not None:
        if predicted_objective is None:
            objective_match = False
        else:
            objective_error = abs(float(predicted_objective) - float(gold_objective))
            tolerance = max(abs_tol, abs(float(gold_objective)) * rel_tol)
            objective_match = objective_error <= tolerance

    return {
        "status_match": status_match,
        "objective_match": objective_match,
        "objective_error": objective_error,
        "correct": status_match and (objective_match is not False),
    }


def load_or_mse_gold_set(path: str | Path) -> list[dict[str, Any]]:
    """Load a small local OR/MSE gold-set JSON file.

    Expected shape:
    {"cases": [{"id": "...", "task_type": "component_extraction|solver|retrieval", ...}]}
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(cases, list):
        raise ValueError("Gold set must be a list or {'cases': [...]} object")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"Gold case {index} must be an object")
        case_id = str(case.get("id") or "").strip()
        task_type = str(case.get("task_type") or "").strip()
        if not case_id:
            raise ValueError(f"Gold case {index} is missing id")
        if case_id in seen:
            raise ValueError(f"Duplicate gold case id: {case_id}")
        if task_type not in {"component_extraction", "solver", "retrieval"}:
            raise ValueError(f"Unsupported task_type for {case_id}: {task_type}")
        seen.add(case_id)
        normalized.append(case)
    return normalized


def evaluate_gold_case_prediction(
    case: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one prediction against a loaded OR/MSE gold case."""
    task_type = case.get("task_type")
    if task_type == "component_extraction":
        return evaluate_component_extraction(
            prediction.get("components") or [],
            case.get("gold_components") or [],
        )
    if task_type == "solver":
        gold = case.get("gold_solver") or {}
        return evaluate_solver_result(
            predicted_status=str(prediction.get("status") or ""),
            predicted_objective=prediction.get("objective_value"),
            gold_status=str(gold.get("status") or ""),
            gold_objective=gold.get("objective_value"),
            abs_tol=float(gold.get("abs_tol", 1e-4)),
            rel_tol=float(gold.get("rel_tol", 0.05)),
        )
    if task_type == "retrieval":
        return evaluate_retrieval(
            [int(item) for item in prediction.get("ranked_ids") or []],
            [int(item) for item in case.get("relevant_ids") or []],
            k=int(case.get("k", 10)),
        )
    raise ValueError(f"Unsupported task_type: {task_type}")


def evaluate_gold_set_predictions(
    cases: list[dict[str, Any]],
    predictions_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate predictions for a mixed OR/MSE gold set."""
    per_case: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for case in cases:
        case_id = str(case["id"])
        prediction = predictions_by_id.get(case_id)
        if prediction is None:
            missing.append(case_id)
            continue
        per_case[case_id] = evaluate_gold_case_prediction(case, prediction)

    correct_values = [
        value["correct"]
        for value in per_case.values()
        if isinstance(value.get("correct"), bool)
    ]
    f1_values = [
        value["overall"]["f1"]
        for value in per_case.values()
        if isinstance(value.get("overall"), dict) and "f1" in value["overall"]
    ]
    retrieval_recalls = [
        metric
        for value in per_case.values()
        for key, metric in value.items()
        if key.startswith("recall@")
    ]
    return {
        "case_count": len(cases),
        "evaluated_count": len(per_case),
        "missing_ids": missing,
        "solver_accuracy": (
            sum(1 for item in correct_values if item) / len(correct_values)
            if correct_values
            else None
        ),
        "mean_component_f1": (
            sum(float(item) for item in f1_values) / len(f1_values)
            if f1_values
            else None
        ),
        "mean_retrieval_recall": (
            sum(float(item) for item in retrieval_recalls) / len(retrieval_recalls)
            if retrieval_recalls
            else None
        ),
        "per_case": per_case,
    }


def _load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"{path} must contain JSON objects")
        return rows
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict) and isinstance(data.get("data"), list):
        rows = data["data"]
    elif isinstance(data, dict):
        rows = [data]
    else:
        raise ValueError(f"{path} must be a JSON list or JSONL file")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain JSON objects")
    return rows


def _gold_solver_from_answer(answer: object) -> dict[str, Any]:
    text = str(answer or "").strip()
    if not text:
        return {"status": "unknown", "objective_value": None}
    lowered = _norm_token(text)
    if "no best solution" in lowered or "no solution" in lowered:
        return {"status": "no_solution_reported", "objective_value": None}
    try:
        objective = float(text)
        if math.isclose(objective, 1_000_000_000_000.0, rel_tol=0.0, abs_tol=1e-6):
            return {"status": "no_solution_reported", "objective_value": None}
        return {"status": "optimal", "objective_value": objective}
    except ValueError:
        return {"status": lowered or "unknown", "objective_value": None}


def load_orq_dataset(root: str | Path = "docs/ORQ_Dataset") -> list[dict[str, Any]]:
    """Load local ORQ_Dataset files into normalized solver-evaluation cases."""
    root_path = Path(root)
    specs = [
        ("IndustryOR", "test", root_path / "IndustryOR" / "IndustryOR.json"),
        ("MAMO", "easy_lp", root_path / "MAMO" / "MAMO_EasyLP.json"),
        ("MAMO", "complex_lp", root_path / "MAMO" / "MAMO_ComplexLP.json"),
        (
            "NL4OPT",
            "test",
            root_path / "NL4OPT" / "NL4OPT_with_optimal_solution.json",
        ),
    ]
    cases: list[dict[str, Any]] = []
    for dataset, split, path in specs:
        if not path.exists():
            continue
        rows = _load_json_or_jsonl(path)
        for index, row in enumerate(rows, start=1):
            raw_id = row.get("id")
            local_id = str(raw_id if raw_id not in (None, "") else index)
            case_id = f"{dataset}:{split}:{local_id}"
            problem_text = row.get("en_question") or row.get("question") or ""
            gold_solver = _gold_solver_from_answer(row.get("en_answer"))
            profile = profile_or_problem_text(
                problem_text=problem_text,
                gold_solver=gold_solver,
            )
            cases.append(
                {
                    "id": case_id,
                    "dataset": dataset,
                    "split": split,
                    "source_style": dataset,
                    "task_type": "solver",
                    "problem_text": problem_text,
                    "difficulty": row.get("difficulty") or "",
                    **profile,
                    "gold_solver": {
                        **gold_solver,
                        "abs_tol": 1e-4,
                        "rel_tol": 0.05,
                    },
                    "raw": row,
                }
            )
    return cases


def summarize_orq_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize normalized ORQ cases without requiring predictions."""
    by_dataset: dict[str, int] = {}
    by_split: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    by_status: dict[str, int] = {}
    by_workflow: dict[str, int] = {}
    by_coverage: dict[str, int] = {}
    by_realism: dict[str, int] = {}
    table_case_count = 0
    for case in cases:
        dataset = str(case.get("dataset") or "unknown")
        split = str(case.get("split") or "unknown")
        difficulty = str(case.get("difficulty") or "unspecified")
        status = str((case.get("gold_solver") or {}).get("status") or "unknown")
        workflow = str(case.get("workflow_hint") or "unknown")
        coverage = str(case.get("orq_coverage") or "unknown")
        realism = str(case.get("industrial_realism") or "unknown")
        by_dataset[dataset] = by_dataset.get(dataset, 0) + 1
        by_split[f"{dataset}:{split}"] = by_split.get(f"{dataset}:{split}", 0) + 1
        by_difficulty[difficulty] = by_difficulty.get(difficulty, 0) + 1
        by_status[status] = by_status.get(status, 0) + 1
        by_workflow[workflow] = by_workflow.get(workflow, 0) + 1
        by_coverage[coverage] = by_coverage.get(coverage, 0) + 1
        by_realism[realism] = by_realism.get(realism, 0) + 1
        table_case_count += int(bool(case.get("has_table")))
    return {
        "case_count": len(cases),
        "datasets": by_dataset,
        "splits": by_split,
        "difficulties": by_difficulty,
        "gold_statuses": by_status,
        "workflow_hints": by_workflow,
        "orq_coverage": by_coverage,
        "industrial_realism": by_realism,
        "table_case_count": table_case_count,
        "sample_case_ids": [case["id"] for case in cases[:20]],
    }


def _candidate_list(prediction: Any) -> list[dict[str, Any]]:
    if isinstance(prediction, list):
        return [item for item in prediction if isinstance(item, dict)]
    if isinstance(prediction, dict):
        candidates = prediction.get("candidates")
        if isinstance(candidates, list):
            return [item for item in candidates if isinstance(item, dict)]
        return [prediction]
    return []


def evaluate_orq_predictions(
    cases: list[dict[str, Any]],
    predictions_by_id: dict[str, Any],
    *,
    k: int = 8,
) -> dict[str, Any]:
    """Evaluate ORQ predictions with pass@1 and pass@k solver metrics.

    Prediction values may be a single object with `status` and `objective_value`,
    or a list / `{candidates: [...]}` of candidate solver outputs.
    """
    per_case: dict[str, dict[str, Any]] = {}
    missing_ids: list[str] = []
    pass1_values: list[bool] = []
    passk_values: list[bool] = []
    by_dataset: dict[str, dict[str, Any]] = {}

    for case in cases:
        case_id = str(case["id"])
        candidates = _candidate_list(predictions_by_id.get(case_id))
        if not candidates:
            missing_ids.append(case_id)
            continue
        gold = case.get("gold_solver") or {}
        evaluated = [
            evaluate_solver_result(
                predicted_status=str(candidate.get("status") or ""),
                predicted_objective=candidate.get("objective_value"),
                gold_status=str(gold.get("status") or ""),
                gold_objective=gold.get("objective_value"),
                abs_tol=float(gold.get("abs_tol", 1e-4)),
                rel_tol=float(gold.get("rel_tol", 0.05)),
            )
            for candidate in candidates[: max(1, k)]
        ]
        pass1 = bool(evaluated and evaluated[0]["correct"])
        passk = any(bool(item["correct"]) for item in evaluated)
        pass1_values.append(pass1)
        passk_values.append(passk)
        dataset = str(case.get("dataset") or "unknown")
        bucket = by_dataset.setdefault(dataset, {"count": 0, "pass1": 0, "passk": 0})
        bucket["count"] += 1
        bucket["pass1"] += int(pass1)
        bucket["passk"] += int(passk)
        per_case[case_id] = {
            "dataset": dataset,
            "split": case.get("split"),
            "difficulty": case.get("difficulty"),
            "candidate_count": len(candidates),
            "pass@1": pass1,
            f"pass@{k}": passk,
            "first": evaluated[0],
            "best_rank": next(
                (rank for rank, item in enumerate(evaluated, start=1) if item["correct"]),
                None,
            ),
        }

    dataset_summary = {
        dataset: {
            "evaluated_count": values["count"],
            "pass@1": values["pass1"] / values["count"] if values["count"] else 0.0,
            f"pass@{k}": values["passk"] / values["count"] if values["count"] else 0.0,
        }
        for dataset, values in by_dataset.items()
    }
    return {
        "case_count": len(cases),
        "evaluated_count": len(per_case),
        "missing_ids": missing_ids,
        "pass@1": (
            sum(1 for item in pass1_values if item) / len(pass1_values)
            if pass1_values
            else None
        ),
        f"pass@{k}": (
            sum(1 for item in passk_values if item) / len(passk_values)
            if passk_values
            else None
        ),
        "by_dataset": dataset_summary,
        "per_case": per_case,
    }
