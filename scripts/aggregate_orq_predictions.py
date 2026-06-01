#!/usr/bin/env python3
"""Aggregate ORQ prediction files into comparison tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

from orbit_or.evaluation import evaluate_orq_predictions, load_orq_dataset


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_predictions(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    if isinstance(data, dict) and isinstance(data.get("predictions"), dict):
        return data["predictions"]
    if isinstance(data, dict):
        return data
    raise ValueError(f"{path} must be a prediction object")


def _parse_method(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--method must use name=path format")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("method name must not be empty")
    path = Path(raw_path.strip())
    if path.is_dir():
        path = path / "predictions.json"
    return name, path


def _filter_cases(
    cases: list[dict[str, Any]],
    *,
    dataset: str | None,
    split: str | None,
    case_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    if case_ids:
        wanted = set(case_ids)
        selected = [case for case in cases if str(case.get("id")) in wanted]
        missing = sorted(wanted - {str(case.get("id")) for case in selected})
        if missing:
            raise ValueError(f"Unknown case ids: {', '.join(missing[:10])}")
        return selected
    selected: list[dict[str, Any]] = []
    for case in cases:
        if dataset and str(case.get("dataset")) != dataset:
            continue
        if split and str(case.get("split")) != split:
            continue
        selected.append(case)
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


def _dataset_rows(
    *,
    method: str,
    evaluation: dict[str, Any],
    case_count: int,
) -> list[dict[str, Any]]:
    rows = [
        {
            "method": method,
            "dataset": "<all>",
            "case_count": case_count,
            "evaluated_count": evaluation.get("evaluated_count"),
            "correct": evaluation.get("correct"),
            "pass@1": evaluation.get("pass@1"),
            "adjusted_pass@1": evaluation.get("adjusted_pass@1"),
            "missing_count": len(evaluation.get("missing_ids") or []),
        }
    ]
    for dataset, values in sorted(evaluation.get("by_dataset", {}).items()):
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "case_count": values.get("case_count"),
                "evaluated_count": values.get("evaluated_count"),
                "correct": values.get("correct"),
                "pass@1": values.get("pass@1"),
                "adjusted_pass@1": values.get("adjusted_pass@1"),
                "missing_count": len(values.get("missing_ids") or []),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "dataset",
        "case_count",
        "evaluated_count",
        "correct",
        "pass@1",
        "adjusted_pass@1",
        "missing_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="docs/ORQ_Dataset")
    parser.add_argument("--dataset", help="Optional dataset filter.")
    parser.add_argument("--split", help="Optional split filter.")
    parser.add_argument("--case-set", help="JSON file with case ids or case entries.")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help="Prediction source as name=path. Directory paths use predictions.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/orq_prediction_summary",
        help="Directory for summary JSON/CSV outputs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    case_ids = _load_case_ids(Path(args.case_set)) if args.case_set else None
    cases = _filter_cases(
        load_orq_dataset(args.root),
        dataset=args.dataset,
        split=args.split,
        case_ids=case_ids,
    )
    output_dir = Path(args.output_dir)
    all_results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for method_arg in args.method:
        name, path = _parse_method(method_arg)
        predictions = _load_predictions(path)
        evaluation = evaluate_orq_predictions(cases, predictions, k=max(1, int(args.k)))
        all_results[name] = {
            "path": str(path),
            "evaluation": evaluation,
        }
        rows.extend(_dataset_rows(method=name, evaluation=evaluation, case_count=len(cases)))

    _write_json(output_dir / "prediction_summary.json", all_results)
    _write_csv(output_dir / "prediction_summary.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
