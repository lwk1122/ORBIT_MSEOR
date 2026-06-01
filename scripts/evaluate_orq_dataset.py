#!/usr/bin/env python3
"""Evaluate local ORQ_Dataset cases and optional solver predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from orbit_or.evaluation import (
    evaluate_orq_predictions,
    load_orq_dataset,
    summarize_orq_cases,
)


def _load_predictions(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("predictions"), dict):
        return data["predictions"]
    if isinstance(data, dict):
        return data
    raise ValueError("Predictions must be an object or {'predictions': {...}}")


def _load_case_ids(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
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


def _filter_cases(
    cases: list[dict],
    *,
    dataset: str | None = None,
    split: str | None = None,
    case_ids: list[str] | None = None,
) -> list[dict]:
    if case_ids:
        wanted = set(case_ids)
        selected = [case for case in cases if str(case.get("id")) in wanted]
        missing = sorted(wanted - {str(case.get("id")) for case in selected})
        if missing:
            raise ValueError(f"Unknown case ids: {', '.join(missing[:10])}")
        return selected
    selected = []
    for case in cases:
        if dataset and str(case.get("dataset")) != dataset:
            continue
        if split and str(case.get("split")) != split:
            continue
        selected.append(case)
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="docs/ORQ_Dataset",
        help="Path to local ORQ_Dataset root.",
    )
    parser.add_argument("--dataset", help="Optional dataset filter.")
    parser.add_argument("--split", help="Optional split filter.")
    parser.add_argument("--case-set", help="JSON file with selected case ids.")
    parser.add_argument(
        "--predictions",
        help=(
            "Optional JSON predictions keyed by normalized case id. Values may be "
            "single solver outputs or candidate lists."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="pass@k candidate cutoff when predictions contain multiple candidates.",
    )
    args = parser.parse_args(argv)

    case_ids = _load_case_ids(Path(args.case_set)) if args.case_set else None
    cases = _filter_cases(
        load_orq_dataset(args.root),
        dataset=args.dataset,
        split=args.split,
        case_ids=case_ids,
    )
    if not args.predictions:
        print(json.dumps(summarize_orq_cases(cases), indent=2, sort_keys=True))
        return 0

    predictions = _load_predictions(Path(args.predictions))
    result = evaluate_orq_predictions(cases, predictions, k=max(1, args.k))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not result["missing_ids"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
