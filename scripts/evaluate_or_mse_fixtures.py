#!/usr/bin/env python3
"""Evaluate local OR/MSE gold fixtures without provider calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from orbit_or.evaluation import evaluate_gold_set_predictions, load_or_mse_gold_set


def _load_predictions(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("predictions"), dict):
        return data["predictions"]
    if isinstance(data, dict):
        return data
    raise ValueError("Predictions must be an object or {'predictions': {...}}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "gold_set",
        nargs="?",
        default="tests/fixtures/or_mse_gold.json",
        help="Path to OR/MSE gold-set JSON.",
    )
    parser.add_argument(
        "--predictions",
        help="Optional predictions JSON keyed by case id. If omitted, only validates the fixture.",
    )
    args = parser.parse_args(argv)

    cases = load_or_mse_gold_set(args.gold_set)
    if not args.predictions:
        summary = {
            "case_count": len(cases),
            "task_types": sorted({case["task_type"] for case in cases}),
            "case_ids": [case["id"] for case in cases],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    predictions = _load_predictions(Path(args.predictions))
    result = evaluate_gold_set_predictions(cases, predictions)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not result["missing_ids"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
