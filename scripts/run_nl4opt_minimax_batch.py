#!/usr/bin/env python3
"""Compatibility wrapper for running NL4OPT cases with the generic ORQ runner."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_orq_minimax_batch import main as _orq_main


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    if "--dataset" not in args:
        args = ["--dataset", "NL4OPT", *args]
    return _orq_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
