#!/usr/bin/env python3
"""Render one explicit, resolved comparison report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_SOURCE = REPOSITORY_ROOT / "research" / "src"
if str(RESEARCH_SOURCE) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SOURCE))

from lidar_model_selection.comparison import load_comparison_report
from lidar_model_selection.plotting import plot_comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render figures from one resolved comparison report."
    )
    parser.add_argument("report", type=Path, help="Comparison report JSON.")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        report = load_comparison_report(arguments.report)
        outputs = plot_comparison(report.rows, arguments.output_dir)
    except Exception as error:
        print(
            f"ERROR: plotting failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
