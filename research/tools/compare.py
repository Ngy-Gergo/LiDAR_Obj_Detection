"""Build one comparison from explicit run and result identities."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_SOURCE = REPOSITORY_ROOT / "research" / "src"
if str(RESEARCH_SOURCE) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SOURCE))

from lidar_model_selection.comparison import (  # noqa: E402
    DEFAULT_RUNS_ROOT,
    KITTI_CAR_AP40_METRICS,
    RUNTIME_SCOPES,
    RUNTIME_STATISTICS,
    CompatibilityWaiver,
    compare_runs,
    write_comparison_report,
)
from lidar_model_selection.runs import load_run, validate_run_id  # noqa: E402


def run_id_argument(value: str) -> str:
    try:
        return validate_run_id(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def assignment_argument(value: str) -> tuple[str, str]:
    key, separator, assigned = value.partition("=")
    if not separator or not key or not assigned:
        raise argparse.ArgumentTypeError("expected RUN_ID=RESULT_ID")
    try:
        validate_run_id(key)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if assigned.strip() != assigned or "\0" in assigned:
        raise argparse.ArgumentTypeError("result ID must be canonical text")
    return key, assigned


def waiver_argument(value: str) -> CompatibilityWaiver:
    field, separator, reason = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("expected FIELD=REASON")
    try:
        return CompatibilityWaiver(field=field, reason=reason)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _unique_assignments(
    values: Sequence[tuple[str, str]],
    *,
    description: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for run_id, result_id in values:
        if run_id in result:
            raise ValueError(f"duplicate {description} for run {run_id}")
        result[run_id] = result_id
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare explicitly named completed runs using exact immutable "
            "evaluation and optional benchmark results."
        )
    )
    parser.add_argument(
        "--run",
        dest="run_ids",
        required=True,
        action="append",
        type=run_id_argument,
        metavar="RUN_ID",
        help="Completed canonical run ID; repeat for every cohort member.",
    )
    parser.add_argument(
        "--evaluation-result",
        action="append",
        default=[],
        type=assignment_argument,
        metavar="RUN_ID=RESULT_ID",
        help=(
            "Exact evaluation result for a run. If omitted, that run must "
            "own exactly one successful evaluation."
        ),
    )
    parser.add_argument(
        "--benchmark-result",
        action="append",
        default=[],
        type=assignment_argument,
        metavar="RUN_ID=RESULT_ID",
        help=(
            "Exact benchmark result for a run. If omitted while runtime is "
            "requested, that run must own exactly one successful benchmark."
        ),
    )
    parser.add_argument(
        "--accuracy-metric",
        required=True,
        choices=tuple(KITTI_CAR_AP40_METRICS),
        help="Resolved KITTI Car AP40 metric used for accuracy ranking.",
    )
    parser.add_argument(
        "--runtime-scope",
        choices=RUNTIME_SCOPES,
        help="Optional benchmark timing scope used for runtime ranking.",
    )
    parser.add_argument(
        "--runtime-statistic",
        choices=RUNTIME_STATISTICS,
        help="Optional statistic within --runtime-scope.",
    )
    parser.add_argument(
        "--waiver",
        action="append",
        default=[],
        type=waiver_argument,
        metavar="FIELD=REASON",
        help="Explicit field-level compatibility waiver; repeat as needed.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="REPORT.json",
        help="Explicit durable JSON report path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if (arguments.runtime_scope is None) != (
        arguments.runtime_statistic is None
    ):
        print(
            "ERROR: --runtime-scope and --runtime-statistic must be used together",
            file=sys.stderr,
        )
        return 2

    try:
        if len(arguments.run_ids) != len(set(arguments.run_ids)):
            raise ValueError("--run values must be unique")
        evaluation_ids = _unique_assignments(
            arguments.evaluation_result,
            description="evaluation result selection",
        )
        benchmark_ids = _unique_assignments(
            arguments.benchmark_result,
            description="benchmark result selection",
        )
        runs = tuple(
            load_run(DEFAULT_RUNS_ROOT / run_id)
            for run_id in arguments.run_ids
        )
        report = compare_runs(
            runs,
            accuracy_metric=arguments.accuracy_metric,
            evaluation_result_ids=evaluation_ids,
            runtime_scope=arguments.runtime_scope,
            runtime_statistic=arguments.runtime_statistic,
            benchmark_result_ids=benchmark_ids,
            waivers=arguments.waiver,
        )
        write_comparison_report(arguments.output, report)
    except Exception as error:
        print(
            f"ERROR: comparison failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    print(
        f"REPORT: path={arguments.output} rows={len(report.rows)} "
        f"accuracy={report.accuracy_metric}"
    )
    for row in sorted(report.rows, key=lambda item: item.accuracy_rank):
        runtime = (
            ""
            if row.runtime_value is None
            else f" runtime={row.runtime_value:.6g} runtime_rank={row.runtime_rank}"
        )
        print(
            f"ROW: run={row.run_id} accuracy={row.accuracy_value:.6g} "
            f"accuracy_rank={row.accuracy_rank}{runtime}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
