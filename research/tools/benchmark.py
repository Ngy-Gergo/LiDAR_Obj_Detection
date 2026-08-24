"""Benchmark one explicit completed canonical run."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = REPOSITORY_ROOT / "research"
RESEARCH_SOURCE = RESEARCH_ROOT / "src"

if str(RESEARCH_SOURCE) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SOURCE))

from lidar_model_selection.benchmarking import (  # noqa: E402
    DEFAULT_RUNS_ROOT,
    benchmark_run,
)
from lidar_model_selection.runs import validate_run_id  # noqa: E402


def run_id_argument(value: str) -> str:
    try:
        return validate_run_id(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {value!r}"
        ) from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def gpu_visibility_argument(value: str) -> str:
    if not value or value.strip() != value or "\0" in value:
        raise argparse.ArgumentTypeError(
            "GPU visibility must be non-empty canonical text"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark one explicit completed canonical run."
    )
    parser.add_argument(
        "--run",
        dest="run_id",
        required=True,
        type=run_id_argument,
        metavar="RUN_ID",
        help="Canonical completed run ID to benchmark.",
    )
    parser.add_argument(
        "--warmup",
        required=True,
        type=positive_integer,
        metavar="COUNT",
        help="Positive number of leading warm-up samples.",
    )
    parser.add_argument(
        "--samples",
        required=True,
        type=positive_integer,
        metavar="COUNT",
        help="Positive number of consecutive measured samples.",
    )
    parser.add_argument(
        "--gpu",
        type=gpu_visibility_argument,
        metavar="VISIBILITY",
        help="Optional CUDA_VISIBLE_DEVICES value set before ML imports.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpu

    try:
        record = benchmark_run(
            DEFAULT_RUNS_ROOT / arguments.run_id,
            warmup=arguments.warmup,
            samples=arguments.samples,
        )
    except Exception as error:
        print(
            f"ERROR: benchmark could not be recorded: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    print(
        f"RESULT: run={record.binding.run_id} "
        f"result={record.result_id} status={record.status}"
    )
    if record.failure is not None:
        print(
            f"ERROR: {record.failure.error_type}: {record.failure.message}",
            file=sys.stderr,
        )
    return 0 if record.successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
