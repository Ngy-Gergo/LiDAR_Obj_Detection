#!/usr/bin/env python3
"""Execute one ordinary end-to-end research pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_SOURCE = REPOSITORY_ROOT / "research" / "src"
if str(RESEARCH_SOURCE) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SOURCE))

from lidar_model_selection.catalog import catalog_slugs
from lidar_model_selection.pipeline import PipelineRequest, run_pipeline


def positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create, preflight, train, evaluate, and benchmark one run."
    )
    parser.add_argument("preset", nargs="?", choices=catalog_slugs())
    parser.add_argument("--config", type=Path, help="Explicit MMEngine config.")
    parser.add_argument("--name", help="Canonical run slug for --config.")
    parser.add_argument("--max-epochs", required=True, type=positive_integer)
    parser.add_argument("--warmup", type=positive_integer, default=100)
    parser.add_argument("--samples", type=positive_integer, default=1000)
    parser.add_argument(
        "--no-sample-check",
        action="store_true",
        help="Skip the one-sample dataset accessibility check.",
    )
    return parser


def _validate(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> None:
    if (arguments.config is None) == (arguments.preset is None):
        parser.error("provide exactly one preset or --config")
    if arguments.config is not None and not arguments.name:
        parser.error("--name is required with --config")
    if arguments.config is None and arguments.name is not None:
        parser.error("--name is only valid with --config")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    _validate(parser, arguments)
    request = PipelineRequest(
        slug=arguments.preset or arguments.name,
        target_epoch=arguments.max_epochs,
        source_config=arguments.config,
        benchmark_warmup=arguments.warmup,
        benchmark_samples=arguments.samples,
        sample_check=not arguments.no_sample_check,
    )
    try:
        run, record = run_pipeline(request)
    except Exception as error:
        print(
            f"ERROR: pipeline failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    print(
        f"COMPLETE: run={run.run_id} pipeline={record.pipeline_id} "
        f"evaluation={record.evaluation_result_id} "
        f"benchmark={record.benchmark_result_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
