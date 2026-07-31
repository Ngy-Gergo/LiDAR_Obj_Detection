"""Benchmark one or all CenterPoint models on one selected GPU."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_SOURCE = REPOSITORY_ROOT / "research/src"
if str(RESEARCH_SOURCE) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SOURCE))


def nonnegative_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {value!r}"
        ) from exc
    if result < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return result


def positive_integer(value: str) -> int:
    result = nonnegative_integer(value)
    if result == 0:
        raise argparse.ArgumentTypeError("must be one or greater")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark one or all trained CenterPoint models."
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        help="CenterPoint config for a single-model benchmark.",
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        help="Checkpoint for a single-model benchmark.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Benchmark all models with discoverable checkpoints.",
    )
    parser.add_argument(
        "--gpu",
        type=nonnegative_integer,
        default=0,
        help="Physical CUDA GPU index (default: 0).",
    )
    parser.add_argument(
        "--warmup",
        type=nonnegative_integer,
        default=100,
        help="Warm-up batch count (default: 100).",
    )
    parser.add_argument(
        "--samples",
        type=positive_integer,
        default=1000,
        help="Measured batch count (default: 1000).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/benchmarks"),
        help="Output directory (default: research/benchmarks).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the benchmark plan without inference.",
    )
    return parser


def validate_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.all and (
        args.config is not None or args.checkpoint is not None
    ):
        parser.error("--all cannot be used with CONFIG or CHECKPOINT")
    if not args.all and (
        args.config is None or args.checkpoint is None
    ):
        parser.error(
            "single-model benchmarking requires CONFIG and CHECKPOINT"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_arguments(parser, args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from lidar_model_selection.benchmarking import run_benchmark

    return run_benchmark(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        all_models=args.all,
        gpu_index=args.gpu,
        warmup=args.warmup,
        samples=args.samples,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
