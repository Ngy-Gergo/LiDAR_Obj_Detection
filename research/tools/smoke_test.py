"""Smoke-test one explicit completed canonical run."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_SOURCE = REPOSITORY_ROOT / "research" / "src"

if str(RESEARCH_SOURCE) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SOURCE))

from lidar_model_selection.evaluation import (  # noqa: E402
    DEFAULT_RUNS_ROOT,
    smoke_run,
)
from lidar_model_selection.runs import validate_run_id  # noqa: E402


def run_id_argument(value: str) -> str:
    try:
        return validate_run_id(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def gpu_visibility_argument(value: str) -> str:
    if not value or value.strip() != value or "\0" in value:
        raise argparse.ArgumentTypeError(
            "GPU visibility must be non-empty canonical text"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test one completed run's selected checkpoint on one GPU."
        ),
    )
    parser.add_argument(
        "--run",
        dest="run_id",
        required=True,
        type=run_id_argument,
        metavar="RUN_ID",
        help="Canonical completed run ID to smoke-test.",
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
        record = smoke_run(DEFAULT_RUNS_ROOT / arguments.run_id)
    except Exception as error:
        print(
            f"ERROR: smoke test failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    result_path = (
        DEFAULT_RUNS_ROOT
        / arguments.run_id
        / "smoke"
        / record.result_id
        / "result.json"
    ).absolute()
    print(f"result ID: {record.result_id}")
    print(f"result path: {result_path}")
    print(f"result status: {record.status}")
    if not record.successful:
        assert record.failure is not None
        print(
            "ERROR: smoke test failed: "
            f"{record.failure.error_type}: {record.failure.message}",
            file=sys.stderr,
        )
        return 1

    summary = record.payload["outputs"]
    print(f"run: {record.binding.run_id}")
    print(
        "selected checkpoint SHA-256: "
        f"{record.binding.checkpoint_sha256}"
    )
    print(f"loss keys: {summary['loss_keys']}")  # type: ignore[index]
    print(f"total loss: {summary['total_loss']:.6f}")  # type: ignore[index]
    print(
        "finite gradient tensors: "
        f"{summary['finite_gradient_tensors']}"  # type: ignore[index]
    )
    print(
        f"prediction boxes: {summary['prediction_boxes_shape']}"  # type: ignore[index]
    )
    print(
        f"prediction scores: {summary['prediction_scores_shape']}"  # type: ignore[index]
    )
    print(
        f"prediction labels: {summary['prediction_labels_shape']}"  # type: ignore[index]
    )
    print("CenterPoint smoke test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
