"""Import one explicitly enumerated historical run."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_SOURCE = REPOSITORY_ROOT / "research" / "src"

if str(RESEARCH_SOURCE) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SOURCE))

from lidar_model_selection.imports import (  # noqa: E402
    import_historical_run,
    read_dataset_identity,
)
from lidar_model_selection.results import list_results  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import one historical completed run from exact, explicitly named "
            "evidence files."
        )
    )
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--final-checkpoint", required=True, type=Path)
    parser.add_argument("--selected-checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-identity", required=True, type=Path)
    parser.add_argument("--evaluation-json", type=Path)
    parser.add_argument("--benchmark-json", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--created-at")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        dataset = read_dataset_identity(arguments.dataset_identity)
        run = import_historical_run(
            arguments.runs_root,
            slug=arguments.slug,
            config_path=arguments.config,
            final_checkpoint_path=arguments.final_checkpoint,
            selected_checkpoint_path=arguments.selected_checkpoint,
            dataset_identity=dataset,
            evaluation_json=arguments.evaluation_json,
            benchmark_json=arguments.benchmark_json,
            run_id=arguments.run_id,
            created_at=arguments.created_at,
        )
    except Exception as error:
        print(
            f"ERROR: historical import failed: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    evaluation_count = len(list_results(run, "evaluation"))
    benchmark_count = len(list_results(run, "benchmark"))
    print(
        f"IMPORTED: run={run.run_id} evaluations={evaluation_count} "
        f"benchmarks={benchmark_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
