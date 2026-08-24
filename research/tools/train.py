"""Create or execute canonical training runs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = REPOSITORY_ROOT / "research"
RESEARCH_SOURCE = RESEARCH_ROOT / "src"
TRAIN_TOOL = Path(__file__).absolute()

if str(RESEARCH_SOURCE) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SOURCE))

from lidar_model_selection.catalog import (  # noqa: E402
    catalog_slugs,
    source_config_for_slug,
)
from lidar_model_selection.runs import (  # noqa: E402
    Run,
    load_run,
    validate_run_id,
)
from lidar_model_selection.scheduling import (  # noqa: E402
    GpuSlot,
    RunOutcome,
    build_train_command,
    discover_gpu_slots,
    schedule_runs,
)
from lidar_model_selection.training import (  # noqa: E402
    DEFAULT_REPOSITORY_ROOT,
    DEFAULT_RUNS_ROOT,
    create_training_run,
    decide_training,
    execute_training,
)


def positive_integer(value: str) -> int:
    """Parse a strictly positive command-line integer."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not an integer"
        ) from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def run_id_argument(value: str) -> str:
    """Parse one canonical run ID for argparse."""
    try:
        return validate_run_id(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create one catalog-backed training run, execute one explicit "
            "run, or queue independent runs for the full catalog."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--model",
        choices=catalog_slugs(),
        metavar="SLUG",
        help="Create and train one canonical catalog model.",
    )
    mode.add_argument(
        "--run",
        dest="run_id",
        type=run_id_argument,
        metavar="RUN_ID",
        help="Execute, resume, or finalize one explicit existing run.",
    )
    mode.add_argument(
        "--all",
        dest="all_models",
        action="store_true",
        help="Create independent runs for every catalog model and queue them.",
    )
    parser.add_argument(
        "--max-epochs",
        type=positive_integer,
        metavar="N",
        help="Required fixed target epoch for --model and --all.",
    )
    parser.add_argument(
        "--parent-run",
        type=run_id_argument,
        metavar="RUN_ID",
        help="Existing parent lineage for a newly created --model run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Preview creation without publishing runs, or inspect the next "
            "decision for an existing --run without executing it."
        ),
    )
    return parser


def validate_arguments(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
) -> None:
    creating = arguments.model is not None or arguments.all_models
    if creating and arguments.max_epochs is None:
        parser.error("--max-epochs is required with --model and --all")
    if arguments.run_id is not None and arguments.max_epochs is not None:
        parser.error("--max-epochs cannot change an existing run")
    if arguments.parent_run is not None and arguments.model is None:
        parser.error("--parent-run is only valid with --model")


def _error(context: str, error: BaseException) -> None:
    print(
        f"ERROR: {context}: {type(error).__name__}: {error}",
        file=sys.stderr,
    )


def _print_creation_plan(
    slugs: Sequence[str],
    target_epoch: int,
    *,
    parent_run_id: str | None = None,
) -> None:
    print("DRY RUN: no run directories will be created")
    for slug in slugs:
        source = source_config_for_slug(slug)
        print(
            f"PLAN: model={slug} target_epoch={target_epoch} "
            f"source_config={source} parent_run={parent_run_id or 'none'}"
        )


def _print_completed(run: Run) -> None:
    outputs = run.manifest.training.outputs
    if outputs is None:
        print(
            f"RUN: {run.run_id} "
            f"status={run.manifest.training.status}"
        )
        return
    print(
        f"COMPLETE: run={run.run_id} "
        f"selected_checkpoint={outputs.selected_checkpoint.path}"
    )


def _run_model(arguments: argparse.Namespace) -> int:
    assert arguments.model is not None
    assert arguments.max_epochs is not None
    try:
        parent_run = (
            None
            if arguments.parent_run is None
            else load_run(DEFAULT_RUNS_ROOT / arguments.parent_run)
        )
    except Exception as error:
        _error(f"could not load parent run {arguments.parent_run!r}", error)
        return 1
    if arguments.dry_run:
        _print_creation_plan(
            (arguments.model,),
            arguments.max_epochs,
            parent_run_id=(None if parent_run is None else parent_run.run_id),
        )
        return 0

    try:
        run = create_training_run(
            arguments.model,
            arguments.max_epochs,
            parent_run=parent_run,
        )
        print(f"CREATED: run={run.run_id} model={arguments.model}")
        completed = execute_training(run)
    except Exception as error:
        _error(f"could not train catalog model {arguments.model!r}", error)
        return 1

    _print_completed(completed)
    return 0


def _run_existing(arguments: argparse.Namespace) -> int:
    assert arguments.run_id is not None
    try:
        run = load_run(DEFAULT_RUNS_ROOT / arguments.run_id)
        if arguments.dry_run:
            decision = decide_training(run)
            resume = (
                "none"
                if decision.resume_checkpoint is None
                else decision.resume_checkpoint.path
            )
            print(
                f"PLAN: run={run.run_id} action={decision.action} "
                f"resume_checkpoint={resume}"
            )
            return 0
        completed = execute_training(run)
    except Exception as error:
        _error(f"could not execute run {arguments.run_id!r}", error)
        return 1

    _print_completed(completed)
    return 0


def _print_gpus(gpus: Sequence[GpuSlot]) -> None:
    print("CUDA workers:")
    for gpu in gpus:
        print(
            f"  logical={gpu.logical_index} "
            f"visibility={gpu.visibility_token} name={gpu.name}"
        )


def _print_outcomes(outcomes: Sequence[RunOutcome]) -> bool:
    failed = False
    print("Training outcomes:")
    for outcome in outcomes:
        if outcome.successful:
            print(
                f"  PASS: run={outcome.run_id} "
                f"gpu={outcome.gpu.visibility_token}"
            )
            continue
        failed = True
        print(
            f"  FAIL: run={outcome.run_id} "
            f"gpu={outcome.gpu.visibility_token} error={outcome.error}",
            file=sys.stderr,
        )
    return failed


def _run_all(arguments: argparse.Namespace) -> int:
    assert arguments.max_epochs is not None
    slugs = catalog_slugs()
    if arguments.dry_run:
        _print_creation_plan(slugs, arguments.max_epochs)
        print("PLAN: published run IDs would be queued across discovered CUDA GPUs")
        return 0

    run_ids = []
    creation_failed = False
    for slug in slugs:
        try:
            run = create_training_run(slug, arguments.max_epochs)
        except Exception as error:
            creation_failed = True
            _error(f"could not create catalog run {slug!r}", error)
            continue
        run_ids.append(run.run_id)
        print(f"CREATED: run={run.run_id} model={slug}")

    if not run_ids:
        return 1

    try:
        gpus = discover_gpu_slots()
    except Exception as error:
        _error("could not discover CUDA workers", error)
        print(
            "Created runs remain pending and can be executed with --run.",
            file=sys.stderr,
        )
        return 1

    _print_gpus(gpus)
    try:
        outcomes = schedule_runs(
            tuple(run_ids),
            gpus,
            lambda run_id: build_train_command(TRAIN_TOOL, run_id),
            cwd=DEFAULT_REPOSITORY_ROOT,
        )
    except Exception as error:
        _error("training queue failed", error)
        return 1

    outcome_failed = _print_outcomes(outcomes)
    return 1 if creation_failed or outcome_failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    validate_arguments(parser, arguments)
    try:
        if arguments.model is not None:
            return _run_model(arguments)
        if arguments.run_id is not None:
            return _run_existing(arguments)
        return _run_all(arguments)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
