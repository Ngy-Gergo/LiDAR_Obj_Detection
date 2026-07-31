"""Train one CenterPoint model or schedule all models across CUDA GPUs."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TextIO


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = REPOSITORY_ROOT / "research"
RESEARCH_SOURCE = RESEARCH_ROOT / "src"
EXPERIMENT_ROOT = RESEARCH_ROOT / "experiments"
DEFAULT_ALL_MAX_EPOCHS = 10
RUNNABLE_ACTIONS = ("train", "resume", "restart")

if str(RESEARCH_SOURCE) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SOURCE))

from lidar_model_selection.checkpoints import (  # noqa: E402
    CENTERPOINT_MODELS,
    checkpoint_epoch,
    is_usable_checkpoint,
)


@dataclass(frozen=True)
class TrainingPlan:
    model: str
    config_path: Path
    work_dir: Path | None
    target_epoch: int
    action: str
    reason: str
    resume_checkpoint: Path | None = None

    @property
    def console_log(self) -> Path | None:
        if self.work_dir is None:
            return None
        return self.work_dir.parent / f"{self.work_dir.name}.console.log"


@dataclass(frozen=True)
class GpuSlot:
    logical_index: int
    visibility_token: str
    name: str


@dataclass
class RunningJob:
    plan: TrainingPlan
    gpu: GpuSlot
    process: subprocess.Popen
    console_stream: TextIO


def positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not an integer"
        ) from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one CenterPoint model or schedule all CenterPoint models."
        )
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        help="Path to one MMEngine configuration file.",
    )
    parser.add_argument(
        "--all",
        dest="all_models",
        action="store_true",
        help="Train all six CenterPoint configurations.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Override the work directory in single-model mode.",
    )
    parser.add_argument(
        "--max-epochs",
        type=positive_integer,
        help=(
            "Override train_cfg.max_epochs. --all defaults to 10; "
            "single-model mode otherwise uses the config value."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an exact numeric checkpoint below the target.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and restart deterministic --all work directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the --all plan without modifying files or training.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help=argparse.SUPPRESS,
    )
    return parser


def validate_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.all_models and args.config is not None:
        parser.error("--all cannot be used together with CONFIG.")
    if not args.all_models and args.config is None:
        parser.error("CONFIG is required unless --all is used.")
    if args.all_models and args.work_dir is not None:
        parser.error("--work-dir cannot be used with --all.")
    if args.force and args.resume:
        parser.error("--force cannot be used together with --resume.")
    if args.force and not args.all_models:
        parser.error("--force is only supported with --all.")
    if args.dry_run and not args.all_models:
        parser.error("--dry-run requires --all.")
    if args.resume_from is not None and not args.resume:
        parser.error("--resume-from requires --resume.")
    if args.resume_from is not None and args.all_models:
        parser.error("--resume-from is only used by single-model children.")


def display_path(path: Path | None) -> str:
    if path is None:
        return "(none)"
    absolute = Path(os.path.abspath(path))
    try:
        resolved = absolute.resolve()
    except (OSError, RuntimeError, ValueError):
        return str(absolute)
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def failed_plan(
    model: str,
    config_path: Path,
    target_epoch: int,
    reason: str,
) -> TrainingPlan:
    return TrainingPlan(
        model,
        config_path,
        None,
        target_epoch,
        "fail",
        reason,
    )


def valid_target_checkpoint(
    work_dir: Path,
    target_epoch: int,
) -> Path | None:
    checkpoint = work_dir / f"epoch_{target_epoch}.pth"
    if checkpoint.is_symlink() or not checkpoint.is_file():
        return None
    if checkpoint.parent.resolve() != work_dir.resolve():
        return None
    if not is_usable_checkpoint(checkpoint):
        return None
    return checkpoint.resolve()


def screen_suffix(model: str, work_dir: Path) -> int | None:
    prefix = f"{model}_screen"
    if not work_dir.name.startswith(prefix):
        return None
    suffix = work_dir.name[len(prefix):]
    if not suffix.isdigit() or int(suffix) < 1:
        return None
    return int(suffix)


def find_experiment_dirs(model: str) -> list[Path]:
    if not EXPERIMENT_ROOT.is_dir():
        return []

    experiment_root = EXPERIMENT_ROOT.resolve()
    directories = []
    for path in EXPERIMENT_ROOT.iterdir():
        if (
            screen_suffix(model, path) is not None
            and not path.is_symlink()
            and path.is_dir()
            and path.resolve().parent == experiment_root
        ):
            directories.append(path.resolve())
    return sorted(
        directories,
        key=lambda path: (
            screen_suffix(model, path),
            path.as_posix(),
        ),
    )


def find_resume_checkpoint(
    work_dir: Path,
    target_epoch: int,
) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in work_dir.glob("epoch_*.pth"):
        epoch = checkpoint_epoch(path)
        if (
            epoch is None
            or path.name != f"epoch_{epoch}.pth"
            or epoch >= target_epoch
            or path.is_symlink()
            or not path.is_file()
            or not is_usable_checkpoint(path)
        ):
            continue
        candidates.append((epoch, path.resolve()))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (item[0], item[1].as_posix()),
    )[1]


def build_resume_plan(
    model: str,
    config_path: Path,
    target_epoch: int,
) -> TrainingPlan:
    if not config_path.is_file():
        reason = f"config does not exist: {display_path(config_path)}"
        return failed_plan(model, config_path, target_epoch, reason)

    directories = find_experiment_dirs(model)
    if not directories:
        reason = (
            f"no existing {model}_screenN experiment directory"
        )
        return failed_plan(model, config_path, target_epoch, reason)

    completed: list[tuple[int, Path]] = []
    for work_dir in directories:
        checkpoint = valid_target_checkpoint(work_dir, target_epoch)
        if checkpoint is not None:
            suffix = screen_suffix(model, work_dir)
            completed.append((suffix, checkpoint))
    if completed:
        checkpoint = max(
            completed,
            key=lambda item: (item[0], item[1].as_posix()),
        )[1]
        return TrainingPlan(
            model,
            config_path,
            checkpoint.parent,
            target_epoch,
            "skip",
            "valid exact target checkpoint: "
            f"{display_path(checkpoint)}",
        )

    candidates: list[tuple[int, int, Path]] = []
    for work_dir in directories:
        checkpoint = find_resume_checkpoint(work_dir, target_epoch)
        if checkpoint is not None:
            epoch = checkpoint_epoch(checkpoint)
            suffix = screen_suffix(model, work_dir)
            candidates.append((epoch, suffix, checkpoint))
    if not candidates:
        reason = (
            "no valid regular epoch_N.pth checkpoint below target "
            f"epoch {target_epoch} exists in an existing experiment "
            "directory"
        )
        return failed_plan(model, config_path, target_epoch, reason)

    checkpoint = max(
        candidates,
        key=lambda item: (item[0], item[1], item[2].as_posix()),
    )[2]
    epoch = checkpoint_epoch(checkpoint)
    return TrainingPlan(
        model,
        config_path,
        checkpoint.parent,
        target_epoch,
        "resume",
        f"resume epoch {epoch} from {display_path(checkpoint)}",
        checkpoint,
    )


def build_fresh_plan(
    model: str,
    config_path: Path,
    target_epoch: int,
    force: bool,
) -> TrainingPlan:
    work_dir = EXPERIMENT_ROOT / f"{model}_screen{target_epoch}"
    if not config_path.is_file():
        reason = f"config does not exist: {display_path(config_path)}"
        return failed_plan(model, config_path, target_epoch, reason)
    if work_dir.is_symlink():
        reason = (
            "deterministic work directory is a symlink: "
            f"{display_path(work_dir)}"
        )
        return failed_plan(model, config_path, target_epoch, reason)
    if work_dir.exists() and not work_dir.is_dir():
        reason = (
            "deterministic work path exists but is not a directory: "
            f"{display_path(work_dir)}"
        )
        return failed_plan(model, config_path, target_epoch, reason)
    if not work_dir.exists():
        return TrainingPlan(
            model,
            config_path,
            work_dir,
            target_epoch,
            "train",
            "deterministic work directory does not exist",
        )
    if force:
        return TrainingPlan(
            model,
            config_path,
            work_dir,
            target_epoch,
            "restart",
            "force requested for the exact deterministic work directory",
        )

    checkpoint = valid_target_checkpoint(work_dir, target_epoch)
    if checkpoint is not None:
        return TrainingPlan(
            model,
            config_path,
            work_dir,
            target_epoch,
            "skip",
            f"valid exact target checkpoint: {display_path(checkpoint)}",
        )
    reason = (
        f"{display_path(work_dir)} exists without a valid exact target "
        "checkpoint; use --resume or --force"
    )
    return failed_plan(model, config_path, target_epoch, reason)


def build_training_plans(
    target_epoch: int,
    *,
    resume: bool,
    force: bool,
) -> list[TrainingPlan]:
    plans: list[TrainingPlan] = []
    for model_spec in CENTERPOINT_MODELS:
        try:
            config_path = model_spec.config_path.resolve()
            if resume:
                plan = build_resume_plan(
                    model_spec.name,
                    config_path,
                    target_epoch,
                )
            else:
                plan = build_fresh_plan(
                    model_spec.name,
                    config_path,
                    target_epoch,
                    force,
                )
        except OSError as error:
            reason = (
                f"cannot inspect experiment paths: {error}"
            )
            plan = failed_plan(
                model_spec.name,
                model_spec.config_path,
                target_epoch,
                reason,
            )
        plans.append(plan)
    return plans


def remove_restart_directory(plan: TrainingPlan) -> None:
    if plan.action != "restart":
        raise RuntimeError(
            f"Refusing restart deletion for action {plan.action!r}."
        )
    if plan.work_dir is None:
        raise RuntimeError("Restart plan has no work directory.")
    if not any(
        model_spec.name == plan.model
        for model_spec in CENTERPOINT_MODELS
    ):
        raise RuntimeError(
            f"Refusing restart deletion for unknown model {plan.model!r}."
        )
    expected = (
        EXPERIMENT_ROOT
        / f"{plan.model}_screen{plan.target_epoch}"
    )
    work_dir = plan.work_dir
    if work_dir != expected:
        raise RuntimeError(
            "Refusing to delete a path other than the exact deterministic "
            f"work directory: {work_dir}"
        )
    if work_dir.parent != EXPERIMENT_ROOT:
        raise RuntimeError(
            f"Refusing to delete outside {EXPERIMENT_ROOT}."
        )
    if EXPERIMENT_ROOT.is_symlink():
        raise RuntimeError(
            f"Refusing deletion through symlinked root: {EXPERIMENT_ROOT}"
        )
    if work_dir.is_symlink():
        raise RuntimeError(
            f"Refusing to delete symlink: {work_dir}"
        )
    if not work_dir.is_dir():
        raise RuntimeError(
            f"Restart path is not a directory: {work_dir}"
        )
    shutil.rmtree(work_dir)


def validate_completed_training(
    work_dir: Path,
    target_epoch: int,
) -> str | None:
    final_checkpoint = valid_target_checkpoint(
        work_dir,
        target_epoch,
    )
    expected = work_dir / f"epoch_{target_epoch}.pth"
    if final_checkpoint is None:
        return (
            "training exited successfully but the exact final checkpoint "
            f"is missing or invalid: {display_path(expected)}"
        )

    marker = work_dir / "last_checkpoint"
    if marker.is_symlink():
        return f"{display_path(marker)} is a symlink"
    if not marker.exists():
        return None
    if not marker.is_file():
        return f"{display_path(marker)} is not a regular file"
    try:
        recorded_text = marker.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as error:
        return f"cannot read {display_path(marker)}: {error}"
    if not recorded_text:
        return f"{display_path(marker)} is empty"

    recorded = Path(recorded_text)
    if not recorded.is_absolute():
        recorded = work_dir / recorded
    try:
        recorded = recorded.resolve()
    except (OSError, RuntimeError, ValueError) as error:
        return (
            f"cannot resolve checkpoint from "
            f"{display_path(marker)}: {error}"
        )
    if recorded.parent != work_dir.resolve():
        return (
            f"{display_path(marker)} points outside its experiment "
            "directory"
        )
    if recorded != final_checkpoint:
        return (
            f"{display_path(marker)} points to "
            f"{display_path(recorded)}, not the exact final checkpoint "
            f"{display_path(final_checkpoint)}"
        )
    return None


def detect_gpus() -> list[GpuSlot]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required to detect CUDA GPUs."
        ) from error

    device_count = torch.cuda.device_count()
    if not torch.cuda.is_available() or device_count < 1:
        raise RuntimeError("At least one CUDA GPU is required for --all.")

    visibility_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visibility_mask is None:
        tokens = [str(index) for index in range(device_count)]
    else:
        tokens = [
            token.strip()
            for token in visibility_mask.split(",")
            if token.strip()
        ]
        if len(tokens) < device_count:
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES exposes fewer tokens than "
                "PyTorch detected."
            )
        tokens = tokens[:device_count]

    return [
        GpuSlot(
            logical_index,
            token,
            torch.cuda.get_device_name(logical_index),
        )
        for logical_index, token in enumerate(tokens)
    ]


def print_detected_gpus(gpus: Sequence[GpuSlot]) -> None:
    print(f"Detected CUDA GPUs ({len(gpus)}):")
    for gpu in gpus:
        print(
            f"  GPU {gpu.visibility_token}: {gpu.name} "
            f"(PyTorch visible index {gpu.logical_index})"
        )


def print_dry_run(
    plans: Sequence[TrainingPlan],
    gpus: Sequence[GpuSlot],
) -> None:
    print("Dry-run model plan:")
    for index, plan in enumerate(plans, start=1):
        resume_epoch = (
            checkpoint_epoch(plan.resume_checkpoint)
            if plan.resume_checkpoint is not None
            else None
        )
        print(f"  {index}. {plan.model}")
        print(f"     config: {display_path(plan.config_path)}")
        print(f"     work directory: {display_path(plan.work_dir)}")
        print(f"     target epoch: {plan.target_epoch}")
        print(
            "     resume checkpoint: "
            + display_path(plan.resume_checkpoint)
        )
        print(
            "     resume epoch: "
            + (str(resume_epoch) if resume_epoch is not None else "(none)")
        )
        print(f"     console log: {display_path(plan.console_log)}")
        print(f"     action: {plan.action.upper()}")
        print(f"     reason: {plan.reason}")

    queued = [
        plan for plan in plans if plan.action in RUNNABLE_ACTIONS
    ]
    queue_text = ", ".join(plan.model for plan in queued) or "(empty)"
    print(f"Model queue: {queue_text}")
    print("Initial assignments:")
    assignments = list(zip(gpus, queued))
    if not assignments:
        print("  (none)")
    for gpu, plan in assignments:
        print(f"  GPU {gpu.visibility_token}: {plan.model}")


def start_training_job(
    plan: TrainingPlan,
    gpu: GpuSlot,
) -> RunningJob:
    if plan.action not in RUNNABLE_ACTIONS:
        raise RuntimeError(
            f"Cannot start {plan.model} with action {plan.action!r}."
        )
    if plan.work_dir is None:
        raise RuntimeError(
            f"Training plan for {plan.model} has no work directory."
        )
    if plan.action == "resume" and plan.resume_checkpoint is None:
        raise RuntimeError(
            f"Resume plan for {plan.model} has no checkpoint."
        )

    console_log = plan.console_log
    if console_log is None:
        raise RuntimeError(
            f"Training plan for {plan.model} has no console log."
        )
    if EXPERIMENT_ROOT.is_symlink():
        raise RuntimeError(
            f"Refusing to write through symlinked root: {EXPERIMENT_ROOT}"
        )
    if console_log.parent.resolve() != EXPERIMENT_ROOT.resolve():
        raise RuntimeError(
            f"Refusing to write log outside {EXPERIMENT_ROOT}."
        )
    if console_log.is_symlink():
        raise RuntimeError(
            f"Refusing to overwrite symlinked log: {console_log}"
        )
    if console_log.exists() and not console_log.is_file():
        raise RuntimeError(
            f"Refusing to overwrite non-file log path: {console_log}"
        )
    if plan.action == "restart":
        remove_restart_directory(plan)

    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    console_stream = console_log.open("w", encoding="utf-8")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(plan.config_path),
        "--work-dir",
        str(plan.work_dir),
        "--max-epochs",
        str(plan.target_epoch),
    ]
    if plan.action == "resume":
        command.extend([
            "--resume",
            "--resume-from",
            str(plan.resume_checkpoint),
        ])

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu.visibility_token
    environment["PYTHONUNBUFFERED"] = "1"
    try:
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=console_stream,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        console_stream.close()
        raise
    return RunningJob(plan, gpu, process, console_stream)


def stop_jobs(running: Sequence[RunningJob]) -> None:
    for job in running:
        try:
            if job.process.poll() is None:
                job.process.terminate()
        except OSError as error:
            print(
                f"WARNING: could not terminate {job.plan.model}: {error}",
                file=sys.stderr,
            )

    for job in running:
        try:
            job.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                job.process.kill()
                job.process.wait()
            except OSError as error:
                print(
                    f"WARNING: could not kill/reap "
                    f"{job.plan.model}: {error}",
                    file=sys.stderr,
                )
        except OSError as error:
            print(
                f"WARNING: could not reap {job.plan.model}: {error}",
                file=sys.stderr,
            )
        finally:
            job.console_stream.close()


def finish_job(job: RunningJob, return_code: int) -> str | None:
    job.console_stream.close()
    if return_code != 0:
        error = f"child exited with code {return_code}"
    else:
        try:
            error = validate_completed_training(
                job.plan.work_dir,
                job.plan.target_epoch,
            )
        except (OSError, RuntimeError, ValueError) as inspection_error:
            error = (
                f"cannot validate completed training: {inspection_error}"
            )

    if error is None:
        print(
            f"GPU {job.gpu.visibility_token}: "
            f"{job.plan.model} finished successfully"
        )
    else:
        print(
            f"GPU {job.gpu.visibility_token}: {job.plan.model} "
            f"failed: {error} "
            f"(log: {display_path(job.plan.console_log)})"
        )
    return error


def run_job_queue(
    plans: Sequence[TrainingPlan],
    gpus: Sequence[GpuSlot],
) -> tuple[
    list[TrainingPlan],
    list[tuple[TrainingPlan, str]],
]:
    pending = deque(plans)
    available_gpus = deque(gpus)
    running: dict[int, RunningJob] = {}
    successes: list[TrainingPlan] = []
    failures: list[tuple[TrainingPlan, str]] = []

    if pending and not available_gpus:
        raise RuntimeError("Cannot schedule training without a CUDA GPU.")

    try:
        while pending or running:
            while pending and available_gpus:
                plan = pending.popleft()
                gpu = available_gpus.popleft()
                try:
                    job = start_training_job(plan, gpu)
                except Exception as error:
                    traceback.print_exc()
                    reason = (
                        f"could not start: {type(error).__name__}: {error}"
                    )
                    failures.append((plan, reason))
                    print(
                        f"GPU {gpu.visibility_token}: {plan.model} "
                        "failed to start "
                        f"(log: {display_path(plan.console_log)})"
                    )
                    available_gpus.append(gpu)
                    continue
                running[gpu.logical_index] = job
                print(
                    f"GPU {gpu.visibility_token}: {plan.model} started"
                )

            completed = []
            for logical_index, job in running.items():
                return_code = job.process.poll()
                if return_code is not None:
                    completed.append(
                        (logical_index, job, return_code)
                    )

            if not completed:
                time.sleep(0.2)
                continue

            for logical_index, job, return_code in completed:
                del running[logical_index]
                available_gpus.append(job.gpu)
                error = finish_job(job, return_code)
                if error is None:
                    successes.append(job.plan)
                else:
                    failures.append((job.plan, error))
    except KeyboardInterrupt:
        print(
            "\nInterrupted; terminating active training processes.",
            file=sys.stderr,
        )
        stop_jobs(list(running.values()))
        raise
    except Exception:
        stop_jobs(list(running.values()))
        raise

    return successes, failures


def print_summary(
    successes: Sequence[TrainingPlan],
    failures: Sequence[tuple[TrainingPlan, str]],
    skipped: Sequence[TrainingPlan],
) -> None:
    print("\nTraining summary:")
    for plan in successes:
        log = f" (log: {display_path(plan.console_log)})"
        print(f"  PASS: {plan.model}{log}")
    for plan in skipped:
        print(f"  SKIP: {plan.model}: {plan.reason}")
    for plan, reason in failures:
        log = (
            f" (log: {display_path(plan.console_log)})"
            if plan.console_log is not None
            else ""
        )
        print(f"  FAIL: {plan.model}: {reason}{log}")
    print(
        f"Totals: {len(successes)} passed, {len(skipped)} skipped, "
        f"{len(failures)} failed"
    )


def train_all(args: argparse.Namespace) -> int:
    target_epoch = args.max_epochs or DEFAULT_ALL_MAX_EPOCHS
    try:
        gpus = detect_gpus()
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print_detected_gpus(gpus)
    plans = build_training_plans(
        target_epoch,
        resume=args.resume,
        force=args.force,
    )
    if args.dry_run:
        print_dry_run(plans, gpus)
        return 1 if any(plan.action == "fail" for plan in plans) else 0

    skipped = [plan for plan in plans if plan.action == "skip"]
    failures = [
        (plan, plan.reason)
        for plan in plans
        if plan.action == "fail"
    ]
    queued = [
        plan for plan in plans if plan.action in RUNNABLE_ACTIONS
    ]

    for plan in skipped:
        print(f"SKIP: {plan.model}: {plan.reason}")
    for plan, reason in failures:
        print(f"FAIL: {plan.model}: {reason}")

    successes, child_failures = run_job_queue(queued, gpus)
    failures.extend(child_failures)
    print_summary(successes, failures, skipped)
    return 1 if failures else 0


def configure_resume(
    cfg: object,
    work_dir: Path,
    target_epoch: int,
    resume_from: Path | None,
) -> Path:
    if resume_from is None:
        checkpoint = find_resume_checkpoint(work_dir, target_epoch)
        if checkpoint is None:
            raise ValueError(
                "no valid regular epoch_N.pth checkpoint below target "
                f"epoch {target_epoch} exists"
            )
    else:
        checkpoint = Path(os.path.abspath(resume_from))
        if checkpoint.is_symlink():
            raise ValueError(
                f"{display_path(checkpoint)} is a symlink"
            )
        if checkpoint.parent.resolve() != work_dir.resolve():
            raise ValueError(
                f"{display_path(checkpoint)} is outside the selected "
                "work directory"
            )
        epoch = checkpoint_epoch(checkpoint)
        if (
            epoch is None
            or checkpoint.name != f"epoch_{epoch}.pth"
        ):
            raise ValueError(
                f"{display_path(checkpoint)} is not named exactly "
                "epoch_N.pth"
            )
        if epoch >= target_epoch:
            raise ValueError(
                f"resume epoch {epoch} is not below target epoch "
                f"{target_epoch}"
            )
        if (
            not checkpoint.is_file()
            or not is_usable_checkpoint(checkpoint)
        ):
            raise ValueError(
                f"{display_path(checkpoint)} is not a usable checkpoint"
            )
        checkpoint = checkpoint.resolve()

    cfg.load_from = str(checkpoint)
    cfg.resume = True
    return checkpoint


def train_single(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    if not config_path.is_file():
        print(f"ERROR: Config does not exist: {config_path}", file=sys.stderr)
        return 1

    from lidar_model_selection.compat.kitti_evaluator import install

    install()
    from mmdet3d.utils import register_all_modules
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmengine.utils import import_modules_from_strings

    register_all_modules(init_default_scope=True)
    try:
        cfg = Config.fromfile(str(config_path))
        if cfg.get("custom_imports"):
            import_modules_from_strings(**cfg.custom_imports)

        configured_work_dir = (
            args.work_dir
            if args.work_dir is not None
            else cfg.get("work_dir")
        )
        if configured_work_dir is None:
            raise ValueError(
                "The config has no work_dir; provide --work-dir."
            )
        work_dir = Path(os.path.abspath(str(configured_work_dir)))

        if args.max_epochs is not None:
            cfg.train_cfg.max_epochs = args.max_epochs
            val_interval = cfg.train_cfg.get("val_interval")
            if (
                isinstance(val_interval, int)
                and args.max_epochs < val_interval
            ):
                cfg.train_cfg.val_interval = args.max_epochs
        target_epoch = int(cfg.train_cfg.max_epochs)
        if target_epoch < 1:
            raise ValueError(
                "train_cfg.max_epochs must be greater than zero."
            )
        if work_dir.is_symlink():
            raise ValueError(
                f"work directory is a symlink: {work_dir}"
            )
        if work_dir.exists() and not work_dir.is_dir():
            raise ValueError(
                f"work path exists but is not a directory: {work_dir}"
            )
        target_checkpoint = valid_target_checkpoint(
            work_dir,
            target_epoch,
        )
    except (
        AttributeError,
        ImportError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(
            f"ERROR: Invalid training configuration: {error}",
            file=sys.stderr,
        )
        return 1

    cfg.launcher = "none"
    cfg.work_dir = str(work_dir)
    if target_checkpoint is not None:
        print(
            f"SKIP: {config_path.stem} already has a valid "
            f"epoch_{target_epoch}.pth checkpoint"
        )
        return 0

    try:
        if args.resume:
            resume_checkpoint = configure_resume(
                cfg,
                work_dir,
                target_epoch,
                args.resume_from,
            )
        else:
            if (
                work_dir.exists()
                and next(work_dir.iterdir(), None) is not None
            ):
                raise ValueError(
                    "nonempty experiment directory without a "
                    f"valid epoch_{target_epoch}.pth checkpoint: "
                    f"{display_path(work_dir)}; use --resume"
                )
            configured_load_from = cfg.get("load_from")
            if configured_load_from is not None:
                print(
                    "INFO: Fresh training ignores configured load_from: "
                    f"{configured_load_from}"
                )
            cfg.resume = False
            cfg.load_from = None
    except (OSError, ValueError) as error:
        operation = "resume" if args.resume else "start fresh training"
        print(
            f"ERROR: Cannot {operation} in {display_path(work_dir)}: "
            f"{error}",
            file=sys.stderr,
        )
        return 1

    if args.resume:
        resume_epoch = checkpoint_epoch(resume_checkpoint)
        print(
            f"RESUME: {config_path.stem} from "
            f"{display_path(resume_checkpoint)} "
            f"(epoch {resume_epoch}) to epoch {target_epoch}"
        )

    try:
        runner = Runner.from_cfg(cfg)
        runner.train()
    except Exception as error:
        traceback.print_exc()
        print(
            f"ERROR: Training failed for {config_path.stem}: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    try:
        error = validate_completed_training(work_dir, target_epoch)
    except (OSError, RuntimeError, ValueError) as inspection_error:
        error = f"cannot validate completed training: {inspection_error}"
    if error is not None:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_arguments(parser, args)
    try:
        if args.all_models:
            return train_all(args)
        return train_single(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
