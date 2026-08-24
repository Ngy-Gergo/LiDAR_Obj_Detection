"""GPU-slot discovery and independent subprocess scheduling for run IDs."""

from __future__ import annotations

import importlib
import math
import os
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runs import validate_run_id


__all__ = (
    "GpuSlot",
    "RunOutcome",
    "discover_gpu_slots",
    "build_train_command",
    "schedule_runs",
)


@dataclass(frozen=True, slots=True)
class GpuSlot:
    """One CUDA device as seen here and its inherited visibility token."""

    logical_index: int
    visibility_token: str
    name: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.logical_index, bool)
            or not isinstance(self.logical_index, int)
            or self.logical_index < 0
        ):
            raise ValueError("GPU logical_index must be a non-negative integer")
        if not isinstance(self.visibility_token, str) or not self.visibility_token:
            raise ValueError("GPU visibility_token must be a non-empty string")
        if self.visibility_token.strip() != self.visibility_token:
            raise ValueError("GPU visibility_token must not have outer whitespace")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("GPU name must contain non-whitespace text")


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """Terminal scheduler outcome for one queued run invocation."""

    run_id: str
    gpu: GpuSlot
    return_code: int | None
    error: str | None

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        if not isinstance(self.gpu, GpuSlot):
            raise TypeError("outcome gpu must be a GpuSlot")
        if self.return_code is not None and (
            isinstance(self.return_code, bool)
            or not isinstance(self.return_code, int)
        ):
            raise TypeError("outcome return_code must be an integer or None")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("outcome error must be a string or None")
        if self.return_code == 0 and self.error is not None:
            raise ValueError("successful outcome must not contain an error")
        if self.return_code != 0 and not self.error:
            raise ValueError("unsuccessful outcome must contain an error")

    @property
    def successful(self) -> bool:
        return self.return_code == 0


@dataclass(slots=True)
class _RunningJob:
    run_id: str
    gpu: GpuSlot
    process: Any


def _visibility_tokens(
    device_count: int,
    environment: Mapping[str, str],
) -> tuple[str, ...]:
    mask = environment.get("CUDA_VISIBLE_DEVICES")
    if mask is None:
        return tuple(str(index) for index in range(device_count))

    tokens = tuple(token.strip() for token in mask.split(",") if token.strip())
    if len(tokens) < device_count:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES exposes fewer tokens than Torch detected"
        )
    selected = tokens[:device_count]
    if len(selected) != len(set(selected)):
        raise RuntimeError("CUDA_VISIBLE_DEVICES contains duplicate active tokens")
    return selected


def discover_gpu_slots(
    *,
    environment: Mapping[str, str] | None = None,
) -> tuple[GpuSlot, ...]:
    """Lazily discover CUDA slots while retaining external token mapping."""
    source_environment = os.environ if environment is None else environment
    if not isinstance(source_environment, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in source_environment.items()
    ):
        raise TypeError("environment must map strings to strings")

    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError("Torch is required to discover CUDA GPUs") from exc

    available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    if not isinstance(available, bool):
        raise RuntimeError("torch.cuda.is_available() returned a non-boolean value")
    if (
        isinstance(device_count, bool)
        or not isinstance(device_count, int)
        or device_count < 0
    ):
        raise RuntimeError("torch.cuda.device_count() returned an invalid value")
    if not available or device_count == 0:
        raise RuntimeError("at least one CUDA GPU is required")

    tokens = _visibility_tokens(device_count, source_environment)
    return tuple(
        GpuSlot(
            logical_index=index,
            visibility_token=tokens[index],
            name=str(torch.cuda.get_device_name(index)),
        )
        for index in range(device_count)
    )


def build_train_command(
    train_tool: Path,
    run_id: str,
    *,
    python_executable: str | Path | None = None,
) -> tuple[str, ...]:
    """Build the explicit command used for one run-owned training child."""
    if not isinstance(train_tool, Path):
        raise TypeError("train_tool must be a pathlib.Path")
    validate_run_id(run_id)
    executable = sys.executable if python_executable is None else os.fspath(
        python_executable
    )
    if not isinstance(executable, str) or not executable:
        raise ValueError("python_executable must be a non-empty text path")
    tool = os.path.abspath(os.fspath(train_tool))
    return (executable, tool, "--run", run_id)


def _validated_run_ids(run_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(run_ids, (str, bytes)) or not isinstance(run_ids, Sequence):
        raise TypeError("run_ids must be a sequence of run ID strings")
    validated = tuple(validate_run_id(run_id) for run_id in run_ids)
    if len(validated) != len(set(validated)):
        raise ValueError("run_ids must not contain duplicates")
    return validated


def _validated_slots(gpus: Sequence[GpuSlot]) -> tuple[GpuSlot, ...]:
    if isinstance(gpus, (str, bytes)) or not isinstance(gpus, Sequence):
        raise TypeError("gpus must be a sequence of GpuSlot values")
    slots = tuple(gpus)
    if not all(isinstance(slot, GpuSlot) for slot in slots):
        raise TypeError("every GPU slot must be a GpuSlot")
    logical_indices = tuple(slot.logical_index for slot in slots)
    visibility_tokens = tuple(slot.visibility_token for slot in slots)
    if len(logical_indices) != len(set(logical_indices)):
        raise ValueError("GPU logical indices must be unique")
    if len(visibility_tokens) != len(set(visibility_tokens)):
        raise ValueError("GPU visibility tokens must be unique")
    return slots


def _validated_seconds(value: object, *, description: str, allow_zero: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{description} must be a real number")
    normalized = float(value)
    minimum_ok = normalized >= 0 if allow_zero else normalized > 0
    if not math.isfinite(normalized) or not minimum_ok:
        comparison = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{description} must be finite and {comparison}")
    return normalized


def _command_for_run(
    command_factory: Callable[[str], Sequence[str]],
    run_id: str,
) -> tuple[str, ...]:
    command = command_factory(run_id)
    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
        raise TypeError("command factory must return a sequence of strings")
    normalized = tuple(command)
    if not normalized or not all(isinstance(argument, str) for argument in normalized):
        raise TypeError("command factory must return non-empty string arguments")
    if any("\x00" in argument for argument in normalized):
        raise ValueError("child command arguments must not contain NUL")
    return normalized


def _stop_and_reap(
    jobs: Sequence[_RunningJob],
    *,
    termination_timeout: float,
) -> None:
    for job in jobs:
        try:
            if job.process.poll() is None:
                job.process.terminate()
        except BaseException:
            try:
                job.process.terminate()
            except BaseException:
                pass

    for job in jobs:
        try:
            job.process.wait(timeout=termination_timeout)
        except BaseException:
            try:
                job.process.kill()
            except BaseException:
                pass
            try:
                job.process.wait()
            except BaseException:
                pass


def schedule_runs(
    run_ids: Sequence[str],
    gpus: Sequence[GpuSlot],
    command_factory: Callable[[str], Sequence[str]],
    *,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
    poll_interval: float = 0.2,
    termination_timeout: float = 10.0,
) -> tuple[RunOutcome, ...]:
    """Run queued IDs independently, with at most one child per GPU slot."""
    queued_ids = _validated_run_ids(run_ids)
    slots = _validated_slots(gpus)
    if not callable(command_factory):
        raise TypeError("command_factory must be callable")
    if queued_ids and not slots:
        raise RuntimeError("cannot schedule training without a CUDA GPU")
    if cwd is not None and not isinstance(cwd, Path):
        raise TypeError("cwd must be a pathlib.Path or None")
    sleep_seconds = _validated_seconds(
        poll_interval,
        description="poll_interval",
        allow_zero=True,
    )
    stop_timeout = _validated_seconds(
        termination_timeout,
        description="termination_timeout",
        allow_zero=False,
    )

    base_environment = os.environ if environment is None else environment
    if not isinstance(base_environment, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in base_environment.items()
    ):
        raise TypeError("environment must map strings to strings")

    pending = deque(queued_ids)
    available = deque(slots)
    running: dict[int, _RunningJob] = {}
    outcomes: dict[str, RunOutcome] = {}

    try:
        while pending or running:
            while pending and available:
                run_id = pending.popleft()
                gpu = available.popleft()
                try:
                    command = _command_for_run(command_factory, run_id)
                    child_environment = dict(base_environment)
                    child_environment["CUDA_VISIBLE_DEVICES"] = gpu.visibility_token
                    child_environment["PYTHONUNBUFFERED"] = "1"
                    process = subprocess.Popen(
                        command,
                        cwd=None if cwd is None else os.fspath(cwd),
                        env=child_environment,
                    )
                except Exception as exc:
                    outcomes[run_id] = RunOutcome(
                        run_id=run_id,
                        gpu=gpu,
                        return_code=None,
                        error=f"could not start: {type(exc).__name__}: {exc}",
                    )
                    available.append(gpu)
                    continue
                running[gpu.logical_index] = _RunningJob(run_id, gpu, process)

            completed = []
            for logical_index, job in running.items():
                return_code = job.process.poll()
                if return_code is not None:
                    completed.append((logical_index, job, return_code))

            if not completed:
                time.sleep(sleep_seconds)
                continue

            for logical_index, job, return_code in completed:
                del running[logical_index]
                available.append(job.gpu)
                outcomes[job.run_id] = RunOutcome(
                    run_id=job.run_id,
                    gpu=job.gpu,
                    return_code=return_code,
                    error=(
                        None
                        if return_code == 0
                        else f"child exited with code {return_code}"
                    ),
                )
    except BaseException:
        _stop_and_reap(
            tuple(running.values()),
            termination_timeout=stop_timeout,
        )
        raise

    return tuple(outcomes[run_id] for run_id in queued_ids)
