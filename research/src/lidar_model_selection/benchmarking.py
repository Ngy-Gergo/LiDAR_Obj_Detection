"""Run-bound synchronized MMDetection3D latency benchmarking."""

from __future__ import annotations

import gc
import importlib
import math
import os
import platform
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpoints import CheckpointArtifact, verify_checkpoint
from .provenance import (
    CodeProvenance,
    EnvironmentInfo,
    capture_code_provenance,
    capture_environment,
)
from .results import (
    ResultFailure,
    ResultRecord,
    binding_for_run,
    create_result,
    publish_result,
)
from .runs import Run, load_run


__all__ = (
    "DEFAULT_REPOSITORY_ROOT",
    "DEFAULT_RUNS_ROOT",
    "BENCHMARK_SCHEMA_VERSION",
    "METHODOLOGY_ID",
    "METHODOLOGY_VERSION",
    "METHODOLOGY_KEY",
    "latency_statistics",
    "benchmark_run",
)

DEFAULT_REPOSITORY_ROOT = Path(__file__).absolute().parents[3]
DEFAULT_RUNS_ROOT = DEFAULT_REPOSITORY_ROOT / "research" / "runs"

BENCHMARK_SCHEMA_VERSION = 1
METHODOLOGY_ID = "mmdet3d_prediction_e2e_sync"
METHODOLOGY_VERSION = 1
METHODOLOGY_KEY = f"{METHODOLOGY_ID}_v{METHODOLOGY_VERSION}"

_TWENTY_HZ_THRESHOLD_MS = 50.0
_CORE_PACKAGES = ("torch", "mmengine", "mmcv", "mmdet", "mmdet3d")
_PROVENANCE_SCOPES = (
    "research/src/lidar_model_selection/benchmarking.py",
    "research/src/lidar_model_selection/checkpoints.py",
    "research/src/lidar_model_selection/compat",
    "research/src/lidar_model_selection/provenance.py",
    "research/src/lidar_model_selection/results.py",
    "research/src/lidar_model_selection/runs.py",
    "research/tools/benchmark.py",
)


def _timestamp() -> datetime:
    return datetime.now(timezone.utc)


def _require_positive_integer(value: object, *, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{description} must be an integer and not a boolean")
    if value <= 0:
        raise ValueError(f"{description} must be greater than zero")
    return value


def _finite_nonnegative(value: object, *, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{description} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{description} must be finite")
    if normalized < 0.0:
        raise ValueError(f"{description} must be non-negative")
    return normalized


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    position = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def latency_statistics(values: Sequence[float]) -> dict[str, object]:
    """Return strict finite latency statistics in milliseconds.

    Percentiles use linear interpolation at ``(n - 1) * q`` and the standard
    deviation is the population value (``ddof=0``).
    """
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("latency measurements must be a sequence of numbers")
    samples = tuple(
        _finite_nonnegative(value, description="latency measurement")
        for value in values
    )
    if not samples:
        raise ValueError("at least one latency measurement is required")

    ordered = tuple(sorted(samples))
    mean = math.fsum(samples) / len(samples)
    variance = math.fsum((value - mean) ** 2 for value in samples) / len(samples)
    statistics = {
        "count": len(samples),
        "mean_ms": mean,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "p50_ms": _percentile(ordered, 50.0),
        "p95_ms": _percentile(ordered, 95.0),
        "p99_ms": _percentile(ordered, 99.0),
        "standard_deviation_ms": math.sqrt(variance),
    }
    for name, value in statistics.items():
        if name != "count":
            _finite_nonnegative(value, description=f"computed statistic {name}")
    return statistics


def _value(container: object, key: str, default: object = None) -> Any:
    if isinstance(container, Mapping):
        return container.get(key, default)
    return getattr(container, key, default)


def _set_value(container: object, key: str, value: object) -> None:
    try:
        container[key] = value  # type: ignore[index]
    except (AttributeError, TypeError):
        setattr(container, key, value)


def _load_canonical_run(run: Run | Path | str) -> Run:
    if isinstance(run, Run):
        return load_run(run.paths.root)
    if not isinstance(run, (Path, str)):
        raise TypeError("run must be a loaded Run or an explicit run directory")
    return load_run(run)


def _checkpoint_path(run: Run, artifact: CheckpointArtifact) -> Path:
    reference = Path(artifact.path)
    root = None if reference.is_absolute() else run.paths.root
    mismatches = verify_checkpoint(artifact, root=root)
    if mismatches:
        details = "; ".join(
            f"{mismatch.field}: expected {mismatch.expected!r}, "
            f"observed {mismatch.actual!r}"
            for mismatch in mismatches
        )
        raise ValueError(f"selected checkpoint identity mismatch: {details}")
    if reference.is_absolute():
        return Path(os.path.abspath(os.fspath(reference)))
    return Path(os.path.abspath(os.fspath(run.paths.root / reference)))


def _require_execution_inputs_unchanged(
    run: Run,
    checkpoint_path: Path,
) -> None:
    """Reverify the exact config and checkpoint just before runner creation."""
    current = load_run(run.paths.root)
    if current.manifest != run.manifest:
        raise ValueError("run manifest changed before benchmarking")
    selected = current.selected_checkpoint
    if selected is None:
        raise ValueError("run no longer has a selected checkpoint")
    if _checkpoint_path(current, selected) != checkpoint_path:
        raise ValueError("selected checkpoint path changed before benchmarking")


def _capture_initial_evidence() -> tuple[CodeProvenance, EnvironmentInfo]:
    return (
        capture_code_provenance(DEFAULT_REPOSITORY_ROOT, _PROVENANCE_SCOPES),
        capture_environment(
            include_packages=True,
            include_torch=False,
            package_names=_CORE_PACKAGES,
        ),
    )


def _capture_execution_environment() -> EnvironmentInfo:
    return capture_environment(
        include_packages=True,
        include_torch=True,
        package_names=_CORE_PACKAGES,
    )


def _cleanup_cuda() -> None:
    """Best-effort release without importing Torch solely for cleanup."""
    try:
        gc.collect()
        torch = sys.modules.get("torch")
        cuda = None if torch is None else getattr(torch, "cuda", None)
        empty_cache = None if cuda is None else getattr(cuda, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
    except BaseException:
        pass


def _host_identity() -> dict[str, str]:
    """Capture only stable host evidence material to E2E timing."""
    cpu_model = platform.processor().strip()
    if not cpu_model:
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    cpu_model = line.partition(":")[2].strip()
                    break
        except OSError:
            pass
    return {
        "cpu_model": " ".join(cpu_model.split()),
        "architecture": platform.machine().strip(),
        "os_class": platform.system().strip(),
    }


def _force_test_dataloader(config: object) -> None:
    dataloader = _value(config, "test_dataloader", None)
    if dataloader is None:
        raise ValueError("canonical config has no test_dataloader")
    sampler = _value(dataloader, "sampler", None)
    if sampler is None:
        raise ValueError("canonical test_dataloader has no sampler")

    _set_value(dataloader, "batch_size", 1)
    _set_value(dataloader, "num_workers", 0)
    _set_value(dataloader, "persistent_workers", False)
    _set_value(dataloader, "drop_last", False)
    _set_value(sampler, "shuffle", False)


def _validate_cuda(torch: object) -> object:
    cuda = getattr(torch, "cuda")
    if not bool(cuda.is_available()):
        raise RuntimeError("CUDA is unavailable for synchronized benchmarking")
    count = int(cuda.device_count())
    if count != 1:
        raise RuntimeError(
            "synchronized benchmarking requires exactly one CUDA-visible "
            f"device, observed {count}; select one with --gpu"
        )
    cuda.set_device(0)
    return cuda


def _next_batch(iterator: object, *, phase: str, index: int) -> object:
    try:
        return next(iterator)  # type: ignore[arg-type]
    except StopIteration as error:
        raise RuntimeError(
            f"test dataloader was exhausted during {phase} sample {index}"
        ) from error


def _model_parameter_dtypes(model: object) -> list[str]:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return []
    return sorted({str(parameter.dtype) for parameter in parameters()})


def _measurement_payload(
    *,
    prediction_times: Sequence[float],
    end_to_end_times: Sequence[float],
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
) -> dict[str, object]:
    prediction = latency_statistics(prediction_times)
    end_to_end = latency_statistics(end_to_end_times)
    over_threshold = sum(
        value > _TWENTY_HZ_THRESHOLD_MS for value in end_to_end_times
    )
    count = len(end_to_end_times)
    end_to_end.update(
        {
            "frames_over_50ms": over_threshold,
            "percentage_over_50ms": 100.0 * over_threshold / count,
            "meets_20hz": end_to_end["p95_ms"] <= _TWENTY_HZ_THRESHOLD_MS,
        }
    )
    return {
        "prediction_ms": prediction,
        "end_to_end_ms": end_to_end,
        "peak_memory": {
            "allocated_bytes": peak_allocated_bytes,
            "reserved_bytes": peak_reserved_bytes,
            "allocated_mib": peak_allocated_bytes / 1024**2,
            "reserved_mib": peak_reserved_bytes / 1024**2,
        },
    }


def _execute_mmengine(
    run: Run,
    checkpoint_path: Path,
    *,
    warmup: int,
    samples: int,
) -> dict[str, object]:
    runner: object | None = None
    test_loop: object | None = None
    iterator: object | None = None
    model: object | None = None
    batch: object | None = None
    prediction: object | None = None
    try:
        compatibility = importlib.import_module(
            "lidar_model_selection.compat.kitti_evaluator"
        )
        compatibility.install()

        torch = importlib.import_module("torch")
        cuda = _validate_cuda(torch)

        mmdet3d_utils = importlib.import_module("mmdet3d.utils")
        mmdet3d_utils.register_all_modules(init_default_scope=True)

        config_class = importlib.import_module("mmengine.config").Config
        config = config_class.fromfile(os.fspath(run.paths.config))
        _force_test_dataloader(config)

        custom_imports = _value(config, "custom_imports", None)
        if custom_imports:
            try:
                options = dict(custom_imports)
            except (TypeError, ValueError) as error:
                raise TypeError("config custom_imports must be a mapping") from error
            importer = importlib.import_module(
                "mmengine.utils"
            ).import_modules_from_strings
            importer(**options)

        prediction_times: list[float] = []
        end_to_end_times: list[float] = []
        with tempfile.TemporaryDirectory(
            prefix=f"lidar-benchmark-{run.run_id}-"
        ) as work_directory:
            _set_value(config, "load_from", os.fspath(checkpoint_path))
            _set_value(config, "resume", False)
            _set_value(config, "launcher", "none")
            _set_value(config, "work_dir", work_directory)

            _require_execution_inputs_unchanged(run, checkpoint_path)
            runner_class = importlib.import_module("mmengine.runner").Runner
            runner = runner_class.from_cfg(config)
            runner.load_or_resume()  # type: ignore[attr-defined]
            test_loop = runner.build_test_loop(  # type: ignore[attr-defined]
                _value(config, "test_cfg")
            )
            model = runner.model  # type: ignore[attr-defined]
            model.eval()  # type: ignore[attr-defined]
            parameter_dtypes = _model_parameter_dtypes(model)
            iterator = iter(test_loop.dataloader)  # type: ignore[attr-defined]

            with torch.inference_mode():
                for index in range(1, warmup + 1):
                    batch = _next_batch(iterator, phase="warm-up", index=index)
                    prediction = model.test_step(batch)  # type: ignore[attr-defined]
                    prediction = None
                    batch = None

                cuda.synchronize()
                gc.collect()
                cuda.empty_cache()
                cuda.reset_peak_memory_stats()

                for index in range(1, samples + 1):
                    cuda.synchronize()
                    end_to_end_started = time.perf_counter_ns()
                    batch = _next_batch(iterator, phase="measured", index=index)
                    cuda.synchronize()
                    prediction_started = time.perf_counter_ns()
                    prediction = model.test_step(batch)  # type: ignore[attr-defined]
                    cuda.synchronize()
                    finished = time.perf_counter_ns()

                    prediction_times.append(
                        _finite_nonnegative(
                            (finished - prediction_started) / 1_000_000.0,
                            description="prediction latency",
                        )
                    )
                    end_to_end_times.append(
                        _finite_nonnegative(
                            (finished - end_to_end_started) / 1_000_000.0,
                            description="end-to-end latency",
                        )
                    )
                    prediction = None
                    batch = None

            peak_allocated = int(cuda.max_memory_allocated())
            peak_reserved = int(cuda.max_memory_reserved())
            if peak_allocated < 0 or peak_reserved < 0:
                raise ValueError("CUDA peak memory counters must be non-negative")

            measured = _measurement_payload(
                prediction_times=prediction_times,
                end_to_end_times=end_to_end_times,
                peak_allocated_bytes=peak_allocated,
                peak_reserved_bytes=peak_reserved,
            )
            measured["hardware"] = {
                "device_type": "cuda",
                "logical_device_index": 0,
                "visible_device_count": 1,
                "device_name": str(cuda.get_device_name(0)),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "host": _host_identity(),
            }
            measured["precision"] = {
                "execution_policy": "torch_inference_mode_no_autocast",
                "inference_mode": True,
                "autocast_enabled_by_benchmark": False,
                "model_parameter_dtypes": parameter_dtypes,
            }
            return measured
    finally:
        prediction = None
        batch = None
        iterator = None
        model = None
        test_loop = None
        runner = None
        _cleanup_cuda()


def _methodology() -> dict[str, object]:
    return {
        "id": METHODOLOGY_ID,
        "version": METHODOLOGY_VERSION,
        "key": METHODOLOGY_KEY,
        "timing_scopes": {
            "prediction_ms": "model.test_step(batch)",
            "end_to_end_ms": "next(iterator) + model.test_step(batch)",
        },
        "synchronization": {
            "device": "CUDA",
            "end_to_end": (
                "synchronize before start, after next(iterator), and after "
                "model.test_step before stop"
            ),
            "prediction": (
                "start after the post-next synchronization and stop after "
                "the post-test_step synchronization"
            ),
        },
        "iterator_policy": (
            "one iterator shared by warm-up and measured samples; no reset "
            "or cycling"
        ),
        "dataloader_policy": (
            "test dataloader forced to batch_size=1, num_workers=0, "
            "persistent_workers=false, drop_last=false, and sampler "
            "shuffle=false"
        ),
        "warmup_policy": (
            "execute the requested leading samples, synchronize, collect "
            "garbage, empty the CUDA cache, then reset peak memory counters"
        ),
        "sample_policy": (
            "measure exactly the requested consecutive samples following "
            "warm-up; fail if the iterator is exhausted"
        ),
        "statistics": {
            "unit": "milliseconds",
            "percentiles": "linear interpolation at (n - 1) * q",
            "standard_deviation": "population standard deviation (ddof=0)",
            "twenty_hz": (
                "end-to-end p95 <= 50 ms; frames strictly over 50 ms are "
                "counted"
            ),
        },
    }


def _payload(
    run: Run,
    *,
    warmup: int,
    samples: int,
    checkpoint: CheckpointArtifact,
    execution: Mapping[str, object] | None,
) -> dict[str, object]:
    evidence = {} if execution is None else dict(execution)
    return {
        "kind": "benchmark",
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "methodology": _methodology(),
        "workload": {
            "semantic_partition": run.manifest.dataset.semantic_partition,
            "framework_key": run.manifest.dataset.framework_key,
            "batch_size": 1,
            "num_workers": 0,
            "persistent_workers": False,
            "drop_last": False,
            "shuffle": False,
            "warmup_count": warmup,
            "measured_sample_count": samples,
        },
        "checkpoint": {
            "size_bytes": checkpoint.size_bytes,
            "size_mib": checkpoint.size_bytes / 1024**2,
        },
        "hardware": evidence.get("hardware"),
        "precision": evidence.get("precision"),
        "prediction_ms": evidence.get("prediction_ms", {}),
        "end_to_end_ms": evidence.get("end_to_end_ms", {}),
        "peak_memory": evidence.get("peak_memory"),
    }


def benchmark_run(
    run: Run | Path | str,
    *,
    warmup: int,
    samples: int,
) -> ResultRecord:
    """Benchmark one explicit completed run and publish one fresh result."""
    warmup = _require_positive_integer(warmup, description="warmup")
    samples = _require_positive_integer(samples, description="samples")
    started_at = _timestamp()
    loaded = _load_canonical_run(run)
    binding = binding_for_run(loaded)
    selected_checkpoint = loaded.selected_checkpoint
    assert selected_checkpoint is not None

    provenance: CodeProvenance | None = None
    environment: EnvironmentInfo | None = None
    execution: Mapping[str, object] | None = None
    try:
        provenance, environment = _capture_initial_evidence()
        checkpoint_path = _checkpoint_path(loaded, selected_checkpoint)
        execution = _execute_mmengine(
            loaded,
            checkpoint_path,
            warmup=warmup,
            samples=samples,
        )
        environment = _capture_execution_environment()
    except BaseException as error:
        failed = create_result(
            result_type="benchmark",
            binding=binding,
            status="failed",
            started_at=started_at,
            finished_at=_timestamp(),
            payload=_payload(
                loaded,
                warmup=warmup,
                samples=samples,
                checkpoint=selected_checkpoint,
                execution=execution,
            ),
            provenance=provenance,
            environment=environment,
            failure=ResultFailure(
                error_type=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
            ),
        )
        publish_result(loaded, failed)
        if not isinstance(error, Exception):
            raise
        return failed

    succeeded = create_result(
        result_type="benchmark",
        binding=binding,
        status="succeeded",
        started_at=started_at,
        finished_at=_timestamp(),
        payload=_payload(
            loaded,
            warmup=warmup,
            samples=samples,
            checkpoint=selected_checkpoint,
            execution=execution,
        ),
        provenance=provenance,
        environment=environment,
        failure=None,
    )
    publish_result(loaded, succeeded)
    return succeeded
