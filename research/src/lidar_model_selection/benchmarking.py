"""Direct MMDetection3D latency benchmarking for CenterPoint models."""

from __future__ import annotations

import csv
import gc
import json
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from lidar_model_selection.checkpoints import (
    CENTERPOINT_MODELS,
    discover_checkpoint,
    is_usable_checkpoint,
)


RESEARCH_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = RESEARCH_ROOT.parent
PREDICTION_SCOPE = (
    "model.test_step: preprocessing + voxelization + network forward + "
    "box decoding + NMS/postprocessing"
)
END_TO_END_SCOPE = (
    "framework dataloader retrieval + CPU transforms + collation + "
    + PREDICTION_SCOPE
)
SUMMARY_FIELDS = (
    "model",
    "checkpoint",
    "checkpoint_selection_type",
    "success",
    "error",
    "samples",
    "prediction_p50_ms",
    "prediction_p95_ms",
    "prediction_p99_ms",
    "end_to_end_mean_ms",
    "end_to_end_p50_ms",
    "end_to_end_p95_ms",
    "end_to_end_p99_ms",
    "percentage_over_50ms",
    "meets_20hz",
    "peak_memory_allocated_mb",
    "peak_memory_reserved_mb",
    "checkpoint_size_mb",
    "gpu_name",
)


@dataclass(frozen=True)
class BenchmarkPlan:
    model: str
    config_path: Path
    checkpoint_path: Path | None
    checkpoint_selection_type: str | None
    planning_error: str | None = None


def resolve_output_dir(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (REPOSITORY_ROOT / path).resolve()


def _display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def _plan_error(plan: BenchmarkPlan) -> str | None:
    if plan.planning_error is not None:
        return plan.planning_error
    if not plan.config_path.exists():
        return f"Config does not exist: {plan.config_path}"
    if not plan.config_path.is_file():
        return f"Config is not a regular file: {plan.config_path}"
    if plan.checkpoint_path is None:
        return f"No usable checkpoint found for {plan.model}."
    if not plan.checkpoint_path.exists():
        return f"Checkpoint does not exist: {plan.checkpoint_path}"
    if not is_usable_checkpoint(plan.checkpoint_path):
        return (
            "Checkpoint is not a usable PyTorch archive: "
            f"{plan.checkpoint_path}"
        )
    return None


def build_plans(
    config_path: Path | None, checkpoint_path: Path | None, all_models: bool
) -> list[BenchmarkPlan]:
    if not all_models:
        if config_path is None or checkpoint_path is None:
            raise ValueError("CONFIG and CHECKPOINT are required.")
        resolved_config = config_path.resolve()
        resolved_checkpoint = checkpoint_path.resolve()
        plan = BenchmarkPlan(
            resolved_config.stem,
            resolved_config,
            resolved_checkpoint,
            "explicit",
        )
        error = _plan_error(plan)
        return [
            BenchmarkPlan(
                plan.model,
                plan.config_path,
                plan.checkpoint_path,
                plan.checkpoint_selection_type,
                error,
            )
        ]

    plans = []
    for model in CENTERPOINT_MODELS:
        try:
            choice = discover_checkpoint(model)
        except OSError as error:
            plans.append(
                BenchmarkPlan(
                    model.name,
                    model.config_path.resolve(),
                    None,
                    None,
                    "checkpoint discovery failed: "
                    f"{type(error).__name__}: {error}",
                )
            )
            continue
        plans.append(
            BenchmarkPlan(
                model.name,
                model.config_path.resolve(),
                choice.path if choice else None,
                choice.selection if choice else None,
            )
        )
    return plans


def _validate_cuda(gpu_index: int) -> tuple[str, str, str | None]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"GPU {gpu_index} is unavailable; expected exactly one "
            "CUDA-visible device after CUDA_VISIBLE_DEVICES selection."
        )
    torch.cuda.set_device(0)
    return (
        torch.cuda.get_device_name(0),
        str(torch.__version__),
        str(torch.version.cuda) if torch.version.cuda is not None else None,
    )


def latency_statistics(values: Sequence[float]) -> dict[str, Any]:
    samples = np.asarray(values, dtype=np.float64)
    if samples.size == 0:
        raise ValueError("No latency measurements were recorded.")
    return {
        "count": int(samples.size),
        "mean_ms": float(np.mean(samples)),
        "min_ms": float(np.min(samples)),
        "max_ms": float(np.max(samples)),
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "p99_ms": float(np.percentile(samples, 99)),
        "standard_deviation_ms": float(np.std(samples)),
    }


def load_runner(plan: BenchmarkPlan, work_dir: str) -> tuple[Any, Any]:
    from lidar_model_selection.compat.kitti_evaluator import install
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmengine.utils import import_modules_from_strings

    install()
    from mmdet3d.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(plan.config_path))
    cfg.load_from = str(plan.checkpoint_path)
    cfg.resume = False
    cfg.launcher = "none"
    cfg.work_dir = work_dir
    cfg.test_dataloader.batch_size = 1
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False
    cfg.test_dataloader.drop_last = False
    cfg.test_dataloader.sampler.shuffle = False
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    if cfg.get("visualizer") is not None:
        cfg.visualizer.name = f"benchmark_visualizer_{plan.model}"
        cfg.visualizer.vis_backends = []
    if cfg.get("default_hooks") is not None:
        cfg.default_hooks.pop("visualization", None)

    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    return runner, runner.build_test_loop(cfg.test_cfg)


def _checkpoint_size(path: Path | None) -> float | None:
    try:
        return path.stat().st_size / 1024**2 if path else None
    except OSError:
        return None


def _result(
    plan: BenchmarkPlan, gpu_index: int,
    gpu: tuple[str, str, str | None],
    warmup: int, samples: int,
    *,
    success: bool, error: str | None,
    prediction: dict[str, Any] | None = None,
    end_to_end: dict[str, Any] | None = None,
    peak_allocated: float | None = None,
    peak_reserved: float | None = None,
) -> dict[str, Any]:
    return {
        "model": plan.model,
        "config_path": _display_path(plan.config_path),
        "checkpoint_path": _display_path(plan.checkpoint_path),
        "checkpoint_selection_type": plan.checkpoint_selection_type,
        "success": success,
        "error": error,
        "gpu_name": gpu[0],
        "gpu_index": gpu_index,
        "pytorch_version": gpu[1],
        "cuda_version": gpu[2],
        "warmup_count": warmup,
        "measured_sample_count": samples,
        "batch_size": 1,
        "precision": "fp32",
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction_scope": PREDICTION_SCOPE,
        "end_to_end_scope": END_TO_END_SCOPE,
        "prediction_ms": prediction or {},
        "end_to_end_ms": end_to_end or {},
        "peak_memory_allocated_mb": peak_allocated,
        "peak_memory_reserved_mb": peak_reserved,
        "checkpoint_size_mb": _checkpoint_size(plan.checkpoint_path),
    }


def benchmark_model(
    plan: BenchmarkPlan, gpu_index: int,
    gpu: tuple[str, str, str | None],
    warmup: int, samples: int,
) -> dict[str, Any]:
    import torch

    if not plan.config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {plan.config_path}")
    if plan.checkpoint_path is None:
        raise FileNotFoundError(
            f"No usable checkpoint found for {plan.model}."
        )
    if not is_usable_checkpoint(plan.checkpoint_path):
        raise ValueError(
            f"Checkpoint is not a usable PyTorch archive: "
            f"{plan.checkpoint_path}"
        )

    runner = test_loop = model = iterator = batch = None
    prediction_times: list[float] = []
    end_to_end_times: list[float] = []
    available = 0
    requested = warmup + samples
    try:
        prefix = f"centerpoint-benchmark-{plan.model}-"
        with tempfile.TemporaryDirectory(prefix=prefix) as work_dir:
            runner, test_loop = load_runner(plan, work_dir)
            model = runner.model.eval()
            iterator = iter(test_loop.dataloader)

            try:
                with torch.inference_mode():
                    for _ in range(warmup):
                        batch = next(iterator)
                        available += 1
                        model.test_step(batch)
                        batch = None

                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                    for _ in range(samples):
                        torch.cuda.synchronize()
                        end_to_end_start = time.perf_counter()
                        batch = next(iterator)
                        available += 1

                        torch.cuda.synchronize()
                        prediction_start = time.perf_counter()
                        model.test_step(batch)
                        torch.cuda.synchronize()
                        prediction_end = time.perf_counter()

                        prediction_times.append(
                            (prediction_end - prediction_start) * 1000.0
                        )
                        end_to_end_times.append(
                            (prediction_end - end_to_end_start) * 1000.0
                        )
                        batch = None
            except StopIteration as exc:
                raise RuntimeError(
                    f"Requested {requested} batches, but the validation "
                    f"dataloader provided only {available}."
                ) from exc

            prediction = latency_statistics(prediction_times)
            end_to_end = latency_statistics(end_to_end_times)
            frames = np.asarray(end_to_end_times)
            frames_over_50ms = int(np.count_nonzero(frames > 50.0))
            end_to_end["frames_over_50ms"] = frames_over_50ms
            end_to_end["percentage_over_50ms"] = (
                frames_over_50ms / samples * 100.0
            )
            end_to_end["meets_20hz"] = end_to_end["p95_ms"] <= 50.0
            return _result(
                plan,
                gpu_index,
                gpu,
                warmup,
                samples,
                success=True,
                error=None,
                prediction=prediction,
                end_to_end=end_to_end,
                peak_allocated=torch.cuda.max_memory_allocated() / 1024**2,
                peak_reserved=torch.cuda.max_memory_reserved() / 1024**2,
            )
    finally:
        batch = iterator = model = test_loop = runner = None


def failure_result(
    plan: BenchmarkPlan, error: Exception, gpu_index: int,
    gpu: tuple[str, str, str | None],
    warmup: int, samples: int,
) -> dict[str, Any]:
    return _result(
        plan,
        gpu_index,
        gpu,
        warmup,
        samples,
        success=False,
        error=f"{type(error).__name__}: {error}",
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


def save_model_result(result: dict[str, Any], output_dir: Path) -> Path:
    path = output_dir / result["model"] / "latency.json"
    _write_json(path, result)
    return path


def _summary_row(result: dict[str, Any]) -> dict[str, Any]:
    prediction = result["prediction_ms"]
    end_to_end = result["end_to_end_ms"]
    return {
        "model": result["model"],
        "checkpoint": result["checkpoint_path"],
        "checkpoint_selection_type": result["checkpoint_selection_type"],
        "success": result["success"],
        "error": result["error"],
        "samples": result["measured_sample_count"],
        "prediction_p50_ms": prediction.get("p50_ms"),
        "prediction_p95_ms": prediction.get("p95_ms"),
        "prediction_p99_ms": prediction.get("p99_ms"),
        "end_to_end_mean_ms": end_to_end.get("mean_ms"),
        "end_to_end_p50_ms": end_to_end.get("p50_ms"),
        "end_to_end_p95_ms": end_to_end.get("p95_ms"),
        "end_to_end_p99_ms": end_to_end.get("p99_ms"),
        "percentage_over_50ms": end_to_end.get("percentage_over_50ms"),
        "meets_20hz": end_to_end.get("meets_20hz"),
        "peak_memory_allocated_mb": result["peak_memory_allocated_mb"],
        "peak_memory_reserved_mb": result["peak_memory_reserved_mb"],
        "checkpoint_size_mb": result["checkpoint_size_mb"],
        "gpu_name": result["gpu_name"],
    }


def _sort_key(row: dict[str, Any]) -> tuple[int, float, str]:
    if row["success"] and row["meets_20hz"]:
        group = 0
    elif row["success"]:
        group = 1
    else:
        group = 2
    p95 = row["end_to_end_p95_ms"]
    return group, float(p95) if p95 is not None else float("inf"), row["model"]


def save_summary(
    results: Sequence[dict[str, Any]], output_dir: Path
) -> list[dict[str, Any]]:
    rows = sorted((_summary_row(result) for result in results), key=_sort_key)
    _write_json(output_dir / "summary.json", rows)
    csv_path = output_dir / "summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=SUMMARY_FIELDS,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def print_ranking(rows: Sequence[dict[str, Any]]) -> None:
    print("\nBenchmark ranking:")
    print(
        f"{'model':<20} {'prediction p95':>15} {'e2e p95':>12} "
        f"{'20 Hz':>8} {'status':>10}"
    )
    for row in rows:
        prediction = row["prediction_p95_ms"]
        end_to_end = row["end_to_end_p95_ms"]
        prediction_text = (
            f"{prediction:.3f}" if prediction is not None else "-"
        )
        end_to_end_text = (
            f"{end_to_end:.3f}" if end_to_end is not None else "-"
        )
        meets = row["meets_20hz"]
        meets_text = (
            "yes" if meets is True else ("no" if meets is False else "-")
        )
        status = "PASS" if row["success"] else "FAIL"
        print(
            f"{row['model']:<20} {prediction_text:>15} "
            f"{end_to_end_text:>12} {meets_text:>8} {status:>10}"
        )
    hardware = rows[0]["gpu_name"] if rows else "workstation"
    print(
        f"\nThis benchmark measures the {hardware} workstation GPU and does "
        "not prove performance on the final deployment hardware."
    )


def cleanup_cuda() -> None:
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _print_dry_run(
    plans: Sequence[BenchmarkPlan], gpu_index: int, gpu_name: str,
    warmup: int, samples: int,
) -> int:
    print(f"Detected GPU {gpu_index}: {gpu_name} (visible as cuda:0)")
    print(f"Warm-up frames: {warmup}")
    print(f"Measured frames: {samples}")
    print("Model order:")
    failed = False
    for index, plan in enumerate(plans, start=1):
        print(f"  {index}. {plan.model}")
        print(f"     config: {_display_path(plan.config_path)}")
        checkpoint = _display_path(plan.checkpoint_path) or "(none)"
        print(f"     checkpoint: {checkpoint}")
        print(
            "     checkpoint selection: "
            f"{plan.checkpoint_selection_type or '(none)'}"
        )
        error = _plan_error(plan)
        if error is not None:
            print(f"     action: FAIL ({error})")
            failed = True
        else:
            print("     action: BENCHMARK")
    return 1 if failed else 0


def run_benchmark(
    *,
    config_path: Path | None, checkpoint_path: Path | None,
    all_models: bool, gpu_index: int, warmup: int, samples: int,
    output_dir: Path, dry_run: bool,
) -> int:
    plans = build_plans(config_path, checkpoint_path, all_models)
    try:
        gpu = _validate_cuda(gpu_index)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 1

    if dry_run:
        return _print_dry_run(plans, gpu_index, gpu[0], warmup, samples)

    output_dir = resolve_output_dir(output_dir)
    results = []
    for plan in plans:
        print(f"GPU {gpu_index}: {plan.model} started")
        try:
            planning_error = _plan_error(plan)
            if planning_error is not None:
                result = _result(
                    plan,
                    gpu_index,
                    gpu,
                    warmup,
                    samples,
                    success=False,
                    error=planning_error,
                )
            else:
                try:
                    result = benchmark_model(
                        plan, gpu_index, gpu, warmup, samples
                    )
                except Exception as exc:
                    traceback.print_exc()
                    result = failure_result(
                        plan, exc, gpu_index, gpu, warmup, samples
                    )

            path = output_dir / plan.model / "latency.json"
            try:
                saved_path = save_model_result(result, output_dir)
            except OSError as error:
                write_error = f"{type(error).__name__}: {error}"
                result["success"] = False
                result["error"] = (
                    f"{result['error']}; {write_error}"
                    if result["error"]
                    else write_error
                )
                print(
                    f"ERROR: {plan.model} latency JSON write failed at "
                    f"{path}: {write_error}"
                )
                saved_path = None

            results.append(result)
            if result["success"]:
                print(
                    f"GPU {gpu_index}: {plan.model} finished successfully"
                )
            else:
                print(
                    f"GPU {gpu_index}: {plan.model} failed: "
                    f"{result['error']}"
                )
            if saved_path is not None:
                print(f"Latency JSON: {_display_path(saved_path)}")
        finally:
            cleanup_cuda()

    try:
        rows = save_summary(results, output_dir)
    except OSError as error:
        print(
            f"ERROR: aggregate benchmark summary write failed in "
            f"{output_dir}: {type(error).__name__}: {error}"
        )
        return 1
    print(f"Summary JSON: {_display_path(output_dir / 'summary.json')}")
    print(f"Summary CSV: {_display_path(output_dir / 'summary.csv')}")
    print_ranking(rows)
    return 1 if any(not result["success"] for result in results) else 0
