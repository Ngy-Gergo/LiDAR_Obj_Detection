"""Evaluate one or all CenterPoint checkpoints on one selected GPU."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import tempfile
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = REPOSITORY_ROOT / "research"
RESEARCH_SOURCE = RESEARCH_ROOT / "src"
EVALUATION_ROOT = RESEARCH_ROOT / "evaluations"
if str(RESEARCH_SOURCE) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SOURCE))

from lidar_model_selection.checkpoints import (  # noqa: E402
    CENTERPOINT_MODELS,
    checkpoint_epoch,
    discover_checkpoint,
    is_usable_checkpoint,
)


KITTI_METRIC_PREFIX = "Kitti metric/pred_instances_3d/KITTI/"
METRIC_KEYS = {
    "car_3d_ap40_easy_strict": (
        KITTI_METRIC_PREFIX + "Car_3D_AP40_easy_strict"
    ),
    "car_3d_ap40_moderate_strict": (
        KITTI_METRIC_PREFIX + "Car_3D_AP40_moderate_strict"
    ),
    "car_3d_ap40_hard_strict": (
        KITTI_METRIC_PREFIX + "Car_3D_AP40_hard_strict"
    ),
    "car_bev_ap40_easy_strict": (
        KITTI_METRIC_PREFIX + "Car_BEV_AP40_easy_strict"
    ),
    "car_bev_ap40_moderate_strict": (
        KITTI_METRIC_PREFIX + "Car_BEV_AP40_moderate_strict"
    ),
    "car_bev_ap40_hard_strict": (
        KITTI_METRIC_PREFIX + "Car_BEV_AP40_hard_strict"
    ),
}
SUMMARY_FIELDS = (
    "model",
    "config_path",
    "checkpoint_path",
    "checkpoint_selection",
    "trained_epochs",
    *METRIC_KEYS,
    "test_success",
    "error_message",
)


@dataclass(frozen=True)
class EvaluationPlan:
    model: str
    config_path: Path
    checkpoint_path: Path | None
    checkpoint_selection_type: str | None


def nonnegative_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {value!r}"
        ) from error
    if result < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one or all trained CenterPoint models."
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        help="CenterPoint config for a single-model evaluation.",
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        help="Checkpoint for a single-model evaluation.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all six CenterPoint models.",
    )
    parser.add_argument(
        "--gpu",
        type=nonnegative_integer,
        default=0,
        help="Physical CUDA GPU index (default: 0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the evaluation plan without loading models.",
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
            "single-model evaluation requires CONFIG and CHECKPOINT"
        )


def display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def validate_cuda(gpu_index: int):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for CenterPoint evaluation."
        ) from error

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"GPU {gpu_index} is unavailable; expected exactly one "
            "CUDA-visible device after CUDA_VISIBLE_DEVICES selection."
        )
    torch.cuda.set_device(0)
    return torch


def build_plans(
    config_path: Path | None,
    checkpoint_path: Path | None,
    all_models: bool,
) -> list[EvaluationPlan]:
    if not all_models:
        if config_path is None or checkpoint_path is None:
            raise ValueError("CONFIG and CHECKPOINT are required.")
        config_path = config_path.resolve()
        checkpoint_path = checkpoint_path.resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Config does not exist: {config_path}")
        if not is_usable_checkpoint(checkpoint_path):
            raise ValueError(
                "Checkpoint is not a usable PyTorch archive: "
                f"{checkpoint_path}"
            )
        return [
            EvaluationPlan(
                config_path.stem,
                config_path,
                checkpoint_path,
                "explicit",
            )
        ]

    plans = []
    for model in CENTERPOINT_MODELS:
        try:
            choice = discover_checkpoint(model)
        except OSError as error:
            print(
                f"ERROR: checkpoint discovery failed for "
                f"{model.name}: {error}",
                file=sys.stderr,
            )
            choice = None
        plans.append(
            EvaluationPlan(
                model.name,
                model.config_path.resolve(),
                choice.path if choice else None,
                choice.selection if choice else None,
            )
        )
    return plans


def print_checkpoint_warning(plan: EvaluationPlan) -> None:
    if plan.checkpoint_selection_type == "candidate":
        print(
            f"WARNING: {plan.model} is using the recorded candidate "
            f"checkpoint: {display_path(plan.checkpoint_path)}"
        )
    elif plan.checkpoint_selection_type == "latest_epoch":
        print(
            f"WARNING: {plan.model} has no best or candidate checkpoint; "
            f"using latest epoch: {display_path(plan.checkpoint_path)}"
        )


def print_dry_run(
    plans: Sequence[EvaluationPlan],
    gpu_index: int,
    gpu_name: str,
    output_root: Path,
) -> int:
    print(f"Selected GPU {gpu_index}: {gpu_name} (visible as cuda:0)")
    print("Model order:")
    failed = False
    for index, plan in enumerate(plans, start=1):
        print_checkpoint_warning(plan)
        print(f"  {index}. {plan.model}")
        print(f"     config: {display_path(plan.config_path)}")
        checkpoint = display_path(plan.checkpoint_path) or "(none)"
        selection = plan.checkpoint_selection_type or "(none)"
        output_path = output_root / plan.model / "metrics.json"
        print(f"     checkpoint: {checkpoint}")
        print(f"     checkpoint selection: {selection}")
        print(f"     output: {display_path(output_path)}")

        errors = []
        if not plan.config_path.is_file():
            errors.append("config does not exist")
        if plan.checkpoint_path is None:
            errors.append("no usable checkpoint")
        elif not is_usable_checkpoint(plan.checkpoint_path):
            errors.append("checkpoint is not usable")
        if errors:
            print(f"     action: FAIL ({'; '.join(errors)})")
            failed = True
        else:
            print("     action: RUN")
    return 1 if failed else 0


def normalize_metrics(raw_metrics: Mapping[str, object]) -> dict[str, float]:
    if not isinstance(raw_metrics, Mapping):
        raise TypeError(
            "Runner.test() did not return a metrics dictionary."
        )

    missing = [
        source_key
        for source_key in METRIC_KEYS.values()
        if source_key not in raw_metrics
    ]
    if missing:
        available = ", ".join(sorted(str(key) for key in raw_metrics))
        raise KeyError(
            "Missing required KITTI metrics: "
            + ", ".join(missing)
            + ". Available metric keys: "
            + (available or "(none)")
        )

    normalized = {}
    for canonical_name, source_key in METRIC_KEYS.items():
        value = raw_metrics[source_key]
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"Metric {source_key!r} is not a numeric scalar: {value!r}"
            )
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(
                f"Metric {source_key!r} is not finite: {value!r}"
            )
        normalized[canonical_name] = value
    return normalized


def metric_values_for_json(
    raw_metrics: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(raw_metrics, Mapping):
        raise TypeError(
            "Runner.test() did not return a metrics dictionary."
        )

    values = {}
    for name, value in raw_metrics.items():
        if hasattr(value, "item"):
            value = value.item()
        if not (
            value is None
            or isinstance(value, (bool, int, float, str))
        ):
            raise TypeError(
                f"Metric {name!r} is not a JSON scalar: {value!r}"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Metric {name!r} is not finite: {value!r}")
        values[str(name)] = value
    return values


def evaluate_model(plan: EvaluationPlan) -> dict[str, object]:
    if not plan.config_path.is_file():
        raise FileNotFoundError(
            f"Config does not exist: {plan.config_path}"
        )
    if plan.checkpoint_path is None:
        raise FileNotFoundError(
            f"No usable checkpoint found for {plan.model}."
        )
    if not is_usable_checkpoint(plan.checkpoint_path):
        raise ValueError(
            "Checkpoint is not a usable PyTorch archive: "
            f"{plan.checkpoint_path}"
        )

    from lidar_model_selection.compat.kitti_evaluator import install
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmengine.utils import import_modules_from_strings

    install()
    from mmdet3d.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    runner = None
    raw_metrics = None
    try:
        prefix = f"centerpoint-evaluation-{plan.model}-"
        with tempfile.TemporaryDirectory(prefix=prefix) as work_dir:
            cfg = Config.fromfile(str(plan.config_path))
            if cfg.get("custom_imports"):
                import_modules_from_strings(**cfg.custom_imports)
            cfg.load_from = str(plan.checkpoint_path)
            cfg.resume = False
            cfg.launcher = "none"
            cfg.work_dir = work_dir
            if cfg.get("visualizer") is not None:
                cfg.visualizer.name = (
                    f"evaluation_visualizer_{plan.model}"
                )
                cfg.visualizer.vis_backends = []
            if cfg.get("default_hooks") is not None:
                cfg.default_hooks.pop("visualization", None)

            runner = Runner.from_cfg(cfg)
            raw_metrics = runner.test()
            metrics = metric_values_for_json(raw_metrics)
            return {
                "model": plan.model,
                "config": display_path(plan.config_path),
                "checkpoint": display_path(plan.checkpoint_path),
                "checkpoint_selection": (
                    plan.checkpoint_selection_type
                ),
                "trained_epochs": checkpoint_epoch(plan.checkpoint_path),
                "test_success": True,
                "error_message": None,
                "metrics": metrics,
            }
    finally:
        raw_metrics = None
        runner = None


def failure_result(
    plan: EvaluationPlan,
    error: Exception,
) -> dict[str, object]:
    return {
        "model": plan.model,
        "config": display_path(plan.config_path),
        "checkpoint": display_path(plan.checkpoint_path),
        "checkpoint_selection": plan.checkpoint_selection_type,
        "trained_epochs": (
            checkpoint_epoch(plan.checkpoint_path)
            if plan.checkpoint_path is not None
            else None
        ),
        "test_success": False,
        "error_message": f"{type(error).__name__}: {error}",
        "metrics": {},
    }


def save_model_result(
    result: dict[str, object],
    output_root: Path,
) -> Path:
    path = output_root / str(result["model"]) / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return path


def summary_sort_key(row: Mapping[str, object]) -> tuple[object, ...]:
    if row["test_success"]:
        return (
            0,
            -float(row["car_3d_ap40_moderate_strict"]),
            row["model"],
        )
    return (1, 0.0, row["model"])


def summary_row(result: Mapping[str, object]) -> dict[str, object]:
    metrics = result["metrics"]
    if not isinstance(metrics, Mapping):
        raise TypeError("Result metrics must be a dictionary.")
    normalized_metrics = (
        normalize_metrics(metrics)
        if result["test_success"]
        else {}
    )
    return {
        "model": result["model"],
        "config_path": result["config"],
        "checkpoint_path": result["checkpoint"],
        "checkpoint_selection": result["checkpoint_selection"],
        "trained_epochs": result["trained_epochs"],
        **{
            metric: normalized_metrics.get(metric)
            for metric in METRIC_KEYS
        },
        "test_success": result["test_success"],
        "error_message": result["error_message"],
    }


def save_summary(
    rows: list[dict[str, object]],
    output_root: Path,
) -> list[dict[str, object]]:
    rows.sort(key=summary_sort_key)

    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "summary.json"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2, allow_nan=False)
        stream.write("\n")

    csv_path = output_root / "summary.csv"
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


def print_ranking(rows: Sequence[Mapping[str, object]]) -> None:
    print("\nAccuracy ranking:")
    print(
        f"{'model':<20} {'3D easy':>9} {'3D moderate':>12} "
        f"{'3D hard':>9} {'BEV moderate':>13} "
        f"{'checkpoint':>13} {'status':>8}"
    )
    for row in rows:
        if row["test_success"]:
            values = [
                f"{float(row[field]):.4f}"
                for field in (
                    "car_3d_ap40_easy_strict",
                    "car_3d_ap40_moderate_strict",
                    "car_3d_ap40_hard_strict",
                    "car_bev_ap40_moderate_strict",
                )
            ]
            status = "PASS"
        else:
            values = ["-", "-", "-", "-"]
            status = "FAIL"
        selection = row["checkpoint_selection"] or "-"
        print(
            f"{str(row['model']):<20} {values[0]:>9} {values[1]:>12} "
            f"{values[2]:>9} {values[3]:>13} "
            f"{str(selection):>13} {status:>8}"
        )
        if not row["test_success"]:
            print(f"  error: {row['error_message']}")

    print(
        "Accuracy alone does not select the runtime winner; "
        "single-GPU latency and recorded JKK testing are still required."
    )


def cleanup_cuda(torch_module) -> None:
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def run_evaluations(
    plans: Sequence[EvaluationPlan],
    gpu_index: int,
    torch_module,
    output_root: Path,
) -> int:
    results = []
    rows = []
    for plan in plans:
        print_checkpoint_warning(plan)
        print(f"GPU {gpu_index}: {plan.model} started")
        output_path = None
        try:
            try:
                result = evaluate_model(plan)
                row = summary_row(result)
            except Exception as error:
                traceback.print_exc()
                result = failure_result(plan, error)
                row = summary_row(result)

            try:
                output_path = save_model_result(result, output_root)
            except OSError as error:
                print(
                    f"ERROR: could not save {plan.model} metrics: {error}",
                    file=sys.stderr,
                )
                result = failure_result(plan, error)
                row = summary_row(result)
            results.append(result)
            rows.append(row)

            if result["test_success"]:
                print(
                    f"GPU {gpu_index}: {plan.model} "
                    "finished successfully"
                )
            else:
                print(
                    f"GPU {gpu_index}: {plan.model} failed: "
                    f"{result['error_message']}"
                )
            if output_path is not None:
                print(f"Metrics JSON: {display_path(output_path)}")
        finally:
            cleanup_cuda(torch_module)

    rows = save_summary(rows, output_root)
    print(f"Summary JSON: {display_path(output_root / 'summary.json')}")
    print(f"Summary CSV: {display_path(output_root / 'summary.csv')}")
    print_ranking(rows)
    return 1 if any(not result["test_success"] for result in results) else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_arguments(parser, args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    try:
        torch_module = validate_cuda(args.gpu)
        plans = build_plans(
            args.config,
            args.checkpoint,
            args.all,
        )
    except (
        FileNotFoundError,
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    gpu_name = torch_module.cuda.get_device_name(0)
    if args.dry_run:
        return print_dry_run(
            plans,
            args.gpu,
            gpu_name,
            EVALUATION_ROOT,
        )

    print(f"Selected GPU {args.gpu}: {gpu_name} (visible as cuda:0)")
    try:
        return run_evaluations(
            plans,
            args.gpu,
            torch_module,
            EVALUATION_ROOT,
        )
    except OSError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
