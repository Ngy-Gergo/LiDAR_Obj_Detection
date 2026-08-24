"""Load one CenterPoint checkpoint and validate one prediction batch."""

from __future__ import annotations

import argparse
import gc
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_SOURCE = REPOSITORY_ROOT / "research/src"
if str(RESEARCH_SOURCE) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SOURCE))

from lidar_model_selection.checkpoints import (  # noqa: E402
    is_usable_checkpoint,
)


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
        description="Smoke-test one CenterPoint checkpoint on one GPU.",
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the CenterPoint configuration.",
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to the checkpoint to load.",
    )
    parser.add_argument(
        "--gpu",
        type=nonnegative_integer,
        default=0,
        help="Physical CUDA GPU index (default: 0).",
    )
    return parser


def validate_inputs(config: Path, checkpoint: Path) -> tuple[Path, Path]:
    config_path = config.resolve()
    checkpoint_path = checkpoint.resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config is not a regular file: {config_path}"
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )
    if not is_usable_checkpoint(checkpoint_path):
        raise ValueError(
            "Checkpoint is not a usable PyTorch archive: "
            f"{checkpoint_path}"
        )
    return config_path, checkpoint_path


def validate_prediction(predictions: Any) -> tuple[Any, Any, Any]:
    import torch

    if not isinstance(predictions, (list, tuple)):
        raise RuntimeError(
            "model.test_step must return a prediction sequence."
        )
    if len(predictions) != 1:
        raise RuntimeError(
            f"Expected one prediction, got {len(predictions)}."
        )

    data_sample = predictions[0]
    instances = getattr(data_sample, "pred_instances_3d", None)
    if instances is None:
        raise RuntimeError("Prediction has no pred_instances_3d output.")

    boxes_3d = getattr(instances, "bboxes_3d", None)
    boxes = getattr(boxes_3d, "tensor", None)
    scores = getattr(instances, "scores_3d", None)
    labels = getattr(instances, "labels_3d", None)
    if boxes is None or scores is None or labels is None:
        raise RuntimeError(
            "Prediction must contain 3D boxes, scores, and labels."
        )

    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise RuntimeError(
            "Prediction boxes must have exactly seven parameters and no "
            f"velocity branch; got shape {tuple(boxes.shape)}."
        )
    if scores.ndim != 1 or labels.ndim != 1:
        raise RuntimeError("Prediction scores and labels must be vectors.")
    if not (len(boxes) == len(scores) == len(labels)):
        raise RuntimeError("Prediction boxes, scores, and labels disagree.")
    if not bool(torch.isfinite(boxes).all()):
        raise RuntimeError("Prediction boxes contain non-finite values.")
    if not bool(torch.isfinite(scores).all()):
        raise RuntimeError("Prediction scores contain non-finite values.")
    if not bool(torch.isfinite(labels).all()):
        raise RuntimeError("Prediction labels contain non-finite values.")
    return boxes, scores, labels


def cleanup_cuda(torch_module: Any) -> None:
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def run_smoke_test(
    config_path: Path,
    checkpoint_path: Path,
    gpu_index: int,
) -> None:
    import torch

    runner = model = iterator = batch = predictions = None
    boxes = scores = labels = None
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                f"GPU {gpu_index} is unavailable; expected exactly one "
                "CUDA-visible device after CUDA_VISIBLE_DEVICES selection."
            )
        torch.cuda.set_device(0)
        from lidar_model_selection.compat.kitti_evaluator import install
        from mmdet3d.utils import register_all_modules
        from mmengine.config import Config
        from mmengine.runner import Runner
        from mmengine.utils import import_modules_from_strings

        install()
        register_all_modules(init_default_scope=True)
        cfg = Config.fromfile(str(config_path))
        if cfg.get("custom_imports"):
            import_modules_from_strings(**cfg.custom_imports)

        with tempfile.TemporaryDirectory(
            prefix="centerpoint-smoke-test-"
        ) as work_dir:
            cfg.load_from = str(checkpoint_path)
            cfg.resume = False
            cfg.launcher = "none"
            cfg.work_dir = work_dir
            cfg.test_dataloader.batch_size = 1
            cfg.test_dataloader.num_workers = 0
            cfg.test_dataloader.persistent_workers = False
            cfg.test_dataloader.drop_last = False
            cfg.test_dataloader.sampler.shuffle = False

            runner = Runner.from_cfg(cfg)
            runner.load_or_resume()
            model = runner.model.eval()
            iterator = iter(runner.test_dataloader)
            try:
                batch = next(iterator)
            except StopIteration as error:
                raise RuntimeError(
                    "The validation dataloader returned no batches."
                ) from error

            with torch.inference_mode():
                predictions = model.test_step(batch)
                torch.cuda.synchronize()

            boxes, scores, labels = validate_prediction(predictions)
            print("prediction: PASS")
            print(f"prediction boxes: {tuple(boxes.shape)}")
            print(f"prediction scores: {tuple(scores.shape)}")
            print(f"prediction labels: {tuple(labels.shape)}")
    finally:
        boxes = scores = labels = predictions = None
        batch = iterator = model = runner = None
        cleanup_cuda(torch)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_path, checkpoint_path = validate_inputs(
            args.config,
            args.checkpoint,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    try:
        run_smoke_test(config_path, checkpoint_path, args.gpu)
    except Exception as error:
        traceback.print_exc()
        print(
            f"ERROR: CenterPoint smoke test failed: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    print("CenterPoint smoke test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
