"""One-batch CenterPoint construction, loss, backward, and prediction test."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate

from lidar_model_selection.compat.kitti_evaluator import install
from mmdet3d.registry import DATASETS, MODELS
from mmdet3d.utils import register_all_modules


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-test one CenterPoint configuration on one GPU.",
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the CenterPoint configuration.",
    )
    return parser


def iter_loss_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
        return

    if isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_loss_tensors(item)
        return

    raise TypeError(
        "Loss values must be tensors or sequences of tensors, "
        f"but got {type(value).__name__}."
    )


def first_valid_sample(dataset: Any, search_limit: int = 32) -> dict:
    limit = min(len(dataset), search_limit)

    for index in range(limit):
        sample = dataset[index]
        instances = sample["data_samples"].gt_instances_3d

        if len(instances.labels_3d) > 0:
            return sample

    raise RuntimeError(
        f"No nonempty training sample found in the first {limit} items."
    )


def validate_training_sample(sample: dict) -> None:
    points = sample["inputs"]["points"]
    instances = sample["data_samples"].gt_instances_3d
    boxes = instances.bboxes_3d.tensor
    labels = instances.labels_3d

    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(
            f"Expected points with shape (N, 4), got {tuple(points.shape)}."
        )

    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise ValueError(
            f"Expected boxes with shape (N, 7), got {tuple(boxes.shape)}."
        )

    if labels.ndim != 1 or labels.shape[0] != boxes.shape[0]:
        raise ValueError(
            "Labels must have shape (N,) and align with the boxes."
        )

    if not torch.isfinite(points).all():
        raise ValueError("Training points contain NaN or infinite values.")

    if not torch.isfinite(boxes).all():
        raise ValueError("Training boxes contain NaN or infinite values.")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test.")

    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")

    install()
    register_all_modules(init_default_scope=True)

    cfg = Config.fromfile(str(config_path))

    train_dataset = DATASETS.build(
        copy.deepcopy(cfg.train_dataloader.dataset)
    )
    train_sample = first_valid_sample(train_dataset)
    validate_training_sample(train_sample)

    model = MODELS.build(copy.deepcopy(cfg.model))
    model = model.cuda()
    model.train()

    training_batch = pseudo_collate([train_sample])
    processed_batch = model.data_preprocessor(
        training_batch,
        training=True,
    )

    losses = model(
        **processed_batch,
        mode="loss",
    )

    loss_tensors = [
        tensor.mean()
        for value in losses.values()
        for tensor in iter_loss_tensors(value)
    ]

    if not loss_tensors:
        raise RuntimeError("The model returned no loss tensors.")

    total_loss = torch.stack(loss_tensors).sum()

    if not torch.isfinite(total_loss):
        raise RuntimeError(f"Non-finite total loss: {total_loss.item()}")

    total_loss.backward()
    torch.cuda.synchronize()

    gradient_count = 0

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        gradient_count += 1

        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError("A model parameter has a non-finite gradient.")

    if gradient_count == 0:
        raise RuntimeError("Backward produced no parameter gradients.")

    print("training sample: PASS")
    print(f"loss keys: {sorted(losses)}")
    print(f"total loss: {total_loss.item():.6f}")
    print(f"finite gradient tensors: {gradient_count}")

    model.zero_grad(set_to_none=True)
    model.eval()

    val_dataset = DATASETS.build(
        copy.deepcopy(cfg.val_dataloader.dataset)
    )
    val_sample = val_dataset[0]
    validation_batch = pseudo_collate([val_sample])

    with torch.no_grad():
        processed_validation = model.data_preprocessor(
            validation_batch,
            training=False,
        )
        predictions = model(
            **processed_validation,
            mode="predict",
        )
        torch.cuda.synchronize()

    if len(predictions) != 1:
        raise RuntimeError(
            f"Expected one prediction, got {len(predictions)}."
        )

    prediction = predictions[0].pred_instances_3d
    boxes = prediction.bboxes_3d.tensor
    scores = prediction.scores_3d
    labels = prediction.labels_3d

    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise RuntimeError(
            f"Prediction boxes must have shape (N, 7), got {boxes.shape}."
        )

    if scores.ndim != 1 or labels.ndim != 1:
        raise RuntimeError("Prediction scores and labels must be vectors.")

    if not (len(boxes) == len(scores) == len(labels)):
        raise RuntimeError("Prediction boxes, scores, and labels disagree.")

    if not torch.isfinite(boxes).all():
        raise RuntimeError("Prediction boxes contain non-finite values.")

    if not torch.isfinite(scores).all():
        raise RuntimeError("Prediction scores contain non-finite values.")

    print("prediction: PASS")
    print(f"prediction boxes: {tuple(boxes.shape)}")
    print(f"prediction scores: {tuple(scores.shape)}")
    print(f"prediction labels: {tuple(labels.shape)}")
    print("CenterPoint smoke test: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())