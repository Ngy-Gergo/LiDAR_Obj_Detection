"""Run-owned MMDetection3D inference for recorded LiDAR frames."""

from __future__ import annotations

import importlib
import os
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from time import perf_counter
from typing import Any

from ..checkpoints import CheckpointArtifact, verify_checkpoint
from ..results import ResultBinding, binding_for_run
from ..runs import Run, load_run
from .frame_source import LidarFrame
from .results import Detection, FrameResult


def _load_canonical_run(run: Run | Path | str) -> Run:
    if isinstance(run, Run):
        return load_run(run.paths.root)
    if not isinstance(run, (Path, str)):
        raise TypeError("run must be a loaded Run or an explicit run directory")
    return load_run(run)


def _selected_checkpoint_path(
    run: Run,
    artifact: CheckpointArtifact,
) -> Path:
    reference = Path(artifact.path)
    if run.manifest.origin == "native":
        if reference.is_absolute():
            raise ValueError("native selected checkpoint must be run-relative")
        root: Path | None = run.paths.root
    else:
        if not reference.is_absolute():
            raise ValueError(
                "historically imported selected checkpoint must be absolute"
            )
        root = None

    mismatches = verify_checkpoint(artifact, root=root)
    if mismatches:
        details = "; ".join(
            f"{mismatch.field}: expected {mismatch.expected!r}, "
            f"observed {mismatch.actual!r}"
            for mismatch in mismatches
        )
        raise ValueError(f"selected checkpoint identity mismatch: {details}")

    path = reference if reference.is_absolute() else run.paths.root / reference
    return Path(os.path.abspath(os.fspath(path)))


def _require_execution_inputs_unchanged(
    run: Run,
    binding: ResultBinding,
    checkpoint_path: Path,
) -> None:
    """Reverify the canonical evidence immediately before model creation."""
    current = load_run(run.paths.root)
    if current.manifest != run.manifest:
        raise ValueError("run manifest changed before detector initialization")
    if binding_for_run(current) != binding:
        raise ValueError("run binding changed before detector initialization")
    selected = current.selected_checkpoint
    if selected is None:
        raise ValueError("run no longer has a selected checkpoint")
    if _selected_checkpoint_path(current, selected) != checkpoint_path:
        raise ValueError(
            "selected checkpoint path changed before detector initialization"
        )


def _prediction_values(
    boxes: Any,
    scores: Any,
    labels: Any,
) -> tuple[list[list[Real]], list[Real], list[Integral]]:
    box_values = boxes.detach().cpu().tolist()
    score_values = scores.detach().cpu().tolist()
    label_values = labels.detach().cpu().tolist()

    for row in box_values:
        if not isinstance(row, list) or len(row) != 7:
            raise ValueError("prediction boxes must contain seven-value rows")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(value)
            for value in row
        ):
            raise ValueError("prediction boxes must contain finite real values")
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(value)
        for value in score_values
    ):
        raise ValueError("prediction scores must contain finite real values")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in label_values
    ):
        raise ValueError("prediction labels must contain integer values")
    return box_values, score_values, label_values


class Mmdet3dDetector:
    def __init__(
        self,
        run: Run | Path | str,
        device: str = "cuda:0",
        score_threshold: float = 0.0,
    ) -> None:
        if not isinstance(device, str):
            raise TypeError("device must be a string")
        if not device.strip():
            raise ValueError("device must contain at least one non-whitespace character")

        if isinstance(score_threshold, bool) or not isinstance(
            score_threshold,
            Real,
        ):
            raise TypeError("score_threshold must be a real number and not a boolean")
        if not isfinite(score_threshold):
            raise ValueError("score_threshold must be finite")

        loaded = _load_canonical_run(run)
        binding = binding_for_run(loaded)
        selected = loaded.selected_checkpoint
        assert selected is not None
        checkpoint_path = _selected_checkpoint_path(loaded, selected)

        torch = importlib.import_module("torch")
        mmdet3d_apis = importlib.import_module("mmdet3d.apis")

        _require_execution_inputs_unchanged(
            loaded,
            binding,
            checkpoint_path,
        )
        model = mmdet3d_apis.init_model(
            config=os.fspath(loaded.paths.config),
            checkpoint=os.fspath(checkpoint_path),
            device=device,
        )

        self._score_threshold = score_threshold
        self._torch = torch
        self._inference_detector = mmdet3d_apis.inference_detector
        self._model = model
        self._model_device = next(model.parameters()).device

    def detect(self, frame: LidarFrame) -> FrameResult:
        if not isinstance(frame, LidarFrame):
            raise TypeError("frame must be a LidarFrame")

        if self._model_device.type == "cuda":
            self._torch.cuda.synchronize(self._model_device)
        start_time = perf_counter()

        result, _ = self._inference_detector(
            self._model,
            os.fspath(frame.path),
        )

        if self._model_device.type == "cuda":
            self._torch.cuda.synchronize(self._model_device)
        end_time = perf_counter()

        predictions = result.pred_instances_3d
        boxes = predictions.bboxes_3d.tensor
        scores = predictions.scores_3d
        labels = predictions.labels_3d

        if boxes.ndim != 2 or boxes.shape[1] != 7:
            raise ValueError("prediction boxes must have shape (N, 7)")
        if scores.ndim != 1:
            raise ValueError("prediction scores must have shape (N,)")
        if labels.ndim != 1:
            raise ValueError("prediction labels must have shape (N,)")
        if boxes.shape[0] != scores.shape[0] or boxes.shape[0] != labels.shape[0]:
            raise ValueError("prediction box, score, and label counts must match")

        box_values, score_values, label_values = _prediction_values(
            boxes,
            scores,
            labels,
        )
        detections = tuple(
            Detection(
                box=tuple(float(value) for value in box),
                score=float(score),
                label=int(label),
            )
            for box, score, label in zip(
                box_values,
                score_values,
                label_values,
            )
            if score >= self._score_threshold
        )

        return FrameResult(
            frame_id=frame.frame_id,
            source_path=frame.path,
            detections=detections,
            inference_ms=(end_time - start_time) * 1000.0,
        )
