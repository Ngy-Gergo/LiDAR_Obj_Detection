from math import isfinite
from numbers import Real
from pathlib import Path
from time import perf_counter

import torch
from mmdet3d.apis import inference_detector, init_model

from .frame_source import LidarFrame
from .results import Detection, FrameResult


class Mmdet3dDetector:
    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path,
        device: str = "cuda:0",
        score_threshold: float = 0.0,
    ) -> None:
        if not isinstance(config_path, Path):
            raise TypeError("config_path must be a pathlib.Path")
        if not config_path.exists():
            raise FileNotFoundError(f"config_path does not exist: {config_path}")
        if not config_path.is_file():
            raise ValueError(f"config_path must be a file: {config_path}")

        if not isinstance(checkpoint_path, Path):
            raise TypeError("checkpoint_path must be a pathlib.Path")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint_path does not exist: {checkpoint_path}")
        if not checkpoint_path.is_file():
            raise ValueError(f"checkpoint_path must be a file: {checkpoint_path}")

        if not isinstance(device, str):
            raise TypeError("device must be a string")
        if not device.strip():
            raise ValueError("device must contain at least one non-whitespace character")

        if isinstance(score_threshold, bool) or not isinstance(score_threshold, Real):
            raise TypeError("score_threshold must be a real number and not a boolean")
        if not isfinite(score_threshold):
            raise ValueError("score_threshold must be finite")

        self._score_threshold = score_threshold
        self._model = init_model(
            config=str(config_path),
            checkpoint=str(checkpoint_path),
            device=device,
        )
        self._model_device = next(self._model.parameters()).device

    def detect(self, frame: LidarFrame) -> FrameResult:
        if not isinstance(frame, LidarFrame):
            raise TypeError("frame must be a LidarFrame")

        if self._model_device.type == "cuda":
            torch.cuda.synchronize(self._model_device)
        start_time = perf_counter()

        result, _ = inference_detector(
            self._model,
            str(frame.path),
        )

        if self._model_device.type == "cuda":
            torch.cuda.synchronize(self._model_device)
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

        keep = scores >= self._score_threshold
        retained_boxes = boxes[keep].detach().cpu().tolist()
        retained_scores = scores[keep].detach().cpu().tolist()
        retained_labels = labels[keep].detach().cpu().tolist()

        detections = tuple(
            Detection(
                box=(
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                    float(box[4]),
                    float(box[5]),
                    float(box[6]),
                ),
                score=float(score),
                label=int(label),
            )
            for box, score, label in zip(
                retained_boxes,
                retained_scores,
                retained_labels,
            )
        )

        return FrameResult(
            frame_id=frame.frame_id,
            source_path=frame.path,
            detections=detections,
            inference_ms=(end_time - start_time) * 1000.0,
        )
