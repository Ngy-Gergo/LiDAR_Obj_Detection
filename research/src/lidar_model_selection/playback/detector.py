"""Run-owned MMDetection3D inference for recorded LiDAR frames."""

from __future__ import annotations

import importlib
import os
import threading
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

from ..checkpoints import CheckpointArtifact, verify_checkpoint
from ..results import ResultBinding, binding_for_run
from ..runs import Run, load_run
from .frame_source import LidarFrame
from .model_registry import (
    FinalistModelIdentity,
    finalist_range_mask,
    resolve_finalist,
)
from .normalization import DETECTOR_COORDINATE_FRAME, KAPOSVAR_FEATURE_PROFILE
from .results import (
    Detection,
    DetectionFrame,
    FrameResult,
    empty_detection_arrays,
)

if TYPE_CHECKING:
    from .contracts import PointCloudFrame


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


def _require_device(device: object) -> str:
    if not isinstance(device, str):
        raise TypeError("device must be a string")
    if not device.strip():
        raise ValueError("device must contain at least one non-whitespace character")
    return device


def _require_score_threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("score_threshold must be a real number and not a boolean")
    threshold = float(value)
    if not isfinite(threshold):
        raise ValueError("score_threshold must be finite")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("score_threshold must be in the closed interval [0, 1]")
    return threshold


def _same_finalist_inputs(
    first: FinalistModelIdentity,
    second: FinalistModelIdentity,
) -> bool:
    return (
        first.model_alias == second.model_alias
        and first.binding == second.binding
        and first.config_path == second.config_path
        and first.checkpoint_path == second.checkpoint_path
        and first.checkpoint_reference == second.checkpoint_reference
        and first.checkpoint_size_bytes == second.checkpoint_size_bytes
    )


def _cpu_numpy(value: Any, *, description: str) -> np.ndarray:
    try:
        detached = value.detach()
        cpu_value = detached.cpu()
        values = cpu_value.numpy()
    except (AttributeError, TypeError, RuntimeError) as error:
        raise ValueError(
            f"{description} could not be materialized as a CPU NumPy array"
        ) from error
    if not isinstance(values, np.ndarray):
        raise ValueError(f"{description} did not materialize as a NumPy array")
    return values


def _validated_prediction_arrays(
    result: Any,
    *,
    score_threshold: float,
    class_names: tuple[str, ...] = ("Car",),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        not isinstance(class_names, tuple)
        or not class_names
        or any(not isinstance(name, str) or not name for name in class_names)
    ):
        raise ValueError("class_names must be a non-empty tuple of strings")
    try:
        predictions = result.pred_instances_3d
        raw_boxes = predictions.bboxes_3d.tensor
        raw_scores = predictions.scores_3d
        raw_labels = predictions.labels_3d
    except AttributeError as error:
        raise ValueError("detector output is missing canonical 3D predictions") from error

    boxes = _cpu_numpy(raw_boxes, description="prediction boxes")
    scores = _cpu_numpy(raw_scores, description="prediction scores")
    labels = _cpu_numpy(raw_labels, description="prediction labels")

    if boxes.ndim != 2 or boxes.shape[1:] != (7,):
        raise ValueError("prediction boxes must have shape (N, 7)")
    if scores.ndim != 1:
        raise ValueError("prediction scores must have shape (N,)")
    if labels.ndim != 1:
        raise ValueError("prediction labels must have shape (N,)")
    if boxes.shape[0] != scores.shape[0] or boxes.shape[0] != labels.shape[0]:
        raise ValueError("prediction box, score, and label counts must match")

    if np.issubdtype(boxes.dtype, np.bool_) or not np.issubdtype(
        boxes.dtype,
        np.number,
    ):
        raise ValueError("prediction boxes must contain real numeric values")
    if np.issubdtype(boxes.dtype, np.complexfloating):
        raise ValueError("prediction boxes must contain real numeric values")
    if np.issubdtype(scores.dtype, np.bool_) or not np.issubdtype(
        scores.dtype,
        np.number,
    ):
        raise ValueError("prediction scores must contain real numeric values")
    if np.issubdtype(scores.dtype, np.complexfloating):
        raise ValueError("prediction scores must contain real numeric values")
    if np.issubdtype(labels.dtype, np.bool_) or not np.issubdtype(
        labels.dtype,
        np.integer,
    ):
        raise ValueError("prediction labels must contain integer values")

    if not np.isfinite(boxes).all():
        raise ValueError("prediction boxes must contain finite real values")
    if boxes.shape[0] and not (boxes[:, 3:6] > 0.0).all():
        raise ValueError("prediction box dimensions must be strictly positive")
    if not np.isfinite(scores).all():
        raise ValueError("prediction scores must contain finite real values")
    if scores.shape[0] and not ((scores >= 0.0) & (scores <= 1.0)).all():
        raise ValueError("prediction scores must be in the closed interval [0, 1]")
    # Boolean indexing retains upstream order, which is the required stable
    # ordering after the user-level score threshold is applied.
    selected = scores >= score_threshold
    if labels[selected].shape[0] and not (
        (labels[selected] >= 0) & (labels[selected] < len(class_names))
    ).all():
        if class_names == ("Car",):
            raise ValueError("retained finalist prediction labels must all equal class 0")
        raise ValueError("retained prediction labels are outside model class_names")
    with np.errstate(over="ignore", invalid="ignore"):
        kept_boxes = np.array(
            boxes[selected],
            dtype=np.float32,
            order="C",
            copy=True,
        )
        kept_scores = np.array(
            scores[selected],
            dtype=np.float32,
            order="C",
            copy=True,
        )
    kept_labels = np.array(
        labels[selected],
        dtype=np.int64,
        order="C",
        copy=True,
    )
    if not np.isfinite(kept_boxes).all():
        raise ValueError("float32 prediction boxes must remain finite")
    if kept_boxes.shape[0] and not (kept_boxes[:, 3:6] > 0.0).all():
        raise ValueError("float32 prediction box dimensions must remain positive")
    if not np.isfinite(kept_scores).all():
        raise ValueError("float32 prediction scores must remain finite")
    for values in (kept_boxes, kept_scores, kept_labels):
        values.setflags(write=False)
    return kept_boxes, kept_scores, kept_labels


class FinalistDetector:
    """Run-bound in-memory inference for the two protected finalists."""

    def __init__(
        self,
        model_alias: str,
        runs_root: Path | str,
        *,
        device: str = "cuda:0",
        score_threshold: float = 0.3,
    ) -> None:
        selected_device = _require_device(device)
        threshold = _require_score_threshold(score_threshold)

        identity = resolve_finalist(model_alias, runs_root)
        torch = importlib.import_module("torch")
        mmdet3d_apis = importlib.import_module("mmdet3d.apis")

        # Re-read and re-hash the canonical execution evidence immediately
        # before model construction. Heavy imports must not silently create a
        # time-of-check/time-of-use gap.
        current = resolve_finalist(model_alias, runs_root)
        if not _same_finalist_inputs(identity, current):
            raise ValueError("finalist execution inputs changed before initialization")
        model = mmdet3d_apis.init_model(
            config=os.fspath(current.config_path),
            checkpoint=os.fspath(current.checkpoint_path),
            device=selected_device,
        )

        try:
            model_device = next(model.parameters()).device
        except (AttributeError, StopIteration) as error:
            raise ValueError("initialized finalist model has no parameter device") from error

        self._identity = current
        dataset = getattr(getattr(current.run, "manifest", None), "dataset", None)
        class_names = tuple(getattr(dataset, "class_names", ("Car",)) or ())
        if not class_names:
            raise ValueError("registered finalist has no dataset class metadata")
        self._class_names = class_names
        self._score_threshold = threshold
        self._device = selected_device
        self._torch = torch
        self._inference_detector = mmdet3d_apis.inference_detector
        self._model = model
        self._model_device = model_device
        self._call_lock = threading.Lock()
        self._closed = False

    @property
    def identity(self) -> FinalistModelIdentity:
        return self._identity

    @property
    def model_alias(self) -> str:
        return self._identity.model_alias

    @property
    def run_id(self) -> str:
        return self._identity.run_id

    @property
    def config_sha256(self) -> str:
        return self._identity.config_sha256

    @property
    def checkpoint_path(self) -> Path:
        return self._identity.checkpoint_path

    @property
    def checkpoint_sha256(self) -> str:
        return self._identity.checkpoint_sha256

    @property
    def checkpoint_size_bytes(self) -> int:
        return self._identity.checkpoint_size_bytes

    @property
    def device(self) -> str:
        return self._device

    @property
    def class_names(self) -> tuple[str, ...]:
        """Canonical class metadata recorded with this selected checkpoint."""

        return self._class_names

    def _result(
        self,
        frame: PointCloudFrame,
        *,
        status: str,
        in_range_point_count: int,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        detector_ms: float,
    ) -> DetectionFrame:
        decode_ms = float(frame.decode_ms)
        return DetectionFrame(
            session_id=frame.session_id,
            frame_index=frame.frame_index,
            timestamp_ns=frame.timestamp_ns,
            storage_timestamp_ns=frame.storage_timestamp_ns,
            source_frame_id=frame.source_frame_id,
            coordinate_frame=frame.coordinate_frame,
            source_key=frame.source_key,
            model_alias=self.model_alias,
            run_id=self.run_id,
            config_sha256=self.config_sha256,
            checkpoint_path=self._identity.checkpoint_reference,
            checkpoint_sha256=self.checkpoint_sha256,
            checkpoint_size_bytes=self.checkpoint_size_bytes,
            source_point_count=frame.source_point_count,
            dropped_nonfinite_count=frame.dropped_nonfinite_count,
            input_point_count=frame.points.shape[0],
            in_range_point_count=in_range_point_count,
            detection_count=boxes.shape[0],
            status=status,
            boxes=boxes,
            scores=scores,
            labels=labels,
            decode_ms=decode_ms,
            detector_ms=detector_ms,
            frame_processing_ms=decode_ms + detector_ms,
            class_names=self.class_names,
        )

    def detect(self, frame: PointCloudFrame) -> DetectionFrame:
        """Infer one canonical immutable ``Nx4`` normalized point frame."""

        # The import is kept here so importing CLI/help remains isolated from
        # source-format modules and their optional ROS dependencies.
        from .contracts import PointCloudFrame

        if not isinstance(frame, PointCloudFrame):
            raise TypeError("frame must be a PointCloudFrame")
        points = frame.points

        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError("concurrent finalist detector calls are not supported")
        try:
            if self._closed:
                raise RuntimeError("finalist detector is closed")
            if self._model_device.type == "cuda":
                self._torch.cuda.synchronize(self._model_device)
            start_time = perf_counter()

            if frame.coordinate_frame != DETECTOR_COORDINATE_FRAME:
                raise ValueError("PointCloudFrame coordinate_frame must be 'lidar'")
            if frame.feature_profile != KAPOSVAR_FEATURE_PROFILE:
                raise ValueError(
                    "PointCloudFrame feature_profile is not compatible with finalists"
                )
            if (
                points.dtype != np.float32
                or points.ndim != 2
                or points.shape[1:] != (4,)
            ):
                raise ValueError(
                    "PointCloudFrame points must be a float32 array of shape (N, 4)"
                )
            if not points.flags.c_contiguous:
                raise ValueError("PointCloudFrame points must be C-contiguous")
            if points.flags.writeable:
                raise ValueError("PointCloudFrame points must be immutable")
            if not np.isfinite(points).all():
                raise ValueError(
                    "PointCloudFrame points must contain only finite values"
                )
            if points.shape[0] and not (
                (points[:, 3] >= 0.0) & (points[:, 3] <= 1.0)
            ).all():
                raise ValueError(
                    "PointCloudFrame normalized reflectivity must be in [0, 1]"
                )

            def completed_result(
                *,
                status: str,
                in_range_point_count: int,
                boxes: np.ndarray,
                scores: np.ndarray,
                labels: np.ndarray,
            ) -> DetectionFrame:
                # Build the immutable result before the ending timestamp so
                # detector_ms includes CPU DetectionFrame materialization.
                if self._model_device.type == "cuda":
                    self._torch.cuda.synchronize(self._model_device)
                published = self._result(
                    frame,
                    status=status,
                    in_range_point_count=in_range_point_count,
                    boxes=boxes,
                    scores=scores,
                    labels=labels,
                    detector_ms=0.0,
                )
                end_time = perf_counter()
                detector_ms = (end_time - start_time) * 1000.0
                if not isfinite(detector_ms) or detector_ms < 0.0:
                    raise RuntimeError(
                        "detector timing clock returned a non-monotonic value"
                    )
                # These internal assignments happen before publication and
                # avoid constructing an untimed second DetectionFrame.
                object.__setattr__(published, "detector_ms", detector_ms)
                object.__setattr__(
                    published,
                    "frame_processing_ms",
                    float(frame.decode_ms) + detector_ms,
                )
                return published

            if frame.source_point_count == 0:
                boxes, scores, labels = empty_detection_arrays()
                return completed_result(
                    status="empty_source",
                    in_range_point_count=0,
                    boxes=boxes,
                    scores=scores,
                    labels=labels,
                )
            if points.shape[0] == 0:
                boxes, scores, labels = empty_detection_arrays()
                return completed_result(
                    status="empty_after_nonfinite_filter",
                    in_range_point_count=0,
                    boxes=boxes,
                    scores=scores,
                    labels=labels,
                )

            in_range = finalist_range_mask(points)
            in_range_count = int(np.count_nonzero(in_range))
            if in_range_count == 0:
                boxes, scores, labels = empty_detection_arrays()
                return completed_result(
                    status="empty_after_range_filter",
                    in_range_point_count=0,
                    boxes=boxes,
                    scores=scores,
                    labels=labels,
                )

            # Pass a private, writable copy because LoadPointsFromDict may
            # normalize in place for other configs. The caller's published
            # immutable array can therefore never be mutated by the backend.
            model_points = np.array(points, dtype=np.float32, order="C", copy=True)
            inference_output = self._inference_detector(self._model, model_points)
            if not isinstance(inference_output, tuple) or len(inference_output) != 2:
                raise ValueError("inference_detector must return (result, input_data)")
            result, _ = inference_output
            boxes, scores, labels = _validated_prediction_arrays(
                result,
                score_threshold=self._score_threshold,
                class_names=self.class_names,
            )

            return completed_result(
                status="success",
                in_range_point_count=in_range_count,
                boxes=boxes,
                scores=scores,
                labels=labels,
            )
        finally:
            self._call_lock.release()

    def close(self) -> None:
        """Release the model reference after the single worker has stopped."""

        with self._call_lock:
            if self._closed:
                return
            self._closed = True
            self._model = None
