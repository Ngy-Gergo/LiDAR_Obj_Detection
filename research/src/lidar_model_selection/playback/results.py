from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from numbers import Integral, Real
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected 3D object with a box ordered as (x, y, z, dx, dy, dz, yaw)."""

    box: tuple[float, float, float, float, float, float, float]
    score: float
    label: int

    def __post_init__(self) -> None:
        if not isinstance(self.box, tuple):
            raise TypeError("box must be a tuple")
        if len(self.box) != 7:
            raise ValueError("box must contain exactly seven values")
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in self.box):
            raise TypeError("every box value must be a real number and not a boolean")

        if isinstance(self.score, bool) or not isinstance(self.score, Real):
            raise TypeError("score must be a real number and not a boolean")
        if not isfinite(self.score):
            raise ValueError("score must be finite")

        if isinstance(self.label, bool) or not isinstance(self.label, Integral):
            raise TypeError("label must be an integer and not a boolean")


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Detections and inference metadata for one LiDAR frame."""

    frame_id: str
    source_path: Path
    detections: tuple[Detection, ...]
    inference_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str):
            raise TypeError("frame_id must be a string")
        if not self.frame_id.strip():
            raise ValueError("frame_id must contain at least one non-whitespace character")

        if not isinstance(self.source_path, Path):
            raise TypeError("source_path must be a pathlib.Path")

        if not isinstance(self.detections, tuple):
            raise TypeError("detections must be a tuple")
        if any(not isinstance(detection, Detection) for detection in self.detections):
            raise TypeError("every detections member must be a Detection")

        if isinstance(self.inference_ms, bool) or not isinstance(self.inference_ms, Real):
            raise TypeError("inference_ms must be a real number and not a boolean")
        if not isfinite(self.inference_ms):
            raise ValueError("inference_ms must be finite")
        if self.inference_ms < 0:
            raise ValueError("inference_ms must be greater than or equal to zero")


def _require_text(value: object, *, description: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{description} must be a string")
    if not value.strip():
        raise ValueError(f"{description} must contain non-whitespace text")
    return value


def _require_count(value: object, *, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{description} must be an integer and not a boolean")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{description} must be greater than or equal to zero")
    return normalized


def _require_duration(value: object, *, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{description} must be a real number and not a boolean")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{description} must be finite")
    if normalized < 0.0:
        raise ValueError(f"{description} must be greater than or equal to zero")
    return normalized


@dataclass(frozen=True, slots=True)
class PlaybackErrorEvidence:
    """One structured, display-safe error attached to a playback frame."""

    phase: str
    code: str
    message: str

    def __post_init__(self) -> None:
        _require_text(self.phase, description="error phase")
        _require_text(self.code, description="error code")
        _require_text(self.message, description="error message")


def empty_detection_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return canonical immutable CPU arrays for a zero-detection result."""

    boxes = np.empty((0, 7), dtype=np.float32)
    scores = np.empty((0,), dtype=np.float32)
    labels = np.empty((0,), dtype=np.int64)
    for values in (boxes, scores, labels):
        values.setflags(write=False)
    return boxes, scores, labels


@dataclass(frozen=True, slots=True)
class DetectionFrame:
    """Run-bound detector output and timing evidence for one recorded frame.

    Arrays are required to be C-contiguous, CPU-backed NumPy arrays that are
    immutable before this value is published. Boxes use bottom-centred LiDAR
    ordering ``(x, y, z, dx, dy, dz, yaw)``.
    """

    session_id: str
    frame_index: int
    timestamp_ns: int | None
    storage_timestamp_ns: int | None
    source_frame_id: str | None
    coordinate_frame: str | None
    source_key: str | None
    model_alias: str
    run_id: str
    config_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    source_point_count: int | None
    dropped_nonfinite_count: int | None
    input_point_count: int | None
    in_range_point_count: int | None
    detection_count: int
    status: str
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    decode_ms: float
    detector_ms: float
    frame_processing_ms: float
    pacing_lag_ms: float = 0.0
    errors: tuple[PlaybackErrorEvidence, ...] = ()
    class_names: tuple[str, ...] = ("Car",)

    @property
    def normalized_point_count(self) -> int | None:
        """Number of finite, normalized points presented to the detector."""

        return self.input_point_count

    def __post_init__(self) -> None:
        for field in (
            "session_id",
            "model_alias",
            "run_id",
            "config_sha256",
            "checkpoint_path",
            "checkpoint_sha256",
            "status",
        ):
            _require_text(getattr(self, field), description=field)
        if (
            not isinstance(self.class_names, tuple)
            or not self.class_names
            or any(
                not isinstance(name, str) or not name or name.strip() != name
                for name in self.class_names
            )
            or len(set(self.class_names)) != len(self.class_names)
        ):
            raise ValueError("class_names must be a non-empty canonical tuple")
        for field in ("source_frame_id", "coordinate_frame", "source_key"):
            value = getattr(self, field)
            if value is not None:
                _require_text(value, description=field)

        _require_count(self.frame_index, description="frame_index")
        if self.timestamp_ns is not None:
            _require_count(self.timestamp_ns, description="timestamp_ns")
            if self.timestamp_ns == 0:
                raise ValueError("timestamp_ns must be positive when present")
        if self.storage_timestamp_ns is not None:
            _require_count(
                self.storage_timestamp_ns,
                description="storage_timestamp_ns",
            )
            if self.storage_timestamp_ns == 0:
                raise ValueError(
                    "storage_timestamp_ns must be positive when present"
                )
        _require_count(
            self.checkpoint_size_bytes,
            description="checkpoint_size_bytes",
        )
        if self.checkpoint_size_bytes == 0:
            raise ValueError("checkpoint_size_bytes must be strictly positive")
        for field in (
            "source_point_count",
            "dropped_nonfinite_count",
            "input_point_count",
            "in_range_point_count",
        ):
            value = getattr(self, field)
            if value is not None:
                _require_count(value, description=field)
        _require_count(self.detection_count, description="detection_count")
        if (
            self.dropped_nonfinite_count is not None
            and self.source_point_count is not None
            and self.dropped_nonfinite_count > self.source_point_count
        ):
            raise ValueError(
                "dropped_nonfinite_count must not exceed source_point_count"
            )
        if (
            self.input_point_count is not None
            and self.dropped_nonfinite_count is not None
            and self.source_point_count is not None
            and self.input_point_count + self.dropped_nonfinite_count
            != self.source_point_count
        ):
            raise ValueError(
                "input_point_count plus dropped_nonfinite_count must equal "
                "source_point_count"
            )
        if (
            self.in_range_point_count is not None
            and self.input_point_count is not None
            and self.in_range_point_count > self.input_point_count
        ):
            raise ValueError("in_range_point_count must not exceed input_point_count")

        successful_statuses = {
            "success",
            "empty_source",
            "empty_after_nonfinite_filter",
            "empty_after_range_filter",
        }
        if self.status in successful_statuses:
            required_source_values = (
                self.timestamp_ns,
                self.source_frame_id,
                self.coordinate_frame,
                self.source_key,
                self.source_point_count,
                self.dropped_nonfinite_count,
                self.input_point_count,
                self.in_range_point_count,
            )
            if any(value is None for value in required_source_values):
                raise ValueError(
                    "successful detection frames require complete source evidence"
                )

        if len(self.config_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.config_sha256
        ):
            raise ValueError("config_sha256 must be a lowercase SHA-256 digest")
        if len(self.checkpoint_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.checkpoint_sha256
        ):
            raise ValueError(
                "checkpoint_sha256 must be a lowercase SHA-256 digest"
            )

        expected_arrays = (
            ("boxes", self.boxes, np.dtype(np.float32), (self.detection_count, 7)),
            ("scores", self.scores, np.dtype(np.float32), (self.detection_count,)),
            ("labels", self.labels, np.dtype(np.int64), (self.detection_count,)),
        )
        for name, values, dtype, shape in expected_arrays:
            if not isinstance(values, np.ndarray):
                raise TypeError(f"{name} must be a NumPy array")
            if values.dtype != dtype:
                raise TypeError(f"{name} must use dtype {dtype}")
            if values.shape != shape:
                raise ValueError(f"{name} must have shape {shape!r}")
            if not values.flags.c_contiguous:
                raise ValueError(f"{name} must be C-contiguous")
            if values.flags.writeable:
                raise ValueError(f"{name} must be immutable")

        if not np.isfinite(self.boxes).all():
            raise ValueError("boxes must contain only finite values")
        if self.detection_count and not (self.boxes[:, 3:6] > 0.0).all():
            raise ValueError("box dimensions must be strictly positive")
        if not np.isfinite(self.scores).all():
            raise ValueError("scores must contain only finite values")
        if self.detection_count and not (
            (self.scores >= 0.0) & (self.scores <= 1.0)
        ).all():
            raise ValueError("scores must be in the closed interval [0, 1]")
        if self.detection_count and not (
            (self.labels >= 0) & (self.labels < len(self.class_names))
        ).all():
            raise ValueError("prediction labels must be within class_names")
        if self.status.startswith("empty_") and self.detection_count != 0:
            raise ValueError("empty frame statuses must not contain detections")

        for field in (
            "decode_ms",
            "detector_ms",
            "frame_processing_ms",
            "pacing_lag_ms",
        ):
            _require_duration(getattr(self, field), description=field)
        expected_processing = float(self.decode_ms) + float(self.detector_ms)
        if abs(float(self.frame_processing_ms) - expected_processing) > 1e-6:
            raise ValueError("frame_processing_ms must equal decode_ms + detector_ms")

        if not isinstance(self.errors, tuple):
            raise TypeError("errors must be a tuple")
        if any(not isinstance(error, PlaybackErrorEvidence) for error in self.errors):
            raise TypeError("every errors member must be PlaybackErrorEvidence")
        if self.status in successful_statuses and self.errors:
            raise ValueError("successful detection frames must not contain errors")
        if self.status not in successful_statuses and not self.errors:
            raise ValueError("non-success detection frames require error evidence")
        if self.errors and self.detection_count != 0:
            raise ValueError("frame errors must not contain fabricated detections")

    def with_processing_timing(
        self,
        *,
        frame_processing_ms: float,
        pacing_lag_ms: float,
    ) -> DetectionFrame:
        """Return this immutable result with pipeline-owned timing attached."""

        return replace(
            self,
            frame_processing_ms=frame_processing_ms,
            pacing_lag_ms=pacing_lag_ms,
        )
