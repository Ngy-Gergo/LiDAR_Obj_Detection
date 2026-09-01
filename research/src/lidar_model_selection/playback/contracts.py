"""Lightweight, immutable contracts for recorded LiDAR playback."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Iterator, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(
            f"{field_name} must contain at least one non-whitespace character"
        )
    return value


def _nonnegative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer and not a boolean")
    result = int(value)
    if result < 0:
        raise ValueError(f"{field_name} must be greater than or equal to zero")
    return result


@dataclass(frozen=True, slots=True)
class PointCloudFrame:
    """One normalized frame, owned by exactly one recording session.

    ``points`` is defensively copied so publication cannot make a caller-owned
    source buffer read-only.  The published copy is always immutable,
    C-contiguous ``float32`` with columns ``x, y, z, normalized reflectivity``.
    """

    session_id: str
    frame_index: int
    timestamp_ns: int
    storage_timestamp_ns: int | None
    source_frame_id: str
    coordinate_frame: str
    source_key: str
    points: NDArray[np.float32]
    source_point_count: int
    dropped_nonfinite_count: int
    feature_profile: str = "kaposvar_center_reflectivity_v1"
    decode_ms: float = 0.0

    def __post_init__(self) -> None:
        _required_text(self.session_id, "session_id")
        frame_index = _nonnegative_integer(self.frame_index, "frame_index")
        timestamp_ns = _nonnegative_integer(self.timestamp_ns, "timestamp_ns")
        storage_timestamp_ns = self.storage_timestamp_ns
        if storage_timestamp_ns is not None:
            storage_timestamp_ns = _nonnegative_integer(
                storage_timestamp_ns,
                "storage_timestamp_ns",
            )
        if timestamp_ns == 0:
            raise ValueError("timestamp_ns must be greater than zero")
        if storage_timestamp_ns == 0:
            raise ValueError("storage_timestamp_ns must be greater than zero")
        _required_text(self.source_frame_id, "source_frame_id")
        _required_text(self.coordinate_frame, "coordinate_frame")
        _required_text(self.source_key, "source_key")
        _required_text(self.feature_profile, "feature_profile")

        if not isinstance(self.points, np.ndarray):
            raise TypeError("points must be a numpy.ndarray")
        if self.points.dtype != np.dtype(np.float32):
            raise TypeError("points must have dtype float32")
        if self.points.ndim != 2 or self.points.shape[1] != 4:
            raise ValueError("points must have shape (N, 4)")
        if not np.isfinite(self.points).all():
            raise ValueError("published points must contain only finite values")

        source_point_count = _nonnegative_integer(
            self.source_point_count,
            "source_point_count",
        )
        dropped_nonfinite_count = _nonnegative_integer(
            self.dropped_nonfinite_count,
            "dropped_nonfinite_count",
        )
        if source_point_count != self.points.shape[0] + dropped_nonfinite_count:
            raise ValueError(
                "source_point_count must equal normalized rows plus "
                "dropped_nonfinite_count"
            )

        if isinstance(self.decode_ms, bool) or not isinstance(self.decode_ms, Real):
            raise TypeError("decode_ms must be a real number and not a boolean")
        decode_ms = float(self.decode_ms)
        if not isfinite(decode_ms) or decode_ms < 0.0:
            raise ValueError("decode_ms must be finite and nonnegative")

        published = np.array(self.points, dtype=np.float32, order="C", copy=True)
        published.setflags(write=False)
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "timestamp_ns", timestamp_ns)
        object.__setattr__(self, "storage_timestamp_ns", storage_timestamp_ns)
        object.__setattr__(self, "source_point_count", source_point_count)
        object.__setattr__(
            self,
            "dropped_nonfinite_count",
            dropped_nonfinite_count,
        )
        object.__setattr__(self, "decode_ms", decode_ms)
        object.__setattr__(self, "points", published)

    @property
    def normalized_point_count(self) -> int:
        return int(self.points.shape[0])


@dataclass(frozen=True, slots=True)
class SessionCalibration:
    """Validated recorded sensor-to-base static calibration evidence."""

    parent_frame_id: str
    child_frame_id: str
    translation_xyz: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    rotation_matrix: NDArray[np.float64]

    def __post_init__(self) -> None:
        _required_text(self.parent_frame_id, "parent_frame_id")
        _required_text(self.child_frame_id, "child_frame_id")
        translation = tuple(float(value) for value in self.translation_xyz)
        quaternion = tuple(float(value) for value in self.quaternion_xyzw)
        if len(translation) != 3 or not all(isfinite(value) for value in translation):
            raise ValueError("translation_xyz must contain three finite values")
        if len(quaternion) != 4 or not all(isfinite(value) for value in quaternion):
            raise ValueError("quaternion_xyzw must contain four finite values")
        norm = sum(value * value for value in quaternion) ** 0.5
        if abs(norm - 1.0) > 1e-9:
            raise ValueError("quaternion_xyzw must be normalized")

        if not isinstance(self.rotation_matrix, np.ndarray):
            raise TypeError("rotation_matrix must be a numpy.ndarray")
        if self.rotation_matrix.shape != (3, 3):
            raise ValueError("rotation_matrix must have shape (3, 3)")
        rotation = np.array(
            self.rotation_matrix,
            dtype=np.float64,
            order="C",
            copy=True,
        )
        if not np.isfinite(rotation).all():
            raise ValueError("rotation_matrix must contain only finite values")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9, rtol=0.0):
            raise ValueError("rotation_matrix must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9, rtol=0.0):
            raise ValueError("rotation_matrix must be a proper rotation")
        rotation.setflags(write=False)
        object.__setattr__(self, "translation_xyz", translation)
        object.__setattr__(self, "quaternion_xyzw", quaternion)
        object.__setattr__(self, "rotation_matrix", rotation)


@dataclass(frozen=True, slots=True)
class FrameErrorEvidence:
    """Identity-rich evidence for a source failure without invented values."""

    code: str
    message: str
    session_id: str
    frame_index: int | None
    header_timestamp_ns: int | None
    storage_timestamp_ns: int | None
    source_key: str | None
    recoverable: bool
    decode_ms: float = 0.0

    def __post_init__(self) -> None:
        _required_text(self.code, "code")
        _required_text(self.message, "message")
        _required_text(self.session_id, "session_id")
        for field_name in (
            "frame_index",
            "header_timestamp_ns",
            "storage_timestamp_ns",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _nonnegative_integer(value, field_name)
        if self.header_timestamp_ns == 0:
            raise ValueError("header_timestamp_ns must be positive when present")
        if self.storage_timestamp_ns == 0:
            raise ValueError("storage_timestamp_ns must be positive when present")
        if self.source_key is not None:
            _required_text(self.source_key, "source_key")
        if not isinstance(self.recoverable, bool):
            raise TypeError("recoverable must be a boolean")
        if isinstance(self.decode_ms, bool) or not isinstance(self.decode_ms, Real):
            raise TypeError("decode_ms must be a real number and not a boolean")
        decode_ms = float(self.decode_ms)
        if not isfinite(decode_ms) or decode_ms < 0.0:
            raise ValueError("decode_ms must be finite and nonnegative")
        object.__setattr__(self, "decode_ms", decode_ms)


class FrameSourceError(RuntimeError):
    """Structured source error; recoverable instances leave iterators usable."""

    def __init__(self, evidence: FrameErrorEvidence) -> None:
        if not isinstance(evidence, FrameErrorEvidence):
            raise TypeError("evidence must be FrameErrorEvidence")
        self.evidence = evidence
        super().__init__(f"{evidence.code}: {evidence.message}")


@runtime_checkable
class RecordingSequence(Protocol):
    @property
    def session_id(self) -> str: ...

    @property
    def frame_count(self) -> int | None: ...

    def iter_frames(
        self,
        start_index: int = 0,
    ) -> Iterator[PointCloudFrame]: ...
