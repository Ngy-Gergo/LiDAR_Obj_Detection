"""Kaposvar PointCloud2 feature and coordinate normalization."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Final

import numpy as np
from numpy.typing import NDArray


KAPOSVAR_FEATURE_PROFILE: Final = "kaposvar_center_reflectivity_v1"
DETECTOR_COORDINATE_FRAME: Final = "lidar"


@dataclass(frozen=True, slots=True)
class NormalizedPointCloud:
    points: NDArray[np.float32]
    source_point_count: int
    dropped_nonfinite_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.points, np.ndarray):
            raise TypeError("points must be a numpy.ndarray")
        if self.points.dtype != np.dtype(np.float32):
            raise TypeError("points must have dtype float32")
        if self.points.ndim != 2 or self.points.shape[1] != 4:
            raise ValueError("points must have shape (N, 4)")
        if not self.points.flags.c_contiguous:
            raise ValueError("points must be C-contiguous")
        if not np.isfinite(self.points).all():
            raise ValueError("points must contain only finite values")
        if self.source_point_count < 0 or self.dropped_nonfinite_count < 0:
            raise ValueError("point counts must be nonnegative")
        if self.source_point_count != len(self.points) + self.dropped_nonfinite_count:
            raise ValueError("point counts are inconsistent")


def normalize_kaposvar_center_reflectivity(
    payload: bytes | bytearray | memoryview,
    *,
    point_count: int,
    point_step: int,
    rotation_matrix: NDArray[np.floating],
) -> NormalizedPointCloud:
    """Build detector points from offset-addressed Ouster PointCloud2 data.

    This fixed cross-sensor profile uses ``clip(reflectivity, 0, 255) / 255``.
    It is an approximation requiring visual acceptance and is not claimed to
    reproduce KITTI reflectance.  Only the recorded rotation is applied; the
    sensor-to-base translation is deliberately excluded so model coordinates
    retain a LiDAR origin.
    """

    if isinstance(point_count, bool) or not isinstance(point_count, Integral):
        raise TypeError("point_count must be an integer and not a boolean")
    if isinstance(point_step, bool) or not isinstance(point_step, Integral):
        raise TypeError("point_step must be an integer and not a boolean")
    point_count = int(point_count)
    point_step = int(point_step)
    if point_count < 0:
        raise ValueError("point_count must be nonnegative")
    if point_step < 26:
        raise ValueError("point_step is too small for the required fields")
    try:
        source = memoryview(payload)
    except TypeError as error:
        raise TypeError("payload must support the buffer protocol") from error
    if source.nbytes != point_count * point_step:
        raise ValueError("payload length does not match point_count * point_step")

    rotation = np.asarray(rotation_matrix, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation_matrix must have shape (3, 3)")
    if not np.isfinite(rotation).all():
        raise ValueError("rotation_matrix must contain only finite values")

    # Explicit offsets preserve all PointCloud2 padding and avoid ever treating
    # the 48-byte Ouster record as packed float32 x 4.
    dtype = np.dtype(
        {
            "names": ("x", "y", "z", "reflectivity"),
            "formats": ("<f4", "<f4", "<f4", "<u2"),
            "offsets": (0, 4, 8, 24),
            "itemsize": point_step,
        }
    )
    records = np.frombuffer(source, dtype=dtype, count=point_count)

    xyz = np.empty((point_count, 3), dtype=np.float32)
    xyz[:, 0] = records["x"]
    xyz[:, 1] = records["y"]
    xyz[:, 2] = records["z"]
    feature = np.clip(records["reflectivity"].astype(np.float32), 0.0, 255.0)
    feature /= np.float32(255.0)

    # Calculate with float64 calibration coefficients, then explicitly publish
    # float32.  Source buffers and views are never modified.
    # Invalid input rows are intentionally handled by the explicit finite mask
    # below.  Suppress only the arithmetic warning produced by inf * 0 here.
    with np.errstate(invalid="ignore"):
        rotated = np.asarray(xyz, dtype=np.float64) @ rotation.T
    points = np.empty((point_count, 4), dtype=np.float32, order="C")
    points[:, :3] = rotated
    points[:, 3] = feature
    finite = np.isfinite(points).all(axis=1)
    dropped = int(point_count - int(np.count_nonzero(finite)))
    if dropped:
        points = np.ascontiguousarray(points[finite], dtype=np.float32)
    else:
        points = np.ascontiguousarray(points, dtype=np.float32)
    return NormalizedPointCloud(
        points=points,
        source_point_count=point_count,
        dropped_nonfinite_count=dropped,
    )


def normalize_points(
    profile: str,
    payload: bytes | bytearray | memoryview,
    *,
    point_count: int,
    point_step: int,
    rotation_matrix: NDArray[np.floating],
) -> NormalizedPointCloud:
    """Dispatch an explicitly named, non-adaptive normalization profile."""

    if profile != KAPOSVAR_FEATURE_PROFILE:
        raise ValueError(f"unsupported feature profile: {profile!r}")
    return normalize_kaposvar_center_reflectivity(
        payload,
        point_count=point_count,
        point_step=point_step,
        rotation_matrix=rotation_matrix,
    )
