from __future__ import annotations

import struct

import numpy as np
import pytest

from lidar_model_selection.playback.contracts import PointCloudFrame
from lidar_model_selection.playback.normalization import (
    KAPOSVAR_FEATURE_PROFILE,
    normalize_kaposvar_center_reflectivity,
    normalize_points,
)


def _payload(rows: list[tuple[float, float, float, int]]) -> bytes:
    payload = bytearray(len(rows) * 48)
    for index, (x, y, z, reflectivity) in enumerate(rows):
        offset = index * 48
        struct.pack_into("<f", payload, offset + 0, x)
        struct.pack_into("<f", payload, offset + 4, y)
        struct.pack_into("<f", payload, offset + 8, z)
        struct.pack_into("<f", payload, offset + 16, 9999.0)
        struct.pack_into("<I", payload, offset + 20, index)
        struct.pack_into("<H", payload, offset + 24, reflectivity)
        struct.pack_into("<H", payload, offset + 26, index)
        struct.pack_into("<H", payload, offset + 28, 12)
        struct.pack_into("<I", payload, offset + 32, 1000)
    return bytes(payload)


def test_offset_extraction_reflectivity_clipping_rotation_and_no_translation() -> None:
    source = _payload(
        [
            (1.0, 2.0, 3.0, 0),
            (-4.0, 5.0, -6.0, 128),
            (7.0, -8.0, 9.0, 65535),
        ]
    )
    before = bytes(source)
    rotation = np.diag((-1.0, -1.0, 1.0))

    normalized = normalize_kaposvar_center_reflectivity(
        source,
        point_count=3,
        point_step=48,
        rotation_matrix=rotation,
    )

    np.testing.assert_allclose(
        normalized.points,
        np.array(
            [
                [-1.0, -2.0, 3.0, 0.0],
                [4.0, -5.0, -6.0, 128.0 / 255.0],
                [-7.0, 8.0, 9.0, 1.0],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )
    # Translation (0.75, 0, 1.91) was deliberately not applied.
    assert normalized.points[0, 0] == -1.0
    assert normalized.points[0, 2] == 3.0
    assert normalized.points.dtype == np.float32
    assert normalized.points.flags.c_contiguous
    assert bytes(source) == before


def test_nonfinite_rows_are_dropped_exactly_while_zero_and_sparse_are_kept() -> None:
    normalized = normalize_kaposvar_center_reflectivity(
        _payload(
            [
                (0.0, 0.0, 0.0, 0),
                (float("nan"), 1.0, 2.0, 20),
                (3.0, float("inf"), 4.0, 30),
            ]
        ),
        point_count=3,
        point_step=48,
        rotation_matrix=np.eye(3),
    )
    assert normalized.source_point_count == 3
    assert normalized.dropped_nonfinite_count == 2
    assert normalized.points.shape == (1, 4)
    np.testing.assert_array_equal(normalized.points[0], np.zeros(4, dtype=np.float32))

    empty = normalize_kaposvar_center_reflectivity(
        b"",
        point_count=0,
        point_step=48,
        rotation_matrix=np.eye(3),
    )
    assert empty.points.shape == (0, 4)
    assert empty.dropped_nonfinite_count == 0


def test_profile_is_fixed_and_not_adaptive() -> None:
    with pytest.raises(ValueError, match="unsupported feature profile"):
        normalize_points(
            "raw_intensity",
            b"",
            point_count=0,
            point_step=48,
            rotation_matrix=np.eye(3),
        )
    assert KAPOSVAR_FEATURE_PROFILE == "kaposvar_center_reflectivity_v1"


def test_point_cloud_frame_defensively_publishes_immutable_c_float32() -> None:
    backing = np.arange(16, dtype=np.float32).reshape((4, 4))
    noncontiguous = backing[::2]
    frame = PointCloudFrame(
        session_id="session-full-name",
        frame_index=2,
        timestamp_ns=10,
        storage_timestamp_ns=11,
        source_frame_id="lexus3/os_center",
        coordinate_frame="lidar",
        source_key="bag.mcap:/points[2]",
        points=noncontiguous,
        source_point_count=2,
        dropped_nonfinite_count=0,
        decode_ms=1.25,
    )
    assert frame.points.flags.c_contiguous
    assert not frame.points.flags.writeable
    assert frame.normalized_point_count == 2
    assert noncontiguous.flags.writeable
    backing[:] = -1.0
    assert np.all(frame.points >= 0.0)
    with pytest.raises(ValueError, match="read-only"):
        frame.points[0, 0] = 3.0
