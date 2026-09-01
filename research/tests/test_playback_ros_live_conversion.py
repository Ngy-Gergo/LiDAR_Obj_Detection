from __future__ import annotations

import struct
from types import SimpleNamespace

import numpy as np
import pytest

from lidar_model_selection.playback.contracts import FrameSourceError
from lidar_model_selection.playback.formats.ros2_mcap import (
    calibration_from_transform,
    pointcloud2_to_frame,
)


def _transform(*, translation=(0.75, 0.0, 1.91), quaternion=(0.0, 0.0, -1.0, 0.0)):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id="lexus3/base_link"),
        child_frame_id="lexus3/os_center",
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=translation[0], y=translation[1], z=translation[2]),
            rotation=SimpleNamespace(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]),
        ),
    )


def _field(name: str, offset: int, datatype: int):
    return SimpleNamespace(name=name, offset=offset, datatype=datatype, count=1)


def _cloud(rows, *, timestamp_ns=10, **changes):
    payload = bytearray(len(rows) * 48)
    for index, (x, y, z, reflectivity) in enumerate(rows):
        base = index * 48
        struct.pack_into("<fff", payload, base, x, y, z)
        struct.pack_into("<f", payload, base + 16, 999.0)
        struct.pack_into("<I", payload, base + 20, index)
        struct.pack_into("<H", payload, base + 24, reflectivity)
        struct.pack_into("<H", payload, base + 26, index)
        struct.pack_into("<H", payload, base + 28, 1)
        struct.pack_into("<I", payload, base + 32, 100)
    values = dict(
        header=SimpleNamespace(
            frame_id="lexus3/os_center",
            stamp=SimpleNamespace(sec=timestamp_ns // 1_000_000_000, nanosec=timestamp_ns % 1_000_000_000),
        ),
        height=1,
        width=len(rows),
        fields=[
            _field("x", 0, 7),
            _field("y", 4, 7),
            _field("z", 8, 7),
            _field("intensity", 16, 7),
            _field("t", 20, 6),
            _field("reflectivity", 24, 4),
            _field("ring", 26, 4),
            _field("ambient", 28, 4),
            _field("range", 32, 6),
        ],
        is_bigendian=False,
        point_step=48,
        row_step=len(payload),
        data=bytes(payload),
    )
    values.update(changes)
    return SimpleNamespace(**values)


def _convert(message):
    return pointcloud2_to_frame(
        message,
        session_id="live:/lexus3/os_center/points",
        frame_index=4,
        calibration=calibration_from_transform(_transform()),
        source_key="/lexus3/os_center/points[4]",
    )


def test_live_conversion_reuses_rotation_offsets_and_reflectivity_profile() -> None:
    frame = _convert(_cloud([(1.0, 2.0, 3.0, 128), (-4.0, 5.0, 6.0, 500)]))
    np.testing.assert_allclose(
        frame.points,
        [[-1.0, -2.0, 3.0, 128.0 / 255.0], [4.0, -5.0, 6.0, 1.0]],
        atol=1e-6,
    )
    assert frame.storage_timestamp_ns is None
    assert frame.timestamp_ns == 10
    assert frame.frame_index == 4
    assert not frame.points.flags.writeable


def test_live_empty_sparse_and_nonfinite_clouds_are_bounded() -> None:
    empty = _convert(_cloud([]))
    assert empty.points.shape == (0, 4)
    sparse = _convert(_cloud([(1.0, 0.0, 0.0, 1)]))
    assert sparse.normalized_point_count == 1
    filtered = _convert(_cloud([(float("nan"), 0.0, 0.0, 1), (1.0, float("inf"), 0.0, 2)]))
    assert filtered.normalized_point_count == 0
    assert filtered.dropped_nonfinite_count == 2


@pytest.mark.parametrize(
    "change",
    [
        {"height": 2},
        {"is_bigendian": True},
        {"point_step": 32},
        {"row_step": 1},
        {"data": b"truncated"},
    ],
)
def test_live_malformed_layout_has_structured_error(change) -> None:
    with pytest.raises(FrameSourceError) as captured:
        _convert(_cloud([(1.0, 2.0, 3.0, 4)], **change))
    assert captured.value.evidence.recoverable
    assert captured.value.evidence.frame_index == 4
    assert captured.value.evidence.storage_timestamp_ns is None


def test_live_transform_must_match_recorded_identity_and_values() -> None:
    with pytest.raises(ValueError, match="translation differs"):
        calibration_from_transform(_transform(translation=(1.0, 0.0, 1.91)))
    invalid = _transform()
    invalid.child_frame_id = "wrong"
    with pytest.raises(ValueError, match="exactly"):
        calibration_from_transform(invalid)


def test_live_cloud_rejects_empty_or_wrong_sensor_frame() -> None:
    for frame_id in ("", "another_sensor"):
        cloud = _cloud([(1.0, 2.0, 3.0, 4)])
        cloud.header.frame_id = frame_id
        with pytest.raises(FrameSourceError) as captured:
            _convert(cloud)
        assert captured.value.evidence.code == "invalid_pointcloud_schema"
