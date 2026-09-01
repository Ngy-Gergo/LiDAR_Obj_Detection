from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from lidar_model_selection.playback.contracts import SessionCalibration
from lidar_model_selection.playback.results import DetectionFrame
from lidar_model_selection.playback.ros_messages import (
    RosMessageBuilder,
    RosMessageTypes,
)


def _pose() -> SimpleNamespace:
    return SimpleNamespace(
        position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0),
    )


class Header:
    def __init__(self) -> None:
        self.stamp = SimpleNamespace(sec=0, nanosec=0)
        self.frame_id = ""


class Point:
    def __init__(self) -> None:
        self.x = self.y = self.z = 0.0


class Detection3DArray:
    def __init__(self) -> None:
        self.header = Header()
        self.detections = []


class Detection3D:
    def __init__(self) -> None:
        self.header = Header()
        self.id = ""
        self.results = []
        self.bbox = SimpleNamespace(center=_pose(), size=SimpleNamespace(x=0.0, y=0.0, z=0.0))


class ObjectHypothesisWithPose:
    def __init__(self) -> None:
        self.hypothesis = SimpleNamespace(class_id="", score=0.0)
        self.pose = SimpleNamespace(pose=_pose())


class MarkerArray:
    def __init__(self) -> None:
        self.markers = []


class Marker:
    ADD = 0
    DELETE = 2
    DELETEALL = 3
    LINE_LIST = 5
    TEXT_VIEW_FACING = 9

    def __init__(self) -> None:
        self.header = Header()
        self.ns = ""
        self.id = 0
        self.type = 0
        self.action = 0
        self.pose = _pose()
        self.scale = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.color = SimpleNamespace(r=0.0, g=0.0, b=0.0, a=0.0)
        self.points = []
        self.text = ""


class DiagnosticArray:
    def __init__(self) -> None:
        self.header = Header()
        self.status = []


class DiagnosticStatus:
    OK = b"\x00"
    WARN = b"\x01"
    ERROR = b"\x02"

    def __init__(self) -> None:
        self.level = self.OK
        self.name = self.message = self.hardware_id = ""
        self.values = []


class KeyValue:
    def __init__(self) -> None:
        self.key = self.value = ""


class PointCloud2:
    def __init__(self) -> None:
        self.header = Header()
        self.height = self.width = 0
        self.fields = []
        self.is_bigendian = False
        self.point_step = self.row_step = 0
        self.data = b""
        self.is_dense = False


class PointField:
    FLOAT32 = 7

    def __init__(self) -> None:
        self.name = ""
        self.offset = self.datatype = self.count = 0


TYPES = RosMessageTypes(
    Header=Header,
    Point=Point,
    Detection3DArray=Detection3DArray,
    Detection3D=Detection3D,
    ObjectHypothesisWithPose=ObjectHypothesisWithPose,
    MarkerArray=MarkerArray,
    Marker=Marker,
    DiagnosticArray=DiagnosticArray,
    DiagnosticStatus=DiagnosticStatus,
    KeyValue=KeyValue,
    PointCloud2=PointCloud2,
    PointField=PointField,
)
STAMP = SimpleNamespace(sec=12, nanosec=34)


def _calibration() -> SessionCalibration:
    return SessionCalibration(
        parent_frame_id="lexus3/base_link",
        child_frame_id="lexus3/os_center",
        translation_xyz=(0.75, 0.0, 1.91),
        quaternion_xyzw=(0.0, 0.0, -1.0, 0.0),
        rotation_matrix=np.diag((-1.0, -1.0, 1.0)),
    )


def _result() -> DetectionFrame:
    boxes = np.array([[1.0, 2.0, 3.0, 4.0, 2.0, 1.0, np.pi / 2]], dtype=np.float32)
    scores = np.array([0.875], dtype=np.float32)
    labels = np.array([0], dtype=np.int64)
    for values in (boxes, scores, labels):
        values.setflags(write=False)
    return DetectionFrame(
        session_id="live:/lexus3/os_center/points",
        frame_index=7,
        timestamp_ns=12_000_000_034,
        storage_timestamp_ns=None,
        source_frame_id="lexus3/os_center",
        coordinate_frame="lidar",
        source_key="/lexus3/os_center/points[7]",
        model_alias="voxel0075",
        run_id="run",
        config_sha256="a" * 64,
        checkpoint_path="checkpoints/epoch_20.pth",
        checkpoint_sha256="b" * 64,
        checkpoint_size_bytes=1,
        source_point_count=1,
        dropped_nonfinite_count=0,
        input_point_count=1,
        in_range_point_count=1,
        detection_count=1,
        status="success",
        boxes=boxes,
        scores=scores,
        labels=labels,
        decode_ms=1.0,
        detector_ms=2.0,
        frame_processing_ms=3.0,
    )


def test_detection_array_applies_translation_and_center_conversion_once() -> None:
    builder = RosMessageBuilder(TYPES, model_alias="voxel0075", base_frame="lexus3/base_link")
    message = builder.detection_array(_result(), stamp=STAMP, calibration=_calibration())

    assert message.header.frame_id == "lexus3/base_link"
    assert (message.header.stamp.sec, message.header.stamp.nanosec) == (12, 34)
    detection = message.detections[0]
    assert detection.bbox.center.position.x == 1.75
    assert detection.bbox.center.position.y == 2.0
    assert detection.bbox.center.position.z == 5.41
    assert detection.bbox.size.x == 4.0
    assert np.isclose(detection.bbox.center.orientation.z, np.sqrt(0.5))
    assert np.isclose(detection.bbox.center.orientation.w, np.sqrt(0.5))
    assert detection.results[0].hypothesis.class_id == "Car"
    assert np.isclose(detection.results[0].hypothesis.score, 0.875)


def test_wireframe_labels_colors_and_stale_marker_deletion() -> None:
    voxel = RosMessageBuilder(TYPES, model_alias="voxel0075", base_frame="lexus3/base_link")
    pillar = RosMessageBuilder(TYPES, model_alias="pillar02", base_frame="lexus3/base_link")
    message = voxel.marker_array(
        _result(),
        stamp=STAMP,
        calibration=_calibration(),
        previous_detection_count=3,
    )

    wire, label = message.markers[:2]
    assert wire.type == Marker.LINE_LIST
    assert len(wire.points) == 24
    assert wire.ns == "voxel0075/boxes"
    assert label.type == Marker.TEXT_VIEW_FACING
    assert label.text == "voxel0075 Car 0.88"
    assert label.ns == "voxel0075/labels"
    stale = message.markers[2:]
    assert [(marker.ns, marker.id, marker.action) for marker in stale] == [
        ("voxel0075/boxes", 1, Marker.DELETE),
        ("voxel0075/labels", 1, Marker.DELETE),
        ("voxel0075/boxes", 2, Marker.DELETE),
        ("voxel0075/labels", 2, Marker.DELETE),
    ]
    assert voxel.marker_color != pillar.marker_color
    clear = voxel.clear_markers(stamp=STAMP)
    assert clear.markers[0].action == Marker.DELETEALL
    with np.testing.assert_raises_regex(ValueError, "model identity"):
        pillar.detection_array(
            _result(),
            stamp=STAMP,
            calibration=_calibration(),
        )


def test_model_cloud_filters_strict_range_and_translates_visual_copy() -> None:
    points = np.array(
        [
            [1.0, 2.0, 0.0, 0.5],
            [0.0, 2.0, 0.0, 0.2],
            [67.2, 0.0, 0.0, 0.3],
        ],
        dtype=np.float32,
    )
    before = points.copy()
    builder = RosMessageBuilder(TYPES, model_alias="pillar02", base_frame="lexus3/base_link")
    message = builder.model_point_cloud(points, stamp=STAMP, calibration=_calibration())

    assert message.width == 1
    assert message.point_step == 16
    assert [field.name for field in message.fields] == ["x", "y", "z", "reflectivity"]
    published = np.frombuffer(message.data, dtype="<f4").reshape(-1, 4)
    np.testing.assert_allclose(published, [[1.75, 2.0, 1.91, 0.5]])
    np.testing.assert_array_equal(points, before)


def test_diagnostics_preserve_byte_level_and_all_values() -> None:
    builder = RosMessageBuilder(TYPES, model_alias="voxel0075", base_frame="lexus3/base_link")
    healthy = builder.diagnostic_array(
        {"checkpoint_sha256": "b" * 64, "received_frames": 2, "failed_frames": 0, "dropped_frames": 0, "last_error": ""},
        stamp=STAMP,
    )
    assert healthy.status[0].level == b"\x00"
    assert healthy.status[0].hardware_id == "b" * 64

    warning = builder.diagnostic_array(
        {"failed_frames": 0, "dropped_frames": 1, "last_error": ""},
        stamp=STAMP,
    )
    assert warning.status[0].level == b"\x01"
    failure = builder.diagnostic_array(
        {"failed_frames": 1, "dropped_frames": 0, "last_error": "bad cloud"},
        stamp=STAMP,
    )
    assert failure.status[0].level == b"\x02"
