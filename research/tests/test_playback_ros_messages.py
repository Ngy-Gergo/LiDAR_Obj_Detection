from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lidar_model_selection.playback.contracts import SessionCalibration
from lidar_model_selection.playback.results import DetectionFrame
from lidar_model_selection.playback.ros_messages import (
    RosMessageBuilder,
    RosMessageTypes,
)
from lidar_model_selection.playback.tracking import (
    TrackedBox,
    TrackedFrame,
    TrackingDiagnostics,
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
    ARROW = 0
    ADD = 0
    DELETE = 2
    DELETEALL = 3
    LINE_STRIP = 4
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


def test_multiclass_ros_labels_follow_detection_metadata() -> None:
    result = _result()
    labels = np.array([1], dtype=np.int64)
    labels.setflags(write=False)
    result = result.__class__(
        **{
            field: getattr(result, field)
            for field in result.__dataclass_fields__
            if field not in {"labels", "class_names"}
        },
        labels=labels,
        class_names=("Car", "Pedestrian", "Cyclist"),
    )
    builder = RosMessageBuilder(TYPES, model_alias="voxel0075", base_frame="lexus3/base_link")
    detections = builder.detection_array(result, stamp=STAMP, calibration=_calibration())
    markers = builder.marker_array(
        result, stamp=STAMP, calibration=_calibration(), previous_detection_count=0
    )
    assert detections.detections[0].results[0].hypothesis.class_id == "Pedestrian"
    assert markers.markers[1].text == "voxel0075 Pedestrian 0.88"


def test_multiclass_alias_uses_distinct_class_marker_colors() -> None:
    result = _result()
    boxes = np.repeat(result.boxes, 3, axis=0)
    scores = np.repeat(result.scores, 3)
    labels = np.array([0, 1, 2], dtype=np.int64)
    for values in (boxes, scores, labels):
        values.setflags(write=False)
    result = result.__class__(
        **{
            field: getattr(result, field)
            for field in result.__dataclass_fields__
            if field
            not in {
                "boxes",
                "scores",
                "labels",
                "class_names",
                "model_alias",
                "detection_count",
            }
        },
        boxes=boxes,
        scores=scores,
        labels=labels,
        class_names=("Car", "Pedestrian", "Cyclist"),
        model_alias="pillar02_multiclass",
        detection_count=3,
    )
    builder = RosMessageBuilder(
        TYPES,
        model_alias="pillar02_multiclass",
        base_frame="lexus3/base_link",
    )
    detections = builder.detection_array(result, stamp=STAMP, calibration=_calibration())
    markers = builder.marker_array(
        result, stamp=STAMP, calibration=_calibration(), previous_detection_count=0
    ).markers

    assert [
        detection.results[0].hypothesis.class_id for detection in detections.detections
    ] == ["Car", "Pedestrian", "Cyclist"]
    assert [marker.text for marker in markers if marker.ns.endswith("/labels")] == [
        "pillar02_multiclass Car 0.88",
        "pillar02_multiclass Pedestrian 0.88",
        "pillar02_multiclass Cyclist 0.88",
    ]
    colors = [
        (marker.color.r, marker.color.g, marker.color.b)
        for marker in markers
        if marker.ns.endswith("/boxes")
    ]
    assert colors == [(1.0, 0.48, 0.0), (0.0, 0.78, 1.0), (0.76, 0.31, 0.96)]


def test_multiclass_tracked_labels_keep_class_and_stable_id() -> None:
    builder = RosMessageBuilder(
        TYPES,
        model_alias="pillar02_multiclass",
        base_frame="lexus3/base_link",
    )
    track = _track(12, coasting=False)
    track = track.__class__(
        **{
            field: getattr(track, field)
            for field in track.__dataclass_fields__
            if field != "label"
        },
        label=2,
    )
    frame = _tracked_frame(tracks=(track,))
    frame = frame.__class__(
        **{
            field: getattr(frame, field)
            for field in frame.__dataclass_fields__
            if field not in {"class_names", "model_alias"}
        },
        class_names=("Car", "Pedestrian", "Cyclist"),
        model_alias="pillar02_multiclass",
    )
    detections = builder.tracked_detection_array(frame, stamp=STAMP)
    markers = builder.tracked_marker_array(
        frame,
        stamp=STAMP,
        previous_track_ids=set(),
    ).markers
    labels = [marker for marker in markers if marker.ns.endswith("tracked_labels")]
    assert detections.detections[0].id == "12"
    assert detections.detections[0].results[0].hypothesis.class_id == "Cyclist"
    assert labels[0].text == "Cyclist #12 0.88 2.0 m/s fresh"
    assert (markers[0].color.r, markers[0].color.g, markers[0].color.b) == (
        0.76,
        0.31,
        0.96,
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
    middleware_warning = builder.diagnostic_array(
        {
            "failed_frames": 0,
            "dropped_frames": 0,
            "middleware_lost_frames": 2,
            "last_error_stage": "middleware",
            "last_error": "middleware: middleware_message_lost: sequence gap",
        },
        stamp=STAMP,
    )
    assert middleware_warning.status[0].level == b"\x01"
    assert middleware_warning.status[0].message == "middleware messages lost"
    failure = builder.diagnostic_array(
        {
            "failed_frames": 1,
            "dropped_frames": 0,
            "last_error_stage": "input",
            "last_error": "bad cloud",
        },
        stamp=STAMP,
    )
    assert failure.status[0].level == b"\x02"


def _tracking_diagnostics() -> TrackingDiagnostics:
    return TrackingDiagnostics(
        active_tracks=2,
        confirmed_tracks=2,
        tentative_tracks=0,
        coasting_tracks=1,
        created_tracks_total=3,
        deleted_tracks_total=1,
        created_tracks=0,
        removed_tracks=0,
        matches=1,
        misses=1,
        unmatched_detections=0,
        unmatched_tracks=1,
        reset_count=0,
        last_reset_reason=None,
        association_ms=0.05,
        update_ms=0.2,
        last_timestamp_ns=12_000_000_034,
        last_dt_seconds=0.1,
        maximum_observed_gap_seconds=0.1,
        tracked_frames_total=3,
    )


def _tracked_frame(*, tracks: tuple[TrackedBox, ...]) -> TrackedFrame:
    return TrackedFrame(
        session_id="live:/lexus3/os_center/points",
        frame_index=7,
        timestamp_ns=12_000_000_034,
        generation=0,
        coordinate_frame="lexus3/base_link",
        model_alias="voxel0075",
        run_id="run",
        checkpoint_sha256="b" * 64,
        tracks=tracks,
        diagnostics=_tracking_diagnostics(),
    )


def _track(track_id: int, *, coasting: bool) -> TrackedBox:
    return TrackedBox(
        track_id=track_id,
        label=0,
        box=(1.75, 2.0, 4.91, 4.0, 2.0, 1.0, np.pi / 2),
        score=0.875,
        velocity_xyz=(2.0, 0.0, 0.0),
        hits=3,
        missed_frames=1 if coasting else 0,
        confirmed=True,
        coasting=coasting,
        timestamp_ns=12_000_000_034,
        trail=((1.0, 2.0, 4.91), (1.75, 2.0, 4.91)),
    )


def test_tracked_messages_use_stable_ids_and_distinguish_coasting() -> None:
    builder = RosMessageBuilder(
        TYPES,
        model_alias="voxel0075",
        base_frame="lexus3/base_link",
    )
    frame = _tracked_frame(tracks=(_track(12, coasting=False), _track(13, coasting=True)))
    detections = builder.tracked_detection_array(frame, stamp=STAMP)
    assert [detection.id for detection in detections.detections] == ["12", "13"]
    assert detections.detections[0].bbox.center.position.z == pytest.approx(5.41)

    markers = builder.tracked_marker_array(
        frame,
        stamp=STAMP,
        previous_track_ids={12, 13, 99},
    ).markers
    boxes = [marker for marker in markers if marker.ns.endswith("tracked_boxes")]
    assert [(marker.id, marker.action) for marker in boxes[:2]] == [
        (12, Marker.ADD),
        (13, Marker.ADD),
    ]
    assert boxes[0].color.a == 1.0
    assert boxes[1].color.a == 0.35
    labels = [
        marker
        for marker in markers
        if marker.ns.endswith("tracked_labels") and marker.action == Marker.ADD
    ]
    assert labels[0].text == "Car #12 0.88 2.0 m/s fresh"
    assert labels[1].text == "Car #13 0.88 2.0 m/s coasting"
    stale = [marker for marker in markers if marker.id == 99]
    assert len(stale) == 4
    assert all(marker.action == Marker.DELETE for marker in stale)


def test_tracked_marker_expiration_and_deleteall_leave_no_marker_ids() -> None:
    builder = RosMessageBuilder(
        TYPES,
        model_alias="voxel0075",
        base_frame="lexus3/base_link",
    )
    empty = _tracked_frame(tracks=())
    deleted = builder.tracked_marker_array(
        empty,
        stamp=STAMP,
        previous_track_ids={12, 13},
    ).markers
    assert len(deleted) == 8
    assert {(marker.id, marker.action) for marker in deleted} == {
        (12, Marker.DELETE),
        (13, Marker.DELETE),
    }
    clear = builder.clear_tracked_markers(stamp=STAMP)
    assert clear.markers[0].action == Marker.DELETEALL


def test_tracking_diagnostic_includes_all_supplied_evidence() -> None:
    builder = RosMessageBuilder(
        TYPES,
        model_alias="voxel0075",
        base_frame="lexus3/base_link",
    )
    values = {
        "checkpoint_sha256": "b" * 64,
        "active_tracks": 2,
        "coasting_tracks": 1,
        "last_error": "",
    }
    message = builder.tracking_diagnostic_array(values, stamp=STAMP)
    status = message.status[0]
    assert status.name == "centerpoint/voxel0075/tracking"
    assert status.level == DiagnosticStatus.OK
    assert {item.key: item.value for item in status.values}[
        "coasting_tracks"
    ] == "1"
