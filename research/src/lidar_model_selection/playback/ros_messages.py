"""ROS-message construction without importing ROS at module import time."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, sin
from typing import Mapping

import numpy as np

from .box_geometry import (
    bottom_to_center,
    box_corners_3d,
    boxes_to_base_frame,
    sensor_translation_to_base_frame,
)
from .contracts import SessionCalibration
from .model_registry import finalist_aliases, finalist_range_mask
from .results import DetectionFrame
from .tracking import TrackedBox, TrackedFrame


_BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)

_MARKER_COLORS = {
    "voxel0075": (0.0, 0.78, 1.0),
    "pillar02": (1.0, 0.48, 0.0),
    "pillar02_multiclass": (1.0, 0.48, 0.0),
}

_MULTICLASS_MARKER_COLORS = {
    "Car": (1.0, 0.48, 0.0),
    "Pedestrian": (0.0, 0.78, 1.0),
    "Cyclist": (0.76, 0.31, 0.96),
}


@dataclass(frozen=True, slots=True)
class RosMessageTypes:
    """Lazily supplied standard ROS message classes used by the builder."""

    Header: type
    Point: type
    Detection3DArray: type
    Detection3D: type
    ObjectHypothesisWithPose: type
    MarkerArray: type
    Marker: type
    DiagnosticArray: type
    DiagnosticStatus: type
    KeyValue: type
    PointCloud2: type
    PointField: type


def _copied_header(types: RosMessageTypes, stamp: object, frame_id: str) -> object:
    header = types.Header()
    header.stamp.sec = int(stamp.sec)
    header.stamp.nanosec = int(stamp.nanosec)
    header.frame_id = frame_id
    return header


def _class_name(class_names: tuple[str, ...], label: int) -> str:
    """Resolve a prediction label through checkpoint-recorded metadata."""

    if label < 0 or label >= len(class_names):
        raise ValueError("prediction label is outside recorded class_names")
    return class_names[label]


class RosMessageBuilder:
    """Build standard detection, marker, diagnostic, and cloud messages."""

    def __init__(
        self,
        types: RosMessageTypes,
        *,
        model_alias: str,
        base_frame: str,
    ) -> None:
        if model_alias not in finalist_aliases():
            raise ValueError(f"unsupported model marker identity: {model_alias!r}")
        if base_frame != "lexus3/base_link":
            raise ValueError("base_frame must be exactly 'lexus3/base_link'")
        self._types = types
        self._model_alias = model_alias
        self._base_frame = base_frame

    @property
    def marker_color(self) -> tuple[float, float, float]:
        return _MARKER_COLORS[self._model_alias]

    def _marker_color_for_label(
        self,
        class_names: tuple[str, ...],
        label: int,
    ) -> tuple[float, float, float]:
        class_name = _class_name(class_names, label)
        if self._model_alias != "pillar02_multiclass":
            return self.marker_color
        try:
            return _MULTICLASS_MARKER_COLORS[class_name]
        except KeyError as error:
            raise ValueError(
                "pillar02_multiclass marker class is not registered"
            ) from error

    def _require_result_identity(self, result: DetectionFrame) -> None:
        if not isinstance(result, DetectionFrame):
            raise TypeError("result must be a DetectionFrame")
        if result.model_alias != self._model_alias:
            raise ValueError("detection model identity does not match topic identity")

    def detection_array(
        self,
        result: DetectionFrame,
        *,
        stamp: object,
        calibration: SessionCalibration,
    ) -> object:
        self._require_result_identity(result)
        boxes = boxes_to_base_frame(result.boxes, calibration)
        centered = bottom_to_center(boxes)
        message = self._types.Detection3DArray()
        message.header = _copied_header(self._types, stamp, self._base_frame)
        message.detections = []
        for index, (box, score, label_index) in enumerate(
            zip(centered, result.scores, result.labels)
        ):
            detection = self._types.Detection3D()
            detection.header = _copied_header(
                self._types,
                stamp,
                self._base_frame,
            )
            detection.id = f"{self._model_alias}:{result.frame_index}:{index}"
            detection.bbox.center.position.x = float(box[0])
            detection.bbox.center.position.y = float(box[1])
            detection.bbox.center.position.z = float(box[2])
            detection.bbox.center.orientation.z = sin(float(box[6]) / 2.0)
            detection.bbox.center.orientation.w = cos(float(box[6]) / 2.0)
            detection.bbox.size.x = float(box[3])
            detection.bbox.size.y = float(box[4])
            detection.bbox.size.z = float(box[5])

            hypothesis = self._types.ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = _class_name(
                result.class_names,
                int(label_index),
            )
            hypothesis.hypothesis.score = float(score)
            hypothesis.pose.pose.position.x = float(box[0])
            hypothesis.pose.pose.position.y = float(box[1])
            hypothesis.pose.pose.position.z = float(box[2])
            hypothesis.pose.pose.orientation.z = sin(float(box[6]) / 2.0)
            hypothesis.pose.pose.orientation.w = cos(float(box[6]) / 2.0)
            detection.results = [hypothesis]
            message.detections.append(detection)
        return message

    def marker_array(
        self,
        result: DetectionFrame,
        *,
        stamp: object,
        calibration: SessionCalibration,
        previous_detection_count: int,
    ) -> object:
        self._require_result_identity(result)
        if previous_detection_count < 0:
            raise ValueError("previous_detection_count must be nonnegative")
        boxes = boxes_to_base_frame(result.boxes, calibration)
        centered = bottom_to_center(boxes)
        corners = box_corners_3d(boxes)
        markers: list[object] = []

        for index, (box_corners, box, score, label_index) in enumerate(
            zip(corners, centered, result.scores, result.labels)
        ):
            red, green, blue = self._marker_color_for_label(
                result.class_names,
                int(label_index),
            )
            wire = self._types.Marker()
            wire.header = _copied_header(self._types, stamp, self._base_frame)
            wire.ns = f"{self._model_alias}/boxes"
            wire.id = index
            wire.type = self._types.Marker.LINE_LIST
            wire.action = self._types.Marker.ADD
            wire.pose.orientation.w = 1.0
            wire.scale.x = 0.08
            wire.color.r = red
            wire.color.g = green
            wire.color.b = blue
            wire.color.a = 1.0
            for first, second in _BOX_EDGES:
                for corner_index in (first, second):
                    point = self._types.Point()
                    point.x = float(box_corners[corner_index, 0])
                    point.y = float(box_corners[corner_index, 1])
                    point.z = float(box_corners[corner_index, 2])
                    wire.points.append(point)
            markers.append(wire)

            label = self._types.Marker()
            label.header = _copied_header(self._types, stamp, self._base_frame)
            label.ns = f"{self._model_alias}/labels"
            label.id = index
            label.type = self._types.Marker.TEXT_VIEW_FACING
            label.action = self._types.Marker.ADD
            label.pose.position.x = float(box[0])
            label.pose.position.y = float(box[1])
            label.pose.position.z = float(box[2] + box[5] / 2.0 + 0.35)
            label.pose.orientation.w = 1.0
            label.scale.z = 0.45
            label.color.r = red
            label.color.g = green
            label.color.b = blue
            label.color.a = 1.0
            label.text = (
                f"{self._model_alias} "
                f"{_class_name(result.class_names, int(label_index))} "
                f"{float(score):.2f}"
            )
            markers.append(label)

        for stale_id in range(result.detection_count, previous_detection_count):
            for suffix in ("boxes", "labels"):
                stale = self._types.Marker()
                stale.header = _copied_header(
                    self._types,
                    stamp,
                    self._base_frame,
                )
                stale.ns = f"{self._model_alias}/{suffix}"
                stale.id = stale_id
                stale.action = self._types.Marker.DELETE
                markers.append(stale)
        message = self._types.MarkerArray()
        message.markers = markers
        return message

    def clear_markers(self, *, stamp: object) -> object:
        marker = self._types.Marker()
        marker.header = _copied_header(self._types, stamp, self._base_frame)
        marker.ns = self._model_alias
        marker.id = 0
        marker.action = self._types.Marker.DELETEALL
        message = self._types.MarkerArray()
        message.markers = [marker]
        return message

    def tracked_detection_array(
        self,
        frame: TrackedFrame,
        *,
        stamp: object,
    ) -> object:
        """Build stable-ID standard detections for confirmed tracks."""

        self._require_tracked_identity(frame)
        message = self._types.Detection3DArray()
        message.header = _copied_header(self._types, stamp, self._base_frame)
        message.detections = []
        for track in frame.visible_tracks:
            box = bottom_to_center(
                np.asarray((track.box,), dtype=np.float64)
            )[0]
            detection = self._types.Detection3D()
            detection.header = _copied_header(
                self._types,
                stamp,
                self._base_frame,
            )
            detection.id = str(track.track_id)
            self._set_detection_box(detection, box)
            hypothesis = self._types.ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = _class_name(
                frame.class_names,
                track.label,
            )
            hypothesis.hypothesis.score = float(track.score)
            hypothesis.pose.pose.position.x = float(box[0])
            hypothesis.pose.pose.position.y = float(box[1])
            hypothesis.pose.pose.position.z = float(box[2])
            hypothesis.pose.pose.orientation.z = sin(float(box[6]) / 2.0)
            hypothesis.pose.pose.orientation.w = cos(float(box[6]) / 2.0)
            detection.results = [hypothesis]
            message.detections.append(detection)
        return message

    def tracked_marker_array(
        self,
        frame: TrackedFrame,
        *,
        stamp: object,
        previous_track_ids: set[int] | frozenset[int],
    ) -> object:
        """Build stable-ID boxes, labels, velocity arrows, trails, and deletes."""

        self._require_tracked_identity(frame)
        if not isinstance(previous_track_ids, (set, frozenset)) or any(
            isinstance(track_id, bool)
            or not isinstance(track_id, int)
            or track_id <= 0
            for track_id in previous_track_ids
        ):
            raise ValueError("previous_track_ids must contain positive integers")
        markers: list[object] = []
        visible_ids: set[int] = set()
        for track in frame.visible_tracks:
            red, green, blue = self._marker_color_for_label(
                frame.class_names,
                track.label,
            )
            visible_ids.add(track.track_id)
            box_values = np.asarray((track.box,), dtype=np.float64)
            centered = bottom_to_center(box_values)[0]
            corners = box_corners_3d(box_values)[0]
            alpha = 0.35 if track.coasting else 1.0

            wire = self._types.Marker()
            wire.header = _copied_header(self._types, stamp, self._base_frame)
            wire.ns = f"{self._model_alias}/tracked_boxes"
            wire.id = track.track_id
            wire.type = self._types.Marker.LINE_LIST
            wire.action = self._types.Marker.ADD
            wire.pose.orientation.w = 1.0
            wire.scale.x = 0.1 if track.coasting else 0.08
            wire.color.r = red
            wire.color.g = green
            wire.color.b = blue
            wire.color.a = alpha
            for first, second in _BOX_EDGES:
                for corner_index in (first, second):
                    wire.points.append(self._point(corners[corner_index]))
            markers.append(wire)

            speed = hypot(track.velocity_xyz[0], track.velocity_xyz[1])
            state = "coasting" if track.coasting else "fresh"
            label = self._types.Marker()
            label.header = _copied_header(self._types, stamp, self._base_frame)
            label.ns = f"{self._model_alias}/tracked_labels"
            label.id = track.track_id
            label.type = self._types.Marker.TEXT_VIEW_FACING
            label.action = self._types.Marker.ADD
            label.pose.position.x = float(centered[0])
            label.pose.position.y = float(centered[1])
            label.pose.position.z = float(centered[2] + centered[5] / 2.0 + 0.35)
            label.pose.orientation.w = 1.0
            label.scale.z = 0.45
            label.color.r = red
            label.color.g = green
            label.color.b = blue
            label.color.a = 0.55 if track.coasting else 1.0
            label.text = (
                f"{_class_name(frame.class_names, track.label)} "
                f"#{track.track_id} {track.score:.2f} "
                f"{speed:.1f} m/s {state}"
            )
            markers.append(label)

            velocity = self._types.Marker()
            velocity.header = _copied_header(self._types, stamp, self._base_frame)
            velocity.ns = f"{self._model_alias}/tracked_velocity"
            velocity.id = track.track_id
            if speed > 0.05:
                velocity.type = self._types.Marker.ARROW
                velocity.action = self._types.Marker.ADD
                velocity.pose.orientation.w = 1.0
                velocity.scale.x = 0.07
                velocity.scale.y = 0.14
                velocity.scale.z = 0.14
                velocity.color.r = red
                velocity.color.g = green
                velocity.color.b = blue
                velocity.color.a = alpha
                velocity.points = [
                    self._point(centered[:3]),
                    self._point(
                        (
                            centered[0] + track.velocity_xyz[0],
                            centered[1] + track.velocity_xyz[1],
                            centered[2] + track.velocity_xyz[2],
                        )
                    ),
                ]
            else:
                velocity.action = self._types.Marker.DELETE
            markers.append(velocity)

            trail = self._types.Marker()
            trail.header = _copied_header(self._types, stamp, self._base_frame)
            trail.ns = f"{self._model_alias}/tracked_trails"
            trail.id = track.track_id
            if len(track.trail) >= 2:
                trail.type = self._types.Marker.LINE_STRIP
                trail.action = self._types.Marker.ADD
                trail.pose.orientation.w = 1.0
                trail.scale.x = 0.07
                trail.color.r = red
                trail.color.g = green
                trail.color.b = blue
                trail.color.a = alpha
                trail.points = [self._point(point) for point in track.trail]
            else:
                trail.action = self._types.Marker.DELETE
            markers.append(trail)

        for stale_id in sorted(set(previous_track_ids) - visible_ids):
            for suffix in (
                "tracked_boxes",
                "tracked_labels",
                "tracked_velocity",
                "tracked_trails",
            ):
                stale = self._types.Marker()
                stale.header = _copied_header(
                    self._types,
                    stamp,
                    self._base_frame,
                )
                stale.ns = f"{self._model_alias}/{suffix}"
                stale.id = stale_id
                stale.action = self._types.Marker.DELETE
                markers.append(stale)
        message = self._types.MarkerArray()
        message.markers = markers
        return message

    def clear_tracked_markers(self, *, stamp: object) -> object:
        marker = self._types.Marker()
        marker.header = _copied_header(self._types, stamp, self._base_frame)
        marker.ns = f"{self._model_alias}/tracked"
        marker.id = 0
        marker.action = self._types.Marker.DELETEALL
        message = self._types.MarkerArray()
        message.markers = [marker]
        return message

    def tracking_diagnostic_array(
        self,
        values: Mapping[str, object],
        *,
        stamp: object,
    ) -> object:
        message = self._types.DiagnosticArray()
        message.header = _copied_header(self._types, stamp, self._base_frame)
        status = self._types.DiagnosticStatus()
        last_error = str(values.get("last_error", ""))
        status.level = (
            self._types.DiagnosticStatus.ERROR
            if last_error
            else self._types.DiagnosticStatus.OK
        )
        status.message = "tracking failure" if last_error else "tracking active"
        status.name = f"centerpoint/{self._model_alias}/tracking"
        status.hardware_id = str(values.get("checkpoint_sha256", ""))
        status.values = []
        for key, value in values.items():
            item = self._types.KeyValue()
            item.key = str(key)
            item.value = str(value)
            status.values.append(item)
        message.status = [status]
        return message

    def _require_tracked_identity(self, frame: TrackedFrame) -> None:
        if not isinstance(frame, TrackedFrame):
            raise TypeError("frame must be a TrackedFrame")
        if frame.model_alias != self._model_alias:
            raise ValueError("tracked model identity does not match topic identity")
        if frame.coordinate_frame != self._base_frame:
            raise ValueError("tracked frame must use the configured base frame")

    def _point(self, values: object) -> object:
        xyz = np.asarray(values, dtype=np.float64)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("marker point must contain three finite values")
        point = self._types.Point()
        point.x = float(xyz[0])
        point.y = float(xyz[1])
        point.z = float(xyz[2])
        return point

    @staticmethod
    def _set_detection_box(detection: object, box: np.ndarray) -> None:
        detection.bbox.center.position.x = float(box[0])
        detection.bbox.center.position.y = float(box[1])
        detection.bbox.center.position.z = float(box[2])
        detection.bbox.center.orientation.z = sin(float(box[6]) / 2.0)
        detection.bbox.center.orientation.w = cos(float(box[6]) / 2.0)
        detection.bbox.size.x = float(box[3])
        detection.bbox.size.y = float(box[4])
        detection.bbox.size.z = float(box[5])

    def model_point_cloud(
        self,
        points: np.ndarray,
        *,
        stamp: object,
        calibration: SessionCalibration,
    ) -> object:
        values = np.asarray(points)
        if values.dtype != np.dtype(np.float32):
            raise TypeError("model points must use float32")
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError("model points must have shape (N, 4)")
        if not np.isfinite(values).all():
            raise ValueError("model points must be finite")
        selected = np.array(
            values[finalist_range_mask(values)],
            dtype=np.float32,
            order="C",
            copy=True,
        )
        selected[:, :3] += sensor_translation_to_base_frame(calibration).astype(
            np.float32
        )

        message = self._types.PointCloud2()
        message.header = _copied_header(self._types, stamp, self._base_frame)
        message.height = 1
        message.width = int(selected.shape[0])
        message.is_bigendian = False
        message.point_step = 16
        message.row_step = message.width * message.point_step
        message.is_dense = True
        message.fields = []
        for name, offset in (("x", 0), ("y", 4), ("z", 8), ("reflectivity", 12)):
            field = self._types.PointField()
            field.name = name
            field.offset = offset
            field.datatype = self._types.PointField.FLOAT32
            field.count = 1
            message.fields.append(field)
        message.data = selected.astype("<f4", copy=False).tobytes(order="C")
        return message

    def diagnostic_array(
        self,
        values: Mapping[str, object],
        *,
        stamp: object,
    ) -> object:
        message = self._types.DiagnosticArray()
        message.header = _copied_header(self._types, stamp, self._base_frame)
        status = self._types.DiagnosticStatus()
        last_error = str(values.get("last_error", ""))
        last_error_stage = str(values.get("last_error_stage", ""))
        failed = int(values.get("failed_frames", 0))
        dropped = int(values.get("dropped_frames", 0))
        middleware_lost = int(values.get("middleware_lost_frames", 0))
        if failed or (last_error and last_error_stage != "middleware"):
            status.level = self._types.DiagnosticStatus.ERROR
            status.message = "frame processing failure"
        elif middleware_lost:
            status.level = self._types.DiagnosticStatus.WARN
            status.message = "middleware messages lost"
        elif dropped:
            status.level = self._types.DiagnosticStatus.WARN
            status.message = "frames dropped or overwritten"
        else:
            status.level = self._types.DiagnosticStatus.OK
            status.message = "ok"
        status.name = f"centerpoint/{self._model_alias}"
        status.hardware_id = str(values.get("checkpoint_sha256", ""))
        status.values = []
        for key, value in values.items():
            item = self._types.KeyValue()
            item.key = str(key)
            item.value = str(value)
            status.values.append(item)
        message.status = [status]
        return message
