"""ROS-message construction without importing ROS at module import time."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from typing import Mapping

import numpy as np

from .box_geometry import bottom_to_center, box_corners_3d
from .contracts import SessionCalibration
from .model_registry import finalist_aliases, finalist_range_mask
from .results import DetectionFrame


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


def _translation(calibration: SessionCalibration) -> np.ndarray:
    if not isinstance(calibration, SessionCalibration):
        raise TypeError("calibration must be a SessionCalibration")
    if calibration.parent_frame_id != "lexus3/base_link":
        raise ValueError("calibration parent must be lexus3/base_link")
    if calibration.child_frame_id != "lexus3/os_center":
        raise ValueError("calibration child must be lexus3/os_center")
    return np.asarray(calibration.translation_xyz, dtype=np.float64)


def _translated_boxes(
    result: DetectionFrame,
    calibration: SessionCalibration,
) -> np.ndarray:
    boxes = np.array(result.boxes, dtype=np.float64, order="C", copy=True)
    boxes[:, :3] += _translation(calibration)
    return boxes


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
        boxes = _translated_boxes(result, calibration)
        centered = bottom_to_center(boxes)
        message = self._types.Detection3DArray()
        message.header = _copied_header(self._types, stamp, self._base_frame)
        message.detections = []
        for index, (box, score) in enumerate(zip(centered, result.scores)):
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
            hypothesis.hypothesis.class_id = "Car"
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
        boxes = _translated_boxes(result, calibration)
        centered = bottom_to_center(boxes)
        corners = box_corners_3d(boxes)
        red, green, blue = self.marker_color
        markers: list[object] = []

        for index, (box_corners, box, score) in enumerate(
            zip(corners, centered, result.scores)
        ):
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
            label.text = f"{self._model_alias} Car {float(score):.2f}"
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
        selected[:, :3] += _translation(calibration).astype(np.float32)

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
        failed = int(values.get("failed_frames", 0))
        dropped = int(values.get("dropped_frames", 0))
        if last_error or failed:
            status.level = self._types.DiagnosticStatus.ERROR
            status.message = "frame processing failure"
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
