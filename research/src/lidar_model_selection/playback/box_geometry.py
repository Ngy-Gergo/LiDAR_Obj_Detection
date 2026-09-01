"""Framework-independent geometry for bottom-centred LiDAR detection boxes."""

from __future__ import annotations

import numpy

from .contracts import SessionCalibration


def _validated_boxes(boxes: numpy.ndarray) -> numpy.ndarray:
    values = numpy.asarray(boxes)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError("boxes must have shape (N, 7)")
    if not numpy.issubdtype(values.dtype, numpy.number):
        raise TypeError("boxes must contain numeric values")

    values = numpy.asarray(values, dtype=numpy.float64)
    if not numpy.isfinite(values).all():
        raise ValueError("boxes must contain only finite values")
    if numpy.any(values[:, 3:6] <= 0.0):
        raise ValueError("box dimensions must be strictly positive")
    return values


def bottom_to_center(boxes: numpy.ndarray) -> numpy.ndarray:
    """Return ``(x, y, z_center, dx, dy, dz, yaw)`` for LiDAR boxes."""

    values = _validated_boxes(boxes)
    centered = numpy.array(values, dtype=numpy.float64, order="C", copy=True)
    centered[:, 2] += centered[:, 5] / 2.0
    centered.setflags(write=False)
    return centered


def boxes_to_base_frame(
    boxes: numpy.ndarray,
    calibration: SessionCalibration,
) -> numpy.ndarray:
    """Translate model-frame boxes into the recorded vehicle base frame.

    Kaposvar normalization has already applied the recorded sensor rotation to
    model points.  Detector boxes therefore need only the validated recorded
    sensor translation here.  Keeping this operation beside the shared box
    geometry makes raw and tracked ROS output use exactly the same conversion.
    """

    values = _validated_boxes(boxes)
    translated = numpy.array(values, dtype=numpy.float64, order="C", copy=True)
    translated[:, :3] += sensor_translation_to_base_frame(calibration)
    translated.setflags(write=False)
    return translated


def sensor_translation_to_base_frame(
    calibration: SessionCalibration,
) -> numpy.ndarray:
    """Return the validated recorded sensor translation as immutable XYZ."""

    if not isinstance(calibration, SessionCalibration):
        raise TypeError("calibration must be a SessionCalibration")
    if calibration.parent_frame_id != "lexus3/base_link":
        raise ValueError("calibration parent must be lexus3/base_link")
    if calibration.child_frame_id != "lexus3/os_center":
        raise ValueError("calibration child must be lexus3/os_center")
    translation = numpy.asarray(
        calibration.translation_xyz,
        dtype=numpy.float64,
    ).copy()
    translation.setflags(write=False)
    return translation


def bev_corners(boxes: numpy.ndarray) -> numpy.ndarray:
    """Return four XY corners for each bottom-centred LiDAR box.

    A zero-yaw box has its ``dx`` dimension along positive/negative X and its
    ``dy`` dimension along positive/negative Y. Positive yaw rotates from X
    toward Y about the positive Z axis.
    """

    values = _validated_boxes(boxes)
    count = values.shape[0]
    if count == 0:
        empty = numpy.empty((0, 4, 2), dtype=numpy.float64)
        empty.setflags(write=False)
        return empty

    signs = numpy.asarray(
        ((1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)),
        dtype=numpy.float64,
    )
    local = signs[numpy.newaxis, :, :] * values[:, numpy.newaxis, 3:5] / 2.0
    cosine = numpy.cos(values[:, 6])[:, numpy.newaxis]
    sine = numpy.sin(values[:, 6])[:, numpy.newaxis]

    corners = numpy.empty((count, 4, 2), dtype=numpy.float64)
    corners[:, :, 0] = (
        values[:, numpy.newaxis, 0]
        + local[:, :, 0] * cosine
        - local[:, :, 1] * sine
    )
    corners[:, :, 1] = (
        values[:, numpy.newaxis, 1]
        + local[:, :, 0] * sine
        + local[:, :, 1] * cosine
    )
    corners = numpy.ascontiguousarray(corners)
    corners.setflags(write=False)
    return corners


def box_corners_3d(boxes: numpy.ndarray) -> numpy.ndarray:
    """Return bottom four then top four XYZ corners for every LiDAR box."""

    values = _validated_boxes(boxes)
    xy = bev_corners(values)
    corners = numpy.empty((values.shape[0], 8, 3), dtype=numpy.float64)
    corners[:, :4, :2] = xy
    corners[:, 4:, :2] = xy
    corners[:, :4, 2] = values[:, numpy.newaxis, 2]
    corners[:, 4:, 2] = values[:, numpy.newaxis, 2] + values[:, numpy.newaxis, 5]
    corners = numpy.ascontiguousarray(corners)
    corners.setflags(write=False)
    return corners
