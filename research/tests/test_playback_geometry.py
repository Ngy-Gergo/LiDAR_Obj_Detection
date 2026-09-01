from __future__ import annotations

from math import pi
from types import SimpleNamespace

import numpy
import pytest

from lidar_model_selection.playback.box_geometry import (
    bev_corners,
    bottom_to_center,
    box_corners_3d,
)


@pytest.mark.parametrize(
    ("yaw", "expected_first"),
    (
        (0.0, (2.0, 1.0)),
        (pi / 2.0, (-1.0, 2.0)),
        (-pi / 2.0, (1.0, -2.0)),
        (pi, (-2.0, -1.0)),
    ),
)
def test_bev_corners_follow_lidar_yaw_convention(
    yaw: float,
    expected_first: tuple[float, float],
) -> None:
    boxes = numpy.asarray([[0.0, 0.0, 3.0, 4.0, 2.0, 6.0, yaw]])

    corners = bev_corners(boxes)

    numpy.testing.assert_allclose(corners[0, 0], expected_first, atol=1e-12)
    assert corners.shape == (1, 4, 2)
    assert not corners.flags.writeable


def test_bottom_center_and_3d_corners_preserve_box_origin() -> None:
    boxes = numpy.asarray([[1.0, 2.0, 3.0, 4.0, 2.0, 6.0, 0.0]])

    centered = bottom_to_center(boxes)
    corners = box_corners_3d(boxes)

    numpy.testing.assert_allclose(centered[0], [1.0, 2.0, 6.0, 4.0, 2.0, 6.0, 0.0])
    numpy.testing.assert_allclose(corners[0, :4, 2], 3.0)
    numpy.testing.assert_allclose(corners[0, 4:, 2], 9.0)
    numpy.testing.assert_allclose(corners[0, :, :2].mean(axis=0), [1.0, 2.0])
    assert not centered.flags.writeable
    assert not corners.flags.writeable


def test_geometry_handles_empty_boxes_and_rejects_invalid_dimensions() -> None:
    assert bev_corners(numpy.empty((0, 7))).shape == (0, 4, 2)
    assert box_corners_3d(numpy.empty((0, 7))).shape == (0, 8, 3)

    with pytest.raises(ValueError, match="strictly positive"):
        bev_corners(numpy.asarray([[0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]]))


def test_bev_viewer_reuses_one_figure_and_closes_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    from lidar_model_selection.playback.bev_viewer import BevViewer

    points = numpy.asarray(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.5], [2.0, 3.0, 0.0, 1.0]],
        dtype=numpy.float32,
    )
    frame = SimpleNamespace(
        coordinate_frame="lidar",
        session_id="session",
        frame_index=4,
        points=points,
    )
    result = SimpleNamespace(
        session_id="session",
        frame_index=4,
        coordinate_frame="lidar",
        boxes=numpy.asarray([[1.0, 2.0, -1.0, 4.0, 2.0, 1.5, 0.0]]),
        scores=numpy.asarray([0.8]),
        labels=numpy.asarray([0]),
        model_alias="voxel0075",
        detector_ms=3.0,
        frame_processing_ms=4.0,
    )
    viewer = BevViewer(max_display_points=2)
    figure = viewer._figure

    assert viewer.render(frame, result)
    assert viewer._figure is figure
    assert len(viewer._axes.collections[0].get_offsets()) == 2
    assert "x forward" in viewer._axes.get_ylabel()
    assert "y left" in viewer._axes.get_xlabel()

    result.coordinate_frame = "base_link"
    with pytest.raises(ValueError, match="coordinate frames"):
        viewer.render(frame, result)
    result.coordinate_frame = "lidar"

    viewer.close()
    assert viewer.closed
    assert viewer.render(frame, result) is False
