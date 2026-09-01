"""Optional interactive BEV rendering for normalized playback frames."""

from __future__ import annotations

from math import ceil
from numbers import Integral
from typing import Any

import numpy

from .box_geometry import bev_corners


class BevViewer:
    """Reuse one Matplotlib figure to display points and detections in LiDAR BEV."""

    def __init__(self, *, max_display_points: int = 30_000) -> None:
        if (
            isinstance(max_display_points, bool)
            or not isinstance(max_display_points, Integral)
        ):
            raise TypeError("max_display_points must be an integer and not a boolean")
        if max_display_points <= 0:
            raise ValueError("max_display_points must be greater than zero")

        # Matplotlib is deliberately optional and isolated from core imports.
        import matplotlib.pyplot as pyplot

        self._pyplot = pyplot
        self._max_display_points = int(max_display_points)
        self._pyplot.ion()
        self._figure, self._axes = self._pyplot.subplots(num="LiDAR playback BEV")
        self._closed = False
        self._figure.canvas.mpl_connect("close_event", self._on_close)

    @property
    def closed(self) -> bool:
        if not self._closed and not self._pyplot.fignum_exists(self._figure.number):
            self._closed = True
        return self._closed

    def _on_close(self, _event: Any) -> None:
        self._closed = True

    def close(self) -> None:
        if not self.closed:
            self._pyplot.close(self._figure)
        self._closed = True

    def render(self, frame: Any, result: Any) -> bool:
        """Draw one frame and return ``False`` after the user closes the figure."""

        if self.closed:
            return False
        if frame.coordinate_frame != "lidar":
            raise ValueError("BEV input points must use the lidar coordinate frame")
        if result.session_id != frame.session_id:
            raise ValueError("point and detection session IDs must match")
        if result.frame_index != frame.frame_index:
            raise ValueError("point and detection frame indices must match")
        if result.coordinate_frame != frame.coordinate_frame:
            raise ValueError("point and detection coordinate frames must match")

        points = numpy.asarray(frame.points)
        boxes = numpy.asarray(result.boxes)
        scores = numpy.asarray(result.scores)
        labels = numpy.asarray(result.labels)
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError("viewer points must have shape (N, 4)")
        if boxes.ndim != 2 or boxes.shape[1] != 7:
            raise ValueError("viewer boxes must have shape (N, 7)")
        if scores.shape != (boxes.shape[0],) or labels.shape != (boxes.shape[0],):
            raise ValueError("viewer box, score, and label counts must match")

        axes = self._axes
        axes.clear()
        if points.shape[0]:
            stride = max(1, ceil(points.shape[0] / self._max_display_points))
            displayed = points[::stride][: self._max_display_points]
            axes.scatter(
                displayed[:, 1],
                displayed[:, 0],
                c=displayed[:, 3],
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                marker=".",
                linewidths=0.0,
                s=1.0,
            )

        for index, (corners, score, label) in enumerate(
            zip(bev_corners(boxes), scores, labels)
        ):
            closed = numpy.concatenate((corners, corners[:1]), axis=0)
            axes.plot(closed[:, 1], closed[:, 0], color="tab:red", linewidth=1.5)
            axes.text(
                float(boxes[index, 1]),
                float(boxes[index, 0]),
                f"Car {float(score):.2f} #{index}" if int(label) == 0 else f"{int(label)} {float(score):.2f} #{index}",
                color="tab:red",
                fontsize=7,
            )

        detector_ms = getattr(result, "detector_ms", 0.0)
        processing_ms = getattr(result, "frame_processing_ms", 0.0)
        axes.set_title(
            f"{frame.session_id}  frame={frame.frame_index}  "
            f"model={result.model_alias}  detections={boxes.shape[0]}\n"
            f"detector_ms={detector_ms:.2f}  frame_processing_ms={processing_ms:.2f}"
        )
        axes.set_xlabel("y left [m]")
        axes.set_ylabel("x forward [m]")
        axes.set_aspect("equal", adjustable="box")
        axes.grid(True, alpha=0.2)
        self._figure.canvas.draw_idle()
        self._figure.canvas.flush_events()
        return not self.closed

    def __enter__(self) -> "BevViewer":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
