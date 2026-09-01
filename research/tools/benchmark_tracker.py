#!/usr/bin/env python3
"""Repeatable CPU-only synthetic latency benchmark for the online tracker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter_ns
from typing import Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_SOURCE = REPOSITORY_ROOT / "research" / "src"
if str(RESEARCH_SOURCE) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SOURCE))

from lidar_model_selection.playback.contracts import SessionCalibration  # noqa: E402
from lidar_model_selection.playback.results import DetectionFrame  # noqa: E402
from lidar_model_selection.playback.tracking import (  # noqa: E402
    OnlineBoxTracker,
    TrackerConfig,
)


def _calibration() -> SessionCalibration:
    return SessionCalibration(
        parent_frame_id="lexus3/base_link",
        child_frame_id="lexus3/os_center",
        translation_xyz=(0.75, 0.0, 1.91),
        quaternion_xyzw=(0.0, 0.0, -1.0, 0.0),
        rotation_matrix=np.diag((-1.0, -1.0, 1.0)),
    )


def _frame(index: int, detection_count: int) -> DetectionFrame:
    columns = 8
    object_ids = np.arange(detection_count, dtype=np.float32)
    boxes = np.empty((detection_count, 7), dtype=np.float32)
    boxes[:, 0] = 4.0 + (object_ids % columns) * 7.0 + index * 0.025
    boxes[:, 1] = -28.0 + (object_ids // columns) * 7.0
    boxes[:, 2] = -1.2
    boxes[:, 3] = 4.2
    boxes[:, 4] = 1.9
    boxes[:, 5] = 1.6
    boxes[:, 6] = 0.02 * (object_ids % 5)
    scores = np.linspace(0.95, 0.55, detection_count, dtype=np.float32)
    labels = np.zeros((detection_count,), dtype=np.int64)
    for values in (boxes, scores, labels):
        values.setflags(write=False)
    return DetectionFrame(
        session_id="synthetic-tracker-benchmark",
        frame_index=index,
        timestamp_ns=1_000_000_000 + index * 100_000_000,
        storage_timestamp_ns=None,
        source_frame_id="lexus3/os_center",
        coordinate_frame="lidar",
        source_key=f"synthetic[{index}]",
        model_alias="voxel0075",
        run_id="synthetic",
        config_sha256="a" * 64,
        checkpoint_path="synthetic.pth",
        checkpoint_sha256="b" * 64,
        checkpoint_size_bytes=1,
        source_point_count=1,
        dropped_nonfinite_count=0,
        input_point_count=1,
        in_range_point_count=1,
        detection_count=detection_count,
        status="success" if detection_count else "empty_source",
        boxes=boxes,
        scores=scores,
        labels=labels,
        decode_ms=0.0,
        detector_ms=0.0,
        frame_processing_ms=0.0,
    )


def run_benchmark(*, iterations: int, detections: int, warmup: int) -> dict[str, object]:
    if iterations <= 0 or detections < 0 or warmup < 0:
        raise ValueError("iterations must be positive; detections and warmup nonnegative")
    tracker = OnlineBoxTracker(
        TrackerConfig(min_confirmed_hits=1),
        model_alias="voxel0075",
    )
    calibration = _calibration()
    latency_ms: list[float] = []
    for index in range(iterations + warmup):
        frame = _frame(index, detections)
        started_ns = perf_counter_ns()
        tracker.update(
            frame,
            calibration=calibration,
            generation=0,
        )
        elapsed_ms = (perf_counter_ns() - started_ns) / 1_000_000.0
        if index >= warmup:
            latency_ms.append(elapsed_ms)
    values = np.asarray(latency_ms, dtype=np.float64)
    return {
        "schema": "centerpoint-tracker-benchmark-v1",
        "iterations": iterations,
        "warmup": warmup,
        "detections_per_frame": detections,
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "maximum_ms": float(values.max()),
        "target_p95_ms": 2.0,
        "target_met": bool(np.percentile(values, 95) <= 2.0),
    }


def main(args: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--detections", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=100)
    values = parser.parse_args(args)
    try:
        report = run_benchmark(
            iterations=values.iterations,
            detections=values.detections,
            warmup=values.warmup,
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
