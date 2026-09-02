from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from lidar_model_selection.playback.contracts import SessionCalibration
from lidar_model_selection.playback.results import DetectionFrame, PlaybackErrorEvidence
from lidar_model_selection.playback.tracking import (
    OnlineBoxTracker,
    TrackerConfig,
    wrap_yaw,
)


def _calibration() -> SessionCalibration:
    return SessionCalibration(
        parent_frame_id="lexus3/base_link",
        child_frame_id="lexus3/os_center",
        translation_xyz=(0.75, 0.0, 1.91),
        quaternion_xyzw=(0.0, 0.0, -1.0, 0.0),
        rotation_matrix=np.diag((-1.0, -1.0, 1.0)),
    )


def _frame(
    timestamp_ns: int,
    boxes: list[tuple[float, float, float, float, float, float, float]],
    *,
    scores: list[float] | None = None,
    labels: list[int] | None = None,
    class_names: tuple[str, ...] = ("Car",),
    frame_index: int = 0,
) -> DetectionFrame:
    box_values = np.asarray(boxes, dtype=np.float32).reshape((-1, 7))
    score_values = np.asarray(
        scores if scores is not None else [0.8] * len(boxes),
        dtype=np.float32,
    )
    label_values = np.asarray(
        labels if labels is not None else [0] * len(boxes), dtype=np.int64
    )
    for values in (box_values, score_values, label_values):
        values.setflags(write=False)
    status = "success" if boxes else "empty_source"
    return DetectionFrame(
        session_id="live:/lexus3/os_center/points",
        frame_index=frame_index,
        timestamp_ns=timestamp_ns,
        storage_timestamp_ns=None,
        source_frame_id="lexus3/os_center",
        coordinate_frame="lidar",
        source_key=f"points[{frame_index}]",
        model_alias="voxel0075",
        run_id="run",
        config_sha256="a" * 64,
        checkpoint_path="training/epoch_20.pth",
        checkpoint_sha256="b" * 64,
        checkpoint_size_bytes=1,
        source_point_count=1,
        dropped_nonfinite_count=0,
        input_point_count=1,
        in_range_point_count=1,
        detection_count=len(boxes),
        status=status,
        boxes=box_values,
        scores=score_values,
        labels=label_values,
        decode_ms=1.0,
        detector_ms=2.0,
        frame_processing_ms=3.0,
        class_names=class_names,
    )


BOX = (1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.0)
MULTICLASS = ("Car", "Pedestrian", "Cyclist")


def test_multiclass_tracks_are_stable_and_never_cross_class() -> None:
    tracker = OnlineBoxTracker(TrackerConfig(min_confirmed_hits=1), model_alias="voxel0075")
    first = [
        (0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0),
        (10.0, 0.0, 0.0, 0.6, 0.6, 1.7, 0.0),
        (20.0, 0.0, 0.0, 1.8, 0.6, 1.6, 0.0),
    ]
    tracker.update(
        _frame(1_000_000_000, first, labels=[0, 1, 2], class_names=MULTICLASS),
        calibration=_calibration(), generation=0,
    )
    stable = tracker.update(
        _frame(
            1_100_000_000,
            [(0.5, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0),
             (10.5, 0.0, 0.0, 0.6, 0.6, 1.7, 0.0),
             (21.0, 0.0, 0.0, 1.8, 0.6, 1.6, 0.0)],
            labels=[0, 1, 2], class_names=MULTICLASS, frame_index=1,
        ), calibration=_calibration(), generation=0,
    )
    assert [(track.track_id, track.label) for track in stable.tracks] == [
        (1, 0), (2, 1), (3, 2)
    ]
    cross_class = tracker.update(
        _frame(
            1_200_000_000,
            [(0.6, 0.0, 0.0, 0.6, 0.6, 1.7, 0.0)],
            labels=[1], class_names=MULTICLASS, frame_index=2,
        ), calibration=_calibration(), generation=0,
    )
    assert cross_class.diagnostics.matches == 0
    assert any(track.label == 1 and track.track_id == 4 for track in cross_class.tracks)


def test_class_specific_association_gates_and_unknown_fallback() -> None:
    def paired(label: int, classes: tuple[str, ...], distance: float) -> int:
        tracker = OnlineBoxTracker(TrackerConfig(min_confirmed_hits=1), model_alias="voxel0075")
        tracker.update(
            _frame(1_000_000_000, [BOX], labels=[label], class_names=classes),
            calibration=_calibration(), generation=0,
        )
        result = tracker.update(
            _frame(
                1_100_000_000,
                [(BOX[0] + distance, *BOX[1:])],
                labels=[label], class_names=classes, frame_index=1,
            ), calibration=_calibration(), generation=0,
        )
        return result.diagnostics.matches

    assert paired(1, MULTICLASS, 1.6) == 0  # Pedestrian gate is 1.5 m.
    assert paired(2, MULTICLASS, 2.4) == 1  # Cyclist gate is 2.5 m.
    assert paired(2, MULTICLASS, 2.6) == 0
    assert paired(0, MULTICLASS, 3.9) == 1  # Car gate remains 4.0 m.
    assert paired(1, ("Car", "Unknown"), 3.9) == 1  # Global fallback.


def test_crossing_pedestrians_keep_predicted_track_ids() -> None:
    tracker = OnlineBoxTracker(
        TrackerConfig(
            min_confirmed_hits=1,
            max_time_gap_seconds=2.0,
            position_smoothing=1.0,
        ),
        model_alias="voxel0075",
    )
    def pedestrians(left: float, right: float):
        return [
            (left, 0.0, 0.0, 0.6, 0.6, 1.7, 0.0),
            (right, 0.0, 0.0, 0.6, 0.6, 1.7, math.pi),
        ]

    tracker.update(
        _frame(
            1_000_000_000,
            pedestrians(0.0, 2.0),
            labels=[1, 1],
            class_names=MULTICLASS,
        ),
        calibration=_calibration(),
        generation=0,
    )
    tracker.update(
        _frame(
            1_100_000_000,
            pedestrians(0.5, 1.5),
            labels=[1, 1],
            class_names=MULTICLASS,
            frame_index=1,
        ),
        calibration=_calibration(),
        generation=0,
    )
    crossed = tracker.update(
        _frame(
            1_200_000_000,
            pedestrians(1.1, 0.9),
            labels=[1, 1],
            class_names=MULTICLASS,
            frame_index=2,
        ),
        calibration=_calibration(),
        generation=0,
    )
    assert [(track.track_id, track.label) for track in crossed.tracks] == [
        (1, 1),
        (2, 1),
    ]
    assert crossed.tracks[0].box[0] > crossed.tracks[1].box[0]


def test_first_track_is_tentative_then_confirms_with_stable_id() -> None:
    tracker = OnlineBoxTracker(TrackerConfig(), model_alias="voxel0075")
    first = tracker.update(_frame(1_000_000_000, [BOX]), calibration=_calibration(), generation=0)
    assert [(track.track_id, track.confirmed) for track in first.tracks] == [(1, False)]
    assert first.visible_tracks == ()

    moved = (1.2, 2.0, 0.0, 4.0, 2.0, 1.5, 0.05)
    second = tracker.update(
        _frame(1_100_000_000, [moved], frame_index=1),
        calibration=_calibration(),
        generation=0,
    )
    assert [(track.track_id, track.confirmed) for track in second.tracks] == [(1, True)]
    assert second.visible_tracks[0].track_id == 1
    assert second.diagnostics.matches == 1
    assert second.diagnostics.tracked_frames_total == 2
    assert second.diagnostics.last_dt_seconds == pytest.approx(0.1)


def test_irregular_timestamp_velocity_and_confidence_smoothing() -> None:
    tracker = OnlineBoxTracker(
        TrackerConfig(min_confirmed_hits=1),
        model_alias="voxel0075",
    )
    tracker.update(
        _frame(1_000_000_000, [BOX], scores=[0.5]),
        calibration=_calibration(),
        generation=0,
    )
    moved = (2.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.0)
    result = tracker.update(
        _frame(1_500_000_000, [moved], scores=[0.8], frame_index=1),
        calibration=_calibration(),
        generation=0,
    )
    track = result.tracks[0]
    assert track.track_id == 1
    assert track.velocity_xyz[0] == pytest.approx(1.3)
    assert track.velocity_xyz[1:] == pytest.approx((0.0, 0.0))
    assert track.score == pytest.approx(0.695)
    assert result.diagnostics.maximum_observed_gap_seconds == pytest.approx(0.5)


def test_multiple_crossing_objects_follow_constant_velocity_predictions() -> None:
    tracker = OnlineBoxTracker(
        TrackerConfig(
            min_confirmed_hits=1,
            association_distance_meters=4.0,
            position_smoothing=1.0,
            max_time_gap_seconds=2.0,
        ),
        model_alias="voxel0075",
    )
    def boxes(left: float, right: float):
        return [
            (left, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0),
            (right, 0.0, 0.0, 4.0, 2.0, 1.5, math.pi),
        ]

    tracker.update(_frame(1_000_000_000, boxes(0.0, 10.0)), calibration=_calibration(), generation=0)
    tracker.update(
        _frame(2_000_000_000, boxes(2.0, 8.0), frame_index=1),
        calibration=_calibration(),
        generation=0,
    )
    crossed = tracker.update(
        _frame(3_000_000_000, boxes(4.1, 5.9), frame_index=2),
        calibration=_calibration(),
        generation=0,
    )
    by_id = {track.track_id: track for track in crossed.tracks}
    assert by_id[1].box[0] == pytest.approx(4.85)
    assert by_id[2].box[0] == pytest.approx(6.65)
    assert crossed.diagnostics.matches == 2


def test_distance_gate_creates_new_id_and_coasts_old_track() -> None:
    tracker = OnlineBoxTracker(
        TrackerConfig(
            min_confirmed_hits=1,
            association_distance_meters=1.0,
        ),
        model_alias="voxel0075",
    )
    tracker.update(
        _frame(1_000_000_000, [BOX]),
        calibration=_calibration(),
        generation=0,
    )
    far = (20.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.0)
    result = tracker.update(
        _frame(1_100_000_000, [far], frame_index=1),
        calibration=_calibration(),
        generation=0,
    )
    assert [track.track_id for track in result.tracks] == [1, 2]
    assert result.tracks[0].coasting
    assert not result.tracks[1].coasting
    assert result.diagnostics.matches == 0
    assert result.diagnostics.unmatched_detections == 1
    assert result.diagnostics.created_tracks == 1
    assert result.diagnostics.misses == 1


def test_equal_distance_association_has_explicit_track_and_detection_tie_break() -> None:
    tracker = OnlineBoxTracker(
        TrackerConfig(
            min_confirmed_hits=1,
            position_smoothing=1.0,
            association_distance_meters=3.0,
        ),
        model_alias="voxel0075",
    )
    initial = [
        (-1.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0),
        (1.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0),
    ]
    tied = [
        (0.0, 0.0, 0.0, 3.0, 1.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 5.0, 2.5, 2.0, 0.0),
    ]
    tracker.update(_frame(1_000_000_000, initial), calibration=_calibration(), generation=0)
    result = tracker.update(
        _frame(1_100_000_000, tied, frame_index=1),
        calibration=_calibration(),
        generation=0,
    )
    assert result.tracks[0].box[3:6] == pytest.approx(tied[0][3:6])
    assert result.tracks[1].box[3:6] == pytest.approx(tied[1][3:6])


def test_yaw_smoothing_uses_short_path_across_pi_boundary() -> None:
    tracker = OnlineBoxTracker(
        TrackerConfig(min_confirmed_hits=1),
        model_alias="voxel0075",
    )
    first = (*BOX[:6], math.pi - 0.05)
    second = (*BOX[:6], -math.pi + 0.05)
    tracker.update(_frame(1_000_000_000, [first]), calibration=_calibration(), generation=0)
    result = tracker.update(
        _frame(1_100_000_000, [second], frame_index=1),
        calibration=_calibration(),
        generation=0,
    )
    assert abs(abs(result.tracks[0].box[6]) - math.pi) < 0.06
    assert wrap_yaw(3.0 * math.pi) == pytest.approx(-math.pi)


def test_empty_frames_coast_then_expire_without_resetting_sequence() -> None:
    tracker = OnlineBoxTracker(
        TrackerConfig(min_confirmed_hits=1, max_missed_frames=2),
        model_alias="voxel0075",
    )
    tracker.update(_frame(1_000_000_000, [BOX]), calibration=_calibration(), generation=0)
    first_empty = tracker.update(
        _frame(1_100_000_000, [], frame_index=1),
        calibration=_calibration(),
        generation=0,
    )
    second_empty = tracker.update(
        _frame(1_200_000_000, [], frame_index=2),
        calibration=_calibration(),
        generation=0,
    )
    expired = tracker.update(
        _frame(1_300_000_000, [], frame_index=3),
        calibration=_calibration(),
        generation=0,
    )
    assert first_empty.visible_tracks[0].coasting
    assert second_empty.visible_tracks[0].missed_frames == 2
    assert expired.tracks == ()
    assert expired.diagnostics.reset_count == 0
    assert expired.diagnostics.deleted_tracks_total == 1
    assert expired.diagnostics.removed_tracks == 1


def test_short_occlusion_reacquires_same_track_id() -> None:
    tracker = OnlineBoxTracker(
        TrackerConfig(min_confirmed_hits=1),
        model_alias="voxel0075",
    )
    tracker.update(_frame(1_000_000_000, [BOX]), calibration=_calibration(), generation=0)
    tracker.update(
        _frame(1_100_000_000, [], frame_index=1),
        calibration=_calibration(),
        generation=0,
    )
    result = tracker.update(
        _frame(1_200_000_000, [(1.2, *BOX[1:])], frame_index=2),
        calibration=_calibration(),
        generation=0,
    )
    assert result.visible_tracks[0].track_id == 1
    assert not result.visible_tracks[0].coasting
    assert result.visible_tracks[0].missed_frames == 0


def test_excessive_gap_generation_change_and_backward_time_reset_ids() -> None:
    tracker = OnlineBoxTracker(
        TrackerConfig(min_confirmed_hits=1, max_time_gap_seconds=0.5),
        model_alias="voxel0075",
    )
    tracker.update(_frame(1_000_000_000, [BOX]), calibration=_calibration(), generation=0)
    gap = tracker.update(
        _frame(2_000_000_000, [BOX], frame_index=1),
        calibration=_calibration(),
        generation=0,
    )
    assert gap.tracks[0].track_id == 1
    assert gap.diagnostics.last_reset_reason == "excessive_time_gap"

    generation = tracker.update(
        _frame(2_100_000_000, [BOX], frame_index=0),
        calibration=_calibration(),
        generation=1,
    )
    assert generation.tracks[0].track_id == 1
    assert generation.diagnostics.last_reset_reason == "generation_change"

    backward = tracker.update(
        _frame(2_050_000_000, [BOX], frame_index=0),
        calibration=_calibration(),
        generation=1,
    )
    assert backward.tracks[0].track_id == 1
    assert backward.diagnostics.last_reset_reason == "nonincreasing_timestamp"
    assert backward.diagnostics.reset_count == 3
    assert len(backward.tracks[0].trail) == 1


def test_trail_history_is_strictly_bounded() -> None:
    tracker = OnlineBoxTracker(
        TrackerConfig(min_confirmed_hits=1, trail_length=3),
        model_alias="voxel0075",
    )
    result = None
    for index in range(6):
        moved = (1.0 + index * 0.1, *BOX[1:])
        result = tracker.update(
            _frame(1_000_000_000 + index * 100_000_000, [moved], frame_index=index),
            calibration=_calibration(),
            generation=0,
        )
    assert result is not None
    assert len(result.tracks[0].trail) == 3


def test_active_state_stays_bounded_under_repeated_unmatched_detections() -> None:
    tracker = OnlineBoxTracker(
        TrackerConfig(
            min_confirmed_hits=1,
            max_missed_frames=2,
            association_distance_meters=0.1,
        ),
        model_alias="voxel0075",
    )
    maximum_active = 0
    for index in range(20):
        boxes = [
            (index * 10.0, float(row), 0.0, 4.0, 2.0, 1.5, 0.0)
            for row in range(8)
        ]
        result = tracker.update(
            _frame(1_000_000_000 + index * 100_000_000, boxes, frame_index=index),
            calibration=_calibration(),
            generation=0,
        )
        maximum_active = max(maximum_active, result.diagnostics.active_tracks)
    assert maximum_active <= 8 * 3


def test_track_ids_stay_within_marker_int32_range_and_skip_active_ids() -> None:
    tracker = OnlineBoxTracker(
        TrackerConfig(min_confirmed_hits=1, association_distance_meters=0.1),
        model_alias="voxel0075",
    )
    tracker.update(_frame(1_000_000_000, [BOX]), calibration=_calibration(), generation=0)
    tracker._next_track_id = (1 << 31) - 1
    result = tracker.update(
        _frame(
            1_100_000_000,
            [
                (100.0, *BOX[1:]),
                (200.0, *BOX[1:]),
            ],
            frame_index=1,
        ),
        calibration=_calibration(),
        generation=0,
    )
    assert [track.track_id for track in result.tracks] == [1, 2, (1 << 31) - 1]
    assert tracker._next_track_id == 3


def test_detector_failure_is_not_treated_as_an_empty_scene() -> None:
    tracker = OnlineBoxTracker(TrackerConfig(), model_alias="voxel0075")
    failed = replace(
        _frame(1_000_000_000, []),
        status="inference_failed",
        errors=(
            PlaybackErrorEvidence(
                phase="inference",
                code="synthetic",
                message="synthetic failure",
            ),
        ),
    )
    with pytest.raises(ValueError, match="valid detector frame"):
        tracker.update(failed, calibration=_calibration(), generation=0)
    assert tracker.snapshot().tracked_frames_total == 0


def test_nonfinite_detection_is_rejected_before_tracker_state_can_change() -> None:
    with pytest.raises(ValueError, match="finite"):
        _frame(
            1_000_000_000,
            [(float("nan"), 0.0, 0.0, 4.0, 2.0, 1.5, 0.0)],
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_confirmed_hits": 0}, "min_confirmed_hits"),
        ({"max_missed_frames": -1}, "max_missed_frames"),
        ({"max_time_gap_seconds": 0.0}, "max_time_gap_seconds"),
        ({"association_distance_meters": float("nan")}, "association_distance"),
        ({"position_smoothing": 1.1}, "position_smoothing"),
        ({"trail_length": 0}, "trail_length"),
    ],
)
def test_invalid_tracker_configuration_is_rejected(kwargs, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        TrackerConfig(**kwargs)
