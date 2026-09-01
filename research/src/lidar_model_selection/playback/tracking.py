"""Deterministic, bounded online tracking for base-frame detection boxes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite, pi
from numbers import Integral, Real
from time import perf_counter_ns

import numpy as np

from .box_geometry import boxes_to_base_frame
from .contracts import SessionCalibration
from .results import DetectionFrame


_SUCCESS_STATUSES = {
    "success",
    "empty_source",
    "empty_after_nonfinite_filter",
    "empty_after_range_filter",
}
_MAX_TRACK_ID = (1 << 31) - 1


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer and not a boolean")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return normalized


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer and not a boolean")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be nonnegative")
    return normalized


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number and not a boolean")
    normalized = float(value)
    if not isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return normalized


def _unit_interval(value: object, name: str) -> float:
    normalized = _positive_float(value, name)
    if normalized > 1.0:
        raise ValueError(f"{name} must be in (0, 1]")
    return normalized


def wrap_yaw(yaw: float) -> float:
    """Wrap one yaw to the half-open interval ``[-pi, pi)``."""

    if isinstance(yaw, bool) or not isinstance(yaw, Real):
        raise TypeError("yaw must be a real number and not a boolean")
    value = float(yaw)
    if not isfinite(value):
        raise ValueError("yaw must be finite")
    return _wrap_yaw_fast(value)


def _wrap_yaw_fast(yaw: float) -> float:
    return (yaw + pi) % (2.0 * pi) - pi


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    """Small, presentation-oriented tracker policy with bounded history."""

    min_confirmed_hits: int = 2
    max_missed_frames: int = 3
    max_time_gap_seconds: float = 0.75
    association_distance_meters: float = 4.0
    position_smoothing: float = 0.65
    score_smoothing: float = 0.65
    trail_length: int = 20

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "min_confirmed_hits",
            _positive_integer(self.min_confirmed_hits, "min_confirmed_hits"),
        )
        object.__setattr__(
            self,
            "max_missed_frames",
            _nonnegative_integer(self.max_missed_frames, "max_missed_frames"),
        )
        object.__setattr__(
            self,
            "max_time_gap_seconds",
            _positive_float(self.max_time_gap_seconds, "max_time_gap_seconds"),
        )
        object.__setattr__(
            self,
            "association_distance_meters",
            _positive_float(
                self.association_distance_meters,
                "association_distance_meters",
            ),
        )
        object.__setattr__(
            self,
            "position_smoothing",
            _unit_interval(self.position_smoothing, "position_smoothing"),
        )
        object.__setattr__(
            self,
            "score_smoothing",
            _unit_interval(self.score_smoothing, "score_smoothing"),
        )
        object.__setattr__(
            self,
            "trail_length",
            _positive_integer(self.trail_length, "trail_length"),
        )


@dataclass(frozen=True, slots=True)
class TrackedBox:
    """One immutable track state in ``lexus3/base_link`` coordinates."""

    track_id: int
    label: int
    box: tuple[float, float, float, float, float, float, float]
    score: float
    velocity_xyz: tuple[float, float, float]
    hits: int
    missed_frames: int
    confirmed: bool
    coasting: bool
    timestamp_ns: int
    trail: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        # The tracker constructs this public value from already validated
        # DetectionFrame arrays. Keep only structural invariants here: an
        # exhaustive finite scan of every bounded trail on every frame would
        # make materialization dominate the tracking update.
        if self.track_id <= 0 or self.hits <= 0 or self.missed_frames < 0:
            raise ValueError("track counters and ID must be valid")
        if len(self.box) != 7 or len(self.velocity_xyz) != 3:
            raise ValueError("box and velocity dimensions are invalid")
        if self.coasting and not self.confirmed:
            raise ValueError("only confirmed tracks may coast")
        if len(self.trail) == 0:
            raise ValueError("trail must contain the current position")


@dataclass(frozen=True, slots=True)
class TrackingDiagnostics:
    active_tracks: int
    confirmed_tracks: int
    tentative_tracks: int
    coasting_tracks: int
    created_tracks_total: int
    deleted_tracks_total: int
    created_tracks: int
    removed_tracks: int
    matches: int
    misses: int
    unmatched_detections: int
    unmatched_tracks: int
    reset_count: int
    last_reset_reason: str | None
    association_ms: float | None
    update_ms: float | None
    last_timestamp_ns: int | None
    last_dt_seconds: float | None
    maximum_observed_gap_seconds: float
    tracked_frames_total: int


@dataclass(frozen=True, slots=True)
class TrackedFrame:
    """Tracker output retaining run and generation provenance."""

    session_id: str
    frame_index: int
    timestamp_ns: int
    generation: int
    coordinate_frame: str
    model_alias: str
    run_id: str
    checkpoint_sha256: str
    tracks: tuple[TrackedBox, ...]
    diagnostics: TrackingDiagnostics

    @property
    def visible_tracks(self) -> tuple[TrackedBox, ...]:
        """Confirmed fresh or coasting tracks safe for presentation output."""

        return tuple(track for track in self.tracks if track.confirmed)


@dataclass(slots=True)
class _Track:
    track_id: int
    label: int
    box: np.ndarray
    score: float
    velocity: np.ndarray
    hits: int
    missed_frames: int
    confirmed: bool
    coasting: bool
    last_update_timestamp_ns: int
    last_observed_timestamp_ns: int
    trail: deque[tuple[float, float, float]]


class OnlineBoxTracker:
    """Constant-velocity prediction with deterministic gated association.

    Assignment is a globally distance-sorted greedy match.  It is deterministic
    and class-gated, and avoids adding SciPy to the declared runtime dependency
    set solely for this small single-class presentation tracker.
    """

    def __init__(self, config: TrackerConfig, *, model_alias: str) -> None:
        if not isinstance(config, TrackerConfig):
            raise TypeError("config must be a TrackerConfig")
        if not isinstance(model_alias, str) or not model_alias.strip():
            raise ValueError("model_alias must contain non-whitespace text")
        self._config = config
        self._model_alias = model_alias
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 1
        self._generation: int | None = None
        self._last_timestamp_ns: int | None = None
        self._created_total = 0
        self._deleted_total = 0
        self._reset_count = 0
        self._last_reset_reason: str | None = None
        self._maximum_gap_seconds = 0.0
        self._last_matches = 0
        self._last_created = 0
        self._last_removed = 0
        self._last_unmatched_detections = 0
        self._last_unmatched_tracks = 0
        self._last_association_ms: float | None = None
        self._last_update_ms: float | None = None
        self._last_dt_seconds: float | None = None
        self._tracked_frames_total = 0

    @property
    def config(self) -> TrackerConfig:
        return self._config

    def reset(self, *, reason: str, generation: int | None = None) -> None:
        """Clear temporal state, bounded trails, and deterministic ID state."""

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reset reason must contain non-whitespace text")
        if generation is not None:
            generation = _nonnegative_integer(generation, "generation")
        self._deleted_total += len(self._tracks)
        self._tracks.clear()
        self._next_track_id = 1
        self._last_timestamp_ns = None
        self._last_matches = 0
        self._last_created = 0
        self._last_removed = 0
        self._last_unmatched_detections = 0
        self._last_unmatched_tracks = 0
        self._last_association_ms = None
        self._last_update_ms = None
        self._last_dt_seconds = None
        self._reset_count += 1
        self._last_reset_reason = reason
        if generation is not None:
            self._generation = generation

    def snapshot(self) -> TrackingDiagnostics:
        tracks = tuple(self._tracks.values())
        confirmed = sum(track.confirmed for track in tracks)
        coasting = sum(track.coasting for track in tracks)
        return TrackingDiagnostics(
            active_tracks=len(tracks),
            confirmed_tracks=confirmed,
            tentative_tracks=len(tracks) - confirmed,
            coasting_tracks=coasting,
            created_tracks_total=self._created_total,
            deleted_tracks_total=self._deleted_total,
            created_tracks=self._last_created,
            removed_tracks=self._last_removed,
            matches=self._last_matches,
            misses=self._last_unmatched_tracks,
            unmatched_detections=self._last_unmatched_detections,
            unmatched_tracks=self._last_unmatched_tracks,
            reset_count=self._reset_count,
            last_reset_reason=self._last_reset_reason,
            association_ms=self._last_association_ms,
            update_ms=self._last_update_ms,
            last_timestamp_ns=self._last_timestamp_ns,
            last_dt_seconds=self._last_dt_seconds,
            maximum_observed_gap_seconds=self._maximum_gap_seconds,
            tracked_frames_total=self._tracked_frames_total,
        )

    def update(
        self,
        frame: DetectionFrame,
        *,
        calibration: SessionCalibration,
        generation: int,
    ) -> TrackedFrame:
        """Advance state using one valid detector outcome and acquisition time."""

        started_ns = perf_counter_ns()
        if not isinstance(frame, DetectionFrame):
            raise TypeError("frame must be a DetectionFrame")
        if frame.model_alias != self._model_alias:
            raise ValueError("detection model identity does not match tracker identity")
        if frame.status not in _SUCCESS_STATUSES:
            raise ValueError("tracker update requires a valid detector frame outcome")
        if frame.timestamp_ns is None:
            raise ValueError("tracker update requires an acquisition timestamp")
        if frame.coordinate_frame != "lidar":
            raise ValueError("tracker input boxes must use the canonical lidar frame")
        checked_generation = _nonnegative_integer(generation, "generation")
        timestamp_ns = frame.timestamp_ns

        if self._generation is None:
            self._generation = checked_generation
        elif checked_generation != self._generation:
            self.reset(
                reason="generation_change",
                generation=checked_generation,
            )

        gap_seconds = 0.0
        observed_dt_seconds: float | None = None
        if self._last_timestamp_ns is not None:
            if timestamp_ns <= self._last_timestamp_ns:
                self.reset(
                    reason="nonincreasing_timestamp",
                    generation=checked_generation,
                )
            else:
                gap_seconds = (
                    timestamp_ns - self._last_timestamp_ns
                ) / 1_000_000_000.0
                observed_dt_seconds = gap_seconds
                self._maximum_gap_seconds = max(
                    self._maximum_gap_seconds,
                    gap_seconds,
                )
                if gap_seconds > self._config.max_time_gap_seconds:
                    self.reset(
                        reason="excessive_time_gap",
                        generation=checked_generation,
                    )
                    gap_seconds = 0.0
                    observed_dt_seconds = None

        boxes = boxes_to_base_frame(frame.boxes, calibration)
        labels = np.asarray(frame.labels, dtype=np.int64)
        scores = np.asarray(frame.scores, dtype=np.float64)
        ordered_ids = tuple(sorted(self._tracks))
        ordered_tracks = [self._tracks[track_id] for track_id in ordered_ids]
        predicted_boxes = self._predicted_boxes(ordered_tracks, gap_seconds)
        association_started_ns = perf_counter_ns()
        matches = self._associate(
            ordered_ids,
            ordered_tracks,
            predicted_boxes,
            boxes,
            labels,
        )
        self._last_association_ms = max(
            0.0,
            (perf_counter_ns() - association_started_ns) / 1_000_000.0,
        )
        matched_track_rows = {track_row for track_row, _ in matches}
        matched_detection_indices = {detection for _, detection in matches}

        self._update_matches(
            ordered_tracks,
            predicted_boxes,
            boxes,
            scores,
            matches,
            timestamp_ns,
            gap_seconds,
        )

        deleted_ids: list[int] = []
        for track_row, track in enumerate(ordered_tracks):
            if track_row in matched_track_rows:
                continue
            track.box = np.array(
                predicted_boxes[track_row],
                dtype=np.float64,
                order="C",
                copy=True,
            )
            track.last_update_timestamp_ns = timestamp_ns
            track.missed_frames += 1
            track.coasting = track.confirmed
            track.trail.append(tuple(track.box[:3].tolist()))
            observed_gap = (
                timestamp_ns - track.last_observed_timestamp_ns
            ) / 1_000_000_000.0
            if (
                not track.confirmed
                or track.missed_frames > self._config.max_missed_frames
                or observed_gap > self._config.max_time_gap_seconds
            ):
                deleted_ids.append(track.track_id)
        for track_id in deleted_ids:
            del self._tracks[track_id]
        self._deleted_total += len(deleted_ids)

        for detection_index in range(frame.detection_count):
            if detection_index in matched_detection_indices:
                continue
            self._create_track(
                boxes[detection_index],
                float(scores[detection_index]),
                int(labels[detection_index]),
                timestamp_ns,
            )

        self._last_timestamp_ns = timestamp_ns
        self._last_dt_seconds = observed_dt_seconds
        self._last_matches = len(matches)
        self._last_created = frame.detection_count - len(matched_detection_indices)
        self._last_removed = len(deleted_ids)
        self._last_unmatched_detections = (
            frame.detection_count - len(matched_detection_indices)
        )
        self._last_unmatched_tracks = len(ordered_tracks) - len(matched_track_rows)
        self._last_update_ms = max(
            0.0,
            (perf_counter_ns() - started_ns) / 1_000_000.0,
        )
        self._tracked_frames_total += 1
        tracks = tuple(
            self._public_track(self._tracks[track_id])
            for track_id in sorted(self._tracks)
        )
        return TrackedFrame(
            session_id=frame.session_id,
            frame_index=frame.frame_index,
            timestamp_ns=timestamp_ns,
            generation=checked_generation,
            coordinate_frame=calibration.parent_frame_id,
            model_alias=frame.model_alias,
            run_id=frame.run_id,
            checkpoint_sha256=frame.checkpoint_sha256,
            tracks=tracks,
            diagnostics=self.snapshot(),
        )

    def _predicted_boxes(
        self,
        tracks: list[_Track],
        elapsed_seconds: float,
    ) -> np.ndarray:
        if not tracks:
            return np.empty((0, 7), dtype=np.float64)
        predicted = np.asarray([track.box for track in tracks], dtype=np.float64)
        velocities = np.asarray(
            [track.velocity for track in tracks],
            dtype=np.float64,
        )
        predicted[:, :3] += velocities * elapsed_seconds
        predicted[:, 6] = (predicted[:, 6] + pi) % (2.0 * pi) - pi
        return predicted

    def _associate(
        self,
        track_ids: tuple[int, ...],
        tracks: list[_Track],
        predicted_boxes: np.ndarray,
        detections: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[tuple[int, int], ...]:
        if not tracks or detections.shape[0] == 0:
            return ()
        delta = predicted_boxes[:, np.newaxis, :2] - detections[np.newaxis, :, :2]
        distance_squared = np.einsum("tni,tni->tn", delta, delta)
        track_labels = np.asarray([track.label for track in tracks], dtype=np.int64)
        compatible = track_labels[:, np.newaxis] == labels[np.newaxis, :]
        gated = compatible & (
            distance_squared <= self._config.association_distance_meters**2
        )
        track_rows, detection_indices = np.nonzero(gated)
        if track_rows.size == 0:
            return ()
        candidate_distances = distance_squared[track_rows, detection_indices]
        candidate_track_ids = np.asarray(track_ids, dtype=np.int64)[track_rows]
        order = np.lexsort(
            (detection_indices, candidate_track_ids, candidate_distances)
        )
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        matches: list[tuple[int, int]] = []
        maximum_matches = min(len(tracks), detections.shape[0])
        for candidate in order:
            track_row = int(track_rows[candidate])
            detection_index = int(detection_indices[candidate])
            if track_row in used_tracks or detection_index in used_detections:
                continue
            used_tracks.add(track_row)
            used_detections.add(detection_index)
            matches.append((track_row, detection_index))
            if len(matches) == maximum_matches:
                break
        matches.sort(key=lambda value: track_ids[value[0]])
        return tuple(matches)

    def _update_matches(
        self,
        tracks: list[_Track],
        predicted_boxes: np.ndarray,
        detections: np.ndarray,
        scores: np.ndarray,
        matches: tuple[tuple[int, int], ...],
        timestamp_ns: int,
        elapsed_seconds: float,
    ) -> None:
        if not matches:
            return
        track_rows = np.fromiter(
            (track_row for track_row, _ in matches),
            dtype=np.int64,
            count=len(matches),
        )
        detection_indices = np.fromiter(
            (detection_index for _, detection_index in matches),
            dtype=np.int64,
            count=len(matches),
        )
        matched_tracks = [tracks[index] for index in track_rows]
        old_boxes = np.asarray(
            [track.box for track in matched_tracks],
            dtype=np.float64,
        )
        old_velocities = np.asarray(
            [track.velocity for track in matched_tracks],
            dtype=np.float64,
        )
        prediction = predicted_boxes[track_rows]
        detection = detections[detection_indices]
        if elapsed_seconds <= 0.0:
            raise RuntimeError("matched tracks require a positive elapsed time")
        alpha = self._config.position_smoothing
        velocities = (1.0 - alpha) * old_velocities + alpha * (
            detection[:, :3] - old_boxes[:, :3]
        ) / elapsed_seconds
        updated = np.array(prediction, dtype=np.float64, order="C", copy=True)
        updated[:, :3] = (
            (1.0 - alpha) * prediction[:, :3] + alpha * detection[:, :3]
        )
        updated[:, 3:6] = (
            (1.0 - alpha) * prediction[:, 3:6] + alpha * detection[:, 3:6]
        )
        yaw_residual = (detection[:, 6] - prediction[:, 6] + pi) % (
            2.0 * pi
        ) - pi
        updated[:, 6] = (
            prediction[:, 6] + alpha * yaw_residual + pi
        ) % (2.0 * pi) - pi
        score_alpha = self._config.score_smoothing
        updated_scores = score_alpha * scores[detection_indices] + (
            1.0 - score_alpha
        ) * np.asarray([track.score for track in matched_tracks])
        for index, track in enumerate(matched_tracks):
            track.box[:] = updated[index]
            track.velocity[:] = velocities[index]
            track.score = float(updated_scores[index])
            track.hits += 1
            track.missed_frames = 0
            track.confirmed = track.hits >= self._config.min_confirmed_hits
            track.coasting = False
            track.last_update_timestamp_ns = timestamp_ns
            track.last_observed_timestamp_ns = timestamp_ns
            track.trail.append(tuple(track.box[:3].tolist()))

    def _create_track(
        self,
        box: np.ndarray,
        score: float,
        label: int,
        timestamp_ns: int,
    ) -> None:
        track_id = self._allocate_track_id()
        position = tuple(box[:3].tolist())
        self._tracks[track_id] = _Track(
            track_id=track_id,
            label=label,
            box=np.array(box, dtype=np.float64, order="C", copy=True),
            score=score,
            velocity=np.zeros((3,), dtype=np.float64),
            hits=1,
            missed_frames=0,
            confirmed=self._config.min_confirmed_hits == 1,
            coasting=False,
            last_update_timestamp_ns=timestamp_ns,
            last_observed_timestamp_ns=timestamp_ns,
            trail=deque((position,), maxlen=self._config.trail_length),
        )
        self._created_total += 1

    def _allocate_track_id(self) -> int:
        """Allocate within ROS Marker's positive signed-int32 ID range."""

        candidate = self._next_track_id
        for _ in range(len(self._tracks) + 1):
            if candidate not in self._tracks:
                self._next_track_id = (
                    1 if candidate == _MAX_TRACK_ID else candidate + 1
                )
                return candidate
            candidate = 1 if candidate == _MAX_TRACK_ID else candidate + 1
        raise RuntimeError("bounded track ID space is exhausted")

    @staticmethod
    def _public_track(track: _Track) -> TrackedBox:
        return TrackedBox(
            track_id=track.track_id,
            label=track.label,
            box=tuple(track.box.tolist()),
            score=float(track.score),
            velocity_xyz=tuple(track.velocity.tolist()),
            hits=track.hits,
            missed_frames=track.missed_frames,
            confirmed=track.confirmed,
            coasting=track.coasting,
            timestamp_ns=track.last_update_timestamp_ns,
            trail=tuple(track.trail),
        )
