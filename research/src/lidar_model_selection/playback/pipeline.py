from dataclasses import dataclass
from math import floor, isfinite
from numbers import Integral, Real
from time import perf_counter, sleep
from typing import Callable, Iterator, Protocol

from .contracts import FrameSourceError, PointCloudFrame, RecordingSequence
from .detector import Mmdet3dDetector
from .frame_source import LidarFrame
from .results import (
    DetectionFrame,
    FrameResult,
    PlaybackErrorEvidence,
    empty_detection_arrays,
)


@dataclass(frozen=True, slots=True)
class ProcessedFrame:
    result: FrameResult
    total_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.result, FrameResult):
            raise TypeError("result must be a FrameResult")
        if isinstance(self.total_ms, bool) or not isinstance(self.total_ms, Real):
            raise TypeError("total_ms must be a real number and not a boolean")
        if not isfinite(self.total_ms):
            raise ValueError("total_ms must be finite")
        if self.total_ms < 0:
            raise ValueError("total_ms must be greater than or equal to zero")


class SequentialDetectionPipeline:
    def __init__(self, detector: Mmdet3dDetector) -> None:
        if not isinstance(detector, Mmdet3dDetector):
            raise TypeError("detector must be an Mmdet3dDetector")
        self._detector = detector

    def run(
        self,
        frames: tuple[LidarFrame, ...],
    ) -> tuple[ProcessedFrame, ...]:
        if not isinstance(frames, tuple):
            raise TypeError("frames must be a tuple")
        if any(not isinstance(frame, LidarFrame) for frame in frames):
            raise TypeError("every frames member must be a LidarFrame")
        if not frames:
            return ()

        processed_frames = []

        for frame in frames:
            start_time = perf_counter()
            result = self._detector.detect(frame)
            end_time = perf_counter()

            processed_frames.append(
                ProcessedFrame(
                    result=result,
                    total_ms=(end_time - start_time) * 1000.0,
                )
            )

        return tuple(processed_frames)


class StreamingDetector(Protocol):
    """Detector surface consumed by the session processor."""

    @property
    def identity(self): ...

    def detect(self, frame: PointCloudFrame) -> DetectionFrame: ...


def _require_nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer and not a boolean")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be greater than or equal to zero")
    return normalized


def _require_playback_rate(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("playback_rate must be a real number and not a boolean")
    normalized = float(value)
    if not isfinite(normalized) or normalized <= 0.0:
        raise ValueError("playback_rate must be finite and strictly positive")
    return normalized


class BlockPacer:
    """Absolute-timeline, no-drop pacing driven by capture timestamps."""

    def __init__(
        self,
        playback_rate: float = 1.0,
        *,
        clock: Callable[[], float] = perf_counter,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        self._playback_rate = _require_playback_rate(playback_rate)
        if not callable(clock):
            raise TypeError("clock must be callable")
        if not callable(sleeper):
            raise TypeError("sleeper must be callable")
        self._clock = clock
        self._sleeper = sleeper
        self.reset()

    @property
    def playback_rate(self) -> float:
        return self._playback_rate

    @property
    def lag_ms(self) -> float:
        return self._lag_ms

    def reset(self) -> None:
        self._capture_anchor_ns: int | None = None
        self._previous_timestamp_ns: int | None = None
        self._wall_anchor_s: float | None = None
        self._current_target_s: float | None = None
        self._lag_ms = 0.0

    def wait(self, timestamp_ns: int) -> float:
        timestamp = _require_nonnegative_integer(
            timestamp_ns,
            name="timestamp_ns",
        )
        now = float(self._clock())
        if not isfinite(now):
            raise ValueError("clock must return a finite value")

        if self._capture_anchor_ns is None:
            self._capture_anchor_ns = timestamp
            self._previous_timestamp_ns = timestamp
            self._wall_anchor_s = now
            self._current_target_s = now
            self._lag_ms = 0.0
            return 0.0

        assert self._previous_timestamp_ns is not None
        assert self._wall_anchor_s is not None
        if timestamp <= self._previous_timestamp_ns:
            raise ValueError("pacing timestamps must be strictly increasing")

        target_s = self._wall_anchor_s + (
            (timestamp - self._capture_anchor_ns)
            / 1_000_000_000.0
            / self._playback_rate
        )
        if now < target_s:
            self._sleeper(target_s - now)
            now = float(self._clock())
            if not isfinite(now):
                raise ValueError("clock must return a finite value")

        self._previous_timestamp_ns = timestamp
        self._current_target_s = target_s
        # This is accumulated lag against the absolute capture-time schedule,
        # not a sum of overlapping per-frame delays.
        self._lag_ms = max(0.0, (now - target_s) * 1000.0)
        return self._lag_ms

    def complete(self) -> float:
        """Record completion lag for the current frame's absolute deadline."""

        if self._current_target_s is None:
            raise RuntimeError("wait(timestamp_ns) must precede complete()")
        now = float(self._clock())
        if not isfinite(now):
            raise ValueError("clock must return a finite value")
        self._lag_ms = max(0.0, (now - self._current_target_s) * 1000.0)
        return self._lag_ms


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """Session-scoped counters and latency aggregates; never cross-session."""

    session_id: str
    frame_count: int
    success_count: int
    detection_count: int
    error_count: int
    empty_frame_count: int
    dropped_nonfinite_point_count: int
    decode_ms_mean: float
    decode_ms_p50: float
    decode_ms_p95: float
    detector_ms_mean: float
    detector_ms_p50: float
    detector_ms_p95: float
    frame_processing_ms_mean: float
    frame_processing_ms_p50: float
    frame_processing_ms_p95: float
    processing_fps: float | None
    final_pacing_lag_ms: float


class SessionProcessor:
    """Stream one independent recording session through one stateless model."""

    def __init__(
        self,
        detector: StreamingDetector,
        *,
        playback_rate: float = 1.0,
        on_frame_error: str = "stop",
        clock: Callable[[], float] = perf_counter,
        sleeper: Callable[[float], None] = sleep,
        observer: Callable[[PointCloudFrame, DetectionFrame], bool | None]
        | None = None,
    ) -> None:
        if not callable(getattr(detector, "detect", None)):
            raise TypeError("detector must provide detect(frame)")
        if on_frame_error not in {"stop", "continue"}:
            raise ValueError("on_frame_error must be 'stop' or 'continue'")
        if observer is not None and not callable(observer):
            raise TypeError("observer must be callable or None")

        self._detector = detector
        self._on_frame_error = on_frame_error
        self._observer = observer
        self._pacer = BlockPacer(
            playback_rate,
            clock=clock,
            sleeper=sleeper,
        )
        self._active = False
        self._reset_metrics("")

    @property
    def summary(self) -> SessionSummary:
        decode_mean = (
            sum(self._decode_timings) / len(self._decode_timings)
            if self._decode_timings
            else 0.0
        )
        detector_mean = (
            sum(self._detector_timings) / len(self._detector_timings)
            if self._detector_timings
            else 0.0
        )
        processing_mean = (
            sum(self._processing_timings) / len(self._processing_timings)
            if self._processing_timings
            else 0.0
        )
        return SessionSummary(
            session_id=self._session_id,
            frame_count=self._frame_count,
            success_count=self._success_count,
            detection_count=self._detection_count,
            error_count=self._error_count,
            empty_frame_count=self._empty_frame_count,
            dropped_nonfinite_point_count=self._dropped_nonfinite_count,
            decode_ms_mean=decode_mean,
            decode_ms_p50=_percentile(self._decode_timings, 0.50),
            decode_ms_p95=_percentile(self._decode_timings, 0.95),
            detector_ms_mean=detector_mean,
            detector_ms_p50=_percentile(self._detector_timings, 0.50),
            detector_ms_p95=_percentile(self._detector_timings, 0.95),
            frame_processing_ms_mean=processing_mean,
            frame_processing_ms_p50=_percentile(
                self._processing_timings,
                0.50,
            ),
            frame_processing_ms_p95=_percentile(
                self._processing_timings,
                0.95,
            ),
            processing_fps=(
                1000.0 / processing_mean if processing_mean > 0.0 else None
            ),
            final_pacing_lag_ms=self._pacer.lag_ms,
        )

    def _reset_metrics(self, session_id: str) -> None:
        self._session_id = session_id
        self._frame_count = 0
        self._success_count = 0
        self._detection_count = 0
        self._error_count = 0
        self._empty_frame_count = 0
        self._dropped_nonfinite_count = 0
        self._decode_timings: list[float] = []
        self._detector_timings: list[float] = []
        self._processing_timings: list[float] = []
        self._pacer.reset()

    def _record(self, result: DetectionFrame) -> None:
        self._frame_count += 1
        # Aggregates describe every published frame outcome. Source failures
        # retain measured decode/processing time and contribute a zero detector
        # duration, keeping the timing denominator equal to ``frame_count``.
        self._decode_timings.append(float(result.decode_ms))
        self._detector_timings.append(float(result.detector_ms))
        self._processing_timings.append(float(result.frame_processing_ms))
        if result.errors:
            self._error_count += 1
            return
        self._success_count += 1
        self._detection_count += result.detection_count
        assert result.dropped_nonfinite_count is not None
        self._dropped_nonfinite_count += result.dropped_nonfinite_count
        if result.status.startswith("empty_"):
            self._empty_frame_count += 1

    def _source_error_result(self, error: FrameSourceError) -> DetectionFrame:
        evidence = error.evidence
        if evidence.frame_index is None:
            # A session-level failure has no safe frame identity and therefore
            # cannot be converted into a fabricated frame result.
            raise error
        identity = self._detector.identity
        boxes, scores, labels = empty_detection_arrays()
        decode_ms = float(evidence.decode_ms)
        return DetectionFrame(
            session_id=evidence.session_id,
            frame_index=evidence.frame_index,
            timestamp_ns=evidence.header_timestamp_ns,
            storage_timestamp_ns=evidence.storage_timestamp_ns,
            source_frame_id=None,
            coordinate_frame=None,
            source_key=evidence.source_key,
            model_alias=identity.model_alias,
            run_id=identity.run_id,
            config_sha256=identity.config_sha256,
            checkpoint_path=identity.checkpoint_reference,
            checkpoint_sha256=identity.checkpoint_sha256,
            checkpoint_size_bytes=identity.checkpoint_size_bytes,
            source_point_count=None,
            dropped_nonfinite_count=None,
            input_point_count=None,
            in_range_point_count=None,
            detection_count=0,
            status="frame_error",
            boxes=boxes,
            scores=scores,
            labels=labels,
            decode_ms=decode_ms,
            detector_ms=0.0,
            frame_processing_ms=decode_ms,
            pacing_lag_ms=self._pacer.lag_ms,
            errors=(
                PlaybackErrorEvidence(
                    phase="source",
                    code=evidence.code,
                    message=evidence.message,
                ),
            ),
        )

    def process(
        self,
        sequence: RecordingSequence,
        *,
        start_index: int = 0,
        max_frames: int | None = None,
    ) -> Iterator[DetectionFrame]:
        """Yield each frame immediately; retain only scalar session metrics."""

        start = _require_nonnegative_integer(start_index, name="start_index")
        if max_frames is not None:
            maximum = _require_nonnegative_integer(
                max_frames,
                name="max_frames",
            )
        else:
            maximum = None
        if not isinstance(sequence.session_id, str) or not sequence.session_id.strip():
            raise ValueError("sequence.session_id must contain non-whitespace text")
        if self._active:
            raise RuntimeError("a session processor iterator is already active")

        self._active = True
        self._reset_metrics(sequence.session_id)
        emitted = 0
        iterator = iter(sequence.iter_frames(start_index=start))
        try:
            while maximum is None or emitted < maximum:
                try:
                    frame = next(iterator)
                except StopIteration:
                    break
                except FrameSourceError as error:
                    if self._on_frame_error == "stop" or not error.evidence.recoverable:
                        raise
                    if error.evidence.session_id != sequence.session_id:
                        raise ValueError(
                            "source error evidence belongs to a different session"
                        ) from error
                    # The MCAP source reads the skipped prefix to preserve
                    # session-wide ordering validation. Under explicit continue
                    # policy, recoverable failures in that prefix are outside
                    # the requested output window and therefore neither publish
                    # an outcome nor consume ``max_frames``.
                    if (
                        error.evidence.frame_index is not None
                        and error.evidence.frame_index < start
                    ):
                        continue
                    result = self._source_error_result(error)
                    self._record(result)
                    emitted += 1
                    yield result
                    continue

                if frame.session_id != sequence.session_id:
                    raise ValueError("source frame belongs to a different session")
                self._pacer.wait(frame.timestamp_ns)
                detected = self._detector.detect(frame)
                if (
                    detected.session_id != frame.session_id
                    or detected.frame_index != frame.frame_index
                    or detected.timestamp_ns != frame.timestamp_ns
                ):
                    raise ValueError(
                        "detector result identity does not match the source frame"
                    )
                # Sample lag after detector completion, but before the observer,
                # so it represents accumulated processing backlog without
                # charging visualization time to the current frame.
                lag_ms = self._pacer.complete()
                result = detected.with_processing_timing(
                    frame_processing_ms=frame.decode_ms + detected.detector_ms,
                    pacing_lag_ms=lag_ms,
                )
                self._record(result)
                emitted += 1
                keep_going = (
                    True
                    if self._observer is None
                    else self._observer(frame, result) is not False
                )
                yield result
                if not keep_going:
                    break
        finally:
            self._active = False
