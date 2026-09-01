from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy
import pytest

from lidar_model_selection.playback.contracts import (
    FrameErrorEvidence,
    FrameSourceError,
    PointCloudFrame,
)
from lidar_model_selection.playback.pipeline import BlockPacer, SessionProcessor
from lidar_model_selection.playback.results import DetectionFrame


def _frame(
    session_id: str,
    frame_index: int,
    timestamp_ns: int,
    *,
    decode_ms: float = 1.0,
) -> PointCloudFrame:
    return PointCloudFrame(
        session_id=session_id,
        frame_index=frame_index,
        timestamp_ns=timestamp_ns,
        storage_timestamp_ns=timestamp_ns + 10,
        source_frame_id="lexus3/os_center",
        coordinate_frame="lidar",
        source_key=f"session_0.mcap:/lexus3/os_center/points#{frame_index}",
        points=numpy.asarray([[1.0, 0.0, 0.0, 0.5]], dtype=numpy.float32),
        source_point_count=1,
        dropped_nonfinite_count=0,
        decode_ms=decode_ms,
    )


@dataclass
class _Sequence:
    session_id: str
    frames: tuple[PointCloudFrame, ...]

    def __post_init__(self) -> None:
        self.starts: list[int] = []

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def iter_frames(self, start_index: int = 0):
        self.starts.append(start_index)
        yield from self.frames[start_index:]


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.sleeps.append(duration)
        self.now += duration


class _Detector:
    def __init__(
        self,
        *,
        detector_ms: float = 2.0,
        clock: _FakeClock | None = None,
        elapsed_s: float = 0.0,
    ) -> None:
        self.calls: list[int] = []
        self.detector_ms = detector_ms
        self.clock = clock
        self.elapsed_s = elapsed_s
        self.identity = SimpleNamespace(
            model_alias="voxel0075",
            run_id="run-id",
            config_sha256="a" * 64,
            checkpoint_reference="training/selected.pth",
            checkpoint_sha256="b" * 64,
            checkpoint_size_bytes=123,
        )

    def detect(self, frame: PointCloudFrame) -> DetectionFrame:
        self.calls.append(frame.frame_index)
        if self.clock is not None:
            self.clock.now += self.elapsed_s
        boxes = numpy.asarray(
            [[1.0, 2.0, -1.0, 4.0, 2.0, 1.5, 0.0]],
            dtype=numpy.float32,
        )
        scores = numpy.asarray([0.75], dtype=numpy.float32)
        labels = numpy.asarray([0], dtype=numpy.int64)
        for values in (boxes, scores, labels):
            values.setflags(write=False)
        return DetectionFrame(
            session_id=frame.session_id,
            frame_index=frame.frame_index,
            timestamp_ns=frame.timestamp_ns,
            storage_timestamp_ns=frame.storage_timestamp_ns,
            source_frame_id=frame.source_frame_id,
            coordinate_frame=frame.coordinate_frame,
            source_key=frame.source_key,
            model_alias=self.identity.model_alias,
            run_id=self.identity.run_id,
            config_sha256=self.identity.config_sha256,
            checkpoint_path=self.identity.checkpoint_reference,
            checkpoint_sha256=self.identity.checkpoint_sha256,
            checkpoint_size_bytes=self.identity.checkpoint_size_bytes,
            source_point_count=frame.source_point_count,
            dropped_nonfinite_count=frame.dropped_nonfinite_count,
            input_point_count=frame.normalized_point_count,
            in_range_point_count=frame.normalized_point_count,
            detection_count=1,
            status="success",
            boxes=boxes,
            scores=scores,
            labels=labels,
            decode_ms=frame.decode_ms,
            detector_ms=self.detector_ms,
            frame_processing_ms=frame.decode_ms + self.detector_ms,
        )


class _ErrorIterator:
    def __init__(self, frame: PointCloudFrame, *, recoverable: bool) -> None:
        self._frame = frame
        self._recoverable = recoverable
        self._state = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._state == 0:
            self._state = 1
            raise FrameSourceError(
                FrameErrorEvidence(
                    code="truncated_pointcloud_payload",
                    message="declared row ended early",
                    session_id=self._frame.session_id,
                    frame_index=0,
                    header_timestamp_ns=None,
                    storage_timestamp_ns=999,
                    source_key="recording.mcap:/points[0]",
                    recoverable=self._recoverable,
                    decode_ms=5.0,
                )
            )
        if self._state == 1:
            self._state = 2
            return self._frame
        raise StopIteration


class _ErrorSequence:
    def __init__(self, *, recoverable: bool = True) -> None:
        self.session_id = "errors"
        self.frame_count = 2
        self._recoverable = recoverable

    def iter_frames(self, start_index: int = 0):
        assert start_index == 0
        return _ErrorIterator(
            _frame("errors", 1, 2_000_000_000),
            recoverable=self._recoverable,
        )


class _PrefixErrorSequence:
    session_id = "prefix-errors"
    frame_count = 3

    def iter_frames(self, start_index: int = 0):
        assert start_index == 2
        return _ErrorIterator(
            _frame("prefix-errors", 2, 3_000_000_000),
            recoverable=True,
        )


def test_session_processor_is_lazy_streaming_and_honors_start_and_max() -> None:
    frames = tuple(_frame("one", index, 1_000 + index) for index in range(4))
    sequence = _Sequence("one", frames)
    detector = _Detector()
    processor = SessionProcessor(detector, playback_rate=1_000_000_000.0)

    stream = processor.process(sequence, start_index=1, max_frames=2)
    assert detector.calls == []
    first = next(stream)
    assert first.frame_index == 1
    assert detector.calls == [1]
    assert [result.frame_index for result in stream] == [2]
    assert detector.calls == [1, 2]
    assert sequence.starts == [1]
    assert processor.summary.frame_count == 2
    assert processor.summary.detection_count == 2
    assert processor.summary.error_count == 0


def test_session_metrics_and_pacing_reset_between_sessions() -> None:
    clock = _FakeClock()
    detector = _Detector(clock=clock)
    processor = SessionProcessor(
        detector,
        playback_rate=2.0,
        clock=clock,
        sleeper=clock.sleep,
    )
    first = _Sequence(
        "first",
        (_frame("first", 0, 1_000_000_000), _frame("first", 1, 2_000_000_000)),
    )
    second = _Sequence("second", (_frame("second", 0, 9_000_000_000),))

    assert len(list(processor.process(first))) == 2
    assert clock.sleeps == pytest.approx([0.5])
    assert processor.summary.session_id == "first"
    assert processor.summary.frame_count == 2

    assert len(list(processor.process(second))) == 1
    assert clock.sleeps == pytest.approx([0.5])
    assert processor.summary.session_id == "second"
    assert processor.summary.frame_count == 1
    assert processor.summary.detection_count == 1
    assert processor.summary.final_pacing_lag_ms == 0.0


def test_block_pacer_reports_absolute_accumulated_lag_without_drops() -> None:
    clock = _FakeClock()
    pacer = BlockPacer(2.0, clock=clock, sleeper=clock.sleep)

    assert pacer.wait(1_000_000_000) == 0.0
    clock.now += 0.8
    assert pacer.wait(2_000_000_000) == pytest.approx(300.0)
    clock.now += 0.8
    assert pacer.wait(3_000_000_000) == pytest.approx(600.0)
    assert clock.sleeps == []


def test_session_lag_is_sampled_after_processing_and_before_observer() -> None:
    clock = _FakeClock()
    detector = _Detector(clock=clock, elapsed_s=2.0)
    observed: list[float] = []

    def observer(_frame: PointCloudFrame, result: DetectionFrame) -> bool:
        observed.append(result.pacing_lag_ms)
        clock.now += 10.0
        return True

    processor = SessionProcessor(
        detector,
        playback_rate=1.0,
        clock=clock,
        sleeper=clock.sleep,
        observer=observer,
    )
    sequence = _Sequence(
        "lag",
        (
            _frame("lag", 0, 1_000_000_000),
            _frame("lag", 1, 2_000_000_000),
        ),
    )

    results = list(processor.process(sequence))

    assert [result.pacing_lag_ms for result in results] == pytest.approx(
        [2_000.0, 13_000.0]
    )
    assert observed == pytest.approx([2_000.0, 13_000.0])
    assert processor.summary.final_pacing_lag_ms == pytest.approx(13_000.0)


@pytest.mark.parametrize("value", (0.0, -1.0, float("inf"), float("nan")))
def test_playback_rate_must_be_finite_and_strictly_positive(value: float) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        BlockPacer(value)


def test_exact_timing_summary_percentiles_and_observer_exclusion() -> None:
    clock = _FakeClock()
    detector = _Detector(detector_ms=2.0, clock=clock)

    def observer(_frame: PointCloudFrame, _result: DetectionFrame) -> bool:
        clock.now += 10.0
        return True

    processor = SessionProcessor(
        detector,
        playback_rate=1.0,
        clock=clock,
        sleeper=clock.sleep,
        observer=observer,
    )
    sequence = _Sequence(
        "timing",
        tuple(
            _frame("timing", index, 1_000_000_000 + index, decode_ms=float(index + 1))
            for index in range(3)
        ),
    )

    results = list(processor.process(sequence))
    summary = processor.summary

    assert [result.frame_processing_ms for result in results] == [3.0, 4.0, 5.0]
    assert summary.decode_ms_mean == 2.0
    assert summary.decode_ms_p50 == 2.0
    assert summary.decode_ms_p95 == pytest.approx(2.9)
    assert summary.detector_ms_mean == 2.0
    assert summary.frame_processing_ms_mean == 4.0
    assert summary.frame_processing_ms_p50 == 4.0
    assert summary.frame_processing_ms_p95 == pytest.approx(4.9)
    # Rendering/observer wall time changes pacing lag, never processing timing.
    assert summary.final_pacing_lag_ms > 0.0


def test_recoverable_source_error_can_continue_without_fabricated_evidence() -> None:
    detector = _Detector()
    processor = SessionProcessor(
        detector,
        playback_rate=1.0,
        on_frame_error="continue",
    )

    results = list(processor.process(_ErrorSequence()))

    assert [result.status for result in results] == ["frame_error", "success"]
    error = results[0]
    assert error.timestamp_ns is None
    assert error.source_frame_id is None
    assert error.coordinate_frame is None
    assert error.source_point_count is None
    assert error.storage_timestamp_ns == 999
    assert error.decode_ms == 5.0
    assert error.errors[0].code == "truncated_pointcloud_payload"
    assert detector.calls == [1]
    assert processor.summary.frame_count == 2
    assert processor.summary.error_count == 1
    assert processor.summary.success_count == 1
    assert processor.summary.decode_ms_mean == 3.0
    assert processor.summary.detector_ms_mean == 1.0
    assert processor.summary.frame_processing_ms_mean == 4.0


def test_default_stop_and_nonrecoverable_errors_never_continue() -> None:
    detector = _Detector()
    stopping = SessionProcessor(detector)
    with pytest.raises(FrameSourceError, match="truncated"):
        next(stopping.process(_ErrorSequence()))
    assert detector.calls == []

    continuing = SessionProcessor(detector, on_frame_error="continue")
    with pytest.raises(FrameSourceError, match="truncated"):
        next(continuing.process(_ErrorSequence(recoverable=False)))
    assert detector.calls == []


def test_continue_does_not_emit_or_count_recoverable_prefix_errors() -> None:
    detector = _Detector()
    processor = SessionProcessor(detector, on_frame_error="continue")

    results = list(
        processor.process(_PrefixErrorSequence(), start_index=2, max_frames=1)
    )

    assert [result.frame_index for result in results] == [2]
    assert detector.calls == [2]
    assert processor.summary.frame_count == 1
    assert processor.summary.error_count == 0
