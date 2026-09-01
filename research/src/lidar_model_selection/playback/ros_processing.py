"""ROS-independent single-worker coordination for live playback inference.

The ROS adapter only has to turn subscription callbacks into :meth:`submit`
calls.  Queue policy, loop-boundary invalidation, serialized detector access,
and diagnostic accounting live here so they can be tested without ROS or a
GPU runtime.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
import threading
import time
from typing import Callable, Deque, Generic, Literal, TypeVar, cast


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
ProcessingPolicy = Literal["all", "latest"]


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer and not a boolean")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field_name} must be greater than zero")
    return result


def _optional_milliseconds(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number and not a boolean")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{field_name} must be finite and nonnegative")
    return result


@dataclass(frozen=True, slots=True)
class ProcessingItem(Generic[InputT]):
    """One accepted input together with its loop-local identity."""

    payload: InputT
    timestamp_ns: int
    input_frame: str | None
    frame_index: int
    generation: int
    received_monotonic_ns: int


@dataclass(frozen=True, slots=True)
class ProcessingResult(Generic[OutputT]):
    """A processor result with optional measured stage timings.

    A processor may return its output directly.  In that case the coordinator
    records the complete processor-call duration as ``detector_ms``.  This
    wrapper lets an adapter report conversion and detector time separately.
    """

    value: OutputT
    conversion_ms: float | None = None
    detector_ms: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "conversion_ms",
            _optional_milliseconds(self.conversion_ms, "conversion_ms"),
        )
        object.__setattr__(
            self,
            "detector_ms",
            _optional_milliseconds(self.detector_ms, "detector_ms"),
        )


@dataclass(frozen=True, slots=True)
class ResetEvent:
    """Evidence emitted when a non-increasing input timestamp starts a loop."""

    previous_timestamp_ns: int
    timestamp_ns: int
    generation: int
    dropped_pending: int


@dataclass(frozen=True, slots=True)
class Submission:
    """Explicit result of attempting to enqueue a subscription input."""

    accepted: bool
    reset: bool
    generation: int
    frame_index: int | None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ProcessingSnapshot:
    """Thread-safe lifetime counters and most-recent timing evidence."""

    policy: ProcessingPolicy
    capacity: int
    received: int
    processed: int
    dropped: int
    failed: int
    rejected: int
    loops: int
    pending: int
    in_flight: bool
    generation: int
    next_frame_index: int
    last_input_timestamp_ns: int | None
    last_input_frame: str | None
    last_frame_index: int | None
    input_age_ms: float | None
    last_queue_ms: float | None
    last_conversion_ms: float | None
    last_detector_ms: float | None
    last_publish_ms: float | None
    last_end_to_end_ms: float | None
    last_error_stage: str | None
    last_error_code: str | None
    last_error_message: str | None
    accepting: bool
    worker_alive: bool


class ProcessingCoordinator(Generic[InputT, OutputT]):
    """Bounded queue feeding exactly one processor/publisher worker.

    ``all`` preserves accepted order and explicitly rejects a submission when
    its fixed pending capacity is full.  ``latest`` has exactly one pending
    slot and replaces that slot when a newer input arrives.

    The reset callback is serialized against publication.  Once a loop reset
    has returned, no result from an older generation can subsequently publish.
    """

    def __init__(
        self,
        process: Callable[
            [ProcessingItem[InputT]],
            OutputT | ProcessingResult[OutputT],
        ],
        publish: Callable[[OutputT], None],
        *,
        policy: ProcessingPolicy,
        capacity: int,
        reset: Callable[[ResetEvent], None] | None = None,
        close: Callable[[], None] | None = None,
        thread_name: str = "centerpoint-playback-worker",
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not callable(process):
            raise TypeError("process must be callable")
        if not callable(publish):
            raise TypeError("publish must be callable")
        if policy not in ("all", "latest"):
            raise ValueError("policy must be either 'all' or 'latest'")
        checked_capacity = _positive_integer(capacity, "capacity")
        if policy == "latest" and checked_capacity != 1:
            raise ValueError("latest policy capacity must be exactly one")
        if reset is not None and not callable(reset):
            raise TypeError("reset must be callable when provided")
        if close is not None and not callable(close):
            raise TypeError("close must be callable when provided")
        if not isinstance(thread_name, str) or not thread_name.strip():
            raise ValueError("thread_name must be a non-empty string")
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")

        self._process = process
        self._publish = publish
        self._reset = reset
        self._close = close
        self._policy = policy
        self._capacity = checked_capacity
        self._clock_ns = clock_ns

        self._condition = threading.Condition()
        self._submit_lock = threading.RLock()
        self._publication_gate = threading.Lock()
        self._queue: Deque[ProcessingItem[InputT]] = deque()
        self._accepting = True
        self._in_flight = False
        self._generation = 0
        self._next_frame_index = 0
        self._last_input_timestamp_ns: int | None = None
        self._last_input_frame: str | None = None
        self._last_input_monotonic_ns: int | None = None
        self._last_frame_index: int | None = None

        self._received = 0
        self._processed = 0
        self._dropped = 0
        self._failed = 0
        self._rejected = 0
        self._loops = 0

        self._last_queue_ms: float | None = None
        self._last_conversion_ms: float | None = None
        self._last_detector_ms: float | None = None
        self._last_publish_ms: float | None = None
        self._last_end_to_end_ms: float | None = None
        self._last_error_stage: str | None = None
        self._last_error_code: str | None = None
        self._last_error_message: str | None = None
        self._close_callback_called = False

        self._worker = threading.Thread(
            target=self._worker_main,
            name=thread_name,
            daemon=True,
        )
        self._worker.start()

    @property
    def policy(self) -> ProcessingPolicy:
        return self._policy

    @property
    def capacity(self) -> int:
        return self._capacity

    def submit(
        self,
        payload: InputT,
        *,
        timestamp_ns: int,
        input_frame: str | None = None,
    ) -> Submission:
        """Submit one input without waiting for detector processing.

        An ``all`` overflow is returned as ``accepted=False`` with reason
        ``"queue_overflow"`` and is also recorded as a failure/rejection.
        """

        checked_timestamp_ns = _positive_integer(timestamp_ns, "timestamp_ns")
        if input_frame is not None:
            if not isinstance(input_frame, str):
                raise TypeError("input_frame must be a string when provided")
            if not input_frame.strip():
                raise ValueError("input_frame must be non-empty when provided")

        with self._submit_lock:
            with self._condition:
                if not self._accepting:
                    return Submission(
                        accepted=False,
                        reset=False,
                        generation=self._generation,
                        frame_index=None,
                        reason="coordinator_closed",
                    )
                previous_timestamp_ns = self._last_input_timestamp_ns

            is_reset = (
                previous_timestamp_ns is not None
                and checked_timestamp_ns <= previous_timestamp_ns
            )
            if is_reset:
                return self._submit_reset(
                    payload,
                    timestamp_ns=checked_timestamp_ns,
                    input_frame=input_frame,
                    previous_timestamp_ns=cast(int, previous_timestamp_ns),
                )

            with self._condition:
                return self._enqueue_locked(
                    payload,
                    timestamp_ns=checked_timestamp_ns,
                    input_frame=input_frame,
                    reset=False,
                )

    def _submit_reset(
        self,
        payload: InputT,
        *,
        timestamp_ns: int,
        input_frame: str | None,
        previous_timestamp_ns: int,
    ) -> Submission:
        # The gate makes reset publication and result publication indivisible:
        # either the old result publishes before the reset marker, or its
        # generation check happens afterward and suppresses it.
        with self._publication_gate:
            with self._condition:
                if not self._accepting:
                    return Submission(
                        accepted=False,
                        reset=False,
                        generation=self._generation,
                        frame_index=None,
                        reason="coordinator_closed",
                    )
                self._received += 1
                self._loops += 1
                self._generation += 1
                self._next_frame_index = 0
                dropped_pending = len(self._queue)
                self._queue.clear()
                self._dropped += dropped_pending
                received_ns = self._clock_ns()
                self._last_input_timestamp_ns = timestamp_ns
                self._last_input_frame = input_frame
                self._last_input_monotonic_ns = received_ns
                event = ResetEvent(
                    previous_timestamp_ns=previous_timestamp_ns,
                    timestamp_ns=timestamp_ns,
                    generation=self._generation,
                    dropped_pending=dropped_pending,
                )

            if self._reset is not None:
                try:
                    self._reset(event)
                except Exception as exc:  # diagnostic evidence, worker survives
                    self._record_error("reset", exc)

            with self._condition:
                return self._enqueue_locked(
                    payload,
                    timestamp_ns=timestamp_ns,
                    input_frame=input_frame,
                    reset=True,
                    already_received=True,
                    received_monotonic_ns=received_ns,
                )

    def _enqueue_locked(
        self,
        payload: InputT,
        *,
        timestamp_ns: int,
        input_frame: str | None,
        reset: bool,
        already_received: bool = False,
        received_monotonic_ns: int | None = None,
    ) -> Submission:
        if not already_received:
            self._received += 1
            received_monotonic_ns = self._clock_ns()
            self._last_input_timestamp_ns = timestamp_ns
            self._last_input_frame = input_frame
            self._last_input_monotonic_ns = received_monotonic_ns
        assert received_monotonic_ns is not None

        if self._policy == "all" and len(self._queue) >= self._capacity:
            self._failed += 1
            self._rejected += 1
            self._set_error_locked(
                "queue",
                "queue_overflow",
                f"all-policy pending capacity {self._capacity} is full",
            )
            return Submission(
                accepted=False,
                reset=reset,
                generation=self._generation,
                frame_index=None,
                reason="queue_overflow",
            )

        if self._policy == "latest" and self._queue:
            self._queue.clear()
            self._dropped += 1

        frame_index = self._next_frame_index
        self._next_frame_index += 1
        item = ProcessingItem(
            payload=payload,
            timestamp_ns=timestamp_ns,
            input_frame=input_frame,
            frame_index=frame_index,
            generation=self._generation,
            received_monotonic_ns=received_monotonic_ns,
        )
        self._queue.append(item)
        self._last_frame_index = frame_index
        self._condition.notify()
        return Submission(
            accepted=True,
            reset=reset,
            generation=self._generation,
            frame_index=frame_index,
        )

    def snapshot(self) -> ProcessingSnapshot:
        """Return one internally consistent diagnostic snapshot."""

        now_ns = self._clock_ns()
        with self._condition:
            input_age_ms = None
            if self._last_input_monotonic_ns is not None:
                input_age_ms = max(
                    0.0,
                    (now_ns - self._last_input_monotonic_ns) / 1_000_000.0,
                )
            return ProcessingSnapshot(
                policy=self._policy,
                capacity=self._capacity,
                received=self._received,
                processed=self._processed,
                dropped=self._dropped,
                failed=self._failed,
                rejected=self._rejected,
                loops=self._loops,
                pending=len(self._queue),
                in_flight=self._in_flight,
                generation=self._generation,
                next_frame_index=self._next_frame_index,
                last_input_timestamp_ns=self._last_input_timestamp_ns,
                last_input_frame=self._last_input_frame,
                last_frame_index=self._last_frame_index,
                input_age_ms=input_age_ms,
                last_queue_ms=self._last_queue_ms,
                last_conversion_ms=self._last_conversion_ms,
                last_detector_ms=self._last_detector_ms,
                last_publish_ms=self._last_publish_ms,
                last_end_to_end_ms=self._last_end_to_end_ms,
                last_error_stage=self._last_error_stage,
                last_error_code=self._last_error_code,
                last_error_message=self._last_error_message,
                accepting=self._accepting,
                worker_alive=self._worker.is_alive(),
            )

    def record_failure(
        self,
        error: Exception | str,
        *,
        timestamp_ns: int | None = None,
        input_frame: str | None = None,
    ) -> None:
        """Record an input failure that happened before :meth:`submit`.

        This seam is for malformed middleware messages whose header or payload
        cannot be converted into a queue item.  It counts one received and one
        failed input while preserving whatever identity was recoverable.
        """

        if not isinstance(error, (Exception, str)):
            raise TypeError("error must be an exception or string")
        if isinstance(error, str) and not error.strip():
            raise ValueError("error must be non-empty")
        checked_timestamp_ns = None
        if timestamp_ns is not None:
            checked_timestamp_ns = _positive_integer(timestamp_ns, "timestamp_ns")
        if input_frame is not None:
            if not isinstance(input_frame, str):
                raise TypeError("input_frame must be a string when provided")
            if not input_frame.strip():
                raise ValueError("input_frame must be non-empty when provided")

        with self._condition:
            self._received += 1
            self._failed += 1
            self._last_input_monotonic_ns = self._clock_ns()
            if checked_timestamp_ns is not None:
                self._last_input_timestamp_ns = checked_timestamp_ns
            if input_frame is not None:
                self._last_input_frame = input_frame
            if isinstance(error, Exception):
                self._set_exception_locked("input", error)
            else:
                self._set_error_locked("input", "input_failure", error)

    def record_drop(self, count: int = 1, *, reason: str) -> None:
        """Record externally observed message loss without inventing a frame."""

        checked_count = _positive_integer(count, "count")
        if not isinstance(reason, str):
            raise TypeError("reason must be a string")
        if not reason.strip():
            raise ValueError("reason must be non-empty")
        with self._condition:
            self._dropped += checked_count
            self._set_error_locked("drop", "external_drop", reason)

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        """Wait until there is neither queued nor in-flight work."""

        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, Real):
                raise TypeError("timeout must be a real number when provided")
            if not isfinite(float(timeout)) or float(timeout) < 0.0:
                raise ValueError("timeout must be finite and nonnegative")
            deadline = time.monotonic() + float(timeout)
        else:
            deadline = None

        with self._condition:
            while self._queue or self._in_flight:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(self, *, drain: bool = False, timeout: float | None = None) -> None:
        """Stop accepting work, join the worker, and close its resource once.

        By default pending and in-flight work are invalidated.  ``drain=True``
        instead lets all already accepted work publish before shutdown.
        """

        if threading.current_thread() is self._worker:
            raise RuntimeError("the processing worker cannot join itself")
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, Real):
                raise TypeError("timeout must be a real number when provided")
            if not isfinite(float(timeout)) or float(timeout) < 0.0:
                raise ValueError("timeout must be finite and nonnegative")

        with self._submit_lock:
            with self._publication_gate:
                with self._condition:
                    if self._accepting:
                        self._accepting = False
                        if not drain:
                            self._generation += 1
                            self._dropped += len(self._queue)
                            self._queue.clear()
                        self._condition.notify_all()

        self._worker.join(None if timeout is None else float(timeout))
        if self._worker.is_alive():
            raise TimeoutError("processing worker did not stop before timeout")

    def __enter__(self) -> "ProcessingCoordinator[InputT, OutputT]":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _worker_main(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._queue and self._accepting:
                        self._condition.wait()
                    if not self._queue:
                        return
                    item = self._queue.popleft()
                    self._in_flight = True
                    started_ns = self._clock_ns()
                    self._last_queue_ms = max(
                        0.0,
                        (started_ns - item.received_monotonic_ns) / 1_000_000.0,
                    )

                process_started_ns = self._clock_ns()
                try:
                    raw_result = self._process(item)
                except Exception as exc:
                    process_finished_ns = self._clock_ns()
                    with self._condition:
                        self._last_detector_ms = max(
                            0.0,
                            (process_finished_ns - process_started_ns) / 1_000_000.0,
                        )
                        self._last_end_to_end_ms = max(
                            0.0,
                            (
                                process_finished_ns
                                - item.received_monotonic_ns
                            )
                            / 1_000_000.0,
                        )
                        self._failed += 1
                        self._set_exception_locked("process", exc)
                        self._in_flight = False
                        self._condition.notify_all()
                    continue
                process_finished_ns = self._clock_ns()

                if isinstance(raw_result, ProcessingResult):
                    result = raw_result.value
                    conversion_ms = raw_result.conversion_ms
                    detector_ms = raw_result.detector_ms
                    if detector_ms is None:
                        detector_ms = max(
                            0.0,
                            (process_finished_ns - process_started_ns)
                            / 1_000_000.0,
                        )
                else:
                    result = cast(OutputT, raw_result)
                    conversion_ms = None
                    detector_ms = max(
                        0.0,
                        (process_finished_ns - process_started_ns) / 1_000_000.0,
                    )

                with self._condition:
                    self._last_conversion_ms = conversion_ms
                    self._last_detector_ms = detector_ms

                with self._publication_gate:
                    with self._condition:
                        if item.generation != self._generation:
                            self._dropped += 1
                            self._last_end_to_end_ms = max(
                                0.0,
                                (self._clock_ns() - item.received_monotonic_ns)
                                / 1_000_000.0,
                            )
                            self._in_flight = False
                            self._condition.notify_all()
                            continue

                    publish_started_ns = self._clock_ns()
                    try:
                        self._publish(result)
                    except Exception as exc:
                        publish_finished_ns = self._clock_ns()
                        with self._condition:
                            self._last_publish_ms = max(
                                0.0,
                                (publish_finished_ns - publish_started_ns)
                                / 1_000_000.0,
                            )
                            self._last_end_to_end_ms = max(
                                0.0,
                                (
                                    publish_finished_ns
                                    - item.received_monotonic_ns
                                )
                                / 1_000_000.0,
                            )
                            self._failed += 1
                            self._set_exception_locked("publish", exc)
                            self._in_flight = False
                            self._condition.notify_all()
                        continue
                    publish_finished_ns = self._clock_ns()

                    with self._condition:
                        self._last_publish_ms = max(
                            0.0,
                            (publish_finished_ns - publish_started_ns)
                            / 1_000_000.0,
                        )
                        self._last_end_to_end_ms = max(
                            0.0,
                            (
                                publish_finished_ns - item.received_monotonic_ns
                            )
                            / 1_000_000.0,
                        )
                        self._processed += 1
                        self._in_flight = False
                        self._condition.notify_all()
        finally:
            self._invoke_close_once()

    def _record_error(self, stage: str, exc: Exception) -> None:
        with self._condition:
            self._failed += 1
            self._set_exception_locked(stage, exc)

    def _set_exception_locked(self, stage: str, exc: Exception) -> None:
        evidence = getattr(exc, "evidence", None)
        code = getattr(evidence, "code", None)
        if not isinstance(code, str) or not code:
            code = type(exc).__name__
        message = str(exc) or repr(exc)
        self._set_error_locked(stage, code, message)

    def _set_error_locked(self, stage: str, code: str, message: str) -> None:
        self._last_error_stage = stage
        self._last_error_code = code
        self._last_error_message = message

    def _invoke_close_once(self) -> None:
        with self._condition:
            if self._close_callback_called:
                return
            self._close_callback_called = True
        if self._close is None:
            return
        try:
            self._close()
        except Exception as exc:
            self._record_error("close", exc)
