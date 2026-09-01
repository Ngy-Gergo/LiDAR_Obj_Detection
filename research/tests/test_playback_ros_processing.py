from __future__ import annotations

import threading

import pytest

from lidar_model_selection.playback.ros_processing import (
    ProcessingCoordinator,
    ProcessingItem,
    ProcessingResult,
    ProcessingStageError,
    ProcessingTimings,
    ResetEvent,
)


def _await(event: threading.Event) -> None:
    assert event.wait(2.0), "worker did not reach the expected state"


def test_all_policy_is_bounded_ordered_and_rejects_overflow_explicitly() -> None:
    entered = threading.Event()
    release = threading.Event()
    active = 0
    max_active = 0
    active_lock = threading.Lock()
    processed: list[str] = []

    def process(item: ProcessingItem[str]) -> str:
        nonlocal active, max_active
        with active_lock:
            active += 1
            max_active = max(max_active, active)
        if item.payload == "first":
            entered.set()
            _await(release)
        with active_lock:
            active -= 1
        return item.payload

    coordinator = ProcessingCoordinator(
        process,
        processed.append,
        policy="all",
        capacity=2,
    )
    try:
        assert coordinator.submit("first", timestamp_ns=10).accepted
        _await(entered)
        assert coordinator.submit("second", timestamp_ns=20).accepted
        assert coordinator.submit("third", timestamp_ns=30).accepted
        overflow = coordinator.submit("fourth", timestamp_ns=40)

        assert not overflow.accepted
        assert overflow.reason == "queue_overflow"
        release.set()
        assert coordinator.wait_until_idle(2.0)

        assert processed == ["first", "second", "third"]
        assert max_active == 1
        snapshot = coordinator.snapshot()
        assert snapshot.received == 4
        assert snapshot.processed == 3
        assert snapshot.failed == 1
        assert snapshot.rejected == 1
        assert snapshot.dropped == 0
        assert snapshot.last_error_stage == "queue"
        assert snapshot.last_error_code == "queue_overflow"
    finally:
        release.set()
        coordinator.close()


def test_latest_policy_keeps_at_most_one_pending_and_counts_replacement() -> None:
    entered = threading.Event()
    release = threading.Event()
    published: list[str] = []

    def process(item: ProcessingItem[str]) -> str:
        if item.payload == "in-flight":
            entered.set()
            _await(release)
        return item.payload

    coordinator = ProcessingCoordinator(
        process,
        published.append,
        policy="latest",
        capacity=1,
    )
    try:
        coordinator.submit("in-flight", timestamp_ns=10)
        _await(entered)
        coordinator.submit("replaced", timestamp_ns=20)
        coordinator.submit("newest", timestamp_ns=30)
        assert coordinator.snapshot().pending == 1

        release.set()
        assert coordinator.wait_until_idle(2.0)

        assert published == ["in-flight", "newest"]
        snapshot = coordinator.snapshot()
        assert snapshot.received == 3
        assert snapshot.processed == 2
        assert snapshot.dropped == 1
        assert snapshot.failed == 0
    finally:
        release.set()
        coordinator.close()


def test_nonincreasing_timestamp_resets_generation_and_suppresses_old_work() -> None:
    entered = threading.Event()
    release = threading.Event()
    published: list[str] = []
    reset_events: list[ResetEvent] = []
    processing_items: list[ProcessingItem[str]] = []

    def process(item: ProcessingItem[str]) -> str:
        processing_items.append(item)
        if item.payload == "old-in-flight":
            entered.set()
            _await(release)
        return item.payload

    coordinator = ProcessingCoordinator(
        process,
        published.append,
        policy="all",
        capacity=4,
        reset=reset_events.append,
    )
    try:
        coordinator.submit(
            "old-in-flight",
            timestamp_ns=100,
            input_frame="sensor",
        )
        _await(entered)
        coordinator.submit("old-pending", timestamp_ns=110, input_frame="sensor")
        loop_submission = coordinator.submit(
            "new-loop",
            timestamp_ns=100,
            input_frame="sensor",
        )

        assert loop_submission.accepted
        assert loop_submission.reset
        assert loop_submission.generation == 1
        assert loop_submission.frame_index == 0
        assert reset_events == [
            ResetEvent(
                reason="point_timestamp",
                previous_timestamp_ns=110,
                timestamp_ns=100,
                generation=1,
                dropped_pending=1,
            )
        ]

        release.set()
        assert coordinator.wait_until_idle(2.0)

        assert published == ["new-loop"]
        assert [item.payload for item in processing_items] == [
            "old-in-flight",
            "new-loop",
        ]
        assert processing_items[-1].frame_index == 0
        assert processing_items[-1].generation == 1
        snapshot = coordinator.snapshot()
        assert snapshot.loops == 1
        assert snapshot.dropped == 2  # one pending plus one stale in-flight
        assert snapshot.processed == 1
        assert snapshot.next_frame_index == 1
        assert snapshot.last_input_timestamp_ns == 100
        assert snapshot.last_input_frame == "sensor"
    finally:
        release.set()
        coordinator.close()


def test_process_and_publish_failures_are_diagnostic_and_worker_continues() -> None:
    published: list[str] = []

    def process(item: ProcessingItem[str]) -> str:
        if item.payload == "process-error":
            raise ValueError("cannot convert")
        return item.payload

    def publish(result: str) -> None:
        if result == "publish-error":
            raise RuntimeError("publisher rejected output")
        published.append(result)

    coordinator = ProcessingCoordinator(
        process,
        publish,
        policy="all",
        capacity=3,
    )
    try:
        coordinator.submit("process-error", timestamp_ns=1)
        coordinator.submit("publish-error", timestamp_ns=2)
        coordinator.submit("success", timestamp_ns=3)
        assert coordinator.wait_until_idle(2.0)

        assert published == ["success"]
        snapshot = coordinator.snapshot()
        assert snapshot.received == 3
        assert snapshot.processed == 1
        assert snapshot.failed == 2
        assert snapshot.last_error_stage == "publish"
        assert snapshot.last_error_code == "RuntimeError"
        assert snapshot.last_inference_ms is None
        assert snapshot.last_publish_ms is not None
        assert snapshot.last_end_to_end_ms is not None
    finally:
        coordinator.close()

def test_processing_result_preserves_stage_timings_and_snapshot_age() -> None:
    clock_lock = threading.Lock()
    now_ns = 1_000_000_000

    def clock() -> int:
        with clock_lock:
            return now_ns

    coordinator = ProcessingCoordinator(
        lambda item: ProcessingResult(
            value=item.payload.upper(),
            tf_lookup_ms=0.75,
            conversion_ms=1.25,
            inference_ms=8.5,
        ),
        lambda _result: None,
        policy="all",
        capacity=1,
        clock_ns=clock,
    )
    try:
        coordinator.submit("frame", timestamp_ns=50)
        assert coordinator.wait_until_idle(2.0)
        with clock_lock:
            now_ns += 5_000_000
        snapshot = coordinator.snapshot()

        assert snapshot.last_tf_lookup_ms == 0.75
        assert snapshot.last_conversion_ms == 1.25
        assert snapshot.last_inference_ms == 8.5
        assert snapshot.last_publish_ms == 0.0
        assert snapshot.last_queue_ms == 0.0
        assert snapshot.last_end_to_end_ms == 0.0
        assert snapshot.input_age_ms == 5.0
    finally:
        coordinator.close()


def test_close_invalidates_inflight_work_joins_and_closes_once() -> None:
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    close_calls: list[str] = []
    published: list[str] = []

    def process(item: ProcessingItem[str]) -> str:
        entered.set()
        _await(release)
        return item.payload

    def close_resource() -> None:
        close_calls.append("closed")
        closed.set()

    coordinator = ProcessingCoordinator(
        process,
        published.append,
        policy="all",
        capacity=1,
        close=close_resource,
    )
    coordinator.submit("in-flight", timestamp_ns=1)
    _await(entered)

    close_finished = threading.Event()

    def close_coordinator() -> None:
        coordinator.close(timeout=2.0)
        close_finished.set()

    close_thread = threading.Thread(target=close_coordinator)
    close_thread.start()
    while coordinator.snapshot().accepting:
        close_thread.join(0.001)
    release.set()
    _await(close_finished)
    close_thread.join()

    assert published == []
    assert close_calls == ["closed"]
    assert coordinator.snapshot().dropped == 1
    assert not coordinator.snapshot().worker_alive
    coordinator.close()
    assert close_calls == ["closed"]


def test_external_failure_and_drop_seams_update_diagnostics() -> None:
    coordinator = ProcessingCoordinator(
        lambda item: item.payload,
        lambda _result: None,
        policy="all",
        capacity=1,
    )
    try:
        coordinator.record_failure(
            ValueError("malformed header"),
            timestamp_ns=90,
            input_frame="sensor",
        )
        failed = coordinator.snapshot()
        assert failed.received == 1
        assert failed.failed == 1
        assert failed.last_error_stage == "input"
        assert failed.last_error_code == "ValueError"
        assert failed.last_input_timestamp_ns == 90
        assert failed.last_input_frame == "sensor"

        coordinator.record_middleware_loss(3, reason="middleware sequence gap")
        dropped = coordinator.snapshot()
        assert dropped.dropped == 0
        assert dropped.middleware_lost == 3
        assert dropped.last_error_stage == "middleware"
        assert dropped.last_error_code == "middleware_message_lost"
        assert dropped.last_error_message == "middleware sequence gap"
    finally:
        coordinator.close()


@pytest.mark.parametrize(
    ("policy", "capacity", "message"),
    [
        ("newest", 1, "policy"),
        ("all", 0, "capacity"),
        ("latest", 2, "exactly one"),
    ],
)
def test_policy_configuration_is_explicit(
    policy: str,
    capacity: int,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        ProcessingCoordinator(
            lambda item: item,
            lambda _result: None,
            policy=policy,  # type: ignore[arg-type]
            capacity=capacity,
        )


def test_reset_callback_failure_is_counted_without_losing_new_loop_input() -> None:
    published: list[str] = []

    def reset(_event: ResetEvent) -> None:
        raise RuntimeError("marker clear failed")

    coordinator = ProcessingCoordinator(
        lambda item: item.payload,
        published.append,
        policy="all",
        capacity=1,
        reset=reset,
    )
    try:
        coordinator.submit("old", timestamp_ns=10)
        assert coordinator.wait_until_idle(2.0)
        submission = coordinator.submit("loop", timestamp_ns=10)
        assert submission.accepted
        assert submission.reset
        assert coordinator.wait_until_idle(2.0)

        assert published == ["old", "loop"]
        snapshot = coordinator.snapshot()
        assert snapshot.loops == 1
        assert snapshot.processed == 2
        assert snapshot.failed == 1
        assert snapshot.last_error_stage == "reset"
    finally:
        coordinator.close()


def test_backward_clock_jump_resets_once_before_first_new_loop_point() -> None:
    resets: list[ResetEvent] = []
    published: list[str] = []
    coordinator = ProcessingCoordinator(
        lambda item: item.payload,
        published.append,
        policy="all",
        capacity=2,
        reset=resets.append,
    )
    try:
        coordinator.submit("old", timestamp_ns=100)
        assert coordinator.wait_until_idle(2.0)

        assert coordinator.reset_generation(reason="clock_jump", timestamp_ns=50)
        submission = coordinator.submit("new", timestamp_ns=50)
        assert submission.accepted
        assert not submission.reset
        assert submission.generation == 1
        assert submission.frame_index == 0
        assert coordinator.wait_until_idle(2.0)

        assert published == ["old", "new"]
        assert resets == [
            ResetEvent(
                reason="clock_jump",
                previous_timestamp_ns=100,
                timestamp_ns=50,
                generation=1,
                dropped_pending=0,
            )
        ]
        assert coordinator.snapshot().loops == 1
        assert coordinator.snapshot().last_reset_reason == "clock_jump"
    finally:
        coordinator.close()


def test_point_timestamp_fallback_deduplicates_following_clock_jump() -> None:
    resets: list[ResetEvent] = []
    coordinator = ProcessingCoordinator(
        lambda item: item.payload,
        lambda _result: None,
        policy="all",
        capacity=2,
        reset=resets.append,
    )
    try:
        coordinator.submit("old", timestamp_ns=100)
        assert coordinator.wait_until_idle(2.0)
        fallback = coordinator.submit("new", timestamp_ns=50)
        assert fallback.reset
        assert not coordinator.reset_generation(
            reason="clock_jump",
            timestamp_ns=50,
        )
        assert coordinator.wait_until_idle(2.0)

        assert len(resets) == 1
        assert resets[0].reason == "point_timestamp"
        assert coordinator.snapshot().generation == 1
        assert coordinator.snapshot().loops == 1
    finally:
        coordinator.close()


def test_tf_timeout_is_not_attributed_to_inference() -> None:
    def process(_item: ProcessingItem[str]) -> str:
        raise ProcessingStageError(
            stage="tf_lookup",
            code="missing_or_stale_tf",
            message="source frame does not exist",
            timings=ProcessingTimings(tf_lookup_ms=204.0),
        )

    coordinator = ProcessingCoordinator(
        process,
        lambda _result: None,
        policy="all",
        capacity=1,
    )
    try:
        coordinator.submit("frame", timestamp_ns=1)
        assert coordinator.wait_until_idle(2.0)
        snapshot = coordinator.snapshot()
        assert snapshot.failed == 1
        assert snapshot.last_error_stage == "tf_lookup"
        assert snapshot.last_error_code == "missing_or_stale_tf"
        assert snapshot.last_tf_lookup_ms == 204.0
        assert snapshot.last_conversion_ms is None
        assert snapshot.last_inference_ms is None
        assert snapshot.last_publish_ms is None
        assert snapshot.last_end_to_end_ms is not None
    finally:
        coordinator.close()


def test_processing_result_can_report_no_inference_for_empty_frame() -> None:
    coordinator = ProcessingCoordinator(
        lambda item: ProcessingResult(
            item.payload,
            tf_lookup_ms=0.1,
            conversion_ms=1.0,
            inference_ms=None,
        ),
        lambda _result: None,
        policy="all",
        capacity=1,
    )
    try:
        coordinator.submit("empty", timestamp_ns=1)
        assert coordinator.wait_until_idle(2.0)
        snapshot = coordinator.snapshot()
        assert snapshot.processed == 1
        assert snapshot.last_tf_lookup_ms == 0.1
        assert snapshot.last_conversion_ms == 1.0
        assert snapshot.last_inference_ms is None
        assert snapshot.last_publish_ms is not None
    finally:
        coordinator.close()
