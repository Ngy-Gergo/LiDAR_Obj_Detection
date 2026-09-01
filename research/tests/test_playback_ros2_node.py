from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from lidar_model_selection.playback import ros2_node


RUNS_ROOT = Path("/home/ws-rtx/Documents/Projects/lidar-centerpoint/research/runs")


def _args(*, model="voxel0075", policy="latest", capacity="1") -> list[str]:
    return [
        "--model",
        model,
        "--runs-root",
        str(RUNS_ROOT),
        "--device",
        "cuda:0",
        "--input-topic",
        "/lexus3/os_center/points",
        "--output-prefix",
        f"/centerpoint/{model}",
        "--base-frame",
        "lexus3/base_link",
        "--feature-profile",
        "kaposvar_center_reflectivity_v1",
        "--score-threshold",
        "0.1",
        "--processing-policy",
        policy,
        "--queue-capacity",
        capacity,
        "--publish-model-cloud",
    ]


def test_import_and_help_contract_do_not_import_ros_or_ml_frameworks() -> None:
    source = Path(ros2_node.__file__).read_text(encoding="utf-8")
    assert "import rclpy\n" not in source
    assert "from rclpy" not in source
    assert "import torch" not in source
    assert "import mmdet3d" not in source
    parser = ros2_node._build_argument_parser()
    assert parser.parse_args(_args()).model == "voxel0075"


def test_cli_is_explicit_and_passes_only_ros_arguments_to_rclpy() -> None:
    config, ros_args = ros2_node._parse_arguments(
        [*_args(policy="all", capacity="32"), "--ros-args", "-p", "use_sim_time:=true"]
    )
    assert config.model == "voxel0075"
    assert config.processing_policy == "all"
    assert config.queue_capacity == 32
    assert config.publish_model_cloud
    assert ros_args == ["--ros-args", "-p", "use_sim_time:=true"]


def test_cli_tracking_is_opt_in_and_validates_useful_bounds() -> None:
    config, _ = ros2_node._parse_arguments(
        [
            *_args(),
            "--enable-tracking",
            "--track-min-hits",
            "3",
            "--track-max-missed",
            "4",
            "--track-max-gap-seconds",
            "1.25",
            "--track-association-distance",
            "5.5",
            "--track-smoothing",
            "0.7",
            "--track-trail-length",
            "24",
        ]
    )
    assert config.enable_tracking
    assert ros2_node._tracker_config(config).min_confirmed_hits == 3
    assert ros2_node._tracker_config(config).max_missed_frames == 4
    assert ros2_node._tracker_config(config).max_time_gap_seconds == 1.25
    assert ros2_node._tracker_config(config).association_distance_meters == 5.5
    assert ros2_node._tracker_config(config).position_smoothing == 0.7
    assert ros2_node._tracker_config(config).score_smoothing == 0.7
    assert ros2_node._tracker_config(config).trail_length == 24

    with pytest.raises(SystemExit):
        ros2_node._parse_arguments([*_args(), "--track-trail-length", "0"])


@pytest.mark.parametrize(
    "replacement",
    [
        ("--output-prefix", "/centerpoint/pillar02"),
        ("--base-frame", "map"),
        ("--input-topic", "/other/points"),
        ("--device", "cpu"),
        ("--feature-profile", "adaptive"),
        ("--queue-capacity", "2"),
    ],
)
def test_cli_rejects_ambiguous_or_incompatible_binding(replacement) -> None:
    args = _args()
    index = args.index(replacement[0]) + 1
    args[index] = replacement[1]
    with pytest.raises(SystemExit):
        ros2_node._parse_arguments(args)


def test_one_cli_model_constructs_exactly_one_bound_detector() -> None:
    config, _ = ros2_node._parse_arguments(_args(model="pillar02"))
    calls = []

    def factory(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    detector = ros2_node._build_detector(config, factory)
    assert detector is not None
    assert calls == [
        (
            ("pillar02", RUNS_ROOT),
            {"device": "cuda:0", "score_threshold": 0.1},
        )
    ]


def test_checkpoint_pin_is_checked_before_detector_construction() -> None:
    config, _ = ros2_node._parse_arguments(
        [*_args(), "--checkpoint-sha256", "0" * 64]
    )
    calls = []
    with pytest.raises(ValueError, match="protected registry"):
        ros2_node._build_detector(
            config,
            lambda *args, **kwargs: calls.append((args, kwargs)),
        )
    assert calls == []


def test_main_forwards_only_ros_args_and_closes_node_cleanly(monkeypatch) -> None:
    events = []

    class FakeRclpy:
        def init(self, *, args, signal_handler_options):
            events.append(("init", args, signal_handler_options))

        def spin(self, node):
            events.append(("spin", node.config.model))

        def try_shutdown(self):
            events.append(("try_shutdown",))

    class FakeNode:
        def __init__(self, config):
            self.config = config

        def close(self):
            events.append(("close",))

        def destroy_node(self):
            events.append(("destroy",))

    runtime = SimpleNamespace(
        rclpy=FakeRclpy(),
        SignalHandlerOptions=SimpleNamespace(NO="no-handlers"),
        ExternalShutdownException=RuntimeError,
    )
    monkeypatch.setattr(ros2_node, "_load_ros_runtime", lambda: runtime)
    monkeypatch.setattr(ros2_node, "_node_class", lambda _runtime: FakeNode)

    ros2_node.main([*_args(), "--ros-args", "-p", "use_sim_time:=true"])

    assert events == [
        (
            "init",
            ["--ros-args", "-p", "use_sim_time:=true"],
            "no-handlers",
        ),
        ("spin", "voxel0075"),
        ("close",),
        ("destroy",),
        ("try_shutdown",),
    ]


def test_main_handles_keyboard_interrupt_before_context_shutdown(monkeypatch) -> None:
    events = []

    class FakeRclpy:
        def init(self, *, args, signal_handler_options):
            events.append("init")

        def spin(self, node):
            events.append("spin")
            raise KeyboardInterrupt

        def try_shutdown(self):
            events.append("try_shutdown")

    class FakeNode:
        def __init__(self, config):
            pass

        def close(self):
            events.append("close")

        def destroy_node(self):
            events.append("destroy")

    runtime = SimpleNamespace(
        rclpy=FakeRclpy(),
        SignalHandlerOptions=SimpleNamespace(NO="no-handlers"),
        ExternalShutdownException=RuntimeError,
    )
    monkeypatch.setattr(ros2_node, "_load_ros_runtime", lambda: runtime)
    monkeypatch.setattr(ros2_node, "_node_class", lambda _runtime: FakeNode)

    ros2_node.main(_args())

    assert events == ["init", "spin", "close", "destroy", "try_shutdown"]


def test_missing_vision_msgs_fails_with_exact_host_install_instruction(monkeypatch) -> None:
    imported = []

    def fake_import(name):
        imported.append(name)
        if name == "vision_msgs.msg":
            raise ImportError("missing")
        return SimpleNamespace()

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(RuntimeError, match="sudo apt install ros-humble-vision-msgs"):
        ros2_node._load_ros_runtime()
    assert "rclpy.qos_event" in imported
    assert "rclpy.event_handler" not in imported


class _TransformError(Exception):
    pass


def _runtime():
    return SimpleNamespace(
        Time=SimpleNamespace(from_msg=lambda stamp: (stamp.sec, stamp.nanosec)),
        Duration=lambda **kwargs: kwargs,
        TransformException=_TransformError,
    )


def _message():
    return SimpleNamespace(
        header=SimpleNamespace(
            frame_id="lexus3/os_center",
            stamp=SimpleNamespace(sec=1, nanosec=2),
        )
    )


def _config():
    return ros2_node._parse_arguments(_args())[0]


def test_missing_tf_stops_frame_before_conversion_or_publication() -> None:
    class Buffer:
        def lookup_transform(self, *args, **kwargs):
            raise _TransformError("not connected")

    clears = []
    with pytest.raises(RuntimeError, match="missing_or_stale_tf"):
        ros2_node._run_with_overlay_guard(
            lambda: ros2_node._lookup_calibration(
                _runtime(), Buffer(), _message(), _config()
            ),
            lambda: clears.append("DELETEALL"),
        )
    assert clears == ["DELETEALL"]


def test_invalid_tf_stops_frame_instead_of_fabricating_overlay() -> None:
    transform = SimpleNamespace(
        header=SimpleNamespace(frame_id="lexus3/base_link"),
        child_frame_id="lexus3/os_center",
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=9.0, y=0.0, z=1.91),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=-1.0, w=0.0),
        ),
    )

    class Buffer:
        def lookup_transform(self, target, source, time, timeout):
            assert (target, source, time) == ("lexus3/base_link", "lexus3/os_center", (1, 2))
            assert timeout == {"seconds": 0.2}
            return transform

    with pytest.raises(RuntimeError, match="invalid_or_ambiguous_tf"):
        ros2_node._lookup_calibration(_runtime(), Buffer(), _message(), _config())


def test_tf_lookup_reports_only_tf_stage_timing() -> None:
    class Buffer:
        def lookup_transform(self, *args, **kwargs):
            raise _TransformError("not connected")

    ticks = iter((10.0, 10.2))
    with pytest.raises(ros2_node.ProcessingStageError) as failure:
        ros2_node._lookup_calibration(
            _runtime(),
            Buffer(),
            _message(),
            _config(),
            clock=lambda: next(ticks),
        )
    assert failure.value.stage == "tf_lookup"
    assert failure.value.code == "missing_or_stale_tf"
    assert failure.value.timings.tf_lookup_ms == pytest.approx(200.0)
    assert failure.value.timings.conversion_ms is None
    assert failure.value.timings.inference_ms is None


@pytest.mark.parametrize(
    ("policy", "capacity", "expected_depth"),
    [("all", 32, 32), ("latest", 1, 1)],
)
def test_pointcloud_subscription_qos_matches_processing_policy(
    policy: str,
    capacity: int,
    expected_depth: int,
) -> None:
    runtime = SimpleNamespace(
        QoSProfile=lambda **kwargs: SimpleNamespace(**kwargs),
        QoSHistoryPolicy=SimpleNamespace(KEEP_LAST="keep_last"),
        QoSReliabilityPolicy=SimpleNamespace(
            RELIABLE="reliable",
            BEST_EFFORT="best_effort",
        ),
        QoSDurabilityPolicy=SimpleNamespace(
            TRANSIENT_LOCAL="transient_local",
            VOLATILE="volatile",
        ),
    )
    qos = ros2_node._pointcloud_qos(
        runtime,
        policy=policy,
        queue_capacity=capacity,
    )
    assert qos.history == "keep_last"
    assert qos.depth == expected_depth
    assert qos.reliability == "best_effort"
    assert qos.durability == "volatile"


def test_repository_replay_qos_retains_complete_tf_static_history() -> None:
    path = RUNS_ROOT.parents[1] / "research/configs/playback/rosbag2_qos.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert document == {
        "/tf_static": {
            "history": "keep_last",
            "depth": 100,
            "reliability": "reliable",
            "durability": "transient_local",
        }
    }


class _FakeLogger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def info(self, message: str) -> None:
        self.messages.append(("info", message))

    def warning(self, message: str) -> None:
        self.messages.append(("warning", message))

    def error(self, message: str) -> None:
        self.messages.append(("error", message))

    def debug(self, message: str) -> None:
        self.messages.append(("debug", message))


class _FakeClock:
    def __init__(self, nanoseconds: int = 50) -> None:
        self._now = SimpleNamespace(
            nanoseconds=nanoseconds,
            to_msg=lambda: SimpleNamespace(sec=0, nanosec=nanoseconds),
        )

    def now(self):
        return self._now


class _FakeNodeBase:
    context = object()

    def get_clock(self):
        return self._clock

    def get_logger(self):
        return self._logger


class _FakePublisher:
    def __init__(self) -> None:
        self.messages: list[object] = []

    def publish(self, message: object) -> None:
        self.messages.append(message)


def _node_test_runtime(valid: dict[str, bool]):
    class FakeRclpy:
        @staticmethod
        def ok(*, context):
            assert context is _FakeNodeBase.context
            return valid["value"]

    return SimpleNamespace(
        Node=_FakeNodeBase,
        rclpy=FakeRclpy,
        RCLError=RuntimeError,
        QoSProfile=lambda **kwargs: SimpleNamespace(**kwargs),
        QoSHistoryPolicy=SimpleNamespace(
            KEEP_LAST="keep_last",
            KEEP_ALL="keep_all",
        ),
        QoSReliabilityPolicy=SimpleNamespace(
            RELIABLE="reliable",
            BEST_EFFORT="best_effort",
        ),
        QoSDurabilityPolicy=SimpleNamespace(
            TRANSIENT_LOCAL="transient_local",
            VOLATILE="volatile",
        ),
        message_types=SimpleNamespace(
            Header=lambda: SimpleNamespace(
                stamp=SimpleNamespace(sec=0, nanosec=0),
            )
        ),
    )


def _bare_node(runtime):
    node_type = ros2_node._node_class(runtime)
    node = node_type.__new__(node_type)
    node._closed = False
    node._close_lock = __import__("threading").Lock()
    node._marker_lock = __import__("threading").RLock()
    node._tf_lock = __import__("threading").RLock()
    node._previous_detection_count = 3
    node._previous_track_ids = set()
    node._tracker = None
    node._clock = _FakeClock()
    node._logger = _FakeLogger()
    node._builder = SimpleNamespace(
        clear_markers=lambda *, stamp: ("DELETEALL", stamp),
    )
    node._markers_publisher = _FakePublisher()
    node._tracked_markers_publisher = None
    node._tracked_detections_publisher = None
    node._tracking_diagnostics_publisher = None
    return node


@pytest.mark.parametrize("context_valid", [True, False])
def test_close_is_idempotent_and_respects_context_validity(context_valid: bool) -> None:
    valid = {"value": context_valid}
    runtime = _node_test_runtime(valid)
    node = _bare_node(runtime)
    events: list[str] = []
    node._coordinator = SimpleNamespace(
        close=lambda *, drain: events.append(f"coordinator:{drain}")
    )
    node._clock_jump_handle = SimpleNamespace(
        unregister=lambda: events.append("jump-unregister")
    )
    node._tf_buffer = object()
    node._tf_listener = SimpleNamespace(
        unregister=lambda: events.append("tf-unregister")
    )

    node.close()
    node.close()

    assert events.count("coordinator:False") == 1
    if context_valid:
        assert events == ["jump-unregister", "coordinator:False", "tf-unregister"]
        assert len(node._markers_publisher.messages) == 1
    else:
        assert events == ["coordinator:False"]
        assert node._markers_publisher.messages == []
        assert any(
            level == "warning" and "context is not valid" in message
            for level, message in node._logger.messages
        )


def test_tf_state_recreation_uses_complete_bounded_static_history() -> None:
    valid = {"value": True}
    runtime = _node_test_runtime(valid)
    created_buffers: list[object] = []
    created_listeners: list[object] = []

    def buffer_factory():
        buffer = object()
        created_buffers.append(buffer)
        return buffer

    class Listener:
        def __init__(self, buffer, node, *, spin_thread, static_qos):
            self.buffer = buffer
            self.node = node
            self.spin_thread = spin_thread
            self.static_qos = static_qos
            self.unregister_calls = 0
            created_listeners.append(self)

        def unregister(self):
            self.unregister_calls += 1

    runtime.Buffer = buffer_factory
    runtime.TransformListener = Listener
    node = _bare_node(runtime)
    old_listener = Listener(object(), node, spin_thread=False, static_qos=object())
    node._tf_listener = old_listener
    node._tf_buffer = old_listener.buffer

    node._replace_tf_state()

    assert old_listener.unregister_calls == 1
    assert node._tf_buffer is created_buffers[-1]
    assert node._tf_listener is created_listeners[-1]
    assert node._tf_listener.static_qos.history == "keep_last"
    assert node._tf_listener.static_qos.depth == 100
    assert node._tf_listener.static_qos.reliability == "reliable"
    assert node._tf_listener.static_qos.durability == "transient_local"


def test_backward_clock_jump_requests_one_clock_generation_reset() -> None:
    valid = {"value": True}
    runtime = _node_test_runtime(valid)
    node = _bare_node(runtime)
    calls: list[tuple[str, int | None]] = []
    node._coordinator = SimpleNamespace(
        reset_generation=lambda *, reason, timestamp_ns: (
            calls.append((reason, timestamp_ns)) or True
        )
    )

    node._clock_jump(SimpleNamespace(delta=SimpleNamespace(nanoseconds=-1)))
    node._clock_jump(SimpleNamespace(delta=SimpleNamespace(nanoseconds=1)))

    assert calls == [("clock_jump", 50)]


def test_reset_recreates_tf_then_clears_markers() -> None:
    valid = {"value": True}
    runtime = _node_test_runtime(valid)
    runtime.message_types = SimpleNamespace(
        Header=lambda: SimpleNamespace(
            stamp=SimpleNamespace(sec=0, nanosec=0),
        )
    )
    node = _bare_node(runtime)
    events: list[str] = []
    node._replace_tf_state = lambda: events.append("tf-reset")
    node._clear_markers = lambda stamp, *, reason: (
        events.append(f"markers:{reason}") or True
    )

    node._reset_sequence(
        SimpleNamespace(
            reason="clock_jump",
            timestamp_ns=50,
            generation=1,
        )
    )

    assert events == ["tf-reset", "markers:clock_jump reset"]


def test_reset_sequence_resets_tracker_once_with_central_generation() -> None:
    valid = {"value": True}
    runtime = _node_test_runtime(valid)
    runtime.message_types = SimpleNamespace(
        Header=lambda: SimpleNamespace(
            stamp=SimpleNamespace(sec=0, nanosec=0),
        )
    )
    node = _bare_node(runtime)
    events = []
    node._tracker = SimpleNamespace(
        reset=lambda *, reason, generation=None: events.append(
            ("tracker", reason, generation)
        )
    )
    node._replace_tf_state = lambda: events.append(("tf",))
    node._clear_markers = lambda stamp, *, reason: (
        events.append(("raw", reason)) or True
    )
    node._clear_tracked_markers = lambda stamp, *, reason: (
        events.append(("tracked", reason)) or True
    )

    node._reset_sequence(
        SimpleNamespace(
            reason="point_timestamp",
            timestamp_ns=50,
            generation=7,
        )
    )

    assert events == [
        ("tf",),
        ("tracker", "point_timestamp reset", 7),
        ("raw", "point_timestamp reset"),
        ("tracked", "point_timestamp reset"),
    ]


@pytest.mark.parametrize(
    ("status", "expected_inference_ms"),
    [("empty_after_range_filter", None), ("success", 9.0)],
)
def test_adapter_reports_inference_only_when_model_execution_occurs(
    monkeypatch,
    status: str,
    expected_inference_ms: float | None,
) -> None:
    valid = {"value": True}
    runtime = _node_test_runtime(valid)
    node = _bare_node(runtime)
    node._config = _config()
    node._current_tf_buffer = lambda: object()
    node._clear_markers = lambda stamp, *, reason: True
    node._detector = SimpleNamespace(
        detect=lambda frame: SimpleNamespace(status=status, detector_ms=9.0)
    )
    calibration = object()
    frame = SimpleNamespace(decode_ms=2.0)
    monkeypatch.setattr(
        ros2_node,
        "_lookup_calibration",
        lambda *args, **kwargs: (calibration, 0.5),
    )
    monkeypatch.setattr(ros2_node, "pointcloud2_to_frame", lambda *args, **kwargs: frame)
    message = SimpleNamespace(header=SimpleNamespace(stamp=object()))
    item = SimpleNamespace(payload=message, frame_index=0, generation=0)

    result = node._process_item(item)

    assert result.tf_lookup_ms == 0.5
    assert result.conversion_ms == 2.0
    assert result.inference_ms == expected_inference_ms


def test_tracking_failure_does_not_suppress_valid_raw_detector_output() -> None:
    valid = {"value": True}
    runtime = _node_test_runtime(valid)
    node = _bare_node(runtime)
    tracker_events = []

    def fail_tracking(*args, **kwargs):
        raise ValueError("synthetic tracking failure")

    node._tracker = SimpleNamespace(
        update=fail_tracking,
        reset=lambda *, reason, generation=None: tracker_events.append(
            (reason, generation)
        ),
    )
    node._failed_tracking_frames = 0
    node._last_tracking_error = ""
    node._detections_publisher = _FakePublisher()
    node._markers_publisher = _FakePublisher()
    node._tracked_detections_publisher = _FakePublisher()
    node._tracked_markers_publisher = _FakePublisher()
    node._model_cloud_publisher = None
    node._builder = SimpleNamespace(
        detection_array=lambda *args, **kwargs: "raw detections",
        marker_array=lambda *args, **kwargs: "raw markers",
        clear_markers=lambda *, stamp: "clear raw",
        clear_tracked_markers=lambda *, stamp: "clear tracked",
    )
    product = ros2_node._PublishedFrame(
        source_message=SimpleNamespace(header=SimpleNamespace(stamp=object())),
        normalized_frame=SimpleNamespace(points=None),
        detections=SimpleNamespace(detection_count=2),
        calibration=object(),
        generation=0,
    )

    node._publish_frame(product)

    assert node._detections_publisher.messages == ["raw detections"]
    assert node._markers_publisher.messages == ["raw markers"]
    assert node._tracked_detections_publisher.messages == []
    assert node._tracked_markers_publisher.messages == ["clear tracked"]
    assert node._previous_detection_count == 2
    assert node._failed_tracking_frames == 1
    assert "synthetic tracking failure" in node._last_tracking_error
    assert tracker_events == [("tracking failure", None)]
