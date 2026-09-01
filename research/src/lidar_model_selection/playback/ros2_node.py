"""One-model ROS2/Foxglove adapter for Kaposvar PointCloud2 playback.

ROS and ML framework imports are deliberately deferred until ``main`` builds
the node. Importing this module, parsing ``--help``, and unit tests need
neither a sourced ROS environment nor a GPU.
"""

from __future__ import annotations

import argparse
import importlib
import re
import threading
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import Any, Callable, Sequence

from .formats.ros2_mcap import (
    BASE_FRAME,
    POINT_TOPIC,
    calibration_from_transform,
    pointcloud2_to_frame,
    pointcloud_header_timestamp,
)
from .model_registry import finalist_aliases, finalist_spec
from .normalization import KAPOSVAR_FEATURE_PROFILE
from .ros_messages import RosMessageBuilder, RosMessageTypes
from .ros_processing import ProcessingStageError, ProcessingTimings
from .tracking import TrackerConfig


_CUDA_DEVICE = re.compile(r"cuda:[0-9]+\Z")


@dataclass(frozen=True, slots=True)
class RosNodeConfig:
    model: str
    runs_root: Path
    device: str
    input_topic: str
    output_prefix: str
    base_frame: str
    feature_profile: str
    score_threshold: float
    processing_policy: str
    queue_capacity: int
    publish_model_cloud: bool
    tf_timeout_seconds: float
    diagnostics_period_seconds: float
    enable_tracking: bool = False
    track_min_hits: int = 2
    track_max_missed: int = 3
    track_max_gap_seconds: float = 0.75
    track_association_distance: float = 4.0
    track_smoothing: float = 0.65
    track_trail_length: int = 20
    checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.model not in finalist_aliases():
            raise ValueError(f"model must be one of: {', '.join(finalist_aliases())}")
        if not isinstance(self.runs_root, Path) or not self.runs_root.is_absolute():
            raise ValueError("runs_root must be an absolute path")
        if not isinstance(self.device, str) or _CUDA_DEVICE.fullmatch(self.device) is None:
            raise ValueError("device must be an explicit CUDA device such as cuda:0")
        if self.input_topic != POINT_TOPIC:
            raise ValueError(f"input_topic must be exactly {POINT_TOPIC}")
        if self.output_prefix != f"/centerpoint/{self.model}":
            raise ValueError("output_prefix must exactly bind the selected model alias")
        if self.base_frame != BASE_FRAME:
            raise ValueError(f"base_frame must be exactly {BASE_FRAME}")
        if self.feature_profile != KAPOSVAR_FEATURE_PROFILE:
            raise ValueError(
                f"feature_profile must be exactly {KAPOSVAR_FEATURE_PROFILE}"
            )
        if not isfinite(self.score_threshold) or not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be finite and in [0, 1]")
        if self.processing_policy not in ("all", "latest"):
            raise ValueError("processing_policy must be 'all' or 'latest'")
        if isinstance(self.queue_capacity, bool) or self.queue_capacity <= 0:
            raise ValueError("queue_capacity must be a positive integer")
        if self.processing_policy == "latest" and self.queue_capacity != 1:
            raise ValueError("latest processing requires queue_capacity=1")
        if not isinstance(self.publish_model_cloud, bool):
            raise TypeError("publish_model_cloud must be a boolean")
        if not isinstance(self.enable_tracking, bool):
            raise TypeError("enable_tracking must be a boolean")
        if self.checkpoint_sha256 is not None and (
            not isinstance(self.checkpoint_sha256, str)
            or len(self.checkpoint_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.checkpoint_sha256
            )
        ):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
        for value, name in (
            (self.tf_timeout_seconds, "tf_timeout_seconds"),
            (self.diagnostics_period_seconds, "diagnostics_period_seconds"),
        ):
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        TrackerConfig(
            min_confirmed_hits=self.track_min_hits,
            max_missed_frames=self.track_max_missed,
            max_time_gap_seconds=self.track_max_gap_seconds,
            association_distance_meters=self.track_association_distance,
            position_smoothing=self.track_smoothing,
            score_smoothing=self.track_smoothing,
            trail_length=self.track_trail_length,
        )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish one protected CenterPoint finalist on standard ROS2 topics.",
    )
    parser.add_argument("--model", required=True, choices=finalist_aliases())
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--input-topic", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--base-frame", required=True)
    parser.add_argument("--feature-profile", required=True)
    parser.add_argument("--score-threshold", required=True, type=float)
    parser.add_argument(
        "--processing-policy",
        required=True,
        choices=("all", "latest"),
    )
    parser.add_argument("--queue-capacity", required=True, type=int)
    cloud = parser.add_mutually_exclusive_group(required=True)
    cloud.add_argument(
        "--publish-model-cloud",
        action="store_true",
        dest="publish_model_cloud",
    )
    cloud.add_argument(
        "--no-publish-model-cloud",
        action="store_false",
        dest="publish_model_cloud",
    )
    parser.add_argument("--tf-timeout-seconds", type=float, default=0.2)
    parser.add_argument("--diagnostics-period-seconds", type=float, default=1.0)
    parser.add_argument(
        "--checkpoint-sha256",
        help="Require the protected registry checkpoint to have this SHA-256.",
    )
    parser.add_argument(
        "--enable-tracking",
        action="store_true",
        help="Publish stable-ID tracked detections, markers, and diagnostics.",
    )
    parser.add_argument("--track-min-hits", type=int, default=2)
    parser.add_argument("--track-max-missed", type=int, default=3)
    parser.add_argument("--track-max-gap-seconds", type=float, default=0.75)
    parser.add_argument(
        "--track-association-distance",
        type=float,
        default=4.0,
    )
    parser.add_argument("--track-smoothing", type=float, default=0.65)
    parser.add_argument("--track-trail-length", type=int, default=20)
    return parser


def _parse_arguments(
    args: Sequence[str] | None,
) -> tuple[RosNodeConfig, list[str]]:
    parser = _build_argument_parser()
    parsed, ros_args = parser.parse_known_args(args)
    if ros_args and ros_args[0] != "--ros-args":
        parser.error(f"unrecognized arguments: {' '.join(ros_args)}")
    try:
        config = RosNodeConfig(
            model=parsed.model,
            runs_root=parsed.runs_root,
            device=parsed.device,
            input_topic=parsed.input_topic,
            output_prefix=parsed.output_prefix,
            base_frame=parsed.base_frame,
            feature_profile=parsed.feature_profile,
            score_threshold=parsed.score_threshold,
            processing_policy=parsed.processing_policy,
            queue_capacity=parsed.queue_capacity,
            publish_model_cloud=parsed.publish_model_cloud,
            tf_timeout_seconds=parsed.tf_timeout_seconds,
            diagnostics_period_seconds=parsed.diagnostics_period_seconds,
            enable_tracking=parsed.enable_tracking,
            track_min_hits=parsed.track_min_hits,
            track_max_missed=parsed.track_max_missed,
            track_max_gap_seconds=parsed.track_max_gap_seconds,
            track_association_distance=parsed.track_association_distance,
            track_smoothing=parsed.track_smoothing,
            track_trail_length=parsed.track_trail_length,
            checkpoint_sha256=parsed.checkpoint_sha256,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    return config, ros_args


def _build_detector(config: RosNodeConfig, factory: Callable[..., object]) -> object:
    """Bind exactly one CLI model identity to one detector instance."""

    if (
        config.checkpoint_sha256 is not None
        and finalist_spec(config.model).checkpoint_sha256
        != config.checkpoint_sha256
    ):
        raise ValueError(
            "requested checkpoint SHA-256 does not match the protected registry"
        )
    return factory(
        config.model,
        config.runs_root,
        device=config.device,
        score_threshold=config.score_threshold,
    )


def _tracker_config(config: RosNodeConfig) -> TrackerConfig:
    return TrackerConfig(
        min_confirmed_hits=config.track_min_hits,
        max_missed_frames=config.track_max_missed,
        max_time_gap_seconds=config.track_max_gap_seconds,
        association_distance_meters=config.track_association_distance,
        position_smoothing=config.track_smoothing,
        score_smoothing=config.track_smoothing,
        trail_length=config.track_trail_length,
    )


@dataclass(frozen=True, slots=True)
class _RosRuntime:
    rclpy: ModuleType
    Node: type
    Duration: type
    Time: type
    JumpThreshold: type
    Buffer: type
    TransformListener: type
    TransformException: type[BaseException]
    RCLError: type[BaseException]
    ExternalShutdownException: type[BaseException]
    SignalHandlerOptions: type
    PointCloud2: type
    QoSProfile: type
    QoSHistoryPolicy: type
    QoSReliabilityPolicy: type
    QoSDurabilityPolicy: type
    SubscriptionEventCallbacks: type
    message_types: RosMessageTypes


def _load_ros_runtime() -> _RosRuntime:
    """Import required standard ROS interfaces only when launching a node."""

    try:
        rclpy = importlib.import_module("rclpy")
        node_module = importlib.import_module("rclpy.node")
        duration_module = importlib.import_module("rclpy.duration")
        time_module = importlib.import_module("rclpy.time")
        clock_module = importlib.import_module("rclpy.clock")
        qos_module = importlib.import_module("rclpy.qos")
        event_module = importlib.import_module("rclpy.qos_event")
        executor_module = importlib.import_module("rclpy.executors")
        implementation_module = importlib.import_module(
            "rclpy.impl.implementation_singleton"
        )
        signals_module = importlib.import_module("rclpy.signals")
        tf2_ros = importlib.import_module("tf2_ros")
        sensor_msgs = importlib.import_module("sensor_msgs.msg")
        visualization_msgs = importlib.import_module("visualization_msgs.msg")
        diagnostic_msgs = importlib.import_module("diagnostic_msgs.msg")
        geometry_msgs = importlib.import_module("geometry_msgs.msg")
        std_msgs = importlib.import_module("std_msgs.msg")
        try:
            vision_msgs = importlib.import_module("vision_msgs.msg")
        except ImportError as error:
            raise RuntimeError(
                "vision_msgs is unavailable; on the ROS2 Humble host run: "
                "sudo apt update && sudo apt install ros-humble-vision-msgs"
            ) from error
    except RuntimeError:
        raise
    except ImportError as error:
        raise RuntimeError(
            "required ROS2 Python support is unavailable; source "
            "/opt/ros/humble/setup.bash before launching"
        ) from error

    message_types = RosMessageTypes(
        Header=std_msgs.Header,
        Point=geometry_msgs.Point,
        Detection3DArray=vision_msgs.Detection3DArray,
        Detection3D=vision_msgs.Detection3D,
        ObjectHypothesisWithPose=vision_msgs.ObjectHypothesisWithPose,
        MarkerArray=visualization_msgs.MarkerArray,
        Marker=visualization_msgs.Marker,
        DiagnosticArray=diagnostic_msgs.DiagnosticArray,
        DiagnosticStatus=diagnostic_msgs.DiagnosticStatus,
        KeyValue=diagnostic_msgs.KeyValue,
        PointCloud2=sensor_msgs.PointCloud2,
        PointField=sensor_msgs.PointField,
    )
    return _RosRuntime(
        rclpy=rclpy,
        Node=node_module.Node,
        Duration=duration_module.Duration,
        Time=time_module.Time,
        JumpThreshold=clock_module.JumpThreshold,
        Buffer=tf2_ros.Buffer,
        TransformListener=tf2_ros.TransformListener,
        TransformException=tf2_ros.TransformException,
        RCLError=implementation_module.rclpy_implementation.RCLError,
        ExternalShutdownException=executor_module.ExternalShutdownException,
        SignalHandlerOptions=signals_module.SignalHandlerOptions,
        PointCloud2=sensor_msgs.PointCloud2,
        QoSProfile=qos_module.QoSProfile,
        QoSHistoryPolicy=qos_module.QoSHistoryPolicy,
        QoSReliabilityPolicy=qos_module.QoSReliabilityPolicy,
        QoSDurabilityPolicy=qos_module.QoSDurabilityPolicy,
        SubscriptionEventCallbacks=event_module.SubscriptionEventCallbacks,
        message_types=message_types,
    )


@dataclass(frozen=True, slots=True)
class _PublishedFrame:
    source_message: object
    normalized_frame: object
    detections: object
    calibration: object
    generation: int


def _stamp_from_ns(types: RosMessageTypes, timestamp_ns: int) -> object:
    header = types.Header()
    header.stamp.sec = timestamp_ns // 1_000_000_000
    header.stamp.nanosec = timestamp_ns % 1_000_000_000
    return header.stamp


def _qos(runtime: _RosRuntime, *, depth: int, reliable: bool, transient: bool) -> object:
    return runtime.QoSProfile(
        history=runtime.QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=(
            runtime.QoSReliabilityPolicy.RELIABLE
            if reliable
            else runtime.QoSReliabilityPolicy.BEST_EFFORT
        ),
        durability=(
            runtime.QoSDurabilityPolicy.TRANSIENT_LOCAL
            if transient
            else runtime.QoSDurabilityPolicy.VOLATILE
        ),
    )


def _pointcloud_qos(
    runtime: _RosRuntime,
    *,
    policy: str,
    queue_capacity: int,
) -> object:
    """Match DDS acquisition depth to the explicit pending-work policy."""

    if policy not in ("all", "latest"):
        raise ValueError("policy must be 'all' or 'latest'")
    if isinstance(queue_capacity, bool) or queue_capacity <= 0:
        raise ValueError("queue_capacity must be a positive integer")
    depth = queue_capacity if policy == "all" else 1
    return _qos(runtime, depth=depth, reliable=False, transient=False)


def _tf_static_qos(runtime: _RosRuntime) -> object:
    """Retain the complete bounded static-transform replay history."""

    return _qos(
        runtime,
        depth=100,
        reliable=True,
        transient=True,
    )


def _elapsed_ms(clock: Callable[[], float], started_at: float) -> float:
    return max(0.0, (clock() - started_at) * 1000.0)


def _lookup_calibration(
    runtime: _RosRuntime,
    buffer: object,
    message: object,
    config: RosNodeConfig,
    *,
    clock: Callable[[], float] = perf_counter,
) -> tuple[object, float]:
    """Resolve the exact-time sensor transform or fail without an overlay."""

    started_at = clock()
    try:
        transform = buffer.lookup_transform(
            config.base_frame,
            message.header.frame_id,
            runtime.Time.from_msg(message.header.stamp),
            timeout=runtime.Duration(seconds=config.tf_timeout_seconds),
        )
    except runtime.TransformException as error:
        elapsed_ms = _elapsed_ms(clock, started_at)
        raise ProcessingStageError(
            stage="tf_lookup",
            code="missing_or_stale_tf",
            message=str(error) or repr(error),
            timings=ProcessingTimings(tf_lookup_ms=elapsed_ms),
        ) from error
    try:
        calibration = calibration_from_transform(transform)
    except (TypeError, ValueError, AttributeError) as error:
        elapsed_ms = _elapsed_ms(clock, started_at)
        raise ProcessingStageError(
            stage="tf_lookup",
            code="invalid_or_ambiguous_tf",
            message=str(error) or repr(error),
            timings=ProcessingTimings(tf_lookup_ms=elapsed_ms),
        ) from error
    return calibration, _elapsed_ms(clock, started_at)


def _run_with_overlay_guard(
    action: Callable[[], object],
    clear_overlay: Callable[[], None],
) -> object:
    """Clear persistent markers when a frame cannot produce a valid overlay."""

    try:
        return action()
    except Exception:
        clear_overlay()
        raise


def _node_class(runtime: _RosRuntime) -> type:
    from .detector import FinalistDetector
    from .ros_processing import ProcessingCoordinator, ProcessingResult
    from .tracking import OnlineBoxTracker

    class CenterPointDetectionNode(runtime.Node):
        def __init__(self, config: RosNodeConfig) -> None:
            super().__init__(f"centerpoint_{config.model}")
            self._config = config
            self._closed = False
            self._close_lock = threading.Lock()
            self._previous_detection_count = 0
            self._previous_track_ids: set[int] = set()
            self._marker_lock = threading.RLock()
            self._tf_lock = threading.RLock()
            self._tf_buffer: object | None = None
            self._tf_listener: object | None = None
            self._clock_jump_handle: object | None = None
            self._builder = RosMessageBuilder(
                runtime.message_types,
                model_alias=config.model,
                base_frame=config.base_frame,
            )
            self._detector = _build_detector(config, FinalistDetector)
            self._tracker = (
                OnlineBoxTracker(
                    _tracker_config(config),
                    model_alias=config.model,
                )
                if config.enable_tracking
                else None
            )
            self._failed_tracking_frames = 0
            self._last_tracking_error = ""
            self._replace_tf_state()

            detection_qos = _qos(runtime, depth=1, reliable=True, transient=False)
            marker_qos = _qos(runtime, depth=1, reliable=True, transient=True)
            diagnostic_qos = _qos(runtime, depth=10, reliable=True, transient=False)
            cloud_qos = _qos(runtime, depth=1, reliable=False, transient=False)
            self._detections_publisher = self.create_publisher(
                runtime.message_types.Detection3DArray,
                f"{config.output_prefix}/detections",
                detection_qos,
            )
            self._markers_publisher = self.create_publisher(
                runtime.message_types.MarkerArray,
                f"{config.output_prefix}/markers",
                marker_qos,
            )
            self._diagnostics_publisher = self.create_publisher(
                runtime.message_types.DiagnosticArray,
                f"{config.output_prefix}/diagnostics",
                diagnostic_qos,
            )
            self._tracked_detections_publisher = (
                self.create_publisher(
                    runtime.message_types.Detection3DArray,
                    f"{config.output_prefix}/tracked_detections",
                    detection_qos,
                )
                if self._tracker is not None
                else None
            )
            self._tracked_markers_publisher = (
                self.create_publisher(
                    runtime.message_types.MarkerArray,
                    f"{config.output_prefix}/tracked_markers",
                    marker_qos,
                )
                if self._tracker is not None
                else None
            )
            self._tracking_diagnostics_publisher = (
                self.create_publisher(
                    runtime.message_types.DiagnosticArray,
                    f"{config.output_prefix}/tracking_diagnostics",
                    diagnostic_qos,
                )
                if self._tracker is not None
                else None
            )
            self._model_cloud_publisher = (
                self.create_publisher(
                    runtime.message_types.PointCloud2,
                    f"{config.output_prefix}/model_points",
                    cloud_qos,
                )
                if config.publish_model_cloud
                else None
            )
            self._clear_markers(
                self.get_clock().now().to_msg(),
                reason="startup",
            )
            self._clear_tracked_markers(
                self.get_clock().now().to_msg(),
                reason="startup",
            )

            close_detector = getattr(self._detector, "close", None)
            self._coordinator = ProcessingCoordinator(
                self._process_item,
                self._publish_frame,
                policy=config.processing_policy,
                capacity=config.queue_capacity,
                reset=self._reset_sequence,
                close=close_detector if callable(close_detector) else None,
                thread_name=f"centerpoint-{config.model}-worker",
            )
            jump_threshold = runtime.JumpThreshold(
                min_forward=None,
                min_backward=runtime.Duration(nanoseconds=-1),
                on_clock_change=False,
            )
            self._clock_jump_handle = self.get_clock().create_jump_callback(
                jump_threshold,
                post_callback=self._clock_jump,
            )
            event_callbacks = runtime.SubscriptionEventCallbacks(
                message_lost=self._message_lost,
                incompatible_qos=self._incompatible_qos,
            )
            input_qos = _pointcloud_qos(
                runtime,
                policy=config.processing_policy,
                queue_capacity=config.queue_capacity,
            )
            self._subscription = self.create_subscription(
                runtime.PointCloud2,
                config.input_topic,
                self._point_cloud_callback,
                input_qos,
                event_callbacks=event_callbacks,
            )
            self._diagnostic_timer = self.create_timer(
                config.diagnostics_period_seconds,
                self._publish_diagnostics,
            )
            identity = self._detector.identity
            self.get_logger().info(
                f"ready model={config.model} run_id={identity.run_id} "
                f"checkpoint_sha256={identity.checkpoint_sha256} "
                f"device={config.device} policy={config.processing_policy} "
                f"queue_capacity={config.queue_capacity} "
                f"subscription_depth={input_qos.depth} "
                f"tracking={config.enable_tracking}"
            )

        def _context_valid(self) -> bool:
            return bool(runtime.rclpy.ok(context=self.context))

        def _publish_message(
            self,
            publisher: object,
            message: object,
            *,
            description: str,
        ) -> bool:
            if not self._context_valid():
                self.get_logger().warning(
                    f"skipped {description}: ROS context is not valid"
                )
                return False
            try:
                publisher.publish(message)
            except runtime.RCLError:
                if not self._context_valid():
                    self.get_logger().warning(
                        f"skipped {description}: ROS context became invalid"
                    )
                    return False
                raise
            return True

        def _replace_tf_state(self) -> None:
            """Re-subscribe TF so no later-timeline samples survive a rewind."""

            with self._tf_lock:
                previous_listener = self._tf_listener
                self._tf_listener = None
                self._tf_buffer = None
            if previous_listener is not None:
                unregister = getattr(previous_listener, "unregister", None)
                if callable(unregister):
                    unregister()

            new_buffer = runtime.Buffer()
            new_listener = runtime.TransformListener(
                new_buffer,
                self,
                spin_thread=False,
                static_qos=_tf_static_qos(runtime),
            )
            with self._tf_lock:
                self._tf_buffer = new_buffer
                self._tf_listener = new_listener

        def _current_tf_buffer(self) -> object:
            with self._tf_lock:
                buffer = self._tf_buffer
            if buffer is None:
                raise ProcessingStageError(
                    stage="tf_lookup",
                    code="tf_reset_in_progress",
                    message="TF state is being reset for a new playback timeline",
                )
            return buffer

        def _clock_jump(self, jump: object) -> None:
            delta = getattr(getattr(jump, "delta", None), "nanoseconds", None)
            if not isinstance(delta, int) or delta >= 0:
                return
            with self._close_lock:
                if self._closed:
                    return
            now_ns = int(self.get_clock().now().nanoseconds)
            created = self._coordinator.reset_generation(
                reason="clock_jump",
                timestamp_ns=now_ns if now_ns > 0 else None,
            )
            if not created:
                self.get_logger().debug(
                    "backward clock jump matched an existing point-timestamp reset"
                )

        def _point_cloud_callback(self, message: object) -> None:
            try:
                timestamp_ns = pointcloud_header_timestamp(message)
            except Exception as error:
                input_frame = getattr(
                    getattr(message, "header", None),
                    "frame_id",
                    None,
                )
                if not isinstance(input_frame, str) or not input_frame.strip():
                    input_frame = None
                self._coordinator.record_failure(
                    f"invalid_header_timestamp: {error}",
                    input_frame=input_frame,
                )
                self._reset_tracking_and_clear(
                    self.get_clock().now().to_msg(),
                    reason="invalid input timestamp",
                )
                self.get_logger().error(f"rejected PointCloud2: {error}")
                return
            input_frame = getattr(message.header, "frame_id", None)
            if not isinstance(input_frame, str) or not input_frame.strip():
                input_frame = None
            submission = self._coordinator.submit(
                message,
                timestamp_ns=timestamp_ns,
                input_frame=input_frame,
            )
            if not submission.accepted:
                self._clear_markers(
                    message.header.stamp,
                    reason="application queue rejection",
                )
                self._clear_tracked_markers(
                    message.header.stamp,
                    reason="application queue rejection",
                )
                self.get_logger().error(
                    f"PointCloud2 not queued: {submission.reason}; "
                    f"policy={self._config.processing_policy}"
                )

        def _process_item(self, item: object) -> object:
            message = item.payload
            stamp = message.header.stamp

            def process() -> object:
                calibration, tf_lookup_ms = _lookup_calibration(
                    runtime,
                    self._current_tf_buffer(),
                    message,
                    self._config,
                )
                conversion_started_at = perf_counter()
                try:
                    frame = pointcloud2_to_frame(
                        message,
                        session_id=f"live:{self._config.input_topic}",
                        frame_index=item.frame_index,
                        calibration=calibration,
                        feature_profile=self._config.feature_profile,
                        source_key=f"{self._config.input_topic}[{item.frame_index}]",
                    )
                except Exception as error:
                    evidence = getattr(error, "evidence", None)
                    code = getattr(evidence, "code", None)
                    if not isinstance(code, str) or not code:
                        code = type(error).__name__
                    conversion_ms = getattr(evidence, "decode_ms", None)
                    if not isinstance(conversion_ms, (int, float)):
                        conversion_ms = _elapsed_ms(
                            perf_counter,
                            conversion_started_at,
                        )
                    raise ProcessingStageError(
                        stage="conversion",
                        code=code,
                        message=str(error) or repr(error),
                        timings=ProcessingTimings(
                            tf_lookup_ms=tf_lookup_ms,
                            conversion_ms=float(conversion_ms),
                        ),
                    ) from error
                try:
                    result = self._detector.detect(frame)
                except Exception as error:
                    raise ProcessingStageError(
                        stage="inference",
                        code=type(error).__name__,
                        message=str(error) or repr(error),
                        timings=ProcessingTimings(
                            tf_lookup_ms=tf_lookup_ms,
                            conversion_ms=frame.decode_ms,
                        ),
                    ) from error
                inference_ms = (
                    result.detector_ms if result.status == "success" else None
                )
                return ProcessingResult(
                    _PublishedFrame(
                        message,
                        frame,
                        result,
                        calibration,
                        item.generation,
                    ),
                    tf_lookup_ms=tf_lookup_ms,
                    conversion_ms=frame.decode_ms,
                    inference_ms=inference_ms,
                )

            return _run_with_overlay_guard(
                process,
                lambda: self._reset_tracking_and_clear(
                    stamp,
                    reason="processing failure",
                ),
            )

        def _publish_frame(self, product: _PublishedFrame) -> None:
            stamp = product.source_message.header.stamp

            def publish() -> None:
                with self._marker_lock:
                    detections = self._builder.detection_array(
                        product.detections,
                        stamp=stamp,
                        calibration=product.calibration,
                    )
                    markers = self._builder.marker_array(
                        product.detections,
                        stamp=stamp,
                        calibration=product.calibration,
                        previous_detection_count=self._previous_detection_count,
                    )
                    model_cloud = (
                        self._builder.model_point_cloud(
                            product.normalized_frame.points,
                            stamp=stamp,
                            calibration=product.calibration,
                        )
                        if self._model_cloud_publisher is not None
                        else None
                    )
                    published = self._publish_message(
                        self._detections_publisher,
                        detections,
                        description="detection array publication",
                    )
                    published = self._publish_message(
                        self._markers_publisher,
                        markers,
                        description="marker array publication",
                    ) and published
                    if model_cloud is not None and self._model_cloud_publisher is not None:
                        published = self._publish_message(
                            self._model_cloud_publisher,
                            model_cloud,
                            description="model cloud publication",
                        ) and published
                    if not published:
                        raise ProcessingStageError(
                            stage="publication",
                            code="context_invalid",
                            message="ROS context became invalid before frame publication completed",
                        )
                    self._previous_detection_count = (
                        product.detections.detection_count
                    )
                    self._publish_tracking(product, stamp=stamp)

            _run_with_overlay_guard(
                publish,
                lambda: self._reset_tracking_and_clear(
                    stamp,
                    reason="publication failure",
                ),
            )

        def _publish_tracking(self, product: _PublishedFrame, *, stamp: object) -> None:
            """Publish derived tracking without changing raw-pipeline success."""

            if self._tracker is None:
                return
            try:
                tracked_frame = self._tracker.update(
                    product.detections,
                    calibration=product.calibration,
                    generation=product.generation,
                )
                tracked_detections = self._builder.tracked_detection_array(
                    tracked_frame,
                    stamp=stamp,
                )
                tracked_markers = self._builder.tracked_marker_array(
                    tracked_frame,
                    stamp=stamp,
                    previous_track_ids=self._previous_track_ids,
                )
                if self._tracked_detections_publisher is None:
                    raise RuntimeError("tracked detection publisher is unavailable")
                if self._tracked_markers_publisher is None:
                    raise RuntimeError("tracked marker publisher is unavailable")
                published = self._publish_message(
                    self._tracked_detections_publisher,
                    tracked_detections,
                    description="tracked detection array publication",
                )
                published = self._publish_message(
                    self._tracked_markers_publisher,
                    tracked_markers,
                    description="tracked marker array publication",
                ) and published
                if not published:
                    raise RuntimeError(
                        "ROS context became invalid during tracked publication"
                    )
                self._previous_track_ids = {
                    track.track_id for track in tracked_frame.visible_tracks
                }
                self._last_tracking_error = ""
            except Exception as error:
                self._failed_tracking_frames += 1
                self._last_tracking_error = (
                    f"{type(error).__name__}: {str(error) or repr(error)}"
                )
                self._reset_tracking(reason="tracking failure")
                self._clear_tracked_markers(stamp, reason="tracking failure")
                self.get_logger().error(
                    "tracking output suppressed; raw detector output remains valid: "
                    f"{self._last_tracking_error}"
                )

        def _clear_markers(self, stamp: object, *, reason: str) -> bool:
            with self._marker_lock:
                self._previous_detection_count = 0
                return self._publish_message(
                    self._markers_publisher,
                    self._builder.clear_markers(stamp=stamp),
                    description=f"marker DELETEALL ({reason})",
                )

        def _clear_tracked_markers(self, stamp: object, *, reason: str) -> bool:
            with self._marker_lock:
                self._previous_track_ids.clear()
                if self._tracked_markers_publisher is None:
                    return True
                return self._publish_message(
                    self._tracked_markers_publisher,
                    self._builder.clear_tracked_markers(stamp=stamp),
                    description=f"tracked marker DELETEALL ({reason})",
                )

        def _reset_tracking(
            self,
            *,
            reason: str,
            generation: int | None = None,
        ) -> None:
            if self._tracker is not None:
                self._tracker.reset(reason=reason, generation=generation)

        def _reset_tracking_and_clear(self, stamp: object, *, reason: str) -> None:
            self._reset_tracking(reason=reason)
            self._clear_markers(stamp, reason=reason)
            self._clear_tracked_markers(stamp, reason=reason)

        def _reset_sequence(self, event: object) -> None:
            stamp = (
                _stamp_from_ns(runtime.message_types, event.timestamp_ns)
                if event.timestamp_ns is not None
                else self.get_clock().now().to_msg()
            )
            try:
                self._replace_tf_state()
            finally:
                self._reset_tracking(
                    reason=f"{event.reason} reset",
                    generation=event.generation,
                )
                self._clear_markers(stamp, reason=f"{event.reason} reset")
                self._clear_tracked_markers(
                    stamp,
                    reason=f"{event.reason} reset",
                )
            self.get_logger().info(
                f"playback reset reason={event.reason} generation={event.generation} "
                f"timestamp_ns={event.timestamp_ns}; pending frames, TF state, and markers cleared"
            )

        def _identity_values(self) -> dict[str, object]:
            identity = self._detector.identity
            snapshot = asdict(self._coordinator.snapshot())
            timestamp = snapshot.get("last_input_timestamp_ns")
            input_frame = snapshot.get("last_input_frame")
            age_ms: object = "n/a"
            if isinstance(timestamp, int):
                delta = self.get_clock().now().nanoseconds - timestamp
                if delta >= 0:
                    age_ms = delta / 1_000_000.0
            return {
                "model_alias": self._config.model,
                "run_id": identity.run_id,
                "selected_checkpoint_sha256": identity.checkpoint_sha256,
                "checkpoint_sha256": identity.checkpoint_sha256,
                "device": self._config.device,
                "score_threshold": self._config.score_threshold,
                "processing_policy": self._config.processing_policy,
                "queue_capacity": self._config.queue_capacity,
                "input_frame": input_frame or "n/a",
                "input_timestamp_ns": timestamp if timestamp is not None else "n/a",
                "message_age_ms": age_ms,
                "received_frames": snapshot["received"],
                "processed_frames": snapshot["processed"],
                "dropped_frames": snapshot["dropped"],
                "failed_frames": snapshot["failed"],
                "rejected_frames": snapshot["rejected"],
                "middleware_lost_frames": snapshot["middleware_lost"],
                "loop_reset_count": snapshot["loops"],
                "last_reset_reason": snapshot["last_reset_reason"],
                "queue_ms": snapshot["last_queue_ms"],
                "tf_lookup_ms": snapshot["last_tf_lookup_ms"],
                "conversion_ms": snapshot["last_conversion_ms"],
                "inference_ms": snapshot["last_inference_ms"],
                "publication_ms": snapshot["last_publish_ms"],
                "end_to_end_ms": snapshot["last_end_to_end_ms"],
                "last_error": ": ".join(
                    str(part)
                    for part in (
                        snapshot["last_error_stage"],
                        snapshot["last_error_code"],
                        snapshot["last_error_message"],
                    )
                    if part
                ),
                **snapshot,
            }

        def _tracking_values(self) -> dict[str, object] | None:
            if self._tracker is None:
                return None
            identity = self._detector.identity
            tracker = asdict(self._tracker.snapshot())
            coordinator = self._coordinator.snapshot()
            last_error = ": ".join(
                part
                for part in (
                    coordinator.last_error_stage,
                    coordinator.last_error_code,
                    coordinator.last_error_message,
                )
                if part
            )
            return {
                "model_alias": self._config.model,
                "run_id": identity.run_id,
                "selected_checkpoint_sha256": identity.checkpoint_sha256,
                "checkpoint_sha256": identity.checkpoint_sha256,
                "generation": coordinator.generation,
                "active_tracks": tracker["active_tracks"],
                "confirmed_tracks": tracker["confirmed_tracks"],
                "tentative_tracks": tracker["tentative_tracks"],
                "coasting_tracks": tracker["coasting_tracks"],
                "created_tracks_total": tracker["created_tracks_total"],
                "deleted_tracks_total": tracker["deleted_tracks_total"],
                "created_tracks": tracker["created_tracks"],
                "removed_tracks": tracker["removed_tracks"],
                "matches": tracker["matches"],
                "misses": tracker["misses"],
                "unmatched_detections": tracker["unmatched_detections"],
                "unmatched_tracks": tracker["unmatched_tracks"],
                "tracking_reset_count": tracker["reset_count"],
                "last_tracking_reset_reason": tracker["last_reset_reason"],
                "tracking_association_ms": tracker["association_ms"],
                "tracking_update_ms": tracker["update_ms"],
                "last_tracking_timestamp_ns": tracker["last_timestamp_ns"],
                "last_tracking_dt_seconds": tracker["last_dt_seconds"],
                "maximum_observed_gap_seconds": tracker[
                    "maximum_observed_gap_seconds"
                ],
                "tracked_frames_total": tracker["tracked_frames_total"],
                "failed_tracking_frames": self._failed_tracking_frames,
                "last_error": self._last_tracking_error or last_error,
            }

        def _publish_diagnostics(self) -> None:
            values = self._identity_values()
            message = self._builder.diagnostic_array(
                values,
                stamp=self.get_clock().now().to_msg(),
            )
            if not self._publish_message(
                self._diagnostics_publisher,
                message,
                description="diagnostic publication",
            ):
                return
            tracking_values = self._tracking_values()
            if tracking_values is not None:
                if self._tracking_diagnostics_publisher is None:
                    raise RuntimeError(
                        "tracking diagnostics publisher is unavailable"
                    )
                self._publish_message(
                    self._tracking_diagnostics_publisher,
                    self._builder.tracking_diagnostic_array(
                        tracking_values,
                        stamp=self.get_clock().now().to_msg(),
                    ),
                    description="tracking diagnostic publication",
                )
            self.get_logger().info(
                "diagnostics "
                f"model={values['model_alias']} run_id={values['run_id']} "
                f"checkpoint_sha256={values['selected_checkpoint_sha256']} "
                f"device={values['device']} threshold={values['score_threshold']} "
                f"policy={values['processing_policy']} "
                f"input={values['input_frame']}@{values['input_timestamp_ns']} "
                f"received={values['received_frames']} "
                f"processed={values['processed_frames']} "
                f"dropped={values['dropped_frames']} "
                f"rejected={values['rejected_frames']} "
                f"middleware_lost={values['middleware_lost_frames']} "
                f"failed={values['failed_frames']} loops={values['loop_reset_count']} "
                f"queue_ms={values['queue_ms']} "
                f"tf_lookup_ms={values['tf_lookup_ms']} "
                f"conversion_ms={values['conversion_ms']} "
                f"inference_ms={values['inference_ms']} "
                f"publication_ms={values['publication_ms']} "
                f"message_age_ms={values['message_age_ms']} "
                f"last_error={values['last_error'] or 'none'}"
            )
            if tracking_values is not None:
                self.get_logger().info(
                    "tracking diagnostics "
                    f"active={tracking_values['active_tracks']} "
                    f"confirmed={tracking_values['confirmed_tracks']} "
                    f"tentative={tracking_values['tentative_tracks']} "
                    f"coasting={tracking_values['coasting_tracks']} "
                    f"matches={tracking_values['matches']} "
                    f"unmatched_detections={tracking_values['unmatched_detections']} "
                    f"resets={tracking_values['tracking_reset_count']} "
                    f"association_ms={tracking_values['tracking_association_ms']} "
                    f"update_ms={tracking_values['tracking_update_ms']}"
                )

        def _message_lost(self, event: object) -> None:
            count = max(1, int(getattr(event, "total_count_change", 1)))
            self._coordinator.record_middleware_loss(
                count,
                reason="middleware reported PointCloud2 loss",
            )
            self._clear_markers(
                self.get_clock().now().to_msg(),
                reason="middleware message loss",
            )
            self._clear_tracked_markers(
                self.get_clock().now().to_msg(),
                reason="middleware message loss",
            )
            self.get_logger().error(f"middleware reported {count} lost PointCloud2 message(s)")

        def _incompatible_qos(self, event: object) -> None:
            self._coordinator.record_failure(
                f"incompatible_input_qos: policy_kind={getattr(event, 'last_policy_kind', 'unknown')}"
            )
            self._reset_tracking_and_clear(
                self.get_clock().now().to_msg(),
                reason="incompatible input QoS",
            )
            self.get_logger().error("input subscription has incompatible QoS")

        def close(self) -> None:
            with self._close_lock:
                if self._closed:
                    return
                self._closed = True
            context_valid = self._context_valid()
            jump_handle = self._clock_jump_handle
            self._clock_jump_handle = None
            if context_valid and jump_handle is not None:
                unregister_jump = getattr(jump_handle, "unregister", None)
                if callable(unregister_jump):
                    unregister_jump()
            try:
                self._coordinator.close(drain=False)
            finally:
                try:
                    shutdown_stamp = (
                        self.get_clock().now().to_msg()
                        if self._context_valid()
                        else _stamp_from_ns(runtime.message_types, 0)
                    )
                    self._reset_tracking_and_clear(
                        shutdown_stamp,
                        reason="shutdown",
                    )
                finally:
                    with self._tf_lock:
                        listener = self._tf_listener
                        self._tf_listener = None
                        self._tf_buffer = None
                    if self._context_valid() and listener is not None:
                        unregister = getattr(listener, "unregister", None)
                        if callable(unregister):
                            unregister()
                    elif listener is not None:
                        self.get_logger().warning(
                            "skipped explicit TF listener unregister: "
                            "ROS context is not valid"
                        )

    return CenterPointDetectionNode


def main(args: Sequence[str] | None = None) -> None:
    config, ros_args = _parse_arguments(args)
    runtime = _load_ros_runtime()
    runtime.rclpy.init(
        args=ros_args,
        signal_handler_options=runtime.SignalHandlerOptions.NO,
    )
    node: Any | None = None
    try:
        node = _node_class(runtime)(config)
        runtime.rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except runtime.ExternalShutdownException:
        pass
    finally:
        try:
            if node is not None:
                try:
                    node.close()
                finally:
                    node.destroy_node()
        finally:
            runtime.rclpy.try_shutdown()


if __name__ == "__main__":
    main()
