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
from types import ModuleType
from typing import Any, Callable, Sequence

from .formats.ros2_mcap import (
    BASE_FRAME,
    POINT_TOPIC,
    calibration_from_transform,
    pointcloud2_to_frame,
    pointcloud_header_timestamp,
)
from .model_registry import finalist_aliases
from .normalization import KAPOSVAR_FEATURE_PROFILE
from .ros_messages import RosMessageBuilder, RosMessageTypes


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
        for value, name in (
            (self.tf_timeout_seconds, "tf_timeout_seconds"),
            (self.diagnostics_period_seconds, "diagnostics_period_seconds"),
        ):
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


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
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    return config, ros_args


def _build_detector(config: RosNodeConfig, factory: Callable[..., object]) -> object:
    """Bind exactly one CLI model identity to one detector instance."""

    return factory(
        config.model,
        config.runs_root,
        device=config.device,
        score_threshold=config.score_threshold,
    )


@dataclass(frozen=True, slots=True)
class _RosRuntime:
    rclpy: ModuleType
    Node: type
    Duration: type
    Time: type
    Buffer: type
    TransformListener: type
    TransformException: type[BaseException]
    PointCloud2: type
    qos_profile_sensor_data: object
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
        qos_module = importlib.import_module("rclpy.qos")
        event_module = importlib.import_module("rclpy.qos_event")
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
        Buffer=tf2_ros.Buffer,
        TransformListener=tf2_ros.TransformListener,
        TransformException=tf2_ros.TransformException,
        PointCloud2=sensor_msgs.PointCloud2,
        qos_profile_sensor_data=qos_module.qos_profile_sensor_data,
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


def _lookup_calibration(
    runtime: _RosRuntime,
    buffer: object,
    message: object,
    config: RosNodeConfig,
) -> object:
    """Resolve the exact-time sensor transform or fail without an overlay."""

    try:
        transform = buffer.lookup_transform(
            config.base_frame,
            message.header.frame_id,
            runtime.Time.from_msg(message.header.stamp),
            timeout=runtime.Duration(seconds=config.tf_timeout_seconds),
        )
    except runtime.TransformException as error:
        raise RuntimeError(f"missing_or_stale_tf: {error}") from error
    try:
        return calibration_from_transform(transform)
    except (TypeError, ValueError, AttributeError) as error:
        raise RuntimeError(f"invalid_or_ambiguous_tf: {error}") from error


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

    class CenterPointDetectionNode(runtime.Node):
        def __init__(self, config: RosNodeConfig) -> None:
            super().__init__(f"centerpoint_{config.model}")
            self._config = config
            self._closed = False
            self._previous_detection_count = 0
            self._marker_lock = threading.RLock()
            self._builder = RosMessageBuilder(
                runtime.message_types,
                model_alias=config.model,
                base_frame=config.base_frame,
            )
            self._detector = _build_detector(config, FinalistDetector)
            self._tf_buffer = runtime.Buffer()
            self._tf_listener = runtime.TransformListener(
                self._tf_buffer,
                self,
                spin_thread=False,
            )

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
            self._model_cloud_publisher = (
                self.create_publisher(
                    runtime.message_types.PointCloud2,
                    f"{config.output_prefix}/model_points",
                    cloud_qos,
                )
                if config.publish_model_cloud
                else None
            )
            self._clear_markers(self.get_clock().now().to_msg())

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
            event_callbacks = runtime.SubscriptionEventCallbacks(
                message_lost=self._message_lost,
                incompatible_qos=self._incompatible_qos,
            )
            self._subscription = self.create_subscription(
                runtime.PointCloud2,
                config.input_topic,
                self._point_cloud_callback,
                runtime.qos_profile_sensor_data,
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
                f"queue_capacity={config.queue_capacity}"
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
                self._clear_markers(self.get_clock().now().to_msg())
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
                self._clear_markers(message.header.stamp)
                self.get_logger().error(
                    f"PointCloud2 not queued: {submission.reason}; "
                    f"policy={self._config.processing_policy}"
                )

        def _process_item(self, item: object) -> object:
            message = item.payload
            stamp = message.header.stamp

            def process() -> object:
                calibration = _lookup_calibration(
                    runtime,
                    self._tf_buffer,
                    message,
                    self._config,
                )
                frame = pointcloud2_to_frame(
                    message,
                    session_id=f"live:{self._config.input_topic}",
                    frame_index=item.frame_index,
                    calibration=calibration,
                    feature_profile=self._config.feature_profile,
                    source_key=f"{self._config.input_topic}[{item.frame_index}]",
                )
                result = self._detector.detect(frame)
                return ProcessingResult(
                    _PublishedFrame(message, frame, result, calibration),
                    conversion_ms=frame.decode_ms,
                    detector_ms=result.detector_ms,
                )

            return _run_with_overlay_guard(
                process,
                lambda: self._clear_markers(stamp),
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
                    self._detections_publisher.publish(detections)
                    self._markers_publisher.publish(markers)
                    if model_cloud is not None:
                        self._model_cloud_publisher.publish(model_cloud)
                    self._previous_detection_count = (
                        product.detections.detection_count
                    )

            _run_with_overlay_guard(
                publish,
                lambda: self._clear_markers(stamp),
            )

        def _clear_markers(self, stamp: object) -> None:
            with self._marker_lock:
                self._previous_detection_count = 0
                self._markers_publisher.publish(
                    self._builder.clear_markers(stamp=stamp)
                )

        def _reset_sequence(self, event: object) -> None:
            stamp = _stamp_from_ns(runtime.message_types, event.timestamp_ns)
            self._clear_markers(stamp)
            self.get_logger().info(
                f"timestamp reset generation={event.generation} "
                f"timestamp_ns={event.timestamp_ns}; pending frames and markers cleared"
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
                "loop_reset_count": snapshot["loops"],
                "conversion_ms": snapshot["last_conversion_ms"],
                "detector_ms": snapshot["last_detector_ms"],
                "publish_ms": snapshot["last_publish_ms"],
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

        def _publish_diagnostics(self) -> None:
            values = self._identity_values()
            message = self._builder.diagnostic_array(
                values,
                stamp=self.get_clock().now().to_msg(),
            )
            self._diagnostics_publisher.publish(message)
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
                f"failed={values['failed_frames']} loops={values['loop_reset_count']} "
                f"conversion_ms={values['conversion_ms']} "
                f"detector_ms={values['detector_ms']} "
                f"publish_ms={values['publish_ms']} "
                f"message_age_ms={values['message_age_ms']} "
                f"last_error={values['last_error'] or 'none'}"
            )

        def _message_lost(self, event: object) -> None:
            count = max(1, int(getattr(event, "total_count_change", 1)))
            self._coordinator.record_drop(count, reason="middleware_message_lost")
            self._clear_markers(self.get_clock().now().to_msg())
            self.get_logger().error(f"middleware reported {count} lost PointCloud2 message(s)")

        def _incompatible_qos(self, event: object) -> None:
            self._coordinator.record_failure(
                f"incompatible_input_qos: policy_kind={getattr(event, 'last_policy_kind', 'unknown')}"
            )
            self._clear_markers(self.get_clock().now().to_msg())
            self.get_logger().error("input subscription has incompatible QoS")

        def close(self) -> None:
            if self._closed:
                return
            self._closed = True
            try:
                self._coordinator.close(drain=False)
            finally:
                try:
                    self._clear_markers(self.get_clock().now().to_msg())
                finally:
                    unregister = getattr(self._tf_listener, "unregister", None)
                    if callable(unregister):
                        unregister()

    return CenterPointDetectionNode


def main(args: Sequence[str] | None = None) -> None:
    config, ros_args = _parse_arguments(args)
    runtime = _load_ros_runtime()
    runtime.rclpy.init(args=ros_args)
    node: Any | None = None
    try:
        node = _node_class(runtime)(config)
        runtime.rclpy.spin(node)
    finally:
        try:
            if node is not None:
                try:
                    node.close()
                finally:
                    node.destroy_node()
        finally:
            if runtime.rclpy.ok():
                runtime.rclpy.shutdown()


if __name__ == "__main__":
    main()
