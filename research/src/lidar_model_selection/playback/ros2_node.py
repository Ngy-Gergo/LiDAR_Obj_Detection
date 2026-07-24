from collections.abc import Sequence
from math import cos, isfinite, sin
from pathlib import Path

import numpy
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

from .detector import Mmdet3dDetector
from .frame_source import DirectoryFrameSource
from .results import FrameResult


class KittiDetectionNode(Node):
    def __init__(self) -> None:
        super().__init__("kitti_detection")

        config_path = self.declare_parameter("config_path", "").value
        checkpoint_path = self.declare_parameter("checkpoint_path", "").value
        input_dir = self.declare_parameter("input_dir", "").value
        device = self.declare_parameter("device", "cuda:0").value
        score_threshold = self.declare_parameter("score_threshold", 0.3).value
        extension = self.declare_parameter("extension", ".bin").value
        max_frames = self.declare_parameter("max_frames", 0).value
        playback_hz = self.declare_parameter("playback_hz", 5.0).value
        loop = self.declare_parameter("loop", True).value
        ros_frame_id = self.declare_parameter("frame_id", "lidar").value

        if not config_path.strip():
            raise ValueError("config_path must contain non-whitespace text")
        if not checkpoint_path.strip():
            raise ValueError("checkpoint_path must contain non-whitespace text")
        if not input_dir.strip():
            raise ValueError("input_dir must contain non-whitespace text")
        if max_frames < 0:
            raise ValueError("max_frames must be greater than or equal to zero")
        if not isfinite(playback_hz) or playback_hz <= 0:
            raise ValueError("playback_hz must be finite and greater than zero")
        if not ros_frame_id.strip():
            raise ValueError("frame_id must contain non-whitespace text")

        source = DirectoryFrameSource(
            directory=Path(input_dir),
            extension=extension,
        )
        self._frames = source.list_frames(
            limit=None if max_frames == 0 else max_frames,
        )
        if not self._frames:
            raise ValueError("no matching LiDAR frames were found")

        self._detector = Mmdet3dDetector(
            config_path=Path(config_path),
            checkpoint_path=Path(checkpoint_path),
            device=device,
            score_threshold=score_threshold,
        )
        self._point_cloud_publisher = self.create_publisher(
            PointCloud2,
            "/lidar/points",
            qos_profile_sensor_data,
        )
        self._marker_publisher = self.create_publisher(
            MarkerArray,
            "/lidar/detections",
            10,
        )
        self._ros_frame_id = ros_frame_id
        self._loop = loop
        self._current_index = 0
        self._timer = self.create_timer(
            1.0 / playback_hz,
            self._process_next_frame,
        )

        self.get_logger().info(
            f"Ready to process {len(self._frames)} frame(s) at "
            f"{playback_hz:.2f} Hz; loop={loop}."
        )

    def _create_point_cloud_message(
        self,
        header: Header,
        points: numpy.ndarray,
    ) -> PointCloud2:
        fields = [
            PointField(
                name="x",
                offset=0,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="y",
                offset=4,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="z",
                offset=8,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="intensity",
                offset=12,
                datatype=PointField.FLOAT32,
                count=1,
            ),
        ]
        return point_cloud2.create_cloud(
            header,
            fields,
            points,
        )

    def _create_marker_array(
        self,
        header: Header,
        result: FrameResult,
    ) -> MarkerArray:
        clear_marker = Marker()
        clear_marker.header = header
        clear_marker.action = Marker.DELETEALL
        markers = [clear_marker]

        for index, detection in enumerate(result.detections):
            x, y, z, dx, dy, dz, yaw = detection.box

            marker = Marker()
            marker.header = header
            marker.ns = "detections"
            marker.id = index
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = z + dz / 2.0

            marker.scale.x = dx
            marker.scale.y = dy
            marker.scale.z = dz

            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = sin(yaw / 2.0)
            marker.pose.orientation.w = cos(yaw / 2.0)

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.5

            markers.append(marker)

        return MarkerArray(markers=markers)

    def _process_next_frame(self) -> None:
        frame = self._frames[self._current_index]

        points = numpy.fromfile(frame.path, dtype=numpy.float32)
        if points.size % 4 != 0:
            raise ValueError(
                f"LiDAR frame must contain x, y, z, intensity float32 values: {frame.path}"
            )
        points = points.reshape(-1, 4)

        result = self._detector.detect(frame)

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self._ros_frame_id

        point_cloud_message = self._create_point_cloud_message(header, points)
        marker_array = self._create_marker_array(header, result)

        self._point_cloud_publisher.publish(point_cloud_message)
        self._marker_publisher.publish(marker_array)
        self.get_logger().info(
            f"{frame.frame_id}: detections={len(result.detections)} "
            f"inference_ms={result.inference_ms:.2f}"
        )

        self._current_index += 1
        if self._current_index == len(self._frames):
            if self._loop:
                self._current_index = 0
            else:
                self._timer.cancel()
                self.get_logger().info("Playback complete.")


def main(args: Sequence[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None

    try:
        node = KittiDetectionNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
