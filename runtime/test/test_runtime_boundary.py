import importlib
import sys


def test_runtime_package_is_lightweight_and_independent() -> None:
    before = set(sys.modules)

    importlib.import_module("lidar_detection_runtime")

    loaded = set(sys.modules) - before
    for forbidden_prefix in (
        "lidar_model_selection",
        "torch",
        "mmdet3d",
        "mmcv",
        "mmengine",
        "rclpy",
        "sensor_msgs",
        "visualization_msgs",
    ):
        assert not any(
            name == forbidden_prefix
            or name.startswith(forbidden_prefix + ".")
            for name in loaded
        )
