import importlib
import importlib.util
import sys


def test_research_packages_are_lightweight() -> None:
    before = set(sys.modules)

    importlib.import_module("lidar_model_selection")
    importlib.import_module("lidar_model_selection.compat")
    importlib.import_module("lidar_model_selection.playback")
    importlib.import_module("lidar_model_selection.pipeline")
    importlib.import_module("lidar_model_selection.plotting")
    importlib.import_module("lidar_model_selection.preflight")

    loaded = set(sys.modules) - before
    for forbidden_prefix in (
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


def test_opt_in_modules_are_discoverable() -> None:
    assert importlib.util.find_spec(
        "lidar_model_selection.compat.center_head_7d"
    )
    assert importlib.util.find_spec(
        "lidar_model_selection.compat.kitti_evaluator"
    )
    assert importlib.util.find_spec(
        "lidar_model_selection.playback.cli"
    )
