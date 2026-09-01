from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def test_main_forwards_only_ros_args_and_closes_node_cleanly(monkeypatch) -> None:
    events = []

    class FakeRclpy:
        def init(self, *, args):
            events.append(("init", args))

        def spin(self, node):
            events.append(("spin", node.config.model))

        def ok(self):
            return True

        def shutdown(self):
            events.append(("shutdown",))

    class FakeNode:
        def __init__(self, config):
            self.config = config

        def close(self):
            events.append(("close",))

        def destroy_node(self):
            events.append(("destroy",))

    runtime = SimpleNamespace(rclpy=FakeRclpy())
    monkeypatch.setattr(ros2_node, "_load_ros_runtime", lambda: runtime)
    monkeypatch.setattr(ros2_node, "_node_class", lambda _runtime: FakeNode)

    ros2_node.main([*_args(), "--ros-args", "-p", "use_sim_time:=true"])

    assert events == [
        ("init", ["--ros-args", "-p", "use_sim_time:=true"]),
        ("spin", "voxel0075"),
        ("close",),
        ("destroy",),
        ("shutdown",),
    ]


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
