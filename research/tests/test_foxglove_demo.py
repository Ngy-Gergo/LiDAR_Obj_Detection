from __future__ import annotations

import importlib.util
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


TOOL_PATH = Path(__file__).parents[1] / "tools" / "foxglove_demo.py"
SPEC = importlib.util.spec_from_file_location("foxglove_demo_tool", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
foxglove_demo = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = foxglove_demo
SPEC.loader.exec_module(foxglove_demo)


def _config(tmp_path: Path, **changes):
    values = {
        "repository_root": tmp_path,
        "bag": tmp_path / "bag",
        "runs_root": tmp_path / "runs",
        "model": "voxel0075",
        "device": "cuda:0",
        "rate": 0.5,
        "loop": True,
        "enable_tracking": True,
        "start_bridge": True,
        "start_bag": True,
        "publish_model_cloud": True,
        "processing_policy": "all",
        "queue_capacity": 32,
        "score_threshold": 0.1,
    }
    values.update(changes)
    return foxglove_demo.DemoConfig(**values)


def test_command_construction_reuses_qos_topics_and_configurable_run_root(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    commands = foxglove_demo.build_commands(config)

    assert commands.bridge[:4] == (
        "ros2",
        "launch",
        "foxglove_bridge",
        "foxglove_bridge_launch.xml",
    )
    assert commands.detector[:3] == (
        foxglove_demo.sys.executable,
        "-m",
        "lidar_model_selection.playback.ros2_node",
    )
    assert commands.detector[commands.detector.index("--runs-root") + 1] == str(
        config.runs_root
    )
    assert "--enable-tracking" in commands.detector
    assert commands.detector[commands.detector.index("--track-smoothing") + 1] == "0.65"
    assert "--publish-model-cloud" in commands.detector
    assert "--loop" in commands.bag
    assert commands.bag[commands.bag.index("--qos-profile-overrides-path") + 1] == str(
        config.qos_path
    )
    topic_start = commands.bag.index("--topics") + 1
    assert commands.bag[topic_start:] == foxglove_demo.REPLAY_TOPICS


def test_command_can_pin_protected_checkpoint_identity(tmp_path: Path) -> None:
    digest = "a" * 64
    commands = foxglove_demo.build_commands(
        _config(tmp_path, checkpoint_sha256=digest)
    )
    assert commands.detector[
        commands.detector.index("--checkpoint-sha256") + 1
    ] == digest


def test_launcher_parser_supports_existing_bridge_and_bag(tmp_path: Path) -> None:
    config = foxglove_demo.parse_args(
        [
            "--model",
            "pillar02_multiclass",
            "--device",
            "cuda:1",
            "--bag",
            str(tmp_path / "bag"),
            "--runs-root",
            str(tmp_path / "runs"),
            "--rate",
            "1.0",
            "--enable-tracking",
            "--no-start-bridge",
            "--no-start-bag",
            "--processing-policy",
            "latest",
            "--queue-capacity",
            "1",
        ]
    )
    assert config.model == "pillar02_multiclass"
    assert config.device == "cuda:1"
    assert config.enable_tracking
    assert not config.start_bridge
    assert not config.start_bag
    assert config.processing_policy == "latest"


@pytest.mark.parametrize(
    "arguments",
    [
        ("device", "cpu"),
        ("rate", 0.0),
        ("queue_capacity", 0),
        ("bridge_port", 70000),
        ("checkpoint_sha256", "not-a-digest"),
    ],
)
def test_invalid_launcher_configuration_is_rejected(tmp_path: Path, arguments) -> None:
    with pytest.raises(ValueError):
        _config(tmp_path, **{arguments[0]: arguments[1]})


def test_launcher_starts_in_order_and_stops_detector_bag_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    starts = []
    stops = []

    class Process:
        next_pid = 100

        def __init__(self, command):
            self.command = tuple(command)
            self.pid = Process.next_pid
            Process.next_pid += 1

        def poll(self):
            return None

    def popen(command, **kwargs):
        process = Process(command)
        starts.append(process.command)
        assert kwargs["start_new_session"] is True
        return process

    monkeypatch.setattr(foxglove_demo, "validate_environment", lambda _config: None)
    monkeypatch.setattr(
        foxglove_demo,
        "_wait_for_detector_topic",
        lambda _config, _detector: None,
    )
    monkeypatch.setattr(
        foxglove_demo,
        "_stop_process",
        lambda process, *, name: stops.append((name, process is not None)),
    )
    monkeypatch.setattr(
        foxglove_demo.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    assert foxglove_demo.run_demo(config, popen=popen) == 0
    commands = foxglove_demo.build_commands(config)
    assert starts == [commands.bridge, commands.detector, commands.bag]
    assert stops == [
        ("bag", True),
        ("detector", True),
        ("bridge", True),
    ]


def test_process_shutdown_begins_with_sigint() -> None:
    events = []

    class Process:
        pid = 42

        def poll(self):
            return None

        def wait(self, *, timeout):
            events.append(("wait", timeout))
            return 0

    foxglove_demo._stop_process(
        Process(),
        name="detector",
        kill_group=lambda pid, signum: events.append(("signal", pid, signum)),
    )
    assert events == [("signal", 42, signal.SIGINT), ("wait", 8.0)]


def test_detector_readiness_wait_is_bounded_and_uses_diagnostic_topic(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, detector_ready_timeout_seconds=1.0)
    ticks = iter((0.0, 0.1, 0.2))
    calls = []
    detector = SimpleNamespace(poll=lambda: None)

    foxglove_demo._wait_for_detector_topic(
        config,
        detector,
        run=lambda *args, **kwargs: (
            calls.append(args[0])
            or SimpleNamespace(
                returncode=0,
                stdout=f"{config.output_prefix}/diagnostics\n",
            )
        ),
        monotonic=lambda: next(ticks),
        sleep=lambda _seconds: None,
    )
    assert calls == [("ros2", "topic", "list")]


def test_dry_run_prints_shell_escaped_commands_without_environment_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path, dry_run=True, start_bridge=False)
    monkeypatch.setattr(
        foxglove_demo,
        "validate_environment",
        lambda _config: (_ for _ in ()).throw(AssertionError("must not validate")),
    )

    assert foxglove_demo.run_demo(config) == 0
    output = capsys.readouterr().out
    assert "bridge:" not in output
    assert "detector:" in output
    assert "bag:" in output
    assert "--enable-tracking" in output
