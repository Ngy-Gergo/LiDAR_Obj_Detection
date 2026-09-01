#!/usr/bin/env python3
"""Launch one tracked CenterPoint/Foxglove presentation graph."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Callable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_SOURCE = REPOSITORY_ROOT / "research" / "src"
if str(RESEARCH_SOURCE) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SOURCE))

from lidar_model_selection.playback.model_registry import finalist_aliases  # noqa: E402
from lidar_model_selection.playback.tracking import TrackerConfig  # noqa: E402


POINT_TOPIC = "/lexus3/os_center/points"
BASE_FRAME = "lexus3/base_link"
CAMERA_TOPIC = "/lexus3/camera/zed/zed_node/left/color/rect/image/compressed"
CAMERA_INFO_TOPIC = (
    "/lexus3/camera/zed/zed_node/left/color/rect/image/camera_info"
)
REPLAY_TOPICS = (
    POINT_TOPIC,
    "/tf",
    "/tf_static",
    CAMERA_TOPIC,
    CAMERA_INFO_TOPIC,
)
_CUDA_DEVICE = re.compile(r"cuda:[0-9]+\Z")


@dataclass(frozen=True, slots=True)
class DemoConfig:
    repository_root: Path
    bag: Path
    runs_root: Path
    model: str
    device: str
    rate: float
    loop: bool
    enable_tracking: bool
    start_bridge: bool
    start_bag: bool
    publish_model_cloud: bool
    processing_policy: str
    queue_capacity: int
    score_threshold: float
    bridge_address: str = "127.0.0.1"
    bridge_port: int = 8765
    detector_ready_timeout_seconds: float = 90.0
    track_min_hits: int = 2
    track_max_missed: int = 3
    track_max_gap_seconds: float = 0.75
    track_association_distance: float = 4.0
    track_smoothing: float = 0.65
    track_trail_length: int = 20
    dry_run: bool = False
    checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        for path, name in (
            (self.repository_root, "repository_root"),
            (self.bag, "bag"),
            (self.runs_root, "runs_root"),
        ):
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"{name} must be an absolute path")
        if self.model not in finalist_aliases():
            raise ValueError(f"model must be one of: {', '.join(finalist_aliases())}")
        if not isinstance(self.device, str) or _CUDA_DEVICE.fullmatch(self.device) is None:
            raise ValueError("device must be an explicit CUDA device such as cuda:0")
        if not isfinite(self.rate) or self.rate <= 0.0:
            raise ValueError("rate must be finite and greater than zero")
        if self.processing_policy not in ("all", "latest"):
            raise ValueError("processing_policy must be 'all' or 'latest'")
        if (
            isinstance(self.queue_capacity, bool)
            or not isinstance(self.queue_capacity, int)
            or self.queue_capacity <= 0
        ):
            raise ValueError("queue_capacity must be a positive integer")
        if self.processing_policy == "latest" and self.queue_capacity != 1:
            raise ValueError("latest processing requires queue_capacity=1")
        if not isfinite(self.score_threshold) or not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be finite and in [0, 1]")
        if not isinstance(self.bridge_address, str) or not self.bridge_address.strip():
            raise ValueError("bridge_address must contain non-whitespace text")
        if (
            isinstance(self.bridge_port, bool)
            or not isinstance(self.bridge_port, int)
            or not 1 <= self.bridge_port <= 65535
        ):
            raise ValueError("bridge_port must be in [1, 65535]")
        if (
            not isfinite(self.detector_ready_timeout_seconds)
            or self.detector_ready_timeout_seconds <= 0.0
        ):
            raise ValueError(
                "detector_ready_timeout_seconds must be finite and positive"
            )
        if not isinstance(self.dry_run, bool):
            raise TypeError("dry_run must be a boolean")
        if self.checkpoint_sha256 is not None and (
            not isinstance(self.checkpoint_sha256, str)
            or len(self.checkpoint_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.checkpoint_sha256
            )
        ):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
        TrackerConfig(
            min_confirmed_hits=self.track_min_hits,
            max_missed_frames=self.track_max_missed,
            max_time_gap_seconds=self.track_max_gap_seconds,
            association_distance_meters=self.track_association_distance,
            position_smoothing=self.track_smoothing,
            score_smoothing=self.track_smoothing,
            trail_length=self.track_trail_length,
        )

    @property
    def output_prefix(self) -> str:
        return f"/centerpoint/{self.model}"

    @property
    def qos_path(self) -> Path:
        return self.repository_root / "research/configs/playback/rosbag2_qos.yaml"


@dataclass(frozen=True, slots=True)
class DemoCommands:
    bridge: tuple[str, ...]
    detector: tuple[str, ...]
    bag: tuple[str, ...]


def build_commands(config: DemoConfig) -> DemoCommands:
    detector = [
        sys.executable,
        "-m",
        "lidar_model_selection.playback.ros2_node",
        "--model",
        config.model,
        "--runs-root",
        str(config.runs_root),
        "--device",
        config.device,
        "--input-topic",
        POINT_TOPIC,
        "--output-prefix",
        config.output_prefix,
        "--base-frame",
        BASE_FRAME,
        "--feature-profile",
        "kaposvar_center_reflectivity_v1",
        "--score-threshold",
        str(config.score_threshold),
        "--processing-policy",
        config.processing_policy,
        "--queue-capacity",
        str(config.queue_capacity),
        "--tf-timeout-seconds",
        "0.2",
        "--diagnostics-period-seconds",
        "1.0",
        (
            "--publish-model-cloud"
            if config.publish_model_cloud
            else "--no-publish-model-cloud"
        ),
    ]
    if config.enable_tracking:
        detector.extend(
            (
                "--enable-tracking",
                "--track-min-hits",
                str(config.track_min_hits),
                "--track-max-missed",
                str(config.track_max_missed),
                "--track-max-gap-seconds",
                str(config.track_max_gap_seconds),
                "--track-association-distance",
                str(config.track_association_distance),
                "--track-smoothing",
                str(config.track_smoothing),
                "--track-trail-length",
                str(config.track_trail_length),
            )
        )
    if config.checkpoint_sha256 is not None:
        detector.extend(("--checkpoint-sha256", config.checkpoint_sha256))
    detector.extend(("--ros-args", "-p", "use_sim_time:=true"))

    bag = [
        "ros2",
        "bag",
        "play",
        str(config.bag),
        "--storage",
        "mcap",
        "--rate",
        str(config.rate),
    ]
    if config.loop:
        bag.append("--loop")
    bag.extend(
        (
            "--clock",
            "100",
            "--qos-profile-overrides-path",
            str(config.qos_path),
            "--topics",
            *REPLAY_TOPICS,
        )
    )
    bridge = (
        "ros2",
        "launch",
        "foxglove_bridge",
        "foxglove_bridge_launch.xml",
        f"address:={config.bridge_address}",
        f"port:={config.bridge_port}",
    )
    return DemoCommands(
        bridge=bridge,
        detector=tuple(detector),
        bag=tuple(bag),
    )


def validate_environment(config: DemoConfig) -> None:
    if not config.repository_root.is_dir():
        raise FileNotFoundError(
            f"repository root does not exist: {config.repository_root}"
        )
    if config.start_bag:
        if not config.bag.is_dir():
            raise FileNotFoundError(f"bag directory does not exist: {config.bag}")
        if not (config.bag / "metadata.yaml").is_file():
            raise FileNotFoundError(
                f"bag metadata does not exist: {config.bag / 'metadata.yaml'}"
            )
    if not config.runs_root.is_dir():
        raise FileNotFoundError(f"runs root does not exist: {config.runs_root}")
    if not config.qos_path.is_file():
        raise FileNotFoundError(f"replay QoS file does not exist: {config.qos_path}")
    if shutil.which("ros2") is None:
        raise RuntimeError(
            "ros2 is unavailable; source /opt/ros/humble/setup.bash first"
        )
    if os.environ.get("ROS_DISTRO") != "humble":
        raise RuntimeError(
            "ROS2 Humble is not active; source /opt/ros/humble/setup.bash first"
        )
    if config.start_bridge:
        bridge = subprocess.run(
            ("ros2", "pkg", "prefix", "foxglove_bridge"),
            check=False,
            capture_output=True,
            text=True,
        )
        if bridge.returncode != 0:
            raise RuntimeError(
                "foxglove_bridge is unavailable in the active ROS2 environment"
            )
    if not Path(sys.executable).is_file():
        raise RuntimeError(f"Python executable does not exist: {sys.executable}")


def _child_environment(config: DemoConfig) -> dict[str, str]:
    environment = dict(os.environ)
    source_paths = [
        str(config.repository_root / "research" / "src"),
        str(config.repository_root / "runtime"),
    ]
    existing = environment.get("PYTHONPATH")
    if existing:
        source_paths.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(source_paths)
    return environment


def _wait_for_detector_topic(
    config: DemoConfig,
    detector: subprocess.Popen[bytes],
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    expected = f"{config.output_prefix}/diagnostics"
    deadline = monotonic() + config.detector_ready_timeout_seconds
    while monotonic() < deadline:
        exit_code = detector.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"detector exited before becoming ready (exit {exit_code})"
            )
        result = run(
            ("ros2", "topic", "list"),
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and expected in result.stdout.splitlines():
            return
        sleep(0.25)
    raise TimeoutError(
        f"detector topic {expected} was not ready within "
        f"{config.detector_ready_timeout_seconds:.1f} seconds"
    )


def _stop_process(
    process: subprocess.Popen[bytes] | None,
    *,
    name: str,
    kill_group: Callable[[int, int], None] = os.killpg,
) -> None:
    if process is None or process.poll() is not None:
        return
    for signal_number, timeout in (
        (signal.SIGINT, 8.0),
        (signal.SIGTERM, 4.0),
        (signal.SIGKILL, 2.0),
    ):
        try:
            kill_group(process.pid, signal_number)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue
    raise RuntimeError(f"{name} did not stop after SIGKILL")


def run_demo(
    config: DemoConfig,
    *,
    popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
) -> int:
    commands = build_commands(config)
    if config.dry_run:
        for name, command in (
            ("bridge", commands.bridge),
            ("detector", commands.detector),
            ("bag", commands.bag),
        ):
            enabled = (
                name == "detector"
                or (name == "bridge" and config.start_bridge)
                or (name == "bag" and config.start_bag)
            )
            if enabled:
                print(f"{name}: {shlex.join(command)}")
        return 0
    validate_environment(config)
    environment = _child_environment(config)
    children: dict[str, subprocess.Popen[bytes] | None] = {
        "bridge": None,
        "detector": None,
        "bag": None,
    }
    print(f"Foxglove: ws://localhost:{config.bridge_port}")
    print(f"Fixed frame: {BASE_FRAME}")
    print(f"Tracked markers: {config.output_prefix}/tracked_markers")
    try:
        if config.start_bridge:
            children["bridge"] = popen(
                commands.bridge,
                cwd=config.repository_root,
                env=environment,
                start_new_session=True,
            )
        children["detector"] = popen(
            commands.detector,
            cwd=config.repository_root,
            env=environment,
            start_new_session=True,
        )
        _wait_for_detector_topic(config, children["detector"])
        if config.start_bag:
            children["bag"] = popen(
                commands.bag,
                cwd=config.repository_root,
                env=environment,
                start_new_session=True,
            )

        while True:
            for name, child in children.items():
                if child is None:
                    continue
                exit_code = child.poll()
                if exit_code is not None:
                    if name == "bag" and not config.loop and exit_code == 0:
                        print("bag playback completed")
                        return 0
                    raise RuntimeError(
                        f"{name} exited unexpectedly with status {exit_code}"
                    )
            time.sleep(0.25)
    except KeyboardInterrupt:
        return 0
    finally:
        stop_errors: list[str] = []
        # Stop new input first, then model output, then the observer bridge.
        for name in ("bag", "detector", "bridge"):
            try:
                _stop_process(children[name], name=name)
            except Exception as error:
                stop_errors.append(f"{name}: {str(error) or repr(error)}")
        if stop_errors:
            raise RuntimeError("child shutdown failure: " + "; ".join(stop_errors))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a tracked CenterPoint ROS2/Foxglove demonstration.",
    )
    parser.add_argument("--model", required=True, choices=finalist_aliases())
    parser.add_argument("--device", required=True)
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPOSITORY_ROOT / "research" / "runs",
    )
    parser.add_argument("--rate", type=float, default=0.5)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--enable-tracking", action="store_true")
    parser.add_argument(
        "--no-bridge",
        "--no-start-bridge",
        action="store_true",
        dest="no_start_bridge",
    )
    parser.add_argument(
        "--no-bag",
        "--no-start-bag",
        action="store_true",
        dest="no_start_bag",
    )
    parser.add_argument("--no-publish-model-cloud", action="store_true")
    parser.add_argument(
        "--processing-policy",
        choices=("all", "latest"),
        default="all",
    )
    parser.add_argument("--queue-capacity", type=int, default=32)
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument(
        "--checkpoint-sha256",
        help="Pin the protected registry checkpoint by exact SHA-256.",
    )
    parser.add_argument("--bridge-address", default="127.0.0.1")
    parser.add_argument("--bridge-port", type=int, default=8765)
    parser.add_argument("--detector-ready-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--track-min-hits", type=int, default=2)
    parser.add_argument("--track-max-missed", type=int, default=3)
    parser.add_argument("--track-max-gap-seconds", type=float, default=0.75)
    parser.add_argument("--track-association-distance", type=float, default=4.0)
    parser.add_argument("--track-smoothing", type=float, default=0.65)
    parser.add_argument("--track-trail-length", type=int, default=20)
    parser.add_argument(
        "--dry-run",
        "--print-command",
        action="store_true",
        dest="dry_run",
        help="Print shell-escaped child commands without validation or execution.",
    )
    return parser


def parse_args(args: Sequence[str] | None = None) -> DemoConfig:
    parser = build_parser()
    values = parser.parse_args(args)
    try:
        return DemoConfig(
            repository_root=REPOSITORY_ROOT,
            bag=values.bag,
            runs_root=values.runs_root,
            model=values.model,
            device=values.device,
            rate=values.rate,
            loop=values.loop,
            enable_tracking=values.enable_tracking,
            start_bridge=not values.no_start_bridge,
            start_bag=not values.no_start_bag,
            publish_model_cloud=not values.no_publish_model_cloud,
            processing_policy=values.processing_policy,
            queue_capacity=values.queue_capacity,
            score_threshold=values.score_threshold,
            bridge_address=values.bridge_address,
            bridge_port=values.bridge_port,
            detector_ready_timeout_seconds=values.detector_ready_timeout_seconds,
            track_min_hits=values.track_min_hits,
            track_max_missed=values.track_max_missed,
            track_max_gap_seconds=values.track_max_gap_seconds,
            track_association_distance=values.track_association_distance,
            track_smoothing=values.track_smoothing,
            track_trail_length=values.track_trail_length,
            dry_run=values.dry_run,
            checkpoint_sha256=values.checkpoint_sha256,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))


def main(args: Sequence[str] | None = None) -> int:
    try:
        return run_demo(parse_args(args))
    except (FileNotFoundError, RuntimeError, TimeoutError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
