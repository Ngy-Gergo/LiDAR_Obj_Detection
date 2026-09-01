from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy
import pytest

import lidar_model_selection.playback.cli as playback_cli
import lidar_model_selection.playback.formats.ros2_mcap as ros2_mcap
import lidar_model_selection.playback.model_registry as model_registry
from lidar_model_selection.playback.contracts import PointCloudFrame
from lidar_model_selection.playback.contracts import (
    FrameErrorEvidence,
    FrameSourceError,
)
from lidar_model_selection.playback.results import DetectionFrame


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "research" / "src"


def test_parser_exposes_explicit_mutually_exclusive_mcap_mode() -> None:
    parser = playback_cli.build_parser()
    arguments = parser.parse_args(
        [
            "--recording-root",
            "/recordings",
            "--session",
            "proba_lexus3_2026-07-27_14-09",
            "--model",
            "voxel0075",
            "--validate-only",
            "--max-frames",
            "3",
        ]
    )

    assert arguments.recording_root == Path("/recordings")
    assert arguments.session == "proba_lexus3_2026-07-27_14-09"
    assert arguments.model == "voxel0075"
    assert arguments.validate_only
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--recording-root",
                "/recordings",
                "--input-dir",
                "/raw",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--recording-root",
                "/recordings",
                "--session",
                "session",
                "--model",
                "unregistered",
            ]
        )


def test_cli_help_imports_no_ml_ros_or_viewer_frameworks() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.fspath(SOURCE_ROOT)
    code = """
import sys
from lidar_model_selection.playback.cli import main
try:
    main(['--help'])
except SystemExit as error:
    assert error.code == 0
for prefix in (
    'torch', 'mmdet3d', 'mmcv', 'mmengine', 'matplotlib',
    'rosbag2_py', 'rclpy', 'sensor_msgs', 'tf2_msgs',
):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_validate_only_decodes_without_run_checkpoint_or_model_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recording_root = tmp_path / "recordings"
    session = recording_root / "proba_lexus3_2026-07-27_14-09"
    session.mkdir(parents=True)
    initial_entries = tuple(session.iterdir())

    class FakeSource:
        def __init__(self, session_directory: Path, *, feature_profile: str) -> None:
            self.session_id = session_directory.name
            self.frame_count = 1
            self.mcap_path = session_directory / "recording_0.mcap"
            self.feature_profile = feature_profile
            self.schema = ros2_mcap.PointCloudSchema()
            self.calibration = SimpleNamespace(
                parent_frame_id="lexus3/base_link",
                child_frame_id="lexus3/os_center",
                translation_xyz=(0.75, 0.0, 1.91),
                quaternion_xyzw=(0.0, 0.0, -1.0, 0.0),
            )

        def iter_frames(self, start_index: int = 0):
            assert start_index == 0
            yield PointCloudFrame(
                session_id=self.session_id,
                frame_index=0,
                timestamp_ns=1_000_000_000,
                storage_timestamp_ns=1_100_000_000,
                source_frame_id="lexus3/os_center",
                coordinate_frame="lidar",
                source_key="recording_0.mcap:/points[0]",
                points=numpy.asarray(
                    [[1.0, -2.0, 0.0, 128.0 / 255.0]],
                    dtype=numpy.float32,
                ),
                source_point_count=1,
                dropped_nonfinite_count=0,
                decode_ms=1.5,
            )

    monkeypatch.setattr(ros2_mcap, "Ros2McapRecordingSequence", FakeSource)
    monkeypatch.setattr(
        model_registry,
        "load_run",
        lambda path: pytest.fail(f"unexpected run/checkpoint access: {path}"),
    )
    monkeypatch.setattr(
        playback_cli,
        "FinalistDetector",
        lambda *args, **kwargs: pytest.fail("validate-only loaded a model"),
    )

    exit_code = playback_cli.main(
        [
            "--recording-root",
            os.fspath(recording_root.resolve()),
            "--session",
            session.name,
            "--model",
            "voxel0075",
            "--validate-only",
            "--max-frames",
            "1",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert f"Session: id={session.name}" in output
    assert "type=sensor_msgs/msg/PointCloud2" in output
    assert "profile=kaposvar_center_reflectivity_v1" in output
    assert "translation_xyz=(0.75,0,1.91)" in output
    assert "model_transform=rotation_only" in output
    assert "run_id=20260827T092043Z-voxel0075" in output
    assert "selected_checkpoint_sha256=5246b24b" in output
    assert "normalized_points=1" in output
    assert "feature_range=(0.501961,0.501961)" in output
    assert "Validation summary:" in output
    assert tuple(session.iterdir()) == initial_entries


def test_validate_only_is_heavy_import_free_in_a_fresh_process(
    tmp_path: Path,
) -> None:
    recording_root = tmp_path / "recordings"
    session = recording_root / "proba_lexus3_2026-07-27_14-09"
    session.mkdir(parents=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.fspath(SOURCE_ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    code = f"""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy
from lidar_model_selection.playback import cli
from lidar_model_selection.playback.contracts import PointCloudFrame
from lidar_model_selection.playback.formats import ros2_mcap

class FakeSource:
    def __init__(self, session_directory, *, feature_profile):
        self.session_id = session_directory.name
        self.frame_count = 1
        self.mcap_path = session_directory / 'recording_0.mcap'
        self.feature_profile = feature_profile
        self.schema = ros2_mcap.PointCloudSchema()
        self.calibration = SimpleNamespace(
            parent_frame_id='lexus3/base_link',
            child_frame_id='lexus3/os_center',
            translation_xyz=(0.75, 0.0, 1.91),
            quaternion_xyzw=(0.0, 0.0, -1.0, 0.0),
        )
    def iter_frames(self, start_index=0):
        assert start_index == 0
        yield PointCloudFrame(
            session_id=self.session_id,
            frame_index=0,
            timestamp_ns=1,
            storage_timestamp_ns=2,
            source_frame_id='lexus3/os_center',
            coordinate_frame='lidar',
            source_key='recording_0.mcap:/points[0]',
            points=numpy.empty((0, 4), dtype=numpy.float32),
            source_point_count=0,
            dropped_nonfinite_count=0,
        )

ros2_mcap.Ros2McapRecordingSequence = FakeSource
cli.FinalistDetector = lambda *args, **kwargs: (_ for _ in ()).throw(
    AssertionError('validate-only constructed a detector')
)
status = cli.main([
    '--recording-root', {os.fspath(recording_root.resolve())!r},
    '--session', {session.name!r},
    '--model', 'voxel0075',
    '--validate-only',
    '--max-frames', '1',
])
assert status == 0
for prefix in ('torch', 'mmdet3d', 'mmcv', 'mmengine', 'matplotlib'):
    assert not any(
        name == prefix or name.startswith(prefix + '.') for name in sys.modules
    ), prefix
"""

    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_validate_only_reports_complete_recoverable_error_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recording_root = tmp_path / "recordings"
    session = recording_root / "error-session"
    session.mkdir(parents=True)

    class ErrorIterator:
        def __init__(self) -> None:
            self.done = False

        def __iter__(self):
            return self

        def __next__(self):
            if self.done:
                raise StopIteration
            self.done = True
            raise FrameSourceError(
                FrameErrorEvidence(
                    code="truncated_pointcloud_payload",
                    message="declared row ended early",
                    session_id=session.name,
                    frame_index=7,
                    header_timestamp_ns=123,
                    storage_timestamp_ns=234,
                    source_key="recording_0.mcap:/points[7]",
                    recoverable=True,
                    decode_ms=6.25,
                )
            )

    class FakeSource:
        def __init__(self, session_directory: Path, *, feature_profile: str) -> None:
            self.session_id = session_directory.name
            self.frame_count = 8
            self.mcap_path = session_directory / "recording_0.mcap"
            self.feature_profile = feature_profile
            self.schema = ros2_mcap.PointCloudSchema()
            self.calibration = SimpleNamespace(
                parent_frame_id="lexus3/base_link",
                child_frame_id="lexus3/os_center",
                translation_xyz=(0.75, 0.0, 1.91),
                quaternion_xyzw=(0.0, 0.0, -1.0, 0.0),
            )

        def iter_frames(self, start_index: int = 0):
            assert start_index == 0
            return ErrorIterator()

    monkeypatch.setattr(ros2_mcap, "Ros2McapRecordingSequence", FakeSource)

    exit_code = playback_cli.main(
        [
            "--recording-root",
            os.fspath(recording_root.resolve()),
            "--session",
            session.name,
            "--model",
            "voxel0075",
            "--validate-only",
            "--max-frames",
            "1",
            "--on-frame-error",
            "continue",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 2
    assert "frame=7" in output
    assert "header_timestamp_ns=123" in output
    assert "storage_timestamp_ns=234" in output
    assert "source_key=recording_0.mcap:/points[7]" in output
    assert "decode_ms=6.250" in output


def test_validate_only_continue_does_not_count_prefix_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    frame = PointCloudFrame(
        session_id="prefix-session",
        frame_index=2,
        timestamp_ns=300,
        storage_timestamp_ns=400,
        source_frame_id="lexus3/os_center",
        coordinate_frame="lidar",
        source_key="recording_0.mcap:/points[2]",
        points=numpy.asarray([[1.0, 0.0, 0.0, 0.5]], dtype=numpy.float32),
        source_point_count=1,
        dropped_nonfinite_count=0,
        decode_ms=1.0,
    )

    class ErrorThenFrame:
        def __init__(self) -> None:
            self.state = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.state == 0:
                self.state = 1
                raise FrameSourceError(
                    FrameErrorEvidence(
                        code="invalid_pointcloud_schema",
                        message="bad skipped frame",
                        session_id="prefix-session",
                        frame_index=0,
                        header_timestamp_ns=100,
                        storage_timestamp_ns=200,
                        source_key="recording_0.mcap:/points[0]",
                        recoverable=True,
                        decode_ms=2.0,
                    )
                )
            if self.state == 1:
                self.state = 2
                return frame
            raise StopIteration

    source = SimpleNamespace(
        session_id="prefix-session",
        iter_frames=lambda start_index=0: ErrorThenFrame(),
    )
    args = SimpleNamespace(
        start_frame=2,
        max_frames=1,
        on_frame_error="continue",
    )

    exit_code = playback_cli._run_validate_only(source, args)

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Frame error:" not in output
    assert "index=2" in output
    assert "frames=1 errors=0" in output


def test_mcap_cli_streams_two_frames_through_one_detector_and_reports_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recording_root = tmp_path / "recordings"
    session = recording_root / "two-frame-session"
    session.mkdir(parents=True)
    constructed: list[object] = []

    class FakeSource:
        def __init__(self, session_directory: Path, *, feature_profile: str) -> None:
            self.session_id = session_directory.name
            self.frame_count = 2
            self.mcap_path = session_directory / "recording_0.mcap"
            self.feature_profile = feature_profile
            self.schema = ros2_mcap.PointCloudSchema()
            self.calibration = SimpleNamespace(
                parent_frame_id="lexus3/base_link",
                child_frame_id="lexus3/os_center",
                translation_xyz=(0.75, 0.0, 1.91),
                quaternion_xyzw=(0.0, 0.0, -1.0, 0.0),
            )

        def iter_frames(self, start_index: int = 0):
            for index in range(start_index, 2):
                yield PointCloudFrame(
                    session_id=self.session_id,
                    frame_index=index,
                    timestamp_ns=1_000_000_000 + index,
                    storage_timestamp_ns=1_100_000_000 + index,
                    source_frame_id="lexus3/os_center",
                    coordinate_frame="lidar",
                    source_key=f"recording_0.mcap:/points[{index}]",
                    points=numpy.asarray(
                        [[1.0, 0.0, 0.0, 0.5]], dtype=numpy.float32
                    ),
                    source_point_count=1,
                    dropped_nonfinite_count=0,
                    decode_ms=1.0,
                )

    class FakeDetector:
        def __init__(
            self,
            model_alias: str,
            runs_root: Path,
            *,
            device: str,
            score_threshold: float,
        ) -> None:
            self.identity = SimpleNamespace(
                model_alias=model_alias,
                run_id=model_registry.FINALIST_RUNS[model_alias],
                config_sha256="a" * 64,
                checkpoint_reference="training/selected.pth",
                checkpoint_sha256="b" * 64,
                checkpoint_size_bytes=41_175_026,
            )
            self.config_sha256 = self.identity.config_sha256
            self.checkpoint_path = runs_root / self.identity.run_id / "training/selected.pth"
            self.checkpoint_sha256 = self.identity.checkpoint_sha256
            self.device = device
            self.calls: list[int] = []
            constructed.append(self)

        def detect(self, frame: PointCloudFrame) -> DetectionFrame:
            self.calls.append(frame.frame_index)
            boxes = numpy.asarray(
                [[1.0, 0.0, -1.0, 4.0, 2.0, 1.5, 0.0]],
                dtype=numpy.float32,
            )
            scores = numpy.asarray([0.8], dtype=numpy.float32)
            labels = numpy.asarray([0], dtype=numpy.int64)
            for values in (boxes, scores, labels):
                values.setflags(write=False)
            return DetectionFrame(
                session_id=frame.session_id,
                frame_index=frame.frame_index,
                timestamp_ns=frame.timestamp_ns,
                storage_timestamp_ns=frame.storage_timestamp_ns,
                source_frame_id=frame.source_frame_id,
                coordinate_frame=frame.coordinate_frame,
                source_key=frame.source_key,
                model_alias=self.identity.model_alias,
                run_id=self.identity.run_id,
                config_sha256=self.identity.config_sha256,
                checkpoint_path=self.identity.checkpoint_reference,
                checkpoint_sha256=self.identity.checkpoint_sha256,
                checkpoint_size_bytes=self.identity.checkpoint_size_bytes,
                source_point_count=1,
                dropped_nonfinite_count=0,
                input_point_count=1,
                in_range_point_count=1,
                detection_count=1,
                status="success",
                boxes=boxes,
                scores=scores,
                labels=labels,
                decode_ms=frame.decode_ms,
                detector_ms=2.0,
                frame_processing_ms=3.0,
            )

    monkeypatch.setattr(ros2_mcap, "Ros2McapRecordingSequence", FakeSource)
    monkeypatch.setattr(playback_cli, "FinalistDetector", FakeDetector)

    exit_code = playback_cli.main(
        [
            "--recording-root",
            os.fspath(recording_root.resolve()),
            "--session",
            session.name,
            "--model",
            "voxel0075",
            "--max-frames",
            "2",
            "--playback-rate",
            "1000000000",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert len(constructed) == 1
    assert constructed[0].calls == [0, 1]
    assert output.count("Detection: session=two-frame-session") == 2
    assert "coordinate_frame=lidar" in output
    assert "dropped_nonfinite=0" in output
    assert "Playback summary: session=two-frame-session frames=2 successes=2" in output


def test_legacy_mode_rejects_mcap_extension_without_detector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        playback_cli,
        "Mmdet3dDetector",
        lambda *args, **kwargs: pytest.fail("raw source validation must run first"),
    )
    exit_code = playback_cli.main(
        [
            "--input-dir",
            os.fspath(tmp_path),
            "--run",
            "20260818T000000Z-pillar02-" + "a" * 24,
            "--extension",
            ".mcap",
        ]
    )
    assert exit_code == 2
