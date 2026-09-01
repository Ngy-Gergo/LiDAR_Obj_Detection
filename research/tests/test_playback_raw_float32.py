from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lidar_model_selection.playback.formats.raw_float32 import (
    RawFloat32DirectorySource,
    read_raw_float32,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "research" / "src"


def test_raw_adapter_import_does_not_load_mcap_or_yaml_dependencies() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.fspath(SOURCE_ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    code = """
import builtins
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'yaml', 'rosbag2_py', 'rclpy', 'sensor_msgs', 'tf2_msgs'}:
        raise AssertionError(f'unexpected MCAP dependency import: {name}')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from lidar_model_selection.playback.formats.raw_float32 import RawFloat32DirectorySource
assert RawFloat32DirectorySource.__name__ == 'RawFloat32DirectorySource'
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_legacy_raw_adapter_preserves_fourth_feature_and_listing(tmp_path: Path) -> None:
    frames = tmp_path / "raw"
    frames.mkdir()
    expected = np.array(
        [[1.0, 2.0, 3.0, 42.0], [4.0, 5.0, 6.0, -7.0]],
        dtype=np.float32,
    )
    expected.tofile(frames / "000002.bin")
    expected[:1].tofile(frames / "000001.bin")
    (frames / "ignored.txt").write_text("not lidar", encoding="utf-8")

    source = RawFloat32DirectorySource(frames)
    listed = source.list_frames()
    assert [frame.frame_id for frame in listed] == ["000001", "000002"]
    observed = read_raw_float32(listed[1].path)
    np.testing.assert_array_equal(observed, expected)
    assert observed.flags.c_contiguous
    # The raw fourth column is not reflectivity/255 and is not calibrated.
    assert observed[0, 3] == 42.0
    assert observed[0, 0] == 1.0


def test_raw_mode_rejects_rosbag_extensions_and_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot accept a rosbag"):
        RawFloat32DirectorySource(tmp_path, ".mcap")
    mcap = tmp_path / "recording.mcap"
    mcap.write_bytes(b"not raw")
    with pytest.raises(ValueError, match="cannot accept a rosbag"):
        read_raw_float32(mcap)
    malformed = tmp_path / "bad.bin"
    malformed.write_bytes(b"0" * 17)
    with pytest.raises(ValueError, match="not divisible by 16"):
        read_raw_float32(malformed)
