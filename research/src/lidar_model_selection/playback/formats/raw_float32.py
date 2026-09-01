"""Explicit adapter for legacy packed float32 x/y/z/feature files.

This path intentionally has no Kaposvar calibration or feature normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..frame_source import DirectoryFrameSource, LidarFrame


_MCAP_SUFFIXES = frozenset((".mcap", ".db3"))


@dataclass(frozen=True, slots=True)
class RawFloat32DirectorySource:
    """List legacy packed-frame files without interpreting recording sessions."""

    directory: Path
    extension: str = ".bin"

    def __post_init__(self) -> None:
        validated = DirectoryFrameSource(self.directory, self.extension)
        if validated.extension.lower() in _MCAP_SUFFIXES:
            raise ValueError("raw float32 mode cannot accept a rosbag recording")
        object.__setattr__(self, "extension", validated.extension)

    def list_frames(self, limit: int | None = None) -> tuple[LidarFrame, ...]:
        return DirectoryFrameSource(self.directory, self.extension).list_frames(limit)


def read_raw_float32(path: Path) -> NDArray[np.float32]:
    """Read one legacy packed float32 x/y/z/feature file.

    The fourth value is preserved verbatim.  No Kaposvar feature profile,
    calibration, finite filtering, or zero-row filtering is applied.
    """

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    if path.suffix.lower() in _MCAP_SUFFIXES:
        raise ValueError("raw float32 mode cannot accept a rosbag recording")
    if not path.exists():
        raise FileNotFoundError(f"raw frame does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"raw frame is not a regular file: {path}")
    if path.stat().st_size % 16 != 0:
        raise ValueError("raw float32 frame byte count is not divisible by 16")
    values = np.fromfile(path, dtype="<f4")
    return np.ascontiguousarray(values.reshape((-1, 4)), dtype=np.float32)
