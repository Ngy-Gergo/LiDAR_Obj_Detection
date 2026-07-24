from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected 3D object with a box ordered as (x, y, z, dx, dy, dz, yaw)."""

    box: tuple[float, float, float, float, float, float, float]
    score: float
    label: int

    def __post_init__(self) -> None:
        if not isinstance(self.box, tuple):
            raise TypeError("box must be a tuple")
        if len(self.box) != 7:
            raise ValueError("box must contain exactly seven values")
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in self.box):
            raise TypeError("every box value must be a real number and not a boolean")

        if isinstance(self.score, bool) or not isinstance(self.score, Real):
            raise TypeError("score must be a real number and not a boolean")
        if not isfinite(self.score):
            raise ValueError("score must be finite")

        if isinstance(self.label, bool) or not isinstance(self.label, Integral):
            raise TypeError("label must be an integer and not a boolean")


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Detections and inference metadata for one LiDAR frame."""

    frame_id: str
    source_path: Path
    detections: tuple[Detection, ...]
    inference_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str):
            raise TypeError("frame_id must be a string")
        if not self.frame_id.strip():
            raise ValueError("frame_id must contain at least one non-whitespace character")

        if not isinstance(self.source_path, Path):
            raise TypeError("source_path must be a pathlib.Path")

        if not isinstance(self.detections, tuple):
            raise TypeError("detections must be a tuple")
        if any(not isinstance(detection, Detection) for detection in self.detections):
            raise TypeError("every detections member must be a Detection")

        if isinstance(self.inference_ms, bool) or not isinstance(self.inference_ms, Real):
            raise TypeError("inference_ms must be a real number and not a boolean")
        if not isfinite(self.inference_ms):
            raise ValueError("inference_ms must be finite")
        if self.inference_ms < 0:
            raise ValueError("inference_ms must be greater than or equal to zero")
