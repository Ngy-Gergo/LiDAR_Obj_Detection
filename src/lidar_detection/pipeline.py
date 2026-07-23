from dataclasses import dataclass
from math import isfinite
from numbers import Real
from time import perf_counter

from .detector import Mmdet3dDetector
from .frame_source import LidarFrame
from .results import FrameResult


@dataclass(frozen=True, slots=True)
class ProcessedFrame:
    result: FrameResult
    total_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.result, FrameResult):
            raise TypeError("result must be a FrameResult")
        if isinstance(self.total_ms, bool) or not isinstance(self.total_ms, Real):
            raise TypeError("total_ms must be a real number and not a boolean")
        if not isfinite(self.total_ms):
            raise ValueError("total_ms must be finite")
        if self.total_ms < 0:
            raise ValueError("total_ms must be greater than or equal to zero")


class SequentialDetectionPipeline:
    def __init__(self, detector: Mmdet3dDetector) -> None:
        if not isinstance(detector, Mmdet3dDetector):
            raise TypeError("detector must be an Mmdet3dDetector")
        self._detector = detector

    def run(
        self,
        frames: tuple[LidarFrame, ...],
    ) -> tuple[ProcessedFrame, ...]:
        if not isinstance(frames, tuple):
            raise TypeError("frames must be a tuple")
        if any(not isinstance(frame, LidarFrame) for frame in frames):
            raise TypeError("every frames member must be a LidarFrame")
        if not frames:
            return ()

        processed_frames = []

        for frame in frames:
            start_time = perf_counter()
            result = self._detector.detect(frame)
            end_time = perf_counter()

            processed_frames.append(
                ProcessedFrame(
                    result=result,
                    total_ms=(end_time - start_time) * 1000.0,
                )
            )

        return tuple(processed_frames)
