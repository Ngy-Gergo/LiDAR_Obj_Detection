import argparse
from collections.abc import Sequence
from pathlib import Path

from ..evaluation import DEFAULT_RUNS_ROOT
from ..runs import validate_run_id
from .detector import Mmdet3dDetector
from .frame_source import DirectoryFrameSource
from .pipeline import SequentialDetectionPipeline


def run_id_argument(value: str) -> str:
    try:
        return validate_run_id(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sequential LiDAR object detection for one completed run.",
    )
    parser.add_argument(
        "--run",
        dest="run_id",
        required=True,
        type=run_id_argument,
        metavar="RUN_ID",
        help="Canonical completed run ID whose selected checkpoint is used.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing LiDAR frame files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device used for inference.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Minimum confidence score retained in the results.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process.",
    )
    parser.add_argument(
        "--extension",
        type=str,
        default=".bin",
        help="LiDAR frame filename extension.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.max_frames is not None and args.max_frames < 0:
        parser.error("--max-frames must be greater than or equal to zero")

    source = DirectoryFrameSource(
        directory=args.input_dir,
        extension=args.extension,
    )
    frames = source.list_frames(limit=args.max_frames)

    if not frames:
        print("No matching frames found.")
        return 1

    detector = Mmdet3dDetector(
        run=DEFAULT_RUNS_ROOT / args.run_id,
        device=args.device,
        score_threshold=args.score_threshold,
    )
    pipeline = SequentialDetectionPipeline(detector)
    processed_frames = pipeline.run(frames)

    for processed in processed_frames:
        print(
            f"{processed.result.frame_id}: "
            f"detections={len(processed.result.detections)} "
            f"inference_ms={processed.result.inference_ms:.2f} "
            f"total_ms={processed.total_ms:.2f}"
        )

    frame_count = len(processed_frames)
    detection_count = sum(
        len(processed.result.detections)
        for processed in processed_frames
    )
    average_inference_ms = sum(
        processed.result.inference_ms
        for processed in processed_frames
    ) / frame_count
    average_total_ms = sum(
        processed.total_ms
        for processed in processed_frames
    ) / frame_count
    average_fps = (
        f"{1000.0 / average_total_ms:.2f}"
        if average_total_ms > 0
        else "n/a"
    )

    print(
        f"Summary: frames={frame_count} "
        f"detections={detection_count} "
        f"average_inference_ms={average_inference_ms:.2f} "
        f"average_total_ms={average_total_ms:.2f} "
        f"average_fps={average_fps}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
