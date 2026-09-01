"""Command-line entry point for explicit raw and ROS2 MCAP playback modes."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from math import isfinite
from pathlib import Path

import numpy

from ..evaluation import DEFAULT_RUNS_ROOT
from ..runs import validate_run_id
from .contracts import FrameErrorEvidence, FrameSourceError
from .detector import FinalistDetector, Mmdet3dDetector
from .frame_source import resolve_session_directory
from .model_registry import (
    FINALIST_POINT_CLOUD_RANGE,
    finalist_aliases,
    finalist_spec,
)
from .normalization import KAPOSVAR_FEATURE_PROFILE
from .pipeline import SequentialDetectionPipeline, SessionProcessor


def run_id_argument(value: str) -> str:
    try:
        return validate_run_id(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run session-aware ROS2 MCAP detection or the explicit legacy "
            "raw-float32 playback path."
        ),
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--recording-root",
        type=Path,
        help="Absolute root whose exact immediate child is the MCAP session.",
    )
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Legacy directory containing raw float32 x/y/z/feature frames.",
    )
    parser.add_argument(
        "--session",
        help="Exact MCAP session-directory basename (no traversal or nesting).",
    )
    parser.add_argument(
        "--model",
        choices=finalist_aliases(),
        help="Registered finalist alias for MCAP playback.",
    )
    parser.add_argument(
        "--run",
        dest="run_id",
        type=run_id_argument,
        metavar="RUN_ID",
        help="Canonical completed run ID for legacy raw-float32 playback.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="Canonical run root containing the selected run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device used for inference (ignored by --validate-only).",
    )
    parser.add_argument(
        "--feature-profile",
        choices=(KAPOSVAR_FEATURE_PROFILE,),
        default=KAPOSVAR_FEATURE_PROFILE,
        help="Fixed MCAP point-feature normalization profile.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help=(
            "Post-NMS confidence threshold; MCAP defaults to 0.3 and legacy "
            "raw playback retains its 0.0 default."
        ),
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Zero-based MCAP frame index at which streaming begins.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frame outcomes to process.",
    )
    parser.add_argument(
        "--playback-rate",
        type=float,
        default=1.0,
        help="Positive capture-time playback multiplier for block pacing.",
    )
    parser.add_argument(
        "--on-frame-error",
        choices=("stop", "continue"),
        default="stop",
        help="Stop by default; continue only after recoverable frame errors.",
    )
    parser.add_argument(
        "--visualize-bev",
        action="store_true",
        help="Render normalized MCAP points and detections interactively in BEV.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Decode/validate/normalize MCAP frames without loading a model.",
    )
    parser.add_argument(
        "--extension",
        type=str,
        default=".bin",
        help="Legacy raw-frame filename extension.",
    )
    return parser


def _validate_common_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.max_frames is not None and args.max_frames < 0:
        parser.error("--max-frames must be greater than or equal to zero")
    if args.start_frame < 0:
        parser.error("--start-frame must be greater than or equal to zero")
    if not isfinite(args.playback_rate) or args.playback_rate <= 0.0:
        parser.error("--playback-rate must be finite and strictly positive")
    if args.score_threshold is not None and not isfinite(args.score_threshold):
        parser.error("--score-threshold must be finite")


def _run_legacy(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.run_id is None:
        parser.error("legacy --input-dir mode requires --run")
    if args.session is not None or args.model is not None:
        parser.error("legacy --input-dir mode does not accept --session or --model")
    if args.validate_only or args.visualize_bev:
        parser.error(
            "--validate-only and --visualize-bev are available only in MCAP mode"
        )
    if args.start_frame != 0:
        parser.error("legacy --input-dir mode does not accept --start-frame")
    if args.playback_rate != 1.0 or args.on_frame_error != "stop":
        parser.error(
            "legacy --input-dir mode does not accept MCAP pacing/error options"
        )

    from .formats.raw_float32 import RawFloat32DirectorySource

    source = RawFloat32DirectorySource(
        directory=args.input_dir,
        extension=args.extension,
    )
    frames = source.list_frames(limit=args.max_frames)
    if not frames:
        print("No matching legacy raw-float32 frames found.")
        return 1

    detector = Mmdet3dDetector(
        run=args.runs_root / args.run_id,
        device=args.device,
        score_threshold=(
            0.0 if args.score_threshold is None else args.score_threshold
        ),
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
        len(processed.result.detections) for processed in processed_frames
    )
    average_inference_ms = sum(
        processed.result.inference_ms for processed in processed_frames
    ) / frame_count
    average_total_ms = sum(
        processed.total_ms for processed in processed_frames
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


def _format_vector(values: numpy.ndarray | tuple[float, ...]) -> str:
    return "(" + ",".join(f"{float(value):.6g}" for value in values) + ")"


def _print_mcap_identity(source: object, model_alias: str) -> None:
    spec = finalist_spec(model_alias)
    calibration = source.calibration
    schema = source.schema
    fields = ",".join(
        f"{field.name}@{field.offset}:{field.datatype_name}"
        for field in schema.required_fields
    )
    print(
        f"Session: id={source.session_id} frames={source.frame_count} "
        f"mcap={source.mcap_path}"
    )
    print(
        f"Schema: topic={schema.topic} type={schema.type_name} "
        f"frame_id={schema.frame_id} "
        f"endianness={'little' if schema.little_endian else 'big'} "
        f"height={schema.height} point_step={schema.point_step} "
        f"fields={fields}"
    )
    print(
        f"Calibration: parent={calibration.parent_frame_id} "
        f"child={calibration.child_frame_id} "
        f"translation_xyz={_format_vector(calibration.translation_xyz)} "
        f"quaternion_xyzw={_format_vector(calibration.quaternion_xyzw)} "
        "model_transform=rotation_only coordinate_frame=lidar"
    )
    print(
        f"Normalization: profile={source.feature_profile} "
        "feature=clip(reflectivity,0,255)/255"
    )
    print(
        f"Model: alias={spec.model_alias} run_id={spec.run_id} "
        f"selected_checkpoint_sha256={spec.checkpoint_sha256} "
        f"selected_checkpoint_size_bytes={spec.checkpoint_size_bytes}"
    )


def _in_range_count(points: numpy.ndarray) -> int:
    x_min, y_min, z_min, x_max, y_max, z_max = FINALIST_POINT_CLOUD_RANGE
    inside = (
        (points[:, 0] > x_min)
        & (points[:, 1] > y_min)
        & (points[:, 2] > z_min)
        & (points[:, 0] < x_max)
        & (points[:, 1] < y_max)
        & (points[:, 2] < z_max)
    )
    return int(numpy.count_nonzero(inside))


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    return float(numpy.percentile(numpy.asarray(values), quantile))


def _format_source_error(prefix: str, evidence: FrameErrorEvidence) -> str:
    return (
        f"{prefix}: session={evidence.session_id} "
        f"frame={evidence.frame_index} code={evidence.code} "
        f"header_timestamp_ns={evidence.header_timestamp_ns} "
        f"storage_timestamp_ns={evidence.storage_timestamp_ns} "
        f"source_key={evidence.source_key} "
        f"decode_ms={evidence.decode_ms:.3f} "
        f"recoverable={evidence.recoverable} message={evidence.message}"
    )


def _run_validate_only(source: object, args: argparse.Namespace) -> int:
    frame_count = 0
    error_count = 0
    empty_count = 0
    source_points = 0
    normalized_points = 0
    dropped_nonfinite = 0
    decode_timings: list[float] = []
    iterator = iter(source.iter_frames(start_index=args.start_frame))
    while args.max_frames is None or frame_count + error_count < args.max_frames:
        try:
            frame = next(iterator)
        except StopIteration:
            break
        except FrameSourceError as error:
            evidence = error.evidence
            if (
                args.on_frame_error == "continue"
                and evidence.recoverable
                and evidence.frame_index is not None
                and evidence.frame_index < args.start_frame
            ):
                # The source validates the skipped prefix for session-wide
                # ordering. Explicit continue keeps recoverable prefix errors
                # outside the requested output/max-frame window.
                continue
            print(_format_source_error("Frame error", evidence))
            error_count += 1
            if args.on_frame_error == "stop" or not evidence.recoverable:
                raise
            continue

        points = frame.points
        in_range_count = _in_range_count(points)
        if frame.source_point_count == 0:
            status = "empty_source"
        elif frame.normalized_point_count == 0:
            status = "empty_after_nonfinite_filter"
        elif in_range_count == 0:
            status = "empty_after_range_filter"
        else:
            status = "validated"
        if status.startswith("empty_"):
            empty_count += 1

        if frame.normalized_point_count:
            xyz_min = _format_vector(points[:, :3].min(axis=0))
            xyz_max = _format_vector(points[:, :3].max(axis=0))
            feature_range = (
                f"({float(points[:, 3].min()):.6g},"
                f"{float(points[:, 3].max()):.6g})"
            )
        else:
            xyz_min = xyz_max = feature_range = "n/a"
        print(
            f"Frame: session={frame.session_id} index={frame.frame_index} "
            f"header_timestamp_ns={frame.timestamp_ns} "
            f"storage_timestamp_ns={frame.storage_timestamp_ns} "
            f"source_frame={frame.source_frame_id} "
            f"coordinate_frame={frame.coordinate_frame} status={status} "
            f"source_points={frame.source_point_count} "
            f"normalized_points={frame.normalized_point_count} "
            f"dropped_nonfinite={frame.dropped_nonfinite_count} "
            f"in_range_points={in_range_count} feature_range={feature_range} "
            f"xyz_min={xyz_min} xyz_max={xyz_max} "
            f"decode_ms={frame.decode_ms:.3f}"
        )
        frame_count += 1
        source_points += frame.source_point_count
        normalized_points += frame.normalized_point_count
        dropped_nonfinite += frame.dropped_nonfinite_count
        decode_timings.append(frame.decode_ms)

    decode_mean = (
        sum(decode_timings) / len(decode_timings) if decode_timings else 0.0
    )
    print(
        f"Validation summary: session={source.session_id} frames={frame_count} "
        f"errors={error_count} empty_frames={empty_count} "
        f"source_points={source_points} normalized_points={normalized_points} "
        f"dropped_nonfinite={dropped_nonfinite} "
        f"decode_ms_mean={decode_mean:.3f} "
        f"decode_ms_p50={_percentile(decode_timings, 50):.3f} "
        f"decode_ms_p95={_percentile(decode_timings, 95):.3f}"
    )
    return 0 if error_count == 0 else 2


def _print_detection_frame(result: object) -> None:
    errors = ";".join(
        f"{error.phase}:{error.code}:{error.message}" for error in result.errors
    )
    print(
        f"Detection: session={result.session_id} frame={result.frame_index} "
        f"header_timestamp_ns={result.timestamp_ns} "
        f"storage_timestamp_ns={result.storage_timestamp_ns} "
        f"source_frame={result.source_frame_id} "
        f"coordinate_frame={result.coordinate_frame} "
        f"source_key={result.source_key} "
        f"model={result.model_alias} run_id={result.run_id} "
        f"checkpoint_sha256={result.checkpoint_sha256} status={result.status} "
        f"source_points={result.source_point_count} "
        f"normalized_points={result.normalized_point_count} "
        f"dropped_nonfinite={result.dropped_nonfinite_count} "
        f"in_range_points={result.in_range_point_count} "
        f"detections={result.detection_count} decode_ms={result.decode_ms:.3f} "
        f"detector_ms={result.detector_ms:.3f} "
        f"frame_processing_ms={result.frame_processing_ms:.3f} "
        f"pacing_lag_ms={result.pacing_lag_ms:.3f} errors={errors or 'none'}"
    )


def _run_mcap(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.session is None or args.model is None:
        parser.error("MCAP --recording-root mode requires --session and --model")
    if args.run_id is not None:
        parser.error("MCAP --recording-root mode does not accept legacy --run")
    if args.extension != ".bin":
        parser.error("MCAP --recording-root mode does not accept --extension")
    threshold = 0.3 if args.score_threshold is None else args.score_threshold
    if not 0.0 <= threshold <= 1.0:
        parser.error("MCAP --score-threshold must be in the closed interval [0, 1]")

    from .formats.ros2_mcap import Ros2McapRecordingSequence

    session_directory = resolve_session_directory(
        args.recording_root,
        args.session,
    )
    source = Ros2McapRecordingSequence(
        session_directory,
        feature_profile=args.feature_profile,
    )
    _print_mcap_identity(source, args.model)
    if args.validate_only:
        if args.visualize_bev:
            parser.error("--validate-only cannot be combined with --visualize-bev")
        return _run_validate_only(source, args)
    if args.max_frames == 0:
        print("No MCAP frames requested (--max-frames 0).")
        return 1

    detector = FinalistDetector(
        args.model,
        args.runs_root,
        device=args.device,
        score_threshold=threshold,
    )
    print(
        f"Execution: config_sha256={detector.config_sha256} "
        f"selected_checkpoint={detector.checkpoint_path} "
        f"selected_checkpoint_sha256={detector.checkpoint_sha256} "
        f"device={detector.device} score_threshold={threshold:.6g}"
    )
    viewer = None
    if args.visualize_bev:
        from .bev_viewer import BevViewer

        viewer = BevViewer()
    processor = SessionProcessor(
        detector,
        playback_rate=args.playback_rate,
        on_frame_error=args.on_frame_error,
        observer=None if viewer is None else viewer.render,
    )
    try:
        for result in processor.process(
            source,
            start_index=args.start_frame,
            max_frames=args.max_frames,
        ):
            _print_detection_frame(result)
    finally:
        if viewer is not None:
            viewer.close()

    summary = processor.summary
    print(
        f"Playback summary: session={summary.session_id} "
        f"frames={summary.frame_count} successes={summary.success_count} "
        f"errors={summary.error_count} empty_frames={summary.empty_frame_count} "
        f"detections={summary.detection_count} "
        f"dropped_nonfinite={summary.dropped_nonfinite_point_count} "
        f"decode_ms_mean={summary.decode_ms_mean:.3f} "
        f"decode_ms_p50={summary.decode_ms_p50:.3f} "
        f"decode_ms_p95={summary.decode_ms_p95:.3f} "
        f"detector_ms_mean={summary.detector_ms_mean:.3f} "
        f"detector_ms_p50={summary.detector_ms_p50:.3f} "
        f"detector_ms_p95={summary.detector_ms_p95:.3f} "
        f"frame_processing_ms_mean={summary.frame_processing_ms_mean:.3f} "
        f"frame_processing_ms_p50={summary.frame_processing_ms_p50:.3f} "
        f"frame_processing_ms_p95={summary.frame_processing_ms_p95:.3f} "
        f"processing_fps={summary.processing_fps or 0.0:.3f} "
        f"final_pacing_lag_ms={summary.final_pacing_lag_ms:.3f}"
    )
    return 0 if summary.error_count == 0 else 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_common_arguments(parser, args)
    if args.recording_root is None and args.input_dir is None:
        parser.error("choose exactly one input mode: --recording-root or --input-dir")

    try:
        if args.recording_root is not None:
            return _run_mcap(parser, args)
        return _run_legacy(parser, args)
    except FrameSourceError as error:
        evidence = error.evidence
        print(
            _format_source_error("Playback source error", evidence),
            file=sys.stderr,
        )
        return 2
    except (
        FileNotFoundError,
        NotADirectoryError,
        ImportError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"Playback error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
