"""Validated streaming ROS2 MCAP source for Kaposvar Ouster recordings."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from time import perf_counter
from typing import Any, Final, Protocol

import numpy as np
import yaml

from ..contracts import (
    FrameErrorEvidence,
    FrameSourceError,
    PointCloudFrame,
    SessionCalibration,
)
from ..normalization import (
    DETECTOR_COORDINATE_FRAME,
    KAPOSVAR_FEATURE_PROFILE,
    normalize_points,
)


POINT_TOPIC: Final = "/lexus3/os_center/points"
POINT_TYPE: Final = "sensor_msgs/msg/PointCloud2"
STATIC_TF_TOPIC: Final = "/tf_static"
STATIC_TF_TYPE: Final = "tf2_msgs/msg/TFMessage"
SENSOR_FRAME: Final = "lexus3/os_center"
BASE_FRAME: Final = "lexus3/base_link"

EXPECTED_TRANSLATION_XYZ: Final = (0.75, 0.0, 1.91)
EXPECTED_QUATERNION_XYZW: Final = (0.0, 0.0, -1.0, 0.0)
# Recorded transforms are floating-point messages.  One micrometre and one
# microradian-scale quaternion-component tolerance accept representation noise
# while rejecting a materially different installation.
CALIBRATION_TRANSLATION_ATOL: Final = 1e-6
CALIBRATION_QUATERNION_ATOL: Final = 1e-6
CALIBRATION_CONSISTENCY_ATOL: Final = 1e-9

POINT_STEP: Final = 48
POINT_HEIGHT: Final = 1


@dataclass(frozen=True, slots=True)
class RequiredPointField:
    name: str
    offset: int
    datatype: int
    datatype_name: str
    size_bytes: int


REQUIRED_POINT_FIELDS: Final = (
    RequiredPointField("x", 0, 7, "float32", 4),
    RequiredPointField("y", 4, 7, "float32", 4),
    RequiredPointField("z", 8, 7, "float32", 4),
    RequiredPointField("intensity", 16, 7, "float32", 4),
    RequiredPointField("t", 20, 6, "uint32", 4),
    RequiredPointField("reflectivity", 24, 4, "uint16", 2),
    RequiredPointField("ring", 26, 4, "uint16", 2),
    RequiredPointField("ambient", 28, 4, "uint16", 2),
    RequiredPointField("range", 32, 6, "uint32", 4),
)

_POINT_FIELD_SIZES: Final = {
    1: 1,  # INT8
    2: 1,  # UINT8
    3: 2,  # INT16
    4: 2,  # UINT16
    5: 4,  # INT32
    6: 4,  # UINT32
    7: 4,  # FLOAT32
    8: 8,  # FLOAT64
}


@dataclass(frozen=True, slots=True)
class PointCloudSchema:
    topic: str = POINT_TOPIC
    type_name: str = POINT_TYPE
    frame_id: str = SENSOR_FRAME
    little_endian: bool = True
    height: int = POINT_HEIGHT
    point_step: int = POINT_STEP
    required_fields: tuple[RequiredPointField, ...] = REQUIRED_POINT_FIELDS


@dataclass(frozen=True, slots=True)
class SerializedBagMessage:
    topic: str
    data: bytes | bytearray | memoryview
    storage_timestamp_ns: object


class _ReaderFactory(Protocol):
    def __call__(
        self,
        session_directory: Path,
        topics: tuple[str, ...],
    ) -> Iterator[SerializedBagMessage]: ...


class _FrameValidationFailure(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        header_timestamp_ns: int | None = None,
    ) -> None:
        self.code = code
        self.header_timestamp_ns = header_timestamp_ns
        super().__init__(message)


def _session_error(session_id: str, code: str, message: str) -> FrameSourceError:
    return FrameSourceError(
        FrameErrorEvidence(
            code=code,
            message=message,
            session_id=session_id,
            frame_index=None,
            header_timestamp_ns=None,
            storage_timestamp_ns=None,
            source_key=None,
            recoverable=False,
        )
    )


def _strict_nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise _FrameValidationFailure(
            "invalid_pointcloud_schema",
            f"{field_name} must be an integer and not a boolean",
        )
    result = int(value)
    if result < 0:
        raise _FrameValidationFailure(
            "invalid_pointcloud_schema",
            f"{field_name} must be nonnegative",
        )
    return result


def _metadata_integer(value: object, description: str, session_id: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise _session_error(
            session_id,
            "invalid_metadata",
            f"{description} must be a nonnegative integer",
        )
    return int(value)


def _resolve_metadata(
    session_directory: Path,
    session_id: str,
) -> tuple[Path, int]:
    metadata_path = session_directory / "metadata.yaml"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise _session_error(
            session_id,
            "invalid_metadata",
            "session must contain one regular metadata.yaml",
        )
    try:
        document = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise _session_error(
            session_id,
            "invalid_metadata",
            f"could not read metadata.yaml: {error}",
        ) from error
    if not isinstance(document, Mapping):
        raise _session_error(session_id, "invalid_metadata", "metadata is not a map")
    information = document.get("rosbag2_bagfile_information")
    if not isinstance(information, Mapping):
        raise _session_error(
            session_id,
            "invalid_metadata",
            "missing rosbag2_bagfile_information",
        )
    if information.get("storage_identifier") != "mcap":
        raise _session_error(
            session_id,
            "invalid_metadata",
            "storage_identifier must be exactly 'mcap'",
        )

    relative_paths = information.get("relative_file_paths")
    if (
        not isinstance(relative_paths, Sequence)
        or isinstance(relative_paths, (str, bytes))
        or len(relative_paths) != 1
        or not isinstance(relative_paths[0], str)
    ):
        raise _session_error(
            session_id,
            "invalid_metadata",
            "metadata must declare exactly one MCAP file",
        )
    declared_name = relative_paths[0]
    declared_relative = Path(declared_name)
    if (
        declared_relative.is_absolute()
        or len(declared_relative.parts) != 1
        or declared_relative.name != declared_name
        or declared_relative.suffix.lower() != ".mcap"
    ):
        raise _session_error(
            session_id,
            "invalid_metadata",
            "declared MCAP must be one immediate relative .mcap file",
        )
    mcap_path = session_directory / declared_relative
    try:
        resolved_mcap = mcap_path.resolve(strict=True)
    except OSError as error:
        raise _session_error(
            session_id,
            "invalid_metadata",
            f"declared MCAP is missing: {declared_name}",
        ) from error
    if (
        mcap_path.is_symlink()
        or not resolved_mcap.is_file()
        or resolved_mcap.parent != session_directory
    ):
        raise _session_error(
            session_id,
            "invalid_metadata",
            "declared MCAP must resolve inside the exact session directory",
        )
    actual_mcaps = tuple(
        sorted(
            path.name
            for path in session_directory.iterdir()
            if path.suffix.lower() == ".mcap"
        )
    )
    if actual_mcaps != (declared_name,):
        raise _session_error(
            session_id,
            "invalid_metadata",
            "session MCAP files do not exactly match metadata",
        )

    topics = information.get("topics_with_message_count")
    if not isinstance(topics, Sequence) or isinstance(topics, (str, bytes)):
        raise _session_error(
            session_id,
            "invalid_metadata",
            "topics_with_message_count must be a sequence",
        )
    required: dict[str, tuple[str, str, int]] = {}
    for entry in topics:
        if not isinstance(entry, Mapping):
            continue
        metadata = entry.get("topic_metadata")
        if not isinstance(metadata, Mapping):
            continue
        name = metadata.get("name")
        if name not in (POINT_TOPIC, STATIC_TF_TOPIC):
            continue
        if name in required:
            raise _session_error(
                session_id,
                "invalid_metadata",
                f"duplicate metadata entry for {name}",
            )
        required[name] = (
            str(metadata.get("type")),
            str(metadata.get("serialization_format")),
            _metadata_integer(entry.get("message_count"), f"{name} message_count", session_id),
        )
    expected = {
        POINT_TOPIC: POINT_TYPE,
        STATIC_TF_TOPIC: STATIC_TF_TYPE,
    }
    for topic, expected_type in expected.items():
        if topic not in required:
            raise _session_error(
                session_id,
                "missing_required_topic",
                f"metadata does not declare {topic}",
            )
        type_name, serialization, _ = required[topic]
        if type_name != expected_type:
            raise _session_error(
                session_id,
                "invalid_topic_type",
                f"{topic} type must be {expected_type}, observed {type_name}",
            )
        if serialization != "cdr":
            raise _session_error(
                session_id,
                "invalid_serialization",
                f"{topic} serialization format must be cdr",
            )
    frame_count = required[POINT_TOPIC][2]
    return resolved_mcap, frame_count


class _RosbagMessageIterator:
    def __init__(self, reader: Any) -> None:
        self._reader = reader

    def __iter__(self) -> _RosbagMessageIterator:
        return self

    def __next__(self) -> SerializedBagMessage:
        if not self._reader.has_next():
            raise StopIteration
        topic, data, timestamp = self._reader.read_next()
        return SerializedBagMessage(topic, data, timestamp)


def _default_reader_factory(
    session_directory: Path,
    topics: tuple[str, ...],
) -> Iterator[SerializedBagMessage]:
    """Open a filtered rosbag2 stream; ROS is imported only on demand."""

    try:
        rosbag2_py = importlib.import_module("rosbag2_py")
    except ImportError as error:
        raise RuntimeError(
            "rosbag2_py is unavailable; source ROS2 Humble before playback"
        ) from error
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(
            uri=str(session_directory),
            storage_id="mcap",
        ),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    return _RosbagMessageIterator(reader)


def _default_deserialize(data: object, type_name: str) -> object:
    """Deserialize only the two explicitly supported ROS message types."""

    try:
        serialization = importlib.import_module("rclpy.serialization")
        if type_name == POINT_TYPE:
            message_class = importlib.import_module("sensor_msgs.msg").PointCloud2
        elif type_name == STATIC_TF_TYPE:
            message_class = importlib.import_module("tf2_msgs.msg").TFMessage
        else:
            raise ValueError(f"unsupported ROS message type: {type_name}")
    except ImportError as error:
        raise RuntimeError(
            "ROS2 Python message support is unavailable; source ROS2 Humble "
            "before playback"
        ) from error
    return serialization.deserialize_message(data, message_class)


def _finite_real(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{description} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{description} must be finite")
    return result


def _quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion)
    return np.array(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _calibration_values(transform: object) -> tuple[np.ndarray, np.ndarray]:
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    translation_values = np.array(
        tuple(
            _finite_real(getattr(translation, axis), f"translation.{axis}")
            for axis in ("x", "y", "z")
        ),
        dtype=np.float64,
    )
    quaternion = np.array(
        tuple(
            _finite_real(getattr(rotation, axis), f"rotation.{axis}")
            for axis in ("x", "y", "z", "w")
        ),
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(quaternion))
    if not isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise ValueError("calibration quaternion has zero or invalid norm")
    quaternion /= norm
    expected = np.asarray(EXPECTED_QUATERNION_XYZW, dtype=np.float64)
    if float(np.dot(quaternion, expected)) < 0.0:
        quaternion *= -1.0
    return translation_values, quaternion


def _validated_sensor_to_base_calibration(
    translation: np.ndarray,
    quaternion: np.ndarray,
) -> SessionCalibration:
    if not np.allclose(
        translation,
        np.asarray(EXPECTED_TRANSLATION_XYZ),
        atol=CALIBRATION_TRANSLATION_ATOL,
        rtol=0.0,
    ):
        raise ValueError(
            f"recorded translation differs from {EXPECTED_TRANSLATION_XYZ}"
        )
    if not np.allclose(
        quaternion,
        np.asarray(EXPECTED_QUATERNION_XYZW),
        atol=CALIBRATION_QUATERNION_ATOL,
        rtol=0.0,
    ):
        raise ValueError(
            f"recorded rotation differs from {EXPECTED_QUATERNION_XYZW}"
        )
    return SessionCalibration(
        parent_frame_id=BASE_FRAME,
        child_frame_id=SENSOR_FRAME,
        translation_xyz=tuple(float(value) for value in translation),
        quaternion_xyzw=tuple(float(value) for value in quaternion),
        rotation_matrix=_quaternion_to_rotation(quaternion),
    )


def calibration_from_transform(transform: object) -> SessionCalibration:
    """Validate one live ``TransformStamped`` against recorded evidence."""

    try:
        parent = transform.header.frame_id
        child = transform.child_frame_id
    except AttributeError as error:
        raise ValueError("transform is missing parent or child frame identity") from error
    if parent != BASE_FRAME or child != SENSOR_FRAME:
        raise ValueError(
            f"transform must be exactly {BASE_FRAME} <- {SENSOR_FRAME}"
        )
    translation, quaternion = _calibration_values(transform)
    return _validated_sensor_to_base_calibration(translation, quaternion)


def _resolve_calibration(
    session_directory: Path,
    session_id: str,
    reader_factory: _ReaderFactory,
    deserialize: Callable[[object, str], object],
) -> SessionCalibration:
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    wrong_parents: set[str] = set()
    try:
        records = iter(reader_factory(session_directory, (STATIC_TF_TOPIC,)))
        for record in records:
            if record.topic != STATIC_TF_TOPIC:
                raise ValueError(f"filtered reader returned {record.topic!r}")
            message = deserialize(record.data, STATIC_TF_TYPE)
            transforms = getattr(message, "transforms")
            for transform in transforms:
                child = getattr(transform, "child_frame_id")
                if child != SENSOR_FRAME:
                    continue
                parent = getattr(transform.header, "frame_id")
                if parent != BASE_FRAME:
                    wrong_parents.add(str(parent))
                    continue
                candidates.append(_calibration_values(transform))
    except FrameSourceError:
        raise
    except Exception as error:
        raise _session_error(
            session_id,
            "invalid_calibration",
            f"could not resolve recorded static transform: {error}",
        ) from error

    if wrong_parents:
        raise _session_error(
            session_id,
            "ambiguous_calibration",
            f"{SENSOR_FRAME} also has unexpected parent(s): "
            + ", ".join(sorted(wrong_parents)),
        )
    if not candidates:
        raise _session_error(
            session_id,
            "missing_calibration",
            f"missing static transform {BASE_FRAME} <- {SENSOR_FRAME}",
        )
    first_translation, first_quaternion = candidates[0]
    for translation, quaternion in candidates[1:]:
        if not np.allclose(
            translation,
            first_translation,
            atol=CALIBRATION_CONSISTENCY_ATOL,
            rtol=0.0,
        ) or not np.allclose(
            quaternion,
            first_quaternion,
            atol=CALIBRATION_CONSISTENCY_ATOL,
            rtol=0.0,
        ):
            raise _session_error(
                session_id,
                "conflicting_calibration",
                "recorded static transforms disagree",
            )

    try:
        return _validated_sensor_to_base_calibration(
            first_translation,
            first_quaternion,
        )
    except ValueError as error:
        raise _session_error(
            session_id,
            "unexpected_calibration",
            str(error),
        ) from error


def pointcloud_header_timestamp(message: object) -> int:
    """Return a validated positive nanosecond PointCloud2 header stamp."""

    try:
        stamp = message.header.stamp
        sec = stamp.sec
        nanosec = stamp.nanosec
    except AttributeError as error:
        raise _FrameValidationFailure(
            "invalid_header_timestamp",
            "PointCloud2 is missing header.stamp",
        ) from error
    if isinstance(sec, bool) or not isinstance(sec, Integral) or sec < 0:
        raise _FrameValidationFailure(
            "invalid_header_timestamp",
            "header stamp seconds must be a nonnegative integer",
        )
    if (
        isinstance(nanosec, bool)
        or not isinstance(nanosec, Integral)
        or nanosec < 0
        or nanosec >= 1_000_000_000
    ):
        raise _FrameValidationFailure(
            "invalid_header_timestamp",
            "header stamp nanoseconds must be in [0, 1000000000)",
        )
    timestamp = int(sec) * 1_000_000_000 + int(nanosec)
    if timestamp <= 0:
        raise _FrameValidationFailure(
            "invalid_header_timestamp",
            "PointCloud2 header timestamp must be greater than zero",
        )
    return timestamp


def pointcloud2_to_frame(
    message: object,
    *,
    session_id: str,
    frame_index: int,
    calibration: SessionCalibration,
    feature_profile: str = KAPOSVAR_FEATURE_PROFILE,
    source_key: str = POINT_TOPIC,
    clock: Callable[[], float] = perf_counter,
) -> PointCloudFrame:
    """Convert one live PointCloud2 using the canonical MCAP schema contract.

    A live ROS subscription has no MCAP storage timestamp, so that evidence is
    intentionally ``None`` rather than copied from the acquisition stamp.
    """

    started_at = clock()
    header_timestamp_ns: int | None = None
    if not isinstance(calibration, SessionCalibration):
        raise TypeError("calibration must be a SessionCalibration")
    if (
        calibration.parent_frame_id != BASE_FRAME
        or calibration.child_frame_id != SENSOR_FRAME
    ):
        raise ValueError("calibration frame identity is incompatible")
    try:
        header_timestamp_ns = pointcloud_header_timestamp(message)
        payload = _validate_point_cloud(message, header_timestamp_ns)
        try:
            normalized = normalize_points(
                feature_profile,
                payload,
                point_count=int(message.width),
                point_step=POINT_STEP,
                rotation_matrix=calibration.rotation_matrix,
            )
        except Exception as error:
            raise _FrameValidationFailure(
                "normalization_failed",
                f"could not normalize PointCloud2: {error}",
                header_timestamp_ns=header_timestamp_ns,
            ) from error
        frame = PointCloudFrame(
            session_id=session_id,
            frame_index=frame_index,
            timestamp_ns=header_timestamp_ns,
            storage_timestamp_ns=None,
            source_frame_id=SENSOR_FRAME,
            coordinate_frame=DETECTOR_COORDINATE_FRAME,
            source_key=source_key,
            points=normalized.points,
            source_point_count=normalized.source_point_count,
            dropped_nonfinite_count=normalized.dropped_nonfinite_count,
            feature_profile=feature_profile,
            decode_ms=0.0,
        )
        object.__setattr__(
            frame,
            "decode_ms",
            max(0.0, (clock() - started_at) * 1000.0),
        )
        return frame
    except _FrameValidationFailure as error:
        elapsed_ms = max(0.0, (clock() - started_at) * 1000.0)
        raise FrameSourceError(
            FrameErrorEvidence(
                code=error.code,
                message=str(error),
                session_id=session_id,
                frame_index=frame_index,
                header_timestamp_ns=(
                    error.header_timestamp_ns
                    if error.header_timestamp_ns is not None
                    else header_timestamp_ns
                ),
                storage_timestamp_ns=None,
                source_key=source_key,
                recoverable=True,
                decode_ms=elapsed_ms,
            )
        ) from error


def _validate_fields(fields: object, header_timestamp_ns: int) -> None:
    if not isinstance(fields, Sequence):
        raise _FrameValidationFailure(
            "invalid_pointcloud_schema",
            "PointCloud2 fields must be a sequence",
            header_timestamp_ns=header_timestamp_ns,
        )
    observed: dict[str, tuple[int, int, int, int]] = {}
    occupied: list[tuple[int, int, str]] = []
    for field in fields:
        try:
            name = field.name
            offset = field.offset
            datatype = field.datatype
            count = field.count
        except AttributeError as error:
            raise _FrameValidationFailure(
                "invalid_pointcloud_schema",
                "PointCloud2 field declaration is incomplete",
                header_timestamp_ns=header_timestamp_ns,
            ) from error
        if not isinstance(name, str) or not name:
            raise _FrameValidationFailure(
                "invalid_pointcloud_schema",
                "PointCloud2 field names must be nonempty strings",
                header_timestamp_ns=header_timestamp_ns,
            )
        if name in observed:
            raise _FrameValidationFailure(
                "invalid_pointcloud_schema",
                f"duplicate PointCloud2 field {name!r}",
                header_timestamp_ns=header_timestamp_ns,
            )
        try:
            field_offset = _strict_nonnegative_int(offset, f"field {name} offset")
            field_datatype = _strict_nonnegative_int(
                datatype,
                f"field {name} datatype",
            )
            field_count = _strict_nonnegative_int(count, f"field {name} count")
        except _FrameValidationFailure as error:
            error.header_timestamp_ns = header_timestamp_ns
            raise
        if field_count == 0 or field_datatype not in _POINT_FIELD_SIZES:
            raise _FrameValidationFailure(
                "invalid_pointcloud_schema",
                f"field {name!r} has unsupported datatype/count",
                header_timestamp_ns=header_timestamp_ns,
            )
        end = field_offset + _POINT_FIELD_SIZES[field_datatype] * field_count
        if end > POINT_STEP:
            raise _FrameValidationFailure(
                "invalid_pointcloud_schema",
                f"field {name!r} exceeds point_step",
                header_timestamp_ns=header_timestamp_ns,
            )
        for prior_start, prior_end, prior_name in occupied:
            if field_offset < prior_end and prior_start < end:
                raise _FrameValidationFailure(
                    "invalid_pointcloud_schema",
                    f"field {name!r} overlaps field {prior_name!r}",
                    header_timestamp_ns=header_timestamp_ns,
                )
        occupied.append((field_offset, end, name))
        observed[name] = (field_offset, field_datatype, field_count, end)

    for required in REQUIRED_POINT_FIELDS:
        declaration = observed.get(required.name)
        expected = (required.offset, required.datatype, 1)
        if declaration is None or declaration[:3] != expected:
            raise _FrameValidationFailure(
                "invalid_pointcloud_schema",
                f"field {required.name!r} must be offset={required.offset}, "
                f"datatype={required.datatype_name}, count=1",
                header_timestamp_ns=header_timestamp_ns,
            )


def _validate_point_cloud(message: object, header_timestamp_ns: int) -> memoryview:
    try:
        frame_id = message.header.frame_id
        height = message.height
        width = message.width
        is_bigendian = message.is_bigendian
        point_step = message.point_step
        row_step = message.row_step
        fields = message.fields
        data = message.data
    except AttributeError as error:
        raise _FrameValidationFailure(
            "invalid_pointcloud_schema",
            "PointCloud2 is missing required members",
            header_timestamp_ns=header_timestamp_ns,
        ) from error
    if frame_id != SENSOR_FRAME:
        raise _FrameValidationFailure(
            "invalid_pointcloud_schema",
            f"frame_id must be exactly {SENSOR_FRAME!r}",
            header_timestamp_ns=header_timestamp_ns,
        )
    try:
        cloud_height = _strict_nonnegative_int(height, "height")
        cloud_width = _strict_nonnegative_int(width, "width")
        cloud_point_step = _strict_nonnegative_int(point_step, "point_step")
        cloud_row_step = _strict_nonnegative_int(row_step, "row_step")
    except _FrameValidationFailure as error:
        error.header_timestamp_ns = header_timestamp_ns
        raise
    if cloud_height != POINT_HEIGHT:
        raise _FrameValidationFailure(
            "invalid_pointcloud_schema",
            "organized PointCloud2 data is unsupported; height must equal 1",
            header_timestamp_ns=header_timestamp_ns,
        )
    if is_bigendian is not False:
        raise _FrameValidationFailure(
            "invalid_pointcloud_schema",
            "PointCloud2 must be little-endian",
            header_timestamp_ns=header_timestamp_ns,
        )
    if cloud_point_step != POINT_STEP:
        raise _FrameValidationFailure(
            "invalid_pointcloud_schema",
            f"point_step must equal {POINT_STEP}",
            header_timestamp_ns=header_timestamp_ns,
        )
    expected_row_step = cloud_width * POINT_STEP
    if cloud_row_step != expected_row_step:
        raise _FrameValidationFailure(
            "invalid_pointcloud_schema",
            "row_step must equal width * point_step",
            header_timestamp_ns=header_timestamp_ns,
        )
    _validate_fields(fields, header_timestamp_ns)
    try:
        payload = memoryview(data)
    except TypeError as error:
        raise _FrameValidationFailure(
            "truncated_pointcloud_payload",
            "PointCloud2 data does not support the buffer protocol",
            header_timestamp_ns=header_timestamp_ns,
        ) from error
    if payload.nbytes != cloud_row_step:
        raise _FrameValidationFailure(
            "truncated_pointcloud_payload",
            f"data length {payload.nbytes} does not equal row_step {cloud_row_step}",
            header_timestamp_ns=header_timestamp_ns,
        )
    return payload


class _McapFrameIterator:
    def __init__(self, source: Ros2McapRecordingSequence, start_index: int) -> None:
        self._source = source
        self._start_index = start_index
        self._records: Iterator[SerializedBagMessage] | None = None
        self._next_frame_index = 0
        self._previous_header_timestamp_ns: int | None = None
        self._exhaustion_checked = False

    def __iter__(self) -> _McapFrameIterator:
        return self

    def _error(
        self,
        *,
        code: str,
        message: str,
        frame_index: int,
        header_timestamp_ns: int | None,
        storage_timestamp_ns: int | None,
        source_key: str,
        started_at: float,
        recoverable: bool = True,
    ) -> FrameSourceError:
        elapsed_ms = max(0.0, (self._source._clock() - started_at) * 1000.0)
        return FrameSourceError(
            FrameErrorEvidence(
                code=code,
                message=message,
                session_id=self._source.session_id,
                frame_index=frame_index,
                header_timestamp_ns=header_timestamp_ns,
                storage_timestamp_ns=storage_timestamp_ns,
                source_key=source_key,
                recoverable=recoverable,
                decode_ms=elapsed_ms,
            )
        )

    def __next__(self) -> PointCloudFrame:
        while True:
            started_at = self._source._clock()
            try:
                if self._records is None:
                    self._records = iter(
                        self._source._reader_factory(
                            self._source.session_directory,
                            (POINT_TOPIC,),
                        )
                    )
                record = next(self._records)
            except StopIteration:
                if (
                    not self._exhaustion_checked
                    and self._next_frame_index != self._source.frame_count
                ):
                    self._exhaustion_checked = True
                    raise self._error(
                        code="message_count_mismatch",
                        message=(
                            f"metadata declared {self._source.frame_count} point "
                            f"messages, reader yielded {self._next_frame_index}"
                        ),
                        frame_index=self._next_frame_index,
                        header_timestamp_ns=None,
                        storage_timestamp_ns=None,
                        source_key=f"{self._source.mcap_path.name}:{POINT_TOPIC}",
                        started_at=started_at,
                        recoverable=False,
                    )
                raise
            except Exception as error:
                raise self._error(
                    code="reader_failure",
                    message=f"could not read filtered MCAP message: {error}",
                    frame_index=self._next_frame_index,
                    header_timestamp_ns=None,
                    storage_timestamp_ns=None,
                    source_key=f"{self._source.mcap_path.name}:{POINT_TOPIC}",
                    started_at=started_at,
                    recoverable=False,
                ) from error

            frame_index = self._next_frame_index
            self._next_frame_index += 1
            source_key = (
                f"{self._source.mcap_path.name}:{POINT_TOPIC}[{frame_index}]"
            )
            storage_timestamp_ns: int | None = None
            header_timestamp_ns: int | None = None
            try:
                if record.topic != POINT_TOPIC:
                    raise _FrameValidationFailure(
                        "unexpected_topic",
                        f"filtered reader returned {record.topic!r}",
                    )
                raw_storage_timestamp = record.storage_timestamp_ns
                if (
                    isinstance(raw_storage_timestamp, bool)
                    or not isinstance(raw_storage_timestamp, Integral)
                    or raw_storage_timestamp <= 0
                ):
                    raise _FrameValidationFailure(
                        "invalid_storage_timestamp",
                        "storage timestamp must be a positive integer",
                    )
                storage_timestamp_ns = int(raw_storage_timestamp)
                try:
                    message = self._source._deserialize(record.data, POINT_TYPE)
                except Exception as error:
                    raise _FrameValidationFailure(
                        "cdr_deserialization_failed",
                        f"could not deserialize PointCloud2: {error}",
                    ) from error
                header_timestamp_ns = pointcloud_header_timestamp(message)
                if (
                    self._previous_header_timestamp_ns is not None
                    and header_timestamp_ns <= self._previous_header_timestamp_ns
                ):
                    raise _FrameValidationFailure(
                        "non_monotonic_timestamp",
                        "PointCloud2 header timestamps must be strictly increasing",
                        header_timestamp_ns=header_timestamp_ns,
                    )
                self._previous_header_timestamp_ns = header_timestamp_ns
                # A valid header participates in the session ordering contract
                # even when this message later fails schema/normalization and an
                # explicit continue policy moves on to the next frame.
                payload = _validate_point_cloud(message, header_timestamp_ns)
                try:
                    normalized = normalize_points(
                        self._source.feature_profile,
                        payload,
                        point_count=int(message.width),
                        point_step=POINT_STEP,
                        rotation_matrix=self._source.calibration.rotation_matrix,
                    )
                except Exception as error:
                    raise _FrameValidationFailure(
                        "normalization_failed",
                        f"could not normalize PointCloud2: {error}",
                        header_timestamp_ns=header_timestamp_ns,
                    ) from error
                frame = PointCloudFrame(
                    session_id=self._source.session_id,
                    frame_index=frame_index,
                    timestamp_ns=header_timestamp_ns,
                    storage_timestamp_ns=storage_timestamp_ns,
                    source_frame_id=SENSOR_FRAME,
                    coordinate_frame=DETECTOR_COORDINATE_FRAME,
                    source_key=source_key,
                    points=normalized.points,
                    source_point_count=normalized.source_point_count,
                    dropped_nonfinite_count=normalized.dropped_nonfinite_count,
                    feature_profile=self._source.feature_profile,
                    decode_ms=0.0,
                )
                # The frame is still private.  End timing only after its immutable
                # array has been materialized, then publish it to the caller.
                object.__setattr__(
                    frame,
                    "decode_ms",
                    max(0.0, (self._source._clock() - started_at) * 1000.0),
                )
            except _FrameValidationFailure as error:
                raise self._error(
                    code=error.code,
                    message=str(error),
                    frame_index=frame_index,
                    header_timestamp_ns=(
                        error.header_timestamp_ns
                        if error.header_timestamp_ns is not None
                        else header_timestamp_ns
                    ),
                    storage_timestamp_ns=storage_timestamp_ns,
                    source_key=source_key,
                    started_at=started_at,
                ) from error

            if frame_index < self._start_index:
                continue
            if frame_index >= self._source.frame_count:
                raise self._error(
                    code="message_count_mismatch",
                    message="reader yielded more point messages than metadata declared",
                    frame_index=frame_index,
                    header_timestamp_ns=header_timestamp_ns,
                    storage_timestamp_ns=storage_timestamp_ns,
                    source_key=source_key,
                    started_at=started_at,
                    recoverable=False,
                )
            return frame


class Ros2McapRecordingSequence:
    """One exact, independent ROS2 MCAP recording session.

    Construction validates metadata and resolves the recorded static transform.
    Point messages remain streaming and are decoded only by ``iter_frames``.
    The underscore-prefixed dependency seams exist solely for deterministic
    synthetic tests; production always uses the supported rosbag2/CDR stack.
    """

    schema: Final = PointCloudSchema()

    def __init__(
        self,
        session_directory: Path,
        *,
        feature_profile: str = KAPOSVAR_FEATURE_PROFILE,
        _reader_factory: _ReaderFactory | None = None,
        _deserialize: Callable[[object, str], object] | None = None,
        _clock: Callable[[], float] = perf_counter,
    ) -> None:
        if not isinstance(session_directory, Path):
            raise TypeError("session_directory must be a pathlib.Path")
        if not session_directory.is_absolute():
            raise ValueError("session_directory must be an absolute path")
        if session_directory.is_symlink():
            raise ValueError("session_directory must not be a symlink")
        try:
            resolved_session = session_directory.resolve(strict=True)
        except OSError as error:
            raise FileNotFoundError(
                f"session directory does not exist: {session_directory}"
            ) from error
        if not resolved_session.is_dir():
            raise NotADirectoryError(
                f"session path is not a directory: {session_directory}"
            )
        if not resolved_session.name or resolved_session.name != session_directory.name:
            raise ValueError("session directory basename must be preserved verbatim")
        if feature_profile != KAPOSVAR_FEATURE_PROFILE:
            raise ValueError(f"unsupported feature profile: {feature_profile!r}")
        if not callable(_clock):
            raise TypeError("_clock must be callable")

        self.session_directory = resolved_session
        self._session_id = session_directory.name
        self.feature_profile = feature_profile
        self._reader_factory = _reader_factory or _default_reader_factory
        self._deserialize = _deserialize or _default_deserialize
        self._clock = _clock
        self.mcap_path, self._frame_count = _resolve_metadata(
            resolved_session,
            self._session_id,
        )
        try:
            self.calibration = _resolve_calibration(
                resolved_session,
                self._session_id,
                self._reader_factory,
                self._deserialize,
            )
        except FrameSourceError:
            raise
        except Exception as error:
            raise _session_error(
                self._session_id,
                "invalid_calibration",
                str(error),
            ) from error

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def iter_frames(self, start_index: int = 0) -> Iterator[PointCloudFrame]:
        if isinstance(start_index, bool) or not isinstance(start_index, Integral):
            raise TypeError("start_index must be an integer and not a boolean")
        start_index = int(start_index)
        if start_index < 0:
            raise ValueError("start_index must be nonnegative")
        return _McapFrameIterator(self, start_index)
