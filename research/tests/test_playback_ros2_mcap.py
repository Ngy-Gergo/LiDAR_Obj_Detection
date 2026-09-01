from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from lidar_model_selection.playback.contracts import FrameSourceError
from lidar_model_selection.playback.formats.ros2_mcap import (
    BASE_FRAME,
    POINT_TOPIC,
    POINT_TYPE,
    SENSOR_FRAME,
    STATIC_TF_TOPIC,
    STATIC_TF_TYPE,
    Ros2McapRecordingSequence,
    SerializedBagMessage,
)


def _field(name: str, offset: int, datatype: int, count: int = 1):
    return SimpleNamespace(
        name=name,
        offset=offset,
        datatype=datatype,
        count=count,
    )


def _required_fields():
    return [
        _field("x", 0, 7),
        _field("y", 4, 7),
        _field("z", 8, 7),
        _field("intensity", 16, 7),
        _field("t", 20, 6),
        _field("reflectivity", 24, 4),
        _field("ring", 26, 4),
        _field("ambient", 28, 4),
        _field("range", 32, 6),
    ]


def _cloud(
    timestamp_ns: int,
    rows: list[tuple[float, float, float, int]] | None = None,
    **changes,
):
    rows = [(1.0, 2.0, 3.0, 128)] if rows is None else rows
    payload = bytearray(len(rows) * 48)
    for index, (x, y, z, reflectivity) in enumerate(rows):
        offset = index * 48
        struct.pack_into("<f", payload, offset, x)
        struct.pack_into("<f", payload, offset + 4, y)
        struct.pack_into("<f", payload, offset + 8, z)
        struct.pack_into("<f", payload, offset + 16, 1.0)
        struct.pack_into("<I", payload, offset + 20, index)
        struct.pack_into("<H", payload, offset + 24, reflectivity)
        struct.pack_into("<H", payload, offset + 26, index)
        struct.pack_into("<H", payload, offset + 28, 5)
        struct.pack_into("<I", payload, offset + 32, 100)
    values = {
        "header": SimpleNamespace(
            stamp=SimpleNamespace(
                sec=timestamp_ns // 1_000_000_000,
                nanosec=timestamp_ns % 1_000_000_000,
            ),
            frame_id=SENSOR_FRAME,
        ),
        "height": 1,
        "width": len(rows),
        "fields": _required_fields(),
        "is_bigendian": False,
        "point_step": 48,
        "row_step": len(rows) * 48,
        "data": bytes(payload),
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _transform(
    *,
    parent: str = BASE_FRAME,
    child: str = SENSOR_FRAME,
    translation: tuple[float, float, float] = (0.75, 0.0, 1.91),
    quaternion: tuple[float, float, float, float] = (0.0, 0.0, -1.0, 4e-11),
):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id=parent),
        child_frame_id=child,
        transform=SimpleNamespace(
            translation=SimpleNamespace(
                x=translation[0],
                y=translation[1],
                z=translation[2],
            ),
            rotation=SimpleNamespace(
                x=quaternion[0],
                y=quaternion[1],
                z=quaternion[2],
                w=quaternion[3],
            ),
        ),
    )


def _metadata(point_count: int = 1, *, point_type: str = POINT_TYPE):
    return {
        "rosbag2_bagfile_information": {
            "storage_identifier": "mcap",
            "relative_file_paths": ["recording_0.mcap"],
            "topics_with_message_count": [
                {
                    "topic_metadata": {
                        "name": POINT_TOPIC,
                        "type": point_type,
                        "serialization_format": "cdr",
                    },
                    "message_count": point_count,
                },
                {
                    "topic_metadata": {
                        "name": STATIC_TF_TOPIC,
                        "type": STATIC_TF_TYPE,
                        "serialization_format": "cdr",
                    },
                    "message_count": 1,
                },
                {
                    "topic_metadata": {
                        "name": "/camera/image_raw",
                        "type": "sensor_msgs/msg/Image",
                        "serialization_format": "cdr",
                    },
                    "message_count": 999,
                },
            ],
        }
    }


def _session_directory(
    tmp_path: Path,
    *,
    name: str = "proba_lexus3_2026-07-27_14-09",
    point_count: int = 1,
    point_type: str = POINT_TYPE,
) -> Path:
    session = tmp_path / name
    session.mkdir()
    (session / "recording_0.mcap").write_bytes(b"synthetic-index-only")
    (session / "metadata.yaml").write_text(
        yaml.safe_dump(_metadata(point_count, point_type=point_type)),
        encoding="utf-8",
    )
    return session


def _source(
    session: Path,
    points: list[SerializedBagMessage],
    *,
    transforms=None,
    clock=None,
):
    calls: list[tuple[str, ...]] = []
    tf_message = SimpleNamespace(
        transforms=[_transform()] if transforms is None else transforms
    )

    def reader_factory(_directory: Path, topics: tuple[str, ...]):
        calls.append(topics)
        if topics == (STATIC_TF_TOPIC,):
            return iter(
                [SerializedBagMessage(STATIC_TF_TOPIC, tf_message, 1)]
            )
        if topics == (POINT_TOPIC,):
            return iter(points)
        raise AssertionError(f"unexpected topic filter: {topics}")

    kwargs = {
        "_reader_factory": reader_factory,
        "_deserialize": lambda data, _type: data,
    }
    if clock is not None:
        kwargs["_clock"] = clock
    return Ros2McapRecordingSequence(session, **kwargs), calls


def _point_record(timestamp_ns: int, storage_timestamp_ns: int, **changes):
    return SerializedBagMessage(
        POINT_TOPIC,
        _cloud(timestamp_ns, **changes),
        storage_timestamp_ns,
    )


def test_exact_session_identity_topic_filters_order_timestamps_and_calibration(
    tmp_path: Path,
) -> None:
    session = _session_directory(tmp_path, point_count=2)
    source, calls = _source(
        session,
        [
            _point_record(10, 101),
            _point_record(20, 202, rows=[(-4.0, 5.0, 6.0, 300)]),
        ],
    )
    frames = list(source.iter_frames())

    assert source.session_id == "proba_lexus3_2026-07-27_14-09"
    assert source.frame_count == 2
    assert source.mcap_path == session / "recording_0.mcap"
    assert source.schema.topic == POINT_TOPIC
    assert source.schema.type_name == POINT_TYPE
    assert source.calibration.parent_frame_id == BASE_FRAME
    assert source.calibration.child_frame_id == SENSOR_FRAME
    assert source.calibration.translation_xyz == (0.75, 0.0, 1.91)
    np.testing.assert_allclose(
        source.calibration.rotation_matrix,
        np.diag((-1.0, -1.0, 1.0)),
        atol=1e-9,
    )
    assert calls == [(STATIC_TF_TOPIC,), (POINT_TOPIC,)]
    assert [frame.frame_index for frame in frames] == [0, 1]
    assert [frame.timestamp_ns for frame in frames] == [10, 20]
    assert [frame.storage_timestamp_ns for frame in frames] == [101, 202]
    assert all(frame.coordinate_frame == "lidar" for frame in frames)
    np.testing.assert_allclose(frames[0].points[0, :3], [-1.0, -2.0, 3.0])
    np.testing.assert_allclose(frames[1].points[0], [4.0, -5.0, 6.0, 1.0])
    assert all(frame.decode_ms >= 0.0 for frame in frames)


def test_start_index_streams_global_indices_and_does_not_preload(tmp_path: Path) -> None:
    session = _session_directory(tmp_path, point_count=3)
    source, calls = _source(
        session,
        [
            _point_record(10, 101),
            _point_record(20, 102),
            _point_record(30, 103),
        ],
    )
    assert calls == [(STATIC_TF_TOPIC,)]
    iterator = source.iter_frames(start_index=2)
    assert calls == [(STATIC_TF_TOPIC,)]
    frame = next(iterator)
    assert calls == [(STATIC_TF_TOPIC,), (POINT_TOPIC,)]
    assert frame.frame_index == 2
    assert frame.timestamp_ns == 30
    with pytest.raises(StopIteration):
        next(iterator)


def test_start_index_still_reports_validation_errors_in_skipped_prefix(
    tmp_path: Path,
) -> None:
    session = _session_directory(tmp_path, point_count=3)
    source, _ = _source(
        session,
        [
            _point_record(10, 101, data=b"short"),
            _point_record(20, 102),
            _point_record(30, 103),
        ],
    )

    iterator = source.iter_frames(start_index=2)
    with pytest.raises(FrameSourceError) as captured:
        next(iterator)
    frame = next(iterator)

    assert captured.value.evidence.frame_index == 0
    assert captured.value.evidence.recoverable
    assert frame.frame_index == 2
    assert frame.timestamp_ns == 30


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"height": 2}, "height must equal 1"),
        ({"is_bigendian": True}, "little-endian"),
        ({"point_step": 32}, "point_step must equal 48"),
        ({"row_step": 47}, "row_step must equal"),
        ({"data": b"short"}, "data length"),
        (
            {
                "header": SimpleNamespace(
                    stamp=SimpleNamespace(sec=0, nanosec=0),
                    frame_id=SENSOR_FRAME,
                )
            },
            "timestamp must be greater",
        ),
    ],
)
def test_invalid_cloud_layout_is_structured(
    tmp_path: Path,
    change: dict[str, object],
    message: str,
) -> None:
    session = _session_directory(tmp_path)
    source, _ = _source(session, [_point_record(10, 100, **change)])
    with pytest.raises(FrameSourceError, match=message) as captured:
        next(source.iter_frames())
    assert captured.value.evidence.frame_index == 0
    assert captured.value.evidence.recoverable
    assert captured.value.evidence.source_key is not None


@pytest.mark.parametrize(
    "fields",
    [
        [field for field in _required_fields() if field.name != "reflectivity"],
        [
            *[field for field in _required_fields() if field.name != "x"],
            _field("x", 1, 7),
        ],
        [
            *[field for field in _required_fields() if field.name != "ring"],
            _field("ring", 26, 2),
        ],
        [*_required_fields(), _field("conflict", 24, 2)],
    ],
)
def test_required_field_name_type_offset_and_overlap_are_strict(
    tmp_path: Path,
    fields: list[object],
) -> None:
    session = _session_directory(tmp_path)
    source, _ = _source(session, [_point_record(10, 100, fields=fields)])
    with pytest.raises(FrameSourceError) as captured:
        next(source.iter_frames())
    assert captured.value.evidence.code == "invalid_pointcloud_schema"


def test_nonconflicting_additional_field_is_allowed(tmp_path: Path) -> None:
    session = _session_directory(tmp_path)
    fields = [*_required_fields(), _field("extra", 12, 2, count=4)]
    source, _ = _source(session, [_point_record(10, 100, fields=fields)])
    assert next(source.iter_frames()).normalized_point_count == 1


def test_duplicate_and_regressing_timestamps_error_but_iterator_can_continue(
    tmp_path: Path,
) -> None:
    session = _session_directory(tmp_path, point_count=4)
    source, _ = _source(
        session,
        [
            _point_record(10, 100),
            _point_record(10, 101),
            _point_record(9, 102),
            _point_record(20, 103),
        ],
    )
    iterator = source.iter_frames()
    assert next(iterator).frame_index == 0
    with pytest.raises(FrameSourceError) as duplicate:
        next(iterator)
    assert duplicate.value.evidence.code == "non_monotonic_timestamp"
    assert duplicate.value.evidence.header_timestamp_ns == 10
    with pytest.raises(FrameSourceError) as regressing:
        next(iterator)
    assert regressing.value.evidence.frame_index == 2
    assert next(iterator).frame_index == 3


def test_valid_header_retains_ordering_state_when_schema_error_is_skipped(
    tmp_path: Path,
) -> None:
    session = _session_directory(tmp_path, point_count=3)
    source, _ = _source(
        session,
        [
            _point_record(10, 100),
            _point_record(30, 101, data=b"truncated"),
            _point_record(20, 102),
        ],
    )
    iterator = source.iter_frames()
    assert next(iterator).timestamp_ns == 10
    with pytest.raises(FrameSourceError) as schema_error:
        next(iterator)
    assert schema_error.value.evidence.header_timestamp_ns == 30
    with pytest.raises(FrameSourceError) as regression:
        next(iterator)
    assert regression.value.evidence.code == "non_monotonic_timestamp"


@pytest.mark.parametrize(
    ("transforms", "code"),
    [
        ([], "missing_calibration"),
        ([_transform(parent="wrong/base")], "ambiguous_calibration"),
        (
            [
                _transform(),
                _transform(translation=(0.7501, 0.0, 1.91)),
            ],
            "conflicting_calibration",
        ),
        ([_transform(translation=(0.8, 0.0, 1.91))], "unexpected_calibration"),
        ([_transform(translation=(float("nan"), 0.0, 1.91))], "invalid_calibration"),
        ([_transform(quaternion=(0.0, 0.0, 0.0, 0.0))], "invalid_calibration"),
    ],
)
def test_missing_ambiguous_conflicting_and_different_calibration_rejected(
    tmp_path: Path,
    transforms: list[object],
    code: str,
) -> None:
    session = _session_directory(tmp_path)
    with pytest.raises(FrameSourceError) as captured:
        _source(session, [_point_record(10, 100)], transforms=transforms)
    assert captured.value.evidence.code == code
    assert not captured.value.evidence.recoverable


def test_calibration_tolerance_and_quaternion_sign_are_supported(tmp_path: Path) -> None:
    session = _session_directory(tmp_path)
    transform = _transform(
        translation=(0.7500005, -0.0000005, 1.9100005),
        quaternion=(0.0, 0.0, 2.0, -8e-11),
    )
    source, _ = _source(
        session,
        [_point_record(10, 100)],
        transforms=[transform],
    )
    np.testing.assert_allclose(
        source.calibration.rotation_matrix,
        np.diag((-1.0, -1.0, 1.0)),
        atol=1e-9,
    )


def test_metadata_requires_exact_mcap_and_topic_type(tmp_path: Path) -> None:
    wrong_type = _session_directory(tmp_path, point_type="wrong/PointCloud2")
    with pytest.raises(FrameSourceError) as type_error:
        _source(wrong_type, [_point_record(10, 100)])
    assert type_error.value.evidence.code == "invalid_topic_type"

    extra = _session_directory(tmp_path, name="extra-mcap")
    (extra / "undeclared.mcap").write_bytes(b"extra")
    with pytest.raises(FrameSourceError, match="exactly match metadata"):
        _source(extra, [_point_record(10, 100)])


def test_metadata_requires_cdr_serialization(tmp_path: Path) -> None:
    session = _session_directory(tmp_path)
    metadata_path = session / "metadata.yaml"
    document = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    topics = document["rosbag2_bagfile_information"]["topics_with_message_count"]
    for entry in topics:
        if entry["topic_metadata"]["name"] == POINT_TOPIC:
            entry["topic_metadata"]["serialization_format"] = "ros1"
    metadata_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(FrameSourceError) as captured:
        _source(session, [_point_record(10, 100)])
    assert captured.value.evidence.code == "invalid_serialization"


def test_storage_timestamp_and_metadata_message_count_are_strict(
    tmp_path: Path,
) -> None:
    invalid_storage = _session_directory(tmp_path, name="invalid-storage")
    source, _ = _source(invalid_storage, [_point_record(10, 0)])
    with pytest.raises(FrameSourceError) as captured:
        next(source.iter_frames())
    assert captured.value.evidence.code == "invalid_storage_timestamp"
    assert captured.value.evidence.storage_timestamp_ns is None

    short = _session_directory(tmp_path, name="short-reader", point_count=2)
    source, _ = _source(short, [_point_record(10, 100)])
    iterator = source.iter_frames()
    assert next(iterator).frame_index == 0
    with pytest.raises(FrameSourceError) as captured:
        next(iterator)
    assert captured.value.evidence.code == "message_count_mismatch"
    assert not captured.value.evidence.recoverable

    long = _session_directory(tmp_path, name="long-reader", point_count=1)
    source, _ = _source(
        long,
        [_point_record(10, 100), _point_record(20, 200)],
    )
    iterator = source.iter_frames()
    assert next(iterator).frame_index == 0
    with pytest.raises(FrameSourceError) as captured:
        next(iterator)
    assert captured.value.evidence.code == "message_count_mismatch"
    assert not captured.value.evidence.recoverable


def test_session_and_declared_mcap_symlinks_are_rejected(tmp_path: Path) -> None:
    real = _session_directory(tmp_path, name="real-session")
    linked_session = tmp_path / "linked-session"
    linked_session.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="must not be a symlink"):
        _source(linked_session, [_point_record(10, 100)])

    linked_mcap_session = _session_directory(tmp_path, name="linked-mcap")
    declared = linked_mcap_session / "recording_0.mcap"
    external = tmp_path / "external.mcap"
    external.write_bytes(b"external")
    declared.unlink()
    declared.symlink_to(external)
    with pytest.raises(FrameSourceError, match="resolve inside"):
        _source(linked_mcap_session, [_point_record(10, 100)])


def test_point_reader_open_failure_is_structured_and_nonrecoverable(
    tmp_path: Path,
) -> None:
    session = _session_directory(tmp_path)
    tf_message = SimpleNamespace(transforms=[_transform()])

    def reader_factory(_directory: Path, topics: tuple[str, ...]):
        if topics == (STATIC_TF_TOPIC,):
            return iter([SerializedBagMessage(STATIC_TF_TOPIC, tf_message, 1)])
        raise OSError("reader open failed")

    source = Ros2McapRecordingSequence(
        session,
        _reader_factory=reader_factory,
        _deserialize=lambda data, _type: data,
    )
    with pytest.raises(FrameSourceError) as captured:
        next(source.iter_frames())
    assert captured.value.evidence.code == "reader_failure"
    assert not captured.value.evidence.recoverable
    assert captured.value.evidence.header_timestamp_ns is None


def test_frame_error_has_measured_decode_scope_without_fabricated_timestamp(
    tmp_path: Path,
) -> None:
    session = _session_directory(tmp_path)
    ticks = iter((1.0, 1.125))
    bad = SerializedBagMessage(POINT_TOPIC, object(), 100)
    source, _ = _source(session, [bad], clock=lambda: next(ticks))
    with pytest.raises(FrameSourceError) as captured:
        next(source.iter_frames())
    evidence = captured.value.evidence
    assert evidence.code == "invalid_header_timestamp"
    assert evidence.header_timestamp_ns is None
    assert evidence.decode_ms == pytest.approx(125.0)


def test_cdr_failure_is_structured_without_fabricated_header_timestamp(
    tmp_path: Path,
) -> None:
    session = _session_directory(tmp_path)
    tf_message = SimpleNamespace(transforms=[_transform()])

    def reader_factory(_directory: Path, topics: tuple[str, ...]):
        if topics == (STATIC_TF_TOPIC,):
            return iter([SerializedBagMessage(STATIC_TF_TOPIC, tf_message, 1)])
        return iter([SerializedBagMessage(POINT_TOPIC, b"bad cdr", 100)])

    def deserialize(data: object, type_name: str):
        if type_name == STATIC_TF_TYPE:
            return data
        raise ValueError("malformed CDR")

    source = Ros2McapRecordingSequence(
        session,
        _reader_factory=reader_factory,
        _deserialize=deserialize,
    )
    with pytest.raises(FrameSourceError) as captured:
        next(source.iter_frames())
    assert captured.value.evidence.code == "cdr_deserialization_failed"
    assert captured.value.evidence.header_timestamp_ns is None


def test_independent_sessions_and_iterators_do_not_share_timestamp_state(
    tmp_path: Path,
) -> None:
    first_dir = _session_directory(tmp_path, name="session-one")
    second_dir = _session_directory(tmp_path, name="session-two")
    first, _ = _source(first_dir, [_point_record(50, 100)])
    second, _ = _source(second_dir, [_point_record(10, 200)])
    assert next(first.iter_frames()).timestamp_ns == 50
    assert next(second.iter_frames()).timestamp_ns == 10
    assert next(first.iter_frames()).frame_index == 0
    assert first.session_id != second.session_id
