from __future__ import annotations

import hashlib
import os
from dataclasses import FrozenInstanceError
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import pytest

from lidar_model_selection.checkpoints import (
    ArtifactMismatch,
    CheckpointArtifact,
    TrainingOutputs,
    checkpoint_epoch,
    identify_checkpoint,
    list_epoch_checkpoints,
    select_training_outputs,
    verify_checkpoint,
)


def _write_checkpoint(
    path: Path,
    *,
    payload: bytes = b"pickle payload",
    data_root: str = "archive",
    version_root: str | None = None,
    version: bytes = b"3\n",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", compression=ZIP_STORED) as archive:
        archive.writestr(f"{data_root}/data.pkl", payload)
        archive.writestr(
            f"{version_root or data_root}/version",
            version,
        )
        archive.writestr(f"{data_root}/data/0", b"tensor storage")
    return path


def _corrupt_member_payload(path: Path, payload: bytes) -> None:
    contents = bytearray(path.read_bytes())
    index = contents.index(payload)
    contents[index] ^= 0xFF
    path.write_bytes(contents)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("epoch_0.pth", 0),
        ("epoch_0003.pth", 3),
        ("epoch_42.pth", 42),
        ("prefix_epoch_3.pth", None),
        ("epoch_3.pth.tmp", None),
        ("epoch_-1.pth", None),
        ("epoch_٣.pth", None),
        ("EPOCH_3.pth", None),
    ],
)
def test_checkpoint_epoch_is_a_strict_full_filename_parser(
    name: str,
    expected: int | None,
) -> None:
    assert checkpoint_epoch(Path(name)) == expected


def test_evidence_is_immutable_and_round_trips() -> None:
    final = CheckpointArtifact(
        path="training/./epoch_4.pth",
        sha256="a" * 64,
        size_bytes=123,
        epoch=4,
        checkpoint_format="pytorch_zip",
        validation_profile="pytorch-zip-structural-v1",
    )
    selected = CheckpointArtifact(
        path="training/best_metric.pth",
        sha256="b" * 64,
        size_bytes=100,
        epoch=None,
        checkpoint_format="pytorch_zip",
        validation_profile="pytorch-zip-structural-v1",
    )
    outputs = TrainingOutputs(final, selected)
    mismatch = ArtifactMismatch("sha256", "a" * 64, "b" * 64)

    assert final.path == "training/epoch_4.pth"
    assert CheckpointArtifact.from_dict(final.to_dict()) == final
    assert TrainingOutputs.from_dict(outputs.to_dict()) == outputs
    assert ArtifactMismatch.from_dict(mismatch.to_dict()) == mismatch
    with pytest.raises(FrozenInstanceError):
        final.path = "changed.pth"  # type: ignore[misc]


def test_evidence_from_dict_rejects_unknown_fields() -> None:
    value = {
        "path": "epoch_1.pth",
        "sha256": "a" * 64,
        "size_bytes": 1,
        "epoch": 1,
        "checkpoint_format": "pytorch_zip",
        "validation_profile": "pytorch-zip-structural-v1",
        "extra": True,
    }
    with pytest.raises(ValueError, match="unexpected"):
        CheckpointArtifact.from_dict(value)


def test_identify_checkpoint_records_streamed_identity_and_relative_path(
    tmp_path: Path,
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "training" / "epoch_12.pth")

    artifact = identify_checkpoint(checkpoint, root=tmp_path)

    contents = checkpoint.read_bytes()
    assert artifact.path == "training/epoch_12.pth"
    assert artifact.sha256 == hashlib.sha256(contents).hexdigest()
    assert artifact.size_bytes == len(contents)
    assert artifact.epoch == 12
    assert artifact.checkpoint_format == "pytorch_zip"
    assert artifact.validation_profile == "pytorch-zip-structural-v1"
    assert verify_checkpoint(artifact, root=tmp_path) == ()


def test_identify_checkpoint_uses_an_absolute_reference_without_root(
    tmp_path: Path,
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "best_score.pth")

    artifact = identify_checkpoint(checkpoint)

    assert Path(artifact.path).is_absolute()
    assert artifact.path == os.path.abspath(checkpoint)
    assert artifact.epoch is None
    assert verify_checkpoint(artifact) == ()


@pytest.mark.parametrize("kind", ["not_zip", "missing_version", "split_root"])
def test_identify_checkpoint_rejects_structurally_invalid_archives(
    tmp_path: Path,
    kind: str,
) -> None:
    checkpoint = tmp_path / "epoch_1.pth"
    if kind == "not_zip":
        checkpoint.write_bytes(b"not a ZIP")
    elif kind == "missing_version":
        with ZipFile(checkpoint, "w") as archive:
            archive.writestr("archive/data.pkl", b"pickle")
    else:
        _write_checkpoint(
            checkpoint,
            data_root="one",
            version_root="two",
        )

    with pytest.raises(ValueError):
        identify_checkpoint(checkpoint)


def test_identify_checkpoint_validates_member_crc(tmp_path: Path) -> None:
    payload = b"unique pickle payload"
    checkpoint = _write_checkpoint(
        tmp_path / "epoch_1.pth",
        payload=payload,
    )
    _corrupt_member_payload(checkpoint, payload)

    with pytest.raises(ValueError):
        identify_checkpoint(checkpoint)


def test_identify_checkpoint_rejects_a_symlink(tmp_path: Path) -> None:
    target = _write_checkpoint(tmp_path / "target.pth")
    link = tmp_path / "epoch_1.pth"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        identify_checkpoint(link)


def test_verify_checkpoint_reports_changed_and_missing_artifacts(
    tmp_path: Path,
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "epoch_2.pth", payload=b"old")
    artifact = identify_checkpoint(checkpoint, root=tmp_path)

    _write_checkpoint(checkpoint, payload=b"new and different")
    mismatches = verify_checkpoint(artifact, root=tmp_path)
    assert {mismatch.field for mismatch in mismatches} >= {"sha256", "size_bytes"}

    checkpoint.unlink()
    missing = verify_checkpoint(artifact, root=tmp_path)
    assert len(missing) == 1
    assert missing[0].field == "checkpoint"
    assert "FileNotFoundError" in str(missing[0].actual)


def test_verify_checkpoint_reports_a_relative_reference_without_root() -> None:
    artifact = CheckpointArtifact(
        path="training/epoch_1.pth",
        sha256="a" * 64,
        size_bytes=1,
        epoch=1,
        checkpoint_format="pytorch_zip",
        validation_profile="pytorch-zip-structural-v1",
    )

    mismatches = verify_checkpoint(artifact)

    assert len(mismatches) == 1
    assert mismatches[0].field == "path"


def test_list_epoch_checkpoints_is_scoped_sorted_and_structurally_strict(
    tmp_path: Path,
) -> None:
    training = tmp_path / "training"
    _write_checkpoint(training / "epoch_10.pth", payload=b"ten")
    _write_checkpoint(training / "epoch_2.pth", payload=b"two")
    _write_checkpoint(training / "best_score.pth", payload=b"best")
    _write_checkpoint(training / "nested" / "epoch_99.pth", payload=b"nested")

    artifacts = list_epoch_checkpoints(training, root=tmp_path)

    assert [artifact.epoch for artifact in artifacts] == [2, 10]
    assert [artifact.path for artifact in artifacts] == [
        "training/epoch_2.pth",
        "training/epoch_10.pth",
    ]


@pytest.mark.parametrize(
    "bad_name",
    ["epoch_.pth", "epoch_latest.pth", "epoch_1.pth.partial", "epoch_1.pt"],
)
def test_list_epoch_checkpoints_rejects_malformed_epoch_state(
    tmp_path: Path,
    bad_name: str,
) -> None:
    training = tmp_path / "training"
    training.mkdir()
    (training / bad_name).write_bytes(b"partial")

    with pytest.raises(ValueError, match="malformed"):
        list_epoch_checkpoints(training)


def test_list_epoch_checkpoints_rejects_duplicate_semantic_epochs(
    tmp_path: Path,
) -> None:
    training = tmp_path / "training"
    _write_checkpoint(training / "epoch_3.pth")
    _write_checkpoint(training / "epoch_0003.pth")

    with pytest.raises(ValueError, match="duplicate semantic epoch 3"):
        list_epoch_checkpoints(training)


def test_list_epoch_checkpoints_rejects_corrupt_and_symlink_state(
    tmp_path: Path,
) -> None:
    corrupt_directory = tmp_path / "corrupt"
    corrupt_directory.mkdir()
    (corrupt_directory / "epoch_1.pth").write_bytes(b"partial")
    with pytest.raises(ValueError):
        list_epoch_checkpoints(corrupt_directory)

    target = _write_checkpoint(tmp_path / "target.pth")
    linked_directory = tmp_path / "linked"
    linked_directory.mkdir()
    (linked_directory / "epoch_1.pth").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        list_epoch_checkpoints(linked_directory)


def test_list_epoch_checkpoints_rejects_a_symlink_training_directory(
    tmp_path: Path,
) -> None:
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    linked_directory = tmp_path / "linked"
    linked_directory.symlink_to(real_directory, target_is_directory=True)

    with pytest.raises(ValueError, match="training directory.*symlink"):
        list_epoch_checkpoints(linked_directory)


def test_select_training_outputs_falls_back_to_literal_final(
    tmp_path: Path,
) -> None:
    training = tmp_path / "training"
    _write_checkpoint(training / "epoch_1.pth", payload=b"one")
    _write_checkpoint(training / "epoch_2.pth", payload=b"two")

    outputs = select_training_outputs(training, 2, root=tmp_path)

    assert outputs.final_checkpoint.path == "training/epoch_2.pth"
    assert outputs.selected_checkpoint == outputs.final_checkpoint


def test_select_training_outputs_prefers_one_valid_best_checkpoint(
    tmp_path: Path,
) -> None:
    training = tmp_path / "training"
    _write_checkpoint(training / "epoch_4.pth", payload=b"final")
    _write_checkpoint(training / "best_metric.pth", payload=b"best")

    outputs = select_training_outputs(training, 4, root=tmp_path)

    assert outputs.final_checkpoint.path == "training/epoch_4.pth"
    assert outputs.selected_checkpoint.path == "training/best_metric.pth"
    assert outputs.selected_checkpoint.epoch is None


def test_select_training_outputs_requires_the_unpadded_literal_final(
    tmp_path: Path,
) -> None:
    training = tmp_path / "training"
    _write_checkpoint(training / "epoch_0004.pth")

    with pytest.raises(FileNotFoundError, match="epoch_4.pth"):
        select_training_outputs(training, 4)


def test_select_training_outputs_rejects_multiple_best_checkpoints(
    tmp_path: Path,
) -> None:
    training = tmp_path / "training"
    _write_checkpoint(training / "epoch_1.pth")
    _write_checkpoint(training / "best_one.pth", payload=b"one")
    _write_checkpoint(training / "best_two.pth", payload=b"two")

    with pytest.raises(ValueError, match="multiple best"):
        select_training_outputs(training, 1)


@pytest.mark.parametrize("kind", ["corrupt", "symlink"])
def test_select_training_outputs_rejects_an_invalid_best_checkpoint(
    tmp_path: Path,
    kind: str,
) -> None:
    training = tmp_path / "training"
    final = _write_checkpoint(training / "epoch_1.pth")
    best = training / "best_metric.pth"
    if kind == "corrupt":
        best.write_bytes(b"partial")
    else:
        best.symlink_to(final)

    with pytest.raises(ValueError):
        select_training_outputs(training, 1)
