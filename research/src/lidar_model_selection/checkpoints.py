"""Checkpoint identity, verification, and run-local output selection."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import zipfile
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO


__all__ = (
    "CheckpointArtifact",
    "TrainingOutputs",
    "ArtifactMismatch",
    "checkpoint_epoch",
    "identify_checkpoint",
    "verify_checkpoint",
    "list_epoch_checkpoints",
    "select_training_outputs",
)

_EPOCH_CHECKPOINT_PATTERN = re.compile(r"epoch_(?P<epoch>[0-9]+)\.pth")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CHECKPOINT_FORMAT = "pytorch_zip"
_VALIDATION_PROFILE = "pytorch-zip-structural-v1"
_HASH_CHUNK_SIZE = 1024 * 1024


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    description: str,
) -> None:
    actual = frozenset(value)
    if actual == expected:
        return

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    details = []
    if missing:
        details.append(f"missing {missing!r}")
    if extra:
        details.append(f"unexpected {extra!r}")
    raise ValueError(f"invalid {description} fields: {', '.join(details)}")


def _normalized_artifact_path(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("checkpoint artifact path must be a string")
    if not value:
        raise ValueError("checkpoint artifact path must not be empty")
    if "\x00" in value:
        raise ValueError("checkpoint artifact path must not contain NUL")

    normalized = os.path.normpath(value)
    if normalized == ".":
        raise ValueError("checkpoint artifact path must name a file")
    if not os.path.isabs(normalized) and ".." in Path(normalized).parts:
        raise ValueError(
            "relative checkpoint artifact path must not escape its root"
        )
    return normalized


def _require_sha256(value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("checkpoint sha256 must be a string")
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("checkpoint sha256 must be 64 lowercase hexadecimal digits")


def _require_nonnegative_integer(value: object, *, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{description} must be an integer and not a boolean")
    if value < 0:
        raise ValueError(f"{description} must be non-negative")
    return value


def _require_optional_epoch(value: object) -> int | None:
    if value is None:
        return None
    return _require_nonnegative_integer(value, description="checkpoint epoch")


def _require_nonempty_string(value: object, *, description: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{description} must be a string")
    if not value.strip():
        raise ValueError(f"{description} must contain non-whitespace text")
    return value


@dataclass(frozen=True, slots=True)
class CheckpointArtifact:
    """Stable identity and structural evidence for one checkpoint file."""

    path: str
    sha256: str
    size_bytes: int
    epoch: int | None
    checkpoint_format: str
    validation_profile: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normalized_artifact_path(self.path))
        _require_sha256(self.sha256)
        _require_nonnegative_integer(
            self.size_bytes,
            description="checkpoint size_bytes",
        )
        _require_optional_epoch(self.epoch)
        _require_nonempty_string(
            self.checkpoint_format,
            description="checkpoint format",
        )
        _require_nonempty_string(
            self.validation_profile,
            description="checkpoint validation profile",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "epoch": self.epoch,
            "checkpoint_format": self.checkpoint_format,
            "validation_profile": self.validation_profile,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CheckpointArtifact:
        if not isinstance(value, Mapping):
            raise TypeError("checkpoint artifact must be a mapping")
        _require_exact_fields(
            value,
            frozenset(
                {
                    "path",
                    "sha256",
                    "size_bytes",
                    "epoch",
                    "checkpoint_format",
                    "validation_profile",
                }
            ),
            description="checkpoint artifact",
        )
        return cls(
            path=value["path"],
            sha256=value["sha256"],
            size_bytes=value["size_bytes"],
            epoch=value["epoch"],
            checkpoint_format=value["checkpoint_format"],
            validation_profile=value["validation_profile"],
        )


_MISMATCH_VALUE_TYPES = (str, int, type(None))


@dataclass(frozen=True, slots=True)
class ArtifactMismatch:
    """One deterministic difference between recorded and observed evidence."""

    field: str
    expected: str | int | None
    actual: str | int | None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.field, description="artifact mismatch field")
        for name, value in (("expected", self.expected), ("actual", self.actual)):
            if isinstance(value, bool) or not isinstance(
                value,
                _MISMATCH_VALUE_TYPES,
            ):
                raise TypeError(
                    f"artifact mismatch {name} must be a string, integer, or None"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactMismatch:
        if not isinstance(value, Mapping):
            raise TypeError("artifact mismatch must be a mapping")
        _require_exact_fields(
            value,
            frozenset({"field", "expected", "actual"}),
            description="artifact mismatch",
        )
        return cls(
            field=value["field"],
            expected=value["expected"],
            actual=value["actual"],
        )


@dataclass(frozen=True, slots=True)
class TrainingOutputs:
    """Exact final and inference-selected checkpoint identities for a run."""

    final_checkpoint: CheckpointArtifact
    selected_checkpoint: CheckpointArtifact

    def __post_init__(self) -> None:
        if not isinstance(self.final_checkpoint, CheckpointArtifact):
            raise TypeError("final_checkpoint must be a CheckpointArtifact")
        if not isinstance(self.selected_checkpoint, CheckpointArtifact):
            raise TypeError("selected_checkpoint must be a CheckpointArtifact")
        if self.final_checkpoint.epoch is None:
            raise ValueError("final_checkpoint must have an epoch filename")

    def to_dict(self) -> dict[str, object]:
        return {
            "final_checkpoint": self.final_checkpoint.to_dict(),
            "selected_checkpoint": self.selected_checkpoint.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrainingOutputs:
        if not isinstance(value, Mapping):
            raise TypeError("training outputs must be a mapping")
        _require_exact_fields(
            value,
            frozenset({"final_checkpoint", "selected_checkpoint"}),
            description="training outputs",
        )
        final_value = value["final_checkpoint"]
        selected_value = value["selected_checkpoint"]
        if not isinstance(final_value, Mapping):
            raise TypeError("final_checkpoint must be a mapping")
        if not isinstance(selected_value, Mapping):
            raise TypeError("selected_checkpoint must be a mapping")
        return cls(
            final_checkpoint=CheckpointArtifact.from_dict(final_value),
            selected_checkpoint=CheckpointArtifact.from_dict(selected_value),
        )


def checkpoint_epoch(path: Path) -> int | None:
    """Return the epoch from a literal ``epoch_<digits>.pth`` filename."""
    if not isinstance(path, Path):
        raise TypeError("checkpoint path must be a pathlib.Path")
    match = _EPOCH_CHECKPOINT_PATTERN.fullmatch(path.name)
    return int(match.group("epoch")) if match is not None else None


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _artifact_reference(path: Path, root: Path | None) -> str:
    absolute_path = _absolute_path(path)
    if root is None:
        return os.fspath(absolute_path)

    absolute_root = _absolute_path(root)
    try:
        relative_path = absolute_path.relative_to(absolute_root)
    except ValueError as exc:
        raise ValueError(
            f"checkpoint is outside the supplied artifact root: {absolute_path}"
        ) from exc
    if relative_path == Path("."):
        raise ValueError("checkpoint path must name a file below its artifact root")
    return relative_path.as_posix()


def _checkpoint_path(artifact: CheckpointArtifact, root: Path | None) -> Path:
    reference = Path(artifact.path)
    if reference.is_absolute():
        if root is not None:
            absolute_root = _absolute_path(root)
            try:
                reference.relative_to(absolute_root)
            except ValueError as exc:
                raise ValueError(
                    "absolute checkpoint artifact path is outside the supplied root"
                ) from exc
        return _absolute_path(reference)

    if root is None:
        raise ValueError(
            "a root is required to verify a relative checkpoint artifact path"
        )
    return _absolute_path(root / reference)


def _require_real_directory(path: Path, *, description: str) -> None:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{description} must not be a symlink: {path}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise NotADirectoryError(f"{description} is not a directory: {path}")


def _require_regular_file(path: Path) -> os.stat_result:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"checkpoint must not be a symlink: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"checkpoint must be a regular file: {path}")
    if metadata.st_size == 0:
        raise ValueError(f"checkpoint must not be empty: {path}")
    return metadata


def _member_parts(info: zipfile.ZipInfo) -> tuple[str, ...]:
    member_path = PurePosixPath(info.filename)
    parts = member_path.parts
    if (
        not info.filename
        or member_path.is_absolute()
        or not parts
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"unsafe ZIP member name: {info.filename!r}")

    unix_mode = info.external_attr >> 16
    member_type = stat.S_IFMT(unix_mode)
    if member_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise ValueError(f"unsupported ZIP member type: {info.filename!r}")
    return parts


def _validate_pytorch_zip(stream: BinaryIO, *, path: Path) -> None:
    try:
        with zipfile.ZipFile(stream, mode="r") as archive:
            infos = archive.infolist()
            if not infos:
                raise ValueError("checkpoint ZIP archive is empty")

            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ValueError("checkpoint ZIP archive has duplicate members")

            members = [(info, _member_parts(info)) for info in infos]
            data_members = [
                (info, parts)
                for info, parts in members
                if not info.is_dir()
                and len(parts) == 2
                and parts[1] == "data.pkl"
            ]
            version_members = [
                (info, parts)
                for info, parts in members
                if not info.is_dir()
                and len(parts) == 2
                and parts[1] == "version"
            ]
            if len(data_members) != 1 or len(version_members) != 1:
                raise ValueError(
                    "checkpoint ZIP archive must contain exactly one data.pkl "
                    "and one version beneath a root directory"
                )

            data_info, data_parts = data_members[0]
            version_info, version_parts = version_members[0]
            common_root = data_parts[0]
            if version_parts[0] != common_root:
                raise ValueError(
                    "checkpoint ZIP data.pkl and version do not share one root"
                )
            if any(parts[0] != common_root for _, parts in members):
                raise ValueError(
                    "checkpoint ZIP members do not share one root directory"
                )
            if data_info.file_size == 0:
                raise ValueError("checkpoint ZIP data.pkl is empty")
            if version_info.file_size == 0 or version_info.file_size > 64:
                raise ValueError("checkpoint ZIP version entry has an invalid size")

            version = archive.read(version_info).strip()
            if not version.isdigit():
                raise ValueError("checkpoint ZIP version entry is not an integer")

            corrupt_member = archive.testzip()
            if corrupt_member is not None:
                raise ValueError(
                    f"checkpoint ZIP CRC failed for member {corrupt_member!r}"
                )
    except (
        EOFError,
        NotImplementedError,
        RuntimeError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        zlib.error,
    ) as exc:
        raise ValueError(f"invalid PyTorch ZIP checkpoint: {path}") from exc


def identify_checkpoint(
    path: Path,
    *,
    root: Path | None = None,
) -> CheckpointArtifact:
    """Validate and identify a checkpoint without importing or loading Torch."""
    if not isinstance(path, Path):
        raise TypeError("checkpoint path must be a pathlib.Path")
    if root is not None and not isinstance(root, Path):
        raise TypeError("checkpoint artifact root must be a pathlib.Path or None")

    lexical_path = _absolute_path(path)
    initial_metadata = _require_regular_file(lexical_path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lexical_path, flags)
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = -1
            opened_metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened_metadata.st_mode):
                raise ValueError(
                    f"checkpoint must be a regular file: {lexical_path}"
                )
            if (
                initial_metadata.st_dev,
                initial_metadata.st_ino,
            ) != (opened_metadata.st_dev, opened_metadata.st_ino):
                raise RuntimeError(
                    f"checkpoint changed while being opened: {lexical_path}"
                )

            _validate_pytorch_zip(stream, path=lexical_path)
            stream.seek(0)
            digest = hashlib.sha256()
            size_bytes = 0
            while True:
                chunk = stream.read(_HASH_CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
                size_bytes += len(chunk)

            final_metadata = os.fstat(stream.fileno())
            if (
                opened_metadata.st_size != final_metadata.st_size
                or opened_metadata.st_mtime_ns != final_metadata.st_mtime_ns
                or opened_metadata.st_ctime_ns != final_metadata.st_ctime_ns
                or size_bytes != final_metadata.st_size
            ):
                raise RuntimeError(
                    f"checkpoint changed while being identified: {lexical_path}"
                )
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    current_metadata = lexical_path.lstat()
    if stat.S_ISLNK(current_metadata.st_mode) or (
        current_metadata.st_dev,
        current_metadata.st_ino,
    ) != (final_metadata.st_dev, final_metadata.st_ino):
        raise RuntimeError(
            f"checkpoint path changed while being identified: {lexical_path}"
        )

    return CheckpointArtifact(
        path=_artifact_reference(lexical_path, root),
        sha256=digest.hexdigest(),
        size_bytes=size_bytes,
        epoch=checkpoint_epoch(lexical_path),
        checkpoint_format=_CHECKPOINT_FORMAT,
        validation_profile=_VALIDATION_PROFILE,
    )


def verify_checkpoint(
    artifact: CheckpointArtifact,
    *,
    root: Path | None = None,
) -> tuple[ArtifactMismatch, ...]:
    """Re-identify a checkpoint and report every evidence mismatch."""
    if not isinstance(artifact, CheckpointArtifact):
        raise TypeError("artifact must be a CheckpointArtifact")
    if root is not None and not isinstance(root, Path):
        raise TypeError("checkpoint artifact root must be a pathlib.Path or None")

    try:
        path = _checkpoint_path(artifact, root)
    except (TypeError, ValueError) as exc:
        return (
            ArtifactMismatch(
                field="path",
                expected=artifact.path,
                actual=f"invalid reference: {exc}",
            ),
        )

    try:
        observed = identify_checkpoint(path, root=root)
    except (OSError, RuntimeError, ValueError) as exc:
        return (
            ArtifactMismatch(
                field="checkpoint",
                expected="valid checkpoint evidence",
                actual=f"{type(exc).__name__}: {exc}",
            ),
        )

    mismatches = []
    for field in (
        "path",
        "sha256",
        "size_bytes",
        "epoch",
        "checkpoint_format",
        "validation_profile",
    ):
        expected = getattr(artifact, field)
        actual = getattr(observed, field)
        if expected != actual:
            mismatches.append(
                ArtifactMismatch(
                    field=field,
                    expected=expected,
                    actual=actual,
                )
            )
    return tuple(mismatches)


def list_epoch_checkpoints(
    training_directory: Path,
    *,
    root: Path | None = None,
) -> tuple[CheckpointArtifact, ...]:
    """Identify exact epoch checkpoints in one explicit training directory."""
    if not isinstance(training_directory, Path):
        raise TypeError("training_directory must be a pathlib.Path")
    if root is not None and not isinstance(root, Path):
        raise TypeError("checkpoint artifact root must be a pathlib.Path or None")

    directory = _absolute_path(training_directory)
    _require_real_directory(directory, description="training directory")
    artifacts_by_epoch: dict[int, CheckpointArtifact] = {}

    for path in sorted(directory.iterdir(), key=lambda candidate: candidate.name):
        if not path.name.startswith("epoch_"):
            continue

        epoch = checkpoint_epoch(path)
        if epoch is None:
            raise ValueError(f"malformed epoch checkpoint name: {path.name!r}")
        if epoch in artifacts_by_epoch:
            other = Path(artifacts_by_epoch[epoch].path).name
            raise ValueError(
                f"duplicate semantic epoch {epoch}: {other!r} and {path.name!r}"
            )

        artifact = identify_checkpoint(path, root=root)
        artifacts_by_epoch[epoch] = artifact

    return tuple(
        artifacts_by_epoch[epoch]
        for epoch in sorted(artifacts_by_epoch)
    )


def select_training_outputs(
    training_directory: Path,
    target_epoch: int,
    *,
    root: Path | None = None,
) -> TrainingOutputs:
    """Select exact final and optional unique best checkpoint identities."""
    _require_nonnegative_integer(target_epoch, description="target_epoch")
    if not isinstance(training_directory, Path):
        raise TypeError("training_directory must be a pathlib.Path")
    if root is not None and not isinstance(root, Path):
        raise TypeError("checkpoint artifact root must be a pathlib.Path or None")

    directory = _absolute_path(training_directory)
    epoch_artifacts = list_epoch_checkpoints(directory, root=root)
    final_name = f"epoch_{target_epoch}.pth"
    final_checkpoint = next(
        (
            artifact
            for artifact in epoch_artifacts
            if Path(artifact.path).name == final_name
        ),
        None,
    )
    if final_checkpoint is None:
        raise FileNotFoundError(
            f"exact final checkpoint is missing: {directory / final_name}"
        )

    best_paths = sorted(
        (
            path
            for path in directory.iterdir()
            if path.name.startswith("best_") and path.name.endswith(".pth")
        ),
        key=lambda path: path.name,
    )
    if len(best_paths) > 1:
        names = ", ".join(repr(path.name) for path in best_paths)
        raise ValueError(f"multiple best checkpoints are ambiguous: {names}")

    selected_checkpoint = (
        final_checkpoint
        if not best_paths
        else identify_checkpoint(best_paths[0], root=root)
    )
    return TrainingOutputs(
        final_checkpoint=final_checkpoint,
        selected_checkpoint=selected_checkpoint,
    )
