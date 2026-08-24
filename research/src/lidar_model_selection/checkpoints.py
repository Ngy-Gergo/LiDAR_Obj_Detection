"""Checkpoint validation and directory-local artifact identity.

Checkpoint capture is observational rather than atomic. Callers must not
mutate checkpoint files or the supplied training directory during capture.
"""

from __future__ import annotations

import os
import re
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from lidar_model_selection.provenance import identify_file

__all__ = [
    "ArtifactMismatch",
    "CheckpointArtifact",
    "CheckpointKind",
    "CheckpointLocation",
    "TrainingOutputs",
    "checkpoint_epoch",
    "identify_checkpoint",
    "is_usable_checkpoint",
    "list_epoch_checkpoints",
    "select_training_outputs",
    "verify_checkpoint",
]

CheckpointKind = Literal["epoch", "best", "external"]
CheckpointLocation = Literal["run", "external"]

_CHECKPOINT_KINDS = frozenset({"epoch", "best", "external"})
_CHECKPOINT_LOCATIONS = frozenset({"run", "external"})
_EPOCH_CHECKPOINT_PATTERN = re.compile(
    r"\Aepoch_(?P<epoch>[0-9]+)\.pth\Z",
    re.ASCII,
)
_BEST_CHECKPOINT_PATTERN = re.compile(
    r"\Abest_[^/]*\.pth\Z",
    re.ASCII,
)
_BEST_EPOCH_PATTERN = re.compile(
    r"\Abest_(?:[^/]*_)?epoch_(?P<epoch>[0-9]+)\.pth\Z",
    re.ASCII,
)
_SHA256_PATTERN = re.compile(r"\A[0-9a-f]{64}\Z", re.ASCII)
_REQUIRED_PYTORCH_MEMBERS = frozenset({"data.pkl", "version"})


@dataclass(frozen=True, slots=True)
class CheckpointArtifact:
    """Immutable evidence identifying one checkpoint file."""

    path: str
    sha256: str
    size_bytes: int
    epoch: int | None
    kind: CheckpointKind
    location: CheckpointLocation


@dataclass(frozen=True, slots=True)
class TrainingOutputs:
    """The exact completion and selected inference checkpoints."""

    final_checkpoint: CheckpointArtifact
    selected_checkpoint: CheckpointArtifact


class ArtifactMismatch(ValueError):
    """A stored checkpoint artifact no longer matches usable file evidence."""


def checkpoint_epoch(path: Path) -> int | None:
    """Return an exact epoch filename or terminal best-checkpoint epoch."""
    name = Path(path).name
    match = _EPOCH_CHECKPOINT_PATTERN.fullmatch(name)
    if match is None:
        match = _BEST_EPOCH_PATTERN.fullmatch(name)
    return int(match.group("epoch")) if match is not None else None


def is_usable_checkpoint(path: Path) -> bool:
    """Return whether *path* is a nonempty, non-symlink PyTorch ZIP.

    Validation reads only ZIP metadata. It never deserializes the checkpoint.
    Required ``data.pkl`` and ``version`` members may live at archive root or
    beneath the same generated PyTorch archive prefix.
    """
    candidate = _absolute_path(Path(path))
    try:
        if _path_contains_symlink(candidate):
            return False
        file_stat = candidate.lstat()
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size == 0:
            return False
        with zipfile.ZipFile(candidate) as archive:
            return _has_required_pytorch_members(archive)
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile):
        return False


def identify_checkpoint(
    path: Path,
    *,
    repo_root: Path,
    kind: CheckpointKind,
    location: CheckpointLocation,
) -> CheckpointArtifact:
    """Validate and identify one stable checkpoint file."""
    _validate_kind(kind)
    _validate_location(location)
    root = _repository_root(repo_root)
    candidate = _input_path(path, repo_root=root)
    relative = _relative_to(candidate, root)

    if location == "run" and relative is None:
        raise ValueError("run checkpoint must be inside repository root")

    _require_usable_checkpoint(candidate)
    evidence = identify_file(
        candidate,
        relative_to=root if relative is not None else None,
    )
    epoch = checkpoint_epoch(candidate)
    _validate_kind_filename(candidate, kind)

    return CheckpointArtifact(
        path=evidence.path,
        sha256=evidence.sha256,
        size_bytes=evidence.size_bytes,
        epoch=epoch,
        kind=kind,
        location=location,
    )


def verify_checkpoint(
    artifact: CheckpointArtifact,
    *,
    repo_root: Path,
) -> Path:
    """Reverify *artifact* and return its absolute concrete path."""
    if not isinstance(artifact, CheckpointArtifact):
        raise TypeError("artifact must be a CheckpointArtifact")

    root = _repository_root(repo_root)
    try:
        _validate_kind(artifact.kind)
        _validate_location(artifact.location)
        _validate_artifact_evidence(artifact)
        candidate, relative_to = _stored_artifact_path(artifact, root)
        if not is_usable_checkpoint(candidate):
            raise ArtifactMismatch(
                f"checkpoint is missing or structurally unusable: {candidate}"
            )
        if candidate.lstat().st_size != artifact.size_bytes:
            raise ArtifactMismatch(
                "checkpoint size does not match stored artifact"
            )
        evidence = identify_file(candidate, relative_to=relative_to)
        if evidence.path != artifact.path:
            raise ArtifactMismatch("checkpoint path representation changed")
        if evidence.size_bytes != artifact.size_bytes:
            raise ArtifactMismatch(
                "checkpoint size does not match stored artifact"
            )
        if evidence.sha256 != artifact.sha256:
            raise ArtifactMismatch(
                "checkpoint SHA-256 does not match stored artifact"
            )
        _validate_kind_filename(candidate, artifact.kind)
        if checkpoint_epoch(candidate) != artifact.epoch:
            raise ArtifactMismatch(
                "checkpoint epoch does not match stored artifact"
            )
    except ArtifactMismatch:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise ArtifactMismatch(str(error)) from error

    return candidate


def list_epoch_checkpoints(
    training_dir: Path,
    *,
    repo_root: Path,
    location: CheckpointLocation = "run",
) -> tuple[CheckpointArtifact, ...]:
    """Identify direct exact numeric epoch checkpoints in ascending order."""
    _validate_location(location)
    root = _repository_root(repo_root)
    directory = _training_directory(
        training_dir,
        repo_root=root,
        location=location,
    )

    candidates_by_epoch: dict[int, list[Path]] = {}
    for path in directory.iterdir():
        match = _EPOCH_CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is not None:
            epoch = int(match.group("epoch"))
            candidates_by_epoch.setdefault(epoch, []).append(path)

    ordered_paths: list[Path] = []
    for epoch, paths in sorted(candidates_by_epoch.items()):
        paths.sort(key=lambda path: os.fsencode(path.name))
        if len(paths) > 1:
            names = ", ".join(path.name for path in paths)
            raise ValueError(
                f"multiple checkpoint files represent epoch {epoch}: {names}"
            )
        ordered_paths.append(paths[0])

    artifacts = [
        identify_checkpoint(
            path,
            repo_root=root,
            kind="epoch",
            location=location,
        )
        for path in ordered_paths
    ]
    return tuple(artifacts)


def select_training_outputs(
    training_dir: Path,
    *,
    target_epoch: int,
    repo_root: Path,
) -> TrainingOutputs:
    """Identify the exact completion checkpoint and unambiguous best output."""
    if isinstance(target_epoch, bool) or not isinstance(target_epoch, int):
        raise TypeError("target_epoch must be an integer")
    if target_epoch <= 0:
        raise ValueError("target_epoch must be positive")

    root = _repository_root(repo_root)
    directory = _training_directory(
        training_dir,
        repo_root=root,
        location="run",
    )
    final_checkpoint = identify_checkpoint(
        directory / f"epoch_{target_epoch}.pth",
        repo_root=root,
        kind="epoch",
        location="run",
    )

    best_paths = sorted(
        (
            path
            for path in directory.iterdir()
            if _BEST_CHECKPOINT_PATTERN.fullmatch(path.name) is not None
        ),
        key=lambda path: os.fsencode(path.name),
    )
    if len(best_paths) > 1:
        names = ", ".join(path.name for path in best_paths)
        raise ValueError(f"multiple best checkpoints are ambiguous: {names}")

    selected_checkpoint = final_checkpoint
    if best_paths:
        selected_checkpoint = identify_checkpoint(
            best_paths[0],
            repo_root=root,
            kind="best",
            location="run",
        )

    return TrainingOutputs(
        final_checkpoint=final_checkpoint,
        selected_checkpoint=selected_checkpoint,
    )


def _has_required_pytorch_members(archive: zipfile.ZipFile) -> bool:
    members_by_parent: dict[PurePosixPath, set[str]] = {}
    for info in archive.infolist():
        member = PurePosixPath(info.filename)
        if member.is_absolute() or ".." in member.parts:
            return False
        if info.is_dir():
            continue
        if member.name in _REQUIRED_PYTORCH_MEMBERS:
            members_by_parent.setdefault(member.parent, set()).add(
                member.name
            )
    return any(
        _REQUIRED_PYTORCH_MEMBERS.issubset(names)
        for names in members_by_parent.values()
    )


def _require_usable_checkpoint(path: Path) -> None:
    try:
        if _path_contains_symlink(path):
            raise ValueError(f"checkpoint path contains a symlink: {path}")
        file_stat = path.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"checkpoint does not exist: {path}") from error

    if stat.S_ISDIR(file_stat.st_mode):
        raise IsADirectoryError(f"checkpoint path is a directory: {path}")
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"checkpoint must be a regular file: {path}")
    if file_stat.st_size == 0:
        raise ValueError(f"checkpoint is empty: {path}")
    if not is_usable_checkpoint(path):
        raise ValueError(f"checkpoint is not a usable PyTorch ZIP: {path}")


def _validate_kind(kind: object) -> None:
    if not isinstance(kind, str):
        raise TypeError("checkpoint kind must be a string")
    if kind not in _CHECKPOINT_KINDS:
        raise ValueError(f"unsupported checkpoint kind: {kind}")


def _validate_location(location: object) -> None:
    if not isinstance(location, str):
        raise TypeError("checkpoint location must be a string")
    if location not in _CHECKPOINT_LOCATIONS:
        raise ValueError(f"unsupported checkpoint location: {location}")


def _validate_kind_filename(path: Path, kind: CheckpointKind) -> None:
    if (
        kind == "epoch"
        and _EPOCH_CHECKPOINT_PATTERN.fullmatch(path.name) is None
    ):
        raise ValueError(
            "epoch checkpoint kind requires an exact epoch_N.pth filename"
        )
    if (
        kind == "best"
        and _BEST_CHECKPOINT_PATTERN.fullmatch(path.name) is None
    ):
        raise ValueError(
            "best checkpoint kind requires a best_*.pth filename"
        )


def _validate_artifact_evidence(artifact: CheckpointArtifact) -> None:
    if (
        isinstance(artifact.size_bytes, bool)
        or not isinstance(artifact.size_bytes, int)
        or artifact.size_bytes < 0
    ):
        raise ArtifactMismatch(
            "checkpoint artifact size must be a nonnegative integer"
        )
    if (
        not isinstance(artifact.sha256, str)
        or _SHA256_PATTERN.fullmatch(artifact.sha256) is None
    ):
        raise ArtifactMismatch(
            "checkpoint artifact SHA-256 must be lowercase hexadecimal"
        )
    if artifact.epoch is not None and (
        isinstance(artifact.epoch, bool)
        or not isinstance(artifact.epoch, int)
        or artifact.epoch < 0
    ):
        raise ArtifactMismatch(
            "checkpoint artifact epoch must be a nonnegative integer or None"
        )


def _repository_root(path: Path) -> Path:
    root = _absolute_path(Path(path))
    if _path_contains_symlink(root):
        raise ValueError(f"repository root contains a symlink: {root}")
    _require_real_directory(root, description="repository root")
    return root


def _training_directory(
    path: Path,
    *,
    repo_root: Path,
    location: CheckpointLocation,
) -> Path:
    directory = _input_path(path, repo_root=repo_root)
    _require_real_directory(directory, description="training directory")
    if location == "run" and _relative_to(directory, repo_root) is None:
        raise ValueError("run training directory must be inside repository root")
    return directory


def _require_real_directory(path: Path, *, description: str) -> None:
    directory_stat = path.lstat()
    if stat.S_ISLNK(directory_stat.st_mode):
        raise ValueError(f"{description} must not be a symlink: {path}")
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise NotADirectoryError(f"{description} is not a directory: {path}")


def _input_path(path: Path, *, repo_root: Path) -> Path:
    supplied = Path(path)
    return _absolute_path(
        supplied if supplied.is_absolute() else repo_root / supplied
    )


def _stored_artifact_path(
    artifact: CheckpointArtifact,
    repo_root: Path,
) -> tuple[Path, Path | None]:
    if not isinstance(artifact.path, str) or not artifact.path:
        raise ArtifactMismatch("checkpoint artifact path must be nonempty")

    stored = Path(artifact.path)
    if stored.is_absolute():
        if artifact.location != "external":
            raise ArtifactMismatch(
                "only external checkpoints may store an absolute path"
            )
        candidate = _absolute_path(stored)
        if _relative_to(candidate, repo_root) is not None:
            raise ArtifactMismatch(
                "checkpoint inside repository must use a relative path"
            )
        return candidate, None

    candidate = _absolute_path(repo_root / stored)
    if _relative_to(candidate, repo_root) is None:
        raise ArtifactMismatch("checkpoint artifact path escapes repository")
    return candidate, repo_root


def _relative_to(path: Path, root: Path) -> Path | None:
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def _path_contains_symlink(path: Path) -> bool:
    anchor = Path(path.anchor)
    current = anchor
    for part in path.relative_to(anchor).parts:
        current /= part
        if stat.S_ISLNK(current.lstat().st_mode):
            return True
    return False


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))
