"""Lightweight hashing and observational provenance capture.

Workspace capture is not an atomic snapshot. Callers must not mutate the Git
working tree or explicitly identified files while they are being captured.
"""

from __future__ import annotations

import errno
import hashlib
import importlib
from importlib import metadata
import os
import platform as platform_module
import stat
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "CodeProvenance",
    "EnvironmentInfo",
    "FileArtifact",
    "FileSetIdentity",
    "capture_code_provenance",
    "capture_environment",
    "identify_file",
    "identify_file_set",
    "sha256_bytes",
    "sha256_file",
]

_FILE_BLOCK_SIZE = 1024 * 1024
_FILE_SET_HASH_SCHEME = "lidar-file-set-v1"
_WORKSPACE_HASH_SCHEME = "lidar-workspace-v1"
_GIT_TIMEOUT_SECONDS = 30
_DISTRIBUTIONS = {
    "torch": "torch",
    "mmengine": "mmengine",
    "mmcv": "mmcv",
    "mmdet": "mmdet",
    "mmdet3d": "mmdet3d",
}


@dataclass(frozen=True, slots=True)
class FileArtifact:
    """Portable identity for one regular file."""

    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class FileSetIdentity:
    """Deterministic identity and evidence for an explicit file set."""

    hash_scheme: str
    profile: str
    sha256: str
    files: tuple[FileArtifact, ...]


@dataclass(frozen=True, slots=True)
class CodeProvenance:
    """Observational Git state for a caller-supplied repository scope."""

    hash_scheme: str
    scope: tuple[str, ...]
    git_commit: str | None
    git_dirty: bool | None
    workspace_sha256: str | None
    capture_error: str | None


@dataclass(frozen=True, slots=True)
class EnvironmentInfo:
    """Lightweight software and optional selected-GPU information."""

    python: str
    platform: str
    machine: str
    torch: str | None
    torch_cuda: str | None
    mmengine: str | None
    mmcv: str | None
    mmdet: str | None
    mmdet3d: str | None
    gpu: str | None


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 hexadecimal digest of *value*."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest of *path*."""
    digest, _ = _hash_file(path)
    return digest


def identify_file(
    path: Path,
    *,
    relative_to: Path | None = None,
) -> FileArtifact:
    """Identify one regular, non-symlink file.

    Relative inputs are interpreted beneath ``relative_to`` when it is
    supplied. Common concurrent file changes are detected and rejected.
    """
    absolute, portable_path = _artifact_path(path, relative_to=relative_to)
    before = _regular_file_metadata(absolute)
    digest, bytes_read = _hash_file(absolute)
    after = _regular_file_metadata(absolute)

    if _file_state(before) != _file_state(after) or bytes_read != after.st_size:
        raise OSError(
            errno.EBUSY,
            "file changed while it was being identified",
            os.fspath(absolute),
        )

    return FileArtifact(
        path=portable_path,
        sha256=digest,
        size_bytes=bytes_read,
    )


def identify_file_set(
    paths: Iterable[Path],
    *,
    relative_to: Path,
    profile: str,
) -> FileSetIdentity:
    """Identify an explicitly supplied set of files in deterministic order."""
    if isinstance(paths, (str, Path)):
        raise TypeError("paths must be an iterable, not one path")
    if not isinstance(profile, str):
        raise TypeError("profile must be a string")
    if not profile:
        raise ValueError("profile must not be empty")

    artifacts_by_path: dict[str, FileArtifact] = {}
    for path in paths:
        artifact = identify_file(path, relative_to=relative_to)
        if artifact.path in artifacts_by_path:
            raise ValueError(f"duplicate file-set path: {artifact.path}")
        artifacts_by_path[artifact.path] = artifact

    files = tuple(
        artifacts_by_path[path]
        for path in sorted(artifacts_by_path, key=os.fsencode)
    )
    digest = hashlib.sha256()
    _update_digest(
        digest,
        b"format",
        _FILE_SET_HASH_SCHEME.encode("ascii"),
    )
    _update_digest(digest, b"profile", profile.encode("utf-8"))
    _update_digest(digest, b"file-count", len(files).to_bytes(8, "big"))
    for artifact in files:
        _update_digest(digest, b"path", os.fsencode(artifact.path))
        _update_digest(
            digest,
            b"sha256",
            bytes.fromhex(artifact.sha256),
        )
        _update_digest(
            digest,
            b"size",
            artifact.size_bytes.to_bytes(8, "big"),
        )

    return FileSetIdentity(
        hash_scheme=_FILE_SET_HASH_SCHEME,
        profile=profile,
        sha256=digest.hexdigest(),
        files=files,
    )


def capture_code_provenance(
    repo_root: Path,
    *,
    scope: Iterable[Path | str],
) -> CodeProvenance:
    """Capture scoped Git provenance without requiring a clean repository.

    Invalid scope specifications are caller errors and raise. Git, repository,
    timeout, and scoped-file failures are returned as observational capture
    errors instead.
    """
    root = _absolute_path(repo_root)
    normalized_scope = _normalize_scope(root, scope)
    commit: str | None = None

    try:
        reported_root = _git_repository_root(root)
        if not os.path.samefile(root, reported_root):
            raise RuntimeError(f"not the Git repository root: {root}")

        commit = _git_commit(root)
    except Exception as error:
        return CodeProvenance(
            hash_scheme=_WORKSPACE_HASH_SCHEME,
            scope=normalized_scope,
            git_commit=None,
            git_dirty=None,
            workspace_sha256=None,
            capture_error=_concise_error(error),
        )

    try:
        tracked_diff = _git_tracked_diff(root, commit, normalized_scope)
        tracked_status = _git_tracked_status(root, normalized_scope)
        untracked_paths = _git_untracked_paths(root, normalized_scope)
        untracked_files = _identify_untracked_files(root, untracked_paths)
        workspace_hash = _workspace_hash(
            commit=commit,
            scope=normalized_scope,
            tracked_diff=tracked_diff,
            untracked_files=untracked_files,
        )
    except Exception as error:
        return CodeProvenance(
            hash_scheme=_WORKSPACE_HASH_SCHEME,
            scope=normalized_scope,
            git_commit=commit,
            git_dirty=None,
            workspace_sha256=None,
            capture_error=_concise_error(error),
        )

    return CodeProvenance(
        hash_scheme=_WORKSPACE_HASH_SCHEME,
        scope=normalized_scope,
        git_commit=commit,
        git_dirty=bool(tracked_status or untracked_paths),
        workspace_sha256=workspace_hash,
        capture_error=None,
    )


def capture_environment(*, include_gpu: bool = False) -> EnvironmentInfo:
    """Capture package metadata and, when requested, lazy Torch GPU facts."""
    versions = {
        field: _distribution_version(distribution)
        for field, distribution in _DISTRIBUTIONS.items()
    }
    torch_version = versions["torch"]
    torch_cuda: str | None = None
    gpu: str | None = None

    if include_gpu:
        torch_version, torch_cuda, gpu = _capture_torch_runtime(torch_version)

    return EnvironmentInfo(
        python=" ".join(sys.version.splitlines()),
        platform=platform_module.platform(),
        machine=platform_module.machine(),
        torch=torch_version,
        torch_cuda=torch_cuda,
        mmengine=versions["mmengine"],
        mmcv=versions["mmcv"],
        mmdet=versions["mmdet"],
        mmdet3d=versions["mmdet3d"],
        gpu=gpu,
    )


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(_FILE_BLOCK_SIZE)
            if not block:
                break
            digest.update(block)
            size_bytes += len(block)
    return digest.hexdigest(), size_bytes


def _artifact_path(
    path: Path,
    *,
    relative_to: Path | None,
) -> tuple[Path, str]:
    supplied = Path(path)
    if relative_to is None:
        absolute = _absolute_path(supplied)
        _reject_symlink_path(absolute)
        return absolute, absolute.as_posix()

    root = _absolute_path(relative_to)
    _reject_symlink_path(root)
    _require_real_directory(root, description="relative_to")
    absolute = _absolute_path(
        supplied if supplied.is_absolute() else root / supplied
    )
    try:
        relative = absolute.relative_to(root)
    except ValueError as error:
        raise ValueError(f"file is outside relative_to: {path}") from error

    _reject_symlink_components(root, relative)
    return absolute, relative.as_posix()


def _regular_file_metadata(path: Path) -> os.stat_result:
    file_stat = path.lstat()
    if stat.S_ISLNK(file_stat.st_mode):
        raise ValueError(f"file artifact must not be a symlink: {path}")
    if stat.S_ISDIR(file_stat.st_mode):
        raise IsADirectoryError(
            errno.EISDIR,
            "file artifact path is a directory",
            os.fspath(path),
        )
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"file artifact must be a regular file: {path}")
    return file_stat


def _file_state(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _require_real_directory(path: Path, *, description: str) -> None:
    directory_stat = path.lstat()
    if stat.S_ISLNK(directory_stat.st_mode):
        raise ValueError(f"{description} must not be a symlink: {path}")
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise NotADirectoryError(
            errno.ENOTDIR,
            f"{description} is not a directory",
            os.fspath(path),
        )


def _reject_symlink_components(root: Path, relative: Path) -> None:
    current = root
    for part in relative.parts:
        current /= part
        if stat.S_ISLNK(current.lstat().st_mode):
            raise ValueError(f"file artifact path contains a symlink: {current}")


def _reject_symlink_path(path: Path) -> None:
    anchor = Path(path.anchor)
    relative = path.relative_to(anchor)
    _reject_symlink_components(anchor, relative)


def _normalize_scope(
    root: Path,
    scope: Iterable[Path | str],
) -> tuple[str, ...]:
    if isinstance(scope, (str, Path)):
        raise TypeError("scope must be an iterable of paths, not one path")

    normalized: set[str] = set()
    for entry in scope:
        if not isinstance(entry, (str, Path)):
            raise TypeError("scope entries must be strings or Path objects")
        if isinstance(entry, str) and not entry:
            raise ValueError("scope entries must not be empty")
        if "\x00" in os.fspath(entry):
            raise ValueError("scope entries must not contain NUL")

        supplied = Path(entry)
        absolute = _absolute_path(
            supplied if supplied.is_absolute() else root / supplied
        )
        try:
            relative = absolute.relative_to(root)
        except ValueError as error:
            raise ValueError(f"scope entry is outside repository: {entry}") from error
        normalized.add(relative.as_posix())

    if not normalized:
        raise ValueError("scope must contain at least one path")
    return tuple(sorted(normalized, key=os.fsencode))


def _git_repository_root(root: Path) -> Path:
    output = _run_git(root, ["rev-parse", "--show-toplevel"])
    value = output.rstrip(b"\n")
    if not value:
        raise RuntimeError("Git returned an empty repository root")
    return Path(os.fsdecode(value))


def _git_commit(root: Path) -> str:
    output = _run_git(root, ["rev-parse", "--verify", "HEAD^{commit}"])
    try:
        commit = output.strip().decode("ascii")
    except UnicodeDecodeError as error:
        raise RuntimeError("Git returned a non-ASCII HEAD object ID") from error
    if len(commit) not in (40, 64) or any(
        character not in "0123456789abcdefABCDEF" for character in commit
    ):
        raise RuntimeError("Git returned an invalid HEAD object ID")
    return commit.lower()


def _git_tracked_diff(
    root: Path,
    commit: str,
    scope: tuple[str, ...],
) -> bytes:
    return _run_git(
        root,
        [
            "diff",
            "--binary",
            "--full-index",
            "--no-color",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            "--diff-algorithm=myers",
            "--no-indent-heuristic",
            "--submodule=short",
            "--unified=3",
            "--src-prefix=a/",
            "--dst-prefix=b/",
            "-O",
            os.devnull,
            "--ignore-submodules=none",
            commit,
            "--",
            *scope,
        ],
    )


def _git_tracked_status(root: Path, scope: tuple[str, ...]) -> bytes:
    return _run_git(
        root,
        [
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=no",
            "--ignored=no",
            "--no-renames",
            "--ignore-submodules=none",
            "--",
            *scope,
        ],
    )


def _git_untracked_paths(root: Path, scope: tuple[str, ...]) -> tuple[bytes, ...]:
    output = _run_git(
        root,
        [
            "ls-files",
            "--others",
            "--exclude-standard",
            "--full-name",
            "-z",
            "--",
            *scope,
        ],
    )
    if not output:
        return ()
    if not output.endswith(b"\x00"):
        raise RuntimeError("Git returned malformed untracked-file output")

    paths = output[:-1].split(b"\x00")
    if any(not path for path in paths) or len(paths) != len(set(paths)):
        raise RuntimeError("Git returned invalid untracked-file paths")
    return tuple(sorted(paths))


def _identify_untracked_files(
    root: Path,
    paths: tuple[bytes, ...],
) -> tuple[FileArtifact, ...]:
    artifacts: list[FileArtifact] = []
    for raw_path in paths:
        relative_text = os.fsdecode(raw_path)
        relative_path = Path(relative_text)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError("Git returned an unsafe untracked-file path")
        artifacts.append(
            identify_file(root / relative_path, relative_to=root)
        )
    return tuple(sorted(artifacts, key=lambda artifact: os.fsencode(artifact.path)))


def _workspace_hash(
    *,
    commit: str,
    scope: tuple[str, ...],
    tracked_diff: bytes,
    untracked_files: tuple[FileArtifact, ...],
) -> str:
    digest = hashlib.sha256()
    _update_digest(
        digest,
        b"format",
        _WORKSPACE_HASH_SCHEME.encode("ascii"),
    )
    _update_digest(digest, b"scope-count", len(scope).to_bytes(8, "big"))
    for path in scope:
        _update_digest(digest, b"scope", os.fsencode(path))
    _update_digest(digest, b"head", commit.encode("ascii"))
    _update_digest(digest, b"tracked-diff", tracked_diff)
    _update_digest(
        digest,
        b"untracked-count",
        len(untracked_files).to_bytes(8, "big"),
    )
    for artifact in untracked_files:
        _update_digest(digest, b"untracked-path", os.fsencode(artifact.path))
        _update_digest(
            digest,
            b"untracked-sha256",
            bytes.fromhex(artifact.sha256),
        )
        _update_digest(
            digest,
            b"untracked-size",
            artifact.size_bytes.to_bytes(8, "big"),
        )
    return digest.hexdigest()


def _update_digest(digest: Any, label: bytes, value: bytes) -> None:
    digest.update(len(label).to_bytes(4, "big"))
    digest.update(label)
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _run_git(root: Path, arguments: list[str]) -> bytes:
    command = [
        "git",
        "--no-pager",
        "--no-optional-locks",
        "-c",
        "core.quotePath=true",
        "-C",
        os.fspath(root),
        "--literal-pathspecs",
        *arguments,
    ]
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    environment["LC_ALL"] = "C"
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_GIT_TIMEOUT_SECONDS,
            env=environment,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Git executable is unavailable") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"Git command timed out after {_GIT_TIMEOUT_SECONDS} seconds"
        ) from error
    except OSError as error:
        raise RuntimeError(f"could not execute Git: {error}") from error

    if completed.returncode != 0:
        detail = " ".join(os.fsdecode(completed.stderr).split())
        if not detail:
            detail = f"exit status {completed.returncode}"
        raise RuntimeError(f"Git {arguments[0]} failed: {detail[:300]}")
    return completed.stdout


def _concise_error(error: Exception) -> str:
    message = " ".join(str(error).split())
    return (message or type(error).__name__)[:500]


def _distribution_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _capture_torch_runtime(
    installed_version: str | None,
) -> tuple[str | None, str | None, str | None]:
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return installed_version, None, None

    runtime_version = _optional_string(getattr(torch, "__version__", None))
    torch_version = runtime_version or installed_version
    version_namespace = getattr(torch, "version", None)
    torch_cuda = _optional_string(getattr(version_namespace, "cuda", None))
    gpu: str | None = None

    try:
        cuda = torch.cuda
        if cuda.is_available() and cuda.device_count() > 0:
            gpu = _optional_string(cuda.get_device_name(cuda.current_device()))
    except Exception:
        gpu = None

    return torch_version, torch_cuda, gpu


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))
