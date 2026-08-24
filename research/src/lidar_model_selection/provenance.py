"""Deterministic file, workspace, and software provenance evidence."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


__all__ = (
    "FILE_SET_SCHEME",
    "WORKSPACE_SCHEME",
    "TRAINING_COMPATIBILITY_PROFILE",
    "TRAINING_COMPATIBILITY_VERSION",
    "FileArtifact",
    "FileSetIdentity",
    "CodeProvenance",
    "EnvironmentInfo",
    "TrainingCompatibilityIdentity",
    "sha256_file",
    "identify_file",
    "identify_file_set",
    "capture_code_provenance",
    "capture_environment",
    "build_training_compatibility",
)

FILE_SET_SCHEME = "lidar-file-set-v1"
WORKSPACE_SCHEME = "lidar-workspace-v1"
TRAINING_COMPATIBILITY_PROFILE = "lidar-training"
TRAINING_COMPATIBILITY_VERSION = 1

_DEFAULT_HASH_CHUNK_SIZE = 1024 * 1024
_DEFAULT_CORE_PACKAGES = (
    "torch",
    "mmengine",
    "mmcv",
    "mmdet",
    "mmdet3d",
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_PACKAGE_SEPARATOR_PATTERN = re.compile(r"[-_.]+")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, *, description: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")
    return value


def _require_string(
    value: object,
    *,
    description: str,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{description} must be a string")
    if not allow_empty and not value:
        raise ValueError(f"{description} must not be empty")
    return value


def _require_optional_string(
    value: object,
    *,
    description: str,
) -> str | None:
    if value is None:
        return None
    return _require_string(value, description=description)


def _require_mapping(value: object, *, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{description} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{description} keys must be strings")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    keys: set[str],
    *,
    description: str,
) -> None:
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise ValueError(
            f"invalid {description} keys; missing={missing}, extra={extra}"
        )


def _normalized_stored_path(value: object, *, description: str) -> str:
    path = _require_string(value, description=description)
    pure_path = PurePosixPath(path)
    if pure_path.is_absolute() or path != pure_path.as_posix():
        raise ValueError(f"{description} must be a normalized relative POSIX path")
    if path == "." or any(part in {"", ".", ".."} for part in pure_path.parts):
        raise ValueError(f"{description} must stay below its identity root")
    return path


def sha256_file(
    path: Path | str,
    *,
    chunk_size: int = _DEFAULT_HASH_CHUNK_SIZE,
) -> str:
    """Return the SHA-256 of *path* without loading the file into memory."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FileArtifact:
    """Content identity and stable root-relative name of one regular file."""

    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _normalized_stored_path(self.path, description="artifact path")
        _require_sha256(self.sha256, description="artifact sha256")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ValueError("artifact size_bytes must be a non-negative integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FileArtifact:
        data = _require_mapping(value, description="file artifact")
        _require_exact_keys(
            data,
            {"path", "sha256", "size_bytes"},
            description="file artifact",
        )
        return cls(
            path=data["path"],  # type: ignore[arg-type]
            sha256=data["sha256"],  # type: ignore[arg-type]
            size_bytes=data["size_bytes"],  # type: ignore[arg-type]
        )


def _file_set_payload(files: Sequence[FileArtifact]) -> dict[str, object]:
    return {
        "scheme": FILE_SET_SCHEME,
        "files": [artifact.to_dict() for artifact in files],
    }


@dataclass(frozen=True)
class FileSetIdentity:
    """Canonical identity of an explicitly enumerated set of files."""

    scheme: str
    identity_sha256: str
    files: tuple[FileArtifact, ...]

    def __post_init__(self) -> None:
        if self.scheme != FILE_SET_SCHEME:
            raise ValueError(f"unsupported file-set scheme: {self.scheme!r}")
        _require_sha256(
            self.identity_sha256,
            description="file-set identity_sha256",
        )
        if not isinstance(self.files, tuple) or not all(
            isinstance(artifact, FileArtifact) for artifact in self.files
        ):
            raise TypeError("file-set files must be a tuple of FileArtifact values")
        paths = tuple(artifact.path for artifact in self.files)
        if paths != tuple(sorted(paths)):
            raise ValueError("file-set files must be sorted by path")
        if len(paths) != len(set(paths)):
            raise ValueError("file-set files must have unique paths")
        expected = _canonical_sha256(_file_set_payload(self.files))
        if self.identity_sha256 != expected:
            raise ValueError("file-set identity_sha256 does not match its files")

    def to_dict(self) -> dict[str, object]:
        return {
            "scheme": self.scheme,
            "identity_sha256": self.identity_sha256,
            "files": [artifact.to_dict() for artifact in self.files],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FileSetIdentity:
        data = _require_mapping(value, description="file-set identity")
        _require_exact_keys(
            data,
            {"scheme", "identity_sha256", "files"},
            description="file-set identity",
        )
        files_value = data["files"]
        if not isinstance(files_value, list):
            raise TypeError("file-set files must be a list")
        return cls(
            scheme=data["scheme"],  # type: ignore[arg-type]
            identity_sha256=data["identity_sha256"],  # type: ignore[arg-type]
            files=tuple(FileArtifact.from_dict(item) for item in files_value),
        )


def _absolute_path(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _relative_artifact_path(path: Path, root: Path | None) -> str:
    if root is None:
        normalized = Path(os.path.normpath(os.fspath(path)))
        if normalized.is_absolute():
            stored = normalized.name
        else:
            stored = normalized.as_posix()
    else:
        absolute_root = _absolute_path(root)
        absolute_path = _absolute_path(path)
        try:
            relative = absolute_path.relative_to(absolute_root)
        except ValueError as error:
            raise ValueError(f"file is outside identity root: {path}") from error
        stored = relative.as_posix()
    return _normalized_stored_path(stored, description="artifact path")


def identify_file(
    path: Path | str,
    root: Path | str | None = None,
) -> FileArtifact:
    """Identify one regular file, optionally naming it relative to *root*."""
    file_path = Path(path)
    metadata = file_path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"artifact is not a regular file: {file_path}")

    stored_path = _relative_artifact_path(
        file_path,
        None if root is None else Path(root),
    )
    with file_path.open("rb") as stream:
        opened_metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened_metadata.st_mode):
            raise ValueError(f"artifact is not a regular file: {file_path}")
        digest = hashlib.sha256()
        while True:
            block = stream.read(_DEFAULT_HASH_CHUNK_SIZE)
            if not block:
                break
            digest.update(block)

    return FileArtifact(
        path=stored_path,
        sha256=digest.hexdigest(),
        size_bytes=opened_metadata.st_size,
    )


def _new_file_set(files: Iterable[FileArtifact]) -> FileSetIdentity:
    ordered = tuple(sorted(files, key=lambda artifact: artifact.path))
    paths = tuple(artifact.path for artifact in ordered)
    if len(paths) != len(set(paths)):
        raise ValueError("explicit file set contains duplicate paths")
    return FileSetIdentity(
        scheme=FILE_SET_SCHEME,
        identity_sha256=_canonical_sha256(_file_set_payload(ordered)),
        files=ordered,
    )


def identify_file_set(
    root: Path | str,
    paths: Iterable[Path | str],
) -> FileSetIdentity:
    """Identify exactly *paths*, using deterministic names relative to *root*."""
    if isinstance(paths, (str, bytes, os.PathLike)):
        raise TypeError("paths must be an iterable of paths, not one path")
    root_path = _absolute_path(root)
    artifacts = []
    for path in paths:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = root_path / candidate
        artifacts.append(identify_file(candidate, root=root_path))
    return _new_file_set(artifacts)


def _workspace_payload(
    *,
    git_commit: str | None,
    git_branch: str | None,
    scopes: Sequence[str],
    tracked_files: FileSetIdentity,
    untracked_files: FileSetIdentity,
    missing_tracked_files: Sequence[str],
    dirty: bool,
) -> dict[str, object]:
    return {
        "scheme": WORKSPACE_SCHEME,
        "git_commit": git_commit,
        "git_branch": git_branch,
        "scopes": list(scopes),
        "tracked_files": tracked_files.to_dict(),
        "untracked_files": untracked_files.to_dict(),
        "missing_tracked_files": list(missing_tracked_files),
        "dirty": dirty,
    }


@dataclass(frozen=True)
class CodeProvenance:
    """Observational identity of selected paths in a Git working tree."""

    scheme: str
    workspace_sha256: str
    git_commit: str | None
    git_branch: str | None
    scopes: tuple[str, ...]
    tracked_files: FileSetIdentity
    untracked_files: FileSetIdentity
    missing_tracked_files: tuple[str, ...]
    dirty: bool

    def __post_init__(self) -> None:
        if self.scheme != WORKSPACE_SCHEME:
            raise ValueError(f"unsupported workspace scheme: {self.scheme!r}")
        _require_sha256(
            self.workspace_sha256,
            description="workspace workspace_sha256",
        )
        if self.git_commit is not None:
            commit = _require_string(
                self.git_commit,
                description="workspace git_commit",
            )
            if (
                len(commit) not in {40, 64}
                or any(character not in "0123456789abcdef" for character in commit)
            ):
                raise ValueError("workspace git_commit is not a canonical object ID")
        _require_optional_string(
            self.git_branch,
            description="workspace git_branch",
        )
        if not isinstance(self.scopes, tuple) or not self.scopes:
            raise ValueError("workspace scopes must be a non-empty tuple")
        if self.scopes != tuple(sorted(set(self.scopes))):
            raise ValueError("workspace scopes must be sorted and unique")
        for scope in self.scopes:
            if scope != ".":
                _normalized_stored_path(scope, description="workspace scope")
        if not isinstance(self.tracked_files, FileSetIdentity):
            raise TypeError("workspace tracked_files must be a FileSetIdentity")
        if not isinstance(self.untracked_files, FileSetIdentity):
            raise TypeError("workspace untracked_files must be a FileSetIdentity")
        if not isinstance(self.missing_tracked_files, tuple):
            raise TypeError("workspace missing_tracked_files must be a tuple")
        if self.missing_tracked_files != tuple(
            sorted(set(self.missing_tracked_files))
        ):
            raise ValueError("missing tracked paths must be sorted and unique")
        for path in self.missing_tracked_files:
            _normalized_stored_path(path, description="missing tracked path")
        tracked_paths = {artifact.path for artifact in self.tracked_files.files}
        untracked_paths = {
            artifact.path for artifact in self.untracked_files.files
        }
        missing_paths = set(self.missing_tracked_files)
        if tracked_paths & untracked_paths or tracked_paths & missing_paths:
            raise ValueError("workspace path classifications overlap")
        if not isinstance(self.dirty, bool):
            raise TypeError("workspace dirty must be a bool")
        expected = _canonical_sha256(
            _workspace_payload(
                git_commit=self.git_commit,
                git_branch=self.git_branch,
                scopes=self.scopes,
                tracked_files=self.tracked_files,
                untracked_files=self.untracked_files,
                missing_tracked_files=self.missing_tracked_files,
                dirty=self.dirty,
            )
        )
        if self.workspace_sha256 != expected:
            raise ValueError("workspace_sha256 does not match workspace evidence")

    def to_dict(self) -> dict[str, object]:
        result = _workspace_payload(
            git_commit=self.git_commit,
            git_branch=self.git_branch,
            scopes=self.scopes,
            tracked_files=self.tracked_files,
            untracked_files=self.untracked_files,
            missing_tracked_files=self.missing_tracked_files,
            dirty=self.dirty,
        )
        result["workspace_sha256"] = self.workspace_sha256
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CodeProvenance:
        data = _require_mapping(value, description="code provenance")
        _require_exact_keys(
            data,
            {
                "scheme",
                "workspace_sha256",
                "git_commit",
                "git_branch",
                "scopes",
                "tracked_files",
                "untracked_files",
                "missing_tracked_files",
                "dirty",
            },
            description="code provenance",
        )
        scopes = data["scopes"]
        missing = data["missing_tracked_files"]
        if not isinstance(scopes, list) or not all(
            isinstance(scope, str) for scope in scopes
        ):
            raise TypeError("code provenance scopes must be a list of strings")
        if not isinstance(missing, list) or not all(
            isinstance(path, str) for path in missing
        ):
            raise TypeError(
                "code provenance missing_tracked_files must be a list of strings"
            )
        return cls(
            scheme=data["scheme"],  # type: ignore[arg-type]
            workspace_sha256=data["workspace_sha256"],  # type: ignore[arg-type]
            git_commit=data["git_commit"],  # type: ignore[arg-type]
            git_branch=data["git_branch"],  # type: ignore[arg-type]
            scopes=tuple(scopes),
            tracked_files=FileSetIdentity.from_dict(
                _require_mapping(
                    data["tracked_files"],
                    description="tracked file-set identity",
                )
            ),
            untracked_files=FileSetIdentity.from_dict(
                _require_mapping(
                    data["untracked_files"],
                    description="untracked file-set identity",
                )
            ),
            missing_tracked_files=tuple(missing),
            dirty=data["dirty"],  # type: ignore[arg-type]
        )


def _git(
    repository: Path,
    arguments: Sequence[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.run(
        ["git", "-C", os.fspath(repository), *arguments],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )


def _git_text(repository: Path, arguments: Sequence[str]) -> str:
    return os.fsdecode(_git(repository, arguments).stdout).strip()


def _git_nul_paths(repository: Path, arguments: Sequence[str]) -> tuple[str, ...]:
    output = _git(repository, arguments).stdout
    if not output:
        return ()
    if not output.endswith(b"\0"):
        raise RuntimeError("Git returned a malformed NUL-delimited path list")
    paths = tuple(os.fsdecode(item) for item in output[:-1].split(b"\0"))
    for path in paths:
        _normalized_stored_path(path, description="Git path")
    return tuple(sorted(set(paths)))


def _normalize_scopes(
    repository: Path,
    scopes: Iterable[Path | str],
) -> tuple[str, ...]:
    if isinstance(scopes, (str, bytes, os.PathLike)):
        raise TypeError("scopes must be an iterable of paths, not one path")
    normalized = set()
    for scope in scopes:
        candidate = Path(scope)
        if not candidate.is_absolute():
            candidate = repository / candidate
        absolute = _absolute_path(candidate)
        try:
            relative = absolute.relative_to(repository)
        except ValueError as error:
            raise ValueError(f"workspace scope is outside repository: {scope}") from error
        stored = relative.as_posix()
        if stored != ".":
            _normalized_stored_path(stored, description="workspace scope")
        normalized.add(stored)
    if not normalized:
        raise ValueError("at least one explicit workspace scope is required")
    return tuple(sorted(normalized))


def _git_pathspecs(scopes: Sequence[str]) -> list[str]:
    return [":(top)" if scope == "." else f":(top,literal){scope}" for scope in scopes]


def _identify_git_paths(
    repository: Path,
    paths: Iterable[str],
) -> FileSetIdentity:
    return _new_file_set(
        identify_file(repository / PurePosixPath(path), root=repository)
        for path in paths
    )


def capture_code_provenance(
    repo_root: Path | str,
    scopes: Iterable[Path | str],
) -> CodeProvenance:
    """Capture current tracked and untracked contents within explicit scopes."""
    repository = _absolute_path(repo_root)
    top_level = _absolute_path(
        _git_text(repository, ("rev-parse", "--show-toplevel"))
    )
    try:
        same_repository = os.path.samefile(repository, top_level)
    except OSError:
        same_repository = False
    if not same_repository:
        raise ValueError(f"repo_root is not the Git worktree root: {repo_root}")

    normalized_scopes = _normalize_scopes(repository, scopes)
    pathspecs = _git_pathspecs(normalized_scopes)
    tracked_paths = _git_nul_paths(
        repository,
        ("ls-files", "--cached", "-z", "--", *pathspecs),
    )
    untracked_paths = _git_nul_paths(
        repository,
        (
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            *pathspecs,
        ),
    )

    present_tracked = []
    missing_tracked = []
    for path in tracked_paths:
        candidate = repository / PurePosixPath(path)
        try:
            candidate.lstat()
        except FileNotFoundError:
            missing_tracked.append(path)
        else:
            present_tracked.append(path)

    head = _git(repository, ("rev-parse", "--verify", "HEAD"), check=False)
    if head.returncode == 0:
        git_commit: str | None = os.fsdecode(head.stdout).strip()
    else:
        git_commit = None
    branch_result = _git(
        repository,
        ("symbolic-ref", "--quiet", "--short", "HEAD"),
        check=False,
    )
    git_branch = (
        os.fsdecode(branch_result.stdout).strip()
        if branch_result.returncode == 0
        else None
    )
    status_result = _git(
        repository,
        (
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--",
            *pathspecs,
        ),
    )
    dirty = bool(status_result.stdout)
    tracked_files = _identify_git_paths(repository, present_tracked)
    untracked_files = _identify_git_paths(repository, untracked_paths)
    missing = tuple(sorted(missing_tracked))
    workspace_sha256 = _canonical_sha256(
        _workspace_payload(
            git_commit=git_commit,
            git_branch=git_branch,
            scopes=normalized_scopes,
            tracked_files=tracked_files,
            untracked_files=untracked_files,
            missing_tracked_files=missing,
            dirty=dirty,
        )
    )
    return CodeProvenance(
        scheme=WORKSPACE_SCHEME,
        workspace_sha256=workspace_sha256,
        git_commit=git_commit,
        git_branch=git_branch,
        scopes=normalized_scopes,
        tracked_files=tracked_files,
        untracked_files=untracked_files,
        missing_tracked_files=missing,
        dirty=dirty,
    )


def _canonical_package_name(value: object) -> str:
    name = _require_string(value, description="package name")
    canonical = _PACKAGE_SEPARATOR_PATTERN.sub("-", name).lower()
    if not canonical:
        raise ValueError("package name must not be empty")
    return canonical


def _normalize_packages(
    packages: Mapping[str, str | None],
) -> tuple[tuple[str, str | None], ...]:
    normalized: dict[str, str | None] = {}
    for raw_name, raw_version in packages.items():
        name = _canonical_package_name(raw_name)
        if raw_version is not None and not isinstance(raw_version, str):
            raise TypeError(f"version for package {name!r} must be a string or null")
        if name in normalized and normalized[name] != raw_version:
            raise ValueError(f"conflicting versions for package {name!r}")
        normalized[name] = raw_version
    return tuple(sorted(normalized.items()))


def _all_package_versions() -> tuple[tuple[str, str | None], ...]:
    versions: dict[str, set[str]] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            continue
        name = _canonical_package_name(raw_name)
        versions.setdefault(name, set()).add(distribution.version)
    flattened = {
        name: " | ".join(sorted(found_versions))
        for name, found_versions in versions.items()
    }
    return _normalize_packages(flattened)


def _selected_package_versions(
    names: Iterable[str],
) -> tuple[tuple[str, str | None], ...]:
    if isinstance(names, str):
        raise TypeError("package names must be an iterable, not one string")
    selected: dict[str, str | None] = {}
    for raw_name in names:
        name = _canonical_package_name(raw_name)
        try:
            version = importlib.metadata.version(raw_name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        if name in selected and selected[name] != version:
            raise ValueError(f"conflicting versions for package {name!r}")
        selected[name] = version
    return _normalize_packages(selected)


@dataclass(frozen=True)
class EnvironmentInfo:
    """Observed Python, package, and optionally Torch/GPU environment."""

    python_version: str
    python_implementation: str
    platform: str
    machine: str
    executable: str
    packages: tuple[tuple[str, str | None], ...]
    torch_version: str | None
    cuda_version: str | None
    cudnn_version: str | None
    gpu_available: bool | None
    gpu_devices: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_string(self.python_version, description="Python version")
        _require_string(
            self.python_implementation,
            description="Python implementation",
        )
        _require_string(self.platform, description="platform")
        _require_string(self.machine, description="machine", allow_empty=True)
        _require_string(self.executable, description="Python executable")
        if not isinstance(self.packages, tuple):
            raise TypeError("environment packages must be a tuple")
        if self.packages != _normalize_packages(dict(self.packages)):
            raise ValueError("environment packages must be canonical and unique")
        for value, description in (
            (self.torch_version, "Torch version"),
            (self.cuda_version, "CUDA version"),
            (self.cudnn_version, "cuDNN version"),
        ):
            _require_optional_string(value, description=description)
        if self.gpu_available is not None and not isinstance(
            self.gpu_available, bool
        ):
            raise TypeError("gpu_available must be bool or null")
        if not isinstance(self.gpu_devices, tuple) or not all(
            isinstance(name, str) and name for name in self.gpu_devices
        ):
            raise TypeError("gpu_devices must be a tuple of non-empty strings")
        if self.gpu_available is None and (
            self.torch_version is not None
            or self.cuda_version is not None
            or self.cudnn_version is not None
            or self.gpu_devices
        ):
            raise ValueError("Torch/GPU fields require an explicit GPU observation")
        if self.gpu_available is False and self.gpu_devices:
            raise ValueError("unavailable GPU observation cannot list devices")

    def to_dict(self) -> dict[str, object]:
        torch_info: dict[str, object] | None
        if self.gpu_available is None:
            torch_info = None
        else:
            torch_info = {
                "torch_version": self.torch_version,
                "cuda_version": self.cuda_version,
                "cudnn_version": self.cudnn_version,
                "gpu_available": self.gpu_available,
                "gpu_devices": list(self.gpu_devices),
            }
        return {
            "python_version": self.python_version,
            "python_implementation": self.python_implementation,
            "platform": self.platform,
            "machine": self.machine,
            "executable": self.executable,
            "packages": dict(self.packages),
            "torch": torch_info,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> EnvironmentInfo:
        data = _require_mapping(value, description="environment info")
        _require_exact_keys(
            data,
            {
                "python_version",
                "python_implementation",
                "platform",
                "machine",
                "executable",
                "packages",
                "torch",
            },
            description="environment info",
        )
        packages_data = _require_mapping(
            data["packages"],
            description="environment packages",
        )
        packages = _normalize_packages(packages_data)  # type: ignore[arg-type]
        torch_data = data["torch"]
        if torch_data is None:
            torch_version = None
            cuda_version = None
            cudnn_version = None
            gpu_available = None
            gpu_devices: tuple[str, ...] = ()
        else:
            torch_mapping = _require_mapping(
                torch_data,
                description="Torch environment",
            )
            _require_exact_keys(
                torch_mapping,
                {
                    "torch_version",
                    "cuda_version",
                    "cudnn_version",
                    "gpu_available",
                    "gpu_devices",
                },
                description="Torch environment",
            )
            devices = torch_mapping["gpu_devices"]
            if not isinstance(devices, list) or not all(
                isinstance(device, str) for device in devices
            ):
                raise TypeError("gpu_devices must be a list of strings")
            torch_version = torch_mapping["torch_version"]
            cuda_version = torch_mapping["cuda_version"]
            cudnn_version = torch_mapping["cudnn_version"]
            gpu_available = torch_mapping["gpu_available"]
            gpu_devices = tuple(devices)
        return cls(
            python_version=data["python_version"],  # type: ignore[arg-type]
            python_implementation=data["python_implementation"],  # type: ignore[arg-type]
            platform=data["platform"],  # type: ignore[arg-type]
            machine=data["machine"],  # type: ignore[arg-type]
            executable=data["executable"],  # type: ignore[arg-type]
            packages=packages,
            torch_version=torch_version,  # type: ignore[arg-type]
            cuda_version=cuda_version,  # type: ignore[arg-type]
            cudnn_version=cudnn_version,  # type: ignore[arg-type]
            gpu_available=gpu_available,  # type: ignore[arg-type]
            gpu_devices=gpu_devices,
        )


def _torch_environment(
) -> tuple[str | None, str | None, str | None, bool, tuple[str, ...]]:
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        return None, None, None, False, ()

    raw_torch_version = getattr(torch, "__version__", None)
    torch_version = (
        str(raw_torch_version) if raw_torch_version is not None else None
    )
    version_namespace = getattr(torch, "version", None)
    raw_cuda_version = getattr(version_namespace, "cuda", None)
    cuda_version = (
        str(raw_cuda_version) if raw_cuda_version is not None else None
    )
    backends = getattr(torch, "backends", None)
    cudnn = getattr(backends, "cudnn", None)
    cudnn_version_function = getattr(cudnn, "version", None)
    raw_cudnn_version = (
        cudnn_version_function() if callable(cudnn_version_function) else None
    )
    cudnn_version = (
        str(raw_cudnn_version) if raw_cudnn_version is not None else None
    )
    cuda = getattr(torch, "cuda")
    available = bool(cuda.is_available())
    devices = (
        tuple(str(cuda.get_device_name(index)) for index in range(cuda.device_count()))
        if available
        else ()
    )
    return torch_version, cuda_version, cudnn_version, available, devices


def capture_environment(
    *,
    include_packages: bool = True,
    include_torch: bool = False,
    package_names: Iterable[str] | None = None,
) -> EnvironmentInfo:
    """Capture lightweight runtime evidence; import Torch only when requested."""
    if not isinstance(include_packages, bool):
        raise TypeError("include_packages must be a bool")
    if not isinstance(include_torch, bool):
        raise TypeError("include_torch must be a bool")
    if package_names is not None and not include_packages:
        raise ValueError("package_names requires include_packages=True")
    if include_packages:
        packages = (
            _all_package_versions()
            if package_names is None
            else _selected_package_versions(package_names)
        )
    else:
        packages = ()
    if include_torch:
        (
            torch_version,
            cuda_version,
            cudnn_version,
            gpu_available,
            gpu_devices,
        ) = _torch_environment()
    else:
        torch_version = None
        cuda_version = None
        cudnn_version = None
        gpu_available = None
        gpu_devices = ()
    return EnvironmentInfo(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform=platform.platform(),
        machine=platform.machine(),
        executable=sys.executable,
        packages=packages,
        torch_version=torch_version,
        cuda_version=cuda_version,
        cudnn_version=cudnn_version,
        gpu_available=gpu_available,
        gpu_devices=gpu_devices,
    )


def _compatibility_payload(
    *,
    config_sha256: str,
    dataset_sha256: str,
    training_sources: FileSetIdentity,
    python_version: str,
    core_packages: Sequence[tuple[str, str | None]],
) -> dict[str, object]:
    return {
        "profile": TRAINING_COMPATIBILITY_PROFILE,
        "version": TRAINING_COMPATIBILITY_VERSION,
        "config_sha256": config_sha256,
        "dataset_sha256": dataset_sha256,
        "training_sources": {
            "scheme": training_sources.scheme,
            "identity_sha256": training_sources.identity_sha256,
        },
        "python_version": python_version,
        "core_packages": dict(core_packages),
    }


@dataclass(frozen=True)
class TrainingCompatibilityIdentity:
    """Narrow, versioned evidence governing training continuation safety."""

    profile: str
    version: int
    compatibility_sha256: str
    config_sha256: str
    dataset_sha256: str
    training_sources: FileSetIdentity
    python_version: str
    core_packages: tuple[tuple[str, str | None], ...]

    def __post_init__(self) -> None:
        if self.profile != TRAINING_COMPATIBILITY_PROFILE:
            raise ValueError(
                f"unsupported training compatibility profile: {self.profile!r}"
            )
        if (
            isinstance(self.version, bool)
            or self.version != TRAINING_COMPATIBILITY_VERSION
        ):
            raise ValueError(
                f"unsupported training compatibility version: {self.version!r}"
            )
        _require_sha256(
            self.compatibility_sha256,
            description="training compatibility_sha256",
        )
        _require_sha256(self.config_sha256, description="canonical config sha256")
        _require_sha256(self.dataset_sha256, description="dataset identity sha256")
        if not isinstance(self.training_sources, FileSetIdentity):
            raise TypeError("training_sources must be a FileSetIdentity")
        _require_string(self.python_version, description="Python version")
        if not isinstance(self.core_packages, tuple):
            raise TypeError("core_packages must be a tuple")
        if self.core_packages != _normalize_packages(dict(self.core_packages)):
            raise ValueError("core_packages must be canonical and unique")
        expected = _canonical_sha256(
            _compatibility_payload(
                config_sha256=self.config_sha256,
                dataset_sha256=self.dataset_sha256,
                training_sources=self.training_sources,
                python_version=self.python_version,
                core_packages=self.core_packages,
            )
        )
        if self.compatibility_sha256 != expected:
            raise ValueError(
                "compatibility_sha256 does not match compatibility evidence"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "version": self.version,
            "compatibility_sha256": self.compatibility_sha256,
            "config_sha256": self.config_sha256,
            "dataset_sha256": self.dataset_sha256,
            "training_sources": self.training_sources.to_dict(),
            "python_version": self.python_version,
            "core_packages": dict(self.core_packages),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> TrainingCompatibilityIdentity:
        data = _require_mapping(value, description="training compatibility")
        _require_exact_keys(
            data,
            {
                "profile",
                "version",
                "compatibility_sha256",
                "config_sha256",
                "dataset_sha256",
                "training_sources",
                "python_version",
                "core_packages",
            },
            description="training compatibility",
        )
        packages_data = _require_mapping(
            data["core_packages"],
            description="training compatibility core_packages",
        )
        return cls(
            profile=data["profile"],  # type: ignore[arg-type]
            version=data["version"],  # type: ignore[arg-type]
            compatibility_sha256=data["compatibility_sha256"],  # type: ignore[arg-type]
            config_sha256=data["config_sha256"],  # type: ignore[arg-type]
            dataset_sha256=data["dataset_sha256"],  # type: ignore[arg-type]
            training_sources=FileSetIdentity.from_dict(
                _require_mapping(
                    data["training_sources"],
                    description="training source identity",
                )
            ),
            python_version=data["python_version"],  # type: ignore[arg-type]
            core_packages=_normalize_packages(packages_data),  # type: ignore[arg-type]
        )


def build_training_compatibility(
    config_sha256: str,
    dataset_sha256: str,
    training_sources: FileSetIdentity,
    *,
    core_packages: Mapping[str, str | None] | Iterable[str] | None = None,
    python_version: str | None = None,
) -> TrainingCompatibilityIdentity:
    """Build resume compatibility without using full workspace provenance."""
    _require_sha256(config_sha256, description="canonical config sha256")
    _require_sha256(dataset_sha256, description="dataset identity sha256")
    if not isinstance(training_sources, FileSetIdentity):
        raise TypeError("training_sources must be a FileSetIdentity")
    if core_packages is None:
        normalized_packages = _selected_package_versions(_DEFAULT_CORE_PACKAGES)
    elif isinstance(core_packages, Mapping):
        normalized_packages = _normalize_packages(core_packages)
    else:
        if isinstance(core_packages, str):
            raise TypeError(
                "core_packages must be a mapping or an iterable of package names"
            )
        normalized_packages = _selected_package_versions(core_packages)
    observed_python_version = (
        platform.python_version() if python_version is None else python_version
    )
    _require_string(observed_python_version, description="Python version")
    payload = _compatibility_payload(
        config_sha256=config_sha256,
        dataset_sha256=dataset_sha256,
        training_sources=training_sources,
        python_version=observed_python_version,
        core_packages=normalized_packages,
    )
    return TrainingCompatibilityIdentity(
        profile=TRAINING_COMPATIBILITY_PROFILE,
        version=TRAINING_COMPATIBILITY_VERSION,
        compatibility_sha256=_canonical_sha256(payload),
        config_sha256=config_sha256,
        dataset_sha256=dataset_sha256,
        training_sources=training_sources,
        python_version=observed_python_version,
        core_packages=normalized_packages,
    )
