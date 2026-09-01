"""Canonical run identity, layout, manifests, and transactional persistence."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .checkpoints import CheckpointArtifact, TrainingOutputs
from .provenance import (
    CodeProvenance,
    EnvironmentInfo,
    FileSetIdentity,
    TrainingCompatibilityIdentity,
)
from .storage import (
    cleanup_staging_directory,
    create_staging_directory,
    publish_directory_exclusive,
    read_json_object,
    write_json_atomic,
)


__all__ = (
    "RUN_SCHEMA_VERSION",
    "DATASET_IDENTITY_SCHEME",
    "RunPaths",
    "ConfigArtifact",
    "DatasetIdentity",
    "TrainingAttempt",
    "TrainingState",
    "RunManifest",
    "Run",
    "generate_run_id",
    "validate_run_id",
    "build_dataset_identity",
    "create_run",
    "load_run",
    "update_run_manifest",
    "update_training_state",
)

RUN_SCHEMA_VERSION = 1
DATASET_IDENTITY_SCHEME = "lidar-dataset-v2"
_LEGACY_DATASET_IDENTITY_SCHEME = "lidar-dataset-v1"

_CONFIG_PATH = "config.py"
_MANIFEST_NAME = "manifest.json"
_RUN_ORIGINS = frozenset({"native", "historical_import"})
_TRAINING_STATUSES = frozenset({"pending", "running", "failed", "completed"})
_ATTEMPT_STATUSES = frozenset({"running", "succeeded", "failed", "interrupted"})
_SLUG_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_TOKEN_PATTERN = re.compile(r"[0-9a-f]{24,}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def _require_mapping(value: object, *, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{description} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{description} keys must be strings")
    return value


def _require_fields(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    description: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"invalid {description} fields; "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_string(
    value: object,
    *,
    description: str,
    optional: bool = False,
) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{description} must be a string")
    if not value or value.strip() != value or "\0" in value:
        raise ValueError(f"{description} must be non-empty canonical text")
    return value


def _require_integer(
    value: object,
    *,
    description: str,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{description} must be an integer")
    if value < minimum:
        raise ValueError(f"{description} must be at least {minimum}")
    return value


def _require_sha256(value: object, *, description: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")
    return value


def _utc_timestamp(value: datetime | None = None) -> str:
    observed = datetime.now(timezone.utc) if value is None else value
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return observed.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _require_utc_timestamp(value: object, *, description: str) -> str:
    timestamp = _require_string(value, description=description)
    assert timestamp is not None
    if not timestamp.endswith("Z"):
        raise ValueError(f"{description} must be an ISO-8601 UTC timestamp")
    try:
        parsed = _parse_utc_timestamp(timestamp)
    except ValueError as error:
        raise ValueError(
            f"{description} must be an ISO-8601 UTC timestamp"
        ) from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{description} must use UTC")
    return timestamp


def _parse_utc_timestamp(value: str) -> datetime:
    for pattern in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(value, pattern).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError("invalid ISO-8601 UTC timestamp")


def _validate_slug(slug: object) -> str:
    if not isinstance(slug, str):
        raise TypeError("run slug must be a string")
    if len(slug) > 48 or _SLUG_PATTERN.fullmatch(slug) is None:
        raise ValueError(
            "run slug must contain 1-48 lowercase letters, digits, and single hyphens"
        )
    return slug


def _run_id_parts(run_id: object) -> tuple[str, str, str]:
    if not isinstance(run_id, str):
        raise TypeError("run_id must be a string")
    if len(run_id) < 16 + 1 + 1 + 1 + 24:
        raise ValueError("invalid run_id")
    timestamp, separator, remainder = run_id.partition("-")
    if not separator:
        raise ValueError("invalid run_id")
    slug, separator, token = remainder.rpartition("-")
    if not separator:
        raise ValueError("invalid run_id")
    try:
        parsed = datetime.strptime(timestamp, "%Y%m%dT%H%M%SZ")
    except ValueError as error:
        raise ValueError("run_id has an invalid UTC timestamp") from error
    if parsed.strftime("%Y%m%dT%H%M%SZ") != timestamp:
        raise ValueError("run_id has a non-canonical UTC timestamp")
    _validate_slug(slug)
    if _TOKEN_PATTERN.fullmatch(token) is None:
        raise ValueError("run_id token must contain at least 96 random bits")
    return timestamp, slug, token


def validate_run_id(run_id: str) -> str:
    """Validate and return one canonical run ID."""
    _run_id_parts(run_id)
    return run_id


def generate_run_id(slug: str, *, now: datetime | None = None) -> str:
    """Create a UTC-, slug-, and 96-random-bit run identifier."""
    normalized_slug = _validate_slug(slug)
    observed = datetime.now(timezone.utc) if now is None else now
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise ValueError("run ID time must be timezone-aware")
    timestamp = observed.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{normalized_slug}-{secrets.token_hex(12)}"


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


@dataclass(frozen=True, slots=True)
class RunPaths:
    root: Path
    manifest: Path
    config: Path
    training: Path
    smoke: Path
    evaluation: Path
    benchmark: Path

    def __post_init__(self) -> None:
        for name in (
            "root",
            "manifest",
            "config",
            "training",
            "smoke",
            "evaluation",
            "benchmark",
        ):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"RunPaths.{name} must be a pathlib.Path")
        root = _absolute(self.root)
        if self.root != root:
            raise ValueError("RunPaths.root must be an absolute lexical path")
        expected = {
            "manifest": root / _MANIFEST_NAME,
            "config": root / _CONFIG_PATH,
            "training": root / "training",
            "smoke": root / "smoke",
            "evaluation": root / "evaluation",
            "benchmark": root / "benchmark",
        }
        for name, path in expected.items():
            if getattr(self, name) != path:
                raise ValueError(f"RunPaths.{name} is not canonical")

    @classmethod
    def for_run(cls, runs_root: Path | str, run_id: str) -> RunPaths:
        validate_run_id(run_id)
        root = _absolute(runs_root) / run_id
        return cls(
            root=root,
            manifest=root / _MANIFEST_NAME,
            config=root / _CONFIG_PATH,
            training=root / "training",
            smoke=root / "smoke",
            evaluation=root / "evaluation",
            benchmark=root / "benchmark",
        )

    @classmethod
    def from_root(cls, root: Path | str) -> RunPaths:
        absolute_root = _absolute(root)
        return cls.for_run(absolute_root.parent, absolute_root.name)


@dataclass(frozen=True, slots=True)
class ConfigArtifact:
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if self.path != _CONFIG_PATH:
            raise ValueError(f"canonical config path must be {_CONFIG_PATH!r}")
        _require_sha256(self.sha256, description="config sha256")
        _require_integer(self.size_bytes, description="config size_bytes")

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ConfigArtifact:
        data = _require_mapping(value, description="config artifact")
        _require_fields(
            data,
            {"path", "sha256", "size_bytes"},
            description="config artifact",
        )
        return cls(
            path=data["path"],
            sha256=data["sha256"],
            size_bytes=data["size_bytes"],
        )


def _normalize_names(
    values: Iterable[str] | None,
    *,
    description: str,
) -> tuple[str, ...] | None:
    if values is None:
        return None
    if isinstance(values, str):
        raise TypeError(f"{description} must be an iterable of strings")
    result = tuple(values)
    if not result:
        raise ValueError(f"{description} must not be empty when known")
    for value in result:
        _require_string(value, description=description)
    if len(result) != len(set(result)):
        raise ValueError(f"{description} must not contain duplicates")
    return result


def _normalize_tasks(
    tasks: Mapping[str, Iterable[str]] | None,
) -> tuple[tuple[str, tuple[str, ...]], ...] | None:
    if tasks is None:
        return None
    if not isinstance(tasks, Mapping):
        raise TypeError("dataset tasks must be a mapping")
    normalized = []
    for name, classes in tasks.items():
        task_name = _require_string(name, description="dataset task name")
        task_classes = _normalize_names(
            classes,
            description=f"classes for dataset task {name!r}",
        )
        assert task_name is not None and task_classes is not None
        normalized.append((task_name, task_classes))
    if not normalized:
        raise ValueError("dataset tasks must not be empty when known")
    normalized.sort(key=lambda item: item[0])
    if len({name for name, _ in normalized}) != len(normalized):
        raise ValueError("dataset task names must be unique")
    return tuple(normalized)


def _dataset_payload(
    *,
    scheme: str = DATASET_IDENTITY_SCHEME,
    name: str | None,
    version: str | None,
    root_reference: str | None,
    semantic_partition: str | None,
    framework_key: str | None,
    annotation_files: FileSetIdentity | None,
    class_names: tuple[str, ...] | None,
    tasks: tuple[tuple[str, tuple[str, ...]], ...] | None,
) -> dict[str, object]:
    return {
        "scheme": scheme,
        "name": name,
        "version": version,
        "root_reference": root_reference,
        "semantic_partition": semantic_partition,
        "framework_key": framework_key,
        "annotation_files": (
            None if annotation_files is None else annotation_files.to_dict()
        ),
        "class_names": None if class_names is None else list(class_names),
        "tasks": (
            None
            if tasks is None
            else [
                {"name": task_name, "classes": list(classes)}
                for task_name, classes in tasks
            ]
        ),
    }


def _dataset_identity_payload(evidence: Mapping[str, object]) -> dict[str, object]:
    """Return scheme-specific semantic evidence used for identity hashing."""
    payload = dict(evidence)
    if payload["scheme"] == DATASET_IDENTITY_SCHEME:
        payload.pop("root_reference")
    return payload


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    scheme: str
    identity_sha256: str
    name: str | None
    version: str | None
    root_reference: str | None
    semantic_partition: str | None
    framework_key: str | None
    annotation_files: FileSetIdentity | None
    class_names: tuple[str, ...] | None
    tasks: tuple[tuple[str, tuple[str, ...]], ...] | None

    def __post_init__(self) -> None:
        if self.scheme not in {
            DATASET_IDENTITY_SCHEME,
            _LEGACY_DATASET_IDENTITY_SCHEME,
        }:
            raise ValueError(f"unsupported dataset identity scheme: {self.scheme!r}")
        _require_sha256(self.identity_sha256, description="dataset identity_sha256")
        for value, description in (
            (self.name, "dataset name"),
            (self.version, "dataset version"),
            (self.root_reference, "dataset root_reference"),
            (self.semantic_partition, "dataset semantic_partition"),
            (self.framework_key, "dataset framework_key"),
        ):
            _require_string(value, description=description, optional=True)
        if self.annotation_files is not None and not isinstance(
            self.annotation_files,
            FileSetIdentity,
        ):
            raise TypeError("annotation_files must be a FileSetIdentity or None")
        if self.class_names != _normalize_names(
            self.class_names,
            description="dataset class_names",
        ):
            raise ValueError("dataset class_names are not canonical")
        task_mapping = None if self.tasks is None else dict(self.tasks)
        if self.tasks != _normalize_tasks(task_mapping):
            raise ValueError("dataset tasks are not canonical")
        expected = _canonical_hash(
            _dataset_identity_payload(
                _dataset_payload(
                    scheme=self.scheme,
                    name=self.name,
                    version=self.version,
                    root_reference=self.root_reference,
                    semantic_partition=self.semantic_partition,
                    framework_key=self.framework_key,
                    annotation_files=self.annotation_files,
                    class_names=self.class_names,
                    tasks=self.tasks,
                )
            )
        )
        if self.identity_sha256 != expected:
            raise ValueError("dataset identity_sha256 does not match its evidence")

    def to_dict(self) -> dict[str, object]:
        result = _dataset_payload(
            scheme=self.scheme,
            name=self.name,
            version=self.version,
            root_reference=self.root_reference,
            semantic_partition=self.semantic_partition,
            framework_key=self.framework_key,
            annotation_files=self.annotation_files,
            class_names=self.class_names,
            tasks=self.tasks,
        )
        result["identity_sha256"] = self.identity_sha256
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DatasetIdentity:
        data = _require_mapping(value, description="dataset identity")
        _require_fields(
            data,
            {
                "scheme",
                "identity_sha256",
                "name",
                "version",
                "root_reference",
                "semantic_partition",
                "framework_key",
                "annotation_files",
                "class_names",
                "tasks",
            },
            description="dataset identity",
        )
        annotation_value = data["annotation_files"]
        annotations = (
            None
            if annotation_value is None
            else FileSetIdentity.from_dict(
                _require_mapping(annotation_value, description="annotation file set")
            )
        )
        class_value = data["class_names"]
        if class_value is not None and (
            not isinstance(class_value, list)
            or not all(isinstance(item, str) for item in class_value)
        ):
            raise TypeError("dataset class_names must be a list of strings or null")
        tasks_value = data["tasks"]
        task_mapping: dict[str, list[str]] | None = None
        if tasks_value is not None:
            if not isinstance(tasks_value, list):
                raise TypeError("dataset tasks must be a list or null")
            task_mapping = {}
            for item in tasks_value:
                task = _require_mapping(item, description="dataset task")
                _require_fields(
                    task,
                    {"name", "classes"},
                    description="dataset task",
                )
                name = task["name"]
                classes = task["classes"]
                if not isinstance(name, str) or not isinstance(classes, list) or not all(
                    isinstance(class_name, str) for class_name in classes
                ):
                    raise TypeError("invalid dataset task name/classes")
                if name in task_mapping:
                    raise ValueError("duplicate dataset task name")
                task_mapping[name] = classes
        return cls(
            scheme=data["scheme"],
            identity_sha256=data["identity_sha256"],
            name=data["name"],
            version=data["version"],
            root_reference=data["root_reference"],
            semantic_partition=data["semantic_partition"],
            framework_key=data["framework_key"],
            annotation_files=annotations,
            class_names=(None if class_value is None else tuple(class_value)),
            tasks=_normalize_tasks(task_mapping),
        )


def build_dataset_identity(
    *,
    name: str | None,
    version: str | None,
    root_reference: str | None,
    semantic_partition: str | None,
    framework_key: str | None,
    annotation_files: FileSetIdentity | None,
    class_names: Iterable[str] | None,
    tasks: Mapping[str, Iterable[str]] | None,
    scheme: str = DATASET_IDENTITY_SCHEME,
) -> DatasetIdentity:
    """Build deterministic dataset evidence without hashing bulk sensor data."""
    if scheme not in {
        DATASET_IDENTITY_SCHEME,
        _LEGACY_DATASET_IDENTITY_SCHEME,
    }:
        raise ValueError(f"unsupported dataset identity scheme: {scheme!r}")
    if annotation_files is not None and not isinstance(
        annotation_files,
        FileSetIdentity,
    ):
        raise TypeError("annotation_files must be a FileSetIdentity or None")
    normalized_classes = _normalize_names(
        class_names,
        description="dataset class_names",
    )
    normalized_tasks = _normalize_tasks(tasks)
    payload = _dataset_payload(
        scheme=scheme,
        name=name,
        version=version,
        root_reference=root_reference,
        semantic_partition=semantic_partition,
        framework_key=framework_key,
        annotation_files=annotation_files,
        class_names=normalized_classes,
        tasks=normalized_tasks,
    )
    return DatasetIdentity(
        scheme=scheme,
        identity_sha256=_canonical_hash(_dataset_identity_payload(payload)),
        name=name,
        version=version,
        root_reference=root_reference,
        semantic_partition=semantic_partition,
        framework_key=framework_key,
        annotation_files=annotation_files,
        class_names=normalized_classes,
        tasks=normalized_tasks,
    )


@dataclass(frozen=True, slots=True)
class TrainingAttempt:
    attempt_id: str
    started_at: str
    finished_at: str | None
    status: str
    resume_checkpoint: CheckpointArtifact | None
    failure: str | None

    def __post_init__(self) -> None:
        _require_string(self.attempt_id, description="training attempt_id")
        _require_utc_timestamp(self.started_at, description="attempt started_at")
        if self.finished_at is not None:
            _require_utc_timestamp(self.finished_at, description="attempt finished_at")
            if _parse_utc_timestamp(self.finished_at) < _parse_utc_timestamp(
                self.started_at
            ):
                raise ValueError("attempt finished_at precedes started_at")
        if self.status not in _ATTEMPT_STATUSES:
            raise ValueError(f"unsupported training attempt status: {self.status!r}")
        if self.resume_checkpoint is not None and not isinstance(
            self.resume_checkpoint,
            CheckpointArtifact,
        ):
            raise TypeError("resume_checkpoint must be a CheckpointArtifact or None")
        _require_string(
            self.failure,
            description="training attempt failure",
            optional=True,
        )
        if self.status == "running":
            if self.finished_at is not None or self.failure is not None:
                raise ValueError("running attempt cannot be finished or failed")
        elif self.status == "succeeded":
            if self.finished_at is None or self.failure is not None:
                raise ValueError("succeeded attempt requires finished_at and no failure")
        elif self.finished_at is None or self.failure is None:
            raise ValueError("failed/interrupted attempt requires finish and failure")

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "resume_checkpoint": (
                None
                if self.resume_checkpoint is None
                else self.resume_checkpoint.to_dict()
            ),
            "failure": self.failure,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TrainingAttempt:
        data = _require_mapping(value, description="training attempt")
        _require_fields(
            data,
            {
                "attempt_id",
                "started_at",
                "finished_at",
                "status",
                "resume_checkpoint",
                "failure",
            },
            description="training attempt",
        )
        checkpoint_value = data["resume_checkpoint"]
        return cls(
            attempt_id=data["attempt_id"],
            started_at=data["started_at"],
            finished_at=data["finished_at"],
            status=data["status"],
            resume_checkpoint=(
                None
                if checkpoint_value is None
                else CheckpointArtifact.from_dict(
                    _require_mapping(
                        checkpoint_value,
                        description="resume checkpoint",
                    )
                )
            ),
            failure=data["failure"],
        )


@dataclass(frozen=True, slots=True)
class TrainingState:
    status: str
    attempts: tuple[TrainingAttempt, ...]
    outputs: TrainingOutputs | None

    def __post_init__(self) -> None:
        if self.status not in _TRAINING_STATUSES:
            raise ValueError(f"unsupported training status: {self.status!r}")
        if not isinstance(self.attempts, tuple) or not all(
            isinstance(attempt, TrainingAttempt) for attempt in self.attempts
        ):
            raise TypeError("training attempts must be a tuple of TrainingAttempt")
        attempt_ids = [attempt.attempt_id for attempt in self.attempts]
        if len(attempt_ids) != len(set(attempt_ids)):
            raise ValueError("training attempt IDs must be unique")
        running = [attempt for attempt in self.attempts if attempt.status == "running"]
        if running and (len(running) != 1 or self.attempts[-1] is not running[0]):
            raise ValueError("only the final training attempt may be running")
        if self.status == "pending":
            if self.attempts or self.outputs is not None:
                raise ValueError("pending training has no attempts or outputs")
        elif self.status == "running":
            if not self.attempts or self.attempts[-1].status != "running":
                raise ValueError("running training requires a running final attempt")
            if self.outputs is not None:
                raise ValueError("running training cannot have outputs")
        elif self.status == "failed":
            if not self.attempts or self.attempts[-1].status not in {
                "failed",
                "interrupted",
            }:
                raise ValueError("failed training requires a failed final attempt")
            if self.outputs is not None:
                raise ValueError("failed training cannot have outputs")
        else:
            if self.outputs is None or not isinstance(self.outputs, TrainingOutputs):
                raise ValueError("completed training requires exact outputs")
            if self.attempts and self.attempts[-1].status != "succeeded":
                raise ValueError("completed training final attempt must have succeeded")

    @classmethod
    def pending(cls) -> TrainingState:
        return cls(status="pending", attempts=(), outputs=None)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "outputs": None if self.outputs is None else self.outputs.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TrainingState:
        data = _require_mapping(value, description="training state")
        _require_fields(
            data,
            {"status", "attempts", "outputs"},
            description="training state",
        )
        attempts_value = data["attempts"]
        if not isinstance(attempts_value, list):
            raise TypeError("training attempts must be a list")
        outputs_value = data["outputs"]
        return cls(
            status=data["status"],
            attempts=tuple(TrainingAttempt.from_dict(item) for item in attempts_value),
            outputs=(
                None
                if outputs_value is None
                else TrainingOutputs.from_dict(
                    _require_mapping(outputs_value, description="training outputs")
                )
            ),
        )


def _require_native_checkpoint_path(
    artifact: CheckpointArtifact,
    *,
    description: str,
    expected_name: str | None = None,
) -> None:
    path = PurePosixPath(artifact.path)
    if path.is_absolute() or len(path.parts) != 2 or path.parts[0] != "training":
        raise ValueError(f"native {description} must be directly run-local training evidence")
    if expected_name is not None and path.name != expected_name:
        raise ValueError(
            f"native {description} must use literal filename {expected_name!r}"
        )


@dataclass(frozen=True, slots=True)
class RunManifest:
    schema_version: int
    revision: int
    run_id: str
    slug: str
    created_at: str
    origin: str
    parent_run_id: str | None
    config: ConfigArtifact
    dataset: DatasetIdentity
    code_provenance: CodeProvenance | None
    environment: EnvironmentInfo | None
    training_compatibility: TrainingCompatibilityIdentity | None
    target_epoch: int
    training: TrainingState
    resumable: bool
    history_complete: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or self.schema_version != RUN_SCHEMA_VERSION
        ):
            raise ValueError(f"unsupported run schema version: {self.schema_version!r}")
        _require_integer(self.revision, description="manifest revision")
        _, run_slug, _ = _run_id_parts(self.run_id)
        if self.slug != run_slug or _validate_slug(self.slug) != self.slug:
            raise ValueError("manifest slug does not match run_id")
        _require_utc_timestamp(self.created_at, description="manifest created_at")
        if self.origin not in _RUN_ORIGINS:
            raise ValueError(f"unsupported run origin: {self.origin!r}")
        if self.parent_run_id is not None:
            validate_run_id(self.parent_run_id)
            if self.parent_run_id == self.run_id:
                raise ValueError("run cannot be its own parent")
        if not isinstance(self.config, ConfigArtifact):
            raise TypeError("manifest config must be a ConfigArtifact")
        if not isinstance(self.dataset, DatasetIdentity):
            raise TypeError("manifest dataset must be a DatasetIdentity")
        if self.code_provenance is not None and not isinstance(
            self.code_provenance,
            CodeProvenance,
        ):
            raise TypeError("code_provenance must be CodeProvenance or None")
        if self.environment is not None and not isinstance(
            self.environment,
            EnvironmentInfo,
        ):
            raise TypeError("environment must be EnvironmentInfo or None")
        if self.training_compatibility is not None and not isinstance(
            self.training_compatibility,
            TrainingCompatibilityIdentity,
        ):
            raise TypeError(
                "training_compatibility must be TrainingCompatibilityIdentity or None"
            )
        _require_integer(self.target_epoch, description="target_epoch", minimum=1)
        if not isinstance(self.training, TrainingState):
            raise TypeError("manifest training must be a TrainingState")
        if not isinstance(self.resumable, bool) or not isinstance(
            self.history_complete,
            bool,
        ):
            raise TypeError("resumable and history_complete must be bool values")

        if self.origin == "historical_import":
            if self.training.status != "completed":
                raise ValueError("historical imported run must be completed")
            assert self.training.outputs is not None
            if self.training.outputs.final_checkpoint.epoch != self.target_epoch:
                raise ValueError("final checkpoint epoch does not match target_epoch")
            if (
                Path(self.training.outputs.final_checkpoint.path).name
                != f"epoch_{self.target_epoch}.pth"
            ):
                raise ValueError(
                    "final checkpoint must use the literal target epoch filename"
                )
            if self.resumable:
                raise ValueError("historical imported run is non-resumable")
            if self.history_complete:
                raise ValueError("historical imported run history is incomplete")
        elif self.training.status == "completed":
            assert self.training.outputs is not None
            if self.training.outputs.final_checkpoint.epoch != self.target_epoch:
                raise ValueError("final checkpoint epoch does not match target_epoch")
            if (
                Path(self.training.outputs.final_checkpoint.path).name
                != f"epoch_{self.target_epoch}.pth"
            ):
                raise ValueError(
                    "final checkpoint must use the literal target epoch filename"
                )
            if self.resumable:
                raise ValueError("completed run cannot be resumable")
        elif not self.resumable:
            raise ValueError("incomplete native training must remain resumable")

        if self.origin == "native":
            if (
                self.code_provenance is None
                or self.environment is None
                or self.training_compatibility is None
            ):
                raise ValueError(
                    "native run requires code, environment, and compatibility evidence"
                )
            if self.training_compatibility.config_sha256 != self.config.sha256:
                raise ValueError(
                    "training compatibility config hash does not match canonical config"
                )
            if (
                self.training_compatibility.dataset_sha256
                != self.dataset.identity_sha256
            ):
                raise ValueError(
                    "training compatibility dataset hash does not match dataset identity"
                )
            if not self.training_compatibility.training_sources.files:
                raise ValueError("native training source identity must not be empty")
            if not self.history_complete:
                raise ValueError("native run must have complete history")
            if self.training.status == "completed" and not self.training.attempts:
                raise ValueError("completed native run requires an attempt record")
            for value, name in (
                (self.dataset.name, "name"),
                (self.dataset.root_reference, "root_reference"),
                (self.dataset.semantic_partition, "semantic_partition"),
                (self.dataset.framework_key, "framework_key"),
                (self.dataset.annotation_files, "annotation_files"),
                (self.dataset.class_names, "class_names"),
                (self.dataset.tasks, "tasks"),
            ):
                if value is None:
                    raise ValueError(f"native dataset identity requires {name}")
            assert self.dataset.annotation_files is not None
            if not self.dataset.annotation_files.files:
                raise ValueError("native dataset annotation file identity must not be empty")
            for attempt in self.training.attempts:
                if attempt.resume_checkpoint is not None:
                    resume_checkpoint = attempt.resume_checkpoint
                    if resume_checkpoint.epoch is None:
                        raise ValueError("native resume checkpoint requires an epoch")
                    if resume_checkpoint.epoch >= self.target_epoch:
                        raise ValueError(
                            "native resume checkpoint epoch must be below target_epoch"
                        )
                    _require_native_checkpoint_path(
                        resume_checkpoint,
                        description="resume checkpoint",
                        expected_name=f"epoch_{resume_checkpoint.epoch}.pth",
                    )
            if self.training.outputs is not None:
                final_checkpoint = self.training.outputs.final_checkpoint
                selected_checkpoint = self.training.outputs.selected_checkpoint
                _require_native_checkpoint_path(
                    final_checkpoint,
                    description="final checkpoint",
                    expected_name=f"epoch_{self.target_epoch}.pth",
                )
                _require_native_checkpoint_path(
                    selected_checkpoint,
                    description="selected checkpoint",
                )
                if selected_checkpoint.path == final_checkpoint.path:
                    if selected_checkpoint != final_checkpoint:
                        raise ValueError(
                            "selected checkpoint conflicts with final checkpoint identity"
                        )
                elif not (
                    PurePosixPath(selected_checkpoint.path).name.startswith("best_")
                    and PurePosixPath(selected_checkpoint.path).name.endswith(".pth")
                    and len(PurePosixPath(selected_checkpoint.path).name) > len("best_.pth")
                ):
                    raise ValueError(
                        "native selected checkpoint must be final or literal best_*.pth"
                    )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "revision": self.revision,
            "run_id": self.run_id,
            "slug": self.slug,
            "created_at": self.created_at,
            "origin": self.origin,
            "parent_run_id": self.parent_run_id,
            "config": self.config.to_dict(),
            "dataset": self.dataset.to_dict(),
            "code_provenance": (
                None if self.code_provenance is None else self.code_provenance.to_dict()
            ),
            "environment": (
                None if self.environment is None else self.environment.to_dict()
            ),
            "training_compatibility": (
                None
                if self.training_compatibility is None
                else self.training_compatibility.to_dict()
            ),
            "target_epoch": self.target_epoch,
            "training": self.training.to_dict(),
            "resumable": self.resumable,
            "history_complete": self.history_complete,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RunManifest:
        data = _require_mapping(value, description="run manifest")
        _require_fields(
            data,
            {
                "schema_version",
                "revision",
                "run_id",
                "slug",
                "created_at",
                "origin",
                "parent_run_id",
                "config",
                "dataset",
                "code_provenance",
                "environment",
                "training_compatibility",
                "target_epoch",
                "training",
                "resumable",
                "history_complete",
            },
            description="run manifest",
        )

        def optional_evidence(
            key: str,
            loader: Any,
        ) -> Any:
            item = data[key]
            return (
                None
                if item is None
                else loader(_require_mapping(item, description=key))
            )

        return cls(
            schema_version=data["schema_version"],
            revision=data["revision"],
            run_id=data["run_id"],
            slug=data["slug"],
            created_at=data["created_at"],
            origin=data["origin"],
            parent_run_id=data["parent_run_id"],
            config=ConfigArtifact.from_dict(
                _require_mapping(data["config"], description="config")
            ),
            dataset=DatasetIdentity.from_dict(
                _require_mapping(data["dataset"], description="dataset")
            ),
            code_provenance=optional_evidence(
                "code_provenance",
                CodeProvenance.from_dict,
            ),
            environment=optional_evidence(
                "environment",
                EnvironmentInfo.from_dict,
            ),
            training_compatibility=optional_evidence(
                "training_compatibility",
                TrainingCompatibilityIdentity.from_dict,
            ),
            target_epoch=data["target_epoch"],
            training=TrainingState.from_dict(
                _require_mapping(data["training"], description="training")
            ),
            resumable=data["resumable"],
            history_complete=data["history_complete"],
        )


@dataclass(frozen=True, slots=True)
class Run:
    paths: RunPaths
    manifest: RunManifest

    def __post_init__(self) -> None:
        if not isinstance(self.paths, RunPaths):
            raise TypeError("Run.paths must be RunPaths")
        if not isinstance(self.manifest, RunManifest):
            raise TypeError("Run.manifest must be RunManifest")
        if self.paths.root.name != self.manifest.run_id:
            raise ValueError("run path does not match manifest run_id")

    @property
    def run_id(self) -> str:
        return self.manifest.run_id

    @property
    def selected_checkpoint(self) -> CheckpointArtifact | None:
        outputs = self.manifest.training.outputs
        return None if outputs is None else outputs.selected_checkpoint


def _require_real_directory(path: Path, *, description: str) -> None:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{description} must not be a symlink: {path}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise NotADirectoryError(f"{description} is not a directory: {path}")


def _require_regular_file(path: Path, *, description: str) -> os.stat_result:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{description} must not be a symlink: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{description} is not a regular file: {path}")
    return metadata


def _verify_config(paths: RunPaths, artifact: ConfigArtifact) -> None:
    initial = _require_regular_file(paths.config, description="canonical config")
    digest = hashlib.sha256()
    size = 0
    descriptor = os.open(
        paths.config,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or (
            initial.st_dev,
            initial.st_ino,
        ) != (opened.st_dev, opened.st_ino):
            raise RuntimeError("canonical config changed while being opened")
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
        final = os.fstat(stream.fileno())
    current = paths.config.lstat()
    if stat.S_ISLNK(current.st_mode) or (
        opened.st_dev,
        opened.st_ino,
    ) != (current.st_dev, current.st_ino):
        raise RuntimeError("canonical config path changed while being verified")
    if (
        opened.st_size != final.st_size
        or opened.st_mtime_ns != final.st_mtime_ns
        or opened.st_ctime_ns != final.st_ctime_ns
    ):
        raise RuntimeError("canonical config changed while being verified")
    if size != artifact.size_bytes or digest.hexdigest() != artifact.sha256:
        raise ValueError("canonical config bytes do not match manifest identity")


def load_run(run_directory: Path | str) -> Run:
    """Load one explicit canonical run and verify its config identity/layout."""
    paths = RunPaths.from_root(run_directory)
    _require_real_directory(paths.root, description="run directory")
    for path, description in (
        (paths.training, "training directory"),
        (paths.evaluation, "evaluation directory"),
        (paths.benchmark, "benchmark directory"),
    ):
        _require_real_directory(path, description=description)
    try:
        _require_real_directory(paths.smoke, description="smoke directory")
    except FileNotFoundError:
        # Runs created before durable smoke evidence have no smoke directory.
        # Absence is a valid, distinguishable missing-stage state.
        pass
    _require_regular_file(paths.manifest, description="run manifest")
    manifest = RunManifest.from_dict(read_json_object(paths.manifest))
    if manifest.run_id != paths.root.name:
        raise ValueError("manifest run_id does not match its directory")
    _verify_config(paths, manifest.config)
    return Run(paths=paths, manifest=manifest)


def create_run(
    runs_root: Path | str,
    *,
    slug: str,
    config_bytes: bytes,
    dataset: DatasetIdentity,
    target_epoch: int,
    code_provenance: CodeProvenance | None = None,
    environment: EnvironmentInfo | None = None,
    training_compatibility: TrainingCompatibilityIdentity | None = None,
    origin: str = "native",
    parent_run_id: str | None = None,
    training_state: TrainingState | None = None,
    run_id: str | None = None,
    created_at: str | None = None,
) -> Run:
    """Transactionally create and exclusively publish one canonical run."""
    normalized_slug = _validate_slug(slug)
    selected_run_id = generate_run_id(normalized_slug) if run_id is None else run_id
    _, id_slug, _ = _run_id_parts(selected_run_id)
    if id_slug != normalized_slug:
        raise ValueError("explicit run_id slug does not match slug")
    if not isinstance(config_bytes, bytes):
        raise TypeError("config_bytes must be bytes")
    if not config_bytes:
        raise ValueError("canonical config snapshot must not be empty")
    try:
        config_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("canonical config snapshot must be UTF-8") from error
    if not isinstance(dataset, DatasetIdentity):
        raise TypeError("dataset must be a DatasetIdentity")
    if origin not in _RUN_ORIGINS:
        raise ValueError(f"unsupported run origin: {origin!r}")
    state = TrainingState.pending() if training_state is None else training_state
    if not isinstance(state, TrainingState):
        raise TypeError("training_state must be a TrainingState")
    config = ConfigArtifact(
        path=_CONFIG_PATH,
        sha256=hashlib.sha256(config_bytes).hexdigest(),
        size_bytes=len(config_bytes),
    )
    manifest = RunManifest(
        schema_version=RUN_SCHEMA_VERSION,
        revision=0,
        run_id=selected_run_id,
        slug=normalized_slug,
        created_at=_utc_timestamp() if created_at is None else created_at,
        origin=origin,
        parent_run_id=parent_run_id,
        config=config,
        dataset=dataset,
        code_provenance=code_provenance,
        environment=environment,
        training_compatibility=training_compatibility,
        target_epoch=target_epoch,
        training=state,
        resumable=origin == "native" and state.status != "completed",
        history_complete=origin == "native",
    )
    final_paths = RunPaths.for_run(runs_root, selected_run_id)
    staging = create_staging_directory(final_paths.root.parent, selected_run_id)
    try:
        (staging / "training").mkdir()
        (staging / "smoke").mkdir()
        (staging / "evaluation").mkdir()
        (staging / "benchmark").mkdir()
        (staging / _CONFIG_PATH).write_bytes(config_bytes)
        write_json_atomic(staging / _MANIFEST_NAME, manifest.to_dict())
        publish_directory_exclusive(staging, final_paths.root)
    except BaseException:
        try:
            cleanup_staging_directory(staging)
        except BaseException:
            pass
        raise
    return load_run(final_paths.root)


def _immutable_manifest_fields(manifest: RunManifest) -> tuple[object, ...]:
    return (
        manifest.schema_version,
        manifest.run_id,
        manifest.slug,
        manifest.created_at,
        manifest.origin,
        manifest.parent_run_id,
        manifest.config,
        manifest.dataset,
        manifest.code_provenance,
        manifest.environment,
        manifest.training_compatibility,
        manifest.target_epoch,
        manifest.history_complete,
    )


def _same_attempt_start(first: TrainingAttempt, second: TrainingAttempt) -> bool:
    return (
        first.attempt_id == second.attempt_id
        and first.started_at == second.started_at
        and first.resume_checkpoint == second.resume_checkpoint
    )


def _require_history_extension(old: TrainingState, new: TrainingState) -> None:
    common = min(len(old.attempts), len(new.attempts))
    for index in range(common):
        previous = old.attempts[index]
        replacement = new.attempts[index]
        if previous == replacement:
            continue
        if (
            index == len(old.attempts) - 1
            and previous.status == "running"
            and replacement.status != "running"
            and _same_attempt_start(previous, replacement)
        ):
            continue
        raise ValueError("training update rewrites immutable attempt history")
    if len(new.attempts) < len(old.attempts):
        raise ValueError("training update removes attempt history")
    if old.attempts and old.attempts[-1].status == "running":
        if len(new.attempts) != len(old.attempts):
            raise ValueError("running attempt must finish before another is appended")
    elif len(new.attempts) > len(old.attempts) + 1:
        raise ValueError("training update may append only one attempt at a time")


def update_run_manifest(
    run_directory: Path | str,
    replacement_manifest: RunManifest,
    *,
    expected_revision: int,
) -> Run:
    """Atomically replace a manifest under an optimistic run-local lock."""
    if not isinstance(replacement_manifest, RunManifest):
        raise TypeError("replacement_manifest must be a RunManifest")
    _require_integer(expected_revision, description="expected_revision")
    paths = RunPaths.from_root(run_directory)
    _require_real_directory(paths.root, description="run directory")
    directory_descriptor = os.open(
        paths.root,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        fcntl.flock(directory_descriptor, fcntl.LOCK_EX)
        current = load_run(paths.root)
        if current.manifest.revision != expected_revision:
            raise RuntimeError(
                "manifest revision conflict: "
                f"expected {expected_revision}, observed {current.manifest.revision}"
            )
        if current.manifest.origin == "historical_import":
            raise ValueError("historical imported run is terminal")
        if current.manifest.training.status == "completed":
            raise ValueError("completed native run is terminal")
        if replacement_manifest.revision != expected_revision + 1:
            raise ValueError("replacement manifest revision must increment by one")
        if _immutable_manifest_fields(replacement_manifest) != _immutable_manifest_fields(
            current.manifest
        ):
            raise ValueError("run identity fields are immutable")
        expected_resumable = replacement_manifest.training.status != "completed"
        if replacement_manifest.resumable != expected_resumable:
            raise ValueError("resumable must reflect native training completion")
        _require_history_extension(
            current.manifest.training,
            replacement_manifest.training,
        )
        write_json_atomic(paths.manifest, replacement_manifest.to_dict())
        return load_run(paths.root)
    finally:
        try:
            fcntl.flock(directory_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(directory_descriptor)


def update_training_state(
    run: Run | Path | str,
    training_state: TrainingState,
    *,
    expected_revision: int,
) -> Run:
    """Persist a replacement training state with optimistic revision checking."""
    if not isinstance(training_state, TrainingState):
        raise TypeError("training_state must be a TrainingState")
    loaded = run if isinstance(run, Run) else load_run(run)
    replacement_manifest = replace(
        loaded.manifest,
        revision=expected_revision + 1,
        training=training_state,
        resumable=training_state.status != "completed",
    )
    return update_run_manifest(
        loaded.paths.root,
        replacement_manifest,
        expected_revision=expected_revision,
    )
