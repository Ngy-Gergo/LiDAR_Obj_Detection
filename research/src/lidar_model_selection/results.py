"""Immutable, run-bound evaluation and benchmark result records."""

from __future__ import annotations

import math
import re
import secrets
import stat
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .provenance import CodeProvenance, EnvironmentInfo
from .runs import Run, validate_run_id
from .storage import (
    cleanup_staging_directory,
    create_staging_directory,
    publish_directory_exclusive,
    read_json_object,
    write_json_exclusive,
)


__all__ = (
    "ResultBinding",
    "ResultBindingMismatch",
    "ResultFailure",
    "ResultRecord",
    "create_result_id",
    "create_result",
    "binding_for_run",
    "verify_result_binding",
    "publish_result",
    "load_result",
    "list_results",
    "select_result",
)

_SCHEMA_VERSION = 1
_RESULT_TYPES = frozenset({"evaluation", "benchmark"})
_RESULT_STATUSES = frozenset({"succeeded", "failed"})
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_RESULT_ID_PATTERN = re.compile(
    r"(?P<timestamp>[0-9]{8}T[0-9]{12}Z)-"
    r"(?P<result_type>[a-z][a-z0-9]*(?:-[a-z0-9]+)*)-"
    r"(?P<random>[0-9a-f]{24,})\Z"
)
_RESULT_FILE_NAME = "result.json"


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
        raise ValueError(
            f"invalid {description} keys; "
            f"missing={sorted(keys - actual)}, extra={sorted(actual - keys)}"
        )


def _require_string(value: object, *, description: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{description} must be a string")
    if not value.strip():
        raise ValueError(f"{description} must contain non-whitespace text")
    return value


def _require_sha256(value: object, *, description: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")
    return value


def _require_result_type(value: object) -> str:
    result_type = _require_string(value, description="result type")
    if result_type not in _RESULT_TYPES:
        raise ValueError(
            "result type must be 'evaluation' or 'benchmark'"
        )
    return result_type


def _require_result_id(value: object) -> re.Match[str]:
    result_id = _require_string(value, description="result ID")
    match = _RESULT_ID_PATTERN.fullmatch(result_id)
    if match is None:
        raise ValueError(
            "result ID must contain a UTC timestamp, result-type slug, "
            "and at least 96 random bits"
        )
    _parse_id_timestamp(match.group("timestamp"))
    return match


def _utc_datetime(value: object, *, description: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{description} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{description} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _timestamp_text(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_timestamp(value: object, *, description: str) -> datetime:
    text = _require_string(value, description=description)
    if not text.endswith("Z"):
        raise ValueError(f"{description} must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{description} is not a valid timestamp") from exc
    normalized = _utc_datetime(parsed, description=description)
    if _timestamp_text(normalized) != text:
        raise ValueError(f"{description} must use six fractional UTC digits")
    return normalized


def _id_timestamp(value: datetime) -> str:
    return value.strftime("%Y%m%dT%H%M%S%fZ")


def _parse_id_timestamp(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%S%fZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError("result ID contains an invalid UTC timestamp") from exc


def _freeze_json(value: object, active: set[int] | None = None) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("result payload must not contain NaN or Infinity")
        return value
    if not isinstance(value, (Mapping, list, tuple)):
        raise TypeError(
            "result payload values must be JSON-compatible, got "
            f"{type(value).__name__}"
        )

    if active is None:
        active = set()
    identity = id(value)
    if identity in active:
        raise ValueError("result payload must not contain a reference cycle")
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            if not all(isinstance(key, str) for key in value):
                raise TypeError("result payload object keys must be strings")
            return MappingProxyType(
                {
                    key: _freeze_json(nested, active)
                    for key, nested in value.items()
                }
            )
        return tuple(_freeze_json(nested, active) for nested in value)
    finally:
        active.remove(identity)


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(nested) for nested in value]
    return value


@dataclass(frozen=True, slots=True)
class ResultBinding:
    """The exact run, effective config, and selected checkpoint used."""

    run_id: str
    config_sha256: str
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        _require_sha256(
            self.config_sha256,
            description="binding config_sha256",
        )
        _require_sha256(
            self.checkpoint_sha256,
            description="binding checkpoint_sha256",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "config_sha256": self.config_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ResultBinding:
        data = _require_mapping(value, description="result binding")
        _require_exact_keys(
            data,
            {"run_id", "config_sha256", "checkpoint_sha256"},
            description="result binding",
        )
        return cls(
            run_id=data["run_id"],  # type: ignore[arg-type]
            config_sha256=data["config_sha256"],  # type: ignore[arg-type]
            checkpoint_sha256=data["checkpoint_sha256"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ResultBindingMismatch:
    """One mismatch between a result binding and its loaded run."""

    field: str
    expected: str
    actual: str

    def __post_init__(self) -> None:
        _require_string(self.field, description="binding mismatch field")
        _require_string(self.expected, description="binding mismatch expected")
        _require_string(self.actual, description="binding mismatch actual")

    def to_dict(self) -> dict[str, object]:
        return {
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ResultBindingMismatch:
        data = _require_mapping(value, description="result binding mismatch")
        _require_exact_keys(
            data,
            {"field", "expected", "actual"},
            description="result binding mismatch",
        )
        return cls(
            field=data["field"],  # type: ignore[arg-type]
            expected=data["expected"],  # type: ignore[arg-type]
            actual=data["actual"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ResultFailure:
    """Serializable failure information for an unsuccessful invocation."""

    error_type: str
    message: str
    traceback: str | None = None

    def __post_init__(self) -> None:
        _require_string(self.error_type, description="failure error_type")
        if not isinstance(self.message, str):
            raise TypeError("failure message must be a string")
        if self.traceback is not None and not isinstance(self.traceback, str):
            raise TypeError("failure traceback must be a string or None")

    def to_dict(self) -> dict[str, object]:
        return {
            "error_type": self.error_type,
            "message": self.message,
            "traceback": self.traceback,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ResultFailure:
        data = _require_mapping(value, description="result failure")
        _require_exact_keys(
            data,
            {"error_type", "message", "traceback"},
            description="result failure",
        )
        return cls(
            error_type=data["error_type"],  # type: ignore[arg-type]
            message=data["message"],  # type: ignore[arg-type]
            traceback=data["traceback"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ResultRecord:
    """One terminal, immutable result from one evaluation or benchmark."""

    schema_version: int
    result_id: str
    result_type: str
    binding: ResultBinding
    status: str
    started_at: datetime
    finished_at: datetime
    payload: Mapping[str, object]
    provenance: CodeProvenance | None
    environment: EnvironmentInfo | None
    failure: ResultFailure | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != _SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported result schema_version: {self.schema_version!r}"
            )
        result_type = _require_result_type(self.result_type)
        result_id_match = _require_result_id(self.result_id)
        if result_id_match.group("result_type") != result_type:
            raise ValueError("result ID type does not match result_type")
        if not isinstance(self.binding, ResultBinding):
            raise TypeError("result binding must be a ResultBinding")
        if self.status not in _RESULT_STATUSES:
            raise ValueError("result status must be 'succeeded' or 'failed'")

        started_at = _utc_datetime(self.started_at, description="started_at")
        finished_at = _utc_datetime(self.finished_at, description="finished_at")
        if finished_at < started_at:
            raise ValueError("finished_at must not precede started_at")
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "finished_at", finished_at)

        payload = _require_mapping(self.payload, description="result payload")
        object.__setattr__(self, "payload", _freeze_json(payload))
        if self.provenance is not None and not isinstance(
            self.provenance,
            CodeProvenance,
        ):
            raise TypeError("result provenance must be CodeProvenance or None")
        if self.environment is not None and not isinstance(
            self.environment,
            EnvironmentInfo,
        ):
            raise TypeError("result environment must be EnvironmentInfo or None")
        if self.failure is not None and not isinstance(
            self.failure,
            ResultFailure,
        ):
            raise TypeError("result failure must be ResultFailure or None")
        if self.status == "succeeded" and self.failure is not None:
            raise ValueError("a succeeded result must not contain failure evidence")
        if self.status == "failed" and self.failure is None:
            raise ValueError("a failed result must contain failure evidence")

    @property
    def successful(self) -> bool:
        return self.status == "succeeded"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "result_id": self.result_id,
            "result_type": self.result_type,
            "binding": self.binding.to_dict(),
            "status": self.status,
            "started_at": _timestamp_text(self.started_at),
            "finished_at": _timestamp_text(self.finished_at),
            "payload": _thaw_json(self.payload),
            "provenance": (
                None if self.provenance is None else self.provenance.to_dict()
            ),
            "environment": (
                None if self.environment is None else self.environment.to_dict()
            ),
            "failure": None if self.failure is None else self.failure.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ResultRecord:
        data = _require_mapping(value, description="result record")
        _require_exact_keys(
            data,
            {
                "schema_version",
                "result_id",
                "result_type",
                "binding",
                "status",
                "started_at",
                "finished_at",
                "payload",
                "provenance",
                "environment",
                "failure",
            },
            description="result record",
        )
        provenance_value = data["provenance"]
        environment_value = data["environment"]
        failure_value = data["failure"]
        return cls(
            schema_version=data["schema_version"],  # type: ignore[arg-type]
            result_id=data["result_id"],  # type: ignore[arg-type]
            result_type=data["result_type"],  # type: ignore[arg-type]
            binding=ResultBinding.from_dict(
                _require_mapping(data["binding"], description="result binding")
            ),
            status=data["status"],  # type: ignore[arg-type]
            started_at=_parse_timestamp(data["started_at"], description="started_at"),
            finished_at=_parse_timestamp(
                data["finished_at"],
                description="finished_at",
            ),
            payload=_require_mapping(data["payload"], description="result payload"),
            provenance=(
                None
                if provenance_value is None
                else CodeProvenance.from_dict(
                    _require_mapping(
                        provenance_value,
                        description="result provenance",
                    )
                )
            ),
            environment=(
                None
                if environment_value is None
                else EnvironmentInfo.from_dict(
                    _require_mapping(
                        environment_value,
                        description="result environment",
                    )
                )
            ),
            failure=(
                None
                if failure_value is None
                else ResultFailure.from_dict(
                    _require_mapping(failure_value, description="result failure")
                )
            ),
        )


def create_result_id(
    result_type: str,
    *,
    timestamp: datetime | None = None,
) -> str:
    """Create a sortable ID with a UTC time and 96 random bits."""
    result_type = _require_result_type(result_type)
    instant = (
        datetime.now(timezone.utc)
        if timestamp is None
        else _utc_datetime(timestamp, description="result ID timestamp")
    )
    result_id = f"{_id_timestamp(instant)}-{result_type}-{secrets.token_hex(12)}"
    _require_result_id(result_id)
    return result_id


def create_result(
    *,
    result_type: str,
    binding: ResultBinding,
    status: str,
    started_at: datetime,
    finished_at: datetime,
    payload: Mapping[str, object],
    provenance: CodeProvenance | None = None,
    environment: EnvironmentInfo | None = None,
    failure: ResultFailure | None = None,
) -> ResultRecord:
    """Create a terminal record with a fresh ID for this invocation."""
    return ResultRecord(
        schema_version=_SCHEMA_VERSION,
        result_id=create_result_id(result_type, timestamp=finished_at),
        result_type=result_type,
        binding=binding,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        payload=payload,
        provenance=provenance,
        environment=environment,
        failure=failure,
    )


def verify_result_binding(
    binding: ResultBinding,
    *,
    run_id: str,
    config_sha256: str,
    selected_checkpoint_sha256: str,
) -> tuple[ResultBindingMismatch, ...]:
    """Compare a result binding with explicit evidence from a loaded run."""
    if not isinstance(binding, ResultBinding):
        raise TypeError("binding must be a ResultBinding")
    expected = ResultBinding(
        run_id=run_id,
        config_sha256=config_sha256,
        checkpoint_sha256=selected_checkpoint_sha256,
    )
    mismatches = []
    for field in ("run_id", "config_sha256", "checkpoint_sha256"):
        expected_value = getattr(expected, field)
        actual_value = getattr(binding, field)
        if expected_value != actual_value:
            mismatches.append(
                ResultBindingMismatch(
                    field=field,
                    expected=expected_value,
                    actual=actual_value,
                )
            )
    return tuple(mismatches)


def _require_real_directory(path: Path, *, description: str) -> None:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{description} must not be a symlink: {path}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise NotADirectoryError(f"{description} is not a directory: {path}")


def _require_regular_file(path: Path, *, description: str) -> None:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{description} must not be a symlink: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{description} must be a regular file: {path}")


def _directory_result_type(result_directory: Path) -> str:
    return _require_result_type(result_directory.name)


def binding_for_run(run: Run) -> ResultBinding:
    """Build the exact binding for a loaded, completed run."""
    if not isinstance(run, Run):
        raise TypeError("run must be a loaded Run")
    if run.manifest.training.status != "completed":
        raise ValueError("results require a completed run")
    selected_checkpoint = run.selected_checkpoint
    if selected_checkpoint is None:
        raise ValueError("completed run has no selected checkpoint")
    return ResultBinding(
        run_id=run.run_id,
        config_sha256=run.manifest.config.sha256,
        checkpoint_sha256=selected_checkpoint.sha256,
    )


def _require_run_binding(run: Run, binding: ResultBinding) -> None:
    expected = binding_for_run(run)
    mismatches = verify_result_binding(
        binding,
        run_id=expected.run_id,
        config_sha256=expected.config_sha256,
        selected_checkpoint_sha256=expected.checkpoint_sha256,
    )
    if mismatches:
        fields = ", ".join(mismatch.field for mismatch in mismatches)
        raise ValueError(f"result binding does not match its run: {fields}")


def _run_result_directory(run: Run, result_type: str) -> Path:
    binding_for_run(run)
    return (
        run.paths.evaluation
        if _require_result_type(result_type) == "evaluation"
        else run.paths.benchmark
    )


def publish_result(run: Run, result: ResultRecord) -> Path:
    """Transactionally publish one immutable result directory."""
    if not isinstance(result, ResultRecord):
        raise TypeError("result must be a ResultRecord")
    _require_run_binding(run, result.binding)
    parent = _run_result_directory(run, result.result_type)

    staging = create_staging_directory(parent, result.result_id)
    destination = parent / result.result_id
    try:
        write_json_exclusive(staging / _RESULT_FILE_NAME, result.to_dict())
        publish_directory_exclusive(staging, destination)
    except BaseException:
        try:
            cleanup_staging_directory(staging)
        except BaseException:
            pass
        raise
    return destination


def _load_result_from_directory(
    result_directory: Path,
    result_id: str,
) -> ResultRecord:
    result_id_match = _require_result_id(result_id)
    expected_type = _directory_result_type(result_directory)
    if result_id_match.group("result_type") != expected_type:
        raise ValueError("result ID type does not match its result directory")
    _require_real_directory(result_directory, description="result directory")

    record_directory = result_directory / result_id
    _require_real_directory(record_directory, description="result record directory")
    record_path = record_directory / _RESULT_FILE_NAME
    _require_regular_file(record_path, description="result record")
    result = ResultRecord.from_dict(read_json_object(record_path))
    if result.result_id != result_id:
        raise ValueError("result record ID does not match its directory name")
    if result.result_type != expected_type:
        raise ValueError("result record type does not match its directory")
    return result


def load_result(
    run: Run,
    result_type: str,
    result_id: str,
) -> ResultRecord:
    """Load one explicitly named immutable result record."""
    parent = _run_result_directory(run, result_type)
    result = _load_result_from_directory(parent, result_id)
    _require_run_binding(run, result.binding)
    return result


def list_results(run: Run, result_type: str) -> tuple[ResultRecord, ...]:
    """Load every published result in one explicit run-owned directory."""
    parent = _run_result_directory(run, result_type)
    _require_real_directory(parent, description="result directory")

    records = []
    for path in sorted(parent.iterdir(), key=lambda item: item.name):
        if path.name.startswith("."):
            continue
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"result directory contains a symlink: {path}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"result directory contains an unexpected entry: {path}")
        _require_result_id(path.name)
        record = _load_result_from_directory(parent, path.name)
        _require_run_binding(run, record.binding)
        records.append(record)
    return tuple(records)


def select_result(
    results: Iterable[ResultRecord],
    *,
    result_id: str | None = None,
) -> ResultRecord:
    """Select an explicit successful result, or the sole successful result."""
    if isinstance(results, (str, bytes, Mapping)):
        raise TypeError("results must be an iterable of ResultRecord values")
    records = tuple(results)
    if not all(isinstance(record, ResultRecord) for record in records):
        raise TypeError("every result must be a ResultRecord")
    identifiers = tuple(record.result_id for record in records)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("result collection contains duplicate result IDs")

    if result_id is not None:
        _require_result_id(result_id)
        matches = [record for record in records if record.result_id == result_id]
        if not matches:
            raise KeyError(f"result ID was not found: {result_id}")
        selected = matches[0]
        if not selected.successful:
            raise ValueError(f"explicit result is not successful: {result_id}")
        return selected

    successful = [record for record in records if record.successful]
    if not successful:
        raise ValueError("no successful result is available; specify a successful result ID")
    if len(successful) > 1:
        raise ValueError(
            "multiple successful results are available; an explicit result ID is required"
        )
    return successful[0]
