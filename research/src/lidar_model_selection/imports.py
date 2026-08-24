"""Explicit import of historical runs and their result evidence."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpoints import (
    CheckpointArtifact,
    TrainingOutputs,
    checkpoint_epoch,
    identify_checkpoint,
    verify_checkpoint,
)
from .results import (
    ResultBinding,
    ResultFailure,
    ResultRecord,
    create_result,
    publish_result,
)
from .runs import (
    DatasetIdentity,
    Run,
    TrainingState,
    create_run,
    generate_run_id,
)


__all__ = (
    "read_dataset_identity",
    "import_historical_run",
)

_CHECKPOINT_PATH_FIELDS = ("checkpoint_path", "checkpoint")


def _absolute_path(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _read_regular_bytes(path: Path, *, description: str) -> tuple[Path, bytes]:
    if not isinstance(path, Path):
        raise TypeError(f"{description} path must be a pathlib.Path")

    absolute = _absolute_path(path)
    initial = absolute.lstat()
    if stat.S_ISLNK(initial.st_mode):
        raise ValueError(f"{description} must not be a symlink: {absolute}")
    if not stat.S_ISREG(initial.st_mode):
        raise ValueError(f"{description} must be a regular file: {absolute}")

    descriptor = os.open(
        absolute,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = -1
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise ValueError(
                    f"{description} must be a regular file: {absolute}"
                )
            if (initial.st_dev, initial.st_ino) != (opened.st_dev, opened.st_ino):
                raise RuntimeError(f"{description} changed while being opened")
            contents = stream.read()
            final = os.fstat(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    current = absolute.lstat()
    if stat.S_ISLNK(current.st_mode) or (
        current.st_dev,
        current.st_ino,
    ) != (final.st_dev, final.st_ino):
        raise RuntimeError(f"{description} path changed while being read")
    if (
        opened.st_size != final.st_size
        or opened.st_mtime_ns != final.st_mtime_ns
        or opened.st_ctime_ns != final.st_ctime_ns
        or len(contents) != final.st_size
    ):
        raise RuntimeError(f"{description} changed while being read")
    return absolute, contents


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(constant: str) -> None:
    raise ValueError(f"invalid JSON constant: {constant}")


def _read_json_source(
    path: Path,
    *,
    description: str,
) -> tuple[dict[str, Any], dict[str, object]]:
    absolute, contents = _read_regular_bytes(path, description=description)
    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{description} must be UTF-8 JSON") from error
    value = json.loads(
        text,
        object_pairs_hook=_strict_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{description} JSON root must be an object")
    return value, {
        "path": os.fspath(absolute),
        "sha256": hashlib.sha256(contents).hexdigest(),
        "size_bytes": len(contents),
    }


def read_dataset_identity(path: Path) -> DatasetIdentity:
    """Read one exact strict-JSON dataset identity document."""
    value, _ = _read_json_source(path, description="dataset identity source")
    return DatasetIdentity.from_dict(value)


def _checkpoint_association(
    source: Mapping[str, object],
    selected_checkpoint: CheckpointArtifact,
) -> dict[str, object]:
    recorded: list[tuple[str, str]] = []
    for field in _CHECKPOINT_PATH_FIELDS:
        if field not in source:
            continue
        value = source[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"historical result {field!r} must be a non-empty string"
            )
        recorded.append((field, value))

    if len({value for _, value in recorded}) > 1:
        raise ValueError(
            "historical result has conflicting checkpoint path fields"
        )
    recorded_field, recorded_path = (
        recorded[0] if recorded else (None, None)
    )
    return {
        "basis": (
            "recorded_checkpoint_path"
            if recorded
            else "explicit_import_specification"
        ),
        "source_field": recorded_field,
        "recorded_checkpoint_path": recorded_path,
        "selected_checkpoint_path_at_import": selected_checkpoint.path,
        "checkpoint_sha256": selected_checkpoint.sha256,
        "checkpoint_sha256_observed_at": "import",
        "checkpoint_sha256_observed_at_measurement": False,
    }


def _source_status(
    source: Mapping[str, object],
) -> tuple[str, tuple[str, object] | None]:
    observations: list[tuple[str, bool]] = []
    for field in ("test_success", "success"):
        if field in source:
            value = source[field]
            if not isinstance(value, bool):
                raise TypeError(f"historical result {field!r} must be a boolean")
            observations.append((field, value))

    if "skipped" in source:
        skipped = source["skipped"]
        if not isinstance(skipped, bool):
            raise TypeError("historical result 'skipped' must be a boolean")
        if skipped:
            observations.append(("skipped", False))

    if "status" in source:
        status = source["status"]
        if not isinstance(status, str):
            raise TypeError("historical result 'status' must be a string")
        if status in {"succeeded", "success"}:
            observations.append(("status", True))
        elif status in {"failed", "failure"}:
            observations.append(("status", False))
        else:
            raise ValueError(
                f"unsupported historical result status: {status!r}"
            )

    if observations and len({success for _, success in observations}) > 1:
        raise ValueError("historical result has conflicting status evidence")
    if not observations:
        raise ValueError(
            "historical result requires explicit success status evidence"
        )
    field, success = observations[0]
    return ("succeeded" if success else "failed"), (field, source[field])


def _failure_for_source(source: Mapping[str, object]) -> ResultFailure:
    for field in ("error_message", "error"):
        value = source.get(field)
        if isinstance(value, str) and value:
            return ResultFailure(
                error_type="HistoricalResultFailure",
                message=value,
                traceback=None,
            )
    return ResultFailure(
        error_type="HistoricalResultFailure",
        message="historical source reports an unsuccessful result",
        traceback=None,
    )


def _historical_payload(
    *,
    result_type: str,
    source: Mapping[str, object],
    source_evidence: Mapping[str, object],
    selected_checkpoint: CheckpointArtifact,
    status_evidence: tuple[str, object] | None,
    imported_at: datetime,
) -> dict[str, object]:
    imported_at_text = imported_at.isoformat(timespec="microseconds").replace(
        "+00:00",
        "Z",
    )
    payload: dict[str, object] = {
        "kind": result_type,
        "historical_import": {
            "imported_at": imported_at_text,
            "result_record_time_semantics": "import_operation_not_measurement",
            "source_json": dict(source_evidence),
            "association": _checkpoint_association(
                source,
                selected_checkpoint,
            ),
            "source_status_evidence": (
                None
                if status_evidence is None
                else {
                    "field": status_evidence[0],
                    "value": status_evidence[1],
                }
            ),
        },
        "source_record": dict(source),
    }
    if result_type == "evaluation":
        payload.update(
            {
                "metric_profile": source.get("metric_profile"),
                "semantic_partition": source.get("semantic_partition"),
                "framework_key": source.get("framework_key"),
                "metrics": source.get("metrics"),
            }
        )
    else:
        payload.update(
            {
                "methodology": source.get("methodology"),
                "semantic_partition": source.get("semantic_partition"),
            }
        )
    return payload


def _historical_result(
    *,
    result_type: str,
    binding: ResultBinding,
    source: Mapping[str, object],
    source_evidence: Mapping[str, object],
    selected_checkpoint: CheckpointArtifact,
    observed_at: datetime,
) -> ResultRecord:
    status, status_evidence = _source_status(source)
    return create_result(
        result_type=result_type,
        binding=binding,
        status=status,
        started_at=observed_at,
        finished_at=observed_at,
        payload=_historical_payload(
            result_type=result_type,
            source=source,
            source_evidence=source_evidence,
            selected_checkpoint=selected_checkpoint,
            status_evidence=status_evidence,
            imported_at=observed_at,
        ),
        provenance=None,
        environment=None,
        failure=None if status == "succeeded" else _failure_for_source(source),
    )


def _require_checkpoint_unchanged(
    checkpoint: CheckpointArtifact,
    *,
    description: str,
) -> None:
    mismatches = verify_checkpoint(checkpoint)
    if mismatches:
        details = "; ".join(
            f"{mismatch.field}: expected {mismatch.expected!r}, "
            f"observed {mismatch.actual!r}"
            for mismatch in mismatches
        )
        raise ValueError(f"{description} changed before import: {details}")


def import_historical_run(
    runs_root: Path | str,
    *,
    slug: str,
    config_path: Path,
    final_checkpoint_path: Path,
    selected_checkpoint_path: Path,
    dataset_identity: DatasetIdentity,
    evaluation_json: Path | None = None,
    benchmark_json: Path | None = None,
    run_id: str | None = None,
    created_at: str | None = None,
) -> Run:
    """Import one explicitly enumerated completed historical run.

    Checkpoint bytes remain at their supplied absolute paths.  Only the exact
    effective-config bytes and immutable metadata are copied into the new run.
    """
    if not isinstance(dataset_identity, DatasetIdentity):
        raise TypeError("dataset_identity must be a DatasetIdentity")
    _, config_bytes = _read_regular_bytes(
        config_path,
        description="historical effective config",
    )
    if not config_bytes:
        raise ValueError("historical effective config must not be empty")
    try:
        config_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("historical effective config must be UTF-8") from error

    final_checkpoint = identify_checkpoint(final_checkpoint_path)
    target_epoch = checkpoint_epoch(final_checkpoint_path)
    if target_epoch is None or final_checkpoint_path.name != f"epoch_{target_epoch}.pth":
        raise ValueError(
            "historical final checkpoint must use canonical epoch_<N>.pth naming"
        )
    if target_epoch < 1:
        raise ValueError("historical target epoch must be at least 1")
    if final_checkpoint.epoch != target_epoch:
        raise ValueError("historical final checkpoint epoch evidence is inconsistent")

    final_absolute = Path(final_checkpoint.path)
    selected_absolute = _absolute_path(selected_checkpoint_path)
    selected_checkpoint = (
        final_checkpoint
        if selected_absolute == final_absolute
        else identify_checkpoint(selected_checkpoint_path)
    )
    outputs = TrainingOutputs(
        final_checkpoint=final_checkpoint,
        selected_checkpoint=selected_checkpoint,
    )
    training_state = TrainingState(
        status="completed",
        attempts=(),
        outputs=outputs,
    )

    source_documents: dict[
        str,
        tuple[dict[str, Any], dict[str, object]],
    ] = {}
    for result_type, source_path in (
        ("evaluation", evaluation_json),
        ("benchmark", benchmark_json),
    ):
        if source_path is not None:
            source_documents[result_type] = _read_json_source(
                source_path,
                description=f"historical {result_type} source",
            )

    selected_run_id = generate_run_id(slug) if run_id is None else run_id
    binding = ResultBinding(
        run_id=selected_run_id,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        checkpoint_sha256=selected_checkpoint.sha256,
    )
    observed_at = datetime.now(timezone.utc)
    prepared_results = tuple(
        _historical_result(
            result_type=result_type,
            binding=binding,
            source=source,
            source_evidence=evidence,
            selected_checkpoint=selected_checkpoint,
            observed_at=observed_at,
        )
        for result_type, (source, evidence) in source_documents.items()
    )

    _require_checkpoint_unchanged(
        final_checkpoint,
        description="historical final checkpoint",
    )
    if selected_checkpoint.path != final_checkpoint.path:
        _require_checkpoint_unchanged(
            selected_checkpoint,
            description="historical selected checkpoint",
        )

    imported = create_run(
        runs_root,
        slug=slug,
        config_bytes=config_bytes,
        dataset=dataset_identity,
        target_epoch=target_epoch,
        code_provenance=None,
        environment=None,
        training_compatibility=None,
        origin="historical_import",
        parent_run_id=None,
        training_state=training_state,
        run_id=selected_run_id,
        created_at=created_at,
    )
    _require_checkpoint_unchanged(
        final_checkpoint,
        description="historical final checkpoint",
    )
    if selected_checkpoint.path != final_checkpoint.path:
        _require_checkpoint_unchanged(
            selected_checkpoint,
            description="historical selected checkpoint",
        )
    for result in prepared_results:
        publish_result(imported, result)
    return imported
