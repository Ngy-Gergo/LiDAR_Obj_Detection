"""One ordinary run-owned training, evaluation, and benchmark workflow."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .benchmarking import benchmark_run
from .evaluation import evaluate_run
from .preflight import PreflightReport, preflight_run
from .results import ResultRecord
from .runs import Run, validate_run_id
from .storage import read_json_object, write_json_exclusive
from .training import create_training_run, execute_training

__all__ = (
    "PIPELINE_SCHEMA_VERSION",
    "PipelineRequest",
    "PipelineRecord",
    "run_pipeline",
    "load_pipeline_record",
)

PIPELINE_SCHEMA_VERSION = 1


def _positive(value: object, *, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{description} must be a positive integer")
    return value


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _text(value: object, *, description: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{description} must be non-empty canonical text")
    return value


def _utc(value: object, *, description: str) -> str:
    text = _text(value, description=description)
    if not text.endswith("Z"):
        raise ValueError(f"{description} must be a UTC timestamp")
    try:
        datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as error:
        raise ValueError(f"{description} must be a UTC timestamp") from error
    return text


@dataclass(frozen=True, slots=True)
class PipelineRequest:
    slug: str
    target_epoch: int
    source_config: Path | None = None
    benchmark_warmup: int = 100
    benchmark_samples: int = 1000
    sample_check: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.slug, str) or not self.slug:
            raise ValueError("pipeline slug must be non-empty")
        _positive(self.target_epoch, description="target_epoch")
        _positive(self.benchmark_warmup, description="benchmark_warmup")
        _positive(self.benchmark_samples, description="benchmark_samples")
        if self.source_config is not None and not isinstance(self.source_config, Path):
            raise TypeError("source_config must be a pathlib.Path or None")
        if not isinstance(self.sample_check, bool):
            raise TypeError("sample_check must be a boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "slug": self.slug,
            "target_epoch": self.target_epoch,
            "source_config": (
                None if self.source_config is None else str(self.source_config)
            ),
            "benchmark_warmup": self.benchmark_warmup,
            "benchmark_samples": self.benchmark_samples,
            "sample_check": self.sample_check,
        }


@dataclass(frozen=True, slots=True)
class PipelineRecord:
    schema_version: int
    pipeline_id: str
    run_id: str
    request: PipelineRequest
    status: str
    started_at: str
    finished_at: str
    preflight: PreflightReport | None
    evaluation_result_id: str | None
    benchmark_result_id: str | None
    failure: str | None

    def __post_init__(self) -> None:
        if self.schema_version != PIPELINE_SCHEMA_VERSION:
            raise ValueError("unsupported pipeline schema version")
        _text(self.pipeline_id, description="pipeline ID")
        validate_run_id(self.run_id)
        if not isinstance(self.request, PipelineRequest):
            raise TypeError("pipeline request must be a PipelineRequest")
        started = _utc(self.started_at, description="pipeline started_at")
        finished = _utc(self.finished_at, description="pipeline finished_at")
        if finished < started:
            raise ValueError("pipeline finished_at precedes started_at")
        if self.preflight is not None and self.preflight.run_id != self.run_id:
            raise ValueError("pipeline preflight belongs to a different run")
        if self.status not in {"succeeded", "failed"}:
            raise ValueError("pipeline status must be succeeded or failed")
        if self.status == "succeeded" and (
            self.preflight is None
            or self.evaluation_result_id is None
            or self.benchmark_result_id is None
            or self.failure is not None
        ):
            raise ValueError("successful pipeline record is incomplete")
        if self.status == "failed" and not self.failure:
            raise ValueError("failed pipeline record requires failure evidence")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "pipeline_id": self.pipeline_id,
            "run_id": self.run_id,
            "request": self.request.to_dict(),
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "preflight": None if self.preflight is None else self.preflight.to_dict(),
            "evaluation_result_id": self.evaluation_result_id,
            "benchmark_result_id": self.benchmark_result_id,
            "failure": self.failure,
        }


def _record_path(run: Run, pipeline_id: str) -> Path:
    return run.paths.root / "pipeline" / f"{pipeline_id}.json"


def _publish(record: PipelineRecord, run: Run) -> None:
    path = _record_path(run, record.pipeline_id)
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    write_json_exclusive(path, record.to_dict())


def run_pipeline(request: PipelineRequest) -> tuple[Run, PipelineRecord]:
    """Execute one request through the existing public run operations."""
    if not isinstance(request, PipelineRequest):
        raise TypeError("request must be a PipelineRequest")
    started = _timestamp()
    pipeline_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + "-"
        + secrets.token_hex(8)
    )
    run = create_training_run(
        request.slug,
        request.target_epoch,
        source_config=request.source_config,
    )
    preflight: PreflightReport | None = None
    evaluation: ResultRecord | None = None
    benchmark: ResultRecord | None = None
    try:
        preflight = preflight_run(
            run, operation="pipeline", sample_check=request.sample_check
        )
        run = execute_training(run)
        evaluation = evaluate_run(run)
        if evaluation.status != "succeeded":
            raise RuntimeError(
                f"evaluation result {evaluation.result_id} did not succeed"
            )
        benchmark = benchmark_run(
            run,
            warmup=request.benchmark_warmup,
            samples=request.benchmark_samples,
        )
        if benchmark.status != "succeeded":
            raise RuntimeError(
                f"benchmark result {benchmark.result_id} did not succeed"
            )
    except BaseException as error:
        record = PipelineRecord(
            schema_version=PIPELINE_SCHEMA_VERSION,
            pipeline_id=pipeline_id,
            run_id=run.run_id,
            request=request,
            status="failed",
            started_at=started,
            finished_at=_timestamp(),
            preflight=preflight,
            evaluation_result_id=(
                None if evaluation is None else evaluation.result_id
            ),
            benchmark_result_id=(None if benchmark is None else benchmark.result_id),
            failure=f"{type(error).__name__}: {error}",
        )
        _publish(record, run)
        raise

    record = PipelineRecord(
        schema_version=PIPELINE_SCHEMA_VERSION,
        pipeline_id=pipeline_id,
        run_id=run.run_id,
        request=request,
        status="succeeded",
        started_at=started,
        finished_at=_timestamp(),
        preflight=preflight,
        evaluation_result_id=evaluation.result_id,
        benchmark_result_id=benchmark.result_id,
        failure=None,
    )
    _publish(record, run)
    return run, record


def load_pipeline_record(path: Path) -> PipelineRecord:
    """Load one strict pipeline record created by this schema."""
    data = read_json_object(path)
    expected = {
        "schema_version",
        "pipeline_id",
        "run_id",
        "request",
        "status",
        "started_at",
        "finished_at",
        "preflight",
        "evaluation_result_id",
        "benchmark_result_id",
        "failure",
    }
    if set(data) != expected:
        raise ValueError("pipeline record fields are incomplete or unexpected")
    request_data = data["request"]
    if not isinstance(request_data, dict):
        raise TypeError("pipeline request must be an object")
    request_keys = {
        "slug",
        "target_epoch",
        "source_config",
        "benchmark_warmup",
        "benchmark_samples",
        "sample_check",
    }
    if set(request_data) != request_keys:
        raise ValueError("pipeline request fields are incomplete or unexpected")
    request = PipelineRequest(
        slug=request_data["slug"],
        target_epoch=request_data["target_epoch"],
        source_config=(
            None
            if request_data["source_config"] is None
            else Path(request_data["source_config"])
        ),
        benchmark_warmup=request_data["benchmark_warmup"],
        benchmark_samples=request_data["benchmark_samples"],
        sample_check=request_data["sample_check"],
    )
    preflight_data = data["preflight"]
    preflight = None
    if preflight_data is not None:
        if not isinstance(preflight_data, dict):
            raise TypeError("pipeline preflight must be an object")
        preflight_keys = {
            "run_id",
            "operation",
            "config_path",
            "dataset_root",
            "annotation_paths",
            "sample_checked",
        }
        if set(preflight_data) != preflight_keys:
            raise ValueError("pipeline preflight fields are incomplete or unexpected")
        if not isinstance(preflight_data["annotation_paths"], list):
            raise TypeError("pipeline preflight annotation_paths must be a list")
        preflight = PreflightReport(
            run_id=preflight_data["run_id"],
            operation=preflight_data["operation"],
            config_path=preflight_data["config_path"],
            dataset_root=preflight_data["dataset_root"],
            annotation_paths=tuple(preflight_data["annotation_paths"]),
            sample_checked=preflight_data["sample_checked"],
        )
    return PipelineRecord(
        schema_version=data["schema_version"],
        pipeline_id=data["pipeline_id"],
        run_id=data["run_id"],
        request=request,
        status=data["status"],
        started_at=data["started_at"],
        finished_at=data["finished_at"],
        preflight=preflight,
        evaluation_result_id=data["evaluation_result_id"],
        benchmark_result_id=data["benchmark_result_id"],
        failure=data["failure"],
    )
