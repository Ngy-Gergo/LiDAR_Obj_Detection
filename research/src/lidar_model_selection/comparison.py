"""Explicit, evidence-bound accuracy and runtime comparisons."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .results import ResultRecord, list_results, select_result
from .runs import Run, load_run, validate_run_id
from .storage import read_json_object, write_json_atomic


__all__ = (
    "COMPARISON_SCHEMA_VERSION",
    "DEFAULT_RUNS_ROOT",
    "KITTI_CAR_AP40_METRICS",
    "RUNTIME_SCOPES",
    "RUNTIME_STATISTICS",
    "ACCURACY_COMPATIBILITY_FIELDS",
    "RUNTIME_COMPATIBILITY_FIELDS",
    "CompatibilityWaiver",
    "ComparisonRow",
    "ComparisonReport",
    "compare_runs",
    "write_comparison_report",
    "load_comparison_report",
)

COMPARISON_SCHEMA_VERSION = 1
DEFAULT_RUNS_ROOT = Path(__file__).absolute().parents[3] / "research" / "runs"

_KITTI_PREFIX = "Kitti metric/pred_instances_3d/KITTI/"
KITTI_CAR_AP40_METRICS: Mapping[str, str] = MappingProxyType(
    {
        "car_3d_ap40_easy_strict": (
            _KITTI_PREFIX + "Car_3D_AP40_easy_strict"
        ),
        "car_3d_ap40_moderate_strict": (
            _KITTI_PREFIX + "Car_3D_AP40_moderate_strict"
        ),
        "car_3d_ap40_hard_strict": (
            _KITTI_PREFIX + "Car_3D_AP40_hard_strict"
        ),
        "car_bev_ap40_easy_strict": (
            _KITTI_PREFIX + "Car_BEV_AP40_easy_strict"
        ),
        "car_bev_ap40_moderate_strict": (
            _KITTI_PREFIX + "Car_BEV_AP40_moderate_strict"
        ),
        "car_bev_ap40_hard_strict": (
            _KITTI_PREFIX + "Car_BEV_AP40_hard_strict"
        ),
    }
)

RUNTIME_SCOPES = ("prediction_ms", "end_to_end_ms")
RUNTIME_STATISTICS = (
    "mean_ms",
    "min_ms",
    "max_ms",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "standard_deviation_ms",
)

ACCURACY_COMPATIBILITY_FIELDS = (
    "accuracy.dataset.identity",
    "accuracy.dataset.name",
    "accuracy.dataset.version",
    "accuracy.semantic_partition",
    "accuracy.task_class_schema",
    "accuracy.metric_profile.id",
    "accuracy.metric_profile.version",
)
RUNTIME_COMPATIBILITY_FIELDS = (
    "runtime.methodology.id",
    "runtime.methodology.version",
    "runtime.timing_scope",
    "runtime.statistic",
    "runtime.hardware_class",
    "runtime.precision",
    "runtime.batch_size",
    "runtime.workload_policy",
    "runtime.warmup_measurement_policy",
    "runtime.software_identity",
)

_RANKING_POLICY = (
    "competition ranking; accuracy descending and runtime ascending; "
    "equal values share a rank and leave the corresponding rank gap"
)
_CORE_RUNTIME_PACKAGES = ("torch", "mmengine", "mmcv", "mmdet", "mmdet3d")


def _require_mapping(value: object, *, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{description} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{description} keys must be strings")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    description: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"invalid {description} keys; "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_text(value: object, *, description: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{description} must be a string")
    if not value or value.strip() != value or "\0" in value:
        raise ValueError(f"{description} must be non-empty canonical text")
    return value


def _require_optional_text(value: object, *, description: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, description=description)


def _require_positive_integer(value: object, *, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{description} must be an integer")
    if value <= 0:
        raise ValueError(f"{description} must be greater than zero")
    return value


def _finite_number(value: object, *, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{description} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{description} must be finite")
    return normalized


def _freeze_json(value: object, active: set[int] | None = None) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("comparison evidence must not contain NaN or Infinity")
        return value
    if not isinstance(value, (Mapping, list, tuple)):
        raise TypeError(
            "comparison evidence must contain only JSON values, got "
            f"{type(value).__name__}"
        )
    if active is None:
        active = set()
    identity = id(value)
    if identity in active:
        raise ValueError("comparison evidence must not contain a reference cycle")
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            if not all(isinstance(key, str) for key in value):
                raise TypeError("comparison evidence object keys must be strings")
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


def _contains_unknown(value: object) -> bool:
    if value is None or value == "":
        return True
    if isinstance(value, Mapping):
        return not value or any(_contains_unknown(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return not value or any(_contains_unknown(item) for item in value)
    return False


@dataclass(frozen=True, slots=True)
class CompatibilityWaiver:
    """An explicit, persisted exception for one compatibility field."""

    field: str
    reason: str

    def __post_init__(self) -> None:
        _require_text(self.field, description="waiver field")
        _require_text(self.reason, description="waiver reason")

    def to_dict(self) -> dict[str, object]:
        return {"field": self.field, "reason": self.reason}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CompatibilityWaiver:
        data = _require_mapping(value, description="compatibility waiver")
        _require_exact_keys(
            data,
            {"field", "reason"},
            description="compatibility waiver",
        )
        return cls(field=data["field"], reason=data["reason"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    """One resolved run row with exact result identities and ranks."""

    run_id: str
    slug: str
    config_sha256: str
    checkpoint_sha256: str
    evaluation_result_id: str
    accuracy_metric: str
    accuracy_raw_key: str
    accuracy_value: float
    accuracy_rank: int
    benchmark_result_id: str | None = None
    runtime_scope: str | None = None
    runtime_statistic: str | None = None
    runtime_value: float | None = None
    runtime_rank: int | None = None

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        _require_text(self.slug, description="row slug")
        if self.run_id[17:].rsplit("-", 1)[0] != self.slug:
            raise ValueError("row slug does not match run ID")
        for digest, description in (
            (self.config_sha256, "row config SHA-256"),
            (self.checkpoint_sha256, "row checkpoint SHA-256"),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{description} must be a lowercase SHA-256 digest")
        _require_text(
            self.evaluation_result_id,
            description="row evaluation result ID",
        )
        if self.accuracy_metric not in KITTI_CAR_AP40_METRICS:
            raise ValueError(f"unsupported accuracy metric: {self.accuracy_metric!r}")
        if self.accuracy_raw_key != KITTI_CAR_AP40_METRICS[self.accuracy_metric]:
            raise ValueError("row accuracy raw key does not match accuracy metric")
        object.__setattr__(
            self,
            "accuracy_value",
            _finite_number(self.accuracy_value, description="row accuracy value"),
        )
        _require_positive_integer(self.accuracy_rank, description="accuracy rank")

        runtime_values = (
            self.benchmark_result_id,
            self.runtime_scope,
            self.runtime_statistic,
            self.runtime_value,
            self.runtime_rank,
        )
        if all(value is None for value in runtime_values):
            return
        if any(value is None for value in runtime_values):
            raise ValueError("row runtime evidence must be wholly present or absent")
        _require_text(
            self.benchmark_result_id,
            description="row benchmark result ID",
        )
        if self.runtime_scope not in RUNTIME_SCOPES:
            raise ValueError(f"unsupported runtime scope: {self.runtime_scope!r}")
        if self.runtime_statistic not in RUNTIME_STATISTICS:
            raise ValueError(
                f"unsupported runtime statistic: {self.runtime_statistic!r}"
            )
        object.__setattr__(
            self,
            "runtime_value",
            _finite_number(self.runtime_value, description="row runtime value"),
        )
        _require_positive_integer(self.runtime_rank, description="runtime rank")

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "slug": self.slug,
            "config_sha256": self.config_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "evaluation_result_id": self.evaluation_result_id,
            "accuracy_metric": self.accuracy_metric,
            "accuracy_raw_key": self.accuracy_raw_key,
            "accuracy_value": self.accuracy_value,
            "accuracy_rank": self.accuracy_rank,
            "benchmark_result_id": self.benchmark_result_id,
            "runtime_scope": self.runtime_scope,
            "runtime_statistic": self.runtime_statistic,
            "runtime_value": self.runtime_value,
            "runtime_rank": self.runtime_rank,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ComparisonRow:
        data = _require_mapping(value, description="comparison row")
        _require_exact_keys(
            data,
            {
                "run_id",
                "slug",
                "config_sha256",
                "checkpoint_sha256",
                "evaluation_result_id",
                "accuracy_metric",
                "accuracy_raw_key",
                "accuracy_value",
                "accuracy_rank",
                "benchmark_result_id",
                "runtime_scope",
                "runtime_statistic",
                "runtime_value",
                "runtime_rank",
            },
            description="comparison row",
        )
        return cls(**data)  # type: ignore[arg-type]


def _rank(values: Mapping[str, float], *, descending: bool) -> dict[str, int]:
    ordered = sorted(
        values.items(),
        key=lambda item: ((-item[1] if descending else item[1]), item[0]),
    )
    ranks: dict[str, int] = {}
    previous: float | None = None
    previous_rank = 0
    for position, (run_id, value) in enumerate(ordered, start=1):
        rank = previous_rank if previous is not None and value == previous else position
        ranks[run_id] = rank
        previous = value
        previous_rank = rank
    return ranks


def _normalize_waivers(
    waivers: Iterable[CompatibilityWaiver],
    *,
    runtime: bool,
) -> tuple[CompatibilityWaiver, ...]:
    if isinstance(waivers, (str, bytes, Mapping)):
        raise TypeError("waivers must be an iterable of CompatibilityWaiver values")
    normalized = tuple(waivers)
    if not all(isinstance(waiver, CompatibilityWaiver) for waiver in normalized):
        raise TypeError("every waiver must be a CompatibilityWaiver")
    fields = [waiver.field for waiver in normalized]
    if len(fields) != len(set(fields)):
        raise ValueError("compatibility waiver fields must be unique")
    supported = set(ACCURACY_COMPATIBILITY_FIELDS)
    if runtime:
        supported.update(RUNTIME_COMPATIBILITY_FIELDS)
    unknown = set(fields) - supported
    if unknown:
        raise ValueError(f"unsupported compatibility waiver fields: {sorted(unknown)}")
    return tuple(sorted(normalized, key=lambda waiver: waiver.field))


def _require_internal_match(
    *,
    field: str,
    left: object,
    right: object,
    waived_fields: set[str],
) -> None:
    if field in waived_fields or _contains_unknown(left) or _contains_unknown(right):
        return
    if left != right:
        raise ValueError(
            f"incompatible evidence within one run for {field}; "
            "provide an explicit field waiver with a reason"
        )


def _validate_compatibility(
    compatibility: Mapping[str, object],
    *,
    run_ids: tuple[str, ...],
    waivers: tuple[CompatibilityWaiver, ...],
    runtime: bool,
) -> Mapping[str, object]:
    expected_fields = set(ACCURACY_COMPATIBILITY_FIELDS)
    if runtime:
        expected_fields.update(RUNTIME_COMPATIBILITY_FIELDS)
    if set(compatibility) != expected_fields:
        raise ValueError(
            "comparison compatibility fields are incomplete or unexpected; "
            f"missing={sorted(expected_fields - set(compatibility))}, "
            f"extra={sorted(set(compatibility) - expected_fields)}"
        )

    waived = {waiver.field for waiver in waivers}
    normalized: dict[str, object] = {}
    for field in sorted(expected_fields):
        observations = _require_mapping(
            compatibility[field],
            description=f"compatibility observations for {field}",
        )
        if set(observations) != set(run_ids):
            raise ValueError(
                f"compatibility field {field!r} must observe every exact run"
            )
        frozen_observations = _freeze_json(dict(observations))
        assert isinstance(frozen_observations, Mapping)
        normalized[field] = frozen_observations
        if field in waived:
            continue
        if field == "accuracy.semantic_partition":
            for observation in observations.values():
                evidence = _require_mapping(
                    observation,
                    description="accuracy semantic partition evidence",
                )
                _require_internal_match(
                    field=field,
                    left=evidence.get("run"),
                    right=evidence.get("evaluation"),
                    waived_fields=waived,
                )
                _require_internal_match(
                    field=field,
                    left=evidence.get("run_framework_key"),
                    right=evidence.get("evaluation_framework_key"),
                    waived_fields=waived,
                )
        elif field == "runtime.workload_policy":
            for observation in observations.values():
                evidence = _require_mapping(
                    observation,
                    description="runtime workload compatibility evidence",
                )
                _require_internal_match(
                    field=field,
                    left=evidence.get("run_semantic_partition"),
                    right=evidence.get("benchmark_semantic_partition"),
                    waived_fields=waived,
                )
                _require_internal_match(
                    field=field,
                    left=evidence.get("run_framework_key"),
                    right=evidence.get("benchmark_framework_key"),
                    waived_fields=waived,
                )
        unknown_runs = [
            run_id
            for run_id in run_ids
            if _contains_unknown(observations[run_id])
        ]
        if unknown_runs:
            raise ValueError(
                f"unknown compatibility metadata for {field}: {unknown_runs}; "
                "provide an explicit field waiver with a reason"
            )
        baseline = observations[run_ids[0]]
        mismatched = [
            run_id
            for run_id in run_ids[1:]
            if observations[run_id] != baseline
        ]
        if mismatched:
            raise ValueError(
                f"incompatible cohort for {field}: {mismatched}; "
                "provide an explicit field waiver with a reason"
            )
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    """A deterministic comparison derived from explicit immutable results."""

    schema_version: int
    accuracy_metric: str
    runtime_scope: str | None
    runtime_statistic: str | None
    ranking_policy: str
    waivers: tuple[CompatibilityWaiver, ...]
    compatibility: Mapping[str, object]
    rows: tuple[ComparisonRow, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or self.schema_version != COMPARISON_SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported comparison schema_version: {self.schema_version!r}"
            )
        if self.accuracy_metric not in KITTI_CAR_AP40_METRICS:
            raise ValueError(f"unsupported accuracy metric: {self.accuracy_metric!r}")
        runtime = self.runtime_scope is not None or self.runtime_statistic is not None
        if runtime:
            if self.runtime_scope not in RUNTIME_SCOPES:
                raise ValueError(f"unsupported runtime scope: {self.runtime_scope!r}")
            if self.runtime_statistic not in RUNTIME_STATISTICS:
                raise ValueError(
                    f"unsupported runtime statistic: {self.runtime_statistic!r}"
                )
        if self.ranking_policy != _RANKING_POLICY:
            raise ValueError("unsupported comparison ranking policy")
        if not isinstance(self.rows, tuple) or not self.rows:
            raise ValueError("comparison report requires at least one row")
        if not all(isinstance(row, ComparisonRow) for row in self.rows):
            raise TypeError("comparison rows must be ComparisonRow values")
        if tuple(row.run_id for row in self.rows) != tuple(
            sorted(row.run_id for row in self.rows)
        ):
            raise ValueError("comparison rows must be sorted by run ID")
        run_ids = tuple(row.run_id for row in self.rows)
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("comparison rows must have unique run IDs")
        evaluation_ids = tuple(row.evaluation_result_id for row in self.rows)
        if len(evaluation_ids) != len(set(evaluation_ids)):
            raise ValueError("comparison rows must have unique evaluation result IDs")
        benchmark_ids = tuple(
            row.benchmark_result_id
            for row in self.rows
            if row.benchmark_result_id is not None
        )
        if len(benchmark_ids) != len(set(benchmark_ids)):
            raise ValueError("comparison rows must have unique benchmark result IDs")
        for row in self.rows:
            if row.accuracy_metric != self.accuracy_metric:
                raise ValueError("row accuracy metric does not match report")
            if runtime:
                if (
                    row.runtime_scope != self.runtime_scope
                    or row.runtime_statistic != self.runtime_statistic
                ):
                    raise ValueError("row runtime selection does not match report")
            elif row.runtime_scope is not None:
                raise ValueError("accuracy-only report must not contain runtime rows")

        normalized_waivers = _normalize_waivers(self.waivers, runtime=runtime)
        if self.waivers != normalized_waivers:
            raise ValueError("comparison waivers must be sorted by field")
        normalized_compatibility = _validate_compatibility(
            self.compatibility,
            run_ids=run_ids,
            waivers=normalized_waivers,
            runtime=runtime,
        )
        object.__setattr__(self, "compatibility", normalized_compatibility)

        accuracy_ranks = _rank(
            {row.run_id: row.accuracy_value for row in self.rows},
            descending=True,
        )
        if any(row.accuracy_rank != accuracy_ranks[row.run_id] for row in self.rows):
            raise ValueError("comparison accuracy ranks are not canonical")
        if runtime:
            runtime_ranks = _rank(
                {
                    row.run_id: row.runtime_value  # type: ignore[misc]
                    for row in self.rows
                },
                descending=False,
            )
            if any(row.runtime_rank != runtime_ranks[row.run_id] for row in self.rows):
                raise ValueError("comparison runtime ranks are not canonical")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "accuracy_metric": self.accuracy_metric,
            "runtime_scope": self.runtime_scope,
            "runtime_statistic": self.runtime_statistic,
            "ranking_policy": self.ranking_policy,
            "waivers": [waiver.to_dict() for waiver in self.waivers],
            "compatibility": _thaw_json(self.compatibility),
            "rows": [row.to_dict() for row in self.rows],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ComparisonReport:
        data = _require_mapping(value, description="comparison report")
        _require_exact_keys(
            data,
            {
                "schema_version",
                "accuracy_metric",
                "runtime_scope",
                "runtime_statistic",
                "ranking_policy",
                "waivers",
                "compatibility",
                "rows",
            },
            description="comparison report",
        )
        waiver_values = data["waivers"]
        row_values = data["rows"]
        if not isinstance(waiver_values, list):
            raise TypeError("comparison waivers must be a list")
        if not isinstance(row_values, list):
            raise TypeError("comparison rows must be a list")
        return cls(
            schema_version=data["schema_version"],  # type: ignore[arg-type]
            accuracy_metric=data["accuracy_metric"],  # type: ignore[arg-type]
            runtime_scope=data["runtime_scope"],  # type: ignore[arg-type]
            runtime_statistic=data["runtime_statistic"],  # type: ignore[arg-type]
            ranking_policy=data["ranking_policy"],  # type: ignore[arg-type]
            waivers=tuple(
                CompatibilityWaiver.from_dict(
                    _require_mapping(item, description="compatibility waiver")
                )
                for item in waiver_values
            ),
            compatibility=_require_mapping(
                data["compatibility"],
                description="comparison compatibility evidence",
            ),
            rows=tuple(
                ComparisonRow.from_dict(
                    _require_mapping(item, description="comparison row")
                )
                for item in row_values
            ),
        )


def _payload(record: ResultRecord, *, expected_kind: str) -> Mapping[str, Any]:
    payload = _require_mapping(record.payload, description=f"{expected_kind} payload")
    if payload.get("kind") != expected_kind:
        raise ValueError(
            f"{expected_kind} result {record.result_id} has an invalid payload kind"
        )
    return payload


def _optional_mapping(value: object, *, description: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return _require_mapping(value, description=description)


def _profile_observations(payload: Mapping[str, Any]) -> tuple[object, object]:
    profile = _optional_mapping(
        payload.get("metric_profile"),
        description="evaluation metric profile",
    )
    if profile is None:
        return None, None
    identifier = profile.get("id")
    version = profile.get("version")
    key = profile.get("key")
    if identifier is not None:
        _require_text(identifier, description="metric profile ID")
    if version is not None and (
        isinstance(version, bool) or not isinstance(version, int)
    ):
        raise TypeError("metric profile version must be an integer or null")
    if isinstance(version, int) and version <= 0:
        raise ValueError("metric profile version must be positive")
    if key is not None and not isinstance(key, str):
        raise TypeError("metric profile key must be a string or null")
    if identifier is not None and version is not None and key is not None:
        if key != f"{identifier}_v{version}":
            raise ValueError("metric profile key does not match its ID/version")
    return identifier, version


def _source_record(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    return _optional_mapping(
        payload.get("source_record"),
        description="historical source record",
    )


def _accuracy_value(
    record: ResultRecord,
    payload: Mapping[str, Any],
    metric: str,
) -> float:
    raw_key = KITTI_CAR_AP40_METRICS[metric]
    containers: list[Mapping[str, Any]] = []
    metrics = _optional_mapping(payload.get("metrics"), description="evaluation metrics")
    if metrics is not None:
        containers.append(metrics)
    source = _source_record(payload)
    if source is not None:
        source_metrics = _optional_mapping(
            source.get("metrics"),
            description="historical source metrics",
        )
        if source_metrics is not None:
            containers.append(source_metrics)
        containers.append(source)
    for container in containers:
        if raw_key in container:
            return _finite_number(
                container[raw_key],
                description=f"accuracy metric {raw_key!r}",
            )
        if metric in container:
            return _finite_number(
                container[metric],
                description=f"historical accuracy metric {metric!r}",
            )
    raise KeyError(
        f"evaluation result {record.result_id} has no exact metric {raw_key!r}"
    )


def _task_class_schema(run: Run) -> object:
    dataset = run.manifest.dataset
    return {
        "class_names": (
            None if dataset.class_names is None else list(dataset.class_names)
        ),
        "tasks": (
            None
            if dataset.tasks is None
            else [
                {"name": name, "classes": list(classes)}
                for name, classes in dataset.tasks
            ]
        ),
    }


def _accuracy_observations(
    run: Run,
    payload: Mapping[str, Any],
) -> dict[str, object]:
    dataset = run.manifest.dataset
    profile_id, profile_version = _profile_observations(payload)
    return {
        "accuracy.dataset.identity": dataset.identity_sha256,
        "accuracy.dataset.name": dataset.name,
        "accuracy.dataset.version": dataset.version,
        "accuracy.semantic_partition": {
            "run": dataset.semantic_partition,
            "evaluation": payload.get("semantic_partition"),
            "run_framework_key": dataset.framework_key,
            "evaluation_framework_key": payload.get("framework_key"),
        },
        "accuracy.task_class_schema": _task_class_schema(run),
        "accuracy.metric_profile.id": profile_id,
        "accuracy.metric_profile.version": profile_version,
    }


def _methodology_observations(
    payload: Mapping[str, Any],
    *,
    scope: str,
    statistic: str,
) -> tuple[object, object, object, object]:
    methodology = _optional_mapping(
        payload.get("methodology"),
        description="benchmark methodology",
    )
    if methodology is None:
        return None, None, {
            "name": scope,
            "definition": None,
            "synchronization": None,
        }, {
            "name": statistic,
            "definitions": None,
        }
    identifier = methodology.get("id")
    version = methodology.get("version")
    key = methodology.get("key")
    if identifier is not None:
        _require_text(identifier, description="benchmark methodology ID")
    if version is not None and (
        isinstance(version, bool) or not isinstance(version, int)
    ):
        raise TypeError("benchmark methodology version must be an integer or null")
    if isinstance(version, int) and version <= 0:
        raise ValueError("benchmark methodology version must be positive")
    if key is not None and not isinstance(key, str):
        raise TypeError("benchmark methodology key must be a string or null")
    if identifier is not None and version is not None and key is not None:
        if key != f"{identifier}_v{version}":
            raise ValueError("benchmark methodology key does not match ID/version")
    timing_scopes = _optional_mapping(
        methodology.get("timing_scopes"),
        description="benchmark timing scopes",
    )
    definitions = _optional_mapping(
        methodology.get("statistics"),
        description="benchmark statistic definitions",
    )
    synchronization = _optional_mapping(
        methodology.get("synchronization"),
        description="benchmark synchronization policy",
    )
    return (
        identifier,
        version,
        {
            "name": scope,
            "definition": None if timing_scopes is None else timing_scopes.get(scope),
            "synchronization": synchronization,
        },
        {"name": statistic, "definitions": definitions},
    )


def _runtime_hardware(
    payload: Mapping[str, Any],
    source: Mapping[str, Any] | None,
) -> object:
    hardware = _optional_mapping(payload.get("hardware"), description="benchmark hardware")
    if hardware is not None:
        return {
            "device_type": hardware.get("device_type"),
            "device_name": hardware.get("device_name"),
        }
    return {
        "device_type": None if source is None else source.get("device_type"),
        "device_name": None if source is None else source.get("gpu_name"),
    }


def _runtime_precision(
    payload: Mapping[str, Any],
    source: Mapping[str, Any] | None,
) -> object:
    precision = payload.get("precision")
    if precision is not None:
        return precision
    return None if source is None else source.get("precision")


def _runtime_software_identity(
    record: ResultRecord,
    source: Mapping[str, Any] | None,
) -> object:
    if record.environment is not None:
        environment = record.environment
        packages = dict(environment.packages)
        return {
            "python_version": environment.python_version,
            "python_implementation": environment.python_implementation,
            "platform": environment.platform,
            "machine": environment.machine,
            "packages": {
                name: packages.get(name) for name in _CORE_RUNTIME_PACKAGES
            },
            "torch_version": environment.torch_version,
            "cuda_version": environment.cuda_version,
            "cudnn_version": environment.cudnn_version,
        }
    return {
        "python_version": None if source is None else source.get("python_version"),
        "python_implementation": None,
        "platform": None,
        "machine": None,
        "packages": {
            "torch": None if source is None else source.get("pytorch_version"),
            "mmengine": None,
            "mmcv": None,
            "mmdet": None,
            "mmdet3d": None,
        },
        "torch_version": None if source is None else source.get("pytorch_version"),
        "cuda_version": None if source is None else source.get("cuda_version"),
        "cudnn_version": None,
    }


def _runtime_observations(
    run: Run,
    record: ResultRecord,
    payload: Mapping[str, Any],
    *,
    scope: str,
    statistic: str,
) -> dict[str, object]:
    source = _source_record(payload)
    workload = _optional_mapping(payload.get("workload"), description="benchmark workload")
    method_id, method_version, timing_scope, statistic_evidence = (
        _methodology_observations(payload, scope=scope, statistic=statistic)
    )

    def workload_value(name: str, historical_name: str | None = None) -> object:
        if workload is not None and name in workload:
            return workload[name]
        if source is None:
            return None
        return source.get(name if historical_name is None else historical_name)

    methodology = _optional_mapping(
        payload.get("methodology"),
        description="benchmark methodology",
    )
    batch_size = workload_value("batch_size")
    if batch_size is not None:
        _require_positive_integer(batch_size, description="benchmark batch size")
    return {
        "runtime.methodology.id": method_id,
        "runtime.methodology.version": method_version,
        "runtime.timing_scope": timing_scope,
        "runtime.statistic": statistic_evidence,
        "runtime.hardware_class": _runtime_hardware(payload, source),
        "runtime.precision": _runtime_precision(payload, source),
        "runtime.batch_size": batch_size,
        "runtime.workload_policy": {
            "run_semantic_partition": run.manifest.dataset.semantic_partition,
            "benchmark_semantic_partition": workload_value("semantic_partition"),
            "run_framework_key": run.manifest.dataset.framework_key,
            "benchmark_framework_key": workload_value("framework_key"),
            "num_workers": workload_value("num_workers"),
            "persistent_workers": workload_value("persistent_workers"),
            "drop_last": workload_value("drop_last"),
            "shuffle": workload_value("shuffle"),
        },
        "runtime.warmup_measurement_policy": {
            "warmup_count": workload_value("warmup_count", "warmup"),
            "measured_sample_count": workload_value(
                "measured_sample_count",
                "samples",
            ),
            "iterator_policy": (
                None if methodology is None else methodology.get("iterator_policy")
            ),
            "warmup_policy": (
                None if methodology is None else methodology.get("warmup_policy")
            ),
            "sample_policy": (
                None if methodology is None else methodology.get("sample_policy")
            ),
        },
        "runtime.software_identity": _runtime_software_identity(
            record,
            source,
        ),
    }


def _runtime_value(
    record: ResultRecord,
    payload: Mapping[str, Any],
    *,
    scope: str,
    statistic: str,
) -> float:
    values = _optional_mapping(
        payload.get(scope),
        description=f"benchmark timing scope {scope}",
    )
    if values is not None and statistic in values:
        result = _finite_number(
            values[statistic],
            description=f"benchmark {scope}.{statistic}",
        )
        if result < 0.0:
            raise ValueError(f"benchmark {scope}.{statistic} must be non-negative")
        return result
    source = _source_record(payload)
    legacy_prefix = "prediction" if scope == "prediction_ms" else "end_to_end"
    legacy_key = f"{legacy_prefix}_{statistic}"
    if source is not None and legacy_key in source:
        result = _finite_number(
            source[legacy_key],
            description=f"historical benchmark field {legacy_key}",
        )
        if result < 0.0:
            raise ValueError(
                f"historical benchmark field {legacy_key} must be non-negative"
            )
        return result
    raise KeyError(
        f"benchmark result {record.result_id} has no exact statistic "
        f"{scope}.{statistic}"
    )


def _canonical_runs(runs: Iterable[Run]) -> tuple[Run, ...]:
    if isinstance(runs, (str, bytes, Mapping)):
        raise TypeError("runs must be an iterable of loaded Run values")
    supplied = tuple(runs)
    if not supplied:
        raise ValueError("comparison requires at least one explicit run")
    if not all(isinstance(run, Run) for run in supplied):
        raise TypeError("every comparison input must be a loaded Run")
    identifiers = [run.run_id for run in supplied]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("comparison run IDs must be unique")

    current_runs = []
    for run in supplied:
        current = load_run(run.paths.root)
        if current.manifest != run.manifest:
            raise ValueError(f"loaded run is stale: {run.run_id}")
        current_runs.append(current)
    return tuple(sorted(current_runs, key=lambda run: run.run_id))


def _selection_map(
    value: Mapping[str, str | None] | None,
    *,
    run_ids: tuple[str, ...],
    description: str,
) -> Mapping[str, str | None]:
    if value is None:
        return MappingProxyType({})
    mapping = _require_mapping(value, description=description)
    unknown = set(mapping) - set(run_ids)
    if unknown:
        raise ValueError(f"{description} contains unknown run IDs: {sorted(unknown)}")
    for run_id, result_id in mapping.items():
        validate_run_id(run_id)
        if result_id is not None:
            _require_text(result_id, description=f"result ID for {run_id}")
    return MappingProxyType(dict(mapping))  # type: ignore[arg-type]


def compare_runs(
    runs: Iterable[Run],
    *,
    accuracy_metric: str,
    evaluation_result_ids: Mapping[str, str | None] | None = None,
    runtime_scope: str | None = None,
    runtime_statistic: str | None = None,
    benchmark_result_ids: Mapping[str, str | None] | None = None,
    waivers: Iterable[CompatibilityWaiver] = (),
) -> ComparisonReport:
    """Resolve exact result records, enforce compatibility, and rank runs.

    A missing per-run result ID delegates only to :func:`select_result`, whose
    sole implicit policy is to accept exactly one successful owned result.
    """
    if accuracy_metric not in KITTI_CAR_AP40_METRICS:
        raise ValueError(f"unsupported accuracy metric: {accuracy_metric!r}")
    runtime = runtime_scope is not None or runtime_statistic is not None
    if runtime:
        if runtime_scope not in RUNTIME_SCOPES:
            raise ValueError(f"unsupported runtime scope: {runtime_scope!r}")
        if runtime_statistic not in RUNTIME_STATISTICS:
            raise ValueError(
                f"unsupported runtime statistic: {runtime_statistic!r}"
            )
    elif benchmark_result_ids:
        raise ValueError(
            "benchmark result selections require a runtime scope and statistic"
        )

    selected_runs = _canonical_runs(runs)
    run_ids = tuple(run.run_id for run in selected_runs)
    evaluation_selections = _selection_map(
        evaluation_result_ids,
        run_ids=run_ids,
        description="evaluation result selections",
    )
    benchmark_selections = _selection_map(
        benchmark_result_ids,
        run_ids=run_ids,
        description="benchmark result selections",
    )
    normalized_waivers = _normalize_waivers(waivers, runtime=runtime)

    evaluations: dict[str, ResultRecord] = {}
    benchmarks: dict[str, ResultRecord] = {}
    accuracy_values: dict[str, float] = {}
    runtime_values: dict[str, float] = {}
    compatibility: dict[str, dict[str, object]] = {
        field: {} for field in ACCURACY_COMPATIBILITY_FIELDS
    }
    if runtime:
        compatibility.update(
            {field: {} for field in RUNTIME_COMPATIBILITY_FIELDS}
        )

    for run in selected_runs:
        evaluation = select_result(
            list_results(run, "evaluation"),
            result_id=evaluation_selections.get(run.run_id),
        )
        evaluation_payload = _payload(evaluation, expected_kind="evaluation")
        evaluations[run.run_id] = evaluation
        accuracy_values[run.run_id] = _accuracy_value(
            evaluation,
            evaluation_payload,
            accuracy_metric,
        )
        accuracy_observations = _accuracy_observations(
            run,
            evaluation_payload,
        )
        for field, observation in accuracy_observations.items():
            compatibility[field][run.run_id] = observation

        if runtime:
            assert runtime_scope is not None and runtime_statistic is not None
            benchmark = select_result(
                list_results(run, "benchmark"),
                result_id=benchmark_selections.get(run.run_id),
            )
            benchmark_payload = _payload(benchmark, expected_kind="benchmark")
            benchmarks[run.run_id] = benchmark
            runtime_values[run.run_id] = _runtime_value(
                benchmark,
                benchmark_payload,
                scope=runtime_scope,
                statistic=runtime_statistic,
            )
            runtime_observations = _runtime_observations(
                run,
                benchmark,
                benchmark_payload,
                scope=runtime_scope,
                statistic=runtime_statistic,
            )
            for field, observation in runtime_observations.items():
                compatibility[field][run.run_id] = observation

    validated_compatibility = _validate_compatibility(
        compatibility,
        run_ids=run_ids,
        waivers=normalized_waivers,
        runtime=runtime,
    )
    accuracy_ranks = _rank(accuracy_values, descending=True)
    runtime_ranks = _rank(runtime_values, descending=False) if runtime else {}

    rows = []
    for run in selected_runs:
        selected_checkpoint = run.selected_checkpoint
        assert selected_checkpoint is not None
        evaluation = evaluations[run.run_id]
        benchmark = benchmarks.get(run.run_id)
        rows.append(
            ComparisonRow(
                run_id=run.run_id,
                slug=run.manifest.slug,
                config_sha256=run.manifest.config.sha256,
                checkpoint_sha256=selected_checkpoint.sha256,
                evaluation_result_id=evaluation.result_id,
                accuracy_metric=accuracy_metric,
                accuracy_raw_key=KITTI_CAR_AP40_METRICS[accuracy_metric],
                accuracy_value=accuracy_values[run.run_id],
                accuracy_rank=accuracy_ranks[run.run_id],
                benchmark_result_id=(
                    None if benchmark is None else benchmark.result_id
                ),
                runtime_scope=runtime_scope,
                runtime_statistic=runtime_statistic,
                runtime_value=runtime_values.get(run.run_id),
                runtime_rank=runtime_ranks.get(run.run_id),
            )
        )

    return ComparisonReport(
        schema_version=COMPARISON_SCHEMA_VERSION,
        accuracy_metric=accuracy_metric,
        runtime_scope=runtime_scope,
        runtime_statistic=runtime_statistic,
        ranking_policy=_RANKING_POLICY,
        waivers=normalized_waivers,
        compatibility=validated_compatibility,
        rows=tuple(rows),
    )


def write_comparison_report(path: Path, report: ComparisonReport) -> None:
    """Durably write one derived comparison report to an explicit path."""
    if not isinstance(path, Path):
        raise TypeError("comparison output path must be a pathlib.Path")
    if not isinstance(report, ComparisonReport):
        raise TypeError("report must be a ComparisonReport")
    write_json_atomic(path, report.to_dict())


def load_comparison_report(path: Path) -> ComparisonReport:
    """Load and strictly validate one persisted comparison report."""
    if not isinstance(path, Path):
        raise TypeError("comparison report path must be a pathlib.Path")
    return ComparisonReport.from_dict(read_json_object(path))
