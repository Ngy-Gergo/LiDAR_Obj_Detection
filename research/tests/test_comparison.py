from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lidar_model_selection.checkpoints import CheckpointArtifact, TrainingOutputs
from lidar_model_selection.comparison import (
    KITTI_CAR_AP40_METRICS,
    CompatibilityWaiver,
    ComparisonReport,
    compare_runs,
    load_comparison_report,
    write_comparison_report,
)
from lidar_model_selection.provenance import EnvironmentInfo
from lidar_model_selection.results import (
    ResultRecord,
    binding_for_run,
    create_result,
    publish_result,
)
from lidar_model_selection.runs import (
    Run,
    TrainingState,
    build_dataset_identity,
    create_run,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COMPARE_TOOL = REPOSITORY_ROOT / "research" / "tools" / "compare.py"
_START = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)
_METRIC = "car_3d_ap40_moderate_strict"
_RAW_METRIC = KITTI_CAR_AP40_METRICS[_METRIC]


def _artifact(path: Path, *, digest: str, epoch: int | None) -> CheckpointArtifact:
    return CheckpointArtifact(
        path=str(path),
        sha256=digest,
        size_bytes=128,
        epoch=epoch,
        checkpoint_format="pytorch_zip",
        validation_profile="pytorch-zip-structural-v1",
    )


def _run(
    tmp_path: Path,
    *,
    slug: str,
    digest_character: str,
    dataset_version: str | None = "kitti-object-v1",
    semantic_partition: str | None = "KITTI validation",
    classes: tuple[str, ...] | None = ("Car",),
) -> Run:
    checkpoint_root = tmp_path / "external" / slug
    outputs = TrainingOutputs(
        final_checkpoint=_artifact(
            checkpoint_root / "epoch_4.pth",
            digest=digest_character * 64,
            epoch=4,
        ),
        selected_checkpoint=_artifact(
            checkpoint_root / "best_score.pth",
            digest=("0" if digest_character != "0" else "1") * 64,
            epoch=None,
        ),
    )
    tasks = None if classes is None else {"3d_detection": classes}
    dataset = build_dataset_identity(
        name="KITTI",
        version=dataset_version,
        root_reference="dataset:kitti-test",
        semantic_partition=semantic_partition,
        framework_key="test_dataloader",
        annotation_files=None,
        class_names=classes,
        tasks=tasks,
    )
    return create_run(
        tmp_path / "runs",
        slug=slug,
        config_bytes=b"model = dict(type='CenterPoint')\n",
        dataset=dataset,
        target_epoch=4,
        origin="historical_import",
        training_state=TrainingState(
            status="completed",
            attempts=(),
            outputs=outputs,
        ),
    )


def _environment(*, torch_version: str = "2.1.2") -> EnvironmentInfo:
    return EnvironmentInfo(
        python_version="3.10.14",
        python_implementation="CPython",
        platform="Linux-test",
        machine="x86_64",
        executable="/usr/bin/python3.10",
        packages=(
            ("mmcv", "2.1.0"),
            ("mmdet", "3.3.0"),
            ("mmdet3d", "1.4.0"),
            ("mmengine", "0.10.7"),
            ("torch", torch_version),
        ),
        torch_version=torch_version,
        cuda_version="12.1",
        cudnn_version="8902",
        gpu_available=True,
        gpu_devices=("NVIDIA Test GPU",),
    )


def _evaluation_payload(
    run: Run,
    score: object,
    *,
    profile_id: object = "mmengine_raw_scalar_metrics",
    profile_version: object = 1,
    source_record: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "kind": "evaluation",
        "metric_profile": (
            None
            if profile_id is None and profile_version is None
            else {
                "id": profile_id,
                "version": profile_version,
                "key": (
                    None
                    if profile_id is None or profile_version is None
                    else f"{profile_id}_v{profile_version}"
                ),
            }
        ),
        "semantic_partition": run.manifest.dataset.semantic_partition,
        "framework_key": run.manifest.dataset.framework_key,
        "metrics": {} if source_record is not None else {_RAW_METRIC: score},
        **({} if source_record is None else {"source_record": source_record}),
    }


def _methodology(*, version: int = 1) -> dict[str, object]:
    identifier = "mmdet3d_prediction_e2e_sync"
    return {
        "id": identifier,
        "version": version,
        "key": f"{identifier}_v{version}",
        "timing_scopes": {
            "prediction_ms": "model.test_step(batch)",
            "end_to_end_ms": "next(iterator) + model.test_step(batch)",
        },
        "synchronization": {"device": "CUDA"},
        "iterator_policy": "one shared iterator",
        "warmup_policy": "leading samples then reset peaks",
        "sample_policy": "consecutive samples without cycling",
        "statistics": {
            "unit": "milliseconds",
            "percentiles": "linear interpolation at (n - 1) * q",
            "standard_deviation": "population (ddof=0)",
        },
    }


def _benchmark_payload(
    run: Run,
    latency: object,
    *,
    hardware: str = "NVIDIA Test GPU",
    driver_version: str = "575.57.08",
    methodology: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "kind": "benchmark",
        "benchmark_schema_version": 2,
        "methodology": _methodology() if methodology is None else methodology,
        "workload": {
            "semantic_partition": run.manifest.dataset.semantic_partition,
            "framework_key": run.manifest.dataset.framework_key,
            "batch_size": 1,
            "num_workers": 0,
            "persistent_workers": False,
            "drop_last": False,
            "shuffle": False,
            "warmup_count": 10,
            "measured_sample_count": 100,
        },
        "hardware": {
            "device_type": "cuda",
            "logical_device_index": 0,
            "visible_device_count": 1,
            "device_name": hardware,
            "nvidia_driver_version": driver_version,
            "cuda_visible_devices": "0",
            "host": {
                "cpu_model": "Test CPU",
                "architecture": "x86_64",
                "os_class": "Linux",
            },
        },
        "precision": {
            "execution_policy": "torch_inference_mode_no_autocast",
            "inference_mode": True,
            "autocast_enabled_by_benchmark": False,
            "model_parameter_dtypes": ["torch.float32"],
        },
        "prediction_ms": {"p95_ms": float(latency) / 2.0},
        "end_to_end_ms": {"p95_ms": latency, "meets_20hz": True},
        "checkpoint": {"size_bytes": 128, "size_mib": 0.0001},
        "peak_memory": {"allocated_bytes": 1024, "reserved_bytes": 2048},
    }


def _record(
    run: Run,
    *,
    result_type: str,
    payload: dict[str, object],
    offset: int = 0,
    environment: EnvironmentInfo | None = None,
) -> ResultRecord:
    started = _START + timedelta(seconds=offset)
    record = create_result(
        result_type=result_type,
        binding=binding_for_run(run),
        status="succeeded",
        started_at=started,
        finished_at=started + timedelta(seconds=1),
        payload=payload,
        environment=_environment() if environment is None else environment,
    )
    publish_result(run, record)
    return record


def _publish_pair(
    run: Run,
    *,
    score: float,
    latency: float,
    offset: int = 0,
    hardware: str = "NVIDIA Test GPU",
) -> tuple[ResultRecord, ResultRecord]:
    evaluation = _record(
        run,
        result_type="evaluation",
        payload=_evaluation_payload(run, score),
        offset=offset,
    )
    benchmark = _record(
        run,
        result_type="benchmark",
        payload=_benchmark_payload(run, latency, hardware=hardware),
        offset=offset + 1,
    )
    return evaluation, benchmark


def test_exact_raw_projection_ranking_round_trip_and_durable_output(
    tmp_path: Path,
) -> None:
    first = _run(tmp_path, slug="alpha", digest_character="a")
    second = _run(tmp_path, slug="beta", digest_character="c")
    first_evaluation, first_benchmark = _publish_pair(
        first,
        score=61.0,
        latency=15.0,
    )
    second_evaluation, second_benchmark = _publish_pair(
        second,
        score=65.0,
        latency=20.0,
        offset=10,
    )

    report = compare_runs(
        (second, first),
        accuracy_metric=_METRIC,
        evaluation_result_ids={
            first.run_id: first_evaluation.result_id,
            second.run_id: second_evaluation.result_id,
        },
        runtime_scope="end_to_end_ms",
        runtime_statistic="p95_ms",
        benchmark_result_ids={
            first.run_id: first_benchmark.result_id,
            second.run_id: second_benchmark.result_id,
        },
    )

    by_run = {row.run_id: row for row in report.rows}
    assert tuple(row.run_id for row in report.rows) == tuple(
        sorted((first.run_id, second.run_id))
    )
    assert by_run[first.run_id].accuracy_raw_key == _RAW_METRIC
    assert by_run[first.run_id].accuracy_rank == 2
    assert by_run[second.run_id].accuracy_rank == 1
    assert by_run[first.run_id].runtime_rank == 1
    assert by_run[second.run_id].runtime_rank == 2
    assert dict(by_run[first.run_id].ap40) == {_METRIC: 61.0}
    latency_statistics = by_run[first.run_id].latency_statistics
    assert latency_statistics is not None
    assert {
        scope: dict(values)
        for scope, values in latency_statistics.items()
    } == {
        "prediction_ms": {"p95_ms": 7.5},
        "end_to_end_ms": {"p95_ms": 15.0},
    }
    assert by_run[first.run_id].peak_memory_allocated_bytes == 1024
    assert by_run[first.run_id].peak_memory_reserved_bytes == 2048
    assert by_run[first.run_id].checkpoint_size_bytes == 128
    assert by_run[first.run_id].meets_20hz is True
    assert ComparisonReport.from_dict(report.to_dict()) == report

    output = tmp_path / "derived" / "comparison.json"
    write_comparison_report(output, report)
    assert load_comparison_report(output) == report
    document = json.loads(output.read_text(encoding="utf-8"))
    assert {row["evaluation_result_id"] for row in document["rows"]} == {
        first_evaluation.result_id,
        second_evaluation.result_id,
    }
    assert {row["benchmark_result_id"] for row in document["rows"]} == {
        first_benchmark.result_id,
        second_benchmark.result_id,
    }


def test_selection_uses_sole_success_or_requires_an_exact_result_id(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path, slug="selection", digest_character="e")
    older = _record(
        run,
        result_type="evaluation",
        payload=_evaluation_payload(run, 55.0),
    )

    implicit = compare_runs((run,), accuracy_metric=_METRIC)
    assert implicit.rows[0].evaluation_result_id == older.result_id

    newer = _record(
        run,
        result_type="evaluation",
        payload=_evaluation_payload(run, 75.0),
        offset=20,
    )
    with pytest.raises(ValueError, match="multiple successful"):
        compare_runs((run,), accuracy_metric=_METRIC)

    explicit = compare_runs(
        (run,),
        accuracy_metric=_METRIC,
        evaluation_result_ids={run.run_id: older.result_id},
    )
    assert explicit.rows[0].accuracy_value == 55.0
    assert explicit.rows[0].evaluation_result_id != newer.result_id


def test_row_rejects_disagreement_between_ranking_and_plot_evidence(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path, slug="consistent", digest_character="d")
    _publish_pair(run, score=61.0, latency=15.0)
    report = compare_runs(
        (run,),
        accuracy_metric=_METRIC,
        runtime_scope="end_to_end_ms",
        runtime_statistic="p95_ms",
    )

    document = report.to_dict()
    document["rows"][0]["accuracy_value"] = 62.0  # type: ignore[index]
    with pytest.raises(ValueError, match="selected accuracy disagrees"):
        ComparisonReport.from_dict(document)

    document = report.to_dict()
    document["rows"][0]["latency_statistics"]["end_to_end_ms"][  # type: ignore[index]
        "p95_ms"
    ] = 16.0
    with pytest.raises(ValueError, match="selected runtime disagrees"):
        ComparisonReport.from_dict(document)


def test_unknown_metadata_requires_the_exact_field_waiver(tmp_path: Path) -> None:
    run = _run(
        tmp_path,
        slug="unknown-version",
        digest_character="1",
        dataset_version=None,
    )
    _record(
        run,
        result_type="evaluation",
        payload=_evaluation_payload(run, 60.0),
    )

    with pytest.raises(ValueError, match=r"accuracy\.dataset\.version"):
        compare_runs((run,), accuracy_metric=_METRIC)

    waiver = CompatibilityWaiver(
        "accuracy.dataset.version",
        "KITTI object dataset release label was not recorded",
    )
    report = compare_runs((run,), accuracy_metric=_METRIC, waivers=(waiver,))
    assert report.waivers == (waiver,)
    assert report.to_dict()["waivers"] == [waiver.to_dict()]


def test_metric_profile_mismatch_refuses_by_default_and_persists_waiver(
    tmp_path: Path,
) -> None:
    first = _run(tmp_path, slug="profile-one", digest_character="3")
    second = _run(tmp_path, slug="profile-two", digest_character="5")
    _record(
        first,
        result_type="evaluation",
        payload=_evaluation_payload(first, 60.0),
    )
    _record(
        second,
        result_type="evaluation",
        payload=_evaluation_payload(second, 61.0, profile_version=2),
    )

    with pytest.raises(ValueError, match=r"accuracy\.metric_profile\.version"):
        compare_runs((first, second), accuracy_metric=_METRIC)

    waiver = CompatibilityWaiver(
        "accuracy.metric_profile.version",
        "reviewed profile migration preserved this exact AP40 definition",
    )
    report = compare_runs(
        (first, second),
        accuracy_metric=_METRIC,
        waivers=(waiver,),
    )
    observations = report.to_dict()["compatibility"]
    assert set(observations[waiver.field].values()) == {1, 2}


def test_runtime_hardware_mismatch_requires_specific_waiver(tmp_path: Path) -> None:
    first = _run(tmp_path, slug="gpu-one", digest_character="7")
    second = _run(tmp_path, slug="gpu-two", digest_character="9")
    _publish_pair(first, score=60.0, latency=10.0, hardware="GPU A")
    _publish_pair(second, score=61.0, latency=11.0, hardware="GPU B")
    options = {
        "accuracy_metric": _METRIC,
        "runtime_scope": "end_to_end_ms",
        "runtime_statistic": "p95_ms",
    }

    with pytest.raises(ValueError, match=r"runtime\.hardware_class"):
        compare_runs((first, second), **options)

    waiver = CompatibilityWaiver(
        "runtime.hardware_class",
        "cross-device exploratory comparison requested by the reviewer",
    )
    report = compare_runs((first, second), waivers=(waiver,), **options)
    assert report.waivers == (waiver,)


def test_runtime_driver_mismatch_requires_exact_driver_waiver(tmp_path: Path) -> None:
    first = _run(tmp_path, slug="driver-one", digest_character="7")
    second = _run(tmp_path, slug="driver-two", digest_character="9")
    for index, (run, driver_version) in enumerate(
        ((first, "575.57.08"), (second, "570.169"))
    ):
        _record(
            run,
            result_type="evaluation",
            payload=_evaluation_payload(run, 60.0 + index),
        )
        _record(
            run,
            result_type="benchmark",
            payload=_benchmark_payload(
                run,
                10.0 + index,
                driver_version=driver_version,
            ),
        )
    options = {
        "accuracy_metric": _METRIC,
        "runtime_scope": "prediction_ms",
        "runtime_statistic": "p95_ms",
    }

    with pytest.raises(ValueError, match=r"runtime\.nvidia_driver_version"):
        compare_runs((first, second), **options)

    waiver = CompatibilityWaiver(
        "runtime.nvidia_driver_version",
        "reviewed driver-version mismatch for this exploratory comparison",
    )
    report = compare_runs((first, second), waivers=(waiver,), **options)
    observations = report.to_dict()["compatibility"][waiver.field]
    assert set(observations.values()) == {"570.169", "575.57.08"}
    assert ComparisonReport.from_dict(report.to_dict()) == report


def test_historical_missing_driver_remains_explicit_and_waivable(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path, slug="driver-unrecorded", digest_character="6")
    _record(
        run,
        result_type="evaluation",
        payload=_evaluation_payload(run, 60.0),
    )
    payload = _benchmark_payload(run, 12.0)
    payload["benchmark_schema_version"] = 1
    payload["hardware"].pop("nvidia_driver_version")  # type: ignore[union-attr]
    benchmark = _record(run, result_type="benchmark", payload=payload)
    historical_hardware = benchmark.payload["hardware"]
    assert "nvidia_driver_version" not in historical_hardware  # type: ignore[operator]
    options = {
        "accuracy_metric": _METRIC,
        "runtime_scope": "prediction_ms",
        "runtime_statistic": "p95_ms",
    }

    with pytest.raises(ValueError, match=r"runtime\.nvidia_driver_version"):
        compare_runs((run,), **options)

    waiver = CompatibilityWaiver(
        "runtime.nvidia_driver_version",
        "historical benchmark predates NVIDIA driver evidence",
    )
    report = compare_runs((run,), waivers=(waiver,), **options)
    observation = report.to_dict()["compatibility"][waiver.field][run.run_id]
    assert observation is None


def test_comparison_rejects_malformed_recorded_driver_evidence(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path, slug="driver-malformed", digest_character="5")
    _record(
        run,
        result_type="evaluation",
        payload=_evaluation_payload(run, 60.0),
    )
    payload = _benchmark_payload(run, 12.0)
    payload["hardware"]["nvidia_driver_version"] = "unknown"  # type: ignore[index]
    _record(run, result_type="benchmark", payload=payload)

    with pytest.raises(ValueError, match="dot-separated numeric version"):
        compare_runs(
            (run,),
            accuracy_metric=_METRIC,
            runtime_scope="prediction_ms",
            runtime_statistic="p95_ms",
        )


def test_comparison_rejects_schema_v2_result_missing_driver_evidence(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path, slug="driver-missing-v2", digest_character="4")
    _record(
        run,
        result_type="evaluation",
        payload=_evaluation_payload(run, 60.0),
    )
    payload = _benchmark_payload(run, 12.0)
    payload["hardware"].pop("nvidia_driver_version")  # type: ignore[union-attr]
    _record(run, result_type="benchmark", payload=payload)

    with pytest.raises(ValueError, match="schema version 2 or newer requires"):
        compare_runs(
            (run,),
            accuracy_metric=_METRIC,
            runtime_scope="prediction_ms",
            runtime_statistic="p95_ms",
            waivers=(
                CompatibilityWaiver(
                    "runtime.nvidia_driver_version",
                    "a waiver cannot fabricate required new evidence",
                ),
            ),
        )


def test_legacy_comparison_schema_v2_without_driver_field_still_loads(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path, slug="legacy-report", digest_character="3")
    _publish_pair(run, score=60.0, latency=12.0)
    report = compare_runs(
        (run,),
        accuracy_metric=_METRIC,
        runtime_scope="prediction_ms",
        runtime_statistic="p95_ms",
    )
    document = report.to_dict()
    document["schema_version"] = 2
    compatibility = document["compatibility"]
    compatibility.pop("runtime.nvidia_driver_version")  # type: ignore[union-attr]

    legacy = ComparisonReport.from_dict(document)

    assert legacy.schema_version == 2
    assert "runtime.nvidia_driver_version" not in legacy.compatibility
    output = tmp_path / "legacy-comparison-v2.json"
    write_comparison_report(output, legacy)
    assert load_comparison_report(output) == legacy


def test_e2e_host_mismatch_or_unknown_requires_persisted_waiver(
    tmp_path: Path,
) -> None:
    first = _run(tmp_path, slug="host-one", digest_character="1")
    second = _run(tmp_path, slug="host-two", digest_character="2")
    for run in (first, second):
        _record(
            run,
            result_type="evaluation",
            payload=_evaluation_payload(run, 60.0),
        )
    first_payload = _benchmark_payload(first, 12.0)
    second_payload = _benchmark_payload(second, 13.0)
    second_payload["hardware"]["host"]["cpu_model"] = "Slower Test CPU"  # type: ignore[index]
    _record(first, result_type="benchmark", payload=first_payload)
    _record(second, result_type="benchmark", payload=second_payload)
    options = {
        "accuracy_metric": _METRIC,
        "runtime_scope": "end_to_end_ms",
        "runtime_statistic": "p95_ms",
    }

    with pytest.raises(ValueError, match=r"runtime\.host_identity"):
        compare_runs((first, second), **options)
    waiver = CompatibilityWaiver(
        "runtime.host_identity", "reviewed cross-host E2E comparison"
    )
    report = compare_runs((first, second), waivers=(waiver,), **options)
    assert waiver in report.waivers

    unknown = _run(tmp_path, slug="host-unknown", digest_character="3")
    _record(
        unknown,
        result_type="evaluation",
        payload=_evaluation_payload(unknown, 59.0),
    )
    unknown_payload = _benchmark_payload(unknown, 14.0)
    unknown_payload["hardware"].pop("host")  # type: ignore[union-attr]
    _record(unknown, result_type="benchmark", payload=unknown_payload, environment=None)
    with pytest.raises(ValueError, match=r"runtime\.host_identity"):
        compare_runs((unknown,), **options)


def test_prediction_scope_does_not_require_matching_host_cpu(tmp_path: Path) -> None:
    first = _run(tmp_path, slug="prediction-host-one", digest_character="4")
    second = _run(tmp_path, slug="prediction-host-two", digest_character="5")
    for index, run in enumerate((first, second)):
        _record(
            run,
            result_type="evaluation",
            payload=_evaluation_payload(run, 60.0 + index),
        )
        payload = _benchmark_payload(run, 12.0 + index)
        payload["hardware"]["host"]["cpu_model"] = f"CPU {index}"  # type: ignore[index]
        _record(run, result_type="benchmark", payload=payload)
    report = compare_runs(
        (first, second),
        accuracy_metric=_METRIC,
        runtime_scope="prediction_ms",
        runtime_statistic="p95_ms",
    )
    assert len(report.rows) == 2


def test_runtime_synchronization_mismatch_is_timing_scope_incompatibility(
    tmp_path: Path,
) -> None:
    first = _run(tmp_path, slug="sync-one", digest_character="4")
    second = _run(tmp_path, slug="sync-two", digest_character="8")
    for run in (first, second):
        _record(
            run,
            result_type="evaluation",
            payload=_evaluation_payload(run, 60.0),
        )
    first_methodology = _methodology()
    second_methodology = _methodology()
    second_methodology["synchronization"] = {"device": "none"}
    _record(
        first,
        result_type="benchmark",
        payload=_benchmark_payload(
            first,
            12.0,
            methodology=first_methodology,
        ),
    )
    _record(
        second,
        result_type="benchmark",
        payload=_benchmark_payload(
            second,
            13.0,
            methodology=second_methodology,
        ),
    )

    with pytest.raises(ValueError, match=r"runtime\.timing_scope"):
        compare_runs(
            (first, second),
            accuracy_metric=_METRIC,
            runtime_scope="end_to_end_ms",
            runtime_statistic="p95_ms",
        )


def test_historical_source_projection_never_makes_unknown_profile_match(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path, slug="historical", digest_character="b")
    _record(
        run,
        result_type="evaluation",
        payload=_evaluation_payload(
            run,
            0.0,
            profile_id=None,
            profile_version=None,
            source_record={_METRIC: 52.5},
        ),
    )

    with pytest.raises(ValueError, match=r"accuracy\.metric_profile\.id"):
        compare_runs((run,), accuracy_metric=_METRIC)

    report = compare_runs(
        (run,),
        accuracy_metric=_METRIC,
        waivers=(
            CompatibilityWaiver(
                "accuracy.metric_profile.id",
                "historical evaluator profile ID was not recorded",
            ),
            CompatibilityWaiver(
                "accuracy.metric_profile.version",
                "historical evaluator profile version was not recorded",
            ),
        ),
    )
    assert report.rows[0].accuracy_value == 52.5


@pytest.mark.parametrize("nested_shape", [False, True])
def test_historical_runtime_methodology_needs_each_exact_unknown_waiver(
    tmp_path: Path,
    nested_shape: bool,
) -> None:
    run = _run(tmp_path, slug="historical-runtime", digest_character="6")
    _record(
        run,
        result_type="evaluation",
        payload=_evaluation_payload(run, 58.0),
    )
    source = {
        **(
            {"end_to_end_ms": {"p95_ms": 19.5, "meets_20hz": True}}
            if nested_shape
            else {"end_to_end_p95_ms": 19.5, "meets_20hz": True}
        ),
        "gpu_name": "NVIDIA Test GPU",
        "device_type": "cuda",
        "nvidia_driver_version": "575.57.08",
        "precision": "fp32",
        "batch_size": 1,
        "semantic_partition": "KITTI validation",
        "framework_key": "test_dataloader",
        "num_workers": 0,
        "persistent_workers": False,
        "drop_last": False,
        "shuffle": False,
        **(
            {"warmup_count": 10, "measured_sample_count": 100}
            if nested_shape
            else {"warmup": 10, "samples": 100}
        ),
        "cpu_model": "Test CPU",
        "architecture": "x86_64",
        "os_class": "Linux",
        "peak_memory_allocated_mb": 1.5,
        "peak_memory_reserved_mb": 2.5,
        "checkpoint_size_mb": 3.5,
    }
    _record(
        run,
        result_type="benchmark",
        payload={
            "kind": "benchmark",
            "methodology": None,
            "source_record": source,
        },
    )
    options = {
        "accuracy_metric": _METRIC,
        "runtime_scope": "end_to_end_ms",
        "runtime_statistic": "p95_ms",
    }

    with pytest.raises(ValueError, match=r"runtime\.methodology\.id"):
        compare_runs((run,), **options)

    waived_fields = (
        "runtime.methodology.id",
        "runtime.methodology.version",
        "runtime.statistic",
        "runtime.timing_scope",
        "runtime.warmup_measurement_policy",
    )
    report = compare_runs(
        (run,),
        waivers=tuple(
            CompatibilityWaiver(field, f"historical evidence lacks {field}")
            for field in waived_fields
        ),
        **options,
    )
    row = report.rows[0]
    assert row.runtime_value == 19.5
    assert dict(row.latency_statistics["end_to_end_ms"]) == {  # type: ignore[index]
        "p95_ms": 19.5
    }
    assert row.peak_memory_allocated_bytes == round(1.5 * 1024**2)
    assert row.peak_memory_reserved_bytes == round(2.5 * 1024**2)
    assert row.checkpoint_size_bytes == round(3.5 * 1024**2)
    assert row.meets_20hz is True
    assert {waiver.field for waiver in report.waivers} == set(waived_fields)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "61"])
def test_accuracy_projection_rejects_nonfinite_or_nonnumeric_values(
    tmp_path: Path,
    value: object,
) -> None:
    run = _run(tmp_path, slug="bad-metric", digest_character="d")
    if isinstance(value, float) and not math.isfinite(value):
        # ResultRecord itself correctly rejects non-finite JSON. Fault-inject a
        # record file to ensure comparison cannot consume corrupt old evidence.
        valid = _record(
            run,
            result_type="evaluation",
            payload=_evaluation_payload(run, 1.0),
        )
        record_path = run.paths.evaluation / valid.result_id / "result.json"
        text = record_path.read_text(encoding="utf-8").replace(
            f'"{_RAW_METRIC}": 1.0',
            f'"{_RAW_METRIC}": {"NaN" if math.isnan(value) else "Infinity"}',
        )
        record_path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            compare_runs((run,), accuracy_metric=_METRIC)
        return

    _record(
        run,
        result_type="evaluation",
        payload=_evaluation_payload(run, value),
    )
    with pytest.raises(TypeError, match="real number"):
        compare_runs((run,), accuracy_metric=_METRIC)


def test_report_rejects_extra_fields_and_mutable_compatibility(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path, slug="strict", digest_character="f")
    _record(
        run,
        result_type="evaluation",
        payload=_evaluation_payload(run, 60.0),
    )
    report = compare_runs((run,), accuracy_metric=_METRIC)
    serialized = report.to_dict()
    serialized["extra"] = True
    with pytest.raises(ValueError, match="extra"):
        ComparisonReport.from_dict(serialized)
    with pytest.raises(TypeError):
        report.compatibility["new"] = {}  # type: ignore[index]


def test_compare_cli_is_thin_and_writes_the_requested_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = _run(tmp_path, slug="cli", digest_character="2")
    evaluation = _record(
        run,
        result_type="evaluation",
        payload=_evaluation_payload(run, 63.0),
    )
    specification = importlib.util.spec_from_file_location("compare_tool", COMPARE_TOOL)
    assert specification is not None and specification.loader is not None
    tool = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(tool)
    monkeypatch.setattr(tool, "DEFAULT_RUNS_ROOT", run.paths.root.parent)
    output = tmp_path / "reports" / "cli.json"

    status = tool.main(
        [
            "--run",
            run.run_id,
            "--evaluation-result",
            f"{run.run_id}={evaluation.result_id}",
            "--accuracy-metric",
            _METRIC,
            "--output",
            str(output),
        ]
    )

    assert status == 0
    assert load_comparison_report(output).rows[0].run_id == run.run_id
    assert "REPORT:" in capsys.readouterr().out


def test_comparison_import_does_not_load_ml_frameworks() -> None:
    script = """
import sys
import lidar_model_selection.comparison
for prefix in ('torch', 'mmengine', 'mmcv', 'mmdet', 'mmdet3d'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        env={"PYTHONPATH": str(REPOSITORY_ROOT / "research" / "src")},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
