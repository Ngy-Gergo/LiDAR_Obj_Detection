from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import lidar_model_selection.pipeline as pipeline
from lidar_model_selection.pipeline import (
    PipelineRequest,
    load_pipeline_record,
    run_pipeline,
)
from lidar_model_selection.preflight import PreflightReport


def _run(tmp_path: Path, slug: str = "explicit") -> SimpleNamespace:
    root = tmp_path / "runs" / slug
    root.mkdir(parents=True)
    return SimpleNamespace(
        run_id=f"20260824T120000Z-{slug}-" + "a" * 24,
        paths=SimpleNamespace(root=root),
    )


def _install_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, slug: str
) -> tuple[SimpleNamespace, list[object]]:
    run = _run(tmp_path, slug)
    completed = SimpleNamespace(
        run_id=run.run_id,
        paths=run.paths,
        completed=True,
    )
    run.completed = completed
    calls: list[object] = []

    def create(slug: str, epoch: int, *, source_config: Path | None):
        calls.append(("create", slug, epoch, source_config))
        return run

    monkeypatch.setattr(pipeline, "create_training_run", create)
    monkeypatch.setattr(
        pipeline,
        "preflight_run",
        lambda supplied, **kwargs: calls.append(("preflight", kwargs))
        or PreflightReport(
            run_id=run.run_id,
            operation="pipeline",
            config_path="config.py",
            dataset_root="data",
            annotation_paths=("train.pkl", "val.pkl"),
            sample_checked=kwargs["sample_check"],
        ),
    )
    def train(supplied: object):
        assert supplied is run
        calls.append("train")
        return completed

    def evaluate(supplied: object):
        assert supplied is completed
        calls.append("evaluate")
        return SimpleNamespace(status="succeeded", result_id="evaluation-id")

    def benchmark(supplied: object, **kwargs: object):
        assert supplied is completed
        calls.append(("benchmark", kwargs))
        return SimpleNamespace(status="succeeded", result_id="benchmark-id")

    monkeypatch.setattr(pipeline, "execute_training", train)
    monkeypatch.setattr(pipeline, "evaluate_run", evaluate)
    monkeypatch.setattr(pipeline, "benchmark_run", benchmark)
    return run, calls


def test_pipeline_explicit_config_reuses_public_operations_and_pins_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, calls = _install_success(tmp_path, monkeypatch, slug="loss-ablation")
    source = tmp_path / "ablation.py"
    request = PipelineRequest(
        slug="loss-ablation",
        target_epoch=4,
        source_config=source,
        benchmark_warmup=3,
        benchmark_samples=7,
        sample_check=False,
    )

    completed, record = run_pipeline(request)
    assert completed is run.completed
    assert calls == [
        ("create", "loss-ablation", 4, source),
        ("preflight", {"operation": "pipeline", "sample_check": False}),
        "train",
        "evaluate",
        ("benchmark", {"warmup": 3, "samples": 7}),
    ]
    assert record.evaluation_result_id == "evaluation-id"
    assert record.benchmark_result_id == "benchmark-id"
    path = run.paths.root / "pipeline" / f"{record.pipeline_id}.json"
    assert load_pipeline_record(path) == record


def test_failed_evaluation_is_persisted_and_benchmark_is_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, calls = _install_success(tmp_path, monkeypatch, slug="pillar02")
    monkeypatch.setattr(
        pipeline,
        "evaluate_run",
        lambda supplied: calls.append("evaluate")
        or SimpleNamespace(status="failed", result_id="failed-evaluation"),
    )
    with pytest.raises(RuntimeError, match="did not succeed"):
        run_pipeline(PipelineRequest(slug="pillar02", target_epoch=2))

    records = tuple((run.paths.root / "pipeline").glob("*.json"))
    assert len(records) == 1
    record = load_pipeline_record(records[0])
    assert record.status == "failed"
    assert record.evaluation_result_id == "failed-evaluation"
    assert record.benchmark_result_id is None
    assert not any(
        isinstance(call, tuple) and call[0] == "benchmark" for call in calls
    )


def test_pipeline_record_strictly_validates_identity_fields_and_time_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _ = _install_success(tmp_path, monkeypatch, slug="pillar02")
    _, record = run_pipeline(PipelineRequest(slug="pillar02", target_epoch=2))
    with pytest.raises(ValueError, match="schema version"):
        replace(record, schema_version=True)
    with pytest.raises(ValueError, match="evaluation result ID"):
        replace(record, evaluation_result_id=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="precedes"):
        replace(
            record,
            started_at="2026-08-24T12:00:00.900000Z",
            finished_at="2026-08-24T12:00:00.100000Z",
        )


def test_run_cli_builds_explicit_config_request_without_execution_logic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tool_path = Path(__file__).parents[1] / "tools" / "run.py"
    specification = importlib.util.spec_from_file_location("run_tool", tool_path)
    assert specification is not None and specification.loader is not None
    tool = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(tool)
    observed: list[PipelineRequest] = []
    run = _run(tmp_path)
    record = SimpleNamespace(
        pipeline_id="pipeline-id",
        evaluation_result_id="evaluation-id",
        benchmark_result_id="benchmark-id",
    )
    monkeypatch.setattr(
        tool,
        "run_pipeline",
        lambda request: observed.append(request) or (run, record),
    )
    config = tmp_path / "ablation.py"
    status = tool.main(
        [
            "--config",
            str(config),
            "--name",
            "my-ablation",
            "--max-epochs",
            "9",
            "--warmup",
            "2",
            "--samples",
            "5",
        ]
    )
    assert status == 0
    assert observed == [
        PipelineRequest(
            slug="my-ablation",
            target_epoch=9,
            source_config=config,
            benchmark_warmup=2,
            benchmark_samples=5,
        )
    ]
    assert f"run={run.run_id}" in capsys.readouterr().out
