from __future__ import annotations

import json
import re
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import lidar_model_selection.results as results_module
from lidar_model_selection.checkpoints import CheckpointArtifact, TrainingOutputs
from lidar_model_selection.provenance import EnvironmentInfo
from lidar_model_selection.results import (
    ResultBinding,
    ResultBindingMismatch,
    ResultFailure,
    ResultRecord,
    binding_for_run,
    create_result,
    create_result_id,
    list_results,
    load_result,
    publish_result,
    select_result,
    verify_result_binding,
)
from lidar_model_selection.runs import (
    Run,
    TrainingState,
    build_dataset_identity,
    create_run,
)


_START = datetime(2026, 8, 18, 12, 0, 0, 123456, tzinfo=timezone.utc)
_CONFIG_SHA256 = "a" * 64
_CHECKPOINT_SHA256 = "b" * 64
_RUN_ID = "20260818T120000Z-result-run-" + "1" * 24


def _binding(
    *,
    run: Run | None = None,
    run_id: str = _RUN_ID,
    config_sha256: str = _CONFIG_SHA256,
    checkpoint_sha256: str = _CHECKPOINT_SHA256,
) -> ResultBinding:
    if run is not None:
        assert run.selected_checkpoint is not None
        run_id = run.run_id
        config_sha256 = run.manifest.config.sha256
        checkpoint_sha256 = run.selected_checkpoint.sha256
    return ResultBinding(run_id, config_sha256, checkpoint_sha256)


def _completed_run(
    tmp_path: Path,
    *,
    token_character: str = "2",
) -> Run:
    run_id = f"20260818T120000Z-result-run-{token_character * 24}"
    final = CheckpointArtifact(
        path="training/epoch_4.pth",
        sha256="c" * 64,
        size_bytes=100,
        epoch=4,
        checkpoint_format="pytorch_zip",
        validation_profile="pytorch-zip-structural-v1",
    )
    selected = CheckpointArtifact(
        path="training/best_score.pth",
        sha256="d" * 64,
        size_bytes=100,
        epoch=None,
        checkpoint_format="pytorch_zip",
        validation_profile="pytorch-zip-structural-v1",
    )
    dataset = build_dataset_identity(
        name=None,
        version=None,
        root_reference=None,
        semantic_partition=None,
        framework_key=None,
        annotation_files=None,
        class_names=None,
        tasks=None,
    )
    return create_run(
        tmp_path / "runs",
        slug="result-run",
        run_id=run_id,
        config_bytes=b"model = dict()\n",
        dataset=dataset,
        target_epoch=4,
        origin="historical_import",
        training_state=TrainingState(
            status="completed",
            attempts=(),
            outputs=TrainingOutputs(final, selected),
        ),
    )


def _environment() -> EnvironmentInfo:
    return EnvironmentInfo(
        python_version="3.10.14",
        python_implementation="CPython",
        platform="Linux-test",
        machine="x86_64",
        executable="/usr/bin/python3.10",
        packages=(("pytest", "9.1.1"),),
        torch_version=None,
        cuda_version=None,
        cudnn_version=None,
        gpu_available=None,
        gpu_devices=(),
    )


def _record(
    *,
    run: Run | None = None,
    result_type: str = "evaluation",
    status: str = "succeeded",
    offset: int = 0,
    payload: dict[str, object] | None = None,
) -> ResultRecord:
    started = _START + timedelta(seconds=offset)
    failure = (
        None
        if status == "succeeded"
        else ResultFailure("RuntimeError", "evaluation failed", "trace")
    )
    return create_result(
        result_type=result_type,
        binding=_binding(run=run),
        status=status,
        started_at=started,
        finished_at=started + timedelta(seconds=1),
        payload={"metrics": {"score": 0.5}, "samples": [1, 2]}
        if payload is None
        else payload,
        environment=_environment(),
        failure=failure,
    )


def test_result_ids_contain_utc_type_and_at_least_96_random_bits() -> None:
    result_id = create_result_id("evaluation", timestamp=_START)

    match = re.fullmatch(
        r"20260818T120000123456Z-evaluation-([0-9a-f]+)",
        result_id,
    )
    assert match is not None
    assert len(match.group(1)) >= 24


def test_repeated_result_creation_always_gets_a_fresh_id() -> None:
    first = _record()
    second = _record()

    assert first.result_id != second.result_id
    assert first.binding == second.binding


def test_evidence_round_trips_and_is_deeply_immutable() -> None:
    result = _record()
    mismatch = ResultBindingMismatch("run_id", "expected", "actual")

    assert ResultBinding.from_dict(result.binding.to_dict()) == result.binding
    assert ResultBindingMismatch.from_dict(mismatch.to_dict()) == mismatch
    assert ResultRecord.from_dict(result.to_dict()) == result
    assert result.started_at.tzinfo == timezone.utc
    with pytest.raises(FrozenInstanceError):
        result.status = "failed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        result.payload["new"] = 1  # type: ignore[index]
    nested = result.payload["metrics"]
    assert isinstance(nested, dict) is False
    with pytest.raises(TypeError):
        nested["score"] = 1.0  # type: ignore[index]


def test_to_dict_returns_detached_mutable_json_values() -> None:
    result = _record()

    serialized = result.to_dict()
    payload = serialized["payload"]
    assert isinstance(payload, dict)
    payload["samples"].append(3)  # type: ignore[union-attr]

    assert result.payload["samples"] == (1, 2)


@pytest.mark.parametrize(
    "payload",
    [
        {"value": float("nan")},
        {"value": float("inf")},
        {1: "non-string key"},
        {"path": Path("not-json")},
    ],
)
def test_result_payload_must_be_strict_json(
    payload: dict[object, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _record(payload=payload)  # type: ignore[arg-type]


def test_result_payload_rejects_reference_cycles() -> None:
    payload: dict[str, object] = {}
    payload["self"] = payload

    with pytest.raises(ValueError, match="cycle"):
        _record(payload=payload)


def test_status_and_failure_evidence_are_consistent() -> None:
    succeeded = _record()
    failed = _record(status="failed")

    assert succeeded.successful is True
    assert succeeded.failure is None
    assert failed.successful is False
    assert failed.failure == ResultFailure(
        "RuntimeError",
        "evaluation failed",
        "trace",
    )
    with pytest.raises(ValueError, match="must not contain failure"):
        replace(succeeded, failure=ResultFailure("Error", "bad"))
    with pytest.raises(ValueError, match="must contain failure"):
        replace(failed, failure=None)


def test_result_rejects_nonterminal_or_reversed_timestamps() -> None:
    result = _record()

    with pytest.raises(ValueError, match="status"):
        replace(result, status="running")
    with pytest.raises(ValueError, match="precede"):
        replace(
            result,
            started_at=result.finished_at,
            finished_at=result.started_at,
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(result, started_at=result.started_at.replace(tzinfo=None))


def test_result_binding_requires_exact_hashes_and_safe_run_id() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        _binding(config_sha256="not-a-hash")
    with pytest.raises(ValueError, match="run_id"):
        _binding(run_id="../run")


def test_verify_result_binding_reports_every_difference() -> None:
    binding = _binding(
        run_id="20260818T120000Z-other-run-" + "3" * 24,
        config_sha256="c" * 64,
        checkpoint_sha256="d" * 64,
    )

    mismatches = verify_result_binding(
        binding,
        run_id=_binding().run_id,
        config_sha256=_CONFIG_SHA256,
        selected_checkpoint_sha256=_CHECKPOINT_SHA256,
    )

    assert [mismatch.field for mismatch in mismatches] == [
        "run_id",
        "config_sha256",
        "checkpoint_sha256",
    ]
    assert verify_result_binding(
        _binding(),
        run_id=_binding().run_id,
        config_sha256=_CONFIG_SHA256,
        selected_checkpoint_sha256=_CHECKPOINT_SHA256,
    ) == ()


def test_publish_and_load_result_transactionally(tmp_path: Path) -> None:
    run = _completed_run(tmp_path)
    result = _record(run=run)

    published = publish_result(run, result)

    assert published == run.paths.evaluation / result.result_id
    assert set(published.iterdir()) == {published / "result.json"}
    assert load_result(run, "evaluation", result.result_id) == result
    raw = json.loads((published / "result.json").read_text(encoding="utf-8"))
    assert raw["binding"]["checkpoint_sha256"] == run.selected_checkpoint.sha256
    assert not list(run.paths.evaluation.glob(".*.staging-*"))


def test_publication_is_exclusive_and_preserves_existing_result(
    tmp_path: Path,
) -> None:
    run = _completed_run(tmp_path)
    result = _record(run=run)
    destination = publish_result(run, result)
    original = (destination / "result.json").read_bytes()

    with pytest.raises(FileExistsError):
        publish_result(run, result)

    assert (destination / "result.json").read_bytes() == original
    assert not list(run.paths.evaluation.glob(".*.staging-*"))


def test_failed_publication_cleans_only_its_staging_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _completed_run(tmp_path)
    unrelated = run.paths.evaluation / "keep"
    unrelated.mkdir()

    def fail_publication(staging: Path, destination: Path) -> None:
        raise OSError("simulated publication failure")

    monkeypatch.setattr(
        results_module,
        "publish_directory_exclusive",
        fail_publication,
    )

    with pytest.raises(OSError, match="simulated"):
        publish_result(run, _record(run=run))

    assert unrelated.is_dir()
    assert set(run.paths.evaluation.iterdir()) == {unrelated}


def test_publication_requires_exact_run_binding_and_derives_result_directory(
    tmp_path: Path,
) -> None:
    run = _completed_run(tmp_path)
    assert binding_for_run(run) == _binding(run=run)
    with pytest.raises(ValueError, match="binding does not match"):
        publish_result(run, _record())

    benchmark = _record(run=run, result_type="benchmark")
    assert publish_result(run, benchmark) == (
        run.paths.benchmark / benchmark.result_id
    )


def test_publication_rejects_an_incomplete_run(tmp_path: Path) -> None:
    run = _completed_run(tmp_path)
    # Fault-inject a stale/incomplete in-memory run to exercise the publication
    # guard; normal RunManifest construction rejects this combination earlier.
    object.__setattr__(run.manifest, "training", TrainingState.pending())

    with pytest.raises(ValueError, match="completed run"):
        publish_result(run, _record())


def test_list_results_is_run_local_and_deterministic(tmp_path: Path) -> None:
    first_run = _completed_run(tmp_path / "one", token_character="4")
    second_run = _completed_run(tmp_path / "two", token_character="5")
    later = _record(run=first_run, offset=10)
    earlier = _record(run=first_run, offset=1)
    external = _record(run=second_run, offset=20)
    publish_result(first_run, later)
    publish_result(first_run, earlier)
    publish_result(second_run, external)

    listed = list_results(first_run, "evaluation")

    assert [record.result_id for record in listed] == sorted(
        [later.result_id, earlier.result_id]
    )
    assert external not in listed
    assert list_results(first_run, "benchmark") == ()


def test_list_and_load_reject_symlinks_and_corrupt_records(
    tmp_path: Path,
) -> None:
    run = _completed_run(tmp_path)
    result = _record(run=run)
    destination = publish_result(run, result)
    record_path = destination / "result.json"
    record_path.unlink()
    record_path.symlink_to(tmp_path / "outside.json")
    (tmp_path / "outside.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="symlink"):
        load_result(run, "evaluation", result.result_id)
    with pytest.raises(ValueError, match="symlink"):
        list_results(run, "evaluation")


def test_load_rejects_record_id_mismatch(tmp_path: Path) -> None:
    run = _completed_run(tmp_path)
    result = _record(run=run)
    destination = publish_result(run, result)
    different = _record(run=run, offset=2)
    (destination / "result.json").write_text(
        json.dumps(different.to_dict()),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        load_result(run, "evaluation", result.result_id)


def test_load_and_list_reject_a_tampered_binding(tmp_path: Path) -> None:
    run = _completed_run(tmp_path)
    result = _record(run=run)
    destination = publish_result(run, result)
    record_path = destination / "result.json"
    data = json.loads(record_path.read_text(encoding="utf-8"))
    data["binding"]["checkpoint_sha256"] = "e" * 64
    record_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="binding does not match"):
        load_result(run, "evaluation", result.result_id)
    with pytest.raises(ValueError, match="binding does not match"):
        list_results(run, "evaluation")


def test_implicit_selection_requires_exactly_one_successful_result() -> None:
    succeeded = _record(offset=1)
    failed = _record(status="failed", offset=2)

    assert select_result((failed, succeeded)) == succeeded
    with pytest.raises(ValueError, match="no successful"):
        select_result((failed,))
    with pytest.raises(ValueError, match="multiple successful"):
        select_result((succeeded, _record(offset=3)))


def test_explicit_selection_uses_the_exact_id_and_never_the_latest() -> None:
    older = _record(offset=1)
    newer = _record(offset=20)

    assert select_result((newer, older), result_id=older.result_id) == older
    with pytest.raises(KeyError, match="not found"):
        select_result((newer,), result_id=older.result_id)


def test_explicit_selection_rejects_a_failed_result() -> None:
    failed = _record(status="failed")

    with pytest.raises(ValueError, match="not successful"):
        select_result((failed,), result_id=failed.result_id)
