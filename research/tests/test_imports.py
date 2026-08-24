from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import pytest

from lidar_model_selection.imports import (
    import_historical_run,
    read_dataset_identity,
)
from lidar_model_selection.results import list_results
from lidar_model_selection.runs import (
    build_dataset_identity,
    load_run,
    update_training_state,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
IMPORTS_MODULE = (
    REPOSITORY_ROOT
    / "research"
    / "src"
    / "lidar_model_selection"
    / "imports.py"
)
IMPORT_TOOL = REPOSITORY_ROOT / "research" / "tools" / "import_run.py"


def _write_checkpoint(path: Path, *, payload: bytes = b"state") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", compression=ZIP_STORED) as archive:
        archive.writestr("archive/data.pkl", payload)
        archive.writestr("archive/version", b"3\n")
        archive.writestr("archive/data/0", b"storage")
    return path


def _dataset():
    return build_dataset_identity(
        name="KITTI",
        version=None,
        root_reference="historical:kitti",
        semantic_partition="KITTI validation",
        framework_key="test_dataloader",
        annotation_files=None,
        class_names=("Car",),
        tasks={"3d_detection": ("Car",)},
    )


def _sources(tmp_path: Path) -> dict[str, Path]:
    config = tmp_path / "old" / "effective.py"
    config.parent.mkdir(parents=True)
    config.write_bytes(b"model = dict(type='CenterPoint')\n")
    final = _write_checkpoint(tmp_path / "old" / "epoch_20.pth", payload=b"final")
    selected = _write_checkpoint(
        tmp_path / "old" / "best_moderate.pth",
        payload=b"selected",
    )
    evaluation = tmp_path / "old" / "metrics.json"
    evaluation.write_text(
        json.dumps(
            {
                "checkpoint_path": "research/experiments/model/best.pth",
                "test_success": True,
                "car_3d_ap40_moderate_strict": 66.9,
            }
        ),
        encoding="utf-8",
    )
    benchmark = tmp_path / "old" / "latency.json"
    benchmark.write_text(
        json.dumps(
            {
                "checkpoint": "research/experiments/model/best.pth",
                "success": True,
                "prediction_p95_ms": 7.5,
            }
        ),
        encoding="utf-8",
    )
    return {
        "config": config,
        "final": final,
        "selected": selected,
        "evaluation": evaluation,
        "benchmark": benchmark,
    }


def _import(tmp_path: Path, **overrides: object):
    sources = _sources(tmp_path)
    arguments: dict[str, object] = {
        "slug": "historical-centerpoint",
        "config_path": sources["config"],
        "final_checkpoint_path": sources["final"],
        "selected_checkpoint_path": sources["selected"],
        "dataset_identity": _dataset(),
        "evaluation_json": sources["evaluation"],
        "benchmark_json": sources["benchmark"],
    }
    arguments.update(overrides)
    return import_historical_run(tmp_path / "runs", **arguments), sources


def test_explicit_import_creates_terminal_run_without_copying_checkpoints(
    tmp_path: Path,
) -> None:
    imported, sources = _import(tmp_path)
    reloaded = load_run(imported.paths.root)
    outputs = reloaded.manifest.training.outputs

    assert reloaded.manifest.origin == "historical_import"
    assert reloaded.manifest.target_epoch == 20
    assert reloaded.manifest.training.status == "completed"
    assert reloaded.manifest.training.attempts == ()
    assert reloaded.manifest.resumable is False
    assert reloaded.manifest.history_complete is False
    assert reloaded.manifest.parent_run_id is None
    assert reloaded.manifest.code_provenance is None
    assert reloaded.manifest.environment is None
    assert reloaded.manifest.training_compatibility is None
    assert reloaded.paths.config.read_bytes() == sources["config"].read_bytes()
    assert list(reloaded.paths.training.iterdir()) == []
    assert outputs is not None
    assert outputs.final_checkpoint.path == os.fspath(sources["final"].absolute())
    assert outputs.selected_checkpoint.path == os.fspath(
        sources["selected"].absolute()
    )
    assert outputs.final_checkpoint.sha256 == hashlib.sha256(
        sources["final"].read_bytes()
    ).hexdigest()

    with pytest.raises(ValueError, match="terminal"):
        update_training_state(
            reloaded,
            reloaded.manifest.training,
            expected_revision=0,
        )


def test_historical_results_are_fresh_immutable_run_owned_evidence(
    tmp_path: Path,
) -> None:
    imported, sources = _import(tmp_path)
    evaluation = list_results(imported, "evaluation")
    benchmark = list_results(imported, "benchmark")

    assert len(evaluation) == 1
    assert len(benchmark) == 1
    evaluation_record = evaluation[0]
    benchmark_record = benchmark[0]
    assert evaluation_record.binding.run_id == imported.run_id
    assert evaluation_record.binding.config_sha256 == imported.manifest.config.sha256
    assert (
        evaluation_record.binding.checkpoint_sha256
        == imported.selected_checkpoint.sha256
    )
    assert evaluation_record.provenance is None
    assert evaluation_record.environment is None
    assert evaluation_record.payload["source_record"] == {
        "checkpoint_path": "research/experiments/model/best.pth",
        "test_success": True,
        "car_3d_ap40_moderate_strict": 66.9,
    }
    assert evaluation_record.payload["metric_profile"] is None
    historical = evaluation_record.payload["historical_import"]
    assert historical["result_record_time_semantics"] == (
        "import_operation_not_measurement"
    )
    assert historical["imported_at"].endswith("Z")
    association = historical["association"]
    assert association["basis"] == "recorded_checkpoint_path"
    assert association["source_field"] == "checkpoint_path"
    assert association["recorded_checkpoint_path"].endswith("best.pth")
    assert association["checkpoint_sha256_observed_at"] == "import"
    assert association["checkpoint_sha256_observed_at_measurement"] is False
    source_json = historical["source_json"]
    assert source_json["path"] == os.fspath(sources["evaluation"].absolute())
    assert source_json["sha256"] == hashlib.sha256(
        sources["evaluation"].read_bytes()
    ).hexdigest()
    assert benchmark_record.payload["methodology"] is None
    assert benchmark_record.provenance is None
    assert benchmark_record.environment is None

    with pytest.raises(TypeError):
        evaluation_record.payload["source_record"]["new"] = True


def test_explicit_association_is_used_when_source_has_no_checkpoint_path(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path)
    sources["evaluation"].write_text(
        json.dumps(
            {
                "test_success": True,
                "metric_profile": {
                    "id": "old-kitti-profile",
                    "version": 2,
                    "key": "old-kitti-profile-v2",
                },
                "metrics": {"score": 0.5},
            }
        ),
        encoding="utf-8",
    )
    imported = import_historical_run(
        tmp_path / "runs",
        slug="historical",
        config_path=sources["config"],
        final_checkpoint_path=sources["final"],
        selected_checkpoint_path=sources["selected"],
        dataset_identity=_dataset(),
        evaluation_json=sources["evaluation"],
    )
    record = list_results(imported, "evaluation")[0]
    association = record.payload["historical_import"]["association"]

    assert association["basis"] == "explicit_import_specification"
    assert association["source_field"] is None
    assert association["recorded_checkpoint_path"] is None
    assert record.payload["metric_profile"] == {
        "id": "old-kitti-profile",
        "version": 2,
        "key": "old-kitti-profile-v2",
    }


def test_unsuccessful_source_is_preserved_as_failed_result(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    sources["evaluation"].write_text(
        json.dumps({"test_success": False, "error_message": "old failure"}),
        encoding="utf-8",
    )
    imported = import_historical_run(
        tmp_path / "runs",
        slug="historical",
        config_path=sources["config"],
        final_checkpoint_path=sources["final"],
        selected_checkpoint_path=sources["selected"],
        dataset_identity=_dataset(),
        evaluation_json=sources["evaluation"],
    )
    record = list_results(imported, "evaluation")[0]

    assert record.status == "failed"
    assert record.failure is not None
    assert record.failure.message == "old failure"
    assert record.payload["source_record"]["test_success"] is False


@pytest.mark.parametrize("source_name", ["config", "evaluation", "benchmark"])
def test_import_rejects_symlinked_small_source_files_before_publication(
    tmp_path: Path,
    source_name: str,
) -> None:
    sources = _sources(tmp_path)
    target = sources[source_name]
    link = tmp_path / f"{source_name}-link"
    link.symlink_to(target)
    keyword = {
        "config": "config_path",
        "evaluation": "evaluation_json",
        "benchmark": "benchmark_json",
    }[source_name]

    with pytest.raises(ValueError, match="symlink"):
        _import(tmp_path / "attempt", **{keyword: link})
    assert not (tmp_path / "attempt" / "runs").exists()


@pytest.mark.parametrize("checkpoint_name", ["final", "selected"])
def test_import_rejects_corrupt_or_symlinked_checkpoints_before_publication(
    tmp_path: Path,
    checkpoint_name: str,
) -> None:
    sources = _sources(tmp_path)
    checkpoint = sources[checkpoint_name]
    if checkpoint_name == "final":
        checkpoint.write_bytes(b"not a checkpoint")
    else:
        original = checkpoint
        link = tmp_path / "selected-link.pth"
        link.symlink_to(original)
        checkpoint = link
    keyword = f"{checkpoint_name}_checkpoint_path"

    with pytest.raises(ValueError):
        _import(tmp_path / "attempt", **{keyword: checkpoint})
    assert not (tmp_path / "attempt" / "runs").exists()


def test_import_rejects_missing_input_and_noncanonical_final_name(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path)
    with pytest.raises(FileNotFoundError):
        import_historical_run(
            tmp_path / "missing-runs",
            slug="historical",
            config_path=tmp_path / "missing-config.py",
            final_checkpoint_path=sources["final"],
            selected_checkpoint_path=sources["selected"],
            dataset_identity=_dataset(),
        )

    malformed = _write_checkpoint(tmp_path / "old" / "epoch_020.pth")
    with pytest.raises(ValueError, match="canonical"):
        import_historical_run(
            tmp_path / "malformed-runs",
            slug="historical",
            config_path=sources["config"],
            final_checkpoint_path=malformed,
            selected_checkpoint_path=sources["selected"],
            dataset_identity=_dataset(),
        )
    assert not (tmp_path / "malformed-runs").exists()


@pytest.mark.parametrize(
    "contents",
    [
        "[]",
        '[{"model": "one"}]',
        '{"metric": NaN}',
        '{"metric": 1, "metric": 2}',
        '{"metric": 1}',
        "not JSON",
    ],
)
def test_import_rejects_summary_lists_and_corrupt_source_json(
    tmp_path: Path,
    contents: str,
) -> None:
    sources = _sources(tmp_path)
    sources["evaluation"].write_text(contents, encoding="utf-8")

    with pytest.raises((json.JSONDecodeError, ValueError)):
        import_historical_run(
            tmp_path / "runs",
            slug="historical",
            config_path=sources["config"],
            final_checkpoint_path=sources["final"],
            selected_checkpoint_path=sources["selected"],
            dataset_identity=_dataset(),
            evaluation_json=sources["evaluation"],
        )
    assert not (tmp_path / "runs").exists()


def test_dataset_identity_reader_requires_exact_object_and_no_symlink(
    tmp_path: Path,
) -> None:
    identity_path = tmp_path / "dataset.json"
    identity_path.write_text(json.dumps(_dataset().to_dict()), encoding="utf-8")
    assert read_dataset_identity(identity_path) == _dataset()

    summary = tmp_path / "dataset-summary.json"
    summary.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="root"):
        read_dataset_identity(summary)

    link = tmp_path / "dataset-link.json"
    link.symlink_to(identity_path)
    with pytest.raises(ValueError, match="symlink"):
        read_dataset_identity(link)


def test_fresh_ids_and_exclusive_nonoverwrite(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    common = {
        "slug": "historical",
        "config_path": sources["config"],
        "final_checkpoint_path": sources["final"],
        "selected_checkpoint_path": sources["selected"],
        "dataset_identity": _dataset(),
    }
    first = import_historical_run(tmp_path / "runs", **common)
    second = import_historical_run(tmp_path / "runs", **common)

    assert first.run_id != second.run_id
    explicit_id = "20260818T120000Z-historical-" + "a" * 24
    explicit = import_historical_run(
        tmp_path / "runs",
        **common,
        run_id=explicit_id,
        created_at="2026-08-18T12:00:00.000000Z",
    )
    manifest_before = explicit.paths.manifest.read_bytes()
    with pytest.raises(FileExistsError):
        import_historical_run(
            tmp_path / "runs",
            **common,
            run_id=explicit_id,
            created_at="2026-08-18T12:00:00.000000Z",
        )
    assert explicit.paths.manifest.read_bytes() == manifest_before


def test_import_module_and_cli_are_lightweight_and_python310_compatible() -> None:
    ast.parse(IMPORTS_MODULE.read_text(encoding="utf-8"), feature_version=(3, 10))
    ast.parse(IMPORT_TOOL.read_text(encoding="utf-8"), feature_version=(3, 10))
    environment = os.environ.copy()
    source_path = os.fspath(REPOSITORY_ROOT / "research" / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_path, environment.get("PYTHONPATH", "")))
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import lidar_model_selection.imports; "
                "assert 'torch' not in sys.modules; "
                "assert 'mmengine' not in sys.modules; "
                "assert 'mmdet3d' not in sys.modules"
            ),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_cli_requires_every_primary_evidence_argument() -> None:
    completed = subprocess.run(
        [sys.executable, os.fspath(IMPORT_TOOL)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert completed.returncode == 2
    assert "--runs-root" in completed.stderr
    assert "--dataset-identity" in completed.stderr
