from __future__ import annotations

import fcntl
import hashlib
import os
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_STORED, ZipFile

import pytest

import lidar_model_selection.catalog as catalog
import lidar_model_selection.training as training
from lidar_model_selection.checkpoints import CheckpointArtifact
from lidar_model_selection.provenance import (
    build_training_compatibility,
    capture_code_provenance,
    capture_environment,
    identify_file_set,
)
from lidar_model_selection.runs import (
    RunPaths,
    TrainingAttempt,
    TrainingState,
    build_dataset_identity,
    create_run,
    update_training_state,
)
from lidar_model_selection.training import (
    TrainingDecision,
    create_training_run,
    decide_training,
    execute_training,
)


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _native_run(tmp_path: Path, *, target_epoch: int = 2):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    _git(evidence, "init", "--quiet")
    _git(evidence, "config", "user.name", "Training Test")
    _git(evidence, "config", "user.email", "training@example.invalid")
    source = evidence / "src" / "train_impl.py"
    source.parent.mkdir()
    source.write_text("def train(): pass\n", encoding="utf-8")
    data = evidence / "data"
    data.mkdir()
    train_info = data / "kitti_infos_train.pkl"
    val_info = data / "kitti_infos_val.pkl"
    train_info.write_bytes(b"train infos")
    val_info.write_bytes(b"validation infos")
    _git(evidence, "add", ".")
    _git(evidence, "commit", "--quiet", "-m", "evidence")

    config_bytes = b"canonical = True\n"
    annotations = identify_file_set(data, (train_info, val_info))
    dataset = build_dataset_identity(
        name="KITTI",
        version=None,
        root_reference=str(data.absolute()),
        semantic_partition="KITTI validation",
        framework_key="test_dataloader",
        annotation_files=annotations,
        class_names=("Car",),
        tasks={"3d_object_detection_7d": ("Car",)},
    )
    environment = capture_environment(include_packages=False)
    sources = identify_file_set(evidence, (source,))
    compatibility = build_training_compatibility(
        hashlib.sha256(config_bytes).hexdigest(),
        dataset.identity_sha256,
        sources,
        core_packages={},
        python_version=environment.python_version,
    )
    return create_run(
        tmp_path / "runs",
        slug="pillar02",
        config_bytes=config_bytes,
        dataset=dataset,
        target_epoch=target_epoch,
        code_provenance=capture_code_provenance(evidence, ("src",)),
        environment=environment,
        training_compatibility=compatibility,
    )


def _write_checkpoint(path: Path, *, payload: bytes = b"state") -> Path:
    with ZipFile(path, "w", compression=ZIP_STORED) as archive:
        archive.writestr("archive/data.pkl", payload)
        archive.writestr("archive/version", b"3\n")
        archive.writestr("archive/data/0", b"tensor")
    return path


def _artifact(epoch: int, *, name: str | None = None) -> CheckpointArtifact:
    return CheckpointArtifact(
        path=f"training/{name or f'epoch_{epoch}.pth'}",
        sha256="a" * 64,
        size_bytes=10,
        epoch=epoch,
        checkpoint_format="pytorch_zip",
        validation_profile="pytorch-zip-structural-v1",
    )


def test_catalog_is_exact_fixed_slug_to_source_mapping() -> None:
    assert catalog.catalog_slugs() == (
        "pillar02",
        "pillar02-dcn",
        "voxel0075",
        "voxel0075-dcn",
        "voxel01",
        "voxel01-dcn",
    )
    assert set(catalog.CENTERPOINT_CONFIGS) == set(catalog.catalog_slugs())
    assert catalog.source_config_for_slug("pillar02-dcn").name == "pillar02_dcn.py"
    with pytest.raises(ValueError, match="unknown"):
        catalog.source_config_for_slug("arbitrary")


def test_training_module_has_no_eager_heavy_imports() -> None:
    source = Path(training.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "from torch" not in source
    assert "import mmengine" not in source
    assert "from mmengine" not in source
    assert (
        "research/src/lidar_model_selection/checkpoints.py"
        in training._TRAINING_SOURCE_PATHS
    )


def test_training_decision_is_frozen_and_validated() -> None:
    decision = TrainingDecision("fresh", None)
    with pytest.raises(FrozenInstanceError):
        decision.action = "resume"  # type: ignore[misc]
    with pytest.raises(ValueError, match="requires"):
        TrainingDecision("resume", None)
    with pytest.raises(ValueError, match="only a resume"):
        TrainingDecision("finalize", _artifact(1))


class _FakeConfig(dict):
    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error

    def __setattr__(self, key: str, value: object) -> None:
        self[key] = value

    def merge_from_dict(self, values: dict[str, object]) -> None:
        self["merged"] = values

    def dump(self, path: str) -> None:
        Path(path).write_text("canonical snapshot\n", encoding="utf-8")


def _effective_config() -> _FakeConfig:
    dataset = {
        "type": "KittiDataset",
        "data_root": "data/KITTI/",
        "ann_file": "kitti_infos_train.pkl",
        "metainfo": {"classes": ("Car",)},
        "box_type_3d": "LiDAR",
        "test_mode": False,
    }
    validation = {
        **dataset,
        "ann_file": "kitti_infos_val.pkl",
        "test_mode": True,
    }
    return _FakeConfig(
        train_cfg={"max_epochs": 20},
        work_dir="stale",
        load_from="stale.pth",
        resume=True,
        launcher="pytorch",
        class_names=("Car",),
        custom_imports={
            "imports": ["lidar_model_selection.compat.center_head_7d"],
            "allow_failed_imports": False,
        },
        model={
            "pts_bbox_head": {
                "tasks": [{"num_class": 1, "class_names": ["Car"]}],
                "bbox_coder": {"code_size": 7},
            }
        },
        train_dataloader={"dataset": dataset},
        val_dataloader={"dataset": validation},
        test_dataloader={"dataset": validation.copy()},
        val_evaluator={"ann_file": "data/KITTI/kitti_infos_val.pkl"},
        test_evaluator={"ann_file": "data/KITTI/kitti_infos_val.pkl"},
    )


@pytest.mark.parametrize("explicit", [False, True])
def test_create_training_run_generates_id_before_config_and_snapshots_exact_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    explicit: bool,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("source = True\n", encoding="utf-8")
    config = _effective_config()
    order: list[str] = []
    run_id = "20260818T120000Z-pillar02-" + "a" * 24
    captured: dict[str, object] = {}
    dataset = SimpleNamespace(identity_sha256="d" * 64)
    environment = SimpleNamespace(packages=(("torch", "2"),), python_version="3.10")

    monkeypatch.setattr(
        training,
        "source_config_for_slug",
        (lambda slug: source)
        if not explicit
        else (lambda slug: (_ for _ in ()).throw(AssertionError("catalog used"))),
    )
    monkeypatch.setattr(
        training,
        "generate_run_id",
        lambda slug: order.append("id") or run_id,
    )

    def load(path: Path):
        order.append("source" if path == source else "snapshot")
        return config

    monkeypatch.setattr(training, "_load_config", load)
    monkeypatch.setattr(training, "_dataset_identity", lambda *a, **k: dataset)
    monkeypatch.setattr(training, "capture_code_provenance", lambda *a: "code")
    monkeypatch.setattr(training, "capture_environment", lambda **k: environment)
    monkeypatch.setattr(training, "identify_file_set", lambda *a: "sources")
    monkeypatch.setattr(
        training,
        "build_training_compatibility",
        lambda *a, **k: "compat",
    )

    def publish(root: Path, **kwargs: object):
        captured["root"] = root
        captured.update(kwargs)
        return "created"

    monkeypatch.setattr(training, "create_run", publish)

    created = create_training_run(
        "pillar02",
        7,
        source_config=(source if explicit else None),
        runs_root=tmp_path / "runs",
        repository_root=tmp_path,
        config_overrides={"optim_wrapper.optimizer.lr": 0.001},
    )

    assert created == "created"
    assert order == ["id", "source", "snapshot"]
    assert captured["run_id"] == run_id
    assert captured["config_bytes"] == b"canonical snapshot\n"
    assert captured["target_epoch"] == 7
    assert config["train_cfg"]["max_epochs"] == 7
    expected_paths = RunPaths.for_run(tmp_path / "runs", run_id)
    assert config["work_dir"] == str(expected_paths.training)
    assert config["load_from"] is None
    assert config["resume"] is False
    assert config["launcher"] == "none"
    assert config["merged"] == {"optim_wrapper.optimizer.lr": 0.001}


def test_creation_rejects_overrides_of_run_owned_runtime_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(training, "source_config_for_slug", lambda slug: source)
    monkeypatch.setattr(training, "_load_config", lambda path: _effective_config())

    with pytest.raises(ValueError, match="run-owned"):
        create_training_run(
            "pillar02",
            2,
            runs_root=tmp_path / "runs",
            repository_root=tmp_path,
            config_overrides={"resume": True},
        )


def test_dataset_identity_captures_train_and_validation_and_separates_keys(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data" / "KITTI"
    data.mkdir(parents=True)
    (data / "kitti_infos_train.pkl").write_bytes(b"train")
    (data / "kitti_infos_val.pkl").write_bytes(b"val")

    identity = training._dataset_identity(_effective_config(), tmp_path)

    assert identity.root_reference == str(data.absolute())
    assert identity.semantic_partition == "KITTI validation"
    assert identity.framework_key == "test_dataloader"
    assert identity.class_names == ("Car",)
    assert len(identity.annotation_files.files) == 2
    assert dict(identity.tasks) == {"3d_object_detection_7d": ("Car",)}


@pytest.mark.parametrize("defect", ["train_mode", "split", "evaluator"])
def test_dataset_identity_rejects_semantically_false_validation_labels(
    tmp_path: Path,
    defect: str,
) -> None:
    data = tmp_path / "data" / "KITTI"
    data.mkdir(parents=True)
    (data / "kitti_infos_train.pkl").write_bytes(b"train")
    (data / "kitti_infos_val.pkl").write_bytes(b"val")
    config = _effective_config()
    if defect == "train_mode":
        config["train_dataloader"]["dataset"]["test_mode"] = True
    elif defect == "split":
        config["test_dataloader"]["dataset"]["ann_file"] = "other.pkl"
    else:
        config["test_evaluator"]["ann_file"] = "data/KITTI/other.pkl"

    with pytest.raises(ValueError):
        training._dataset_identity(config, tmp_path)


def test_decision_uses_only_run_local_canonical_epochs(tmp_path: Path) -> None:
    run = _native_run(tmp_path, target_epoch=4)
    _write_checkpoint(run.paths.training / "epoch_1.pth")
    _write_checkpoint(run.paths.training / "epoch_3.pth")
    outside = tmp_path / "epoch_4.pth"
    _write_checkpoint(outside)

    decision = decide_training(run)

    assert decision.action == "resume"
    assert decision.resume_checkpoint is not None
    assert decision.resume_checkpoint.path == "training/epoch_3.pth"


def test_decision_fresh_and_exact_target_finalization(tmp_path: Path) -> None:
    run = _native_run(tmp_path, target_epoch=2)
    assert decide_training(run) == TrainingDecision("fresh", None)

    _write_checkpoint(run.paths.training / "epoch_2.pth")
    assert decide_training(run) == TrainingDecision("finalize", None)


@pytest.mark.parametrize("name", ["epoch_0001.pth", "epoch_3.pth"])
def test_decision_refuses_noncanonical_or_beyond_target_epoch(
    tmp_path: Path,
    name: str,
) -> None:
    run = _native_run(tmp_path, target_epoch=2)
    _write_checkpoint(run.paths.training / name)
    with pytest.raises(ValueError):
        decide_training(run)


def test_decision_validates_best_state_without_a_final_checkpoint(
    tmp_path: Path,
) -> None:
    run = _native_run(tmp_path)
    (run.paths.training / "best_score.pth").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        decide_training(run)


def test_decision_refuses_multiple_valid_best_candidates_without_final(
    tmp_path: Path,
) -> None:
    run = _native_run(tmp_path)
    _write_checkpoint(run.paths.training / "best_a.pth")
    _write_checkpoint(run.paths.training / "best_b.pth")
    with pytest.raises(ValueError, match="multiple best"):
        decide_training(run)


def test_decision_refuses_best_checkpoint_without_resumable_epoch(
    tmp_path: Path,
) -> None:
    run = _native_run(tmp_path)
    _write_checkpoint(run.paths.training / "best_score.pth")

    with pytest.raises(ValueError, match="without a resumable epoch"):
        decide_training(run)


def _patch_execution_boundary(
    monkeypatch: pytest.MonkeyPatch,
    runner,
) -> None:
    monkeypatch.setattr(training, "_verify_current_compatibility", lambda *a: None)
    monkeypatch.setattr(
        training,
        "_runtime_config",
        lambda run, decision: {"decision": decision},
    )
    monkeypatch.setattr(training, "_run_mmengine", runner)


def _completed_parent_and_child(tmp_path: Path, *, child_target: int = 3):
    parent = _native_run(tmp_path, target_epoch=2)
    _write_checkpoint(parent.paths.training / "epoch_2.pth")
    parent = execute_training(parent, repository_root=tmp_path)
    child = create_run(
        parent.paths.root.parent,
        slug="pillar02",
        config_bytes=parent.paths.config.read_bytes(),
        dataset=parent.manifest.dataset,
        target_epoch=child_target,
        code_provenance=parent.manifest.code_provenance,
        environment=parent.manifest.environment,
        training_compatibility=parent.manifest.training_compatibility,
        parent_run_id=parent.run_id,
    )
    return parent, child


def test_execute_fresh_records_attempt_console_and_exact_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _native_run(tmp_path)

    def runner(config: object) -> None:
        os.write(1, b"native stdout evidence\n")
        os.write(2, b"native stderr evidence\n")
        _write_checkpoint(run.paths.training / "epoch_2.pth")

    _patch_execution_boundary(monkeypatch, runner)
    completed = execute_training(run, repository_root=tmp_path)

    assert completed.manifest.training.status == "completed"
    outputs = completed.manifest.training.outputs
    assert outputs.final_checkpoint.path == "training/epoch_2.pth"
    assert outputs.selected_checkpoint.path == "training/epoch_2.pth"
    assert len(completed.manifest.training.attempts) == 1
    assert completed.manifest.training.attempts[0].status == "succeeded"
    logs = tuple(completed.paths.training.glob("attempt-*.console.log"))
    assert len(logs) == 1
    log_text = logs[0].read_text(encoding="utf-8")
    assert "native stdout evidence" in log_text
    assert "native stderr evidence" in log_text


def test_execute_resumes_highest_epoch_and_reverifies_it_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _native_run(tmp_path, target_epoch=3)
    _write_checkpoint(run.paths.training / "epoch_1.pth")
    _write_checkpoint(run.paths.training / "epoch_2.pth")
    seen: list[str] = []

    def runner(config: object) -> None:
        decision = config["decision"]
        seen.append(decision.resume_checkpoint.path)
        _write_checkpoint(run.paths.training / "epoch_3.pth")

    _patch_execution_boundary(monkeypatch, runner)
    completed = execute_training(run, repository_root=tmp_path)

    assert seen == ["training/epoch_2.pth"]
    recorded_resume = completed.manifest.training.attempts[-1].resume_checkpoint
    assert recorded_resume.path == "training/epoch_2.pth"


def test_changed_resume_checkpoint_fails_recorded_attempt_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _native_run(tmp_path, target_epoch=2)
    resume = _write_checkpoint(run.paths.training / "epoch_1.pth")
    called = False
    monkeypatch.setattr(training, "_verify_current_compatibility", lambda *a: None)

    def runtime_config(run, decision):
        resume.write_bytes(b"changed after selection")
        return {}

    def runner(config: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(training, "_runtime_config", runtime_config)
    monkeypatch.setattr(training, "_run_mmengine", runner)

    with pytest.raises(ValueError, match="changed after selection"):
        execute_training(run, repository_root=tmp_path)

    assert called is False
    failed = training.load_run(run.paths.root)
    assert failed.manifest.training.status == "failed"
    assert failed.manifest.training.attempts[-1].status == "failed"


def test_runner_failure_is_isolated_and_persisted_with_console(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _native_run(tmp_path)

    def runner(config: object) -> None:
        print("runner evidence")
        raise RuntimeError("CUDA exploded")

    _patch_execution_boundary(monkeypatch, runner)
    with pytest.raises(RuntimeError, match="CUDA exploded"):
        execute_training(run, repository_root=tmp_path)

    failed = training.load_run(run.paths.root)
    attempt = failed.manifest.training.attempts[-1]
    assert failed.manifest.training.status == "failed"
    assert attempt.status == "failed"
    assert attempt.failure == "RuntimeError: CUDA exploded"
    log = next(failed.paths.training.glob("attempt-*.console.log"))
    assert "runner evidence" in log.read_text(encoding="utf-8")
    assert "Traceback" in log.read_text(encoding="utf-8")


def test_existing_target_finalizes_without_environment_gate_or_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _native_run(tmp_path)
    _write_checkpoint(run.paths.training / "epoch_2.pth")
    monkeypatch.setattr(
        training,
        "_verify_current_compatibility",
        lambda *a: pytest.fail("finalization must not gate on current environment"),
    )
    monkeypatch.setattr(
        training,
        "_run_mmengine",
        lambda config: pytest.fail("Runner must not run"),
    )

    completed = execute_training(run, repository_root=tmp_path)
    assert completed.manifest.training.status == "completed"
    with pytest.raises(ValueError, match="completed"):
        execute_training(completed, repository_root=tmp_path)


def test_terminal_run_is_rejected_before_training_lock_is_opened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _native_run(tmp_path)
    _write_checkpoint(run.paths.training / "epoch_2.pth")
    completed = execute_training(run, repository_root=tmp_path)
    before = {
        path.name: path.read_bytes()
        for path in completed.paths.training.iterdir()
        if path.is_file()
    }

    def forbidden_lock(run):
        pytest.fail("terminal execution must not open the training lock")

    monkeypatch.setattr(training, "_training_lock", forbidden_lock)
    with pytest.raises(ValueError, match="completed"):
        execute_training(completed, repository_root=tmp_path)

    after = {
        path.name: path.read_bytes()
        for path in completed.paths.training.iterdir()
        if path.is_file()
    }
    assert after == before


def test_imported_run_is_rejected_without_creating_training_lock(
    tmp_path: Path,
) -> None:
    native = _native_run(tmp_path)
    _write_checkpoint(native.paths.training / "epoch_2.pth")
    native = execute_training(native, repository_root=tmp_path)
    imported = create_run(
        tmp_path / "imported-runs",
        slug="pillar02",
        config_bytes=native.paths.config.read_bytes(),
        dataset=native.manifest.dataset,
        target_epoch=2,
        origin="historical_import",
        training_state=native.manifest.training,
    )

    with pytest.raises(ValueError, match="historical imported"):
        execute_training(imported, repository_root=tmp_path)

    assert not (imported.paths.training / ".training.lock").exists()


def test_parent_checkpoint_initializes_fresh_child_but_local_resume_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, child = _completed_parent_and_child(tmp_path, child_target=4)

    def config_for_child(path: Path):
        config = _effective_config()
        config["work_dir"] = str(child.paths.training)
        config["train_cfg"]["max_epochs"] = 4
        config["load_from"] = None
        config["resume"] = False
        config["launcher"] = "none"
        return config

    monkeypatch.setattr(training, "_load_config", config_for_child)
    fresh = training._runtime_config(child, TrainingDecision("fresh", None))
    assert fresh["load_from"] == str(
        parent.paths.root / parent.selected_checkpoint.path
    )
    assert fresh["resume"] is False

    _write_checkpoint(child.paths.training / "epoch_1.pth")
    resume = decide_training(child)
    resumed = training._runtime_config(child, resume)
    assert resumed["load_from"] == str(child.paths.training / "epoch_1.pth")
    assert resumed["resume"] is True


def test_parent_initialization_is_verified_and_not_recorded_as_local_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, child = _completed_parent_and_child(tmp_path)

    def config_for_child(path: Path):
        config = _effective_config()
        config["work_dir"] = str(child.paths.training)
        config["train_cfg"]["max_epochs"] = 3
        config["load_from"] = None
        config["resume"] = False
        config["launcher"] = "none"
        return config

    monkeypatch.setattr(training, "_load_config", config_for_child)
    monkeypatch.setattr(training, "_verify_current_compatibility", lambda *a: None)

    def runner(config: object) -> None:
        assert config["load_from"] == str(
            parent.paths.root / parent.selected_checkpoint.path
        )
        assert config["resume"] is False
        _write_checkpoint(child.paths.training / "epoch_3.pth")

    monkeypatch.setattr(training, "_run_mmengine", runner)
    completed = execute_training(child, repository_root=tmp_path)
    assert completed.manifest.training.attempts[-1].resume_checkpoint is None


def test_parent_checkpoint_tamper_refuses_child_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, child = _completed_parent_and_child(tmp_path)
    (parent.paths.training / "epoch_2.pth").write_bytes(b"tampered")
    config = _effective_config()
    config["work_dir"] = str(child.paths.training)
    config["train_cfg"]["max_epochs"] = 3
    config["load_from"] = None
    config["resume"] = False
    config["launcher"] = "none"
    monkeypatch.setattr(training, "_load_config", lambda path: config)

    with pytest.raises(ValueError, match="parent selected"):
        training._runtime_config(child, TrainingDecision("fresh", None))


def test_creation_requires_parent_from_same_runs_root(tmp_path: Path) -> None:
    parent = _native_run(tmp_path)
    with pytest.raises(ValueError, match="same runs_root"):
        create_training_run(
            "pillar02",
            3,
            runs_root=tmp_path / "other-runs",
            repository_root=tmp_path,
            parent_run=parent,
        )


def test_stale_running_attempt_is_closed_before_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _native_run(tmp_path)
    stale = TrainingAttempt(
        attempt_id="stale-attempt",
        started_at="2000-01-01T00:00:00.000000Z",
        finished_at=None,
        status="running",
        resume_checkpoint=None,
        failure=None,
    )
    run = update_training_state(
        run,
        TrainingState("running", (stale,), None),
        expected_revision=run.manifest.revision,
    )

    def runner(config: object) -> None:
        _write_checkpoint(run.paths.training / "epoch_2.pth")

    _patch_execution_boundary(monkeypatch, runner)
    completed = execute_training(run, repository_root=tmp_path)
    assert [item.status for item in completed.manifest.training.attempts] == [
        "interrupted",
        "succeeded",
    ]


def _with_running_attempt(run):
    attempt = TrainingAttempt(
        attempt_id="crashed-attempt",
        started_at="2000-01-01T00:00:00.000000Z",
        finished_at=None,
        status="running",
        resume_checkpoint=None,
        failure=None,
    )
    return update_training_state(
        run,
        TrainingState("running", (attempt,), None),
        expected_revision=run.manifest.revision,
    )


def test_corrupt_checkpoint_closes_stale_running_attempt(
    tmp_path: Path,
) -> None:
    run = _with_running_attempt(_native_run(tmp_path))
    (run.paths.training / "epoch_1.pth").write_bytes(b"corrupt")

    with pytest.raises(ValueError):
        execute_training(run, repository_root=tmp_path)

    failed = training.load_run(run.paths.root)
    assert failed.manifest.training.status == "failed"
    assert failed.manifest.training.attempts[-1].status == "failed"
    assert "ValueError" in failed.manifest.training.attempts[-1].failure


def test_invalid_last_pointer_closes_stale_running_attempt(
    tmp_path: Path,
) -> None:
    run = _with_running_attempt(_native_run(tmp_path))
    _write_checkpoint(run.paths.training / "epoch_2.pth")
    (run.paths.training / "last_checkpoint").write_text(
        "epoch_1.pth\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exact final"):
        execute_training(run, repository_root=tmp_path)

    failed = training.load_run(run.paths.root)
    assert failed.manifest.training.status == "failed"
    assert failed.manifest.training.attempts[-1].status == "failed"


def test_training_lock_refuses_a_concurrent_executor(
    tmp_path: Path,
) -> None:
    run = _native_run(tmp_path)
    lock = run.paths.training / ".training.lock"
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already active"):
            execute_training(run, repository_root=tmp_path)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_last_checkpoint_pointer_must_name_literal_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _native_run(tmp_path)

    def runner(config: object) -> None:
        _write_checkpoint(run.paths.training / "epoch_2.pth")
        (run.paths.training / "last_checkpoint").write_text(
            "epoch_1.pth\n", encoding="utf-8"
        )

    _patch_execution_boundary(monkeypatch, runner)
    with pytest.raises(ValueError, match="exact final"):
        execute_training(run, repository_root=tmp_path)
    failed = training.load_run(run.paths.root)
    assert failed.manifest.training.status == "failed"


def test_last_checkpoint_pointer_must_not_be_a_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _native_run(tmp_path)
    outside = tmp_path / "pointer.txt"
    outside.write_text("epoch_2.pth\n", encoding="utf-8")

    def runner(config: object) -> None:
        _write_checkpoint(run.paths.training / "epoch_2.pth")
        (run.paths.training / "last_checkpoint").symlink_to(outside)

    _patch_execution_boundary(monkeypatch, runner)
    with pytest.raises(ValueError, match="non-symlink"):
        execute_training(run, repository_root=tmp_path)


def test_output_mutation_before_terminal_commit_is_a_recorded_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _native_run(tmp_path)

    def runner(config: object) -> None:
        _write_checkpoint(run.paths.training / "epoch_2.pth")

    _patch_execution_boundary(monkeypatch, runner)
    original_outputs = training._outputs

    def mutate_after_identity(current):
        outputs = original_outputs(current)
        (run.paths.training / "epoch_2.pth").write_bytes(b"changed")
        return outputs

    monkeypatch.setattr(training, "_outputs", mutate_after_identity)
    with pytest.raises(ValueError, match="changed before completion"):
        execute_training(run, repository_root=tmp_path)
    failed = training.load_run(run.paths.root)
    assert failed.manifest.training.status == "failed"


def test_ambiguous_best_added_before_terminal_commit_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _native_run(tmp_path)

    def runner(config: object) -> None:
        _write_checkpoint(run.paths.training / "epoch_2.pth")
        _write_checkpoint(run.paths.training / "best_initial.pth")

    _patch_execution_boundary(monkeypatch, runner)
    original_outputs = training._outputs
    first = True

    def add_ambiguous_best(current):
        nonlocal first
        outputs = original_outputs(current)
        if first:
            first = False
            _write_checkpoint(run.paths.training / "best_late.pth")
        return outputs

    monkeypatch.setattr(training, "_outputs", add_ambiguous_best)
    with pytest.raises(ValueError, match="output state changed"):
        execute_training(run, repository_root=tmp_path)

    failed = training.load_run(run.paths.root)
    assert failed.manifest.training.status == "failed"


def test_config_mutation_during_runtime_load_is_refused_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _native_run(tmp_path)
    canonical = run.paths.config.read_bytes()
    called = False
    monkeypatch.setattr(training, "_verify_current_compatibility", lambda *a: None)

    def mutate_config(current, decision):
        current.paths.config.write_bytes(b"tampered = True\n")
        return {}

    def runner(config: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(training, "_runtime_config", mutate_config)
    monkeypatch.setattr(training, "_run_mmengine", runner)
    try:
        with pytest.raises(ValueError, match="canonical config bytes"):
            execute_training(run, repository_root=tmp_path)
    finally:
        run.paths.config.write_bytes(canonical)

    assert called is False
    unchanged = training.load_run(run.paths.root)
    assert unchanged.manifest.training == TrainingState.pending()


def test_process_output_cleanup_does_not_mask_active_runner_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []
    closed: list[int] = []
    saved = iter((101, 102))
    monkeypatch.setattr(training.os, "dup", lambda descriptor: next(saved))

    def dup2(source: int, destination: int) -> None:
        calls.append((source, destination))
        if len(calls) == 3:
            raise OSError("stdout restoration failed")

    monkeypatch.setattr(training.os, "dup2", dup2)
    monkeypatch.setattr(training.os, "close", closed.append)
    console = (tmp_path / "console.log").open("w", encoding="utf-8")
    try:
        with pytest.raises(RuntimeError, match="runner failed"):
            with training._capture_process_output(console):
                raise RuntimeError("runner failed")
    finally:
        console.close()

    assert calls[-2:] == [(101, 1), (102, 2)]
    assert closed == [101, 102]


def test_current_compatibility_uses_explicit_repository_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_sources = SimpleNamespace(
        files=(SimpleNamespace(path="research/source.py"),)
    )
    recorded = SimpleNamespace(
        training_sources=recorded_sources,
        core_packages=(("torch", "2"),),
    )
    dataset = SimpleNamespace(
        root_reference=str(tmp_path),
        identity_sha256="b" * 64,
    )
    run = SimpleNamespace(
        manifest=SimpleNamespace(
            training_compatibility=recorded,
            dataset=dataset,
            config=SimpleNamespace(sha256="a" * 64),
        ),
        paths=SimpleNamespace(config=tmp_path / "config.py"),
    )
    observed_roots: list[Path] = []
    monkeypatch.setattr(training, "_load_config", lambda path: {})
    monkeypatch.setattr(training, "_dataset_identity", lambda *a, **k: dataset)

    def identify(root: Path, paths: object):
        observed_roots.append(root)
        return recorded_sources

    monkeypatch.setattr(training, "identify_file_set", identify)
    monkeypatch.setattr(
        training,
        "build_training_compatibility",
        lambda *a, **k: recorded,
    )

    training._verify_current_compatibility(run, tmp_path)
    assert observed_roots == [tmp_path]
