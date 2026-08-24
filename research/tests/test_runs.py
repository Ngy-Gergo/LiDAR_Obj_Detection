from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import lidar_model_selection.runs as runs
from lidar_model_selection.checkpoints import CheckpointArtifact, TrainingOutputs
from lidar_model_selection.provenance import (
    build_training_compatibility,
    capture_code_provenance,
    capture_environment,
    identify_file_set,
)
from lidar_model_selection.runs import (
    DATASET_IDENTITY_SCHEME,
    RUN_SCHEMA_VERSION,
    DatasetIdentity,
    RunManifest,
    RunPaths,
    TrainingAttempt,
    TrainingState,
    build_dataset_identity,
    create_run,
    generate_run_id,
    load_run,
    update_run_manifest,
    update_training_state,
    validate_run_id,
)
from lidar_model_selection.storage import read_json_object, write_json_atomic


_STARTED = "2026-08-18T10:00:00.000000Z"
_FINISHED = "2026-08-18T10:05:00.000000Z"


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _evidence(tmp_path: Path, config_bytes: bytes = b"model = dict()\n") -> dict:
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    _git(evidence_root, "init", "--quiet")
    _git(evidence_root, "config", "user.name", "Run Test")
    _git(evidence_root, "config", "user.email", "run@example.invalid")
    source = evidence_root / "src" / "training.py"
    source.parent.mkdir()
    source.write_text("def train(): pass\n", encoding="utf-8")
    annotation = evidence_root / "data" / "kitti_infos_val.pkl"
    annotation.parent.mkdir()
    annotation.write_bytes(b"annotation identity")
    _git(evidence_root, "add", ".")
    _git(evidence_root, "commit", "--quiet", "-m", "evidence")

    source_files = identify_file_set(evidence_root, [source])
    annotations = identify_file_set(evidence_root, [annotation])
    dataset = build_dataset_identity(
        name="KITTI",
        version=None,
        root_reference="dataset:kitti-local-v1",
        semantic_partition="KITTI validation",
        framework_key="test_dataloader",
        annotation_files=annotations,
        class_names=["Car"],
        tasks={"3d_detection": ["Car"]},
    )
    return {
        "config_bytes": config_bytes,
        "dataset": dataset,
        "code_provenance": capture_code_provenance(evidence_root, ["src"]),
        "environment": capture_environment(include_packages=False),
        "training_compatibility": build_training_compatibility(
            hashlib.sha256(config_bytes).hexdigest(),
            dataset.identity_sha256,
            source_files,
            core_packages={"torch": "2.1.2", "mmengine": "0.10.7"},
            python_version="3.10.14",
        ),
    }


def _native_run(tmp_path: Path, *, target_epoch: int = 4):
    evidence = _evidence(tmp_path)
    return create_run(
        tmp_path / "runs",
        slug="centerpoint-car",
        target_epoch=target_epoch,
        **evidence,
    )


def _checkpoint(
    path: str,
    *,
    epoch: int | None,
    digest_character: str,
) -> CheckpointArtifact:
    return CheckpointArtifact(
        path=path,
        sha256=digest_character * 64,
        size_bytes=100,
        epoch=epoch,
        checkpoint_format="pytorch_zip",
        validation_profile="pytorch-zip-structural-v1",
    )


def _outputs(target_epoch: int) -> TrainingOutputs:
    return TrainingOutputs(
        final_checkpoint=_checkpoint(
            f"training/epoch_{target_epoch}.pth",
            epoch=target_epoch,
            digest_character="a",
        ),
        selected_checkpoint=_checkpoint(
            "training/best_score.pth",
            epoch=None,
            digest_character="b",
        ),
    )


def _attempt(status: str, *, attempt_id: str = "attempt-1") -> TrainingAttempt:
    return TrainingAttempt(
        attempt_id=attempt_id,
        started_at=_STARTED,
        finished_at=None if status == "running" else _FINISHED,
        status=status,
        resume_checkpoint=None,
        failure="worker failed" if status in {"failed", "interrupted"} else None,
    )


def test_run_id_contains_utc_slug_and_at_least_96_random_bits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runs.secrets, "token_hex", lambda size: "ab" * size)
    run_id = generate_run_id(
        "centerpoint-car",
        now=datetime(2026, 8, 18, 12, 34, 56, tzinfo=timezone(timedelta(hours=2))),
    )

    assert run_id == "20260818T103456Z-centerpoint-car-" + "ab" * 12
    assert validate_run_id(run_id) == run_id
    assert len(run_id.rsplit("-", 1)[1]) == 24


@pytest.mark.parametrize(
    "slug",
    ["", "Upper", "has space", "double--hyphen", "-leading", "trailing-"],
)
def test_run_id_rejects_invalid_slugs(slug: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        generate_run_id(slug)


def test_run_paths_are_absolute_and_canonical(tmp_path: Path) -> None:
    run_id = "20260818T103456Z-model-" + "1" * 24
    paths = RunPaths.for_run(tmp_path / "relative" / "runs", run_id)

    assert paths.root.is_absolute()
    assert paths.manifest == paths.root / "manifest.json"
    assert paths.config == paths.root / "config.py"
    assert paths.training == paths.root / "training"
    assert paths.evaluation == paths.root / "evaluation"
    assert paths.benchmark == paths.root / "benchmark"
    assert RunPaths.from_root(paths.root) == paths


def test_dataset_identity_is_deterministic_and_keeps_semantics_separate(
    tmp_path: Path,
) -> None:
    annotation = tmp_path / "infos.pkl"
    annotation.write_bytes(b"infos")
    files = identify_file_set(tmp_path, [annotation])
    first = build_dataset_identity(
        name="KITTI",
        version=None,
        root_reference="kitti:local",
        semantic_partition="KITTI validation",
        framework_key="test_dataloader",
        annotation_files=files,
        class_names=["Car", "Cyclist"],
        tasks={"detection": ["Car", "Cyclist"]},
    )
    repeated = build_dataset_identity(
        name="KITTI",
        version=None,
        root_reference="kitti:local",
        semantic_partition="KITTI validation",
        framework_key="test_dataloader",
        annotation_files=files,
        class_names=["Car", "Cyclist"],
        tasks={"detection": ["Car", "Cyclist"]},
    )

    assert first == repeated
    assert first.scheme == DATASET_IDENTITY_SCHEME
    assert first.semantic_partition == "KITTI validation"
    assert first.framework_key == "test_dataloader"
    assert DatasetIdentity.from_dict(first.to_dict()) == first

    changed_framework_key = build_dataset_identity(
        name="KITTI",
        version=None,
        root_reference="kitti:local",
        semantic_partition="KITTI validation",
        framework_key="val_dataloader",
        annotation_files=files,
        class_names=["Car", "Cyclist"],
        tasks={"detection": ["Car", "Cyclist"]},
    )
    assert changed_framework_key.identity_sha256 != first.identity_sha256


def test_dataset_identity_is_portable_but_semantic_changes_are_distinct(
    tmp_path: Path,
) -> None:
    roots = (tmp_path / "machine-a" / "KITTI", tmp_path / "machine-b" / "data")
    identities = []
    for root in roots:
        root.mkdir(parents=True)
        annotation = root / "kitti_infos_train.pkl"
        annotation.write_bytes(b"same annotation bytes")
        identities.append(
            build_dataset_identity(
                name="KITTI",
                version="object-v1",
                root_reference=str(root.absolute()),
                semantic_partition="KITTI train/validation",
                framework_key="test_dataloader",
                annotation_files=identify_file_set(root, [annotation]),
                class_names=["Car"],
                tasks={"3d_detection": ["Car"]},
            )
        )

    first, relocated = identities
    assert first.root_reference != relocated.root_reference
    assert first.identity_sha256 == relocated.identity_sha256

    def changed(**overrides: object) -> DatasetIdentity:
        values = {
            "name": "KITTI",
            "version": "object-v1",
            "root_reference": first.root_reference,
            "semantic_partition": "KITTI train/validation",
            "framework_key": "test_dataloader",
            "annotation_files": first.annotation_files,
            "class_names": ["Car"],
            "tasks": {"3d_detection": ["Car"]},
        }
        values.update(overrides)
        return build_dataset_identity(**values)  # type: ignore[arg-type]

    altered_root = tmp_path / "altered"
    altered_root.mkdir()
    altered_file = altered_root / "kitti_infos_train.pkl"
    altered_file.write_bytes(b"different annotation bytes")
    altered_annotations = identify_file_set(altered_root, [altered_file])
    assert changed(annotation_files=altered_annotations).identity_sha256 != first.identity_sha256
    assert changed(semantic_partition="KITTI test").identity_sha256 != first.identity_sha256
    assert changed(class_names=["Car", "Cyclist"]).identity_sha256 != first.identity_sha256
    assert changed(tasks={"3d_detection": ["Cyclist"]}).identity_sha256 != first.identity_sha256
    assert changed(version="object-v2").identity_sha256 != first.identity_sha256


def test_dataset_identity_preserves_explicit_unknowns() -> None:
    unknown = build_dataset_identity(
        name=None,
        version=None,
        root_reference=None,
        semantic_partition=None,
        framework_key=None,
        annotation_files=None,
        class_names=None,
        tasks=None,
    )

    assert DatasetIdentity.from_dict(unknown.to_dict()) == unknown
    assert unknown.to_dict()["semantic_partition"] is None


def test_legacy_path_bound_dataset_identity_records_keep_v1_meaning() -> None:
    current = build_dataset_identity(
        name="KITTI",
        version="object-v1",
        root_reference="/legacy/mount/KITTI",
        semantic_partition="KITTI validation",
        framework_key="test_dataloader",
        annotation_files=None,
        class_names=["Car"],
        tasks={"3d_detection": ["Car"]},
    )
    document = current.to_dict()
    document["scheme"] = "lidar-dataset-v1"
    identity_payload = dict(document)
    identity_payload.pop("identity_sha256")
    document["identity_sha256"] = hashlib.sha256(
        json.dumps(
            identity_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    loaded = DatasetIdentity.from_dict(document)
    assert loaded.scheme == "lidar-dataset-v1"
    assert loaded.root_reference == "/legacy/mount/KITTI"
    assert loaded.identity_sha256 != current.identity_sha256


def test_create_native_run_transactionally_publishes_canonical_layout(
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path)
    run_id = "20260818T103456Z-centerpoint-car-" + "2" * 24
    created = create_run(
        tmp_path / "runs",
        slug="centerpoint-car",
        run_id=run_id,
        target_epoch=40,
        **evidence,
    )

    assert created.run_id == run_id
    assert created.paths.config.read_bytes() == evidence["config_bytes"]
    assert created.manifest.config.sha256 == hashlib.sha256(
        evidence["config_bytes"]
    ).hexdigest()
    assert created.manifest.config.path == "config.py"
    assert created.manifest.schema_version == RUN_SCHEMA_VERSION
    assert created.manifest.revision == 0
    assert created.manifest.training == TrainingState.pending()
    assert created.manifest.resumable is True
    assert created.manifest.history_complete is True
    assert created.paths.training.is_dir()
    assert created.paths.evaluation.is_dir()
    assert created.paths.benchmark.is_dir()
    assert load_run(created.paths.root) == created
    assert RunManifest.from_dict(created.manifest.to_dict()) == created.manifest
    with pytest.raises(FrozenInstanceError):
        created.manifest.revision = 9  # type: ignore[misc]


def test_create_run_rejects_explicit_id_for_another_slug(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    run_id = "20260818T103456Z-other-model-" + "3" * 24

    with pytest.raises(ValueError, match="does not match"):
        create_run(
            tmp_path / "runs",
            slug="centerpoint-car",
            run_id=run_id,
            target_epoch=4,
            **evidence,
        )


def test_creation_is_exclusive_and_cleans_failed_staging(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    run_id = "20260818T103456Z-centerpoint-car-" + "4" * 24
    first = create_run(
        tmp_path / "runs",
        slug="centerpoint-car",
        run_id=run_id,
        target_epoch=4,
        **evidence,
    )
    original_manifest = first.paths.manifest.read_bytes()

    with pytest.raises(FileExistsError):
        create_run(
            tmp_path / "runs",
            slug="centerpoint-car",
            run_id=run_id,
            target_epoch=4,
            **evidence,
        )

    assert first.paths.manifest.read_bytes() == original_manifest
    assert not list(first.paths.root.parent.glob(f".{run_id}.staging-*"))


def test_creation_cleans_stage_when_manifest_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence(tmp_path)
    run_id = "20260818T103456Z-centerpoint-car-" + "5" * 24
    failure = OSError("injected write failure")

    def fail_write(path: Path, value: object) -> None:
        raise failure

    monkeypatch.setattr(runs, "write_json_atomic", fail_write)
    with pytest.raises(OSError) as raised:
        create_run(
            tmp_path / "runs",
            slug="centerpoint-car",
            run_id=run_id,
            target_epoch=4,
            **evidence,
        )

    assert raised.value is failure
    assert not (tmp_path / "runs" / run_id).exists()
    assert not list((tmp_path / "runs").glob(f".{run_id}.staging-*"))


def test_creation_cleanup_does_not_mask_the_original_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence(tmp_path)
    original = OSError("original write failure")

    def fail_write(path: Path, value: object) -> None:
        raise original

    def fail_cleanup(path: Path) -> None:
        raise OSError("cleanup failure")

    monkeypatch.setattr(runs, "write_json_atomic", fail_write)
    monkeypatch.setattr(runs, "cleanup_staging_directory", fail_cleanup)
    with pytest.raises(OSError) as raised:
        create_run(
            tmp_path / "runs",
            slug="centerpoint-car",
            target_epoch=4,
            **evidence,
        )
    assert raised.value is original


def test_load_rejects_config_content_or_manifest_path_tampering(
    tmp_path: Path,
) -> None:
    created = _native_run(tmp_path)
    created.paths.config.write_text("changed = True\n", encoding="utf-8")

    with pytest.raises(ValueError, match="config bytes"):
        load_run(created.paths.root)

    created.paths.config.write_bytes(b"model = dict()\n")
    manifest = read_json_object(created.paths.manifest)
    manifest["run_id"] = "20260818T103456Z-other-" + "6" * 24
    manifest["slug"] = "other"
    write_json_atomic(created.paths.manifest, manifest)
    with pytest.raises(ValueError, match="does not match its directory"):
        load_run(created.paths.root)


def test_load_rejects_a_symlinked_canonical_directory(tmp_path: Path) -> None:
    created = _native_run(tmp_path)
    created.paths.evaluation.rmdir()
    target = tmp_path / "elsewhere"
    target.mkdir()
    created.paths.evaluation.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        load_run(created.paths.root)


def test_load_rejects_a_symlinked_manifest(tmp_path: Path) -> None:
    created = _native_run(tmp_path)
    saved = tmp_path / "saved-manifest.json"
    created.paths.manifest.replace(saved)
    created.paths.manifest.symlink_to(saved)

    with pytest.raises(ValueError, match="manifest must not be a symlink"):
        load_run(created.paths.root)


def test_native_compatibility_hashes_must_match_config_and_dataset(
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path)
    sources = evidence["training_compatibility"].training_sources
    wrong_config = build_training_compatibility(
        "f" * 64,
        evidence["dataset"].identity_sha256,
        sources,
        core_packages={},
        python_version="3.10.14",
    )
    with pytest.raises(ValueError, match="config hash"):
        create_run(
            tmp_path / "runs-config",
            slug="model",
            target_epoch=4,
            training_compatibility=wrong_config,
            **{
                key: value
                for key, value in evidence.items()
                if key != "training_compatibility"
            },
        )

    wrong_dataset = build_training_compatibility(
        hashlib.sha256(evidence["config_bytes"]).hexdigest(),
        "e" * 64,
        sources,
        core_packages={},
        python_version="3.10.14",
    )
    with pytest.raises(ValueError, match="dataset hash"):
        create_run(
            tmp_path / "runs-dataset",
            slug="model",
            target_epoch=4,
            training_compatibility=wrong_dataset,
            **{
                key: value
                for key, value in evidence.items()
                if key != "training_compatibility"
            },
        )


def test_imported_run_allows_unknown_historical_metadata_but_is_completed(
    tmp_path: Path,
) -> None:
    unknown_dataset = build_dataset_identity(
        name="KITTI",
        version=None,
        root_reference=None,
        semantic_partition="KITTI validation",
        framework_key=None,
        annotation_files=None,
        class_names=None,
        tasks=None,
    )
    completed = TrainingState(
        status="completed",
        attempts=(),
        outputs=_outputs(40),
    )
    imported = create_run(
        tmp_path / "runs",
        slug="historical-centerpoint",
        config_bytes=b"model = dict()\n",
        dataset=unknown_dataset,
        target_epoch=40,
        origin="historical_import",
        training_state=completed,
    )

    assert imported.manifest.code_provenance is None
    assert imported.manifest.environment is None
    assert imported.manifest.training_compatibility is None
    assert imported.manifest.training.outputs is not None
    assert (
        imported.manifest.training.outputs.final_checkpoint
        != imported.manifest.training.outputs.selected_checkpoint
    )
    assert imported.manifest.resumable is False
    assert imported.manifest.history_complete is False


def test_imported_run_must_be_completed_and_native_requires_evidence(
    tmp_path: Path,
) -> None:
    unknown_dataset = build_dataset_identity(
        name=None,
        version=None,
        root_reference=None,
        semantic_partition=None,
        framework_key=None,
        annotation_files=None,
        class_names=None,
        tasks=None,
    )
    with pytest.raises(ValueError, match="must be completed"):
        create_run(
            tmp_path / "runs",
            slug="historical",
            config_bytes=b"model = dict()\n",
            dataset=unknown_dataset,
            target_epoch=4,
            origin="historical_import",
        )
    with pytest.raises(ValueError, match="requires code"):
        create_run(
            tmp_path / "runs",
            slug="native",
            config_bytes=b"model = dict()\n",
            dataset=unknown_dataset,
            target_epoch=4,
        )


def test_parent_lineage_is_validated_and_persisted(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    parent_id = "20260818T103456Z-parent-" + "7" * 24
    child = create_run(
        tmp_path / "runs",
        slug="child",
        config_bytes=evidence["config_bytes"],
        dataset=evidence["dataset"],
        code_provenance=evidence["code_provenance"],
        environment=evidence["environment"],
        training_compatibility=evidence["training_compatibility"],
        target_epoch=4,
        parent_run_id=parent_id,
    )
    assert child.manifest.parent_run_id == parent_id

    with pytest.raises(ValueError, match="own parent"):
        create_run(
            tmp_path / "other-runs",
            slug="child",
            run_id="20260818T103456Z-child-" + "8" * 24,
            config_bytes=evidence["config_bytes"],
            dataset=evidence["dataset"],
            code_provenance=evidence["code_provenance"],
            environment=evidence["environment"],
            training_compatibility=evidence["training_compatibility"],
            target_epoch=4,
            parent_run_id="20260818T103456Z-child-" + "8" * 24,
        )


def test_optimistic_training_updates_reload_disk_revision(tmp_path: Path) -> None:
    initial = _native_run(tmp_path)
    running = TrainingState(
        status="running",
        attempts=(_attempt("running"),),
        outputs=None,
    )
    updated = update_training_state(
        initial,
        running,
        expected_revision=0,
    )

    assert updated.manifest.revision == 1
    assert updated.manifest.training == running
    with pytest.raises(RuntimeError, match="revision conflict"):
        update_training_state(initial, running, expected_revision=0)
    assert load_run(initial.paths.root).manifest.revision == 1


def test_low_level_update_rejects_identity_changes_and_attempt_rewrites(
    tmp_path: Path,
) -> None:
    initial = _native_run(tmp_path)
    changed_identity = replace(
        initial.manifest,
        revision=1,
        target_epoch=5,
    )
    with pytest.raises(ValueError, match="identity fields"):
        update_run_manifest(
            initial.paths.root,
            changed_identity,
            expected_revision=0,
        )

    running = TrainingState("running", (_attempt("running"),), None)
    first = update_training_state(initial, running, expected_revision=0)
    rewritten = TrainingState(
        "running",
        (_attempt("running", attempt_id="different"),),
        None,
    )
    replacement = replace(first.manifest, revision=2, training=rewritten)
    with pytest.raises(ValueError, match="attempt history"):
        update_run_manifest(
            first.paths.root,
            replacement,
            expected_revision=1,
        )


def test_completed_native_run_is_terminal(tmp_path: Path) -> None:
    initial = _native_run(tmp_path, target_epoch=4)
    running = update_training_state(
        initial,
        TrainingState("running", (_attempt("running"),), None),
        expected_revision=0,
    )
    completed_attempt = _attempt("succeeded")
    completed = update_training_state(
        running,
        TrainingState("completed", (completed_attempt,), _outputs(4)),
        expected_revision=1,
    )

    assert completed.manifest.resumable is False
    assert completed.selected_checkpoint == _outputs(4).selected_checkpoint
    with pytest.raises(ValueError, match="terminal"):
        update_training_state(
            completed,
            completed.manifest.training,
            expected_revision=2,
        )


def test_native_completion_rejects_noncanonical_selected_checkpoint(
    tmp_path: Path,
) -> None:
    initial = _native_run(tmp_path, target_epoch=4)
    running = update_training_state(
        initial,
        TrainingState("running", (_attempt("running"),), None),
        expected_revision=0,
    )
    invalid_outputs = TrainingOutputs(
        final_checkpoint=_outputs(4).final_checkpoint,
        selected_checkpoint=_checkpoint(
            "training/arbitrary.pth",
            epoch=None,
            digest_character="c",
        ),
    )
    with pytest.raises(ValueError, match=r"best_\*\.pth"):
        update_training_state(
            running,
            TrainingState(
                "completed",
                (_attempt("succeeded"),),
                invalid_outputs,
            ),
            expected_revision=1,
        )


@pytest.mark.parametrize(
    ("checkpoint", "message"),
    [
        (
            _checkpoint(
                "training/best_score.pth",
                epoch=None,
                digest_character="d",
            ),
            "requires an epoch",
        ),
        (
            _checkpoint(
                "training/epoch_04.pth",
                epoch=4,
                digest_character="d",
            ),
            "below target_epoch",
        ),
        (
            _checkpoint(
                "training/epoch_03.pth",
                epoch=3,
                digest_character="d",
            ),
            "literal filename",
        ),
    ],
)
def test_native_resume_checkpoint_is_a_lower_literal_epoch(
    tmp_path: Path,
    checkpoint: CheckpointArtifact,
    message: str,
) -> None:
    initial = _native_run(tmp_path, target_epoch=4)
    attempt = TrainingAttempt(
        attempt_id="resume",
        started_at=_STARTED,
        finished_at=None,
        status="running",
        resume_checkpoint=checkpoint,
        failure=None,
    )
    with pytest.raises(ValueError, match=message):
        update_training_state(
            initial,
            TrainingState("running", (attempt,), None),
            expected_revision=0,
        )


def test_attempt_timestamp_order_uses_parsed_instants() -> None:
    with pytest.raises(ValueError, match="precedes"):
        TrainingAttempt(
            attempt_id="attempt",
            started_at="2026-08-18T10:00:00.9Z",
            finished_at="2026-08-18T10:00:00.10Z",
            status="succeeded",
            resume_checkpoint=None,
            failure=None,
        )


def test_manifest_rejects_boolean_schema_version(tmp_path: Path) -> None:
    created = _native_run(tmp_path)
    serialized = created.manifest.to_dict()
    serialized["schema_version"] = True
    with pytest.raises(ValueError, match="schema version"):
        RunManifest.from_dict(serialized)


def test_imported_run_is_terminal(tmp_path: Path) -> None:
    dataset = build_dataset_identity(
        name="KITTI",
        version=None,
        root_reference=None,
        semantic_partition="KITTI validation",
        framework_key=None,
        annotation_files=None,
        class_names=None,
        tasks=None,
    )
    imported = create_run(
        tmp_path / "runs",
        slug="imported",
        config_bytes=b"model = dict()\n",
        dataset=dataset,
        target_epoch=4,
        origin="historical_import",
        training_state=TrainingState("completed", (), _outputs(4)),
    )
    replacement = replace(imported.manifest, revision=1)

    with pytest.raises(ValueError, match="terminal"):
        update_run_manifest(
            imported.paths.root,
            replacement,
            expected_revision=0,
        )


def test_manifest_parser_rejects_unknown_fields(tmp_path: Path) -> None:
    created = _native_run(tmp_path)
    serialized = created.manifest.to_dict()
    serialized["unexpected"] = True

    with pytest.raises(ValueError, match="extra"):
        RunManifest.from_dict(serialized)
