from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from lidar_model_selection.provenance import (
    FILE_SET_SCHEME,
    TRAINING_COMPATIBILITY_PROFILE,
    TRAINING_COMPATIBILITY_VERSION,
    WORKSPACE_SCHEME,
    CodeProvenance,
    EnvironmentInfo,
    FileArtifact,
    FileSetIdentity,
    TrainingCompatibilityIdentity,
    build_training_compatibility,
    capture_code_provenance,
    capture_environment,
    identify_file,
    identify_file_set,
    sha256_file,
)


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "Provenance Test")
    _git(repository, "config", "user.email", "provenance@example.invalid")
    (repository / "source").mkdir()
    (repository / "outside").mkdir()
    (repository / "source" / "kept.py").write_text("original\n", encoding="utf-8")
    (repository / "source" / "missing.py").write_text("remove me\n", encoding="utf-8")
    (repository / "outside" / "ignored.py").write_text("outside\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "--quiet", "-m", "initial")
    return repository


def test_sha256_file_streams_exact_bytes(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 9000
    path = tmp_path / "large.bin"
    path.write_bytes(payload)

    assert sha256_file(path, chunk_size=1021) == hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="positive"):
        sha256_file(path, chunk_size=0)


def test_identify_file_uses_normalized_root_relative_path(tmp_path: Path) -> None:
    root = tmp_path / "one"
    nested = root / "nested" / "artifact.bin"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"content")

    artifact = identify_file(nested, root=root)

    assert artifact == FileArtifact(
        path="nested/artifact.bin",
        sha256=hashlib.sha256(b"content").hexdigest(),
        size_bytes=7,
    )
    assert FileArtifact.from_dict(artifact.to_dict()) == artifact
    with pytest.raises(FrozenInstanceError):
        artifact.path = "changed"  # type: ignore[misc]


def test_identify_file_rejects_outside_root_and_non_regular_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")

    with pytest.raises(ValueError, match="outside identity root"):
        identify_file(outside, root=root)
    with pytest.raises(ValueError, match="not a regular file"):
        identify_file(root, root=tmp_path)


def test_explicit_file_set_is_order_independent_and_root_independent(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    for root in (first_root, second_root):
        (root / "nested").mkdir(parents=True)
        (root / "a.txt").write_text("A", encoding="utf-8")
        (root / "nested" / "b.txt").write_text("B", encoding="utf-8")

    first = identify_file_set(first_root, ["nested/b.txt", "a.txt"])
    second = identify_file_set(second_root, ["a.txt", "nested/b.txt"])

    assert first == second
    assert first.scheme == FILE_SET_SCHEME
    assert [artifact.path for artifact in first.files] == ["a.txt", "nested/b.txt"]
    assert FileSetIdentity.from_dict(first.to_dict()) == first


def test_explicit_file_set_rejects_duplicates_and_changes_with_content(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.py"
    path.write_text("one", encoding="utf-8")
    initial = identify_file_set(tmp_path, [path])

    with pytest.raises(ValueError, match="duplicate"):
        identify_file_set(tmp_path, [path, "source.py"])

    path.write_text("two", encoding="utf-8")
    changed = identify_file_set(tmp_path, [path])
    assert changed.identity_sha256 != initial.identity_sha256


def test_file_set_rejects_tampered_persisted_digest(tmp_path: Path) -> None:
    path = tmp_path / "source.py"
    path.write_text("source", encoding="utf-8")
    identity = identify_file_set(tmp_path, [path])
    serialized = identity.to_dict()
    serialized["identity_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="does not match"):
        FileSetIdentity.from_dict(serialized)


def test_scoped_git_provenance_hashes_untracked_content_and_missing_files(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    (repository / "source" / "kept.py").write_text("modified\n", encoding="utf-8")
    (repository / "source" / "missing.py").unlink()
    (repository / "source" / "untracked.py").write_text(
        "untracked one\n",
        encoding="utf-8",
    )
    (repository / "outside" / "new.py").write_text("excluded\n", encoding="utf-8")

    provenance = capture_code_provenance(repository, ["source"])

    assert provenance.scheme == WORKSPACE_SCHEME
    assert provenance.scopes == ("source",)
    assert provenance.dirty is True
    assert [item.path for item in provenance.tracked_files.files] == [
        "source/kept.py"
    ]
    assert provenance.tracked_files.files[0].sha256 == hashlib.sha256(
        b"modified\n"
    ).hexdigest()
    assert [item.path for item in provenance.untracked_files.files] == [
        "source/untracked.py"
    ]
    assert provenance.untracked_files.files[0].sha256 == hashlib.sha256(
        b"untracked one\n"
    ).hexdigest()
    assert provenance.missing_tracked_files == ("source/missing.py",)
    assert "outside/new.py" not in json.dumps(provenance.to_dict())
    assert CodeProvenance.from_dict(provenance.to_dict()) == provenance

    repeated = capture_code_provenance(repository, [repository / "source"])
    assert repeated.workspace_sha256 == provenance.workspace_sha256

    (repository / "source" / "untracked.py").write_text(
        "untracked two\n",
        encoding="utf-8",
    )
    changed = capture_code_provenance(repository, ["source"])
    assert changed.workspace_sha256 != provenance.workspace_sha256


def test_scoped_git_provenance_excludes_dirty_paths_outside_scope(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    (repository / "outside" / "ignored.py").write_text("changed\n", encoding="utf-8")

    provenance = capture_code_provenance(repository, ["source"])

    assert provenance.dirty is False
    assert all(
        artifact.path.startswith("source/")
        for artifact in provenance.tracked_files.files
    )


def test_code_provenance_requires_explicit_valid_scope(tmp_path: Path) -> None:
    repository = _repository(tmp_path)

    with pytest.raises(ValueError, match="explicit workspace scope"):
        capture_code_provenance(repository, [])
    with pytest.raises(ValueError, match="outside repository"):
        capture_code_provenance(repository, [tmp_path / "elsewhere"])
    with pytest.raises(subprocess.CalledProcessError):
        capture_code_provenance(tmp_path / "not-a-repository", ["source"])


def test_environment_capture_does_not_import_torch_unless_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import_module = importlib.import_module

    def guarded_import(name: str, package: str | None = None) -> object:
        if name == "torch":
            raise AssertionError("Torch was imported without opt-in")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", guarded_import)
    environment = capture_environment(include_packages=False, include_torch=False)

    assert environment.gpu_available is None
    assert environment.torch_version is None
    assert environment.packages == ()
    assert EnvironmentInfo.from_dict(environment.to_dict()) == environment


def test_environment_capture_records_selected_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def version(name: str) -> str:
        if name == "missing-framework":
            raise importlib.metadata.PackageNotFoundError(name)
        return {"Torch": "2.1.2", "mm_engine": "0.10.7"}[name]

    monkeypatch.setattr(importlib.metadata, "version", version)
    environment = capture_environment(
        package_names=["mm_engine", "Torch", "missing-framework"],
    )

    assert environment.packages == (
        ("missing-framework", None),
        ("mm-engine", "0.10.7"),
        ("torch", "2.1.2"),
    )


def test_environment_capture_imports_torch_on_explicit_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        get_device_name=lambda index: f"GPU {index}",
    )
    fake_torch = SimpleNamespace(
        __version__="2.1.2",
        version=SimpleNamespace(cuda="12.1"),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 8900)),
        cuda=fake_cuda,
    )
    imported = []

    def import_module(name: str) -> object:
        imported.append(name)
        assert name == "torch"
        return fake_torch

    monkeypatch.setattr(importlib, "import_module", import_module)
    environment = capture_environment(
        include_packages=False,
        include_torch=True,
    )

    assert imported == ["torch"]
    assert environment.torch_version == "2.1.2"
    assert environment.cuda_version == "12.1"
    assert environment.cudnn_version == "8900"
    assert environment.gpu_available is True
    assert environment.gpu_devices == ("GPU 0", "GPU 1")


def test_training_compatibility_is_narrow_versioned_and_deterministic(
    tmp_path: Path,
) -> None:
    source = tmp_path / "train.py"
    source.write_text("train()\n", encoding="utf-8")
    sources = identify_file_set(tmp_path, [source])
    config_sha256 = hashlib.sha256(b"config").hexdigest()
    dataset_sha256 = hashlib.sha256(b"dataset").hexdigest()

    first = build_training_compatibility(
        config_sha256,
        dataset_sha256,
        sources,
        core_packages={"Torch": "2.1.2", "MMEngine": "0.10.7"},
        python_version="3.10.14",
    )
    repeated = build_training_compatibility(
        config_sha256,
        dataset_sha256,
        sources,
        core_packages={"mmengine": "0.10.7", "torch": "2.1.2"},
        python_version="3.10.14",
    )

    assert first == repeated
    assert first.profile == TRAINING_COMPATIBILITY_PROFILE
    assert first.version == TRAINING_COMPATIBILITY_VERSION
    assert TrainingCompatibilityIdentity.from_dict(first.to_dict()) == first
    serialized = first.to_dict()
    assert "workspace_sha256" not in serialized
    assert "git_commit" not in serialized

    changed_dataset = build_training_compatibility(
        config_sha256,
        hashlib.sha256(b"other dataset").hexdigest(),
        sources,
        core_packages=dict(first.core_packages),
        python_version=first.python_version,
    )
    assert changed_dataset.compatibility_sha256 != first.compatibility_sha256


def test_training_compatibility_rejects_tampering(tmp_path: Path) -> None:
    source = tmp_path / "train.py"
    source.write_text("train()\n", encoding="utf-8")
    sources = identify_file_set(tmp_path, [source])
    identity = build_training_compatibility(
        "1" * 64,
        "2" * 64,
        sources,
        core_packages={},
        python_version="3.10.0",
    )
    serialized = identity.to_dict()
    serialized["config_sha256"] = "3" * 64

    with pytest.raises(ValueError, match="does not match"):
        TrainingCompatibilityIdentity.from_dict(serialized)


def test_import_remains_lightweight() -> None:
    script = """
import json
import sys
import lidar_model_selection.provenance
forbidden = [
    name for name in sys.modules
    if name == 'torch' or name.startswith('torch.')
    or name == 'mmengine' or name.startswith('mmengine.')
    or name == 'mmcv' or name.startswith('mmcv.')
    or name == 'mmdet' or name.startswith('mmdet.')
    or name == 'mmdet3d' or name.startswith('mmdet3d.')
]
print(json.dumps(forbidden))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert json.loads(completed.stdout) == []
