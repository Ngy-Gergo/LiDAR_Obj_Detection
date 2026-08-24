from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

import lidar_model_selection.provenance as provenance
from lidar_model_selection.provenance import (
    CodeProvenance,
    EnvironmentInfo,
    FileArtifact,
    FileSetIdentity,
    capture_code_provenance,
    capture_environment,
    identify_file,
    identify_file_set,
    sha256_bytes,
    sha256_file,
)


_FORBIDDEN_IMPORT_PREFIXES = (
    "torch",
    "mmengine",
    "mmcv",
    "mmdet",
    "mmdet3d",
    "rclpy",
)


def test_sha256_bytes_matches_known_values_and_distinguishes_payloads() -> None:
    assert sha256_bytes(b"") == (
        "e3b0c44298fc1c149afbf4c8996fb924"
        "27ae41e4649b934ca495991b7852b855"
    )
    assert sha256_bytes(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )
    assert sha256_bytes(b"first") != sha256_bytes(b"second")


def test_sha256_file_handles_empty_small_and_multiblock_files(
    tmp_path: Path,
) -> None:
    payloads = {
        "empty.bin": b"",
        "small.bin": b"text\x00payload\xff\n",
        "large.bin": bytes(range(256)) * 8193,
    }
    assert len(payloads["large.bin"]) > 2 * 1024 * 1024

    for name, payload in payloads.items():
        path = tmp_path / name
        path.write_bytes(payload)

        assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_identify_file_records_portable_relative_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve() / "repository"
    path = root / "research" / "config.py"
    path.parent.mkdir(parents=True)
    payload = b"model = {'type': 'CenterPoint'}\n"
    path.write_bytes(payload)

    artifact = identify_file(path, relative_to=root)

    assert isinstance(artifact, FileArtifact)
    assert artifact.path == "research/config.py"
    assert artifact.sha256 == hashlib.sha256(payload).hexdigest()
    assert artifact.size_bytes == len(payload)
    assert str(root) not in artifact.path


def test_identify_file_without_root_records_absolute_path(
    tmp_path: Path,
) -> None:
    path = tmp_path.resolve() / "artifact.bin"
    payload = b"absolute artifact\n"
    path.write_bytes(payload)

    artifact = identify_file(path)

    recorded = Path(artifact.path)
    assert recorded.is_absolute()
    assert recorded == path
    assert artifact.sha256 == hashlib.sha256(payload).hexdigest()
    assert artifact.size_bytes == len(payload)


def test_identify_file_rejects_directory(tmp_path: Path) -> None:
    directory = tmp_path.resolve() / "directory"
    directory.mkdir()

    with pytest.raises(IsADirectoryError):
        identify_file(directory)


def test_identify_file_rejects_leaf_and_component_symlinks(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve() / "repository"
    real_directory = root / "real"
    real_directory.mkdir(parents=True)
    target = real_directory / "source.py"
    target.write_text("value = 1\n", encoding="utf-8")

    leaf_link = root / "source-link.py"
    _symlink_or_skip(leaf_link, target)
    with pytest.raises(ValueError, match="symlink"):
        identify_file(leaf_link, relative_to=root)

    directory_link = root / "directory-link"
    _symlink_or_skip(
        directory_link,
        real_directory,
        target_is_directory=True,
    )
    with pytest.raises(ValueError, match="symlink"):
        identify_file(directory_link / target.name, relative_to=root)


def test_identify_file_rejects_outside_and_parent_escape(
    tmp_path: Path,
) -> None:
    base = tmp_path.resolve()
    root = base / "repository"
    root.mkdir()
    outside = base / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside relative_to"):
        identify_file(outside, relative_to=root)

    with pytest.raises(ValueError, match="outside relative_to"):
        identify_file(root / ".." / outside.name, relative_to=root)


def test_identify_file_detects_change_during_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path.resolve() / "changing.bin"
    path.write_bytes(b"before")
    real_hash_file = provenance._hash_file

    def hash_then_change(candidate: Path) -> tuple[str, int]:
        result = real_hash_file(candidate)
        candidate.write_bytes(b"changed to a different size")
        return result

    monkeypatch.setattr(provenance, "_hash_file", hash_then_change)

    with pytest.raises(OSError, match="changed while it was being identified"):
        identify_file(path)


def test_file_set_identity_is_order_independent_and_sorted(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve() / "repository"
    root.mkdir()
    paths = {
        name: root / name
        for name in ("a.py", "b.py", "c.py")
    }
    for name, path in paths.items():
        path.write_text(f"name = {name!r}\n", encoding="utf-8")

    first = identify_file_set(
        [paths["b.py"], paths["a.py"], paths["c.py"]],
        relative_to=root,
        profile="training-source-v1",
    )
    second = identify_file_set(
        [paths["c.py"], paths["b.py"], paths["a.py"]],
        relative_to=root,
        profile="training-source-v1",
    )

    assert isinstance(first, FileSetIdentity)
    assert first == second
    assert first.hash_scheme == "lidar-file-set-v1"
    assert first.profile == "training-source-v1"
    assert tuple(file.path for file in first.files) == (
        "a.py",
        "b.py",
        "c.py",
    )


def test_file_set_v1_has_exact_golden_identity(tmp_path: Path) -> None:
    root = tmp_path.resolve() / "repository"
    first = root / "configs" / "a.py"
    second = root / "configs" / "nested" / "b.bin"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"alpha\n")
    second.write_bytes(b"\x00\xffB")

    identity = identify_file_set(
        [second, first],
        relative_to=root,
        profile="training-source-v1",
    )

    assert identity.hash_scheme == "lidar-file-set-v1"
    assert identity.sha256 == (
        "919d9524f0342ee73f534afa6f9a1c22"
        "076f0c618984d9500aab128ee2df1eed"
    )
    assert identity.files == (
        FileArtifact(
            path="configs/a.py",
            sha256=(
                "b6a98d9ce9a2d9149288fa3df42d377"
                "c3e42737afdcdaf714e33c0a100b51060"
            ),
            size_bytes=6,
        ),
        FileArtifact(
            path="configs/nested/b.bin",
            sha256=(
                "f803bec586282caafe409609aae90eb09"
                "f6d4cddb6e04431ddf76d22e7dcacd6"
            ),
            size_bytes=3,
        ),
    )


def test_file_set_content_change_changes_identity(tmp_path: Path) -> None:
    root = tmp_path.resolve() / "repository"
    root.mkdir()
    path = root / "source.py"
    path.write_text("value = 1\n", encoding="utf-8")
    before = identify_file_set(
        [path],
        relative_to=root,
        profile="source-v1",
    )

    path.write_text("value = 2\n", encoding="utf-8")
    after = identify_file_set(
        [path],
        relative_to=root,
        profile="source-v1",
    )

    assert after.sha256 != before.sha256


def test_file_set_path_change_changes_identity(tmp_path: Path) -> None:
    root = tmp_path.resolve() / "repository"
    root.mkdir()
    first = root / "first.py"
    second = root / "second.py"
    first.write_bytes(b"identical\n")
    second.write_bytes(b"identical\n")

    first_identity = identify_file_set(
        [first],
        relative_to=root,
        profile="source-v1",
    )
    second_identity = identify_file_set(
        [second],
        relative_to=root,
        profile="source-v1",
    )

    assert first_identity.files[0].sha256 == second_identity.files[0].sha256
    assert first_identity.sha256 != second_identity.sha256


def test_file_set_profile_change_changes_identity(tmp_path: Path) -> None:
    root = tmp_path.resolve() / "repository"
    root.mkdir()
    path = root / "source.py"
    path.write_text("value = 1\n", encoding="utf-8")

    first = identify_file_set(
        [path],
        relative_to=root,
        profile="training-source-v1",
    )
    second = identify_file_set(
        [path],
        relative_to=root,
        profile="evaluation-source-v1",
    )

    assert first.files == second.files
    assert first.sha256 != second.sha256


def test_file_set_rejects_duplicate_normalized_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve() / "repository"
    root.mkdir()
    path = root / "a.py"
    path.write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate file-set path"):
        identify_file_set(
            [path, Path("a.py")],
            relative_to=root,
            profile="source-v1",
        )


def test_file_set_rejects_empty_and_non_string_profile(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve() / "repository"
    root.mkdir()
    path = root / "a.py"
    path.write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="profile must not be empty"):
        identify_file_set([path], relative_to=root, profile="")

    with pytest.raises(TypeError, match="profile must be a string"):
        identify_file_set(
            [path],
            relative_to=root,
            profile=1,  # type: ignore[arg-type]
        )


def test_clean_git_provenance_is_stable_and_self_describing(
    git_repository: Path,
) -> None:
    scope = ["research/tools", "research/src"]

    first = capture_code_provenance(git_repository, scope=scope)
    second = capture_code_provenance(git_repository, scope=reversed(scope))

    assert isinstance(first, CodeProvenance)
    assert first == second
    assert first.hash_scheme == "lidar-workspace-v1"
    assert first.scope == ("research/src", "research/tools")
    assert first.git_commit == _run_git(
        git_repository,
        "rev-parse",
        "HEAD",
    ).strip()
    assert first.git_dirty is False
    assert first.workspace_sha256 is not None
    assert first.capture_error is None


def test_unstaged_tracked_change_changes_workspace_identity(
    git_repository: Path,
) -> None:
    before = capture_code_provenance(
        git_repository,
        scope=["research/src"],
    )
    (git_repository / "research/src/a.py").write_text(
        "value = 2\n",
        encoding="utf-8",
    )

    after = capture_code_provenance(
        git_repository,
        scope=["research/src"],
    )

    assert after.git_commit == before.git_commit
    assert after.git_dirty is True
    assert after.workspace_sha256 != before.workspace_sha256
    assert after.capture_error is None


def test_staged_only_change_changes_workspace_identity(
    git_repository: Path,
) -> None:
    before = capture_code_provenance(
        git_repository,
        scope=["research/src"],
    )
    path = git_repository / "research/src/a.py"
    path.write_text("value = 3\n", encoding="utf-8")
    _run_git(git_repository, "add", "--", "research/src/a.py")

    after = capture_code_provenance(
        git_repository,
        scope=["research/src"],
    )

    assert _run_git(git_repository, "diff", "--quiet").strip() == ""
    assert after.git_commit == before.git_commit
    assert after.git_dirty is True
    assert after.workspace_sha256 != before.workspace_sha256
    assert after.capture_error is None


def test_deleted_tracked_file_changes_workspace_identity(
    git_repository: Path,
) -> None:
    before = capture_code_provenance(
        git_repository,
        scope=["research/src"],
    )
    (git_repository / "research/src/a.py").unlink()

    after = capture_code_provenance(
        git_repository,
        scope=["research/src"],
    )

    assert after.git_commit == before.git_commit
    assert after.git_dirty is True
    assert after.workspace_sha256 != before.workspace_sha256
    assert after.capture_error is None


def test_untracked_file_content_changes_workspace_identity(
    git_repository: Path,
) -> None:
    path = git_repository / "research/src/new.py"
    path.write_text("new_value = 1\n", encoding="utf-8")
    first = capture_code_provenance(
        git_repository,
        scope=["research/src"],
    )

    path.write_text("new_value = 2\n", encoding="utf-8")
    second = capture_code_provenance(
        git_repository,
        scope=["research/src"],
    )

    assert first.git_dirty is True
    assert second.git_dirty is True
    assert first.workspace_sha256 != second.workspace_sha256
    assert first.capture_error is None
    assert second.capture_error is None


def test_ignored_file_does_not_change_workspace_identity(
    git_repository: Path,
) -> None:
    ignore_file = git_repository / ".gitignore"
    ignore_file.write_text(
        "research/src/generated.py\n",
        encoding="utf-8",
    )
    _commit_all(git_repository, "add ignore rule")
    before = capture_code_provenance(
        git_repository,
        scope=["research/src"],
    )

    (git_repository / "research/src/generated.py").write_text(
        "generated = True\n",
        encoding="utf-8",
    )
    after = capture_code_provenance(
        git_repository,
        scope=["research/src"],
    )

    assert before.git_dirty is False
    assert after.git_dirty is False
    assert after.workspace_sha256 == before.workspace_sha256
    assert after.capture_error is None


def test_out_of_scope_change_does_not_affect_scoped_provenance(
    git_repository: Path,
) -> None:
    before = capture_code_provenance(
        git_repository,
        scope=["research/src"],
    )
    (git_repository / "README.md").write_text(
        "changed outside scope\n",
        encoding="utf-8",
    )

    after = capture_code_provenance(
        git_repository,
        scope=["research/src"],
    )

    assert before.git_dirty is False
    assert after.git_dirty is False
    assert after.workspace_sha256 == before.workspace_sha256
    assert after.capture_error is None


def test_scope_itself_changes_workspace_identity(
    git_repository: Path,
) -> None:
    narrow = capture_code_provenance(
        git_repository,
        scope=["research/src"],
    )
    broad = capture_code_provenance(
        git_repository,
        scope=["research/src", "research/tools"],
    )

    assert narrow.scope == ("research/src",)
    assert broad.scope == ("research/src", "research/tools")
    assert narrow.git_commit == broad.git_commit
    assert narrow.workspace_sha256 != broad.workspace_sha256


def test_git_capture_handles_unicode_and_spaces_deterministically(
    git_repository: Path,
) -> None:
    path = git_repository / "research/src/naïve model.py"
    path.write_text("unicode_value = 1\n", encoding="utf-8")
    _commit_all(git_repository, "add unicode source")
    clean = capture_code_provenance(
        git_repository,
        scope=["research/src"],
    )

    path.write_text("unicode_value = 2\n", encoding="utf-8")
    first = capture_code_provenance(
        git_repository,
        scope=["research/src"],
    )
    second = capture_code_provenance(
        git_repository,
        scope=["research/src"],
    )

    assert clean.capture_error is None
    assert first.capture_error is None
    assert first.git_dirty is True
    assert first.workspace_sha256 != clean.workspace_sha256
    assert first.workspace_sha256 == second.workspace_sha256


def test_git_capture_supports_detached_head(git_repository: Path) -> None:
    expected_commit = _run_git(
        git_repository,
        "rev-parse",
        "HEAD",
    ).strip()
    _run_git(
        git_repository,
        "checkout",
        "--detach",
        "--quiet",
        "HEAD",
    )

    result = capture_code_provenance(
        git_repository,
        scope=["research/src"],
    )

    assert result.git_commit == expected_commit
    assert result.git_dirty is False
    assert result.workspace_sha256 is not None
    assert result.capture_error is None


def test_unborn_head_returns_self_describing_observational_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_git_environment(monkeypatch)
    repository = _init_git_repo(tmp_path.resolve() / "unborn")

    result = capture_code_provenance(
        repository,
        scope=["research/src"],
    )

    assert result.hash_scheme == "lidar-workspace-v1"
    assert result.scope == ("research/src",)
    assert result.git_commit is None
    assert result.git_dirty is None
    assert result.workspace_sha256 is None
    assert result.capture_error is not None


def test_non_git_directory_returns_observational_failure(
    tmp_path: Path,
) -> None:
    directory = tmp_path.resolve() / "ordinary"
    directory.mkdir()

    result = capture_code_provenance(
        directory,
        scope=["research/tools", "research/src"],
    )

    assert result.hash_scheme == "lidar-workspace-v1"
    assert result.scope == ("research/src", "research/tools")
    assert result.git_commit is None
    assert result.git_dirty is None
    assert result.workspace_sha256 is None
    assert result.capture_error is not None


def test_missing_git_executable_returns_observational_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path.resolve() / "repository"
    directory.mkdir()

    def git_unavailable(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr(provenance.subprocess, "run", git_unavailable)

    result = capture_code_provenance(
        directory,
        scope=["research/src"],
    )

    assert result.hash_scheme == "lidar-workspace-v1"
    assert result.scope == ("research/src",)
    assert result.git_commit is None
    assert result.git_dirty is None
    assert result.workspace_sha256 is None
    assert result.capture_error == "Git executable is unavailable"


def test_git_timeout_before_head_returns_observational_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path.resolve() / "repository"
    directory.mkdir()

    def git_timeout(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd=["git"], timeout=30)

    monkeypatch.setattr(provenance.subprocess, "run", git_timeout)

    result = capture_code_provenance(
        directory,
        scope=["research/src"],
    )

    assert result.git_commit is None
    assert result.git_dirty is None
    assert result.workspace_sha256 is None
    assert result.capture_error == "Git command timed out after 30 seconds"


def test_workspace_failure_after_head_preserves_commit_and_scope(
    git_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_commit = _run_git(
        git_repository,
        "rev-parse",
        "HEAD",
    ).strip()

    def fail_workspace_capture(
        _root: Path,
        _commit: str,
        _scope: tuple[str, ...],
    ) -> bytes:
        raise RuntimeError("simulated workspace failure\nwith details")

    monkeypatch.setattr(
        provenance,
        "_git_tracked_diff",
        fail_workspace_capture,
    )

    result = capture_code_provenance(
        git_repository,
        scope=["research/tools", "research/src"],
    )

    assert result.hash_scheme == "lidar-workspace-v1"
    assert result.scope == ("research/src", "research/tools")
    assert result.git_commit == expected_commit
    assert result.git_dirty is None
    assert result.workspace_sha256 is None
    assert result.capture_error == (
        "simulated workspace failure with details"
    )


def test_invalid_scope_remains_caller_error(
    git_repository: Path,
) -> None:
    with pytest.raises(ValueError, match="at least one path"):
        capture_code_provenance(git_repository, scope=[])

    with pytest.raises(TypeError, match="iterable of paths"):
        capture_code_provenance(
            git_repository,
            scope="research/src",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="outside repository"):
        capture_code_provenance(
            git_repository,
            scope=[git_repository.parent / "outside"],
        )

    with pytest.raises(ValueError, match="must not contain NUL"):
        capture_code_provenance(
            git_repository,
            scope=["research/src/bad\x00path"],
        )


def test_environment_capture_without_gpu_is_lightweight() -> None:
    environment = capture_environment(include_gpu=False)

    assert isinstance(environment, EnvironmentInfo)
    assert environment.python
    assert environment.platform
    assert environment.machine
    for version in (
        environment.torch,
        environment.mmengine,
        environment.mmcv,
        environment.mmdet,
        environment.mmdet3d,
    ):
        assert version is None or isinstance(version, str)
    assert environment.torch_cuda is None
    assert environment.gpu is None


def test_missing_distribution_metadata_becomes_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def distribution_version(distribution: str) -> str:
        if distribution == "mmcv":
            raise provenance.metadata.PackageNotFoundError(distribution)
        return f"{distribution}-test-version"

    monkeypatch.setattr(
        provenance.metadata,
        "version",
        distribution_version,
    )

    environment = capture_environment(include_gpu=False)

    assert environment.torch == "torch-test-version"
    assert environment.mmengine == "mmengine-test-version"
    assert environment.mmcv is None
    assert environment.mmdet == "mmdet-test-version"
    assert environment.mmdet3d == "mmdet3d-test-version"


def test_import_and_cpu_environment_capture_are_lightweight_subprocess() -> None:
    script = """
import importlib
import json
import sys

before = set(sys.modules)
module = importlib.import_module("lidar_model_selection.provenance")
environment = module.capture_environment(include_gpu=False)
loaded = set(sys.modules) - before
forbidden = (
    "torch",
    "mmengine",
    "mmcv",
    "mmdet",
    "mmdet3d",
    "rclpy",
)
unexpected = sorted(
    name
    for name in loaded
    if any(
        name == prefix or name.startswith(prefix + ".")
        for prefix in forbidden
    )
)
print(json.dumps({
    "unexpected": unexpected,
    "torch_cuda": environment.torch_cuda,
    "gpu": environment.gpu,
}))
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "unexpected": [],
        "torch_cuda": None,
        "gpu": None,
    }


def test_gpu_environment_capture_imports_torch_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports: list[str] = []
    fake_cuda = _FakeCuda(available=True, gpu_name="Test GPU")
    fake_torch = SimpleNamespace(
        __version__="2.5.0+cu124",
        version=SimpleNamespace(cuda="12.4"),
        cuda=fake_cuda,
    )

    def import_module(name: str) -> object:
        imports.append(name)
        return fake_torch

    monkeypatch.setattr(
        provenance.importlib,
        "import_module",
        import_module,
    )

    without_gpu = capture_environment(include_gpu=False)
    assert imports == []
    assert without_gpu.torch_cuda is None
    assert without_gpu.gpu is None

    with_gpu = capture_environment(include_gpu=True)
    assert imports == ["torch"]
    assert with_gpu.torch == "2.5.0+cu124"
    assert with_gpu.torch_cuda == "12.4"
    assert with_gpu.gpu == "Test GPU"
    assert fake_cuda.device_name_requests == [0]


def test_gpu_environment_capture_handles_unavailable_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cuda = _FakeCuda(available=False, gpu_name="must not be read")
    fake_torch = SimpleNamespace(
        __version__="2.4.0+cu121",
        version=SimpleNamespace(cuda="12.1"),
        cuda=fake_cuda,
    )
    monkeypatch.setattr(
        provenance.importlib,
        "import_module",
        lambda _name: fake_torch,
    )

    environment = capture_environment(include_gpu=True)

    assert environment.torch == "2.4.0+cu121"
    assert environment.torch_cuda == "12.1"
    assert environment.gpu is None
    assert fake_cuda.device_name_requests == []


def test_gpu_environment_capture_retains_metadata_on_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def distribution_version(distribution: str) -> str:
        if distribution == "torch":
            return "2.3.0-installed"
        raise provenance.metadata.PackageNotFoundError(distribution)

    def fail_import(_name: str) -> object:
        raise ImportError("simulated Torch import failure")

    monkeypatch.setattr(
        provenance.metadata,
        "version",
        distribution_version,
    )
    monkeypatch.setattr(
        provenance.importlib,
        "import_module",
        fail_import,
    )

    environment = capture_environment(include_gpu=True)

    assert environment.torch == "2.3.0-installed"
    assert environment.torch_cuda is None
    assert environment.gpu is None


def test_public_evidence_dataclasses_are_immutable() -> None:
    artifact = FileArtifact(path="a.py", sha256="0" * 64, size_bytes=0)
    records = (
        artifact,
        FileSetIdentity(
            hash_scheme="lidar-file-set-v1",
            profile="source-v1",
            sha256="1" * 64,
            files=(artifact,),
        ),
        CodeProvenance(
            hash_scheme="lidar-workspace-v1",
            scope=("research/src",),
            git_commit=None,
            git_dirty=None,
            workspace_sha256=None,
            capture_error="unavailable",
        ),
        EnvironmentInfo(
            python="3.10",
            platform="test-platform",
            machine="test-machine",
            torch=None,
            torch_cuda=None,
            mmengine=None,
            mmcv=None,
            mmdet=None,
            mmdet3d=None,
            gpu=None,
        ),
    )

    for record in records:
        field_name = next(iter(record.__dataclass_fields__))
        with pytest.raises(FrozenInstanceError):
            setattr(record, field_name, getattr(record, field_name))


@pytest.fixture
def git_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    _isolate_git_environment(monkeypatch)
    repository = _init_git_repo(tmp_path.resolve() / "repository")
    source = repository / "research/src/a.py"
    tool = repository / "research/tools/tool.py"
    source.parent.mkdir(parents=True)
    tool.parent.mkdir(parents=True)
    source.write_text("value = 1\n", encoding="utf-8")
    tool.write_text("tool = True\n", encoding="utf-8")
    (repository / "README.md").write_text("fixture\n", encoding="utf-8")
    _commit_all(repository, "initial")
    return repository


def _isolate_git_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Provenance Test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Provenance Test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "test@example.invalid")
    monkeypatch.setenv("GIT_AUTHOR_DATE", "2000-01-02T03:04:05+0000")
    monkeypatch.setenv("GIT_COMMITTER_DATE", "2000-01-02T03:04:05+0000")


def _init_git_repo(path: Path) -> Path:
    if shutil.which("git") is None:
        pytest.skip("Git executable is unavailable")

    path.mkdir()
    _run_git(
        path,
        "init",
        "--quiet",
        "--template=",
        "--object-format=sha1",
    )
    _run_git(path, "config", "user.email", "test@example.invalid")
    _run_git(path, "config", "user.name", "Provenance Test")
    _run_git(path, "config", "commit.gpgSign", "false")
    _run_git(path, "config", "core.autocrlf", "false")
    _run_git(path, "config", "core.fileMode", "false")
    return path


def _commit_all(repository: Path, message: str) -> None:
    _run_git(repository, "add", "--all")
    _run_git(
        repository,
        "commit",
        "--quiet",
        "--no-gpg-sign",
        "--no-verify",
        "-m",
        message,
    )


def _run_git(repository: Path, *arguments: str) -> str:
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    completed = subprocess.run(
        ["git", "-C", os.fspath(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def _symlink_or_skip(
    link: Path,
    target: Path,
    *,
    target_is_directory: bool = False,
) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"symlink creation is unavailable: {error}")


class _FakeCuda:
    def __init__(self, *, available: bool, gpu_name: str) -> None:
        self.available = available
        self.gpu_name = gpu_name
        self.device_name_requests: list[int] = []

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return 1 if self.available else 0

    def current_device(self) -> int:
        return 0

    def get_device_name(self, device: int) -> str:
        self.device_name_requests.append(device)
        return self.gpu_name
