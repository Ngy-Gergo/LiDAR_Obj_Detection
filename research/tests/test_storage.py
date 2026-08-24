from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

import lidar_model_selection.storage as storage
from lidar_model_selection.storage import (
    cleanup_staging_directory,
    create_staging_directory,
    publish_directory_exclusive,
    read_json_object,
    write_json_atomic,
    write_json_exclusive,
)


LINUX_ONLY = pytest.mark.skipif(
    sys.platform != "linux",
    reason="exclusive directory publication requires Linux renameat2",
)


def test_atomic_json_output_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    value = {
        "z": 1,
        "a": {
            "b": 2,
        },
    }

    write_json_atomic(path, value)

    text = path.read_bytes().decode("utf-8")
    assert text == '{\n  "a": {\n    "b": 2\n  },\n  "z": 1\n}\n'
    assert text.endswith("\n")
    assert not text.endswith("\n\n")
    assert read_json_object(path) == value


@pytest.mark.parametrize(
    "non_finite",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
    ],
)
def test_write_rejects_nested_non_finite_values(
    tmp_path: Path,
    non_finite: float,
) -> None:
    path = tmp_path / "metrics.json"

    with pytest.raises(ValueError):
        write_json_atomic(path, {"metrics": {"loss": non_finite}})

    assert not path.exists()


@pytest.mark.parametrize(
    "value",
    [
        pytest.param({1: "invalid"}, id="root"),
        pytest.param({"nested": {1: "invalid"}}, id="nested"),
    ],
)
def test_write_rejects_non_string_object_keys(
    tmp_path: Path,
    value: object,
) -> None:
    path = tmp_path / "manifest.json"

    with pytest.raises(TypeError, match="JSON object keys must be strings"):
        write_json_atomic(path, value)  # type: ignore[arg-type]

    assert not path.exists()


def test_non_string_key_failure_preserves_atomic_destination(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    write_json_atomic(path, {"version": 1})

    with pytest.raises(TypeError, match="JSON object keys must be strings"):
        write_json_atomic(
            path,
            {"nested": {1: "invalid"}},  # type: ignore[dict-item]
        )

    assert read_json_object(path) == {"version": 1}


def test_unsupported_value_preserves_atomic_destination(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_json_atomic(path, {"good": True})

    with pytest.raises(TypeError):
        write_json_atomic(path, {"unsupported": object()})

    assert read_json_object(path) == {"good": True}


def test_read_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_json_object(tmp_path / "missing.json")


def test_read_malformed_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "malformed.json"
    path.write_text('{"broken":', encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        read_json_object(path)


@pytest.mark.parametrize(
    ("text", "root_type"),
    [
        pytest.param("[]\n", "list", id="array"),
        pytest.param("42\n", "int", id="number"),
        pytest.param("null\n", "NoneType", id="null"),
    ],
)
def test_read_rejects_non_object_roots(
    tmp_path: Path,
    text: str,
    root_type: str,
) -> None:
    path = tmp_path / "non-object.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=f"got {root_type}"):
        read_json_object(path)


@pytest.mark.parametrize(
    "constant",
    [
        pytest.param("NaN", id="nan"),
        pytest.param("Infinity", id="positive-infinity"),
        pytest.param("-Infinity", id="negative-infinity"),
    ],
)
def test_read_rejects_non_finite_constants(
    tmp_path: Path,
    constant: str,
) -> None:
    path = tmp_path / "non-finite.json"
    path.write_text(f'{{"value": {constant}}}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite JSON number"):
        read_json_object(path)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            '{"run_id": "a", "run_id": "b"}\n',
            id="top-level",
        ),
        pytest.param(
            '{"training": {"status": "running", "status": "completed"}}\n',
            id="nested",
        ),
    ],
)
def test_read_rejects_duplicate_object_keys(
    tmp_path: Path,
    text: str,
) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        read_json_object(path)


def test_atomic_write_replaces_mutable_destination(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"

    write_json_atomic(path, {"revision": 1})
    write_json_atomic(path, {"revision": 2})

    assert read_json_object(path) == {"revision": 2}
    assert stat.S_ISREG(path.stat().st_mode)
    assert [entry.name for entry in tmp_path.iterdir()] == ["manifest.json"]


def test_atomic_replace_failure_preserves_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "manifest.json"
    write_json_atomic(path, {"revision": 1})
    temporary_paths: list[Path] = []

    def fail_replace(source: object, _destination: object) -> None:
        temporary_paths.append(Path(source))  # type: ignore[arg-type]
        raise OSError("simulated replace failure")

    monkeypatch.setattr(storage.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        write_json_atomic(path, {"revision": 2})

    assert read_json_object(path) == {"revision": 1}
    assert len(temporary_paths) == 1
    assert not temporary_paths[0].exists()


def test_atomic_post_replace_fsync_failure_reports_failure_with_new_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "manifest.json"
    write_json_atomic(path, {"revision": 1})

    def fail_directory_sync(_path: Path) -> None:
        raise OSError("simulated directory fsync failure")

    monkeypatch.setattr(storage, "_fsync_directory", fail_directory_sync)

    with pytest.raises(OSError, match="simulated directory fsync failure"):
        write_json_atomic(path, {"revision": 2})

    assert read_json_object(path) == {"revision": 2}


def test_exclusive_publication_succeeds(tmp_path: Path) -> None:
    path = tmp_path / "result.json"

    write_json_exclusive(path, {"result": 1})

    assert path.is_file()
    assert read_json_object(path) == {"result": 1}


def test_exclusive_collision_preserves_first_result(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    write_json_exclusive(path, {"result": 1})

    with pytest.raises(FileExistsError):
        write_json_exclusive(path, {"result": 2})

    assert read_json_object(path) == {"result": 1}


def test_exclusive_link_failure_does_not_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "result.json"
    temporary_paths: list[Path] = []

    def fail_link(source: object, _destination: object) -> None:
        temporary_paths.append(Path(source))  # type: ignore[arg-type]
        raise OSError("simulated link failure")

    monkeypatch.setattr(storage.os, "link", fail_link)

    with pytest.raises(OSError, match="simulated link failure"):
        write_json_exclusive(path, {"result": 1})

    assert not path.exists()
    assert len(temporary_paths) == 1
    assert not temporary_paths[0].exists()


def test_exclusive_first_parent_fsync_failure_is_precommit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "result.json"
    temporary_paths: list[Path] = []
    real_link = storage.os.link

    def record_link(source: object, destination: object) -> None:
        temporary_paths.append(Path(source))  # type: ignore[arg-type]
        real_link(source, destination)  # type: ignore[arg-type]

    def fail_directory_sync(_path: Path) -> None:
        raise OSError("simulated first parent fsync failure")

    monkeypatch.setattr(storage.os, "link", record_link)
    monkeypatch.setattr(storage, "_fsync_directory", fail_directory_sync)

    with pytest.raises(OSError, match="simulated first parent fsync failure"):
        write_json_exclusive(path, {"result": 1})

    assert len(temporary_paths) == 1
    assert not temporary_paths[0].exists()
    if path.exists():
        assert read_json_object(path) == {"result": 1}


def test_exclusive_alias_unlink_failure_after_commit_is_successful(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "result.json"
    temporary_paths: list[Path] = []
    real_link = storage.os.link
    real_unlink = Path.unlink

    def record_link(source: object, destination: object) -> None:
        temporary_paths.append(Path(source))  # type: ignore[arg-type]
        real_link(source, destination)  # type: ignore[arg-type]

    def fail_temporary_unlink(
        candidate: Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        if temporary_paths and candidate == temporary_paths[0]:
            raise PermissionError("simulated alias cleanup failure")
        real_unlink(candidate, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(storage.os, "link", record_link)
    monkeypatch.setattr(Path, "unlink", fail_temporary_unlink)

    write_json_exclusive(path, {"result": 1})

    assert read_json_object(path) == {"result": 1}
    assert len(temporary_paths) == 1
    assert temporary_paths[0].exists()


def test_exclusive_second_parent_fsync_failure_after_commit_is_successful(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "result.json"
    real_fsync_directory = storage._fsync_directory
    calls = 0

    def fail_second_directory_sync(directory: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated second parent fsync failure")
        real_fsync_directory(directory)

    monkeypatch.setattr(
        storage,
        "_fsync_directory",
        fail_second_directory_sync,
    )

    write_json_exclusive(path, {"result": 1})

    assert calls == 2
    assert read_json_object(path) == {"result": 1}


@LINUX_ONLY
def test_create_staging_directory_is_hidden_unique_sibling(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "runs"

    first = create_staging_directory(parent, "run-123")
    second = create_staging_directory(parent, "run-123")

    assert first.is_dir()
    assert second.is_dir()
    assert first.parent == parent
    assert second.parent == parent
    assert first.name.startswith(".run-123.staging-")
    assert second.name.startswith(".run-123.staging-")
    assert first != second


@pytest.mark.parametrize(
    "final_name",
    [
        pytest.param("", id="empty"),
        pytest.param(".", id="dot"),
        pytest.param("..", id="dot-dot"),
        pytest.param("a/b", id="nested"),
        pytest.param("/absolute", id="absolute"),
        pytest.param("bad\x00name", id="nul"),
    ],
)
def test_create_staging_directory_rejects_invalid_final_names(
    tmp_path: Path,
    final_name: str,
) -> None:
    parent = tmp_path / "runs"

    with pytest.raises(ValueError):
        create_staging_directory(parent, final_name)

    assert not parent.exists()


@LINUX_ONLY
def test_staged_directory_publication_preserves_tree(tmp_path: Path) -> None:
    parent = tmp_path / "runs"
    staging = create_staging_directory(parent, "run-123")
    (staging / "manifest.json").write_bytes(b'{"run_id": "run-123"}\n')
    (staging / "config.py").write_bytes(b"model = dict()\n")
    (staging / "nested").mkdir()
    (staging / "nested" / "metadata.json").write_bytes(b'{"version": 1}\n')
    destination = parent / "run-123"

    publish_directory_exclusive(staging, destination)

    assert not staging.exists()
    assert destination.is_dir()
    assert (destination / "manifest.json").read_bytes() == (
        b'{"run_id": "run-123"}\n'
    )
    assert (destination / "config.py").read_bytes() == b"model = dict()\n"
    assert (destination / "nested" / "metadata.json").read_bytes() == (
        b'{"version": 1}\n'
    )


@LINUX_ONLY
def test_staged_publication_collision_preserves_empty_destination_and_stage(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "runs"
    staging = create_staging_directory(parent, "run-123")
    (staging / "payload.txt").write_text("staged", encoding="utf-8")
    destination = parent / "run-123"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        publish_directory_exclusive(staging, destination)

    assert destination.is_dir()
    assert list(destination.iterdir()) == []
    assert staging.is_dir()
    assert (staging / "payload.txt").read_text(encoding="utf-8") == "staged"


@LINUX_ONLY
def test_staged_publication_requires_distinct_sibling_paths(
    tmp_path: Path,
) -> None:
    first_parent = tmp_path / "first"
    second_parent = tmp_path / "second"
    second_parent.mkdir()
    staging = create_staging_directory(first_parent, "run-123")

    with pytest.raises(ValueError, match="same parent"):
        publish_directory_exclusive(staging, second_parent / "run-123")

    with pytest.raises(ValueError, match="different paths"):
        publish_directory_exclusive(staging, staging)

    assert staging.is_dir()


@LINUX_ONLY
def test_staged_publication_rejects_symlink(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "runs"
    staging = create_staging_directory(parent, "run-123")
    external = tmp_path / "external.txt"
    external.write_text("outside", encoding="utf-8")
    link = staging / "link"
    _symlink_or_skip(link, external)
    destination = parent / "run-123"

    with pytest.raises(ValueError, match="symlink"):
        publish_directory_exclusive(staging, destination)

    assert not destination.exists()
    assert staging.is_dir()
    assert link.is_symlink()


@LINUX_ONLY
def test_staged_publication_rejects_fifo(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("os.mkfifo is unavailable")

    parent = tmp_path / "runs"
    staging = create_staging_directory(parent, "run-123")
    fifo = staging / "events.fifo"
    try:
        os.mkfifo(fifo)
    except OSError as error:
        pytest.skip(f"cannot create FIFO in test environment: {error}")
    destination = parent / "run-123"

    with pytest.raises(ValueError, match="unsupported file type"):
        publish_directory_exclusive(staging, destination)

    assert not destination.exists()
    assert staging.is_dir()


@LINUX_ONLY
def test_cleanup_removes_tree_without_following_symlinks(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "runs"
    staging = create_staging_directory(parent, "run-123")
    nested = staging / "nested"
    nested.mkdir()
    (nested / "metadata.json").write_text("{}\n", encoding="utf-8")
    external = tmp_path / "external.txt"
    external.write_text("preserve", encoding="utf-8")
    _symlink_or_skip(nested / "external-link", external)

    cleanup_staging_directory(staging)

    assert not staging.exists()
    assert external.read_text(encoding="utf-8") == "preserve"


@LINUX_ONLY
def test_cleanup_staging_directory_is_idempotent(tmp_path: Path) -> None:
    parent = tmp_path / "runs"
    staging = create_staging_directory(parent, "run-123")
    (staging / "payload.txt").write_text("payload", encoding="utf-8")

    cleanup_staging_directory(staging)
    cleanup_staging_directory(staging)

    assert not staging.exists()


def test_cleanup_refuses_arbitrary_directory(tmp_path: Path) -> None:
    important = tmp_path / "important"
    important.mkdir()
    payload = important / "payload.txt"
    payload.write_text("preserve", encoding="utf-8")

    with pytest.raises(ValueError, match="staging path name"):
        cleanup_staging_directory(important)

    assert important.is_dir()
    assert payload.read_text(encoding="utf-8") == "preserve"


def test_storage_import_is_lightweight_in_clean_subprocess() -> None:
    script = """
import importlib
import json
import sys

before = set(sys.modules)
importlib.import_module("lidar_model_selection.storage")
loaded = set(sys.modules) - before
forbidden = (
    "torch",
    "mmengine",
    "mmcv",
    "mmdet",
    "mmdet3d",
    "rclpy",
    "sensor_msgs",
    "visualization_msgs",
)
unexpected = sorted(
    name
    for name in loaded
    if any(
        name == prefix or name.startswith(prefix + ".")
        for prefix in forbidden
    )
)
print(json.dumps(unexpected))
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"symlink creation is unavailable: {error}")
