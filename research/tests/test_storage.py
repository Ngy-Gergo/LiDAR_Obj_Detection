from __future__ import annotations

import ctypes
import errno
import json
import os
import stat
from collections import UserDict
from pathlib import Path
from typing import Any, Callable

import pytest

import lidar_model_selection.storage as storage


_WRITERS = ("write_json_atomic", "write_json_exclusive")


def _temporary_aliases(parent: Path, destination_name: str) -> list[Path]:
    if not parent.exists():
        return []
    return list(parent.glob(f".{destination_name}.tmp-*"))


def _write_json(
    writer_name: str,
    path: Path,
    value: Any,
) -> None:
    getattr(storage, writer_name)(path, value)


def _assert_same_exception(
    expected: BaseException,
    operation: Callable[[], None],
) -> None:
    with pytest.raises(type(expected)) as raised:
        operation()
    assert raised.value is expected


class _FakeCFunction:
    def __init__(self, callback: Callable[..., int]) -> None:
        self._callback = callback
        self.argtypes: object = None
        self.restype: object = None

    def __call__(self, *args: object) -> int:
        return self._callback(*args)


class _FakeLibrary:
    def __init__(self, callback: Callable[..., int]) -> None:
        self.renameat2 = _FakeCFunction(callback)


def test_public_api_is_exact() -> None:
    assert storage.__all__ == (
        "read_json_object",
        "write_json_atomic",
        "write_json_exclusive",
        "create_staging_directory",
        "publish_directory_exclusive",
        "cleanup_staging_directory",
    )


def test_read_json_object_reads_utf8_object(tmp_path: Path) -> None:
    path = tmp_path / "object.json"
    path.write_text('{"name": "LiDAR", "nested": {"value": 7}}', encoding="utf-8")

    assert storage.read_json_object(path) == {
        "name": "LiDAR",
        "nested": {"value": 7},
    }


def test_read_json_object_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        storage.read_json_object(tmp_path / "missing.json")


def test_read_json_object_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "malformed.json"
    path.write_text('{"value":', encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        storage.read_json_object(path)


def test_read_json_object_rejects_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "invalid-utf8.json"
    path.write_bytes(b'{"value": "\xff"}')

    with pytest.raises(UnicodeDecodeError):
        storage.read_json_object(path)


@pytest.mark.parametrize(
    "document",
    ("[]", '"text"', "1", "true", "false", "null"),
)
def test_read_json_object_rejects_non_object_root(
    tmp_path: Path,
    document: str,
) -> None:
    path = tmp_path / "non-object.json"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError, match="root must be an object"):
        storage.read_json_object(path)


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
def test_read_json_object_rejects_nonfinite_constants(
    tmp_path: Path,
    constant: str,
) -> None:
    path = tmp_path / "constant.json"
    path.write_text(f'{{"nested": [{constant}]}}', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON constant"):
        storage.read_json_object(path)


@pytest.mark.parametrize(
    "document",
    (
        '{"key": 1, "key": 2}',
        '{"outer": {"key": 1, "key": 2}}',
    ),
)
def test_read_json_object_rejects_duplicate_keys_at_every_depth(
    tmp_path: Path,
    document: str,
) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key: 'key'"):
        storage.read_json_object(path)


@pytest.mark.parametrize("writer_name", _WRITERS)
def test_json_writer_uses_sorted_indented_json_with_one_newline(
    tmp_path: Path,
    writer_name: str,
) -> None:
    path = tmp_path / f"{writer_name}.json"

    _write_json(writer_name, path, {"z": 1, "a": {"b": 2}})

    assert path.read_bytes() == b'{\n  "a": {\n    "b": 2\n  },\n  "z": 1\n}\n'


@pytest.mark.parametrize("writer_name", _WRITERS)
def test_json_writer_accepts_a_custom_root_mapping(
    tmp_path: Path,
    writer_name: str,
) -> None:
    path = tmp_path / f"{writer_name}.json"
    value = UserDict({"second": 2, "first": 1})

    _write_json(writer_name, path, value)

    assert storage.read_json_object(path) == {"first": 1, "second": 2}


@pytest.mark.parametrize("writer_name", _WRITERS)
def test_json_writer_rejects_a_non_mapping_root(
    tmp_path: Path,
    writer_name: str,
) -> None:
    with pytest.raises(TypeError, match="must be a mapping"):
        _write_json(writer_name, tmp_path / "value.json", ["not", "an", "object"])


@pytest.mark.parametrize("writer_name", _WRITERS)
@pytest.mark.parametrize(
    "value",
    (
        {1: "root"},
        {"nested": {False: "mapping"}},
        {"nested": [{2: "list"}]},
        {"nested": (UserDict({3: "custom mapping"}),)},
    ),
)
def test_json_writer_rejects_non_string_keys_at_every_depth(
    tmp_path: Path,
    writer_name: str,
    value: object,
) -> None:
    with pytest.raises(TypeError, match="JSON object keys must be strings"):
        _write_json(writer_name, tmp_path / "value.json", value)


@pytest.mark.parametrize("writer_name", _WRITERS)
@pytest.mark.parametrize("constant", (float("nan"), float("inf"), float("-inf")))
def test_json_writer_rejects_nonfinite_values_before_creating_parent(
    tmp_path: Path,
    writer_name: str,
    constant: float,
) -> None:
    path = tmp_path / "not-created" / "value.json"

    with pytest.raises(ValueError):
        _write_json(writer_name, path, {"constant": constant})

    assert not path.parent.exists()


@pytest.mark.parametrize("writer_name", _WRITERS)
def test_json_writer_rejects_unsupported_values_before_modifying_destination(
    tmp_path: Path,
    writer_name: str,
) -> None:
    path = tmp_path / "value.json"
    original = b"original bytes\n"
    path.write_bytes(original)

    with pytest.raises(TypeError):
        _write_json(writer_name, path, {"unsupported": object()})

    assert path.read_bytes() == original
    assert _temporary_aliases(tmp_path, path.name) == []


@pytest.mark.parametrize("writer_name", _WRITERS)
def test_json_writer_creates_missing_parents_and_uses_same_directory_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer_name: str,
) -> None:
    path = tmp_path / "missing" / "nested" / "value.json"
    real_mkstemp = storage.tempfile.mkstemp
    calls: list[tuple[str, Path]] = []

    def tracked_mkstemp(*, prefix: str, dir: Path) -> tuple[int, str]:
        calls.append((prefix, Path(dir)))
        return real_mkstemp(prefix=prefix, dir=dir)

    monkeypatch.setattr(storage.tempfile, "mkstemp", tracked_mkstemp)

    _write_json(writer_name, path, {"value": 1})

    assert calls == [(".value.json.tmp-", path.parent)]
    assert storage.read_json_object(path) == {"value": 1}
    assert _temporary_aliases(path.parent, path.name) == []


def test_atomic_replace_failure_preserves_destination_and_removes_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "mutable.json"
    original = b'{"old": true}\n'
    path.write_bytes(original)
    failure = OSError("replace failed")

    def fail_replace(source: Path, destination: Path) -> None:
        assert Path(source).parent == tmp_path
        assert destination == path
        raise failure

    monkeypatch.setattr(storage.os, "replace", fail_replace)

    _assert_same_exception(
        failure,
        lambda: storage.write_json_atomic(path, {"new": True}),
    )
    assert path.read_bytes() == original
    assert _temporary_aliases(tmp_path, path.name) == []


def test_atomic_file_fsync_failure_preserves_destination_and_removes_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "mutable.json"
    original = b'{"old": true}\n'
    path.write_bytes(original)
    failure = OSError("file fsync failed")

    monkeypatch.setattr(storage.os, "fsync", lambda descriptor: (_ for _ in ()).throw(failure))

    _assert_same_exception(
        failure,
        lambda: storage.write_json_atomic(path, {"new": True}),
    )
    assert path.read_bytes() == original
    assert _temporary_aliases(tmp_path, path.name) == []


def test_atomic_fdopen_failure_closes_descriptor_and_removes_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "mutable.json"
    path.write_text('{"old": true}\n', encoding="utf-8")
    failure = OSError("fdopen failed")
    descriptors: list[int] = []

    def fail_fdopen(descriptor: int, mode: str) -> None:
        descriptors.append(descriptor)
        assert mode == "wb"
        raise failure

    monkeypatch.setattr(storage.os, "fdopen", fail_fdopen)

    _assert_same_exception(
        failure,
        lambda: storage.write_json_atomic(path, {"new": True}),
    )
    assert len(descriptors) == 1
    with pytest.raises(OSError) as closed:
        os.fstat(descriptors[0])
    assert closed.value.errno == errno.EBADF
    assert path.read_text(encoding="utf-8") == '{"old": true}\n'
    assert _temporary_aliases(tmp_path, path.name) == []


def test_atomic_temp_cleanup_failure_does_not_mask_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "mutable.json"
    failure = OSError("replace failed")

    monkeypatch.setattr(storage.os, "replace", lambda source, destination: (_ for _ in ()).throw(failure))
    monkeypatch.setattr(storage, "_unlink_best_effort", lambda temporary: False)

    _assert_same_exception(
        failure,
        lambda: storage.write_json_atomic(path, {"value": 1}),
    )


def test_atomic_parent_fsync_failure_raises_after_new_value_is_installed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "mutable.json"
    path.write_text('{"old": true}\n', encoding="utf-8")
    failure = OSError("parent fsync failed")

    monkeypatch.setattr(storage, "_fsync_directory", lambda parent: (_ for _ in ()).throw(failure))

    _assert_same_exception(
        failure,
        lambda: storage.write_json_atomic(path, {"new": True}),
    )
    assert storage.read_json_object(path) == {"new": True}
    assert _temporary_aliases(tmp_path, path.name) == []


def test_exclusive_write_never_overwrites_existing_destination(
    tmp_path: Path,
) -> None:
    path = tmp_path / "immutable.json"
    original = b'{"original": true}\n'
    path.write_bytes(original)

    with pytest.raises(FileExistsError):
        storage.write_json_exclusive(path, {"replacement": True})

    assert path.read_bytes() == original
    assert _temporary_aliases(tmp_path, path.name) == []


def test_exclusive_hard_link_failure_is_precommit_and_removes_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "immutable.json"
    failure = OSError("link failed")

    monkeypatch.setattr(storage.os, "link", lambda source, destination: (_ for _ in ()).throw(failure))

    _assert_same_exception(
        failure,
        lambda: storage.write_json_exclusive(path, {"value": 1}),
    )
    assert not path.exists()
    assert _temporary_aliases(tmp_path, path.name) == []


def test_exclusive_file_fsync_failure_is_precommit_and_removes_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "immutable.json"
    failure = OSError("file fsync failed")

    monkeypatch.setattr(storage.os, "fsync", lambda descriptor: (_ for _ in ()).throw(failure))

    _assert_same_exception(
        failure,
        lambda: storage.write_json_exclusive(path, {"value": 1}),
    )
    assert not path.exists()
    assert _temporary_aliases(tmp_path, path.name) == []


def test_exclusive_first_parent_fsync_failure_reports_unconfirmed_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "immutable.json"
    failure = OSError("first parent fsync failed")

    monkeypatch.setattr(storage, "_fsync_directory", lambda parent: (_ for _ in ()).throw(failure))

    _assert_same_exception(
        failure,
        lambda: storage.write_json_exclusive(path, {"value": 1}),
    )
    assert storage.read_json_object(path) == {"value": 1}
    assert _temporary_aliases(tmp_path, path.name) == []


def test_exclusive_postcommit_alias_unlink_failure_is_suppressed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "immutable.json"
    aliases: list[Path] = []

    def fail_unlink(alias: Path) -> None:
        aliases.append(Path(alias))
        raise OSError("alias unlink failed")

    monkeypatch.setattr(storage.os, "unlink", fail_unlink)

    storage.write_json_exclusive(path, {"committed": True})

    assert storage.read_json_object(path) == {"committed": True}
    assert len(aliases) == 1
    assert aliases[0].exists()
    assert aliases[0].stat().st_ino == path.stat().st_ino


def test_exclusive_second_parent_fsync_failure_is_suppressed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "immutable.json"
    real_fsync_directory = storage._fsync_directory
    calls: list[Path] = []

    def fail_second_fsync(parent: Path) -> None:
        calls.append(parent)
        if len(calls) == 1:
            real_fsync_directory(parent)
            return
        raise OSError("alias-removal fsync failed")

    monkeypatch.setattr(storage, "_fsync_directory", fail_second_fsync)

    storage.write_json_exclusive(path, {"committed": True})

    assert calls == [tmp_path, tmp_path]
    assert storage.read_json_object(path) == {"committed": True}
    assert _temporary_aliases(tmp_path, path.name) == []


def test_create_staging_directory_creates_parent_hidden_unique_directories(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "missing" / "parent"

    first = storage.create_staging_directory(parent, "result")
    second = storage.create_staging_directory(parent, "result")

    assert parent.is_dir()
    assert first.parent == parent
    assert second.parent == parent
    assert first != second
    assert first.is_dir() and not first.is_symlink()
    assert second.is_dir() and not second.is_symlink()
    assert first.name.startswith(".result.staging-")
    assert first.name.removeprefix(".result.staging-")
    assert second.name.startswith(".result.staging-")
    assert second.name.removeprefix(".result.staging-")


@pytest.mark.parametrize("final_name", ("", ".", "..", "nested/name", "nul\0name"))
def test_create_staging_directory_rejects_invalid_final_name(
    tmp_path: Path,
    final_name: str,
) -> None:
    with pytest.raises(ValueError):
        storage.create_staging_directory(tmp_path, final_name)


@pytest.mark.parametrize("final_name", (Path("result"), 7, None))
def test_create_staging_directory_requires_string_final_name(
    tmp_path: Path,
    final_name: object,
) -> None:
    with pytest.raises(TypeError):
        storage.create_staging_directory(tmp_path, final_name)  # type: ignore[arg-type]


def test_create_staging_directory_rejects_symlink_parent(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    alias = tmp_path / "parent-alias"
    alias.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        storage.create_staging_directory(alias, "result")

    assert list(real_parent.iterdir()) == []


def test_create_staging_directory_rejects_file_parent(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    parent.write_text("not a directory", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        storage.create_staging_directory(parent, "result")


def test_create_staging_directory_fsyncs_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    real_fsync_directory = storage._fsync_directory
    calls: list[Path] = []

    def tracked_fsync(directory: Path) -> None:
        calls.append(directory)
        real_fsync_directory(directory)

    monkeypatch.setattr(storage, "_fsync_directory", tracked_fsync)

    staging = storage.create_staging_directory(parent, "result")

    assert staging.is_dir()
    assert calls == [parent]


def test_create_staging_directory_removes_empty_stage_if_parent_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    sibling = parent / "keep.txt"
    sibling.write_text("keep", encoding="utf-8")
    failure = OSError("parent fsync failed")

    monkeypatch.setattr(storage, "_fsync_directory", lambda directory: (_ for _ in ()).throw(failure))

    _assert_same_exception(
        failure,
        lambda: storage.create_staging_directory(parent, "result"),
    )
    assert list(parent.iterdir()) == [sibling]


def test_create_staging_directory_does_not_broadly_clean_nonempty_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    real_mkdtemp = storage.tempfile.mkdtemp
    failure = OSError("parent fsync failed")

    def create_nonempty_stage(*, prefix: str, dir: Path) -> str:
        created = Path(real_mkdtemp(prefix=prefix, dir=dir))
        (created / "unexpected.txt").write_text("keep", encoding="utf-8")
        return os.fspath(created)

    monkeypatch.setattr(storage.tempfile, "mkdtemp", create_nonempty_stage)
    monkeypatch.setattr(storage, "_fsync_directory", lambda directory: (_ for _ in ()).throw(failure))

    _assert_same_exception(
        failure,
        lambda: storage.create_staging_directory(parent, "result"),
    )
    remaining = list(parent.glob(".result.staging-*"))
    assert len(remaining) == 1
    assert (remaining[0] / "unexpected.txt").read_text(encoding="utf-8") == "keep"


def test_publish_directory_fsyncs_tree_bottom_up_and_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = storage.create_staging_directory(tmp_path, "result")
    root_file = staging / "root.txt"
    root_file.write_text("root", encoding="utf-8")
    nested = staging / "nested"
    nested.mkdir()
    nested_file = nested / "payload.txt"
    nested_file.write_text("payload", encoding="utf-8")
    destination = tmp_path / "result"
    real_fsync = storage.os.fsync
    fsynced_paths: list[str] = []

    def tracked_fsync(descriptor: int) -> None:
        fsynced_paths.append(os.readlink(f"/proc/self/fd/{descriptor}"))
        real_fsync(descriptor)

    monkeypatch.setattr(storage.os, "fsync", tracked_fsync)

    storage.publish_directory_exclusive(staging, destination)

    root_file_path = os.fspath(root_file)
    nested_path = os.fspath(nested)
    nested_file_path = os.fspath(nested_file)
    staging_path = os.fspath(staging)
    parent_path = os.fspath(tmp_path)
    assert fsynced_paths.index(root_file_path) < fsynced_paths.index(staging_path)
    assert fsynced_paths.index(nested_file_path) < fsynced_paths.index(nested_path)
    assert fsynced_paths.index(nested_path) < fsynced_paths.index(staging_path)
    assert fsynced_paths.index(staging_path) < fsynced_paths.index(parent_path)
    assert not staging.exists()
    assert (destination / "root.txt").read_text(encoding="utf-8") == "root"
    assert (destination / "nested" / "payload.txt").read_text(encoding="utf-8") == "payload"


def test_publish_directory_tree_traversal_opens_entries_fd_relative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = storage.create_staging_directory(tmp_path, "result")
    (staging / "root.txt").write_text("root", encoding="utf-8")
    nested = staging / "nested"
    nested.mkdir()
    (nested / "payload.txt").write_text("payload", encoding="utf-8")
    destination = tmp_path / "result"
    real_open = storage.os.open
    calls: list[tuple[object, int, int | None]] = []

    def tracked_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        calls.append((path, flags, dir_fd))
        if dir_fd is None:
            return real_open(path, flags, mode)  # type: ignore[arg-type]
        return real_open(path, flags, mode, dir_fd=dir_fd)  # type: ignore[arg-type]

    monkeypatch.setattr(storage.os, "open", tracked_open)

    storage.publish_directory_exclusive(staging, destination)

    relative_calls = {
        os.fspath(path): (flags, dir_fd)
        for path, flags, dir_fd in calls
        if dir_fd is not None
    }
    for name in ("root.txt", "nested", "payload.txt"):
        flags, directory_descriptor = relative_calls[name]
        assert directory_descriptor is not None
        assert flags & os.O_NOFOLLOW
    assert relative_calls["nested"][0] & os.O_DIRECTORY
    assert relative_calls["root.txt"][0] & os.O_NONBLOCK
    assert relative_calls["payload.txt"][0] & os.O_NONBLOCK


def test_publish_directory_rejects_arbitrary_sibling_directory(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "arbitrary-directory"
    staging.mkdir()

    with pytest.raises(ValueError, match="staging directory name"):
        storage.publish_directory_exclusive(staging, tmp_path / "result")

    assert staging.is_dir()
    assert not (tmp_path / "result").exists()


def test_publish_directory_rejects_same_staging_and_destination(
    tmp_path: Path,
) -> None:
    staging = storage.create_staging_directory(tmp_path, "result")

    with pytest.raises(ValueError, match="must be different"):
        storage.publish_directory_exclusive(staging, staging)

    assert staging.is_dir()


def test_publish_directory_rejects_different_parents(tmp_path: Path) -> None:
    first_parent = tmp_path / "first"
    second_parent = tmp_path / "second"
    staging = storage.create_staging_directory(first_parent, "result")
    second_parent.mkdir()

    with pytest.raises(ValueError, match="same parent"):
        storage.publish_directory_exclusive(staging, second_parent / "result")

    assert staging.is_dir()


def test_publish_directory_normalizes_relative_lexical_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    staging = storage.create_staging_directory(parent, "result")
    (staging / "payload.txt").write_text("payload", encoding="utf-8")
    (parent / "unused").mkdir()
    monkeypatch.chdir(tmp_path)
    lexical_staging = Path("parent") / "unused" / ".." / staging.name
    destination = Path("parent") / "result"

    storage.publish_directory_exclusive(lexical_staging, destination)

    assert (parent / "result" / "payload.txt").read_text(encoding="utf-8") == "payload"


def test_publish_directory_detects_same_path_after_lexical_normalization(
    tmp_path: Path,
) -> None:
    staging = storage.create_staging_directory(tmp_path, "result")
    intermediary = tmp_path / "intermediary"
    intermediary.mkdir()
    lexical_alias = intermediary / ".." / staging.name

    with pytest.raises(ValueError, match="must be different"):
        storage.publish_directory_exclusive(staging, lexical_alias)

    assert staging.is_dir()


def test_publish_directory_rejects_symlink_parent_without_resolving_it(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    staging = storage.create_staging_directory(real_parent, "result")
    (staging / "payload.txt").write_text("payload", encoding="utf-8")
    alias = tmp_path / "parent-alias"
    alias.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        storage.publish_directory_exclusive(
            alias / staging.name,
            alias / "result",
        )

    assert staging.is_dir()
    assert not (real_parent / "result").exists()


def test_publish_directory_rejects_symlink_staging_path(
    tmp_path: Path,
) -> None:
    real_staging = tmp_path / "real-staging"
    real_staging.mkdir()
    staging = tmp_path / ".result.staging-alias"
    staging.symlink_to(real_staging, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        storage.publish_directory_exclusive(staging, tmp_path / "result")

    assert staging.is_symlink()
    assert not (tmp_path / "result").exists()


def test_publish_directory_rejects_regular_file_staging_path(
    tmp_path: Path,
) -> None:
    staging = tmp_path / ".result.staging-file"
    staging.write_text("not a directory", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        storage.publish_directory_exclusive(staging, tmp_path / "result")

    assert staging.is_file()


def test_publish_directory_rejects_nested_symlink_without_following_target(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    staging = storage.create_staging_directory(tmp_path, "result")
    (staging / "link").symlink_to(outside)

    with pytest.raises(ValueError, match="contains a symlink"):
        storage.publish_directory_exclusive(staging, tmp_path / "result")

    assert outside.read_text(encoding="utf-8") == "outside"
    assert staging.is_dir()
    assert not (tmp_path / "result").exists()


def test_publish_directory_rejects_fifo(tmp_path: Path) -> None:
    staging = storage.create_staging_directory(tmp_path, "result")
    fifo = staging / "fifo"
    os.mkfifo(fifo)

    with pytest.raises(ValueError, match="special file"):
        storage.publish_directory_exclusive(staging, tmp_path / "result")

    assert stat.S_ISFIFO(fifo.lstat().st_mode)
    assert not (tmp_path / "result").exists()


def test_publish_directory_does_not_replace_existing_empty_directory(
    tmp_path: Path,
) -> None:
    staging = storage.create_staging_directory(tmp_path, "result")
    (staging / "payload.txt").write_text("payload", encoding="utf-8")
    destination = tmp_path / "result"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        storage.publish_directory_exclusive(staging, destination)

    assert staging.is_dir()
    assert destination.is_dir()
    assert list(destination.iterdir()) == []


def test_publish_directory_does_not_replace_existing_symlink(
    tmp_path: Path,
) -> None:
    staging = storage.create_staging_directory(tmp_path, "result")
    target = tmp_path / "target"
    target.mkdir()
    destination = tmp_path / "result"
    destination.symlink_to(target, target_is_directory=True)

    with pytest.raises(FileExistsError):
        storage.publish_directory_exclusive(staging, destination)

    assert destination.is_symlink()
    assert staging.is_dir()


def test_publish_final_rename_uses_leaf_names_and_same_parent_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = storage.create_staging_directory(tmp_path, "result")
    (staging / "payload.txt").write_text("payload", encoding="utf-8")
    destination = tmp_path / "result"
    rename_calls: list[tuple[int, bytes, int, bytes, int]] = []
    renamed = False
    fsync_after_rename: list[int] = []

    def fake_renameat2(
        source_directory: object,
        source_name: object,
        destination_directory: object,
        destination_name: object,
        flags: object,
    ) -> int:
        nonlocal renamed
        call = (
            int(source_directory),
            bytes(source_name),
            int(destination_directory),
            bytes(destination_name),
            int(flags),
        )
        rename_calls.append(call)
        os.rename(
            call[1],
            call[3],
            src_dir_fd=call[0],
            dst_dir_fd=call[2],
        )
        renamed = True
        return 0

    fake_library = _FakeLibrary(fake_renameat2)
    monkeypatch.setattr(storage.ctypes, "CDLL", lambda *args, **kwargs: fake_library)
    real_fsync = storage.os.fsync

    def tracked_fsync(descriptor: int) -> None:
        if renamed:
            fsync_after_rename.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(storage.os, "fsync", tracked_fsync)

    storage.publish_directory_exclusive(staging, destination)

    assert len(rename_calls) == 1
    parent_fd, source_name, destination_fd, destination_name, flags = rename_calls[0]
    assert parent_fd == destination_fd
    assert parent_fd != -100  # Linux AT_FDCWD
    assert source_name == os.fsencode(staging.name)
    assert destination_name == os.fsencode(destination.name)
    assert b"/" not in source_name and b"/" not in destination_name
    assert flags == 1
    assert fsync_after_rename == [parent_fd]
    with pytest.raises(OSError) as closed:
        os.fstat(parent_fd)
    assert closed.value.errno == errno.EBADF
    assert (destination / "payload.txt").read_text(encoding="utf-8") == "payload"


def test_publish_translates_kernel_enosys_to_not_implemented(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = storage.create_staging_directory(tmp_path, "result")
    destination = tmp_path / "result"

    def unavailable_renameat2(*args: object) -> int:
        ctypes.set_errno(errno.ENOSYS)
        return -1

    fake_library = _FakeLibrary(unavailable_renameat2)
    monkeypatch.setattr(storage.ctypes, "CDLL", lambda *args, **kwargs: fake_library)

    with pytest.raises(NotImplementedError, match="not supported by this kernel"):
        storage.publish_directory_exclusive(staging, destination)

    assert staging.is_dir()
    assert not destination.exists()


def test_publish_reports_missing_libc_renameat2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = storage.create_staging_directory(tmp_path, "result")
    destination = tmp_path / "result"
    monkeypatch.setattr(storage.ctypes, "CDLL", lambda *args, **kwargs: object())

    with pytest.raises(NotImplementedError, match="libc does not provide renameat2"):
        storage.publish_directory_exclusive(staging, destination)

    assert staging.is_dir()
    assert not destination.exists()


def test_publish_parent_fsync_failure_raises_after_successful_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = storage.create_staging_directory(tmp_path, "result")
    (staging / "payload.txt").write_text("payload", encoding="utf-8")
    destination = tmp_path / "result"
    real_rename = storage._rename_directory_noreplace
    real_fsync = storage.os.fsync
    renamed = False
    failure = OSError("publication parent fsync failed")

    def tracked_rename(
        parent_descriptor: int,
        source: Path,
        target: Path,
    ) -> None:
        nonlocal renamed
        real_rename(parent_descriptor, source, target)
        renamed = True

    def fail_after_rename(descriptor: int) -> None:
        if renamed:
            raise failure
        real_fsync(descriptor)

    monkeypatch.setattr(storage, "_rename_directory_noreplace", tracked_rename)
    monkeypatch.setattr(storage.os, "fsync", fail_after_rename)

    _assert_same_exception(
        failure,
        lambda: storage.publish_directory_exclusive(staging, destination),
    )
    assert not staging.exists()
    assert (destination / "payload.txt").read_text(encoding="utf-8") == "payload"


def test_cleanup_staging_directory_removes_only_supplied_tree_and_fsyncs_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = storage.create_staging_directory(tmp_path, "result")
    sibling = storage.create_staging_directory(tmp_path, "result")
    (staging / "nested").mkdir()
    (staging / "nested" / "payload.txt").write_text("payload", encoding="utf-8")
    real_fsync_directory = storage._fsync_directory
    calls: list[Path] = []

    def tracked_fsync(parent: Path) -> None:
        calls.append(parent)
        real_fsync_directory(parent)

    monkeypatch.setattr(storage, "_fsync_directory", tracked_fsync)

    storage.cleanup_staging_directory(staging)

    assert not staging.exists()
    assert sibling.is_dir()
    assert calls == [tmp_path]


def test_cleanup_staging_directory_does_not_follow_nested_symlink(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "keep.txt"
    outside_file.write_text("keep", encoding="utf-8")
    staging = storage.create_staging_directory(tmp_path, "result")
    (staging / "outside-link").symlink_to(outside, target_is_directory=True)

    storage.cleanup_staging_directory(staging)

    assert not staging.exists()
    assert outside_file.read_text(encoding="utf-8") == "keep"


def test_cleanup_staging_directory_is_idempotent_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / ".result.staging-missing"

    def unexpected_fsync(parent: Path) -> None:
        raise AssertionError("missing cleanup must not fsync an unchanged parent")

    monkeypatch.setattr(storage, "_fsync_directory", unexpected_fsync)

    storage.cleanup_staging_directory(missing)
    storage.cleanup_staging_directory(missing)


@pytest.mark.parametrize(
    "name",
    (
        "arbitrary-directory",
        ".result.staging-",
        ".staging-random",
        "result.staging-random",
        ".result.stage-random",
    ),
)
def test_cleanup_staging_directory_rejects_invalid_name_without_deleting(
    tmp_path: Path,
    name: str,
) -> None:
    path = tmp_path / name
    path.mkdir()

    with pytest.raises(ValueError, match="staging directory name"):
        storage.cleanup_staging_directory(path)

    assert path.is_dir()


def test_cleanup_staging_directory_rejects_symlink_path(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    staging = tmp_path / ".result.staging-link"
    staging.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="refusing to clean up"):
        storage.cleanup_staging_directory(staging)

    assert staging.is_symlink()
    assert target.is_dir()


def test_cleanup_staging_directory_rejects_non_directory(
    tmp_path: Path,
) -> None:
    staging = tmp_path / ".result.staging-file"
    staging.write_text("keep", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        storage.cleanup_staging_directory(staging)

    assert staging.read_text(encoding="utf-8") == "keep"


def test_cleanup_parent_fsync_failure_raises_after_tree_is_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = storage.create_staging_directory(tmp_path, "result")
    (staging / "payload.txt").write_text("payload", encoding="utf-8")
    failure = OSError("cleanup parent fsync failed")
    monkeypatch.setattr(storage, "_fsync_directory", lambda parent: (_ for _ in ()).throw(failure))

    _assert_same_exception(
        failure,
        lambda: storage.cleanup_staging_directory(staging),
    )
    assert not staging.exists()
