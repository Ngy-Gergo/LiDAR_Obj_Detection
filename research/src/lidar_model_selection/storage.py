"""Strict, durable JSON and staged-directory persistence helpers."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


__all__ = (
    "read_json_object",
    "write_json_atomic",
    "write_json_exclusive",
    "create_staging_directory",
    "publish_directory_exclusive",
    "cleanup_staging_directory",
)

_RENAME_NOREPLACE = 1
_STAGING_MARKER = ".staging-"


def _strict_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(constant: str) -> None:
    raise ValueError(f"invalid JSON constant: {constant}")


def read_json_object(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON file whose root is a strict JSON object."""
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(
            stream,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )

    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _validate_json_object_keys(
    value: object,
    active_containers: set[int] | None = None,
) -> None:
    if not isinstance(value, (Mapping, list, tuple)):
        return

    if active_containers is None:
        active_containers = set()
    identity = id(value)
    if identity in active_containers:
        return

    active_containers.add(identity)
    try:
        if isinstance(value, Mapping):
            for key, nested_value in value.items():
                if not isinstance(key, str):
                    raise TypeError(
                        "JSON object keys must be strings, got "
                        f"{type(key).__name__}"
                    )
                _validate_json_object_keys(
                    nested_value,
                    active_containers,
                )
        else:
            for nested_value in value:
                _validate_json_object_keys(
                    nested_value,
                    active_containers,
                )
    finally:
        active_containers.remove(identity)


def _serialize_json(value: Mapping[str, object]) -> bytes:
    if not isinstance(value, Mapping):
        raise TypeError("JSON value must be a mapping")
    _validate_json_object_keys(value)
    serialized = json.dumps(
        dict(value),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    return (serialized + "\n").encode("utf-8")


def _validate_path(path: Path, description: str) -> None:
    raw_path = os.fspath(path)
    if not isinstance(raw_path, str):
        raise TypeError(f"{description} must be a text path")
    if "\0" in raw_path:
        raise ValueError(f"{description} must not contain NUL")


def _validate_leaf_path(path: Path, description: str) -> None:
    _validate_path(path, description)
    if path.name in {"", ".", ".."}:
        raise ValueError(f"{description} must name one filesystem entry")


def _unlink_best_effort(path: Path) -> bool:
    try:
        os.unlink(path)
    except Exception:
        return False
    return True


def _write_temporary_file(
    parent: Path,
    destination_name: str,
    payload: bytes,
) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_name}.tmp-",
        dir=parent,
    )
    temporary_path = Path(temporary_name)

    try:
        try:
            stream = os.fdopen(descriptor, "wb")
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise

        with stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        _unlink_best_effort(temporary_path)
        raise

    return temporary_path


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_atomic(
    path: Path,
    value: Mapping[str, object],
) -> None:
    """Atomically replace a mutable JSON object and make it durable."""
    payload = _serialize_json(value)
    _validate_leaf_path(path, "JSON destination")
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _write_temporary_file(
        parent,
        path.name,
        payload,
    )

    try:
        os.replace(temporary_path, path)
    except BaseException:
        _unlink_best_effort(temporary_path)
        raise

    _fsync_directory(parent)


def write_json_exclusive(
    path: Path,
    value: Mapping[str, object],
) -> None:
    """Publish an immutable JSON object without replacing any destination."""
    payload = _serialize_json(value)
    _validate_leaf_path(path, "JSON destination")
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _write_temporary_file(
        parent,
        path.name,
        payload,
    )

    try:
        os.link(temporary_path, path)
        _fsync_directory(parent)
    except BaseException:
        _unlink_best_effort(temporary_path)
        raise

    if not _unlink_best_effort(temporary_path):
        return

    try:
        _fsync_directory(parent)
    except Exception:
        pass


def _validate_final_name(final_name: str) -> None:
    if not isinstance(final_name, str):
        raise TypeError("final_name must be a string")
    if final_name in {"", ".", ".."}:
        raise ValueError("final_name must be one non-special path component")
    if "\0" in final_name:
        raise ValueError("final_name must not contain NUL")
    if Path(final_name).name != final_name:
        raise ValueError("final_name must be one path component")


def _require_real_directory(
    path: Path,
    *,
    description: str,
) -> None:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{description} must not be a symlink: {path}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise NotADirectoryError(
            errno.ENOTDIR,
            f"{description} is not a directory",
            path,
        )


def create_staging_directory(
    parent: Path,
    final_name: str,
) -> Path:
    """Create a hidden, uniquely named staging directory under *parent*."""
    _validate_final_name(final_name)
    _validate_path(parent, "staging parent")
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass
    _require_real_directory(parent, description="staging parent")
    created_path = Path(
        tempfile.mkdtemp(
            prefix=f".{final_name}{_STAGING_MARKER}",
            dir=parent,
        )
    )
    staging = parent / created_path.name
    try:
        _fsync_directory(parent)
    except BaseException:
        try:
            os.rmdir(created_path)
        except BaseException:
            pass
        raise
    return staging


def _fsync_regular_file_at(
    parent_descriptor: int,
    name: str,
    path: Path,
) -> None:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
        dir_fd=parent_descriptor,
    )
    try:
        file_status = os.fstat(descriptor)
        if not stat.S_ISREG(file_status.st_mode):
            raise ValueError(
                f"staging entry is not a regular file: {path}"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_open_directory_tree(
    directory_descriptor: int,
    path: Path,
) -> None:
    with os.scandir(directory_descriptor) as entries:
        for entry in entries:
            entry_path = path / entry.name
            entry_status = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(entry_status.st_mode):
                raise ValueError(
                    f"staging tree contains a symlink: {entry_path}"
                )
            if stat.S_ISREG(entry_status.st_mode):
                _fsync_regular_file_at(
                    directory_descriptor,
                    entry.name,
                    entry_path,
                )
            elif stat.S_ISDIR(entry_status.st_mode):
                child_descriptor = os.open(
                    entry.name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_descriptor,
                )
                try:
                    child_status = os.fstat(child_descriptor)
                    if not stat.S_ISDIR(child_status.st_mode):
                        raise ValueError(
                            "staging entry is not a directory: "
                            f"{entry_path}"
                        )
                    _fsync_open_directory_tree(
                        child_descriptor,
                        entry_path,
                    )
                finally:
                    os.close(child_descriptor)
            else:
                raise ValueError(
                    "staging tree contains a special file: "
                    f"{entry_path}"
                )

    os.fsync(directory_descriptor)


def _fsync_staging_tree(path: Path) -> None:
    path_status = os.lstat(path)
    if stat.S_ISLNK(path_status.st_mode):
        raise ValueError(f"staging tree contains a symlink: {path}")
    if not stat.S_ISDIR(path_status.st_mode):
        raise NotADirectoryError(
            errno.ENOTDIR,
            "staging path is not a directory",
            path,
        )

    directory_descriptor = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        opened_status = os.fstat(directory_descriptor)
        if not stat.S_ISDIR(opened_status.st_mode):
            raise ValueError(f"staging path is not a directory: {path}")
        _fsync_open_directory_tree(directory_descriptor, path)
    finally:
        os.close(directory_descriptor)


def _encoded_name(name: str) -> bytes:
    if "\0" in name:
        raise ValueError("name passed to renameat2 must not contain NUL")
    return os.fsencode(name)


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _rename_directory_noreplace(
    parent_descriptor: int,
    staging: Path,
    destination: Path,
) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = library.renameat2
    except AttributeError as error:
        raise NotImplementedError(
            "libc does not provide renameat2"
        ) from error

    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int

    ctypes.set_errno(0)
    result = renameat2(
        parent_descriptor,
        _encoded_name(staging.name),
        parent_descriptor,
        _encoded_name(destination.name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return

    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            destination,
        )
    if error_number == errno.ENOSYS:
        raise NotImplementedError(
            "renameat2 is not supported by this kernel"
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        os.fspath(staging),
        None,
        os.fspath(destination),
    )


def publish_directory_exclusive(
    staging: Path,
    destination: Path,
) -> None:
    """Durably publish a staged directory with no-replace semantics."""
    staging = _absolute_path(staging)
    destination = _absolute_path(destination)
    _validate_staging_path(staging)
    _validate_leaf_path(destination, "directory destination")
    if staging == destination:
        raise ValueError("staging and destination must be different")
    if staging.parent != destination.parent:
        raise ValueError("staging and destination must have the same parent")

    parent = staging.parent
    _require_real_directory(parent, description="publication parent")
    _require_real_directory(staging, description="staging path")
    _fsync_staging_tree(staging)
    parent_descriptor = os.open(
        parent,
        os.O_RDONLY | os.O_DIRECTORY,
    )
    try:
        _rename_directory_noreplace(
            parent_descriptor,
            staging,
            destination,
        )
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _validate_staging_path(path: Path) -> None:
    _validate_leaf_path(path, "staging cleanup path")
    name = path.name
    if not name.startswith("."):
        raise ValueError("path does not have a staging directory name")

    final_name, marker, random_suffix = name[1:].rpartition(
        _STAGING_MARKER
    )
    if not marker or not final_name or not random_suffix:
        raise ValueError("path does not have a staging directory name")
    _validate_final_name(final_name)


def cleanup_staging_directory(path: Path) -> None:
    """Remove one explicitly named staging directory, if it exists."""
    _validate_staging_path(path)
    try:
        path_status = os.lstat(path)
    except FileNotFoundError:
        return

    if stat.S_ISLNK(path_status.st_mode):
        raise ValueError("refusing to clean up a staging-directory symlink")
    if not stat.S_ISDIR(path_status.st_mode):
        raise NotADirectoryError(
            errno.ENOTDIR,
            "staging cleanup path is not a directory",
            path,
        )

    shutil.rmtree(path)
    _fsync_directory(path.parent)
