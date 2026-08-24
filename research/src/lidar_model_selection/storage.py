"""Strict JSON and durable filesystem publication primitives.

The directory-publication primitive intentionally targets Linux. Python does
not expose Linux's no-replace rename operation, so this module uses the libc
``renameat2`` entry point rather than falling back to a racy existence check.

Durability guarantees apply to entries in the immediate parent directory.
Callers should create long-lived persistence roots before publishing beneath
them; recursively-created ancestors are not individually synced here.
"""

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
from typing import Any, NoReturn

__all__ = [
    "cleanup_staging_directory",
    "create_staging_directory",
    "publish_directory_exclusive",
    "read_json_object",
    "write_json_atomic",
    "write_json_exclusive",
]

_RENAME_NOREPLACE = 1
_STAGING_MARKER = ".staging-"

try:
    _RENAMEAT2 = ctypes.CDLL(None, use_errno=True).renameat2
except (AttributeError, OSError):
    _RENAMEAT2 = None
else:
    _RENAMEAT2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    _RENAMEAT2.restype = ctypes.c_int


def read_json_object(path: Path) -> dict[str, Any]:
    """Read *path* as strict UTF-8 JSON and require an object at its root."""
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(
            stream,
            parse_constant=_reject_non_finite_constant,
            object_pairs_hook=_strict_object,
        )

    if not isinstance(value, dict):
        raise ValueError(
            f"expected JSON object in {path}, got {type(value).__name__}"
        )

    return value


def write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    """Atomically replace *path* with a fully written and synced JSON object.

    A failure before ``os.replace`` leaves any existing destination intact.  A
    failure while syncing the parent after replacement is reported, although
    the complete replacement may already be visible.
    """
    payload = _encode_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _write_fsynced_temporary(path, payload)

    try:
        os.replace(temporary, path)
    except BaseException:
        _unlink_best_effort(temporary)
        raise

    _fsync_directory(path.parent)


def write_json_exclusive(path: Path, value: Mapping[str, object]) -> None:
    """Publish a synced JSON object without replacing any existing entry.

    Publication uses a same-directory hard link, whose creation fails
    atomically with ``FileExistsError`` when the destination already exists.
    The destination is committed once its link has been synced in the parent;
    removing the temporary alias after that point is best effort.
    """
    payload = _encode_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _write_fsynced_temporary(path, payload)

    try:
        os.link(temporary, path)
    except BaseException:
        _unlink_best_effort(temporary)
        raise

    try:
        _fsync_directory(path.parent)
    except BaseException:
        _unlink_best_effort(temporary)
        raise

    # The immutable destination is now durably committed. Failure to remove or
    # durably forget the temporary alias must not report publication failure.
    try:
        temporary.unlink()
    except OSError:
        return

    try:
        _fsync_directory(path.parent)
    except OSError:
        pass


def create_staging_directory(parent: Path, final_name: str) -> Path:
    """Create and return a hidden, unique staging directory below *parent*."""
    _validate_final_name(final_name)
    parent.mkdir(parents=True, exist_ok=True)
    _require_real_directory(parent, description="staging parent")

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{final_name}{_STAGING_MARKER}",
            dir=parent,
        )
    )

    try:
        _fsync_directory(parent)
    except BaseException:
        try:
            staging.rmdir()
        except OSError:
            pass
        raise

    return staging


def publish_directory_exclusive(staging: Path, destination: Path) -> None:
    """Durably publish a sibling staging tree without replacing a destination.

    Regular files and directories in the staging tree are synced bottom-up.
    Symlinks and special files are rejected rather than followed. This helper
    is intended for small metadata/config trees; callers must not populate it
    with large data trees or mutate it concurrently with publication.
    """
    staging = _absolute_path(staging)
    destination = _absolute_path(destination)

    if staging == destination:
        raise ValueError("staging and destination must be different paths")
    if staging.parent != destination.parent:
        raise ValueError("staging and destination must share the same parent")
    _require_staging_name(staging)
    _require_leaf_path(destination, description="destination")

    parent = staging.parent
    _require_real_directory(parent, description="publication parent")
    _require_real_directory(staging, description="staging path")

    _fsync_staging_tree(staging)

    parent_fd = _open_directory(parent)
    try:
        _rename_noreplace(
            parent_fd,
            staging.name,
            destination.name,
            destination,
        )
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def cleanup_staging_directory(path: Path) -> None:
    """Remove only the explicitly supplied staging tree, without scanning."""
    _require_staging_name(path)
    try:
        _require_real_directory(path, description="staging path")
    except FileNotFoundError:
        return

    shutil.rmtree(path)
    _fsync_directory(path.parent)


def _encode_json(value: Mapping[str, object]) -> bytes:
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a mapping, got {type(value).__name__}")
    _require_string_object_keys(value)

    text = json.dumps(
        dict(value),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def _reject_non_finite_constant(constant: str) -> NoReturn:
    raise ValueError(f"non-finite JSON number is not permitted: {constant}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _require_string_object_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"JSON object keys must be strings, got {type(key).__name__}"
                )
            _require_string_object_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _require_string_object_keys(item)


def _write_fsynced_temporary(path: Path, payload: bytes) -> Path:
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)

    try:
        stream = os.fdopen(fd, "wb")
        fd = -1
        with stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        _unlink_best_effort(temporary)
        raise

    return temporary


def _unlink_best_effort(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        # Preserve the operation's primary failure. An orphaned hidden
        # temporary file is preferable to masking the cause of the failure.
        pass


def _fsync_staging_tree(path: Path) -> None:
    _require_real_directory(path, description="staging directory")

    with os.scandir(path) as iterator:
        entries = list(iterator)

    for entry in entries:
        child = Path(entry.path)
        if entry.is_symlink():
            raise ValueError(f"staging tree contains a symlink: {child}")
        if entry.is_dir(follow_symlinks=False):
            _fsync_staging_tree(child)
        elif entry.is_file(follow_symlinks=False):
            _fsync_regular_file(child)
        else:
            raise ValueError(
                f"staging tree contains an unsupported file type: {child}"
            )

    _fsync_directory(path)


def _fsync_regular_file(path: Path) -> None:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise NotImplementedError("safe staged-file syncing requires O_NOFOLLOW")

    fd = os.open(path, os.O_RDONLY | no_follow)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"staging entry is not a regular file: {path}")
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_directory(path: Path) -> None:
    fd = _open_directory(path)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _open_directory(path: Path) -> int:
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if directory_flag is None:
        raise NotImplementedError("directory syncing requires O_DIRECTORY")
    return os.open(path, os.O_RDONLY | directory_flag)


def _require_real_directory(path: Path, *, description: str) -> os.stat_result:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{description} must not be a symlink: {path}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise NotADirectoryError(
            errno.ENOTDIR,
            f"{description} is not a directory",
            os.fspath(path),
        )
    return metadata


def _validate_final_name(final_name: str) -> None:
    if (
        not final_name
        or final_name in {".", ".."}
        or "\x00" in final_name
        or Path(final_name).name != final_name
    ):
        raise ValueError("final_name must be one non-empty path component")


def _require_staging_name(path: Path) -> None:
    _require_leaf_path(path, description="staging path")
    prefix, marker, suffix = path.name.rpartition(_STAGING_MARKER)
    if not marker or not prefix.startswith(".") or len(prefix) == 1 or not suffix:
        raise ValueError(
            "staging path name must match the create_staging_directory pattern"
        )


def _require_leaf_path(path: Path, *, description: str) -> None:
    if not path.name or path.name in {".", ".."} or "\x00" in path.name:
        raise ValueError(f"{description} must name a directory entry")


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _rename_noreplace(
    parent_fd: int,
    source_name: str,
    destination_name: str,
    destination: Path,
) -> None:
    if "\x00" in source_name or "\x00" in destination_name:
        raise ValueError("directory publication names must not contain NUL")

    if _RENAMEAT2 is None:
        raise NotImplementedError(
            "exclusive directory publication requires Linux renameat2"
        )

    ctypes.set_errno(0)
    result = _RENAMEAT2(
        parent_fd,
        os.fsencode(source_name),
        parent_fd,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return

    error_number = ctypes.get_errno()
    if error_number == errno.ENOSYS:
        raise NotImplementedError(
            "the running Linux kernel does not support renameat2"
        )
    if error_number == errno.EEXIST:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            os.fspath(destination),
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        os.fspath(destination),
    )
