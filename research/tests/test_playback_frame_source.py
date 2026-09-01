from pathlib import Path

import pytest

from lidar_model_selection.playback.frame_source import resolve_session_directory


def test_resolve_session_preserves_exact_immediate_basename(tmp_path: Path) -> None:
    root = tmp_path / "recordings"
    session = root / "proba_lexus3_2026-07-27_14-09"
    session.mkdir(parents=True)

    resolved = resolve_session_directory(root.resolve(), session.name)

    assert resolved == session.resolve()
    assert resolved.name == "proba_lexus3_2026-07-27_14-09"


@pytest.mark.parametrize(
    "session_id",
    ("../outside", "nested/session", ".", "..", "/absolute/session"),
)
def test_resolve_session_rejects_traversal_and_nested_paths(
    tmp_path: Path,
    session_id: str,
) -> None:
    root = tmp_path / "recordings"
    root.mkdir()

    with pytest.raises(ValueError, match="immediate directory basename"):
        resolve_session_directory(root.resolve(), session_id)


def test_resolve_session_rejects_missing_non_directory_and_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "recordings"
    root.mkdir()
    (root / "file").write_text("not a session", encoding="utf-8")
    target = root / "target"
    target.mkdir()
    (root / "alias").symlink_to(target, target_is_directory=True)

    with pytest.raises(FileNotFoundError, match="not an immediate"):
        resolve_session_directory(root.resolve(), "missing")
    with pytest.raises(NotADirectoryError, match="not a directory"):
        resolve_session_directory(root.resolve(), "file")
    with pytest.raises(ValueError, match="symbolic link"):
        resolve_session_directory(root.resolve(), "alias")


def test_resolve_session_requires_absolute_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        resolve_session_directory(Path("recordings"), "session")
