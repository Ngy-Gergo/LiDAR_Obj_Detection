from dataclasses import dataclass
from numbers import Integral
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LidarFrame:
    frame_id: str
    path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str):
            raise TypeError("frame_id must be a string")
        if not self.frame_id.strip():
            raise ValueError("frame_id must contain at least one non-whitespace character")
        if not isinstance(self.path, Path):
            raise TypeError("path must be a pathlib.Path")


@dataclass(frozen=True, slots=True)
class DirectoryFrameSource:
    directory: Path
    extension: str = ".bin"

    def __post_init__(self) -> None:
        if not isinstance(self.directory, Path):
            raise TypeError("directory must be a pathlib.Path")
        if not self.directory.exists():
            raise FileNotFoundError(f"directory does not exist: {self.directory}")
        if not self.directory.is_dir():
            raise NotADirectoryError(f"directory is not a directory: {self.directory}")

        if not isinstance(self.extension, str):
            raise TypeError("extension must be a string")
        if not self.extension or not self.extension.strip():
            raise ValueError("extension must contain at least one non-whitespace character")

        normalized_extension = (
            self.extension if self.extension.startswith(".") else f".{self.extension}"
        )
        if not normalized_extension[1:].strip():
            raise ValueError("extension must contain non-whitespace content after the dot")
        object.__setattr__(self, "extension", normalized_extension)

    def list_frames(
        self,
        limit: int | None = None,
    ) -> tuple[LidarFrame, ...]:
        if isinstance(limit, bool) or (limit is not None and not isinstance(limit, Integral)):
            raise TypeError("limit must be an integer or None, and not a boolean")
        if limit is not None and limit < 0:
            raise ValueError("limit must be greater than or equal to zero")
        if limit == 0:
            return ()

        matching_paths = sorted(
            (
                path
                for path in self.directory.iterdir()
                if path.is_file() and path.suffix == self.extension
            ),
            key=lambda path: path.name,
        )
        selected_paths = matching_paths if limit is None else matching_paths[:limit]
        return tuple(LidarFrame(path.stem, path) for path in selected_paths)


def resolve_session_directory(recording_root: Path, session_id: str) -> Path:
    """Resolve one exact, non-symlinked immediate recording-root child."""

    if not isinstance(recording_root, Path):
        raise TypeError("recording_root must be a pathlib.Path")
    if not recording_root.is_absolute():
        raise ValueError("recording_root must be an absolute path")
    if not recording_root.exists():
        raise FileNotFoundError(f"recording root does not exist: {recording_root}")
    if not recording_root.is_dir():
        raise NotADirectoryError(
            f"recording root is not a directory: {recording_root}"
        )
    if not isinstance(session_id, str):
        raise TypeError("session_id must be a string")
    if not session_id or not session_id.strip():
        raise ValueError("session_id must contain non-whitespace text")

    session_component = Path(session_id)
    if (
        session_component.is_absolute()
        or session_component.name != session_id
        or session_id in {".", ".."}
    ):
        raise ValueError(
            "session_id must be one exact immediate directory basename"
        )

    root = recording_root.resolve(strict=True)
    matches = tuple(entry for entry in recording_root.iterdir() if entry.name == session_id)
    if not matches:
        raise FileNotFoundError(
            f"session is not an immediate recording-root child: {session_id}"
        )
    if len(matches) != 1:
        raise ValueError(f"session name is ambiguous: {session_id}")

    candidate = matches[0]
    if candidate.is_symlink():
        raise ValueError(f"session directory must not be a symbolic link: {candidate}")
    if not candidate.is_dir():
        raise NotADirectoryError(f"session path is not a directory: {candidate}")

    resolved = candidate.resolve(strict=True)
    if resolved.parent != root or resolved.name != session_id:
        raise ValueError(
            "session must resolve to the named immediate recording-root child"
        )
    return resolved
