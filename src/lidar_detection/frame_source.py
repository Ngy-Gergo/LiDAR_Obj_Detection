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
