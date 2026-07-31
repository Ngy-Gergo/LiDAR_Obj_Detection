"""Shared CenterPoint model definitions and checkpoint discovery."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import NamedTuple, Sequence


RESEARCH_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = RESEARCH_ROOT / "configs/centerpoint"
EXPERIMENT_ROOT = RESEARCH_ROOT / "experiments"
EXPERIMENT_PATTERN = re.compile(r"(?P<model>.+)_screen(?P<epochs>\d+)")
CHECKPOINT_EPOCH_PATTERN = re.compile(r"(?:^|_)epoch_(?P<epoch>\d+)\.pth")
LATEST_EPOCH_PATTERN = re.compile(r"epoch_(?P<epoch>\d+)\.pth")


class ModelSpec(NamedTuple):
    name: str
    config_path: Path
    candidate_checkpoint: Path | None = None


class CheckpointChoice(NamedTuple):
    path: Path
    selection: str
    trained_epochs: int | None


CENTERPOINT_MODELS = (
    ModelSpec(
        "pillar02",
        CONFIG_ROOT / "pillar02.py",
        EXPERIMENT_ROOT / "pillar02_full/epoch_10.pth",
    ),
    ModelSpec("pillar02_dcn", CONFIG_ROOT / "pillar02_dcn.py"),
    ModelSpec("voxel01", CONFIG_ROOT / "voxel01.py"),
    ModelSpec("voxel01_dcn", CONFIG_ROOT / "voxel01_dcn.py"),
    ModelSpec("voxel0075", CONFIG_ROOT / "voxel0075.py"),
    ModelSpec("voxel0075_dcn", CONFIG_ROOT / "voxel0075_dcn.py"),
)


def checkpoint_epoch(path: Path) -> int | None:
    match = CHECKPOINT_EPOCH_PATTERN.search(path.name)
    return int(match.group("epoch")) if match else None


def is_usable_checkpoint(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        with zipfile.ZipFile(path) as archive:
            members = {Path(name).name for name in archive.namelist()}
        return {"data.pkl", "version"}.issubset(members)
    except (OSError, zipfile.BadZipFile):
        return False


def _experiment_directories(model_name: str) -> list[Path]:
    directories = {EXPERIMENT_ROOT / f"{model_name}_screen10"}
    if EXPERIMENT_ROOT.is_dir():
        for path in EXPERIMENT_ROOT.iterdir():
            match = EXPERIMENT_PATTERN.fullmatch(path.name)
            if path.is_dir() and match and match.group("model") == model_name:
                directories.add(path)
    return sorted(
        directories,
        key=lambda path: (
            int(EXPERIMENT_PATTERN.fullmatch(path.name).group("epochs")),
            path.as_posix(),
        ),
    )


def _highest_epoch(paths: Sequence[Path]) -> Path:
    return max(
        paths,
        key=lambda path: (
            -1
            if (epoch := checkpoint_epoch(path)) is None
            else epoch,
            path.as_posix(),
        ),
    )


def discover_checkpoint(spec: ModelSpec) -> CheckpointChoice | None:
    directories = _experiment_directories(spec.name)
    if spec.candidate_checkpoint is not None:
        directories = sorted(
            {*directories, spec.candidate_checkpoint.parent},
            key=Path.as_posix,
        )

    best = [
        path
        for directory in directories
        for path in directory.glob("best_*.pth")
        if is_usable_checkpoint(path)
    ]
    if best:
        path = _highest_epoch(best)
        return CheckpointChoice(
            path.resolve(), "best", checkpoint_epoch(path)
        )

    candidate = spec.candidate_checkpoint
    if candidate is not None and is_usable_checkpoint(candidate):
        return CheckpointChoice(
            candidate.resolve(), "candidate", checkpoint_epoch(candidate)
        )

    latest = [
        path
        for directory in directories
        for path in directory.glob("epoch_*.pth")
        if LATEST_EPOCH_PATTERN.fullmatch(path.name)
        and is_usable_checkpoint(path)
    ]
    if latest:
        path = _highest_epoch(latest)
        return CheckpointChoice(
            path.resolve(), "latest_epoch", checkpoint_epoch(path)
        )
    return None
