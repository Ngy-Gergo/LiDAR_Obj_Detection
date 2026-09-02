"""The fixed CenterPoint model-selection source-config catalog."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Mapping


__all__ = (
    "CENTERPOINT_CONFIGS",
    "catalog_slugs",
    "source_config_for_slug",
)

_RESEARCH_ROOT = Path(__file__).absolute().parents[2]
_CONFIG_ROOT = _RESEARCH_ROOT / "configs" / "centerpoint"

CENTERPOINT_CONFIGS: Mapping[str, Path] = MappingProxyType(
    {
        "pillar02": _CONFIG_ROOT / "pillar02.py",
        "pillar02-multiclass": _CONFIG_ROOT / "pillar02_multiclass.py",
        "pillar02-dcn": _CONFIG_ROOT / "pillar02_dcn.py",
        "voxel0075": _CONFIG_ROOT / "voxel0075.py",
        "voxel0075-dcn": _CONFIG_ROOT / "voxel0075_dcn.py",
        "voxel01": _CONFIG_ROOT / "voxel01.py",
        "voxel01-dcn": _CONFIG_ROOT / "voxel01_dcn.py",
    }
)


def catalog_slugs() -> tuple[str, ...]:
    """Return the six canonical slugs in deterministic order."""
    return tuple(sorted(CENTERPOINT_CONFIGS))


def source_config_for_slug(slug: str) -> Path:
    """Return the explicit source config for one canonical model slug."""
    if not isinstance(slug, str):
        raise TypeError("model slug must be a string")
    try:
        return CENTERPOINT_CONFIGS[slug]
    except KeyError as error:
        raise ValueError(f"unknown CenterPoint model slug: {slug!r}") from error
