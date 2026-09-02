"""Closed finalist registry for real-recording playback."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from ..checkpoints import verify_checkpoint
from ..results import ResultBinding, binding_for_run
from ..runs import Run, load_run


FINALIST_RUNS = {
    "voxel0075": "20260827T092043Z-voxel0075-e583a40f435e3071e0cbd6fc",
    "pillar02": "20260901T195416Z-pillar02-duration30-2720f37cf422c4e55bafd0a6",
}

_FINALIST_CONFIGS = MappingProxyType(
    {
        "voxel0075": (
            "723749a5dc262ed1e57304092f12694d8f062c4a4158e2d65be685a47874c1b5"
        ),
        "pillar02": (
            "ebed7d29b96cae0812ede9e572ffb1ba054d650ad62cb1c6c8895697fcb3a5d9"
        ),
    }
)

_FINALIST_CHECKPOINTS = MappingProxyType(
    {
        "voxel0075": (
            41_175_026,
            "5246b24bfe66a81df3bc6ca94db982f0188b33043f25771c40d02be4bcb22507",
        ),
        "pillar02": (
            34_256_294,
            "2606a3448cd9edc97b662b0ea8631ea828ed1ba7fe64578bba1f2f5b650c8cac",
        ),
    }
)

# Both protected finalists use this exact inference range. The canonical
# MMDetection3D PointsRangeFilter applies strict inequalities at all six faces.
FINALIST_POINT_CLOUD_RANGE = (0.0, -38.4, -3.0, 67.2, 38.4, 1.0)


def finalist_range_mask(points: np.ndarray) -> np.ndarray:
    """Return the canonical strict MMDetection3D range-filter mask."""

    values = np.asarray(points)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError("points must have shape (N, M) with at least XYZ columns")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("points must contain numeric values")
    x_min, y_min, z_min, x_max, y_max, z_max = FINALIST_POINT_CLOUD_RANGE
    mask = np.asarray(
        (values[:, 0] > x_min)
        & (values[:, 1] > y_min)
        & (values[:, 2] > z_min)
        & (values[:, 0] < x_max)
        & (values[:, 1] < y_max)
        & (values[:, 2] < z_max),
        dtype=np.bool_,
    )
    mask.setflags(write=False)
    return mask


@dataclass(frozen=True, slots=True)
class FinalistSpec:
    """Static declared identity, safe to inspect without touching run files."""

    model_alias: str
    run_id: str
    config_sha256: str
    checkpoint_size_bytes: int
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class FinalistModelIdentity:
    """Fully verified canonical inputs for one registered finalist."""

    model_alias: str
    run: Run
    binding: ResultBinding
    config_path: Path
    checkpoint_path: Path
    checkpoint_reference: str
    checkpoint_size_bytes: int

    @property
    def run_id(self) -> str:
        return self.binding.run_id

    @property
    def config_sha256(self) -> str:
        return self.binding.config_sha256

    @property
    def checkpoint_sha256(self) -> str:
        return self.binding.checkpoint_sha256


def finalist_aliases() -> tuple[str, ...]:
    """Return the accepted aliases in deterministic display order."""

    return tuple(FINALIST_RUNS)


def finalist_spec(model_alias: str) -> FinalistSpec:
    """Return declared identity without loading or verifying any artifact."""

    if not isinstance(model_alias, str):
        raise TypeError("model_alias must be a string")
    if model_alias not in FINALIST_RUNS:
        accepted = ", ".join(finalist_aliases())
        raise ValueError(
            f"unknown finalist model alias {model_alias!r}; expected one of: {accepted}"
        )
    checkpoint_size, checkpoint_sha256 = _FINALIST_CHECKPOINTS[model_alias]
    return FinalistSpec(
        model_alias=model_alias,
        run_id=FINALIST_RUNS[model_alias],
        config_sha256=_FINALIST_CONFIGS[model_alias],
        checkpoint_size_bytes=checkpoint_size,
        checkpoint_sha256=checkpoint_sha256,
    )


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def resolve_finalist(
    model_alias: str,
    runs_root: Path | str,
) -> FinalistModelIdentity:
    """Resolve and verify one exact protected finalist below ``runs_root``.

    Resolution is intentionally closed: arbitrary run IDs, configs, and
    checkpoints are not accepted by the real-recording path.
    """

    spec = finalist_spec(model_alias)
    if not isinstance(runs_root, (Path, str)):
        raise TypeError("runs_root must be a pathlib.Path or string")

    root = _absolute(runs_root)
    run_id = spec.run_id
    loaded = load_run(root / run_id)
    if loaded.run_id != run_id:
        raise ValueError("resolved finalist run identity does not match the registry")
    if loaded.paths.root != root / run_id:
        raise ValueError("finalist run must be the exact immediate child of runs_root")
    if loaded.manifest.origin != "native":
        raise ValueError("registered finalist must be a native canonical run")
    if loaded.manifest.config.sha256 != spec.config_sha256:
        raise ValueError(
            "registered finalist config SHA-256 does not match the registry"
        )

    selected = loaded.selected_checkpoint
    if selected is None:
        raise ValueError("registered finalist does not have a selected checkpoint")
    expected_size = spec.checkpoint_size_bytes
    expected_sha256 = spec.checkpoint_sha256
    if selected.size_bytes != expected_size:
        raise ValueError(
            "registered finalist selected checkpoint size does not match the registry"
        )
    if selected.sha256 != expected_sha256:
        raise ValueError(
            "registered finalist selected checkpoint SHA-256 does not match the registry"
        )

    mismatches = verify_checkpoint(selected, root=loaded.paths.root)
    if mismatches:
        details = "; ".join(
            f"{mismatch.field}: expected {mismatch.expected!r}, "
            f"observed {mismatch.actual!r}"
            for mismatch in mismatches
        )
        raise ValueError(f"selected checkpoint identity mismatch: {details}")

    binding = binding_for_run(loaded)
    if binding.run_id != run_id:
        raise ValueError("finalist binding run identity does not match the registry")
    if binding.config_sha256 != spec.config_sha256:
        raise ValueError("finalist binding config does not match the registry")
    if binding.checkpoint_sha256 != expected_sha256:
        raise ValueError("finalist binding checkpoint does not match the registry")

    checkpoint_path = loaded.paths.root / selected.path
    return FinalistModelIdentity(
        model_alias=model_alias,
        run=loaded,
        binding=binding,
        config_path=loaded.paths.config,
        checkpoint_path=checkpoint_path,
        checkpoint_reference=selected.path,
        checkpoint_size_bytes=selected.size_bytes,
    )
