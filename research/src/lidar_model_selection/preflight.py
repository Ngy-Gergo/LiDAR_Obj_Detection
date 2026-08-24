"""Cheap readiness validation for one canonical run."""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .runs import Run, load_run, validate_run_id

__all__ = ("PreflightReport", "preflight_run")

Operation = Literal["train", "evaluate", "benchmark", "pipeline"]
_REPOSITORY_ROOT = Path(__file__).absolute().parents[3]


def _value(container: object, key: str, default: object = None) -> Any:
    if isinstance(container, Mapping):
        return container.get(key, default)
    return getattr(container, key, default)


def _loaded(run: Run | Path | str) -> Run:
    return run if isinstance(run, Run) else load_run(run)


def _regular_contained(root: Path, relative: str, *, description: str) -> Path:
    candidate = (root / relative).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{description} escapes dataset root") from error
    if not candidate.is_file():
        raise ValueError(f"{description} is not a regular file: {candidate}")
    return candidate


def _dataset_config(config: object, loader_key: str) -> object:
    loader = _value(config, loader_key)
    dataset = _value(loader, "dataset")
    while _value(dataset, "dataset") is not None:
        dataset = _value(dataset, "dataset")
    if dataset is None:
        raise ValueError(f"canonical config has no {loader_key}.dataset")
    return dataset


@dataclass(frozen=True, slots=True)
class PreflightReport:
    run_id: str
    operation: Operation
    config_path: str
    dataset_root: str
    annotation_paths: tuple[str, ...]
    sample_checked: bool

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        if self.operation not in {"train", "evaluate", "benchmark", "pipeline"}:
            raise ValueError("invalid preflight operation")
        for value, description in (
            (self.config_path, "config path"),
            (self.dataset_root, "dataset root"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"preflight {description} must be non-empty")
        if not isinstance(self.annotation_paths, tuple) or not self.annotation_paths:
            raise ValueError("preflight annotation paths must be a non-empty tuple")
        if not all(isinstance(path, str) and path for path in self.annotation_paths):
            raise ValueError("preflight annotation paths must be non-empty strings")
        if not isinstance(self.sample_checked, bool):
            raise TypeError("preflight sample_checked must be a boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "operation": self.operation,
            "config_path": self.config_path,
            "dataset_root": self.dataset_root,
            "annotation_paths": list(self.annotation_paths),
            "sample_checked": self.sample_checked,
        }


def preflight_run(
    run: Run | Path | str,
    *,
    operation: Operation = "pipeline",
    repository_root: Path = _REPOSITORY_ROOT,
    sample_check: bool = True,
) -> PreflightReport:
    """Validate cheap, material prerequisites without loading a full dataset."""
    if operation not in {"train", "evaluate", "benchmark", "pipeline"}:
        raise ValueError(f"unsupported preflight operation: {operation!r}")
    if not isinstance(repository_root, Path):
        raise TypeError("repository_root must be a pathlib.Path")
    loaded = _loaded(run)
    if not loaded.paths.root.is_dir() or not loaded.paths.training.is_dir():
        raise ValueError("run output paths are not ready")

    mmengine_config = importlib.import_module("mmengine.config")
    config_class = getattr(mmengine_config, "Config")
    config = config_class.fromfile(os.fspath(loaded.paths.config))
    loader_keys = (
        ("train_dataloader", "test_dataloader")
        if operation in {"train", "pipeline"}
        else ("test_dataloader",)
    )
    datasets = tuple(_dataset_config(config, key) for key in loader_keys)
    roots = []
    annotations = []
    repository = repository_root.resolve(strict=True)
    for key, dataset in zip(loader_keys, datasets):
        root_text = _value(dataset, "data_root")
        annotation_text = _value(dataset, "ann_file")
        if not isinstance(root_text, str) or not root_text.strip():
            raise ValueError(f"{key} dataset root is missing")
        if not isinstance(annotation_text, str) or not annotation_text.strip():
            raise ValueError(f"{key} annotation file is missing")
        root = Path(root_text)
        if not root.is_absolute():
            root = repository / root
        root = root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"{key} dataset root is not a directory: {root}")
        roots.append(root)
        annotations.append(
            _regular_contained(root, annotation_text, description=f"{key} annotation")
        )
        metainfo = _value(dataset, "metainfo", {})
        classes = _value(metainfo, "classes")
        if classes is not None and tuple(classes) != loaded.manifest.dataset.class_names:
            raise ValueError(f"{key} classes do not match run dataset identity")
    if len(set(roots)) != 1:
        raise ValueError("canonical loaders use different dataset roots")
    recorded_root = loaded.manifest.dataset.root_reference
    if not isinstance(recorded_root, str) or not recorded_root:
        raise ValueError("run has no observed dataset root evidence")
    recorded = Path(recorded_root)
    if not recorded.is_absolute():
        recorded = repository / recorded
    if recorded.resolve(strict=True) != roots[0]:
        raise ValueError("configured dataset root differs from run evidence")

    checked = False
    if sample_check:
        mmdet3d_utils = importlib.import_module("mmdet3d.utils")
        register = getattr(mmdet3d_utils, "register_all_modules")
        register(init_default_scope=True)
        registry = importlib.import_module("mmdet3d.registry")
        dataset = getattr(registry, "DATASETS").build(datasets[-1])
        if len(dataset) <= 0:
            raise ValueError("configured dataset is empty")
        dataset[0]
        checked = True

    if operation in {"train", "evaluate", "benchmark", "pipeline"}:
        torch = importlib.import_module("torch")
        cuda = getattr(torch, "cuda", None)
        if cuda is None or not bool(cuda.is_available()):
            raise RuntimeError(f"CUDA is unavailable for {operation}")

    return PreflightReport(
        run_id=loaded.run_id,
        operation=operation,
        config_path=os.fspath(loaded.paths.config),
        dataset_root=os.fspath(roots[0]),
        annotation_paths=tuple(os.fspath(path) for path in annotations),
        sample_checked=checked,
    )
