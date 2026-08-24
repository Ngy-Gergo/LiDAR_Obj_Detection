from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import lidar_model_selection.preflight as preflight


def _run(tmp_path: Path) -> SimpleNamespace:
    root = tmp_path / "runs" / "run"
    training = root / "training"
    training.mkdir(parents=True)
    config = root / "config.py"
    config.write_text("config\n", encoding="utf-8")
    return SimpleNamespace(
        run_id="20260824T120000Z-preflight-" + "a" * 24,
        paths=SimpleNamespace(root=root, training=training, config=config),
        manifest=SimpleNamespace(
            dataset=SimpleNamespace(
                class_names=("Car",),
                root_reference=str(tmp_path / "data" / "KITTI"),
            )
        ),
    )


def _config() -> dict[str, object]:
    train = {
        "data_root": "data/KITTI",
        "ann_file": "train.pkl",
        "metainfo": {"classes": ("Car",)},
    }
    test = {
        "data_root": "data/KITTI",
        "ann_file": "val.pkl",
        "metainfo": {"classes": ("Car",)},
    }
    return {
        "train_dataloader": {"dataset": train},
        "test_dataloader": {"dataset": test},
    }


def test_preflight_checks_config_files_one_sample_and_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _run(tmp_path)
    dataset_root = tmp_path / "data" / "KITTI"
    dataset_root.mkdir(parents=True)
    (dataset_root / "train.pkl").write_bytes(b"train")
    (dataset_root / "val.pkl").write_bytes(b"val")
    built = SimpleNamespace(__len__=lambda self: 1)

    class Dataset:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, int]:
            assert index == 0
            return {"sample": index}

    modules = {
        "mmengine.config": SimpleNamespace(
            Config=SimpleNamespace(fromfile=lambda path: _config())
        ),
        "mmdet3d.utils": SimpleNamespace(
            register_all_modules=lambda **kwargs: None
        ),
        "mmdet3d.registry": SimpleNamespace(
            DATASETS=SimpleNamespace(build=lambda config: Dataset())
        ),
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True)
        ),
    }
    monkeypatch.setattr(preflight, "_loaded", lambda supplied: run)
    monkeypatch.setattr(
        preflight.importlib, "import_module", lambda name: modules[name]
    )

    report = preflight.preflight_run(
        run, repository_root=tmp_path, operation="pipeline"
    )
    assert report.sample_checked is True
    assert report.dataset_root == str(dataset_root)
    assert {Path(path).name for path in report.annotation_paths} == {
        "train.pkl",
        "val.pkl",
    }


def test_preflight_rejects_missing_annotation_before_heavy_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _run(tmp_path)
    dataset_root = tmp_path / "data" / "KITTI"
    dataset_root.mkdir(parents=True)
    (dataset_root / "train.pkl").write_bytes(b"train")
    monkeypatch.setattr(preflight, "_loaded", lambda supplied: run)
    monkeypatch.setattr(
        preflight.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            Config=SimpleNamespace(fromfile=lambda path: _config())
        ),
    )
    with pytest.raises(FileNotFoundError):
        preflight.preflight_run(
            run, repository_root=tmp_path, operation="pipeline"
        )
