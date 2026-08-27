from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import lidar_model_selection.preflight as preflight
from lidar_model_selection.provenance import identify_file_set
from lidar_model_selection.runs import build_dataset_identity


def _run(
    tmp_path: Path,
    *,
    scheme: str = "lidar-dataset-v2",
    recorded_root: Path | None = None,
    dataset: object | None = None,
) -> SimpleNamespace:
    root = tmp_path / "runs" / "run"
    training = root / "training"
    training.mkdir(parents=True)
    config = root / "config.py"
    config.write_text("config\n", encoding="utf-8")
    return SimpleNamespace(
        run_id="20260824T120000Z-preflight-" + "a" * 24,
        paths=SimpleNamespace(root=root, training=training, config=config),
        manifest=SimpleNamespace(
            dataset=(
                dataset
                if dataset is not None
                else SimpleNamespace(
                    scheme=scheme,
                    class_names=("Car",),
                    root_reference=str(
                        recorded_root or tmp_path / "data" / "KITTI"
                    ),
                )
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


def test_preflight_reloads_supplied_run_and_rejects_stale_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRun:
        def __init__(self, manifest: object) -> None:
            self.run_id = "20260824T120000Z-preflight-" + "a" * 24
            self.paths = SimpleNamespace(root=tmp_path / "run")
            self.manifest = manifest

    supplied = FakeRun(SimpleNamespace(revision=1))
    current = FakeRun(SimpleNamespace(revision=2))
    monkeypatch.setattr(preflight, "Run", FakeRun)
    monkeypatch.setattr(preflight, "load_run", lambda path: current)

    with pytest.raises(ValueError, match="stale"):
        preflight._loaded(supplied)


def test_preflight_checks_config_files_one_sample_and_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _run(tmp_path)
    dataset_root = tmp_path / "data" / "KITTI"
    dataset_root.mkdir(parents=True)
    (dataset_root / "train.pkl").write_bytes(b"train")
    (dataset_root / "val.pkl").write_bytes(b"val")
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


@pytest.mark.parametrize(
    "scheme, current_validation, expected_error",
    [
        ("lidar-dataset-v2", b"val", None),
        ("lidar-dataset-v2", b"changed", "identity differs"),
        ("lidar-dataset-v1", b"val", "root differs"),
    ],
)
def test_preflight_preserves_versioned_dataset_location_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scheme: str,
    current_validation: bytes,
    expected_error: str | None,
) -> None:
    current_root = tmp_path / "data" / "KITTI"
    current_root.mkdir(parents=True)
    (current_root / "train.pkl").write_bytes(b"train")
    (current_root / "val.pkl").write_bytes(current_validation)
    recorded_root = tmp_path / "old-machine" / "KITTI"
    recorded_root.mkdir(parents=True)
    (recorded_root / "train.pkl").write_bytes(b"train")
    (recorded_root / "val.pkl").write_bytes(b"val")
    dataset = build_dataset_identity(
        name="KITTI",
        version="object-v1",
        root_reference=str(recorded_root),
        semantic_partition="KITTI validation",
        framework_key="test_dataloader",
        annotation_files=identify_file_set(
            recorded_root,
            (Path("train.pkl"), Path("val.pkl")),
        ),
        class_names=("Car",),
        tasks={"3d_detection": ("Car",)},
        scheme=scheme,
    )
    run = _run(tmp_path, dataset=dataset)
    modules = {
        "mmengine.config": SimpleNamespace(
            Config=SimpleNamespace(fromfile=lambda path: _config())
        ),
        "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    }
    monkeypatch.setattr(preflight, "_loaded", lambda supplied: run)
    monkeypatch.setattr(
        preflight.importlib,
        "import_module",
        lambda name: modules[name],
    )

    if expected_error is None:
        report = preflight.preflight_run(
            run,
            repository_root=tmp_path,
            operation="pipeline",
            sample_check=False,
        )
        assert report.dataset_root == str(current_root)
    else:
        with pytest.raises(ValueError, match=expected_error):
            preflight.preflight_run(
                run,
                repository_root=tmp_path,
                operation="pipeline",
                sample_check=False,
            )
