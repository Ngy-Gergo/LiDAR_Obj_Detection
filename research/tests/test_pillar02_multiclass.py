from __future__ import annotations

from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPOSITORY_ROOT / "research/configs/centerpoint/pillar02_multiclass.py"
DATA_ROOT = Path(
    "/home/ws-rtx/Documents/Projects/lidar-centerpoint/data/KITTI_Obj_Detect"
)
CANONICAL_CLASSES = ("Car", "Pedestrian", "Cyclist")


def test_multiclass_config_preserves_pillar_geometry_and_recipe() -> None:
    from mmengine.config import Config

    from lidar_model_selection.training import _validate_model_schema

    config = Config.fromfile(CONFIG)
    assert tuple(config.class_names) == CANONICAL_CLASSES
    task = config.model.pts_bbox_head.tasks[0]
    assert tuple(task.class_names) == CANONICAL_CLASSES
    assert task.num_class == 3
    assert tuple(config.voxel_size) == (0.2, 0.2, 4.0)
    assert tuple(config.point_cloud_range) == (0.0, -38.4, -3.0, 67.2, 38.4, 1.0)
    assert config.train_dataloader.batch_size == 2
    assert config.optim_wrapper.accumulative_counts == 4
    assert config.train_cfg.max_epochs == 60
    assert config.train_cfg.val_interval == 5
    sampler = config.train_pipeline[2]
    assert sampler.type == "ObjectSample"
    assert sampler.use_ground_plane is False
    assert sampler.db_sampler.sample_groups == dict(
        Car=15, Pedestrian=15, Cyclist=15
    )
    _validate_model_schema(config)


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="read-only KITTI root unavailable")
def test_multiclass_pipeline_retains_each_canonical_class() -> None:
    from mmengine.config import Config
    from mmdet3d.registry import DATASETS
    from mmdet3d.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    config = Config.fromfile(CONFIG)
    dataset = DATASETS.build(config.train_dataloader.dataset)
    seen: set[int] = set()
    for index in range(len(dataset)):
        sample = dataset.prepare_data(index)
        assert sample is not None
        labels = sample["data_samples"].gt_instances_3d.labels_3d.tolist()
        seen.update(int(label) for label in labels)
        if seen == {0, 1, 2}:
            break
    assert seen == {0, 1, 2}


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="read-only KITTI root unavailable")
def test_multiclass_model_cpu_loss_accepts_all_labels() -> None:
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmdet3d.registry import DATASETS, MODELS
    from mmdet3d.utils import register_all_modules

    import lidar_model_selection.compat.center_head_7d  # noqa: F401
    import lidar_model_selection.compat.kitti_db_sampler  # noqa: F401

    register_all_modules(init_default_scope=True)
    config = Config.fromfile(CONFIG)
    dataset = DATASETS.build(config.train_dataloader.dataset)
    model = MODELS.build(config.model)
    samples = [dataset.prepare_data(0), dataset.prepare_data(1)]
    batch = model.data_preprocessor(pseudo_collate(samples), training=True)
    losses = model(**batch, mode="loss")
    total_loss = sum(losses.values())
    total_loss.backward()
    labels = {
        int(label)
        for sample in samples
        for label in sample["data_samples"].gt_instances_3d.labels_3d.tolist()
    }
    assert set(losses) == {"task0.loss_bbox", "task0.loss_heatmap"}
    assert labels == {0, 1, 2}
