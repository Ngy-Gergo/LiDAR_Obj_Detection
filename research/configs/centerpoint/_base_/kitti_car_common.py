"""Common KITTI Car-only settings for CenterPoint model selection."""

custom_imports = dict(
    imports=[
        "lidar_model_selection.compat.center_head_7d",
    ],
    allow_failed_imports=False,
)

default_scope = "mmdet3d"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

dataset_type = "KittiDataset"
data_root = "data/KITTI_Obj_Detect/"

class_names = ("Car",)
metainfo = dict(classes=class_names)

# The same range must be used by every model in the comparison.
#
# Grid dimensions:
#   0.20 m  The same range must be used by every model in -> 336 x 384
#   0.10 m  -> 672 x 768
#   0.075 m -> 896 x 1024
point_cloud_range = [
    0.0,
    -38.4,
    -3.0,
    67.2,
    38.4,
    1.0,
]

input_modality = dict(
    use_lidar=True,
    use_camera=False,
)

backend_args = None

train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=4,
        use_dim=4,
        backend_args=backend_args,
    ),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
    ),

    # Deliberately no ObjectSample yet.
    # This avoids the ground-plane and np.long compatibility problems.

    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.0, 0.0, 0.0],
    ),
    dict(
        type="RandomFlip3D",
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.0,
    ),
    dict(
        type="PointsRangeFilter",
        point_cloud_range=point_cloud_range,
    ),
    dict(
        type="ObjectRangeFilter",
        point_cloud_range=point_cloud_range,
    ),
    dict(
        type="ObjectNameFilter",
        classes=class_names,
    ),
    dict(type="PointShuffle"),
    dict(
        type="Pack3DDetInputs",
        keys=[
            "points",
            "gt_bboxes_3d",
            "gt_labels_3d",
        ],
    ),
]

test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=4,
        use_dim=4,
        backend_args=backend_args,
    ),
    dict(
        type="PointsRangeFilter",
        point_cloud_range=point_cloud_range,
    ),
    dict(
        type="Pack3DDetInputs",
        keys=["points"],
    ),
]

eval_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=4,
        use_dim=4,
        backend_args=backend_args,
    ),
    dict(
        type="Pack3DDetInputs",
        keys=["points"],
    ),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(
        type="DefaultSampler",
        shuffle=True,
    ),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="kitti_infos_train.pkl",
        data_prefix=dict(
            pts="training/velodyne_reduced",
        ),
        pipeline=train_pipeline,
        modality=input_modality,
        metainfo=metainfo,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(
        type="DefaultSampler",
        shuffle=False,
    ),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="kitti_infos_val.pkl",
        data_prefix=dict(
            pts="training/velodyne_reduced",
        ),
        pipeline=test_pipeline,
        modality=input_modality,
        metainfo=metainfo,
        box_type_3d="LiDAR",
        test_mode=True,
        backend_args=backend_args,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type="KittiMetric",
    ann_file=data_root + "kitti_infos_val.pkl",
    metric="bbox",
    backend_args=backend_args,
)

test_evaluator = val_evaluator

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

learning_rate = 1e-4
max_epochs = 20

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=learning_rate,
        weight_decay=0.01,
    ),
    clip_grad=dict(
        max_norm=35,
        norm_type=2,
    ),
)

param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        T_max=8,
        eta_min=learning_rate * 10,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=12,
        eta_min=learning_rate * 1e-4,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=8,
        eta_min=0.85 / 0.95,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=12,
        eta_min=1.0,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=max_epochs,
    val_interval=5,
)

val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

auto_scale_lr = dict(
    enable=False,
    base_batch_size=1,
)

# ---------------------------------------------------------------------------
# Runtime configuration used during research
# ---------------------------------------------------------------------------

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(
        type="LoggerHook",
        interval=50,
    ),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=3,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="Det3DVisualizationHook"),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(
        mp_start_method="fork",
        opencv_num_threads=0,
    ),
    dist_cfg=dict(backend="nccl"),
)

vis_backends = [
    dict(type="LocalVisBackend"),
]

visualizer = dict(
    type="Det3DLocalVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
)

log_processor = dict(
    type="LogProcessor",
    window_size=50,
    by_epoch=True,
)

log_level = "INFO"

randomness = dict(
    seed=20260724,
    deterministic=False,
)

load_from = None
resume = False