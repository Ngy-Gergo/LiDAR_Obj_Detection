"""Canonical KITTI Car/Pedestrian/Cyclist CenterPoint settings."""

_base_ = ["./kitti_car_common.py"]

custom_imports = dict(
    imports=[
        "lidar_model_selection.compat.center_head_7d",
        "lidar_model_selection.compat.kitti_db_sampler",
    ],
    allow_failed_imports=False,
)

# The order is a contract shared by dataset remapping, the single CenterHead
# task, evaluation output, and later playback metadata.
class_names = ("Car", "Pedestrian", "Cyclist")
metainfo = dict(classes=class_names)

# A worktree does not contain a second dataset copy.  This is the existing,
# read-only KITTI preparation owned by the primary checkout.
data_root = "/home/ws-rtx/Documents/Projects/lidar-centerpoint/data/KITTI_Obj_Detect/"
backend_args = None
point_cloud_range = [0.0, -38.4, -3.0, 67.2, 38.4, 1.0]

# Values follow the official MMDetection3D v1.4.0 KITTI three-class
# PointPillars recipe.  Ground-plane files are absent from this prepared KITTI
# root, so ObjectSample intentionally does not request a plane.
db_sampler = dict(
    type="KittiDataBaseSampler",
    data_root=data_root,
    info_path=data_root + "kitti_dbinfos_train.pkl",
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5),
    ),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=15, Cyclist=15),
    points_loader=dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=4,
        use_dim=4,
        backend_args=backend_args,
    ),
    backend_args=backend_args,
)

train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=4,
        use_dim=4,
        backend_args=backend_args,
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    dict(type="ObjectSample", db_sampler=db_sampler, use_ground_plane=False),
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
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="PointShuffle"),
    dict(type="Pack3DDetInputs", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
]

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root=data_root,
        pipeline=train_pipeline,
        metainfo=metainfo,
    ),
)
val_dataloader = dict(dataset=dict(data_root=data_root, metainfo=metainfo))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + "kitti_infos_val.pkl")
test_evaluator = dict(ann_file=data_root + "kitti_infos_val.pkl")

learning_rate = 2.5e-4
max_epochs = 60
optim_wrapper = dict(
    optimizer=dict(lr=learning_rate),
    accumulative_counts=4,
)
param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        T_max=24,
        eta_min=learning_rate * 10,
        begin=0,
        end=24,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=36,
        eta_min=learning_rate * 1e-4,
        begin=24,
        end=60,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=24,
        eta_min=0.85 / 0.95,
        begin=0,
        end=24,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=36,
        eta_min=1.0,
        begin=24,
        end=60,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]
train_cfg = dict(max_epochs=max_epochs, val_interval=5)
auto_scale_lr = dict(enable=False, base_batch_size=8)

primary_metric = (
    "Kitti metric/pred_instances_3d/KITTI/"
    "Car_3D_AP40_moderate_strict"
)
