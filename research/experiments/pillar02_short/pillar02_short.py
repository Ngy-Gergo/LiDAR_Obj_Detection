auto_scale_lr = dict(base_batch_size=1, enable=False)
backend_args = None
class_names = ('Car', )
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'lidar_model_selection.compat.center_head_7d',
    ])
data_root = 'data/KITTI_Obj_Detect/'
dataset_type = 'KittiDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, interval=500, max_keep_ckpts=1, type='CheckpointHook'),
    logger=dict(interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
eval_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
grid_size = [
    336,
    384,
    1,
]
input_modality = dict(use_camera=False, use_lidar=True)
launcher = 'none'
learning_rate = 0.0001
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 20
metainfo = dict(classes=('Car', ))
model = dict(
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=20,
            max_voxels=(
                30000,
                40000,
            ),
            point_cloud_range=[
                0.0,
                -38.4,
                -3.0,
                67.2,
                38.4,
                1.0,
            ],
            voxel_size=[
                0.2,
                0.2,
                4.0,
            ])),
    pts_backbone=dict(
        conv_cfg=dict(bias=False, type='Conv2d'),
        in_channels=64,
        layer_nums=[
            3,
            5,
            5,
        ],
        layer_strides=[
            2,
            2,
            2,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.01, type='BN'),
        out_channels=[
            64,
            128,
            256,
        ],
        type='SECOND'),
    pts_bbox_head=dict(
        bbox_coder=dict(
            code_size=7,
            max_num=100,
            out_size_factor=4,
            pc_range=[
                0.0,
                -38.4,
            ],
            post_center_range=[
                0.0,
                -38.4,
                -3.0,
                67.2,
                38.4,
                1.0,
            ],
            score_threshold=0.1,
            type='CenterPointBBoxCoder',
            voxel_size=[
                0.2,
                0.2,
            ]),
        common_heads=dict(
            dim=(
                3,
                2,
            ), height=(
                1,
                2,
            ), reg=(
                2,
                2,
            ), rot=(
                2,
                2,
            )),
        in_channels=384,
        loss_bbox=dict(
            loss_weight=0.25, reduction='mean', type='mmdet.L1Loss'),
        loss_cls=dict(reduction='mean', type='mmdet.GaussianFocalLoss'),
        norm_bbox=True,
        separate_head=dict(
            final_kernel=3, init_bias=-2.19, type='SeparateHead'),
        share_conv_channel=64,
        tasks=[
            dict(class_names=[
                'Car',
            ], num_class=1),
        ],
        type='KittiCenterHead'),
    pts_middle_encoder=dict(
        in_channels=64, output_shape=(
            384,
            336,
        ), type='PointPillarsScatter'),
    pts_neck=dict(
        in_channels=[
            64,
            128,
            256,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.01, type='BN'),
        out_channels=[
            128,
            128,
            128,
        ],
        type='SECONDFPN',
        upsample_cfg=dict(bias=False, type='deconv'),
        upsample_strides=[
            0.5,
            1,
            2,
        ],
        use_conv_for_no_stride=True),
    pts_voxel_encoder=dict(
        feat_channels=[
            64,
        ],
        in_channels=4,
        legacy=False,
        norm_cfg=dict(eps=0.001, momentum=0.01, type='BN1d'),
        point_cloud_range=[
            0.0,
            -38.4,
            -3.0,
            67.2,
            38.4,
            1.0,
        ],
        type='PillarFeatureNet',
        voxel_size=[
            0.2,
            0.2,
            4.0,
        ],
        with_distance=False),
    test_cfg=dict(
        pts=dict(
            max_per_img=100,
            max_pool_nms=False,
            min_radius=[
                2,
            ],
            nms_thr=0.2,
            nms_type='rotate',
            out_size_factor=4,
            pc_range=[
                0.0,
                -38.4,
            ],
            post_center_limit_range=[
                0.0,
                -38.4,
                -3.0,
                67.2,
                38.4,
                1.0,
            ],
            post_max_size=100,
            pre_max_size=1000,
            score_threshold=0.1,
            voxel_size=[
                0.2,
                0.2,
            ])),
    train_cfg=dict(
        pts=dict(
            code_weights=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dense_reg=1,
            gaussian_overlap=0.1,
            grid_size=[
                336,
                384,
                1,
            ],
            max_objs=100,
            min_radius=2,
            out_size_factor=4,
            point_cloud_range=[
                0.0,
                -38.4,
                -3.0,
                67.2,
                38.4,
                1.0,
            ],
            voxel_size=[
                0.2,
                0.2,
                4.0,
            ])),
    type='CenterPoint')
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.01),
    type='OptimWrapper')
param_scheduler = []
point_cloud_range = [
    0.0,
    -38.4,
    -3.0,
    67.2,
    38.4,
    1.0,
]
randomness = dict(deterministic=False, seed=20260724)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='kitti_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(pts='training/velodyne_reduced'),
        data_root='data/KITTI_Obj_Detect/',
        metainfo=dict(classes=('Car', )),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=4),
            dict(
                point_cloud_range=[
                    0.0,
                    -38.4,
                    -3.0,
                    67.2,
                    38.4,
                    1.0,
                ],
                type='PointsRangeFilter'),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='KittiDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/KITTI_Obj_Detect/kitti_infos_val.pkl',
    backend_args=None,
    metric='bbox',
    type='KittiMetric')
test_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    dict(
        point_cloud_range=[
            0.0,
            -38.4,
            -3.0,
            67.2,
            38.4,
            1.0,
        ],
        type='PointsRangeFilter'),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
train_cfg = dict(max_iters=500, type='IterBasedTrainLoop', val_interval=500)
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='kitti_infos_train.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(pts='training/velodyne_reduced'),
        data_root='data/KITTI_Obj_Detect/',
        filter_empty_gt=True,
        metainfo=dict(classes=('Car', )),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=4),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                rot_range=[
                    -0.78539816,
                    0.78539816,
                ],
                scale_ratio_range=[
                    0.95,
                    1.05,
                ],
                translation_std=[
                    0.0,
                    0.0,
                    0.0,
                ],
                type='GlobalRotScaleTrans'),
            dict(
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.0,
                sync_2d=False,
                type='RandomFlip3D'),
            dict(
                point_cloud_range=[
                    0.0,
                    -38.4,
                    -3.0,
                    67.2,
                    38.4,
                    1.0,
                ],
                type='PointsRangeFilter'),
            dict(
                point_cloud_range=[
                    0.0,
                    -38.4,
                    -3.0,
                    67.2,
                    38.4,
                    1.0,
                ],
                type='ObjectRangeFilter'),
            dict(classes=('Car', ), type='ObjectNameFilter'),
            dict(type='PointShuffle'),
            dict(
                keys=[
                    'points',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=False,
        type='KittiDataset'),
    num_workers=1,
    persistent_workers=False,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        rot_range=[
            -0.78539816,
            0.78539816,
        ],
        scale_ratio_range=[
            0.95,
            1.05,
        ],
        translation_std=[
            0.0,
            0.0,
            0.0,
        ],
        type='GlobalRotScaleTrans'),
    dict(
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.0,
        sync_2d=False,
        type='RandomFlip3D'),
    dict(
        point_cloud_range=[
            0.0,
            -38.4,
            -3.0,
            67.2,
            38.4,
            1.0,
        ],
        type='PointsRangeFilter'),
    dict(
        point_cloud_range=[
            0.0,
            -38.4,
            -3.0,
            67.2,
            38.4,
            1.0,
        ],
        type='ObjectRangeFilter'),
    dict(classes=('Car', ), type='ObjectNameFilter'),
    dict(type='PointShuffle'),
    dict(
        keys=[
            'points',
            'gt_bboxes_3d',
            'gt_labels_3d',
        ],
        type='Pack3DDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='kitti_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(pts='training/velodyne_reduced'),
        data_root='data/KITTI_Obj_Detect/',
        metainfo=dict(classes=('Car', )),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=4),
            dict(
                point_cloud_range=[
                    0.0,
                    -38.4,
                    -3.0,
                    67.2,
                    38.4,
                    1.0,
                ],
                type='PointsRangeFilter'),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='KittiDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/KITTI_Obj_Detect/kitti_infos_val.pkl',
    backend_args=None,
    metric='bbox',
    type='KittiMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
voxel_size = [
    0.2,
    0.2,
    4.0,
]
work_dir = 'research/experiments/pillar02_short'
