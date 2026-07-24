"""CenterPoint Pillar 0.2 for KITTI Car-only model selection."""

_base_ = ["./_base_/kitti_car_common.py"]

point_cloud_range = [
    0.0,
    -38.4,
    -3.0,
    67.2,
    38.4,
    1.0,
]

voxel_size = [
    0.2,
    0.2,
    4.0,
]

grid_size = [
    336,
    384,
    1,
]

model = dict(
    type="CenterPoint",

    data_preprocessor=dict(
        type="Det3DDataPreprocessor",
        voxel=True,
        voxel_layer=dict(
            max_num_points=20,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(30000, 40000),
        ),
    ),

    pts_voxel_encoder=dict(
        type="PillarFeatureNet",
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(
            type="BN1d",
            eps=1e-3,
            momentum=0.01,
        ),
        legacy=False,
    ),

    pts_middle_encoder=dict(
        type="PointPillarsScatter",
        in_channels=64,
        output_shape=(384, 336),
    ),

    pts_backbone=dict(
        type="SECOND",
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(
            type="BN",
            eps=1e-3,
            momentum=0.01,
        ),
        conv_cfg=dict(
            type="Conv2d",
            bias=False,
        ),
    ),

    pts_neck=dict(
        type="SECONDFPN",
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[0.5, 1, 2],
        norm_cfg=dict(
            type="BN",
            eps=1e-3,
            momentum=0.01,
        ),
        upsample_cfg=dict(
            type="deconv",
            bias=False,
        ),
        use_conv_for_no_stride=True,
    ),

    pts_bbox_head=dict(
        type="KittiCenterHead",
        in_channels=384,

        tasks=[
            dict(
                num_class=1,
                class_names=["Car"],
            ),
        ],

        common_heads=dict(
            reg=(2, 2),
            height=(1, 2),
            dim=(3, 2),
            rot=(2, 2),
        ),

        share_conv_channel=64,

        bbox_coder=dict(
            type="CenterPointBBoxCoder",
            pc_range=point_cloud_range[:2],
            post_center_range=point_cloud_range,
            max_num=100,
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            code_size=7,
        ),

        separate_head=dict(
            type="SeparateHead",
            init_bias=-2.19,
            final_kernel=3,
        ),

        loss_cls=dict(
            type="mmdet.GaussianFocalLoss",
            reduction="mean",
        ),

        loss_bbox=dict(
            type="mmdet.L1Loss",
            reduction="mean",
            loss_weight=0.25,
        ),

        norm_bbox=True,
    ),

    train_cfg=dict(
        pts=dict(
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=100,
            min_radius=2,
            code_weights=[1.0] * 8,
        ),
    ),

    test_cfg=dict(
        pts=dict(
            post_center_limit_range=point_cloud_range,
            max_per_img=100,
            max_pool_nms=False,
            min_radius=[2],
            score_threshold=0.1,
            pc_range=point_cloud_range[:2],
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            nms_type="rotate",
            pre_max_size=1000,
            post_max_size=100,
            nms_thr=0.2,
        ),
    ),
)

work_dir = "research/experiments/pillar02"