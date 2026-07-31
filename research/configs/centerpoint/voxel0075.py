"""CenterPoint Voxel 0.075 for KITTI Car."""

_base_ = ["./voxel01.py"]

point_cloud_range = [
    0.0,
    -38.4,
    -3.0,
    67.2,
    38.4,
    1.0,
]

voxel_size = [
    0.075,
    0.075,
    0.1,
]

grid_size = [
    896,
    1024,
    40,
]

sparse_shape = [
    41,
    1024,
    896,
]

model = dict(
    data_preprocessor=dict(
        voxel_layer=dict(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=10,
            max_voxels=(120000, 160000),
        ),
    ),

    pts_middle_encoder=dict(
        sparse_shape=sparse_shape,
    ),

    pts_bbox_head=dict(
        bbox_coder=dict(
            pc_range=point_cloud_range[:2],
            post_center_range=point_cloud_range,
            voxel_size=voxel_size[:2],
        ),
    ),

    train_cfg=dict(
        pts=dict(
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
        ),
    ),

    test_cfg=dict(
        pts=dict(
            post_center_limit_range=point_cloud_range,
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
        ),
    ),
)

work_dir = "research/experiments/voxel0075"