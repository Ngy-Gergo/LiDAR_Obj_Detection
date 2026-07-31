"""CenterPoint Voxel 0.1 with DCN for KITTI Car."""

_base_ = ["./voxel01.py"]

model = dict(
    pts_bbox_head=dict(
        separate_head=dict(
            _delete_=True,
            type="DCNSeparateHead",
            dcn_config=dict(
                type="DCN",
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                groups=4,
            ),
            init_bias=-2.19,
            final_kernel=3,
        ),
    ),
)

work_dir = "research/experiments/voxel01_dcn"