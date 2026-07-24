# LiDAR model selection

This installable Python package owns research-only concerns: MMDetection3D
compatibility, experiment configurations, evaluation, benchmarking, and
recorded-data playback.

Install it from the repository root after provisioning the pinned CUDA,
PyTorch, MMCV, and MMDetection3D environment:

```bash
python -m pip install -e research
```

The package metadata pins the versions used by this project, but it is not a
CUDA wheel lock. PyTorch and MMCV must be installed from indexes matching the
target CUDA toolkit.

Run the existing tools with:

```bash
lidar-offline-detect --help
lidar-kitti-eval --help
python -m lidar_model_selection.playback.ros2_node
```

Run MMDetection3D commands from the repository root because the current
configuration intentionally keeps dataset paths relative to that directory:

```text
research/configs/pointpillars/pointpillars_kitti_car.py
```

Future CenterPoint configurations that use the custom 7D head must opt in:

```python
custom_imports = dict(
    imports=["lidar_model_selection.compat.center_head_7d"],
    allow_failed_imports=False,
)
```

Nothing in this package is part of the production vehicle runtime.
