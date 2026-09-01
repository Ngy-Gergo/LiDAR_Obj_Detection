# Real-recording playback

The research package has two explicit, non-interchangeable playback modes:

- ROS2 MCAP session playback for the Kaposvár center Ouster recording profile.
- Legacy packed float32 `x/y/z/feature` files selected with `--input-dir` and
  `--run`.

The offline MCAP path does not publish ROS messages, track objects, retain
temporal state, or create output files. The separate thin live ROS2 adapter is
documented in [`foxglove_playback.md`](foxglove_playback.md); it consumes the
same canonical normalization, finalist detector, and box geometry.

## Runtime prerequisites

Run from the repository root because the protected MMDetection3D run configs
still contain repository-relative dataset paths. Source ROS2 Humble before
using the exact project interpreter:

```bash
cd /home/ws-rtx/Documents/Projects/lidar-centerpoint
source /opt/ros/humble/setup.bash
/home/ws-rtx/anaconda3/envs/lidar_centerpoint_g/bin/python -c "import rosbag2_py, yaml; from rclpy.serialization import deserialize_message; from sensor_msgs.msg import PointCloud2; from tf2_msgs.msg import TFMessage; print('ROS2 MCAP/CDR support ready; PyYAML', yaml.__version__)"
```

ROS imports are lazy. `lidar-offline-detect --help`, the playback contracts,
and `--validate-only` model selection do not import Torch or MMDetection3D.
Validation does use the installed ROS2 reader and CDR message packages.

## MCAP contract

`--recording-root` must be absolute and `--session` must name one exact,
non-symlinked immediate child. The source requires the single MCAP declared by
that child's `metadata.yaml`, reads only `/tf_static` and
`/lexus3/os_center/points`, and preserves both PointCloud2 header time and MCAP
storage time.

The accepted PointCloud2 profile is little-endian, unorganized (`height=1`),
with a 48-byte point stride and exact declarations for `x`, `y`, `z`,
`intensity`, `t`, `reflectivity`, `ring`, `ambient`, and `range`. Padding is
never interpreted as packed float32 data. Schema changes, short payloads, and
duplicate or regressing header timestamps are errors.

The recorded `lexus3/base_link <- lexus3/os_center` static transform is
resolved and checked with 1e-6 translation/quaternion tolerances. Its
translation `(0.75, 0, 1.91)` is retained as session evidence but deliberately
not applied to detector points. Only its rotation is applied, producing the
LiDAR-origin `lidar` frame (approximately `[-x, -y, z]`). KITTI calibration is
not used.

The only MCAP feature profile is
`kaposvar_center_reflectivity_v1`: the fourth model feature is
`clip(reflectivity, 0, 255) / 255`. This is a fixed cross-sensor approximation
that requires visual acceptance; it is not claimed to reproduce KITTI
reflectance. Nonfinite required rows are counted and removed, while zero rows
and sparse nonempty clouds are retained.

## Streaming, errors, and timing

One session is streamed without retaining its frames or detection results.
State, pacing anchors, errors, and summary statistics reset at every fresh
session invocation. Default `--on-frame-error stop` aborts on the first error;
`continue` yields identity-rich error evidence only for failures explicitly
marked recoverable by the source. Frames before `--start-frame` are still read
for session-wide timestamp-order validation: the default policy reports those
errors, while explicit `continue` suppresses recoverable prefix outcomes so
they cannot consume the requested `--max-frames` window.

Empty source, empty-after-nonfinite, and empty-after-protected-range frames
return successful zero-detection statuses without calling model inference. A
nonempty sparse frame is sent to the model. The finalist backend validates
bottom-centred `(x, y, z, dx, dy, dz, yaw)` boxes, scores, and Car-only label
zero before publishing immutable CPU arrays.

Timing names are intentionally distinct from campaign benchmarks:

- `decode_ms`: MCAP read, CDR deserialization, schema checks, normalization,
  and immutable point-frame materialization.
- `detector_ms`: canonical array entry through synchronized inference,
  postprocessing, validation, and immutable CPU detection materialization.
- `frame_processing_ms`: `decode_ms + detector_ms`; deliberate pacing and BEV
  rendering are excluded.

Session timing aggregates cover every published frame outcome. A recoverable
source-error outcome retains its measured decode/processing duration and has a
zero detector duration, so the aggregate denominator remains the reported
frame count.

Block pacing follows the absolute capture-time schedule divided by
`--playback-rate`, never intentionally drops a frame, and reports lag against
that schedule. The optional Matplotlib BEV reuses one figure and decimates only
the displayed points, never model input.

## Decoder-only validation

This bounded command reads and normalizes the first three bring-up frames
without loading a run, checkpoint, Torch, MMDetection3D, CUDA, or a model:

```bash
cd /home/ws-rtx/Documents/Projects/lidar-centerpoint
source /opt/ros/humble/setup.bash
/home/ws-rtx/anaconda3/envs/lidar_centerpoint_g/bin/python -m lidar_model_selection.playback.cli \
  --recording-root /media/ws-rtx/datastore1/2026-07-27_kaposvar \
  --session proba_lexus3_2026-07-27_14-09 \
  --model voxel0075 \
  --runs-root /home/ws-rtx/Documents/Projects/lidar-centerpoint/research/runs \
  --feature-profile kaposvar_center_reflectivity_v1 \
  --start-frame 0 \
  --max-frames 3 \
  --validate-only
```

## Host acceptance commands

GPU preflight:

```bash
cd /home/ws-rtx/Documents/Projects/lidar-centerpoint
source /opt/ros/humble/setup.bash
/home/ws-rtx/anaconda3/envs/lidar_centerpoint_g/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); torch.zeros(1, device='cuda:0'); torch.cuda.synchronize()"
```

Ten-frame `proba` inference validation with `voxel0075`:

```bash
cd /home/ws-rtx/Documents/Projects/lidar-centerpoint
source /opt/ros/humble/setup.bash
/home/ws-rtx/anaconda3/envs/lidar_centerpoint_g/bin/python -m lidar_model_selection.playback.cli \
  --recording-root /media/ws-rtx/datastore1/2026-07-27_kaposvar \
  --session proba_lexus3_2026-07-27_14-09 \
  --model voxel0075 \
  --runs-root /home/ws-rtx/Documents/Projects/lidar-centerpoint/research/runs \
  --device cuda:0 \
  --feature-profile kaposvar_center_reflectivity_v1 \
  --score-threshold 0.3 \
  --start-frame 0 \
  --max-frames 10 \
  --playback-rate 1.0 \
  --on-frame-error stop
```

Full 129-frame `proba` run with interactive BEV:

```bash
cd /home/ws-rtx/Documents/Projects/lidar-centerpoint
source /opt/ros/humble/setup.bash
/home/ws-rtx/anaconda3/envs/lidar_centerpoint_g/bin/python -m lidar_model_selection.playback.cli \
  --recording-root /media/ws-rtx/datastore1/2026-07-27_kaposvar \
  --session proba_lexus3_2026-07-27_14-09 \
  --model voxel0075 \
  --runs-root /home/ws-rtx/Documents/Projects/lidar-centerpoint/research/runs \
  --device cuda:0 \
  --feature-profile kaposvar_center_reflectivity_v1 \
  --score-threshold 0.3 \
  --start-frame 0 \
  --playback-rate 1.0 \
  --on-frame-error stop \
  --visualize-bev
```

Bounded 100-frame `korforgalom_9` visual acceptance:

```bash
cd /home/ws-rtx/Documents/Projects/lidar-centerpoint
source /opt/ros/humble/setup.bash
/home/ws-rtx/anaconda3/envs/lidar_centerpoint_g/bin/python -m lidar_model_selection.playback.cli \
  --recording-root /media/ws-rtx/datastore1/2026-07-24_kaposvar \
  --session korforgalom_9_lexus3_2026-07-24_10-58 \
  --model voxel0075 \
  --runs-root /home/ws-rtx/Documents/Projects/lidar-centerpoint/research/runs \
  --device cuda:0 \
  --feature-profile kaposvar_center_reflectivity_v1 \
  --score-threshold 0.3 \
  --start-frame 0 \
  --max-frames 100 \
  --playback-rate 1.0 \
  --on-frame-error stop \
  --visualize-bev
```

Model execution and visual semantic acceptance must be performed from that
GPU-capable host terminal. The ROS2/Foxglove stage publishes frame-local
detections but does not establish tracking, motion compensation, TensorRT
parity, NuScenes performance, or equivalence between Ouster reflectivity and
KITTI reflectance.
