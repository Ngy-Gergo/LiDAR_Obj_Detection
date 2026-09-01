# Tracked Foxglove presentation

This presentation path adds a deterministic, lightweight tracker after the
existing run-bound detector. It does not change model input, inference,
thresholding, raw detections, checkpoints, or training evidence.

## One-command demonstration

Use the synchronized July 27 session with the working left camera. Source ROS2
and invoke the repository launcher from the repository that contains the code
you intend to present:

```bash
cd /tmp/lidar-centerpoint-presentation-final-711e4f
source /opt/ros/humble/setup.bash
PYTHONPATH=research/src:runtime \
  /home/ws-rtx/anaconda3/envs/lidar_centerpoint_g/bin/python \
  research/tools/foxglove_demo.py \
  --model voxel0075 \
  --device cuda:0 \
  --bag /media/ws-rtx/datastore1/2026-07-27_kaposvar/elso_kor_lexus3_2026-07-27_14-15 \
  --runs-root /home/ws-rtx/Documents/Projects/lidar-centerpoint/research/runs \
  --checkpoint-sha256 5246b24bfe66a81df3bc6ca94db982f0188b33043f25771c40d02be4bcb22507 \
  --rate 0.5 \
  --loop \
  --enable-tracking
```

The launcher validates the bag, runs root, QoS file, Python interpreter, and
ROS2 command before starting anything. It starts Foxglove Bridge, then the
single selected detector, waits for its diagnostics topic, and finally starts
QoS-correct bag playback. It prints:

```text
Foxglove: ws://localhost:8765
Fixed frame: lexus3/base_link
```

`Ctrl-C` stops the detector first, then bag playback, then the bridge. Each
child runs in its own process group and escalates from `SIGINT` to `SIGTERM`
and finally `SIGKILL` only if a process does not stop within its bounded
grace period.

`--model` and `--runs-root` are forwarded to the existing closed, run-bound
detector registry. `--checkpoint-sha256` optionally pins that registry
selection and is rejected before model construction if it does not match; it
does not bypass run or checkpoint verification with an arbitrary path. The
current registry selects the immutable 20-epoch voxel finalist. A later
accepted final run can be selected in the existing registry without changing
launcher or tracker logic.

Inspect the exact shell-escaped child commands without requiring ROS, a bag,
or a GPU by adding `--dry-run` (alias `--print-command`). If a bridge or bag is
already running, retain the same command but add one or both of:

```text
--no-bridge
--no-bag
```

For a later 1.0x freshness check, use:

```text
--rate 1.0 --processing-policy latest --queue-capacity 1
```

`latest` remains an explicit freshness policy; its replacements are counted
and are not presented as zero-drop real-time evidence.

## Tracker policy

The tracker runs only with `--enable-tracking`. It uses actual acquisition
timestamp differences for constant-velocity prediction, then performs a
class-compatible distance-gated deterministic greedy assignment. SciPy is not
a declared project dependency, so no new assignment dependency is introduced.

Default lifecycle controls are:

```text
--track-min-hits 2
--track-max-missed 3
--track-max-gap-seconds 0.75
--track-association-distance 4.0
--track-trail-length 20
```

Tracks begin tentative and become visible after the configured hit count.
Unmatched confirmed tracks coast only within both missed-frame and elapsed-time
limits. Coasting boxes are predictions, rendered with lower opacity and the
word `coasting`; they are not shown as fresh observations. Trails are strictly
bounded. Track IDs restart at 1 after every reset.

The existing coordinator generation is the primary ROS loop/reset signal. A
backward `/clock` jump or deduplicated point-timestamp fallback clears pending
work, suppresses stale in-flight results, resets TF, resets the tracker once,
and publishes tracked-marker `DELETEALL`. Invalid TF, malformed input, or a
processing/publication failure clears track state instead of fabricating an
overlay. A valid empty detector frame advances coasting/expiration normally.

## Topics and Foxglove layout

Connect Foxglove to `ws://localhost:8765` and set the fixed frame to
`lexus3/base_link`.

Use these panels/topics:

| Panel | Topic |
|---|---|
| 3D point cloud | `/centerpoint/voxel0075/model_points` |
| Raw detector boxes | `/centerpoint/voxel0075/markers` |
| Tracked boxes, IDs, velocity, trails | `/centerpoint/voxel0075/tracked_markers` |
| Camera | `/lexus3/camera/zed/zed_node/left/color/rect/image/compressed` |
| Detector diagnostics | `/centerpoint/voxel0075/diagnostics` |
| Tracking diagnostics | `/centerpoint/voxel0075/tracking_diagnostics` |

For a clear presentation, use separate raw and tracked 3D panels with the same
model-range point cloud. The raw panel shows thresholded frame-local detector
output. The tracked panel shows stable `Car #ID score speed state` labels,
velocity arrows, and short trails. Toggle raw markers off when emphasizing
temporal stability.

Tracking publishes only standard Humble messages:

| Topic | Type | QoS |
|---|---|---|
| `/centerpoint/<model>/tracked_detections` | `vision_msgs/msg/Detection3DArray` | reliable, volatile, keep last 1 |
| `/centerpoint/<model>/tracked_markers` | `visualization_msgs/msg/MarkerArray` | reliable, transient local, keep last 1 |
| `/centerpoint/<model>/tracking_diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | reliable, volatile, keep last 10 |

The standard `Detection3D.id` contains the stable integer ID. Markers use
model-owned stable namespaces and IDs. Disappeared tracks receive explicit
deletes; reset and shutdown publish `DELETEALL` while the ROS context remains
valid.

Tracking diagnostics include active, confirmed, tentative and coasting counts;
per-frame and total creations/removals, matches and misses; reset count/reason;
total tracked and failed tracking frames; tracking-update latency; last
timestamp, actual `dt`, and maximum observed gap; coordinator generation; model
alias; run ID; and checkpoint SHA-256. Tracker latency is kept separate from
detector inference latency. A tracker exception clears only derived tracking
state and markers; already-valid raw detections and markers remain published.

Run the repeatable CPU-only synthetic benchmark with:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=research/src \
  python research/tools/benchmark_tracker.py \
  --iterations 1000 --warmup 100 --detections 100
```

The tool reports p50, p95, and maximum update latency. Its 2 ms target is
reported as evidence, not enforced as a machine-dependent CI assertion.

## Record a short fallback MCAP

Do this later while the live tracked demo is healthy. First preflight free
space and confirm that the explicit new output directory does not exist:

```bash
df -h /media/ws-rtx/datastore1
test ! -e /media/ws-rtx/datastore1/centerpoint_presentation_demos/voxel0075_tracked_20260902
mkdir -p /media/ws-rtx/datastore1/centerpoint_presentation_demos
```

Then record a bounded 30-second MCAP:

```bash
source /opt/ros/humble/setup.bash
timeout --signal=INT --kill-after=10s 30s \
  ros2 bag record \
  --storage mcap \
  --output /media/ws-rtx/datastore1/centerpoint_presentation_demos/voxel0075_tracked_20260902 \
  --qos-profile-overrides-path /home/ws-rtx/Documents/Projects/lidar-centerpoint/research/configs/playback/rosbag2_qos.yaml \
  /clock \
  /tf \
  /tf_static \
  /lexus3/os_center/points \
  /lexus3/camera/zed/zed_node/left/color/rect/image/compressed \
  /lexus3/camera/zed/zed_node/left/color/rect/image/camera_info \
  /centerpoint/voxel0075/model_points \
  /centerpoint/voxel0075/detections \
  /centerpoint/voxel0075/markers \
  /centerpoint/voxel0075/diagnostics \
  /centerpoint/voxel0075/tracked_detections \
  /centerpoint/voxel0075/tracked_markers \
  /centerpoint/voxel0075/tracking_diagnostics
```

`timeout` sends `SIGINT` at 30 seconds so rosbag2 closes the MCAP cleanly. You
can also stop earlier with `Ctrl-C`. Confirm `metadata.yaml` exists before
using the recording, then validate finalization and topic coverage:

```bash
ros2 bag info /media/ws-rtx/datastore1/centerpoint_presentation_demos/voxel0075_tracked_20260902
```

Replay the fallback without a detector or GPU by starting Foxglove Bridge
first and then running:

```bash
source /opt/ros/humble/setup.bash
ros2 bag play \
  /media/ws-rtx/datastore1/centerpoint_presentation_demos/voxel0075_tracked_20260902 \
  --storage mcap \
  --rate 0.5 \
  --loop \
  --clock 100 \
  --qos-profile-overrides-path \
  /home/ws-rtx/Documents/Projects/lidar-centerpoint/research/configs/playback/rosbag2_qos.yaml
```

The fallback is presentation evidence only. It is not labeled tracking ground
truth and cannot support AMOTA, identity-switch, or deployment-performance
claims.

## Acceptance checklist and limitations

Run the one-command graph first at `--rate 0.5 --processing-policy all
--queue-capacity 32`, then at `--rate 1.0 --processing-policy latest
--queue-capacity 1`. At both rates verify the point cloud, normal camera image,
raw boxes, stable tracked IDs, bounded trails, useful velocity arrows, brief
coasting, loop-reset `DELETEALL`, fresh TF, increasing processed counts,
explained drops/failures, and traceback-free `Ctrl-C` with no owned processes
left behind. Visual review is presentation acceptance, not quantitative
tracking validation.

This is post-detection temporal stabilization and identity continuity. It does
not fuse the camera, alter detector AP, add learned temporal memory, or make
validation-set measurements equivalent to untouched test performance. The
protected finalist registry continues to select and hash-check the checkpoint;
`--runs-root` selects its immutable artifact root and `--checkpoint-sha256`
pins the selected identity. Register a newly accepted finalist through the
existing evidence-reviewed registry rather than bypassing checkpoint
verification with an arbitrary path.

Post-presentation work: evaluate multi-sweep input and learned BEV memory;
extract a smaller deployment package; perform the broader runtime/research
cleanup; investigate TensorRT and distillation; and expand datasets and sealed
evaluation coverage. None is implemented in this deadline branch.
