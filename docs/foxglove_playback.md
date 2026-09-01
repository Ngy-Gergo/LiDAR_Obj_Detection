# ROS2 and Foxglove playback

The stage-1 adapter subscribes to the original bag-played cloud and runs one
protected finalist per process. It publishes only standard ROS2 messages;
Foxglove Bridge observes the same ROS graph as the detector nodes. Do not open
the MCAP directly in Foxglove for this workflow.

The acceptance session is the July 27 recording with working synchronized
camera images:

```text
/media/ws-rtx/datastore1/2026-07-27_kaposvar/elso_kor_lexus3_2026-07-27_14-15
```

Do not use the July 24 `korforgalom_9` camera for visual acceptance: its JPEG
payloads render solid green.

## Host prerequisite

ROS2 Humble, rosbag2 MCAP support, Foxglove Bridge, and `vision_msgs` must be
installed. Source ROS in every terminal. A dependency check is:

```bash
cd /home/ws-rtx/Documents/Projects/lidar-centerpoint
source /opt/ros/humble/setup.bash
/home/ws-rtx/anaconda3/envs/lidar_centerpoint_g/bin/python -c "import rclpy, rosbag2_py, sensor_msgs, visualization_msgs, diagnostic_msgs, vision_msgs, tf2_ros; print('ROS2 adapter dependencies ready')"
```

If `vision_msgs` is absent, install the Humble package on the host:

```bash
sudo apt update
sudo apt install ros-humble-vision-msgs
```

## Required static-TF replay policy

The recording contains 44 `/tf_static` messages, while its recorded offered
QoS is transient-local `KEEP_LAST(depth=1)`. An unmodified rosbag publisher can
therefore retain only one sample for a detector that joins late, not the
complete static frame graph. Always play this session with the repository-owned
override:

```text
/home/ws-rtx/Documents/Projects/lidar-centerpoint/research/configs/playback/rosbag2_qos.yaml
```

It changes only the replay publisher for `/tf_static` to reliable,
transient-local `KEEP_LAST(depth=100)`. This bounded history holds the complete
44-message graph without accumulating duplicate samples forever under
`--loop`. It does not create, replace, or hardcode any transform. The
detector's static-TF subscription uses the same bounded history. Consequently
both supported start orders are deterministic:

- detector ready before bag playback: it receives the static messages live;
- detector started after playback begins: DDS delivers the retained complete
  static history from the configured rosbag publisher.

Omitting the override is unsupported for acceptance and can reproduce the
late-join failure.

## Start the graph

Terminal 1 — start Foxglove Bridge on the local interface:

```bash
cd /home/ws-rtx/Documents/Projects/lidar-centerpoint
source /opt/ros/humble/setup.bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml address:=127.0.0.1 port:=8765
```

Terminal 2 — start the voxel finalist on GPU 0. The bounded `all` policy is the
initial inspection mode; its DDS subscription and application queue both use
the explicit capacity 32. Queue overflow is a counted rejection, never a
silent policy change.

```bash
cd /home/ws-rtx/Documents/Projects/lidar-centerpoint
source /opt/ros/humble/setup.bash
/home/ws-rtx/anaconda3/envs/lidar_centerpoint_g/bin/python -m lidar_model_selection.playback.ros2_node \
  --model voxel0075 \
  --runs-root /home/ws-rtx/Documents/Projects/lidar-centerpoint/research/runs \
  --device cuda:0 \
  --input-topic /lexus3/os_center/points \
  --output-prefix /centerpoint/voxel0075 \
  --base-frame lexus3/base_link \
  --feature-profile kaposvar_center_reflectivity_v1 \
  --score-threshold 0.1 \
  --processing-policy all \
  --queue-capacity 32 \
  --tf-timeout-seconds 0.2 \
  --diagnostics-period-seconds 1.0 \
  --publish-model-cloud \
  --ros-args -p use_sim_time:=true
```

Terminal 3 — optionally start the pillar finalist on GPU 1. It need not publish
a duplicate visualization cloud because both finalists use the same protected
input range; the voxel node's cloud can be shared by both Foxglove panels.

```bash
cd /home/ws-rtx/Documents/Projects/lidar-centerpoint
source /opt/ros/humble/setup.bash
/home/ws-rtx/anaconda3/envs/lidar_centerpoint_g/bin/python -m lidar_model_selection.playback.ros2_node \
  --model pillar02 \
  --runs-root /home/ws-rtx/Documents/Projects/lidar-centerpoint/research/runs \
  --device cuda:1 \
  --input-topic /lexus3/os_center/points \
  --output-prefix /centerpoint/pillar02 \
  --base-frame lexus3/base_link \
  --feature-profile kaposvar_center_reflectivity_v1 \
  --score-threshold 0.1 \
  --processing-policy all \
  --queue-capacity 32 \
  --tf-timeout-seconds 0.2 \
  --diagnostics-period-seconds 1.0 \
  --no-publish-model-cloud \
  --ros-args -p use_sim_time:=true
```

Terminal 4 — play the full synchronized LiDAR/TF/camera set at 0.5x:

```bash
cd /home/ws-rtx/Documents/Projects/lidar-centerpoint
source /opt/ros/humble/setup.bash
ros2 bag play /media/ws-rtx/datastore1/2026-07-27_kaposvar/elso_kor_lexus3_2026-07-27_14-15 \
  --storage mcap \
  --rate 0.5 \
  --loop \
  --clock 100 \
  --qos-profile-overrides-path /home/ws-rtx/Documents/Projects/lidar-centerpoint/research/configs/playback/rosbag2_qos.yaml \
  --topics \
    /lexus3/os_center/points \
    /tf \
    /tf_static \
    /lexus3/camera/zed/zed_node/left/color/rect/image/compressed \
    /lexus3/camera/zed/zed_node/left/color/rect/image/camera_info
```

This order exercises detector-before-bag behavior. To exercise supported late
joining, start Terminal 4 first with this same override and then start either
detector; no bag restart or timing guess is required.

## Foxglove layout

Connect to `ws://localhost:8765` and use `lexus3/base_link` as the fixed/display
frame.

For a side-by-side comparison, create two 3D panels. Add the same
`/centerpoint/voxel0075/model_points` point cloud to both. Enable only
`/centerpoint/voxel0075/markers` in the left panel and only
`/centerpoint/pillar02/markers` in the right panel. Add an Image panel using:

```text
/lexus3/camera/zed/zed_node/left/color/rect/image/compressed
```

The corresponding camera calibration topic is replayed as
`/lexus3/camera/zed/zed_node/left/color/rect/image/camera_info`. Useful
diagnostic topics are `/centerpoint/voxel0075/diagnostics` and
`/centerpoint/pillar02/diagnostics`.

## Later 1.0x freshness acceptance

Stop both detector processes and rosbag, then repeat the commands with
`--rate 1.0`. For freshness inspection, change each detector to:

```text
--processing-policy latest --queue-capacity 1
```

`latest` uses one DDS sample and one pending application slot. Every pending
replacement remains counted as an application drop. This is a freshness
policy, not a zero-loss or real-time performance claim. Retain the static-TF
QoS override at every playback rate.

## Topics, frames, reset, and diagnostics

| Topic | Type | Frame | QoS |
|---|---|---|---|
| `/lexus3/os_center/points` | `sensor_msgs/msg/PointCloud2` | `lexus3/os_center` | best effort, volatile, keep last `queue_capacity` for `all`, 1 for `latest` |
| `/centerpoint/<model>/detections` | `vision_msgs/msg/Detection3DArray` | `lexus3/base_link` | reliable, volatile, keep last 1 |
| `/centerpoint/<model>/markers` | `visualization_msgs/msg/MarkerArray` | `lexus3/base_link` | reliable, transient local, keep last 1 |
| `/centerpoint/<model>/diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | `lexus3/base_link` | reliable, volatile, keep last 10 |
| `/centerpoint/<model>/model_points` | `sensor_msgs/msg/PointCloud2` | `lexus3/base_link` | best effort, volatile, keep last 1 |

The exact acquisition stamp propagates to detections, markers, and the optional
model cloud. Missing, stale, mismatched, or invalid TF clears prior markers,
publishes no overlay, and never invokes inference.

A backward ROS-clock jump invalidates the current generation before the node
recreates its TF buffer and subscriptions. Pending work is cleared, old
in-flight output is suppressed, retained static history is delivered to the
new listener, marker state is cleared, and the new loop starts at frame index
zero. A point-timestamp fallback provides the same reset when `/clock` is not
available; the two paths deduplicate one logical rewind.

Diagnostics keep middleware-reported loss separate from queue rejection and
pending replacement. Stage timings are `queue_ms`, `tf_lookup_ms`,
`conversion_ms`, `inference_ms`, `publication_ms`, and `end_to_end_ms`. A TF
timeout has only TF and end-to-end timing; later stages remain unavailable.
Only successful model execution has inference timing.

Stop safely with `Ctrl-C` in detector terminals first, then rosbag, then the
bridge. The node disables rclpy's eager SIGINT context shutdown so it can stop
and join its worker, suppress stale output, and publish `DELETEALL` before
destroying the node and shutting down the ROS context. If another actor has
already invalidated the context, cleanup publication is skipped and logged.
