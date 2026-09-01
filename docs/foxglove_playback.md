# ROS2 and Foxglove playback

The stage-1 adapter subscribes to the original bag-played cloud and runs one
protected finalist per process. It publishes only standard ROS2 messages;
Foxglove Bridge observes the same ROS graph as the detector nodes. Do not open
the MCAP directly in Foxglove for this workflow.

## Host prerequisite

ROS2 Humble, rosbag2 MCAP support, and Foxglove Bridge are installed on the
host. `vision_msgs` was the one missing dependency during the implementation
audit. Install it on the host before acceptance (these commands were not run
by Codex):

```bash
sudo apt update
sudo apt install ros-humble-vision-msgs
```

Source ROS in every terminal. A quick dependency check is:

```bash
cd /home/ws-rtx/Documents/Projects/lidar-centerpoint
source /opt/ros/humble/setup.bash
/home/ws-rtx/anaconda3/envs/lidar_centerpoint_g/bin/python -c "import rclpy, rosbag2_py, sensor_msgs, visualization_msgs, diagnostic_msgs, vision_msgs, tf2_ros; print('ROS2 adapter dependencies ready')"
```

The commands below deliberately leave GPU execution to the host user.

## Initial 0.5x inspection

Terminal 1 — start Foxglove Bridge on the local interface:

```bash
cd /home/ws-rtx/Documents/Projects/lidar-centerpoint
source /opt/ros/humble/setup.bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml address:=127.0.0.1 port:=8765
```

Terminal 2 — loop the exact session at half speed and publish `/clock`:

```bash
cd /home/ws-rtx/Documents/Projects/lidar-centerpoint
source /opt/ros/humble/setup.bash
ros2 bag play /media/ws-rtx/datastore1/2026-07-24_kaposvar/korforgalom_9_lexus3_2026-07-24_10-58 \
  --storage mcap \
  --rate 0.5 \
  --loop \
  --clock 100 \
  --topics /lexus3/os_center/points /tf /tf_static
```

Terminal 3 — run only `voxel0075` on GPU 0. The bounded `all` policy is the
initial careful-inspection mode and reports queue overflow as a failure rather
than silently dropping a frame:

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

Terminal 4 (optional) — independently run only `pillar02` on GPU 1:

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
  --publish-model-cloud \
  --ros-args -p use_sim_time:=true
```

Connect Foxglove to `ws://localhost:8765`. In a 3D panel, select
`lexus3/base_link` as the fixed frame and add either the original
`/lexus3/os_center/points` cloud or the model-range cloud(s), plus the marker
topics. Add a diagnostics panel for each active model:

- `/centerpoint/voxel0075/model_points`
- `/centerpoint/voxel0075/detections`
- `/centerpoint/voxel0075/markers`
- `/centerpoint/voxel0075/diagnostics`
- `/centerpoint/pillar02/model_points` (when the second node is running)
- `/centerpoint/pillar02/detections` (when the second node is running)
- `/centerpoint/pillar02/markers` (when the second node is running)
- `/centerpoint/pillar02/diagnostics` (when the second node is running)

The original raw cloud remains owned by bag playback. The adapter never
republishes it.

## Later 1.0x freshness acceptance

After the 0.5x inspection, stop the detector nodes and bag process, then
restart the bag at real-time capture rate:

```bash
cd /home/ws-rtx/Documents/Projects/lidar-centerpoint
source /opt/ros/humble/setup.bash
ros2 bag play /media/ws-rtx/datastore1/2026-07-24_kaposvar/korforgalom_9_lexus3_2026-07-24_10-58 \
  --storage mcap \
  --rate 1.0 \
  --loop \
  --clock 100 \
  --topics /lexus3/os_center/points /tf /tf_static
```

Restart each desired detector with:

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
  --processing-policy latest \
  --queue-capacity 1 \
  --tf-timeout-seconds 0.2 \
  --diagnostics-period-seconds 1.0 \
  --publish-model-cloud \
  --ros-args -p use_sim_time:=true
```

For `pillar02`, change the model/prefix to `pillar02` and the device to
`cuda:1`. `latest` retains no more than one pending frame and reports every
overwritten pending frame. Dropping stale work is a freshness policy, not a
real-time performance claim.

To disable the inspection cloud for deployment, replace
`--publish-model-cloud` with `--no-publish-model-cloud`.

## Topics, frames, and QoS

| Topic | Type | Frame | QoS |
|---|---|---|---|
| `/lexus3/os_center/points` | `sensor_msgs/msg/PointCloud2` | `lexus3/os_center` | best effort, volatile, keep last 5 |
| `/centerpoint/<model>/detections` | `vision_msgs/msg/Detection3DArray` | `lexus3/base_link` | reliable, volatile, keep last 1 |
| `/centerpoint/<model>/markers` | `visualization_msgs/msg/MarkerArray` | `lexus3/base_link` | reliable, transient local, keep last 1 |
| `/centerpoint/<model>/diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | `lexus3/base_link` | reliable, volatile, keep last 10 |
| `/centerpoint/<model>/model_points` | `sensor_msgs/msg/PointCloud2` | `lexus3/base_link` | best effort, volatile, keep last 1 |

`tf2_ros` subscribes to `/tf` as reliable/volatile and `/tf_static` as
reliable/transient-local. The exact input acquisition timestamp is propagated
to detections, every marker, and the optional model cloud.

The canonical normalizer applies the recorded approximately 180-degree Z
rotation to model input and deliberately omits translation. For publication,
the validated sensor translation `(0.75, 0, 1.91)` is added exactly once to
bottom-centred detector boxes and the optional visualization cloud. No second
rotation is applied. Missing, stale, mismatched, or invalid TF causes a
diagnostic failure and no overlay.

Each frame publishes wireframe boxes and labels in a stable per-model color.
Disappeared IDs receive explicit deletes. A bag timestamp reset increments the
loop counter, clears pending work and all model-owned markers, resets the
sequence index, and suppresses any old in-flight result. Shutdown does the
same marker clear after stopping the worker and closing the detector.

Diagnostics include model alias, run ID, selected-checkpoint SHA-256, device,
threshold, policy/capacity, last input frame and acquisition timestamp,
received/processed/dropped/failed/rejected counts, loop count, queue depth and
generation, conversion/detector/publish/end-to-end timings, message and
subscription age, worker state, and the last structured error stage/code/text.

Stop safely with `Ctrl-C` in the detector terminals first, then the bag-play
terminal, then Foxglove Bridge. This gives each node an opportunity to publish
its final marker clear and join its worker.
