# Architecture

The repository is a monorepo with independent research and deployment
packages.

```text
research configuration and training
              |
              v
      frozen model artifact
              |
              v
production vehicle runtime
```

## Research

`research/` is an installable Python project named `lidar-model-selection`.
It owns MMDetection3D configuration, version-pinned compatibility adapters,
evaluation, benchmarking, and recorded-data playback.

The existing ROS 2 playback node remains here because it reads recorded KITTI
files and reconstructs a model from a training configuration and checkpoint.
It is an experiment tool, not the vehicle runtime.

## Artifact boundary

`artifacts/` is the only model hand-off between research and runtime. Binary
artifacts are not committed. Runtime metadata defines the backend, input
features, spatial range, classes, box convention, score threshold, and model
filename.

## Runtime

`runtime/` is an independent ROS 2 `ament_python` package named
`lidar_detection_runtime`. It must not import the research package,
MMDetection3D training configuration, KITTI evaluation code, or recorded-file
playback.

The runtime currently contains packaging only. Production modules should be
added after export and numerical-parity validation, with separate
responsibilities for the node, detector backend, latest-frame mailbox,
point-cloud conversion, result conversion, and diagnostics.

## Repository-relative paths

The current MMDetection3D configuration uses `data/...` paths relative to the
repository root. Research commands using that configuration must therefore be
started from the repository root until a dedicated path configuration is
introduced.
