# Architecture

The repository is a monorepo with independent research and deployment
packages.

```text
config or catalog preset -> canonical Run -> train -> exact checkpoint
                                                |-> evaluation result
                                                |-> benchmark result
                                                        |
                                                   comparison -> plots

selected research Run -> future export/parity -> frozen artifact -> runtime
```

## Research

`research/` is an installable Python project named `lidar-model-selection`.
It owns MMDetection3D configuration, version-pinned compatibility adapters,
run/config/checkpoint evidence, training, evaluation, benchmarking, comparison,
plotting, and recorded-data playback. One canonical `Run` owns all native
experiment identity. Evaluation and benchmarking consume only its verified
selected checkpoint; they never search globally for artifacts.

`storage.py` owns generic durable persistence; `provenance.py` evidence;
`checkpoints.py` checkpoint identity; `runs.py` run lifecycle; `results.py`
immutable results; `training.py` MMEngine training; `evaluation.py` and
`benchmarking.py` one-run execution; `comparison.py` compatibility/ranking;
`plotting.py` rendering; `preflight.py` cheap readiness checks; and
`pipeline.py` one ordinary experiment orchestration. Command-line tools remain
thin callers of these package functions.

The Kaposvár MCAP research path is documented in
[`playback.md`](playback.md). It streams one session through immutable point
and detection contracts. A thin, one-model-per-process research adapter wraps
that same core for ROS2 bag playback and standard-message Foxglove inspection;
its operational contract is documented in
[`foxglove_playback.md`](foxglove_playback.md). It is an acceptance and
presentation tool, not the vehicle runtime.

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
