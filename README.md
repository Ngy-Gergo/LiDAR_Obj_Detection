# LiDAR CenterPoint

This repository separates reproducible CenterPoint research from the future
vehicle runtime. Research is run-owned: one immutable run identity binds its
canonical MMEngine config, dataset evidence, provenance, exact checkpoints,
and immutable evaluation and benchmark results.

## Layout

```text
research/configs/                 named CenterPoint presets
research/src/lidar_model_selection/
                                  run, training, result, and comparison logic
research/tools/                   thin research command-line interfaces
research/runs/                    generated local run records (ignored)
artifacts/                        future frozen deployment hand-off
runtime/                          independent ROS 2 runtime package
```

The runtime does not import research training, comparison, plotting, or
MMDetection3D machinery. Only a future validated frozen artifact crosses the
`artifacts/` boundary.

## Setup

Provision the project’s compatible CUDA, PyTorch, and OpenMMLab environment,
then install the research package:

```bash
python -m pip install -e research
```

Run commands from the repository root. The current configs refer to KITTI at
`data/KITTI_Obj_Detect/`.

## Ordinary workflow

Run one complete catalog preset:

```bash
python research/tools/run.py voxel0075 --max-epochs 20
```

Run an experimental config through the exact same run pipeline:

```bash
python research/tools/run.py \
  --config research/configs/my_ablation.py \
  --name my-ablation \
  --max-epochs 20
```

The pipeline creates the run, performs focused preflight checks, trains,
evaluates the run-owned selected checkpoint, benchmarks it, and writes an
immutable pipeline record pinning both result IDs.

Individual operations remain available for explicit runs:

```bash
python research/tools/train.py --model voxel0075 --max-epochs 20
python research/tools/train.py --run RUN_ID
python research/tools/evaluate.py --run RUN_ID
python research/tools/benchmark.py --run RUN_ID --warmup 100 --samples 1000
```

Use each command’s `--help` for exact options. Repeated evaluation or benchmark
invocations create new immutable results; ambiguous result selection must be
resolved explicitly.

## Comparison and plotting

Create a comparison from explicit run/result evidence:

```bash
python research/tools/compare.py \
  --run RUN_ID_A --run RUN_ID_B \
  --accuracy-metric car_3d_ap40_moderate_strict \
  --runtime-scope end_to_end_ms --runtime-statistic p95_ms \
  --output /tmp/comparison.json

python research/tools/plot.py /tmp/comparison.json --output-dir /tmp/figures
```

Compatibility metadata never silently matches when unknown. Any deliberate
exception is a field-specific, persisted waiver. Plotting consumes only the
already-resolved comparison rows; it performs no run or checkpoint discovery.

See [research/PROGRESS.md](research/PROGRESS.md) for validated milestone state
and [docs/architecture.md](docs/architecture.md) for responsibility boundaries.

## Presentation demo

The presentation path uses the protected 20-epoch `voxel0075` accuracy model,
with the accepted 30-epoch `pillar02` model as its low-latency fallback. The
single canonical tracked Foxglove launcher command, layout, reset behavior,
and bounded fallback-recording procedure are in
[docs/tracked_foxglove_demo.md](docs/tracked_foxglove_demo.md). Final model
identities and canonical evidence are summarized in
[docs/presentation_handoff.md](docs/presentation_handoff.md).
