# LiDAR CenterPoint

> **Architecture recovery checkpoint:** the detailed command guide below still
> documents the legacy GitHub workflow and must not be used for new work.
> The validated run-owned implementation and exact continuation state are in
> [research/PROGRESS.md](research/PROGRESS.md).

This repository separates CenterPoint research from the future vehicle
runtime. Run all research commands from the repository root because the KITTI
dataset paths in the MMDetection3D configs are repository-relative.

## Repository layout

```text
.
├── research/
│   ├── configs/centerpoint/       CenterPoint model configurations
│   ├── tools/train.py             Single-model and queued multi-GPU training
│   ├── tools/test.py              Single-model and sequential evaluation
│   ├── tools/benchmark.py         Sequential single-GPU latency benchmark
│   ├── tools/plot_results.py      Accuracy/runtime Matplotlib reports
│   ├── experiments/               Checkpoints, config snapshots, and logs
│   ├── evaluations/               Per-model and aggregate test results
│   ├── benchmarks/                Per-model and aggregate latency results
│   ├── reports/figures/           Trackable comparison plots
│   ├── src/lidar_model_selection/ Research-only compatibility and playback
│   ├── tests/                     Research package boundary tests
│   └── pyproject.toml             Pinned research package dependencies
├── runtime/                       Independent ROS 2 runtime package
├── artifacts/                     Frozen-model hand-off to runtime
└── docs/architecture.md           Research/runtime boundary description
```

Checkpoints, per-model evaluation details, datasets, and model binaries are
ignored by Git. Aggregate comparison summaries and report figures remain
trackable; none of these generated files are runtime source code.

## Setup

Activate the project environment and install the research package:

```bash
conda activate lidar_centerpoint_g
python -m pip install -e research
```

PyTorch and MMCV must be installed for the CUDA version used by the machine.
The local KITTI dataset is expected below `data/KITTI_Obj_Detect/`.

## CenterPoint models

The automated commands use exactly these six configurations:

```text
research/configs/centerpoint/pillar02.py
research/configs/centerpoint/pillar02_dcn.py
research/configs/centerpoint/voxel01.py
research/configs/centerpoint/voxel01_dcn.py
research/configs/centerpoint/voxel0075.py
research/configs/centerpoint/voxel0075_dcn.py
```

## Command flow

```text
CenterPoint configs
        |
        |  train.py
        v
research/experiments/<model>_screen<N>/
        |                              |
        |  test.py                     |  benchmark.py
        v                              v
research/evaluations/          research/benchmarks/
        |                              |
        +--------------+---------------+
                       |  plot_results.py
                       v
             research/reports/figures/
                       |
                       |  finalist training + JKK testing
                       |  + target-hardware testing + export
                       v
                   artifacts/
                       |
                       v
                    runtime/
```

Training checkpoints are research outputs. Only a selected, exported model
and its metadata contract cross the `artifacts/` boundary into runtime.

## Train all models

Preview GPU assignments without creating experiments or starting training:

```bash
python research/tools/train.py --all --max-epochs 10 --dry-run
```

Train all missing models:

```bash
python research/tools/train.py --all --max-epochs 10
```

The launcher detects CUDA devices with PyTorch, gives each child process one
physical GPU through `CUDA_VISIBLE_DEVICES`, and trains one complete model per
GPU. Remaining models wait in a queue and start when a GPU becomes free. DDP
is not used.

For ten epochs, each model uses:

```text
research/experiments/<model>_screen10/
research/experiments/<model>_screen10.console.log
```

Changing `--max-epochs` changes the deterministic suffix, for example
`voxel01_screen20`.

A model with a valid `best_*.pth` is skipped. A nonempty experiment without a
best checkpoint is not silently restarted.

Resume incomplete experiments explicitly:

```bash
python research/tools/train.py --all --max-epochs 10 --resume
```

Delete and restart the deterministic experiment directories explicitly:

```bash
python research/tools/train.py --all --max-epochs 10 --force
```

`--force` and `--resume` cannot be combined. Successful training must produce
a best checkpoint, the final `epoch_N.pth`, and a `last_checkpoint` pointer to
that final epoch.

## Train one model

The same single-model path used by the multi-GPU launcher remains available:

```bash
python research/tools/train.py \
    research/configs/centerpoint/pillar02.py \
    --work-dir research/experiments/pillar02_screen10 \
    --max-epochs 10
```

Add `--resume` only when continuing an incomplete experiment.

## Test all available models

Evaluate every model sequentially on one requested GPU:

```bash
python research/tools/test.py --all --gpu 0
```

Checkpoint discovery follows this order:

1. A valid `best_*.pth` in a deterministic `<model>_screen<N>` directory.
2. The explicitly recorded Pillar 0.2 candidate
   `research/experiments/pillar02_full/epoch_10.pth`, when present.
3. The highest numbered valid `epoch_N.pth`.

Candidate and latest-epoch fallbacks print warnings. Models without a usable
checkpoint are skipped. Evaluation failures are recorded and do not prevent
the remaining models from being tested.

Each model writes:

```text
research/evaluations/<model>/metrics.json
research/evaluations/<model>/console.log
```

The aggregate files are:

```text
research/evaluations/summary.json
research/evaluations/summary.csv
```

They contain the config and checkpoint paths, checkpoint-selection type,
trained epoch, six strict Car AP40 metrics, test status, and any error message.
Results are ranked by Car 3D AP40 moderate strict.

## Test one model

```bash
python research/tools/test.py \
    research/configs/centerpoint/pillar02.py \
    research/experiments/pillar02_full/epoch_10.pth \
    --work-dir research/evaluations/pillar02
```

## Benchmark and plot results

Benchmark every available model sequentially on one requested GPU:

```bash
python research/tools/benchmark.py --all --gpu 0
```

Preview checkpoint selection and output paths without loading a model:

```bash
python research/tools/benchmark.py --all --gpu 0 --dry-run
```

The benchmark records synchronized end-to-end latency, stage timings, peak
CUDA memory, and checkpoint size. Its primary screening requirement is
end-to-end p95 latency at or below 50 ms for 20 Hz operation.

After benchmarking, combine the tracked evaluation and benchmark summaries
into Matplotlib comparison figures:

```bash
python research/tools/plot_results.py
```

Detailed per-model benchmark output remains local. Aggregate CSV/JSON
summaries and PNG figures remain trackable for model comparison.

## Runtime selection

Accuracy ranking and the workstation benchmark do not select the production
model by themselves. Final selection still requires 20-epoch finalist
training, recorded JKK testing, and target-hardware testing, followed by
export and numerical-parity validation.

The ROS 2 package in `runtime/` must remain independent of the research
package, MMDetection3D, KITTI evaluation code, and recorded-data playback.
See [docs/architecture.md](docs/architecture.md),
[research/README.md](research/README.md), and
[runtime/README.md](runtime/README.md) for package-specific details.
