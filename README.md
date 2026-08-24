# LiDAR CenterPoint

This repository separates CenterPoint model research from the independent ROS 2
runtime. Run research commands from the repository root because dataset and
output paths in the MMEngine configurations are repository-relative.

## Repository layout

```text
.
├── research/
│   ├── configs/centerpoint/       Six CenterPoint configurations
│   ├── tools/train.py             Single-model and queued multi-GPU training
│   ├── tools/test.py              Single-model and sequential evaluation
│   ├── tools/benchmark.py         Sequential single-GPU latency benchmark
│   ├── tools/plot_results.py      Accuracy/runtime Matplotlib reports
│   ├── tools/smoke_test.py        One-batch checkpoint validation
│   ├── experiments/               Generated checkpoints and training logs
│   ├── evaluations/               Per-model and aggregate accuracy results
│   ├── benchmarks/                Per-model and aggregate latency results
│   ├── reports/figures/           Trackable comparison figures
│   ├── src/lidar_model_selection/ Research-only Python package
│   └── tests/                     Research package boundary tests
├── runtime/                       Independent ROS 2 runtime package
├── artifacts/                     Frozen-model hand-off to runtime
└── docs/architecture.md           Research/runtime boundary description
```

Training experiments and per-model evaluation and benchmark details are
generated locally. Aggregate `summary.csv` and `summary.json` files and report
figures may be retained for comparison. Checkpoints, datasets, and model
binaries are not source files.

## Setup

Activate the project environment and install the research package:

```bash
conda activate lidar_centerpoint_g
python -m pip install -e research
```

PyTorch and MMCV must match the machine's CUDA installation. The local KITTI
dataset is expected below `data/KITTI_Obj_Detect/`.

## Model set and data contract

The automated commands use exactly these six models:

```text
pillar02
pillar02_dcn
voxel01
voxel01_dcn
voxel0075
voxel0075_dcn
```

Their configurations are the matching files below
`research/configs/centerpoint/`. The comparison is KITTI Car-only, uses LiDAR
points `(x, y, z, intensity)`, and predicts seven-parameter boxes
`(x, y, z, dx, dy, dz, yaw)` with no velocity branch. The shared point-cloud
range is `[0.0, -38.4, -3.0, 67.2, 38.4, 1.0]`. Active training, validation,
testing, and benchmarking use batch size 1.

## Research flow

```text
configs ── train.py ──> research/experiments/
                           │              │
                     test.py        benchmark.py
                           │              │
                           v              v
                  research/evaluations/  research/benchmarks/
                           └──────┬───────┘
                           plot_results.py
                                  │
                                  v
                    research/reports/figures/
                                  │
                    JKK + deployment-hardware validation
                                  │
                                  v
                         artifacts/ ──> runtime/
```

Only the selected exported model and its metadata contract cross the
`artifacts/` boundary into runtime.

## Training

Train one configuration with its configured work directory and epoch count:

```bash
python research/tools/train.py CONFIG
```

The public single-model options are `--work-dir PATH`, `--max-epochs N`, and
`--resume`. For example:

```bash
python research/tools/train.py \
    research/configs/centerpoint/pillar02.py \
    --work-dir research/experiments/pillar02_screen10 \
    --max-epochs 20 \
    --resume
```

Train or inspect all six models with:

```bash
python research/tools/train.py --all
python research/tools/train.py --all --max-epochs 20
python research/tools/train.py --all --max-epochs 20 --resume
python research/tools/train.py --all --max-epochs 20 --force
python research/tools/train.py --all --max-epochs 20 --dry-run
```

`--all` defaults to 10 epochs when `--max-epochs` is omitted. It discovers
PyTorch-visible CUDA GPUs, starts one independent model process per GPU, queues
the remaining models, and reuses a GPU when its child exits. It does not use
DDP or `torchrun`. Each child receives one physical GPU through
`CUDA_VISIBLE_DEVICES` and writes
`research/experiments/<work-directory-name>.console.log`. An individual child
failure does not stop the other jobs; the command returns nonzero if any plan
or child fails. Dry-run prints the detected GPUs, complete plan, queue, and
initial assignments without starting training or writing outputs, but it still
requires at least one visible CUDA GPU. A started job replaces its previous
console log rather than appending to it. Epoch values must be positive.

Training supplies these conservative thread defaults:

```text
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
OPENBLAS_NUM_THREADS=2
NUMEXPR_NUM_THREADS=2
```

Existing user values are preserved. Scheduler children receive the defaults in
their copied environments, while direct single-model training receives them
before framework imports. The all-model controller and dry-run environment are
not changed.

### Work directories, completion, and resume

Fresh all-model training toward epoch `N` uses the deterministic directory
`research/experiments/<model>_screen<N>`. Its behavior is:

- missing directory: start fresh training;
- valid exact `epoch_<N>.pth`: skip as complete;
- existing directory without that exact checkpoint: fail and request
  `--resume` or `--force`;
- `--force`: delete and restart only that exact deterministic fresh directory.

A `best_*.pth` file records an inference candidate from a validation epoch; it
does not prove that training reached its target. Completion requires a usable
exact `epoch_<N>.pth`, which is also the skip criterion. After an executed run
exits successfully, an existing `last_checkpoint` marker is additionally
required to point to that exact file; the marker is optional. A best checkpoint
is not required by the training completion check.

Resume mode does not derive a new directory from the target epoch. It searches
existing real `<model>_screenN` directories, selects the highest usable exact
numeric `epoch_N.pth` below the target, and continues inside that checkpoint's
directory. For example:

```text
research/experiments/pillar02_screen10/epoch_10.pth
    -> target epoch 20 in research/experiments/pillar02_screen10/
```

It does not create `pillar02_screen20` for that continuation. No usable lower
checkpoint is an error, never an implicit restart from epoch 1. Resume sets the
selected checkpoint as `cfg.load_from` and enables `cfg.resume`; MMEngine then
restores model, optimizer, parameter-scheduler, epoch, and iteration state.
An existing valid exact target checkpoint is skipped. `--resume` and `--force`
are mutually exclusive.

## Checkpoint selection for inference

All-model evaluation and benchmarking share this priority:

1. highest usable `best_*.pth`;
2. configured candidate checkpoint;
3. highest usable, exactly named numeric `epoch_N.pth`;
4. no usable checkpoint: record a failure.

This inference priority is separate from numeric-only training resume. The
configured Pillar 0.2 candidate, when needed, is
`research/experiments/pillar02_full/epoch_10.pth`.
Evaluation prints warnings for candidate and latest-epoch fallbacks.

## Evaluation

Evaluate an explicit pair or all discoverable model checkpoints:

```bash
python research/tools/test.py CONFIG CHECKPOINT --gpu 0
python research/tools/test.py CONFIG CHECKPOINT --gpu 0 --dry-run
python research/tools/test.py --all --gpu 0
python research/tools/test.py --all --gpu 0 --dry-run
```

The nonnegative physical GPU index defaults to 0. The requested GPU is the
process's only visible CUDA device and is used as `cuda:0`. Models run
sequentially. Evaluation installs the KITTI evaluator compatibility, registers
MMDetection3D, processes inherited custom imports, and calls `Runner.test()`.
Per-model failures are recorded and later models continue; the final exit code
is nonzero when any model fails. Dry-run resolves and validates configs and
checkpoints without loading models or writing results, but still validates the
requested GPU.

Outputs are:

```text
research/evaluations/<model>/metrics.json
research/evaluations/summary.json
research/evaluations/summary.csv
```

The summaries contain the six strict Car 3D/BEV AP40 metrics and failure
details. Successful models rank by **Car 3D AP40 Moderate strict**, descending.
The output root is fixed; an explicit single-model evaluation also rewrites the
two aggregate summaries with its one result.

## Latency benchmark

Benchmark an explicit pair or all discoverable checkpoints:

```bash
python research/tools/benchmark.py CONFIG CHECKPOINT --gpu 0
python research/tools/benchmark.py --all --gpu 0
python research/tools/benchmark.py --all --gpu 0 \
    --warmup 100 --samples 1000
python research/tools/benchmark.py --all --gpu 0 --dry-run
```

Defaults are GPU 0, 100 warm-up batches, 1000 measured batches, and output
directory `research/benchmarks`. `--output-dir PATH` overrides the output
directory. Warm-up may be zero; measured samples must be positive.

The benchmark runs models sequentially in one process on the selected GPU. It
forces batch size 1, no dataloader workers, and unshuffled validation order. A
fresh iterator for each model consumes the first warm-up batches and then the
next measured batches deterministically. Warm-up is excluded from timing;
there is no cache-priming or frame pre-reading pass. CUDA timing boundaries are
synchronized.

Only two combined latency scopes are measured:

- `prediction_ms`: `model.test_step`, including data preprocessing,
  voxelization, forward inference, decoding, NMS, and postprocessing;
- `end_to_end_ms`: framework dataloader retrieval, CPU transforms, collation,
  and the complete prediction scope.

Each scope reports count, mean, minimum, maximum, p50, p95, p99, and population
standard deviation. The result also records peak CUDA allocated/reserved
memory and checkpoint size. The 20 Hz requirement is
`end_to_end_ms.p95_ms <= 50`. Missing the requirement is a performance result,
not an execution failure.

Regular per-model errors are recorded and later models continue. Runner/model
references are released and garbage collection plus CUDA cache cleanup runs
between attempted models. There is no controller/worker subprocess layer and
no per-model benchmark console log. Any recorded model failure produces a
nonzero final exit code, whereas a successful measurement above 50 ms does not.
Dry-run validates the requested GPU, configs, and checkpoints without loading
a Runner, running inference, or writing outputs.

Outputs are:

```text
research/benchmarks/<model>/latency.json
research/benchmarks/summary.json
research/benchmarks/summary.csv
```

These measurements characterize the selected workstation GPU only; they do
not prove latency on deployment hardware.

## Plot results

Generate the comparison report from the two aggregate CSV files:

```bash
python research/tools/plot_results.py
```

The default inputs are `research/evaluations/summary.csv` and
`research/benchmarks/summary.csv`; the default output directory is
`research/reports/figures/`. They can be overridden with
`--evaluation-summary`, `--benchmark-summary`, and `--output-dir`.

Seven figures are generated:

```text
accuracy_3d_ap40.png
accuracy_bev_ap40.png
latency_percentiles.png
accuracy_vs_latency.png
peak_gpu_memory.png
checkpoint_size.png
comparison_table.png
```

Recommendation ranking first requires end-to-end p95 at or below 50 ms, then
sorts eligible models by Car 3D AP40 Moderate strict descending. It does not
use an arbitrary combined score.

## One-batch smoke test

Validate one explicit config/checkpoint pair without a full evaluation:

```bash
python research/tools/smoke_test.py CONFIG CHECKPOINT --gpu 0
```

The smoke test validates both paths, initializes MMDetection3D and inherited
custom imports, loads the checkpoint through MMEngine, and processes exactly
one runner-provided test batch in inference mode. It requires one prediction
with aligned finite boxes, scores, and labels, and verifies that each box has
the seven KITTI parameters. It does not train, calculate gradients, compute
losses, run full KITTI evaluation, or benchmark latency, and it writes no
result artifacts. The nonnegative physical GPU index defaults to 0 and is
visible inside the process as `cuda:0`.

## Current stage

All six model variants have completed 20 epochs and have usable exact
`epoch_20.pth` checkpoints. Some continuations correctly remain in their
original `*_screen10` directories; the exact checkpoint, not the directory
suffix or a best checkpoint, proves completion.

The evaluation and benchmark tools have been cleaned and validated. Before
final selection:

1. run or confirm the final epoch-20 evaluation with the current evaluation
   path;
2. rerun the full benchmark with the current two-scope implementation—older
   benchmark summaries are not directly comparable;
3. rank accuracy first among models satisfying end-to-end p95 <= 50 ms;
4. complete recorded JKK qualitative testing;
5. validate latency and behavior on deployment hardware.

Runtime export and numerical-parity work follow final model selection. The ROS
2 package in `runtime/` must remain independent of the research package,
MMDetection3D, KITTI evaluation code, and recorded-data playback. See
[docs/architecture.md](docs/architecture.md),
[research/README.md](research/README.md), and
[runtime/README.md](runtime/README.md) for package-specific details.
