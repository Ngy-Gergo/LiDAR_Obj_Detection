# LiDAR model selection

The installable research package owns CenterPoint configuration, MMDetection3D
compatibility, checkpoint discovery, evaluation, benchmarking, and
recorded-data playback. It is not part of the production ROS 2 runtime.

Install it from the repository root after provisioning compatible CUDA,
PyTorch, MMCV, and MMDetection3D versions:

```bash
python -m pip install -e research
```

Run every command below from the repository root. The project compares exactly
`pillar02`, `pillar02_dcn`, `voxel01`, `voxel01_dcn`, `voxel0075`, and
`voxel0075_dcn` on KITTI Car with batch size 1.

## Training

Single-model mode uses the config's work directory and epoch count unless they
are overridden:

```bash
python research/tools/train.py CONFIG
python research/tools/train.py CONFIG \
    --work-dir PATH --max-epochs 20
python research/tools/train.py CONFIG \
    --work-dir PATH --max-epochs 20 --resume
```

All-model mode defaults to 10 epochs:

```bash
python research/tools/train.py --all
python research/tools/train.py --all --max-epochs 20
python research/tools/train.py --all --max-epochs 20 --resume
python research/tools/train.py --all --max-epochs 20 --force
python research/tools/train.py --all --max-epochs 20 --dry-run
```

One independent model child is assigned to each PyTorch-visible CUDA GPU. The
remaining models wait in a queue, and a GPU is reused after its current child
finishes. There is no DDP or `torchrun`. Every child has one physical GPU and a
separate `research/experiments/<work-directory-name>.console.log`. Failures do
not stop other children, and any failure makes the final exit code nonzero.
Dry-run prints the plan and initial assignments without starting training or
writing outputs, but still requires CUDA. Starting a job replaces its previous
console log rather than appending to it. Epoch values must be positive.

Fresh all-model target `N` uses
`research/experiments/<model>_screen<N>`. A usable exact `epoch_<N>.pth` means
complete and is skipped. An existing directory without that checkpoint fails
until `--resume` or `--force` is chosen. `--force` deletes and restarts only
the exact deterministic fresh directory.

Resume searches existing `<model>_screenN` directories for the highest usable
numeric epoch checkpoint below the target and continues in that same
directory. For example, `pillar02_screen10/epoch_10.pth` resumes toward epoch
20 inside `pillar02_screen10`; it does not create `pillar02_screen20`. MMEngine
resume restores model, optimizer, scheduler, epoch, and iteration state. No
valid lower checkpoint is an error. `--resume` and `--force` cannot be
combined.

A best checkpoint is an inference candidate, not completion proof. Completion
and skipping require exact `epoch_<target>.pth`. After an executed run exits,
an existing `last_checkpoint` marker must point to that exact file; the marker
is optional.

Training supplies `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`,
`OPENBLAS_NUM_THREADS=2`, and `NUMEXPR_NUM_THREADS=2` as defaults. Existing
user values are preserved. Scheduled children receive them in their copied
environment, and direct single-model training receives them before framework
imports.

## Evaluation

```bash
python research/tools/test.py CONFIG CHECKPOINT --gpu 0
python research/tools/test.py CONFIG CHECKPOINT --gpu 0 --dry-run
python research/tools/test.py --all --gpu 0
python research/tools/test.py --all --gpu 0 --dry-run
```

The nonnegative physical GPU index defaults to 0 and is visible as `cuda:0`;
models run sequentially. The evaluation path initializes KITTI compatibility,
MMDetection3D, and inherited custom imports, then calls `Runner.test()`.
Dry-run resolves configs and checkpoints without loading models or writing
outputs, but still validates the requested GPU. Failures are recorded per model
and later models continue.

Outputs are `research/evaluations/<model>/metrics.json`,
`research/evaluations/summary.json`, and
`research/evaluations/summary.csv`. The primary ranking metric is Car 3D AP40
Moderate strict, descending. There is no output override; an explicit
single-model evaluation rewrites the aggregate files with its one result.

Evaluation and all-model benchmarking share inference checkpoint discovery:

1. highest usable `best_*.pth`;
2. configured candidate checkpoint;
3. highest usable exact numeric `epoch_N.pth`;
4. no usable checkpoint: failure.

The configured Pillar 0.2 candidate is
`research/experiments/pillar02_full/epoch_10.pth`. This inference order is
separate from numeric-only training resume. Evaluation prints warnings for
candidate and latest-epoch fallbacks.

## Benchmark

```bash
python research/tools/benchmark.py CONFIG CHECKPOINT --gpu 0
python research/tools/benchmark.py --all --gpu 0
python research/tools/benchmark.py --all --gpu 0 \
    --warmup 100 --samples 1000
python research/tools/benchmark.py --all --gpu 0 --dry-run
```

Defaults are GPU 0, 100 warm-up batches, 1000 measured batches, and
`research/benchmarks` as the output directory. Use `--output-dir PATH` to
change it. Benchmarking is sequential in one process on one GPU, with batch
size 1, no workers, unshuffled validation order, and synchronized timing.
Warm-up batches are excluded; there is no cache-priming or pre-reading stage.

The two measured scopes are:

- `prediction_ms`: `model.test_step`, including preprocessing, voxelization,
  forward inference, decoding, NMS, and postprocessing;
- `end_to_end_ms`: dataloader retrieval, CPU transforms, collation, and the
  complete prediction scope.

The 20 Hz requirement is end-to-end p95 <= 50 ms. Errors are recorded per
model when caught, later models continue, and model references, garbage, and
the CUDA cache are cleaned between attempts. There is no subprocess isolation.
Any recorded failure produces a nonzero exit code; a successful measurement
above 50 ms does not. Dry-run validates the requested GPU, configs, and
checkpoints without loading a Runner, running inference, or writing outputs.
Workstation results do not prove deployment-hardware performance.

Outputs are `research/benchmarks/<model>/latency.json`,
`research/benchmarks/summary.json`, and `research/benchmarks/summary.csv`.

## Plots

```bash
python research/tools/plot_results.py
```

The required aggregate inputs are `research/evaluations/summary.csv` and
`research/benchmarks/summary.csv`. The default output directory is
`research/reports/figures/`; input and output paths may be overridden with
`--evaluation-summary`, `--benchmark-summary`, and `--output-dir`.

The command generates `accuracy_3d_ap40.png`, `accuracy_bev_ap40.png`,
`latency_percentiles.png`, `accuracy_vs_latency.png`,
`peak_gpu_memory.png`, `checkpoint_size.png`, and `comparison_table.png`.
Recommendation ranking first applies end-to-end p95 <= 50 ms eligibility,
then sorts by Car 3D AP40 Moderate strict descending.

## One-batch smoke test

```bash
python research/tools/smoke_test.py CONFIG CHECKPOINT --gpu 0
```

This validates the explicit paths, initializes MMDetection3D and inherited
custom imports, loads the checkpoint through MMEngine, and processes exactly
one runner-provided test batch. It validates one prediction with aligned,
finite boxes, scores, and labels and exactly seven box parameters. It does not
train, calculate gradients, run full KITTI evaluation, benchmark latency, or
write result artifacts. The nonnegative physical GPU index defaults to 0 and
is visible as `cuda:0`.

## Current stage

All six variants completed 20 epochs and have usable exact epoch-20
checkpoints. The evaluation and benchmark tools have been cleaned and
validated. Final epoch-20 evaluation must be run or confirmed, and the full
benchmark must be rerun with the current two-scope implementation; older
benchmark results are not directly comparable. Final selection ranks accuracy
first among models meeting the 50 ms p95 requirement, then requires recorded
JKK qualitative testing and actual deployment-hardware validation. Runtime
export follows final selection.
