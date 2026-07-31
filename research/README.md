# LiDAR model selection

This installable Python package owns research-only concerns: MMDetection3D
compatibility, experiment configurations, evaluation, benchmarking, and
recorded-data playback.

Install it from the repository root after provisioning the pinned CUDA,
PyTorch, MMCV, and MMDetection3D environment:

```bash
python -m pip install -e research
```

The package metadata pins the versions used by this project, but it is not a
CUDA wheel lock. PyTorch and MMCV must be installed from indexes matching the
target CUDA toolkit.

Run research commands from the repository root because dataset paths are
repository-relative.

```bash
python research/tools/train.py --all --max-epochs 10
```

This trains all missing models. Experiment directories use deterministic
`<model>_screen<N>` names, where `N` is the requested epoch count, such as
`pillar02_screen10`. One complete model is assigned to each detected GPU, and
the remaining models wait in a queue.

Preview assignments without starting training:

```bash
python research/tools/train.py --all --max-epochs 10 --dry-run
```

Resume incomplete experiments explicitly:

```bash
python research/tools/train.py --all --max-epochs 10 --resume
```

Test all available models:

```bash
python research/tools/test.py --all --gpu 0
```

Testing is sequential on the requested single GPU. Best checkpoints are
preferred automatically; candidate or latest-epoch fallbacks emit warnings.
Accuracy results alone do not select the runtime model: single-GPU latency
benchmarking and JKK testing are still required.

Benchmark all available models:

```bash
python research/tools/benchmark.py --all --gpu 0
```

Use custom warm-up and measured sample counts:

```bash
python research/tools/benchmark.py \
    --all \
    --gpu 0 \
    --warmup 100 \
    --samples 1000
```

Benchmarking runs the models sequentially on one GPU. Do not run training or
testing at the same time. The primary runtime measurement is p95 end-to-end
latency, and 50 ms is the current 20 Hz requirement. The reported
`prediction_ms` stage includes network forward, box decoding, and
NMS/postprocessing because the current CenterPoint API does not expose a
reliable split between them. Each selected validation frame is read once
before warm-up so sequential models use the same primed file-cache condition.

Generate the accuracy and runtime comparison plots:

```bash
python research/tools/plot_results.py
```

The plots combine existing KITTI metrics from
`research/evaluations/summary.csv` with single-GPU results from
`research/benchmarks/summary.csv`.

Nothing in this package is part of the production vehicle runtime.
