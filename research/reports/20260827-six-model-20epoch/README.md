# Six-model KITTI 20-epoch campaign

This directory records the resolved evidence from one fresh 20-epoch training
run for each of the six KITTI Car-only CenterPoint configurations. Every run
owns one successful immutable smoke result, one successful final KITTI
evaluation result, and one successful synchronized benchmark result.

The resolved reports are:

- [end-to-end p95 comparison](fresh-end-to-end-p95.json), with figures under
  [`figures/end-to-end-p95/`](figures/end-to-end-p95/);
- [prediction p95 comparison](fresh-prediction-p95.json), with figures under
  [`figures/prediction-p95/`](figures/prediction-p95/).

## Protected runs

- `20260827T092033Z-pillar02-3367910930525d0c12ddc346`
- `20260827T092042Z-pillar02-dcn-17f8f3e630e66376d794960d`
- `20260827T092043Z-voxel0075-e583a40f435e3071e0cbd6fc`
- `20260827T092044Z-voxel0075-dcn-04444a4b945b155c3942a099`
- `20260827T092045Z-voxel01-40cd6123fa5b4cdee59306ed`
- `20260827T092046Z-voxel01-dcn-bc6db6f99a45864e67165106`

## Measurements

| Model | 3D AP40 moderate | Prediction p95 | End-to-end p95 | Peak allocated | Peak reserved | Checkpoint | 20 Hz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `pillar02` | 55.1909 | 13.076175 ms | 15.003207 ms | 147.00 MiB | 428.00 MiB | 27.92 MiB | yes |
| `pillar02-dcn` | 54.4764 | 13.294470 ms | 15.022747 ms | 147.15 MiB | 428.00 MiB | 28.07 MiB | yes |
| `voxel01` | 63.4699 | 30.335071 ms | 32.126355 ms | 175.38 MiB | 338.00 MiB | 39.27 MiB | yes |
| `voxel01-dcn` | 63.2037 | 33.544498 ms | 35.432026 ms | 183.97 MiB | 230.00 MiB | 39.42 MiB | yes |
| `voxel0075` | 66.1696 | 33.659217 ms | 35.330952 ms | 100.44 MiB | 170.00 MiB | 39.27 MiB | yes |
| `voxel0075-dcn` | 66.0839 | 35.621944 ms | 37.666962 ms | 117.02 MiB | 186.00 MiB | 39.42 MiB | yes |

Benchmarks ran on an NVIDIA GeForce RTX 2080 Ti with NVIDIA driver
`575.57.08`. The canonical methodology used batch size 1, 100 warmup samples,
1,000 consecutive measured samples, and synchronized CUDA timing for both the
prediction-only and end-to-end scopes.

Both comparisons contain one explicit `accuracy.dataset.version` waiver:
the KITTI release label was not recorded. The comparison remains semantically
valid because all six runs have the same `lidar-dataset-v2` semantic identity
and matching train/validation annotation hashes.

## Selection

The measured accuracy/latency Pareto frontier is `pillar02`, `voxel01`, and
`voxel0075`:

- **Accuracy finalist:** `voxel0075`.
- **Speed finalist:** `pillar02`.
- **Middle Pareto reference:** `voxel01`.

`voxel01` remains a useful measured reference, but it is not a primary
deadline finalist: relative to `voxel0075`, it saves about 3.2 ms end-to-end
p95 while losing about 2.70 points of moderate 3D AP40. Each DCN variant is
dominated by its corresponding non-DCN model in this campaign, so none is
selected as a finalist.

## Limits

All six models pass the preliminary 20 Hz requirement on this workstation.
This does not establish performance on low-end deployment hardware. The
campaign contains one run per model and therefore does not provide multi-seed
statistical significance. It also does not include an official pretrained
checkpoint baseline.
