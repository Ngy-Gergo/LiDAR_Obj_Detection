# Paired 20- versus 30-epoch finalist evidence

This report records the owner-managed post-training evaluation of the
`voxel0075` and `pillar02` 30-epoch finalists against their corresponding
20-epoch baselines. Each 30-epoch run was evaluated once and benchmarked once
with its manifest-recorded selected-best checkpoint. Both selected-best
filenames resolve to epoch 30; selection was not inferred from the target
duration.

The machine-readable comparison is
[`paired-end-to-end-p95.json`](paired-end-to-end-p95.json). It pins all four
evaluation and benchmark result IDs and validates cohort compatibility before
ranking.

## Canonical results

| Model | Epochs | 3D AP40 E / M / H | BEV AP40 E / M / H | Best epoch | Selected checkpoint SHA-256 |
| --- | ---: | --- | --- | ---: | --- |
| `voxel0075` | 20 | 78.4480 / 66.1696 / 62.1959 | 88.0176 / 82.0890 / 77.8536 | 20 | `5246b24bfe66a81df3bc6ca94db982f0188b33043f25771c40d02be4bcb22507` |
| `voxel0075` | 30 | 77.2638 / 65.8049 / 62.0019 | 87.5701 / 80.0926 / 77.7030 | 30 | `3e850ff7fc7e597971b028351a36257cd19b44a69770a5f6dda5222591e9c8c5` |
| `pillar02` | 20 | 68.9758 / 55.1909 / 49.7370 | 85.6702 / 75.6608 / 71.1751 | 20 | `7814db42c341be87c09ae4e68a0266288227aeac6a98cfb83420b4ffb5caaf8d` |
| `pillar02` | 30 | 73.2703 / 59.3263 / 53.6976 | 87.7711 / 77.0989 / 74.4106 | 30 | `2606a3448cd9edc97b662b0ea8631ea828ed1ba7fe64578bba1f2f5b650c8cac` |

| Model | Epochs | Prediction p50 / p95 | End-to-end p50 / p95 | Peak GPU allocated / reserved | 20 Hz | Terminal result |
| --- | ---: | --- | --- | --- | :---: | --- |
| `voxel0075` | 20 | 31.2469 / 33.6592 ms | 32.8995 / 35.3310 ms | 100.44 / 170.00 MiB | yes | training, evaluation, benchmark succeeded |
| `voxel0075` | 30 | 32.0940 / 39.6069 ms | 33.9328 / 42.0169 ms | 100.44 / 170.00 MiB | yes | training, evaluation, benchmark succeeded |
| `pillar02` | 20 | 12.5046 / 13.0762 ms | 14.2425 / 15.0032 ms | 147.00 / 428.00 MiB | yes | training, evaluation, benchmark succeeded |
| `pillar02` | 30 | 12.6915 / 13.2319 ms | 14.4154 / 15.0919 ms | 147.00 / 428.00 MiB | yes | training, evaluation, benchmark succeeded |

## Absolute 30-minus-20 deltas

| Model | 3D AP40 E / M / H | BEV AP40 E / M / H | Prediction p50 / p95 | End-to-end p50 / p95 | GPU allocated / reserved |
| --- | --- | --- | --- | --- | --- |
| `voxel0075` | -1.1842 / -0.3647 / -0.1940 | -0.4475 / -1.9964 / -0.1506 | +0.8471 / +5.9476 ms | +1.0333 / +6.6860 ms | 0.00 / 0.00 MiB |
| `pillar02` | +4.2945 / +4.1354 / +3.9606 | +2.1009 / +1.4381 / +3.2355 | +0.1870 / +0.1557 ms | +0.1729 / +0.0886 ms | 0.00 / 0.00 MiB |

The last validation in the final five-epoch interval improved moderate 3D AP40
from epoch 25 to epoch 30 for both new runs: `voxel0075` rose from 65.2178 to
65.8049 (+0.5871), and `pillar02` rose from 56.8705 to 59.3263 (+2.4558).
Epoch 30 therefore became the recorded best checkpoint within each 30-epoch
run. Across independent runs, however, the 30-epoch voxel checkpoint remained
0.3647 AP below the 20-epoch voxel baseline on the primary moderate metric.

## Result identity and status

| Run | Evaluation result | Benchmark result |
| --- | --- | --- |
| `20260827T092043Z-voxel0075-e583a40f435e3071e0cbd6fc` | `20260901T093742934083Z-evaluation-1da12da1dcea037b32fcd47a` | `20260901T104014393729Z-benchmark-8788ddfdc671d76144efcac4` |
| `20260901T195406Z-voxel0075-duration30-2ad23907052ef315ba8f8675` | `20260902T082628281593Z-evaluation-2fdaa37945bd2675c557dfef` | `20260902T083009602325Z-benchmark-f58a76e9462024acdb0df19e` |
| `20260827T092033Z-pillar02-3367910930525d0c12ddc346` | `20260901T093536573158Z-evaluation-e06d68a8b48bd9b854eaf052` | `20260901T103846232075Z-benchmark-3e3b0fb771aaacf47599a66d` |
| `20260901T195416Z-pillar02-duration30-2720f37cf422c4e55bafd0a6` | `20260902T082842231699Z-evaluation-f67bfeac549a966c8ff58b73` | `20260902T083106887597Z-benchmark-943a7b5978dd17570bd393c9` |

Both 30-epoch manifests have `training.status=completed`, one succeeded
training attempt, and no failure payload. Their final and selected checkpoints
are structurally valid and match their recorded SHA-256 values. Each new run
owns exactly one successful evaluation and exactly one successful benchmark;
no retry or failed result was published.

The strict comparison confirms matching dataset semantic identity and split,
KITTI Car task/metric profile, benchmark workload, hardware, driver, host,
precision, and core software identity. Its sole explicit waiver is the same
known warning as the 20-epoch campaign: the KITTI release label is unrecorded.
Exact `lidar-dataset-v2` identity and train/validation annotation hashes match.

Interpretation remains single-seed and single-benchmark-run evidence. The
30-epoch experiments also couple longer duration with a proportionally
stretched two-phase schedule, so the duration and schedule-duration effects
cannot be separated. The voxel p95 shift is much larger than its p50 shift
while memory is identical, making tail timing sensitive to one-shot host
jitter; the immutable canonical value is retained without a retry. Validation
has been used repeatedly and is not an untouched final test set.

## Presentation selection

Use the 20-epoch `voxel0075` selected-best checkpoint as the primary
presentation checkpoint:

```text
run_id: 20260827T092043Z-voxel0075-e583a40f435e3071e0cbd6fc
best_epoch: 20
sha256: 5246b24bfe66a81df3bc6ca94db982f0188b33043f25771c40d02be4bcb22507
```

It leads all four candidates in every requested 3D AP40 slice and also beats
the 30-epoch voxel checkpoint in every BEV AP40 slice and both canonical
latency percentiles. Use the 30-epoch `pillar02` selected-best checkpoint as
the low-latency fallback: it preserves approximately 15.09 ms end-to-end p95
while materially improving all six requested AP40 values over its baseline.

No additional training run was started. Canonical score/NMS settings remain
unchanged. The later integration acceptance retained the documented 0.10
presentation threshold and did not reinterpret it as an AP optimum; see
[`../../../docs/presentation_handoff.md`](../../../docs/presentation_handoff.md).

## Validation

- Strict schema-v3 report reload: 4 rows, 1 persisted waiver.
- Recomputed SHA-256 values match both 30-epoch selected checkpoints and
  run-owned configs.
- Result ownership audit: exactly 1 successful evaluation and 1 successful
  benchmark per 30-epoch run, with checkpoint bindings matching the manifests.
- Full CPU-only suite with `CUDA_VISIBLE_DEVICES=''`: 552 passed in 28.41 s.
- `git diff --check`: passed.
