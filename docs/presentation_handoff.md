# Presentation release handoff

This document records the final evidence-backed model selection and release
acceptance state. The canonical one-command tracked demonstration, Foxglove
layout, lifecycle behavior, and fallback-recording procedure are documented in
[`tracked_foxglove_demo.md`](tracked_foxglove_demo.md).

## Protected presentation selection

| Role | Model | Epochs | Run ID | Config SHA-256 | Selected checkpoint SHA-256 | Score threshold |
| --- | --- | ---: | --- | --- | --- | ---: |
| Primary accuracy demo | voxel0075 | 20 | `20260827T092043Z-voxel0075-e583a40f435e3071e0cbd6fc` | `723749a5dc262ed1e57304092f12694d8f062c4a4158e2d65be685a47874c1b5` | `5246b24bfe66a81df3bc6ca94db982f0188b33043f25771c40d02be4bcb22507` | 0.10 |
| Low-latency fallback | pillar02 | 30 | `20260901T195416Z-pillar02-duration30-2720f37cf422c4e55bafd0a6` | `ebed7d29b96cae0812ede9e572ffb1ba054d650ad62cb1c6c8895697fcb3a5d9` | `2606a3448cd9edc97b662b0ea8631ea828ed1ba7fe64578bba1f2f5b650c8cac` | 0.10 |

The closed playback registry pins the run, config, checkpoint size, and
checkpoint SHA-256 for both aliases. `voxel0075` remains the default
presentation model. The 30-epoch voxel checkpoint is deliberately not
selected because it was worse than the 20-epoch checkpoint in every requested
3D/BEV AP40 slice and in canonical p95 latency.

The 0.10 presentation threshold preserves the existing canonical NMS and
detector setting. It has not been relabeled as an AP-optimal visual tuning
result.

## Canonical paired comparison

| Model / epochs | 3D AP40 E / M / H | BEV AP40 E / M / H | Prediction p50 / p95 ms | E2E p50 / p95 ms | Peak allocated / reserved MiB | Best epoch | Status / delta vs 20 |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| voxel0075 / 20 | 78.4480 / 66.1696 / 62.1959 | 88.0176 / 82.0890 / 77.8536 | 31.2469 / 33.6592 | 32.8995 / 35.3310 | 100.44 / 170.00 | 20 | completed / baseline |
| voxel0075 / 30 | 77.2638 / 65.8049 / 62.0019 | 87.5701 / 80.0926 / 77.7030 | 32.0940 / 39.6069 | 33.9328 / 42.0169 | 100.44 / 170.00 | 30 | completed / moderate Δ −0.3647 |
| pillar02 / 20 | 68.9758 / 55.1909 / 49.7370 | 85.6702 / 75.6608 / 71.1751 | 12.5046 / 13.0762 | 14.2425 / 15.0032 | 147.00 / 428.00 | 20 | completed / baseline |
| pillar02 / 30 | 73.2703 / 59.3263 / 53.6976 | 87.7711 / 77.0989 / 74.4106 | 12.6915 / 13.2319 | 14.4154 / 15.0919 | 147.00 / 428.00 | 30 | completed / moderate Δ +4.1354 |

The immutable result IDs, full deltas, methodology, compatibility evidence,
and limitations are preserved in
[`../research/reports/20260902-finalists-duration30/`](../research/reports/20260902-finalists-duration30/).
Both 30-epoch runs completed with one successful training attempt, one
successful evaluation, one successful benchmark, and no failure record.

The comparison has one explicit compatibility waiver: the KITTI release label
was not recorded. Exact semantic dataset identity and train/validation
annotation hashes match. These remain single-seed validation results, not
untouched sealed-test or statistical-significance claims.

## Training decision

No additional training was started. The available presentation training
budget remains unconsumed: the strongest accuracy checkpoint is already the
20-epoch voxel model, and the accepted 30-epoch pillar model provides the
improved low-latency alternative.

## Release acceptance

The final integration passed the complete CPU suite, compilation/import,
strict comparison reload, registry artifact verification, launcher dry-run,
and tracked playback tests. The 1,000-iteration synthetic tracker benchmark
with 100 detections per frame measured 1.141 ms p50 and 1.620 ms p95 against
the documented 2 ms p95 target.

Live voxel acceptance used the exact primary checkpoint and unchanged 0.10
threshold. At 0.5× (`all`, queue 32) and 1.0× (`latest`, queue 1), the live
graph and connected Foxglove Bridge carried a base-frame model point cloud,
decoded 1920×1080 camera images, raw boxes, stable tracked IDs, Car labels,
scores, velocity arrows, coasting labels/opacity, and trails bounded to 20
points. TF included `map -> lexus3/base_link`; detector and tracking diagnostics
advanced with zero processing or tracking failures. A backward seek caused a
second generation/reset and raw/tracked `DELETEALL` before fresh tracks. Both
launches and the pillar30 smoke launch stopped without traceback or owned
processes left behind. The pillar smoke diagnostics pinned run
`20260901T195416Z-pillar02-duration30-2720f37cf422c4e55bafd0a6` and checkpoint
`2606a3448cd9edc97b662b0ea8631ea828ed1ba7fe64578bba1f2f5b650c8cac`.

The live DDS reported non-fatal PointCloud2 loss at both rates. The detector
diagnostic records that separately as a warning; it is not a processing or
tracking failure. At the sampled 1.0× point the freshness policy also reported
103 intentional replacements from 327 received frames, so this run is not
zero-drop real-time evidence. Raw detector output remained valid, and the
tracking diagnostic stayed OK with `failed_tracking_frames=0`.

The bounded fallback is
`/media/ws-rtx/datastore1/centerpoint_presentation_demos/voxel0075_tracked_20260902T091653Z`:
29.342 seconds, 4,651 messages on all 13 required topics, 443.8 MiB. It replayed
through Foxglove Bridge with no detector node, CUDA hidden, and no GPU compute
process.
