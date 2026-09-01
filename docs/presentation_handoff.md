# Presentation readiness handoff

This branch implements the independent CPU/runtime portion of the presentation
goal. The paired 30-epoch campaign remained owned by another session while this
branch was prepared, so no run artifact was modified and no GPU evaluation,
operating-point sweep, extra training, or live ROS/Foxglove acceptance was
started here.

## Current protected selection

Until both 30-epoch runs and their owner-managed evaluation/benchmark phases
are terminal, retain the existing protected registry selection:

| Role | Model | Run ID | Selected checkpoint SHA-256 | Score threshold |
|---|---|---|---|---:|
| Primary accuracy demo | voxel0075 | `20260827T092043Z-voxel0075-e583a40f435e3071e0cbd6fc` | `5246b24bfe66a81df3bc6ca94db982f0188b33043f25771c40d02be4bcb22507` | 0.10 |
| Low-latency fallback | pillar02 | `20260827T092033Z-pillar02-3367910930525d0c12ddc346` | `7814db42c341be87c09ae4e68a0266288227aeac6a98cfb83420b4ffb5caaf8d` | 0.10 |

The 0.10 threshold is the existing configurable demo default, not a completed
presentation operating-point optimization. Preserve canonical evaluation
settings and record any later threshold-only selection separately.

## Canonical comparison status

The 20-epoch rows below are canonical completed artifacts. At the last
read-only inspection, pillar30 training was complete but voxel30 was still
`running`; the paired comparison is therefore intentionally pending. Do not
fill 30-epoch cells from per-epoch training validation or partial artifacts.

| Model / run | 3D AP40 E/M/H | BEV AP40 E/M/H | Pred. p50/p95 ms | E2E p50/p95 ms | Peak alloc./reserved MiB | Best epoch / trend | Status / delta vs 20 |
|---|---|---|---|---|---|---|---|
| voxel0075 20 | 78.4480 / 66.1696 / 62.1959 | 88.0176 / 82.0890 / 77.8536 | 31.2469 / 33.6592 | 32.8995 / 35.3310 | 100.44 / 170.00 | 20 / canonical selected | completed / baseline |
| voxel0075 30 | pending | pending | pending | pending | pending | pending | owner run still active |
| pillar02 20 | 68.9758 / 55.1909 / 49.7370 | 85.6702 / 75.6608 / 71.1751 | 12.5046 / 13.0762 | 14.2425 / 15.0032 | 147.00 / 428.00 | 20 / canonical selected | completed / baseline |
| pillar02 30 | pending | pending | pending | pending | pending | pending | wait for paired owner evaluation |
| evidence-gated final | not started | not started | not started | not started | not started | not decided | blocked by paired evidence and GPU ownership |

Both baseline runs use KITTI validation dataset identity
`0bb26013400c77313f2720b2295f78fc84ba30ef1711b3c769608fa02aa3c8df`.
Repeated KITTI validation use does not support claims equivalent to an untouched
sealed test set or statistical significance.

## Deferred acceptance

After the active campaign owner releases the GPUs, follow the 0.5x and 1.0x
checklist and bounded MCAP procedure in
[`tracked_foxglove_demo.md`](tracked_foxglove_demo.md). Only then register a
new checkpoint, select a separately recorded presentation threshold, and decide
whether the evidence permits the single optional final training run.
