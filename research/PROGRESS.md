# Clean run-owned research foundation

The M1–M9 foundation is complete on branch `dev`. The preserved handoff was
commit `a50e4c5` (`research: checkpoint run-owned architecture recovery`); this
continuation did not revert to or reconstruct the legacy `main` architecture.

## Milestones

- **M1–M4 — complete at handoff.** Durable storage, provenance, checkpoint
  identity, canonical runs, immutable results, training/scheduling, evaluation,
  synchronized benchmarking, smoke execution, and playback were retained.
- **M5 — complete.** Comparison schema v3 carries resolved available AP40,
  latency, memory, checkpoint-size, and 20 Hz evidence. `plotting.py` renders
  only resolved rows, with a thin `tools/plot.py`. Dataset scheme
  `lidar-dataset-v2` excludes observed machine paths from semantic identity
  while preserving `root_reference`; v1 records retain their old path-bound
  meaning through resume, lineage, and relocated-data checks. E2E comparisons
  include a normalized CPU model, architecture, and OS class; architecture-only
  CPU evidence remains unknown, and unknown/mismatched host evidence requires a
  persisted waiver. Prediction-only timing does not require host matching, and
  plotting renders only the selected timing scope.
- **M6 — complete for available evidence.** Exact effective configs, literal
  epoch-20 checkpoints, selected-best checkpoints, evaluation JSON, benchmark
  JSON, and KITTI info files were present for all six historical models. Six
  imported, non-resumable runs were created locally under ignored
  `research/runs/`. Accuracy ranking exactly reproduced 66.9838, 63.9242,
  63.2985, 62.8605, 56.0098, and 53.3350 AP40 moderate. E2E p95 exactly
  reproduced 14.9157, 15.3299, 32.2739, 34.7634, 37.1552, and 40.1738 ms.
  Historical unknowns remain unknown and comparisons persist explicit waivers.
- **M7 — complete.** `preflight.py` performs focused config, dataset root,
  metadata containment, class, one-sample, output-path, framework, and CUDA
  readiness checks. `pipeline.py` calls the normal public train, evaluate, and
  benchmark functions and publishes immutable records pinning result IDs.
  `tools/run.py` supports both catalog presets and explicit configs through the
  same `create_training_run` materialization and evidence path.
- **M8 — complete.** Obsolete `tools/test.py`, `tools/plot_results.py`, and
  `tools/train_screening_wave.sh` were removed after behavior review. Root and
  research documentation now describe only the run-owned workflow. Searches
  find no active screen-directory discovery, checkpoint fallback, mutable
  result-summary workflow, or detached config/checkpoint execution path.
- **M9 — complete.** Responsibility, large-module, no-bloat, secrets,
  generated-file, whitespace, CLI, grammar, and runtime-boundary reviews pass.
  No Manager/Service/Repository/Controller layers or competing Experiment
  concept were introduced.
- **M10 — ROS2 playback adapter hardened; visual acceptance remains.** The
  standard-message Foxglove adapter now has repository-owned complete
  `/tf_static` replay QoS, backward-clock generation/TF reset, lifecycle-safe
  shutdown, policy-matched PointCloud2 subscription depth, distinct middleware
  and application loss accounting, and stage-accurate live diagnostics. The
  July 27 synchronized LiDAR/camera acceptance procedure is documented in
  [`docs/foxglove_playback.md`](../docs/foxglove_playback.md). This does not
  claim that the remaining user-run GPU visual acceptance has passed.

## Fresh six-model campaign

The six fresh KITTI Car-only CenterPoint runs created on 2026-08-27 completed
all 20 training epochs. Each protected run subsequently produced one successful
immutable prediction-smoke result, one final KITTI evaluation result, and one
synchronized benchmark result. The resolved evidence is recorded under
[`reports/20260827-six-model-20epoch/`](reports/20260827-six-model-20epoch/):

- [`fresh-end-to-end-p95.json`](reports/20260827-six-model-20epoch/fresh-end-to-end-p95.json)
  and [`figures/end-to-end-p95/`](reports/20260827-six-model-20epoch/figures/end-to-end-p95/);
- [`fresh-prediction-p95.json`](reports/20260827-six-model-20epoch/fresh-prediction-p95.json)
  and [`figures/prediction-p95/`](reports/20260827-six-model-20epoch/figures/prediction-p95/).

The measured Pareto frontier selects `voxel0075` as the accuracy finalist,
`pillar02` as the speed finalist, and retains `voxel01` as the middle reference.
Each DCN variant was dominated by its corresponding non-DCN model in this
campaign. All six measured end-to-end p95 latencies were below 50 ms on the
RTX 2080 Ti workstation; this preliminary result is not a low-end deployment
claim. The report README records the exact metrics, methodology, hardware,
dataset-version waiver, and study limitations.

## Paired 30-epoch finalist screen

The two fresh 30-epoch finalist runs completed cleanly and each produced one
successful canonical evaluation and one successful synchronized benchmark from
its recorded selected-best epoch-30 checkpoint. The immutable paired report is
under [`reports/20260902-finalists-duration30/`](reports/20260902-finalists-duration30/).

The 30-epoch `voxel0075` run reached 65.8049 moderate 3D AP40 versus 66.1696
for its 20-epoch baseline, while end-to-end p95 rose from 35.3310 to 42.0169
ms. The 30-epoch `pillar02` run improved moderate 3D AP40 from 55.1909 to
59.3263 while end-to-end p95 changed only from 15.0032 to 15.0919 ms. Peak
allocated/reserved GPU memory was unchanged within each architecture.

The presentation accuracy checkpoint therefore remains the 20-epoch
`voxel0075` selected-best artifact (`5246b24bfe66a81df3bc6ca94db982f0188b33043f25771c40d02be4bcb22507`),
with the 30-epoch `pillar02` artifact retained as the improved low-latency
fallback. No additional training was started. Presentation operating-point
tuning and live Foxglove acceptance remain separate from this canonical
post-training comparison.

## Validation

Full real Python 3.10 environment command:

```bash
MPLCONFIGDIR=/tmp/lidar-matplotlib \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=research/src:runtime \
/home/ws-rtx/anaconda3/envs/lidar_centerpoint_g/bin/python \
  -m pytest -p no:cacheprovider research/tests runtime/test -q
```

Foundation result: **373 passed, 0 failed, 0 skipped** in 14.66 seconds.
Current post-campaign result: **406 passed, 0 failed, 0 skipped** in 16.73
seconds.
Current post-playback-hardening result: **550 passed, 0 failed, 0 skipped** in
17.42 seconds.

Also passed:

- Python 3.10 grammar parsing across 109 Python files;
- lightweight research/runtime import boundaries;
- eight research CLI `--help` checks;
- whitespace/conflict-marker/secret/generated-junk checks;
- all six configs loaded with real MMEngine/MMDetection3D;
- real KITTI validation dataset construction (3,769 samples) and sample 0
  access, producing `inputs` and a `Det3DDataSample`.

## Real stack

Validated imports and real dataset access with:

- Python 3.10.20
- Torch 2.1.2+cu121
- NumPy 1.26.4
- Matplotlib 3.10.9
- MMEngine 0.10.7
- MMCV 2.1.0
- MMDetection 3.3.0
- MMDetection3D 1.4.0

The Codex validation environment remains GPU-inaccessible. The completed
campaign's immutable host-generated records, however, bind successful smoke,
evaluation, and synchronized benchmark execution on an NVIDIA GeForce RTX 2080
Ti with NVIDIA driver `575.57.08`. Historical imported evidence remains
separate from these fresh results.

## Architecture review

The larger modules remain cohesive by responsibility: `runs.py` owns run
schema/lifecycle/revision; `provenance.py` evidence; `training.py` training
lifecycle; `results.py` immutable results; `checkpoints.py` checkpoint
structure/identity; and `benchmarking.py` one-run timing execution.
`comparison.py` remains the largest module because strict serialization,
compatibility, KITTI AP40 projection, historical translation, waivers, and
ranking form the current comparison boundary. KITTI projection was not moved
to a speculative profile package: with only one metric family, extraction
would mostly relocate helpers rather than simplify ownership. Revisit that
boundary when nuScenes adds a genuinely second metric profile.

No remaining production file appears bloated through unrelated ownership, and
logic is not scattered across CLIs. Generated local runs and checkpoints remain
ignored; tracked historical summaries/figures remain intentional evidence.

## Next phase

The next acceptance step is the documented host-GPU side-by-side Kaposvár
bounding-box inspection using the working July 27 camera recording. Runtime
comparison and detection-level tracking remain later work and are not part of
the adapter hardening. TensorRT, loss changes, NuScenes training, multi-seed
experiments, longer finalist training, and an official-pretrained checkpoint
comparison remain explicitly deferred.
