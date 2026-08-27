# Clean run-owned research foundation

The M1–M9 foundation is complete on branch `dev`. The preserved handoff was
commit `a50e4c5` (`research: checkpoint run-owned architecture recovery`); this
continuation did not revert to or reconstruct the legacy `main` architecture.

## Milestones

- **M1–M4 — complete at handoff.** Durable storage, provenance, checkpoint
  identity, canonical runs, immutable results, training/scheduling, evaluation,
  synchronized benchmarking, smoke execution, and playback were retained.
- **M5 — complete.** Comparison schema v2 carries resolved available AP40,
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

## Validation

Full real Python 3.10 environment command:

```bash
MPLCONFIGDIR=/tmp/lidar-matplotlib \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=research/src:runtime \
/home/ws-rtx/anaconda3/envs/lidar_centerpoint_g/bin/python \
  -m pytest -p no:cacheprovider research/tests runtime/test -q
```

Result: **373 passed, 0 failed, 0 skipped** in 14.66 seconds.

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

GPU execution remains blocked: `nvidia-smi` cannot communicate with the NVIDIA
driver and Torch reports zero CUDA devices. Therefore no new real CUDA
training, checkpoint evaluation, synchronized benchmark, or smoke execution
was claimed. Exact historical outputs were migrated and parity-checked without
re-running their GPU measurement.

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

The next major roadmap phase is **TensorRT feasibility and numerical parity**.
Do not begin it as part of this foundation milestone.
