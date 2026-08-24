# Current architecture recovery state

This is a safe development checkpoint created on 2026-08-24 so work can move
to another machine. The active architecture goal is intentionally paused here;
it is not complete.

## Completed milestones

### M1 — persistence, provenance, and checkpoints

- `storage.py` implements strict JSON, durable atomic/exclusive publication,
  Linux no-replace directory publication, fd-relative staged-tree fsync, and
  safe explicit staging cleanup.
- `provenance.py` records deterministic file sets, scoped Git workspace
  identity (including untracked content), environment evidence, and narrow
  training compatibility.
- `checkpoints.py` owns strict checkpoint names, structural PyTorch ZIP/CRC
  validation, hashing, verification, and run-local output selection. It has no
  model catalog or global checkpoint discovery.
- Focused tests: storage 94, provenance 15, checkpoints 33 (142 total).

### M2 — canonical runs and immutable results

- `runs.py` implements transactionally created canonical runs, stable IDs,
  exact config/dataset/evidence identity, strict native/imported invariants,
  optimistic manifest revisions, and locked training-state updates.
- `results.py` implements immutable run-owned evaluation/benchmark records,
  exact run/config/selected-checkpoint bindings, fresh result IDs, and strict
  explicit-or-sole-successful selection.
- Focused tests: runs 33, results 25 (58 total).

### M3 — training and scheduling

- `catalog.py` is only the six-slug-to-source-config catalog; it owns no
  checkpoint or output paths.
- `training.py` snapshots and verifies effective configs, captures dataset and
  compatibility evidence, uses one run-local checkpoint lineage, records
  attempts/logs, locks execution, handles fresh/resume/finalize/parent
  initialization, and refuses corrupt or ambiguous output state.
- `scheduling.py` queues run IDs across explicit GPU slots with child failure
  isolation and Ctrl-C cleanup.
- `tools/train.py` is run-owned (`--model`, `--run`, or `--all`) with no
  `screenN`, cross-directory resume, candidate fallback, or destructive force
  reuse.
- Focused tests: training/catalog 38, scheduling 16 (54 total).

### M4 — evaluation, benchmarking, and smoke execution

- `evaluation.py` evaluates one completed run, preserves raw MMEngine scalar
  metric names, publishes immutable results, and also owns the selected-run
  smoke execution boundary.
- `benchmarking.py` benchmarks one completed run with methodology
  `mmdet3d_prediction_e2e_sync_v1`: one iterator, fixed batch-1 loader,
  synchronized prediction/end-to-end scopes, warm-up/reset policy, strict
  statistics, GPU memory, hardware, precision, workload, checkpoint size, and
  20 Hz evidence.
- `tools/evaluate.py`, `tools/benchmark.py`, and `tools/smoke_test.py` are thin,
  run-only CLIs. Config and selected-checkpoint identity are reverified before
  model/Runner construction.
- Focused tests: evaluation 20, benchmarking 22, smoke 3 (45 total).

### Additional stable cutover work

- Recorded-data playback now accepts a completed run and its verified selected
  checkpoint; its CLI and ROS parameters no longer accept detached config and
  checkpoint pairs.
- KITTI compatibility retains `install()` and a deterministic `self-test` CLI;
  the arbitrary official-test delegate and detached `eval-pkl` path are gone.
- Focused playback/compatibility boundary tests: 9.

## Current milestone

M5 is partially complete and stable:

- `imports.py` plus `tools/import_run.py` explicitly imports exact historical
  config/checkpoint/result evidence without copying checkpoint bytes or
  stamping current provenance. Imported runs are completed, non-resumable, and
  history-incomplete. Focused tests: 20.
- `comparison.py` plus `tools/compare.py` selects exact run/result identities,
  performs KITTI AP40 projection/ranking, enforces accuracy/runtime
  compatibility, and persists field-specific waivers. Focused tests: 15.
- `plotting.py` and `test_plotting.py` have **not** been created. Before plotting,
  extend resolved comparison rows with available ancillary AP40, latency,
  memory, and checkpoint-size evidence so plotting never has to load/discover
  results itself.

Last fully completed milestone: M4. Current partially completed milestone: M5.

## Validation at this checkpoint

The complete CPU suite passed on Python 3.14.4:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/tmp/lidar-test-packages:research/src:runtime \
python3 -m pytest -p no:cacheprovider research/tests runtime/test
```

Result: **346 passed, 0 failed, 0 skipped** in 3.06 seconds.

The `/tmp/lidar-test-packages` directory contains only the pinned test tools
(`pytest==9.1.1`, `typeguard==4.4.4`, and their dependencies); it is not part of
the repository. Changed Python sources also pass Python 3.10 AST grammar checks
and lightweight-import tests.

## Known intentional temporary breakages

- `research/tools/test.py` is the obsolete evaluator and imports APIs removed
  by the checkpoint cutover. Do not restore those APIs; delete this file after
  the verified `tools/evaluate.py` replacement at the next cleanup step.
- `research/tools/plot_results.py` still reads mutable global summaries, uses a
  fixed model order, and joins by model name. It remains only until the new
  resolved-row `plotting.py` is implemented and verified.
- `research/tools/train_screening_wave.sh` is an obsolete workstation-specific
  `screenN` launcher. The run-owned scheduler replaces it; delete it during
  the next legacy cleanup.
- The detailed command sections in the root and research READMEs are legacy.
  Both files carry a warning pointing here; rewrite them after plotting and
  legacy deletion so the final documentation describes only the cutover.

## Historical evidence status

The six tracked evaluation/benchmark summaries and seven PNG figures are
preserved as historical evidence. This clone does not contain their referenced
checkpoint files, saved effective configs, per-run metrics/latency files, or
logs. Therefore no historical run was fabricated or migrated. The explicit
importer is ready when the exact artifacts are supplied.

## Known environment blockers

- GPU access is blocked in this WSL environment (`nvidia-smi` cannot access a
  GPU), so real CUDA/OpenMMLab training, evaluation, benchmark, and smoke parity
  were not executed. CPU tests mock only the heavy execution boundary.
- KITTI data and the historical checkpoints/effective configs are absent.
- Python 3.10 is not installed locally. Runtime tests use Python 3.14.4; Python
  3.10 compatibility is checked with `ast.parse(..., feature_version=(3, 10))`.
- Torch, MMEngine, MMCV, MMDetection, MMDetection3D, NumPy, and Matplotlib are
  not installed in the system interpreter. Dependency declarations were not
  changed to hide this environmental limitation.

## Next actions

1. Extend `ComparisonRow` with immutable already-resolved ancillary AP40,
   latency-statistics, GPU-memory, and checkpoint-size evidence.
2. Implement `plotting.py` and `test_plotting.py` using lazy Matplotlib and only
   resolved comparison rows; preserve the useful legacy plots without model
   discovery or a fixed model universe.
3. Run the focused comparison/plotting suite and the full CPU suite.
4. Delete obsolete `tools/test.py`, `tools/plot_results.py`, and
   `tools/train_screening_wave.sh`; search again for active `screenN`, candidate,
   global checkpoint, mutable-summary, and detached config/checkpoint paths.
5. Rewrite root/research documentation for the final run-owned commands and
   document that historical migration is unavailable until exact artifacts
   arrive.
6. Perform the M8 diff audit: unnecessary abstractions, duplicated ownership,
   dead code, compatibility wrappers, blobs, generated files, and boundary
   imports; then run the real Python 3.10 and GPU parity lanes when available.

## Architectural invariants

- One canonical run owns its effective config, dataset/evidence identity,
  attempts, and exact final/selected checkpoint identities.
- No `screenN` discovery, global checkpoint guessing, candidate fallback, or
  cross-directory resume may return to active code.
- Results and result bindings are immutable; repeated invocations create fresh
  IDs, and comparisons consume exact IDs (or the sole successful result).
- Exact config and checkpoint bytes are hashed and reverified at execution
  boundaries.
- CLIs remain thin; scientific and persistence logic belongs in the package.
- Preserve useful behavior, not deprecated compatibility interfaces.
- The future ROS 2 vehicle runtime remains independent of the research package
  and MMDetection3D training stack.
