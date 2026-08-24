# LiDAR model selection

The `lidar-model-selection` package owns the research lifecycle:

- durable run, provenance, checkpoint, and result evidence;
- canonical MMEngine config materialization and run-local training/resume;
- one-run evaluation, synchronized research benchmarking, and smoke execution;
- explicit historical import, compatible comparison, and renderer-only plots;
- focused preflight and one-experiment pipeline orchestration.

Install from the repository root:

```bash
python -m pip install -e research
```

The convenient end-to-end command accepts either a catalog preset or an
explicit config. Both use the same ordinary `Run` creation and execution path:

```bash
python research/tools/run.py pillar02 --max-epochs 20

python research/tools/run.py \
  --config research/configs/my_experiment.py \
  --name my-experiment \
  --max-epochs 20
```

The lower-level run-owned commands are `train.py`, `evaluate.py`,
`benchmark.py`, `smoke_test.py`, `compare.py`, `plot.py`, and `import_run.py`.
They accept explicit run or result identities and never scan global experiment
directories or guess checkpoints.

Generated runs live under `research/runs/` and are ignored by Git. Each native
run owns its canonical config, training directory, exact final and selected
checkpoint identities, immutable results, and pipeline records. Imported
historical runs remain non-resumable and retain unknown provenance as unknown.

Nothing in this package is part of the production vehicle runtime.
