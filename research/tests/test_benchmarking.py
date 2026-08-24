from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

import lidar_model_selection.benchmarking as benchmarking
from lidar_model_selection.checkpoints import TrainingOutputs, identify_checkpoint
from lidar_model_selection.provenance import (
    CodeProvenance,
    EnvironmentInfo,
    build_training_compatibility,
    capture_code_provenance,
    identify_file_set,
)
from lidar_model_selection.results import list_results, load_result
from lidar_model_selection.runs import (
    Run,
    TrainingState,
    build_dataset_identity,
    create_run,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_TOOL = REPOSITORY_ROOT / "research" / "tools" / "benchmark.py"


def _write_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr("archive/data.pkl", b"structural checkpoint")
        archive.writestr("archive/version", b"3")


def _dataset():
    return build_dataset_identity(
        name="KITTI",
        version=None,
        root_reference="dataset:kitti-test",
        semantic_partition="KITTI validation",
        framework_key="test_dataloader",
        annotation_files=None,
        class_names=("Car",),
        tasks={"3d_detection": ("Car",)},
    )


def _completed_run(tmp_path: Path) -> Run:
    final_path = tmp_path / "external" / "epoch_4.pth"
    selected_path = tmp_path / "external" / "best_score.pth"
    _write_checkpoint(final_path)
    _write_checkpoint(selected_path)
    outputs = TrainingOutputs(
        final_checkpoint=identify_checkpoint(final_path),
        selected_checkpoint=identify_checkpoint(selected_path),
    )
    return create_run(
        tmp_path / "runs",
        slug="benchmark-run",
        config_bytes=(
            b"custom_imports = dict(imports=['example_plugin'], "
            b"allow_failed_imports=False)\nmodel = dict()\n"
        ),
        dataset=_dataset(),
        target_epoch=4,
        origin="historical_import",
        training_state=TrainingState(
            status="completed",
            attempts=(),
            outputs=outputs,
        ),
    )


def _environment(*, torch_observed: bool) -> EnvironmentInfo:
    return EnvironmentInfo(
        python_version="3.10.14",
        python_implementation="CPython",
        platform="Linux-test",
        machine="x86_64",
        executable="/usr/bin/python3.10",
        packages=(("mmengine", "0.10.7"), ("torch", "2.1.2")),
        torch_version="2.1.2" if torch_observed else None,
        cuda_version="12.1" if torch_observed else None,
        cudnn_version="8902" if torch_observed else None,
        gpu_available=True if torch_observed else None,
        gpu_devices=("Fake GPU",) if torch_observed else (),
    )


@pytest.fixture(scope="module")
def code_provenance() -> CodeProvenance:
    return capture_code_provenance(
        REPOSITORY_ROOT,
        ("research/src/lidar_model_selection/benchmarking.py",),
    )


def _stub_evidence(
    monkeypatch: pytest.MonkeyPatch,
    code_provenance: CodeProvenance,
) -> None:
    monkeypatch.setattr(
        benchmarking,
        "_capture_initial_evidence",
        lambda: (code_provenance, _environment(torch_observed=False)),
    )
    monkeypatch.setattr(
        benchmarking,
        "_capture_execution_environment",
        lambda: _environment(torch_observed=True),
    )


def _fake_mmengine(
    monkeypatch: pytest.MonkeyPatch,
    run: Run,
    *,
    warmup: int,
    samples: int,
    available_batches: int | None = None,
    mutate_after_config: str | None = None,
) -> tuple[list[str], list[dict[str, object]], list[int]]:
    events: list[str] = []
    configs: list[dict[str, object]] = []
    iterator_counts = [0]
    batch_count = (
        warmup + samples if available_batches is None else available_batches
    )

    class Config(dict):
        @classmethod
        def fromfile(cls, path: str):
            events.append(f"config:{path}")
            if mutate_after_config == "config":
                Path(path).write_text("model = dict(tampered=True)\n", encoding="utf-8")
            elif mutate_after_config == "checkpoint":
                assert run.selected_checkpoint is not None
                Path(run.selected_checkpoint.path).write_bytes(b"tampered")
            return cls(
                custom_imports={
                    "imports": ["example_plugin"],
                    "allow_failed_imports": False,
                },
                test_cfg={"type": "TestLoop"},
                test_dataloader={
                    "batch_size": 8,
                    "num_workers": 4,
                    "persistent_workers": True,
                    "drop_last": True,
                    "sampler": {"type": "DefaultSampler", "shuffle": True},
                },
            )

    class Dataloader:
        def __iter__(self):
            iterator_counts[0] += 1
            events.append("iterator")
            return iter(range(1, batch_count + 1))

    class Model:
        def eval(self) -> None:
            events.append("eval")

        def parameters(self):
            return iter((types.SimpleNamespace(dtype="torch.float32"),))

        def test_step(self, batch: object) -> list[object]:
            events.append(f"test_step:{batch}")
            return [batch]

    class RunnerInstance:
        def __init__(self) -> None:
            self.model = Model()

        def load_or_resume(self) -> None:
            events.append("load_or_resume")

        def build_test_loop(self, test_cfg: object):
            events.append("build_test_loop")
            assert test_cfg == {"type": "TestLoop"}
            return types.SimpleNamespace(dataloader=Dataloader())

    class Runner:
        @classmethod
        def from_cfg(cls, config):
            events.append("runner")
            configs.append(config)
            assert Path(config["work_dir"]).is_dir()
            return RunnerInstance()

    class Cuda:
        def is_available(self) -> bool:
            return True

        def device_count(self) -> int:
            return 1

        def set_device(self, index: int) -> None:
            events.append(f"set_device:{index}")

        def synchronize(self) -> None:
            events.append("synchronize")

        def empty_cache(self) -> None:
            events.append("empty_cache")

        def reset_peak_memory_stats(self) -> None:
            events.append("reset_peak")

        def max_memory_allocated(self) -> int:
            return 2 * 1024**2

        def max_memory_reserved(self) -> int:
            return 3 * 1024**2

        def get_device_name(self, index: int) -> str:
            assert index == 0
            return "Fake GPU"

    class InferenceMode:
        def __enter__(self):
            events.append("inference_enter")
            return self

        def __exit__(self, exc_type, exc, traceback_value) -> None:
            events.append("inference_exit")

    fake_torch = types.SimpleNamespace(
        cuda=Cuda(),
        inference_mode=lambda: InferenceMode(),
    )
    modules = {
        "lidar_model_selection.compat.kitti_evaluator": types.SimpleNamespace(
            install=lambda: events.append("compat")
        ),
        "torch": fake_torch,
        "mmdet3d.utils": types.SimpleNamespace(
            register_all_modules=lambda **options: events.append(
                f"register:{options['init_default_scope']}"
            )
        ),
        "mmengine.config": types.SimpleNamespace(Config=Config),
        "mmengine.utils": types.SimpleNamespace(
            import_modules_from_strings=lambda **options: events.append(
                f"custom:{options['imports'][0]}"
            )
        ),
        "mmengine.runner": types.SimpleNamespace(Runner=Runner),
    }

    def import_module(name: str):
        if name not in modules:
            raise AssertionError(f"unexpected lazy import: {name}")
        return modules[name]

    monkeypatch.setattr(
        benchmarking,
        "importlib",
        types.SimpleNamespace(import_module=import_module),
    )
    monkeypatch.setattr(benchmarking.gc, "collect", lambda: 0)

    clock_values = []
    for index in range(samples):
        base = index * 100_000_000
        clock_values.extend((base, base + 10_000_000, base + 20_000_000))
    clock = iter(clock_values)
    monkeypatch.setattr(
        benchmarking.time,
        "perf_counter_ns",
        lambda: next(clock),
    )
    return events, configs, iterator_counts


def test_latency_statistics_are_finite_linear_and_population_based() -> None:
    statistics = benchmarking.latency_statistics([1.0, 2.0, 3.0, 4.0])

    assert statistics == pytest.approx(
        {
            "count": 4,
            "mean_ms": 2.5,
            "min_ms": 1.0,
            "max_ms": 4.0,
            "p50_ms": 2.5,
            "p95_ms": 3.85,
            "p99_ms": 3.97,
            "standard_deviation_ms": math.sqrt(1.25),
        }
    )


@pytest.mark.parametrize(
    "values, error_type",
    [
        ([], ValueError),
        ([float("nan")], ValueError),
        ([float("inf")], ValueError),
        ([-0.1], ValueError),
        ([True], TypeError),
        (["1.0"], TypeError),
    ],
)
def test_latency_statistics_reject_non_strict_measurements(
    values: list[object],
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        benchmarking.latency_statistics(values)  # type: ignore[arg-type]


def test_threshold_evidence_uses_end_to_end_p95_and_strictly_over_50ms() -> None:
    measured = benchmarking._measurement_payload(
        prediction_times=[1.0, 2.0, 3.0, 4.0],
        end_to_end_times=[49.0, 50.0, 50.1, 100.0],
        peak_allocated_bytes=1024,
        peak_reserved_bytes=2048,
    )

    end_to_end = measured["end_to_end_ms"]
    assert end_to_end["frames_over_50ms"] == 2  # type: ignore[index]
    assert end_to_end["percentage_over_50ms"] == 50.0  # type: ignore[index]
    assert end_to_end["meets_20hz"] is False  # type: ignore[index]


def test_success_uses_one_iterator_synchronized_scopes_and_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code_provenance: CodeProvenance,
) -> None:
    run = _completed_run(tmp_path)
    _stub_evidence(monkeypatch, code_provenance)
    events, configs, iterator_counts = _fake_mmengine(
        monkeypatch,
        run,
        warmup=2,
        samples=3,
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-fake")

    record = benchmarking.benchmark_run(run, warmup=2, samples=3)

    assert record.successful is True
    assert record.provenance == code_provenance
    assert record.environment == _environment(torch_observed=True)
    assert record.binding.run_id == run.run_id
    assert record.payload["kind"] == "benchmark"
    assert record.payload["benchmark_schema_version"] == 1
    assert dict(record.payload["methodology"])["key"] == (
        "mmdet3d_prediction_e2e_sync_v1"
    )
    workload = record.payload["workload"]
    assert dict(workload) == {
        "semantic_partition": "KITTI validation",
        "framework_key": "test_dataloader",
        "batch_size": 1,
        "num_workers": 0,
        "persistent_workers": False,
        "drop_last": False,
        "shuffle": False,
        "warmup_count": 2,
        "measured_sample_count": 3,
    }
    prediction = record.payload["prediction_ms"]
    end_to_end = record.payload["end_to_end_ms"]
    assert prediction["count"] == end_to_end["count"] == 3  # type: ignore[index]
    assert prediction["mean_ms"] == pytest.approx(10.0)  # type: ignore[index]
    assert end_to_end["mean_ms"] == pytest.approx(20.0)  # type: ignore[index]
    assert end_to_end["frames_over_50ms"] == 0  # type: ignore[index]
    assert end_to_end["meets_20hz"] is True  # type: ignore[index]
    assert dict(record.payload["peak_memory"]) == {
        "allocated_bytes": 2 * 1024**2,
        "reserved_bytes": 3 * 1024**2,
        "allocated_mib": 2.0,
        "reserved_mib": 3.0,
    }
    assert dict(record.payload["hardware"])["device_name"] == "Fake GPU"
    host = dict(dict(record.payload["hardware"])["host"])
    assert set(host) == {"cpu_model", "architecture", "os_class"}
    assert all(isinstance(value, str) for value in host.values())
    assert dict(record.payload["precision"])["model_parameter_dtypes"] == (
        "torch.float32",
    )
    assert iterator_counts == [1]
    assert events.count("synchronize") == 1 + 3 * 3
    assert events.count("empty_cache") == 1
    assert events.count("reset_peak") == 1
    assert [event for event in events if event.startswith("test_step:")] == [
        "test_step:1",
        "test_step:2",
        "test_step:3",
        "test_step:4",
        "test_step:5",
    ]

    config = configs[0]
    assert config["load_from"] == run.selected_checkpoint.path
    assert config["resume"] is False
    assert config["launcher"] == "none"
    assert config["test_dataloader"] == {
        "batch_size": 1,
        "num_workers": 0,
        "persistent_workers": False,
        "drop_last": False,
        "sampler": {"type": "DefaultSampler", "shuffle": False},
    }
    assert not Path(config["work_dir"]).exists()
    assert load_result(run, "benchmark", record.result_id) == record


@pytest.mark.parametrize("mutated_input", ["config", "checkpoint"])
def test_execution_inputs_are_reverified_immediately_before_runner_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code_provenance: CodeProvenance,
    mutated_input: str,
) -> None:
    run = _completed_run(tmp_path)
    _stub_evidence(monkeypatch, code_provenance)
    events, _, _ = _fake_mmengine(
        monkeypatch,
        run,
        warmup=1,
        samples=1,
        mutate_after_config=mutated_input,
    )

    record = benchmarking.benchmark_run(run, warmup=1, samples=1)

    assert record.status == "failed"
    assert record.failure is not None
    assert "runner" not in events
    assert record.failure.error_type == "ValueError"
    if mutated_input == "config":
        assert "config bytes" in record.failure.message
    else:
        assert "checkpoint identity mismatch" in record.failure.message
    assert len(list_results(run, "benchmark")) == 1


def test_checkpoint_tamper_is_recorded_before_any_lazy_ml_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code_provenance: CodeProvenance,
) -> None:
    run = _completed_run(tmp_path)
    assert run.selected_checkpoint is not None
    Path(run.selected_checkpoint.path).write_bytes(b"not a checkpoint")
    _stub_evidence(monkeypatch, code_provenance)

    def forbidden_import(name: str):
        raise AssertionError(f"heavy import occurred after failed verification: {name}")

    monkeypatch.setattr(
        benchmarking,
        "importlib",
        types.SimpleNamespace(import_module=forbidden_import),
    )

    record = benchmarking.benchmark_run(run, warmup=1, samples=1)

    assert record.status == "failed"
    assert record.failure is not None
    assert record.failure.error_type == "ValueError"
    assert "checkpoint identity mismatch" in record.failure.message
    assert record.payload["prediction_ms"] == {}


def test_initial_config_tamper_fails_before_publication_or_ml_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _completed_run(tmp_path)
    run.paths.config.write_text("model = dict(tampered=True)\n", encoding="utf-8")
    monkeypatch.setattr(
        benchmarking,
        "importlib",
        types.SimpleNamespace(
            import_module=lambda name: pytest.fail(f"unexpected heavy import {name}")
        ),
    )

    with pytest.raises(ValueError, match="config bytes"):
        benchmarking.benchmark_run(run, warmup=1, samples=1)

    assert tuple(run.paths.benchmark.iterdir()) == ()


def test_dataloader_exhaustion_becomes_a_fresh_failure_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code_provenance: CodeProvenance,
) -> None:
    run = _completed_run(tmp_path)
    _stub_evidence(monkeypatch, code_provenance)
    _fake_mmengine(
        monkeypatch,
        run,
        warmup=1,
        samples=2,
        available_batches=2,
    )

    record = benchmarking.benchmark_run(run, warmup=1, samples=2)

    assert record.status == "failed"
    assert record.failure is not None
    assert record.failure.error_type == "RuntimeError"
    assert "measured sample 2" in record.failure.message
    assert record.payload["prediction_ms"] == {}
    assert load_result(run, "benchmark", record.result_id) == record


@pytest.mark.parametrize(
    "warmup, samples, error_type",
    [
        (0, 1, ValueError),
        (1, 0, ValueError),
        (-1, 1, ValueError),
        (True, 1, TypeError),
        (1, 1.5, TypeError),
    ],
)
def test_counts_must_be_explicit_positive_integers(
    tmp_path: Path,
    warmup: object,
    samples: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        benchmarking.benchmark_run(
            tmp_path / "not-loaded",
            warmup=warmup,  # type: ignore[arg-type]
            samples=samples,  # type: ignore[arg-type]
        )


def test_incomplete_run_is_rejected_before_ml_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code_provenance: CodeProvenance,
) -> None:
    config = b"model = dict()\n"
    environment = _environment(torch_observed=False)
    annotation = tmp_path / "kitti_infos_val.pkl"
    annotation.write_bytes(b"validation annotation identity")
    annotations = identify_file_set(tmp_path, (annotation,))
    dataset = build_dataset_identity(
        name="KITTI",
        version=None,
        root_reference="dataset:kitti-test",
        semantic_partition="KITTI validation",
        framework_key="test_dataloader",
        annotation_files=annotations,
        class_names=("Car",),
        tasks={"3d_detection": ("Car",)},
    )
    sources = identify_file_set(
        REPOSITORY_ROOT,
        (REPOSITORY_ROOT / "research/src/lidar_model_selection/benchmarking.py",),
    )
    compatibility = build_training_compatibility(
        hashlib.sha256(config).hexdigest(),
        dataset.identity_sha256,
        sources,
        core_packages=dict(environment.packages),
        python_version=environment.python_version,
    )
    run = create_run(
        tmp_path / "runs",
        slug="pending-benchmark",
        config_bytes=config,
        dataset=dataset,
        target_epoch=4,
        code_provenance=code_provenance,
        environment=environment,
        training_compatibility=compatibility,
    )
    monkeypatch.setattr(
        benchmarking,
        "importlib",
        types.SimpleNamespace(
            import_module=lambda name: pytest.fail(f"unexpected heavy import {name}")
        ),
    )

    with pytest.raises(ValueError, match="completed run"):
        benchmarking.benchmark_run(run, warmup=1, samples=1)

    assert tuple(run.paths.benchmark.iterdir()) == ()


def test_benchmark_module_and_cli_have_no_eager_ml_imports() -> None:
    source = REPOSITORY_ROOT / "research" / "src"
    script = f"""
import importlib.util
import sys
before = set(sys.modules)
import lidar_model_selection.benchmarking
spec = importlib.util.spec_from_file_location(
    'benchmark_tool_test',
    {str(BENCHMARK_TOOL)!r},
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
forbidden = {{'torch', 'numpy', 'mmengine', 'mmcv', 'mmdet', 'mmdet3d'}}
loaded = sorted(
    name for name in forbidden if name in sys.modules and name not in before
)
assert not loaded, loaded
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(source), environment.get("PYTHONPATH", ""))
    )

    subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _load_benchmark_tool():
    spec = importlib.util.spec_from_file_location(
        "benchmark_tool_under_test",
        BENCHMARK_TOOL,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_accepts_only_one_run_and_explicit_positive_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_benchmark_tool()
    run_id = "20260818T120000Z-benchmark-run-" + "1" * 24
    observed: list[tuple[Path, int, int]] = []
    record = types.SimpleNamespace(
        binding=types.SimpleNamespace(run_id=run_id),
        result_id="20260818T120000000000Z-benchmark-" + "2" * 24,
        status="succeeded",
        successful=True,
        failure=None,
    )

    monkeypatch.setattr(tool, "DEFAULT_RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(
        tool,
        "benchmark_run",
        lambda path, *, warmup, samples: (
            observed.append((path, warmup, samples)) or record
        ),
    )

    assert tool.main(
        [
            "--run",
            run_id,
            "--warmup",
            "3",
            "--samples",
            "7",
            "--gpu",
            "GPU-deadbeef",
        ]
    ) == 0
    assert observed == [(tmp_path / "runs" / run_id, 3, 7)]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-deadbeef"

    parser = tool.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["config.py", "checkpoint.pth"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--all"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--run", run_id, "--warmup", "1"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--run", run_id, "--warmup", "0", "--samples", "1"]
        )
