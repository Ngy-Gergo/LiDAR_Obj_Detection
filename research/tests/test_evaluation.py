from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

import lidar_model_selection.evaluation as evaluation
from lidar_model_selection.checkpoints import (
    CheckpointArtifact,
    TrainingOutputs,
    identify_checkpoint,
)
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
EVALUATE_TOOL = REPOSITORY_ROOT / "research" / "tools" / "evaluate.py"
_RAW_NAME = "Kitti metric/pred_instances_3d/KITTI/Car_3D_AP40_moderate_strict"


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
        slug="evaluation-run",
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
        gpu_available=False if torch_observed else None,
        gpu_devices=(),
    )


@pytest.fixture(scope="module")
def code_provenance() -> CodeProvenance:
    return capture_code_provenance(
        REPOSITORY_ROOT,
        ("research/src/lidar_model_selection/evaluation.py",),
    )


def _stub_evidence(
    monkeypatch: pytest.MonkeyPatch,
    code_provenance: CodeProvenance,
) -> None:
    monkeypatch.setattr(
        evaluation,
        "_capture_initial_evidence",
        lambda: (code_provenance, _environment(torch_observed=False)),
    )
    monkeypatch.setattr(
        evaluation,
        "_capture_execution_environment",
        lambda: _environment(torch_observed=True),
    )


def _fake_mmengine(
    monkeypatch: pytest.MonkeyPatch,
    *,
    metrics: object | None = None,
    test_error: BaseException | None = None,
) -> tuple[list[str], list[dict[str, object]]]:
    events: list[str] = []
    configs: list[dict[str, object]] = []
    returned_metrics = {_RAW_NAME: 0.625} if metrics is None else metrics

    class Config(dict):
        @classmethod
        def fromfile(cls, path: str):
            events.append(f"config:{path}")
            return cls(
                custom_imports={
                    "imports": ["example_plugin"],
                    "allow_failed_imports": False,
                }
            )

    class RunnerInstance:
        def test(self):
            events.append("test")
            if test_error is not None:
                raise test_error
            return returned_metrics

    class Runner:
        @classmethod
        def from_cfg(cls, config):
            events.append("runner")
            configs.append(config)
            assert Path(config["work_dir"]).is_dir()
            return RunnerInstance()

    modules = {
        "lidar_model_selection.compat.kitti_evaluator": types.SimpleNamespace(
            install=lambda: events.append("compat")
        ),
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
        evaluation,
        "importlib",
        types.SimpleNamespace(import_module=import_module),
    )
    return events, configs


class _Scalar:
    def __init__(self, value: object) -> None:
        self.value = value
        self.calls = 0

    def item(self) -> object:
        self.calls += 1
        return self.value


def test_normalize_metrics_preserves_raw_names_and_json_scalar_types() -> None:
    scalar = _Scalar(0.375)
    raw = {
        _RAW_NAME: scalar,
        "iterations": 8,
        "converged": True,
        "note": "raw",
        "optional": None,
    }

    normalized = evaluation.normalize_metrics(raw)

    assert normalized == {
        _RAW_NAME: 0.375,
        "iterations": 8,
        "converged": True,
        "note": "raw",
        "optional": None,
    }
    assert scalar.calls == 1
    assert "car_3d_ap40_moderate_strict" not in normalized


@pytest.mark.parametrize(
    "raw, error_type",
    [
        ([('score', 1.0)], TypeError),
        ({1: 1.0}, TypeError),
        ({"score": [1.0]}, TypeError),
        ({"score": _Scalar([1.0])}, TypeError),
        ({"score": float("nan")}, ValueError),
        ({"score": float("inf")}, ValueError),
        ({"score": _Scalar(float("-inf"))}, ValueError),
    ],
)
def test_normalize_metrics_rejects_non_strict_values(
    raw: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        evaluation.normalize_metrics(raw)  # type: ignore[arg-type]


def test_success_verifies_then_uses_exact_lazy_boundary_and_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code_provenance: CodeProvenance,
) -> None:
    run = _completed_run(tmp_path)
    _stub_evidence(monkeypatch, code_provenance)
    events, configs = _fake_mmengine(monkeypatch)
    real_load = evaluation.load_run
    real_verify = evaluation.verify_checkpoint

    def load(path):
        events.append("load")
        return real_load(path)

    roots: list[Path | None] = []

    def verify(artifact, *, root=None):
        events.append("verify")
        roots.append(root)
        return real_verify(artifact, root=root)

    monkeypatch.setattr(evaluation, "load_run", load)
    monkeypatch.setattr(evaluation, "verify_checkpoint", verify)

    record = evaluation.evaluate_run(run)

    assert record.successful is True
    assert record.provenance == code_provenance
    assert record.environment == _environment(torch_observed=True)
    assert record.binding.run_id == run.run_id
    assert record.binding.config_sha256 == run.manifest.config.sha256
    assert record.binding.checkpoint_sha256 == run.selected_checkpoint.sha256
    assert record.payload["kind"] == "evaluation"
    assert record.payload["semantic_partition"] == "KITTI validation"
    assert record.payload["framework_key"] == "test_dataloader"
    assert record.payload["metrics"][_RAW_NAME] == 0.625  # type: ignore[index]
    assert dict(record.payload["metric_profile"]) == {
        "id": evaluation.RAW_METRIC_PROFILE_ID,
        "version": 1,
        "key": evaluation.RAW_METRIC_PROFILE_KEY,
    }
    assert roots == [None, None]
    assert events == [
        "load",
        "verify",
        "compat",
        "register:True",
        f"config:{run.paths.config}",
        "custom:example_plugin",
        "load",
        "verify",
        "runner",
        "test",
    ]

    config = configs[0]
    assert config["load_from"] == run.selected_checkpoint.path
    assert config["resume"] is False
    assert config["launcher"] == "none"
    assert not Path(config["work_dir"]).exists()
    assert load_result(run, "evaluation", record.result_id) == record


def test_relative_checkpoint_verification_is_rooted_at_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _completed_run(tmp_path)
    local_path = run.paths.training / "best_local.pth"
    _write_checkpoint(local_path)
    artifact = identify_checkpoint(local_path, root=run.paths.root)
    real_verify = evaluation.verify_checkpoint
    observed_roots: list[Path | None] = []

    def verify(candidate, *, root=None):
        observed_roots.append(root)
        return real_verify(candidate, root=root)

    monkeypatch.setattr(evaluation, "verify_checkpoint", verify)

    path = evaluation._checkpoint_path(run, artifact)

    assert observed_roots == [run.paths.root]
    assert path == local_path


def test_checkpoint_tamper_is_published_as_failure_before_heavy_imports(
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
        evaluation,
        "importlib",
        types.SimpleNamespace(import_module=forbidden_import),
    )

    record = evaluation.evaluate_run(run)

    assert record.successful is False
    assert record.failure is not None
    assert record.failure.error_type == "ValueError"
    assert "checkpoint identity mismatch" in record.failure.message
    assert record.payload["metrics"] == {}
    assert record.provenance == code_provenance
    assert len(list_results(run, "evaluation")) == 1


def test_config_tamper_fails_reload_before_runner_or_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _completed_run(tmp_path)
    run.paths.config.write_text("model = dict(tampered=True)\n", encoding="utf-8")

    monkeypatch.setattr(
        evaluation,
        "importlib",
        types.SimpleNamespace(
            import_module=lambda name: pytest.fail(f"unexpected heavy import {name}")
        ),
    )

    with pytest.raises(ValueError, match="config bytes"):
        evaluation.evaluate_run(run)

    assert tuple(run.paths.evaluation.iterdir()) == ()


def test_config_change_during_runtime_load_is_recorded_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code_provenance: CodeProvenance,
) -> None:
    run = _completed_run(tmp_path)
    _stub_evidence(monkeypatch, code_provenance)
    events, _ = _fake_mmengine(monkeypatch)
    real_import = evaluation.importlib.import_module

    class MutatingConfig(dict):
        @classmethod
        def fromfile(cls, path: str):
            run.paths.config.write_text("tampered = True\n", encoding="utf-8")
            return cls()

    def import_module(name: str):
        if name == "mmengine.config":
            return types.SimpleNamespace(Config=MutatingConfig)
        return real_import(name)

    monkeypatch.setattr(
        evaluation,
        "importlib",
        types.SimpleNamespace(import_module=import_module),
    )

    record = evaluation.evaluate_run(run)

    assert record.status == "failed"
    assert record.failure is not None
    assert record.failure.error_type == "ValueError"
    assert "config bytes" in record.failure.message
    assert "runner" not in events


def test_runner_failure_becomes_fresh_immutable_failure_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code_provenance: CodeProvenance,
) -> None:
    run = _completed_run(tmp_path)
    _stub_evidence(monkeypatch, code_provenance)
    _fake_mmengine(monkeypatch, test_error=RuntimeError("model test failed"))

    first = evaluation.evaluate_run(run)
    second = evaluation.evaluate_run(run)

    assert first.result_id != second.result_id
    assert first.status == second.status == "failed"
    assert first.failure is not None
    assert first.failure.error_type == "RuntimeError"
    assert first.failure.message == "model test failed"
    assert "in test" in (first.failure.traceback or "")
    assert len(list_results(run, "evaluation")) == 2
    with pytest.raises(TypeError):
        first.payload["metrics"]["new"] = 1  # type: ignore[index]


def test_repeated_success_never_overwrites_previous_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code_provenance: CodeProvenance,
) -> None:
    run = _completed_run(tmp_path)
    _stub_evidence(monkeypatch, code_provenance)
    _fake_mmengine(monkeypatch, metrics={"raw/score": _Scalar(0.5)})

    first = evaluation.evaluate_run(run)
    second = evaluation.evaluate_run(run)

    assert first.result_id != second.result_id
    assert first.successful and second.successful
    assert {item.result_id for item in list_results(run, "evaluation")} == {
        first.result_id,
        second.result_id,
    }
    assert load_result(run, "evaluation", first.result_id) == first


def test_publication_error_is_not_hidden_by_execution_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code_provenance: CodeProvenance,
) -> None:
    run = _completed_run(tmp_path)
    _stub_evidence(monkeypatch, code_provenance)
    _fake_mmengine(monkeypatch, test_error=RuntimeError("primary failure"))

    def fail_publication(run, record):
        raise OSError("publication failed")

    monkeypatch.setattr(evaluation, "publish_result", fail_publication)

    with pytest.raises(OSError, match="publication failed") as captured:
        evaluation.evaluate_run(run)

    assert isinstance(captured.value.__context__, RuntimeError)
    assert tuple(run.paths.evaluation.iterdir()) == ()


def test_cuda_cleanup_is_best_effort_without_importing_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def empty_cache() -> None:
        calls.append("empty")
        raise RuntimeError("cleanup failed")

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(empty_cache=empty_cache)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    evaluation._cleanup_cuda()

    assert calls == ["empty"]


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
        (REPOSITORY_ROOT / "research/src/lidar_model_selection/evaluation.py",),
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
        slug="pending-evaluation",
        config_bytes=config,
        dataset=dataset,
        target_epoch=4,
        code_provenance=code_provenance,
        environment=environment,
        training_compatibility=compatibility,
    )
    monkeypatch.setattr(
        evaluation,
        "importlib",
        types.SimpleNamespace(
            import_module=lambda name: pytest.fail(f"unexpected heavy import {name}")
        ),
    )

    with pytest.raises(ValueError, match="completed run"):
        evaluation.evaluate_run(run)

    assert tuple(run.paths.evaluation.iterdir()) == ()


def test_evaluation_module_and_cli_have_no_eager_ml_imports() -> None:
    source = REPOSITORY_ROOT / "research" / "src"
    script = f"""
import importlib.util
import sys
before = set(sys.modules)
import lidar_model_selection.evaluation
spec = importlib.util.spec_from_file_location(
    'evaluate_tool_test',
    {str(EVALUATE_TOOL)!r},
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
forbidden = {{'torch', 'mmengine', 'mmcv', 'mmdet', 'mmdet3d'}}
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


def _load_evaluate_tool():
    spec = importlib.util.spec_from_file_location(
        "evaluate_tool_under_test",
        EVALUATE_TOOL,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_accepts_only_one_run_id_and_optional_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_evaluate_tool()
    run_id = "20260818T120000Z-evaluation-run-" + "1" * 24
    observed: list[Path] = []
    record = types.SimpleNamespace(
        binding=types.SimpleNamespace(run_id=run_id),
        result_id="20260818T120000000000Z-evaluation-" + "2" * 24,
        status="succeeded",
        successful=True,
        failure=None,
    )

    monkeypatch.setattr(tool, "DEFAULT_RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(
        tool,
        "evaluate_run",
        lambda path: observed.append(path) or record,
    )

    assert tool.main(["--run", run_id, "--gpu", "GPU-deadbeef"]) == 0
    assert observed == [tmp_path / "runs" / run_id]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-deadbeef"

    parser = tool.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["config.py", "checkpoint.pth"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--all"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--run", "not-a-run-id"])
