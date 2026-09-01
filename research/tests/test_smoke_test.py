from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
import types
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import pytest

import lidar_model_selection.evaluation as evaluation
import lidar_model_selection.results as results_module
from lidar_model_selection.checkpoints import TrainingOutputs, identify_checkpoint
from lidar_model_selection.provenance import (
    CodeProvenance,
    EnvironmentInfo,
    capture_code_provenance,
)
from lidar_model_selection.results import (
    list_results,
    load_result,
    select_result,
)
from lidar_model_selection.runs import (
    Run,
    TrainingState,
    build_dataset_identity,
    create_run,
    load_run,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SMOKE_TOOL = REPOSITORY_ROOT / "research" / "tools" / "smoke_test.py"


def _load_tool():
    specification = importlib.util.spec_from_file_location(
        "lidar_smoke_test_tool",
        SMOKE_TOOL,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _write_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", b"smoke checkpoint")
        archive.writestr("archive/version", b"3")


def _completed_run(tmp_path: Path):
    final = tmp_path / "external" / "epoch_2.pth"
    selected = tmp_path / "external" / "best_score.pth"
    _write_checkpoint(final)
    _write_checkpoint(selected)
    outputs = TrainingOutputs(
        final_checkpoint=identify_checkpoint(final),
        selected_checkpoint=identify_checkpoint(selected),
    )
    dataset = build_dataset_identity(
        name="KITTI",
        version=None,
        root_reference="historical:kitti",
        semantic_partition="KITTI validation",
        framework_key="test_dataloader",
        annotation_files=None,
        class_names=("Car",),
        tasks={"3d_detection": ("Car",)},
    )
    return create_run(
        tmp_path / "runs",
        slug="smoke-run",
        config_bytes=b"model = dict()\n",
        dataset=dataset,
        target_epoch=2,
        origin="historical_import",
        training_state=TrainingState("completed", (), outputs),
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
        ("research/src/lidar_model_selection/evaluation.py",),
    )


def _stub_evidence(
    monkeypatch: pytest.MonkeyPatch,
    code_provenance: CodeProvenance,
) -> None:
    monkeypatch.setattr(
        evaluation,
        "_capture_smoke_initial_evidence",
        lambda: (code_provenance, _environment(torch_observed=False)),
    )
    monkeypatch.setattr(
        evaluation,
        "_capture_execution_environment",
        lambda: _environment(torch_observed=True),
    )


def _outputs() -> dict[str, object]:
    return {
        "loss_keys": ["loss_centerpoint"],
        "total_loss": 1.25,
        "finite_gradient_tensors": 3,
        "prediction_boxes_shape": [2, 7],
        "prediction_scores_shape": [2],
        "prediction_labels_shape": [2],
    }


def _stub_execution(
    monkeypatch: pytest.MonkeyPatch,
    *,
    error: BaseException | None = None,
) -> None:
    def execute(run: Run, checkpoint_path: Path, runtime: dict[str, object]):
        runtime["selected_cuda_device_index"] = 0
        runtime["selected_cuda_device_name"] = "Fake GPU"
        if error is not None:
            raise error
        return _outputs()

    monkeypatch.setattr(evaluation, "_execute_smoke", execute)


def test_selected_checkpoint_is_run_bound_and_tamper_detected(
    tmp_path: Path,
) -> None:
    run = _completed_run(tmp_path)
    selected = run.selected_checkpoint
    assert selected is not None

    assert evaluation._checkpoint_path(run, selected) == Path(selected.path)
    Path(selected.path).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="identity mismatch"):
        evaluation._checkpoint_path(run, selected)


def test_smoke_execution_loads_selected_checkpoint_after_final_recheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code_provenance: CodeProvenance,
) -> None:
    run = _completed_run(tmp_path)
    _stub_evidence(monkeypatch, code_provenance)
    events: list[str] = []

    class Tensor:
        def mean(self):
            return self

    class Total:
        def sum(self):
            return self

        def backward(self):
            events.append("backward")

        def item(self):
            return 1.25

    class Finite:
        def all(self):
            return True

        def __bool__(self):
            return True

    class Parameter:
        grad = object()

    class Model:
        def cuda(self):
            return self

        def train(self):
            events.append("train")

        def eval(self):
            events.append("eval")

        def data_preprocessor(self, batch, *, training):
            return {}

        def __call__(self, **options):
            if options["mode"] == "loss":
                return {"loss_centerpoint": Tensor()}
            return ["prediction"]

        def parameters(self):
            return (Parameter(),)

        def zero_grad(self, *, set_to_none):
            assert set_to_none is True

    model = Model()
    fake_torch = types.SimpleNamespace(
        Tensor=Tensor,
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_name=lambda index: "Fake GPU",
            synchronize=lambda: events.append("synchronize"),
        ),
        stack=lambda values: Total(),
        isfinite=lambda value: Finite(),
        no_grad=lambda: nullcontext(),
    )

    class Config(dict):
        @classmethod
        def fromfile(cls, path):
            events.append("config")
            return cls(
                model={},
                train_dataloader={"dataset": {"split": "train"}},
                val_dataloader={"dataset": {"split": "validation"}},
            )

    class DatasetRegistry:
        @staticmethod
        def build(config):
            return [object()]

    class ModelRegistry:
        @staticmethod
        def build(config):
            events.append("model")
            return model

    modules = {
        "lidar_model_selection.compat.kitti_evaluator": types.SimpleNamespace(
            install=lambda: events.append("compat")
        ),
        "torch": fake_torch,
        "mmdet3d.utils": types.SimpleNamespace(
            register_all_modules=lambda **options: events.append("register")
        ),
        "mmengine.config": types.SimpleNamespace(Config=Config),
        "mmdet3d.registry": types.SimpleNamespace(
            DATASETS=DatasetRegistry,
            MODELS=ModelRegistry,
        ),
        "mmengine.dataset": types.SimpleNamespace(
            pseudo_collate=lambda samples: samples
        ),
        "mmengine.runner": types.SimpleNamespace(
            load_checkpoint=lambda candidate, path, **options: events.append(
                f"checkpoint:{path}"
            )
        ),
    }
    monkeypatch.setattr(
        evaluation,
        "importlib",
        types.SimpleNamespace(import_module=lambda name: modules[name]),
    )
    monkeypatch.setattr(evaluation, "_first_valid_sample", lambda dataset: {})
    monkeypatch.setattr(
        evaluation,
        "_validate_training_sample",
        lambda sample: None,
    )
    monkeypatch.setattr(
        evaluation,
        "_validate_predictions",
        lambda predictions: (
            types.SimpleNamespace(shape=(2, 7)),
            types.SimpleNamespace(shape=(2,)),
            types.SimpleNamespace(shape=(2,)),
        ),
    )
    original_recheck = evaluation._require_execution_inputs_unchanged

    def recheck(current, checkpoint):
        events.append("recheck")
        return original_recheck(current, checkpoint)

    monkeypatch.setattr(
        evaluation,
        "_require_execution_inputs_unchanged",
        recheck,
    )

    record = evaluation.smoke_run(run)

    checkpoint_event = f"checkpoint:{run.selected_checkpoint.path}"
    assert events.count("recheck") == 3
    assert events.index("model") < events.index("recheck", events.index("model"))
    assert events.index("recheck", events.index("model")) < events.index(
        checkpoint_event
    )
    assert record.successful
    assert record.binding.run_id == run.run_id
    assert record.binding.checkpoint_sha256 == run.selected_checkpoint.sha256
    outputs = record.payload["outputs"]
    assert outputs["finite_gradient_tensors"] == 1  # type: ignore[index]
    assert outputs["prediction_boxes_shape"] == (2, 7)  # type: ignore[index]


def test_success_persists_complete_run_bound_smoke_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code_provenance: CodeProvenance,
) -> None:
    run = _completed_run(tmp_path)
    _stub_evidence(monkeypatch, code_provenance)
    _stub_execution(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")

    record = evaluation.smoke_run(run)

    assert record.successful
    assert record.result_type == "smoke"
    assert record.provenance == code_provenance
    assert record.environment == _environment(torch_observed=True)
    assert record.binding.run_id == run.run_id
    assert record.binding.config_sha256 == run.manifest.config.sha256
    assert record.binding.checkpoint_sha256 == run.selected_checkpoint.sha256
    document = record.to_dict()
    payload = document["payload"]
    assert payload["kind"] == "smoke"  # type: ignore[index]
    assert payload["smoke_schema_version"] == 1  # type: ignore[index]
    assert payload["model"] == {  # type: ignore[index]
        "slug": run.manifest.slug,
        "config": run.manifest.config.to_dict(),
    }
    assert payload["dataset"] == run.manifest.dataset.to_dict()  # type: ignore[index]
    assert payload["checkpoint"] == run.selected_checkpoint.to_dict()  # type: ignore[union-attr,index]
    assert payload["runtime"] == {  # type: ignore[index]
        "cuda_visible_devices": "2",
        "selected_cuda_device_index": 0,
        "selected_cuda_device_name": "Fake GPU",
    }
    assert payload["outputs"] == _outputs()  # type: ignore[index]
    evaluation.validate_smoke_result(run, record)

    destination = run.paths.smoke / record.result_id
    assert set(destination.iterdir()) == {destination / "result.json"}
    assert load_result(run, "smoke", record.result_id) == record
    assert evaluation.smoke_stage_status(run) == "successful"
    assert not list(run.paths.smoke.glob(f".{record.result_id}.staging-*"))


def test_failed_attempt_persists_failure_and_retry_gets_unique_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code_provenance: CodeProvenance,
) -> None:
    run = _completed_run(tmp_path)
    _stub_evidence(monkeypatch, code_provenance)
    _stub_execution(monkeypatch, error=RuntimeError("backward failed"))

    first = evaluation.smoke_run(run)
    second = evaluation.smoke_run(run)

    assert first.result_id != second.result_id
    assert first.status == second.status == "failed"
    assert first.failure is not None
    assert first.failure.error_type == "RuntimeError"
    assert first.failure.message == "backward failed"
    assert "in execute" in (first.failure.traceback or "")
    assert first.payload["outputs"] == {}
    assert {record.result_id for record in list_results(run, "smoke")} == {
        first.result_id,
        second.result_id,
    }
    assert evaluation.smoke_stage_status(run) == "failed"


def test_base_exception_is_persisted_before_it_is_reraised(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code_provenance: CodeProvenance,
) -> None:
    run = _completed_run(tmp_path)
    _stub_evidence(monkeypatch, code_provenance)
    _stub_execution(monkeypatch, error=KeyboardInterrupt("interrupted"))

    with pytest.raises(KeyboardInterrupt, match="interrupted"):
        evaluation.smoke_run(run)

    records = list_results(run, "smoke")
    assert len(records) == 1
    assert records[0].status == "failed"
    assert records[0].failure is not None
    assert records[0].failure.error_type == "KeyboardInterrupt"
    assert records[0].failure.message == "interrupted"


def test_repeated_success_never_overwrites_and_requires_explicit_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code_provenance: CodeProvenance,
) -> None:
    run = _completed_run(tmp_path)
    _stub_evidence(monkeypatch, code_provenance)
    _stub_execution(monkeypatch)

    first = evaluation.smoke_run(run)
    first_path = run.paths.smoke / first.result_id / "result.json"
    first_bytes = first_path.read_bytes()
    second = evaluation.smoke_run(run)

    assert first.result_id != second.result_id
    assert first_path.read_bytes() == first_bytes
    records = list_results(run, "smoke")
    assert len(records) == 2
    assert evaluation.smoke_stage_status(run) == "successful"
    with pytest.raises(ValueError, match="multiple successful"):
        select_result(records)
    assert select_result(records, result_id=first.result_id) == first


def test_failed_atomic_publication_leaves_no_partial_smoke_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code_provenance: CodeProvenance,
) -> None:
    run = _completed_run(tmp_path)
    _stub_evidence(monkeypatch, code_provenance)
    _stub_execution(monkeypatch)

    def fail_publication(staging: Path, destination: Path) -> None:
        assert (staging / "result.json").is_file()
        raise OSError("publication failed")

    monkeypatch.setattr(
        results_module,
        "publish_directory_exclusive",
        fail_publication,
    )

    with pytest.raises(OSError, match="publication failed"):
        evaluation.smoke_run(run)

    assert tuple(run.paths.smoke.iterdir()) == ()


def test_checkpoint_tamper_is_immutably_recorded_before_smoke_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code_provenance: CodeProvenance,
) -> None:
    run = _completed_run(tmp_path)
    selected = run.selected_checkpoint
    assert selected is not None
    Path(selected.path).write_bytes(b"tampered")
    _stub_evidence(monkeypatch, code_provenance)

    def forbidden_execution(*args, **kwargs):
        raise AssertionError("smoke execution must not start after tampering")

    monkeypatch.setattr(evaluation, "_execute_smoke", forbidden_execution)

    record = evaluation.smoke_run(run)

    assert record.status == "failed"
    assert record.failure is not None
    assert record.failure.error_type == "ValueError"
    assert "checkpoint identity mismatch" in record.failure.message
    assert record.to_dict()["payload"]["checkpoint"] == selected.to_dict()  # type: ignore[index]
    assert load_result(run, "smoke", record.result_id) == record


def test_smoke_schema_rejects_model_dataset_or_checkpoint_rebinding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code_provenance: CodeProvenance,
) -> None:
    run = _completed_run(tmp_path)
    _stub_evidence(monkeypatch, code_provenance)
    _stub_execution(monkeypatch)
    record = evaluation.smoke_run(run)
    payload = record.to_dict()["payload"]

    changed_model = dict(payload)  # type: ignore[arg-type]
    changed_model["model"] = {
        "slug": "another-model",
        "config": run.manifest.config.to_dict(),
    }
    with pytest.raises(ValueError, match="model/config identity"):
        evaluation.validate_smoke_result(
            run,
            replace(record, payload=changed_model),
        )

    changed_dataset = dict(payload)  # type: ignore[arg-type]
    changed_dataset["dataset"] = {"identity_sha256": "0" * 64}
    with pytest.raises(ValueError, match="dataset identity"):
        evaluation.validate_smoke_result(
            run,
            replace(record, payload=changed_dataset),
        )

    changed_checkpoint = dict(payload)  # type: ignore[arg-type]
    checkpoint = dict(changed_checkpoint["checkpoint"])  # type: ignore[arg-type]
    checkpoint["sha256"] = "0" * 64
    changed_checkpoint["checkpoint"] = checkpoint
    with pytest.raises(ValueError, match="checkpoint identity"):
        evaluation.validate_smoke_result(
            run,
            replace(record, payload=changed_checkpoint),
        )


def test_invalid_gpu_selection_is_persisted_when_initial_evidence_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code_provenance: CodeProvenance,
) -> None:
    run = _completed_run(tmp_path)
    _stub_evidence(monkeypatch, code_provenance)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    record = evaluation.smoke_run(run)

    assert record.status == "failed"
    assert record.failure is not None
    assert record.failure.error_type == "ValueError"
    assert "CUDA_VISIBLE_DEVICES" in record.failure.message
    assert record.to_dict()["payload"]["runtime"]["cuda_visible_devices"] == ""  # type: ignore[index]
    assert load_result(run, "smoke", record.result_id) == record


def test_evidence_capture_failure_is_not_published_without_fabricated_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _completed_run(tmp_path)

    def fail_capture():
        raise OSError("workspace evidence unavailable")

    monkeypatch.setattr(evaluation, "_capture_smoke_initial_evidence", fail_capture)

    with pytest.raises(OSError, match="workspace evidence unavailable"):
        evaluation.smoke_run(run)

    assert tuple(run.paths.smoke.iterdir()) == ()


def test_historical_run_without_smoke_directory_loads_and_is_missing(
    tmp_path: Path,
) -> None:
    run = _completed_run(tmp_path)
    run.paths.smoke.rmdir()

    historical = load_run(run.paths.root)

    assert not historical.paths.smoke.exists()
    assert list_results(historical, "smoke") == ()
    assert evaluation.smoke_stage_status(historical) == "missing"
    assert not historical.paths.smoke.exists()


def test_historical_smoke_directory_is_created_safely_on_first_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code_provenance: CodeProvenance,
) -> None:
    run = _completed_run(tmp_path)
    run.paths.smoke.rmdir()
    historical = load_run(run.paths.root)
    _stub_evidence(monkeypatch, code_provenance)
    _stub_execution(monkeypatch)

    record = evaluation.smoke_run(historical)

    assert record.successful
    assert historical.paths.smoke.is_dir()
    assert load_result(historical, "smoke", record.result_id) == record


def test_concurrent_smoke_attempts_publish_distinct_complete_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code_provenance: CodeProvenance,
) -> None:
    run = _completed_run(tmp_path)
    run.paths.smoke.rmdir()
    _stub_evidence(monkeypatch, code_provenance)
    _stub_execution(monkeypatch)

    with ThreadPoolExecutor(max_workers=4) as executor:
        records = tuple(executor.map(lambda _: evaluation.smoke_run(run), range(4)))

    assert len({record.result_id for record in records}) == 4
    assert all(record.successful for record in records)
    published = list_results(run, "smoke")
    assert {record.result_id for record in published} == {
        record.result_id for record in records
    }
    assert not list(run.paths.smoke.glob(".*.staging-*"))


def test_cli_reports_result_id_full_path_and_terminal_exit_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    code_provenance: CodeProvenance,
) -> None:
    run = _completed_run(tmp_path)
    _stub_evidence(monkeypatch, code_provenance)
    _stub_execution(monkeypatch)
    succeeded = evaluation.smoke_run(run)
    smoke = _load_tool()
    monkeypatch.setattr(smoke, "DEFAULT_RUNS_ROOT", run.paths.root.parent)
    monkeypatch.setattr(smoke, "smoke_run", lambda path: succeeded)

    assert smoke.main(["--run", run.run_id, "--gpu", "2"]) == 0
    captured = capsys.readouterr()
    result_path = run.paths.smoke / succeeded.result_id / "result.json"
    assert f"result ID: {succeeded.result_id}" in captured.out
    assert f"result path: {result_path}" in captured.out
    assert "result status: succeeded" in captured.out
    assert "CenterPoint smoke test: PASS" in captured.out
    assert captured.err == ""

    _stub_execution(monkeypatch, error=RuntimeError("prediction failed"))
    failed = evaluation.smoke_run(run)
    monkeypatch.setattr(smoke, "smoke_run", lambda path: failed)
    assert smoke.main(["--run", run.run_id, "--gpu", "2"]) == 1
    captured = capsys.readouterr()
    failed_path = run.paths.smoke / failed.result_id / "result.json"
    assert f"result ID: {failed.result_id}" in captured.out
    assert f"result path: {failed_path}" in captured.out
    assert "result status: failed" in captured.out
    assert "RuntimeError: prediction failed" in captured.err


def test_cli_is_run_only_lightweight_and_python310_compatible() -> None:
    smoke = _load_tool()
    destinations = {action.dest for action in smoke.build_parser()._actions}
    assert "run_id" in destinations
    assert "config" not in destinations
    assert "checkpoint" not in destinations
    ast.parse(SMOKE_TOOL.read_text(encoding="utf-8"), feature_version=(3, 10))

    environment = os.environ.copy()
    source = os.fspath(REPOSITORY_ROOT / "research" / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source, environment.get("PYTHONPATH", "")))
    )
    completed = subprocess.run(
        [sys.executable, os.fspath(SMOKE_TOOL), "--help"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util, pathlib, sys; "
                f"p=pathlib.Path({os.fspath(SMOKE_TOOL)!r}); "
                "s=importlib.util.spec_from_file_location('smoke_probe', p); "
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                "assert 'torch' not in sys.modules; "
                "assert 'mmengine' not in sys.modules; "
                "assert 'mmdet3d' not in sys.modules"
            ),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    assert probe.returncode == 0, probe.stderr
