from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
import types
import zipfile
from contextlib import nullcontext
from pathlib import Path

import pytest

import lidar_model_selection.evaluation as evaluation
from lidar_model_selection.checkpoints import TrainingOutputs, identify_checkpoint
from lidar_model_selection.runs import (
    TrainingState,
    build_dataset_identity,
    create_run,
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
) -> None:
    run = _completed_run(tmp_path)
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

    summary = evaluation.smoke_run(run)

    checkpoint_event = f"checkpoint:{run.selected_checkpoint.path}"
    assert events.count("recheck") == 3
    assert events.index("model") < events.index("recheck", events.index("model"))
    assert events.index("recheck", events.index("model")) < events.index(
        checkpoint_event
    )
    assert summary["run_id"] == run.run_id
    assert summary["checkpoint_sha256"] == run.selected_checkpoint.sha256
    assert summary["finite_gradient_tensors"] == 1
    assert summary["prediction_boxes_shape"] == [2, 7]


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
