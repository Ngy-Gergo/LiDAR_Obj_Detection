from __future__ import annotations

import argparse
import ast
import importlib
import os
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

import lidar_model_selection.playback.cli as playback_cli
import lidar_model_selection.playback.detector as detector_module
from lidar_model_selection.checkpoints import (
    TrainingOutputs,
    identify_checkpoint,
)
from lidar_model_selection.playback.detector import Mmdet3dDetector
from lidar_model_selection.playback.frame_source import LidarFrame
from lidar_model_selection.runs import (
    Run,
    TrainingState,
    build_dataset_identity,
    create_run,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "research" / "src"
_RUN_ID = "20260818T000000Z-pillar02-" + "a" * 24


def _write_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr("archive/data.pkl", b"structural checkpoint")
        archive.writestr("archive/version", b"3")


def _completed_run(tmp_path: Path) -> Run:
    final_path = tmp_path / "external" / "epoch_2.pth"
    selected_path = tmp_path / "external" / "best_score.pth"
    _write_checkpoint(final_path)
    _write_checkpoint(selected_path)
    outputs = TrainingOutputs(
        final_checkpoint=identify_checkpoint(final_path),
        selected_checkpoint=identify_checkpoint(selected_path),
    )
    dataset = build_dataset_identity(
        name="KITTI",
        version=None,
        root_reference="dataset:kitti-playback-test",
        semantic_partition="KITTI validation",
        framework_key="test_dataloader",
        annotation_files=None,
        class_names=("Car",),
        tasks={"3d_object_detection_7d": ("Car",)},
    )
    return create_run(
        tmp_path / "runs",
        slug="playback-run",
        config_bytes=b"model = dict(type='CenterPoint')\n",
        dataset=dataset,
        target_epoch=2,
        origin="historical_import",
        training_state=TrainingState(
            status="completed",
            attempts=(),
            outputs=outputs,
        ),
    )


class _FakeTensor:
    def __init__(self, values: list[object]) -> None:
        self._values = values
        self.ndim = 2 if values and isinstance(values[0], list) else 1
        self.shape = (
            (len(values), len(values[0]))
            if self.ndim == 2
            else (len(values),)
        )

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._values


def _fake_ml_modules(
    events: list[str],
    *,
    boxes: list[list[object]] | None = None,
    scores: list[object] | None = None,
    labels: list[object] | None = None,
):
    device = types.SimpleNamespace(type="cuda")

    class Model:
        def parameters(self):
            return iter((types.SimpleNamespace(device=device),))

    prediction = types.SimpleNamespace(
        bboxes_3d=types.SimpleNamespace(
            tensor=_FakeTensor(
                boxes
                or [
                    [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.1],
                    [5.0, 6.0, 7.0, 2.0, 2.0, 2.0, -0.2],
                ]
            )
        ),
        scores_3d=_FakeTensor(scores or [0.25, 0.75]),
        labels_3d=_FakeTensor(labels or [0, 1]),
    )

    def initialize(*, config: str, checkpoint: str, device: str):
        events.append(f"init:{config}:{checkpoint}:{device}")
        return Model()

    def infer(model: object, frame_path: str):
        events.append(f"infer:{frame_path}")
        return types.SimpleNamespace(pred_instances_3d=prediction), None

    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            synchronize=lambda selected: events.append(
                f"sync:{selected.type}"
            )
        )
    )
    apis = types.SimpleNamespace(
        init_model=initialize,
        inference_detector=infer,
    )
    return torch, apis


def test_detector_reloads_binding_reverifies_and_uses_selected_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _completed_run(tmp_path)
    events: list[str] = []
    torch, apis = _fake_ml_modules(events)
    real_load = detector_module.load_run
    real_verify = detector_module.verify_checkpoint
    real_binding = detector_module.binding_for_run
    roots: list[Path | None] = []

    def load(path):
        events.append("load")
        return real_load(path)

    def verify(artifact, *, root=None):
        events.append("verify")
        roots.append(root)
        return real_verify(artifact, root=root)

    def binding(candidate):
        events.append("binding")
        return real_binding(candidate)

    def import_module(name: str):
        events.append(f"import:{name}")
        return {"torch": torch, "mmdet3d.apis": apis}[name]

    monkeypatch.setattr(detector_module, "load_run", load)
    monkeypatch.setattr(detector_module, "verify_checkpoint", verify)
    monkeypatch.setattr(detector_module, "binding_for_run", binding)
    monkeypatch.setattr(detector_module.importlib, "import_module", import_module)

    detector = Mmdet3dDetector(run, device="cuda:0", score_threshold=0.5)

    selected_path = Path(run.selected_checkpoint.path)  # type: ignore[union-attr]
    assert roots == [None, None]
    assert events == [
        "load",
        "binding",
        "verify",
        "import:torch",
        "import:mmdet3d.apis",
        "load",
        "binding",
        "verify",
        f"init:{run.paths.config}:{selected_path}:cuda:0",
    ]

    frame = LidarFrame("000001", tmp_path / "000001.bin")
    result = detector.detect(frame)

    assert result.frame_id == "000001"
    assert len(result.detections) == 1
    assert result.detections[0].label == 1
    assert result.detections[0].score == 0.75
    assert events[-3:] == [
        "sync:cuda",
        f"infer:{frame.path}",
        "sync:cuda",
    ]


def test_checkpoint_tamper_fails_before_heavy_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _completed_run(tmp_path)
    assert run.selected_checkpoint is not None
    Path(run.selected_checkpoint.path).write_bytes(b"tampered")

    monkeypatch.setattr(
        detector_module.importlib,
        "import_module",
        lambda name: pytest.fail(f"unexpected heavy import: {name}"),
    )

    with pytest.raises(ValueError, match="checkpoint identity mismatch"):
        Mmdet3dDetector(run.paths.root)


def test_runtime_config_tamper_is_rejected_before_init_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _completed_run(tmp_path)
    events: list[str] = []
    torch, apis = _fake_ml_modules(events)

    def import_module(name: str):
        events.append(f"import:{name}")
        if name == "torch":
            run.paths.config.write_text("tampered = True\n", encoding="utf-8")
            return torch
        return apis

    monkeypatch.setattr(detector_module.importlib, "import_module", import_module)

    with pytest.raises(ValueError, match="config bytes"):
        Mmdet3dDetector(run)
    assert not any(event.startswith("init:") for event in events)


def test_detector_rejects_nonfinite_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _completed_run(tmp_path)
    events: list[str] = []
    torch, apis = _fake_ml_modules(
        events,
        scores=[float("nan"), 0.75],
    )
    monkeypatch.setattr(
        detector_module.importlib,
        "import_module",
        lambda name: {"torch": torch, "mmdet3d.apis": apis}[name],
    )
    detector = Mmdet3dDetector(run)

    with pytest.raises(ValueError, match="scores.*finite"):
        detector.detect(LidarFrame("frame", tmp_path / "frame.bin"))


def test_playback_parser_accepts_only_run_owned_model_selection() -> None:
    parser = playback_cli.build_parser()
    arguments = parser.parse_args(
        ["--run", _RUN_ID, "--input-dir", "/recording"]
    )

    assert arguments.run_id == _RUN_ID
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--run" in option_strings
    assert "--config" not in option_strings
    assert "--checkpoint" not in option_strings
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--config",
                "config.py",
                "--checkpoint",
                "model.pth",
                "--input-dir",
                "/recording",
            ]
        )


def test_playback_main_resolves_run_id_below_default_runs_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_directory = tmp_path / "frames"
    input_directory.mkdir()
    (input_directory / "000001.bin").write_bytes(b"")
    observed: dict[str, object] = {}

    class Detector:
        def __init__(self, **options: object) -> None:
            observed.update(options)

    class Pipeline:
        def __init__(self, detector: object) -> None:
            observed["detector"] = detector

        def run(self, frames):
            result = types.SimpleNamespace(
                frame_id=frames[0].frame_id,
                detections=(),
                inference_ms=1.5,
            )
            return (types.SimpleNamespace(result=result, total_ms=2.0),)

    monkeypatch.setattr(playback_cli, "DEFAULT_RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(playback_cli, "Mmdet3dDetector", Detector)
    monkeypatch.setattr(playback_cli, "SequentialDetectionPipeline", Pipeline)

    assert playback_cli.main(
        ["--run", _RUN_ID, "--input-dir", os.fspath(input_directory)]
    ) == 0
    assert observed["run"] == tmp_path / "runs" / _RUN_ID
    assert "Summary: frames=1" in capsys.readouterr().out


def test_playback_modules_do_not_import_ml_frameworks() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.fspath(SOURCE_ROOT)
    code = """
import sys
import lidar_model_selection.playback.detector
import lidar_model_selection.playback.cli
for prefix in ('torch', 'mmdet3d', 'mmcv', 'mmengine'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=REPOSITORY_ROOT,
        env=environment,
    )


def test_compatibility_cli_exposes_self_test_alone() -> None:
    module_name = "lidar_model_selection.compat.kitti_evaluator"
    previous_module = sys.modules.pop(module_name, None)
    previous_numpy = sys.modules.get("numpy")
    if previous_numpy is None:
        sys.modules["numpy"] = types.ModuleType("numpy")
    try:
        compatibility = importlib.import_module(module_name)
        parser = compatibility._build_argument_parser()
        subparsers = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        assert set(subparsers.choices) == {"self-test"}
        assert parser.parse_args(["self-test", "--device", "cpu"]).device == "cpu"
    finally:
        sys.modules.pop(module_name, None)
        if previous_module is not None:
            sys.modules[module_name] = previous_module
        if previous_numpy is None:
            sys.modules.pop("numpy", None)


def test_ros_parameters_and_python310_grammar_are_run_owned() -> None:
    paths = (
        Path(detector_module.__file__),
        Path(playback_cli.__file__),
        SOURCE_ROOT / "lidar_model_selection" / "playback" / "ros2_node.py",
        SOURCE_ROOT / "lidar_model_selection" / "compat" / "kitti_evaluator.py",
    )
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 10))

    ros_source = paths[2].read_text(encoding="utf-8")
    assert 'declare_parameter("run_id"' in ros_source
    assert '"runs_root"' in ros_source
    assert '"config_path"' not in ros_source
    assert '"checkpoint_path"' not in ros_source
