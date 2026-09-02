from __future__ import annotations

import threading
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import lidar_model_selection.playback.detector as detector_module
import lidar_model_selection.playback.model_registry as registry_module
from lidar_model_selection.checkpoints import CheckpointArtifact
from lidar_model_selection.playback.contracts import PointCloudFrame
from lidar_model_selection.playback.detector import FinalistDetector
from lidar_model_selection.playback.model_registry import (
    FINALIST_RUNS,
    FinalistModelIdentity,
    finalist_spec,
    resolve_finalist,
)
from lidar_model_selection.playback.results import (
    DetectionFrame,
    PlaybackErrorEvidence,
    empty_detection_arrays,
)
from lidar_model_selection.results import ResultBinding


_VOXEL_RUN_ID = "20260827T092043Z-voxel0075-e583a40f435e3071e0cbd6fc"
_VOXEL_SHA256 = (
    "5246b24bfe66a81df3bc6ca94db982f0188b33043f25771c40d02be4bcb22507"
)


class _FakeTensor:
    def __init__(self, values: object, events: list[str] | None = None) -> None:
        self.values = np.asarray(values)
        self._events = [] if events is None else events

    def detach(self):
        self._events.append("detach")
        return self

    def cpu(self):
        self._events.append("cpu")
        return self

    def numpy(self):
        self._events.append("numpy")
        return self.values


def _identity(tmp_path: Path) -> FinalistModelIdentity:
    binding = ResultBinding(
        run_id=_VOXEL_RUN_ID,
        config_sha256="a" * 64,
        checkpoint_sha256=_VOXEL_SHA256,
    )
    run_root = tmp_path / _VOXEL_RUN_ID
    return FinalistModelIdentity(
        model_alias="voxel0075",
        run=types.SimpleNamespace(),  # type: ignore[arg-type]
        binding=binding,
        config_path=run_root / "config.py",
        checkpoint_path=run_root / "training" / "selected.pth",
        checkpoint_reference="training/selected.pth",
        checkpoint_size_bytes=41_175_026,
    )


def _frame(
    points: np.ndarray,
    *,
    source_point_count: int | None = None,
    dropped_nonfinite_count: int = 0,
) -> PointCloudFrame:
    return PointCloudFrame(
        session_id="session",
        frame_index=4,
        timestamp_ns=1_000_000_000,
        storage_timestamp_ns=1_100_000_000,
        source_frame_id="lexus3/os_center",
        coordinate_frame="lidar",
        source_key="session_0.mcap:/lexus3/os_center/points#4",
        points=np.asarray(points, dtype=np.float32),
        source_point_count=(
            points.shape[0] + dropped_nonfinite_count
            if source_point_count is None
            else source_point_count
        ),
        dropped_nonfinite_count=dropped_nonfinite_count,
        decode_ms=1.25,
    )


def _fake_detector_modules(
    events: list[str],
    *,
    prediction: object,
    infer_hook=None,
):
    model_device = types.SimpleNamespace(type="cuda")

    class Model:
        def parameters(self):
            return iter((types.SimpleNamespace(device=model_device),))

    def init_model(*, config: str, checkpoint: str, device: str):
        events.append(f"init:{device}")
        return Model()

    def inference_detector(model: object, points: np.ndarray):
        events.append("infer")
        if infer_hook is not None:
            infer_hook(points)
        return types.SimpleNamespace(pred_instances_3d=prediction), {"points": points}

    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            synchronize=lambda selected: events.append(f"sync:{selected.type}")
        )
    )
    apis = types.SimpleNamespace(
        init_model=init_model,
        inference_detector=inference_detector,
    )
    return torch, apis


def _prediction(
    events: list[str],
    *,
    boxes: object = (
        (10.0, 1.0, -1.0, 4.0, 2.0, 1.5, 0.1),
        (11.0, 2.0, -1.0, 4.0, 2.0, 1.5, 0.2),
        (12.0, 3.0, -1.0, 4.0, 2.0, 1.5, 0.3),
    ),
    scores: object = (0.8, 0.1, 0.6),
    labels: object = (0, 1, 0),
):
    return types.SimpleNamespace(
        bboxes_3d=types.SimpleNamespace(tensor=_FakeTensor(boxes, events)),
        scores_3d=_FakeTensor(scores, events),
        labels_3d=_FakeTensor(labels, events),
    )


def _build_detector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    *,
    prediction: object | None = None,
    infer_hook=None,
) -> FinalistDetector:
    identity = _identity(tmp_path)
    monkeypatch.setattr(detector_module, "resolve_finalist", lambda *args: identity)
    selected_prediction = _prediction(events) if prediction is None else prediction
    torch, apis = _fake_detector_modules(
        events,
        prediction=selected_prediction,
        infer_hook=infer_hook,
    )
    monkeypatch.setattr(
        detector_module.importlib,
        "import_module",
        lambda name: {"torch": torch, "mmdet3d.apis": apis}[name],
    )
    return FinalistDetector("voxel0075", tmp_path, score_threshold=0.3)


def test_finalist_registry_is_closed_and_exact() -> None:
    assert FINALIST_RUNS == {
        "voxel0075": _VOXEL_RUN_ID,
        "pillar02": (
            "20260901T195416Z-pillar02-duration30-2720f37cf422c4e55bafd0a6"
        ),
    }
    with pytest.raises(ValueError, match="unknown finalist model alias"):
        resolve_finalist("arbitrary-run", "/runs")


def test_finalist_spec_is_lightweight_and_does_not_touch_run_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        registry_module,
        "load_run",
        lambda path: pytest.fail(f"unexpected run access: {path}"),
    )

    spec = finalist_spec("pillar02")

    assert spec.run_id == FINALIST_RUNS["pillar02"]
    assert spec.config_sha256 == (
        "ebed7d29b96cae0812ede9e572ffb1ba054d650ad62cb1c6c8895697fcb3a5d9"
    )
    assert spec.checkpoint_size_bytes == 34_256_294
    assert spec.checkpoint_sha256 == (
        "2606a3448cd9edc97b662b0ea8631ea828ed1ba7fe64578bba1f2f5b650c8cac"
    )


def test_registry_resolves_only_matching_canonical_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_root = tmp_path / "runs"
    run_root = runs_root / _VOXEL_RUN_ID
    selected = CheckpointArtifact(
        path="training/selected.pth",
        sha256=_VOXEL_SHA256,
        size_bytes=41_175_026,
        epoch=None,
        checkpoint_format="pytorch_zip",
        validation_profile="pytorch-zip-structural-v1",
    )
    loaded = types.SimpleNamespace(
        run_id=_VOXEL_RUN_ID,
        paths=types.SimpleNamespace(root=run_root, config=run_root / "config.py"),
        manifest=types.SimpleNamespace(
            origin="native",
            config=types.SimpleNamespace(
                sha256=(
                    "723749a5dc262ed1e57304092f12694d8f062c4a4158e2d65be685a47874c1b5"
                )
            ),
        ),
        selected_checkpoint=selected,
    )
    binding = ResultBinding(
        _VOXEL_RUN_ID,
        "723749a5dc262ed1e57304092f12694d8f062c4a4158e2d65be685a47874c1b5",
        _VOXEL_SHA256,
    )
    roots: list[Path] = []

    monkeypatch.setattr(registry_module, "load_run", lambda path: loaded)
    monkeypatch.setattr(registry_module, "binding_for_run", lambda run: binding)

    def verify(artifact, *, root):
        roots.append(root)
        return ()

    monkeypatch.setattr(registry_module, "verify_checkpoint", verify)

    identity = resolve_finalist("voxel0075", runs_root)

    assert identity.run_id == _VOXEL_RUN_ID
    assert identity.checkpoint_sha256 == _VOXEL_SHA256
    assert identity.checkpoint_size_bytes == 41_175_026
    assert identity.checkpoint_path == run_root / "training" / "selected.pth"
    assert roots == [run_root]

    loaded.manifest.config.sha256 = "b" * 64
    with pytest.raises(ValueError, match="config SHA-256 does not match"):
        resolve_finalist("voxel0075", runs_root)
    loaded.manifest.config.sha256 = (
        "723749a5dc262ed1e57304092f12694d8f062c4a4158e2d65be685a47874c1b5"
    )

    selected = types.SimpleNamespace(
        path="training/selected.pth",
        sha256=_VOXEL_SHA256,
        size_bytes=1,
    )
    loaded.selected_checkpoint = selected
    with pytest.raises(ValueError, match="size does not match"):
        resolve_finalist("voxel0075", runs_root)


def test_finalist_binding_failure_precedes_heavy_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        detector_module,
        "resolve_finalist",
        lambda *args: (_ for _ in ()).throw(
            ValueError("selected checkpoint binding failed")
        ),
    )
    monkeypatch.setattr(
        detector_module.importlib,
        "import_module",
        lambda name: pytest.fail(f"unexpected heavy import: {name}"),
    )

    with pytest.raises(ValueError, match="checkpoint binding failed"):
        FinalistDetector("voxel0075", tmp_path)


def test_in_memory_detection_is_run_bound_stable_timed_and_nonmutating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    frame = _frame(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.5],
                [-1.0, 0.0, 0.0, 0.25],
            ],
            dtype=np.float32,
        )
    )
    original = frame.points.copy()

    def mutate_private_input(points: np.ndarray) -> None:
        assert points is not frame.points
        assert points.flags.writeable
        points[0, 0] = 999.0

    detector = _build_detector(
        tmp_path,
        monkeypatch,
        events,
        infer_hook=mutate_private_input,
    )
    real_result = detector._result

    def materialize_detection_frame(*args, **kwargs):
        events.append("detection_frame")
        return real_result(*args, **kwargs)

    monkeypatch.setattr(detector, "_result", materialize_detection_frame)
    times = iter((10.0, 10.0125))

    def clock() -> float:
        events.append("clock")
        return next(times)

    monkeypatch.setattr(detector_module, "perf_counter", clock)
    result = detector.detect(frame)

    np.testing.assert_array_equal(frame.points, original)
    assert result.status == "success"
    assert result.model_alias == "voxel0075"
    assert result.run_id == _VOXEL_RUN_ID
    assert result.checkpoint_sha256 == _VOXEL_SHA256
    assert result.source_point_count == 2
    assert result.normalized_point_count == 2
    assert result.in_range_point_count == 1
    assert result.detection_count == 2
    np.testing.assert_array_equal(result.scores, np.array([0.8, 0.6], np.float32))
    np.testing.assert_array_equal(result.boxes[:, 0], np.array([10.0, 12.0]))
    assert not result.boxes.flags.writeable
    assert not result.scores.flags.writeable
    assert not result.labels.flags.writeable
    assert result.detector_ms == pytest.approx(12.5)
    assert result.frame_processing_ms == pytest.approx(13.75)
    assert events.count("sync:cuda") == 2
    assert events.index("infer") < events.index("detach")
    assert events[-3:] == ["sync:cuda", "detection_frame", "clock"]


@pytest.mark.parametrize(
    ("points", "source_count", "dropped", "status"),
    (
        (np.empty((0, 4), np.float32), 0, 0, "empty_source"),
        (
            np.empty((0, 4), np.float32),
            3,
            3,
            "empty_after_nonfinite_filter",
        ),
        (
            np.array([[-1.0, 0.0, 0.0, 0.2]], np.float32),
            1,
            0,
            "empty_after_range_filter",
        ),
    ),
)
def test_empty_inputs_bypass_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    points: np.ndarray,
    source_count: int,
    dropped: int,
    status: str,
) -> None:
    events: list[str] = []
    detector = _build_detector(tmp_path, monkeypatch, events)
    result = detector.detect(
        _frame(
            points,
            source_point_count=source_count,
            dropped_nonfinite_count=dropped,
        )
    )

    assert result.status == status
    assert result.detection_count == 0
    assert result.detector_ms >= 0.0
    assert "infer" not in events
    assert events.count("sync:cuda") == 2


def test_one_in_range_point_is_not_rejected_as_too_sparse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    detector = _build_detector(tmp_path, monkeypatch, events)

    result = detector.detect(_frame(np.array([[1.0, 0.0, 0.0, 0.2]])))

    assert result.status == "success"
    assert events.count("infer") == 1


def test_one_initialized_finalist_model_is_reused_serially(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    detector = _build_detector(tmp_path, monkeypatch, events)
    frame = _frame(np.array([[1.0, 0.0, 0.0, 0.2]], np.float32))

    detector.detect(frame)
    detector.detect(frame)

    assert sum(event.startswith("init:") for event in events) == 1
    assert events.count("infer") == 2


def test_finalist_detector_close_is_idempotent_and_rejects_future_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    detector = _build_detector(tmp_path, monkeypatch, events)
    detector.close()
    detector.close()

    assert detector._model is None
    with pytest.raises(RuntimeError, match="detector is closed"):
        detector.detect(_frame(np.array([[1.0, 0.0, 0.0, 0.2]], np.float32)))


def test_detector_rejects_incompatible_feature_identity_before_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    detector = _build_detector(tmp_path, monkeypatch, events)
    frame = _frame(np.array([[1.0, 0.0, 0.0, 0.2]]))

    with pytest.raises(ValueError, match="feature_profile.*compatible"):
        detector.detect(replace(frame, feature_profile="raw_float32_intensity"))
    with pytest.raises(ValueError, match="reflectivity.*\\[0, 1\\]"):
        detector.detect(_frame(np.array([[1.0, 0.0, 0.0, 2.0]])))
    assert "infer" not in events


@pytest.mark.parametrize(
    ("prediction", "message"),
    (
        (
            _prediction([], boxes=((1.0, 2.0, 3.0, 4.0, 5.0, 6.0),)),
            "shape \\(N, 7\\)",
        ),
        (
            _prediction(
                [],
                boxes=((1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0),),
                scores=(0.8,),
                labels=(0,),
            ),
            "dimensions.*positive",
        ),
        (
            _prediction(
                [],
                boxes=((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0),),
                scores=(1.1,),
                labels=(0,),
            ),
            "scores.*\\[0, 1\\]",
        ),
        (
            _prediction(
                [],
                boxes=((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0),),
                scores=(0.8,),
                labels=(1,),
            ),
            "labels.*class 0",
        ),
    ),
)
def test_invalid_detector_outputs_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prediction: object,
    message: str,
) -> None:
    events: list[str] = []
    detector = _build_detector(
        tmp_path,
        monkeypatch,
        events,
        prediction=prediction,
    )
    with pytest.raises(ValueError, match=message):
        detector.detect(_frame(np.array([[1.0, 0.0, 0.0, 0.2]])))


def test_concurrent_calls_are_rejected_without_waiting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    entered = threading.Event()
    release = threading.Event()
    failure: list[BaseException] = []

    def block(points: np.ndarray) -> None:
        entered.set()
        if not release.wait(timeout=5.0):
            raise AssertionError("test did not release fake inference")

    detector = _build_detector(
        tmp_path,
        monkeypatch,
        events,
        infer_hook=block,
    )
    frame = _frame(np.array([[1.0, 0.0, 0.0, 0.2]]))

    def first_call() -> None:
        try:
            detector.detect(frame)
        except BaseException as error:  # pragma: no cover - asserted below
            failure.append(error)

    worker = threading.Thread(target=first_call)
    worker.start()
    assert entered.wait(timeout=5.0)
    with pytest.raises(RuntimeError, match="concurrent"):
        detector.detect(frame)
    release.set()
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert failure == []


def test_detection_frame_allows_missing_source_evidence_only_for_errors() -> None:
    boxes, scores, labels = empty_detection_arrays()
    result = DetectionFrame(
        session_id="session",
        frame_index=3,
        timestamp_ns=None,
        storage_timestamp_ns=None,
        source_frame_id=None,
        coordinate_frame=None,
        source_key=None,
        model_alias="voxel0075",
        run_id=_VOXEL_RUN_ID,
        config_sha256="a" * 64,
        checkpoint_path="training/selected.pth",
        checkpoint_sha256=_VOXEL_SHA256,
        checkpoint_size_bytes=41_175_026,
        source_point_count=None,
        dropped_nonfinite_count=None,
        input_point_count=None,
        in_range_point_count=None,
        detection_count=0,
        status="frame_error",
        boxes=boxes,
        scores=scores,
        labels=labels,
        decode_ms=0.5,
        detector_ms=0.0,
        frame_processing_ms=0.5,
        errors=(
            PlaybackErrorEvidence(
                phase="decode",
                code="truncated_payload",
                message="payload ended before one declared point",
            ),
        ),
    )

    assert result.timestamp_ns is None
    assert result.source_point_count is None
    assert result.errors[0].code == "truncated_payload"
