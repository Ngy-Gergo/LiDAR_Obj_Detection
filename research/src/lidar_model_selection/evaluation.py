"""Run-bound MMEngine evaluation with immutable evidence publication."""

from __future__ import annotations

import copy
import gc
import importlib
import math
import os
import sys
import tempfile
import traceback
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpoints import CheckpointArtifact, verify_checkpoint
from .provenance import (
    CodeProvenance,
    EnvironmentInfo,
    capture_code_provenance,
    capture_environment,
)
from .results import (
    ResultFailure,
    ResultRecord,
    binding_for_run,
    create_result,
    list_results,
    publish_result,
)
from .runs import Run, load_run


__all__ = (
    "DEFAULT_REPOSITORY_ROOT",
    "DEFAULT_RUNS_ROOT",
    "RAW_METRIC_PROFILE_ID",
    "RAW_METRIC_PROFILE_VERSION",
    "RAW_METRIC_PROFILE_KEY",
    "SMOKE_SCHEMA_VERSION",
    "normalize_metrics",
    "evaluate_run",
    "smoke_run",
    "list_smoke_results",
    "smoke_stage_status",
    "validate_smoke_result",
)

DEFAULT_REPOSITORY_ROOT = Path(__file__).absolute().parents[3]
DEFAULT_RUNS_ROOT = DEFAULT_REPOSITORY_ROOT / "research" / "runs"

RAW_METRIC_PROFILE_ID = "mmengine_raw_scalar_metrics"
RAW_METRIC_PROFILE_VERSION = 1
RAW_METRIC_PROFILE_KEY = (
    f"{RAW_METRIC_PROFILE_ID}_v{RAW_METRIC_PROFILE_VERSION}"
)
SMOKE_SCHEMA_VERSION = 1

_CORE_PACKAGES = ("torch", "mmengine", "mmcv", "mmdet", "mmdet3d")
_PROVENANCE_SCOPES = (
    "research/src/lidar_model_selection/checkpoints.py",
    "research/src/lidar_model_selection/compat",
    "research/src/lidar_model_selection/evaluation.py",
    "research/src/lidar_model_selection/provenance.py",
    "research/src/lidar_model_selection/results.py",
    "research/src/lidar_model_selection/runs.py",
    "research/tools/evaluate.py",
)
_SMOKE_PROVENANCE_SCOPES = (
    "research/src/lidar_model_selection/checkpoints.py",
    "research/src/lidar_model_selection/compat",
    "research/src/lidar_model_selection/evaluation.py",
    "research/src/lidar_model_selection/provenance.py",
    "research/src/lidar_model_selection/results.py",
    "research/src/lidar_model_selection/runs.py",
    "research/tools/smoke_test.py",
)
_SMOKE_OUTPUT_KEYS = frozenset(
    {
        "loss_keys",
        "total_loss",
        "finite_gradient_tensors",
        "prediction_boxes_shape",
        "prediction_scores_shape",
        "prediction_labels_shape",
    }
)


def _timestamp() -> datetime:
    return datetime.now(timezone.utc)


def _value(container: object, key: str, default: object = None) -> Any:
    if isinstance(container, Mapping):
        return container.get(key, default)
    return getattr(container, key, default)


def _set_value(container: object, key: str, value: object) -> None:
    try:
        container[key] = value  # type: ignore[index]
    except (AttributeError, TypeError):
        setattr(container, key, value)


def normalize_metrics(raw_metrics: Mapping[str, object]) -> dict[str, object]:
    """Convert raw MMEngine metrics to strict JSON scalar values.

    Metric names are retained verbatim. Tensor- and NumPy-like scalar values
    may expose ``item()``; containers, non-finite floats, and non-string keys
    are rejected rather than projected into an evaluator-specific schema.
    """
    if not isinstance(raw_metrics, Mapping):
        raise TypeError("Runner.test() must return a mapping of metric names")

    metrics: dict[str, object] = {}
    for name, original in raw_metrics.items():
        if not isinstance(name, str):
            raise TypeError("MMEngine metric names must be strings")

        value = original
        if value is not None and not isinstance(
            value,
            (str, bool, int, float),
        ):
            item = getattr(value, "item", None)
            if callable(item):
                value = item()

        if not (
            value is None
            or isinstance(value, (str, bool, int, float))
        ):
            raise TypeError(
                f"metric {name!r} is not a JSON scalar after item() conversion"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"metric {name!r} must be finite")
        metrics[name] = value
    return metrics


def _load_canonical_run(run: Run | Path | str) -> Run:
    if isinstance(run, Run):
        return load_run(run.paths.root)
    if not isinstance(run, (Path, str)):
        raise TypeError("run must be a loaded Run or an explicit run directory")
    return load_run(run)


def _checkpoint_path(run: Run, artifact: CheckpointArtifact) -> Path:
    reference = Path(artifact.path)
    root = None if reference.is_absolute() else run.paths.root
    mismatches = verify_checkpoint(artifact, root=root)
    if mismatches:
        details = "; ".join(
            f"{mismatch.field}: expected {mismatch.expected!r}, "
            f"observed {mismatch.actual!r}"
            for mismatch in mismatches
        )
        raise ValueError(f"selected checkpoint identity mismatch: {details}")
    if reference.is_absolute():
        return Path(os.path.abspath(os.fspath(reference)))
    return Path(os.path.abspath(os.fspath(run.paths.root / reference)))


def _require_execution_inputs_unchanged(
    run: Run,
    checkpoint_path: Path,
) -> None:
    """Reverify the exact config and checkpoint just before runner creation."""
    current = load_run(run.paths.root)
    if current.manifest != run.manifest:
        raise ValueError("run manifest changed before evaluation")
    selected = current.selected_checkpoint
    if selected is None:
        raise ValueError("run no longer has a selected checkpoint")
    if _checkpoint_path(current, selected) != checkpoint_path:
        raise ValueError("selected checkpoint path changed before evaluation")


def _capture_initial_evidence() -> tuple[CodeProvenance, EnvironmentInfo]:
    provenance = capture_code_provenance(
        DEFAULT_REPOSITORY_ROOT,
        _PROVENANCE_SCOPES,
    )
    environment = capture_environment(
        include_packages=True,
        include_torch=False,
        package_names=_CORE_PACKAGES,
    )
    return provenance, environment


def _capture_smoke_initial_evidence() -> tuple[CodeProvenance, EnvironmentInfo]:
    provenance = capture_code_provenance(
        DEFAULT_REPOSITORY_ROOT,
        _SMOKE_PROVENANCE_SCOPES,
    )
    environment = capture_environment(
        include_packages=True,
        include_torch=False,
        package_names=_CORE_PACKAGES,
    )
    return provenance, environment


def _capture_execution_environment() -> EnvironmentInfo:
    return capture_environment(
        include_packages=True,
        include_torch=True,
        package_names=_CORE_PACKAGES,
    )


def _cleanup_cuda() -> None:
    """Release transient evaluation state without importing Torch to do so."""
    try:
        gc.collect()
        torch = sys.modules.get("torch")
        cuda = None if torch is None else getattr(torch, "cuda", None)
        empty_cache = None if cuda is None else getattr(cuda, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
    except BaseException:
        # Cleanup is advisory and must never replace evaluation/publication state.
        pass


def _execute_mmengine(run: Run, checkpoint_path: Path) -> dict[str, object]:
    runner: object | None = None
    raw_metrics: object | None = None
    try:
        # The KITTI alias must exist before MMDetection3D registers evaluators.
        compatibility = importlib.import_module(
            "lidar_model_selection.compat.kitti_evaluator"
        )
        compatibility.install()

        mmdet3d_utils = importlib.import_module("mmdet3d.utils")
        mmdet3d_utils.register_all_modules(init_default_scope=True)

        config_class = importlib.import_module("mmengine.config").Config
        config = config_class.fromfile(os.fspath(run.paths.config))

        custom_imports = _value(config, "custom_imports", None)
        if custom_imports:
            try:
                options = dict(custom_imports)
            except (TypeError, ValueError) as error:
                raise TypeError("config custom_imports must be a mapping") from error
            importer = importlib.import_module(
                "mmengine.utils"
            ).import_modules_from_strings
            importer(**options)

        with tempfile.TemporaryDirectory(
            prefix=f"lidar-evaluation-{run.run_id}-"
        ) as work_directory:
            _set_value(config, "load_from", os.fspath(checkpoint_path))
            _set_value(config, "resume", False)
            _set_value(config, "launcher", "none")
            _set_value(config, "work_dir", work_directory)

            _require_execution_inputs_unchanged(run, checkpoint_path)
            runner_class = importlib.import_module("mmengine.runner").Runner
            runner = runner_class.from_cfg(config)
            raw_metrics = runner.test()  # type: ignore[attr-defined]
            return normalize_metrics(raw_metrics)  # type: ignore[arg-type]
    finally:
        raw_metrics = None
        runner = None
        _cleanup_cuda()


def _payload(run: Run, metrics: Mapping[str, object]) -> dict[str, object]:
    dataset = run.manifest.dataset
    return {
        "kind": "evaluation",
        "metric_profile": {
            "id": RAW_METRIC_PROFILE_ID,
            "version": RAW_METRIC_PROFILE_VERSION,
            "key": RAW_METRIC_PROFILE_KEY,
        },
        "semantic_partition": dataset.semantic_partition,
        "framework_key": dataset.framework_key,
        "metrics": dict(metrics),
    }


def _iter_loss_tensors(value: Any) -> Iterable[Any]:
    torch = importlib.import_module("torch")
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_loss_tensors(item)
        return
    raise TypeError(
        "loss values must be tensors or sequences of tensors, got "
        f"{type(value).__name__}"
    )


def _first_valid_sample(dataset: Any, search_limit: int = 32) -> dict[str, Any]:
    if isinstance(search_limit, bool) or not isinstance(search_limit, int):
        raise TypeError("search_limit must be an integer")
    if search_limit <= 0:
        raise ValueError("search_limit must be greater than zero")
    limit = min(len(dataset), search_limit)
    for index in range(limit):
        sample = dataset[index]
        instances = sample["data_samples"].gt_instances_3d
        if len(instances.labels_3d) > 0:
            return sample
    raise RuntimeError(
        f"no nonempty training sample found in the first {limit} items"
    )


def _validate_training_sample(sample: Mapping[str, Any]) -> None:
    torch = importlib.import_module("torch")
    points = sample["inputs"]["points"]
    instances = sample["data_samples"].gt_instances_3d
    boxes = instances.bboxes_3d.tensor
    labels = instances.labels_3d
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"expected points with shape (N, 4), got {points.shape}")
    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise ValueError(f"expected boxes with shape (N, 7), got {boxes.shape}")
    if labels.ndim != 1 or labels.shape[0] != boxes.shape[0]:
        raise ValueError("labels must have shape (N,) and align with boxes")
    if not torch.isfinite(points).all():
        raise ValueError("training points contain NaN or infinite values")
    if not torch.isfinite(boxes).all():
        raise ValueError("training boxes contain NaN or infinite values")


def _validate_predictions(predictions: object) -> tuple[object, object, object]:
    torch = importlib.import_module("torch")
    if len(predictions) != 1:  # type: ignore[arg-type]
        raise RuntimeError(
            f"expected one prediction, got {len(predictions)}"  # type: ignore[arg-type]
        )
    prediction = predictions[0].pred_instances_3d  # type: ignore[index]
    boxes = prediction.bboxes_3d.tensor
    scores = prediction.scores_3d
    labels = prediction.labels_3d
    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise RuntimeError(f"prediction boxes must have shape (N, 7), got {boxes.shape}")
    if scores.ndim != 1 or labels.ndim != 1:
        raise RuntimeError("prediction scores and labels must be vectors")
    if not (len(boxes) == len(scores) == len(labels)):
        raise RuntimeError("prediction boxes, scores, and labels disagree")
    if not torch.isfinite(boxes).all():
        raise RuntimeError("prediction boxes contain non-finite values")
    if not torch.isfinite(scores).all():
        raise RuntimeError("prediction scores contain non-finite values")
    return boxes, scores, labels


def _execute_smoke(
    loaded: Run,
    checkpoint_path: Path,
    runtime: dict[str, object],
) -> dict[str, object]:
    try:
        importlib.import_module(
            "lidar_model_selection.compat.kitti_evaluator"
        ).install()
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this smoke test")
        selected_device_index = torch.cuda.current_device()
        if isinstance(selected_device_index, bool) or not isinstance(
            selected_device_index,
            int,
        ):
            raise TypeError("Torch returned an invalid current CUDA device index")
        selected_device_name = str(
            torch.cuda.get_device_name(selected_device_index)
        ).strip()
        if not selected_device_name:
            raise ValueError("Torch returned an empty current CUDA device name")
        runtime["selected_cuda_device_index"] = selected_device_index
        runtime["selected_cuda_device_name"] = selected_device_name
        importlib.import_module("mmdet3d.utils").register_all_modules(
            init_default_scope=True
        )
        config = importlib.import_module("mmengine.config").Config.fromfile(
            os.fspath(loaded.paths.config)
        )
        custom_imports = _value(config, "custom_imports", None)
        if custom_imports:
            importlib.import_module(
                "mmengine.utils"
            ).import_modules_from_strings(**dict(custom_imports))
        _require_execution_inputs_unchanged(loaded, checkpoint_path)

        registry = importlib.import_module("mmdet3d.registry")
        pseudo_collate = importlib.import_module(
            "mmengine.dataset"
        ).pseudo_collate
        load_checkpoint = importlib.import_module(
            "mmengine.runner"
        ).load_checkpoint

        train_dataloader = _value(config, "train_dataloader")
        train_dataset = registry.DATASETS.build(
            copy.deepcopy(_value(train_dataloader, "dataset"))
        )
        train_sample = _first_valid_sample(train_dataset)
        _validate_training_sample(train_sample)

        _require_execution_inputs_unchanged(loaded, checkpoint_path)
        model = registry.MODELS.build(copy.deepcopy(_value(config, "model")))
        _require_execution_inputs_unchanged(loaded, checkpoint_path)
        load_checkpoint(model, os.fspath(checkpoint_path), map_location="cpu")
        model = model.cuda()
        model.train()

        training_batch = pseudo_collate([train_sample])
        processed_batch = model.data_preprocessor(training_batch, training=True)
        losses = model(**processed_batch, mode="loss")
        if not isinstance(losses, Mapping):
            raise TypeError("model loss output must be a mapping")
        loss_tensors = [
            tensor.mean()
            for value in losses.values()
            for tensor in _iter_loss_tensors(value)
        ]
        if not loss_tensors:
            raise RuntimeError("model returned no loss tensors")
        total_loss = torch.stack(loss_tensors).sum()
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"non-finite total loss: {total_loss.item()}")

        total_loss.backward()
        torch.cuda.synchronize()
        gradient_count = 0
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            gradient_count += 1
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError("a model parameter has a non-finite gradient")
        if gradient_count == 0:
            raise RuntimeError("backward produced no parameter gradients")

        model.zero_grad(set_to_none=True)
        model.eval()
        val_dataloader = _value(config, "val_dataloader")
        val_dataset = registry.DATASETS.build(
            copy.deepcopy(_value(val_dataloader, "dataset"))
        )
        validation_batch = pseudo_collate([val_dataset[0]])
        with torch.no_grad():
            processed_validation = model.data_preprocessor(
                validation_batch,
                training=False,
            )
            predictions = model(**processed_validation, mode="predict")
            torch.cuda.synchronize()
        boxes, scores, labels = _validate_predictions(predictions)
        return {
            "loss_keys": sorted(losses),
            "total_loss": float(total_loss.item()),
            "finite_gradient_tensors": gradient_count,
            "prediction_boxes_shape": list(boxes.shape),  # type: ignore[attr-defined]
            "prediction_scores_shape": list(scores.shape),  # type: ignore[attr-defined]
            "prediction_labels_shape": list(labels.shape),  # type: ignore[attr-defined]
        }
    finally:
        _cleanup_cuda()


def _smoke_runtime() -> dict[str, object]:
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    return {
        "cuda_visible_devices": visibility,
        "selected_cuda_device_index": None,
        "selected_cuda_device_name": None,
    }


def _smoke_payload(
    run: Run,
    selected_checkpoint: CheckpointArtifact,
    *,
    runtime: Mapping[str, object],
    outputs: Mapping[str, object],
) -> dict[str, object]:
    return {
        "kind": "smoke",
        "smoke_schema_version": SMOKE_SCHEMA_VERSION,
        "model": {
            "slug": run.manifest.slug,
            "config": run.manifest.config.to_dict(),
        },
        "dataset": run.manifest.dataset.to_dict(),
        "checkpoint": selected_checkpoint.to_dict(),
        "runtime": dict(runtime),
        "outputs": dict(outputs),
    }


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _require_shape(value: object, *, rank: int, description: str) -> list[int]:
    plain = _plain_json(value)
    if not isinstance(plain, list) or len(plain) != rank or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in plain
    ):
        raise ValueError(f"{description} must be a non-negative rank-{rank} shape")
    return plain


def validate_smoke_result(run: Run, result: ResultRecord) -> None:
    """Validate the smoke-specific schema and its exact run identities."""
    if not isinstance(run, Run):
        raise TypeError("run must be a loaded Run")
    if not isinstance(result, ResultRecord):
        raise TypeError("result must be a ResultRecord")
    if result.result_type != "smoke":
        raise ValueError("smoke validation requires a smoke result")
    expected_binding = binding_for_run(run)
    if result.binding != expected_binding:
        raise ValueError("smoke result binding does not match its run")
    if result.provenance is None:
        raise ValueError("smoke result requires code/workspace provenance")
    if result.environment is None:
        raise ValueError("smoke result requires runtime environment evidence")

    payload = _plain_json(result.payload)
    if not isinstance(payload, dict):
        raise TypeError("smoke payload must be an object")
    expected_keys = {
        "kind",
        "smoke_schema_version",
        "model",
        "dataset",
        "checkpoint",
        "runtime",
        "outputs",
    }
    if set(payload) != expected_keys:
        raise ValueError("smoke payload has invalid fields")
    if payload["kind"] != "smoke":
        raise ValueError("smoke payload kind must be 'smoke'")
    if payload["smoke_schema_version"] != SMOKE_SCHEMA_VERSION:
        raise ValueError("unsupported smoke payload schema version")
    if payload["model"] != {
        "slug": run.manifest.slug,
        "config": run.manifest.config.to_dict(),
    }:
        raise ValueError("smoke model/config identity does not match its run")
    if payload["dataset"] != run.manifest.dataset.to_dict():
        raise ValueError("smoke dataset identity does not match its run")
    selected = run.selected_checkpoint
    assert selected is not None
    if payload["checkpoint"] != selected.to_dict():
        raise ValueError("smoke checkpoint identity does not match its run")

    runtime = payload["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != {
        "cuda_visible_devices",
        "selected_cuda_device_index",
        "selected_cuda_device_name",
    }:
        raise ValueError("smoke runtime evidence has invalid fields")
    visibility = runtime["cuda_visible_devices"]
    if visibility is not None and not isinstance(visibility, str):
        raise TypeError("smoke CUDA visibility must be text or null")
    device_index = runtime["selected_cuda_device_index"]
    device_name = runtime["selected_cuda_device_name"]
    if (device_index is None) != (device_name is None):
        raise ValueError("smoke selected CUDA device evidence is incomplete")
    if device_index is not None and (
        isinstance(device_index, bool)
        or not isinstance(device_index, int)
        or device_index < 0
    ):
        raise ValueError("smoke selected CUDA device index is invalid")
    if device_name is not None and (
        not isinstance(device_name, str) or not device_name.strip()
    ):
        raise ValueError("smoke selected CUDA device name is invalid")

    outputs = payload["outputs"]
    if not isinstance(outputs, dict):
        raise TypeError("smoke outputs must be an object")
    if result.status == "failed":
        if outputs:
            raise ValueError("failed smoke result must not claim successful outputs")
        return

    if visibility is not None and (
        not visibility or visibility.strip() != visibility or "\0" in visibility
    ):
        raise ValueError("successful smoke CUDA visibility is invalid")
    if device_index is None or device_name is None:
        raise ValueError("successful smoke result requires selected GPU evidence")
    if result.environment.gpu_available is not True:
        raise ValueError("successful smoke result requires an available GPU")
    if device_name not in result.environment.gpu_devices:
        raise ValueError("selected smoke GPU is absent from environment evidence")
    if set(outputs) != _SMOKE_OUTPUT_KEYS:
        raise ValueError("successful smoke outputs have invalid fields")
    loss_keys = outputs["loss_keys"]
    if (
        not isinstance(loss_keys, list)
        or not loss_keys
        or not all(isinstance(key, str) and key for key in loss_keys)
        or loss_keys != sorted(set(loss_keys))
    ):
        raise ValueError("smoke loss keys must be sorted, unique text")
    total_loss = outputs["total_loss"]
    if isinstance(total_loss, bool) or not isinstance(total_loss, (int, float)):
        raise TypeError("smoke total loss must be numeric")
    if not math.isfinite(float(total_loss)):
        raise ValueError("smoke total loss must be finite")
    gradient_count = outputs["finite_gradient_tensors"]
    if (
        isinstance(gradient_count, bool)
        or not isinstance(gradient_count, int)
        or gradient_count <= 0
    ):
        raise ValueError("smoke finite gradient count must be positive")
    boxes = _require_shape(
        outputs["prediction_boxes_shape"],
        rank=2,
        description="smoke prediction boxes",
    )
    scores = _require_shape(
        outputs["prediction_scores_shape"],
        rank=1,
        description="smoke prediction scores",
    )
    labels = _require_shape(
        outputs["prediction_labels_shape"],
        rank=1,
        description="smoke prediction labels",
    )
    if boxes[1] != 7 or not (boxes[0] == scores[0] == labels[0]):
        raise ValueError("smoke prediction output shapes disagree")


def list_smoke_results(run: Run | Path | str) -> tuple[ResultRecord, ...]:
    """Load and validate every immutable smoke attempt owned by one run."""
    loaded = _load_canonical_run(run)
    records = list_results(loaded, "smoke")
    for record in records:
        validate_smoke_result(loaded, record)
    return records


def smoke_stage_status(run: Run | Path | str) -> str:
    """Classify validated smoke evidence as missing, failed, or successful.

    Multiple successful records remain classified as successful evidence, but
    no record is silently selected; consumers must still use an explicit ID.
    """
    records = list_smoke_results(run)
    if not records:
        return "missing"
    if any(record.successful for record in records):
        return "successful"
    return "failed"


def smoke_run(run: Run | Path | str) -> ResultRecord:
    """Exercise and immutably publish one run-bound smoke attempt."""
    started_at = _timestamp()
    loaded = _load_canonical_run(run)
    binding = binding_for_run(loaded)
    selected = loaded.selected_checkpoint
    assert selected is not None
    provenance: CodeProvenance | None = None
    environment: EnvironmentInfo | None = None
    runtime = _smoke_runtime()

    try:
        provenance, environment = _capture_smoke_initial_evidence()
        visibility = runtime["cuda_visible_devices"]
        if visibility is not None and (
            not isinstance(visibility, str)
            or not visibility
            or visibility.strip() != visibility
            or "\0" in visibility
        ):
            raise ValueError("CUDA_VISIBLE_DEVICES must be non-empty canonical text")
        checkpoint_path = _checkpoint_path(loaded, selected)
        outputs = _execute_smoke(loaded, checkpoint_path, runtime)
        environment = _capture_execution_environment()
        succeeded = create_result(
            result_type="smoke",
            binding=binding,
            status="succeeded",
            started_at=started_at,
            finished_at=_timestamp(),
            payload=_smoke_payload(
                loaded,
                selected,
                runtime=runtime,
                outputs=outputs,
            ),
            provenance=provenance,
            environment=environment,
            failure=None,
        )
        validate_smoke_result(loaded, succeeded)
    except BaseException as error:
        if provenance is None or environment is None:
            # Without code and environment identity there is not enough
            # trustworthy evidence to publish a schema-valid smoke result.
            raise
        failed = create_result(
            result_type="smoke",
            binding=binding,
            status="failed",
            started_at=started_at,
            finished_at=_timestamp(),
            payload=_smoke_payload(
                loaded,
                selected,
                runtime=runtime,
                outputs={},
            ),
            provenance=provenance,
            environment=environment,
            failure=ResultFailure(
                error_type=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
            ),
        )
        validate_smoke_result(loaded, failed)
        publish_result(loaded, failed)
        if not isinstance(error, Exception):
            raise
        return failed

    publish_result(loaded, succeeded)
    return succeeded


def evaluate_run(run: Run | Path | str) -> ResultRecord:
    """Evaluate one explicit completed run and publish one fresh result.

    Normal execution failures become immutable failed evaluation records.  A
    failure to publish that terminal record is deliberately propagated.
    Invalid or incomplete run inputs fail before publication because no
    trustworthy completed-run binding has been established.
    """
    started_at = _timestamp()
    loaded = _load_canonical_run(run)
    binding = binding_for_run(loaded)
    selected_checkpoint = loaded.selected_checkpoint
    assert selected_checkpoint is not None

    provenance: CodeProvenance | None = None
    environment: EnvironmentInfo | None = None
    try:
        provenance, environment = _capture_initial_evidence()
        checkpoint_path = _checkpoint_path(loaded, selected_checkpoint)
        metrics = _execute_mmengine(loaded, checkpoint_path)
        environment = _capture_execution_environment()
    except BaseException as error:
        failed = create_result(
            result_type="evaluation",
            binding=binding,
            status="failed",
            started_at=started_at,
            finished_at=_timestamp(),
            payload=_payload(loaded, {}),
            provenance=provenance,
            environment=environment,
            failure=ResultFailure(
                error_type=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
            ),
        )
        publish_result(loaded, failed)
        if not isinstance(error, Exception):
            raise
        return failed

    succeeded = create_result(
        result_type="evaluation",
        binding=binding,
        status="succeeded",
        started_at=started_at,
        finished_at=_timestamp(),
        payload=_payload(loaded, metrics),
        provenance=provenance,
        environment=environment,
        failure=None,
    )
    publish_result(loaded, succeeded)
    return succeeded
