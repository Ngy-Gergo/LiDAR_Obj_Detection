"""Native run creation and run-local MMEngine training execution."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib
import os
import secrets
import stat
import tempfile
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TextIO

from .catalog import source_config_for_slug
from .checkpoints import (
    CheckpointArtifact,
    TrainingOutputs,
    identify_checkpoint,
    list_epoch_checkpoints,
    select_training_outputs,
    verify_checkpoint,
)
from .provenance import (
    TrainingCompatibilityIdentity,
    build_training_compatibility,
    capture_code_provenance,
    capture_environment,
    identify_file_set,
)
from .runs import (
    DATASET_IDENTITY_SCHEME,
    DatasetIdentity,
    Run,
    RunPaths,
    TrainingAttempt,
    TrainingState,
    build_dataset_identity,
    create_run,
    generate_run_id,
    load_run,
    update_training_state,
)


__all__ = (
    "DEFAULT_REPOSITORY_ROOT",
    "DEFAULT_RUNS_ROOT",
    "TrainingDecision",
    "create_training_run",
    "decide_training",
    "execute_training",
)

DEFAULT_REPOSITORY_ROOT = Path(__file__).absolute().parents[3]
DEFAULT_RUNS_ROOT = DEFAULT_REPOSITORY_ROOT / "research" / "runs"

_TRAINING_LOCK_NAME = ".training.lock"
_LAST_CHECKPOINT_NAME = "last_checkpoint"
_CORE_PACKAGES = ("torch", "mmengine", "mmcv", "mmdet", "mmdet3d")
_PROVENANCE_SCOPES = (
    "research/configs/centerpoint",
    "research/src/lidar_model_selection/catalog.py",
    "research/src/lidar_model_selection/checkpoints.py",
    "research/src/lidar_model_selection/compat",
    "research/src/lidar_model_selection/runs.py",
    "research/src/lidar_model_selection/training.py",
    "research/tools/train.py",
)
_TRAINING_SOURCE_PATHS = (
    "research/src/lidar_model_selection/training.py",
    "research/src/lidar_model_selection/checkpoints.py",
    "research/src/lidar_model_selection/compat/center_head_7d.py",
    "research/src/lidar_model_selection/compat/kitti_evaluator.py",
)
_PROTECTED_OVERRIDE_KEYS = frozenset(
    {"work_dir", "load_from", "resume", "launcher", "train_cfg.max_epochs"}
)
_MISSING = object()


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _positive_epoch(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("target_epoch must be a positive integer")
    return value


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _value(container: object, key: str, default: object = _MISSING) -> Any:
    if isinstance(container, Mapping):
        if key in container:
            return container[key]
    else:
        try:
            return getattr(container, key)
        except AttributeError:
            pass
    if default is _MISSING:
        raise ValueError(f"effective config is missing {key!r}")
    return default


def _nested(container: object, *keys: str) -> Any:
    current = container
    for key in keys:
        current = _value(current, key)
    return current


def _set_value(container: object, key: str, value: object) -> None:
    try:
        container[key] = value  # type: ignore[index]
    except (TypeError, AttributeError):
        setattr(container, key, value)


def _set_nested(container: object, keys: tuple[str, ...], value: object) -> None:
    current = container
    for key in keys[:-1]:
        current = _value(current, key)
    _set_value(current, keys[-1], value)


def _config_class() -> Any:
    try:
        return importlib.import_module("mmengine.config").Config
    except (AttributeError, ImportError) as error:
        raise RuntimeError(
            "MMEngine is required for training config handling"
        ) from error


def _load_config(path: Path) -> Any:
    return _config_class().fromfile(os.fspath(path))


def _apply_overrides(config: object, overrides: Mapping[str, object] | None) -> None:
    if overrides is None:
        return
    if not isinstance(overrides, Mapping) or not all(
        isinstance(key, str) for key in overrides
    ):
        raise TypeError("config_overrides must map strings to values")
    conflicts = sorted(_PROTECTED_OVERRIDE_KEYS.intersection(overrides))
    if conflicts:
        raise ValueError(
            "creation overrides cannot replace run-owned fields: "
            + ", ".join(conflicts)
        )
    merger = getattr(config, "merge_from_dict", None)
    if not callable(merger):
        raise TypeError("MMEngine Config does not support creation overrides")
    merger(dict(overrides))


def _dataset_config(config: object, loader_key: str) -> object:
    dataset = _nested(config, loader_key, "dataset")
    while _value(dataset, "data_root", None) is None:
        nested = _value(dataset, "dataset", None)
        if nested is None or nested is dataset:
            raise ValueError(f"{loader_key} does not expose a concrete dataset")
        dataset = nested
    return dataset


def _string(value: object, *, description: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{description} must be non-empty canonical text")
    return value


def _classes(value: object, *, description: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (tuple, list)):
        raise ValueError(f"{description} must be a class sequence")
    result = tuple(value)
    if not result or not all(isinstance(item, str) and item for item in result):
        raise ValueError(f"{description} must contain class names")
    return result


def _validate_model_schema(config: object) -> None:
    class_names = _classes(_value(config, "class_names"), description="class_names")
    if class_names != ("Car",):
        raise ValueError("native training requires the KITTI Car-only class schema")

    head = _nested(config, "model", "pts_bbox_head")
    if _nested(head, "bbox_coder", "code_size") != 7:
        raise ValueError("native training requires a 7D checkpoint box schema")
    tasks = _value(head, "tasks")
    if not isinstance(tasks, (tuple, list)) or len(tasks) != 1:
        raise ValueError("native training requires exactly one model task")
    task = tasks[0]
    if _classes(_value(task, "class_names"), description="task classes") != (
        "Car",
    ) or _value(task, "num_class", 1) != 1:
        raise ValueError("native training model task must be Car-only")

    custom_imports = _value(config, "custom_imports", None)
    imports = () if custom_imports is None else _value(custom_imports, "imports", ())
    if (
        isinstance(imports, str)
        or "lidar_model_selection.compat.center_head_7d" not in tuple(imports)
    ):
        raise ValueError("effective config is missing the 7D CenterHead import")


def _dataset_identity(
    config: object,
    repository_root: Path,
    *,
    recorded_root: str | None = None,
    scheme: str = DATASET_IDENTITY_SCHEME,
) -> DatasetIdentity:
    train = _dataset_config(config, "train_dataloader")
    validation = _dataset_config(config, "val_dataloader")
    test = _dataset_config(config, "test_dataloader")
    for dataset, description, expected_test_mode in (
        (train, "training", False),
        (validation, "validation", True),
        (test, "test", True),
    ):
        if _value(dataset, "type") != "KittiDataset":
            raise ValueError(f"{description} dataset must be KittiDataset")
        if _value(dataset, "box_type_3d", None) != "LiDAR":
            raise ValueError(f"{description} dataset must use LiDAR boxes")
        if _value(dataset, "test_mode", None) is not expected_test_mode:
            raise ValueError(
                f"{description} dataset test_mode must be {expected_test_mode}"
            )
        metainfo = _value(dataset, "metainfo")
        if _classes(_value(metainfo, "classes"), description="dataset classes") != (
            "Car",
        ):
            raise ValueError(f"{description} dataset must be Car-only")

    train_root = _string(_value(train, "data_root"), description="training data_root")
    validation_root = _string(
        _value(validation, "data_root"), description="validation data_root"
    )
    test_root = _string(_value(test, "data_root"), description="test data_root")
    if len(
        {
            os.path.normpath(train_root),
            os.path.normpath(validation_root),
            os.path.normpath(test_root),
        }
    ) != 1:
        raise ValueError("training, validation, and test data roots differ")

    configured_root = Path(train_root)
    observed_root = (
        _absolute(recorded_root)
        if recorded_root is not None
        else _absolute(
            configured_root
            if configured_root.is_absolute()
            else repository_root / configured_root
        )
    )
    train_ann = _string(_value(train, "ann_file"), description="training ann_file")
    validation_ann = _string(
        _value(validation, "ann_file"), description="validation ann_file"
    )
    test_ann = _string(_value(test, "ann_file"), description="test ann_file")
    if os.path.normpath(validation_ann) != os.path.normpath(test_ann):
        raise ValueError("val_dataloader and test_dataloader identify different splits")
    if os.path.normpath(train_ann) == os.path.normpath(validation_ann):
        raise ValueError("training and validation annotation files must differ")

    configured_validation = os.path.normpath(
        os.path.join(validation_root, validation_ann)
    )
    accepted_evaluator_names = {
        os.path.normpath(validation_ann),
        configured_validation,
    }
    for evaluator_key in ("val_evaluator", "test_evaluator"):
        evaluator_ann = _string(
            _nested(config, evaluator_key, "ann_file"),
            description=f"{evaluator_key} ann_file",
        )
        if os.path.normpath(evaluator_ann) not in accepted_evaluator_names:
            raise ValueError(
                f"{evaluator_key} does not identify the validation annotation file"
            )
    annotation_files = identify_file_set(
        observed_root,
        (Path(train_ann), Path(validation_ann)),
    )
    return build_dataset_identity(
        name="KITTI",
        version=None,
        root_reference=os.fspath(observed_root),
        semantic_partition="KITTI validation",
        framework_key="test_dataloader",
        annotation_files=annotation_files,
        class_names=("Car",),
        tasks={"3d_object_detection_7d": ("Car",)},
        scheme=scheme,
    )


def _validate_snapshot(config: object, paths: RunPaths, target_epoch: int) -> None:
    work_dir = _string(_value(config, "work_dir"), description="work_dir")
    if _absolute(work_dir) != paths.training:
        raise ValueError(
            "effective config work_dir is not the final run training directory"
        )
    if _nested(config, "train_cfg", "max_epochs") != target_epoch:
        raise ValueError("effective config max_epochs does not match target_epoch")
    if _value(config, "load_from", None) is not None:
        raise ValueError("canonical config load_from must be cleared")
    if _value(config, "resume", None) is not False:
        raise ValueError("canonical config resume must be false")
    if _value(config, "launcher", None) != "none":
        raise ValueError("canonical config launcher must be 'none'")
    _validate_model_schema(config)
    _dataset_config(config, "train_dataloader")
    _dataset_config(config, "test_dataloader")


def _require_parent_compatible(
    parent: Run,
    compatibility: TrainingCompatibilityIdentity,
    dataset: DatasetIdentity,
) -> None:
    loaded = load_run(parent.paths.root)
    if (
        loaded.manifest.training.status != "completed"
        or loaded.selected_checkpoint is None
    ):
        raise ValueError("parent run must be completed with a selected checkpoint")
    if verify_checkpoint(loaded.selected_checkpoint, root=loaded.paths.root):
        raise ValueError("parent selected checkpoint does not match its evidence")
    previous = loaded.manifest.training_compatibility
    if previous is None:
        raise ValueError("parent run lacks training compatibility evidence")
    current = compatibility
    parent_dataset = loaded.manifest.dataset
    if previous.dataset_sha256 != parent_dataset.identity_sha256:
        raise ValueError("parent dataset evidence is inconsistent")
    if current.dataset_sha256 != dataset.identity_sha256:
        raise ValueError("child dataset evidence is inconsistent")
    observed_for_parent = build_dataset_identity(
        name=dataset.name,
        version=dataset.version,
        root_reference=dataset.root_reference,
        semantic_partition=dataset.semantic_partition,
        framework_key=dataset.framework_key,
        annotation_files=dataset.annotation_files,
        class_names=dataset.class_names,
        tasks=None if dataset.tasks is None else dict(dataset.tasks),
        scheme=parent_dataset.scheme,
    )
    if observed_for_parent.identity_sha256 != parent_dataset.identity_sha256:
        raise ValueError("parent run is incompatible in dataset")
    checks = (
        (
            previous.training_sources.identity_sha256,
            current.training_sources.identity_sha256,
            "training sources",
        ),
        (previous.python_version, current.python_version, "Python"),
        (previous.core_packages, current.core_packages, "core packages"),
    )
    for expected, actual, description in checks:
        if expected != actual:
            raise ValueError(f"parent run is incompatible in {description}")


def create_training_run(
    model_slug: str,
    target_epoch: int,
    *,
    source_config: Path | None = None,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    repository_root: Path = DEFAULT_REPOSITORY_ROOT,
    config_overrides: Mapping[str, object] | None = None,
    parent_run: Run | None = None,
) -> Run:
    """Create one native run from a catalog preset or explicit MMEngine config."""
    target = _positive_epoch(target_epoch)
    if not isinstance(runs_root, Path) or not isinstance(repository_root, Path):
        raise TypeError("runs_root and repository_root must be pathlib.Path values")
    if parent_run is not None and not isinstance(parent_run, Run):
        raise TypeError("parent_run must be a loaded Run or None")
    if parent_run is not None and (
        _absolute(parent_run.paths.root.parent) != _absolute(runs_root)
    ):
        raise ValueError("parent run must belong to the same runs_root")

    if source_config is not None and not isinstance(source_config, Path):
        raise TypeError("source_config must be a pathlib.Path or None")
    source = (
        source_config_for_slug(model_slug)
        if source_config is None
        else _absolute(source_config)
    )
    metadata = source.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"source config must be a regular file: {source}")

    # The ID comes first because its final directory is part of the exact config.
    run_id = generate_run_id(model_slug)
    paths = RunPaths.for_run(runs_root, run_id)
    config = _load_config(source)
    _apply_overrides(config, config_overrides)
    _set_nested(config, ("train_cfg", "max_epochs"), target)
    _set_value(config, "work_dir", os.fspath(paths.training))
    _set_value(config, "load_from", None)
    _set_value(config, "resume", False)
    _set_value(config, "launcher", "none")

    with tempfile.TemporaryDirectory(prefix="lidar-config-") as temporary:
        snapshot_path = Path(temporary) / "config.py"
        dumper = getattr(config, "dump", None)
        if not callable(dumper):
            raise TypeError("MMEngine Config does not support canonical dumps")
        dumper(os.fspath(snapshot_path))
        reloaded = _load_config(snapshot_path)
        _validate_snapshot(reloaded, paths, target)
        config_bytes = snapshot_path.read_bytes()

    repository = _absolute(repository_root)
    dataset = _dataset_identity(reloaded, repository)
    code_provenance = capture_code_provenance(repository, _PROVENANCE_SCOPES)
    environment = capture_environment(
        include_packages=True,
        include_torch=False,
        package_names=_CORE_PACKAGES,
    )
    training_sources = identify_file_set(repository, _TRAINING_SOURCE_PATHS)
    compatibility = build_training_compatibility(
        hashlib.sha256(config_bytes).hexdigest(),
        dataset.identity_sha256,
        training_sources,
        core_packages=dict(environment.packages),
        python_version=environment.python_version,
    )
    if parent_run is not None:
        _require_parent_compatible(parent_run, compatibility, dataset)

    return create_run(
        runs_root,
        slug=model_slug,
        config_bytes=config_bytes,
        dataset=dataset,
        target_epoch=target,
        code_provenance=code_provenance,
        environment=environment,
        training_compatibility=compatibility,
        parent_run_id=None if parent_run is None else parent_run.run_id,
        run_id=run_id,
    )


@dataclass(frozen=True, slots=True)
class TrainingDecision:
    """One deterministic decision based only on a run's training directory."""

    action: Literal["finalize", "resume", "fresh"]
    resume_checkpoint: CheckpointArtifact | None

    def __post_init__(self) -> None:
        if self.action not in {"finalize", "resume", "fresh"}:
            raise ValueError(f"unsupported training action: {self.action!r}")
        if self.action == "resume":
            if not isinstance(self.resume_checkpoint, CheckpointArtifact):
                raise ValueError("resume decision requires a checkpoint")
        elif self.resume_checkpoint is not None:
            raise ValueError("only a resume decision may carry a checkpoint")


def _loaded(run: Run | Path | str) -> Run:
    if isinstance(run, Run):
        return load_run(run.paths.root)
    return load_run(run)


def _decide(run: Run) -> TrainingDecision:
    _require_trainable(run)
    manifest = run.manifest

    best_paths = sorted(
        path
        for path in run.paths.training.iterdir()
        if path.name.startswith("best_") and path.name.endswith(".pth")
    )
    for path in best_paths:
        if len(path.name) <= len("best_.pth"):
            raise ValueError(f"malformed best checkpoint name: {path.name!r}")
        identify_checkpoint(path, root=run.paths.root)
    if len(best_paths) > 1:
        raise ValueError("multiple best checkpoints are ambiguous")

    artifacts = list_epoch_checkpoints(run.paths.training, root=run.paths.root)
    for artifact in artifacts:
        assert artifact.epoch is not None
        if Path(artifact.path).name != f"epoch_{artifact.epoch}.pth":
            raise ValueError(
                f"non-canonical epoch checkpoint name: {Path(artifact.path).name!r}"
            )
        if artifact.epoch > manifest.target_epoch:
            raise ValueError("training directory contains an epoch beyond target_epoch")

    target = next(
        (item for item in artifacts if item.epoch == manifest.target_epoch),
        None,
    )
    if target is not None:
        # Also reject ambiguous/corrupt best-checkpoint state before finalization.
        select_training_outputs(
            run.paths.training,
            manifest.target_epoch,
            root=run.paths.root,
        )
        return TrainingDecision("finalize", None)
    if artifacts:
        return TrainingDecision("resume", artifacts[-1])
    if best_paths:
        raise ValueError("best checkpoint exists without a resumable epoch")
    return TrainingDecision("fresh", None)


def _require_trainable(run: Run) -> None:
    manifest = run.manifest
    if manifest.origin != "native":
        raise ValueError("historical imported runs cannot be trained")
    if manifest.training.status == "completed" or not manifest.resumable:
        raise ValueError("completed runs cannot be trained again")


def decide_training(run: Run | Path | str) -> TrainingDecision:
    """Inspect one explicit run and return its next training action."""
    return _decide(_loaded(run))


@contextlib.contextmanager
def _training_lock(run: Run):
    path = run.paths.training / _TRAINING_LOCK_NAME
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("training lock must be a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"training is already active for run {run.run_id}"
            ) from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _new_attempt_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{secrets.token_hex(12)}"


def _open_console(run: Run, attempt_id: str) -> TextIO:
    path = run.paths.training / f"attempt-{attempt_id}.console.log"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    return os.fdopen(descriptor, "w", encoding="utf-8", buffering=1)


def _finish_console(stream: TextIO) -> None:
    stream.flush()
    os.fsync(stream.fileno())


def _finish_attempt(
    run: Run,
    *,
    status: Literal["succeeded", "failed", "interrupted"],
    failure: str | None,
    outputs: TrainingOutputs | None = None,
) -> Run:
    current = _loaded(run)
    running = current.manifest.training.attempts[-1]
    if running.status != "running":
        raise RuntimeError("training attempt is no longer running")
    finished = replace(
        running,
        finished_at=_timestamp(),
        status=status,
        failure=failure,
    )
    if status == "succeeded":
        if not isinstance(outputs, TrainingOutputs):
            raise TypeError("successful training requires exact outputs")
        _verify_output_evidence(current, outputs)
    state = TrainingState(
        status="completed" if status == "succeeded" else "failed",
        attempts=current.manifest.training.attempts[:-1] + (finished,),
        outputs=outputs if status == "succeeded" else None,
    )
    return update_training_state(
        current,
        state,
        expected_revision=current.manifest.revision,
    )


def _failure(error: BaseException) -> str:
    message = " ".join(str(error).split())
    return f"{type(error).__name__}: {message or 'no message'}"


def _mark_stale_attempt(run: Run) -> Run:
    if run.manifest.training.status != "running":
        return run
    return _finish_attempt(
        run,
        status="interrupted",
        failure=(
            "InterruptedError: previous training process ended without "
            "a terminal record"
        ),
    )


def _record_stale_validation_failure(run: Run, error: BaseException) -> None:
    try:
        _finish_attempt(
            run,
            status="failed",
            failure=_failure(error),
        )
    except BaseException:
        # Preserve the validation failure which caused this recovery path.
        pass


def _verify_current_compatibility(run: Run, repository_root: Path) -> None:
    recorded = run.manifest.training_compatibility
    if recorded is None:
        raise ValueError("native run lacks training compatibility evidence")
    config = _load_config(run.paths.config)
    recorded_dataset = run.manifest.dataset
    observed_dataset = _dataset_identity(
        config,
        repository_root,
        recorded_root=(
            recorded_dataset.root_reference
            if recorded_dataset.scheme != DATASET_IDENTITY_SCHEME
            else None
        ),
        scheme=recorded_dataset.scheme,
    )
    if observed_dataset.identity_sha256 != recorded_dataset.identity_sha256:
        raise ValueError("dataset annotation identity changed since run creation")
    source_paths = tuple(item.path for item in recorded.training_sources.files)
    current_sources = identify_file_set(repository_root, source_paths)
    current = build_training_compatibility(
        run.manifest.config.sha256,
        run.manifest.dataset.identity_sha256,
        current_sources,
        core_packages=tuple(name for name, _ in recorded.core_packages),
    )
    if current != recorded:
        raise ValueError("current training environment is incompatible with this run")


def _runtime_config(run: Run, decision: TrainingDecision) -> object:
    config = _load_config(run.paths.config)
    _validate_snapshot(config, run.paths, run.manifest.target_epoch)
    if decision.action == "resume":
        assert decision.resume_checkpoint is not None
        checkpoint = run.paths.root / decision.resume_checkpoint.path
        _set_value(config, "load_from", os.fspath(checkpoint))
        _set_value(config, "resume", True)
    elif run.manifest.parent_run_id is not None:
        parent = load_run(run.paths.root.parent / run.manifest.parent_run_id)
        if parent.manifest.training.status != "completed":
            raise ValueError("parent run is no longer completed")
        checkpoint = parent.selected_checkpoint
        if checkpoint is None:
            raise ValueError("parent run no longer has a selected checkpoint")
        mismatches = verify_checkpoint(checkpoint, root=parent.paths.root)
        if mismatches:
            raise ValueError(
                "parent selected checkpoint no longer matches its evidence"
            )
        _set_value(
            config,
            "load_from",
            os.fspath(parent.paths.root / checkpoint.path),
        )
        _set_value(config, "resume", False)
    else:
        _set_value(config, "load_from", None)
        _set_value(config, "resume", False)
    return config


def _require_config_unchanged(run: Run) -> None:
    observed = load_run(run.paths.root)
    if observed.manifest.revision != run.manifest.revision:
        raise RuntimeError("run manifest changed while preparing training")
    if observed.manifest.config != run.manifest.config:
        raise ValueError("canonical config identity changed while preparing training")


def _require_resume_checkpoint_unchanged(
    run: Run,
    decision: TrainingDecision,
) -> None:
    if decision.action != "resume":
        return
    assert decision.resume_checkpoint is not None
    mismatches = verify_checkpoint(
        decision.resume_checkpoint,
        root=run.paths.root,
    )
    if mismatches:
        fields = ", ".join(mismatch.field for mismatch in mismatches)
        raise ValueError(f"resume checkpoint changed after selection: {fields}")


def _require_parent_checkpoint_unchanged(
    run: Run,
    decision: TrainingDecision,
) -> None:
    if decision.action != "fresh" or run.manifest.parent_run_id is None:
        return
    parent = load_run(run.paths.root.parent / run.manifest.parent_run_id)
    checkpoint = parent.selected_checkpoint
    if parent.manifest.training.status != "completed" or checkpoint is None:
        raise ValueError("parent run is not completed with a selected checkpoint")
    if verify_checkpoint(checkpoint, root=parent.paths.root):
        raise ValueError("parent selected checkpoint changed before training")


def _run_mmengine(config: object) -> None:
    # The compatibility alias must exist before MMDetection3D imports evaluators.
    importlib.import_module(
        "lidar_model_selection.compat.kitti_evaluator"
    ).install()
    importlib.import_module("mmdet3d.utils").register_all_modules(
        init_default_scope=True
    )
    custom_imports = _value(config, "custom_imports", None)
    if custom_imports is not None:
        importer = importlib.import_module(
            "mmengine.utils"
        ).import_modules_from_strings
        importer(**dict(custom_imports))
    runner = importlib.import_module("mmengine.runner").Runner.from_cfg(config)
    runner.train()


@contextlib.contextmanager
def _capture_process_output(stream: TextIO):
    """Capture Python and native fd 1/2 writes in the attempt console."""
    stream.flush()
    saved_stdout = os.dup(1)
    try:
        saved_stderr = os.dup(2)
    except BaseException:
        try:
            os.close(saved_stdout)
        except BaseException:
            pass
        raise
    active_error: BaseException | None = None
    try:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            yield
    except BaseException as error:
        active_error = error
        raise
    finally:
        cleanup_error: BaseException | None = None
        operations = (
            stream.flush,
            lambda: os.dup2(saved_stdout, 1),
            lambda: os.dup2(saved_stderr, 2),
            lambda: os.close(saved_stdout),
            lambda: os.close(saved_stderr),
        )
        for operation in operations:
            try:
                operation()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
        if active_error is None and cleanup_error is not None:
            raise cleanup_error


def _validate_last_checkpoint(run: Run, final_path: Path) -> None:
    pointer = run.paths.training / _LAST_CHECKPOINT_NAME
    try:
        metadata = pointer.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("last_checkpoint must be a regular non-symlink file")
    if metadata.st_size > 4096:
        raise ValueError("last_checkpoint pointer is unreasonably large")
    descriptor = os.open(
        pointer,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (
            metadata.st_dev,
            metadata.st_ino,
        ) != (opened.st_dev, opened.st_ino):
            raise RuntimeError("last_checkpoint changed while being opened")
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = -1
            contents = stream.read(4097)
            final = os.fstat(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    current = pointer.lstat()
    if stat.S_ISLNK(current.st_mode) or (
        opened.st_dev,
        opened.st_ino,
    ) != (current.st_dev, current.st_ino):
        raise RuntimeError("last_checkpoint changed while being read")
    if len(contents) > 4096:
        raise ValueError("last_checkpoint pointer is unreasonably large")
    if (
        opened.st_size != final.st_size
        or opened.st_mtime_ns != final.st_mtime_ns
        or opened.st_ctime_ns != final.st_ctime_ns
    ):
        raise RuntimeError("last_checkpoint changed while being read")
    try:
        target = contents.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise ValueError("last_checkpoint pointer must be UTF-8") from error
    if not target:
        raise ValueError("last_checkpoint pointer is empty")
    candidate = Path(target)
    if not candidate.is_absolute():
        candidate = run.paths.training / candidate
    if _absolute(candidate) != _absolute(final_path):
        raise ValueError("last_checkpoint does not name the exact final checkpoint")


def _outputs(run: Run) -> TrainingOutputs:
    outputs = select_training_outputs(
        run.paths.training,
        run.manifest.target_epoch,
        root=run.paths.root,
    )
    _validate_last_checkpoint(
        run,
        run.paths.root / outputs.final_checkpoint.path,
    )
    return outputs


def _verify_output_evidence(run: Run, outputs: TrainingOutputs) -> None:
    try:
        decision = _decide(run)
        if decision.action != "finalize":
            raise ValueError("exact target checkpoint is no longer finalizable")
        observed = _outputs(run)
    except Exception as error:
        raise ValueError(
            "training output state changed before completion commit"
        ) from error
    if observed != outputs:
        raise ValueError("training output selection changed before completion commit")


def _finalize_existing(run: Run) -> Run:
    outputs = _outputs(run)
    state = run.manifest.training
    if state.status == "running":
        return _finish_attempt(run, status="succeeded", failure=None, outputs=outputs)

    attempt_id = _new_attempt_id()
    with _open_console(run, attempt_id) as console:
        console.write("Validated an already-present exact target checkpoint.\n")
        _finish_console(console)
    attempt = TrainingAttempt(
        attempt_id=attempt_id,
        started_at=_timestamp(),
        finished_at=_timestamp(),
        status="succeeded",
        resume_checkpoint=None,
        failure=None,
    )
    _verify_output_evidence(run, outputs)
    completed = TrainingState(
        status="completed",
        attempts=state.attempts + (attempt,),
        outputs=outputs,
    )
    return update_training_state(
        run,
        completed,
        expected_revision=run.manifest.revision,
    )


def execute_training(
    run: Run | Path | str,
    *,
    repository_root: Path = DEFAULT_REPOSITORY_ROOT,
) -> Run:
    """Execute, resume, or finalize one explicit native run under its lock."""
    if not isinstance(repository_root, Path):
        raise TypeError("repository_root must be a pathlib.Path")
    repository = _absolute(repository_root)
    initial = _loaded(run)
    _require_trainable(initial)
    with _training_lock(initial):
        current = _loaded(initial)
        if current.manifest.training.status == "running":
            try:
                decision = _decide(current)
            except BaseException as error:
                _record_stale_validation_failure(current, error)
                raise
            if decision.action == "finalize":
                try:
                    return _finalize_existing(current)
                except BaseException as error:
                    _record_stale_validation_failure(current, error)
                    raise
            current = _mark_stale_attempt(current)
        else:
            decision = _decide(current)
            if decision.action == "finalize":
                return _finalize_existing(current)

        _verify_current_compatibility(current, repository)
        decision = _decide(current)
        if decision.action == "finalize":
            return _finalize_existing(current)
        runtime_config = _runtime_config(current, decision)
        _require_config_unchanged(current)

        attempt_id = _new_attempt_id()
        with _open_console(current, attempt_id) as console:
            attempt = TrainingAttempt(
                attempt_id=attempt_id,
                started_at=_timestamp(),
                finished_at=None,
                status="running",
                resume_checkpoint=decision.resume_checkpoint,
                failure=None,
            )
            running = TrainingState(
                status="running",
                attempts=current.manifest.training.attempts + (attempt,),
                outputs=None,
            )
            current = update_training_state(
                current,
                running,
                expected_revision=current.manifest.revision,
            )
            try:
                with _capture_process_output(console):
                    _require_config_unchanged(current)
                    _require_resume_checkpoint_unchanged(current, decision)
                    _require_parent_checkpoint_unchanged(current, decision)
                    _run_mmengine(runtime_config)
                outputs = _outputs(current)
                _finish_console(console)
                completed = _finish_attempt(
                    current,
                    status="succeeded",
                    failure=None,
                    outputs=outputs,
                )
            except BaseException as error:
                traceback.print_exc(file=console)
                try:
                    _finish_console(console)
                    _finish_attempt(
                        current,
                        status=(
                            "interrupted"
                            if isinstance(error, (KeyboardInterrupt, SystemExit))
                            else "failed"
                        ),
                        failure=_failure(error),
                    )
                except BaseException:
                    traceback.print_exc(file=console)
                    try:
                        _finish_console(console)
                    except BaseException:
                        pass
                raise

        return completed
