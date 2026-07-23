"""Opt-in KITTI rotated-IoU compatibility for MMDetection3D 1.4.

This module replaces only the lazily imported
``mmdet3d.evaluation.functional.kitti_utils.rotate_iou`` module.  The
official MMDetection3D KITTI annotation conversion and AP implementation
remain responsible for every other part of evaluation.
"""

from __future__ import annotations

import argparse
import copy
import importlib.machinery
import importlib.util
import json
import math
import operator
import runpy
import sys
import types
import warnings
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np


_TARGET_MODULE = (
    "mmdet3d.evaluation.functional.kitti_utils.rotate_iou"
)
_ALIAS_MARKER = "_lidar_detection_kitti_eval_compat"
_installed_alias: types.ModuleType | None = None
_cpu_fallback_warned = False

_DETERMINISTIC_ATOL = 1e-6
_DETERMINISTIC_RTOL = 1e-5
_RANDOM_IOU_ATOL = 5e-6
_RANDOM_IOU_RTOL = 1e-5
_RANDOM_INTERSECTION_ATOL = 5e-5
_RANDOM_INTERSECTION_RTOL = 1e-5

_KITTI_PREDICTION_FIELDS = (
    "name",
    "truncated",
    "occluded",
    "alpha",
    "bbox",
    "dimensions",
    "location",
    "rotation_y",
    "score",
    "sample_idx",
)


def _normalize_boxes(value: Any, name: str) -> np.ndarray:
    """Validate an ``(N, 5)`` input and return contiguous float32."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(
            f"{name} must be convertible to a numeric NumPy array"
        ) from exc

    if array.ndim != 2 or array.shape[1] != 5:
        raise ValueError(
            f"{name} must have shape (N, 5); got {array.shape}"
        )
    if (
        array.dtype == np.dtype("O")
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise ValueError(
            f"{name} must contain real numeric values; got dtype {array.dtype}"
        )

    try:
        normalized = np.ascontiguousarray(array, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} could not be converted to contiguous float32"
        ) from exc

    if not normalized.flags.writeable:
        normalized = normalized.copy()
    return normalized


def _rotate_iou_on_device(
    boxes: np.ndarray,
    query_boxes: np.ndarray,
    criterion: int,
    device: Any,
) -> np.ndarray:
    """Evaluate normalized boxes on an explicit torch device."""
    output_shape = (boxes.shape[0], query_boxes.shape[0])
    if boxes.shape[0] == 0 or query_boxes.shape[0] == 0:
        return np.zeros(output_shape, dtype=np.float32)

    import torch
    from mmcv.ops import box_iou_rotated

    device = torch.device(device)
    device_context = (
        torch.cuda.device(device)
        if device.type == "cuda"
        else nullcontext()
    )

    with device_context, torch.no_grad():
        boxes_tensor = torch.from_numpy(boxes).to(device=device)
        query_boxes_tensor = torch.from_numpy(query_boxes).to(device=device)

        if criterion == -1:
            overlaps = box_iou_rotated(
                boxes_tensor,
                query_boxes_tensor,
                mode="iou",
                aligned=False,
                clockwise=False,
            )
        else:
            iof = box_iou_rotated(
                boxes_tensor,
                query_boxes_tensor,
                mode="iof",
                aligned=False,
                clockwise=False,
            )
            first_box_areas = boxes_tensor[:, 2] * boxes_tensor[:, 3]
            intersection = iof * first_box_areas[:, None]

            if criterion == 0:
                overlaps = iof
            elif criterion == 1:
                query_box_areas = (
                    query_boxes_tensor[:, 2] * query_boxes_tensor[:, 3]
                )
                overlaps = intersection / query_box_areas[None, :]
            else:
                overlaps = intersection

        result = overlaps.detach().cpu().numpy()

    return np.ascontiguousarray(
        result.reshape(output_shape),
        dtype=np.float32,
    )


def rotate_iou_gpu_eval(
    boxes: Any,
    query_boxes: Any,
    criterion: int = -1,
    device_id: int = 0,
) -> np.ndarray:
    """Compute rotated overlap with the MMDetection3D-compatible interface.

    Criterion semantics are:

    * ``-1``: intersection over union;
    * ``0``: intersection over the first-input box area;
    * ``1``: intersection over the second-input query-box area;
    * any other value: raw intersection area.
    """
    normalized_boxes = _normalize_boxes(boxes, "boxes")
    normalized_queries = _normalize_boxes(query_boxes, "query_boxes")
    output_shape = (
        normalized_boxes.shape[0],
        normalized_queries.shape[0],
    )
    if output_shape[0] == 0 or output_shape[1] == 0:
        return np.zeros(output_shape, dtype=np.float32)

    import torch

    if torch.cuda.is_available():
        try:
            selected_device = operator.index(device_id)
        except TypeError as exc:
            raise ValueError(
                f"device_id must be an integer; got {device_id!r}"
            ) from exc

        device_count = torch.cuda.device_count()
        if selected_device < 0 or selected_device >= device_count:
            raise ValueError(
                "device_id must identify an available CUDA device; "
                f"got {selected_device}, available device count is "
                f"{device_count}"
            )
        device = torch.device("cuda", selected_device)
    else:
        global _cpu_fallback_warned
        if not _cpu_fallback_warned:
            warnings.warn(
                "CUDA is unavailable; KITTI rotated-IoU evaluation is using "
                "the MMCV CPU operator.",
                RuntimeWarning,
                stacklevel=2,
            )
            _cpu_fallback_warned = True
        device = torch.device("cpu")

    return _rotate_iou_on_device(
        normalized_boxes,
        normalized_queries,
        criterion,
        device,
    )


def install() -> None:
    """Install the controlled synthetic rotated-IoU module alias."""
    global _installed_alias

    existing = sys.modules.get(_TARGET_MODULE)
    if existing is not None:
        if (
            existing is _installed_alias
            and getattr(existing, _ALIAS_MARKER, False) is True
            and getattr(existing, "rotate_iou_gpu_eval", None)
            is rotate_iou_gpu_eval
        ):
            return
        raise RuntimeError(
            f"Cannot install KITTI evaluation compatibility: "
            f"{_TARGET_MODULE!r} is already loaded by another module. "
            "Start a fresh process and invoke this compatibility module "
            "before importing the official rotated-IoU evaluator."
        )

    alias = types.ModuleType(
        _TARGET_MODULE,
        "Project-local MMCV rotated-IoU compatibility module.",
    )
    alias.__package__ = _TARGET_MODULE.rpartition(".")[0]
    alias.__spec__ = importlib.machinery.ModuleSpec(
        _TARGET_MODULE,
        loader=None,
    )
    alias.__dict__["__all__"] = ("rotate_iou_gpu_eval",)
    alias.__dict__["rotate_iou_gpu_eval"] = rotate_iou_gpu_eval
    setattr(alias, _ALIAS_MARKER, True)

    sys.modules[_TARGET_MODULE] = alias
    _installed_alias = alias


def _assert_allclose(
    actual: np.ndarray,
    expected: np.ndarray | float,
    *,
    atol: float = _DETERMINISTIC_ATOL,
    rtol: float = _DETERMINISTIC_RTOL,
) -> None:
    np.testing.assert_allclose(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
    )


def _self_test_overlap(
    boxes: Any,
    query_boxes: Any,
    criterion: int,
    device: Any,
) -> np.ndarray:
    normalized_boxes = _normalize_boxes(boxes, "boxes")
    normalized_queries = _normalize_boxes(query_boxes, "query_boxes")
    return _rotate_iou_on_device(
        normalized_boxes,
        normalized_queries,
        criterion,
        device,
    )


def _run_input_contract_tests() -> None:
    invalid_values = (
        (np.zeros((5,), dtype=np.float32), "shape"),
        (np.zeros((2, 4), dtype=np.float32), "shape"),
        (np.array([["x"] * 5]), "dtype"),
        (np.array([[object()] * 5], dtype=object), "dtype"),
        (np.ones((1, 5), dtype=np.complex64), "dtype"),
        (np.ones((1, 5), dtype=np.bool_), "dtype"),
    )
    valid_query = np.zeros((1, 5), dtype=np.float32)
    for value, description in invalid_values:
        try:
            rotate_iou_gpu_eval(value, valid_query)
        except ValueError:
            continue
        raise AssertionError(
            f"Invalid {description} input was not rejected: {value!r}"
        )


def _run_deterministic_tests(device: Any) -> None:
    identical_boxes = np.array(
        [
            [0.0, 0.0, 2.0, 4.0, angle]
            for angle in (
                0.0,
                0.1,
                0.2,
                0.37,
                math.pi / 4.0,
                math.pi / 2.0,
            )
        ],
        dtype=np.float32,
    )
    for box in identical_boxes:
        overlap = _self_test_overlap(
            box[None, :],
            box[None, :],
            -1,
            device,
        )
        _assert_allclose(overlap, np.array([[1.0]], dtype=np.float32))

        translated = box.copy()
        translated[:2] = (17.25, -8.5)
        translated_overlap = _self_test_overlap(
            translated[None, :],
            translated[None, :],
            -1,
            device,
        )
        _assert_allclose(
            translated_overlap,
            np.array([[1.0]], dtype=np.float32),
        )

    first = np.array([[0.0, 0.0, 2.0, 2.0, 0.0]], dtype=np.float32)
    partial = np.array([[1.0, 0.0, 2.0, 2.0, 0.0]], dtype=np.float32)
    far = np.array([[20.0, 20.0, 2.0, 2.0, 0.0]], dtype=np.float32)

    partial_iou = _self_test_overlap(first, partial, -1, device)
    reverse_partial_iou = _self_test_overlap(partial, first, -1, device)
    _assert_allclose(partial_iou, np.array([[1.0 / 3.0]], np.float32))
    _assert_allclose(partial_iou, reverse_partial_iou)
    _assert_allclose(
        _self_test_overlap(first, partial, 0, device),
        np.array([[0.5]], np.float32),
    )
    _assert_allclose(
        _self_test_overlap(first, partial, 1, device),
        np.array([[0.5]], np.float32),
    )
    _assert_allclose(
        _self_test_overlap(first, partial, 2, device),
        np.array([[2.0]], np.float32),
    )
    _assert_allclose(
        _self_test_overlap(first, far, -1, device),
        np.array([[0.0]], np.float32),
    )

    small = np.array([[0.0, 0.0, 2.0, 2.0, 0.37]], dtype=np.float32)
    large = np.array([[0.0, 0.0, 4.0, 4.0, 0.37]], dtype=np.float32)
    expected_small_large = {
        -1: 0.25,
        0: 1.0,
        1: 0.25,
        2: 4.0,
    }
    expected_large_small = {
        -1: 0.25,
        0: 0.25,
        1: 1.0,
        2: 4.0,
    }
    for criterion, expected in expected_small_large.items():
        _assert_allclose(
            _self_test_overlap(
                small,
                large,
                criterion,
                device,
            ),
            np.array([[expected]], np.float32),
        )
    for criterion, expected in expected_large_small.items():
        _assert_allclose(
            _self_test_overlap(
                large,
                small,
                criterion,
                device,
            ),
            np.array([[expected]], np.float32),
        )

    matrix_boxes = np.concatenate((first, far), axis=0)
    matrix_queries = np.concatenate((first, partial, far), axis=0)
    matrix = _self_test_overlap(
        matrix_boxes,
        matrix_queries,
        -1,
        device,
    )
    expected_matrix = np.array(
        [
            [1.0, 1.0 / 3.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    _assert_allclose(matrix, expected_matrix)
    assert matrix.shape == (2, 3)
    assert matrix.dtype == np.float32
    assert np.all(matrix >= 0.0)
    assert np.all(matrix <= 1.0)

    empty_first = _self_test_overlap(
        np.empty((0, 5), dtype=np.float32),
        matrix_queries,
        -1,
        device,
    )
    empty_second = _self_test_overlap(
        matrix_boxes,
        np.empty((0, 5), dtype=np.float32),
        -1,
        device,
    )
    assert empty_first.shape == (0, 3)
    assert empty_second.shape == (2, 0)
    assert empty_first.dtype == np.float32
    assert empty_second.dtype == np.float32


def _corners_for_shapely(box: np.ndarray) -> np.ndarray:
    center_x, center_y, width, height, angle = map(float, box)
    local = np.array(
        [
            [-width / 2.0, -height / 2.0],
            [-width / 2.0, height / 2.0],
            [width / 2.0, height / 2.0],
            [width / 2.0, -height / 2.0],
        ],
        dtype=np.float64,
    )
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = np.array(
        [[cosine, sine], [-sine, cosine]],
        dtype=np.float64,
    )
    return local @ rotation.T + np.array(
        [center_x, center_y],
        dtype=np.float64,
    )


def _run_optional_shapely_tests(device: Any) -> str:
    try:
        from shapely.geometry import Polygon
    except ImportError:
        return "SKIP (Shapely is not installed)"

    rng = np.random.default_rng(20260723)
    boxes = np.column_stack(
        (
            rng.uniform(-5.0, 5.0, 16),
            rng.uniform(-5.0, 5.0, 16),
            rng.uniform(0.5, 4.0, 16),
            rng.uniform(0.5, 4.0, 16),
            rng.uniform(-math.pi, math.pi, 16),
        )
    ).astype(np.float32)
    queries = np.column_stack(
        (
            rng.uniform(-5.0, 5.0, 13),
            rng.uniform(-5.0, 5.0, 13),
            rng.uniform(0.5, 4.0, 13),
            rng.uniform(0.5, 4.0, 13),
            rng.uniform(-math.pi, math.pi, 13),
        )
    ).astype(np.float32)

    expected_iou = np.zeros((len(boxes), len(queries)), dtype=np.float64)
    expected_first_fraction = np.zeros_like(expected_iou)
    expected_query_fraction = np.zeros_like(expected_iou)
    expected_intersection = np.zeros_like(expected_iou)

    box_polygons = [Polygon(_corners_for_shapely(box)) for box in boxes]
    query_polygons = [
        Polygon(_corners_for_shapely(box)) for box in queries
    ]
    for row, first_polygon in enumerate(box_polygons):
        for column, query_polygon in enumerate(query_polygons):
            intersection = first_polygon.intersection(query_polygon).area
            union = first_polygon.area + query_polygon.area - intersection
            expected_intersection[row, column] = intersection
            expected_iou[row, column] = intersection / union
            expected_first_fraction[row, column] = (
                intersection / first_polygon.area
            )
            expected_query_fraction[row, column] = (
                intersection / query_polygon.area
            )

    actual_iou = _self_test_overlap(boxes, queries, -1, device)
    actual_first_fraction = _self_test_overlap(boxes, queries, 0, device)
    actual_query_fraction = _self_test_overlap(boxes, queries, 1, device)
    actual_intersection = _self_test_overlap(boxes, queries, 2, device)

    _assert_allclose(
        actual_iou,
        expected_iou,
        atol=_RANDOM_IOU_ATOL,
        rtol=_RANDOM_IOU_RTOL,
    )
    _assert_allclose(
        actual_first_fraction,
        expected_first_fraction,
        atol=_RANDOM_IOU_ATOL,
        rtol=_RANDOM_IOU_RTOL,
    )
    _assert_allclose(
        actual_query_fraction,
        expected_query_fraction,
        atol=_RANDOM_IOU_ATOL,
        rtol=_RANDOM_IOU_RTOL,
    )
    _assert_allclose(
        actual_intersection,
        expected_intersection,
        atol=_RANDOM_INTERSECTION_ATOL,
        rtol=_RANDOM_INTERSECTION_RTOL,
    )

    maximum_iou_error = float(
        np.max(np.abs(actual_iou - expected_iou))
    )
    return f"PASS (maximum IoU error {maximum_iou_error:.3e})"


def _run_self_test(device_mode: str) -> int:
    install()
    _run_input_contract_tests()

    import torch

    devices: list[Any] = []
    if device_mode in ("all", "cpu"):
        devices.append(torch.device("cpu"))
    if device_mode in ("all", "cuda"):
        if torch.cuda.is_available():
            devices.append(torch.device("cuda", 0))
        else:
            print("CUDA self-test: SKIP (CUDA is unavailable)")

    if not devices:
        print("No requested self-test device is available.")
        return 0

    for device in devices:
        _run_deterministic_tests(device)
        oracle_result = _run_optional_shapely_tests(device)
        print(f"{device} deterministic tests: PASS")
        print(f"{device} Shapely oracle: {oracle_result}")

    print("KITTI rotated-IoU compatibility self-test: PASS")
    return 0


def _find_official_test_script() -> Path:
    spec = importlib.util.find_spec("mmdet3d")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError(
            "Cannot locate the installed MMDetection3D package."
        )

    candidates = [
        Path(location) / ".mim" / "tools" / "test.py"
        for location in spec.submodule_search_locations
    ]
    scripts = [candidate for candidate in candidates if candidate.is_file()]
    if len(scripts) != 1:
        rendered = ", ".join(str(candidate) for candidate in candidates)
        raise RuntimeError(
            "Expected exactly one installed MMDetection3D test script; "
            f"searched: {rendered}"
        )
    return scripts[0]


def _run_official_test(arguments: Sequence[str]) -> int:
    if not arguments:
        raise ValueError(
            "The test command requires the arguments accepted by the "
            "official MMDetection3D test script."
        )

    install()
    test_script = _find_official_test_script()
    previous_argv = sys.argv
    sys.argv = [str(test_script), *arguments]
    try:
        runpy.run_path(str(test_script), run_name="__main__")
    finally:
        sys.argv = previous_argv
    return 0


def _select_kitti_evaluator_config(config: Any) -> dict[str, Any]:
    configured = config.get("test_evaluator")
    if configured is None:
        raise ValueError("The config does not define test_evaluator.")

    candidates = (
        list(configured)
        if isinstance(configured, (list, tuple))
        else [configured]
    )
    matches: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_dict = copy.deepcopy(dict(candidate))
        metric_type = candidate_dict.get("type")
        if isinstance(metric_type, str):
            type_name = metric_type.rsplit(".", 1)[-1]
        else:
            type_name = getattr(metric_type, "__name__", "")
        if type_name == "KittiMetric":
            matches.append(candidate_dict)

    if len(matches) != 1:
        raise ValueError(
            "The config must define exactly one KittiMetric in "
            "test_evaluator."
        )
    if matches[0].get("format_only", False):
        raise ValueError(
            "eval-pkl cannot evaluate a KittiMetric configured with "
            "format_only=True."
        )
    return matches[0]


def _test_dataset_config(config: Any) -> dict[str, Any]:
    test_dataloader = config.get("test_dataloader")
    if test_dataloader is None or "dataset" not in test_dataloader:
        raise ValueError(
            "The config does not define test_dataloader.dataset."
        )

    dataset = dict(test_dataloader["dataset"])
    while "metainfo" not in dataset and "dataset" in dataset:
        dataset = dict(dataset["dataset"])
    return dataset


def _dataset_metainfo(config: Any) -> dict[str, Any]:
    dataset = _test_dataset_config(config)
    metainfo = dataset.get("metainfo")
    if metainfo is None:
        metainfo = config.get("metainfo")
    if metainfo is None:
        raise ValueError(
            "The config does not define test dataset metainfo."
        )

    result = copy.deepcopy(dict(metainfo))
    classes = result.get("classes")
    if (
        not isinstance(classes, (list, tuple))
        or not classes
        or not all(isinstance(name, str) for name in classes)
    ):
        raise ValueError(
            "The test dataset metainfo must define a non-empty classes list."
        )
    result["classes"] = tuple(classes)
    return result


def _require_sequential_test_sampler(config: Any) -> None:
    test_dataloader = config.get("test_dataloader")
    sampler = (
        test_dataloader.get("sampler")
        if test_dataloader is not None
        else None
    )
    if sampler is None or sampler.get("shuffle") is not False:
        raise ValueError(
            "eval-pkl requires test_dataloader.sampler.shuffle=False so "
            "empty predictions can be aligned by verified list order."
        )


def _validate_prediction_annotation(
    prediction: Any,
    position: int,
) -> bool:
    if not isinstance(prediction, dict):
        raise ValueError(
            f"Prediction {position} must be a KITTI annotation dictionary."
        )
    missing = [
        field
        for field in _KITTI_PREDICTION_FIELDS
        if field not in prediction
    ]
    if missing:
        raise ValueError(
            f"Prediction {position} is missing fields: {missing}"
        )

    score = np.asarray(prediction["score"])
    if score.ndim != 1:
        raise ValueError(
            f"Prediction {position} score must have shape (N,); "
            f"got {score.shape}"
        )
    count = score.shape[0]

    vector_fields = (
        "name",
        "truncated",
        "occluded",
        "alpha",
        "rotation_y",
        "score",
        "sample_idx",
    )
    for field in vector_fields:
        value = np.asarray(prediction[field])
        if value.ndim != 1 or value.shape[0] != count:
            raise ValueError(
                f"Prediction {position} field {field!r} must have shape "
                f"({count},); got {value.shape}"
            )

    matrix_shapes = {
        "bbox": (count, 4),
        "dimensions": (count, 3),
        "location": (count, 3),
    }
    for field, expected_shape in matrix_shapes.items():
        value = np.asarray(prediction[field])
        if value.shape != expected_shape:
            raise ValueError(
                f"Prediction {position} field {field!r} must have shape "
                f"{expected_shape}; got {value.shape}"
            )

    sample_indices = np.asarray(prediction["sample_idx"])
    if not np.issubdtype(sample_indices.dtype, np.integer):
        raise ValueError(
            f"Prediction {position} sample_idx must use an integer dtype; "
            f"got {sample_indices.dtype}"
        )
    if count == 0:
        return True
    if not np.all(sample_indices == sample_indices[0]):
        raise ValueError(
            f"Prediction {position} contains inconsistent sample_idx values."
        )
    declared_position = int(sample_indices[0])
    if declared_position != position:
        raise ValueError(
            f"Prediction {position} declares sample_idx "
            f"{declared_position}; expected positional index {position}."
        )
    return False


def _validate_annotation_information(annotation_data: Any) -> list[dict]:
    if not isinstance(annotation_data, dict):
        raise ValueError(
            "The KITTI annotation file must contain a dictionary."
        )
    if not isinstance(annotation_data.get("metainfo"), dict):
        raise ValueError(
            "The KITTI annotation file is missing dictionary metainfo."
        )
    data_list = annotation_data.get("data_list")
    if not isinstance(data_list, list):
        raise ValueError(
            "The KITTI annotation file is missing a data_list."
        )

    raw_sample_indices: list[int] = []
    for position, data_info in enumerate(data_list):
        if not isinstance(data_info, dict) or "sample_idx" not in data_info:
            raise ValueError(
                f"Annotation data_list entry {position} lacks sample_idx."
            )
        try:
            raw_sample_indices.append(operator.index(data_info["sample_idx"]))
        except TypeError as exc:
            raise ValueError(
                f"Annotation data_list entry {position} has a non-integral "
                "raw sample_idx."
            ) from exc
    if len(set(raw_sample_indices)) != len(raw_sample_indices):
        raise ValueError(
            "The KITTI annotation file contains duplicate raw sample_idx "
            "values."
        )
    return data_list


def _evaluate_prediction_pickle(
    *,
    config_path: Path,
    predictions_path: Path,
    annotation_path: Path,
    output_json_path: Path,
) -> int:
    install()

    from mmengine import load
    from mmengine.config import Config
    from mmdet3d.registry import METRICS
    from mmdet3d.utils import register_all_modules

    config = Config.fromfile(str(config_path))
    _require_sequential_test_sampler(config)
    evaluator_config = _select_kitti_evaluator_config(config)
    evaluator_config["ann_file"] = str(annotation_path)
    dataset_metainfo = _dataset_metainfo(config)

    register_all_modules(init_default_scope=False)
    metric = METRICS.build(evaluator_config)
    metric.dataset_meta = dataset_metainfo
    classes = list(dataset_metainfo["classes"])

    predictions = load(str(predictions_path))
    if not isinstance(predictions, list):
        raise ValueError(
            "The prediction pkl must contain the KITTI annotation list "
            "written by KittiMetric.bbox2result_kitti."
        )

    annotation_data = load(
        str(annotation_path),
        backend_args=metric.backend_args,
    )
    raw_data_list = _validate_annotation_information(annotation_data)
    converted_infos = metric.convert_annos_to_kitti_annos(annotation_data)

    if len(converted_infos) != len(raw_data_list):
        raise ValueError(
            "Official KITTI annotation conversion changed the data-list "
            "length unexpectedly."
        )
    if len(predictions) != len(converted_infos):
        raise ValueError(
            "Prediction/annotation length mismatch: "
            f"{len(predictions)} predictions versus "
            f"{len(converted_infos)} annotations."
        )

    empty_positions = 0
    verified_positions = 0
    for position, prediction in enumerate(predictions):
        is_empty = _validate_prediction_annotation(prediction, position)
        if is_empty:
            empty_positions += 1
        else:
            verified_positions += 1
    if verified_positions == 0:
        raise ValueError(
            "Every prediction is empty, so the converted pkl contains no "
            "sample_idx values with which to verify positional alignment."
        )

    gt_annos: list[dict] = []
    for position, data_info in enumerate(converted_infos):
        if "kitti_annos" not in data_info:
            raise ValueError(
                f"Converted annotation {position} lacks kitti_annos."
            )
        gt_annos.append(data_info["kitti_annos"])

    result_dict = {"pred_instances_3d": predictions}
    metrics: dict[str, float] = {}
    for requested_metric in metric.metrics:
        evaluated = metric.kitti_evaluate(
            result_dict,
            gt_annos,
            metric=requested_metric,
            classes=classes,
            logger=None,
        )
        duplicate_keys = set(metrics).intersection(evaluated)
        if duplicate_keys:
            raise RuntimeError(
                f"Duplicate KITTI metric keys: {sorted(duplicate_keys)}"
            )
        metrics.update(evaluated)

    if metric.prefix:
        metrics = {
            f"{metric.prefix}/{key}": value
            for key, value in metrics.items()
        }

    serialized = json.dumps(
        metrics,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(serialized, encoding="utf-8")

    print(
        f"Verified positional alignment for {verified_positions} nonempty "
        f"predictions; used verified list order for {empty_positions} empty "
        "predictions."
    )
    print(f"Wrote {len(metrics)} metrics to {output_json_path}")
    return 0


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Opt-in MMCV rotated-IoU compatibility for MMDetection3D KITTI "
            "evaluation."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    self_test_parser = subparsers.add_parser(
        "self-test",
        help="Run deterministic and optional Shapely overlap checks.",
    )
    self_test_parser.add_argument(
        "--device",
        choices=("all", "cpu", "cuda"),
        default="all",
        help="Backends to test. Defaults to both available backends.",
    )

    test_parser = subparsers.add_parser(
        "test",
        add_help=False,
        help="Delegate to the installed official MMDetection3D test script.",
    )
    test_parser.add_argument(
        "test_arguments",
        nargs=argparse.REMAINDER,
    )

    eval_parser = subparsers.add_parser(
        "eval-pkl",
        help="Evaluate a persisted converted KITTI prediction pkl.",
    )
    eval_parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="MMDetection3D configuration used for the predictions.",
    )
    eval_parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="Converted pred_instances_3d.pkl path.",
    )
    eval_parser.add_argument(
        "--ann-file",
        required=True,
        type=Path,
        help="Matching MMDetection3D KITTI information pkl.",
    )
    eval_parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
        help="Destination for deterministic sorted JSON metrics.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the compatibility command-line interface."""
    parser = _build_argument_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)

    if arguments and arguments[0] == "test":
        try:
            return _run_official_test(arguments[1:])
        except ValueError as exc:
            parser.error(str(exc))

    args = parser.parse_args(arguments)

    if args.command == "self-test":
        return _run_self_test(args.device)
    if args.command == "eval-pkl":
        return _evaluate_prediction_pickle(
            config_path=args.config,
            predictions_path=args.predictions,
            annotation_path=args.ann_file,
            output_json_path=args.output_json,
        )

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
