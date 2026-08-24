"""Opt-in KITTI rotated-IoU compatibility for MMDetection3D 1.4.

This module replaces only the lazily imported
``mmdet3d.evaluation.functional.kitti_utils.rotate_iou`` module.  The
official MMDetection3D KITTI annotation conversion and AP implementation
remain responsible for every other part of evaluation.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import math
import operator
import sys
import types
import warnings
from collections.abc import Sequence
from contextlib import nullcontext
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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the compatibility command-line interface."""
    parser = _build_argument_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)

    if args.command == "self-test":
        return _run_self_test(args.device)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
