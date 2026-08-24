from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from lidar_model_selection.comparison import (
    KITTI_CAR_AP40_METRICS,
    ComparisonRow,
)
from lidar_model_selection.plotting import plot_comparison


def _row(slug: str, token: str, *, complete: bool = True) -> ComparisonRow:
    ap40 = {
        metric: 50.0 + index
        for index, metric in enumerate(KITTI_CAR_AP40_METRICS)
    }
    if not complete:
        ap40 = {"car_3d_ap40_moderate_strict": 61.0}
    return ComparisonRow(
        run_id=f"20260824T120000Z-{slug}-{token * 24}",
        slug=slug,
        config_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        evaluation_result_id=f"evaluation-{slug}",
        accuracy_metric="car_3d_ap40_moderate_strict",
        accuracy_raw_key=KITTI_CAR_AP40_METRICS[
            "car_3d_ap40_moderate_strict"
        ],
        accuracy_value=ap40["car_3d_ap40_moderate_strict"],
        accuracy_rank=1,
        ap40=ap40,
        benchmark_result_id=(f"benchmark-{slug}" if complete else None),
        runtime_scope=("end_to_end_ms" if complete else None),
        runtime_statistic=("p95_ms" if complete else None),
        runtime_value=(15.0 if complete else None),
        runtime_rank=(1 if complete else None),
        latency_statistics=(
            {
                "prediction_ms": {
                    "p50_ms": 10.0,
                    "p95_ms": 11.0,
                    "p99_ms": 12.0,
                },
                "end_to_end_ms": {
                    "p50_ms": 13.0,
                    "p95_ms": 15.0,
                    "p99_ms": 17.0,
                },
            }
            if complete
            else None
        ),
        peak_memory_allocated_bytes=(256 * 1024**2 if complete else None),
        peak_memory_reserved_bytes=(300 * 1024**2 if complete else None),
        checkpoint_size_bytes=(40 * 1024**2 if complete else None),
        meets_20hz=(True if complete else None),
    )


def test_plotting_import_does_not_import_matplotlib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in tuple(sys.modules):
        if name == "lidar_model_selection.plotting" or name.startswith("matplotlib"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    importlib.import_module("lidar_model_selection.plotting")
    assert "matplotlib" not in sys.modules


def test_complete_rows_render_all_legacy_evidence_views(tmp_path: Path) -> None:
    outputs = plot_comparison((_row("alpha", "1"),), tmp_path / "figures")
    assert {path.name for path in outputs} == {
        "accuracy_3d_ap40.png",
        "accuracy_bev_ap40.png",
        "latency_percentiles.png",
        "accuracy_vs_latency.png",
        "peak_gpu_memory.png",
        "checkpoint_size.png",
        "comparison_table.png",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)


def test_sparse_resolved_evidence_skips_unsupported_plots(tmp_path: Path) -> None:
    outputs = plot_comparison(
        (_row("historical", "2", complete=False),), tmp_path / "figures"
    )
    assert tuple(path.name for path in outputs) == ("comparison_table.png",)


def test_plotting_rejects_unresolved_or_empty_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        plot_comparison((), tmp_path)
    with pytest.raises(ValueError, match="ComparisonRow"):
        plot_comparison((object(),), tmp_path)  # type: ignore[arg-type]
