"""Render already-resolved comparison evidence.

This module deliberately knows nothing about runs, result selection, checkpoint
discovery, or compatibility.  Its sole input is a validated comparison report.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .comparison import ComparisonRow

__all__ = ("plot_comparison",)

_THREE_D = (
    "car_3d_ap40_easy_strict",
    "car_3d_ap40_moderate_strict",
    "car_3d_ap40_hard_strict",
)
_BEV = (
    "car_bev_ap40_easy_strict",
    "car_bev_ap40_moderate_strict",
    "car_bev_ap40_hard_strict",
)
_PERCENTILES = ("p50_ms", "p95_ms", "p99_ms")


def _pyplot() -> Any:
    """Import Matplotlib only when rendering is explicitly requested."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


def _finish(figure: Any, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    import matplotlib.pyplot as pyplot

    pyplot.close(figure)
    return output


def _grouped_bars(
    pyplot: Any,
    rows: Sequence[ComparisonRow],
    fields: Sequence[str],
    labels: Sequence[str],
    *,
    title: str,
    ylabel: str,
    output: Path,
    value: Callable[[ComparisonRow, str], float | None],
    threshold: float | None = None,
) -> Path | None:
    eligible = [row for row in rows if all(value(row, field) is not None for field in fields)]
    if not eligible:
        return None
    figure, axis = pyplot.subplots(figsize=(max(8.0, len(eligible) * 1.5), 6.0))
    width = 0.8 / len(fields)
    centers = list(range(len(eligible)))
    for index, (field, label) in enumerate(zip(fields, labels)):
        offset = (index - (len(fields) - 1) / 2) * width
        axis.bar(
            [center + offset for center in centers],
            [value(row, field) for row in eligible],
            width=width,
            label=label,
        )
    if threshold is not None:
        axis.axhline(threshold, linestyle="--", linewidth=1.4, label="20 Hz (50 ms)")
    axis.set_xticks(centers, [row.slug for row in eligible], rotation=20, ha="right")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_ylim(bottom=0.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    return _finish(figure, output)


def _simple_bars(
    pyplot: Any,
    rows: Sequence[ComparisonRow],
    *,
    value: Callable[[ComparisonRow], float | None],
    title: str,
    ylabel: str,
    output: Path,
) -> Path | None:
    evidence = [(row, value(row)) for row in rows]
    eligible = [(row, number) for row, number in evidence if number is not None]
    if not eligible:
        return None
    figure, axis = pyplot.subplots(figsize=(max(8.0, len(eligible) * 1.5), 6.0))
    axis.bar(range(len(eligible)), [number for _, number in eligible])
    axis.set_xticks(
        range(len(eligible)),
        [row.slug for row, _ in eligible],
        rotation=20,
        ha="right",
    )
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_ylim(bottom=0.0)
    axis.grid(axis="y", alpha=0.25)
    return _finish(figure, output)


def _accuracy_latency(
    pyplot: Any, rows: Sequence[ComparisonRow], output: Path
) -> Path | None:
    metric = "car_3d_ap40_moderate_strict"
    eligible = [
        row
        for row in rows
        if metric in row.ap40
        and row.latency_statistics is not None
        and "p95_ms" in row.latency_statistics.get("end_to_end_ms", {})
    ]
    if not eligible:
        return None
    latency = [row.latency_statistics["end_to_end_ms"]["p95_ms"] for row in eligible]  # type: ignore[index]
    accuracy = [row.ap40[metric] for row in eligible]
    figure, axis = pyplot.subplots(figsize=(9.0, 6.0))
    axis.scatter(latency, accuracy, s=55)
    for row, x_value, y_value in zip(eligible, latency, accuracy):
        axis.annotate(row.slug, (x_value, y_value), xytext=(6, 6), textcoords="offset points")
    axis.axvline(50.0, linestyle="--", linewidth=1.4, label="20 Hz (50 ms)")
    axis.set_title("CenterPoint accuracy versus end-to-end latency")
    axis.set_xlabel("End-to-end p95 latency (ms)")
    axis.set_ylabel("Car 3D AP40 Moderate strict")
    axis.set_xlim(left=0.0)
    axis.set_ylim(0.0, 100.0)
    axis.grid(alpha=0.25)
    axis.legend()
    return _finish(figure, output)


def _comparison_table(
    pyplot: Any, rows: Sequence[ComparisonRow], output: Path
) -> Path:
    headers = (
        "Run",
        "3D AP40 mod.",
        "BEV AP40 mod.",
        "E2E p95 (ms)",
        "Peak CUDA (MiB)",
        "Checkpoint (MiB)",
        "20 Hz",
    )

    def number(value: float | None, decimals: int = 2) -> str:
        return "N/A" if value is None else f"{value:.{decimals}f}"

    cells = []
    for row in rows:
        latency = None
        if row.latency_statistics is not None:
            latency = row.latency_statistics.get("end_to_end_ms", {}).get("p95_ms")
        cells.append(
            (
                row.slug,
                number(row.ap40.get("car_3d_ap40_moderate_strict")),
                number(row.ap40.get("car_bev_ap40_moderate_strict")),
                number(latency),
                number(
                    None
                    if row.peak_memory_allocated_bytes is None
                    else row.peak_memory_allocated_bytes / 1024**2,
                    1,
                ),
                number(
                    None
                    if row.checkpoint_size_bytes is None
                    else row.checkpoint_size_bytes / 1024**2,
                    1,
                ),
                "N/A" if row.meets_20hz is None else ("Yes" if row.meets_20hz else "No"),
            )
        )
    figure, axis = pyplot.subplots(figsize=(15.0, max(3.0, 0.55 * len(rows) + 2.0)))
    axis.axis("off")
    table = axis.table(cellText=cells, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    axis.set_title("CenterPoint accuracy and runtime comparison", pad=14)
    return _finish(figure, output)


def plot_comparison(
    rows: Sequence[ComparisonRow], output_dir: Path
) -> tuple[Path, ...]:
    """Render useful plots from validated, already-resolved comparison rows."""
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence of ComparisonRow values")
    if not rows or not all(isinstance(row, ComparisonRow) for row in rows):
        raise ValueError("rows must contain at least one ComparisonRow")
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")
    pyplot = _pyplot()
    rendered: list[Path | None] = [
        _grouped_bars(
            pyplot,
            rows,
            _THREE_D,
            ("Easy", "Moderate", "Hard"),
            title="CenterPoint Car 3D AP40 accuracy",
            ylabel="Car 3D AP40 strict",
            output=output_dir / "accuracy_3d_ap40.png",
            value=lambda row, field: row.ap40.get(field),
        ),
        _grouped_bars(
            pyplot,
            rows,
            _BEV,
            ("Easy", "Moderate", "Hard"),
            title="CenterPoint Car BEV AP40 accuracy",
            ylabel="Car BEV AP40 strict",
            output=output_dir / "accuracy_bev_ap40.png",
            value=lambda row, field: row.ap40.get(field),
        ),
        _grouped_bars(
            pyplot,
            rows,
            _PERCENTILES,
            ("p50", "p95", "p99"),
            title="CenterPoint end-to-end latency percentiles",
            ylabel="Latency (ms)",
            output=output_dir / "latency_percentiles.png",
            value=lambda row, field: (
                None
                if row.latency_statistics is None
                else row.latency_statistics.get("end_to_end_ms", {}).get(field)
            ),
            threshold=50.0,
        ),
        _accuracy_latency(pyplot, rows, output_dir / "accuracy_vs_latency.png"),
        _simple_bars(
            pyplot,
            rows,
            value=lambda row: (
                None
                if row.peak_memory_allocated_bytes is None
                else row.peak_memory_allocated_bytes / 1024**2
            ),
            title="CenterPoint peak allocated CUDA memory",
            ylabel="Peak allocated CUDA memory (MiB)",
            output=output_dir / "peak_gpu_memory.png",
        ),
        _simple_bars(
            pyplot,
            rows,
            value=lambda row: (
                None
                if row.checkpoint_size_bytes is None
                else row.checkpoint_size_bytes / 1024**2
            ),
            title="CenterPoint checkpoint size",
            ylabel="Checkpoint size (MiB)",
            output=output_dir / "checkpoint_size.png",
        ),
        _comparison_table(pyplot, rows, output_dir / "comparison_table.png"),
    ]
    return tuple(path for path in rendered if path is not None)
