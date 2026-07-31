"""Plot CenterPoint accuracy and single-GPU benchmark summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION_SUMMARY = (
    REPOSITORY_ROOT / "research" / "evaluations" / "summary.csv"
)
DEFAULT_BENCHMARK_SUMMARY = (
    REPOSITORY_ROOT / "research" / "benchmarks" / "summary.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "research" / "reports" / "figures"
)

MODEL_ORDER = (
    "pillar02",
    "pillar02_dcn",
    "voxel01",
    "voxel01_dcn",
    "voxel0075",
    "voxel0075_dcn",
)

ACCURACY_3D_FIELDS = (
    "car_3d_ap40_easy_strict",
    "car_3d_ap40_moderate_strict",
    "car_3d_ap40_hard_strict",
)
ACCURACY_BEV_FIELDS = (
    "car_bev_ap40_easy_strict",
    "car_bev_ap40_moderate_strict",
    "car_bev_ap40_hard_strict",
)
ACCURACY_FIELDS = (*ACCURACY_3D_FIELDS, *ACCURACY_BEV_FIELDS)
LATENCY_FIELDS = (
    "end_to_end_p50_ms",
    "end_to_end_p95_ms",
    "end_to_end_p99_ms",
)
BENCHMARK_FIELDS = (
    *LATENCY_FIELDS,
    "peak_memory_allocated_mb",
    "checkpoint_size_mb",
)

EVALUATION_REQUIRED_COLUMNS = {
    "model",
    "test_success",
    *ACCURACY_FIELDS,
}
BENCHMARK_REQUIRED_COLUMNS = {
    "model",
    "success",
    *BENCHMARK_FIELDS,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate CenterPoint accuracy and single-GPU benchmark plots."
        )
    )
    parser.add_argument(
        "--evaluation-summary",
        type=Path,
        default=DEFAULT_EVALUATION_SUMMARY,
        help=(
            "Evaluation aggregate CSV "
            "(default: research/evaluations/summary.csv)."
        ),
    )
    parser.add_argument(
        "--benchmark-summary",
        type=Path,
        default=DEFAULT_BENCHMARK_SUMMARY,
        help=(
            "Benchmark aggregate CSV "
            "(default: research/benchmarks/summary.csv)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "PNG output directory "
            "(default: research/reports/figures)."
        ),
    )
    return parser


def warning(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def parse_success(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.strip().str.lower()
    parsed = normalized.map(
        {
            "true": True,
            "1": True,
            "1.0": True,
            "yes": True,
            "false": False,
            "0": False,
            "0.0": False,
            "no": False,
        }
    )
    return parsed.astype("boolean")


def read_summary(
    path: Path,
    source: str,
    required_columns: set[str],
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"{source} summary does not exist: {display_path(path)}"
        )

    frame = pd.read_csv(path)
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"{source} summary {display_path(path)} is missing required "
            f"columns: {', '.join(missing_columns)}"
        )

    frame["model"] = frame["model"].astype("string")
    empty_model = (
        frame["model"].isna()
        | frame["model"].str.strip().eq("").fillna(True)
    )
    if empty_model.any():
        csv_row = int(frame.index[empty_model][0]) + 2
        raise ValueError(
            f"{source} summary {display_path(path)} has an empty model "
            f"at CSV row {csv_row}"
        )

    duplicates = frame["model"].duplicated(keep=False)
    if duplicates.any():
        models = sorted(frame.loc[duplicates, "model"].unique())
        raise ValueError(
            f"{source} summary {display_path(path)} contains duplicate "
            f"model rows: {', '.join(models)}"
        )
    return frame


def validate_summary_values(
    frame: pd.DataFrame,
    source: str,
    success_column: str,
    metric_columns: Sequence[str],
) -> None:
    parsed_success = parse_success(frame[success_column])
    invalid_success = parsed_success.isna()
    if invalid_success.any():
        row = frame.loc[
            invalid_success,
            ["model", success_column],
        ].iloc[0]
        raise ValueError(
            f"{source} result for {row['model']!r} has invalid "
            f"{success_column}: {row[success_column]!r}"
        )
    frame[success_column] = parsed_success

    columns = list(metric_columns)
    numeric_text = frame[columns].astype("string")
    frame[columns] = numeric_text.apply(
        pd.to_numeric,
        errors="coerce",
    )
    successful = frame[success_column].eq(True).to_numpy(dtype=bool)
    for column in columns:
        finite = np.isfinite(
            frame[column].to_numpy(dtype=float, na_value=np.nan)
        )
        invalid_numeric = successful & ~finite
        if invalid_numeric.any():
            model = frame.loc[invalid_numeric, "model"].iloc[0]
            raise ValueError(
                f"{source} result for {model!r} has invalid or missing "
                f"required numeric column {column!r}"
            )


def warn_result_issues(data: pd.DataFrame) -> None:
    for model in data.loc[~data["evaluation_present"], "model"]:
        warning(f"evaluation summary has no row for {model}")
    for model in data.loc[~data["benchmark_present"], "model"]:
        warning(f"benchmark summary has no row for {model}")

    failed_evaluations = data[
        data["evaluation_present"]
        & data["evaluation_success"].eq(False).fillna(False)
    ]
    for row in failed_evaluations.itertuples(index=False):
        detail = row.error_message
        if pd.isna(detail) or not str(detail).strip():
            detail = "no error message recorded"
        warning(f"evaluation failed for {row.model}: {detail}")

    failed_benchmarks = data[
        data["benchmark_present"]
        & data["benchmark_success"].eq(False).fillna(False)
    ]
    for row in failed_benchmarks.itertuples(index=False):
        detail = row.error
        if pd.isna(detail) or not str(detail).strip():
            detail = "no error message recorded"
        warning(f"benchmark failed for {row.model}: {detail}")


def load_results(
    evaluation_path: Path,
    benchmark_path: Path,
) -> pd.DataFrame:
    evaluation = read_summary(
        evaluation_path,
        "evaluation",
        EVALUATION_REQUIRED_COLUMNS,
    )
    benchmark = read_summary(
        benchmark_path,
        "benchmark",
        BENCHMARK_REQUIRED_COLUMNS,
    )

    models = pd.concat(
        [evaluation["model"], benchmark["model"]],
        ignore_index=True,
    ).drop_duplicates()
    unknown_models = models[~models.isin(MODEL_ORDER)]
    for model in unknown_models:
        warning(f"unknown model {model!r} is unsupported and was excluded")
    evaluation = evaluation[evaluation["model"].isin(MODEL_ORDER)].copy()
    benchmark = benchmark[benchmark["model"].isin(MODEL_ORDER)].copy()

    validate_summary_values(
        evaluation,
        "evaluation",
        "test_success",
        ACCURACY_FIELDS,
    )
    validate_summary_values(
        benchmark,
        "benchmark",
        "success",
        BENCHMARK_FIELDS,
    )

    if "error_message" not in evaluation:
        evaluation["error_message"] = pd.NA
    if "error" not in benchmark:
        benchmark["error"] = pd.NA

    evaluation = evaluation[
        ["model", "test_success", "error_message", *ACCURACY_FIELDS]
    ].rename(columns={"test_success": "evaluation_success"})
    benchmark = benchmark[
        ["model", "success", "error", *BENCHMARK_FIELDS]
    ].rename(columns={"success": "benchmark_success"})

    merged = evaluation.merge(
        benchmark,
        on="model",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    expected = pd.DataFrame({"model": MODEL_ORDER})
    data = expected.merge(
        merged,
        on="model",
        how="left",
        validate="one_to_one",
    )
    data["model"] = pd.Categorical(
        data["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )
    data = data.sort_values("model").reset_index(drop=True)
    data["evaluation_present"] = data["_merge"].isin(
        ("left_only", "both")
    )
    data["benchmark_present"] = data["_merge"].isin(
        ("right_only", "both")
    )
    data = data.drop(columns="_merge")
    warn_result_issues(data)

    evaluation_success = (
        data["evaluation_success"].eq(True).fillna(False)
    )
    benchmark_success = data["benchmark_success"].eq(True).fillna(False)
    data.loc[~evaluation_success, list(ACCURACY_FIELDS)] = np.nan
    data.loc[~benchmark_success, list(BENCHMARK_FIELDS)] = np.nan

    data["meets_20hz"] = pd.Series(
        pd.NA,
        index=data.index,
        dtype="boolean",
    )
    data.loc[benchmark_success, "meets_20hz"] = data.loc[
        benchmark_success,
        "end_to_end_p95_ms",
    ].le(50.0)
    return data


def annotate_bars(axis, bars, values: np.ndarray, decimals: int) -> None:
    labels = [f"{value:.{decimals}f}" for value in values]
    axis.bar_label(
        bars,
        labels=labels,
        padding=3,
        rotation=90,
        fontsize=8,
    )


def finish_bar_axis(
    axis,
    positions: np.ndarray,
    model_labels: pd.Series,
    title: str,
    y_label: str,
    *,
    show_legend: bool,
) -> None:
    axis.set_title(title)
    axis.set_xlabel("CenterPoint model")
    axis.set_ylabel(y_label)
    axis.set_xticks(
        positions,
        model_labels,
        rotation=18,
        ha="right",
    )
    axis.set_ylim(bottom=0.0)
    axis.margins(y=0.15)
    axis.grid(axis="y", alpha=0.25)
    if show_legend:
        axis.legend()


def save_figure(figure, path: Path) -> None:
    try:
        figure.tight_layout()
        figure.savefig(path, dpi=180)
    finally:
        plt.close(figure)
    print(f"Saved: {display_path(path)}")


def grouped_bar_plot(
    data: pd.DataFrame,
    fields: Sequence[str],
    series_labels: Sequence[str],
    title: str,
    y_label: str,
    output_path: Path,
    *,
    decimals: int,
    requirement_line: float | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(13, 7))
    positions = np.arange(len(data), dtype=float)
    width = 0.8 / len(fields)

    for index, (field, label) in enumerate(
        zip(fields, series_labels)
    ):
        values = data[field].to_numpy(dtype=float)
        offset = (index - (len(fields) - 1) / 2) * width
        bars = axis.bar(
            positions + offset,
            values,
            width,
            label=label,
        )
        annotate_bars(axis, bars, values, decimals)

    if requirement_line is not None:
        axis.axhline(
            requirement_line,
            linestyle="--",
            linewidth=1.5,
            label="20 Hz requirement (50 ms)",
        )

    finish_bar_axis(
        axis,
        positions,
        data["model"].astype("string"),
        title,
        y_label,
        show_legend=True,
    )
    save_figure(figure, output_path)


def simple_bar_plot(
    data: pd.DataFrame,
    field: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(12, 7))
    positions = np.arange(len(data), dtype=float)
    values = data[field].to_numpy(dtype=float)
    bars = axis.bar(positions, values)
    annotate_bars(axis, bars, values, decimals=1)

    finish_bar_axis(
        axis,
        positions,
        data["model"].astype("string"),
        title,
        y_label,
        show_legend=False,
    )
    save_figure(figure, output_path)


def plot_accuracy_vs_latency(
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(12, 7))
    latency = data["end_to_end_p95_ms"].to_numpy(dtype=float)
    accuracy = data[
        "car_3d_ap40_moderate_strict"
    ].to_numpy(dtype=float)
    axis.scatter(latency, accuracy, s=55)

    labels = data["model"].astype("string")
    for index, (model, x_value, y_value) in enumerate(
        zip(labels, latency, accuracy)
    ):
        axis.annotate(
            model,
            xy=(x_value, y_value),
            xytext=(6, 6 if index % 2 == 0 else -12),
            textcoords="offset points",
            fontsize=9,
        )

    axis.axvline(
        50.0,
        linestyle="--",
        linewidth=1.5,
        label="20 Hz requirement (50 ms)",
    )
    axis.set_title("CenterPoint accuracy versus end-to-end latency")
    axis.set_xlabel("End-to-end p95 latency (ms)")
    axis.set_ylabel("Car 3D AP40 Moderate strict")
    axis.set_xlim(left=0.0)
    axis.set_ylim(0.0, 100.0)
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(figure, output_path)


def plot_comparison_table(
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    display = data[
        [
            "model",
            "car_3d_ap40_moderate_strict",
            "car_bev_ap40_moderate_strict",
            "end_to_end_p95_ms",
            "peak_memory_allocated_mb",
            "checkpoint_size_mb",
            "meets_20hz",
        ]
    ].copy()
    display.columns = (
        "Model",
        "3D AP40 Moderate",
        "BEV AP40 Moderate",
        "E2E p95",
        "Peak CUDA memory",
        "Checkpoint size",
        "Meets 20 Hz",
    )
    display["Model"] = display["Model"].astype("string")
    for column in ("3D AP40 Moderate", "BEV AP40 Moderate", "E2E p95"):
        display[column] = display[column].map(
            lambda value: "N/A" if pd.isna(value) else f"{value:.2f}"
        )
    for column in ("Peak CUDA memory", "Checkpoint size"):
        display[column] = display[column].map(
            lambda value: "N/A" if pd.isna(value) else f"{value:.1f}"
        )
    display["Meets 20 Hz"] = (
        display["Meets 20 Hz"]
        .map({True: "Yes", False: "No"})
        .fillna("N/A")
    )

    figure, axis = plt.subplots(figsize=(16, 5.2))
    axis.axis("off")
    table = axis.table(
        cellText=display.to_numpy(),
        colLabels=display.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    axis.set_title(
        "CenterPoint accuracy and single-GPU runtime comparison",
        pad=14,
    )
    save_figure(figure, output_path)


def generate_plots(
    evaluation_plot_data: pd.DataFrame,
    benchmark_plot_data: pd.DataFrame,
    comparison_data: pd.DataFrame,
    recommendation_data: pd.DataFrame,
    output_dir: Path,
) -> None:
    grouped_bar_plot(
        evaluation_plot_data,
        ACCURACY_3D_FIELDS,
        ("Easy", "Moderate", "Hard"),
        "CenterPoint Car 3D AP40 accuracy",
        "Car 3D AP40 strict",
        output_dir / "accuracy_3d_ap40.png",
        decimals=2,
    )
    grouped_bar_plot(
        evaluation_plot_data,
        ACCURACY_BEV_FIELDS,
        ("Easy", "Moderate", "Hard"),
        "CenterPoint Car BEV AP40 accuracy",
        "Car BEV AP40 strict",
        output_dir / "accuracy_bev_ap40.png",
        decimals=2,
    )
    grouped_bar_plot(
        benchmark_plot_data,
        LATENCY_FIELDS,
        ("p50", "p95", "p99"),
        "CenterPoint end-to-end latency percentiles",
        "Latency (ms)",
        output_dir / "latency_percentiles.png",
        decimals=1,
        requirement_line=50.0,
    )
    plot_accuracy_vs_latency(
        recommendation_data,
        output_dir / "accuracy_vs_latency.png",
    )
    simple_bar_plot(
        benchmark_plot_data,
        "peak_memory_allocated_mb",
        "CenterPoint peak allocated CUDA memory",
        "Peak allocated CUDA memory (MiB)",
        output_dir / "peak_gpu_memory.png",
    )
    simple_bar_plot(
        benchmark_plot_data,
        "checkpoint_size_mb",
        "CenterPoint checkpoint size",
        "Checkpoint size (MiB)",
        output_dir / "checkpoint_size.png",
    )
    plot_comparison_table(
        comparison_data,
        output_dir / "comparison_table.png",
    )


def print_recommendation(
    benchmark_plot_data: pd.DataFrame,
    recommendation_data: pd.DataFrame,
) -> None:
    eligible_benchmarks = benchmark_plot_data[
        benchmark_plot_data["meets_20hz"].eq(True)
    ].sort_values(
        ["end_to_end_p95_ms", "model"],
        ascending=[True, True],
    )

    print("\nModels satisfying end-to-end p95 <= 50 ms:")
    if eligible_benchmarks.empty:
        print("  None")
    else:
        for row in eligible_benchmarks.itertuples(index=False):
            print(f"  {row.model}: {row.end_to_end_p95_ms:.2f} ms")

    eligible_by_accuracy = recommendation_data[
        recommendation_data["meets_20hz"].eq(True)
    ].sort_values(
        ["car_3d_ap40_moderate_strict", "model"],
        ascending=[False, True],
    )
    print("Eligible models ranked by Car 3D AP40 Moderate strict:")
    if eligible_by_accuracy.empty:
        print("  None with successful accuracy data")
        print("Highest-accuracy eligible model: unavailable")
    else:
        for rank, row in enumerate(
            eligible_by_accuracy.itertuples(index=False),
            start=1,
        ):
            print(
                f"  {rank}. {row.model}: "
                f"{row.car_3d_ap40_moderate_strict:.4f}"
            )
        best = eligible_by_accuracy.iloc[0]
        print(
            "Highest-accuracy eligible model: "
            f"{best['model']} "
            f"({best['car_3d_ap40_moderate_strict']:.4f})"
        )

    print("Fastest successful benchmark model (fallback):")
    if benchmark_plot_data.empty:
        print("  Unavailable")
    else:
        fastest = benchmark_plot_data.sort_values(
            ["end_to_end_p95_ms", "model"],
            ascending=[True, True],
        ).iloc[0]
        print(
            f"  {fastest['model']}: "
            f"{fastest['end_to_end_p95_ms']:.2f} ms p95"
        )

    print("Final selection still requires:")
    print("  - recorded JKK qualitative testing")
    print("  - actual target-hardware testing")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluation_path = args.evaluation_summary.resolve()
    benchmark_path = args.benchmark_summary.resolve()
    output_dir = args.output_dir.resolve()

    try:
        comparison_data = load_results(
            evaluation_path,
            benchmark_path,
        )
        if output_dir.exists() and not output_dir.is_dir():
            raise ValueError(
                f"output path is not a directory: "
                f"{display_path(output_dir)}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    evaluation_plot_data = comparison_data[
        comparison_data["evaluation_success"].eq(True)
        & comparison_data[list(ACCURACY_FIELDS)].notna().all(axis=1)
    ]
    benchmark_plot_data = comparison_data[
        comparison_data["benchmark_success"].eq(True)
        & comparison_data[list(BENCHMARK_FIELDS)].notna().all(axis=1)
    ]
    recommendation_data = comparison_data[
        comparison_data["evaluation_success"].eq(True)
        & comparison_data["benchmark_success"].eq(True)
        & comparison_data[
            [
                "car_3d_ap40_moderate_strict",
                "end_to_end_p95_ms",
            ]
        ].notna().all(axis=1)
    ]

    try:
        generate_plots(
            evaluation_plot_data,
            benchmark_plot_data,
            comparison_data,
            recommendation_data,
            output_dir,
        )
    except OSError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print_recommendation(
        benchmark_plot_data,
        recommendation_data,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
