#!/usr/bin/env python3
"""Értéket adó prezentációs ábrák renderelése immutable JSON-forrásokból.

A program kizárólag meglévő, feloldott összehasonlító JSON-okat és a tartós
multiclass run manifest/result fájljait olvassa. Nem indít tanítást,
kiértékelést, benchmarkot vagy modellbetöltést.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

_cache_root = Path(tempfile.gettempdir()) / "lidar-centerpoint-presentation-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root / "xdg"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
SIX_END_TO_END = (
    ROOT / "research/reports/20260827-six-model-20epoch/fresh-end-to-end-p95.json"
)
SIX_PREDICTION = (
    ROOT / "research/reports/20260827-six-model-20epoch/fresh-prediction-p95.json"
)
FINALISTS = (
    ROOT / "research/reports/20260902-finalists-duration30/paired-end-to-end-p95.json"
)

RUN_ID = "20260902T125737Z-pillar02-multiclass-faa487143efbe3dba808d9ac"
EVALUATION_ID = "20260902T195616343969Z-evaluation-513fc499ed4e49415ee237c2"
BENCHMARK_ID = "20260902T195651982215Z-benchmark-0001486e79f9c7dd48f28e10"
CONFIG_SHA256 = "b31131058eb44367a6d7daa7a3ee0620d41cf3324fdfd701f1f41d402a50231d"
CHECKPOINT_REFERENCE = (
    "training/best_Kitti metric_pred_instances_3d_KITTI_Car_3D_AP40_"
    "moderate_strict_epoch_55.pth"
)
CHECKPOINT_SIZE = 32_452_134
CHECKPOINT_SHA256 = "cf62f3c99ce8ebdbb96eaa467cc44c3bde0a152aa25470ce1cfd11b8ac7c7427"

MODEL_ORDER = (
    "pillar02",
    "pillar02-dcn",
    "voxel01",
    "voxel01-dcn",
    "voxel0075",
    "voxel0075-dcn",
)
MODEL_LABELS = {
    "pillar02": "Pillar02",
    "pillar02-dcn": "Pillar02 DCN",
    "voxel01": "Voxel01",
    "voxel01-dcn": "Voxel01 DCN",
    "voxel0075": "Voxel0075",
    "voxel0075-dcn": "Voxel0075 DCN",
}
MODEL_COLORS = {
    "pillar02": "#F28E2B",
    "pillar02-dcn": "#EDC948",
    "voxel01": "#59A14F",
    "voxel01-dcn": "#8CD17D",
    "voxel0075": "#4E79A7",
    "voxel0075-dcn": "#76B7B2",
}
CLASSES = ("Car", "Pedestrian", "Cyclist")
DIFFICULTIES = ("easy", "moderate", "hard")
DIFFICULTY_LABELS = ("Easy", "Moderate", "Hard")

FIGURE_PARETO = "hatmodell_pareto_3d_ap40_p95"
FIGURE_LATENCY = "hatmodell_p95_meresi_scope"
FIGURE_FINALISTS = "finalistak_eredmenymatrix"
FIGURE_MULTICLASS = "pillar02_tobbosztalyos_ap40_matrix"
FIGURE_NAMES = (
    FIGURE_PARETO,
    FIGURE_LATENCY,
    FIGURE_FINALISTS,
    FIGURE_MULTICLASS,
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"A JSON gyökéreleme nem objektum: {path}")
    return payload


def _required(value: Any, description: str) -> Any:
    if value is None:
        raise ValueError(f"Hiányzó kötelező adat: {description}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows_by_slug(payload: dict[str, Any], source: Path) -> dict[str, dict[str, Any]]:
    rows = _required(payload.get("rows"), f"rows ({source})")
    if not isinstance(rows, list):
        raise ValueError(f"A rows mező nem lista: {source}")
    indexed = {str(_required(row.get("slug"), f"slug ({source})")): row for row in rows}
    if set(indexed) != set(MODEL_ORDER) or len(indexed) != len(MODEL_ORDER):
        raise ValueError(f"A hatmodell-forrás modellkészlete hibás: {source}")
    return indexed


def _metric(row: dict[str, Any], key: str) -> float:
    return float(_required(_required(row.get("ap40"), "ap40").get(key), key))


def _latency(row: dict[str, Any], scope: str) -> float:
    statistics = _required(row.get("latency_statistics"), "latency_statistics")
    return float(_required(_required(statistics.get(scope), scope).get("p95_ms"), f"{scope}.p95_ms"))


def _binding_matches(payload: dict[str, Any], *, result_id: str, result_type: str) -> bool:
    binding = payload.get("binding")
    return bool(
        payload.get("result_id") == result_id
        and payload.get("result_type") == result_type
        and payload.get("status") == "succeeded"
        and isinstance(binding, dict)
        and binding.get("run_id") == RUN_ID
        and binding.get("config_sha256") == CONFIG_SHA256
        and binding.get("checkpoint_sha256") == CHECKPOINT_SHA256
    )


def _validate_multiclass(
    run_root: Path,
    manifest: dict[str, Any],
    evaluation: dict[str, Any],
    benchmark: dict[str, Any],
) -> None:
    if manifest.get("run_id") != RUN_ID or manifest.get("origin") != "native":
        raise ValueError("A multiclass manifest nem a várt natív futáshoz tartozik.")
    config = _required(manifest.get("config"), "manifest.config")
    if config.get("sha256") != CONFIG_SHA256:
        raise ValueError("A multiclass config SHA-256 értéke eltér.")
    outputs = _required(
        _required(_required(manifest.get("training"), "training").get("outputs"), "outputs"),
        "training.outputs",
    )
    selected = _required(outputs.get("selected_checkpoint"), "selected_checkpoint")
    if (
        selected.get("path") != CHECKPOINT_REFERENCE
        or selected.get("size_bytes") != CHECKPOINT_SIZE
        or selected.get("sha256") != CHECKPOINT_SHA256
    ):
        raise ValueError("A kiválasztott multiclass checkpoint manifestkötése eltér.")
    checkpoint = run_root / CHECKPOINT_REFERENCE
    if not checkpoint.is_file() or checkpoint.stat().st_size != CHECKPOINT_SIZE:
        raise ValueError("A tartós multiclass checkpoint hiányzik vagy hibás méretű.")
    if _sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("A tartós multiclass checkpoint SHA-256 értéke eltér.")
    if not _binding_matches(evaluation, result_id=EVALUATION_ID, result_type="evaluation"):
        raise ValueError("A multiclass evaluation/result.json kötése eltér.")
    if not _binding_matches(benchmark, result_id=BENCHMARK_ID, result_type="benchmark"):
        raise ValueError("A multiclass benchmark/result.json kötése eltér.")


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 21,
            "axes.labelsize": 17,
            "legend.fontsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "svg.hashsalt": "lidar-centerpoint-presentation-v1",
        }
    )


def _save(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "lidar-centerpoint render_presentation_assets_final.py", "Date": None}
    figure.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight", metadata=metadata)
    svg_path = output_dir / f"{stem}.svg"
    figure.savefig(svg_path, bbox_inches="tight", metadata=metadata)
    plt.close(figure)
    # A Matplotlib SVG backend a pathadatok sorvégein szóközt hagy. A
    # normalizálás tartalmat nem változtat, viszont determinisztikus kimenetet
    # és tiszta ``git diff --check`` eredményt ad.
    normalized = "\n".join(
        line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()
    )
    svg_path.write_text(normalized + "\n", encoding="utf-8")


def _pareto_slugs(rows: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    frontier = []
    for slug in MODEL_ORDER:
        x = _latency(rows[slug], "end_to_end_ms")
        y = _metric(rows[slug], "car_3d_ap40_moderate_strict")
        dominated = any(
            other != slug
            and _latency(rows[other], "end_to_end_ms") <= x
            and _metric(rows[other], "car_3d_ap40_moderate_strict") >= y
            and (
                _latency(rows[other], "end_to_end_ms") < x
                or _metric(rows[other], "car_3d_ap40_moderate_strict") > y
            )
            for other in MODEL_ORDER
        )
        if not dominated:
            frontier.append(slug)
    return tuple(sorted(frontier, key=lambda item: _latency(rows[item], "end_to_end_ms")))


def _render_pareto(context: dict[str, Any], output_dir: Path) -> None:
    rows = context["end_to_end"]
    frontier = _pareto_slugs(rows)
    figure, axis = plt.subplots(figsize=(13.333, 7.5), layout="constrained")
    axis.plot(
        [_latency(rows[slug], "end_to_end_ms") for slug in frontier],
        [_metric(rows[slug], "car_3d_ap40_moderate_strict") for slug in frontier],
        color="#2F5597",
        linewidth=2.2,
        linestyle="--",
        label="Pareto frontier",
        zorder=2,
    )
    offsets = {
        "pillar02": (8, 8),
        "pillar02-dcn": (8, -8),
        "voxel01": (8, 8),
        "voxel01-dcn": (8, -22),
        "voxel0075": (8, 8),
        "voxel0075-dcn": (8, -22),
    }
    for slug in MODEL_ORDER:
        latency = _latency(rows[slug], "end_to_end_ms")
        accuracy = _metric(rows[slug], "car_3d_ap40_moderate_strict")
        axis.scatter(
            latency,
            accuracy,
            s=250,
            color=MODEL_COLORS[slug],
            edgecolor="#1F1F1F" if slug in frontier else "white",
            linewidth=2.0 if slug in frontier else 1.2,
            zorder=3,
        )
        axis.annotate(
            MODEL_LABELS[slug],
            (latency, accuracy),
            xytext=offsets[slug],
            textcoords="offset points",
            fontsize=12,
        )
    axis.axvline(50, color="#E15759", linewidth=1.8, linestyle=":", label="Real-time target (50 ms)")
    axis.set_xlim(12.5, 51.5)
    axis.set_ylim(53.6, 66.75)
    axis.set_xlabel("End-to-end p95 latency [ms]")
    axis.set_ylabel("KITTI Car 3D AP40 (moderate, strict) [%]")
    axis.set_title("Accuracy–latency trade-off — six 20-epoch Car-only models")
    axis.legend(loc="lower right")
    axis.set_axisbelow(True)
    _save(figure, output_dir, FIGURE_PARETO)


def _render_latency(context: dict[str, Any], output_dir: Path) -> None:
    prediction = context["prediction"]
    end_to_end = context["end_to_end"]
    positions = np.arange(len(MODEL_ORDER))
    width = 0.37
    pred = [_latency(prediction[slug], "prediction_ms") for slug in MODEL_ORDER]
    e2e = [_latency(end_to_end[slug], "end_to_end_ms") for slug in MODEL_ORDER]
    figure, axis = plt.subplots(figsize=(13.333, 7.5), layout="constrained")
    pred_bars = axis.bar(positions - width / 2, pred, width, label="Prediction latency (p95)", color="#4E79A7")
    e2e_bars = axis.bar(positions + width / 2, e2e, width, label="End-to-end latency (p95)", color="#F28E2B")
    axis.bar_label(pred_bars, fmt="%.2f", padding=3, fontsize=10)
    axis.bar_label(e2e_bars, fmt="%.2f", padding=3, fontsize=10)
    axis.axhline(50, color="#E15759", linewidth=1.8, linestyle=":", label="Real-time target (50 ms)")
    axis.set_xticks(positions, [MODEL_LABELS[slug] for slug in MODEL_ORDER], rotation=17)
    axis.set_ylabel("Latency [ms]")
    axis.set_title("Prediction and end-to-end p95 latency — six 20-epoch Car-only models")
    axis.set_ylim(0, 54)
    axis.legend(loc="upper left", ncols=1)
    axis.set_axisbelow(True)
    _save(figure, output_dir, FIGURE_LATENCY)


def _finalist_key(row: dict[str, Any]) -> tuple[str, int]:
    run_id = str(_required(row.get("run_id"), "finalista run_id"))
    architecture = "Voxel0075" if "voxel0075" in run_id else "Pillar02" if "pillar02" in run_id else ""
    if not architecture:
        raise ValueError(f"Ismeretlen finalista futás: {run_id}")
    return architecture, 30 if "duration30" in run_id else 20


def _format_value(value: float) -> str:
    return f"{value:.2f}"


def _render_finalists(context: dict[str, Any], output_dir: Path) -> None:
    raw_rows = _required(context["finalists"].get("rows"), "finalista rows")
    indexed = {_finalist_key(row): row for row in raw_rows}
    expected = {(architecture, epoch) for architecture in ("Voxel0075", "Pillar02") for epoch in (20, 30)}
    if set(indexed) != expected or len(raw_rows) != 4:
        raise ValueError("A finalista forrás nem a két architektúra párosított 20/30 epochos futásait tartalmazza.")

    metric_specs = (
        ("3D AP40\nEasy", "accuracy", "car_3d_ap40_easy_strict"),
        ("3D AP40\nModerate", "accuracy", "car_3d_ap40_moderate_strict"),
        ("3D AP40\nHard", "accuracy", "car_3d_ap40_hard_strict"),
        ("BEV AP40\nEasy", "accuracy", "car_bev_ap40_easy_strict"),
        ("BEV AP40\nModerate", "accuracy", "car_bev_ap40_moderate_strict"),
        ("BEV AP40\nHard", "accuracy", "car_bev_ap40_hard_strict"),
        ("Prediction\np95 [ms]", "latency", "prediction_ms"),
        ("End-to-end\np95 [ms]", "latency", "end_to_end_ms"),
    )
    row_keys = (("Voxel0075", 20), ("Voxel0075", 30), ("Pillar02", 20), ("Pillar02", 30))
    values = np.zeros((4, len(metric_specs)), dtype=float)
    direction = np.zeros_like(values, dtype=int)
    annotations: list[list[str]] = [[""] * len(metric_specs) for _ in row_keys]
    for row_index, (architecture, epoch) in enumerate(row_keys):
        row = indexed[(architecture, epoch)]
        baseline = indexed[(architecture, 20)]
        for column, (_, kind, key) in enumerate(metric_specs):
            value = _metric(row, key) if kind == "accuracy" else _latency(row, key)
            reference = _metric(baseline, key) if kind == "accuracy" else _latency(baseline, key)
            values[row_index, column] = value
            if epoch == 20:
                annotations[row_index][column] = _format_value(value)
                continue
            delta = value - reference
            is_better = delta > 0 if kind == "accuracy" else delta < 0
            direction[row_index, column] = 1 if is_better else -1 if delta != 0 else 0
            annotations[row_index][column] = f"{_format_value(value)}\n({delta:+.2f})"

    figure, axis = plt.subplots(figsize=(13.333, 7.5))
    figure.subplots_adjust(left=0.15, right=0.82, top=0.70, bottom=0.18)
    axis.imshow(direction, cmap=ListedColormap(("#F4B6B2", "#ECECEC", "#B8D8BA")), vmin=-1, vmax=1, aspect="auto")
    for row_index in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row_index, annotations[row_index][column], ha="center", va="center", fontsize=11)
    axis.set_xticks(np.arange(len(metric_specs)), [item[0] for item in metric_specs])
    axis.set_yticks(
        np.arange(len(row_keys)),
        [f"{architecture} — {epoch} epochs" for architecture, epoch in row_keys],
    )
    axis.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    axis.set_xticks(np.arange(-0.5, len(metric_specs), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(row_keys), 1), minor=True)
    axis.grid(which="minor", color="white", linestyle="-", linewidth=2)
    axis.grid(which="major", visible=False)
    figure.suptitle(
        "Finalist result matrix — KITTI AP40 and p95 latency",
        fontsize=21,
        y=0.98,
    )
    axis.legend(
        handles=(
            Patch(facecolor="#ECECEC", label="20-epoch reference"),
            Patch(facecolor="#B8D8BA", label="Improved at 30 epochs"),
            Patch(facecolor="#F4B6B2", label="Degraded at 30 epochs"),
        ),
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncols=1,
        frameon=False,
    )
    axis.set_xlabel(
        "Parentheses show the 30 − 20 epoch difference; AP40 is reported in percentage points.",
        labelpad=24,
        fontsize=11.5,
    )
    _save(figure, output_dir, FIGURE_FINALISTS)


def _multiclass_metric(payload: dict[str, Any], class_name: str, domain: str, difficulty: str) -> float:
    key = f"Kitti metric/pred_instances_3d/KITTI/{class_name}_{domain}_AP40_{difficulty}_strict"
    metrics = _required(_required(payload.get("payload"), "evaluation.payload").get("metrics"), "evaluation.metrics")
    return float(_required(metrics.get(key), key))


def _render_multiclass(context: dict[str, Any], output_dir: Path) -> None:
    evaluation = context["evaluation"]
    benchmark = context["benchmark"]
    matrices = {
        domain: np.asarray(
            [[_multiclass_metric(evaluation, class_name, domain, difficulty) for difficulty in DIFFICULTIES] for class_name in CLASSES],
            dtype=float,
        )
        for domain in ("3D", "BEV")
    }
    figure, axes = plt.subplots(1, 2, figsize=(13.333, 7.5))
    figure.subplots_adjust(left=0.08, right=0.89, top=0.79, bottom=0.22, wspace=0.22)
    rendered = None
    for axis, domain in zip(axes, ("3D", "BEV")):
        matrix = matrices[domain]
        rendered = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                color = "white" if matrix[row, column] >= 58 else "#1F1F1F"
                axis.text(column, row, _format_value(float(matrix[row, column])), ha="center", va="center", color=color, fontsize=15, fontweight="bold")
        axis.set_xticks(np.arange(3), DIFFICULTY_LABELS)
        axis.set_yticks(np.arange(3), CLASSES)
        axis.set_xlabel("Difficulty")
        axis.set_title(f"KITTI {domain} AP40 (strict)")
        axis.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        axis.grid(which="minor", color="white", linestyle="-", linewidth=2)
        axis.grid(which="major", visible=False)
        axis.tick_params(which="both", length=0)
    assert rendered is not None
    colorbar = figure.colorbar(rendered, ax=axes, shrink=0.86, pad=0.025)
    colorbar.set_label("AP40 (strict) [%]")
    figure.suptitle(
        "Pillar02 multiclass results — class-wise AP40 (strict)",
        fontsize=21,
        y=0.96,
    )
    e2e_p95 = float(benchmark["payload"]["end_to_end_ms"]["p95_ms"])
    figure.text(
        0.5,
        0.035,
        f"Separate 60-epoch experiment; selected checkpoint: epoch 55; end-to-end p95: {_format_value(e2e_p95)} ms. "
        "Not directly rankable against the six 20-epoch Car-only models.",
        ha="center",
        fontsize=10.5,
    )
    _save(figure, output_dir, FIGURE_MULTICLASS)


def _load_context(runs_root: Path) -> dict[str, Any]:
    end_to_end = _rows_by_slug(_load_json(SIX_END_TO_END), SIX_END_TO_END)
    prediction = _rows_by_slug(_load_json(SIX_PREDICTION), SIX_PREDICTION)
    for slug in MODEL_ORDER:
        if end_to_end[slug].get("checkpoint_sha256") != prediction[slug].get("checkpoint_sha256"):
            raise ValueError(f"Eltérő checkpoint a két hatmodell-forrásban: {slug}")
    finalists = _load_json(FINALISTS)
    run_root = runs_root / RUN_ID
    manifest = _load_json(run_root / "manifest.json")
    evaluation = _load_json(run_root / "evaluation" / EVALUATION_ID / "result.json")
    benchmark = _load_json(run_root / "benchmark" / BENCHMARK_ID / "result.json")
    _validate_multiclass(run_root, manifest, evaluation, benchmark)
    return {
        "end_to_end": end_to_end,
        "prediction": prediction,
        "finalists": finalists,
        "evaluation": evaluation,
        "benchmark": benchmark,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        required=True,
        type=Path,
        help="A tartós kísérleti futások abszolút gyökérkönyvtára.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs/presentation_assets/final/figures",
        help="A PNG- és SVG-kimenetek könyvtára.",
    )
    parser.add_argument(
        "--figure",
        action="append",
        choices=FIGURE_NAMES,
        help="Csak a megadott ábrát rendereli; többször megadható.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Csak a forrás- és artifactkötéseket ellenőrzi.",
    )
    arguments = parser.parse_args()
    if not arguments.runs_root.is_absolute():
        parser.error("--runs-root csak abszolút útvonal lehet")

    context = _load_context(arguments.runs_root)
    print(
        "FORRÁSOK_RENDBEN: "
        f"run={RUN_ID} checkpoint_sha256={CHECKPOINT_SHA256} "
        f"checkpoint_size={CHECKPOINT_SIZE}"
    )
    if arguments.verify_only:
        return 0

    _configure_style()
    renderers: dict[str, Callable[[dict[str, Any], Path], None]] = {
        FIGURE_PARETO: _render_pareto,
        FIGURE_LATENCY: _render_latency,
        FIGURE_FINALISTS: _render_finalists,
        FIGURE_MULTICLASS: _render_multiclass,
    }
    selected = tuple(dict.fromkeys(arguments.figure or FIGURE_NAMES))
    for name in selected:
        renderers[name](context, arguments.output_dir)
        print(f"ÁBRA_KÉSZ: {arguments.output_dir / name}.[png|svg]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
