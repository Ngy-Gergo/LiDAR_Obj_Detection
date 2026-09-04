#!/usr/bin/env python3
"""Read-only readiness check for the Pillar02 three-class experiment."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_SOURCE = REPOSITORY_ROOT / "research" / "src"
if str(RESEARCH_SOURCE) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SOURCE))


CANONICAL_CLASSES = ("Car", "Pedestrian", "Cyclist")


def _value(container: Any, key: str) -> Any:
    return container[key] if isinstance(container, dict) else getattr(container, key)


def _dataset(config: Any) -> Any:
    return _value(_value(config, "train_dataloader"), "dataset")


def _count_annotations(path: Path, class_names: tuple[str, ...]) -> tuple[int, Counter[str]]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    data_list = payload["data_list"]
    categories = payload["metainfo"]["categories"]
    labels_to_names = {index: name for name, index in categories.items()}
    counts = Counter(
        labels_to_names[instance["bbox_label_3d"]]
        for info in data_list
        for instance in info["instances"]
        if instance["bbox_label_3d"] >= 0
        and labels_to_names[instance["bbox_label_3d"]] in class_names
    )
    return len(data_list), counts


def _usable_database_counts(path: Path, class_names: tuple[str, ...]) -> Counter[str]:
    with path.open("rb") as stream:
        database = pickle.load(stream)
    return Counter(
        {
            name: sum(entry["num_points_in_gt"] >= 5 for entry in database[name])
            for name in class_names
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify read-only KITTI three-class data and config readiness."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "research/configs/centerpoint/pillar02_multiclass.py",
        help="Explicit multiclass config to inspect.",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    from mmengine.config import Config

    config = Config.fromfile(os.fspath(arguments.config))
    classes = tuple(config.class_names)
    if classes != CANONICAL_CLASSES:
        raise ValueError(f"class order must be {CANONICAL_CLASSES!r}, got {classes!r}")
    head = config.model.pts_bbox_head
    task = head.tasks[0]
    if tuple(task.class_names) != classes or task.num_class != len(classes):
        raise ValueError("single CenterHead task does not match dataset class order")

    dataset = _dataset(config)
    root = Path(dataset.data_root).resolve(strict=True)
    required_files = (
        "kitti_infos_train.pkl",
        "kitti_infos_val.pkl",
        "kitti_dbinfos_train.pkl",
    )
    missing = [name for name in required_files if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required KITTI artifacts: {', '.join(missing)}")
    reduced = root / "training/velodyne_reduced"
    if not reduced.is_dir() or not any(reduced.glob("*.bin")):
        raise FileNotFoundError("missing reduced KITTI training points")

    train_total, train_counts = _count_annotations(
        root / "kitti_infos_train.pkl", classes
    )
    val_total, val_counts = _count_annotations(root / "kitti_infos_val.pkl", classes)
    database_counts = _usable_database_counts(
        root / "kitti_dbinfos_train.pkl", classes
    )
    if any(count == 0 for count in train_counts.values()) or any(
        count == 0 for count in val_counts.values()
    ) or any(count == 0 for count in database_counts.values()):
        raise ValueError("one or more canonical classes has no usable data")

    print(f"CONFIG: {arguments.config.resolve()}")
    print(f"DATASET: {root}")
    print(f"CLASS_ORDER: {', '.join(classes)}")
    print(f"TRAIN: samples={train_total} counts={dict(train_counts)}")
    print(f"VAL: samples={val_total} counts={dict(val_counts)}")
    print(f"DATABASE_MIN_POINTS_5: counts={dict(database_counts)}")
    planes = root / "training/planes"
    if planes.is_dir():
        print(f"GROUND_PLANES: present files={sum(1 for _ in planes.iterdir())}")
    else:
        print("GROUND_PLANES: missing; ObjectSample use_ground_plane=False")
        print(
            "GROUND_PLANES_COMMAND: "
            "do not generate in place; prepare a separate KITTI root with "
            "MMDetection3D tools/create_data.py kitti --root-path <KITTI_ROOT> "
            "--out-dir <SEPARATE_OUTPUT_ROOT> --extra-tag kitti"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
