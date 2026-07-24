"""Single-GPU MMEngine training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from mmengine.config import Config
from mmengine.runner import Runner

from lidar_model_selection.compat.kitti_evaluator import install
from mmdet3d.utils import register_all_modules


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one CenterPoint model on one GPU.",
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to an MMEngine configuration file.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Override the work directory from the configuration.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in the work directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")

    # Install the project-local rotated-IoU implementation before the
    # official KITTI evaluator can import its incompatible implementation.
    install()

    register_all_modules(init_default_scope=True)

    cfg = Config.fromfile(str(config_path))
    cfg.launcher = "none"

    if args.work_dir is not None:
        cfg.work_dir = str(args.work_dir.resolve())

    if args.resume:
        cfg.resume = True

    runner = Runner.from_cfg(cfg)
    runner.train()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())