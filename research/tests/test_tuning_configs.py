"""Fidelity checks for the frozen first paired finalist experiment."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from mmengine.config import Config


CONFIG_ROOT = Path(__file__).parents[1] / "configs" / "centerpoint"
CAMPAIGN_ID = "20260901-finalists-duration30-screening"
DATASET_IDENTITY = (
    "0bb26013400c77313f2720b2295f78fc84ba30ef1711b3c769608fa02aa3c8df"
)
EXPERIMENT_FIELDS = {
    "benchmark_methodology_version",
    "campaign_id",
    "comparison_group",
    "dataset_identity",
    "explicit_overrides",
    "factor_group",
    "factor_name",
    "hypothesis",
    "model_alias",
    "parent_baseline_run_id",
    "resolved_config",
    "seed",
    "selection_metric",
    "split_identity",
}


def _resolved(relative: str) -> dict[str, object]:
    return Config.fromfile(CONFIG_ROOT / relative).to_dict()


def _without_frozen_factor(config: dict[str, object]) -> dict[str, object]:
    result = deepcopy(config)
    for field in EXPERIMENT_FIELDS:
        result.pop(field, None)
    result["max_epochs"] = 20
    train_cfg = dict(result["train_cfg"])
    train_cfg["max_epochs"] = 20
    result["train_cfg"] = train_cfg
    result.pop("param_scheduler")
    return result


def test_duration30_candidates_change_only_the_frozen_factor_group() -> None:
    pairs = (
        ("voxel0075.py", "tuning/voxel0075_duration30.py", "voxel0075"),
        ("pillar02.py", "tuning/pillar02_duration30.py", "pillar02"),
    )
    for baseline_path, candidate_path, alias in pairs:
        baseline = _resolved(baseline_path)
        candidate = _resolved(candidate_path)
        assert _without_frozen_factor(candidate) == {
            key: value for key, value in baseline.items() if key != "param_scheduler"
        }
        assert candidate["model_alias"] == alias
        assert candidate["campaign_id"] == CAMPAIGN_ID
        assert candidate["dataset_identity"] == DATASET_IDENTITY
        assert candidate["seed"] == baseline["randomness"]["seed"]
        assert candidate["max_epochs"] == 30
        assert candidate["train_cfg"]["max_epochs"] == 30


def test_duration30_candidates_share_the_proportional_schedule() -> None:
    voxel = _resolved("tuning/voxel0075_duration30.py")
    pillar = _resolved("tuning/pillar02_duration30.py")
    assert voxel["param_scheduler"] == pillar["param_scheduler"]
    assert [item["begin"] for item in voxel["param_scheduler"]] == [0, 12, 0, 12]
    assert [item["end"] for item in voxel["param_scheduler"]] == [12, 30, 12, 30]
    assert [item["T_max"] for item in voxel["param_scheduler"]] == [12, 18, 12, 18]
