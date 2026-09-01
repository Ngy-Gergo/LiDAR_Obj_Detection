"""First paired duration-screening candidate for the pillar02 finalist."""

_base_ = ["../pillar02.py"]

model_alias = "pillar02"
parent_baseline_run_id = (
    "20260827T092033Z-pillar02-3367910930525d0c12ddc346"
)
campaign_id = "20260901-finalists-duration30-screening"
comparison_group = "finalists-duration30-paired"
hypothesis = (
    "Validation AP40 and training loss still improve through epoch 20; fresh "
    "30-epoch training with the existing two-phase schedule stretched "
    "proportionally will test whether premature stopping limits both finalists."
)
factor_name = "training_duration"
factor_group = "training_duration_plus_proportional_scheduler_duration"
seed = 20260724
resolved_config = "run-owned config.py"
explicit_overrides = dict(
    primary=dict(
        max_epochs=dict(baseline=20, candidate=30),
        train_cfg_max_epochs=dict(baseline=20, candidate=30),
    ),
    coupled=dict(
        scheduler_phase_1_epochs=dict(baseline=[0, 8], candidate=[0, 12]),
        scheduler_phase_2_epochs=dict(baseline=[8, 20], candidate=[12, 30]),
    ),
)
dataset_identity = "0bb26013400c77313f2720b2295f78fc84ba30ef1711b3c769608fa02aa3c8df"
split_identity = dict(
    train=dict(
        name="KITTI train",
        samples=3712,
        sha256="b6417a1d9b18c8fdb085128e633d28ff321b7674a6d1b3841b8f43d865b281cb",
    ),
    validation=dict(
        name="KITTI validation",
        samples=3769,
        sha256="657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86",
    ),
    overlap_count=0,
)
selection_metric = (
    "Kitti metric/pred_instances_3d/KITTI/"
    "Car_3D_AP40_moderate_strict"
)
benchmark_methodology_version = 1

max_epochs = 30
train_cfg = dict(max_epochs=max_epochs)

param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        T_max=12,
        eta_min=0.001,
        begin=0,
        end=12,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=18,
        eta_min=1e-8,
        begin=12,
        end=30,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=12,
        eta_min=0.85 / 0.95,
        begin=0,
        end=12,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=18,
        eta_min=1.0,
        begin=12,
        end=30,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]
