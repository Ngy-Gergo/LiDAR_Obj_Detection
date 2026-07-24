"""Short 500-iteration training check for CenterPoint Pillar 0.2."""

_base_ = ["./pillar02.py"]

train_cfg = dict(
    _delete_=True,
    type="IterBasedTrainLoop",
    max_iters=500,
    val_interval=500,
)

# The full epoch-based scheduler is unnecessary for this short stability test.
param_scheduler = []

default_hooks = dict(
    logger=dict(
        type="LoggerHook",
        interval=20,
    ),
    checkpoint=dict(
        _delete_=True,
        type="CheckpointHook",
        by_epoch=False,
        interval=500,
        max_keep_ckpts=1,
    ),
)

work_dir = "research/experiments/pillar02_short"