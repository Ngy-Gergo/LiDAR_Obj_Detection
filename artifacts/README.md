# Exported model artifacts

This directory is the hand-off boundary between research and vehicle runtime.
Model binaries and instance-specific metadata are generated or retrieved
locally and are ignored by Git.

A selected artifact directory should contain one frozen backend model and one
metadata document following `model-contract.example.json`. Training
checkpoints and arbitrary MMDetection3D configurations are not runtime
artifacts.
