# LiDAR detection runtime

This directory is an independent ROS 2 `ament_python` package for the future
vehicle runtime.

The package is intentionally only a boundary today. It does not contain the
recorded KITTI playback node, import the research package, reconstruct an
MMDetection3D training model, or expose a placeholder production executable.
Those choices prevent research dependencies from leaking into deployment.

Runtime implementation starts after model selection and export. It will
consume a frozen artifact plus the metadata contract in
`../artifacts/model-contract.example.json`, then provide preprocessing,
inference, postprocessing, live sensor communication, latest-frame handling,
and diagnostics.

Before a release, replace the `TODO` license metadata with the repository
owner's chosen license.
