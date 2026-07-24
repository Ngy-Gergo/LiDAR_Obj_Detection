"""KITTI-specific CenterPoint head without velocity regression."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.models.dense_heads import CenterHead
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.registry import MODELS
from mmdet3d.structures import LiDARInstance3DBoxes


@MODELS.register_module()
class KittiCenterHead(CenterHead):
    """Adapt CenterHead training and rotated NMS to 7D KITTI boxes.

    KITTI boxes contain position, dimensions, and yaw but no velocity.
    Targets therefore have eight regression values, while decoded boxes have
    seven values.
    """

    def get_targets_single(
        self,
        gt_instances_3d: InstanceData,
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        """Generate official CenterHead targets, then remove fake velocity."""
        if not isinstance(gt_instances_3d, InstanceData):
            raise TypeError(
                "gt_instances_3d must be InstanceData, but got "
                f"{type(gt_instances_3d).__name__}."
            )

        fields = set(gt_instances_3d.keys())
        missing_fields = {"bboxes_3d", "labels_3d"} - fields
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(
                "gt_instances_3d is missing required field(s): "
                f"{missing}."
            )

        boxes = gt_instances_3d.bboxes_3d
        labels = gt_instances_3d.labels_3d
        if not isinstance(boxes, LiDARInstance3DBoxes):
            raise TypeError(
                "bboxes_3d must be LiDARInstance3DBoxes, but got "
                f"{type(boxes).__name__}."
            )
        if not isinstance(boxes.tensor, Tensor):
            raise TypeError(
                "bboxes_3d.tensor must be a torch.Tensor, but got "
                f"{type(boxes.tensor).__name__}."
            )
        if boxes.tensor.ndim != 2:
            raise ValueError(
                "bboxes_3d.tensor must have rank 2, but got shape "
                f"{tuple(boxes.tensor.shape)}."
            )
        if boxes.tensor.shape[1] != 7:
            raise ValueError(
                "bboxes_3d.tensor must have width 7, but got shape "
                f"{tuple(boxes.tensor.shape)}."
            )
        if boxes.box_dim != 7:
            raise ValueError(
                "bboxes_3d.box_dim must be 7, but got "
                f"{boxes.box_dim} for tensor shape "
                f"{tuple(boxes.tensor.shape)}."
            )
        if not isinstance(labels, Tensor):
            raise TypeError(
                "labels_3d must be a torch.Tensor, but got "
                f"{type(labels).__name__}."
            )
        if labels.ndim != 1:
            raise ValueError(
                "labels_3d must have rank 1, but got shape "
                f"{tuple(labels.shape)}."
            )
        if labels.shape[0] != boxes.tensor.shape[0]:
            raise ValueError(
                "labels_3d and bboxes_3d must contain the same number "
                f"of objects, but got labels shape {tuple(labels.shape)} "
                f"and boxes shape {tuple(boxes.tensor.shape)}."
            )

        fake_velocity = boxes.tensor.new_zeros((len(boxes), 2))
        temporary_tensor = torch.cat((boxes.tensor, fake_velocity), dim=1)
        temporary_boxes = type(boxes)(
            temporary_tensor,
            box_dim=9,
            with_yaw=True,
            origin=(0.5, 0.5, 0.0),
        )
        temporary_gt_instances = InstanceData()
        temporary_gt_instances.bboxes_3d = temporary_boxes
        temporary_gt_instances.labels_3d = labels.clone()

        heatmaps, anno_boxes, inds, masks = super().get_targets_single(
            temporary_gt_instances
        )

        kitti_anno_boxes = []
        for task_id, anno_box in enumerate(anno_boxes):
            if not isinstance(anno_box, Tensor):
                raise TypeError(
                    "Official CenterHead annotation target for task "
                    f"{task_id} must be a torch.Tensor, but got "
                    f"{type(anno_box).__name__}."
                )
            if anno_box.ndim != 2 or anno_box.shape[-1] != 10:
                raise ValueError(
                    "Official CenterHead annotation target for task "
                    f"{task_id} must have shape (max_objs, 10), but got "
                    f"{tuple(anno_box.shape)}."
                )
            kitti_anno_boxes.append(anno_box[:, :8].contiguous())

        return heatmaps, kitti_anno_boxes, inds, masks

    def loss_by_feat(
        self,
        preds_dicts: Sequence[Sequence[Dict[str, Tensor]]],
        batch_gt_instances_3d: List[InstanceData],
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Tensor]:
        """Calculate stock CenterHead losses without velocity regression."""
        if not isinstance(batch_gt_instances_3d, (list, tuple)):
            raise TypeError(
                "batch_gt_instances_3d must be a list or tuple, but got "
                f"{type(batch_gt_instances_3d).__name__}."
            )
        batch_size = len(batch_gt_instances_3d)
        if batch_size == 0:
            raise ValueError(
                "KittiCenterHead loss requires at least one batch element."
            )

        code_size = getattr(self.bbox_coder, "code_size", None)
        if code_size != 7:
            raise ValueError(
                "KittiCenterHead requires bbox_coder.code_size == 7, "
                f"but got {code_size}."
            )

        if self.train_cfg is None or "code_weights" not in self.train_cfg:
            raise ValueError(
                "KittiCenterHead requires train_cfg.code_weights with "
                "exactly 8 values."
            )
        code_weights = self.train_cfg["code_weights"]
        if code_weights is None:
            raise ValueError(
                "KittiCenterHead requires train_cfg.code_weights, but "
                "got None."
            )
        if isinstance(code_weights, Tensor):
            code_weights_tensor = code_weights
        else:
            try:
                code_weights_tensor = torch.as_tensor(code_weights)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "train_cfg.code_weights must be a one-dimensional "
                    f"numeric sequence, but got {type(code_weights).__name__}."
                ) from error
        if code_weights_tensor.ndim != 1:
            raise ValueError(
                "train_cfg.code_weights must be one-dimensional, but got "
                f"shape {tuple(code_weights_tensor.shape)}."
            )
        if code_weights_tensor.numel() != 8:
            raise ValueError(
                "train_cfg.code_weights must contain exactly 8 values, "
                f"but got shape {tuple(code_weights_tensor.shape)}."
            )

        if not isinstance(preds_dicts, (list, tuple)):
            raise TypeError(
                "preds_dicts must be a list or tuple of prediction tasks, "
                f"but got {type(preds_dicts).__name__}."
            )
        expected_tasks = len(self.task_heads)
        if expected_tasks == 0:
            raise ValueError(
                "KittiCenterHead requires at least one configured task."
            )
        if len(preds_dicts) != expected_tasks:
            raise ValueError(
                "Prediction task count must match the configured head: "
                f"expected {expected_tasks}, got {len(preds_dicts)}."
            )

        required_keys = ("heatmap", "reg", "height", "dim", "rot")
        expected_channels = {
            "reg": 2,
            "height": 1,
            "dim": 3,
            "rot": 2,
        }
        predictions = []
        regression_predictions = []
        for task_id, prediction_levels in enumerate(preds_dicts):
            if not isinstance(prediction_levels, (list, tuple)):
                raise TypeError(
                    f"Prediction task {task_id} must contain prediction "
                    "levels in a list or tuple, but got "
                    f"{type(prediction_levels).__name__}."
                )
            if len(prediction_levels) == 0:
                raise ValueError(
                    f"Prediction task {task_id} has no prediction levels."
                )
            prediction = prediction_levels[0]
            if not isinstance(prediction, dict):
                raise TypeError(
                    f"Prediction task {task_id}, level 0 must be a dict, "
                    f"but got {type(prediction).__name__}."
                )

            missing_keys = [
                key for key in required_keys if key not in prediction
            ]
            if missing_keys:
                raise KeyError(
                    f"Prediction task {task_id} is missing required key(s): "
                    f"{', '.join(missing_keys)}."
                )
            if "vel" in prediction:
                raise ValueError(
                    f"Prediction task {task_id} contains forbidden "
                    "velocity branch 'vel'."
                )

            for branch_name in required_keys:
                branch = prediction[branch_name]
                if not isinstance(branch, Tensor):
                    raise TypeError(
                        f"Prediction task {task_id} branch "
                        f"'{branch_name}' must be a torch.Tensor, but got "
                        f"{type(branch).__name__}."
                    )
                if branch.ndim != 4:
                    raise ValueError(
                        f"Prediction task {task_id} branch "
                        f"'{branch_name}' must have rank 4, but got shape "
                        f"{tuple(branch.shape)}."
                    )

            heatmap = prediction["heatmap"]
            if heatmap.shape[0] != batch_size:
                raise ValueError(
                    f"Prediction task {task_id} batch size must be "
                    f"{batch_size}, but got heatmap shape "
                    f"{tuple(heatmap.shape)}."
                )
            expected_heatmap_channels = len(self.class_names[task_id])
            if heatmap.shape[1] != expected_heatmap_channels:
                raise ValueError(
                    f"Prediction task {task_id} heatmap must have "
                    f"{expected_heatmap_channels} channels, but got shape "
                    f"{tuple(heatmap.shape)}."
                )
            reference_layout = (
                heatmap.shape[0],
                heatmap.shape[2],
                heatmap.shape[3],
            )
            for branch_name, channel_count in expected_channels.items():
                branch = prediction[branch_name]
                if branch.shape[1] != channel_count:
                    raise ValueError(
                        f"Prediction task {task_id} branch "
                        f"'{branch_name}' must have {channel_count} "
                        f"channels, but got shape {tuple(branch.shape)}."
                    )
                branch_layout = (
                    branch.shape[0],
                    branch.shape[2],
                    branch.shape[3],
                )
                if branch_layout != reference_layout:
                    raise ValueError(
                        f"Prediction task {task_id} branch "
                        f"'{branch_name}' batch/spatial shape "
                        f"{branch_layout} does not match heatmap "
                        f"batch/spatial shape {reference_layout}."
                    )

            regression_reference = prediction["reg"]
            for branch_name in ("height", "dim", "rot"):
                branch = prediction[branch_name]
                if branch.device != regression_reference.device:
                    raise ValueError(
                        f"Prediction task {task_id} branch "
                        f"'{branch_name}' is on {branch.device}, but reg "
                        f"is on {regression_reference.device}."
                    )
                if branch.dtype != regression_reference.dtype:
                    raise ValueError(
                        f"Prediction task {task_id} branch "
                        f"'{branch_name}' has dtype {branch.dtype}, but reg "
                        f"has dtype {regression_reference.dtype}."
                    )

            regression_prediction = torch.cat(
                (
                    prediction["reg"],
                    prediction["height"],
                    prediction["dim"],
                    prediction["rot"],
                ),
                dim=1,
            )
            if regression_prediction.shape[1] != 8:
                raise ValueError(
                    f"Prediction task {task_id} concatenated regression "
                    "must have width 8, but got shape "
                    f"{tuple(regression_prediction.shape)}."
                )
            predictions.append(prediction)
            regression_predictions.append(regression_prediction)

        heatmaps, anno_boxes, inds, masks = self.get_targets(
            batch_gt_instances_3d
        )
        target_collections = {
            "heatmaps": heatmaps,
            "anno_boxes": anno_boxes,
            "inds": inds,
            "masks": masks,
        }
        for target_name, targets in target_collections.items():
            if len(targets) != expected_tasks:
                raise ValueError(
                    f"Generated {target_name} task count must be "
                    f"{expected_tasks}, but got {len(targets)}."
                )

        for task_id, prediction in enumerate(predictions):
            heatmap_target = heatmaps[task_id]
            target_box = anno_boxes[task_id]
            ind = inds[task_id]
            mask = masks[task_id]

            if not isinstance(heatmap_target, Tensor):
                raise TypeError(
                    f"Generated heatmap target for task {task_id} must be "
                    "a torch.Tensor, but got "
                    f"{type(heatmap_target).__name__}."
                )
            if tuple(heatmap_target.shape) != tuple(
                prediction["heatmap"].shape
            ):
                raise ValueError(
                    f"Prediction and target heatmap shapes for task "
                    f"{task_id} must match, but got prediction "
                    f"{tuple(prediction['heatmap'].shape)} and target "
                    f"{tuple(heatmap_target.shape)}."
                )
            if not isinstance(target_box, Tensor):
                raise TypeError(
                    f"Generated regression target for task {task_id} must "
                    f"be a torch.Tensor, but got "
                    f"{type(target_box).__name__}."
                )
            if target_box.ndim != 3 or target_box.shape[-1] != 8:
                raise ValueError(
                    f"Generated regression target for task {task_id} must "
                    f"have shape (batch, max_objs, 8), but got "
                    f"{tuple(target_box.shape)}."
                )
            if not isinstance(ind, Tensor) or not isinstance(mask, Tensor):
                raise TypeError(
                    f"Generated inds and masks for task {task_id} must be "
                    "torch.Tensor objects, but got "
                    f"{type(ind).__name__} and {type(mask).__name__}."
                )
            if ind.ndim != 2 or mask.ndim != 2:
                raise ValueError(
                    f"Generated inds and masks for task {task_id} must "
                    f"have rank 2, but got {tuple(ind.shape)} and "
                    f"{tuple(mask.shape)}."
                )
            if tuple(ind.shape) != tuple(mask.shape):
                raise ValueError(
                    f"Generated inds and masks for task {task_id} must "
                    f"have matching shapes, but got {tuple(ind.shape)} "
                    f"and {tuple(mask.shape)}."
                )
            if tuple(target_box.shape[:2]) != tuple(ind.shape):
                raise ValueError(
                    f"Generated regression target and inds for task "
                    f"{task_id} disagree: target shape "
                    f"{tuple(target_box.shape)}, inds shape "
                    f"{tuple(ind.shape)}."
                )

        loss_dict = {}
        for task_id, prediction in enumerate(predictions):
            prediction["heatmap"] = clip_sigmoid(prediction["heatmap"])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            loss_heatmap = self.loss_cls(
                prediction["heatmap"],
                heatmaps[task_id],
                avg_factor=max(num_pos, 1),
            )

            target_box = anno_boxes[task_id]
            prediction["anno_box"] = regression_predictions[task_id]
            ind = inds[task_id]
            num = masks[task_id].float().sum()
            pred = prediction["anno_box"].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan

            bbox_weights = mask * mask.new_tensor(code_weights)
            loss_bbox = self.loss_bbox(
                pred,
                target_box,
                bbox_weights,
                avg_factor=(num + 1e-4),
            )
            loss_dict[f"task{task_id}.loss_heatmap"] = loss_heatmap
            loss_dict[f"task{task_id}.loss_bbox"] = loss_bbox

        return loss_dict

    def get_task_detections(
        self,
        num_class_with_bg: int,
        batch_cls_preds: Sequence[Tensor],
        batch_reg_preds: Sequence[Tensor],
        batch_cls_labels: Sequence[Tensor],
        img_metas: Sequence[dict],
    ) -> List[Dict[str, Tensor]]:
        """Normalize singleton scores, then delegate all rotated NMS logic."""
        code_size = getattr(self.bbox_coder, "code_size", None)
        if code_size != 7:
            raise ValueError(
                "KittiCenterHead requires bbox_coder.code_size == 7 during "
                f"prediction, but got {code_size}."
            )

        batch_inputs = {
            "scores": batch_cls_preds,
            "boxes": batch_reg_preds,
            "labels": batch_cls_labels,
            "img_metas": img_metas,
        }
        for input_name, batch_input in batch_inputs.items():
            if not isinstance(batch_input, (list, tuple)):
                raise TypeError(
                    f"Batch {input_name} must be a list or tuple, but got "
                    f"{type(batch_input).__name__}."
                )

        batch_size = len(batch_reg_preds)
        if batch_size == 0:
            raise ValueError(
                "Rotated NMS requires at least one batch element."
            )
        for input_name, batch_input in batch_inputs.items():
            if len(batch_input) != batch_size:
                raise ValueError(
                    f"Batch {input_name} length must be {batch_size}, but "
                    f"got {len(batch_input)}."
                )

        normalized_scores = []
        for batch_id, (scores, boxes, labels) in enumerate(
            zip(batch_cls_preds, batch_reg_preds, batch_cls_labels)
        ):
            if not isinstance(boxes, Tensor):
                raise TypeError(
                    f"Decoded boxes for batch {batch_id} must be a "
                    f"torch.Tensor, but got {type(boxes).__name__}."
                )
            if boxes.ndim != 2 or boxes.shape[1] != 7:
                raise ValueError(
                    f"Decoded boxes for batch {batch_id} must have shape "
                    f"(N, 7), but got {tuple(boxes.shape)}."
                )
            if not isinstance(scores, Tensor):
                raise TypeError(
                    f"Scores for batch {batch_id} must be a torch.Tensor, "
                    f"but got {type(scores).__name__}."
                )
            if scores.ndim == 1:
                normalized_score = scores.unsqueeze(-1)
            elif scores.ndim == 2 and scores.shape[1] == 1:
                normalized_score = scores
            else:
                raise ValueError(
                    f"Scores for batch {batch_id} must have shape (N,) or "
                    f"(N, 1), but got {tuple(scores.shape)}."
                )
            if not isinstance(labels, Tensor):
                raise TypeError(
                    f"Labels for batch {batch_id} must be a torch.Tensor, "
                    f"but got {type(labels).__name__}."
                )
            if labels.ndim != 1:
                raise ValueError(
                    f"Labels for batch {batch_id} must have shape (N,), "
                    f"but got {tuple(labels.shape)}."
                )
            candidate_count = boxes.shape[0]
            if scores.shape[0] != candidate_count:
                raise ValueError(
                    f"Scores and boxes for batch {batch_id} must contain "
                    f"the same candidates, but got scores shape "
                    f"{tuple(scores.shape)} and boxes shape "
                    f"{tuple(boxes.shape)}."
                )
            if labels.shape[0] != candidate_count:
                raise ValueError(
                    f"Labels and boxes for batch {batch_id} must contain "
                    f"the same candidates, but got labels shape "
                    f"{tuple(labels.shape)} and boxes shape "
                    f"{tuple(boxes.shape)}."
                )
            normalized_scores.append(normalized_score)

        return super().get_task_detections(
            num_class_with_bg,
            normalized_scores,
            batch_reg_preds,
            batch_cls_labels,
            img_metas,
        )
