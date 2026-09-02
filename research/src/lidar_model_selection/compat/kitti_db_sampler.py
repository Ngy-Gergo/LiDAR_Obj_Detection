"""Local MMDetection3D v1.4.0 database-sampler compatibility adapter."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from mmdet3d.datasets.transforms.dbsampler import DataBaseSampler
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class KittiDataBaseSampler(DataBaseSampler):
    """Use the upstream sampler logic with NumPy's supported int64 dtype.

    MMDetection3D v1.4.0 calls the removed ``np.long`` alias in one line of
    ``sample_all``.  Keeping the override here avoids both an upstream edit and
    a process-global NumPy compatibility mutation.
    """

    def sample_all(
        self,
        gt_bboxes: np.ndarray,
        gt_labels: np.ndarray,
        img: np.ndarray | None = None,
        ground_plane: np.ndarray | None = None,
    ) -> dict[str, Any] | None:
        sampled_num_per_class: list[int] = []
        for class_name, max_sample_num in zip(
            self.sample_classes,
            self.sample_max_nums,
        ):
            class_label = self.cat2label[class_name]
            sampled_num = int(max_sample_num - np.sum(gt_labels == class_label))
            sampled_num_per_class.append(
                int(np.round(self.rate * sampled_num).astype(np.int64))
            )

        sampled: list[dict[str, Any]] = []
        sampled_gt_bboxes: list[np.ndarray] = []
        avoid_coll_boxes = gt_bboxes
        for class_name, sampled_num in zip(
            self.sample_classes,
            sampled_num_per_class,
        ):
            if sampled_num <= 0:
                continue
            sampled_class = self.sample_class_v2(
                class_name,
                sampled_num,
                avoid_coll_boxes,
            )
            sampled.extend(sampled_class)
            if not sampled_class:
                continue
            boxes = np.stack(
                [entry["box3d_lidar"] for entry in sampled_class],
                axis=0,
            )
            sampled_gt_bboxes.append(boxes)
            avoid_coll_boxes = np.concatenate((avoid_coll_boxes, boxes), axis=0)

        if not sampled:
            return None
        sampled_boxes = np.concatenate(sampled_gt_bboxes, axis=0)
        sampled_points = []
        for entry in sampled:
            file_path = (
                os.path.join(self.data_root, entry["path"])
                if self.data_root
                else entry["path"]
            )
            points = self.points_loader(
                dict(lidar_points=dict(lidar_path=file_path))
            )["points"]
            points.translate(entry["box3d_lidar"][:3])
            sampled_points.append(points)

        labels = np.asarray(
            [self.cat2label[entry["name"]] for entry in sampled],
            dtype=np.int64,
        )
        if ground_plane is not None:
            xyz = sampled_boxes[:, :3]
            dz = (ground_plane[:3][None, :] * xyz).sum(-1) + ground_plane[3]
            sampled_boxes[:, 2] -= dz
            for index, points in enumerate(sampled_points):
                points.tensor[:, 2].sub_(dz[index])
        return {
            "gt_labels_3d": labels,
            "gt_bboxes_3d": sampled_boxes,
            "points": sampled_points[0].cat(sampled_points),
            "group_ids": np.arange(
                gt_bboxes.shape[0],
                gt_bboxes.shape[0] + len(sampled),
            ),
        }
