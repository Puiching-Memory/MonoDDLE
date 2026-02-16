"""
KITTI3DDataset: A KITTI dataset adapter that produces ultralytics-compatible batches
with additional monocular 3D target annotations.

This wraps the existing KITTI_Dataset's label parsing logic and produces:
  - Standard ultralytics keys: img, bboxes (xywh normalized), cls, batch_idx
  - 3D keys: mono3d_targets (depth, offset3d, size3d, heading_bin, heading_res, mask_3d)
  - Calibration/image metadata needed for 3D inference evaluation
"""

import os
import numpy as np
import torch
import torch.utils.data as data
from copy import deepcopy
from PIL import Image

from lib.datasets.utils import angle2class
from lib.datasets.kitti.kitti_utils import get_objects_from_label, Calibration
from lib.datasets.transforms import get_affine_transform, affine_transform


class KITTI3DDataset(data.Dataset):
    """
    KITTI dataset for YOLO3D training.

    Produces per-image dicts compatible with ultralytics' collate_fn and
    augmented with 3D monocular detection targets.
    """

    def __init__(self, split, cfg):
        self.root_dir = cfg.get('root_dir', 'data/KITTI')
        self.split = split
        self.num_classes = 3
        self.max_objs = 50
        self.class_name = ['Pedestrian', 'Car', 'Cyclist']
        self.cls2id = {'Pedestrian': 0, 'Car': 1, 'Cyclist': 2}
        self.writelist = cfg.get('writelist', ['Car'])
        self.bbox2d_type = cfg.get('bbox2d_type', 'anno')
        self.meanshape = cfg.get('meanshape', False)
        self.class_merging = cfg.get('class_merging', False)
        self.use_dontcare = cfg.get('use_dontcare', False)
        self.use_3d_center = cfg.get('use_3d_center', True)

        if self.class_merging:
            self.writelist.extend(['Van', 'Truck'])
        if self.use_dontcare:
            self.writelist.extend(['DontCare'])

        # Target image size (W, H)
        resolution = cfg.get('resolution', [1280, 384])
        self.resolution = np.array(resolution)

        # Data split
        assert split in ['train', 'val', 'trainval', 'test']
        self.split_file = os.path.join(self.root_dir, 'ImageSets', split + '.txt')
        self.idx_list = [x.strip() for x in open(self.split_file).readlines()]

        # Paths
        self.data_dir = os.path.join(self.root_dir, 'testing' if split == 'test' else 'training')
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label_2')

        # Augmentation
        self.data_augmentation = split in ['train', 'trainval']
        self.random_flip = cfg.get('random_flip', 0.5)
        self.random_crop = cfg.get('random_crop', 0.5)
        self.scale = cfg.get('scale', 0.4)
        self.shift = cfg.get('shift', 0.1)

        # Image stats
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Class mean sizes (H, W, L) — used as anchor for 3D size residual
        self.cls_mean_size = np.array([
            [1.76255119, 0.66068622, 0.84422524],
            [1.52563191, 1.62856739, 3.52588311],
            [1.73698127, 0.59706367, 1.76282397],
        ], dtype=np.float32)
        if not self.meanshape:
            self.cls_mean_size = np.zeros_like(self.cls_mean_size)

        # Cache
        self._label_cache = {}
        self._calib_cache = {}

    def __len__(self):
        return len(self.idx_list)

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file), f"Image not found: {img_file}"
        return Image.open(img_file)

    def get_label(self, idx):
        if idx in self._label_cache:
            return self._label_cache[idx]
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        labels = get_objects_from_label(label_file)
        self._label_cache[idx] = labels
        return labels

    def get_calib(self, idx):
        if idx in self._calib_cache:
            return self._calib_cache[idx]
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        calib = Calibration(calib_file)
        self._calib_cache[idx] = calib
        return calib

    def eval(self, results_dir, logger, epoch=None):
        """Run KITTI official evaluation with Rich formatted output and CSV saving."""
        import json
        from lib.datasets.kitti.kitti_eval import get_official_eval_result
        import lib.datasets.kitti.kitti_eval.utils as kitti

        logger.info("Loading detections and GTs...")
        img_ids = [int(id) for id in self.idx_list]
        dt_annos = kitti.get_label_annos(results_dir, img_ids)
        gt_annos = kitti.get_label_annos(self.label_dir, img_ids)
        test_id = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        all_metrics = {}

        for category in self.writelist:
            if category not in test_id:
                continue

            results_str, results_dict, rich_data = get_official_eval_result(
                gt_annos, dt_annos, test_id[category]
            )
            all_metrics.update(results_dict)

            # ── Rich 打印 + 历史结果对比 ──
            prev_rich_data = None
            try:
                from lib.helpers.logger_helper import print_kitti_eval_results, save_eval_to_csv

                # 尝试加载上次评估结果
                epoch_dir = os.path.dirname(results_dir)
                viz_dir = os.path.dirname(epoch_dir)
                last_result_path = os.path.join(viz_dir, "last_eval_result.json")

                if not os.path.exists(viz_dir):
                    last_result_path = os.path.join(epoch_dir, "last_eval_result.json")

                # 加载历史
                if os.path.exists(last_result_path):
                    with open(last_result_path, "r") as f:
                        prev_rich_data = json.load(f)

                print_kitti_eval_results(rich_data, prev_rich_data)

                # 更新历史存储
                current_disk_data = []
                try:
                    if os.path.exists(last_result_path):
                        with open(last_result_path, "r") as f:
                            loaded = json.load(f)
                            if isinstance(loaded, list):
                                current_disk_data = [
                                    d for d in loaded
                                    if isinstance(d, dict) and d.get("class_name") != rich_data[0]["class_name"]
                                ]
                except Exception:
                    current_disk_data = []

                current_disk_data.extend(rich_data)
                with open(last_result_path, "w") as f:
                    json.dump(
                        current_disk_data, f,
                        default=lambda o: o.tolist() if hasattr(o, 'tolist') else None,
                    )

                # ── CSV 保存 ──
                csv_path = os.path.join(viz_dir, "eval_results.csv")
                if not os.path.isdir(viz_dir):
                    csv_path = os.path.join(epoch_dir, "eval_results.csv")
                save_eval_to_csv(rich_data, csv_path, model_name="yolo3d", epoch=epoch)

            except ImportError:
                logger.info(results_str)
            except Exception as e:
                logger.warning(f"Failed to save evaluation results: {e}")

        return all_metrics

    def __getitem__(self, item):
        index = int(self.idx_list[item])
        img = self.get_image(index)
        img_size = np.array(img.size)  # (W, H)

        # ---- Augmentation ----
        center = np.array(img_size) / 2
        aug_scale, crop_size = 1.0, img_size
        random_crop_flag, random_flip_flag = False, False

        if self.data_augmentation:
            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.random() < self.random_crop:
                random_crop_flag = True
                aug_scale = np.clip(np.random.randn() * self.scale + 1,
                                    1 - self.scale, 1 + self.scale)
                crop_size = img_size * aug_scale
                center[0] += img_size[0] * np.clip(np.random.randn() * self.shift,
                                                    -2 * self.shift, 2 * self.shift)
                center[1] += img_size[1] * np.clip(np.random.randn() * self.shift,
                                                    -2 * self.shift, 2 * self.shift)

        # Affine transform to target resolution
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        img = img.transform(
            tuple(self.resolution.tolist()),
            method=Image.AFFINE,
            data=tuple(trans_inv.reshape(-1).tolist()),
            resample=Image.BILINEAR,
        )

        # Encode image
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))  # [C, H, W], 0~1

        # ---- Build result dict ----
        # ratio_pad: ultralytics format (ratio_w, ratio_h), (pad_w, pad_h)
        # Our affine transform is equivalent to resizing, no letterbox padding
        W_target, H_target = int(self.resolution[0]), int(self.resolution[1])
        W_orig, H_orig = int(img_size[0]), int(img_size[1])
        ratio = (W_target / W_orig, H_target / H_orig)
        result = {
            "img": img_tensor,
            "im_file": os.path.join(self.image_dir, '%06d.png' % index),
            "ori_shape": (H_orig, W_orig),  # (H, W)
            "resized_shape": (H_target, W_target),  # (H, W)
            "ratio_pad": (ratio, (0.0, 0.0)),  # (ratio, pad)
            "img_id": index,
        }

        if self.split == 'test':
            # Return with empty labels for testing
            result.update({
                "cls": torch.zeros(0, 1),
                "bboxes": torch.zeros(0, 4),
                "batch_idx": torch.zeros(0),
                "mono3d_targets": torch.zeros(0, 9),
            })
            return result

        # ---- Parse labels ----
        objects = self.get_label(index)
        calib = self.get_calib(index)

        if self.bbox2d_type == 'proj':
            for obj in objects:
                obj.box2d_proj = np.array(
                    calib.corners3d_to_img_boxes(obj.generate_corners3d()[None, :])[0][0],
                    dtype=np.float32,
                )
                obj.box2d = obj.box2d_proj.copy()

        # Flip labels
        if random_flip_flag:
            for obj in objects:
                x1, _, x2, _ = obj.box2d
                obj.box2d[0], obj.box2d[2] = img_size[0] - x2, img_size[0] - x1
                obj.alpha = np.pi - obj.alpha
                obj.ry = np.pi - obj.ry
                if obj.alpha > np.pi:
                    obj.alpha -= 2 * np.pi
                if obj.alpha < -np.pi:
                    obj.alpha += 2 * np.pi
                if obj.ry > np.pi:
                    obj.ry -= 2 * np.pi
                if obj.ry < -np.pi:
                    obj.ry += 2 * np.pi

        bboxes_list = []    # normalized xywh
        cls_list = []       # class ids
        mono3d_list = []    # [depth, off3d_x, off3d_y, sz3d_h, sz3d_w, sz3d_l, hbin, hres, mask_3d]

        W, H = self.resolution  # target image size

        for obj in objects:
            if obj.cls_type not in self.writelist:
                continue
            if obj.cls_type in ['Van', 'Truck', 'DontCare']:
                continue
            if obj.level_str == 'UnKnown' or obj.pos[-1] < 2:
                continue
            if obj.pos[-1] > 65:
                continue

            cls_id = self.cls2id[obj.cls_type]

            # Transform 2D bbox to target image coordinates
            bbox_2d = obj.box2d.copy()
            bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
            bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)

            # Clip to image
            bbox_2d[0] = np.clip(bbox_2d[0], 0, W - 1)
            bbox_2d[1] = np.clip(bbox_2d[1], 0, H - 1)
            bbox_2d[2] = np.clip(bbox_2d[2], 0, W - 1)
            bbox_2d[3] = np.clip(bbox_2d[3], 0, H - 1)

            w = bbox_2d[2] - bbox_2d[0]
            h = bbox_2d[3] - bbox_2d[1]
            if w < 2 or h < 2:
                continue

            # Normalized xywh
            cx = (bbox_2d[0] + bbox_2d[2]) / 2.0 / W
            cy = (bbox_2d[1] + bbox_2d[3]) / 2.0 / H
            nw = w / W
            nh = h / H

            bboxes_list.append([cx, cy, nw, nh])
            cls_list.append(cls_id)

            # ---- 3D targets ----
            # Project 3D center to image
            center_3d = obj.pos + [0, -obj.h / 2, 0]
            center_3d = center_3d.reshape(-1, 3)
            center_3d_img, _ = calib.rect_to_img(center_3d)
            center_3d_img = center_3d_img[0]
            if random_flip_flag:
                center_3d_img[0] = img_size[0] - center_3d_img[0]
            center_3d_img = affine_transform(center_3d_img.reshape(-1), trans)

            # Offset 3D: offset of projected 3D center from 2D center, normalized
            center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2.0,
                                   (bbox_2d[1] + bbox_2d[3]) / 2.0])
            offset_3d = center_3d_img - center_2d  # in pixel coords

            depth = obj.pos[-1] * aug_scale

            heading_bin, heading_res = angle2class(obj.alpha)

            mean_size = self.cls_mean_size[cls_id]
            size_3d = np.array([obj.h, obj.w, obj.l], dtype=np.float32) - mean_size

            mask_3d = 0.0 if random_crop_flag else 1.0

            mono3d_list.append([
                depth,
                offset_3d[0], offset_3d[1],
                size_3d[0], size_3d[1], size_3d[2],
                heading_bin, heading_res,
                mask_3d,
            ])

        # Pack into tensors
        if len(bboxes_list) > 0:
            result["bboxes"] = torch.tensor(bboxes_list, dtype=torch.float32)
            result["cls"] = torch.tensor(cls_list, dtype=torch.float32).unsqueeze(-1)
            result["mono3d_targets"] = torch.tensor(mono3d_list, dtype=torch.float32)
        else:
            result["bboxes"] = torch.zeros(0, 4, dtype=torch.float32)
            result["cls"] = torch.zeros(0, 1, dtype=torch.float32)
            result["mono3d_targets"] = torch.zeros(0, 9, dtype=torch.float32)

        # batch_idx placeholder (per-sample, will be adjusted by collate_fn)
        result["batch_idx"] = torch.zeros(result["bboxes"].shape[0], dtype=torch.float32)

        return result

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function compatible with ultralytics training loop.

        Stacks images, concatenates annotations with batch_idx tracking.
        Passes through mono3d_targets with the same convention.
        """
        new_batch = {}
        keys = batch[0].keys()

        for k in keys:
            values = [b[k] for b in batch]
            if k == "img":
                new_batch[k] = torch.stack(values, 0)
            elif k in ("bboxes", "cls", "mono3d_targets"):
                new_batch[k] = torch.cat(values, 0)
            elif k == "batch_idx":
                # Adjust batch indices
                adjusted = []
                for i, v in enumerate(values):
                    adjusted.append(v + i)
                new_batch[k] = torch.cat(adjusted, 0)
            elif k in ("im_file", "img_id"):
                new_batch[k] = values  # keep as list
            elif k in ("ori_shape", "resized_shape", "ratio_pad"):
                new_batch[k] = values
            else:
                new_batch[k] = values

        return new_batch
