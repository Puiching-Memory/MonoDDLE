import os
import numpy as np
import torch.utils.data as data
from PIL import Image

from lib.datasets.utils import angle2class
from lib.datasets.utils import gaussian_radius
from lib.datasets.utils import draw_umich_gaussian
from lib.datasets.kitti.kitti_utils import get_objects_from_label
from lib.datasets.kitti.kitti_utils import Calibration
from lib.datasets.kitti.kitti_utils import get_affine_transform
from lib.datasets.kitti.kitti_utils import affine_transform
from lib.datasets.kitti.kitti_eval import get_official_eval_result, get_distance_eval_result
import lib.datasets.kitti.kitti_eval.utils as kitti


class KITTI_Dataset(data.Dataset):
    def __init__(self, split, cfg):
        # basic configuration
        self.root_dir = cfg.get('root_dir', '../../data/KITTI')
        self.split = split
        self.num_classes = 3
        self.max_objs = 50
        self.class_name = ['Pedestrian', 'Car', 'Cyclist']
        self.cls2id = {'Pedestrian': 0, 'Car': 1, 'Cyclist': 2}
        self.resolution = np.array([1280, 384])  # W * H
        self.use_3d_center = cfg.get('use_3d_center', True)
        self.writelist = cfg.get('writelist', ['Car'])
        # anno: use src annotations as GT, proj: use projected 2d bboxes as GT
        self.bbox2d_type = cfg.get('bbox2d_type', 'anno')
        assert self.bbox2d_type in ['anno', 'proj']
        self.meanshape = cfg.get('meanshape', False)
        self.class_merging = cfg.get('class_merging', False)
        self.use_dontcare = cfg.get('use_dontcare', False)

        if self.class_merging:
            self.writelist.extend(['Van', 'Truck'])
        if self.use_dontcare:
            self.writelist.extend(['DontCare'])


        # data split loading
        assert self.split in ['train', 'val', 'trainval', 'test']
        self.split_file = os.path.join(self.root_dir, 'ImageSets', self.split + '.txt')
        self.idx_list = [x.strip() for x in open(self.split_file).readlines()]

        # path configuration
        self.data_dir = os.path.join(self.root_dir, 'testing' if split == 'test' else 'training')
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.depth_dir = os.path.join(self.data_dir, 'depth')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label_2')

        # data augmentation configuration
        self.data_augmentation = True if split in ['train', 'trainval'] else False
        self.random_flip = cfg.get('random_flip', 0.5)
        self.random_crop = cfg.get('random_crop', 0.5)
        self.scale = cfg.get('scale', 0.4)
        self.shift = cfg.get('shift', 0.1)

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.cls_mean_size = np.array([[1.76255119, 0.66068622, 0.84422524],
                                       [1.52563191, 1.62856739, 3.52588311],
                                       [1.73698127, 0.59706367, 1.76282397]], dtype=np.float32)  # H*W*L
        if not self.meanshape:
            self.cls_mean_size = np.zeros_like(self.cls_mean_size, dtype=np.float32)

        # others
        self.downsample = 4
        # cache settings
        self.cache_labels = cfg.get('cache_labels', True)
        self.cache_calibs = cfg.get('cache_calibs', True)
        self._label_cache = {}
        self._calib_cache = {}



    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return Image.open(img_file)


    def get_label(self, idx):
        if self.cache_labels and idx in self._label_cache:
            return self._label_cache[idx]
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        labels = get_objects_from_label(label_file)
        if self.cache_labels:
            self._label_cache[idx] = labels
        return labels


    def get_calib(self, idx):
        if self.cache_calibs and idx in self._calib_cache:
            return self._calib_cache[idx]
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        calib = Calibration(calib_file)
        if self.cache_calibs:
            self._calib_cache[idx] = calib
        return calib

    def eval(self, results_dir, logger):
        logger.info("Loading detections and GTs...")
        img_ids = [int(id) for id in self.idx_list]
        dt_annos = kitti.get_label_annos(results_dir, img_ids)
        gt_annos = kitti.get_label_annos(self.label_dir, img_ids)

        test_id = {'Car': 0, 'Pedestrian':1, 'Cyclist': 2}

        logger.info('Evaluating (official) ...')
        all_metrics = {}
        for category in self.writelist:
            # results_str, results_dict = get_official_eval_result(gt_annos, dt_annos, test_id[category])
            # 接收新增的 rich_data
            results_str, results_dict, rich_data = get_official_eval_result(gt_annos, dt_annos, test_id[category])
            
            # 不再打印 raw string
            # logger.info(results_str) 
            
            all_metrics.update(results_dict)
            
            # 使用 logger_helper 中的 rich 打印函数
            # 注意: logger 可能是 Loguru logger，也可能是我们自定义的 something holding console
            # 这里我们直接从 lib.helpers.logger_helper 导入 print_kitti_eval_results
            from lib.helpers.logger_helper import print_kitti_eval_results
            import os
            import json
            
            # --- Load/Save previous results mechanism ---
            prev_rich_data = None
            try:
                # results_dir typically: .../experiments/run_id/visualizations/epoch_XX/data
                # We want to store in: .../experiments/run_id/visualizations/last_eval_result.json
                
                # Heuristic: go up 3 levels from 'data' to get to 'visualizations' context (roughly)
                # results_dir: .../epoch_1/data
                # parent: .../epoch_1
                # grandparent: .../visualizations (or checkpoints etc)
                
                # Careful handling of paths
                epoch_dir = os.path.dirname(results_dir)    # .../epoch_1
                visualizations_dir = os.path.dirname(epoch_dir) # .../visualizations
                
                last_result_path = os.path.join(visualizations_dir, 'last_eval_result.json')
                
                # If we are not in expected structure, fallback to writing in results_dir parent
                if not os.path.exists(visualizations_dir):
                    last_result_path = os.path.join(epoch_dir, 'last_eval_result.json')

                if os.path.exists(last_result_path):
                     with open(last_result_path, 'r') as f:
                        prev_all_rich = json.load(f)
                        prev_rich_data = prev_all_rich # This might be a list of multiple classes
                
                # To support multiple classes in loop, we need to be careful.
                # 'rich_data' returned by 'get_official_eval_result' is a list (usually len=1 logic per call if single class)
                # But 'all_metrics' accumulates.
                # Let's read/write the full list.
                # However, this loop processes one category at a time.
                # We should append to a persistent list or read partial. 
                # For simplicity, let's just pass what we have loaded.
                
            except Exception as e:
                pass
            
            print_kitti_eval_results(rich_data, prev_rich_data)
            
            # Update the stored json on disk with current new data for THIS category
            if 'last_result_path' in locals():
               # 1. Try to load existing data
               current_disk_data = []
               try:
                   if os.path.exists(last_result_path):
                       with open(last_result_path, 'r') as f:
                           current_disk_data = json.load(f)
               except Exception:
                   # If load fails (e.g. corruption), start fresh
                   current_disk_data = []
               
               # 2. Update and Write
               try:
                   # Remove old entry for this class if exists & data is valid list
                   if isinstance(current_disk_data, list):
                        current_disk_data = [d for d in current_disk_data if isinstance(d, dict) and d.get('class_name') != rich_data[0]['class_name']]
                   else:
                        current_disk_data = []
                        
                   # Add new
                   current_disk_data.extend(rich_data)
                   
                   with open(last_result_path, 'w') as f:
                       # Use default=... to handle numpy arrays
                       json.dump(current_disk_data, f, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else None)
               except Exception as e:
                   logger.warning(f"Failed to save evaluation results: {e}")
        
        return all_metrics

    def __len__(self):
        return self.idx_list.__len__()


    def __getitem__(self, item):
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id
        # image loading
        img = self.get_image(index)
        img_size = np.array(img.size)
        features_size = self.resolution // self.downsample    # W * H

        # data augmentation for image
        center = np.array(img_size) / 2
        aug_scale, crop_size = 1.0, img_size
        random_crop_flag, random_flip_flag = False, False
        if self.data_augmentation:
            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            if np.random.random() < self.random_crop:
                random_crop_flag = True
                aug_scale = np.clip(np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
                crop_size = img_size * aug_scale
                center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)

        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        img = img.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)
        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W

        info = {'img_id': index,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size/features_size}

        if self.split == 'test':
            return img, img, info   # img / placeholder(fake label) / info


        #  ============================   get labels   ==============================
        objects = self.get_label(index)
        calib = self.get_calib(index)

        # computed 3d projected box
        if self.bbox2d_type == 'proj':
            for object in objects:
                object.box2d_proj = np.array(calib.corners3d_to_img_boxes(object.generate_corners3d()[None, :])[0][0], dtype=np.float32)
                object.box2d = object.box2d_proj.copy()

        # data augmentation for labels
        if random_flip_flag:
            for object in objects:
                [x1, _, x2, _] = object.box2d
                object.box2d[0],  object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                object.alpha = np.pi - object.alpha
                object.ry = np.pi - object.ry
                if object.alpha > np.pi:  object.alpha -= 2 * np.pi  # check range
                if object.alpha < -np.pi: object.alpha += 2 * np.pi
                if object.ry > np.pi:  object.ry -= 2 * np.pi
                if object.ry < -np.pi: object.ry += 2 * np.pi


        # labels encoding
        heatmap = np.zeros((self.num_classes, features_size[1], features_size[0]), dtype=np.float32) # C * H * W
        size_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
        offset_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
        depth = np.zeros((self.max_objs, 1), dtype=np.float32)
        heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
        heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
        src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        offset_3d = np.zeros((self.max_objs, 2), dtype=np.float32)
        indices = np.zeros((self.max_objs), dtype=np.int64)
        mask_2d = np.zeros((self.max_objs), dtype=np.uint8)
        mask_3d = np.zeros((self.max_objs), dtype=np.uint8)
        object_num = len(objects) if len(objects) < self.max_objs else self.max_objs
        for i in range(object_num):
            # filter objects by writelist
            if objects[i].cls_type not in self.writelist:
                continue

            # filter inappropriate samples
            if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
                continue

            # ignore the samples beyond the threshold [hard encoding]
            threshold = 65
            if objects[i].pos[-1] > threshold:
                continue

            # process 2d bbox & get 2d center
            bbox_2d = objects[i].box2d.copy()

            # add affine transformation for 2d boxes.
            bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
            bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
            # modify the 2d bbox according to pre-compute downsample ratio
            bbox_2d[:] /= self.downsample

            # process 3d bbox & get 3d center
            center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)  # W * H
            center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
            center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
            center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
            center_3d = center_3d[0]  # shape adjustment
            if random_flip_flag:  # random flip for center3d
                center_3d[0] = img_size[0] - center_3d[0]
            center_3d = affine_transform(center_3d.reshape(-1), trans)
            center_3d /= self.downsample

            # generate the center of gaussian heatmap [optional: 3d center or 2d center]
            center_heatmap = center_3d.astype(np.int32) if self.use_3d_center else center_2d.astype(np.int32)
            if center_heatmap[0] < 0 or center_heatmap[0] >= features_size[0]: continue
            if center_heatmap[1] < 0 or center_heatmap[1] >= features_size[1]: continue

            # generate the radius of gaussian heatmap
            w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
            radius = gaussian_radius((w, h))
            radius = max(0, int(radius))

            if objects[i].cls_type in ['Van', 'Truck', 'DontCare']:
                draw_umich_gaussian(heatmap[1], center_heatmap, radius)
                continue

            cls_id = self.cls2id[objects[i].cls_type]
            draw_umich_gaussian(heatmap[cls_id], center_heatmap, radius)

            # encoding 2d/3d offset & 2d size
            indices[i] = center_heatmap[1] * features_size[0] + center_heatmap[0]
            offset_2d[i] = center_2d - center_heatmap
            size_2d[i] = 1. * w, 1. * h

            # encoding depth
            depth[i] = objects[i].pos[-1] * aug_scale

            # encoding heading angle
            heading_angle = objects[i].alpha
            heading_bin[i], heading_res[i] = angle2class(heading_angle)

            # encoding 3d offset & size_3d
            offset_3d[i] = center_3d - center_heatmap
            src_size_3d[i] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
            mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
            size_3d[i] = src_size_3d[i] - mean_size

            mask_2d[i] = 1
            mask_3d[i] = 0 if random_crop_flag else 1


        # collect return data
        inputs = img
        targets = {'depth': depth,
                   'size_2d': size_2d,
                   'heatmap': heatmap,
                   'offset_2d': offset_2d,
                   'indices': indices,
                   'size_3d': size_3d,
                   'src_size_3d': src_size_3d,
                   'offset_3d': offset_3d,
                   'heading_bin': heading_bin,
                   'heading_res': heading_res,
                   'mask_2d': mask_2d,
                   'mask_3d': mask_3d}
        info = {'img_id': index,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size/features_size}
        return inputs, targets, info




if __name__ == '__main__':
    from torch.utils.data import DataLoader
    cfg = {'root_dir': '../../../data/KITTI',
           'random_flip':0.0, 'random_crop':1.0, 'scale':0.8, 'shift':0.1, 'use_dontcare': False,
           'class_merging': False, 'writelist':['Pedestrian', 'Car', 'Cyclist'], 'use_3d_center':False}
    dataset = KITTI_Dataset('train', cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    print(dataset.writelist)

    for batch_idx, (inputs, targets, info) in enumerate(dataloader):
        # test image
        img = inputs[0].numpy().transpose(1, 2, 0)
        img = (img * dataset.std + dataset.mean) * 255
        img = Image.fromarray(img.astype(np.uint8))
        img.show()
        # print(targets['size_3d'][0][0])

        # test heatmap
        heatmap = targets['heatmap'][0]  # image id
        heatmap = Image.fromarray(heatmap[0].numpy() * 255)  # cats id
        heatmap.show()

        break


    # print ground truth fisrt
    objects = dataset.get_label(0)
    for object in objects:
        print(object.to_kitti_format())
