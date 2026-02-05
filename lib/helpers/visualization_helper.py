import cv2
import numpy as np
import os
import torch
from lib.datasets.kitti.kitti_utils import get_affine_transform, affine_transform

def visualize_results(images, outputs, targets, info, calibs, cls_mean_size, threshold=0.2, output_dir='visualizations', epoch=0, gt_objects=None, dataset=None):
    """
    Visualizes the predictions and ground truths for a batch of images.
    images: (B, 3, H, W) tensor, normalized
    outputs: model outputs
    targets: ground truth targets
    info: info dict
    calibs: list of calibration objects
    cls_mean_size: mean sizes for classes
    gt_objects: list of lists of Object3d objects
    dataset: Dataset object, used for affine transform information
    """
    from lib.helpers.decode_helper import extract_dets_from_outputs, decode_detections
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract detections
    dets = extract_dets_from_outputs(outputs, K=50)
    dets = dets.detach().cpu().numpy()
    
    # info needs to be in numpy for decode_detections
    info_np = {key: val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else val for key, val in info.items()}
    
    # decode detections
    results = decode_detections(dets=dets,
                                info=info_np,
                                calibs=calibs,
                                cls_mean_size=cls_mean_size,
                                threshold=threshold)
    
    # image normalization parameters
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Extract heatmaps for visualization
    if 'heatmap' in outputs:
        heatmaps = outputs['heatmap'].detach().cpu().numpy()
    else:
        heatmaps = None
    
    for i in range(images.shape[0]):
        img_id = info_np['img_id'][i]
        img_base = images[i].detach().cpu().numpy().transpose(1, 2, 0)
        img_base = (img_base * std + mean) * 255
        img_base = img_base.astype(np.uint8).copy()
        img_base = cv2.cvtColor(img_base, cv2.COLOR_RGB2BGR)

        # Calculate transform if dataset is provided
        trans = None
        if dataset is not None:
             img_w, img_h = info_np['img_size'][i]
             center = np.array([img_w / 2, img_h / 2], dtype=np.float32)
             crop_size = np.array([img_w, img_h], dtype=np.float32)
             trans = get_affine_transform(center, crop_size, 0, dataset.resolution, inv=0)
        
        # --- 1. Visualize Heatmap ---
        if heatmaps is not None:
             hm = heatmaps[i] # (C, H_down, W_down)
             hm_max = np.max(hm, axis=0) # (H_down, W_down)
             # Normalize and Colorize
             hm_max = np.clip(hm_max * 255, 0, 255).astype(np.uint8)
             hm_img = cv2.resize(hm_max, (img_base.shape[1], img_base.shape[0]))
             hm_color = cv2.applyColorMap(hm_img, cv2.COLORMAP_JET)
             img_heatmap = cv2.addWeighted(img_base, 0.6, hm_color, 0.4, 0)
             cv2.imwrite(os.path.join(output_dir, f'epoch_{epoch}_id_{img_id:06d}_heatmap.png'), img_heatmap)

        # --- 2. Visualize 2D Boxes ---
        img_2d = img_base.copy()
        # Draw ground truth 2D
        if gt_objects is not None and i < len(gt_objects):
            for obj in gt_objects[i]:
                if obj.cls_type == 'DontCare': continue
                x1, y1, x2, y2 = obj.box2d
                if trans is not None:
                     pt1 = affine_transform(np.array([x1, y1]), trans)
                     pt2 = affine_transform(np.array([x2, y2]), trans)
                     x1, y1 = pt1
                     x2, y2 = pt2
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(img_2d, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue for GT

        # Draw predictions 2D
        if img_id in results:
            for pred in results[img_id]:
                score = pred[-1]
                if score < threshold: continue
                # pred: [cls_id, alpha, x1, y1, x2, y2, ...]
                x1, y1, x2, y2 = pred[2:6]
                if trans is not None:
                     pt1 = affine_transform(np.array([x1, y1]), trans)
                     pt2 = affine_transform(np.array([x2, y2]), trans)
                     x1, y1 = pt1
                     x2, y2 = pt2
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(img_2d, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green for Pred
        
        cv2.imwrite(os.path.join(output_dir, f'epoch_{epoch}_id_{img_id:06d}_2d.png'), img_2d)

        # --- 3. Visualize 3D Boxes ---
        img_3d = img_base.copy()
        
        # Draw ground truth 3D
        if gt_objects is not None and i < len(gt_objects):
            for obj in gt_objects[i]:
                if obj.cls_type == 'DontCare': continue
                corners3d = obj.generate_corners3d()
                pts_2d, _ = calibs[i].rect_to_img(corners3d)
                if trans is not None:
                     pts_2d_trans = []
                     for pt in pts_2d:
                         pts_2d_trans.append(affine_transform(pt, trans))
                     pts_2d = np.array(pts_2d_trans)

                img_3d = draw_projected_box3d(img_3d, pts_2d, color=(255, 0, 0)) # Blue for GT
        
        # Draw predictions 3D
        if img_id in results:
            for pred in results[img_id]:
                score = pred[-1]
                if score < threshold:
                    continue
                
                # pred: [cls_id, alpha, x1, y1, x2, y2, h, w, l, x, y, z, ry, score]
                h, w, l = pred[6:9]
                pos = pred[9:12]
                ry = pred[12]
                
                # generate corners
                x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
                y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
                z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
                R = np.array([[np.cos(ry), 0, np.sin(ry)],
                              [0, 1, 0],
                              [-np.sin(ry), 0, np.cos(ry)]])
                corners3d = np.vstack([x_corners, y_corners, z_corners])
                corners3d = np.dot(R, corners3d).T + pos
                
                # project to image
                pts_2d, _ = calibs[i].rect_to_img(corners3d)

                if trans is not None:
                     pts_2d_trans = []
                     for pt in pts_2d:
                         pts_2d_trans.append(affine_transform(pt, trans))
                     pts_2d = np.array(pts_2d_trans)

                img_3d = draw_projected_box3d(img_3d, pts_2d, color=(0, 255, 0)) # Green for pred
        
        # Save image
        save_path = os.path.join(output_dir, f'epoch_{epoch}_id_{img_id:06d}_3d.png')
        cv2.imwrite(save_path, img_3d)

def draw_projected_box3d(image, qs, color=(0, 255, 0), thickness=2):
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
    return image
