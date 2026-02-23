"""
可视化脚本：绘制 KITTI 2D/3D 框的真值与预测值对比图，并可选地
渲染检测热力图、DA3 深度图（GT 与模型预测）及不确定性图。

用法 (推荐，使用项目 YAML 配置):
    cd MonoDDLE
    python tools/visualize.py \
        --config experiments/configs/monodle/kitti_da3.yaml \
        --num_images 10

用法 (完全手动指定路径):
    python tools/visualize.py \
        --data_dir data/KITTI \
        --pred_dir experiments/results/monodle/kitti_da3/<timestamp>/outputs/data \
        --split val \
        --num_images 10 \
        --output_dir vis_output

说明:
  --config 会自动从 YAML 中读取 dataset.root_dir、dataset.writelist、
  tester.threshold，并从 tester.checkpoint 路径推导 pred_dir 和 checkpoint。
  若 checkpoint 路径含 <timestamp> 占位符，则自动选择最新的实验目录。
  提供 checkpoint 时会额外输出:
    heatmap/       - 检测中心热力图叠加原图
    da3_gt/        - DA3 真值深度图 (如有 .npz 文件)
    da3_pred/      - 模型预测的密集深度图
    uncertainty/   - 深度预测不确定性图 (仅限 use_distill_uncertainty=True 的模型)
  所有命令行参数均可显式传入以覆盖配置值。
"""
import os
import sys
import glob
import argparse
import random
import numpy as np
import cv2

# ---------- 加入项目根目录到路径 ----------
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from lib.datasets.kitti.kitti_utils import get_objects_from_label, Calibration, get_affine_transform

# ======================== 预处理常量 ========================
# 与 KITTI_Dataset 保持一致
_RESOLUTION = np.array([1280, 384])  # W × H
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# 类别 ID 映射 (同 KITTI_Dataset.cls2id)
_CLS2ID = {'Pedestrian': 0, 'Car': 1, 'Cyclist': 2}
_CLASS_NAMES = ['Pedestrian', 'Car', 'Cyclist']

# ======================== 颜色/类别配置 ========================
CLASS_COLORS = {
    'Car':        (0, 255, 0),    # 绿色
    'Pedestrian': (0, 255, 255),  # 黄色
    'Cyclist':    (255, 0, 255),  # 品红
    'Van':        (255, 128, 0),  # 橙色
    'Truck':      (0, 128, 255),  # 浅蓝
    'Person_sitting': (128, 0, 255),
    'Tram':       (128, 128, 0),
    'Misc':       (128, 128, 128),
}

GT_ALPHA = 1.0
PRED_ALPHA = 0.8

# 默认类别，可被 --config 中的 dataset.writelist 覆盖
VALID_CLASSES = {'Car'}


def get_color(cls_type, is_pred=False):
    """获取类别颜色, 预测值偏红色调"""
    base = CLASS_COLORS.get(cls_type, (200, 200, 200))
    if is_pred:
        # 预测值: 偏红偏亮
        return (min(base[0] + 100, 255), max(base[1] - 50, 0), base[2])
    return base


# ======================== 3D 框绘制 ========================
# 8个角点的连接顺序 (12条边)
EDGES_3D = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
    (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
    (0, 4), (1, 5), (2, 6), (3, 7),  # 侧面竖线
]

def draw_3d_box(img, corners_2d, color, thickness=2):
    """
    在图像上绘制投影后的3D框。
    corners_2d: (8, 2), 8个角点在图像上的坐标
    """
    corners = corners_2d.astype(np.int32)

    for i, j in EDGES_3D:
        pt1 = tuple(corners[i])
        pt2 = tuple(corners[j])
        cv2.line(img, pt1, pt2, color, thickness)

    return img


def project_3d_box_to_image(obj, calib):
    """
    将 Object3d 的3D框投影到图像平面。
    返回 (8, 2) 角点, 如果有角点在相机后方则返回 None。
    """
    corners3d = obj.generate_corners3d()  # (8, 3)

    # 检查是否有角点在相机后方 (z <= 0)
    if np.any(corners3d[:, 2] <= 0):
        return None

    _, corners_2d = calib.corners3d_to_img_boxes(corners3d.reshape(1, 8, 3))
    corners_2d = corners_2d.reshape(8, 2)  # (8, 2)

    return corners_2d


# ======================== 2D 框绘制 ========================
def draw_2d_box(img, box2d, color, thickness=2, label=None):
    """
    绘制2D框。
    box2d: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = [int(v) for v in box2d]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    return img


# ======================== 辅助：按深度排序 ========================
def sort_by_depth(objects):
    """将目标按深度从大到小排序（远→近），近处的框后画压在上面"""
    valid = [o for o in objects if o.cls_type in VALID_CLASSES and o.cls_type != 'DontCare']
    return sorted(valid, key=lambda o: o.pos[2], reverse=True)


# ======================== 主绘制函数 ========================
def _draw_3d_objects(canvas, objects, calib, is_pred, thickness):
    """在 canvas 上按深度顺序绘制一组3D框"""
    for obj in sort_by_depth(objects):
        corners_2d = project_3d_box_to_image(obj, calib)
        if corners_2d is None:
            continue
        color = (0, 0, 255) if is_pred else get_color(obj.cls_type, is_pred=False)
        draw_3d_box(canvas, corners_2d, color, thickness=thickness)


def _draw_2d_objects(canvas, objects, is_pred, thickness):
    """在 canvas 上按深度顺序绘制一组2D框"""
    for obj in sort_by_depth(objects):
        if is_pred:
            color = (0, 0, 255)
            score_str = f"{obj.score:.2f}" if obj.score >= 0 else ""
            label = f"Pred:{obj.cls_type} {score_str}"
        else:
            color = get_color(obj.cls_type, is_pred=False)
            label = f"GT:{obj.cls_type}"
        draw_2d_box(canvas, obj.box2d, color, thickness=thickness, label=label)


def make_split_image(img, gt_canvas, pred_canvas, divider=4):
    """
    上下拼图：上=GT, 下=Pred，中间加分隔线。
    """
    sep = np.zeros((divider, img.shape[1], 3), dtype=np.uint8)
    sep[:] = (80, 80, 80)
    return np.vstack([gt_canvas, sep, pred_canvas])


def add_title_bar(canvas, text, bar_h=24, color=(40, 40, 40), text_color=(255, 255, 255)):
    """在图像顶部插入一个标题栏"""
    h, w = canvas.shape[:2]
    bar = np.full((bar_h, w, 3), color, dtype=np.uint8)
    cv2.putText(bar, text, (8, bar_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)
    return np.vstack([bar, canvas])


def visualize_3d(img, gt_objects, pred_objects, calib, thickness=2, layout='split'):
    """
    layout='split': 上下拼图，上=GT only，下=Pred only （推荐，避免重叠）
    layout='overlay': 叠加在同一张图上（传统方式）
    """
    if layout == 'overlay':
        vis = img.copy()
        _draw_3d_objects(vis, gt_objects, calib, is_pred=False, thickness=thickness)
        _draw_3d_objects(vis, pred_objects, calib, is_pred=True, thickness=thickness)
        add_legend(vis, mode='3d')
        return vis

    # split 模式
    gt_canvas = img.copy()
    _draw_3d_objects(gt_canvas, gt_objects, calib, is_pred=False, thickness=thickness)
    add_legend(gt_canvas, mode='3d', only_gt=True)
    gt_canvas = add_title_bar(gt_canvas, 'GT  (3D)')

    pred_canvas = img.copy()
    _draw_3d_objects(pred_canvas, pred_objects, calib, is_pred=True, thickness=thickness)
    add_legend(pred_canvas, mode='3d', only_pred=True)
    pred_canvas = add_title_bar(pred_canvas, 'Pred  (3D)')

    return make_split_image(img, gt_canvas, pred_canvas)


def visualize_2d(img, gt_objects, pred_objects, thickness=2, layout='split'):
    """
    layout='split': 上下拼图，上=GT only，下=Pred only
    layout='overlay': 叠加在同一张图上
    """
    if layout == 'overlay':
        vis = img.copy()
        _draw_2d_objects(vis, gt_objects, is_pred=False, thickness=thickness)
        _draw_2d_objects(vis, pred_objects, is_pred=True, thickness=thickness)
        add_legend(vis, mode='2d')
        return vis

    # split 模式
    gt_canvas = img.copy()
    _draw_2d_objects(gt_canvas, gt_objects, is_pred=False, thickness=thickness)
    add_legend(gt_canvas, mode='2d', only_gt=True)
    gt_canvas = add_title_bar(gt_canvas, 'GT  (2D)')

    pred_canvas = img.copy()
    _draw_2d_objects(pred_canvas, pred_objects, is_pred=True, thickness=thickness)
    add_legend(pred_canvas, mode='2d', only_pred=True)
    pred_canvas = add_title_bar(pred_canvas, 'Pred  (2D)')

    return make_split_image(img, gt_canvas, pred_canvas)


def add_legend(img, mode='3d', only_gt=False, only_pred=False):
    """在图像左上角添加图例"""
    overlay = img.copy()

    all_items = [
        ("GT (Green)", (0, 255, 0)),
        ("Pred (Red)", (0, 0, 255)),
    ]
    if only_gt:
        legend_items = [all_items[0]]
    elif only_pred:
        legend_items = [all_items[1]]
    else:
        legend_items = all_items

    x0, y0 = 10, 10
    dy = 22
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55

    # 半透明背景
    bg_h = y0 + dy * len(legend_items) + 10
    bg_w = 160
    cv2.rectangle(overlay, (x0 - 5, y0 - 5), (x0 + bg_w, bg_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    for i, (text, color) in enumerate(legend_items):
        y = y0 + dy * i + 15
        cv2.rectangle(img, (x0, y - 10), (x0 + 15, y + 2), color, -1)
        cv2.putText(img, text, (x0 + 22, y), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def _add_class_color_legend(img):
    """在图像左上角添加类别颜色图例（按类名分色）。"""
    items = [(cls, CLASS_COLORS[cls]) for cls in sorted(VALID_CLASSES) if cls in CLASS_COLORS]
    if not items:
        return
    overlay = img.copy()
    x0, y0, dy = 10, 10, 22
    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.55
    bg_h = y0 + dy * len(items) + 10
    bg_w = 180
    cv2.rectangle(overlay, (x0 - 5, y0 - 5), (x0 + bg_w, bg_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    for i, (cls_name, color) in enumerate(items):
        y = y0 + dy * i + 15
        cv2.rectangle(img, (x0, y - 10), (x0 + 15, y + 2), color, -1)
        cv2.putText(img, cls_name, (x0 + 22, y), font, fs,
                    (255, 255, 255), 1, cv2.LINE_AA)


# ======================== 模型推理辅助 ========================

def preprocess_image_for_model(img_bgr):
    """
    将 BGR OpenCV 图像预处理为网络输入 tensor，与 KITTI_Dataset val/test 保持一致。

    变换流程: 仿射变换缩放到 1280×384 → 归一化 → NCHW float32 tensor。

    Args:
        img_bgr: uint8 BGR numpy array (H, W, 3)
    Returns:
        tensor: float32 numpy array (1, 3, 384, 1280), 已按数据集均值/标准差归一化
        trans_mat: (2, 3) 仿射矩阵, 用于把 feature-map 坐标映射回原图
    """
    import PIL.Image as PILImage
    h, w = img_bgr.shape[:2]
    img_size = np.array([w, h], dtype=np.float32)
    center = img_size / 2.0
    trans, _ = get_affine_transform(center, img_size, 0, _RESOLUTION, inv=0)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(img_rgb)
    _, trans_inv = get_affine_transform(center, img_size, 0, _RESOLUTION, inv=1)
    pil_img = pil_img.transform(
        tuple(_RESOLUTION.tolist()),
        method=PILImage.AFFINE,
        data=tuple(trans_inv.reshape(-1).tolist()),
        resample=PILImage.BILINEAR,
    )
    img_arr = np.array(pil_img, dtype=np.float32) / 255.0
    img_arr = (img_arr - _MEAN) / _STD
    tensor = img_arr.transpose(2, 0, 1)[np.newaxis, ...]  # 1 × C × H × W
    return tensor, trans


def load_model_from_config(cfg, checkpoint_path, device):
    """
    根据 YAML config dict 构建并加载模型权重。

    Args:
        cfg: yaml.load 得到的 dict，需含 model 段
        checkpoint_path: checkpoint 文件路径
        device: torch.device
    Returns:
        model: eval 模式下的 nn.Module
    """
    import torch
    from lib.helpers.model_helper import build_model
    from lib.helpers.save_helper import load_checkpoint

    model = build_model(cfg['model'])
    # 创建 dummy logger 兼容 load_checkpoint
    import logging
    dummy_logger = logging.getLogger('vis_loader')
    dummy_logger.addHandler(logging.NullHandler())
    load_checkpoint(
        model=model,
        optimizer=None,
        filename=checkpoint_path,
        map_location=device,
        logger=dummy_logger,
    )
    model.to(device)
    model.eval()
    return model


def run_model_inference(model, img_bgr, device):
    """
    对单张图像进行模型前向推理。

    Args:
        model: 已加载权重的 CenterNet3D
        img_bgr: uint8 BGR numpy array
        device: torch.device
    Returns:
        outputs: 模型输出 dict (tensor 已转 numpy, 去掉 batch dim)
        trans_mat: (2,3) 仿射变换矩阵 (feature grid → 原图)
    """
    import torch
    tensor, trans = preprocess_image_for_model(img_bgr)
    inp = torch.from_numpy(tensor).to(device)
    with torch.no_grad():
        raw = model(inp)
    # 转 numpy，去 batch 维度
    outputs = {}
    for k, v in raw.items():
        if isinstance(v, torch.Tensor):
            outputs[k] = v[0].cpu().numpy()  # (C, H_feat, W_feat)
    return outputs, trans


# ======================== 深度 / 热力图可视化 ========================

def scalar_map_to_bgr(arr_2d, colormap=cv2.COLORMAP_MAGMA, target_hw=None,
                       vmin=None, vmax=None, invert=False):
    """
    将单通道浮点数组转为伪彩色 BGR 图像 (uint8)。

    Args:
        arr_2d: (H, W) float32
        colormap: OpenCV 颜色映射
        target_hw: (target_h, target_w) 输出尺寸; None 则保持原尺寸
        vmin/vmax: 归一化范围; None 则从数据自动推导
        invert: 反转归一化方向 (近→暖 深→冷 时有用)
    Returns:
        BGR uint8 image
    """
    arr = arr_2d.astype(np.float32).copy()
    lo = float(np.nanpercentile(arr, 2)) if vmin is None else vmin
    hi = float(np.nanpercentile(arr, 98)) if vmax is None else vmax
    if hi - lo < 1e-6:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    if invert:
        arr = 1.0 - arr
    arr_u8 = (arr * 255).astype(np.uint8)
    colored = cv2.applyColorMap(arr_u8, colormap)
    if target_hw is not None:
        colored = cv2.resize(colored, (target_hw[1], target_hw[0]),
                             interpolation=cv2.INTER_LINEAR)
    return colored


def draw_colorbar_legend(canvas, label, vmin, vmax, colormap=cv2.COLORMAP_MAGMA,
                          bar_w=20, margin=8):
    """
    在 canvas 右侧绘制垂直渐变色条和最大/最小值标注。

    Returns:
        新 canvas (宽度增加了 bar_w + margin)
    """
    h, w = canvas.shape[:2]
    bar_h = h - 2 * margin
    gradient = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(-1, 1)
    bar = cv2.applyColorMap(np.tile(gradient, (1, bar_w)), colormap)  # (bar_h, bar_w, 3)
    out = np.zeros((h, w + bar_w + margin * 2, 3), dtype=np.uint8)
    out[:, :w] = canvas
    out[margin:margin + bar_h, w + margin:w + margin + bar_w] = bar

    font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1
    cv2.putText(out, f'{vmax:.0f}m', (w + margin, margin + 12), font, fs, (220, 220, 220), ft, cv2.LINE_AA)
    cv2.putText(out, f'{vmin:.0f}m', (w + margin, margin + bar_h - 4), font, fs, (220, 220, 220), ft, cv2.LINE_AA)
    cv2.putText(out, label, (w + margin, h - 4), font, 0.36, (180, 180, 180), 1, cv2.LINE_AA)
    return out


def visualize_heatmap(img_bgr, outputs):
    """
    渲染检测中心热力图（所有类别 max 叠加），叠加在原图上。

    Args:
        img_bgr: 原始 BGR 图像 (H, W, 3)
        outputs: 模型输出 dict，含 'heatmap' key (num_class, H_feat, W_feat)
    Returns:
        BGR uint8 image
    """
    hm = outputs['heatmap']  # (num_class, H_feat, W_feat)
    # sigmoid 激活后取所有类的逐像素最大值
    hm_sigmoid = 1.0 / (1.0 + np.exp(-hm))
    hm_max = hm_sigmoid.max(axis=0)  # (H_feat, W_feat)

    h, w = img_bgr.shape[:2]
    hm_color = scalar_map_to_bgr(hm_max, cv2.COLORMAP_JET, target_hw=(h, w), vmin=0.0, vmax=1.0)
    blended = cv2.addWeighted(img_bgr, 0.45, hm_color, 0.55, 0)
    blended = add_title_bar(blended, 'Heatmap  (detection center, max over classes)')
    return blended


def visualize_heatmap_perclass(img_bgr, outputs):
    """
    逐类别热力图，并排拼合为一张图。

    Args:
        img_bgr: 原始 BGR 图像 (H, W, 3)
        outputs: 模型输出 dict，含 'heatmap' key (num_class, H_feat, W_feat)
    Returns:
        BGR uint8 image 或 None (heatmap 通道数为 0 时)
    """
    hm = outputs['heatmap']  # (C, H_feat, W_feat)
    num_class = hm.shape[0]
    h, w = img_bgr.shape[:2]
    panels = []
    for c in range(num_class):
        hmc = 1.0 / (1.0 + np.exp(-hm[c]))
        colored = scalar_map_to_bgr(hmc, cv2.COLORMAP_JET, target_hw=(h, w), vmin=0.0, vmax=1.0)
        blended = cv2.addWeighted(img_bgr, 0.45, colored, 0.55, 0)
        cls_name = _CLASS_NAMES[c] if c < len(_CLASS_NAMES) else f'cls{c}'
        blended = add_title_bar(blended, f'Heatmap  {cls_name}')
        panels.append(blended)
    if not panels:
        return None
    sep = np.zeros((4, w, 3), dtype=np.uint8)
    sep[:] = (80, 80, 80)
    out = panels[0]
    for p in panels[1:]:
        out = np.vstack([out, sep, p])
    return out


def visualize_da3_gt(img_bgr, da3_dir, idx):
    """
    渲染 DA3 真值深度图 (来自 .npz 文件)。

    Args:
        img_bgr: 原始 BGR 图像 (H, W, 3)
        da3_dir: DA3_depth_results 目录路径
        idx: 图像 ID 字符串 (如 '002657')
    Returns:
        BGR uint8 image, 或 None (文件不存在时)
    """
    npz_path = os.path.join(da3_dir, f'{idx}.npz')
    if not os.path.exists(npz_path):
        return None
    depth = np.load(npz_path)['depth'].astype(np.float32)  # (H_orig, W_orig)
    h, w = img_bgr.shape[:2]
    colored = scalar_map_to_bgr(depth, cv2.COLORMAP_MAGMA, target_hw=(h, w), invert=False)
    colored = add_title_bar(colored, 'DA3 GT Depth  (Depth-Anything-3, metric)')
    vmin = float(np.nanpercentile(depth[depth > 0], 2)) if (depth > 0).any() else 0.0
    vmax = float(np.nanpercentile(depth[depth > 0], 98)) if (depth > 0).any() else 80.0
    colored = draw_colorbar_legend(colored, 'depth(m)', vmin, vmax, cv2.COLORMAP_MAGMA)
    return colored


def visualize_da3_pred(img_bgr, outputs, has_dense_supervision=True):
    """
    渲染模型预测的密集深度图 (CenterNet3D depth 头第 0 通道)。

    模型 depth 头输出的是未经激活的 raw logits，需做 inverse-sigmoid 变换
    还原为真实深度（米）：depth = 1 / (sigmoid(raw) + eps) - 1

    注意: 若模型未经密集深度蒸馏 (distill.lambda=0)，depth 头仅在物体中心点
    被稀疏监督，密集深度图不具参考价值。

    Args:
        img_bgr: 原始 BGR 图像 (H, W, 3)
        outputs: 模型输出 dict，含 'depth' key (2, H_feat, W_feat)
                 channel 0 = raw depth logit, channel 1 = log_variance
        has_dense_supervision: 模型是否经过密集深度蒸馏训练
    Returns:
        BGR uint8 image
    """
    raw = outputs['depth'][0].astype(np.float32)  # (H_feat, W_feat) raw logit
    # inverse-sigmoid: 与 decode_helper.py / centernet_loss.py 保持一致
    sigmoid_raw = 1.0 / (1.0 + np.exp(-raw))
    depth_pred = 1.0 / (sigmoid_raw + 1e-6) - 1.0  # 真实深度 (米)
    h, w = img_bgr.shape[:2]
    colored = scalar_map_to_bgr(depth_pred, cv2.COLORMAP_MAGMA, target_hw=(h, w), invert=False)
    if has_dense_supervision:
        title = 'Predicted Dense Depth  (depth head, ch0)'
    else:
        title = 'Predicted Dense Depth  (depth head, ch0)  [NO dense supervision — sparse only, not meaningful]'
    colored = add_title_bar(colored, title)
    vmin = float(np.percentile(depth_pred, 2))
    vmax = float(np.percentile(depth_pred, 98))
    colored = draw_colorbar_legend(colored, 'depth(m)', vmin, vmax, cv2.COLORMAP_MAGMA)
    return colored


def visualize_uncertainty(img_bgr, outputs):
    """
    渲染密集深度不确定性图 (仅当模型包含 dense_depth_uncertainty 头时)。

    log_variance → sigma = exp(0.5 * log_var), 较大的 sigma 表示更高的不确定性。

    Args:
        img_bgr: 原始 BGR 图像 (H, W, 3)
        outputs: 模型输出 dict; 若不含 'dense_depth_uncertainty' 则返回 None
    Returns:
        BGR uint8 image 或 None
    """
    if 'dense_depth_uncertainty' not in outputs:
        return None
    log_var = outputs['dense_depth_uncertainty'][0]  # (H_feat, W_feat)
    sigma = np.exp(0.5 * log_var.astype(np.float32))  # aleatoric std
    h, w = img_bgr.shape[:2]
    # 高不确定性显示为暖色 (反转: sigma 大 → 值大 → 暖色)
    colored = scalar_map_to_bgr(sigma, cv2.COLORMAP_HOT, target_hw=(h, w), invert=False)
    blended = cv2.addWeighted(img_bgr, 0.3, colored, 0.7, 0)
    blended = add_title_bar(blended, 'Depth Uncertainty  sigma = exp(0.5*log_var)  [hot = high]')
    vmin = float(np.percentile(sigma, 2))
    vmax = float(np.percentile(sigma, 98))
    blended = draw_colorbar_legend(blended, 'sigma', vmin, vmax, cv2.COLORMAP_HOT)
    return blended


# ======================== LiDAR BEV 可视化 ========================

# BEV 范围 (相机坐标系: x=右, y=下, z=前)
_BEV_X_RANGE = (-40.0, 40.0)   # 左右 (m)
_BEV_Z_RANGE = (-5.0, 70.0)    # 前方深度 (m), 负值含自车后方
_BEV_Y_RANGE = (-3.0, 2.0)     # 高度过滤 (m), 包含地面点 (y≈1.65)
_BEV_RESOLUTION = 0.05          # 每像素对应 0.05m

# 参考图配色: GT=cyan, Pred=orange
_BEV_GT_COLOR = (255, 255, 0)    # BGR: cyan
_BEV_PRED_COLOR = (0, 165, 255)  # BGR: orange

# ---- KITTI 自车参数 (VW Passat B6) ----
# 参考 KITTI 官方 setup: camera height=1.65m, car≈4.78×1.82×1.47m
# 相机坐标系原点在 Cam2 (x=right, y=down, z=forward)
_EGO_LENGTH = 4.78   # 纵向 (m)
_EGO_WIDTH  = 1.82   # 横向 (m)
_EGO_HEIGHT = 1.47   # 高度 (m)
_EGO_CAM_H  = 1.65   # 相机离地高度 (m)
# 相机距前保险杠约 1.74m (前悬0.94 + 前轴到相机0.80)
_EGO_CAM_TO_FRONT = 1.74
_EGO_CAM_TO_REAR  = _EGO_LENGTH - _EGO_CAM_TO_FRONT  # ≈3.04m
# 相机大致居中, 略偏左 ~0.06m, 此处忽略
_EGO_HALF_W = _EGO_WIDTH / 2.0   # 0.91m


def _ego_corners_3d():
    """生成自车 8 个角点 (相机 rect 坐标系)。

    Returns:
        corners: (8, 3) 点阵, 与 Object3d.generate_corners3d() 一致的顺序
        底面 0-3 先行, 顶面 4-7 随后, 前面 = 0,1
    """
    x_half = _EGO_HALF_W
    z_front = _EGO_CAM_TO_FRONT     # +z 前方
    z_rear  = -_EGO_CAM_TO_REAR     # -z 后方
    y_top   = _EGO_CAM_H - _EGO_HEIGHT  # 车顶 (≈0.18)
    y_bot   = _EGO_CAM_H               # 车底 = 地面 (1.65)
    #       x        y      z
    return np.array([
        [ x_half, y_bot, z_front],   # 0: 右前底
        [-x_half, y_bot, z_front],   # 1: 左前底
        [-x_half, y_bot, z_rear ],   # 2: 左后底
        [ x_half, y_bot, z_rear ],   # 3: 右后底
        [ x_half, y_top, z_front],   # 4: 右前顶
        [-x_half, y_top, z_front],   # 5: 左前顶
        [-x_half, y_top, z_rear ],   # 6: 左后顶
        [ x_half, y_top, z_rear ],   # 7: 右后顶
    ], dtype=np.float64)


def load_velodyne(velo_path):
    """加载 KITTI velodyne 点云 (.bin, float32×4)。

    Args:
        velo_path: .bin 文件路径
    Returns:
        points: (N, 4) [x, y, z, reflectance] in velodyne coordinates
    """
    return np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)


_BEV_POINT_RADIUS = 2  # 点云绘制半径 (像素)


def _points_to_bev(pts_rect, y_range=_BEV_Y_RANGE, x_range=_BEV_X_RANGE,
                   z_range=_BEV_Z_RANGE, res=_BEV_RESOLUTION,
                   point_radius=_BEV_POINT_RADIUS):
    """将相机坐标系点云渲染为 BEV 伪彩色图。

    使用高度作为颜色通道 (低→暗紫, 高→亮黄, 类似 magma)。

    Args:
        pts_rect: (N, 3+) 相机坐标系点云, 至少含 x, y, z
        point_radius: 每个点的绘制半径 (像素)
    Returns:
        bev_bgr: uint8 BGR image, shape (H_bev, W_bev, 3)
    """
    x, y, z = pts_rect[:, 0], pts_rect[:, 1], pts_rect[:, 2]
    # 过滤范围
    mask = ((x >= x_range[0]) & (x <= x_range[1]) &
            (y >= y_range[0]) & (y <= y_range[1]) &
            (z >= z_range[0]) & (z <= z_range[1]))
    x, y, z = x[mask], y[mask], z[mask]

    W_bev = int((x_range[1] - x_range[0]) / res)
    H_bev = int((z_range[1] - z_range[0]) / res)

    # BEV 像素坐标: u=x→右, v=z→上 (远在顶部)
    u = ((x - x_range[0]) / res).astype(np.int32)
    v = H_bev - 1 - ((z - z_range[0]) / res).astype(np.int32)
    u = np.clip(u, 0, W_bev - 1)
    v = np.clip(v, 0, H_bev - 1)

    # 按高度归一化到 0-255, 用 MAGMA 查表得到 BGR 颜色
    y_norm = np.clip((y - y_range[0]) / (y_range[1] - y_range[0]), 0.0, 1.0)
    y_u8 = (y_norm * 255).astype(np.uint8)
    # 预生成 256 色 LUT
    lut_gray = np.arange(256, dtype=np.uint8).reshape(1, 256)
    lut_bgr = cv2.applyColorMap(lut_gray, cv2.COLORMAP_MAGMA).reshape(256, 3)

    # 浅色背景
    bev_bgr = np.full((H_bev, W_bev, 3), [235, 235, 230], dtype=np.uint8)

    # 按高度排序: 低点先画, 高点后画 (高的覆盖低的)
    order = np.argsort(-y)  # y 大 = 低处, 先画
    for idx in order:
        color = tuple(int(c) for c in lut_bgr[y_u8[idx]])
        cv2.circle(bev_bgr, (int(u[idx]), int(v[idx])), point_radius,
                   color, -1, cv2.LINE_AA)

    return bev_bgr


def _rect_to_bev_coords(pts_3d, x_range=_BEV_X_RANGE, z_range=_BEV_Z_RANGE,
                        res=_BEV_RESOLUTION):
    """将相机坐标系 3D 点转为 BEV 像素坐标。

    Args:
        pts_3d: (N, 3) 相机坐标系点
    Returns:
        pts_bev: (N, 2) int32, [u, v]
    """
    H_bev = int((z_range[1] - z_range[0]) / res)
    u = ((pts_3d[:, 0] - x_range[0]) / res).astype(np.int32)
    v = H_bev - 1 - ((pts_3d[:, 2] - z_range[0]) / res).astype(np.int32)
    return np.stack([u, v], axis=1)


def _draw_bev_3d_box(canvas, obj, color, thickness=2):
    """在 BEV 画布上绘制单个目标的俯视旋转框 + 朝向箭头。

    Args:
        canvas: BEV BGR image
        obj: Object3d 实例
        color: BGR tuple
        thickness: 线宽
    """
    corners3d = obj.generate_corners3d()  # (8, 3) 相机坐标
    # 底面四角 (indices 0-3) 即为 BEV 投影
    bottom4 = corners3d[:4]  # (4, 3)
    bev4 = _rect_to_bev_coords(bottom4)  # (4, 2)

    # 绘制四边形
    for i in range(4):
        pt1 = tuple(bev4[i])
        pt2 = tuple(bev4[(i + 1) % 4])
        cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)

    # 朝向箭头: 前面中点 → 中心
    front_mid = ((bev4[0] + bev4[1]) / 2).astype(np.int32)
    center = bev4.mean(axis=0).astype(np.int32)
    cv2.arrowedLine(canvas, tuple(center), tuple(front_mid), color,
                    max(1, thickness - 1), cv2.LINE_AA, tipLength=0.3)

    # 在框中心画×
    arm = 4
    cx, cy = int(center[0]), int(center[1])
    cv2.line(canvas, (cx - arm, cy - arm), (cx + arm, cy + arm), color, 1, cv2.LINE_AA)
    cv2.line(canvas, (cx - arm, cy + arm), (cx + arm, cy - arm), color, 1, cv2.LINE_AA)


def visualize_lidar_bev(velo_path, calib, gt_objects, pred_objects,
                        thickness=2, img_size=None):
    """渲染 LiDAR 点云 BEV 视图 + GT/Pred 3D 框 + 相机 FOV。

    Args:
        velo_path: velodyne .bin 文件路径
        calib: Calibration 实例
        gt_objects: GT Object3d 列表
        pred_objects: Pred Object3d 列表 (已过滤)
        thickness: 线宽
        img_size: (W, H) 图像尺寸, 用于绘制 FOV
    Returns:
        BEV BGR uint8 image, 或 None (文件不存在时)
    """
    if not os.path.exists(velo_path):
        return None

    # 加载点云并转到相机坐标
    pts_velo = load_velodyne(velo_path)
    pts_rect = calib.lidar_to_rect(pts_velo[:, :3])

    # 渲染 BEV 底图
    canvas = _points_to_bev(pts_rect)

    gt_valid = sort_by_depth([o for o in gt_objects if o.cls_type in VALID_CLASSES])
    pred_valid = sort_by_depth(pred_objects)

    for obj in gt_valid:
        _draw_bev_3d_box(canvas, obj, _BEV_GT_COLOR, thickness)
    for obj in pred_valid:
        _draw_bev_3d_box(canvas, obj, _BEV_PRED_COLOR, thickness)
    _draw_bev_camera_fov(canvas, calib, img_size)
    _add_bev_legend(canvas)
    _draw_bev_distance_grid(canvas)
    canvas = add_title_bar(canvas, 'LiDAR BEV  (GT: cyan | Pred: orange)')
    return canvas


def _draw_bev_distance_grid(canvas, x_range=_BEV_X_RANGE, z_range=_BEV_Z_RANGE,
                            res=_BEV_RESOLUTION):
    """在 BEV 图上绘制距离参考线 (每 10m 一条水平线 + 每 10m 一条垂直线)。"""
    H_bev = int((z_range[1] - z_range[0]) / res)
    W_bev = int((x_range[1] - x_range[0]) / res)
    font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    line_color = (180, 180, 180)  # 淡灰
    text_color = (100, 100, 100)

    # 水平线: 每 10m 深度
    for z_m in range(int(z_range[0]), int(z_range[1]) + 1, 10):
        if z_m == 0:
            continue
        v = H_bev - 1 - int((z_m - z_range[0]) / res)
        if 0 <= v < H_bev:
            cv2.line(canvas, (0, v), (W_bev - 1, v), line_color, 1)
            cv2.putText(canvas, f'{z_m}m', (W_bev - 50, v - 4),
                        font, fs, text_color, ft, cv2.LINE_AA)

    # 垂直线: 每 10m 左右
    for x_m in range(int(x_range[0]), int(x_range[1]) + 1, 10):
        u = int((x_m - x_range[0]) / res)
        if 0 <= u < W_bev:
            cv2.line(canvas, (u, 0), (u, H_bev - 1), line_color, 1)
            if x_m != 0:
                cv2.putText(canvas, f'{x_m}m', (u + 3, H_bev - 8),
                            font, fs, text_color, ft, cv2.LINE_AA)

    # 自车 3D 框 (红色)
    _draw_bev_ego_car(canvas, x_range, z_range, res)


def _draw_bev_ego_car(canvas, x_range=_BEV_X_RANGE,
                      z_range=_BEV_Z_RANGE, res=_BEV_RESOLUTION):
    """在 BEV 上绘制 KITTI 自车 (VW Passat) 俯视 3D 框 + 朝向箭头。"""
    H_bev = int((z_range[1] - z_range[0]) / res)
    corners = _ego_corners_3d()  # (8,3)
    # 底面 4 角: indices 0-3
    bottom4 = corners[:4]
    bev4 = np.zeros((4, 2), dtype=np.int32)
    for i in range(4):
        bev4[i, 0] = int((bottom4[i, 0] - x_range[0]) / res)
        bev4[i, 1] = H_bev - 1 - int((bottom4[i, 2] - z_range[0]) / res)

    ego_color = (0, 0, 220)  # BGR: 深红
    # 半透明填充
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [bev4], (0, 0, 120))
    cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
    # 边框
    for i in range(4):
        pt1 = tuple(bev4[i])
        pt2 = tuple(bev4[(i + 1) % 4])
        cv2.line(canvas, pt1, pt2, ego_color, 2, cv2.LINE_AA)
    # 朝向箭头 (前面中点)
    front_mid = ((bev4[0] + bev4[1]) / 2).astype(np.int32)
    center = bev4.mean(axis=0).astype(np.int32)
    cv2.arrowedLine(canvas, tuple(center), tuple(front_mid), ego_color,
                    2, cv2.LINE_AA, tipLength=0.25)
    # 标注 "EGO"
    cv2.putText(canvas, 'EGO', (center[0] - 15, center[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    # 相机位置十字 (x=0, z=0)
    cam_u = int((0 - x_range[0]) / res)
    cam_v = H_bev - 1 - int((0 - z_range[0]) / res)
    arm = 5
    cv2.line(canvas, (cam_u - arm, cam_v), (cam_u + arm, cam_v),
             (0, 255, 255), 1, cv2.LINE_AA)
    cv2.line(canvas, (cam_u, cam_v - arm), (cam_u, cam_v + arm),
             (0, 255, 255), 1, cv2.LINE_AA)


def _draw_bev_camera_fov(canvas, calib, img_size=None,
                          x_range=_BEV_X_RANGE, z_range=_BEV_Z_RANGE,
                          res=_BEV_RESOLUTION):
    """在 BEV 图上绘制相机可视角度 (FOV) 的两条边线。

    根据 P2 内参 (fu, cu) 和图像宽度计算水平 FOV，
    从相机原点向前方绘制两条射线。
    """
    if img_size is None:
        return
    img_w, _img_h = img_size
    fu, cu = calib.fu, calib.cu

    # 图像左/右边缘对应的 x 方向正切值
    tan_left = (0 - cu) / fu        # 负值 (左侧)
    tan_right = (img_w - cu) / fu    # 正值 (右侧)

    H_bev = int((z_range[1] - z_range[0]) / res)
    W_bev = int((x_range[1] - x_range[0]) / res)

    # 相机原点 BEV 坐标
    cam_u = int((0 - x_range[0]) / res)
    cam_v = H_bev - 1 - int((0 - z_range[0]) / res)

    # z=z_range[1] 处的左右 x 坐标
    z_far = z_range[1]
    x_left = tan_left * z_far
    x_right = tan_right * z_far

    far_left_u = int((x_left - x_range[0]) / res)
    far_left_v = H_bev - 1 - int((z_far - z_range[0]) / res)
    far_right_u = int((x_right - x_range[0]) / res)
    far_right_v = far_left_v

    # 绘制 FOV 边线 (黄绿色, 半透明)
    fov_color = (50, 200, 200)  # BGR: 黄绿
    overlay = canvas.copy()
    cv2.line(overlay, (cam_u, cam_v), (far_left_u, far_left_v),
             fov_color, 2, cv2.LINE_AA)
    cv2.line(overlay, (cam_u, cam_v), (far_right_u, far_right_v),
             fov_color, 2, cv2.LINE_AA)
    # 填充 FOV 区域 (极淡透明)
    pts_poly = np.array([[cam_u, cam_v],
                         [far_left_u, far_left_v],
                         [far_right_u, far_right_v]], dtype=np.int32)
    cv2.fillPoly(overlay, [pts_poly], fov_color)
    cv2.addWeighted(overlay, 0.10, canvas, 0.90, 0, canvas)
    # 再画边线 (实线不透明)
    cv2.line(canvas, (cam_u, cam_v), (far_left_u, far_left_v),
             fov_color, 2, cv2.LINE_AA)
    cv2.line(canvas, (cam_u, cam_v), (far_right_u, far_right_v),
             fov_color, 2, cv2.LINE_AA)

    # 标注 FOV 角度
    import math
    hfov = math.degrees(math.atan(tan_right) - math.atan(tan_left))
    cv2.putText(canvas, f'HFOV {hfov:.0f}', (cam_u + 8, cam_v - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, fov_color, 1, cv2.LINE_AA)


def _add_bev_legend(img, only_gt=False, only_pred=False):
    """在 BEV 图左上角添加图例。"""
    all_items = [
        ("GT (Cyan)", _BEV_GT_COLOR),
        ("Pred (Orange)", _BEV_PRED_COLOR),
    ]
    if only_gt:
        items = [all_items[0]]
    elif only_pred:
        items = [all_items[1]]
    else:
        items = all_items

    overlay = img.copy()
    x0, y0, dy = 10, 10, 22
    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.55
    bg_h = y0 + dy * len(items) + 10
    cv2.rectangle(overlay, (5, 5), (175, bg_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    for i, (text, color) in enumerate(items):
        y = y0 + dy * i + 15
        cv2.rectangle(img, (x0, y - 10), (x0 + 15, y + 2), color, -1)
        cv2.putText(img, text, (x0 + 22, y), font, fs, (255, 255, 255), 1, cv2.LINE_AA)


# ======================== LiDAR 透视视角 (45° 俯视) ========================

# 透视投影参数
_PERSP_IMG_W = 1600
_PERSP_IMG_H = 900
_PERSP_POINT_RADIUS = 2

# 虚拟相机: 从自车后上方俯视前方, 保证自车区域可见
_CAM_EYE = np.array([0.0, -10.0, -8.0])     # 自车后方 8m, 上方 10m
_CAM_LOOKAT = np.array([0.0, 0.5, 20.0])   # 注视前方 20m 处地面偏上
_CAM_UP = np.array([0.0, -1.0, 0.0])       # 上方向

_PERSP_FOCAL = 500.0  # 焦距 (稍广角减少空白)


def _build_view_matrix(eye, lookat, up):
    """构建 view matrix (world → camera)。

    Args:
        eye: (3,) 相机位置
        lookat: (3,) 注视点
        up: (3,) 上方向
    Returns:
        R: (3, 3), t: (3,)  满足 p_cam = R @ p_world + t
    """
    fwd = lookat - eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)  # 相机 y 轴 (向下)
    R = np.stack([right, down, fwd], axis=0)  # (3, 3)
    t = -R @ eye
    return R, t


def _project_perspective(pts_3d, R, t, focal, cx, cy):
    """将 3D 点投影到虚拟透视相机。

    Args:
        pts_3d: (N, 3)
        R, t: view matrix
        focal, cx, cy: 内参
    Returns:
        uv: (N, 2) int32 像素坐标
        depth: (N,) 深度 (用于 z-buffer 排序)
        valid: (N,) bool mask (深度 > 0 的点)
    """
    pts_cam = (R @ pts_3d.T).T + t  # (N, 3)
    depth = pts_cam[:, 2]
    valid = depth > 0.5
    u = (focal * pts_cam[:, 0] / np.maximum(depth, 1e-6) + cx).astype(np.int32)
    v = (focal * pts_cam[:, 1] / np.maximum(depth, 1e-6) + cy).astype(np.int32)
    uv = np.stack([u, v], axis=1)
    return uv, depth, valid


def _draw_persp_3d_box(canvas, corners3d, R, t, focal, cx, cy,
                       color, thickness=2):
    """在透视视角画布上绘制 3D 框 (12 条边)。

    Args:
        canvas: BGR image
        corners3d: (8, 3) 3D 角点
        R, t: view matrix
        focal, cx, cy: 内参
        color: BGR tuple
        thickness: 线宽
    """
    uv, depth, valid = _project_perspective(corners3d, R, t, focal, cx, cy)
    if not valid.all():
        return
    h, w = canvas.shape[:2]
    # 检查是否有点在画面内
    in_frame = ((uv[:, 0] >= -200) & (uv[:, 0] < w + 200) &
                (uv[:, 1] >= -200) & (uv[:, 1] < h + 200))
    if not in_frame.any():
        return
    for i, j in EDGES_3D:
        pt1 = (int(uv[i, 0]), int(uv[i, 1]))
        pt2 = (int(uv[j, 0]), int(uv[j, 1]))
        cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)


def visualize_lidar_perspective(velo_path, calib, gt_objects, pred_objects,
                                thickness=2):
    """渲染从车顶 45° 斜向下的 LiDAR 透视视图 + GT/Pred 3D 框。

    Args:
        velo_path: velodyne .bin 文件路径
        calib: Calibration 实例
        gt_objects: GT Object3d 列表
        pred_objects: Pred Object3d 列表 (已过滤)
        thickness: 线宽
    Returns:
        BGR uint8 image, 或 None
    """
    if not os.path.exists(velo_path):
        return None

    pts_velo = load_velodyne(velo_path)
    pts_rect = calib.lidar_to_rect(pts_velo[:, :3])

    # 过滤范围
    mask = ((pts_rect[:, 0] >= _BEV_X_RANGE[0]) &
            (pts_rect[:, 0] <= _BEV_X_RANGE[1]) &
            (pts_rect[:, 1] >= _BEV_Y_RANGE[0]) &
            (pts_rect[:, 1] <= _BEV_Y_RANGE[1]) &
            (pts_rect[:, 2] >= _BEV_Z_RANGE[0]) &
            (pts_rect[:, 2] <= _BEV_Z_RANGE[1]))
    pts = pts_rect[mask]

    W, H = _PERSP_IMG_W, _PERSP_IMG_H
    cx, cy = W / 2.0, H / 2.0
    focal = _PERSP_FOCAL

    R, t = _build_view_matrix(_CAM_EYE, _CAM_LOOKAT, _CAM_UP)

    # 投影点云
    uv, depth, valid = _project_perspective(pts, R, t, focal, cx, cy)
    uv, depth, pts_y = uv[valid], depth[valid], pts[valid, 1]

    # 浅色背景
    canvas = np.full((H, W, 3), [235, 235, 230], dtype=np.uint8)

    # 先绘制距离参考线 (在点云层之下)
    _draw_persp_distance_lines(canvas, R, t, focal, cx, cy, W, H)

    # 按深度排序: 远处先画, 近处后画
    order = np.argsort(-depth)
    uv, pts_y = uv[order], pts_y[order]

    # 高度着色
    y_norm = np.clip((pts_y - _BEV_Y_RANGE[0]) /
                     (_BEV_Y_RANGE[1] - _BEV_Y_RANGE[0]), 0.0, 1.0)
    y_u8 = (y_norm * 255).astype(np.uint8)
    lut_gray = np.arange(256, dtype=np.uint8).reshape(1, 256)
    lut_bgr = cv2.applyColorMap(lut_gray, cv2.COLORMAP_MAGMA).reshape(256, 3)

    for k in range(len(uv)):
        px, py = int(uv[k, 0]), int(uv[k, 1])
        if 0 <= px < W and 0 <= py < H:
            color = tuple(int(c) for c in lut_bgr[y_u8[k]])
            cv2.circle(canvas, (px, py), _PERSP_POINT_RADIUS, color,
                       -1, cv2.LINE_AA)

    # 绘制 3D 框
    gt_valid = sort_by_depth([o for o in gt_objects if o.cls_type in VALID_CLASSES])
    pred_valid = sort_by_depth(pred_objects)
    for obj in gt_valid:
        corners3d = obj.generate_corners3d()
        _draw_persp_3d_box(canvas, corners3d, R, t, focal, cx, cy,
                           _BEV_GT_COLOR, thickness)
    for obj in pred_valid:
        corners3d = obj.generate_corners3d()
        _draw_persp_3d_box(canvas, corners3d, R, t, focal, cx, cy,
                           _BEV_PRED_COLOR, thickness)

    # 自车 3D 框 (深红色)
    ego_corners = _ego_corners_3d()
    _draw_persp_3d_box(canvas, ego_corners, R, t, focal, cx, cy,
                       (0, 0, 220), thickness + 1)
    # 标注 "EGO"
    ego_center = ego_corners.mean(axis=0, keepdims=True)
    uv_ego, _, v_ego = _project_perspective(ego_center, R, t, focal, cx, cy)
    if v_ego.any():
        cv2.putText(canvas, 'EGO',
                    (int(uv_ego[0, 0]) - 15, int(uv_ego[0, 1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2, cv2.LINE_AA)

    # 裁剪顶部空白区域 (背景色 [235,235,230])
    canvas = _crop_persp_top_blank(canvas)
    _add_bev_legend(canvas)
    canvas = add_title_bar(canvas, 'LiDAR Perspective  (elevated rear view)')
    return canvas


def _crop_persp_top_blank(canvas, bg_color=(235, 235, 230), margin=20):
    """裁剪透视图顶部的空白行, 保留少量 margin。

    逐行检测是否与背景色接近, 找到第一行有内容的位置后裁剪。
    """
    h, w = canvas.shape[:2]
    # 每行与背景色的差异
    bg = np.array(bg_color, dtype=np.float32)
    row_diff = np.abs(canvas.astype(np.float32) - bg).max(axis=2).max(axis=1)
    # 差异 > 15 即认为有内容
    content_rows = np.where(row_diff > 15)[0]
    if len(content_rows) == 0:
        return canvas
    first_row = max(0, content_rows[0] - margin)
    if first_row <= 10:
        return canvas  # 几乎没有空白, 不裁剪
    return canvas[first_row:, :, :]


def _draw_persp_distance_lines(canvas, R, t, focal, cx, cy, W, H):
    """在透视视图中绘制深度参考线 (每 10m 一条, 在路面平面 y≈1.65 上)。

    参考线使用浅色绘制, 距离标注放在画面左右边缘, 不干扰主视图。
    """
    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.45
    road_y = 1.65  # 路面高度 (相机坐标系, y 向下, KITTI 典型值)
    line_color = (210, 210, 205)  # 极浅灰, 接近背景
    text_color = (140, 140, 140)
    margin = 8  # 距边缘像素
    for z_m in range(10, 71, 10):
        # 在 z=z_m, y=road_y 平面绘一条 x 方向的线
        pts = np.array([[-40, road_y, z_m], [40, road_y, z_m]], dtype=np.float32)
        uv, _, valid = _project_perspective(pts, R, t, focal, cx, cy)
        if not valid.all():
            continue
        x1, y1 = int(uv[0, 0]), int(uv[0, 1])
        x2, y2 = int(uv[1, 0]), int(uv[1, 1])
        if (y1 < 0 and y2 < 0) or (y1 >= H and y2 >= H):
            continue
        cv2.line(canvas, (x1, y1), (x2, y2), line_color, 1, cv2.LINE_AA)
        # 标注距离 — 左侧边缘
        label = f'{z_m}m'
        (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
        # 计算线在左边缘 (x=margin) 处的 y 坐标 (线性插值)
        if x2 != x1:
            y_at_left = int(y1 + (y2 - y1) * (margin - x1) / (x2 - x1))
        else:
            y_at_left = y1
        if 0 <= y_at_left < H:
            cv2.putText(canvas, label, (margin, y_at_left - 3),
                        font, fs, text_color, 1, cv2.LINE_AA)


# ======================== 组合 3D 可视化 (Pred + LiDAR 透视, 按类别着色) ========================

def visualize_combined_pred_perspective(img, pred_objects, calib, velo_path,
                                       gt_objects=None, thickness=2):
    """生成组合图: 上=相机视角 Pred 3D 框, 下=LiDAR 透视 3D 视图, 按类别着色。

    Args:
        img: 原始 BGR 图像 (H, W, 3)
        pred_objects: 预测目标 Object3d 列表 (已过滤)
        calib: Calibration 实例
        velo_path: velodyne .bin 文件路径
        gt_objects: GT 目标 Object3d 列表 (可选, 在透视图中薄线显示)
        thickness: 线宽
    Returns:
        BGR uint8 image, 或 None (点云文件不存在时)
    """
    if not os.path.exists(velo_path):
        return None

    # ---- 上半部: 相机视角 Pred 3D 框 (按类别着色) ----
    pred_canvas = img.copy()
    for obj in sort_by_depth(pred_objects):
        corners_2d = project_3d_box_to_image(obj, calib)
        if corners_2d is None:
            continue
        color = CLASS_COLORS.get(obj.cls_type, (200, 200, 200))
        draw_3d_box(pred_canvas, corners_2d, color, thickness=thickness)
    _add_class_color_legend(pred_canvas)

    # ---- 下半部: LiDAR 透视视图 (仅 Pred, 按类别着色) ----
    persp_canvas = _render_perspective_by_class(
        velo_path, calib, pred_objects, thickness)
    if persp_canvas is None:
        return pred_canvas  # 无点云时仅返回上半部

    # ---- 统一宽度后上下拼接 (等比缩放, 不留黑边) ----
    tw = max(pred_canvas.shape[1], persp_canvas.shape[1])
    panels = []
    for p in [pred_canvas, persp_canvas]:
        if p.shape[1] != tw:
            new_h = int(p.shape[0] * tw / p.shape[1])
            p = cv2.resize(p, (tw, new_h), interpolation=cv2.INTER_LINEAR)
        panels.append(p)
    sep = np.full((4, tw, 3), 80, dtype=np.uint8)
    return np.vstack([panels[0], sep, panels[1]])


def _render_perspective_by_class(velo_path, calib, pred_objects,
                                 thickness=2):
    """渲染 LiDAR 透视 3D 视图, 仅绘制 Pred 框并按类别着色。

    Args:
        velo_path: velodyne .bin 文件路径
        calib: Calibration 实例
        pred_objects: Pred Object3d 列表 (已过滤)
        thickness: Pred 线宽
    Returns:
        BGR uint8 image, 或 None
    """
    if not os.path.exists(velo_path):
        return None

    pts_velo = load_velodyne(velo_path)
    pts_rect = calib.lidar_to_rect(pts_velo[:, :3])

    # 过滤范围
    mask = ((pts_rect[:, 0] >= _BEV_X_RANGE[0]) &
            (pts_rect[:, 0] <= _BEV_X_RANGE[1]) &
            (pts_rect[:, 1] >= _BEV_Y_RANGE[0]) &
            (pts_rect[:, 1] <= _BEV_Y_RANGE[1]) &
            (pts_rect[:, 2] >= _BEV_Z_RANGE[0]) &
            (pts_rect[:, 2] <= _BEV_Z_RANGE[1]))
    pts = pts_rect[mask]

    W, H = _PERSP_IMG_W, _PERSP_IMG_H
    cx, cy = W / 2.0, H / 2.0
    focal = _PERSP_FOCAL
    R, t = _build_view_matrix(_CAM_EYE, _CAM_LOOKAT, _CAM_UP)

    # 投影并渲染点云
    uv, depth, valid = _project_perspective(pts, R, t, focal, cx, cy)
    uv, depth, pts_y = uv[valid], depth[valid], pts[valid, 1]
    canvas = np.full((H, W, 3), [235, 235, 230], dtype=np.uint8)

    # 先绘制距离参考线 (在点云层之下)
    _draw_persp_distance_lines(canvas, R, t, focal, cx, cy, W, H)

    order = np.argsort(-depth)
    uv, pts_y = uv[order], pts_y[order]
    y_norm = np.clip((pts_y - _BEV_Y_RANGE[0]) /
                     (_BEV_Y_RANGE[1] - _BEV_Y_RANGE[0]), 0.0, 1.0)
    y_u8 = (y_norm * 255).astype(np.uint8)
    lut_gray = np.arange(256, dtype=np.uint8).reshape(1, 256)
    lut_bgr = cv2.applyColorMap(lut_gray, cv2.COLORMAP_MAGMA).reshape(256, 3)
    for k in range(len(uv)):
        px, py = int(uv[k, 0]), int(uv[k, 1])
        if 0 <= px < W and 0 <= py < H:
            c = tuple(int(v) for v in lut_bgr[y_u8[k]])
            cv2.circle(canvas, (px, py), _PERSP_POINT_RADIUS, c, -1, cv2.LINE_AA)

    # Pred 框: 类别颜色
    for obj in sort_by_depth(pred_objects):
        corners3d = obj.generate_corners3d()
        cls_color = CLASS_COLORS.get(obj.cls_type, (200, 200, 200))
        _draw_persp_3d_box(canvas, corners3d, R, t, focal, cx, cy,
                           cls_color, thickness)

    # 自车框
    ego_corners = _ego_corners_3d()
    _draw_persp_3d_box(canvas, ego_corners, R, t, focal, cx, cy,
                       (0, 0, 220), thickness + 1)
    ego_center = ego_corners.mean(axis=0, keepdims=True)
    uv_ego, _, v_ego = _project_perspective(ego_center, R, t, focal, cx, cy)
    if v_ego.any():
        cv2.putText(canvas, 'EGO',
                    (int(uv_ego[0, 0]) - 15, int(uv_ego[0, 1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2, cv2.LINE_AA)

    canvas = _crop_persp_top_blank(canvas)
    _add_class_color_legend(canvas)
    return canvas


# ======================== 主流程 ========================
def parse_args():
    parser = argparse.ArgumentParser(description='MonoDLE KITTI Visualization')
    parser.add_argument('--config', type=str, default=None,
                        help='YAML 配置文件路径 (自动读取 data_dir / pred_dir / score_thresh / valid_classes)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='KITTI 数据集根目录 (覆盖 config 中 dataset.root_dir)')
    parser.add_argument('--pred_dir', type=str, default=None,
                        help='预测结果目录 (KITTI 格式 txt, 覆盖从 config 自动推导的路径)')
    parser.add_argument('--split', type=str, default='val',
                        help='数据集切分 (train/val/test)')
    parser.add_argument('--num_images', type=int, default=10,
                        help='可视化图片数量')
    parser.add_argument('--output_dir', type=str, default='vis_output',
                        help='输出目录')
    parser.add_argument('--seed', type=int, default=43,
                        help='随机种子 (用于采样图片)')
    parser.add_argument('--score_thresh', type=float, default=None,
                        help='预测结果分数阈值 (覆盖 config 中 tester.threshold)')
    parser.add_argument('--no_random', action='store_true',
                        help='不随机采样，使用前N张图')
    parser.add_argument('--thickness', type=int, default=2,
                        help='框线条粗细 (像素)')
    parser.add_argument('--layout', type=str, default='split', choices=['split', 'overlay'],
                        help='split=上下拼图(GT/Pred分开，推荐); overlay=叠加在同一张图')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型权重路径，用于生成热力图/深度/不确定性视图 (可覆盖 config 中 tester.checkpoint)')
    return parser.parse_args()


def load_yaml_config(config_path):
    """加载 YAML 配置文件，返回 dict。"""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.load(f, Loader=yaml.Loader)


def resolve_path(path, base_dir):
    """将相对路径解析为相对于 base_dir 的绝对路径。"""
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def find_latest_run_dir(pattern):
    """
    给定含 <timestamp> 占位符的路径 pattern，返回最新匹配的实际路径。
    例: experiments/results/monodle/kitti_da3/<timestamp>/checkpoints/xxx.pth
    """
    # 将 <timestamp> 替换为 glob 通配符
    glob_pattern = pattern.replace('<timestamp>', '*')
    matches = glob.glob(glob_pattern)
    if not matches:
        return None
    # 按修改时间降序，取最新
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def derive_pred_dir_from_checkpoint(checkpoint_path, base_dir):
    """
    从 checkpoint 路径推导预测结果目录。

    例:
      checkpoint: experiments/results/monodle/kitti_da3/20260219/checkpoints/ckpt.pth
      pred_dir  : experiments/results/monodle/kitti_da3/20260219/outputs/data
    """
    # 若含 <timestamp>，先解析为最新路径
    if '<timestamp>' in checkpoint_path:
        resolved = find_latest_run_dir(resolve_path(checkpoint_path, base_dir))
        if resolved is None:
            return None
        checkpoint_path = resolved
    else:
        checkpoint_path = resolve_path(checkpoint_path, base_dir)

    if not os.path.exists(checkpoint_path):
        return None

    # checkpoints/ → run_dir/ → outputs/data/
    run_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    return os.path.join(run_dir, 'outputs', 'data')


def apply_yaml_config(args):
    """
    从 YAML 配置中读取默认值并填充到 args，CLI 显式传入的值优先级更高。

    更新的字段:
      args.data_dir      ← dataset.root_dir
      args.pred_dir      ← 从 tester.checkpoint 推导
      args.score_thresh  ← tester.threshold
      VALID_CLASSES      ← dataset.writelist
    """
    if args.config is None:
        return

    config_path = resolve_path(args.config, PROJ_ROOT)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_yaml_config(config_path)
    print(f"[INFO] 已加载配置: {config_path}")

    # --- data_dir ---
    if args.data_dir is None:
        root_dir = cfg.get('dataset', {}).get('root_dir', None)
        if root_dir is not None:
            args.data_dir = resolve_path(root_dir, PROJ_ROOT)
            print(f"[INFO] data_dir (from config): {args.data_dir}")

    # --- score_thresh ---
    if args.score_thresh is None:
        args.score_thresh = cfg.get('tester', {}).get('threshold', 0.2)
        print(f"[INFO] score_thresh (from config): {args.score_thresh}")

    # --- valid_classes ---
    writelist = cfg.get('dataset', {}).get('writelist', None)
    if writelist:
        global VALID_CLASSES
        VALID_CLASSES = set(writelist)
        print(f"[INFO] VALID_CLASSES (from config): {VALID_CLASSES}")

    # --- pred_dir & checkpoint ---
    cfg_checkpoint = cfg.get('tester', {}).get('checkpoint', None)
    if args.pred_dir is None:
        if cfg_checkpoint is not None:
            pred_dir = derive_pred_dir_from_checkpoint(cfg_checkpoint, PROJ_ROOT)
            if pred_dir is not None:
                args.pred_dir = pred_dir
                print(f"[INFO] pred_dir (from config checkpoint): {args.pred_dir}")
            else:
                print(f"[WARNING] 无法从 checkpoint 路径推导 pred_dir: {cfg_checkpoint}")

    if args.checkpoint is None and cfg_checkpoint is not None:
        if '<timestamp>' in cfg_checkpoint:
            resolved = find_latest_run_dir(resolve_path(cfg_checkpoint, PROJ_ROOT))
        else:
            resolved = resolve_path(cfg_checkpoint, PROJ_ROOT)
        if resolved is not None and os.path.exists(resolved):
            args.checkpoint = resolved
            print(f"[INFO] checkpoint (from config): {args.checkpoint}")
        else:
            print(f"[WARNING] checkpoint 文件不存在，跳过模型推理视图: {cfg_checkpoint}")

    return cfg


def load_prediction(pred_path, score_thresh=0.2):
    """加载预测文件并按 score 过滤"""
    if not os.path.exists(pred_path):
        return []
    objects = get_objects_from_label(pred_path)
    # 过滤低分预测
    filtered = [obj for obj in objects if obj.score >= score_thresh]
    return filtered


def select_images(idx_list, label_dir, num_images, seed, no_random):
    """
    从 idx_list 中选择图片进行可视化。
    只要求 GT 标签文件存在即可入选，不依赖预测结果，
    确保相同种子下不同配置/模型选出完全相同的图片。
    """
    candidates = []
    for idx in idx_list:
        gt_path = os.path.join(label_dir, f'{idx}.txt')
        if not os.path.exists(gt_path):
            continue
        candidates.append(idx)

    if not candidates:
        print("[WARNING] 没有找到有效的标签文件!")
        return idx_list[:num_images]

    if no_random:
        selected = candidates[:num_images]
    else:
        random.seed(seed)
        random.shuffle(candidates)
        selected = candidates[:num_images]

    return selected


def main():
    args = parse_args()

    # 从 YAML 配置读取默认值（CLI 显式传入的值不受影响）
    cfg = apply_yaml_config(args) or {}

    # 判断模型是否经过密集深度蒸馏 (distill.lambda > 0)
    distill_lambda = cfg.get('distill', {}).get('lambda', 0.0)
    has_dense_supervision = distill_lambda > 0.0
    if not has_dense_supervision:
        print("[INFO] distill.lambda=0 → 模型未经密集深度蒸馏，密集深度图仅供参考")

    # 最终回退默认值
    if args.data_dir is None:
        args.data_dir = os.path.join(PROJ_ROOT, 'data', 'KITTI')
        print(f"[INFO] data_dir (default): {args.data_dir}")
    if args.score_thresh is None:
        args.score_thresh = 0.2
    if args.pred_dir is None:
        args.pred_dir = os.path.join(PROJ_ROOT, 'experiments', 'results', 'outputs', 'data')

    # 规范化为绝对路径
    args.data_dir = resolve_path(args.data_dir, PROJ_ROOT)
    args.pred_dir = resolve_path(args.pred_dir, PROJ_ROOT)

    # 路径配置
    data_dir = args.data_dir
    split_file = os.path.join(data_dir, 'ImageSets', f'{args.split}.txt')
    data_subdir = 'testing' if args.split == 'test' else 'training'
    image_dir = os.path.join(data_dir, data_subdir, 'image_2')
    label_dir = os.path.join(data_dir, data_subdir, 'label_2')
    calib_dir = os.path.join(data_dir, data_subdir, 'calib')
    velo_dir = os.path.join(data_dir, data_subdir, 'velodyne')
    pred_dir = args.pred_dir

    # 输出目录
    out_3d = os.path.join(args.output_dir, '3d_bbox')
    out_2d = os.path.join(args.output_dir, '2d_bbox')
    out_hm  = os.path.join(args.output_dir, 'heatmap')
    out_da3 = os.path.join(args.output_dir, 'da3_depth')
    out_unc = os.path.join(args.output_dir, 'uncertainty')
    out_bev = os.path.join(args.output_dir, 'lidar_bev')
    out_combined = os.path.join(args.output_dir, 'combined_3d')
    os.makedirs(out_3d, exist_ok=True)
    os.makedirs(out_2d, exist_ok=True)
    if os.path.isdir(velo_dir):
        os.makedirs(out_bev, exist_ok=True)
        os.makedirs(out_combined, exist_ok=True)
        print(f"[INFO] 发现 velodyne 目录, 将生成 LiDAR BEV 视图 + 组合3D视图")
    else:
        print(f"[INFO] 无 velodyne 目录 ({velo_dir}), 跳过 LiDAR BEV / 组合3D视图")

    # DA3 深度目录
    da3_dir = os.path.join(data_dir, 'DA3_depth_results')

    # 加载模型（如果有 checkpoint）
    import torch
    vis_model = None
    vis_device = None
    has_uncertainty = False
    checkpoint_path = getattr(args, 'checkpoint', None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[INFO] 加载模型权重: {checkpoint_path}")
        vis_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        try:
            vis_model = load_model_from_config(cfg, checkpoint_path, vis_device)
            has_uncertainty = cfg.get('model', {}).get('use_distill_uncertainty', False)
            os.makedirs(out_hm, exist_ok=True)
            os.makedirs(out_da3, exist_ok=True)
            if has_uncertainty:
                os.makedirs(out_unc, exist_ok=True)
            print(f"[INFO] 模型加载成功  (uncertainty head: {has_uncertainty})")
        except Exception as e:
            print(f"[WARNING] 模型加载失败，跳过推理视图: {e}")
            vis_model = None
    elif checkpoint_path:
        print(f"[WARNING] checkpoint 不存在，跳过推理视图: {checkpoint_path}")

    if os.path.isdir(da3_dir):
        os.makedirs(out_da3, exist_ok=True)

    # 检查输入路径
    assert os.path.exists(split_file), f"Split file not found: {split_file}"
    assert os.path.exists(image_dir), f"Image dir not found: {image_dir}"
    assert os.path.exists(label_dir), f"Label dir not found: {label_dir}"
    assert os.path.exists(calib_dir), f"Calib dir not found: {calib_dir}"

    if not os.path.exists(pred_dir):
        print(f"[WARNING] 预测目录不存在: {pred_dir}")
        print("将只绘制真值框。")

    # 加载分割文件
    with open(split_file, 'r') as f:
        idx_list = [x.strip() for x in f.readlines()]
    print(f"[INFO] {args.split} 集共 {len(idx_list)} 张图片")

    # 选择要可视化的图片
    selected = select_images(
        idx_list, label_dir, args.num_images, args.seed, args.no_random
    )
    print(f"[INFO] 选择 {len(selected)} 张图片进行可视化")
    print(f"[INFO] 图片ID: {selected}")

    # 逐张可视化
    for i, idx in enumerate(selected):
        img_path = os.path.join(image_dir, f'{idx}.png')
        gt_path = os.path.join(label_dir, f'{idx}.txt')
        calib_path = os.path.join(calib_dir, f'{idx}.txt')
        pred_path = os.path.join(pred_dir, f'{idx}.txt')

        if not os.path.exists(img_path):
            print(f"[SKIP] 图片不存在: {img_path}")
            continue

        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"[SKIP] 无法读取图片: {img_path}")
            continue

        # 读取标定
        calib = Calibration(calib_path)

        # 读取真值
        gt_objects = get_objects_from_label(gt_path) if os.path.exists(gt_path) else []

        # 读取预测
        pred_objects = load_prediction(pred_path, args.score_thresh)

        # 统计
        gt_valid = [o for o in gt_objects if o.cls_type in VALID_CLASSES]
        print(f"  [{i+1}/{len(selected)}] {idx}: GT={len(gt_valid)} objs, Pred={len(pred_objects)} objs")

        # --- 3D 可视化 ---
        vis_3d = visualize_3d(img, gt_objects, pred_objects, calib,
                              thickness=args.thickness, layout=args.layout)
        out_path_3d = os.path.join(out_3d, f'{idx}_3d.png')
        cv2.imwrite(out_path_3d, vis_3d)

        # --- 2D 可视化 ---
        vis_2d = visualize_2d(img, gt_objects, pred_objects,
                              thickness=args.thickness, layout=args.layout)
        out_path_2d = os.path.join(out_2d, f'{idx}_2d.png')
        cv2.imwrite(out_path_2d, vis_2d)

        # --- LiDAR BEV + 透视可视化 (合并为一张图) ---
        if os.path.isdir(velo_dir):
            velo_path = os.path.join(velo_dir, f'{idx}.bin')
            img_size = (img.shape[1], img.shape[0])
            bev_img = visualize_lidar_bev(velo_path, calib, gt_objects, pred_objects,
                                          thickness=args.thickness, img_size=img_size)
            persp_img = visualize_lidar_perspective(
                velo_path, calib, gt_objects, pred_objects,
                thickness=args.thickness)
            lidar_panels = [p for p in [bev_img, persp_img] if p is not None]
            if lidar_panels:
                # 宽度对齐后上下拼接
                tw = max(p.shape[1] for p in lidar_panels)
                aligned = []
                for p in lidar_panels:
                    if p.shape[1] < tw:
                        pad = np.zeros((p.shape[0], tw - p.shape[1], 3), dtype=np.uint8)
                        p = np.hstack([p, pad])
                    elif p.shape[1] > tw:
                        p = cv2.resize(p, (tw, int(p.shape[0] * tw / p.shape[1])),
                                       interpolation=cv2.INTER_LINEAR)
                    aligned.append(p)
                sep = np.full((6, tw, 3), 80, dtype=np.uint8)
                combined = aligned[0]
                for a in aligned[1:]:
                    combined = np.vstack([combined, sep, a])
                cv2.imwrite(os.path.join(out_bev, f'{idx}_lidar.png'), combined)

            # --- 组合 3D 视图 (Pred 相机 + LiDAR 透视, 按类别着色) ---
            velo_path_combined = os.path.join(velo_dir, f'{idx}.bin')
            combined_img = visualize_combined_pred_perspective(
                img, pred_objects, calib, velo_path_combined,
                gt_objects=gt_objects, thickness=args.thickness)
            if combined_img is not None:
                cv2.imwrite(os.path.join(out_combined, f'{idx}_combined.png'),
                            combined_img)

        # --- 模型推理视图 ---
        da3_pred_img = None
        if vis_model is not None:
            try:
                outputs, _ = run_model_inference(vis_model, img, vis_device)

                hm_img = visualize_heatmap(img, outputs)
                cv2.imwrite(os.path.join(out_hm, f'{idx}_hm.png'), hm_img)

                hm_per = visualize_heatmap_perclass(img, outputs)
                if hm_per is not None:
                    cv2.imwrite(os.path.join(out_hm, f'{idx}_hm_perclass.png'), hm_per)

                da3_pred_img = visualize_da3_pred(img, outputs,
                                                    has_dense_supervision=has_dense_supervision)

                if has_uncertainty:
                    unc_img = visualize_uncertainty(img, outputs)
                    if unc_img is not None:
                        cv2.imwrite(os.path.join(out_unc, f'{idx}_unc.png'), unc_img)
            except Exception as e:
                print(f"[WARNING] {idx} 推理失败: {e}")

        # --- DA3 GT + Pred 合并深度图（上下排列）---
        da3_gt_img = None
        if os.path.isdir(da3_dir):
            da3_gt_img = visualize_da3_gt(img, da3_dir, idx)

        if da3_gt_img is not None or da3_pred_img is not None:
            panels = [p for p in [da3_gt_img, da3_pred_img] if p is not None]
            if len(panels) == 2:
                # 宽度对齐后拼接
                tw = max(p.shape[1] for p in panels)
                aligned = []
                for p in panels:
                    if p.shape[1] < tw:
                        pad = np.zeros((p.shape[0], tw - p.shape[1], 3), dtype=np.uint8)
                        p = np.hstack([p, pad])
                    aligned.append(p)
                sep = np.full((6, tw, 3), 80, dtype=np.uint8)
                combined = np.vstack([aligned[0], sep, aligned[1]])
            else:
                combined = panels[0]
            cv2.imwrite(os.path.join(out_da3, f'{idx}_da3.png'), combined)

    print(f"\n[DONE] 可视化结果已保存到 {args.output_dir}/")
    print(f"  3D 框: {out_3d}/")
    print(f"  2D 框: {out_2d}/")
    if vis_model is not None:
        print(f"  热力图: {out_hm}/")
        if has_uncertainty:
            print(f"  不确定性: {out_unc}/")
    if os.path.isdir(velo_dir):
        print(f"  LiDAR BEV: {out_bev}/")
        print(f"  组合3D (Pred+LiDAR, 按类别着色): {out_combined}/")
    if os.path.isdir(da3_dir) or vis_model is not None:
        print(f"  DA3 深度 (GT/Pred 合并): {out_da3}/")


if __name__ == '__main__':
    main()
