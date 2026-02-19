"""
可视化脚本：绘制 KITTI 2D/3D 框的真值与预测值对比图。
用法:
    cd monodle_modern
    python tools/visualize.py \
        --data_dir ../../data/KITTI \
        --pred_dir experiments/example/outputs/data \
        --split val \
        --num_images 10 \
        --output_dir vis_output
"""
import os
import sys
import argparse
import random
import numpy as np
import cv2

# ---------- 加入项目根目录到路径 ----------
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from lib.datasets.kitti.kitti_utils import get_objects_from_label, Calibration

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

VALID_CLASSES = {
                'Car', 
                #  'Pedestrian', 
                #  'Cyclist', 
                #  'Van', 
                #  'Truck'
                 }


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


# ======================== 主流程 ========================
def parse_args():
    parser = argparse.ArgumentParser(description='MonoDLE KITTI Visualization')
    parser.add_argument('--data_dir', type=str, default='../../data/KITTI',
                        help='KITTI 数据集根目录')
    parser.add_argument('--pred_dir', type=str, default='experiments/example/outputs/data',
                        help='预测结果目录 (KITTI 格式 txt)')
    parser.add_argument('--split', type=str, default='val',
                        help='数据集切分 (train/val/test)')
    parser.add_argument('--num_images', type=int, default=10,
                        help='可视化图片数量')
    parser.add_argument('--output_dir', type=str, default='vis_output',
                        help='输出目录')
    parser.add_argument('--seed', type=int, default=43,
                        help='随机种子 (用于采样图片)')
    parser.add_argument('--score_thresh', type=float, default=0.2,
                        help='预测结果分数阈值')
    parser.add_argument('--no_random', action='store_true',
                        help='不随机采样，使用前N张图')
    parser.add_argument('--thickness', type=int, default=2,
                        help='框线条粗细 (像素)')
    parser.add_argument('--layout', type=str, default='split', choices=['split', 'overlay'],
                        help='split=上下拼图(GT/Pred分开，推荐); overlay=叠加在同一张图')
    return parser.parse_args()


def load_prediction(pred_path, score_thresh=0.2):
    """加载预测文件并按 score 过滤"""
    if not os.path.exists(pred_path):
        return []
    objects = get_objects_from_label(pred_path)
    # 过滤低分预测
    filtered = [obj for obj in objects if obj.score >= score_thresh]
    return filtered


def select_images_with_objects(idx_list, label_dir, pred_dir, num_images, seed, no_random):
    """
    选择同时有真值和预测结果的图片。
    优先选择目标数量较多的图片以获得更好的可视化效果。
    """
    candidates = []
    for idx in idx_list:
        gt_path = os.path.join(label_dir, f'{idx}.txt')
        pred_path = os.path.join(pred_dir, f'{idx}.txt')

        if not os.path.exists(gt_path):
            continue

        gt_objects = get_objects_from_label(gt_path)
        valid_gt = [obj for obj in gt_objects if obj.cls_type in VALID_CLASSES]

        pred_objects = []
        if os.path.exists(pred_path):
            pred_objects = get_objects_from_label(pred_path)
            pred_objects = [obj for obj in pred_objects if obj.cls_type in VALID_CLASSES]

        n_objects = len(valid_gt) + len(pred_objects)
        if n_objects > 0:
            candidates.append((idx, n_objects))

    if not candidates:
        print("[WARNING] 没有找到包含目标的图片!")
        return idx_list[:num_images]

    if no_random:
        # 按目标数量排序，选最多的
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [c[0] for c in candidates[:num_images]]
    else:
        # 随机采样，但倾向于目标多的图片
        random.seed(seed)
        # 从有较多目标的图片中随机选
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_pool = candidates[:min(len(candidates), num_images * 5)]
        random.shuffle(top_pool)
        selected = [c[0] for c in top_pool[:num_images]]

    return selected


def main():
    args = parse_args()

    # 路径配置
    data_dir = args.data_dir
    split_file = os.path.join(data_dir, 'ImageSets', f'{args.split}.txt')
    data_subdir = 'testing' if args.split == 'test' else 'training'
    image_dir = os.path.join(data_dir, data_subdir, 'image_2')
    label_dir = os.path.join(data_dir, data_subdir, 'label_2')
    calib_dir = os.path.join(data_dir, data_subdir, 'calib')
    pred_dir = args.pred_dir

    # 输出目录
    out_3d = os.path.join(args.output_dir, '3d_bbox')
    out_2d = os.path.join(args.output_dir, '2d_bbox')
    os.makedirs(out_3d, exist_ok=True)
    os.makedirs(out_2d, exist_ok=True)

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
    selected = select_images_with_objects(
        idx_list, label_dir, pred_dir, args.num_images, args.seed, args.no_random
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

    print(f"\n[DONE] 可视化结果已保存到 {args.output_dir}/")
    print(f"  3D 框: {out_3d}/")
    print(f"  2D 框: {out_2d}/")


if __name__ == '__main__':
    main()
