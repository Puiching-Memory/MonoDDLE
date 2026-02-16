"""
KITTI 数据集工具类
=================

提供 KITTI 3D 目标检测所需的核心数据结构和坐标变换工具：

- :class:`Object3d` — 单个标注对象
- :class:`Calibration` — 相机标定参数及坐标系互转
- 仿射变换辅助函数

所有坐标系遵循 KITTI 相机坐标系约定：
- X 轴指向右方
- Y 轴指向下方
- Z 轴指向前方
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

__all__ = [
    'Object3d', 'Calibration',
    'get_objects_from_label', 'get_calib_from_file',
]


# ═══════════════════════ Object3d ═══════════════════════

class Object3d:
    """KITTI 格式的单个 3D 目标标注。

    KITTI 标注格式 (每行 15 或 16 个字段)::

        type truncation occlusion alpha
        bbox_left bbox_top bbox_right bbox_bottom
        height width length
        x y z rotation_y [score]

    Attributes:
        cls_type: 类别名 (Car, Pedestrian, Cyclist, ...)。
        trucation: 截断程度 [0, 1]。
        occlusion: 遮挡级别 (0=可见, 1=部分, 2=大部分, 3=未知)。
        alpha: 观测角 [-π, π]。
        box2d: 2D 边界框 [x1, y1, x2, y2]。
        h, w, l: 3D 尺寸 (高度, 宽度, 长度)。
        pos: 3D 位置 [x, y, z] (相机坐标系)。
        ry: 绕 Y 轴旋转角。
        score: 检测置信度 (GT 默认 -1)。
        level_str: 难度级别名称。
        level: 难度级别编号。
    """

    __slots__ = (
        'src', 'cls_type', 'trucation', 'occlusion', 'alpha',
        'box2d', 'h', 'w', 'l', 'pos', 'dis_to_cam', 'ry', 'score',
        'level_str', 'level',
    )

    def __init__(self, line: str) -> None:
        label = line.strip().split(' ')
        self.src: str = line
        self.cls_type: str = label[0]
        self.trucation: float = float(label[1])
        self.occlusion: float = float(label[2])
        self.alpha: float = float(label[3])
        self.box2d: np.ndarray = np.array(
            (float(label[4]), float(label[5]), float(label[6]), float(label[7])),
            dtype=np.float32,
        )
        self.h: float = float(label[8])
        self.w: float = float(label[9])
        self.l: float = float(label[10])
        self.pos: np.ndarray = np.array(
            (float(label[11]), float(label[12]), float(label[13])),
            dtype=np.float32,
        )
        self.dis_to_cam: float = float(np.linalg.norm(self.pos))
        self.ry: float = float(label[14])
        self.score: float = float(label[15]) if len(label) == 16 else -1.0
        self.level_str: Optional[str] = None
        self.level: int = self._compute_difficulty()

    def _compute_difficulty(self) -> int:
        """计算难度等级 (Easy=1, Moderate=2, Hard=3, Unknown=4, DontCare=0)。"""
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if self.trucation == -1:
            self.level_str = 'DontCare'
            return 0
        if height >= 40 and self.trucation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 1
        elif height >= 25 and self.trucation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 2
        elif height >= 25 and self.trucation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 3
        else:
            self.level_str = 'UnKnown'
            return 4

    # ── 3D 操作 ──

    def generate_corners3d(self) -> np.ndarray:
        """生成 3D 边界框的 8 个顶点 (相机坐标系)。

        Returns:
            corners_3d: 形状 (8, 3) 的顶点坐标::

                1 -------- 0
               /|         /|
              2 -------- 3 .
              | |        | |
              . 5 -------- 4
              |/         |/
              6 -------- 7
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([
            [np.cos(self.ry), 0, np.sin(self.ry)],
            [0, 1, 0],
            [-np.sin(self.ry), 0, np.cos(self.ry)],
        ])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.pos
        return corners3d

    def to_bev_box2d(self, oblique: bool = True, voxel_size: float = 0.1) -> np.ndarray:
        """转换为 BEV 2D 框。"""
        if oblique:
            corners3d = self.generate_corners3d()
            xz_corners = corners3d[0:4, [0, 2]]
            box2d = np.zeros((4, 2), dtype=np.int32)
            box2d[:, 0] = ((xz_corners[:, 0] - Object3d.MIN_XZ[0]) / voxel_size).astype(np.int32)
            box2d[:, 1] = (Object3d.BEV_SHAPE[0] - 1
                           - ((xz_corners[:, 1] - Object3d.MIN_XZ[1]) / voxel_size)).astype(np.int32)
            box2d[:, 0] = np.clip(box2d[:, 0], 0, Object3d.BEV_SHAPE[1])
            box2d[:, 1] = np.clip(box2d[:, 1], 0, Object3d.BEV_SHAPE[0])
        else:
            box2d = np.zeros(4, dtype=np.int32)
            cu = np.floor((self.pos[0] - Object3d.MIN_XZ[0]) / voxel_size).astype(np.int32)
            cv = (Object3d.BEV_SHAPE[0] - 1
                  - ((self.pos[2] - Object3d.MIN_XZ[1]) / voxel_size)).astype(np.int32)
            half_l, half_w = int(self.l / voxel_size / 2), int(self.w / voxel_size / 2)
            box2d[0], box2d[1] = cu - half_l, cv - half_w
            box2d[2], box2d[3] = cu + half_l, cv + half_w
        return box2d

    # ── 序列化 ──

    def to_str(self) -> str:
        return (
            f'{self.cls_type} {self.trucation:.3f} {self.occlusion:.3f} '
            f'{self.alpha:.3f} box2d: {self.box2d} '
            f'hwl: [{self.h:.3f} {self.w:.3f} {self.l:.3f}] '
            f'pos: {self.pos} ry: {self.ry:.3f}'
        )

    def to_kitti_format(self) -> str:
        """输出标准 KITTI 格式字符串。"""
        return (
            f'{self.cls_type} {self.trucation:.2f} {int(self.occlusion)} '
            f'{self.alpha:.2f} '
            f'{self.box2d[0]:.2f} {self.box2d[1]:.2f} '
            f'{self.box2d[2]:.2f} {self.box2d[3]:.2f} '
            f'{self.h:.2f} {self.w:.2f} {self.l:.2f} '
            f'{self.pos[0]:.2f} {self.pos[1]:.2f} {self.pos[2]:.2f} '
            f'{self.ry:.2f}'
        )

    def __repr__(self) -> str:
        return f'Object3d({self.cls_type}, level={self.level_str})'


def get_objects_from_label(label_file: str) -> List[Object3d]:
    """从标注文件解析所有 3D 目标。

    Args:
        label_file: 标注文件路径。

    Returns:
        Object3d 对象列表。
    """
    with open(label_file, 'r') as f:
        lines = f.readlines()
    return [Object3d(line) for line in lines if line.strip()]


# ═══════════════════════ Calibration ═══════════════════════

def get_calib_from_file(calib_file: str) -> Dict[str, np.ndarray]:
    """从标定文件读取标定矩阵。

    Args:
        calib_file: KITTI 标定文件路径。

    Returns:
        包含 P2, P3, R0, Tr_velo2cam 的字典。
    """
    with open(calib_file) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {
        'P2': P2.reshape(3, 4),
        'P3': P3.reshape(3, 4),
        'R0': R0.reshape(3, 3),
        'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4),
    }


class Calibration:
    """KITTI 相机标定参数与坐标变换。

    支持以下坐标系转换:
    - LiDAR ↔ 相机矩形坐标系 (Rect)
    - 相机矩形坐标系 ↔ 图像像素坐标系
    - 深度图 → 3D 点云

    Args:
        calib_file: 标定文件路径或已解析的标定字典。
    """

    def __init__(self, calib_file: str | Dict[str, np.ndarray]) -> None:
        if isinstance(calib_file, str):
            calib = get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.P2: np.ndarray = calib['P2']            # (3, 4)
        self.R0: np.ndarray = calib['R0']            # (3, 3)
        self.V2C: np.ndarray = calib['Tr_velo2cam']  # (3, 4)
        self.C2V: np.ndarray = self._inverse_rigid_trans(self.V2C)

        # 相机内参
        self.cu: float = float(self.P2[0, 2])
        self.cv: float = float(self.P2[1, 2])
        self.fu: float = float(self.P2[0, 0])
        self.fv: float = float(self.P2[1, 1])
        self.tx: float = float(self.P2[0, 3] / (-self.fu))
        self.ty: float = float(self.P2[1, 3] / (-self.fv))

    # ── 坐标变换 ──

    @staticmethod
    def cart_to_hom(pts: np.ndarray) -> np.ndarray:
        """笛卡尔坐标 → 齐次坐标。

        Args:
            pts: (N, D)

        Returns:
            (N, D+1)
        """
        return np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))

    def lidar_to_rect(self, pts_lidar: np.ndarray) -> np.ndarray:
        """LiDAR 坐标 → 相机矩形坐标。

        Args:
            pts_lidar: (N, 3) LiDAR 点。

        Returns:
            (N, 3) 矩形坐标系点。
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        return np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))

    def rect_to_lidar(self, pts_rect: np.ndarray) -> np.ndarray:
        """相机矩形坐标 → LiDAR 坐标。"""
        pts_ref = np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_rect)))
        pts_ref = self.cart_to_hom(pts_ref)
        return np.dot(pts_ref, np.transpose(self.C2V))

    def rect_to_img(self, pts_rect: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """相机矩形坐标 → 图像像素坐标。

        Args:
            pts_rect: (N, 3)

        Returns:
            pts_img: (N, 2) 像素坐标。
            pts_rect_depth: (N,) 深度。
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """LiDAR 坐标 → 图像像素坐标。"""
        pts_rect = self.lidar_to_rect(pts_lidar)
        return self.rect_to_img(pts_rect)

    def img_to_rect(self, u: np.ndarray, v: np.ndarray, depth_rect: np.ndarray) -> np.ndarray:
        """图像像素坐标 + 深度 → 相机矩形坐标。

        Args:
            u, v: (N,) 像素坐标。
            depth_rect: (N,) 深度。

        Returns:
            (N, 3) 矩形坐标系点。
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        return np.concatenate(
            (x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)),
            axis=1,
        )

    def depthmap_to_rect(self, depth_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """深度图 → 3D 点云。

        Args:
            depth_map: (H, W) 深度图。

        Returns:
            pts_rect: (N, 3) 矩形坐标系点。
            x_idxs, y_idxs: 像素索引。
        """
        x_range = np.arange(0, depth_map.shape[1])
        y_range = np.arange(0, depth_map.shape[0])
        x_idxs, y_idxs = np.meshgrid(x_range, y_range)
        x_idxs, y_idxs = x_idxs.reshape(-1), y_idxs.reshape(-1)
        depth = depth_map[y_idxs, x_idxs]
        pts_rect = self.img_to_rect(x_idxs, y_idxs, depth)
        return pts_rect, x_idxs, y_idxs

    def corners3d_to_img_boxes(
        self, corners3d: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """3D 框顶点 → 2D 图像框。

        Args:
            corners3d: (N, 8, 3) 矩形坐标系顶点。

        Returns:
            boxes: (N, 4) [x1, y1, x2, y2]。
            boxes_corner: (N, 8, 2) 各顶点像素坐标。
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate(
            (corners3d, np.ones((sample_num, 8, 1))), axis=2
        )
        img_pts = np.matmul(corners3d_hom, self.P2.T)
        x = img_pts[:, :, 0] / img_pts[:, :, 2]
        y = img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate(
            (x1.reshape(-1, 1), y1.reshape(-1, 1),
             x2.reshape(-1, 1), y2.reshape(-1, 1)),
            axis=1,
        )
        boxes_corner = np.concatenate(
            (x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2
        )
        return boxes, boxes_corner

    def camera_dis_to_rect(
        self, u: np.ndarray, v: np.ndarray, d: np.ndarray,
    ) -> np.ndarray:
        """相机距离 → 相机矩形坐标。"""
        assert self.fu == self.fv, f'{self.fu:.8f} != {self.fv:.8f}'
        fd = np.sqrt((u - self.cu) ** 2 + (v - self.cv) ** 2 + self.fu ** 2)
        x = ((u - self.cu) * d) / fd + self.tx
        y = ((v - self.cv) * d) / fd + self.ty
        z = np.sqrt(d ** 2 - x ** 2 - y ** 2)
        return np.concatenate(
            (x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=1
        )

    # ── 角度转换 ──

    def alpha2ry(self, alpha: float, u: float) -> float:
        """观测角 → 旋转角。"""
        ry = alpha + np.arctan2(u - self.cu, self.fu)
        if ry > np.pi:
            ry -= 2 * np.pi
        if ry < -np.pi:
            ry += 2 * np.pi
        return ry

    def ry2alpha(self, ry: float, u: float) -> float:
        """旋转角 → 观测角。"""
        alpha = ry - np.arctan2(u - self.cu, self.fu)
        if alpha > np.pi:
            alpha -= 2 * np.pi
        if alpha < -np.pi:
            alpha += 2 * np.pi
        return alpha

    # ── 内部工具 ──

    @staticmethod
    def _inverse_rigid_trans(Tr: np.ndarray) -> np.ndarray:
        """刚体变换逆矩阵 (3×4 → 3×4)。"""
        inv_Tr = np.zeros_like(Tr)
        inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
        inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
        return inv_Tr

    def __repr__(self) -> str:
        return f'Calibration(fu={self.fu:.1f}, fv={self.fv:.1f}, cu={self.cu:.1f}, cv={self.cv:.1f})'


# ═══════════════════ 仿射变换工具 (from official MonoDLE) ═══════════════════

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    import cv2
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        trans_inv = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        return trans, trans_inv
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]



