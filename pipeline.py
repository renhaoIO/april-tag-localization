"""
pipeline.py — 核心定位流水线模块

实现从 AprilTag 检测到位姿输出的完整算法链:

    检测结果 (corners, ids)
         │
    process_markers()     ← 亚像素优化 + PnP 解算 + 区分参考/目标 Tag
         │
    compute_homography()  ← 4 个参考 Tag → 单应矩阵 H (像素↔世界)
         │
    estimate_target_pose() ← 像素→世界 + Yaw 计算 + KalmanFilter2D + 轨迹更新
         │
    输出: TargetPose(x, z, yaw_deg)

plus 卡尔曼滤波器: KalmanFilter2D (6 状态: x, z, yaw, vx, vz, vyaw)
"""

from typing import Optional, Tuple, Dict, Deque
from collections import deque

import cv2
import numpy as np

from config import (
    MARKER_LENGTH,
    FIELD_SIZE_Z,
    SCALE,
    MARGIN,
    TRAIL_LENGTH,
    REF_TAG_CONFIG,
    CAMERA_MATRIX,
    DIST_COEFFS,
)
from datatypes import ReferenceTagInfo, TargetObservation, TargetPose
from utils import normalize_angle, normalize_angle_diff, pixel_to_world


# ===================== 卡尔曼滤波器 =====================
class KalmanFilter2D:
    """
    2D 卡尔曼滤波器 — 含角度跳变处理

    状态向量: [x, z, yaw, vx, vz, vyaw] (Y→Z 语义)

    过程噪声 (Q 矩阵):
        位置维度: 1e-4 (≈ 1cm 预测不确定性) — 保持位置平滑
        角度维度: 1e-1 (≈ 18° 预测不确定性) — 允许快速转向响应

    核心优化:
        - normalize_angle_diff: 处理 ±180° 新息跳变
        - 初始化和每次更新后归一化 yaw

    Attributes:
        dt:                时间步长 [秒]，默认 1/30
        A:                 状态转移矩阵 (6x6, 匀速运动模型)
        H:                 观测矩阵 (3x6, 仅观测 x,z,yaw)
        Q:                 过程噪声协方差 (6x6)
        R:                 观测噪声协方差 (3x3)
        P:                 估计误差协方差 (6x6)
        x:                 当前状态估计 (6x1)
        initialized:       是否已初始化
    """

    def __init__(
        self,
        dt: float = 1 / 30,
        process_noise: float = 1e-4,
        measurement_noise: float = 5e-3,
    ):
        """
        初始化卡尔曼滤波器

        Args:
            dt:               时间步长 [秒]
            process_noise:    位置维度的基础过程噪声
            measurement_noise: 观测噪声 (x, z, yaw 统一)
        """
        self.dt = dt

        # 状态转移矩阵 A — 匀速运动模型
        self.A = np.array(
            [
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # 观测矩阵 H — 只观测位置和角度
        self.H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        # 过程噪声协方差 Q
        self.Q = np.eye(6, dtype=np.float32) * process_noise
        self.Q[2, 2] = 10.0  # yaw 预测不确定性极大, 几乎无平滑, 纯跟踪测量
        self.Q[5, 5] = 10.0  # vyaw 同理

        # 观测噪声协方差 R
        self.R = np.eye(3, dtype=np.float32) * measurement_noise

        # 估计误差协方差 P (初始完全不确定)
        self.P = np.eye(6, dtype=np.float32)

        # 状态估计 x (初始为零)
        self.x = np.zeros((6, 1), dtype=np.float32)

        self.initialized = False

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        执行卡尔曼滤波更新 (预测 + 校正)

        Args:
            measurement: 观测值 [x, z, yaw_deg], shape (3,)

        Returns:
            平滑后的状态估计 [x, z, yaw_deg], shape (3,)

        处理流程:
            1. 首次调用: 用测量值初始化状态
            2. 后续调用: 预测 → 新息归一化 → 卡尔曼增益 → 状态更新 → 角度归一化
        """
        if not self.initialized:
            self.x[0:3, 0] = measurement
            self.x[2, 0] = normalize_angle(self.x[2, 0])
            self.initialized = True
            return measurement

        self.x = self.A @ self.x               # 预测
        self.P = self.A @ self.P @ self.A.T + self.Q

        y = measurement.reshape(-1, 1) - self.H @ self.x
        y[2, 0] = normalize_angle_diff(y[2, 0])  # 新息 + 角度跳变修正

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # 卡尔曼增益

        self.x = self.x + K @ y                   # 状态更新
        self.P = (np.eye(6) - K @ self.H) @ self.P
        self.x[2, 0] = normalize_angle(self.x[2, 0])

        return self.x[0:3, 0].flatten()


# ===================== Marker 处理 =====================
def process_markers(
    gray: np.ndarray,
    corners: Tuple[np.ndarray, ...],
    ids: np.ndarray,
    target_tag_id: int,
) -> Tuple[Dict[int, ReferenceTagInfo], Optional[TargetObservation]]:
    """
    处理所有检测到的 AprilTag Marker

    对每个检测到的 Tag:
        1. 亚像素角点优化 (cornerSubPix) — 提高精度
        2. PnP 位姿解算 (SOLVEPNP_IPPE_SQUARE) — 专为方形 Tag 优化
        3. 区分参考 Tag (场地四角，供 homography 使用) 和目标 Tag (小车)

    对目标 Tag 额外计算:
        - 前端点: Tag 本地 X 轴正方向上的点投影到图像，用于计算 Yaw

    Args:
        gray:          灰度图像 (用于亚像素优化)
        corners:       OpenCV 检测到的角点列表
        ids:           OpenCV 检测到的 Tag ID 数组
        target_tag_id: 要跟踪的目标 Tag ID

    Returns:
        (reference_tags, target_obs):
            reference_tags: {tag_id: ReferenceTagInfo} 检测到的参考 Tag
            target_obs:     TargetObservation 或 None (目标 Tag 未检测到或 PnP 失败)
    """
    # 亚像素优化 — 提高角点检测精度
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    for corner in corners:
        cv2.cornerSubPix(gray, corner[0], (3, 3), (-1, -1), criteria)

    reference_tags: Dict[int, ReferenceTagInfo] = {}
    target_obs: Optional[TargetObservation] = None

    # Tag 本地 3D 角点定义 — 中心为原点 (尺寸: MARKER_LENGTH x MARKER_LENGTH)
    tag_corners_3d = np.float32(
        [
            [-MARKER_LENGTH / 2, MARKER_LENGTH / 2, 0],
            [MARKER_LENGTH / 2, MARKER_LENGTH / 2, 0],
            [MARKER_LENGTH / 2, -MARKER_LENGTH / 2, 0],
            [-MARKER_LENGTH / 2, -MARKER_LENGTH / 2, 0],
        ]
    )

    for i in range(len(ids)):
        tag_id = int(ids[i][0])
        corner = corners[i][0]
        cx = float(np.mean(corner[:, 0]))
        cy = float(np.mean(corner[:, 1]))

        if tag_id in REF_TAG_CONFIG:
            # --- 处理参考 Tag (场地四角) ---
            ret_pnp, rvec, tvec = cv2.solvePnP(
                tag_corners_3d,
                corner.astype(np.float32),
                CAMERA_MATRIX,
                DIST_COEFFS,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if ret_pnp:
                config = REF_TAG_CONFIG[tag_id]
                reference_tags[tag_id] = ReferenceTagInfo(
                    tag_id=tag_id,
                    corner_id=config["corner_id"],
                    center_px=(cx, cy),
                    world_pos=config["world_pos"],
                    rvec=rvec,
                    tvec=tvec,
                )

        elif tag_id == target_tag_id:
            # --- 处理目标 Tag (小车上的 Tag) ---
            ret_pnp, rvec, tvec = cv2.solvePnP(
                tag_corners_3d,
                corner.astype(np.float32),
                CAMERA_MATRIX,
                DIST_COEFFS,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if ret_pnp:
                # 计算前端点 — Tag 本地 X 轴正方向 [MARKER_LENGTH/2, 0, 0]
                front_point_3d = np.array(
                    [[MARKER_LENGTH / 2, 0.0, 0.0]], dtype=np.float32
                )
                front_point_2d, _ = cv2.projectPoints(
                    front_point_3d, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS
                )
                fx, fy = (
                    float(front_point_2d[0][0][0]),
                    float(front_point_2d[0][0][1]),
                )
                target_obs = TargetObservation(
                    center_px=(cx, cy), front_px=(fx, fy), corners=corner
                )

    return reference_tags, target_obs


# ===================== 单应矩阵计算 =====================
def compute_homography(
    reference_tags: Dict[int, ReferenceTagInfo],
    corners: Tuple[np.ndarray, ...],
    ids: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    基于参考 Tag 计算单应性矩阵 (Homography)

    建立 像素坐标 ↔ 世界坐标 (XZ 平面) 的映射关系。

    算法流程:
        1. 对每个参考 Tag，根据其 corner_id 确定旋转角度 (yaw_map)
        2. 将 Tag 的 4 个本地角点旋转+平移到世界坐标
        3. 收集所有 像素↔世界 点对
        4. 使用 RANSAC 鲁棒估计单应矩阵 H
        5. 构建 世界→逆透视视图 的组合矩阵 new_H

    Args:
        reference_tags: 当前帧检测到的参考 Tag 字典
        corners:        OpenCV 检测到的所有角点
        ids:            OpenCV 检测到的所有 Tag ID

    Returns:
        (H, new_H):
            H:     Pixel → World 单应矩阵 (3x3), 或 None
            new_H: Pixel → WarpedView 组合矩阵 (3x3), 或 None

    前置条件: 至少检测到 4 个参考 Tag (场地四角齐)
    """
    if len(reference_tags) < 4:
        return None, None

    # corner_id → yaw 映射 (参考 Tag 在场地中的旋转角度)
    yaw_map = {0: 0, 1: np.pi, 2: 3 * np.pi / 2, 3: np.pi / 2}

    src_points = []  # 图像像素点
    dst_points = []  # 对应的世界坐标点

    for tag_id, tag_info in reference_tags.items():
        cid = tag_info.corner_id
        if cid not in yaw_map:
            print(f"⚠️ 参考 Tag {tag_id} 的 corner_id={cid} 不在 yaw_map 中，已跳过")
            continue

        # 从原始检测结果中匹配像素角点
        found_corner = None
        for i in range(len(ids)):
            if int(ids[i][0]) == tag_id:
                found_corner = corners[i][0]
                break
        if found_corner is None:
            continue

        corner_px = found_corner
        wx_base, wz_base = tag_info.world_pos
        yaw = yaw_map[cid]
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        # 将 Tag 的 4 个本地角点旋转+平移得到世界坐标
        for j in range(4):
            lx = [
                -MARKER_LENGTH / 2,
                MARKER_LENGTH / 2,
                MARKER_LENGTH / 2,
                -MARKER_LENGTH / 2,
            ][j]
            ly = [
                MARKER_LENGTH / 2,
                MARKER_LENGTH / 2,
                -MARKER_LENGTH / 2,
                -MARKER_LENGTH / 2,
            ][j]
            world_x = wx_base + lx * cos_y - ly * sin_y
            world_z = wz_base + lx * sin_y + ly * cos_y
            src_points.append(corner_px[j])
            dst_points.append([world_x, world_z])

    if len(src_points) < 4:
        return None, None

    src = np.array(src_points, dtype=np.float32)
    dst = np.array(dst_points, dtype=np.float32)

    # RANSAC 鲁棒估计单应矩阵
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return None, None

    # 构建 世界→逆透视视图 的组合矩阵
    # 变换: 平移到世界原点外 + 缩放 + Y 轴翻转
    world_to_warp = np.array(
        [
            [SCALE, 0, MARGIN * SCALE],
            [0, -SCALE, (FIELD_SIZE_Z + MARGIN) * SCALE],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    new_H = world_to_warp @ H

    return H, new_H


# ===================== 目标位姿估计 =====================
def estimate_target_pose(
    target_obs: Optional[TargetObservation],
    H: Optional[np.ndarray],
    kf: KalmanFilter2D,
    last_pose: Optional[TargetPose],
) -> Tuple[Optional[TargetPose], Deque[Tuple[float, float]]]:
    """
    估计目标 (小车) 在场地 XZ 平面坐标系下的位姿

    处理流程:
        1. 坐标转换: 中心点和前端点 像素 → 世界 (通过单应矩阵 H)
        2. 朝向计算: atan2(fz - car_z, fx - car_x)
        3. 卡尔曼滤波: 平滑位置和角度
        4. 角度归一化: normalize_angle
        5. 轨迹更新: 保留最近 TRAIL_LENGTH 个历史位置点

    Args:
        target_obs: 目标 Tag 的原始观测 (None=未检测到)
        H:          Pixel→World 单应矩阵 (None=不可用)
        kf:         卡尔曼滤波器实例
        last_pose:  上一帧的位姿 (用于未检测到时保持状态)

    Returns:
        (current_pose, trail_points):
            current_pose: 当前帧的 TargetPose，或 last_pose (观测无效时)
            trail_points: 历史轨迹点队列

    Note:
        当 H=None 或 target_obs=None 时，返回 last_pose 保持上一帧状态。
        trail_points 通过 setattr 绑定到 current_pose.trail。
    """
    trail_points: Deque[Tuple[float, float]] = deque(maxlen=TRAIL_LENGTH)
    if last_pose is not None and hasattr(last_pose, "trail"):
        trail_points = getattr(last_pose, "trail")

    if H is None or target_obs is None:
        return last_pose, trail_points

    cx_px, cy_px = target_obs.center_px

    car_pos = pixel_to_world(cx_px, cy_px, H)
    if car_pos is None:
        return last_pose, trail_points
    car_x, car_z = car_pos

    # Yaw from 4 tag corners mapped through homography
    # corner[0]→corner[1] = tag's +X axis, full 0.5m span → 2x better SNR
    world_corners = []
    for j in range(4):
        pos = pixel_to_world(target_obs.corners[j][0], target_obs.corners[j][1], H)
        if pos is None:
            break
        world_corners.append(pos)

    if len(world_corners) == 4:
        dx = world_corners[1][0] - world_corners[0][0]
        dz = world_corners[1][1] - world_corners[0][1]
        car_yaw = np.degrees(np.arctan2(dz, dx))
        print(f"[YAW_RAW] {car_yaw:.1f}°", end="  ")
    else:
        return last_pose, trail_points

    filtered = kf.update(np.array([car_x, car_z, car_yaw]))
    car_x, car_z, car_yaw = filtered
    car_yaw = normalize_angle(car_yaw)
    print(f"→ KF:{car_yaw:.1f}°")  # debug: 对比 raw vs filtered

    trail_points.append((car_x, car_z))

    current_pose = TargetPose(x=car_x, z=car_z, yaw_deg=car_yaw)
    setattr(current_pose, "trail", trail_points)

    return current_pose, trail_points
