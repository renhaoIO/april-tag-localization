"""
utils.py — 数学工具与坐标转换模块

提供定位系统中所有底层数学计算:
    normalize_angle()          : 角度归一化到 [-180, 180)
    normalize_angle_diff()     : 角度差异跳变处理 (KF 专用)
    pixel_to_world()           : 单应矩阵: 像素坐标 → 世界坐标
    world_to_vis()             : 世界坐标 → 可视化窗口像素坐标 (Y 轴翻转)
    world_to_warped()          : 世界坐标 → 逆透视图像浮点坐标
    transform_box_to_warped()  : 矩形框的级联坐标变换

所有函数均为纯函数 (无副作用)，输入输出明确。
"""

from typing import Optional, Tuple

from config import SCALE, MARGIN, FIELD_SIZE_X, FIELD_SIZE_Z
import numpy as np


def normalize_angle(angle_deg: float) -> float:
    """
    将角度归一化到 [-180, 180) 区间

    使用模运算避免角度突变 (如从 179° 突跳到 -179° 时保持连续性)。
    常用于卡尔曼滤波后的 yaw 值归一化。

    Args:
        angle_deg: 输入角度 [度]，可以是任意值

    Returns:
        归一化后的角度，范围 [-180, 180)

    Example:
        >>> normalize_angle(190)
        -170.0
        >>> normalize_angle(-190)
        170.0
    """
    return (angle_deg + 180.0) % 360.0 - 180.0


def normalize_angle_diff(diff_deg: float) -> float:
    """
    处理角度差异的 ±180° 跳变问题 (卡尔曼滤波专用)

    在 KF 的新息 (Innovation) 计算中使用。当测量 yaw 和预测 yaw
    分别位于 +179° 和 -179° 时，实际差异为 2°，而非 358°。
    本函数确保差异始终在 [-180, 180) 区间内。

    Args:
        diff_deg: 角度差 [度]

    Returns:
        修正后的角度差，范围 [-180, 180)

    Example:
        >>> normalize_angle_diff(358)   # 实际应理解为 -2°
        -2.0
    """
    diff_deg = diff_deg % 360.0
    if diff_deg >= 180.0:
        diff_deg -= 360.0
    return diff_deg


def pixel_to_world(px: float, py: float, H: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    利用单应性矩阵将像素坐标转换为世界坐标

    通过齐次坐标变换 p' = H @ [px, py, 1]^T，再除以第三个分量得到
    世界坐标 (wx, wz)。如果第三个分量接近零或异常则返回 None。

    Args:
        px: 像素 x 坐标
        py: 像素 y 坐标
        H:  3x3 单应性矩阵 (Pixel → World)

    Returns:
        (x, z) 世界坐标 [米] 的 Tuple，若转换失败返回 None

    Note:
        单应矩阵假设所有点位于同一平面上 (地面)，因此 XZ 平面的
        高度信息 (Y 轴) 无法从单应矩阵恢复。
    """
    try:
        p = np.array([px, py, 1.0])
        wp = H @ p  # 齐次坐标变换
        if abs(wp[2]) < 1e-6:
            return None  # 避免除以零 (无穷远点)
        return wp[0] / wp[2], wp[1] / wp[2]
    except Exception:
        return None


def world_to_vis(wx: float, wz: float) -> Tuple[int, int]:
    """
    将世界坐标 (XZ 平面) 转换为可视化窗口的像素坐标

    坐标变换包含:
        1. 添加边距 (MARGIN)
        2. 缩放 (SCALE)
        3. Y 轴翻转 (图像坐标系 Y 向下 vs 世界坐标系 Z 向上)

    Args:
        wx: 世界 X 坐标 [米]
        wz: 世界 Z 坐标 [米] (Y→Z 交换)

    Returns:
        (px, py) 可视化窗口像素坐标的 Tuple
    """
    px = int((wx + MARGIN) * SCALE)
    py = int((FIELD_SIZE_Z + MARGIN - wz) * SCALE)
    return px, py


def world_to_warped(wx: float, wz: float) -> Tuple[float, float]:
    """
    将世界坐标 (XZ 平面) 转换为逆透视图像的浮点坐标

    与 world_to_vis 相同的变换逻辑，但返回浮点数以支持
    精确的亚像素绘制。

    Args:
        wx: 世界 X 坐标 [米]
        wz: 世界 Z 坐标 [米] (Y→Z 交换)

    Returns:
        (px, py) 逆透视图像浮点坐标的 Tuple
    """
    px = (wx + MARGIN) * SCALE
    py = (FIELD_SIZE_Z + MARGIN - wz) * SCALE
    return px, py


def transform_box_to_warped(
    box_points: np.ndarray,
    H: np.ndarray,
    new_H: np.ndarray,
) -> Optional[np.ndarray]:
    """
    将原始图像中的矩形框角点，通过两级变换映射到逆透视图像坐标

    变换链路:
        原始图像像素 → [H] → 世界坐标 → [world_to_warped] → 逆透视视图坐标

    用于在 Warped 窗口中绘制检测到的 Tag 框。

    Args:
        box_points: 原始图像的 4 个角点 (shape: Nx2)
        H:     Pixel → World 单应矩阵 (3x3)
        new_H: 预留参数，实际由 world_to_warped 完成第二步变换

    Returns:
        逆透视图像中的 N 个角点 (int32)，若变换失败返回 None

    Note:
        每步变换都会检查齐次坐标的第三分量是否接近零。
        如果某个点变换失败，该点被跳过。
    """
    try:
        warped_points = []
        for (px, py) in box_points:
            # Step 1: Pixel → World
            p = np.array([px, py, 1.0])
            wp = H @ p
            if abs(wp[2]) < 1e-6:
                continue
            wx, wz = wp[0] / wp[2], wp[1] / wp[2]

            # Step 2: World → Warped Image
            warped_x, warped_y = world_to_warped(wx, wz)
            warped_points.append([warped_x, warped_y])

        if len(warped_points) == 4:
            return np.array(warped_points, dtype=np.int32)
        return None
    except Exception:
        return None
