"""
visualizer.py — 可视化绘制模块

提供三个 OpenCV 窗口的绘制逻辑:

    draw_axis()           : 在图像上绘制 3D 坐标轴 (X:红, Y:绿, Z:蓝)
    draw_raw_window()     : 窗口 1 — 原始摄像头画面 + Tag 检测 + UDP 统计
    draw_warped_window()  : 窗口 2 — 全画面逆透视矫正 (Bird's Eye View)
    draw_map_window()     : 窗口 3 — 纯抽象坐标可视化 (网格 + 轨迹 + 朝向箭头)

所有绘制函数都不修改输入帧 (通过 .copy() 或零图创建新画布)。
"""

from typing import Optional, Tuple, Dict, Deque
from collections import deque

import cv2
import numpy as np

from config import (
    MARKER_LENGTH,
    FIELD_SIZE_X,
    FIELD_SIZE_Z,
    SCALE,
    MARGIN,
    WARP_WIDTH,
    WARP_HEIGHT,
    REF_TAG_CONFIG,
    CAMERA_MATRIX,
    DIST_COEFFS,
)
from datatypes import ReferenceTagInfo, TargetObservation, TargetPose
from utils import world_to_vis, world_to_warped, transform_box_to_warped


# ===================== 3D 坐标轴绘制 =====================
def draw_axis(img: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, length: float) -> None:
    """
    在图像上绘制 3D 坐标轴，用于可视化 Tag 的姿态估计结果

    坐标轴颜色约定:
        X 轴 — 红色 (Tag 的前方)
        Y 轴 — 绿色 (Tag 的左侧)
        Z 轴 — 蓝色 (垂直于 Tag 平面向上)

    使用 OpenCV 的 projectPoints 将 3D 轴端点投影到图像上。

    Args:
        img:    要绘制的目标图像 (就地修改)
        rvec:   PnP 解算的旋转向量 (Rodrigues)
        tvec:   PnP 解算的平移向量
        length: 坐标轴长度 [米]
    """
    # 定义 4 个轴端点 (包括原点)
    points = np.float32(
        [[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]]
    ).reshape(-1, 3)

    # 投影到图像平面
    img_points, _ = cv2.projectPoints(points, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS)
    img_points = np.int32(img_points).reshape(-1, 2)

    # 绘制轴线 (从原点到各方向)
    cv2.line(img, tuple(img_points[0]), tuple(img_points[1]), (0, 0, 255), 2)  # X 红
    cv2.line(img, tuple(img_points[0]), tuple(img_points[2]), (0, 255, 0), 2)  # Y 绿
    cv2.line(img, tuple(img_points[0]), tuple(img_points[3]), (255, 0, 0), 2)  # Z 蓝


# ===================== 窗口 1 — 原始摄像头画面 =====================
def draw_raw_window(
    frame: np.ndarray,
    reference_tags: Dict[int, ReferenceTagInfo],
    target_obs: Optional[TargetObservation],
    avg_fps: float,
    detected_count: int,
    udp_stats: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    绘制窗口 1 — 原始摄像头画面

    叠加显示内容:
        1. 参考 Tag 的 3D 坐标轴 (绿色→场地边框连线)
        2. 目标 Tag 的检测框 (蓝色) + 朝向箭头 (黄色)
        3. 左上角状态信息: FPS / Tags / UDP FPS / UDP Latency

    Args:
        frame:          原始摄像头帧 (BGR)
        reference_tags: 检测到的参考 Tag 字典
        target_obs:     目标 Tag 观测 (None=未检测到)
        avg_fps:        滑动平均帧率
        detected_count: 检测到的参考 Tag 数量
        udp_stats:      UDP 发送统计 (可选，用于显示实时速率)

    Returns:
        绘制完成后的图像 (新副本)
    """
    win1 = frame.copy()

    for tag in reference_tags.values():
        draw_axis(win1, tag.rvec, tag.tvec, MARKER_LENGTH / 2)

    if len(reference_tags) == 4:
        sorted_tags = sorted(reference_tags.values(), key=lambda x: x.corner_id)
        field_corners = np.array([t.center_px for t in sorted_tags], dtype=np.int32)
        cv2.polylines(win1, [field_corners], True, (0, 255, 0), 3)

    if target_obs is not None:
        box = np.int32(target_obs.corners)
        cv2.polylines(win1, [box], True, (255, 0, 0), 2)
        cx, cy = target_obs.center_px
        fx, fy = target_obs.front_px
        cv2.arrowedLine(win1, (int(cx), int(cy)), (int(fx), int(fy)), (0, 255, 255), 2)

    cv2.putText(win1, f"FPS: {avg_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(win1, f"Tags: {detected_count}/4", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if udp_stats:
        cv2.putText(win1, f"UDP FPS: {udp_stats.get('udp_fps', 0):.1f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(win1, f"UDP Latency: {udp_stats.get('avg_latency_ms', 0):.1f}ms",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return win1


# ===================== 窗口 2 — 逆透视矫正 (鸟瞰图) =====================
def draw_warped_window(
    raw_frame: np.ndarray,
    H: Optional[np.ndarray],
    new_H: Optional[np.ndarray],
    reference_tags: Dict[int, ReferenceTagInfo],
    target_obs: Optional[TargetObservation],
    corners: Tuple[np.ndarray, ...],
    ids: Optional[np.ndarray],
    target_pose: Optional[TargetPose],
    detected_count: int,
) -> np.ndarray:
    """
    绘制窗口 2 — 全画面逆透视矫正 (Bird's Eye View)

    将整个摄像画面通过 warped 矩阵变换到鸟瞰视角，叠加:
        1. 场地边框 (绿色多边形，基于 REF_TAG_CONFIG 的世界坐标)
        2. 所有检测到的 Tag 框 (参考 Tag 黄色，其他灰色)
        3. 目标小车位置 (红色框 + 朝向箭头 + 坐标标注)

    Args:
        raw_frame:      未经绘制的原始帧 (用于 warped 变换)
        H:              Pixel→World 单应矩阵
        new_H:          Pixel→WarpedView 组合矩阵
        reference_tags: 检测到的参考 Tag
        target_obs:     目标 Tag 观测
        corners:        所有检测到的角点
        ids:            所有检测到的 Tag ID
        target_pose:    目标位姿
        detected_count: 检测到的参考 Tag 数量

    Returns:
        鸟瞰图图像
    """
    win2 = np.zeros((WARP_HEIGHT, WARP_WIDTH, 3), dtype=np.uint8)

    if H is None or new_H is None:
        cv2.putText(
            win2,
            f"等待所有参考 Tag... 已检测：{detected_count}/4",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            1,
        )
        return win2

    try:
        warped_frame = cv2.warpPerspective(raw_frame, new_H, (WARP_WIDTH, WARP_HEIGHT))
        win2 = warped_frame.copy()
        cv2.putText(win2, "Full Warped View", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        field_warped = []  # 按 corner_id 0→1→2→3 绘制场地方框
        for cid in range(4):
            for tag_id, config in REF_TAG_CONFIG.items():
                if config["corner_id"] == cid:
                    wx, wz = config["world_pos"]
                    px, py = world_to_warped(wx, wz)
                    field_warped.append([int(px), int(py)])
                    break
        if len(field_warped) == 4:
            cv2.polylines(win2, [np.array(field_warped, dtype=np.int32)], True, (0, 255, 0), 3)

        if ids is not None:  # 绘制所有检测到的 Tag
            for i in range(len(ids)):
                tag_id = int(ids[i][0])
                corner = corners[i][0]
                warped_tag = transform_box_to_warped(corner, H, new_H)
                if warped_tag is not None:
                    color = (0, 255, 255) if tag_id in reference_tags else (128, 128, 128)
                    cv2.polylines(win2, [warped_tag], True, color, 2)
                    cx = int(np.mean(warped_tag[:, 0]))
                    cy = int(np.mean(warped_tag[:, 1]))
                    cv2.putText(win2, str(tag_id), (cx - 10, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if target_obs is not None and target_pose is not None:  # 目标小车
            warped_car = transform_box_to_warped(target_obs.corners, H, new_H)
            if warped_car is not None:
                cv2.polylines(win2, [warped_car], True, (0, 0, 255), 3)
                center_x = int(np.mean(warped_car[:, 0]))
                center_y = int(np.mean(warped_car[:, 1]))

                arrow_len = 0.3 * SCALE
                end_x = center_x + int(arrow_len * np.cos(np.radians(target_pose.yaw_deg)))
                end_y = center_y - int(arrow_len * np.sin(np.radians(target_pose.yaw_deg)))
                cv2.arrowedLine(win2, (center_x, center_y), (end_x, end_y), (255, 0, 0), 3)

                cv2.putText(win2, f"Car: ({target_pose.x:.2f}, {target_pose.z:.2f})",
                            (center_x + 20, center_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    except Exception as e:
        cv2.putText(
            win2, f"Warp Error: {str(e)[:50]}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
        )

    return win2


# ===================== 窗口 3 — 坐标可视化 (地图) =====================
def draw_map_window(
    target_pose: Optional[TargetPose],
    trail_points: Deque[Tuple[float, float]],
    reference_tags: Dict[int, ReferenceTagInfo],
    detected_count: int,
) -> np.ndarray:
    """
    绘制窗口 3 — 纯抽象的 XZ 平面坐标可视化

    不包括摄像头图像纹理，仅显示:
        1. 网格背景 (灰色) + 刻度标注 (米)
        2. 场地边界 (白色矩形)
        3. 参考 Tag 位置 (绿色圆点 / 灰色未检测)
        4. 历史轨迹 (黄色折线, 最多 TRAIL_LENGTH 点)
        5. 当前小车位置 (绿色大圆) + 朝向箭头 (红色)

    Args:
        target_pose:     目标位姿 (None=未检测到)
        trail_points:    历史轨迹点队列 [(x, z), ...]
        reference_tags:  检测到的参考 Tag
        detected_count:  检测到的参考 Tag 数量

    Returns:
        坐标可视化图像
    """
    win3 = np.zeros((WARP_HEIGHT, WARP_WIDTH, 3), dtype=np.uint8)

    for x in range(-1, int(FIELD_SIZE_X) + 2):  # X 方向网格
        px1, py1 = world_to_vis(x, -1)
        px2, py2 = world_to_vis(x, FIELD_SIZE_Z + 1)
        cv2.line(win3, (px1, py1), (px2, py2), (50, 50, 50), 1)
        if 0 <= x <= FIELD_SIZE_X:
            tx, ty = world_to_vis(x, 0)
            cv2.putText(win3, f"{x}m", (tx - 10, ty + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    for z in range(-1, int(FIELD_SIZE_Z) + 2):  # Z 方向网格
        px1, py1 = world_to_vis(-1, z)
        px2, py2 = world_to_vis(FIELD_SIZE_X + 1, z)
        cv2.line(win3, (px1, py1), (px2, py2), (50, 50, 50), 1)
        if 0 <= z <= FIELD_SIZE_Z:
            tx, ty = world_to_vis(0, z)
            cv2.putText(win3, f"{z}m", (tx + 5, ty + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    bl_x, bl_y = world_to_vis(0, 0)
    tr_x, tr_y = world_to_vis(FIELD_SIZE_X, FIELD_SIZE_Z)
    cv2.rectangle(win3, (bl_x, tr_y), (tr_x, bl_y), (255, 255, 255), 2)  # 场地边框

    for tag_id, config in REF_TAG_CONFIG.items():  # 参考 Tag 标记
        wx, wz = config["world_pos"]
        px, py = world_to_vis(wx, wz)
        color = (0, 255, 0) if tag_id in reference_tags else (80, 80, 80)
        cv2.circle(win3, (px, py), 10, color, -1)
        cv2.putText(win3, str(tag_id), (px - 5, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if len(trail_points) > 1:  # 历史轨迹
        trail_pts = np.array([world_to_vis(x, z) for x, z in trail_points], np.int32)
        cv2.polylines(win3, [trail_pts], False, (0, 165, 255), 2)

    if target_pose is not None:  # 当前小车位置 + 朝向
        px, py = world_to_vis(target_pose.x, target_pose.z)
        cv2.circle(win3, (px, py), 12, (0, 220, 0), -1)

        arrow_len = 0.4
        dx = arrow_len * np.cos(np.radians(target_pose.yaw_deg))
        dz = arrow_len * np.sin(np.radians(target_pose.yaw_deg))
        ex, ey = world_to_vis(target_pose.x + dx, target_pose.z + dz)
        cv2.arrowedLine(win3, (px, py), (ex, ey), (255, 0, 0), 3)

        cv2.putText(win3, f"({target_pose.x:.2f}, {target_pose.z:.2f})",
                    (px + 15, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 25), 1)

    cv2.putText(win3, "Coordinate Visualization (XZ plane)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if detected_count < 4:
        cv2.putText(win3, f"等待所有参考 Tag... 已检测：{detected_count}/4",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    return win3
