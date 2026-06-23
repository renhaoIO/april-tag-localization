"""
datatypes.py — 数据结构定义模块

定义定位系统中流通的三个核心数据结构:
    ReferenceTagInfo   : 参考 Tag (场地四角) 的检测与位姿信息
    TargetObservation  : 目标 Tag (小车) 的原始观测数据
    TargetPose         : 解算后的目标位姿 (XZ 平面 + Yaw)

三个结构体形成数据流: Tag 检测 → 观测 → 位姿
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ReferenceTagInfo:
    """
    参考 Tag 的检测与位姿信息

    用于构建单应性矩阵 (Homography)，建立像素坐标与世界坐标的映射关系。
    每个参考 Tag 对应场地的一个角点，通过 PnP 解算得到其在相机坐标系下的姿态。

    Attributes:
        tag_id:     AprilTag 编号 (物理贴在场地上的 Tag ID)
        corner_id:  场地角点索引 (0=左下/原点, 1=右下, 2=右上, 3=左上)
        center_px:  图像像素中心坐标 (u, v)
        world_pos:  对应的世界坐标 (x, z) [米] — 从 REF_TAG_CONFIG 读取
        rvec:       PnP 解算的旋转向量 (Rodrigues 格式)
        tvec:       PnP 解算的平移向量
    """
    tag_id: int
    corner_id: int
    center_px: Tuple[float, float]
    world_pos: Tuple[float, float]
    rvec: np.ndarray
    tvec: np.ndarray


@dataclass
class TargetObservation:
    """
    目标 Tag 的原始观测数据

    包含中心点和前端点，用于计算小车的朝向 (Yaw)。
    前端点通过 PnP 解算的位姿将 Tag 本地坐标系的 X 轴正方向点投影到图像中得到。

    Attributes:
        center_px: 图像像素中心坐标 (u, v) — 用于计算位置
        front_px:  图像像素前端点坐标 — 用于计算朝向
        corners:   4 个角点的像素坐标数组 (shape: 4x2)
    """
    center_px: Tuple[float, float]
    front_px: Tuple[float, float]
    corners: np.ndarray


@dataclass
class TargetPose:
    """
    解算后的目标位姿 (场地 XZ 平面坐标系)

    经过单应矩阵变换和卡尔曼滤波后的最终定位结果。
    坐标系语义 (Y→Z 交换):
        X 轴: 场地长轴方向
        Z 轴: 场地短轴方向 (原 Y 轴)
        Y 轴: 高度 (不在此结构中，由 CAR_HEIGHT 常量表示)

    Attributes:
        x:       X 坐标 [米]
        z:       Z 坐标 [米] (Y→Z 交换: 原 y → z)
        yaw_deg: 偏航角 (Yaw) [度], 范围 [-180, 180)
    """
    x: float
    z: float
    yaw_deg: float
