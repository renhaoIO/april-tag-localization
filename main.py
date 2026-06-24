"""
main.py — 主程序入口模块

职责:
    - 解析命令行参数 (parse_args)
    - 构建 AprilTag 检测器 (build_detector)
    - 打开摄像头 (open_camera)
    - 创建通信组件 (create_udp_transmitter)
    - 编排主循环 (main): 检测 → 计算 → UDP 发送 → JSON 输出 → 可视化

主循环流程:
    相机帧 → detectMarkers → process_markers → compute_homography
    → estimate_target_pose → UDP 发送 + JSON 输出 + 三窗口绘制
"""

import argparse
import time
import traceback
from collections import deque
from typing import Optional, Dict, Deque

import cv2
import numpy as np

from config import (
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_FPS,
    DEFAULT_UDP_IP,
    DEFAULT_UDP_PORT,
    DEFAULT_TARGET_TAG_ID,
    CAR_HEIGHT,
    FIELD_SIZE_X,
    FIELD_SIZE_Z,
    WINDOW1_WIDTH,
    WINDOW1_HEIGHT,
    WINDOW_DISPLAY_FPS,
    JSON_OUTPUT_MODES,
    DEFAULT_JSON_MODE,
    DEFAULT_JSON_FILE,
)
from datatypes import ReferenceTagInfo, TargetObservation, TargetPose
from comms import JSONOutputHandler, LowLatencyUDPTransmitter
from pipeline import KalmanFilter2D, process_markers, compute_homography, estimate_target_pose
from utils import normalize_angle
from visualizer import draw_raw_window, draw_warped_window, draw_map_window


# ===================== 命令行参数解析 =====================
def parse_args() -> argparse.Namespace:
    """
    解析命令行参数

    支持的参数组:
        - 摄像头: --camera, --width, --height, --fps
        - UDP:    --udp-ip, --udp-port, --udp-mode (json|binary)
        - Tag:    --target-tag-id
        - JSON:   --json-output (none|console|file|both), --json-file

    Returns:
        解析后的 Namespace 对象

    Example:
        python main.py --camera 0 --udp-ip 192.168.1.105 --udp-mode json --json-output both
    """
    parser = argparse.ArgumentParser(
        description="三窗口 AprilTag 定位调试工具 v3.0 模块化版 (Y→Z + JSON输出 + UDP双模式)"
    )

    camera_group = parser.add_argument_group("摄像头配置")
    camera_group.add_argument("--camera", type=int, default=None, help="摄像头索引 (默认自动检测)")
    camera_group.add_argument("--width", type=int, default=DEFAULT_CAMERA_WIDTH, help="摄像头画面宽度")
    camera_group.add_argument("--height", type=int, default=DEFAULT_CAMERA_HEIGHT, help="摄像头画面高度")
    camera_group.add_argument("--fps", type=int, default=DEFAULT_CAMERA_FPS, help="摄像头帧率设置")

    udp_group = parser.add_argument_group("UDP 网络配置")
    udp_group.add_argument("--udp-ip", type=str, default=DEFAULT_UDP_IP, help="UDP 目标 IP (设为 'none' 关闭)")
    udp_group.add_argument("--udp-port", type=int, default=DEFAULT_UDP_PORT, help="UDP 目标端口")
    udp_group.add_argument(
        "--udp-mode", type=str, default="json", choices=["json", "binary"],
        help="UDP 发送格式: json(默认) 或 binary",
    )

    tag_group = parser.add_argument_group("Tag 配置")
    tag_group.add_argument("--target-tag-id", type=int, default=DEFAULT_TARGET_TAG_ID, help="要跟踪的目标 Tag ID")

    json_group = parser.add_argument_group("JSON 输出配置")
    json_group.add_argument(
        "--json-output", type=str, default=DEFAULT_JSON_MODE,
        choices=JSON_OUTPUT_MODES,
        help="JSON 输出模式: none(不输出), console(控制台), file(文件), both(双重)",
    )
    json_group.add_argument("--json-file", type=str, default=DEFAULT_JSON_FILE, help="JSON 输出文件路径")

    return parser.parse_args()


# ===================== AprilTag 检测器构建 =====================
def build_detector() -> cv2.aruco.ArucoDetector:
    """
    构建 AprilTag 检测器

    优先使用高精度 AprilTag 25h9 字典，不可用时回退到 ArUco 6x6_250。

    Returns:
        配置好的 OpenCV ArucoDetector 实例
    """
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9)
        print("✅ 使用 AprilTag 25h9 字典 (高精度)")
    except AttributeError:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        print("⚠️ AprilTag 字典不可用，回退到标准 ArUco 6X6_250 字典")

    parameters = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(aruco_dict, parameters)


# ===================== 摄像头初始化 =====================
def open_camera(
    camera_index: Optional[int],
    width: int,
    height: int,
    fps: int,
) -> tuple[cv2.VideoCapture, int]:
    """
    打开摄像头，自动重试多个索引和后端

    尝试策略:
        - 指定索引时只尝试该索引
        - 未指定时尝试 [0] (Windows 默认)
        - 后端: 优先 CAP_DSHOW (Windows DirectShow)，失败回退默认后端
        - 设置: 分辨率 + 帧率 + BUFFERSIZE=1 (最小缓冲)

    Args:
        camera_index: 摄像头索引 (None=自动检测)
        width:        期望画面宽度
        height:       期望画面高度
        fps:          期望帧率

    Returns:
        (capture, index): VideoCapture 对象和实际使用的索引

    Raises:
        RuntimeError: 所有尝试均失败
    """
    candidate_indices = [camera_index] if camera_index is not None else [0]
    backends = [cv2.CAP_DSHOW] if hasattr(cv2, "CAP_DSHOW") else []
    backends.append(None)

    attempts = []
    for index in candidate_indices:
        for backend in backends:
            capture = (
                cv2.VideoCapture(index)
                if backend is None
                else cv2.VideoCapture(index, backend)
            )
            backend_name = "default" if backend is None else "CAP_DSHOW"
            attempts.append(f"{index}@{backend_name}")

            if not capture.isOpened():
                capture.release()
                continue

            # 配置摄像头参数
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            capture.set(cv2.CAP_PROP_FPS, fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小缓冲区，降低延迟

            ok, _ = capture.read()
            if ok:
                print(f"✅ 已打开摄像头：索引{index}, 后端{backend_name}, 分辨率{width}x{height}")
                return capture, index
            capture.release()

    raise RuntimeError(f"❌ 无法打开摄像头，已尝试：{'、'.join(attempts)}")


# ===================== UDP 发送器创建 =====================
def create_udp_transmitter(
    udp_ip: Optional[str],
    udp_port: Optional[int],
    mode: str = "json",
) -> Optional[LowLatencyUDPTransmitter]:
    """
    创建并启动 UDP 发送器

    对 udp_ip='none' 或 None 的情况优雅降级 (不启动网络功能)。

    Args:
        udp_ip:   目标 IP 地址 ("none"=关闭)
        udp_port: 目标端口号
        mode:     发送格式 "json" 或 "binary"

    Returns:
        LowLatencyUDPTransmitter 实例 (已启动), 或 None (功能关闭)
    """
    if not udp_ip or udp_ip.lower() == "none":
        print("ℹ️ UDP 功能已关闭")
        return None
    if udp_port is None:
        return None

    transmitter = LowLatencyUDPTransmitter(udp_ip, udp_port, mode=mode)
    if transmitter.start():
        return transmitter
    return None


# ===================== 主程序 =====================
def main():
    """
    主程序入口 — 编排整个定位系统的运行

    初始化 → 主循环 (无限循环直到用户按 q 退出) → 清理

    主循环每秒执行 ~100 次 (受摄像头帧率影响):
        1. 读帧 + FPS 计算
        2. detectMarkers → 检测所有 AprilTag
        3. process_markers → 区分参考/目标 Tag, PnP 解算
        4. compute_homography → 基于 4 个参考 Tag 计算单应矩阵
        5. estimate_target_pose → 位姿估计 + 卡尔曼滤波 + 轨迹更新
        6. UDP 异步发送 (非阻塞, 独立线程)
        7. JSON 输出 (可选)
        8. 帧率节流显示 (每 N 帧刷新一次三窗口)

    异常处理:
        - KeyboardInterrupt: 用户 Ctrl+C 正常退出
        - 其他异常: 打印 traceback 并清理
    """
    # 初始化
    args = parse_args()
    detector = build_detector()
    cap, _ = open_camera(args.camera, args.width, args.height, args.fps)

    json_output = JSONOutputHandler(args.json_output, args.json_file)
    udp_tx = create_udp_transmitter(args.udp_ip, args.udp_port, mode=args.udp_mode)

    kf = KalmanFilter2D()
    last_target_pose: Optional[TargetPose] = None
    fps_history: Deque[float] = deque(maxlen=30)
    last_time = time.time()
    frame_count = 0
    display_interval = max(1, int(DEFAULT_CAMERA_FPS / WINDOW_DISPLAY_FPS))

    print("=" * 50)
    print("=== 三窗口 AprilTag 定位调试工具 v3.0 模块化版 ===")
    print("=== Y→Z坐标系 + JSON输出 + UDP双模式 ===")
    print("窗口 1: 原始画面 (Raw) + UDP 实时速率")
    print("窗口 2: 全画面逆透视矫正 (Warped) - XZ平面")
    print("窗口 3: 坐标可视化 (Map) - XZ平面")
    print(f"JSON 输出: {args.json_output.upper()} → {args.json_file}")
    print(f"UDP 格式: {args.udp_mode.upper()}")
    print("按 q 退出")
    print(f"场地: {FIELD_SIZE_X}m x {FIELD_SIZE_Z}m | 目标 Tag ID: {args.target_tag_id}")
    print("=" * 50)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 无法获取摄像头画面")
                break

            raw_frame = frame.copy()
            frame_count += 1

            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            fps = 1.0 / dt if dt > 0 else 0
            fps_history.append(fps)
            avg_fps = np.mean(fps_history)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            reference_tags: Dict[int, ReferenceTagInfo] = {}
            target_obs: Optional[TargetObservation] = None

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                reference_tags, target_obs = process_markers(gray, corners, ids, args.target_tag_id)

            H, new_H = None, None
            if ids is not None and len(reference_tags) >= 4:
                H, new_H = compute_homography(reference_tags, corners, ids)

            target_pose, trail_points = estimate_target_pose(target_obs, H, kf, last_target_pose)
            last_target_pose = target_pose

            if target_pose is not None:
                # 坐标系映射: 世界(X=右,Z=前) → 智能车(X=前,Y=右), yaw偏移-90°
                sm_x = target_pose.z
                sm_y = target_pose.x
                sm_yaw = normalize_angle(target_pose.yaw_deg - 90.0)

                if udp_tx is not None:
                    success = udp_tx.send(sm_x, CAR_HEIGHT, sm_y, sm_yaw)
                    if not success and frame_count % 30 == 0:
                        print("⚠️ UDP 队列已满，数据可能丢失")

                if json_output.mode != "none":
                    json_data = {
                        "timestamp": time.time(),
                        "frame": frame_count,
                        "position": {
                            "x": round(sm_x, 4),
                            "y": CAR_HEIGHT,
                            "z": round(sm_y, 4),
                        },
                        "orientation": {
                            "yaw": round(sm_yaw, 4),
                            "pitch": 0.0,
                            "roll": 0.0,
                        },
                        "detection": {
                            "tags_found": len(reference_tags),
                            "target_tag_id": args.target_tag_id,
                        },
                    }
                    json_output.output(json_data)

            if frame_count % display_interval == 0:  # 帧率节流显示
                udp_stats = udp_tx.get_stats() if udp_tx else {}
                win1 = draw_raw_window(frame, reference_tags, target_obs, avg_fps,
                                       len(reference_tags), udp_stats)
                win2 = draw_warped_window(raw_frame, H, new_H, reference_tags, target_obs,
                                          corners, ids, target_pose, len(reference_tags))
                win3 = draw_map_window(target_pose, trail_points, reference_tags, len(reference_tags))

                win1 = cv2.resize(win1, (WINDOW1_WIDTH, WINDOW1_HEIGHT))
                cv2.imshow("1. Raw Camera View", win1)
                cv2.imshow("2. Full Warped View", win2)
                cv2.imshow("3. Coordinate Visualization", win3)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nℹ️ 用户请求退出")
                break

    except KeyboardInterrupt:
        print("\nℹ️ 用户中断")
    except Exception as e:
        print(f"❌ 运行异常: {e}")
        traceback.print_exc()
    finally:
        cap.release()
        if udp_tx is not None:
            udp_tx.stop()
        json_output.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
