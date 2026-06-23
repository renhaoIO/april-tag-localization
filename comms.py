"""
comms.py — 通信模块

提供定位系统的数据输出能力，包含两个组件:

    JSONOutputHandler           : JSON Lines 格式输出器 (控制台/文件/双模式)
    LowLatencyUDPTransmitter    : 低延迟 UDP 异步发送器 (JSON/binary 双模式)

UDP 发送器运行在独立 daemon 线程中，不阻塞主循环。
通过有界队列 (max 50) 实现背压控制，队列满时丢弃旧帧避免累积延迟。
"""

import json
import socket
import struct
import threading
import time
import queue
from collections import deque
from typing import Optional, Dict, Deque

from config import (
    UDP_MAX_QUEUE_SIZE,
    UDP_BUFFER_SIZE,
    UDP_SEND_TIMEOUT,
)


# ===================== JSON 输出器 =====================
class JSONOutputHandler:
    """
    JSON 数据输出器 — 支持控制台/文件/双模式输出

    输出格式: JSON Lines (每行一个完整的 JSON 对象)，便于流式解析。

    四种输出模式:
        "none"    : 不输出 (默认)
        "console" : 仅输出到控制台 (stdout)
        "file"    : 仅输出到文件 (追加模式)
        "both"    : 同时输出到控制台和文件

    Attributes:
        mode:         当前输出模式
        filepath:     输出文件路径
        file_handle:  文件句柄 (file/both 模式时有效)

    Example:
        >>> handler = JSONOutputHandler("file", "output.jsonl")
        >>> handler.output({"type": "pose", "x": 1.5, "z": 3.2})
        >>> handler.close()
    """

    def __init__(self, mode: str, filepath: str):
        """
        初始化 JSON 输出器

        Args:
            mode:     输出模式 ("none" | "console" | "file" | "both")
            filepath: 输出文件路径 (仅在 "file" 或 "both" 模式下使用)
        """
        self.mode = mode
        self.filepath = filepath
        self.file_handle: Optional[object] = None

        if mode in ["file", "both"]:
            try:
                self.file_handle = open(filepath, 'a', encoding='utf-8')
                print(f"✅ JSON 输出已启用: {filepath}")
            except Exception as e:
                print(f"⚠️ JSON 文件打开失败: {e}，切换到控制台模式")
                self.mode = "console" if mode == "both" else "none"

    def output(self, data: dict) -> None:
        """
        输出一条 JSON 数据

        根据当前模式决定输出到控制台和/或文件。
        JSON 使用紧凑格式 (无空格)，每行以换行符结束。

        Args:
            data: 要输出的字典数据
        """
        json_str = json.dumps(data, ensure_ascii=False, separators=(',', ':'))

        if self.mode in ["console", "both"]:
            print(json_str)

        if self.mode in ["file", "both"] and self.file_handle:
            try:
                self.file_handle.write(json_str + '\n')
                self.file_handle.flush()
            except Exception as e:
                print(f"⚠️ JSON 写入失败: {e}")

    def close(self) -> None:
        """关闭文件句柄，释放资源"""
        if self.file_handle:
            self.file_handle.close()


# ===================== 低延迟 UDP 发送器 =====================
class LowLatencyUDPTransmitter:
    """
    低延迟 UDP 异步发送器 — 支持 JSON/二进制双模式

    核心设计:
        - 独立 daemon 发送线程: 不阻塞主循环，保障最大检测帧率
        - 有界队列 (50): 队列满时丢弃旧帧，避免累积延迟
        - 1ms 队列超时: 线程快速轮询，保障低延迟
        - 2MB 发送缓冲区: 减少操作系统层丢包

    双模式:
        - JSON 模式: {"type":"robot_position","pos":[x,y,z],"euler":[0,0,yaw]}
        - binary 模式: seq(I) + x,y,z,yaw(4f) + timestamp(d) = 28 bytes

    实时统计 (通过 get_stats() 获取):
        udp_fps         : 每秒实际发送次数
        success_rate    : 发送成功率 (%)
        queue_size      : 当前队列深度
        avg_latency_ms  : 平均排队+发送延迟 (毫秒)

    Example:
        >>> tx = LowLatencyUDPTransmitter("192.168.1.105", 9005, mode="json")
        >>> tx.start()
        >>> tx.send(1.5, 0.0, 3.2, 45.0)
        >>> stats = tx.get_stats()
        >>> tx.stop()
    """

    def __init__(self, ip: str, port: int, mode: str = "json"):
        """
        初始化 UDP 发送器

        Args:
            ip:   目标 IP 地址
            port: 目标端口号
            mode: 发送格式 "json" (默认) 或 "binary"
        """
        self._ip = ip
        self._port = port
        self._mode = mode
        self._queue: queue.Queue = queue.Queue(maxsize=UDP_MAX_QUEUE_SIZE)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._socket: Optional[socket.socket] = None
        self._seq = 0
        self._seq_lock = threading.Lock()

        # 统计变量
        self._sent_count = 0
        self._failed_count = 0
        self._last_stats_time = time.time()
        self._current_fps = 0.0
        self._success_rate = 100.0
        self._avg_latency = 0.0
        self._latency_samples: Deque[float] = deque(maxlen=100)
        self._lock = threading.Lock()

    def start(self) -> bool:
        """
        启动 UDP 发送线程

        创建 socket，设置缓冲区大小和超时，启动 daemon 线程。

        Returns:
            True=启动成功, False=启动失败
        """
        if self._running:
            return True
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, UDP_BUFFER_SIZE)
            self._socket.settimeout(0.001)

            self._running = True
            self._thread = threading.Thread(target=self._send_loop, daemon=True)
            self._thread.start()
            print(
                f"✅ UDP 发送器启动: {self._ip}:{self._port} | "
                f"模式:{self._mode} | 队列:{UDP_MAX_QUEUE_SIZE} | "
                f"缓冲:{UDP_BUFFER_SIZE // 1024}KB"
            )
            return True
        except Exception as e:
            print(f"⚠️ UDP 发送器启动失败: {e}")
            return False

    def stop(self) -> None:
        """
        停止 UDP 发送线程

        清空队列，发送终止信号 (None)，等待线程退出，关闭 socket。
        """
        self._running = False
        if self._queue:
            try:
                while True:
                    self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._socket:
            self._socket.close()
        print("ℹ️ UDP 发送器已停止")

    def send(self, x: float, y: float, z: float, yaw: float) -> bool:
        """
        非阻塞入队 — 立即返回，不阻塞主循环

        将位姿数据放入发送队列，由独立线程异步发送。
        队列满时直接丢弃数据 (不阻塞)。

        Args:
            x:   X 坐标 [米]
            y:   高度 [米] (通常为 CAR_HEIGHT)
            z:   Z 坐标 [米] (Y→Z 交换)
            yaw: 偏航角 [度]

        Returns:
            True=入队成功, False=队列满或发送器未运行
        """
        if not self._running:
            return False
        try:
            self._queue.put_nowait((x, y, z, yaw, time.time()))
            return True
        except queue.Full:
            with self._lock:
                self._failed_count += 1
            return False

    def get_stats(self) -> Dict[str, float]:
        """
        获取实时统计信息

        Returns:
            包含 udp_fps, success_rate, queue_size, avg_latency_ms 的字典

        Note:
            线程安全 (通过内部锁保护统计变量)
        """
        with self._lock:
            return {
                "udp_fps": self._current_fps,
                "success_rate": self._success_rate,
                "queue_size": self._queue.qsize(),
                "avg_latency_ms": self._avg_latency * 1000,
            }

    # ---------- 内部方法 ----------

    def _send_loop(self) -> None:
        """
        UDP 发送主循环 — 运行在独立 daemon 线程中

        从队列取出数据，调用 _send_packet 发送。
        """
        while self._running:
            try:
                item = self._queue.get(timeout=UDP_SEND_TIMEOUT)
                if item is None:
                    break
                x, y, z, yaw, send_time = item
                self._send_packet(x, y, z, yaw, send_time)
            except queue.Empty:
                continue
            except Exception:
                pass

    def _send_packet(
        self, x: float, y: float, z: float, yaw: float, send_time: float
    ) -> None:
        """
        JSON/binary 双模式发送单个数据包

        JSON 模式: 输出紧凑的 robot_position 格式
        binary 模式: 输出 28 字节紧凑包 (seq + 4float + timestamp)

        Args:
            x, y, z, yaw: 位姿数据
            send_time:     数据入队时的系统时间 (用于计算发送延迟)
        """
        try:
            with self._seq_lock:
                self._seq += 1
                seq = self._seq

            if self._mode == "json":
                # JSON 格式 — 精简扁平结构
                pose_data = {
                    "type": "robot_position",
                    "pos": [round(x, 4), round(y, 4), round(z, 4)],
                    "euler": [0.0, 0.0, round(yaw, 4)],
                }
                data = json.dumps(pose_data, ensure_ascii=False).encode("utf-8")
            else:
                # 精简 binary: seq(I) + x,y,z,yaw(4f) + timestamp(d) = 28 bytes
                data = struct.pack('<Iffffd', seq, x, y, z, yaw, time.time())

            self._socket.sendto(data, (self._ip, self._port))

            latency = time.time() - send_time
            self._latency_samples.append(latency)
            with self._lock:
                self._sent_count += 1
                self._avg_latency = (
                    sum(self._latency_samples) / len(self._latency_samples)
                )
                self._update_stats()
        except OSError:
            with self._lock:
                self._failed_count += 1
                self._update_stats()

    def _update_stats(self) -> None:
        """
        每秒更新一次统计信息 (线程安全)

        计算: udp_fps = 每秒发送帧数, success_rate = 成功率百分比
        统计变量每秒清零重新计数。
        """
        now = time.time()
        elapsed = now - self._last_stats_time
        if elapsed >= 1.0:
            self._current_fps = (
                self._sent_count / elapsed if elapsed > 0 else 0.0
            )
            total = self._sent_count + self._failed_count
            self._success_rate = (
                (self._sent_count / total * 100) if total > 0 else 100.0
            )
            self._sent_count = 0
            self._failed_count = 0
            self._last_stats_time = now
