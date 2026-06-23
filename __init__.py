"""
newtest — AprilTag 定位调试工具 模块化版本

模块结构:
    config.py       系统常量配置 (场地尺寸、相机内参、Tag 布局、UDP/JSON 参数)
    datatypes.py    数据结构定义 (ReferenceTagInfo, TargetObservation, TargetPose)
    utils.py        数学工具与坐标转换 (角度归一化、单应矩阵映射、世界坐标变换)
    comms.py        通信模块 (LowLatencyUDPTransmitter, JSONOutputHandler)
    pipeline.py     核心定位流水线 (KalmanFilter2D, process_markers, compute_homography, estimate_target_pose)
    visualizer.py   可视化绘制 (三窗口 + 坐标轴)
    main.py         主程序入口 (参数解析、摄像头初始化、主循环)

运行方式:
    python main.py [--camera 0] [--udp-ip 192.168.1.105] [--udp-port 9005] [--target-tag-id 7]
"""
