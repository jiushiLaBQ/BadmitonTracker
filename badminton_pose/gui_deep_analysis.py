# -*- coding: utf-8 -*-
"""
羽毛球深度视频分析系统 — 一体化 GUI
4 面板布局：
  左：原始视频 + 骨架 + 球轨迹叠加
  右上：标准球场俯视图 + 落点标记
  右中：落点热力图
  右下：动作识别结果 + 置信度
支持本地视频、摄像头输入
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
from collections import deque, Counter

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QGroupBox, QGridLayout,
    QProgressBar, QTextEdit, QSplitter, QFrame, QComboBox
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from modules.keypoint_extractor import KeypointExtractor
from modules.preprocessor import Preprocessor
from modules.model import build_model
from modules.court_detector import CourtDetector, manual_court_selection
from modules.court_mapper import CourtMapper, draw_court_diagram
from modules.ball_detector import BallDetector, CentroidTracker, TrajectoryFitter
from modules.heatmap_generator import HeatmapGenerator

# COCO 骨架连线
SKELETON_CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
]

SKELETON_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
]


class DeepAnalysisWorker(QThread):
    """
    深度分析工作线程
    并行执行：球场检测、姿态估计、球检测跟踪、轨迹拟合、热力图、动作识别
    """
    # 信号：原始帧、俯视图、热力图、动作结果、状态日志、统计信息
    frame_ready = pyqtSignal(np.ndarray)
    birdeye_ready = pyqtSignal(np.ndarray)
    heatmap_ready = pyqtSignal(np.ndarray)
    result_ready = pyqtSignal(str, float, list)
    status_update = pyqtSignal(str)
    stats_ready = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.cap = None
        self.mode = "video"
        self.court_detected = False
        self.H_warp = None
        self.manual_corners = None

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载 BiLSTM 模型
        model_path = os.path.join(config.MODELS_DIR, 'best_model.pth')
        if not os.path.exists(model_path):
            self.status_update.emit(f"模型文件不存在: {model_path}")
            self.model = None
            self.class_names = []
            self.num_classes = 0
            self.scaler = None
        else:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            from modules.model import BiLSTMClassifier
            saved_cfg = checkpoint.get('config', {})
            self.model = BiLSTMClassifier(
                input_dim=saved_cfg.get('input_dim', config.TOTAL_FEATURES),
                hidden_size=saved_cfg.get('hidden_size', config.LSTM_HIDDEN_SIZE),
                num_layers=saved_cfg.get('num_layers', config.LSTM_NUM_LAYERS),
                num_classes=checkpoint.get('num_classes', config.NUM_CLASSES),
                dropout=saved_cfg.get('dropout', config.DROPOUT_RATE),
                fc_hidden=saved_cfg.get('fc_hidden', config.FC_HIDDEN_SIZE),
            ).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.class_names = checkpoint['class_names']
            self.num_classes = checkpoint['num_classes']

            # 加载StandardScaler参数
            scaler_mean = checkpoint.get('scaler_mean', None)
            scaler_scale = checkpoint.get('scaler_scale', None)
            if scaler_mean is not None and scaler_scale is not None:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                self.scaler.mean_ = scaler_mean
                self.scaler.scale_ = scaler_scale
                self.scaler.n_features_in_ = len(scaler_mean)
            else:
                self.scaler = None

            self.status_update.emit(f"模型加载完成 | 设备: {self.device} | 类别数: {self.num_classes}")

        # 推理组件
        self.kpt_extractor = KeypointExtractor()
        self.preprocessor = Preprocessor()

        # 球场检测
        self.court_detector = CourtDetector()
        self.court_mapper = CourtMapper()

        # 球检测 + 跟踪
        self.ball_detector = BallDetector()
        self.tracker = CentroidTracker()
        self.trajectory_fitter = TrajectoryFitter()

        # 热力图
        self.heatmap_gen = HeatmapGenerator()

        # 时序缓冲
        self.frame_buffer = deque(maxlen=config.SEQ_LENGTH)
        self.prediction_history = deque(maxlen=5)

        # 当前预测类别索引（用于热力图分类）
        self.current_class_idx = None
        self._last_trajectory_id = -1

    def set_source(self, source, mode="video"):
        """设置输入源"""
        self.source = source
        self.mode = mode
        self.frame_buffer.clear()
        self.prediction_history.clear()
        self.court_detected = False
        self.current_class_idx = None
        self._last_trajectory_id = -1
        self.manual_corners = None
        # 重置跟踪器
        self.tracker = CentroidTracker()

    def run(self):
        """主处理循环"""
        self.running = True

        if self.mode == "camera":
            self.cap = cv2.VideoCapture(int(self.source))
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            self.status_update.emit(f"无法打开视频源: {self.source}")
            return

        frame_count = 0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if self.mode == "video":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            frame_count += 1
            self._process_frame(frame, frame_count)

        if self.cap:
            self.cap.release()

    def stop(self):
        """停止处理"""
        self.running = False
        if self.cap is not None:
            self.cap.release()

    def set_manual_corners(self, corners):
        """
        接收手动标定的角点，计算透视变换
        用户需按 左上→右上→右下→左下 顺序点击

        Args:
            corners: np.ndarray (4,2) 用户点击的角点
        """
        H_warp, H_court = self.court_detector.compute_homography(corners)
        self.H_warp = H_warp
        self.manual_corners = corners
        self.court_mapper.set_homography(H_court)
        self.court_detected = True
        # 调试：验证角点映射
        labels = ['TL', 'TR', 'BR', 'BL']
        expected = [[0, 0], [13.4, 0], [13.4, 6.1], [0, 6.1]]
        for i, (c, lbl, exp) in enumerate(zip(corners, labels, expected)):
            court_pt = self.court_mapper.pixel_to_court(c.reshape(1, 2))[0]
            print(f"  Corner {i} ({lbl}): pixel=({c[0]:.0f}, {c[1]:.0f}) -> court=({court_pt[0]:.2f}, {court_pt[1]:.2f})  expected=({exp[0]}, {exp[1]})")
        self.status_update.emit("手动球场标定完成")

    def reset_heatmap(self):
        """重置热力图数据"""
        self.heatmap_gen.reset()
        self.status_update.emit("热力图数据已重置")

    def _process_frame(self, frame, frame_count):
        """
        处理单帧：球场检测 → 骨架 → 球检测 → 轨迹 → 热力图 → 动作识别

        Args:
            frame: BGR 图像
            frame_count: 帧号
        """
        annotated = frame.copy()

        # ========== 1. 球场检测（仅首帧）==========
        if not self.court_detected:
            result = self.court_detector.detect_court(frame)
            if result['success']:
                self.H_warp = result['homography_warp']
                self.court_mapper.set_homography(result['homography_court'])
                self.court_detected = True
                if result.get('fallback', False):
                    self.status_update.emit("球场自动检测失败，使用默认映射（热力图可用，精度一般）")
                else:
                    self.status_update.emit("球场自动检测成功")

        # ========== 2. 姿态估计 + 骨架绘制 ==========
        kpts = self.kpt_extractor._detect_pose(frame)
        if kpts is not None:
            self._draw_skeleton(annotated, kpts)
            self.frame_buffer.append(kpts)

        # 画球场轮廓（标定后）
        if self.manual_corners is not None:
            pts = self.manual_corners.astype(np.int32)
            cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
            for i, pt in enumerate(pts):
                cv2.circle(annotated, tuple(pt), 6, (0, 0, 255), -1)
                cv2.putText(annotated, str(i + 1), (int(pt[0]) + 8, int(pt[1]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ========== 3. 球检测 + 跟踪 ==========
        ball_det = self.ball_detector.detect(frame, person_keypoints=kpts)
        ball_pos = self.tracker.update(ball_det, frame_count)

        # 在原始帧上绘制球位置
        if ball_pos is not None:
            bx, by = int(ball_pos[0]), int(ball_pos[1])
            cv2.circle(annotated, (bx, by), 8, (0, 0, 255), -1)
            cv2.circle(annotated, (bx, by), 12, (0, 255, 255), 2)

            # 绘制轨迹历史
            traj = self.tracker.get_active_trajectory()
            if len(traj) >= 2:
                pts = np.array([(int(t[0]), int(t[1])) for t in traj[-30:]], dtype=np.int32)
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    color = (0, int(255 * (1 - alpha)), int(255 * alpha))
                    cv2.line(annotated, tuple(pts[i - 1]), tuple(pts[i]), color, 2)

        # ========== 4. 轨迹拟合 + 落点预测 ==========
        landing_point = None
        traj_data = self.tracker.get_active_trajectory()
        if len(traj_data) >= config.TRAJECTORY_MIN_POINTS:
            fit_result = self.trajectory_fitter.fit(traj_data)
            if fit_result['valid']:
                h, w = frame.shape[:2]
                landing = self.trajectory_fitter.predict_landing(fit_result['coeffs'], h)
                if landing is not None:
                    landing_point = landing
                    lx, ly = int(landing[0]), int(landing[1])
                    # 在原始帧绘制预测落点
                    cv2.drawMarker(annotated, (lx, ly), (255, 0, 255),
                                   cv2.MARKER_TILTED_CROSS, 20, 2)

                    # 绘制拟合曲线
                    curve_pts = self.trajectory_fitter.get_predicted_positions(
                        fit_result['coeffs'],
                        fit_result['x_coords'].min(),
                        max(lx, fit_result['x_coords'].max()),
                        50
                    )
                    if curve_pts is not None:
                        for i in range(1, len(curve_pts)):
                            p1 = (int(curve_pts[i - 1, 0]), int(curve_pts[i - 1, 1]))
                            p2 = (int(curve_pts[i, 0]), int(curve_pts[i, 1]))
                            if 0 <= p1[0] < w and 0 <= p2[0] < w:
                                cv2.line(annotated, p1, p2, (255, 0, 255), 2)

        # ========== 5. 动作识别 ==========
        label = "Collecting..."
        confidence = 0.0
        all_probs = []

        if self.model is not None and len(self.frame_buffer) >= config.SEQ_LENGTH:
            seq = np.array(list(self.frame_buffer))
            features = self.preprocessor.process_sequence(seq)

            if features.shape[0] >= config.SEQ_LENGTH:
                feat_seq = features[-config.SEQ_LENGTH:]

                # 应用StandardScaler（训练时拟合，推理时必须transform）
                if self.scaler is not None:
                    feat_seq = self.scaler.transform(feat_seq)

                with torch.no_grad():
                    x = torch.FloatTensor(feat_seq).unsqueeze(0).to(self.device)
                    logits = self.model(x)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                pred_idx = np.argmax(probs)
                confidence = probs[pred_idx]
                all_probs = probs.tolist()

                # 5帧投票平滑
                self.prediction_history.append(pred_idx)
                if len(self.prediction_history) >= 3:
                    vote = Counter(self.prediction_history).most_common(1)[0]
                    pred_idx = vote[0]
                    confidence = max(confidence, probs[pred_idx])

                label = self.class_names[pred_idx]
                self.current_class_idx = pred_idx

        # 绘制动作标签
        self._draw_label(annotated, label, confidence)

        # ========== 6. 坐标映射 + 热力图 ==========
        birdeye_img = draw_court_diagram(config.BIRDEYE_WIDTH, config.BIRDEYE_HEIGHT)
        heatmap_img = draw_court_diagram(config.BIRDEYE_WIDTH, config.BIRDEYE_HEIGHT)
        stats = {}

        if self.court_detected:
            # 俯视图
            if self.H_warp is not None:
                birdeye_img = cv2.warpPerspective(
                    frame, self.H_warp,
                    (config.BIRDEYE_WIDTH, config.BIRDEYE_HEIGHT)
                )

            # 落点映射 + 热力图累积（轨迹结束后取末端落点）
            completed = self.tracker.get_new_completed_trajectories(self._last_trajectory_id)
            for traj_idx, traj_points in completed:
                self._last_trajectory_id = max(self._last_trajectory_id, traj_idx)
                if len(traj_points) >= 3:
                    tail = traj_points[-min(5, len(traj_points)):]
                    tail_positions = [(p[0], p[1]) for p in tail]
                    try:
                        court_pts = self.court_mapper.pixel_to_court(tail_positions)
                        valid = [pt for pt in court_pts if self.court_mapper.is_in_court(pt)]
                        if valid:
                            landing = np.mean(valid, axis=0)
                            self.heatmap_gen.add_point(tuple(landing), self.current_class_idx)

                            # 在俯视图上标记落点
                            px = int(landing[0] / config.COURT_LENGTH_M * config.BIRDEYE_WIDTH)
                            py = int(landing[1] / config.COURT_WIDTH_M * config.BIRDEYE_HEIGHT)
                            if 0 <= px < config.BIRDEYE_WIDTH and 0 <= py < config.BIRDEYE_HEIGHT:
                                cv2.circle(birdeye_img, (px, py), 5, (0, 0, 255), -1)
                    except Exception:
                        pass

            # 生成热力图
            heatmap_img = self.heatmap_gen.generate_heatmap()

            # 统计信息
            if len(self.heatmap_gen.all_points) > 0:
                stats = self.court_mapper.get_court_stats(
                    np.array(self.heatmap_gen.all_points)
                )

        # ========== 7. 发射信号 ==========
        self.frame_ready.emit(annotated)
        self.birdeye_ready.emit(birdeye_img)
        self.heatmap_ready.emit(heatmap_img)
        self.result_ready.emit(label, confidence, all_probs)
        if stats:
            self.stats_ready.emit(stats)

    def _draw_skeleton(self, frame, keypoints):
        """绘制骨骼骨架"""
        for i in range(17):
            x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
            if x == 0 and y == 0:
                continue
            conf = keypoints[i, 2]
            color = (0, 255, 0) if conf > 0.7 else (0, 255, 255)
            cv2.circle(frame, (x, y), 5, color, -1)

        for idx, (p1, p2) in enumerate(SKELETON_CONNECTIONS):
            x1, y1 = int(keypoints[p1, 0]), int(keypoints[p1, 1])
            x2, y2 = int(keypoints[p2, 0]), int(keypoints[p2, 1])
            if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
                continue
            color = SKELETON_COLORS[idx % len(SKELETON_COLORS)]
            cv2.line(frame, (x1, y1), (x2, y2), color, 3)

    def _draw_label(self, frame, label, confidence):
        """绘制动作标签"""
        text = f"{label}: {confidence:.1%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(frame, (10, 10), (tw + 30, th + 30), (0, 0, 0), -1)

        if confidence > 0.8:
            color = (0, 255, 0)
        elif confidence > 0.5:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        cv2.putText(frame, text, (20, th + 20), font, scale, color, thickness)


class DeepAnalysisGUI(QMainWindow):
    """
    羽毛球深度视频分析系统主界面
    4 面板布局：原始视频 | 俯视图 | 热力图 | 动作识别
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("羽毛球深度视频分析系统 — YOLOv8-Pose + BiLSTM")
        self.setGeometry(30, 30, config.DEEP_ANALYSIS_WINDOW_W, config.DEEP_ANALYSIS_WINDOW_H)

        self.worker = None
        self.current_frame = None  # 保存当前帧用于手动标定

        self._init_ui()
        self._apply_style()

    def _init_ui(self):
        """初始化界面布局"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # ========== 标题 ==========
        title = QLabel("羽毛球深度视频分析系统")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #2c3e50; padding: 8px; background: #ecf0f1; border-radius: 8px;")
        main_layout.addWidget(title)

        # ========== 工具栏 ==========
        toolbar = QHBoxLayout()

        btn_video = QPushButton("加载视频")
        btn_video.clicked.connect(self.load_video)
        toolbar.addWidget(btn_video)

        btn_camera = QPushButton("摄像头")
        btn_camera.clicked.connect(self.toggle_camera)
        self.btn_camera = btn_camera
        toolbar.addWidget(btn_camera)

        btn_court = QPushButton("手动标定球场")
        btn_court.clicked.connect(self.manual_court)
        toolbar.addWidget(btn_court)

        btn_stop = QPushButton("停止")
        btn_stop.clicked.connect(self.stop_processing)
        toolbar.addWidget(btn_stop)

        btn_reset = QPushButton("重置热力图")
        btn_reset.clicked.connect(self.reset_heatmap)
        toolbar.addWidget(btn_reset)

        # 热力图类别选择
        self.class_combo = QComboBox()
        self.class_combo.addItem("全局热力图", -1)
        self._populate_class_combo()
        self.class_combo.currentIndexChanged.connect(self._on_class_changed)
        toolbar.addWidget(self.class_combo)

        toolbar.addStretch()
        main_layout.addLayout(toolbar)

        # ========== 主内容区 ==========
        content = QSplitter(Qt.Horizontal)
        main_layout.addWidget(content, 1)

        # --- 左面板：原始视频 ---
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel("请选择输入源")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(720, 540)
        self.video_label.setStyleSheet(
            "background: #1a1a2e; border: 2px solid #3498db; border-radius: 8px; color: #ecf0f1;"
        )
        left_layout.addWidget(self.video_label)
        content.addWidget(left)

        # --- 右面板：俯视图 + 热力图 + 动作结果 ---
        right = QSplitter(Qt.Vertical)

        # 右上：俯视球场图
        birdeye_group = QGroupBox("俯视球场图")
        birdeye_layout = QVBoxLayout(birdeye_group)
        self.birdeye_label = QLabel()
        self.birdeye_label.setAlignment(Qt.AlignCenter)
        self.birdeye_label.setMinimumSize(500, 220)
        self.birdeye_label.setStyleSheet("background: #2c3e50; border-radius: 5px;")
        birdeye_layout.addWidget(self.birdeye_label)
        right.addWidget(birdeye_group)

        # 右中：热力图
        heatmap_group = QGroupBox("落点热力图")
        heatmap_layout = QVBoxLayout(heatmap_group)
        self.heatmap_label = QLabel()
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        self.heatmap_label.setMinimumSize(500, 220)
        self.heatmap_label.setStyleSheet("background: #2c3e50; border-radius: 5px;")
        heatmap_layout.addWidget(self.heatmap_label)
        right.addWidget(heatmap_group)

        # 右下：动作识别结果 + 统计
        result_group = QGroupBox("识别结果")
        result_layout = QVBoxLayout(result_group)

        self.label_action = QLabel("动作类别: --")
        self.label_action.setFont(QFont("Arial", 14, QFont.Bold))
        self.label_action.setStyleSheet("color: #2c3e50; padding: 3px;")
        result_layout.addWidget(self.label_action)

        self.label_conf = QLabel("置信度: --")
        self.label_conf.setFont(QFont("Arial", 12))
        self.label_conf.setStyleSheet("color: #7f8c8d; padding: 3px;")
        result_layout.addWidget(self.label_conf)

        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        self.conf_bar.setValue(0)
        self.conf_bar.setStyleSheet("""
            QProgressBar { border: 2px solid #bdc3c7; border-radius: 5px;
                           text-align: center; height: 20px; }
            QProgressBar::chunk { background-color: #27ae60; border-radius: 3px; }
        """)
        result_layout.addWidget(self.conf_bar)

        self.top5_label = QLabel("")
        self.top5_label.setFont(QFont("Arial", 9))
        self.top5_label.setWordWrap(True)
        self.top5_label.setStyleSheet("color: #2c3e50; padding: 3px;")
        result_layout.addWidget(self.top5_label)

        self.stats_label = QLabel("")
        self.stats_label.setFont(QFont("Arial", 9))
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("color: #2c3e50; padding: 3px;")
        result_layout.addWidget(self.stats_label)

        right.addWidget(result_group)
        right.setSizes([250, 250, 300])
        content.addWidget(right)

        content.setSizes([900, 600])

        # ========== 状态日志 ==========
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        self.log_text.setFont(QFont("Consolas", 9))
        main_layout.addWidget(self.log_text)

        self._log("系统初始化完成")
        model_path = os.path.join(config.MODELS_DIR, 'best_model.pth')
        if os.path.exists(model_path):
            self._log(f"模型: {model_path}")
        else:
            self._log(f"警告: 模型文件不存在 {model_path}，请先运行训练流水线")

    def _populate_class_combo(self):
        """填充类别下拉框"""
        try:
            config.load_class_names()
            if config.CLASS_NAMES:
                for i, name in enumerate(config.CLASS_NAMES):
                    self.class_combo.addItem(f"类别 {i}: {name}", i)
        except Exception:
            pass

    def _on_class_changed(self, index):
        """切换热力图类别"""
        class_idx = self.class_combo.currentData()
        if self.worker:
            if class_idx == -1:
                self._log("切换到全局热力图")
            else:
                self._log(f"切换到类别 {class_idx} 热力图")

    def _apply_style(self):
        """全局样式"""
        self.setStyleSheet("""
            QMainWindow { background: #f5f6fa; }
            QGroupBox { font-weight: bold; border: 2px solid #bdc3c7;
                        border-radius: 5px; margin-top: 10px; padding-top: 15px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background: #3498db; color: white; border: none;
                          padding: 8px 16px; border-radius: 5px; font-size: 12px; }
            QPushButton:hover { background: #2980b9; }
            QPushButton:pressed { background: #1a5276; }
            QTextEdit { background: #2c3e50; color: #ecf0f1; border-radius: 5px; }
        """)

    def _log(self, msg):
        """追加日志"""
        self.log_text.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def _cv2_to_qpixmap(self, img, target_size=None):
        """
        BGR numpy → QPixmap

        Args:
            img: BGR 图像
            target_size: (w, h) 缩放目标，None 则保持原尺寸

        Returns:
            QPixmap
        """
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        if target_size:
            pixmap = pixmap.scaled(target_size[0], target_size[1],
                                   Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return pixmap

    def load_video(self):
        """加载本地视频"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if not path:
            return
        self._log(f"加载视频: {path}")
        self._start_processing(path, "video")

    def toggle_camera(self):
        """切换摄像头"""
        if self.worker and self.worker.isRunning():
            self.stop_processing()
            self.btn_camera.setText("摄像头")
        else:
            self._log("开启摄像头...")
            self.btn_camera.setText("关闭摄像头")
            self._start_processing(str(config.DEEP_ANALYSIS_CAMERA_INDEX), "camera")

    def manual_court(self):
        """手动标定球场角点"""
        if self.current_frame is None:
            self._log("请先加载视频或开启摄像头")
            return

        self._log("请在弹出窗口中依次点击 4 个球场角点（ENTER 确认，R 重置）")
        corners = manual_court_selection(self.current_frame)
        if corners is not None and len(corners) == 4:
            if self.worker:
                self.worker.set_manual_corners(corners)
            self._log("手动球场标定已应用")

    def reset_heatmap(self):
        """重置热力图"""
        if self.worker:
            self.worker.reset_heatmap()
        self._log("热力图已重置")

    def _start_processing(self, source, mode):
        """启动处理线程"""
        self.stop_processing()

        self.worker = DeepAnalysisWorker(self)
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.birdeye_ready.connect(self._on_birdeye_ready)
        self.worker.heatmap_ready.connect(self._on_heatmap_ready)
        self.worker.result_ready.connect(self._on_result_ready)
        self.worker.status_update.connect(self._log)
        self.worker.stats_ready.connect(self._on_stats_ready)

        self.worker.set_source(source, mode)
        self.worker.start()

    def stop_processing(self):
        """停止处理"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None

    # ========== 信号槽：更新 UI ==========

    def _on_frame_ready(self, frame):
        """更新原始视频面板"""
        self.current_frame = frame.copy()
        pixmap = self._cv2_to_qpixmap(frame, (self.video_label.width(), self.video_label.height()))
        self.video_label.setPixmap(pixmap)

    def _on_birdeye_ready(self, image):
        """更新俯视图面板"""
        pixmap = self._cv2_to_qpixmap(image, (self.birdeye_label.width(), self.birdeye_label.height()))
        self.birdeye_label.setPixmap(pixmap)

    def _on_heatmap_ready(self, image):
        """更新热力图面板"""
        pixmap = self._cv2_to_qpixmap(image, (self.heatmap_label.width(), self.heatmap_label.height()))
        self.heatmap_label.setPixmap(pixmap)

    def _on_result_ready(self, label, confidence, all_probs):
        """更新识别结果"""
        if not label:
            return

        self.label_action.setText(f"动作类别: {label}")
        self.label_conf.setText(f"置信度: {confidence:.2%}")
        self.conf_bar.setValue(int(confidence * 100))

        # 颜色指示
        if confidence > 0.8:
            color = "#27ae60"
        elif confidence > 0.5:
            color = "#f39c12"
        else:
            color = "#e74c3c"
        self.label_action.setStyleSheet(f"color: {color}; padding: 3px;")

        # Top-5
        if all_probs and self.worker and hasattr(self.worker, 'class_names') and self.worker.class_names:
            top5_idx = np.argsort(all_probs)[-5:][::-1]
            top5_text = "  |  ".join([
                f"{self.worker.class_names[i]}: {all_probs[i]:.1%}"
                for i in top5_idx
            ])
            self.top5_label.setText(f"Top-5: {top5_text}")

    def _on_stats_ready(self, stats):
        """更新落点统计"""
        if not stats:
            return

        text = (
            f"总落点: {stats.get('total', 0)} | "
            f"前场: {stats.get('front_count', 0)} 后场: {stats.get('back_count', 0)} | "
            f"左半场: {stats.get('left_count', 0)} 右半场: {stats.get('right_count', 0)} | "
            f"场内: {stats.get('in_court_count', 0)}"
        )
        self.stats_label.setText(text)

    def closeEvent(self, event):
        """关闭窗口清理"""
        self.stop_processing()
        event.accept()


def main():
    """主入口"""
    config.load_class_names()

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = DeepAnalysisGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
