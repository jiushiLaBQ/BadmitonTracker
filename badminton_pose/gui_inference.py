# -*- coding: utf-8 -*-
"""
羽毛球运动姿态识别 GUI
左图：视频 + 骨架 + 球轨迹 + 脚步热力叠加
右图：球场俯视图 + 球落点热力图 + 球员步伐图
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
    QLabel, QPushButton, QFileDialog, QGroupBox,
    QTextEdit, QSplitter
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from modules.keypoint_extractor import KeypointExtractor
from modules.preprocessor import Preprocessor
from modules.ball_detector import BallDetector, CentroidTracker
from modules.court_detector import CourtDetector
from modules.court_mapper import CourtMapper, draw_court_diagram
from modules.heatmap_generator import HeatmapGenerator

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


class InferenceWorker(QThread):
    """推理工作线程"""
    frame_ready = pyqtSignal(np.ndarray, str, float, list)
    ball_heatmap_ready = pyqtSignal(np.ndarray)
    footwork_heatmap_ready = pyqtSignal(np.ndarray)
    status_update = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.cap = None
        self.mode = "video"

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 动作识别模型
        model_path = os.path.join(config.MODELS_DIR, 'best_model.pth')
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

        # 推理组件
        self.kpt_extractor = KeypointExtractor()
        self.preprocessor = Preprocessor()
        self.ball_detector = BallDetector()
        self.tracker = CentroidTracker()

        # 球场检测 + 映射
        self.court_detector = CourtDetector()
        self.court_mapper = CourtMapper()
        self.court_detected = False
        self.manual_corners = None

        # 热力图
        self.ball_heatmap = HeatmapGenerator()
        self.footwork_heatmap = HeatmapGenerator()

        # 脚步像素位置（用于视频上叠加）
        self.foot_pixel_positions = []

        # 时序
        self.frame_buffer = deque(maxlen=config.SEQ_LENGTH)
        self.prediction_history = deque(maxlen=5)
        self.ball_court_history = deque(maxlen=10)

    def set_source(self, source, mode="video"):
        self.source = source
        self.mode = mode
        self.frame_buffer.clear()
        self.prediction_history.clear()
        self.ball_court_history.clear()
        self.foot_pixel_positions.clear()
        self.court_detected = False
        self.manual_corners = None
        self.tracker = CentroidTracker()
        self.ball_heatmap.reset()
        self.footwork_heatmap.reset()

    def set_manual_corners(self, corners):
        """用户手动标定球场4角"""
        self.manual_corners = corners
        H_warp, H_court = self.court_detector.compute_homography(corners)
        self.court_mapper.set_homography(H_court)
        self.court_detected = True
        self.status_update.emit("球场手动标定完成，映射已更新")

    def run(self):
        self.running = True
        if self.mode == "camera":
            self.cap = cv2.VideoCapture(int(self.source))
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            self.status_update.emit(f"无法打开视频源: {self.source}")
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = max(1, int(fps / 15))
        frame_count = 0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if self.mode == "video":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            frame_count += 1
            if frame_count % frame_interval != 0:
                continue

            self._process_frame(frame, frame_count)

        if self.cap:
            self.cap.release()

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()

    def _process_frame(self, frame, frame_count):
        h, w = frame.shape[:2]
        annotated = frame.copy()

        # 1. 球场检测（仅自动检测，手动标定优先）
        if not self.court_detected:
            result = self.court_detector.detect_court(frame)
            self.court_mapper.set_homography(result['homography_court'])
            self.court_detected = True
            if result.get('fallback'):
                self.status_update.emit("球场自动检测失败，点击 [手动标定球场] 提升精度")
            else:
                self.status_update.emit("球场自动检测成功")

        # 2. 姿态估计
        kpts = self.kpt_extractor._detect_pose(frame)
        if kpts is not None:
            self._draw_skeleton(annotated, kpts)
            self.frame_buffer.append(kpts)

            # 脚踝位置（像素坐标，用于视频叠加）
            for ankle_idx in [15, 16]:
                conf = kpts[ankle_idx, 2]
                ax, ay = int(kpts[ankle_idx, 0]), int(kpts[ankle_idx, 1])
                if ax > 0 and ay > 0 and conf > 0.5:
                    self.foot_pixel_positions.append((ax, ay))

            # 脚踝映射到球场（俯视图步伐）
            if self.manual_corners is not None:  # 只在手动标定后才映射
                for ankle_idx in [15, 16]:
                    conf = kpts[ankle_idx, 2]
                    ax, ay = int(kpts[ankle_idx, 0]), int(kpts[ankle_idx, 1])
                    if ax > 0 and ay > 0 and conf > 0.6:
                        try:
                            court_pos = self.court_mapper.pixel_to_court(np.array([ax, ay]))
                            if court_pos is not None and len(court_pos) > 0:
                                cp = court_pos[0]
                                if self.court_mapper.is_in_court(cp):
                                    self.footwork_heatmap.add_point(cp)
                        except Exception:
                            pass

        # 3. 球检测 + 跟踪
        ball_det = self.ball_detector.detect(frame, person_keypoints=kpts)
        ball_pos = self.tracker.update(ball_det, frame_count)

        # 视频上画球标记（用跟踪位置，含卡尔曼预测）
        if ball_pos is not None:
            bx, by = int(ball_pos[0]), int(ball_pos[1])
            cv2.circle(annotated, (bx, by), 8, (0, 0, 255), -1)
            cv2.circle(annotated, (bx, by), 12, (0, 255, 255), 2)

            # 视频上的像素轨迹线
            traj = self.tracker.get_active_trajectory()
            if len(traj) >= 2:
                pts = np.array([(int(t[0]), int(t[1])) for t in traj[-30:]], dtype=np.int32)
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    color = (0, int(255 * (1 - alpha)), int(255 * alpha))
                    cv2.line(annotated, tuple(pts[i - 1]), tuple(pts[i]), color, 2)

        # 球场热力：真检测时累积，静态目标过滤
        if ball_det is not None and self.manual_corners is not None and ball_pos is not None:
            try:
                court_pos = self.court_mapper.pixel_to_court(ball_pos)
                if court_pos is not None and len(court_pos) > 0:
                    cp = court_pos[0]
                    if self.court_mapper.is_in_court(cp):
                        is_moving = True
                        if len(self.ball_court_history) >= 3:
                            recent = np.array(list(self.ball_court_history)[-5:])
                            spread = np.sqrt(((recent - recent.mean(axis=0))**2).sum(axis=1)).max()
                            if spread < 0.5:
                                is_moving = False
                        if is_moving:
                            self.ball_heatmap.add_point(cp)
                        self.ball_court_history.append(cp)
            except Exception:
                pass

        # 4. 在视频上叠加脚步热力图（像素坐标，不需要球场映射）
        foot_overlay = self._draw_footwork_on_video(annotated)
        if foot_overlay is not None:
            annotated = foot_overlay

        # 5. 动作识别
        label = "Collecting..."
        confidence = 0.0
        all_probs = []
        if len(self.frame_buffer) >= config.SEQ_LENGTH:
            seq = np.array(list(self.frame_buffer))
            features = self.preprocessor.process_sequence(seq)
            if features.shape[0] >= config.SEQ_LENGTH:
                feat_seq = features[-config.SEQ_LENGTH:]
                with torch.no_grad():
                    x = torch.FloatTensor(feat_seq).unsqueeze(0).to(self.device)
                    logits = self.model(x)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = np.argmax(probs)
                confidence = probs[pred_idx]
                all_probs = probs.tolist()
                self.prediction_history.append(pred_idx)
                if len(self.prediction_history) >= 3:
                    vote = Counter(self.prediction_history).most_common(1)[0]
                    pred_idx = vote[0]
                    confidence = max(confidence, probs[pred_idx])
                label = self.class_names[pred_idx]

        self._draw_label(annotated, label, confidence)

        # 6. 发射信号
        self.frame_ready.emit(annotated, label, confidence, all_probs)

        ball_img = self.ball_heatmap.generate_heatmap()
        self.ball_heatmap_ready.emit(ball_img)

        footwork_img = self.footwork_heatmap.generate_heatmap()
        self.footwork_heatmap_ready.emit(footwork_img)

    def _draw_footwork_on_video(self, frame):
        """在视频帧上叠加脚步密度热力图"""
        if len(self.foot_pixel_positions) < 5:
            return None

        h, w = frame.shape[:2]
        # 用最近200个点
        recent = self.foot_pixel_positions[-200:]

        # 生成密度图
        heatmap = np.zeros((h, w), dtype=np.float32)
        for (px, py) in recent:
            if 0 <= px < w and 0 <= py < h:
                heatmap[py, px] += 1.0

        # 高斯模糊
        kernel_size = max(h, w) // 20
        if kernel_size % 2 == 0:
            kernel_size += 1
        heatmap = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), 0)

        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)

            # 只在有热力的区域叠加
            mask = heatmap > 10
            overlay = frame.copy()
            overlay[mask] = cv2.addWeighted(frame, 0.5, heatmap_color, 0.5, 0)[mask]
            return overlay

        return None

    def _draw_skeleton(self, frame, keypoints):
        for i in range(17):
            x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
            if x == 0 and y == 0:
                continue
            color = (0, 255, 0) if keypoints[i, 2] > 0.7 else (0, 255, 255)
            cv2.circle(frame, (x, y), 5, color, -1)
        for idx, (p1, p2) in enumerate(SKELETON_CONNECTIONS):
            x1, y1 = int(keypoints[p1, 0]), int(keypoints[p1, 1])
            x2, y2 = int(keypoints[p2, 0]), int(keypoints[p2, 1])
            if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
                continue
            cv2.line(frame, (x1, y1), (x2, y2), SKELETON_COLORS[idx % len(SKELETON_COLORS)], 3)

    def _draw_label(self, frame, label, confidence):
        text = f"{label}: {confidence:.1%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.2
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


class BadmintonGUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("羽毛球运动姿态识别与动作分类系统")
        self.setGeometry(50, 50, 1400, 900)
        self.worker = None
        self.calibrating = False
        self.calib_corners = []
        self._init_ui()
        self._apply_style()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        title = QLabel("羽毛球运动姿态识别与动作分类系统")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #2c3e50; padding: 10px; background: #ecf0f1; border-radius: 8px;")
        main_layout.addWidget(title)

        # 工具栏
        toolbar = QHBoxLayout()
        btn_video = QPushButton("加载视频")
        btn_video.clicked.connect(self.load_video)
        toolbar.addWidget(btn_video)
        btn_camera = QPushButton("摄像头")
        btn_camera.clicked.connect(self.toggle_camera)
        self.btn_camera = btn_camera
        toolbar.addWidget(btn_camera)
        btn_calib = QPushButton("手动标定球场")
        btn_calib.clicked.connect(self.start_calibration)
        toolbar.addWidget(btn_calib)
        self.calib_label = QLabel("")
        self.calib_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        toolbar.addWidget(self.calib_label)
        btn_flip_h = QPushButton("左右翻转")
        btn_flip_h.setCheckable(True)
        btn_flip_h.clicked.connect(self.toggle_mirror_x)
        toolbar.addWidget(btn_flip_h)
        btn_flip_v = QPushButton("上下翻转")
        btn_flip_v.setCheckable(True)
        btn_flip_v.clicked.connect(self.toggle_mirror_y)
        toolbar.addWidget(btn_flip_v)
        btn_stop = QPushButton("停止")
        btn_stop.clicked.connect(self.stop_inference)
        toolbar.addWidget(btn_stop)
        btn_reset = QPushButton("重置热力图")
        btn_reset.clicked.connect(self.reset_heatmaps)
        toolbar.addWidget(btn_reset)
        toolbar.addStretch()
        main_layout.addLayout(toolbar)

        # 主内容
        content = QSplitter(Qt.Horizontal)
        main_layout.addWidget(content, 1)

        # 左：视频
        self.video_label = ClickableLabel("请选择视频开始\n\n提示: 点击 [手动标定球场] 后\n依次点击球场4个角点")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(720, 540)
        self.video_label.setStyleSheet("background: #1a1a2e; border: 2px solid #3498db; border-radius: 8px; color: #ecf0f1;")
        self.video_label.click_callback = self._on_video_click
        content.addWidget(self.video_label)

        # 右
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(8)

        # 球落点热力图
        ball_group = QGroupBox("羽毛球落点热力图（需手动标定球场）")
        ball_layout = QVBoxLayout(ball_group)
        self.ball_heatmap_label = QLabel()
        self.ball_heatmap_label.setAlignment(Qt.AlignCenter)
        self.ball_heatmap_label.setMinimumSize(500, 220)
        self.ball_heatmap_label.setStyleSheet("background: #2c3e50; border-radius: 5px;")
        ball_layout.addWidget(self.ball_heatmap_label)
        right_layout.addWidget(ball_group)

        # 球员步伐热力图
        foot_group = QGroupBox("球员步伐落点图（需手动标定球场）")
        foot_layout = QVBoxLayout(foot_group)
        self.footwork_label = QLabel()
        self.footwork_label.setAlignment(Qt.AlignCenter)
        self.footwork_label.setMinimumSize(500, 220)
        self.footwork_label.setStyleSheet("background: #2c3e50; border-radius: 5px;")
        foot_layout.addWidget(self.footwork_label)
        right_layout.addWidget(foot_group)

        # 识别结果
        result_group = QGroupBox("动作识别")
        result_layout = QVBoxLayout(result_group)
        self.label_action = QLabel("动作类别: --")
        self.label_action.setFont(QFont("Arial", 14, QFont.Bold))
        self.label_action.setStyleSheet("color: #2c3e50; padding: 3px;")
        result_layout.addWidget(self.label_action)
        self.label_conf = QLabel("置信度: --")
        self.label_conf.setFont(QFont("Arial", 12))
        self.label_conf.setStyleSheet("color: #7f8c8d; padding: 3px;")
        result_layout.addWidget(self.label_conf)
        self.top5_label = QLabel("")
        self.top5_label.setFont(QFont("Arial", 9))
        self.top5_label.setWordWrap(True)
        self.top5_label.setStyleSheet("color: #2c3e50; padding: 3px;")
        result_layout.addWidget(self.top5_label)
        right_layout.addWidget(result_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(80)
        self.log_text.setFont(QFont("Consolas", 9))
        right_layout.addWidget(self.log_text)

        content.addWidget(right)
        content.setSizes([900, 500])

        # 初始化所有面板显示球场底图
        self._init_heatmap_displays()

        self._log("系统初始化完成")

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background: #f5f6fa; }
            QGroupBox { font-weight: bold; border: 2px solid #bdc3c7; border-radius: 5px; margin-top: 8px; padding-top: 12px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background: #3498db; color: white; border: none; padding: 8px 16px; border-radius: 5px; font-size: 12px; }
            QPushButton:hover { background: #2980b9; }
            QTextEdit { background: #2c3e50; color: #ecf0f1; border-radius: 5px; }
        """)

    def _log(self, msg):
        self.log_text.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video (*.mp4 *.avi *.mov *.mkv)")
        if not path:
            return
        self._log(f"加载视频: {path}")
        self._start_inference(path, "video")

    def toggle_camera(self):
        if self.worker and self.worker.isRunning():
            self.stop_inference()
            self.btn_camera.setText("摄像头")
        else:
            self.btn_camera.setText("关闭摄像头")
            self._start_inference("0", "camera")

    def toggle_mirror_x(self, checked):
        """左右翻转"""
        if self.worker:
            self.worker.court_mapper.mirror_x = checked
            self._log(f"球场左右翻转: {'开' if checked else '关'}")

    def toggle_mirror_y(self, checked):
        """上下翻转"""
        if self.worker:
            self.worker.court_mapper.mirror_y = checked
            self._log(f"球场上下翻转: {'开' if checked else '关'}")

    def start_calibration(self):
        """开始手动标定模式"""
        self.calibrating = True
        self.calib_corners = []
        self.calib_label.setText("请在视频上依次点击球场4个角点 (0/4)")
        self._log("标定模式: 请在视频上点击球场4个角点")

    def _on_video_click(self, x, y):
        """视频点击回调"""
        if not self.calibrating:
            return
        self.calib_corners.append([x, y])
        count = len(self.calib_corners)
        self.calib_label.setText(f"已标记 {count}/4 个角点")

        if count >= 4:
            self.calibrating = False
            corners = np.array(self.calib_corners, dtype=np.float32)
            if self.worker:
                self.worker.set_manual_corners(corners)
            self.calib_label.setText("标定完成!")
            self._log(f"球场标定完成: {self.calib_corners}")

    def _start_inference(self, source, mode):
        self.stop_inference()
        self.worker = InferenceWorker(self)
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.ball_heatmap_ready.connect(self._on_ball_heatmap)
        self.worker.footwork_heatmap_ready.connect(self._on_footwork_heatmap)
        self.worker.status_update.connect(self._log)
        self.worker.set_source(source, mode)
        self.worker.start()

    def stop_inference(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None

    def reset_heatmaps(self):
        if self.worker:
            self.worker.ball_heatmap.reset()
            self.worker.footwork_heatmap.reset()
            self.worker.ball_court_history.clear()
            self.worker.foot_pixel_positions.clear()
        self._init_heatmap_displays()
        self._log("热力图已重置")

    def _on_frame_ready(self, frame, label, confidence, all_probs):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        self.video_label.set_frame_size(w, h)
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        if label:
            self.label_action.setText(f"动作类别: {label}")
            self.label_conf.setText(f"置信度: {confidence:.2%}")
            if confidence > 0.8:
                color = "#27ae60"
            elif confidence > 0.5:
                color = "#f39c12"
            else:
                color = "#e74c3c"
            self.label_action.setStyleSheet(f"color: {color}; padding: 3px;")
            if all_probs and self.worker:
                top5_idx = np.argsort(all_probs)[-5:][::-1]
                top5_text = "\n".join([f"  {self.worker.class_names[i]}: {all_probs[i]:.2%}" for i in top5_idx])
                self.top5_label.setText(f"Top-5:\n{top5_text}")

    def _on_ball_heatmap(self, img):
        self._show_court(img, self.ball_heatmap_label)

    def _on_footwork_heatmap(self, img):
        self._show_court(img, self.footwork_label)

    def _init_heatmap_displays(self):
        """初始化右侧球场底图"""
        court_img = draw_court_diagram(config.BIRDEYE_WIDTH, config.BIRDEYE_HEIGHT)
        self._show_court(court_img, self.ball_heatmap_label)
        self._show_court(court_img, self.footwork_label)

    def _show_court(self, img, label):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, event):
        self.stop_inference()
        event.accept()


class ClickableLabel(QLabel):
    """支持鼠标点击的QLabel，自动将点击坐标映射回原始帧"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.click_callback = None
        self.frame_w = 0   # 原始帧宽度
        self.frame_h = 0   # 原始帧高度

    def set_frame_size(self, w, h):
        """更新原始帧尺寸（每帧处理时调用）"""
        self.frame_w = w
        self.frame_h = h

    def mousePressEvent(self, event):
        if self.click_callback and self.pixmap() and self.frame_w > 0:
            label_w = self.width()
            label_h = self.height()
            pixmap = self.pixmap()
            pm_w = pixmap.width()
            pm_h = pixmap.height()

            # pixmap在label中居中显示
            offset_x = (label_w - pm_w) / 2
            offset_y = (label_h - pm_h) / 2

            click_x = event.x() - offset_x
            click_y = event.y() - offset_y

            if 0 <= click_x < pm_w and 0 <= click_y < pm_h:
                # 缩放回原始帧坐标
                orig_x = click_x * self.frame_w / pm_w
                orig_y = click_y * self.frame_h / pm_h
                self.click_callback(int(orig_x), int(orig_y))


def main():
    config.load_class_names()
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = BadmintonGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
