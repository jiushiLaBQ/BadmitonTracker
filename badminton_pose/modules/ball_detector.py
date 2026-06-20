# -*- coding: utf-8 -*-
"""
羽毛球检测与跟踪模块
混合检测策略：帧差运动检测 + 颜色筛选 + YOLO辅助
卡尔曼滤波器平滑跟踪
多项式拟合飞行轨迹并预测落点
"""

import cv2
import numpy as np
from collections import deque
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class BallDetector:
    """
    羽毛球检测器
    混合策略：帧差运动检测(主) + 亮色区域筛选 + YOLO检测(辅助)
    羽毛球特征：小、亮(白/黄)、快速移动
    """

    def __init__(self):
        self.prev_gray = None
        self.max_area = config.BALL_MAX_AREA
        # 尝试加载YOLO（可选辅助）
        self.yolo_model = None
        try:
            from ultralytics import YOLO
            custom_model = os.path.join(config.MODELS_DIR, "best_ball.pt")
            if os.path.exists(custom_model):
                self.yolo_model = YOLO(custom_model)
            else:
                self.yolo_model = YOLO(config.BALL_MODEL)
        except Exception:
            pass

    def detect(self, frame, person_keypoints=None):
        """
        检测单帧中的羽毛球位置

        Args:
            frame: BGR 图像
            person_keypoints: (17, 3) 人物关键点，用于排除人体区域

        Returns:
            np.ndarray (cx, cy) 像素坐标 或 None
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 创建人物掩膜（排除人头、身体区域）
        person_mask = self._build_person_mask(person_keypoints, h, w)

        candidates = []

        # ========== 方法1：帧差运动检测（最可靠）==========
        if self.prev_gray is not None:
            motion_candidates = self._motion_detect(self.prev_gray, gray, hsv, h, w, person_mask)
            candidates.extend(motion_candidates)
        self.prev_gray = gray

        # ========== 方法2：亮色小目标检测 ==========
        color_candidates = self._color_detect(frame, hsv, h, w, person_mask)
        candidates.extend(color_candidates)

        # ========== 方法3：YOLO辅助（低权重）==========
        if self.yolo_model is not None:
            yolo_candidates = self._yolo_detect(frame, hsv, h, w, person_mask)
            candidates.extend(yolo_candidates)

        if not candidates:
            return None

        # 去重：合并相近位置的候选
        merged = self._merge_candidates(candidates, dist_thresh=20)

        # 返回得分最高的
        merged.sort(key=lambda c: c[2], reverse=True)
        return np.array([merged[0][0], merged[0][1]], dtype=np.float32)

    def _build_person_mask(self, keypoints, h, w):
        """
        根据人物关键点创建掩膜，排除人体区域
        球不应出现在人体内部

        Args:
            keypoints: (17, 3) COCO关键点 或 None
            h, w: 图像尺寸

        Returns:
            np.ndarray (h, w) uint8 掩膜，255=人体区域
        """
        mask = np.zeros((h, w), dtype=np.uint8)

        if keypoints is None:
            return mask

        COCO_SKELETON = [
            (5, 7), (7, 9), (6, 8), (8, 10),  # 手臂
            (5, 6), (5, 11), (6, 12), (11, 12),  # 躯干
            (11, 13), (13, 15), (12, 14), (14, 16),  # 腿
            (0, 5), (0, 6),  # 头到肩
        ]

        # 围绕每个关键点画圆（覆盖身体区域）
        for i in range(17):
            x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
            if x == 0 and y == 0:
                continue
            # 头部区域用更大半径（人头容易被误检为球）
            radius = 40 if i in (0, 1, 2, 3, 4) else 25
            cv2.circle(mask, (x, y), radius, 255, -1)

        # 连线也画粗线覆盖
        for p1, p2 in COCO_SKELETON:
            x1, y1 = int(keypoints[p1, 0]), int(keypoints[p1, 1])
            x2, y2 = int(keypoints[p2, 0]), int(keypoints[p2, 1])
            if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
                continue
            cv2.line(mask, (x1, y1), (x2, y2), 255, 20)

        # 膨胀确保完全覆盖
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.dilate(mask, kernel)

        return mask

    def _motion_detect(self, prev_gray, gray, hsv, h, w, person_mask):
        """
        帧差法检测运动区域中的亮色小目标
        羽毛球移动速度快，帧间差异明显
        """
        candidates = []

        # 帧差
        diff = cv2.absdiff(prev_gray, gray)
        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # 形态学去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_DILATE, kernel)

        # 排除人物区域
        if person_mask is not None and person_mask.any():
            motion_mask = cv2.bitwise_and(motion_mask, cv2.bitwise_not(person_mask))

        # 亮色掩膜（白球或黄球）
        bright_mask = (
            ((hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 180)) |  # 白色
            ((hsv[:, :, 0] < 35) & (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 180))  # 黄色
        )

        # 运动 + 亮色 交集
        combined = cv2.bitwise_and(motion_mask, bright_mask.astype(np.uint8) * 255)

        # 查找连通区域
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 4 or area > self.max_area:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            cx = x + cw / 2.0
            cy = y + ch / 2.0
            # 小面积+运动 = 高置信度
            score = 3.0 / (area + 1)
            candidates.append((cx, cy, score, 'motion'))

        return candidates

    def _color_detect(self, frame, hsv, h, w, person_mask):
        """
        检测画面中亮色小区域（无需运动）
        适用于球静止或刚出现的情况
        """
        candidates = []

        # 白色或黄色小目标
        bright_mask = (
            ((hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 190)) |  # 白色
            ((hsv[:, :, 0] < 35) & (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 190))  # 黄色
        )

        bright_uint8 = bright_mask.astype(np.uint8) * 255
        # 去掉大面积区域（球场线、天花板等）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bright_uint8 = cv2.morphologyEx(bright_uint8, cv2.MORPH_OPEN, kernel)

        # 排除人物区域
        if person_mask is not None and person_mask.any():
            bright_uint8 = cv2.bitwise_and(bright_uint8, cv2.bitwise_not(person_mask))

        contours, _ = cv2.findContours(bright_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 4 or area > self.max_area:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            ratio = max(cw, ch) / (min(cw, ch) + 1e-6)
            if ratio > 3.0:
                continue
            cx = x + cw / 2.0
            cy = y + ch / 2.0
            score = 1.5 / (area + 1)
            candidates.append((cx, cy, score, 'color'))

        return candidates

    def _yolo_detect(self, frame, hsv, h, w, person_mask):
        """YOLO检测作为辅助信号，权重较低"""
        candidates = []
        try:
            results = self.yolo_model.predict(
                [frame], conf=config.BALL_CONF,
                iou=config.BALL_IOU, imgsz=config.BALL_IMG_SIZE,
                verbose=False
            )
            if len(results) == 0 or results[0].boxes is None:
                return candidates

            boxes = results[0].boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)
                area = (x2 - x1) * (y2 - y1)
                if area > self.max_area or area < 10:
                    continue
                cw = x2 - x1
                ch = y2 - y1
                if cw == 0 or ch == 0:
                    continue
                ratio = max(cw, ch) / min(cw, ch)
                if ratio > 3.0:
                    continue
                # 排除人物区域
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                if person_mask is not None and person_mask.any():
                    ix, iy = int(cx), int(cy)
                    if 0 <= iy < person_mask.shape[0] and 0 <= ix < person_mask.shape[1]:
                        if person_mask[iy, ix] > 0:
                            continue
                # 检查亮色
                roi = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                if roi.size == 0:
                    continue
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                bright = ((roi_hsv[:,:,1] < 80) & (roi_hsv[:,:,2] > 150)).sum()
                total = roi.shape[0] * roi.shape[1]
                if bright / (total + 1e-6) < 0.3:
                    continue
                score = 1.0 / (area + 1)
                candidates.append((cx, cy, score, 'yolo'))
        except Exception:
            pass
        return candidates

    def _merge_candidates(self, candidates, dist_thresh=20):
        """合并距离相近的候选，取最高分"""
        if len(candidates) <= 1:
            return [(c[0], c[1], c[2]) for c in candidates]

        # 按分数降序
        sorted_cands = sorted(candidates, key=lambda c: c[2], reverse=True)
        merged = []

        for cx, cy, score, src in sorted_cands:
            found = False
            for i, (mx, my, ms) in enumerate(merged):
                if np.sqrt((cx - mx)**2 + (cy - my)**2) < dist_thresh:
                    # 合并：取最高分的位置
                    merged[i] = (mx, my, ms + score)
                    found = True
                    break
            if not found:
                merged.append((cx, cy, score))

        return merged


class CentroidTracker:
    """
    质心跟踪器
    基于最近邻匹配 + 卡尔曼滤波的羽毛球跟踪
    支持预测位置辅助匹配（检测丢失时用卡尔曼预测）
    """

    def __init__(self, max_disappeared=None, max_distance=None):
        """
        Args:
            max_disappeared: 目标消失后最大保留帧数
            max_distance: 关联匹配最大像素距离
        """
        self.max_disappeared = max_disappeared or config.BALL_TRACK_MAX_DISAPPEARED
        self.max_distance = max_distance or config.BALL_TRACK_MAX_DISTANCE

        self.next_id = 0
        self.objects = {}          # id -> (cx, cy)
        self.kalman_filters = {}   # id -> cv2.KalmanFilter
        self.disappeared = {}      # id -> 连续消失帧数
        self.trajectory = {}       # id -> [(cx, cy, frame_num), ...]
        self.all_trajectories = [] # 所有历史轨迹（球出界/换球后保留）

    def update(self, detection, frame_num):
        """
        用新检测更新跟踪状态

        Args:
            detection: np.ndarray (cx, cy) 或 None
            frame_num: 当前帧号

        Returns:
            np.ndarray (cx, cy) 当前跟踪位置 或 None
        """
        # 没有检测到
        if detection is None:
            to_remove = []
            for obj_id in self.disappeared:
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    to_remove.append(obj_id)
            for obj_id in to_remove:
                self._deregister(obj_id)

            # 返回卡尔曼预测位置（如果有活跃目标）
            if self.objects:
                best_id = min(self.objects.keys())
                kf = self.kalman_filters[best_id]
                prediction = kf.predict()
                pos = np.array([prediction[0, 0], prediction[1, 0]], dtype=np.float32)
                self.objects[best_id] = (float(pos[0]), float(pos[1]))
                self.trajectory[best_id].append((float(pos[0]), float(pos[1]), frame_num))
                return pos
            return None

        det = (float(detection[0]), float(detection[1]))

        # 没有已有目标 → 注册
        if not self.objects:
            self._register(det, frame_num)
            return np.array(det, dtype=np.float32)

        # 已有目标 → 最近邻匹配
        best_id = min(self.objects.keys())
        best_dist = np.linalg.norm(
            np.array(det) - np.array(self.objects[best_id])
        )

        if best_dist < self.max_distance:
            # 匹配成功 → 卡尔曼更新
            kf = self.kalman_filters[best_id]
            measurement = np.array([[np.float32(det[0])], [np.float32(det[1])]])
            kf.correct(measurement)
            prediction = kf.predict()
            smooth_pos = (float(prediction[0, 0]), float(prediction[1, 0]))
            self.objects[best_id] = smooth_pos
            self.disappeared[best_id] = 0
            self.trajectory[best_id].append((smooth_pos[0], smooth_pos[1], frame_num))
            return np.array(smooth_pos, dtype=np.float32)
        else:
            # 距离太远 → 可能是新球（如发球），保存旧轨迹后重新注册
            self._deregister(best_id)
            self._register(det, frame_num)
            return np.array(det, dtype=np.float32)

    def _register(self, centroid, frame_num):
        """注册新跟踪目标"""
        obj_id = self.next_id
        self.next_id += 1

        self.objects[obj_id] = centroid
        self.disappeared[obj_id] = 0
        self.trajectory[obj_id] = [(centroid[0], centroid[1], frame_num)]

        # 初始化卡尔曼滤波器
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                          [0, 1, 0, 0]], dtype=np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                         [0, 1, 0, 1],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]], dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * config.KALMAN_PROCESS_NOISE
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * config.KALMAN_MEASUREMENT_NOISE

        # 初始化状态
        kf.statePost = np.array([[centroid[0]], [centroid[1]], [0], [0]], dtype=np.float32)
        self.kalman_filters[obj_id] = kf

    def _deregister(self, obj_id):
        """注销跟踪目标"""
        if obj_id in self.trajectory and len(self.trajectory[obj_id]) >= 3:
            self.all_trajectories.append(list(self.trajectory[obj_id]))
        del self.objects[obj_id]
        del self.kalman_filters[obj_id]
        del self.disappeared[obj_id]
        if obj_id in self.trajectory:
            del self.trajectory[obj_id]

    def get_new_completed_trajectories(self, after_id=-1):
        """
        获取ID大于after_id的已完成轨迹

        Args:
            after_id: 上次处理的最大轨迹索引

        Returns:
            list of (index, trajectory_points)
        """
        results = []
        for i, traj in enumerate(self.all_trajectories):
            if i > after_id:
                results.append((i, traj))
        return results

    def get_active_trajectory(self):
        """
        获取当前活跃目标的轨迹

        Returns:
            list of (cx, cy, frame_num) 或空列表
        """
        if not self.objects:
            return []
        best_id = min(self.objects.keys())
        return self.trajectory.get(best_id, [])


class TrajectoryFitter:
    """
    飞行轨迹拟合器
    用时间参数化拟合抛物线：x(t)和y(t)分别用二次多项式拟合
    支持基于时间的落点预测和轨迹可视化
    """

    def __init__(self, fit_order=None, min_points=None):
        """
        Args:
            fit_order: 多项式阶数（2=抛物线）
            min_points: 拟合最少点数
        """
        self.fit_order = fit_order or config.TRAJECTORY_FIT_ORDER
        self.min_points = min_points or config.TRAJECTORY_MIN_POINTS

    def fit(self, trajectory_points):
        """
        拟合飞行轨迹（时间参数化）

        Args:
            trajectory_points: list of (cx, cy, frame_num)

        Returns:
            dict: {
                'coeffs_x': x(t)多项式系数,
                'coeffs_y': y(t)多项式系数,
                'valid': bool,
                't_range': (t_min, t_max),
                'x_coords': 原始 x 坐标,
                'y_coords': 原始 y 坐标
            }
        """
        if len(trajectory_points) < self.min_points:
            return {'coeffs': None, 'valid': False, 'x_coords': None, 'y_coords': None}

        pts = np.array(trajectory_points)
        x_coords = pts[:, 0]
        y_coords = pts[:, 1]
        t_coords = pts[:, 2].astype(float)

        # 需要足够的时间跨度
        if t_coords.max() - t_coords.min() < 3:
            return {'coeffs': None, 'valid': False, 'x_coords': x_coords, 'y_coords': y_coords}

        # 过滤卡尔曼预测的轨迹（检测丢失时的预测可能不准）
        # 只保留有显著运动的点
        x_range = x_coords.max() - x_coords.min()
        y_range = y_coords.max() - y_coords.min()
        if x_range < 5 and y_range < 5:
            return {'coeffs': None, 'valid': False, 'x_coords': x_coords, 'y_coords': y_coords}

        try:
            # 时间参数化：用帧号归一化到[0,1]
            t_norm = (t_coords - t_coords.min()) / (t_coords.max() - t_coords.min() + 1e-6)

            # 用原始帧号做多项式拟合（更稳定）
            coeffs_x = np.polyfit(t_coords, x_coords, self.fit_order)
            coeffs_y = np.polyfit(t_coords, y_coords, self.fit_order)

            # 验证拟合质量：R²
            x_pred = np.polyval(coeffs_x, t_coords)
            y_pred = np.polyval(coeffs_y, t_coords)
            r2_x = 1 - np.sum((x_coords - x_pred)**2) / (np.sum((x_coords - x_coords.mean())**2) + 1e-6)
            r2_y = 1 - np.sum((y_coords - y_pred)**2) / (np.sum((y_coords - y_coords.mean())**2) + 1e-6)

            # 至少一个方向拟合良好
            if max(r2_x, r2_y) < 0.5:
                return {'coeffs': None, 'valid': False, 'x_coords': x_coords, 'y_coords': y_coords}

            # 兼容旧接口：存储为 (coeffs_x, coeffs_y, t_range) 元组
            return {
                'coeffs': (coeffs_x, coeffs_y, (t_coords.min(), t_coords.max())),
                'valid': True,
                'x_coords': x_coords,
                'y_coords': y_coords
            }
        except (np.linalg.LinAlgError, ValueError):
            return {'coeffs': None, 'valid': False, 'x_coords': x_coords, 'y_coords': y_coords}

    def predict_landing(self, coeffs, frame_height):
        """
        预测落点（y达到frame_height时的x坐标和时间）

        Args:
            coeffs: (coeffs_x, coeffs_y, (t_min, t_max)) 时间参数化系数
            frame_height: 图像高度（像素）

        Returns:
            np.ndarray (x, y) 预测落点 或 None
        """
        if coeffs is None:
            return None

        # 兼容旧格式（直接x-y多项式）
        if not isinstance(coeffs, tuple):
            eq = coeffs.copy()
            eq[-1] -= frame_height
            roots = np.roots(eq)
            real_roots = roots[np.isreal(roots)].real
            if len(real_roots) == 0:
                return None
            return np.array([real_roots[-1], float(frame_height)], dtype=np.float32)

        coeffs_x, coeffs_y, (t_min, t_max) = coeffs

        # 解 y(t) = frame_height
        eq_y = coeffs_y.copy()
        eq_y[-1] -= frame_height

        roots = np.roots(eq_y)
        real_roots = roots[np.isreal(roots)].real

        if len(real_roots) == 0:
            return None

        # 取最接近当前轨迹末端时间的根（t > t_max表示将来落地）
        valid_roots = real_roots[real_roots >= t_min]
        if len(valid_roots) == 0:
            valid_roots = real_roots

        t_landing = valid_roots[np.argmin(np.abs(valid_roots - t_max))]
        x_landing = np.polyval(coeffs_x, t_landing)

        return np.array([float(x_landing), float(frame_height)], dtype=np.float32)

    def get_predicted_positions(self, coeffs, x_min, x_max, num_points=50):
        """
        生成轨迹曲线上的点（用于可视化）

        Args:
            coeffs: 多项式系数（兼容新旧格式）
            x_min, x_max: x 范围
            num_points: 采样点数

        Returns:
            np.ndarray (num_points, 2) 或 None
        """
        if coeffs is None:
            return None

        # 新格式：时间参数化
        if isinstance(coeffs, tuple):
            coeffs_x, coeffs_y, (t_min, t_max) = coeffs
            ts = np.linspace(t_min, t_max * 1.5, num_points)
            xs = np.polyval(coeffs_x, ts)
            ys = np.polyval(coeffs_y, ts)
            return np.column_stack([xs, ys]).astype(np.float32)

        # 旧格式：x-y多项式
        xs = np.linspace(x_min, x_max, num_points)
        ys = np.polyval(coeffs, xs)
        return np.column_stack([xs, ys]).astype(np.float32)
