# -*- coding: utf-8 -*-
"""
球场检测模块
自动检测羽毛球场地标线，计算透视变换矩阵
支持手动4点标定回退
"""

import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class CourtDetector:
    """
    球场检测器
    通过 HSV 颜色过滤 → Canny 边缘 → Hough 线段 → 交点聚类 → 角点排序
    自动检测球场四角并计算透视变换矩阵
    """

    def __init__(self):
        """初始化球场检测器，加载配置参数"""
        self.hsv_lower = config.COURT_HSV_LOWER
        self.hsv_upper = config.COURT_HSV_UPPER
        self.canny_low = config.COURT_CANNY_LOW
        self.canny_high = config.COURT_CANNY_HIGH
        self.hough_threshold = config.COURT_HOUGH_THRESHOLD
        self.hough_min_line = config.COURT_HOUGH_MIN_LINE_LENGTH
        self.hough_max_gap = config.COURT_HOUGH_MAX_LINE_GAP
        self.angle_tol = config.COURT_ANGLE_TOLERANCE

    def detect_court(self, frame):
        """
        检测球场角点并计算透视变换矩阵
        自动检测失败时使用基于帧尺寸的默认映射（保证热力图可用）

        Args:
            frame: BGR 图像

        Returns:
            dict: {
                'corners': np.ndarray (4,2) float32 或 None,
                'homography_warp': np.ndarray (3,3) 像素→俯视图像素 或 None,
                'homography_court': np.ndarray (3,3) 像素→米制坐标 或 None,
                'success': bool
            }
        """
        corners = self._auto_detect(frame)

        if corners is None:
            # 自动检测失败，使用基于帧尺寸的默认映射
            h, w = frame.shape[:2]
            corners = self._default_corners(w, h)
            H_warp, H_court = self.compute_homography(corners)
            return {
                'corners': corners,
                'homography_warp': H_warp,
                'homography_court': H_court,
                'success': True,
                'fallback': True
            }

        H_warp, H_court = self.compute_homography(corners)
        return {
            'corners': corners,
            'homography_warp': H_warp,
            'homography_court': H_court,
            'success': True,
            'fallback': False
        }

    def _default_corners(self, frame_w, frame_h):
        """
        基于帧尺寸生成默认球场角点
        假设球场占据画面中心约80%区域

        Args:
            frame_w, frame_h: 帧宽高

        Returns:
            np.ndarray (4,2) float32
        """
        margin_x = frame_w * 0.10
        margin_y = frame_h * 0.10
        return np.array([
            [margin_x, margin_y],
            [frame_w - margin_x, margin_y],
            [frame_w - margin_x, frame_h - margin_y],
            [margin_x, frame_h - margin_y],
        ], dtype=np.float32)

    def _auto_detect(self, frame):
        """
        自动检测球场四角

        Pipeline: HSV掩膜 → 形态学 → Canny → HoughLinesP → 交点聚类

        Args:
            frame: BGR 图像

        Returns:
            np.ndarray (4,2) float32 或 None
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # 形态学去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Canny 边缘检测
        edges = cv2.Canny(mask, self.canny_low, self.canny_high)

        # Hough 线段检测
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line,
            maxLineGap=self.hough_max_gap
        )

        if lines is None or len(lines) < 2:
            return None

        # 按角度筛选：保留近似水平或垂直的线段
        h_lines, v_lines = self._filter_lines_by_angle(lines)

        if len(h_lines) < 1 or len(v_lines) < 1:
            return None

        # 计算所有水平线和垂直线的交点
        intersections = []
        for hl in h_lines:
            for vl in v_lines:
                pt = self._line_intersection(hl, vl)
                if pt is not None:
                    # 检查交点是否在图像范围内
                    h, w = frame.shape[:2]
                    if 0 <= pt[0] <= w and 0 <= pt[1] <= h:
                        intersections.append(pt)

        if len(intersections) < 4:
            return None

        intersections = np.array(intersections, dtype=np.float32)

        # K-means 聚类为 4 个角点
        corners = self._cluster_corners(intersections)
        if corners is None:
            return None

        # 排序角点：TL, TR, BR, BL
        corners = self._order_corners(corners)

        # 验证凸四边形 + 长宽比
        if not self._validate_corners(corners):
            return None

        return corners

    def _filter_lines_by_angle(self, lines):
        """
        按角度筛选水平/垂直线段

        Args:
            lines: HoughLinesP 输出 (N, 1, 4)

        Returns:
            (h_lines, v_lines): 水平线列表, 垂直线列表
        """
        h_lines = []
        v_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

            # 水平线：角度接近 0 或 180
            if angle < self.angle_tol or angle > (180 - self.angle_tol):
                h_lines.append(np.array([x1, y1, x2, y2], dtype=np.float32))
            # 垂直线：角度接近 90
            elif abs(angle - 90) < self.angle_tol:
                v_lines.append(np.array([x1, y1, x2, y2], dtype=np.float32))

        return h_lines, v_lines

    def _line_intersection(self, line1, line2):
        """
        计算两条线段的交点

        Args:
            line1: [x1, y1, x2, y2]
            line2: [x1, y1, x2, y2]

        Returns:
            np.ndarray (2,) 或 None（平行时返回 None）
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-8:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)

        return np.array([ix, iy], dtype=np.float32)

    def _cluster_corners(self, points):
        """
        将交点聚类为 4 个角点

        Args:
            points: np.ndarray (N, 2) 所有交点

        Returns:
            np.ndarray (4, 2) 或 None
        """
        if len(points) < 4:
            return None

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
        _, labels, centers = cv2.kmeans(
            points, 4, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )

        if centers is not None and len(centers) == 4:
            return centers.astype(np.float32)
        return None

    def _order_corners(self, corners):
        """
        将 4 个角点排序为 TL, TR, BR, BL 顺序

        策略：以质心为原点，按 atan2 角度排序，从最左上角开始顺时针排列。
        对任意摄像头角度都稳定。

        Args:
            corners: np.ndarray (4, 2)

        Returns:
            np.ndarray (4, 2) 排序后的角点
        """
        center = corners.mean(axis=0)
        # atan2: 右=0, 下=π/2, 左=±π, 上=-π/2
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])

        # 按角度从小到大排序（从右上方逆时针方向）
        sorted_idx = np.argsort(angles)

        # 找到 TL：x+y 最小的那个角点
        s = corners.sum(axis=1)
        tl_in_sorted = np.argmin(s[sorted_idx])

        # 从 TL 开始，顺时针取 4 个角点
        ordered = np.zeros((4, 2), dtype=np.float32)
        for i in range(4):
            ordered[i] = corners[sorted_idx[(tl_in_sorted + i) % 4]]

        return ordered

    def _validate_corners(self, corners):
        """
        验证 4 个角点是否构成合理的凸四边形

        检查：凸性 + 长宽比在 [1.5, 3.0] 范围内（标准比 13.4/6.1≈2.2）

        Args:
            corners: np.ndarray (4, 2)

        Returns:
            bool
        """
        hull = cv2.convexHull(corners)
        if len(hull) != 4:
            return False

        # 计算长宽比
        rect = cv2.minAreaRect(corners)
        w, h = rect[1]
        if w < 1 or h < 1:
            return False

        ratio = max(w, h) / min(w, h)
        return 1.5 <= ratio <= 3.0

    def compute_homography(self, corners):
        """
        计算两组透视变换矩阵

        Args:
            corners: np.ndarray (4,2) 球场像素角点 TL/TR/BR/BL

        Returns:
            (H_warp, H_court):
                H_warp: 像素→俯视图像素（用于 warpPerspective 显示）
                H_court: 像素→球场米制坐标（用于坐标映射）
        """
        # 目标1：俯视图像素坐标
        dst_warp = np.array([
            [0, 0],
            [config.BIRDEYE_WIDTH, 0],
            [config.BIRDEYE_WIDTH, config.BIRDEYE_HEIGHT],
            [0, config.BIRDEYE_HEIGHT]
        ], dtype=np.float32)

        H_warp = cv2.getPerspectiveTransform(corners, dst_warp)

        # 目标2：标准球场米制坐标
        H_court = cv2.getPerspectiveTransform(corners, config.COURT_CORNERS_REAL)

        return H_warp, H_court

    def get_warp_corners(self):
        """
        返回俯视图目标角点（像素坐标）

        Returns:
            np.ndarray (4,2)
        """
        return np.array([
            [0, 0],
            [config.BIRDEYE_WIDTH, 0],
            [config.BIRDEYE_WIDTH, config.BIRDEYE_HEIGHT],
            [0, config.BIRDEYE_HEIGHT]
        ], dtype=np.float32)


def manual_court_selection(frame):
    """
    手动标定球场四角（OpenCV 鼠标回调交互）

    用户按顺序点击 4 个角点（左上→右上→右下→左下），
    按 ENTER 确认，按 R 重置

    Args:
        frame: BGR 图像

    Returns:
        np.ndarray (4,2) float32 用户选择的角点
    """
    points = []
    clone = frame.copy()

    def _mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            cv2.circle(clone, (x, y), 6, (0, 0, 255), -1)
            if len(points) > 1:
                cv2.line(clone, tuple(points[-2]), tuple(points[-1]), (0, 255, 0), 2)
            if len(points) == 4:
                cv2.line(clone, tuple(points[3]), tuple(points[0]), (0, 255, 0), 2)
            cv2.imshow("手动标定球场 - 点击4个角点 (ENTER确认, R重置)", clone)

    cv2.imshow("手动标定球场 - 点击4个角点 (ENTER确认, R重置)", clone)
    cv2.setMouseCallback("手动标定球场 - 点击4个角点 (ENTER确认, R重置)", _mouse_callback)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and len(points) == 4:  # ENTER
            break
        elif key == ord('r') or key == ord('R'):  # 重置
            points = []
            clone = frame.copy()
            cv2.imshow("手动标定球场 - 点击4个角点 (ENTER确认, R重置)", clone)

    cv2.destroyWindow("手动标定球场 - 点击4个角点 (ENTER确认, R重置)")
    return np.array(points, dtype=np.float32)
