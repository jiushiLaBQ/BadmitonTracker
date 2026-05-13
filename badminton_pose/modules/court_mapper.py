# -*- coding: utf-8 -*-
"""
球场坐标映射模块
像素坐标 ↔ 标准球场米制坐标 转换
标准球场图绘制
"""

import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class CourtMapper:
    """
    球场坐标映射器
    利用透视变换矩阵实现像素坐标与标准球场米制坐标的互转
    """

    def __init__(self, homography=None):
        """
        Args:
            homography: np.ndarray (3,3) 像素→米制的透视变换矩阵
        """
        self.H = homography
        self.H_inv = None
        if homography is not None:
            self.H_inv = np.linalg.inv(homography)

    def set_homography(self, homography):
        """设置/更新透视变换矩阵"""
        self.H = homography
        self.H_inv = np.linalg.inv(homography)

    def pixel_to_court(self, points):
        """
        像素坐标 → 球场米制坐标

        Args:
            points: np.ndarray (2,) 或 (N,2) 像素坐标

        Returns:
            np.ndarray (N,2) 米制坐标
        """
        if self.H is None:
            raise ValueError("未设置透视变换矩阵，请先调用 set_homography()")

        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        court_pts = cv2.perspectiveTransform(pts, self.H)
        return court_pts.reshape(-1, 2)

    def court_to_pixel(self, court_points):
        """
        球场米制坐标 → 像素坐标

        Args:
            court_points: np.ndarray (2,) 或 (N,2) 米制坐标

        Returns:
            np.ndarray (N,2) 像素坐标
        """
        if self.H_inv is None:
            raise ValueError("未设置透视变换矩阵，请先调用 set_homography()")

        pts = np.array(court_points, dtype=np.float32).reshape(-1, 1, 2)
        pixel_pts = cv2.perspectiveTransform(pts, self.H_inv)
        return pixel_pts.reshape(-1, 2)

    def get_court_stats(self, court_points):
        """
        统计落点分布

        Args:
            court_points: np.ndarray (N,2) 米制坐标

        Returns:
            dict: {
                'total': 总点数,
                'front_count': 前场点数 (x < 6.7),
                'back_count': 后场点数 (x >= 6.7),
                'left_count': 左半场点数 (y < 3.05),
                'right_count': 右半场点数 (y >= 3.05),
                'in_court_count': 场内点数,
                'mean_position': (mean_x, mean_y),
                'coverage_area': 覆盖面积
            }
        """
        pts = np.array(court_points).reshape(-1, 2)
        net = config.NET_POSITION_M
        half_w = config.COURT_WIDTH_M / 2

        front = pts[pts[:, 0] < net]
        back = pts[pts[:, 0] >= net]
        left = pts[pts[:, 1] < half_w]
        right = pts[pts[:, 1] >= half_w]

        in_court = sum(1 for p in pts if self.is_in_court(p))

        # 覆盖面积（包围盒）
        if len(pts) >= 2:
            area = (pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min())
        else:
            area = 0.0

        return {
            'total': len(pts),
            'front_count': len(front),
            'back_count': len(back),
            'left_count': len(left),
            'right_count': len(right),
            'in_court_count': in_court,
            'mean_position': (float(pts[:, 0].mean()), float(pts[:, 1].mean())) if len(pts) > 0 else (0, 0),
            'coverage_area': float(area)
        }

    def is_in_court(self, court_point):
        """
        判断点是否在标准球场范围内

        Args:
            court_point: (x, y) 米制坐标

        Returns:
            bool
        """
        x, y = court_point
        return 0 <= x <= config.COURT_LENGTH_M and 0 <= y <= config.COURT_WIDTH_M


def draw_court_diagram(width, height):
    """
    在空白画布上绘制标准双打羽毛球场地

    Args:
        width: 画布宽度（像素）
        height: 画布高度（像素）

    Returns:
        np.ndarray BGR 图像 (height, width, 3)
    """
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    # 坐标缩放因子
    sx = width / config.COURT_LENGTH_M
    sy = height / config.COURT_WIDTH_M

    def _m2px(mx, my):
        """米制坐标转像素坐标"""
        return int(mx * sx), int(my * sy)

    # 颜色定义
    line_color = (0, 0, 0)        # 黑色标线
    net_color = (0, 0, 200)       # 红色球网
    line_t = 2                     # 线宽

    # 1. 外边界（双打边线）
    tl = _m2px(0, 0)
    tr = _m2px(config.COURT_LENGTH_M, 0)
    br = _m2px(config.COURT_LENGTH_M, config.COURT_WIDTH_M)
    bl = _m2px(0, config.COURT_WIDTH_M)
    cv2.rectangle(canvas, tl, br, line_color, line_t)

    # 2. 球网（x = 6.7m）
    net_x = _m2px(config.NET_POSITION_M, 0)[0]
    cv2.line(canvas, (net_x, tl[1]), (net_x, bl[1]), net_color, line_t + 1)

    # 3. 前发球线（距球网 1.98m）
    # 左半场：x = 6.7 - 1.98 = 4.72
    # 右半场：x = 6.7 + 1.98 = 8.68
    fsl_left = _m2px(config.NET_POSITION_M - 1.98, 0)[0]
    fsl_right = _m2px(config.NET_POSITION_M + 1.98, 0)[0]
    cv2.line(canvas, (fsl_left, tl[1]), (fsl_left, bl[1]), line_color, line_t)
    cv2.line(canvas, (fsl_right, tl[1]), (fsl_right, bl[1]), line_color, line_t)

    # 4. 双打后发球线（距两端 0.76m）
    bsl_left = _m2px(0.76, 0)[0]
    bsl_right = _m2px(config.COURT_LENGTH_M - 0.76, 0)[0]
    cv2.line(canvas, (bsl_left, tl[1]), (bsl_left, bl[1]), line_color, line_t)
    cv2.line(canvas, (bsl_right, tl[1]), (bsl_right, bl[1]), line_color, line_t)

    # 5. 中线（y = 3.05m，前发球线与后发球线之间）
    mid_y = _m2px(0, config.COURT_WIDTH_M / 2)[1]
    # 左半场中线
    cv2.line(canvas, (bsl_left, mid_y), (fsl_left, mid_y), line_color, line_t)
    # 右半场中线
    cv2.line(canvas, (fsl_right, mid_y), (bsl_right, mid_y), line_color, line_t)

    # 6. 单打边线（y = 0.46m 和 y = 5.64m）
    sl_left = _m2px(0, 0.46)[1]
    sl_right = _m2px(0, config.COURT_WIDTH_M - 0.46)[1]
    cv2.line(canvas, (tl[0], sl_left), (br[0], sl_left), line_color, line_t)
    cv2.line(canvas, (tl[0], sl_right), (br[0], sl_right), line_color, line_t)

    return canvas
