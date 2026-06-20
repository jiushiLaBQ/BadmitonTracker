# -*- coding: utf-8 -*-
"""
热力图生成模块
基于球场坐标的高斯核密度估计热力图
支持全局落点热力 + 按动作分类热力
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class HeatmapGenerator:
    """
    热力图生成器
    累积球场坐标落点，生成高斯平滑热力图并叠加到标准球场图上
    """

    def __init__(self):
        """初始化热力图生成器"""
        self.all_points = []                # 全局落点 [(x, y), ...]
        self.class_points = {}              # 按类别分类 {class_idx: [(x, y), ...]}
        self.resolution = config.HEATMAP_RESOLUTION
        self.blur_kernel = config.HEATMAP_BLUR_KERNEL
        self.alpha = config.HEATMAP_ALPHA

    def add_point(self, court_point, class_idx=None):
        """
        添加一个落点

        Args:
            court_point: (x, y) 米制坐标
            class_idx: 动作类别索引（可选）
        """
        pt = (float(court_point[0]), float(court_point[1]))
        self.all_points.append(pt)

        if class_idx is not None:
            if class_idx not in self.class_points:
                self.class_points[class_idx] = []
            self.class_points[class_idx].append(pt)

    def reset(self):
        """清除所有累积落点"""
        self.all_points = []
        self.class_points = {}

    def reset_class(self, class_idx):
        """清除指定类别的落点"""
        if class_idx in self.class_points:
            self.class_points[class_idx] = []

    def generate_heatmap(self, class_idx=None):
        """
        生成热力图叠加到球场图

        Args:
            class_idx: None=全局热力图, int=指定类别热力图

        Returns:
            np.ndarray BGR 图像 (height, width, 3)
        """
        from .court_mapper import draw_court_diagram

        width = config.BIRDEYE_WIDTH
        height = config.BIRDEYE_HEIGHT

        # 获取对应点集
        if class_idx is not None:
            points = self.class_points.get(class_idx, [])
        else:
            points = self.all_points

        # 绘制基础球场图
        court_img = draw_court_diagram(width, height)

        if len(points) < 3:
            return court_img

        pts = np.array(points)

        # 调试：打印前几个点的坐标
        print(f"[HeatmapDebug] 总点数={len(pts)}, 前5个: {pts[:5].tolist()}")

        # 生成 2D 直方图
        heatmap, xedges, yedges = self._histogram_heatmap(pts, width, height)

        # 高斯模糊平滑
        if self.blur_kernel > 0:
            k = self.blur_kernel if self.blur_kernel % 2 == 1 else self.blur_kernel + 1
            heatmap = gaussian_filter(heatmap, sigma=k // 4 + 1)

        # 归一化到 [0, 255]
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        else:
            return court_img

        # 应用颜色映射
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 叠加热力图到球场图
        result = cv2.addWeighted(court_img, 1.0 - self.alpha, heatmap_color, self.alpha, 0)

        return result

    def _histogram_heatmap(self, points, width, height):
        """
        使用 2D 直方图生成密度图

        Args:
            points: np.ndarray (N, 2) 米制坐标
            width, height: 输出画布尺寸

        Returns:
            (heatmap, xedges, yedges)
        """
        x = points[:, 0]
        y = points[:, 1]

        heatmap, xedges, yedges = np.histogram2d(
            x, y,
            bins=[self.resolution, self.resolution],
            range=[[0, config.COURT_LENGTH_M], [0, config.COURT_WIDTH_M]]
        )

        # histogram2d 返回 (x_bins, y_bins)，axis0=x(0~13.4m), axis1=y(0~6.1m)
        # 左右翻转 + 左旋90° → axis0=y(行), axis1=x(列)
        heatmap = np.fliplr(heatmap)         # 左右翻转
        heatmap = np.rot90(heatmap, k=1)     # 左旋90° → (y_bins, x_bins)
        heatmap = cv2.resize(
            heatmap.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

        # 调试：打印热点位置
        if heatmap.max() > 0:
            peak = np.unravel_index(heatmap.argmax(), heatmap.shape)
            print(f"[HeatmapDebug] shape={heatmap.shape}, peak=row{peak[0]},col{peak[1]}, "
                  f"court_x≈{peak[1]/width*config.COURT_LENGTH_M:.1f}m, "
                  f"court_y≈{peak[0]/height*config.COURT_WIDTH_M:.1f}m")

        return heatmap, xedges, yedges

    def get_point_density(self, class_idx=None):
        """
        获取点密度统计（不渲染）

        Args:
            class_idx: None=全局, int=指定类别

        Returns:
            dict: {
                'count': 点数,
                'max_density': 最大密度位置 (x, y),
                'mean_position': 平均位置 (x, y),
                'std_position': 标准差 (std_x, std_y)
            }
        """
        if class_idx is not None:
            points = self.class_points.get(class_idx, [])
        else:
            points = self.all_points

        if len(points) == 0:
            return {
                'count': 0,
                'max_density': (0, 0),
                'mean_position': (0, 0),
                'std_position': (0, 0)
            }

        pts = np.array(points)

        # 用 2D 直方图找最大密度位置
        if len(pts) >= 3:
            heatmap, xe, ye = self._histogram_heatmap(
                pts, config.BIRDEYE_WIDTH, config.BIRDEYE_HEIGHT
            )
            max_idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
            # 转换回米制坐标
            max_x = xe[0] + (xe[-1] - xe[0]) * max_idx[0] / heatmap.shape[0]
            max_y = ye[0] + (ye[-1] - ye[0]) * max_idx[1] / heatmap.shape[1]
        else:
            max_x, max_y = float(pts[:, 0].mean()), float(pts[:, 1].mean())

        return {
            'count': len(pts),
            'max_density': (float(max_x), float(max_y)),
            'mean_position': (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
            'std_position': (float(pts[:, 0].std()), float(pts[:, 1].std()))
        }
