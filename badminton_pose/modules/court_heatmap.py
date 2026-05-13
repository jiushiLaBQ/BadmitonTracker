# -*- coding: utf-8 -*-
"""
球场热力图生成模块
- 接收球员在视频帧中的坐标（如脚踝）
- 将视频坐标通过单应性矩阵转换为球场平面坐标
- 在空白的球场图上累积热力点
- 生成并返回叠加了热力图的球场图像
"""

import cv2
import numpy as np

class CourtHeatmap:
    def __init__(self, court_template_path, width, height):
        self.court_template = cv2.imread(court_template_path)
        if self.court_template is None:
            raise ValueError(f"无法加载球场模板图片: {court_template_path}")
        
        self.width = width
        self.height = height
        self.heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        
        # 这个单应性矩阵是关键，用于将视频坐标映射到球场图片坐标
        # 需要根据实际视频的拍摄角度进行调整
        self.homography_matrix = np.array([
            [ 8.0e-01,  0.0e+00, -2.0e+02],
            [ 0.0e+00,  1.5e+00, -1.5e+02],
            [ 0.0e+00,  0.0e+00,  1.0e+00]
        ])
        print("[Info] CourtHeatmap 初始化成功")

    def set_homography_matrix(self, matrix):
        """从外部设置单应性矩阵"""
        self.homography_matrix = matrix
        print("[Info] CourtHeatmap 已同步更新单应性矩阵")

    def add_point(self, point_coord):
        if point_coord is None:
            return

        # 防御性编程：确保输入格式绝对正确
        # 1. 确保输入是numpy数组
        point_video_flat = np.array(point_coord, dtype=np.float32)
        # 2. 检查基本形状是否为 (2,)
        if point_video_flat.shape != (2,):
            print(f"[ERROR] CourtHeatmap: Unexpected point_coord format: {point_coord}")
            return
        # 3. 强制重塑为 (1, 1, 2)
        point_video_reshaped = point_video_flat.reshape(1, 1, 2)

        # 使用单应性矩阵进行坐标变换
        point_court = cv2.perspectiveTransform(point_video_reshaped, self.homography_matrix)
        
        if point_court is None:
            return
            
        # 提取变换后的坐标
        cx, cy = point_court[0][0]

        # 限制坐标在热力图范围内
        cx = np.clip(cx, 0, self.width - 1)
        cy = np.clip(cy, 0, self.height - 1)

        # 在热力图上增加权重，这里简单处理，可以直接增加一个值
        # 也可以使用高斯核来创建一个更平滑的热点
        # 为了性能，我们先用简单的方式
        # 确保在边界内
        int_cx, int_cy = int(cx), int(cy)
        if 0 <= int_cx < self.width and 0 <= int_cy < self.height:
            self.heatmap[int_cy, int_cx] += 50 # 增加热力值

    def generate_heatmap_image(self):
        # 归一化热力图
        norm_heatmap = self.heatmap.copy()
        if np.max(norm_heatmap) > 0:
            norm_heatmap = cv2.normalize(norm_heatmap, None, 0, 255, cv2.NORM_MINMAX)
        
        norm_heatmap = norm_heatmap.astype(np.uint8)

        # 应用颜色映射
        heatmap_colored = cv2.applyColorMap(norm_heatmap, cv2.COLORMAP_JET)

        # 创建一个纯黑色的背景以突出显示热力图
        heatmap_on_black = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # 创建一个掩码，只在有热度的区域进行绘制
        mask = norm_heatmap > 0
        
        # 将彩色的热力图部分直接绘制到黑色背景上
        heatmap_on_black[mask] = heatmap_colored[mask]

        return heatmap_on_black

    def reset(self):
        """重置热力图"""
        self.heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        print("[Info] CourtHeatmap 已重置")