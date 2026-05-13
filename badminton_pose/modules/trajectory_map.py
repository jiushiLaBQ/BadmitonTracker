# -*- coding: utf-8 -*-
"""
羽毛球轨迹图生成模块
- 接收羽毛球在视频帧中的坐标
- 将视频坐标通过单应性矩阵转换为球场平面坐标
- 在空白的球场图上绘制轨迹
- 使用颜色渐变表示轨迹的时间顺序
"""

import cv2
import numpy as np
from collections import deque

class TrajectoryMap:
    def __init__(self, court_template_path, max_points=100):
        self.court_template = cv2.imread(court_template_path)
        if self.court_template is None:
            raise ValueError(f"无法加载球场模板图片: {court_template_path}")
        
        self.trajectory_points = deque(maxlen=max_points)
        
        # 这个单应性矩阵需要与 CourtHeatmap 中的保持一致
        # 我们先使用一个默认值，之后会从主程序同步
        self.homography_matrix = np.array([
            [ 8.0e-01,  0.0e+00, -2.0e+02], # X轴：拉长并重新居中
            [ 0.0e+00,  1.5e+00, -1.5e+02], # Y轴：整体下移
            [ 0.0e+00,  0.0e+00,  1.0e+00]
        ])
        print("[Info] TrajectoryMap 初始化成功")

    def set_homography_matrix(self, matrix):
        """从外部设置单应性矩阵，以保持与热力图模块一致"""
        self.homography_matrix = matrix
        print("[Info] TrajectoryMap 已同步更新单应性矩阵")

    def add_point(self, point_coord):
        """
        添加一个新的视频坐标点，并将其转换到球场坐标系中
        """
        if point_coord is None:
            return

        # 防御性编程：确保输入格式绝对正确
        # 1. 确保输入是numpy数组
        point_video_flat = np.array(point_coord, dtype=np.float32)
        # 2. 检查基本形状是否为 (2,)
        if point_video_flat.shape != (2,):
            print(f"[ERROR] TrajectoryMap: Unexpected point_coord format: {point_coord}")
            return
        # 3. 强制重塑为 (1, 1, 2)
        point_video_reshaped = point_video_flat.reshape(1, 1, 2)

        # 使用单应性矩阵进行坐标变换
        point_court = cv2.perspectiveTransform(point_video_reshaped, self.homography_matrix)
        
        if point_court is None:
            return
        
        # 提取变换后的坐标
        cx, cy = point_court[0][0]

        # 限制坐标在球场图片范围内
        height, width, _ = self.court_template.shape
        cx = np.clip(cx, 0, width - 1)
        cy = np.clip(cy, 0, height - 1)

        self.trajectory_points.append((int(cx), int(cy)))
        print(f"[Debug Trajectory] Video: {point_coord} -> Court: [{cx:10.2f} {cy:10.2f}]")


    def generate_map(self):
        """在球场模板上绘制轨迹并返回图像"""
        trajectory_map = self.court_template.copy()
        num_points = len(self.trajectory_points)

        for i in range(1, num_points):
            start_point = self.trajectory_points[i-1]
            end_point = self.trajectory_points[i]
            
            # 使用颜色渐变（从蓝色到红色）
            # i / num_points 的值从 0 -> 1
            color_ratio = i / num_points
            # BGR: (255, 0, 0) -> (0, 0, 255)
            b = 255 * (1 - color_ratio)
            r = 255 * color_ratio
            color = (b, 0, r)
            
            cv2.line(trajectory_map, start_point, end_point, color, 2)
            cv2.circle(trajectory_map, end_point, 3, color, -1)

        return trajectory_map