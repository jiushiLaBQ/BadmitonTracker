# -*- coding: utf-8 -*-
"""
通用物体检测器模块
- 使用YOLOv8通用模型 (yolov8n.pt)
- 专门用于检测羽毛球 (在COCO数据集中被归类为 'sports ball')
"""

import os
import torch
from ultralytics import YOLO

class ObjectDetector:
    """
    一个用于检测特定物体的类，这里我们用它来找羽毛球。
    """
    def __init__(self, model_name='yolov8n.pt', conf=0.25):
        """
        初始化检测器。
        :param model_name: 模型文件名，例如 'yolov8n.pt'。
        :param conf: 置信度阈值，低于此值的检测结果将被忽略。
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 根据用户指定，直接在项目根目录查找模型
        model_path = os.path.join(os.path.dirname(__file__), '..', model_name)
        
        # 检查模型文件是否存在。如果不存在，YOLO会自动尝试下载。
        if not os.path.exists(model_path):
            print(f"警告: 模型文件 {model_path} 不存在。将尝试从网络下载 '{model_name}'。")
            self.model = YOLO(model_name)
        else:
            self.model = YOLO(model_path)

        self.model.to(self.device)
        
        # 在COCO数据集中, 'sports ball' 的类别ID是32
        self.target_class_id = 32
        self.conf = conf
        
        print(f"物体检测器加载完成 | 模型: {model_name} | 设备: {self.device}")

    def detect_ball(self, frame):
        """
        在给定的图像帧中检测羽毛球。
        
        :param frame: BGR格式的OpenCV图像。
        :return: 如果找到球，则返回其中心坐标 (x, y)；否则返回 None。
        """
        # 使用模型进行推理，只关注'sports ball'类别
        results = self.model(frame, conf=self.conf, classes=[self.target_class_id], verbose=False)
        
        best_ball = None
        max_conf = 0

        # 遍历所有检测到的物体（虽然我们只筛选了球）
        # 选择置信度最高的那个作为最终结果
        for box in results[0].boxes:
            conf = box.conf[0].item()
            if conf > max_conf:
                max_conf = conf
                xyxy = box.xyxy[0].cpu().numpy()
                center_x = (xyxy[0] + xyxy[2]) / 2
                center_y = (xyxy[1] + xyxy[3]) / 2
                best_ball = (center_x, center_y)
                
        return best_ball