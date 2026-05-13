# -*- coding: utf-8 -*-
"""
关键点批量提取模块
使用YOLOv8-Pose从视频中提取COCO 17个关键点坐标与置信度
支持批量帧处理加速推理
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class KeypointExtractor:
    """
    基于YOLOv8-Pose的关键点提取器
    对每个视频进行均匀间隔采样，批量提取人物姿态关键点
    """

    def __init__(self, model_path=None, conf=None, iou=None, img_size=None):
        """
        初始化提取器

        Args:
            model_path: YOLOv8-Pose模型路径，None则自动下载
            conf: 检测置信度阈值
            iou: NMS IoU阈值
            img_size: 推理图像尺寸
        """
        self.conf = conf or config.YOLO_CONF
        self.iou = iou or config.YOLO_IOU
        self.img_size = img_size or config.YOLO_IMG_SIZE

        print(f"[KeypointExtractor] 加载 YOLOv8-Pose 模型: {config.YOLO_MODEL}")
        self.model = YOLO(model_path or config.YOLO_MODEL)
        print(f"[KeypointExtractor] conf={self.conf}, iou={self.iou}, img_size={self.img_size}")

    def extract_from_video(self, video_path, sample_interval=None):
        """
        从单个视频中均匀采样并批量提取关键点

        Args:
            video_path: 视频文件路径
            sample_interval: 采样间隔，None则使用配置默认值

        Returns:
            keypoints_seq: np.ndarray, shape (num_frames, 17, 3),
                           每帧17个关键点的 (x, y, confidence)
            如果没有检测到人，返回空数组
        """
        interval = sample_interval or config.SAMPLE_INTERVAL
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return np.array([])

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 2:
            cap.release()
            return np.array([])

        # Step 1: 读取所有采样帧到列表
        frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                frames.append(frame)
            frame_idx += 1
        cap.release()

        if len(frames) == 0:
            return np.array([])

        # Step 2: 批量推理（一次送入所有帧，大幅提速）
        results = self.model.predict(
            frames,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,
            classes=[0],       # 只检测person类别
            verbose=False
        )

        # Step 3: 逐帧提取最佳关键点
        keypoints_list = []
        for result in results:
            kpts = self._parse_result(result)
            if kpts is not None:
                keypoints_list.append(kpts)

        if len(keypoints_list) == 0:
            return np.array([])

        return np.array(keypoints_list)  # (N, 17, 3)

    def _parse_result(self, result):
        """
        从单帧YOLO结果中解析最佳人物的关键点

        Args:
            result: YOLO单帧结果

        Returns:
            keypoints: np.ndarray (17, 3) 或 None
        """
        if result.keypoints is None or len(result.keypoints) == 0:
            return None

        kpts_data = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
        boxes_conf = result.boxes.conf.cpu().numpy() if result.boxes is not None else None

        if len(kpts_data) == 0:
            return None

        if len(kpts_data) == 1:
            kpts = kpts_data[0]
        else:
            if boxes_conf is not None:
                best_idx = np.argmax(boxes_conf)
            else:
                avg_conf = kpts_data[:, :, 2].mean(axis=1)
                best_idx = np.argmax(avg_conf)
            kpts = kpts_data[best_idx]

        # 过滤低置信度关键点
        kpts[kpts[:, 2] < config.PERSON_CONF, :2] = 0

        return kpts

    def _detect_pose(self, frame):
        """
        单帧姿态检测（用于实时 GUI 推理）

        Args:
            frame: BGR 图像（单帧）

        Returns:
            keypoints: np.ndarray (17, 3) 或 None（未检测到人物）
        """
        results = self.model.predict(
            [frame],
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,
            classes=[0],
            verbose=False
        )
        if len(results) == 0:
            return None
        return self._parse_result(results[0])

    def extract_batch(self, video_root=None, output_dir=None, sample_interval=None):
        """
        批量提取整个数据集所有视频的关键点

        Args:
            video_root: 视频数据根目录
            output_dir: 关键点保存目录
            sample_interval: 采样间隔

        Returns:
            所有成功提取的视频数量
        """
        video_root = video_root or config.VIDEO_ROOT
        output_dir = output_dir or config.KEYPOINTS_DIR
        os.makedirs(output_dir, exist_ok=True)

        class_dirs = sorted([
            d for d in os.listdir(video_root)
            if os.path.isdir(os.path.join(video_root, d))
        ])

        total_videos = 0
        success_videos = 0
        skip_videos = 0
        failed_videos = 0

        print(f"\n{'='*60}")
        print(f"开始批量提取关键点")
        print(f"视频根目录: {video_root}")
        print(f"类别数量: {len(class_dirs)}")
        print(f"保存目录: {output_dir}")
        print(f"{'='*60}\n")

        for class_idx, class_name in enumerate(class_dirs):
            class_dir = os.path.join(video_root, class_name)
            class_out = os.path.join(output_dir, class_name)
            os.makedirs(class_out, exist_ok=True)

            video_files = [
                f for f in os.listdir(class_dir)
                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
            ]

            print(f"[{class_idx+1}/{len(class_dirs)}] 类别: {class_name} "
                  f"({len(video_files)} 个视频)")

            for video_file in tqdm(video_files, desc=f"  {class_name}", leave=False):
                video_path = os.path.join(class_dir, video_file)
                out_path = os.path.join(class_out, video_file.rsplit('.', 1)[0] + '.npy')

                total_videos += 1

                # 跳过已提取的
                if os.path.exists(out_path):
                    skip_videos += 1
                    continue

                try:
                    keypoints_seq = self.extract_from_video(video_path, sample_interval)
                except Exception:
                    failed_videos += 1
                    continue

                if keypoints_seq.shape[0] > 0:
                    np.save(out_path, keypoints_seq)
                    success_videos += 1

        print(f"\n{'='*60}")
        print(f"关键点提取完成!")
        print(f"  总视频数: {total_videos}")
        print(f"  成功提取: {success_videos}")
        print(f"  已跳过:   {skip_videos}")
        print(f"  失败/损坏: {failed_videos}")
        print(f"{'='*60}\n")

        return success_videos


if __name__ == "__main__":
    extractor = KeypointExtractor()
    extractor.extract_batch()
