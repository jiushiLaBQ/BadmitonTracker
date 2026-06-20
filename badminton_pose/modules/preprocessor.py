# -*- coding: utf-8 -*-
"""
姿态降噪与特征工程模块
- 关键点平滑滤波（中值滤波+线性插值）
- 异常点检测与修复
- 关节夹角计算
- 全局坐标归一化
"""

import os
import numpy as np
from scipy.signal import medfilt
from scipy.interpolate import interp1d
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class Preprocessor:
    """
    姿态数据预处理器
    输入: (T, 17, 3) 的关键点序列
    输出: (T', 128) 的归一化特征序列 (34坐标 + 10角度 + 34速度 + 34加速度 + 10角速度 + 6几何)
    """

    def __init__(self):
        self.smooth_window = config.SMOOTH_WINDOW
        self.anomaly_threshold = config.ANOMALY_THRESHOLD
        self.joint_angles = config.JOINT_ANGLES

    def process_sequence(self, keypoints_seq):
        """
        完整预处理流水线

        Args:
            keypoints_seq: np.ndarray, shape (T, 17, 3)
                           T帧，每帧17个关键点 (x, y, conf)

        Returns:
            features: np.ndarray, shape (T', 128)
                      归一化特征序列 (34坐标 + 10角度 + 34速度 + 34加速度 + 10角速度 + 6几何)
        """
        if len(keypoints_seq) < 4:
            return np.array([])

        # Step 1: 平滑滤波
        smoothed = self._smooth_filter(keypoints_seq)

        # Step 2: 异常点检测与插值修复
        repaired = self._anomaly_repair(smoothed)

        # Step 3: 坐标归一化（消除尺度和位置影响）
        normalized_coords = self._normalize_coordinates(repaired)

        # Step 4: 计算关节夹角
        angles = self._compute_angles(repaired)

        # Step 5: 计算坐标速度特征（帧间差分）
        velocity = self._compute_velocity(normalized_coords)

        # Step 6: 计算加速度特征（二阶差分）
        acceleration = self._compute_acceleration(normalized_coords)

        # Step 7: 计算角速度（角度的帧间差分）
        angular_velocity = self._compute_angular_velocity(angles)

        # Step 8: 计算几何特征（扭转、站姿、肢体偏移等）
        geometric = self._compute_geometric_features(repaired)

        # Step 9: 拼接所有特征
        features = np.concatenate([
            normalized_coords,    # 34
            angles,               # 10
            velocity,             # 34
            acceleration,         # 34
            angular_velocity,     # 10
            geometric,            # 6
        ], axis=1)

        return features  # (T, 128)

    def _smooth_filter(self, keypoints_seq):
        """
        对每个关键点的x, y坐标分别做中值滤波降噪

        Args:
            keypoints_seq: (T, 17, 3)

        Returns:
            smoothed: (T, 17, 3)
        """
        T = keypoints_seq.shape[0]
        smoothed = keypoints_seq.copy()

        # 中值滤波窗口必须为奇数
        win = self.smooth_window
        if win % 2 == 0:
            win += 1
        if T < win:
            return smoothed

        for kpt_idx in range(17):
            for coord_idx in range(2):  # x, y
                series = keypoints_seq[:, kpt_idx, coord_idx]
                # 仅对非零值做滤波（0表示未检测到）
                valid_mask = series != 0
                if valid_mask.sum() > win:
                    filtered = medfilt(series[valid_mask], kernel_size=win)
                    smoothed[valid_mask, kpt_idx, coord_idx] = filtered

        return smoothed

    def _anomaly_repair(self, keypoints_seq):
        """
        检测并修复异常跳变的关键点
        相邻帧位移超过阈值的点视为异常，用线性插值替代

        Args:
            keypoints_seq: (T, 17, 3)

        Returns:
            repaired: (T, 17, 3)
        """
        T = keypoints_seq.shape[0]
        repaired = keypoints_seq.copy()

        if T < 3:
            return repaired

        for kpt_idx in range(17):
            for coord_idx in range(2):
                series = repaired[:, kpt_idx, coord_idx].copy()
                valid_mask = series != 0

                if valid_mask.sum() < 3:
                    continue

                # 计算相邻帧差值
                diff = np.abs(np.diff(series))
                # 找到异常跳变的索引
                anomaly_idx = np.where(diff > self.anomaly_threshold)[0]

                for idx in anomaly_idx:
                    if not valid_mask[idx] or not valid_mask[idx + 1]:
                        continue
                    # 用前后有效值做线性插值
                    repaired[idx + 1, kpt_idx, coord_idx] = (
                        repaired[idx, kpt_idx, coord_idx]
                    )

        return repaired

    def _normalize_coordinates(self, keypoints_seq):
        """
        全局坐标归一化：
        1. 以躯干中心为原点平移（消除位置影响）
        2. 以躯干长度为基准缩放（消除尺度影响）
        跳过零值关键点（历史数据中低置信度点被置零的情况）

        Args:
            keypoints_seq: (T, 17, 3)

        Returns:
            normalized: (T, 34)  17个关键点的归一化x,y坐标
        """
        T = keypoints_seq.shape[0]
        coords = keypoints_seq[:, :, :2].copy()  # (T, 17, 2)

        normalized = np.zeros((T, 17 * 2))

        for t in range(T):
            frame_coords = coords[t]  # (17, 2)

            # 躯干中心：用有效（非零）的肩和髋计算
            ref_indices = [5, 6, 11, 12]  # 左肩、右肩、左髋、右髋
            ref_points = []
            for idx in ref_indices:
                if frame_coords[idx, 0] != 0 or frame_coords[idx, 1] != 0:
                    ref_points.append(frame_coords[idx])

            if len(ref_points) >= 2:
                ref_points = np.array(ref_points)
                body_center = ref_points.mean(axis=0)
            else:
                # 所有参考点都为零，用全部非零点的中心
                nonzero = frame_coords[(frame_coords[:, 0] != 0) | (frame_coords[:, 1] != 0)]
                if len(nonzero) > 0:
                    body_center = nonzero.mean(axis=0)
                else:
                    body_center = np.array([0.0, 0.0])

            # 平移：以躯干中心为原点
            centered = frame_coords - body_center

            # 缩放：以躯干长度为基准（用有效点计算）
            shoulder_pts = [frame_coords[i] for i in [5, 6]
                           if frame_coords[i, 0] != 0 or frame_coords[i, 1] != 0]
            hip_pts = [frame_coords[i] for i in [11, 12]
                      if frame_coords[i, 0] != 0 or frame_coords[i, 1] != 0]

            torso_length = 0.0
            if len(shoulder_pts) >= 1 and len(hip_pts) >= 1:
                sc = np.array(shoulder_pts).mean(axis=0)
                hc = np.array(hip_pts).mean(axis=0)
                torso_length = np.linalg.norm(sc - hc)

            if torso_length > 1e-6:
                centered = centered / torso_length

            normalized[t] = centered.flatten()

        return normalized  # (T, 34)

    def _compute_angles(self, keypoints_seq):
        """
        计算所有定义的关节夹角

        Args:
            keypoints_seq: (T, 17, 3)

        Returns:
            angles: (T, num_angles)  归一化的角度特征
        """
        T = keypoints_seq.shape[0]
        num_angles = len(self.joint_angles)
        angles = np.zeros((T, num_angles))

        coords = keypoints_seq[:, :, :2]  # (T, 17, 2)

        angle_names = list(self.joint_angles.keys())

        for t in range(T):
            for i, name in enumerate(angle_names):
                vertex_idx, p1_idx, p2_idx = self.joint_angles[name]

                vertex = coords[t, vertex_idx]
                p1 = coords[t, p1_idx]
                p2 = coords[t, p2_idx]

                # 计算角度
                v1 = p1 - vertex
                v2 = p2 - vertex

                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)

                if norm1 < 1e-6 or norm2 < 1e-6:
                    angles[t, i] = 0.0
                    continue

                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle_rad = np.arccos(cos_angle)

                # 归一化到 [0, 1]（角度范围 0~π → 0~1）
                angles[t, i] = angle_rad / np.pi

        return angles  # (T, 10)

    def _compute_velocity(self, normalized_coords):
        """
        计算归一化坐标的帧间速度（差分特征）
        捕捉动作的动态变化信息，对区分相似姿态但不同节奏的动作很有帮助

        Args:
            normalized_coords: (T, 34) 归一化后的坐标

        Returns:
            velocity: (T, 34) 每帧的速度特征（第一帧用零填充）
        """
        T = normalized_coords.shape[0]
        velocity = np.zeros_like(normalized_coords)

        if T < 2:
            return velocity

        # 帧间差分: v[t] = x[t] - x[t-1]
        velocity[1:] = normalized_coords[1:] - normalized_coords[:-1]

        return velocity  # (T, 34)

    def _compute_acceleration(self, normalized_coords):
        """
        计算归一化坐标的加速度（二阶差分）
        捕捉动作节奏变化：启动、制动、挥拍加速等

        Args:
            normalized_coords: (T, 34)

        Returns:
            acceleration: (T, 34) 前两帧用零填充
        """
        T = normalized_coords.shape[0]
        acceleration = np.zeros_like(normalized_coords)

        if T < 3:
            return acceleration

        # 二阶差分: a[t] = x[t+1] - 2*x[t] + x[t-1]
        acceleration[1:-1] = normalized_coords[2:] - 2 * normalized_coords[1:-1] + normalized_coords[:-2]

        return acceleration  # (T, 34)

    def _compute_angular_velocity(self, angles):
        """
        计算关节夹角的角速度（角度的帧间差分）
        区分快速挥拍（smash肘关节高速转动）vs 慢速控制（clear的平缓运动）

        Args:
            angles: (T, 10) 归一化角度

        Returns:
            angular_velocity: (T, 10) 第一帧用零填充
        """
        T = angles.shape[0]
        angular_velocity = np.zeros_like(angles)

        if T < 2:
            return angular_velocity

        angular_velocity[1:] = angles[1:] - angles[:-1]

        return angular_velocity  # (T, 10)

    def _compute_geometric_features(self, keypoints_seq):
        """
        计算领域相关的几何特征

        Args:
            keypoints_seq: (T, 17, 3) 原始关键点（未归一化）

        Returns:
            geometric: (T, 6) 包含:
                [0] 髋肩扭转角 (hip-shoulder twist)
                [1] 站姿宽度 (stance width / torso_length)
                [2] 左腕-肩垂直偏移
                [3] 右腕-肩垂直偏移
                [4] 左前臂向量x分量（归一化）
                [5] 右前臂向量x分量（归一化）
        """
        T = keypoints_seq.shape[0]
        coords = keypoints_seq[:, :, :2]  # (T, 17, 2)
        geometric = np.zeros((T, 6))

        for t in range(T):
            left_shoulder = coords[t, 5]
            right_shoulder = coords[t, 6]
            left_hip = coords[t, 11]
            right_hip = coords[t, 12]
            left_elbow = coords[t, 7]
            right_elbow = coords[t, 8]
            left_wrist = coords[t, 9]
            right_wrist = coords[t, 10]
            left_ankle = coords[t, 15]
            right_ankle = coords[t, 16]

            shoulder_vec = right_shoulder - left_shoulder
            hip_vec = right_hip - left_hip

            # 躯干长度（用于归一化）
            shoulder_center = (left_shoulder + right_shoulder) / 2.0
            hip_center = (left_hip + right_hip) / 2.0
            torso_length = np.linalg.norm(shoulder_center - hip_center)
            if torso_length < 1e-6:
                torso_length = 1.0

            # [0] 髋肩扭转角：肩线和髋线的方向差
            if np.linalg.norm(shoulder_vec) > 1e-6 and np.linalg.norm(hip_vec) > 1e-6:
                cos_twist = np.dot(shoulder_vec, hip_vec) / (
                    np.linalg.norm(shoulder_vec) * np.linalg.norm(hip_vec)
                )
                cos_twist = np.clip(cos_twist, -1.0, 1.0)
                geometric[t, 0] = np.arccos(cos_twist) / np.pi  # 归一化到[0,1]
            else:
                geometric[t, 0] = 0.0

            # [1] 站姿宽度：两踝水平距离 / 躯干长度
            ankle_dist = abs(left_ankle[0] - right_ankle[0])
            geometric[t, 1] = ankle_dist / torso_length

            # [2-3] 腕-肩垂直偏移（归一化）：正值=手腕低于肩膀，负值=高于（挥拍/过顶）
            geometric[t, 2] = (left_wrist[1] - left_shoulder[1]) / torso_length
            geometric[t, 3] = (right_wrist[1] - right_shoulder[1]) / torso_length

            # [4-5] 前臂向量x分量（归一化）：捕捉手臂水平伸展程度
            left_forearm = left_wrist - left_elbow
            right_forearm = right_wrist - right_elbow
            left_forearm_len = np.linalg.norm(left_forearm)
            right_forearm_len = np.linalg.norm(right_forearm)
            geometric[t, 4] = left_forearm[0] / (left_forearm_len + 1e-6)
            geometric[t, 5] = right_forearm[0] / (right_forearm_len + 1e-6)

        return geometric  # (T, 6)

    def process_batch(self, keypoints_dir=None, output_dir=None):
        """
        批量预处理所有提取好的关键点数据

        Args:
            keypoints_dir: 关键点numpy文件目录
            output_dir: 预处理后特征保存目录
        """
        keypoints_dir = keypoints_dir or config.KEYPOINTS_DIR
        output_dir = output_dir or config.KEYPOINTS_DIR
        processed_dir = os.path.join(output_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        class_dirs = sorted([
            d for d in os.listdir(keypoints_dir)
            if os.path.isdir(os.path.join(keypoints_dir, d)) and d != "processed"
        ])

        total = 0
        success = 0

        print(f"\n{'='*60}")
        print(f"开始特征预处理（平滑+归一化+角度计算）")
        print(f"{'='*60}\n")

        for class_name in class_dirs:
            class_in = os.path.join(keypoints_dir, class_name)
            class_out = os.path.join(processed_dir, class_name)
            os.makedirs(class_out, exist_ok=True)

            npy_files = [f for f in os.listdir(class_in) if f.endswith('.npy')]

            for npy_file in npy_files:
                total += 1
                out_path = os.path.join(class_out, npy_file)

                if os.path.exists(out_path):
                    continue

                kpts = np.load(os.path.join(class_in, npy_file))
                features = self.process_sequence(kpts)

                if features.shape[0] >= config.SEQ_LENGTH:
                    np.save(out_path, features)
                    success += 1

        print(f"\n预处理完成: 总计 {total} 个文件, 成功 {success} 个\n")
        return success


if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.process_batch()
