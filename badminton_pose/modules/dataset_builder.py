# -*- coding: utf-8 -*-
"""
数据集构建模块
- 滑动窗口切割时序样本
- 分层抽样划分训练/验证/测试集
- 数据增强（关键点偏移、角度扰动、时序翻转）
- PyTorch Dataset封装
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config


class BadmintonDataset(Dataset):
    """
    羽毛球动作分类数据集
    支持训练模式下的数据增强
    """

    def __init__(self, features_list, labels, augment=False):
        """
        Args:
            features_list: list of np.ndarray, 每个 (seq_len, feature_dim)
            labels: list of int, 类别标签
            augment: 是否启用数据增强
        """
        self.features = features_list
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx].copy()
        y = self.labels[idx]

        if self.augment:
            x = self._augment(x)

        # 转为 tensor
        x = torch.FloatTensor(x)   # (seq_len, feature_dim)
        y = torch.LongTensor([y]).squeeze()
        return x, y

    def _augment(self, features):
        """
        数据增强策略

        Args:
            features: (seq_len, feature_dim)

        Returns:
            augmented: (seq_len, feature_dim)
        """
        augmented = features.copy()
        feature_dim = augmented.shape[1]

        # 1. 关键点坐标微小偏移（数据已标准化，用小扰动）
        if np.random.random() < config.AUG_PROB:
            shift = np.random.uniform(-0.1, 0.1, size=(1, config.COORD_FEATURES))
            augmented[:, :config.COORD_FEATURES] += shift

        # 2. 角度轻微扰动（数据已标准化，用小扰动）
        if np.random.random() < config.AUG_PROB:
            angle_start = config.COORD_FEATURES
            angle_end = angle_start + config.ANGLE_FEATURES
            perturb = np.random.uniform(-0.05, 0.05, size=(1, config.ANGLE_FEATURES))
            augmented[:, angle_start:angle_end] += perturb

        # 3. 时序翻转增强（时间反向播放）
        if config.AUG_TEMPORAL_FLIP and np.random.random() < config.AUG_PROB:
            augmented = augmented[::-1].copy()

        # 4. 随机帧丢弃（轻微过拟合）
        if np.random.random() < 0.3:
            drop_idx = np.random.randint(0, augmented.shape[0], size=1)
            augmented[drop_idx] = augmented[max(0, drop_idx[0] - 1)]

        return augmented


class DatasetBuilder:
    """
    数据集构建器
    从预处理好的特征文件中构建训练/验证/测试数据集
    """

    def __init__(self):
        self.seq_length = config.SEQ_LENGTH
        self.stride = config.SEQ_STRIDE
        self.feature_dim = config.TOTAL_FEATURES

    def build_from_processed(self, processed_dir=None, output_dir=None):
        """
        从预处理后的特征文件构建完整的滑动窗口数据集

        Args:
            processed_dir: 预处理特征目录
            output_dir: 数据集保存目录
        """
        processed_dir = processed_dir or os.path.join(config.KEYPOINTS_DIR, "processed")
        output_dir = output_dir or config.DATASETS_DIR
        os.makedirs(output_dir, exist_ok=True)

        # 加载原始18类目录名，并解析出原始类别名称
        raw_dirs = sorted([
            d for d in os.listdir(processed_dir)
            if os.path.isdir(os.path.join(processed_dir, d))
        ])
        # 从 "00_Short Serve" -> "Short Serve"
        original_class_names = [
            " ".join(d.split('_')[1:]) for d in raw_dirs
        ]

        all_sequences = []
        all_labels = []

        print(f"\n{'='*60}")
        print(f"开始构建滑动窗口数据集 (18类模式)")
        print(f"序列长度: {self.seq_length}, 步长: {self.stride}")
        print(f"特征维度: {self.feature_dim}")
        print(f"类别合并: 已禁用")
        print(f"{'='*60}\n")

        # 统计每个原始类的样本数
        original_class_counts = [0] * len(original_class_names)

        for i, class_name in enumerate(raw_dirs):
            class_dir = os.path.join(processed_dir, class_name)
            # 原始类别索引就是它在排序后列表中的位置
            class_idx = i

            npy_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
            file_count = 0

            for npy_file in npy_files:
                features = np.load(os.path.join(class_dir, npy_file))  # (T, 44)

                if features.shape[0] < self.seq_length:
                    continue

                # 滑动窗口切割
                sequences = self._sliding_window(features)
                for seq in sequences:
                    all_sequences.append(seq)
                    all_labels.append(class_idx)
                    original_class_counts[class_idx] += 1
                    file_count += 1

            print(f"  {class_name}: {len(npy_files)} 文件, {file_count} 样本")

        print()
        for i, name in enumerate(original_class_names):
            print(f"  [{i+1:2d}] {name}: {original_class_counts[i]} 个样本")

        print(f"\n总样本数: {len(all_sequences)}")

        # 分层抽样划分数据集
        X_train, X_val, X_test, y_train, y_val, y_test = self._stratified_split(
            all_sequences, all_labels
        )

        # 特征标准化：用训练集拟合，统一变换所有集
        # 先把所有时序展平为 (N*T, feat_dim) 做统计
        all_train_flat = np.concatenate(X_train, axis=0)  # (N_train*T, 44)
        scaler = StandardScaler()
        scaler.fit(all_train_flat)

        # 对每个样本的每帧做变换
        def scale_sequences(seqs):
            return [scaler.transform(s) for s in seqs]

        X_train = scale_sequences(X_train)
        X_val = scale_sequences(X_val)
        X_test = scale_sequences(X_test)

        print(f"\n特征标准化完成: mean≈{scaler.mean_.mean():.4f}, std≈{scaler.scale_.mean():.4f}")

        # 保存（包含scaler供推理时使用）
        dataset = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val,     'y_val': y_val,
            'X_test': X_test,   'y_test': y_test,
            'class_names': original_class_names,
            'scaler_mean': scaler.mean_.copy(),
            'scaler_scale': scaler.scale_.copy(),
        }

        save_path = os.path.join(output_dir, 'dataset.pt')
        torch.save(dataset, save_path)

        print(f"\n{'='*60}")
        print(f"数据集划分完成:")
        print(f"  训练集: {len(y_train)} 样本")
        print(f"  验证集: {len(y_val)} 样本")
        print(f"  测试集: {len(y_test)} 样本")
        print(f"保存至: {save_path}")
        print(f"{'='*60}\n")

        return dataset

    def _sliding_window(self, features):
        """
        对单个视频的特征序列做滑动窗口切割

        Args:
            features: (T, feature_dim)

        Returns:
            sequences: list of (seq_length, feature_dim)
        """
        T = features.shape[0]
        sequences = []

        for start in range(0, T - self.seq_length + 1, self.stride):
            seq = features[start:start + self.seq_length]
            sequences.append(seq)

        # 如果没有足够的帧，做padding
        if len(sequences) == 0 and T >= self.seq_length:
            sequences.append(features[:self.seq_length])

        return sequences

    def _stratified_split(self, sequences, labels):
        """
        按类别分层抽样划分数据集

        Args:
            sequences: list of np.ndarray
            labels: list of int

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        indices = list(range(len(labels)))

        # 第一次划分：训练+临时
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=(config.VAL_RATIO + config.TEST_RATIO),
            stratify=[labels[i] for i in indices],
            random_state=config.RANDOM_SEED
        )

        # 第二次划分：验证+测试
        temp_labels = [labels[i] for i in temp_idx]
        val_ratio_adjusted = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=(1 - val_ratio_adjusted),
            stratify=temp_labels,
            random_state=config.RANDOM_SEED
        )

        X_train = [sequences[i] for i in train_idx]
        X_val = [sequences[i] for i in val_idx]
        X_test = [sequences[i] for i in test_idx]
        y_train = [labels[i] for i in train_idx]
        y_val = [labels[i] for i in val_idx]
        y_test = [labels[i] for i in test_idx]

        return X_train, X_val, X_test, y_train, y_val, y_test

    @staticmethod
    def create_dataloaders(dataset_dict, batch_size=None, num_workers=None):
        """
        从数据集字典创建 PyTorch DataLoader

        Args:
            dataset_dict: 包含 X_train/y_train 等的字典
            batch_size: 批次大小
            num_workers: 工作线程数

        Returns:
            train_loader, val_loader, test_loader
        """
        bs = batch_size or config.BATCH_SIZE
        nw = num_workers or config.NUM_WORKERS

        train_ds = BadmintonDataset(
            dataset_dict['X_train'], dataset_dict['y_train'], augment=True
        )
        val_ds = BadmintonDataset(
            dataset_dict['X_val'], dataset_dict['y_val'], augment=False
        )
        test_ds = BadmintonDataset(
            dataset_dict['X_test'], dataset_dict['y_test'], augment=False
        )

        # 训练集使用加权采样器处理类别不均衡
        label_counts = np.bincount(dataset_dict['y_train'])
        class_weights = 1.0 / (label_counts + 1e-6)
        sample_weights = class_weights[dataset_dict['y_train']]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        use_pin = torch.cuda.is_available()

        train_loader = DataLoader(
            train_ds, batch_size=bs, sampler=sampler,
            num_workers=nw, pin_memory=use_pin, drop_last=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=bs, shuffle=False,
            num_workers=nw, pin_memory=use_pin
        )
        test_loader = DataLoader(
            test_ds, batch_size=bs, shuffle=False,
            num_workers=nw, pin_memory=use_pin
        )

        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.build_from_processed()