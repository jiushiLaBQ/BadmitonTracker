# -*- coding: utf-8 -*-
"""
训练脚本
- AdamW优化器 + 余弦退火学习率调度
- 早停策略 + 保存最优模型
- 训练/验证损失与准确率曲线绘制
- 支持GPU/CPU自适应训练
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from modules.model import build_model
from modules.dataset_builder import DatasetBuilder


# 中文字体设置
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False


class Trainer:
    """
    模型训练器
    """

    def __init__(self):
        # 设备选择
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Trainer] 使用设备: {self.device}")
        if self.device.type == 'cuda':
            print(f"[Trainer] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[Trainer] 显存: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

        # 加载数据集
        dataset_path = os.path.join(config.DATASETS_DIR, 'dataset.pt')
        if not os.path.exists(dataset_path):
            print("[ERROR] 数据集不存在，请先运行 build_dataset.py")
            sys.exit(1)

        print(f"[Trainer] 加载数据集: {dataset_path}")
        self.dataset = torch.load(dataset_path, weights_only=False)
        config.NUM_CLASSES = len(self.dataset['class_names'])
        config.CLASS_NAMES = self.dataset['class_names']
        print(f"[Trainer] 类别数: {config.NUM_CLASSES}")
        print(f"[Trainer] 训练样本: {len(self.dataset['y_train'])}")
        print(f"[Trainer] 验证样本: {len(self.dataset['y_val'])}")
        print(f"[Trainer] 测试样本: {len(self.dataset['y_test'])}")

        # 创建DataLoader
        self.train_loader, self.val_loader, self.test_loader = \
            DatasetBuilder.create_dataloaders(self.dataset)

        # 构建模型
        self.model = build_model().to(self.device)
        info = self.model.get_model_info()
        print(f"[Trainer] 模型参数量: {info['total_params']:,}")

        # 损失函数 - CrossEntropy + 标签平滑（软化硬标签，缓解类别不平衡的过拟合）
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # 学习率调度器 - 线性warmup + 余弦退火
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=config.WARMUP_EPOCHS
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.NUM_EPOCHS - config.WARMUP_EPOCHS,
            eta_min=config.MIN_LR
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.WARMUP_EPOCHS]
        )

        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.lr_history = []
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

    def train_one_epoch(self, epoch):
        """
        训练一个epoch

        Args:
            epoch: 当前epoch编号

        Returns:
            avg_loss, accuracy
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_idx, (data, labels) in enumerate(self.train_loader):
            data = data.to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            logits = self.model(data)
            loss = self.criterion(logits, labels)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * data.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(all_labels)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self, data_loader):
        """
        在给定数据集上验证

        Args:
            data_loader: 数据加载器

        Returns:
            avg_loss, accuracy, all_preds, all_labels
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for data, labels in data_loader:
            data = data.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(data)
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * data.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(all_labels)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)

    def train(self):
        """
        完整训练流程
        """
        print(f"\n{'='*60}")
        print(f"开始训练 | 设备: {self.device}")
        print(f"最大轮数: {config.NUM_EPOCHS} | 早停耐心: {config.EARLY_STOP_PATIENCE}")
        print(f"{'='*60}\n")

        for epoch in range(1, config.NUM_EPOCHS + 1):
            start_time = time.time()

            # 训练
            train_loss, train_acc = self.train_one_epoch(epoch)

            # 验证
            val_loss, val_acc, _, _ = self.validate(self.val_loader)

            # 学习率调度
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            self.lr_history.append(current_lr)

            elapsed = time.time() - start_time

            # 打印进度
            print(f"Epoch [{epoch:3d}/{config.NUM_EPOCHS}] | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s")

            # 保存最优模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_acc)
                print(f"  >>> 新最优模型! Val Acc: {val_acc:.4f}")
            else:
                self.patience_counter += 1

            # 早停检查
            if self.patience_counter >= config.EARLY_STOP_PATIENCE:
                print(f"\n早停触发! {config.EARLY_STOP_PATIENCE}轮无改善")
                print(f"最优轮次: Epoch {self.best_epoch}, Val Acc: {self.best_val_acc:.4f}")
                break

        # 绘制训练曲线
        self._plot_curves()

        # 最终测试集评估
        self._final_test()

    def _save_checkpoint(self, epoch, val_acc):
        """保存模型检查点"""
        save_path = os.path.join(config.MODELS_DIR, 'best_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'class_names': config.CLASS_NAMES,
            'num_classes': config.NUM_CLASSES,
            'config': {
                'input_dim': config.TOTAL_FEATURES,
                'hidden_size': config.LSTM_HIDDEN_SIZE,
                'num_layers': config.LSTM_NUM_LAYERS,
                'dropout': config.DROPOUT_RATE,
                'fc_hidden': config.FC_HIDDEN_SIZE,
            }
        }, save_path)

    def _plot_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        epochs = range(1, len(self.train_losses) + 1)

        # 损失曲线
        axes[0].plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 准确率曲线
        axes[1].plot(epochs, self.train_accs, 'b-', label='Train Acc', linewidth=2)
        axes[1].plot(epochs, self.val_accs, 'r-', label='Val Acc', linewidth=2)
        axes[1].axhline(y=0.85, color='g', linestyle='--', alpha=0.5, label='Target: 85%')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training & Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 学习率曲线
        axes[2].plot(epochs, self.lr_history, 'g-', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(config.CURVES_DIR, 'training_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n训练曲线已保存: {save_path}")

    def _final_test(self):
        """最终测试集评估"""
        # 加载最优模型
        best_path = os.path.join(config.MODELS_DIR, 'best_model.pth')
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"\n加载最优模型 (Epoch {checkpoint['epoch']}, "
                  f"Val Acc: {checkpoint['val_acc']:.4f})")

        test_loss, test_acc, test_preds, test_labels = self.validate(self.test_loader)

        print(f"\n{'='*60}")
        print(f"测试集评估结果:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"{'='*60}\n")

        # 分类报告
        print("分类报告:")
        print(classification_report(
            test_labels, test_preds,
            target_names=config.CLASS_NAMES,
            digits=4,
            zero_division=0
        ))


def main():
    """主训练入口"""
    config.load_class_names()
    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    main()