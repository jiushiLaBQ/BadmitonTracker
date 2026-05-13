# -*- coding: utf-8 -*-
"""
评估脚本
- 测试集准确率评估
- 混淆矩阵绘制
- 每类别精度/召回率/F1分析
- 分类报告导出
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score
)
import seaborn as sns

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from modules.model import build_model
from modules.dataset_builder import DatasetBuilder

# 中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False


class Evaluator:
    """
    模型评估器
    """

    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载数据集
        dataset_path = os.path.join(config.DATASETS_DIR, 'dataset.pt')
        self.dataset = torch.load(dataset_path, weights_only=False)
        config.CLASS_NAMES = self.dataset['class_names']
        config.NUM_CLASSES = len(config.CLASS_NAMES)

        # 创建DataLoader
        _, _, self.test_loader = DatasetBuilder.create_dataloaders(self.dataset)

        # 加载模型
        model_path = model_path or os.path.join(config.MODELS_DIR, 'best_model.pth')
        if not os.path.exists(model_path):
            print(f"[ERROR] 模型文件不存在: {model_path}")
            sys.exit(1)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # 使用 checkpoint 中保存的模型配置构建模型（避免架构不匹配）
        from modules.model import BiLSTMClassifier
        saved_cfg = checkpoint.get('config', {})
        self.model = BiLSTMClassifier(
            input_dim=saved_cfg.get('input_dim', config.TOTAL_FEATURES),
            hidden_size=saved_cfg.get('hidden_size', config.LSTM_HIDDEN_SIZE),
            num_layers=saved_cfg.get('num_layers', config.LSTM_NUM_LAYERS),
            num_classes=checkpoint.get('num_classes', config.NUM_CLASSES),
            dropout=saved_cfg.get('dropout', config.DROPOUT_RATE),
            fc_hidden=saved_cfg.get('fc_hidden', config.FC_HIDDEN_SIZE),
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.class_names = checkpoint.get('class_names', config.CLASS_NAMES)
        self.best_epoch = checkpoint.get('epoch', '?')
        self.best_val_acc = checkpoint.get('val_acc', 0.0)

        print(f"[Evaluator] 加载模型: {model_path}")
        print(f"[Evaluator] 最优轮次: Epoch {self.best_epoch}")
        print(f"[Evaluator] 验证集准确率: {self.best_val_acc:.4f}")
        print(f"[Evaluator] 类别数: {len(self.class_names)}")

    @torch.no_grad()
    def evaluate(self, use_tta=True, tta_n=5):
        """
        在测试集上进行全面评估

        Args:
            use_tta: 是否启用测试时增强
            tta_n: TTA增强次数

        Returns:
            评估结果字典
        """
        all_preds = []
        all_labels = []
        all_probs = []

        for data, labels in self.test_loader:
            data = data.to(self.device)

            if use_tta:
                # 测试时增强：多次加噪取平均概率
                batch_probs = []
                # 原始预测
                logits = self.model(data)
                batch_probs.append(torch.softmax(logits, dim=1))
                # 多次加噪预测
                for _ in range(tta_n - 1):
                    noise = torch.randn_like(data) * 0.05
                    noisy_data = data + noise
                    logits_noisy = self.model(noisy_data)
                    batch_probs.append(torch.softmax(logits_noisy, dim=1))
                # 平均概率
                probs = torch.stack(batch_probs).mean(dim=0)
            else:
                logits = self.model(data)
                probs = torch.softmax(logits, dim=1)

            preds = probs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # 计算各项指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
        }

        # 打印结果
        print(f"\n{'='*60}")
        tta_str = f" (TTA x{tta_n})" if use_tta else ""
        print(f"测试集评估结果{tta_str}")
        print(f"{'='*60}")
        print(f"  准确率 (Accuracy):  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  精确率 (Precision): {precision:.4f}")
        print(f"  召回率 (Recall):    {recall:.4f}")
        print(f"  F1分数 (F1-Score):  {f1:.4f}")
        print(f"{'='*60}\n")

        # 分类报告
        report = classification_report(
            all_labels, all_preds,
            target_names=self.class_names,
            digits=4,
            zero_division=0
        )
        print("详细分类报告:")
        print(report)

        return results

    def plot_confusion_matrix(self, results=None):
        """
        绘制混淆矩阵

        Args:
            results: 评估结果字典，None则自动评估
        """
        if results is None:
            results = self.evaluate()

        cm = confusion_matrix(results['labels'], results['predictions'])
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        # 计算每个类别的准确率
        per_class_acc = np.diag(cm_normalized)

        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))

        # 原始计数矩阵
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=axes[0]
        )
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=14)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].tick_params(axis='y', rotation=0)

        # 归一化矩阵
        sns.heatmap(
            cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=axes[1],
            vmin=0, vmax=1
        )
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        axes[1].set_ylabel('True Label', fontsize=12)
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].tick_params(axis='y', rotation=0)

        plt.suptitle(
            f'Test Accuracy: {results["accuracy"]*100:.2f}% | '
            f'F1: {results["f1_score"]:.4f} | '
            f'Epoch: {self.best_epoch}',
            fontsize=16, fontweight='bold'
        )
        plt.tight_layout()

        save_path = os.path.join(config.CURVES_DIR, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"混淆矩阵已保存: {save_path}")

        # 绘制每类别准确率柱状图
        self._plot_per_class_accuracy(per_class_acc)

    def _plot_per_class_accuracy(self, per_class_acc):
        """绘制每类别准确率柱状图"""
        fig, ax = plt.subplots(figsize=(14, 6))

        colors = ['#e74c3c' if acc < 0.7 else '#f39c12' if acc < 0.85 else '#27ae60'
                  for acc in per_class_acc]

        bars = ax.bar(range(len(self.class_names)), per_class_acc, color=colors, edgecolor='white')
        ax.set_xticks(range(len(self.class_names)))
        ax.set_xticklabels(self.class_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Per-Class Accuracy', fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.85, color='g', linestyle='--', alpha=0.5, label='Target: 85%')
        ax.legend()

        # 在柱子上标数值
        for bar, acc in zip(bars, per_class_acc):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom', fontsize=8)

        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(config.CURVES_DIR, 'per_class_accuracy.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"每类别准确率图已保存: {save_path}")

    def plot_top_errors(self, results=None, top_k=5):
        """
        绘制最容易混淆的类别对

        Args:
            results: 评估结果
            top_k: 展示前K个最容易混淆的对
        """
        if results is None:
            results = self.evaluate()

        cm = confusion_matrix(results['labels'], results['predictions'])

        # 找到非对角线最大值
        cm_no_diag = cm.copy()
        np.fill_diagonal(cm_no_diag, 0)

        error_pairs = []
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j and cm_no_diag[i, j] > 0:
                    error_pairs.append((i, j, cm_no_diag[i, j], cm[i, i]))

        error_pairs.sort(key=lambda x: x[2], reverse=True)

        print(f"\nTop-{top_k} 最容易混淆的类别对:")
        print("-" * 50)
        for idx, (true_idx, pred_idx, count, true_total) in enumerate(error_pairs[:top_k]):
            pct = f"{count/true_total*100:.1f}%" if true_total > 0 else "N/A"
            print(f"  {idx+1}. {self.class_names[true_idx]} → "
                  f"{self.class_names[pred_idx]}: {count} 次 "
                  f"(占该类 {pct})")


def main():
    """评估主入口"""
    config.load_class_names()
    evaluator = Evaluator()
    results = evaluator.evaluate()
    evaluator.plot_confusion_matrix(results)
    evaluator.plot_top_errors(results)


if __name__ == "__main__":
    main()
