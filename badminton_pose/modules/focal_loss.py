# -*- coding: utf-8 -*-
"""
Focal Loss - 焦点损失函数
用于解决类别不平衡问题，特别是当简单样本数量远超困难样本时。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss 实现

    Args:
        alpha (float, optional): 类别权重因子，用于平衡正负样本的重要性。
                                 对于类别i，alpha值为alpha[i]，否则为1-alpha[i]。
                                 默认为0.25。
        gamma (float, optional): 聚焦参数，用于调节简单样本的权重。
                                 gamma越大，对简单样本的抑制越强。
                                 默认为2.0。
        reduction (str, optional): 损失计算方式，'mean', 'sum', 'none'。
                                   默认为 'mean'。
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        前向传播

        Args:
            inputs: (N, C), 模型的原始输出 (logits)
            targets: (N), 真实标签
        """
        # 计算标准的交叉熵损失，但不进行reduction
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 计算pt，即模型对正确类别的预测概率
        pt = torch.exp(-ce_loss)

        # 计算Focal Loss
        # alpha_t 是对应类别的alpha权重
        if isinstance(self.alpha, (float, int)):
            alpha_t = self.alpha
        else:
            # 如果alpha是list或tensor，则根据target选择
            alpha_t = self.alpha[targets]
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss