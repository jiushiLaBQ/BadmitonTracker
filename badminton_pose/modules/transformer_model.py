# -*- coding: utf-8 -*-
"""
Transformer 动作分类模型
- 基于Transformer Encoder的时序特征提取器
- 使用Positional Encoding注入时序信息
- 替换BiLSTM，旨在捕获更复杂的长距离依赖
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    为输入序列注入位置信息
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (seq_len, batch, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """
    基于Transformer Encoder的分类器
    """
    def __init__(self, input_dim, model_dim, n_heads, n_layers, num_classes, dropout=0.5):
        super(TransformerClassifier, self).__init__()
        self.model_type = 'Transformer'
        self.d_model = model_dim

        # 输入层: 将原始特征维度映射到Transformer的模型维度
        self.input_embedding = nn.Linear(input_dim, model_dim)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        
        # Transformer Encoder层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=n_heads, 
            dim_feedforward=model_dim * 4, # 惯例设置为4倍
            dropout=dropout,
            batch_first=True # 重要：输入格式为 (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier初始化"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, src):
        """
        前向传播
        Args:
            src: (batch, seq_len, feature_dim)
        Returns:
            logits: (batch, num_classes)
        """
        # 1. 输入嵌入
        src = self.input_embedding(src) * math.sqrt(self.d_model)
        
        # 2. 添加位置编码
        # TransformerEncoderLayer默认输入是(seq, batch, feature), 但我们用了batch_first=True
        # PositionalEncoding的输入是(seq, batch, feature)，所以需要转换一下
        src = src.transpose(0, 1) # (seq_len, batch, model_dim)
        src = self.pos_encoder(src)
        src = src.transpose(0, 1) # (batch, seq_len, model_dim)

        # 3. Transformer编码
        output = self.transformer_encoder(src)
        
        # 4. 全局平均池化
        # 取所有时间步输出的平均值作为整个序列的表示
        output = output.mean(dim=1)
        
        # 5. 分类
        logits = self.classifier(output)
        
        return logits

    def get_model_info(self):
        """获取模型参数量信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
        }