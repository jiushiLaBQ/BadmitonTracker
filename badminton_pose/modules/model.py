# -*- coding: utf-8 -*-
"""
BiLSTM 动作分类模型
- 多层双向LSTM编码时序特征
- 全连接分类头 + BatchNorm + Dropout
- 支持多类别羽毛球动作分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class AttentionPooling(nn.Module):
    """
    可学习的注意力池化层
    替代均值池化，让模型自动学习哪些帧对分类最重要
    """

    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_size)

        Returns:
            pooled: (batch, hidden_size)
        """
        # 计算注意力权重
        attn_weights = self.attention(x)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # (batch, seq_len, 1)

        # 加权求和
        pooled = (x * attn_weights).sum(dim=1)  # (batch, hidden_size)
        return pooled, attn_weights.squeeze(-1)


class BiLSTMClassifier(nn.Module):
    """
    深层双向BiLSTM + 全连接分类头

    结构:
        Input (batch, seq_len, feature_dim)
        → BiLSTM Layer 1 (hidden_size) + BatchNorm + Dropout
        → BiLSTM Layer 2 (hidden_size) + BatchNorm + Dropout
        → BiLSTM Layer 3 (hidden_size) + Dropout
        → 取最后时间步输出
        → FC (hidden_size*2 → fc_hidden) + ReLU + BatchNorm + Dropout
        → FC (fc_hidden → num_classes)
    """

    def __init__(self, input_dim=None, hidden_size=None, num_layers=None,
                 num_classes=None, dropout=None, fc_hidden=None, use_bn=None):
        super(BiLSTMClassifier, self).__init__()

        self.input_dim = input_dim or config.TOTAL_FEATURES
        self.hidden_size = hidden_size or config.LSTM_HIDDEN_SIZE
        self.num_layers = num_layers or config.LSTM_NUM_LAYERS
        self.num_classes = num_classes or config.NUM_CLASSES
        self.dropout = dropout or config.DROPOUT_RATE
        self.fc_hidden = fc_hidden or config.FC_HIDDEN_SIZE
        self.use_bn = use_bn if use_bn is not None else config.USE_BATCHNORM

        # LSTM输入层归一化
        self.input_bn = nn.BatchNorm1d(self.input_dim)

        # 多层双向LSTM
        lstm_layers = []
        for i in range(self.num_layers):
            input_size = self.input_dim if i == 0 else self.hidden_size * 2
            lstm_layers.append(
                nn.LSTM(
                    input_size=input_size,
                    hidden_size=self.hidden_size,
                    bidirectional=True,
                    batch_first=True,
                    dropout=0  # 手动堆叠LSTM层，由外部Dropout控制
                )
            )
        self.lstm_layers = nn.ModuleList(lstm_layers)

        # 每层LSTM后的BatchNorm和Dropout
        self.lstm_bns = nn.ModuleList([
            nn.BatchNorm1d(self.hidden_size * 2) for _ in range(self.num_layers)
        ]) if self.use_bn else None

        self.lstm_dropouts = nn.ModuleList([
            nn.Dropout(self.dropout) for _ in range(self.num_layers)
        ])

        # 新增：多头自注意力层，增强时序特征捕捉
        embed_dim = self.hidden_size * 2
        num_heads = config.ATTN_NUM_HEADS
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) 必须能被 num_heads ({num_heads}) 整除！"

        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(self.hidden_size * 2)

        # 注意力池化层（替代均值池化）
        self.attn_pool = AttentionPooling(self.hidden_size * 2)

        # 全连接分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.fc_hidden),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.fc_hidden) if self.use_bn else nn.Identity(),
            nn.Dropout(self.dropout),
            nn.Linear(self.fc_hidden, self.fc_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout / 2),
            nn.Linear(self.fc_hidden // 2, self.num_classes)
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """Xavier初始化"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        """
        前向传播

        Args:
            x: (batch, seq_len, feature_dim)

        Returns:
            logits: (batch, num_classes)
        """
        batch_size, seq_len, feat_dim = x.shape

        # 输入层BatchNorm
        # 将 (batch, seq_len, feat_dim) reshape 为 (batch*seq_len, feat_dim)
        x_reshaped = x.reshape(-1, feat_dim)
        x_reshaped = self.input_bn(x_reshaped)
        x = x_reshaped.reshape(batch_size, seq_len, feat_dim)

        # 逐层LSTM
        for i, lstm in enumerate(self.lstm_layers):
            x, _ = lstm(x)  # (batch, seq_len, hidden*2)

            if self.lstm_bns is not None:
                # BatchNorm: (batch, seq_len, hidden*2) → permute → bn → permute
                x = x.permute(0, 2, 1)  # (batch, hidden*2, seq_len)
                x = self.lstm_bns[i](x)
                x = x.permute(0, 2, 1)  # (batch, seq_len, hidden*2)

            x = self.lstm_dropouts[i](x)

        # 新增：自注意力模块 + 残差连接
        attn_input = x
        attn_output, _ = self.self_attn(attn_input, attn_input, attn_input)
        x = self.attn_norm(attn_input + attn_output) # 残差连接

        # 注意力池化（学习哪些帧对分类最重要）
        x, attn_weights = self.attn_pool(x)  # (batch, hidden*2), (batch, seq_len)

        # 全连接分类
        logits = self.classifier(x)  # (batch, num_classes)

        # 在训练时只返回logits，在评估/推理时可以返回更多信息
        if self.training:
            return logits
        else:
            return logits, attn_weights

    def get_model_info(self):
        """获取模型参数量信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'input_dim': self.input_dim,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
        }


def build_model(num_classes=None):
    """
    根据配置构建并返回模型实例

    Args:
        num_classes: 类别数，None则从config获取

    Returns:
        model: nn.Module 实例
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES

    model_type = config.MODEL_TYPE.lower()
    print(f"[Model Builder] 构建模型: {model_type}")

    if model_type == 'bilstm':
        model = BiLSTMClassifier(
            input_dim=config.TOTAL_FEATURES,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_NUM_LAYERS,
            num_classes=num_classes,
            dropout=config.DROPOUT_RATE,
            fc_hidden=config.FC_HIDDEN_SIZE,
            use_bn=config.USE_BATCHNORM
        )
    elif model_type == 'transformer':
        # 动态导入，避免不使用时也加载
        from .transformer_model import TransformerClassifier
        model = TransformerClassifier(
            input_dim=config.TOTAL_FEATURES,
            model_dim=config.TRANSFORMER_MODEL_DIM,
            n_heads=config.TRANSFORMER_N_HEADS,
            n_layers=config.TRANSFORMER_N_LAYERS,
            num_classes=num_classes,
            dropout=config.DROPOUT_RATE
        )
    else:
        raise ValueError(f"未知的模型类型: {config.MODEL_TYPE}. "
                         f"请在 'BiLSTM' 或 'Transformer' 中选择。")

    return model


if __name__ == "__main__":
    config.load_class_names()
    model = build_model()
    info = model.get_model_info()
    print(f"模型参数总量: {info['total_params']:,}")
    print(f"可训练参数量: {info['trainable_params']:,}")

    # 测试前向传播
    dummy = torch.randn(4, config.SEQ_LENGTH, config.TOTAL_FEATURES)
    output = model(dummy)
    print(f"输入形状: {dummy.shape}")
    print(f"输出形状: {output.shape}")