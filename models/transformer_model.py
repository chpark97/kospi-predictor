"""Transformer 모델 - 앙상블 구성원

Positional Encoding + Multi-head Self-Attention 기반
시계열 패턴 학습. LSTM과 다른 관점으로 장기 의존성 포착.
"""
import math

import torch
import torch.nn as nn

from config.settings import ATTENTION_HEADS, DROPOUT


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding"""

    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    """Transformer 기반 시계열 예측 모델

    입력: (batch, seq_len, num_features)
    출력: pred_return (batch,), confidence (batch,)
    """

    def __init__(self, num_features, d_model=128, nhead=None, num_layers=2,
                 dim_feedforward=256, dropout=None, seq_length=20):
        super().__init__()
        nhead = nhead or ATTENTION_HEADS
        dropout = dropout or DROPOUT

        # 입력 프로젝션
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length + 10, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(d_model)

        # 출력 헤드
        self.fc_return = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.fc_confidence = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.fc_direction = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # (batch, seq_len, features) -> (batch, seq_len, d_model)
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        # Transformer Encoder
        x = self.transformer_encoder(x)
        x = self.layer_norm(x)

        # 마지막 시점 사용
        last = x[:, -1, :]

        pred_return = self.fc_return(last).squeeze(-1)
        confidence = self.fc_confidence(last).squeeze(-1)
        pred_direction = self.fc_direction(last).squeeze(-1)

        return pred_return, confidence, pred_direction
