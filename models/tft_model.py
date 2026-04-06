"""Temporal Fusion Transformer (TFT) 경량 구현

정적 피처(요일, 월, 이벤트)와 동적 피처(가격, 지수)를 분리 처리.
GRN(Gated Residual Network) + Attention 기반 해석 가능한 예측.
"""
import math

import torch
import torch.nn as nn

from config.settings import DROPOUT


class GatedResidualNetwork(nn.Module):
    """GRN: TFT의 핵심 빌딩 블록"""

    def __init__(self, d_input, d_hidden, d_output=None, dropout=0.1):
        super().__init__()
        d_output = d_output or d_input
        self.fc1 = nn.Linear(d_input, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_output)
        self.gate = nn.Linear(d_hidden, d_output)
        self.layer_norm = nn.LayerNorm(d_output)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(d_input, d_output) if d_input != d_output else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        h = torch.relu(self.fc1(x))
        h = self.dropout(h)
        out = self.fc2(h)
        gate = torch.sigmoid(self.gate(h))
        return self.layer_norm(residual + gate * out)


class TemporalFusionTransformer(nn.Module):
    """경량 TFT 모델

    구조:
    1. 입력 프로젝션 (GRN)
    2. LSTM 인코더
    3. Multi-head Attention (해석 가능)
    4. GRN 디코더
    5. 예측 헤드
    """

    def __init__(self, num_features, d_model=64, nhead=4, num_layers=1,
                 dropout=None, seq_length=20):
        super().__init__()
        dropout = dropout or DROPOUT

        # 입력 프로젝션 (GRN)
        self.input_grn = GatedResidualNetwork(num_features, d_model * 2, d_model, dropout)

        # LSTM 인코더
        self.lstm = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)

        # Self-Attention
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)

        # 디코더 GRN
        self.decoder_grn = GatedResidualNetwork(d_model, d_model * 2, d_model, dropout)

        # 출력
        self.fc_return = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.fc_confidence = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.fc_direction = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # (batch, seq, features) → GRN → (batch, seq, d_model)
        h = self.input_grn(x)

        # LSTM
        lstm_out, _ = self.lstm(h)

        # Self-Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        h = self.attn_norm(lstm_out + attn_out)

        # 디코더 GRN (마지막 시점)
        last = self.decoder_grn(h[:, -1, :])

        pred_return = self.fc_return(last).squeeze(-1)
        confidence = self.fc_confidence(last).squeeze(-1)
        pred_direction = self.fc_direction(last).squeeze(-1)

        return pred_return, confidence, pred_direction
