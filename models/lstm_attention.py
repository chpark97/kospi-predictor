"""Phase 4: LSTM + Multi-head Attention 모델"""
import torch
import torch.nn as nn

from config.settings import (
    ATTENTION_HEADS, DROPOUT, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS,
)


class LSTMAttention(nn.Module):
    """LSTM + Multi-head Self-Attention 기반 코스피 예측 모델

    LSTM으로 시계열 특성을 추출한 후, Multi-head Attention으로
    중요한 시점에 가중치를 부여합니다.
    """

    def __init__(self, num_features, hidden_size=None, num_layers=None,
                 num_heads=None, dropout=None):
        super().__init__()
        hidden_size = hidden_size or LSTM_HIDDEN_SIZE
        num_layers = num_layers or LSTM_NUM_LAYERS
        num_heads = num_heads or ATTENTION_HEADS
        dropout = dropout or DROPOUT

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.fc_return = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.fc_confidence = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)

        # Multi-head Self-Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(lstm_out + attn_out)  # residual connection

        # 마지막 시점의 출력 사용
        last_hidden = attn_out[:, -1, :]

        pred_return = self.fc_return(last_hidden).squeeze(-1)
        confidence = self.fc_confidence(last_hidden).squeeze(-1)

        return pred_return, confidence
