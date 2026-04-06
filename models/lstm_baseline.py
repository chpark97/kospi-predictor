"""Phase 3: LSTM 베이스라인 모델"""
import torch
import torch.nn as nn

from config.settings import (
    DROPOUT, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS,
)


class LSTMBaseline(nn.Module):
    """LSTM 기반 코스피 등락률 예측 모델

    입력: (batch, seq_len, num_features)
    출력: (batch, 1) - 예측 등락률(%)
          (batch, 1) - 예측 확신도(sigmoid, 0~1)
    """

    def __init__(self, num_features, hidden_size=None, num_layers=None, dropout=None):
        super().__init__()
        hidden_size = hidden_size or LSTM_HIDDEN_SIZE
        num_layers = num_layers or LSTM_NUM_LAYERS
        dropout = dropout or DROPOUT

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

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

        self.fc_direction = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # 마지막 시점

        pred_return = self.fc_return(last_hidden).squeeze(-1)
        confidence = self.fc_confidence(last_hidden).squeeze(-1)
        pred_direction = self.fc_direction(last_hidden).squeeze(-1)

        return pred_return, confidence, pred_direction
