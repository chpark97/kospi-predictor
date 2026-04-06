"""멀티스케일 시퀀스 모델

단기(5일) + 중기(20일) + 장기(60일) 시퀀스를 동시 입력하여
각 스케일별 LSTM으로 인코딩 후 concat하여 예측.
"""
import torch
import torch.nn as nn

from config.settings import DROPOUT


class MultiScaleLSTM(nn.Module):
    """Multi-Scale LSTM

    3개 스케일의 시퀀스를 개별 LSTM으로 인코딩 후 결합.
    표준 20일 입력을 받아 내부에서 5일/20일 슬라이스로 분할.
    """

    def __init__(self, num_features, hidden_size=64, dropout=None, seq_length=20):
        super().__init__()
        dropout = dropout or DROPOUT
        self.seq_length = seq_length

        # 단기 LSTM (최근 5일)
        self.lstm_short = nn.LSTM(num_features, hidden_size, batch_first=True)
        # 중기 LSTM (최근 20일 = 전체)
        self.lstm_mid = nn.LSTM(num_features, hidden_size, batch_first=True)
        # 장기 요약: 20일 데이터를 4일 간격으로 서브샘플링 → 5 step
        self.lstm_long = nn.LSTM(num_features, hidden_size, batch_first=True)

        concat_size = hidden_size * 3

        self.fc_return = nn.Sequential(
            nn.Linear(concat_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.fc_confidence = nn.Sequential(
            nn.Linear(concat_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.fc_direction = nn.Sequential(
            nn.Linear(concat_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        batch_size = x.size(0)
        seq_len = x.size(1)

        # 단기: 최근 5일
        x_short = x[:, -5:, :]
        _, (h_short, _) = self.lstm_short(x_short)

        # 중기: 전체 시퀀스
        _, (h_mid, _) = self.lstm_mid(x)

        # 장기: 4일 간격 서브샘플링 (인덱스 0, 4, 8, 12, 16)
        indices = list(range(0, seq_len, max(seq_len // 5, 1)))[:5]
        if len(indices) < 2:
            indices = [0, seq_len - 1]
        x_long = x[:, indices, :]
        _, (h_long, _) = self.lstm_long(x_long)

        # concat
        combined = torch.cat([
            h_short[-1],  # (batch, hidden)
            h_mid[-1],
            h_long[-1],
        ], dim=1)

        pred_return = self.fc_return(combined).squeeze(-1)
        confidence = self.fc_confidence(combined).squeeze(-1)
        pred_direction = self.fc_direction(combined).squeeze(-1)

        return pred_return, confidence, pred_direction
