"""1D-CNN 모델 - 앙상블 구성원"""
import torch
import torch.nn as nn

from config.settings import DROPOUT


class CNN1D(nn.Module):
    """1D Convolutional Neural Network for time series

    CNN은 LSTM과 다른 관점으로 패턴을 학습:
    - 로컬 패턴 (단기 가격 움직임) 포착에 강점
    - 병렬 연산으로 빠른 학습
    """

    def __init__(self, num_features, seq_length=20, dropout=None):
        super().__init__()
        dropout = dropout or DROPOUT

        self.conv_block = nn.Sequential(
            # Block 1: 넓은 커널로 중기 패턴
            nn.Conv1d(num_features, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Block 2: 중간 커널
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Block 3: 좁은 커널로 단기 패턴
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc_return = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        self.fc_confidence = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.fc_direction = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len) for Conv1d
        x = x.transpose(1, 2)
        x = self.conv_block(x)
        x = self.global_pool(x).squeeze(-1)  # (batch, 64)

        pred_return = self.fc_return(x).squeeze(-1)
        confidence = self.fc_confidence(x).squeeze(-1)
        pred_direction = self.fc_direction(x).squeeze(-1)

        return pred_return, confidence, pred_direction
