"""앙상블 모델 - 다수의 모델 예측을 결합

앙상블 전략:
1. 아키텍처 다양성: LSTM, LSTM+Attention, 1D-CNN, Transformer, TFT
2. 시드 다양성: 동일 아키텍처를 다른 시드로 학습
3. 가중 투표: validation 성능 기반 가중치 부여
"""
import logging

import numpy as np
import torch

from models.lstm_baseline import LSTMBaseline
from models.lstm_attention import LSTMAttention
from models.cnn_model import CNN1D
from models.transformer_model import TransformerPredictor
from models.multiscale_model import MultiScaleLSTM
from models.tft_model import TemporalFusionTransformer

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "baseline": LSTMBaseline,
    "attention": LSTMAttention,
    "cnn": CNN1D,
    "transformer": TransformerPredictor,
    "multiscale": MultiScaleLSTM,
    "tft": TemporalFusionTransformer,
}

# 앙상블 구성: (모델 타입, 시드) - 총 11개
# multiscale 모델 제거: dynamic_weights.json에서 항상 correct=false, 가중치 0.1
# → 20일 시퀀스에서 서브샘플링이 의미 없고 학습 실패 패턴 반복
ENSEMBLE_MEMBERS = [
    ("attention", 42),
    ("attention", 123),
    ("attention", 777),
    ("baseline", 42),
    ("baseline", 123),
    ("cnn", 42),
    ("cnn", 123),
    ("transformer", 42),
    ("transformer", 123),
    ("tft", 42),
    ("tft", 123),
]


def create_model(model_type, num_features, seq_length=20):
    """모델 타입에 따라 인스턴스 생성"""
    cls = MODEL_REGISTRY[model_type]
    if model_type in ("cnn", "transformer", "multiscale", "tft"):
        return cls(num_features, seq_length=seq_length)
    return cls(num_features)


class EnsemblePredictor:
    """학습된 앙상블 모델들의 예측을 결합"""

    def __init__(self):
        self.models = []       # (model, weight, model_type)
        self.scaler = None

    def add_model(self, model, weight=1.0, model_type="attention"):
        model.eval()
        self.models.append((model, weight, model_type))

    def predict(self, X):
        """가중 앙상블 예측"""
        if not self.models:
            raise ValueError("앙상블에 모델이 없습니다")

        X_tensor = torch.FloatTensor(X) if not isinstance(X, torch.Tensor) else X

        all_returns = []
        all_confs = []
        weights = []

        with torch.no_grad():
            for model, weight, model_type in self.models:
                pred_ret, pred_conf = model(X_tensor)
                all_returns.append(pred_ret.numpy())
                all_confs.append(pred_conf.numpy())
                weights.append(weight)

        all_returns = np.array(all_returns)
        all_confs = np.array(all_confs)
        weights = np.array(weights)
        weights = weights / weights.sum()

        pred_return = np.average(all_returns, axis=0, weights=weights)

        directions = (all_returns > 0).astype(float)
        up_vote_ratio = np.average(directions, axis=0, weights=weights)
        agreement = np.abs(up_vote_ratio - 0.5) * 2

        avg_conf = np.average(all_confs, axis=0, weights=weights)
        confidence = avg_conf * (0.5 + 0.5 * agreement)

        details = {
            "individual_returns": all_returns,
            "individual_confs": all_confs,
            "up_vote_ratio": up_vote_ratio,
            "agreement": agreement,
            "weights": weights,
        }

        return pred_return, confidence, details
