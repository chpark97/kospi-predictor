"""Monte Carlo Dropout - 예측 불확실성 추정

추론 시 Dropout을 유지한 채 N번 샘플링하여
평균 → 예측값, 표준편차 → 불확실성을 추정.
"""
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _enable_dropout(model):
    """모델의 Dropout 레이어만 train 모드로 전환, BatchNorm은 명시적 eval 유지

    CNN 모델에 BatchNorm1d가 있으므로, Dropout만 train으로 바꾸고
    BatchNorm은 eval 유지하여 running statistics 사용.
    이렇게 하지 않으면 MC Dropout의 확률적 변동과 BatchNorm이 충돌.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()
        elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            m.eval()


def mc_dropout_predict(ensemble, X, n_samples=50):
    """Monte Carlo Dropout 예측

    Args:
        ensemble: EnsemblePredictor
        X: 입력 시퀀스 (1, seq_len, features)
        n_samples: 샘플링 횟수

    Returns:
        mean_return: 평균 예측 등락률
        std_return: 예측 표준편차 (불확실성)
        uncertainty_level: "low" / "medium" / "high"
    """
    X_tensor = torch.FloatTensor(X) if not isinstance(X, torch.Tensor) else X

    all_samples = []

    for _ in range(n_samples):
        sample_preds = []
        for model, weight, _ in ensemble.models:
            # Dropout만 train 모드로 (BatchNorm은 eval 유지)
            _enable_dropout(model)

            with torch.no_grad():
                pred, _ = model(X_tensor)
                sample_preds.append(pred.item() * weight)

            # 다시 eval 모드로
            model.eval()

        all_samples.append(sum(sample_preds) / sum(w for _, w, _ in ensemble.models))

    all_samples = np.array(all_samples)
    mean_return = float(np.mean(all_samples))
    std_return = float(np.std(all_samples))

    # 불확실성 레벨
    if std_return < 0.3:
        level = "low"
        emoji = "🟢"
        label = "낮음"
    elif std_return < 0.7:
        level = "medium"
        emoji = "🟡"
        label = "보통"
    else:
        level = "high"
        emoji = "🔴"
        label = "높음"

    logger.info(f"  MC Dropout: mean={mean_return:.3f}%, std={std_return:.3f}% → {label} {emoji}")

    return mean_return, std_return, level, emoji, label
