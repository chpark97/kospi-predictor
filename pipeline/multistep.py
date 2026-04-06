"""멀티스텝 예측 (1일/2일/3일 후)

기존 앙상블의 시퀀스를 1칸씩 밀어서 2일/3일 후 예측을 근사.
3일 모두 같은 방향이면 신뢰도 +10% 보너스.
"""
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def predict_multistep(ensemble, X_recent, n_steps=3):
    """멀티스텝 예측

    Args:
        ensemble: EnsemblePredictor
        X_recent: 최근 시퀀스 (1, seq_len, features) — 이미 스케일링됨
        n_steps: 예측 단계 수

    Returns:
        directions: ["up", "down", ...] 각 단계 방향
        returns: [0.3, -0.1, 0.5] 각 단계 예측 등락률
        all_same: 3일 모두 같은 방향인지
    """
    directions = []
    returns = []

    X = X_recent.copy()

    for step in range(n_steps):
        pred_ret, _, _ = ensemble.predict(X)
        ret = pred_ret[0]
        directions.append("up" if ret > 0 else "down")
        returns.append(round(float(ret), 4))

        if step < n_steps - 1:
            # 다음 단계: 시퀀스를 1칸 밀고 마지막 행을 복제 (근사)
            X = np.roll(X, -1, axis=1)
            X[:, -1, :] = X[:, -2, :]

    all_same = len(set(directions)) == 1

    dir_symbols = ["▲" if d == "up" else "▼" for d in directions]
    logger.info(f"  멀티스텝 전망: {''.join(dir_symbols)} (1일/2일/3일)")

    return directions, returns, all_same


def format_multistep(directions):
    """멀티스텝 결과를 슬랙용 문자열로 변환"""
    symbols = ["▲" if d == "up" else "▼" for d in directions]
    return "".join(symbols) + " (1일/2일/3일)"
