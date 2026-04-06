"""앙상블 Stacking 메타 모델

11개 모델의 예측값(등락률 + 확신도)을 입력받아
로지스틱 회귀로 최종 방향을 결정.
Walk-forward로 메타 모델을 학습하여 미래 누수 방지.
"""
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

META_MODEL_PATH = Path(__file__).parent / "meta_model_state.json"


class MetaModel:
    """Stacking 메타 모델 (로지스틱 회귀)"""

    def __init__(self):
        self.model = LogisticRegression(C=1.0, max_iter=1000)
        self.is_fitted = False

    def fit(self, individual_preds, y_true):
        """메타 모델 학습

        Args:
            individual_preds: (n_samples, n_models) 개별 모델 예측 등락률
            y_true: (n_samples,) 실제 방향 (1=상승, 0=하락)
        """
        if len(y_true) < 10:
            logger.info("[MetaModel] 데이터 부족 (최소 10건)")
            return

        self.model.fit(individual_preds, y_true)
        self.is_fitted = True

        train_acc = self.model.score(individual_preds, y_true) * 100
        logger.info(f"[MetaModel] 학습 완료: {len(y_true)}건, train_acc={train_acc:.1f}%")
        self.save()

    def predict_proba(self, individual_preds):
        """상승 확률 예측"""
        if not self.is_fitted:
            return None
        return self.model.predict_proba(individual_preds)[:, 1]

    def save(self):
        if not self.is_fitted:
            return
        state = {
            "coef": self.model.coef_.tolist(),
            "intercept": self.model.intercept_.tolist(),
            "classes": self.model.classes_.tolist(),
        }
        with open(META_MODEL_PATH, "w") as f:
            json.dump(state, f)
        logger.info(f"[MetaModel] 저장: {META_MODEL_PATH}")

    def load(self):
        if not META_MODEL_PATH.exists():
            return False
        with open(META_MODEL_PATH) as f:
            state = json.load(f)
        self.model.coef_ = np.array(state["coef"])
        self.model.intercept_ = np.array(state["intercept"])
        self.model.classes_ = np.array(state["classes"])
        self.is_fitted = True
        logger.info("[MetaModel] 로드 완료")
        return True


def train_meta_from_history(ensemble, X_data, y_data):
    """과거 데이터로 메타 모델을 Walk-forward 방식으로 학습

    마지막 30%를 validation으로 사용하여 메타 모델 학습.
    """
    import torch

    n = len(X_data)
    split_idx = int(n * 0.7)

    # 개별 모델 예측 수집
    all_individual = []
    with torch.no_grad():
        for i in range(n):
            X_single = torch.FloatTensor(X_data[i:i+1])
            preds = []
            for model, _, _ in ensemble.models:
                ret, _ = model(X_single)
                preds.append(ret.item())
            all_individual.append(preds)

    individual_preds = np.array(all_individual)  # (n, n_models)
    y_binary = (y_data > 0).astype(int)

    # 마지막 30%로 학습 (미래 누수 방지: 앙상블 학습 후 별도)
    meta = MetaModel()
    meta.fit(individual_preds[split_idx:], y_binary[split_idx:])

    # validation 정확도
    if meta.is_fitted:
        val_proba = meta.predict_proba(individual_preds[split_idx:])
        val_pred = (val_proba > 0.5).astype(int)
        val_acc = np.mean(val_pred == y_binary[split_idx:]) * 100
        logger.info(f"[MetaModel] val_acc={val_acc:.1f}%")

    return meta
