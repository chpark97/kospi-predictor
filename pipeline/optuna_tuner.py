"""Optuna 하이퍼파라미터 자동 튜닝

Walk-forward 기준 평균 방향정확도를 최대화하는 하이퍼파라미터 탐색.
최적 파라미터를 config/best_params.json에 저장.
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import WALK_FORWARD_SPLITS
from models.ensemble import create_model
from preprocessing.feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)

BEST_PARAMS_PATH = Path(__file__).parent.parent / "config" / "best_params.json"


class _DirectionalLoss(nn.Module):
    def __init__(self, dw=0.5, cw=0.3):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.dw, self.cw = dw, cw

    def forward(self, pred, conf, y):
        loss_mse = self.mse(pred, y)
        loss_dir = self.bce(torch.sigmoid(pred * 3), (y > 0).float())
        with torch.no_grad():
            correct = ((y > 0).float() == (pred > 0).float()).float()
        loss_conf = self.bce(conf, correct)
        return loss_mse + self.dw * loss_dir + self.cw * loss_conf


def _quick_train(model, X_tr, y_tr, X_val, y_val, lr, batch_size, epochs=60, patience=15):
    """빠른 학습 (튜닝용, 에폭 축소)"""
    train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = _DirectionalLoss()
    mse_fn = nn.MSELoss()

    best_val_da = 0.0
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            pred, conf = model(X_b)
            loss = criterion(pred, conf, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_da_list = []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                pred, _ = model(X_b)
                da = ((pred > 0) == (y_b > 0)).float().mean().item()
                val_da_list.append(da)

        val_da = np.mean(val_da_list) * 100

        if val_da > best_val_da:
            best_val_da = val_da
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            if epoch >= 20:
                wait += 1
                if wait >= patience:
                    break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_da


def objective(trial, fe, df):
    """Optuna 목적함수: Walk-forward 평균 방향정확도"""
    # 하이퍼파라미터 탐색 공간
    lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    direction_weight = trial.suggest_float("direction_weight", 0.2, 1.0)

    all_da = []

    # 속도를 위해 첫 2개 split만 사용
    for split in WALK_FORWARD_SPLITS[:2]:
        train_df = df[df["date"] <= split["train_end"]].copy()
        X_train, y_train, _, scaler = fe.prepare_sequences(train_df, fit_scaler=True)

        full_test = df[df["date"] <= split["test_end"]].copy()
        X_full, y_full, dates_full, _ = fe.prepare_sequences(full_test, scaler=scaler, fit_scaler=False)
        test_mask = np.array([d >= split["test_start"] for d in dates_full])
        X_test, y_test = X_full[test_mask], y_full[test_mask]

        if len(X_test) < 10:
            return 0.0

        val_size = max(int(len(X_train) * 0.15), 1)
        X_tr, y_tr = X_train[:-val_size], y_train[:-val_size]
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]

        num_features = X_train.shape[2]
        seq_length = X_train.shape[1]

        # attention 모델로 테스트
        torch.manual_seed(42)
        from models.lstm_attention import LSTMAttention
        model = LSTMAttention(num_features, hidden_size=hidden_size,
                              num_layers=num_layers, dropout=dropout)

        model, val_da = _quick_train(model, X_tr, y_tr, X_val, y_val, lr, batch_size)

        # 테스트 DA
        model.eval()
        with torch.no_grad():
            pred, _ = model(torch.FloatTensor(X_test))
            test_da = ((pred.numpy() > 0) == (y_test > 0)).mean() * 100
        all_da.append(test_da)

        # Pruning
        trial.report(np.mean(all_da), len(all_da) - 1)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(all_da)


def run_optuna_tuning(n_trials=30):
    """Optuna 튜닝 실행"""
    logger.info("=" * 60)
    logger.info("Optuna 하이퍼파라미터 튜닝 시작")
    logger.info(f"Trials: {n_trials}")
    logger.info("=" * 60)

    fe = FeatureEngineer()
    df = fe.build_dataset()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )

    study.optimize(lambda trial: objective(trial, fe, df), n_trials=n_trials)

    # 결과 요약
    logger.info(f"\n{'='*60}")
    logger.info("Optuna 튜닝 결과")
    logger.info(f"{'='*60}")
    logger.info(f"최적 방향정확도: {study.best_value:.2f}%")
    logger.info(f"최적 파라미터:")
    for k, v in study.best_params.items():
        logger.info(f"  {k}: {v}")

    # 상위 5개 trial
    logger.info(f"\n상위 5개 Trial:")
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)
    for t in trials_sorted[:5]:
        if t.value:
            logger.info(f"  Trial {t.number}: DA={t.value:.2f}% | {t.params}")

    # 최적 파라미터 저장
    best = study.best_params
    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(best, f, indent=2)
    logger.info(f"\n최적 파라미터 저장: {BEST_PARAMS_PATH}")

    return best


def load_best_params():
    """저장된 최적 파라미터 로드 (없으면 기본값)"""
    if BEST_PARAMS_PATH.exists():
        with open(BEST_PARAMS_PATH) as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_optuna_tuning(n_trials=30)
