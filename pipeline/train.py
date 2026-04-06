"""학습 파이프라인: Walk-forward validation"""
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    BATCH_SIZE, EARLY_STOPPING_PATIENCE, EPOCHS, LEARNING_RATE,
    WALK_FORWARD_SPLITS,
)
from models.lstm_baseline import LSTMBaseline
from models.lstm_attention import LSTMAttention
from preprocessing.feature_engineer import FeatureEngineer
from evaluation.backtest import evaluate_predictions

logger = logging.getLogger(__name__)

SAVE_DIR = Path(__file__).parent.parent / "saved_models"


def train_model(model, train_loader, val_loader, epochs=None, lr=None, patience=None):
    """모델 학습 (Early Stopping 포함)"""
    epochs = epochs or EPOCHS
    lr = lr or LEARNING_RATE
    patience = patience or EARLY_STOPPING_PATIENCE

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 손실 함수: 등락률 MSE + 방향 정확도 보조 손실
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred_return, confidence = model(X_batch)

            # 메인 손실: 등락률 예측
            loss_return = mse_loss(pred_return, y_batch)

            # 보조 손실: 방향 맞추기 (confidence가 방향 정확도를 학습)
            direction_target = (y_batch > 0).float()
            direction_pred = (pred_return > 0).float()
            correct = (direction_target == direction_pred).float()
            loss_conf = bce_loss(confidence, correct)

            loss = loss_return + 0.3 * loss_conf
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred_return, _ = model(X_batch)
                val_losses.append(mse_loss(pred_return, y_batch).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_loss


def walk_forward_train(model_type="attention"):
    """Walk-forward validation으로 학습 및 평가

    Args:
        model_type: "baseline" (LSTM) 또는 "attention" (LSTM+Attention)
    """
    fe = FeatureEngineer()
    df = fe.build_dataset()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for i, split in enumerate(WALK_FORWARD_SPLITS):
        train_end = split["train_end"]
        test_start = split["test_start"]
        test_end = split["test_end"]

        logger.info(f"\n{'='*50}")
        logger.info(f"Split {i+1}: train ~{train_end} / test {test_start}~{test_end}")
        logger.info(f"{'='*50}")

        # 데이터 분할
        train_df = df[df["date"] <= train_end].copy()
        test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()

        if len(train_df) < 100 or len(test_df) < 10:
            logger.warning(f"Split {i+1}: 데이터 부족 (train={len(train_df)}, test={len(test_df)})")
            continue

        # 시퀀스 생성
        X_train, y_train, dates_train, scaler = fe.prepare_sequences(train_df, fit_scaler=True)

        # 테스트는 전체 기간 데이터로 시퀀스를 만들되, 테스트 기간 날짜만 필터
        full_for_test = df[df["date"] <= test_end].copy()
        X_full, y_full, dates_full, _ = fe.prepare_sequences(full_for_test, scaler=scaler, fit_scaler=False)

        # 테스트 기간만 필터링
        test_mask = np.array([d >= test_start for d in dates_full])
        X_test = X_full[test_mask]
        y_test = y_full[test_mask]
        dates_test = np.array(dates_full)[test_mask]

        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

        # 학습/검증 분할 (train의 마지막 15%를 validation으로)
        val_size = max(int(len(X_train) * 0.15), 1)
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        X_tr, y_tr = X_train[:-val_size], y_train[:-val_size]

        # DataLoader
        train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        # 모델 생성
        num_features = X_train.shape[2]
        if model_type == "attention":
            model = LSTMAttention(num_features)
        else:
            model = LSTMBaseline(num_features)

        logger.info(f"모델: {model_type}, 파라미터: {sum(p.numel() for p in model.parameters()):,}")

        # 학습
        model, best_val_loss = train_model(model, train_loader, val_loader)
        logger.info(f"Best val_loss: {best_val_loss:.6f}")

        # 테스트 예측
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test)
            pred_returns, pred_confs = model(X_test_t)
            pred_returns = pred_returns.numpy()
            pred_confs = pred_confs.numpy()

        # 평가
        metrics = evaluate_predictions(y_test, pred_returns, pred_confs, dates_test)
        metrics["split"] = f"{test_start[:4]}"
        all_results.append(metrics)

        # 모델 저장
        save_path = SAVE_DIR / f"{model_type}_split{i+1}.pt"
        torch.save({
            "model_state": model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "feature_names": fe.feature_names,
            "num_features": num_features,
            "model_type": model_type,
            "split": split,
        }, save_path)
        logger.info(f"모델 저장: {save_path}")

    # 전체 결과 요약
    logger.info(f"\n{'='*60}")
    logger.info("Walk-Forward 전체 결과 요약")
    logger.info(f"{'='*60}")
    logger.info(f"{'Split':<8} {'방향정확도':>10} {'누적수익률':>10} {'샤프비율':>10}")
    logger.info("-" * 42)
    for r in all_results:
        logger.info(
            f"{r['split']:<8} {r['direction_accuracy']:>9.1f}% "
            f"{r['cumulative_return']:>9.2f}% {r['sharpe_ratio']:>10.2f}"
        )

    # 최종 모델: 전체 데이터로 재학습
    logger.info(f"\n최종 모델 학습 (전체 데이터)...")
    X_all, y_all, _, scaler = fe.prepare_sequences(df, fit_scaler=True)
    val_size = max(int(len(X_all) * 0.1), 1)
    train_ds = TensorDataset(
        torch.FloatTensor(X_all[:-val_size]),
        torch.FloatTensor(y_all[:-val_size]),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_all[-val_size:]),
        torch.FloatTensor(y_all[-val_size:]),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    num_features = X_all.shape[2]
    if model_type == "attention":
        model = LSTMAttention(num_features)
    else:
        model = LSTMBaseline(num_features)

    model, _ = train_model(model, train_loader, val_loader)

    final_path = SAVE_DIR / f"{model_type}_final.pt"
    torch.save({
        "model_state": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "feature_names": fe.feature_names,
        "num_features": num_features,
        "model_type": model_type,
    }, final_path)
    logger.info(f"최종 모델 저장: {final_path}")

    return all_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="attention", choices=["baseline", "attention"])
    args = parser.parse_args()

    walk_forward_train(args.model)
