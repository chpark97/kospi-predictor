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
    MIN_EPOCHS, WALK_FORWARD_SPLITS,
)
from models.lstm_baseline import LSTMBaseline
from models.lstm_attention import LSTMAttention
from preprocessing.feature_engineer import FeatureEngineer
from evaluation.backtest import evaluate_predictions

logger = logging.getLogger(__name__)

SAVE_DIR = Path(__file__).parent.parent / "saved_models"


class DirectionalLoss(nn.Module):
    """등락률 예측 + 방향 정확도를 동시에 최적화하는 커스텀 손실 함수

    - MSE: 등락률 크기 예측
    - 방향 손실: 예측 부호가 실제와 다를 때 페널티
    - 확신도 손실: confidence가 방향 정확도와 일치하도록 학습
    """

    def __init__(self, direction_weight=0.5, confidence_weight=0.3):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.direction_weight = direction_weight
        self.confidence_weight = confidence_weight

    def forward(self, pred_return, confidence, y_true):
        # 1) 등락률 MSE 손실
        loss_mse = self.mse(pred_return, y_true)

        # 2) 방향 손실: 실제 방향과 예측 방향이 다르면 페널티
        true_dir = (y_true > 0).float()
        pred_prob = torch.sigmoid(pred_return * 3)  # 부호를 확률로 변환 (스케일 3)
        loss_dir = self.bce(pred_prob, true_dir)

        # 3) 확신도 학습: confidence가 방향 맞춤 여부를 예측
        with torch.no_grad():
            pred_dir = (pred_return > 0).float()
            correct = (true_dir == pred_dir).float()
        loss_conf = self.bce(confidence, correct)

        total = loss_mse + self.direction_weight * loss_dir + self.confidence_weight * loss_conf
        return total, loss_mse.item(), loss_dir.item(), loss_conf.item()


def train_model(model, train_loader, val_loader, epochs=None, lr=None, patience=None):
    """모델 학습 (Early Stopping + 방향 손실 포함)"""
    epochs = epochs or EPOCHS
    lr = lr or LEARNING_RATE
    patience = patience or EARLY_STOPPING_PATIENCE
    min_epochs = MIN_EPOCHS

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    criterion = DirectionalLoss(direction_weight=0.5, confidence_weight=0.3)
    mse_loss = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        train_dir_acc = []

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred_return, confidence = model(X_batch)

            loss, mse_val, dir_val, conf_val = criterion(pred_return, confidence, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

            # 방향 정확도 추적
            with torch.no_grad():
                dir_acc = ((pred_return > 0) == (y_batch > 0)).float().mean().item()
                train_dir_acc.append(dir_acc)

        scheduler.step(epoch)

        # Validation
        model.eval()
        val_mse_losses = []
        val_dir_accs = []
        val_confs = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred_return, confidence = model(X_batch)
                val_mse_losses.append(mse_loss(pred_return, y_batch).item())
                dir_acc = ((pred_return > 0) == (y_batch > 0)).float().mean().item()
                val_dir_accs.append(dir_acc)
                val_confs.append(confidence.mean().item())

        train_loss = np.mean(train_losses)
        train_da = np.mean(train_dir_acc) * 100
        val_loss = np.mean(val_mse_losses)
        val_da = np.mean(val_dir_accs) * 100
        val_conf = np.mean(val_confs) * 100
        current_lr = optimizer.param_groups[0]["lr"]

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"  Epoch {epoch+1:3d}: "
                f"train_loss={train_loss:.4f} train_DA={train_da:.1f}% | "
                f"val_loss={val_loss:.4f} val_DA={val_da:.1f}% conf={val_conf:.1f}% | "
                f"lr={current_lr:.2e}"
            )

        # Early stopping (최소 에폭 이후부터)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            if epoch >= min_epochs:
                wait += 1
                if wait >= patience:
                    logger.info(f"  Early stopping at epoch {epoch+1} (best_val_loss={best_val_loss:.6f})")
                    break

    if best_state:
        model.load_state_dict(best_state)

    total_epochs = epoch + 1
    logger.info(f"  학습 완료: {total_epochs} epochs, best_val_loss={best_val_loss:.6f}")
    return model, best_val_loss


def walk_forward_train(model_type="attention"):
    """Walk-forward validation으로 학습 및 평가"""
    fe = FeatureEngineer()
    df = fe.build_dataset()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for i, split in enumerate(WALK_FORWARD_SPLITS):
        train_end = split["train_end"]
        test_start = split["test_start"]
        test_end = split["test_end"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Split {i+1}: train ~{train_end} / test {test_start}~{test_end}")
        logger.info(f"{'='*60}")

        # 데이터 분할
        train_df = df[df["date"] <= train_end].copy()

        if len(train_df) < 100:
            logger.warning(f"Split {i+1}: 학습 데이터 부족 ({len(train_df)}행)")
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

        if len(X_test) < 10:
            logger.warning(f"Split {i+1}: 테스트 데이터 부족 ({len(X_test)}건)")
            continue

        # 학습/검증 분할 (train의 마지막 15%를 validation으로)
        val_size = max(int(len(X_train) * 0.15), 1)
        X_tr, y_tr = X_train[:-val_size], y_train[:-val_size]
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]

        logger.info(f"Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        logger.info(f"Train y 분포: mean={y_tr.mean():.4f}%, std={y_tr.std():.4f}%, 상승비율={(y_tr>0).mean()*100:.1f}%")

        # DataLoader
        train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        # 모델 생성
        num_features = X_train.shape[2]
        if model_type == "attention":
            model = LSTMAttention(num_features)
        else:
            model = LSTMBaseline(num_features)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"모델: {model_type}, 파라미터: {total_params:,}")

        # 학습
        model, best_val_loss = train_model(model, train_loader, val_loader)

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
        metrics["val_loss"] = best_val_loss
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
            "metrics": metrics,
        }, save_path)
        logger.info(f"모델 저장: {save_path}")

    # 전체 결과 요약
    logger.info(f"\n{'='*70}")
    logger.info(f"  Walk-Forward 전체 결과 요약 ({model_type})")
    logger.info(f"{'='*70}")
    logger.info(f"{'Split':<8} {'방향정확도':>10} {'누적수익률':>10} {'샤프비율':>8} {'승률':>8} {'거래수':>6}")
    logger.info("-" * 55)
    for r in all_results:
        logger.info(
            f"{r['split']:<8} {r['direction_accuracy']:>9.1f}% "
            f"{r['cumulative_return']:>9.2f}% {r['sharpe_ratio']:>8.2f} "
            f"{r['win_rate']:>7.1f}% {r['trade_count']:>5d}"
        )

    # 평균
    if all_results:
        avg_da = np.mean([r["direction_accuracy"] for r in all_results])
        avg_ret = np.mean([r["cumulative_return"] for r in all_results])
        avg_sharpe = np.mean([r["sharpe_ratio"] for r in all_results])
        logger.info("-" * 55)
        logger.info(f"{'평균':<8} {avg_da:>9.1f}% {avg_ret:>9.2f}% {avg_sharpe:>8.2f}")

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
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    num_features = X_all.shape[2]
    if model_type == "attention":
        model = LSTMAttention(num_features)
    else:
        model = LSTMBaseline(num_features)

    model, final_val_loss = train_model(model, train_loader, val_loader)

    final_path = SAVE_DIR / f"{model_type}_final.pt"
    torch.save({
        "model_state": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "feature_names": fe.feature_names,
        "num_features": num_features,
        "model_type": model_type,
    }, final_path)
    logger.info(f"최종 모델 저장: {final_path} (val_loss={final_val_loss:.6f})")

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
