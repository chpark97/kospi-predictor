"""학습 파이프라인: Walk-forward validation + 앙상블"""
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    BATCH_SIZE, EARLY_STOPPING_PATIENCE, EPOCHS, LEARNING_RATE,
    MIN_EPOCHS, WALK_FORWARD_SPLITS,
)
from models.ensemble import ENSEMBLE_MEMBERS, EnsemblePredictor, create_model
from preprocessing.feature_engineer import FeatureEngineer
from evaluation.backtest import evaluate_predictions, evaluate_with_thresholds
from pipeline.optuna_tuner import load_best_params
from pipeline.feature_selection import rank_features_by_importance, select_top_features
from pipeline.training_visualizer import TrainingVisualizer, format_training_report
from notifications.slack_notifier import send_slack_message

logger = logging.getLogger(__name__)

SAVE_DIR = Path(__file__).parent.parent / "saved_models"


class DirectionalLoss(nn.Module):
    """MSE + 방향 BCE(클래스 가중치 적용) + 확신도 BCE 커스텀 손실"""

    def __init__(self, direction_weight=0.5, confidence_weight=0.3, pos_weight=None):
        super().__init__()
        self.mse = nn.MSELoss()
        self.dw = direction_weight
        self.cw = confidence_weight
        # 클래스 불균형 보정: 하락일이 적으면 하락 맞추는 데 더 큰 가중치
        if pos_weight is not None:
            self.bce_dir = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
            self.use_logits = True
        else:
            self.bce_dir = nn.BCELoss()
            self.use_logits = False
        self.bce_conf = nn.BCELoss()

    def forward(self, pred_return, confidence, y_true):
        loss_mse = self.mse(pred_return, y_true)

        true_dir = (y_true > 0).float()
        if self.use_logits:
            loss_dir = self.bce_dir(pred_return * 3, true_dir)
        else:
            pred_prob = torch.sigmoid(pred_return * 3)
            loss_dir = self.bce_dir(pred_prob, true_dir)

        with torch.no_grad():
            pred_dir = (pred_return > 0).float()
            correct = (true_dir == pred_dir).float()
        loss_conf = self.bce_conf(confidence, correct)

        return loss_mse + self.dw * loss_dir + self.cw * loss_conf


def _compute_class_weight(y):
    """상승/하락 비율로 클래스 가중치 계산

    BCEWithLogitsLoss의 pos_weight는 레이블=1(상승일)에 대한 패널티 배율.
    코스피는 상승일이 다수이므로 pos_weight < 1로 설정해야
    하락일을 놓쳤을 때의 상대 패널티가 커져 상승 편향이 제거된다.
    """
    n_up = (y > 0).sum()
    n_down = (y <= 0).sum()
    if n_down == 0 or n_up == 0:
        return None
    # pos_weight = 하락일 수 / 상승일 수
    # 상승일 多 → pos_weight < 1 → 상승 레이블 패널티 감소
    #                             → 하락 레이블 상대 패널티 증가 → 편향 교정
    weight = n_down / n_up
    return float(weight)


def compute_time_weights(n_samples):
    """시간 기반 샘플 가중치 — 최근 데이터일수록 높은 가중치

    최근 1년(~250): 3.0, 2년(~500): 2.0, 3년(~750): 1.5, 이전: 1.0
    """
    weights = np.ones(n_samples)
    if n_samples > 250:
        weights[-250:] = 3.0
    if n_samples > 500:
        weights[-500:-250] = 2.0
    if n_samples > 750:
        weights[-750:-500] = 1.5
    # 정규화
    weights = weights / weights.mean()
    return weights


def train_single_model(model, train_loader, val_loader, epochs=None, lr=None,
                       patience=None, min_epochs=None, time_weights=None,
                       visualizer=None):
    """단일 모델 학습

    Args:
        model: 학습할 모델
        train_loader: 훈련 데이터 로더
        val_loader: 검증 데이터 로더
        epochs: 최대 에폭 수
        lr: 학습률
        patience: 조기 종료 인내도
        min_epochs: 최소 학습 에폭
        time_weights: 시간 가중치 텐서 (샘플별)
        visualizer: TrainingVisualizer 인스턴스 (실시간 시각화용)

    Returns:
        (model, best_val_loss, best_val_da, epoch_metrics)
        epoch_metrics: {"train_losses": [], "val_losses": [], "train_das": [], "val_das": []}
    """
    epochs = epochs or EPOCHS
    lr = lr or LEARNING_RATE
    patience = patience or EARLY_STOPPING_PATIENCE
    min_epochs = min_epochs or MIN_EPOCHS

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)  # 강화된 L2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    # 클래스 불균형 보정: train loader에서 y 추출
    all_y = []
    for _, y_b in train_loader:
        all_y.append(y_b)
    all_y_cat = torch.cat(all_y)
    pos_w = _compute_class_weight(all_y_cat.numpy())
    # 조건 완화: 0.1 → 0.05 (상승일 54% → pos_weight≈0.85 → 이제 적용됨)
    if pos_w and abs(pos_w - 1.0) > 0.05:
        logger.debug(f"    클래스 가중치: pos_weight={pos_w:.2f}")

    criterion = DirectionalLoss(pos_weight=pos_w)
    mse_loss = nn.MSELoss()

    best_val_loss = float("inf")
    best_val_da = 0.0
    best_state = None
    wait = 0

    # 에폭별 메트릭 수집
    epoch_metrics = {
        "train_losses": [],
        "val_losses": [],
        "train_das": [],
        "val_das": [],
    }

    for epoch in range(epochs):
        model.train()
        train_losses, train_dir_acc = [], []

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred_return, confidence = model(X_batch)
            loss = criterion(pred_return, confidence, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

            with torch.no_grad():
                da = ((pred_return > 0) == (y_batch > 0)).float().mean().item()
                train_dir_acc.append(da)

        scheduler.step(epoch)

        model.eval()
        val_mse_list, val_da_list = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred_return, _ = model(X_batch)
                val_mse_list.append(mse_loss(pred_return, y_batch).item())
                da = ((pred_return > 0) == (y_batch > 0)).float().mean().item()
                val_da_list.append(da)

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_mse_list)
        train_da = np.mean(train_dir_acc) * 100
        val_da = np.mean(val_da_list) * 100

        # 메트릭 저장
        epoch_metrics["train_losses"].append(train_loss)
        epoch_metrics["val_losses"].append(val_loss)
        epoch_metrics["train_das"].append(train_da)
        epoch_metrics["val_das"].append(val_da)

        # 실시간 시각화 업데이트
        if visualizer:
            visualizer.update_epoch(epoch + 1, train_loss, val_loss, train_da, val_da)

        if (epoch + 1) % 20 == 0:
            logger.info(
                f"    Epoch {epoch+1:3d}: train_DA={train_da:.1f}% | "
                f"val_loss={val_loss:.4f} val_DA={val_da:.1f}%"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_da = val_da
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            if epoch >= min_epochs:
                wait += 1
                if wait >= patience:
                    break

    if best_state:
        model.load_state_dict(best_state)

    return model, best_val_loss, best_val_da, epoch_metrics


def _compute_feature_selection(X_train, y_train, feature_names, target_count=40):
    """Train 데이터만 사용하여 피처 선택 (미래 누수 방지)

    Returns:
        selected_indices: 선택된 피처 인덱스 리스트
        keep_names: 유지할 피처 이름 리스트
    """
    num_features = X_train.shape[2]
    seq_length = X_train.shape[1]

    # 빠른 피처 선택을 위해 간단한 baseline 모델로 학습
    quick_model = create_model("baseline", num_features, seq_length)

    # 짧은 학습으로 최소한의 피처 관계 학습
    val_size = max(int(len(X_train) * 0.15), 1)
    train_ds = TensorDataset(
        torch.FloatTensor(X_train[:-val_size]),
        torch.FloatTensor(y_train[:-val_size])
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_train[-val_size:]),
        torch.FloatTensor(y_train[-val_size:])
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # 빠른 학습 (20 에폭)
    quick_model, _, _ = train_single_model(
        quick_model, train_loader, val_loader,
        epochs=20, patience=10, min_epochs=10
    )

    # EnsemblePredictor로 감싸서 importance 계산
    quick_ensemble = EnsemblePredictor()
    quick_ensemble.add_model(quick_model, 1.0, "baseline")

    # train 데이터로 피처 중요도 계산
    ranked = rank_features_by_importance(
        quick_ensemble, X_train, feature_names, n_samples=min(100, len(X_train))
    )

    # 상위 피처 선택
    keep_names, remove_names = select_top_features(ranked, target_count=target_count)

    # 선택된 피처의 인덱스 계산
    selected_indices = [i for i, name in enumerate(feature_names) if name in keep_names]

    logger.info(f"피처 선택 완료: {num_features}개 → {len(selected_indices)}개")

    return selected_indices, keep_names


def _apply_feature_selection(X, selected_indices):
    """선택된 피처 인덱스만 추출"""
    return X[:, :, selected_indices]


def walk_forward_ensemble(enable_visualization=True):
    """앙상블 Walk-forward validation

    Args:
        enable_visualization: matplotlib 실시간 시각화 활성화 여부
    """
    # Optuna 최적 파라미터 로드
    best_params = load_best_params()
    if best_params:
        logger.info(f"Optuna 최적 파라미터 로드: {best_params}")
    else:
        best_params = {}

    fe = FeatureEngineer()
    df = fe.build_dataset()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    # 피처 선택 결과 저장 (최종 앙상블에 재사용)
    global_selected_indices = None
    global_feature_ranking = None

    # 모델별 DA 저장 (최종 split 기준)
    last_split_model_das = {}

    # 실시간 시각화 초기화
    visualizer = None
    if enable_visualization:
        try:
            visualizer = TrainingVisualizer(n_models=len(ENSEMBLE_MEMBERS))
            visualizer.initialize()
        except Exception as e:
            logger.warning(f"시각화 초기화 실패 (headless 환경?): {e}")
            visualizer = None

    for i, split in enumerate(WALK_FORWARD_SPLITS):
        train_end = split["train_end"]
        test_start = split["test_start"]
        test_end = split["test_end"]

        logger.info(f"\n{'='*70}")
        logger.info(f"Split {i+1}: train ~{train_end} / test {test_start}~{test_end}")
        logger.info(f"{'='*70}")

        train_df = df[df["date"] <= train_end].copy()
        if len(train_df) < 100:
            continue

        X_train, y_train, _, scaler = fe.prepare_sequences(train_df, fit_scaler=True)

        full_for_test = df[df["date"] <= test_end].copy()
        X_full, y_full, dates_full, _ = fe.prepare_sequences(full_for_test, scaler=scaler, fit_scaler=False)
        test_mask = np.array([d >= test_start for d in dates_full])
        X_test, y_test = X_full[test_mask], y_full[test_mask]
        dates_test = np.array(dates_full)[test_mask]

        if len(X_test) < 10:
            continue

        # ── 피처 선택 (train 데이터만 사용) ──
        selected_indices, keep_names = _compute_feature_selection(
            X_train, y_train, fe.feature_names, target_count=40
        )
        global_selected_indices = selected_indices  # 마지막 split 결과를 최종 앙상블에 사용

        # 선택된 피처만 사용
        X_train = _apply_feature_selection(X_train, selected_indices)
        X_test = _apply_feature_selection(X_test, selected_indices)

        val_size = max(int(len(X_train) * 0.15), 1)
        X_tr, y_tr = X_train[:-val_size], y_train[:-val_size]
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]

        num_features = X_train.shape[2]
        seq_length = X_train.shape[1]

        logger.info(f"Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        logger.info(f"피처 수: {num_features}, 앙상블 멤버 수: {len(ENSEMBLE_MEMBERS)}")

        # ── 시간 가중치 계산 ──
        time_weights = compute_time_weights(len(X_tr))
        time_weights_tensor = torch.FloatTensor(time_weights)

        # ── 개별 모델 학습 ──
        ensemble = EnsemblePredictor()
        split_model_das = {}

        for j, (model_type, seed) in enumerate(ENSEMBLE_MEMBERS):
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = create_model(model_type, num_features, seq_length)
            params = sum(p.numel() for p in model.parameters())

            logger.info(f"  [{j+1}/{len(ENSEMBLE_MEMBERS)}] {model_type} (seed={seed}, params={params:,})")

            # 시간 가중치를 WeightedRandomSampler로 적용
            train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
            val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

            hp_lr = best_params.get("learning_rate", LEARNING_RATE)
            hp_bs = best_params.get("batch_size", BATCH_SIZE)

            # 시간 가중치 적용된 샘플러 생성
            sampler = WeightedRandomSampler(
                weights=time_weights_tensor,
                num_samples=len(X_tr),
                replacement=True
            )
            train_loader = DataLoader(train_ds, batch_size=hp_bs, sampler=sampler, drop_last=True)
            val_loader = DataLoader(val_ds, batch_size=hp_bs)

            # 시각화 에폭 메트릭 초기화
            if visualizer:
                visualizer.reset_epoch_metrics()

            model, val_loss, val_da, epoch_metrics = train_single_model(
                model, train_loader, val_loader, lr=hp_lr, visualizer=visualizer
            )
            logger.info(f"    → val_loss={val_loss:.4f}, val_DA={val_da:.1f}%")

            weight = max(val_da - 45, 1.0)
            ensemble.add_model(model, weight, model_type)
            split_model_das[model_type] = val_da

        # ── 앙상블 테스트 ──
        pred_returns, pred_confs, details = ensemble.predict(X_test)

        logger.info(f"\n  앙상블 결과:")
        logger.info(f"  모델 가중치: {details['weights'].round(3)}")

        # 개별 모델 DA도 출력 및 저장
        test_model_das = {}
        for j, (model_type, seed) in enumerate(ENSEMBLE_MEMBERS):
            indiv_da = np.mean((details["individual_returns"][j] > 0) == (y_test > 0)) * 100
            logger.info(f"    {model_type}(seed={seed}): DA={indiv_da:.1f}%")
            test_model_das[f"{model_type}"] = indiv_da

        # 최종 split의 모델 DA 저장
        last_split_model_das = test_model_das

        metrics = evaluate_predictions(y_test, pred_returns, pred_confs, dates_test)
        metrics["split"] = f"{test_start[:4]}"
        metrics["agreement"] = float(details["agreement"].mean())
        all_results.append(metrics)

        # 피처 중요도 계산 (train 데이터 기반)
        feature_ranking = rank_features_by_importance(
            ensemble, X_train, [fe.feature_names[idx] for idx in selected_indices],
            n_samples=min(100, len(X_train))
        )
        global_feature_ranking = feature_ranking

        # 실시간 시각화 업데이트
        if visualizer:
            daily_preds = pred_returns > 0
            daily_actuals = y_test > 0
            visualizer.update_split_complete(
                split_idx=i,
                model_das=test_model_das,
                daily_preds=daily_preds,
                daily_actuals=daily_actuals,
                feature_ranking=feature_ranking,
                dates=dates_test
            )

        # 앙상블 저장
        save_path = SAVE_DIR / f"ensemble_split{i+1}.pt"
        member_states = []
        for model, weight, model_type in ensemble.models:
            member_states.append({
                "model_state": model.state_dict(),
                "weight": weight,
                "model_type": model_type,
            })

        torch.save({
            "members": member_states,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "feature_names": fe.feature_names,
            "num_features": num_features,
            "seq_length": seq_length,
            "split": split,
            "metrics": metrics,
            "selected_feature_indices": selected_indices,
        }, save_path)
        logger.info(f"  앙상블 저장: {save_path}")

    # ── 전체 결과 요약 ──
    logger.info(f"\n{'='*75}")
    logger.info(f"  Walk-Forward 앙상블 결과 요약")
    logger.info(f"{'='*75}")
    logger.info(f"{'Split':<8} {'방향정확도':>10} {'누적수익률':>10} {'샤프비율':>8} {'승률':>8} {'합의도':>8} {'거래수':>6}")
    logger.info("-" * 65)
    for r in all_results:
        logger.info(
            f"{r['split']:<8} {r['direction_accuracy']:>9.1f}% "
            f"{r['cumulative_return']:>9.2f}% {r['sharpe_ratio']:>8.2f} "
            f"{r['win_rate']:>7.1f}% {r['agreement']:>7.1f}% {r['trade_count']:>5d}"
        )

    if all_results:
        avg_da = np.mean([r["direction_accuracy"] for r in all_results])
        avg_ret = np.mean([r["cumulative_return"] for r in all_results])
        avg_sharpe = np.mean([r["sharpe_ratio"] for r in all_results])
        logger.info("-" * 65)
        logger.info(f"{'평균':<8} {avg_da:>9.1f}% {avg_ret:>9.2f}% {avg_sharpe:>8.2f}")

    # 최종 시각화 업데이트
    if visualizer:
        visualizer.show_final_summary(all_results)

    # ── 최종 앙상블: 전체 데이터로 재학습 ──
    logger.info(f"\n최종 앙상블 학습 (전체 데이터)...")
    X_all, y_all, _, scaler = fe.prepare_sequences(df, fit_scaler=True)

    # 피처 선택 적용 (walk-forward에서 계산된 인덱스 사용, 없으면 새로 계산)
    if global_selected_indices is None:
        global_selected_indices, _ = _compute_feature_selection(
            X_all, y_all, fe.feature_names, target_count=40
        )

    X_all = _apply_feature_selection(X_all, global_selected_indices)
    logger.info(f"최종 앙상블 피처 수: {X_all.shape[2]}개 (선택됨)")

    val_size = max(int(len(X_all) * 0.1), 1)
    X_tr, y_tr = X_all[:-val_size], y_all[:-val_size]
    X_val, y_val = X_all[-val_size:], y_all[-val_size:]

    num_features = X_all.shape[2]
    seq_length = X_all.shape[1]
    ensemble_final = EnsemblePredictor()

    for j, (model_type, seed) in enumerate(ENSEMBLE_MEMBERS):
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = create_model(model_type, num_features, seq_length)
        logger.info(f"  [{j+1}/{len(ENSEMBLE_MEMBERS)}] {model_type} (seed={seed})")

        train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        model, val_loss, val_da = train_single_model(model, train_loader, val_loader)
        logger.info(f"    → val_loss={val_loss:.4f}, val_DA={val_da:.1f}%")

        weight = max(val_da - 45, 1.0)
        ensemble_final.add_model(model, weight, model_type)

    # 최종 저장
    member_states = []
    for model, weight, model_type in ensemble_final.models:
        member_states.append({
            "model_state": model.state_dict(),
            "weight": weight,
            "model_type": model_type,
        })

    final_path = SAVE_DIR / "ensemble_final.pt"
    torch.save({
        "members": member_states,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "feature_names": fe.feature_names,
        "num_features": num_features,
        "seq_length": seq_length,
        "selected_feature_indices": global_selected_indices,
    }, final_path)
    logger.info(f"최종 앙상블 저장: {final_path}")
    logger.info(f"  선택된 피처 인덱스 저장: {len(global_selected_indices)}개")

    # ── 차트 저장 및 슬랙 리포트 ──
    if visualizer:
        chart_path = visualizer.save_charts()
        logger.info(f"학습 차트 저장 완료: {chart_path}")

    # 모델 랭킹 정렬
    model_rankings = sorted(last_split_model_das.items(), key=lambda x: x[1], reverse=True)

    # 슬랙 텍스트 요약 전송
    try:
        slack_report = format_training_report(
            all_split_results=all_results,
            model_rankings=model_rankings,
            feature_rankings=global_feature_ranking,
            best_threshold=None,  # 임계값 비교는 evaluate_with_thresholds에서 수행
            threshold_metrics=None
        )
        send_slack_message(slack_report)
        logger.info("[Slack] 학습 리포트 전송 완료")
    except Exception as e:
        logger.warning(f"슬랙 리포트 전송 실패: {e}")

    # 시각화 블로킹 표시 (사용자가 창을 닫을 때까지)
    if visualizer:
        logger.info("차트 창을 닫으면 학습이 완료됩니다...")
        visualizer.show_blocking()
        visualizer.close()

    return all_results


# 단일 모델 학습도 유지 (하위 호환)
def walk_forward_train(model_type="ensemble"):
    if model_type == "ensemble":
        return walk_forward_ensemble()

    # 기존 단일 모델 학습 로직
    fe = FeatureEngineer()
    df = fe.build_dataset()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for i, split in enumerate(WALK_FORWARD_SPLITS):
        train_end, test_start, test_end = split["train_end"], split["test_start"], split["test_end"]
        logger.info(f"\nSplit {i+1}: train ~{train_end} / test {test_start}~{test_end}")

        train_df = df[df["date"] <= train_end].copy()
        X_train, y_train, _, scaler = fe.prepare_sequences(train_df, fit_scaler=True)
        full_for_test = df[df["date"] <= test_end].copy()
        X_full, y_full, dates_full, _ = fe.prepare_sequences(full_for_test, scaler=scaler, fit_scaler=False)
        test_mask = np.array([d >= test_start for d in dates_full])
        X_test, y_test, dates_test = X_full[test_mask], y_full[test_mask], np.array(dates_full)[test_mask]

        val_size = max(int(len(X_train) * 0.15), 1)
        train_ds = TensorDataset(torch.FloatTensor(X_train[:-val_size]), torch.FloatTensor(y_train[:-val_size]))
        val_ds = TensorDataset(torch.FloatTensor(X_train[-val_size:]), torch.FloatTensor(y_train[-val_size:]))
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        model = create_model(model_type, X_train.shape[2], X_train.shape[1])
        model, val_loss, val_da = train_single_model(model, train_loader, val_loader)

        model.eval()
        with torch.no_grad():
            pred_returns, pred_confs = model(torch.FloatTensor(X_test))
            pred_returns, pred_confs = pred_returns.numpy(), pred_confs.numpy()

        metrics = evaluate_predictions(y_test, pred_returns, pred_confs, dates_test)
        metrics["split"] = test_start[:4]
        all_results.append(metrics)

    return all_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ensemble",
                        choices=["baseline", "attention", "cnn", "transformer", "ensemble"])
    args = parser.parse_args()

    walk_forward_train(args.model)
