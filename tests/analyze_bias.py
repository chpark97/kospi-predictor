"""모델 출력 분포 + 학습 데이터 편향 분석"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import logging
logging.basicConfig(level=logging.WARNING)

from pipeline.predict import load_ensemble
from preprocessing.feature_engineer import FeatureEngineer


def analyze():
    # 1. 앙상블 로드
    ensemble, scaler, feature_names = load_ensemble()
    print(f"앙상블 모델 수: {len(ensemble.models)}")
    for i, (m, w, mt) in enumerate(ensemble.models):
        print(f"  [{i}] {mt} weight={w:.3f}")

    # 2. 데이터셋 빌드
    fe = FeatureEngineer()
    df = fe.build_dataset()
    X_all, y_all, dates_all, _ = fe.prepare_sequences(df, scaler=scaler, fit_scaler=False)

    print(f"\n전체 데이터: {len(X_all)}일")
    print(f"최근 60일: {dates_all[-60]} ~ {dates_all[-1]}")

    # 3. 학습 데이터 y 분포
    print(f"\n{'='*60}")
    print("학습 데이터 (y) 분포 분석")
    print(f"{'='*60}")
    y_flat = y_all.flatten()
    n_up = (y_flat > 0).sum()
    n_down = (y_flat <= 0).sum()
    print(f"  총 샘플: {len(y_flat)}")
    print(f"  상승(y>0): {n_up} ({n_up/len(y_flat):.1%})")
    print(f"  하락(y≤0): {n_down} ({n_down/len(y_flat):.1%})")
    print(f"  y 평균: {y_flat.mean():.4f}%")
    print(f"  y 중앙값: {np.median(y_flat):.4f}%")
    print(f"  y 표준편차: {y_flat.std():.4f}%")
    print(f"  y 범위: {y_flat.min():.4f}% ~ {y_flat.max():.4f}%")

    # 최근 60일 y
    y_recent = y_all[-60:].flatten()
    n_up_r = (y_recent > 0).sum()
    print(f"\n  최근 60일 y 평균: {y_recent.mean():.4f}%")
    print(f"  최근 60일 상승비율: {n_up_r}/{len(y_recent)} ({n_up_r/len(y_recent):.1%})")

    # 4. 개별 모델 출력 분포 (최근 60일)
    print(f"\n{'='*60}")
    print("개별 모델 출력 분포 (최근 60일)")
    print(f"{'='*60}")

    X_recent = X_all[-60:]
    X_tensor = torch.FloatTensor(X_recent)

    model_outputs = []
    with torch.no_grad():
        for i, (model, weight, model_type) in enumerate(ensemble.models):
            pred_ret, pred_conf = model(X_tensor)
            rets = pred_ret.numpy().flatten()
            model_outputs.append(rets)
            n_pos = (rets > 0).sum()
            print(
                f"  [{i:2d}] {model_type:<12} | "
                f"mean={rets.mean():+.4f} std={rets.std():.4f} "
                f"min={rets.min():+.4f} max={rets.max():+.4f} | "
                f"상승예측: {n_pos}/{len(rets)} ({n_pos/len(rets):.0%})"
            )

    # 5. 앙상블 합산 결과
    print(f"\n{'='*60}")
    print("앙상블 합산 결과 (최근 60일)")
    print(f"{'='*60}")
    pred_returns, _, details = ensemble.predict(X_recent)
    pred_flat = pred_returns.flatten()
    n_up_pred = (pred_flat > 0).sum()
    print(f"  앙상블 평균: {pred_flat.mean():+.4f}%")
    print(f"  앙상블 std: {pred_flat.std():.4f}%")
    print(f"  앙상블 범위: {pred_flat.min():+.4f}% ~ {pred_flat.max():+.4f}%")
    print(f"  상승 예측: {n_up_pred}/{len(pred_flat)} ({n_up_pred/len(pred_flat):.0%})")

    # 방향 정확도
    y_recent_flat = y_recent.flatten()
    correct = ((pred_flat > 0) == (y_recent_flat > 0)).sum()
    print(f"  방향 정확도: {correct}/{len(pred_flat)} ({correct/len(pred_flat):.1%})")

    # 6. 편향 크기 정량화
    print(f"\n{'='*60}")
    print("편향 정량화")
    print(f"{'='*60}")
    ensemble_mean = pred_flat.mean()
    y_mean = y_recent_flat.mean()
    print(f"  예측 평균 - 실제 평균 = {ensemble_mean - y_mean:+.4f}%")
    print(f"  de-mean 적용 시 상승예측: {((pred_flat - ensemble_mean) > 0).sum()}/{len(pred_flat)}")

    # 적응적 임계값 시뮬레이션
    for window in [10, 20, 30]:
        adjusted_ups = 0
        total = 0
        correct_adj = 0
        for j in range(window, len(pred_flat)):
            rolling_mean = pred_flat[j-window:j].mean()
            adjusted = pred_flat[j] - rolling_mean
            is_up = adjusted > 0
            actual_up = y_recent_flat[j] > 0
            adjusted_ups += int(is_up)
            total += 1
            correct_adj += int(is_up == actual_up)
        if total > 0:
            print(
                f"  적응적 임계값 (window={window}): 상승예측 {adjusted_ups}/{total} "
                f"({adjusted_ups/total:.0%}), DA={correct_adj/total:.1%}"
            )


if __name__ == "__main__":
    analyze()
