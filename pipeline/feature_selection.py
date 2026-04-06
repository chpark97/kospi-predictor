"""SHAP 기반 피처 선택 및 앙상블 다양성 분석

과적합 방지를 위해 중요도 낮은 피처를 자동 제거.
"""
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def rank_features_by_importance(ensemble, X_sample, feature_names, n_samples=50):
    """피처 중요도 순위 (perturbation 방식)

    Args:
        X_sample: (n, seq_len, features) 샘플
        feature_names: 피처 이름 리스트
        n_samples: 평가할 샘플 수

    Returns:
        sorted list of (feature_name, importance_score)
    """
    n = min(n_samples, len(X_sample))
    indices = np.random.choice(len(X_sample), n, replace=False)
    X = X_sample[indices]

    # 베이스라인 예측
    base_preds, _, _ = ensemble.predict(X)

    n_features = X.shape[2]
    importances = np.zeros(n_features)

    for f_idx in range(n_features):
        X_perm = X.copy()
        # 해당 피처를 셔플 (permutation importance)
        perm = np.random.permutation(n)
        X_perm[:, :, f_idx] = X[perm, :, f_idx]

        perm_preds, _, _ = ensemble.predict(X_perm)
        # 예측 변화량의 평균 절대값
        importances[f_idx] = np.mean(np.abs(base_preds - perm_preds))

    # 정렬
    ranked = sorted(zip(feature_names, importances), key=lambda x: -x[1])

    logger.info(f"[FeatureSelect] 피처 중요도 계산 완료 ({n_features}개)")
    for name, score in ranked[:10]:
        logger.info(f"  {name}: {score:.4f}")

    return ranked


def select_top_features(ranked_features, target_count=100):
    """상위 N개 피처 선택, 하위 피처 제거 목록 반환"""
    keep = [name for name, _ in ranked_features[:target_count]]
    remove = [name for name, _ in ranked_features[target_count:]]

    logger.info(f"[FeatureSelect] 유지: {len(keep)}개, 제거: {len(remove)}개")
    if remove:
        logger.info(f"  제거 대상 (하위): {remove[:10]}...")

    return keep, remove


def measure_ensemble_diversity(ensemble, X_sample):
    """앙상블 모델 간 예측 상관관계 분석"""
    X = X_sample[:100] if len(X_sample) > 100 else X_sample
    X_tensor = torch.FloatTensor(X)

    preds = []
    with torch.no_grad():
        for model, _, mt in ensemble.models:
            p, _ = model(X_tensor)
            preds.append(p.numpy())

    preds = np.array(preds)  # (n_models, n_samples)
    n_models = len(preds)

    # 모델 간 상관계수 행렬
    corr_matrix = np.corrcoef(preds)
    member_names = [f"{mt}_{i}" for i, (_, _, mt) in enumerate(ensemble.models)]

    # 높은 상관관계 쌍 감지
    high_corr_pairs = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            if corr_matrix[i, j] > 0.9:
                high_corr_pairs.append((member_names[i], member_names[j], corr_matrix[i, j]))

    avg_corr = (corr_matrix.sum() - n_models) / (n_models * (n_models - 1))

    logger.info(f"[Diversity] 평균 모델 간 상관: {avg_corr:.3f}")
    if high_corr_pairs:
        for a, b, c in high_corr_pairs:
            logger.info(f"  ⚠ 높은 상관: {a} ↔ {b} ({c:.3f}) → 중복 제거 권고")

    return avg_corr, high_corr_pairs
