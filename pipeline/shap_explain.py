"""SHAP 기반 피처 중요도 분석 (경량 버전)

SHAP 설치 실패 시 순열 중요도(permutation importance)로 대체.
매 예측 시 상위 5개 영향 피처를 산출.
"""
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_top_features(ensemble, X_recent, feature_names, top_k=5):
    """예측에 가장 큰 영향을 미친 피처 상위 K개 산출

    방법: 각 피처를 0으로 마스킹한 후 예측 변화량 측정 (perturbation)

    Returns:
        list of (feature_name, impact_direction, impact_score)
    """
    X = X_recent.copy()  # (1, seq_len, n_features)
    base_pred, _, _ = ensemble.predict(X)
    base_pred = base_pred[0]

    n_features = X.shape[2]
    impacts = []

    for f_idx in range(n_features):
        X_perturbed = X.copy()
        X_perturbed[:, :, f_idx] = 0  # 해당 피처를 0으로 (스케일링 기준 평균)

        perturbed_pred, _, _ = ensemble.predict(X_perturbed)
        delta = base_pred - perturbed_pred[0]
        impacts.append((f_idx, delta))

    # 절대 영향력 기준 정렬
    impacts.sort(key=lambda x: abs(x[1]), reverse=True)

    results = []
    for f_idx, delta in impacts[:top_k]:
        name = feature_names[f_idx] if f_idx < len(feature_names) else f"feature_{f_idx}"
        direction = "↑" if delta > 0 else "↓"
        results.append((name, direction, abs(delta)))

    return results


def format_shap_results(top_features, feature_values=None):
    """SHAP 결과를 슬랙용 문자열로 변환"""
    lines = []
    for i, (name, direction, score) in enumerate(top_features, 1):
        # 피처 이름을 읽기 쉽게 변환
        display_name = _prettify_feature_name(name)
        lines.append(f"  {i}. {display_name} ({direction})")
    return "\n".join(lines)


def _prettify_feature_name(name):
    """피처 이름을 사람이 읽기 쉽게 변환"""
    mapping = {
        "sp500_ret1d": "S&P500 전일",
        "nasdaq_ret1d": "나스닥 전일",
        "dow_ret1d": "다우 전일",
        "vix_ret1d": "VIX 변화",
        "vix": "VIX 수준",
        "usdkrw_ret1d": "달러/원 변화",
        "usdkrw": "달러/원 환율",
        "nikkei_ret1d": "닛케이 전일",
        "hangseng_ret1d": "항셍 전일",
        "wti_ret1d": "유가 변화",
        "copper_ret1d": "구리 변화",
        "rsi_14": "RSI(14)",
        "rsi_7": "RSI(7)",
        "macd_hist": "MACD 히스토그램",
        "bb_position": "볼린저밴드 위치",
        "stoch_k": "스토캐스틱 %K",
        "return_1d": "코스피 전일 수익률",
        "return_5d": "코스피 5일 수익률",
        "volatility_20d": "20일 변동성",
        "volume_ma5_ratio": "거래량/5일평균",
        "corr_sp500_20d": "S&P500 상관계수",
        "ma_20_ratio": "20일선 이격도",
        "us10y_change": "미국 금리 변화",
        "dxy_ret1d": "달러인덱스 변화",
    }
    return mapping.get(name, name.replace("_", " "))
