"""예측 확률 보정 (Calibration)

상승 편향을 감지하고 오프셋 기반으로 방향까지 교정.
기존 scale 방식은 크기만 줄이고 부호(방향)를 바꾸지 못하는 문제를 수정.
"""
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CALIBRATION_PATH = Path(__file__).parent.parent / "data" / "calibration_state.json"


def detect_bias(prediction_history, window=30):
    """최근 N일 상승/하락 예측 비율 분석

    Returns:
        (up_ratio, actual_up_ratio, is_biased, msg)
    """
    recent = prediction_history[-window:] if len(prediction_history) >= window else prediction_history
    if len(recent) < 10:
        return 0.5, 0.5, False, None

    up_count = sum(1 for h in recent if h.get("predicted_direction") == "up")
    up_ratio = up_count / len(recent)

    # 실제 상승 비율
    verified = [h for h in recent if h.get("actual_direction")]
    actual_up = sum(1 for h in verified if h["actual_direction"] == "up") / len(verified) if verified else 0.5

    is_biased = up_ratio > 0.55  # 55% 초과면 편향 (기존 0.65에서 하향)
    msg = None

    if is_biased:
        msg = (
            f"⚠️ *상승 편향 감지*: 최근 {len(recent)}일 상승 예측 {up_ratio:.0%}"
            f" (실제 상승 {actual_up:.0%})\n→ 오프셋 보정 적용 중"
        )
        logger.warning(f"[Calibration] 편향: 예측상승={up_ratio:.0%}, 실제상승={actual_up:.0%}")

    return up_ratio, actual_up, is_biased, msg


def calibrate_prediction(pred_return, individual_returns, prediction_history):
    """편향 감지 시 오프셋 기반으로 예측 등락률을 보정

    기존 scale 방식의 문제:
      pred_return * scale → 양수에 양수를 곱하면 여전히 양수 (방향 불변)

    개선된 오프셋 방식:
      1) 예측 상승 비율과 실제 상승 비율의 차이를 오프셋으로 변환
      2) 개별 모델 예측의 분산을 고려하여 오프셋 크기 조절
      3) 이를 통해 약한 상승 예측을 하락으로 전환 가능
    """
    up_ratio, actual_up, is_biased, bias_msg = detect_bias(prediction_history)

    if not is_biased:
        return pred_return, individual_returns, None

    # 오프셋 계산: 예측값 대비 비율 기반으로 부호 전환 가능하도록 개선
    # 기존: bias_gap * spread → 예측값(0.04~0.30%)보다 오프셋이 작아 부호 전환 불가
    # 개선: 예측값 절대값 대비 비율 기반으로 오프셋 계산
    bias_gap = up_ratio - actual_up  # up_ratio=0.8, actual_up=0.5 → bias_gap=0.3

    individual_rets = individual_returns[:, 0] if individual_returns.ndim > 1 else individual_returns

    # 오프셋을 예측값 절대값의 배수로 계산 (부호 전환 가능)
    # bias_gap=0.3 → multiplier=0.6, bias_gap=0.5 → multiplier=1.0
    pred_abs = abs(pred_return)
    multiplier = bias_gap * 2  # bias_gap 30%면 예측값의 60%를 오프셋으로
    offset = pred_abs * multiplier

    # 추가 임계값 보정: 편향이 극심하면(up_ratio > 0.8) 더 강하게 보정
    threshold_adj = max(0, (up_ratio - 0.7)) * pred_abs  # 70% 초과분만 추가

    total_offset = offset + threshold_adj

    calibrated_return = pred_return - total_offset
    calibrated_individual = individual_returns - total_offset

    logger.info(
        f"  [Calibration] 오프셋 보정: {pred_return:.3f}% → {calibrated_return:.3f}% "
        f"(offset={total_offset:.3f}, bias_gap={bias_gap:.2f}, spread={spread:.3f})"
    )

    return calibrated_return, calibrated_individual, bias_msg
