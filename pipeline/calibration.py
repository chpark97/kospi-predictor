"""예측 확률 보정 (Calibration)

상승 편향을 감지하고 Platt Scaling으로 교정.
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
        (up_ratio, is_biased, msg)
    """
    recent = prediction_history[-window:] if len(prediction_history) >= window else prediction_history
    if len(recent) < 10:
        return 0.5, False, None

    up_count = sum(1 for h in recent if h.get("predicted_direction") == "up")
    up_ratio = up_count / len(recent)

    # 실제 상승 비율
    verified = [h for h in recent if h.get("actual_direction")]
    actual_up = sum(1 for h in verified if h["actual_direction"] == "up") / len(verified) if verified else 0.5

    is_biased = up_ratio > 0.65  # 65% 초과면 편향
    msg = None

    if is_biased:
        msg = (
            f"⚠️ *상승 편향 감지*: 최근 {len(recent)}일 상승 예측 {up_ratio:.0%}"
            f" (실제 상승 {actual_up:.0%})\n→ 확률 보정 적용 중"
        )
        logger.warning(f"[Calibration] 편향: 예측상승={up_ratio:.0%}, 실제상승={actual_up:.0%}")

    return up_ratio, is_biased, msg


def calibrate_prediction(pred_return, individual_returns, prediction_history):
    """편향 감지 시 예측 등락률을 보정

    방법: 최근 상승 예측이 65% 초과면
          예측 등락률을 실제 상승비율 방향으로 축소
    """
    up_ratio, is_biased, bias_msg = detect_bias(prediction_history)

    if not is_biased:
        return pred_return, individual_returns, None

    # 보정 계수: 편향이 클수록 강하게 축소
    # 예: up_ratio=0.8 → scale=0.625 (상승 예측 축소)
    target_ratio = 0.55  # 목표 상승 예측 비율
    scale = target_ratio / up_ratio

    calibrated_return = pred_return * scale
    calibrated_individual = individual_returns * scale

    logger.info(f"  [Calibration] 보정: {pred_return:.3f}% → {calibrated_return:.3f}% (scale={scale:.2f})")

    return calibrated_return, calibrated_individual, bias_msg
