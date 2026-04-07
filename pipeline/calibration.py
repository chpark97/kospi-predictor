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

    is_biased = up_ratio > 0.65  # 65% 초과면 편향
    msg = None

    if is_biased:
        msg = (
            f"⚠️ *상승 편향 감지*: 최근 {len(recent)}일 상승 예측 {up_ratio:.0%}"
            f" (실제 상승 {actual_up:.0%})\n→ 오프셋 보정 적용 중"
        )
        logger.warning(f"[Calibration] 편향: 예측상승={up_ratio:.0%}, 실제상승={actual_up:.0%}")

    return up_ratio, actual_up, is_biased, msg


def calibrate_prediction(pred_return, individual_returns, prediction_history):
    """편향 감지 시 분위수 기반으로 예측 등락률을 보정

    핵심: 예측 상승 비율(97%)을 실제 상승 비율(60%)에 맞추는 오프셋 계산.
    최근 예측값의 (1-실제상승비율) 분위수를 빼면,
    보정 후 상승 예측 비율이 실제 상승 비율에 근접함.

    예: 실제 하락 40% → 예측값의 40th percentile을 오프셋으로 사용
    → 보정 후 약 60%가 양수(상승), 40%가 음수(하락)
    """
    up_ratio, actual_up, is_biased, bias_msg = detect_bias(prediction_history)

    if not is_biased:
        return pred_return, individual_returns, None

    # 최근 예측값 수집
    recent_preds = [
        h["predicted_return"] for h in prediction_history[-30:]
        if h.get("predicted_return") is not None
    ]
    if not recent_preds or len(recent_preds) < 5:
        return pred_return, individual_returns, bias_msg

    # 분위수 기반 오프셋: 실제 하락 비율에 해당하는 예측값 분위수
    # actual_up=0.6 → down_ratio=0.4 → 40th percentile을 빼면 60%가 양수로 남음
    down_ratio = 1.0 - actual_up
    offset = float(np.percentile(recent_preds, down_ratio * 100))

    calibrated_return = pred_return - offset
    calibrated_individual = individual_returns - offset

    logger.info(
        f"  [Calibration] 분위수 보정: {pred_return:.3f}% → {calibrated_return:.3f}% "
        f"(offset={offset:.4f}, target_up={actual_up:.0%}, pred_up={up_ratio:.0%})"
    )

    return calibrated_return, calibrated_individual, bias_msg
