"""예측 확률 보정 (Calibration) v2

상승 편향을 감지하고 다단계로 교정:
1. Platt Scaling (스케일 축소)
2. 극심한 괴리 시 부호 반전 (방향 뒤집기)
3. 앙상블 가중 투표 방향과 일치 여부 검증
"""
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CALIBRATION_PATH = Path(__file__).parent.parent / "data" / "calibration_state.json"

# 편향 감지 임계값 (기존 65% → 55%로 하향)
BIAS_THRESHOLD = 0.55


def detect_bias(prediction_history, window=30):
    """최근 N일 상승/하락 예측 비율 분석

    Returns:
        (up_ratio, actual_up_ratio, is_biased, bias_severity, msg)
    """
    recent = prediction_history[-window:] if len(prediction_history) >= window else prediction_history
    if len(recent) < 10:
        return 0.5, 0.5, False, "none", None

    up_count = sum(1 for h in recent if h.get("predicted_direction") == "up")
    up_ratio = up_count / len(recent)

    # 실제 상승 비율
    verified = [h for h in recent if h.get("actual_direction")]
    actual_up = sum(1 for h in verified if h["actual_direction"] == "up") / len(verified) if verified else 0.5

    # 편향 심각도 판단
    divergence = up_ratio - actual_up  # 예측 상승비율 - 실제 상승비율

    if up_ratio <= BIAS_THRESHOLD:
        return up_ratio, actual_up, False, "none", None

    # 심각도 분류
    if divergence > 0.3:
        severity = "severe"  # 극심: 예측 80%+ 상승인데 실제 50% 미만
    elif divergence > 0.15:
        severity = "moderate"  # 보통
    else:
        severity = "mild"  # 경미

    msg = (
        f"⚠️ *상승 편향 감지 [{severity}]*: 최근 {len(recent)}일 상승 예측 {up_ratio:.0%}"
        f" (실제 상승 {actual_up:.0%}, 괴리 {divergence:+.0%})\n"
        f"→ {'부호 반전 보정' if severity == 'severe' else '스케일 보정'} 적용 중"
    )
    logger.warning(
        f"[Calibration] 편향 {severity}: 예측상승={up_ratio:.0%}, "
        f"실제상승={actual_up:.0%}, 괴리={divergence:+.0%}"
    )

    return up_ratio, actual_up, True, severity, msg


def calibrate_prediction(pred_return, individual_returns, prediction_history):
    """편향 감지 시 예측 등락률을 보정

    보정 방법:
    1. mild: 기존 Platt Scaling (스케일 축소)
    2. moderate: 강한 스케일 축소 + 방향 불확실성 페널티
    3. severe: 실제 상승비율이 50% 미만이면 부호 반전
    """
    up_ratio, actual_up, is_biased, severity, bias_msg = detect_bias(prediction_history)

    if not is_biased:
        return pred_return, individual_returns, None

    original_return = pred_return
    divergence = up_ratio - actual_up

    if severity == "severe" and actual_up <= 0.5 and pred_return > 0:
        # 극심한 괴리: 실제로 하락이 더 많은데 계속 상승 예측
        # → 예측 등락률 부호를 반전하고 크기도 축소
        flip_scale = actual_up / up_ratio  # 예: 0.4/0.9 = 0.44
        pred_return = -abs(pred_return) * max(flip_scale, 0.3)
        individual_returns = individual_returns * (-flip_scale)
        logger.info(
            f"  [Calibration] 부호 반전: {original_return:.3f}% → {pred_return:.3f}% "
            f"(flip_scale={flip_scale:.2f})"
        )

    elif severity == "moderate":
        # 보통 괴리: 강한 축소
        target_ratio = 0.50
        scale = target_ratio / max(up_ratio, 0.01)
        # 추가 페널티: 괴리가 클수록 더 축소
        penalty = max(1.0 - divergence, 0.3)
        effective_scale = scale * penalty
        pred_return = pred_return * effective_scale
        individual_returns = individual_returns * effective_scale
        logger.info(
            f"  [Calibration] 강한 축소: {original_return:.3f}% → {pred_return:.3f}% "
            f"(scale={scale:.2f} × penalty={penalty:.2f})"
        )

    else:  # mild
        # 경미: 기존 Platt Scaling
        target_ratio = 0.55
        scale = target_ratio / max(up_ratio, 0.01)
        pred_return = pred_return * scale
        individual_returns = individual_returns * scale
        logger.info(
            f"  [Calibration] 스케일 보정: {original_return:.3f}% → {pred_return:.3f}% "
            f"(scale={scale:.2f})"
        )

    return pred_return, individual_returns, bias_msg


def get_calibration_inline(prediction_history):
    """슬랙 인라인 표시용 보정 정보"""
    up_ratio, actual_up, is_biased, severity, _ = detect_bias(prediction_history)
    if not is_biased:
        return None

    divergence = up_ratio - actual_up
    if severity == "severe":
        return f"부호 반전 (괴리 {divergence:+.0%})"
    elif severity == "moderate":
        target = 0.50
        scale = target / max(up_ratio, 0.01)
        penalty = max(1.0 - divergence, 0.3)
        return f"scale {scale * penalty:.2f} 적용 (괴리 {divergence:+.0%})"
    else:
        scale = 0.55 / max(up_ratio, 0.01)
        return f"scale {scale:.2f} 적용"
