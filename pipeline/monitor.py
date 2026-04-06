"""모델 성능 모니터링

최근 N일 방향정확도를 추적하고,
성능 저하 시 슬랙 경고를 발송.
"""
import logging

logger = logging.getLogger(__name__)

ALERT_THRESHOLD = 45.0  # 이 이하면 경고


def check_model_performance(prediction_history, n_days=10):
    """최근 N일 방향정확도 계산 및 경고

    Returns:
        (accuracy, alert_msg) — alert_msg는 경고 필요 시에만 값
    """
    verified = [h for h in prediction_history
                if h.get("actual_direction") is not None]

    recent = verified[-n_days:] if len(verified) >= n_days else verified

    if len(recent) < 5:
        return None, None

    correct = sum(1 for h in recent
                  if h["predicted_direction"] == h["actual_direction"])
    accuracy = correct / len(recent) * 100

    alert_msg = None
    if accuracy <= ALERT_THRESHOLD:
        alert_msg = (
            f"⚠️ *모델 성능 저하 감지*\n"
            f"최근 {len(recent)}일 정확도: {accuracy:.1f}%\n"
            f"→ 재학습 권장 (`python main.py retrain`)"
        )
        logger.warning(f"[Monitor] 성능 저하: {accuracy:.1f}% (최근 {len(recent)}일)")
    else:
        logger.info(f"[Monitor] 최근 {len(recent)}일 정확도: {accuracy:.1f}% (정상)")

    return accuracy, alert_msg
