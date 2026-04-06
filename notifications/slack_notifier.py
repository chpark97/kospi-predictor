"""슬랙 웹훅 알림 모듈

환경변수 SLACK_WEBHOOK_URL에 웹훅 URL을 설정하세요.
예: export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/XXX/YYY/ZZZ"
"""
import json
import logging
import os

import requests

logger = logging.getLogger(__name__)

SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")


def send_prediction_alert(result: dict):
    """예측 결과를 슬랙으로 전송

    Args:
        result: run_prediction()이 반환하는 dict
            - date, direction, predicted_return, confidence,
              signal_valid, vix, agreement, up_vote,
              risk_warnings, sentiment
    """
    date = result.get("date", "N/A")
    pred_ret = result.get("predicted_return", 0)
    confidence = result.get("confidence", 0)
    signal_valid = result.get("signal_valid", False)
    vix = result.get("vix")
    agreement = result.get("agreement", 0)
    up_vote = result.get("up_vote", "?/?")
    sentiment = result.get("sentiment")
    warnings = result.get("risk_warnings", [])

    # 방향
    if not signal_valid:
        direction_str = "― 신호없음"
    elif pred_ret > 0:
        direction_str = "▲ 상승"
    else:
        direction_str = "▼ 하락"

    # 등락률 부호
    sign = "+" if pred_ret > 0 else ""

    # VIX
    if vix is not None:
        vix_status = "경고" if vix >= 30 else "정상"
        vix_str = f"{vix:.1f} ({vix_status})"
    else:
        vix_str = "N/A"

    # 감성
    if sentiment is not None:
        if sentiment > 0.1:
            sent_label = "긍정"
        elif sentiment < -0.1:
            sent_label = "부정"
        else:
            sent_label = "중립"
        sent_str = f"{sent_label} ({sentiment:+.1f})"
    else:
        sent_str = "N/A"

    # 신호 상태
    if signal_valid:
        status_line = "✅ 거래 신호 있음"
    else:
        status_line = "⚠️ 리스크 필터 발동"

    # 메시지 조립
    text = (
        f"📊 *[{date}] 코스피 예측*\n\n"
        f"방향: *{direction_str}*\n"
        f"예측 등락률: *{sign}{pred_ret:.2f}%*\n"
        f"신뢰도: {confidence:.1f}%\n"
        f"모델 합의: {up_vote}\n"
        f"VIX: {vix_str}\n"
        f"감성: {sent_str}\n\n"
        f"{status_line}"
    )

    if warnings:
        text += "\n" + "\n".join(warnings)

    payload = {"text": text}

    if not SLACK_WEBHOOK_URL:
        logger.warning("[Slack] SLACK_WEBHOOK_URL 환경변수가 설정되지 않았습니다")
        return

    try:
        resp = requests.post(
            SLACK_WEBHOOK_URL,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if resp.status_code == 200:
            logger.info("[Slack] 알림 전송 완료")
        else:
            logger.warning(f"[Slack] 전송 실패: {resp.status_code} {resp.text}")
    except Exception as e:
        logger.warning(f"[Slack] 전송 오류: {e}")
