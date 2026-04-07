"""슬랙 웹훅 알림 모듈"""
import json
import logging
import os

import requests

logger = logging.getLogger(__name__)

SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")


def _post_to_slack(text):
    if not SLACK_WEBHOOK_URL:
        logger.warning("[Slack] SLACK_WEBHOOK_URL 미설정")
        return False
    try:
        resp = requests.post(
            SLACK_WEBHOOK_URL,
            data=json.dumps({"text": text}),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if resp.status_code == 200:
            logger.info("[Slack] 알림 전송 완료")
            return True
        logger.warning(f"[Slack] 전송 실패: {resp.status_code}")
        return False
    except Exception as e:
        logger.warning(f"[Slack] 전송 오류: {e}")
        return False


def send_slack_message(text):
    return _post_to_slack(text)


def send_prediction_alert(result: dict):
    """최종 슬랙 예측 알림 — 모든 기능 통합"""
    date = result.get("date", "N/A")
    pred_ret = result.get("predicted_return", 0)
    confidence = result.get("confidence", 0)
    signal_valid = result.get("signal_valid", False)
    vix = result.get("vix")
    up_vote = result.get("up_vote", "?/?")
    agreement = result.get("agreement", 0)
    sentiment = result.get("sentiment")
    regime_label = result.get("regime_label", "")
    threshold = result.get("confidence_threshold", 65)
    ms_str = result.get("multistep", "")
    top_features = result.get("top_features", [])
    mc_std = result.get("mc_std", 0)
    mc_emoji = result.get("mc_emoji", "")
    mc_label = result.get("mc_label", "")
    portfolio = result.get("portfolio", "")
    meta_proba = result.get("meta_proba")

    # 방향 및 매매 전략 (일일 양방향 매매)
    if pred_ret > 0:
        direction_str = "▲ 상승"
        trade_action = "오늘 롱 매수"
    else:
        direction_str = "▼ 하락"
        trade_action = "오늘 인버스 매수"

    sign = "+" if pred_ret > 0 else ""

    # VIX
    vix_str = f"{vix:.1f} ({'경고' if vix >= 30 else '정상'})" if vix else "N/A"

    # 감성
    if sentiment is not None:
        sent_label = "긍정" if sentiment > 0.1 else ("부정" if sentiment < -0.1 else "중립")
        sent_str = f"{sent_label} ({sentiment:+.1f})"
    else:
        sent_str = "N/A"

    status_line = "✅ 거래 신호 있음" if signal_valid else "⚠️ 리스크 필터 발동"

    # 주요 근거
    reasons = ""
    if top_features:
        lines = [f"  {i}. {n} ({d})" for i, (n, d) in enumerate(top_features[:3], 1)]
        reasons = "\n🔍 *주요 근거:*\n" + "\n".join(lines)

    # 메시지 조립
    event = result.get("event", "")

    text = f"📊 *[{date}] 코스피 예측*\n\n"
    text += f"🌍 시장 레짐: {regime_label}\n"

    if event:
        text += f"📅 이벤트: {event}\n"

    if ms_str:
        text += f"📅 단기 전망: {ms_str}\n"

    text += (
        f"\n방향: *{direction_str}*\n"
        f"매매: *{trade_action}* (장중 청산)\n"
        f"예측 등락률: *{sign}{pred_ret:.2f}%* (±{mc_std:.2f}%) {mc_emoji} {mc_label}\n"
        f"신뢰도: {confidence:.1f}% (임계 {threshold:.0f}%)\n"
        f"모델 합의: {up_vote} ({agreement:.0f}%)\n"
    )

    if meta_proba is not None:
        text += f"메타 모델: {'상승' if meta_proba > 0.5 else '하락'} ({meta_proba:.0%})\n"

    text += f"VIX: {vix_str}\n"
    text += f"감성: {sent_str}"

    if reasons:
        text += f"\n{reasons}"

    text += f"\n\n{status_line}"

    warnings = result.get("risk_warnings", [])
    if warnings:
        text += "\n" + "\n".join(warnings)

    if portfolio:
        text += f"\n\n{portfolio}"

    return _post_to_slack(text)
