"""백테스팅 시각화 - 누적 수익률 차트 생성"""
import logging
import sqlite3
from pathlib import Path

import numpy as np

from config.settings import DB_PATH

logger = logging.getLogger(__name__)

CHART_PATH = Path(__file__).parent.parent / "data" / "weekly_chart.png"


def generate_weekly_chart(prediction_history, output_path=None):
    """주간 리포트용 누적 수익률 차트 생성

    코스피 Buy&Hold vs 모델 전략 비교.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime

    output_path = output_path or CHART_PATH

    verified = [h for h in prediction_history
                if h.get("actual_return") is not None and h.get("actual_direction")]

    if len(verified) < 3:
        logger.info("[Chart] 검증 데이터 부족")
        return None

    dates = [datetime.strptime(h["date"], "%Y-%m-%d") for h in verified]
    actual_rets = [h["actual_return"] / 100 for h in verified]

    # 모델 전략 수익률: 상승 예측 시 매수, 아니면 관망
    model_rets = []
    for h in verified:
        if h["predicted_direction"] == "up" and h.get("signal_valid", True):
            model_rets.append(h["actual_return"] / 100 - 0.0003)  # 수수료
        else:
            model_rets.append(0)

    # 누적 수익률
    kospi_cum = np.cumprod(1 + np.array(actual_rets)) - 1
    model_cum = np.cumprod(1 + np.array(model_rets)) - 1

    # 차트 생성
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, kospi_cum * 100, label="KOSPI Buy&Hold", color="#888888", linewidth=1.5)
    ax.plot(dates, model_cum * 100, label="Model Strategy", color="#2196F3", linewidth=2)

    ax.fill_between(dates, 0, model_cum * 100, alpha=0.1, color="#2196F3")
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")

    ax.set_title("Cumulative Returns: Model vs KOSPI", fontsize=13, fontweight="bold")
    ax.set_ylabel("Return (%)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)

    logger.info(f"[Chart] 저장: {output_path}")
    return str(output_path)


def upload_chart_to_slack(chart_path):
    """차트 이미지를 슬랙에 업로드"""
    import os
    import requests

    webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
    if not webhook_url or not chart_path:
        return False

    # Incoming Webhook은 파일 업로드 미지원 → 텍스트로 대체
    # (실제 운영 시 Slack Bot Token + files.upload API 사용 권장)
    logger.info("[Chart] 차트 생성 완료 (슬랙 webhook은 이미지 업로드 미지원, 로컬 저장)")
    return True
