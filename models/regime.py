"""시장 레짐 감지 모듈

VIX, 이동평균 배열, 변동성 기반으로 상승장/하락장/횡보장 분류.
레짐별 다른 임계값을 적용하여 보수적/공격적 운영.
"""
import logging
import sqlite3

import numpy as np

from config.settings import DB_PATH

logger = logging.getLogger(__name__)

# 레짐 정의
REGIME_BULL = "bull"       # 상승장
REGIME_BEAR = "bear"       # 하락장
REGIME_SIDEWAYS = "sideways"  # 횡보장

# 레짐별 신뢰도 임계값
REGIME_THRESHOLDS = {
    REGIME_BULL: 50.0,      # 상승장: 공격적 (50% 이상이면 신호)
    REGIME_SIDEWAYS: 60.0,  # 횡보장: 보통
    REGIME_BEAR: 70.0,      # 하락장: 보수적 (70% 이상만 신호)
}

REGIME_LABELS = {
    REGIME_BULL: "🟢 상승장",
    REGIME_BEAR: "🔴 하락장",
    REGIME_SIDEWAYS: "🟡 횡보장",
}


def detect_regime(db_path=None):
    """현재 시장 레짐 감지

    판단 기준:
    1. 코스피 20일 MA vs 60일 MA 배열
    2. VIX 수준
    3. 최근 20일 수익률
    4. 최근 변동성 수준

    Returns:
        (regime, details): 레짐 문자열과 판단 근거
    """
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)

    try:
        # 코스피 최근 80일 종가
        rows = conn.execute(
            "SELECT close FROM kospi_index ORDER BY date DESC LIMIT 80"
        ).fetchall()
        kospi_closes = np.array([r[0] for r in rows])[::-1]  # 오래된→최신

        # VIX 최신값
        vix_row = conn.execute(
            "SELECT close FROM yahoo_vix ORDER BY date DESC LIMIT 1"
        ).fetchone()
        vix = vix_row[0] if vix_row else 20.0
    finally:
        conn.close()

    if len(kospi_closes) < 60:
        return REGIME_SIDEWAYS, {"reason": "데이터 부족", "vix": vix}

    # 지표 계산
    ma20 = np.mean(kospi_closes[-20:])
    ma60 = np.mean(kospi_closes[-60:])
    current_price = kospi_closes[-1]

    # 20일 수익률
    ret_20d = (kospi_closes[-1] / kospi_closes[-20] - 1) * 100

    # 20일 변동성 (연환산)
    daily_rets = np.diff(kospi_closes[-21:]) / kospi_closes[-21:-1]
    volatility = np.std(daily_rets) * np.sqrt(252) * 100

    # 점수 기반 판단 (-3 ~ +3)
    score = 0

    # MA 배열: 가격 > MA20 > MA60 → 상승장
    if current_price > ma20 > ma60:
        score += 2
    elif current_price < ma20 < ma60:
        score -= 2
    elif current_price > ma20:
        score += 1
    elif current_price < ma20:
        score -= 1

    # 20일 수익률
    if ret_20d > 3:
        score += 1
    elif ret_20d < -3:
        score -= 1

    # VIX
    if vix < 18:
        score += 1
    elif vix > 28:
        score -= 1

    # 변동성
    if volatility > 25:
        score -= 1

    # 레짐 결정
    if score >= 2:
        regime = REGIME_BULL
    elif score <= -2:
        regime = REGIME_BEAR
    else:
        regime = REGIME_SIDEWAYS

    details = {
        "score": score,
        "ma20": round(ma20, 1),
        "ma60": round(ma60, 1),
        "price": round(current_price, 1),
        "ret_20d": round(ret_20d, 2),
        "volatility": round(volatility, 1),
        "vix": round(vix, 1),
    }

    logger.info(f"레짐 감지: {REGIME_LABELS[regime]} (score={score}, VIX={vix:.1f}, ret20d={ret_20d:+.1f}%)")
    return regime, details


def get_regime_threshold(regime):
    """레짐에 따른 신뢰도 임계값 반환"""
    return REGIME_THRESHOLDS.get(regime, 60.0)
