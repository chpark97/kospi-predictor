"""ETF 매매 신호 전략 시스템

신뢰도와 레짐에 따른 양방향 ETF 자동 선택:
- 상승 예측: KODEX 200 / TIGER 200 / KODEX 레버리지
- 하락 예측: KODEX 인버스 / TIGER 인버스 / KODEX 200선물인버스2X
- 지정학적 위험 / 신뢰도 미달 → 현금 보유
"""
import logging

logger = logging.getLogger(__name__)

# ETF 종목 정의
ETF_CATALOG = {
    "KODEX 200": {
        "code": "069500",
        "direction": "long",
        "leverage": 1.0,
        "multiplier": 1.0,
        "risk": "안전",
        "description": "코스피 1배 추종",
    },
    "TIGER 200": {
        "code": "102110",
        "direction": "long",
        "leverage": 1.0,
        "multiplier": 1.0,
        "risk": "안전",
        "description": "코스피 1배 추종",
    },
    "KODEX 레버리지": {
        "code": "122630",
        "direction": "long",
        "leverage": 2.0,
        "multiplier": 1.95,  # 레버리지 비용 고려
        "risk": "공격",
        "description": "코스피 2배 추종 (일별 복리)",
    },
    "KODEX 인버스": {
        "code": "114800",
        "direction": "short",
        "leverage": 1.0,
        "multiplier": -1.0,
        "risk": "안전",
        "description": "코스피 역방향 1배",
    },
    "TIGER 인버스": {
        "code": "123310",
        "direction": "short",
        "leverage": 1.0,
        "multiplier": -1.0,
        "risk": "안전",
        "description": "코스피 역방향 1배",
    },
    "KODEX 200선물인버스2X": {
        "code": "252670",
        "direction": "short",
        "leverage": 2.0,
        "multiplier": -1.9,  # 인버스 레버리지 비용 고려
        "risk": "공격",
        "description": "코스피 역방향 2배 (일별 복리)",
    },
}

# 현금 (거래 없음)
CASH_SIGNAL = {
    "etf_name": "현금",
    "etf_code": None,
    "direction": "none",
    "strategy_desc": "현금 보유",
    "multiplier": 0.0,
    "leverage": 0.0,
    "risk": "안전",
    "reason": "",
    "sizing_ratio": 0.0,
    "expected_return": 0.0,
}


def select_etf(predicted_direction, confidence, regime, geo_level="safe",
               mc_level="medium", bias_corrected=False, pred_return=0.0):
    """ETF 매매 신호 결정

    결정 로직:
    1. 지정학적 필터 → 위험 시 현금
    2. 신뢰도 미달 → 현금
    3. 편향 보정 차단 → 현금 (signal_valid=False인 경우 외부에서 처리)
    4. 레짐 + 방향 + 신뢰도 → ETF 종목 선택
    5. MC 불확실성 → 포지션 크기 결정

    Args:
        predicted_direction: "up" or "down"
        confidence: 복합 신뢰도 (0~100)
        regime: "bull" / "bear" / "sideways"
        geo_level: "safe" / "caution" / "danger"
        mc_level: "low" / "medium" / "high"
        bias_corrected: 편향 보정이 적용되었는지
        pred_return: 예측 등락률 (%)

    Returns:
        dict: ETF 신호 정보
    """
    # 1. 지정학적 위험 → 현금
    if geo_level == "danger":
        signal = CASH_SIGNAL.copy()
        signal["reason"] = "지정학적 리스크 → 거래 중단"
        logger.info("  [ETF] 지정학적 위험 → 현금 보유")
        return signal

    # 2. 신뢰도 70% 미만 → 현금
    if confidence < 70:
        signal = CASH_SIGNAL.copy()
        signal["reason"] = f"신뢰도 미달 ({confidence:.1f}% < 70%)"
        logger.info(f"  [ETF] 신뢰도 미달 ({confidence:.1f}%) → 현금 보유")
        return signal

    # 3. MC 불확실성 기반 포지션 사이징
    sizing_map = {"low": 1.0, "medium": 0.6, "high": 0.3}
    sizing_ratio = sizing_map.get(mc_level, 0.6)

    # 4. 방향 + 신뢰도 + 레짐 → ETF 선택
    if predicted_direction == "up":
        etf_name, strategy_desc = _select_long_etf(confidence, regime)
    else:
        etf_name, strategy_desc = _select_short_etf(confidence, regime)

    etf_info = ETF_CATALOG[etf_name]
    expected_return = abs(pred_return) * abs(etf_info["multiplier"])

    signal = {
        "etf_name": etf_name,
        "etf_code": etf_info["code"],
        "direction": etf_info["direction"],
        "strategy_desc": strategy_desc,
        "multiplier": etf_info["multiplier"],
        "leverage": etf_info["leverage"],
        "risk": etf_info["risk"],
        "reason": f"{regime} + 신뢰도 {confidence:.0f}%",
        "sizing_ratio": sizing_ratio,
        "expected_return": round(expected_return, 2),
    }

    logger.info(
        f"  [ETF] {etf_name} ({strategy_desc}) | "
        f"투입 {sizing_ratio:.0%} | 예상 수익 {expected_return:+.2f}%"
    )
    return signal


def _select_long_etf(confidence, regime):
    """상승 예측 시 ETF 선택"""
    # 신뢰도 90%+ & 상승장 → KODEX 레버리지 (2배, 공격)
    if confidence >= 90 and regime == "bull":
        return "KODEX 레버리지", "정방향 2배 (상승장 + 고신뢰)"

    # 신뢰도 80~90% & 상승장 → TIGER 200 (1배)
    if confidence >= 80 and regime == "bull":
        return "TIGER 200", "정방향 1배 (상승장 + 신뢰도 80%+)"

    # 신뢰도 70~80% & 횡보/상승장 → KODEX 200 (1배, 안전)
    if confidence >= 70 and regime in ("sideways", "bull"):
        return "KODEX 200", "정방향 1배 (안전)"

    # 하락장에서 상승 예측 → 보수적으로 KODEX 200
    return "KODEX 200", "정방향 1배 (하락장 내 상승 신호)"


def _select_short_etf(confidence, regime):
    """하락 예측 시 ETF 선택"""
    # 신뢰도 90%+ & 하락장 → KODEX 200선물인버스2X (2배, 공격)
    if confidence >= 90 and regime == "bear":
        return "KODEX 200선물인버스2X", "역방향 2배 (하락장 + 고신뢰)"

    # 신뢰도 80~90% & 하락장 → TIGER 인버스 (1배)
    if confidence >= 80 and regime == "bear":
        return "TIGER 인버스", "역방향 1배 (하락장 + 신뢰도 80%+)"

    # 신뢰도 70~80% & 횡보/하락장 → KODEX 인버스 (1배, 안전)
    if confidence >= 70 and regime in ("sideways", "bear"):
        return "KODEX 인버스", "역방향 1배 (안전)"

    # 상승장에서 하락 예측 → 보수적으로 KODEX 인버스
    return "KODEX 인버스", "역방향 1배 (상승장 내 하락 신호)"


def format_etf_signal(signal):
    """슬랙용 ETF 매매 신호 포맷"""
    if signal["etf_name"] == "현금":
        return f"📈 *ETF 매매 신호*\n  현금 보유 ({signal['reason']})"

    sizing_pct = int(signal["sizing_ratio"] * 100)
    sign = "+" if signal["expected_return"] > 0 else ""

    return (
        f"📈 *ETF 매매 신호*\n"
        f"  추천 ETF: {signal['etf_name']}\n"
        f"  전략: {signal['strategy_desc']}\n"
        f"  투입비율: {sizing_pct}% ({signal['risk']})\n"
        f"  예상 수익: {sign}{signal['expected_return']:.2f}%"
    )
