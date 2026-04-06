"""ETF 백테스팅 모듈 — 4가지 전략 비교 (2022~2024)

전략:
  A. Buy & Hold (KODEX 200만 보유)
  B. 기존 단순 매수/현금 전략
  C. ETF 양방향 전략 (레버리지 없음)
  D. ETF 양방향 전략 (레버리지/인버스 포함)
"""
import logging
import sqlite3

import numpy as np

from config.settings import COMMISSION_RATE, DB_PATH

logger = logging.getLogger(__name__)

INITIAL_CAPITAL = 100_000_000
STOP_LOSS = -2.0
TAKE_PROFIT = 3.0
MAX_HOLD_DAYS = 5


def _compute_metrics(daily_values, wins, losses, trades):
    """전략 지표 계산"""
    daily_vals = np.array(daily_values)
    final_return = (daily_vals[-1] / daily_vals[0] - 1) * 100

    daily_rets = np.diff(daily_vals) / daily_vals[:-1]
    sharpe = 0.0
    if len(daily_rets) > 1 and np.std(daily_rets) > 0:
        sharpe = float(np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252))

    # MDD
    running_max = np.maximum.accumulate(daily_vals)
    drawdowns = (daily_vals / running_max - 1) * 100
    mdd = float(np.min(drawdowns))

    total_trades = wins + losses
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0.0

    return {
        "total_return": round(final_return, 2),
        "sharpe": round(sharpe, 2),
        "mdd": round(mdd, 1),
        "wins": wins,
        "losses": losses,
        "trades": trades,
        "win_rate": round(win_rate, 1),
    }


def _simulate_position(closes, i, entry_price, multiplier, hold_days,
                        invested_amount):
    """포지션 수익률 계산 (ETF multiplier 적용)"""
    kospi_change = (closes[i] / entry_price - 1)
    return kospi_change * multiplier * 100  # unrealized %


def run_etf_backtest(db_path=None):
    """ETF 4전략 백테스팅 (2022~2024)

    Returns:
        dict: 4개 전략 결과
    """
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT date, close FROM kospi_index "
        "WHERE date >= '2022-01-01' AND date <= '2024-12-31' ORDER BY date"
    ).fetchall()
    conn.close()

    if len(rows) < 50:
        logger.warning("백테스트 데이터 부족 (50일 미만)")
        return None

    dates = [r[0] for r in rows]
    closes = np.array([r[1] for r in rows])
    n = len(closes)
    daily_rets = np.diff(closes) / closes[:-1]  # 일별 수익률

    # 간단한 모멘텀 시뮬레이션용 신호 생성
    # 실제 모델 없이 전일 수익률 기반 방향 + 변동성 기반 신뢰도
    signals = _generate_backtest_signals(closes, daily_rets)

    results = {}

    # ═══ A. Buy & Hold (KODEX 200) ═══
    results["A_BuyHold"] = _strategy_buy_hold(closes)

    # ═══ B. 기존 단순 매수/현금 ═══
    results["B_LongOnly"] = _strategy_long_only(closes, daily_rets, signals)

    # ═══ C. ETF 양방향 (레버리지 없음) ═══
    results["C_BiDir_1x"] = _strategy_bidirectional(
        closes, daily_rets, signals, use_leverage=False
    )

    # ═══ D. ETF 양방향 (레버리지 포함) ═══
    results["D_BiDir_Lev"] = _strategy_bidirectional(
        closes, daily_rets, signals, use_leverage=True
    )

    # 결과 출력
    _print_results(results)
    return results


def _generate_backtest_signals(closes, daily_rets):
    """백테스트용 매매 신호 생성

    모멘텀 + 변동성 기반:
    - 5일 모멘텀 양수 → 상승 신호
    - 5일 모멘텀 음수 → 하락 신호
    - 20일 변동성 기반 신뢰도 추정
    - 변동성 기반 레짐 (bull/bear/sideways)
    """
    n = len(closes)
    signals = []

    for i in range(n):
        # 5일 모멘텀
        if i >= 5:
            mom5 = (closes[i] / closes[i - 5] - 1) * 100
        else:
            mom5 = 0.0

        # 20일 변동성
        if i >= 20:
            vol20 = np.std(daily_rets[max(0, i - 20):i]) * np.sqrt(252) * 100
        else:
            vol20 = 15.0

        # 3일 단기 모멘텀 (추가)
        if i >= 3:
            mom3 = (closes[i] / closes[i - 3] - 1) * 100
        else:
            mom3 = 0.0

        # 방향 결정 (5일 + 3일 모멘텀 가중 합산)
        combined = mom5 * 0.6 + mom3 * 0.4
        if combined > 0.1:
            direction = "up"
        elif combined < -0.1:
            direction = "down"
        else:
            direction = "up" if mom5 > 0 else "down"

        # 신뢰도 추정 (모멘텀 강도 + 변동성 역수)
        mom_strength = min(abs(combined) / 2.0, 1.0) * 50  # 0~50
        vol_score = max(0, (30 - vol20) / 30) * 50  # 변동성 낮으면 높은 점수
        confidence = 50 + mom_strength + vol_score
        confidence = min(max(confidence, 40), 99)

        # 레짐 (20일 이동평균 vs 60일)
        if i >= 60:
            ma20 = np.mean(closes[i - 19:i + 1])
            ma60 = np.mean(closes[i - 59:i + 1])
            if closes[i] > ma20 > ma60:
                regime = "bull"
            elif closes[i] < ma20 < ma60:
                regime = "bear"
            else:
                regime = "sideways"
        else:
            regime = "sideways"

        # 신호 유효성 (신뢰도 70% 이상)
        signal_valid = confidence >= 70

        # 불확실성 레벨 (변동성 기반)
        if vol20 < 12:
            mc_level = "low"
        elif vol20 < 22:
            mc_level = "medium"
        else:
            mc_level = "high"

        signals.append({
            "direction": direction,
            "confidence": confidence,
            "signal_valid": signal_valid,
            "regime": regime,
            "mc_level": mc_level,
            "pred_return": combined * 0.3,  # 예상 등락률 스케일링
        })

    return signals


def _strategy_buy_hold(closes):
    """A. Buy & Hold (KODEX 200)"""
    daily_values = closes / closes[0] * INITIAL_CAPITAL
    total_return = (closes[-1] / closes[0] - 1) * 100

    running_max = np.maximum.accumulate(daily_values)
    drawdowns = (daily_values / running_max - 1) * 100
    mdd = float(np.min(drawdowns))

    daily_rets = np.diff(daily_values) / daily_values[:-1]
    sharpe = 0.0
    if np.std(daily_rets) > 0:
        sharpe = float(np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252))

    return {
        "total_return": round(total_return, 2),
        "sharpe": round(sharpe, 2),
        "mdd": round(mdd, 1),
        "wins": 0,
        "losses": 0,
        "trades": 0,
        "win_rate": 0.0,
        "name": "Buy & Hold (KODEX 200)",
    }


def _strategy_long_only(closes, daily_rets, signals):
    """B. 기존 단순 매수/현금 전략

    상승 예측 시 매수, 하락 예측 시 현금
    """
    capital = float(INITIAL_CAPITAL)
    position = "cash"
    entry_price = None
    hold_days = 0
    wins, losses, trades = 0, 0, 0
    daily_values = []

    sizing_map = {"low": 1.0, "medium": 0.6, "high": 0.3}

    for i in range(len(closes)):
        sig = signals[i]

        if position == "long" and entry_price:
            hold_days += 1
            unrealized = (closes[i] / entry_price - 1) * 100

            should_sell = (
                unrealized <= STOP_LOSS
                or unrealized >= TAKE_PROFIT
                or hold_days >= MAX_HOLD_DAYS
                or (sig["signal_valid"] and sig["direction"] == "down")
            )

            if should_sell:
                sizing = sizing_map.get(sig["mc_level"], 0.6)
                invested = capital * sizing / (1 + sizing)  # 투입 비율에서 역산
                pnl = invested * (closes[i] / entry_price - 1)
                commission = abs(invested + pnl) * COMMISSION_RATE
                capital += pnl - commission
                trades += 1
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                position = "cash"
                entry_price = None
                hold_days = 0

        elif position == "cash":
            if sig["signal_valid"] and sig["direction"] == "up":
                position = "long"
                entry_price = closes[i]
                hold_days = 0

        daily_values.append(capital)

    metrics = _compute_metrics(daily_values, wins, losses, trades)
    metrics["name"] = "단순 매수/현금"
    return metrics


def _strategy_bidirectional(closes, daily_rets, signals, use_leverage=False):
    """C/D. ETF 양방향 전략

    상승 예측 → 정방향 ETF 매수
    하락 예측 → 인버스 ETF 매수
    """
    capital = float(INITIAL_CAPITAL)
    position = "cash"
    entry_price = None
    hold_days = 0
    multiplier = 1.0
    wins, losses, trades = 0, 0, 0
    daily_values = []

    sizing_map = {"low": 1.0, "medium": 0.6, "high": 0.3}

    for i in range(len(closes)):
        sig = signals[i]

        if position != "cash" and entry_price:
            hold_days += 1
            kospi_change = (closes[i] / entry_price - 1) * 100
            unrealized = kospi_change * multiplier

            # 방향 전환 감지
            direction_change = False
            if sig["signal_valid"]:
                if position == "long" and sig["direction"] == "down":
                    direction_change = True
                elif position == "short" and sig["direction"] == "up":
                    direction_change = True

            should_sell = (
                unrealized <= STOP_LOSS
                or unrealized >= TAKE_PROFIT
                or hold_days >= MAX_HOLD_DAYS
                or direction_change
            )

            if should_sell:
                sizing = sizing_map.get(sig["mc_level"], 0.6)
                invested_ratio = sizing
                invested = capital * invested_ratio / (1 + invested_ratio)
                pnl = invested * (kospi_change / 100) * multiplier
                commission = abs(invested + pnl) * COMMISSION_RATE
                capital += pnl - commission
                trades += 1
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                position = "cash"
                entry_price = None
                hold_days = 0
                multiplier = 1.0

        if position == "cash" and sig["signal_valid"]:
            confidence = sig["confidence"]
            regime = sig["regime"]
            direction = sig["direction"]

            # ETF 선택 (레버리지 유무에 따라)
            if direction == "up":
                multiplier = _get_long_multiplier(
                    confidence, regime, use_leverage
                )
                position = "long"
            else:
                multiplier = _get_short_multiplier(
                    confidence, regime, use_leverage
                )
                position = "short"

            entry_price = closes[i]
            hold_days = 0

        daily_values.append(capital)

    metrics = _compute_metrics(daily_values, wins, losses, trades)
    if use_leverage:
        metrics["name"] = "ETF 양방향 (레버리지)"
    else:
        metrics["name"] = "ETF 양방향 (1배)"
    return metrics


def _get_long_multiplier(confidence, regime, use_leverage):
    """상승 예측 시 multiplier 결정"""
    if use_leverage and confidence >= 90 and regime == "bull":
        return 1.95  # KODEX 레버리지
    return 1.0  # KODEX 200 / TIGER 200


def _get_short_multiplier(confidence, regime, use_leverage):
    """하락 예측 시 multiplier 결정"""
    if use_leverage and confidence >= 90 and regime == "bear":
        return -1.9  # KODEX 200선물인버스2X
    return -1.0  # KODEX 인버스 / TIGER 인버스


def _print_results(results):
    """결과 표 출력"""
    logger.info("")
    logger.info("=" * 80)
    logger.info("  ETF 백테스팅 결과 (2022~2024)")
    logger.info("=" * 80)
    logger.info(
        f"{'전략':<28} {'수익률':>8} {'샤프비율':>8} {'MDD':>8} "
        f"{'승률':>8} {'거래':>6}"
    )
    logger.info("-" * 80)

    for key in ["A_BuyHold", "B_LongOnly", "C_BiDir_1x", "D_BiDir_Lev"]:
        r = results[key]
        name = r.get("name", key)
        wr_str = f"{r['win_rate']:.1f}%" if r["trades"] > 0 else "N/A"
        tr_str = f"{r['wins']}승{r['losses']}패" if r["trades"] > 0 else "-"
        logger.info(
            f"  {name:<26} {r['total_return']:>+7.2f}% {r['sharpe']:>7.2f} "
            f"{r['mdd']:>7.1f}% {wr_str:>7} {tr_str:>8}"
        )

    logger.info("=" * 80)

    # 최고 전략 표시
    best = max(results.values(), key=lambda x: x["total_return"])
    logger.info(f"  최고 수익 전략: {best.get('name', '?')} ({best['total_return']:+.2f}%)")


def format_backtest_for_slack(results):
    """슬랙 메시지용 백테스트 결과 포맷"""
    if not results:
        return "백테스트 결과 없음"

    lines = ["📊 *ETF 백테스팅 결과 (2022~2024)*\n"]

    for key in ["A_BuyHold", "B_LongOnly", "C_BiDir_1x", "D_BiDir_Lev"]:
        r = results[key]
        name = r.get("name", key)
        wr = f"{r['win_rate']:.0f}%" if r["trades"] > 0 else "-"
        lines.append(
            f"  {name}: {r['total_return']:+.2f}% | "
            f"샤프 {r['sharpe']:.2f} | MDD {r['mdd']:.1f}% | "
            f"승률 {wr} ({r['trades']}건)"
        )

    best = max(results.values(), key=lambda x: x["total_return"])
    lines.append(f"\n  🏆 최고: {best.get('name')} ({best['total_return']:+.2f}%)")

    return "\n".join(lines)
