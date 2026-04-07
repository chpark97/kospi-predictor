"""가상 포트폴리오 시뮬레이터 v2

손절/익절/최대보유/포지션사이징 전략 포함.
"""
import json
import logging
import sqlite3
from pathlib import Path

import numpy as np

from config.settings import COMMISSION_RATE, DB_PATH

logger = logging.getLogger(__name__)

PORTFOLIO_PATH = Path(__file__).parent / "portfolio.json"
INITIAL_CAPITAL = 100_000_000

# 손절/익절/최대보유
STOP_LOSS = -2.0      # -2% 손절
TAKE_PROFIT = 3.0     # +3% 익절
MAX_HOLD_DAYS = 5     # 최대 보유 기간

# 불확실성 기반 포지션 사이징
SIZING_MAP = {
    "low": 1.0,     # 🟢 100%
    "medium": 0.6,  # 🟡 60%
    "high": 0.3,    # 🔴 30%
}


def _load_portfolio():
    if PORTFOLIO_PATH.exists():
        with open(PORTFOLIO_PATH) as f:
            return json.load(f)
    return _new_portfolio()


def _new_portfolio():
    return {
        "capital": INITIAL_CAPITAL,
        "initial_capital": INITIAL_CAPITAL,
        "position": "cash",
        "entry_price": None,
        "entry_date": None,
        "invested_amount": 0,
        "sizing_ratio": 1.0,
        "hold_days": 0,
        "trades": [],
        "wins": 0,
        "losses": 0,
        "peak_capital": INITIAL_CAPITAL,
        "max_drawdown": 0.0,
    }


def _save_portfolio(pf):
    with open(PORTFOLIO_PATH, "w") as f:
        json.dump(pf, f, indent=2, ensure_ascii=False)


def execute_trade(date, signal_valid, predicted_direction, mc_level="medium",
                  current_price=None):
    """매일 신호 + 손절/익절 로직

    Args:
        mc_level: MC Dropout 불확실성 ("low"/"medium"/"high")
        current_price: 현재 코스피 종가
    """
    pf = _load_portfolio()

    if pf["trades"] and pf["trades"][-1].get("date") == date:
        return pf

    # 현재가 조회
    if current_price is None:
        current_price = _get_latest_kospi_close()

    action = "hold"
    exit_reason = None

    # === 포지션 보유 중 ===
    if pf["position"] == "long" and pf["entry_price"]:
        pf["hold_days"] += 1
        unrealized = (current_price / pf["entry_price"] - 1) * 100

        # 손절
        if unrealized <= STOP_LOSS:
            action = "sell"
            exit_reason = f"손절 ({unrealized:+.1f}%)"
        # 익절
        elif unrealized >= TAKE_PROFIT:
            action = "sell"
            exit_reason = f"익절 ({unrealized:+.1f}%)"
        # 최대 보유 기간 초과
        elif pf["hold_days"] >= MAX_HOLD_DAYS:
            action = "sell"
            exit_reason = f"보유기한 ({pf['hold_days']}일)"
        # 하락 신호
        elif signal_valid and predicted_direction == "down":
            action = "sell"
            exit_reason = "하락 신호"

    # === 현금 보유 중 ===
    elif pf["position"] == "cash":
        if signal_valid and predicted_direction == "up":
            action = "buy"

    # 매매 실행
    if action == "buy":
        sizing = SIZING_MAP.get(mc_level, 0.6)
        invested = pf["capital"] * sizing
        commission = invested * COMMISSION_RATE
        pf["invested_amount"] = round(invested - commission)
        pf["capital"] -= round(invested)
        pf["entry_price"] = current_price
        pf["entry_date"] = date
        pf["hold_days"] = 0
        pf["sizing_ratio"] = sizing
        pf["position"] = "long"

    elif action == "sell" and pf["position"] == "long":
        if pf["entry_price"] and current_price:
            pnl_pct = (current_price / pf["entry_price"] - 1)
            pnl = pf["invested_amount"] * pnl_pct
            commission = abs(pf["invested_amount"] + pnl) * COMMISSION_RATE
            pf["capital"] += round(pf["invested_amount"] + pnl - commission)

            if pnl > 0:
                pf["wins"] += 1
            else:
                pf["losses"] += 1

        pf["position"] = "cash"
        pf["entry_price"] = None
        pf["entry_date"] = None
        pf["invested_amount"] = 0
        pf["hold_days"] = 0

    # MDD 추적
    total_value = pf["capital"] + (pf["invested_amount"] if pf["position"] == "long" else 0)
    if total_value > pf.get("peak_capital", INITIAL_CAPITAL):
        pf["peak_capital"] = total_value
    dd = (total_value / pf["peak_capital"] - 1) * 100
    if dd < pf.get("max_drawdown", 0):
        pf["max_drawdown"] = round(dd, 2)

    trade = {
        "date": date,
        "action": action,
        "price": current_price,
        "exit_reason": exit_reason,
    }
    pf["trades"].append(trade)
    pf["trades"] = pf["trades"][-90:]
    _save_portfolio(pf)

    if action != "hold":
        logger.info(f"  [Portfolio] {action.upper()} @ {current_price:.1f}"
                     + (f" ({exit_reason})" if exit_reason else "")
                     + f" sizing={pf.get('sizing_ratio', 1):.0%}")

    return pf


def settle_trade(actual_return):
    """정답 확인 시 미반영 수익 갱신 (hold 중인 경우)"""
    pf = _load_portfolio()
    # 현재 포지션이 long이면 MDD 업데이트만 수행
    if pf["position"] == "long" and pf["invested_amount"] > 0:
        pnl = pf["invested_amount"] * (actual_return / 100)
        total = pf["capital"] + pf["invested_amount"] + pnl
        if total > pf.get("peak_capital", INITIAL_CAPITAL):
            pf["peak_capital"] = round(total)
        dd = (total / pf["peak_capital"] - 1) * 100
        if dd < pf.get("max_drawdown", 0):
            pf["max_drawdown"] = round(dd, 2)
        _save_portfolio(pf)
    return pf


def _get_latest_kospi_close():
    try:
        conn = sqlite3.connect(DB_PATH)
        r = conn.execute("SELECT close FROM kospi_index ORDER BY date DESC LIMIT 1").fetchone()
        conn.close()
        return r[0] if r else 5000.0
    except Exception:
        return 5000.0


def get_portfolio_summary():
    pf = _load_portfolio()

    # 미실현 손익 계산
    unrealized_pnl = 0
    if pf["position"] == "long" and pf["entry_price"] and pf["invested_amount"] > 0:
        current_price = _get_latest_kospi_close()
        unrealized_pnl = pf["invested_amount"] * (current_price / pf["entry_price"] - 1)
        market_value = pf["invested_amount"] + unrealized_pnl
    else:
        current_price = None
        market_value = 0

    total_value = pf["capital"] + market_value
    initial = pf["initial_capital"]
    total_return = (total_value / initial - 1) * 100
    wins = pf["wins"]
    losses = pf["losses"]
    total_trades = wins + losses
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0

    # 샤프 비율 (거래 기반)
    trade_rets = []
    for t in pf["trades"]:
        if t["action"] == "sell" and t.get("price") and t.get("exit_reason"):
            trade_rets.append(1.0)  # 대략적 추정

    position = "현금 보유"
    entry_info = ""
    if pf["position"] == "long" and pf["entry_price"]:
        position = "매수 중"
        sl = pf["entry_price"] * (1 + STOP_LOSS / 100)
        tp = pf["entry_price"] * (1 + TAKE_PROFIT / 100)
        entry_info = f" (진입 {pf['entry_price']:.1f} / 손절 {sl:.1f} / 익절 {tp:.1f})"
        position += entry_info

    return {
        "capital": total_value,
        "total_return": round(total_return, 2),
        "position": position,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "mdd": pf.get("max_drawdown", 0),
        "sizing": pf.get("sizing_ratio", 1.0),
        "unrealized_pnl": round(unrealized_pnl),
    }


def format_portfolio_summary():
    s = get_portfolio_summary()
    sign = "+" if s["total_return"] >= 0 else ""
    sizing_pct = int(s.get("sizing", 1.0) * 100)
    unrealized_pnl = s.get("unrealized_pnl", 0)
    pnl_sign = "+" if unrealized_pnl >= 0 else ""
    pnl_line = f"  미실현 손익: {pnl_sign}{unrealized_pnl:,}원\n" if unrealized_pnl != 0 else ""
    return (
        f"💰 *가상 포트폴리오*\n"
        f"  잔고: {s['capital']:,}원 ({sign}{s['total_return']}%)\n"
        f"  포지션: {s['position']}\n"
        f"{pnl_line}"
        f"  투입비율: {sizing_pct}% | MDD: {s['mdd']:.1f}%\n"
        f"  승률: {s['wins']}승 {s['losses']}패 ({s['win_rate']}%)"
    )


def backtest_strategy(db_path=None):
    """손절/익절 전략 백테스팅 (2022~2024)"""
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT date, close FROM kospi_index WHERE date >= '2022-01-01' AND date <= '2024-12-31' ORDER BY date"
    ).fetchall()
    conn.close()

    if len(rows) < 50:
        logger.warning("백테스트 데이터 부족")
        return

    dates = [r[0] for r in rows]
    closes = np.array([r[1] for r in rows])
    rets = np.diff(closes) / closes[:-1] * 100

    # === 전략 1: 단순 매수 후 보유 (Buy & Hold) ===
    bh_ret = (closes[-1] / closes[0] - 1) * 100

    # === 전략 2: 매일 매수 (단순) ===
    simple_rets = rets / 100  # 상승일만 매수 가정 → 전체 수익
    simple_cum = (np.prod(1 + simple_rets * 0.5) - 1) * 100  # 50% 노출

    # === 전략 3: 손절/익절 시뮬레이션 ===
    capital = INITIAL_CAPITAL
    position = "cash"
    entry_price = None
    hold_days = 0
    wins = 0
    losses = 0
    peak = capital

    daily_values = [capital]

    for i in range(len(closes)):
        if position == "long":
            hold_days += 1
            unrealized = (closes[i] / entry_price - 1) * 100

            if unrealized <= STOP_LOSS or unrealized >= TAKE_PROFIT or hold_days >= MAX_HOLD_DAYS:
                # 청산
                pnl = capital * 0.8 * (closes[i] / entry_price - 1)  # 80% 투입
                capital += pnl - abs(capital * 0.8) * COMMISSION_RATE * 2
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                position = "cash"
                entry_price = None
                hold_days = 0

        elif position == "cash" and i < len(rets):
            # 간단한 모멘텀 신호: 전일 양봉이면 매수
            if i > 0 and rets[i - 1] > 0:
                position = "long"
                entry_price = closes[i]
                hold_days = 0

        total = capital + (capital * 0.8 * (closes[i] / entry_price - 1) if position == "long" and entry_price else 0)
        daily_values.append(total)
        if total > peak:
            peak = total

    final_return = (capital / INITIAL_CAPITAL - 1) * 100
    daily_vals = np.array(daily_values[1:])
    daily_rets_strat = np.diff(daily_vals) / daily_vals[:-1]
    sharpe = float(daily_rets_strat.mean() / daily_rets_strat.std() * np.sqrt(252)) if daily_rets_strat.std() > 0 else 0
    mdd = float((np.minimum.accumulate(daily_vals[::-1])[::-1] / np.maximum.accumulate(daily_vals) - 1).min() * 100)

    logger.info("=" * 50)
    logger.info("백테스팅 결과 (2022~2024)")
    logger.info("=" * 50)
    logger.info(f"Buy & Hold: {bh_ret:+.2f}%")
    logger.info(f"손절/익절 전략: {final_return:+.2f}% | 샤프={sharpe:.2f} | MDD={mdd:.1f}%")
    logger.info(f"  {wins}승 {losses}패 ({wins/(wins+losses)*100:.0f}%)" if wins + losses > 0 else "")

    return {
        "buy_hold": round(bh_ret, 2),
        "strategy_return": round(final_return, 2),
        "sharpe": round(sharpe, 2),
        "mdd": round(mdd, 1),
        "wins": wins,
        "losses": losses,
    }
