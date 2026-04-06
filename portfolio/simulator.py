"""가상 포트폴리오 시뮬레이터 v3 — ETF 양방향 전략

ETF별 수익률 시뮬레이션:
  KODEX 200:             코스피 수익률 × 1.0
  TIGER 200:             코스피 수익률 × 1.0
  KODEX 레버리지:        코스피 수익률 × 1.95
  KODEX 인버스:          코스피 수익률 × -1.0
  TIGER 인버스:          코스피 수익률 × -1.0
  KODEX 200선물인버스2X: 코스피 수익률 × -1.9
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
        "etf_name": None,
        "etf_multiplier": None,
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
        # ETF 월간 통계
        "monthly_etf_trades": {},  # {"inverse": 0, "leverage": 0, "normal": 0}
        "monthly_reset_date": None,
    }


def _save_portfolio(pf):
    with open(PORTFOLIO_PATH, "w") as f:
        json.dump(pf, f, indent=2, ensure_ascii=False)


def _classify_etf_type(etf_name):
    """ETF 종류 분류"""
    if etf_name is None:
        return "none"
    if "인버스" in etf_name:
        return "inverse"
    if "레버리지" in etf_name:
        return "leverage"
    return "normal"


def _get_total_value(pf, current_price=None):
    """총자산 = capital + 미실현 평가금 (정확한 계산)"""
    total = pf["capital"]
    if pf["position"] != "cash" and pf["invested_amount"] > 0 and pf.get("entry_price"):
        if current_price and pf["entry_price"]:
            multiplier = pf.get("etf_multiplier", 1.0)
            kospi_change = (current_price / pf["entry_price"] - 1)
            total += pf["invested_amount"] * (1 + kospi_change * multiplier)
        else:
            total += pf["invested_amount"]
    return total


def execute_etf_trade(date, signal_valid, etf_signal, current_price=None):
    """ETF 전략 기반 매매 실행

    Args:
        date: 거래일
        signal_valid: 신호 유효 여부
        etf_signal: etf_strategy.select_etf() 결과 dict
        current_price: 현재 코스피 종가
    """
    pf = _load_portfolio()

    if pf["trades"] and pf["trades"][-1].get("date") == date:
        return pf

    if current_price is None:
        current_price = _get_latest_kospi_close()

    # 월간 통계 리셋
    month_key = date[:7]  # "2026-04"
    if pf.get("monthly_reset_date") != month_key:
        pf["monthly_etf_trades"] = {"inverse": 0, "leverage": 0, "normal": 0}
        pf["monthly_reset_date"] = month_key

    action = "hold"
    exit_reason = None

    # === 포지션 보유 중 ===
    if pf["position"] != "cash" and pf["entry_price"]:
        pf["hold_days"] += 1
        multiplier = pf.get("etf_multiplier", 1.0)
        kospi_change = (current_price / pf["entry_price"] - 1)
        unrealized = kospi_change * multiplier * 100

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
        # 반대 방향 신호 발생
        elif signal_valid and etf_signal.get("etf_name") != "현금":
            current_dir = pf.get("position", "cash")
            new_dir = etf_signal.get("direction", "none")
            if current_dir != new_dir and new_dir != "none":
                action = "sell"
                exit_reason = "방향 전환"

        # 같은 방향 신호가 다시 올 때는 중복 매수 방지 (hold 유지)

    # === 현금 보유 중 ===
    elif pf["position"] == "cash":
        if signal_valid and etf_signal.get("etf_name") != "현금":
            action = "buy"

    # 매매 실행
    if action == "buy":
        # 포지션 사이징: 총자산 기준 (capital이 0이어도 정확)
        total_value = _get_total_value(pf, current_price)
        sizing = etf_signal.get("sizing_ratio", 0.6)
        # capital 기준이 아닌 총자산 기준
        invest_target = total_value * sizing
        # capital에서 실제 차감 가능한 금액으로 제한
        invest_actual = min(invest_target, pf["capital"])
        if invest_actual <= 0:
            # 자금 부족 → 매수 불가
            action = "hold"
        else:
            commission = invest_actual * COMMISSION_RATE
            pf["invested_amount"] = round(invest_actual - commission)
            pf["capital"] -= round(invest_actual)
            pf["entry_price"] = current_price
            pf["entry_date"] = date
            pf["hold_days"] = 0
            pf["sizing_ratio"] = sizing
            pf["position"] = etf_signal.get("direction", "long")
            pf["etf_name"] = etf_signal.get("etf_name")
            pf["etf_multiplier"] = etf_signal.get("multiplier", 1.0)

            # 월간 거래 카운트
            etf_type = _classify_etf_type(pf["etf_name"])
            pf["monthly_etf_trades"][etf_type] = pf["monthly_etf_trades"].get(etf_type, 0) + 1

    if action == "sell" and pf["position"] != "cash":
        if pf["entry_price"] and current_price:
            multiplier = pf.get("etf_multiplier", 1.0)
            kospi_change = (current_price / pf["entry_price"] - 1)
            pnl_pct = kospi_change * multiplier
            pnl = pf["invested_amount"] * pnl_pct
            commission = abs(pf["invested_amount"] + pnl) * COMMISSION_RATE
            returned = round(pf["invested_amount"] + pnl - commission)
            # capital이 음수가 되지 않도록 보호
            pf["capital"] = max(pf["capital"] + returned, 0)

            if pnl > 0:
                pf["wins"] += 1
            else:
                pf["losses"] += 1

        pf["position"] = "cash"
        pf["etf_name"] = None
        pf["etf_multiplier"] = None
        pf["entry_price"] = None
        pf["entry_date"] = None
        pf["invested_amount"] = 0
        pf["hold_days"] = 0

        # 반대 방향 매수 즉시 진입
        if exit_reason == "방향 전환" and signal_valid and etf_signal.get("etf_name") != "현금":
            total_value = _get_total_value(pf, current_price)
            sizing = etf_signal.get("sizing_ratio", 0.6)
            invest_target = total_value * sizing
            invest_actual = min(invest_target, pf["capital"])
            if invest_actual > 0:
                commission_buy = invest_actual * COMMISSION_RATE
                pf["invested_amount"] = round(invest_actual - commission_buy)
                pf["capital"] -= round(invest_actual)
                pf["entry_price"] = current_price
                pf["entry_date"] = date
                pf["hold_days"] = 0
                pf["sizing_ratio"] = sizing
                pf["position"] = etf_signal.get("direction", "long")
                pf["etf_name"] = etf_signal.get("etf_name")
                pf["etf_multiplier"] = etf_signal.get("multiplier", 1.0)

                etf_type = _classify_etf_type(pf["etf_name"])
                pf["monthly_etf_trades"][etf_type] = pf["monthly_etf_trades"].get(etf_type, 0) + 1

    # MDD 추적
    total_value = _get_total_value(pf, current_price)
    if total_value > pf.get("peak_capital", INITIAL_CAPITAL):
        pf["peak_capital"] = total_value
    dd = (total_value / pf["peak_capital"] - 1) * 100
    if dd < pf.get("max_drawdown", 0):
        pf["max_drawdown"] = round(dd, 2)

    trade = {
        "date": date,
        "action": action,
        "price": current_price,
        "etf_name": pf.get("etf_name"),
        "exit_reason": exit_reason,
    }
    pf["trades"].append(trade)
    pf["trades"] = pf["trades"][-90:]
    _save_portfolio(pf)

    if action != "hold":
        etf_str = pf.get("etf_name") or ""
        logger.info(
            f"  [Portfolio] {action.upper()} {etf_str} @ {current_price:.1f}"
            + (f" ({exit_reason})" if exit_reason else "")
            + f" sizing={pf.get('sizing_ratio', 1):.0%}"
        )

    return pf


# 하위 호환: 기존 execute_trade도 유지
def execute_trade(date, signal_valid, predicted_direction, mc_level="medium",
                  current_price=None):
    """기존 단순 매수/현금 전략 (하위 호환)"""
    if predicted_direction == "up" and signal_valid:
        etf_signal = {
            "etf_name": "KODEX 200",
            "direction": "long",
            "multiplier": 1.0,
            "sizing_ratio": SIZING_MAP.get(mc_level, 0.6),
        }
    else:
        etf_signal = {"etf_name": "현금", "direction": "none", "sizing_ratio": 0.0}

    return execute_etf_trade(date, signal_valid, etf_signal, current_price)


def settle_trade(actual_return):
    """정답 확인 시 미실현 손익 기록 및 MDD 업데이트"""
    pf = _load_portfolio()
    current_price = _get_latest_kospi_close()

    if pf["position"] != "cash" and pf["invested_amount"] > 0 and pf.get("entry_price"):
        multiplier = pf.get("etf_multiplier", 1.0)
        kospi_change = (current_price / pf["entry_price"] - 1)
        unrealized_pnl = round(pf["invested_amount"] * kospi_change * multiplier)
        unrealized_pct = round(kospi_change * multiplier * 100, 2)
        pf["unrealized_pnl"] = unrealized_pnl
        pf["unrealized_pct"] = unrealized_pct

        total = pf["capital"] + pf["invested_amount"] + unrealized_pnl
        if total > pf.get("peak_capital", INITIAL_CAPITAL):
            pf["peak_capital"] = round(total)
        dd = (total / pf["peak_capital"] - 1) * 100
        if dd < pf.get("max_drawdown", 0):
            pf["max_drawdown"] = round(dd, 2)
    else:
        pf["unrealized_pnl"] = 0
        pf["unrealized_pct"] = 0.0

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
    # 총자산 = capital + 미실현 평가금 (정확한 계산)
    current_price = _get_latest_kospi_close()
    total_value = _get_total_value(pf, current_price)
    initial = pf["initial_capital"]
    total_return = (total_value / initial - 1) * 100
    wins = pf["wins"]
    losses = pf["losses"]
    total_trades = wins + losses
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0

    # 샤프 비율 추정
    trade_rets = []
    for t in pf["trades"]:
        if t["action"] == "sell":
            trade_rets.append(1.0)

    position = "현금 보유"
    entry_info = ""
    etf_name = pf.get("etf_name")

    if pf["position"] != "cash" and pf.get("entry_price"):
        multiplier = pf.get("etf_multiplier", 1.0)
        position = f"{etf_name} 매수 중" if etf_name else "매수 중"
        sl = pf["entry_price"] * (1 + STOP_LOSS / 100 / abs(multiplier)) if multiplier else pf["entry_price"]
        tp = pf["entry_price"] * (1 + TAKE_PROFIT / 100 / abs(multiplier)) if multiplier else pf["entry_price"]
        entry_info = f" (진입 {pf['entry_price']:.0f} / 손절 {sl:.0f} / 익절 {tp:.0f})"
        position += entry_info

    # 월간 ETF 거래 통계
    monthly = pf.get("monthly_etf_trades", {})
    inv_cnt = monthly.get("inverse", 0)
    lev_cnt = monthly.get("leverage", 0)
    norm_cnt = monthly.get("normal", 0)

    unrealized_pnl = pf.get("unrealized_pnl", 0)
    unrealized_pct = pf.get("unrealized_pct", 0.0)

    return {
        "capital": total_value,
        "total_return": round(total_return, 2),
        "position": position,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "mdd": pf.get("max_drawdown", 0),
        "sizing": pf.get("sizing_ratio", 1.0),
        "etf_name": etf_name,
        "monthly_inverse": inv_cnt,
        "monthly_leverage": lev_cnt,
        "monthly_normal": norm_cnt,
        "unrealized_pnl": unrealized_pnl,
        "unrealized_pct": unrealized_pct,
    }


def format_portfolio_summary():
    s = get_portfolio_summary()
    sign = "+" if s["total_return"] >= 0 else ""
    sizing_pct = int(s.get("sizing", 1.0) * 100)

    # 월간 ETF 거래 내역
    inv = s.get("monthly_inverse", 0)
    lev = s.get("monthly_leverage", 0)
    norm = s.get("monthly_normal", 0)
    etf_trades_str = ""
    parts = []
    if inv > 0:
        parts.append(f"인버스 {inv}회")
    if lev > 0:
        parts.append(f"레버리지 {lev}회")
    if norm > 0:
        parts.append(f"일반 {norm}회")
    if parts:
        etf_trades_str = f"\n  이번 달 ETF 거래: {' / '.join(parts)}"

    # 미실현 손익 라인
    upnl = s.get("unrealized_pnl", 0)
    upnl_pct = s.get("unrealized_pct", 0.0)
    upnl_str = ""
    if upnl != 0:
        upnl_sign = "+" if upnl > 0 else ""
        upnl_str = f"\n  미실현 손익: {upnl_sign}{upnl:,}원 ({upnl_sign}{upnl_pct:.1f}%)"

    return (
        f"💰 *가상 포트폴리오 (ETF 전략)*\n"
        f"  잔고: {s['capital']:,.0f}원 ({sign}{s['total_return']}%)\n"
        f"  현재 포지션: {s['position']}{upnl_str}\n"
        f"  투입비율: {sizing_pct}% | MDD: {s['mdd']:.1f}%\n"
        f"  승률: {s['wins']}승 {s['losses']}패 ({s['win_rate']}%)"
        f"{etf_trades_str}"
    )


def reset_portfolio():
    """포트폴리오를 초기 상태로 리셋 (백업 후)"""
    from datetime import datetime

    pf = _load_portfolio()
    prev_capital = _get_total_value(pf, _get_latest_kospi_close())

    # 현재 상태 백업
    backup_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = PORTFOLIO_PATH.parent / f"portfolio_backup_{backup_date}.json"
    with open(backup_path, "w") as f:
        json.dump(pf, f, indent=2, ensure_ascii=False)
    logger.info(f"[Portfolio] 백업 저장: {backup_path}")

    # 초기화
    new_pf = _new_portfolio()
    _save_portfolio(new_pf)

    logger.info(
        f"[Portfolio] 포트폴리오 초기화: {prev_capital:,.0f}원 → {INITIAL_CAPITAL:,.0f}원"
    )
    return {"prev_capital": prev_capital, "new_capital": INITIAL_CAPITAL, "backup": str(backup_path)}


def backtest_strategy(db_path=None):
    """기존 단순 백테스팅 (하위 호환)"""
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT date, close FROM kospi_index WHERE date >= '2022-01-01' AND date <= '2024-12-31' ORDER BY date"
    ).fetchall()
    conn.close()

    if len(rows) < 50:
        logger.warning("백테스트 데이터 부족")
        return

    closes = np.array([r[1] for r in rows])
    rets = np.diff(closes) / closes[:-1] * 100
    bh_ret = (closes[-1] / closes[0] - 1) * 100

    # 손절/익절 시뮬레이션
    capital = INITIAL_CAPITAL
    position = "cash"
    entry_price = None
    hold_days = 0
    wins, losses = 0, 0
    peak = capital
    daily_values = [capital]

    for i in range(len(closes)):
        if position == "long":
            hold_days += 1
            unrealized = (closes[i] / entry_price - 1) * 100
            if unrealized <= STOP_LOSS or unrealized >= TAKE_PROFIT or hold_days >= MAX_HOLD_DAYS:
                pnl = capital * 0.8 * (closes[i] / entry_price - 1)
                capital += pnl - abs(capital * 0.8) * COMMISSION_RATE * 2
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                position = "cash"
                entry_price = None
                hold_days = 0
        elif position == "cash" and i < len(rets):
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
    if wins + losses > 0:
        logger.info(f"  {wins}승 {losses}패 ({wins/(wins+losses)*100:.0f}%)")

    return {
        "buy_hold": round(bh_ret, 2),
        "strategy_return": round(final_return, 2),
        "sharpe": round(sharpe, 2),
        "mdd": round(mdd, 1),
        "wins": wins,
        "losses": losses,
    }
