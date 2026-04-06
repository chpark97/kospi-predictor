"""가상 포트폴리오 시뮬레이터

초기 자금 1억원으로 시작, 매일 신호에 따라 가상 매매.
"""
import json
import logging
from pathlib import Path

from config.settings import COMMISSION_RATE

logger = logging.getLogger(__name__)

PORTFOLIO_PATH = Path(__file__).parent / "portfolio.json"
INITIAL_CAPITAL = 100_000_000  # 1억원


def _load_portfolio():
    if PORTFOLIO_PATH.exists():
        with open(PORTFOLIO_PATH) as f:
            return json.load(f)
    return {
        "capital": INITIAL_CAPITAL,
        "initial_capital": INITIAL_CAPITAL,
        "position": "cash",  # "cash" or "long"
        "trades": [],
        "wins": 0,
        "losses": 0,
    }


def _save_portfolio(pf):
    with open(PORTFOLIO_PATH, "w") as f:
        json.dump(pf, f, indent=2, ensure_ascii=False)


def execute_trade(date, signal_valid, predicted_direction, actual_return=None):
    """매일 신호에 따라 가상 매매

    Args:
        date: 거래일
        signal_valid: 신호 유효 여부
        predicted_direction: "up" or "down"
        actual_return: 실제 등락률 (%) — check 시점에서 사후 기입

    Returns:
        포트폴리오 상태 dict
    """
    pf = _load_portfolio()

    # 이미 같은 날 거래 기록이 있으면 건너뛰기
    if pf["trades"] and pf["trades"][-1].get("date") == date:
        return pf

    action = "hold"

    if signal_valid and predicted_direction == "up":
        if pf["position"] == "cash":
            action = "buy"
            pf["position"] = "long"
    else:
        if pf["position"] == "long":
            action = "sell"
            pf["position"] = "cash"

    trade = {
        "date": date,
        "action": action,
        "capital_before": pf["capital"],
        "actual_return": actual_return,
    }

    pf["trades"].append(trade)
    # 최근 90거래일만 보관
    pf["trades"] = pf["trades"][-90:]
    _save_portfolio(pf)

    return pf


def settle_trade(actual_return):
    """정답 확인 시 마지막 거래 정산

    포지션이 long이었으면 actual_return 만큼 수익/손실 반영.
    """
    pf = _load_portfolio()

    if not pf["trades"]:
        return pf

    last = pf["trades"][-1]
    if last.get("actual_return") is not None:
        return pf  # 이미 정산됨

    last["actual_return"] = round(actual_return, 4)

    # long 포지션이었으면 수익 반영
    if last["action"] == "buy" or (last["action"] == "hold" and pf["position"] == "long"):
        pnl = pf["capital"] * (actual_return / 100) - pf["capital"] * COMMISSION_RATE * 2
        pf["capital"] = round(pf["capital"] + pnl)
        if pnl > 0:
            pf["wins"] += 1
        else:
            pf["losses"] += 1
        last["pnl"] = round(pnl)

    _save_portfolio(pf)
    return pf


def get_portfolio_summary():
    """포트폴리오 현황 요약"""
    pf = _load_portfolio()
    capital = pf["capital"]
    initial = pf["initial_capital"]
    total_return = (capital / initial - 1) * 100
    wins = pf["wins"]
    losses = pf["losses"]
    total_trades = wins + losses
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    position = "매수 중" if pf["position"] == "long" else "현금 보유"

    return {
        "capital": capital,
        "total_return": round(total_return, 2),
        "position": position,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
    }


def format_portfolio_summary():
    """슬랙용 포트폴리오 문자열"""
    s = get_portfolio_summary()
    sign = "+" if s["total_return"] >= 0 else ""
    return (
        f"💰 *가상 포트폴리오*\n"
        f"  잔고: {s['capital']:,}원 ({sign}{s['total_return']}%)\n"
        f"  포지션: {s['position']}\n"
        f"  승률: {s['wins']}승 {s['losses']}패 ({s['win_rate']}%)"
    )
