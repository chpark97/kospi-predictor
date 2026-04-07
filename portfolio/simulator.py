"""가상 포트폴리오 시뮬레이터 v3

일일 양방향 매매 전략: 롱/인버스, 오버나이트 없음.
매일 open 매수 → 목표가 또는 장마감 청산.
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

# 일일 매매 설정
COMMISSION_PER_TRADE = COMMISSION_RATE * 2  # 매수 + 매도 = 0.03%
TARGET_PROFIT = 0.5  # 목표 수익률 0.5% (수수료 제외 전)

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
        "position": "cash",  # cash/long/inverse
        "direction": None,   # "up" or "down"
        "entry_price": None,
        "entry_date": None,
        "invested_amount": 0,
        "sizing_ratio": 1.0,
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
                  ohlc=None, confidence=None, confidence_threshold=None):
    """일일 양방향 매매 실행 (신뢰도 필터 포함)

    매일 open 매수 → 목표가 도달 시 청산, 미도달 시 장마감(close) 청산.
    오버나이트 없음: 매일 position이 cash로 리셋.
    신뢰도 필터: confidence < confidence_threshold이면 현금 보유.

    Args:
        date: 거래일 (YYYY-MM-DD)
        signal_valid: 신호 유효 여부 (현재는 항상 매매하므로 참고용)
        predicted_direction: "up" 또는 "down"
        mc_level: MC Dropout 불확실성 ("low"/"medium"/"high")
        ohlc: dict with "open", "high", "low", "close" (None이면 DB 조회)
        confidence: 예측 신뢰도 (0-100, None이면 필터 미적용)
        confidence_threshold: 신뢰도 임계값 (0-100, None이면 필터 미적용)
    """
    pf = _load_portfolio()

    if pf["trades"] and pf["trades"][-1].get("date") == date:
        return pf

    # OHLC 조회
    if ohlc is None:
        ohlc = _get_ohlc_for_date(date)
    if ohlc is None:
        logger.warning(f"  [Portfolio] {date} OHLC 데이터 없음")
        return pf

    # ── 신뢰도 필터: 임계값 미달 시 현금 보유 ──
    if confidence is not None and confidence_threshold is not None:
        if confidence < confidence_threshold:
            logger.info(f"  [Portfolio] 신뢰도 {confidence:.1f}% < 임계값 {confidence_threshold}% → 현금 보유 (skip)")
            trade = {
                "date": date,
                "direction": None,
                "action": "skip",
                "entry_price": None,
                "exit_price": None,
                "exit_reason": f"신뢰도 미달 ({confidence:.1f}% < {confidence_threshold}%)",
                "pnl": 0,
                "return_pct": 0,
            }
            pf["trades"].append(trade)
            pf["trades"] = pf["trades"][-90:]
            _save_portfolio(pf)
            return pf

    open_price = ohlc["open"]
    high_price = ohlc["high"]
    low_price = ohlc["low"]
    close_price = ohlc["close"]

    # 포지션 사이징
    sizing = SIZING_MAP.get(mc_level, 0.6)
    invested = pf["capital"] * sizing
    buy_commission = invested * COMMISSION_RATE

    # 매수가 = 당일 open
    entry_price = open_price
    direction = predicted_direction  # "up" -> 롱, "down" -> 인버스

    # 목표 수익률 (수수료 포함)
    target_pct = (TARGET_PROFIT + COMMISSION_PER_TRADE * 100) / 100

    # 장중 목표가 도달 여부 확인
    if direction == "up":
        # 롱: 코스피 상승 시 수익
        target_price = entry_price * (1 + target_pct)
        target_reached = high_price >= target_price
        if target_reached:
            exit_price = target_price
            exit_reason = f"목표가 도달 ({target_pct*100:.2f}%)"
        else:
            exit_price = close_price
            exit_reason = "장마감"
        # 실제 수익률 (롱)
        raw_return = (exit_price / entry_price - 1)
    else:
        # 인버스: 코스피 하락 시 수익 (수익률 = -(close/open - 1))
        target_price = entry_price * (1 - target_pct)
        target_reached = low_price <= target_price
        if target_reached:
            exit_price = target_price
            exit_reason = f"목표가 도달 ({target_pct*100:.2f}%)"
        else:
            exit_price = close_price
            exit_reason = "장마감"
        # 실제 수익률 (인버스): 코스피 하락 = 수익
        raw_return = -(exit_price / entry_price - 1)

    # 수수료 차감
    net_return = raw_return - COMMISSION_PER_TRADE
    pnl = invested * net_return

    # 자본 업데이트
    pf["capital"] += round(pnl)

    if pnl > 0:
        pf["wins"] += 1
    else:
        pf["losses"] += 1

    # MDD 추적
    total_value = pf["capital"]
    if total_value > pf.get("peak_capital", INITIAL_CAPITAL):
        pf["peak_capital"] = total_value
    dd = (total_value / pf["peak_capital"] - 1) * 100
    if dd < pf.get("max_drawdown", 0):
        pf["max_drawdown"] = round(dd, 2)

    # 포지션은 항상 cash로 리셋 (오버나이트 없음)
    pf["position"] = "cash"
    pf["direction"] = None
    pf["entry_price"] = None
    pf["entry_date"] = None
    pf["invested_amount"] = 0
    pf["sizing_ratio"] = sizing

    trade = {
        "date": date,
        "direction": direction,
        "action": "롱" if direction == "up" else "인버스",
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "pnl": round(pnl),
        "return_pct": round(net_return * 100, 2),
    }
    pf["trades"].append(trade)
    pf["trades"] = pf["trades"][-90:]
    _save_portfolio(pf)

    dir_label = "롱" if direction == "up" else "인버스"
    logger.info(f"  [Portfolio] {dir_label} 매매 @ {entry_price:.1f}→{exit_price:.1f} "
                f"({exit_reason}) P/L: {pnl:+,.0f}원 ({net_return*100:+.2f}%)")

    return pf


def _get_ohlc_for_date(date):
    """특정 날짜의 OHLC 데이터 조회"""
    try:
        conn = sqlite3.connect(DB_PATH)
        r = conn.execute(
            "SELECT open, high, low, close FROM kospi_index WHERE date = ?",
            (date,)
        ).fetchone()
        conn.close()
        if r:
            return {"open": r[0], "high": r[1], "low": r[2], "close": r[3]}
    except Exception as e:
        logger.error(f"OHLC 조회 실패: {e}")
    return None


def settle_trade(actual_return):
    """정답 확인 시 호출 (오버나이트 없으므로 MDD만 업데이트)"""
    pf = _load_portfolio()
    # 일일 청산이므로 포지션이 없음 - MDD만 확인
    total = pf["capital"]
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

    # 오버나이트 없으므로 미실현 손익은 항상 0
    total_value = pf["capital"]
    initial = pf.get("initial_capital", INITIAL_CAPITAL)
    total_return = (total_value / initial - 1) * 100
    wins = pf["wins"]
    losses = pf["losses"]
    total_trades = wins + losses
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0

    # 최근 거래 통계
    recent_trades = pf.get("trades", [])[-30:]
    recent_pnl = sum(t.get("pnl", 0) for t in recent_trades)

    # 포지션 상태 (일일 청산이므로 항상 현금)
    position = "현금 보유 (일일 청산)"

    return {
        "capital": total_value,
        "total_return": round(total_return, 2),
        "position": position,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "mdd": pf.get("max_drawdown", 0),
        "sizing": pf.get("sizing_ratio", 1.0),
        "unrealized_pnl": 0,  # 오버나이트 없음
        "recent_pnl": recent_pnl,
    }


def format_portfolio_summary(next_direction=None):
    """포트폴리오 요약 포맷

    Args:
        next_direction: 다음 거래 방향 ("up" -> 롱, "down" -> 인버스)
    """
    s = get_portfolio_summary()
    sign = "+" if s["total_return"] >= 0 else ""
    sizing_pct = int(s.get("sizing", 1.0) * 100)

    # 다음 거래 방향 표시
    if next_direction == "up":
        next_trade = "오늘 롱 매수"
    elif next_direction == "down":
        next_trade = "오늘 인버스 매수"
    else:
        next_trade = "대기"

    # 최근 30일 손익
    recent_pnl = s.get("recent_pnl", 0)
    recent_sign = "+" if recent_pnl >= 0 else ""

    return (
        f"💰 *가상 포트폴리오* (일일 양방향 매매)\n"
        f"  잔고: {s['capital']:,}원 ({sign}{s['total_return']}%)\n"
        f"  다음 거래: {next_trade}\n"
        f"  최근 30일 손익: {recent_sign}{recent_pnl:,}원\n"
        f"  투입비율: {sizing_pct}% | MDD: {s['mdd']:.1f}%\n"
        f"  승률: {s['wins']}승 {s['losses']}패 ({s['win_rate']}%)"
    )


def backtest_strategy(db_path=None, direction_accuracy=0.52):
    """일일 양방향 매매 전략 백테스팅 (2022~2024)

    Args:
        db_path: DB 경로
        direction_accuracy: 방향 예측 정확도 (기본 52%)
    """
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT date, open, high, low, close FROM kospi_index "
        "WHERE date >= '2022-01-01' AND date <= '2024-12-31' ORDER BY date"
    ).fetchall()
    conn.close()

    if len(rows) < 50:
        logger.warning("백테스트 데이터 부족")
        return

    dates = [r[0] for r in rows]
    opens = np.array([r[1] for r in rows])
    highs = np.array([r[2] for r in rows])
    lows = np.array([r[3] for r in rows])
    closes = np.array([r[4] for r in rows])

    # 실제 방향 (open -> close)
    actual_directions = closes > opens  # True = 상승, False = 하락

    # === 전략 1: 단순 매수 후 보유 (Buy & Hold) ===
    bh_ret = (closes[-1] / closes[0] - 1) * 100

    # === 전략 2: 일일 양방향 매매 시뮬레이션 ===
    np.random.seed(42)  # 재현성
    capital = INITIAL_CAPITAL
    wins = 0
    losses = 0
    peak = capital
    daily_values = [capital]

    target_pct = (TARGET_PROFIT + COMMISSION_PER_TRADE * 100) / 100

    for i in range(len(dates)):
        # 방향 예측 (direction_accuracy 확률로 맞춤)
        is_correct = np.random.random() < direction_accuracy
        actual_up = actual_directions[i]
        pred_up = actual_up if is_correct else not actual_up

        open_p = opens[i]
        high_p = highs[i]
        low_p = lows[i]
        close_p = closes[i]

        invested = capital * 0.6  # 60% 투입

        if pred_up:
            # 롱: 상승 예측
            target_price = open_p * (1 + target_pct)
            target_reached = high_p >= target_price
            exit_price = target_price if target_reached else close_p
            raw_return = (exit_price / open_p - 1)
        else:
            # 인버스: 하락 예측
            target_price = open_p * (1 - target_pct)
            target_reached = low_p <= target_price
            exit_price = target_price if target_reached else close_p
            raw_return = -(exit_price / open_p - 1)

        net_return = raw_return - COMMISSION_PER_TRADE
        pnl = invested * net_return
        capital += pnl

        if pnl > 0:
            wins += 1
        else:
            losses += 1

        daily_values.append(capital)
        if capital > peak:
            peak = capital

    final_return = (capital / INITIAL_CAPITAL - 1) * 100
    daily_vals = np.array(daily_values[1:])
    daily_rets_strat = np.diff(daily_vals) / daily_vals[:-1]
    sharpe = float(daily_rets_strat.mean() / daily_rets_strat.std() * np.sqrt(252)) if daily_rets_strat.std() > 0 else 0

    # MDD 계산
    peak_values = np.maximum.accumulate(daily_vals)
    drawdowns = (daily_vals / peak_values - 1) * 100
    mdd = float(drawdowns.min())

    total_trades = wins + losses
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0

    logger.info("=" * 50)
    logger.info(f"백테스팅 결과 (2022~2024) - DA {direction_accuracy*100:.0f}%")
    logger.info("=" * 50)
    logger.info(f"Buy & Hold: {bh_ret:+.2f}%")
    logger.info(f"양방향 매매: {final_return:+.2f}% | 샤프={sharpe:.2f} | MDD={mdd:.1f}%")
    logger.info(f"  {wins}승 {losses}패 ({win_rate:.0f}%) | 거래 {total_trades}건")

    return {
        "buy_hold": round(bh_ret, 2),
        "strategy_return": round(final_return, 2),
        "sharpe": round(sharpe, 2),
        "mdd": round(mdd, 1),
        "wins": wins,
        "losses": losses,
        "total_trades": total_trades,
    }
