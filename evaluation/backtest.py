"""평가 지표 및 백테스팅"""
import logging

import numpy as np

from config.settings import COMMISSION_RATE

logger = logging.getLogger(__name__)


def evaluate_predictions(y_true, y_pred, confidences, dates):
    """예측 결과 평가 (일일 양방향 매매 전략)

    Args:
        y_true: 실제 등락률 (%)
        y_pred: 예측 등락률 (%)
        confidences: 예측 확신도 (0~1)
        dates: 날짜 배열

    Returns:
        dict: 평가 지표
    """
    n = len(y_true)

    # 1. 방향 정확도 (Directional Accuracy)
    true_direction = y_true > 0
    pred_direction = y_pred > 0
    direction_accuracy = np.mean(true_direction == pred_direction) * 100

    # 2. 백테스팅 누적 수익률 (양방향 매매, 수수료 반영)
    # 전략: 상승 예측 → 롱 (actual_return - 수수료)
    #       하락 예측 → 인버스 (-actual_return - 수수료)
    # 매일 거래하므로 관망 없음
    daily_returns = []
    commission = COMMISSION_RATE * 2  # 매수+매도 수수료

    for i in range(n):
        actual_ret = y_true[i] / 100  # % -> 비율
        if y_pred[i] > 0:  # 상승 예측 -> 롱
            ret = actual_ret - commission
        else:  # 하락 예측 -> 인버스
            ret = -actual_ret - commission
        daily_returns.append(ret)

    daily_returns = np.array(daily_returns)
    cumulative_return = (np.prod(1 + daily_returns) - 1) * 100

    # 3. 샤프 비율 (연간화)
    if daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # 4. 추가 지표
    avg_confidence = np.mean(confidences) * 100
    win_rate = np.mean(daily_returns > 0) * 100  # 매일 거래하므로 전체 대상
    trade_count = n  # 매일 거래

    logger.info(f"  방향 정확도: {direction_accuracy:.1f}%")
    logger.info(f"  누적 수익률: {cumulative_return:.2f}% (거래 {trade_count}건)")
    logger.info(f"  샤프 비율: {sharpe_ratio:.2f}")
    logger.info(f"  승률: {win_rate:.1f}%")
    logger.info(f"  평균 확신도: {avg_confidence:.1f}%")

    return {
        "direction_accuracy": direction_accuracy,
        "cumulative_return": cumulative_return,
        "sharpe_ratio": sharpe_ratio,
        "win_rate": win_rate,
        "avg_confidence": avg_confidence,
        "trade_count": int(trade_count),
        "n_samples": n,
    }
