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


def evaluate_with_thresholds(y_true, y_pred, confidences, dates,
                              thresholds=None):
    """신뢰도 임계값별 성과 비교

    Args:
        y_true: 실제 등락률 (%)
        y_pred: 예측 등락률 (%)
        confidences: 예측 확신도 (0~1)
        dates: 날짜 배열
        thresholds: 임계값 리스트 (기본: [40, 50, 55, 60, 65, 70])

    Returns:
        dict: {threshold: metrics} 형태의 임계값별 결과
    """
    if thresholds is None:
        thresholds = [40, 50, 55, 60, 65, 70]

    n = len(y_true)
    commission = COMMISSION_RATE * 2  # 매수+매도 수수료

    # confidences를 0-100 스케일로 변환 (0-1 범위인 경우)
    conf_pct = confidences * 100 if np.max(confidences) <= 1 else confidences

    results = {}

    logger.info(f"\n{'='*70}")
    logger.info(f"신뢰도 임계값별 성과 비교")
    logger.info(f"{'='*70}")
    logger.info(f"{'임계값':<10} {'거래수':>8} {'방향정확도':>12} {'누적수익률':>12} {'샤프비율':>10} {'승률':>8}")
    logger.info("-" * 60)

    for threshold in thresholds:
        # 임계값 이상인 날만 거래
        trade_mask = conf_pct >= threshold
        trade_count = np.sum(trade_mask)

        if trade_count == 0:
            results[threshold] = {
                "direction_accuracy": 0,
                "cumulative_return": 0,
                "sharpe_ratio": 0,
                "win_rate": 0,
                "trade_count": 0,
            }
            logger.info(f"{threshold}%       {'거래 없음':>8}")
            continue

        # 거래한 날만 필터링
        y_true_filtered = y_true[trade_mask]
        y_pred_filtered = y_pred[trade_mask]

        # 방향 정확도
        true_dir = y_true_filtered > 0
        pred_dir = y_pred_filtered > 0
        direction_accuracy = np.mean(true_dir == pred_dir) * 100

        # 일별 수익률 계산
        daily_returns = []
        for i in range(len(y_true_filtered)):
            actual_ret = y_true_filtered[i] / 100
            if y_pred_filtered[i] > 0:  # 상승 예측 -> 롱
                ret = actual_ret - commission
            else:  # 하락 예측 -> 인버스
                ret = -actual_ret - commission
            daily_returns.append(ret)

        daily_returns = np.array(daily_returns)
        cumulative_return = (np.prod(1 + daily_returns) - 1) * 100

        # 샤프 비율 (연간화)
        if daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # 승률
        win_rate = np.mean(daily_returns > 0) * 100

        results[threshold] = {
            "direction_accuracy": direction_accuracy,
            "cumulative_return": cumulative_return,
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate,
            "trade_count": int(trade_count),
        }

        sign = '+' if cumulative_return >= 0 else ''
        logger.info(
            f"{threshold}%       {trade_count:>8} {direction_accuracy:>11.1f}% "
            f"{sign}{cumulative_return:>10.2f}% {sharpe_ratio:>10.2f} {win_rate:>7.1f}%"
        )

    # 최적 임계값 찾기 (수익률 기준)
    best_threshold = max(results.keys(), key=lambda t: results[t].get("cumulative_return", float("-inf")))
    best_metrics = results[best_threshold]
    logger.info("-" * 60)
    logger.info(f"최적 임계값: {best_threshold}% (수익률 {best_metrics['cumulative_return']:+.2f}%)")

    return results, best_threshold, best_metrics
