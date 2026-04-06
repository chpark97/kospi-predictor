"""앙상블 동적 가중치 관리

최근 30일 각 모델별 방향정확도를 추적하고,
성과에 비례하여 가중치를 동적으로 조정합니다.
"""
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

WEIGHTS_PATH = Path(__file__).parent / "dynamic_weights.json"


def load_weights():
    """저장된 동적 가중치 로드"""
    if WEIGHTS_PATH.exists():
        with open(WEIGHTS_PATH) as f:
            return json.load(f)
    return None


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def save_weights(data):
    """동적 가중치 저장"""
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
    logger.info(f"[DynamicWeights] 저장: {WEIGHTS_PATH}")


def update_weights(individual_predictions, actual_direction, member_names):
    """새로운 실제 결과를 반영하여 가중치 업데이트

    Args:
        individual_predictions: 각 모델의 예측 방향 (list of bool, True=상승)
        actual_direction: 실제 방향 (True=상승)
        member_names: 모델 이름 리스트 ["attention_42", "baseline_123", ...]
    """
    data = load_weights() or {"history": [], "weights": {}}
    history = data["history"]

    # 새 기록 추가
    record = {
        "correct": {},
    }
    for i, name in enumerate(member_names):
        if i < len(individual_predictions):
            predicted_up = bool(individual_predictions[i])
            record["correct"][name] = (predicted_up == actual_direction)
        else:
            record["correct"][name] = False  # 기록 없는 모델은 오답 처리

    history.append(record)

    # 최근 30일만 유지
    if len(history) > 30:
        history = history[-30:]
    data["history"] = history

    # 각 모델별 최근 DA 계산 → 가중치
    weights = {}
    recent_10 = history[-10:] if len(history) >= 10 else history

    for name in member_names:
        correct_list = [h["correct"].get(name, False) for h in history]
        if correct_list:
            da = sum(correct_list) / len(correct_list)
        else:
            da = 0.5

        # 자동 비활성화: 최근 10일 DA가 20% 이하면 가중치 0
        if len(recent_10) >= 10:
            recent_correct = [h["correct"].get(name, False) for h in recent_10]
            recent_da = sum(recent_correct) / len(recent_correct)
            if recent_da <= 0.2:
                weights[name] = 0.0
                logger.warning(f"  ⚠ {name} 자동 비활성화: 최근 10일 DA={recent_da*100:.0f}%")
                continue

        # 가중치: DA에 비례, 최소 0.1
        weights[name] = max(da, 0.1)

    data["weights"] = weights
    save_weights(data)

    logger.info(f"[DynamicWeights] 업데이트 ({len(history)}일 기록)")
    for name, w in weights.items():
        logger.info(f"  {name}: DA={w*100:.0f}% → weight={w:.2f}")

    return weights


def get_dynamic_weights(member_names):
    """현재 동적 가중치 조회. 없으면 균등 가중치."""
    data = load_weights()
    if not data or not data.get("weights"):
        return None

    weights = []
    for name in member_names:
        weights.append(data["weights"].get(name, 0.5))

    weights_arr = np.array(weights)

    # 가중치 0인 모델이 과반이면 경고
    n_disabled = np.sum(weights_arr == 0)
    if n_disabled > len(weights_arr) / 2:
        logger.warning(
            f"[DynamicWeights] ⚠ 비활성화 모델 과반: {n_disabled}/{len(weights_arr)}개 → "
            f"앙상블 신뢰도 저하 주의"
        )

    return weights_arr


def get_recent_accuracy(n_days=30):
    """최근 N일 전체 앙상블 정확도 조회"""
    data = load_weights()
    if not data or not data.get("history"):
        return None, 0

    history = data["history"][-n_days:]
    if not history:
        return None, 0

    # 다수결 기준 정확도
    total_correct = 0
    for record in history:
        correct_votes = sum(record["correct"].values())
        total_votes = len(record["correct"])
        if correct_votes > total_votes / 2:
            total_correct += 1

    accuracy = total_correct / len(history) * 100
    return accuracy, len(history)
