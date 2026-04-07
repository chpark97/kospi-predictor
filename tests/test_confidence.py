"""confidence 계산 수정 전후 비교 테스트"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def old_compute_composite_confidence(details, prediction_history=None):
    """수정 전 (기존) 로직 - 비교용"""
    all_returns = details["individual_returns"]
    individual_rets = all_returns[:, 0]

    up_vote_ratio = details["up_vote_ratio"][0]
    agreement_score = abs(up_vote_ratio - 0.5) * 200

    mean_abs = np.mean(np.abs(individual_rets))
    strength_score = min(mean_abs / 0.5 * 100, 100)

    majority_up = up_vote_ratio > 0.5
    same_dir = individual_rets[individual_rets > 0] if majority_up else individual_rets[individual_rets <= 0]
    if len(same_dir) > 1:
        consistency_score = max(0, 100 - np.std(same_dir) * 200)
    else:
        consistency_score = 50

    accuracy_score = 50.0  # backfill에서는 항상 이 값

    composite = (
        agreement_score * 0.30
        + strength_score * 0.15
        + consistency_score * 0.25
        + accuracy_score * 0.30
    )
    return composite, agreement_score, strength_score, consistency_score, accuracy_score


def new_compute_composite_confidence(details, prediction_history=None):
    """수정 후 (개선) 로직 — 상승 편향 패널티 포함"""
    all_returns = details["individual_returns"]
    individual_rets = all_returns[:, 0]

    up_vote_ratio = details["up_vote_ratio"][0]
    raw_agreement = abs(up_vote_ratio - 0.5) * 2
    agreement_score = raw_agreement ** 0.6 * 100

    # 상승 편향 패널티: up_ratio > 0.8이면 합의도 점수 감쇄
    if up_vote_ratio > 0.8:
        bias_penalty = (up_vote_ratio - 0.8) * 2.5  # 0.8→0, 1.0→0.5
        agreement_score *= (1 - bias_penalty)

    mean_abs = np.mean(np.abs(individual_rets))
    strength_score = min((mean_abs / 0.15) ** 0.7 * 70, 100)

    majority_up = up_vote_ratio > 0.5
    same_dir = individual_rets[individual_rets > 0] if majority_up else individual_rets[individual_rets <= 0]
    if len(same_dir) > 1:
        cv = np.std(same_dir) / (np.abs(np.mean(same_dir)) + 1e-8)
        consistency_score = max(0, min(100, 100 - cv * 100))
    else:
        consistency_score = 30

    predicted_up = np.mean(individual_rets) > 0
    # prediction_history에서 검증된 기록 활용
    # 기본값: 35.0 (보수적 운영, 기존 50.0에서 하향)
    accuracy_score = 35.0
    if prediction_history:
        recent_verified = [h for h in prediction_history[-20:] if h.get("actual_direction")]
        if len(recent_verified) >= 3:
            same_pred = [
                h for h in recent_verified
                if (h["predicted_direction"] == "up") == predicted_up
            ]
            if len(same_pred) >= 2:
                dir_acc = sum(
                    1 for h in same_pred
                    if h["predicted_direction"] == h["actual_direction"]
                ) / len(same_pred)
                accuracy_score = dir_acc * 100

    composite = (
        agreement_score * 0.30
        + strength_score * 0.15
        + consistency_score * 0.25
        + accuracy_score * 0.30
    )
    return composite, agreement_score, strength_score, consistency_score, accuracy_score


def make_details(individual_rets):
    """테스트용 details dict 생성"""
    rets = np.array(individual_rets).reshape(-1, 1)
    directions = (rets > 0).astype(float)
    weights = np.ones(len(rets)) / len(rets)
    up_vote_ratio = np.average(directions, axis=0, weights=weights)
    return {
        "individual_returns": rets,
        "up_vote_ratio": up_vote_ratio,
    }


# 실제 prediction_history.json의 데이터로 히스토리 구축
SAMPLE_HISTORY = [
    {"date": f"2026-03-{d:02d}", "predicted_direction": "up", "actual_direction": act}
    for d, act in [
        (9, "down"), (10, "up"), (11, "up"), (12, "down"),
        (13, "down"), (16, "up"), (17, "up"), (18, "up"),
        (19, "down"), (20, "up"), (23, "down"), (24, "up"),
        (25, "up"), (26, "down"), (27, "down"), (30, "down"),
        (31, "down"),
    ]
]


def test_all_up_weak_predictions_below_threshold():
    """전부 상승 약한 예측값 시나리오에서 신뢰도가 임계값(50%) 미만인지 검증

    상승 편향 패널티(up_ratio > 0.8)와 보수적 fallback(35.0)이 적용되면
    전부 상승 예측 시 신뢰도가 50% 미만으로 나와야 리스크 필터가 작동함.
    """
    # 실제 문제 상황: 9개 모델 전부 약한 상승 예측
    weak_all_up = [0.04, 0.05, 0.03, 0.06, 0.04, 0.05, 0.03, 0.04, 0.05]
    details = make_details(weak_all_up)

    # 히스토리 없이 테스트 (fallback=35.0 적용)
    composite, agreement, strength, consistency, accuracy = new_compute_composite_confidence(details)

    # 검증: 신뢰도가 50% 미만이어야 bull 레짐에서 리스크 필터 발동
    BULL_THRESHOLD = 50.0
    assert composite < BULL_THRESHOLD, (
        f"전부 상승 약한 예측 시 신뢰도({composite:.1f}%)가 "
        f"bull 임계값({BULL_THRESHOLD}%) 이상 - 편향 교정 실패"
    )

    # up_ratio=1.0 → 상승 편향 패널티로 합의도가 크게 감소해야 함
    # 패널티 없을 때 100점 → 패널티 적용 후 ~50점 (부동소수점 오차 허용)
    assert agreement < 51, f"up_ratio=1.0인데 합의도({agreement:.0f})가 50 초과 - 패널티 미적용"

    print(f"✅ 테스트 통과: 전부 상승 약한 예측 → 신뢰도 {composite:.1f}% < {BULL_THRESHOLD}%")
    print(f"   (합의={agreement:.0f}, 강도={strength:.0f}, 일관={consistency:.0f}, 정확={accuracy:.0f})")
    return True


def main():
    # 편향 교정 테스트 먼저 실행
    print("=" * 80)
    print("편향 교정 테스트")
    print("=" * 80)
    test_all_up_weak_predictions_below_threshold()
    print()

    scenarios = {
        "9개 전부 상승 (강한 합의, 큰 예측값)": [0.25, 0.20, 0.22, 0.18, 0.24, 0.21, 0.19, 0.23, 0.20],
        "9개 전부 상승 (약한 예측값)":          [0.04, 0.05, 0.03, 0.06, 0.04, 0.05, 0.03, 0.04, 0.05],
        "8:1 상승 (전형적 데이터)":             [0.10, 0.12, 0.08, 0.11, 0.09, 0.13, 0.10, -0.05, 0.11],
        "7:2 상승":                             [0.10, 0.12, 0.08, -0.03, 0.09, 0.13, 0.10, -0.05, 0.11],
        "6:3 분할":                             [0.08, 0.05, -0.04, -0.06, 0.09, 0.07, -0.03, 0.06, 0.04],
        "5:4 분할 (낮은 합의)":                 [0.05, -0.03, 0.04, -0.02, 0.06, -0.04, -0.01, 0.03, 0.02],
        "전부 하락":                            [-0.15, -0.10, -0.12, -0.18, -0.14, -0.11, -0.13, -0.16, -0.10],
        "2:7 하락 우세":                        [0.03, -0.08, -0.10, -0.06, -0.09, -0.07, 0.02, -0.11, -0.08],
    }

    print("=" * 110)
    print(f"{'시나리오':<35} | {'OLD':>8} | {'NEW':>8} | {'NEW+hist':>8} | 지표 변화 (합의/강도/일관/정확)")
    print("=" * 110)

    for name, rets in scenarios.items():
        details = make_details(rets)

        old_c, old_a, old_s, old_con, old_acc = old_compute_composite_confidence(details)
        new_c, new_a, new_s, new_con, new_acc = new_compute_composite_confidence(details)
        new_h, new_ha, new_hs, new_hcon, new_hacc = new_compute_composite_confidence(details, SAMPLE_HISTORY)

        print(
            f"{name:<35} | {old_c:>7.1f}% | {new_c:>7.1f}% | {new_h:>7.1f}% | "
            f"합의 {old_a:>4.0f}→{new_a:>4.0f}  "
            f"강도 {old_s:>4.0f}→{new_s:>4.0f}  "
            f"일관 {old_con:>4.0f}→{new_con:>4.0f}  "
            f"정확 {old_acc:>4.0f}→{new_hacc:>4.0f}"
        )

    print("=" * 110)
    print()

    # 범위 확인
    all_old = []
    all_new = []
    all_new_h = []
    for rets in scenarios.values():
        details = make_details(rets)
        old_c, *_ = old_compute_composite_confidence(details)
        new_c, *_ = new_compute_composite_confidence(details)
        new_h, *_ = new_compute_composite_confidence(details, SAMPLE_HISTORY)
        all_old.append(old_c)
        all_new.append(new_c)
        all_new_h.append(new_h)

    print(f"OLD 범위: {min(all_old):.1f}% ~ {max(all_old):.1f}%  (차이: {max(all_old)-min(all_old):.1f}%)")
    print(f"NEW 범위: {min(all_new):.1f}% ~ {max(all_new):.1f}%  (차이: {max(all_new)-min(all_new):.1f}%)")
    print(f"NEW+hist: {min(all_new_h):.1f}% ~ {max(all_new_h):.1f}%  (차이: {max(all_new_h)-min(all_new_h):.1f}%)")


if __name__ == "__main__":
    main()
