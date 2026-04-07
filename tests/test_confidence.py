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
    """수정 후 (개선) 로직"""
    all_returns = details["individual_returns"]
    individual_rets = all_returns[:, 0]

    up_vote_ratio = details["up_vote_ratio"][0]
    raw_agreement = abs(up_vote_ratio - 0.5) * 2
    agreement_score = raw_agreement ** 0.6 * 100

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
    accuracy_score = 50.0
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


def main():
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
