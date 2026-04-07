"""상승 편향 수정 전후 비교 검증

prediction_history.json의 실제 데이터를 기반으로
수정 전/후 direction 분포를 비교합니다.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


# === 실제 prediction_history.json 데이터 로드 ===
HISTORY_PATH = Path(__file__).parent.parent / "data" / "prediction_history.json"
with open(HISTORY_PATH) as f:
    history = json.load(f)


def test_calibration_before_after():
    """calibration 수정 전/후 비교"""
    print("=" * 70)
    print("1. 캘리브레이션 수정 전후 비교")
    print("=" * 70)

    # 시뮬레이션: 30건의 predicted_return으로 테스트
    pred_returns = [h["predicted_return"] for h in history]
    actual_dirs = [h.get("actual_direction") for h in history]

    print(f"\n원본 predicted_return 분포:")
    print(f"  평균: {np.mean(pred_returns):+.4f}%")
    print(f"  범위: {np.min(pred_returns):+.4f} ~ {np.max(pred_returns):+.4f}%")
    print(f"  양수 비율: {sum(1 for r in pred_returns if r > 0)}/{len(pred_returns)}")

    # 기존 캘리브레이션 (약한 오프셋)
    up_ratio = sum(1 for h in history if h["predicted_direction"] == "up") / len(history)
    verified = [h for h in history if h.get("actual_direction")]
    actual_up_ratio = sum(1 for h in verified if h["actual_direction"] == "up") / len(verified) if verified else 0.5

    individual_rets = np.array(pred_returns)
    spread = np.std(individual_rets)
    bias_gap = up_ratio - actual_up_ratio
    old_offset = bias_gap * max(spread, 0.05) + (up_ratio - 0.55) * 0.5
    old_calibrated = [r - old_offset for r in pred_returns]

    print(f"\n기존 캘리브레이션 (spread-based):")
    print(f"  offset = {old_offset:.4f}")
    print(f"  보정 후 양수: {sum(1 for r in old_calibrated if r > 0)}/{len(old_calibrated)}")
    print(f"  보정 후 범위: {min(old_calibrated):+.4f} ~ {max(old_calibrated):+.4f}")

    # 새 캘리브레이션 (분위수 기반)
    down_ratio = 1.0 - actual_up_ratio
    new_offset = float(np.percentile(pred_returns[-30:], down_ratio * 100))
    new_calibrated = [r - new_offset for r in pred_returns]

    print(f"\n새 캘리브레이션 (분위수 기반):")
    print(f"  actual_up={actual_up_ratio:.0%}, down_ratio={down_ratio:.0%}")
    print(f"  {down_ratio*100:.0f}th percentile offset = {new_offset:.4f}")
    print(f"  보정 후 양수: {sum(1 for r in new_calibrated if r > 0)}/{len(new_calibrated)}")
    print(f"  보정 후 범위: {min(new_calibrated):+.4f} ~ {max(new_calibrated):+.4f}")

    # 방향 정확도 비교
    if verified:
        old_correct = 0
        new_correct = 0
        orig_correct = 0
        for i, h in enumerate(history):
            if h.get("actual_direction") is None:
                continue
            actual_up = h["actual_direction"] == "up"
            orig_correct += int((pred_returns[i] > 0) == actual_up)
            old_correct += int((old_calibrated[i] > 0) == actual_up)
            new_correct += int((new_calibrated[i] > 0) == actual_up)
        n = len(verified)
        print(f"\n방향 정확도 (검증된 {n}건):")
        print(f"  원본: {orig_correct}/{n} ({orig_correct/n:.1%})")
        print(f"  기존 보정: {old_correct}/{n} ({old_correct/n:.1%})")
        print(f"  새 보정: {new_correct}/{n} ({new_correct/n:.1%})")


def test_adaptive_threshold():
    """적응적 임계값 시뮬레이션"""
    print(f"\n{'='*70}")
    print("2. 적응적 임계값 시뮬레이션")
    print(f"{'='*70}")

    pred_returns = [h["predicted_return"] for h in history]

    for window in [5, 10, 20, 30]:
        up_count = 0
        correct = 0
        total = 0
        for i in range(window, len(pred_returns)):
            rolling_median = np.median(pred_returns[max(0, i-window):i])
            is_up = pred_returns[i] > rolling_median
            actual = history[i].get("actual_direction")
            if actual:
                actual_up = actual == "up"
                correct += int(is_up == actual_up)
                total += 1
            up_count += int(is_up)

        n = len(pred_returns) - window
        print(
            f"  window={window:2d}: 상승예측 {up_count}/{n} ({up_count/n:.0%})"
            f", DA={correct}/{total} ({correct/total:.1%})" if total > 0 else ""
        )


def test_directional_loss_impact():
    """DirectionalLoss 가중치 변경의 예상 효과"""
    print(f"\n{'='*70}")
    print("3. DirectionalLoss 가중치 변경 분석")
    print(f"{'='*70}")

    print("\n기존 가중치:")
    print("  loss = 1.0*MSE + 0.5*BCE_dir + 0.3*BCE_conf")
    print("  → MSE가 지배적: 양수 드리프트 학습 → 항상 상승 예측")

    print("\n새 가중치:")
    print("  loss = 0.3*MSE + 1.0*BCE_dir + 0.3*BCE_conf")
    print("  → BCE_dir가 지배적: 방향 정확도 최적화")
    print("  → 재학습 후 효과 발생 (다음 주간 재학습 시)")


def test_combined_effect():
    """수정의 종합 효과 — 역할 분리"""
    print(f"\n{'='*70}")
    print("4. 종합 효과 (역할 분리 아키텍처)")
    print(f"{'='*70}")

    pred_returns = [h["predicted_return"] for h in history]

    # 캘리브레이션: pred_return 보정 → 최종 방향 판단 (pred_return > 0)
    up_ratio = sum(1 for h in history if h["predicted_direction"] == "up") / len(history)
    verified = [h for h in history if h.get("actual_direction")]
    actual_up_ratio = sum(1 for h in verified if h["actual_direction"] == "up") / len(verified)

    down_ratio_c = 1.0 - actual_up_ratio
    cal_offset = float(np.percentile(pred_returns[-30:], down_ratio_c * 100))
    calibrated = [r - cal_offset for r in pred_returns]

    # 적응적 임계값: 개별 모델 투표 합의도에만 사용 (방향 판단 아님)
    # → 신뢰도 계산의 agreement_score에 영향
    adapt_thresholds = []
    for i in range(len(pred_returns)):
        if i < 5:
            adapt_thresholds.append(0.0)
        else:
            adapt_thresholds.append(np.median(pred_returns[max(0, i-30):i]))

    up_cal = sum(1 for r in calibrated if r > 0)
    print(f"\n역할 분리:")
    print(f"  캘리브레이션 → pred_return 보정 → 방향 판단")
    print(f"  적응적 임계값 → 개별 모델 투표 → agreement_score → 신뢰도")
    print(f"  DirectionalLoss → 재학습 시 근본 원인 해결")

    print(f"\n상승 예측 비율:")
    print(f"  수정 전: {sum(1 for r in pred_returns if r > 0)}/{len(pred_returns)} ({sum(1 for r in pred_returns if r > 0)/len(pred_returns):.0%})")
    print(f"  수정 후: {up_cal}/{len(calibrated)} ({up_cal/len(calibrated):.0%})")
    print(f"  실제:    {sum(1 for h in verified if h['actual_direction']=='up')}/{len(verified)} ({actual_up_ratio:.0%})")

    # 방향 정확도
    correct_orig = correct_cal = 0
    total = 0
    for i, h in enumerate(history):
        if h.get("actual_direction") is None:
            continue
        actual_up = h["actual_direction"] == "up"
        correct_orig += int((pred_returns[i] > 0) == actual_up)
        correct_cal += int((calibrated[i] > 0) == actual_up)
        total += 1

    print(f"\n방향 정확도:")
    print(f"  수정 전: {correct_orig}/{total} ({correct_orig/total:.1%})")
    print(f"  수정 후: {correct_cal}/{total} ({correct_cal/total:.1%})")

    # 적응적 임계값의 효과: 투표 변화 시뮬레이션
    print(f"\n적응적 임계값 효과 (투표 합의도 변화):")
    for i in [0, 10, 20, 29]:
        if i >= len(history):
            continue
        h = history[i]
        dirs_orig = h.get("individual_dirs", [])
        threshold = adapt_thresholds[i]
        # 원본 투표는 individual_dirs에 저장되어 있음
        up_orig = sum(dirs_orig)
        total_m = len(dirs_orig)
        print(f"  [{h['date']}] 원본투표: {up_orig}/{total_m} | threshold={threshold:+.4f}")


if __name__ == "__main__":
    test_calibration_before_after()
    test_adaptive_threshold()
    test_directional_loss_impact()
    test_combined_effect()
