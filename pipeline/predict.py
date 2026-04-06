"""일일 예측 + 정답 확인 + 주간 리포트 파이프라인"""
import json
import logging
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DB_PATH, LARGE_MOVE_THRESHOLD, LOG_PATH, VIX_THRESHOLD
from models.ensemble import ENSEMBLE_MEMBERS, EnsemblePredictor, create_model
from models.regime import detect_regime, get_regime_threshold, REGIME_LABELS
from models.dynamic_weights import (
    get_dynamic_weights, get_recent_accuracy, update_weights,
)
from preprocessing.feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)

SAVE_DIR = Path(__file__).parent.parent / "saved_models"
PREDICTION_LOG = Path(__file__).parent.parent / "data" / "prediction_history.json"


# ── 모델 로드 ──

def load_ensemble(model_path=None):
    if model_path is None:
        ensemble_path = SAVE_DIR / "ensemble_final.pt"
        if ensemble_path.exists():
            model_path = ensemble_path
        else:
            candidates = list(SAVE_DIR.glob("*_final.pt"))
            if not candidates:
                raise FileNotFoundError("학습된 모델이 없습니다. 먼저 train을 실행해주세요.")
            model_path = max(candidates, key=lambda p: p.stat().st_mtime)

    checkpoint = torch.load(model_path, weights_only=False)
    num_features = checkpoint["num_features"]
    seq_length = checkpoint.get("seq_length", 20)

    scaler = StandardScaler()
    scaler.mean_ = checkpoint["scaler_mean"]
    scaler.scale_ = checkpoint["scaler_scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = num_features

    if "members" in checkpoint:
        ensemble = EnsemblePredictor()
        for member in checkpoint["members"]:
            model = create_model(member["model_type"], num_features, seq_length)
            model.load_state_dict(member["model_state"])
            model.eval()
            ensemble.add_model(model, member["weight"], member["model_type"])
        logger.info(f"앙상블 로드: {len(checkpoint['members'])}개 모델")
        return ensemble, scaler, checkpoint["feature_names"]
    else:
        from models.lstm_attention import LSTMAttention
        from models.lstm_baseline import LSTMBaseline
        mt = checkpoint.get("model_type", "attention")
        m = LSTMAttention(num_features) if mt == "attention" else LSTMBaseline(num_features)
        m.load_state_dict(checkpoint["model_state"])
        m.eval()
        ensemble = EnsemblePredictor()
        ensemble.add_model(m, 1.0, mt)
        return ensemble, scaler, checkpoint["feature_names"]


# ── DB 조회 ──

def get_latest_vix():
    conn = sqlite3.connect(DB_PATH)
    try:
        r = conn.execute("SELECT close, date FROM yahoo_vix ORDER BY date DESC LIMIT 1").fetchone()
        return (r[0], r[1]) if r else (None, None)
    finally:
        conn.close()


def get_previous_kospi_return():
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("SELECT close FROM kospi_index ORDER BY date DESC LIMIT 2").fetchall()
        return (rows[0][0] / rows[1][0] - 1) * 100 if len(rows) >= 2 else 0.0
    finally:
        conn.close()


def get_latest_sentiment():
    conn = sqlite3.connect(DB_PATH)
    try:
        r = conn.execute("SELECT sentiment_score FROM news_sentiment ORDER BY date DESC LIMIT 1").fetchone()
        return r[0] if r else None
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


# ── 예측 이력 관리 ──

def _load_prediction_history():
    if PREDICTION_LOG.exists():
        with open(PREDICTION_LOG) as f:
            return json.load(f)
    return []


def _save_prediction_history(history):
    PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PREDICTION_LOG, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def _save_today_prediction(result, individual_dirs):
    history = _load_prediction_history()
    entry = {
        "date": result["date"],
        "predicted_direction": result["direction"],
        "predicted_return": result["predicted_return"],
        "confidence": result["confidence"],
        "signal_valid": result["signal_valid"],
        "individual_dirs": [bool(d) for d in individual_dirs],
        "actual_direction": None,
        "actual_return": None,
    }
    # 같은 날짜 덮어쓰기
    history = [h for h in history if h["date"] != result["date"]]
    history.append(entry)
    # 최근 90일만 유지
    history = history[-90:]
    _save_prediction_history(history)


# ── 복합 신뢰도 계산 ──

def compute_composite_confidence(details, regime, db_path=None):
    """복합 신뢰도 계산

    구성요소:
    1. 모델 합의도 (40%) - 9개 모델의 방향 일치율
    2. 예측 분포 집중도 (30%) - 예측값이 한 방향으로 몰려있는 정도
    3. 최근 앙상블 정확도 (30%) - 최근 30일 성과

    Returns:
        confidence (0~100)
    """
    all_returns = details["individual_returns"]  # (n_models, batch)
    agreement = details["agreement"][0]  # 0~1

    # 1) 합의도 점수 (0~100)
    agreement_score = agreement * 100

    # 2) 예측 분포 집중도: 개별 예측값의 부호 강도
    individual_rets = all_returns[:, 0]
    mean_abs = np.mean(np.abs(individual_rets))
    spread = np.std(individual_rets)
    # 평균 절대 예측이 크고 분산이 작으면 → 높은 집중도
    if mean_abs > 0:
        concentration = min(mean_abs / max(spread, 0.01), 5.0) / 5.0 * 100
    else:
        concentration = 0

    # 3) 최근 정확도
    recent_acc, n_days = get_recent_accuracy(30)
    if recent_acc is not None and n_days >= 5:
        accuracy_score = recent_acc
    else:
        accuracy_score = 50.0  # 기록 부족 시 기본값

    # 가중 합산
    composite = (
        agreement_score * 0.40 +
        concentration * 0.30 +
        accuracy_score * 0.30
    )

    logger.info(
        f"  복합신뢰도: {composite:.1f}% "
        f"(합의={agreement_score:.0f}% × 0.4 + "
        f"집중={concentration:.0f}% × 0.3 + "
        f"실적={accuracy_score:.0f}% × 0.3)"
    )

    return composite


# ── 일일 예측 ──

def run_prediction():
    today = datetime.now().strftime("%Y-%m-%d")

    # 1. 데이터 업데이트
    logger.info("데이터 업데이트 중...")
    from collectors import YahooCollector, KRXCollector, FREDCollector, InvestorCollector, NewsCollector
    YahooCollector().collect()
    KRXCollector().collect()
    FREDCollector().collect()
    InvestorCollector().collect()
    NewsCollector().collect(days_back=3)

    # 2. 피처 생성
    fe = FeatureEngineer()
    df = fe.build_dataset()

    # 3. 앙상블 로드 + 동적 가중치 적용
    ensemble, scaler, feature_names = load_ensemble()

    member_names = [f"{mt}_{sd}" for mt, sd in ENSEMBLE_MEMBERS]
    dyn_weights = get_dynamic_weights(member_names)
    if dyn_weights is not None:
        logger.info(f"동적 가중치 적용: {dict(zip(member_names, dyn_weights.round(2)))}")
        for i, (model, _, mt) in enumerate(ensemble.models):
            ensemble.models[i] = (model, float(dyn_weights[i]), mt)

    # 4. 레짐 감지
    regime, regime_details = detect_regime()
    regime_label = REGIME_LABELS[regime]
    confidence_threshold = get_regime_threshold(regime)

    # 5. 예측
    X, _, dates, _ = fe.prepare_sequences(df, scaler=scaler, fit_scaler=False)
    pred_return, _, details = ensemble.predict(X[-1:])
    pred_return = pred_return[0]

    individual_dirs = details["individual_returns"][:, 0] > 0
    up_count = individual_dirs.sum()
    total_models = len(individual_dirs)

    # 6. 복합 신뢰도
    confidence = compute_composite_confidence(details, regime)

    # 7. 리스크 필터
    vix_value, vix_date = get_latest_vix()
    prev_return = get_previous_kospi_return()
    sentiment = get_latest_sentiment()

    direction = "▲ 상승" if pred_return > 0 else "▼ 하락"
    sign = "+" if pred_return > 0 else ""

    risk_warnings = []
    signal_valid = True

    if vix_value and vix_value >= VIX_THRESHOLD:
        risk_warnings.append(f"⚠ 고변동성 경고 (VIX={vix_value:.1f})")
        signal_valid = False

    if abs(prev_return) >= LARGE_MOVE_THRESHOLD:
        risk_warnings.append(f"⚠ 전일 대폭 변동 ({prev_return:+.2f}%)")
        confidence *= 0.8

    if sentiment is not None and abs(sentiment) >= 0.3:
        if (pred_return > 0 and sentiment < -0.3) or (pred_return < 0 and sentiment > 0.3):
            risk_warnings.append(f"⚠ 감성 역행 (감성={sentiment:+.2f})")
            confidence *= 0.85

    if confidence < confidence_threshold:
        risk_warnings.append(f"⚠ 신뢰도 부족 ({confidence:.1f}% < {confidence_threshold:.0f}% [{regime}])")
        signal_valid = False

    vix_status = "정상" if (vix_value and vix_value < VIX_THRESHOLD) else "경고"
    sent_str = f"{sentiment:+.2f}" if sentiment is not None else "N/A"

    # 8. 결과 출력
    output_lines = [
        f"\n[{today}] 코스피 예측",
        f"  레짐: {regime_label}",
        f"  방향: {direction}" if signal_valid else f"  방향: ― (신호 무력화)",
        f"  예측 등락률: {sign}{pred_return:.2f}%",
        f"  신뢰도: {confidence:.1f}% (임계={confidence_threshold:.0f}%)",
        f"  모델 합의: {int(up_count)}/{total_models} 상승",
        f"  VIX: {vix_value:.1f} ({vix_status})" if vix_value else "  VIX: N/A",
        f"  감성: {sent_str}",
    ]
    for w in risk_warnings:
        output_lines.append(f"  {w}")
    if not signal_valid:
        output_lines.append("  → 리스크 필터 발동: 거래 신호 없음")
    if signal_valid:
        output_lines.append("  → ✅ 거래 신호 활성")

    result_text = "\n".join(output_lines)
    logger.info(result_text)

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(result_text + "\n\n")

    result = {
        "date": today,
        "direction": "up" if pred_return > 0 else "down",
        "predicted_return": round(pred_return, 4),
        "confidence": round(confidence, 1),
        "signal_valid": signal_valid,
        "vix": vix_value,
        "agreement": round(details["agreement"][0] * 100, 1),
        "up_vote": f"{int(up_count)}/{total_models}",
        "risk_warnings": risk_warnings,
        "sentiment": sentiment,
        "regime": regime,
        "regime_label": regime_label,
        "confidence_threshold": confidence_threshold,
    }

    # 예측 이력 저장
    _save_today_prediction(result, individual_dirs)

    # 9. 슬랙 알림
    try:
        from notifications.slack_notifier import send_prediction_alert
        send_prediction_alert(result)
    except Exception as e:
        logger.warning(f"슬랙 알림 실패: {e}")

    return result


# ── 정답 확인 (매일 오후 4시) ──

def run_verify_yesterday():
    """어제 예측의 정답을 확인하고 슬랙으로 전송"""
    logger.info("정답 확인 중...")

    # 데이터 업데이트
    from collectors import YahooCollector, KRXCollector
    YahooCollector().collect()
    KRXCollector().collect()

    # 최근 코스피 종가 2일분
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT date, close FROM kospi_index ORDER BY date DESC LIMIT 2").fetchall()
    conn.close()

    if len(rows) < 2:
        logger.warning("코스피 데이터 부족")
        return

    today_date, today_close = rows[0]
    yesterday_date, yesterday_close = rows[1]
    actual_return = (today_close / yesterday_close - 1) * 100
    actual_up = actual_return > 0

    # 예측 이력에서 해당 날짜 찾기
    history = _load_prediction_history()
    target_entry = None
    for entry in history:
        if entry["date"] == yesterday_date or entry["date"] == today_date:
            target_entry = entry
            break

    if not target_entry:
        # 가장 최근 미확인 예측
        for entry in reversed(history):
            if entry.get("actual_direction") is None:
                target_entry = entry
                break

    if not target_entry:
        logger.info("확인할 예측 없음")
        return

    # 정답 기록
    predicted_up = target_entry["predicted_direction"] == "up"
    correct = predicted_up == actual_up
    target_entry["actual_direction"] = "up" if actual_up else "down"
    target_entry["actual_return"] = round(actual_return, 4)
    _save_prediction_history(history)

    # 동적 가중치 업데이트
    member_names = [f"{mt}_{sd}" for mt, sd in ENSEMBLE_MEMBERS]
    if target_entry.get("individual_dirs"):
        update_weights(target_entry["individual_dirs"], actual_up, member_names)

    # 결과 출력
    pred_dir_str = "▲ 상승" if predicted_up else "▼ 하락"
    actual_dir_str = f"+{actual_return:.1f}%" if actual_return > 0 else f"{actual_return:.1f}%"
    icon = "✅" if correct else "❌"

    msg = f"{icon} 어제 예측 {'정답' if correct else '오답'}! ({pred_dir_str} 예측 → 실제 {actual_dir_str})"
    logger.info(msg)

    # 슬랙 전송
    try:
        from notifications.slack_notifier import send_slack_message
        send_slack_message(msg)
    except Exception as e:
        logger.warning(f"슬랙 전송 실패: {e}")

    return {"correct": correct, "actual_return": actual_return}


# ── 주간 리포트 (매주 월요일) ──

def run_weekly_report():
    """주간 성과 리포트 생성 및 슬랙 전송"""
    logger.info("주간 리포트 생성 중...")

    # 데이터 업데이트
    from collectors import YahooCollector, KRXCollector
    YahooCollector().collect()
    KRXCollector().collect()

    history = _load_prediction_history()
    if not history:
        logger.info("예측 이력 없음")
        return

    # 최근 7일 예측 필터
    today = datetime.now()
    week_ago = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    weekly = [h for h in history if h["date"] >= week_ago]

    if not weekly:
        logger.info("이번 주 예측 없음")
        return

    # 통계
    total = len(weekly)
    signaled = [h for h in weekly if h.get("signal_valid")]
    blocked = total - len(signaled)
    verified = [h for h in weekly if h.get("actual_direction") is not None]
    correct = sum(1 for h in verified if h["predicted_direction"] == h["actual_direction"])
    accuracy = correct / len(verified) * 100 if verified else 0

    # 이번 주 코스피 수익률
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT close FROM kospi_index WHERE date >= ? ORDER BY date", (week_ago,)
    ).fetchall()
    conn.close()

    if len(rows) >= 2:
        weekly_return = (rows[-1][0] / rows[0][0] - 1) * 100
    else:
        weekly_return = 0

    first_date = weekly[0]["date"]
    last_date = weekly[-1]["date"]

    report = (
        f"📈 *주간 성과 리포트 ({first_date} ~ {last_date})*\n\n"
        f"예측 횟수: {total}회 (신호 {len(signaled)}회, 차단 {blocked}회)\n"
        f"정답률: {correct}/{len(verified)} ({accuracy:.1f}%)\n"
        f"이번 주 코스피: {weekly_return:+.1f}%"
    )

    logger.info(report)

    try:
        from notifications.slack_notifier import send_slack_message
        send_slack_message(report)
    except Exception as e:
        logger.warning(f"슬랙 전송 실패: {e}")

    return {"total": total, "accuracy": accuracy, "weekly_return": weekly_return}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_prediction()
