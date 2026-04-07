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
    get_dynamic_weights, get_recent_accuracy, load_weights, update_weights,
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
                raise FileNotFoundError("학습된 모델이 없습니다.")
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
        ens = EnsemblePredictor()
        ens.add_model(m, 1.0, mt)
        return ens, scaler, checkpoint["feature_names"]


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


# ── 예측 이력 ──

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
    history = [h for h in history if h["date"] != result["date"]]
    history.append(entry)
    history = history[-90:]
    _save_prediction_history(history)


# ── 복합 신뢰도 ──

def compute_composite_confidence(details, regime, prediction_history=None):
    """개선된 복합 신뢰도 — 4개 지표의 가중 합산

    각 지표가 20~90% 범위를 커버하도록 스케일링 조정.
    상승 편향 감지 시 강제 신뢰도 패널티 적용.
    """
    all_returns = details["individual_returns"]
    individual_rets = all_returns[:, 0]

    # 1. 모델 합의도 (0~100)
    # up_vote_ratio가 극단일수록 (0 또는 1에 가까울수록) 높은 점수
    # 비선형 매핑으로 6:3 이상 분할에서 의미 있는 차이 생성
    up_vote_ratio = details["up_vote_ratio"][0]
    raw_agreement = abs(up_vote_ratio - 0.5) * 2  # 0~1
    agreement_score = raw_agreement ** 0.6 * 100   # 비선형: 5:4→18, 6:3→53, 7:2→72, 8:1→87, 9:0→100

    # 상승 편향 패널티: up_ratio > 0.8이면 합의도 점수 감쇄
    # 전부 상승 예측(up_ratio=1.0)은 "합의"가 아니라 "편향"일 가능성
    if up_vote_ratio > 0.8:
        bias_penalty = (up_vote_ratio - 0.8) * 2.5  # 0.8→0, 1.0→0.5
        agreement_score *= (1 - bias_penalty)
        logger.debug(f"  상승 편향 패널티: up_ratio={up_vote_ratio:.2f} → 합의도 {bias_penalty:.0%} 감쇄")

    # 2. 예측 강도 (0~100)
    # 실제 예측값 분포(0.04~0.3%)에 맞게 분모 조정
    # 기존: mean_abs/0.5*100 → 항상 8~60 (변별력 부족)
    # 개선: 0.1%를 중간점으로, 비선형 스케일링
    mean_abs = np.mean(np.abs(individual_rets))
    strength_score = min((mean_abs / 0.15) ** 0.7 * 70, 100)  # 0.04→27, 0.1→53, 0.15→70, 0.25→90

    # 3. 모델 간 일관성 (0~100)
    # 다수결 방향과 같은 모델들의 예측치 변동계수(CV)로 측정
    majority_up = up_vote_ratio > 0.5
    same_dir = individual_rets[individual_rets > 0] if majority_up else individual_rets[individual_rets <= 0]
    if len(same_dir) > 1:
        cv = np.std(same_dir) / (np.abs(np.mean(same_dir)) + 1e-8)  # 변동계수
        consistency_score = max(0, min(100, 100 - cv * 100))  # CV 0→100, CV 1→0
    else:
        consistency_score = 30  # 다수결 방향 모델이 1개뿐이면 낮은 일관성

    # 4. 최근 방향별 정확도 (0~100)
    # 현재 예측 방향(상승/하락)에서의 최근 20일 정확도
    predicted_up = np.mean(individual_rets) > 0
    accuracy_score = _compute_accuracy_score(predicted_up, prediction_history)

    composite = (
        agreement_score * 0.30
        + strength_score * 0.15
        + consistency_score * 0.25
        + accuracy_score * 0.30
    )

    logger.info(
        f"  복합신뢰도: {composite:.1f}% "
        f"(합의={agreement_score:.0f}×0.3 + 강도={strength_score:.0f}×0.15 "
        f"+ 일관={consistency_score:.0f}×0.25 + 정확={accuracy_score:.0f}×0.3)"
    )
    return composite


def _compute_accuracy_score(predicted_up, prediction_history):
    """방향별 정확도 점수 계산. 다중 소스 활용."""
    # 1순위: prediction_history에서 검증된 기록 사용
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
                return dir_acc * 100

    # 2순위: dynamic_weights.json의 모델별 정확도 활용
    dw_data = load_weights()
    if dw_data and dw_data.get("history") and len(dw_data["history"]) >= 3:
        history = dw_data["history"]
        # 각 모델의 최근 정확도를 평균
        model_accuracies = []
        for name in dw_data.get("weights", {}):
            correct_list = [h["correct"].get(name, False) for h in history[-20:]]
            if correct_list:
                model_accuracies.append(sum(correct_list) / len(correct_list))
        if model_accuracies:
            avg_acc = np.mean(model_accuracies)
            return avg_acc * 100

    # 3순위: get_recent_accuracy fallback
    recent_acc, n_days = get_recent_accuracy(30)
    if recent_acc is not None and n_days >= 5:
        return recent_acc

    # 데이터 없음 → 보수적 운영 (기존 50.0에서 35.0으로 하향)
    # 신뢰할 데이터가 없을 때 bull 레짐 임계값(50%) 미만으로 설정하여
    # 리스크 필터가 작동하도록 함
    return 35.0


# ── 일일 예측 ──

def run_prediction():
    today = datetime.now().strftime("%Y-%m-%d")

    # 1. 데이터 업데이트 (수집 실패해도 기존 데이터로 예측 진행)
    logger.info("데이터 업데이트 중...")
    from collectors import YahooCollector, KRXCollector, FREDCollector, InvestorCollector, NewsCollector, KoreaSpecificCollector, ECOSCollector

    collectors = [
        ("Yahoo", YahooCollector),
        ("KRX", KRXCollector),
        ("FRED", FREDCollector),
        ("Investor", InvestorCollector),
        ("News", lambda: NewsCollector().collect(days_back=3)),
        ("Korea", KoreaSpecificCollector),
        ("ECOS", ECOSCollector),
    ]
    failed = []
    for name, cls in collectors:
        try:
            if callable(cls) and not isinstance(cls, type):
                cls()
            else:
                cls().collect()
        except Exception as e:
            logger.error(f"[{name}] 수집 실패: {e}")
            failed.append(name)
    if failed:
        logger.warning(f"수집 실패: {failed} — 기존 데이터로 예측 진행")

    # 2. 피처 생성
    fe = FeatureEngineer()
    df = fe.build_dataset()

    # 3. 앙상블 + 동적 가중치
    ensemble, scaler, feature_names = load_ensemble()

    member_names = [f"{mt}_{sd}" for mt, sd in ENSEMBLE_MEMBERS]
    dyn_weights = get_dynamic_weights(member_names)
    if dyn_weights is not None:
        for i, (model, _, mt) in enumerate(ensemble.models):
            ensemble.models[i] = (model, float(dyn_weights[i]), mt)

    # 3a. 지정학적 리스크 감지
    from pipeline.geopolitical_filter import check_geopolitical_risk, format_geo_alert
    geo_level, geo_details = check_geopolitical_risk()
    geo_alert = format_geo_alert(geo_level, geo_details)
    geo_forced_stop = (geo_level == "danger")

    if geo_level != "safe":
        logger.warning(f"  지정학적 리스크: {geo_level} ({geo_details['keyword_count']}건)")

    # 4. 레짐
    regime, _ = detect_regime()
    regime_label = REGIME_LABELS[regime]
    confidence_threshold = get_regime_threshold(regime)

    # 지정학적 주의 → 임계값 +20%
    if geo_level == "caution":
        confidence_threshold += 20
        logger.info(f"  지정학적 주의 → 임계값 +20% = {confidence_threshold:.0f}%")

    # 4a. 이벤트 감지
    from collectors.event_calendar import get_nearest_event
    nearest_event = get_nearest_event(today)
    event_str = ""
    if nearest_event and nearest_event["days_until"] <= 3:
        event_str = f"{nearest_event['name']} ({nearest_event['date']})"
        if nearest_event["days_until"] == 0:
            confidence_threshold += 10  # 이벤트 당일 임계값 상향
            logger.info(f"  이벤트 당일: {event_str} → 임계값 +10%")
        else:
            logger.info(f"  이벤트 {nearest_event['days_until']}일 후: {event_str}")

    # 5. 예측
    X, _, dates, _ = fe.prepare_sequences(df, scaler=scaler, fit_scaler=False)
    pred_return, _, details = ensemble.predict(X[-1:])
    pred_return = pred_return[0]

    # 5-1. 캘리브레이션 (상승 편향 제거)
    from pipeline.calibration import calibrate_prediction
    history = _load_prediction_history()
    pred_return, cal_individual, bias_msg = calibrate_prediction(
        pred_return, details["individual_returns"], history
    )
    if cal_individual is not None:
        details["individual_returns"] = cal_individual

    individual_dirs = details["individual_returns"][:, 0] > 0
    up_count = individual_dirs.sum()
    total_models = len(individual_dirs)

    # 5a. 메타 모델 (Stacking)
    from models.meta_model import MetaModel
    meta = MetaModel()
    meta_proba = None
    if meta.load():
        individual_preds = details["individual_returns"][:, 0].reshape(1, -1)
        meta_proba = meta.predict_proba(individual_preds)
        if meta_proba is not None:
            meta_proba = meta_proba[0]
            # 메타 모델이 앙상블과 다른 방향이면 플래그
            meta_up = meta_proba > 0.5
            ensemble_up = pred_return > 0
            if meta_up != ensemble_up:
                logger.info(f"  ⚡ 메타 모델 불일치: meta={meta_proba:.1%} vs ensemble={'상승' if ensemble_up else '하락'}")

    # 5b. MC Dropout 불확실성
    from pipeline.mc_dropout import mc_dropout_predict
    mc_mean, mc_std, mc_level, mc_emoji, mc_label = mc_dropout_predict(ensemble, X[-1:])

    # 5c. 멀티스텝
    from pipeline.multistep import predict_multistep, format_multistep
    ms_dirs, ms_rets, ms_all_same = predict_multistep(ensemble, X[-1:])
    ms_str = format_multistep(ms_dirs)

    # 5d. SHAP
    from pipeline.shap_explain import get_top_features, format_shap_results
    top_features = get_top_features(ensemble, X[-1:], fe.feature_names, top_k=5)
    shap_str = format_shap_results(top_features)

    # 6. 리스크 필터 (신뢰도 계산 전에 먼저 적용)
    # 기존 순서: 신뢰도 → 멀티스텝 보너스 → MC Dropout → 리스크 필터
    # 변경 순서: 리스크 필터 → 신뢰도 계산 → MC Dropout → 임계값 비교
    vix_value, _ = get_latest_vix()
    prev_return = get_previous_kospi_return()
    sentiment = get_latest_sentiment()

    direction = "▲ 상승" if pred_return > 0 else "▼ 하락"
    sign = "+" if pred_return > 0 else ""

    risk_warnings = []
    signal_valid = True
    risk_confidence_penalty = 1.0  # 리스크에 의한 신뢰도 감쇄 계수

    # 지정학적 리스크 강제 중단
    if geo_forced_stop:
        risk_warnings.append("🚨 지정학적 리스크 → 거래 전면 중단")
        signal_valid = False

    if vix_value and vix_value >= VIX_THRESHOLD:
        risk_warnings.append(f"⚠ 고변동성 (VIX={vix_value:.1f})")
        signal_valid = False

    if abs(prev_return) >= LARGE_MOVE_THRESHOLD:
        risk_warnings.append(f"⚠ 전일 대폭 변동 ({prev_return:+.2f}%)")
        risk_confidence_penalty *= 0.8

    if sentiment is not None and abs(sentiment) >= 0.3:
        if (pred_return > 0 and sentiment < -0.3) or (pred_return < 0 and sentiment > 0.3):
            risk_warnings.append(f"⚠ 감성 역행")
            risk_confidence_penalty *= 0.85

    # 7. 복합 신뢰도 (리스크 감쇄 후 계산)
    confidence = compute_composite_confidence(details, regime, history)
    confidence *= risk_confidence_penalty
    if risk_confidence_penalty < 1.0:
        logger.info(f"  리스크 감쇄: ×{risk_confidence_penalty:.2f} → {confidence:.1f}%")

    # 멀티스텝 결과는 슬랙 메시지에만 표시, 신뢰도 보너스 없음

    # MC Dropout 고불확실성이면 신뢰도 감소
    if mc_level == "high":
        confidence *= 0.85
        logger.info(f"  MC Dropout 고불확실성: 신뢰도 -15% → {confidence:.1f}%")

    if confidence < confidence_threshold:
        risk_warnings.append(f"⚠ 신뢰도 부족 ({confidence:.1f}%<{confidence_threshold:.0f}%)")
        signal_valid = False

    vix_status = "정상" if (vix_value and vix_value < VIX_THRESHOLD) else "경고"
    sent_str = f"{sentiment:+.2f}" if sentiment is not None else "N/A"

    # 8. 포트폴리오 (손절/익절 + 포지션 사이징)
    from portfolio.simulator import execute_trade, format_portfolio_summary
    execute_trade(today, signal_valid, "up" if pred_return > 0 else "down",
                  mc_level=mc_level)
    portfolio_str = format_portfolio_summary()

    # 9. 결과 출력
    output_lines = [
        f"\n[{today}] 코스피 예측",
        f"  레짐: {regime_label}",
        f"  단기전망: {ms_str}",
        f"  방향: {direction}" if signal_valid else f"  방향: ― (신호 무력화)",
        f"  예측 등락률: {sign}{pred_return:.2f}% (±{mc_std:.2f}%) 불확실성: {mc_label} {mc_emoji}",
        f"  신뢰도: {confidence:.1f}% (임계={confidence_threshold:.0f}%)",
        f"  모델 합의: {int(up_count)}/{total_models}",
        f"  VIX: {vix_value:.1f} ({vix_status})" if vix_value else "  VIX: N/A",
        f"  감성: {sent_str}",
        f"  주요 근거:",
        shap_str,
    ]
    for w in risk_warnings:
        output_lines.append(f"  {w}")
    if signal_valid:
        output_lines.append("  → ✅ 거래 신호 활성")
    else:
        output_lines.append("  → 리스크 필터 발동")

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
        "multistep": ms_str,
        "top_features": [(n, d) for n, d, _ in top_features],
        "mc_std": round(mc_std, 3),
        "mc_label": mc_label,
        "mc_emoji": mc_emoji,
        "portfolio": format_portfolio_summary(),
        "meta_proba": round(float(meta_proba), 3) if meta_proba is not None else None,
        "event": event_str,
        "geo_level": geo_level,
    }

    _save_today_prediction(result, individual_dirs)

    # 10. 슬랙 알림
    try:
        from notifications.slack_notifier import send_prediction_alert, send_slack_message
        # 지정학적 긴급 알림 (예측보다 먼저)
        if geo_alert:
            send_slack_message(geo_alert)
        # 편향 경고
        if bias_msg:
            send_slack_message(bias_msg)
        # 예측 알림
        send_prediction_alert(result)
    except Exception as e:
        logger.warning(f"슬랙 알림 실패: {e}")

    return result


# ── 정답 확인 ──

def run_verify_yesterday():
    logger.info("정답 확인 중...")

    from collectors import YahooCollector, KRXCollector
    YahooCollector().collect()
    KRXCollector().collect()

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

    history = _load_prediction_history()
    target = None
    for e in history:
        if e["date"] in (yesterday_date, today_date):
            target = e
            break
    if not target:
        for e in reversed(history):
            if e.get("actual_direction") is None:
                target = e
                break
    if not target:
        logger.info("확인할 예측 없음")
        return

    predicted_up = target["predicted_direction"] == "up"
    correct = predicted_up == actual_up
    target["actual_direction"] = "up" if actual_up else "down"
    target["actual_return"] = round(actual_return, 4)
    _save_prediction_history(history)

    # 동적 가중치
    member_names = [f"{mt}_{sd}" for mt, sd in ENSEMBLE_MEMBERS]
    if target.get("individual_dirs"):
        update_weights(target["individual_dirs"], actual_up, member_names)

    # 포트폴리오 정산
    from portfolio.simulator import settle_trade
    settle_trade(actual_return)

    # 모니터링
    from pipeline.monitor import check_model_performance
    _, alert_msg = check_model_performance(history)

    # 결과
    pred_str = "▲ 상승" if predicted_up else "▼ 하락"
    actual_str = f"+{actual_return:.1f}%" if actual_return > 0 else f"{actual_return:.1f}%"
    icon = "✅" if correct else "❌"
    msg = f"{icon} 어제 예측 {'정답' if correct else '오답'}! ({pred_str} 예측 → 실제 {actual_str})"

    from portfolio.simulator import format_portfolio_summary
    msg += f"\n\n{format_portfolio_summary()}"

    logger.info(msg)

    try:
        from notifications.slack_notifier import send_slack_message
        send_slack_message(msg)
        if alert_msg:
            send_slack_message(alert_msg)
    except Exception as e:
        logger.warning(f"슬랙 전송 실패: {e}")

    return {"correct": correct, "actual_return": actual_return}


# ── 주간 리포트 ──

def run_weekly_report():
    logger.info("주간 리포트 생성 중...")

    from collectors import YahooCollector, KRXCollector
    YahooCollector().collect()
    KRXCollector().collect()

    history = _load_prediction_history()
    if not history:
        logger.info("예측 이력 없음")
        return

    today = datetime.now()
    week_ago = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    weekly = [h for h in history if h["date"] >= week_ago]

    if not weekly:
        logger.info("이번 주 예측 없음")
        return

    total = len(weekly)
    signaled = [h for h in weekly if h.get("signal_valid")]
    blocked = total - len(signaled)
    verified = [h for h in weekly if h.get("actual_direction") is not None]
    correct = sum(1 for h in verified if h["predicted_direction"] == h["actual_direction"])
    accuracy = correct / len(verified) * 100 if verified else 0

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT close FROM kospi_index WHERE date >= ? ORDER BY date", (week_ago,)
    ).fetchall()
    conn.close()
    weekly_return = (rows[-1][0] / rows[0][0] - 1) * 100 if len(rows) >= 2 else 0

    first_date = weekly[0]["date"]
    last_date = weekly[-1]["date"]

    from portfolio.simulator import format_portfolio_summary
    portfolio = format_portfolio_summary()

    report = (
        f"📈 *주간 성과 리포트 ({first_date} ~ {last_date})*\n\n"
        f"예측 횟수: {total}회 (신호 {len(signaled)}회, 차단 {blocked}회)\n"
        f"정답률: {correct}/{len(verified)} ({accuracy:.1f}%)\n"
        f"이번 주 코스피: {weekly_return:+.1f}%\n\n"
        f"{portfolio}"
    )

    # 차트 생성
    from pipeline.chart import generate_weekly_chart
    chart_path = generate_weekly_chart(history)
    if chart_path:
        report += f"\n\n📊 차트 저장: {chart_path}"

    logger.info(report)

    try:
        from notifications.slack_notifier import send_slack_message
        send_slack_message(report)
    except Exception as e:
        logger.warning(f"슬랙 전송 실패: {e}")

    return {"total": total, "accuracy": accuracy, "weekly_return": weekly_return}


# ── 소급 초기화 ──

def backfill_prediction_history():
    logger.info("소급 초기화 중...")

    try:
        ensemble, scaler, _ = load_ensemble()
    except FileNotFoundError:
        logger.warning("모델 없음")
        return

    fe = FeatureEngineer()
    df = fe.build_dataset()
    X_all, y_all, dates_all, _ = fe.prepare_sequences(df, scaler=scaler, fit_scaler=False)

    history = _load_prediction_history()
    existing = {h["date"] for h in history}
    member_names = [f"{mt}_{sd}" for mt, sd in ENSEMBLE_MEMBERS]
    regime, _ = detect_regime()
    added = 0

    for i in range(max(0, len(X_all) - 30), len(X_all)):
        if dates_all[i] in existing:
            continue
        pred_ret, _, details = ensemble.predict(X_all[i:i+1])
        dirs = list(details["individual_returns"][:, 0] > 0)
        actual_up = y_all[i] > 0

        # 실제 신뢰도 계산 (기존 히스토리를 prediction_history로 전달)
        confidence = compute_composite_confidence(details, regime, history)

        history.append({
            "date": dates_all[i],
            "predicted_direction": "up" if pred_ret[0] > 0 else "down",
            "predicted_return": round(float(pred_ret[0]), 4),
            "confidence": round(confidence, 1),
            "signal_valid": True,
            "individual_dirs": [bool(d) for d in dirs],
            "actual_direction": "up" if actual_up else "down",
            "actual_return": round(float(y_all[i]), 4),
        })
        update_weights(dirs, actual_up, member_names)
        added += 1

    history = history[-90:]
    _save_prediction_history(history)

    # 메타 모델 학습
    from models.meta_model import train_meta_from_history
    train_meta_from_history(ensemble, X_all[-60:], y_all[-60:])

    logger.info(f"소급 완료: {added}건 (총 {len(history)}건)")


# ── 주간 재학습 ──

def run_retrain():
    import time
    start = time.time()
    logger.info("주간 모델 재학습 시작")

    from collectors import YahooCollector, KRXCollector, FREDCollector
    YahooCollector().collect()
    KRXCollector().collect()
    FREDCollector().collect()

    from pipeline.train import walk_forward_train
    walk_forward_train("ensemble")

    backfill_prediction_history()

    elapsed = int(time.time() - start)

    conn = sqlite3.connect(DB_PATH)
    r = conn.execute("SELECT MIN(date), MAX(date) FROM kospi_index").fetchone()
    conn.close()

    msg = (
        f"🔄 *주간 모델 재학습 완료*\n\n"
        f"학습 데이터: {r[0]} ~ {r[1]}\n"
        f"소요 시간: {elapsed // 60}분\n"
        f"앙상블: {len(ENSEMBLE_MEMBERS)}개 모델"
    )
    logger.info(msg)

    try:
        from notifications.slack_notifier import send_slack_message
        send_slack_message(msg)
    except Exception as e:
        logger.warning(f"슬랙 전송 실패: {e}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_prediction()
