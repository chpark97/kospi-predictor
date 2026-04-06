"""일일 예측 실행 파이프라인 (앙상블 + 리스크 필터)"""
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    CONFIDENCE_THRESHOLD, DB_PATH, LARGE_MOVE_THRESHOLD, LOG_PATH,
    VIX_THRESHOLD,
)
from models.ensemble import EnsemblePredictor, create_model
from preprocessing.feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)

SAVE_DIR = Path(__file__).parent.parent / "saved_models"


def load_ensemble(model_path=None):
    """앙상블 모델 로드"""
    if model_path is None:
        # ensemble_final.pt 우선, 없으면 단일 모델 fallback
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

    # Scaler 복원
    scaler = StandardScaler()
    scaler.mean_ = checkpoint["scaler_mean"]
    scaler.scale_ = checkpoint["scaler_scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = num_features

    if "members" in checkpoint:
        # 앙상블 모델
        ensemble = EnsemblePredictor()
        for member in checkpoint["members"]:
            model = create_model(member["model_type"], num_features, seq_length)
            model.load_state_dict(member["model_state"])
            model.eval()
            ensemble.add_model(model, member["weight"], member["model_type"])

        logger.info(f"앙상블 로드: {len(checkpoint['members'])}개 모델")
        return ensemble, scaler, checkpoint["feature_names"]
    else:
        # 단일 모델 fallback
        from models.lstm_attention import LSTMAttention
        from models.lstm_baseline import LSTMBaseline
        model_type = checkpoint.get("model_type", "attention")
        model = LSTMAttention(num_features) if model_type == "attention" else LSTMBaseline(num_features)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        ensemble = EnsemblePredictor()
        ensemble.add_model(model, 1.0, model_type)
        return ensemble, scaler, checkpoint["feature_names"]


def get_latest_vix():
    conn = sqlite3.connect(DB_PATH)
    try:
        result = conn.execute("SELECT close, date FROM yahoo_vix ORDER BY date DESC LIMIT 1").fetchone()
        return (result[0], result[1]) if result else (None, None)
    finally:
        conn.close()


def get_previous_kospi_return():
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("SELECT close FROM kospi_index ORDER BY date DESC LIMIT 2").fetchall()
        if len(rows) >= 2:
            return (rows[0][0] / rows[1][0] - 1) * 100
        return 0.0
    finally:
        conn.close()


def get_latest_sentiment():
    """최신 뉴스 감성 점수 조회"""
    conn = sqlite3.connect(DB_PATH)
    try:
        result = conn.execute(
            "SELECT sentiment_score FROM news_sentiment ORDER BY date DESC LIMIT 1"
        ).fetchone()
        return result[0] if result else None
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


def run_prediction():
    """일일 예측 실행"""
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

    # 3. 앙상블 로드
    ensemble, scaler, feature_names = load_ensemble()

    # 4. 예측
    X, _, dates, _ = fe.prepare_sequences(df, scaler=scaler, fit_scaler=False)
    pred_return, confidence, details = ensemble.predict(X[-1:])
    pred_return = pred_return[0]
    confidence = confidence[0] * 100

    # 개별 모델 방향
    individual_dirs = details["individual_returns"][:, 0] > 0
    up_count = individual_dirs.sum()
    total_models = len(individual_dirs)
    agreement_pct = details["agreement"][0] * 100

    # 5. 리스크 필터 + 감성 필터
    vix_value, vix_date = get_latest_vix()
    prev_return = get_previous_kospi_return()
    sentiment = get_latest_sentiment()

    direction = "▲ 상승" if pred_return > 0 else "▼ 하락"
    sign = "+" if pred_return > 0 else ""

    risk_warnings = []
    signal_valid = True

    if vix_value and vix_value >= VIX_THRESHOLD:
        risk_warnings.append(f"⚠ 고변동성 경고 (VIX={vix_value:.1f} >= {VIX_THRESHOLD})")
        signal_valid = False

    if abs(prev_return) >= LARGE_MOVE_THRESHOLD:
        risk_warnings.append(f"⚠ 전일 대폭 변동 ({prev_return:+.2f}%) - 신뢰도 하향")
        confidence *= 0.8

    # 감성 필터: 예측 방향과 감성이 반대일 때 신뢰도 하향
    if sentiment is not None and abs(sentiment) >= 0.3:
        if (pred_return > 0 and sentiment < -0.3) or (pred_return < 0 and sentiment > 0.3):
            risk_warnings.append(f"⚠ 감성 역행 (감성={sentiment:+.2f} vs 예측={sign}{pred_return:.2f}%) - 신뢰도 하향")
            confidence *= 0.85

    if confidence < CONFIDENCE_THRESHOLD:
        risk_warnings.append(f"⚠ 낮은 확신도 ({confidence:.1f}% < {CONFIDENCE_THRESHOLD}%) - 신호 미출력")
        signal_valid = False

    vix_status = "정상" if (vix_value and vix_value < VIX_THRESHOLD) else "경고"
    sent_str = f"{sentiment:+.2f}" if sentiment is not None else "N/A"

    # 6. 결과 출력
    output_lines = [
        f"\n[{today}] 코스피 예측",
        f"  방향: {direction}" if signal_valid else f"  방향: ― (신호 무력화)",
        f"  예측 등락률: {sign}{pred_return:.2f}%",
        f"  신뢰도: {confidence:.1f}%",
        f"  모델 합의: {int(up_count)}/{total_models} 상승 (합의도 {agreement_pct:.0f}%)",
        f"  VIX: {vix_value:.1f} ({vix_status})" if vix_value else "  VIX: N/A",
        f"  감성: {sent_str}",
    ]

    for warning in risk_warnings:
        output_lines.append(f"  {warning}")

    if not signal_valid:
        output_lines.append("  → 리스크 필터 발동: 거래 신호 없음")

    result_text = "\n".join(output_lines)
    logger.info(result_text)

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(result_text + "\n\n")

    logger.info(f"결과 저장: {LOG_PATH}")

    return {
        "date": today,
        "direction": "up" if pred_return > 0 else "down",
        "predicted_return": pred_return,
        "confidence": confidence,
        "signal_valid": signal_valid,
        "vix": vix_value,
        "agreement": agreement_pct,
        "up_vote": f"{int(up_count)}/{total_models}",
        "risk_warnings": risk_warnings,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_prediction()
