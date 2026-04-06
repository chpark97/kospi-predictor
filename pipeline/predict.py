"""일일 예측 실행 파이프라인 (리스크 필터 포함)"""
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
from models.lstm_attention import LSTMAttention
from models.lstm_baseline import LSTMBaseline
from preprocessing.feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)

SAVE_DIR = Path(__file__).parent.parent / "saved_models"


def load_model(model_path=None):
    """저장된 모델 로드"""
    if model_path is None:
        # 최신 final 모델 찾기
        candidates = list(SAVE_DIR.glob("*_final.pt"))
        if not candidates:
            raise FileNotFoundError("학습된 모델이 없습니다. 먼저 train을 실행해주세요.")
        model_path = max(candidates, key=lambda p: p.stat().st_mtime)

    checkpoint = torch.load(model_path, weights_only=False)

    num_features = checkpoint["num_features"]
    model_type = checkpoint.get("model_type", "attention")

    if model_type == "attention":
        model = LSTMAttention(num_features)
    else:
        model = LSTMBaseline(num_features)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Scaler 복원
    scaler = StandardScaler()
    scaler.mean_ = checkpoint["scaler_mean"]
    scaler.scale_ = checkpoint["scaler_scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = num_features

    return model, scaler, checkpoint["feature_names"]


def get_latest_vix():
    """최신 VIX 값 조회"""
    conn = sqlite3.connect(DB_PATH)
    try:
        result = conn.execute(
            "SELECT close, date FROM yahoo_vix ORDER BY date DESC LIMIT 1"
        ).fetchone()
        return (result[0], result[1]) if result else (None, None)
    finally:
        conn.close()


def get_previous_kospi_return():
    """전일 코스피 등락률 조회"""
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT close FROM kospi_index ORDER BY date DESC LIMIT 2"
        ).fetchall()
        if len(rows) >= 2:
            return (rows[0][0] / rows[1][0] - 1) * 100
        return 0.0
    finally:
        conn.close()


def run_prediction():
    """일일 예측 실행"""
    today = datetime.now().strftime("%Y-%m-%d")

    # 1. 데이터 수집 (최신 데이터 업데이트)
    logger.info("데이터 업데이트 중...")
    from collectors import YahooCollector, KRXCollector, FREDCollector
    YahooCollector().collect()
    KRXCollector().collect()
    FREDCollector().collect()

    # 2. 피처 생성
    fe = FeatureEngineer()
    df = fe.build_dataset()

    # 3. 모델 로드
    model, scaler, feature_names = load_model()

    # 4. 최근 시퀀스로 예측
    X, _, dates, _ = fe.prepare_sequences(df, scaler=scaler, fit_scaler=False)
    latest_seq = torch.FloatTensor(X[-1:])

    with torch.no_grad():
        pred_return, confidence = model(latest_seq)
        pred_return = pred_return.item()
        confidence = confidence.item() * 100  # %

    # 5. 리스크 필터
    vix_value, vix_date = get_latest_vix()
    prev_return = get_previous_kospi_return()

    direction = "▲ 상승" if pred_return > 0 else "▼ 하락"
    sign = "+" if pred_return > 0 else ""

    # 리스크 상태 판단
    risk_warnings = []
    signal_valid = True

    if vix_value and vix_value >= VIX_THRESHOLD:
        risk_warnings.append(f"⚠ 고변동성 경고 (VIX={vix_value:.1f} >= {VIX_THRESHOLD})")
        signal_valid = False

    if abs(prev_return) >= LARGE_MOVE_THRESHOLD:
        risk_warnings.append(f"⚠ 전일 대폭 변동 ({prev_return:+.2f}%) - 신뢰도 하향")
        confidence *= 0.8  # 20% 신뢰도 감소

    if confidence < CONFIDENCE_THRESHOLD:
        risk_warnings.append(f"⚠ 낮은 확신도 ({confidence:.1f}% < {CONFIDENCE_THRESHOLD}%) - 신호 미출력")
        signal_valid = False

    vix_status = "정상" if (vix_value and vix_value < VIX_THRESHOLD) else "경고"

    # 6. 결과 출력
    output_lines = [
        f"\n[{today}] 코스피 예측",
        f"  방향: {direction}" if signal_valid else f"  방향: ― (신호 무력화)",
        f"  예측 등락률: {sign}{pred_return:.2f}%",
        f"  신뢰도: {confidence:.1f}%",
        f"  VIX: {vix_value:.1f} ({vix_status})" if vix_value else "  VIX: N/A",
    ]

    for warning in risk_warnings:
        output_lines.append(f"  {warning}")

    if not signal_valid:
        output_lines.append("  → 리스크 필터 발동: 거래 신호 없음")

    result_text = "\n".join(output_lines)
    logger.info(result_text)

    # 7. 로그 저장
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
        "risk_warnings": risk_warnings,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_prediction()
