"""전역 설정"""
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "kospi.db"
LOG_PATH = PROJECT_ROOT / "results.log"

# 데이터 수집 범위
START_DATE = "2015-01-01"
END_DATE = None  # None이면 오늘까지

# FRED API 키 (환경변수에서 읽기)
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# yfinance 티커 매핑
YAHOO_TICKERS = {
    # 미국 지수
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "dow": "^DJI",
    "vix": "^VIX",
    "dxy": "DX-Y.NYB",
    # 아시아 지수
    "nikkei": "^N225",
    "hangseng": "^HSI",
    "shanghai": "000001.SS",
    # 원자재
    "wti": "CL=F",
    "copper": "HG=F",
    # 환율
    "usdkrw": "KRW=X",
}

# FRED 시리즈 ID
FRED_SERIES = {
    "us10y": "DGS10",  # 미국 10년물 금리
}

# 코스피 지수 티커 (pykrx)
KOSPI_TICKER = "1001"  # 코스피 지수

# 모델 설정
SEQUENCE_LENGTH = 20
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
ATTENTION_HEADS = 4
DROPOUT = 0.3  # 과적합 방지 강화
LEARNING_RATE = 0.0005
BATCH_SIZE = 64
EPOCHS = 80
EARLY_STOPPING_PATIENCE = 15
MIN_EPOCHS = 15  # 최소 학습 에폭

# 리스크 필터 임계값
VIX_THRESHOLD = 30.0
LARGE_MOVE_THRESHOLD = 3.0  # ±3%
CONFIDENCE_THRESHOLD = 65.0  # 65%

# 백테스트 수수료
COMMISSION_RATE = 0.00015  # 0.015%

# Walk-forward 설정
WALK_FORWARD_SPLITS = [
    {"train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31"},
    {"train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-12-31"},
    {"train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31"},
    {"train_end": "2024-12-31", "test_start": "2025-01-01", "test_end": "2025-12-31"},
    {"train_end": "2025-12-31", "test_start": "2026-01-01", "test_end": "2026-04-06"},
]
