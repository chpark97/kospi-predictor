"""전처리 및 피처 엔지니어링

- 멀티소스 데이터를 날짜 기준으로 병합
- 기술적 지표 계산 (RSI, MACD, 볼린저밴드)
- 정규화 및 시퀀스 생성
"""
import logging
import sqlite3

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config.settings import DB_PATH, SEQUENCE_LENGTH, START_DATE

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """DB에서 데이터를 읽어 학습용 시퀀스 데이터를 생성"""

    def __init__(self, db_path=None, seq_length=None):
        self.db_path = db_path or DB_PATH
        self.seq_length = seq_length or SEQUENCE_LENGTH
        self.scaler = StandardScaler()
        self.feature_names = []

    def build_dataset(self):
        """전체 파이프라인: DB -> 병합 -> 피처생성 -> 정규화 -> 시퀀스"""
        df = self._load_and_merge()
        df = self._add_technical_indicators(df)
        df = self._add_derived_features(df)
        df = df.dropna().reset_index(drop=True)

        logger.info(f"최종 피처 수: {len(df.columns) - 1}, 행 수: {len(df)}")
        logger.info(f"기간: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")

        return df

    def prepare_sequences(self, df, scaler=None, fit_scaler=True):
        """DataFrame -> (X, y, dates, scaler) 시퀀스 변환

        y: 다음날 코스피 등락률
        """
        dates = df["date"].values
        # 타겟: 다음날 코스피 종가 등락률
        kospi_close = df["kospi_close"].values
        returns = np.diff(kospi_close) / kospi_close[:-1] * 100  # %

        # 피처 (date 컬럼 제외)
        feature_cols = [c for c in df.columns if c != "date"]
        self.feature_names = feature_cols
        features = df[feature_cols].values.astype(np.float64)

        # inf -> nan -> forward fill
        features = np.where(np.isinf(features), np.nan, features)
        features = pd.DataFrame(features).ffill().bfill().values

        # 정규화
        if fit_scaler:
            scaler = self.scaler
            features_scaled = scaler.fit_transform(features)
        else:
            features_scaled = scaler.transform(features)

        # 시퀀스 생성
        X, y, seq_dates = [], [], []
        for i in range(self.seq_length, len(features_scaled) - 1):
            X.append(features_scaled[i - self.seq_length:i])
            y.append(returns[i])  # i번째 날의 등락률 = close[i]/close[i-1]-1
            seq_dates.append(dates[i + 1])  # 예측 대상 날짜

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        logger.info(f"시퀀스 생성: X={X.shape}, y={y.shape}")
        return X, y, seq_dates, scaler

    def _load_and_merge(self):
        """DB의 모든 테이블을 날짜 기준으로 병합"""
        conn = sqlite3.connect(self.db_path)

        # 코스피 지수 (기준 테이블)
        kospi = pd.read_sql("SELECT * FROM kospi_index", conn)
        kospi = kospi.rename(columns={
            "open": "kospi_open", "high": "kospi_high",
            "low": "kospi_low", "close": "kospi_close",
            "volume": "kospi_volume",
        })

        merged = kospi.copy()

        # Yahoo 데이터 병합 (close만 사용, 코스피 OHLCV는 별도)
        yahoo_tables = [
            "yahoo_sp500", "yahoo_nasdaq", "yahoo_dow", "yahoo_vix",
            "yahoo_dxy", "yahoo_nikkei", "yahoo_hangseng", "yahoo_shanghai",
            "yahoo_wti", "yahoo_copper", "yahoo_usdkrw",
        ]

        for table in yahoo_tables:
            try:
                df = pd.read_sql(f"SELECT date, close FROM {table}", conn)
                col_name = table.replace("yahoo_", "")
                df = df.rename(columns={"close": col_name})
                merged = merged.merge(df, on="date", how="left")
            except Exception as e:
                logger.warning(f"{table} 병합 실패: {e}")

        # FRED 데이터 병합
        try:
            fred = pd.read_sql("SELECT date, value as us10y FROM fred_us10y", conn)
            merged = merged.merge(fred, on="date", how="left")
        except Exception as e:
            logger.warning(f"fred_us10y 병합 실패: {e}")

        conn.close()

        # 날짜 정렬
        merged = merged.sort_values("date").reset_index(drop=True)

        # 결측치 처리: forward fill -> backward fill
        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        merged[numeric_cols] = merged[numeric_cols].ffill().bfill()

        logger.info(f"병합 완료: {merged.shape}")
        return merged

    def _add_technical_indicators(self, df):
        """코스피 기술적 지표 추가"""
        close = df["kospi_close"]
        high = df["kospi_high"]
        low = df["kospi_low"]

        # RSI (14일)
        df["rsi_14"] = self._calc_rsi(close, 14)

        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # 볼린저 밴드 (20일)
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df["bb_upper"] = sma20 + 2 * std20
        df["bb_lower"] = sma20 - 2 * std20
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20
        df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        return df

    def _add_derived_features(self, df):
        """파생 피처 추가"""
        close = df["kospi_close"]

        # 이동평균 (5, 10, 20, 60일)
        for period in [5, 10, 20, 60]:
            df[f"ma_{period}"] = close.rolling(period).mean()
            df[f"ma_{period}_ratio"] = close / df[f"ma_{period}"]

        # 수익률 (1, 5, 20일)
        for period in [1, 5, 20]:
            df[f"return_{period}d"] = close.pct_change(period) * 100

        # 거래량 변화율
        df["volume_change"] = df["kospi_volume"].pct_change()
        df["volume_ma5_ratio"] = df["kospi_volume"] / df["kospi_volume"].rolling(5).mean()

        # 변동성 (20일 표준편차)
        df["volatility_20d"] = close.pct_change().rolling(20).std() * np.sqrt(252) * 100

        # 이동평균 절대값 정리 (비율만 유지하고 절대값 제거)
        df = df.drop(columns=[f"ma_{p}" for p in [5, 10, 20, 60]])

        return df

    @staticmethod
    def _calc_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
