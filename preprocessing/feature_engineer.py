"""전처리 및 피처 엔지니어링 v2

v1 대비 추가:
- 글로벌 지수 수익률 (1d, 5d) 및 시차(lag) 피처
- 크로스마켓 상호작용 (미국-코스피 상관계수, 스프레드)
- 추가 기술적 지표 (스토캐스틱, ATR, OBV, CCI, Williams %R)
- 캘린더 피처 (요일, 월, 월말/월초)
- VIX 파생 피처 (변화율, 레벨)
- 환율 모멘텀
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
        """전체 파이프라인: DB -> 병합 -> 피처생성 -> 시퀀스"""
        df = self._load_and_merge()
        df = self._add_technical_indicators(df)
        df = self._add_global_market_features(df)
        df = self._add_cross_market_features(df)
        df = self._add_calendar_features(df)
        df = self._add_derived_features(df)
        df = df.dropna().reset_index(drop=True)

        logger.info(f"최종 피처 수: {len(df.columns) - 1}, 행 수: {len(df)}")
        logger.info(f"기간: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")

        return df

    def prepare_sequences(self, df, scaler=None, fit_scaler=True):
        """DataFrame -> (X, y, dates, scaler) 시퀀스 변환"""
        dates = df["date"].values
        kospi_close = df["kospi_close"].values
        returns = np.diff(kospi_close) / kospi_close[:-1] * 100

        feature_cols = [c for c in df.columns if c != "date"]
        self.feature_names = feature_cols
        features = df[feature_cols].values.astype(np.float64)

        # inf/nan 정리
        features = np.where(np.isinf(features), np.nan, features)
        features = pd.DataFrame(features).ffill().bfill().values

        if fit_scaler:
            scaler = self.scaler
            features_scaled = scaler.fit_transform(features)
        else:
            features_scaled = scaler.transform(features)

        X, y, seq_dates = [], [], []
        for i in range(self.seq_length, len(features_scaled) - 1):
            X.append(features_scaled[i - self.seq_length:i])
            y.append(returns[i])
            seq_dates.append(dates[i + 1])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        logger.info(f"시퀀스 생성: X={X.shape}, y={y.shape}")
        return X, y, seq_dates, scaler

    # ── 데이터 로드 ──

    def _load_and_merge(self):
        """DB의 모든 테이블을 날짜 기준으로 병합"""
        conn = sqlite3.connect(self.db_path)

        kospi = pd.read_sql("SELECT * FROM kospi_index", conn)
        kospi = kospi.rename(columns={
            "open": "kospi_open", "high": "kospi_high",
            "low": "kospi_low", "close": "kospi_close",
            "volume": "kospi_volume",
        })
        merged = kospi.copy()

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

        try:
            fred = pd.read_sql("SELECT date, value as us10y FROM fred_us10y", conn)
            merged = merged.merge(fred, on="date", how="left")
        except Exception as e:
            logger.warning(f"fred_us10y 병합 실패: {e}")

        conn.close()

        merged = merged.sort_values("date").reset_index(drop=True)
        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        merged[numeric_cols] = merged[numeric_cols].ffill().bfill()

        logger.info(f"병합 완료: {merged.shape}")
        return merged

    # ── 코스피 기술적 지표 ──

    def _add_technical_indicators(self, df):
        close = df["kospi_close"]
        high = df["kospi_high"]
        low = df["kospi_low"]
        volume = df["kospi_volume"]

        # RSI (7일, 14일)
        df["rsi_7"] = self._calc_rsi(close, 7)
        df["rsi_14"] = self._calc_rsi(close, 14)

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # 볼린저 밴드
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df["bb_width"] = (4 * std20) / sma20
        df["bb_position"] = (close - (sma20 - 2 * std20)) / (4 * std20)

        # 스토캐스틱 (14, 3)
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        df["stoch_k"] = (close - low14) / (high14 - low14) * 100
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # ATR (Average True Range, 14일)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean() / close * 100  # % 단위

        # CCI (Commodity Channel Index, 20일)
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        df["cci_20"] = (typical_price - sma_tp) / (0.015 * mad)

        # Williams %R (14일)
        df["williams_r"] = (high14 - close) / (high14 - low14) * -100

        # OBV (On-Balance Volume) 변화율
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        df["obv_change"] = obv.pct_change(5)

        # 캔들스틱 패턴 피처
        df["candle_body"] = (close - df["kospi_open"]) / df["kospi_open"] * 100
        df["upper_shadow"] = (high - pd.concat([close, df["kospi_open"]], axis=1).max(axis=1)) / close * 100
        df["lower_shadow"] = (pd.concat([close, df["kospi_open"]], axis=1).min(axis=1) - low) / close * 100

        return df

    # ── 글로벌 시장 수익률/시차 피처 ──

    def _add_global_market_features(self, df):
        """글로벌 지수의 수익률 및 lag 피처"""
        global_cols = ["sp500", "nasdaq", "dow", "nikkei", "hangseng", "shanghai",
                       "wti", "copper", "usdkrw"]

        for col in global_cols:
            if col not in df.columns:
                continue

            # 1일, 5일 수익률
            df[f"{col}_ret1d"] = df[col].pct_change(1) * 100
            df[f"{col}_ret5d"] = df[col].pct_change(5) * 100

            # 1일 lag (전일 미국 시장 → 당일 코스피 영향)
            df[f"{col}_ret1d_lag1"] = df[f"{col}_ret1d"].shift(1)

        # VIX 파생 피처
        if "vix" in df.columns:
            df["vix_ret1d"] = df["vix"].pct_change(1) * 100
            df["vix_ret5d"] = df["vix"].pct_change(5) * 100
            df["vix_ma5_ratio"] = df["vix"] / df["vix"].rolling(5).mean()
            df["vix_level"] = pd.cut(df["vix"], bins=[0, 15, 20, 25, 30, 100],
                                     labels=[0, 1, 2, 3, 4]).astype(float)

        # DXY 파생
        if "dxy" in df.columns:
            df["dxy_ret1d"] = df["dxy"].pct_change(1) * 100
            df["dxy_ret1d_lag1"] = df["dxy_ret1d"].shift(1)

        # 미국 10년물 금리 파생
        if "us10y" in df.columns:
            df["us10y_change"] = df["us10y"].diff()
            df["us10y_change_lag1"] = df["us10y_change"].shift(1)

        return df

    # ── 크로스마켓 상호작용 피처 ──

    def _add_cross_market_features(self, df):
        """시장 간 상관관계, 스프레드, 레짐 피처"""
        kospi_ret = df["kospi_close"].pct_change()

        # S&P500-코스피 롤링 상관계수 (20일)
        if "sp500" in df.columns:
            sp_ret = df["sp500"].pct_change()
            df["corr_sp500_20d"] = kospi_ret.rolling(20).corr(sp_ret)
            df["corr_sp500_60d"] = kospi_ret.rolling(60).corr(sp_ret)

        # 나스닥-코스피 상관계수
        if "nasdaq" in df.columns:
            nq_ret = df["nasdaq"].pct_change()
            df["corr_nasdaq_20d"] = kospi_ret.rolling(20).corr(nq_ret)

        # USD/KRW-코스피 역상관 (환율 상승 = 코스피 약세 경향)
        if "usdkrw" in df.columns:
            krw_ret = df["usdkrw"].pct_change()
            df["corr_usdkrw_20d"] = kospi_ret.rolling(20).corr(krw_ret)

        # 미국 vs 아시아 모멘텀 스프레드
        if all(c in df.columns for c in ["sp500", "nikkei"]):
            sp_mom = df["sp500"].pct_change(20)
            nk_mom = df["nikkei"].pct_change(20)
            df["us_asia_momentum_spread"] = (sp_mom - nk_mom) * 100

        # 위험선호 지표: 구리/금 비율 대용 (구리 모멘텀)
        if "copper" in df.columns:
            df["copper_momentum_20d"] = df["copper"].pct_change(20) * 100

        # 유가-코스피 상관 (에너지 의존도)
        if "wti" in df.columns:
            wti_ret = df["wti"].pct_change()
            df["corr_wti_20d"] = kospi_ret.rolling(20).corr(wti_ret)

        return df

    # ── 캘린더 피처 ──

    def _add_calendar_features(self, df):
        """요일, 월, 월말효과 등 캘린더 기반 피처"""
        dates = pd.to_datetime(df["date"])

        # 요일 (0=월~4=금) → sin/cos 인코딩
        dow = dates.dt.dayofweek
        df["dow_sin"] = np.sin(2 * np.pi * dow / 5)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 5)

        # 월 → sin/cos 인코딩
        month = dates.dt.month
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)

        # 월초/월말 효과 (영업일 기준 근사)
        df["is_month_start"] = (dates.dt.day <= 3).astype(float)
        df["is_month_end"] = (dates.dt.day >= 27).astype(float)

        # 분기말
        df["is_quarter_end"] = ((dates.dt.month % 3 == 0) & (dates.dt.day >= 25)).astype(float)

        return df

    # ── 파생 피처 ──

    def _add_derived_features(self, df):
        close = df["kospi_close"]

        # 이동평균 비율 (5, 10, 20, 60, 120일)
        for period in [5, 10, 20, 60, 120]:
            ma = close.rolling(period).mean()
            df[f"ma_{period}_ratio"] = close / ma

        # 골든크로스/데드크로스 시그널
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        df["ma5_above_ma20"] = (ma5 > ma20).astype(float)
        df["ma20_above_ma60"] = (ma20 > ma60).astype(float)

        # 수익률 (1, 2, 3, 5, 10, 20일)
        for period in [1, 2, 3, 5, 10, 20]:
            df[f"return_{period}d"] = close.pct_change(period) * 100

        # 연속 상승/하락 카운트
        daily_ret = close.pct_change()
        up = (daily_ret > 0).astype(int)
        down = (daily_ret < 0).astype(int)
        # 연속 상승일수
        df["consec_up"] = up * (up.groupby((up != up.shift()).cumsum()).cumcount() + 1)
        df["consec_down"] = down * (down.groupby((down != down.shift()).cumsum()).cumcount() + 1)

        # 거래량 피처
        df["volume_change"] = df["kospi_volume"].pct_change()
        df["volume_ma5_ratio"] = df["kospi_volume"] / df["kospi_volume"].rolling(5).mean()
        df["volume_ma20_ratio"] = df["kospi_volume"] / df["kospi_volume"].rolling(20).mean()

        # 변동성 (10, 20, 60일)
        daily_pct = close.pct_change()
        for period in [10, 20, 60]:
            df[f"volatility_{period}d"] = daily_pct.rolling(period).std() * np.sqrt(252) * 100

        # 변동성 비율 (단기/장기) - 변동성 레짐 전환 감지
        df["vol_ratio_10_60"] = df["volatility_10d"] / df["volatility_60d"]

        # 고저 비율 (당일 변동 범위)
        df["hl_ratio"] = (df["kospi_high"] - df["kospi_low"]) / close * 100

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
