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
        df = self._add_investor_features(df)
        df = self._add_sentiment_features(df)
        df = self._add_korea_specific_features(df)
        df = self._add_event_features(df)
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

        # 외국인/기관 매매동향 (선택적)
        try:
            inv = pd.read_sql("SELECT date, foreign_net, inst_net FROM investor_trading", conn)
            merged = merged.merge(inv, on="date", how="left")
        except Exception as e:
            logger.info(f"investor_trading 없음 (선택적): {e}")
            merged["foreign_net"] = 0.0
            merged["inst_net"] = 0.0

        # 뉴스 감성 점수 (선택적)
        try:
            news = pd.read_sql("SELECT date, sentiment_score FROM news_sentiment", conn)
            merged = merged.merge(news, on="date", how="left")
        except Exception as e:
            logger.info(f"news_sentiment 없음 (선택적): {e}")
            merged["sentiment_score"] = 0.0

        # 한국 특화: 삼성전자, SOX, 코스닥 (선택적)
        for name in ["samsung", "sox", "kosdaq"]:
            try:
                kdf = pd.read_sql(f"SELECT date, close as {name} FROM korea_{name}", conn)
                merged = merged.merge(kdf, on="date", how="left")
            except Exception:
                merged[name] = np.nan

        # ECOS 매크로 지표 (선택적)
        try:
            ecos = pd.read_sql("SELECT * FROM ecos_macro", conn)
            merged = merged.merge(ecos, on="date", how="left")
        except Exception as e:
            logger.info(f"ecos_macro 없음 (선택적): {e}")
            for col in ["base_rate", "cpi_index", "export_index"]:
                merged[col] = 0.0

        conn.close()

        merged = merged.sort_values("date").reset_index(drop=True)
        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        merged[numeric_cols] = merged[numeric_cols].ffill().bfill()
        # 여전히 NaN인 컬럼은 0으로 채움 (선택적 데이터원)
        merged[numeric_cols] = merged[numeric_cols].fillna(0)

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

        # 캔들스틱 패턴 피처는 _add_derived_features()에서 shift(1) 적용하여 생성

        return df

    # ── 글로벌 시장 수익률/시차 피처 ──

    def _add_global_market_features(self, df):
        """글로벌 지수의 수익률 및 lag 피처

        주의: 미국/유럽 지수는 한국 장마감 후에 거래되므로,
        당일 수익률(ret1d)을 당일 피처로 쓰면 미래 데이터 누수.
        모든 글로벌 수익률에 shift(1)을 적용하여 전일까지만 사용.
        """
        global_cols = ["sp500", "nasdaq", "dow", "nikkei", "hangseng", "shanghai",
                       "wti", "copper", "usdkrw"]

        for col in global_cols:
            if col not in df.columns:
                continue

            ret1d = df[col].pct_change(1) * 100
            ret5d = df[col].pct_change(5) * 100

            # 모든 수익률에 shift(1) 적용 — 전일 데이터만 사용
            df[f"{col}_ret1d"] = ret1d.shift(1)
            df[f"{col}_ret5d"] = ret5d.shift(1)
            # lag2도 추가 (이틀 전)
            df[f"{col}_ret1d_lag2"] = ret1d.shift(2)

        # VIX 파생 피처 (VIX도 미국 시장이므로 shift 필요)
        if "vix" in df.columns:
            df["vix_ret1d"] = df["vix"].pct_change(1).shift(1) * 100
            df["vix_ret5d"] = df["vix"].pct_change(5).shift(1) * 100
            df["vix_ma5_ratio"] = (df["vix"] / df["vix"].rolling(5).mean()).shift(1)
            df["vix_level"] = pd.cut(df["vix"].shift(1), bins=[0, 15, 20, 25, 30, 100],
                                     labels=[0, 1, 2, 3, 4]).astype(float)

        # DXY 파생 (미국)
        if "dxy" in df.columns:
            df["dxy_ret1d"] = df["dxy"].pct_change(1).shift(1) * 100
            df["dxy_ret1d_lag2"] = df["dxy"].pct_change(1).shift(2) * 100

        # 미국 10년물 금리 파생
        if "us10y" in df.columns:
            df["us10y_change"] = df["us10y"].diff().shift(1)
            df["us10y_change_lag2"] = df["us10y"].diff().shift(2)

        return df

    # ── 크로스마켓 상호작용 피처 ──

    def _add_cross_market_features(self, df):
        """시장 간 상관관계, 스프레드, 레짐 피처

        상관계수는 전일까지의 윈도우를 사용(shift(1))하여 누수 방지.
        """
        kospi_ret = df["kospi_close"].pct_change()

        if "sp500" in df.columns:
            sp_ret = df["sp500"].pct_change()
            df["corr_sp500_20d"] = kospi_ret.rolling(20).corr(sp_ret).shift(1)
            df["corr_sp500_60d"] = kospi_ret.rolling(60).corr(sp_ret).shift(1)

        if "nasdaq" in df.columns:
            nq_ret = df["nasdaq"].pct_change()
            df["corr_nasdaq_20d"] = kospi_ret.rolling(20).corr(nq_ret).shift(1)

        if "usdkrw" in df.columns:
            krw_ret = df["usdkrw"].pct_change()
            df["corr_usdkrw_20d"] = kospi_ret.rolling(20).corr(krw_ret).shift(1)

        if all(c in df.columns for c in ["sp500", "nikkei"]):
            sp_mom = df["sp500"].pct_change(20)
            nk_mom = df["nikkei"].pct_change(20)
            df["us_asia_momentum_spread"] = ((sp_mom - nk_mom) * 100).shift(1)

        if "copper" in df.columns:
            df["copper_momentum_20d"] = (df["copper"].pct_change(20) * 100).shift(1)

        if "wti" in df.columns:
            wti_ret = df["wti"].pct_change()
            df["corr_wti_20d"] = kospi_ret.rolling(20).corr(wti_ret).shift(1)

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

    # ── 외국인/기관 매매동향 피처 ──

    def _add_investor_features(self, df):
        """외국인/기관 순매수 파생 피처"""
        for col in ["foreign_net", "inst_net"]:
            if col not in df.columns or df[col].abs().sum() == 0:
                continue

            # 5일, 20일 이동평균
            df[f"{col}_ma5"] = df[col].rolling(5).mean()
            df[f"{col}_ma20"] = df[col].rolling(20).mean()

            # 누적 순매수 방향 (5일)
            df[f"{col}_cum5"] = df[col].rolling(5).sum()
            df[f"{col}_direction"] = (df[f"{col}_cum5"] > 0).astype(float)

            # 연속 순매수/순매도 일수
            is_buy = (df[col] > 0).astype(int)
            df[f"{col}_consec_buy"] = is_buy * (
                is_buy.groupby((is_buy != is_buy.shift()).cumsum()).cumcount() + 1
            )

        return df

    # ── 뉴스 감성 피처 ──

    def _add_sentiment_features(self, df):
        """뉴스 감성 점수 파생 피처"""
        if "sentiment_score" not in df.columns:
            return df

        sent = df["sentiment_score"]

        # 3일, 5일 이동평균
        df["sentiment_ma3"] = sent.rolling(3).mean()
        df["sentiment_ma5"] = sent.rolling(5).mean()

        # 감성 변화율
        df["sentiment_change"] = sent.diff()

        # 감성 레벨 (강한 부정/부정/중립/긍정/강한 긍정)
        df["sentiment_level"] = pd.cut(
            sent, bins=[-1.1, -0.5, -0.1, 0.1, 0.5, 1.1],
            labels=[-2, -1, 0, 1, 2]
        ).astype(float)

        return df

    # ── 한국 특화 피처 ──

    def _add_korea_specific_features(self, df):
        close = df["kospi_close"]

        # 삼성전자 수익률 + 상대강도 (shift 적용)
        if "samsung" in df.columns and df["samsung"].notna().sum() > 20:
            df["samsung_ret1d"] = df["samsung"].pct_change(1).shift(1) * 100
            df["samsung_ret5d"] = df["samsung"].pct_change(5).shift(1) * 100
            # 삼성전자 vs 코스피 상대강도 (RS)
            sam_ret = df["samsung"].pct_change(20)
            kos_ret = close.pct_change(20)
            df["samsung_relative_strength"] = ((1 + sam_ret) / (1 + kos_ret) - 1).shift(1) * 100

        # SOX 반도체 수익률 + 모멘텀
        if "sox" in df.columns and df["sox"].notna().sum() > 20:
            df["sox_ret1d"] = df["sox"].pct_change(1).shift(1) * 100
            df["sox_momentum_5d"] = df["sox"].pct_change(5).shift(1) * 100

        # 코스닥 수익률 + 코스피/코스닥 비율
        if "kosdaq" in df.columns and df["kosdaq"].notna().sum() > 20:
            df["kosdaq_ret1d"] = df["kosdaq"].pct_change(1).shift(1) * 100
            df["kospi_kosdaq_ratio"] = (close / df["kosdaq"]).shift(1)

        # ECOS 매크로 파생 피처 (실제 데이터 있을 때만)
        if "base_rate" in df.columns and df["base_rate"].replace(0, np.nan).notna().sum() > 5:
            df["base_rate_change"] = df["base_rate"].diff().fillna(0)
            df["base_rate_level"] = df["base_rate"]

        if "cpi_index" in df.columns and df["cpi_index"].replace(0, np.nan).notna().sum() > 5:
            df["cpi_mom"] = df["cpi_index"].pct_change().fillna(0) * 100

        if "export_index" in df.columns and df["export_index"].replace(0, np.nan).notna().sum() > 5:
            df["export_growth"] = df["export_index"].pct_change().fillna(0) * 100

        return df

    # ── 이벤트 피처 ──

    def _add_event_features(self, df):
        try:
            from collectors.event_calendar import add_event_features
            df = add_event_features(df)
        except Exception as e:
            logger.warning(f"이벤트 피처 추가 실패: {e}")
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

        # 수익률 — return_1d/2d/3d 제거 (타겟과 1~3일 차이 → 누수)
        # return_5d/10d/20d만 유지하되 shift(1) 적용 (전일 기준)
        for period in [5, 10, 20]:
            df[f"return_{period}d"] = (close.pct_change(period) * 100).shift(1)

        # 연속 상승/하락 카운트 — shift(1) 적용 (당일 종가 방향 누수 방지)
        daily_ret = close.pct_change()
        up = (daily_ret > 0).astype(int)
        down = (daily_ret < 0).astype(int)
        df["consec_up"] = (up * (up.groupby((up != up.shift()).cumsum()).cumcount() + 1)).shift(1)
        df["consec_down"] = (down * (down.groupby((down != down.shift()).cumsum()).cumcount() + 1)).shift(1)

        # 거래량 피처 — shift(1) 적용 (당일 거래량은 장중 데이터)
        df["volume_change"] = df["kospi_volume"].pct_change().shift(1)
        df["volume_ma5_ratio"] = (df["kospi_volume"] / df["kospi_volume"].rolling(5).mean()).shift(1)
        df["volume_ma20_ratio"] = (df["kospi_volume"] / df["kospi_volume"].rolling(20).mean()).shift(1)

        # 변동성 (10, 20, 60일)
        daily_pct = close.pct_change()
        for period in [10, 20, 60]:
            df[f"volatility_{period}d"] = daily_pct.rolling(period).std() * np.sqrt(252) * 100

        # 변동성 비율 (단기/장기) - 변동성 레짐 전환 감지
        df["vol_ratio_10_60"] = df["volatility_10d"] / df["volatility_60d"]

        # 고저 비율 — shift(1) 적용 (당일 OHLC는 장중 데이터)
        df["hl_ratio"] = ((df["kospi_high"] - df["kospi_low"]) / close * 100).shift(1)

        # 캔들스틱 패턴 피처 — shift(1) 적용 (당일 OHLC는 장중)
        df["candle_body"] = ((close - df["kospi_open"]) / df["kospi_open"] * 100).shift(1)
        df["upper_shadow"] = ((df["kospi_high"] - pd.concat([close, df["kospi_open"]], axis=1).max(axis=1)) / close * 100).shift(1)
        df["lower_shadow"] = ((pd.concat([close, df["kospi_open"]], axis=1).min(axis=1) - df["kospi_low"]) / close * 100).shift(1)

        # ── 멀티 타임프레임 피처 ──
        # 52주 고/저점 대비 위치
        high_252 = close.rolling(252).max()
        low_252 = close.rolling(252).min()
        df["pct_from_52w_high"] = (close / high_252 - 1) * 100
        df["pct_from_52w_low"] = (close / low_252 - 1) * 100

        # 주간 이동평균 추세 방향 (5주=25일)
        ma25 = close.rolling(25).mean()
        df["weekly_ma_trend"] = (ma25 - ma25.shift(5)) / ma25.shift(5) * 100

        # 월간 vs 일간 변동성 비율
        vol_daily = daily_pct.rolling(5).std() * np.sqrt(252) * 100
        vol_monthly = daily_pct.rolling(60).std() * np.sqrt(252) * 100
        df["vol_daily_monthly_ratio"] = vol_daily / vol_monthly.replace(0, np.nan)

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
