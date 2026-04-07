"""FRED API를 사용한 경제지표 데이터 수집"""
import logging
import sqlite3
from datetime import datetime

import pandas as pd

from config.settings import DB_PATH, FRED_API_KEY, FRED_SERIES, START_DATE

logger = logging.getLogger(__name__)


class FREDCollector:
    """FRED API로 미국 경제지표를 수집하여 SQLite에 저장

    FRED API 키가 없는 경우 yfinance의 ^TNX(10년물 금리 ETF)로 대체합니다.
    """

    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH

    def _get_last_date(self, conn, table_name):
        try:
            result = conn.execute(f"SELECT MAX(date) FROM {table_name}").fetchone()[0]
            if result:
                return result
        except sqlite3.OperationalError:
            pass
        return None

    def collect(self, start_date=None, end_date=None):
        """FRED 데이터 수집 (API 키 없으면 yfinance 대체)"""
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        conn = sqlite3.connect(self.db_path)

        if FRED_API_KEY:
            self._collect_from_fred(conn, start_date, end_date)
        else:
            logger.warning("[FRED] API 키 미설정 -> yfinance ^TNX로 대체")
            self._collect_from_yfinance_fallback(conn, start_date, end_date)

        conn.close()
        logger.info("[FRED] 수집 완료")

    def _collect_from_fred(self, conn, start_date, end_date):
        """FRED API 직접 수집"""
        from fredapi import Fred

        fred = Fred(api_key=FRED_API_KEY)

        for name, series_id in FRED_SERIES.items():
            table_name = f"fred_{name}"
            last_date = self._get_last_date(conn, table_name)
            fetch_start = last_date if (last_date and not start_date) else (start_date or START_DATE)

            try:
                series = fred.get_series(
                    series_id,
                    observation_start=fetch_start,
                    observation_end=end_date,
                )
            except Exception as e:
                logger.error(f"[FRED] {name}({series_id}) 수집 실패: {e}")
                continue

            # RangeIndex인 경우 DatetimeIndex로 변환
            if not isinstance(series.index, pd.DatetimeIndex):
                series.index = pd.to_datetime(series.index)

            df = pd.DataFrame({
                "date": series.index.strftime("%Y-%m-%d"),
                "value": series.values,
            }).dropna()

            if last_date:
                df = df[df["date"] > last_date]

            if df.empty:
                logger.info(f"[FRED] {name}: 새로운 데이터 없음")
                continue

            df.to_sql(table_name, conn, if_exists="append", index=False)
            logger.info(f"[FRED] {name}: {len(df)}건 저장")

    def _collect_from_yfinance_fallback(self, conn, start_date, end_date):
        """FRED API 키 없을 때 yfinance로 미국 10년물 금리 대체 수집"""
        import yfinance as yf

        table_name = "fred_us10y"
        last_date = self._get_last_date(conn, table_name)
        fetch_start = last_date if (last_date and not start_date) else (start_date or START_DATE)

        logger.info(f"[FRED/yf] ^TNX: {fetch_start} ~ {end_date}")

        data = yf.download("^TNX", start=fetch_start, end=end_date, progress=False)
        if data.empty:
            logger.warning("[FRED/yf] ^TNX: 데이터 없음")
            return

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        df = pd.DataFrame({
            "date": data.index.strftime("%Y-%m-%d"),
            "value": data["Close"].values,
        }).dropna()

        if last_date:
            df = df[df["date"] > last_date]

        if df.empty:
            logger.info("[FRED/yf] ^TNX: 새로운 데이터 없음")
            return

        df.to_sql(table_name, conn, if_exists="append", index=False)
        logger.info(f"[FRED/yf] ^TNX: {len(df)}건 저장")
