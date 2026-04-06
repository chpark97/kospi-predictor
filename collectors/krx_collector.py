"""코스피 지수 데이터 수집 (pykrx 우선, 실패 시 yfinance 대체)"""
import logging
import sqlite3
from datetime import datetime

import pandas as pd

from config.settings import DB_PATH, START_DATE

logger = logging.getLogger(__name__)

TABLE_NAME = "kospi_index"


class KRXCollector:
    """코스피 지수 OHLCV를 수집하여 SQLite에 저장

    1차: pykrx (KRX 직접)
    2차: yfinance ^KS11 (fallback)
    """

    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH

    def _get_last_date(self, conn):
        try:
            result = conn.execute(f"SELECT MAX(date) FROM {TABLE_NAME}").fetchone()[0]
            if result:
                return result
        except sqlite3.OperationalError:
            pass
        return None

    def collect(self, start_date=None, end_date=None):
        """코스피 지수 OHLCV 수집 (pykrx -> yfinance fallback)"""
        conn = sqlite3.connect(self.db_path)
        last_date = self._get_last_date(conn)

        if last_date and not start_date:
            fetch_start = last_date
        else:
            fetch_start = start_date or START_DATE

        end_date = end_date or datetime.now().strftime("%Y-%m-%d")

        # 1차: pykrx 시도
        df = self._try_pykrx(fetch_start, end_date)

        # 2차: yfinance fallback
        if df is None or df.empty:
            logger.warning("[KRX] pykrx 실패 -> yfinance ^KS11로 대체")
            df = self._try_yfinance(fetch_start, end_date)

        if df is None or df.empty:
            logger.warning("[KRX] 코스피 지수: 데이터 수집 실패")
            conn.close()
            return

        # 중복 제거
        if last_date:
            df = df[df["date"] > last_date]

        if df.empty:
            logger.info("[KRX] 코스피 지수: 새로운 데이터 없음")
            conn.close()
            return

        df.to_sql(TABLE_NAME, conn, if_exists="append", index=False)
        logger.info(f"[KRX] 코스피 지수: {len(df)}건 저장")
        conn.close()

    def _try_pykrx(self, start_date, end_date):
        """pykrx로 코스피 지수 수집 시도"""
        try:
            from pykrx import stock as pykrx_stock

            start_fmt = start_date.replace("-", "")
            end_fmt = end_date.replace("-", "")
            logger.info(f"[KRX/pykrx] 코스피 지수: {start_fmt} ~ {end_fmt}")

            df = pykrx_stock.get_index_ohlcv(start_fmt, end_fmt, "1001")

            if df.empty:
                return None

            return pd.DataFrame({
                "date": df.index.strftime("%Y-%m-%d"),
                "open": df["시가"].values,
                "high": df["고가"].values,
                "low": df["저가"].values,
                "close": df["종가"].values,
                "volume": df["거래량"].values,
            })
        except Exception as e:
            logger.error(f"[KRX/pykrx] 수집 실패: {e}")
            return None

    def _try_yfinance(self, start_date, end_date):
        """yfinance ^KS11로 코스피 지수 대체 수집"""
        try:
            import yfinance as yf

            logger.info(f"[KRX/yf] ^KS11: {start_date} ~ {end_date}")
            data = yf.download("^KS11", start=start_date, end=end_date, progress=False)

            if data.empty:
                return None

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            return pd.DataFrame({
                "date": data.index.strftime("%Y-%m-%d"),
                "open": data["Open"].values,
                "high": data["High"].values,
                "low": data["Low"].values,
                "close": data["Close"].values,
                "volume": data["Volume"].values,
            })
        except Exception as e:
            logger.error(f"[KRX/yf] 수집 실패: {e}")
            return None
