"""한국 시장 특화 데이터 수집

삼성전자, SOX 반도체 지수, 코스피200 선물 등.
"""
import logging
import sqlite3
from datetime import datetime

import pandas as pd
import yfinance as yf

from config.settings import DB_PATH, START_DATE

logger = logging.getLogger(__name__)

KOREA_TICKERS = {
    "samsung": "005930.KS",   # 삼성전자
    "sox": "^SOX",            # 필라델피아 반도체 지수
    "kosdaq": "^KQ11",        # 코스닥 지수
}


class KoreaSpecificCollector:
    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH

    def _get_last_date(self, conn, table):
        try:
            r = conn.execute(f"SELECT MAX(date) FROM {table}").fetchone()[0]
            return r
        except sqlite3.OperationalError:
            return None

    def collect(self, start_date=None, end_date=None):
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        conn = sqlite3.connect(self.db_path)

        for name, ticker in KOREA_TICKERS.items():
            try:
                self._collect_one(conn, name, ticker, start_date, end_date)
            except Exception as e:
                logger.warning(f"[Korea] {name}({ticker}) 수집 실패: {e}")

        conn.close()

    def _collect_one(self, conn, name, ticker, start_date, end_date):
        table = f"korea_{name}"
        last = self._get_last_date(conn, table)
        fetch_start = last if (last and not start_date) else (start_date or START_DATE)

        data = yf.download(ticker, start=fetch_start, end=end_date, progress=False)
        if data.empty:
            return

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        df = pd.DataFrame({
            "date": data.index.strftime("%Y-%m-%d"),
            "close": data["Close"].values,
            "volume": data["Volume"].values,
        })

        if last:
            df = df[df["date"] > last]
        if df.empty:
            return

        df.to_sql(table, conn, if_exists="append", index=False)
        logger.info(f"[Korea] {name}: {len(df)}건 저장")
