"""yfinance를 사용한 글로벌 지수/원자재/환율 데이터 수집"""
import logging
import sqlite3
from datetime import datetime

import pandas as pd
import yfinance as yf

from config.settings import DB_PATH, START_DATE, YAHOO_TICKERS

logger = logging.getLogger(__name__)


class YahooCollector:
    """yfinance로 글로벌 시장 데이터를 수집하여 SQLite에 저장"""

    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH

    def _get_last_date(self, conn, table_name):
        """DB에 저장된 마지막 날짜 조회"""
        try:
            query = f"SELECT MAX(date) FROM {table_name}"
            result = conn.execute(query).fetchone()[0]
            if result:
                return result
        except sqlite3.OperationalError:
            pass
        return None

    def collect(self, start_date=None, end_date=None):
        """모든 Yahoo Finance 티커 데이터 수집"""
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        conn = sqlite3.connect(self.db_path)

        for name, ticker in YAHOO_TICKERS.items():
            try:
                self._collect_ticker(conn, name, ticker, start_date, end_date)
            except Exception as e:
                logger.error(f"[Yahoo] {name}({ticker}) 수집 실패: {e}")

        conn.close()
        logger.info("[Yahoo] 전체 수집 완료")

    def _collect_ticker(self, conn, name, ticker, start_date, end_date):
        """개별 티커 데이터 수집 및 저장"""
        table_name = f"yahoo_{name}"

        # 증분 수집: DB에 데이터가 있으면 마지막 날짜 이후부터
        last_date = self._get_last_date(conn, table_name)
        if last_date and not start_date:
            fetch_start = last_date
        else:
            fetch_start = start_date or START_DATE

        logger.info(f"[Yahoo] {name}({ticker}): {fetch_start} ~ {end_date}")

        data = yf.download(ticker, start=fetch_start, end=end_date, progress=False)

        if data.empty:
            logger.warning(f"[Yahoo] {name}: 데이터 없음")
            return

        # MultiIndex 컬럼 처리 (yfinance >= 0.2.31)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        df = pd.DataFrame({
            "date": data.index.strftime("%Y-%m-%d"),
            "open": data["Open"].values,
            "high": data["High"].values,
            "low": data["Low"].values,
            "close": data["Close"].values,
            "volume": data["Volume"].values,
        })

        # 기존 데이터와 중복 제거
        if last_date:
            df = df[df["date"] > last_date]

        if df.empty:
            logger.info(f"[Yahoo] {name}: 새로운 데이터 없음")
            return

        df.to_sql(table_name, conn, if_exists="append", index=False)
        logger.info(f"[Yahoo] {name}: {len(df)}건 저장")
