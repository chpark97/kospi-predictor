"""외국인/기관 매매동향 데이터 수집

pykrx 우선, 실패 시 빈 DataFrame 반환 (선택적 피처).
"""
import logging
import sqlite3
from datetime import datetime

import pandas as pd

from config.settings import DB_PATH, START_DATE

logger = logging.getLogger(__name__)

TABLE_NAME = "investor_trading"


class InvestorCollector:
    """외국인/기관 순매수 데이터 수집"""

    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH

    def _get_last_date(self, conn):
        try:
            result = conn.execute(f"SELECT MAX(date) FROM {TABLE_NAME}").fetchone()[0]
            return result
        except sqlite3.OperationalError:
            return None

    def collect(self, start_date=None, end_date=None):
        """외국인/기관 순매수 수집"""
        conn = sqlite3.connect(self.db_path)
        last_date = self._get_last_date(conn)

        if last_date and not start_date:
            fetch_start = last_date.replace("-", "")
        else:
            fetch_start = (start_date or START_DATE).replace("-", "")

        end_date = end_date or datetime.now().strftime("%Y%m%d")
        end_fmt = end_date.replace("-", "")

        # pykrx 시도
        df = self._try_pykrx(fetch_start, end_fmt)

        if df is None or df.empty:
            logger.warning("[Investor] pykrx 실패 -> 외국인/기관 데이터 건너뜀 (선택적 피처)")
            conn.close()
            return

        # 중복 제거
        if last_date:
            df = df[df["date"] > last_date]

        if df.empty:
            logger.info("[Investor] 새로운 데이터 없음")
            conn.close()
            return

        df.to_sql(TABLE_NAME, conn, if_exists="append", index=False)
        logger.info(f"[Investor] {len(df)}건 저장")
        conn.close()

    def _try_pykrx(self, start_date, end_date):
        """pykrx로 투자자별 매매동향 수집"""
        try:
            from pykrx import stock as pykrx_stock

            logger.info(f"[Investor/pykrx] {start_date} ~ {end_date}")

            # 코스피 전체 투자자별 순매수
            df = pykrx_stock.get_market_trading_value_by_date(
                start_date, end_date, "KOSPI"
            )

            if df.empty:
                return None

            result = pd.DataFrame({
                "date": df.index.strftime("%Y-%m-%d"),
            })

            # 컬럼명이 한글 - 외국인, 기관 순매수 추출
            for col in df.columns:
                if "외국인" in col:
                    result["foreign_net"] = df[col].values
                elif "기관" in col and "합계" in col:
                    result["inst_net"] = df[col].values

            # 컬럼을 못 찾은 경우 위치 기반 fallback
            if "foreign_net" not in result.columns:
                if len(df.columns) >= 4:
                    result["foreign_net"] = df.iloc[:, 3].values  # 외국인 위치
                else:
                    result["foreign_net"] = 0

            if "inst_net" not in result.columns:
                if len(df.columns) >= 2:
                    result["inst_net"] = df.iloc[:, 1].values  # 기관 위치
                else:
                    result["inst_net"] = 0

            return result

        except Exception as e:
            logger.error(f"[Investor/pykrx] 수집 실패: {e}")
            return None
