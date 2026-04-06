"""한국은행 ECOS API 경제지표 수집

ECOS 오픈 API로 기준금리, CPI, 수출 등 매크로 지표 수집.
API 키: 환경변수 ECOS_API_KEY (없으면 샘플키 'sample' 사용)
"""
import logging
import os
import sqlite3
from datetime import datetime

import pandas as pd
import requests

from config.settings import DB_PATH, START_DATE

logger = logging.getLogger(__name__)

ECOS_API_KEY = os.environ.get("ECOS_API_KEY", "sample")
ECOS_BASE_URL = "https://ecos.bok.or.kr/api/StatisticSearch"

TABLE_NAME = "ecos_macro"

# 수집 대상 (통계코드, 항목코드, 피처명, 주기)
ECOS_SERIES = [
    ("722Y001", "0101000", "base_rate", "M"),       # 기준금리 (월별)
    ("901Y009", "0", "cpi_index", "M"),              # 소비자물가지수 (월별)
    ("403Y003", "0", "export_index", "M"),           # 수출금액지수 (월별)
]


class ECOSCollector:
    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH

    def _get_last_date(self, conn):
        try:
            r = conn.execute(f"SELECT MAX(date) FROM {TABLE_NAME}").fetchone()[0]
            return r
        except sqlite3.OperationalError:
            return None

    def collect(self, start_date=None, end_date=None):
        """ECOS 경제지표 수집"""
        conn = sqlite3.connect(self.db_path)
        last_date = self._get_last_date(conn)

        if last_date and not start_date:
            start_ym = last_date[:7].replace("-", "")  # YYYYMM
        else:
            start_ym = (start_date or START_DATE)[:7].replace("-", "")

        end_ym = (end_date or datetime.now().strftime("%Y-%m"))[:7].replace("-", "")

        all_data = {}

        for stat_code, item_code, feat_name, freq in ECOS_SERIES:
            try:
                values = self._fetch_series(stat_code, item_code, start_ym, end_ym, freq)
                if values:
                    all_data[feat_name] = values
                    logger.info(f"[ECOS] {feat_name}: {len(values)}건")
            except Exception as e:
                logger.warning(f"[ECOS] {feat_name} 수집 실패: {e}")

        if not all_data:
            logger.info("[ECOS] 수집 데이터 없음 (API 키 확인 필요)")
            conn.close()
            return

        # 월별 → 일별 forward fill
        df = self._monthly_to_daily(all_data)

        if last_date:
            df = df[df["date"] > last_date]

        if df.empty:
            logger.info("[ECOS] 새로운 데이터 없음")
            conn.close()
            return

        df.to_sql(TABLE_NAME, conn, if_exists="append", index=False)
        logger.info(f"[ECOS] {len(df)}건 저장")
        conn.close()

    def _fetch_series(self, stat_code, item_code, start_ym, end_ym, freq):
        """ECOS API 호출"""
        url = (
            f"{ECOS_BASE_URL}/{ECOS_API_KEY}/json/kr/1/100/"
            f"{stat_code}/{freq}/{start_ym}/{end_ym}/{item_code}"
        )

        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None

        data = resp.json()

        if "StatisticSearch" not in data:
            # API 에러 (키 미등록 등)
            if "RESULT" in data:
                logger.debug(f"[ECOS] {stat_code}: {data['RESULT'].get('MESSAGE', '')}")
            return None

        rows = data["StatisticSearch"].get("row", [])
        result = {}
        for row in rows:
            period = row.get("TIME", "")
            value = row.get("DATA_VALUE", "")
            if period and value:
                try:
                    result[period] = float(value)
                except ValueError:
                    continue

        return result

    def _monthly_to_daily(self, all_data):
        """월별 데이터를 일별로 확장 (forward fill)"""
        # 모든 시리즈의 월 키 통합
        all_months = set()
        for values in all_data.values():
            all_months.update(values.keys())

        if not all_months:
            return pd.DataFrame()

        rows = []
        for ym in sorted(all_months):
            # YYYYMM → YYYY-MM-01
            if len(ym) == 6:
                date_str = f"{ym[:4]}-{ym[4:6]}-01"
            else:
                continue

            row = {"date": date_str}
            for feat_name, values in all_data.items():
                row[feat_name] = values.get(ym)
            rows.append(row)

        df = pd.DataFrame(rows)

        # 일별로 리샘플링 (forward fill)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df = df.resample("D").ffill()
        df = df.reset_index()
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")

        return df
