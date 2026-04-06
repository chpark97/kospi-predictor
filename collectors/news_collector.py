"""네이버 금융 뉴스 감성 분석 수집기

네이버 금융 증시 뉴스를 크롤링하고 한국어 감성 사전으로 점수 산출.
"""
import logging
import re
import sqlite3
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup

from config.settings import DB_PATH

logger = logging.getLogger(__name__)

TABLE_NAME = "news_sentiment"

# 한국어 금융 감성 사전
POSITIVE_WORDS = [
    "상승", "급등", "강세", "호재", "반등", "회복", "최고", "돌파", "랠리",
    "성장", "호조", "개선", "흑자", "수혜", "기대", "긍정", "상향", "매수",
    "강화", "확대", "증가", "활황", "호황", "쾌속", "사상최고", "신고가",
    "순매수", "외국인매수", "기관매수", "서프라이즈", "초과달성",
    "완화", "부양", "인하", "유입", "상한가", "수급개선",
]

NEGATIVE_WORDS = [
    "하락", "급락", "약세", "악재", "폭락", "저점", "최저", "위기", "침체",
    "손실", "적자", "부진", "리스크", "우려", "경고", "하향", "매도",
    "감소", "축소", "둔화", "불안", "공포", "패닉", "경기침체",
    "순매도", "외국인매도", "기관매도", "쇼크", "미달",
    "긴축", "인상", "유출", "하한가", "수급악화", "디폴트",
]


class NewsCollector:
    """네이버 금융 뉴스 감성 분석"""

    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
        })

    def _get_last_date(self, conn):
        try:
            result = conn.execute(f"SELECT MAX(date) FROM {TABLE_NAME}").fetchone()[0]
            return result
        except sqlite3.OperationalError:
            return None

    def collect(self, start_date=None, end_date=None, days_back=5):
        """뉴스 감성 데이터 수집

        Args:
            days_back: 최근 N일치 수집 (일일 실행 시 1~5일)
        """
        conn = sqlite3.connect(self.db_path)
        last_date = self._get_last_date(conn)

        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()

        if last_date and not start_date:
            start_dt = datetime.strptime(last_date, "%Y-%m-%d")
        elif start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start_dt = end_dt - timedelta(days=days_back)

        current = start_dt
        rows = []

        while current <= end_dt:
            date_str = current.strftime("%Y-%m-%d")

            # 이미 수집된 날짜 건너뛰기
            if last_date and date_str <= last_date:
                current += timedelta(days=1)
                continue

            score = self._get_daily_sentiment(date_str)
            if score is not None:
                rows.append({"date": date_str, "sentiment_score": score})

            current += timedelta(days=1)

        if rows:
            import pandas as pd
            df = pd.DataFrame(rows)
            df.to_sql(TABLE_NAME, conn, if_exists="append", index=False)
            logger.info(f"[News] 감성 점수 {len(rows)}건 저장")
        else:
            logger.info("[News] 새로운 감성 데이터 없음")

        conn.close()

    def _get_daily_sentiment(self, date_str):
        """특정 날짜의 네이버 금융 뉴스 감성 점수 계산"""
        try:
            date_compact = date_str.replace("-", "")
            url = (
                f"https://finance.naver.com/news/mainnews.naver"
                f"?date={date_compact}"
            )

            resp = self.session.get(url, timeout=10)
            if resp.status_code != 200:
                return None

            soup = BeautifulSoup(resp.text, "html.parser")

            # 뉴스 제목 추출
            titles = []
            for tag in soup.select("dd.articleSubject a, li.block1 a"):
                title = tag.get_text(strip=True)
                if title:
                    titles.append(title)

            if not titles:
                # 대안: 전체 텍스트에서 제목급 텍스트 추출
                for tag in soup.find_all("a"):
                    text = tag.get_text(strip=True)
                    if 5 < len(text) < 100 and any(
                        kw in text for kw in ["코스피", "증시", "주가", "지수", "시장"]
                    ):
                        titles.append(text)

            if not titles:
                return 0.0

            # 감성 점수 계산
            pos_count = 0
            neg_count = 0

            for title in titles:
                for word in POSITIVE_WORDS:
                    if word in title:
                        pos_count += 1
                for word in NEGATIVE_WORDS:
                    if word in title:
                        neg_count += 1

            total = pos_count + neg_count
            if total == 0:
                return 0.0

            # -1 (극 부정) ~ +1 (극 긍정)
            score = (pos_count - neg_count) / total
            return round(score, 4)

        except Exception as e:
            logger.debug(f"[News] {date_str} 크롤링 실패: {e}")
            return None
