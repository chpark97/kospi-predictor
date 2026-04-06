"""뉴스 감성 분석 수집기 (RSS 기반)

다중 RSS 소스에서 금융 뉴스 제목을 수집하고
한국어 감성 사전으로 점수를 산출합니다.
"""
import logging
import sqlite3
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup

from config.settings import DB_PATH

logger = logging.getLogger(__name__)

TABLE_NAME = "news_sentiment"

# RSS 피드 소스
RSS_FEEDS = [
    ("hankyung", "https://www.hankyung.com/feed/finance"),
    ("yna_economy", "https://www.yna.co.kr/rss/economy.xml"),
    ("yna_stock", "https://www.yna.co.kr/rss/stock.xml"),
]

# 네이버 금융 뉴스 URL (RSS 불가 시 fallback)
NAVER_FINANCE_URL = "https://finance.naver.com/news/mainnews.naver"

# 한국어 금융 감성 사전
POSITIVE_WORDS = [
    "상승", "급등", "강세", "호재", "반등", "회복", "최고", "돌파", "랠리",
    "성장", "호조", "개선", "흑자", "수혜", "기대", "긍정", "상향", "매수",
    "강화", "확대", "증가", "활황", "호황", "사상최고", "신고가",
    "순매수", "외국인매수", "기관매수", "서프라이즈", "초과달성",
    "완화", "부양", "인하", "유입", "상한가", "수급개선",
    "낙관", "동반상승", "급반등", "저가매수", "강보합", "최고치",
]

NEGATIVE_WORDS = [
    "하락", "급락", "약세", "악재", "폭락", "저점", "최저", "위기", "침체",
    "손실", "적자", "부진", "리스크", "우려", "경고", "하향", "매도",
    "감소", "축소", "둔화", "불안", "공포", "패닉", "경기침체",
    "순매도", "외국인매도", "기관매도", "쇼크", "미달",
    "긴축", "인상", "유출", "하한가", "수급악화", "디폴트",
    "비관", "동반하락", "급전직하", "투매", "약보합", "최저치",
]


class NewsCollector:
    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        })

    def _get_last_date(self, conn):
        try:
            return conn.execute(f"SELECT MAX(date) FROM {TABLE_NAME}").fetchone()[0]
        except sqlite3.OperationalError:
            return None

    def collect(self, start_date=None, end_date=None, days_back=5):
        """뉴스 감성 수집 — RSS 우선, fallback 네이버"""
        conn = sqlite3.connect(self.db_path)
        last_date = self._get_last_date(conn)
        today_str = datetime.now().strftime("%Y-%m-%d")

        # 오늘 이미 수집했으면 스킵
        if last_date and last_date >= today_str:
            logger.info("[News] 오늘 이미 수집됨")
            conn.close()
            return

        # RSS에서 오늘 감성 점수 수집
        titles = self._collect_rss_titles()

        if not titles:
            # fallback: 네이버 금융
            titles = self._collect_naver_titles()

        if not titles:
            logger.info("[News] 뉴스 수집 실패 — 스킵")
            conn.close()
            return

        score = self._compute_sentiment(titles)

        import pandas as pd
        df = pd.DataFrame([{"date": today_str, "sentiment_score": score}])

        # 기존 오늘 데이터 덮어쓰기
        try:
            conn.execute(f"DELETE FROM {TABLE_NAME} WHERE date = ?", (today_str,))
        except sqlite3.OperationalError:
            pass

        df.to_sql(TABLE_NAME, conn, if_exists="append", index=False)
        logger.info(f"[News] 감성 점수: {score:+.3f} (뉴스 {len(titles)}건)")
        conn.close()

    def _collect_rss_titles(self):
        """RSS 피드에서 뉴스 제목 수집"""
        all_titles = []

        for name, url in RSS_FEEDS:
            try:
                resp = self.session.get(url, timeout=10)
                if resp.status_code != 200:
                    continue

                # XML 파싱
                root = ET.fromstring(resp.content)

                # RSS 2.0 형식
                for item in root.findall(".//item"):
                    title_el = item.find("title")
                    if title_el is not None and title_el.text:
                        title = title_el.text.strip()
                        # CDATA 처리
                        title = title.replace("<![CDATA[", "").replace("]]>", "")
                        if len(title) > 5:
                            all_titles.append(title)

                # Atom 형식 fallback
                if not all_titles:
                    ns = {"atom": "http://www.w3.org/2005/Atom"}
                    for entry in root.findall(".//atom:entry", ns):
                        title_el = entry.find("atom:title", ns)
                        if title_el is not None and title_el.text:
                            all_titles.append(title_el.text.strip())

                if all_titles:
                    logger.info(f"[News/{name}] {len(all_titles)}건 수집")

            except Exception as e:
                logger.debug(f"[News/{name}] RSS 실패: {e}")
                continue

        return all_titles

    def _collect_naver_titles(self):
        """네이버 금융 뉴스 제목 수집 (fallback)"""
        try:
            resp = self.session.get(NAVER_FINANCE_URL, timeout=10)
            if resp.status_code != 200:
                return []

            soup = BeautifulSoup(resp.text, "html.parser")
            titles = []

            for tag in soup.select("dd.articleSubject a, li.block1 a"):
                t = tag.get_text(strip=True)
                if t and len(t) > 5:
                    titles.append(t)

            if not titles:
                for tag in soup.find_all("a"):
                    t = tag.get_text(strip=True)
                    if 5 < len(t) < 100 and any(
                        kw in t for kw in ["코스피", "증시", "주가", "지수", "시장", "코스닥"]
                    ):
                        titles.append(t)

            if titles:
                logger.info(f"[News/naver] {len(titles)}건 수집")
            return titles

        except Exception as e:
            logger.debug(f"[News/naver] 실패: {e}")
            return []

    def _compute_sentiment(self, titles):
        """뉴스 제목 목록에서 감성 점수 산출"""
        pos = 0
        neg = 0

        for title in titles:
            for w in POSITIVE_WORDS:
                if w in title:
                    pos += 1
            for w in NEGATIVE_WORDS:
                if w in title:
                    neg += 1

        total = pos + neg
        if total == 0:
            return 0.0

        score = (pos - neg) / total
        return round(score, 4)
