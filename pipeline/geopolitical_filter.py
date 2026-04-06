"""지정학적 리스크 자동 감지 및 거래 중단

RSS 뉴스 제목에서 위험 키워드를 감지하여
위기 상황에서 자동으로 거래를 중단합니다.
"""
import logging
import xml.etree.ElementTree as ET

import requests

logger = logging.getLogger(__name__)

# 위험 키워드 (한/영)
DANGER_KEYWORDS = [
    # 한국어
    "전쟁", "침공", "제재", "서킷브레이커", "폭락", "사이드카",
    "금융위기", "긴급", "패닉", "봉쇄", "계엄", "미사일", "핵실험",
    "디폴트", "뱅크런", "대공황", "블랙먼데이", "투매",
    # 영어
    "war", "invasion", "sanction", "crisis", "crash", "panic",
    "default", "emergency", "collapse", "missile", "nuclear",
]

# 주의 레벨
CAUTION_THRESHOLD = 5    # 🟡 주의
DANGER_THRESHOLD = 15    # 🔴 위험

RSS_FEEDS = [
    "https://www.hankyung.com/feed/finance",
    "https://www.yna.co.kr/rss/economy.xml",
    "https://www.yna.co.kr/rss/stock.xml",
]


def check_geopolitical_risk():
    """지정학적 리스크 감지

    Returns:
        (risk_level, details): "safe"/"caution"/"danger", 상세 정보 dict
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    all_titles = []
    for url in RSS_FEEDS:
        try:
            resp = session.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            root = ET.fromstring(resp.content)
            for item in root.findall(".//item"):
                title_el = item.find("title")
                if title_el is not None and title_el.text:
                    t = title_el.text.strip().replace("<![CDATA[", "").replace("]]>", "")
                    if len(t) > 3:
                        all_titles.append(t)
        except Exception:
            continue

    if not all_titles:
        return "safe", {"keyword_count": 0, "keywords_found": [], "news_count": 0}

    # 위험 키워드 검색
    found_keywords = {}
    for title in all_titles:
        title_lower = title.lower()
        for kw in DANGER_KEYWORDS:
            if kw in title_lower or kw in title:
                found_keywords[kw] = found_keywords.get(kw, 0) + 1

    total_hits = sum(found_keywords.values())
    top_keywords = sorted(found_keywords.items(), key=lambda x: -x[1])[:5]

    if total_hits >= DANGER_THRESHOLD:
        level = "danger"
    elif total_hits >= CAUTION_THRESHOLD:
        level = "caution"
    else:
        level = "safe"

    details = {
        "keyword_count": total_hits,
        "keywords_found": top_keywords,
        "news_count": len(all_titles),
    }

    if level != "safe":
        kw_str = ", ".join(f"{k}({v}건)" for k, v in top_keywords)
        logger.warning(f"[GeoRisk] {level.upper()}: {total_hits}건 감지 — {kw_str}")

    return level, details


def format_geo_alert(level, details):
    """슬랙용 긴급 알림 포맷"""
    if level == "safe":
        return None

    icon = "🟡" if level == "caution" else "🚨"
    label = "주의" if level == "caution" else "긴급"
    kw_str = ", ".join(f"{k}" for k, _ in details["keywords_found"][:5])

    return (
        f"{icon} *[{label}] 지정학적 리스크 감지*\n"
        f"감지 키워드: {kw_str} (총 {details['keyword_count']}건 / 뉴스 {details['news_count']}건)\n"
        f"→ {'오늘 거래 신호 전면 중단' if level == 'danger' else '임계값 +20% 상향'}"
    )
