"""매크로 이벤트 캘린더

FOMC, 금통위, CPI, NFP 등 주요 경제 이벤트 일정.
이벤트 전후 3일 플래그를 피처로 생성.
"""
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# 2025~2026년 FOMC 일정 (결정 발표일)
FOMC_DATES = [
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
]

# 2025~2026년 한국은행 금통위 일정
BOK_DATES = [
    "2025-01-16", "2025-02-27", "2025-04-17", "2025-05-29",
    "2025-07-10", "2025-08-28", "2025-10-16", "2025-11-27",
    "2026-01-15", "2026-02-26", "2026-04-16", "2026-05-28",
    "2026-07-09", "2026-08-27", "2026-10-15", "2026-11-26",
]

# 2025~2026년 미국 CPI 발표일 (대략)
CPI_DATES = [
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
    "2025-05-13", "2025-06-11", "2025-07-11", "2025-08-12",
    "2025-09-10", "2025-10-14", "2025-11-12", "2025-12-10",
    "2026-01-14", "2026-02-11", "2026-03-11", "2026-04-10",
    "2026-05-12", "2026-06-10", "2026-07-14", "2026-08-12",
    "2026-09-10", "2026-10-13", "2026-11-10", "2026-12-10",
]

# 2025~2026년 미국 고용지표(NFP) 발표일 (매월 첫째 금요일)
NFP_DATES = [
    "2025-01-10", "2025-02-07", "2025-03-07", "2025-04-04",
    "2025-05-02", "2025-06-06", "2025-07-03", "2025-08-01",
    "2025-09-05", "2025-10-03", "2025-11-07", "2025-12-05",
    "2026-01-09", "2026-02-06", "2026-03-06", "2026-04-03",
    "2026-05-01", "2026-06-05", "2026-07-02", "2026-08-07",
    "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04",
]

ALL_EVENTS = {
    "fomc": FOMC_DATES,
    "bok": BOK_DATES,
    "cpi": CPI_DATES,
    "nfp": NFP_DATES,
}


def get_event_flags(date_str, window=3):
    """특정 날짜의 이벤트 플래그 반환

    Args:
        date_str: "YYYY-MM-DD"
        window: 전후 N일

    Returns:
        dict: {event_name: distance} — distance=0은 당일, 음수=이전, 양수=이후
              해당 없으면 빈 dict
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    flags = {}

    for event_name, dates in ALL_EVENTS.items():
        for ed in dates:
            edt = datetime.strptime(ed, "%Y-%m-%d")
            diff = (dt - edt).days
            if -window <= diff <= window:
                flags[event_name] = diff
                break

    return flags


def get_nearest_event(date_str):
    """가장 가까운 이벤트 정보 반환"""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    nearest = None
    min_days = 999

    event_labels = {
        "fomc": "FOMC",
        "bok": "금통위",
        "cpi": "미국 CPI",
        "nfp": "미국 고용(NFP)",
    }

    for event_name, dates in ALL_EVENTS.items():
        for ed in dates:
            edt = datetime.strptime(ed, "%Y-%m-%d")
            diff = (edt - dt).days
            if 0 <= diff < min_days:
                min_days = diff
                nearest = {
                    "name": event_labels.get(event_name, event_name),
                    "date": ed,
                    "days_until": diff,
                }

    return nearest


def add_event_features(df):
    """DataFrame에 이벤트 플래그 피처 추가"""
    import numpy as np

    event_names = list(ALL_EVENTS.keys())

    # 이벤트 전후 3일 이내 플래그 (0 or 1)
    for ev in event_names:
        df[f"event_{ev}"] = 0.0

    # 이벤트까지 거리 (0=당일, 양수=N일 전)
    df["event_any_distance"] = 99.0

    for idx, row in df.iterrows():
        flags = get_event_flags(row["date"])
        for ev, dist in flags.items():
            df.at[idx, f"event_{ev}"] = 1.0
            abs_dist = abs(dist)
            if abs_dist < df.at[idx, "event_any_distance"]:
                df.at[idx, "event_any_distance"] = float(abs_dist)

    # 99 → NaN → 0 (이벤트 없는 날)
    df["event_any_distance"] = df["event_any_distance"].replace(99.0, np.nan).fillna(10.0)

    return df
