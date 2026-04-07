"""대시보드 /portfolio 엔드포인트 테스트"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from dashboard.app import app

client = TestClient(app)


def test_portfolio_returns_non_zero_capital_when_long():
    """position == 'long'일 때 capital이 0이 아닌 값을 반환하는지 검증"""
    mock_summary = {
        "capital": 105_000_000,
        "total_return": 5.0,
        "position": "매수 중 (진입 2800.0 / 손절 2744.0 / 익절 2884.0)",
        "wins": 3,
        "losses": 1,
        "win_rate": 75.0,
        "unrealized_pnl": 2_500_000,
        "mdd": -1.5,
        "sizing": 0.8,
    }

    with patch("dashboard.app.get_portfolio_summary", return_value=mock_summary):
        response = client.get("/portfolio")
        data = response.json()

        assert response.status_code == 200
        assert data["capital"] == 105_000_000
        assert data["capital"] != 0


def test_portfolio_response_includes_unrealized_pnl():
    """unrealized_pnl 필드가 응답에 포함되는지 검증"""
    mock_summary = {
        "capital": 102_000_000,
        "total_return": 2.0,
        "position": "매수 중",
        "wins": 2,
        "losses": 1,
        "win_rate": 66.7,
        "unrealized_pnl": 1_500_000,
        "mdd": -1.0,
        "sizing": 1.0,
    }

    with patch("dashboard.app.get_portfolio_summary", return_value=mock_summary):
        response = client.get("/portfolio")
        data = response.json()

        assert response.status_code == 200
        assert "unrealized_pnl" in data
        assert data["unrealized_pnl"] == 1_500_000


def test_portfolio_with_cash_position():
    """position == 'cash'일 때 정상 응답 확인"""
    mock_summary = {
        "capital": 100_000_000,
        "total_return": 0.0,
        "position": "현금 보유",
        "wins": 0,
        "losses": 0,
        "win_rate": 0,
        "unrealized_pnl": 0,
        "mdd": 0,
        "sizing": 1.0,
    }

    with patch("dashboard.app.get_portfolio_summary", return_value=mock_summary):
        response = client.get("/portfolio")
        data = response.json()

        assert response.status_code == 200
        assert data["capital"] == 100_000_000
        assert data["unrealized_pnl"] == 0
        assert data["position"] == "현금 보유"


def test_portfolio_error_handling():
    """포트폴리오 조회 실패 시 에러 응답"""
    with patch("dashboard.app.get_portfolio_summary", side_effect=Exception("파일 없음")):
        response = client.get("/portfolio")
        data = response.json()

        assert response.status_code == 200
        assert "error" in data
