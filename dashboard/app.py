"""코스피 예측 웹 대시보드 (FastAPI)

실행: python dashboard/app.py → http://localhost:8000
"""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from config.settings import DB_PATH

app = FastAPI(title="코스피 예측 대시보드")

DATA_DIR = Path(__file__).parent.parent / "data"
HISTORY_PATH = DATA_DIR / "prediction_history.json"
PORTFOLIO_PATH = Path(__file__).parent.parent / "portfolio" / "portfolio.json"


def _load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@app.get("/predict")
def get_prediction():
    """오늘 예측 결과"""
    history = _load_json(HISTORY_PATH) or []
    if history:
        return history[-1]
    return {"error": "예측 이력 없음"}


@app.get("/history")
def get_history():
    """최근 30일 예측 히스토리"""
    history = _load_json(HISTORY_PATH) or []
    return history[-30:]


@app.get("/performance")
def get_performance():
    """방향정확도, 누적수익률, 샤프비율"""
    import numpy as np
    history = _load_json(HISTORY_PATH) or []
    verified = [h for h in history if h.get("actual_direction")]

    if not verified:
        return {"error": "검증 데이터 없음"}

    correct = sum(1 for h in verified if h["predicted_direction"] == h["actual_direction"])
    da = correct / len(verified) * 100

    rets = []
    for h in verified:
        ar = h.get("actual_return", 0) or 0
        if h["predicted_direction"] == "up" and h.get("signal_valid", True):
            rets.append(ar / 100 - 0.0003)
        else:
            rets.append(0)

    rets = np.array(rets)
    cum_ret = float((np.prod(1 + rets) - 1) * 100)
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0

    return {
        "direction_accuracy": round(da, 1),
        "cumulative_return": round(cum_ret, 2),
        "sharpe_ratio": round(sharpe, 2),
        "total_predictions": len(verified),
    }


@app.get("/portfolio")
def get_portfolio():
    """가상 포트폴리오 현황"""
    pf = _load_json(PORTFOLIO_PATH)
    if not pf:
        return {"error": "포트폴리오 없음"}

    total_ret = (pf["capital"] / pf["initial_capital"] - 1) * 100
    total_trades = pf["wins"] + pf["losses"]
    return {
        "capital": pf["capital"],
        "initial_capital": pf["initial_capital"],
        "total_return": round(total_ret, 2),
        "position": pf["position"],
        "wins": pf["wins"],
        "losses": pf["losses"],
        "win_rate": round(pf["wins"] / total_trades * 100, 1) if total_trades > 0 else 0,
    }


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """HTML 대시보드"""
    return """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>코스피 예측 대시보드</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
  h1 { color: #1a237e; }
  .card { background: white; border-radius: 12px; padding: 20px; margin: 16px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
  .stat { display: inline-block; text-align: center; padding: 10px 24px; }
  .stat .value { font-size: 28px; font-weight: bold; color: #1a237e; }
  .stat .label { font-size: 13px; color: #666; margin-top: 4px; }
  .up { color: #e53935; } .down { color: #1565c0; }
  table { width: 100%; border-collapse: collapse; margin-top: 12px; }
  th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }
  th { background: #f5f5f5; font-weight: 600; }
</style>
</head><body>
<h1>📊 코스피 예측 대시보드</h1>

<div class="card" id="predict-card"><h3>오늘 예측</h3><p>로딩 중...</p></div>
<div class="card"><h3>성과 지표</h3><div id="perf"></div></div>
<div class="card"><h3>포트폴리오</h3><div id="pf"></div></div>
<div class="card"><h3>누적 수익률</h3><canvas id="chart" height="200"></canvas></div>
<div class="card"><h3>최근 예측 히스토리</h3><div id="hist"></div></div>

<script>
async function load() {
  // 예측
  const pred = await (await fetch('/predict')).json();
  const dir = pred.predicted_direction === 'up' ? '▲ 상승' : '▼ 하락';
  const cls = pred.predicted_direction === 'up' ? 'up' : 'down';
  document.getElementById('predict-card').innerHTML = `<h3>오늘 예측 (${pred.date || 'N/A'})</h3>
    <div class="stat"><div class="value ${cls}">${dir}</div><div class="label">방향</div></div>
    <div class="stat"><div class="value">${pred.predicted_return > 0 ? '+' : ''}${(pred.predicted_return||0).toFixed(2)}%</div><div class="label">예측 등락률</div></div>
    <div class="stat"><div class="value">${(pred.confidence||0).toFixed(1)}%</div><div class="label">신뢰도</div></div>
    <div class="stat"><div class="value">${pred.signal_valid ? '✅' : '⚠️'}</div><div class="label">신호</div></div>`;

  // 성과
  const perf = await (await fetch('/performance')).json();
  document.getElementById('perf').innerHTML = `
    <div class="stat"><div class="value">${(perf.direction_accuracy||0).toFixed(1)}%</div><div class="label">방향정확도</div></div>
    <div class="stat"><div class="value">${(perf.cumulative_return||0).toFixed(2)}%</div><div class="label">누적수익률</div></div>
    <div class="stat"><div class="value">${(perf.sharpe_ratio||0).toFixed(2)}</div><div class="label">샤프비율</div></div>
    <div class="stat"><div class="value">${perf.total_predictions||0}</div><div class="label">예측 수</div></div>`;

  // 포트폴리오
  const pf = await (await fetch('/portfolio')).json();
  const ret = pf.total_return || 0;
  document.getElementById('pf').innerHTML = `
    <div class="stat"><div class="value">${(pf.capital||0).toLocaleString()}원</div><div class="label">잔고</div></div>
    <div class="stat"><div class="value ${ret>=0?'up':'down'}">${ret>=0?'+':''}${ret.toFixed(2)}%</div><div class="label">수익률</div></div>
    <div class="stat"><div class="value">${pf.wins||0}승 ${pf.losses||0}패</div><div class="label">전적</div></div>`;

  // 히스토리 + 차트
  const hist = await (await fetch('/history')).json();
  const verified = hist.filter(h => h.actual_return !== null);

  let cumRet = 0, chartData = [], labels = [];
  verified.forEach(h => {
    const r = h.predicted_direction === 'up' && h.signal_valid ? h.actual_return/100 : 0;
    cumRet = (1+cumRet)*(1+r)-1;
    labels.push(h.date.slice(5));
    chartData.push((cumRet*100).toFixed(2));
  });

  new Chart(document.getElementById('chart'), {
    type: 'line',
    data: { labels, datasets: [{ label: '누적수익률(%)', data: chartData, borderColor: '#2196F3', fill: true, backgroundColor: 'rgba(33,150,243,0.1)' }] },
    options: { responsive: true, scales: { y: { beginAtZero: true } } }
  });

  // 테이블
  let rows = hist.slice(-10).reverse().map(h => {
    const icon = h.actual_direction ? (h.predicted_direction === h.actual_direction ? '✅' : '❌') : '⏳';
    return `<tr><td>${h.date}</td><td>${h.predicted_direction==='up'?'▲':'▼'}</td><td>${h.confidence?.toFixed(1)||'-'}%</td><td>${h.actual_return!==null?(h.actual_return>0?'+':'')+h.actual_return.toFixed(2)+'%':'—'}</td><td>${icon}</td></tr>`;
  }).join('');
  document.getElementById('hist').innerHTML = `<table><tr><th>날짜</th><th>예측</th><th>신뢰도</th><th>실제</th><th>결과</th></tr>${rows}</table>`;
}
load();
</script>
</body></html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
