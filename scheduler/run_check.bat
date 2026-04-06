@echo off
REM 매일 오후 4시 실행: 어제 예측 정답 확인 + 가중치 업데이트 + 슬랙 전송
cd /d "%~dp0\.."
python main.py check
