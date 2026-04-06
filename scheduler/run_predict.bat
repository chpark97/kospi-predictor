@echo off
REM 매일 오전 8시 실행: 코스피 일일 예측 + 슬랙 알림
cd /d "%~dp0\.."
python main.py predict
