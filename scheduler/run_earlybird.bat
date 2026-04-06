@echo off
REM 매일 오전 6시 실행: 미국 장 마감 후 데이터 수집 + 예측
REM 한국 06:00 = 미국 장 마감 직후 (동부 17:00)
cd /d "%~dp0\.."
python main.py predict
