@echo off
REM 매주 일요일 오전 2시 실행: 주간 모델 재학습
cd /d "%~dp0\.."
python main.py retrain
