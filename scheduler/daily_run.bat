@echo off
REM 코스피 예측 일일 실행 스크립트
REM Windows Task Scheduler에서 매일 오전 8시 실행 설정:
REM   프로그램: cmd.exe
REM   인수: /c "C:\path\to\kospi-predictor\scheduler\daily_run.bat"

cd /d "%~dp0\.."
python main.py predict

pause
