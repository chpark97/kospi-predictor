@echo off
REM 매주 월요일 오전 8시 30분 실행: 주간 성과 리포트 + 슬랙 전송
cd /d "%~dp0\.."
python main.py weekly-report
