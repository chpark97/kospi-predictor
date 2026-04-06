# Windows Task Scheduler 설정 가이드

## 환경변수 설정 (필수)

```
변수명: SLACK_WEBHOOK_URL
값: https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

## Task Scheduler 등록

| 작업명 | 배치파일 | 스케줄 | 설명 |
|-------|---------|--------|------|
| KOSPI_EarlyBird | run_earlybird.bat | 매일 06:00 | 미국 장 마감 직후 수집+예측 |
| KOSPI_Predict | run_predict.bat | 매일 08:00 | 일일 예측 (백업) |
| KOSPI_Check | run_check.bat | 매일 16:00 | 정답 확인 + 가중치 업데이트 |
| KOSPI_Weekly | run_weekly.bat | 매주 월 08:30 | 주간 성과 리포트 |
| KOSPI_Retrain | run_retrain.bat | 매주 일 02:00 | 모델 재학습 |

### 등록 방법
1. `taskschd.msc` 실행
2. "작업 만들기" → 이름 입력
3. 트리거: 해당 스케줄 설정
4. 동작: `cmd.exe /c "C:\path\to\kospi-predictor\scheduler\배치파일명"`
5. "사용자가 로그온하지 않아도 실행" 체크

## CLI 명령어

| 명령어 | 설명 |
|--------|------|
| `python main.py predict` | 일일 예측 + 멀티스텝 + SHAP + 슬랙 |
| `python main.py check` | 정답 확인 + 동적 가중치 업데이트 |
| `python main.py weekly-report` | 주간 리포트 |
| `python main.py retrain` | 전체 모델 재학습 |
| `python main.py backfill` | 과거 예측 실적 소급 |
| `python main.py train` | Walk-forward 학습 |
| `python main.py tune --trials 30` | Optuna 튜닝 |
| `python main.py collect` | 데이터 수집 |
