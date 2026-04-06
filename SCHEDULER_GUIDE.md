# Windows Task Scheduler 설정 가이드

## 환경변수 설정 (필수)

시스템 환경변수에 슬랙 웹훅 URL을 등록합니다:

```
변수명: SLACK_WEBHOOK_URL
값: https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

설정 방법: 시스템 속성 → 고급 → 환경 변수 → 시스템 변수 → 새로 만들기

## Task Scheduler 등록

### 작업 1: 일일 예측 (매일 오전 8시)

1. Task Scheduler 열기 (taskschd.msc)
2. "작업 만들기" 클릭
3. **일반 탭**
   - 이름: `KOSPI_Predict`
   - "사용자가 로그온하지 않아도 실행" 선택
4. **트리거 탭** → 새로 만들기
   - 시작: 매일
   - 시간: 오전 8:00:00
5. **동작 탭** → 새로 만들기
   - 프로그램: `cmd.exe`
   - 인수: `/c "C:\path\to\kospi-predictor\scheduler\run_predict.bat"`
6. 확인 → 비밀번호 입력

### 작업 2: 정답 확인 (매일 오후 4시)

1. "작업 만들기" 클릭
2. 이름: `KOSPI_Check`
3. 트리거: 매일 오후 4:00:00
4. 동작: `cmd.exe /c "C:\path\to\kospi-predictor\scheduler\run_check.bat"`

### 작업 3: 주간 리포트 (매주 월요일 오전 8시 30분)

1. "작업 만들기" 클릭
2. 이름: `KOSPI_Weekly`
3. 트리거: 매주 월요일 오전 8:30:00
4. 동작: `cmd.exe /c "C:\path\to\kospi-predictor\scheduler\run_weekly.bat"`

## CLI 명령어 정리

| 명령어 | 설명 | 스케줄 |
|--------|------|--------|
| `python main.py predict` | 일일 예측 + 슬랙 알림 | 매일 08:00 |
| `python main.py check` | 정답 확인 + 가중치 업데이트 | 매일 16:00 |
| `python main.py weekly-report` | 주간 성과 리포트 | 매주 월 08:30 |
| `python main.py collect` | 데이터 수집 | 수동 |
| `python main.py train` | 모델 학습 | 수동 |
| `python main.py tune --trials 30` | 하이퍼파라미터 튜닝 | 수동 |
| `python main.py verify` | DB 현황 확인 | 수동 |
