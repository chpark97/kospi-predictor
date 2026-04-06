"""코스피 예측 시스템 - 메인 엔트리포인트"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import DATA_DIR, DB_PATH
from collectors import YahooCollector, KRXCollector, FREDCollector, KoreaSpecificCollector


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )


def collect_data(start_date=None, end_date=None):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("데이터 수집 시작")
    YahooCollector().collect(start_date=start_date, end_date=end_date)
    KRXCollector().collect(start_date=start_date, end_date=end_date)
    FREDCollector().collect(start_date=start_date, end_date=end_date)
    KoreaSpecificCollector().collect(start_date=start_date, end_date=end_date)
    logging.info("데이터 수집 완료")
    verify_database()


def verify_database():
    import sqlite3
    if not DB_PATH.exists():
        logging.warning("DB 파일이 존재하지 않습니다.")
        return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cursor.fetchall()
    logging.info(f"\n{'테이블':<25} {'행 수':>8} {'시작일':>12} {'종료일':>12}")
    logging.info("-" * 60)
    for (t,) in tables:
        try:
            c = cursor.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            mn = cursor.execute(f"SELECT MIN(date) FROM {t}").fetchone()[0]
            mx = cursor.execute(f"SELECT MAX(date) FROM {t}").fetchone()[0]
            logging.info(f"{t:<25} {c:>8} {mn or 'N/A':>12} {mx or 'N/A':>12}")
        except Exception as e:
            logging.error(f"{t}: {e}")
    conn.close()


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="코스피 예측 시스템")
    sub = parser.add_subparsers(dest="command")

    cp = sub.add_parser("collect", help="데이터 수집")
    cp.add_argument("--start", default=None)
    cp.add_argument("--end", default=None)

    sub.add_parser("verify", help="DB 현황 확인")

    tp = sub.add_parser("train", help="모델 학습")
    tp.add_argument("--model", default="ensemble",
                     choices=["baseline", "attention", "cnn", "transformer", "tft", "ensemble"])

    sub.add_parser("predict", help="일일 예측 실행")
    sub.add_parser("check", help="정답 확인 + 가중치 업데이트")
    sub.add_parser("weekly-report", help="주간 성과 리포트")
    sub.add_parser("retrain", help="모델 재학습")
    sub.add_parser("backfill", help="과거 예측 실적 소급")
    sub.add_parser("dashboard", help="웹 대시보드 실행 (localhost:8000)")

    tn = sub.add_parser("tune", help="Optuna 하이퍼파라미터 튜닝")
    tn.add_argument("--trials", type=int, default=30)

    args = parser.parse_args()

    if args.command == "collect":
        collect_data(start_date=args.start, end_date=args.end)
    elif args.command == "verify":
        verify_database()
    elif args.command == "train":
        from pipeline.train import walk_forward_train
        walk_forward_train(args.model)
    elif args.command == "predict":
        from pipeline.predict import run_prediction
        run_prediction()
    elif args.command == "check":
        from pipeline.predict import run_verify_yesterday
        run_verify_yesterday()
    elif args.command == "weekly-report":
        from pipeline.predict import run_weekly_report
        run_weekly_report()
    elif args.command == "retrain":
        from pipeline.predict import run_retrain
        run_retrain()
    elif args.command == "backfill":
        from pipeline.predict import backfill_prediction_history
        backfill_prediction_history()
    elif args.command == "tune":
        from pipeline.optuna_tuner import run_optuna_tuning
        run_optuna_tuning(n_trials=args.trials)
    elif args.command == "dashboard":
        from dashboard.app import app
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
