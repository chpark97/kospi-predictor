"""코스피 예측 시스템 - 메인 엔트리포인트"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import DATA_DIR, DB_PATH
from collectors import YahooCollector, KRXCollector, FREDCollector


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )


def collect_data(start_date=None, end_date=None):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("=" * 50)
    logging.info("데이터 수집 시작")
    logging.info("=" * 50)

    YahooCollector().collect(start_date=start_date, end_date=end_date)
    KRXCollector().collect(start_date=start_date, end_date=end_date)
    FREDCollector().collect(start_date=start_date, end_date=end_date)

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
    for (table_name,) in tables:
        try:
            count = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            mn = cursor.execute(f"SELECT MIN(date) FROM {table_name}").fetchone()[0]
            mx = cursor.execute(f"SELECT MAX(date) FROM {table_name}").fetchone()[0]
            logging.info(f"{table_name:<25} {count:>8} {mn or 'N/A':>12} {mx or 'N/A':>12}")
        except Exception as e:
            logging.error(f"{table_name}: {e}")
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
                     choices=["baseline", "attention", "cnn", "transformer", "ensemble"])

    sub.add_parser("predict", help="일일 예측 실행")
    sub.add_parser("check", help="어제 예측 정답 확인 + 가중치 업데이트")
    sub.add_parser("weekly-report", help="주간 성과 리포트")

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
    elif args.command == "tune":
        from pipeline.optuna_tuner import run_optuna_tuning
        run_optuna_tuning(n_trials=args.trials)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
