"""코스피 예측 시스템 - 메인 엔트리포인트"""
import argparse
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import DATA_DIR, DB_PATH
from collectors import YahooCollector, KRXCollector, FREDCollector


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
        ],
    )


def collect_data(start_date=None, end_date=None):
    """Phase 1: 전체 데이터 수집 파이프라인 실행"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 50)
    logging.info("데이터 수집 시작")
    logging.info("=" * 50)

    # 1. Yahoo Finance (글로벌 지수, 원자재, 환율)
    logging.info("[1/3] Yahoo Finance 데이터 수집 중...")
    yahoo = YahooCollector()
    yahoo.collect(start_date=start_date, end_date=end_date)

    # 2. KRX (코스피 지수)
    logging.info("[2/3] KRX 코스피 데이터 수집 중...")
    krx = KRXCollector()
    krx.collect(start_date=start_date, end_date=end_date)

    # 3. FRED (미국 경제지표)
    logging.info("[3/3] FRED 경제지표 데이터 수집 중...")
    fred = FREDCollector()
    fred.collect(start_date=start_date, end_date=end_date)

    logging.info("=" * 50)
    logging.info("데이터 수집 완료")
    logging.info(f"DB 경로: {DB_PATH}")
    logging.info("=" * 50)

    verify_database()


def verify_database():
    """수집된 데이터 현황 출력"""
    import sqlite3

    if not DB_PATH.exists():
        logging.warning("DB 파일이 존재하지 않습니다.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 모든 테이블 목록
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cursor.fetchall()

    logging.info("\n[DB 현황]")
    logging.info(f"{'테이블':<25} {'행 수':>8} {'시작일':>12} {'종료일':>12}")
    logging.info("-" * 60)

    for (table_name,) in tables:
        try:
            count = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            min_date = cursor.execute(f"SELECT MIN(date) FROM {table_name}").fetchone()[0]
            max_date = cursor.execute(f"SELECT MAX(date) FROM {table_name}").fetchone()[0]
            logging.info(f"{table_name:<25} {count:>8} {min_date or 'N/A':>12} {max_date or 'N/A':>12}")
        except Exception as e:
            logging.error(f"{table_name}: 조회 실패 - {e}")

    conn.close()


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="코스피 예측 시스템")
    subparsers = parser.add_subparsers(dest="command")

    # collect 명령
    collect_parser = subparsers.add_parser("collect", help="데이터 수집")
    collect_parser.add_argument("--start", default=None, help="시작일 (YYYY-MM-DD)")
    collect_parser.add_argument("--end", default=None, help="종료일 (YYYY-MM-DD)")

    # verify 명령
    subparsers.add_parser("verify", help="DB 현황 확인")

    # train 명령
    train_parser = subparsers.add_parser("train", help="모델 학습 (Walk-forward)")
    train_parser.add_argument("--model", default="ensemble",
                              choices=["baseline", "attention", "cnn", "transformer", "ensemble"],
                              help="모델 타입 (ensemble: 앙상블 추천)")

    # predict 명령
    subparsers.add_parser("predict", help="일일 예측 실행")

    # tune 명령
    tune_parser = subparsers.add_parser("tune", help="Optuna 하이퍼파라미터 튜닝")
    tune_parser.add_argument("--trials", type=int, default=30, help="탐색 횟수")

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
    elif args.command == "tune":
        from pipeline.optuna_tuner import run_optuna_tuning
        run_optuna_tuning(n_trials=args.trials)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
