"""실시간 학습 시각화 모듈

matplotlib의 interactive mode를 사용하여 학습 진행 상황을 실시간으로 시각화.
"""
import logging
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('TkAgg')  # Interactive backend
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

CHART_DIR = Path(__file__).parent.parent / "data" / "training_charts"


class TrainingVisualizer:
    """실시간 학습 시각화 클래스"""

    def __init__(self, n_models=11):
        """시각화 초기화

        Args:
            n_models: 앙상블 모델 수
        """
        self.n_models = n_models
        self.fig = None
        self.axes = None
        self.initialized = False

        # 에폭별 메트릭 저장
        self.epoch_train_losses = []
        self.epoch_val_losses = []
        self.epoch_train_das = []
        self.epoch_val_das = []

        # Split별 결과 저장
        self.split_results = []
        self.model_das_per_split = []
        self.feature_rankings = []

        # 임계값별 결과 저장
        self.threshold_results = {}

    def initialize(self):
        """차트 창 초기화 (2x3 서브플롯)"""
        try:
            plt.ion()  # Interactive mode ON
            self.fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            self.axes = axes.flatten()

            # 서브플롯 제목 설정
            titles = [
                "(a) Train/Val Loss",
                "(b) Train/Val DA",
                "(c) 모델별 Test DA",
                "(d) 일별 적중 현황",
                "(e) 피처 중요도 TOP 15",
                "(f) 누적 수익률",
            ]
            for ax, title in zip(self.axes, titles):
                ax.set_title(title, fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3)

            self.fig.suptitle("KOSPI 예측 모델 학습 모니터링", fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            self.fig.canvas.draw()
            plt.pause(0.1)

            self.initialized = True
            logger.info("[Visualizer] 차트 초기화 완료")
        except Exception as e:
            logger.warning(f"[Visualizer] 초기화 실패 (headless 환경?): {e}")
            self.initialized = False

    def update_epoch(self, epoch, train_loss, val_loss, train_da, val_da):
        """에폭별 메트릭 업데이트 (실시간)

        Args:
            epoch: 현재 에폭 번호
            train_loss: 훈련 손실
            val_loss: 검증 손실
            train_da: 훈련 방향 정확도 (%)
            val_da: 검증 방향 정확도 (%)
        """
        self.epoch_train_losses.append(train_loss)
        self.epoch_val_losses.append(val_loss)
        self.epoch_train_das.append(train_da)
        self.epoch_val_das.append(val_da)

        if not self.initialized:
            return

        try:
            # Loss 곡선 업데이트
            ax_loss = self.axes[0]
            ax_loss.clear()
            epochs = range(1, len(self.epoch_train_losses) + 1)
            ax_loss.plot(epochs, self.epoch_train_losses, 'b-', label='Train Loss', linewidth=1.5)
            ax_loss.plot(epochs, self.epoch_val_losses, 'r-', label='Val Loss', linewidth=1.5)
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss')
            ax_loss.set_title('(a) Train/Val Loss', fontsize=10, fontweight='bold')
            ax_loss.legend(loc='upper right', fontsize=8)
            ax_loss.grid(True, alpha=0.3)

            # DA 곡선 업데이트
            ax_da = self.axes[1]
            ax_da.clear()
            ax_da.plot(epochs, self.epoch_train_das, 'b-', label='Train DA', linewidth=1.5)
            ax_da.plot(epochs, self.epoch_val_das, 'r-', label='Val DA', linewidth=1.5)
            ax_da.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
            ax_da.set_xlabel('Epoch')
            ax_da.set_ylabel('Direction Accuracy (%)')
            ax_da.set_title('(b) Train/Val DA', fontsize=10, fontweight='bold')
            ax_da.legend(loc='lower right', fontsize=8)
            ax_da.grid(True, alpha=0.3)

            self.fig.canvas.draw()
            plt.pause(0.01)
        except Exception as e:
            logger.debug(f"[Visualizer] 에폭 업데이트 오류: {e}")

    def reset_epoch_metrics(self):
        """새 모델 학습 시작 시 에폭 메트릭 초기화"""
        self.epoch_train_losses = []
        self.epoch_val_losses = []
        self.epoch_train_das = []
        self.epoch_val_das = []

    def update_split_complete(self, split_idx, model_das, daily_preds, daily_actuals,
                               feature_ranking, dates=None):
        """Split 완료 시 차트 갱신

        Args:
            split_idx: Split 인덱스 (0-based)
            model_das: 모델별 DA 딕셔너리 {model_name: da_value}
            daily_preds: 일별 예측 방향 배열 (True=상승)
            daily_actuals: 일별 실제 방향 배열 (True=상승)
            feature_ranking: 피처 중요도 리스트 [(name, importance), ...]
            dates: 날짜 배열 (optional)
        """
        self.model_das_per_split.append(model_das)
        self.feature_rankings.append(feature_ranking)

        if not self.initialized:
            return

        try:
            # 모델별 DA 막대 차트
            ax_models = self.axes[2]
            ax_models.clear()
            model_names = list(model_das.keys())
            das = list(model_das.values())
            colors = ['green' if d > 52 else 'orange' if d > 50 else 'red' for d in das]
            bars = ax_models.bar(range(len(model_names)), das, color=colors, alpha=0.7)
            ax_models.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
            ax_models.set_xticks(range(len(model_names)))
            ax_models.set_xticklabels(model_names, rotation=45, ha='right', fontsize=7)
            ax_models.set_ylabel('DA (%)')
            ax_models.set_title(f'(c) 모델별 Test DA (Split {split_idx + 1})', fontsize=10, fontweight='bold')
            ax_models.grid(True, alpha=0.3, axis='y')

            # 일별 적중 차트
            ax_daily = self.axes[3]
            ax_daily.clear()
            correct = daily_preds == daily_actuals
            n_days = len(correct)
            colors_daily = ['green' if c else 'red' for c in correct]
            ax_daily.bar(range(n_days), [1] * n_days, color=colors_daily, alpha=0.7, width=1.0)
            ax_daily.set_xlim(0, n_days)
            ax_daily.set_ylim(0, 1.5)
            ax_daily.set_xlabel(f'거래일 (총 {n_days}일, 적중 {sum(correct)}일)')
            ax_daily.set_ylabel('')
            ax_daily.set_yticks([])
            hit_rate = sum(correct) / n_days * 100 if n_days > 0 else 0
            ax_daily.set_title(f'(d) 일별 적중 현황 ({hit_rate:.1f}%)', fontsize=10, fontweight='bold')

            # 피처 중요도 차트
            ax_features = self.axes[4]
            ax_features.clear()
            if feature_ranking:
                top_15 = feature_ranking[:15]
                names = [f[0][:12] for f in top_15]  # 이름 truncate
                importances = [f[1] for f in top_15]
                y_pos = range(len(names))
                ax_features.barh(y_pos, importances, color='steelblue', alpha=0.7)
                ax_features.set_yticks(y_pos)
                ax_features.set_yticklabels(names, fontsize=7)
                ax_features.invert_yaxis()
                ax_features.set_xlabel('Importance')
                ax_features.set_title('(e) 피처 중요도 TOP 15', fontsize=10, fontweight='bold')
                ax_features.grid(True, alpha=0.3, axis='x')

            self.fig.canvas.draw()
            plt.pause(0.1)
        except Exception as e:
            logger.debug(f"[Visualizer] Split 완료 업데이트 오류: {e}")

    def show_final_summary(self, all_split_results, threshold_results=None):
        """최종 요약 차트 표시

        Args:
            all_split_results: 모든 split 결과 리스트
            threshold_results: 임계값별 성과 딕셔너리 {threshold: metrics}
        """
        self.split_results = all_split_results
        self.threshold_results = threshold_results or {}

        if not self.initialized:
            return

        try:
            # 누적 수익률 차트 (Split별)
            ax_returns = self.axes[5]
            ax_returns.clear()

            if all_split_results:
                splits = [r.get('split', str(i)) for i, r in enumerate(all_split_results)]
                returns = [r.get('cumulative_return', 0) for r in all_split_results]
                das = [r.get('direction_accuracy', 50) for r in all_split_results]

                x = range(len(splits))
                width = 0.35

                bars1 = ax_returns.bar([i - width/2 for i in x], returns, width,
                                       label='수익률 (%)', color='steelblue', alpha=0.7)
                ax_returns.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

                ax2 = ax_returns.twinx()
                bars2 = ax2.bar([i + width/2 for i in x], das, width,
                               label='DA (%)', color='green', alpha=0.5)
                ax2.axhline(y=50, color='green', linestyle='--', alpha=0.3)
                ax2.set_ylabel('DA (%)', color='green')

                ax_returns.set_xticks(x)
                ax_returns.set_xticklabels(splits, fontsize=9)
                ax_returns.set_xlabel('Split (연도)')
                ax_returns.set_ylabel('수익률 (%)', color='steelblue')
                ax_returns.set_title('(f) Split별 성과', fontsize=10, fontweight='bold')
                ax_returns.legend(loc='upper left', fontsize=8)
                ax2.legend(loc='upper right', fontsize=8)
                ax_returns.grid(True, alpha=0.3, axis='y')

            self.fig.canvas.draw()
            plt.pause(0.1)
            logger.info("[Visualizer] 최종 요약 차트 갱신 완료")
        except Exception as e:
            logger.debug(f"[Visualizer] 최종 요약 오류: {e}")

    def save_charts(self, output_dir=None):
        """차트를 PNG로 저장

        Args:
            output_dir: 저장 디렉토리 (기본: data/training_charts/)

        Returns:
            저장된 파일 경로
        """
        output_dir = Path(output_dir) if output_dir else CHART_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_report_{timestamp}.png"
        filepath = output_dir / filename

        if self.fig:
            try:
                self.fig.savefig(filepath, dpi=150, bbox_inches='tight',
                               facecolor='white', edgecolor='none')
                logger.info(f"[Visualizer] 차트 저장: {filepath}")
                return str(filepath)
            except Exception as e:
                logger.warning(f"[Visualizer] 차트 저장 실패: {e}")

        return None

    def show_blocking(self):
        """차트 창을 blocking 모드로 표시 (사용자가 닫을 때까지 유지)"""
        if self.initialized and self.fig:
            try:
                plt.ioff()  # Interactive mode OFF
                plt.show(block=True)
            except Exception as e:
                logger.debug(f"[Visualizer] Blocking 표시 오류: {e}")

    def close(self):
        """차트 창 닫기"""
        if self.fig:
            try:
                plt.close(self.fig)
                self.fig = None
                self.initialized = False
            except Exception:
                pass


def format_training_report(all_split_results, model_rankings=None, feature_rankings=None,
                           best_threshold=None, threshold_metrics=None):
    """학습 리포트를 슬랙 텍스트로 포맷

    Args:
        all_split_results: Split별 결과 리스트
        model_rankings: 모델별 성능 랭킹 [(model_name, da), ...]
        feature_rankings: 피처 중요도 랭킹 [(name, importance), ...]
        best_threshold: 최적 신뢰도 임계값
        threshold_metrics: 최적 임계값의 메트릭

    Returns:
        슬랙 메시지 문자열
    """
    today = datetime.now().strftime("%Y-%m-%d")

    lines = [f"*학습 리포트 ({today})*\n"]
    lines.append("*Walk-forward 결과:*")

    for r in all_split_results:
        split = r.get('split', '?')
        da = r.get('direction_accuracy', 0)
        ret = r.get('cumulative_return', 0)
        sharpe = r.get('sharpe_ratio', 0)
        sign = '+' if ret >= 0 else ''
        lines.append(f"  Split {split}: DA={da:.1f}% | 수익={sign}{ret:.1f}% | 샤프={sharpe:.2f}")

    if all_split_results:
        avg_da = np.mean([r.get('direction_accuracy', 0) for r in all_split_results])
        avg_ret = np.mean([r.get('cumulative_return', 0) for r in all_split_results])
        sign = '+' if avg_ret >= 0 else ''
        lines.append(f"  *평균: DA={avg_da:.1f}% | 수익={sign}{avg_ret:.1f}%*")

    # 모델 성능 랭킹
    if model_rankings and len(model_rankings) >= 3:
        lines.append(f"\n*모델 성능 (최종 Split 기준):*")
        lines.append(f"  1. {model_rankings[0][0]} ({model_rankings[0][1]:.1f}%)")
        lines.append(f"  2. {model_rankings[1][0]} ({model_rankings[1][1]:.1f}%)")
        lines.append(f"  3. {model_rankings[2][0]} ({model_rankings[2][1]:.1f}%)")

    # 피처 TOP 5
    if feature_rankings:
        top5 = feature_rankings[:5]
        feature_str = ", ".join([f[0] for f in top5])
        lines.append(f"\n*피처 TOP 5:* {feature_str}")

    # 최적 임계값
    if best_threshold is not None and threshold_metrics:
        ret = threshold_metrics.get('cumulative_return', 0)
        trades = threshold_metrics.get('trade_count', 0)
        sign = '+' if ret >= 0 else ''
        lines.append(f"\n*최적 임계값:* {best_threshold}% (수익 {sign}{ret:.1f}%, 거래 {trades}건)")

    lines.append(f"\n*차트 저장:* data/training_charts/")

    return "\n".join(lines)
