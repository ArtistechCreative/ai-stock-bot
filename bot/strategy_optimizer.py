"""
策略自动更新引擎
- 结合 DL 预测信号 + 市场评分信号
- 基于回测表现自动调整参数（止损/止盈/仓位）
- 记录每次调参的原因和结果
"""
import os, json, copy
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class StrategyParams:
    """当前策略参数"""
    stop_loss_pct: float = 8.0
    take_profit_pct: float = 15.0
    max_position_pct: float = 20.0
    max_positions: int = 5
    dl_weight: float = 0.5     # DL 信号权重 (0-1)
    score_weight: float = 0.5  # 评分信号权重
    min_confidence: float = 65.0  # 最低信心度（低于此忽略 DL 信号）

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PerformanceRecord:
    """每次调参后的表现记录"""
    date: str
    params: dict
    backtest_result: dict  # {total_return, max_drawdown, win_rate, sharpe}
    dl_accuracy: float     # DL 预测准确率
    composite_score: float  # 综合评分（回测分 * DL准确率）
    action: str            # "ADJUST" / "KEEP" / "RESET"


class StrategyOptimizer:
    """
    策略自动优化器：
    - 维护当前参数
    - 每次调参后记录到历史
    - 根据历史表现决定是否继续调整
    """

    def __init__(
        self,
        initial_params: StrategyParams = None,
        state_path: str = None,
    ):
        self.params = initial_params or StrategyParams()
        self.state_path = state_path or str(DATA_DIR / "strategy_state.json")
        self.history: list[PerformanceRecord] = []
        self.best_score = -999.0
        self.best_params: Optional[StrategyParams] = None
        self._load()

    def _load(self):
        if Path(self.state_path).exists():
            data = json.load(open(self.state_path))
            self.params = StrategyParams(**data.get("params", {}))
            self.history = [PerformanceRecord(**r) for r in data.get("history", [])]
            self.best_score = data.get("best_score", -999.0)
            if data.get("best_params"):
                self.best_params = StrategyParams(**data["best_params"])

    def save(self):
        data = {
            "params": self.params.to_dict(),
            "best_params": self.best_params.to_dict() if self.best_params else None,
            "best_score": self.best_score,
            "history": [asdict(r) for r in self.history],
        }
        with open(self.state_path, "w") as f:
            json.dump(data, f, indent=2)

    def record(self, backtest_result: dict, dl_accuracy: float, composite_score: float):
        """记录一次策略表现"""
        record = PerformanceRecord(
            date=datetime.now().strftime("%Y-%m-%d"),
            params=self.params.to_dict(),
            backtest_result=backtest_result,
            dl_accuracy=dl_accuracy,
            composite_score=composite_score,
            action="ADJUST",
        )
        self.history.append(record)

        if composite_score > self.best_score:
            self.best_score = composite_score
            self.best_params = copy.copy(self.params)

        self.save()

    def adjust_params(
        self,
        backtest_return: float,
        max_drawdown: float,
        win_rate: float,
        sharpe: float,
        dl_accuracy: float,
    ) -> StrategyParams:
        """
        根据回测结果 + DL 准确率调整参数
        返回调整后的参数（但不立即应用，需确认）
        """
        new_params = copy.copy(self.params)
        changes = []

        # 综合评分
        # 权重：收益率40%，回撤30%，胜率15%，Sharpe 15%
        score = (
            backtest_return * 0.40
            - max_drawdown * 0.30
            + win_rate * 0.15
            + sharpe * 0.15
        ) * (0.5 + dl_accuracy * 0.5)  # DL 准确率加成/惩罚

        composite_score = round(score, 3)

        # 调整逻辑
        if max_drawdown > 15:
            # 回撤过大 → 收紧止损
            new_params.stop_loss_pct = max(3.0, self.params.stop_loss_pct - 1.0)
            changes.append(f"回撤{max_drawdown:.1f}%→收紧止损→{new_params.stop_loss_pct}%")

        if backtest_return < 0 and dl_accuracy < 0.55:
            # 亏损 + DL 准确率低 → 降低 DL 权重
            new_params.dl_weight = max(0.2, self.params.dl_weight - 0.1)
            changes.append(f"亏损+DL准确率{dl_accuracy:.0%}→降低DL权重→{new_params.dl_weight}")

        if win_rate > 0.65 and dl_accuracy > 0.6:
            # 高胜率 + DL 靠谱 → 可以放大仓位
            new_params.max_position_pct = min(40.0, self.params.max_position_pct + 5.0)
            changes.append(f"高胜率({win_rate:.0%})+DL({dl_accuracy:.0%})→放大仓位→{new_params.max_position_pct}%")

        if dl_accuracy < 0.5 and self.params.dl_weight > 0.3:
            # DL 不准 → 降低 DL 权重
            new_params.dl_weight = max(0.1, self.params.dl_weight - 0.15)
            changes.append(f"DL准确率{dl_accuracy:.0%}<50%→降DL权重→{new_params.dl_weight}")

        if max_drawdown < 5 and backtest_return > 5:
            # 回撤低 + 收益好 → 可以放宽止盈
            new_params.take_profit_pct = min(30.0, self.params.take_profit_pct + 2.0)
            changes.append(f"回撤低+收益好→放宽止盈→{new_params.take_profit_pct}%")

        # 记录
        self.record(
            backtest_result={
                "return": backtest_return,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "sharpe": sharpe,
            },
            dl_accuracy=dl_accuracy,
            composite_score=composite_score,
        )

        print(f"  📊 综合评分: {composite_score:.3f}")
        print(f"  📝 调整: {'; '.join(changes) if changes else '无调整（KEEP）'}")

        return new_params, changes, composite_score

    def apply_params(self, new_params: StrategyParams):
        """确认应用新参数"""
        self.params = new_params
        self.save()

    def reset_to_best(self):
        """重置为历史最佳参数"""
        if self.best_params:
            self.params = copy.copy(self.best_params)
            self.save()
            print(f"  ↩️ 重置为历史最佳参数 (score={self.best_score:.3f})")

    def get_signal(
        self,
        dl_signals: list[dict],   # [{ticker, signal, confidence}]
        score_rank: list[dict],    # [{ticker, score}]
    ) -> list[dict]:
        """
        综合 DL 信号 + 评分信号，输出最终交易信号
        dl_weight=0.5: 各50%权重
        """
        combined = {}

        # 评分信号权重
        for i, s in enumerate(score_rank):
            ticker = s["ticker"]
            score_signal = 1 - (i / len(score_rank))  # 排名1=1.0, 排名n=接近0
            combined[ticker] = {
                "score_signal": score_signal,
                "dl_signal": 0,
                "dl_confidence": 0,
            }

        # DL 信号权重
        for sig in dl_signals:
            ticker = sig["ticker"]
            if ticker in combined:
                confidence = sig.get("confidence", 0)
                if confidence >= self.params.min_confidence:
                    if sig["signal"] == "BUY":
                        combined[ticker]["dl_signal"] = confidence / 100
                    elif sig["signal"] == "SELL":
                        combined[ticker]["dl_signal"] = -confidence / 100

        # 综合评分
        results = []
        for ticker, c in combined.items():
            total = (
                c["score_signal"] * self.params.score_weight
                + c["dl_signal"] * self.params.dl_weight
                * (c["dl_confidence"] / 100)
            )
            # 归一化到 0-1
            score = max(0, min(1, (total + 1) / 2))

            results.append({
                "ticker": ticker,
                "combined_score": round(score, 3),
                "dl_signal": c["dl_signal"],
                "score_signal": c["score_signal"],
            })

        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results