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
    dl_weight: float = 0.2     # DL 信号奖励权重（仅在高信心时加分，不参与综合决策）
    score_weight: float = 1.0  # 评分权重（固定1.0，DL为加分器）
    min_confidence: float = 70.0  # 最低信心度（低于此忽略 DL 信号）
    trailing_stop_pct: float = 2.0  # 移动止盈（%，TP激活后从最高点回撤触发）

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
        backtest_result=None,
    ) -> StrategyParams:
        """
        根据回测结果 + DL 准确率调整参数
        backtest_result: BacktestResult 对象（兼容旧 dict）
        """
        new_params = copy.copy(self.params)
        changes = []

        # ── 兼容：支持 BacktestResult 对象或 dict ──────────────
        if hasattr(backtest_result, 'total_trades'):
            n_trades = backtest_result.total_trades
            tp_rate = backtest_result.take_profit_rate
            sl_rate = backtest_result.stop_loss_rate
        elif isinstance(backtest_result, dict):
            n_trades = backtest_result.get("n_trades", 0)
            tp_rate = backtest_result.get("take_profit_rate", 0)
            sl_rate = backtest_result.get("stop_loss_rate", 0)
        else:
            n_trades = 0; tp_rate = 0; sl_rate = 0

        # ── 综合评分（改进版 v2）────────────────────────────
        # 归一化到可比规模：收益率±15% → 0-10，回撤→负分，胜率→0-10，Sharpe→0-10
        # 交易频率奖励（期望 20-40 笔/180 天）
        expected_trades = 30
        trade_bonus = min(2.5, n_trades / expected_trades * 1.8)  # 0 ~ 2.5 分

        ret_norm = max(-5, min(15, backtest_return)) / 1.5        # ±15% → ~±10 分
        dd_penalty = -min(max_drawdown, 30) / 3                   # 回撤 → 负分（-10 分封顶）
        win_norm = win_rate * 10                                   # 胜率 → 0-10
        sharpe_norm = min(sharpe, 4) * 2.5                         # Sharpe → 0-10（4.0=满分）

        # ── 不对称奖励：赚钱时的效率 vs 亏损时的保护 ──────────
        # 止盈成功率（take_profit 触发次数 / 总卖出次数）
        pnl_asymmetry = tp_rate * 3 - sl_rate * 2                # 止盈奖励 > 止损惩罚

        raw_score = ret_norm + dd_penalty + win_norm + sharpe_norm + trade_bonus + pnl_asymmetry

        # ── DL 调制因子（0.3~1.6）── 更激进的区分度 ─────────
        # 低于 45% 严重惩罚（模型不可靠），高于 65% 强烈加成
        if dl_accuracy < 0.45:
            dl_factor = 0.3 + dl_accuracy * 1.0   # 0.40 acc → 0.70 factor
        elif dl_accuracy < 0.52:
            dl_factor = 0.75 + (dl_accuracy - 0.45) * 2.5  # 0.45→0.75, 0.50→0.875
        elif dl_accuracy < 0.60:
            dl_factor = 0.9 + (dl_accuracy - 0.52) * 2.5   # 0.52→0.9, 0.57→1.0
        elif dl_accuracy < 0.68:
            dl_factor = 1.05 + (dl_accuracy - 0.60) * 3.5  # 0.62→1.12, 0.65→1.23
        else:
            dl_factor = 1.3 + (dl_accuracy - 0.68) * 4.0   # 0.70→1.38, 0.75→1.58

        score = raw_score * dl_factor
        composite_score = round(score, 3)

        # ── 记录详细分解 ────────────────────────────────────
        backtest_result_detail = {
            "return": backtest_return,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "sharpe": sharpe,
            "n_trades": n_trades,
            "dl_accuracy": dl_accuracy,
            "dl_factor": round(dl_factor, 3),
            "raw_score": round(raw_score, 3),
            "ret_norm": round(ret_norm, 2),
            "dd_penalty": round(dd_penalty, 2),
            "win_norm": round(win_norm, 2),
            "sharpe_norm": round(sharpe_norm, 2),
            "trade_bonus": round(trade_bonus, 3),
            "pnl_asymmetry": round(pnl_asymmetry, 2),
        }

        # ── 参数调整规则 ──────────────────────────────────
        if max_drawdown > 20:
            # 重大回撤 → 立即收紧止损 + 降仓位
            new_params.stop_loss_pct = max(3.0, self.params.stop_loss_pct - 2.0)
            new_params.max_position_pct = max(10.0, self.params.max_position_pct - 5.0)
            changes.append(f"回撤>{max_drawdown:.0f}%→止损{new_params.stop_loss_pct}%+仓位{new_params.max_position_pct}%")
        elif max_drawdown > 12:
            new_params.stop_loss_pct = max(4.0, self.params.stop_loss_pct - 1.0)
            changes.append(f"回撤>{max_drawdown:.0f}%→收紧止损→{new_params.stop_loss_pct}%")

        if dl_accuracy < 0.45:
            # DL 模型不准 → 大幅降权重，提高最低门槛
            new_params.dl_weight = max(0.1, self.params.dl_weight - 0.2)
            new_params.min_confidence = max(70.0, self.params.min_confidence + 5.0)
            changes.append(f"DL准确率{dl_accuracy:.0%}<45%→dl_weight={new_params.dl_weight}, min_conf={new_params.min_confidence}%")
        elif dl_accuracy < 0.52:
            new_params.dl_weight = max(0.2, self.params.dl_weight - 0.1)
            changes.append(f"DL准确率{dl_accuracy:.0%}<52%→dl_weight={new_params.dl_weight}")

        if dl_accuracy > 0.60 and win_rate > 0.60:
            # DL 靠谱 + 高胜率 → 放大仓位和持仓数
            new_params.max_position_pct = min(50.0, self.params.max_position_pct + 5.0)
            new_params.max_positions = min(8, self.params.max_positions + 1)
            changes.append(f"DL{dl_accuracy:.0%}+胜率{win_rate:.0%}→仓位{new_params.max_position_pct}%,持仓数{new_params.max_positions}")

        if backtest_return > 10 and max_drawdown < 5:
            # 强劲表现 → 放宽止盈
            new_params.take_profit_pct = min(35.0, self.params.take_profit_pct + 3.0)
            changes.append(f"高回报+低回撤→放宽止盈→{new_params.take_profit_pct}%")

        if sharpe > 1.5 and dl_accuracy > 0.55:
            # 高夏普比 + DL 信号可靠 → 提高 DL 权重
            new_params.dl_weight = min(0.8, self.params.dl_weight + 0.1)
            changes.append(f"Sharpe>{sharpe:.1f}+DL>{dl_accuracy:.0%}→dl_weight={new_params.dl_weight}")

        # 当 DL 准确率回升，自动调低最低信心门槛
        if dl_accuracy > 0.58 and self.params.min_confidence > 60:
            new_params.min_confidence = max(55.0, self.params.min_confidence - 3.0)
            changes.append(f"DL回升→min_conf={new_params.min_confidence}%")

        # 记录
        self.record(
            backtest_result={
                "return": backtest_return,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "sharpe": sharpe,
                "dl_accuracy": dl_accuracy,
            },
            dl_accuracy=dl_accuracy,
            composite_score=composite_score,
        )

        print(f"  📊 综合评分: {composite_score:.3f} (dl_acc={dl_accuracy:.1%}, raw={raw_score:.3f}, tp={tp_rate:.0%}, sl={sl_rate:.0%})")
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
        综合信号：评分排名为基准，DL 高信心时加分奖励。

        旧逻辑（已废弃）：DL信号和评分信号各50%权重混合
        新逻辑：composite = score_signal + (dl_conf ≥ 70% ? confidence_bonus : 0)
                DL权重 0.2 = 最多给 top 信号加 0.2 分（20%提升空间）
                DL 不达标时不扣分，只是不给奖励
        """
        combined = {}

        # 评分信号权重
        for i, s in enumerate(score_rank):
            ticker = s["ticker"]
            score_signal = 1 - (i / len(score_rank))  # 排名1=1.0, 排名n=接近0
            combined[ticker] = {
                "score_signal": score_signal,
                "dl_confidence": 0,
                "dl_bonus": 0,
            }

        # DL 信号：只给高信心信号加分，不参与综合评分直接决策
        for sig in dl_signals:
            ticker = sig["ticker"]
            if ticker in combined:
                confidence = sig.get("confidence", 0)
                # 信心阈值 70%，低于此不做任何操作
                if confidence >= self.params.min_confidence:
                    dl_weight = self.params.dl_weight  # 默认 0.2
                    if sig["signal"] == "BUY":
                        # 正向bonus = (confidence/100) × dl_weight，范围 [0, 0.2]
                        combined[ticker]["dl_bonus"] = (confidence / 100) * dl_weight
                        combined[ticker]["dl_confidence"] = confidence
                    elif sig["signal"] == "SELL":
                        # SELL 信号给负分（降低排名），权重更小
                        combined[ticker]["dl_bonus"] = -(confidence / 100) * (dl_weight * 0.5)
                        combined[ticker]["dl_confidence"] = -confidence

        # 综合评分 = 评分排名 + DL信心bonus
        results = []
        for ticker, c in combined.items():
            total = c["score_signal"] + c["dl_bonus"]
            # 归一化到 0-1（score_signal 范围 [0,1]，bonus 范围 [-0.1, +0.2]）
            score = max(0, min(1, total))

            results.append({
                "ticker": ticker,
                "combined_score": round(score, 3),
                "score_signal": round(c["score_signal"], 3),
                "dl_bonus": round(c["dl_bonus"], 3),
                "dl_confidence": c["dl_confidence"],
            })

        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results