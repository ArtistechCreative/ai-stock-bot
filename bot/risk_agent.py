"""
Risk Agent — 不可绕过的一票否决权
=====================================
基于 Wayland Zhang《AI量化交易从0到1》第 15 课框架

核心原则:
  Risk Agent 拥有否决权，且不可被任何"更好的理由"覆盖。
  所有交易请求必须经过 Risk Agent 的 6 级审核流水线。

用法:
    from risk_agent import RiskAgent, RiskConfig
    agent = RiskAgent(portfolio_context)
    result = agent.review(signal)
    # result.verdict: "APPROVE" / "REDUCE" / "REJECT"
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import json, os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ── 数据类型 ─────────────────────────────────────────────────────

@dataclass
class Position:
    ticker: str
    shares: float
    avg_cost: float
    side: str = "LONG"  # LONG / SHORT


@dataclass
class PortfolioContext:
    """当前组合状态"""
    cash: float = 0.0
    portfolio_value: float = 0.0
    positions: list = field(default_factory=list)  # [Position, ...]
    daily_trades: int = 0
    daily_pnl_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    is_market_hours: bool = True


@dataclass
class TradingSignal:
    """一个待审核的交易请求"""
    ticker: str
    direction: str  # BUY / SELL / SHORT / COVER
    suggested_shares: float
    suggested_price: float
    reason: str = ""
    source: str = "signal_agent"  # 信号来源
    is_crypto: bool = False


@dataclass
class ReviewResult:
    """审核结果"""
    ticker: str
    direction: str
    verdict: str = ""  # APPROVE / REDUCE / REJECT；空=未审核
    approved_shares: float = 0.0
    approved_price: float = 0.0
    approved_stop: float = 0.0
    approved_target: float = 0.0
    reason: str = ""
    review_trail: list = field(default_factory=list)  # 每级审核记录


@dataclass
class RiskState:
    """持久化风控状态（每日/跨运行）"""
    date: str = ""
    daily_trades: int = 0
    daily_pnl: float = 0.0
    peak_value: float = 0.0
    consecutive_losses: int = 0
    circuit_breaker_active: bool = False
    circuit_breaker_reason: str = ""


# ═══════════════════════════════════════════════════════════════════
# Half-Kelly + Van Tharp 仓位计算
# ═══════════════════════════════════════════════════════════════════

def half_kelly_position(win_rate: float, avg_win: float, avg_loss: float,
                        portfolio_value: float, max_position_pct: float = 0.20) -> float:
    """
    Half-Kelly 仓位计算。
    
    参数:
        win_rate: 历史胜率 (0-1)
        avg_win: 平均盈利比例 (如 0.15 = 15%)
        avg_loss: 平均亏损比例 (如 0.08 = 8%)，正数
        portfolio_value: 当前组合总值
        max_position_pct: 单笔最大仓位比例
    
    返回: 建议投入金额
    """
    if avg_loss <= 0:
        return portfolio_value * max_position_pct
    
    # Kelly 公式: f = (b * p - q) / b, 其中 b = avg_win/avg_loss, p = win_rate, q = 1-p
    b = avg_win / avg_loss
    p = win_rate
    q = 1 - p
    kelly_f = (b * p - q) / b if b > 0 else 0
    
    # Half-Kelly: 保守一半
    half_kelly_f = max(0, kelly_f * 0.5)
    
    # 上限：不超过 max_position_pct
    return min(half_kelly_f * portfolio_value, portfolio_value * max_position_pct)


def van_tharp_position(portfolio_value: float, current_price: float,
                       stop_loss_pct: float, max_risk_pct: float = 0.01) -> float:
    """
    Van Tharp R-Multiple 仓位计算。
    单笔亏损永不致命 — 最大亏损不超过组合的 max_risk_pct%。
    
    参数:
        portfolio_value: 当前组合总值
        current_price: 当前价格
        stop_loss_pct: 止损比例 (如 0.08 = 8%)
        max_risk_pct: 单笔最大风险 (如 0.01 = 1%)
    
    返回: 建议买入金额
    """
    risk_per_share = current_price * stop_loss_pct
    if risk_per_share <= 0:
        return 0
    max_risk_amount = portfolio_value * max_risk_pct
    shares = max_risk_amount / risk_per_share
    return shares * current_price


# ═══════════════════════════════════════════════════════════════════
# Risk Agent — 核心审核引擎
# ═══════════════════════════════════════════════════════════════════

class RiskAgent:
    """
    Risk Agent — 所有交易请求的必经门槛。
    
    6 级审核流水线（不可跳过、不可覆盖）:
      Level 1: 单笔金额上限
      Level 2: 标的集中度
      Level 3: 行业/板块集中度
      Level 4: 总仓位/杠杆
      Level 5: 回撤状态
      Level 6: 熔断状态
    """

    def __init__(self, context: PortfolioContext = None, state_path: str = None):
        self.context = context or PortfolioContext()
        self.state_path = state_path or os.path.join(DATA_DIR, "risk_state.json")
        self.state = self._load_state()
        self._update_daily_state()

    # ── 外部接口 ──────────────────────────────────────────────────

    def review(self, signal: TradingSignal) -> ReviewResult:
        """
        审核单个交易信号。返回审核结果（APPROVE / REDUCE / REJECT）。
        这是 Risk Agent 的唯一外部入口。
        """
        result = ReviewResult(
            ticker=signal.ticker,
            direction=signal.direction,
            approved_shares=signal.suggested_shares,
            approved_price=signal.suggested_price,
        )

        # ── Level 1: 单笔金额上限 ──────────────────────────────
        result = self._level1_single_position(result, signal)
        result.review_trail.append(("Level 1 单笔金额上限", result.verdict, result.reason))
        if result.verdict == "REJECT":
            return self._finalize(result)

        # ── Level 2: 标的集中度 ────────────────────────────────
        result = self._level2_concentration(result, signal)
        result.review_trail.append(("Level 2 标的集中度", result.verdict, result.reason))
        if result.verdict == "REJECT":
            return self._finalize(result)

        # ── Level 3: 行业/板块集中度 ───────────────────────────
        result = self._level3_sector_concentration(result, signal)
        result.review_trail.append(("Level 3 行业/板块集中度", result.verdict, result.reason))
        if result.verdict == "REJECT":
            return self._finalize(result)

        # ── Level 4: 总仓位/杠杆 ──────────────────────────────
        result = self._level4_total_exposure(result, signal)
        result.review_trail.append(("Level 4 总仓位/杠杆", result.verdict, result.reason))
        if result.verdict == "REJECT":
            return self._finalize(result)

        # ── Level 5: 回撤状态 ──────────────────────────────────
        result = self._level5_drawdown(result, signal)
        result.review_trail.append(("Level 5 回撤状态", result.verdict, result.reason))
        if result.verdict == "REJECT":
            return self._finalize(result)

        # ── Level 6: 熔断状态 ──────────────────────────────────
        result = self._level6_circuit_breaker(result, signal)
        result.review_trail.append(("Level 6 熔断状态", result.verdict, result.reason))
        if result.verdict == "REJECT":
            return self._finalize(result)

        # ── 全部通过（取最终裁定：如果任何一级为 REDUCE，保持 REDUCE） ──
        final_verdict = "APPROVE"
        for _, verdict, _ in result.review_trail:
            if verdict == "REDUCE":
                final_verdict = "REDUCE"
                break
        result.verdict = final_verdict
        result.reason = f"{len(result.review_trail)} 级审核完成，最终裁定: {final_verdict}"
        return self._finalize(result)

    def review_batch(self, signals: list[TradingSignal]) -> list[ReviewResult]:
        """批量审核多个信号。按优先级排序后逐个审核。"""
        # 按信号强度排序（分数高的优先）
        sorted_signals = sorted(signals, key=lambda s: s.suggested_shares, reverse=True)
        results = []
        for sig in sorted_signals:
            result = self.review(sig)
            results.append(result)
            # 如果被拒绝，释放的额度给下一个信号
        return results

    def update_context(self, context: PortfolioContext):
        """更新组合上下文（每次信号周期前调用）"""
        self.context = context
        self._update_daily_state()

    # ── Level 实现 ──────────────────────────────────────────────

    def _level1_single_position(self, result: ReviewResult, signal: TradingSignal) -> ReviewResult:
        """单笔金额上限 + Half-Kelly/Van Tharp 仓位"""
        val = self.context.portfolio_value
        if val <= 0:
            result.verdict = "REJECT"
            result.reason = "组合价值为0"
            return result

        # 硬上限：单笔不超过组合的 20%
        hard_max = val * 0.20
        if signal.is_crypto:
            hard_max = val * 0.10  # 加密货币更保守

        suggested_amount = signal.suggested_shares * signal.suggested_price

        if suggested_amount <= hard_max:
            result.approved_shares = signal.suggested_shares
            result.reason = f"金额 ${suggested_amount:.0f} < 上限 ${hard_max:.0f}"
            return result
        else:
            # 缩减到上限
            reduced_shares = int(hard_max / signal.suggested_price)
            if reduced_shares <= 0:
                result.verdict = "REJECT"
                result.reason = f"单笔 ${suggested_amount:.0f} 超上限 ${hard_max:.0f}，缩减后为0"
            else:
                result.verdict = "REDUCE"
                result.approved_shares = reduced_shares
                result.reason = f"金额 ${suggested_amount:.0f} > 上限 ${hard_max:.0f}，缩减至 {reduced_shares} 股"
            return result

    def _level2_concentration(self, result: ReviewResult, signal: TradingSignal) -> ReviewResult:
        """标的集中度：同一标的不能持有过重"""
        # 检查是否已有同一标的持仓
        existing = [p for p in self.context.positions if p.ticker == signal.ticker]
        if not existing:
            return result  # 新标的，通过

        current_shares = sum(p.shares for p in existing)
        new_total = current_shares + result.approved_shares
        current_value = current_shares * signal.suggested_price
        new_value = new_total * signal.suggested_price
        val = self.context.portfolio_value

        # 同一标的占比不得超过 20%
        max_conc = val * 0.20
        if signal.is_crypto:
            max_conc = val * 0.10

        if new_value <= max_conc:
            return result
        elif current_value < max_conc:
            # 缩减新增部分
            available = max_conc - current_value
            reduced = int(available / signal.suggested_price)
            if reduced > 0:
                result.verdict = "REDUCE"
                result.approved_shares = reduced
                result.reason = f"标的集中度限制 {signal.ticker}: 缩减至 {reduced} 股"
            else:
                result.verdict = "REJECT"
                result.reason = f"{signal.ticker} 已达集中度上限，无法新增"
        else:
            result.verdict = "REJECT"
            result.reason = f"{signal.ticker} 已达集中度上限"

        return result

    def _level3_sector_concentration(self, result: ReviewResult, signal: TradingSignal) -> ReviewResult:
        """行业/板块集中度"""
        # 简化实现：检查 crypto vs stock 大类
        # 实际使用中可传入 ticker→sector 映射
        if signal.is_crypto:
            # 检查加密仓位占比
            crypto_value = sum(
                p.shares * (p.avg_cost if p.avg_cost else signal.suggested_price)
                for p in self.context.positions
                if "/" in p.ticker
            )
            new_crypto = crypto_value + result.approved_shares * signal.suggested_price
            max_crypto = self.context.portfolio_value * 0.30  # 加密总仓位不超过30%
            if new_crypto > max_crypto:
                result.verdict = "REDUCE"
                result.reason = f"加密板块集中度限制: 当前${crypto_value:.0f}+新增 > 上限${max_crypto:.0f}"
                # 尝试缩减到可用空间
                available = max_crypto - crypto_value
                reduced = int(available / signal.suggested_price)
                if reduced > 0:
                    result.approved_shares = reduced
                else:
                    result.verdict = "REJECT"
                    result.reason = f"加密板块已达上限"
        # 股票板块——可扩展为真实行业映射
        return result

    def _level4_total_exposure(self, result: ReviewResult, signal: TradingSignal) -> ReviewResult:
        """总仓位/杠杆控制"""
        total_long = sum(
            p.shares * signal.suggested_price
            for p in self.context.positions if p.side == "LONG"
        )
        total_short = sum(
            p.shares * signal.suggested_price
            for p in self.context.positions if p.side == "SHORT"
        )
        net_exposure = total_long - total_short
        val = self.context.portfolio_value

        # 计算新增后的净风险暴露
        if signal.direction in ("BUY",):
            new_exposure = net_exposure + result.approved_shares * signal.suggested_price
        elif signal.direction in ("SHORT",):
            new_exposure = net_exposure - result.approved_shares * signal.suggested_price
        else:
            new_exposure = net_exposure  # SELL/COVER 减少风险

        # 杠杆上限：净暴露不超过组合的 150%（保证金账户标准）
        max_leverage = val * 1.5
        if abs(new_exposure) > max_leverage:
            result.verdict = "REJECT"
            result.reason = f"净暴露 ${new_exposure:.0f} 超杠杆上限 ${max_leverage:.0f}"
        return result

    def _level5_drawdown(self, result: ReviewResult, signal: TradingSignal) -> ReviewResult:
        """回撤状态检查"""
        dd = self.context.current_drawdown_pct
        
        # 回撤 < 10%: 正常
        if dd < 10:
            return result
        
        # 回撤 10-15%: 警告模式，缩减仓位
        if dd < 15:
            if signal.direction in ("BUY", "SHORT"):
                # 缩减至 50%
                result.approved_shares = int(result.approved_shares * 0.5)
                if result.approved_shares <= 0:
                    result.verdict = "REJECT"
                    result.reason = f"回撤 {dd:.1f}% 进入警告区，开仓被拒"
                else:
                    result.verdict = "REDUCE"
                    result.reason = f"回撤 {dd:.1f}% 警告，仓位缩减50%"
            return result
        
        # 回撤 > 15%: 停止所有新开仓
        if signal.direction in ("BUY", "SHORT"):
            result.verdict = "REJECT"
            result.reason = f"回撤 {dd:.1f}% 超过上限 15%，禁止新开仓"
        
        return result

    def _level6_circuit_breaker(self, result: ReviewResult, signal: TradingSignal) -> ReviewResult:
        """熔断状态检查"""
        if self.state.circuit_breaker_active:
            # 熔断期间只允许平仓
            if signal.direction in ("BUY", "SHORT"):
                result.verdict = "REJECT"
                result.reason = f"熔断激活: {self.state.circuit_breaker_reason}"
            else:
                result.reason = f"熔断期间执行平仓: {signal.ticker}"
        return result

    # ── 熔断控制 ──────────────────────────────────────────────────

    def trigger_circuit_breaker(self, reason: str):
        """手动触发熔断"""
        self.state.circuit_breaker_active = True
        self.state.circuit_breaker_reason = reason
        self._save_state()
        print(f"  🛑 熔断触发: {reason}")

    def release_circuit_breaker(self):
        """解除熔断"""
        self.state.circuit_breaker_active = False
        self.state.circuit_breaker_reason = ""
        self._save_state()
        print(f"  ✅ 熔断解除")

    # ── 内部方法 ──────────────────────────────────────────────────

    def _finalize(self, result: ReviewResult) -> ReviewResult:
        """审核完成：计算止损/止盈价"""
        if result.verdict != "REJECT" and result.approved_price > 0:
            price = result.approved_price
            is_crypto = "/" in result.ticker
            sl = 0.15 if is_crypto else 0.08
            tp = 0.25 if is_crypto else 0.15
            result.approved_stop = round(price * (1 - sl), 2)
            result.approved_target = round(price * (1 + tp), 2)
        self._save_state()
        return result

    def _load_state(self) -> RiskState:
        """加载持久化风控状态"""
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r') as f:
                    data = json.load(f)
                return RiskState(**data)
        except Exception:
            pass
        return RiskState()

    def _save_state(self):
        """保存风控状态"""
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, 'w') as f:
                json.dump({
                    "date": self.state.date,
                    "daily_trades": self.state.daily_trades,
                    "daily_pnl": self.state.daily_pnl,
                    "peak_value": self.state.peak_value,
                    "consecutive_losses": self.state.consecutive_losses,
                    "circuit_breaker_active": self.state.circuit_breaker_active,
                    "circuit_breaker_reason": self.state.circuit_breaker_reason,
                }, f, indent=2)
        except Exception:
            pass

    def _update_daily_state(self):
        """每日状态更新"""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.state.date != today:
            self.state.date = today
            self.state.daily_trades = 0
            self.state.daily_pnl = 0.0
            self._save_state()


# ═══════════════════════════════════════════════════════════════════
# 便捷函数：将 run_cron 中的信号格式转为 TradingSignal
# ═══════════════════════════════════════════════════════════════════

def signal_from_dict(d: dict) -> TradingSignal:
    """将 run_cron step3 输出的信号 dict 转为 TradingSignal"""
    ticker = d.get("ticker", "")
    return TradingSignal(
        ticker=ticker,
        direction="BUY",
        suggested_shares=d.get("dl_confidence", 50) * 2,  # 保守估计股数
        suggested_price=d.get("price", 0) or 100,
        reason=d.get("reason", ""),
        source="strategy_optimizer",
        is_crypto="/" in ticker,
    )


def result_to_dict(result: ReviewResult) -> dict:
    """将审核结果转为 dict（用于日志/Telegram）"""
    icon = {"APPROVE": "✅", "REDUCE": "⚠️", "REJECT": "❌"}
    return {
        "ticker": result.ticker,
        "direction": result.direction,
        "verdict": result.verdict,
        "icon": icon.get(result.verdict, "❓"),
        "approved_shares": result.approved_shares,
        "approved_stop": result.approved_stop,
        "approved_target": result.approved_target,
        "reason": result.reason,
    }
