"""
风险管理器 — 支持多头+空头
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class RiskConfig:
    """用户自定义风险管理参数"""
    # === 资金管理 ===
    max_single_position_pct: float = 20.0   # 单笔最大仓位（%总资金）
    max_total_exposure_pct: float = 80.0    # 总持仓上限（%）

    # === 止损/止盈 ===
    stop_loss_default_pct: float = 8.0      # 默认止损（%）
    trailing_stop_pct: float = 5.0          # 移动止盈（%）
    profit_taking_pct: float = 15.0         # 止盈线（%）

    # === 短线规则 ===
    max_positions: int = 5                  # 最大同时持仓（含多空）
    min_holding_days: int = 1               # 最短持有
    max_holding_days: int = 5               # 最长持有（天）
    max_trades_per_day: int = 3             # 每天最大交易次数
    max_loss_per_day_pct: float = 5.0       # 每天最大亏损（%）

    # === 空头专属 ===
    short_margin_pct: float = 50.0          # 空头保证金要求（%开仓价值）
    short_max_position_pct: float = 20.0    # 空头单笔最大仓位（%总资金）
    short_stop_loss_pct: float = 8.0        # 空头止损（股价涨幅超此%则止损）
    short_take_profit_pct: float = 15.0     # 空头止盈（股价跌幅超此%则止盈）
    allow_short: bool = True                # 是否允许做空

    # ── 加密货币专属风控（7x24 高波动）──────────────────────────
    crypto_stop_loss_pct: float = 15.0    # 加密货币止损（更宽）
    crypto_take_profit_pct: float = 25.0  # 加密货币止盈（更高目标）
    crypto_trailing_stop_pct: float = 8.0 # 加密货币追踪止损（从峰值回撤 8% 触发）
    crypto_max_position_pct: float = 10.0 # 加密货币单笔仓位上限

    # === 回撤限制 ===
    max_drawdown_pct: float = 15.0          # 最大回撤（触发强平）

    def total_short_margin_required(self, price: float, shares: float) -> float:
        """空头占用保证金 = 股数 × 价格 × 保证金%"""
        return shares * price * (self.short_margin_pct / 100)


@dataclass
class RiskStatus:
    can_open_long: bool = True
    can_open_short: bool = True
    reason: str = ""
    positions_count: int = 0
    long_exposure_pct: float = 0.0
    short_exposure_pct: float = 0.0
    portfolio_value: float = 0.0
    daily_pnl_pct: float = 0.0


class RiskManager:
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()

    # ---- 多头风控 ----

    def can_open_long(
        self,
        price: float,
        portfolio_value: float,
        current_positions: dict,  # {ticker: {position_type, shares, avg_cost}}
        daily_trades: int = 0,
    ) -> tuple[bool, str, float]:
        cfg = self.config

        # 持仓数量
        long_count = sum(1 for p in current_positions.values() if p.get("position_type") != "SHORT")
        if long_count >= cfg.max_positions:
            return False, f"多头已达最大持仓数({cfg.max_positions})", 0

        # 日内次数
        if daily_trades >= cfg.max_trades_per_day:
            return False, f"今日交易次数已达上限({cfg.max_trades_per_day})", 0

        max_amount = portfolio_value * (cfg.max_single_position_pct / 100)
        shares = max_amount / price

        if max_amount <= 0:
            return False, "资金不足", 0

        return True, "OK", shares

    def should_stop_loss_long(
        self,
        entry_price: float,
        current_price: float,
        stop_loss_price: float,
    ) -> tuple[bool, str]:
        """多头止损：价格跌破止损价"""
        loss_pct = (entry_price - current_price) / entry_price * 100
        if current_price <= stop_loss_price and stop_loss_price > 0:
            return True, f"触发止损（亏损{loss_pct:.1f}%）"
        return False, ""

    def should_take_profit_long(
        self,
        entry_price: float,
        current_price: float,
    ) -> tuple[bool, str]:
        """多头止盈"""
        gain_pct = (current_price - entry_price) / entry_price * 100
        if gain_pct >= self.config.profit_taking_pct:
            return True, f"达到止盈线(+{gain_pct:.1f}%)"
        return False, ""

    # ---- 空头风控 ----

    def can_open_short(
        self,
        price: float,
        shares: float,
        portfolio_value: float,
        cash: float,
        current_positions: dict,
        daily_trades: int = 0,
    ) -> tuple[bool, str, float]:
        """
        检查是否可以开空头
        保证金 = shares × price × margin_pct
        """
        cfg = self.config

        if not cfg.allow_short:
            return False, "系统禁止做空", 0

        short_count = sum(1 for p in current_positions.values() if p.get("position_type") == "SHORT")
        if short_count >= cfg.max_positions:
            return False, f"空头已达最大持仓数({cfg.max_positions})", 0

        if daily_trades >= cfg.max_trades_per_day:
            return False, f"今日交易次数已达上限({cfg.max_trades_per_day})", 0

        margin_required = cfg.total_short_margin_required(price, shares)
        if margin_required > cash:
            return False, f"保证金不足(需要${margin_required:.0f}, 有${cash:.0f})", 0

        # 空头占用保证金后，剩余可开的保证金
        if margin_required > cash * 0.5:
            return False, f"保证金不足({margin_required / cash * 100:.0f}%cash)", 0

        return True, "OK", shares

    def should_stop_loss_short(
        self,
        entry_price: float,
        current_price: float,
    ) -> tuple[bool, str]:
        """
        空头止损：股价上涨超过 short_stop_loss_pct 时触发
        """
        cfg = self.config
        loss_pct = (current_price - entry_price) / entry_price * 100
        if loss_pct >= cfg.short_stop_loss_pct:
            return True, f"触发空头止损（股价上涨{loss_pct:.1f}%）"
        return False, ""

    def should_take_profit_short(
        self,
        entry_price: float,
        current_price: float,
    ) -> tuple[bool, str]:
        """
        空头止盈：股价下跌超过 short_take_profit_pct 时触发
        """
        cfg = self.config
        gain_pct = (entry_price - current_price) / entry_price * 100
        if gain_pct >= cfg.short_take_profit_pct:
            return True, f"达到空头止盈线(盈利{gain_pct:.1f}%)"
        return False, ""

    # ---- 共通 ----

    def check_max_drawdown(
        self,
        peak_value: float,
        current_value: float,
    ) -> tuple[bool, float]:
        """检查是否触发最大回撤"""
        dd_pct = (peak_value - current_value) / peak_value * 100 if peak_value else 0
        return dd_pct >= self.config.max_drawdown_pct, dd_pct

    def get_buy_quantity(
        self,
        price: float,
        portfolio_value: float,
        risk_pct: float = None,
    ) -> float:
        pct = risk_pct or self.config.max_single_position_pct
        max_amount = portfolio_value * (pct / 100)
        return round(max_amount / price, 2)

    def get_crypto_stop_loss(self, entry_price: float, is_long: bool = True) -> float:
        """加密货币止损价（更宽，适用于 7x24 高波动）"""
        if is_long:
            return round(entry_price * (1 - self.config.crypto_stop_loss_pct / 100), 4)
        else:
            return round(entry_price * (1 + self.config.crypto_stop_loss_pct / 100), 4)

    def get_crypto_take_profit(self, entry_price: float, is_long: bool = True) -> float:
        """加密货币止盈价（更高目标）"""
        if is_long:
            return round(entry_price * (1 + self.config.crypto_take_profit_pct / 100), 4)
        else:
            return round(entry_price * (1 - self.config.crypto_take_profit_pct / 100), 4)

    def get_short_quantity(
        self,
        price: float,
        cash: float,
        risk_pct: float = None,
    ) -> float:
        """基于保证金计算可做空股数"""
        cfg = self.config
        pct = risk_pct or cfg.short_max_position_pct
        max_amount = cash * (pct / 100)  # 用现金的X%作为保证金上限
        margin_per_share = price * (cfg.short_margin_pct / 100)
        return round(max_amount / margin_per_share, 0)