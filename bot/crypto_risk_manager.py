"""
加密货币风险管理器 — 支持做多 + 做空 + 高杠杆
- 杠杆仓位保证金管理
- 强平价计算
- ATR 动态止损（加密货币波动大）
- 资金费率成本计算
- 每笔仓位最大风险控制
"""
from dataclasses import dataclass
from typing import Optional

from crypto_data import PERP_INFO


@dataclass
class CryptoRiskConfig:
    """加密货币风险管理参数"""
    # === 仓位控制 ===
    max_single_position_pct: float = 10.0     # 单笔最大仓位（%总资金，保守10%）
    max_total_exposure_pct: float = 60.0      # 总持仓上限（%总资金）

    # === 杠杆（加密货币高杠杆） ===
    default_leverage: int = 10                # 默认杠杆（保守10x，用户可调）
    max_leverage: int = 75                    # 交易所允许最大杠杆（BTC 75x）
    allow_high_leverage: bool = False         # 是否允许 >20x 杠杆

    # === 止损/止盈（ATR 动态） ===
    stop_loss_atr_multiplier: float = 2.0     # 止损 = entry ± ATR × multiplier
    take_profit_atr_multiplier: float = 4.0   # 止盈 = entry ± ATR × multiplier
    stop_loss_default_pct: float = 5.0        # 默认止损（%，当无ATR数据时）
    profit_taking_pct: float = 10.0           # 止盈（%）

    # === 短线规则 ===
    max_positions: int = 5                     # 最大同时持仓（含多空）
    min_holding_hours: int = 1                # 最短持有（小时）
    max_holding_hours: int = 48               # 最长持有（小时，2天）
    max_trades_per_day: int = 6               # 每天最大交易次数

    # === 空头专属 ===
    short_margin_pct: float = 50.0            # 空头保证金要求（%开仓价值，交易所标准）
    short_max_position_pct: float = 10.0     # 空头单笔最大仓位
    allow_short: bool = True                  # 允许做空

    # === 回撤限制 ===
    max_drawdown_pct: float = 15.0           # 最大回撤（触发强平警告）

    # === 资金费率成本 ===
    funding_cost_budget_pct: float = 1.0      # 每24小时资金费率成本上限（%资金）
    max_funding_rate: float = 0.001           # 允许的最高资金费率（0.1%）

    def total_short_margin_required(self, price: float, size: float) -> float:
        """空头占用保证金 = 合约数量 × 价格 × 保证金%"""
        return size * price * (self.short_margin_pct / 100)


@dataclass
class CryptoRiskStatus:
    can_open_long: bool = True
    can_open_short: bool = True
    reason: str = ""
    positions_count: int = 0
    long_exposure_pct: float = 0.0
    short_exposure_pct: float = 0.0
    portfolio_value: float = 0.0
    daily_pnl_pct: float = 0.0


class CryptoRiskManager:
    """
    加密货币风控系统
    支持做多、做空、高杠杆、强平计算
    """

    def __init__(self, config: CryptoRiskConfig = None):
        self.config = config or CryptoRiskConfig()

    # ---- 强平价格计算 ----

    def calc_liquidation_price(
        self,
        entry_price: float,
        side: str,          # "LONG" or "SHORT"
        leverage: int,
        maint_margin_ratio: float = 0.005,  # 维持保证金率 0.5%（行业标准）
    ) -> float:
        """
        计算强平价格
        多头：entry_price × (1 - 1/leverage × (1 - maint_ratio))
        空头：entry_price × (1 + 1/leverage × (1 - maint_ratio))
        """
        if leverage <= 0:
            return 0
        if side == "LONG":
            return entry_price * (1 - (1 / leverage) * (1 - maint_margin_ratio))
        else:
            return entry_price * (1 + (1 / leverage) * (1 - maint_margin_ratio))

    # ---- ATR 动态止损/止盈 ----

    def calc_stop_loss(
        self,
        entry_price: float,
        atr: float,
        atr_pct: float,
        side: str,
        use_atr: bool = True,
    ) -> float:
        """
        计算止损价
        - ATR 模式（默认）：根据市场波动率
        - 回撤%模式（无ATR）：entry × (1 - stop_loss_pct)
        """
        if use_atr and atr > 0:
            multiplier = self.config.stop_loss_atr_multiplier
            if side == "LONG":
                return round(entry_price - atr * multiplier, 4)
            else:
                return round(entry_price + atr * multiplier, 4)
        else:
            pct = self.config.stop_loss_default_pct / 100
            if side == "LONG":
                return round(entry_price * (1 - pct), 4)
            else:
                return round(entry_price * (1 + pct), 4)

    def calc_take_profit(
        self,
        entry_price: float,
        atr: float,
        atr_pct: float,
        side: str,
        use_atr: bool = True,
    ) -> float:
        """计算止盈价"""
        if use_atr and atr > 0:
            multiplier = self.config.take_profit_atr_multiplier
            if side == "LONG":
                return round(entry_price + atr * multiplier, 4)
            else:
                return round(entry_price - atr * multiplier, 4)
        else:
            pct = self.config.profit_taking_pct / 100
            if side == "LONG":
                return round(entry_price * (1 + pct), 4)
            else:
                return round(entry_price * (1 - pct), 4)

    # ---- 做多风控 ----

    def can_open_long(
        self,
        price: float,
        portfolio_value: float,
        current_positions: dict,
        daily_trades: int = 0,
    ) -> tuple[bool, str, float]:
        cfg = self.config
        long_count = sum(1 for p in current_positions.values() if p.get("position_type") != "SHORT")
        if long_count >= cfg.max_positions:
            return False, f"多头已达最大持仓数({cfg.max_positions})", 0
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
        """多头止损"""
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

    # ---- 做空风控 ----

    def can_open_short(
        self,
        price: float,
        size: float,
        portfolio_value: float,
        cash: float,
        current_positions: dict,
        daily_trades: int = 0,
    ) -> tuple[bool, str, float]:
        cfg = self.config
        if not cfg.allow_short:
            return False, "系统禁止做空", 0
        short_count = sum(1 for p in current_positions.values() if p.get("position_type") == "SHORT")
        if short_count >= cfg.max_positions:
            return False, f"空头已达最大持仓数({cfg.max_positions})", 0
        if daily_trades >= cfg.max_trades_per_day:
            return False, f"今日交易次数已达上限({cfg.max_trades_per_day})", 0
        margin_required = cfg.total_short_margin_required(price, size)
        if margin_required > cash * 0.5:
            return False, f"保证金不足({margin_required/cash*100:.0f}%cash)", 0
        return True, "OK", size

    def should_stop_loss_short(
        self,
        entry_price: float,
        current_price: float,
    ) -> tuple[bool, str]:
        """空头止损：股价上涨超过配置%时触发"""
        loss_pct = (current_price - entry_price) / entry_price * 100
        if loss_pct >= self.config.stop_loss_default_pct:
            return True, f"触发空头止损（价格上涨{loss_pct:.1f}%）"
        return False, ""

    def should_take_profit_short(
        self,
        entry_price: float,
        current_price: float,
    ) -> tuple[bool, str]:
        """空头止盈：股价下跌超过配置%时触发"""
        gain_pct = (entry_price - current_price) / entry_price * 100
        if gain_pct >= self.config.profit_taking_pct:
            return True, f"达到空头止盈线(盈利{gain_pct:.1f}%)"
        return False, ""

    # ---- 资金费率成本 ----

    def calc_funding_cost(
        self,
        price: float,
        size: float,
        funding_rate: float,  # 8小时费率（正数=多头付钱，空头收钱）
        intervals: int = 3,   # 计算几个小时（默认3个周期=24小时）
    ) -> float:
        """
        计算资金费率成本（每小时）
        资金费率 = price × size × rate（每小时）
        做多：正资金费率是成本
        做空：负资金费率是成本（空头付钱给多头）
        """
        return price * size * funding_rate * intervals  # 3个8小时周期=24小时

    def check_funding_affordable(
        self,
        price: float,
        size: float,
        funding_rate: float,
        portfolio_value: float,
    ) -> tuple[bool, str]:
        """检查资金费率是否在预算内"""
        cost_24h = self.calc_funding_cost(price, size, funding_rate, intervals=3)
        cost_pct = cost_24h / portfolio_value * 100
        if cost_pct > self.config.funding_cost_budget_pct:
            return False, f"资金费率成本过高({cost_pct:.2f}% > {self.config.funding_cost_budget_pct}%)"
        return True, "OK"

    # ---- 通用 ----

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
        return round(max_amount / price, 4)

    def get_short_quantity(
        self,
        price: float,
        cash: float,
        risk_pct: float = None,
    ) -> float:
        """基于保证金计算可做空合约数"""
        cfg = self.config
        pct = risk_pct or cfg.short_max_position_pct
        max_margin = cash * (pct / 100)
        margin_per_contract = price * (cfg.short_margin_pct / 100)
        return round(max_margin / margin_per_contract, 4)

    def validate_leverage(
        self,
        leverage: int,
        symbol: str = None,
    ) -> int:
        """确保杠杆在允许范围内"""
        cfg = self.config
        max_leverage = cfg.max_leverage
        if symbol:
            perp_info = PERP_INFO.get(symbol, {})
            max_leverage = min(max_leverage, perp_info.get("leverage", 75))
        if not cfg.allow_high_leverage and leverage > 20:
            leverage = min(leverage, 20)
        return min(leverage, max_leverage)

    def get_position_size_in_contracts(
        self,
        price: float,
        portfolio_value: float,
        leverage: int,
        risk_pct: float = None,
    ) -> float:
        """
        根据杠杆计算开仓合约数量
        - 仓位价值 = 保证金 × 杠杆
        - 合约数 = 仓位价值 / 价格
        """
        cfg = self.config
        pct = risk_pct or cfg.max_single_position_pct
        margin = portfolio_value * (pct / 100)  # 保证金
        position_value = margin * leverage        # 仓位价值
        return round(position_value / price, 4)  # 合约数量