"""
投资组合管理：资金、多头/空头持仓、P&L 追踪
"""
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class Position:
    ticker: str
    shares: float          # 正=多头股数, 负=空头股数
    avg_cost: float       # 多头:买入均价 | 空头:卖出均价
    stop_loss: float = 0.0
    target: float = 0.0   # 多头止盈价 / 空头止损价
    entry_date: str = ""
    strategy: str = ""
    position_type: str = "LONG"   # "LONG" 或 "SHORT"


@dataclass
class Trade:
    date: str
    action: str           # BUY / SELL / SHORT / COVER
    ticker: str
    shares: float         # 股数（多头正，空头负表示卖空）
    price: float
    pnl: float = 0.0
    reason: str = ""
    position_type: str = "LONG"


class Portfolio:
    """支持多头+空头的投资组合"""

    def __init__(self, initial_cash: float = 10000.0, json_path: str = None):
        self.initial_cash = initial_cash
        self.path = Path(json_path) if json_path else None
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self._load()

    def _load(self):
        if self.path and self.path.exists():
            data = json.load(open(self.path))
            self.initial_cash = data.get("initial_cash", self.initial_cash)
            self.positions = {
                k: Position(**v) for k, v in data.get("positions", {}).items()
            }
            self.trades = [Trade(**t) for t in data.get("trades", [])]

    def save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "initial_cash": self.initial_cash,
                "positions": {k: asdict(v) for k, v in self.positions.items()},
                "trades": [asdict(t) for t in self.trades],
            }
            with open(self.path, "w") as f:
                json.dump(data, f, indent=2)

    def reset(self, new_cash: float = None):
        if new_cash:
            self.initial_cash = new_cash
        self.positions.clear()
        self.trades.clear()
        self.save()

    # ---- 多头操作 ----

    def buy(
        self,
        ticker: str,
        shares: float,
        price: float,
        stop_loss: float = 0,
        target: float = 0,
        strategy: str = "",
        reason: str = "",
    ):
        cost = shares * price
        if cost > self.cash:
            raise ValueError(f"现金不足: 需要${cost:.2f}, 只有${self.cash:.2f}")

        if ticker in self.positions:
            pos = self.positions[ticker]
            if pos.position_type != "LONG":
                raise ValueError(f"{ticker} 已有空头仓位，需先平仓")
            total_shares = pos.shares + shares
            new_avg = (pos.shares * pos.avg_cost + shares * price) / total_shares
            pos.shares = total_shares
            pos.avg_cost = new_avg
            pos.stop_loss = stop_loss or pos.stop_loss
            pos.target = target or pos.target
        else:
            self.positions[ticker] = Position(
                ticker=ticker,
                shares=shares,
                avg_cost=price,
                stop_loss=stop_loss,
                target=target,
                entry_date=datetime.now().strftime("%Y-%m-%d"),
                strategy=strategy,
                position_type="LONG",
            )

        self.trades.append(Trade(
            date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            action="BUY",
            ticker=ticker,
            shares=shares,
            price=price,
            reason=reason,
            position_type="LONG",
        ))
        self.save()

    def sell(self, ticker: str, shares: float, price: float, reason: str = ""):
        if ticker not in self.positions:
            raise ValueError(f"{ticker} 没有多头仓位")
        pos = self.positions[ticker]
        if pos.position_type != "LONG":
            raise ValueError(f"{ticker} 是空头仓位，应用 COVER 平仓")

        pnl = (price - pos.avg_cost) * min(shares, pos.shares)
        if shares >= pos.shares:
            # 全平
            self.trades.append(Trade(
                date=datetime.now().strftime("%Y-%m-%d %H:%M"),
                action="SELL",
                ticker=ticker,
                shares=pos.shares,
                price=price,
                pnl=pnl,
                reason=reason,
                position_type="LONG",
            ))
            del self.positions[ticker]
        else:
            # 部分平
            pos.shares -= shares
            self.trades.append(Trade(
                date=datetime.now().strftime("%Y-%m-%d %H:%M"),
                action="SELL",
                ticker=ticker,
                shares=shares,
                price=price,
                pnl=pnl,
                reason=reason,
                position_type="LONG",
            ))
        self.save()
        return pnl

    # ---- 空头操作 ----

    def short(
        self,
        ticker: str,
        shares: float,
        price: float,
        stop_loss: float = 0,  # 空头止损（价格超过此价买入平仓）
        target: float = 0,     # 空头止盈（价格跌到此价买入平仓）
        strategy: str = "",
        reason: str = "",
    ):
        """
        卖空：先借股票卖出，等价格下跌再买回来平仓
        利润 = 卖出价 - 买入价（价差）
        保证金要求 = 卖出总价值（通常需要50%保证金）
        """
        proceeds = shares * price
        # 保证金：空头需要冻结保证金（通常是仓位价值的50%）
        margin_required = proceeds * 0.5
        if margin_required > self.cash:
            raise ValueError(f"保证金不足: 需要${margin_required:.2f}, 只有${self.cash:.2f}")

        if ticker in self.positions:
            raise ValueError(f"{ticker} 已有仓位，需先平仓")

        self.positions[ticker] = Position(
            ticker=ticker,
            shares=shares,         # 空头股数为正（代表卖出数量）
            avg_cost=price,        # 卖出开仓价格
            stop_loss=stop_loss,   # 空头止损价（比开仓价高）
            target=target,        # 空头止盈价（比开仓价低）
            entry_date=datetime.now().strftime("%Y-%m-%d"),
            strategy=strategy,
            position_type="SHORT",
        )

        self.trades.append(Trade(
            date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            action="SHORT",
            ticker=ticker,
            shares=shares,
            price=price,
            reason=reason,
            position_type="SHORT",
        ))
        self.save()

    def cover(
        self,
        ticker: str,
        shares: float,
        price: float,
        reason: str = "",
    ):
        """买回股票平仓空头"""
        if ticker not in self.positions:
            raise ValueError(f"{ticker} 没有空头仓位")
        pos = self.positions[ticker]
        if pos.position_type != "SHORT":
            raise ValueError(f"{ticker} 是多头仓位，应用 SELL 平仓")

        # 空头盈亏 = 卖出开仓价 - 买回平仓价（价差为正=盈利）
        cost = shares * price
        pnl = (pos.avg_cost - price) * min(shares, pos.shares)

        if shares >= pos.shares:
            self.trades.append(Trade(
                date=datetime.now().strftime("%Y-%m-%d %H:%M"),
                action="COVER",
                ticker=ticker,
                shares=pos.shares,
                price=price,
                pnl=pnl,
                reason=reason,
                position_type="SHORT",
            ))
            del self.positions[ticker]
        else:
            pos.shares -= shares
            self.trades.append(Trade(
                date=datetime.now().strftime("%Y-%m-%d %H:%M"),
                action="COVER",
                ticker=ticker,
                shares=shares,
                price=price,
                pnl=pnl,
                reason=reason,
                position_type="SHORT",
            ))
        self.save()
        return pnl

    # ---- 持仓保证金（空头）----

    @property
    def cash(self) -> float:
        """可用现金（扣除多头占用 + 空头保证金）"""
        long_used = sum(p.shares * p.avg_cost for p in self.positions.values() if p.position_type == "LONG")
        short_margin = sum(p.shares * p.avg_cost * 0.5 for p in self.positions.values() if p.position_type == "SHORT")
        return self.initial_cash + self.realized_pnl - long_used - short_margin

    @property
    def realized_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    def portfolio_summary(self, live_prices: dict[str, float] = None) -> dict:
        """组合汇总（支持多头+空头）"""
        positions_out = []
        total_market_value = 0.0
        total_unrealized_pnl = 0.0
        long_exposure = 0.0
        short_exposure = 0.0

        for ticker, pos in self.positions.items():
            price = live_prices.get(ticker, pos.avg_cost) if live_prices else pos.avg_cost

            if pos.position_type == "LONG":
                market_value = pos.shares * price
                cost = pos.shares * pos.avg_cost
                pnl = market_value - cost
                pnl_pct = (price / pos.avg_cost - 1) * 100
                unrealized = pnl
                exposure = cost
            else:  # SHORT
                # 空头市值 = 股数 * 当前价（表示需要多少买回）
                market_value = pos.shares * price
                # 空头占用保证金 = 卖出时收到的钱 + 50%额外保证金
                margin_posted = pos.shares * pos.avg_cost * 1.5  # 卖出收入 + 50%
                pnl = (pos.avg_cost - price) * pos.shares   # 价差盈利
                pnl_pct = (pos.avg_cost / price - 1) * 100
                unrealized = pnl
                exposure = margin_posted   # 空头用保证金计算暴露

            positions_out.append({
                "ticker": ticker,
                "type": pos.position_type,
                "shares": pos.shares,
                "avg_cost": pos.avg_cost,
                "current_price": price,
                "market_value": round(market_value, 2),
                "pnl": round(unrealized, 2),
                "pnl_pct": round(pnl_pct, 2),
                "stop_loss": pos.stop_loss,
                "target": pos.target,
                "strategy": pos.strategy,
                "entry_date": pos.entry_date,
            })
            total_market_value += market_value
            total_unrealized_pnl += unrealized
            if pos.position_type == "LONG":
                long_exposure += exposure
            else:
                short_exposure += exposure

        total_pnl = self.realized_pnl + total_unrealized_pnl
        total_pnl_pct = (total_pnl / self.initial_cash) * 100

        return {
            "initial_cash": self.initial_cash,
            "cash": round(self.cash, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(total_unrealized_pnl, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "positions": positions_out,
            "long_exposure": round(long_exposure, 2),
            "short_exposure": round(short_exposure, 2),
            "total_exposure": round(long_exposure + short_exposure, 2),
        }
