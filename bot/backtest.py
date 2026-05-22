"""
回测引擎：历史数据回测 + 回撤分析
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
import json

DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class BacktestResult:
    start_date: str
    end_date: str
    initial_cash: float
    final_value: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    total_trades: int
    win_rate: float
    avg_holding_days: float
    trades: list
    n_take_profit: int = 0
    n_stop_loss: int = 0
    n_trailing_stop: int = 0    # 追踪止损触发次数
    take_profit_rate: float = 0.0
    stop_loss_rate: float = 0.0


class BacktestEngine:
    def __init__(self, initial_cash: float = 10000):
        self.initial_cash = initial_cash

    def run(
        self,
        tickers: list[str],
        strategy_fn,           # func(portfolio, prices, date) -> action dict
        start_date: str = None,
        end_date: str = None,
        stop_loss_pct: float = 8.0,
        take_profit_pct: float = 15.0,
        trailing_trigger_pct: float = 5.0,  # 涨超5%后激活追踪止损
        trailing_stop_pct: float = 3.0,     # 激活后从高点回撤3%即出
    ) -> BacktestResult:
        """
        跑回测
        strategy_fn: (portfolio, date, prices_df) -> list of {action:'BUY/SELL/HOLD', ticker, shares}
        """
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        start_date = start_date or (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

        # 下载历史数据
        print(f"📥 下载历史数据 {start_date} → {end_date}...")
        price_data = {}
        for t in tickers:
            try:
                df = yf.download(t, start=start_date, end=end_date, progress=False)
                if len(df) > 20:
                    price_data[t] = df["Close"]
            except Exception as e:
                print(f"  [!] {t}: {e}")

        if not price_data:
            print("❌ 没有下载到数据")
            return None

        # 构建每日价格矩阵
        all_dates = sorted(set().union(*[set(p.index) for p in price_data.values()]))
        prices_df = pd.DataFrame(index=all_dates)
        for t, s in price_data.items():
            prices_df[t] = s

        prices_df = prices_df.dropna()
        print(f"   数据日期: {prices_df.index[0].strftime('%Y-%m-%d')} → {prices_df.index[-1].strftime('%Y-%m-%d')}, 共 {len(prices_df)} 天")

        # 模拟组合
        cash = self.initial_cash
        # 持仓：ticker -> {shares, avg_cost, entry_date, stop, target, high_price}
        positions = {}
        short_positions = {}  # 做空持仓：ticker -> {shares(正数), avg_cost, entry_date, stop, target, low_price}
        trades = []
        daily_values = []
        peak_value = self.initial_cash

        # 持仓记录
        entry_prices = {}

        for date in prices_df.index:
            date_str = date.strftime("%Y-%m-%d")
            row = prices_df.loc[date]
            portfolio_value = cash + sum(
                row[t] * positions[t]["shares"]
                for t in positions if t in row
            )
            peak_value = max(peak_value, portfolio_value)
            daily_values.append({
                "date": date_str,
                "value": portfolio_value,
                "peak": peak_value,
                "drawdown_pct": (peak_value - portfolio_value) / peak_value * 100 if peak_value else 0
            })

            # 止损检查（对每支持仓）
            for ticker in list(positions.keys()):
                if ticker not in row or np.isnan(row[ticker]):
                    continue
                pos = positions[ticker]
                current_price = row[ticker]
                entry = pos["avg_cost"]
                stop = pos["stop"]
                target = pos["target"]
                high_price = pos.get("high_price", entry)

                # 更新高点（追踪止损用）
                if current_price > high_price:
                    pos["high_price"] = current_price
                    high_price = current_price

                gain_pct = (current_price - entry) / entry * 100

                # ── 追踪止损（trailing stop）─────────────────────────────
                # 激活条件：浮盈 ≥ trailing_trigger_pct（默认5%）
                # 触发后：从高点回撤 trailing_stop_pct（默认3%）即出
                trailing_activated = gain_pct >= trailing_trigger_pct
                if trailing_activated:
                    # 计算从高点回撤了多少 %
                    drawback_from_high = (high_price - current_price) / high_price * 100
                    if drawback_from_high >= trailing_stop_pct:
                        pnl = (current_price - entry) * pos["shares"]
                        cash += current_price * pos["shares"]
                        trades.append({"date": date_str, "ticker": ticker, "action": "SELL",
                                       "reason": "TRAILING_STOP",
                                       "price": current_price, "pnl": round(pnl, 2),
                                       "return_pct": round(gain_pct, 2)})
                        del positions[ticker]
                        entry_prices.pop(ticker, None)
                        continue

                # ── 固定止损 ─────────────────────────────────────────
                if current_price <= stop:
                    pnl = (current_price - entry) * pos["shares"]
                    cash += current_price * pos["shares"]
                    trades.append({"date": date_str, "ticker": ticker, "action": "SELL", "reason": "STOP_LOSS", "price": current_price, "pnl": round(pnl, 2), "return_pct": round(pnl/entry/pos["shares"]*100, 2)})
                    del positions[ticker]
                    entry_prices.pop(ticker, None)
                    continue

                # ── 止盈 ─────────────────────────────────────────────
                if gain_pct >= take_profit_pct:
                    pnl = (current_price - entry) * pos["shares"]
                    cash += current_price * pos["shares"]
                    trades.append({"date": date_str, "ticker": ticker, "action": "SELL", "reason": "TAKE_PROFIT", "price": current_price, "pnl": round(pnl, 2), "return_pct": round(gain_pct, 2)})
                    del positions[ticker]
                    entry_prices.pop(ticker, None)
                    continue

            # ── 做空仓位自动止损/止盈检查 ──────────────────────────────
            for ticker in list(short_positions.keys()):
                if ticker not in row or np.isnan(row[ticker]):
                    continue
                pos = short_positions[ticker]
                current_price = row[ticker]
                entry = pos["avg_cost"]
                stop = pos["stop"]
                target = pos["target"]
                low_price = pos.get("low_price", entry)

                # 更新最低点（追踪最高点用，做空从高点回落）
                if current_price < low_price:
                    pos["low_price"] = current_price
                    low_price = current_price

                # 做空盈亏：方向相反
                gain_pct = (entry - current_price) / entry * 100  # 正数 = 赚钱

                # ── 追踪止损（做空方向反转）────────────────────────────
                trailing_activated = gain_pct >= trailing_trigger_pct
                if trailing_activated:
                    rise_from_low = (current_price - low_price) / low_price * 100
                    if rise_from_low >= trailing_stop_pct:
                        pnl = (entry - current_price) * pos["shares"]
                        # 平空：花 current_price 买回，之前收了 entry * shares
                        cash += entry * pos["shares"] - pnl
                        trades.append({"date": date_str, "ticker": ticker, "action": "COVER",
                                       "reason": "TRAILING_STOP",
                                       "price": current_price, "pnl": round(pnl, 2),
                                       "return_pct": round(gain_pct, 2)})
                        del short_positions[ticker]
                        continue

                # ── 做空止损（价格向不利的方向移动）──────────────────
                if current_price >= stop:
                    pnl = (entry - current_price) * pos["shares"]
                    cash += entry * pos["shares"] - pnl
                    trades.append({"date": date_str, "ticker": ticker, "action": "COVER", "reason": "STOP_LOSS", "price": current_price, "pnl": round(pnl, 2), "return_pct": round((entry - current_price) / entry * 100, 2)})
                    del short_positions[ticker]
                    continue

                # ── 做空止盈（价格向有利的方向移动）──────────────────
                if gain_pct >= take_profit_pct:
                    pnl = (entry - current_price) * pos["shares"]
                    cash += entry * pos["shares"] - pnl
                    trades.append({"date": date_str, "ticker": ticker, "action": "COVER", "reason": "TAKE_PROFIT", "price": current_price, "pnl": round(pnl, 2), "return_pct": round(gain_pct, 2)})
                    del short_positions[ticker]
                    continue

            # 调用策略生成信号
            try:
                signals = strategy_fn(positions, cash, row, date_str)
                if signals:
                    for sig in signals:
                        action = sig.get("action")
                        ticker = sig.get("ticker")
                        if action == "BUY" and ticker in row and not np.isnan(row[ticker]):
                            price = row[ticker]
                            shares = sig.get("shares", 0)
                            cost = shares * price
                            if cost > 0 and cost <= cash * 0.9:
                                cash -= cost
                                positions[ticker] = {
                                    "shares": shares,
                                    "avg_cost": price,
                                    "stop": round(price * (1 - stop_loss_pct/100), 2),
                                    "target": round(price * (1 + take_profit_pct/100), 2),
                                    "entry_date": date_str,
                                    "high_price": price,  # 追踪止损用
                                }
                                trades.append({"date": date_str, "ticker": ticker, "action": "BUY", "price": price, "shares": shares, "reason": sig.get("reason", "")})
                        elif action == "SHORT" and ticker in row and not np.isnan(row[ticker]):
                            price = row[ticker]
                            shares = sig.get("shares", 0)
                            proceeds = shares * price
                            # 做空：收取卖出收入，保证金占用等量资金
                            if proceeds > 0 and proceeds <= cash * 0.45:
                                cash += proceeds  # 收到卖空收入
                                short_positions[ticker] = {
                                    "shares": shares,
                                    "avg_cost": price,   # 做空开仓价
                                    "stop": round(price * (1 + stop_loss_pct/100), 2),  # 做空止损价（更高）
                                    "target": round(price * (1 - take_profit_pct/100), 2),  # 做空止盈价（更低）
                                    "entry_date": date_str,
                                    "low_price": price,  # 追踪最高点用的"高点"
                                }
                                trades.append({"date": date_str, "ticker": ticker, "action": "SHORT", "price": price, "shares": shares, "reason": sig.get("reason", "")})
                        elif action == "SELL" and ticker in positions:
                            # 平掉多头仓位
                            pos = positions[ticker]
                            current_price = row[ticker]
                            pnl = (current_price - pos["avg_cost"]) * pos["shares"]
                            cash += current_price * pos["shares"]
                            trades.append({"date": date_str, "ticker": ticker, "action": "SELL", "price": current_price, "pnl": round(pnl, 2), "return_pct": round((current_price - pos["avg_cost"]) / pos["avg_cost"] * 100, 2), "reason": sig.get("reason", "MANUAL")})
                            del positions[ticker]
                            entry_prices.pop(ticker, None)
                        elif action == "COVER" and ticker in short_positions:
                            # 平掉做空仓位
                            pos = short_positions[ticker]
                            current_price = row[ticker]
                            pnl = (pos["avg_cost"] - current_price) * pos["shares"]  # 做空盈利 = 卖出价 - 买入价
                            cash += pos["avg_cost"] * pos["shares"] - pnl  # 还入保证金 + 利润
                            trades.append({"date": date_str, "ticker": ticker, "action": "COVER", "price": current_price, "pnl": round(pnl, 2), "return_pct": round((pos["avg_cost"] - current_price) / pos["avg_cost"] * 100, 2), "reason": sig.get("reason", "MANUAL")})
                            del short_positions[ticker]
            except Exception as e:
                import traceback
                traceback.print_exc()

            # 计算当天收盘后的组合价值（用于回撤追踪）
            try:
                long_value = sum(
                    row[t] * positions[t]["shares"]
                    for t in positions if t in row
                )
                # 做空仓位价值：持仓成本 - 当前市值（做空盈利时价值为负）
                short_value = sum(
                    (pos["avg_cost"] - row[t]) * pos["shares"]
                    for t, pos in short_positions.items() if t in row
                )
                portfolio_value = cash + long_value + short_value
            except Exception:
                portfolio_value = cash

            peak_value = max(peak_value, portfolio_value)
            daily_values.append({
                "date": date_str,
                "value": portfolio_value,
                "peak": peak_value,
                "drawdown_pct": (peak_value - portfolio_value) / peak_value * 100 if peak_value else 0
            })

        # 计算最终结果（循环结束后）
        last_row = prices_df.iloc[-1]
        final_value = cash
        # 多头价值
        for t, pos in positions.items():
            if t in last_row.index and not pd.isna(last_row[t]):
                final_value += last_row[t] * pos["shares"]
            else:
                if t in prices_df.columns:
                    last_price = prices_df[t].iloc[-1]
                    if not pd.isna(last_price):
                        final_value += last_price * pos["shares"]
        # 做空价值：盈利 = (entry - current) * shares
        for t, pos in short_positions.items():
            if t in last_row.index and not pd.isna(last_row[t]):
                final_value += (pos["avg_cost"] - last_row[t]) * pos["shares"]
            else:
                if t in prices_df.columns:
                    last_price = prices_df[t].iloc[-1]
                    if not pd.isna(last_price):
                        final_value += (pos["avg_cost"] - last_price) * pos["shares"]

        total_return = (final_value - self.initial_cash) / self.initial_cash * 100

        # 回撤分析
        daily_df = pd.DataFrame(daily_values)
        max_dd = daily_df["drawdown_pct"].max() if len(daily_df) > 0 else 0

        # Sharpe ratio（简化：用日收益率 std）
        if len(daily_df) > 2:
            daily_df["return_pct"] = daily_df["value"].pct_change() * 100
            returns = daily_df["return_pct"].dropna()
            sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        else:
            sharpe = 0

        # 交易统计（多头 SELL + 做空 COVER 合并统计）
        exit_trades = [t for t in trades if t["action"] in ("SELL", "COVER")]
        win_trades = [t for t in exit_trades if t.get("pnl", 0) > 0]
        n_take_profit = len([t for t in exit_trades if t.get("reason") == "TAKE_PROFIT"])
        n_stop_loss = len([t for t in exit_trades if t.get("reason") == "STOP_LOSS"])
        n_trailing = len([t for t in exit_trades if t.get("reason") == "TRAILING_STOP"])
        total_exits = len(exit_trades) or 1

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_cash=self.initial_cash,
            final_value=round(final_value, 2),
            total_return_pct=round(total_return, 2),
            max_drawdown_pct=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 2),
            total_trades=len(trades),
            win_rate=round(len(win_trades)/total_exits*100, 1),
            avg_holding_days=0,
            trades=trades,
            n_take_profit=n_take_profit,
            n_stop_loss=n_stop_loss,
            n_trailing_stop=n_trailing,
            take_profit_rate=round(n_take_profit / total_exits, 3),
            stop_loss_rate=round(n_stop_loss / total_exits, 3),
        )

    def summary_text(self, result: BacktestResult) -> str:
        """生成回测报告文字版"""
        return f"""📊 回测报告：{result.start_date} → {result.end_date}

💰 初始资金: ${result.initial_cash:,.0f}
📈 最终价值: ${result.final_value:,.2f}
📊 总收益率: {'+' if result.total_return_pct >= 0 else ''}{result.total_return_pct}%

📉 最大回撤: {result.max_drawdown_pct}%
⚡ Sharpe: {result.sharpe_ratio}
📋 总交易: {result.total_trades} 笔
🏆 胜率: {result.win_rate}%

⚠️ 回测结果仅供参考，不构成投资建议。
"""