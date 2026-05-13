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
        positions = {}  # ticker -> {shares, avg_cost, entry_date, stop, target}
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

                # 止损
                if current_price <= stop:
                    pnl = (current_price - entry) * pos["shares"]
                    cash += current_price * pos["shares"]
                    trades.append({"date": date_str, "ticker": ticker, "action": "SELL", "reason": "STOP_LOSS", "price": current_price, "pnl": round(pnl, 2), "return_pct": round(pnl/entry/pos["shares"]*100, 2)})
                    del positions[ticker]
                    del entry_prices[ticker]
                    continue

                # 止盈
                gain_pct = (current_price - entry) / entry * 100
                if gain_pct >= take_profit_pct:
                    pnl = (current_price - entry) * pos["shares"]
                    cash += current_price * pos["shares"]
                    trades.append({"date": date_str, "ticker": ticker, "action": "SELL", "reason": "TAKE_PROFIT", "price": current_price, "pnl": round(pnl, 2), "return_pct": round(gain_pct, 2)})
                    del positions[ticker]
                    del entry_prices[ticker]
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
                                }
                                trades.append({"date": date_str, "ticker": ticker, "action": "BUY", "price": price, "shares": shares, "reason": sig.get("reason", "")})
            except Exception as e:
                pass  # 策略出错不影响回测

        # 计算结果
        final_value = cash + sum(
            prices_df.iloc[-1][t] * positions[t]["shares"]
            for t in positions if t in prices_df.iloc[-1]
        )
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

        # 交易统计
        sell_trades = [t for t in trades if t["action"] == "SELL"]
        win_trades = [t for t in sell_trades if t.get("pnl", 0) > 0]

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_cash=self.initial_cash,
            final_value=round(final_value, 2),
            total_return_pct=round(total_return, 2),
            max_drawdown_pct=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 2),
            total_trades=len(trades),
            win_rate=round(len(win_trades)/len(sell_trades)*100, 1) if sell_trades else 0,
            avg_holding_days=0,  # 可加
            trades=trades,
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