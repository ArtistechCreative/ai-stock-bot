"""
参数扫描：找最优止损/止盈/信心组合
"""
import sys, os, json
sys.path.insert(0, 'bot')
import pandas as pd
from datetime import datetime, timedelta

from scorer import rank_stocks
from dl_strategy import batch_predict
from strategy_optimizer import StrategyOptimizer
from backtest import BacktestEngine

WATCHLIST = ['NVDA','TSLA','AMD','MSFT','META','AAPL','AMZN','GOOGL','JPM','V','UNH','XOM','JNJ','KO','DIS','NFLX','PLTR','COIN','SOFI']

start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

# 评分 + DL预测（只做一次）
ranked = rank_stocks(WATCHLIST, top_n=8)
top_tickers = [s['ticker'] for s in ranked[:5]]
preds = batch_predict(top_tickers, model_type='MLP')
dl_preds = {p['ticker']: p for p in preds if 'error' not in p}
dl_list = [{'ticker': t, 'signal': dl_preds[t]['signal'], 'confidence': dl_preds[t]['confidence']} for t in dl_preds]
print(f'DL signals: {[(d["ticker"], d["signal"], d["confidence"]) for d in dl_list]}')

# 固定策略函数（追踪止损 4% 高点，回撤 3%）
def make_strategy_fn(dl_list, combined, params):
    def strategy_fn(positions, cash, row, date_str):
        signals = []
        holdings = set(positions.keys())
        for ticker, pos in list(positions.items()):
            if ticker not in row or pd.isna(row[ticker]):
                continue
            price = float(row[ticker])
            entry = pos.get('avg_cost', price)
            high_since_entry = entry * 1.04
            drawdown = (high_since_entry - price) / high_since_entry
            if price < high_since_entry and drawdown > 0.03:
                signals.append({'action': 'SELL', 'ticker': ticker, 'shares': pos['shares'],
                                'reason': f'TRAIL_STOP d={drawdown*100:.1f}%'})

        candidates = [c for c in combined if c['ticker'] not in holdings]
        candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        slots = min(4, params.max_positions - len(holdings))
        bought = 0
        for candidate in candidates:
            if bought >= slots:
                break
            ticker = candidate['ticker']
            if ticker not in row or pd.isna(row[ticker]):
                continue
            price = float(row[ticker])
            dl_info = next((d for d in dl_list if d['ticker'] == ticker), None)
            if not dl_info or dl_info['signal'] != 'BUY':
                continue
            if dl_info['confidence'] < params.min_confidence:
                continue
            conf_ratio = (dl_info['confidence'] - params.min_confidence) / (100 - params.min_confidence)
            leverage = 0.6 + conf_ratio * 0.9
            adjusted_pct = min(params.max_position_pct / 100 * leverage, 0.5)
            position_value = cash * adjusted_pct
            shares = max(1, int(position_value / price))
            cost = shares * price
            if cost > 0 and cost <= cash * 0.85:
                signals.append({'action': 'BUY', 'ticker': ticker, 'shares': shares,
                                'reason': f'DL_conf={dl_info["confidence"]:.0f}%(x{leverage:.1f})'})
                bought += 1

        for ticker, pos in list(positions.items()):
            if ticker not in row or pd.isna(row[ticker]):
                continue
            dl_info = next((d for d in dl_list if d['ticker'] == ticker), None)
            if not dl_info or dl_info['signal'] != 'SELL':
                continue
            current_price = float(row[ticker])
            entry = pos.get('avg_cost', current_price)
            pnl_pct = (current_price - entry) / entry * 100
            if pnl_pct >= params.stop_loss_pct * 0.3:  # 盈亏平衡就出
                signals.append({'action': 'SELL', 'ticker': ticker, 'shares': pos['shares'],
                                'reason': f'DL反转 pnl={pnl_pct:.1f}%'})
        return signals
    return strategy_fn

# 参数网格
stop_losses = [4, 5, 6, 8]
take_profits = [8, 10, 12, 15]
min_confidences = [42, 45, 48, 50]

results = []
best = None

for sl in stop_losses:
    for tp in take_profits:
        for mc in min_confidences:
            state = {
                'params': {
                    'min_confidence': mc,
                    'stop_loss_pct': sl,
                    'take_profit_pct': tp,
                    'max_position_pct': 25.0,
                    'max_positions': 8,
                    'dl_weight': 0.6,
                    'score_weight': 0.4,
                }
            }
            optimizer = StrategyOptimizer(state_path=None)
            optimizer.params = type('P', (), {**state['params']})()
            score_list = [{'ticker': s['ticker'], 'score': s['score']} for s in ranked]
            combined = optimizer.get_signal(dl_list, score_list)
            strategy_fn = make_strategy_fn(dl_list, combined, optimizer.params)

            be = BacktestEngine(initial_cash=10000)
            result = be.run(
                tickers=WATCHLIST,
                strategy_fn=strategy_fn,
                start_date=start_date,
                end_date=end_date,
                stop_loss_pct=sl,
                take_profit_pct=tp,
            )
            if result:
                r = {
                    'sl': sl, 'tp': tp, 'mc': mc,
                    'return': result.total_return_pct,
                    'max_dd': result.max_drawdown_pct,
                    'sharpe': result.sharpe_ratio,
                    'win_rate': result.win_rate,
                    'trades': result.total_trades,
                    'final': result.final_value
                }
                results.append(r)
                print(f"SL={sl}% TP={tp}% MC={mc}% | ret={result.total_return_pct:+.2f}% dd={result.max_drawdown_pct:.2f}% sharpe={result.sharpe_ratio:.2f} win={result.win_rate:.0f}% trades={result.total_trades}")
                if best is None or result.total_return_pct > best['return']:
                    best = r

print(f'\n=== 最佳参数 ===')
for k, v in best.items():
    print(f'  {k}: {v}')