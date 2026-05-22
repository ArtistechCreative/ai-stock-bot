"""
180天短线回测脚本（优化版）
用法: python test_180day.py
"""
import sys, os, json
sys.path.insert(0, 'bot')
import pandas as pd

# 设置短线参数
state_path = 'data/strategy_state.json'
with open(state_path) as f:
    s = json.load(f)
s['params']['min_confidence'] = 45.0
s['params']['stop_loss_pct'] = 5.0
s['params']['take_profit_pct'] = 8.0
s['params']['max_position_pct'] = 25.0
s['params']['max_positions'] = 8
s['params']['dl_weight'] = 0.6
with open(state_path, 'w') as f:
    json.dump(s, f, indent=2)
print('Params: stop=5%, profit=8%, min_conf=45%, max_pos=8')

from scorer import rank_stocks
from dl_strategy import batch_predict
from strategy_optimizer import StrategyOptimizer
from backtest import BacktestEngine
from datetime import datetime, timedelta

STOCK_WATCHLIST = [
    "NVDA","MSFT","AMD","META","AAPL","AMZN","GOOGL","GOOG","TSLA","NFLX",
    "AVGO","CRM","ORCL","CSCO","ADBE","ACN","IBM","INTC","QCOM","TXN","MU","LRCX",
    "JPM","V","MA","BAC","WFC","GS","MS","BLK","AXP","C",
    "SCHW","SPGI","MCO","CME","ICE","USB","PNC","TFC","COF","ADP","PLTR",
    "LLY","UNH","JNJ","ABBV","MRK","PFE","ABT","TMO","DHR","BMY",
    "AMGN","GILD","VRTX","REGN","ISRG","MDT","SYK","ZTS","BSX","EW",
    "CAT","DE","HON","GE","RTX","LMT","NOC","BA","UPS","FDX",
    "XOM","CVX","COP","SLB","EOG","MPC","VLO","PSX","OXY","CTAS",
    "PG","KO","PEP","COST","WMT","HD","MCD","SBUX","NKE","DIS",
    "CMCSA","VZ","T","TMUS","CHTR","EA","TTWO","LEN","DRI","NEE",
    "EXC","AEP","ORLY","AZO","ROST","BKR","PCAR","EL",
]
WATCHLIST = STOCK_WATCHLIST

start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')
print(f'Backtest: {start_date} → {end_date} (180天)')

# 评分：对全部 WATCHLIST 排名
ranked = rank_stocks(WATCHLIST, top_n=8)

# DL 预测：只对有模型的 ticker 做预测
trained_tickers = [
    "AAPL","AMD","AMZN","COIN","DIS","GOOGL","JNJ","JPM","KO",
    "META","MSFT","NFLX","NVDA","PLTR","SOFI","TSLA","UNH","V","XOM"
]
available = [t for t in trained_tickers if t in WATCHLIST]
print(f"  DL模型覆盖: {len(available)}/{len(WATCHLIST)} 支")
preds = batch_predict(available, model_type='MLP')
dl_preds = {p['ticker']: p for p in preds if 'error' not in p}

# 构建 DL 信号列表（有模型 + 有预测结果的 ticker）
dl_list = [
    {'ticker': t, 'signal': dl_preds[t]['signal'], 'confidence': dl_preds[t]['confidence']}
    for t in dl_preds
]
print(f'  DL signals: {[(d["ticker"], d["signal"], d["confidence"]) for d in dl_list]}')

optimizer = StrategyOptimizer(state_path=state_path)
params = optimizer.params

# 评分排名：基于全股票池
score_list = [{'ticker': s['ticker'], 'score': s['score']} for s in ranked]

# 综合信号：DL + 评分
combined = optimizer.get_signal(dl_list, score_list)

# ── 短线策略函数 ────────────────────────────────────────────
def strategy_fn(positions, cash, row, date_str):
    """
    激进短线策略：
    - 追踪止损（回撤超过持仓均价 5% 就出）
    - 快速止盈 8%（backtest引擎自动触发）
    - DL BUY 信号 + 信心 > min_conf → 买入
    - 每次最多加 4 支，总持仓不超过 8 支
    - DL 反转 SELL → 有利润（>=2%）才出
    """
    signals = []
    holdings = set(positions.keys())

    # 1. 持仓：追踪止损
    for ticker, pos in list(positions.items()):
        if ticker not in row or pd.isna(row[ticker]):
            continue
        price = float(row[ticker])
        entry = pos.get('avg_cost', price)
        high_since_entry = entry * 1.04
        drawdown = (high_since_entry - price) / high_since_entry
        if price < high_since_entry and drawdown > 0.05:
            signals.append({
                'action': 'SELL', 'ticker': ticker,
                'shares': pos['shares'],
                'reason': f'TRAIL_STOP drawdown={drawdown*100:.1f}%'
            })

    # 2. 买入：DL BUY 信号 + 信心达标
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

        # 找 DL 信号（只在有模型的 ticker 中找）
        dl_info = next((d for d in dl_list if d['ticker'] == ticker), None)
        if not dl_info:
            continue
        if dl_info['signal'] != 'BUY':
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
            signals.append({
                'action': 'BUY', 'ticker': ticker, 'shares': shares,
                'reason': f'DL_conf={dl_info["confidence"]:.0f}%(x{leverage:.1f})'
            })
            bought += 1

    # 3. 持仓：DL 反转 SELL → 保盈利离场
    for ticker, pos in list(positions.items()):
        if ticker not in row or pd.isna(row[ticker]):
            continue
        dl_info = next((d for d in dl_list if d['ticker'] == ticker), None)
        if not dl_info or dl_info['signal'] != 'SELL':
            continue
        current_price = float(row[ticker])
        entry = pos.get('avg_cost', current_price)
        pnl_pct = (current_price - entry) / entry * 100
        if pnl_pct >= 2.0:
            signals.append({
                'action': 'SELL', 'ticker': ticker, 'shares': pos['shares'],
                'reason': f'DL反转SELL pnl={pnl_pct:.1f}%'
            })

    return signals


# 运行回测
print(f'\n开始回测 180天...')
be = BacktestEngine(initial_cash=10000)
result = be.run(
    tickers=WATCHLIST,
    strategy_fn=strategy_fn,
    start_date=start_date,
    end_date=end_date,
    stop_loss_pct=params.stop_loss_pct,
    take_profit_pct=params.take_profit_pct,
)

if result:
    print(f'\n=== 180天回测结果 ===')
    print(f'总收益率: {result.total_return_pct:.2f}%')
    print(f'最大回撤: {result.max_drawdown_pct:.2f}%')
    print(f'Sharpe: {result.sharpe_ratio:.2f}')
    print(f'胜率: {result.win_rate:.1f}%')
    trades = result.trades
    buy_trades = [t for t in trades if t.get('action') == 'BUY']
    sell_trades = [t for t in trades if t.get('action') == 'SELL']
    print(f'总交易: {len(trades)} 笔 (BUY={len(buy_trades)}, SELL={len(sell_trades)})')
    print(f'最终价值: ${result.final_value:.2f}')
    print('\n交易明细:')
    for t in trades:
        pnl_str = f'${t.get("pnl")}' if t.get('pnl') not in (None, 0) else '?'
        shares_str = t.get('shares', '?')
        print(f'  {t.get("date","?")} {t.get("action","?"):4s} {t.get("ticker","?"):4s} price={t.get("price","?")} shares={shares_str} pnl={pnl_str} reason={t.get("reason","")}')
else:
    print('回测失败')