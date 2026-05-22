import sys, os, json
import pandas as pd
sys.path.insert(0, 'bot')
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.hermes/.env"))

from scorer import rank_stocks, score_stock
from backtest import BacktestEngine
from dl_strategy import DLStrategy, batch_predict
from strategy_optimizer import StrategyOptimizer
from datetime import datetime, timedelta

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

WATCHLIST = ['NVDA','TSLA','AMD','MSFT','META','AAPL','AMZN','GOOGL','JPM','V','UNH','XOM','JNJ','KO','DIS','NFLX','PLTR','COIN','SOFI']

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] === AI Stock Bot Learning Loop ===")

# 1. Scoring
print("[1/6] Scoring stocks...")
ranked = rank_stocks(WATCHLIST, top_n=8)
top_tickers = [s['ticker'] for s in ranked[:5]]

# 2. DL predictions
print("[2/6] Running DL predictions...")
dl_preds = {}
try:
    preds = batch_predict(top_tickers, model_type="MLP")
    for p in preds:
        if 'error' not in p:
            dl_preds[p['ticker']] = p
except Exception as e:
    print(f"  DL prediction failed: {e}")

# 3. Combined signals
print("[3/6] Computing combined signals...")
optimizer = StrategyOptimizer(state_path=f"{DATA_DIR}/strategy_state.json")
score_list = [{'ticker': s['ticker'], 'score': s['score']} for s in ranked]
dl_list = [{'ticker': t, 'signal': dl_preds[t]['signal'], 'confidence': dl_preds[t]['confidence']} for t in dl_preds]
combined = optimizer.get_signal(dl_list, score_list)

high_score = [c for c in combined if c['combined_score'] > 0.6 and any(d['signal'] == 'BUY' for d in dl_list if d['ticker'] == c['ticker'])]

# 4. Backtest (silent learning)
print("[4/6] Running backtest...")
has_result = False

# ── 真正的 strategy_fn：基于评分 + DL 信号生成买卖信号 ──────────
# 获取当前策略参数
params = optimizer.params

def strategy_fn(positions, cash, row, date_str):
    """
    策略函数：每天被回测引擎调用一次
    positions: 当前持仓 {ticker: {shares, avg_cost, stop, target, entry_date}}
    cash: 可用资金
    row: 当天收盘价 pd.Series
    date_str: 日期字符串
    返回: [{"action": "BUY"|"SELL", "ticker": "...", "shares": N}, ...]
    """
    from bot.data_fetcher import fetch_quote

    signals = []
    holdings = set(positions.keys())

    # ── 1. 持仓止损/止盈/趋势检查 ──────────────────────────
    for ticker, pos in list(positions.items()):
        if ticker not in row or pd.isna(row[ticker]):
            continue
        price = float(row[ticker])
        entry = pos.get("avg_cost", price)
        # 追踪止损（trailing stop）：从高点回撤 5% 即出
        highest_since_entry = entry * 1.05  # 简化版：entry+5%算高位
        if price < highest_since_entry and (entry - price) / entry > 0.05:
            signals.append({
                "action": "SELL",
                "ticker": ticker,
                "shares": pos["shares"],
                "reason": f"TRAIL_STOP 回撤{(entry-price)/entry*100:.1f}%",
            })
            continue
        # 弱势股检查（RSI 过滤）：如果可用数据的话，跳过已超卖的
        # 此处省略 RSI 计算（需要历史数据），简化为均线偏离

    # ── 2. 选股：综合评分 Top 8 候选（扩大候选池）──────────
    candidates = [c for c in combined if c["ticker"] not in holdings]
    # 按 combined_score 排序
    candidates.sort(key=lambda x: x["combined_score"], reverse=True)

    slots = min(3, params.max_positions - len(holdings))  # 每次最多加 3 支
    bought = 0

    for candidate in candidates:
        if bought >= slots:
            break
        ticker = candidate["ticker"]
        if ticker not in row or pd.isna(row[ticker]):
            continue

        price = float(row[ticker])
        dl_info = next((d for d in dl_list if d["ticker"] == ticker), None)
        if not dl_info:
            continue

        # ── 信号过滤：DL 必须是 BUY，信心度 > min_confidence ──
        if dl_info["signal"] != "BUY":
            continue
        if dl_info["confidence"] < params.min_confidence:
            continue

        # ── 动态仓位：信心度越高，仓位越大 ──────────────────
        # 基础仓位 = min(cash * max_position_pct, price * max_shares)
        base_position_pct = params.max_position_pct / 100
        # 信心度调整：[min_conf, 100] → [0.6x, 1.5x]
        conf_ratio = (dl_info["confidence"] - params.min_confidence) / (100 - params.min_confidence)
        leverage = 0.6 + conf_ratio * 0.9  # 0.6 ~ 1.5
        adjusted_pct = min(base_position_pct * leverage, 0.5)  # 上限 50% 单笔

        position_value = cash * adjusted_pct
        shares = max(1, int(position_value / price))
        cost = shares * price

        if cost > 0 and cost <= cash * 0.85:
            signals.append({
                "action": "BUY",
                "ticker": ticker,
                "shares": shares,
                "reason": (
                    f"DL_conf={dl_info['confidence']:.0f}% "
                    f"(x{leverage:.1f}) score={candidate['combined_score']:.2f} "
                    f"dl={candidate['dl_signal']:.2f} rank={candidate['score_signal']:.2f}"
                ),
            })
            bought += 1

    # ── 3. 卖出：DL 反转信号（SELL）且持仓盈利/保本 ────────
    for ticker, pos in list(positions.items()):
        if ticker not in row or pd.isna(row[ticker]):
            continue
        dl_info = next((d for d in dl_list if d["ticker"] == ticker), None)
        if not dl_info:
            continue
        current_price = float(row[ticker])
        entry = pos.get("avg_cost", current_price)
        pnl_pct = (current_price - entry) / entry * 100

        # DL 说 SELL + 盈利或保本 → 止盈离场
        if dl_info["signal"] == "SELL" and pnl_pct >= -1.0:
            signals.append({
                "action": "SELL",
                "ticker": ticker,
                "shares": pos["shares"],
                "reason": f"DL反转SELL pnl{pnl_pct:.1f}% conf={dl_info['confidence']:.0f}%",
            })
        # DL 说 SELL + 小亏（<3%）→ 止损保住本金
        elif dl_info["signal"] == "SELL" and pnl_pct < -1.0 and pnl_pct > -4.0:
            signals.append({
                "action": "SELL",
                "ticker": ticker,
                "shares": pos["shares"],
                "reason": f"DL警告SELL pnl{pnl_pct:.1f}%，保护本金",
            })

    return signals

try:
    be = BacktestEngine(initial_cash=10000)
    result = be.run(
        tickers=WATCHLIST,  # 全部 19 支股票，不要只取前 8
        strategy_fn=strategy_fn,
        start_date=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
        end_date=datetime.now().strftime("%Y-%m-%d"),
        stop_loss_pct=params.stop_loss_pct,
        take_profit_pct=params.take_profit_pct,
    )
    has_result = result is not None
    if has_result:
        print(f"  Backtest result: return={result.total_return_pct:.2f}%, max_dd={result.max_drawdown_pct:.2f}%, win_rate={result.win_rate:.2f}%, sharpe={result.sharpe_ratio:.2f}")
        print(f"  Total trades: {result.total_trades} (BUY={len([t for t in result.trades if t['action']=='BUY'])}, SELL={len([t for t in result.trades if t['action']=='SELL'])})")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  Backtest failed: {e}")

# 5. Update strategy
print("[5/6] Updating strategy...")
if has_result:
    dl_acc = 0.55
    for t, p in dl_preds.items():
        dl_acc = p.get('confidence', 55) / 100
    new_params, changes, comp_score = optimizer.adjust_params(
        backtest_return=result.total_return_pct if result else 0,
        max_drawdown=result.max_drawdown_pct if result else 0,
        win_rate=result.win_rate if result else 0,
        sharpe=result.sharpe_ratio if result else 0,
        dl_accuracy=dl_acc,
        backtest_result=result,
    )
    print(f"  Strategy update | composite_score={comp_score:.2f} | changes: {'; '.join(changes) if changes else 'none'}")
    optimizer.apply_params(new_params)
else:
    print("  Skipping strategy update (no backtest result)")

# 6. Push signals to Telegram
print("[6/6] Telegram push...")
if high_score:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_HOME_CHANNEL", "6801255591")
    if token:
        import requests
        top_signals = []
        for c in high_score[:3]:
            ticker = c['ticker']
            score_val = c['combined_score']
            dl_info = next((d for d in dl_list if d['ticker'] == ticker), None)
            sig = dl_info['signal'] if dl_info else 'BUY'
            conf = dl_info['confidence'] if dl_info else 0
            price = next((s['price'] for s in ranked if s['ticker'] == ticker), None)
            if price:
                entry = round(price, 2)
                stop = round(price * 0.92, 2)
                target = round(price * 1.15, 2)
                top_signals.append(f"GREEN {ticker} | Buy ${entry} | Stop ${stop} | Target ${target} | Conf {conf}% | Score {score_val:.2f}")

        msg = f"AI Stock Signals - {datetime.now().strftime('%m/%d %H:%M')}\n\n" + "\n".join(top_signals) + "\n\nDISCLAIMER: For reference only, not investment advice."
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={'chat_id': chat_id, 'text': msg}, timeout=10)
            print(f"  Telegram push success: {len(top_signals)} signals")
        except Exception as e:
            print(f"  Telegram push failed: {e}")
else:
    print("  No high-score signals to push")

print("=== Learning Loop Complete ===")
