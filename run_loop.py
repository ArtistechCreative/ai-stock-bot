import sys, os, json
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

# 1. Scoring and ranking
print("[1/6] Scoring and ranking...")
ranked = rank_stocks(WATCHLIST, top_n=8)
top_tickers = [s['ticker'] for s in ranked[:5]]
print(f"    Ranked: {ranked[:3]}")

# 2. DL predictions
print("[2/6] DL predictions...")
dl_preds = {}
try:
    preds = batch_predict(top_tickers, model_type="MLP")
    for p in preds:
        if 'error' not in p:
            dl_preds[p['ticker']] = p
    print(f"    Predictions: {dl_preds}")
except Exception as e:
    print(f"    DL prediction failed: {e}")

# 3. Combined signals
print("[3/6] Combined signals...")
optimizer = StrategyOptimizer(state_path=f"{DATA_DIR}/strategy_state.json")
score_list = [{'ticker': s['ticker'], 'score': s['score']} for s in ranked]
dl_list = [{'ticker': t, 'signal': dl_preds[t]['signal'], 'confidence': dl_preds[t]['confidence']} for t in dl_preds]
combined = optimizer.get_signal(dl_list, score_list)
print(f"    Combined: {combined}")

# Find high-score signals
high_score = [c for c in combined if c['combined_score'] > 0.6 and any(d['signal'] == 'BUY' for d in dl_list if d['ticker'] == c['ticker'])]
print(f"    High-score signals (>0.6 combined_score + BUY): {high_score}")

# 4. Backtesting (silent learning)
print("[4/6] Silent backtesting...")
try:
    be = BacktestEngine(initial_cash=10000)
    result = be.run(
        tickers=WATCHLIST[:8],
        strategy_fn=lambda *args: [],
        start_date=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
        end_date=datetime.now().strftime("%Y-%m-%d"),
        stop_loss_pct=8,
        take_profit_pct=15,
    )
    has_result = result is not None
    if result:
        print(f"    Backtest result: return={result.total_return_pct:.2f}%, max_dd={result.max_drawdown_pct:.2f}%, win_rate={result.win_rate:.2f}, sharpe={result.sharpe_ratio:.2f}")
except Exception as e:
    print(f"    Backtest failed: {e}")
    has_result = False

# 5. Update strategy
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
    )
    print(f"[5/6] Strategy updated | Score: {comp_score:.2f} | Changes: {changes}")
    optimizer.apply_params(new_params)
else:
    print("[5/6] Skipped strategy update (no backtest data)")

# 6. Push high-score signals to Telegram
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
            # Find price
            price = next((s['price'] for s in ranked if s['ticker'] == ticker), None)
            if price:
                entry = round(price, 2)
                stop = round(price * 0.92, 2)
                target = round(price * 1.15, 2)
                top_signals.append(f"BUY {ticker} @${entry} | SL${stop} | TP${target} | Conf{conf}% | Score{score_val:.2f}")

        msg = f"AI Stock Bot Signals - {datetime.now().strftime('%m/%d %H:%M')}\n\n" + "\n".join(top_signals) + "\n\nDisclaimer: Not investment advice"
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={'chat_id': chat_id, 'text': msg}, timeout=10)
            print(f"[6/6] Telegram push success: {len(top_signals)} signals")
        except Exception as e:
            print(f"[6/6] Telegram push failed: {e}")
    else:
        print("[6/6] No Telegram token configured")
else:
    print("[6/6] No high-score signals to push")

print("=== Background learning complete ===")
