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

from bot.config import WATCHLIST

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] === AI股神后台学习循环 ===")

# 1. 评分排名
print("评分排名...")
ranked = rank_stocks(WATCHLIST, top_n=8)
top_tickers = [s['ticker'] for s in ranked[:5]]

# 2. DL预测
print("DL预测...")
dl_preds = {}
try:
    preds = batch_predict(top_tickers, model_type="MLP")
    for p in preds:
        if 'error' not in p:
            dl_preds[p['ticker']] = p
except Exception as e:
    print(f"  DL预测失败: {e}")

# 3. 综合信号
optimizer = StrategyOptimizer(state_path=f"{DATA_DIR}/strategy_state.json")
score_list = [{'ticker': s['ticker'], 'score': s['score']} for s in ranked]
dl_list = [{'ticker': t, 'signal': dl_preds[t]['signal'], 'confidence': dl_preds[t]['confidence']} for t in dl_preds]
combined = optimizer.get_signal(dl_list, score_list)

# 找高分信号
high_score = [c for c in combined if c['combined_score'] > 0.6 and any(d['signal'] == 'BUY' for d in dl_list if d['ticker'] == c['ticker'])]

# 4. 回测（静默学习）
print("静默回测...")
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
except Exception as e:
    print(f"  回测失败: {e}")
    has_result = False

# 5. 更新策略
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
    print(f"  策略更新 | 综合评分: {comp_score:.2f} | {'; '.join(changes) if changes else '无调整'}")
    optimizer.apply_params(new_params)
else:
    print("  跳过策略更新（无回测数据）")

# 6. 推送高分信号到 Telegram
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
                top_signals.append(f"[BUY] {ticker} | Entry${entry} | Stop${stop} | Target${target} | Conf{conf}% | Score{score_val:.2f}")

        msg = f"AI股神信号 - {datetime.now().strftime('%m/%d %H:%M')}\n\n" + "\n".join(top_signals) + "\n\n仅供参考，不构成投资建议"
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={'chat_id': chat_id, 'text': msg}, timeout=10)
            print(f"  推送成功: {len(top_signals)} 条信号")
        except Exception as e:
            print(f"  推送失败: {e}")

print("=== 后台学习完成 ===")
