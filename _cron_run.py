#!/usr/bin/env python3
"""
AI股神后台学习 Cron 脚本（每15分钟）
=======================================
使用板块路由 DL 预测 + 评分系统综合信号。

用法（Hermes cronjob）：
  路径：/home/aitistech/projects/ai-stock-bot/_cron_run.py
  计划：*/15 * * * *
"""

import sys, os, json
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "bot"))

from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.hermes/.env"))

from scorer import rank_stocks
from dl_strategy import batch_predict_sector
from strategy_optimizer import StrategyOptimizer
from datetime import datetime, timedelta
import numpy as np

DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
STATE_FILE = DATA_DIR / "strategy_state.json"

# ── 三大板块配置（31 ticker）── 与 train_dl.py / run_cron.py 完全一致 ──
SECTOR_CONFIG = {
    "tech_high_vol": [
        "NVDA", "TSLA", "AMD", "MSFT", "META", "AAPL", "AMZN", "GOOGL",
        "AVGO", "NFLX", "TSM", "INTC", "BABA", "PDD", "ORCL"
    ],
    "traditional_defensive": [
        "JPM", "V", "UNH", "XOM", "COST", "WMT", "BA", "HON"
    ],
    "cryptocurrency": [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
        "XRP/USDT", "DOGE/USDT", "ADA/USDT"
    ],
}
ALL_TICKERS = [t for tickers in SECTOR_CONFIG.values() for t in tickers]

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] === AI股神后台学习 ===")
print(f"  板块: {list(SECTOR_CONFIG.keys())} ({len(ALL_TICKERS)} ticker)")

# ── Step 1: 市场评分（as_of_date 防止数据泄漏）───
print("\n[1/6] 市场评分...")
try:
    ranked = rank_stocks(ALL_TICKERS, top_n=12)
    print(f"  评分完成: {len(ranked)} 个资产")
    for r in ranked[:5]:
        icon = "🪙" if r.get("_is_crypto") else "💻"
        print(f"  {icon} {r['ticker']}: score={r.get('score', 0):.1f} rsi={r.get('rsi', 'N/A')}")
except Exception as e:
    print(f"  [!] 评分失败: {e}")
    ranked = []

# ── Step 2: 板块路由 DL 预测 ──
print("\n[2/6] 板块路由 DL 预测...")
dl_preds = {}
try:
    preds = batch_predict_sector(ALL_TICKERS)
    for p in preds:
        if "error" not in p and p.get("signal") in ("BUY", "SELL"):
            dl_preds[p["ticker"]] = p
    print(f"  有效预测: {len(dl_preds)}/{len(ALL_TICKERS)}")
    for p in list(dl_preds.values())[:5]:
        print(f"  📈 {p['ticker']}: {p['signal']} {p['confidence']:.0f}%")
except Exception as e:
    print(f"  [!] DL预测失败: {e}")

# ── Step 3: 综合信号（get_signal 新架构）───
print("\n[3/6] 综合信号...")
combined = []
if ranked:
    try:
        score_list = [{"ticker": s["ticker"], "score": s["score"]} for s in ranked]
        dl_list = [
            {"ticker": t, "signal": dl_preds[t]["signal"], "confidence": dl_preds[t]["confidence"]}
            for t in dl_preds
        ]
        optimizer = StrategyOptimizer(state_path=str(STATE_FILE))
        combined = optimizer.get_signal(dl_list, score_list)
        print(f"  综合排名: {len(combined)} 个资产")
    except Exception as e:
        print(f"  [!] 综合信号失败: {e}")

# ── Step 4: 快速回测（评分买 top 3 策略）───
print("\n[4/6] 快速回测...")
backtest_score = 0.0
has_result = False
try:
    from backtest import BacktestEngine

    def make_strategy(ranked_list):
        """用评分排名生成买入信号"""
        top = {s["ticker"] for s in ranked_list[:3]}
        def strategy_fn(positions, cash, row, date_str):
            signals = []
            for ticker in top:
                if ticker in positions:
                    continue
                if ticker not in row or np.isnan(row[ticker]):
                    continue
                price = float(row[ticker])
                if price <= 0 or price * 1 > cash * 0.9:
                    continue
                shares = max(1, int((cash * 0.1) / price))
                signals.append({
                    "action": "BUY", "ticker": ticker,
                    "shares": shares,
                    "reason": f"score_top3"
                })
            return signals
        return strategy_fn

    be = BacktestEngine(initial_cash=10000)
    result = be.run(
        tickers=ALL_TICKERS,
        strategy_fn=make_strategy(ranked),
        start_date=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
        end_date=datetime.now().strftime("%Y-%m-%d"),
        stop_loss_pct=8,
        take_profit_pct=15,
        trailing_trigger_pct=5.0,
        trailing_stop_pct=3.0,
    )
    if result:
        has_result = True
        backtest_score = max(-10, min(15, result.total_return_pct))  # clamp
        print(f"  回测结果: return={result.total_return_pct:.2f}% "
              f"max_dd={result.max_drawdown_pct:.2f}% "
              f"win_rate={result.win_rate:.2f}% "
              f"sharpe={result.sharpe_ratio:.2f} "
              f"trades={result.total_trades}")
except Exception as e:
    print(f"  [!] 回测失败: {e}")

# ── Step 5: 策略参数更新 ──
print("\n[5/6] 策略更新...")
if has_result:
    try:
        optimizer = StrategyOptimizer(state_path=str(STATE_FILE))
        # 估算 DL 准确率
        dl_acc = 0.55
        if dl_preds:
            dl_acc = min(0.70, sum(p.get("confidence", 50) for p in dl_preds.values()) / max(len(dl_preds), 1) / 100)

        new_params, changes, comp_score = optimizer.adjust_params(
            backtest_return=backtest_score,
            max_drawdown=result.max_drawdown_pct if result else 0,
            win_rate=result.win_rate / 100 if result else 0,
            sharpe=result.sharpe_ratio if result else 0,
            dl_accuracy=dl_acc,
        )
        change_str = "; ".join(changes) if changes else "无调整"
        print(f"  策略更新完成 | composite={comp_score:.2f} | {change_str}")
        optimizer.apply_params(new_params)
    except Exception as e:
        print(f"  [!] 策略更新失败: {e}")
else:
    print("  跳过（无回测数据）")

# ── Step 6: Telegram 推送（高分信号）───
print("\n[6/6] Telegram 推送...")
high_score = [
    c for c in combined
    if c.get("combined_score", 0) > 0.55
    and (abs(c.get("dl_confidence", 0)) >= 60 or c.get("combined_score", 0) > 0.75)
][:4]

if high_score:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_HOME_CHANNEL", "6801255591")
    if token:
        import requests
        lines = []
        for c in high_score:
            ticker = c["ticker"]
            dl_p = dl_preds.get(ticker, {})
            sig = dl_p.get("signal", "BUY")
            conf = dl_p.get("confidence", 0)
            price = next((s["price"] for s in ranked if s["ticker"] == ticker), None)
            if price:
                entry = round(price, 2)
                stop = round(price * 0.92, 2)
                target = round(price * 1.15, 2)
                dl_b = c.get("dl_bonus", 0)
                dl_c = abs(c.get("dl_confidence", 0))
                bonus_str = f"+{dl_b:.3f}({dl_c:.0f}%)" if dl_b > 0 else ("-" if dl_b < 0 else "无DL")
                icon = "🪙" if "/" in ticker else "💻"
                lines.append(
                    f"{icon}[{sig}] {ticker} | ${entry} | SL${stop} TP${target} "
                    f"| 综合={c['combined_score']:.3f} 评分={c['score_signal']:.3f} DL={bonus_str}"
                )

        msg = (
            f"🟢 AI股神信号 - {datetime.now().strftime('%m/%d %H:%M')}\n"
            f"回测评分: {backtest_score:.2f}\n\n"
            + "\n".join(lines)
            + "\n\n仅供参考，不构成投资建议"
        )
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": msg},
                timeout=10
            )
            print(f"  推送成功: {len(lines)} 条信号")
        except Exception as e:
            print(f"  [!] 推送失败: {e}")
    else:
        print("  无 bot token，跳过")
else:
    print("  无高分信号，跳过推送")

print(f"\n=== 后台学习完成 {datetime.now().strftime('%H:%M:%S')} ===")