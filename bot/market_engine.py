"""
市场监控引擎（统一版）
===================
合并原 monitor.py + auto_trade.py，每 15 分钟执行一次。

职责：
  1. 数据拉取（统一一次，stocks + crypto）
  2. 市场异动警报（涨跌超阈值）
  3. 持仓止损/止盈检查 + 追踪止损
  4. AI 评分 + DL 预测 + 新信号生成
  5. Telegram 推送（只在有新内容时才发）

不再做的事：
  - 每轮重复推送相同的持仓状态
  - 重复写入已存在的信号到 Sheet
"""
import sys, os, json, time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

import sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.expanduser("~/.hermes/skills/productivity/ai-stock-trading-bot/references"))

from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.hermes/.env"))

from data_fetcher import fetch_quotes_batch
from config import WATCHLIST
from risk_manager import RiskConfig
from scorer import rank_stocks
from telegram_bot import send_report

# ---------- 引用路径 ----------
SKILL_REF = os.path.expanduser("~/.hermes/skills/productivity/ai-stock-trading-bot/references")
sys.path.insert(0, SKILL_REF)

from google_sheets_portfolio import (
    append_signal,
    get_pending_signals,
    get_summary,
    close_signal,
    get_all_signals_for_close,
)
from auto_trade import (
    get_live_quotes as _get_live_quotes,
    ASSET_LEVERAGE_TIERS,
    _signal_leverage,
)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

ALERT_THRESHOLD_PCT = 5.0   # 涨跌超 5% 触发异动警报
TRAIL_STATE_FILE = os.path.expanduser("~/.hermes/ai-stock-bot/trail_state.json")

# ═══════════════════════════════════════════════
# 追踪止损状态
# ═══════════════════════════════════════════════

TRAILING_STOP_TIERS = {
    ("BTC/USDT", "BUY"): 30,   ("BTC/USDT", "SHORT"): 30,
    ("ETH/USDT", "BUY"):  30,  ("ETH/USDT", "SHORT"): 30,
    ("BNB/USDT", "BUY"):  30,  ("BNB/USDT", "SHORT"): 30,
    ("SOL/USDT", "BUY"):  50,  ("SOL/USDT", "SHORT"): 50,
    ("XRP/USDT", "BUY"):  50,  ("XRP/USDT", "SHORT"): 50,
    ("ADA/USDT", "BUY"):  50,  ("ADA/USDT", "SHORT"): 50,
    ("AVAX/USDT", "BUY"): 50,  ("AVAX/USDT", "SHORT"):50,
    ("LINK/USDT", "BUY"): 50,  ("LINK/USDT", "SHORT"):50,
    ("DOT/USDT", "BUY"):  50,  ("DOT/USDT", "SHORT"): 50,
    ("MATIC/USDT","BUY"):50,  ("MATIC/USDT","SHORT"):50,
    ("LTC/USDT", "BUY"):  50,  ("LTC/USDT", "SHORT"): 50,
    ("UNI/USDT", "BUY"):  50,  ("UNI/USDT", "SHORT"): 50,
    ("APT/USDT", "BUY"):  50,  ("APT/USDT", "SHORT"): 50,
    ("ARB/USDT", "BUY"):  50,  ("ARB/USDT", "SHORT"): 50,
    ("INJ/USDT", "BUY"):  50,  ("INJ/USDT", "SHORT"): 50,
    ("SUI/USDT", "BUY"):  50,  ("SUI/USDT", "SHORT"): 50,
    ("TIA/USDT", "BUY"):  50,  ("TIA/USDT", "SHORT"): 50,
    ("DOGE/USDT","BUY"): 100,  ("DOGE/USDT","SHORT"):100,
    ("SHIB/USDT","BUY"): 100,  ("SHIB/USDT","SHORT"):100,
    ("PEPE/USDT","BUY"): 100,  ("PEPE/USDT","SHORT"):100,
    ("WIF/USDT", "BUY"): 100,  ("WIF/USDT", "SHORT"):100,
    ("EURUSD=X","BUY"): 0.0030, ("EURUSD=X","SHORT"):0.0030,
    ("GBPUSD=X","BUY"): 0.0030, ("GBPUSD=X","SHORT"):0.0030,
    ("AUDUSD=X","BUY"): 0.0030, ("AUDUSD=X","SHORT"):0.0030,
    ("USDJPY=X","BUY"): 0.30,   ("USDJPY=X","SHORT"):0.30,
    ("EURGBP=X","BUY"): 0.0030, ("EURGBP=X","SHORT"):0.0030,
    ("EURJPY=X","BUY"): 0.30,   ("EURJPY=X","SHORT"):0.30,
    ("GBPJPY=X","BUY"): 0.30,   ("GBPJPY=X","SHORT"):0.30,
    ("GC=F",   "BUY"): 3.0,    ("GC=F",   "SHORT"): 3.0,
    ("CL=F",   "BUY"): 0.50,   ("CL=F",   "SHORT"): 0.50,
    ("ES=F",   "BUY"): 1.0,    ("ES=F",   "SHORT"): 1.0,
    ("NQ=F",   "BUY"): 2.0,    ("NQ=F",   "SHORT"): 2.0,
    ("^KLSE",  "BUY"): 0.5,    ("^KLSE",  "SHORT"): 0.5,
    ("^STI",   "BUY"): 0.5,    ("^STI",   "SHORT"): 0.5,
}


def get_trailing_points(ticker: str, direction: str) -> float:
    key = (ticker.upper(), direction)
    return TRAILING_STOP_TIERS.get(key, 50)


def _load_trail_state() -> dict:
    if os.path.exists(TRAIL_STATE_FILE):
        try:
            return json.load(open(TRAIL_STATE_FILE))
        except Exception:
            pass
    return {}


def _save_trail_state(state: dict):
    os.makedirs(os.path.dirname(TRAIL_STATE_FILE), exist_ok=True)
    with open(TRAIL_STATE_FILE, "w") as f:
        json.dump(state, f)


# ═══════════════════════════════════════════════
# 统一数据拉取
# ═══════════════════════════════════════════════

def fetch_all_quotes(tickers: list[str]) -> dict:
    """一次性获取所有股票 + 加密货币行情"""
    quotes = {}

    # 股票/外汇/期货
    stock_tickers = [t for t in tickers if "/" not in t]
    if stock_tickers:
        try:
            fetched = fetch_quotes_batch(stock_tickers)
            quotes.update(fetched)
        except Exception as e:
            print(f"  [!] fetch_quotes_batch 失败: {e}")

    # 加密货币（走 CCXT）
    crypto_tickers = [t for t in tickers if "/" in t]
    if crypto_tickers:
        try:
            from crypto_data import CryptoData
            cd = CryptoData(exchange="okx")
            for sym in crypto_tickers:
                try:
                    q = cd.fetch_quote(sym, use_cache=False)
                    if q:
                        quotes[sym] = {
                            "price": q.last_price,
                            "prev_close": None,
                            "volume": q.volume_24h,
                            "avg_volume": q.volume_24h or 1,
                            "market_cap": 0,
                            "pe": None,
                            "beta": 1.0,
                            "change_24h_pct": q.change_24h_pct,
                        }
                except Exception as e:
                    print(f"  [!] {sym} 获取失败: {e}")
        except ImportError as e:
            print(f"  [!] crypto_data 导入失败: {e}")

    return quotes


# ═══════════════════════════════════════════════
# 市场异动警报
# ═══════════════════════════════════════════════

def check_market_alerts(quotes: dict) -> list[str]:
    """涨跌超阈值时返回警报消息列表"""
    alerts = []
    prev_prices = {}
    pfile = DATA_DIR / "prev_prices.json"
    if pfile.exists():
        try:
            prev_prices = json.load(open(pfile))
        except Exception:
            pass

    for ticker, q in quotes.items():
        price = q.get("price")
        prev = prev_prices.get(ticker, price)
        if not price or not prev:
            continue
        change_pct = (price - prev) / prev * 100
        if abs(change_pct) >= ALERT_THRESHOLD_PCT:
            direction = "📈暴涨" if change_pct > 0 else "📉暴跌"
            alerts.append(
                f"{direction} {ticker} {'+' if change_pct >= 0 else ''}{change_pct:.2f}% → ${price:.2f}"
            )

    # 保存当前价格
    new_prev = {t: q.get("price", 0) for t, q in quotes.items() if q.get("price")}
    with open(pfile, "w") as f:
        json.dump(new_prev, f, indent=2)

    return alerts


# ═══════════════════════════════════════════════
# 持仓止损/止盈检查（含追踪止损）
# ═══════════════════════════════════════════════

def check_portfolio_alerts(quotes: dict) -> dict:
    """
    检查所有 OPEN 持仓，触发止损/止盈/追踪止损时自动平仓。
    返回 {closed: [...], alerts: [...]}
    """
    trail_state = _load_trail_state()
    closed = []
    alerts = []

    try:
        positions = get_summary(live_prices=None)
    except Exception as e:
        print(f"  [!] 读取持仓汇总失败: {e}")
        return {"closed": [], "alerts": [], "errors": [str(e)]}

    open_positions = [p for p in positions if p.get("status") == "OPEN"]
    if not open_positions:
        return {"closed": [], "alerts": [], "errors": []}

    errors = []

    for pos in open_positions:
        ticker  = pos["ticker"]
        direction = pos.get("direction", "BUY")
        avg_cost  = pos.get("avg_cost", 0)
        stop_loss = pos.get("stop_loss", 0)
        take_profit = pos.get("take_profit", 0)
        shares    = pos.get("shares", 0)
        row_index = pos.get("row")

        q = quotes.get(ticker, {})
        raw_price = q.get("price", 0) or 0
        try:
            current_price = float(raw_price)
        except (ValueError, TypeError):
            current_price = 0.0
        if current_price <= 0:
            current_price = avg_cost

        trailing_points = get_trailing_points(ticker, direction)

        # 盈亏计算
        if direction == "BUY":
            pnl_pct = (current_price - avg_cost) / avg_cost * 100 if avg_cost > 0 else 0
        else:
            pnl_pct = (avg_cost - current_price) / avg_cost * 100 if avg_cost > 0 else 0

        # 追踪状态
        key = f"{ticker}_{direction}"
        state = trail_state.get(key, {})
        peak_price   = state.get("peak_price", avg_cost)
        valley_price = state.get("valley_price", avg_cost)
        trailing_activated = state.get("trailing_activated", False)

        # 更新 peak / valley
        if direction == "BUY":
            if current_price > peak_price:
                peak_price = current_price
        else:
            if current_price < valley_price:
                valley_price = current_price

        # TP1 激活追踪止损
        hit_tp = False
        if take_profit > 0:
            hit_tp = (direction == "BUY" and current_price >= take_profit) or \
                     (direction == "SHORT" and current_price <= take_profit)

        if hit_tp and not trailing_activated:
            trailing_activated = True
            alerts.append(
                f"✅ 追踪止损已激活 — {ticker}\n"
                f"  激活价: ${current_price:.5f} | 档位: {trailing_points} 点"
            )

        # 止损触发
        hit_stop = stop_loss > 0 and (
            (direction == "BUY" and current_price <= stop_loss) or
            (direction == "SHORT" and current_price >= stop_loss)
        )

        # 追踪止损触发
        hit_trailing = False
        if trailing_activated and trailing_points > 0:
            if direction == "BUY":
                hit_trailing = current_price <= (peak_price - trailing_points)
            else:
                hit_trailing = current_price >= (valley_price + trailing_points)

        triggered = None
        if hit_trailing:
            triggered = "TRAILING_STOP"
        elif hit_stop:
            triggered = "STOP_LOSS"
        elif hit_tp:
            triggered = "TAKE_PROFIT"

        # 持久化追踪状态
        trail_state[key] = {
            "peak_price": peak_price,
            "valley_price": valley_price,
            "trailing_activated": trailing_activated,
        }

        if triggered:
            print(f"  🎯 {ticker} 触发 {triggered} | 现价=${current_price:.5f}")
            try:
                matches = get_all_signals_for_close(ticker)
                if not matches:
                    errors.append(f"{ticker}: 信号池找不到未平仓行")
                    del trail_state[key]
                    _save_trail_state(trail_state)
                    continue
                target_row = matches[0]["row"] + 1
            except Exception as e:
                errors.append(f"{ticker}: 查找信号行失败 - {e}")
                del trail_state[key]
                _save_trail_state(trail_state)
                continue

            try:
                result = close_signal(row_index=target_row, close_price=current_price)
                del trail_state[key]
                _save_trail_state(trail_state)
                if result.get("status") == "closed":
                    closed.append({
                        "ticker": ticker,
                        "triggered": triggered,
                        "close_price": current_price,
                        "pnl": result.get("pnl", 0),
                        "pnl_pct": pnl_pct,
                        "holding_days": result.get("holding_days", 0),
                    })
                    emoji = {"STOP_LOSS": "🛑", "TAKE_PROFIT": "🎯", "TRAILING_STOP": "📍"}.get(triggered, "🎯")
                    alerts.append(
                        f"{emoji} {triggered} — {ticker}\n"
                        f"  平仓价: ${current_price:.2f} | 盈亏: ${result.get('pnl', 0):.2f} ({pnl_pct:.1f}%)\n"
                        f"  持仓天数: {result.get('holding_days', 0)} 天"
                    )
                else:
                    errors.append(f"{ticker}: close_signal 返回 {result}")
            except Exception as e:
                errors.append(f"{ticker}: 平仓写入失败 - {e}")

    _save_trail_state(trail_state)
    return {"closed": closed, "alerts": alerts, "errors": errors}


# ═══════════════════════════════════════════════
# AI 信号生成
# ═══════════════════════════════════════════════

def generate_and_push_signals(quotes: dict) -> list[str]:
    """
    基于当前行情生成 AI 信号，有新信号才推送。
    返回警报消息列表。
    """
    from auto_trade import generate_signals, execute_signals

    alerts = []

    # 已存在的 PENDING ticker（防重复）
    try:
        pending = get_pending_signals()
        existing_pending = {p["ticker"].upper() for p in pending}
    except Exception:
        existing_pending = set()

    # AI 评分
    try:
        ranked = rank_stocks(WATCHLIST, top_n=8)
    except Exception as e:
        print(f"  [!] 评分失败: {e}")
        return alerts

    top_tickers = [s["ticker"] for s in ranked[:5]]

    # DL 预测（板块路由版）
    dl_preds = {}
    try:
        from dl_strategy import batch_predict_sector
        preds = batch_predict_sector(WATCHLIST)
        for p in preds:
            if "error" not in p and p.get("signal") in ("BUY", "SELL"):
                dl_preds[p["ticker"]] = p
        print(f"  🧠 DL板块预测: {len(dl_preds)}/{len(WATCHLIST)} 个有效信号")
    except Exception as e:
        print(f"  [!] DL 预测失败: {e}")

    # 模拟账户（简化）
    portfolio_cash = 10000.0
    max_pos_pct = 10.0
    risk_config = RiskConfig()

    # 生成信号
    try:
        signals = generate_signals(
            ranked_stocks=ranked,
            dl_predictions=list(dl_preds.values()),
            portfolio_cash=portfolio_cash,
            max_position_pct=max_pos_pct,
            risk_config=risk_config,
        )
    except Exception as e:
        print(f"  [!] 信号生成失败: {e}")
        return alerts

    # 过滤已有 PENDING
    new_signals = [s for s in signals if s["ticker"].upper() not in existing_pending]

    if not new_signals:
        print("  ✅ 暂无新信号（现有持仓继续跟踪）")
        return alerts

    # 执行（Sheet 写入 + Telegram 推送）
    try:
        results = execute_signals(new_signals, dry_run=False)
        if results:
            alerts.append(f"🟢 生成 {len(results)} 个新信号，已写入信号池")
    except Exception as e:
        print(f"  [!] 信号执行失败: {e}")

    return alerts


# ═══════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════

def run():
    now = datetime.now()
    print(f"\n[{now.strftime('%H:%M:%S')}] === 市场监控引擎 ===")
    all_alerts = []

    # 1. 统一数据拉取
    print("  [1/4] 拉取行情数据...")
    quotes = fetch_all_quotes(WATCHLIST)
    print(f"      获取到 {len(quotes)} 个标的行情")

    # 2. 市场异动
    print("  [2/4] 检查市场异动...")
    market_alerts = check_market_alerts(quotes)
    all_alerts.extend(market_alerts)

    # 3. 持仓止损/止盈（含追踪止损）
    print("  [3/4] 检查持仓状态...")
    portfolio_result = check_portfolio_alerts(quotes)
    all_alerts.extend(portfolio_result.get("alerts", []))

    # 4. AI 信号
    print("  [4/4] 生成 AI 信号...")
    signal_alerts = generate_and_push_signals(quotes)
    all_alerts.extend(signal_alerts)

    # 统一推送
    if all_alerts:
        header = f"📊 市场监控 — {now.strftime('%H:%M')}"
        body = "\n\n".join(all_alerts)
        chat_id = os.getenv("TELEGRAM_HOME_CHANNEL") or os.getenv("TELEGRAM_CHAT_ID") or "6801255591"
        try:
            send_report(f"{header}\n\n{body}", chat_id=chat_id)
            print(f"  ✅ 推送 {len(all_alerts)} 条警报到 Telegram")
        except Exception as e:
            print(f"  [!] Telegram 推送失败: {e}")
    else:
        print("  ✅ 无异常，继续持有")

    print(f"=== 监控完成 [{now.strftime('%H:%M:%S')}] ===\n")
    return all_alerts


if __name__ == "__main__":
    run()
