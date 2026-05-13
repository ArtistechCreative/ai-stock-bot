"""
持仓跟踪器：检查 OPEN 仓位是否触达止盈/止损/追踪止损，自动记录平仓
- 每 5 分钟由 cronjob 驱动
- 追踪止损：TP1 激活后，从持仓期间最高/最低价向不利方向回撤 N 点触发
- 状态持久化在 ~/.hermes/ai-stock-bot/trail_state.json（跨 cron 保持 peak_price）
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.expanduser("~/.hermes/skills/productivity/ai-stock-trading-bot/references"))

from datetime import datetime
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.hermes/.env"))

from auto_trade import get_live_quotes

# ---------- 追踪止损状态文件 ----------
TRAIL_STATE_FILE = os.path.expanduser("~/.hermes/ai-stock-bot/trail_state.json")

def _load_trail_state() -> dict:
    if os.path.exists(TRAIL_STATE_FILE):
        try:
            with open(TRAIL_STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_trail_state(state: dict):
    os.makedirs(os.path.dirname(TRAIL_STATE_FILE), exist_ok=True)
    with open(TRAIL_STATE_FILE, "w") as f:
        json.dump(state, f)


# ---------- Google Sheets 持仓汇总 API（复制引用避免循环依赖） ----------
PORTFOLIO_MODULE = os.path.expanduser(
    "~/.hermes/skills/productivity/ai-stock-trading-bot/references/google_sheets_portfolio.py"
)
PORTFOLIO_DIR = os.path.dirname(PORTFOLIO_MODULE)
sys.path.insert(0, PORTFOLIO_DIR)

from google_sheets_portfolio import (
    get_summary,
    close_signal,
    get_all_signals_for_close,
    SHEET_ID,
    ASSET_LEVERAGE_TIERS,
)
from telegram_bot import send_report


# ---------- 追踪止损分级配置 ----------
# 点数（points）：加密货币直用点数；外汇用小数点后4位（pip 概念）；股票/指数用百分比换算
# 规则：价格越高 → 点数绝对值越大（绝对值保护）
TRAILING_STOP_TIERS = {
    # tier 1 — 主流币（高价值，绝对点数要能覆盖正常波动）
    ("BTC/USDT", "BUY"): 30,
    ("BTC/USDT", "SHORT"): 30,
    ("ETH/USDT", "BUY"):  30,
    ("ETH/USDT", "SHORT"): 30,
    ("BNB/USDT", "BUY"):  30,
    ("BNB/USDT", "SHORT"): 30,
    # tier 2 — 主流山寨（中等价格）
    ("SOL/USDT", "BUY"):  50,
    ("SOL/USDT", "SHORT"): 50,
    ("XRP/USDT", "BUY"):  50,
    ("XRP/USDT", "SHORT"): 50,
    ("ADA/USDT", "BUY"):  50,
    ("ADA/USDT", "SHORT"): 50,
    ("AVAX/USDT", "BUY"): 50,
    ("AVAX/USDT", "SHORT"): 50,
    ("LINK/USDT", "BUY"): 50,
    ("LINK/USDT", "SHORT"): 50,
    ("DOT/USDT", "BUY"):  50,
    ("DOT/USDT", "SHORT"): 50,
    ("MATIC/USDT","BUY"): 50,
    ("MATIC/USDT","SHORT"):50,
    ("LTC/USDT", "BUY"):  50,
    ("LTC/USDT", "SHORT"): 50,
    ("UNI/USDT", "BUY"):  50,
    ("UNI/USDT", "SHORT"): 50,
    ("APT/USDT", "BUY"):  50,
    ("APT/USDT", "SHORT"): 50,
    ("ARB/USDT", "BUY"):  50,
    ("ARB/USDT", "SHORT"): 50,
    ("INJ/USDT", "BUY"):  50,
    ("INJ/USDT", "SHORT"): 50,
    ("SUI/USDT", "BUY"):  50,
    ("SUI/USDT", "SHORT"): 50,
    ("TIA/USDT", "BUY"):  50,
    ("TIA/USDT", "SHORT"): 50,
    # tier 3 — MEME/高波动币（价格低，百分比波动大）
    ("DOGE/USDT","BUY"): 100,
    ("DOGE/USDT","SHORT"):100,
    ("SHIB/USDT","BUY"): 100,
    ("SHIB/USDT","SHORT"):100,
    ("PEPE/USDT","BUY"): 100,
    ("PEPE/USDT","SHORT"):100,
    ("WIF/USDT", "BUY"): 100,
    ("WIF/USDT", "SHORT"):100,
    # tier 4 — 外汇 CFD（用 pip-like 点数，小数点后4位）
    ("EURUSD=X","BUY"):  0.0030,   # 30 pips
    ("EURUSD=X","SHORT"):0.0030,
    ("GBPUSD=X","BUY"):  0.0030,
    ("GBPUSD=X","SHORT"):0.0030,
    ("AUDUSD=X","BUY"):  0.0030,
    ("AUDUSD=X","SHORT"):0.0030,
    ("USDJPY=X","BUY"):  0.30,     # JPY pip
    ("USDJPY=X","SHORT"):0.30,
    ("EURGBP=X","BUY"):  0.0030,
    ("EURGBP=X","SHORT"):0.0030,
    ("EURJPY=X","BUY"):  0.30,
    ("EURJPY=X","SHORT"):0.30,
    ("GBPJPY=X","BUY"):  0.30,
    ("GBPJPY=X","SHORT"):0.30,
    # tier 5 — 贵金属/大宗
    ("GC=F",   "BUY"):  3.0,       # 黄金 $3
    ("GC=F",   "SHORT"):3.0,
    ("CL=F",   "BUY"):  0.50,      # 原油 $0.5
    ("CL=F",   "SHORT"):0.50,
    # tier 6 — 指数 ETF（用百分比价格，非点数）
    ("ES=F",   "BUY"):  1.0,       # S&P 500 点数（= $1）
    ("NQ=F",   "BUY"):  2.0,       # Nasdaq 100 点数（= $2）
    ("ES=F",   "SHORT"):1.0,
    ("NQ=F",   "SHORT"):2.0,
    ("^KLSE",  "BUY"):  0.5,       # 马来西亚综指
    ("^KLSE",  "SHORT"):0.5,
    ("^STI",   "BUY"):  0.5,       # 海峡时报
    ("^STI",   "SHORT"):0.5,
}


def get_trailing_stop(ticker: str, direction: str, current_price: float = 0) -> float:
    """
    返回追踪止损点数。
    优先用固定档位；无法识别时根据价格自动估算。
    """
    key = (ticker.upper(), direction)
    if key in TRAILING_STOP_TIERS:
        return TRAILING_STOP_TIERS[key]

    # 自动估算：价格 > 1000 → 资产价值高，点数随之放大；< 1 → 超小价格
    if current_price > 0:
        if current_price >= 10000:
            return 30
        elif current_price >= 1000:
            return 50
        elif current_price >= 100:
            return 100
        elif current_price >= 10:
            return 200
        else:
            return 500
    return 50  # 默认


def check_and_close_positions() -> dict:
    """
    检查所有 OPEN 持仓，触达止盈/止损则自动平仓。
    返回 {closed: [...], alerts: [...], errors: [...]}
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now}] 🔍 持仓跟踪检查...")

    # 1. 读取持仓汇总（所有 OPEN 仓位）
    try:
        positions = get_summary(live_prices=None)
    except Exception as e:
        print(f"[!] 读取持仓汇总失败: {e}")
        return {"closed": [], "alerts": [], "errors": [str(e)]}

    open_positions = [p for p in positions if p.get("status") == "OPEN"]
    if not open_positions:
        print("  ✅ 无 OPEN 持仓，跳过")
        return {"closed": [], "alerts": [], "errors": []}

    print(f"  📋 检测到 {len(open_positions)} 个 OPEN 持仓")

    # 2. 获取所有持仓标的现价
    tickers = [p["ticker"] for p in open_positions]
    quotes = get_live_quotes(tickers)
    if not quotes:
        print("  [!] 无法获取实时行情")
        return {"closed": [], "alerts": [], "errors": ["No quote data"]}

    closed = []
    alerts = []
    errors = []

    # 加载追踪止损状态
    trail_state = _load_trail_state()

    for pos in open_positions:
        ticker = pos["ticker"]
        direction = pos.get("direction", "BUY")
        avg_cost = pos.get("avg_cost", 0)
        stop_loss = pos.get("stop_loss", 0)
        take_profit = pos.get("take_profit", 0)
        # 追踪止损点数从分级配置获取（不再从持仓记录读取）
        shares = pos.get("shares", 0)
        row_index = pos.get("row")

        q = quotes.get(ticker, {})
        raw_price = q.get("price", 0) or 0
        try:
            current_price = float(raw_price)
        except (ValueError, TypeError):
            current_price = 0.0
        if current_price <= 0:
            current_price = avg_cost  # fallback

        # 追踪止损点数（根据资产类别分级）
        trailing_stop_points = get_trailing_stop(ticker, direction, current_price)

        # 计算当前盈亏%
        if direction == "BUY":
            pnl_pct = (current_price - avg_cost) / avg_cost * 100 if avg_cost > 0 else 0
        else:
            pnl_pct = (avg_cost - current_price) / avg_cost * 100 if avg_cost > 0 else 0

        # 初始化/读取该持仓的追踪状态
        key = f"{ticker}_{direction}"
        state = trail_state.get(key, {})
        peak_price   = state.get("peak_price", avg_cost)
        valley_price = state.get("valley_price", avg_cost)
        trailing_activated = state.get("trailing_activated", False)

        # 更新 peak / valley
        if direction == "BUY":
            if current_price > peak_price:
                peak_price = current_price
        else:  # SHORT
            if current_price < valley_price:
                valley_price = current_price

        # TP1 激活判断（未激活时先激活）
        hit_tp = False
        if take_profit > 0:
            if direction == "BUY":
                hit_tp = current_price >= take_profit
            else:
                hit_tp = current_price <= take_profit

        if hit_tp and not trailing_activated:
            trailing_activated = True
            alerts.append(
                f"✅ *追踪止损已激活* — {ticker}\n"
                f"  激活价: ${current_price:.2f} | 档位: {trailing_stop_points} 点"
            )

        # 止损触发判断
        hit_stop = stop_loss > 0
        if direction == "BUY":
            hit_stop = stop_loss > 0 and current_price <= stop_loss
        else:
            hit_stop = stop_loss > 0 and current_price >= stop_loss

        # 追踪止损触发判断（TP1 激活后才生效，朝有利方向移动）
        hit_trailing = False
        if trailing_activated and trailing_stop_points > 0:
            if direction == "BUY":
                trailing_trigger_price = peak_price - trailing_stop_points
                hit_trailing = current_price <= trailing_trigger_price
            else:  # SHORT
                trailing_trigger_price = valley_price + trailing_stop_points
                hit_trailing = current_price >= trailing_trigger_price

        triggered = None
        if hit_trailing:
            triggered = "TRAILING_STOP"
        elif hit_stop:
            triggered = "STOP_LOSS"
        elif hit_tp:
            triggered = "TAKE_PROFIT"

        # 保存追踪状态（即使未触发也要更新 peak/valley）
        trail_state[key] = {
            "peak_price": peak_price,
            "valley_price": valley_price,
            "trailing_activated": trailing_activated,
        }

        if triggered:
            print(f"  🎯 {ticker} 触发 {triggered} | 现价=${current_price:.2f} | {'≤' if triggered == 'STOP_LOSS' or triggered == 'TRAILING_STOP' else '≥'} 目标价=${trailing_trigger_price if triggered == 'TRAILING_STOP' else (stop_loss if triggered == 'STOP_LOSS' else take_profit):.2f}")
            # 用 get_all_signals_for_close 找行号
            try:
                matches = get_all_signals_for_close(ticker)
                if not matches:
                    errors.append(f"{ticker}: 信号池找不到未平仓行")
                    del trail_state[key]  # 平仓后删除状态
                    _save_trail_state(trail_state)
                    continue
                target_row = matches[0]["row"] + 1
            except Exception as e:
                errors.append(f"{ticker}: 查找信号行失败 - {e}")
                del trail_state[key]
                _save_trail_state(trail_state)
                continue

            # 执行平仓
            try:
                result = close_signal(row_index=target_row, close_price=current_price)
                del trail_state[key]  # 平仓后删除状态
                _save_trail_state(trail_state)
                if result.get("status") == "closed":
                    closed.append({
                        "ticker": ticker,
                        "triggered": triggered,
                        "close_price": current_price,
                        "pnl": result.get("pnl", 0),
                        "pnl_pct": result.get("pnl_pct", 0),
                        "holding_days": result.get("holding_days", 0),
                        "row": target_row,
                    })
                    emoji_map = {"STOP_LOSS": "🛑", "TAKE_PROFIT": "🎯", "TRAILING_STOP": "📍"}
                    emoji = emoji_map.get(triggered, "🎯")
                    alerts.append(
                        f"{emoji} *{triggered}* — {ticker}\n"
                        f"  平仓价: ${current_price:.2f} | 盈亏: ${result.get('pnl', 0):.2f} ({pnl_pct:.1f}%)\n"
                        f"  持仓天数: {result.get('holding_days', 0)} 天"
                    )
                else:
                    errors.append(f"{ticker}: close_signal 返回 {result}")
            except Exception as e:
                errors.append(f"{ticker}: 平仓写入失败 - {e}")

    # 4. 保存追踪止损状态（即使没有触发平仓也持久化 peak_price）
    _save_trail_state(trail_state)

    # 5. 发送 Telegram 通知
    if alerts:
        header = f"🚨 *持仓自动平仓* — {datetime.now().strftime('%H:%M')}"
        body = "\n\n".join(alerts)
        try:
            chat_id = os.getenv("TELEGRAM_HOME_CHANNEL") or os.getenv("TELEGRAM_CHAT_ID") or "6801255591"
            send_report(f"{header}\n\n{body}", chat_id=chat_id)
            print(f"  📱 已推送 Telegram 通知")
        except Exception as e:
            print(f"  [!] Telegram 推送失败: {e}")
    else:
        print("  ✅ 暂无触发平仓的持仓")

    print(f"\n  完成: 平仓 {len(closed)} 笔，错误 {len(errors)} 笔")
    return {"closed": closed, "alerts": alerts, "errors": errors}


if __name__ == "__main__":
    result = check_and_close_positions()
    print(f"\n结果: {result}")