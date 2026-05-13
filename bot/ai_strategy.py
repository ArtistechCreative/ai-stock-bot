"""
AI 交易策略引擎
支持多头(BUY/SELL)和空头(SHORT/COVER)
"""
import os, sys, json
import requests
from datetime import datetime
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.hermes/.env"))

from scorer import rank_stocks, score_stock
from risk_manager import RiskManager, RiskConfig

MODEL = "MiniMax-M2.7"
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/anthropic")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")


def get_minimax_response(messages: list) -> str:
    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": MODEL,
        "max_tokens": 1024,
        "messages": messages,
    }
    resp = requests.post(f"{MINIMAX_BASE_URL}/v1/messages", headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        return f"❌ API Error: {resp.status_code}"
    data = resp.json()
    text_blocks = [c.get("text", "") for c in data.get("content", []) if c.get("type") == "text"]
    return "\n".join(text_blocks) if text_blocks else "（无内容）"


def ai_generate_signals(
    watchlist: list[str],
    portfolio_cash: float,
    portfolio_value: float,
    current_positions: dict,
    risk_config: RiskConfig,
    top_n: int = 8,
    short_allowed: bool = True,
) -> list[dict]:
    """
    调用 AI 生成交易信号，支持：
    - BUY n股 — 开多头
    - SELL n股 — 平多头
    - SHORT n股 — 开空头
    - COVER n股 — 平空头
    - HOLD — 继续持有
    """

    ranked = rank_stocks(watchlist, top_n=top_n)
    ranked_data = []
    for s in ranked:
        ranked_data.append(
            f"- {s['ticker']} ({s['name']}): "
            f"价格=${s['price']}, PE={s['pe']}, "
            f"5日={s['change_5d_pct']}%, 量={s['volume_ratio']}x, "
            f"β={s['beta']}, 市值=${s['market_cap_B']}B, "
            f"亮点: {', '.join(s['reasons']) if s['reasons'] else '无明显催化剂'}"
        )
    market_table = "\n".join(ranked_data)

    pos_lines = []
    for ticker, pos in current_positions.items():
        ptype = pos.get("position_type", "LONG")
        direction = "🔴做空" if ptype == "SHORT" else "🟢做多"
        pnl_pct = pos.get("pnl_pct", 0)
        pos_lines.append(
            f"- {ticker}: {direction} {pos.get('shares', 0)}股, "
            f"均价${pos.get('avg_cost', 0)}, 当前${pos.get('current_price', 0)}, "
            f"盈亏{pnl_pct}%"
        )
    positions_text = "\n".join(pos_lines) if pos_lines else "空仓"

    risk_text = (
        f"止损: {risk_config.stop_loss_default_pct}% | "
        f"止盈: {risk_config.profit_taking_pct}% | "
        f"单笔上限: {risk_config.max_single_position_pct}% | "
        f"最长持仓: {risk_config.max_holding_days}天 | "
        f"最大同时持仓: {risk_config.max_positions} | "
        f"允许做空: {'是' if short_allowed else '否'}"
    )

    short_instruction = """
做空规则（当 short_allowed=true 时）：
- SHORT n股 — 估计股价将下跌，先借股卖出，等跌了再买回平仓
- COVER n股 — 买回股票平掉空头仓位
- 做空仓位需要50%保证金（可用资金必须充足）
- 做空止损：股价超过开仓价的 {stop_loss}% 时强制止损
- 做空止盈：股价跌破开仓价的 {take_profit}% 时止盈

做空盈亏计算：空头盈利 = 卖出开仓价 - 买回平仓价（价格下跌赚钱）
""".format(
        stop_loss=risk_config.stop_loss_default_pct,
        take_profit=risk_config.profit_taking_pct,
    ) if short_allowed else ""

    system_prompt = f"""你是一个短线股票交易员，擅长做多和做空。

交易指令：
- BUY n股 — 开多头仓位
- SELL n股 — 卖出/平掉多头仓位
- SHORT n股 — 卖出开仓（做空，需要账户有足够保证金）
- COVER n股 — 买回平仓（做空仓位）
- HOLD — 继续持有

{short_instruction}
风险规则：
- 单笔仓位不超过总资金的{risk_config.max_single_position_pct}%
- 同时持仓不超过{risk_config.max_positions}支（含做空）
- 短线持仓不超过{risk_config.max_holding_days}天

按以下格式输出（每一行一个指令）：
BUY [ticker] [shares]股 | 原因
SELL [ticker] | 原因（空仓不写股数）
SHORT [ticker] [shares]股 | 原因（需要保证金充足）
COVER [ticker] | 原因
HOLD [ticker] | 原因

如果没有任何信号，写：NONE
"""

    user_prompt = f"""市场排名：
{market_table}

当前投资组合：
{positions_text}

账户资金：${portfolio_cash:.2f} | 组合总价值：${portfolio_value:.2f}

风险规则：{risk_text}

请给出今日交易指令（中文理由）：
"""

    response = get_minimax_response([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])

    signals = []
    for line in response.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or "NONE" in line.upper():
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        action = parts[0].upper()
        if action not in ("BUY", "SELL", "SHORT", "COVER", "HOLD"):
            continue
        ticker = parts[1]
        shares = 0
        reason = " ".join(parts[2:]) if len(parts) > 2 else ""
        if action in ("BUY", "SHORT") and len(parts) >= 3 and "股" in parts[2]:
            try:
                shares = float(parts[2].replace("股", ""))
            except:
                shares = 0
        signals.append({
            "action": action,
            "ticker": ticker,
            "shares": shares,
            "reason": reason,
            "raw": line,
        })

    return signals


def simulate_ai_trading(
    watchlist: list[str],
    initial_cash: float,
    risk_config: RiskConfig,
    top_n: int = 8,
    short_allowed: bool = True,
) -> dict:
    """
    模拟 AI 执行交易（支持做空）
    返回完整的模拟结果和交易记录
    """
    rm = RiskManager(risk_config)
    ranked = rank_stocks(watchlist, top_n=top_n)
    live_prices = {s["ticker"]: s["price"] for s in ranked}

    cash = initial_cash
    positions = {}  # ticker -> {shares, avg_cost, entry_date, stop, target, position_type}
    trade_log = []
    start_date = datetime.now().strftime("%Y-%m-%d")

    # 空头卖出的"虚拟保证金"冻结
    short_margin_used = 0.0

    signals = ai_generate_signals(
        watchlist=watchlist,
        portfolio_cash=cash,
        portfolio_value=initial_cash,
        current_positions={},
        risk_config=risk_config,
        top_n=top_n,
        short_allowed=short_allowed,
    )

    for sig in signals:
        ticker = sig["ticker"]
        action = sig["action"]

        if ticker not in live_prices:
            continue
        price = live_prices[ticker]

        if action == "BUY":
            max_amount = initial_cash * (risk_config.max_single_position_pct / 100)
            shares = sig["shares"] if sig["shares"] > 0 else int(max_amount / price)
            cost = shares * price
            available = cash - short_margin_used
            if cost <= available * 0.95 and ticker not in positions:
                cash -= cost
                positions[ticker] = {
                    "shares": shares,
                    "avg_cost": price,
                    "stop": round(price * (1 - risk_config.stop_loss_default_pct / 100), 2),
                    "target": round(price * (1 + risk_config.profit_taking_pct / 100), 2),
                    "entry_date": start_date,
                    "position_type": "LONG",
                }
                trade_log.append({
                    "date": start_date, "action": "BUY", "ticker": ticker,
                    "price": price, "shares": shares, "cost": round(cost, 2),
                    "reason": sig["reason"],
                })

        elif action == "SHORT" and short_allowed:
            max_shares = int((cash * 0.5) / price)  # 50%保证金
            shares = sig["shares"] if sig["shares"] > 0 else max_shares
            shares = min(shares, max_shares)
            if shares > 0 and ticker not in positions:
                proceeds = shares * price
                margin = proceeds * 0.5  # 50%保证金
                cash += proceeds  # 卖空获得资金（但冻结保证金）
                short_margin_used += margin
                positions[ticker] = {
                    "shares": shares,
                    "avg_cost": price,
                    "stop": round(price * (1 + risk_config.stop_loss_default_pct / 100), 2),
                    "target": round(price * (1 - risk_config.profit_taking_pct / 100), 2),
                    "entry_date": start_date,
                    "position_type": "SHORT",
                }
                trade_log.append({
                    "date": start_date, "action": "SHORT", "ticker": ticker,
                    "price": price, "shares": shares, "cost": round(proceeds, 2),
                    "reason": sig["reason"],
                })

        elif action == "SELL" and ticker in positions and positions[ticker]["position_type"] == "LONG":
            pos = positions[ticker]
            pnl = (price - pos["avg_cost"]) * pos["shares"]
            cash += pos["shares"] * price
            trade_log.append({
                "date": start_date, "action": "SELL", "ticker": ticker,
                "price": price, "shares": pos["shares"], "pnl": round(pnl, 2),
                "reason": sig["reason"],
            })
            del positions[ticker]

        elif action == "COVER" and ticker in positions and positions[ticker]["position_type"] == "SHORT":
            pos = positions[ticker]
            pnl = (pos["avg_cost"] - price) * pos["shares"]  # 空头盈利=开仓价-平仓价
            cover_cost = pos["shares"] * price
            cash -= cover_cost
            short_margin_used -= pos["avg_cost"] * pos["shares"] * 0.5
            trade_log.append({
                "date": start_date, "action": "COVER", "ticker": ticker,
                "price": price, "shares": pos["shares"], "pnl": round(pnl, 2),
                "reason": sig["reason"],
            })
            del positions[ticker]

    # 计算模拟组合价值
    long_value = sum(p["shares"] * live_prices.get(t, p["avg_cost"])
                     for t, p in positions.items() if p["position_type"] == "LONG")
    short_cost = sum(p["shares"] * live_prices.get(t, p["avg_cost"])
                     for t, p in positions.items() if p["position_type"] == "SHORT")
    # 空头持仓的市值表示"需要花多少钱买回"
    portfolio_value = cash + long_value - short_cost + short_margin_used
    pnl = portfolio_value - initial_cash
    pnl_pct = (pnl / initial_cash) * 100

    return {
        "signals": signals,
        "positions": {t: {k: v for k, v in p.items() if k != "position_type"}
                      for t, p in positions.items()},
        "trade_log": trade_log,
        "cash": round(cash, 2),
        "short_margin_used": round(short_margin_used, 2),
        "portfolio_value": round(portfolio_value, 2),
        "initial_cash": initial_cash,
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
        "long_positions": [t for t, p in positions.items() if p["position_type"] == "LONG"],
        "short_positions": [t for t, p in positions.items() if p["position_type"] == "SHORT"],
    }