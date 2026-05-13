"""
AI 加密货币交易策略引擎
支持做多(BUY)和做空(SHORT)，使用 MiniMax API 生成交易信号
broker-agnostic：可用于任何支持永续合约的交易所
"""
import os
import sys
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.hermes/.env"))

sys.path.insert(0, os.path.dirname(__file__))

from crypto_scorer import rank_cryptos, score_crypto
from crypto_risk_manager import CryptoRiskManager, CryptoRiskConfig
from crypto_data import DEFAULT_EXCHANGE, PERP_INFO

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


def ai_generate_crypto_signals(
    symbols: list[str],
    portfolio_value: float,
    current_positions: dict,   # {symbol: {side, size, entry_price, leverage, ...}}
    risk_config: CryptoRiskConfig,
    exchange: str = DEFAULT_EXCHANGE,
    top_n: int = 8,
    short_allowed: bool = True,
) -> list[dict]:
    """
    调用 AI 生成加密货币交易信号
    支持：
    - BUY [symbol] [contracts]张 — 开多头仓位
    - SELL [symbol] — 平多头仓位
    - SHORT [symbol] [contracts]张 — 开空头仓位（高杠杆，需保证金充足）
    - COVER [symbol] — 平空头仓位
    - HOLD — 继续持有
    """

    ranked = rank_cryptos(symbols, exchange=exchange, top_n=top_n)
    if not ranked:
        return []

    ranked_data = []
    for s in ranked:
        perp_info = PERP_INFO.get(s["symbol"], {})
        leverage = s.get("leverage", perp_info.get("leverage", 50))
        reasons = s.get("reasons", [])
        ranked_data.append(
            f"- {s['symbol']}: "
            f"价格=${s['price']}, RSI={s['rsi']}, "
            f"5日={s['change_5d_pct']}%, ATR={s['atr_pct']}%, "
            f"布林={s['bb_pct']}, 资金费率={s['funding_rate']}%, "
            f"杠杆上限={leverage}x, "
            f"信号: {', '.join(reasons) if reasons else '中性观望'}"
        )
    market_table = "\n".join(ranked_data)

    pos_lines = []
    for symbol, pos in current_positions.items():
        side = pos.get("side", "LONG")
        direction = "🔴做空" if side == "SHORT" else "🟢做多"
        pnl_pct = pos.get("unrealized_pnl_pct", 0)
        pos_lines.append(
            f"- {symbol}: {direction} {pos.get('size', 0)}张, "
            f"均价${pos.get('entry_price', 0)}, 当前${pos.get('mark_price', 0)}, "
            f"盈亏{pnl_pct}%, 杠杆{pos.get('leverage', 1)}x, "
            f"强平价${pos.get('liquidation_price', 0)}"
        )
    positions_text = "\n".join(pos_lines) if pos_lines else "空仓"

    risk_text = (
        f"止损(ATR×2) | 止盈(ATR×4) | "
        f"单笔上限:{risk_config.max_single_position_pct}% | "
        f"最长持仓:{risk_config.max_holding_hours}h | "
        f"最大同时持仓:{risk_config.max_positions}个 | "
        f"允许做空:{'是' if short_allowed else '否'} | "
        f"默认杠杆:{risk_config.default_leverage}x"
    )

    short_instruction = """
做空规则（当 short_allowed=true 时）：
- SHORT n张 — 预计币价下跌，通过永续合约做空获利（可高达75x杠杆）
- COVER n张 — 买回仓位平掉空头
- 做空仓位需要50%保证金（可用资金必须充足）
- 做空止损：币价超过开仓价 + stop_loss_default_pct% 时强制止损
- 做空止盈：币价跌破开仓价 - profit_taking_pct% 时止盈
- 做空盈亏计算：空头盈利 = 卖出开仓价 - 买回平仓价（价格下跌赚钱）
- 资金费率：每8小时结算，做空时若资金费率为正则需支付费用

加密货币高杠杆警告：
- 高杠杆意味着高风险，建议保守杠杆(≤10x)操作
- 75x杠杆可在1分钟内爆仓，必须严格止损
- 建议使用 ATR 动态止损替代固定%止损
""" if short_allowed else ""

    system_prompt = f"""你是一个加密货币永续合约交易员，擅长做多和做空。

交易指令：
- BUY [symbol] [contracts]张 — 开多头仓位（用保证金买入合约）
- SELL [symbol] — 卖出平掉多头仓位
- SHORT [symbol] [contracts]张 — 卖出开仓（做空，需要保证金充足）
- COVER [symbol] — 买回仓位平掉空头
- HOLD — 继续持有，不操作

{short_instruction}
风险规则：
- 单笔仓位不超过总资金的{risk_config.max_single_position_pct}%
- 同时持仓不超过{risk_config.max_positions}个（含做空）
- 最长持仓 {risk_config.max_holding_hours} 小时
- 建议使用 ATR 动态止损，不建议固定%止损

按以下格式输出（每一行一个指令）：
BUY [symbol] [contracts]张 | 原因（如：RSI超卖，反弹做多）
SELL [symbol] | 原因（如：达到止盈目标）
SHORT [symbol] [contracts]张 | 原因（如：RSI超买，做空回调）
COVER [symbol] | 原因（如：触及止损线）
HOLD [symbol] | 原因（如：波动过大，观望）

如果没有任何信号，写：NONE
"""

    user_prompt = f"""加密货币市场排名（按技术评分）：
{market_table}

当前投资组合：
{positions_text}

组合总价值：${portfolio_value:,.2f}

风险规则：{risk_text}

请给出交易指令（中文理由）：
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
        symbol = parts[1].replace("张", "")
        contracts = 0
        reason = " ".join(parts[2:]).replace("|", "").strip()
        if action in ("BUY", "SHORT") and len(parts) >= 3:
            try:
                contracts = float(parts[2].replace("张", ""))
            except:
                contracts = 0
        signals.append({
            "action": action,
            "symbol": symbol,
            "contracts": contracts,
            "reason": reason,
            "raw": line,
        })

    return signals


def simulate_ai_crypto_trading(
    symbols: list[str],
    initial_cash: float,
    risk_config: CryptoRiskConfig,
    exchange: str = DEFAULT_EXCHANGE,
    top_n: int = 8,
    short_allowed: bool = True,
) -> dict:
    """
    模拟 AI 执行加密货币交易（支持做空 + 高杠杆）
    """
    rm = CryptoRiskManager(risk_config)
    ranked = rank_cryptos(symbols, exchange=exchange, top_n=top_n)
    live_prices = {s["symbol"]: s["price"] for s in ranked if s.get("price")}

    cash = initial_cash
    positions = {}  # symbol -> {side, size, entry_price, leverage, stop, target, ...}
    trade_log = []
    start_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    signals = ai_generate_crypto_signals(
        symbols=symbols,
        portfolio_value=initial_cash,
        current_positions={},
        risk_config=risk_config,
        exchange=exchange,
        top_n=top_n,
        short_allowed=short_allowed,
    )

    for sig in signals:
        symbol = sig["symbol"]
        action = sig["action"]

        if symbol not in live_prices:
            continue
        price = live_prices[symbol]
        perp_info = PERP_INFO.get(symbol, {})
        leverage = min(risk_config.default_leverage, perp_info.get("leverage", 50))

        if action == "BUY":
            max_margin = initial_cash * (risk_config.max_single_position_pct / 100)
            contracts = sig["contracts"] if sig["contracts"] > 0 else int(max_margin / price)
            cost = contracts * price
            available = cash * 0.95  # 保留5%手续费缓冲
            if cost <= available and symbol not in positions:
                positions[symbol] = {
                    "side": "LONG",
                    "size": contracts,
                    "entry_price": price,
                    "mark_price": price,
                    "stop": round(price * (1 - risk_config.stop_loss_default_pct / 100), 4),
                    "target": round(price * (1 + risk_config.profit_taking_pct / 100), 4),
                    "entry_date": start_date,
                    "leverage": leverage,
                    "liquidation_price": rm.calc_liquidation_price(price, "LONG", leverage),
                    "margin": cost / leverage,
                }
                trade_log.append({
                    "date": start_date, "action": "BUY", "symbol": symbol,
                    "price": price, "contracts": contracts, "cost": round(cost, 4),
                    "reason": sig["reason"],
                })

        elif action == "SHORT" and short_allowed:
            max_margin = initial_cash * (risk_config.short_max_position_pct / 100)
            margin_per_contract = price * (risk_config.short_margin_pct / 100)
            max_contracts = int(max_margin / margin_per_contract)
            contracts = sig["contracts"] if sig["contracts"] > 0 else max_contracts
            contracts = min(contracts, max_contracts)
            if contracts > 0 and symbol not in positions:
                positions[symbol] = {
                    "side": "SHORT",
                    "size": contracts,
                    "entry_price": price,
                    "mark_price": price,
                    "stop": round(price * (1 + risk_config.stop_loss_default_pct / 100), 4),
                    "target": round(price * (1 - risk_config.profit_taking_pct / 100), 4),
                    "entry_date": start_date,
                    "leverage": leverage,
                    "liquidation_price": rm.calc_liquidation_price(price, "SHORT", leverage),
                    "margin": contracts * price * (risk_config.short_margin_pct / 100),
                }
                trade_log.append({
                    "date": start_date, "action": "SHORT", "symbol": symbol,
                    "price": price, "contracts": contracts, "cost": round(contracts * price, 4),
                    "reason": sig["reason"],
                })

        elif action == "SELL" and symbol in positions and positions[symbol]["side"] == "LONG":
            pos = positions[symbol]
            pnl = (price - pos["entry_price"]) * pos["size"]
            trade_log.append({
                "date": start_date, "action": "SELL", "symbol": symbol,
                "price": price, "contracts": pos["size"], "pnl": round(pnl, 4),
                "reason": sig["reason"],
            })
            del positions[symbol]

        elif action == "COVER" and symbol in positions and positions[symbol]["side"] == "SHORT":
            pos = positions[symbol]
            pnl = (pos["entry_price"] - price) * pos["size"]  # 空头盈利 = 开仓价 - 平仓价
            trade_log.append({
                "date": start_date, "action": "COVER", "symbol": symbol,
                "price": price, "contracts": pos["size"], "pnl": round(pnl, 4),
                "reason": sig["reason"],
            })
            del positions[symbol]

    # 计算模拟组合价值
    long_value = sum(p["size"] * live_prices.get(s, p["entry_price"])
                     for s, p in positions.items() if p["side"] == "LONG")
    short_cost = sum(p["size"] * live_prices.get(s, p["entry_price"])
                     for s, p in positions.items() if p["side"] == "SHORT")
    margin_used = sum(p.get("margin", 0) for p in positions.values())
    portfolio_value = cash + long_value - short_cost
    pnl = portfolio_value - initial_cash
    pnl_pct = (pnl / initial_cash) * 100

    return {
        "signals": signals,
        "positions": positions,
        "trade_log": trade_log,
        "cash": round(cash, 2),
        "margin_used": round(margin_used, 2),
        "portfolio_value": round(portfolio_value, 2),
        "initial_cash": initial_cash,
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
        "long_positions": [s for s, p in positions.items() if p["side"] == "LONG"],
        "short_positions": [s for s, p in positions.items() if p["side"] == "SHORT"],
    }


# ======== CLI ========

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(os.path.expanduser("~/.hermes/.env"))
    from config import CRYPTO_WATCHLIST

    risk_config = CryptoRiskConfig(
        max_single_position_pct=10.0,
        default_leverage=10,
        short_allowed=True,
        max_positions=5,
    )

    print("📊 AI 加密货币策略信号\n")
    signals = ai_generate_crypto_signals(
        symbols=CRYPTO_WATCHLIST,
        portfolio_value=10000.0,
        current_positions={},
        risk_config=risk_config,
        exchange=DEFAULT_EXCHANGE,
        top_n=8,
        short_allowed=True,
    )

    if signals:
        for sig in signals:
            print(f"  {sig['action']} {sig['symbol']} {sig['contracts']}张 | {sig['reason']}")
    else:
        print("  无信号（HOLD）")