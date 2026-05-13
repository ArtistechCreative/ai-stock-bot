"""
AI 交易建议引擎（本地计算 + MiniMax AI 双轨）
- 本地计算：基于技术指标和风控规则，计算买入/止损/止盈价
- AI 增强：调用 MiniMax 提供更深层的逻辑分析和催化剂判断
- 两轨并行，本地优先；AI 失败时自动降级到本地计算
"""
import os, sys, json, requests, time, re
from datetime import datetime
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.hermes/.env"))

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/anthropic")
MODEL = "MiniMax-M2.7"


def local_recommendation(ticker: str, name: str, price: float,
                          pe: float, change_5d: float,
                          volume_ratio: float, beta: float,
                          rsi: float = None, macd: float = None,
                          stop_loss_pct: float = 8.0,
                          profit_target_pct: float = 15.0,
                          risk_level: str = "normal") -> dict:
    """
    本地计算交易建议（不需要 AI）
    基于：技术指标当前位置 + 波动率调整 + 支撑阻力位估算
    """
    # 波动率调整：beta 越高，止损越宽
    vol_adjust = max(0.5, min(2.0, beta / 1.0))

    # RSI 调整：超买时降低目标位
    rsi_adjust = 1.0
    if rsi is not None:
        if rsi > 75:
            rsi_adjust = 0.85  # 超买，降低目标
        elif rsi < 30:
            rsi_adjust = 1.15  # 超卖，提高目标

    # 成交量调整：放量突破更可信
    vol_confirm = 1.0
    if volume_ratio > 2.0:
        vol_confirm = 1.1  # 量能强劲
    elif volume_ratio < 0.7:
        vol_confirm = 0.9  # 量能萎缩

    # 动态止损
    adjusted_sl = stop_loss_pct * vol_adjust
    stop_loss = round(price * (1 - adjusted_sl / 100), 2)

    # 动态止盈（第一目标）
    tp1_multiplier = profit_target_pct * vol_confirm * rsi_adjust
    take_profit_1 = round(price * (1 + tp1_multiplier / 100), 2)

    # 追踪止损（达到 TP1 后激活，从最高盈利点回撤 N 点触发）
    trailing_stop_points = 50  # 点（加密货币适用）

    # 买入价：回踩 5 日均线附近买入（估算）
    buy_price = round(price, 2)  # 默认直接市价买入

    # 仓位计算（Kelly Criterion 简化版）
    # 基于 PE 和波动率估算仓位
    if pe and pe > 0:
        if pe < 20:
            pe_factor = 1.2  # 低估值，可以重仓
        elif pe < 40:
            pe_factor = 1.0
        else:
            pe_factor = 0.8  # 高估值，控制仓位

    pos = min(20, max(5, int(10 * pe_factor * (2.0 / vol_adjust))))

    # 趋势判断
    if change_5d > 8:
        entry_type = "追涨买入（强势突破）"
    elif change_5d > 3:
        entry_type = "顺势买入（趋势确立）"
    elif change_5d > 0:
        entry_type = "回调买入（震荡偏强）"
    else:
        entry_type = "观望或轻仓试多"

    # 风险评级
    risk = "高波动" if beta > 1.5 else ("防御型" if beta < 0.8 else "中等波动")

    return {
        "ticker": ticker,
        "name": name,
        "source": "LOCAL",
        "action": "BUY" if change_5d > -5 else "WATCH",
        "buy_price": buy_price,
        "stop_loss": stop_loss,
        "take_profit_1": take_profit_1,
        "trailing_stop_points": trailing_stop_points,
        "position_size": pos,
        "adjusted_stop_loss_pct": round(adjusted_sl, 1),
        "adjusted_tp1_pct": round(tp1_multiplier, 1),
        "entry_reason": f"{entry_type} | {risk} | 5日+{change_5d:.1f}%",
        "risk": f"止损{adjusted_sl:.1f}% | Beta={beta} | PE={pe}",
        "rsi": rsi,
        "volume_ratio": volume_ratio,
    }


def ai_recommendation_with_retry(ticker: str, name: str, price: float,
                                   pe: float, change_5d: float,
                                   volume_ratio: float, beta: float,
                                   rsi: float = None, macd: float = None,
                                   stop_loss_pct: float = 8.0,
                                   profit_target_pct: float = 15.0,
                                   max_retries: int = 3) -> dict:
    """
    调用 MiniMax AI 生成建议，带重试（529 过载时自动降级到本地）
    """
    indicators = []
    if rsi is not None:
        indicators.append(f"RSI={rsi:.1f}")
    if macd is not None:
        indicators.append(f"MACD={macd:.2f}")
    ind_str = ", ".join(indicators) if indicators else "暂无"

    body = f"""你是短线交易策略助手。请为以下股票给出具体交易建议：

{ticker} ({name})
价格：${price:.2f}
PE：{pe}
5日涨跌：{change_5d:.2f}%
成交量比：{volume_ratio}x
Beta：{beta}
技术指标：{ind_str}

请严格按以下格式输出（每行必须有冒号）：

TICKER: {ticker}
ACTION: BUY / WATCH / SKIP
BUY_PRICE: ${price:.2f}
STOP_LOSS: $XXX.XX
TAKE_PROFIT_1: $XXX.XX
TAKE_PROFIT_2: $XXX.XX
POSITION_SIZE: XX%
ENTRY_REASON: 一句话说明为什么买
RISK: 一句话说明最大风险
"""

    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    payload = {
        "model": MODEL,
        "max_tokens": 600,
        "messages": [{"role": "user", "content": body}]
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{MINIMAX_BASE_URL}/v1/messages",
                headers=headers,
                json=payload,
                timeout=35,
            )

            if resp.status_code == 529:
                wait = 2 ** attempt * 3
                print(f"  ⏳ API 过载（529），{wait}秒后重试（{attempt+1}/{max_retries}）...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()
            content = data.get("content", [])
            text_blocks = [c.get("text", "") for c in content if c.get("type") == "text"]
            raw = "\n".join(text_blocks) if text_blocks else ""

            if not raw.strip():
                raise ValueError("Empty AI response")

            result = _parse_ai_output(raw, price, stop_loss_pct, profit_target_pct)
            result["source"] = "AI"
            result["raw"] = raw
            return result

        except Exception as e:
            print(f"  ⚠️ AI 请求失败（尝试 {attempt+1}/{max_retries}）: {e}")
            if attempt == max_retries - 1:
                # 降级到本地计算
                print(f"  ↩️ 降级到本地计算...")
                result = local_recommendation(
                    ticker=ticker, name=name, price=price, pe=pe,
                    change_5d=change_5d, volume_ratio=volume_ratio, beta=beta,
                    rsi=rsi, macd=macd,
                    stop_loss_pct=stop_loss_pct, profit_target_pct=profit_target_pct,
                )
                result["ai_fallback"] = True
                return result

    # 如果全部重试失败，直接用本地
    result = local_recommendation(
        ticker=ticker, name=name, price=price, pe=pe,
        change_5d=change_5d, volume_ratio=volume_ratio, beta=beta,
        rsi=rsi, macd=macd,
        stop_loss_pct=stop_loss_pct, profit_target_pct=profit_target_pct,
    )
    result["ai_fallback"] = True
    return result


def _parse_ai_output(raw: str, current_price: float,
                     stop_loss_pct: float, profit_target_pct: float) -> dict:
    """解析 AI 输出"""
    lines = raw.strip().split("\n")

    result = {
        "ticker": "",
        "action": "WATCH",
        "buy_price": current_price,
        "stop_loss": round(current_price * (1 - stop_loss_pct / 100), 2),
        "take_profit_1": round(current_price * (1 + profit_target_pct / 100), 2),
        "trailing_stop_points": 50,
        "position_size": 10,
        "entry_reason": "",
        "risk": "",
        "raw": raw,
    }

    for line in lines:
        line = line.strip()
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().upper()
        val = val.strip()

        if key == "TICKER":
            result["ticker"] = val
        elif key == "ACTION":
            if "BUY" in val.upper() and "SKIP" not in val.upper():
                result["action"] = "BUY"
            elif "SKIP" in val.upper():
                result["action"] = "SKIP"
            else:
                result["action"] = "WATCH"
        elif key == "BUY_PRICE":
            p = _extract_price(val)
            if p: result["buy_price"] = p
        elif key == "STOP_LOSS":
            p = _extract_price(val)
            if p: result["stop_loss"] = p
        elif key == "TAKE_PROFIT_1":
            p = _extract_price(val)
            if p: result["take_profit_1"] = p
        elif key == "TRAILING_STOP_POINTS":
            m = re.search(r'(\d+)', val)
            if m: result["trailing_stop_points"] = int(m.group(1))
        elif key == "POSITION_SIZE":
            m = re.search(r'(\d+)', val)
            if m: result["position_size"] = min(20, max(1, int(m.group(1))))
        elif key == "ENTRY_REASON":
            result["entry_reason"] = val
        elif key == "RISK":
            result["risk"] = val

    return result


def _extract_price(text: str):
    m = re.search(r'\$?([\d.]+)', text)
    if m:
        try: return float(m.group(1))
        except: pass
    return None


def _get_per_trade_limit() -> float:
    """从 Google Sheet 账户总览读取单笔最大投入金额。"""
    try:
        _ref = os.path.expanduser("~/.hermes/skills/productivity/ai-stock-trading-bot/references")
        if _ref not in sys.path:
            sys.path.insert(0, _ref)
        from google_sheets_portfolio import get_account_overview
        overview = get_account_overview()
        capital = float(overview.get("total_capital") or 25)
        risk_ratio = float(overview.get("risk_ratio") or 25)
        return capital * (risk_ratio / 100)
    except Exception:
        return 6.25  # 默认 $25 × 25%


_per_trade_limit = _get_per_trade_limit()


def format_signal(rec: dict) -> str:
    """格式化为信号消息"""
    ticker = rec["ticker"]
    action = rec["action"]

    if action == "SKIP":
        return ""

    src_emoji = "🤖" if rec.get("source") == "AI" else "⚙️"
    action_emoji = {"BUY": "🟢", "WATCH": "🟡"}.get(action, "⚪")

    lines = [
        f"{src_emoji}{action_emoji} **{ticker}** ({rec.get('name', '')}) — {action}",
        f"━━━━━━━━━━━━━━━━━━",
        f"📍 买入价：${rec['price']:.2f}",
        f"🛑 止损价：${rec['stop_loss']:.2f}",
        f"🎯 第一止盈：${rec['take_profit_1']:.2f}",
    ]

    if rec.get("trailing_stop_points") is not None:
        lines.append(f"📍 追踪止损：{rec['trailing_stop_points']} 点（TP1 激活后从最高点回撤触发）")

    # position_size（%）：以 total_capital × risk_ratio% 为单笔限额基准
    # 注意：这是金额上限（%），不是Kelly公式的杠杆比例
    # 外汇/大宗以"合约数/lots"表示；加密以"币数量"表示
    pos = rec.get("position_size", 10)
    lines.append(f"📊 建议仓位：{pos}%（${_per_trade_limit:.2f}限额内）")

    # RSI 如果有的话
    if rec.get("rsi") is not None:
        lines.append(f"📈 RSI：{rec['rsi']:.1f}")

    if rec.get("entry_reason"):
        lines.append(f"💡 {rec['entry_reason']}")

    if rec.get("risk"):
        lines.append(f"⚠️ {rec['risk']}")

    if rec.get("ai_fallback"):
        lines.append(f"[AI 服务降级，本地计算]")

    return "\n".join(lines)


def get_all_signals(tickers: list[str], stop_loss_pct: float = 8.0,
                    profit_target_pct: float = 15.0,
                    use_ai: bool = True) -> list[dict]:
    """
    对多个股票生成交易信号
    """
    from scorer import score_stock

    signals = []
    for ticker in tickers:
        try:
            s = score_stock(ticker)
            if not s:
                continue

            if use_ai and MINIMAX_API_KEY:
                rec = ai_recommendation_with_retry(
                    ticker=ticker, name=s["name"], price=s["price"],
                    pe=s["pe"], change_5d=s["change_5d_pct"],
                    volume_ratio=s["volume_ratio"], beta=s["beta"],
                    rsi=s.get("rsi"), macd=s.get("macd"),
                    stop_loss_pct=stop_loss_pct,
                    profit_target_pct=profit_target_pct,
                )
            else:
                rec = local_recommendation(
                    ticker=ticker, name=s["name"], price=s["price"],
                    pe=s["pe"], change_5d=s["change_5d_pct"],
                    volume_ratio=s["volume_ratio"], beta=s["beta"],
                    rsi=s.get("rsi"), macd=s.get("macd"),
                    stop_loss_pct=stop_loss_pct,
                    profit_target_pct=profit_target_pct,
                )

            rec["price"] = s["price"]
            rec["change_5d_pct"] = s["change_5d_pct"]
            rec["score"] = s.get("score", 0)
            signals.append(rec)

        except Exception as e:
            print(f"  [!] {ticker}: {e}")

    # 排序：BUY 信号优先，然后按综合评分
    def sort_key(s):
        action_order = {"BUY": 0, "WATCH": 1, "SKIP": 2}
        return (action_order.get(s["action"], 3), -s.get("score", 0))

    signals.sort(key=sort_key)
    return signals


# ======== CLI 测试 ========
if __name__ == "__main__":
    print("🧪 测试交易建议引擎...")
    signals = get_all_signals(["NVDA", "TSLA", "AAPL"], use_ai=False)
    for s in signals:
        print(format_signal(s))
        print()