"""
AI 选股 + 自动交易引擎
去除 Moomoo 依赖，改用 yfinance 数据层 + broker-agnostic 信号推送
实际下单由用户手动执行（MITRADE/其他平台），系统仅负责：
  1. 数据拉取（yfinance）
  2. AI 评分排序
  3. DL 预测（可选）
  4. 生成交易信号（含建议杠杆）+ Telegram 推送
  5. 持仓监控 + 止损/止盈提醒
"""
import os, sys, json, time
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional

# ── 资产杠杆分层（MITRADE 实际支持 200x 外汇 + 加密，100x 主流币）─────────────
# tier 1 — 主流加密永续（BTC/ETH/BNB）：200x
# tier 2 — 主流山寨（SOL/XRP/ADA/AVAX/LINK/DOT等）：100x
# tier 3 — 高波动币MEME/新币：50x
# tier 4 — 外汇 CFD（EUR/USD GBP/USD AUD/USD USD/JPY）：200x（MITRADE）
# tier 5 — 黄金/原油 CFD：100x
# tier 6 — 指数 ETF：20x
# tier 7 — 港股/马来西亚股 CFD：10x
ASSET_LEVERAGE_TIERS = {
    # tier 1 — 主流币（BTC/ETH/BNB）
    "BTC/USDT":  {"tier": 1, "max_leverage": 100},
    "ETH/USDT":  {"tier": 1, "max_leverage": 100},
    "BNB/USDT":  {"tier": 1, "max_leverage": 75},
    # tier 2 — 主流山寨
    "SOL/USDT":  {"tier": 2, "max_leverage": 50},
    "XRP/USDT":  {"tier": 2, "max_leverage": 50},
    "ADA/USDT":  {"tier": 2, "max_leverage": 50},
    "AVAX/USDT": {"tier": 2, "max_leverage": 50},
    "LINK/USDT": {"tier": 2, "max_leverage": 50},
    "DOT/USDT":  {"tier": 2, "max_leverage": 50},
    "MATIC/USDT":{"tier": 2, "max_leverage": 50},
    "LTC/USDT":  {"tier": 2, "max_leverage": 50},
    "UNI/USDT":  {"tier": 2, "max_leverage": 50},
    "APT/USDT":  {"tier": 2, "max_leverage": 50},
    "ARB/USDT":  {"tier": 2, "max_leverage": 50},
    "INJ/USDT":  {"tier": 2, "max_leverage": 50},
    "SUI/USDT":  {"tier": 2, "max_leverage": 50},
    "TIA/USDT":  {"tier": 2, "max_leverage": 50},
    # tier 3 — 高波动 MEME 币
    "DOGE/USDT": {"tier": 3, "max_leverage": 50},
    "SHIB/USDT": {"tier": 3, "max_leverage": 50},
    "PEPE/USDT": {"tier": 3, "max_leverage": 50},
    "WIF/USDT":  {"tier": 3, "max_leverage": 50},
    # tier 4 — 外汇 CFD（MITRADE 200x）
    "EURUSD=X":  {"tier": 4, "max_leverage": 200},
    "GBPUSD=X":  {"tier": 4, "max_leverage": 200},
    "AUDUSD=X":  {"tier": 4, "max_leverage": 200},
    "USDJPY=X":  {"tier": 4, "max_leverage": 200},
    "EURGBP=X":  {"tier": 4, "max_leverage": 200},
    "EURJPY=X":  {"tier": 4, "max_leverage": 200},
    "GBPJPY=X":  {"tier": 4, "max_leverage": 200},
    # tier 5 — 大宗商品 CFD（MITRADE）
    "GC=F":      {"tier": 5, "max_leverage": 100},  # 黄金
    "CL=F":      {"tier": 5, "max_leverage": 100},  # 原油
    # tier 6 — 指数 ETF
    "ES=F":      {"tier": 6, "max_leverage": 20},
    "NQ=F":      {"tier": 6, "max_leverage": 20},
    "^KLSE":     {"tier": 6, "max_leverage": 10},
    "^STI":      {"tier": 6, "max_leverage": 10},
}


def _read_sheet_total_capital() -> float:
    """从 Google Sheet 账户总览读取真实总投入本金，失败时返回 None。"""
    try:
        import sys as _sys
        _ref = os.path.expanduser("~/.hermes/skills/productivity/ai-stock-trading-bot/references")
        if _ref not in _sys.path:
            _sys.path.insert(0, _ref)
        from google_sheets_portfolio import get_account_overview
        overview = get_account_overview()
        capital = overview.get("total_capital")
        if capital is not None:
            return float(capital)
    except Exception:
        pass
    return None


def _signal_leverage(ticker: str, direction: str, price: float, stop_loss: float, user_risk: str = "medium") -> float:
    """
    根据「每笔最多亏本金 X%」反推可用最大杠杆。

    核心公式：leverage = (max_loss_pct / stop_loss_pct) * 100
    逻辑：亏 max_loss_pct 本金 能扛 stop_loss_pct 的波动 → 杠杆 = 该比例 × 100
    例子：max_loss=10%, stop_loss=8% → (10/8)*100 = 125x（tier1封顶100x）
          max_loss=10%, stop_loss=5% → (10/5)*100 = 200x（tier1封顶100x）
          max_loss=5%,  stop_loss=2% → (5/2)*100  = 250x（tier1封顶100x）
    """
    tier_info = ASSET_LEVERAGE_TIERS.get(ticker, None)
    tier_max = tier_info["max_leverage"] if tier_info else 5.0

    # aggressive=20% 单笔最大亏损（激进模式可以一单亏20%本金）
    max_loss_pcts = {"conservative": 5.0, "medium": 10.0, "aggressive": 20.0}
    max_loss_pct = max_loss_pcts.get(user_risk, 10.0)

    if price > 0 and stop_loss > 0:
        stop_loss_pct = abs(stop_loss - price) / price * 100
    else:
        stop_loss_pct = 8.0

    # 核心：杠杆 = 能亏的 / 止损幅度 × 100
    raw_leverage = (max_loss_pct / stop_loss_pct) * 100.0
    if direction in ("SHORT", "SELL"):
        raw_leverage *= 0.8

    suggested = min(tier_max, max(2.0, raw_leverage))
    return float(suggested)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════════════
# 数据层（统一用 data_fetcher 的链式 fallback）
# yfinance + Alpha Vantage + Finnhub + Twelvedata
# 加密货币（带 '/' 如 BTC/USDT）走 OKX + CCXT
# ════════════════════════════════════════════════════════════════════

def get_live_quotes(tickers: list[str]) -> dict:
    """
    用 data_fetcher 批量获取股票/外汇/期货实时报价（自动 fallback）。
    加密货币（带 '/' 如 BTC/USDT）走 OKX + CCXT。
    """
    from data_fetcher import fetch_quotes_batch

    # ── 加密货币路由：走 OKX CCXT ───────────────────────────────
    crypto_tickers  = [t for t in tickers if "/" in t]
    stock_tickers   = [t for t in tickers if "/" not in t]
    quotes = {}

    if crypto_tickers:
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            from crypto_data import CryptoData
            cd = CryptoData(exchange="okx")
            for sym in crypto_tickers:
                try:
                    q = cd.fetch_quote(sym, use_cache=False)
                    if q:
                        quotes[sym] = {
                            "price"          : q.last_price,
                            "prev_close"     : None,
                            "volume"         : q.volume_24h,
                            "avg_volume"     : q.volume_24h or 1,
                            "market_cap"     : 0,
                            "pe"             : None,
                            "beta"           : 1.0,
                            "change_24h_pct" : q.change_24h_pct,
                        }
                except Exception as e:
                    print(f"  [!] {sym} CCXT 获取失败: {e}")
        except ImportError as e:
            print(f"  [!] crypto_data 模块导入失败: {e}")

    # ── 股票/外汇/期货：走 data_fetcher（自动 fallback）──────────
    if stock_tickers:
        fetched = fetch_quotes_batch(stock_tickers)
        quotes.update(fetched)

    return quotes


def get_account_info(portfolio_value: float, cash: float = None) -> dict:
    """模拟账户信息（实际数据从用户输入或 portfolio.json 读取）"""
    if cash is None:
        cash = portfolio_value * 0.3  # 默认假设 30% 现金
    return {
        "available_cash": cash,
        "portfolio_value": portfolio_value,
    }


# ======== 交易信号生成 ========

def generate_signals(
    ranked_stocks: list[dict],
    dl_predictions: list[dict],
    portfolio_cash: float,
    max_position_pct: float,
    risk_config,
) -> list[dict]:
    """
    综合 AI 评分 + DL 预测生成交易信号（支持多空）
    新架构（2026-05-18）：调用 strategy_optimizer.get_signal()，
    以评分排名为基准 + DL 高信心时加分奖励。板块路由（batch_predict_sector）
    自动识别 crypto/tech/defensive 板块并加载对应模型。

    ranked_stocks: rank_stocks() 返回的列表（含 score/rsi/macd_hist 等字段）
    dl_predictions: batch_predict_sector() 返回的列表（含 signal/confidence）
    返回: [{ticker, action, qty, price, reason, stop_loss, take_profit, ...}]
    """
    from bot.strategy_optimizer import StrategyOptimizer

    signals = []

    # ── Step 1: 调用 get_signal() 获取综合排名 ───────────────
    try:
        opt = StrategyOptimizer()
        combined = opt.get_signal(dl_predictions, ranked_stocks)
    except Exception as e:
        print(f"  [!] get_signal() 调用失败: {e}，回退到纯评分逻辑")
        combined = None

    # ── Step 2: 解包 combined，按评分+DL综合分选股 ──────────
    # combined 按 combined_score 降序排列
    if combined is not None:
        top_candidates = combined  # 已经是排序后的
    else:
        # 回退：只用评分排名
        top_candidates = [
            {"ticker": s["ticker"], "score_signal": 1 - i / max(len(ranked_stocks), 1), "dl_bonus": 0, "dl_confidence": 0}
            for i, s in enumerate(ranked_stocks)
        ]

    # ── Step 3: 生成 BUY 信号（综合分 > 0.6 且 DL信心足够 OR 综合分极高）──
    for c in top_candidates:
        ticker = c["ticker"]
        combined_score = c.get("combined_score", 0)
        dl_conf = abs(c.get("dl_confidence", 0))
        dl_bonus = c.get("dl_bonus", 0)

        # 找到对应的评分数据和 DL 预测
        stock = next((s for s in ranked_stocks if s["ticker"] == ticker), None)
        dl_pred = next((d for d in dl_predictions if d["ticker"] == ticker), None)

        if not stock:
            continue

        price = stock.get("price")
        if not price or price <= 0:
            continue

        is_crypto = stock.get("_is_crypto", False)

        # ── 加密货币特殊规则（技术指标驱动，无需 DL）───────
        if is_crypto:
            rsi = stock.get("rsi")
            macd_hist = stock.get("macd_hist")
            mom5 = stock.get("change_5d_pct", 0)
            bb_pct = stock.get("bb_pct")

            crypto_buy = (rsi is not None and rsi < 40) or (macd_hist is not None and macd_hist > 0 and mom5 > 0)
            crypto_short = rsi is not None and rsi > 70

            if crypto_buy:
                max_shares = (portfolio_cash * max_position_pct / 100) / price
                sl = round(price * (1 - risk_config.stop_loss_default_pct / 100), 2)
                tp1 = round(price * (1 + risk_config.profit_taking_pct / 100), 2)
                reason = f"AI评分{stock.get('score', 0)} + 加密技术信号(RSI={rsi:.1f}超卖)" if rsi and rsi < 40 else f"AI评分{stock.get('score', 0)} + MACD金叉"
                signals.append({
                    "ticker": ticker, "action": "BUY",
                    "qty": max_shares, "price": price,
                    "reason": reason,
                    "score": stock.get("score", 0),
                    "stop_loss": sl, "take_profit_1": tp1,
                    "trailing_stop_points": 50,
                    "position_type": "LONG", "strategy": "AI_SCORING",
                    "leverage": _signal_leverage(ticker, "BUY", price, sl),
                })
            elif crypto_short:
                max_shares = (portfolio_cash * max_position_pct / 100) / price
                sl = round(price * (1 + risk_config.stop_loss_default_pct / 100), 2)
                tp1 = round(price * (1 - risk_config.profit_taking_pct / 100), 2)
                signals.append({
                    "ticker": ticker, "action": "SHORT",
                    "qty": max_shares, "price": price,
                    "reason": f"AI评分{stock.get('score', 0)} + RSI={rsi:.1f}超买做空",
                    "score": stock.get("score", 0),
                    "stop_loss": sl, "take_profit_1": tp1,
                    "trailing_stop_points": 50,
                    "position_type": "SHORT", "strategy": "CRYPTO_TECH_SHORT",
                    "leverage": _signal_leverage(ticker, "SHORT", price, sl),
                })
            continue

        # ── 股票/ETF: 综合分阈值判断 ─────────────────────────
        # 条件：combined_score > 0.6 且 (dl_conf >= 70% 或 combined_score > 0.8)
        if combined_score > 0.6 and (dl_conf >= 70 or combined_score > 0.8):
            max_shares = (portfolio_cash * max_position_pct / 100) / price
            sl = round(price * (1 - risk_config.stop_loss_default_pct / 100), 2)
            tp1 = round(price * (1 + risk_config.profit_taking_pct / 100), 2)
            dl_info = f"DL+{dl_bonus:.3f}({dl_conf:.0f}%)" if dl_conf > 0 else "无DL"
            signals.append({
                "ticker": ticker, "action": "BUY",
                "qty": round(max_shares, 4), "price": price,
                "reason": f"综合分{combined_score:.3f} = 评分{c.get('score_signal', 0):.3f} + {dl_info}",
                "score": stock.get("score", 0),
                "stop_loss": sl, "take_profit_1": tp1,
                "trailing_stop_points": 50,
                "position_type": "LONG", "strategy": "AI_SCORING",
                "leverage": _signal_leverage(ticker, "BUY", price, sl),
            })

        # ── 股票做空信号：DL SELL + 高信心且综合分为负 ──────
        elif dl_pred and dl_pred.get("signal") == "SELL" and dl_conf >= 70 and combined_score < 0.3:
            max_shares = (portfolio_cash * risk_config.short_max_position_pct / 100) / price
            sl = round(price * (1 + risk_config.short_stop_loss_pct / 100), 2)
            tp1 = round(price * (1 - risk_config.short_take_profit_pct / 100), 2)
            signals.append({
                "ticker": ticker, "action": "SHORT",
                "qty": round(max_shares, 4), "price": price,
                "reason": f"综合分{combined_score:.3f} + DL SELL {dl_conf:.0f}%",
                "score": stock.get("score", 0),
                "stop_loss": sl, "take_profit_1": tp1,
                "trailing_stop_points": 50,
                "position_type": "SHORT", "strategy": "DL_SIGNAL",
                "leverage": _signal_leverage(ticker, "SHORT", price, sl),
            })

    return signals


# ======== 信号执行（改为 Telegram 推送，用户手动下单） ========

def execute_signals(
    signals: list[dict],
    dry_run: bool = True,
) -> list[dict]:
    """
    执行交易信号 → 写入 Google Sheet 信号池 + Telegram 推送
    用户在 Sheet 手动确认后才进入持仓汇总
    dry_run=True: 模拟执行
    """
    from ai_recommender import format_signal
    from telegram_bot import send_report

    # 延迟导入（需先加入 skill 引用目录）
    SHEET_WRITE = False
    try:
        _ref_dir = os.path.expanduser("~/.hermes/skills/productivity/ai-stock-trading-bot/references")
        if _ref_dir not in sys.path:
            sys.path.insert(0, _ref_dir)
        from google_sheets_portfolio import append_signal, get_pending_signals
        SHEET_WRITE = True
    except Exception as e:
        print(f"  ⚠️  Sheet 模块加载失败: {e}")

    # ── 防重复：读取现有 PENDING 信号列表 ────────────────────────────────────
    # 已存在于池中的 ticker 不再写入（Sheet 和 Telegram 都不重复）
    existing_pending_tickers: set = set()
    if SHEET_WRITE:
        try:
            pending = get_pending_signals()
            existing_pending_tickers = {p["ticker"].upper() for p in pending}
            if existing_pending_tickers:
                print(f"  ℹ️  信号池已有 PENDING: {existing_pending_tickers}")
        except Exception as e:
            print(f"  ⚠️  读取 PENDING 信号失败: {e}")

    results = []
    messages = []

    for sig in signals:
        ticker = sig["ticker"]
        action = sig["action"]
        qty = sig["qty"]
        price = sig["price"]
        reason = sig["reason"]

        # ── 防重复：已存在于 PENDING 的 ticker 跳过 ──────────────────────────
        ticker_upper = ticker.upper()
        if ticker_upper in existing_pending_tickers:
            print(f"  ⏭️  [{action}] {ticker} → 已在 PENDING，跳过")
            continue

        msg = format_signal(sig)
        if not msg:
            continue

        messages.append(msg)

        result = {
            "ticker": ticker,
            "action": action,
            "qty": qty,
            "price": price,
            "result": "SIMULATED" if dry_run else "REAL",
            "reason": reason,
            "order_id": f"SIM_{datetime.now().strftime('%H%M%S')}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        results.append(result)

        # 写入 Google Sheet 信号池（静默）
        if SHEET_WRITE:
            try:
                append_signal(
                    date=datetime.now().strftime("%Y-%m-%d"),
                    ticker=ticker,
                    direction=action,
                    qty=qty,
                    price=price,
                    strategy=sig.get("strategy", ""),
                    stop_loss=sig.get("stop_loss", 0),
                    take_profit=sig.get("take_profit_1", 0),
                    invested_amount=price * qty,
                    notes=reason,
                    leverage=sig.get("leverage", 0),
                )
                print(f"  📝 [{action}] {ticker} → 已写入信号池")
            except Exception as e:
                print(f"  ⚠️  Sheet 写入失败: {e}")

        direction_icon = "🟢" if action == "BUY" else "🔴"
        print(f"  {direction_icon} [{action}] {ticker} {qty}股 @ ${price:.2f} | {reason}")

    # 发送 Telegram 报告
    if messages:
        full_report = "\n\n".join(messages)
        chat_id = os.getenv("TELEGRAM_HOME_CHANNEL") or os.getenv("TELEGRAM_CHAT_ID") or "6801255591"
        ok = send_report(full_report, chat_id=chat_id)
        if ok:
            print("  ✅ Telegram 信号推送成功")
        else:
            print("  ❌ Telegram 推送失败")

    return results


# ======== 持仓监控 + 止损/止盈提醒 ========

def monitor_and_alert(
    positions: list[dict],
    risk_config,
) -> list[dict]:
    """
    监控持仓，触发止损/止盈时发送 Telegram 提醒
    positions: [{ticker, shares, avg_cost, position_type, entry_date, stop, target}]
    不再自动平仓，改为提醒用户手动操作
    """
    from telegram_bot import send_report

    alerts = []
    if not positions:
        return alerts

    quotes = get_live_quotes([p["ticker"] for p in positions])

    for pos in positions:
        ticker = pos["ticker"]
        shares = pos["shares"]
        avg_cost = pos["avg_cost"]
        pos_type = pos.get("position_type", "LONG")
        entry_date = pos.get("entry_date", "")
        stop_price = pos.get("stop", 0)
        target_price = pos.get("target", 0)

        q = quotes.get(ticker, {})
        current_price = q.get("price", avg_cost)

        if current_price <= 0:
            continue

        if pos_type == "LONG":
            pnl_pct = (current_price - avg_cost) / avg_cost * 100
            # 止损
            if stop_price > 0 and current_price <= stop_price:
                alerts.append({
                    "type": "STOP_LOSS",
                    "ticker": ticker,
                    "action": "SELL",
                    "price": current_price,
                    "pnl_pct": round(pnl_pct, 2),
                    "shares": shares,
                    "message": f"🛑 止损提醒：{ticker} 现价${current_price:.5f} ≤ 止损价${stop_price:.5f}（亏损{pnl_pct:.1f}%）请手动卖出！",
                })
            # 止盈
            elif target_price > 0 and current_price >= target_price:
                alerts.append({
                    "type": "TAKE_PROFIT",
                    "ticker": ticker,
                    "action": "SELL",
                    "price": current_price,
                    "pnl_pct": round(pnl_pct, 2),
                    "shares": shares,
                    "message": f"🎯 止盈提醒：{ticker} 现价${current_price:.5f} ≥ 目标价${target_price:.5f}（盈利{pnl_pct:.1f}%）请手动卖出！",
                })
        elif pos_type == "SHORT":
            # 空头：盈利 = 开仓价 - 当前价
            pnl_pct = (avg_cost - current_price) / avg_cost * 100
            # 空头止损（股价上涨超过 stop_loss%）
            if stop_price > 0 and current_price >= stop_price:
                alerts.append({
                    "type": "STOP_LOSS_SHORT",
                    "ticker": ticker,
                    "action": "COVER",
                    "price": current_price,
                    "pnl_pct": round(pnl_pct, 2),
                    "shares": shares,
                    "message": f"🛑 做空止损提醒：{ticker} 现价${current_price:.5f} ≥ 止损价${stop_price:.5f}（{'盈利' if pnl_pct > 0 else '亏损'}{abs(pnl_pct):.1f}%）请手动买回平仓！",
                })
            # 空头止盈（股价下跌超过 target%）
            elif target_price > 0 and current_price <= target_price:
                alerts.append({
                    "type": "TAKE_PROFIT_SHORT",
                    "ticker": ticker,
                    "action": "COVER",
                    "price": current_price,
                    "pnl_pct": round(pnl_pct, 2),
                    "shares": shares,
                    "message": f"🎯 做空止盈提醒：{ticker} 现价${current_price:.5f} ≤ 目标价${target_price:.5f}（盈利{pnl_pct:.1f}%）请手动买回平仓！",
                })

    # 发送告警
    if alerts:
        lines = [f"🚨 *持仓告警* — {datetime.now().strftime('%H:%M:%S')}"]
        for a in alerts:
            lines.append(a["message"])
        chat_id = os.getenv("TELEGRAM_HOME_CHANNEL") or os.getenv("TELEGRAM_CHAT_ID") or "6801255591"
        send_report("\n".join(lines), chat_id=chat_id)

    return alerts


# ======== 主程序：每日选股 + 信号推送 ========

def daily_trading_cycle(
    tickers: list[str],
    initial_cash: float,
    risk_config,
    dry_run: bool = True,
    use_dl: bool = True,
    model_type: str = "MLP",
) -> dict:
    """
    每日交易循环（无 broker 依赖）：
    1. yfinance 获取实时行情
    2. AI 评分排序
    3. DL 预测（可选）
    4. 生成信号
    5. Telegram 推送
    6. 返回报告
    """
    from scorer import rank_stocks
    from dl_strategy import batch_predict_sector
    from ai_analyzer import generate_report

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ===== 每日选股交易循环 =====")
    print(f"  dry_run={dry_run}, use_dl={use_dl}, 模型={model_type}")

    # 1. 获取行情
    print("  📥 获取实时行情（yfinance）...")
    quotes = get_live_quotes(tickers)
    if not quotes:
        print("  [!] 无法获取行情数据")
        return {"error": "No quote data"}

    # 2. 评分排序（股票 + 加密货币混合排名）
    print("  📊 AI 评分排序（全资产池）...")
    ranked = rank_stocks(tickers, top_n=20)  # 20=给加密资产足够空间

    # 3. DL 预测（板块路由版，可识别 crypto 板块模型）
    # 先在评分股票基础上混入加密货币（从 config 读取 CRYPTO_WATCHLIST）
    all_tickers_for_dl = list(ranked_stocks)  # 保留评分结果顺序
    crypto_tickers = [t for t in tickers if "/" in t]
    for t in crypto_tickers:
        if t not in [s["ticker"] for s in all_tickers_for_dl]:
            all_tickers_for_dl.append({"ticker": t})

    dl_predictions = []
    if use_dl:
        print("  🧠 DL 预测（板块路由）...")
        try:
            ticker_list = [s["ticker"] for s in all_tickers_for_dl[:15]]
            dl_predictions = batch_predict_sector(ticker_list)
        except Exception as e:
            print(f"  [!] DL 预测失败: {e}")

    # 4. 生成信号
    print("  🎯 生成交易信号...")
    account = get_account_info(initial_cash)
    cash = account.get("available_cash", initial_cash)

    signals = generate_signals(
        ranked_stocks=ranked,
        dl_predictions=dl_predictions,
        portfolio_cash=initial_cash,
        max_position_pct=risk_config.max_single_position_pct,
        risk_config=risk_config,
    )

    # 5. 推送信号
    print(f"  📋 推送 {len(signals)} 个信号到 Telegram...")
    exec_results = execute_signals(signals, dry_run=dry_run)

    # 6. AI 分析报告
    top_picks = ranked[:5]
    ai_report = generate_report(top_picks, tickers)

    # 保存结果
    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dry_run": dry_run,
        "ranked_stocks": ranked[:8],
        "dl_predictions": dl_predictions,
        "signals": signals,
        "exec_results": exec_results,
        "ai_report": ai_report,
        "account": account,
    }

    # 存档
    result_path = DATA_DIR / f"trading_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"  ✅ 完成！结果保存: {result_path}")
    return result


# ======== CLI 入口 ========

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(os.path.expanduser("~/.hermes/.env"))
    from config import WATCHLIST
    from risk_manager import RiskConfig

    risk_config = RiskConfig()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="实盘（目前仅改变推送标签）")
    parser.add_argument("--no-dl", action="store_true", help="禁用深度学习")
    args = parser.parse_args()

    dry_run = not args.live

    result = daily_trading_cycle(
        tickers=WATCHLIST,
        initial_cash=10000,
        risk_config=risk_config,
        dry_run=dry_run,
        use_dl=not args.no_dl,
    )

    if "error" not in result:
        print("\n📊 今日 Top 5:")
        for i, s in enumerate(result["ranked_stocks"][:5], 1):
            print(f"  {i}. {s['ticker']} (score={s.get('score',0)})")
        print(f"\n📋 信号:")
        for r in result["exec_results"]:
            icon = "🟢" if r["action"] == "BUY" else "🔴"
            print(f"  {icon} {r['action']} {r['ticker']} {r['qty']}股 @ ${r['price']:.2f} → {r['result']}")