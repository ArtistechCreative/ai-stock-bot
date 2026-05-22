"""
加密货币评分系统
不再用"全部满足"硬过滤，而是综合评分 + 做空优先

评分维度（满分 100 + 加分项 35）：
- 趋势动量（5日、1日）        +30
- RSI 超买超卖               +15
- MACD 方向与动能            +15
- 布林带位置                 +10
- 成交量异动                 +10
- 波动率（高波动利于做空）     +10
- 资金费率（做空成本参考）     +5
- 加分：AI做空信号 +25
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from crypto_data import CryptoData, Quote, DEFAULT_WATCHLIST, PERP_INFO, DEFAULT_EXCHANGE, MultiExchangeRouter

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# 预设可连通的交易所列表（WSL 实测 OKX 可用，Binance/Bybit 被墙）
# 用途：MultiExchangeRouter 默认交易所池（会动态跳过不可用的）
_PREFERRABLE_EXCHANGES = ["okx", "gateio", "bitget", "kucoin"]


def get_default_exchanges() -> list[str]:
    """自动检测并返回可连通的交易所列表（按速度排序）"""
    import ccxt
    exchanges_to_test = ["okx", "gateio", "bitget", "kucoin", "bybit", "binance"]
    available = []
    for ex_id in exchanges_to_test:
        try:
            cls = getattr(ccxt, ex_id)
            ex = cls({"enableRateLimit": True})
            # 只测 fetch_ticker（最轻量）
            ex.fetch_ticker("BTC/USDT", timeout=5)
            available.append(ex_id)
        except Exception:
            pass
        if len(available) >= 2:  # 找到 2 个就不继续测了
            break
    return available or ["okx"]  # 保底至少有一个


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """从 OHLCV DataFrame 计算技术指标"""
    df = df.copy()
    close = df["close"]

    # 均线
    df["ma5"] = close.rolling(5).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma60"] = close.rolling(60).mean()

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # 布林带
    ma20_std = close.rolling(20).std()
    df["bb_upper"] = df["ma20"] + 2 * ma20_std
    df["bb_lower"] = df["ma20"] - 2 * ma20_std
    df["bb_pct"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, 1e-10)

    # ATR（Average True Range — 波动率）
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - close.shift(1))
    low_close = abs(df["low"] - close.shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr"] / close * 100  # ATR 占价格的百分比（波动率）

    # 动量
    df["mom5"] = close / close.shift(5) - 1
    df["mom1"] = close / close.shift(1) - 1

    # 成交量比率
    df["vol_ma5"] = df["volume"].rolling(5).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma5"].replace(0, 1e-10)

    # 趋势方向
    df["above_ma20"] = (close > df["ma20"]).astype(int)
    df["above_ma5"] = (close > df["ma5"]).astype(int)

    return df


def score_crypto(
    symbol: str,
    exchange: str = DEFAULT_EXCHANGE,
    timeframe: str = "1h",
    use_cache: bool = True,
    router: MultiExchangeRouter = None,
) -> dict | None:
    """
    综合评分加密货币
    router: MultiExchangeRouter 实例，传入后自动多交易所拉取（推荐）
    返回: {
        symbol, price, score,
        change_5d_pct, mom1, rsi, macd_hist, bb_pct, atr_pct, vol_ratio, funding_rate,
        trend_score, momentum_score, signals (list of strings)
    }
    """
    if router is None:
        router = MultiExchangeRouter([exchange])

    quote, err = router.fetch_quote(symbol)
    if not quote:
        return None

    df = router.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)
    if df.empty or len(df) < 30:
        return None

    df = compute_indicators(df)
    latest = df.iloc[-1]
    prev5 = df.iloc[-6] if len(df) >= 6 else df.iloc[0]

    price = quote.last_price
    score = 0
    signals = []

    # ── 1. 趋势动量（5日）+15 ──────────────────────────────
    mom5 = latest["mom5"] * 100  # 转为 %
    if mom5 > 15:
        score += 15
        signals.append(f"强势上涨({mom5:.1f}%)")
    elif mom5 > 5:
        score += 10
        signals.append(f"上涨({mom5:.1f}%)")
    elif mom5 > 0:
        score += 5
        signals.append(f"小幅上涨({mom5:.1f}%)")
    elif mom5 > -5:
        score += 2  # 微跌不扣分（有利于做空）
    elif mom5 < -15:
        score += 8  # 大跌 → 做空机会大
        signals.append(f"大幅下跌({mom5:.1f}%) — 做空信号")

    # ── 2. 1日动量 +15 ────────────────────────────────────
    mom1 = latest["mom1"] * 100
    if mom1 > 3:
        score += 10
        signals.append(f"日内强势({mom1:.2f}%)")
    elif mom1 > 0:
        score += 5
    elif mom1 < -3:
        score += 8  # 下跌动能强，做空信号
        signals.append(f"日内走弱({mom1:.2f}%) — 做空信号")

    # ── 3. RSI 超买超卖 +15 ───────────────────────────────
    rsi = latest["rsi"]
    if rsi < 30:
        score += 15
        signals.append(f"RSI超卖({rsi:.1f})")
    elif rsi < 40:
        score += 10
        signals.append(f"RSI偏低({rsi:.1f})")
    elif rsi > 70:
        score += 12  # 超买 → 做空信号
        signals.append(f"RSI超买({rsi:.1f}) — 做空信号")
    elif rsi > 60:
        score += 6
    elif 40 <= rsi <= 60:
        score += 3

    # ── 4. MACD 方向与动能 +15 ───────────────────────────
    macd_hist = latest["macd_hist"]
    macd = latest["macd"]
    macd_sig = latest["macd_signal"]

    if macd_hist > 0 and macd > macd_sig:
        score += 15
        signals.append("MACD 金叉")
    elif macd_hist > 0:
        score += 8
    elif macd_hist < -0.01 * price:  # 跌破零轴明显
        score += 10
        signals.append("MACD 死叉 — 做空信号")
    elif macd_hist < 0:
        score += 4

    # ── 5. 布林带位置 +10 ─────────────────────────────────
    bb_pct = latest["bb_pct"]
    if bb_pct < 0.2:  # 触碰下轨
        score += 10
        signals.append("触碰布林下轨 — 超卖反弹做多信号")
    elif bb_pct < 0.4:
        score += 6
    elif bb_pct > 0.8:  # 触碰上轨
        score += 8
        signals.append("触碰布林上轨 — 超买回落做空信号")
    elif bb_pct > 0.6:
        score += 3
    else:
        score += 1  # 中性位置

    # ── 6. 成交量异动 +10 ─────────────────────────────────
    vol_ratio = latest["vol_ratio"]
    if vol_ratio >= 2.0:
        score += 10
        signals.append(f"成交量爆发({vol_ratio:.2f}x)")
    elif vol_ratio >= 1.5:
        score += 7
        signals.append(f"量能放大({vol_ratio:.2f}x)")
    elif vol_ratio >= 1.2:
        score += 4

    # ── 7. 波动率（高波动利于做空） +10 ──────────────────
    atr_pct = latest["atr_pct"]
    if atr_pct > 5:
        score += 10
        signals.append(f"高波动(ATR={atr_pct:.2f}%) — 做空机会")
    elif atr_pct > 3:
        score += 6
        signals.append(f"中等波动(ATR={atr_pct:.2f}%)")
    elif atr_pct > 1.5:
        score += 3
    else:
        score += 1  # 低波动

    # ── 8. 资金费率（做空成本参考） +5 ───────────────────
    perp_info = PERP_INFO.get(symbol, {})
    funding_rate = perp_info.get("funding_rate", quote.funding_rate)
    if funding_rate > 0.0005:  # > 0.05% 偏高
        score += 3
        signals.append(f"高资金费率({funding_rate*100:.3f}%) — 多头需谨慎")
    elif funding_rate < -0.0002:  # 负数，多头补贴
        score += 5
        signals.append(f"负资金费率({funding_rate*100:.3f}%) — 多头优势")
    elif funding_rate > 0:
        score += 2

    perp_info = PERP_INFO.get(symbol, {})

    return {
        "symbol": symbol,
        "name": symbol.replace("/USDT", ""),
        "price": round(price, 4) if price else None,
        "pe": None,  # 加密货币无 PE
        "beta": None,
        "change_5d_pct": round(mom5, 2),
        "change_1d_pct": round(mom1, 2),
        "volume_ratio": round(vol_ratio, 2),
        "rsi": round(rsi, 1),
        "macd_hist": round(macd_hist, 4),
        "bb_pct": round(bb_pct, 3),
        "atr_pct": round(atr_pct, 3),
        "trend_score": round(score, 1),
        "reasons": signals,
        # 额外字段（供 AI / 风控使用）
        "above_ma20": bool(latest["above_ma20"]),
        "above_ma5": bool(latest["above_ma5"]),
        "leverage": perp_info.get("leverage", 50),
        "funding_rate": round(funding_rate * 100, 4),  # 转为 %
        "taker_fee": perp_info.get("taker_fee", 0.0005),
        "timeframe": timeframe,
        "exchange": exchange,
    }


def rank_cryptos(
    symbols: list[str] = None,
    exchange: str = DEFAULT_EXCHANGE,
    timeframe: str = "1h",
    top_n: int = 8,
    multi_exchange: bool = True,
) -> list[dict]:
    """
    评分 + 排序
    multi_exchange=True（默认）：自动创建 MultiExchangeRouter，多交易所同时拉取
    multi_exchange=False：退化为单交易所（使用 exchange 参数指定的）
    """
    syms = symbols or DEFAULT_WATCHLIST

    if multi_exchange:
        exchanges = get_default_exchanges()
        router = MultiExchangeRouter(
            exchanges,
            mode="round_robin",
            requests_per_second=0.8,
        )
    else:
        router = MultiExchangeRouter([exchange])

    # 批量拉取所有币种行情（多交易所分散）
    all_quotes = router.fetch_quotes(syms)
    if not all_quotes:
        return []

    results = []
    for sym in syms:
        s = score_crypto(sym, exchange=exchange, timeframe=timeframe, use_cache=False, router=router)
        if s:
            results.append(s)

    results.sort(key=lambda x: x["trend_score"], reverse=True)
    return results[:top_n]


# ======== CLI ========

if __name__ == "__main__":
    print("📊 加密货币综合评分\n")
    ranked = rank_cryptos(exchange=DEFAULT_EXCHANGE, top_n=8)
    print(f"🏆 Top {len(ranked)} 加密货币：\n")
    for i, s in enumerate(ranked, 1):
        dir_icon = "🔴" if s["rsi"] > 60 or s["change_5d_pct"] < 0 else "🟢"
        print(f"  {i}. {s['symbol']} ({s['name']}) — 分数: {s['trend_score']}/100")
        print(f"     价格: ${s['price']} | RSI: {s['rsi']} | MACD_hist: {s['macd_hist']}")
        print(f"     5日: {s['change_5d_pct']}% | 成交量: {s['volume_ratio']}x | ATR: {s['atr_pct']}%")
        print(f"     布林: {s['bb_pct']} | 资金费率: {s['funding_rate']}% | 杠杆: {s['leverage']}x")
        print(f"     信号: {', '.join(s['reasons']) if s['reasons'] else '中性观望'}")
        print()