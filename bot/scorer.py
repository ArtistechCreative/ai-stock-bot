"""
股票评分系统：给每支股票打分，排序输出 Top N
不再用"全部满足"硬过滤，而是综合评分
"""
import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def score_stock(ticker: str, detailed: bool = False) -> dict | None:
    """
    综合评分：
    - PE 合理（20-40） +20
    - 净利润增长 +10
    - 5日涨幅正 +10
    - 成交量放大 +10
    - 低 beta（抗波动） +10
    - 市值 > 50B +5
    - 营收增长正 +10
    总分 75 分
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        price = info.get("regularMarketPrice") or info.get("currentPrice")
        prev_close = info.get("previousClose")
        volume = info.get("volume", 0)
        avg_volume = info.get("averageVolume", 1)
        market_cap = info.get("marketCap", 0)
        pe = info.get("trailingPE")
        eps = info.get("trailingEps")
        beta = info.get("beta", 1.0)

        # 5日价格变化
        hist = stock.history(period="10d")  # 多拿几天防数据少
        if len(hist) >= 3:
            prices = hist["Close"].values
            change_5d = (prices[-1] - prices[0]) / prices[0] * 100
            vol_ratio = volume / avg_volume if avg_volume else 0
        else:
            change_5d = 0
            vol_ratio = 0

        score = 0
        reasons = []

        # PE 评分
        if pe and 0 < pe <= 30:
            score += 25
            reasons.append(f"PE极低({pe})")
        elif pe and 30 < pe <= 50:
            score += 15
            reasons.append(f"PE合理({pe})")
        elif pe and 50 < pe <= 80:
            score += 5
            reasons.append(f"PE偏高({pe})")
        # pe<=0 或 >80 不加分

        # 5日涨幅
        if change_5d > 5:
            score += 15
            reasons.append(f"强势上涨({change_5d:.1f}%)")
        elif change_5d > 0:
            score += 8
            reasons.append(f"小幅上涨({change_5d:.1f}%)")
        elif change_5d > -3:
            score += 2  # 微跌不扣分

        # 成交量
        if vol_ratio >= 1.5:
            score += 10
            reasons.append(f"量能放大({vol_ratio:.2f}x)")
        elif vol_ratio >= 1.2:
            score += 7
            reasons.append(f"成交量正常({vol_ratio:.2f}x)")
        elif vol_ratio >= 1.0:
            score += 3

        # beta（波动性）
        if beta and beta < 1.0:
            score += 10
            reasons.append(f"低波动(β={beta})")
        elif beta and beta < 1.5:
            score += 5
        # beta > 2 不加分

        # 市值
        if market_cap >= 100e9:
            score += 10
            reasons.append(f"大盘股(\${market_cap/1e9:.0f}B)")
        elif market_cap >= 10e9:
            score += 5

        # 营收增长（通过 financials）
        try:
            fin = stock.financials
            if fin is not None and len(fin) >= 2:
                rev = fin.loc["Total Revenue"] if "Total Revenue" in fin.index else None
                if rev is not None and len(rev) >= 2 and rev.iloc[1] != 0:
                    rev_growth = (rev.iloc[0] - rev.iloc[1]) / rev.iloc[1] * 100
                    if rev_growth > 20:
                        score += 10
                        reasons.append(f"营收高增长({rev_growth:.0f}%)")
                    elif rev_growth > 0:
                        score += 5
        except:
            pass

        return {
            "ticker": ticker,
            "name": info.get("shortName", ticker),
            "price": round(price, 2) if price else None,
            "pe": round(pe, 1) if pe else None,
            "beta": round(beta, 2) if beta else None,
            "change_5d_pct": round(change_5d, 2),
            "volume_ratio": round(vol_ratio, 2),
            "market_cap_B": round(market_cap / 1e9, 1) if market_cap else None,
            "score": score,
            "reasons": reasons,
        }
    except Exception as e:
        return None


def _score_crypto_ticker(symbol: str, exchange: str = "okx") -> dict | None:
    """
    封装 crypto_scorer.score_crypto 为股票式返回格式（ticker/price/score 等字段）
    使其与 score_stock 输出格式兼容，方便 rank_stocks 统一排序
    """
    try:
        from crypto_scorer import score_crypto
        result = score_crypto(symbol, exchange=exchange, timeframe="1h", use_cache=False)
        if not result:
            return None
        # 映射为股票式字段
        return {
            "ticker": symbol,
            "name": result.get("name", symbol),
            "price": result.get("price"),
            "pe": None,
            "beta": None,
            "change_5d_pct": result.get("change_5d_pct", 0),
            "volume_ratio": result.get("volume_ratio", 1),
            "market_cap_B": None,
            "score": result.get("trend_score", 0),
            "reasons": result.get("reasons", []),
            # 加密货币额外字段（供 generate_signals 使用）
            "rsi": result.get("rsi"),
            "macd_hist": result.get("macd_hist"),
            "bb_pct": result.get("bb_pct"),
            "atr_pct": result.get("atr_pct"),
            "leverage": result.get("leverage", 50),
            "funding_rate": result.get("funding_rate", 0),
            "exchange": result.get("exchange", exchange),
            "_is_crypto": True,
        }
    except Exception as e:
        print(f"  [!] _score_crypto_ticker({symbol}) failed: {e}")
        return None


def rank_stocks(tickers: list[str], top_n: int = 5) -> list[dict]:
    """
    评分 + 排序（股票用 yfinance，加密货币用 CCXT/OKX）
    加密货币识别：ticker 含 '/' → 走 _score_crypto_ticker
    """
    results = []
    crypto_tickers = [t for t in tickers if "/" in t]
    stock_tickers = [t for t in tickers if "/" not in t]

    # 股票走 yfinance scorer
    for t in stock_tickers:
        s = score_stock(t)
        if s:
            results.append(s)

    # 加密货币走 CCXT/OKX scorer
    for t in crypto_tickers:
        s = _score_crypto_ticker(t, exchange="okx")
        if s:
            results.append(s)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


if __name__ == "__main__":
    from config import WATCHLIST

    print("📊 股票评分中...\n")
    ranked = rank_stocks(WATCHLIST)

    print(f"🏆 Top {len(ranked)} 股票：\n")
    for i, s in enumerate(ranked, 1):
        print(f"  {i}. {s['ticker']} ({s['name']}) — 分数: {s['score']}/75")
        print(f"     价格: ${s['price']} | PE: {s['pe']} | β: {s['beta']}")
        print(f"     5日: {s['change_5d_pct']}% | 成交量: {s['volume_ratio']}x | 市值: {s['market_cap_B']}B")
        print(f"     亮点: {', '.join(s['reasons']) if s['reasons'] else '数据不足'}")
        print()