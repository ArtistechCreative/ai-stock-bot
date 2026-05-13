"""
数据层：用 yfinance 拉股价、财务数据、筛选
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json, os
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def fetch_quote(ticker: str) -> dict | None:
    """拉单个股票的基础信息"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # 基础数据
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        prev_close = info.get("previousClose")
        volume = info.get("volume")
        avg_volume = info.get("averageVolume")
        market_cap = info.get("marketCap", 0)
        pe = info.get("trailingPE")
        eps = info.get("trailingEps")
        beta = info.get("beta")

        # 5日价格变化
        hist = stock.history(period="5d")
        if len(hist) >= 2:
            price_5d_ago = hist["Close"].iloc[0]
            change_5d = (price - price_5d_ago) / price_5d_ago * 100
        else:
            change_5d = 0

        # 成交量比率
        volume_ratio = volume / avg_volume if avg_volume else 0

        return {
            "ticker": ticker,
            "price": price,
            "prev_close": prev_close,
            "change_5d_pct": round(change_5d, 2),
            "volume": volume,
            "avg_volume": avg_volume,
            "volume_ratio": round(volume_ratio, 2),
            "market_cap": market_cap,
            "pe": round(pe, 2) if pe else None,
            "eps": round(eps, 2) if eps else None,
            "beta": round(beta, 2) if beta else None,
            "name": info.get("shortName", ticker),
            "sector": info.get("sector", "Unknown"),
        }
    except Exception as e:
        print(f"  [!] {ticker}: {e}")
        return None


def fetch_financials(ticker: str) -> dict:
    """拉财务数据：营收、利润、负债"""
    try:
        stock = yf.Ticker(ticker)
        fin = stock.financials
        bal = stock.balance_sheet

        # 营收增长率
        if fin is not None and len(fin) >= 2:
            revenues = fin.loc["Total Revenue"] if "Total Revenue" in fin.index else None
            if revenues is not None and len(revenues) >= 2:
                revGrowth = (revenues.iloc[0] - revenues.iloc[1]) / revenues.iloc[1] * 100
            else:
                revGrowth = None
        else:
            revGrowth = None

        # 负债率
        if bal is not None and len(bal) >= 1:
            total_debt = bal.loc["Total Debt"].iloc[0] if "Total Debt" in bal.index else None
            total_assets = bal.loc["Total Assets"].iloc[0] if "Total Assets" in bal.index else None
            debt_ratio = total_debt / total_assets if (total_debt and total_assets) else None
        else:
            debt_ratio = None

        return {
            "revenue_growth_pct": round(revGrowth, 2) if revGrowth else None,
            "debt_ratio": round(debt_ratio, 3) if debt_ratio else None,
        }
    except Exception as e:
        return {"revenue_growth_pct": None, "debt_ratio": None}


def screen_stocks(tickers: list[str], config: dict) -> list[dict]:
    """筛选股票"""
    results = []

    for ticker in tickers:
        print(f"  分析 {ticker}...")
        quote = fetch_quote(ticker)
        if not quote:
            continue

        # 基础筛选
        if quote.get("pe") is None or quote["pe"] > config["pe_max"] or quote["pe"] <= config["pe_min"]:
            continue
        if quote.get("market_cap", 0) < config["market_cap_min"]:
            continue
        if quote.get("change_5d_pct", 0) < config["price_change_min"]:
            continue
        if quote.get("volume_ratio", 0) < config["volume_ratio_min"]:
            continue

        # 财务数据
        fin = fetch_financials(ticker)
        quote.update(fin)

        results.append(quote)

    return results


def save_screening_results(results: list[dict], filename: str = None):
    """保存筛选结果到 CSV"""
    if not filename:
        filename = f"screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    path = DATA_DIR / filename

    if results:
        df = pd.DataFrame(results)
        df.to_csv(path, index=False)
        print(f"  保存到 {path}")
    return path