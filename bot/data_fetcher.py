"""
数据层：yfinance + 多源 fallback（Alpha Vantage / Finnhub / Twelvedata）
优先级：yfinance → Alpha Vantage → Finnhub → Twelvedata
每次只尝试一个源，失败才跳下一个，避免单点限流
"""
import yfinance as yf
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── API Key 配置（从环境变量或 ~/.env 读取）────────────────────────
import os
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.hermes/.env"))

ALPHA_VANTAGE_KEY  = os.getenv("ALPHA_VANTAGE_API_KEY",  "")
FINNHUB_KEY        = os.getenv("FINNHUB_API_KEY",         "")
TWELVEDATA_KEY     = os.getenv("TWELVEDATA_API_KEY",      "")


# ════════════════════════════════════════════════════════════════════
# 通用工具
# ════════════════════════════════════════════════════════════════════

def _rate_limit(min_secs: float = 0.3):
    """两次请求之间最小间隔（避免触发限流）"""
    time.sleep(min_secs)


# ════════════════════════════════════════════════════════════════════
# 数据源 1 — yfinance（主源）
# ════════════════════════════════════════════════════════════════════

def _fetch_quote_yfinance(ticker: str) -> dict | None:
    """通过 yfinance 获取股票基础信息"""
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info

        price     = info.get("regularMarketPrice") or info.get("currentPrice")
        prev_close= info.get("previousClose")
        volume    = info.get("volume", 0)
        avg_vol   = info.get("averageVolume", 1)
        market_cap= info.get("marketCap", 0)
        pe        = info.get("trailingPE")
        eps       = info.get("trailingEps")
        beta      = info.get("beta", 1.0)
        name      = info.get("shortName", ticker)
        sector    = info.get("sector", "Unknown")

        # 5日价格变化
        hist = stock.history(period="5d")
        if len(hist) >= 2:
            price_5d_ago = hist["Close"].iloc[0]
            change_5d    = (price - price_5d_ago) / price_5d_ago * 100
        else:
            change_5d = 0

        volume_ratio = volume / avg_vol if avg_vol else 0

        return {
            "ticker"        : ticker,
            "price"         : price,
            "prev_close"    : prev_close,
            "change_5d_pct": round(change_5d, 2),
            "volume"        : volume,
            "avg_volume"    : avg_vol,
            "volume_ratio"  : round(volume_ratio, 2),
            "market_cap"    : market_cap,
            "pe"            : round(pe, 2) if pe else None,
            "eps"           : round(eps, 2) if eps else None,
            "beta"          : round(beta, 2) if beta else None,
            "name"          : name,
            "sector"        : sector,
            "_source"       : "yfinance",
        }

    # ── Layer 1: 网络层错误（yfinance 底层用 requests）─────────────────────
    except (
        ConnectionRefusedError,
        ConnectionResetError,
        ConnectionError,
        TimeoutError,
        OSError,
    ) as e:
        print(f"⚠️  [网络连接失败] {ticker} yfinance | {type(e).__name__}: {e}")
        return None

    # ── Layer 2: yfinance 自身异常（超时、无数据、HTTP 错误）──────────────
    except Exception as e:
        err_str = str(e).lower()
        # 网络类异常：记录后返回 None，触发 fallback
        if any(kw in err_str for kw in [
                'connection', 'timeout', 'network', 'resolve',
                'getaddrinfo', '429', 'rate limit', '503', '502',
                'connection refused', 'read timeout', 'read timed out']):
            print(f"⚠️  [网络/YF 错误] {ticker} yfinance 失败: {type(e).__name__}: {e}")
            return None
        # 业务类异常（如 ticker 不存在）：继续向上抛，避免静默失败
        print(f"  [!] {ticker} yfinance 业务错误（终止 fallback 链）: {e}")
        raise


# ════════════════════════════════════════════════════════════════════
# 数据源 2 — Alpha Vantage（yfinance 失败时 fallback）
# 免费额度：25 req/day，足够覆盖日常使用
# ════════════════════════════════════════════════════════════════════

def _fetch_quote_alpha_vantage(ticker: str) -> dict | None:
    """通过 Alpha Vantage Global Quote API 获取实时报价"""
    if not ALPHA_VANTAGE_KEY:
        print(f"  [!] {ticker} Alpha Vantage: 未配置 API_KEY（跳过）")
        return None

    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function"     : "GLOBAL_QUOTE",
            "symbol"       : ticker,
            "apikey"       : ALPHA_VANTAGE_KEY,
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        quote = data.get("Global Quote", {})
        if not quote or "05. price" not in quote:
            print(f"  [!] {ticker} Alpha Vantage: 无有效数据（{data})")
            return None

        price      = float(quote["05. price"])
        prev_close = float(quote["08. previous close"])
        volume     = int(quote["06. volume"])
        change_pct = float(quote["10. change percent"].replace("%", ""))

        return {
            "ticker"        : ticker,
            "price"         : price,
            "prev_close"    : prev_close,
            "change_5d_pct" : change_pct,          # Alpha Vantage 只给今日%，近似用 change_pct
            "volume"        : volume,
            "avg_volume"    : volume,               # 无平均成交量，用当日量代替
            "volume_ratio"  : 1.0,                 # 缺少 avg_volume，保守设为 1.0
            "market_cap"    : 0,
            "pe"            : None,
            "eps"           : None,
            "beta"          : 1.0,
            "name"          : ticker,
            "sector"        : "Unknown",
            "_source"       : "alpha_vantage",
        }
    except (
        ConnectionRefusedError, ConnectionResetError,
        ConnectionError, TimeoutError, OSError,
        requests.exceptions.RequestException,
    ) as e:
        print(f"⚠️  [网络错误] {ticker} Alpha Vantage | {type(e).__name__}: {e}")
        return None
    except Exception as e:
        err_str = str(e).lower()
        if any(kw in err_str for kw in ['429', 'rate limit', '503', '502', 'connection', 'timeout', 'network']):
            print(f"⚠️  [网络/限速] {ticker} Alpha Vantage 失败: {e}")
            return None
        print(f"  [!] {ticker} Alpha Vantage 业务错误: {e}")
        raise


# ════════════════════════════════════════════════════════════════════
# 数据源 3 — Finnhub（Alpha Vantage 失败时 fallback）
# 免费额度：60 req/sec，非常宽松
# ════════════════════════════════════════════════════════════════════

def _fetch_quote_finnhub(ticker: str) -> dict | None:
    """通过 Finnhub Quote API 获取实时报价"""
    if not FINNHUB_KEY:
        print(f"  [!] {ticker} Finnhub: 未配置 API_KEY（跳过）")
        return None

    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={FINNHUB_KEY}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        if not data or data.get("c") == 0:   # c = current price，0 表示无数据
            print(f"  [!] {ticker} Finnhub: 无有效数据（{data})")
            return None

        price      = data["c"]   # current
        prev_close = data["pc"]  # previous close
        high_52    = data["52WeekHigh"]
        low_52     = data["52WeekLow"]
        change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0

        return {
            "ticker"        : ticker,
            "price"         : price,
            "prev_close"    : prev_close,
            "change_5d_pct" : round(change_pct, 2),
            "volume"        : 0,
            "avg_volume"    : 0,
            "volume_ratio"  : 1.0,
            "market_cap"    : 0,
            "pe"            : None,
            "eps"           : None,
            "beta"          : 1.0,
            "name"          : ticker,
            "sector"        : "Unknown",
            "_source"       : "finnhub",
        }
    except (
        ConnectionRefusedError, ConnectionResetError,
        ConnectionError, TimeoutError, OSError,
        requests.exceptions.RequestException,
    ) as e:
        print(f"⚠️  [网络错误] {ticker} Finnhub | {type(e).__name__}: {e}")
        return None
    except Exception as e:
        err_str = str(e).lower()
        if any(kw in err_str for kw in ['429', 'rate limit', '503', '502', 'connection', 'timeout', 'network']):
            print(f"⚠️  [网络/限速] {ticker} Finnhub 失败: {e}")
            return None
        print(f"  [!] {ticker} Finnhub 业务错误: {e}")
        raise


# ════════════════════════════════════════════════════════════════════
# 数据源 4 — Twelvedata（Finnhub 失败时 fallback）
# 免费额度：800 req/day，覆盖充足
# ════════════════════════════════════════════════════════════════════

def _fetch_quote_twelvedata(ticker: str) -> dict | None:
    """通过 Twelvedata Real-Time Price API 获取实时报价"""
    if not TWELVEDATA_KEY:
        print(f"  [!] {ticker} Twelvedata: 未配置 API_KEY（跳过）")
        return None

    try:
        url = "https://api.twelvedata.com/price"
        params = {
            "symbol" : ticker,
            "apikey" : TWELVEDATA_KEY,
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        if "status" in data and data["status"] != "ok":
            print(f"  [!] {ticker} Twelvedata: {data}")
            return None

        price_str = data.get("price")
        if not price_str:
            print(f"  [!] {ticker} Twelvedata: 无 price 字段（{data})")
            return None

        price = float(price_str)

        # Twelvedata 的 quote 接口不提供 prev_close/change 等，
        # 需要额外调用 endpoint 获取完整数据
        url2 = "https://api.twelvedata.com/quote"
        params2 = {
            "symbol" : ticker,
            "apikey" : TWELVEDATA_KEY,
        }
        r2 = requests.get(url2, params=params2, timeout=10)
        r2.raise_for_status()
        data2 = r2.json()

        prev_close = float(data2.get("prev_close", price))
        volume     = int(data2.get("volume", 0))
        change_pct = float(data2.get("percent_change", 0))

        return {
            "ticker"        : ticker,
            "price"         : price,
            "prev_close"    : prev_close,
            "change_5d_pct" : round(change_pct, 2),
            "volume"        : volume,
            "avg_volume"    : volume,
            "volume_ratio"  : 1.0,
            "market_cap"    : 0,
            "pe"            : None,
            "eps"           : None,
            "beta"          : 1.0,
            "name"          : ticker,
            "sector"        : "Unknown",
            "_source"       : "twelvedata",
        }
    except (
        ConnectionRefusedError, ConnectionResetError,
        ConnectionError, TimeoutError, OSError,
        requests.exceptions.RequestException,
    ) as e:
        print(f"⚠️  [网络错误] {ticker} Twelvedata | {type(e).__name__}: {e}")
        return None
    except Exception as e:
        err_str = str(e).lower()
        if any(kw in err_str for kw in ['429', 'rate limit', '503', '502', 'connection', 'timeout', 'network']):
            print(f"⚠️  [网络/限速] {ticker} Twelvedata 失败: {e}")
            return None
        print(f"  [!] {ticker} Twelvedata 业务错误: {e}")
        raise


# ════════════════════════════════════════════════════════════════════
# 对外接口：fetch_quote — 链式 fallback
# ════════════════════════════════════════════════════════════════════

def fetch_quote(ticker: str) -> dict | None:
    """
    获取单支股票基础信息，链式 fallback：
    yfinance → Alpha Vantage → Finnhub → Twelvedata

    返回 dict 包含 _source 字段标记数据来源。
    所有字段都与原有格式兼容。
    """
    providers = [
        ("yfinance",     _fetch_quote_yfinance),
        ("Alpha Vantage", lambda t: _fetch_quote_alpha_vantage(t) if ALPHA_VANTAGE_KEY else None),
        ("Finnhub",       lambda t: _fetch_quote_finnhub(t)       if FINNHUB_KEY       else None),
        ("Twelvedata",    lambda t: _fetch_quote_twelvedata(t)   if TWELVEDATA_KEY    else None),
    ]

    for name, fn in providers:
        if fn is None:
            continue
        result = fn(ticker)
        if result is not None:
            return result
        # 当前源失败，rate limit 后尝试下一个
        _rate_limit(0.5)

    print(f"  [!] {ticker} 所有数据源均失败")
    return None


# ════════════════════════════════════════════════════════════════════
# 对外接口：批量抓取（带 fallback）
# ════════════════════════════════════════════════════════════════════

def fetch_quotes_batch(tickers: list[str]) -> dict[str, dict]:
    """
    批量抓取多支股票行情，失败自动 fallback。
    返回 {ticker: quote_dict}，不含 ticker 索引。
    """
    quotes = {}
    for ticker in tickers:
        q = fetch_quote(ticker)
        if q:
            quotes[ticker] = q
    return quotes


# ════════════════════════════════════════════════════════════════════
# 财务数据（仍通过 yfinance，财务数据 Yahoo 最完整，暂不加备用）
# ════════════════════════════════════════════════════════════════════

def fetch_financials(ticker: str) -> dict:
    """拉财务数据：营收增长率、负债率（仅 yfinance）"""
    try:
        stock = yf.Ticker(ticker)
        fin   = stock.financials
        bal   = stock.balance_sheet

        if fin is not None and len(fin) >= 2:
            revenues = fin.loc["Total Revenue"] if "Total Revenue" in fin.index else None
            if revenues is not None and len(revenues) >= 2:
                revGrowth = (revenues.iloc[0] - revenues.iloc[1]) / revenues.iloc[1] * 100
            else:
                revGrowth = None
        else:
            revGrowth = None

        if bal is not None and len(bal) >= 1:
            total_debt  = bal.loc["Total Debt"].iloc[0]  if "Total Debt"  in bal.index else None
            total_assets= bal.loc["Total Assets"].iloc[0]if "Total Assets" in bal.index else None
            debt_ratio  = total_debt / total_assets if (total_debt and total_assets) else None
        else:
            debt_ratio = None

        return {
            "revenue_growth_pct": round(revGrowth, 2) if revGrowth else None,
            "debt_ratio"        : round(debt_ratio, 3) if debt_ratio else None,
        }
    except Exception as e:
        return {"revenue_growth_pct": None, "debt_ratio": None}


# ════════════════════════════════════════════════════════════════════
# 股票筛选（使用新的 fetch_quote）
# ════════════════════════════════════════════════════════════════════

def screen_stocks(tickers: list[str], config: dict) -> list[dict]:
    """筛选股票"""
    results = []

    for ticker in tickers:
        print(f"  分析 {ticker}...")
        quote = fetch_quote(ticker)
        if not quote:
            continue

        if quote.get("pe") is None or quote["pe"] > config["pe_max"] or quote["pe"] <= config["pe_min"]:
            continue
        if quote.get("market_cap", 0) < config["market_cap_min"]:
            continue
        if quote.get("change_5d_pct", 0) < config["price_change_min"]:
            continue
        if quote.get("volume_ratio", 0) < config["volume_ratio_min"]:
            continue

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