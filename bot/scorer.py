"""
股票评分系统：给每支股票打分，排序输出 Top N
数据来自 data_fetcher（yfinance + Alpha Vantage + Finnhub + Twelvedata fallback）
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from bot.data_fetcher import fetch_quote

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def score_stock(ticker: str, detailed: bool = False, as_of_date: str = None) -> dict | None:
    """
    综合评分（75分制）：
    - PE 合理（20-40） +25
    - 净利润增长 +10
    - 5日涨幅正 +15
    - 成交量放大 +10
    - 低 beta（抗波动） +10
    - 市值 > 50B +10
    - 营收增长正 +10

    数据来源：data_fetcher（自动 fallback，兼容股票/外汇/期货）
    财务数据（营收增长）：直接用 yfinance，因为暂无备用

    as_of_date: 回测时指定日期（YYYY-MM-DD），计算技术指标时只能使用该日期及之前的数据。
                不指定则使用实时价格（即正常评分）。
    """
    try:
        quote = fetch_quote(ticker)
        if not quote:
            return None

        price     = quote.get("price")
        prev_close= quote.get("prev_close")
        volume    = quote.get("volume", 0)
        avg_vol   = quote.get("avg_volume", 1)
        market_cap= quote.get("market_cap", 0)
        pe        = quote.get("pe")
        beta      = quote.get("beta", 1.0)
        change_5d = quote.get("change_5d_pct", 0)
        vol_ratio = quote.get("volume_ratio", 1.0)
        name      = quote.get("name", ticker)
        sector    = quote.get("sector", "Unknown")

        # ── 营收增长率（仍走 yfinance，暂无备用）────────────────
        rev_growth = None
        try:
            stock = yf.Ticker(ticker)
            fin   = stock.financials
            if fin is not None and len(fin) >= 2:
                rev = fin.loc["Total Revenue"] if "Total Revenue" in fin.index else None
                if rev is not None and len(rev) >= 2 and rev.iloc[1] != 0:
                    rev_growth = (rev.iloc[0] - rev.iloc[1]) / rev.iloc[1] * 100
        except Exception:
            pass

        # ── 评分逻辑 ────────────────────────────────────────────
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

        # 5日涨幅
        if change_5d > 5:
            score += 15
            reasons.append(f"强势上涨({change_5d:.1f}%)")
        elif change_5d > 0:
            score += 8
            reasons.append(f"小幅上涨({change_5d:.1f}%)")
        elif change_5d > -3:
            score += 2

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

        # 市值
        if market_cap >= 100e9:
            score += 10
            reasons.append(f"大盘股(${market_cap/1e9:.0f}B)")
        elif market_cap >= 10e9:
            score += 5

        # 营收增长
        if rev_growth is not None:
            if rev_growth > 20:
                score += 10
                reasons.append(f"营收高增长({rev_growth:.0f}%)")
            elif rev_growth > 0:
                score += 5

        return {
            "ticker"        : ticker,
            "name"          : name,
            "price"         : round(price, 2) if price else None,
            "pe"            : round(pe, 1) if pe else None,
            "beta"          : round(beta, 2) if beta else None,
            "change_5d_pct" : round(change_5d, 2),
            "volume_ratio"  : round(vol_ratio, 2),
            "market_cap_B"  : round(market_cap / 1e9, 1) if market_cap else None,
            "score"         : score,
            "reasons"       : reasons,
            "sector"        : sector,
            "_source"       : quote.get("_source"),
        }
    except Exception as e:
        print(f"  [!] score_stock({ticker}) 失败: {e}")
        return None


def _score_stock_historical(ticker: str, as_of_date: str) -> dict | None:
    """
    回测专用：基于历史数据（截止 as_of_date）计算评分和技术指标。
    严格只用 as_of_date 当天及之前的数据，绝不获取未来价格。
    """
    try:
        end = (pd.Timestamp(as_of_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        start = (pd.Timestamp(as_of_date) - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty or len(df) < 20:
            return None

        # 扁平化 MultiIndex 列
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        close = df["Close"]
        high = df.get("High", close)
        low = df.get("Low", close)
        volume = df["Volume"]

        # 计算技术指标（只用截止 as_of_date 的历史数据）
        ma5   = close.rolling(5).mean()
        ma10  = close.rolling(10).mean()
        ma20  = close.rolling(20).mean()
        ma60  = close.rolling(60).mean()

        # 5日变化率
        change_5d_pct = close.pct_change(5).iloc[-1] * 100 if len(close) >= 5 else 0

        # 成交量比率（今日量 / 5日均量）
        vol_ma5 = volume.rolling(5).mean()
        vol_ratio = (volume.iloc[-1] / vol_ma5.iloc[-1]) if vol_ma5.iloc[-1] > 0 else 1.0

        # RSI (14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = (100 - (100 / (1 + rs))).iloc[-1]

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9).mean()
        macd_hist = (macd - macd_signal).iloc[-1]

        # 布林带 %B
        ma20_std = close.rolling(20).std()
        bb_upper = ma20 + 2 * ma20_std
        bb_lower = ma20 - 2 * ma20_std
        bb_pct_b = ((close - bb_lower) / (bb_upper - bb_lower + 1e-10)).iloc[-1]

        # ATR (14)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        atr_pct = (atr / close.iloc[-1] * 100) if close.iloc[-1] > 0 else 0

        # 最新价格
        price = float(close.iloc[-1])

        # PE / market_cap — 从 yfinance info 拿（info 是当前快照，backtest 时近似使用）
        # 注意：info 数据不是历史回溯的，是当前最新。对 backtest 来说这是已知局限。
        pe = None
        beta = 1.0
        market_cap = 0
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            pe = info.get("trailingPE") or info.get("forwardPE")
            beta = info.get("beta", 1.0)
            market_cap = info.get("marketCap", 0)
        except Exception:
            pass

        # 营收增长（从 financials，不回溯，用最近一期 vs 前一期）
        rev_growth = None
        try:
            fin = stock.financials
            if fin is not None and len(fin) >= 2:
                rev = fin.loc["Total Revenue"] if "Total Revenue" in fin.index else None
                if rev is not None and len(rev) >= 2 and rev.iloc[1] != 0:
                    rev_growth = (rev.iloc[0] - rev.iloc[1]) / rev.iloc[1] * 100
        except Exception:
            pass

        # ── 评分逻辑（与 score_stock 一致）─────────────────────────
        score = 0
        reasons = []

        # PE 评分
        if pe and 0 < pe <= 30:
            score += 25
            reasons.append(f"PE极低({pe:.1f})")
        elif pe and 30 < pe <= 50:
            score += 15
            reasons.append(f"PE合理({pe:.1f})")
        elif pe and 50 < pe <= 80:
            score += 5
            reasons.append(f"PE偏高({pe:.1f})")

        # 5日涨幅
        if change_5d_pct > 5:
            score += 15
            reasons.append(f"强势上涨({change_5d_pct:.1f}%)")
        elif change_5d_pct > 0:
            score += 8
            reasons.append(f"小幅上涨({change_5d_pct:.1f}%)")
        elif change_5d_pct > -3:
            score += 2

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
            reasons.append(f"低波动(β={beta:.2f})")
        elif beta and beta < 1.5:
            score += 5

        # 市值
        if market_cap >= 100e9:
            score += 10
            reasons.append(f"大盘股(${market_cap/1e9:.0f}B)")
        elif market_cap >= 10e9:
            score += 5

        # 营收增长
        if rev_growth is not None:
            if rev_growth > 20:
                score += 10
                reasons.append(f"营收高增长({rev_growth:.0f}%)")
            elif rev_growth > 0:
                score += 5

        return {
            "ticker": ticker,
            "name": ticker,
            "price": round(price, 2),
            "pe": round(pe, 1) if pe else None,
            "beta": round(beta, 2) if beta else None,
            "change_5d_pct": round(change_5d_pct, 2),
            "volume_ratio": round(vol_ratio, 2),
            "market_cap_B": round(market_cap / 1e9, 1) if market_cap else None,
            "score": score,
            "reasons": reasons,
            "sector": "Unknown",
            "_source": "historical",
            # 技术指标（供 generate_signals 加密货币规则使用）
            "rsi": round(rsi, 1) if not np.isnan(rsi) else None,
            "macd_hist": round(macd_hist, 4),
            "bb_pct": round(bb_pct_b, 4),
            "atr_pct": round(atr_pct, 4),
        }
    except Exception as e:
        print(f"  [!] _score_stock_historical({ticker}, {as_of_date}) 失败: {e}")
        return None


def _score_crypto_ticker(symbol: str, exchange: str = "okx", as_of_date: str = None) -> dict | None:
    """加密货币评分（兼容股票格式）— 回测模式使用 CCXT 历史 K 线计算技术指标"""
    try:
        import sys as _sys
        # 动态导入 crypto_data（避免循环依赖）
        _ref_dir = os.path.expanduser("~/.hermes/skills/productivity/ai-stock-trading-bot/references")
        if _ref_dir not in _sys.path:
            _sys.path.insert(0, _ref_dir)
        from bot.crypto_data import CryptoData

        cd = CryptoData(exchange=exchange)
        lookback_days = 200 if as_of_date is None else 200
        limit = lookback_days

        if as_of_date is not None:
            # 回测模式：取 as_of_date 之前的历史 K 线（避免未来价格泄漏）
            # CCXT fetch_ohlcv 不支持 end 参数，用 limit 近似
            since_ms = int((pd.Timestamp(as_of_date) - pd.Timedelta(days=lookback_days)).timestamp() * 1000)
            df = cd.fetch_ohlcv_dataframe(symbol, timeframe="1d", limit=limit, since=since_ms)
        else:
            df = cd.fetch_ohlcv_dataframe(symbol, timeframe="1d", limit=limit)

        if df.empty or len(df) < 30:
            return None

        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume"
        })
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        if as_of_date is not None:
            df = df[df.index <= pd.Timestamp(as_of_date).value]

        # 用 dl_strategy 的 compute_technical_indicators 计算技术指标（与 DL 模型特征一致）
        from bot.dl_strategy import compute_technical_indicators
        df_tech = compute_technical_indicators(df)

        close = df_tech["Close"]
        price = float(close.iloc[-1])
        rsi = float(df_tech["rsi"].iloc[-1]) if "rsi" in df_tech.columns and not np.isnan(df_tech["rsi"].iloc[-1]) else 50.0
        macd_hist = float(df_tech["macd_hist"].iloc[-1]) if "macd_hist" in df_tech.columns else 0.0
        bb_pct = float(df_tech["bb_pct_b"].iloc[-1]) if "bb_pct_b" in df_tech.columns else 0.5
        atr_pct = float(df_tech["atr_pct"].iloc[-1]) if "atr_pct" in df_tech.columns else 0.0
        mom5 = float(df_tech["mom5"].iloc[-1]) if "mom5" in df_tech.columns else 0.0
        vol_ratio = float(df_tech["vol_ratio"].iloc[-1]) if "vol_ratio" in df_tech.columns else 1.0

        # 基础趋势评分（与股票 75 分制对齐）
        score = 30.0
        reasons = []
        if rsi < 40:
            score += 20; reasons.append(f"RSI超卖({rsi:.1f})")
        elif rsi > 70:
            score -= 10; reasons.append(f"RSI超买({rsi:.1f})")
        if macd_hist > 0:
            score += 15; reasons.append("MACD金叉")
        else:
            score -= 5; reasons.append("MACD死叉")
        if mom5 > 0.05:
            score += 10; reasons.append(f"5日强势({mom5:.1%})")
        elif mom5 < -0.05:
            score -= 5; reasons.append(f"5日弱势({mom5:.1%})")
        if bb_pct < 0.2:
            score += 10; reasons.append("布林下轨超卖")
        elif bb_pct > 0.8:
            score -= 5; reasons.append("布林上轨超买")

        return {
            "ticker"        : symbol,
            "name"          : symbol,
            "price"         : price,
            "pe"            : None,
            "beta"          : None,
            "change_5d_pct" : round(mom5 * 100, 2),
            "volume_ratio"  : round(vol_ratio, 2),
            "market_cap_B"  : None,
            "score"         : min(75, max(0, score)),
            "reasons"       : reasons,
            "rsi"           : round(rsi, 1),
            "macd_hist"     : round(macd_hist, 4),
            "bb_pct"        : round(bb_pct, 3),
            "atr_pct"       : round(atr_pct, 2),
            "leverage"      : 50,
            "funding_rate"  : 0,
            "exchange"      : exchange,
            "_is_crypto"    : True,
        }
    except Exception as e:
        print(f"  [!] _score_crypto_ticker({symbol}) failed: {e}")
        return None


def rank_stocks(tickers: list[str], top_n: int = 5, as_of_date: str = None) -> list[dict]:
    """
    评分 + 排序（股票用 data_fetcher + fallback，加密货币用 CCXT/OKX）

    as_of_date: 回测时指定日期（YYYY-MM-DD）。指定时：
      - 股票改用 _score_stock_historical() 从历史数据计算评分（不泄露未来价格）
      - 加密货币仍走 CCXT/OKX（实时价格，对回测场景有数据污染，仅用于无历史数据的币种）
    """
    results = []
    crypto_tickers = [t for t in tickers if "/" in t]
    stock_tickers = [t for t in tickers if "/" not in t]

    # ── 回测模式：使用历史评分 ────────────────────────────────
    if as_of_date is not None:
        for t in stock_tickers:
            s = _score_stock_historical(t, as_of_date)
            if s:
                results.append(s)
        for t in crypto_tickers:
            # 回测模式下：CCXT 历史 K 线计算技术指标（与DL模型特征一致，避免未来价格泄漏）
            s = _score_crypto_ticker(t, exchange="okx", as_of_date=as_of_date)
            if s:
                results.append(s)

    # ── 实时模式：使用 live 数据 ──────────────────────────────
    else:
        for t in stock_tickers:
            s = score_stock(t)
            if s:
                results.append(s)

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
        src = f" [{s.get('_source')}]" if s.get("_source") else ""
        print(f"  {i}. {s['ticker']}{src} ({s['name']}) — 分数: {s['score']}/75")
        print(f"     价格: ${s['price']} | PE: {s['pe']} | β: {s['beta']}")
        print(f"     5日: {s['change_5d_pct']}% | 成交量: {s['volume_ratio']}x | 市值: {s['market_cap_B']}B")
        print(f"     亮点: {', '.join(s['reasons']) if s['reasons'] else '数据不足'}")
        print()