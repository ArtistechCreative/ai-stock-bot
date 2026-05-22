"""
加密货币数据层 — broker-agnostic
支持 Binance / OKX / Bybit / Gate.io / Bitget / KuCoin 等交易所
通过 CCXT 统一接口获取行情、K线、订单簿、资金费率、持仓数据

多交易所架构：
- MultiExchangeRouter: 同时连接多个交易所，自动故障转移 + 请求分发
- RoundRobin 模式：轮询各交易所分散请求，防限速
- Fallback 模式：主交易所失败时自动切换备交易所
- 请求级锁：避免同一交易所并发请求触发限速

用法：
  # 单交易所（如之前）
  cd = CryptoData(exchange="binance")

  # 多交易所（新增）
  router = MultiExchangeRouter(["binance", "okx", "bybit"])
  quotes = router.fetch_quotes(["BTC/USDT", "ETH/USDT"])

  # 带 API key（私有接口）
  router = MultiExchangeRouter(["binance", "okx"], api_keys={...})
"""
import os
import time
import math
import random
import asyncio
import threading
from dataclasses import dataclass, field
from typing import Optional, Literal
from collections import defaultdict
import pandas as pd
import numpy as np

import ccxt
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 默认观察列表（主流币 + MEME + Defi）
DEFAULT_WATCHLIST = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "XRP/USDT",
    "BNB/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
    "MATIC/USDT", "SHIB/USDT", "LTC/USDT", "UNI/USDT", "APT/USDT",
    "ARB/USDT",  # Layer2
    "INJ/USDT",  # Cosmos Defi
    "SUI/USDT",  # 新公链
    "TIA/USDT",  # Celestia
]

# 合约基础信息（USDT本位永续合约）
PERP_INFO = {
    "BTC/USDT":  {"leverage": 75, "funding_rate": 0.0001, "maker_fee": 0.0002, "taker_fee": 0.0005},
    "ETH/USDT":  {"leverage": 75, "funding_rate": 0.0001, "maker_fee": 0.0002, "taker_fee": 0.0005},
    "SOL/USDT":  {"leverage": 50, "funding_rate": 0.0002, "maker_fee": 0.0002, "taker_fee": 0.0005},
    "DOGE/USDT": {"leverage": 50, "funding_rate": 0.0001, "maker_fee": 0.0002, "taker_fee": 0.0005},
    "XRP/USDT":  {"leverage": 50, "funding_rate": 0.0001, "maker_fee": 0.0002, "taker_fee": 0.0005},
    "BNB/USDT":  {"leverage": 25, "funding_rate": 0.0001, "maker_fee": 0.0002, "taker_fee": 0.0005},
    "AVAX/USDT": {"leverage": 50, "funding_rate": 0.0002, "maker_fee": 0.0002, "taker_fee": 0.0005},
    "LINK/USDT": {"leverage": 50, "funding_rate": 0.0001, "maker_fee": 0.0002, "taker_fee": 0.0005},
    "DOT/USDT":  {"leverage": 50, "funding_rate": 0.0002, "maker_fee": 0.0002, "taker_fee": 0.0005},
    "ADA/USDT":  {"leverage": 50, "funding_rate": 0.0001, "maker_fee": 0.0002, "taker_fee": 0.0005},
}

# 各交易所 API 地址（Binance 为例，其他可扩展）
EXCHANGE_CONFIG = {
    "binance":   {"id": "binance",   "name": "Binance",   "default_markets": ["USDT"]},
    "okx":       {"id": "okx",       "name": "OKX",       "default_markets": ["USDT"]},
    "bybit":     {"id": "bybit",     "name": "Bybit",     "default_markets": ["USDT"]},
    "gateio":    {"id": "gateio",    "name": "Gate.io",   "default_markets": ["USDT"]},
    "bitget":    {"id": "bitget",    "name": "Bitget",    "default_markets": ["USDT"]},
    "kucoin":    {"id": "kucoin",    "name": "KuCoin",    "default_markets": ["USDT"]},
}

# 默认交易所（在当前网络可访问的）
DEFAULT_EXCHANGE = "okx"


def get_exchanges() -> list[dict]:
    """返回支持的交易所列表"""
    return list(EXCHANGE_CONFIG.values())


# ======== 数据类 ========

@dataclass
class OHLCV:
    """单根 K 线"""
    timestamp: int       # Unix ms
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def datetime(self) -> str:
        import datetime as dt
        return dt.datetime.fromtimestamp(self.timestamp / 1000).strftime("%Y-%m-%d %H:%M")


@dataclass
class Quote:
    """实时行情快照"""
    symbol: str
    last_price: float
    bid: float
    ask: float
    volume_24h: float
    change_24h_pct: float
    high_24h: float
    low_24h: float
    funding_rate: float       # 资金费率（8小时）
    open_interest: float      # 未平仓合约
    timestamp: int


@dataclass
class Position:
    """仓位（支持多空）"""
    symbol: str
    side: str          # "LONG" or "SHORT"
    size: float        # 合约数量
    entry_price: float
    mark_price: float  # 当前标记价格
    liquidation_price: float
    leverage: int
    unrealized_pnl: float
    realized_pnl: float
    margin: float      # 占用保证金
    timestamp: int


# ======== 主数据类 ========

class CryptoData:
    """
    加密货币数据层 — broker-agnostic
    纯公共接口无需 API key；私有接口（下单/查询持仓）需要传入 key/secret
    """

    def __init__(
        self,
        exchange: str = "binance",
        api_key: str = None,
        api_secret: str = None,
        password: str = None,     # OKX 等需要 password
        testnet: bool = False,
    ):
        self.exchange_id = exchange
        self.exchange: ccxt.Exchange = self._init_exchange(exchange, api_key, api_secret, password, testnet)
        self.symbols: list[str] = []
        self._cache: dict = {}

    def _init_exchange(self, exchange, api_key, api_secret, password, testnet) -> ccxt.Exchange:
        """初始化 CCXT 交易所"""
        cls = getattr(ccxt, exchange)
        config = {
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},  # 永续合约
        }
        if testnet:
            config["testnet"] = True
            # Binance testnet
            if exchange == "binance":
                config["urls"] = {
                    "api": "https://testnet.binancefuture.com",
                    "v3": "https://testnet.binancefuture.com",
                }
        if api_key:
            config["apiKey"] = api_key
        if api_secret:
            config["secret"] = api_secret
        if password:
            config["password"] = password

        ex = cls(config)
        return ex

    # ---- 市场数据 ----

    def set_symbols(self, symbols: list[str]):
        """设置观察的币种列表"""
        self.symbols = symbols

    def fetch_quote(self, symbol: str, use_cache: bool = True) -> Quote | None:
        """
        获取单个币种实时行情（公共接口，不需要签名）
        """
        cache_key = f"quote:{symbol}"
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            if time.time() - cached.get("_ts", 0) < 5:  # 5秒缓存
                return cached.get("data")

        try:
            ticker = self.exchange.fetch_ticker(symbol)
            perp_info = PERP_INFO.get(symbol, {})
            change_24h = ticker.get("change", 0) or 0
            change_pct = ticker.get("changePercent", 0) or (change_24h / ticker["previousClose"] * 100) if ticker.get("previousClose") else 0

            quote = Quote(
                symbol=symbol,
                last_price=ticker["last"],
                bid=ticker.get("bid", 0) or 0,
                ask=ticker.get("ask", 0) or 0,
                volume_24h=ticker["baseVolume"] or 0,
                change_24h_pct=round(change_pct, 2),
                high_24h=ticker["high"] or 0,
                low_24h=ticker["low"] or 0,
                funding_rate=perp_info.get("funding_rate", 0),
                open_interest=0,  # CCXT 公共接口不返回 OI，需要单独查询
                timestamp=ticker["timestamp"],
            )
            self._cache[cache_key] = {"data": quote, "_ts": time.time()}
            return quote

        # ── Layer 1: Python 原生连接错误（防 WSL 被墙崩溃）─────────────────
        except (
            ConnectionRefusedError, ConnectionResetError,
            ConnectionError, TimeoutError, OSError, EOFError,
        ) as e:
            print(f"⚠️  [网络连接失败] {symbol} @ {self.exchange.id} | {type(e).__name__}: {e}")
            return None

        # ── Layer 2: requests / urllib3 网络错误 ─────────────────────────
        except requests.exceptions.RequestException as e:
            print(f"⚠️  [HTTP 网络错误] {symbol} @ {self.exchange.id} | {type(e).__name__}: {e}")
            return None

        # ── Layer 3: CCXT 专属错误类 ─────────────────────────────────────
        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.ExchangeError) as e:
            err_str = str(e).lower()
            is_conn_err = any(kw in err_str for kw in [
                'connection', 'timeout', 'network', 'resolve', 'getaddrinfo',
                'connection refused', 'unreachable', 'service unavailable'])
            tag = "⚠️  [CCXT 连接错误]" if is_conn_err else "⚠️  [CCXT API 错误]"
            print(f"{tag} {symbol} @ {self.exchange.id} | {e}")
            return None

        # ── Layer 4: 其他业务异常（如 429）────────────────────────────────
        except Exception as e:
            err_str = str(e).lower()
            if '429' in err_str or 'rate limit' in err_str:
                print(f"🛑 [429 限速] {symbol} @ {self.exchange.id}，跳过。")
            else:
                print(f"  [!] fetch_quote({symbol}) failed: {e}")
            return None

    def fetch_quotes(self, symbols: list[str] = None) -> dict[str, Quote]:
        """批量获取行情（默认不用缓存）"""
        syms = symbols or self.symbols
        results = {}
        for s in syms:
            q = self.fetch_quote(s, use_cache=False)
            if q:
                results[s] = q
        return results

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
        since: int = None,
    ) -> list[OHLCV]:
        """
        获取 K 线数据
        timeframe: "1m","5m","15m","1h","4h","1d","1w"
        """
        timeframe_map = {
            "1m": "1m", "5m": "5m", "15m": "15m",
            "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w",
        }
        tf = timeframe_map.get(timeframe, "1h")

        try:
            ohlcv_list = self.exchange.fetch_ohlcv(symbol, tf, since=since, limit=limit)
            return [
                OHLCV(
                    timestamp=c[0],
                    open=float(c[1]),
                    high=float(c[2]),
                    low=float(c[3]),
                    close=float(c[4]),
                    volume=float(c[5]),
                )
                for c in ohlcv_list
            ]

        # ── Layer 1: Python 原生连接错误 ─────────────────────────────────
        except (ConnectionRefusedError, ConnectionResetError, ConnectionError, TimeoutError, OSError, EOFError) as e:
            print(f"⚠️  [网络连接失败] fetch_ohlcv {symbol} @ {self.exchange.id} | {type(e).__name__}: {e}")
            return []
        # ── Layer 2: requests / urllib3 网络错误 ─────────────────────────
        except requests.exceptions.RequestException as e:
            print(f"⚠️  [HTTP 网络错误] fetch_ohlcv {symbol} @ {self.exchange.id} | {type(e).__name__}: {e}")
            return []
        # ── Layer 3: CCXT 专属错误类 ─────────────────────────────────────
        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.ExchangeError) as e:
            err_str = str(e).lower()
            is_conn_err = any(kw in err_str for kw in [
                'connection', 'timeout', 'network', 'resolve', 'getaddrinfo',
                'connection refused', 'unreachable', 'service unavailable'])
            tag = "⚠️  [CCXT 连接错误]" if is_conn_err else "⚠️  [CCXT API 错误]"
            print(f"{tag} fetch_ohlcv {symbol} @ {self.exchange.id} | {e}")
            return []
        # ── Layer 4: 其他异常（如 429）───────────────────────────────────
        except Exception as e:
            err_str = str(e).lower()
            if '429' in err_str or 'rate limit' in err_str:
                print(f"🛑 [429 限速] fetch_ohlcv {symbol} @ {self.exchange.id}，跳过。")
            else:
                print(f"  [!] fetch_ohlcv({symbol},{tf}) failed: {e}")
            return []

    def fetch_ohlcv_dataframe(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
        since: int = None,
    ) -> pd.DataFrame:
        """返回 DataFrame 格式的 K 线（方便计算指标）"""
        ohlcv = self.fetch_ohlcv(symbol, timeframe, limit, since)
        if not ohlcv:
            return pd.DataFrame()
        df = pd.DataFrame([{
            "timestamp": o.timestamp,
            "open": o.open,
            "high": o.high,
            "low": o.low,
            "close": o.close,
            "volume": o.volume,
        } for o in ohlcv])
        return df

    # ---- 订单簿 ----

    def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        """获取订单簿"""
        try:
            ob = self.exchange.fetch_order_book(symbol, limit=limit)
            return {
                "symbol": symbol,
                "bids": [[float(p), float(s)] for p, s in ob.get("bids", [])[:10]],
                "asks": [[float(p), float(s)] for p, s in ob.get("asks", [])[:10]],
                "timestamp": ob.get("timestamp", 0),
            }
        except Exception as e:
            print(f"  [!] fetch_order_book({symbol}) failed: {e}")
            return {"symbol": symbol, "bids": [], "asks": [], "timestamp": 0}

    # ---- 资金费率 ----

    def fetch_funding_rate(self, symbol: str) -> float:
        """获取当前资金费率"""
        try:
            funding = self.exchange.fetch_funding_rate(symbol)
            return funding.get("fundingRate", 0) or 0
        except Exception:
            return PERP_INFO.get(symbol, {}).get("funding_rate", 0)

    # ---- 持仓（需要签名） ----

    def fetch_positions(self, symbol: str = None) -> list[Position]:
        """
        查询持仓（需要 API key）
        symbol=None 时返回所有持仓
        """
        try:
            positions = self.exchange.fetch_positions(symbols=[symbol] if symbol else None)
            result = []
            for p in positions:
                if not p.get("info", {}).get("positionAmt"):
                    continue
                size = float(p.get("info", {}).get("positionAmt", 0))
                if size == 0:
                    continue
                result.append(Position(
                    symbol=p.get("symbol", symbol or ""),
                    side="LONG" if size > 0 else "SHORT",
                    size=abs(size),
                    entry_price=float(p.get("entryPrice", 0)),
                    mark_price=float(p.get("markPrice", 0)),
                    liquidation_price=float(p.get("liquidationPrice", 0) or 0),
                    leverage=int(p.get("leverage", 1)),
                    unrealized_pnl=float(p.get("unrealizedPnl", 0)),
                    realized_pnl=float(p.get("realizedPnl", 0)),
                    margin=float(p.get("isolatedMargin", 0) or p.get("maintMargin", 0) or 0),
                    timestamp=p.get("timestamp", 0),
                ))
            return result
        except Exception as e:
            print(f"  [!] fetch_positions({symbol}) failed: {e}")
            return []

    # ---- 账户余额（需要签名） ----

    def fetch_balance(self) -> dict:
        """查询账户余额"""
        try:
            bal = self.exchange.fetch_balance()
            free = bal.get("free", {})
            total = bal.get("total", {})
            used = bal.get("used", {})
            return {
                "free": {k: float(v) for k, v in free.items() if isinstance(v, (int, float)) and v > 0},
                "total": {k: float(v) for k, v in total.items() if isinstance(v, (int, float)) and v > 0},
                "used": {k: float(v) for k, v in used.items() if isinstance(v, (int, float)) and v > 0},
            }
        except Exception as e:
            print(f"  [!] fetch_balance failed: {e}")
            return {}

    # ---- 下单（需要签名） ----

    def place_order(
        self,
        symbol: str,
        side: str,        # "buy" or "sell"
        order_type: str,  # "market", "limit"
        qty: float,
        price: float = None,
        reduce_only: bool = False,
        stop_loss: float = None,   # 止损价格
        take_profit: float = None, # 止盈价格
    ) -> dict:
        """
        下单（需要 API key）
        side: "buy"(开多/平空) / "sell"(开空/平多)
        reduce_only: True = 平仓单（不新增仓位）
        返回: {success, order_id, message, filled_qty, avg_price}
        """
        try:
            params = {}
            if reduce_only:
                params["reduceOnly"] = True

            # 止损/止盈作为附加止盈损
            if stop_loss or take_profit:
                params["stopLossPrice"] = stop_loss
                params["takeProfitPrice"] = take_profit

            if order_type == "market":
                ret = self.exchange.create_market_order(symbol, side, qty, params=params)
            else:
                ret = self.exchange.create_limit_order(symbol, side, qty, price, params=params)

            return {
                "success": True,
                "order_id": str(ret.get("id", "")),
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": price or 0,
                "filled_qty": float(ret.get("filled", 0) or 0),
                "avg_price": float(ret.get("average", 0) or 0),
                "status": ret.get("status", "unknown"),
                "timestamp": ret.get("timestamp", 0),
            }
        except Exception as e:
            return {"success": False, "symbol": symbol, "message": str(e), "order_id": None}

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        """取消订单"""
        try:
            ret = self.exchange.cancel_order(order_id, symbol)
            return {"success": True, "order_id": order_id, "message": "已取消"}
        except Exception as e:
            return {"success": False, "order_id": order_id, "message": str(e)}

    # ---- 批量下单 ----

    def close_position(self, symbol: str, side: str = None) -> dict:
        """
        平掉指定币种的全部仓位
        side: 不填则自动判断（多头用 sell，平多头；空头用 buy，平空头）
        """
        positions = self.fetch_positions(symbol)
        if not positions:
            return {"success": True, "message": "无持仓"}
        pos = positions[0]
        actual_side = side or ("sell" if pos.side == "LONG" else "buy")
        return self.place_order(symbol, actual_side, "market", pos.size, reduce_only=True)

    # ---- 工具方法 ----

    def get_funding_rate_next(self, symbol: str) -> str:
        """预测下一次资金费率时间（UTC 0/8/16 点）"""
        now_h = time.gmtime().tm_hour
        next_slot = (8 - now_h % 8) % 8 or 8
        import datetime as dt
        next_time = dt.datetime.utcnow() + dt.timedelta(hours=next_slot)
        return next_time.strftime("%H:%M UTC")

    def get_liquidation_price(
        self,
        entry_price: float,
        side: str,       # "LONG" or "SHORT"
        leverage: int,
        maint_margin_ratio: float = 0.005,  # 维持保证金率（大部分币种 0.5%）
    ) -> float:
        """
        计算强平价格
        以 BTC 为例，维持保证金率 0.5%，75x 杠杆：
        多头强平价 = entry_price × (1 - 1/leverage × (1 - maint_ratio))
        空头强平价 = entry_price × (1 + 1/leverage × (1 - maint_ratio))
        """
        if leverage <= 0:
            return 0
        if side == "LONG":
            return entry_price * (1 - (1 / leverage) * (1 - maint_margin_ratio))
        else:
            return entry_price * (1 + (1 / leverage) * (1 - maint_margin_ratio))

    def normalize_symbol(self, symbol: str) -> str:
        """统一 symbol 格式为 BTC/USDT"""
        return symbol.replace("-", "/").replace("_", "/").upper()

    def get_taker_fee(self, symbol: str = None) -> float:
        """获取 taker 手续费率"""
        if symbol:
            return PERP_INFO.get(symbol, {}).get("taker_fee", 0.0005)
        return 0.0005

    def __repr__(self):
        return f"CryptoData(exchange={self.exchange_id}, symbols={len(self.symbols)})"


# ======== 多交易所路由器 ========
# 核心设计：请求级锁 + 轮询分发 + 自动故障转移

class RateLimiter:
    """单交易所请求频率限制器（线程安全）"""

    def __init__(self, requests_per_second: float = 1.0, burst: int = 2):
        self.per_second = requests_per_second
        self.burst = burst
        self._lock = threading.Lock()
        self._last_times: list[float] = []
        self._consecutive_errors = 0  # 连续失败计数

    def wait(self):
        """阻塞直到允许发送下一个请求"""
        with self._lock:
            now = time.time()
            # 清理过期记录（只保留1秒内的）
            self._last_times = [t for t in self._last_times if now - t < 1.0]

            if len(self._last_times) >= self.burst:
                # 达到突发上限，等待最旧请求过期
                sleep_time = 1.0 - (now - self._last_times[0]) + 0.05
                if sleep_time > 0:
                    time.sleep(sleep_time)
                now = time.time()
                self._last_times = [t for t in self._last_times if now - t < 1.0]

            self._last_times.append(time.time())
            self._consecutive_errors = 0  # 成功后重置

    def record_error(self, is_connection_error: bool = False):
        """记录一次失败，触发更严格的限速"""
        with self._lock:
            self._consecutive_errors += 1
            if is_connection_error:
                # 连接错误比 API 限速更严重，跳过时间拉长
                self._consecutive_errors += 2

    def is_rate_limited(self) -> bool:
        return self._consecutive_errors >= 3


class MultiExchangeRouter:
    """
    多交易所路由器 — 同时连接多个交易所，自动故障转移 + 请求分发

    两种工作模式：
    - round_robin：轮询分发请求到不同交易所（推荐，最均衡）
    - fallback：优先主交易所，失败才切换（适合有主次偏好的场景）

    防封策略：
    1. 每个交易所独立的 RateLimiter
    2. 请求自动分发到不同交易所（相同 symbol 也不会全压一个交易所）
    3. 某个交易所连续失败3次，临时跳过（直到恢复）
    4. 单交易所请求间隔 >1秒，避免触发 Binance 429
    """

    def __init__(
        self,
        exchanges: list[str] = None,
        api_keys: dict[str, dict] = None,
        mode: Literal["round_robin", "fallback"] = "round_robin",
        requests_per_second: float = 0.8,  # 每交易所每秒请求数（保守，防封）
    ):
        self.exchanges = exchanges or ["binance", "okx", "bybit"]
        self.api_keys = api_keys or {}
        self.mode = mode

        # 初始化每个交易所的连接和限速器
        self._connections: dict[str, CryptoData] = {}
        self._rate_limiters: dict[str, RateLimiter] = {}
        self._round_robin_index: dict[str, int] = defaultdict(int)  # 每个symbol的轮询位置
        self._lock = threading.Lock()

        for ex_id in self.exchanges:
            keys = self.api_keys.get(ex_id, {})
            self._connections[ex_id] = CryptoData(
                exchange=ex_id,
                api_key=keys.get("api_key"),
                api_secret=keys.get("api_secret"),
                password=keys.get("password"),
            )
            self._rate_limiters[ex_id] = RateLimiter(requests_per_second=requests_per_second)

    def _get_available_exchanges(self) -> list[str]:
        """返回当前可用的交易所列表（排除被限速的）"""
        return [
            ex_id for ex_id in self.exchanges
            if not self._rate_limiters[ex_id].is_rate_limited()
        ]

    def _get_exchange_for_symbol(self, symbol: str) -> str:
        """根据模式为 symbol 分配交易所"""
        available = self._get_available_exchanges()
        if not available:
            # 所有交易所都被限速，降级用第一个
            return self.exchanges[0]

        if self.mode == "round_robin":
            idx = self._round_robin_index[symbol] % len(available)
            chosen = available[idx]
            self._round_robin_index[symbol] += 1
            return chosen
        else:  # fallback
            return available[0]

    def _request(self, exchange_id: str, fn, *args, **kwargs):
        """在指定交易所执行请求，自动限速 + 错误记录"""
        limiter = self._rate_limiters[exchange_id]
        conn = self._connections[exchange_id]

        limiter.wait()
        try:
            result = fn(conn, *args, **kwargs)
            return result, None

        # ── Layer 1: Python 原生连接错误（最优先拦截，防止程序崩溃）─────────
        except (
            ConnectionRefusedError,
            ConnectionResetError,
            ConnectionAbortedError,
            ConnectionError,
            TimeoutError,
            OSError,           # 包含 "Connection refused" 在不同平台的变体
            EOFError,          # 某些 CCXT 握手失败会抛出
        ) as e:
            print(f"⚠️  [网络连接失败] 交易所={exchange_id} | 原因: WSL 网络受限/被墙 ({type(e).__name__}: {e})")
            limiter.record_error(is_connection_error=True)
            return None, f"[网络错误] {type(e).__name__}: {e}"

        # ── Layer 2: requests / urllib3 层级网络错误 ───────────────────────
        except requests.exceptions.RequestException as e:
            print(f"⚠️  [HTTP 网络错误] 交易所={exchange_id} | 原因: {type(e).__name__}: {e}")
            limiter.record_error(is_connection_error=True)
            return None, f"[HTTP 错误] {type(e).__name__}: {e}"

        # ── Layer 3: CCXT 专属网络错误类（显式类名匹配，不过度依赖字符串）──
        except (
            ccxt.NetworkError,
            ccxt.RequestTimeout,
            ccxt.ExchangeNotAvailable,
            ccxt.ExchangeError,    # 包含 "connection failed" 等
        ) as e:
            err_str = str(e).lower()
            is_conn_err = any(
                kw in err_str
                for kw in ['connection', 'timeout', 'network', 'resolve',
                           'getaddrinfo', 'connection refused', 'unreachable',
                           'service unavailable', '503', '502', '504']
            )
            tag = "⚠️  [CCXT 连接错误]" if is_conn_err else "⚠️  [CCXT API 错误]"
            print(f"{tag} 交易所={exchange_id} | 原因: {e}")
            limiter.record_error(is_connection_error=is_conn_err)
            return None, str(e)

        # ── Layer 4: 其他业务异常（429 等 HTTP 状态码，保持原有逻辑）───────
        except Exception as e:
            err_str = str(e).lower()
            # 429 限速
            if '429' in err_str or 'rate limit' in err_str:
                print(f"🛑 [429 限速] 交易所={exchange_id}，自动跳过。")
                limiter.record_error(is_connection_error=False)
                return None, f"[429 限速] {e}"
            # 兜底：记录文本匹配方式的连接错误（CCXT 某些版本不抛标准异常）
            is_conn_err = any(
                kw in err_str
                for kw in ['connection', 'timeout', 'network', 'resolve',
                           'getaddrinfo', 'connection refused']
            )
            limiter.record_error(is_connection_error=is_conn_err)
            return None, str(e)

    # ---- 公开 API ----

    def fetch_quote(self, symbol: str) -> tuple[Optional[Quote], Optional[str]]:
        """获取单个币种实时行情（自动选交易所）"""
        ex_id = self._get_exchange_for_symbol(symbol)
        return self._request(ex_id, lambda conn: conn.fetch_quote(symbol, use_cache=False))

    def fetch_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        """
        批量获取行情 — 自动将请求分散到多个交易所
        返回格式: {symbol: Quote}
        """
        results = {}
        errors = []

        for symbol in symbols:
            ex_id = self._get_exchange_for_symbol(symbol)
            quote, err = self._request(ex_id, lambda conn: conn.fetch_quote(symbol, use_cache=False))
            if quote:
                results[symbol] = quote
            else:
                errors.append(f"{symbol}: {err}")

        if errors:
            print(f"  [!] MultiExchange 部分失败 ({len(errors)}/{len(symbols)}): {errors[:3]}{'...' if len(errors) > 3 else ''}")

        return results

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
    ) -> pd.DataFrame:
        """获取 K 线数据（优先 OKX，其次 Gate.io/Bitget，避免 Binance）"""
        # WSL 实测：OKX 可用，Binance/Bybit 被墙
        # 按实际连通性排序尝试
        for ex_id in self._get_available_exchanges():
            limiter = self._rate_limiters[ex_id]
            if limiter.is_rate_limited():
                continue
            limiter.wait()
            try:
                conn = self._connections[ex_id]
                df = conn.fetch_ohlcv_dataframe(symbol, timeframe, limit)
                if not df.empty:
                    return df

            # ── Layer 1: Python 原生连接错误 ──────────────────────────────
            except (
                ConnectionRefusedError,
                ConnectionResetError,
                ConnectionAbortedError,
                ConnectionError,
                TimeoutError,
                OSError,
                EOFError,
            ) as e:
                print(f"⚠️  [网络连接失败] fetch_ohlcv {symbol} @ {ex_id} | {type(e).__name__}: {e}")
                limiter.record_error(is_connection_error=True)
                continue

            # ── Layer 2: requests / urllib3 网络错误 ───────────────────────
            except requests.exceptions.RequestException as e:
                print(f"⚠️  [HTTP 网络错误] fetch_ohlcv {symbol} @ {ex_id} | {type(e).__name__}: {e}")
                limiter.record_error(is_connection_error=True)
                continue

            # ── Layer 3: CCXT 专属网络错误类 ─────────────────────────────
            except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.ExchangeError) as e:
                err_str = str(e).lower()
                is_conn_err = any(kw in err_str for kw in [
                    'connection', 'timeout', 'network', 'resolve',
                    'getaddrinfo', 'connection refused', 'unreachable',
                    'service unavailable', '503', '502', '504'])
                tag = "⚠️  [CCXT 连接错误]" if is_conn_err else "⚠️  [CCXT API 错误]"
                print(f"{tag} fetch_ohlcv {symbol} @ {ex_id} | {e}")
                limiter.record_error(is_connection_error=is_conn_err)
                continue

            # ── Layer 4: 其他业务异常（429 等）────────────────────────────
            except Exception as e:
                err_str = str(e).lower()
                if '429' in err_str or 'rate limit' in err_str:
                    print(f"🛑 [429 限速] fetch_ohlcv {symbol} @ {ex_id}，跳过。")
                    limiter.record_error(is_connection_error=False)
                else:
                    is_conn_err = any(kw in err_str for kw in [
                        'connection', 'timeout', 'network', 'resolve', 'getaddrinfo'])
                    limiter.record_error(is_connection_error=is_conn_err)
                continue

    def fetch_ohlcv_multi(
        self,
        symbols: list[str],
        timeframe: str = "1h",
        limit: int = 500,
    ) -> dict[str, pd.DataFrame]:
        """批量获取多个币种 K 线（分散到不同交易所）"""
        results = {}
        for symbol in symbols:
            df = self.fetch_ohlcv(symbol, timeframe, limit)
            if not df.empty:
                results[symbol] = df
        return results

    def fetch_funding_rate(self, symbol: str) -> float:
        """获取资金费率"""
        ex_id = self._get_exchange_for_symbol(symbol)
        conn = self._connections[ex_id]
        limiter = self._rate_limiters[ex_id]
        limiter.wait()
        try:
            return conn.fetch_funding_rate(symbol)
        except Exception:
            return 0.0

    def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        """获取订单簿"""
        ex_id = self._get_exchange_for_symbol(symbol)
        conn = self._connections[ex_id]
        limiter = self._rate_limiters[ex_id]
        limiter.wait()
        try:
            return conn.fetch_order_book(symbol, limit)
        except Exception as e:
            return {"symbol": symbol, "bids": [], "asks": [], "timestamp": 0, "error": str(e)}

    def get_status(self) -> dict:
        """返回各交易所健康状态"""
        status = {}
        for ex_id in self.exchanges:
            limiter = self._rate_limiters[ex_id]
            status[ex_id] = {
                "rate_limited": limiter.is_rate_limited(),
                "consecutive_errors": limiter._consecutive_errors,
            }
        return status

    def __repr__(self):
        return f"MultiExchangeRouter(exchanges={self.exchanges}, mode={self.mode})"


# ======== 数据获取辅助函数 ========

def fetch_crypto_quotes(
    symbols: list[str] = None,
    exchange: str = "binance",
) -> dict[str, Quote]:
    """快速获取多个币种的实时行情"""
    syms = symbols or DEFAULT_WATCHLIST
    cd = CryptoData(exchange=exchange)
    return cd.fetch_quotes(syms)


def fetch_crypto_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 500,
    exchange: str = "binance",
) -> pd.DataFrame:
    """快速获取 K 线 DataFrame"""
    cd = CryptoData(exchange=exchange)
    return cd.fetch_ohlcv_dataframe(symbol, timeframe, limit)


# ======== CLI 测试 ========

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(os.path.expanduser("~/.hermes/.env"))

    print("📊 加密货币数据层测试\n")

    cd = CryptoData(exchange="binance")

    # 测试获取多个币行情
    syms = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    quotes = cd.fetch_quotes(syms)

    print(f"🟢 交易所: {cd.exchange_id.upper()}")
    print(f"📋 监控币种: {len(syms)}\n")
    print(f"{'币种':<15} {'价格':<12} {'24h涨跌':<10} {'成交量':<12} {'资金费率':<10}")
    print("-" * 60)
    for sym, q in quotes.items():
        change_emoji = "🟢" if q.change_24h_pct >= 0 else "🔴"
        print(f"{sym:<15} ${q.last_price:<11,.2f} {change_emoji}{q.change_24h_pct:>+.2f}%  {q.volume_24h:>12,.0f}  {q.funding_rate*100:>+.3f}%")

    print("\n📈 BTC/USDT K线（最近5根1h）：")
    btc_ohlcv = cd.fetch_ohlcv("BTC/USDT", "1h", limit=5)
    for c in btc_ohlcv:
        print(f"  {c.datetime} | O:{c.open:,.2f} H:{c.high:,.2f} L:{c.low:,.2f} C:{c.close:,.2f} V:{c.volume:,.0f}")

    print("\n" + "=" * 60)
    print("🔄 MultiExchangeRouter 测试（同时连接多个交易所）")
    print("=" * 60)

    # 多交易所同时拉取
    router = MultiExchangeRouter(["binance", "okx", "bybit"])
    quotes = router.fetch_quotes(["BTC/USDT", "ETH/USDT", "SOL/USDT"])

    print(f"\n📊 多交易所行情（round_robin 模式）")
    print(f"{'币种':<15} {'价格':<14} {'24h涨跌':<12}")
    print("-" * 43)
    for sym, q in quotes.items():
        change_emoji = "🟢" if q.change_24h_pct >= 0 else "🔴"
        print(f"{sym:<15} ${q.last_price:<13,.2f} {change_emoji}{q.change_24h_pct:>+.2f}%")

    # 各交易所健康状态
    status = router.get_status()
    print(f"\n🏥 交易所健康状态：")
    for ex, info in status.items():
        icon = "🔴" if info["rate_limited"] else "🟢"
        print(f"  {icon} {ex:<10} 连续错误: {info['consecutive_errors']}")