"""
Moomoo/Futu OpenAPI 连接管理
- 行情上下文 (OpenQuoteContext) — 实时行情
- 交易上下文 (OpenSecTradeContext) — 下单/持仓/账户
"""
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from futu import OpenQuoteContext, OpenSecTradeContext, TrdMarket, TrdEnv

DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class MoomooConfig:
    """Moomoo 连接配置"""
    # 连接地址（默认本地 OpenD）
    quote_host: str = "127.0.0.1"
    quote_port: int = 11111

    trade_host: str = "127.0.0.1"
    trade_port: int = 11111

    # 交易环境：SIMULATE = 模拟 / REAL = 实盘
    trd_env: str = "SIMULATE"  # "SIMULATE" or "REAL"

    # 市场
    trd_market: str = "US"     # "US" / "HK" / "SG"

    # 是否启用实盘交易（设为 True 时 trd_env 必须为 REAL）
    enable_live_trading: bool = False


class MoomooConnection:
    """
    Moomoo 连接管理器
    - 提供行情 + 交易上下文
    - 自动重连
    - 上下文生命周期管理
    """

    def __init__(self, config: MoomooConfig = None):
        self.config = config or MoomooConfig()
        self.quote_ctx: Optional[OpenQuoteContext] = None
        self.trade_ctx: Optional[OpenSecTradeContext] = None
        self._connected = False

    def connect_quote(self) -> bool:
        """连接行情"""
        try:
            self.quote_ctx = OpenQuoteContext(
                host=self.config.quote_host,
                port=self.config.quote_port,
            )
            ret, data = self.quote_ctx.start()
            if ret != 0:
                print(f"  [!] 行情连接失败: {data}")
                return False
            print(f"  ✅ 行情连接成功 ({self.config.quote_host}:{self.config.quote_port})")
            return True
        except Exception as e:
            print(f"  [!] 行情连接异常: {e}")
            return False

    def connect_trade(self) -> bool:
        """连接交易（模拟/实盘）"""
        try:
            trd_env = TrdEnv.SIMULATE if self.config.trd_env == "SIMULATE" else TrdEnv.REAL
            trd_market = TrdMarket.US if self.config.trd_market == "US" else (
                TrdMarket.HK if self.config.trd_market == "HK" else TrdMarket.SG
            )

            self.trade_ctx = OpenSecTradeContext(
                host=self.config.trade_host,
                port=self.config.trade_port,
                trd_env=trd_env,
                trd_market=trd_market,
            )
            ret, data = self.trade_ctx.start()
            if ret != 0:
                print(f"  [!] 交易连接失败: {data}")
                return False
            print(f"  ✅ 交易连接成功 (市场={self.config.trd_market}, 环境={self.config.trd_env})")
            return True
        except Exception as e:
            print(f"  [!] 交易连接异常: {e}")
            return False

    def connect_all(self) -> bool:
        """同时连接行情 + 交易"""
        q_ok = self.connect_quote()
        t_ok = self.connect_trade() if self.config.enable_live_trading or self.config.trd_env == "SIMULATE" else False
        self._connected = q_ok
        return q_ok

    def close(self):
        """关闭所有连接"""
        if self.quote_ctx:
            self.quote_ctx.close()
        if self.trade_ctx:
            self.trade_ctx.close()
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    # ---- 行情接口 ----

    def get_quote(self, ticker: str) -> dict:
        """获取单支股票实时报价"""
        if not self.quote_ctx:
            return {}
        ret, data = self.quote_ctx.get_stock_quote([ticker])
        if ret != 0 or data.empty:
            return {}
        row = data.iloc[0]
        return {
            "ticker": ticker,
            "last_price": row.get("last_price", 0),
            "open_price": row.get("open_price", 0),
            "high_price": row.get("high_price", 0),
            "low_price": row.get("low_price", 0),
            "volume": row.get("volume", 0),
            "bid_price": row.get("bid_price", 0),
            "ask_price": row.get("ask_price", 0),
            "timestamp": row.get("timestamp", 0),
        }

    def get_quotes(self, tickers: list[str]) -> dict:
        """批量获取报价"""
        result = {}
        for t in tickers:
            try:
                q = self.get_quote(t)
                if q:
                    result[t] = q
            except:
                pass
        return result

    def get_kline(
        self,
        ticker: str,
        ktype: str = "K_DAY",
        count: int = 100,
        start_date: str = None,
        end_date: str = None,
    ) -> list[dict]:
        """获取 K 线数据"""
        if not self.quote_ctx:
            return []

        from futu import KLType
        kt = {
            "K_DAY": KLType.K_DAY,
            "K_1MIN": KLType.K_1M,
            "K_5MIN": KLType.K_5M,
            "K_1H": KLType.K_1H,
            "K_WEEK": KLType.K_WEEK,
        }.get(ktype, KLType.K_DAY)

        ret, data = self.quote_ctx.request_history_kline(
            ticker,
            start=start_date,
            end=end_date,
            kl_type=kt,
            count=count,
        )

        if ret != 0 or data is None:
            return []

        records = []
        for _, row in data.iterrows():
            records.append({
                "time": row.get("time", ""),
                "open": row.get("open", 0),
                "high": row.get("high", 0),
                "low": row.get("low", 0),
                "close": row.get("close", 0),
                "volume": row.get("volume", 0),
            })
        return records

    # ---- 交易接口 ----

    def get_positions(self) -> list[dict]:
        """获取当前持仓"""
        if not self.trade_ctx:
            return []
        ret, data = self.trade_ctx.position_list_query()
        if ret != 0:
            return []
        positions = []
        for _, row in data.iterrows():
            positions.append({
                "ticker": row.get("stock_code", ""),
                "shares": row.get("qty", 0),
                "avg_cost": row.get("cost_price", 0),
                "market_value": row.get("market_val", 0),
                "pnl": row.get("pl_val", 0),
                "pnl_pct": row.get("pl_ratio", 0),
            })
        return positions

    def get_account_info(self) -> dict:
        """获取账户资金信息"""
        if not self.trade_ctx:
            return {}
        ret, data = self.trade_ctx.account_info_query()
        if ret != 0 or data.empty:
            return {}
        row = data.iloc[0]
        return {
            "cash": row.get("cash", 0),
            "available_cash": row.get("available_cash", 0),
            "total_assets": row.get("total_assets", 0),
            "market_value": row.get("market_val", 0),
            "currency": row.get("currency", "USD"),
        }

    def place_order(
        self,
        ticker: str,
        qty: int,
        side: str,  # "BUY" or "SELL"
        order_type: str = "NORMAL",  # NORMAL / MARKET / STOP
        price: float = 0,
    ) -> dict:
        """
        下单
        返回: {success: bool, order_id: str, message: str}
        """
        if not self.trade_ctx:
            return {"success": False, "message": "交易上下文未连接"}

        from futu import TrdSide, OrderType, PriceType

        trd_side = TrdSide.BUY if side == "BUY" else TrdSide.SELL

        # 如果没指定价格，用市价单
        if price <= 0:
            # 市价单
            ret, data = self.trade_ctx.place_order(
                stock_code=ticker,
                qty=qty,
                trd_side=trd_side,
                order_type=OrderType.MARKET,
            )
        else:
            # 限价单
            ret, data = self.trade_ctx.place_order(
                stock_code=ticker,
                qty=qty,
                trd_side=trd_side,
                order_type=OrderType.NORMAL,
                price=price,
            )

        if ret != 0:
            return {"success": False, "message": str(data), "order_id": None}

        order_id = data.iloc[0].get("order_id", "") if not data.empty else ""
        return {"success": True, "order_id": order_id, "message": "下单成功"}

    def cancel_order(self, order_id: str) -> dict:
        """取消订单"""
        if not self.trade_ctx:
            return {"success": False, "message": "交易上下文未连接"}
        ret, data = self.trade_ctx.cancel_order(order_id)
        if ret != 0:
            return {"success": False, "message": str(data)}
        return {"success": True, "message": "取消成功"}

    def get_orders(self, status: str = "UNFINISHED") -> list[dict]:
        """获取订单列表"""
        if not self.trade_ctx:
            return []
        from futu import OrderStatus
        status_map = {
            "UNFINISHED": OrderStatus.UNFINISHED,
            "FINISHED": OrderStatus.FILLED,
            "CANCELLED": OrderStatus.CANCEL,
            "ALL": OrderStatus.ALL,
        }
        ret, data = self.trade_ctx.order_list_query(status_map.get(status, OrderStatus.ALL))
        if ret != 0:
            return []
        orders = []
        for _, row in data.iterrows():
            orders.append({
                "order_id": row.get("order_id", ""),
                "ticker": row.get("stock_code", ""),
                "qty": row.get("qty", 0),
                "traded_qty": row.get("traded_qty", 0),
                "price": row.get("price", 0),
                "side": row.get("trd_side", ""),
                "status": row.get("status", ""),
                "create_time": row.get("create_time", ""),
            })
        return orders


# ======== 单例连接 ========
_connection: Optional[MoomooConnection] = None


def get_connection(config: MoomooConfig = None) -> MoomooConnection:
    global _connection
    if _connection is None:
        _connection = MoomooConnection(config)
    return _connection


def close_connection():
    global _connection
    if _connection:
        _connection.close()
        _connection = None


# ======== CLI 测试 ========
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(os.path.expanduser("~/.hermes/.env"))

    print("🔌 测试 Moomoo 连接...")

    # 先检查 OpenD 是否在跑
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(("127.0.0.1", 11111))
    sock.close()

    if result != 0:
        print("❌ OpenD 未运行！请先在电脑打开 Moomoo/OpenD")
        print("   下载: https://www.moomoo.com/openapi")
        print("   启动后保持运行，再重新运行此脚本")
    else:
        conn = MoomooConnection(MoomooConfig(trd_env="SIMULATE"))
        if conn.connect_all():
            print(conn.get_quote("NVDA"))
            conn.close()