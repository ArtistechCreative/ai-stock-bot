"""
股票行情监控服务（后台常驻）
- 每 N 分钟检查股票行情
- 发现异动（涨幅/跌幅超阈值）→ Telegram 推送
- 持仓触发止损/止盈 → 自动推送警报
- 记录所有事件到日志
"""
import time
import threading
import requests
import yfinance as yf
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import json
import os

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


@dataclass
class Alert:
    timestamp: str
    level: str      # INFO / WARNING / CRITICAL
    ticker: str
    message: str
    price: float
    change_pct: float


class StockMonitor:
    """后台股票行情监控"""

    def __init__(
        self,
        tickers: list[str],
        check_interval_minutes: int = 5,
        alert_threshold_pct: float = 5.0,    # 涨跌超 5% 警报
        portfolio_path: str = None,
        telegram_token: str = None,
        telegram_chat_id: str = None,
    ):
        self.tickers = tickers
        self.interval = check_interval_minutes * 60
        self.alert_threshold = alert_threshold_pct
        self.portfolio_path = portfolio_path or str(DATA_DIR / "portfolio.json")
        self.telegram_token = telegram_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = telegram_chat_id or os.getenv("TELEGRAM_HOME_CHANNEL") or "6801255591"

        self.running = False
        self._thread: Optional[threading.Thread] = None
        self.alerts: list[Alert] = []
        self.alert_history_path = DATA_DIR / "alert_history.json"

        # 上一次的价格（计算变化）
        self._prev_prices: dict[str, float] = {}
        self._prev_prices = self._load_prev_prices()

    def _load_prev_prices(self) -> dict:
        p = DATA_DIR / "prev_prices.json"
        if p.exists():
            return json.load(open(p))
        return {}

    def _save_prev_prices(self):
        with open(DATA_DIR / "prev_prices.json", "w") as f:
            json.dump(self._prev_prices, f, indent=2)

    # ---- 数据获取 ----

    def fetch_quotes(self) -> dict:
        """获取所有股票当前行情"""
        quotes = {}
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                hist = stock.history(period="5d")

                price = info.get("regularMarketPrice") or info.get("currentPrice")
                prev_close = info.get("previousClose") or hist["Close"].iloc[-2] if len(hist) >= 2 else price
                volume = info.get("volume", 0)
                avg_vol = info.get("averageVolume", 1)

                # 计算各种指标
                change_pct = (price - prev_close) / prev_close * 100 if prev_close else 0

                quotes[ticker] = {
                    "price": price,
                    "prev_close": prev_close,
                    "change_pct": round(change_pct, 3),
                    "volume": volume,
                    "avg_volume": avg_vol,
                    "volume_ratio": round(volume / avg_vol, 2) if avg_vol else 0,
                    "market_cap": info.get("marketCap", 0),
                    "pe": info.get("trailingPE"),
                    "beta": info.get("beta"),
                }
            except Exception as e:
                print(f"  [!] {ticker}: {e}")

        return quotes

    def check_portfolio_alerts(self, quotes: dict):
        """检查持仓是否触发止损/止盈"""
        if not Path(self.portfolio_path).exists():
            return []

        alerts = []
        portfolio = json.load(open(self.portfolio_path))
        positions = portfolio.get("positions", {})

        for ticker, pos in positions.items():
            if ticker not in quotes:
                continue

            cp = quotes[ticker]["price"]
            entry = pos.get("avg_cost", 0)
            stop = pos.get("stop_loss", 0)
            target = pos.get("target", 0)

            if not entry or entry <= 0:
                continue

            pnl_pct = (cp - entry) / entry * 100

            if cp <= stop and stop > 0:
                alerts.append(Alert(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                    level="CRITICAL",
                    ticker=ticker,
                    message=f"🚨 触发止损！现价${cp:.2f}，亏损{pnl_pct:.1f}%",
                    price=cp,
                    change_pct=pnl_pct,
                ))
            elif cp >= target and target > 0:
                alerts.append(Alert(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                    level="WARNING",
                    ticker=ticker,
                    message=f"🎯 达到止盈目标！现价${cp:.2f}，盈利{pnl_pct:.1f}%",
                    price=cp,
                    change_pct=pnl_pct,
                ))

        return alerts

    def check_market_alerts(self, quotes: dict) -> list[Alert]:
        """检查市场异动（大幅涨跌）"""
        alerts = []
        for ticker, q in quotes.items():
            change = q["change_pct"]
            price = q["price"]

            if abs(change) >= self.alert_threshold:
                prev = self._prev_prices.get(ticker, price)
                direction = "📈暴涨" if change > 0 else "📉暴跌"
                alerts.append(Alert(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                    level="INFO",
                    ticker=ticker,
                    message=f"{direction} {ticker} {'+' if change >= 0 else ''}{change:.2f}% → ${price:.2f}",
                    price=price,
                    change_pct=change,
                ))

        return alerts

    def send_telegram_alert(self, text: str):
        """发送 Telegram 警报"""
        if not self.telegram_token:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
                json={"chat_id": self.telegram_chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
        except Exception as e:
            print(f"  [!] Telegram 发送失败: {e}")

    def save_alert(self, alert: Alert):
        """保存警报到历史"""
        self.alerts.append(alert)
        # 保存到文件
        history = []
        if self.alert_history_path.exists():
            history = json.load(open(self.alert_history_path))
        history.append(asdict(alert))
        with open(self.alert_history_path, "w") as f:
            json.dump(history[-100:], f, indent=2)  # 保留最近100条

    def one_poll(self) -> list[Alert]:
        """执行一次轮询，返回所有警报"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 轮询 {len(self.tickers)} 支股票...")
        quotes = self.fetch_quotes()

        all_alerts = []

        # 市场异动警报
        market_alerts = self.check_market_alerts(quotes)
        all_alerts.extend(market_alerts)

        # 持仓止损/止盈检查
        portfolio_alerts = self.check_portfolio_alerts(quotes)
        all_alerts.extend(portfolio_alerts)

        # 发送 Telegram
        for a in all_alerts:
            print(f"  ⚠️ [{a.level}] {a.ticker}: {a.message}")
            self.send_telegram_alert(f"📢 [{a.level}] {a.ticker}\n{a.message}")
            self.save_alert(a)

        # 更新价格记录
        self._prev_prices = {t: q["price"] for t, q in quotes.items()}
        self._save_prev_prices()

        return all_alerts

    def _poll_loop(self):
        """轮询循环（后台线程）"""
        while self.running:
            try:
                self.one_poll()
            except Exception as e:
                print(f"  [!] 轮询异常: {e}")

            # 等待下次轮询
            for _ in range(self.interval):
                if not self.running:
                    break
                time.sleep(1)

    def start(self, poll_once: bool = False):
        """启动监控（后台线程）"""
        if self.running:
            print("⚠️ 监控已在运行")
            return

        print(f"🚀 启动股票监控: 每 {self.interval//60} 分钟轮询一次")
        self.running = True

        if poll_once:
            self.one_poll()
        else:
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()

    def stop(self):
        """停止监控"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("🛑 股票监控已停止")


# ---- CLI 入口 ----
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    from dotenv import load_dotenv
    load_dotenv(os.path.expanduser("~/.hermes/.env"))
    from config import WATCHLIST

    monitor = StockMonitor(
        tickers=WATCHLIST,
        check_interval_minutes=5,
        alert_threshold_pct=5.0,
    )
    monitor.start(poll_once=True)