"""
配置
"""
import os
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.hermes/.env"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")  # 你的 Telegram ID

# ===== Google Sheets 持仓记录 =====
GOOGLE_SHEETS_ID = "18GN184JSGv-L_xSKclBne6PtPJAsAkCRRSo51r3iHzU"
GOOGLE_SHEETS_URL = "https://docs.google.com/spreadsheets/d/18GN184JSGv-L_xSKclBne6PtPJAsAkCRRSo51r3iHzU/edit"
PORTFOLIO_MODULE = os.path.expanduser("~/.hermes/skills/productivity/ai-stock-trading-bot/references/google_sheets_portfolio.py")

# ===== 加密货币配置 =====
CRYPTO_EXCHANGE = os.getenv("CRYPTO_EXCHANGE", "okx")  # 默认交易所（okx/bybit/binance/gateio/bitget/kucoin）

# 加密货币观察列表
CRYPTO_WATCHLIST = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "XRP/USDT",
    "BNB/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
    "MATIC/USDT", "LTC/USDT", "UNI/USDT", "APT/USDT",
    "ARB/USDT",  # Layer2
    "INJ/USDT",  # Cosmos DeFi
    "SUI/USDT",  # 新公链
    "TIA/USDT",  # Celestia
    "PEPE/USDT",  # MEME
    "WIF/USDT",   # MEME
]

# 观察列表（外汇 + 加密 + 黄金，股票已不适合 $25 本金）
WATCHLIST = [
    # 外汇 CFD（MITRADE 200x）
    "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "USDJPY=X",
    "EURJPY=X", "GBPJPY=X",
    # 加密（MITRADE 100x 合约）
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "XRP/USDT",
    # 黄金（MITRADE 100x）
    "GC=F",
]

# 筛选条件（宽松版 — 演示用）
SCREENER_CONFIG = {
    "pe_max": 60,            # PE < 60（科技股合理区间）
    "pe_min": 0,             # PE > 0（排除负盈利）
    "price_change_min": 2,   # 近5日涨超2%
    "volume_ratio_min": 1.0, # 成交量持平即可
    "market_cap_min": 5e9,   # 市值 > 50亿美元
}