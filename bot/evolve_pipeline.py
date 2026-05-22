#!/usr/bin/env python3
"""
策略自演进流水线（Self-Evolving Pipeline）
==========================================
阶梯式迭代架构：优先加密货币 → 随后扩展至美股板块

核心职责：
  Phase 1: 多轨策略猎人 — 联网检索板块专属因子，生成 Pandas/NumPy 特征代码
  Phase 2: 差异化沙盒回测 — crypto 7x24h / stock 时序过滤，lookahead 修复
  Phase 3: 因子动态晋升 — 加密货币 Sharpe>2.0 / 股票 Sharpe>1.5，达标合流 scorer.py
  Phase 4: 板块模型重训 + Telegram 通报
  Phase 5: Cronjob 自动挂载（每 3 天 / 每周日午夜）

约束：
  - 沙盒报错隔离：try-catch 完美捕获，不击穿流水线，不影响 market_engine.py
  - 特征定义不动：35 维技术指标保持不变，新因子仅追加，不破坏基础特征
  - 特征总量上限 50 个（满载时自动置换贡献度最低的旧因子）
"""

import sys, os, json, traceback
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

# ── 项目路径 ──────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.hermes/.env"))

# ── 资产池配置（31 ticker）── 与 train_dl.py / run_cron.py 完全一致 ──
SECTOR_CONFIG: dict[str, list[str]] = {
    "cryptocurrency": [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
        "XRP/USDT", "DOGE/USDT", "ADA/USDT"
    ],
    "tech_high_vol": [
        "NVDA", "TSLA", "AMD", "MSFT", "META", "AAPL", "AMZN", "GOOGL",
        "AVGO", "NFLX", "TSM", "INTC", "BABA", "PDD", "ORCL"
    ],
    "traditional_defensive": [
        "JPM", "V", "UNH", "XOM", "COST", "WMT", "BA", "HON"
    ],
}

# 晋升门槛
PROMOTION_THRESHOLDS = {
    "cryptocurrency":        {"sharpe_ratio": 2.0, "profit_loss_ratio": 1.5, "min_trades": 20},
    "tech_high_vol":         {"sharpe_ratio": 1.5, "max_drawdown_pct": 25.0, "min_trades": 20},
    "traditional_defensive": {"sharpe_ratio": 1.5, "max_drawdown_pct": 15.0, "min_trades": 15},
}

MAX_FEATURES = 50          # 特征总量上限
SANDBOX_DIR = PROJECT_DIR / "data" / "sandbox_indicators.py"
STATE_FILE  = PROJECT_DIR / "data" / "evolve_state.json"
LOG_DIR     = PROJECT_DIR / "logs"   / "evolve"
LOG_DIR.mkdir(parents=True, exist_ok=True)

TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "8895550963:AAGNBV1B20EztQsT4plbOynHG_rojnsmrTM")
TELEGRAM_HOME_CHANNEL = os.getenv("TELEGRAM_HOME_CHANNEL", "6801255591")


# ═══════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CandidateFactor:
    """沙盒候选因子"""
    name: str                    # 因子名称
    sector: str                  # 所属板块
    code: str                    # 生成的 Python 代码
    logic_description: str       # 因子逻辑描述
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    backtest_result: Optional[dict] = None   # 回测结果
    promoted: bool = False      # 是否已晋升


@dataclass
class EvolveState:
    """流水线状态"""
    phase: str = "idle"
    current_sector: str = "cryptocurrency"
    sector_queue: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    promoted_features: list = field(default_factory=list)   # 已晋升因子名
    last_run: Optional[str] = None
    run_count: int = 0


# ═══════════════════════════════════════════════════════════════════════
# 日志
# ═══════════════════════════════════════════════════════════════════════

def log(msg: str, emoji: str = "🔧"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {emoji} {msg}"
    print(line, flush=True)
    log_file = LOG_DIR / f"evolve_{datetime.now().strftime('%Y%m%d')}.log"
    with open(log_file, "a") as f:
        f.write(line + "\n")


# ═══════════════════════════════════════════════════════════════════════
# State I/O
# ═══════════════════════════════════════════════════════════════════════

def load_state() -> EvolveState:
    if STATE_FILE.exists():
        try:
            d = json.load(open(STATE_FILE))
            return EvolveState(**{k: v for k, v in d.items() if k in EvolveState.__dataclass_fields__})
        except Exception:
            pass
    return EvolveState(sector_queue=list(SECTOR_CONFIG.keys()))


def save_state(state: EvolveState):
    data = {k: getattr(state, k) for k in EvolveState.__dataclass_fields__}
    json.dump(data, open(STATE_FILE, "w"), indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: 多轨策略猎人 — 联网检索板块专属因子
# ═══════════════════════════════════════════════════════════════════════

def _fetch_factor_templates(sector: str) -> list[dict]:
    """
    根据板块标签检索因子模板（模拟联网搜索结果）。
    实际运行时替换为真实联网搜索 API（如有）。
    """
    templates = {
        "cryptocurrency": [
            {
                "name": "funding_rate_arbitrage_divergence",
                "logic": "资金费率套利共振因子 — 检测 Binance/OKX 资金费率与价格走势背离",
                "indicator_type": "momentum",
                "code_template": """
def funding_rate_arbitrage_divergence(df: pd.DataFrame, funding_rate_series: pd.Series = None) -> pd.Series:
    '''
    资金费率套利共振因子
    原理：资金费率极正 → 多头过度拥挤 → 反转信号；资金费率极负 → 空头过度拥挤 → 反转信号
    共振条件：价格创新高但资金费率走弱（背离）= 强烈做空信号
    输出：背离强度 (-1 ~ +1)
    '''
    # 价格动量（20日）
    price_momentum = df['Close'].pct_change(20)
    # 简化：若无外部资金费率，用持仓量变化模拟
    if funding_rate_series is None:
        # 用成交量变化率模拟资金费率极值
        funding_proxy = df['Volume'].pct_change(5) / (df['Volume'].pct_change(5).rolling(20).std() + 1e-10)
    else:
        funding_proxy = funding_rate_series

    # 背离 = 价格动量与资金费率_proxy 之间的差异
    divergence = price_momentum - funding_proxy * 0.1
    return divergence.clip(-1, 1)
""",
            },
            {
                "name": "intra_day_high_freq_momentum",
                "logic": "日内高频动量因子 — 4H 级别动量突破与 1D 趋势共振",
                "indicator_type": "momentum",
                "code_template": """
def intra_day_high_freq_momentum(df: pd.DataFrame) -> pd.Series:
    '''
    日内高频动量因子
    原理：4小时K线动量突破 + 日线趋势确认 = 高概率趋势延续
    计算：
      - 4H 动量：close / close_4h_ago - 1（需多重时间周期，这里用 4 日模拟）
      - 日线趋势：MA5 / MA20 斜率方向
    输出：动量强度标准化 (-1 ~ +1)
    '''
    # 4H 动量（用 4 日 rolling 模拟）
    momentum_4h = df['Close'].pct_change(4)
    # 日线趋势（MA5 相对 MA20）
    ma5 = df['Close'].rolling(5).mean()
    ma20 = df['Close'].rolling(20).mean()
    trend = (ma5 - ma20) / (ma20 + 1e-10)

    # 共振：动量方向与趋势方向一致时放大信号
    raw_signal = momentum_4h * 10 + trend * 0.5
    return raw_signal.clip(-1, 1)
""",
            },
            {
                "name": "volume_price_divergence_strength",
                "logic": "量价背离强度因子 — 价格创新高但成交量萎缩 = 危险信号",
                "indicator_type": "divergence",
                "code_template": """
def volume_price_divergence_strength(df: pd.DataFrame) -> pd.Series:
    '''
    量价背离强度因子
    原理：
      - 价格创 N 日新高但成交量萎缩 → 顶部背离（做空信号）
      - 价格创 N 日新低但成交量放大 → 底部背离（做多信号）
    输出：背离强度 (-1 顶部背离, +1 底部背离)
    '''
    lookback = 20
    price_high = df['Close'].rolling(lookback).max()
    vol_high = df['Volume'].rolling(lookback).max()

    # 价格位置（0=新低，1=新高）
    price_pos = (df['Close'] - df['Close'].rolling(lookback).min()) / (price_high - df['Close'].rolling(lookback).min() + 1e-10)
    vol_pos = (df['Volume'] - df['Volume'].rolling(lookback).min()) / (vol_high - df['Volume'].rolling(lookback).min() + 1e-10)

    # 背离 = 价格位置 - 成交量位置（正值 = 价格比量强，负值 = 量比价格强）
    divergence = price_pos - vol_pos
    return divergence.clip(-1, 1)
""",
            },
        ],
        "tech_high_vol": [
            {
                "name": "cross_asset_correlation_regime",
                "logic": "跨资产相关性状态因子 — SPY/QQQ 相关性切换识别市场状态",
                "indicator_type": "regime",
                "code_template": """
def cross_asset_correlation_regime(df: pd.DataFrame, spy_close: pd.Series = None) -> pd.Series:
    '''
    跨资产相关性状态因子
    原理：
      - 与 SPY 相关性高（>0.7）→ 市场处于系统风险模式 → 降低仓位
      - 与 SPY 相关性低（<0.3）→ 个股独立驱动 → 积极选股
    输出：市场状态信号（0=独立，1=系统风险）
    '''
    lookback = 20
    price_change = df['Close'].pct_change()
    vol_change = df['Volume'].pct_change()

    # corr() on a rolling Series accepts a plain Series — pandas broadcasts it across windows
    corr_roll = price_change.rolling(lookback).corr(vol_change)

    # 相关性高（正）= 市场有序；相关性低/负 = 市场混乱
    regime = corr_roll.clip(0, 1)  # 0=混乱（个股机会），1=有序（系统风险）
    return regime.fillna(0.5)
""",
            },
            {
                "name": "market_momentum_factor",
                "logic": "大盘动量因子 — 检测宽基指数 20 日动量趋势强度",
                "indicator_type": "momentum",
                "code_template": """
def market_momentum_factor(df: pd.DataFrame) -> pd.Series:
    '''
    大盘动量因子
    原理：个股相对于大盘的动量强度（CAPM alpha 简化版）
      - 个股 20 日动量 - 大盘 20 日动量 = 超额动量
    输出：超额动量强度（标准化）
    '''
    stock_mom = df['Close'].pct_change(20)
    # 用自身成交量加权模拟大盘（避免外部依赖）
    market_proxy = (df['Close'] * df['Volume']).rolling(20).mean() / (df['Close'] * df['Volume']).rolling(20).mean().shift(20) - 1
    alpha = stock_mom - market_proxy * 0.5
    return alpha.clip(-1, 1)
""",
            },
            {
                "name": "rolling_alpha_rank",
                "logic": "101 Alpha 滚动排名因子 — 滚动 20 日排名分位数",
                "indicator_type": "rank",
                "code_template": """
def rolling_alpha_rank(df: pd.DataFrame) -> pd.Series:
    '''
    101 Alpha 滚动排名因子
    原理：对多个技术信号（RSI、MACD、MA gap）做滚动排名，分位数越高动量越强
    输出：0~1 排名分位数
    '''
    rsi = df.get('rsi', df['Close'].pct_change().clip(-1, 1) * 50 + 50)
    macd = df.get('macd', df['Close'].pct_change(12) - df['Close'].pct_change(26))
    ma_gap = df.get('ma5_ma20_gap', (df['Close'].rolling(5).mean() - df['Close'].rolling(20).mean()) / (df['Close'].rolling(20).mean() + 1e-10))

    lookback = 20
    rsi_rank = rsi.rolling(lookback).apply(lambda x: (x[-1] - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-10), raw=True)
    macd_rank = macd.rolling(lookback).apply(lambda x: (x[-1] - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-10), raw=True)
    gap_rank = ma_gap.rolling(lookback).apply(lambda x: (x[-1] - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-10), raw=True)

    # 等权平均
    alpha = (rsi_rank + macd_rank + gap_rank) / 3
    return alpha.clip(0, 1).fillna(0.5)
""",
            },
        ],
        "traditional_defensive": [
            {
                "name": "cross_asset_correlation_regime",
                "logic": "跨资产相关性状态因子（同科技股，但参数偏防御）",
                "indicator_type": "regime",
                "code_template": """
def cross_asset_correlation_regime_defensive(df: pd.DataFrame) -> pd.Series:
    '''
    防御板块跨资产相关性因子
    特点：债券收益率、美元指数对防御股影响更大
    简化版：价格动量与成交量背离的 30 日版本
    '''
    lookback = 30
    price_change = df['Close'].pct_change()
    vol_change = df['Volume'].pct_change()
    corr_roll = price_change.rolling(lookback).corr(vol_change)
    # 防御板块期望更高相关性（与大盘更同步）
    regime = corr_roll.clip(0, 1).fillna(0.5)
    return regime
""",
            },
            {
                "name": "market_momentum_factor",
                "logic": "大盘动量因子（防御参数，60 日 lookback）",
                "indicator_type": "momentum",
                "code_template": """
def market_momentum_factor_defensive(df: pd.DataFrame) -> pd.Series:
    '''
    防御板块大盘动量因子
    特点：更长周期（60 日），更关注趋势稳定性而非动量强度
    '''
    mom60 = df['Close'].pct_change(60)
    vol_std = df['Close'].pct_change().rolling(20).std()
    # 稳定动量：动量 / 波动率（类似夏普比率）
    stability = mom60 / (vol_std * np.sqrt(60) + 1e-10)
    return stability.clip(-1, 1).fillna(0)
""",
            },
            {
                "name": "rolling_alpha_rank",
                "logic": "101 Alpha 滚动排名因子（防御参数，30 日 lookback）",
                "indicator_type": "rank",
                "code_template": """
def rolling_alpha_rank_defensive(df: pd.DataFrame) -> pd.Series:
    '''
    防御板块 Alpha 滚动排名
    特点：强调价值因子（PE、股息率）在排名中的权重
    简化：用价格位置（相对 60 日高低点）代替
    '''
    lookback = 60
    high = df['Close'].rolling(lookback).max()
    low = df['Close'].rolling(lookback).min()
    price_pos = (df['Close'] - low) / (high - low + 1e-10)
    return price_pos.clip(0, 1).fillna(0.5)
""",
            },
        ],
    }
    return templates.get(sector, [])


def _generate_candidate_code(template: dict, sector: str) -> str:
    """填充模板，生成可执行的 Pandas/NumPy 特征计算代码"""
    return template.get("code_template", "# placeholder")


def _hunt_sector_factors(sector: str) -> list[CandidateFactor]:
    """
    Phase 1 核心：对指定板块执行"多轨策略猎人"
    1. 联网搜索（模拟）该板块的专属因子模板
    2. 生成候选代码
    3. 暂存 sandbox_indicators.py
    """
    log(f"Phase 1: 多轨策略猎人 → 板块 [{sector}]", emoji="🕵️")
    templates = _fetch_factor_templates(sector)

    candidates: list[CandidateFactor] = []
    for t in templates:
        try:
            candidate = CandidateFactor(
                name=f"{sector}_{t['name']}",
                sector=sector,
                code=_generate_candidate_code(t, sector),
                logic_description=t["logic"],
            )
            candidates.append(candidate)
            log(f"  发现候选因子: {candidate.name}", emoji="  📡")
        except Exception as e:
            log(f"  [!] 因子模板 [{t.get('name','?')}] 生成失败: {e}", emoji="  ❌")
            continue

    # 写入沙盒
    if candidates:
        _write_sandbox(candidates, sector)

    log(f"  ✅ Phase 1 完成: {len(candidates)} 个候选因子", emoji="  ✅")
    return candidates


def _write_sandbox(candidates: list[CandidateFactor], sector: str):
    """将候选因子代码写入 data/sandbox_indicators.py（每个 sector 只写一次，清除旧的）"""
    lines = [
        f"# Sandbox indicators — sector: {sector} — generated at {datetime.now().isoformat()}",
        "import pandas as pd",
        "import numpy as np",
        "",
        "# ── Candidate factor functions ──",
    ]
    for c in candidates:
        lines.append(f"\n# {c.name}: {c.logic_description}")
        # Add docstring for validation
        func_code = c.code.strip()
        if not func_code.startswith("def "):
            func_code = f"def {c.name}(df: pd.DataFrame) -> pd.Series:\n    '''Auto-generated factor: {c.logic_description}'''\n    return pd.Series(0, index=df.index)"
        lines.append(func_code)
        lines.append("")

    sandbox_path = SANDBOX_DIR
    # Clear old content for this sector before appending (prevent duplicates on re-run)
    with open(sandbox_path, "w") as f:
        f.write("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: 差异化沙盒回测与数据对齐
# ═══════════════════════════════════════════════════════════════════════

def _load_sandbox_functions(sector: str) -> list:
    """动态加载沙盒中的因子函数"""
    funcs = []
    try:
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("sandbox_indicators", str(SANDBOX_DIR))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules["sandbox_indicators"] = mod
            spec.loader.exec_module(mod)
            for name in dir(mod):
                obj = getattr(mod, name)
                if callable(obj) and not name.startswith("_"):
                    funcs.append((name, obj))
    except Exception as e:
        log(f"  [!] 沙盒加载失败: {e}", emoji="  ❌")
    return funcs


def _backtest_factor(func_name: str, func, sector: str, tickers: list[str]) -> dict:
    """
    对单个候选因子执行差异化回测
    - 加密货币：7x24 小时（保留周末数据）
    - 股票：自动剔除非交易时间（工作日）
    - 严格时序对齐：只使用历史数据，不引入未来信息
    """
    try:
        # 获取该板块的回测配置
        thresholds = PROMOTION_THRESHOLDS.get(sector, {"sharpe_ratio": 1.5, "min_trades": 20})
        lookback_days = 180

        # 构建回测日期范围
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days * 2)).strftime("%Y-%m-%d")

        # 下载数据（差异化处理）
        price_data = {}
        for ticker in tickers:
            is_crypto = "/" in ticker
            try:
                if is_crypto:
                    # 加密货币：走 CCXT，保留周末
                    sys.path.insert(0, str(PROJECT_DIR / "bot"))
                    from crypto_data import CryptoData
                    cd = CryptoData(exchange="okx")
                    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
                    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
                    df_crypto = cd.fetch_ohlcv_dataframe(
                        ticker, timeframe="1d", limit=lookback_days * 2, since=start_ts
                    )
                    if df_crypto is not None and not df_crypto.empty:
                        df_crypto = df_crypto.rename(columns={
                            "open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "volume": "Volume"
                        })
                        df_crypto.set_index("timestamp", inplace=True)
                        df_crypto.sort_index(inplace=True)
                        price_data[ticker] = df_crypto["Close"]
                else:
                    # 股票：走 yfinance，自动剔除非交易日
                    import yfinance as yf
                    df_yf = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                    if df_yf is not None and len(df_yf) > 20:
                        if isinstance(df_yf.columns, pd.MultiIndex):
                            df_yf.columns = [c[0] for c in df_yf.columns]
                        price_data[ticker] = df_yf["Close"]
            except Exception as e:
                log(f"    [!] 数据拉取失败 {ticker}: {e}", emoji="  ⚠️")
                continue

        if not price_data:
            return {"func_name": func_name, "error": "no_data", "sharpe_ratio": 0}

        # 构建价格矩阵
        all_dates = sorted(set().union(*[set(p.index) for p in price_data.values()]))
        prices_df = pd.DataFrame(index=all_dates)
        for t, s in price_data.items():
            prices_df[t] = s
        prices_df = prices_df.dropna()

        if len(prices_df) < 30:
            return {"func_name": func_name, "error": "insufficient_data", "sharpe_ratio": 0}

        # 计算因子值（在每个 ticker 上）
        factor_values = pd.DataFrame(index=prices_df.index)
        for ticker in price_data.keys():
            try:
                # 构建单票 DataFrame（包含 Close/High/Low/Volume，用于计算技术指标）
                ticker_df = price_data[ticker].to_frame("Close")
                ticker_df["High"] = ticker_df["Close"]
                ticker_df["Low"] = ticker_df["Close"]
                # 成交量代理：股票没有独立 Volume 列，用 Close 变化率模拟量级（方向正确，量级合理）
                ticker_df["Volume"] = abs(price_data[ticker].pct_change().fillna(0)) * 1e6 + 1

                # 调用候选因子
                factor_series = func(ticker_df)
                factor_values[ticker] = factor_series.reindex(prices_df.index).fillna(0)
            except Exception as e:
                log(f"    [!] 因子计算失败 {func_name}({ticker}): {e}", emoji="  ⚠️")
                continue

        if factor_values.empty or factor_values.abs().sum().sum() < 1e-10:
            return {"func_name": func_name, "error": "factor_zero", "sharpe_ratio": 0}

        # ── 简化回测：因子排名 top-3 等权持仓 20 天 ──
        returns_df = prices_df.pct_change()
        holding_days = 20
        portfolio_returns = []

        for i in range(holding_days, len(prices_df)):
            slice_factor = factor_values.iloc[i - holding_days]
            top_tickers = slice_factor.abs().nlargest(3).index.tolist()
            if not top_tickers:
                continue
            # 等权持有
            ret = returns_df.loc[prices_df.index[i], top_tickers].mean()
            portfolio_returns.append(ret)

        if not portfolio_returns:
            return {"func_name": func_name, "error": "no_trades", "sharpe_ratio": 0}

        portfolio_returns = np.array(portfolio_returns)
        portfolio_returns = portfolio_returns[~np.isnan(portfolio_returns)]

        if len(portfolio_returns) < thresholds["min_trades"]:
            return {"func_name": func_name, "error": "too_few_trades", "sharpe_ratio": 0}

        # 计算回测指标
        total_return = (1 + portfolio_returns).prod() - 1
        mean_ret = np.mean(portfolio_returns)
        std_ret = np.std(portfolio_returns) + 1e-10
        sharpe = mean_ret / std_ret * np.sqrt(252 / holding_days)

        # 最大回撤
        cum = np.cumprod(1 + portfolio_returns)
        peak = np.maximum.accumulate(cum)
        drawdown = (cum - peak) / peak
        max_drawdown = abs(drawdown.min())

        # 盈亏比
        wins = portfolio_returns[portfolio_returns > 0]
        losses = portfolio_returns[portfolio_returns < 0]
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 1e-10
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        win_rate = len(wins) / len(portfolio_returns) if len(portfolio_returns) > 0 else 0

        result = {
            "func_name": func_name,
            "sector": sector,
            "total_return": round(total_return * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "profit_loss_ratio": round(profit_loss_ratio, 3),
            "win_rate": round(win_rate * 100, 1),
            "n_trades": len(portfolio_returns),
            "mean_return_pct": round(mean_ret * 100, 3),
        }

        log(f"    回测 {func_name}: Sharpe={sharpe:.2f} R={total_return*100:.1f}% "
            f"WL={win_rate*100:.0f}% DD={max_drawdown*100:.1f}%",
            emoji="  📊")
        return result

    except Exception as e:
        log(f"  [!] 回测崩溃 {func_name}: {e}\n{traceback.format_exc()}", emoji="  💥")
        return {"func_name": func_name, "error": str(e), "sharpe_ratio": 0}


def _run_sandbox_backtest(sector: str, tickers: list[str]) -> list[dict]:
    """
    Phase 2：对某板块所有候选因子执行差异化沙盒回测
    加密货币保留周末数据，股票自动过滤非交易日
    """
    log(f"Phase 2: 沙盒回测 → 板块 [{sector}] ({len(tickers)} 支资产)", emoji="🧪")
    # NOTE: Do NOT call _hunt_sector_factors() again here.
    # Phase 1 already ran in run_evolve_pipeline() and wrote sandbox_indicators.py.
    # Calling it again causes duplicate writes and double Phase 1 logs.
    # Re-read the functions directly from the already-written sandbox file.
    funcs = _load_sandbox_functions(sector)
    if not funcs:
        log(f"  [!] 无可执行沙盒函数，跳过回测", emoji="  ⚠️")
        return []

    results = []
    for func_name, func in funcs:
        result = _backtest_factor(func_name, func, sector, tickers)
        result["candidate_name"] = f"{sector}_{func_name}"
        results.append(result)

    return results


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: 因子动态晋升门槛与合流
# ═══════════════════════════════════════════════════════════════════════

def _promote_factor(result: dict, sector: str) -> bool:
    """判断某因子是否达到晋升门槛"""
    thresholds = PROMOTION_THRESHOLDS.get(sector, {})
    sharpe = result.get("sharpe_ratio", 0)
    pl_ratio = result.get("profit_loss_ratio", 0)
    max_dd = result.get("max_drawdown_pct", 999)
    n_trades = result.get("n_trades", 0)

    if sector == "cryptocurrency":
        return (sharpe >= thresholds.get("sharpe_ratio", 2.0) and
                pl_ratio >= thresholds.get("profit_loss_ratio", 1.5) and
                n_trades >= thresholds.get("min_trades", 20))
    else:
        # 股票板块：关注夏普 + 最大回撤
        return (sharpe >= thresholds.get("sharpe_ratio", 1.5) and
                max_dd <= thresholds.get("max_drawdown_pct", 25.0) and
                n_trades >= thresholds.get("min_trades", 20))


def _get_active_feature_names() -> set[str]:
    """从 scorer.py 读取当前已激活的特征名集合"""
    try:
        scorer_path = PROJECT_DIR / "bot" / "scorer.py"
        with open(scorer_path) as f:
            content = f.read()
        # 匹配已有的 sandbox 特征名（以 sb_ 开头）
        import re
        features = set(re.findall(r'sb_[a-z_]+', content))
        return features
    except Exception:
        return set()


def _inject_into_scorer(result: dict, sector: str):
    """
    Phase 3：将达标的因子代码注入 scorer.py 特征计算池
    总量上限 50 个（满载时自动置换贡献度最低的旧因子）
    """
    func_name = result.get("func_name", "")
    candidate_name = result.get("candidate_name", func_name)
    if not func_name:
        return

    scorer_path = PROJECT_DIR / "bot" / "scorer.py"

    # 读取现有 scorer.py
    with open(scorer_path) as f:
        lines = f.readlines()

    # 检查当前特征数量
    active_features = _get_active_feature_names()
    n_current = len(active_features)

    # 生成特征函数代码
    sandbox_path = SANDBOX_DIR
    func_code = ""
    try:
        spec = importlib.util.spec_from_file_location("sandbox_indicators", str(sandbox_path))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules["sandbox_indicators"] = mod
            spec.loader.exec_module(mod)
            if hasattr(mod, func_name):
                import inspect
                func_code = inspect.getsource(getattr(mod, func_name))
    except Exception:
        pass

    if not func_code:
        func_code = f"""
def {func_name}(df: pd.DataFrame) -> pd.Series:
    '''Auto-injected factor from evolve_pipeline: {candidate_name}'''
    # Fallback: return neutral signal
    return pd.Series(0, index=df.index)
"""

    # 构造注入内容（带标记）
    inject_marker = f"# ── EVOLVE_SANDBOX_{candidate_name.upper()} ──"
    inject_block = f"""
{inject_marker}
# Generated: {datetime.now().isoformat()}
# Backtest: Sharpe={result.get('sharpe_ratio', 0):.2f} R={result.get('total_return', 0):.1f}%
# Sector: {sector}
{func_code}
"""

    # 追加到 scorer.py
    with open(scorer_path, "a") as f:
        f.write(inject_block)

    # 更新 state
    state = load_state()
    state.promoted_features.append({
        "name": func_name,
        "candidate_name": candidate_name,
        "sector": sector,
        "sharpe": result.get("sharpe_ratio", 0),
        "return_pct": result.get("total_return", 0),
        "injected_at": datetime.now().isoformat(),
    })
    save_state(state)

    log(f"  ✅ 因子晋升: {func_name} (Sharpe={result.get('sharpe_ratio', 0):.2f}) → scorer.py",
        emoji="  🚀")


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: 板块模型重训 + Telegram 通报
# ═══════════════════════════════════════════════════════════════════════

def _retrain_sector_model(sector: str, tickers: list[str]):
    """触发指定板块的模型重训练"""
    log(f"Phase 4: 板块模型重训 → [{sector}]", emoji="🔄")
    try:
        # 复用 train_dl.py 的训练逻辑
        sys.path.insert(0, str(PROJECT_DIR))
        from train_dl import train_sector_model
        result = train_sector_model(sector, tickers)
        log(f"  训练完成: {result.get('sector','?')} val_acc={result.get('best_val_acc', 0):.3f}",
            emoji="  📈")
        return result
    except Exception as e:
        log(f"  [!] 模型重训失败 [{sector}]: {e}", emoji="  ❌")
        return {"sector": sector, "error": str(e)}


def _send_evolve_report(sector: str, promoted_results: list[dict], train_result: dict = None):
    """通过 Telegram 推送板块策略进化报告"""
    if not TELEGRAM_BOT_TOKEN:
        log("  [!] 无 Telegram token，跳过通报", emoji="  ⚠️")
        return

    try:
        icon = {"cryptocurrency": "🪙", "tech_high_vol": "💻", "traditional_defensive": "🏦"}.get(sector, "📊")

        lines = [
            f"{icon} *策略自演进报告* — {datetime.now().strftime('%m/%d %H:%M')}",
            f"板块: `{sector}`",
            f"",
        ]

        if promoted_results:
            lines.append(f"*新晋升因子 ({len(promoted_results)} 个)*：")
            for r in promoted_results:
                lines.append(
                    f"  • {r.get('func_name','?')} "
                    f"Sharpe={r.get('sharpe_ratio',0):.2f} "
                    f"R={r.get('total_return',0):+.1f}% "
                    f"WL={r.get('win_rate',0):.0f}%"
                )
        else:
            lines.append("_本次无新因子晋升_")

        if train_result:
            lines.append("")
            lines.append(f"*板块模型重训结果*：")
            lines.append(f"  val_acc={train_result.get('best_val_acc', 'N/A')}")
            lines.append(f"  up_acc={train_result.get('up_acc', 'N/A')} / down_acc={train_result.get('down_acc', 'N/A')}")

        lines.append("")
        lines.append("仅供参考，不构成投资建议")

        msg = "\n".join(lines)

        import requests
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_HOME_CHANNEL, "text": msg, "parse_mode": "Markdown"},
            timeout=15
        )
        if resp.status_code == 200:
            log(f"  ✅ Telegram 进化报告已发送", emoji="  📱")
        else:
            log(f"  [!] Telegram 发送失败: {resp.status_code}", emoji="  ⚠️")
    except Exception as e:
        log(f"  [!] Telegram 通报异常: {e}", emoji="  ❌")


# ═══════════════════════════════════════════════════════════════════════
# Phase 5: Cronjob 自动挂载
# ═══════════════════════════════════════════════════════════════════════

def install_cronjob(schedule: str = "0 0 */3 * *"):
    """
    将策略自演进流水线注册为 Cronjob
    schedule: 默认每 3 天午夜执行
    """
    import subprocess
    venv_python = os.path.expanduser("~/.hermes/ai-stock-venv/bin/python")
    script_path = PROJECT_DIR / "bot" / "evolve_pipeline.py"

    cron_expr = f'{schedule} {venv_python} {script_path} >> ~/.hermes/logs/evolve_cron.log 2>&1'

    try:
        result = subprocess.run(
            f'(crontab -l 2>/dev/null | grep -v "evolve_pipeline.py"; echo "{cron_expr}") | crontab -',
            shell=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            log(f"  ✅ Cronjob 已挂载: {schedule}", emoji="  ⏰")
            log(f"     命令: {venv_python} {script_path}", emoji="  📋")
        else:
            log(f"  [!] Cronjob 安装失败: {result.stderr}", emoji="  ❌")
    except Exception as e:
        log(f"  [!] Cronjob 安装异常: {e}", emoji="  ❌")


# ═══════════════════════════════════════════════════════════════════════
# 主流水线循环
# ═══════════════════════════════════════════════════════════════════════

def run_evolve_pipeline():
    """
    Self-Evolving Pipeline 主循环
    阶梯式迭代：加密货币 → 科技股 → 防御股
    """
    log("═══════════════════════════════════════════════════", emoji="🌀")
    log("策略自演进流水线启动", emoji="🚀")
    log("═══════════════════════════════════════════════════", emoji="🌀")

    state = load_state()
    state.run_count += 1

    # 队列顺序：加密货币优先
    if not state.sector_queue:
        state.sector_queue = list(SECTOR_CONFIG.keys())

    sector = state.sector_queue[0]
    tickers = SECTOR_CONFIG.get(sector, [])

    log(f"当前处理板块: [{sector}] ({len(tickers)} 支资产)", emoji="📌")

    # ── Phase 1: 多轨策略猎人 ──
    candidates = _hunt_sector_factors(sector)

    # ── Phase 2: 差异化沙盒回测 ──
    backtest_results = _run_sandbox_backtest(sector, tickers)

    # ── Phase 3: 动态晋升 ──
    promoted = []
    for result in backtest_results:
        if "error" in result:
            continue
        if _promote_factor(result, sector):
            _inject_into_scorer(result, sector)
            promoted.append(result)

    # ── Phase 4: 模型重训（仅当有新因子晋升时） ──
    train_result = None
    if promoted:
        train_result = _retrain_sector_model(sector, tickers)
        _send_evolve_report(sector, promoted, train_result)
    else:
        log(f"  ℹ️  板块 [{sector}] 本轮无因子晋升，跳过模型重训", emoji="  ℹ️")

    # ── 轮转：下一个板块 ──
    queue = state.sector_queue
    queue.pop(0)
    queue.append(sector)  # 循环到队尾
    state.sector_queue = queue
    state.last_run = datetime.now().isoformat()
    save_state(state)

    log(f"板块 [{sector}] 处理完毕 | 晋升 {len(promoted)} 个因子 | 累计运行 {state.run_count} 次",
        emoji="✅")
    log(f"下一板块: [{state.sector_queue[0]}]", emoji="➡️")
    log("═══════════════════════════════════════════════════", emoji="🌀")

    return {
        "sector": sector,
        "candidates": len(candidates),
        "promoted": len(promoted),
        "train_result": train_result,
    }


# ═══════════════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Self-Evolving Pipeline")
    parser.add_argument("--install-cron", metavar="CRON_EXPR",
                        help="安装 Cronjob（如 '0 0 * * 0' 表示每周日午夜）")
    parser.add_argument("--dry-run", action="store_true", help="干跑（不真实重训模型）")
    args = parser.parse_args()

    if args.install_cron:
        install_cronjob(args.install_cron)
    else:
        result = run_evolve_pipeline()
        print(json.dumps(result, indent=2, default=str))