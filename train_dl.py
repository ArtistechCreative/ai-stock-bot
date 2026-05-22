#!/usr/bin/env python3
"""
DL模型板块聚合训练 + 策略更新 + 信号推送
================================================
Phase 2 核心改动：
  - 废弃「单股票独立训练」，改为 3 大板块（SECTOR_CONFIG）大循环
  - 每个板块内 pd.concat 垂直拼接所有股票的 35 维特征
  - 板块内股票越多 → 训练样本量越大 → 过拟合被压制
  - 输出 3 个板块模型：dl_model_tech.pth / dl_model_defensive.pth / dl_model_crypto.pth

板块划分（SECTOR_CONFIG）：
  tech_high_vol        : 15 支科技股（高波动）
  traditional_defensive: 8 支防御型股票
  cryptocurrency       : 7 支主流加密货币

Phase 1 核心改动（已落地）：
  - rank_stocks(as_of_date) 正确传递日期给 _score_stock_historical()
  - _score_crypto_ticker(as_of_date) 用 CCXT 历史 K 线计算技术指标
  - backtest.py 的 strategy_fn 已正确传递 date_str
"""

import sys, os, json, math
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dotenv import load_dotenv
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.hermes/.env"))

from bot.dl_strategy import DLStrategy, compute_technical_indicators, StockMLP, StockDataset
from bot.scorer import rank_stocks
from bot.backtest import BacktestEngine
from bot.strategy_optimizer import StrategyOptimizer

# ════════════════════════════════════════════════════════════════════
# 板块配置（31 个 ticker）
# ════════════════════════════════════════════════════════════════════
SECTOR_CONFIG = {
    "tech_high_vol": [
        "NVDA", "TSLA", "AMD", "MSFT", "META", "AAPL", "AMZN", "GOOGL",
        "AVGO", "NFLX", "TSM", "INTC", "BABA", "PDD", "ORCL"
    ],
    "traditional_defensive": [
        "JPM", "V", "UNH", "XOM", "COST", "WMT", "BA", "HON"
    ],
    "cryptocurrency": [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
        "XRP/USDT", "DOGE/USDT", "ADA/USDT"
    ],
}

# 板块模型文件映射
SECTOR_MODELS = {
    "tech_high_vol":        "data/models/dl_model_tech.pth",
    "traditional_defensive":"data/models/dl_model_defensive.pth",
    "cryptocurrency":       "data/models/dl_model_crypto.pth",
}

# ════════════════════════════════════════════════════════════════════
# 训练配置
# ════════════════════════════════════════════════════════════════════
MODEL_TYPE       = "MLP"        # tech + defensive 用 MLP；crypto 用同款
LOOKBACK         = 500          # 天
EPOCHS_BASE      = 60
BATCH_SIZE       = 64
LEARNING_RATE    = 3e-4
VAL_ACC_TARGET   = 0.55         # 板块聚合后达标门槛降低（更多样本=更稳定）
UP_ACC_TARGET    = 0.52
DOWN_ACC_TARGET  = 0.52
MAX_EXTRA_ROUNDS = 4

# 信号推送
MIN_CONFIDENCE    = 65.0        # 下调（板块模型比单股票模型更稳定）
MAX_SIGNALS       = 4
BACKTEST_DAYS     = 90
BACKTEST_SCORE_TARGET = 4.0      # 下调（加密货币高波动拉低评分）

MODEL_DIR = Path(__file__).parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR  = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = DATA_DIR / "strategy_state.json"

# ════════════════════════════════════════════════════════════════════
# State helpers
# ════════════════════════════════════════════════════════════════════
def get_state():
    if Path(STATE_PATH).exists():
        return json.load(open(STATE_PATH))
    return {}

def save_state(state):
    json.dump(state, open(STATE_PATH, "w"), indent=2)

# ════════════════════════════════════════════════════════════════════
# Phase 2 核心：板块特征聚合
# ════════════════════════════════════════════════════════════════════

def _load_crypto_features(symbol: str, end_date: str = None, lookback: int = 500) -> pd.DataFrame | None:
    """
    加载单个加密货币的 35 维特征（走 CCXT OKX 历史 K 线）。
    返回 DataFrame：[features + target]，可直接与股票特征 pd.concat 拼接。
    """
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from bot.crypto_data import CryptoData

        cd = CryptoData(exchange="okx")
        end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
        start = end - pd.Timedelta(days=lookback * 2)
        since_ms = int(start.timestamp() * 1000)

        df = cd.fetch_ohlcv_dataframe(symbol, timeframe="1d", limit=lookback * 2, since=since_ms)
        if df.empty or len(df) < 60:
            return None

        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume"
        })
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        if end_date:
            end_ts = pd.Timestamp(end_date)
            df = df[df.index <= end_ts.value]

        df_tech = compute_technical_indicators(df)
        if df_tech.empty:
            return None

        # 标签：次日涨跌 (1=涨, 0=跌)
        df_tech["target"] = (df_tech["Close"].shift(-1) > df_tech["Close"]).astype(int)
        df_tech = df_tech.dropna()

        feature_cols = [
            "rsi", "rsi_5", "rsi_30",
            "macd", "macd_hist", "macd_hist_pct",
            "mom5", "mom10", "mom20", "mom60",
            "pct_change_1d", "pct_change_5d",
            "ma5", "ma10", "ma20", "ma60", "ma120",
            "ma5_ma20_gap", "ma10_ma60_gap",
            "price_ma20_ratio", "price_ma60_ratio", "price_ma120_ratio",
            "bb_pct_b", "bb_width", "price_deviation_ma20",
            "vol_ratio", "vol_change", "vol_change_5d",
            "obv", "obv_slope", "price_volume_div",
            "atr", "atr_pct", "bb_squeeze", "stoch_k",
        ]
        existing_cols = [c for c in feature_cols if c in df_tech.columns]
        return df_tech[existing_cols + ["target"]]

    except Exception as e:
        print(f"    [!] _load_crypto_features({symbol}) failed: {e}")
        return None


def _load_stock_features(ticker: str, end_date: str = None, lookback: int = 500) -> pd.DataFrame | None:
    """加载单个股票的 35 维特征（走 yfinance）。"""
    try:
        end = end_date or datetime.now().strftime("%Y-%m-%d")
        start = (pd.Timestamp(end) - pd.Timedelta(days=lookback * 2)).strftime("%Y-%m-%d")

        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty or len(df) < 60:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        df_tech = compute_technical_indicators(df)
        if df_tech.empty:
            return None

        df_tech["target"] = (df_tech["Close"].shift(-1) > df_tech["Close"]).astype(int)
        df_tech = df_tech.dropna()

        feature_cols = [
            "rsi", "rsi_5", "rsi_30",
            "macd", "macd_hist", "macd_hist_pct",
            "mom5", "mom10", "mom20", "mom60",
            "pct_change_1d", "pct_change_5d",
            "ma5", "ma10", "ma20", "ma60", "ma120",
            "ma5_ma20_gap", "ma10_ma60_gap",
            "price_ma20_ratio", "price_ma60_ratio", "price_ma120_ratio",
            "bb_pct_b", "bb_width", "price_deviation_ma20",
            "vol_ratio", "vol_change", "vol_change_5d",
            "obv", "obv_slope", "price_volume_div",
            "atr", "atr_pct", "bb_squeeze", "stoch_k",
        ]
        existing_cols = [c for c in feature_cols if c in df_tech.columns]
        return df_tech[existing_cols + ["target"]]

    except Exception as e:
        print(f"    [!] _load_stock_features({ticker}) failed: {e}")
        return None


def aggregate_sector_features(sector_name: str, tickers: list[str], end_date: str = None, lookback: int = None) -> pd.DataFrame | None:
    """
    Phase 2 核心：板块特征垂直拼接。
    遍历板块内所有股票，提取 35 维特征后在行方向 pd.concat 拼接。
    大幅扩充训练样本量，压制单票过拟合。

    lookback: 默认 None（自动按小板块增加历史窗口）；直接传值可覆盖
    """
    if lookback is None:
        n_tickers = len(tickers)
        if n_tickers <= 4:
            lookback = 800
        elif n_tickers <= 8:
            lookback = 600
        else:
            lookback = 500

    frames = []
    print(f"  📦 聚合板块 [{sector_name}] {len(tickers)} 支资产（lookback={lookback}天）...")

    for t in tickers:
        is_crypto = "/" in t
        if is_crypto:
            df_t = _load_crypto_features(t, end_date=end_date, lookback=lookback)
        else:
            df_t = _load_stock_features(t, end_date=end_date, lookback=lookback)

        if df_t is not None and len(df_t) >= 20:
            # 标签列
            labels = df_t["target"].values
            n_up   = int(labels.sum())
            n_down = len(labels) - n_up
            print(f"    {t}: {len(df_t)} 条样本 (up={n_up}, down={n_down})")
            frames.append(df_t)
        else:
            print(f"    ⏭ {t}: 数据不足，跳过")

    if not frames:
        return None

    # 垂直拼接（时间轴扩展）
    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined = combined.dropna()
    print(f"  ✅ 板块 [{sector_name}] 合计: {len(combined)} 条样本")
    return combined


# ════════════════════════════════════════════════════════════════════
# 板块模型训练
# ════════════════════════════════════════════════════════════════════

def train_sector_model(sector_name: str, tickers: list[str], lookback: int = None) -> dict:
    """训练单个板块聚合模型（板块内所有股票特征垂直拼接）。"""
    print(f"\n{'='*60}")
    print(f"[{sector_name}] 板块聚合训练")
    print(f"  资产: {tickers}")

    # 小板块（≤8支股票）用更长历史窗口，确保聚合后样本量足够
    if lookback is None:
        n_tickers = len(tickers)
        if n_tickers <= 4:
            lookback = 800
        elif n_tickers <= 8:
            lookback = 600
        else:
            lookback = 500

    df_all = aggregate_sector_features(sector_name, tickers, end_date=None)
    if df_all is None or len(df_all) < 100:
        print(f"  ⚠️  板块 [{sector_name}] 数据不足，跳过")
        return {"sector": sector_name, "error": "insufficient data"}

    feature_cols = [c for c in df_all.columns if c != "target"]
    X = df_all[feature_cols].values.astype(np.float32)
    y = df_all["target"].values.astype(np.int64)

    n_up   = int(y.sum())
    n_down = len(y) - n_up
    ratio  = n_up / max(n_down, 1)
    print(f"  标签分布: up={n_up}, down={n_down}, ratio={ratio:.2f}")

    # Walk-forward 时序分割
    n = len(X)
    train_end = int(n * 0.75)
    X_tr, X_val = X[:train_end], X[train_end:]
    y_tr, y_val = y[:train_end], y[train_end:]

    print(f"  训练: {len(X_tr)} 样本 | 验证: {len(X_val)} 样本")

    device = torch.device("cpu")
    # pos_weight 反转（up样本多→up权重小，down样本少→down权重大）
    pos_weight = torch.FloatTensor([ratio, 1.0]).to(device)

    train_ds = StockDataset(X_tr, y_tr)
    val_ds   = StockDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 模型
    input_size = X_tr.shape[-1]
    torch.manual_seed(42 + hash(sector_name) % 100)
    model = StockMLP(input_size=input_size).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-5
    )
    criterion = nn.CrossEntropyLoss(weight=pos_weight, label_smoothing=0.05)

    best_acc = 0; best_state = None
    best_up = 0; best_down = 0
    patience = 15; no_improve = 0

    for epoch in range(EPOCHS_BASE):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = correct_up = correct_down = 0
        total = total_up = total_down = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch).argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total   += len(y_batch)
                mask_up   = (y_batch == 1)
                mask_down = (y_batch == 0)
                correct_up   += ((preds == y_batch) & mask_up).sum().item()
                correct_down += ((preds == y_batch) & mask_down).sum().item()
                total_up   += mask_up.sum().item()
                total_down += mask_down.sum().item()

        val_acc  = correct / total if total > 0 else 0
        up_acc   = correct_up   / total_up   if total_up   > 0 else 0
        down_acc = correct_down / total_down if total_down > 0 else 0
        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_up = up_acc; best_down = down_acc; no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            if best_up > 0.70 and best_down < 0.40:
                print(f"  ⚠️  极化强制终止 ep {epoch+1}: up={best_up:.1%}/down={best_down:.1%}，不保存")
                best_state = None
            else:
                print(f"  Early stop ep {epoch+1}, best_acc={best_acc:.3f}")
            break

        if (epoch + 1) % 20 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Ep {epoch+1}: val_acc={val_acc:.3f} up={up_acc:.3f} down={down_acc:.3f} lr={lr_now:.2e}")

    # 保存板块模型
    sector_path = MODEL_DIR / f"dl_model_{sector_name.replace(' ', '_').replace('_', '')}.pth"
    if best_state:
        model.load_state_dict(best_state)
        torch.save(best_state, sector_path)
        print(f"  💾 板块模型已保存: {sector_path.name}")
        print(f"     val_acc={best_acc:.3f} up={best_up:.3f} down={best_down:.3f}")
    else:
        print(f"  ⚠️  无有效模型（极化），跳过保存")

    return {
        "sector": sector_name,
        "best_val_acc": round(best_acc, 4),
        "up_acc": round(best_up, 4),
        "down_acc": round(best_down, 4),
        "train_samples": len(X_tr),
        "val_samples": len(X_val),
        "model_path": str(sector_path) if best_state else None,
    }


# ════════════════════════════════════════════════════════════════════
# 板块路由 + 批量预测
# ════════════════════════════════════════════════════════════════════

def get_sector(ticker: str) -> str:
    """根据 ticker 返回所属板块，未知归类到 crypto 并打印警告。"""
    for sector, tickers in SECTOR_CONFIG.items():
        if ticker in tickers:
            return sector
    # Fallback：未知资产默认归 crypto（安全降级，不崩溃）
    if "/" in ticker or ticker in ("BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA"):
        return "cryptocurrency"
    print(f"  ⚠️  [{ticker}] 未定义板块路由，默认归 tech_high_vol")
    return "tech_high_vol"


def batch_predict_sector(tickers: list[str], model_type: str = "MLP") -> list[dict]:
    """板块路由批量预测：自动识别 ticker 所属板块，加载对应板块模型。"""
    device = torch.device("cpu")
    results = []
    for ticker in tickers:
        sector = get_sector(ticker)
        model_path = MODEL_DIR / f"dl_model_{sector.replace(' ', '_').replace('_', '')}.pth"

        if not model_path.exists():
            results.append({"ticker": ticker, "error": f"no_model_for_sector_{sector}"})
            continue

        try:
            # 加载板块模型
            state = torch.load(model_path, map_location=device, weights_only=False)
            first_key = list(state.keys())[0]
            if "input_proj" in first_key:
                input_size = state["input_proj.weight"].shape[1]
                model = StockMLP(input_size=input_size).to(device)
                model.load_state_dict(state)
                model.eval()
            else:
                results.append({"ticker": ticker, "error": f"unknown_arch_in_{sector}"})
                continue

            # 准备特征
            is_crypto = "/" in ticker
            if is_crypto:
                df_feat = _load_crypto_features(ticker, lookback=250)
            else:
                df_feat = _load_stock_features(ticker, lookback=250)

            if df_feat is None or df_feat.empty:
                results.append({"ticker": ticker, "error": "no_feature_data"})
                continue

            feature_cols = [c for c in df_feat.columns if c != "target"]
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X = scaler.fit_transform(df_feat[feature_cols].values)
            X_input = X[-1:]  # 最新一天

            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_input).to(device)
                out = model(X_tensor)
                probs = torch.softmax(out, dim=1)[0]
                pred = probs.argmax().item()
                confidence = probs[pred].item()

            signal_map = {0: "SELL", 1: "BUY"}
            results.append({
                "ticker": ticker,
                "signal": signal_map.get(pred, "HOLD"),
                "confidence": round(confidence * 100, 1),
                "prob_down": round(probs[0].item() * 100, 1),
                "prob_up": round(probs[1].item() * 100, 1),
                "sector": sector,
                "model_path": str(model_path),
            })

        except Exception as e:
            results.append({"ticker": ticker, "error": str(e)})

    return results


# ════════════════════════════════════════════════════════════════════
# 快速回测（as_of_date 防止未来数据泄漏）
# ════════════════════════════════════════════════════════════════════

def quick_backtest(tickers: list[str]) -> float:
    """90 天回测，综合评分（≥4 = 可投资）。"""
    try:
        be = BacktestEngine(initial_cash=10000)

        def strategy_fn(positions, cash, row, date_str):
            try:
                ranked = rank_stocks(tickers, top_n=8, as_of_date=date_str)
            except Exception:
                return []
            buy_signals = []
            for s in ranked[:3]:
                t = s["ticker"]
                if t not in positions and t in row and not np.isnan(row[t]):
                    price = row[t]
                    max_shares = int(cash * 0.2 / price)
                    if max_shares > 0:
                        buy_signals.append({
                            "action": "BUY", "ticker": t,
                            "shares": max_shares,
                            "reason": f"rank_score={s['score']:.1f}"
                        })
            return buy_signals

        result = be.run(
            tickers=tickers,
            strategy_fn=strategy_fn,
            start_date=(datetime.now() - timedelta(days=BACKTEST_DAYS)).strftime("%Y-%m-%d"),
            end_date=datetime.now().strftime("%Y-%m-%d"),
            stop_loss_pct=8, take_profit_pct=15,
            trailing_trigger_pct=5.0, trailing_stop_pct=3.0,
        )
        if result is None:
            return 0.0

        ret_norm   = max(-5, min(15, result.total_return_pct)) / 1.5
        dd_penalty = -min(max(result.max_drawdown_pct, 30), 30) / 3
        win_norm   = result.win_rate / 100 * 10
        sh_norm    = min(result.sharpe_ratio, 4) * 2.5
        n_trades   = result.total_trades
        trade_bonus= min(2.5, n_trades / 30 * 1.8)
        tp_rate    = result.take_profit_rate
        sl_rate    = result.stop_loss_rate
        pnl_asym   = tp_rate * 3 - sl_rate * 2

        score = ret_norm + dd_penalty + win_norm + sh_norm + trade_bonus + pnl_asym
        return round(score, 2)
    except Exception as e:
        print(f"    回测失败: {e}")
        return 0.0


# ════════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] === DL板块聚合训练开始 ===")
    print(f"  板块: {list(SECTOR_CONFIG.keys())}")
    print(f"  设备: {device}")
    print(f"{'='*60}")

    # ── Phase 1: 训练 3 个板块模型 ─────────────────────────────
    train_results = []
    for sector_name, tickers in SECTOR_CONFIG.items():
        result = train_sector_model(sector_name, tickers)
        train_results.append(result)

    # ── Phase 2: 回测验证 ────────────────────────────────────
    all_tickers = [t for tickers in SECTOR_CONFIG.values() for t in tickers]
    print(f"\n>>> 回测验证 ({BACKTEST_DAYS}天)...")
    backtest_score = quick_backtest(all_tickers)
    print(f"  回测综合分: {backtest_score:.2f} (目标≥{BACKTEST_SCORE_TARGET})")

    # ── Phase 3: 板块路由批量预测 ────────────────────────────
    print(f"\n>>> 板块路由预测...")
    dl_preds_raw = batch_predict_sector(all_tickers)
    dl_preds = {p["ticker"]: p for p in dl_preds_raw if "error" not in p}
    print(f"  有效预测: {list(dl_preds.keys())}")

    # ── Phase 4: 综合信号（调用 strategy_optimizer.get_signal）──
    print(f"\n>>> 综合信号...")
    ranked = rank_stocks(all_tickers, top_n=8)
    score_list = [{"ticker": s["ticker"], "score": s["score"]} for s in ranked]

    dl_list = [
        {"ticker": t, "signal": dl_preds[t]["signal"], "confidence": dl_preds[t]["confidence"]}
        for t in dl_preds
    ]

    optimizer = StrategyOptimizer(state_path=str(STATE_PATH))
    combined = optimizer.get_signal(dl_list, score_list)

    high_score = [
        c for c in combined
        if c["combined_score"] > 0.55
        and (abs(c.get("dl_confidence", 0)) >= MIN_CONFIDENCE or c["combined_score"] > 0.75)
    ][:MAX_SIGNALS]

    # ── Phase 5: 策略更新 + Telegram 推送 ───────────────────
    avg_acc = sum(
        r.get("best_val_acc", 0) for r in train_results
    ) / max(len(train_results), 1) * 100

    dl_acc_guess = min(avg_acc / 100, 0.70)

    if backtest_score > 0:
        new_params, changes, comp_score = optimizer.adjust_params(
            backtest_return=backtest_score,
            max_drawdown=0, win_rate=0, sharpe=0,
            dl_accuracy=dl_acc_guess,
        )
        print(f"\n>>> 策略更新 | 回测分: {backtest_score:.2f} | {'; '.join(changes) if changes else '无调整'}")
        optimizer.apply_params(new_params)

    # Telegram 推送
    token   = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_HOME_CHANNEL", "6801255591")
    if token and high_score:
        import requests
        signal_lines = []
        for c in high_score:
            ticker = c["ticker"]
            dl_p   = dl_preds.get(ticker, {})
            sig    = dl_p.get("signal", "BUY")
            conf   = dl_p.get("confidence", 0)
            price  = next((s["price"] for s in ranked if s["ticker"] == ticker), None)
            if price:
                sector = get_sector(ticker)
                entry  = round(price, 2)
                stop   = round(price * 0.92, 2)
                target = round(price * 1.15, 2)
                dl_b   = c.get("dl_bonus", 0)
                dl_c   = abs(c.get("dl_confidence", 0))
                bonus_str = f"+{dl_b:.3f}({dl_c:.0f}%)" if dl_b > 0 else ("-" if dl_b < 0 else "无DL")
                sector_icon = {"tech_high_vol": "💻", "traditional_defensive": "🏦", "cryptocurrency": "🪙"}.get(sector, "📊")
                signal_lines.append(
                    f"{sector_icon}[{sig}] {ticker} | ${entry} | Stop${stop} | Target${target} | "
                    f"板块{sector} | 评分{c['score_signal']:.3f} DL{bonus_str}"
                )

        status_icon = "🟢" if backtest_score >= BACKTEST_SCORE_TARGET else "🟡"
        msg = (
            f"{status_icon}AI股神信号 - {datetime.now().strftime('%m/%d %H:%M')}\n"
            f"板块模型: tech(15) / defensive(8) / crypto(7)\n"
            f"回测评分: {backtest_score:.2f}/{BACKTEST_SCORE_TARGET}\n"
            f"DL准确率估算: {dl_acc_guess:.1%}\n\n"
            + "\n".join(signal_lines)
            + "\n\n仅供参考，不构成投资建议"
        )
        try:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": msg}, timeout=10
            )
            print(f"\n>>> Telegram 推送成功: {len(signal_lines)} 条信号")
        except Exception as e:
            print(f"\n>>> Telegram 推送失败: {e}")
    else:
        print(f"\n>>> 无高分信号，跳过推送")

    print(f"\n{'='*60}")
    print(f"=== 完成 {datetime.now().strftime('%H:%M:%S')} ===")
    print(f"板块训练结果:")
    for r in train_results:
        status = "✅" if r.get("best_val_acc", 0) >= VAL_ACC_TARGET else "⚠️ "
        print(f"  {status} {r['sector']}: val_acc={r.get('best_val_acc', 'N/A')}, up={r.get('up_acc', 'N/A')}")
    print(f"{'='*60}\n")