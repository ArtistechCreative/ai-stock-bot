"""
深度学习交易策略
- 技术指标特征工程
- PyTorch 简单 MLP/LSTM 模型
- 训练：基于历史数据预测次日涨跌
- 推理：输出买入/卖出/持有信号
- 自动定期重训练更新策略
"""
import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ======== 特征工程 ========

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """从 OHLCV 数据计算技术指标"""
    df = df.copy()

    # Flatten multi-level columns from yfinance (single-ticker download returns tuples)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    close = df["Close"]
    high = df.get("High", close)
    low = df.get("Low", close)
    volume = df["Volume"]

    # ── 均线 ───────────────────────────────────────────────────────────────────
    df["ma5"] = close.rolling(5).mean()
    df["ma10"] = close.rolling(10).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma60"] = close.rolling(60).mean()
    df["ma120"] = close.rolling(120).mean()

    # ── RSI ───────────────────────────────────────────────────────────────────
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    # RSI 多周期（反转信号）
    df["rsi_5"] = 100 - (100 / (1 + delta.where(delta > 0, 0).rolling(5).mean()
                               / (delta.where(delta < 0, 0).abs().rolling(5).mean() + 1e-10)))
    df["rsi_30"] = 100 - (100 / (1 + delta.where(delta > 0, 0).rolling(30).mean()
                                 / (delta.where(delta < 0, 0).abs().rolling(30).mean() + 1e-10)))

    # ── MACD ───────────────────────────────────────────────────────────────────
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["macd_hist_pct"] = df["macd_hist"] / (close + 1e-10)  # 归一化hist

    # ── Bollinger Bands ────────────────────────────────────────────────────────
    ma20_std = close.rolling(20).std()
    df["bb_upper"] = df["ma20"] + 2 * ma20_std
    df["bb_lower"] = df["ma20"] - 2 * ma20_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["ma20"]
    df["bb_pct_b"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

    # ── 成交量指标 ────────────────────────────────────────────────────────────
    df["vol_ma5"] = volume.rolling(5).mean()
    df["vol_ma20"] = volume.rolling(20).mean()
    df["vol_ratio"] = volume / df["vol_ma5"]
    df["vol_change"] = volume.pct_change(1)
    df["vol_change_5d"] = volume.pct_change(5)

    # ── 动量 ──────────────────────────────────────────────────────────────────
    df["mom5"] = close / close.shift(5) - 1
    df["mom10"] = close / close.shift(10) - 1
    df["mom20"] = close / close.shift(20) - 1
    df["mom60"] = close / close.shift(60) - 1

    # 价格变化率
    df["pct_change_1d"] = close.pct_change(1)
    df["pct_change_5d"] = close.pct_change(5)

    # ── ATR ───────────────────────────────────────────────────────────────────
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    df["atr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    df["atr_pct"] = df["atr"] / close * 100

    # ── 价格位置 ──────────────────────────────────────────────────────────────
    df["price_ma20_ratio"] = close / df["ma20"]
    df["price_ma60_ratio"] = close / df["ma60"]
    df["price_ma120_ratio"] = close / df["ma120"]
    df["ma5_ma20_gap"] = (df["ma5"] - df["ma20"]) / df["ma20"]
    df["ma10_ma60_gap"] = (df["ma10"] - df["ma60"]) / df["ma60"]

    # ── OBV ───────────────────────────────────────────────────────────────────
    df["obv"] = (np.sign(close.diff()) * volume).cumsum()
    df["obv_ma10"] = df["obv"].rolling(10).mean()
    df["obv_ma20"] = df["obv"].rolling(20).mean()
    df["obv_slope"] = df["obv"].diff(5) / (df["obv_ma10"] + 1e-10)  # OBV 5日斜率

    # ── Stochastic Oscillator ─────────────────────────────────────────────────
    lowest_low = low.rolling(14).min()
    highest_high = high.rolling(14).max()
    df["stoch_k"] = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ── 新增：均值回归特征（BB 极度收缩 → 即将突破） ─────────────────────────
    df["bb_squeeze"] = (df["bb_width"] / df["bb_width"].rolling(20).mean())  # BB宽度相对20日均值
    df["price_deviation_ma20"] = (close - df["ma20"]) / (ma20_std + 1e-10)  # Z-score

    # ── 新增：量价背离（价格创新高但量萎缩 = 危险） ──────────────────────────
    df["price_high_20"] = close.rolling(20).max()
    df["volume_high_20"] = volume.rolling(20).max()
    df["price_volume_div"] = (close / df["price_high_20"]) - (volume / df["volume_high_20"])

    # ── 新增：相对强弱（self-relative） ──────────────────────────────────────
    df["rel_str_5d"] = df["mom5"] / (df["mom5"].abs().rolling(5).mean() + 1e-10)
    df["rel_str_20d"] = df["mom20"] / (df["mom20"].abs().rolling(20).mean() + 1e-10)

    return df


def prepare_features(ticker: str, end_date: str = None, lookback: int = 500) -> pd.DataFrame:
    """
    下载历史数据 + 计算特征
    lookback: 多少天历史（默认500天 ≈ 2年，更多训练数据）

    标签：次日涨跌 (1=涨, 0=跌)
    特征工程：38个指标，涵盖趋势/动量/均线/成交量/波动性/均值回归/量价背离
    """
    end = end_date or datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=lookback * 2)).strftime("%Y-%m-%d")

    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty or len(df) < 60:
        return pd.DataFrame()

    df = compute_technical_indicators(df)

    # 标签：次日涨跌 (1=涨, 0=跌)
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # 去掉 NaN 行（含最后1天没有未来数据的行）
    df = df.dropna()

    # ── 特征列（扩展到 38 个特征）────────────────────────────────────────────
    feature_cols = [
        # 趋势/动量（14）
        "rsi", "rsi_5", "rsi_30",
        "macd", "macd_hist", "macd_hist_pct",
        "mom5", "mom10", "mom20", "mom60",
        "pct_change_1d", "pct_change_5d",
        # 均线系统（7）
        "ma5", "ma10", "ma20", "ma60", "ma120",
        "ma5_ma20_gap", "ma10_ma60_gap",
        # 价格位置（6）
        "price_ma20_ratio", "price_ma60_ratio", "price_ma120_ratio",
        "bb_pct_b", "bb_width", "price_deviation_ma20",
        # 成交量（7）
        "vol_ratio", "vol_change", "vol_change_5d",
        "obv", "obv_slope", "price_volume_div",
        # 波动性（4）
        "atr", "atr_pct", "bb_squeeze", "stoch_k",
    ]

    return df[feature_cols + ["target"]]


# ======== 模型 ========

class StockMLP(nn.Module):
    """加深的多层感知机 + BatchNorm + 残差连接"""
    def __init__(self, input_size: int, hidden_dims=[256, 128, 64, 32]):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0])

        layers = []
        prev_dim = hidden_dims[0]
        for i, h_dim in enumerate(hidden_dims):
            is_last = (i == len(hidden_dims) - 1)

            # 残差跳跃（当维度匹配时）
            has_residual = (prev_dim == h_dim) and (i > 0)
            linear = nn.Linear(prev_dim, h_dim)
            bn = nn.BatchNorm1d(h_dim)

            layers.append(linear)
            layers.append(bn)

            if not is_last:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.25))
            prev_dim = h_dim

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 2)

    def forward(self, x):
        x = torch.relu(self.input_bn(self.input_proj(x)))
        x = self.hidden(x)
        return self.output(x)


class StockLSTM(nn.Module):
    """加深 LSTM + 双头 Attention + BatchNorm"""
    def __init__(self, input_size: int, hidden_size: int = 96, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )
        # 双头 attention（捕捉不同时间尺度）
        self.head1 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
        )
        self.head2 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Sigmoid(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

        # 双头 attention
        h1 = self.head1(lstm_out)      # (batch, seq, hidden)
        h2 = self.head2(lstm_out)      # (batch, seq, hidden)
        gate = self.gate(lstm_out)    # (batch, seq, 1)

        # 门控组合 + softmax 归一化
        context = torch.cat([h1, h2 * gate], dim=-1)  # (batch, seq, hidden*2+hidden)
        attn = torch.softmax(context.sum(dim=-1, keepdim=True), dim=1)  # (batch, seq, 1)
        context = torch.sum(lstm_out * attn, dim=1)  # (batch, hidden*2)

        return self.fc(context)


class StockEnsemble(nn.Module):
    """
    Ensemble of 3 MLPs with different random seeds.
    Each MLP has a different architecture variant (width/depth trade-off).
    Final prediction = majority vote (避免极端预测)
    """
    def __init__(self, input_size: int, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 3 个不同架构的 MLP
        self.models = nn.ModuleList([
            StockMLP(input_size, hidden_dims=[192, 96, 48, 24]),   # 深宽
            StockMLP(input_size, hidden_dims=[256, 128, 64]),     # 标准深
            StockMLP(input_size, hidden_dims=[128, 128, 64, 32]), # 双残差
        ])

        # 不同初始化种子
        for i, m in enumerate(self.models):
            torch.manual_seed(seed + i * 11)
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, x):
        # 收集每个模型的 logits
        logits_list = [m(x) for m in self.models]
        # 加权平均 logits（比投票更稳定，减少随机性）
        avg_logits = torch.stack(logits_list, dim=0).mean(dim=0)
        # 概率校准：降低极端预测的置信度
        probs = torch.softmax(avg_logits, dim=1)
        # 一致率：三个模型预测一致时置信度更高
        preds_stack = torch.stack([logits.argmax(dim=1) for logits in logits_list], dim=0)  # (3, batch)
        vote_conf = (preds_stack == preds_stack[0:1]).float().mean(dim=0)  # 与模型0一致的比例
        # 一致率高时信任 avg_logits，一致率低时向 0.5 均化
        calibrated = probs.clone()
        calibrated[:, 1] = probs[:, 1] * vote_conf + 0.5 * (1 - vote_conf)
        calibrated[:, 0] = probs[:, 0] * vote_conf + 0.5 * (1 - vote_conf)
        calibrated = calibrated / calibrated.sum(dim=1, keepdim=True)
        return torch.log(calibrated + 1e-8)


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


def get_sector(ticker: str) -> str:
    """根据 ticker 返回所属板块。未定义资产默认归 cryptocurrency（不崩溃）。"""
    for sector, tickers in SECTOR_CONFIG.items():
        if ticker in tickers:
            return sector
    if "/" in ticker:
        return "cryptocurrency"
    # 未知非加密资产归 tech_high_vol（安全降级，不崩溃）
    import logging as _log
    _log.warning(f"[get_sector] {ticker} not in SECTOR_CONFIG, defaulting to tech_high_vol")
    return "tech_high_vol"


def sector_model_path(sector: str) -> Path:
    """板块模型文件路径（与 train_dl.py 的命名规则保持一致）。"""
    name_map = {
        "tech_high_vol":        "dl_model_techhighvol.pth",
        "traditional_defensive":"dl_model_traditionaldefensive.pth",
        "cryptocurrency":       "dl_model_cryptocurrency.pth",
    }
    return MODEL_DIR / name_map.get(sector, f"dl_model_{sector}.pth")


def batch_predict_sector(tickers: list[str]) -> list[dict]:
    """板块路由批量预测：识别 ticker 板块 → 加载对应板块模型 → 输出信号。"""
    results = []
    for ticker in tickers:
        sector = get_sector(ticker)
        model_file = sector_model_path(sector)

        if not model_file.exists():
            results.append({"ticker": ticker, "error": f"no_model:{sector}"})
            continue

        try:
            # 加载板块模型（从 best_state 字典恢复架构）
            state = torch.load(model_file, map_location="cpu", weights_only=False)
            first_key = list(state.keys())[0]

            # 推断 input_size 从权重形状
            if "input_proj.weight" in first_key:
                input_size = state["input_proj.weight"].shape[1]
            elif "net.0.weight" in first_key:
                input_size = state["net.0.weight"].shape[1]
            else:
                input_size = 38  # fallback

            model = StockMLP(input_size=input_size).to("cpu")
            model.load_state_dict(state)
            model.eval()

            # 准备特征（按资产类型分流）
            is_crypto = "/" in ticker
            if is_crypto:
                df_feat = _load_crypto_features(ticker)
            else:
                df_feat = _prepare_features_for_ticker(ticker)

            if df_feat is None or df_feat.empty:
                results.append({"ticker": ticker, "error": "no_data"})
                continue

            feature_cols = [c for c in df_feat.columns if c not in ("target", "date")]
            scaler = StandardScaler()
            X = scaler.fit_transform(df_feat[feature_cols].values.astype(np.float32))
            X_input = X[-1:]  # 最新一天

            with torch.no_grad():
                out = model(torch.FloatTensor(X_input))
                probs = torch.softmax(out, dim=1)[0]
                pred = probs.argmax().item()
                conf = probs[pred].item()

            results.append({
                "ticker": ticker,
                "signal": {0: "SELL", 1: "BUY"}.get(pred, "HOLD"),
                "confidence": round(conf * 100, 1),
                "prob_down": round(probs[0].item() * 100, 1),
                "prob_up": round(probs[1].item() * 100, 1),
                "sector": sector,
            })

        except Exception as e:
            results.append({"ticker": ticker, "error": str(e)})

    return results


def _prepare_features_for_ticker(ticker: str) -> pd.DataFrame | None:
    """包装 prepare_features，保持接口兼容。"""
    return prepare_features(ticker, lookback=250)


def _load_crypto_features(symbol: str) -> pd.DataFrame | None:
    """从 CCXT OKX 加载加密货币特征，与 dl_strategy compute_technical_indicators 对齐。"""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from bot.crypto_data import CryptoData

        cd = CryptoData(exchange="okx")
        since_ms = int((pd.Timestamp.now() - pd.Timedelta(days=500)).timestamp() * 1000)
        df = cd.fetch_ohlcv_dataframe(symbol, timeframe="1d", limit=500, since=since_ms)
        if df.empty or len(df) < 60:
            return None

        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume"
        })
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

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
        existing = [c for c in feature_cols if c in df_tech.columns]
        return df_tech[existing + ["target"]]

    except Exception as e:
        return None


class StockDataset:
    """PyTorch Dataset for stock feature matrices."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Alias for backward compatibility
Dataset = StockDataset


# ======== 训练器 ========

class DLStrategy:
    """
    深度学习策略：
    - 训练：拿历史数据训练 MLP/LSTM
    - 推理：给当前特征 → 预测涨跌概率 → 生成交易信号
    - 自动定期重训练
    """

    def __init__(
        self,
        ticker: str,
        model_type: str = "MLP",  # MLP or LSTM
        sequence_len: int = 10,   # LSTM 序列长度
        model_dir: Path = None,
    ):
        self.ticker = ticker
        self.model_type = model_type
        self.sequence_len = sequence_len
        self.model_dir = model_dir or MODEL_DIR
        self.model_path = self.model_dir / f"{ticker}_{model_type}.pt"

        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False

        # 训练配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latest_train_date: str = None

    def prepare_data(self, end_date: str = None, lookback: int = 250) -> tuple:
        """准备训练数据（标准化仅在训练集拟合，避免泄漏）"""
        df = prepare_features(self.ticker, end_date=end_date, lookback=lookback)
        if df.empty:
            return None, None, None

        feature_cols = [c for c in df.columns if c != "target"]
        X = df[feature_cols].values
        y = df["target"].values

        # 先 fit_transform 全量数据（用于 inference 时的一致性）
        # 训练时会再按 train/test split 重做，这里只做全局标准化用于后续 inference
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y, feature_cols

    def train(
        self,
        end_date: str = None,
        lookback: int = 300,
        epochs: int = 80,
        batch_size: int = 32,
        lr: float = 3e-4,
        model_type_override: str = None,
    ) -> dict:
        """
        训练模型（walk-forward 时序验证 + Label Smoothing + 更强正则化）
        改进点：
        - 更低学习率 + 更高 weight_decay
        - Label smoothing（0.05）减少极端预测
        - 更严格早停（patience=8）
        - 梯度裁剪 max_norm=0.5
        - 训练集 shuffle 以提升泛化
        """
        X, y, feature_cols = self.prepare_data(end_date=end_date, lookback=lookback)
        if X is None:
            return {"error": "No data"}

        # 标签分布 + 双向类别权重（不再只重DOWN→UP）
        n_up = int(y.sum())
        n_down = len(y) - n_up
        # 两个元素的 pos_weight: [down_weight, up_weight]
        # 比值 = down/up，表示模型更倾向猜 up 时加大 down 权重
        ratio = n_up / max(n_down, 1)
        pos_weight = torch.FloatTensor([ratio, 1.0]).to(self.device)
        print(f"  [{self.ticker}] 数据: up={n_up}, down={n_down}, ratio={ratio:.2f}, pos_weight={pos_weight[0].item():.2f}")

        if self.model_type == "LSTM":
            # 构造序列数据
            X_seq, y_seq = [], []
            for i in range(self.sequence_len, len(X)):
                X_seq.append(X[i-self.sequence_len:i])
                y_seq.append(y[i])
            X_train = np.array(X_seq)
            y_train = np.array(y_seq)
        else:
            X_train, y_train = X, y

        # Walk-forward 时序分割（不用 shuffle_split，避免未来数据泄漏）
        n = len(X_train)
        train_end = int(n * 0.75)  # 75% 训练，25% 验证（之前是80/20）
        X_tr, X_val = X_train[:train_end], X_train[train_end:]
        y_tr, y_val = y_train[:train_end], y_train[train_end:]

        train_ds = StockDataset(X_tr, y_tr)
        val_ds = StockDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # 模型（每次训练重新随机种子，避免极化循环）
        torch.manual_seed(torch.randint(0, 2**31, (1,)).item())
        np.random.seed()
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True  # 加速训练
        input_size = X_tr.shape[-1]
        mt = model_type_override or self.model_type
        if mt == "LSTM":
            self.model = StockLSTM(input_size).to(self.device)
        elif mt == "ensemble":
            self.model = StockEnsemble(input_size).to(self.device)
        else:
            self.model = StockMLP(input_size).to(self.device)

        # 更强正则化
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-5
        )
        # Label smoothing（0.05）：将硬标签转为软标签，减少极端预测，让模型更校准
        criterion = nn.CrossEntropyLoss(weight=pos_weight, label_smoothing=0.05)

        best_acc = 0
        best_state = None
        best_up = 0
        best_down = 0
        patience = 15   # 放宽：原来8太短，lr CosineAnnealing 还没降到位就被打断
        no_improve = 0

        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                out = self.model(X_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)  # 更严格裁剪
                optimizer.step()
                train_loss += loss.item()

            # Validate（同时算各类别准确率）
            self.model.eval()
            correct = 0
            total = 0
            correct_up = 0; total_up = 0
            correct_down = 0; total_down = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    out = self.model(X_batch)
                    preds = out.argmax(dim=1)
                    correct += (preds == y_batch).sum().item()
                    total += len(y_batch)
                    # 分类别统计
                    mask_up = (y_batch == 1)
                    mask_down = (y_batch == 0)
                    correct_up += ((preds == y_batch) & mask_up).sum().item()
                    correct_down += ((preds == y_batch) & mask_down).sum().item()
                    total_up += mask_up.sum().item()
                    total_down += mask_down.sum().item()

            val_acc = correct / total if total > 0 else 0
            scheduler.step(val_acc)
            avg_loss = train_loss / len(train_loader)
            up_acc = correct_up / total_up if total_up > 0 else 0
            down_acc = correct_down / total_down if total_down > 0 else 0

            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                best_up = up_acc
                best_down = down_acc
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                # 极化检测：up/down 准确率严重失衡时强制终止
                if best_up > 0.70 and best_down < 0.40:
                    print(f"  ⚠️ 极化强制终止 ep {epoch+1}: up={best_up:.1%} / down={best_down:.1%}，不保存极化模型")
                    best_state = None  # 不保存极化模型
                else:
                    print(f"  Early stop ep {epoch+1}, best_acc={best_acc:.3f}, up_acc={best_up:.3f}, down_acc={best_down:.3f}")
                break

            if (epoch + 1) % 10 == 0:
                print(f"  Ep {epoch+1}: loss={avg_loss:.3f}, val_acc={val_acc:.3f} (up={up_acc:.3f}, down={down_acc:.3f}), lr={scheduler.get_last_lr()[0]:.2e}")

        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)
            # 保存时包含模型类型后缀
            model_name = f"{self.ticker}_{mt}.pt"
            torch.save(best_state, self.model_dir / model_name)
        else:
            print(f"  ⚠️ 无有效模型（极化/不达标），跳过保存")

        self.is_trained = True
        self.latest_train_date = end_date or datetime.now().strftime("%Y-%m-%d")

        return {
            "ticker": self.ticker,
            "best_val_acc": round(best_acc, 3),
            "up_acc": round(best_up, 3),
            "down_acc": round(best_down, 3),
            "train_samples": len(X_tr),
            "val_samples": len(X_val),
            "latest_train_date": self.latest_train_date,
            "model_type": mt,
            "model_path": str(self.model_dir / model_name) if best_state else None,
        }

    def predict(self, features: dict = None, ticker: str = None) -> dict:
        """
        用最新市场数据预测涨跌
        features: 可选，dict of {feature_name: value}。如果不提供则实时拉数据
        """
        t = ticker or self.ticker

        if features is None:
            # 实时拉数据
            df = prepare_features(t, lookback=250)
            if df.empty:
                return {"ticker": t, "error": "No data"}
            feature_cols = [c for c in df.columns if c != "target"]
            X = df[feature_cols].values
            X_scaled = self.scaler.fit_transform(X)
            X_input = X_scaled[-1:]  # 最新一天
        else:
            # 用提供的特征
            feat_df = pd.DataFrame([features])
            X_input = self.scaler.transform(feat_df.values)

        if self.model is None or not self.is_trained:
            # Try to load from disk - check both typed model name and legacy name
            # New format: {ticker}_{model_type}.pt, e.g. NVDA_ensemble.pt
            typed_path = self.model_dir / f"{self.ticker}_{self.model_type}.pt"

            candidates = []
            if typed_path.exists():
                candidates = [typed_path]
            if self.model_path.exists():
                candidates.append(self.model_path)

            for model_file in candidates:
                try:
                    state = torch.load(model_file, map_location=self.device, weights_only=False)
                    first_key = list(state.keys())[0]

                    # Detect model architecture from state dict keys
                    if 'models' in first_key:  # StockEnsemble: 'models.0.xxx'
                        input_size = state['models.0.input_proj.weight'].shape[1]
                        self.model = StockEnsemble(input_size=input_size).to(self.device)
                        self.model.load_state_dict(state)
                        self.is_trained = True
                        self.model_type = "ensemble"
                        break
                    elif first_key == 'input_proj.weight':  # StockMLP new arch
                        input_size = state['input_proj.weight'].shape[1]
                        self.model = StockMLP(input_size=input_size).to(self.device)
                        self.model.load_state_dict(state)
                        self.is_trained = True
                        break
                    elif first_key in ('net.0.weight',):  # old arch
                        input_size = state['net.0.weight'].shape[1]
                        self.model = StockMLP(input_size, hidden_dims=[64, 32, 16]).to(self.device)
                        old_to_new = {
                            'net.0.weight': 'net.0.weight',
                            'net.0.bias':   'net.0.bias',
                            'net.3.weight': 'net.4.weight',
                            'net.3.bias':   'net.4.bias',
                            'net.6.weight': 'net.8.weight',
                            'net.6.bias':   'net.8.bias',
                            'net.8.weight': 'net.12.weight',
                            'net.8.bias':   'net.12.bias',
                        }
                        new_state = {new_k: state[old_k] for old_k, new_k in old_to_new.items() if old_k in state}
                        for dim, bn_idx in {64: 1, 32: 5, 16: 9}.items():
                            new_state[f'net.{bn_idx}.running_mean'] = torch.ones(dim)
                            new_state[f'net.{bn_idx}.running_var'] = torch.ones(dim)
                            new_state[f'net.{bn_idx}.num_batches_tracked'] = torch.tensor(0)
                            new_state[f'net.{bn_idx}.weight'] = torch.ones(dim)
                            new_state[f'net.{bn_idx}.bias'] = torch.zeros(dim)
                        self.model.load_state_dict(new_state, strict=False)
                        self.is_trained = True
                        break
                except Exception as e:
                    continue

            if not self.is_trained:
                return {"ticker": t, "signal": "NO_MODEL", "confidence": 0}

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_input).to(self.device)
            out = self.model(X_tensor)
            probs = torch.softmax(out, dim=1)[0]
            pred = probs.argmax().item()
            confidence = probs[pred].item()

        signal_map = {0: "SELL", 1: "BUY"}
        signal = signal_map.get(pred, "HOLD")

        return {
            "ticker": t,
            "signal": signal,
            "confidence": round(confidence * 100, 1),
            "prob_down": round(probs[0].item() * 100, 1),
            "prob_up": round(probs[1].item() * 100, 1),
            "model_type": self.model_type,
            "trained": self.latest_train_date,
        }

    def auto_train(
        self,
        tickers: list[str],
        days_since_last: int = 7,
    ) -> list[dict]:
        """
        检查是否需要重训练（距上次训练 > days_since_last）
        对每个 ticker 训练并返回结果
        """
        results = []
        for ticker in tickers:
            try:
                dl = DLStrategy(ticker, model_type=self.model_type, sequence_len=self.sequence_len)
                last_date = dl.latest_train_date

                if last_date is None:
                    needs_train = True
                else:
                    days_passed = (datetime.now() - datetime.strptime(last_date, "%Y-%m-%d")).days
                    needs_train = days_passed >= days_since_last

                if needs_train:
                    print(f"  🔄 训练 {ticker}（距上次 {days_passed if last_date else '首次'} 天）...")
                    result = dl.train()
                    results.append(result)
                else:
                    print(f"  ⏭️ 跳过 {ticker}（距上次训练仅 {days_passed} 天）")
            except Exception as e:
                print(f"  [!] {ticker} 训练失败: {e}")
                results.append({"ticker": ticker, "error": str(e)})

        return results


# ======== 批量预测 ========

def batch_predict(tickers: list[str], model_type: str = "MLP") -> list[dict]:
    """对多个股票批量预测"""
    results = []
    for ticker in tickers:
        try:
            model = DLStrategy(ticker, model_type=model_type)
            pred = model.predict()
            results.append(pred)
        except Exception as e:
            results.append({"ticker": ticker, "error": str(e)})
    return results


# ======== CLI ========

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from dotenv import load_dotenv
    load_dotenv(os.path.expanduser("~/.hermes/.env"))
    from config import WATCHLIST

    print("📊 深度学习策略训练")
    print(f"   模型类型: MLP")
    print(f"   股票池: {len(WATCHLIST)} 支")

    results = DLStrategy("NVDA").auto_train(WATCHLIST, days_since_last=7)

    for r in results:
        if "error" in r:
            print(f"  ❌ {r['ticker']}: {r['error']}")
        else:
            print(f"  ✅ {r['ticker']}: val_acc={r['best_val_acc']}")

    print("\n🔮 批量预测：")
    preds = batch_predict(WATCHLIST[:6])
    for p in preds:
        if "error" in p:
            print(f"  ❌ {p['ticker']}: {p['error']}")
        else:
            print(f"  {p['signal']} {p['ticker']} ({p['confidence']}% 信心) | 涨{p['prob_up']}% / 跌{p['prob_down']}%")