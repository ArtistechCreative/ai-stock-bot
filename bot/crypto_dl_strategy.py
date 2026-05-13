"""
加密货币深度学习策略
基于技术指标的 PyTorch MLP/LSTM 模型
支持做空方向预测（SHORT 多头市场反向下注）
使用 yfinance 加密货币数据或 CCXT 数据
"""
import os
import sys
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

sys.path.insert(0, os.path.dirname(__file__))
from crypto_data import CryptoData, PERP_INFO, DEFAULT_EXCHANGE

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ======== 特征工程（加密货币专用）=======

def compute_crypto_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 OHLCV 数据计算加密货币技术指标
    特别适合永续合约交易（做空方向）
    """
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    close = df["Close"] if "Close" in df.columns else df["close"]
    high  = df["High"]  if "High"  in df.columns else df["high"]
    low   = df["Low"]   if "Low"   in df.columns else df["low"]
    vol   = df["Volume"] if "Volume" in df.columns else df["volume"]

    # 均线
    df["ma5"]  = close.rolling(5).mean()
    df["ma10"] = close.rolling(10).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma60"] = close.rolling(60).mean()
    df["ma120"] = close.rolling(120).mean()

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # 布林带
    ma20_std = close.rolling(20).std()
    df["bb_upper"] = df["ma20"] + 2 * ma20_std
    df["bb_lower"] = df["ma20"] - 2 * ma20_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["ma20"]
    df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, 1e-10)

    # ATR（波动率）
    high_low = high - low
    high_close = abs(high - close.shift(1))
    low_close = abs(low - close.shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr"] / close * 100  # 波动率百分比

    # 成交量
    df["vol_ma5"] = vol.rolling(5).mean()
    df["vol_ratio"] = vol / df["vol_ma5"].replace(0, 1e-10)

    # 动量（关键：下跌时做空更容易赚钱）
    df["mom1"]  = close / close.shift(1) - 1
    df["mom5"]  = close / close.shift(5) - 1
    df["mom20"] = close / close.shift(20) - 1

    # 价格变化率
    df["pct_change_1d"] = close.pct_change(1)
    df["pct_change_5d"] = close.pct_change(5)

    # 相对位置
    df["price_ma20_ratio"] = close / df["ma20"]

    # 做空相关：最近的急剧下跌（适合做空回调）
    df["drawdown_5d"] = (close - close.rolling(5).max()) / close.rolling(5).max() * 100
    df["drawup_5d"]   = (close.rolling(5).min() - close) / close * 100  # 最近5天内从低点反弹

    # 趋势强度（适合顺势做空）
    df["trend_strength"] = abs(close - df["ma20"]) / df["atr"]

    return df


def prepare_crypto_features(
    symbol: str,
    exchange: str = DEFAULT_EXCHANGE,
    end_date: str = None,
    lookback: int = 250,
    timeframe: str = "1h",
) -> pd.DataFrame:
    """
    下载历史数据 + 计算特征
    lookback: 多少小时历史（默认250 ≈ 10天小时数据）
    """
    end = end_date or datetime.now().strftime("%Y-%m-%d")

    # 尝试 CCXT（永续合约数据更准确）
    cd = CryptoData(exchange=exchange)
    df = cd.fetch_ohlcv_dataframe(symbol, timeframe=timeframe, limit=lookback * 2)
    if df.empty or len(df) < 60:
        # Fallback: yfinance 加密货币数据
        yf_symbol = symbol.replace("/", "-") + "-USD"
        df = yf.download(yf_symbol, period="6mo", progress=False)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c.lower() for c in df.columns]

    df = compute_crypto_indicators(df)

    # 标签：次日涨跌 (1=涨, 0=跌)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # 去掉 NaN 行
    df = df.dropna()

    feature_cols = [
        "rsi", "macd", "macd_hist", "bb_width", "bb_position",
        "mom5", "mom20", "mom1",
        "pct_change_1d", "pct_change_5d",
        "price_ma20_ratio", "vol_ratio",
        "atr_pct", "drawdown_5d", "trend_strength",
        "ma5", "ma20", "ma60", "ma120",
    ]

    available = [c for c in feature_cols if c in df.columns]
    return df[available + ["target"]]


# ======== 模型（与 dl_strategy.py 相同架构）=======

class CryptoMLP(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # 0=跌, 1=涨
        )

    def forward(self, x):
        return self.net(x)


class CryptoLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class CryptoDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ======== 训练器 ========

class CryptoDLStrategy:
    """
    加密货币深度学习策略：
    - 训练：拿历史数据训练 MLP/LSTM
    - 推理：给当前特征 → 预测涨跌概率 → 生成做空/做多信号
    - 自动定期重训练
    """

    def __init__(
        self,
        symbol: str,
        model_type: str = "MLP",
        sequence_len: int = 10,
        model_dir: Path = None,
    ):
        self.symbol = symbol
        self.model_type = model_type
        self.sequence_len = sequence_len
        self.model_dir = model_dir or MODEL_DIR
        self.model_path = self.model_dir / f"crypto_{symbol.replace('/', '_')}_{model_type}.pt"

        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latest_train_date: str = None

    def prepare_data(self, end_date: str = None, lookback: int = 250, timeframe: str = "1h") -> tuple:
        """准备训练数据"""
        df = prepare_crypto_features(
            self.symbol, end_date=end_date, lookback=lookback, timeframe=timeframe
        )
        if df.empty:
            return None, None, None

        feature_cols = [c for c in df.columns if c != "target"]
        X = df[feature_cols].values
        y = df["target"].values
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y, feature_cols

    def train(
        self,
        end_date: str = None,
        lookback: int = 250,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        timeframe: str = "1h",
    ) -> dict:
        """训练模型"""
        X, y, feature_cols = self.prepare_data(end_date=end_date, lookback=lookback, timeframe=timeframe)
        if X is None:
            return {"error": "No data", "symbol": self.symbol}

        if self.model_type == "LSTM":
            X_seq, y_seq = [], []
            for i in range(self.sequence_len, len(X)):
                X_seq.append(X[i - self.sequence_len:i])
                y_seq.append(y[i])
            X_train = np.array(X_seq)
            y_train = np.array(y_seq)
        else:
            X_train, y_train = X, y

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, shuffle=False
        )

        train_ds = CryptoDataset(X_tr, y_tr)
        val_ds = CryptoDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        input_size = X_tr.shape[-1]
        if self.model_type == "LSTM":
            self.model = CryptoLSTM(input_size).to(self.device)
        else:
            self.model = CryptoMLP(input_size).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        best_acc = 0
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                out = self.model(X_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    out = self.model(X_batch)
                    preds = out.argmax(dim=1)
                    correct += (preds == y_batch).sum().item()
                    total += len(y_batch)

            val_acc = correct / total if total > 0 else 0
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = self.model.state_dict().copy()

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: loss={train_loss/len(train_loader):.4f}, val_acc={val_acc:.3f}")

        if best_state:
            self.model.load_state_dict(best_state)
            torch.save(best_state, self.model_path)

        self.is_trained = True
        self.latest_train_date = end_date or datetime.now().strftime("%Y-%m-%d")

        return {
            "symbol": self.symbol,
            "best_val_acc": round(best_acc, 3),
            "train_samples": len(X_tr),
            "val_samples": len(X_val),
            "latest_train_date": self.latest_train_date,
            "model_path": str(self.model_path),
        }

    def predict(self, features: dict = None, exchange: str = DEFAULT_EXCHANGE) -> dict:
        """
        用最新市场数据预测涨跌
        返回做多/做空信号及信心度
        """
        if features is None:
            df = prepare_crypto_features(self.symbol, lookback=250, exchange=exchange)
            if df.empty:
                return {"symbol": self.symbol, "signal": "NO_DATA", "confidence": 0}
            feature_cols = [c for c in df.columns if c != "target"]
            X = df[feature_cols].values
            X_scaled = self.scaler.fit_transform(X)
            X_input = X_scaled[-1:]
        else:
            feat_df = pd.DataFrame([features])
            X_input = self.scaler.transform(feat_df.values)

        if self.model is None or not self.is_trained:
            if self.model_path.exists():
                try:
                    state = torch.load(self.model_path, map_location=self.device, weights_only=False)
                    input_size = state["net.0.weight"].shape[1] if self.model_type == "MLP" else 32
                    if self.model_type == "LSTM":
                        self.model = CryptoLSTM(input_size).to(self.device)
                    else:
                        self.model = CryptoMLP(input_size).to(self.device)
                    self.model.load_state_dict(state)
                    self.is_trained = True
                except Exception:
                    return {"symbol": self.symbol, "signal": "NO_MODEL", "confidence": 0}
            else:
                return {"symbol": self.symbol, "signal": "NO_MODEL", "confidence": 0}

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_input).to(self.device)
            out = self.model(X_tensor)
            probs = torch.softmax(out, dim=1)[0]
            pred = probs.argmax().item()
            confidence = probs[pred].item()

        # 信号映射
        if pred == 1:
            signal = "BUY"   # 预测上涨 → 做多
        else:
            signal = "SHORT" # 预测下跌 → 做空

        perp_info = PERP_INFO.get(self.symbol, {})
        leverage = perp_info.get("leverage", 50)

        return {
            "symbol": self.symbol,
            "signal": signal,
            "direction": "LONG" if signal == "BUY" else "SHORT",
            "confidence": round(confidence * 100, 1),
            "prob_down": round(probs[0].item() * 100, 1),
            "prob_up": round(probs[1].item() * 100, 1),
            "leverage": leverage,
            "model_type": self.model_type,
            "trained": self.latest_train_date,
        }

    def auto_train(
        self,
        symbols: list[str],
        days_since_last: int = 7,
        timeframe: str = "1h",
    ) -> list[dict]:
        """检查是否需要重训练并执行"""
        results = []
        for symbol in symbols:
            try:
                dl = CryptoDLStrategy(symbol, model_type=self.model_type, sequence_len=self.sequence_len)
                last_date = dl.latest_train_date
                needs_train = True
                if last_date:
                    days_passed = (datetime.now() - datetime.strptime(last_date, "%Y-%m-%d")).days
                    needs_train = days_passed >= days_since_last
                    print(f"  ⏭️ {symbol}: 距上次训练 {days_passed} 天 {'→ 跳过' if not needs_train else ''}")

                if needs_train:
                    print(f"  🔄 训练 {symbol}...")
                    result = dl.train(timeframe=timeframe)
                    results.append(result)
            except Exception as e:
                print(f"  [!] {symbol} 训练失败: {e}")
                results.append({"symbol": symbol, "error": str(e)})
        return results


# ======== 批量预测 ========

def batch_crypto_predict(
    symbols: list[str],
    exchange: str = DEFAULT_EXCHANGE,
    model_type: str = "MLP",
) -> list[dict]:
    """对多个加密货币批量预测"""
    results = []
    for symbol in symbols:
        try:
            model = CryptoDLStrategy(symbol, model_type=model_type)
            pred = model.predict(exchange=exchange)
            results.append(pred)
        except Exception as e:
            results.append({"symbol": symbol, "error": str(e)})
    return results


# ======== CLI ========

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(os.path.expanduser("~/.hermes/.env"))

    syms = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    exchange = DEFAULT_EXCHANGE

    print(f"📊 加密货币 DL 策略（{exchange.upper()}）")
    print(f"   模型: MLP | 币种: {len(syms)} 个\n")

    # 训练
    print("🔄 训练模型...")
    for symbol in syms:
        dl = CryptoDLStrategy(symbol, model_type="MLP")
        result = dl.train(timeframe="1h", epochs=30)
        if "error" not in result:
            print(f"  ✅ {symbol}: val_acc={result['best_val_acc']}, samples={result['train_samples']}")
        else:
            print(f"  ❌ {symbol}: {result['error']}")

    # 预测
    print("\n🔮 批量预测：")
    preds = batch_crypto_predict(syms, exchange=exchange)
    for p in preds:
        if "error" in p:
            print(f"  ❌ {p['symbol']}: {p['error']}")
        else:
            emoji = "📈" if p["signal"] == "BUY" else "📉"
            print(f"  {emoji} {p['signal']} {p['symbol']} ({p['confidence']}% 信心) "
                  f"| 涨{p['prob_up']}% / 跌{p['prob_down']}% | 杠杆:{p['leverage']}x")