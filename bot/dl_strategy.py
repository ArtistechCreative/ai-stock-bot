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

    # 均线
    df["ma5"] = close.rolling(5).mean()
    df["ma10"] = close.rolling(10).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma60"] = close.rolling(60).mean()

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

    # Bollinger Bands
    ma20_std = close.rolling(20).std()
    df["bb_upper"] = df["ma20"] + 2 * ma20_std
    df["bb_lower"] = df["ma20"] - 2 * ma20_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["ma20"]

    # 成交量指标
    df["vol_ma5"] = df["Volume"].rolling(5).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_ma5"]

    # 动量
    df["mom5"] = close / close.shift(5) - 1
    df["mom10"] = close / close.shift(10) - 1
    df["mom20"] = close / close.shift(20) - 1

    # 价格变化率
    df["pct_change_1d"] = close.pct_change(1)
    df["pct_change_5d"] = close.pct_change(5)

    # 相对位置（价格在均线间）
    df["price_ma20_ratio"] = close / df["ma20"]

    return df


def prepare_features(ticker: str, end_date: str = None, lookback: int = 250) -> pd.DataFrame:
    """
    下载历史数据 + 计算特征
    lookback: 多少天历史（默认250天 ≈ 1年）
    """
    end = end_date or datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=lookback * 2)).strftime("%Y-%m-%d")

    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty or len(df) < 60:
        return pd.DataFrame()

    df = compute_technical_indicators(df)

    # 打标签：次日涨跌 (1=涨, 0=跌)
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # 去掉 NaN 行
    df = df.dropna()

    # 特征列
    feature_cols = [
        "rsi", "macd", "macd_hist", "bb_width",
        "mom5", "mom10", "mom20",
        "pct_change_1d", "pct_change_5d",
        "price_ma20_ratio", "vol_ratio",
        "ma5", "ma10", "ma20", "ma60",
    ]

    return df[feature_cols + ["target"]]


# ======== 模型 ========

class StockMLP(nn.Module):
    """简单多层感知机"""
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


class StockLSTM(nn.Module):
    """简单 LSTM（用于时序）"""
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
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个 time step
        return self.fc(out)


class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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
        """准备训练数据"""
        df = prepare_features(self.ticker, end_date=end_date, lookback=lookback)
        if df.empty:
            return None, None, None

        feature_cols = [c for c in df.columns if c != "target"]

        X = df[feature_cols].values
        y = df["target"].values

        # 标准化
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y, feature_cols

    def train(
        self,
        end_date: str = None,
        lookback: int = 250,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
    ) -> dict:
        """训练模型"""
        X, y, feature_cols = self.prepare_data(end_date=end_date, lookback=lookback)
        if X is None:
            return {"error": "No data"}

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

        # 分割
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, shuffle=False
        )

        train_ds = StockDataset(X_tr, y_tr)
        val_ds = StockDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # 模型
        input_size = X_tr.shape[-1]
        if self.model_type == "LSTM":
            self.model = StockLSTM(input_size).to(self.device)
        else:
            self.model = StockMLP(input_size).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0
        best_state = None

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
                optimizer.step()
                train_loss += loss.item()

            # Validate
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
            avg_loss = train_loss / len(train_loader)

            if val_acc > best_acc:
                best_acc = val_acc
                best_state = self.model.state_dict().copy()

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, val_acc={val_acc:.3f}")

        # 恢复最佳模型
        if best_state:
            self.model.load_state_dict(best_state)
            torch.save(best_state, self.model_path)

        self.is_trained = True
        self.latest_train_date = end_date or datetime.now().strftime("%Y-%m-%d")

        return {
            "ticker": self.ticker,
            "best_val_acc": round(best_acc, 3),
            "train_samples": len(X_tr),
            "val_samples": len(X_val),
            "latest_train_date": self.latest_train_date,
            "model_path": str(self.model_path),
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
            # Try to load from disk
            if self.model_path.exists():
                try:
                    state = torch.load(self.model_path, map_location=self.device, weights_only=False)
                    input_size = state['net.0.weight'].shape[1]  # infer from first layer
                    if self.model_type == "LSTM":
                        self.model = StockLSTM(input_size).to(self.device)
                    else:
                        self.model = StockMLP(input_size).to(self.device)
                    self.model.load_state_dict(state)
                    self.is_trained = True
                except Exception:
                    return {"ticker": t, "signal": "NO_MODEL", "confidence": 0}
            else:
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