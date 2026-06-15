"""
Regime Agent — 市场状态识别 + Meta Agent 降级链路
=====================================================
基于 Wayland Zhang《AI量化交易从0到1》第 12-13 课框架

四状态市场模型: TRENDING / MEAN_REVERTING / CRISIS / UNCERTAIN
四级降级链路: Normal → Caution → Defensive → Safe

用法:
    from regime_agent import RegimeAgent
    agent = RegimeAgent()
    state = agent.analyze()
    print(f"市场状态: {state.regime}, 降级: L{state.degradation_level}")
"""

import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import json, os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ── 数据类型 ─────────────────────────────────────────────────────

@dataclass
class RegimeState:
    """当前市场状态"""
    regime: str = "UNCERTAIN"      # TRENDING / MEAN_REVERTING / CRISIS / UNCERTAIN
    degradation_level: int = 0     # 0=正常 1=谨慎 2=防御 3=安全
    confidence: float = 0.0        # 0-100
    timestamp: str = ""
    
    # 关键指标
    adx: float = 0.0               # 趋势强度
    volatility: float = 0.0        # 年化波动率
    vix: float = 0.0               # VIX 恐慌指数
    spx_return_5d: float = 0.0     # SPX 5日收益率
    spx_return_20d: float = 0.0    # SPX 20日收益率
    spx_drawdown: float = 0.0      # SPX 从高点回撤%
    correlation_spike: bool = False  # 相关性飙升标志
    
    # 推荐参数（供下游 Agent 使用）
    suggested_max_position_pct: float = 0.20   # 建议单笔上限
    suggested_slippage_buffer: float = 0.002    # 建议滑点缓冲
    trade_cooling: bool = False                 # 是否冷却（暂停交易）
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "regime": self.regime,
            "degradation_level": self.degradation_level,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "adx": round(self.adx, 1),
            "volatility": round(self.volatility * 100, 1),
            "vix": round(self.vix, 1),
            "spx_return_5d": round(self.spx_return_5d, 1),
            "spx_return_20d": round(self.spx_return_20d, 1),
            "spx_drawdown": round(self.spx_drawdown, 1),
            "correlation_spike": self.correlation_spike,
            "description": self.description,
        }


# ═══════════════════════════════════════════════════════════════════
# Regime Agent
# ═══════════════════════════════════════════════════════════════════

class RegimeAgent:
    """
    市场状态识别 Agent。
    
    使用规则法（ADX + 波动率 + VIX + 回撤）判断当前市场状态。
    规则法被 Wayland 评价为"最实用"——简单可靠，适合个人量化系统。
    """

    def __init__(self):
        self.state_path = os.path.join(DATA_DIR, "regime_state.json")
        self._last_state = None

    def analyze(self, force_refresh: bool = False) -> RegimeState:
        """
        分析当前市场状态。结果缓存 1 小时避免重复请求。
        
        返回 RegimeState 包含:
          - regime: TRENDING / MEAN_REVERTING / CRISIS / UNCERTAIN
          - degradation_level: 0-3
          - 各项市场指标
        """
        # 缓存检查（1小时内不重复请求）
        if not force_refresh and self._last_state:
            age = (datetime.now() - datetime.fromisoformat(self._last_state.timestamp)).total_seconds()
            if age < 3600:  # 1小时缓存
                return self._last_state

        try:
            # 1. 获取市场数据
            indicators = self._fetch_market_data()
            
            # 2. 计算各项指标
            state = self._classify_regime(indicators)
            
            # 3. 确定降级级别
            state.degradation_level = self._determine_degradation(state)
            
            # 4. 生成描述文本
            state.description = self._describe_state(state)
            state.timestamp = datetime.now().isoformat()
            
            # 5. 缓存并持久化
            self._last_state = state
            self._save_state(state)
            
            return state
            
        except Exception as e:
            print(f"  [!] Regime 分析失败: {e}")
            # 返回上次已知状态或默认
            return self._load_state() or RegimeState(
                regime="UNCERTAIN", description=f"分析失败: {e}"
            )

    # ── 数据获取 ──────────────────────────────────────────────────

    def _fetch_market_data(self) -> dict:
        """获取市场指标数据"""
        result = {}
        
        # 获取 SPY 数据（用于 ADX、波动率、回撤）
        spy = yf.download("SPY", period="6mo", interval="1d", progress=False, auto_adjust=True)
        if len(spy) >= 50:
            # yfinance >=1.3 returns MultiIndex - flatten
            for col in ["Close", "High", "Low"]:
                vals = spy[col].values
                result[f"spy_{col.lower()}"] = vals.flatten() if hasattr(vals, 'flatten') and len(vals.shape) > 1 else vals
            result["spy_dates"] = spy.index
        
        # 获取 VIX 数据
        vix = yf.download("^VIX", period="1mo", interval="1d", progress=False, auto_adjust=True)
        if len(vix) >= 2:
            vix_close = vix["Close"].values.flatten() if hasattr(vix["Close"].values, 'flatten') and len(vix["Close"].values.shape) > 1 else vix["Close"].values
            result["vix_close"] = vix_close[-1]
            result["vix_ma20"] = vix_close[-20:].mean() if len(vix_close) >= 20 else vix_close.mean()
        else:
            result["vix_close"] = 15.0  # 默认低波动
            result["vix_ma20"] = 15.0
        
        return result

    # ── 指标计算 ──────────────────────────────────────────────────

    def _compute_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """计算 ADX（平均趋向指数）"""
        if len(close) < period + 1:
            return 25.0  # 默认中性
        
        # True Range
        high_low = high[1:] - low[1:]
        high_close = np.abs(high[1:] - close[:-1])
        low_close = np.abs(low[1:] - close[:-1])
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth
        atr = pd.Series(tr).ewm(span=period).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period).mean().values / (atr + 1e-10)
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period).mean().values / (atr + 1e-10)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(span=period).mean().values[-1]
        
        return float(adx)

    def _classify_regime(self, d: dict) -> RegimeState:
        """基于规则法判断市场状态"""
        state = RegimeState()
        
        close = d.get("spy_close", np.array([]))
        high = d.get("spy_high", np.array([]))
        low = d.get("spy_low", np.array([]))
        
        if len(close) < 20:
            return state
        
        # ── 计算基础指标 ───────────────────────────────────────
        
        # ADX（趋势强度）
        adx = self._compute_adx(high, low, close)
        state.adx = adx
        
        # 年化波动率（20日）
        daily_returns = np.diff(close) / close[:-1]
        state.volatility = float(np.std(daily_returns[-20:]) * np.sqrt(252))
        
        # VIX
        state.vix = float(d.get("vix_close", 15))
        vix_ma = float(d.get("vix_ma20", 15))
        
        # SPX 近期收益
        state.spx_return_5d = float((close[-1] - close[-6]) / close[-6] * 100) if len(close) >= 6 else 0
        state.spx_return_20d = float((close[-1] - close[-21]) / close[-21] * 100) if len(close) >= 21 else 0
        
        # 从高点回撤
        peak = np.max(close[-60:]) if len(close) >= 60 else np.max(close)
        state.spx_drawdown = float((peak - close[-1]) / peak * 100)
        
        # 相关性飙升检测（简化：检查各板块是否同涨同跌）
        # 用20日收益率的横截面标准差衡量——低std = 同涨同跌 = 相关性飙升
        if len(daily_returns) >= 20:
            sector_std = np.std(daily_returns[-5:])
            overall_std = np.std(daily_returns[-20:])
            state.correlation_spike = sector_std < overall_std * 0.3 and overall_std > 0.01
        
        # ── 状态分类 ───────────────────────────────────────────
        
        # 危机检测（最高优先级）
        is_crisis = (
            state.vix > 30
            or state.spx_drawdown > 10
            or (state.spx_return_20d < -10 and state.volatility > 0.30)
        )
        
        # 趋势检测
        is_trending = adx > 25 and abs(state.spx_return_20d) > 3
        
        # 震荡检测
        is_mean_reverting = adx < 20 and state.volatility < 0.25
        
        # ── 裁决 ───────────────────────────────────────────────
        if is_crisis:
            state.regime = "CRISIS"
            state.confidence = min(90 + state.vix / 5, 99)
        elif is_trending:
            state.regime = "TRENDING"
            state.confidence = min(adx * 2, 85)
        elif is_mean_reverting:
            state.regime = "MEAN_REVERTING"
            state.confidence = 65
        else:
            # 不确定状态
            state.regime = "UNCERTAIN"
            state.confidence = 50
        
        return state

    def _determine_degradation(self, state: RegimeState) -> int:
        """
        基于市场状态确定降级级别。
        
        Level 0: 正常 — 全功能
        Level 1: 谨慎 — 降仓30%，仅接受高分信号
        Level 2: 防御 — 降仓60%，仅接受最稳健信号
        Level 3: 安全 — 停止所有新开仓，只平仓
        """
        if state.regime == "CRISIS":
            if state.vix > 40 or state.spx_drawdown > 15:
                return 3  # 安全模式
            return 2  # 防御模式
        
        if state.regime == "UNCERTAIN":
            return 1  # 谨慎模式
        
        if state.regime == "TRENDING":
            if abs(state.spx_return_20d) > 8:
                return 1  # 快速趋势中保持谨慎
            return 0  # 正常
        
        # MEAN_REVERTING
        return 0

    def _describe_state(self, state: RegimeState) -> str:
        """生成可读的市场状态描述"""
        regime_names = {
            "TRENDING": "趋势市",
            "MEAN_REVERTING": "震荡市",
            "CRISIS": "危机模式",
            "UNCERTAIN": "不确定",
        }
        deg_names = ["正常", "谨慎", "防御", "安全"]
        
        parts = [
            f"{regime_names.get(state.regime, '?')}(L{state.degradation_level} {deg_names[state.degradation_level]})",
            f"ADX={state.adx:.0f} VIX={state.vix:.0f}",
        ]
        
        if state.regime == "CRISIS":
            parts.append(f"回撤{state.spx_drawdown:.1f}% VIX={state.vix:.0f}")
        elif state.regime == "TRENDING":
            direction = "上涨" if state.spx_return_20d > 0 else "下跌"
            parts.append(f"20日{state.spx_return_20d:+.0f}% {direction}")
        elif state.regime == "UNCERTAIN":
            switch_hints = []
            if state.adx > 18:
                switch_hints.append("ADX接近趋势阈值")
            if state.vix > 25:
                switch_hints.append("VIX偏高")
            parts.extend(switch_hints)
        
        return " | ".join(parts)

    # ── 持久化 ──────────────────────────────────────────────────

    def _save_state(self, state: RegimeState):
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)
        except Exception:
            pass

    def _load_state(self) -> Optional[RegimeState]:
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r') as f:
                    data = json.load(f)
                return RegimeState(**data)
        except Exception:
            pass
        return None


# ═══════════════════════════════════════════════════════════════════
# 便捷函数：生成降级建议（供下游 Agent 使用）
# ═══════════════════════════════════════════════════════════════════

def get_regime_adjustments(state: RegimeState) -> dict:
    """根据当前市场状态返回调整参数"""
    adj = {
        "max_position_pct": 0.20,
        "stop_loss_pct": 8.0,
        "take_profit_pct": 15.0,
        "score_threshold": 50,
        "slippage_buffer": 0.002,
        "trade_cooling": False,
        "risk_multiplier": 1.0,
    }
    
    level = state.degradation_level
    
    if level >= 3:
        # 安全模式
        adj["max_position_pct"] = 0.0
        adj["trade_cooling"] = True
        adj["risk_multiplier"] = 0.0
    elif level == 2:
        # 防御模式
        adj["max_position_pct"] = 0.08
        adj["stop_loss_pct"] = 5.0
        adj["take_profit_pct"] = 10.0
        adj["score_threshold"] = 70
        adj["slippage_buffer"] = 0.005
        adj["risk_multiplier"] = 0.4
    elif level == 1:
        # 谨慎模式
        adj["max_position_pct"] = 0.14
        adj["stop_loss_pct"] = 6.0
        adj["take_profit_pct"] = 12.0
        adj["score_threshold"] = 60
        adj["slippage_buffer"] = 0.003
        adj["risk_multiplier"] = 0.7
    
    # 趋势市：放宽止损（趋势延续概率高）
    if state.regime == "TRENDING" and level == 0:
        adj["stop_loss_pct"] = 10.0
        adj["take_profit_pct"] = 20.0
    
    # 高波动：收紧仓位
    if state.volatility > 0.35:
        adj["max_position_pct"] = min(adj["max_position_pct"], 0.10)
        adj["risk_multiplier"] *= 0.7
    
    return adj
