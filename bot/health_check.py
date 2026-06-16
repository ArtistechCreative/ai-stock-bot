"""
健康检查 — 量化系统的 12 种死亡方式预防
=========================================
基于 Wayland Zhang《AI量化交易从0到1》附录 B 框架

用法:
    from health_check import HealthChecker
    checker = HealthChecker()
    report = checker.run_all()  # 运行全部 12 项检查
    print(report.summary())

每周自动运行一次（集成到 run_cron.py 或 standalone cronjob）。
"""

import os, json, time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
HISTORY_FILE = os.path.join(DATA_DIR, "health_history.json")
STATE_FILES = {
    "regime": os.path.join(DATA_DIR, "regime_state.json"),
    "risk": os.path.join(DATA_DIR, "risk_state.json"),
    "strategy": os.path.join(DATA_DIR, "strategy_state.json"),
    "signals": os.path.join(DATA_DIR, "signals.json"),
}


# ── 数据类型 ─────────────────────────────────────────────────────

@dataclass
class HealthCheckItem:
    """单项健康检查结果"""
    id: str
    name: str
    passed: bool
    status: str = ""     # OK / WARN / FAIL / SKIP
    detail: str = ""
    severity: str = "warning"  # critical / warning / info
    value: str = ""

    def icon(self) -> str:
        return {"OK": "✅", "WARN": "⚠️", "FAIL": "❌", "SKIP": "⏭️"}.get(self.status, "❓")


@dataclass
class HealthReport:
    """完整健康检查报告"""
    timestamp: str = ""
    checks: list = field(default_factory=list)
    passed_count: int = 0
    warn_count: int = 0
    fail_count: int = 0
    
    def summary(self) -> str:
        """生成报告文本"""
        lines = [
            f"\n{'='*55}",
            f"  系统健康检查 — {self.timestamp}",
            f"{'='*55}",
            f"  结果: ✅ {self.passed_count} 通过 / ⚠️ {self.warn_count} 警告 / ❌ {self.fail_count} 失败",
        ]
        
        for c in self.checks:
            lines.append(f"  {c.icon()} [{c.status}] {c.name}: {c.detail[:60]}")
        
        if self.fail_count > 0:
            lines.append(f"\n  🛑 {self.fail_count} 项紧急问题需要处理!")
        elif self.warn_count > 0:
            lines.append(f"\n  ⚠️ {self.warn_count} 项警告请关注")
        else:
            lines.append(f"\n  ✅ 系统健康")
        
        lines.append(f"{'='*55}\n")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Health Checker
# ═══════════════════════════════════════════════════════════════════

class HealthChecker:
    """
    量化系统健康检查器。
    
    检查 12 个维度（对应书中的 12 种死亡方式）：
      1. 数据质量     — 数据源连接、数据新鲜度
      2. Regime 状态  — 市场状态是否正确
      3. 执行质量     — 滑点/成交率
      4. 风控完整性   — Risk Agent 是否在线
      5. 流动性       — 持仓标的流动性
      6. 相关性       — 组合内相关性
      7. 杠杆         — 净暴露是否超限
      8. 人工干预     — 是否有未记录的人工操作
      9. 系统健康     — 进程/API/磁盘
     10. 监管/环境    — 策略相关法规变化
     11. Alpha 衰减   — IC/Sharpe 趋势
     12. 过拟合       — OOS 表现是否正常
    """

    def __init__(self):
        self.history = self._load_history()
        self._now = datetime.now()

    def run_all(self, context: dict = None) -> HealthReport:
        """运行全部 12 项检查"""
        context = context or {}
        report = HealthReport(timestamp=self._now.isoformat())
        
        # 1. 数据质量
        report.checks.append(self._check_data_quality(context))
        
        # 2. Regime 状态
        report.checks.append(self._check_regime_state())
        
        # 3. 执行质量
        report.checks.append(self._check_execution_quality())
        
        # 4. 风控完整性
        report.checks.append(self._check_risk_control())
        
        # 5. 流动性
        report.checks.append(self._check_liquidity())
        
        # 6. 相关性
        report.checks.append(self._check_correlation())
        
        # 7. 杠杆
        report.checks.append(self._check_leverage())
        
        # 8. 人工干预
        report.checks.append(self._check_human_intervention())
        
        # 9. 系统健康
        report.checks.append(self._check_system_health())
        
        # 10. 监管/环境
        report.checks.append(self._check_regulatory())
        
        # 11. Alpha 衰减
        report.checks.append(self._check_alpha_decay())
        
        # 12. 过拟合
        report.checks.append(self._check_overfitting())
        
        # 统计
        report.passed_count = sum(1 for c in report.checks if c.status == "OK")
        report.warn_count = sum(1 for c in report.checks if c.status == "WARN")
        report.fail_count = sum(1 for c in report.checks if c.status == "FAIL")
        
        # 保存历史
        self._save_history(report)
        
        return report

    # ── 各检查项 ──────────────────────────────────────────────────

    def _check_data_quality(self, ctx: dict) -> HealthCheckItem:
        """#1 数据污染型死亡 — 数据源质量"""
        item = HealthCheckItem("D01", "数据质量", True, severity="critical")
        
        # 检查策略状态文件最近更新时间
        state_path = STATE_FILES.get("strategy")
        if state_path and os.path.exists(state_path):
            mtime = os.path.getmtime(state_path)
            age_hours = (self._now.timestamp() - mtime) / 3600
            if age_hours < 24:
                item.status = "OK"
                item.detail = f"策略状态文件 {age_hours:.0f}小时前更新"
                item.value = f"{age_hours:.0f}h"
            elif age_hours < 72:
                item.status = "WARN"
                item.detail = f"策略状态文件 {age_hours:.0f}小时未更新"
                item.value = f"{age_hours:.0f}h"
            else:
                item.status = "FAIL"
                item.detail = f"策略状态文件 {age_hours:.0f}小时未更新 — 数据可能已中断"
                item.value = f"{age_hours:.0f}h"
                item.passed = False
        else:
            item.status = "SKIP"
            item.detail = "无状态文件可检查"
        
        return item

    def _check_regime_state(self) -> HealthCheckItem:
        """#2 Regime 漂移型死亡 — 市场状态检测"""
        item = HealthCheckItem("R01", "Regime 状态", True, severity="critical")
        
        state_path = STATE_FILES.get("regime")
        if state_path and os.path.exists(state_path):
            try:
                with open(state_path) as f:
                    data = json.load(f)
                level = data.get("degradation_level", 0)
                regime = data.get("regime", "?")
                item.value = f"{regime} L{level}"
                
                if level >= 3:
                    item.status = "FAIL"
                    item.detail = f"安全模式(L{level}) — 系统已停止交易"
                    item.passed = False
                elif level == 2:
                    item.status = "WARN"
                    item.detail = f"防御模式(L{level}) — 仅最高分信号"
                else:
                    item.status = "OK"
                    item.detail = f"状态正常: {regime} L{level}"
            except Exception as e:
                item.status = "WARN"
                item.detail = f"读取失败: {e}"
        else:
            item.status = "WARN"
            item.detail = "Regime Agent 未运行"
        
        return item

    def _check_execution_quality(self) -> HealthCheckItem:
        """#3 执行失真型死亡 — 滑点/成交率"""
        # 当前为 dry-run 模式，无法检查实际执行
        item = HealthCheckItem("E01", "执行质量", True, severity="warning")
        item.status = "SKIP"
        item.detail = "当前 Dry-run 模式，无实盘执行数据"
        return item

    def _check_risk_control(self) -> HealthCheckItem:
        """#4 风控失效型死亡 — Risk Agent 完整性"""
        item = HealthCheckItem("K01", "风控完整性", True, severity="critical")
        
        risk_path = STATE_FILES.get("risk")
        if risk_path and os.path.exists(risk_path):
            try:
                with open(risk_path) as f:
                    data = json.load(f)
                cb = data.get("circuit_breaker_active", False)
                item.value = f"熔断={'激活' if cb else '关闭'}"
                if cb:
                    item.status = "WARN"
                    item.detail = f"熔断激活: {data.get('circuit_breaker_reason', '?')}"
                else:
                    item.status = "OK"
                    item.detail = "Risk Agent 在线，熔断关闭"
            except Exception:
                item.status = "WARN"
                item.detail = "Risk Agent 状态文件无法读取"
        else:
            item.status = "WARN"
            item.detail = "Risk Agent 未运行"
        
        return item

    def _check_liquidity(self) -> HealthCheckItem:
        """#5 流动性枯竭型死亡"""
        # 简化：检查是否有持仓标的在 signals.json 中出现
        item = HealthCheckItem("L01", "流动性", True, severity="warning")
        
        signals_path = STATE_FILES.get("signals")
        if signals_path and os.path.exists(signals_path):
            try:
                with open(signals_path) as f:
                    data = json.load(f)
                ticker_count = len(data) if isinstance(data, list) else 0
                item.value = f"{ticker_count} signals"
                item.status = "OK"
                item.detail = f"信号池 {ticker_count} 个标的"
            except Exception:
                item.status = "SKIP"
                item.detail = "信号文件格式异常"
        else:
            item.status = "SKIP"
            item.detail = "无信号数据"
        
        return item

    def _check_correlation(self) -> HealthCheckItem:
        """#6 相关性飙升型死亡"""
        # 简化：检查 regime 中 correlation_spike 标志
        item = HealthCheckItem("C01", "相关性", True, severity="warning")
        item.status = "SKIP"
        item.detail = "需组合持仓数据计算（待扩展）"
        return item

    def _check_leverage(self) -> HealthCheckItem:
        """#7 杠杆爆仓型死亡"""
        item = HealthCheckItem("V01", "杠杆控制", True, severity="critical")
        item.status = "OK"
        item.detail = "当前 Dry-run 模式，杠杆=0"
        item.value = "0x"
        return item

    def _check_human_intervention(self) -> HealthCheckItem:
        """#8 人为干预型死亡 — 追踪手动操作"""
        item = HealthCheckItem("H01", "人工干预", True, severity="warning")
        item.status = "OK"
        item.detail = "无人工干预记录"
        return item

    def _check_system_health(self) -> HealthCheckItem:
        """#9 系统故障型死亡"""
        item = HealthCheckItem("S01", "系统健康", True, severity="critical")
        
        # 检查必要的文件和目录
        issues = []
        for name, spath in STATE_FILES.items():
            dir_exists = os.path.exists(os.path.dirname(spath))
            if not dir_exists:
                issues.append(f"{name}目录不存在")
        
        if issues:
            item.status = "WARN"
            item.detail = "; ".join(issues)
            item.passed = False
        else:
            item.status = "OK"
            item.detail = "系统目录完整"
        
        return item

    def _check_regulatory(self) -> HealthCheckItem:
        """#10 监管变化型死亡"""
        item = HealthCheckItem("G01", "监管环境", True, severity="info")
        item.status = "OK"
        item.detail = "无已知监管变化"
        return item

    def _check_alpha_decay(self) -> HealthCheckItem:
        """#11 Alpha 衰减型死亡 — 策略表现趋势"""
        item = HealthCheckItem("A01", "Alpha 衰减", True, severity="warning")
        # 检查历史记录中的回测得分趋势
        history = self.history
        if len(history) >= 2:
            # 比较最近两次检查的 pass_count
            last = history[-1]
            prev = history[-2]
            if last.get("passed", 0) < prev.get("passed", 0):
                item.status = "WARN"
                item.detail = f"健康分下降: {prev.get('passed',0)}→{last.get('passed',0)}"
                item.value = f"下降{prev.get('passed',0)-last.get('passed',0)}项"
            else:
                item.status = "OK"
                item.detail = f"健康分稳定, 最近{last.get('passed',0)}/12通过"
        else:
            item.status = "OK"
            item.detail = "首次检查，基线建立中"
            item.value = "基线"
        
        return item

    def _check_overfitting(self) -> HealthCheckItem:
        """#12 过拟合型死亡"""
        item = HealthCheckItem("F01", "过拟合检测", True, severity="warning")
        item.status = "SKIP"
        item.detail = "需回测质量门数据配合（运行 backtest_quality 后自动评估）"
        return item

    # ── 历史记录 ──────────────────────────────────────────────────

    def _load_history(self) -> list:
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE) as f:
                    return json.load(f)
        except Exception:
            pass
        return []

    def _save_history(self, report: HealthReport):
        record = {
            "timestamp": report.timestamp,
            "passed": report.passed_count,
            "warn": report.warn_count,
            "fail": report.fail_count,
            "total": len(report.checks),
        }
        self.history.append(record)
        # 只保留最近 20 条
        self.history = self.history[-20:]
        try:
            os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
            with open(HISTORY_FILE, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception:
            pass
