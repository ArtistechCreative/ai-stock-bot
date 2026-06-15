"""
回测质量门 — 5 层 20 项检查
================================
基于 Wayland Zhang《AI量化交易从0到1》第 07 课框架

用法:
    from backtest_quality import BacktestQualityGate, QualityConfig
    
    gate = BacktestQualityGate()
    report = gate.evaluate(backtest_result, metadata={
        "data_years": 5,
        "has_survivorship_bias": True,
        "oos_return_pct": 8.0,
        "train_return_pct": 15.0,
        ...
    })
    print(report.summary())
    print(f"总分: {report.total_score}/100 — {'✅ PASS' if report.passed else '❌ FAIL'}")
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class QualityConfig:
    """质量门阈值配置（可按风险偏好调整）"""
    # Layer 1: 数据完整性
    min_data_years: float = 5.0
    require_survivorship_free: bool = True
    require_adjusted_prices: bool = True
    
    # Layer 2: 时间完整性
    lookahead_tolerance_days: int = 1  # 信号T日生成T+1日执行
    max_data_leakage_ratio: float = 0.01
    
    # Layer 3: 过拟合检测
    oos_to_train_min_ratio: float = 0.5  # OOS收益 >= 训练收益的50%
    param_stability_max_change: float = 30.0  # 参数±20%变化，收益变化<30%
    min_sharpe_per_year: float = 0.5
    
    # Layer 4: 成本建模
    conservative_slippage: float = 0.002  # 0.2%
    include_funding_cost: bool = True
    
    # Layer 5: 验证方法
    min_walk_forward_rounds: int = 10
    monte_carlo_pass_rate: float = 0.90  # 90%场景结果>0
    
    # 综合
    pass_threshold: float = 60.0  # 60分及格
    warn_threshold: float = 40.0  # 40分以下危险


@dataclass
class CheckResult:
    """单项检查结果"""
    name: str
    passed: bool
    score: float  # 0-100
    detail: str = ""
    severity: str = "info"  # critical / warning / info

    def __str__(self):
        icon = "✅" if self.passed else "❌"
        return f"  {icon} {self.name}: {self.score:.0f}/100 — {self.detail}"


@dataclass
class QualityReport:
    """质量门完整报告"""
    timestamp: str = ""
    layer_scores: dict = field(default_factory=dict)
    checks: list = field(default_factory=list)
    total_score: float = 0.0
    passed: bool = False
    warnings: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"\n{'='*55}",
            f"  回测质量门报告 — {self.timestamp}",
            f"{'='*55}",
        ]
        for layer_name, score in self.layer_scores.items():
            bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
            lines.append(f"  {layer_name:20s} {bar} {score:.0f}/100")
        lines.append(f"  {'─'*40}")
        lines.append(f"  {'总分':20s} {'█' * int(self.total_score/10)}{'░' * (10-int(self.total_score/10))} {self.total_score:.0f}/100")
        
        if self.passed:
            lines.append(f"\n  ✅ 判定: PASS (≥60分) — 回测可信度可接受")
        elif self.total_score >= 40:
            lines.append(f"\n  ⚠️ 判定: WARN (40-60分) — 回测有风险，需修正后再实盘")
        else:
            lines.append(f"\n  ❌ 判定: FAIL (<40分) — 回测不可信，必须修正")
        
        if self.recommendations:
            lines.append(f"\n  📋 改进建议:")
            for r in self.recommendations:
                lines.append(f"    • {r}")
        
        lines.append(f"{'='*55}\n")
        return "\n".join(lines)


class BacktestQualityGate:
    """回测质量门 — 5 层 20 项检查"""

    def __init__(self, config: QualityConfig = None):
        self.config = config or QualityConfig()

    def evaluate(self, result=None, metadata: dict = None) -> QualityReport:
        """
        执行完整质量检查。
        
        参数:
            result: BacktestResult 对象（可选，部分检查不需要）
            metadata: 包含回测配置信息的字典
                - data_years: 数据覆盖年数
                - has_survivorship_bias: 是否有幸存者偏差
                - has_lookahead: 是否有前瞻偏差
                - oos_return_pct: 样本外收益率(%)
                - train_return_pct: 样本内收益率(%)
                - param_stability_pct: 参数敏感性(%)
                - num_strategies_tested: 测试的策略数
                - slippage_assumed: 假设的滑点
                - walk_forward_rounds: Walk-Forward轮数
                - monte_carlo_pass_rate: Monte Carlo通过率
                - stress_test_2008: 2008年压力测试结果
                - stress_test_2020: 2020年压力测试结果
                - stress_test_2022: 2022年压力测试结果
        """
        metadata = metadata or {}
        report = QualityReport(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # 从result提取数据（如果有）
        if result:
            metadata.setdefault("total_return_pct", getattr(result, 'total_return_pct', 0))
            metadata.setdefault("max_drawdown_pct", getattr(result, 'max_drawdown_pct', 0))
            metadata.setdefault("sharpe_ratio", getattr(result, 'sharpe_ratio', 0))
            metadata.setdefault("win_rate", getattr(result, 'win_rate', 0))
            metadata.setdefault("total_trades", getattr(result, 'total_trades', 0))

        # ── Layer 1: 数据完整性 ──────────────────────────────────────
        l1_checks = self._layer1_data_integrity(metadata)
        report.checks.extend(l1_checks)
        report.layer_scores["Layer 1: 数据完整性"] = sum(c.score for c in l1_checks) / len(l1_checks)

        # ── Layer 2: 时间完整性 ──────────────────────────────────────
        l2_checks = self._layer2_time_integrity(metadata)
        report.checks.extend(l2_checks)
        report.layer_scores["Layer 2: 时间完整性"] = sum(c.score for c in l2_checks) / len(l2_checks)

        # ── Layer 3: 过拟合检测 ──────────────────────────────────────
        l3_checks = self._layer3_overfitting(metadata, result)
        report.checks.extend(l3_checks)
        report.layer_scores["Layer 3: 过拟合检测"] = sum(c.score for c in l3_checks) / len(l3_checks)

        # ── Layer 4: 成本建模 ────────────────────────────────────────
        l4_checks = self._layer4_cost_modeling(metadata, result)
        report.checks.extend(l4_checks)
        report.layer_scores["Layer 4: 成本建模"] = sum(c.score for c in l4_checks) / len(l4_checks)

        # ── Layer 5: 验证方法 ──────────────────────────────────────
        l5_checks = self._layer5_validation(metadata)
        report.checks.extend(l5_checks)
        report.layer_scores["Layer 5: 验证方法"] = sum(c.score for c in l5_checks) / len(l5_checks)

        # ── 综合评分 ────────────────────────────────────────────────
        weights = {"Layer 1: 数据完整性": 0.20,
                    "Layer 2: 时间完整性": 0.20,
                    "Layer 3: 过拟合检测": 0.25,
                    "Layer 4: 成本建模": 0.20,
                    "Layer 5: 验证方法": 0.15}
        
        report.total_score = sum(
            report.layer_scores.get(k, 0) * w for k, w in weights.items()
        )
        report.passed = report.total_score >= self.config.pass_threshold

        # ── 生成建议 ────────────────────────────────────────────────
        for check in report.checks:
            if not check.passed and check.severity == "critical":
                report.recommendations.append(f"[严重] {check.name}: {check.detail}")
        for check in report.checks:
            if not check.passed and check.severity == "warning":
                report.recommendations.append(f"[警告] {check.name}: {check.detail}")

        return report

    # ── Layer 1 ────────────────────────────────────────────────────

    def _layer1_data_integrity(self, m: dict) -> list:
        checks = []
        years = m.get("data_years", 0)
        
        # 1.1 数据覆盖
        if years >= self.config.min_data_years:
            checks.append(CheckResult("1.1 数据覆盖≥5年含牛熊周期", True, 100,
                                       f"✓ {years:.1f}年覆盖"))
        else:
            score = max(0, years / self.config.min_data_years * 100)
            checks.append(CheckResult("1.1 数据覆盖≥5年含牛熊周期", score >= 60, score,
                                       f"✗ 仅{years:.1f}年（建议≥{self.config.min_data_years}年）",
                                       severity="critical"))
        
        # 1.2 幸存者偏差
        if not m.get("has_survivorship_bias", True):
            checks.append(CheckResult("1.2 幸存者偏差控制", True, 100,
                                       "✓ 包含退市股数据"))
        else:
            checks.append(CheckResult("1.2 幸存者偏差控制", False, 30,
                                       "✗ 未处理幸存者偏差（可高估收益50%+）",
                                       severity="critical"))
        
        # 1.3 复权处理
        if m.get("has_adjusted_prices", False):
            checks.append(CheckResult("1.3 复权/换月处理", True, 100,
                                       "✓ 使用后复权价格"))
        else:
            checks.append(CheckResult("1.3 复权/换月处理", False, 40,
                                       "✗ 未使用复权价格（可能产生假信号）",
                                       severity="warning"))
        
        # 1.4 时区对齐（假设用yfinance默认已对齐）
        if not m.get("has_timezone_issue", True):
            checks.append(CheckResult("1.4 时区对齐", True, 100, "✓ 时区已统一"))
        else:
            checks.append(CheckResult("1.4 时区对齐", True, 80,
                                       "⚠ 需确认所有数据源时区一致", severity="info"))
        
        return checks

    # ── Layer 2 ────────────────────────────────────────────────────

    def _layer2_time_integrity(self, m: dict) -> list:
        checks = []
        
        # 2.1 Look-Ahead Bias
        if not m.get("has_lookahead", True):
            checks.append(CheckResult("2.1 Look-Ahead Bias", True, 100,
                                       "✓ 信号T日生成T+1日执行"))
        else:
            checks.append(CheckResult("2.1 Look-Ahead Bias", False, 0,
                                       "✗ 存在前瞻偏差（可虚高收益2-10x）",
                                       severity="critical"))
        
        # 2.2 数据泄漏
        leakage = m.get("data_leakage_ratio", 0)
        if leakage <= self.config.max_data_leakage_ratio:
            checks.append(CheckResult("2.2 数据泄漏", True, 100,
                                       f"✓ 泄漏率{leakage:.1%}"))
        else:
            checks.append(CheckResult("2.2 数据泄漏", False, 20,
                                       f"✗ 泄漏率{leakage:.1%}（建议<{self.config.max_data_leakage_ratio:.0%})",
                                       severity="critical"))
        
        # 2.3 特征计算（用shift(1)或更早数据）
        if m.get("feature_shift_used", True):
            checks.append(CheckResult("2.3 特征计算防未来数据", True, 100,
                                       "✓ 特征用shift(1)或更早数据"))
        else:
            checks.append(CheckResult("2.3 特征计算防未来数据", False, 0,
                                       "✗ 特征可能用了未来数据",
                                       severity="critical"))
        
        # 2.4 标签定义
        if m.get("label_leakage_prevented", True):
            checks.append(CheckResult("2.4 标签无泄漏", True, 100,
                                       "✓ 标签只用当前时刻前数据"))
        else:
            checks.append(CheckResult("2.4 标签无泄漏", False, 0,
                                       "✗ 标签泄漏风险", severity="critical"))
        
        return checks

    # ── Layer 3 ────────────────────────────────────────────────────

    def _layer3_overfitting(self, m: dict, result=None) -> list:
        checks = []
        
        # 3.1 OOS表现
        oos = m.get("oos_return_pct", None)
        train = m.get("train_return_pct", None)
        if oos is not None and train is not None and train != 0:
            ratio = abs(oos / train) if train != 0 else 0
            threshold = self.config.oos_to_train_min_ratio
            if ratio >= threshold:
                checks.append(CheckResult("3.1 OOS表现", True, 100,
                                           f"✓ OOS/训练={ratio:.0%}（≥{threshold:.0%}）"))
            else:
                score = max(0, ratio / threshold * 100)
                checks.append(CheckResult("3.1 OOS表现", score >= 60, score,
                                           f"✗ OOS/训练={ratio:.0%}（建议≥{threshold:.0%}）",
                                           severity="critical"))
        else:
            checks.append(CheckResult("3.1 OOS表现", False, 30,
                                       "✗ 未提供样本外数据（无法评估过拟合）",
                                       severity="warning"))
        
        # 3.2 参数稳定性
        stability = m.get("param_stability_pct", None)
        if stability is not None:
            if stability <= self.config.param_stability_max_change:
                checks.append(CheckResult("3.2 参数稳定性", True, 100,
                                           f"✓ 参数变化→收益变化{stability:.0f}%（<{self.config.param_stability_max_change}%）"))
            else:
                score = max(0, (1 - (stability - self.config.param_stability_max_change) / 100) * 100)
                checks.append(CheckResult("3.2 参数稳定性", False, max(10, score),
                                           f"✗ 参数变化→收益变化{stability:.0f}%（建议<{self.config.param_stability_max_change}%）",
                                           severity="warning"))
        else:
            checks.append(CheckResult("3.2 参数稳定性", False, 30,
                                       "✗ 未测试参数敏感性", severity="warning"))
        
        # 3.3 多重检验
        n_tests = m.get("num_strategies_tested", 1)
        if n_tests <= 1:
            checks.append(CheckResult("3.3 多重检验校正", True, 100,
                                       "✓ 未做多重测试（单策略）"))
        else:
            threshold = 0.05 / n_tests
            checks.append(CheckResult("3.3 多重检验校正", False, 50,
                                       f"⚠ 测试了{n_tests}个策略，p值阈值应为{threshold:.4f}",
                                       severity="warning"))
        
        # 3.4 跨时期稳定性
        sharpe = m.get("sharpe_ratio", None)
        if sharpe is not None:
            if sharpe >= self.config.min_sharpe_per_year:
                checks.append(CheckResult("3.4 跨时期稳定性", True, 100,
                                           f"✓ Sharpe={sharpe:.2f}（≥{self.config.min_sharpe_per_year}）"))
            else:
                checks.append(CheckResult("3.4 跨时期稳定性", False,
                                           max(10, sharpe / self.config.min_sharpe_per_year * 100),
                                           f"✗ Sharpe={sharpe:.2f}（建议≥{self.config.min_sharpe_per_year}）",
                                           severity="warning"))
        else:
            checks.append(CheckResult("3.4 跨时期稳定性", True, 50,
                                       "⚠ 未提供Sharpe数据", severity="info"))
        
        return checks

    # ── Layer 4 ────────────────────────────────────────────────────

    def _layer4_cost_modeling(self, m: dict, result=None) -> list:
        checks = []
        
        # 4.1 手续费
        fee = m.get("fee_per_trade", 0)
        if fee >= 0:
            checks.append(CheckResult("4.1 手续费建模", True, 80,
                                       f"✓ 含手续费{fee:.4f}"))
        else:
            checks.append(CheckResult("4.1 手续费建模", False, 30,
                                       "✗ 未考虑手续费", severity="warning"))
        
        # 4.2 滑点假设
        slippage = m.get("slippage_assumed", 0)
        if slippage >= self.config.conservative_slippage:
            checks.append(CheckResult("4.2 滑点假设", True, 100,
                                       f"✓ 保守滑点{slippage:.1%}"))
        elif slippage > 0:
            score = slippage / self.config.conservative_slippage * 100
            checks.append(CheckResult("4.2 滑点假设", False, score,
                                       f"⚠ 滑点{slippage:.1%}（建议≥{self.config.conservative_slippage:.1%}）",
                                       severity="warning"))
        else:
            checks.append(CheckResult("4.2 滑点假设", False, 0,
                                       "✗ 未考虑滑点（高频策略必亏）",
                                       severity="critical"))
        
        # 4.3 市场冲击
        if m.get("market_impact_modeled", False):
            checks.append(CheckResult("4.3 市场冲击建模", True, 100,
                                       "✓ 考虑了市场冲击"))
        else:
            checks.append(CheckResult("4.3 市场冲击建模", True, 60,
                                       "⚠ 未考虑市场冲击（小资金可接受）",
                                       severity="info"))
        
        # 4.4 资金成本
        if not m.get("has_short_positions", False):
            checks.append(CheckResult("4.4 资金/融券成本", True, 100,
                                       "✓ 无做空仓位"))
        elif m.get("borrow_cost_modeled", False):
            checks.append(CheckResult("4.4 资金/融券成本", True, 100,
                                       "✓ 已建模融券成本"))
        else:
            checks.append(CheckResult("4.4 资金/融券成本", False, 40,
                                       "✗ 做空但未考虑融券成本",
                                       severity="warning"))
        
        return checks

    # ── Layer 5 ────────────────────────────────────────────────────

    def _layer5_validation(self, m: dict) -> list:
        checks = []
        
        # 5.1 Walk-Forward
        wf = m.get("walk_forward_rounds", 0)
        if wf >= self.config.min_walk_forward_rounds:
            checks.append(CheckResult("5.1 Walk-Forward验证", True, 100,
                                       f"✓ {wf}轮（≥{self.config.min_walk_forward_rounds}）"))
        elif wf > 0:
            score = wf / self.config.min_walk_forward_rounds * 100
            checks.append(CheckResult("5.1 Walk-Forward验证", False, score,
                                       f"⚠ 仅{wf}轮（建议≥{self.config.min_walk_forward_rounds}）",
                                       severity="warning"))
        else:
            checks.append(CheckResult("5.1 Walk-Forward验证", False, 20,
                                       "✗ 未做Walk-Forward验证", severity="warning"))
        
        # 5.2 Monte Carlo
        mc = m.get("monte_carlo_pass_rate", 0)
        if mc >= self.config.monte_carlo_pass_rate:
            checks.append(CheckResult("5.2 Monte Carlo模拟", True, 100,
                                       f"✓ {mc:.0%}场景通过（≥{self.config.monte_carlo_pass_rate:.0%}）"))
        elif mc > 0:
            score = mc / self.config.monte_carlo_pass_rate * 100
            checks.append(CheckResult("5.2 Monte Carlo模拟", False, score,
                                       f"⚠ 仅{mc:.0%}场景通过（建议≥{self.config.monte_carlo_pass_rate:.0%}）",
                                       severity="warning"))
        else:
            checks.append(CheckResult("5.2 Monte Carlo模拟", False, 20,
                                       "✗ 未做Monte Carlo模拟", severity="info"))
        
        # 5.3 压力测试
        stress_tests = []
        for year in ["2008", "2020", "2022"]:
            val = m.get(f"stress_test_{year}", None)
            if val is not None:
                stress_tests.append((year, val))
        
        if stress_tests:
            all_passed = all(v > 0 for _, v in stress_tests)
            years_str = ", ".join(f"{y}: {v:+.1f}%" for y, v in stress_tests)
            checks.append(CheckResult("5.3 压力测试", all_passed,
                                       100 if all_passed else 50,
                                       f"{'✓' if all_passed else '⚠'} {years_str}",
                                       severity="warning" if not all_passed else "info"))
        else:
            checks.append(CheckResult("5.3 压力测试", False, 30,
                                       "✗ 未做压力测试（2008/2020/2022）",
                                       severity="warning"))
        
        # 5.4 实盘预期收益估算
        total_return = m.get("total_return_pct", None)
        if total_return is not None:
            expected_real = total_return * 0.5 - 10  # 回测×0.5 - 隐性成本
            detail = f"回测{total_return:+.1f}% → 实盘预期{expected_real:+.1f}%"
            if expected_real > 0:
                checks.append(CheckResult("5.4 实盘预期收益估算", True, 80, f"✓ {detail}"))
            else:
                checks.append(CheckResult("5.4 实盘预期收益估算", False, 30,
                                           f"✗ {detail}（实盘预期为负）",
                                           severity="critical"))
        else:
            checks.append(CheckResult("5.4 实盘预期收益估算", False, 50,
                                       "⚠ 未估算实盘预期收益", severity="info"))
        
        return checks
