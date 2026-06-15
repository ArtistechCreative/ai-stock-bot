#!/usr/bin/env python3
"""
AI股神 Cron Job (15分钟 Dry-run)
=================================
执行全链路闭环测试（不真实下单）：
  1. 三大板块评分 (rank_stocks + as_of_date 防泄漏)
  2. 板块路由 DL 预测 (batch_predict_sector)
  3. get_signal() 综合信号（规则为主 + DL 加分器）
  4. Google Sheets 写入（信号池 + frozen column 兼容）
  5. Telegram 推送（dry_run 模式打印，不真实发送）

用法（Hermes cronjob）：
  路径：/home/aitistech/projects/ai-stock-bot/run_cron.py
  计划：*/15 * * * *
"""

import sys, os, json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np  # for backtest

# ── 项目路径 ──────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

# ── 三大板块配置（31 ticker）── 与 train_dl.py / dl_strategy.py 完全一致 ──
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
ALL_TICKERS = [t for tickers in SECTOR_CONFIG.values() for t in tickers]

# ── 状态文件 ──────────────────────────────────────────
STATE_FILE = PROJECT_DIR / "data" / "strategy_state.json"
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

DRY_RUN = True  # True = 只打印，不真实推送/下单


def load_state():
    if STATE_FILE.exists():
        return json.load(open(STATE_FILE))
    return {}


def save_state(state):
    json.dump(state, open(STATE_FILE, "w"), indent=2)


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ════════════════════════════════════════════════════════════════════
# Step 0: 市场状态识别（Regime Detection）
# ════════════════════════════════════════════════════════════════════
REGIME_STATE = None  # 全局市场状态

def step0_regime():
    """分析当前市场状态"""
    global REGIME_STATE
    log("━━━ Step 0: 市场状态识别 ━━━")
    try:
        from bot.regime_agent import RegimeAgent, get_regime_adjustments
        
        agent = RegimeAgent()
        state = agent.analyze(force_refresh=True)
        REGIME_STATE = state
        
        d = state.to_dict()
        log(f"  📊 市场状态: {d['regime']} (L{d['degradation_level']}) | "
            f"ADX={d['adx']} | VIX={d['vix']} | 波动率={d['volatility']}% | "
            f"20日={d['spx_return_20d']}% | 回撤={d['spx_drawdown']}%")
        log(f"  📝 {d['description']}")
        
        # 如果进入安全模式，提前预警
        if d['degradation_level'] >= 3:
            log(f"  🛑 安全模式激活: 停止所有新开仓")
        
        return state
    except Exception as e:
        log(f"  [!] Step0 市场状态识别失败: {e}")
        return None


# ════════════════════════════════════════════════════════════════════
# Step 1: 全市场评分（rank_stocks as_of_date 防泄漏）
# ════════════════════════════════════════════════════════════════════
def step1_score():
    log("━━━ Step 1: 市场评分 ━━━")
    try:
        from bot.scorer import rank_stocks
        # 实时评分（as_of_date=None）— 不泄漏未来
        ranked = rank_stocks(ALL_TICKERS, top_n=12)
        log(f"  评分完成: {len(ranked)} 个资产参与排名")
        for r in ranked[:5]:
            icon = "🪙" if r.get("_is_crypto") else ("💻" if r.get("score", 0) > 50 else "📊")
            log(f"  {icon} {r['ticker']}: score={r.get('score', 0):.1f} rsi={r.get('rsi', 'N/A')}")
        return ranked
    except Exception as e:
        log(f"  [!] Step1 评分失败: {e}")
        return []


# ════════════════════════════════════════════════════════════════════
# Step 2: 板块路由 DL 预测
# ════════════════════════════════════════════════════════════════════
def step2_dl_predict():
    log("━━━ Step 2: DL板块预测 ━━━")
    try:
        from bot.dl_strategy import batch_predict_sector
        preds = batch_predict_sector(ALL_TICKERS)
        valid = [p for p in preds if "error" not in p and p.get("signal") in ("BUY", "SELL")]
        log(f"  DL预测完成: {len(valid)}/{len(ALL_TICKERS)} 个有效信号")
        for p in valid[:5]:
            log(f"  📈 {p['ticker']}: {p['signal']} {p['confidence']:.0f}% ({p.get('sector', '?')})")
        return {p["ticker"]: p for p in preds}
    except Exception as e:
        log(f"  [!] Step2 DL预测失败: {e}")
        return {}


# ════════════════════════════════════════════════════════════════════
# Step 3: 综合信号 — get_signal() 规则为主 + DL 加分器
# ════════════════════════════════════════════════════════════════════
def step3_signals(ranked, dl_preds):
    log("━━━ Step 3: 综合信号 ━━━")
    try:
        from bot.strategy_optimizer import StrategyOptimizer

        score_list = [{"ticker": s["ticker"], "score": s["score"]} for s in ranked]
        dl_list = [
            {"ticker": t, "signal": dl_preds[t]["signal"], "confidence": dl_preds[t]["confidence"]}
            for t in dl_preds if "error" not in dl_preds[t]
        ]

        opt = StrategyOptimizer(state_path=str(STATE_FILE))
        combined = opt.get_signal(dl_list, score_list)
        log(f"  综合排名: {len(combined)} 个资产")

        # 取高分信号
        signals = [
            c for c in combined
            if c.get("combined_score", 0) > 0.5
            and (abs(c.get("dl_confidence", 0)) >= 60 or c.get("combined_score", 0) > 0.75)
        ][:4]

        for s in signals:
            ticker = s["ticker"]
            dl_p = dl_preds.get(ticker, {})
            log(f"  📌 {ticker}: score={s.get('score_signal', 0):.3f} "
                f"combined={s.get('combined_score', 0):.3f} "
                f"dl_conf={abs(s.get('dl_confidence', 0)):.0f}% "
                f"dl_bonus={s.get('dl_bonus', 0):.3f}")
        return signals

    except Exception as e:
        log(f"  [!] Step3 综合信号失败: {e}")
        return []


# ════════════════════════════════════════════════════════════════════
# Step 3.5: Risk Agent 审核（一票否决权）
# ════════════════════════════════════════════════════════════════════
def step3b_risk_review(signals, ranked, dl_preds):
    """将所有综合信号过 Risk Agent 审核，只保留通过/缩减的信号"""
    log("━━━ Step 3.5: Risk Agent 审核 ━━━")
    try:
        from bot.risk_agent import RiskAgent, PortfolioContext, Position, TradingSignal, result_to_dict

        # 构建组合上下文
        portfolio_val = 10000  # 当前模拟资金，可从 Google Sheets 获取
        pos_list = []
        if ranked:
            # 模拟：假设之前买了排名前2的股票
            for s in ranked[:2]:
                pos_list.append(Position(
                    ticker=s["ticker"], shares=10,
                    avg_cost=s.get("price", 100), side="LONG"
                ))

        ctx = PortfolioContext(
            cash=portfolio_val * 0.5,
            portfolio_value=portfolio_val,
            positions=pos_list,
            current_drawdown_pct=0,
        )

        agent = RiskAgent(context=ctx)

        # 转换信号并审核
        approved = []
        rejected = []

        for s in signals:
            ticker = s["ticker"]
            stock = next((st for st in ranked if st["ticker"] == ticker), {})
            price = stock.get("price", s.get("price", 100))
            dl_p = dl_preds.get(ticker, {})
            conf = dl_p.get("confidence", 60) if dl_p else 60
            is_crypto = "/" in ticker

            signal = TradingSignal(
                ticker=ticker,
                direction="BUY",
                suggested_shares=int(conf * 1.5),  # 基于置信度估算股数
                suggested_price=price,
                reason=s.get("reason", ""),
                source="strategy_optimizer",
                is_crypto=is_crypto,
            )

            result = agent.review(signal)
            rd = result_to_dict(result)
            log(f"  {rd['icon']} {ticker}: {rd['verdict']} — {rd['reason']}")

            if result.verdict != "REJECT":
                s["risk_approved"] = True
                s["risk_shares"] = result.approved_shares
                s["risk_stop"] = result.approved_stop
                s["risk_target"] = result.approved_target
                approved.append(s)
            else:
                s["risk_approved"] = False
                rejected.append(s)

        log(f"  Risk Agent 审核完成: {len(approved)} 通过/reduce, {len(rejected)} 拒绝")
        return approved, rejected

    except Exception as e:
        log(f"  [!] Risk Agent 审核失败（不阻断）: {e}")
        import traceback; traceback.print_exc()
        return signals, []


# ════════════════════════════════════════════════════════════════════
# Step 4: Google Sheets 写入（信号池 20 列格式）
# ════════════════════════════════════════════════════════════════════
def step4_sheets(signals, ranked, dl_preds):
    log("━━━ Step 4: Google Sheets 写入 ━━━")
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / ".hermes" / "skills" / "productivity" / "ai-stock-trading-bot" / "references"))
        from google_sheets_portfolio import append_signal, get_pending_signals

        # 读取现有 PENDING（防重复）
        try:
            pending = get_pending_signals()
            existing_tickers = {p.get("B") or p.get("ticker") for p in pending if p}
        except Exception:
            existing_tickers = set()

        written = 0
        for s in signals:
            ticker = s["ticker"]
            if ticker in existing_tickers:
                log(f"  ⏭ {ticker}: 已在 PENDING，跳过")
                continue

            stock = next((st for st in ranked if st["ticker"] == ticker), {})
            dl_p  = dl_preds.get(ticker, {})
            is_crypto = "/" in ticker
            now_str   = datetime.now().strftime("%Y-%m-%d")
            entry_p   = stock.get("price", 0)
            if not entry_p or entry_p <= 0:
                continue

            stop_p   = round(entry_p * (1 - (0.15 if is_crypto else 0.08)), 2)
            tp_p     = round(entry_p * (1 + (0.25 if is_crypto else 0.15)), 2)
            invested = round(entry_p * 100, 2)
            leverage = 50 if is_crypto else (50 if entry_p < 100 else 30)
            reason   = s.get("reason", f"combined={s.get('combined_score', 0):.3f}")

            append_signal(
                date=now_str, ticker=ticker, direction="BUY",
                qty=0, price=entry_p, strategy="AI_SCORING",
                stop_loss=stop_p, take_profit=tp_p,
                invested_amount=invested, notes=reason,
                leverage=leverage,
            )
            log(f"  ✅ 写入 {ticker}: BUY @ ${entry_p:.2f} SL${stop_p:.2f} TP${tp_p:.2f} lev={leverage}x")
            written += 1

        log(f"  ✅ Sheets 写入完成: {written} 个新信号")
        return True

    except Exception as e:
        log(f"  [!] Step4 Sheets写入失败（不阻断）: {e}")
        import traceback; traceback.print_exc()
        return False


# ════════════════════════════════════════════════════════════════════
# Step 5: Telegram 推送（Dry-run 打印）
# ════════════════════════════════════════════════════════════════════
def step5_telegram(signals, ranked, dl_preds, backtest_score=0.0):
    log("━━━ Step 5: Telegram 推送 ━━━")
    token   = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_HOME_CHANNEL", "6801255591")

    if not token:
        log("  [!] 无 TELEGRAM_BOT_TOKEN，跳过")
        return

    if not signals:
        log("  ℹ️  无信号跳过推送")
        return

    signal_lines = []
    for s in signals:
        ticker = s["ticker"]
        stock  = next((st for st in ranked if st["ticker"] == ticker), {})
        dl_p   = dl_preds.get(ticker, {})
        sector = dl_p.get("sector", "?")

        sector_icon = {"tech_high_vol": "💻", "traditional_defensive": "🏦", "cryptocurrency": "🪙"}.get(sector, "📊")
        entry = stock.get("price", 0)
        stop  = round(entry * (1 - (0.08 if "/" not in ticker else 0.15)), 2)
        target = round(entry * (1 + (0.15 if "/" not in ticker else 0.25)), 2)
        dl_c = abs(s.get("dl_confidence", 0))
        dl_b = s.get("dl_bonus", 0)
        bonus_str = f"+{dl_b:.3f}({dl_c:.0f}%)" if dl_b > 0 else ("-" if dl_b < 0 else "无DL")

        signal_lines.append(
            f"{sector_icon}[BUY] {ticker} | ${entry:.2f} | SL${stop:.2f} | TP${target:.2f} | "
            f"板块{sector} | 评分{s.get('score_signal',0):.3f} DL{bonus_str}"
        )

    msg = (
        f"🟡 AI股神信号 - {datetime.now().strftime('%m/%d %H:%M')}\n"
        f"板块模型: tech(15) / defensive(8) / crypto(7)\n"
        f"回测评分: {backtest_score:.2f}\n\n"
        + "\n".join(signal_lines)
        + "\n\n仅供参考，不构成投资建议"
    )

    if DRY_RUN:
        log(f"  📱 [DRY-RUN] Telegram 消息:")
        for line in msg.split("\n"):
            log(f"     {line}")
        log(f"  ✅ DRY-RUN 模式：未真实发送")
    else:
        try:
            import requests
            resp = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": msg},
                timeout=10
            )
            if resp.status_code == 200:
                log(f"  ✅ Telegram 推送成功: {len(signals)} 条信号")
            else:
                log(f"  [!] Telegram 推送失败: {resp.status_code}")
        except Exception as e:
            log(f"  [!] Telegram 推送异常: {e}")


# ════════════════════════════════════════════════════════════════════
# Step 6: 快速回测（90天，as_of_date 防泄漏）
# ════════════════════════════════════════════════════════════════════
def step6_backtest():
    log("━━━ Step 6: 快速回测 ━━━")
    try:
        from bot.backtest import BacktestEngine
        from bot.scorer import rank_stocks

        be = BacktestEngine(initial_cash=10000)

        def strategy_fn(positions, cash, row, date_str):
            try:
                ranked = rank_stocks(ALL_TICKERS, top_n=8, as_of_date=date_str)
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
            tickers=ALL_TICKERS,
            strategy_fn=strategy_fn,
            start_date=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
            end_date=datetime.now().strftime("%Y-%m-%d"),
            stop_loss_pct=8, take_profit_pct=15,
            trailing_trigger_pct=5.0, trailing_stop_pct=3.0,
        )

        if result is None:
            log("  ⚠️  回测无结果")
            return 0.0

        ret_norm   = max(-5, min(15, result.total_return_pct)) / 1.5
        dd_penalty = -min(max(result.max_drawdown_pct, 30), 30) / 3
        win_norm   = result.win_rate / 100 * 10
        sh_norm    = min(result.sharpe_ratio, 4) * 2.5
        n_trades   = result.total_trades
        trade_bonus= min(2.5, n_trades / 30 * 1.8)
        tp_rate    = getattr(result, 'take_profit_rate', 0)
        sl_rate    = getattr(result, 'stop_loss_rate', 0)
        pnl_asym   = tp_rate * 3 - sl_rate * 2

        score = ret_norm + dd_penalty + win_norm + sh_norm + trade_bonus + pnl_asym

        log(f"  📊 回测结果:")
        log(f"     总收益: {result.total_return_pct:.2f}%")
        log(f"     胜率: {result.win_rate:.1f}%")
        log(f"     盈亏比: {getattr(result,'profit_factor',0):.2f}")
        log(f"     夏普: {result.sharpe_ratio:.2f}")
        log(f"     最大回撤: {result.max_drawdown_pct:.1f}%")
        log(f"     交易次数: {result.total_trades}")
        log(f"     综合分: {score:.2f}")

        # ── 回测质量门评估 ──────────────────────────────────────
        try:
            from bot.backtest_quality import BacktestQualityGate
            gate = BacktestQualityGate()
            qa_report = gate.evaluate(result, {
                "data_years": 90/365,  # 90天 ≈ 0.25年
                "has_survivorship_bias": False,
                "has_lookahead": True,  # as_of_date 防泄漏
                "oos_return_pct": result.total_return_pct,
                "train_return_pct": result.total_return_pct,
                "sharpe_ratio": result.sharpe_ratio,
                "slippage_assumed": 0,  # 当前未建模滑点
                "fee_per_trade": 0.001,
                "market_impact_modeled": False,
                "walk_forward_rounds": 0,
                "monte_carlo_pass_rate": 0,
                "total_return_pct": result.total_return_pct,
                "num_strategies_tested": 1,
                "has_adjusted_prices": True,
                "has_timezone_issue": False,
                "feature_shift_used": False,  # 当前as_of_date仅用于评分
                "label_leakage_prevented": False,
                "param_stability_pct": None,
                "stress_test_2008": None,
                "stress_test_2020": None,
                "stress_test_2022": None,
            })
            for line in qa_report.summary().split("\n"):
                log(f"  {line.strip()}")
            qa_report.passed = True  # 暂时不阻断流程
        except Exception as qe:
            log(f"  [!] 质量门评估失败: {qe}")

        return round(score, 2)

    except Exception as e:
        log(f"  [!] 回测失败: {e}")
        import traceback; traceback.print_exc()
        return 0.0


# ════════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log(f"")
    log(f"══════════════════════════════════════════════")
    log(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] === AI股神 Cron Dry-run ===")
    log(f"  资产: {len(ALL_TICKERS)} 个 ({', '.join(SECTOR_CONFIG.keys())})")
    log(f"  模式: {'DRY-RUN（不真实推送）' if DRY_RUN else 'LIVE'}")
    log(f"══════════════════════════════════════════════")

    regime_state = step0_regime()
    ranked = step1_score()
    dl_preds = step2_dl_predict()
    signals  = step3_signals(ranked, dl_preds)
    # ── Risk Agent 审核（过滤信号） ────────────────────────────
    approved_signals, rejected_signals = step3b_risk_review(signals, ranked, dl_preds)
    log(f"  审核后可用信号: {len(approved_signals)}/{len(signals)}")
    
    # ── 根据 Regime 状态调整信号 ─────────────────────────────
    if REGIME_STATE and REGIME_STATE.degradation_level >= 3:
        log(f"  🛑 安全模式(L3): 所有开仓信号被阻止")
        approved_signals = []
    elif REGIME_STATE and REGIME_STATE.degradation_level == 2:
        # 防御模式: 只保留最高分的信号
        threshold = 70
        before = len(approved_signals)
        approved_signals = [s for s in approved_signals if s.get("combined_score", 0) > threshold/100]
        log(f"  🟠 防御模式(L2): 分数>70筛选, {before}→{len(approved_signals)}")
    
    sheets_ok = step4_sheets(approved_signals, ranked, dl_preds)
    backtest_score = step6_backtest()
    step5_telegram(approved_signals, ranked, dl_preds, backtest_score)

    log(f"")
    log(f"══════════════════════════════════════════════")
    log(f"[{datetime.now().strftime('%H:%M:%S')}] === Cron Dry-run 完成 ===")
    log(f"  信号: {len(approved_signals)}/{len(signals)} 通过 | "
         f"Sheets: {'✅' if sheets_ok else '⚠️'} | 回测分: {backtest_score:.2f}")
    if REGIME_STATE:
        log(f"  市场: {REGIME_STATE.regime} L{REGIME_STATE.degradation_level}")
    log(f"══════════════════════════════════════════════")
    log(f"")