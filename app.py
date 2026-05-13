"""
Streamlit Web 界面 — AI 股神（无 Moomoo 依赖版）
路径: ~/projects/ai-stock-bot/app.py
运行: streamlit run ~/projects/ai-stock-bot/app.py --server.port 8501
数据源: yfinance（无需 OpenD）
信号推送: Telegram（用户手动下单参考）
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys, os, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "bot"))

# ====== Google Sheets 持仓模块 ======
SHEETS_REF = os.path.expanduser("~/.hermes/skills/productivity/ai-stock-trading-bot/references")
sys.path.insert(0, SHEETS_REF)
try:
    from google_sheets_portfolio import (
        append_trade, update_summary, get_all_trades,
        get_account_summary, log_trade_from_signal,
        append_signal, get_pending_signals, confirm_to_summary,
        get_summary, update_summary_invested_amount,
        close_signal, get_closed_signals, get_all_signals_for_close,
        sync_confirmed_signals,
        SHEET_ID,
    )
    SHEET_AVAILABLE = True
except Exception as e:
    SHEET_AVAILABLE = False
    append_trade = None
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.hermes/.env"))

st.set_page_config(page_title="AI 股神 · 量化系统", page_icon="📈", layout="wide")

# ---- 模块（去除 Moomoo 依赖） ----
from portfolio import Portfolio
from risk_manager import RiskConfig, RiskManager
from scorer import rank_stocks, score_stock
from monitor import StockMonitor
from backtest import BacktestEngine
from dl_strategy import DLStrategy, batch_predict
from strategy_optimizer import StrategyOptimizer, StrategyParams
from auto_trade import daily_trading_cycle, monitor_and_alert

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PORTFOLIO_PATH = os.path.join(DATA_DIR, "portfolio.json")

# ====== 侧边栏 ======
st.sidebar.title("⚙️ 系统配置")

# 资金
st.sidebar.subheader("💰 资金")
initial_cash = st.sidebar.number_input("初始资金", 100, 1000000, 10000, 500, format="%d")

# 风险管理
st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ 风险管理")
max_position_pct = st.sidebar.slider("单笔最大仓位", 5, 50, 20, 5)
stop_loss_pct = st.sidebar.slider("止损线", 3, 20, 8, 1)
take_profit_pct = st.sidebar.slider("止盈线", 5, 30, 15, 5)
max_positions = st.sidebar.slider("最大同时持仓", 1, 10, 5, 1)
max_holding_days = st.sidebar.slider("最长持有天数", 1, 14, 5, 1)

risk_config = RiskConfig(
    max_single_position_pct=max_position_pct,
    stop_loss_default_pct=stop_loss_pct,
    profit_taking_pct=take_profit_pct,
    max_positions=max_positions,
    max_holding_days=max_holding_days,
)

# 股票池
st.sidebar.markdown("---")
st.sidebar.subheader("📋 股票池")
default_tickers = "NVDA,TSLA,AMD,MSFT,META,AAPL,AMZN,GOOGL,JPM,V,UNH,XOM,JNJ,KO,DIS,NFLX,PLTR,COIN,SOFI"
tickers_input = st.sidebar.text_area("股票代码（逗号/换行分隔）", value=default_tickers, height=120).strip()
WATCHLIST = [t.strip().upper() for t in tickers_input.replace("\n", ",").split(",") if t.strip()]

# AI 策略
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 AI 策略")
use_dl = st.sidebar.checkbox("启用深度学习", value=True)
model_type = st.sidebar.selectbox("DL 模型", ["MLP", "LSTM"], index=0)

# 回测
st.sidebar.markdown("---")
st.sidebar.subheader("📊 回测")
backtest_days = st.sidebar.selectbox("回测区间", [30, 60, 90, 180], index=2)

# 操作按钮
st.sidebar.markdown("---")
col_run1, col_run2 = st.sidebar.columns(2)
run_backtest = col_run1.button("🚀 回测", use_container_width=True)
run_ai_sim = col_run2.button("🤖 AI 信号", use_container_width=True, type="primary")

# Google Sheet 信号池
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Sheet 信号池")
if SHEET_AVAILABLE:
    try:
        pending = get_pending_signals()
        # 自动同步：用户在 Sheet 手动改"已确认=Yes"后自动补到持仓汇总
        sync_result = sync_confirmed_signals()
        if sync_result.get("synced", 0) > 0:
            st.sidebar.success(f"🔄 已同步 {sync_result['synced']} 条信号到持仓汇总")
        elif sync_result.get("skipped", 0) > 0:
            pass  # 静默
        st.sidebar.caption(f"⏳ 待确认信号: {len(pending)} 条")
        st.sidebar.caption(f"Sheet: `…{SHEET_ID[-8:]}`")
    except:
        st.sidebar.caption("Sheet: 连接正常")
    st.sidebar.markdown(f"📎 [打开 Sheet](https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit)")
else:
    st.sidebar.error("⚠️ Sheet 未连接")

# ====== 手动持仓输入（旧版 JSON）======

# ====== 主页面 ======
st.title("📈 AI 股神 · 量化系统")

# 状态栏
status_col1, status_col2, status_col3, status_col4 = st.columns(4)
with status_col1:
    st.success("🟢 系统就绪")
with status_col2:
    st.info(f"股票池: {len(WATCHLIST)} 支")
with status_col3:
    st.caption(f"初始资金: ${initial_cash:,}")
with status_col4:
    st.caption(datetime.now().strftime("更新: %H:%M:%S"))

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 组合概览",
    "🤖 AI 策略",
    "📉 回测报告",
    "🧠 深度学习",
    "📋 信号池",
])

# ====== TAB 1: 组合概览 ======
with tab1:
    col1, col2, col3, col4 = st.columns(4)

    port = Portfolio(initial_cash=initial_cash, json_path=PORTFOLIO_PATH)
    live_prices = {}
    for t in WATCHLIST[:15]:
        try:
            s = score_stock(t)
            if s:
                live_prices[t] = s["price"]
        except:
            pass

    summary = port.portfolio_summary(live_prices)

    with col1:
        st.metric("💰 初始资金", f"${summary['initial_cash']:,.0f}")
    with col2:
        st.metric("💵 可用电额", f"${summary['cash']:,.2f}")
    with col3:
        delta = summary["total_pnl"]
        st.metric("📈 总盈亏", f"{'+' if delta >= 0 else ''}{delta:,.2f}")
    with col4:
        pct = summary["total_pnl_pct"]
        st.metric("📊 收益率", f"{'+' if pct >= 0 else ''}{pct:.2f}%")

    st.markdown("---")

    if summary["positions"]:
        pos_df = pd.DataFrame(summary["positions"])
        pos_display = pos_df.copy()
        pos_display["market_value"] = pos_display["market_value"].apply(lambda x: f"${x:,.2f}")
        pos_display["pnl"] = pos_display["pnl"].apply(lambda x: f"{'+' if x >= 0 else ''}{x:.2f}")
        pos_display["pnl_pct"] = pos_display["pnl_pct"].apply(lambda x: f"{'+' if x >= 0 else ''}{x:.2f}%")
        st.dataframe(pos_display, use_container_width=True, hide_index=True)
    else:
        st.info("📭 空仓 — 去「AI 策略」生成交易信号")

    if summary["positions"] and len(summary["positions"]) > 0:
        fig = go.Figure()
        tickers = [p["ticker"] for p in summary["positions"]]
        pnl_vals = [p["pnl"] for p in summary["positions"]]
        colors = ["#00D084" if p >= 0 else "#FF4757" for p in pnl_vals]
        fig.add_trace(go.Bar(x=tickers, y=pnl_vals, marker_color=colors))
        fig.update_layout(title="📊 持仓盈亏", height=250, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    if port.trades:
        st.subheader("📋 历史交易")
        trades_df = pd.DataFrame([
            {"时间": t.date, "操作": t.action, "股票": t.ticker,
             "价格": f"${t.price:.2f}", "股数": t.shares,
             "盈亏": f"{'+' if t.pnl >= 0 else ''}{t.pnl:.2f}"}
            for t in port.trades[-20:]
        ])
        st.dataframe(trades_df, use_container_width=True, hide_index=True)

    # 手动持仓更新弹窗
# ====== TAB 2: AI 策略 ======
with tab2:
    st.subheader("🤖 AI 交易信号")

    if run_ai_sim:
        with st.spinner("🤖 AI 分析 + 生成信号..."):
            ranked = rank_stocks(WATCHLIST, top_n=10)

            dl_signals = []
            if use_dl:
                try:
                    dl_signals = batch_predict([s["ticker"] for s in ranked[:5]], model_type=model_type)
                except Exception as e:
                    st.warning(f"DL 预测失败: {e}")

            optimizer = StrategyOptimizer(state_path=os.path.join(DATA_DIR, "strategy_state.json"))
            combined = optimizer.get_signal(dl_signals, ranked)

            col_left, col_right = st.columns([1, 1])
            with col_left:
                st.markdown("### 📊 市场评分 Top 8")
                for i, s in enumerate(ranked, 1):
                    medal = ["🥇", "🥈", "🥉"]
                    icon = medal[i-1] if i <= 3 else f"{i}."
                    st.markdown(f"{icon} **{s['ticker']}** — {s['name']}")
                    st.caption(f"   PE={s['pe']} | 5日{s['change_5d_pct']}% | 量{s['volume_ratio']}x | {', '.join(s['reasons'][:2])}")

            with col_right:
                st.markdown("### 🧠 综合信号排名")
                for i, c in enumerate(combined[:8], 1):
                    dl_ind = ""
                    for d in dl_signals:
                        if d["ticker"] == c["ticker"]:
                            dl_ind = f" | DL:{d['signal']}({d['confidence']}%)"
                    st.markdown(f"{i}. **{c['ticker']}** | 综合:{c['combined_score']:.2f}{dl_ind}")

            st.markdown("---")
            st.markdown("### 📋 信号（请手动在 MITRADE 下单）")

            from auto_trade import generate_signals
            signals = generate_signals(ranked, dl_signals, initial_cash, max_position_pct, risk_config)

            if signals:
                for sig in signals:
                    emoji = {"BUY": "🟢", "SHORT": "🔴", "SELL": "🔴", "HOLD": "🟡"}.get(sig["action"], "⚪")
                    pos_type = sig.get("position_type", "LONG")
                    direction = "做多" if pos_type == "LONG" else "做空"
                    lev = sig.get("leverage", 0)
                    lev_display = f" | ⚡{lev:.0f}x" if lev else ""
                    st.success(
                        f"{emoji} [{sig['action']}] **{sig['ticker']}** "
                        f"{direction} {sig['qty']}股 @ ${sig['price']:.2f}{lev_display}\n"
                        f"   止损: ${sig.get('stop_loss','?')} | 止盈1: ${sig.get('take_profit_1','?')} | "
                        f"理由: {sig['reason']}"
                    )
                    # 自动记录到 Google Sheet 信号池
                    if SHEET_AVAILABLE and sig["action"] in ("BUY", "SELL"):
                        try:
                            log_trade_from_signal(
                                signal=sig,
                                live_price=sig["price"],
                                stop_loss=sig.get("stop_loss", 0),
                                take_profit=sig.get("take_profit_1", 0),
                                strategy="AI_SCORING",
                            )
                            st.toast(f"📝 已写入信号池: {sig['ticker']}", icon="✅")
                        except Exception as sheet_err:
                            st.warning(f"Sheet 写入失败: {sheet_err}")
                st.info("📌 以上信号已推送 Telegram，请在 MITRADE 手动下单执行")
            else:
                st.info("暂无可执行信号 — 市场条件不满足")

    else:
        st.info("👈 点击「🤖 AI 信号」开始分析")
        st.caption("综合市场评分 + 深度学习预测 → 生成 Telegram 信号推送（用户手动下单）")

# ====== TAB 3: 回测 ======
with tab3:
    st.subheader("📉 历史回测")

    if run_backtest:
        with st.spinner(f"📥 下载 {backtest_days} 天历史数据 + 回测..."):
            be = BacktestEngine(initial_cash=initial_cash)
            def strategy_fn(positions, cash, row, date_str):
                return []
            result = be.run(
                tickers=WATCHLIST,
                strategy_fn=strategy_fn,
                start_date=(datetime.now() - timedelta(days=backtest_days)).strftime("%Y-%m-%d"),
                end_date=datetime.now().strftime("%Y-%m-%d"),
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
            )

        if result:
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.metric("初始资金", f"${result.initial_cash:,.0f}")
            with r2:
                st.metric("最终价值", f"${result.final_value:,.2f}")
            with r3:
                pnl = result.total_return_pct
                st.metric("总收益", f"{'+' if pnl >= 0 else ''}{pnl:.2f}%")
            with r4:
                st.metric("最大回撤", f"{result.max_drawdown_pct:.2f}%")

            r5, r6, r7 = st.columns(3)
            with r5:
                st.metric("Sharpe", f"{result.sharpe_ratio:.2f}")
            with r6:
                st.metric("交易次数", f"{result.total_trades}")
            with r7:
                st.metric("胜率", f"{result.win_rate}%")

            st.markdown("---")
            st.text(be.summary_text(result))

            if result.trades:
                sell_df = pd.DataFrame([t for t in result.trades if t["action"] == "SELL"])
                if len(sell_df) > 0:
                    sell_df["return_pct"] = sell_df["return_pct"].apply(lambda x: f"{'+' if x >= 0 else ''}{x:.2f}%")
                    st.subheader("📋 交易明细")
                    st.dataframe(sell_df[["date", "ticker", "action", "reason", "price", "pnl", "return_pct"]], use_container_width=True, hide_index=True)
    else:
        st.info("👈 点击「🚀 回测」开始回测")

# ====== TAB 4: 深度学习 ======
with tab4:
    st.subheader("🧠 深度学习训练与预测")

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        train_ticker = st.selectbox("选择股票", WATCHLIST[:10], index=0)
        train_btn = st.button("🔥 训练模型", type="primary")

        if train_btn:
            with st.spinner(f"🔥 训练 {train_ticker} ({model_type})..."):
                try:
                    dl = DLStrategy(train_ticker, model_type=model_type)
                    result = dl.train(epochs=30)
                    if "error" not in result:
                        st.success(f"✅ 训练完成！验证准确率: {result['best_val_acc']:.1%}")
                        st.caption(f"模型: {result['model_path']} | 日期: {result['latest_train_date']}")
                    else:
                        st.error(f"失败: {result['error']}")
                except Exception as e:
                    st.error(f"错误: {e}")

        st.markdown("### 🔮 批量预测")
        predict_btn = st.button("🔮 预测所有股票")
        if predict_btn:
            with st.spinner("🔮 DL 批量预测..."):
                preds = batch_predict(WATCHLIST[:10], model_type=model_type)
                for p in preds:
                    if "error" in p:
                        st.error(f"{p['ticker']}: {p['error']}")
                    else:
                        sig_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(p["signal"], "⚪")
                        st.markdown(f"{sig_emoji} {p['ticker']} | {p['signal']} ({p['confidence']}% 信心) | 涨{p['prob_up']}% / 跌{p['prob_down']}%")

    with col_dl2:
        st.markdown("### ⚙️ 策略参数")
        opt = StrategyOptimizer(state_path=os.path.join(DATA_DIR, "strategy_state.json"))
        st.json(opt.params.to_dict())

        if opt.history:
            st.markdown("**历史调整（最近5次）：**")
            for rec in opt.history[-5:]:
                color = "🟢" if rec.composite_score > 0 else "🔴"
                st.markdown(f"{color} {rec.date} | 综合:{rec.composite_score:.2f} | 收益:{rec.backtest_result.get('return','?')}% | DD:{rec.backtest_result.get('max_drawdown','?')}%")

        st.markdown("**权重配置：**")
        dl_w = st.slider("DL 信号权重", 0.0, 1.0, opt.params.dl_weight, 0.05, key="dl_w")
        sc_w = st.slider("评分信号权重", 0.0, 1.0, opt.params.score_weight, 0.05, key="sc_w")

    st.markdown("---")
    st.caption("🧠 技术指标(RSI/MACD/布林带) + LSTM/MLP 预测次日涨跌 | 每周自动重训练")

# ====== TAB 5: 信号池 + 持仓汇总 ======
with tab5:
    st.subheader("📋 AI 信号池 / 持仓汇总")

    sub_tab_signals, sub_tab_summary, sub_tab_history = st.tabs([
        "📨 待确认信号",
        "📊 持仓汇总",
        "📜 历史平仓",
    ])

    with sub_tab_signals:
        if not SHEET_AVAILABLE:
            st.error("⚠️ Sheet 未连接")
        else:
            # 读取待确认信号
            pending = get_pending_signals()
            if not pending:
                st.info("📭 信号池暂无待确认信号 — 点击「🤖 AI 信号」生成")
            else:
                st.caption(f"共 {len(pending)} 条待确认信号，勾选后录入持仓汇总")
                confirmed = []
                for p in pending:
                    lev = p.get("leverage", 0)
                    lev_str = f" ⚡{lev:.0f}x" if lev else ""
                    checked = st.checkbox(
                        f"**{'🟢' if p['direction']=='BUY' else '🔴'} {p['ticker']}** "
                        f"{p['direction']} × {p['qty']} @ ${p['price']:.2f}{lev_str} "
                        f"| 策略:{p['strategy']} | {p['date']} "
                        f"| 止损:${p.get('stop_loss','-')} 止盈:${p.get('take_profit','-')}",
                        key=f"sig_{p['row']}",
                    )
                    if checked:
                        confirmed.append(p)

                if confirmed:
                    if st.button("✅ 确认录入持仓汇总", type="primary"):
                        for p in confirmed:
                            try:
                                r = confirm_to_summary(p["row"])
                                st.success(f"✅ {r['ticker']} 已录入持仓汇总")
                            except Exception as e:
                                st.error(f"❌ {p['ticker']}: {e}")
                        st.rerun()

                st.markdown("---")
                st.caption("💡 录入持仓汇总后，可修正投入金额（见「持仓汇总」 Tab）")

    with sub_tab_summary:
        if not SHEET_AVAILABLE:
            st.error("⚠️ Sheet 未连接")
        else:
            import yfinance
            # 先拿持仓列表，再抓实时价格
            positions_no_price = get_summary()
            tickers = [p["ticker"] for p in positions_no_price]

            live_prices = {}
            for t in tickers:
                try:
                    s = yfinance.Ticker(t).fast_info
                    live_prices[t] = s.get("regularPrice", 0) or 0
                except:
                    pass

            positions = get_summary(live_prices=live_prices)
            if not positions:
                st.info("📭 持仓汇总为空 — 从「待确认信号」勾选录入")
            else:
                total_pnl = sum(p["pnl"] for p in positions)
                total_pnl_pct = sum(p["pnl_pct"] * p["qty"] for p in positions) / sum(p["qty"] for p in positions) if sum(p["qty"] for p in positions) > 0 else 0

                k1, k2, k3 = st.columns(3)
                k1.metric("📊 总盈亏", f"{'+' if total_pnl>=0 else ''}{total_pnl:.2f}")
                k2.metric("📈 平均收益率", f"{'+' if total_pnl_pct>=0 else ''}{total_pnl_pct:.2f}%")
                k3.metric("🗂️ 持仓数", f"{len(positions)}")
                st.markdown("---")

                # 可编辑投入金额
                st.markdown("**✏️ 修正投入金额**（在 Sheet 直接修改 M 列也有效）")
                for p in positions:
                    lev = p.get("leverage", 0)
                    lev_str = f" ⚡{lev:.0f}x" if lev else ""
                    with st.expander(f"**{'🟢' if p['direction']=='BUY' else '🔴'} {p['ticker']}**{lev_str} — {p['qty']} 股 @ ${p['avg_cost']:.2f} | 现价 ${p['live_price']:.2f} | {'+' if p['pnl']>=0 else ''}{p['pnl_pct']:.2f}% ({'+' if p['pnl']>=0 else ''}{p['pnl']:.2f})"):
                        current_invested = p["invested"]
                        new_invested = st.number_input(
                            "投入金额 (USD)",
                            min_value=0.0,
                            value=float(current_invested),
                            step=50.0,
                            key=f"inv_{p['ticker']}",
                        )
                        if new_invested != current_invested:
                            try:
                                update_summary_invested_amount(p["ticker"], new_invested)
                                st.success(f"✅ {p['ticker']} 投入金额已更新: ${new_invested:.2f}")
                            except Exception as e:
                                st.error(f"更新失败: {e}")
                        st.caption(
                            f"方向:{p['direction']} | 数量:{p['qty']} | "
                            f"平均成本:${p['avg_cost']:.2f} | 现价:${p['live_price']:.2f} | "
                            f"止损:${p.get('stop_loss') or '-'} | 止盈:${p.get('take_profit') or '-'} | "
                            f"建仓:{p.get('open_date','-')} | {p.get('notes','')}"
                        )
                        # 平仓按钮
                        col_close, col_price = st.columns([1, 2])
                        with col_close:
                            do_close = st.button(f"🔚 平仓", key=f"close_{p['ticker']}")
                        with col_price:
                            close_price_val = st.number_input(
                                "平仓价格", value=float(p["live_price"]), step=1.0,
                                key=f"cp_{p['ticker']}"
                            )
                        if do_close:
                            try:
                                # 用 get_all_signals_for_close 找对应信号的行号（含已确认的）
                                all_sigs = get_all_signals_for_close(p["ticker"])
                                if not all_sigs:
                                    st.warning(f"信号池中未找到 {p['ticker']} 对应信号，请手动在Sheet中标记平仓")
                                else:
                                    row_to_close = all_sigs[0]["row"]  # 取最新一条
                                    cr = close_signal(row_to_close, float(close_price_val))
                                    st.success(
                                        f"🔚 {p['ticker']} 已平仓 | "
                                        f"开${cr['open_price']:.2f} → 平${cr['close_price']:.2f} | "
                                        f"{'+' if cr['pnl']>=0 else ''}{cr['pnl']:.2f} ({cr['pnl_pct']:.2f}%) | "
                                        f"持仓{cr['holding_days']}天"
                                    )
                                    st.rerun()
                            except Exception as e:
                                st.error(f"平仓失败: {e}")

    with sub_tab_history:
        if not SHEET_AVAILABLE:
            st.error("⚠️ Sheet 未连接")
        else:
            closed = get_closed_signals()
            if not closed:
                st.info("📭 暂无历史平仓记录 — 确认信号后会移动到持仓汇总，平仓后记录会显示在这里")
            else:
                # 统计摘要
                total_pnl = sum(c["pnl"] for c in closed)
                win_count = sum(1 for c in closed if c["pnl"] > 0)
                win_rate = win_count / len(closed) * 100 if closed else 0
                avg_days = sum(c["holding_days"] for c in closed) / len(closed) if closed else 0

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("📜 总交易", f"{len(closed)} 笔")
                m2.metric("🎯 胜率", f"{win_rate:.0f}%")
                m3.metric("⏳ 平均持仓", f"{avg_days:.1f} 天")
                m4.metric("💰 总盈亏", f"{'+' if total_pnl >= 0 else ''}{total_pnl:.2f}")
                st.markdown("---")

                st.markdown("**📋 历史平仓明细**（按平仓日期倒序）")
                for c in closed:
                    emoji = "🟢" if c["pnl"] >= 0 else "🔴"
                    pnl_disp = f"{'+' if c['pnl'] >= 0 else ''}{c['pnl']:.2f}"
                    pnl_pct_disp = c.get("pnl_pct_str", "0%")
                    lev = c.get("leverage", 0)
                    lev_str = f" ⚡{lev:.0f}x" if lev else ""
                    st.markdown(
                        f"{emoji} **{c['ticker']}**{lev_str} "
                        f"{c['direction']} | 开${c['open_price']:.2f} → 平${c['close_price']:.2f} "
                        f"| {pnl_disp} ({pnl_pct_disp}) | 持仓{c['holding_days']}天 "
                        f"| {c['strategy']} | 平仓:{c['close_date']}"
                    )
                st.caption(f"共 {len(closed)} 条记录 | 数据来源: Google Sheet 信号池 O~S列")

st.markdown("---")
st.caption("⚠️ 仅供参考，不构成投资建议 | 数据来源: yfinance | 信号通过 Telegram 推送")