"""
每日选股报告（新版本）
- 生成 BUY 信号：买入价 + 止损价 + 止盈价
- 格式化为友好消息推送 Telegram
- 不再用旧 run.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.hermes/.env"))

from config import WATCHLIST, SCREENER_CONFIG
from ai_recommender import get_all_signals, format_signal
from scorer import rank_stocks
from telegram_bot import send_report


def build_report():
    date = datetime.now().strftime("%Y-%m-%d")

    # 生成所有信号（本地计算，不走 AI）
    signals = get_all_signals(
        tickers=WATCHLIST,
        stop_loss_pct=8.0,
        profit_target_pct=15.0,
        use_ai=False,
    )

    # 过滤 BUY 信号
    buy_signals = [s for s in signals if s["action"] == "BUY"]
    watch_signals = [s for s in signals if s["action"] == "WATCH"]

    # 市场评分
    try:
        ranked = rank_stocks(WATCHLIST, top_n=6)
    except:
        ranked = []

    lines = [
        f"📈 *AI 股神每日选股报告*",
        f"🗓 {date} {datetime.now().strftime('%H:%M')}",
        f"━━━━━━━━━━━━━━━━━━",
    ]

    # BUY 信号
    if buy_signals:
        lines.append(f"🟢 *精选买入信号 ({len(buy_signals)} 个)*\n")
        for i, sig in enumerate(buy_signals, 1):
            lines.append(
                f"{i}. *{sig['ticker']}* ({sig.get('name', '')})"
            )
            # 加密货币显示更多小数位（5位），股票/外汇保持2位
            is_crypto = "/" in sig.get("ticker", "")
            price_fmt = ".5f" if is_crypto else ".2f"

            lines.append(
                f"   💰 买入价: ${sig['buy_price']:{price_fmt}}"
            )
            lines.append(
                f"   🛑 止损价: ${sig['stop_loss']:{price_fmt}} ({sig.get('adjusted_stop_loss_pct', 8.0):.1f}% 止损)"
            )
            lines.append(
                f"   🎯 止盈1: ${sig['take_profit_1']:{price_fmt}}"
            )
            if sig.get('trailing_stop_points') is not None:
                lines.append(
                    f"   📍 追踪止损: {sig['trailing_stop_points']} 点（TP1 激活后从最高点回撤触发）"
                )
            pos = sig.get('position_size', 10)
            pe = sig.get('pe', '?')
            lines.append(
                f"   📊 仓位: {pos}% | PE: {pe} | {sig.get('entry_reason', '')}"
            )
            if sig.get('rsi') is not None:
                lines.append(f"   📈 RSI: {sig['rsi']:.1f}")
            lines.append("")  # 空行分隔

    else:
        lines.append("\n🟡 今日暂无 BUY 信号（市场条件不满足）")

    # WATCH 信号
    if watch_signals:
        lines.append(f"🟡 *观望 ({len(watch_signals)} 个)*")
        for sig in watch_signals[:4]:
            lines.append(
                f"• {sig['ticker']} — {sig.get('entry_reason', '观望')}"
            )

    # 市场情绪（简版）
    if ranked:
        lines.append(f"\n📊 *市场评分 Top 3*")
        medals = ["🥇", "🥈", "🥉"]
        for i, s in enumerate(ranked[:3], 0):
            medal = medals[i] if i < 3 else f"{i+1}."
            lines.append(
                f"{medal} {s['ticker']} — {s['name']}"
                f" | PE={s['pe']} | 5日{s['change_5d_pct']}% | 量{s['volume_ratio']}x"
            )

    lines.append(f"\n⚠️ 仅供参考，不构成投资建议")
    lines.append(f"🔧 止损/止盈已按 Beta 波动率调整")

    return "\n".join(lines)


def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 生成每日选股报告...")
    report = build_report()
    print(report)
    print("\n--- 发送到 Telegram ---")

    chat_id = os.getenv("TELEGRAM_HOME_CHANNEL") or os.getenv("TELEGRAM_CHAT_ID") or "6801255591"
    ok = send_report(report, chat_id=chat_id)
    if ok:
        print("✅ 发送成功")
    else:
        print("❌ 发送失败（检查 Telegram 配置）")

    return report


if __name__ == "__main__":
    main()