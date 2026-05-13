"""
主程序：跑评分 → AI 分析 → Telegram 推送
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from scorer import rank_stocks
from ai_analyzer import generate_report
from telegram_bot import send_report
from config import WATCHLIST
from datetime import datetime


def build_text_report(ranked: list[dict]) -> str:
    """构建纯文本格式的报告（不需要 AI）"""
    lines = [
        f"📊 今日股票评分报告 — {datetime.now().strftime('%Y-%m-%d')}",
        "",
        f"🏆 Top {len(ranked)} 选股：",
    ]
    medal = ["🥇", "🥈", "🥉"]
    for i, s in enumerate(ranked):
        m = medal[i] if i < 3 else f"{i+1}."
        reasons_str = ", ".join(s["reasons"]) if s["reasons"] else "数据不足"
        lines.append(
            f"{m} {s['ticker']} ({s['name']}) — 分数: {s['score']}/75\n"
            f"   原因: {reasons_str}\n"
            f"   价格: ${s['price']} | PE: {s['pe']} | β: {s['beta']} | 5日: {s['change_5d_pct']}%"
        )
    lines.extend([
        "",
        "📋 完整数据：",
    ])
    for s in ranked:
        lines.append(
            f"{s['ticker']} | ${s['price']} | PE={s['pe']} | "
            f"5d={s['change_5d_pct']}% | vol={s['volume_ratio']}x | "
            f"β={s['beta']} | ${s['market_cap_B']}B"
        )
    lines.extend([
        "",
        "⚠️ 仅供参考，不构成投资建议。",
    ])
    return "\n".join(lines)


def main():
    print(f"⏰ [{datetime.now().strftime('%H:%M:%S')}] 开始股票分析...")

    # 1. 评分排名
    print("📊 评分排名中...")
    ranked = rank_stocks(WATCHLIST, top_n=8)
    print(f"   完成，共 {len(ranked)} 支股票")

    # 2. 构建文字报告
    text_report = build_text_report(ranked)
    print("\n" + text_report[:500] + "...")

    # 3. AI 分析（如果 API 可用）
    print("\n🤖 生成 AI 分析...")
    ai_report = generate_report(ranked, WATCHLIST)
    print(ai_report[:300])

    # 4. 合并报告推送到 Telegram
    full_report = f"{text_report}\n\n{'='*40}\n\n🤖 AI 点评：\n{ai_report}"
    print("\n📱 推送到 Telegram...")
    success = send_report(full_report)

    if success:
        print("✅ 完成！")
    else:
        print("⚠️ Telegram 推送失败（缺少 Token/Chat ID）")
        print("\n要启用推送，设置环境变量：")
        print("  TELEGRAM_BOT_TOKEN=你的机器人token")
        print("  TELEGRAM_CHAT_ID=你的 Telegram ID")


if __name__ == "__main__":
    main()