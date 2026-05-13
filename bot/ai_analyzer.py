"""
AI 分析层：调用 MiniMax 分析选股理由
"""
import os, json, requests
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.hermes/.env"))

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/anthropic")

MODEL = "MiniMax-M2.7"  # 或 "MiniMax-M2"


def analyze_with_ai(stocks: list[dict], watchlist: list[str]) -> str:
    """
    把筛选结果喂给 AI，生成选股报告
    """
    if not stocks:
        # 没有符合条件的股票时，AI 也生成一份观察名单分析
        body = f"""你是一个股票分析助手。本周筛选没有找到同时满足条件的股票（PE<40、近5日涨超3%、成交量放大1.2x）。

请从以下观察名单中，挑选你认为值得关注的3-5支，说明理由：
{', '.join(watchlist)}

输出格式（中文）：
📊 今日选股报告

🔍 观察名单精选：
1. [ticker] - 原因
2. [ticker] - 原因

⚠️ 风险提示：仅供参考，不构成投资建议。
"""
    else:
        # 构建股票表格
        lines = []
        for s in stocks:
            lines.append(
                f"- {s['ticker']} ({s['name']}): "
                f"现价${s['price']:.2f}, PE={s['pe']}, "
                f"5日涨幅{s['change_5d_pct']}%, "
                f"成交量比{s['volume_ratio']}x, "
                f"营收增长{s.get('revenue_growth_pct')}%, "
                f"负债率{s.get('debt_ratio')}, "
                f"beta={s.get('beta')}"
            )

        stock_table = "\n".join(lines)

        body = f"""你是一个股票分析助手。本周技术面筛选出以下股票：

{stock_table}

请做以下分析（中文输出）：
1. 📈 排序并选出最值得关注的3支，给出简短理由
2. 💡 每支股票的核心逻辑（趋势/价值/催化剂）
3. ⚠️ 风险提示

输出格式：
📊 今日 AI 选股报告

🥇 Top Picks:
1. [ticker] — 理由
2. [ticker] — 理由
3. [ticker] — 理由

📋 完整筛选数据：
[表格]

⚠️ 风险提示：仅供参考，不构成投资建议。
"""

    # 调用 MiniMax
    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    payload = {
        "model": MODEL,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": body}]
    }

    try:
        resp = requests.post(
            f"{MINIMAX_BASE_URL}/v1/messages",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        # MiniMax 返回: content=[{"type":"thinking","thinking":"..."}, {"type":"text","text":"..."}]
        content = data.get("content", [])
        # 找 text 类型的 block
        text_blocks = [c.get("text", "") for c in content if c.get("type") == "text"]
        return "\n".join(text_blocks) if text_blocks else "（AI 未返回有效内容）"
    except Exception as e:
        return f"❌ AI 分析失败: {e}\n\n原始数据：\n{body[:500]}"


def generate_report(stocks: list[dict], watchlist: list[str]) -> str:
    """生成完整报告（筛选数据 + AI 分析）"""
    report = analyze_with_ai(stocks, watchlist)
    return report