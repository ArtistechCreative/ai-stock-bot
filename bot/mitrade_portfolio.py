"""
MITRADE 持仓记录 — Notion API 管理工具
用法:
  python3 mitrade_portfolio.py list          # 查看所有持仓
  python3 mitrade_portfolio.py add NVDA LONG 2 220.00 228.50 --lev 5 --sl 200.00 --tp 260.00
  python3 mitrade_portfolio.py update NVDA --cur 235.00
  python3 mitrade_portfolio.py close NVDA --reason "止盈平仓"
  python3 mitrade_portfolio.py refresh NVDA   # 从 yfinance 更新当前价
"""
import os, sys, json, datetime, re
import urllib.request, urllib.error
from pathlib import Path

# ===== Config =====
NOTION_KEY = os.popen("grep NOTION_API_KEY ~/.hermes/.env | cut -d= -f2").read().strip()
MAIN_PAGE_ID = "35e7ed2e-a0a0-81aa-85ac-d514d475a183"  # MITRADE 持仓记录 page

# ===== Notion API helpers =====
def notion_req(path, payload=None, method="GET"):
    url = f"https://api.notion.com/v1{path}"
    headers = {
        "Authorization": f"Bearer {NOTION_KEY}",
        "Notion-Version": "2025-09-03",
        "Content-Type": "application/json; charset=utf-8"
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode() if payload else None, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err = json.loads(e.read().decode())
        raise Exception(f"Notion API {e.code}: {err.get('message', err)}")

def search_pages(query):
    result = notion_req("/search", {"query": query, "filter": {"property": "object", "value": "page"}, "page_size": 5})
    return result.get("results", [])

def create_page(parent_id, title, children=None):
    payload = {
        "parent": {"page_id": parent_id},
        "properties": {"title": {"title": [{"text": {"content": title}}]}},
        "children": children or []
    }
    return notion_req("/pages", payload, "POST")

def append_blocks(page_id, blocks):
    """Append blocks to a page"""
    payload = {"children": blocks}
    return notion_req(f"/blocks/{page_id}/children", payload, "PATCH")

def get_page_blocks(page_id):
    """Get all blocks in a page"""
    return notion_req(f"/blocks/{page_id}/children?page_size=50")

def update_page_properties(page_id, properties):
    """Update page properties"""
    payload = {"properties": properties}
    return notion_req(f"/pages/{page_id}", payload, "PATCH")

# ===== Portfolio Operations =====
def create_position_page(ticker, direction, qty, entry_price, current_price,
                         leverage=5, stop_loss=None, take_profit=None,
                         note="", status="持仓中"):
    """Create a new position as a child page"""
    pnl = (current_price - entry_price) * qty if "LONG" in direction else (entry_price - current_price) * qty
    pnl_pct = (pnl / (entry_price * qty / leverage)) * 100 if leverage > 0 else 0
    contract_val = qty * current_price
    margin = contract_val / leverage if leverage > 0 else 0

    emoji = "🟢" if "LONG" in direction else "🔴"
    title = f"{emoji} {ticker} {direction.split()[0]} | {datetime.date.today().isoformat()}"

    children = [
        {"object": "block", "type": "heading_2", "heading_2": {"rich_text": [{"text": {"content": "📊 持仓详情"}}], "color": "default"}},
        {"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"text": {"content": f"标的代码: {ticker}"}}]}},
        {"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"text": {"content": f"方向: {direction} | 杠杆: {leverage}x"}}]}},
        {"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"text": {"content": f"持仓数量: {qty} | 开仓价: ${entry_price:.2f} | 当前价: ${current_price:.2f}"}}]}},
        {"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"text": {"content": f"合约价值: ${contract_val:.2f} | 保证金: ${margin:.2f}"}}]}},
        {"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"text": {"content": f"{'🟢' if pnl >= 0 else '🔴'} 浮盈亏: {'+' if pnl >= 0 else ''}${pnl:.2f} ({'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%)"}}]}},
        {"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"text": {"content": f"止损: ${stop_loss:.2f} | 止盈: ${take_profit:.2f}"}}]}},
        {"object": "block", "type": "callout", "callout": {"rich_text": [{"text": {"content": note or "AI信号买入 | 模拟账户"}}], "icon": {"emoji": "🤖"}, "color": "blue_background"}},
        {"object": "block", "type": "divider", "divider": {}},
        {"object": "block", "type": "heading_2", "heading_2": {"rich_text": [{"text": {"content": "📋 交易记录"}}], "color": "default"}},
        {"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"text": {"content": f"开仓时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"}}]}},
        {"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"text": {"content": f"状态: {status}"}}]}},
    ]

    page = create_page(MAIN_PAGE_ID, title, children)
    return page

def list_positions():
    """List all position pages under main page"""
    blocks = get_page_blocks(MAIN_PAGE_ID)
    positions = []
    for block in blocks.get("results", []):
        if block.get("type") == "child_page":
            child = block.get("child_page", {})
            title = child.get("title", "")
            page_id = block.get("id", "")
            # Parse title: "🟢 NVDA LONG | 2026-05-12"
            if any(x in title for x in ["LONG", "SHORT", "多", "空"]):
                positions.append({"title": title, "page_id": page_id})
    return positions

def update_position_current_price(page_id, ticker, current_price, entry_price, qty, leverage, direction):
    """Update the current price in a position page by appending a new block"""
    pnl = (current_price - entry_price) * qty if "LONG" in direction else (entry_price - current_price) * qty
    pnl_pct = (pnl / (entry_price * qty / leverage)) * 100 if leverage > 0 else 0

    update_block = {
        "object": "block",
        "type": "callout",
        "callout": {
            "rich_text": [{"text": {"content": f"📈 价格更新 {datetime.datetime.now().strftime('%H:%M')}: ${current_price:.2f} | 浮盈亏: {'+' if pnl >= 0 else ''}${pnl:.2f} ({'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%)"}}],
            "icon": {"emoji": "🔄"},
            "color": "green_background" if pnl >= 0 else "red_background"
        }
    }
    append_blocks(page_id, [update_block])

# ===== CLI =====
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "list":
        print("📋 当前持仓:")
        positions = list_positions()
        if not positions:
            print("  （空仓）")
        for p in positions:
            print(f"  {p['title']}")
            print(f"    ID: {p['page_id']}")

    elif cmd == "add":
        # python3 mitrade_portfolio.py add NVDA LONG 2 220.00 228.50 --lev 5 --sl 200.00 --tp 260.00 --note "AI信号"
        ticker = sys.argv[2]
        direction = sys.argv[3]
        qty = float(sys.argv[4])
        entry = float(sys.argv[5])
        cur = float(sys.argv[6])

        args = sys.argv[7:]
        kwargs = {}
        i = 0
        while i < len(args):
            if args[i] == "--lev": kwargs["leverage"] = float(args[i+1]); i += 2
            elif args[i] == "--sl": kwargs["stop_loss"] = float(args[i+1]); i += 2
            elif args[i] == "--tp": kwargs["take_profit"] = float(args[i+1]); i += 2
            elif args[i] == "--note": kwargs["note"] = args[i+1]; i += 2
            else: i += 1

        page = create_position_page(ticker, direction, qty, entry, cur, **kwargs)
        print(f"✅ 持仓已创建: {page.get('id')}")
        print(f"   URL: {page.get('url','')}")

    elif cmd == "update":
        # python3 mitrade_portfolio.py update NVDA --cur 235.00
        ticker = sys.argv[2]
        args = sys.argv[3:]
        cur_price = None
        i = 0
        while i < len(args):
            if args[i] == "--cur": cur_price = float(args[i+1]); i += 2
            else: i += 1

        if not cur_price:
            print("❌ 需要提供 --cur 参数")
            sys.exit(1)

        positions = list_positions()
        target = [p for p in positions if ticker.upper() in p["title"]]
        if not target:
            print(f"❌ 未找到 {ticker} 的持仓页面")
            sys.exit(1)

        page_id = target[0]["page_id"]
        # For simplicity, just log update - in production you'd parse entry price from the page
        print(f"📝 更新 {ticker} 当前价: ${cur_price}")

        # We need entry_price, qty etc to calculate P&L - parse from page blocks
        blocks = get_page_blocks(page_id)
        entry_price = qty = leverage = direction = None
        for b in blocks.get("results", []):
            t = b.get("type","")
            txt = b.get(t,{}).get("rich_text",[])
            content = "".join([x["text"]["content"] for x in txt if x])
            if "开仓价:" in content:
                m = re.search(r"\$?([\d.]+)", content)
                if m: entry_price = float(m.group(1))
            if "持仓数量:" in content:
                m = re.search(r"(\d+)", content)
                if m: qty = float(m.group(1))
            if "杠杆:" in content:
                m = re.search(r"(\d+)x", content)
                if m: leverage = float(m.group(1))
            if "方向:" in content:
                direction = content.split("方向:")[1].split("|")[0].strip() if "方向:" in content else ""

        if entry_price and qty and leverage:
            update_position_current_price(page_id, ticker, cur_price, entry_price, qty, leverage, direction)
            print(f"✅ 价格已更新")
        else:
            print(f"⚠️ 无法解析持仓详情，请手动更新")

    elif cmd == "help":
        print(__doc__)

    else:
        print("未知命令:", cmd)
        print(__doc__)