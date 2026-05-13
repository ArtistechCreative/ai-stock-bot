"""
Telegram 推送 Bot（修复 async warning，用 requests 直发）
"""
import os
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.hermes/.env"))


def send_report(report: str, chat_id: str = None) -> bool:
    """推送报告到 Telegram（直接用 requests，不依赖 telegram 库的 async）"""
    import requests

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    cid = chat_id or os.getenv("TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_HOME_CHANNEL") or "6801255591"

    if not token:
        print("⚠️ 缺少 TELEGRAM_BOT_TOKEN，跳过推送")
        return False

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": int(cid), "text": report, "parse_mode": "HTML"},
            timeout=15,
        )
        result = resp.json()
        if result.get("ok"):
            print(f"  ✅ 推送成功")
            return True
        else:
            print(f"  ❌ 推送失败: {result}")
            return False
    except Exception as e:
        print(f"  ❌ 推送异常: {e}")
        return False


def send_image(image_path: str, caption: str = None, chat_id: str = None) -> bool:
    """发送图片（K线图用）"""
    import requests

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    cid = chat_id or os.getenv("TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_HOME_CHANNEL") or "6801255591"

    if not token:
        return False

    try:
        with open(image_path, "rb") as f:
            resp = requests.post(
                f"https://api.telegram.org/bot{token}/sendPhoto",
                files={"photo": f},
                data={"chat_id": int(cid), "caption": caption or ""},
                timeout=15,
            )
        return resp.json().get("ok", False)
    except Exception as e:
        print(f"  ❌ 图片发送失败: {e}")
        return False