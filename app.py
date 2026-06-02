
from flask import Flask, request, abort

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

app = Flask(__name__)

# ⚠️ 請之後建議重新產生 token / secret
LINE_CHANNEL_ACCESS_TOKEN = "WTABxCBcujxYRnghgAE25sholokOVwn9fL1Dj9EmQNYK1Ok6VJn+wENCUHHQaYy0GIP5Nb8HiLADBwYPQWnlZgSQG+GG69CwMV5LNPlfCRNRZZeJzoOGEtDzD25AvdBq5A73cKh04HdNNbiITkwRtQdB04t89/1O/w1cDnyilFU="
LINE_CHANNEL_SECRET = "d77b9c7e6dd980ecba1694d1a54faa54"

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# 🔍 詐騙判斷
def scam_check(text):
    keywords = ["凍結", "帳號", "中獎", "點擊", "驗證", "轉帳", "OTP", "密碼"]
    score = sum(1 for k in keywords if k in text)

    if score >= 3:
        return "🔴 高風險詐騙"
    elif score >= 1:
        return "🟠 可疑訊息"
    else:
        return "🟢 正常訊息"

# LINE webhook
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)

    print("📩 收到訊息：", body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("❌ Signature 驗證失敗")
        abort(400)

    return "OK"

# 收訊息
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_text = event.message.text

    result = scam_check(user_text)

    reply = f"{result}\n\n你輸入：{user_text}"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )

    print("✅ 已回覆：", reply)
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
