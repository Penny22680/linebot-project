from flask import Flask, request, abort
import os
import traceback
import requests

from linebot.v3.messaging import MessagingApi, ApiClient
from linebot.v3.messaging.configuration import Configuration
from linebot.v3.messaging.models import ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.webhook import WebhookParser

app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    raise RuntimeError("LINE Token 或 Secret 沒有設定")

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
api_client = ApiClient(configuration)
line_bot_api = MessagingApi(api_client)
parser = WebhookParser(LINE_CHANNEL_SECRET)

HF_API_URL = "https://penny0922-linebot-bert-api.hf.space/gradio_api/call/predict"

def predict_bert(text):
    response = requests.post(
        HF_API_URL,
        json={"data": [text]},
        timeout=60
    )
    response.raise_for_status()
    result = response.json()
    return result["data"][0]

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)

    try:
        events = parser.parse(body, signature)
    except Exception:
        print("❌ Webhook parse error")
        print(traceback.format_exc())
        abort(400)

    for event in events:
        if isinstance(event, MessageEvent) and isinstance(event.message, TextMessageContent):
            try:
                text = event.message.text
                result = predict_bert(text)
                reply_text = f"🔍 判斷結果：{result}"
            except Exception as e:
                print("❌ 系統錯誤")
                print(traceback.format_exc())
                reply_text = f"❌ 系統錯誤：\n{str(e)}"

            try:
                line_bot_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text=reply_text)]
                    )
                )
            except Exception:
                print("❌ LINE 回覆失敗")
                print(traceback.format_exc())

    return "OK", 200

@app.route("/")
def home():
    return "LINE Bot Running", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
