from flask import Flask, request, abort
import os
import traceback
from gradio_client import Client

from linebot.v3.messaging import MessagingApi, ApiClient
from linebot.v3.messaging.configuration import Configuration
from linebot.v3.messaging.models import ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.webhook import WebhookParser

app = Flask(__name__)

# =========================
# LINE 設定
# =========================
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    raise RuntimeError("LINE Token 或 Secret 沒有設定")

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
api_client = ApiClient(configuration)
line_bot_api = MessagingApi(api_client)
parser = WebhookParser(LINE_CHANNEL_SECRET)

# =========================
# Hugging Face Space
# =========================
HF_SPACE_URL = "https://penny0922-linebot-bert-binary-api.hf.space"
hf_client = Client(HF_SPACE_URL)


def predict_bert(text):
    """
    呼叫 Hugging Face Space
    """
    result = hf_client.predict(
        text,
        api_name="/predict"
    )

    print("HF 回傳結果：")
    print(result)

    return str(result)


# =========================
# LINE Webhook
# =========================
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)

    try:
        events = parser.parse(body, signature)
    except Exception:
        print("Webhook Parse Error")
        print(traceback.format_exc())
        abort(400)

    for event in events:

        if isinstance(event, MessageEvent) and isinstance(event.message, TextMessageContent):

            user_text = event.message.text

            try:
                reply_text = predict_bert(user_text)

            except Exception as e:
                print(traceback.format_exc())
                reply_text = f"❌ 系統錯誤\n{str(e)}"

            try:
                line_bot_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[
                            TextMessage(text=reply_text)
                        ]
                    )
                )

            except Exception:
                print("LINE Reply Error")
                print(traceback.format_exc())

    return "OK", 200


@app.route("/")
def home():
    return "LINE Bot Running!", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
