from flask import Flask, request, abort
import os
import torch
import shutil
import gdown
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from linebot.v3.messaging import MessagingApi, ApiClient, Configuration
from linebot.v3.messaging.models import ReplyMessageRequest, TextMessage
from linebot.v3.webhook import WebhookParser
from linebot.v3.webhooks import MessageEvent

app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
api_client = ApiClient(configuration)
line_bot_api = MessagingApi(api_client)
parser = WebhookParser(LINE_CHANNEL_SECRET)

MODEL_PATH = "./model"

tokenizer = None
model = None

LABEL_MAPPING = {
    0: "真新聞",
    1: "假新聞"
}

def load_model_lazy():
    global tokenizer, model

    if tokenizer and model:
        return

    if not os.path.exists(MODEL_PATH):
        print("下載模型中...")

        FILE_ID = "1dL8l2KrWo41l8qOlW9OIG1cl6Nsqj9KJ"
        zip_path = "./model.zip"

        gdown.download(id=FILE_ID, output=zip_path, quiet=False)
        shutil.unpack_archive(zip_path, MODEL_PATH)
        os.remove(zip_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

def predict(text):
    if not text.strip():
        return "空內容"

    load_model_lazy()

    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(outputs.logits, dim=-1).item()
    return LABEL_MAPPING.get(pred, "未知")

@app.route("/callback", methods=["POST"])
def callback():
    body = request.get_data(as_text=True)
    signature = request.headers.get("X-Line-Signature")

    try:
        events = parser.parse(body, signature)
    except Exception:
        abort(400)

    for event in events:
        if isinstance(event, MessageEvent):
            text = event.message.text

            try:
                result = predict(text)
                reply = f"🔍 判斷結果：{result}"
            except Exception as e:
                reply = f"錯誤：{str(e)}"

            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply)]
                )
            )

    return "OK", 200

@app.route("/")
def home():
    return "LINE Bot Running", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
