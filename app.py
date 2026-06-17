from flask import Flask, request, abort
import os
import torch
import shutil
import gdown
import traceback

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from linebot.v3.messaging import (
    MessagingApi,
    ApiClient
)
from linebot.v3.messaging.configuration import Configuration
from linebot.v3.messaging.models import (
    ReplyMessageRequest,
    TextMessage
)
from linebot.v3.webhooks import MessageEvent
from linebot.v3.webhook import WebhookParser

app = Flask(__name__)

# =========================
# LINE 設定
# =========================
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

configuration = Configuration(
    access_token=LINE_CHANNEL_ACCESS_TOKEN
)

api_client = ApiClient(configuration)
line_bot_api = MessagingApi(api_client)
parser = WebhookParser(LINE_CHANNEL_SECRET)

# =========================
# 模型設定
# =========================
MODEL_PATH = "./model"
ZIP_PATH = "./model.zip"
FILE_ID = "1LJzbFRxYjORxOpxnHwLtj_3L1ZddpeI0"

tokenizer = None
model = None

LABEL_MAPPING = {
    0: "健康醫療",
    1: "其他",
    2: "愛情",
    3: "投資",
    4: "財務金融",
    5: "社會/生活",
    6: "購物",
    7: "虛假中獎"
}

# =========================
# 檢查模型
# =========================
def model_exists():
    return (
        os.path.exists(MODEL_PATH)
        and os.path.exists(os.path.join(MODEL_PATH, "config.json"))
    )

# =========================
# 安全下載模型
# =========================
def download_model():
    try:
        print("🚀 下載模型中...")

        gdown.download(
            id=FILE_ID,
            output=ZIP_PATH,
            quiet=False
        )

        print("📦 解壓模型...")

        shutil.unpack_archive(ZIP_PATH, MODEL_PATH)

        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)

        print("✅ 模型下載完成")

    except Exception as e:
        print("❌ 模型下載失敗：", str(e))
        raise

# =========================
# 載入模型（穩定版）
# =========================
def load_model_lazy():
    global tokenizer, model

    if tokenizer is not None and model is not None:
        return

    if not model_exists():
        download_model()

    print("🧠 載入 tokenizer & model...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            use_fast=False,          # ⭐ 關鍵修復
            local_files_only=True,
            trust_remote_code=True
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            trust_remote_code=True
        )

        model.eval()

        print("✅ 模型載入成功")

    except Exception as e:
        print("❌ 模型載入失敗")
        print(traceback.format_exc())
        raise

# =========================
# 預測
# =========================
def predict_bert(text):

    if not text or not text.strip():
        return "無法辨識空文字"

    load_model_lazy()

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(outputs.logits, dim=-1).item()

    return LABEL_MAPPING.get(pred, "其他")

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
        print("❌ Webhook parse error")
        abort(400)

    for event in events:

        if isinstance(event, MessageEvent):

            try:
                text = event.message.text
                result = predict_bert(text)

                reply_text = f"🔍 判斷結果：{result}"

            except Exception as e:
                print(traceback.format_exc())
                reply_text = f"❌ 系統錯誤：\n{str(e)}"

            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply_text)]
                )
            )

    return "OK", 200

# =========================
# 健康檢查
# =========================
@app.route("/")
def home():
    return "LINE Bot Running", 200

# =========================
# 啟動
# =========================
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
