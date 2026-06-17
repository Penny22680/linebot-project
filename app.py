from flask import Flask, request, abort
import os
import torch
import shutil
import gdown
import traceback

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from linebot.v3.messaging import MessagingApi, ApiClient
from linebot.v3.messaging.configuration import Configuration
from linebot.v3.messaging.models import ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.webhook import WebhookParser

app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    raise RuntimeError("LINE_CHANNEL_ACCESS_TOKEN 或 LINE_CHANNEL_SECRET 沒有設定")

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
api_client = ApiClient(configuration)
line_bot_api = MessagingApi(api_client)
parser = WebhookParser(LINE_CHANNEL_SECRET)

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


def model_exists():
    return (
        os.path.isdir(MODEL_PATH)
        and os.path.exists(os.path.join(MODEL_PATH, "config.json"))
        and (
            os.path.exists(os.path.join(MODEL_PATH, "model.safetensors"))
            or os.path.exists(os.path.join(MODEL_PATH, "pytorch_model.bin"))
        )
    )


def clean_model_files():
    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH)

    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    os.makedirs(MODEL_PATH, exist_ok=True)


def download_model():
    print("🚀 下載模型中...")

    clean_model_files()

    url = f"https://drive.google.com/uc?id={FILE_ID}"

    downloaded = gdown.download(
        url=url,
        output=ZIP_PATH,
        quiet=False
    )

    if downloaded is None:
        raise RuntimeError(
            "Google Drive 下載失敗：請確認 model.zip 權限是「知道連結的任何人都能檢視」，並確認 FILE_ID 正確"
        )

    if not os.path.exists(ZIP_PATH):
        raise RuntimeError("模型下載失敗：model.zip 沒有成功建立")

    print("📦 解壓模型...")

    shutil.unpack_archive(ZIP_PATH, MODEL_PATH)

    print("📂 model 目錄內容：", os.listdir(MODEL_PATH))

    if not os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        raise RuntimeError(
            f"模型解壓後找不到 config.json，目前 model 內容：{os.listdir(MODEL_PATH)}"
        )

    os.remove(ZIP_PATH)

    print("✅ 模型下載完成")


def load_model_lazy():
    global tokenizer, model

    if tokenizer is not None and model is not None:
        return

    if not model_exists():
        download_model()

    print("🧠 載入 tokenizer & model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True,
        local_files_only=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    model.eval()

    print("✅ 模型載入成功")


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
