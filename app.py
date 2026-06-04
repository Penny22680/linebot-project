from flask import Flask, request, abort
import os
import torch
import numpy as np
import shutil
import gdown
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from linebot.v3.webhook import WebhookHandler
from linebot.v3.messaging import MessagingApi
from linebot.v3.messaging.configuration import Configuration
from linebot.v3.messaging import ApiClient
from linebot.v3.messaging.models import ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.webhook import WebhookParser

app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
api_client = ApiClient(configuration)
line_bot_api = MessagingApi(api_client)
parser = WebhookParser(LINE_CHANNEL_SECRET)

MODEL_PATH = "./model"

# 全域變數，初始為 None
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

def load_model_lazy():
    """動態懶載入模型：第一次觸發訊息時下載並加載大腦"""
    global tokenizer, model
    
    if tokenizer is not None and model is not None:
        return

    if not os.path.exists(MODEL_PATH):
        print("🚀 [動態載入] 偵測到本機無模型，開始從 Google Drive 抓取壓縮包...")
        DRIVE_ZIP_URL = "https://drive.google.com/file/d/1dL8l2KrWo41l8qOlW9OIG1cl6Nsqj9KJ"
        try:
            zip_path = "./model.zip"
            # 🌟 已移除 fuzzy=True 參數
            gdown.download(url=DRIVE_ZIP_URL, output=zip_path, quiet=False)
            
            print("📦 [動態載入] 下載成功，正在解壓縮...")
            shutil.unpack_archive(zip_path, MODEL_PATH)
            
            if os.path.exists(zip_path):
                os.remove(zip_path)
            print("✅ [動態載入] 模型檔案已解壓至 model 目錄！")
        except Exception as e:
            print(f"❌ 模型下載或解壓失敗: {e}")
            raise e

    if tokenizer is None or model is None:
        print("🧠 正在將 BERT 模型載入至記憶體...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        print("🎉 模型成功完全載入！")

def predict_bert(text):
    if not text.strip():
        return "無法辨識空文字"
    
    load_model_lazy()
    
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    return LABEL_MAPPING.get(predicted_class_id, "其他")

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)
    try:
        events = parser.parse(body, signature)
    except Exception:
        abort(400)
    for event in events:
        if isinstance(event, MessageEvent):
            user_text = event.message.text
            result_label = predict_bert(user_text)
            reply_text = f"【BERT 模型偵測結果】\n這段文字屬於：{result_label}"
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply_text)]
                )
            )
    return "OK", 200

@app.route("/", methods=["GET"])
def index():
    return "BERT LINE Bot 服務正常運行中！", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
