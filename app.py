from flask import Flask, request, abort
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from linebot.v3.webhook import WebhookHandler
from linebot.v3.messaging import MessagingApi
from linebot.v3.messaging.configuration import Configuration
from linebot.v3.messaging import ApiClient
from linebot.v3.messaging.models import ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.webhook import WebhookParser

app = Flask(__name__)

# 1. 讀取環境變數 (在 Render 後台設定)
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
api_client = ApiClient(configuration)
line_bot_api = MessagingApi(api_client)

handler = WebhookHandler(LINE_CHANNEL_SECRET)
parser = WebhookParser(LINE_CHANNEL_SECRET)

# 2. 載入你的 BERT 模型 (路徑指向專案資料夾內的 model 資料夾)
MODEL_PATH = "./model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# 🌟 請根據你當初訓練模型時，Label Encoder 轉出來的數字順序修改這裡的對應！
# 這裡先幫你列出 CSV 裡有的 10 個類別
LABEL_MAPPING = {
    0: "健康醫療",
    1: "投資",
    2: "其他",
    3: "購物",
    4: "財務金融",
    5: "社會/生活",
    6: "愛情",
    7: "假新聞",
    8: "職場",
    9: "虛假中獎"
}

def predict_bert(text):
    """將 LINE 收到的文字餵給 BERT 模型做類別預測"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    
    # 根據數字找出中文類別名稱，找不到就回傳其他
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
            # 聽：拿到使用者輸入的文字
            user_text = event.message.text
            
            # 算：呼叫 BERT 模型進行分類預測
            result_label = predict_bert(user_text)

            # 組裝回覆：告知使用者這段文字是什麼類別
            reply_text = f"【BERT 模型偵測結果】\n這段文字屬於：{result_label}"

            # 回：把結果傳送給使用者
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply_text)]
                )
            )
    return "OK"

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
