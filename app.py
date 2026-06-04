# =================================================================
# 🎯 2026-06-04 終極無敵懶載入完全體 (解決大檔直連下載與快取問題)
# =================================================================
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

# 讀取環境變數
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
api_client = ApiClient(configuration)
line_bot_api = MessagingApi(api_client)
parser = WebhookParser(LINE_CHANNEL_SECRET)

MODEL_PATH = "./model"

# 全域變數，一開始維持空大腦，避開啟動超時
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
    """動態懶載入大腦：收到訊息才下載並加載"""
    global tokenizer, model
    
    # 1. 檢查是否早已載入完成
    if tokenizer is not None and model is not None:
        return

    # 2. 如果資料夾不存在，啟動直連下載
    if not os.path.exists(MODEL_PATH):
        print("🚀 [核心日誌] 偵測到雲端無模型大腦，啟動 Google Drive 直連下載...")
        
        # 🌟 關鍵修正：加上 &confirm=t 參數，直接繞過 Google Drive 的大檔案防毒警告頁面
        DRIVE_ZIP_URL = "https://docs.google.com/uc?export=download&id=1dL8l2KrWo41l8qOlW9OIG1cl6Nsqj9KJ&confirm=t"
        
        try:
            zip_path = "./model.zip"
            
            # 物理破除快取鬼魂：拆分參數傳遞，絕對無 fuzzy 參數
            target_url = str(DRIVE_ZIP_URL)
            output_file = str(zip_path)
            gdown.download(target_url, output_file, quiet=False)
            
            print("📦 [核心日誌] 壓縮包下載完畢，開始解壓縮...")
            shutil.unpack_archive(output_file, MODEL_PATH)
            
            # 清理多餘的壓縮檔
            if os.path.exists(output_file):
                os.remove(output_file)
            print("✅ [核心日誌] 雲端大腦物理建置成功！")
            
        except Exception as e:
            print(f"❌ [核心日誌] 下載或解壓失敗: {e}")
            raise RuntimeError(f"雲端大腦下載/解壓失敗，原因: {e}")

    # 3. 確保解壓完成後，正式把 BERT 塞進記憶體
    if tokenizer is None or model is None:
        print("🧠 [核心日誌] 正在將 BERT 模型載入記憶體...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
            print("🎉 [核心日誌] BERT 大腦成功完全就位！")
        except Exception as e:
            print(f"❌ [核心日誌] 模型加載至記憶體失敗: {e}")
            raise RuntimeError(f"模型載入記憶體失敗，請檢查資料夾結構。原因: {e}")

def predict_bert(text):
    """BERT 核心預測邏輯"""
    if not text.strip():
        return "無法辨識空文字"
    
    # 觸發大腦載入機制
    load_model_lazy()
    
    # 開始進行預測
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
            
            # 建立多重安全警報機制，哪怕出錯也直接回傳到 LINE 畫面上
            try:
                result_label = predict_bert(user_text)
                reply_text = f"【BERT 模型偵測結果】\n這段文字屬於：{result_label}"
            except Exception as model_err:
                reply_text = f"❌ 系統在解析新大腦時發生錯誤：\n{str(model_err)}"
            
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply_text)]
                )
            )
    return "OK", 200

@app.route("/", methods=["GET"])
def index():
    return "BERT LINE Bot 終極完全體版本正完美運行中！", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
