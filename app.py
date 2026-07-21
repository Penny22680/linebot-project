import os
import time
import logging

from flask import Flask, abort, request
from gradio_client import Client

from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent


# =========================================================
# 基本設定
# =========================================================

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =========================================================
# Render Environment Variables
# =========================================================

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

# 這裡改成你目前成功運作的 CPU Basic Space
HF_SPACE_URL = "https://penny0922-linebot-bert-api.hf.space"

if not LINE_CHANNEL_ACCESS_TOKEN:
    raise RuntimeError(
        "找不到 LINE_CHANNEL_ACCESS_TOKEN，"
        "請到 Render 的 Environment Variables 設定。"
    )

if not LINE_CHANNEL_SECRET:
    raise RuntimeError(
        "找不到 LINE_CHANNEL_SECRET，"
        "請到 Render 的 Environment Variables 設定。"
    )


# =========================================================
# LINE SDK 設定
# =========================================================

configuration = Configuration(
    access_token=LINE_CHANNEL_ACCESS_TOKEN
)

handler = WebhookHandler(
    LINE_CHANNEL_SECRET
)


# =========================================================
# Hugging Face 呼叫函式
# =========================================================

def predict_with_huggingface(text: str) -> str:
    """
    呼叫 Hugging Face Gradio Space。

    不在程式啟動時建立 Client，
    避免 Hugging Face 暫時休眠或重建時，
    造成 Render 啟動失敗。
    """

    last_error = None

    for attempt in range(1, 4):
        try:
            logger.info(
                "正在呼叫 Hugging Face，第 %s 次嘗試",
                attempt,
            )

            client = Client(
                HF_SPACE_URL,
                verbose=False,
            )

            result = client.predict(
                text,
                api_name="/predict",
            )

            logger.info("Hugging Face 預測成功")

            return str(result)

        except Exception as error:
            last_error = error

            logger.exception(
                "第 %s 次呼叫 Hugging Face 失敗",
                attempt,
            )

            if attempt < 3:
                time.sleep(attempt * 2)

    raise RuntimeError(
        f"Hugging Face 呼叫失敗：{last_error}"
    )


# =========================================================
# 文字檢查
# =========================================================

def validate_user_text(text: str) -> str | None:
    """
    驗證使用者輸入。
    回傳 None 代表可以進行模型預測。
    """

    text = text.strip()

    if not text:
        return "請輸入想要辨識的文字內容。"

    if len(text) < 20:
        return (
            "⚠️ 目前輸入內容太短，可能影響判斷準確度。\n\n"
            "請貼上較完整的訊息或新聞內容，建議至少 20 個字。"
        )

    if len(text) > 3000:
        return (
            "⚠️ 輸入內容過長。\n\n"
            "請將文字縮短至 3000 字以內再試一次。"
        )

    return None


# =========================================================
# 統一整理模型回覆
# =========================================================

def format_prediction_result(result: str) -> str:
    """
    在模型結果下方加入提醒文字。
    """

    result_lower = result.lower()

    if "詐騙" in result:
        warning = (
            "\n\n⚠️ 此結果僅供輔助判斷。"
            "請勿因模型結果直接匯款、提供驗證碼、"
            "帳戶資料或個人資料。"
        )

    elif "真實" in result:
        warning = (
            "\n\nℹ️ 模型無法保證內容完全真實，"
            "仍建議確認新聞來源、發布日期與官方公告。"
        )

    elif "scam" in result_lower:
        warning = (
            "\n\n⚠️ 此結果僅供輔助判斷，"
            "請勿直接匯款或提供個人資料。"
        )

    else:
        warning = (
            "\n\nℹ️ 此結果由 AI 模型產生，"
            "僅供參考，請搭配其他來源查證。"
        )

    return result + warning


# =========================================================
# 首頁與健康檢查
# =========================================================

@app.route("/", methods=["GET"])
def home():
    return {
        "status": "ok",
        "service": "LINE BERT scam detection bot",
    }, 200


@app.route("/health", methods=["GET"])
def health():
    return "OK", 200


# =========================================================
# LINE Webhook
# =========================================================

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get(
        "X-Line-Signature",
        "",
    )

    body = request.get_data(
        as_text=True
    )

    logger.info(
        "收到 LINE Webhook，內容長度：%s",
        len(body),
    )

    try:
        handler.handle(
            body,
            signature,
        )

    except InvalidSignatureError:
        logger.warning("LINE Webhook 簽章驗證失敗")
        abort(400)

    except Exception:
        logger.exception("處理 LINE Webhook 時發生錯誤")
        abort(500)

    return "OK", 200


# =========================================================
# 處理 LINE 文字訊息
# =========================================================

@handler.add(
    MessageEvent,
    message=TextMessageContent,
)
def handle_text_message(event):
    user_text = event.message.text.strip()

    logger.info(
        "收到使用者文字，字數：%s",
        len(user_text),
    )

    validation_message = validate_user_text(
        user_text
    )

    if validation_message:
        reply_text = validation_message

    else:
        start_time = time.time()

        try:
            model_result = predict_with_huggingface(
                user_text
            )

            reply_text = format_prediction_result(
                model_result
            )

            elapsed = time.time() - start_time

            logger.info(
                "預測完成，耗時 %.2f 秒",
                elapsed,
            )

        except Exception as error:
            logger.exception(
                "模型預測失敗：%s",
                error,
            )

            reply_text = (
                "⚠️ 系統目前正在啟動、更新或忙碌中。\n\n"
                "請稍候約 30 秒後，再重新傳送一次。"
            )

    try:
        with ApiClient(configuration) as api_client:
            messaging_api = MessagingApi(api_client)

            messaging_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[
                        TextMessage(
                            text=reply_text
                        )
                    ],
                )
            )

        logger.info("LINE 回覆成功")

    except Exception:
        logger.exception("LINE 回覆失敗")


# =========================================================
# 本機測試用
# Render 使用 Gunicorn 時不會執行這一段
# =========================================================

if __name__ == "__main__":
    port = int(
        os.environ.get(
            "PORT",
            5000,
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
    )
