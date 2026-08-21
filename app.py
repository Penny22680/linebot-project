from __future__ import annotations

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import joblib
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
from linebot.v3.messaging.exceptions import ApiException
from linebot.v3.webhooks import MessageEvent, TextMessageContent

from predict_url_v3 import format_url_result, predict_url_v3


# =========================================================
# 1. 基本設定
# =========================================================

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================================
# 2. Render Environment Variables
# =========================================================

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

HF_SPACE_URL = os.getenv(
    "HF_SPACE_URL",
    "https://penny0922-linebot-bert-v3.hf.space",
)

MAX_PREDICTION_SECONDS = int(os.getenv("MAX_PREDICTION_SECONDS", "60"))
MIN_TEXT_LENGTH = int(os.getenv("MIN_TEXT_LENGTH", "10"))
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "3000"))

if not LINE_CHANNEL_ACCESS_TOKEN:
    raise RuntimeError(
        "找不到 LINE_CHANNEL_ACCESS_TOKEN，請到 Render 的 Environment Variables 設定。"
    )

if not LINE_CHANNEL_SECRET:
    raise RuntimeError(
        "找不到 LINE_CHANNEL_SECRET，請到 Render 的 Environment Variables 設定。"
    )


# =========================================================
# 3. LINE SDK
# =========================================================

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)


# =========================================================
# 4. LightGBM V3 網址模型
# =========================================================

PHISHING_MODEL_PATH = os.path.join(BASE_DIR, "phishing_model_v3.pkl")
TFIDF_VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer_v3.pkl")

try:
    phishing_model = joblib.load(PHISHING_MODEL_PATH)
    tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)

    if int(phishing_model.n_features_in_) != 1018:
        raise RuntimeError(
            f"網址模型應需要 1018 個特徵，實際為 {phishing_model.n_features_in_}。"
        )

    if len(tfidf_vectorizer.get_feature_names_out()) != 1000:
        raise RuntimeError("TF-IDF 向量器特徵數不是 1000。")

    logger.info(
        "LightGBM V3 載入成功：model_features=%s, tfidf_features=%s",
        phishing_model.n_features_in_,
        len(tfidf_vectorizer.get_feature_names_out()),
    )
except Exception as error:
    logger.exception("LightGBM V3 初始化失敗")
    raise RuntimeError(f"網址模型初始化失敗：{error}") from error


# =========================================================
# 5. URL 判斷與網址預測
# =========================================================

URL_PATTERN = re.compile(r"^https?://[^\s]+$", re.IGNORECASE)


def is_url(text: str) -> bool:
    return bool(URL_PATTERN.fullmatch(str(text).strip()))


def predict_phishing_url(url: str) -> str:
    logger.info("開始分析網址：%s", url)
    start_time = time.time()

    result = predict_url_v3(
        url=url,
        model=phishing_model,
        tfidf_vectorizer=tfidf_vectorizer,
    )
    reply_text = format_url_result(result)

    logger.info(
        "網址分析完成：label=%s, phishing=%.4f, elapsed=%.2fs",
        result["label"],
        result["phishing_probability"],
        time.time() - start_time,
    )
    return reply_text


# =========================================================
# 6. Hugging Face BERT V3
# =========================================================


def _call_huggingface(text: str) -> str:
    logger.info("正在建立 Hugging Face Client")
    client = Client(HF_SPACE_URL, verbose=False)
    result = client.predict(text, api_name="/predict")
    logger.info("Hugging Face 預測成功")
    return str(result)


def predict_with_huggingface(text: str) -> str:
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_call_huggingface, text)

    try:
        return future.result(timeout=MAX_PREDICTION_SECONDS)
    except FutureTimeoutError as error:
        future.cancel()
        logger.error("Hugging Face 預測超過 %s 秒", MAX_PREDICTION_SECONDS)
        raise RuntimeError(
            f"模型處理超過 {MAX_PREDICTION_SECONDS} 秒"
        ) from error
    except Exception as error:
        logger.exception("Hugging Face 呼叫失敗")
        raise RuntimeError(f"Hugging Face 呼叫失敗：{error}") from error
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


# =========================================================
# 7. 一般文字驗證
# =========================================================


def validate_user_text(text: str) -> str | None:
    text = str(text).strip()

    if not text:
        return "請輸入想要辨識的文字內容。"

    if len(text) < MIN_TEXT_LENGTH:
        return (
            "⚠️ 輸入內容過短，可能影響 AI 判斷準確度。\n\n"
            f"請貼上較完整的訊息內容，建議至少 {MIN_TEXT_LENGTH} 個字。"
        )

    if len(text) > MAX_TEXT_LENGTH:
        return (
            "⚠️ 輸入內容過長。\n\n"
            f"請將文字縮短至 {MAX_TEXT_LENGTH} 字以內再試一次。"
        )

    return None


# =========================================================
# 8. Flask Routes
# =========================================================

@app.route("/", methods=["GET"])
def home():
    return {
        "status": "ok",
        "service": "LINE scam + phishing detection bot",
        "text_model": "BERT V3",
        "text_explainability": "Integrated Gradients",
        "url_model": "LightGBM V3",
        "url_features": "18 structural + 1000 character TF-IDF",
        "url_explainability": "human-readable structural feature explanation",
        "hf_space": HF_SPACE_URL,
    }, 200


@app.route("/health", methods=["GET"])
def health():
    return "OK", 200


@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    logger.info("收到 LINE Webhook，內容長度：%s", len(body))

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logger.warning("LINE Webhook 簽章驗證失敗")
        abort(400)
    except Exception:
        logger.exception("處理 LINE Webhook 時發生錯誤")
        abort(500)

    return "OK", 200


# =========================================================
# 9. LINE 文字訊息處理
# =========================================================

@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event):
    user_text = event.message.text.strip()
    logger.info("收到使用者訊息，字數：%s", len(user_text))

    if is_url(user_text):
        logger.info("偵測到 URL，進入 LightGBM V3")
        try:
            reply_text = predict_phishing_url(user_text)
        except Exception as error:
            logger.exception("網址分析失敗：%s", error)
            reply_text = (
                "⚠️ 網址目前無法完成分析。\n\n"
                "請確認網址格式，或稍後再試一次。"
            )
    else:
        validation_message = validate_user_text(user_text)
        if validation_message:
            reply_text = validation_message
        else:
            start_time = time.time()
            try:
                # Hugging Face V3 已完成分類、機率與 Integrated Gradients，直接回傳。
                reply_text = predict_with_huggingface(user_text)
                logger.info(
                    "BERT V3 + IG 預測完成，耗時 %.2f 秒",
                    time.time() - start_time,
                )
            except Exception as error:
                logger.exception(
                    "BERT V3 預測失敗，耗時 %.2f 秒：%s",
                    time.time() - start_time,
                    error,
                )
                reply_text = (
                    "⚠️ 文字模型目前正在啟動、更新或忙碌中。\n\n"
                    "請稍候約 30 秒後重新傳送一次。"
                )

    try:
        with ApiClient(configuration) as api_client:
            messaging_api = MessagingApi(api_client)
            messaging_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply_text)],
                )
            )
        logger.info("LINE 回覆成功")
    except ApiException as error:
        error_text = str(error)
        if "invalid reply token" in error_text.lower():
            logger.warning("LINE reply token 已失效或已使用")
        else:
            logger.exception("LINE Messaging API 回覆失敗：%s", error)
    except Exception:
        logger.exception("LINE 回覆失敗")


# =========================================================
# 10. 本機 / Render 執行
# =========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
