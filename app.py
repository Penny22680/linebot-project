from __future__ import annotations

import logging
import os
import re
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)

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
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
)

from predict_url_v4_2 import predict_url


# =========================================================
# 1. 基本設定
# =========================================================

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


# =========================================================
# 2. Render Environment Variables
# =========================================================

LINE_CHANNEL_ACCESS_TOKEN = os.getenv(
    "LINE_CHANNEL_ACCESS_TOKEN"
)

LINE_CHANNEL_SECRET = os.getenv(
    "LINE_CHANNEL_SECRET"
)

HF_SPACE_URL = os.getenv(
    "HF_SPACE_URL",
    "https://penny0922-linebot-bert-v3.hf.space",
)

MAX_PREDICTION_SECONDS = int(
    os.getenv(
        "MAX_PREDICTION_SECONDS",
        "60",
    )
)

MIN_TEXT_LENGTH = int(
    os.getenv(
        "MIN_TEXT_LENGTH",
        "10",
    )
)

MAX_TEXT_LENGTH = int(
    os.getenv(
        "MAX_TEXT_LENGTH",
        "3000",
    )
)


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
# 3. LINE SDK
# =========================================================

configuration = Configuration(
    access_token=LINE_CHANNEL_ACCESS_TOKEN
)

handler = WebhookHandler(
    LINE_CHANNEL_SECRET
)


# =========================================================
# 4. URL 判斷與網址預測
# =========================================================

URL_PATTERN = re.compile(
    r"(https?://[^\s]+|www\.[^\s]+)",
    re.IGNORECASE,
)


def extract_url(
    text: str,
) -> str | None:
    """
    從使用者訊息中抓出第一個網址。

    支援：
    https://example.com
    http://example.com
    www.example.com
    """

    match = URL_PATTERN.search(
        str(text).strip()
    )

    if not match:
        return None

    url = match.group(0)

    # 移除句尾常見標點，
    # 避免「網址。」「網址）」一起送進模型
    return url.rstrip(
        ".,!?;:，。！？；：)]}>'\""
    )


def get_risk_level(
    score: float,
) -> str:
    if score >= 80:
        return "極高風險"

    if score >= 60:
        return "高風險"

    if score >= 40:
        return "中度風險"

    if score >= 20:
        return "低度風險"

    return "極低風險"


def predict_phishing_url(
    url: str,
) -> str:
    logger.info(
        "開始分析網址：%s",
        url,
    )

    start_time = time.time()

    result = predict_url(
        url
    )

    risk_score = result.get(
        "final_risk_score"
    )

    if risk_score is None:
        risk_score = 0.0

    risk_level = get_risk_level(
        float(risk_score)
    )

    reply: list[str] = []

    reply.append("🔍 網址分析結果")
    reply.append("")

    reply.append("🌐 網址：")
    reply.append(
        result["normalized_url"]
    )
    reply.append("")

    reply.append(
        f"📌 判斷：{result['result']}"
    )

    reply.append(
        f"⚠️ 風險分數："
        f"{result['final_risk_score']}/100"
    )

    reply.append(
        f"📊 風險等級：{risk_level}"
    )

    reply.append("")

    reply.append("📖 判斷原因：")

    reasons = result.get(
        "reasons",
        [],
    )

    if reasons:
        for reason in reasons:
            reply.append(
                f"• {reason}"
            )
    else:
        reply.append(
            "• 依網址字元與結構特徵綜合判斷"
        )

    reply.append("")

    if result["result"] == "釣魚網址":
        reply.append("🚨 安全建議：")
        reply.append("• 不要輸入帳號或密碼")
        reply.append("• 不要提供信用卡或驗證碼")
        reply.append("• 不要下載任何不明檔案")
        reply.append("• 建議立即關閉網頁")

    elif result["result"] == "短網址，需進一步檢查":
        reply.append("⚠️ 安全建議：")
        reply.append("• 短網址會隱藏真正目的地")
        reply.append("• 請先展開網址後再檢查")
        reply.append("• 不要直接登入或付款")

    else:
        reply.append("✅ 安全提醒：")
        reply.append(
            "• 此結果僅依網址特徵判斷"
        )
        reply.append(
            "• 仍請確認網站內容與來源"
        )

    logger.info(
        "網址分析完成："
        "result=%s, risk_score=%s, elapsed=%.2fs",
        result.get("result"),
        result.get("final_risk_score"),
        time.time() - start_time,
    )

    return "\n".join(
        reply
    )


# =========================================================
# 5. Hugging Face BERT V3
# =========================================================

def _call_huggingface(
    text: str,
) -> str:
    logger.info(
        "正在建立 Hugging Face Client"
    )

    client = Client(
        HF_SPACE_URL,
        verbose=False,
    )

    result = client.predict(
        text,
        api_name="/predict",
    )

    logger.info(
        "Hugging Face 預測成功"
    )

    return str(
        result
    )


def predict_with_huggingface(
    text: str,
) -> str:
    executor = ThreadPoolExecutor(
        max_workers=1
    )

    future = executor.submit(
        _call_huggingface,
        text,
    )

    try:
        return future.result(
            timeout=MAX_PREDICTION_SECONDS
        )

    except FutureTimeoutError as error:
        future.cancel()

        logger.error(
            "Hugging Face 預測超過 %s 秒",
            MAX_PREDICTION_SECONDS,
        )

        raise RuntimeError(
            f"模型處理超過 "
            f"{MAX_PREDICTION_SECONDS} 秒"
        ) from error

    except Exception as error:
        logger.exception(
            "Hugging Face 呼叫失敗"
        )

        raise RuntimeError(
            f"Hugging Face 呼叫失敗：{error}"
        ) from error

    finally:
        executor.shutdown(
            wait=False,
            cancel_futures=True,
        )


# =========================================================
# 6. 一般文字驗證
# =========================================================

def validate_user_text(
    text: str,
) -> str | None:
    text = str(
        text
    ).strip()

    if not text:
        return (
            "請輸入想要辨識的文字內容。"
        )

    if len(text) < MIN_TEXT_LENGTH:
        return (
            "⚠️ 輸入內容過短，"
            "可能影響 AI 判斷準確度。\n\n"
            f"請貼上較完整的訊息內容，"
            f"建議至少 {MIN_TEXT_LENGTH} 個字。"
        )

    if len(text) > MAX_TEXT_LENGTH:
        return (
            "⚠️ 輸入內容過長。\n\n"
            f"請將文字縮短至 "
            f"{MAX_TEXT_LENGTH} 字以內再試一次。"
        )

    return None


# =========================================================
# 7. Flask Routes
# =========================================================

@app.route(
    "/",
    methods=["GET"],
)
def home():
    return {
        "status": "ok",
        "service":
            "LINE scam + phishing detection bot",
        "text_model":
            "BERT V3",
        "text_explainability":
            "Integrated Gradients",
        "url_model":
            "LightGBM V4.2",
        "url_features":
            "26 structural + "
            "3000 character TF-IDF",
        "url_threshold":
            0.538741,
        "url_explainability":
            "human-readable structural "
            "feature explanation",
        "hf_space":
            HF_SPACE_URL,
    }, 200


@app.route(
    "/health",
    methods=["GET"],
)
def health():
    return "OK", 200


@app.route(
    "/callback",
    methods=["POST"],
)
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
        logger.warning(
            "LINE Webhook 簽章驗證失敗"
        )
        abort(400)

    except Exception:
        logger.exception(
            "處理 LINE Webhook 時發生錯誤"
        )
        abort(500)

    return "OK", 200


# =========================================================
# 8. LINE 文字訊息處理
# =========================================================

@handler.add(
    MessageEvent,
    message=TextMessageContent,
)
def handle_text_message(
    event,
):
    user_text = (
        event.message.text
        .strip()
    )

    logger.info(
        "收到使用者訊息，字數：%s",
        len(user_text),
    )

    url = extract_url(
        user_text
    )

    if url:
        logger.info(
            "偵測到 URL，"
            "進入 LightGBM V4.2"
        )

        try:
            reply_text = (
                predict_phishing_url(
                    url
                )
            )

        except Exception as error:
            logger.exception(
                "網址分析失敗：%s",
                error,
            )

            reply_text = (
                "⚠️ 網址目前無法完成分析。\n\n"
                "請確認網址格式，"
                "或稍後再試一次。"
            )

    else:
        validation_message = (
            validate_user_text(
                user_text
            )
        )

        if validation_message:
            reply_text = (
                validation_message
            )

        else:
            start_time = time.time()

            try:
                reply_text = (
                    predict_with_huggingface(
                        user_text
                    )
                )

                logger.info(
                    "BERT V3 + IG 預測完成，"
                    "耗時 %.2f 秒",
                    time.time() - start_time,
                )

            except Exception as error:
                logger.exception(
                    "BERT V3 預測失敗，"
                    "耗時 %.2f 秒：%s",
                    time.time() - start_time,
                    error,
                )

                reply_text = (
                    "⚠️ 文字模型目前正在啟動、"
                    "更新或忙碌中。\n\n"
                    "請稍候約 30 秒後"
                    "重新傳送一次。"
                )

    try:
        with ApiClient(
            configuration
        ) as api_client:
            messaging_api = (
                MessagingApi(
                    api_client
                )
            )

            messaging_api.reply_message(
                ReplyMessageRequest(
                    reply_token=(
                        event.reply_token
                    ),
                    messages=[
                        TextMessage(
                            text=reply_text
                        )
                    ],
                )
            )

        logger.info(
            "LINE 回覆成功"
        )

    except ApiException as error:
        error_text = str(
            error
        )

        if (
            "invalid reply token"
            in error_text.lower()
        ):
            logger.warning(
                "LINE reply token "
                "已失效或已使用"
            )

        else:
            logger.exception(
                "LINE Messaging API "
                "回覆失敗：%s",
                error,
            )

    except Exception:
        logger.exception(
            "LINE 回覆失敗"
        )


# =========================================================
# 9. 本機 / Render 執行
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
