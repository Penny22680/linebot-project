import os
import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

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

HF_SPACE_URL = os.getenv(
    "HF_SPACE_URL",
    "https://penny0922-linebot-bert-api.hf.space",
)

MAX_PREDICTION_SECONDS = int(
    os.getenv("MAX_PREDICTION_SECONDS", "40")
)
MIN_TEXT_LENGTH = int(
    os.getenv("MIN_TEXT_LENGTH", "50")
)
MAX_TEXT_LENGTH = int(
    os.getenv("MAX_TEXT_LENGTH", "3000")
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

def _call_huggingface(text: str) -> str:
    logger.info("正在建立 Hugging Face Client")

    client = Client(
        HF_SPACE_URL,
        verbose=False,
    )

    logger.info("正在呼叫 Hugging Face /predict")

    result = client.predict(
        text,
        api_name="/predict",
    )

    logger.info("Hugging Face 預測成功")
    return str(result)


def predict_with_huggingface(text: str) -> str:
    """
    呼叫 Hugging Face，並限制最長等待時間。
    不做多次重試，避免 LINE reply token 因等待太久失效。
    """

    executor = ThreadPoolExecutor(max_workers=1)
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
            f"模型處理超過 {MAX_PREDICTION_SECONDS} 秒"
        ) from error

    except Exception as error:
        logger.exception("Hugging Face 呼叫失敗")
        raise RuntimeError(
            f"Hugging Face 呼叫失敗：{error}"
        ) from error

    finally:
        executor.shutdown(
            wait=False,
            cancel_futures=True,
        )


# =========================================================
# 文字檢查
# =========================================================

def validate_user_text(text: str) -> str | None:
    text = text.strip()

    if not text:
        return "請輸入想要辨識的文字內容。"

    if len(text) < MIN_TEXT_LENGTH:
        return (
            "⚠️ 輸入內容過短，可能影響 AI 判斷準確度。\n\n"
            f"請貼上較完整的新聞或訊息內容，"
            f"建議至少 {MIN_TEXT_LENGTH} 個字。"
        )

    if len(text) > MAX_TEXT_LENGTH:
        return (
            "⚠️ 輸入內容過長。\n\n"
            f"請將文字縮短至 {MAX_TEXT_LENGTH} 字以內再試一次。"
        )

    return None


# =========================================================
# 模型結果解析工具
# =========================================================

def extract_percentage(
    pattern: str,
    result: str,
) -> float | None:
    match = re.search(
        pattern,
        result,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def format_prediction_result(result: str) -> str:
    logger.info(
        "模型原始回傳結果：%s",
        result,
    )

    result_lower = result.lower()

    label_match = re.search(
        r"判斷結果[：:]\s*"
        r"(詐騙訊息|真實新聞|真實訊息|詐騙|真實)",
        result,
        flags=re.IGNORECASE,
    )

    confidence = extract_percentage(
        r"模型信心度[：:]\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        result,
    )
    scam_probability = extract_percentage(
        r"詐騙機率[：:]\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        result,
    )
    real_probability = extract_percentage(
        r"真實機率[：:]\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        result,
    )

    if label_match:
        label = label_match.group(1)
    elif "scam" in result_lower:
        label = "詐騙"
    elif "real" in result_lower or "true" in result_lower:
        label = "真實"
    else:
        return (
            result
            + "\n\nℹ️ 此結果由 AI 模型產生，"
            "僅供參考，請搭配其他可靠來源查證。"
        )

    if confidence is None:
        if "詐騙" in label and scam_probability is not None:
            confidence = scam_probability
        elif "真實" in label and real_probability is not None:
            confidence = real_probability
        else:
            confidence = 0.0

    probability_lines = []

    if scam_probability is not None:
        probability_lines.append(
            f"🚨 詐騙機率：{scam_probability:.2f}%"
        )

    if real_probability is not None:
        probability_lines.append(
            f"✅ 真實機率：{real_probability:.2f}%"
        )

    probability_text = "\n".join(
        probability_lines
    )

    if "詐騙" in label:
        if confidence >= 95:
            title = "🔴【模型高度傾向詐騙】"
            description = (
                "模型對詐騙類別的判斷信心很高，"
                "請先停止交易並進一步查證。"
            )
        elif confidence >= 80:
            title = "🟠【高風險訊息】"
            description = (
                "此內容具有較高的詐騙風險，"
                "請提高警覺並避免立即操作。"
            )
        elif confidence >= 60:
            title = "🟡【疑似詐騙】"
            description = (
                "模型偏向詐騙，但判斷仍有不確定性，"
                "建議透過官方來源查證。"
            )
        else:
            title = "⚪【判斷信心不足】"
            description = (
                "模型目前無法明確判斷，"
                "請勿只依賴此結果做出決定。"
            )

        sections = [
            title,
            f"📊 模型信心度：{confidence:.2f}%",
        ]

        if probability_text:
            sections.append(probability_text)

        sections.extend([
            "🔎 風險說明：\n" + description,
            (
                "⚠️ 防詐提醒：\n"
                "• 請勿立即匯款或轉帳\n"
                "• 請勿提供銀行帳號或信用卡資料\n"
                "• 請勿提供密碼或簡訊驗證碼\n"
                "• 請勿點擊不明連結\n"
                "• 建議透過官方管道再次查證\n"
                "• 必要時撥打 165 反詐騙專線"
            ),
            (
                "ℹ️ 此結果僅供輔助判斷，"
                "不代表最終事實認定。"
            ),
        ])

        return "\n\n".join(sections)

    if "真實" in label:
        if confidence >= 95:
            title = "🟢【模型高度傾向真實】"
            description = (
                "模型高度傾向此內容為真實，"
                "但仍無法保證內容完全正確。"
            )
        elif confidence >= 80:
            title = "🟢【較可能是真實內容】"
            description = (
                "模型傾向此內容為真實，"
                "仍建議確認消息來源。"
            )
        elif confidence >= 60:
            title = "🟡【可能是真實內容】"
            description = (
                "模型初步判斷為真實，"
                "但信心度有限，建議再次查證。"
            )
        else:
            title = "⚪【判斷信心不足】"
            description = (
                "模型目前無法確定內容是否真實，"
                "請參考其他可靠來源。"
            )

        sections = [
            title,
            f"📊 模型信心度：{confidence:.2f}%",
        ]

        if probability_text:
            sections.append(probability_text)

        sections.extend([
            "🔎 判斷說明：\n" + description,
            (
                "✅ 查證建議：\n"
                "• 確認發布媒體或機構名稱\n"
                "• 確認文章發布日期\n"
                "• 搜尋其他媒體是否有相同報導\n"
                "• 優先參考政府或官方公告"
            ),
            (
                "ℹ️ AI 模型無法保證內容完全真實，"
                "仍建議透過可靠來源再次查證。"
            ),
        ])

        return "\n\n".join(sections)

    return (
        result
        + "\n\nℹ️ 此結果由 AI 模型產生，"
        "僅供參考，請搭配其他可靠來源查證。"
    )


# =========================================================
# 首頁與健康檢查
# =========================================================

@app.route("/", methods=["GET"])
def home():
    return {
        "status": "ok",
        "service": "LINE BERT scam detection bot",
        "hf_space": HF_SPACE_URL,
        "min_text_length": MIN_TEXT_LENGTH,
        "max_prediction_seconds": MAX_PREDICTION_SECONDS,
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
            elapsed = time.time() - start_time

            logger.exception(
                "模型預測失敗，耗時 %.2f 秒：%s",
                elapsed,
                error,
            )

            reply_text = (
                "⚠️ 模型目前正在啟動、更新或忙碌中。\n\n"
                "本次等待時間過長，為避免 LINE 回覆逾時，"
                "已先停止等待。\n\n"
                "請稍候約 30 秒後，再重新傳送一次。"
            )

    try:
        with ApiClient(configuration) as api_client:
            messaging_api = MessagingApi(
                api_client
            )

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

    except ApiException as error:
        error_text = str(error)

        if "invalid reply token" in error_text.lower():
            logger.warning(
                "LINE reply token 已失效或已使用，"
                "本次無法回覆。請檢查模型處理時間。"
            )
        else:
            logger.exception(
                "LINE Messaging API 回覆失敗：%s",
                error,
            )

    except Exception:
        logger.exception(
            "LINE 回覆失敗"
        )


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
