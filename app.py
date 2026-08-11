import os
import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import joblib
import numpy as np
import shap

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

from url_features import extract_url_features_25


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
# LINE SDK
# =========================================================

configuration = Configuration(
    access_token=LINE_CHANNEL_ACCESS_TOKEN
)
handler = WebhookHandler(LINE_CHANNEL_SECRET)


# =========================================================
# Random Forest / SHAP
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PHISHING_MODEL_PATH = os.path.join(
    BASE_DIR,
    "phishing_rf_model_25.pkl",
)

FEATURE_COLUMNS_PATH = os.path.join(
    BASE_DIR,
    "feature_columns_25.pkl",
)

try:
    phishing_model = joblib.load(PHISHING_MODEL_PATH)
    feature_columns_25 = joblib.load(FEATURE_COLUMNS_PATH)

    if int(phishing_model.n_features_in_) != len(feature_columns_25):
        raise RuntimeError(
            "Random Forest 模型特徵數量與 feature_columns_25.pkl 不一致。"
        )

    phishing_explainer = shap.TreeExplainer(phishing_model)

    logger.info(
        "Random Forest 模型載入成功，特徵數量：%s",
        phishing_model.n_features_in_,
    )
    logger.info("SHAP TreeExplainer 建立成功")

except Exception as error:
    logger.exception("網址模型或 SHAP 初始化失敗")
    raise RuntimeError(
        f"網址模型初始化失敗：{error}"
    ) from error


URL_FEATURE_NAMES = {
    "having_IPhaving_IP_Address": "網址是否直接使用 IP 位址",
    "URLURL_Length": "網址長度",
    "Shortining_Service": "短網址服務",
    "having_At_Symbol": "網址中的 @ 符號",
    "double_slash_redirecting": "網址重新導向結構",
    "Prefix_Suffix": "網域命名結構",
    "having_Sub_Domain": "子網域結構",
    "SSLfinal_State": "SSL / HTTPS 安全狀態",
    "Domain_registeration_length": "網域註冊期間",
    "Favicon": "網站圖示來源",
    "port": "網站連接埠",
    "HTTPS_token": "網域中的 HTTPS 字樣",
    "Request_URL": "網頁外部資源來源",
    "URL_of_Anchor": "網頁超連結結構",
    "Links_in_tags": "HTML 標籤連結結構",
    "SFH": "表單提交位置",
    "Submitting_to_email": "Email 資料提交行為",
    "Abnormal_URL": "網址結構一致性",
    "Redirect": "網址重新導向特徵",
    "on_mouseover": "滑鼠事件程式碼",
    "RightClick": "滑鼠右鍵限制",
    "popUpWidnow": "彈出視窗行為",
    "Iframe": "Iframe 嵌入內容",
    "age_of_domain": "網域建立時間",
    "DNSRecord": "DNS 紀錄",
}


# =========================================================
# URL 工具
# =========================================================

def is_url(text: str) -> bool:
    text = text.strip()
    return bool(
        re.fullmatch(
            r"https?://[^\s]+",
            text,
            flags=re.IGNORECASE,
        )
    )


def get_shap_values_for_class(features, predicted_class: int) -> np.ndarray:
    shap_values = phishing_explainer.shap_values(features)

    if isinstance(shap_values, list):
        class_values = np.asarray(shap_values[predicted_class])
        if class_values.ndim == 2:
            return class_values[0]
        return class_values.reshape(-1)

    shap_array = np.asarray(shap_values)

    if shap_array.ndim == 3:
        return shap_array[0, :, predicted_class]

    if shap_array.ndim == 2:
        return shap_array[0]

    if shap_array.ndim == 1:
        return shap_array

    raise RuntimeError(
        "無法辨識 SHAP 輸出格式："
        f"{shap_array.shape}"
    )


def build_shap_explanation(
    features,
    predicted_class: int,
    max_features: int = 5,
) -> str:
    current_shap = get_shap_values_for_class(
        features,
        predicted_class,
    )

    if len(current_shap) != len(feature_columns_25):
        raise RuntimeError(
            "SHAP 特徵數量與模型欄位數量不一致。"
        )

    explanation_items = []

    for index, feature_name in enumerate(feature_columns_25):
        shap_value = float(current_shap[index])
        explanation_items.append(
            {
                "feature": feature_name,
                "display_name": URL_FEATURE_NAMES.get(
                    feature_name,
                    feature_name,
                ),
                "shap_value": shap_value,
                "abs_shap": abs(shap_value),
            }
        )

    supporting = [
        item
        for item in explanation_items
        if item["shap_value"] > 0
    ]

    supporting.sort(
        key=lambda item: item["shap_value"],
        reverse=True,
    )

    if supporting:
        top_items = supporting[:max_features]
    else:
        top_items = sorted(
            explanation_items,
            key=lambda item: item["abs_shap"],
            reverse=True,
        )[:max_features]

    class_name = (
        "釣魚網站"
        if predicted_class == 1
        else "正常網站"
    )

    lines = [
        f"以下因素對本次「{class_name}」判斷影響較大："
    ]

    for number, item in enumerate(top_items, 1):
        lines.append(
            (
                f"\n{number}. {item['display_name']}\n"
                f"   → 提高模型判斷為「{class_name}」的傾向\n"
                f"   SHAP 貢獻值：{item['shap_value']:.4f}"
            )
        )

    lines.append(
        "\n📌 上述因素由 SHAP 依照本次模型預測計算，"
        "不是以人工關鍵字或固定規則產生。"
    )

    return "\n".join(lines)


def predict_phishing_url(url: str) -> str:
    logger.info("開始分析網址：%s", url)
    start_time = time.time()

    features = extract_url_features_25(
        url,
        feature_columns_25,
    )

    prediction = int(
        phishing_model.predict(features)[0]
    )

    probabilities = phishing_model.predict_proba(
        features
    )[0]

    normal_probability = float(probabilities[0] * 100)
    phishing_probability = float(probabilities[1] * 100)

    explanation = build_shap_explanation(
        features,
        prediction,
    )

    logger.info(
        "網址分析完成，耗時 %.2f 秒",
        time.time() - start_time,
    )

    if prediction == 1:
        if phishing_probability >= 90:
            title = "🔴【高度疑似釣魚網站】"
        elif phishing_probability >= 70:
            title = "🟠【高風險網站】"
        else:
            title = "🟡【疑似釣魚網站】"

        return (
            "🌐 網址安全檢測\n\n"
            f"{title}\n\n"
            f"🚨 釣魚機率：{phishing_probability:.2f}%\n"
            f"🟢 正常機率：{normal_probability:.2f}%\n\n"
            "🧠 AI 模型判斷依據（SHAP）\n\n"
            f"{explanation}\n\n"
            "⚠️ 安全提醒：\n"
            "• 請勿輸入帳號或密碼\n"
            "• 請勿輸入信用卡資料\n"
            "• 請勿提供簡訊驗證碼\n"
            "• 請勿下載不明檔案\n"
            "• 建議改由官方網站或 App 登入\n\n"
            "ℹ️ 此結果僅供輔助判斷，不代表最終安全認定。"
        )

    return (
        "🌐 網址安全檢測\n\n"
        "🟢【較可能為正常網站】\n\n"
        f"🟢 正常機率：{normal_probability:.2f}%\n"
        f"🚨 釣魚機率：{phishing_probability:.2f}%\n\n"
        "🧠 AI 模型判斷依據（SHAP）\n\n"
        f"{explanation}\n\n"
        "⚠️ 即使模型判斷為正常網站，仍請確認網址拼字、"
        "網域名稱與網站來源。\n\n"
        "ℹ️ 此結果僅供輔助判斷，不代表網站絕對安全。"
    )


# =========================================================
# Hugging Face BERT
# =========================================================

def _call_huggingface(text: str) -> str:
    logger.info("正在建立 Hugging Face Client")

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


def predict_with_huggingface(text: str) -> str:
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_call_huggingface, text)

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
# 一般文字驗證
# =========================================================

def validate_user_text(text: str) -> str | None:
    text = text.strip()

    if not text:
        return "請輸入想要辨識的文字內容。"

    if len(text) < MIN_TEXT_LENGTH:
        return (
            "⚠️ 輸入內容過短，可能影響 AI 判斷準確度。\n\n"
            "請貼上較完整的新聞或訊息內容，"
            f"建議至少 {MIN_TEXT_LENGTH} 個字。"
        )

    if len(text) > MAX_TEXT_LENGTH:
        return (
            "⚠️ 輸入內容過長。\n\n"
            f"請將文字縮短至 {MAX_TEXT_LENGTH} 字以內再試一次。"
        )

    return None


# =========================================================
# BERT 結果解析
# =========================================================

def extract_percentage(pattern: str, result: str) -> float | None:
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


def format_prediction_result(result: str, user_text: str) -> str:
    logger.info("BERT 原始回傳結果：%s", result)

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
            + "\n\nℹ️ 此結果由 AI 模型產生，僅供參考，"
            "請搭配其他可靠來源查證。"
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

    probability_text = "\n".join(probability_lines)

    if "詐騙" in label:
        if confidence >= 95:
            title = "🔴【模型高度傾向詐騙】"
        elif confidence >= 80:
            title = "🟠【高風險訊息】"
        elif confidence >= 60:
            title = "🟡【疑似詐騙】"
        else:
            title = "⚪【判斷信心不足】"

        sections = [
            title,
            f"📊 模型信心度：{confidence:.2f}%",
        ]

        if probability_text:
            sections.append(probability_text)

        sections.extend(
            [
                (
                    "🔎 模型判斷說明：\n"
                    "此結果來自 BERT 對整段文字語意的分類結果。"
                ),
                (
                    "🧠 可解釋性狀態：\n"
                    "已停用人工關鍵字作為模型判斷依據。"
                    "後續將改用 Token Attribution 分析真正影響 "
                    "BERT 預測的文字片段。"
                ),
                (
                    "⚠️ 防詐提醒：\n"
                    "• 請勿立即匯款或轉帳\n"
                    "• 請勿提供銀行帳號或信用卡資料\n"
                    "• 請勿提供密碼或簡訊驗證碼\n"
                    "• 請勿點擊不明連結\n"
                    "• 建議透過官方管道再次查證"
                ),
                (
                    "ℹ️ 此結果僅供輔助判斷，不代表最終事實認定。"
                ),
            ]
        )

        return "\n\n".join(sections)

    if "真實" in label:
        if confidence >= 95:
            title = "🟢【模型高度傾向真實】"
        elif confidence >= 80:
            title = "🟢【較可能是真實內容】"
        elif confidence >= 60:
            title = "🟡【可能是真實內容】"
        else:
            title = "⚪【判斷信心不足】"

        sections = [
            title,
            f"📊 模型信心度：{confidence:.2f}%",
        ]

        if probability_text:
            sections.append(probability_text)

        sections.extend(
            [
                (
                    "🔎 模型判斷說明：\n"
                    "此結果來自 BERT 對整段文字語意的分類結果。"
                ),
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
            ]
        )

        return "\n\n".join(sections)

    return (
        result
        + "\n\nℹ️ 此結果由 AI 模型產生，僅供參考。"
    )


# =========================================================
# 首頁 / Health Check
# =========================================================

@app.route("/", methods=["GET"])
def home():
    return {
        "status": "ok",
        "service": "LINE scam + phishing detection bot",
        "text_model": "BERT",
        "url_model": "Random Forest",
        "url_features": 25,
        "url_explainability": "SHAP TreeExplainer",
        "text_explainability": "pending token attribution",
        "hf_space": HF_SPACE_URL,
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

    body = request.get_data(as_text=True)

    logger.info(
        "收到 LINE Webhook，內容長度：%s",
        len(body),
    )

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
# LINE 文字訊息
# =========================================================

@handler.add(
    MessageEvent,
    message=TextMessageContent,
)
def handle_text_message(event):
    user_text = event.message.text.strip()

    logger.info(
        "收到使用者訊息，字數：%s",
        len(user_text),
    )

    if is_url(user_text):
        logger.info(
            "偵測到 URL，進入 Random Forest + SHAP"
        )

        try:
            reply_text = predict_phishing_url(user_text)

        except Exception as error:
            logger.exception(
                "網址分析失敗：%s",
                error,
            )

            reply_text = (
                "⚠️ 網址目前無法完成分析。\n\n"
                "可能原因包括網站無法連線、"
                "網域資料暫時無法取得，"
                "或網站阻擋自動分析。\n\n"
                "請稍後再試一次。"
            )

    else:
        validation_message = validate_user_text(user_text)

        if validation_message:
            reply_text = validation_message

        else:
            start_time = time.time()

            try:
                model_result = predict_with_huggingface(
                    user_text
                )

                reply_text = format_prediction_result(
                    model_result,
                    user_text,
                )

                logger.info(
                    "BERT 預測完成，耗時 %.2f 秒",
                    time.time() - start_time,
                )

            except Exception as error:
                logger.exception(
                    "BERT 預測失敗，耗時 %.2f 秒：%s",
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
                    messages=[
                        TextMessage(text=reply_text)
                    ],
                )
            )

        logger.info("LINE 回覆成功")

    except ApiException as error:
        error_text = str(error)

        if "invalid reply token" in error_text.lower():
            logger.warning(
                "LINE reply token 已失效或已使用"
            )
        else:
            logger.exception(
                "LINE Messaging API 回覆失敗：%s",
                error,
            )

    except Exception:
        logger.exception("LINE 回覆失敗")


# =========================================================
# 本機執行
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
