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

# Hugging Face BERT V3 Space
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


# =========================================================
# 3. 檢查 LINE 環境變數
# =========================================================

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
# 4. LINE SDK
# =========================================================

configuration = Configuration(
    access_token=LINE_CHANNEL_ACCESS_TOKEN
)

handler = WebhookHandler(
    LINE_CHANNEL_SECRET
)


# =========================================================
# 5. Random Forest / SHAP
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PHISHING_MODEL_PATH = os.path.join(
    BASE_DIR,
    "phishing_rf_model_25.pkl",
)

FEATURE_COLUMNS_PATH = os.path.join(
    BASE_DIR,
    "feature_columns_25.pkl",
)


try:

    phishing_model = joblib.load(
        PHISHING_MODEL_PATH
    )

    feature_columns_25 = joblib.load(
        FEATURE_COLUMNS_PATH
    )

    if (
        int(phishing_model.n_features_in_)
        != len(feature_columns_25)
    ):
        raise RuntimeError(
            "Random Forest 模型特徵數量與 "
            "feature_columns_25.pkl 不一致。"
        )

    phishing_explainer = shap.TreeExplainer(
        phishing_model
    )

    logger.info(
        "Random Forest 模型載入成功，特徵數量：%s",
        phishing_model.n_features_in_,
    )

    logger.info(
        "SHAP TreeExplainer 建立成功"
    )

except Exception as error:

    logger.exception(
        "網址模型或 SHAP 初始化失敗"
    )

    raise RuntimeError(
        f"網址模型初始化失敗：{error}"
    ) from error


# =========================================================
# 6. URL 特徵中文名稱
# =========================================================

URL_FEATURE_NAMES = {

    "having_IPhaving_IP_Address":
        "網址是否直接使用 IP 位址",

    "URLURL_Length":
        "網址長度",

    "Shortining_Service":
        "短網址服務",

    "having_At_Symbol":
        "網址中的 @ 符號",

    "double_slash_redirecting":
        "網址重新導向結構",

    "Prefix_Suffix":
        "網域命名結構",

    "having_Sub_Domain":
        "子網域結構",

    "SSLfinal_State":
        "SSL / HTTPS 安全狀態",

    "Domain_registeration_length":
        "網域註冊期間",

    "Favicon":
        "網站圖示來源",

    "port":
        "網站連接埠",

    "HTTPS_token":
        "網域中的 HTTPS 字樣",

    "Request_URL":
        "網頁外部資源來源",

    "URL_of_Anchor":
        "網頁超連結結構",

    "Links_in_tags":
        "HTML 標籤連結結構",

    "SFH":
        "表單提交位置",

    "Submitting_to_email":
        "Email 資料提交行為",

    "Abnormal_URL":
        "網址結構一致性",

    "Redirect":
        "網址重新導向特徵",

    "on_mouseover":
        "滑鼠事件程式碼",

    "RightClick":
        "滑鼠右鍵限制",

    "popUpWidnow":
        "彈出視窗行為",

    "Iframe":
        "Iframe 嵌入內容",

    "age_of_domain":
        "網域建立時間",

    "DNSRecord":
        "DNS 紀錄",
}


# =========================================================
# 7. 判斷是不是 URL
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


# =========================================================
# 8. 取得 SHAP values
# =========================================================

def get_shap_values_for_class(
    features,
    predicted_class: int,
) -> np.ndarray:

    shap_values = (
        phishing_explainer.shap_values(
            features
        )
    )

    # 舊版 SHAP
    if isinstance(shap_values, list):

        class_values = np.asarray(
            shap_values[predicted_class]
        )

        if class_values.ndim == 2:
            return class_values[0]

        return class_values.reshape(-1)

    shap_array = np.asarray(
        shap_values
    )

    # 新版 SHAP
    if shap_array.ndim == 3:

        return shap_array[
            0,
            :,
            predicted_class
        ]

    if shap_array.ndim == 2:

        return shap_array[0]

    if shap_array.ndim == 1:

        return shap_array

    raise RuntimeError(
        "無法辨識 SHAP 輸出格式："
        f"{shap_array.shape}"
    )


# =========================================================
# 9. 建立 SHAP 解釋
# =========================================================

def build_shap_explanation(
    features,
    predicted_class: int,
    max_features: int = 5,
) -> str:

    current_shap = (
        get_shap_values_for_class(
            features,
            predicted_class,
        )
    )

    if (
        len(current_shap)
        != len(feature_columns_25)
    ):
        raise RuntimeError(
            "SHAP 特徵數量與模型欄位數量不一致。"
        )

    explanation_items = []

    for index, feature_name in enumerate(
        feature_columns_25
    ):

        shap_value = float(
            current_shap[index]
        )

        explanation_items.append(
            {
                "feature": feature_name,

                "display_name":
                    URL_FEATURE_NAMES.get(
                        feature_name,
                        feature_name,
                    ),

                "shap_value":
                    shap_value,

                "abs_shap":
                    abs(shap_value),
            }
        )

    supporting = [
        item
        for item in explanation_items
        if item["shap_value"] > 0
    ]

    supporting.sort(
        key=lambda item:
            item["shap_value"],
        reverse=True,
    )

    if supporting:

        top_items = supporting[
            :max_features
        ]

    else:

        top_items = sorted(
            explanation_items,
            key=lambda item:
                item["abs_shap"],
            reverse=True,
        )[:max_features]

    class_name = (
        "釣魚網站"
        if predicted_class == 1
        else "正常網站"
    )

    lines = [
        f"以下因素對本次「{class_name}」"
        "判斷影響較大："
    ]

    for number, item in enumerate(
        top_items,
        1,
    ):

        lines.append(
            (
                f"\n{number}. "
                f"{item['display_name']}\n"

                f"   → 提高模型判斷為"
                f"「{class_name}」的傾向\n"

                f"   SHAP 貢獻值："
                f"{item['shap_value']:.4f}"
            )
        )

    lines.append(
        "\n📌 上述因素由 SHAP "
        "依照本次模型預測計算，"
        "不是以人工關鍵字或固定規則產生。"
    )

    return "\n".join(lines)


# =========================================================
# 10. Random Forest 網址預測
# =========================================================

def predict_phishing_url(
    url: str
) -> str:

    logger.info(
        "開始分析網址：%s",
        url,
    )

    start_time = time.time()

    features = (
        extract_url_features_25(
            url,
            feature_columns_25,
        )
    )

    prediction = int(
        phishing_model.predict(
            features
        )[0]
    )

    probabilities = (
        phishing_model.predict_proba(
            features
        )[0]
    )

    normal_probability = float(
        probabilities[0] * 100
    )

    phishing_probability = float(
        probabilities[1] * 100
    )

    explanation = (
        build_shap_explanation(
            features,
            prediction,
        )
    )

    logger.info(
        "網址分析完成，耗時 %.2f 秒",
        time.time() - start_time,
    )

    # -----------------------------------------------------
    # 釣魚網站
    # -----------------------------------------------------

    if prediction == 1:

        if phishing_probability >= 90:

            title = (
                "🔴【高度疑似釣魚網站】"
            )

        elif phishing_probability >= 70:

            title = (
                "🟠【高風險網站】"
            )

        else:

            title = (
                "🟡【疑似釣魚網站】"
            )

        return (
            "🌐 網址安全檢測\n\n"

            f"{title}\n\n"

            f"🚨 釣魚機率："
            f"{phishing_probability:.2f}%\n"

            f"🟢 正常機率："
            f"{normal_probability:.2f}%\n\n"

            "🧠 AI 模型判斷依據（SHAP）\n\n"

            f"{explanation}\n\n"

            "⚠️ 安全提醒：\n"

            "• 請勿輸入帳號或密碼\n"
            "• 請勿輸入信用卡資料\n"
            "• 請勿提供簡訊驗證碼\n"
            "• 請勿下載不明檔案\n"
            "• 建議改由官方網站或 App 登入\n\n"

            "ℹ️ 此結果僅供輔助判斷，"
            "不代表最終安全認定。"
        )

    # -----------------------------------------------------
    # 正常網站
    # -----------------------------------------------------

    return (
        "🌐 網址安全檢測\n\n"

        "🟢【較可能為正常網站】\n\n"

        f"🟢 正常機率："
        f"{normal_probability:.2f}%\n"

        f"🚨 釣魚機率："
        f"{phishing_probability:.2f}%\n\n"

        "🧠 AI 模型判斷依據（SHAP）\n\n"

        f"{explanation}\n\n"

        "⚠️ 即使模型判斷為正常網站，"
        "仍請確認網址拼字、"
        "網域名稱與網站來源。\n\n"

        "ℹ️ 此結果僅供輔助判斷，"
        "不代表網站絕對安全。"
    )


# =========================================================
# 11. Hugging Face BERT V3
# =========================================================

def _call_huggingface(
    text: str
) -> str:

    logger.info(
        "正在建立 Hugging Face Client"
    )

    client = Client(
        HF_SPACE_URL,
        verbose=False,
    )

    # 呼叫 Hugging Face Space
    # BERT V3 + Integrated Gradients

    result = client.predict(
        text,
        api_name="/predict",
    )

    logger.info(
        "Hugging Face 預測成功"
    )

    return str(result)


# =========================================================
# 12. Hugging Face Timeout
# =========================================================

def predict_with_huggingface(
    text: str
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
            "Hugging Face 呼叫失敗："
            f"{error}"
        ) from error

    finally:

        executor.shutdown(
            wait=False,
            cancel_futures=True,
        )


# =========================================================
# 13. 一般文字驗證
# =========================================================

def validate_user_text(
    text: str
) -> str | None:

    text = text.strip()

    if not text:

        return (
            "請輸入想要辨識的文字內容。"
        )

    if len(text) < MIN_TEXT_LENGTH:

        return (
            "⚠️ 輸入內容過短，"
            "可能影響 AI 判斷準確度。\n\n"

            "請貼上較完整的訊息內容，"
            f"建議至少 {MIN_TEXT_LENGTH} 個字。"
        )

    if len(text) > MAX_TEXT_LENGTH:

        return (
            "⚠️ 輸入內容過長。\n\n"

            f"請將文字縮短至 "
            f"{MAX_TEXT_LENGTH} 字以內"
            "再試一次。"
        )

    return None


# =========================================================
# 14. 首頁
# =========================================================

@app.route(
    "/",
    methods=["GET"],
)
def home():

    return {

        "status":
            "ok",

        "service":
            "LINE scam + phishing detection bot",

        "text_model":
            "BERT V3",

        "text_explainability":
            "Integrated Gradients",

        "url_model":
            "Random Forest",

        "url_features":
            25,

        "url_explainability":
            "SHAP TreeExplainer",

        "hf_space":
            HF_SPACE_URL,

    }, 200


# =========================================================
# 15. Health Check
# =========================================================

@app.route(
    "/health",
    methods=["GET"],
)
def health():

    return "OK", 200


# =========================================================
# 16. LINE Webhook
# =========================================================

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
# 17. LINE 文字訊息
# =========================================================

@handler.add(
    MessageEvent,
    message=TextMessageContent,
)
def handle_text_message(event):

    user_text = (
        event.message.text.strip()
    )

    logger.info(
        "收到使用者訊息，字數：%s",
        len(user_text),
    )

    # =====================================================
    # A. URL
    # Random Forest + SHAP
    # =====================================================

    if is_url(user_text):

        logger.info(
            "偵測到 URL，"
            "進入 Random Forest + SHAP"
        )

        try:

            reply_text = (
                predict_phishing_url(
                    user_text
                )
            )

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

    # =====================================================
    # B. 一般文字
    # BERT V3 + Integrated Gradients
    # =====================================================

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

                # =========================================
                # 呼叫 Hugging Face V3
                # =========================================

                model_result = (
                    predict_with_huggingface(
                        user_text
                    )
                )

                # =========================================
                # 重要：
                #
                # Hugging Face V3 已經完成：
                #
                # 1. BERT 詐騙 / 正常分類
                # 2. 模型信心度
                # 3. 詐騙機率
                # 4. 正常機率
                # 5. Integrated Gradients
                # 6. 高影響文字片段
                #
                # 所以不要再重新格式化。
                # 直接完整回傳給 LINE。
                # =========================================

                reply_text = (
                    model_result
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


    # =====================================================
    # 18. 回覆 LINE
    # =====================================================

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

                    reply_token=
                        event.reply_token,

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

        error_text = str(error)

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
# 19. 本機 / Render 執行
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
