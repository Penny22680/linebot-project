import os
import re
import time
import logging
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)

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
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
)

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

LINE_CHANNEL_ACCESS_TOKEN = os.getenv(
    "LINE_CHANNEL_ACCESS_TOKEN"
)

LINE_CHANNEL_SECRET = os.getenv(
    "LINE_CHANNEL_SECRET"
)

HF_SPACE_URL = os.getenv(
    "HF_SPACE_URL",
    "https://penny0922-linebot-bert-api.hf.space",
)

MAX_PREDICTION_SECONDS = int(
    os.getenv(
        "MAX_PREDICTION_SECONDS",
        "40",
    )
)

MIN_TEXT_LENGTH = int(
    os.getenv(
        "MIN_TEXT_LENGTH",
        "50",
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
        "請到 Render Environment Variables 設定。"
    )

if not LINE_CHANNEL_SECRET:
    raise RuntimeError(
        "找不到 LINE_CHANNEL_SECRET，"
        "請到 Render Environment Variables 設定。"
    )


# =========================================================
# LINE SDK
# =========================================================

configuration = Configuration(
    access_token=LINE_CHANNEL_ACCESS_TOKEN
)

handler = WebhookHandler(
    LINE_CHANNEL_SECRET
)


# =========================================================
# Random Forest 模型檔案位置
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


# =========================================================
# 載入 Random Forest
# =========================================================

try:

    phishing_model = joblib.load(
        PHISHING_MODEL_PATH
    )

    feature_columns_25 = joblib.load(
        FEATURE_COLUMNS_PATH
    )

    logger.info(
        "✅ Random Forest 模型載入成功"
    )

    logger.info(
        "模型特徵數量：%s",
        phishing_model.n_features_in_,
    )

    logger.info(
        "欄位數量：%s",
        len(feature_columns_25),
    )

except Exception as error:

    logger.exception(
        "❌ Random Forest 模型載入失敗"
    )

    raise RuntimeError(
        f"Random Forest 模型載入失敗：{error}"
    ) from error


# =========================================================
# 建立 SHAP TreeExplainer
# =========================================================

try:

    phishing_explainer = shap.TreeExplainer(
        phishing_model
    )

    logger.info(
        "✅ SHAP TreeExplainer 建立成功"
    )

except Exception as error:

    logger.exception(
        "❌ SHAP TreeExplainer 建立失敗"
    )

    raise RuntimeError(
        f"SHAP 初始化失敗：{error}"
    ) from error


# =========================================================
# URL 特徵中文名稱
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
# 判斷使用者輸入是否為 URL
# =========================================================

def is_url(text: str) -> bool:

    text = text.strip()

    pattern = re.compile(
        r"^https?://[^\s]+$",
        re.IGNORECASE,
    )

    return bool(
        pattern.match(text)
    )


# =========================================================
# SHAP 輸出格式相容處理
# =========================================================

def get_shap_values_for_class(
    features,
    predicted_class: int,
):

    shap_values = (
        phishing_explainer.shap_values(
            features
        )
    )

    # =============================================
    # 舊版 SHAP：
    # [
    #   class_0_values,
    #   class_1_values
    # ]
    # =============================================

    if isinstance(
        shap_values,
        list,
    ):

        class_values = np.asarray(
            shap_values[
                predicted_class
            ]
        )

        return class_values[0]


    shap_array = np.asarray(
        shap_values
    )

    # =============================================
    # 新版：
    # (samples, features, classes)
    # 例如你目前：
    # (1, 25, 2)
    # =============================================

    if shap_array.ndim == 3:

        return shap_array[
            0,
            :,
            predicted_class,
        ]


    # =============================================
    # 部分 SHAP 版本：
    # (samples, features)
    # =============================================

    if shap_array.ndim == 2:

        return shap_array[0]


    raise RuntimeError(
        "無法辨識 SHAP 輸出格式："
        f"{shap_array.shape}"
    )


# =========================================================
# 產生真正的 SHAP 使用者解釋
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

    feature_values = (
        features.iloc[0].values
    )

    explanations = []

    for index, feature_name in enumerate(
        feature_columns_25
    ):

        shap_value = float(
            current_shap[index]
        )

        feature_value = (
            feature_values[index]
        )

        explanations.append({
            "feature": feature_name,
            "display_name":
                URL_FEATURE_NAMES.get(
                    feature_name,
                    feature_name,
                ),
            "value": feature_value,
            "shap_value": shap_value,
            "abs_shap": abs(
                shap_value
            ),
        })


    # =====================================================
    # 優先找真正支持「本次預測類別」的正 SHAP
    # =====================================================

    supporting = [
        item
        for item in explanations
        if item["shap_value"] > 0
    ]

    supporting.sort(
        key=lambda item:
            item["shap_value"],
        reverse=True,
    )


    # 如果極端情況沒有正值，再以絕對值排序
    if not supporting:

        supporting = sorted(
            explanations,
            key=lambda item:
                item["abs_shap"],
            reverse=True,
        )


    top_items = supporting[
        :max_features
    ]


    if predicted_class == 1:

        class_name = (
            "釣魚網站"
        )

    else:

        class_name = (
            "正常網站"
        )


    lines = [
        (
            f"以下因素對本次"
            f"「{class_name}」判斷"
            f"影響較大："
        )
    ]


    for number, item in enumerate(
        top_items,
        1,
    ):

        display_name = (
            item["display_name"]
        )

        shap_value = (
            item["shap_value"]
        )

        lines.append(
            (
                f"\n{number}. "
                f"{display_name}\n"
                f"   → 提高模型判斷為"
                f"「{class_name}」的傾向\n"
                f"   SHAP 貢獻值："
                f"{shap_value:.4f}"
            )
        )


    lines.append(
        "\n📌 上述因素由 SHAP "
        "依照本次模型預測計算，"
        "並非使用人工關鍵字規則。"
    )

    return "\n".join(
        lines
    )


# =========================================================
# Random Forest URL 預測
# =========================================================

def predict_phishing_url(
    url: str,
) -> str:

    logger.info(
        "開始分析網址：%s",
        url,
    )

    start_time = time.time()


    # =====================================================
    # 25 個即時特徵
    # =====================================================

    features = extract_url_features_25(
        url,
        feature_columns_25,
    )


    # =====================================================
    # 模型預測
    # =====================================================

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
        probabilities[0]
        * 100
    )

    phishing_probability = float(
        probabilities[1]
        * 100
    )


    # =====================================================
    # SHAP Local Explanation
    # =====================================================

    explanation = (
        build_shap_explanation(
            features,
            prediction,
        )
    )


    elapsed = (
        time.time()
        - start_time
    )

    logger.info(
        "網址分析完成，耗時 %.2f 秒",
        elapsed,
    )


    # =====================================================
    # 預測為釣魚
    # =====================================================

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


    # =====================================================
    # 預測為正常
    # =====================================================

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
# Hugging Face BERT
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

    logger.info(
        "正在呼叫 Hugging Face /predict"
    )

    result = client.predict(
        text,
        api_name="/predict",
    )

    logger.info(
        "Hugging Face 預測成功"
    )

    return str(result)


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
            f"Hugging Face 呼叫失敗："
            f"{error}"
        ) from error

    finally:

        executor.shutdown(
            wait=False,
            cancel_futures=True,
        )


# =========================================================
# 一般文字長度檢查
# =========================================================

def validate_user_text(
    text: str,
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
            "請貼上較完整的新聞或訊息內容，"
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
# BERT 百分比解析
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

        return float(
            match.group(1)
        )

    except (
        TypeError,
        ValueError,
    ):

        return None


# =========================================================
# BERT 回覆格式
#
# 重要：
# 目前移除人工關鍵字「判斷依據」。
#
# 因為 BERT 真正可解釋性之後要改成
# Integrated Gradients / SHAP Token attribution。
# =========================================================

def format_prediction_result(
    result: str,
    user_text: str,
) -> str:

    logger.info(
        "BERT 原始回傳結果：%s",
        result,
    )

    result_lower = (
        result.lower()
    )


    label_match = re.search(
        r"判斷結果[：:]\s*"
        r"(詐騙訊息|真實新聞|"
        r"真實訊息|詐騙|真實)",
        result,
        flags=re.IGNORECASE,
    )


    confidence = extract_percentage(
        r"模型信心度[：:]\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*%",
        result,
    )


    scam_probability = extract_percentage(
        r"詐騙機率[：:]\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*%",
        result,
    )


    real_probability = extract_percentage(
        r"真實機率[：:]\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*%",
        result,
    )


    if label_match:

        label = (
            label_match.group(1)
        )

    elif "scam" in result_lower:

        label = "詐騙"

    elif (
        "real" in result_lower
        or "true" in result_lower
    ):

        label = "真實"

    else:

        return (
            result
            + "\n\nℹ️ 此結果由 AI 模型產生，"
            "僅供參考，請搭配其他可靠來源查證。"
        )


    if confidence is None:

        if (
            "詐騙" in label
            and scam_probability
            is not None
        ):

            confidence = (
                scam_probability
            )

        elif (
            "真實" in label
            and real_probability
            is not None
        ):

            confidence = (
                real_probability
            )

        else:

            confidence = 0.0


    probability_lines = []


    if scam_probability is not None:

        probability_lines.append(
            f"🚨 詐騙機率："
            f"{scam_probability:.2f}%"
        )


    if real_probability is not None:

        probability_lines.append(
            f"✅ 真實機率："
            f"{real_probability:.2f}%"
        )


    probability_text = "\n".join(
        probability_lines
    )


    # =====================================================
    # 詐騙
    # =====================================================

    if "詐騙" in label:

        if confidence >= 95:

            title = (
                "🔴【模型高度傾向詐騙】"
            )

        elif confidence >= 80:

            title = (
                "🟠【高風險訊息】"
            )

        elif confidence >= 60:

            title = (
                "🟡【疑似詐騙】"
            )

        else:

            title = (
                "⚪【判斷信心不足】"
            )


        sections = [
            title,
            (
                f"📊 模型信心度："
                f"{confidence:.2f}%"
            ),
        ]


        if probability_text:

            sections.append(
                probability_text
            )


        sections.extend([

            (
                "🔎 模型判斷說明：\n"
                "此結果來自 BERT 對整段文字"
                "語意的分類結果。"
            ),

            (
                "🧠 可解釋性狀態：\n"
                "目前已停用人工關鍵字作為"
                "模型判斷依據。\n"
                "後續將使用 Token attribution "
                "方法分析真正影響 BERT "
                "預測的文字片段。"
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
                "ℹ️ 此結果僅供輔助判斷，"
                "不代表最終事實認定。"
            ),
        ])


        return "\n\n".join(
            sections
        )


    # =====================================================
    # 真實
    # =====================================================

    if "真實" in label:

        if confidence >= 95:

            title = (
                "🟢【模型高度傾向真實】"
            )

        elif confidence >= 80:

            title = (
                "🟢【較可能是真實內容】"
            )

        elif confidence >= 60:

            title = (
                "🟡【可能是真實內容】"
            )

        else:

            title = (
                "⚪【判斷信心不足】"
            )


        sections = [
            title,

            (
                f"📊 模型信心度："
                f"{confidence:.2f}%"
            ),
        ]


        if probability_text:

            sections.append(
                probability_text
            )


        sections.extend([

            (
                "🔎 模型判斷說明：\n"
                "此結果來自 BERT 對整段文字"
                "語意的分類結果。"
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
        ])


        return "\n\n".join(
            sections
        )


    return (
        result
        + "\n\nℹ️ 此結果由 AI 模型產生，"
        "僅供參考。"
    )


# =========================================================
# 首頁
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
            "BERT",

        "url_model":
            "Random Forest",

        "url_features":
            25,

        "url_explainability":
            "SHAP TreeExplainer",

        "text_explainability":
            "pending token attribution",

        "hf_space":
            HF_SPACE_URL,

    }, 200


@app.route(
    "/health",
    methods=["GET"],
)
def health():

    return "OK", 200


# =========================================================
# LINE Webhook
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
# LINE 文字訊息
# =========================================================

@handler.add(
    MessageEvent,
    message=TextMessageContent,
)
def handle_text_message(
    event,
):

    user_text = (
        event.message.text.strip()
    )


    logger.info(
        "收到使用者訊息，字數：%s",
        len(user_text),
    )


    # =====================================================
    # URL → Random Forest + SHAP
    # =====================================================

    if is_url(
        user_text
    ):

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
    # 一般文字 → BERT
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

            start_time = (
                time.time()
            )


            try:

                model_result = (
                    predict_with_huggingface(
                        user_text
                    )
                )


                reply_text = (
                    format_prediction_result(
                        model_result,
                        user_text,
                    )
                )


                elapsed = (
                    time.time()
                    - start_time
                )


                logger.info(
                    "BERT 預測完成，"
                    "耗時 %.2f 秒",
                    elapsed,
                )


            except Exception as error:

                elapsed = (
                    time.time()
                    - start_time
                )


                logger.exception(
                    "BERT 預測失敗，"
                    "耗時 %.2f 秒：%s",
                    elapsed,
                    error,
                )


                reply_text = (
                    "⚠️ 文字模型目前正在啟動、"
                    "更新或忙碌中。\n\n"
                    "請稍候約 30 秒後"
                    "重新傳送一次。"
                )


    # =====================================================
    # LINE Reply
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
    )app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


# =========================================================
# Render Environment Variables
# =========================================================

LINE_CHANNEL_ACCESS_TOKEN = os.getenv(
    "LINE_CHANNEL_ACCESS_TOKEN"
)

LINE_CHANNEL_SECRET = os.getenv(
    "LINE_CHANNEL_SECRET"
)

HF_SPACE_URL = os.getenv(
    "HF_SPACE_URL",
    "https://penny0922-linebot-bert-api.hf.space",
)

MAX_PREDICTION_SECONDS = int(
    os.getenv(
        "MAX_PREDICTION_SECONDS",
        "40",
    )
)

MIN_TEXT_LENGTH = int(
    os.getenv(
        "MIN_TEXT_LENGTH",
        "50",
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
# LINE SDK
# =========================================================

configuration = Configuration(
    access_token=LINE_CHANNEL_ACCESS_TOKEN
)

handler = WebhookHandler(
    LINE_CHANNEL_SECRET
)


# =========================================================
# 載入 Random Forest 網址模型
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

    logger.info(
        "釣魚網址 Random Forest 模型載入成功"
    )

    logger.info(
        "模型特徵數量：%s",
        phishing_model.n_features_in_,
    )

    logger.info(
        "欄位數量：%s",
        len(feature_columns_25),
    )

except Exception as error:

    logger.exception(
        "釣魚網址模型載入失敗"
    )

    raise RuntimeError(
        f"釣魚網址模型載入失敗：{error}"
    ) from error


# =========================================================
# URL 判斷
# =========================================================

def is_url(text: str) -> bool:

    text = text.strip()

    pattern = re.compile(
        r"^https?://"
        r"[^\s]+$",
        re.IGNORECASE,
    )

    return bool(
        pattern.match(text)
    )


# =========================================================
# URL 特徵說明
# =========================================================

def build_url_explanation(features) -> str:

    row = features.iloc[0]

    findings = []

    # SSL
    if row["SSLfinal_State"] == -1:
        findings.append(
            "• SSL / HTTPS 狀態具有風險"
        )

    # IP
    if row["having_IPhaving_IP_Address"] == -1:
        findings.append(
            "• 網址直接使用 IP 位址"
        )

    # URL 長度
    if row["URLURL_Length"] == -1:
        findings.append(
            "• 網址長度異常偏長"
        )

    # 短網址
    if row["Shortining_Service"] == -1:
        findings.append(
            "• 使用短網址服務，"
            "可能隱藏實際目的網址"
        )

    # @
    if row["having_At_Symbol"] == -1:
        findings.append(
            "• 網址包含 @ 符號"
        )

    # //
    if row["double_slash_redirecting"] == -1:
        findings.append(
            "• 網址存在可疑重新導向結構"
        )

    # -
    if row["Prefix_Suffix"] == -1:
        findings.append(
            "• 網域名稱包含可疑連字號結構"
        )

    # Subdomain
    if row["having_Sub_Domain"] == -1:
        findings.append(
            "• 網址包含較複雜的子網域結構"
        )

    # Domain registration
    if row["Domain_registeration_length"] == -1:
        findings.append(
            "• 網域註冊期間較短"
        )

    # Favicon
    if row["Favicon"] == -1:
        findings.append(
            "• 網站圖示來源與主要網域不一致"
        )

    # Port
    if row["port"] == -1:
        findings.append(
            "• 網址使用非標準連接埠"
        )

    # HTTPS token
    if row["HTTPS_token"] == -1:
        findings.append(
            "• 網域名稱中出現可疑 HTTPS 字樣"
        )

    # Request URL
    if row["Request_URL"] == -1:
        findings.append(
            "• 網頁載入大量外部來源資源"
        )

    # Anchor
    if row["URL_of_Anchor"] == -1:
        findings.append(
            "• 網頁連結結構具有較高風險"
        )

    # Links in tags
    if row["Links_in_tags"] == -1:
        findings.append(
            "• HTML 標籤中的外部連結比例異常"
        )

    # SFH
    if row["SFH"] == -1:
        findings.append(
            "• 表單提交位置具有可疑特徵"
        )

    # Email
    if row["Submitting_to_email"] == -1:
        findings.append(
            "• 網站可能透過 Email 提交資料"
        )

    # Abnormal URL
    if row["Abnormal_URL"] == -1:
        findings.append(
            "• 網址結構具有異常特徵"
        )

    # Redirect
    if row["Redirect"] == -1:
        findings.append(
            "• 偵測到可疑重新導向特徵"
        )

    # Mouseover
    if row["on_mouseover"] == -1:
        findings.append(
            "• 網頁使用可疑滑鼠事件程式碼"
        )

    # Right click
    if row["RightClick"] == -1:
        findings.append(
            "• 網頁可能限制滑鼠右鍵操作"
        )

    # Popup
    if row["popUpWidnow"] == -1:
        findings.append(
            "• 網頁包含可疑彈出視窗行為"
        )

    # iframe
    if row["Iframe"] == -1:
        findings.append(
            "• 網頁使用 iframe 嵌入內容"
        )

    # Domain age
    if row["age_of_domain"] == -1:
        findings.append(
            "• 網域建立時間較短"
        )

    # DNS
    if row["DNSRecord"] == -1:
        findings.append(
            "• DNS 紀錄具有異常或無法確認"
        )

    if not findings:
        return (
            "• 未偵測到明顯的高風險網址特徵\n"
            "• 模型主要根據 25 項網址與網站結構特徵綜合判斷"
        )

    return "\n".join(
        findings[:6]
    )


# =========================================================
# Random Forest 網址預測
# =========================================================

def predict_phishing_url(url: str) -> str:

    logger.info(
        "開始分析網址：%s",
        url,
    )

    start_time = time.time()

    features = extract_url_features_25(
        url,
        feature_columns_25,
    )

    prediction = phishing_model.predict(
        features
    )[0]

    probabilities = (
        phishing_model.predict_proba(
            features
        )[0]
    )

    normal_probability = (
        probabilities[0] * 100
    )

    phishing_probability = (
        probabilities[1] * 100
    )

    explanation = build_url_explanation(
        features
    )

    elapsed = (
        time.time() - start_time
    )

    logger.info(
        "網址分析完成，耗時 %.2f 秒",
        elapsed,
    )

    # =====================================================
    # 釣魚
    # =====================================================

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
            "🔎 判斷依據：\n"
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

    # =====================================================
    # 正常
    # =====================================================

    return (
        "🌐 網址安全檢測\n\n"
        "🟢【較可能為正常網站】\n\n"
        f"🟢 正常機率："
        f"{normal_probability:.2f}%\n"
        f"🚨 釣魚機率："
        f"{phishing_probability:.2f}%\n\n"
        "🔎 判斷說明：\n"
        "模型未偵測到足以判定為釣魚網站的"
        "高風險特徵。\n\n"
        "⚠️ 即使模型判斷為正常網站，"
        "仍請確認網址拼字、網域名稱與網站來源。\n\n"
        "ℹ️ 此結果僅供輔助判斷，"
        "不代表網站絕對安全。"
    )


# =========================================================
# 詐騙特徵字典
# =========================================================

SCAM_FEATURES = {

    "投資詐騙": {

        "keywords": [
            "保證獲利",
            "保證賺錢",
            "穩賺不賠",
            "穩賺",
            "高報酬",
            "高收益",
            "零風險",
            "低風險高報酬",
            "內線消息",
            "內幕消息",
            "老師帶單",
            "老師報牌",
            "代操",
            "飆股",
            "明牌",
            "投資群組",
            "股票群組",
            "虛擬貨幣投資",
            "加密貨幣投資",
            "穩定獲利",
            "每天獲利",
            "快速翻倍",
            "本金翻倍",
        ],

        "patterns": [
            ("加入", "投資群"),
            ("加入", "股票群"),
            ("加LINE", "投資"),
            ("加賴", "投資"),
            ("入金", "獲利"),
            ("儲值", "投資"),
            ("匯款", "代操"),
        ],
    },

    "金融與帳戶詐騙": {

        "keywords": [
            "解除分期",
            "重複扣款",
            "誤設分期",
            "取消分期",
            "監管帳戶",
            "安全帳戶",
            "帳戶凍結",
            "帳戶異常",
            "帳戶解凍",
            "提款機操作",
            "ATM操作",
            "網路銀行操作",
            "提供驗證碼",
            "簡訊驗證碼",
            "OTP驗證碼",
            "銀行密碼",
            "網銀密碼",
            "信用卡卡號",
            "信用卡背面末三碼",
        ],

        "patterns": [
            ("ATM", "解除"),
            ("ATM", "取消"),
            ("轉帳", "驗證"),
            ("匯款", "帳戶"),
            ("銀行", "驗證碼"),
            ("客服", "分期"),
        ],
    },

    "假冒政府或司法機關": {

        "keywords": [
            "涉及洗錢",
            "涉嫌洗錢",
            "涉及刑案",
            "涉嫌刑案",
            "偵查不公開",
            "法院傳票",
            "檢察官指示",
            "檢警辦案",
            "地檢署通知",
            "警察局通知",
            "健保卡遭冒用",
            "身分遭冒用",
            "配合調查",
            "不得告知家人",
        ],

        "patterns": [
            ("警察", "匯款"),
            ("檢察官", "匯款"),
            ("法院", "轉帳"),
            ("地檢署", "帳戶"),
            ("涉案", "監管"),
        ],
    },

    "中獎與獎金詐騙": {

        "keywords": [
            "恭喜中獎",
            "幸運得主",
            "領取獎金",
            "領取獎品",
            "中獎通知",
            "兌獎期限",
            "領獎手續費",
            "稅金後領獎",
            "先繳稅金",
            "保證金後領取",
        ],

        "patterns": [
            ("中獎", "手續費"),
            ("中獎", "匯款"),
            ("領獎", "稅金"),
            ("獎金", "帳戶"),
        ],
    },

    "網路購物詐騙": {

        "keywords": [
            "私下交易",
            "跳過平台",
            "離開平台交易",
            "先匯款後出貨",
            "匯款後出貨",
            "保留商品請先付款",
            "訂金保留",
            "客服要求操作ATM",
            "賣場認證",
            "買家認證",
            "金流認證",
        ],

        "patterns": [
            ("匯款", "出貨"),
            ("訂金", "保留"),
            ("客服", "ATM"),
            ("賣場", "驗證"),
            ("買家", "驗證"),
        ],
    },

    "愛情與交友詐騙": {

        "keywords": [
            "跨國軍官",
            "海外軍官",
            "戰地軍官",
            "聯合國醫生",
            "海外醫生",
            "海外工程師",
            "包裹卡海關",
            "見面需要費用",
            "急需借錢",
            "幫忙匯款",
            "代收包裹",
            "愛你但需要錢",
            "未見面先借錢",
        ],

        "patterns": [
            ("交友", "借錢"),
            ("愛你", "匯款"),
            ("見面", "機票"),
            ("包裹", "手續費"),
            ("海關", "費用"),
            ("軍官", "匯款"),
        ],
    },

    "釣魚連結與個資詐騙": {

        "keywords": [
            "點擊連結驗證",
            "立即點擊連結",
            "登入驗證帳戶",
            "帳號即將停用",
            "帳號即將封鎖",
            "更新個人資料",
            "填寫銀行資料",
            "提供身分證字號",
            "提供信用卡資料",
            "重新認證帳號",
        ],

        "patterns": [
            ("點擊", "驗證"),
            ("連結", "登入"),
            ("帳號", "停用"),
            ("帳戶", "認證"),
            ("填寫", "信用卡"),
        ],
    },

    "急迫與施壓話術": {

        "keywords": [
            "立即處理",
            "限時處理",
            "最後通知",
            "逾期失效",
            "今天截止",
            "馬上匯款",
            "立刻轉帳",
            "不得告知他人",
            "不要告訴家人",
            "保持通話",
            "不要掛電話",
        ],

        "patterns": [
            ("立即", "匯款"),
            ("立刻", "轉帳"),
            ("限時", "付款"),
            ("最後", "機會"),
            ("不要", "報警"),
        ],
    },
}


# =========================================================
# 文字詐騙可解釋功能
# =========================================================

def normalize_for_matching(text: str) -> str:

    normalized = text.lower()

    normalized = re.sub(
        r"\s+",
        "",
        normalized,
    )

    normalized = normalized.replace(
        "line群組",
        "line群",
    )

    normalized = normalized.replace(
        "line群组",
        "line群",
    )

    normalized = normalized.replace(
        "加入line",
        "加line",
    )

    normalized = normalized.replace(
        "加line好友",
        "加line",
    )

    return normalized


def analyze_scam_features(
    text: str,
) -> list[dict]:

    normalized_text = normalize_for_matching(
        text
    )

    findings = []

    for category, rules in SCAM_FEATURES.items():

        matched_items = []

        for keyword in rules.get(
            "keywords",
            [],
        ):

            if (
                normalize_for_matching(keyword)
                in normalized_text
            ):
                matched_items.append(
                    keyword
                )

        for pattern_words in rules.get(
            "patterns",
            [],
        ):

            normalized_words = [
                normalize_for_matching(word)
                for word in pattern_words
            ]

            if all(
                word in normalized_text
                for word in normalized_words
            ):

                matched_items.append(
                    "＋".join(
                        pattern_words
                    )
                )

        unique_items = list(
            dict.fromkeys(
                matched_items
            )
        )

        if unique_items:

            findings.append({
                "category": category,
                "matches": unique_items,
            })

    findings.sort(
        key=lambda item:
            len(item["matches"]),
        reverse=True,
    )

    return findings


def build_explanation_text(
    user_text: str,
    max_categories: int = 3,
    max_matches_per_category: int = 4,
) -> str:

    findings = analyze_scam_features(
        user_text
    )

    if not findings:

        return (
            "未偵測到明確的規則型詐騙關鍵字。\n"
            "本次結果主要來自模型對整段文字語意的判斷，"
            "建議仍透過官方來源查證。"
        )

    lines = []

    for finding in findings[
        :max_categories
    ]:

        matches = finding["matches"][
            :max_matches_per_category
        ]

        match_text = "、".join(
            f"「{item}」"
            for item in matches
        )

        lines.append(
            f"• {finding['category']}："
            f"偵測到 {match_text}"
        )

    lines.append(
        "\n以上是系統偵測到的常見風險特徵，"
        "用來輔助說明判斷結果。"
    )

    return "\n".join(lines)


# =========================================================
# Hugging Face
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

    logger.info(
        "正在呼叫 Hugging Face /predict"
    )

    result = client.predict(
        text,
        api_name="/predict",
    )

    logger.info(
        "Hugging Face 預測成功"
    )

    return str(result)


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
            f"Hugging Face 呼叫失敗："
            f"{error}"
        ) from error

    finally:

        executor.shutdown(
            wait=False,
            cancel_futures=True,
        )


# =========================================================
# 一般文字長度檢查
# =========================================================

def validate_user_text(
    text: str,
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
            "請貼上較完整的新聞或訊息內容，"
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
# BERT 模型結果解析
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
        return float(
            match.group(1)
        )

    except (
        TypeError,
        ValueError,
    ):
        return None


def format_prediction_result(
    result: str,
    user_text: str,
) -> str:

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
        r"模型信心度[：:]\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*%",
        result,
    )

    scam_probability = extract_percentage(
        r"詐騙機率[：:]\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*%",
        result,
    )

    real_probability = extract_percentage(
        r"真實機率[：:]\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*%",
        result,
    )

    if label_match:

        label = label_match.group(1)

    elif "scam" in result_lower:

        label = "詐騙"

    elif (
        "real" in result_lower
        or "true" in result_lower
    ):

        label = "真實"

    else:

        return (
            result
            + "\n\nℹ️ 此結果由 AI 模型產生，"
            "僅供參考，請搭配其他可靠來源查證。"
        )

    if confidence is None:

        if (
            "詐騙" in label
            and scam_probability is not None
        ):

            confidence = scam_probability

        elif (
            "真實" in label
            and real_probability is not None
        ):

            confidence = real_probability

        else:

            confidence = 0.0

    probability_lines = []

    if scam_probability is not None:

        probability_lines.append(
            f"🚨 詐騙機率："
            f"{scam_probability:.2f}%"
        )

    if real_probability is not None:

        probability_lines.append(
            f"✅ 真實機率："
            f"{real_probability:.2f}%"
        )

    probability_text = "\n".join(
        probability_lines
    )

    # =====================================================
    # 詐騙
    # =====================================================

    if "詐騙" in label:

        if confidence >= 95:

            title = (
                "🔴【模型高度傾向詐騙】"
            )

            description = (
                "模型對詐騙類別的判斷信心很高，"
                "請先停止交易並進一步查證。"
            )

        elif confidence >= 80:

            title = (
                "🟠【高風險訊息】"
            )

            description = (
                "此內容具有較高的詐騙風險，"
                "請提高警覺並避免立即操作。"
            )

        elif confidence >= 60:

            title = (
                "🟡【疑似詐騙】"
            )

            description = (
                "模型偏向詐騙，"
                "但判斷仍有不確定性，"
                "建議透過官方來源查證。"
            )

        else:

            title = (
                "⚪【判斷信心不足】"
            )

            description = (
                "模型目前無法明確判斷，"
                "請勿只依賴此結果做出決定。"
            )

        sections = [
            title,
            (
                f"📊 模型信心度："
                f"{confidence:.2f}%"
            ),
        ]

        if probability_text:
            sections.append(
                probability_text
            )

        explanation_text = (
            build_explanation_text(
                user_text
            )
        )

        sections.extend([
            (
                "🔎 風險說明：\n"
                + description
            ),
            (
                "🧩 判斷依據：\n"
                + explanation_text
            ),
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

        return "\n\n".join(
            sections
        )

    # =====================================================
    # 真實
    # =====================================================

    if "真實" in label:

        if confidence >= 95:

            title = (
                "🟢【模型高度傾向真實】"
            )

            description = (
                "模型高度傾向此內容為真實，"
                "但仍無法保證內容完全正確。"
            )

        elif confidence >= 80:

            title = (
                "🟢【較可能是真實內容】"
            )

            description = (
                "模型傾向此內容為真實，"
                "仍建議確認消息來源。"
            )

        elif confidence >= 60:

            title = (
                "🟡【可能是真實內容】"
            )

            description = (
                "模型初步判斷為真實，"
                "但信心度有限，"
                "建議再次查證。"
            )

        else:

            title = (
                "⚪【判斷信心不足】"
            )

            description = (
                "模型目前無法確定內容是否真實，"
                "請參考其他可靠來源。"
            )

        sections = [
            title,
            (
                f"📊 模型信心度："
                f"{confidence:.2f}%"
            ),
        ]

        if probability_text:
            sections.append(
                probability_text
            )

        sections.extend([
            (
                "🔎 判斷說明：\n"
                + description
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
        ])

        return "\n\n".join(
            sections
        )

    return (
        result
        + "\n\nℹ️ 此結果由 AI 模型產生，"
        "僅供參考，請搭配其他可靠來源查證。"
    )


# =========================================================
# 首頁
# =========================================================

@app.route(
    "/",
    methods=["GET"],
)
def home():

    return {
        "status": "ok",
        "service": (
            "LINE scam + phishing detection bot"
        ),
        "text_model": "BERT",
        "url_model": "Random Forest",
        "url_features": 25,
        "hf_space": HF_SPACE_URL,
        "min_text_length": MIN_TEXT_LENGTH,
        "max_prediction_seconds":
            MAX_PREDICTION_SECONDS,
        "explainable_rules": True,
    }, 200


@app.route(
    "/health",
    methods=["GET"],
)
def health():

    return "OK", 200


# =========================================================
# LINE Webhook
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
# LINE 文字訊息
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
        "收到使用者文字，字數：%s",
        len(user_text),
    )

    # =====================================================
    # A. URL → Random Forest
    # =====================================================

    if is_url(user_text):

        logger.info(
            "偵測到 URL，進入釣魚網址模型"
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
    # B. 一般文字 → BERT
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

                model_result = (
                    predict_with_huggingface(
                        user_text
                    )
                )

                reply_text = (
                    format_prediction_result(
                        model_result,
                        user_text,
                    )
                )

                elapsed = (
                    time.time()
                    - start_time
                )

                logger.info(
                    "文字預測完成，耗時 %.2f 秒",
                    elapsed,
                )

            except Exception as error:

                elapsed = (
                    time.time()
                    - start_time
                )

                logger.exception(
                    "模型預測失敗，"
                    "耗時 %.2f 秒：%s",
                    elapsed,
                    error,
                )

                reply_text = (
                    "⚠️ 模型目前正在啟動、"
                    "更新或忙碌中。\n\n"
                    "本次等待時間過長，"
                    "為避免 LINE 回覆逾時，"
                    "已先停止等待。\n\n"
                    "請稍候約 30 秒後，"
                    "再重新傳送一次。"
                )

    # =====================================================
    # LINE 回覆
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

        error_text = str(error)

        if (
            "invalid reply token"
            in error_text.lower()
        ):

            logger.warning(
                "LINE reply token 已失效或已使用，"
                "本次無法回覆。"
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
# 本機測試
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
