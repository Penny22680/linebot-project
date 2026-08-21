import re
from urllib.parse import urlparse

from url_features import extract_universal_features

SUSPICIOUS_KEYWORDS = {
    "login", "signin", "verify", "verification", "account", "password",
    "secure", "security", "update", "confirm", "bank", "wallet",
    "refund", "tax", "bonus", "prize", "payment", "authenticate",
    "authentication", "gov", "pay", "paypal", "line", "support",
}


def explain_url(url: str) -> list[str]:
    """以模型使用的結構特徵產生人類可讀的輔助說明。"""
    url = str(url).strip()
    features = extract_universal_features(url)
    reasons: list[str] = []

    if features["has_ip"]:
        reasons.append("網址直接使用 IP 位址，而不是一般網域名稱。")

    if features["is_https"] == 0:
        reasons.append("網址未使用 HTTPS 加密連線。")

    if features["is_shortened"]:
        reasons.append("網址使用縮網址服務，實際目的地較難直接確認。")

    if features["has_at_symbol"]:
        reasons.append("網址包含 @ 符號，可能用來混淆真正的目的網域。")

    if features["double_slash_redirect"]:
        reasons.append("網址在協定後又出現 //，可能具有可疑重新導向結構。")

    if features["subdomain_count"] >= 3:
        reasons.append("網址包含較多層子網域，可能用來模仿官方網站。")

    if features["count_hyphen"] >= 3:
        reasons.append("網址包含多個連字號，網域命名結構較可疑。")

    if features["url_length"] >= 75:
        reasons.append("網址長度偏長，較不容易由使用者直接辨識。")

    if features["count_dots"] >= 4:
        reasons.append("網址中的句點數量偏多。")

    if features["count_digits"] >= 8:
        reasons.append("網址包含大量數字。")

    if features["digits_ratio"] >= 0.20:
        reasons.append("網址中的數字比例偏高。")

    if features["url_entropy"] >= 4.2:
        reasons.append("網址字元分布較隨機，熵值偏高。")

    if features["count_question"] >= 1:
        reasons.append("網址包含查詢參數，請確認參數內容與來源。")

    if features["count_equal"] >= 2:
        reasons.append("網址包含多個參數指派符號。")

    parsed = urlparse(url if re.match(r"^https?://", url, re.I) else f"http://{url}")
    tokens = {
        token for token in re.split(r"[^a-z0-9]+", f"{parsed.netloc}{parsed.path}{parsed.query}".lower())
        if token
    }
    matched_keywords = sorted(tokens.intersection(SUSPICIOUS_KEYWORDS))
    if matched_keywords:
        reasons.append("網址包含高風險情境字詞：" + "、".join(matched_keywords[:5]) + "。")

    return reasons
