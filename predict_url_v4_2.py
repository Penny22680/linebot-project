
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import tldextract

from scipy.sparse import csr_matrix, hstack

from url_features_v4 import (
    STRUCTURE_COLUMNS,
    extract_universal_features,
)


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "phishing_model_v4_1.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer_v4_1.pkl"
METADATA_PATH = BASE_DIR / "model_metadata_v4_1.pkl"

BEST_THRESHOLD = 0.53874125094324


TRUSTED_HOMEPAGES = {
    "google.com",
    "youtube.com",
    "github.com",
    "openai.com",
    "microsoft.com",
    "apple.com",
    "cloudflare.com",
    "wikipedia.org",
    "amazon.com",
    "facebook.com",
    "instagram.com",
    "linkedin.com",
}


SHORTENER_DOMAINS = {
    "bit.ly",
    "goo.gl",
    "tinyurl.com",
    "is.gd",
    "cli.gs",
    "t.co",
    "ow.ly",
    "adf.ly",
    "rb.gy",
    "reurl.cc",
    "cutt.ly",
    "rebrand.ly",
}


_TLD_EXTRACTOR = tldextract.TLDExtract(
    suffix_list_urls=None
)


model = joblib.load(MODEL_PATH)
tfidf = joblib.load(VECTORIZER_PATH)
metadata = joblib.load(METADATA_PATH)


def normalize_url_for_model(url: str) -> str:
    if not isinstance(url, str):
        url = ""

    url = url.strip()

    if not url:
        return "http://unknown.invalid/"

    url = (
        url
        .replace("[.]", ".")
        .replace("\\.", ".")
        .replace("hxxps://", "https://")
        .replace("hxxp://", "http://")
    )

    if not url.lower().startswith(
        ("http://", "https://")
    ):
        url = "http://" + url

    return url


def get_registered_domain(hostname: str) -> str:
    hostname = hostname.lower().strip(".")

    extracted = _TLD_EXTRACTOR(hostname)

    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}"

    return hostname


def is_trusted_homepage(normalized_url: str) -> bool:
    try:
        parsed = urlparse(normalized_url)

        hostname = (
            parsed.hostname
            or ""
        ).lower()

        registered_domain = get_registered_domain(
            hostname
        )

        path = parsed.path or "/"

        return (
            registered_domain in TRUSTED_HOMEPAGES
            and path == "/"
            and parsed.query == ""
            and parsed.fragment == ""
        )

    except Exception:
        return False


def build_reasons(
    feature_dict: dict,
    trusted_homepage: bool,
) -> list[str]:
    reasons = []

    if trusted_homepage:
        reasons.append("符合已確認的官方首頁")

    if feature_dict["has_ip"] == 1:
        reasons.append("網址使用 IP 位址")

    if feature_dict["is_shortened"] == 1:
        reasons.append(
            "網址使用短網址服務，需檢查實際目的地"
        )

    if feature_dict["has_sensitive_keyword"] == 1:
        reasons.append(
            "網址包含登入、驗證或付款等敏感字詞"
        )

    if feature_dict["suspicious_tld"] == 1:
        reasons.append(
            "網址使用較高風險的頂級網域"
        )

    if feature_dict["has_at_symbol"] == 1:
        reasons.append("網址包含 @ 符號")

    if feature_dict["double_slash_redirect"] == 1:
        reasons.append(
            "網址路徑中出現額外雙斜線"
        )

    if feature_dict["subdomain_count"] >= 3:
        reasons.append(
            "網址包含較多層子網域"
        )

    if feature_dict["is_https"] == 1:
        reasons.append("網址使用 HTTPS")
    else:
        reasons.append("網址未使用 HTTPS")

    return reasons


def predict_url(url: str) -> dict:
    original_url = str(url).strip()

    if not original_url:
        return {
            "url": "",
            "normalized_url": "",
            "prediction": None,
            "result": "輸入網址為空",
            "model_phishing_probability": None,
            "final_risk_score": None,
            "threshold": BEST_THRESHOLD,
            "trusted_homepage": False,
            "decision_source": "輸入檢查",
            "reasons": ["請輸入有效網址"],
        }

    normalized_url = normalize_url_for_model(
        original_url
    )

    tfidf_vector = tfidf.transform(
        [normalized_url]
    )

    feature_dict = extract_universal_features(
        normalized_url
    )

    structure_vector = csr_matrix(
        np.asarray(
            [[
                feature_dict[column]
                for column in STRUCTURE_COLUMNS
            ]],
            dtype=np.float32,
        )
    )

    final_vector = hstack(
        [
            tfidf_vector,
            structure_vector,
        ],
        format="csr",
        dtype=np.float32,
    )

    model_probability = float(
        model.predict_proba(
            final_vector
        )[0, 1]
    )

    trusted_homepage = is_trusted_homepage(
        normalized_url
    )

    is_shortened = (
        feature_dict["is_shortened"] == 1
    )

    if trusted_homepage:
        prediction = 0
        result = "正常網址"
        decision_source = "可信官方首頁規則"

        final_risk_score = min(
            model_probability * 100,
            5.0,
        )

    elif is_shortened:
        prediction = None
        result = "短網址，需進一步檢查"
        decision_source = "短網址安全規則"
        final_risk_score = (
            model_probability * 100
        )

    else:
        prediction = int(
            model_probability
            >= BEST_THRESHOLD
        )

        result = (
            "釣魚網址"
            if prediction == 1
            else "正常網址"
        )

        decision_source = (
            "LightGBM v4.1 模型"
        )

        final_risk_score = (
            model_probability * 100
        )

    return {
        "url": original_url,
        "normalized_url": normalized_url,
        "prediction": prediction,
        "result": result,
        "model_phishing_probability": round(
            model_probability,
            6,
        ),
        "final_risk_score": round(
            final_risk_score,
            2,
        ),
        "threshold": round(
            BEST_THRESHOLD,
            6,
        ),
        "trusted_homepage": trusted_homepage,
        "decision_source": decision_source,
        "reasons": build_reasons(
            feature_dict,
            trusted_homepage,
        ),
        "structure_features": feature_dict,
    }
