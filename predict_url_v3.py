from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, hstack

from url_explainer import explain_url
from url_features import STRUCTURE_COLUMNS, extract_universal_features


def predict_url_v3(url: str, model, tfidf_vectorizer) -> dict:
    """使用 18 個結構特徵 + 1000 維字元 TF-IDF 進行網址分類。"""
    url = str(url).strip()
    if not url:
        raise ValueError("網址不能是空白。")

    feature_dict = extract_universal_features(url)
    structure_values = np.array(
        [[float(feature_dict[column]) for column in STRUCTURE_COLUMNS]],
        dtype=np.float32,
    )
    structure_matrix = csr_matrix(structure_values)
    tfidf_part = tfidf_vectorizer.transform([url]).astype(np.float32)

    # 必須與訓練順序一致：18 個結構特徵在前，TF-IDF 在後。
    input_matrix = hstack([structure_matrix, tfidf_part], format="csr")

    if input_matrix.shape[1] != int(model.n_features_in_):
        raise ValueError(
            f"輸入特徵數為 {input_matrix.shape[1]}，模型需要 {model.n_features_in_}。"
        )

    prediction = int(model.predict(input_matrix)[0])
    probabilities = model.predict_proba(input_matrix)[0]

    return {
        "url": url,
        "prediction": prediction,
        "label": "釣魚網站" if prediction == 1 else "正常網站",
        "normal_probability": float(probabilities[0]),
        "phishing_probability": float(probabilities[1]),
        "features": feature_dict,
        "reasons": explain_url(url),
    }


def format_url_result(result: dict) -> str:
    prediction = int(result["prediction"])
    normal_percent = float(result["normal_probability"]) * 100
    phishing_percent = float(result["phishing_probability"]) * 100
    reasons = list(result.get("reasons", []))

    if prediction == 1:
        if phishing_percent >= 90:
            title = "🔴【高度疑似釣魚網站】"
        elif phishing_percent >= 70:
            title = "🟠【高風險網站】"
        else:
            title = "🟡【疑似釣魚網站】"
        confidence = phishing_percent
    else:
        title = "🟢【較可能為正常網站】"
        confidence = normal_percent

    lines = [
        "🌐 網址安全檢測",
        "",
        title,
        "",
        f"🔍 判斷結果：{result['label']}",
        f"📊 模型信心度：{confidence:.2f}%",
        "",
        f"🟢 正常機率：{normal_percent:.2f}%",
        f"🚨 釣魚機率：{phishing_percent:.2f}%",
        "",
        "🧠 AI 分析依據",
    ]

    if reasons:
        for index, reason in enumerate(reasons[:6], start=1):
            lines.append(f"{index}. {reason}")
    else:
        lines.append("未發現明顯可疑的網址結構特徵。")

    lines.extend([
        "",
        "📌 說明：以上原因來自模型使用的網址結構特徵與網址字串分析，"
        "用來輔助理解結果，不等同於單一因素直接決定分類。",
    ])

    if prediction == 1:
        lines.extend([
            "",
            "⚠️ 安全建議",
            "• 不要輸入帳號、密碼或信用卡資料",
            "• 不要提供簡訊驗證碼",
            "• 不要下載不明檔案",
            "• 建議自行開啟官方網站或官方 App",
        ])
    else:
        lines.extend([
            "",
            "⚠️ 即使模型判斷為正常，仍請確認網址拼字、網域名稱與來源。",
        ])

    lines.extend([
        "",
        "ℹ️ 此結果僅供輔助判斷，不代表網站絕對安全或一定為詐騙。",
    ])
    return "\n".join(lines)
