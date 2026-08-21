# LINE 詐騙文字與釣魚網址辨識 Bot V3

## 功能

- 一般文字：呼叫 Hugging Face BERT V3，回傳分類、機率與 Integrated Gradients。
- 網址：使用 LightGBM V3，輸入為 18 個結構特徵與 1000 維字元 TF-IDF。
- LINE Messaging API：收到訊息後自動判斷是網址或一般文字。

## 必要檔案

- `app.py`
- `predict_url_v3.py`
- `url_features.py`
- `url_explainer.py`
- `phishing_model_v3.pkl`
- `tfidf_vectorizer_v3.pkl`
- `requirements.txt`
- `runtime.txt`

## Render 環境變數

- `LINE_CHANNEL_ACCESS_TOKEN`
- `LINE_CHANNEL_SECRET`
- `HF_SPACE_URL`（預設已設定）

## Render Start Command

```bash
gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 300
```

## LINE Webhook

部署完成後，將 LINE Developers Webhook URL 設成：

```text
https://你的-render-網址.onrender.com/callback
```

並啟用 Webhook。
