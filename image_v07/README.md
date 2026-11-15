# 圖片詐騙辨識 V0.7｜完全獨立測試部署版

**本專案與 `Penny22680/linebot-project` 完全分開。** 不需修改原來的 `app.py`、`render.yaml`、BERT 文字模型、LightGBM 網址模型、Render 服務或 LINE Webhook。

## 本版包含什麼？

- DeiT Tiny 視覺模型（需要從你的 Mac 複製 `model.safetensors` 和 `config.json`）
- 已封裝的 V0.7 OCR TF-IDF + Logistic Regression 分類器（models/ocr_text_model.joblib）
- 固定 V0.7 的分數後期融合：影像 0.20 + OCR 0.80；二元門檻 0.45；三段式 0.35/0.65 + 模型不一致則人工確認
- Flask API、LINE Webhook 簽章驗證、LINE 圖片下載、單人聊天室快速 reply + 背景 push
- Dockerfile + Tesseract 繁中/英文語言套件
- `preflight.py` 模型檔與部署前敏感檔檢查

**不包含：** 你的 DeiT 真實權重、任何 LINE 憑證、私人圖片資料或對真實 LINE/Render 的已完成部署。分類分數未經校準，並非詐騙機率。模型還沒有通過全新來源獨立測試，只適合封閉測試。

## 第 1 步｜下載 ZIP 後在 Mac 安裝檔案

```bash
cd ~/Downloads
unzip -o linebot_image_v07_independent.zip
cd linebot_image_v07_independent

# 複製之前 Mac 上真正訓練好的 DeiT 權重
cp ~/Downloads/scam_v06_research/deit_macos_output/best_model/config.json models/deit_best_model/
cp ~/Downloads/scam_v06_research/deit_macos_output/best_model/model.safetensors models/deit_best_model/

python3 preflight.py
python3 -m unittest discover -s tests -p 'test_smoke.py' -v
```

`preflight.py` 通過才進行 GitHub 上傳；如果失敗，把錯誤訊息當作下一步排錯依據。不要上傳任何 `.env`、LINE 憑證、OCR 快取或私人測試圖片。

## 第 2 步｜先在 Mac 跑一次模型

```bash
cd ~/Downloads/linebot_image_v07_independent
source ../scam_v06_research/.venv/bin/activate
python3 -m pip install 'Flask>=3.1,<4' 'requests>=2.32,<3' 'gunicorn>=23,<25'
# 用你本機實際存在、獲授權的測試圖片路徑替換
python3 local_test.py --image '/你的圖片路徑/sample.jpg'
```

若成功，會得到模型分數、OCR 狀態和三段式分類；不需要 LINE 憑證。

## 第 3 步｜建立全新的私有 GitHub 儲存庫

到 <https://github.com/new> 建立名稱 `linebot-image-v07` 的 **Private** repository，建立時先不要加入 README。**不得使用原本的 `linebot-project`，也不要在舊 repo 建分支。**

在專案根目錄執行以下指令（把 URL 換成 GitHub 實際建立的私人儲存庫網址）：

```bash
python3 preflight.py
git init
git branch -M main
git add .
git status --short  # 人工確認：僅程式碼、joblib OCR 模型、DeiT 權重；沒有 .env/私人圖片/資料集
git commit -m 'Isolated V0.7 image beta'
git remote add origin https://github.com/<你的帳號>/linebot-image-v07.git
git push -u origin main
```

**確認 GitHub 網頁標示 Private，才可上傳模型權重。** 模型約 21 MB，私有儲存庫適合短期封測；仍應只讓有權者存取。

## 第 4 步｜建立新的 Render 服務，不動原服務

<https://dashboard.render.com/> → **New → Web Service** → 選擇新私人 repo `linebot-image-v07` → **Language: Docker** → `Dockerfile` 路徑 `./Dockerfile`。

- 名稱：`linebot-image-v07-test`（新名稱，不能與舊 Render service 重複）
- Health Check Path：`/health`（只表示 HTTP 存活與檔案檢查；**另外一定要手動檢查 `/ready`**）
- Instance：此模型要同時載入 Torch、Transformers、OCR，512 MB 計畫可能記憶體不足；請按 Render 顯示的規格與價格選擇，**沒有你的同意不要建立付費實例**
- 第一次不設 LINE 憑證也能測 `/health`、`/ready`，但 `/callback` 會返回 503
- 成功後：`https://<新服務網址>/ready` 必須回 `"status": "ready"`

**沒有 Render 實測前，不保證 Docker 映像已成功建立，也不保證所選方案資源足夠。** 日誌如有 OOM，應增加 RAM 或改用較小架構。

## 第 5 步｜另開測試 LINE Channel，別動原 Bot

到 <https://developers.line.biz/console/> 為**另一個測試用 LINE 官方帳號／Messaging API Channel**設置 Webhook：

```text
https://<新 Render 服務網址>/callback
```

只把**新測試 Channel**的 `LINE_CHANNEL_SECRET`、`LINE_CHANNEL_ACCESS_TOKEN` 放到新 Render 服務的 Environment。不要輸入原本 Bot 的憑證；也不要改舊 Channel 的 Webhook。LINE 每個 Channel 只能設定一個 Webhook URL，舊 LINE Bot 不受影響。

使用測試帳號一對一傳送圖片，觀察先收到「已收到圖片」，接著收到風險評估訊息。後續若要合併同一 LINE Bot，必須另行規劃，不在本次範圍內。

## 風險、限制與清理

- 機率未經校準，不能保證圖片安全；先限制測試使用者，不得宣稱完成正式對外發布。
- 本機 ThreadPoolExecutor 工作佇列沒有持久化；Render 重啟可能遺失已排隊的工作。收到圖後用 push 傳回結果，可能計入 LINE 推播配額。
- 不要把圖片或 OCR 原文寫入日誌；正式對外使用前要完成個資聲明、源自不同網站的新資料驗證與服務可靠性測試。
- 訓練與舊有文字、網址服務不相容的套件版本，放在**獨立 Docker 環境**，不會安裝到原本 Render。

參考：<https://render.com/docs/docker>，<https://developers.line.biz/en/docs/messaging-api/receiving-messages/>。
