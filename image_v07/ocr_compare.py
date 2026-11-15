"""Compare old/new OCR on a LOCAL authorized picture; never prints recognized text."""
from pathlib import Path
from PIL import Image, ImageOps
from ocr_pipeline import extract_text
import argparse
import pytesseract

parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True, type=Path)
parser.add_argument('--show-text', action='store_true', help='僅在本機顯示 OCR 原文，勿貼到聊天室或日誌')
args = parser.parse_args()
with Image.open(args.image) as source:
    image = ImageOps.exif_transpose(source).convert('RGB')
    old = image.copy()
    old.thumbnail((1100, 1100), Image.Resampling.BILINEAR)
    try:
        old_text = pytesseract.image_to_string(old, lang='chi_tra+eng', config='--psm 6', timeout=6)
        print('舊版 OCR 字元數：', sum(ch.isalnum() for ch in old_text))
        if args.show_text:
            print('--- 舊版擷取文字（僅本機） ---\n' + old_text)
    except Exception as e:
        print('舊版 OCR 失敗：', type(e).__name__)
    new = extract_text(image)
    print('新版狀態：', new.status)
    print('新版 OCR 字元數：', new.char_count)
    print('新版模式：', new.method)
    if args.show_text:
        print('--- 新版擷取文字（僅本機；低品質文字會隱藏） ---\n' + new.text)
    print('新版 OCR 內部信心值：', None if new.confidence is None else round(new.confidence, 1))
    print('注意：文字數量及 Tesseract 信心值均不等於辨識正確率，請使用圖片原始文字人工核對。')
