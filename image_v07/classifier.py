"""V0.7 DeiT + OCR inference, matching original run_v07.py research behavior.

Scam label index = 1. The scores are NOT calibrated fraud probabilities.
OCR-only fallback: if OCR missing, use visual score but send to manual review.
"""
from __future__ import annotations

import io
import logging
import os
import threading
from pathlib import Path

from ocr_pipeline import extract_text

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / 'models' / 'deit_best_model'
OCR_MODEL_PATH = BASE / 'models' / 'ocr_text_model.joblib'
OCR_WEIGHT = 0.8  # frozen on original 80-image validation set
BINARY_THRESHOLD = 0.45
TRIAGE_LOW = 0.35
TRIAGE_HIGH = 0.65
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000


def triage(visual: float, ocr: float | None, fused: float) -> str:
    """Exact V0.7 three-way triage (not the binary research classifier)."""
    if ocr is None:
        return '需要人工確認'
    if (visual >= .5) != (ocr >= .5):
        return '需要人工確認'
    if fused >= TRIAGE_HIGH and visual >= .5 and ocr >= .5:
        return '疑似詐騙'
    if fused <= TRIAGE_LOW and visual < .5 and ocr < .5:
        return '未偵測到明顯線索（不代表安全）'
    return '需要人工確認'


def combine_scores(visual: float, ocr: float | None) -> float:
    return visual if ocr is None else (1.0 - OCR_WEIGHT) * visual + OCR_WEIGHT * ocr


def image_from_bytes(blob: bytes):
    from PIL import Image, ImageOps
    if not blob or len(blob) > MAX_IMAGE_BYTES:
        raise ValueError('圖片為空或超過 10 MB')
    # Verify file type rather than trusting its name or MIME header.
    with Image.open(io.BytesIO(blob)) as im:
        if im.format not in ('JPEG', 'PNG', 'WEBP'):
            raise ValueError('僅支援 JPG、PNG、WEBP 圖片')
        if im.width * im.height > MAX_IMAGE_PIXELS:
            raise ValueError('圖片像素超過安全限制')
        rgb = ImageOps.exif_transpose(im).convert('RGB')
    return rgb


class FusionEngine:
    def __init__(self, device: str | None = None):
        import joblib
        import torch
        from transformers import AutoModelForImageClassification
        import pytesseract

        from model_store import ensure_model_files
        ensure_model_files()
        if not (MODEL_PATH / 'model.safetensors').is_file() or not (MODEL_PATH / 'config.json').is_file():
            raise FileNotFoundError(f'缺少 Mac 上訓練好的 DeiT 權重：{MODEL_PATH}')
        if not OCR_MODEL_PATH.is_file():
            raise FileNotFoundError('缺少可信任的 OCR 分類器模型')
        lang = set(pytesseract.get_languages(config=''))
        if not {'chi_tra', 'eng'}.issubset(lang):
            raise RuntimeError('尚未安裝 Tesseract chi_tra + eng 語言套件')

        torch.set_num_threads(max(1, min(int(os.getenv('TORCH_NUM_THREADS', '2')), 4)))
        choice = device or ('mps' if torch.backends.mps.is_available() else
                            ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.device = torch.device(choice)
        self.model = AutoModelForImageClassification.from_pretrained(str(MODEL_PATH))
        if self.model.config.num_labels != 2:
            raise RuntimeError('DeiT 模型並非二元分類器')
        label_map = self.model.config.id2label
        if label_map.get(1) not in ('scam', 'LABEL_1'):
            raise RuntimeError(f'分類索引需要人工確認：{label_map!r}')
        self.model = self.model.to(self.device).eval()
        # Load only trusted, bundled joblib. Never load a user-uploaded model.
        self.ocr_model = joblib.load(OCR_MODEL_PATH)
        self._lock = threading.Lock()

    def evaluate(self, blob: bytes) -> dict:
        import numpy as np
        import pytesseract
        import torch
        from PIL import Image, ImageOps

        im = image_from_bytes(blob)
        deit_im = ImageOps.pad(im, (224, 224), method=Image.Resampling.BILINEAR,
                               color=(255, 255, 255))
        arr = np.asarray(deit_im, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        mean = torch.tensor([.485, .456, .406])[:, None, None]
        std = torch.tensor([.229, .224, .225])[:, None, None]
        pixel_values = ((tensor - mean) / std)[None].to(self.device)
        with self._lock, torch.inference_mode():
            logits = self.model(pixel_values=pixel_values).logits.float()
            visual = float(torch.softmax(logits, dim=-1)[0, 1].cpu().item())

        ocr_status = 'unavailable'
        ocr_method = 'none'
        ocr_char_count = 0
        ocr_confidence = None
        ocr_score = None
        try:
            extraction = extract_text(im)
            ocr_status = extraction.status
            ocr_method = extraction.method
            ocr_char_count = extraction.char_count
            ocr_confidence = extraction.confidence
            if extraction.status == 'ok':
                ocr_score = float(self.ocr_model.predict_proba([extraction.text])[0, 1])
        except Exception as exc:
            # Fail closed: never treat OCR errors as evidence that an image is safe.
            logging.getLogger('scam-v07-ocr').warning(
                'OCR extraction/scoring failed: %s', type(exc).__name__
            )
            ocr_status = 'unavailable'
            ocr_score = None
        fused = combine_scores(visual, ocr_score)
        final = triage(visual, ocr_score, fused)
        return {
            'version': 'V0.8.1 DeiT + OCR (research beta)',
            'label': final,
            'binary_label': '疑似詐騙' if fused >= BINARY_THRESHOLD else '未達二元門檻',
            'visual_model_score': round(visual, 4),
            'ocr_text_model_score': round(ocr_score, 4) if ocr_score is not None else None,
            'fusion_score': round(fused, 4),
            'ocr_status': ocr_status,
            'ocr_method': ocr_method,
            'ocr_char_count': ocr_char_count,
            'ocr_confidence': round(ocr_confidence, 1) if ocr_confidence is not None else None,
            'warning': '模型分數未校準，不是詐騙機率；無法證明圖片安全。'
        }
