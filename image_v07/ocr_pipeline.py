"""Conservative Traditional-Chinese/English OCR for LINE scam-image research beta.

This module has no model inference or LINE integration. It deliberately never
logs or transmits recognized text. Quality measures are heuristics, not accuracy.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from PIL import Image, ImageFilter, ImageOps

log = logging.getLogger('scam-v07-ocr')

# Designed for the existing Render free instance; at most two Tesseract passes.
MAX_OCR_EDGE = 1100
MIN_OCR_CHARS = 6
MIN_OCR_CONFIDENCE = 30.0
RETRY_IF_CONFIDENCE_BELOW = 55.0
RETRY_IF_CHARS_BELOW = 24
TESSERACT_TIMEOUT_PER_PASS = 15


@dataclass(frozen=True)
class OCRResult:
    text: str
    status: str  # ok, empty, low_quality, unavailable
    method: str  # psm6, psm11, none
    char_count: int
    confidence: float | None


def _prepare(image: Image.Image) -> Image.Image:
    """Resize within a bounded area, then boost low-contrast small glyphs."""
    im = ImageOps.grayscale(image)
    width, height = im.size
    edge = max(width, height)
    # Enlarge small screenshot text modestly; never exceed MAX_OCR_EDGE.
    scale = min(2.0, MAX_OCR_EDGE / edge) if edge < MAX_OCR_EDGE else MAX_OCR_EDGE / edge
    if abs(scale - 1.0) > .05:
        im = im.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)
    im = ImageOps.autocontrast(im, cutoff=1)
    return im.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))


def _join_cjk_spaces(text: str) -> str:
    return re.sub(r'(?<=[\u3400-\u9fff])[ \t]+(?=[\u3400-\u9fff])', '', text)


def _extract_once(im: Image.Image, psm: int):
    import pytesseract

    data = pytesseract.image_to_data(
        im,
        lang='chi_tra+eng',
        config=f'--oem 1 --psm {psm} -c preserve_interword_spaces=1',
        output_type=pytesseract.Output.DICT,
        timeout=TESSERACT_TIMEOUT_PER_PASS,
    )
    # Reassemble lines instead of flattening punctuation and separate blocks.
    groups: dict[tuple[int, int, int, int], list[str]] = {}
    confs: list[float] = []
    for i, raw in enumerate(data.get('text', [])):
        word = str(raw).replace('\x00', '').strip()
        if not word:
            continue
        line_key = tuple(int(data[k][i]) for k in ('page_num', 'block_num', 'par_num', 'line_num'))
        groups.setdefault(line_key, []).append(word)
        try:
            confidence = float(data['conf'][i])
        except (TypeError, ValueError, KeyError, IndexError):
            continue
        if 0 <= confidence <= 100:
            confs.append(confidence)
    text = _join_cjk_spaces('\n'.join(' '.join(words) for words in groups.values()))
    char_count = sum(ch.isalnum() for ch in text)
    avg_conf = sum(confs) / len(confs) if confs else None
    return text, char_count, avg_conf


def _quality(pass_result) -> tuple[bool, float]:
    _, char_count, conf = pass_result
    is_usable = char_count >= MIN_OCR_CHARS and conf is not None and conf >= MIN_OCR_CONFIDENCE
    # Penalize very low-confidence garbage but don't just reward long noisy text.
    ranking = min(char_count, 180) / 3 + (conf or 0)
    if char_count < MIN_OCR_CHARS:
        ranking -= 60
    return is_usable, ranking


def extract_text(image: Image.Image) -> OCRResult:
    """OCR one or two passes; keep manual-review fallback when uncertain.

    `status=ok` is just a heuristic quality signal; it does NOT guarantee that
    the extracted characters match the picture.
    """
    im = _prepare(image)
    candidates = []
    errors = 0
    for psm in (6, 11):
        if psm == 11 and candidates:
            best = max(candidates, key=lambda x: _quality(x[1])[1])
            _text, count, confidence = best[1]
            # Avoid extra work when block-mode OCR has enough readable text.
            if count >= RETRY_IF_CHARS_BELOW and confidence is not None and confidence >= RETRY_IF_CONFIDENCE_BELOW:
                break
        try:
            result = _extract_once(im, psm)
            candidates.append((psm, result))
        except Exception as exc:
            # Do not print OCR text or uploaded image information.
            errors += 1
            if type(exc) is RuntimeError and str(exc) == 'Tesseract process timeout':
                log.warning('OCR attempt psm=%s timed out after %ss', psm, TESSERACT_TIMEOUT_PER_PASS)
                break
            log.warning('OCR attempt psm=%s failed: %s', psm, type(exc).__name__)

    if not candidates:
        return OCRResult('', 'unavailable', 'none', 0, None)

    usable = [(psm, r) for psm, r in candidates if _quality(r)[0]]
    selected_pool = usable or candidates
    psm, (text, count, confidence) = max(selected_pool, key=lambda x: _quality(x[1])[1])
    if usable:
        status = 'ok'
    elif count == 0:
        status = 'unavailable' if errors else 'empty'
    else:
        status = 'low_quality'
    log.info('OCR completed: status=%s method=psm%s chars=%s confidence=%s',
             status, psm, count, None if confidence is None else round(confidence))
    return OCRResult(text if status == 'ok' else '', status, f'psm{psm}', count, confidence)
