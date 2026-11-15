"""Closed beta LINE image risk triage service.

Signature verification is mandatory. One-to-one image messages only.
Reply immediately, analyze in bounded background executor, then push result.
The in-process queue is NOT durable across restarts; see README.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests
from flask import Flask, abort, jsonify, request

from classifier import FusionEngine, MODEL_PATH, OCR_MODEL_PATH, MAX_IMAGE_BYTES

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
log = logging.getLogger('scam-v07-linebot')
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_IMAGE_BYTES + 4096

LINE_REPLY_API = 'https://api.line.me/v2/bot/message/reply'
LINE_PUSH_API = 'https://api.line.me/v2/bot/message/push'
LINE_CONTENT_API = 'https://api-data.line.me/v2/bot/message/{}/content'

# This is a lightweight *beta* queue, not a durable job system.
workers = ThreadPoolExecutor(max_workers=1, thread_name_prefix='line-image')
queue_slots = threading.BoundedSemaphore(value=4)
_engine = None
_engine_lock = threading.Lock()
_seen: dict[str, float] = {}
_seen_lock = threading.Lock()


def get_engine():
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = FusionEngine()
    return _engine


def push_target(source: dict) -> str | None:
    # Avoid accidental public posting of an image report in groups.
    if source.get('type') != 'user':
        return None
    user_id = source.get('userId')
    return user_id if isinstance(user_id, str) and user_id.startswith('U') else None


def line_headers() -> dict:
    token = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN', '').strip()
    if not token:
        raise RuntimeError('LINE_CHANNEL_ACCESS_TOKEN 未設定')
    return {'Authorization': f'Bearer {token}'}


def reply(reply_token: str, text: str):
    resp = requests.post(
        LINE_REPLY_API, headers=line_headers(),
        json={'replyToken': reply_token, 'messages': [{'type': 'text', 'text': text}]},
        timeout=12,
    )
    resp.raise_for_status()


def push(to: str, text: str, event_id: str):
    headers = {**line_headers(), 'X-Line-Retry-Key': str(uuid.uuid5(uuid.NAMESPACE_URL, event_id))}
    resp = requests.post(
        LINE_PUSH_API, headers=headers,
        json={'to': to, 'messages': [{'type': 'text', 'text': text}]},
        timeout=15,
    )
    resp.raise_for_status()


def download_image(message_id: str) -> bytes:
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', str(message_id)):
        raise ValueError('Invalid message ID')
    with requests.get(LINE_CONTENT_API.format(message_id), headers=line_headers(),
                      stream=True, timeout=(5, 25)) as resp:
        resp.raise_for_status()
        size = 0
        chunks = []
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            if chunk:
                size += len(chunk)
                if size > MAX_IMAGE_BYTES:
                    raise ValueError('LINE image exceeds 10MB')
                chunks.append(chunk)
        if not chunks:
            raise ValueError('Empty LINE image')
        return b''.join(chunks)


def format_ocr_status(status: str) -> str:
    """Avoid implying OCR accuracy and keep user images out of the reply."""
    return {
        'ok': '已擷取可用文字（仍可能有錯字）',
        'empty': '未偵測到可用文字，請人工確認',
        'low_quality': '文字品質不足，請人工確認',
        'unavailable': '文字擷取暫時失敗，請稍後重試',
    }.get(status, '無法確認文字品質，請人工檢查')


def format_result(outcome: dict) -> str:
    """User-facing explanation: rule-matched OCR phrases, not ML causal attribution."""
    status = outcome.get('ocr_status', 'unavailable')
    label = str(outcome.get('label', '需要人工確認'))
    lines = [
        '🔍 圖片詐騙辨識｜V0.8.3 測試版',
        f'📌 判斷：{label}',
        f'📝 文字擷取：{format_ocr_status(status)}',
    ]
    if status == 'ok':
        clues = outcome.get('text_clues') or []
        if clues:
            lines.extend(['', '⚠️ OCR 辨識到的警示文字：'])
            for clue in clues[:3]:
                lines.append(f"•「{clue['quote']}」：{clue['explanation']}")
        else:
            lines.extend(['', '🔎 文字線索：未找到本版規則所涵蓋的特定警示語句；不代表安全。'])
    else:
        lines.extend(['', '🔎 文字線索：OCR 尚無可靠結果，無法提供具體文字線索。'])
    lines.extend([
        '',
        '📖 說明：最終結果來自影像模型與可用的文字模型；上方警示語句是獨立的規則比對，'
        '不是模型內部決策的因果解釋，不能只憑這些文字斷言是詐騙。',
        '⚠️ 此為研究模型，分數並非真實詐騙機率。請勿只憑結果匯款或提供密碼、個資；'
        '請透過官方網站或客服獨立查證。',
    ])
    return '\n'.join(lines)[:4900]


def analyze_and_push(message_id: str, to: str, event_id: str):
    try:
        blob = download_image(message_id)
        outcome = get_engine().evaluate(blob)
        message = format_result(outcome)
    except Exception:
        log.exception('Image analysis failed (no image/OCR content logged)')
        message = '圖片辨識失敗，請改用官方管道查證，或稍後再傳送圖片測試。'
    try:
        push(to, message, event_id)
    except Exception:
        log.exception('LINE push failed; beta queue does not retry after worker restart')
    finally:
        queue_slots.release()


def mark_event_once(event_id: str) -> bool:
    with _seen_lock:
        now = time.monotonic()
        for key, at in list(_seen.items()):
            if now - at > 3600:
                del _seen[key]
        if event_id in _seen:
            return False
        _seen[event_id] = now
        return True


@app.get('/health')
def health():
    files_present = ((MODEL_PATH / 'config.json').is_file() and
                     (MODEL_PATH / 'model.safetensors').is_file() and OCR_MODEL_PATH.is_file())
    return jsonify(status='ok', service='scam-v07-linebot-beta', model_files_present=files_present)


@app.get('/ready')
def ready():
    try:
        engine = get_engine()
        return jsonify(status='ready', model='V0.8.3 DeiT + OCR', device=str(engine.device))
    except Exception:
        log.exception('readiness failed')
        return jsonify(status='unavailable', message='模型或 OCR 尚未就緒'), 503


@app.post('/callback')
def callback():
    secret = os.getenv('LINE_CHANNEL_SECRET', '').strip()
    if not secret or not os.getenv('LINE_CHANNEL_ACCESS_TOKEN', '').strip():
        return jsonify(error='LINE env vars not configured'), 503

    raw = request.get_data(cache=True)
    sig = request.headers.get('X-Line-Signature', '')
    expected = base64.b64encode(hmac.new(secret.encode(), raw, hashlib.sha256).digest()).decode()
    if not hmac.compare_digest(sig, expected):
        abort(400, description='Invalid LINE signature')
    payload = request.get_json(silent=True) or {}
    events = payload.get('events', [])
    if not isinstance(events, list):
        return jsonify(error='invalid events'), 400

    for event in events:
        if not isinstance(event, dict) or event.get('type') != 'message':
            continue
        msg = event.get('message', {})
        if not isinstance(msg, dict) or msg.get('type') != 'image':
            # No unsolicited greeting or menu replies; image messages only.
            continue
        token = event.get('replyToken')
        if not isinstance(token, str) or not token:
            continue
        to = push_target(event.get('source', {}))
        if to is None:
            # Do not push results to a group or multi-person room by default.
            try:
                reply(token, '圖片辨識測試目前只支援與機器人的一對一聊天，以保護隱私。')
            except Exception:
                log.exception('LINE group guidance reply failed')
            continue
        message_id = str(msg.get('id', ''))
        event_id = str(event.get('webhookEventId') or message_id)
        if not message_id or not mark_event_once(event_id):
            continue
        if not queue_slots.acquire(blocking=False):
            try:
                reply(token, '目前有其他圖片正在辨識，請稍後重新傳送。')
            except Exception:
                log.exception('LINE busy reply failed')
            continue
        try:
            reply(token, '已收到圖片，正在辨識。完成後會另外傳送結果。')
        except Exception:
            log.exception('LINE acknowledgment failed; attempting push anyway')
        try:
            workers.submit(analyze_and_push, message_id, to, event_id)
        except Exception:
            queue_slots.release()
            log.exception('Unable to queue image job')
    return 'OK', 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', '5055')), debug=False)
