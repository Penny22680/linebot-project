"""Conservative OCR phrase clues for LINE image scam-screening beta.

Rules find literal, short risk phrases in OCR *text*. They do not explain the
internal decision of DeiT / the text classifier and are not proof of fraud.
No full OCR transcripts, URLs, phone numbers or private images are logged.
"""
from __future__ import annotations

import re

# (category, phrase expression, why it deserves independent verification)
_RULES = (
    ('guarantee', r'保證(?:獲利|收益|賺錢)|絕不虧損|穩賺不賠|零風險獲利',
     '宣稱投資保證獲利或完全不會虧損，屬於常見的高風險招攬話術。'),
    ('returns', r'(?:每(?:日|週|月))?固定收益|高額回報|穩定獲利|高報酬低風險',
     '強調固定或高額報酬，應查證投資來源、風險與合法性。'),
    ('guide', r'專人帶單|老師帶單|帶單操作|跟單操作|老師全程協助操作',
     '以帶單、跟單或代操作吸引投資，應先查核對方身分。'),
    ('urgency', r'立即驗證|儘速驗證|限時驗證|否則將暫停|帳戶(?:將)?(?:被)?(?:凍結|停用)',
     '要求緊急處理帳戶問題，可能利用壓力誘導點擊連結。'),
    ('expiry', r'(?:24|48|二十四|四十八)小時(?:後)?(?:失效|到期)|連結即將失效',
     '強調時間限制，應避免在未確認官方來源前匆忙操作。'),
    ('login', r'(?:提供|輸入|填寫|回傳|傳送)(?:您的|個人)?(?:帳戶密碼|登入密碼|簡訊驗證碼|OTP|信用卡資訊)',
     '涉及帳密、驗證碼或信用卡資訊，請先確認官方管道。'),
    ('transfer', r'先(?:匯款|付款|繳費|支付)|繳(?:交)?保證金|首次參與只需',
     '要求先付費或繳款，請查明用途及對方身分。'),
    ('prize', r'(?:中獎|領獎)(?:需|要|須|必須|請|先)?(?:支付|繳交|付款|匯款|繳納)(?:手續費|稅金|保證金)?',
     '領獎卻要求先支付費用，可能涉及詐騙。'),
    ('bank', r'帳戶存在異常交易|異地登入',
     '以帳戶異常為由要求操作時，應另由官方管道查核。'),
)


def find_text_clues(text: str, max_items: int = 3) -> list[dict[str, str]]:
    """Return at most 3 short matched phrases, without exposing full OCR text.

    No confidence or model-feature attribution is implied. Normalization makes
    ordinary OCR spacing less brittle while limiting quoted text to known rules.
    """
    if not isinstance(text, str) or max_items <= 0:
        return []
    normalized = re.sub(r'\s+', '', text)
    normalized = normalized.replace('，', ',').replace('％', '%')
    results = []
    for category, pattern, explanation in _RULES:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
            # A simple negation safeguard: don't present "不保證獲利" as a promise.
            prefix = normalized[max(0, match.start() - 3):match.start()]
            if re.search(r'(?:不|無|非|未|勿|別|拒絕)$', prefix):
                continue
            results.append({'quote': match.group(0)[:28], 'explanation': explanation})
            break  # at most one clue per category
        if len(results) >= min(max_items, 3):
            break
    return results
