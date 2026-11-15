"""Non-network preflight. Doesn't print tokens, image contents or model data."""
from __future__ import annotations
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
model = root / 'models' / 'deit_best_model'
checks = [
    (model/'config.json', 100),
    (model/'model.safetensors', 1_000_000),
    (root/'models'/'ocr_text_model.joblib', 1_000),
    (root/'Dockerfile', 10),
    (root/'app.py', 100),
]
fail = False
for path, min_bytes in checks:
    ok = path.is_file() and path.stat().st_size >= min_bytes
    print(('OK' if ok else 'MISSING') + ' ' + str(path.relative_to(root)))
    fail |= not ok
for name in ('.env', '.env.local', '.env.production', 'private_ocr_cache.json'):
    if (root/name).exists():
        print('PRIVATE FILE PRESENT; remove before Git upload: ' + name)
        fail = True
# This is not a comprehensive repository security audit. Check `git status` too.
if fail:
    print('結果：未通過；請先補齊模型或排除敏感檔案。')
    sys.exit(1)
print('結果：通過靜態檔案檢查；仍需本機推論與 Render 實測。')
