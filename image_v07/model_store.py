
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "models"

FILES = {
    "deit_best_model/config.json": 100,
    "deit_best_model/model.safetensors": 1000000,
    "ocr_text_model.joblib": 1000,
}

def ensure_model_files():
    missing = [
        name for name, size in FILES.items()
        if not (ROOT / name).is_file()
        or (ROOT / name).stat().st_size < size
    ]

    if not missing:
        return

    repo = os.getenv("HF_PRIVATE_MODEL_REPO", "").strip()
    token = os.getenv("HF_TOKEN", "").strip()

    if repo != "Penny0922/linebot-image-v07-private" or not token:
        raise RuntimeError("私人模型庫或 HF_TOKEN 尚未正確設定")

    from huggingface_hub import hf_hub_download

    ROOT.mkdir(parents=True, exist_ok=True)

    for name in missing:
        hf_hub_download(
            repo_id=repo,
            filename=name,
            repo_type="model",
            token=token,
            local_dir=str(ROOT),
        )

    for name, size in FILES.items():
        path = ROOT / name
        if not path.is_file() or path.stat().st_size < size:
            raise RuntimeError("模型下載不完整：" + name)
