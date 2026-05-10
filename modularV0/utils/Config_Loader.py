import json
from pathlib import Path
from typing import Dict, Any
from dataclasses import asdict
from modularV0.utils.Configurations import Configurations

CONFIG_FILE = Path("x.json")

def _is_empty_file(fp: Path) -> bool:
    """بررسی خالی بودن فایل."""
    try:
        return fp.stat().st_size == 0
    except FileNotFoundError:
        return True

def load_config(path:   str = CONFIG_FILE) -> Configurations:
    """بارگذاری فایل JSON و تبدیل آن به شیء Configurations."""
    path = Path(path)

    # 1️⃣ اگر فایل وجود ندارد یا خالی است → مقادیر پیش‌فرض برگردان
    if not path.is_file() or _is_empty_file(path):
        print(f"[Config] فایل '{path}' یافت نشد یا خالی بود. از مقادیر پیش‌فرض استفاده می‌شود.")
        return Configurations()

    # 2️⃣ فایل را بخوانیم؛ در صورت خطای JSON، هم‌چنان به مقادیر پیش‌فرض برگردیم
    try:
        with path.open("r", encoding="utf-8") as f:
            raw_data: Dict[str, Any] = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[Config] خطا در خواندن JSON: {e}. فایل خالی می‌شود و مقادیر پیش‌فرض برمی‌گردد.")
        # می‌توانیم فایل را پاک/نوشتن مجدد کنیم
        path.write_text("", encoding="utf-8")
        return Configurations()

    # 3️⃣ فیلدهای نامعتبر را حذف کنیم
    valid_keys = set(Configurations.__dataclass_fields__.keys())
    filtered = {k: v for k, v in raw_data.items() if k in valid_keys}

    return Configurations(**filtered)

def save_config(cfg: Configurations, path: str = CONFIG_FILE) -> None:
    """ذخیره‌سازی شیء Configurations به صورت JSON."""
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)