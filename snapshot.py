"""انبار نتیجه های از پیش محاسبه شده.

چرا؟ Render رایگان فقط ۰٫۱ CPU دارد و اجرای زنده ایجنت ۱۴۰ ثانیه طول
می کشد. به جای اینکه کاربر منتظر بماند، گیت هاب اکشنز هر ۱۵ دقیقه محاسبه
را انجام می دهد و نتیجه را به این ماژول «پوش» می کند.

چرا پوش و نه کامیت در ریپو؟ چون هر کامیت باعث ری دیپلوی Render می شود
(روزی ۹۶ بار) و ریپو هم خصوصی است پس raw.githubusercontent بدون توکن
کار نمی کند. پوش مستقیم هر دو مشکل را حل می کند و به علاوه سایت را
بیدار نگه می دارد.

هیچ داده حدسی اینجا ساخته نمی شود — فقط همان چیزی که موتور واقعی
محاسبه کرده، ذخیره و پس داده می شود.
"""
import json
import os
import threading
import time
from typing import Any, Dict, Optional

# محل ذخیره روی دیسک. روی Render موقتی است (با ری استارت پاک می شود)
# ولی چون اکشنز هر ۱۵ دقیقه دوباره پوش می کند، خودش را ترمیم می کند.
_PATH = os.environ.get("SNAPSHOT_PATH", "/tmp/dow_snapshot.json")

# کلید مشترک. اگر تنظیم نشده باشد، پوش کاملاً غیرفعال است.
_KEY = (os.environ.get("SNAPSHOT_KEY") or "").strip()

# حداکثر عمر قابل قبول برای یک نتیجه ذخیره شده (ثانیه).
MAX_AGE = int(os.environ.get("SNAPSHOT_MAX_AGE", "2700"))      # ۴۵ دقیقه

_LOCK = threading.Lock()
_MEM: Dict[str, Any] = {}


def enabled() -> bool:
    """آیا کلید تنظیم شده و پوش مجاز است؟"""
    return bool(_KEY)


def check_key(given: Optional[str]) -> bool:
    """مقایسه امن کلید (مقاوم در برابر حمله زمان سنجی)."""
    if not _KEY or not given:
        return False
    import hmac
    return hmac.compare_digest(_KEY, given.strip())


def _read_disk() -> Dict[str, Any]:
    try:
        with open(_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_disk(blob: Dict[str, Any]) -> None:
    try:
        tmp = _PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(blob, f, ensure_ascii=False)
        os.replace(tmp, _PATH)         # جایگزینی اتمیک
    except Exception:
        pass                            # دیسک پر/فقط خواندنی → فقط حافظه


def _all() -> Dict[str, Any]:
    """همه ورودی ها؛ اول حافظه، اگر خالی بود از دیسک بخوان."""
    global _MEM
    with _LOCK:
        if _MEM:
            return _MEM
        _MEM = _read_disk()
        return _MEM


def save(key: str, payload: Any, built_at: Optional[float] = None) -> Dict:
    """ذخیره یک نتیجه محاسبه شده زیر کلید دلخواه."""
    now = time.time()
    entry = {
        "payload": payload,
        "built_at": float(built_at or now),
        "saved_at": now,
    }
    with _LOCK:
        # نکته: باید کپی گرفت. اگر blob همان شیء _MEM باشد،
        # _MEM.clear() هر دو را پاک می کند و همه چیز از دست می رود.
        blob = dict(_MEM) if _MEM else _read_disk()
        blob[key] = entry
        _MEM.clear()
        _MEM.update(blob)
        out = dict(_MEM)
    _write_disk(out)
    return {"ok": True, "key": key, "age": 0.0}


def get(key: str, max_age: Optional[int] = None) -> Optional[Dict]:
    """اگر نتیجه تازه باشد برگردان، وگرنه None."""
    lim = MAX_AGE if max_age is None else max_age
    e = (_all() or {}).get(key)
    if not isinstance(e, dict):
        return None
    age = time.time() - float(e.get("built_at") or 0)
    if lim > 0 and age > lim:
        return None
    return {"payload": e.get("payload"), "age": age,
            "built_at": e.get("built_at")}


def age_of(key: str) -> Optional[float]:
    """عمر یک ورودی به ثانیه، بدون توجه به تازگی."""
    e = (_all() or {}).get(key)
    if not isinstance(e, dict):
        return None
    return time.time() - float(e.get("built_at") or 0)


def status() -> Dict:
    """وضعیت انبار — برای نمایش به کاربر و عیب یابی."""
    blob = _all() or {}
    now = time.time()
    items = []
    for k, e in sorted(blob.items()):
        if not isinstance(e, dict):
            continue
        age = now - float(e.get("built_at") or 0)
        items.append({
            "key": k,
            "age_sec": round(age, 1),
            "age_fa": _age_fa(age),
            "fresh": age <= MAX_AGE,
        })
    return {
        "ok": True,
        "enabled": enabled(),
        "max_age_sec": MAX_AGE,
        "count": len(items),
        "items": items,
        "note": ("کلید تنظیم نشده — پوش غیرفعال است"
                 if not enabled() else "آماده دریافت"),
    }


def _age_fa(sec: float) -> str:
    if sec < 90:
        return f"{sec:.0f} ثانیه پیش"
    if sec < 5400:
        return f"{sec / 60:.0f} دقیقه پیش"
    return f"{sec / 3600:.1f} ساعت پیش"
