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

# ── انبار پشتیبان روی ریپو (اصلاح ۲۰۲۶-۰۹-۳۰) ─────────────────────
#
# مشکلی که کشف شد: /tmp روی Render رایگان موقتی است. سرویس بعد از
# ۱۵ دقیقه بی کاری می خوابد و با بیدار شدن، انبار خالی است. فرض
# قبلی این بود که «اکشنز هر ۱۵ دقیقه ترمیمش می کند» ولی بررسی
# تاریخچه اجراها نشان داد گیت هاب کران را throttle می کند و واقعا
# هر ۳ تا ۴ ساعت اجرا می شود.
#
# نتیجه: /api/snapshot همیشه count: 0 بود و هر درخواست ایجنت یک
# محاسبه زنده ۱۴۰ تا ۳۰۰ ثانیه ای می شد که روی ۰٫۱ CPU کل سرویس
# را قفل می کرد.
#
# راه حل: همان الگوی دفترچه. اکشنز نتیجه را در ریپو کامیت می کند و
# سایت وقتی حافظه و دیسکش خالی است، یک بار از raw.githubusercontent
# می خواند. اندازه گیری شد: ۰٫۱۵ ثانیه برای ۳۴ کیلوبایت فشرده.
# این با هر ری استارت خودش را ترمیم می کند.
#
# اگر ریپو پرایوت شود این آدرس ۴۰۴ می دهد و بی صدا به محاسبه زنده
# برمی گردیم — یعنی رفتار بدتر از قبل نمی شود.
# آدرس فایل پشتیبان. اگر دستی تنظیم نشود، خودش از متغیرهایی که
# Render به طور خودکار می سازد ساخته می شود — یعنی کاربر لازم نیست
# هیچ چیزی در پنل Render تنظیم کند.
#
#     RENDER_GIT_REPO_SLUG = نام کاربری/نام ریپو
#     RENDER_GIT_BRANCH    = main
#
# خارج از Render (مثلا اجرای محلی) این متغیرها وجود ندارند و
# _REMOTE خالی می ماند، یعنی رفتار دقیقا مثل قبل است.
def _default_remote() -> str:
    slug = (os.environ.get("RENDER_GIT_REPO_SLUG") or "").strip().strip("/")
    if not slug or "/" not in slug:
        return ""
    branch = (os.environ.get("RENDER_GIT_BRANCH") or "main").strip() or "main"
    return ("https://raw.githubusercontent.com/%s/%s/snapshot_cache.json"
            % (slug, branch))


_REMOTE = (os.environ.get("SNAPSHOT_REMOTE") or "").strip() or _default_remote()
_REMOTE_TTL = 300.0          # حداکثر هر ۵ دقیقه یک بار از گیت هاب بخوان
_remote_at = 0.0

# اگر تازه ترین ورودی از این کهنه تر بود، از ریپو بپرس شاید
# اکشنز نسخه جدیدی کامیت کرده باشد. (۱۵ دقیقه)
_REFRESH_AFTER = float(os.environ.get("SNAPSHOT_REFRESH_AFTER", "900"))

# کلید مشترک. اگر تنظیم نشده باشد، پوش کاملاً غیرفعال است.
_KEY = (os.environ.get("SNAPSHOT_KEY") or "").strip()

# حداکثر عمر قابل قبول برای یک نتیجه ذخیره شده (ثانیه).
# ⚠ اصلاح ۲۰۲۶-۰۹-۳۰: پیش فرض از ۴۵ دقیقه به ۴ ساعت رفت.
#
# چرا: کرون گیت هاب روی ریپوهای کم فعالیت شدیدا throttle می شود.
# کران روی */15 تنظیم بود ولی بررسی تاریخچه اجراها نشان داد واقعا
# هر ۳ تا ۴ ساعت اجرا می شود. با سقف ۴۵ دقیقه، انبار بیشتر وقت ها
# منقضی بود (/api/snapshot → count: 0) و سایت مجبور می شد محاسبه
# زنده ۱۴۰ تا ۲۰۰ ثانیه ای انجام دهد که مرورگر قطعش می کرد و کاربر
# عدد کهنه می دید بدون اینکه بداند کهنه است.
#
# نتیجه کهنه با برچسب صریح، بی نهایت بهتر از تایم اوت است.
MAX_AGE = int(os.environ.get("SNAPSHOT_MAX_AGE", "14400"))     # ۴ ساعت

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


def _read_remote() -> Dict[str, Any]:
    """انبار پشتیبان را از ریپو بخوان. بی صدا شکست می خورد."""
    if not _REMOTE:
        return {}
    import urllib.request
    try:
        req = urllib.request.Request(
            _REMOTE, headers={"User-Agent": "dow-dashboard",
                              "Accept-Encoding": "gzip"})
        with urllib.request.urlopen(req, timeout=12) as r:
            raw = r.read()
            if r.headers.get("Content-Encoding") == "gzip":
                import gzip as _gz
                raw = _gz.decompress(raw)
        blob = json.loads(raw.decode("utf-8"))
        return blob if isinstance(blob, dict) else {}
    except Exception:
        return {}


def _newest(blob: Dict[str, Any]) -> float:
    """تازه ترین built_at بین همه ورودی ها (۰ اگر هیچ)."""
    best = 0.0
    for e in (blob or {}).values():
        if not isinstance(e, dict):
            continue
        try:
            b = float(e.get("built_at") or 0)
        except (TypeError, ValueError):
            b = 0.0
        if b > best:
            best = b
    return best


def _merge(dst: Dict[str, Any], src: Dict[str, Any]) -> bool:
    """ورودی های تازه تر src را روی dst بنشان. True اگر چیزی عوض شد.

    کلید به کلید مقایسه می کنیم، چون ممکن است یک کلید روی سایت
    تازه تر باشد (پوش مستقیم اکشنز) و کلید دیگر روی ریپو.
    """
    changed = False
    for k, e in (src or {}).items():
        if not isinstance(e, dict):
            continue
        try:
            nb = float(e.get("built_at") or 0)
        except (TypeError, ValueError):
            nb = 0.0
        old = dst.get(k)
        ob = -1.0
        if isinstance(old, dict):
            try:
                ob = float(old.get("built_at") or 0)
            except (TypeError, ValueError):
                ob = 0.0
        if nb > ob:
            dst[k] = e
            changed = True
    return changed


def _pull_remote() -> bool:
    """یک بار از ریپو بخوان و ادغام کن. True اگر داده تازه تری آمد."""
    blob = _read_remote()
    if not blob:
        return False
    with _LOCK:
        changed = _merge(_MEM, blob)
        out = dict(_MEM)
    if changed:
        _write_disk(out)
    return changed


def _refresh_async() -> None:
    """تازه سازی در پس زمینه — درخواست وب منتظر شبکه نمی ماند."""
    t = threading.Thread(target=_pull_remote, daemon=True,
                         name="snapshot-refresh")
    t.start()


def _all() -> Dict[str, Any]:
    """همه ورودی ها. ترتیب: حافظه → دیسک → ریپو.

    ⚠ اصلاح ۲۰۲۶-۱۰-۰۱ — باگ «شاخه ریپو هیچ وقت اجرا نمی شد».

    نسخه قبلی اگر دیسک چیزی داشت همان جا return می کرد، پس
    _read_remote عملا کد مرده بود. اما snapshot_cache.json همراه
    دیپلوی روی دیسک می آید و کامیت های انبار برچسب [skip render]
    دارند، یعنی نسخه دیسک فقط موقع دیپلوی واقعی عوض می شود.
    نتیجه: داده روی لحظه آخرین دیپلوی یخ می زد، از MAX_AGE رد
    می شد، get() مقدار None می داد و هر درخواست یک محاسبه زنده
    چند دقیقه ای راه می انداخت که با تک ورکر کل سایت را قفل می کرد.

    حالا: اگر تازه ترین ورودی از _REFRESH_AFTER کهنه تر باشد سراغ
    ریپو می رویم — در پس زمینه اگر داده ای (هرچند کهنه) داریم، و
    فقط در حالت دست خالی منتظر شبکه می مانیم.
    """
    global _MEM, _remote_at
    with _LOCK:
        if not _MEM:
            _MEM = _read_disk()
        now = time.time()
        have = bool(_MEM)
        stale = _newest(_MEM) < (now - _REFRESH_AFTER)
        if have and not stale:
            return dict(_MEM)
        due = bool(_REMOTE) and (now - _remote_at) >= _REMOTE_TTL
        if not due:
            return dict(_MEM)
        _remote_at = now
        cur = dict(_MEM)
    if have:
        _refresh_async()           # کهنه ولی موجود ⇒ کاربر منتظر نماند
        return cur
    # دست خالی ⇒ چاره ای جز انتظار نیست (فقط اولین درخواست بعد از بوت)
    _pull_remote()
    with _LOCK:
        return dict(_MEM)


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


def get_any(key: str) -> Optional[Dict]:
    """ورودی را بدون توجه به سن برگردان، با برچسب کهنگی.

    ⚠ اصلاح ۲۰۲۶-۱۰-۰۱. روی Render رایگان (۰.۱ هسته) یک محاسبه
    زنده agent.decide بیش از ۵ دقیقه طول می کشد و چون فقط یک
    ورکر داریم، تمام سایت در آن مدت بی پاسخ می شود. پس وقتی
    انبار از MAX_AGE رد شده، به جای محاسبه زنده همان عدد کهنه
    را می دهیم و سنش را صریح اعلام می کنیم — دقیقا همان چیزی
    که در توضیح MAX_AGE قول داده شده بود ولی پیاده نشده بود.
    """
    e = (_all() or {}).get(key)
    if not isinstance(e, dict):
        return None
    try:
        built = float(e.get("built_at") or 0)
    except (TypeError, ValueError):
        built = 0.0
    age = time.time() - built
    return {"payload": e.get("payload"), "age": age, "built_at": built,
            "stale": age > MAX_AGE, "age_fa": _age_fa(age)}


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
