# -*- coding: utf-8 -*-
"""لایه «پیش بینی در مقابل واقعی» برای تقویم اقتصادی.

تقویم فعلی ما (real_data.economic_calendar) از فدرال رزرو و FRED
می آید و تاریخ ها ۱۰۰٪ واقعی اند — اما فقط می گوید «رویدادی هست».
نمی گوید عدد منتشر شده بهتر از انتظار بود یا بدتر.

biquote.io این سه عدد را می دهد: actual / forecast / previous.
بدون کلید، JSON تمیز، پنجره حدود ۷ روز.

⚠️ رتبه بندی اهمیت biquote قابل اعتماد نیست (تست شد: از ۳۹ رویداد
آمریکا فقط ۱ مورد high بود، آن هم موجودی نفت). بنابراین:
  • اهمیت را از جدول خودمان می خوانیم (IMPORTANCE)
  • اهمیت biquote فقط به عنوان fallback برای رویدادهای ناشناخته

هر تابع در خطا ok=False برمی گرداند و هرگز استثنا پرتاب نمی کند.
"""
from __future__ import annotations

import io
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Optional

CAL_URL = "https://biquote.io/api/calendar"

UA = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

# اهمیت بر اساس قضاوت خودمان — نه biquote
# کلیدواژه (lowercase) → (وزن ۱ تا ۴، نام فارسی)
IMPORTANCE = {
    "non farm payroll": (4, "اشتغال غیرکشاورزی"),
    "nonfarm payroll": (4, "اشتغال غیرکشاورزی"),
    "inflation rate": (4, "نرخ تورم"),
    "core inflation": (4, "تورم هسته"),
    "cpi": (4, "شاخص قیمت مصرف کننده"),
    "fed interest rate": (4, "نرخ بهره فدرال رزرو"),
    "fomc": (4, "نشست فدرال رزرو"),
    "fed press conference": (4, "کنفرانس خبری فدرال رزرو"),
    "gdp growth": (3, "رشد تولید ناخالص"),
    "unemployment rate": (3, "نرخ بیکاری"),
    "ppi": (3, "شاخص قیمت تولیدکننده"),
    "producer price": (3, "قیمت تولیدکننده"),
    "retail sales": (3, "خرده فروشی"),
    "pce": (3, "مخارج مصرف شخصی"),
    "ism manufacturing": (3, "شاخص تولید ISM"),
    "ism services": (3, "شاخص خدمات ISM"),
    "fed chair": (3, "سخنرانی رئیس فدرال رزرو"),
    "powell": (3, "سخنرانی پاول"),
    "initial jobless": (2, "مدعیان بیکاری"),
    "jobless claims": (2, "مدعیان بیکاری"),
    "durable goods": (2, "سفارش کالاهای بادوام"),
    "consumer confidence": (2, "اعتماد مصرف کننده"),
    "michigan": (2, "اعتماد مصرف کننده میشیگان"),
    "housing starts": (2, "شروع ساخت مسکن"),
    "industrial production m/m": (2, "تولید صنعتی ماهانه"),
    "industrial production y/y": (2, "تولید صنعتی سالانه"),
    "manufacturing production": (1, "تولید کارخانه ای"),
    "capacity utilization": (1, "نرخ بهره برداری ظرفیت"),
    "leading economic index": (2, "شاخص پیشرو اقتصادی"),
    "richmond fed": (1, "شاخص فد ریچموند"),
    "chicago fed": (1, "شاخص فد شیکاگو"),
    "fed governor": (2, "سخنرانی مقام فدرال رزرو"),
    "fed vice chair": (2, "سخنرانی نایب رئیس فدرال رزرو"),
    "fed president": (2, "سخنرانی رئیس فد منطقه ای"),
    "fed speech": (2, "سخنرانی فدرال رزرو"),
    "note auction": (1, "مزایده اوراق"),
    "bill auction": (1, "مزایده اوراق کوتاه مدت"),
    "rig count": (1, "شمار دکل های نفتی"),
    "crude oil stocks": (1, "موجودی نفت خام"),
    "eia": (1, "گزارش انرژی EIA"),
    "baker hughes": (1, "شمار دکل های نفتی"),
    "cftc copper": (1, "پوزیشن CFTC مس"),
    "cftc nasdaq": (1, "پوزیشن CFTC نزدک"),
    "cftc natural gas": (1, "پوزیشن CFTC گاز"),
    "cftc corn": (1, "پوزیشن CFTC ذرت"),
    "cftc wheat": (1, "پوزیشن CFTC گندم"),
    "cftc soybean": (1, "پوزیشن CFTC سویا"),
    "cftc aluminium": (1, "پوزیشن CFTC آلومینیوم"),
    "cftc gold": (1, "پوزیشن CFTC طلا"),
    "cftc silver": (1, "پوزیشن CFTC نقره"),
    "cftc s&p 500": (1, "پوزیشن CFTC اس اند پی"),
    "cftc crude oil": (1, "پوزیشن CFTC نفت"),
    "cftc": (1, "گزارش پوزیشن CFTC"),
}

IMPACT_FA = {4: "بسیار بالا", 3: "بالا", 2: "متوسط", 1: "کم"}

# آیا عدد بالاتر برای دلار/سهام خوب است؟
# +1 = بالاتر بهتر (رشد)، -1 = بالاتر بدتر (تورم/بیکاری)
DIRECTION = {
    "inflation": -1, "cpi": -1, "ppi": -1, "producer price": -1,
    "unemployment rate": -1, "jobless": -1, "claims": -1,
    "gdp": +1, "retail sales": +1, "payroll": +1,
    "industrial production": +1, "confidence": +1, "ism": +1,
    "durable goods": +1, "housing starts": +1,
}

_CACHE: Dict[str, tuple] = {}


def _cached(key: str, ttl: float, fn):
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    val = fn()
    _CACHE[key] = (now, val)
    return val


def _get(url: str, timeout: float = 25.0):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return json.loads(r.read().decode("utf-8", "ignore"))


def _num(v) -> Optional[float]:
    """«۱.۴٪» یا «230.3K» یا None → عدد."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "").replace("%", "")
    if not s or s in ("-", "--"):
        return None
    mult = 1.0
    if s and s[-1] in "KMB":
        mult = {"K": 1e3, "M": 1e6, "B": 1e9}[s[-1]]
        s = s[:-1]
    m = re.match(r"^-?\d*\.?\d+$", s)
    return float(s) * mult if m else None


def _classify(name: str) -> tuple:
    """نام انگلیسی → (وزن، نام فارسی، جهت)."""
    low = name.lower()
    weight, fa = 0, ""
    for kw, (w, f) in IMPORTANCE.items():
        if kw in low and w > weight:
            weight, fa = w, f
    direction = 0
    for kw, d in DIRECTION.items():
        if kw in low:
            direction = d
            break
    return weight, fa, direction


def _surprise(actual, forecast, previous, direction: int) -> Dict:
    """محاسبه «سورپرایز» — واقعی در مقابل انتظار."""
    a, f, p = _num(actual), _num(forecast), _num(previous)
    base = f if f is not None else p
    if a is None or base is None:
        return dict(ok=False, note="عدد منتشر نشده" if a is None else "پیش بینی موجود نیست")
    diff = a - base
    denom = abs(base) if abs(base) > 1e-9 else 1.0
    pct = diff / denom * 100.0
    if abs(diff) < 1e-9:
        tone, fa = 0, "دقیقا مطابق انتظار"
    else:
        beat = diff > 0
        # جهت: برای تورم/بیکاری، بالاتر = بد
        good = beat if direction >= 0 else (not beat)
        ref = "انتظار" if f is not None else "دوره قبل"
        if direction == 0:
            tone, fa = 0, ("بالاتر از %s" % ref if beat else "پایین تر از %s" % ref)
        else:
            tone = 1 if good else -1
            fa = ("بهتر از %s" % ref if good else "بدتر از %s" % ref)
    return dict(ok=True, actual=a, forecast=f, previous=p,
                diff=round(diff, 4), pct=round(pct, 2),
                tone=tone, tone_fa=fa,
                vs="پیش بینی" if f is not None else "دوره قبل")


def calendar_actuals(country: str = "US", _ttl: float = 900) -> Dict:
    """تقویم biquote با actual/forecast/previous و اهمیت بازتعریف شده."""
    def _fetch():
        try:
            raw = _get(CAL_URL)
        except urllib.error.HTTPError as e:
            return dict(ok=False, error="HTTP %s" % e.code, events=[])
        except Exception as e:
            return dict(ok=False, error=type(e).__name__, events=[])
        if not isinstance(raw, list):
            return dict(ok=False, error="شکل پاسخ غیرمنتظره", events=[])

        now = datetime.now(timezone.utc)
        out: List[Dict] = []
        for e in raw:
            if country and e.get("countryCode") != country:
                continue
            name = str(e.get("name") or "")
            weight, fa, direction = _classify(name)
            if weight == 0:
                bq = str(e.get("importance") or "").lower()
                weight = {"high": 3, "medium": 2, "low": 1}.get(bq, 1)
                fa = name
            try:
                when = datetime.fromisoformat(str(e["time"]).replace("Z", "+00:00"))
            except Exception:
                continue
            mins = (when - now).total_seconds() / 60.0
            sur = _surprise(e.get("actual"), e.get("forecast"),
                            e.get("previous"), direction)
            out.append(dict(
                name_en=name, name=fa or name,
                when_iso=when.isoformat(),
                when=when.strftime("%Y-%m-%d %H:%M"),
                minutes_until=round(mins), hours_until=round(mins / 60, 1),
                days_until=round(mins / 1440, 1),
                released=mins < 0 and sur.get("ok", False),
                weight=weight, impact=IMPACT_FA.get(weight, "کم"),
                direction=direction,
                actual=e.get("actual"), forecast=e.get("forecast"),
                previous=e.get("previous"),
                unit=e.get("unit") or "", surprise=sur,
                source="biquote.io", source_tier="real"))

        out.sort(key=lambda x: (x["when_iso"], -x["weight"]))
        seen, dedup = set(), []
        for x in out:
            sig = (x["name_en"], x["when_iso"])
            if sig in seen:
                continue
            seen.add(sig)
            dedup.append(x)
        out = dedup
        rel = [x for x in out if x["released"]]
        up = [x for x in out if x["minutes_until"] >= 0]
        return dict(
            ok=True, events=out, count=len(out),
            released=rel, upcoming=up,
            released_count=len(rel), upcoming_count=len(up),
            window="%s تا %s" % (out[0]["when"][:10], out[-1]["when"][:10]) if out else "",
            source="biquote.io", source_tier="real",
            note="پنجره حدود ۷ روزه — اهمیت بر اساس جدول خودمان بازتعریف شده")

    try:
        return _cached("cal_%s" % country, _ttl, _fetch)
    except Exception as e:
        return dict(ok=False, error=type(e).__name__, events=[])


def _match(fa_name: str, bq: Dict) -> bool:
    """آیا رویداد فارسی ما با رویداد biquote یکی است؟"""
    pairs = [
        ("مدعیان بیکاری", ("jobless", "claims")),
        ("میشیگان", ("michigan",)),
        ("اعتماد مصرف", ("consumer confidence", "michigan")),
        ("قیمت مصرف کننده", ("inflation rate", "cpi")),
        ("CPI", ("inflation rate", "cpi")),
        ("اشتغال", ("payroll", "unemployment")),
        ("NFP", ("payroll",)),
        ("تولید ناخالص", ("gdp",)),
        ("GDP", ("gdp",)),
        ("تولیدکننده", ("producer price", "ppi")),
        ("PPI", ("producer price", "ppi")),
        ("خرده فروشی", ("retail sales",)),
        ("PCE", ("pce", "personal spending", "personal income")),
        ("فدرال رزرو", ("fed ", "fomc")),
    ]
    low = bq["name_en"].lower()
    for fa_key, kws in pairs:
        if fa_key in fa_name and any(k in low for k in kws):
            return True
    return False


def enrich_calendar(cal: Dict, country: str = "US") -> Dict:
    """به تقویم FRED/فدرال رزرو ما، actual/forecast اضافه می کند.

    تقویم ما تاریخ های رسمی دارد (دقیق تر)، biquote اعداد دارد.
    این تابع دو را ترکیب می کند بدون اینکه تاریخ های ما تغییر کند.
    """
    if not isinstance(cal, dict):
        return dict(ok=False, error="ورودی نامعتبر")
    bq = calendar_actuals(country)
    out = dict(cal)
    out["actuals_ok"] = bool(bq.get("ok"))
    if not bq.get("ok"):
        out["actuals_note"] = "اعداد در دسترس نیست: %s" % bq.get("error", "?")
        return out

    evs = list(cal.get("events") or [])
    matched = 0
    for e in evs:
        for b in bq["events"]:
            if _match(e.get("name", ""), b) and abs(
                    (b["minutes_until"] - e.get("minutes_until", 0))) < 2880:
                e["actual"] = b["actual"]
                e["forecast"] = b["forecast"]
                e["previous"] = b["previous"]
                e["surprise"] = b["surprise"]
                e["actual_source"] = "biquote.io"
                matched += 1
                break
    out["events"] = evs
    out["actuals_matched"] = matched
    # رویدادهای مهمی که تقویم ما ندارد
    extra = [b for b in bq["events"] if b["weight"] >= 3]
    out["extra_high"] = extra
    out["extra_high_count"] = len(extra)
    out["actuals_note"] = ("%d رویداد با عدد واقعی تکمیل شد · %d رویداد مهم "
                           "اضافی از biquote" % (matched, len(extra)))
    return out


def surprise_score(country: str = "US", hours_back: float = 72.0) -> Dict:
    """جمع بندی سورپرایزهای اخیر → یک لحن کلی برای موتور تصمیم."""
    bq = calendar_actuals(country)
    if not bq.get("ok"):
        return dict(ok=False, error=bq.get("error"), tone=0.0)
    rec = [e for e in bq["released"]
           if -hours_back <= e["hours_until"] <= 0 and e["weight"] >= 2
           and e["surprise"].get("ok") and e["surprise"]["tone"] != 0
           and e["surprise"].get("forecast") is not None]
    if not rec:
        return dict(ok=True, tone=0.0, n=0,
                    note="داده مهمی در %d ساعت گذشته منتشر نشده" % int(hours_back),
                    source="biquote.io", source_tier="real", items=[])
    num = sum(e["surprise"]["tone"] * e["weight"] for e in rec)
    den = sum(e["weight"] for e in rec)
    tone = num / den if den else 0.0
    if tone >= 0.5:
        fa = "داده های اقتصادی بهتر از انتظار"
    elif tone <= -0.5:
        fa = "داده های اقتصادی بدتر از انتظار"
    else:
        fa = "داده های اقتصادی مختلط"
    items = [dict(name=e["name"], name_en=e["name_en"], impact=e["impact"],
                  actual=e["actual"], forecast=e["forecast"],
                  tone=e["surprise"]["tone"], tone_fa=e["surprise"]["tone_fa"],
                  hours_ago=abs(e["hours_until"]))
             for e in sorted(rec, key=lambda x: -x["weight"])[:8]]
    return dict(ok=True, tone=round(tone, 3), tone_fa=fa, n=len(rec),
                items=items, hours_back=hours_back,
                source="biquote.io", source_tier="real",
                note="میانگین وزنی سورپرایز %d داده مهم اخیر" % len(rec))


# ================================================================
# ForexFactory — فقط برای رویدادهای «بسیار مهم» که biquote ندارد
# ================================================================
# چرا لازم است: تست ۲۰۲۶-۰۹-۱۹ نشان داد biquote کل هفته FOMC را
# از دست داده بود — Federal Funds Rate، FOMC Statement و کنفرانس
# خبری، یعنی مهم ترین رویداد ماه.
#
# ⚠️ خطر جدی: این منبع بعد از ۲ درخواست کد ۴۲۹ می دهد و ۳ دقیقه
# بعد هم هنوز بلاک است. پس:
#   • کش روی دیسک با عمر ۶ ساعت (بین ری استارت ها هم می ماند)
#   • حداکثر ۴ درخواست در ۲۴ ساعت (شمارنده روی دیسک)
#   • پس از ۴۲۹، ۹۰ دقیقه کامل سکوت
#   • هرگز در مسیر داغ صدا زده نمی شود — فقط کش خوانده می شود
#
# این منبع actual ندارد (فقط forecast/previous) پس جایگزین
# biquote نیست؛ مکمل آن است برای رویدادهای High.

FF_URLS = {
    "thisweek": "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "nextweek": "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
}
FF_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache_ff")
FF_TTL = 6 * 3600.0          # عمر کش
FF_MAX_CALLS_DAY = 4         # سقف درخواست روزانه
FF_COOLDOWN_429 = 5400.0     # ۹۰ دقیقه سکوت بعد از ۴۲۹

# فقط این ها ارزش درخواست زدن دارند
FF_HIGH_ONLY = True

FF_NAMES = {
    "federal funds rate": "نرخ بهره فدرال رزرو",
    "fomc statement": "بیانیه فدرال رزرو",
    "fomc press conference": "کنفرانس خبری فدرال رزرو",
    "fomc economic projections": "پیش بینی های اقتصادی فدرال رزرو",
    "fomc meeting minutes": "صورتجلسه فدرال رزرو",
    "fomc member": "سخنرانی عضو فدرال رزرو",
    "cpi m/m": "تورم ماهانه",
    "cpi y/y": "تورم سالانه",
    "core cpi": "تورم هسته",
    "non-farm employment change": "اشتغال غیرکشاورزی",
    "unemployment rate": "نرخ بیکاری",
    "average hourly earnings": "دستمزد ساعتی",
    "ppi m/m": "قیمت تولیدکننده ماهانه",
    "core ppi": "قیمت تولیدکننده هسته",
    "retail sales": "خرده فروشی",
    "core retail sales": "خرده فروشی هسته",
    "advance gdp": "تولید ناخالص اولیه",
    "gdp q/q": "تولید ناخالص فصلی",
    "core pce": "مخارج مصرف شخصی هسته",
    "ism manufacturing pmi": "شاخص تولید ISM",
    "ism services pmi": "شاخص خدمات ISM",
    "jolts job openings": "فرصت های شغلی JOLTS",
    "unemployment claims": "مدعیان بیکاری",
    "consumer confidence": "اعتماد مصرف کننده",
    "prelim um consumer sentiment": "اعتماد مصرف کننده میشیگان",
    "fed chair": "سخنرانی رئیس فدرال رزرو",
    "treasury sec": "سخنرانی وزیر خزانه داری",
}


def _ff_state_path():
    return os.path.join(FF_CACHE_DIR, "state.json")


def _ff_load_state():
    try:
        with io.open(_ff_state_path(), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return dict(calls=[], blocked_until=0.0)


def _ff_save_state(st):
    try:
        os.makedirs(FF_CACHE_DIR, exist_ok=True)
        with io.open(_ff_state_path(), "w", encoding="utf-8") as f:
            json.dump(st, f)
    except Exception:
        pass


def _ff_cache_path(week):
    return os.path.join(FF_CACHE_DIR, "%s.json" % week)


def _ff_read_cache(week):
    try:
        p = _ff_cache_path(week)
        age = time.time() - os.path.getmtime(p)
        with io.open(p, encoding="utf-8") as f:
            return json.load(f), age
    except Exception:
        return None, 1e9


def _ff_write_cache(week, data):
    try:
        os.makedirs(FF_CACHE_DIR, exist_ok=True)
        with io.open(_ff_cache_path(week), "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def _ff_budget_ok(st):
    """آیا اجازه یک درخواست تازه داریم؟"""
    now = time.time()
    if now < st.get("blocked_until", 0):
        return False, "در دوره سکوت پس از ۴۲۹"
    recent = [t for t in st.get("calls", []) if now - t < 86400]
    st["calls"] = recent
    if len(recent) >= FF_MAX_CALLS_DAY:
        return False, "سقف %d درخواست روزانه پر شده" % FF_MAX_CALLS_DAY
    return True, ""


def _ff_name(title):
    low = title.lower().strip()
    for k, v in FF_NAMES.items():
        if k in low:
            return v
    return title


def forexfactory(week="thisweek", force=False):
    """رویدادهای High آمریکا از ForexFactory — کش ۶ ساعته اجباری."""
    if week not in FF_URLS:
        week = "thisweek"
    cached, age = _ff_read_cache(week)
    if cached is not None and age < FF_TTL and not force:
        out = dict(cached)
        out["from_cache"] = True
        out["cache_age_min"] = round(age / 60, 1)
        return out
    # کش منفی ۴۰۴ عمر کوتاه تری دارد — شاید فایل منتشر شده باشد
    if (cached is not None and not cached.get("ok")
            and cached.get("error") == "HTTP 404" and age < 3600 and not force):
        out = dict(cached)
        out["from_cache"] = True
        return out

    st = _ff_load_state()
    allowed, why = _ff_budget_ok(st)
    if not allowed:
        if cached is not None:
            out = dict(cached)
            out["from_cache"] = True
            out["stale"] = True
            out["cache_age_min"] = round(age / 60, 1)
            out["note"] = "کش قدیمی — %s" % why
            return out
        return dict(ok=False, error=why, events=[], source="ForexFactory")

    try:
        st["calls"] = list(st.get("calls", [])) + [time.time()]
        _ff_save_state(st)
        raw = _get(FF_URLS[week], timeout=20.0)
    except urllib.error.HTTPError as e:
        if e.code == 429:
            st["blocked_until"] = time.time() + FF_COOLDOWN_429
            _ff_save_state(st)
        elif e.code == 404:
            # فایل هفته آینده تا اواخر هفته ساخته نمی شود — کش منفی
            neg = dict(ok=False, error="HTTP 404", events=[], count=0,
                       week=week, source="ForexFactory",
                       note="فایل این هفته هنوز منتشر نشده")
            _ff_write_cache(week, neg)
            st["calls"] = [t for t in st.get("calls", [])][:-1]
            _ff_save_state(st)
            return neg
        if cached is not None:
            out = dict(cached)
            out["from_cache"] = True
            out["stale"] = True
            out["note"] = "HTTP %s — کش قدیمی استفاده شد" % e.code
            return out
        return dict(ok=False, error="HTTP %s" % e.code, events=[],
                    source="ForexFactory")
    except Exception as e:
        if cached is not None:
            out = dict(cached)
            out["from_cache"] = True
            out["stale"] = True
            return out
        return dict(ok=False, error=type(e).__name__, events=[],
                    source="ForexFactory")

    if not isinstance(raw, list):
        return dict(ok=False, error="شکل پاسخ غیرمنتظره", events=[],
                    source="ForexFactory")

    evs = []
    for e in raw:
        if e.get("country") != "USD":
            continue
        imp = str(e.get("impact") or "")
        if FF_HIGH_ONLY and imp != "High":
            continue
        title = str(e.get("title") or "")
        try:
            when = datetime.fromisoformat(str(e["date"]))
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            when = when.astimezone(timezone.utc)
        except Exception:
            continue
        evs.append(dict(
            name_en=title, name=_ff_name(title),
            when_iso=when.isoformat(),
            when=when.strftime("%Y-%m-%d %H:%M"),
            impact_raw=imp, weight=4, impact="بسیار بالا",
            forecast=e.get("forecast") or None,
            previous=e.get("previous") or None,
            actual=None,
            source="ForexFactory", source_tier="real"))

    evs.sort(key=lambda x: x["when_iso"])
    out = dict(ok=True, events=evs, count=len(evs), week=week,
               fetched_at=datetime.now(timezone.utc).isoformat(),
               source="ForexFactory", source_tier="real",
               note="فقط رویدادهای High آمریکا · کش ۶ ساعته")
    _ff_write_cache(week, out)
    out["from_cache"] = False
    out["cache_age_min"] = 0.0
    return out


def critical_events(hours_ahead=72.0):
    """رویدادهای بسیار مهم پیش رو — ترکیب ForexFactory و biquote.

    خروجی برای «وتوی خبری» استفاده می شود: اگر رویداد بسیار مهمی
    نزدیک باشد، بهتر است پوزیشن جدید باز نشود.
    """
    now = datetime.now(timezone.utc)
    items, srcs = [], []

    ff = forexfactory("thisweek")
    if ff.get("ok"):
        srcs.append("ForexFactory")
        items += ff["events"]
    nx = forexfactory("nextweek")
    if nx.get("ok"):
        items += nx["events"]
    nx_note = "" if nx.get("ok") else "تقویم هفته آینده هنوز منتشر نشده"

    bq = calendar_actuals()
    if bq.get("ok"):
        srcs.append("biquote.io")
        items += [e for e in bq["events"] if e.get("weight", 0) >= 3]

    seen, merged = set(), []
    for e in items:
        try:
            when = datetime.fromisoformat(e["when_iso"])
        except Exception:
            continue
        mins = (when - now).total_seconds() / 60.0
        if mins < -60 or mins > hours_ahead * 60:
            continue
        sig = (e["name"], e["when_iso"][:16])
        if sig in seen:
            continue
        seen.add(sig)
        d = dict(e)
        d["minutes_until"] = round(mins)
        d["hours_until"] = round(mins / 60, 1)
        merged.append(d)

    merged.sort(key=lambda x: x["minutes_until"])
    nxt = merged[0] if merged else None
    soon = [e for e in merged if 0 <= e["hours_until"] <= 4]
    return dict(
        ok=True, events=merged, count=len(merged), next_event=nxt,
        imminent=soon, imminent_count=len(soon),
        veto=bool(soon),
        veto_reason=("رویداد بسیار مهم در %s ساعت آینده: %s"
                     % (soon[0]["hours_until"], soon[0]["name"])) if soon else "",
        sources=srcs, source_tier="real",
        ff_cached=ff.get("from_cache"), ff_age_min=ff.get("cache_age_min"),
        nextweek_note=nx_note,
        note="رویدادهای بسیار مهم %d ساعت آینده" % int(hours_ahead))


if __name__ == "__main__":
    import sys
    c = sys.argv[1] if len(sys.argv) > 1 else "US"
    r = calendar_actuals(c)
    print("=" * 62)
    print("تقویم %s — ok=%s count=%s" % (c, r.get("ok"), r.get("count")))
    print("پنجره:", r.get("window"), "| منتشرشده:", r.get("released_count"),
          "| آینده:", r.get("upcoming_count"))
    print("=" * 62)
    for e in r.get("events", [])[:25]:
        s = e["surprise"]
        tag = ""
        if s.get("ok"):
            sym = {1: "✅", -1: "❌", 0: "➖"}[s["tone"]]
            tag = "%s %s (a=%s f=%s)" % (sym, s["tone_fa"], s["actual"], s["forecast"])
        print("  [%d %-9s] %-34s %s  %s" % (
            e["weight"], e["impact"], e["name"][:34], e["when"][5:], tag))
    print("-" * 62)
    ss = surprise_score(c)
    print("لحن کلی:", ss.get("tone"), ss.get("tone_fa"), "| n=", ss.get("n"))
    for i in ss.get("items", []):
        print("   %s %-30s a=%s f=%s" % (
            {1: "✅", -1: "❌"}.get(i["tone"], "➖"), i["name"][:30],
            i["actual"], i["forecast"]))
