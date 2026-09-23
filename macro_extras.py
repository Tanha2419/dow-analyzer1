# -*- coding: utf-8 -*-
"""
سه لایه کمکی برای طلا و داوجونز:

  ۱. تایمر کیل زون ICT      — شمارش معکوس تا پنجره بعدی
  ۲. همبستگی زنده با دلار    — و تشخیص شکست رابطه
  ۳. شمارش معکوس فدرال رزرو  — از تقویم رسمی خود فدرال رزرو

همه اعداد از داده واقعی می آیند. هیچ عدد حدسی ساخته نمی شود؛
اگر منبعی در دسترس نباشد، مقدار None برمی گردد و لایه
با برچسب «—» نمایش داده می شود.
"""
from __future__ import annotations

import datetime as _dt
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ImportError:                                    # pragma: no cover
    ZoneInfo = None                                    # type: ignore

_TEHRAN = ZoneInfo("Asia/Tehran") if ZoneInfo else None
_NY = ZoneInfo("America/New_York") if ZoneInfo else None

_CACHE: Dict[str, tuple] = {}


def _cached(key: str, ttl: int, fn):
    """کش ساده در حافظه تا به منابع بیرونی فشار نیاید."""
    import time
    now = time.time()
    hit = _CACHE.get(key)
    if hit and (now - hit[0]) < ttl:
        return hit[1]
    val = fn()
    _CACHE[key] = (now, val)
    return val


# ═══════════════════════════════════════════════ ۱. تایمر کیل زون
def killzone_timer(asset: str = "US30") -> Dict:
    """
    وضعیت کیل زون های ICT به همراه شمارش معکوس دقیق.

    پنجره ها از market_hours می آیند که به بازگشایی واقعی نیویورک
    لنگر شده اند، پس تغییر ساعت تابستانی خودکار اعمال می شود.
    """
    try:
        import market_hours as mh
        zones = mh.killzones_tehran()
    except Exception as e:
        return dict(ok=False, error=str(e)[:120])

    now = _dt.datetime.now(_TEHRAN) if _TEHRAN else _dt.datetime.now()
    h = now.hour + now.minute / 60 + now.second / 3600
    weekend = now.weekday() in (3, 4) and False   # بازار جمعه شب بسته می شود

    out: List[Dict] = []
    active = None
    for z in sorted(zones, key=lambda k: k["start"]):
        on = z["start"] <= h < z["end"]
        if on:
            active = z
        if on:
            mins = (z["end"] - h) * 60
            state, note = "active", "پایان تا"
        elif z["start"] > h:
            mins = (z["start"] - h) * 60
            state, note = "upcoming", "شروع تا"
        else:
            mins = (24 - h + z["start"]) * 60
            state, note = "passed", "فردا تا"
        out.append(dict(
            key=z.get("key"), name=z.get("name"),
            start=round(z["start"], 2), end=round(z["end"], 2),
            weight=z.get("weight"), note=z.get("note"),
            state=state, label=note,
            minutes=round(mins, 1),
        ))

    upcoming = [z for z in out if z["state"] == "upcoming"]
    nxt = min(upcoming, key=lambda z: z["minutes"]) if upcoming else None
    if nxt is None:
        passed = [z for z in out if z["state"] == "passed"]
        nxt = min(passed, key=lambda z: z["minutes"]) if passed else None

    return dict(
        ok=True,
        now_tehran=now.strftime("%H:%M:%S"),
        active=active and dict(
            name=active.get("name"), weight=active.get("weight"),
            ends_in_min=round((active["end"] - h) * 60, 1),
            note=active.get("note")),
        next=nxt,
        zones=out,
        source="market_hours (لنگر به بازگشایی واقعی نیویورک)",
    )


# ═══════════════════════════════════════════════ ۲. همبستگی با دلار
_DXY_SYMBOL = "DX-Y.NYB"       # تنها نماد شاخص دلار که در یاهو پایدار است


def _series(symbol: str, period: str = "1y") -> Optional[pd.Series]:
    try:
        import yfinance as yf
        d = yf.Ticker(symbol).history(interval="1d", period=period)
        if d is None or d.empty or "Close" not in d:
            return None
        d = d.dropna(subset=["Close"])
        idx = pd.to_datetime(d.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        s = pd.Series(d["Close"].to_numpy(float), index=idx.normalize())
        return s[~s.index.duplicated(keep="last")]
    except Exception:
        return None


def dxy_correlation(asset: str = "XAUUSD") -> Dict:
    """
    همبستگی روزانه دارایی با شاخص دلار در سه پنجره زمانی.

    نکته تحلیلی: طلا و دلار معمولا خلاف هم حرکت می کنند. وقتی
    همبستگی کوتاه مدت به شکل معناداری از بلندمدت فاصله می گیرد،
    یعنی رابطه شکسته و معمولا یکی از دو بازار در حال قیمت گذاری
    یک خبر تازه است.
    """
    sym = {"XAUUSD": "GC=F", "US30": "DIA"}.get(asset, "DIA")

    def _build():
        a = _series(sym)
        d = _series(_DXY_SYMBOL)
        if a is None or d is None or len(a) < 30 or len(d) < 30:
            return dict(ok=False, error="داده کافی برای همبستگی نیست")

        j = pd.concat([a.rename("a"), d.rename("d")], axis=1,
                      sort=True).dropna()
        if len(j) < 30:
            return dict(ok=False, error="روز مشترک کافی نیست")

        r = j.pct_change().dropna()
        cors = {}
        for w in (20, 60, 120):
            if len(r) >= w:
                c = r["a"].tail(w).corr(r["d"].tail(w))
                cors[w] = None if (c is None or not np.isfinite(c)) \
                    else round(float(c), 3)
            else:
                cors[w] = None

        short, long = cors.get(20), cors.get(120)
        broken, gap = False, None
        if short is not None and long is not None:
            gap = round(short - long, 3)
            # شکست یعنی: یا علامت برعکس شده، یا فاصله بزرگ است
            broken = (abs(gap) >= 0.35) or (short * long < 0 and abs(short) > .2)

        if short is None:
            tone, tone_fa = "unknown", "نامشخص"
        elif short <= -0.4:
            tone, tone_fa = "strong_inverse", "معکوس قوی (عادی)"
        elif short <= -0.15:
            tone, tone_fa = "inverse", "معکوس ملایم"
        elif short < 0.15:
            tone, tone_fa = "decoupled", "بی ارتباط"
        else:
            tone, tone_fa = "positive", "هم جهت (غیرعادی)"

        dxy_last = float(d.iloc[-1])
        dxy_chg = float((d.iloc[-1] / d.iloc[-2] - 1) * 100) if len(d) > 1 else None

        return dict(
            ok=True, asset=asset, symbol=sym, dxy_symbol=_DXY_SYMBOL,
            corr_20=cors.get(20), corr_60=cors.get(60), corr_120=cors.get(120),
            gap=gap, broken=bool(broken), tone=tone, tone_fa=tone_fa,
            dxy_last=round(dxy_last, 3),
            dxy_change_pct=None if dxy_chg is None else round(dxy_chg, 3),
            days=len(j), last_date=str(j.index[-1].date()),
            note=("رابطه شکسته — یکی از دو بازار خبر تازه ای را قیمت گذاری می کند"
                  if broken else "رابطه در محدوده عادی"),
            source="Yahoo Finance · روزانه · ۱ سال",
        )

    try:
        return _cached(f"dxy_{asset}", 1800, _build)
    except Exception as e:
        return dict(ok=False, error=str(e)[:150])


# ═══════════════════════════════════════════════ ۳. شمارش معکوس فدرال رزرو
# منبع: تقویم رسمی federalreserve.gov/monetarypolicy/fomccalendars.htm
# تاریخ ها روز دوم نشست اند (روزی که تصمیم اعلام می شود).
# ستاره = همراه با جدول پیش بینی های اقتصادی (SEP) — این جلسات
# معمولا نوسان بیشتری می سازند.
_FOMC: List[tuple] = [
    # (سال, ماه, روز اعلام, دارای SEP)
    (2026, 1, 28, False),
    (2026, 3, 18, True),
    (2026, 4, 29, False),
    (2026, 6, 17, True),
    (2026, 7, 29, False),
    (2026, 9, 16, True),
    (2026, 10, 28, False),
    (2026, 12, 9, True),
    (2027, 1, 27, False),
    (2027, 3, 17, True),
    (2027, 4, 28, False),
    (2027, 6, 9, True),
    (2027, 7, 28, False),
    (2027, 9, 15, True),
    (2027, 10, 27, False),
    (2027, 12, 8, True),
    (2028, 1, 26, False),
]

# اعلام تصمیم ساعت ۱۴:۰۰ به وقت نیویورک، کنفرانس خبری ۱۴:۳۰
_FOMC_HOUR, _FOMC_MIN = 14, 0


def fomc_countdown() -> Dict:
    """شمارش معکوس تا جلسه بعدی فدرال رزرو، از تقویم رسمی."""
    if _NY is None:
        return dict(ok=False, error="منطقه زمانی در دسترس نیست")

    now = _dt.datetime.now(_NY)
    nxt = None
    for (y, m, d, sep) in _FOMC:
        when = _dt.datetime(y, m, d, _FOMC_HOUR, _FOMC_MIN, tzinfo=_NY)
        if when > now:
            nxt = (when, sep)
            break

    if nxt is None:
        return dict(ok=False, error="تقویم فدرال رزرو نیاز به به روزرسانی دارد")

    when, sep = nxt
    delta = when - now
    total_min = delta.total_seconds() / 60
    days = int(total_min // 1440)
    hours = int((total_min % 1440) // 60)
    mins = int(total_min % 60)

    teh = when.astimezone(_TEHRAN) if _TEHRAN else when

    if total_min <= 24 * 60:
        risk, risk_fa = "extreme", "بسیار بالا — معامله جدید باز نکنید"
    elif total_min <= 72 * 60:
        risk, risk_fa = "high", "بالا — حجم را کم کنید"
    elif total_min <= 7 * 24 * 60:
        risk, risk_fa = "elevated", "رو به افزایش"
    else:
        risk, risk_fa = "normal", "عادی"

    # آخرین جلسه برگزار شده
    prev = None
    for (y, m, d, s) in reversed(_FOMC):
        w = _dt.datetime(y, m, d, _FOMC_HOUR, _FOMC_MIN, tzinfo=_NY)
        if w <= now:
            prev = w
            break

    return dict(
        ok=True,
        when_ny=when.strftime("%Y-%m-%d %H:%M"),
        when_tehran=teh.strftime("%Y-%m-%d %H:%M"),
        date=when.strftime("%Y-%m-%d"),
        days=days, hours=hours, minutes=mins,
        total_minutes=round(total_min, 1),
        total_days=round(total_min / 1440, 2),
        has_projections=bool(sep),
        projections_note=("همراه با جدول پیش بینی اقتصادی — نوسان معمولا بیشتر"
                          if sep else "بدون جدول پیش بینی"),
        risk=risk, risk_fa=risk_fa,
        last_meeting=prev.strftime("%Y-%m-%d") if prev else None,
        press_conference_ny=when.replace(hour=14, minute=30).strftime("%H:%M"),
        source="federalreserve.gov — تقویم رسمی FOMC",
    )


# ═══════════════════════════════════════════════ بسته کامل
def build_extras(asset: str = "US30") -> Dict:
    """هر سه لایه در یک فراخوانی. هر خطا جدا مهار می شود."""
    out: Dict = dict(ok=True, asset=asset)
    for key, fn in (("killzone", lambda: killzone_timer(asset)),
                    ("dxy", lambda: dxy_correlation(asset)),
                    ("fomc", fomc_countdown)):
        try:
            out[key] = fn()
        except Exception as e:
            out[key] = dict(ok=False, error=str(e)[:150])
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(build_extras("XAUUSD"), ensure_ascii=False, indent=2))
