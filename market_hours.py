#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
market_hours.py — منبع واحد حقیقت برای ساعت بازار

چرا این ماژول ساخته شد:
  پیش از این دو تعریف ناسازگار وجود داشت —
    live_feed  : ساعت ثابت ۱۷:۰۰ تا ۲۳:۳۰ تهران
    market_context : کیل زون ICT که از ۱۶:۳۰ شروع می شد
  نتیجه: بین ۱۶:۳۰ تا ۱۷:۰۰ یکی می گفت «بازار بسته» و دیگری «قابل معامله».

  بدتر اینکه ساعت ثابت فقط در فصل تابستان آمریکا درست بود.
  با پایان DST (اول نوامبر) بازار ۱۸:۰۰ تهران باز می شود نه ۱۷:۰۰.

این ماژول همه چیز را از منطقه زمانی رسمی نیویورک محاسبه می کند،
بنابراین DST خودکار اعمال می شود و تعطیلات رسمی هم لحاظ می گردد.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
TEHRAN = ZoneInfo("Asia/Tehran")

# ساعت رسمی بورس نیویورک (به وقت محلی نیویورک)
PRE_OPEN = time(4, 0)
REG_OPEN = time(9, 30)
REG_CLOSE = time(16, 0)
POST_CLOSE = time(20, 0)

# نیم روزهای معاملاتی: بسته شدن ساعت ۱۳:۰۰ نیویورک
HALF_DAY_CLOSE = time(13, 0)


# ================================================================ تعطیلات
def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n اُمین weekday ماه. n منفی یعنی از آخر."""
    if n > 0:
        d = date(year, month, 1)
        shift = (weekday - d.weekday()) % 7
        return d + timedelta(days=shift + 7 * (n - 1))
    d = date(year, month, 1)
    nxt = date(year + (month == 12), (month % 12) + 1, 1)
    last = nxt - timedelta(days=1)
    shift = (last.weekday() - weekday) % 7
    return last - timedelta(days=shift + 7 * (-n - 1))


def _observed(d: date) -> date:
    """اگر تعطیلی شنبه باشد جمعه قبل، اگر یکشنبه باشد دوشنبه بعد."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _easter(year: int) -> date:
    """الگوریتم Anonymous Gregorian برای یافتن عید پاک."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def nyse_holidays(year: int) -> Dict[date, str]:
    """تعطیلات رسمی بورس نیویورک برای یک سال."""
    h: Dict[date, str] = {}
    h[_observed(date(year, 1, 1))] = "سال نو میلادی"
    h[_nth_weekday(year, 1, 0, 3)] = "روز مارتین لوتر کینگ"
    h[_nth_weekday(year, 2, 0, 3)] = "روز رؤسای جمهور"
    h[_easter(year) - timedelta(days=2)] = "جمعه نیک"
    h[_nth_weekday(year, 5, 0, -1)] = "روز یادبود"
    if year >= 2022:
        h[_observed(date(year, 6, 19))] = "روز آزادی (Juneteenth)"
    h[_observed(date(year, 7, 4))] = "روز استقلال"
    h[_nth_weekday(year, 9, 0, 1)] = "روز کارگر"
    h[_nth_weekday(year, 11, 3, 4)] = "روز شکرگزاری"
    h[_observed(date(year, 12, 25))] = "کریسمس"
    return h


def half_days(year: int) -> Dict[date, str]:
    """روزهایی که بازار ساعت ۱۳:۰۰ نیویورک زودتر می بندد."""
    d: Dict[date, str] = {}
    thanks = _nth_weekday(year, 11, 3, 4)
    d[thanks + timedelta(days=1)] = "جمعه پس از شکرگزاری"
    xmas = date(year, 12, 24)
    if xmas.weekday() < 5:
        d[xmas] = "شب کریسمس"
    jul3 = date(year, 7, 3)
    if jul3.weekday() < 5:
        d[jul3] = "شب روز استقلال"
    return d


# ================================================================ وضعیت بازار
def _ny_now(ts: Optional[datetime] = None) -> datetime:
    if ts is None:
        return datetime.now(NY)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=TEHRAN)
    return ts.astimezone(NY)


def session_bounds(d: date) -> Tuple[time, time]:
    """ساعت باز و بسته شدن جلسه عادی برای یک روز مشخص."""
    if d in half_days(d.year):
        return REG_OPEN, HALF_DAY_CLOSE
    return REG_OPEN, REG_CLOSE


def market_status(ts: Optional[datetime] = None) -> Dict:
    """
    وضعیت کامل بازار در یک لحظه.

    خروجی همیشه شامل ساعت معادل تهران است تا هیچ ماژولی
    مجبور به تبدیل دستی و اشتباه نشود.
    """
    ny = _ny_now(ts)
    teh = ny.astimezone(TEHRAN)
    d = ny.date()
    t = ny.time()

    hol = nyse_holidays(d.year)
    half = half_days(d.year)
    is_weekend = ny.weekday() >= 5
    is_holiday = d in hol
    is_half = d in half
    open_t, close_t = session_bounds(d)

    def to_teh(tt: time) -> str:
        return datetime.combine(d, tt, tzinfo=NY).astimezone(TEHRAN).strftime("%H:%M")

    # ---------- فاز ----------
    if is_weekend:
        phase, label, is_open = "weekend", "تعطیل آخر هفته", False
    elif is_holiday:
        phase, label, is_open = "holiday", f"تعطیل رسمی — {hol[d]}", False
    elif t < PRE_OPEN:
        phase, label, is_open = "closed", "بازار بسته", False
    elif t < open_t:
        phase, label, is_open = "pre", "پیش گشایش", False
    elif t < close_t:
        phase, label, is_open = "regular", "بازار باز", True
    elif t < POST_CLOSE:
        phase, label, is_open = "post", "پس از بسته شدن", False
    else:
        phase, label, is_open = "closed", "بازار بسته", False

    if is_half and phase == "regular":
        label = f"بازار باز — نیم روز ({half[d]})"

    # ---------- بازگشایی بعدی ----------
    nxt = d
    for _ in range(12):
        if nxt > d or not (is_weekend or is_holiday) and t < open_t and nxt == d:
            pass
        cand = nxt
        chol = nyse_holidays(cand.year)
        if cand.weekday() < 5 and cand not in chol:
            if cand > d:
                break
            if cand == d and t < session_bounds(cand)[0]:
                break
        nxt = nxt + timedelta(days=1)
    next_open_dt = datetime.combine(nxt, session_bounds(nxt)[0], tzinfo=NY)
    mins_to_open = max(0.0, (next_open_dt - ny).total_seconds() / 60.0)

    close_dt = datetime.combine(d, close_t, tzinfo=NY)
    mins_to_close = (close_dt - ny).total_seconds() / 60.0 if is_open else None

    return dict(
        is_open=is_open, phase=phase, label=label,
        is_weekend=is_weekend, is_holiday=is_holiday,
        holiday_name=hol.get(d), is_half_day=is_half,
        half_day_name=half.get(d),
        ny_time=ny.strftime("%Y-%m-%d %H:%M:%S"),
        tehran_time=teh.strftime("%Y-%m-%d %H:%M:%S"),
        dst=bool(ny.dst() and ny.dst().total_seconds() > 0),
        session=dict(
            open_ny=open_t.strftime("%H:%M"), close_ny=close_t.strftime("%H:%M"),
            open_tehran=to_teh(open_t), close_tehran=to_teh(close_t),
            pre_tehran=to_teh(PRE_OPEN), post_tehran=to_teh(POST_CLOSE),
        ),
        minutes_to_open=round(mins_to_open, 1),
        minutes_to_close=None if mins_to_close is None else round(mins_to_close, 1),
        next_open_tehran=next_open_dt.astimezone(TEHRAN).strftime("%Y-%m-%d %H:%M"),
        countdown=_countdown_text(is_open, mins_to_open, mins_to_close),
    )


def _countdown_text(is_open: bool, to_open: float,
                    to_close: Optional[float]) -> str:
    if is_open and to_close is not None:
        if to_close < 60:
            return f"{int(to_close)} دقیقه تا بسته شدن"
        return f"{to_close/60:.1f} ساعت تا بسته شدن"
    if to_open < 60:
        return f"{int(to_open)} دقیقه تا بازگشایی"
    if to_open < 24 * 60:
        return f"{to_open/60:.1f} ساعت تا بازگشایی"
    return f"{to_open/1440:.1f} روز تا بازگشایی"


def killzones_tehran(ts: Optional[datetime] = None) -> List[Dict]:
    """
    کیل زون های ICT لنگر شده به بازگشایی واقعی نیویورک.

    به جای ساعت ثابت، همه بازه ها نسبت به زمان واقعی بازگشایی
    محاسبه می شوند تا با تغییر DST خودکار جابه جا شوند.
    """
    ny = _ny_now(ts)
    d = ny.date()
    open_t, close_t = session_bounds(d)
    base = datetime.combine(d, open_t, tzinfo=NY)
    endb = datetime.combine(d, close_t, tzinfo=NY)

    def teh_h(dt: datetime) -> float:
        x = dt.astimezone(TEHRAN)
        return x.hour + x.minute / 60

    return [
        dict(key="asia", name="آسیا (انباشت)", weight=0.35,
             start=teh_h(base - timedelta(hours=13)),
             end=teh_h(base - timedelta(hours=7.5)),
             note="دامنه آسیا معمولا نقدینگی لندن را می سازد",
             requires_open=False),
        dict(key="london", name="باز شدن لندن", weight=0.85,
             start=teh_h(base - timedelta(hours=5.5)),
             end=teh_h(base - timedelta(hours=2.5)),
             note="تعیین جهت روز و شکار نقدینگی آسیا",
             requires_open=False),
        dict(key="ny_pre", name="پیش گشایش نیویورک", weight=0.30,
             start=teh_h(base - timedelta(minutes=30)),
             end=teh_h(base),
             note="فیوچرز فعال است ولی بورس هنوز باز نشده",
             requires_open=False),
        dict(key="ny_am", name="باز شدن نیویورک", weight=1.00,
             start=teh_h(base), end=teh_h(base + timedelta(hours=3)),
             note="طلایی ترین پنجره برای داوجونز", requires_open=True),
        dict(key="ny_pm", name="بعدازظهر نیویورک", weight=0.55,
             start=teh_h(base + timedelta(hours=3)),
             end=teh_h(endb - timedelta(hours=1)),
             note="ادامه روند یا بازگشت به میانگین", requires_open=True),
        dict(key="ny_close", name="بسته شدن نیویورک", weight=0.20,
             start=teh_h(endb - timedelta(hours=1)), end=teh_h(endb),
             note="زمان بستن پوزیشن ها، نه باز کردن", requires_open=True),
    ]


def upcoming_holidays(n: int = 5, ts: Optional[datetime] = None) -> List[Dict]:
    ny = _ny_now(ts)
    today = ny.date()
    out: List[Dict] = []
    for yr in (today.year, today.year + 1):
        for d, name in sorted(nyse_holidays(yr).items()):
            if d >= today:
                out.append(dict(date=str(d), name=name,
                                days=(d - today).days, type="تعطیل کامل"))
        for d, name in sorted(half_days(yr).items()):
            if d >= today:
                out.append(dict(date=str(d), name=name,
                                days=(d - today).days, type="نیم روز"))
    out.sort(key=lambda x: x["days"])
    return out[:n]


if __name__ == "__main__":
    s = market_status()
    print("=== وضعیت بازار ===")
    print(f"نیویورک: {s['ny_time']}  |  تهران: {s['tehran_time']}")
    print(f"وضعیت: {s['label']}  (باز={s['is_open']}, فاز={s['phase']})")
    print(f"ساعت وقت تابستانی فعال: {s['dst']}")
    ss = s["session"]
    print(f"جلسه امروز: نیویورک {ss['open_ny']}–{ss['close_ny']}  |  "
          f"تهران {ss['open_tehran']}–{ss['close_tehran']}")
    print(f"شمارش: {s['countdown']}")
    print(f"بازگشایی بعدی: {s['next_open_tehran']}")

    print("\n=== کیل زون ها (لنگر شده به بازگشایی واقعی) ===")
    for k in killzones_tehran():
        print(f"  {k['name']:<24}{k['start']:>5.2f} – {k['end']:<6.2f}"
              f" وزن {k['weight']}  {'(نیاز به بازار باز)' if k['requires_open'] else ''}")

    print("\n=== تعطیلات پیش رو ===")
    for hd in upcoming_holidays(6):
        print(f"  {hd['date']}  {hd['name']:<26}{hd['days']:>4} روز  {hd['type']}")

    print("\n=== آزمون تغییر DST ===")
    for ds in ["2026-09-16", "2026-10-30", "2026-11-05", "2026-12-15"]:
        y, m, dd = map(int, ds.split("-"))
        st = market_status(datetime(y, m, dd, 18, 0, tzinfo=TEHRAN))
        print(f"  {ds} ساعت ۱۸:۰۰ تهران -> {st['label']:<22}"
              f"(بازگشایی تهران {st['session']['open_tehran']})")
