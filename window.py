# -*- coding: utf-8 -*-
"""پنجره معاملاتی — کدام ساعت ها اصلا ارزش معامله دارند.

ایده
────
معیار درست برای «آیا می شود اسکلپ کرد» عدد خام هزینه نیست،
بلکه هزینه نسبت به نوسان همان ساعت است:

    نسبت = هزینه رفت و برگشت ÷ ATR کندل ۵ دقیقه آن ساعت

اندازه گیری ۲۰۲۶-۰۹-۲۹ روی ۶۰ روز داده:

    ساعت ۰۱ UTC (۰۴:۳۰ تهران)
        طلا      ۱۲.۴٪  ✅ قابل معامله
        داوجونز  ۴۲.۹٪  🔴 عملا مرده

    ساعت ۱۳ UTC (۱۶:۳۰ تهران)
        طلا      ۱۰.۱٪  ✅
        داوجونز   ۹.۷٪  ✅ بهترین ساعتش

یعنی «کی معامله می کنید» از «چه ستاپی می زنید» مهم تر است.

⚠ این یک پیش بینی نیست — آمار توصیفی است. می گوید فضای حرکت
هست یا نه، نه اینکه قیمت کجا می رود. ولی نوسان ساعتی پایدارترین
الگوی بازار است، برخلاف جهت قیمت.

پروفایل روی داده واقعی حساب می شود و هاردکد نیست، چون نوسان
ساعتی با رویدادهای کلان تغییر می کند.
"""
from __future__ import annotations

import datetime as dt
import threading
import time
from typing import Dict, List, Optional

import numpy as np

# هزینه رفت و برگشت بر حسب درصد قیمت — از engine_backtest
COST_PCT = {"XAUUSD": 0.020, "US30": 0.012}

# نمادی که نوسان واقعی ۲۳ ساعته را دارد (نه ETF که فقط ۶ ساعت است)
VOL_SYMBOL = {"XAUUSD": "GC=F", "US30": "YM=F"}
TO_SPOT = {"XAUUSD": 0.99194589, "US30": 1.0}

GREEN, YELLOW = 0.15, 0.22       # آستانه نسبت هزینه به نوسان
_TTL = 6 * 3600.0                # پروفایل ساعتی کند تغییر می کند
_lock = threading.Lock()
_cache: Dict[str, Dict] = {}


def _fetch(symbol: str):
    import yfinance as yf
    df = yf.Ticker(symbol).history(interval="5m", period="60d")
    if df is None or df.empty:
        return None
    df = df.dropna(subset=["High", "Low", "Close"])
    if df.empty:
        return None
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df


def profile(asset: str = "US30") -> Dict:
    """نسبت هزینه به نوسان برای هر ساعت UTC."""
    asset = (asset or "US30").upper()
    now = time.time()
    with _lock:
        hit = _cache.get(asset)
        if hit and now - hit["at"] < _TTL:
            d = dict(hit["data"])
            d["age_sec"] = round(now - hit["at"], 1)
            d["from_cache"] = True
            return d

    sym = VOL_SYMBOL.get(asset, "YM=F")
    k = TO_SPOT.get(asset, 1.0)
    df = _fetch(sym)
    if df is None:
        return dict(ok=False, error="داده نوسان در دسترس نیست", asset=asset)

    px = float(df["Close"].iloc[-1]) * k
    cost = px * COST_PCT.get(asset, 0.015) / 100.0

    hi = df["High"].to_numpy(float) * k
    lo = df["Low"].to_numpy(float) * k
    hrs = np.array([t.hour for t in df.index])

    hours: Dict[int, Dict] = {}
    for h in range(24):
        m = hrs == h
        n = int(m.sum())
        if n < 40:                       # نمونه کم = قضاوت نکن
            continue
        atr = float(np.mean(hi[m] - lo[m]))
        if atr <= 0:
            continue
        ratio = cost / atr
        hours[h] = dict(
            atr=round(atr, 2),
            ratio=round(ratio, 4),
            n=n,
            grade=("good" if ratio < GREEN
                   else ("fair" if ratio < YELLOW else "poor")),
        )

    out = dict(ok=True, asset=asset, symbol=sym, price=round(px, 2),
               cost=round(cost, 3),
               cost_pct=COST_PCT.get(asset, 0.015),
               hours=hours,
               thresholds=dict(good=GREEN, fair=YELLOW),
               days=int(len({t.date() for t in df.index})),
               note=("نسبت = هزینه رفت‌وبرگشت ÷ نوسان متوسط کندل ۵ دقیقه "
                     "آن ساعت. هرچه کمتر، فضای بیشتر. این آمار توصیفی "
                     "است نه پیش‌بینی جهت."))
    with _lock:
        _cache[asset] = dict(at=now, data=out)
    out = dict(out)
    out["age_sec"] = 0.0
    out["from_cache"] = False
    return out


def _fa_time(minutes: int) -> str:
    minutes %= 1440
    return "%02d:%02d" % (minutes // 60, minutes % 60)


def status(asset: str = "US30", at: Optional[dt.datetime] = None) -> Dict:
    """وضعیت همین لحظه + پنجره بعدی."""
    p = profile(asset)
    if not p.get("ok"):
        return p
    now = at or dt.datetime.now(dt.UTC)
    h = now.hour
    hours = p["hours"]
    cur = hours.get(h)

    # ── پنجره های سبز پیوسته ──
    good = sorted(x for x, v in hours.items() if v["grade"] == "good")
    windows: List[Dict] = []
    if good:
        start = prev = good[0]
        for x in good[1:]:
            if x == prev + 1:
                prev = x
                continue
            windows.append(dict(from_h=start, to_h=prev + 1))
            start = prev = x
        windows.append(dict(from_h=start, to_h=prev + 1))
        # اگر ۲۳ و ۰ هر دو سبزند، دور زدن روز
        if len(windows) > 1 and windows[0]["from_h"] == 0 \
                and windows[-1]["to_h"] == 24:
            windows[0]["from_h"] = windows[-1]["from_h"] - 24
            windows.pop()

    for w in windows:
        # تهران = UTC + ۳:۳۰
        w["from_fa"] = _fa_time(w["from_h"] * 60 + 210)
        w["to_fa"] = _fa_time(w["to_h"] * 60 + 210)

    in_window = bool(cur and cur["grade"] == "good")

    # چقدر تا پایان پنجره فعلی، یا تا شروع پنجره بعدی
    mins_now = now.hour * 60 + now.minute
    nxt = None
    if in_window:
        for w in windows:
            if w["from_h"] <= h < w["to_h"]:
                nxt = dict(kind="ends_in",
                           minutes=w["to_h"] * 60 - mins_now,
                           at_fa=w["to_fa"])
                break
    else:
        best = None
        for w in windows:
            delta = (w["from_h"] * 60 - mins_now) % 1440
            if best is None or delta < best[0]:
                best = (delta, w)
        if best:
            nxt = dict(kind="starts_in", minutes=best[0],
                       at_fa=best[1]["from_fa"])

    return dict(
        ok=True, asset=asset,
        hour_utc=h,
        now_fa=_fa_time(mins_now + 210),
        in_window=in_window,
        grade=(cur or {}).get("grade", "unknown"),
        ratio=(cur or {}).get("ratio"),
        atr=(cur or {}).get("atr"),
        cost=p["cost"],
        windows=windows,
        next=nxt,
        price=p["price"],
        days=p["days"],
        note=p["note"],
        verdict=_verdict((cur or {}).get("grade", "unknown"),
                         (cur or {}).get("ratio")),
    )


def _verdict(grade: str, ratio: Optional[float]) -> str:
    if grade == "good":
        return ("پنجره مناسب — هزینه %.0f٪ نوسان این ساعت است، "
                "فضای کافی هست." % (100 * (ratio or 0)))
    if grade == "fair":
        return ("پنجره متوسط — هزینه %.0f٪ نوسان. فقط با ستاپ "
                "قوی وارد شوید." % (100 * (ratio or 0)))
    if grade == "poor":
        return ("خارج از پنجره — هزینه %.0f٪ نوسان این ساعت را "
                "می‌خورد. اسکلپ در این ساعت از نظر ریاضی بازنده "
                "است." % (100 * (ratio or 0)))
    return "داده کافی برای این ساعت نیست."
