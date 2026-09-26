# -*- coding: utf-8 -*-
"""تابلوی لحظه ای — نقشه سطوح داوجونز.

فلسفه
─────
موتور ایجنت یک «امتیاز جهت دار» می دهد که ترکیب ۳۲ وزن است، و
تقریبا همه آن وزن ها کند هستند (COT هفتگی، ترس و طمع روزانه،
فصلی ماهانه). تحقیق ۷۷۳ روزه نشان داد برای اسکلپ هیچ کدام از
۳۲ پیکربندی آزمایش شده معنادار نبود.

در مقابل، سطوح ساختاری — POC، خوشه نقدینگی، خلأ قیمتی — پیش بینی
نیستند؛ نقشه اند. می گویند «حجم واقعی کجا معامله شده» و «سفارش ها
احتمالا کجا انباشته اند». این یک واقعیت اندازه گیری شدنی است.

پس این ماژول هیچ «بخر/بفروش» نمی دهد. نقشه می دهد.

چرا فیوچرز
──────────
اندازه گیری ۲۰۲۶-۰۹-۲۵، کندل ۵ دقیقه در ۵ روز:

    DIA   ۳۸۹ کندل    ساعت ۴:۳۰ صبح تهران داده ندارد
    YM=F  ۱۳۴۱ کندل   ۲۳ ساعته، حجم واقعی نهادی

اعداد روی فیوچرز حساب می شوند ولی با «پایه» به مقیاس نقدی
برمی گردند تا با پلتفرم کاربر یکی باشند.

پایه و قیمت نقدی از dow_cash.py می آید — همان ماژولی که نوار
«داوجونز نقدی» بالای صفحه را هم تغذیه می کند، تا هر دو همیشه یک
عدد بگویند.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import dow_cash as DC

FUT = "YM=F"
_TTL = 300.0                     # سطوح کند تکان می خورند
_lock = threading.Lock()
_cache: Dict[str, object] = {"at": 0.0, "data": None}

# مرزهای جلسه به وقت UTC (تهران +۳:۳۰)
#
# ⚠ دقت: روز معاملاتی فیوچرز CME ساعت ۲۲:۰۰ UTC شروع می شود
# (۱۷:۰۰ شیکاگو)، نه با بسته شدن بازار نقدی. اولین نسخه این را
# ۲۰:۰۰ گرفته بود و باعث می شد «جلسه جاری» فقط ۵۵ دقیقه باشد و
# پنجره «شب» کلا خالی در بیاید.
SESSION_ROLL_H = 22              # شروع روز معاملاتی فیوچرز
NY_OPEN_H, NY_OPEN_M = 13, 30    # ۱۷:۰۰ تهران — بازگشایی نقدی
NY_CLOSE_H = 20                  # ۲۳:۳۰ تهران — بسته شدن نقدی
LONDON_OPEN_H = 7                # ۱۰:۳۰ تهران


# ─────────────────────────────────────────────────────────────
#  داده
# ─────────────────────────────────────────────────────────────

def _load(interval: str = "5m", period: str = "5d") -> Optional[pd.DataFrame]:
    import yfinance as yf
    df = yf.Ticker(FUT).history(interval=interval, period=period)
    if df is None or df.empty:
        return None
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if df.empty:
        return None
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df


# ─────────────────────────────────────────────────────────────
#  پروفایل حجمی
# ─────────────────────────────────────────────────────────────

def volume_profile(df: pd.DataFrame, bins: int = 60) -> Dict:
    """POC و محدوده ارزش، از حجم واقعی معامله شده.

    هر کندل حجمش را به طور یکنواخت بین کف و سقف خودش پخش
    می کند. این روش استاندارد «پروفایل حجمی» است.
    """
    if df is None or df.empty or "Volume" not in df:
        return dict(ok=False)
    lo, hi = float(df["Low"].min()), float(df["High"].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return dict(ok=False)

    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    acc = np.zeros(bins, dtype=float)

    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    v = df["Volume"].to_numpy(float)
    for i in range(len(df)):
        if not np.isfinite(v[i]) or v[i] <= 0:
            continue
        a, b = l[i], h[i]
        if b <= a:
            idx = int(np.clip(np.searchsorted(edges, a) - 1, 0, bins - 1))
            acc[idx] += v[i]
            continue
        first = int(np.clip(np.searchsorted(edges, a) - 1, 0, bins - 1))
        last = int(np.clip(np.searchsorted(edges, b) - 1, 0, bins - 1))
        n = last - first + 1
        if n <= 0:
            continue
        acc[first:last + 1] += v[i] / n

    total = float(acc.sum())
    if total <= 0:
        return dict(ok=False)

    poc_i = int(np.argmax(acc))
    # محدوده ارزش ۷۰٪: از POC به بیرون گسترش بده
    target = total * 0.70
    lo_i = hi_i = poc_i
    got = acc[poc_i]
    while got < target and (lo_i > 0 or hi_i < bins - 1):
        down = acc[lo_i - 1] if lo_i > 0 else -1.0
        up = acc[hi_i + 1] if hi_i < bins - 1 else -1.0
        if up >= down:
            hi_i += 1
            got += acc[hi_i]
        else:
            lo_i -= 1
            got += acc[lo_i]

    # گره های کم حجم = خلأ نقدینگی
    thin = float(np.percentile(acc[acc > 0], 20)) if (acc > 0).any() else 0.0
    lvn = [float(centers[i]) for i in range(bins)
           if 0 < acc[i] <= thin]

    return dict(ok=True,
                poc=float(centers[poc_i]),
                poc_vol_pct=round(100.0 * acc[poc_i] / total, 2),
                vah=float(centers[hi_i]), val=float(centers[lo_i]),
                lvn=lvn, total_vol=total,
                bin_width=float(edges[1] - edges[0]))


# ─────────────────────────────────────────────────────────────
#  نوسان ها = استخر نقدینگی
# ─────────────────────────────────────────────────────────────

def swing_levels(df: pd.DataFrame, left: int = 3, right: int = 3,
                 limit: int = 40) -> List[Dict]:
    """سقف و کف های محلی. «دست نخورده» یعنی قیمت بعدا از آن رد نشده.

    چرا مهم است: بالای سقف های دست نخورده سفارش های حد ضرر
    خریداران و حد سود فروشندگان جمع می شود. همان چیزی که به آن
    «نقدینگی» می گویند.
    """
    if df is None or len(df) < left + right + 2:
        return []
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    idx = df.index
    out: List[Dict] = []

    for i in range(left, len(df) - right):
        win_h = h[i - left:i + right + 1]
        win_l = l[i - left:i + right + 1]
        if h[i] == win_h.max() and (win_h.argmax() == left):
            after = h[i + right + 1:]
            out.append(dict(kind="سقف", level=float(h[i]),
                            at=idx[i].isoformat(),
                            untested=bool(len(after) == 0 or after.max() < h[i])))
        if l[i] == win_l.min() and (win_l.argmin() == left):
            after = l[i + right + 1:]
            out.append(dict(kind="کف", level=float(l[i]),
                            at=idx[i].isoformat(),
                            untested=bool(len(after) == 0 or after.min() > l[i])))

    out.sort(key=lambda r: r["at"], reverse=True)
    return out[:limit]


# ─────────────────────────────────────────────────────────────
#  خلأ قیمتی (FVG)
# ─────────────────────────────────────────────────────────────

def imbalances(df: pd.DataFrame, limit: int = 12) -> List[Dict]:
    """شکاف سه کندلی — جایی که قیمت بدون معامله رد شده."""
    if df is None or len(df) < 3:
        return []
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    idx = df.index
    out: List[Dict] = []
    for i in range(2, len(df)):
        if l[i] > h[i - 2]:                        # خلأ صعودی
            top, bot = float(l[i]), float(h[i - 2])
            filled = bool(l[i:].min() <= bot)
            out.append(dict(dir="صعودی", top=top, bottom=bot,
                            mid=(top + bot) / 2.0,
                            at=idx[i].isoformat(), filled=filled))
        elif h[i] < l[i - 2]:                      # خلأ نزولی
            top, bot = float(l[i - 2]), float(h[i])
            filled = bool(h[i:].max() >= top)
            out.append(dict(dir="نزولی", top=top, bottom=bot,
                            mid=(top + bot) / 2.0,
                            at=idx[i].isoformat(), filled=filled))
    out = [g for g in out if not g["filled"]]
    out.sort(key=lambda r: r["at"], reverse=True)
    return out[:limit]


# ─────────────────────────────────────────────────────────────
#  ساخت تابلو
# ─────────────────────────────────────────────────────────────

def _session_start(ts: pd.Timestamp) -> pd.Timestamp:
    """شروع روز معاملاتی فیوچرز که این زمان داخلش است."""
    start = ts.normalize() + pd.Timedelta(hours=SESSION_ROLL_H)
    if ts < start:
        start -= pd.Timedelta(days=1)
    return start


def _sessions(df: pd.DataFrame) -> Dict:
    """سقف و کف جلسه جاری، پنجره شب، و روز معاملاتی قبل."""
    out: Dict[str, object] = {}
    if df is None or df.empty:
        return out
    last = df.index[-1]
    day_start = _session_start(last)

    cur = df[df.index >= day_start]
    if len(cur):
        out["session_high"] = float(cur["High"].max())
        out["session_low"] = float(cur["Low"].min())
        out["session_open"] = float(cur["Open"].iloc[0])
        out["_session_hours"] = round(
            (last - day_start).total_seconds() / 3600.0, 1)

    # پنجره شب: از شروع جلسه تا بازگشایی بازار نقدی نیویورک
    ny_open = day_start.normalize() + pd.Timedelta(
        days=1, hours=NY_OPEN_H, minutes=NY_OPEN_M)
    if ny_open <= day_start:
        ny_open += pd.Timedelta(days=1)
    overnight = df[(df.index >= day_start) & (df.index < ny_open)]
    if len(overnight) >= 3:
        out["overnight_high"] = float(overnight["High"].max())
        out["overnight_low"] = float(overnight["Low"].min())

    prev_start = day_start - pd.Timedelta(days=1)
    prev = df[(df.index >= prev_start) & (df.index < day_start)]
    if len(prev) >= 3:
        out["prev_high"] = float(prev["High"].max())
        out["prev_low"] = float(prev["Low"].min())
        out["prev_close"] = float(prev["Close"].iloc[-1])
    return out


# قاعده های واکنش — گزینه «ب» که کاربر انتخاب کرد.
# ⚠ این ها بک تست نشده اند و با برچسب صریح نمایش داده می شوند.
_RULES = {
    "POC": "قیمت معمولا به POC برمی گردد. برخورد اول معمولا واکنش دارد.",
    "سقف": "اگر جارو شد و بلافاصله پس گرفته شد، شکار نقدینگی بوده.",
    "کف": "اگر جارو شد و بلافاصله پس گرفته شد، شکار نقدینگی بوده.",
    "خلأ": "خلأ پر نشده معمولا مغناطیس است تا وقتی پر شود.",
    "محدوده": "لبه محدوده ارزش، مرز پذیرش قیمت است.",
}


def build(top: int = 10) -> Dict:
    """نقشه کامل سطوح، روی مقیاس نقدی (= پلتفرم کاربر)."""
    t0 = time.time()
    df = _load("5m", "5d")
    if df is None or df.empty:
        return dict(ok=False, error="داده فیوچرز در دسترس نیست")

    try:
        bs = DC.compute_basis()
        basis = float(bs.get("basis") or 0.0) if bs.get("ok") else 0.0
    except Exception:
        bs, basis = {}, 0.0
    try:
        cp = DC.cash_price()
    except Exception:
        cp = {}
    fut_last = float(df["Close"].iloc[-1])
    price = round(fut_last - basis, 1)          # مقیاس نقدی

    def cash(x):                                 # فیوچرز → نقدی
        return None if x is None else round(float(x) - basis, 1)

    ses = _sessions(df)

    # ── پنجره پروفایل ──
    # استاندارد، پروفایل «جلسه جاری» است. ولی درست بعد از رول اور
    # ساعت ۲۲:۰۰ فقط چند کندل داریم و POC بی معنی می شود؛ در آن
    # حالت به ۲۴ ساعت اخیر برمی گردیم.
    day_start = _session_start(df.index[-1])
    cur = df[df.index >= day_start]
    if len(cur) >= 72:                           # حداقل ۶ ساعت
        prof_df, prof_label = cur, "جلسه جاری"
    else:
        cutoff = df.index[-1] - pd.Timedelta(hours=24)
        prof_df, prof_label = df[df.index >= cutoff], "۲۴ ساعت اخیر"
    vp = volume_profile(prof_df)

    sw = swing_levels(df)
    gaps = imbalances(df)

    levels: List[Dict] = []

    def add(name, raw, kind, why, pinned=False, extra=None):
        if raw is None:
            return
        try:
            v = float(raw)
        except Exception:
            return
        if not np.isfinite(v):
            return
        lv = cash(v)
        levels.append(dict(
            name=name, level=lv, kind=kind, why=why, pinned=bool(pinned),
            dist=round(lv - price, 1), dist_abs=abs(round(lv - price, 1)),
            rule=_RULES.get(kind), **(extra or {})))

    # ── سنجاق شده: همیشه دیده می شوند، هرچقدر هم دور باشند ──
    if vp.get("ok"):
        add("POC (%s)" % prof_label, vp["poc"], "POC",
            "بیشترین حجم واقعی معامله شده — %.1f٪ کل حجم"
            % vp["poc_vol_pct"], pinned=True,
            extra=dict(vol_pct=vp["poc_vol_pct"]))
        add("سقف محدوده ارزش", vp["vah"], "محدوده",
            "لبه بالای ۷۰٪ حجم", pinned=True)
        add("کف محدوده ارزش", vp["val"], "محدوده",
            "لبه پایین ۷۰٪ حجم", pinned=True)

    # ── سطوح جلسه ──
    for k, nm, why in (
            ("session_high", "سقف جلسه", "بالاترین قیمت از شروع روز معاملاتی"),
            ("session_low", "کف جلسه", "پایین ترین قیمت از شروع روز معاملاتی"),
            ("overnight_high", "سقف شب", "سقف پیش از بازگشایی نیویورک"),
            ("overnight_low", "کف شب", "کف پیش از بازگشایی نیویورک"),
            ("prev_high", "سقف روز قبل", "سقف روز معاملاتی قبل"),
            ("prev_low", "کف روز قبل", "کف روز معاملاتی قبل"),
            ("prev_close", "بسته روز قبل", "قیمت بسته شدن قبلی")):
        if k in ses:
            kind = "سقف" if "high" in k else ("کف" if "low" in k else "محدوده")
            add(nm, ses[k], kind, why)

    # ── نقدینگی دست نخورده ──
    for sv in [x for x in sw if x["untested"]][:8]:
        add("%s دست‌نخورده" % sv["kind"], sv["level"], sv["kind"],
            "نقدینگی انباشته — قیمت هنوز از آن رد نشده",
            extra=dict(at=sv["at"], untested=True))

    # ── خلأ پر نشده ──
    for g in gaps[:5]:
        add("خلأ %s" % g["dir"], g["mid"], "خلأ",
            "شکاف پر نشده بین %s و %s" % (cash(g["bottom"]), cash(g["top"])),
            extra=dict(top=cash(g["top"]), bottom=cash(g["bottom"])))

    # ── ادغام سطوح تقریبا یکسان ──
    # مثلا «کف جلسه» و «کف دست نخورده» اغلب یک عدد اند؛ دوبار
    # نشان دادنشان تابلو را شلوغ می کند.
    tol = max(5.0, price * 0.0002)               # حدود ۱۰ واحد
    merged: List[Dict] = []
    for L in sorted(levels, key=lambda r: (-int(r["pinned"]), r["dist_abs"])):
        hit = None
        for m in merged:
            if abs(m["level"] - L["level"]) <= tol:
                hit = m
                break
        if hit is None:
            L["also"] = []
            merged.append(L)
        else:
            if L["name"] not in hit["also"]:
                hit["also"].append(L["name"])
            hit["pinned"] = hit["pinned"] or L["pinned"]

    pinned = [m for m in merged if m["pinned"]]
    rest = sorted([m for m in merged if not m["pinned"]],
                  key=lambda r: r["dist_abs"])
    out_levels = pinned + rest[:max(0, top - len(pinned))]
    out_levels.sort(key=lambda r: -r["level"])   # از بالا به پایین

    return dict(
        ok=True,
        price=price,
        price_fut=round(fut_last, 1),
        basis=round(basis, 1),
        asof=df.index[-1].isoformat(),
        candles=int(len(df)),
        profile_window=prof_label,
        session_hours=ses.get("_session_hours"),
        levels=out_levels,
        align=dict(cash=cp.get("price"),
                   cash_src=cp.get("source_fa"),
                   market_open=cp.get("market_open"),
                   age_min=cp.get("age_min"),
                   fut=bs.get("futures_at"),
                   basis_samples=bs.get("samples"),
                   basis_age_h=bs.get("age_hours")),
        rules_validated=False,
        rules_note=("قاعده ها راهنمای عمومی معامله گری اند و روی این "
                    "سیستم بک تست نشده اند. سطح ها واقعی و اندازه گیری "
                    "شده اند؛ قاعده ها نه."),
        source=dict(symbol=FUT,
                    note="سطوح روی فیوچرز حساب و با پایه به مقیاس "
                         "نقدی برگردانده شده"),
        took=round(time.time() - t0, 2),
    )


def cached(force: bool = False) -> Dict:
    now = time.time()
    with _lock:
        if (not force and _cache["data"] is not None
                and now - float(_cache["at"]) < _TTL):
            d = dict(_cache["data"])         # type: ignore[arg-type]
            d["age_sec"] = round(now - float(_cache["at"]), 1)
            d["from_cache"] = True
            return d
    d = build()
    if d.get("ok"):
        with _lock:
            _cache["at"] = now
            _cache["data"] = d
    d = dict(d)
    d["age_sec"] = 0.0
    d["from_cache"] = False
    return d
