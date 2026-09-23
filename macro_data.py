#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
macro_data.py — داده های کلان اقتصادی واقعی (بدون کلید API)

منابع تست شده:
  * FRED از طریق pandas_datareader  — ۱۵ سری، کاملا رایگان
  * فیوچرز نرخ بهره ZQ=F           — پیش بینی فدرال رزرو
  * تقویم درآمدی yfinance           — ۳۰ شرکت داوجونز

هیچ عدد تخمینی ساخته نمی شود؛ هر مقدار مستقیما از منبع می آید.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_CACHE: Dict[str, tuple] = {}


def _cached(key: str, ttl: float, fn):
    import time
    now = time.time()
    if key in _CACHE:
        ts, val = _CACHE[key]
        if now - ts < ttl:
            return val
    val = fn()
    _CACHE[key] = (now, val)
    return val


# ================================================================ FRED
FRED_SERIES = {
    # کد          (عنوان فارسی، دسته، جهت مطلوب برای سهام)
    "DFF":        ("نرخ موثر فدرال", "نرخ بهره", -1),
    "DGS2":       ("بازده ۲ ساله", "نرخ بهره", -1),
    "DGS10":      ("بازده ۱۰ ساله", "نرخ بهره", -1),
    "T10Y2Y":     ("اسپرد ۱۰ منهای ۲ ساله", "منحنی بازده", +1),
    "T10YIE":     ("انتظار تورم ۱۰ ساله", "تورم", -1),
    "CPIAUCSL":   ("شاخص قیمت مصرف کننده", "تورم", -1),
    "UNRATE":     ("نرخ بیکاری", "اشتغال", -1),
    "PAYEMS":     ("اشتغال غیرکشاورزی", "اشتغال", +1),
    "GDPC1":      ("تولید ناخالص داخلی حقیقی", "رشد", +1),
    "UMCSENT":    ("اعتماد مصرف کننده میشیگان", "پیشرو", +1),
    "BAMLH0A0HYM2": ("اسپرد اوراق پرریسک", "ریسک اعتباری", -1),
    "NFCI":       ("شاخص شرایط مالی شیکاگو", "شرایط مالی", -1),
    "WALCL":      ("ترازنامه فدرال رزرو", "نقدینگی", +1),
    "M2SL":       ("حجم نقدینگی M2", "نقدینگی", +1),
    "VIXCLS":     ("شاخص نوسان VIX", "ریسک", -1),
}

# طبقه بندی پیشرو / همزمان / تاخیری
INDICATOR_CLASS = {
    "UMCSENT": "پیشرو", "NFCI": "پیشرو", "T10Y2Y": "پیشرو",
    "BAMLH0A0HYM2": "پیشرو", "VIXCLS": "پیشرو",
    "PAYEMS": "همزمان", "GDPC1": "همزمان", "WALCL": "همزمان",
    "M2SL": "همزمان", "DFF": "همزمان",
    "UNRATE": "تاخیری", "CPIAUCSL": "تاخیری",
    "DGS2": "همزمان", "DGS10": "همزمان", "T10YIE": "همزمان",
}


def fred_block(days: int = 900) -> Dict:
    """دریافت همه سری های FRED و محاسبه تغییرات و امتیاز کلان."""
    def _fetch():
        try:
            import pandas_datareader.data as web
        except Exception:
            return dict(ok=False, error="pandas_datareader نصب نیست", items=[])

        end = datetime.now()
        start = end - timedelta(days=days)
        items: List[Dict] = []
        errors: List[str] = []

        try:
            df = web.DataReader(list(FRED_SERIES.keys()), "fred", start, end)
        except Exception as e:
            return dict(ok=False, error=f"FRED در دسترس نیست: {type(e).__name__}",
                        items=[])

        for sid, (name, cat, want) in FRED_SERIES.items():
            if sid not in df.columns:
                errors.append(sid)
                continue
            s = df[sid].dropna()
            if s.empty:
                errors.append(sid)
                continue
            last = float(s.iloc[-1])
            prev = float(s.iloc[-2]) if len(s) > 1 else last
            chg = last - prev
            # تغییر ۳ ماهه
            cutoff = s.index[-1] - pd.Timedelta(days=90)
            s3 = s[s.index >= cutoff]
            chg3 = float(last - s3.iloc[0]) if len(s3) > 1 else 0.0
            pct3 = (chg3 / abs(s3.iloc[0]) * 100) if len(s3) > 1 and s3.iloc[0] else 0.0
            # صدک تاریخی
            pctile = float((s < last).mean() * 100)

            items.append(dict(
                id=sid, name=name, category=cat,
                klass=INDICATOR_CLASS.get(sid, "همزمان"),
                value=round(last, 4), prev=round(prev, 4),
                change=round(chg, 4),
                change_3m=round(chg3, 4), change_3m_pct=round(pct3, 3),
                percentile=round(pctile, 1),
                date=str(s.index[-1].date()),
                n=len(s), want=want,
            ))
        return dict(ok=True, items=items, missing=errors,
                    fetched=datetime.now().isoformat())

    return _cached("fred", 3600.0, _fetch)


def macro_score(fred: Dict) -> Dict:
    """امتیاز کلان: هر شاخص بر اساس جهت مطلوب و شدت تغییر ۳ ماهه."""
    if not fred.get("ok"):
        return dict(ok=False, score=0.0, label="داده کلان در دسترس نیست")

    contrib: List[Dict] = []
    total = 0.0
    for it in fred["items"]:
        p3 = it["change_3m_pct"]
        # نرمال سازی: تغییر ۳ ماهه بیش از ۱۵٪ اشباع می شود
        mag = float(np.clip(p3 / 15.0, -1, 1))
        c = mag * it["want"]
        # شاخص های پیشرو وزن بیشتری دارند
        w = {"پیشرو": 1.3, "همزمان": 1.0, "تاخیری": 0.6}[it["klass"]]
        c *= w
        total += c
        if abs(c) > 0.08:
            contrib.append(dict(name=it["name"], impact=round(c, 3),
                                change_3m_pct=p3, klass=it["klass"]))

    contrib.sort(key=lambda x: -abs(x["impact"]))
    n = max(len(fred["items"]), 1)
    norm = float(np.clip(total / n * 6, -3, 3))

    if norm > 1.0:
        label = "شرایط کلان حامی رشد سهام"
    elif norm > 0.3:
        label = "شرایط کلان نسبتا مثبت"
    elif norm < -1.0:
        label = "شرایط کلان منقبض — فشار بر سهام"
    elif norm < -0.3:
        label = "شرایط کلان نسبتا منفی"
    else:
        label = "شرایط کلان خنثی"

    return dict(ok=True, score=round(norm, 3), label=label,
                top=contrib[:6], raw=round(total, 3))


def yield_curve(fred: Dict) -> Dict:
    """وضعیت منحنی بازده — معکوس شدن، هشدار رکود."""
    if not fred.get("ok"):
        return dict(ok=False)
    g = {i["id"]: i for i in fred["items"]}
    spread = g.get("T10Y2Y", {}).get("value")
    if spread is None:
        return dict(ok=False)
    if spread < 0:
        state, note = "معکوس", "منحنی بازده معکوس — هشدار تاریخی رکود"
    elif spread < 0.25:
        state, note = "تخت", "منحنی بازده بسیار تخت — نزدیک به معکوس شدن"
    elif spread < 1.0:
        state, note = "نرمال باریک", "شیب مثبت ولی باریک"
    else:
        state, note = "شیب دار", "منحنی بازده سالم و شیب دار"
    return dict(ok=True, spread_10y2y=spread, state=state, note=note,
                dgs2=g.get("DGS2", {}).get("value"),
                dgs10=g.get("DGS10", {}).get("value"),
                dff=g.get("DFF", {}).get("value"),
                inflation_expect=g.get("T10YIE", {}).get("value"))


# ================================================================ Fed Watch
def fed_watch() -> Dict:
    """
    پیش بینی نرخ بهره از فیوچرز ZQ=F.
    قیمت فیوچرز = 100 منهای نرخ موثر مورد انتظار.
    """
    def _fetch():
        import yfinance as yf
        out: Dict = dict(ok=False)
        try:
            h = yf.Ticker("ZQ=F").history(period="3mo")
            if h.empty:
                return dict(ok=False, error="فیوچرز در دسترس نیست")
            price = float(h["Close"].iloc[-1])
            implied = 100.0 - price
            prev = float(h["Close"].iloc[-2]) if len(h) > 1 else price
            implied_prev = 100.0 - prev
            m1 = h["Close"].iloc[-22] if len(h) >= 22 else h["Close"].iloc[0]
            implied_m1 = 100.0 - float(m1)

            fr = fred_block()
            cur = None
            if fr.get("ok"):
                for i in fr["items"]:
                    if i["id"] == "DFF":
                        cur = i["value"]
            out = dict(ok=True, futures_price=round(price, 4),
                       implied_rate=round(implied, 4),
                       implied_prev=round(implied_prev, 4),
                       implied_1m_ago=round(implied_m1, 4),
                       current_rate=cur)
            if cur is not None:
                diff_bp = (implied - cur) * 100
                out["diff_bp"] = round(diff_bp, 1)
                if diff_bp <= -12:
                    out["expectation"] = "بازار کاهش نرخ را قیمت گذاری کرده"
                    out["direction"] = -1
                elif diff_bp >= 12:
                    out["expectation"] = "بازار افزایش نرخ را قیمت گذاری کرده"
                    out["direction"] = 1
                else:
                    out["expectation"] = "بازار تغییر نرخ را انتظار ندارد"
                    out["direction"] = 0
                out["cuts_priced"] = round(-diff_bp / 25.0, 2)
            shift = (implied - implied_m1) * 100
            out["shift_1m_bp"] = round(shift, 1)
            out["trend"] = ("انتظارات به سمت سیاست انبساطی" if shift < -8 else
                            ("انتظارات به سمت سیاست انقباضی" if shift > 8 else
                             "انتظارات تقریبا ثابت"))
        except Exception as e:
            out = dict(ok=False, error=f"{type(e).__name__}: {str(e)[:70]}")
        return out

    return _cached("fedwatch", 1800.0, _fetch)


# ================================================================ درآمدها
DOW30 = ["AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
         "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
         "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ",
         "WMT", "DOW"]


def earnings_calendar(days_ahead: int = 30, limit: int = 30) -> Dict:
    """تقویم گزارش درآمدی شرکت های داوجونز از yfinance."""
    def _fetch():
        import yfinance as yf
        today = datetime.now().date()
        horizon = today + timedelta(days=days_ahead)
        rows: List[Dict] = []
        errs = 0
        for sym in DOW30[:limit]:
            try:
                cal = yf.Ticker(sym).calendar
                if not isinstance(cal, dict):
                    continue
                ed = cal.get("Earnings Date")
                if not ed:
                    continue
                d0 = ed[0] if isinstance(ed, (list, tuple)) else ed
                if not hasattr(d0, "year"):
                    continue
                if not (today <= d0 <= horizon):
                    continue
                avg = cal.get("Earnings Average")
                hi = cal.get("Earnings High")
                lo = cal.get("Earnings Low")
                disp = None
                if avg and hi and lo and avg:
                    disp = round((hi - lo) / abs(avg) * 100, 1)
                rows.append(dict(
                    symbol=sym, date=str(d0),
                    days=(d0 - today).days,
                    eps_avg=None if avg is None else round(float(avg), 4),
                    eps_high=None if hi is None else round(float(hi), 4),
                    eps_low=None if lo is None else round(float(lo), 4),
                    dispersion_pct=disp,
                    revenue_avg=cal.get("Revenue Average"),
                ))
            except Exception:
                errs += 1
                continue
        rows.sort(key=lambda r: r["days"])
        cluster: Dict[str, int] = {}
        for r in rows:
            cluster[r["date"]] = cluster.get(r["date"], 0) + 1
        heavy = sorted([(d, c) for d, c in cluster.items() if c >= 2],
                       key=lambda t: t[0])
        return dict(ok=True, items=rows, n=len(rows), errors=errs,
                    heavy_days=[dict(date=d, count=c) for d, c in heavy],
                    note=("روزهای شلوغ گزارش درآمد ریسک نوسان بالا دارند"
                          if heavy else "تجمع گزارش درآمدی مشاهده نشد"))

    return _cached(f"earn{days_ahead}", 7200.0, _fetch)


# ================================================================ تجمیع
def build_macro(with_earnings: bool = True) -> Dict:
    fr = fred_block()
    out: Dict = dict(fred=fr, score=macro_score(fr),
                     curve=yield_curve(fr), fed=fed_watch())
    if with_earnings:
        out["earnings"] = earnings_calendar()
    # دسته بندی برای نمایش
    if fr.get("ok"):
        groups: Dict[str, List[Dict]] = {}
        for it in fr["items"]:
            groups.setdefault(it["klass"], []).append(it)
        out["by_class"] = groups
    return out


if __name__ == "__main__":
    import json
    m = build_macro()
    fr = m["fred"]
    print(f"=== FRED: {len(fr.get('items', []))} سری ===")
    print(f"{'شاخص':<26}{'مقدار':>14}{'۳ماهه٪':>10}{'صدک':>7}  دسته")
    for i in fr.get("items", []):
        print(f"{i['name']:<26}{i['value']:>14,.2f}{i['change_3m_pct']:>+10.2f}"
              f"{i['percentile']:>7.0f}  {i['klass']}")
    s = m["score"]
    print(f"\nامتیاز کلان: {s['score']:+.2f} — {s['label']}")
    for t in s.get("top", [])[:5]:
        print(f"   {t['name']:<28}{t['impact']:+.3f}")
    c = m["curve"]
    if c.get("ok"):
        print(f"\nمنحنی بازده: {c['state']} (اسپرد {c['spread_10y2y']}) — {c['note']}")
    f = m["fed"]
    if f.get("ok"):
        print(f"\nFed Watch: نرخ ضمنی {f['implied_rate']:.3f}٪ در برابر فعلی "
              f"{f.get('current_rate')}٪ ({f.get('diff_bp')} بیپ)")
        print(f"  {f.get('expectation')} | {f.get('trend')}")
    e = m.get("earnings", {})
    if e.get("ok"):
        print(f"\nگزارش درآمدی ۳۰ روز آینده: {e['n']} شرکت")
        for r in e["items"][:8]:
            print(f"   {r['symbol']:<6}{r['date']}  ({r['days']} روز)  EPS {r['eps_avg']}")
