#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
market_context.py — بخش ۱: حواس پنج گانه ایجنت (Data Ingestion)

  1) تحلیل چند تایم فریمی (Multi-Timeframe)
  2) همبستگی های بین بازاری (DXY / US10Y / SPX / NDX) + SMT Divergence
  3) تقویم اقتصادی و فیلتر خبری (CPI / NFP / FOMC / GDP)
  4) شاخص های احساسی (VIX / VXN / Put-Call proxy)
  +) Killzone های ICT بر اساس ساعت تهران
"""

from __future__ import annotations

import re
import warnings
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

import assets as A

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- نمادها
INTERMARKET = {
    "DX-Y.NYB": dict(name="شاخص دلار (DXY)", key="dxy", expect=-1),
    "^TNX":     dict(name="بازده ۱۰ ساله (US10Y)", key="us10y", expect=-1),
    "^GSPC":    dict(name="اس اند پی ۵۰۰ (SPX)", key="spx", expect=+1),
    "^NDX":     dict(name="نزدک ۱۰۰ (NDX)", key="ndx", expect=+1),
    "^VIX":     dict(name="شاخص ترس (VIX)", key="vix", expect=-1),
}

# ساعات به وقت تهران (UTC+3:30)
TEHRAN_TZ = timezone(timedelta(hours=3, minutes=30))

KILLZONES = [
    dict(key="asia",   name="آسیا (انباشت)",       start=3.5,  end=9.0,
         weight=0.35, note="دامنه آسیا معمولا نقدینگی لندن را می سازد"),
    dict(key="london", name="باز شدن لندن",         start=11.5, end=14.5,
         weight=0.85, note="تعیین جهت روز و شکار نقدینگی آسیا"),
    dict(key="ny_am",  name="باز شدن نیویورک",      start=16.5, end=19.5,
         weight=1.00, note="طلایی ترین پنجره برای داوجونز"),
    dict(key="ny_pm",  name="بعدازظهر نیویورک",     start=19.5, end=22.5,
         weight=0.55, note="ادامه روند یا بازگشت به میانگین"),
    dict(key="ny_close", name="بسته شدن نیویورک",   start=22.5, end=23.9,
         weight=0.20, note="زمان بستن پوزیشن ها، نه باز کردن"),
]

# رویدادهای اقتصادی پرتاثیر (ساعت انتشار به وقت تهران)
HIGH_IMPACT = [
    dict(key="CPI",  name="شاخص قیمت مصرف کننده (CPI)", hour=16.0,
         rule="روز ۱۰ تا ۱۵ هر ماه، معمولا سه شنبه/چهارشنبه"),
    dict(key="NFP",  name="اشتغال غیرکشاورزی (NFP)",    hour=16.0,
         rule="اولین جمعه هر ماه"),
    dict(key="FOMC", name="نشست فدرال رزرو (FOMC)",     hour=21.5,
         rule="هر ۶ تا ۸ هفته، چهارشنبه"),
    dict(key="GDP",  name="تولید ناخالص داخلی (GDP)",   hour=16.0,
         rule="پایان هر فصل، پنجشنبه"),
]

_CACHE: Dict[str, dict] = {}


# ================================================================ کمکی
def _now_tehran() -> datetime:
    return datetime.now(TEHRAN_TZ)


def _hour_float(dt: datetime) -> float:
    return dt.hour + dt.minute / 60.0


def _pct(a: float, b: float) -> float:
    return (a - b) / b * 100 if b else 0.0


def _slope(arr: np.ndarray, k: int = 20) -> float:
    k = min(k, len(arr) - 1)
    if k < 3:
        return 0.0
    y = np.asarray(arr[-k:], dtype=float)
    x = np.arange(k)
    rng = np.ptp(y)
    return float(np.polyfit(x, y, 1)[0] / (rng / k + 1e-9))


def _rsi(c: np.ndarray, n: int = 14) -> np.ndarray:
    s = pd.Series(c)
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50).to_numpy()


# ================================================================ 1) چند تایم فریمی
def _bias_from(df: pd.DataFrame) -> Dict:
    """سوگیری ساختاری یک تایم فریم بر پایه ساختار سوئینگ و میانگین ها."""
    from smart_money import market_structure, atr_array

    if df is None or len(df) < 60:
        return dict(bias=0, label="نامشخص", ema_state="—", rsi=None, atr=None)

    st = market_structure(df, 3, 3)
    c = df["Close"].to_numpy(float)
    ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().to_numpy()
    ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().to_numpy()
    rsi = _rsi(c)
    atr = atr_array(df)

    score = 0
    score += 1 if st["bias"] > 0 else (-1 if st["bias"] < 0 else 0)
    score += 1 if ema20[-1] > ema50[-1] else -1
    score += 1 if c[-1] > ema20[-1] else -1

    bias = 1 if score >= 2 else (-1 if score <= -2 else 0)
    label = "صعودی" if bias > 0 else ("نزولی" if bias < 0 else "خنثی")
    last_ev = st["events"][-1] if st["events"] else None

    return dict(
        bias=bias, label=label, score=int(score),
        ema_state="EMA20 بالای EMA50" if ema20[-1] > ema50[-1] else "EMA20 زیر EMA50",
        rsi=float(rsi[-1]), atr=float(atr[-1]),
        price=float(c[-1]),
        last_event=None if last_ev is None else dict(
            type=last_ev["type"], dir=last_ev["dir"],
            level=float(last_ev["level"]),
            time=pd.Timestamp(last_ev["time"]).isoformat()),
    )


MTF_PLAN = [
    ("1d",  "5y",   "ساختار کلان",  0.30),
    ("4h",  "730d", "ساختار میانی", 0.25),
    ("1h",  "730d", "سطوح کلیدی",   0.20),
    ("15m", "60d",  "SMC میانی",    0.15),
    ("5m",  "60d",  "تریگر ورود",   0.10),
]


def multi_timeframe(symbol: str = "DIA") -> Dict:
    """تحلیل ۵ تایم فریم و ساخت سوگیری وزنی یکپارچه."""
    out: Dict[str, Dict] = {}
    t = yf.Ticker(symbol)
    for iv, per, role, w in MTF_PLAN:
        try:
            if iv == "4h":
                d = t.history(interval="1h", period=per)
                if not d.empty:
                    d = d.resample("4h").agg({
                        "Open": "first", "High": "max", "Low": "min",
                        "Close": "last", "Volume": "sum"}).dropna()
            else:
                d = t.history(interval=iv, period=per)
            if d.empty:
                continue
            d.index = pd.to_datetime(d.index)
            if d.index.tz is not None:
                d.index = d.index.tz_localize(None)
            b = _bias_from(d)
            b.update(role=role, weight=w, candles=len(d))
            out[iv] = b
        except Exception:
            continue

    tot = sum(v["weight"] for v in out.values()) or 1.0
    agg = sum(v["bias"] * v["weight"] for v in out.values()) / tot
    align = [v["bias"] for v in out.values() if v["bias"] != 0]
    agreement = (max((np.array(align) > 0).mean(), (np.array(align) < 0).mean())
                 if align else 0.0)

    if agg >= 0.45:
        verdict = "همراستایی صعودی قوی در تایم فریم ها"
    elif agg <= -0.45:
        verdict = "همراستایی نزولی قوی در تایم فریم ها"
    elif abs(agg) < 0.18:
        verdict = "تضاد بین تایم فریم ها — بازار بدون جهت مشخص"
    else:
        verdict = "همراستایی ضعیف — نیاز به تایید بیشتر"

    return dict(frames=out, aggregate=float(agg), agreement=float(agreement),
                verdict=verdict,
                direction=1 if agg >= 0.45 else (-1 if agg <= -0.45 else 0))


# ================================================================ 2) بین بازاری
def intermarket(ref_df: Optional[pd.DataFrame] = None, period: str = "6mo",
                asset: Optional[str] = None) -> Dict:
    """همبستگی بین بازاری + تشخیص SMT Divergence — وابسته به پروفایل دارایی.

    داوجونز با SPX/NDX مقایسه می شود، ولی طلا با نقره و پلاتین.
    """
    prof = A.profile(asset)
    imap = prof["intermarket"]
    ref_sym = prof["candle_symbol"]
    ref_name = prof["name"]
    syms = list(imap.keys())
    try:
        raw = yf.download(syms + [ref_sym], period=period, interval="1d",
                          group_by="ticker", progress=False,
                          auto_adjust=True, threads=True)
    except Exception as e:
        return dict(ok=False, error=str(e), assets=[], risk_score=0.0)

    def close_of(s):
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                return raw[s]["Close"].dropna()
            return raw["Close"].dropna()
        except Exception:
            return pd.Series(dtype=float)

    dia = close_of(ref_sym)
    if dia.empty:
        return dict(ok=False, error="داده مرجع نیامد", assets=[], risk_score=0.0)

    dia_r = dia.pct_change().dropna()
    assets: List[Dict] = []
    risk = 0.0

    for s, meta in imap.items():
        c = close_of(s)
        if len(c) < 30:
            continue
        r = c.pct_change().dropna()
        j = pd.concat([dia_r, r], axis=1).dropna()
        corr = float(j.corr().iloc[0, 1]) if len(j) > 10 else 0.0
        corr20 = (float(j.tail(20).corr().iloc[0, 1])
                  if len(j) > 20 else corr)

        arr = c.to_numpy(float)
        k = min(10, len(arr) - 1)
        chg = _pct(arr[-1], arr[-k])
        sma20 = float(pd.Series(arr).rolling(20).mean().iloc[-1])
        above = arr[-1] > sma20
        sl = _slope(arr, 20)

        # سهم این دارایی در ریسک پوزیشن لانگ روی داوجونز
        impact = meta["expect"] * np.tanh(chg / 2.0)
        risk += impact

        if meta["key"] == "vix":
            spike = chg > 12
            if prof["rules"].get("vix_safe_haven"):
                state = ("جهش ترس — معمولا به نفع طلا" if spike
                         else ("افزایش ترس — حمایت از طلا" if chg > 4
                               else ("آرامش بازار — تقاضای پناهگاه کمتر"
                                     if chg < -4 else "نرمال")))
            else:
                state = ("جهش ترس — خطر برای لانگ" if spike
                         else ("افزایش ترس" if chg > 4
                               else ("آرامش بازار" if chg < -4 else "نرمال")))
        else:
            state = ("صعودی" if above and sl > 0 else
                     ("نزولی" if (not above) and sl < 0 else "خنثی"))

        assets.append(dict(
            symbol=s, key=meta["key"], name=meta["name"],
            price=float(arr[-1]), chg_pct=float(chg),
            corr=corr, corr20=corr20,
            expect=meta["expect"], above_sma20=bool(above),
            slope=float(sl), state=state,
            impact=float(impact),
            supportive=bool(impact > 0.05),
            headwind=bool(impact < -0.05),
        ))

    # ---------- SMT Divergence: داوجونز در برابر SPX و NDX ----------
    smt: List[Dict] = []
    look = 12
    peer_names = prof.get("smt_peer_names", {})
    for peer in prof["smt_peers"]:
        pc = close_of(peer)
        j = pd.concat([dia, pc], axis=1).dropna()
        if len(j) < look + 2:
            continue
        a = j.iloc[:, 0].to_numpy(float)
        b = j.iloc[:, 1].to_numpy(float)
        a_hi = a[-1] >= a[-look:].max() * 0.999
        b_hi = b[-1] >= b[-look:].max() * 0.999
        a_lo = a[-1] <= a[-look:].min() * 1.001
        b_lo = b[-1] <= b[-look:].min() * 1.001
        pn = peer_names.get(peer, peer)
        if a_hi and not b_hi:
            smt.append(dict(peer=peer, type="bearish",
                            text=f"{ref_name} سقف جدید زد ولی {pn} تایید نکرد "
                                 f"— واگرایی SMT نزولی"))
        elif b_hi and not a_hi:
            smt.append(dict(peer=peer, type="bearish_peer",
                            text=f"{pn} سقف جدید زد ولی {ref_name} عقب ماند "
                                 f"— ضعف نسبی {ref_name}"))
        elif a_lo and not b_lo:
            smt.append(dict(peer=peer, type="bullish",
                            text=f"{ref_name} کف جدید زد ولی {pn} تایید نکرد "
                                 f"— واگرایی SMT صعودی"))
        elif b_lo and not a_lo:
            smt.append(dict(peer=peer, type="bullish_peer",
                            text=f"{pn} کف جدید زد ولی {ref_name} مقاوم ماند "
                                 f"— قدرت نسبی {ref_name}"))

    risk = float(np.clip(risk, -3, 3))
    if risk > 0.6:
        verdict = f"شرایط بین بازاری حامی صعود {ref_name}"
    elif risk < -0.6:
        verdict = "شرایط بین بازاری مخالف صعود (باد مخالف)"
    else:
        verdict = "شرایط بین بازاری خنثی"

    return dict(ok=True, assets=assets, smt=smt, risk_score=risk,
                verdict=verdict,
                headwinds=[a["name"] for a in assets if a["headwind"]],
                supports=[a["name"] for a in assets if a["supportive"]])


# ================================================================ 3) تقویم و اخبار
def _nth_weekday(year: int, month: int, weekday: int, n: int) -> datetime:
    d = datetime(year, month, 1)
    add = (weekday - d.weekday()) % 7 + 7 * (n - 1)
    return d + timedelta(days=add)


def _economic_calendar_legacy(days_ahead: int = 10) -> Dict:
    """
    تقویم اقتصادی تخمینی بر پایه قواعد تکرارشونده انتشار.
    توجه: این یک تقویم قاعده محور است، نه فید رسمی. برای دقت کامل باید
    به API تقویم (ForexFactory / Investing) متصل شود.
    """
    now = _now_tehran()
    events: List[Dict] = []

    for m_off in range(0, 2):
        y = now.year + (now.month - 1 + m_off) // 12
        m = (now.month - 1 + m_off) % 12 + 1

        # NFP: اولین جمعه ماه، ساعت ۱۶:۰۰ تهران
        nfp = _nth_weekday(y, m, 4, 1).replace(hour=16, tzinfo=TEHRAN_TZ)
        events.append(dict(key="NFP", name="اشتغال غیرکشاورزی (NFP)", when=nfp,
                           impact="بسیار بالا"))

        # CPI: حدود روز ۱۳ ماه، ساعت ۱۶:۰۰
        try:
            cpi = datetime(y, m, 13, 16, tzinfo=TEHRAN_TZ)
            while cpi.weekday() > 4:
                cpi += timedelta(days=1)
            events.append(dict(key="CPI", name="شاخص قیمت مصرف کننده (CPI)",
                               when=cpi, impact="بسیار بالا"))
        except ValueError:
            pass

        # GDP: حدود روز ۲۸، پنجشنبه نزدیک
        try:
            g = datetime(y, m, 28, 16, tzinfo=TEHRAN_TZ)
            while g.weekday() != 3:
                g += timedelta(days=1)
            events.append(dict(key="GDP", name="تولید ناخالص داخلی (GDP)",
                               when=g, impact="بالا"))
        except ValueError:
            pass

    # FOMC: نشست های تقریبی سال (چهارشنبه، ۲۱:۳۰ تهران)
    fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]
    for m in fomc_months:
        for y in {now.year, now.year + 1}:
            try:
                f = _nth_weekday(y, m, 2, 3).replace(hour=21, minute=30,
                                                     tzinfo=TEHRAN_TZ)
                events.append(dict(key="FOMC", name="نشست فدرال رزرو (FOMC)",
                                   when=f, impact="بسیار بالا"))
            except Exception:
                continue

    horizon = now + timedelta(days=days_ahead)
    up = sorted([e for e in events if now - timedelta(hours=2) <= e["when"] <= horizon],
                key=lambda e: e["when"])

    # آیا الان در پنجره ممنوعه هستیم؟
    blocked, blocker = False, None
    for e in events:
        delta = abs((e["when"] - now).total_seconds()) / 60.0
        if delta <= 15:
            blocked, blocker = True, e
            break

    nxt = up[0] if up else None
    hrs = ((nxt["when"] - now).total_seconds() / 3600.0) if nxt else None

    return dict(
        ok=True, now=now.isoformat(), blocked=blocked,
        blocker=None if not blocker else dict(
            key=blocker["key"], name=blocker["name"],
            when=blocker["when"].isoformat()),
        next_event=None if not nxt else dict(
            key=nxt["key"], name=nxt["name"], when=nxt["when"].isoformat(),
            impact=nxt["impact"], hours=round(hrs, 1)),
        upcoming=[dict(key=e["key"], name=e["name"],
                       when=e["when"].isoformat(), impact=e["impact"],
                       hours=round((e["when"] - now).total_seconds() / 3600, 1))
                  for e in up[:6]],
        note="تقویم قاعده محور — برای معاملات واقعی با فید رسمی تطبیق دهید",
    )


# ---------- تحلیل احساسی تیترهای خبری (NLP سبک، بدون مدل سنگین) ----------
POS_WORDS = {
    "rally","surge","jump","gain","gains","soar","climb","rise","rises","higher",
    "beat","beats","boost","strong","upgrade","bullish","record","optimism",
    "recovery","rebound","cut","cuts","dovish","ease","eases","growth","profit",
}
NEG_WORDS = {
    "fall","falls","drop","drops","plunge","slump","sink","lower","tumble",
    "miss","misses","weak","weakness","downgrade","bearish","fear","fears",
    "recession","selloff","crash","hike","hikes","hawkish","inflation","tariff",
    "concern","concerns","risk","warns","warning","cut jobs","layoff","slowdown",
}
MACRO_WORDS = {
    "fed","fomc","powell","cpi","inflation","rate","rates","treasury","yield",
    "jobs","payroll","gdp","tariff","recession",
}

def economic_calendar(days_ahead: int = 10) -> Dict:
    """تقویم اقتصادی واقعی از فدرال رزرو و FRED.

    نسخه قبلی تاریخ ها را حدس می زد (مثلا «CPI حدود روز ۱۳ ماه»).
    حالا هر تاریخ از منبع رسمی می آید. اگر شبکه قطع بود، به نسخه
    قاعده محور برمی گردد و با source_tier=estimated علامت می خورد.
    """
    try:
        import real_data
        r = real_data.economic_calendar(max(days_ahead, 30))
        if r.get("ok") and r.get("events"):
            evs = []
            for e in r["events"][:40]:
                evs.append(dict(
                    key=e["key"], name=e["name"], impact=e["impact"],
                    when=e["when"], when_iso=e.get("when_iso"),
                    minutes_until=e["minutes_until"],
                    hours_until=e["hours_until"],
                    days_until=e["days_until"],
                    # نام های سازگار با کد قدیمی
                    hours=e["hours_until"],
                    minutes=e["minutes_until"],
                    days=e["days_until"],
                    note=e.get("note", ""), source=e.get("source", "")))
            nxt = evs[0] if evs else None
            return dict(ok=True, events=evs, count=len(evs), next_event=nxt,
                        sources=r.get("sources", []),
                        source_tier="real",
                        note="تقویم واقعی — فدرال رزرو و FRED")
    except Exception:
        pass
    out = _economic_calendar_legacy(days_ahead)
    out["source_tier"] = "estimated"
    out["note"] = "بازگشت به تقویم قاعده محور (منبع رسمی در دسترس نبود)"
    return out


def _news_sentiment_yahoo(symbol: str = "DIA", limit: int = 12) -> Dict:
    """امتیاز احساسی تیترهای خبری با روش واژگانی (Lexicon-based NLP)."""
    try:
        items = yf.Ticker(symbol).news or []
    except Exception as e:
        return dict(ok=False, error=str(e), score=0.0, items=[])

    out: List[Dict] = []
    tot, macro_hits = 0.0, 0
    now = datetime.now(timezone.utc)

    for a in items[:limit]:
        c = a.get("content", a)
        title = str(c.get("title", "")).strip()
        if not title:
            continue
        pub = c.get("pubDate") or c.get("providerPublishTime")
        try:
            ts = (pd.Timestamp(pub).tz_convert("UTC") if isinstance(pub, str)
                  else pd.Timestamp(pub, unit="s", tz="UTC"))
            age_h = (now - ts.to_pydatetime()).total_seconds() / 3600
        except Exception:
            ts, age_h = None, 48.0

        words = set(re.findall(r"[a-z']+", title.lower()))
        p = len(words & POS_WORDS)
        n = len(words & NEG_WORDS)
        is_macro = bool(words & MACRO_WORDS)
        if is_macro:
            macro_hits += 1

        raw = (p - n) / max(1, p + n)
        decay = float(np.exp(-max(age_h, 0) / 36.0))   # اخبار کهنه وزن کمتر
        w = raw * decay * (1.35 if is_macro else 1.0)
        tot += w

        out.append(dict(
            title=title[:130],
            sentiment=("مثبت" if raw > 0.15 else
                       ("منفی" if raw < -0.15 else "خنثی")),
            raw=round(raw, 3), weight=round(w, 3),
            macro=is_macro, age_h=round(age_h, 1),
            when=None if ts is None else ts.isoformat(),
        ))

    score = float(np.clip(tot / max(1, len(out)) * 2.2, -1, 1))
    if score > 0.22:
        label = "فضای خبری مثبت"
    elif score < -0.22:
        label = "فضای خبری منفی"
    else:
        label = "فضای خبری خنثی"

    return dict(ok=True, score=score, label=label, n=len(out),
                macro_share=round(macro_hits / max(1, len(out)), 2),
                items=out[:8])


# ================================================================ 4) احساسات


def news_sentiment(symbol: str = "DIA", limit: int = 40) -> Dict:
    """تحلیل احساسات خبری از منابع واقعی.

    ترتیب اولویت:
      1) Alpha Vantage  — امتیاز NLP حرفه ای (سقف ۲۵ درخواست روزانه)
      2) Finnhub        — ۱۰۰ خبر بلادرنگ + واژگان داخلی
      3) Yahoo          — نسخه قدیمی، فقط ۸ خبر (آخرین سنگر)
    """
    result = None

    # ۱) امتیاز عددی حرفه ای
    try:
        import real_data
        av = real_data.av_sentiment("AAPL,MSFT,JPM,WMT,CAT")
        if av.get("ok") and av.get("count", 0) >= 5:
            items = [dict(title=i["title"], score=i["score"],
                          label=i["label"], source=i.get("source", ""))
                     for i in av["items"][:12]]
            result = dict(
                ok=True, score=round(av["avg_score"] * 100, 2),
                raw_score=av["avg_score"], label=av["label"],
                bullish=av["bullish"], bearish=av["bearish"],
                neutral=av["neutral"], n=av["count"], items=items,
                engine="Alpha Vantage NLP", source_tier="real",
                note="امتیاز احساسات حرفه ای روی سهام بزرگ داو")
    except Exception:
        pass

    # ۲) حجم بالای خبر با واژگان داخلی
    if result is None:
        try:
            import real_data
            fn = real_data.finnhub_news(limit=limit)
            if fn.get("ok") and fn.get("items"):
                pos = neg = 0
                scored = []
                for it in fn["items"]:
                    words = set(re.findall(r"[a-z']+", it["title"].lower()))
                    p = len(words & POS_WORDS)
                    n = len(words & NEG_WORDS)
                    if p > n:
                        pos += 1
                        lab = "مثبت"
                    elif n > p:
                        neg += 1
                        lab = "منفی"
                    else:
                        lab = "خنثی"
                    scored.append(dict(title=it["title"][:150],
                                       score=round((p - n) * 0.1, 3),
                                       label=lab, source=it.get("source", ""),
                                       time=it.get("time", "")))
                tot = len(scored) or 1
                sc = round((pos - neg) / tot * 100, 2)
                result = dict(
                    ok=True, score=sc, label=("مثبت" if sc > 8 else
                                              "منفی" if sc < -8 else "خنثی"),
                    bullish=pos, bearish=neg, neutral=tot - pos - neg,
                    n=tot, items=scored[:12],
                    engine="Finnhub + واژگان", source_tier="real",
                    note="۱۰۰ خبر بلادرنگ با امتیازدهی واژگانی")
        except Exception:
            pass

    # ۳) آخرین سنگر
    if result is None:
        result = _news_sentiment_yahoo(symbol, 12)
        result["engine"] = "Yahoo (محدود)"
        result["source_tier"] = "estimated"

    return result


def sentiment_block() -> Dict:
    """VIX، VXN و تخمین Put/Call از روی ساختار نوسان."""
    res: Dict = dict(ok=True)
    try:
        v = yf.Ticker("^VIX").history(period="3mo", interval="1d")["Close"].dropna()
        vx = yf.Ticker("^VXN").history(period="3mo", interval="1d")["Close"].dropna()
    except Exception as e:
        return dict(ok=False, error=str(e))

    if v.empty:
        return dict(ok=False, error="داده VIX نیامد")

    arr = v.to_numpy(float)
    last = float(arr[-1])
    prev = float(arr[-2]) if len(arr) > 1 else last
    chg = _pct(last, prev)
    ma20 = float(pd.Series(arr).rolling(20).mean().iloc[-1])
    pctile = float((arr < last).mean() * 100)

    spike = chg > 12 or last > ma20 * 1.25
    if last < 14:
        regime = "آرامش بیش از حد — ریسک غافلگیری"
    elif last < 20:
        regime = "نوسان نرمال — شرایط مناسب روند"
    elif last < 28:
        regime = "نوسان بالا — کاهش حجم معاملات"
    else:
        regime = "ترس شدید — فقط معاملات دفاعی"

    res.update(vix=dict(
        last=last, chg_pct=float(chg), ma20=ma20,
        percentile=pctile, spike=bool(spike), regime=regime,
        close_longs=bool(spike and last > 20),
    ))

    if not vx.empty:
        vxa = vx.to_numpy(float)
        ratio = float(vxa[-1] / last) if last else 1.0
        res["vxn"] = dict(last=float(vxa[-1]), ratio=ratio,
                          note=("ریسک تکنولوژی بالاتر از بازار" if ratio > 1.25
                                else "ریسک متوازن"))

    # تخمین Put/Call از شیب ترم استراکچر نوسان
    try:
        v9 = yf.Ticker("^VIX9D").history(period="1mo", interval="1d")["Close"].dropna()
        if not v9.empty:
            ts = float(v9.iloc[-1] / last)
            res["term_structure"] = dict(
                ratio=ts,
                state=("بک واردیشن — ترس کوتاه مدت (اشباع فروش محتمل)"
                       if ts > 1.02 else "کنتانگو — بازار آرام"),
            )
    except Exception:
        pass

    # نسبت واقعی Put/Call از زنجیره آپشن — جایگزین فرمول ساختگی قبلی
    try:
        import real_data
        pcr = real_data.put_call_real("DIA")
        if pcr.get("ok"):
            val = pcr.get("put_call_volume") or pcr.get("put_call_oi")
            res["put_call"] = dict(
                value=val,
                volume_ratio=pcr.get("put_call_volume"),
                oi_ratio=pcr.get("put_call_oi"),
                put_volume=pcr.get("put_volume"),
                call_volume=pcr.get("call_volume"),
                state=pcr.get("state"),
                contrarian=pcr.get("contrarian"),
                source_tier="computed",
                note="از حجم و Open Interest واقعی آپشن")
            res["put_call_proxy"] = res["put_call"]
    except Exception:
        pass

    if "put_call" not in res:
        pc = float(np.clip(0.75 + (last - 18) / 45, 0.45, 1.6))
        res["put_call_proxy"] = dict(
            value=round(pc, 3),
            state=("اشباع فروش (پوت زیاد)" if pc > 1.05 else
                   ("اشباع خرید (کال زیاد)" if pc < 0.72 else "متعادل")),
            source_tier="estimated",
            note="تخمین از VIX — زنجیره آپشن در دسترس نبود",
        )

    # صدک واقعی VIX از تاریخچه ۱۹۹۰ به بعد
    try:
        import real_data
        vp = real_data.vix_percentile(last)
        if vp.get("ok"):
            res["vix_percentile"] = dict(
                value=vp["value"], all_time=vp["percentile_all"],
                one_year=vp["percentile_1y"], five_year=vp["percentile_5y"],
                regime=vp["regime"], n_days=vp["n_days"],
                quartiles=vp["quartiles"],
                source_tier="real", source="Cboe از ۱۹۹۰")
    except Exception:
        pass

    return res


# ================================================================ Killzone
def _dynamic_killzones(ts: Optional[datetime] = None):
    """کیل زون های لنگر شده به بازگشایی واقعی (DST-aware)."""
    try:
        import market_hours as mh
        return mh.killzones_tehran(ts)
    except Exception:
        return KILLZONES


def killzone_state(ts: Optional[datetime] = None,
                   asset: Optional[str] = None) -> Dict:
    now = ts or _now_tehran()
    h = _hour_float(now)
    zones = _dynamic_killzones(now)
    active = None
    for kz in zones:
        if kz["start"] <= h < kz["end"]:
            active = kz
            break

    nxt, wait = None, None
    for kz in sorted(zones, key=lambda k: k["start"]):
        if kz["start"] > h:
            nxt, wait = kz, kz["start"] - h
            break
    if nxt is None:
        nxt = zones[0]
        wait = 24 - h + nxt["start"]

    weekday = now.weekday()

    # --- وضعیت واقعی بازار از منبع واحد ---
    _prof = A.profile(asset)
    _metal = _prof["hours"]["mode"] == "metal"
    try:
        if _metal:
            # طلا تقویم بورس ندارد؛ تقریبا ۲۴ ساعته معامله می شود
            ms = A.market_state(_prof["key"])
            market_open = bool(ms.get("is_open"))
            phase = "regular" if market_open else "closed"
            phase_label = ("بازار طلا باز" if market_open
                           else str(ms.get("reason", "بازار طلا بسته")))
            countdown = ms.get("reason")
            session = dict(open_tehran="۰۱:۰۰ یکشنبه",
                           close_tehran="۲۳:۰۰ جمعه",
                           note=_prof["hours"]["note"])
            holiday = None
        else:
            import market_hours as mh
            ms = mh.market_status(now)
            market_open = ms["is_open"]
            phase = ms["phase"]
            phase_label = ms["label"]
            countdown = ms["countdown"]
            session = ms["session"]
            holiday = ms.get("holiday_name")
    except Exception:
        market_open = weekday < 5
        phase = "unknown"
        phase_label = "نامشخص"
        countdown = None
        session = {}
        holiday = None

    # کیل زونی که به بازار باز نیاز دارد، وقتی بورس بسته است فعال نمی شود
    # (برای طلا این قید اعمال نمی شود چون بازارش تقریبا ۲۴ ساعته است)
    if active and active.get("requires_open") and not market_open and not _metal:
        active = None

    return dict(
        now=now.strftime("%Y-%m-%d %H:%M"), hour=round(h, 2),
        weekday=["دوشنبه", "سه شنبه", "چهارشنبه", "پنجشنبه",
                 "جمعه", "شنبه", "یکشنبه"][weekday],
        market_open=market_open,
        phase=phase, phase_label=phase_label, countdown=countdown,
        session=session, holiday=holiday,
        active=None if not active else dict(
            key=active["key"], name=active["name"],
            weight=active["weight"], note=active["note"],
            ends_in=round(active["end"] - h, 2)),
        next=dict(key=nxt["key"], name=nxt["name"],
                  starts_in=round(wait, 2), weight=nxt["weight"]),
        weight=0.0 if not market_open else (active["weight"] if active else 0.15),
        tradeable=bool(active and active["weight"] >= 0.5 and market_open),
        zones=[dict(key=k["key"], name=k["name"],
                    window=f"{int(k['start']):02d}:{int(k['start'] % 1 * 60):02d}"
                           f"–{int(k['end']):02d}:{int(k['end'] % 1 * 60):02d}",
                    weight=k["weight"], note=k["note"]) for k in zones],
    )


# ================================================================ تجمیع
def build_context(symbol: str = "DIA", with_mtf: bool = True,
                  asset: Optional[str] = None) -> Dict:
    prof = A.profile(asset) if asset else None
    if prof is not None:
        symbol = prof["candle_symbol"]
    ctx: Dict = {}
    ctx["asset"] = (prof or A.profile("US30"))["key"]
    ctx["killzone"] = killzone_state(asset=asset)
    ctx["calendar"] = economic_calendar()
    try:
        ctx["sentiment"] = sentiment_block()
    except Exception as e:
        ctx["sentiment"] = dict(ok=False, error=str(e))
    try:
        _nsym = (prof["news_symbols"][0] if prof else symbol)
        ctx["news"] = news_sentiment(_nsym)
    except Exception as e:
        ctx["news"] = dict(ok=False, error=str(e))
    try:
        ctx["intermarket"] = intermarket(asset=asset)
    except Exception as e:
        ctx["intermarket"] = dict(ok=False, error=str(e))
    if with_mtf:
        try:
            ctx["mtf"] = multi_timeframe(symbol)
        except Exception as e:
            ctx["mtf"] = dict(frames={}, aggregate=0.0, error=str(e))

    # ---------- دروازه معاملاتی (Trade Gate) ----------
    blocks: List[str] = []
    warns: List[str] = []

    cal = ctx["calendar"]
    if cal.get("blocked"):
        blocks.append(f"پنجره خبری: {cal['blocker']['name']} (±۱۵ دقیقه)")
    elif cal.get("next_event") and cal["next_event"]["hours"] is not None \
            and 0 <= cal["next_event"]["hours"] <= 2:
        warns.append(f"{cal['next_event']['name']} تا "
                     f"{cal['next_event']['hours']:.1f} ساعت دیگر")

    sen = ctx.get("sentiment", {})
    if sen.get("ok") and sen.get("vix", {}).get("close_longs"):
        blocks.append(f"جهش VIX به {sen['vix']['last']:.1f} — بستن پوزیشن های لانگ")
    elif sen.get("ok") and sen.get("vix", {}).get("spike"):
        warns.append("افزایش ناگهانی VIX")

    kz = ctx["killzone"]
    if not kz["market_open"]:
        blocks.append(kz.get("phase_label") or "بازار تعطیل است (آخر هفته)")
    elif not kz["tradeable"]:
        warns.append(f"خارج از Killzone اصلی — وزن جلسه {kz['weight']:.2f}")

    im = ctx.get("intermarket", {})
    if im.get("ok") and abs(im.get("risk_score", 0)) > 1.2:
        warns.append(f"فشار بین بازاری قوی: {im['verdict']}")

    ctx["gate"] = dict(
        allowed=len(blocks) == 0,
        mode=("NO-TRADE" if blocks else ("CAUTION" if warns else "NORMAL")),
        blocks=blocks, warnings=warns,
        session_weight=kz["weight"],
    )
    return ctx
