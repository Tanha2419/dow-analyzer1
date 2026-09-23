#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
real_data.py — منابع داده واقعی، بدون هیچ عدد تخیلی

هر تابع یکی از حدس های قبلی سیستم را با داده رسمی جایگزین می کند:

  1) fomc_calendar()      تاریخ نشست های فدرال رزرو    <- سایت رسمی Fed   (بدون کلید)
  2) fred_release_dates() تقویم انتشار CPI/NFP/GDP     <- FRED API        (کلید)
  3) economic_calendar()  تقویم یکپارچه واقعی          <- ترکیب دو مورد بالا
  4) vix_history()        ۹۲۰۰+ روز VIX از ۱۹۹۰        <- Cboe CSV        (بدون کلید)
  5) treasury_yields()    نرخ خزانه داری رسمی          <- Treasury CSV    (بدون کلید)
  6) put_call_real()      نسبت واقعی Put/Call          <- زنجیره آپشن یاهو
  7) finnhub_news()       ۱۰۰ خبر بلادرنگ              <- Finnhub         (کلید)
  8) av_sentiment()       امتیاز احساسات عددی          <- Alpha Vantage   (کلید)

قرارداد شفافیت — هر خروجی کلید source_tier دارد:
    "real"      داده مستقیم از منبع رسمی
    "computed"  از داده واقعی محاسبه شده
    "estimated" مدل تقریبی (با احتیاط استفاده شود)
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

TEHRAN = timezone(timedelta(hours=3, minutes=30))
NY_OFFSET_HINT = "America/New_York"
UA = {"User-Agent": "Mozilla/5.0 (compatible; DowAnalyzer/1.0)"}

_ENV_CACHE: Optional[Dict[str, str]] = None
_CACHE: Dict[str, tuple] = {}
_LOCK = threading.RLock()


# ================================================================ زیرساخت

def env(key: str, default: str = "") -> str:
    """خواندن کلید از .env — یک بار خوانده و نگهداری می شود."""
    global _ENV_CACHE
    if _ENV_CACHE is None:
        _ENV_CACHE = {}
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    _ENV_CACHE[k.strip()] = v.strip().strip('"').strip("'")
        except FileNotFoundError:
            pass
    return os.environ.get(key) or _ENV_CACHE.get(key, default)


def _cached(key: str, ttl: float, fn):
    """کش ساده حافظه ای تا از سقف نرخ سرویس ها عبور نکنیم."""
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and (time.time() - hit[0]) < ttl:
            return hit[1]
    val = fn()
    with _LOCK:
        _CACHE[key] = (time.time(), val)
    return val


def _get(url: str, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=timeout).read()


def _get_json(url: str, timeout: float = 20.0):
    return json.loads(_get(url, timeout))


def _now_teh() -> datetime:
    return datetime.now(TEHRAN)


# ================================================================ ۱) FOMC

FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}


def fomc_calendar(_ttl: float = 43200) -> Dict:
    """تاریخ واقعی نشست های FOMC از سایت رسمی فدرال رزرو (بدون کلید)."""

    def _fetch() -> Dict:
        try:
            html = _get(FOMC_URL, 25).decode("utf-8", "ignore")
        except Exception as exc:
            return dict(ok=False, error=str(exc), events=[], source_tier="real")

        events: List[Dict] = []
        # هر بلوک سال، سپس هر ردیف نشست داخل آن
        for ym in re.finditer(r'(\d{4})\s+FOMC Meetings(.*?)(?=\d{4}\s+FOMC Meetings|$)',
                              html, re.S):
            year = int(ym.group(1))
            block = ym.group(2)
            for row in re.finditer(
                    r'fomc-meeting__month[^>]*>\s*(?:<strong>)?\s*([A-Za-z]+)'
                    r'.*?fomc-meeting__date[^>]*>\s*([^<]+)<', block, re.S):
                mname = row.group(1).strip().lower()
                dtxt = row.group(2)
                mon = _MONTHS.get(mname)
                if not mon:
                    continue
                nums = re.findall(r"\d+", dtxt)
                if not nums:
                    continue
                day = int(nums[-1])          # روز دوم = روز اعلام نرخ
                mon_end = mon
                # حالت گذر از ماه مثل "April/May 28-1"
                if len(nums) > 1 and int(nums[-1]) < int(nums[0]):
                    mon_end = mon + 1
                    if mon_end > 12:
                        mon_end = 1
                try:
                    when = datetime(year, mon_end, day, 21, 30, tzinfo=TEHRAN)
                except ValueError:
                    continue
                projection = "Projection" in block[:row.end() + 400]
                events.append(dict(
                    key="FOMC", name="نشست فدرال رزرو (FOMC)",
                    when=when, impact="بسیار بالا",
                    note=("اعلام نرخ بهره + پیش بینی اقتصادی"
                          if projection else "اعلام نرخ بهره"),
                    source="Federal Reserve", source_tier="real"))

        # حذف تکراری ها
        seen = set()
        uniq = []
        for e in sorted(events, key=lambda x: x["when"]):
            k = e["when"].strftime("%Y-%m-%d")
            if k in seen:
                continue
            seen.add(k)
            uniq.append(e)
        events = uniq

        events.sort(key=lambda e: e["when"])
        return dict(ok=bool(events), events=events, count=len(events),
                    source="federalreserve.gov", source_tier="real")

    return _cached("fomc", _ttl, _fetch)


# ================================================================ ۲) FRED

FRED_BASE = "https://api.stlouisfed.org/fred"

# انتشارهایی که واقعا بازار را تکان می دهند
FRED_RELEASES = {
    10:  ("شاخص قیمت مصرف کننده (CPI)", "بسیار بالا", 17.0),
    50:  ("گزارش اشتغال (NFP)", "بسیار بالا", 17.0),
    53:  ("تولید ناخالص داخلی (GDP)", "بالا", 17.0),
    46:  ("شاخص قیمت تولیدکننده (PPI)", "بالا", 17.0),
    25:  ("خرده فروشی", "بالا", 17.0),
    21:  ("مدعیان بیکاری هفتگی", "متوسط", 17.0),
    54:  ("درآمد و مخارج شخصی (PCE)", "بالا", 17.0),
    97:  ("اعتماد مصرف کننده میشیگان", "متوسط", 18.5),
}


def fred_release_dates(days_ahead: int = 45, _ttl: float = 21600) -> Dict:
    """تقویم واقعی انتشار داده های اقتصادی از FRED (نیازمند کلید)."""
    key = env("FRED_KEY")
    if not key:
        return dict(ok=False, error="کلید FRED تنظیم نشده", events=[],
                    source_tier="real")

    def _fetch() -> Dict:
        today = _now_teh().date()
        end = today + timedelta(days=days_ahead)
        url = (f"{FRED_BASE}/releases/dates?api_key={key}&file_type=json"
               f"&realtime_start={today}&realtime_end={end}"
               f"&include_release_dates_with_no_data=true&sort_order=asc&limit=1000")
        try:
            data = _get_json(url, 25)
        except Exception as exc:
            return dict(ok=False, error=str(exc), events=[], source_tier="real")

        events: List[Dict] = []
        for row in data.get("release_dates", []):
            rid = row.get("release_id")
            meta = FRED_RELEASES.get(rid)
            if not meta:
                continue
            name, impact, hour = meta
            try:
                d = datetime.strptime(row["date"], "%Y-%m-%d").date()
            except Exception:
                continue
            if d < today or d > end:
                continue
            hh = int(hour)
            mm = int(round((hour - hh) * 60))
            when = datetime(d.year, d.month, d.day, hh, mm, tzinfo=TEHRAN)
            events.append(dict(
                key=f"FRED{rid}", name=name, when=when, impact=impact,
                note=row.get("release_name", ""),
                source="FRED", source_tier="real"))

        events.sort(key=lambda e: e["when"])
        return dict(ok=True, events=events, count=len(events),
                    source="FRED releases API", source_tier="real")

    return _cached(f"fredrel:{days_ahead}", _ttl, _fetch)


def fred_series(series_id: str, limit: int = 5, _ttl: float = 3600) -> Dict:
    """آخرین مقادیر یک سری FRED با کلید مستقیم (تازه تر از pandas_datareader)."""
    key = env("FRED_KEY")
    if not key:
        return dict(ok=False, error="کلید FRED تنظیم نشده", source_tier="real")

    def _fetch() -> Dict:
        url = (f"{FRED_BASE}/series/observations?series_id={series_id}"
               f"&api_key={key}&file_type=json&sort_order=desc&limit={limit}")
        try:
            obs = _get_json(url, 20).get("observations", [])
        except Exception as exc:
            return dict(ok=False, error=str(exc), source_tier="real")
        vals = []
        for o in obs:
            try:
                vals.append(dict(date=o["date"], value=float(o["value"])))
            except (ValueError, KeyError):
                continue
        if not vals:
            return dict(ok=False, error="بدون داده معتبر", source_tier="real")
        latest = vals[0]
        prev = vals[1] if len(vals) > 1 else None
        return dict(ok=True, id=series_id, value=latest["value"],
                    date=latest["date"],
                    change=(latest["value"] - prev["value"]) if prev else None,
                    history=vals, source="FRED", source_tier="real")

    return _cached(f"fred:{series_id}:{limit}", _ttl, _fetch)


# ================================================================ ۳) تقویم یکپارچه

def economic_calendar(days_ahead: int = 30) -> Dict:
    """تقویم اقتصادی کاملا واقعی — جایگزین نسخه قاعده محور قبلی."""
    now = _now_teh()
    horizon = now + timedelta(days=days_ahead)

    events: List[Dict] = []
    sources: List[str] = []
    problems: List[str] = []

    fo = fomc_calendar()
    if fo.get("ok"):
        sources.append("Federal Reserve")
        events += [e for e in fo["events"] if now <= e["when"] <= horizon]
    else:
        problems.append("FOMC: " + str(fo.get("error"))[:60])

    fr = fred_release_dates(days_ahead)
    if fr.get("ok"):
        sources.append("FRED")
        events += [e for e in fr["events"] if now <= e["when"] <= horizon]
    else:
        problems.append("FRED: " + str(fr.get("error"))[:60])

    events.sort(key=lambda e: e["when"])

    out: List[Dict] = []
    for e in events:
        delta = (e["when"] - now).total_seconds() / 60.0
        out.append(dict(
            key=e["key"], name=e["name"], impact=e["impact"],
            when=e["when"].strftime("%Y-%m-%d %H:%M"),
            when_iso=e["when"].isoformat(),
            minutes_until=round(delta),
            hours_until=round(delta / 60, 1),
            days_until=round(delta / 1440, 1),
            note=e.get("note", ""), source=e.get("source", ""),
            source_tier="real"))

    nxt = out[0] if out else None
    return dict(
        ok=bool(out), events=out, count=len(out), next_event=nxt,
        sources=sources, problems=problems,
        source_tier="real",
        note="تقویم واقعی از فدرال رزرو و FRED — هیچ تاریخ حدسی نیست",
    )


# ================================================================ ۴) VIX تاریخی

CBOE_VIX = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"


def vix_history(_ttl: float = 21600) -> Dict:
    """تاریخچه کامل VIX از ۱۹۹۰ — برای محاسبه صدک واقعی (بدون کلید)."""

    def _fetch() -> Dict:
        try:
            raw = _get(CBOE_VIX, 30).decode("utf-8", "ignore")
        except Exception as exc:
            return dict(ok=False, error=str(exc), source_tier="real")

        closes: List[float] = []
        dates: List[str] = []
        for row in csv.DictReader(io.StringIO(raw)):
            try:
                c = float(row["CLOSE"])
            except (ValueError, KeyError, TypeError):
                continue
            if c <= 0:
                continue
            closes.append(c)
            dates.append(row.get("DATE", ""))
        if not closes:
            return dict(ok=False, error="بدون داده", source_tier="real")

        return dict(ok=True, closes=closes, dates=dates, n=len(closes),
                    first=dates[0], last=dates[-1], last_close=closes[-1],
                    source="Cboe", source_tier="real")

    return _cached("vixhist", _ttl, _fetch)


def vix_percentile(value: float) -> Dict:
    """صدک واقعی VIX نسبت به کل تاریخ و بازه های اخیر."""
    h = vix_history()
    if not h.get("ok"):
        return dict(ok=False, error=h.get("error"), source_tier="real")

    closes = h["closes"]
    n = len(closes)

    def _pct(arr: List[float]) -> float:
        if not arr:
            return 0.0
        below = sum(1 for x in arr if x < value)
        return round(below / len(arr) * 100, 1)

    y1 = closes[-252:] if n >= 252 else closes
    y5 = closes[-1260:] if n >= 1260 else closes
    srt = sorted(closes)

    def _q(p: float) -> float:
        return round(srt[min(len(srt) - 1, int(len(srt) * p))], 2)

    return dict(
        ok=True, value=round(value, 2),
        percentile_all=_pct(closes),
        percentile_1y=_pct(y1),
        percentile_5y=_pct(y5),
        n_days=n, since=h["first"], until=h["last"],
        quartiles=dict(q10=_q(.10), q25=_q(.25), median=_q(.50),
                       q75=_q(.75), q90=_q(.90)),
        regime=("آرامش شدید" if value < _q(.10) else
                "آرام" if value < _q(.25) else
                "عادی" if value < _q(.75) else
                "پرتنش" if value < _q(.90) else "بحرانی"),
        source="Cboe (از ۱۹۹۰)", source_tier="real")


# ================================================================ ۵) خزانه داری

TREASURY_CSV = ("https://home.treasury.gov/resource-center/data-chart-center/"
                "interest-rates/daily-treasury-rates.csv/{year}/all"
                "?type=daily_treasury_yield_curve&field_tdr_date_value={year}"
                "&page&_format=csv")


def treasury_yields(_ttl: float = 10800) -> Dict:
    """نرخ رسمی خزانه داری آمریکا — معمولا یک روز تازه تر از FRED (بدون کلید)."""

    def _fetch() -> Dict:
        year = _now_teh().year
        try:
            raw = _get(TREASURY_CSV.format(year=year), 25).decode("utf-8", "ignore")
        except Exception as exc:
            return dict(ok=False, error=str(exc), source_tier="real")

        rows = list(csv.DictReader(io.StringIO(raw)))
        if not rows:
            return dict(ok=False, error="بدون ردیف", source_tier="real")

        cur, prev = rows[0], (rows[1] if len(rows) > 1 else None)

        def _f(row, key):
            try:
                return float(row[key])
            except (ValueError, KeyError, TypeError):
                return None

        out = dict(ok=True, date=cur.get("Date"), rows=len(rows),
                   source="U.S. Treasury", source_tier="real", tenors={})
        for label, col in [("2Y", "2 Yr"), ("5Y", "5 Yr"), ("10Y", "10 Yr"),
                           ("20Y", "20 Yr"), ("30Y", "30 Yr")]:
            v = _f(cur, col)
            if v is None:
                continue
            p = _f(prev, col) if prev else None
            out["tenors"][label] = dict(
                value=v, prev=p,
                change=round(v - p, 3) if p is not None else None,
                change_pct=round((v / p - 1) * 100, 2) if p else None)

        t2 = out["tenors"].get("2Y", {}).get("value")
        t10 = out["tenors"].get("10Y", {}).get("value")
        if t2 is not None and t10 is not None:
            sp = round(t10 - t2, 3)
            out["curve"] = dict(
                spread_10y2y=sp,
                state=("معکوس — هشدار رکود" if sp < 0 else
                       "صاف" if sp < 0.2 else
                       "نرمال باریک" if sp < 0.8 else "نرمال"))
        return out

    return _cached("treasury", _ttl, _fetch)


# ================================================================ ۶) Put/Call واقعی

def put_call_real(symbol: str = "DIA", max_exp: int = 4,
                  _ttl: float = 900) -> Dict:
    """نسبت واقعی Put/Call از زنجیره آپشن — جایگزین فرمول ساختگی قبلی.

    فرمول قبلی این بود:  pc = 0.75 + (VIX - 18) / 45
    که هیچ ارتباطی با داده واقعی آپشن نداشت.
    """

    def _fetch() -> Dict:
        try:
            import yfinance as yf
        except Exception as exc:
            return dict(ok=False, error=str(exc), source_tier="computed")

        try:
            tk = yf.Ticker(symbol)
            exps = list(tk.options or [])[:max_exp]
            if not exps:
                return dict(ok=False, error="زنجیره آپشن خالی",
                            source_tier="computed")

            pv = cv = 0.0
            poi = coi = 0.0
            used: List[str] = []
            for e in exps:
                try:
                    ch = tk.option_chain(e)
                except Exception:
                    continue
                calls, puts = ch.calls, ch.puts
                if calls is None or puts is None:
                    continue
                cv += float(calls["volume"].fillna(0).sum())
                pv += float(puts["volume"].fillna(0).sum())
                coi += float(calls["openInterest"].fillna(0).sum())
                poi += float(puts["openInterest"].fillna(0).sum())
                used.append(e)

            if cv <= 0 and coi <= 0:
                return dict(ok=False, error="حجم و OI صفر",
                            source_tier="computed")

            pcv = round(pv / cv, 4) if cv > 0 else None
            pco = round(poi / coi, 4) if coi > 0 else None
            ref = pcv if pcv is not None else pco

            def _state(r: Optional[float]) -> str:
                if r is None:
                    return "نامشخص"
                if r > 1.20:
                    return "ترس شدید — پوت بسیار زیاد (احتمال کف)"
                if r > 1.00:
                    return "محتاطانه — پوت بیشتر از کال"
                if r > 0.70:
                    return "متعادل"
                if r > 0.55:
                    return "خوش بینانه — کال بیشتر"
                return "طمع شدید — کال بسیار زیاد (احتمال سقف)"

            return dict(
                ok=True, symbol=symbol,
                put_call_volume=pcv, put_call_oi=pco,
                put_volume=int(pv), call_volume=int(cv),
                put_oi=int(poi), call_oi=int(coi),
                expirations=used, state=_state(ref),
                contrarian=("صعودی" if (ref or 0) > 1.15 else
                            "نزولی" if 0 < (ref or 1) < 0.60 else "خنثی"),
                source="Yahoo option chain",
                source_tier="computed",
                note="محاسبه مستقیم از حجم و Open Interest واقعی آپشن")
        except Exception as exc:
            return dict(ok=False, error=str(exc), source_tier="computed")

    return _cached(f"pcr:{symbol}", _ttl, _fetch)


# ================================================================ ۷) اخبار Finnhub

FINNHUB = "https://finnhub.io/api/v1"


def finnhub_news(category: str = "general", limit: int = 60,
                 _ttl: float = 600) -> Dict:
    """اخبار بلادرنگ بازار — جایگزین ۸ خبر ضعیف قبلی."""
    key = env("FINNHUB_KEY")
    if not key:
        return dict(ok=False, error="کلید Finnhub تنظیم نشده", items=[],
                    source_tier="real")

    def _fetch() -> Dict:
        try:
            data = _get_json(f"{FINNHUB}/news?category={category}&token={key}", 20)
        except Exception as exc:
            return dict(ok=False, error=str(exc), items=[], source_tier="real")

        items = []
        for n in data[:limit]:
            ts = n.get("datetime") or 0
            items.append(dict(
                title=n.get("headline", ""),
                summary=(n.get("summary") or "")[:400],
                source=n.get("source", ""),
                url=n.get("url", ""),
                ts=ts,
                time=(datetime.fromtimestamp(ts, TEHRAN).strftime("%m-%d %H:%M")
                      if ts else ""),
                age_min=(round((time.time() - ts) / 60) if ts else None)))
        return dict(ok=True, items=items, count=len(items),
                    source="Finnhub", source_tier="real")

    return _cached(f"fnews:{category}:{limit}", _ttl, _fetch)


def finnhub_quote(symbol: str = "DIA", _ttl: float = 20) -> Dict:
    """قیمت لحظه ای از Finnhub — منبع دوم برای راستی آزمایی."""
    key = env("FINNHUB_KEY")
    if not key:
        return dict(ok=False, error="کلید تنظیم نشده", source_tier="real")

    def _fetch() -> Dict:
        try:
            d = _get_json(f"{FINNHUB}/quote?symbol={symbol}&token={key}", 15)
        except Exception as exc:
            return dict(ok=False, error=str(exc), source_tier="real")
        if not d.get("c"):
            return dict(ok=False, error="قیمت خالی", source_tier="real")
        return dict(ok=True, symbol=symbol, price=d["c"],
                    change=d.get("d"), change_pct=d.get("dp"),
                    high=d.get("h"), low=d.get("l"), open=d.get("o"),
                    prev_close=d.get("pc"), ts=d.get("t"),
                    source="Finnhub", source_tier="real")

    return _cached(f"fq:{symbol}", _ttl, _fetch)


def finnhub_earnings(days_ahead: int = 30, _ttl: float = 21600) -> Dict:
    """تقویم درآمدی واقعی شرکت ها."""
    key = env("FINNHUB_KEY")
    if not key:
        return dict(ok=False, error="کلید تنظیم نشده", items=[],
                    source_tier="real")

    DOW = {"AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX",
           "DIS", "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM",
           "MRK", "MSFT", "NKE", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT"}

    def _fetch() -> Dict:
        today = _now_teh().date()
        end = today + timedelta(days=days_ahead)
        url = (f"{FINNHUB}/calendar/earnings?from={today}&to={end}&token={key}")
        try:
            data = _get_json(url, 25).get("earningsCalendar", [])
        except Exception as exc:
            return dict(ok=False, error=str(exc), items=[], source_tier="real")

        rows = []
        for e in data:
            sym = e.get("symbol", "")
            if sym not in DOW:
                continue
            rows.append(dict(
                symbol=sym, date=e.get("date"),
                hour=e.get("hour", ""),
                eps_estimate=e.get("epsEstimate"),
                eps_actual=e.get("epsActual"),
                revenue_estimate=e.get("revenueEstimate"),
                is_dow=True))
        rows.sort(key=lambda r: (r["date"] or ""))
        return dict(ok=True, items=rows, count=len(rows),
                    total_scanned=len(data),
                    source="Finnhub", source_tier="real")

    return _cached(f"fearn:{days_ahead}", _ttl, _fetch)


# ================================================================ ۸) احساسات AV

AV = "https://www.alphavantage.co/query"
# سقف رایگان ۲۵ درخواست در روز -> کش طولانی الزامی است
_AV_TTL = 21600


def av_sentiment(tickers: str = "AAPL,MSFT,JPM", limit: int = 50) -> Dict:
    """امتیاز عددی احساسات خبری — جایگزین واژه نامه ۶۰ کلمه ای قبلی.

    توجه: سقف رایگان ۲۵ درخواست در روز است، پس ۶ ساعت کش می شود.
    نمادهای شاخصی مثل DIA و SPY پاسخ خالی می دهند؛ فقط سهام تکی کار می کند.
    """
    key = env("ALPHAVANTAGE_KEY")
    if not key:
        return dict(ok=False, error="کلید Alpha Vantage تنظیم نشده",
                    items=[], source_tier="real")

    def _fetch() -> Dict:
        url = (f"{AV}?function=NEWS_SENTIMENT&tickers={urllib.parse.quote(tickers)}"
               f"&limit={limit}&apikey={key}")
        try:
            d = _get_json(url, 25)
        except Exception as exc:
            return dict(ok=False, error=str(exc), items=[], source_tier="real")

        if "feed" not in d:
            msg = d.get("Information") or d.get("Note") or "پاسخ نامعتبر"
            return dict(ok=False, error=str(msg)[:160], items=[],
                        rate_limited=True, source_tier="real")

        feed = d.get("feed", [])
        items, scores = [], []
        for it in feed:
            try:
                sc = float(it.get("overall_sentiment_score", 0))
            except (TypeError, ValueError):
                continue
            scores.append(sc)
            items.append(dict(
                title=it.get("title", "")[:160],
                score=round(sc, 4),
                label=it.get("overall_sentiment_label", ""),
                source=it.get("source", ""),
                time=it.get("time_published", "")[:13]))

        if not scores:
            return dict(ok=False, error="بدون امتیاز", items=[],
                        source_tier="real")

        avg = sum(scores) / len(scores)
        bull = sum(1 for s in scores if s >= 0.15)
        bear = sum(1 for s in scores if s <= -0.15)
        return dict(
            ok=True, items=items[:25], count=len(items),
            avg_score=round(avg, 4),
            bullish=bull, bearish=bear,
            neutral=len(scores) - bull - bear,
            label=("صعودی" if avg >= 0.15 else
                   "نزولی" if avg <= -0.15 else "خنثی"),
            source="Alpha Vantage", source_tier="real",
            note="امتیاز NLP حرفه ای — سقف ۲۵ درخواست در روز")

    return _cached(f"av:{tickers}:{limit}", _AV_TTL, _fetch)


# ================================================================ تجمیع

def build_real_data(symbol: str = "DIA", asset: Optional[str] = None) -> Dict:
    """همه منابع واقعی در یک فراخوانی."""
    out: Dict = dict(ok=True, generated=_now_teh().strftime("%Y-%m-%d %H:%M:%S"))

    # نماد آپشن بر اساس پروفایل دارایی (طلا → GLD)
    _osym = symbol
    _prof = None
    try:
        import assets as _A
        if asset:
            _prof = _A.profile(asset)
            _osym = _prof["options_symbol"]
            out["asset"] = _prof["key"]
            out["asset_name"] = _prof["name"]
    except Exception:
        pass

    out["calendar"] = economic_calendar(30)
    out["treasury"] = treasury_yields()
    out["put_call"] = put_call_real(_osym)
    if _prof is not None and _osym != _prof["candle_symbol"]:
        try:
            out["put_call"]["proxy_symbol"] = _osym
            out["put_call"]["proxy_note"] = (
                f"آپشن طلا از صندوق {_osym} خوانده شد "
                f"(فیوچرز GC=F زنجیره آپشن ندارد).")
        except Exception:
            pass

    # قیمت اسپات زنده برای دارایی هایی که فید اسپات دارند
    if _prof is not None and _prof.get("spot_source"):
        try:
            import assets as _A2
            sp = _A2.live_spot(_prof["key"])
            bs = _A2.basis(_prof["key"])
            out["spot"] = dict(sp, basis=bs)
        except Exception as _e:
            out["spot"] = dict(ok=False, error=str(_e))
    out["news"] = finnhub_news(limit=40)
    out["earnings"] = finnhub_earnings(45)

    tr = out["treasury"]
    if tr.get("ok"):
        out["fred_fresh"] = dict(
            dgs10=fred_series("DGS10"), dgs2=fred_series("DGS2"),
            dff=fred_series("DFF"))

    # صدک VIX از سطح جاری
    try:
        import yfinance as yf
        v = yf.Ticker("^VIX").history(period="5d", interval="1d")["Close"].dropna()
        if len(v):
            out["vix"] = vix_percentile(float(v.iloc[-1]))
    except Exception as exc:
        out["vix"] = dict(ok=False, error=str(exc)[:80], source_tier="real")

    tiers = {"real": 0, "computed": 0, "estimated": 0}
    for k, v in out.items():
        if isinstance(v, dict) and v.get("source_tier"):
            tiers[v["source_tier"]] = tiers.get(v["source_tier"], 0) + 1
    out["transparency"] = tiers
    return out


# ================================================================ خودآزمون

if __name__ == "__main__":
    import sys

    def hr(t):
        print("\n" + "=" * 64)
        print(" " + t)
        print("=" * 64)

    hr("۱) تقویم FOMC (بدون کلید)")
    f = fomc_calendar()
    print(f"   ok={f.get('ok')}  تعداد={f.get('count')}")
    for e in f.get("events", [])[:4]:
        print(f"     {e['when'].strftime('%Y-%m-%d %H:%M')}  {e['name']}")

    hr("۲) تقویم انتشار FRED")
    r = fred_release_dates(45)
    print(f"   ok={r.get('ok')}  تعداد={r.get('count')}  {r.get('error','')}")
    for e in r.get("events", [])[:6]:
        print(f"     {e['when'].strftime('%Y-%m-%d %H:%M')}  {e['name']}  [{e['impact']}]")

    hr("۳) تقویم یکپارچه واقعی")
    c = economic_calendar(30)
    print(f"   ok={c.get('ok')}  رویداد={c.get('count')}  منابع={c.get('sources')}")
    if c.get("next_event"):
        n = c["next_event"]
        print(f"   بعدی: {n['name']} در {n['days_until']} روز ({n['when']})")
    for e in c.get("events", [])[:8]:
        print(f"     {e['when']}  {e['name'][:38]:<38} [{e['impact']}]")

    hr("۴) صدک VIX (Cboe از ۱۹۹۰)")
    h = vix_history()
    print(f"   ردیف={h.get('n')}  از {h.get('first')} تا {h.get('last')}")
    if h.get("ok"):
        p = vix_percentile(h["last_close"])
        print(f"   VIX={p['value']}  صدک کل={p['percentile_all']}%  "
              f"یک ساله={p['percentile_1y']}%  رژیم={p['regime']}")
        print(f"   چارک ها: {p['quartiles']}")

    hr("۵) نرخ خزانه داری رسمی")
    t = treasury_yields()
    print(f"   ok={t.get('ok')}  تاریخ={t.get('date')}")
    for k, v in (t.get("tenors") or {}).items():
        print(f"     {k:<4} {v['value']}%  تغییر {v.get('change')}")
    print(f"   منحنی: {t.get('curve')}")

    hr("۶) Put/Call واقعی (جایگزین فرمول ساختگی)")
    pc = put_call_real("DIA")
    print(f"   ok={pc.get('ok')}  حجمی={pc.get('put_call_volume')}  "
          f"OI={pc.get('put_call_oi')}")
    print(f"   وضعیت: {pc.get('state')}  | معکوس: {pc.get('contrarian')}")
    print(f"   پوت={pc.get('put_volume')} کال={pc.get('call_volume')}")

    hr("۷) اخبار Finnhub")
    n = finnhub_news(limit=40)
    print(f"   ok={n.get('ok')}  تعداد={n.get('count')}")
    for i in n.get("items", [])[:4]:
        print(f"     [{i['time']}] {i['title'][:56]}")

    hr("۸) تقویم درآمدی داو")
    e = finnhub_earnings(45)
    print(f"   ok={e.get('ok')}  شرکت داو={e.get('count')}  "
          f"از {e.get('total_scanned')} کل")
    for i in e.get("items", [])[:5]:
        print(f"     {i['date']}  {i['symbol']:<6} تخمین EPS={i.get('eps_estimate')}")

    if "--av" in sys.argv:
        hr("۹) احساسات Alpha Vantage (مصرف سهمیه)")
        a = av_sentiment()
        print(f"   ok={a.get('ok')}  تعداد={a.get('count')}  "
              f"میانگین={a.get('avg_score')}  {a.get('label','')}")
        print(f"   صعودی={a.get('bullish')} نزولی={a.get('bearish')} "
              f"خنثی={a.get('neutral')}")
        for i in a.get("items", [])[:3]:
            print(f"     {i['score']:+.3f} {i['label']:<14} {i['title'][:44]}")
        if not a.get("ok"):
            print("   ", a.get("error"))
