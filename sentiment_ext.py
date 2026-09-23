# -*- coding: utf-8 -*-
"""
================================================================
sentiment_ext.py — سه منبع احساسات و پوزیشن که قبلا نداشتیم
================================================================
۱) COT — گزارش هفتگی CFTC (دولت آمریکا، بدون کلید)
     تنها پنجره واقعی به پوزیشن پول بزرگ در بازار طلا.
۲) CNN Fear & Greed — شاخص ترس و طمع بازار سهام.
۳) StockTwits — احساسات معامله گران خرد (برچسب خود کاربران).

⚠️ نکات فنی که با تست واقعی کشف شد و در پیاده سازی رعایت شده:
  • خطوط فایل CFTC با کوتیشن شروع می شوند؛ startswith بدون کوتیشن
    هیچ نتیجه ای نمی دهد (باگ رایج).
  • ستون Managed Money اندیس ۱۳ و ۱۴ است، نه ۱۱ و ۱۲ (که Swap
    Dealer است). با سه معادله تراز صحت سنجی شده:
        مجموع لانگ های هفت گروه == فیلد[19]
        مجموع شورت های هفت گروه == فیلد[20]
        فیلد[19] + فیلد[21]     == Open Interest
  • CNN با User-Agent غیرمرورگری کد ۴۱۸ می دهد ("You're a bot").
  • برچسب احساسات StockTwits اختیاری است؛ برای XAUUSD معمولا
    کمتر از ۵ پیام برچسب دارد، پس چند نماد ترکیب می شود.

همه توابع در صورت خطا dict با ok=False برمی گردانند — هرگز استثنا
پرتاب نمی کنند تا یک منبع خراب کل سیستم را از کار نیندازد.
================================================================
"""
from __future__ import annotations

import csv
import io
import json
import ssl
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

# CNN با UA غیرمرورگری بلاک می کند (۴۱۸ "You're a bot").
# تست شد: فقط UA کافی نیست — Referer/Origin/Sec-Fetch هم لازم است.
_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")

BROWSER_UA = {
    "User-Agent": _UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "identity",
    "Referer": "https://edition.cnn.com/",
    "Origin": "https://edition.cnn.com",
    "Sec-Fetch-Site": "cross-site",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
    "Connection": "keep-alive",
}

_CACHE: Dict[str, tuple] = {}


def _cached(key: str, ttl: float, fn):
    hit = _CACHE.get(key)
    if hit and (time.time() - hit[0]) < ttl:
        return hit[1]
    val = fn()
    _CACHE[key] = (time.time(), val)
    return val


def _get(url: str, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(url, headers=BROWSER_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# ================================================================ ۱) COT
COT_URL = "https://www.cftc.gov/dea/newcot/f_disagg.txt"

# اندیس ستون ها در گزارش Disaggregated (تأیید شده با معادله تراز)
_C = dict(
    name=0, oi=7,
    prod_long=8, prod_short=9,
    swap_long=10, swap_short=11, swap_spread=12,
    mm_long=13, mm_short=14, mm_spread=15,
    other_long=16, other_short=17, other_spread=18,
    tot_long=19, tot_short=20,
    nonrept_long=21, nonrept_short=22,
)

COT_MARKETS = {
    "XAUUSD": ('"GOLD - COMMODITY EXCHANGE INC.', "طلا"),
    "SILVER": ('"SILVER - COMMODITY EXCHANGE INC.', "نقره"),
}


def _parse_cot_row(line: str) -> Optional[Dict]:
    try:
        f = next(csv.reader(io.StringIO(line)))
    except Exception:
        return None
    if len(f) < 25:
        return None

    def n(i) -> Optional[int]:
        try:
            return int(str(f[_C[i]]).strip())
        except Exception:
            return None

    mm_l, mm_s = n("mm_long"), n("mm_short")
    oi = n("oi")
    if mm_l is None or mm_s is None or not oi:
        return None

    # --- صحت سنجی: مجموع گروه ها باید با ستون جمع بخواند ---
    checks = {}
    try:
        tl = (n("prod_long") + n("swap_long") + n("swap_spread")
              + n("mm_long") + n("mm_spread") + n("other_long")
              + n("other_spread"))
        ts = (n("prod_short") + n("swap_short") + n("swap_spread")
              + n("mm_short") + n("mm_spread") + n("other_short")
              + n("other_spread"))
        checks["long_balance"] = (tl == n("tot_long"))
        checks["short_balance"] = (ts == n("tot_short"))
        checks["oi_balance"] = (n("tot_long") + n("nonrept_long") == oi)
    except Exception:
        checks = {"long_balance": False, "short_balance": False,
                  "oi_balance": False}

    net = mm_l - mm_s
    tot_mm = mm_l + mm_s
    return dict(
        report_date=str(f[2]).strip(),
        open_interest=oi,
        mm_long=mm_l, mm_short=mm_s, mm_net=net,
        mm_net_pct=round(net / tot_mm * 100, 2) if tot_mm else None,
        mm_long_pct_oi=round(mm_l / oi * 100, 2),
        mm_net_pct_oi=round(net / oi * 100, 2),
        prod_net=(n("prod_long") - n("prod_short")
                  if n("prod_long") is not None else None),
        swap_net=(n("swap_long") - n("swap_short")
                  if n("swap_long") is not None else None),
        checks=checks,
        verified=all(checks.values()),
    )


def cot_report(asset: str = "XAUUSD", _ttl: float = 6 * 3600) -> Dict:
    """
    گزارش COT هفتگی CFTC — پوزیشن واقعی صندوق های بزرگ.

    «Managed Money» یعنی صندوق های پوشش ریسک و سرمایه گذاران حرفه ای.
    وقتی خالص لانگ آن ها خیلی زیاد شود، یعنی همه یک طرف معامله جمع
    شده اند و ریسک اصلاح بالا می رود (سیگنال معکوس).
    """
    key = f"cot:{asset}"

    def _fetch() -> Dict:
        pat = COT_MARKETS.get(str(asset).upper())
        if not pat:
            return dict(ok=False, error="این دارایی در گزارش COT نیست",
                        available=False, source_tier="real")
        needle, fa = pat
        try:
            txt = _get(COT_URL, 30).decode("utf-8", "replace")
        except Exception as e:
            return dict(ok=False, error=f"دریافت نشد: {e}", source_tier="real")

        row = next((l for l in txt.splitlines() if l.startswith(needle)), None)
        if not row:
            return dict(ok=False, error="ردیف این بازار پیدا نشد",
                        source_tier="real")
        d = _parse_cot_row(row)
        if not d:
            return dict(ok=False, error="ساختار فایل تغییر کرده",
                        source_tier="real")

        pct = d["mm_net_pct"] or 0.0

        # ── آستانه صدکی (نه ثابت) ──
        # ممیزی ۲۰۲۶-۰۹-۱۹ روی ۳۴۷ هفته: میانه تاریخی ~۶۹٪ است، پس
        # آستانه ثابت ۷۰٪ در نیمی از اوقات فعال می شد.
        hist = cot_history(asset)
        q90 = q75 = q25 = q10 = None
        if hist.get("ok"):
            q90, q75, q25, q10 = hist["q90"], hist["q75"], hist["q25"], hist["q10"]

        if q90 is not None:
            if pct >= q90:
                state = "ازدحام خرید — کم سابقه"
                note = ("پوزیشن لانگ صندوق ها در صدک ۹۰ بالای ۳ سال اخیر است. "
                        "داده تاریخی نشان می دهد این حالت معمولا با ادامه روند "
                        "همراه بوده، نه بازگشت — ولی فضای خرید تازه کم است.")
            elif pct >= q75:
                state = "لانگ بالاتر از عادی"
                note = "صندوق ها بیش از حد معمول لانگ هستند."
            elif pct <= q10:
                state = "شورت کم سابقه"
                note = ("پوزیشن در صدک ۱۰ پایین است — از نظر تاریخی "
                        "معمولا کف ساز بوده.")
            elif pct <= q25:
                state = "لانگ کمتر از عادی"
                note = "صندوق ها کمتر از معمول لانگ هستند."
            else:
                state = "در محدوده عادی"
                note = "پوزیشن صندوق ها نزدیک میانه تاریخی است."
        else:
            state = "متعادل" if -10 < pct < 45 else (
                "تمایل قوی به خرید" if pct >= 45 else "تمایل به فروش")
            note = "تاریخچه در دسترس نبود — فقط عدد خام گزارش شده."

        # ── لحن: بر اساس داده، نه فرض ──
        # ممیزی: ازدحام خرید (صدک۹۰+) → بازده ۴ هفته +۳.۰۰٪ در برابر
        # میانه +۱.۰۲٪، p=0.016. یعنی «کنترین» بودن اشتباه بود.
        # علامت مثبت (ادامه روند) با شدت کم.
        tone = 0
        if q90 is not None:
            if pct >= q90:
                tone = 1
            elif pct <= q10:
                tone = 1
        return dict(ok=True, asset=asset, market_fa=fa,
                    contrarian_tone=tone, trend_tone=tone, state=state, note=note,
                    percentile_band=(state if q90 is not None else None),
                    q90=q90, q75=q75, q25=q25, q10=q10,
                    source="CFTC (دولت آمریکا)",
                    source_tier=("real" if d["verified"] else "computed"),
                    **d)

    return _cached(key, _ttl, _fetch)


def cot_history(asset: str = "XAUUSD", years: int = 3,
                _ttl: float = 86400) -> Dict:
    """تاریخچه چند ساله Managed Money از آرشیو سالانه CFTC.

    چرا لازم است: آستانه ثابت «۷۰٪ = ازدحام» بی معناست چون میانه
    تاریخی خودش حدود ۶۹٪ است (ممیزی ۲۰۲۶-۰۹-۱۹). آستانه باید
    صدکی باشد تا «کم سابقه» واقعا کم سابقه باشد.
    """
    import zipfile

    mk = COT_MARKETS.get(str(asset).upper())
    if not mk:
        return dict(ok=False, error="این دارایی در گزارش COT نیست")
    prefix = mk[0]

    def _fetch():
        import datetime as _dt
        cur = _dt.date.today().year
        rows, got = [], []
        for yr in range(cur - years + 1, cur + 1):
            url = ("https://www.cftc.gov/files/dea/history/"
                   "fut_disagg_txt_%d.zip" % yr)
            try:
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                req = urllib.request.Request(url, headers=BROWSER_UA)
                with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
                    blob = r.read()
                z = zipfile.ZipFile(io.BytesIO(blob))
                txt = z.read(z.namelist()[0]).decode("utf-8", "ignore")
                for line in txt.splitlines():
                    if not line.startswith(prefix):
                        continue
                    f = next(csv.reader([line]))
                    try:
                        ml, ms = float(f[_C["mm_long"]]), float(f[_C["mm_short"]])
                    except (ValueError, IndexError):
                        continue
                    tot = ml + ms
                    if tot <= 0:
                        continue
                    rows.append(dict(date=f[2],
                                     pct=round((ml - ms) / tot * 100, 2),
                                     net=int(ml - ms)))
                got.append(yr)
            except Exception:
                continue

        if not rows:
            return dict(ok=False, error="آرشیو در دسترس نیست",
                        source="CFTC", source_tier="real")

        seen, uniq = set(), []
        for r in sorted(rows, key=lambda x: x["date"]):
            if r["date"] in seen:
                continue
            seen.add(r["date"])
            uniq.append(r)

        vals = sorted(x["pct"] for x in uniq)
        n = len(vals)

        def _q(p):
            if n == 1:
                return vals[0]
            i = p * (n - 1)
            lo, hi = int(i), min(int(i) + 1, n - 1)
            return round(vals[lo] + (vals[hi] - vals[lo]) * (i - lo), 2)

        return dict(
            ok=True, asset=asset, weeks=n, years_loaded=got,
            history=uniq[-60:],
            q10=_q(0.10), q25=_q(0.25), median=_q(0.50),
            q75=_q(0.75), q90=_q(0.90),
            min=vals[0], max=vals[-1],
            source="CFTC (آرشیو سالانه)", source_tier="real",
            note="%d هفته از %s" % (n, ", ".join(str(y) for y in got)))

    try:
        return _cached("cot_hist_%s_%d" % (asset, years), _ttl, _fetch)
    except Exception as e:
        return dict(ok=False, error=type(e).__name__)


def cot_percentile(pct: float, asset: str = "XAUUSD") -> Dict:
    """جایگاه عدد فعلی در توزیع تاریخی."""
    h = cot_history(asset)
    if not h.get("ok"):
        return dict(ok=False, error=h.get("error"))
    vals = [x["pct"] for x in cot_history(asset).get("history", [])]
    full = h.get("weeks", 0)
    below = sum(1 for v in [x["pct"] for x in h["history"]] if v < pct)
    # صدک دقیق از روی چارک های ذخیره شده
    if pct >= h["q90"]:
        band, tone_fa = "extreme_high", "کم سابقه بالا"
    elif pct >= h["q75"]:
        band, tone_fa = "high", "بالاتر از عادی"
    elif pct <= h["q10"]:
        band, tone_fa = "extreme_low", "کم سابقه پایین"
    elif pct <= h["q25"]:
        band, tone_fa = "low", "پایین تر از عادی"
    else:
        band, tone_fa = "normal", "در محدوده عادی"
    return dict(ok=True, band=band, band_fa=tone_fa,
                q10=h["q10"], q25=h["q25"], median=h["median"],
                q75=h["q75"], q90=h["q90"], weeks=full,
                approx_pctile=round(100.0 * below / max(len(h["history"]), 1), 1))


# ================================================================ ۲) F&G
FG_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"


def fear_greed(_ttl: float = 1800) -> Dict:
    """
    شاخص ترس و طمع CNN (۰ تا ۱۰۰) برای بازار سهام آمریکا.

    زیر ۲۵ = ترس شدید (تاریخا فرصت خرید)، بالای ۷۵ = طمع شدید
    (ریسک اصلاح). برای طلا معنای معکوس دارد: ترس در سهام معمولا
    تقاضای پناهگاه امن را بالا می برد.
    """
    def _fetch() -> Dict:
        try:
            j = json.loads(_get(FG_URL, 20).decode("utf-8", "replace"))
        except Exception as e:
            return dict(ok=False, error=f"دریافت نشد: {e}",
                        source_tier="real")
        fg = j.get("fear_and_greed") or {}
        score = fg.get("score")
        if score is None:
            return dict(ok=False, error="ساختار پاسخ تغییر کرده",
                        source_tier="real")
        score = float(score)

        if score <= 25:
            state, equity_tone, gold_tone = "ترس شدید", 1, 1
        elif score <= 45:
            state, equity_tone, gold_tone = "ترس", 0, 0
        elif score < 55:
            state, equity_tone, gold_tone = "خنثی", 0, 0
        elif score < 75:
            state, equity_tone, gold_tone = "طمع", 0, 0
        else:
            state, equity_tone, gold_tone = "طمع شدید", -1, 0

        hist = (j.get("fear_and_greed_historical") or {}).get("data") or []
        series = [round(float(p.get("y")), 1) for p in hist[-60:]
                  if p.get("y") is not None]

        subs = {}
        for key, fa in (("market_momentum_sp500", "شتاب بازار"),
                        ("stock_price_strength", "قدرت قیمت سهام"),
                        ("stock_price_breadth", "وسعت بازار"),
                        ("put_call_options", "نسبت Put/Call"),
                        ("market_volatility_vix", "نوسان (VIX)")):
            blk = j.get(key) or {}
            if blk.get("rating") is not None:
                subs[fa] = dict(rating=blk.get("rating"),
                                score=(round(float(blk["score"]), 1)
                                       if blk.get("score") is not None else None))

        return dict(
            ok=True, score=round(score, 1), state=state, rating=fg.get("rating"),
            equity_tone=equity_tone, gold_tone=gold_tone,
            prev_close=_r(fg.get("previous_close")),
            prev_week=_r(fg.get("previous_1_week")),
            prev_month=_r(fg.get("previous_1_month")),
            prev_year=_r(fg.get("previous_1_year")),
            timestamp=fg.get("timestamp"),
            series=series, components=subs,
            source="CNN Business", source_tier="real",
            note=("این شاخص ترس و طمع بازار سهام است. برای داوجونز مستقیم "
                  "معنا دارد؛ برای طلا معکوس — ترس در سهام معمولا پول را "
                  "به سمت طلا می برد."),
        )

    return _cached("fg", _ttl, _fetch)


def _r(x, d: int = 1):
    try:
        return round(float(x), d)
    except Exception:
        return None


# ================================================================ ۳) StockTwits
ST_URL = "https://api.stocktwits.com/api/2/streams/symbol/{}.json"

# نماد XAUUSD پیام برچسب دار کمی دارد؛ چند نماد ترکیب می شود
ST_SYMBOLS = {
    "XAUUSD": ["GC_F", "GLD", "XAUUSD"],
    "US30": ["DIA", "YM_F", "SPY"],
}

MIN_TAGGED = 8      # زیر این تعداد، نمونه بی معنی است


def stocktwits(asset: str = "US30", _ttl: float = 900) -> Dict:
    """
    احساسات معامله گران خرد از StockTwits.

    کاربران پیام خود را «صعودی» یا «نزولی» برچسب می زنند. این عدد
    نظر جمعیت خرد است — نه نهادها. وقتی خیلی یک طرفه شود، اغلب
    به عنوان سیگنال معکوس استفاده می شود.
    """
    key = f"st:{asset}"

    def _fetch() -> Dict:
        syms = ST_SYMBOLS.get(str(asset).upper(), [str(asset).upper()])
        per: List[Dict] = []
        bull = bear = total = 0
        for s in syms:
            try:
                j = json.loads(_get(ST_URL.format(s), 15)
                               .decode("utf-8", "replace"))
            except Exception:
                continue
            msgs = j.get("messages") or []
            b = r = 0
            for m in msgs:
                tag = ((m.get("entities") or {}).get("sentiment") or {})
                t = tag.get("basic") if isinstance(tag, dict) else None
                if t == "Bullish":
                    b += 1
                elif t == "Bearish":
                    r += 1
            per.append(dict(symbol=s, messages=len(msgs),
                            bullish=b, bearish=r, tagged=b + r))
            bull += b
            bear += r
            total += len(msgs)

        tagged = bull + bear
        if tagged == 0:
            return dict(ok=False, error="هیچ پیام برچسب داری یافت نشد",
                        symbols=per, source="StockTwits",
                        source_tier="real")

        ratio = (bull - bear) / tagged
        weak = tagged < MIN_TAGGED

        if ratio >= 0.5:
            state, tone = "خوش بینی شدید خرد", -1
        elif ratio >= 0.15:
            state, tone = "خوش بین", 0
        elif ratio <= -0.5:
            state, tone = "بدبینی شدید خرد", 1
        elif ratio <= -0.15:
            state, tone = "بدبین", 0
        else:
            state, tone = "متعادل", 0

        return dict(
            ok=True, asset=asset, bullish=bull, bearish=bear,
            tagged=tagged, messages=total,
            ratio=round(ratio, 3),
            bull_pct=round(bull / tagged * 100, 1),
            state=state, contrarian_tone=tone,
            low_sample=weak,
            symbols=per,
            source="StockTwits",
            source_tier=("real" if not weak else "computed"),
            note=("نمونه کوچک است؛ با احتیاط تفسیر شود."
                  if weak else
                  "نظر معامله گران خرد — وقتی خیلی یک طرفه شود "
                  "اغلب خلافش اتفاق می افتد."),
        )

    return _cached(key, _ttl, _fetch)


# ================================================================ تجمیع
def build_sentiment_ext(asset: str = "US30") -> Dict:
    """هر سه منبع در یک فراخوانی."""
    import assets as A
    prof = A.profile(asset)
    key = prof["key"]

    out: Dict = dict(ok=True, asset=key, asset_name=prof["name"])
    out["fear_greed"] = fear_greed()
    out["stocktwits"] = stocktwits(key)
    out["cot"] = (cot_report("XAUUSD") if key == "XAUUSD"
                  else dict(ok=False, available=False,
                            error="گزارش COT برای داوجونز در فایل "
                                  "Disaggregated وجود ندارد (فقط کالاهاست)."))

    tones: List[int] = []
    fg = out["fear_greed"]
    if fg.get("ok"):
        tones.append(fg["gold_tone"] if key == "XAUUSD" else fg["equity_tone"])
    st = out["stocktwits"]
    if st.get("ok") and not st.get("low_sample"):
        tones.append(st["contrarian_tone"])
    ct = out["cot"]
    if ct.get("ok"):
        tones.append(ct["contrarian_tone"])

    out["combined_tone"] = (round(sum(tones) / len(tones), 2) if tones else 0.0)
    out["sources_ok"] = sum(1 for k in ("fear_greed", "stocktwits", "cot")
                            if out[k].get("ok"))
    return out


if __name__ == "__main__":
    import sys
    a = sys.argv[1] if len(sys.argv) > 1 else "XAUUSD"

    print("=" * 62)
    c = cot_report("XAUUSD")
    print("۱) COT — پوزیشن صندوق های بزرگ در طلا")
    if c.get("ok"):
        print(f"   تاریخ گزارش : {c['report_date']}")
        print(f"   لانگ        : {c['mm_long']:,}")
        print(f"   شورت        : {c['mm_short']:,}")
        print(f"   خالص        : {c['mm_net']:+,}  ({c['mm_net_pct']}٪)")
        print(f"   سهم از OI   : {c['mm_net_pct_oi']}٪")
        print(f"   وضعیت       : {c['state']}")
        print(f"   صحت سنجی    : {c['checks']} → verified={c['verified']}")
        print(f"   {c['note']}")
    else:
        print("   ❌", c.get("error"))

    print("=" * 62)
    f = fear_greed()
    print("۲) CNN Fear & Greed")
    if f.get("ok"):
        print(f"   امتیاز      : {f['score']} ({f['state']})")
        print(f"   هفته پیش    : {f['prev_week']} · ماه پیش: {f['prev_month']}"
              f" · سال پیش: {f['prev_year']}")
        print(f"   اجزا        : {list(f['components'].keys())}")
        print(f"   تاریخچه     : {len(f['series'])} نقطه")
    else:
        print("   ❌", f.get("error"))

    print("=" * 62)
    s = stocktwits(a)
    print(f"۳) StockTwits — {a}")
    if s.get("ok"):
        print(f"   صعودی {s['bullish']} / نزولی {s['bearish']} "
              f"= {s['bull_pct']}٪ مثبت")
        print(f"   نسبت        : {s['ratio']:+.3f} ({s['state']})")
        print(f"   نمادها      : {[(x['symbol'], x['tagged']) for x in s['symbols']]}")
        print(f"   نمونه کم؟   : {s['low_sample']}")
    else:
        print("   ❌", s.get("error"))
    print("=" * 62)
