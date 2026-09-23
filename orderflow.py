#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orderflow.py — جریان سفارش، فصلی بودن و الگوها

بخش اول — جریان سفارش از کندل یک دقیقه ای:
    Delta, CVD, واگرایی CVD, معاملات بلوکی, نمایه حجم

  هشدار صداقت: داده tick با جهت معامله رایگان نیست.
  Delta از موقعیت بسته شدن در دامنه کندل بازسازی می شود و
  در همه خروجی ها با estimated=True علامت می خورد.

بخش دوم — فصلی بودن (کاملا واقعی، از داده تاریخی):
    اثر روز هفته، ماه، ساعت، روزهای حول تعطیلات

بخش سوم — الگوها (محاسبه عددی، نه بینایی ماشین):
    کندلی کلاسیک + سر و شانه / مثلث / دو قله
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ================================================================ جریان سفارش
def delta_cvd(df: pd.DataFrame) -> Dict:
    """
    Delta و CVD بازسازی شده از OHLCV.

    روش: نسبت بسته شدن در دامنه کندل (CLV) سهم خرید را تعیین می کند.
      CLV = ((C-L) - (H-C)) / (H-L)   در بازه ۱- تا ۱+
      Delta = CLV × Volume
    این تقریب است نه داده واقعی جریان سفارش.
    """
    if df.empty or len(df) < 5:
        return dict(ok=False, error="داده کافی نیست")

    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    o = df["Open"].to_numpy(float)
    v = df["Volume"].to_numpy(float)

    rng = h - l
    clv = np.where(rng > 0, ((c - l) - (h - c)) / np.where(rng == 0, 1, rng), 0.0)
    delta = clv * v
    cvd = np.cumsum(delta)

    buy_vol = v * (clv + 1) / 2
    sell_vol = v - buy_vol

    # واگرایی CVD در برابر قیمت
    n = min(40, len(c) - 1)
    div = None
    if n >= 12:
        pr = (c[-1] - c[-n]) / c[-n] * 100
        rngc = np.max(np.abs(cvd[-n:])) or 1.0
        cd = (cvd[-1] - cvd[-n]) / rngc * 100
        if pr > 0.12 and cd < -6:
            div = dict(type="bearish", price_chg=round(pr, 3), cvd_chg=round(cd, 2),
                       note="قیمت بالا می رود اما CVD پایین — فروش پنهان نهادی")
        elif pr < -0.12 and cd > 6:
            div = dict(type="bullish", price_chg=round(pr, 3), cvd_chg=round(cd, 2),
                       note="قیمت پایین می آید اما CVD بالا — جذب نهادی")

    w = min(30, len(delta))
    recent = float(delta[-w:].sum())
    dmean = float(np.mean(np.abs(delta[-200:]))) if len(delta) >= 20 else 1.0
    pressure = float(np.clip(recent / (dmean * w + 1e-9), -3, 3))

    cvd_slope = 0.0
    if len(cvd) >= 20:
        yy = cvd[-20:]
        cvd_slope = float(np.polyfit(np.arange(20), yy, 1)[0] /
                          (np.std(yy) + 1e-9))

    return dict(
        ok=True, estimated=True,
        method="بازسازی شده از CLV — داده tick واقعی رایگان نیست",
        delta_last=round(float(delta[-1]), 1),
        delta_sum_30=round(recent, 1),
        cvd_last=round(float(cvd[-1]), 1),
        cvd_slope=round(cvd_slope, 4),
        pressure=round(pressure, 3),
        pressure_label=("فشار خرید قوی" if pressure > 1.0 else
                        "فشار خرید" if pressure > 0.3 else
                        "فشار فروش قوی" if pressure < -1.0 else
                        "فشار فروش" if pressure < -0.3 else "متعادل"),
        buy_volume=int(buy_vol[-30:].sum()), sell_volume=int(sell_vol[-30:].sum()),
        divergence=div,
        series=dict(
            cvd=[round(float(x), 1) for x in cvd[-160:]],
            delta=[round(float(x), 1) for x in delta[-160:]],
            close=[round(float(x), 4) for x in c[-160:]],
        ),
    )


def block_trades(df: pd.DataFrame, z_thr: float = 2.2) -> Dict:
    """کندل هایی با حجم به شدت غیرعادی — رد پای معامله بلوکی."""
    if len(df) < 40:
        return dict(ok=False, items=[])
    v = df["Volume"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    vm = pd.Series(v).rolling(30).mean().to_numpy()
    vs = pd.Series(v).rolling(30).std().to_numpy()
    z = (v - vm) / np.where((vs == 0) | np.isnan(vs), np.nan, vs)
    rng = h - l
    clv = np.where(rng > 0, ((c - l) - (h - c)) / np.where(rng == 0, 1, rng), 0)

    items = []
    for i in range(len(df)):
        if np.isfinite(z[i]) and z[i] >= z_thr:
            items.append(dict(
                i=int(i), time=pd.Timestamp(df.index[i]).isoformat(),
                price=round(float(c[i]), 4), volume=int(v[i]),
                z=round(float(z[i]), 2),
                vol_x=round(float(v[i] / vm[i]), 2) if vm[i] else None,
                side=("خرید" if clv[i] > 0.25 else
                      "فروش" if clv[i] < -0.25 else "خنثی"),
                move_pct=round(float((c[i] - o[i]) / o[i] * 100), 3),
            ))
    recent = [x for x in items if x["i"] >= len(df) - 60]
    nb = sum(1 for x in recent if x["side"] == "خرید")
    ns = sum(1 for x in recent if x["side"] == "فروش")
    return dict(ok=True, n=len(items), recent=len(recent),
                buy=nb, sell=ns,
                bias=float(np.clip((nb - ns) / max(len(recent), 1) * 2, -2, 2)),
                items=items[-12:])


# ================================================================ فصلی بودن
WD_FA = ["دوشنبه", "سه شنبه", "چهارشنبه", "پنجشنبه", "جمعه", "شنبه", "یکشنبه"]
MON_FA = ["ژانویه", "فوریه", "مارس", "آوریل", "مه", "ژوئن", "ژوئیه",
          "اوت", "سپتامبر", "اکتبر", "نوامبر", "دسامبر"]


def _stats(x: pd.Series) -> Dict:
    if len(x) == 0:
        return dict(n=0)
    a = x.to_numpy(float)
    win = float((a > 0).mean() * 100)
    mean = float(a.mean())
    sd = float(a.std()) or 1e-9
    # آماره t برای معناداری
    tstat = mean / (sd / np.sqrt(len(a))) if len(a) > 1 else 0.0
    return dict(n=int(len(a)), mean=round(mean, 4),
                median=round(float(np.median(a)), 4),
                win_rate=round(win, 1), std=round(sd, 4),
                t_stat=round(float(tstat), 2),
                significant=bool(abs(tstat) > 1.96))


def seasonality(symbol: str = "DIA", years: int = 10) -> Dict:
    """اثر روز هفته، ماه، و روزهای حول تعطیلات — از داده روزانه واقعی."""
    import yfinance as yf
    df = yf.Ticker(symbol).history(period=f"{years}y")
    if df.empty or len(df) < 200:
        return dict(ok=False, error="داده کافی نیست")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    r = df["Close"].pct_change().dropna() * 100
    d = pd.DataFrame(dict(ret=r))
    d["wd"] = d.index.dayofweek
    d["mon"] = d.index.month
    d["dom"] = d.index.day

    by_wd = []
    for k in range(5):
        s = _stats(d[d["wd"] == k]["ret"])
        if s.get("n"):
            by_wd.append(dict(key=k, name=WD_FA[k], **s))
    by_mon = []
    for k in range(1, 13):
        s = _stats(d[d["mon"] == k]["ret"])
        if s.get("n"):
            by_mon.append(dict(key=k, name=MON_FA[k - 1], **s))

    # اثر ابتدا/انتهای ماه
    tom = _stats(d[(d["dom"] <= 3) | (d["dom"] >= 28)]["ret"])
    mid = _stats(d[(d["dom"] > 3) & (d["dom"] < 28)]["ret"])

    # روزهای حول تعطیل: فاصله بیش از یک روز تقویمی بین دو جلسه
    idx = d.index
    gaps = np.diff(idx.values).astype("timedelta64[D]").astype(int)
    pre, post = [], []
    for i, g in enumerate(gaps):
        if g >= 3:                       # آخر هفته عادی ۳ روز است
            if i < len(d):
                pre.append(d["ret"].iloc[i])
            if i + 1 < len(d):
                post.append(d["ret"].iloc[i + 1])
    hol = dict(pre_holiday=_stats(pd.Series(pre)),
               post_holiday=_stats(pd.Series(post)))

    best_wd = max(by_wd, key=lambda x: x["mean"]) if by_wd else None
    worst_wd = min(by_wd, key=lambda x: x["mean"]) if by_wd else None
    best_mon = max(by_mon, key=lambda x: x["mean"]) if by_mon else None
    worst_mon = min(by_mon, key=lambda x: x["mean"]) if by_mon else None

    today = datetime.now()
    cur_wd = next((x for x in by_wd if x["key"] == today.weekday()), None)
    cur_mon = next((x for x in by_mon if x["key"] == today.month), None)

    return dict(ok=True, symbol=symbol, years=years, n_days=int(len(d)),
                period=f"{d.index.min().date()} تا {d.index.max().date()}",
                by_weekday=by_wd, by_month=by_mon,
                turn_of_month=tom, mid_month=mid, holiday=hol,
                best_weekday=best_wd, worst_weekday=worst_wd,
                best_month=best_mon, worst_month=worst_mon,
                current_weekday=cur_wd, current_month=cur_mon)


def intraday_seasonality(symbol: str = "DIA", days: int = 59) -> Dict:
    """الگوی ساعتی داخل روز — از کندل ۳۰ دقیقه ای، به وقت تهران."""
    import yfinance as yf
    df = yf.Ticker(symbol).history(interval="30m", period=f"{days}d")
    if df.empty or len(df) < 50:
        return dict(ok=False, error="داده کافی نیست")
    df.index = pd.to_datetime(df.index)
    try:
        teh = df.index.tz_convert("Asia/Tehran")
    except Exception:
        teh = df.index
    r = df["Close"].pct_change() * 100
    d = pd.DataFrame(dict(ret=r.to_numpy()), index=teh).dropna()
    d["slot"] = [f"{t.hour:02d}:{'30' if t.minute >= 30 else '00'}" for t in d.index]

    rows = []
    for slot, g in d.groupby("slot"):
        s = _stats(g["ret"])
        if s.get("n", 0) >= 5:
            rows.append(dict(slot=slot, **s))
    rows.sort(key=lambda x: x["slot"])
    if not rows:
        return dict(ok=False, error="نمونه کافی نیست")
    best = max(rows, key=lambda x: x["mean"])
    worst = min(rows, key=lambda x: x["mean"])
    return dict(ok=True, symbol=symbol, timezone="Asia/Tehran",
                n=int(len(d)), slots=rows, best=best, worst=worst,
                note="ساعت ها به وقت تهران — بازگشایی نیویورک حدود ۱۷:۰۰")


# ================================================================ الگوها
def candle_patterns(df: pd.DataFrame, lookback: int = 12) -> Dict:
    """الگوهای کندلی کلاسیک — محاسبه عددی روی OHLC."""
    if len(df) < 6:
        return dict(ok=False, items=[])
    o = df["Open"].to_numpy(float); h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float); c = df["Close"].to_numpy(float)
    body = np.abs(c - o)
    rng = np.maximum(h - l, 1e-9)
    upsh = h - np.maximum(o, c)
    dnsh = np.minimum(o, c) - l
    avg = pd.Series(body).rolling(14).mean().to_numpy()

    found: List[Dict] = []
    s = max(1, len(df) - lookback)
    for i in range(s, len(df)):
        t = pd.Timestamp(df.index[i]).isoformat()
        ab = avg[i] if np.isfinite(avg[i]) and avg[i] > 0 else body[i] or 1e-9

        def add(name, direction, strength=1.0):
            found.append(dict(name=name, dir=direction, i=int(i), time=t,
                              bars_ago=int(len(df) - 1 - i),
                              strength=round(float(strength), 2)))

        if body[i] / rng[i] < 0.1:
            add("دوجی", 0, 0.5)
        if dnsh[i] > body[i] * 2 and upsh[i] < body[i] * 0.6 and body[i] > 0:
            add("چکش" if c[i] >= o[i] else "چکش معکوس نزولی", 1,
                min(dnsh[i] / (body[i] + 1e-9) / 2, 3))
        if upsh[i] > body[i] * 2 and dnsh[i] < body[i] * 0.6 and body[i] > 0:
            add("ستاره ثاقب", -1, min(upsh[i] / (body[i] + 1e-9) / 2, 3))
        if i >= 1:
            if c[i] > o[i] and c[i - 1] < o[i - 1] and c[i] >= o[i - 1] and o[i] <= c[i - 1]:
                add("پوشای صعودی", 1, min(body[i] / ab, 3))
            if c[i] < o[i] and c[i - 1] > o[i - 1] and o[i] >= c[i - 1] and c[i] <= o[i - 1]:
                add("پوشای نزولی", -1, min(body[i] / ab, 3))
        if i >= 2:
            if (c[i-2] < o[i-2] and body[i-1] / rng[i-1] < 0.35
                    and c[i] > o[i] and c[i] > (o[i-2] + c[i-2]) / 2):
                add("ستاره صبحگاهی", 1, 2.0)
            if (c[i-2] > o[i-2] and body[i-1] / rng[i-1] < 0.35
                    and c[i] < o[i] and c[i] < (o[i-2] + c[i-2]) / 2):
                add("ستاره عصرگاهی", -1, 2.0)
        if body[i] > ab * 2.2:
            add("مارابوزو صعودی" if c[i] > o[i] else "مارابوزو نزولی",
                1 if c[i] > o[i] else -1, min(body[i] / ab / 2, 3))

    found.sort(key=lambda x: x["bars_ago"])
    bias = sum(x["dir"] * x["strength"] * max(0.25, 1 - x["bars_ago"] / 14)
               for x in found)
    return dict(ok=True, items=found[:12], n=len(found),
                bias=round(float(np.clip(bias, -4, 4)), 2))


def _pivots(x: np.ndarray, w: int = 5) -> Tuple[List[int], List[int]]:
    hi, lo = [], []
    for i in range(w, len(x) - w):
        seg = x[i - w:i + w + 1]
        if seg.argmax() == w:
            hi.append(i)
        if seg.argmin() == w:
            lo.append(i)
    return hi, lo


def chart_patterns(df: pd.DataFrame, lookback: int = 130) -> Dict:
    """سر و شانه، دو قله/دو دره، مثلث — با هندسه روی پیوت ها."""
    d = df.tail(min(lookback, len(df)))
    if len(d) < 40:
        return dict(ok=False, items=[])
    h = d["High"].to_numpy(float); l = d["Low"].to_numpy(float)
    c = d["Close"].to_numpy(float)
    ph, pl = _pivots(c, 5)
    items: List[Dict] = []
    N = len(c)

    def ago(i):
        return int(N - 1 - i)

    # --- دو قله / دو دره ---
    for a, b in zip(ph, ph[1:]):
        if b - a < 6:
            continue
        if abs(h[a] - h[b]) / max(h[a], 1e-9) < 0.018:
            trough = l[a:b].min() if b > a else None
            if trough and (min(h[a], h[b]) - trough) / trough > 0.012:
                items.append(dict(name="دو قله", dir=-1, bars_ago=ago(b),
                                  level=round(float((h[a] + h[b]) / 2), 4),
                                  neckline=round(float(trough), 4)))
    for a, b in zip(pl, pl[1:]):
        if b - a < 6:
            continue
        if abs(l[a] - l[b]) / max(l[a], 1e-9) < 0.018:
            peak = h[a:b].max() if b > a else None
            if peak and (peak - max(l[a], l[b])) / max(l[a], 1e-9) > 0.012:
                items.append(dict(name="دو دره", dir=1, bars_ago=ago(b),
                                  level=round(float((l[a] + l[b]) / 2), 4),
                                  neckline=round(float(peak), 4)))

    # --- سر و شانه ---
    if len(ph) >= 3:
        for x, y, z in zip(ph, ph[1:], ph[2:]):
            ls, hd, rs = h[x], h[y], h[z]
            if hd > ls * 1.012 and hd > rs * 1.012 and abs(ls - rs) / max(ls, 1e-9) < 0.03:
                nk = float(min(l[x:y].min(), l[y:z].min()))
                items.append(dict(name="سر و شانه", dir=-1, bars_ago=ago(z),
                                  level=round(float(hd), 4), neckline=round(nk, 4)))
    if len(pl) >= 3:
        for x, y, z in zip(pl, pl[1:], pl[2:]):
            ls, hd, rs = l[x], l[y], l[z]
            if hd < ls * 0.988 and hd < rs * 0.988 and abs(ls - rs) / max(ls, 1e-9) < 0.03:
                nk = float(max(h[x:y].max(), h[y:z].max()))
                items.append(dict(name="سر و شانه معکوس", dir=1, bars_ago=ago(z),
                                  level=round(float(hd), 4), neckline=round(nk, 4)))

    # --- مثلث: همگرایی خط روند سقف ها و کف ها ---
    if len(ph) >= 2 and len(pl) >= 2:
        hh, ll = ph[-2:], pl[-2:]
        sh = (h[hh[1]] - h[hh[0]]) / max(hh[1] - hh[0], 1)
        sl = (l[ll[1]] - l[ll[0]]) / max(ll[1] - ll[0], 1)
        span = abs(h[hh[-1]] - l[ll[-1]])
        if span > 0:
            if sh < -1e-5 and sl > 1e-5:
                items.append(dict(name="مثلث متقارن", dir=0,
                                  bars_ago=ago(max(hh[1], ll[1])),
                                  level=round(float((h[hh[1]] + l[ll[1]]) / 2), 4)))
            elif abs(sh) < abs(sl) * 0.35 and sl > 1e-5:
                items.append(dict(name="مثلث صعودی", dir=1,
                                  bars_ago=ago(max(hh[1], ll[1])),
                                  level=round(float(h[hh[1]]), 4)))
            elif abs(sl) < abs(sh) * 0.35 and sh < -1e-5:
                items.append(dict(name="مثلث نزولی", dir=-1,
                                  bars_ago=ago(max(hh[1], ll[1])),
                                  level=round(float(l[ll[1]]), 4)))

    items = sorted(items, key=lambda x: x["bars_ago"])[:8]
    bias = sum(x["dir"] * max(0.3, 1 - x["bars_ago"] / 40) for x in items)
    return dict(ok=True, items=items, n=len(items),
                bias=round(float(np.clip(bias, -3, 3)), 2))


# ================================================================ تجمیع
def build_orderflow(symbol: str = "DIA", interval: str = "1h",
                    with_seasonality: bool = True) -> Dict:
    import yfinance as yf
    per = {"1m": "5d", "5m": "60d", "15m": "60d", "30m": "60d",
           "1h": "730d", "1d": "5y"}.get(interval, "1y")
    df = yf.Ticker(symbol).history(interval=interval, period=per)
    if df.empty:
        return dict(ok=False, error="داده دریافت نشد")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    out: Dict = dict(ok=True, symbol=symbol, interval=interval, bars=len(df))

    # جریان سفارش روی ریزترین تایم فریم ممکن
    fine = df
    if interval in ("1d", "1h"):
        try:
            fine = yf.Ticker(symbol).history(interval="1m", period="5d")
            fine.index = pd.to_datetime(fine.index)
            if fine.index.tz is not None:
                fine.index = fine.index.tz_localize(None)
            out["flow_source"] = "کندل ۱ دقیقه ای (۵ روز)"
        except Exception:
            fine = df
            out["flow_source"] = f"کندل {interval}"
    else:
        out["flow_source"] = f"کندل {interval}"

    out["delta"] = delta_cvd(fine)
    out["blocks"] = block_trades(fine)
    out["candles"] = candle_patterns(df)
    out["chart"] = chart_patterns(df)
    if with_seasonality:
        out["seasonality"] = seasonality(symbol)
        out["intraday"] = intraday_seasonality(symbol)
    return out


if __name__ == "__main__":
    r = build_orderflow("DIA", "1h")
    d = r["delta"]
    print("=== جریان سفارش ===")
    print(f"منبع: {r['flow_source']}")
    if d.get("ok"):
        print(f"⚠️  {d['method']}")
        print(f"Delta آخر {d['delta_last']:,.0f} | مجموع ۳۰ کندل {d['delta_sum_30']:,.0f}")
        print(f"CVD {d['cvd_last']:,.0f} | شیب {d['cvd_slope']:+.3f}")
        print(f"فشار {d['pressure']:+.2f} — {d['pressure_label']}")
        print(f"حجم خرید {d['buy_volume']:,} در برابر فروش {d['sell_volume']:,}")
        if d.get("divergence"):
            print(f"🔀 واگرایی CVD: {d['divergence']['note']}")
    b = r["blocks"]
    if b.get("ok"):
        print(f"\n=== معاملات بلوکی === کل {b['n']} | اخیر {b['recent']} "
              f"(خرید {b['buy']} فروش {b['sell']}) بایاس {b['bias']:+.2f}")
        for x in b["items"][-5:]:
            print(f"   {x['time'][:16]}  {x['side']:<6} حجم×{x['vol_x']}  z={x['z']}")

    s = r.get("seasonality", {})
    if s.get("ok"):
        print(f"\n=== فصلی بودن ({s['period']}, {s['n_days']} روز) ===")
        print(f"{'روز':<12}{'میانگین٪':>10}{'برد٪':>8}{'نمونه':>7}{'t':>7}  معنادار")
        for x in s["by_weekday"]:
            print(f"{x['name']:<12}{x['mean']:>+10.4f}{x['win_rate']:>8.1f}"
                  f"{x['n']:>7}{x['t_stat']:>7.2f}  {'✓' if x['significant'] else ''}")
        print(f"\nبهترین روز: {s['best_weekday']['name']} | "
              f"بدترین: {s['worst_weekday']['name']}")
        print(f"بهترین ماه: {s['best_month']['name']} ({s['best_month']['mean']:+.3f}٪) | "
              f"بدترین: {s['worst_month']['name']} ({s['worst_month']['mean']:+.3f}٪)")
        tm, mm = s["turn_of_month"], s["mid_month"]
        print(f"ابتدا/انتهای ماه {tm['mean']:+.4f}٪ در برابر میانه ماه {mm['mean']:+.4f}٪")
        hp = s["holiday"]
        print(f"پیش از تعطیل {hp['pre_holiday'].get('mean')}٪ | "
              f"پس از تعطیل {hp['post_holiday'].get('mean')}٪")
        if s.get("current_weekday"):
            cw = s["current_weekday"]
            print(f"\nامروز {cw['name']}: میانگین {cw['mean']:+.4f}٪ "
                  f"برد {cw['win_rate']}٪ {'(معنادار)' if cw['significant'] else ''}")

    it = r.get("intraday", {})
    if it.get("ok"):
        print(f"\n=== الگوی ساعتی (تهران) ===")
        print(f"بهترین {it['best']['slot']} ({it['best']['mean']:+.4f}٪) | "
              f"بدترین {it['worst']['slot']} ({it['worst']['mean']:+.4f}٪)")

    cp = r["candles"]
    if cp.get("ok") and cp["items"]:
        print(f"\n=== الگوهای کندلی (بایاس {cp['bias']:+.2f}) ===")
        for x in cp["items"][:6]:
            print(f"   {x['name']:<20}{'صعودی' if x['dir']>0 else ('نزولی' if x['dir']<0 else 'خنثی'):<8}"
                  f"{x['bars_ago']:>3} کندل قبل  قدرت {x['strength']}")
    ch = r["chart"]
    if ch.get("ok") and ch["items"]:
        print(f"\n=== الگوهای نموداری (بایاس {ch['bias']:+.2f}) ===")
        for x in ch["items"]:
            print(f"   {x['name']:<18}{x['bars_ago']:>4} کندل قبل  سطح {x['level']}")
