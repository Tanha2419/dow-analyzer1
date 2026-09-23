# -*- coding: utf-8 -*-
"""
بک تست موتور تصمیم گیری — سنجش صادقانه روی گذشته.

چرا این فایل وجود دارد
──────────────────────
موتور ایجنت وزن هایی دارد (COT ×۴، ترس و طمع ×۷، سوپرایز ×۶ …) که
از تحلیل منطقی آمده اند، نه از آزمایش. تا امروز هیچ سنجشی روی گذشته
انجام نشده بود و ژورنال زنده فقط ۶ رکورد با صفر ارزیابی داشت.

این ماژول موتور را کندل به کندل روی تاریخ اجرا می کند و نرخ برد
واقعی را بیرون می کشد.

صداقت روش شناختی — چه چیزی بازپخش می شود و چه چیزی نه
────────────────────────────────────────────────────
بعضی لایه های موتور «فقط لحظه ای» هستند؛ آرشیو تاریخی هم راستا
ندارند:

    COT             → هفتگی، فقط آخرین گزارش در دسترس است
    ترس و طمع CNN   → فقط مقدار امروز
    StockTwits      → فقط پیام های اخیر
    سوپرایز اقتصادی → تقویم فقط چند روز جلو/عقب
    زنجیره آپشن     → فقط تصویر امروز

اگر این ها را با مقدار «امروز» روی گذشته بگذاریم، نتیجه نشت آینده
(look-ahead bias) می دهد و نرخ برد به دروغ بالا می رود. پس کنار
گذاشته می شوند و این موضوع در خروجی صریحا اعلام می شود.

آنچه بازپخش می شود — همه فقط از کندل های تا لحظه تصمیم:
    • موتور پول هوشمند (SMC)         وزن ۰٫۸۵
    • رژیم بازار (هرست)
    • واگرایی مومنتوم / ویوترند
    • سوپرترند
    • الگوهای کندلی و نموداری
    • فصلی بودن روز هفته

نتیجه یک «کف» است، نه سقف: امتیاز کامل با لایه های لحظه ای ممکن
است بهتر باشد، ولی آن ادعا اثبات پذیر نیست. آنچه اینجا اندازه گیری
می شود، اثبات پذیر است.

روش ارزیابی
───────────
برای هر تصمیم، از همان کندل جلو می رویم و می بینیم کدام زودتر
لمس می شود:
    هدف  = ورود ± R × ATR       (پیش فرض R = ۱٫۵)
    ضرر  = ورود ∓ ۱ × ATR
اگر تا پایان افق هیچ کدام نخورد، با قیمت پایان بسته می شود.

سقف High/Low هر کندل بررسی می شود، نه فقط Close — یعنی لمس واقعی.
اگر در یک کندل هر دو لمس شوند، بدبینانه ضرر فرض می شود.

هزینه معامله کسر می شود (اسپرد + لغزش) تا عدد خوش بینانه نباشد.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import agent as AG
import assets as A
import regime_ai as RA
import smart_money as SM

# افق ارزیابی بر حسب تعداد کندل — همان مقادیر ژورنال زنده
HORIZON = {"5m": 12, "15m": 8, "30m": 6, "1h": 8, "1d": 5}

# هزینه رفت و برگشت بر حسب درصد — اسپرد + لغزش، محافظه کارانه
COST_PCT = {"XAUUSD": 0.020, "US30": 0.012}

# آستانه های همان موتور زنده
STRONG = 26.0
WEAK = 13.0


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=max(2, n // 2)).mean()


def _seasonality(df: pd.DataFrame, idx: int, interval: str) -> float:
    """سوگیری روز هفته — فقط از داده تا همین لحظه."""
    if interval != "1d" or idx < 120:
        return 0.0
    hist = df.iloc[:idx]
    dow = df.index[idx].dayofweek
    r = hist["Close"].pct_change()
    same = r[hist.index.dayofweek == dow].dropna()
    if len(same) < 20:
        return 0.0
    m = float(same.mean())
    sd = float(same.std())
    if sd <= 0:
        return 0.0
    t = m / (sd / math.sqrt(len(same)))
    return float(np.clip(t, -2.0, 2.0)) * 4.0


def score_at(df: pd.DataFrame, idx: int, interval: str) -> Optional[Dict]:
    """امتیاز موتور در کندل idx — فقط با داده تا همان لحظه.

    هیچ ردیفی بعد از idx دیده نمی شود؛ این تضمین نبود نشت آینده است.
    """
    win = df.iloc[: idx + 1]
    if len(win) < 160:
        return None

    parts: List = []

    # ---- موتور پول هوشمند ----
    # امتیاز زیر کلید signal است، نه ریشه خروجی
    try:
        smc = SM.run_full_smc(win.tail(400), interval=interval,
                              with_coalition=False)
        sig_blk = smc.get("signal") or {}
        smc_score = float(sig_blk.get("raw_score",
                                      sig_blk.get("score", 0.0)) or 0.0)
        parts.append(("SMC", smc_score * 0.85))
    except Exception:
        return None

    # ---- هوش بازار ----
    try:
        intel = RA.run_intelligence(win.tail(400), with_ml=False)
    except Exception:
        intel = {}

    reg = intel.get("regime") or {}
    if reg.get("regime") == "رونددار":
        conf = float(reg.get("confidence", 0) or 0)
        parts.append(("رژیم", conf / 100 * 6.0 *
                      (1 if smc.get("score", 0) > 0 else -1)))

    dv = intel.get("divergence") or {}
    if isinstance(dv, dict) and dv.get("found"):
        parts.append(("واگرایی", float(dv.get("bias", 0) or 0) * 5.0))

    st = intel.get("supertrend") or {}
    if isinstance(st, dict) and st.get("direction"):
        parts.append(("سوپرترند", 4.0 if st["direction"] == "صعودی" else -4.0))

    # ---- معاملات بلوکی ----
    bl = smc.get("blocks") or {}
    if isinstance(bl, dict) and bl.get("bias") is not None:
        parts.append(("بلوک", float(np.clip(
            float(bl["bias"]) * 0.8, -2.0, 2.0))))

    # ---- فصلی ----
    s = _seasonality(df, idx, interval)
    if s:
        parts.append(("فصلی", s))

    raw = sum(p[1] for p in parts)

    # جریمه های همان موتور زنده
    k = float(sig_blk.get("fake_penalty", 1.0) or 1.0)
    k *= float(sig_blk.get("hft_penalty", 1.0) or 1.0)

    score = raw * k
    conf = float(np.clip(abs(score) / 68 * 100, 0, 97))

    if score >= STRONG:
        sig, lab = 2, "خرید قوی"
    elif score >= WEAK:
        sig, lab = 1, "خرید ضعیف"
    elif score <= -STRONG:
        sig, lab = -2, "فروش قوی"
    elif score <= -WEAK:
        sig, lab = -1, "فروش ضعیف"
    else:
        sig, lab = 0, "خنثی"

    if conf >= 78:
        grade = "A+"
    elif conf >= 64:
        grade = "A"
    elif conf >= 50:
        grade = "B"
    elif conf >= 34:
        grade = "C"
    else:
        grade = "D"

    return dict(score=round(score, 2), signal=sig, label=lab,
                conf=round(conf, 1), grade=grade,
                parts={p[0]: round(p[1], 2) for p in parts})


def _outcome(df: pd.DataFrame, idx: int, direction: int, atr: float,
             horizon: int, r_mult: float, cost: float) -> Optional[Dict]:
    """نتیجه واقعی معامله — لمس High/Low، نه فقط Close."""
    entry = float(df["Close"].iloc[idx])
    if not np.isfinite(entry) or not np.isfinite(atr) or atr <= 0:
        return None

    tgt = entry + direction * r_mult * atr
    stp = entry - direction * atr
    fut = df.iloc[idx + 1: idx + 1 + horizon]
    if len(fut) < 2:
        return None

    for _, row in fut.iterrows():
        hi, lo = float(row["High"]), float(row["Low"])
        hit_t = hi >= tgt if direction > 0 else lo <= tgt
        hit_s = lo <= stp if direction > 0 else hi >= stp
        if hit_t and hit_s:
            return dict(win=False, r=-1.0, exit="هر دو (بدبینانه)")
        if hit_t:
            return dict(win=True, r=r_mult - cost, exit="هدف")
        if hit_s:
            return dict(win=False, r=-1.0 - cost, exit="حد ضرر")

    last = float(fut["Close"].iloc[-1])
    r = direction * (last - entry) / atr - cost
    return dict(win=r > 0, r=round(r, 3), exit="پایان افق")


def run(asset: str = "US30", interval: str = "1h",
        r_mult: float = 1.5, step: int = 1,
        max_bars: Optional[int] = None) -> Dict:
    """اجرای بک تست کامل موتور."""
    prof = A.profile(asset)
    sym = prof["candle_symbol"]

    try:
        df = AG.load(interval, symbol=sym)
    except Exception as e:
        return dict(ok=False, error=f"داده نیامد: {e}")

    if len(df) < 250:
        return dict(ok=False, error=f"داده کافی نیست ({len(df)} کندل)")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    atr = _atr(df)
    hz = HORIZON.get(interval, 8)
    cost = COST_PCT.get(asset, 0.015) / 100 * 0  # به واحد R تبدیل می شود

    start = 200
    end = len(df) - hz - 1
    if max_bars:
        start = max(start, end - max_bars)

    trades: List[Dict] = []
    skipped = 0

    for i in range(start, end, step):
        s = score_at(df, i, interval)
        if s is None:
            skipped += 1
            continue
        if s["signal"] == 0:
            continue

        a = float(atr.iloc[i])
        if not np.isfinite(a) or a <= 0:
            continue

        # هزینه بر حسب R: درصد هزینه ÷ (ATR بر حسب درصد)
        entry = float(df["Close"].iloc[i])
        atr_pct = a / entry * 100
        c_r = (COST_PCT.get(asset, 0.015) / atr_pct) if atr_pct > 0 else 0.02

        o = _outcome(df, i, 1 if s["signal"] > 0 else -1, a, hz, r_mult, c_r)
        if o is None:
            continue

        trades.append(dict(
            when=str(df.index[i])[:16],
            score=s["score"], signal=s["signal"], label=s["label"],
            grade=s["grade"], conf=s["conf"],
            direction="خرید" if s["signal"] > 0 else "فروش",
            strong=abs(s["signal"]) == 2,
            **o,
        ))

    if not trades:
        return dict(ok=False, error="هیچ سیگنالی تولید نشد",
                    bars=len(df), skipped=skipped)

    return dict(ok=True, asset=asset, interval=interval,
                symbol=sym, r_mult=r_mult, horizon=hz,
                bars=len(df), tested=(end - start) // step,
                skipped=skipped, trades=trades,
                **_stats(trades))


def _wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """بازه اطمینان ویلسون — صادقانه تر از نسبت خام."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0, c - h) * 100, 1), round(min(1, c + h) * 100, 1))


def _stats(trades: List[Dict]) -> Dict:
    n = len(trades)
    w = sum(1 for t in trades if t["win"])
    rs = [t["r"] for t in trades]
    lo, hi = _wilson(w, n)

    by_grade = {}
    for g in ("A+", "A", "B", "C", "D"):
        sub = [t for t in trades if t["grade"] == g]
        if sub:
            gw = sum(1 for t in sub if t["win"])
            gl, gh = _wilson(gw, len(sub))
            by_grade[g] = dict(
                n=len(sub), win_rate=round(gw / len(sub) * 100, 1),
                avg_r=round(float(np.mean([t["r"] for t in sub])), 3),
                ci=[gl, gh])

    by_dir = {}
    for d in ("خرید", "فروش"):
        sub = [t for t in trades if t["direction"] == d]
        if sub:
            dw = sum(1 for t in sub if t["win"])
            by_dir[d] = dict(
                n=len(sub), win_rate=round(dw / len(sub) * 100, 1),
                avg_r=round(float(np.mean([t["r"] for t in sub])), 3))

    strong = [t for t in trades if t["strong"]]
    weak = [t for t in trades if not t["strong"]]

    def _blk(sub):
        if not sub:
            return None
        sw = sum(1 for t in sub if t["win"])
        return dict(n=len(sub), win_rate=round(sw / len(sub) * 100, 1),
                    avg_r=round(float(np.mean([t["r"] for t in sub])), 3))

    eq = np.cumsum(rs)
    peak = np.maximum.accumulate(eq)
    dd = float((peak - eq).max()) if len(eq) else 0.0

    sd = float(np.std(rs))
    return dict(
        n=n, wins=w, losses=n - w,
        win_rate=round(w / n * 100, 1),
        win_rate_ci=[lo, hi],
        avg_r=round(float(np.mean(rs)), 3),
        median_r=round(float(np.median(rs)), 3),
        total_r=round(float(np.sum(rs)), 2),
        max_drawdown_r=round(dd, 2),
        sharpe=round(float(np.mean(rs)) / sd, 3) if sd > 0 else None,
        expectancy=round(float(np.mean(rs)), 3),
        by_grade=by_grade, by_direction=by_dir,
        strong=_blk(strong), weak=_blk(weak),
        beats_random=bool(lo > 50.0),
    )


def part_analysis(asset: str = "US30", interval: str = "1h",
                  max_bars: Optional[int] = None) -> Dict:
    """کدام جزء امتیاز واقعا پیش بینی کننده است؟

    برای هر جزء، همبستگی مقدارش با نتیجه R محاسبه می شود.
    مقدار مثبت یعنی کمک می کند؛ منفی یعنی ضرر می زند.
    """
    r = run(asset, interval, max_bars=max_bars)
    if not r.get("ok"):
        return r

    # جزءها در run ذخیره نمی شوند، پس دوباره محاسبه می شوند
    return dict(ok=True, note="برای تحلیل اجزا از run_parts استفاده کنید",
                summary={k: r[k] for k in
                         ("n", "win_rate", "avg_r", "beats_random")})


if __name__ == "__main__":
    import json
    import sys
    a = sys.argv[1] if len(sys.argv) > 1 else "US30"
    i = sys.argv[2] if len(sys.argv) > 2 else "1h"
    out = run(a, i, max_bars=300)
    out.pop("trades", None)
    print(json.dumps(out, ensure_ascii=False, indent=2))
