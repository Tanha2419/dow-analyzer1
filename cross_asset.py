# -*- coding: utf-8 -*-
"""
همبستگی طلا ↔ داوجونز و اندازه گیری پوزیشن بر پایه نوسان.

چرا
───
۱) وقتی هر دو دارایی سیگنال هم جهت می دهند ولی تاریخا معکوس
   حرکت می کنند، یکی از دو سیگنال احتمالا نویز است. این یک فیلتر
   ارزان و واقعی است.

۲) داشبورد حد ضرر می داد ولی حجم معامله را حساب نمی کرد. حد ضرر
   بدون حجم، مدیریت ریسک نیست.

همه اعداد از کندل های واقعی یاهو می آیند. اگر داده نیاید، None
برگردانده می شود — نه تخمین.
"""
from __future__ import annotations

import datetime as _dt
import time as _time
from typing import Dict, Optional

import numpy as np
import pandas as pd

_cache: Dict[str, tuple] = {}

# ارزش هر واحد حرکت، برای یک لات استاندارد
CONTRACT = {
    "XAUUSD": dict(unit="اونس", per_point=100.0, lot_name="لات (۱۰۰ اونس)"),
    "US30":   dict(unit="واحد", per_point=1.0,   lot_name="قرارداد (۱$ هر واحد)"),
}


def _hist(symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(period=period, interval="1d")
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df.dropna(subset=["Close"])
    except Exception:
        return None


def correlation(ttl: float = 1800.0) -> Dict:
    """همبستگی بازده روزانه طلا و داوجونز در سه پنجره."""
    hit = _cache.get("corr")
    if hit and (_time.time() - hit[0]) < ttl:
        return hit[1]

    g = _hist("GC=F")
    d = _hist("DIA")
    if g is None or d is None:
        return dict(ok=False, error="داده تاریخی نیامد")

    gg = g["Close"].pct_change().dropna()
    dd = d["Close"].pct_change().dropna()
    gg.index = gg.index.normalize()
    dd.index = dd.index.normalize()
    j = pd.concat([gg.rename("gold"), dd.rename("dow")],
                  axis=1, join="inner", sort=True).dropna()

    if len(j) < 40:
        return dict(ok=False, error=f"همپوشانی کم ({len(j)} روز)")

    out: Dict = dict(ok=True, days=len(j),
                     last_date=str(j.index[-1].date()))
    for w in (20, 60, 120):
        if len(j) >= w:
            sub = j.tail(w)
            out[f"corr_{w}"] = round(float(
                sub["gold"].corr(sub["dow"])), 3)
        else:
            out[f"corr_{w}"] = None

    c = out.get("corr_20")
    if c is None:
        return dict(ok=False, error="پنجره کوتاه محاسبه نشد")

    if c <= -0.3:
        tone, fa = "inverse", "معکوس — معمولا خلاف هم حرکت می کنند"
    elif c >= 0.3:
        tone, fa = "aligned", "هم جهت — با هم حرکت می کنند"
    else:
        tone, fa = "decoupled", "مستقل — رابطه روشنی ندارند"

    out.update(tone=tone, tone_fa=fa,
               source="Yahoo Finance · GC=F و DIA · بازده روزانه")
    _cache["corr"] = (_time.time(), out)
    return out


def check_signals(gold_dir: Optional[int], dow_dir: Optional[int]) -> Dict:
    """آیا دو سیگنال با همبستگی تاریخی سازگارند؟

    gold_dir / dow_dir: ‎+1 خرید، ‎-1 فروش، 0 یا None خنثی
    """
    c = correlation()
    if not c.get("ok"):
        return dict(ok=False, error=c.get("error"))

    base = dict(ok=True, corr_20=c["corr_20"], tone=c["tone"],
                tone_fa=c["tone_fa"], days=c["days"])

    if not gold_dir or not dow_dir:
        return dict(**base, verdict="single",
                    verdict_fa="فقط یک سیگنال فعال است",
                    warn=False,
                    note="برای بررسی سازگاری، هر دو دارایی باید "
                         "سیگنال داشته باشند.")

    same = (gold_dir > 0) == (dow_dir > 0)
    tone = c["tone"]

    if tone == "inverse" and same:
        return dict(**base, verdict="conflict", warn=True,
                    verdict_fa="⚠️ ناسازگار",
                    note=f"همبستگی {c['corr_20']} یعنی این دو معمولا "
                         "خلاف هم می روند، ولی هر دو سیگنال هم جهت "
                         "دادند. احتمالا یکی نویز است — حجم را نصف "
                         "کنید یا منتظر تأیید بمانید.")

    if tone == "aligned" and not same:
        return dict(**base, verdict="conflict", warn=True,
                    verdict_fa="⚠️ ناسازگار",
                    note=f"همبستگی {c['corr_20']} یعنی این دو با هم "
                         "می روند، ولی سیگنال ها مخالف هم اند. یکی "
                         "احتمالا اشتباه است.")

    if tone == "inverse" and not same:
        return dict(**base, verdict="confirm", warn=False,
                    verdict_fa="✅ سازگار",
                    note="سیگنال ها مخالف هم اند و همبستگی هم معکوس "
                         "است — با هم جور در می آید.")

    if tone == "aligned" and same:
        return dict(**base, verdict="confirm", warn=False,
                    verdict_fa="✅ سازگار",
                    note="هر دو هم جهت و همبستگی مثبت — تأیید "
                         "متقابل. ولی ریسک کل را حواستان باشد؛ "
                         "دو معامله هم جهت یعنی ریسک دو برابر.")

    return dict(**base, verdict="neutral", warn=False,
                verdict_fa="بدون تداخل",
                note="رابطه دو دارایی فعلا روشن نیست، پس سیگنال ها "
                     "مستقل بررسی می شوند.")


def position_size(asset: str, entry: float, stop: float,
                  equity: float = 10_000.0,
                  risk_pct: float = 1.0) -> Dict:
    """حجم معامله بر پایه فاصله حد ضرر — نه عدد دلخواه.

    قاعده: هرگز بیش از risk_pct درصد سرمایه در یک معامله ریسک نشود.
    فاصله حد ضرر تعیین می کند چند واحد می توان گرفت.
    """
    if not entry or not stop or entry <= 0 or stop <= 0:
        return dict(ok=False, error="ورود یا حد ضرر معتبر نیست")
    if abs(entry - stop) < 1e-9:
        return dict(ok=False, error="ورود و حد ضرر یکی است")

    spec = CONTRACT.get(asset, CONTRACT["US30"])
    risk_money = equity * risk_pct / 100.0
    dist = abs(entry - stop)
    units = risk_money / dist
    lots = units / spec["per_point"]

    return dict(
        ok=True, asset=asset,
        equity=round(equity, 2), risk_pct=risk_pct,
        risk_money=round(risk_money, 2),
        entry=round(entry, 2), stop=round(stop, 2),
        stop_distance=round(dist, 2),
        stop_distance_pct=round(dist / entry * 100, 3),
        units=round(units, 4),
        lots=round(lots, 4),
        lot_name=spec["lot_name"],
        unit=spec["unit"],
        note=(f"با {equity:,.0f} دلار سرمایه و ریسک {risk_pct}٪، "
              f"حداکثر {risk_money:,.0f} دلار در خطر است. چون فاصله "
              f"حد ضرر {dist:,.2f} {spec['unit']} است، حجم مجاز "
              f"{lots:.3f} {spec['lot_name']} می شود."),
        beginner=("حجم معامله را فاصله حد ضرر تعیین می کند، نه حدس. "
                  "حد ضرر دورتر ⟵ حجم کمتر. این طور ضرر هر معامله "
                  "همیشه ثابت می ماند."),
    )


def build(gold_dir: Optional[int] = None,
          dow_dir: Optional[int] = None) -> Dict:
    """بسته کامل برای نمایش در داشبورد."""
    out: Dict = {}
    try:
        out["correlation"] = correlation()
    except Exception as e:
        out["correlation"] = dict(ok=False, error=str(e)[:150])
    try:
        out["check"] = check_signals(gold_dir, dow_dir)
    except Exception as e:
        out["check"] = dict(ok=False, error=str(e)[:150])
    out["checked_at"] = _dt.datetime.now(_dt.timezone.utc)\
        .strftime("%Y-%m-%d %H:%M UTC")
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(build(1, 1), ensure_ascii=False, indent=2))
    print(json.dumps(position_size("XAUUSD", 4320, 4295),
                     ensure_ascii=False, indent=2))
