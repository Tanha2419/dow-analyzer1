# -*- coding: utf-8 -*-
"""سیگنال اعتبارسنجی شده — تنها لبه ای که بک تست تاییدش کرده.

چرا این ماژول جدا از ایجنت است
──────────────────────────────
ایجنت (agent.decide) ۲۵ جزء امتیاز دارد. اندازه گیری روی دفترچه
واقعی نشان داد ۷۰٪ وزن این اجزا از منابع شبکه ای می آید — اخبار،
StockTwits، انتظار نرخ بهره، COT، FRED، DXY، بازده اوراق. هیچ کدام
قابل بازپخش تاریخی نیستند، پس روی امتیاز ایجنت هیچ آستانه ای
اعتبارسنجی نشده و نمی شود.

آستانه ۳۲ که ما داریم از engine_backtest آمده؛ موتوری که فقط از
کندل استفاده می کند. تایید عددی (۱ اکتبر ۲۰۲۶): اسکن ۶۰۰ روزه
همین موتور برای طلا ۵۵ سیگنال بالای ۳۲ داد که مقیاس شده به بازه
بک تست اصلی می شود ۷۱ — و بک تست اصلی ۶۹ گزارش کرده بود.

پس کار درست عوض کردن آستانه نیست، اعمالش روی موتور درست است.
این ماژول همان موتور را زنده اجرا می کند (حدود ۰.۴ ثانیه).

⚠ این عدد با امتیاز ایجنت یکی نیست و نباید با هم مقایسه شوند.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional

# نتیجه بک تست اصلی — مبنای آستانه. دست نزنید مگر بک تست دوباره اجرا شود.
VALIDATED: Dict[str, Dict] = {
    "US30":   dict(threshold=32, target_r=2.5, n=144, win_rate=56.9,
                   avg_r=0.399, t_stat=3.98),
    "XAUUSD": dict(threshold=32, target_r=2.0, n=69, win_rate=62.3,
                   avg_r=0.639, t_stat=3.99),
}

# توزیع امتیاز از اسکن ۶۰۰ روزه (۱ اکتبر ۲۰۲۶) — برای نمایش صدک
_DIST: Dict[str, Dict] = {
    "US30":   dict(p50=16.2, p80=27.4, p90=36.2, p95=42.3, p99=51.2,
                   mean_abs=18.37, hit_rate=15.5, every_days=6),
    "XAUUSD": dict(p50=16.7, p80=24.9, p90=30.8, p95=38.7, p99=48.0,
                   mean_abs=17.01, hit_rate=9.2, every_days=11),
}

_TTL = 900.0                      # ۱۵ دقیقه — کندل روزانه کند تغییر می کند
_cache: Dict[str, Dict] = {}
_lock = threading.Lock()


def _pctile(asset: str, a: float) -> Optional[int]:
    """صدک تقریبی این امتیاز در توزیع تاریخی."""
    d = _DIST.get(asset)
    if not d:
        return None
    for p, key in ((99, "p99"), (95, "p95"), (90, "p90"),
                   (80, "p80"), (50, "p50")):
        if a >= d[key]:
            return p
    return None


def compute(asset: str = "US30", interval: str = "1d") -> Dict:
    """امتیاز موتور اعتبارسنجی شده روی آخرین کندل بسته شده."""
    import agent
    import assets as A
    import engine_backtest as eb

    prof = A.profile(asset)
    key = prof["key"]
    cfg = VALIDATED.get(key)
    if not cfg:
        return dict(ok=False, error="این دارایی بک تست اعتبارسنجی شده ندارد")
    if interval != "1d":
        return dict(ok=False,
                    error="فقط تایم فریم روزانه اعتبارسنجی شده است",
                    note_fa="بک تست روی ۵ دقیقه، ۱۵ دقیقه، ۳۰ دقیقه و "
                            "یک ساعته سود نداد؛ برای آنها آستانه ای نداریم.")

    df = agent.load(interval, symbol=prof["candle_symbol"])
    if df is None or len(df) < 260:
        return dict(ok=False, error="کندل کافی نیست")

    s = eb.score_at(df, len(df) - 1, interval)
    if not s:
        return dict(ok=False, error="موتور امتیاز نداد")

    score = float(s["score"])
    a = abs(score)
    th = float(cfg["threshold"])
    qualifies = a >= th
    direction = 0 if not qualifies else (1 if score > 0 else -1)

    # قیمت خام فید است؛ برای نمایش باید به مقیاس دارایی برود
    raw_price = float(df["Close"].iloc[-1])
    sc = (float(prof["display_scale"])
          if prof.get("basis_mode") == "scale" else 1.0)

    dist = _DIST.get(key, {})
    return dict(
        ok=True, asset=key, interval=interval,
        score=round(score, 2), abs_score=round(a, 2),
        threshold=th, qualifies=qualifies, direction=direction,
        direction_fa=("خرید" if direction > 0 else
                      "فروش" if direction < 0 else "بدون سیگنال"),
        gap=round(th - a, 2) if not qualifies else 0.0,
        percentile=_pctile(key, a),
        target_r=cfg["target_r"],
        parts=s.get("parts") or {},
        bar_date=str(df.index[-1])[:10],
        price=round(raw_price * sc, 2),
        evidence=dict(n=cfg["n"], win_rate=cfg["win_rate"],
                      avg_r=cfg["avg_r"], t_stat=cfg["t_stat"]),
        frequency=dict(hit_rate_pct=dist.get("hit_rate"),
                       every_days=dist.get("every_days")),
        note_fa=(
            "این عدد از موتوری می آید که روی ۷۷۳ روز تاریخی "
            "اعتبارسنجی شده و فقط از کندل استفاده می کند. با امتیاز "
            "کارت ایجنت یکی نیست و نباید با هم مقایسه شوند."),
        beginner_fa=(
            "فقط وقتی قدرمطلق امتیاز از %d رد شود سیگنال معتبر است. "
            "در %d روز گذشته این اتفاق حدود هر %s روز یک بار افتاده. "
            "بقیه روزها یعنی «کاری نکن» — که خودش یک تصمیم است."
            % (int(th), 600, dist.get("every_days", "؟"))),
    )


def cached(asset: str = "US30", interval: str = "1d",
           force: bool = False) -> Dict:
    """نسخه کش شده — محاسبه حدود ۰.۴ ثانیه است ولی روی ۰.۱ هسته
    CPU رندر همان هم ارزش کش کردن دارد."""
    k = "%s:%s" % (asset, interval)
    now = time.time()
    with _lock:
        hit = _cache.get(k)
        if hit and not force and (now - hit["at"]) < _TTL:
            out = dict(hit["data"])
            out["age_sec"] = round(now - hit["at"], 1)
            return out
    data = compute(asset, interval)
    if data.get("ok"):
        with _lock:
            _cache[k] = dict(at=now, data=data)
    data = dict(data)
    data["age_sec"] = 0.0
    return data
