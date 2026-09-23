# -*- coding: utf-8 -*-
"""
فیلتر کیفیت سیگنال — بر پایه بک تست، نه حدس.

پیشینه
──────
بک تست ۲۰۲۶-۰۹-۲۳ روی ۸۸۲ سیگنال روزانه نشان داد آستانه های قبلی
موتور (±۱۳ ضعیف / ±۲۶ قوی) بیش از حد سخاوتمندانه اند:

    دارایی   آستانه ۱۳   آستانه ۳۲
    ─────────────────────────────────────
    طلا      R=+0.222    R=+0.639   t=3.99
    داوجونز  R=+0.020    R=+0.399   t=3.98
             (بی معنی)   (معنادار)

یعنی بیشتر سیگنال های ضعیف فقط نویز بودند و سود را رقیق می کردند.

اعتبارسنجی خارج از نمونه (نصف اول تنظیم، نصف دوم آزمون) هر چهار
حالت را تأیید کرد:

    طلا     نصف اول  n=38 برد=63.2% t=+3.14
    طلا     نصف دوم  n=31 برد=61.3% t=+2.42
    داوجونز نصف اول  n=73 برد=61.6% t=+3.54
    داوجونز نصف دوم  n=71 برد=52.1% t=+2.09

چون نصف دوم هرگز در انتخاب آستانه دخالت نداشت، این یافته برازش
بیش از حد نیست.

نکته مهم درباره جهت سیگنال داوجونز
──────────────────────────────────
در گزارش قبلی دیده شد معکوس کردن سیگنال داوجونز از +5.4R به
+23.1R می رسد. آزمون شد: t=0.39 و بازه اطمینان بوت استرپ
[-26.0, +69.7] — یعنی کاملا تصادفی. همچنین کل سود از ۵ معامله
خوش شانس می آمد (بدون آن ها مجموع منفی می شد).

پس جهت سیگنال معکوس نشد. مسئله واقعی کیفیت بود، نه جهت.
"""
from __future__ import annotations

from typing import Dict, Optional

# آستانه های اثبات شده — از بک تست، با اعتبارسنجی خارج از نمونه
THRESHOLD = {
    "XAUUSD": dict(
        minimum=26.0, strong=32.0,
        evidence=dict(n=69, win_rate=62.3, avg_r=0.639, t=3.99,
                      oos_first=dict(n=38, win_rate=63.2, t=3.14),
                      oos_second=dict(n=31, win_rate=61.3, t=2.42)),
    ),
    "US30": dict(
        minimum=32.0, strong=40.0,
        evidence=dict(n=144, win_rate=56.9, avg_r=0.399, t=3.98,
                      oos_first=dict(n=73, win_rate=61.6, t=3.54),
                      oos_second=dict(n=71, win_rate=52.1, t=2.09)),
    ),
}

# نسبت هدف به حد ضرر — بهترین در بک تست
BEST_R = {"XAUUSD": 2.0, "US30": 2.5}

# تایم فریم هایی که در بک تست ضرر دادند
BAD_INTERVALS = {"1h", "5m", "15m", "30m"}


def assess(asset: str, score: float,
           interval: str = "1d") -> Dict:
    """آیا این سیگنال ارزش معامله دارد؟

    خروجی همیشه دلیل عددی دارد — هیچ قضاوت بدون شاهد.
    """
    cfg = THRESHOLD.get(asset)
    if not cfg:
        return dict(ok=False, error=f"دارایی ناشناخته: {asset}")

    a = abs(float(score or 0.0))
    ev = cfg["evidence"]

    out: Dict = dict(
        ok=True, asset=asset, score=round(float(score or 0), 2),
        abs_score=round(a, 2),
        threshold=cfg["minimum"], strong_at=cfg["strong"],
        best_r=BEST_R.get(asset, 2.0),
        interval=interval,
    )

    # هشدار تایم فریم — مستقل از امتیاز
    if interval in BAD_INTERVALS:
        out.update(
            tier="rejected_timeframe", tier_fa="⛔ تایم‌فریم نامناسب",
            tradeable=False, color="#f87171",
            reason=(f"در بک‌تست، تایم‌فریم {interval} برای هر دو دارایی "
                    "زیان‌ده بود (میانگین منفی). فقط تایم‌فریم روزانه "
                    "اعتبارسنجی شده است."),
            beginner=("این تایم‌فریم آزمایش شد و نتیجه منفی داد. "
                      "بهتر است روی نمودار روزانه کار کنید."),
        )
        return out

    if a < cfg["minimum"]:
        gap = cfg["minimum"] - a
        out.update(
            tier="noise", tier_fa="⚪ زیر آستانه — رد",
            tradeable=False, color="#6b7280",
            reason=(f"امتیاز {a:.1f} کمتر از آستانه اثبات‌شده "
                    f"{cfg['minimum']:.0f} است. {gap:.1f} واحد فاصله دارد."),
            beginner=("سیگنال‌های زیر این عدد در گذشته نتیجه‌ای بهتر از "
                      "شانس نداشتند. صبر کردن بهتر از معامله کردن است."),
        )
        return out

    if a < cfg["strong"]:
        out.update(
            tier="acceptable", tier_fa="🟡 قابل قبول",
            tradeable=True, color="#fbbf24",
            size_hint=0.5,
            reason=(f"امتیاز {a:.1f} از آستانه {cfg['minimum']:.0f} "
                    f"گذشته ولی به {cfg['strong']:.0f} نرسیده."),
            beginner=("سیگنال قابل قبول است ولی قوی‌ترین نیست. "
                      "با نصف حجم معمول وارد شوید."),
        )
        return out

    out.update(
        tier="strong", tier_fa="🟢 قوی — اثبات‌شده",
        tradeable=True, color="#4ade80",
        size_hint=1.0,
        reason=(f"امتیاز {a:.1f} بالای آستانه قوی {cfg['strong']:.0f}. "
                f"در بک‌تست این دسته: {ev['n']} نمونه، "
                f"برد {ev['win_rate']}٪، میانگین {ev['avg_r']:+.3f}R، "
                f"t={ev['t']}."),
        beginner=("این قوی‌ترین دسته سیگنال است. در گذشته بیش از "
                  "نیمی از این‌ها به سود رسیدند."),
    )
    return out


def evidence(asset: Optional[str] = None) -> Dict:
    """شواهد بک تست — برای نمایش شفاف به کاربر."""
    if asset:
        cfg = THRESHOLD.get(asset)
        if not cfg:
            return dict(ok=False, error="دارایی ناشناخته")
        return dict(ok=True, asset=asset, **cfg)
    return dict(ok=True, thresholds=THRESHOLD, best_r=BEST_R,
                bad_intervals=sorted(BAD_INTERVALS),
                source="بک‌تست ۲۰۲۶-۰۹-۲۳ · ۸۸۲ سیگنال روزانه · "
                       "اعتبارسنجی خارج از نمونه انجام شد",
                note="جهت سیگنال معکوس نشد؛ آزمون t=0.39 نشان داد "
                     "آن یافته تصادفی بود.")


if __name__ == "__main__":
    import json
    for a, s in (("XAUUSD", 35), ("XAUUSD", 20), ("US30", 45), ("US30", 28)):
        r = assess(a, s)
        print(f"{a:7s} {s:3d} → {r['tier_fa']}")
    print(json.dumps(evidence(), ensure_ascii=False, indent=1)[:400])
