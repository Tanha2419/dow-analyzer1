# -*- coding: utf-8 -*-
"""
قیمت نقدی داوجونز — منطبق با پلتفرم های معاملاتی (ترِندو و مشابه).

مسئله
─────
سه عدد متفاوت برای «داوجونز» وجود دارد:

    ^DJI   شاخص نقدی    ← پلتفرم های معاملاتی این را نشان می دهند
    YM=F   فیوچرز        ← همیشه گران تر، ولی تقریبا ۲۴ ساعته
    DIA    ETF           ← ×۱۰۰ تقریبا برابر شاخص

اندازه گیری ۲۰۲۶-۰۹-۲۳ در برابر عدد واقعی ترِندو (۵۱٬۹۱۶):

    ^DJI     51,864   اختلاف  −52   ✅
    DIA×100  51,800   اختلاف −116
    YM=F     52,325   اختلاف +409   ❌

پس مرجع درست ^DJI است، نه فیوچرز.

اما ^DJI فقط ۶.۵ ساعت در روز به روز می شود (۱۶:۳۰–۲۳:۰۰ تهران).
کاربری که در جلسه لندن معامله می کند، عددی یخ زده می بیند.

راه حل
──────
وقتی بورس بسته است، از فیوچرز استفاده می کنیم و «پایه» را کم می کنیم:

    تخمین نقدی = فیوچرز الان − پایه

پایه = اختلاف فیوچرز و شاخص در آخرین لحظه ای که هر دو باز بودند.
اندازه گیری نشان داد پایه ثابت نیست (۱۲ تا ۴۲۶ در ۲۰ روز)، پس
هر بار زنده محاسبه می شود — هرگز عدد ثابت.

اعتبارسنجی همین روش روی داده واقعی:
    فیوچرز 52,333 − پایه 419 = 51,914   در برابر ترِندو 51,916
    اختلاف: ۲ واحد (۰٫۰۰۴٪)

هیچ عددی بدون برچسب منبع برنمی گردد. اگر منبعی در دسترس نباشد،
مقدار None و دلیل آن برگردانده می شود — نه عدد حدسی.
"""
from __future__ import annotations

import datetime as _dt
import json as _json
import time as _time
import urllib.request as _url
from typing import Dict, Optional

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"

_CASH = "^DJI"        # شاخص نقدی — مرجع پلتفرم های معاملاتی
_FUT = "YM=F"         # فیوچرز E-mini Dow — زنده در ساعات بسته بودن بورس

_cache: Dict[str, tuple] = {}


def _get(symbol: str, interval: str = "1m", rng: str = "1d",
         prepost: bool = True, timeout: float = 12.0) -> Optional[dict]:
    """یک فراخوانی به نمودار یاهو. در صورت خطا None."""
    q = f"?interval={interval}&range={rng}"
    if prepost:
        q += "&includePrePost=true"
    try:
        req = _url.Request(_CHART.format(sym=symbol) + q, headers=_UA)
        with _url.urlopen(req, timeout=timeout) as r:
            d = _json.load(r)
        res = (d.get("chart") or {}).get("result")
        return res[0] if res else None
    except Exception:
        return None


def _last_tick(res: dict) -> tuple:
    """(قیمت, زمان epoch) از آخرین کندل معتبر."""
    if not res:
        return None, None
    meta = res.get("meta") or {}
    px = meta.get("regularMarketPrice")
    ts = meta.get("regularMarketTime")

    # اگر متادیتا ناقص بود، از آخرین کندل غیرخالی بخوان
    if px is None:
        try:
            q = res["indicators"]["quote"][0]
            for t, c in zip(reversed(res["timestamp"]),
                            reversed(q["close"])):
                if c is not None:
                    return float(c), int(t)
        except Exception:
            return None, None
    return (float(px) if px is not None else None,
            int(ts) if ts else None)


def _fresh_minutes(ts: Optional[int]) -> Optional[float]:
    if not ts:
        return None
    return (_time.time() - ts) / 60.0


def compute_basis(max_age_hours: float = 36.0) -> Dict:
    """
    پایه = فیوچرز − شاخص، در آخرین دقیقه ای که هر دو داده داشتند.

    از کندل های یک دقیقه ای دو روز اخیر استفاده می کند و آخرین
    زمانِ مشترک را پیدا می کند — یعنی لحظه ای که بورس هنوز باز بود.
    """
    fut = _get(_FUT, "1m", "2d", prepost=True)
    cash = _get(_CASH, "1m", "2d", prepost=True)
    if not fut or not cash:
        return dict(ok=False, error="داده یک دقیقه ای در دسترس نیست")

    def _map(res):
        out = {}
        try:
            q = res["indicators"]["quote"][0]
            for t, c in zip(res["timestamp"], q["close"]):
                if c is not None:
                    out[int(t)] = float(c)
        except Exception:
            pass
        return out

    fm, cm = _map(fut), _map(cash)
    common = sorted(set(fm) & set(cm))
    if not common:
        return dict(ok=False, error="لحظه مشترکی بین فیوچرز و شاخص نبود")

    t = common[-1]
    age_h = (_time.time() - t) / 3600
    if age_h > max_age_hours:
        return dict(ok=False, error=f"آخرین همپوشانی {age_h:.1f} ساعت پیش بود",
                    stale=True, age_hours=round(age_h, 2))

    basis = fm[t] - cm[t]
    return dict(
        ok=True,
        basis=round(basis, 2),
        futures_at=round(fm[t], 2),
        cash_at=round(cm[t], 2),
        at_iso=_dt.datetime.fromtimestamp(t, _dt.timezone.utc)
                          .strftime("%Y-%m-%d %H:%M UTC"),
        age_hours=round(age_h, 2),
        samples=len(common),
    )


def cash_price(ttl: float = 20.0) -> Dict:
    """
    بهترین تخمین از قیمتی که پلتفرم معاملاتی نشان می دهد.

    خروجی همیشه دارای `source_fa` است تا کاربر بداند عدد از کجا آمده:
      • «شاخص نقدی» — بورس باز است، عدد مستقیم و دقیق
      • «از فیوچرز» — بورس بسته است، فیوچرز منهای پایه زنده
    """
    hit = _cache.get("cash")
    if hit and (_time.time() - hit[0]) < ttl:
        return hit[1]

    cash_res = _get(_CASH, "1m", "1d", prepost=True)
    cash_px, cash_ts = _last_tick(cash_res)
    cash_age = _fresh_minutes(cash_ts)

    fut_res = _get(_FUT, "1m", "1d", prepost=True)
    fut_px, fut_ts = _last_tick(fut_res)
    fut_age = _fresh_minutes(fut_ts)

    prev_close = None
    try:
        prev_close = float((cash_res.get("meta") or {}).get("chartPreviousClose"))
    except Exception:
        pass

    out: Dict

    # حالت ۱ — شاخص تازه است (بورس باز): مستقیم و دقیق
    if cash_px and cash_age is not None and cash_age <= 15:
        out = dict(
            ok=True, price=round(cash_px, 2),
            mode="cash", source_fa="شاخص نقدی (^DJI)",
            symbol=_CASH, age_min=round(cash_age, 1),
            market_open=True, basis=None,
            accuracy_fa="دقیق — مستقیم از شاخص",
        )

    # حالت ۲ — بورس بسته: فیوچرز منهای پایه زنده
    elif fut_px and fut_age is not None and fut_age <= 30:
        b = compute_basis()
        if b.get("ok"):
            est = fut_px - b["basis"]
            out = dict(
                ok=True, price=round(est, 2),
                mode="futures_adjusted",
                source_fa=f"از فیوچرز (پایه {b['basis']:+,.0f})",
                symbol=_FUT, age_min=round(fut_age, 1),
                market_open=False,
                basis=b["basis"], basis_at=b.get("at_iso"),
                futures_raw=round(fut_px, 2),
                cash_last=round(cash_px, 2) if cash_px else None,
                cash_age_min=None if cash_age is None else round(cash_age, 1),
                accuracy_fa="تخمین — بورس بسته است، از فیوچرز محاسبه شد",
            )
        else:
            # پایه محاسبه نشد → عدد ساختگی نمی سازیم
            out = dict(
                ok=True, price=round(cash_px, 2) if cash_px else None,
                mode="cash_stale",
                source_fa="آخرین قیمت بسته شدن",
                symbol=_CASH,
                age_min=None if cash_age is None else round(cash_age, 1),
                market_open=False, basis=None,
                basis_error=b.get("error"),
                futures_raw=round(fut_px, 2),
                accuracy_fa="قدیمی — پایه محاسبه نشد",
            )

    # حالت ۳ — هیچ کدام تازه نیست (آخر هفته)
    else:
        out = dict(
            ok=bool(cash_px), price=round(cash_px, 2) if cash_px else None,
            mode="closed", source_fa="بسته شدن آخرین روز کاری",
            symbol=_CASH,
            age_min=None if cash_age is None else round(cash_age, 1),
            market_open=False, basis=None,
            accuracy_fa="بازار تعطیل — قیمت ثابت",
        )
        if not cash_px:
            out["error"] = "هیچ منبعی پاسخ نداد"

    if prev_close and out.get("price"):
        out["prev_close"] = round(prev_close, 2)
        out["change"] = round(out["price"] - prev_close, 2)
        out["change_pct"] = round((out["price"] / prev_close - 1) * 100, 3)

    out["checked_at"] = _dt.datetime.now(_dt.timezone.utc)\
        .strftime("%Y-%m-%d %H:%M:%S UTC")
    _cache["cash"] = (_time.time(), out)
    return out


def compare() -> Dict:
    """سه نما کنار هم — برای تشخیص عیب و شفافیت."""
    res = {}
    for sym in (_CASH, _FUT, "DIA"):
        r = _get(sym, "1m", "1d", prepost=True)
        px, ts = _last_tick(r)
        if px is None:
            res[sym] = dict(ok=False)
            continue
        res[sym] = dict(
            ok=True,
            raw=round(px, 2),
            as_index=round(px * 100, 2) if sym == "DIA" else round(px, 2),
            age_min=None if ts is None else round(_fresh_minutes(ts), 1),
        )
    res["recommended"] = cash_price()
    return res


if __name__ == "__main__":
    print(_json.dumps(cash_price(), ensure_ascii=False, indent=2))
    print(_json.dumps(compute_basis(), ensure_ascii=False, indent=2))
