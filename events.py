# -*- coding: utf-8 -*-
"""رویدادهای بازار — اعلام واقعیت، نه پیش بینی.

چرا این ماژول هست
─────────────────
هر آزمونی که روی سیگنال جهت دار زدیم شکست خورد:

    اسکلپ ۵ دقیقه داوجونز   ۰ از ۳۲ حالت معنادار (۷۷۳ روز)
    ستاپ لندن → نیویورک     ۰ از ۱۶ ترکیب سودده (داوجونز)
    همان روی طلا            ۰ از ۶

ولی چیزی که کار کرد، سطوح بود. کاربر خودش گفت:
«امتیاز می گفت long ضعیف است ولی POC و خلأ نقدینگی درست بود».

پس این ماژول «بخر / بفروش» نمی گوید. فقط می گوید همین الان
چه اتفاقی افتاد:

    سقف جلسه جارو شد
    قیمت وارد خلأ پر نشده شد
    قیمت به POC نزدیک شد

اینها واقعیت اند. تصمیم با معامله گر است.

حافظه
─────
وضعیت در حافظه فرایند نگه داشته می شود. روی Render با هر
ری استارت پاک می شود — که اشکالی ندارد، چون رویدادها درون روزی
اند و ارزش ذخیره دائمی ندارند.
"""
from __future__ import annotations

import datetime as dt
import threading
import time
from typing import Dict, List, Optional

MAX_EVENTS = 40
NEAR_ATR_FRAC = 0.35          # «نزدیک» یعنی کمتر از این کسر از ATR جلسه
_lock = threading.Lock()

# به ازای هر دارایی: آخرین قیمت، سمت هر سطح، رویدادهای اعلام شده
_state: Dict[str, Dict] = {}


def _blank() -> Dict:
    return dict(last_price=None, side={}, near={}, log=[], updated=0.0)


def _fa_now() -> str:
    t = dt.datetime.now(dt.UTC) + dt.timedelta(hours=3, minutes=30)
    return t.strftime("%H:%M")


def _push(st: Dict, kind: str, icon: str, text: str, level=None,
          name: str = "") -> None:
    st["log"].insert(0, dict(
        at=_fa_now(), ts=time.time(), kind=kind, icon=icon,
        text=text, level=level, name=name))
    del st["log"][MAX_EVENTS:]


def update(board: Dict, asset: str = "US30") -> Dict:
    """نقشه تازه را می گیرد و رویدادهای جدید را تشخیص می دهد.

    board خروجی board.build() است.
    """
    asset = (asset or "US30").upper()
    if not board or not board.get("ok"):
        return dict(ok=False, error="نقشه در دسترس نیست")

    price = board.get("price")
    levels = board.get("levels") or []
    if price is None:
        return dict(ok=False, error="قیمت در دسترس نیست")

    dec = 1 if asset == "US30" else 2
    # مقیاس «نزدیکی»: از پراکندگی خود سطوح برآورد می شود تا برای
    # طلا (۴۱۷۰ دلار) و داوجونز (۵۱۸۰۰ واحد) هر دو معنی بدهد.
    spans = [abs(l["level"] - price) for l in levels if l.get("level")]
    scale = (sorted(spans)[len(spans) // 2] if spans else price * 0.001)
    near_gap = max(scale * NEAR_ATR_FRAC, price * 0.0002)

    with _lock:
        st = _state.setdefault(asset, _blank())
        prev_price = st["last_price"]
        fresh: List[Dict] = []

        for L in levels:
            lv = L.get("level")
            nm = L.get("name") or ""
            if lv is None or not nm:
                continue
            key = "%s@%.4f" % (nm, lv)
            side_now = "above" if price >= lv else "below"
            side_old = st["side"].get(key)

            # ── عبور از سطح ──
            if side_old and side_old != side_now:
                arrow = "بالا" if side_now == "above" else "پایین"
                icon = "⚡"
                kind = "cross"
                txt = "%s (%s) رد شد — قیمت حالا %s آن است" % (
                    nm, format(lv, ",.%df" % dec), arrow)
                _push(st, kind, icon, txt, lv, nm)
                fresh.append(st["log"][0])
            st["side"][key] = side_now

            # ── نزدیک شدن ──
            d = abs(lv - price)
            was_near = st["near"].get(key, False)
            is_near = d <= near_gap
            if is_near and not was_near:
                _push(st, "near", "🎯",
                      "قیمت به %s (%s) نزدیک شد — فاصله %s" % (
                          nm, format(lv, ",.%df" % dec),
                          format(d, ",.%df" % dec)), lv, nm)
                fresh.append(st["log"][0])
            st["near"][key] = is_near

            # ── ورود به خلأ پر نشده ──
            if L.get("kind") == "خلأ" and L.get("top") is not None \
                    and L.get("bottom") is not None:
                inside = L["bottom"] <= price <= L["top"]
                k2 = key + "#in"
                if inside and not st["near"].get(k2):
                    _push(st, "gap", "🕳",
                          "قیمت وارد %s شد (%s تا %s)" % (
                              nm, format(L["bottom"], ",.%df" % dec),
                              format(L["top"], ",.%df" % dec)), lv, nm)
                    fresh.append(st["log"][0])
                st["near"][k2] = inside

        st["last_price"] = price
        st["updated"] = time.time()
        log = list(st["log"])

    return dict(ok=True, asset=asset, price=price,
                new_count=len(fresh), events=log,
                near_gap=round(near_gap, dec),
                note=("این‌ها رویدادهای واقعی‌اند، نه سیگنال. "
                      "هیچ جهتی پیشنهاد نمی‌شود."))


def get(asset: str = "US30") -> Dict:
    asset = (asset or "US30").upper()
    with _lock:
        st = _state.get(asset)
        if not st:
            return dict(ok=True, asset=asset, events=[], note="هنوز رویدادی ثبت نشده")
        return dict(ok=True, asset=asset, events=list(st["log"]),
                    price=st["last_price"],
                    age_sec=round(time.time() - st["updated"], 1))


def reset(asset: Optional[str] = None) -> None:
    with _lock:
        if asset:
            _state.pop(asset.upper(), None)
        else:
            _state.clear()
