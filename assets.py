"""
================================================================
لایه پروفایل دارایی (Asset Profile Layer)
================================================================
هر دارایی قوانین خودش را دارد. این ماژول مشخص می کند که برای هر
نماد:
  • داده تاریخی از کجا بیاید (کندل)
  • قیمت زنده از کجا بیاید (اسپات واقعی)
  • زنجیره آپشن از کدام نماد خوانده شود
  • کدام دارایی ها برای همبستگی و واگرایی SMT مقایسه شوند
  • قوانین وتو (دلار / VIX) چطور عمل کنند
  • ساعت معاملاتی چیست
  • ضریب نمایش چند است

دلیل وجود این فایل: قوانینی که برای داوجونز درست بودند برای طلا
غلط هستند. مثلا «VIX بالا رفت پس خرید ممنوع» برای سهام منطقی است
ولی طلا دارایی پناهگاه امن است و با ترس بالا می رود.
================================================================
"""
from __future__ import annotations

import json
import urllib.request
from typing import Dict, List, Optional

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0 Safari/537.36"}

_spot_cache: Dict[str, tuple] = {}


# ================================================================ پروفایل ها
PROFILES: Dict[str, Dict] = {

    # ------------------------------------------------------------ داوجونز
    "US30": dict(
        key="US30",
        name="داوجونز",
        name_full="شاخص داوجونز صنعتی (US30)",
        emoji="📈",
        asset_class="equity_index",

        # --- داده ---
        candle_symbol="DIA",          # ETF داوجونز — کندل تاریخی
        options_symbol="DIA",         # زنجیره آپشن
        spot_source=None,             # اسپات جدا ندارد
        display_scale=100.0,          # DIA × 100 ≈ US30
        basis_mode="scale",           # فقط ضرب ساده
        unit="واحد",
        decimals=0,

        # --- بین بازاری ---
        intermarket={
            "DX-Y.NYB": dict(name="شاخص دلار (DXY)", key="dxy", expect=-1),
            "^TNX":     dict(name="بازده ۱۰ ساله (US10Y)", key="us10y", expect=-1),
            "^GSPC":    dict(name="اس اند پی ۵۰۰ (SPX)", key="spx", expect=+1),
            "^NDX":     dict(name="نزدک ۱۰۰ (NDX)", key="ndx", expect=+1),
            "^VIX":     dict(name="شاخص ترس (VIX)", key="vix", expect=-1),
        },
        smt_peers=["^GSPC", "^NDX"],
        smt_peer_names={"^GSPC": "اس اند پی ۵۰۰", "^NDX": "نزدک ۱۰۰"},

        # --- قوانین وتو ---
        rules=dict(
            dxy_veto=True,            # دلار قوی = وتوی خرید
            dxy_veto_strength=1.0,
            dxy_corr_ref=-0.276,      # همبستگی اندازه گیری شده
            # ممیزی ۲۰۲۶-۰۹-۱۹ (۱۰ سال، ۲۵۱۱ روز): VIX>22.9 → بازده
            # ۵ روزه داوجونز +۰.۶۳٪ در برابر +۰.۱۹٪ عادی، p=0.010.
            # یعنی ترس بالا تاریخا نقطه خرید بوده نه وتو. وتو برداشته شد؛
            # فقط به عنوان هشدار نوسان باقی می ماند.
            vix_veto_long=False,
            # ممیزی ۲۰۲۶-۰۹-۱۹ با بازه اطمینان بوت‌استرپ:
            # RSI روی داوجونز ۱ساعته IC=-0.036 [-0.067,-0.006] ← بازگشتی
            momentum_sign=-1,
            vix_safe_haven=False,
            yields_veto=True,
            news_blackout=True,
        ),

        # --- ساعت بازار ---
        hours=dict(
            mode="equity",            # ۱۷:۰۰ تا ۲۳:۳۰ تهران
            nearly_24h=False,
            note="بازار سهام آمریکا — فقط در ساعات رسمی",
        ),
        intervals=["5m", "15m", "30m", "1h", "1d"],
        news_symbols=["DIA", "^DJI"],
    ),

    # ------------------------------------------------------------ طلا
    "XAUUSD": dict(
        key="XAUUSD",
        name="طلا",
        name_full="طلای جهانی اسپات (XAU/USD)",
        emoji="🥇",
        asset_class="commodity_metal",

        # --- داده ---
        # یاهو اسپات XAUUSD ندارد. فیوچرز GC=F تاریخچه کامل دارد و
        # همبستگی اش با اسپات عملا ۱ است. قیمت نهایی با بیسیس زنده
        # به اسپات واقعی تبدیل می شود.
        candle_symbol="GC=F",
        options_symbol="GLD",         # GC=F هیچ سررسیدی ندارد → از GLD
        spot_source="xauusd",         # فید اسپات واقعی بدون کلید
        display_scale=1.0,
        basis_mode="spot_basis",      # تصحیح اختلاف فیوچرز/اسپات
        unit="دلار",
        decimals=2,

        # --- بین بازاری ---
        intermarket={
            "DX-Y.NYB": dict(name="شاخص دلار (DXY)", key="dxy", expect=-1),
            "^TNX":     dict(name="بازده ۱۰ ساله (US10Y)", key="us10y", expect=-1),
            "SLV":      dict(name="نقره (Silver)", key="slv", expect=+1),
            "PPLT":     dict(name="پلاتین (Platinum)", key="pplt", expect=+1),
            "^VIX":     dict(name="شاخص ترس (VIX)", key="vix", expect=+1),  # پناهگاه امن
        },
        smt_peers=["SLV", "PPLT"],
        smt_peer_names={"SLV": "نقره", "PPLT": "پلاتین"},

        # --- قوانین وتو ---
        rules=dict(
            dxy_veto=True,
            dxy_veto_strength=1.55,   # |-0.433| / |-0.276| ≈ ۱.۵۷ → دلار مهم تر
            dxy_corr_ref=-0.433,
            vix_veto_long=False,      # ⚠ طلا پناهگاه امن است — ترس بالا مانع خرید نیست
            # RSI روی طلای ۱ساعته IC=+0.055 [+0.034,+0.073] ← ادامه‌دهنده
            momentum_sign=+1,
            # ⚠ ممیزی ۱۰ ساله: VIX>25 → طلا +۰.۲۹٪ در برابر +۰.۲۶٪ عادی،
            # p=0.89. VIX>30 هم p=0.13. اثر «پناهگاه» اثبات نشد.
            # مقدار True نگه داشته شد چون فقط وتو را برمی‌دارد (محافظه‌کارانه)
            # ولی هیچ امتیاز مثبتی تولید نمی‌کند.
            vix_safe_haven=True,
            yields_veto=True,
            news_blackout=True,
        ),

        # --- ساعت بازار ---
        hours=dict(
            mode="metal",             # تقریبا ۲۳ ساعته
            nearly_24h=True,
            note="طلا تقریبا ۲۴ ساعته معامله می شود (یکشنبه ۰۱:۰۰ تا جمعه ۲۳:۰۰ تهران)",
        ),
        intervals=["5m", "15m", "30m", "1h", "1d"],
        news_symbols=["GLD", "GC=F"],
    ),
}

DEFAULT_ASSET = "US30"

# نگاشت نام های مستعار → کلید پروفایل
_ALIASES = {
    "us30": "US30", "dia": "US30", "dow": "US30", "dji": "US30",
    "^dji": "US30", "داوجونز": "US30", "ym=f": "US30",
    "xauusd": "XAUUSD", "xau": "XAUUSD", "gold": "XAUUSD",
    "gc=f": "XAUUSD", "gld": "XAUUSD", "طلا": "XAUUSD",
}


def resolve(asset: Optional[str]) -> str:
    """نام ورودی کاربر را به کلید پروفایل تبدیل می کند."""
    if not asset:
        return DEFAULT_ASSET
    a = str(asset).strip().lower()
    return _ALIASES.get(a, DEFAULT_ASSET)


def profile(asset: Optional[str] = None) -> Dict:
    """پروفایل کامل یک دارایی."""
    return PROFILES[resolve(asset)]


def candle_symbol(asset: Optional[str] = None) -> str:
    return profile(asset)["candle_symbol"]


def options_symbol(asset: Optional[str] = None) -> str:
    return profile(asset)["options_symbol"]


def list_assets() -> List[Dict]:
    """فهرست کوتاه برای سوییچ داشبورد."""
    return [dict(key=p["key"], name=p["name"], emoji=p["emoji"],
                 full=p["name_full"], unit=p["unit"])
            for p in PROFILES.values()]


# ================================================================ اسپات زنده
def _get_json(url: str, timeout: float = 8.0):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def spot_swissquote() -> Dict:
    """قیمت زنده XAUUSD از فید عمومی سوییس کوت — bid/ask واقعی."""
    url = ("https://forex-data-feed.swissquote.com/public-quotes/"
           "bboquotes/instrument/XAU/USD")
    j = _get_json(url)
    prof = j[0]["spreadProfilePrices"][0]
    bid = float(prof["bid"])
    ask = float(prof["ask"])
    return dict(ok=True, price=(bid + ask) / 2.0, bid=bid, ask=ask,
                spread=round(ask - bid, 3),
                source="Swissquote", source_tier="real")


def spot_goldapi() -> Dict:
    """قیمت زنده XAUUSD از gold-api — منبع پشتیبان."""
    j = _get_json("https://api.gold-api.com/price/XAU")
    return dict(ok=True, price=float(j["price"]), bid=None, ask=None,
                spread=None, updated=j.get("updatedAt"),
                source="gold-api.com", source_tier="real")


def live_spot(asset: Optional[str] = None, ttl: float = 3.0) -> Dict:
    """
    قیمت اسپات واقعی با آبشار منبع:
      ۱) سوییس کوت (تیک به تیک + bid/ask)
      ۲) gold-api (پشتیبان)
    """
    import time as _t
    p = profile(asset)
    src = p.get("spot_source")
    if not src:
        return dict(ok=False, error="این دارایی فید اسپات جدا ندارد")

    hit = _spot_cache.get(src)
    if hit and (_t.time() - hit[0]) < ttl:
        return hit[1]

    errs = []
    for fn in (spot_swissquote, spot_goldapi):
        try:
            out = fn()
            if out.get("ok") and out.get("price", 0) > 0:
                out["fetched"] = _t.time()
                _spot_cache[src] = (_t.time(), out)
                return out
        except Exception as e:
            errs.append(f"{fn.__name__}: {e}")
    return dict(ok=False, error=" | ".join(errs) or "هیچ منبعی پاسخ نداد")


# ================================================================ بیسیس
_basis_cache: Dict[str, tuple] = {}


def basis(asset: Optional[str] = None, futures_price: Optional[float] = None,
          ttl: float = 60.0) -> Dict:
    """
    اختلاف بین قیمت فیوچرز و اسپات واقعی.

    فیوچرز طلا همیشه کمی بالاتر از اسپات معامله می شود (هزینه نگهداری
    و بهره). این تابع آن اختلاف را اندازه می گیرد تا بتوانیم تحلیل را
    روی فیوچرز انجام دهیم ولی عدد نهایی را اسپات واقعی نشان دهیم.
    """
    import time as _t
    p = profile(asset)
    if p["basis_mode"] != "spot_basis":
        return dict(ok=False, mode=p["basis_mode"], factor=p["display_scale"],
                    offset=0.0)

    key = p["key"]
    hit = _basis_cache.get(key)
    if hit and (_t.time() - hit[0]) < ttl and futures_price is None:
        return hit[1]

    sp = live_spot(asset)
    if not sp.get("ok"):
        return dict(ok=False, error=sp.get("error"), factor=1.0, offset=0.0,
                    mode="spot_basis")

    spot = float(sp["price"])
    fut = futures_price
    if fut is None:
        try:
            import yfinance as yf
            h = yf.Ticker(p["candle_symbol"]).history(period="5d", interval="1h")
            fut = float(h["Close"].iloc[-1]) if not h.empty else None
        except Exception:
            fut = None

    if not fut or fut <= 0:
        return dict(ok=False, error="قیمت فیوچرز نیامد", factor=1.0,
                    offset=0.0, mode="spot_basis")

    out = dict(
        ok=True, mode="spot_basis",
        spot=round(spot, 2), futures=round(fut, 2),
        offset=round(spot - fut, 3),            # افزودنی
        factor=round(spot / fut, 8),            # ضربی (دقیق تر برای سطوح)
        pct=round((fut - spot) / spot * 100, 4),
        bid=sp.get("bid"), ask=sp.get("ask"), spread=sp.get("spread"),
        source=sp.get("source"), source_tier="real",
        note=f"فیوچرز {p['candle_symbol']} حدود {abs((fut-spot)/spot*100):.2f}٪ "
             f"{'بالاتر' if fut > spot else 'پایین تر'} از اسپات است؛ "
             f"همه قیمت ها به اسپات تبدیل شدند.",
    )
    _basis_cache[key] = (_t.time(), out)
    return out


def to_display(value: Optional[float], asset: Optional[str] = None,
               bas: Optional[Dict] = None) -> Optional[float]:
    """
    تبدیل یک قیمت خام (از کندل) به عددی که به کاربر نشان داده می شود.
      • داوجونز : × ۱۰۰
      • طلا     : × ضریب بیسیس (فیوچرز → اسپات)
    """
    if value is None:
        return None
    p = profile(asset)
    if p["basis_mode"] == "scale":
        return float(value) * p["display_scale"]
    b = bas if bas is not None else basis(asset)
    if b.get("ok"):
        return float(value) * float(b["factor"])
    return float(value)


def fmt(value: Optional[float], asset: Optional[str] = None) -> str:
    """قالب بندی عدد با تعداد رقم اعشار مناسب همان دارایی."""
    if value is None:
        return "—"
    d = profile(asset)["decimals"]
    return f"{value:,.{d}f}"


# ================================================================ ساعت بازار
def market_state(asset: Optional[str] = None) -> Dict:
    """
    وضعیت باز/بسته بودن بازار همان دارایی.
      • سهام : تقویم رسمی NYSE
      • فلز  : یکشنبه ۰۱:۰۰ تا جمعه ۲۳:۰۰ تهران با وقفه روزانه کوتاه
    """
    from datetime import datetime, timedelta, timezone as _tz
    p = profile(asset)
    if p["hours"]["mode"] != "metal":
        try:
            import market_hours as mh
            return mh.market_status()
        except Exception as e:
            return dict(ok=False, error=str(e))

    TEH = _tz(timedelta(hours=3, minutes=30))
    now = datetime.now(TEH)
    wd = now.weekday()               # دوشنبه=۰ ... یکشنبه=۶
    hour = now.hour + now.minute / 60.0

    # بازار فلزات: یکشنبه ۰۱:۰۰ باز → جمعه ۲۳:۰۰ بسته (وقت تهران)
    if wd == 5:                                    # شنبه
        is_open, why = False, "آخر هفته — بازار جهانی طلا بسته است"
    elif wd == 6 and hour < 1.0:                   # یکشنبه قبل از ۰۱:۰۰
        is_open, why = False, "بازار هنوز باز نشده (یکشنبه ۰۱:۰۰ تهران)"
    elif wd == 4 and hour >= 23.0:                 # جمعه بعد از ۲۳:۰۰
        is_open, why = False, "بازار بسته شد (جمعه ۲۳:۰۰ تهران)"
    elif 0.0 <= hour < 1.0 and wd not in (5, 6):   # وقفه روزانه
        is_open, why = False, "وقفه روزانه بازار (۰۰:۰۰ تا ۰۱:۰۰ تهران)"
    else:
        is_open, why = True, "بازار طلا باز است"

    return dict(ok=True, is_open=is_open, market_open=is_open,
                status_fa="باز" if is_open else "بسته",
                reason=why, tehran_time=now.strftime("%H:%M"),
                weekday=wd, nearly_24h=True,
                note=p["hours"]["note"])


# ================================================================ تست
if __name__ == "__main__":
    import sys
    a = sys.argv[1] if len(sys.argv) > 1 else "XAUUSD"
    p = profile(a)
    print(f"پروفایل: {p['emoji']} {p['name_full']}")
    print(f"  کندل      : {p['candle_symbol']}")
    print(f"  آپشن      : {p['options_symbol']}")
    print(f"  همتا SMT  : {p['smt_peers']}")
    print(f"  وتوی VIX  : {p['rules']['vix_veto_long']}")
    print(f"  پناهگاه   : {p['rules']['vix_safe_haven']}")
    print()
    if p.get("spot_source"):
        s = live_spot(a)
        print("اسپات زنده:", s)
        b = basis(a)
        print("بیسیس     :", b)
        print("نمونه تبدیل: 4434.80 →", fmt(to_display(4434.80, a), a))
    print("وضعیت بازار:", market_state(a))
