#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
institutional.py — پنج لایه دید تریدر نهادی + مهندسی ویژگی

لایه ۱: وتوی بین بازاری  (DXY / US10Y / VIX / SMT)
لایه ۲: نقشه نقدینگی    (PDH/PDL/PWH/PWL، خوشه استاپ، انتظار Sweep)
لایه ۳: کلان و تقویم     (پنجره نامتقارن ۱۵ قبل / ۳۰ بعد، احتمال FedWatch)
لایه ۴: احساسات         (NLP خبری + جریان آپشن)
لایه ۵: زمان و فصلی      (Killzone + اثر روز هفته)

خروجی نهایی: بردار ویژگی عددی نرمال شده برای تغذیه ایجنت.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import assets as A

warnings.filterwarnings("ignore")

TEHRAN = timezone(timedelta(hours=3, minutes=30))


# ================================================================ ابزار
def _f(x, d=0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else d
    except Exception:
        return d


def _clip(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(np.clip(_f(v), lo, hi))


# ================================================================ لایه ۱: وتوی بین بازاری
DXY_BREAK_ATR = 0.6          # شکست مقاومت DXY بیش از این مقدار ATR = وتو
US10Y_JUMP_PCT = 2.0         # جهش بازده بیش از این درصد = فشار فروش
VIX_SPIKE_ABS = 20.0         # آستانه مطلق VIX
VIX_SPIKE_PCT = 8.0          # جهش درصدی VIX


def _levels(df: pd.DataFrame, lookback: int = 60) -> Tuple[float, float]:
    """مقاومت و حمایت اخیر بر اساس سقف/کف lookback کندل."""
    d = df.tail(lookback)
    return float(d["High"].max()), float(d["Low"].min())


def intermarket_veto(intermarket: Dict, ref_df: Optional[pd.DataFrame] = None,
                     asset: Optional[str] = None) -> Dict:
    """
    وتوی صریح سیگنال بر اساس شرایط بین بازاری — وابسته به پروفایل دارایی.

    تفاوت کلیدی طلا با سهام:
      • دلار برای طلا مهم تر است (همبستگی ‎-0.43‎ در برابر ‎-0.28‎)
      • VIX بالا برای سهام خطر است ولی طلا پناهگاه امن است
      • بازده بالا برای طلا هزینه فرصت است (نه فشار فروش سهام)
    """
    _prof = A.profile(asset)
    _rules = _prof["rules"]
    out = dict(ok=False, veto_long=False, veto_short=False,
               reasons=[], warnings=[], features={})
    if not intermarket or not intermarket.get("ok"):
        return out
    out["ok"] = True

    assets = {a.get("key") or a.get("symbol") or a.get("name"): a
              for a in intermarket.get("assets", [])}

    def find(*keys) -> Optional[Dict]:
        for k in keys:
            for kk, v in assets.items():
                if kk and k.lower() in str(kk).lower():
                    return v
            for a in intermarket.get("assets", []):
                if k.lower() in str(a.get("name", "")).lower() \
                        or k.lower() in str(a.get("symbol", "")).lower():
                    return a
        return None

    # --- DXY ---
    dxy = find("dxy", "دلار", "DX-Y")
    if dxy:
        chg = _f(dxy.get("chg_pct"))
        out["features"]["dxy_chg_pct"] = round(chg, 4)
        out["features"]["dxy_corr"] = round(_f(dxy.get("corr")), 4)
        breaking = False
        dxy_df = dxy.get("_df")
        if isinstance(dxy_df, pd.DataFrame) and len(dxy_df) > 20:
            res, _ = _levels(dxy_df.iloc[:-1], 60)
            last = float(dxy_df["Close"].iloc[-1])
            rngs = float((dxy_df["High"] - dxy_df["Low"]).tail(14).mean()) or 1e-9
            over = (last - res) / rngs
            out["features"]["dxy_break_atr"] = round(over, 3)
            breaking = over > 0
        _ds = float(_rules.get("dxy_veto_strength", 1.0))
        _t_veto = 0.35 / _ds          # طلا حساس تر → آستانه پایین تر
        _t_warn = 0.15 / _ds
        if _rules.get("dxy_veto") and (chg > _t_veto or breaking):
            out["veto_long"] = True
            out["reasons"].append(
                f"شاخص دلار در حال قدرت گرفتن ({chg:+.2f}٪) — وتوی سیگنال خرید"
                + (" (طلا به دلار حساس تر است)" if _ds > 1.2 else ""))
        elif chg > _t_warn:
            out["warnings"].append(f"شاخص دلار مثبت ({chg:+.2f}٪) — احتیاط در خرید")
        elif chg < -_t_veto:
            out["warnings"].append(f"ضعف دلار ({chg:+.2f}٪) — حامی خرید")

    # --- US10Y ---
    t10 = find("tnx", "۱۰ ساله", "10Y", "بازده")
    if t10:
        chg = _f(t10.get("chg_pct"))
        out["features"]["us10y_chg_pct"] = round(chg, 4)
        _gold = _rules.get("vix_safe_haven")
        _why = ("هزینه فرصت نگهداری طلا بالا می رود" if _gold
                else "فشار فروش بر سهام")
        if _rules.get("yields_veto") and chg >= US10Y_JUMP_PCT:
            out["veto_long"] = True
            out["reasons"].append(
                f"جهش بازده ۱۰ ساله ({chg:+.2f}٪) — {_why}")
        elif chg >= 1.0:
            out["warnings"].append(f"بازده ۱۰ ساله بالا رفته ({chg:+.2f}٪)")
        elif chg <= -1.5:
            out["warnings"].append(
                f"افت بازده ۱۰ ساله ({chg:+.2f}٪) — "
                + ("حامی طلا" if _gold else "حامی سهام"))

    # --- VIX ---
    vix = find("vix", "ترس")
    if vix:
        lvl = _f(vix.get("price"))
        chg = _f(vix.get("chg_pct"))
        out["features"]["vix_level"] = round(lvl, 3)
        out["features"]["vix_chg_pct"] = round(chg, 3)
        spike = (chg >= VIX_SPIKE_PCT) or (lvl >= VIX_SPIKE_ABS and chg > 3)
        out["features"]["vix_spike"] = 1 if spike else 0
        if spike:
            out["warnings"].append(
                f"جهش شاخص ترس (VIX {lvl:.1f}, {chg:+.1f}٪) — "
                "شکست ها احتمالا فیک هستند")
            out["breakout_unreliable"] = True
        if lvl >= 25:
            if _rules.get("vix_veto_long", True):
                out["veto_long"] = True
                out["reasons"].append(f"VIX بالای ۲۵ ({lvl:.1f}) — رژیم پرریسک")
            else:
                if _rules.get("vix_safe_haven"):
                    out["warnings"].append(
                        f"VIX بالای ۲۵ ({lvl:.1f}) — ترس بالا معمولا "
                        "تقاضای طلا را زیاد می کند (وتو اعمال نشد)")
                    out["features"]["vix_haven_bid"] = 1
                else:
                    out["warnings"].append(
                        f"VIX بالای ۲۵ ({lvl:.1f}) — نوسان زیاد است، حد ضرر "
                        "را پهن تر بگیرید. طبق داده ۱۰ ساله این سطح تاریخا "
                        "نقطه خرید بوده، پس وتو اعمال نشد")

    # --- SMT ---
    smt = intermarket.get("smt", []) or []
    out["features"]["smt_divergence"] = 0
    for s in smt:
        typ = str(s.get("type", ""))
        if typ.startswith("bear"):
            out["veto_long"] = True
            out["features"]["smt_divergence"] = -1
            out["reasons"].append("واگرایی SMT نزولی — " + str(s.get("text", ""))[:60])
        elif typ.startswith("bull"):
            out["veto_short"] = True
            out["features"]["smt_divergence"] = 1
            out["reasons"].append("واگرایی SMT صعودی — " + str(s.get("text", ""))[:60])

    rs = _f(intermarket.get("risk_score"))
    out["features"]["intermarket_risk"] = round(_clip(rs / 3.0), 4)
    if rs <= -2.0:
        out["veto_long"] = True
        out["reasons"].append(f"امتیاز ریسک بین بازاری {rs:+.2f} — جبهه مخالف صعود")
    elif rs >= 2.0:
        out["veto_short"] = True
        out["reasons"].append(f"امتیاز ریسک بین بازاری {rs:+.2f} — جبهه حامی صعود")

    return out


# ================================================================ لایه ۲: نقشه نقدینگی
def reference_levels(daily: pd.DataFrame, weekly: Optional[pd.DataFrame] = None
                     ) -> Dict:
    """PDH/PDL/PDC/PD-MID و PWH/PWL — سطوح مرجع نهادی."""
    out: Dict = {}
    if daily is None or len(daily) < 2:
        return out
    prev = daily.iloc[-2]
    out["PDH"] = float(prev["High"])
    out["PDL"] = float(prev["Low"])
    out["PDC"] = float(prev["Close"])
    out["PDM"] = (out["PDH"] + out["PDL"]) / 2.0
    cur = daily.iloc[-1]
    out["TDO"] = float(cur["Open"])
    if weekly is None:
        try:
            weekly = daily.resample("W").agg(
                {"High": "max", "Low": "min", "Close": "last"}).dropna()
        except Exception:
            weekly = None
    if weekly is not None and len(weekly) >= 2:
        out["PWH"] = float(weekly["High"].iloc[-2])
        out["PWL"] = float(weekly["Low"].iloc[-2])
    return out


def liquidation_map(price: float, atr: float, refs: Dict,
                    liq: Optional[Dict] = None,
                    scale: float = 100.0) -> Dict:
    """
    نقشه حرارتی نقدینگی: کجا استاپ ها خوشه شده اند.

    منطق نهادی: استاپ خریداران زیر PDL/PWL و استاپ فروشندگان بالای PDH/PWH
    جمع می شود. اعداد رند هم مغناطیس نقدینگی اند.
    """
    if atr <= 0:
        return dict(ok=False)
    pools: List[Dict] = []

    def add(name: str, lvl: float, side: str, weight: float, kind: str):
        if not np.isfinite(lvl) or lvl <= 0:
            return
        dist = (lvl - price) / atr
        pools.append(dict(
            name=name, level=round(lvl, 4), level_display=round(lvl * scale, 1),
            side=side, kind=kind, weight=round(weight, 2),
            dist_atr=round(dist, 3), dist_pct=round((lvl / price - 1) * 100, 4),
            above=bool(lvl > price),
        ))

    # سطوح مرجع
    add("سقف روز قبل (PDH)", refs.get("PDH", np.nan), "BSL", 1.00, "reference")
    add("کف روز قبل (PDL)", refs.get("PDL", np.nan), "SSL", 1.00, "reference")
    add("سقف هفته قبل (PWH)", refs.get("PWH", np.nan), "BSL", 1.25, "reference")
    add("کف هفته قبل (PWL)", refs.get("PWL", np.nan), "SSL", 1.25, "reference")
    add("بسته روز قبل (PDC)", refs.get("PDC", np.nan), "MID", 0.45, "reference")
    add("میانه روز قبل", refs.get("PDM", np.nan), "MID", 0.35, "reference")

    # اعداد رند در مقیاس نمایش
    us = price * scale
    for step, w in ((250, 0.4), (500, 0.6), (1000, 0.85)):
        for lvl_us in (np.floor(us / step) * step, np.ceil(us / step) * step):
            lvl = lvl_us / scale
            if abs(lvl - price) / atr < 6:
                add(f"عدد رند {int(lvl_us):,}", lvl,
                    "BSL" if lvl > price else "SSL", w, "round")

    # سطوح نقدینگی موتور SMC
    if liq:
        for key, side in (("bsl", "BSL"), ("ssl", "SSL")):
            for lv in (liq.get(key) or [])[:14]:
                if lv.get("taken"):
                    continue
                lvl = _f(lv.get("price"))
                hits = int(lv.get("hits", 1) or 1)
                w = 0.5 + min(hits, 4) * 0.22 + (0.35 if lv.get("eq") else 0)
                if abs(lvl - price) / atr < 8:
                    add(f"{side} ({hits} برخورد{'، مساوی' if lv.get('eq') else ''})",
                        lvl, side, w, "smc")

    if not pools:
        return dict(ok=False)

    # خوشه بندی: سطوح نزدیک تر از ۰.۳ ATR یکی می شوند
    pools.sort(key=lambda p: p["level"])
    clusters: List[Dict] = []
    for p in pools:
        if clusters and abs(p["level"] - clusters[-1]["level"]) < atr * 0.3:
            c = clusters[-1]
            tot = c["weight"] + p["weight"]
            c["level"] = (c["level"] * c["weight"] + p["level"] * p["weight"]) / tot
            c["level_display"] = round(c["level"] * scale, 1)
            c["weight"] = round(tot, 2)
            c["members"].append(p["name"])
            c["dist_atr"] = round((c["level"] - price) / atr, 3)
            c["above"] = bool(c["level"] > price)
        else:
            clusters.append(dict(level=p["level"], level_display=p["level_display"],
                                 side=p["side"], weight=p["weight"],
                                 dist_atr=p["dist_atr"], above=p["above"],
                                 members=[p["name"]]))

    for c in clusters:
        c["level"] = round(c["level"], 4)
        c["strength"] = round(min(c["weight"] / 2.5, 1.0), 3)
        c["label"] = " + ".join(c["members"][:3])

    above = sorted([c for c in clusters if c["above"]], key=lambda c: c["dist_atr"])
    below = sorted([c for c in clusters if not c["above"]],
                   key=lambda c: -c["dist_atr"])

    nearest_up = above[0] if above else None
    nearest_dn = below[0] if below else None

    # قوی ترین استخر در ۳ ATR
    near_pools = [c for c in clusters if abs(c["dist_atr"]) <= 3.0]
    magnet = max(near_pools, key=lambda c: c["weight"]) if near_pools else None

    # انتظار Sweep: اگر استخر قوی در ۱ ATR است، احتمال شکار نقدینگی
    sweep_setup = None
    for c in (nearest_up, nearest_dn):
        if c and abs(c["dist_atr"]) <= 1.0 and c["weight"] >= 1.0:
            sweep_setup = dict(
                target=c["level"], target_display=c["level_display"],
                side=c["side"], dist_atr=c["dist_atr"], label=c["label"],
                expect=("قیمت احتمالا سقف را جارو می کند سپس برمی گردد — "
                        "منتظر Sweep و بازگشت برای فروش"
                        if c["above"] else
                        "قیمت احتمالا کف را جارو می کند سپس برمی گردد — "
                        "منتظر Sweep و بازگشت برای خرید"),
                bias_after=-1 if c["above"] else 1,
            )
            break

    return dict(ok=True, price=round(price, 4), atr=round(atr, 4),
                clusters=clusters[:18],
                nearest_above=nearest_up, nearest_below=nearest_dn,
                magnet=magnet, sweep_setup=sweep_setup,
                n_above=len(above), n_below=len(below))


# ================================================================ لایه ۳: تقویم نامتقارن
def news_window(calendar: Dict, before_min: int = 15,
                after_min: int = 30) -> Dict:
    """
    پنجره ممنوعه نامتقارن: ۱۵ دقیقه قبل تا ۳۰ دقیقه بعد از خبر مهم.
    (نسخه قبلی متقارن ±۱۵ بود.)
    """
    out = dict(blocked=False, in_window=False, reason=None,
               next_event=None, minutes_to=None, phase="عادی")
    if not calendar or not calendar.get("ok", True):
        return out
    now = datetime.now(TEHRAN)
    events = calendar.get("upcoming") or []
    ne = calendar.get("next_event")

    for e in events:
        hrs = e.get("hours")
        if hrs is None:
            continue
        mins = hrs * 60.0
        impact = str(e.get("impact", ""))
        high = ("بالا" in impact) or ("بسیار" in impact) or e.get("high_impact")
        if not high:
            continue
        if -after_min <= mins <= before_min:
            out.update(blocked=True, in_window=True,
                       reason=f"{e.get('name')} — پنجره ممنوعه "
                              f"({before_min} دقیقه قبل تا {after_min} دقیقه بعد)",
                       minutes_to=round(mins, 1),
                       phase="قبل از انتشار" if mins > 0 else "پس از انتشار")
            break
    if ne:
        out["next_event"] = dict(name=ne.get("name"), key=ne.get("key"),
                                 hours=ne.get("hours"),
                                 impact=ne.get("impact"))
        if out["minutes_to"] is None and ne.get("hours") is not None:
            out["minutes_to"] = round(ne["hours"] * 60, 1)
    return out


def fedwatch_probability() -> Dict:
    """
    احتمال تغییر نرخ به سبک CME FedWatch.

    روش: اختلاف نرخ ضمنی فیوچرز با نرخ فعلی، تقسیم بر گام ۲۵ بیپ.
    نتیجه به احتمال هر سناریو تبدیل می شود.
    """
    try:
        import macro_data as macro
    except Exception:
        return dict(ok=False, error="ماژول کلان در دسترس نیست")

    fw = macro.fed_watch()
    if not fw.get("ok"):
        return dict(ok=False, error=fw.get("error", "داده فیوچرز نیست"))

    cur = fw.get("current_rate")
    imp = fw.get("implied_rate")
    if cur is None or imp is None:
        return dict(ok=False, error="نرخ مرجع در دسترس نیست")

    diff_bp = (imp - cur) * 100.0
    step = 25.0
    steps = diff_bp / step                       # مثبت = افزایش
    k = int(np.floor(abs(steps)))
    frac = abs(steps) - k
    direction = 1 if steps > 0 else (-1 if steps < 0 else 0)

    scen: List[Dict] = []
    if direction == 0 or abs(steps) < 0.04:
        scen.append(dict(move_bp=0, label="بدون تغییر", prob=1.0))
    else:
        base = k * step * direction
        nxt = (k + 1) * step * direction
        p_next = float(np.clip(frac, 0.0, 1.0))
        p_base = 1.0 - p_next
        for mv, p in ((base, p_base), (nxt, p_next)):
            if p > 0.005:
                lbl = ("بدون تغییر" if mv == 0 else
                       f"{'افزایش' if mv > 0 else 'کاهش'} {abs(int(mv))} بیپ")
                scen.append(dict(move_bp=int(mv), label=lbl, prob=round(p, 4)))
    scen.sort(key=lambda s: -s["prob"])

    return dict(ok=True, current_rate=cur, implied_rate=imp,
                diff_bp=round(diff_bp, 1), steps=round(steps, 3),
                direction=direction, scenarios=scen,
                most_likely=scen[0] if scen else None,
                hike_prob=round(sum(s["prob"] for s in scen if s["move_bp"] > 0), 4),
                cut_prob=round(sum(s["prob"] for s in scen if s["move_bp"] < 0), 4),
                hold_prob=round(sum(s["prob"] for s in scen if s["move_bp"] == 0), 4),
                note="تخمین از فیوچرز ZQ=F — روش ساده شده CME FedWatch")


# ================================================================ مهندسی ویژگی
FEATURE_SPEC = [
    # (نام، توضیح فارسی، دامنه)
    ("distance_to_pdh", "فاصله تا سقف روز قبل بر حسب ATR", "ATR"),
    ("distance_to_pdl", "فاصله تا کف روز قبل بر حسب ATR", "ATR"),
    ("distance_to_pwh", "فاصله تا سقف هفته قبل بر حسب ATR", "ATR"),
    ("distance_to_pwl", "فاصله تا کف هفته قبل بر حسب ATR", "ATR"),
    ("distance_to_poc", "فاصله تا نقطه کنترل حجم بر حسب ATR", "ATR"),
    ("value_area_pos", "موقعیت در ناحیه ارزش", "-1..1"),
    ("nearest_pool_atr", "فاصله تا نزدیک ترین استخر نقدینگی", "ATR"),
    ("pool_strength", "قدرت آن استخر", "0..1"),
    ("sweep_expected", "انتظار شکار نقدینگی", "0/1"),
    ("smt_divergence", "واگرایی SMT", "-1/0/1"),
    ("dxy_chg_pct", "تغییر شاخص دلار", "٪"),
    ("us10y_chg_pct", "تغییر بازده ۱۰ ساله", "٪"),
    ("vix_level", "سطح شاخص ترس", "مطلق"),
    ("vix_spike", "جهش شاخص ترس", "0/1"),
    ("intermarket_risk", "ریسک بین بازاری", "-1..1"),
    ("news_risk", "ریسک خبری", "0/1"),
    ("minutes_to_news", "دقیقه تا خبر مهم", "دقیقه"),
    ("fed_hike_prob", "احتمال افزایش نرخ", "0..1"),
    ("fed_cut_prob", "احتمال کاهش نرخ", "0..1"),
    ("candle_strength", "قدرت کندل آخر", "0..1"),
    ("candle_dir", "جهت کندل آخر", "-1/1"),
    ("upper_wick_ratio", "نسبت سایه بالا", "0..1"),
    ("lower_wick_ratio", "نسبت سایه پایین", "0..1"),
    ("cvd_pressure", "فشار جریان سفارش", "-3..3"),
    ("cvd_divergence", "واگرایی CVD", "-1/0/1"),
    ("killzone_weight", "وزن جلسه معاملاتی", "0..1"),
    ("is_ny_open", "در کیل زون نیویورک", "0/1"),
    ("weekday", "روز هفته", "0..4"),
    ("weekday_edge", "لبه آماری روز هفته", "٪"),
    ("weekday_significant", "معناداری آماری روز", "0/1"),
    ("pcr_volume", "نسبت پوت به کال", "نسبت"),
    ("iv_atm", "نوسان ضمنی اتم پول", "٪"),
    ("iv_skew", "شیب نوسان", "٪"),
    ("vrp", "پریمیوم ریسک نوسان", "٪"),
    ("regime_trending", "رژیم رونددار", "0/1"),
    ("hurst", "نمای هرست", "0..1"),
    ("adx", "قدرت روند", "0..100"),
]


def build_features(price: float, atr: float, df: pd.DataFrame,
                   refs: Dict, liqmap: Dict, veto: Dict,
                   newsw: Dict, fedp: Dict, ctx: Dict,
                   vol: Dict, flow: Dict, intel: Dict,
                   vp: Optional[Dict] = None) -> Dict:
    """
    ساخت بردار ویژگی عددی — همان جدولی که کاربر خواست.
    هر ویژگی مقدار خام + مقدار نرمال شده دارد.
    """
    F: Dict[str, float] = {}

    # --- سطوح مرجع ---
    for key, name in (("PDH", "distance_to_pdh"), ("PDL", "distance_to_pdl"),
                      ("PWH", "distance_to_pwh"), ("PWL", "distance_to_pwl")):
        lvl = refs.get(key)
        F[name] = round((price - lvl) / atr, 4) if (lvl and atr > 0) else 0.0

    # --- نمایه حجم ---
    if vp:
        poc = _f(vp.get("poc"))
        vah = _f(vp.get("vah"))
        val = _f(vp.get("val"))
        F["distance_to_poc"] = round((price - poc) / atr, 4) if (poc and atr > 0) else 0.0
        if vah > val:
            F["value_area_pos"] = round(_clip((price - (vah + val) / 2) /
                                              ((vah - val) / 2)), 4)
        else:
            F["value_area_pos"] = 0.0
    else:
        F["distance_to_poc"] = 0.0
        F["value_area_pos"] = 0.0

    # --- نقشه نقدینگی ---
    if liqmap.get("ok"):
        na, nb = liqmap.get("nearest_above"), liqmap.get("nearest_below")
        cands = [c for c in (na, nb) if c]
        if cands:
            nearest = min(cands, key=lambda c: abs(c["dist_atr"]))
            F["nearest_pool_atr"] = round(nearest["dist_atr"], 4)
            F["pool_strength"] = round(nearest["strength"], 4)
        else:
            F["nearest_pool_atr"] = 0.0
            F["pool_strength"] = 0.0
        F["sweep_expected"] = 1.0 if liqmap.get("sweep_setup") else 0.0
    else:
        F["nearest_pool_atr"] = 0.0
        F["pool_strength"] = 0.0
        F["sweep_expected"] = 0.0

    # --- بین بازاری ---
    vf = veto.get("features", {}) if veto else {}
    for k in ("smt_divergence", "dxy_chg_pct", "us10y_chg_pct",
              "vix_level", "vix_spike", "intermarket_risk"):
        F[k] = round(_f(vf.get(k)), 4)

    # --- خبر ---
    F["news_risk"] = 1.0 if newsw.get("blocked") else 0.0
    mt = newsw.get("minutes_to")
    F["minutes_to_news"] = round(_f(mt, 9999), 1) if mt is not None else 9999.0

    # --- فدرال رزرو ---
    F["fed_hike_prob"] = round(_f(fedp.get("hike_prob")), 4) if fedp.get("ok") else 0.0
    F["fed_cut_prob"] = round(_f(fedp.get("cut_prob")), 4) if fedp.get("ok") else 0.0

    # --- قدرت کندل ---
    if len(df) >= 1:
        o = _f(df["Open"].iloc[-1]); h = _f(df["High"].iloc[-1])
        l = _f(df["Low"].iloc[-1]); c = _f(df["Close"].iloc[-1])
        rng = max(h - l, 1e-9)
        F["candle_strength"] = round(abs(c - o) / rng, 4)
        F["candle_dir"] = 1.0 if c >= o else -1.0
        F["upper_wick_ratio"] = round((h - max(o, c)) / rng, 4)
        F["lower_wick_ratio"] = round((min(o, c) - l) / rng, 4)
    else:
        F["candle_strength"] = F["candle_dir"] = 0.0
        F["upper_wick_ratio"] = F["lower_wick_ratio"] = 0.0

    # --- جریان سفارش ---
    fd = (flow or {}).get("delta", {})
    F["cvd_pressure"] = round(_f(fd.get("pressure")), 4) if fd.get("ok") else 0.0
    dv = fd.get("divergence") if fd.get("ok") else None
    F["cvd_divergence"] = (1.0 if (dv and dv.get("type") == "bullish")
                           else (-1.0 if dv else 0.0))

    # --- زمان ---
    kz = (ctx or {}).get("killzone", {})
    F["killzone_weight"] = round(_f(kz.get("weight")), 4)
    act = kz.get("active") or {}
    F["is_ny_open"] = 1.0 if "نیویورک" in str(act.get("name", "")) else 0.0
    now = datetime.now(TEHRAN)
    F["weekday"] = float(now.weekday())

    sz = (flow or {}).get("seasonality", {})
    cwd = sz.get("current_weekday") if sz.get("ok") else None
    F["weekday_edge"] = round(_f(cwd.get("mean")), 4) if cwd else 0.0
    F["weekday_significant"] = 1.0 if (cwd and cwd.get("significant")) else 0.0

    # --- آپشن ---
    op = (vol or {}).get("options", {})
    F["pcr_volume"] = round(_f(op.get("pcr_volume")), 4) if op.get("ok") else 0.0
    F["iv_atm"] = round(_f(op.get("atm_iv")), 4) if op.get("ok") else 0.0
    F["iv_skew"] = round(_f(op.get("skew")), 4) if op.get("ok") else 0.0
    F["vrp"] = round(_f((vol or {}).get("vrp")), 4)

    # --- رژیم ---
    rg = (intel or {}).get("regime", {})
    F["regime_trending"] = 1.0 if rg.get("regime") == "trending" else 0.0
    F["hurst"] = round(_f(rg.get("hurst"), 0.5), 4)
    F["adx"] = round(_f(rg.get("adx")), 3)

    rows = []
    spec = {k: (fa, rngv) for k, fa, rngv in FEATURE_SPEC}
    for k, v in F.items():
        fa, rngv = spec.get(k, (k, ""))
        rows.append(dict(key=k, name=fa, value=v, unit=rngv))

    return dict(ok=True, n=len(F), values=F, table=rows,
                generated=datetime.now(TEHRAN).isoformat())


# ================================================================ تجمیع پنج لایه
def build_institutional(price: float, atr: float, df: pd.DataFrame,
                        daily: Optional[pd.DataFrame], ctx: Dict,
                        vol: Dict, flow: Dict, intel: Dict,
                        liq: Optional[Dict] = None,
                        vp: Optional[Dict] = None,
                        scale: float = 100.0) -> Dict:
    """اجرای پنج لایه و ساخت بردار ویژگی."""
    veto = intermarket_veto(ctx.get("intermarket", {}), df,
                            asset=ctx.get("asset"))
    refs = reference_levels(daily) if daily is not None else {}
    lmap = liquidation_map(price, atr, refs, liq, scale)
    newsw = news_window(ctx.get("calendar", {}))
    fedp = fedwatch_probability()
    feats = build_features(price, atr, df, refs, lmap, veto, newsw,
                           fedp, ctx, vol, flow, intel, vp)

    # جمع بندی وتو
    blocks: List[str] = []
    if newsw.get("blocked"):
        blocks.append(newsw["reason"])
    veto_long = bool(veto.get("veto_long")) or bool(newsw.get("blocked"))
    veto_short = bool(veto.get("veto_short")) or bool(newsw.get("blocked"))

    return dict(
        ok=True,
        layer1_intermarket=veto,
        layer2_liquidity=dict(refs={k: dict(raw=round(v, 4),
                                            display=round(v * scale, 1))
                                    for k, v in refs.items()},
                              map=lmap),
        layer3_macro=dict(news=newsw, fedwatch=fedp),
        layer4_sentiment=dict(news=ctx.get("news", {}),
                              options=dict(
                                  pcr=vol.get("options", {}).get("pcr_volume"),
                                  pcr_state=vol.get("options", {}).get("pcr_state"),
                                  skew=vol.get("options", {}).get("skew"),
                                  iv=vol.get("options", {}).get("atm_iv"))),
        layer5_time=dict(killzone=ctx.get("killzone", {}),
                         seasonality=(flow or {}).get("seasonality", {}).get(
                             "current_weekday"),
                         intraday=(flow or {}).get("intraday", {}).get("best")),
        veto=dict(long=veto_long, short=veto_short,
                  reasons=veto.get("reasons", []) + blocks,
                  warnings=veto.get("warnings", []),
                  breakout_unreliable=bool(veto.get("breakout_unreliable"))),
        features=feats,
    )
