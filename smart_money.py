#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smart_money.py
موتور تحلیل Smart Money / ICT / Order-Flow برای داوجونز (DIA)

ماژول ها:
  1) hft_bank_coalition ...... ائتلاف بانک ها و ردپای ربات های HFT
  2) market_structure/FVG .... BOS / CHoCH / Fair Value Gap / Order Block
  3) liquidity_engine ........ تسویه نقدینگی، گره حجمی (HVN/LVN/POC)، اصلاح به گره
  4) institutional_flow ...... داده موسسات، Smart Money Index، جریان پول ICT
  5) fake_trend_detector ..... تشخیص روند فیک و شکست جعلی
  6) block_order_scanner ..... سفارشات سنگین و حجم های عمده خرید/فروش
  7) sweep_detector .......... جمع آوری نقدینگی BSL / SSL Sweep
  +) build_master_signal ..... سیگنال نهایی دقیق با ورود/حد ضرر/اهداف
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================================
# ابزارهای پایه
# ============================================================================
def atr_array(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(n, min_periods=1).mean().to_numpy()


def clv_array(df: pd.DataFrame) -> np.ndarray:
    """Close Location Value: +1 یعنی بسته شدن روی High (فشار خرید)."""
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    rng = np.where((h - l) == 0, np.nan, h - l)
    clv = ((c - l) - (h - c)) / rng
    return np.nan_to_num(clv, nan=0.0)


def delta_proxy(df: pd.DataFrame) -> np.ndarray:
    """تقریب دلتای حجم (خریدار تهاجمی منهای فروشنده تهاجمی)."""
    return clv_array(df) * df["Volume"].to_numpy(float)


def zscore(x: np.ndarray, n: int = 20) -> np.ndarray:
    s = pd.Series(x)
    m = s.rolling(n, min_periods=max(3, n // 3)).mean()
    sd = s.rolling(n, min_periods=max(3, n // 3)).std()
    return ((s - m) / sd.replace(0, np.nan)).fillna(0.0).to_numpy()


def _rsi(c: np.ndarray, n: int = 14) -> np.ndarray:
    s = pd.Series(c)
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50).to_numpy()


def _adx(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    up = np.concatenate([[0.0], np.diff(h)])
    dn = np.concatenate([[0.0], -np.diff(l)])
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = atr_array(df, n)
    tr = np.where(tr == 0, np.nan, tr)
    pdi = 100 * pd.Series(plus_dm).ewm(alpha=1 / n, adjust=False).mean().to_numpy() / tr
    mdi = 100 * pd.Series(minus_dm).ewm(alpha=1 / n, adjust=False).mean().to_numpy() / tr
    dx = 100 * np.abs(pdi - mdi) / np.where((pdi + mdi) == 0, np.nan, pdi + mdi)
    return pd.Series(dx).ewm(alpha=1 / n, adjust=False).mean().fillna(0).to_numpy()


def pct(a: float, b: float) -> float:
    return (a - b) / b * 100 if b else 0.0


# ============================================================================
# 2-الف) ساختار بازار: سوئینگ ها، BOS، CHoCH
# ============================================================================
def detect_swings(df: pd.DataFrame, left: int = 3, right: int = 3):
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    n = len(df)
    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    for i in range(left, n - right):
        wh = h[i - left:i + right + 1]
        wl = l[i - left:i + right + 1]
        if wh.argmax() == left:
            sh[i] = h[i]
        if wl.argmin() == left:
            sl[i] = l[i]
    return sh, sl


def market_structure(df: pd.DataFrame, left: int = 3, right: int = 3) -> Dict:
    """
    BOS  = Break of Structure  (ادامه روند)
    CHoCH= Change of Character (تغییر کاراکتر / احتمال بازگشت روند)
    سوئینگ ها فقط بعد از تایید (i+right) در دسترس قرار می گیرند => بدون دید به آینده.
    """
    sh, sl = detect_swings(df, left, right)
    c = df["Close"].to_numpy(float)
    n = len(df)
    events: List[Dict] = []
    bias = 0
    bias_series = np.zeros(n, dtype=int)
    last_sh_v, last_sh_i = np.nan, -1
    last_sl_v, last_sl_i = np.nan, -1
    prot_high, prot_low = np.nan, np.nan  # سطوح محافظ فعلی

    for i in range(n):
        j = i - right
        if j >= 0:
            if not np.isnan(sh[j]):
                last_sh_v, last_sh_i = sh[j], j
            if not np.isnan(sl[j]):
                last_sl_v, last_sl_i = sl[j], j

        if last_sh_i >= 0 and c[i] > last_sh_v:
            kind = "BOS" if bias == 1 else "CHoCH"
            events.append(dict(i=i, time=df.index[i], type=kind, dir="bull",
                               level=float(last_sh_v), ref_i=int(last_sh_i),
                               price=float(c[i])))
            bias = 1
            prot_low = last_sl_v
            last_sh_v, last_sh_i = np.nan, -1
        elif last_sl_i >= 0 and c[i] < last_sl_v:
            kind = "BOS" if bias == -1 else "CHoCH"
            events.append(dict(i=i, time=df.index[i], type=kind, dir="bear",
                               level=float(last_sl_v), ref_i=int(last_sl_i),
                               price=float(c[i])))
            bias = -1
            prot_high = last_sh_v
            last_sl_v, last_sl_i = np.nan, -1

        bias_series[i] = bias

    return dict(swing_high=sh, swing_low=sl, events=events, bias=bias,
                bias_series=bias_series, protected_high=prot_high,
                protected_low=prot_low)


# ============================================================================
# 2-ب) Fair Value Gap (عدم توازن قیمتی) + Order Block
# ============================================================================
def find_fvg(df: pd.DataFrame, min_atr_frac: float = 0.08) -> List[Dict]:
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    atr = atr_array(df)
    out: List[Dict] = []
    for i in range(2, len(df)):
        a = atr[i] if atr[i] > 0 else 1e-9
        if l[i] > h[i - 2]:
            size = l[i] - h[i - 2]
            if size >= min_atr_frac * a:
                out.append(dict(i=i, time=df.index[i], dir="bull",
                                top=float(l[i]), bottom=float(h[i - 2]),
                                size=float(size), atr_x=float(size / a)))
        if h[i] < l[i - 2]:
            size = l[i - 2] - h[i]
            if size >= min_atr_frac * a:
                out.append(dict(i=i, time=df.index[i], dir="bear",
                                top=float(l[i - 2]), bottom=float(h[i]),
                                size=float(size), atr_x=float(size / a)))
    for g in out:
        k = g["i"]
        fl = l[k + 1:]
        fh = h[k + 1:]
        mid = (g["top"] + g["bottom"]) / 2
        if g["dir"] == "bull":
            g["touched"] = bool(fl.size and (fl <= g["top"]).any())
            g["mitigated"] = bool(fl.size and (fl <= mid).any())
            g["filled"] = bool(fl.size and (fl <= g["bottom"]).any())
        else:
            g["touched"] = bool(fh.size and (fh >= g["bottom"]).any())
            g["mitigated"] = bool(fh.size and (fh >= mid).any())
            g["filled"] = bool(fh.size and (fh >= g["top"]).any())
    return out


def find_order_blocks(df: pd.DataFrame, disp_atr: float = 1.1,
                      body_frac: float = 0.55, max_back: int = 5) -> List[Dict]:
    """
    Order Block = آخرین کندل مخالف قبل از حرکت ایمپالسیو (Displacement).
    ردپای مستقیم سفارش گذاری موسسات.
    """
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    v = df["Volume"].to_numpy(float)
    atr = atr_array(df)
    vz = zscore(v, 20)
    obs: List[Dict] = []
    for i in range(3, len(df)):
        a = atr[i] if atr[i] > 0 else 1e-9
        rng = h[i] - l[i]
        body = abs(c[i] - o[i])
        if rng <= 0 or body < body_frac * rng or rng < disp_atr * a:
            continue
        bull_disp = c[i] > o[i] and l[i] > h[i - 2]
        bear_disp = c[i] < o[i] and h[i] < l[i - 2]
        if not (bull_disp or bear_disp):
            continue
        direction = "bull" if bull_disp else "bear"
        for k in range(i - 1, max(i - max_back - 1, 0), -1):
            is_opp = (c[k] < o[k]) if direction == "bull" else (c[k] > o[k])
            if is_opp:
                obs.append(dict(i=k, time=df.index[k], dir=direction,
                                top=float(max(o[k], c[k], h[k])),
                                bottom=float(min(o[k], c[k], l[k])),
                                disp_i=i, disp_atr=float(rng / a),
                                vol_z=float(vz[k])))
                break
    for b in obs:
        k = b["i"]
        fl = l[k + 1:]
        fh = h[k + 1:]
        if b["dir"] == "bull":
            b["mitigated"] = bool(fl.size and (fl <= b["top"]).any())
        else:
            b["mitigated"] = bool(fh.size and (fh >= b["bottom"]).any())
    return obs


# ============================================================================
# 3) موتور نقدینگی: EQH/EQL، سطوح دوره ای، پروفایل حجم (گره ها)
# ============================================================================
def liquidity_levels(df: pd.DataFrame, left: int = 3, right: int = 3,
                     tol_atr: float = 0.25, lookback: int = 300) -> Dict:
    d = df.tail(lookback)
    off = len(df) - len(d)
    sh, sl = detect_swings(d, left, right)
    atr = float(np.nanmedian(atr_array(d)))
    tol = tol_atr * atr

    def cluster(vals_idx):
        vals_idx = sorted(vals_idx, key=lambda t: t[1])
        groups = []
        for idx, v in vals_idx:
            if groups and abs(v - groups[-1]["price"]) <= tol:
                g = groups[-1]
                g["members"].append(idx)
                g["price"] = float(np.mean([g["price"], v]))
            else:
                groups.append(dict(price=float(v), members=[idx]))
        return groups

    hi_pts = [(i + off, sh[i]) for i in range(len(d)) if not np.isnan(sh[i])]
    lo_pts = [(i + off, sl[i]) for i in range(len(d)) if not np.isnan(sl[i])]
    hg = cluster(hi_pts)
    lg = cluster(lo_pts)

    cur = float(df["Close"].iloc[-1])
    h_all = df["High"].to_numpy(float)
    l_all = df["Low"].to_numpy(float)

    bsl, ssl = [], []
    for g in hg:
        last_i = max(g["members"])
        taken = bool(h_all[last_i + 1:].size and (h_all[last_i + 1:] > g["price"]).any())
        bsl.append(dict(price=g["price"], hits=len(g["members"]), last_i=last_i,
                        eq=len(g["members"]) >= 2, taken=taken, kind="BSL",
                        dist_pct=pct(g["price"], cur)))
    for g in lg:
        last_i = max(g["members"])
        taken = bool(l_all[last_i + 1:].size and (l_all[last_i + 1:] < g["price"]).any())
        ssl.append(dict(price=g["price"], hits=len(g["members"]), last_i=last_i,
                        eq=len(g["members"]) >= 2, taken=taken, kind="SSL",
                        dist_pct=pct(g["price"], cur)))

    # سطوح دوره ای (روز/هفته/ماه قبل) = استخرهای اصلی نقدینگی
    periodic = {}
    try:
        idx = pd.DatetimeIndex(df.index)
        intraday = (idx[1] - idx[0]) < pd.Timedelta("1D") if len(idx) > 1 else False
        rules = [("روز قبل", "D"), ("هفته قبل", "W")] if intraday else \
                [("هفته قبل", "W"), ("ماه قبل", "ME")]
        for label, rule in rules:
            g = df.resample(rule).agg({"High": "max", "Low": "min"}).dropna()
            if len(g) >= 2:
                periodic[label] = dict(high=float(g["High"].iloc[-2]),
                                       low=float(g["Low"].iloc[-2]))
    except Exception:
        pass

    # اولویت با استخرهای دست نخورده و نزدیک به قیمت
    fresh_bsl = sorted([x for x in bsl if not x["taken"] and x["price"] > cur],
                       key=lambda x: x["price"])
    fresh_ssl = sorted([x for x in ssl if not x["taken"] and x["price"] < cur],
                       key=lambda x: -x["price"])
    return dict(bsl=bsl, ssl=ssl, fresh_bsl=fresh_bsl, fresh_ssl=fresh_ssl,
                periodic=periodic, tol=tol)


def volume_profile(df: pd.DataFrame, bins: int = 64, lookback: int = 150) -> Dict:
    """
    پروفایل حجم: POC، Value Area، گره های پرحجم (HVN) و کم حجم (LVN).
    HVN = گره تعادل و مقصد اصلاح قیمت | LVN = خلاء نقدینگی و مسیر حرکت سریع
    """
    d = df.tail(lookback)
    lo = float(d["Low"].min())
    hi = float(d["High"].max())
    if hi <= lo:
        return {}
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    vp = np.zeros(bins)
    vpd = np.zeros(bins)          # پروفایل دلتا (خرید منهای فروش)
    clv = clv_array(d)
    H = d["High"].to_numpy(float)
    L = d["Low"].to_numpy(float)
    V = d["Volume"].to_numpy(float)
    step = edges[1] - edges[0]
    for k in range(len(d)):
        a, b = L[k], H[k]
        i0 = max(0, int((a - lo) / step))
        i1 = min(bins - 1, int((b - lo) / step))
        m = i1 - i0 + 1
        vp[i0:i1 + 1] += V[k] / m
        vpd[i0:i1 + 1] += (V[k] * clv[k]) / m

    poc_i = int(vp.argmax())
    poc = float(centers[poc_i])
    total = vp.sum()
    order = np.argsort(vp)[::-1]
    acc, sel = 0.0, []
    for i in order:
        acc += vp[i]
        sel.append(i)
        if acc >= 0.7 * total:
            break
    vah = float(centers[max(sel)])
    val = float(centers[min(sel)])

    mean, sd = vp.mean(), vp.std()
    hvn, lvn = [], []
    for i in range(1, bins - 1):
        if vp[i] >= vp[i - 1] and vp[i] >= vp[i + 1] and vp[i] > mean + 0.6 * sd:
            hvn.append(float(centers[i]))
        if vp[i] <= vp[i - 1] and vp[i] <= vp[i + 1] and vp[i] < mean - 0.4 * sd:
            lvn.append(float(centers[i]))

    cur = float(df["Close"].iloc[-1])
    near_hvn = min(hvn, key=lambda x: abs(x - cur)) if hvn else None
    near_lvn = min(lvn, key=lambda x: abs(x - cur)) if lvn else None
    return dict(centers=centers, profile=vp, delta_profile=vpd, poc=poc,
                vah=vah, val=val, hvn=hvn, lvn=lvn, near_hvn=near_hvn,
                near_lvn=near_lvn, lo=lo, hi=hi,
                in_value=bool(val <= cur <= vah),
                poc_dist_pct=pct(poc, cur))


# ============================================================================
# 7) شکار نقدینگی: BSL / SSL Sweep
# ============================================================================
def detect_sweeps(df: pd.DataFrame, liq: Dict, window: int = 20,
                  lookback: int = 80, wick_frac: float = 0.30) -> List[Dict]:
    """
    Sweep = قیمت سطح نقدینگی را می زند (استاپ ها جمع می شوند) ولی داخل برمی گردد.
    این دقیقا رفتار Smart Money قبل از حرکت اصلی است.
    """
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    o = df["Open"].to_numpy(float)
    v = df["Volume"].to_numpy(float)
    vz = zscore(v, 20)
    atr = atr_array(df)
    n = len(df)
    prior_hi = pd.Series(h).rolling(window).max().shift(1).to_numpy()
    prior_lo = pd.Series(l).rolling(window).min().shift(1).to_numpy()

    levels_b = [x["price"] for x in liq.get("bsl", []) if x["hits"] >= 2]
    levels_s = [x["price"] for x in liq.get("ssl", []) if x["hits"] >= 2]
    for lab, pv in liq.get("periodic", {}).items():
        levels_b.append(pv["high"])
        levels_s.append(pv["low"])

    out: List[Dict] = []
    start = max(2, n - lookback)
    for i in range(start, n):
        rng = h[i] - l[i]
        if rng <= 0:
            continue
        up_w = (h[i] - max(o[i], c[i])) / rng
        dn_w = (min(o[i], c[i]) - l[i]) / rng

        ref_b = [x for x in levels_b + [prior_hi[i]]
                 if x is not None and not np.isnan(x) and h[i] > x > c[i]]
        if ref_b and up_w >= wick_frac:
            lvl = float(max(ref_b))
            out.append(dict(i=i, time=df.index[i], type="BSL_SWEEP", dir="bear",
                            level=lvl, extreme=float(h[i]), close=float(c[i]),
                            wick=float(up_w), vol_z=float(vz[i]),
                            depth_atr=float((h[i] - lvl) / max(atr[i], 1e-9))))

        ref_s = [x for x in levels_s + [prior_lo[i]]
                 if x is not None and not np.isnan(x) and l[i] < x < c[i]]
        if ref_s and dn_w >= wick_frac:
            lvl = float(min(ref_s))
            out.append(dict(i=i, time=df.index[i], type="SSL_SWEEP", dir="bull",
                            level=lvl, extreme=float(l[i]), close=float(c[i]),
                            wick=float(dn_w), vol_z=float(vz[i]),
                            depth_atr=float((lvl - l[i]) / max(atr[i], 1e-9))))
    return out


# ============================================================================
# 6) اسکنر سفارشات سنگین و حجم های عمده
# ============================================================================
def block_order_scanner(df: pd.DataFrame, z_thr: float = 1.8,
                        lookback: int = 60) -> Dict:
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    v = df["Volume"].to_numpy(float)
    atr = atr_array(df)
    vz = zscore(v, 20)
    clv = clv_array(df)
    n = len(df)
    start = max(1, n - lookback)
    blocks, absorption, icebergs = [], [], []

    for i in range(start, n):
        if vz[i] < z_thr:
            continue
        rng = h[i] - l[i]
        a = max(atr[i], 1e-9)
        notional = v[i] * c[i]
        side = "BUY" if clv[i] > 0.15 else ("SELL" if clv[i] < -0.15 else "NEUTRAL")
        rec = dict(i=i, time=df.index[i], vol=float(v[i]), vol_z=float(vz[i]),
                   notional=float(notional), side=side, clv=float(clv[i]),
                   price=float(c[i]), range_atr=float(rng / a),
                   ret_pct=float(pct(c[i], o[i])))
        blocks.append(rec)
        # جذب سفارش: حجم بسیار بالا ولی دامنه قیمتی کوچک => دیوار سفارش موسسات
        if rng < 0.6 * a:
            absorption.append(rec | {"note": "جذب سفارش (Absorption)"})

    # آیسبرگ: چند کندل پیاپی پرحجم در محدوده فشرده قیمتی
    for i in range(start + 2, n):
        seg_v = vz[i - 2:i + 1]
        seg_h, seg_l = h[i - 2:i + 1].max(), l[i - 2:i + 1].min()
        if (seg_v > 1.0).all() and (seg_h - seg_l) < 1.1 * max(atr[i], 1e-9):
            icebergs.append(dict(i=i, time=df.index[i],
                                 price=float((seg_h + seg_l) / 2),
                                 vol_z=float(seg_v.mean()),
                                 side="BUY" if clv[i - 2:i + 1].mean() > 0 else "SELL"))

    buy_n = sum(b["notional"] for b in blocks if b["side"] == "BUY")
    sell_n = sum(b["notional"] for b in blocks if b["side"] == "SELL")
    tot = buy_n + sell_n
    ratio = (buy_n - sell_n) / tot if tot else 0.0
    return dict(blocks=sorted(blocks, key=lambda x: -x["vol_z"]),
                absorption=absorption, icebergs=icebergs,
                buy_notional=buy_n, sell_notional=sell_n, imbalance=ratio,
                n_blocks=len(blocks))


# ============================================================================
# 4) داده موسسات + جریان پول هوشمند (ICT Smart Money)
# ============================================================================
def institutional_flow(df: pd.DataFrame, intraday: Optional[pd.DataFrame] = None) -> Dict:
    c = df["Close"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    v = df["Volume"].to_numpy(float)
    clv = clv_array(df)

    obv = np.cumsum(np.sign(np.concatenate([[0.0], np.diff(c)])) * v)
    ad = np.cumsum(clv * v)
    cd = np.cumsum(delta_proxy(df))

    mfv = pd.Series(clv * v)
    cmf = (mfv.rolling(20).sum() / pd.Series(v).rolling(20).sum()).fillna(0).to_numpy()

    tp = (h + l + c) / 3
    pos_mf = pd.Series(np.where(np.concatenate([[0.0], np.diff(tp)]) > 0, tp * v, 0.0))
    neg_mf = pd.Series(np.where(np.concatenate([[0.0], np.diff(tp)]) < 0, tp * v, 0.0))
    mr = pos_mf.rolling(14).sum() / neg_mf.rolling(14).sum().replace(0, np.nan)
    mfi = (100 - 100 / (1 + mr)).fillna(50).to_numpy()

    vwap_n = min(60, len(df))
    vwap = float((tp[-vwap_n:] * v[-vwap_n:]).sum() / max(v[-vwap_n:].sum(), 1e-9))

    def slope(arr, k=20):
        k = min(k, len(arr) - 1)
        if k < 3:
            return 0.0
        y = arr[-k:]
        x = np.arange(k)
        rng = np.ptp(y)
        return float(np.polyfit(x, y, 1)[0] / (rng / k + 1e-9))

    # واگرایی قیمت با جریان پول = ردپای توزیع/انباشت پنهان موسسات
    k = min(30, len(df) - 1)
    p_chg = pct(c[-1], c[-k])
    obv_chg = obv[-1] - obv[-k]
    cd_chg = cd[-1] - cd[-k]
    divergence = "ندارد"
    if p_chg > 0.6 and cd_chg < 0:
        divergence = "واگرایی منفی (توزیع پنهان موسسات)"
    elif p_chg < -0.6 and cd_chg > 0:
        divergence = "واگرایی مثبت (انباشت پنهان موسسات)"

    # Smart Money Index واقعی (نیازمند داده درون روزی)
    smi_trend, smi_series = None, None
    if intraday is not None and len(intraday) > 30:
        try:
            g = intraday.groupby(intraday.index.date)
            rows = []
            for day, dd in g:
                if len(dd) < 8:
                    continue
                first_close = float(dd["Close"].iloc[1]) if len(dd) > 1 else float(dd["Close"].iloc[0])
                day_open = float(dd["Open"].iloc[0])
                last_close = float(dd["Close"].iloc[-1])
                hour_ago = float(dd["Close"].iloc[-5]) if len(dd) >= 5 else float(dd["Close"].iloc[0])
                rows.append((day, (first_close - day_open), (last_close - hour_ago)))
            if len(rows) >= 5:
                arr = np.array([(r[2] - r[1]) for r in rows])
                smi_series = np.cumsum(arr)
                smi_trend = slope(smi_series, min(20, len(smi_series)))
        except Exception:
            pass

    return dict(obv=obv, ad=ad, cum_delta=cd, cmf=cmf, mfi=mfi, vwap=vwap,
                obv_slope=slope(obv), cd_slope=slope(cd), ad_slope=slope(ad),
                cmf_last=float(cmf[-1]), mfi_last=float(mfi[-1]),
                vwap_dev_pct=pct(c[-1], vwap), divergence=divergence,
                smi_trend=smi_trend, smi_series=smi_series,
                price_chg_pct=p_chg, obv_chg=float(obv_chg), cd_chg=float(cd_chg))


# ============================================================================
# 1) ائتلاف بانک ها + ردپای ربات های HFT
# ============================================================================
BANK_BASKET = ["GS", "JPM", "V", "AXP", "TRV", "XLF"]
BANK_NAMES = {"GS": "گلدمن ساکس", "JPM": "جی پی مورگان", "V": "ویزا",
              "AXP": "امریکن اکسپرس", "TRV": "تراولرز", "XLF": "ETF بخش مالی"}


def hft_footprint(df: pd.DataFrame, lookback: int = 120) -> Dict:
    """
    ردپای الگوریتم های فرکانس بالا:
      - انفجار حجم بدون حرکت قیمت (تامین نقدینگی / جذب)
      - شکار استاپ با سایه بلند و حجم بالا
      - خودهمبستگی منفی بازده (غلبه الگوریتم های بازگشت به میانگین)
      - میخکوب شدن روی اعداد رند (Pinning)
    """
    d = df.tail(min(lookback, len(df)))
    o = d["Open"].to_numpy(float)
    h = d["High"].to_numpy(float)
    l = d["Low"].to_numpy(float)
    c = d["Close"].to_numpy(float)
    v = d["Volume"].to_numpy(float)
    atr = atr_array(d)
    vz = zscore(v, 20)
    ret = np.concatenate([[0.0], np.diff(c) / c[:-1]])
    absret = np.abs(ret)
    mean_abs = absret[absret > 0].mean() if (absret > 0).any() else 1e-9

    burst = (vz > 1.5) & (absret < 0.35 * mean_abs)
    burst_rate = float(burst.mean())

    rng = np.where((h - l) == 0, np.nan, h - l)
    up_w = (h - np.maximum(o, c)) / rng
    dn_w = (np.minimum(o, c) - l) / rng
    hunt = ((np.nan_to_num(up_w) > 0.55) | (np.nan_to_num(dn_w) > 0.55)) & (vz > 0.8)
    hunt_rate = float(hunt.mean())

    ac1 = float(pd.Series(ret[1:]).autocorr(lag=1) or 0.0)

    step = 1.0 if c[-1] > 100 else 0.5
    pin = np.abs(c - np.round(c / step) * step) / c
    pin_rate = float((pin < 0.0006).mean())

    vol_of_vol = float(pd.Series(absret).rolling(10).std().std() * 1e4)

    idx = 0.0
    idx += min(burst_rate / 0.12, 1) * 30
    idx += min(hunt_rate / 0.28, 1) * 25
    idx += min(max(-ac1, 0) / 0.25, 1) * 25
    idx += min(pin_rate / 0.12, 1) * 20
    idx = float(np.clip(idx, 0, 100))

    if ac1 < -0.08:
        regime = "غلبه الگوریتم های بازگشت به میانگین (محیط رنج/شکار استاپ)"
    elif ac1 > 0.08:
        regime = "غلبه الگوریتم های مومنتوم (محیط روندی)"
    else:
        regime = "بازار متعادل بین الگوریتم های مومنتوم و بازگشتی"

    return dict(hft_index=idx, burst_rate=burst_rate, hunt_rate=hunt_rate,
                autocorr=ac1, pin_rate=pin_rate, vol_of_vol=vol_of_vol,
                regime=regime,
                n_absorb=int(burst.sum()), n_hunt=int(hunt.sum()))


def bank_coalition(interval: str = "1d", period: str = "6mo",
                   ref: Optional[pd.DataFrame] = None) -> Dict:
    """
    ائتلاف بانک ها: آیا سنگین وزن های مالی داوجونز هم جهت و با حجم غیرعادی
    در حال انباشت یا توزیع هستند؟ هماهنگی بالا = جریان سفارش نهادی هماهنگ.
    """
    import yfinance as yf
    members: List[Dict] = []
    try:
        raw = yf.download(BANK_BASKET, period=period, interval=interval,
                          group_by="ticker", progress=False, auto_adjust=True,
                          threads=True)
    except Exception as e:
        return dict(ok=False, error=str(e), members=[], score=0.0, agreement=0.0)

    for sym in BANK_BASKET:
        try:
            d = raw[sym].dropna() if isinstance(raw.columns, pd.MultiIndex) else raw.dropna()
            if len(d) < 30:
                continue
            d = d.rename(columns=str.title)
            clv = clv_array(d)
            v = d["Volume"].to_numpy(float)
            vz = zscore(v, 20)
            cd = np.cumsum(clv * v)
            c = d["Close"].to_numpy(float)
            k = min(10, len(c) - 1)
            r10 = pct(c[-1], c[-k])
            cd_sl = float(np.polyfit(np.arange(min(20, len(cd))),
                                     cd[-min(20, len(cd)):], 1)[0])
            cd_norm = cd_sl / (abs(cd[-min(20, len(cd)):]).mean() + 1e-9)
            sma20 = float(pd.Series(c).rolling(20).mean().iloc[-1])
            above = c[-1] > sma20
            strength = np.tanh(r10 / 3.0) * 0.5 + np.tanh(cd_norm * 8) * 0.35 + \
                       (0.15 if above else -0.15)
            members.append(dict(symbol=sym, name=BANK_NAMES.get(sym, sym),
                                ret_pct=float(r10), vol_z=float(vz[-1]),
                                cd_slope=float(cd_norm), above_sma20=bool(above),
                                strength=float(strength)))
        except Exception:
            continue

    if not members:
        return dict(ok=False, error="داده ای دریافت نشد", members=[], score=0.0,
                    agreement=0.0)

    s = np.array([m["strength"] for m in members])
    score = float(s.mean())
    agreement = float(max((s > 0).mean(), (s < 0).mean()))
    heavy = float(np.mean([1 if m["vol_z"] > 1.0 else 0 for m in members]))

    corr = None
    if ref is not None and len(ref) > 40:
        try:
            rr = ref["Close"].pct_change().dropna().tail(60)
            xlf = raw["XLF"]["Close"].pct_change().dropna().tail(60) \
                if isinstance(raw.columns, pd.MultiIndex) else None
            if xlf is not None:
                j = pd.concat([rr.reset_index(drop=True), xlf.reset_index(drop=True)],
                              axis=1).dropna()
                corr = float(j.corr().iloc[0, 1])
        except Exception:
            corr = None

    if score > 0.18 and agreement >= 0.7:
        verdict = "ائتلاف صعودی بانک ها (انباشت هماهنگ)"
    elif score < -0.18 and agreement >= 0.7:
        verdict = "ائتلاف نزولی بانک ها (توزیع هماهنگ)"
    elif agreement < 0.6:
        verdict = "بانک ها پراکنده و بدون هماهنگی (بی جهتی نهادی)"
    else:
        verdict = "ائتلاف ضعیف / در حال شکل گیری"

    return dict(ok=True, members=sorted(members, key=lambda m: -m["strength"]),
                score=score, agreement=agreement, heavy_vol_share=heavy,
                corr_xlf=corr, verdict=verdict)


# ============================================================================
# 5) تشخیص روند فیک
# ============================================================================
def fake_trend_detector(df: pd.DataFrame, struct: Dict, flow: Dict,
                        lookback: int = 40) -> Dict:
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    o = df["Open"].to_numpy(float)
    v = df["Volume"].to_numpy(float)
    atr = atr_array(df)
    adx = _adx(df)
    rsi = _rsi(c)
    vma = pd.Series(v).rolling(20).mean().to_numpy()
    n = len(df)
    reasons: List[str] = []
    score = 0.0

    # 1) روند بدون قدرت
    if adx[-1] < 18:
        score += 22
        reasons.append(f"ADX پایین ({adx[-1]:.1f}) - حرکت فاقد قدرت روندی واقعی")
    elif adx[-1] < 23:
        score += 10
        reasons.append(f"ADX ضعیف ({adx[-1]:.1f}) - روند شکننده")

    # 2) شکست بدون تایید حجم
    w = 20
    ph = pd.Series(h).rolling(w).max().shift(1).to_numpy()
    pl = pd.Series(l).rolling(w).min().shift(1).to_numpy()
    fake_breaks = 0
    for i in range(max(w + 1, n - lookback), n):
        if not np.isnan(ph[i]) and h[i] > ph[i] and c[i] < ph[i]:
            fake_breaks += 1
        if not np.isnan(pl[i]) and l[i] < pl[i] and c[i] > pl[i]:
            fake_breaks += 1
    if fake_breaks >= 3:
        score += 18
        reasons.append(f"{fake_breaks} شکست جعلی سطح در {lookback} کندل اخیر")
    elif fake_breaks > 0:
        score += 7
        reasons.append(f"{fake_breaks} شکست ناموفق سطح - بازار در حال فریب")

    last_break_vol = None
    for i in range(n - 1, max(n - 12, 0), -1):
        if not np.isnan(ph[i]) and c[i] > ph[i]:
            last_break_vol = v[i] / max(vma[i], 1e-9)
            break
        if not np.isnan(pl[i]) and c[i] < pl[i]:
            last_break_vol = v[i] / max(vma[i], 1e-9)
            break
    if last_break_vol is not None and last_break_vol < 0.95:
        score += 16
        reasons.append(f"آخرین شکست با حجم ضعیف ({last_break_vol:.2f}x میانگین) - فاقد تایید")

    # 3) واگرایی جریان پول با قیمت
    if "توزیع" in flow.get("divergence", "") or "انباشت" in flow.get("divergence", ""):
        score += 20
        reasons.append(f"{flow['divergence']} - حرکت قیمت پشتوانه جریان سفارش ندارد")

    # 4) غلبه سایه ها (رد شدن قیمت)
    seg = slice(max(0, n - 15), n)
    rng = np.where((h[seg] - l[seg]) == 0, np.nan, h[seg] - l[seg])
    up_w = np.nanmean((h[seg] - np.maximum(o[seg], c[seg])) / rng)
    dn_w = np.nanmean((np.minimum(o[seg], c[seg]) - l[seg]) / rng)
    if max(up_w, dn_w) > 0.42:
        score += 14
        side = "بالا" if up_w > dn_w else "پایین"
        reasons.append(f"غلبه سایه های سمت {side} ({max(up_w, dn_w) * 100:.0f}%) - رد قیمت توسط بازار")

    # 5) واگرایی RSI
    k = min(25, n - 2)
    if c[-1] > c[-k] and rsi[-1] < rsi[-k] - 4:
        score += 12
        reasons.append("واگرایی منفی RSI - سقف های بالاتر بدون قدرت")
    elif c[-1] < c[-k] and rsi[-1] > rsi[-k] + 4:
        score += 12
        reasons.append("واگرایی مثبت RSI - کف های پایین تر بدون قدرت")

    # 6) نوسان بدون پیشروی (چاپینس)
    net = abs(c[-1] - c[-min(20, n - 1)])
    path = np.abs(np.diff(c[-min(20, n - 1):])).sum()
    eff = net / max(path, 1e-9)
    if eff < 0.22:
        score += 12
        reasons.append(f"کارایی حرکت پایین ({eff * 100:.0f}%) - نوسان بدون پیشروی واقعی")

    score = float(np.clip(score, 0, 100))
    if score >= 62:
        verdict = "احتمال بسیار بالای روند فیک - ورود ممنوع"
    elif score >= 42:
        verdict = "روند مشکوک - فقط با تایید چندگانه وارد شوید"
    elif score >= 24:
        verdict = "روند نسبتا معتبر با نقاط ضعف جزئی"
    else:
        verdict = "روند سالم و معتبر"
    return dict(score=score, verdict=verdict, reasons=reasons,
                adx=float(adx[-1]), efficiency=float(eff),
                fake_breaks=fake_breaks)


# ============================================================================
# سیگنال نهایی: تجمیع همه ماژول ها
# ============================================================================
def build_master_signal(df: pd.DataFrame, struct: Dict, fvgs: List[Dict],
                        obs: List[Dict], liq: Dict, vprof: Dict,
                        sweeps: List[Dict], blocks: Dict, flow: Dict,
                        fake: Dict, hft: Dict, coalition: Dict,
                        htf_bias: Optional[int] = None,
                        recent: int = 12) -> Dict:
    n = len(df)
    price = float(df["Close"].iloc[-1])
    atr = float(atr_array(df)[-1])
    score = 0.0
    parts: List[Tuple[str, float, str]] = []

    # --- ساختار تایم فریم بالاتر
    if htf_bias is not None and htf_bias != 0:
        w = 15 * (1 if htf_bias > 0 else -1)
        score += w
        parts.append(("ساختار تایم فریم بالاتر (روزانه)", w,
                      "صعودی" if htf_bias > 0 else "نزولی"))

    # --- ساختار تایم فریم جاری (BOS / CHoCH)
    ev = [e for e in struct["events"] if e["i"] >= n - recent]
    if ev:
        last = ev[-1]
        base = 22 if last["type"] == "CHoCH" else 18
        w = base * (1 if last["dir"] == "bull" else -1)
        score += w
        parts.append((f"{last['type']} در {recent} کندل اخیر", w,
                      f"{'صعودی' if last['dir'] == 'bull' else 'نزولی'} @ {last['level']:.2f}"))
    elif struct["bias"] != 0:
        w = 8 * struct["bias"]
        score += w
        parts.append(("بایاس ساختاری جاری", w,
                      "صعودی" if struct["bias"] > 0 else "نزولی"))

    # --- شکار نقدینگی
    sw = [s for s in sweeps if s["i"] >= n - recent]
    if sw:
        last = max(sw, key=lambda s: s["i"])
        w = 20 * (1 if last["dir"] == "bull" else -1)
        w *= min(1.0, 0.6 + 0.2 * max(last["vol_z"], 0))
        score += w
        parts.append((f"{last['type']} (شکار نقدینگی)", w,
                      f"سطح {last['level']:.2f} | عمق {last['depth_atr']:.2f} ATR"))

    # --- FVG / Order Block نزدیک و پرنشده
    act_bull = [g for g in fvgs if g["dir"] == "bull" and not g["filled"]
                and g["bottom"] <= price * 1.02 and g["i"] > n - 120]
    act_bear = [g for g in fvgs if g["dir"] == "bear" and not g["filled"]
                and g["top"] >= price * 0.98 and g["i"] > n - 120]
    nb = len(act_bull)
    ns = len(act_bear)
    if nb or ns:
        w = float(np.clip((nb - ns) * 4.5, -12, 12))
        score += w
        parts.append(("عدم توازن FVG فعال", w, f"{nb} صعودی / {ns} نزولی"))

    ob_bull = [b for b in obs if b["dir"] == "bull" and not b["mitigated"] and b["i"] > n - 120]
    ob_bear = [b for b in obs if b["dir"] == "bear" and not b["mitigated"] and b["i"] > n - 120]
    if ob_bull or ob_bear:
        w = float(np.clip((len(ob_bull) - len(ob_bear)) * 3.5, -9, 9))
        score += w
        parts.append(("بلوک سفارش دست نخورده", w,
                      f"{len(ob_bull)} صعودی / {len(ob_bear)} نزولی"))

    # --- جریان سفارش نهادی
    of = 0.0
    of += np.clip(flow["cd_slope"] * 10, -1, 1) * 6
    of += np.clip(flow["obv_slope"] * 10, -1, 1) * 3
    of += np.clip(flow["cmf_last"] * 10, -1, 1) * 3
    of += (1.5 if flow["mfi_last"] < 25 else (-1.5 if flow["mfi_last"] > 78 else 0))
    score += of
    parts.append(("جریان پول نهادی (Delta/OBV/CMF/MFI)", float(of),
                  f"CMF={flow['cmf_last']:+.3f} | MFI={flow['mfi_last']:.0f}"))

    if flow.get("smi_trend") is not None:
        w = float(np.clip(flow["smi_trend"] * 6, -6, 6))
        score += w
        parts.append(("شاخص پول هوشمند (SMI)", w,
                      "صعودی" if w > 0 else "نزولی"))

    # --- سفارشات سنگین
    if blocks["n_blocks"] > 0:
        w = float(np.clip(blocks["imbalance"] * 11, -11, 11))
        score += w
        parts.append(("عدم توازن سفارشات سنگین", w,
                      f"{blocks['n_blocks']} بلوک | نامتوازنی {blocks['imbalance'] * 100:+.0f}%"))

    # --- ائتلاف بانک ها
    if coalition.get("ok"):
        w = float(np.clip(coalition["score"] * 14, -10, 10)) * \
            (1.0 if coalition["agreement"] >= 0.7 else 0.5)
        score += w
        parts.append(("ائتلاف بانک ها", w,
                      f"امتیاز {coalition['score']:+.2f} | هماهنگی {coalition['agreement'] * 100:.0f}%"))

    # --- پروفایل حجم / گره ها
    if vprof:
        poc = vprof["poc"]
        if price < vprof["val"]:
            w = 7
            note = "زیر ناحیه ارزش - کشش بازگشتی به گره POC"
        elif price > vprof["vah"]:
            w = -7
            note = "بالای ناحیه ارزش - کشش اصلاحی به گره POC"
        else:
            w = 3 if price > poc else -3
            note = "داخل ناحیه ارزش (تعادل)"
        score += w
        parts.append(("موقعیت نسبت به گره حجمی", float(w),
                      f"{note} | POC={poc:.2f}"))

    # --- جریمه روند فیک و محیط HFT
    raw = score
    fake_pen = 1 - (fake["score"] / 100) * 0.55
    hft_pen = 1 - max(0.0, (hft["hft_index"] - 55) / 100) * 0.35
    score = raw * fake_pen * hft_pen

    conf = float(np.clip(abs(score) / 70 * 100, 0, 97))
    if score >= 24:
        direction, label = 1, "خرید (LONG)"
    elif score <= -24:
        direction, label = -1, "فروش (SHORT)"
    elif score >= 12:
        direction, label = 1, "خرید ضعیف - منتظر تایید"
    elif score <= -12:
        direction, label = -1, "فروش ضعیف - منتظر تایید"
    else:
        direction, label = 0, "خنثی - بدون موقعیت"

    grade = "A+" if conf >= 75 else "A" if conf >= 62 else "B" if conf >= 48 \
        else "C" if conf >= 33 else "D"

    # ---------------- طرح معامله ----------------
    plan: Dict = {}
    if direction != 0:
        if direction > 0:
            zones = [(g["bottom"], g["top"]) for g in act_bull if g["top"] <= price] + \
                    [(b["bottom"], b["top"]) for b in ob_bull if b["top"] <= price]
            zones = sorted(zones, key=lambda z: -z[1])
            entry = float(np.mean(zones[0])) if zones else price
            entry = min(entry, price)
            sw_lows = [s["extreme"] for s in sweeps if s["dir"] == "bull" and s["i"] >= n - recent]
            struct_low = float(np.nanmin(df["Low"].to_numpy(float)[-recent:]))
            base_sl = min([struct_low] + sw_lows) if sw_lows else struct_low
            if zones:
                base_sl = min(base_sl, zones[0][0])
            sl = base_sl - 0.5 * atr
            tg = [x["price"] for x in liq.get("fresh_bsl", []) if x["price"] > entry]
            tg += [p["high"] for p in liq.get("periodic", {}).values() if p["high"] > entry]
            tg = sorted(set(tg))
            t1 = tg[0] if tg else entry + 1.5 * (entry - sl)
            t2 = tg[1] if len(tg) > 1 else entry + 2.5 * (entry - sl)
            t3 = tg[2] if len(tg) > 2 else entry + 4.0 * (entry - sl)
            if vprof and entry < vprof["poc"] < t2:
                t1 = min(t1, vprof["poc"])
        else:
            zones = [(g["bottom"], g["top"]) for g in act_bear if g["bottom"] >= price] + \
                    [(b["bottom"], b["top"]) for b in ob_bear if b["bottom"] >= price]
            zones = sorted(zones, key=lambda z: z[0])
            entry = float(np.mean(zones[0])) if zones else price
            entry = max(entry, price)
            sw_hi = [s["extreme"] for s in sweeps if s["dir"] == "bear" and s["i"] >= n - recent]
            struct_hi = float(np.nanmax(df["High"].to_numpy(float)[-recent:]))
            base_sl = max([struct_hi] + sw_hi) if sw_hi else struct_hi
            if zones:
                base_sl = max(base_sl, zones[0][1])
            sl = base_sl + 0.5 * atr
            tg = [x["price"] for x in liq.get("fresh_ssl", []) if x["price"] < entry]
            tg += [p["low"] for p in liq.get("periodic", {}).values() if p["low"] < entry]
            tg = sorted(set(tg), reverse=True)
            t1 = tg[0] if tg else entry - 1.5 * (sl - entry)
            t2 = tg[1] if len(tg) > 1 else entry - 2.5 * (sl - entry)
            t3 = tg[2] if len(tg) > 2 else entry - 4.0 * (sl - entry)
            if vprof and t2 < vprof["poc"] < entry:
                t1 = max(t1, vprof["poc"])

        risk = abs(entry - sl)
        # حذف اهدافی که نسبت سود به زیان قابل قبول ندارند (حداقل 1R)
        if risk > 0:
            tps = sorted({t1, t2, t3}, reverse=(direction < 0))
            tps = [t for t in tps if abs(t - entry) / risk >= 1.0]
            if direction > 0:
                fallback = [entry + m * risk for m in (1.5, 2.5, 4.0)]
            else:
                fallback = [entry - m * risk for m in (1.5, 2.5, 4.0)]
            while len(tps) < 3:
                nxt = fallback[len(tps)]
                if not tps or (abs(nxt - entry) > abs(tps[-1] - entry)):
                    tps.append(nxt)
                else:
                    tps.append(tps[-1] + (risk if direction > 0 else -risk))
            t1, t2, t3 = tps[0], tps[1], tps[2]
        rr1 = abs(t1 - entry) / risk if risk else 0
        rr2 = abs(t2 - entry) / risk if risk else 0
        rr3 = abs(t3 - entry) / risk if risk else 0
        plan = dict(entry=float(entry), stop=float(sl), tp1=float(t1),
                    tp2=float(t2), tp3=float(t3), risk=float(risk),
                    rr1=float(rr1), rr2=float(rr2), rr3=float(rr3),
                    risk_pct=float(risk / entry * 100),
                    at_market=bool(abs(entry - price) / price < 0.001))

    return dict(score=float(score), raw_score=float(raw), confidence=conf,
                direction=direction, label=label, grade=grade, parts=parts,
                plan=plan, fake_penalty=float(fake_pen), hft_penalty=float(hft_pen),
                price=price, atr=atr)


# ============================================================================
# سیگنال های علّی برای بکتست (بدون دید به آینده)
# ============================================================================
def causal_smc_signals(df: pd.DataFrame, left: int = 2, right: int = 2,
                       sweep_win: int = 12, confirm: int = 6,
                       htf_bias_series: Optional[np.ndarray] = None,
                       regime_filter: bool = True) -> pd.DataFrame:
    """
    ستون های ورود/خروج SMC که فقط از اطلاعات گذشته ساخته می شوند (بدون دید به آینده).

    الگوی ورود (مدل ICT):
       1) شکار نقدینگی: قیمت کف/سقف دوره را می زند و داخل برمی گردد (Sweep)
       2) بازپس گیری سطح: بسته شدن کندل بعدی داخل محدوده (Reclaim)
       3) تایید: CHoCH/BOS هم جهت  یا  کندل جابجایی (Displacement)
       4) فیلتر: دلتای تجمعی هم جهت + فیلتر روند فیک
    """
    out = df.copy()
    n = len(df)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    o = df["Open"].to_numpy(float)
    atr = atr_array(df)
    st = market_structure(df, left, right)
    adx = _adx(df)
    cd = np.cumsum(delta_proxy(df))
    cd_sl = pd.Series(cd).diff(5).to_numpy()
    ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().to_numpy()

    ph = pd.Series(h).rolling(sweep_win).max().shift(1).to_numpy()
    pl = pd.Series(l).rolling(sweep_win).min().shift(1).to_numpy()
    rng = np.where((h - l) == 0, np.nan, h - l)
    up_w = np.nan_to_num((h - np.maximum(o, c)) / rng)
    dn_w = np.nan_to_num((np.minimum(o, c) - l) / rng)

    # --- گام 1: شکار نقدینگی
    sweep_lo = np.zeros(n, bool)   # کف زده شد -> سوگیری صعودی
    sweep_hi = np.zeros(n, bool)   # سقف زده شد -> سوگیری نزولی
    sweep_lo_lvl = np.full(n, np.nan)
    sweep_hi_lvl = np.full(n, np.nan)
    for i in range(sweep_win + 1, n):
        if not np.isnan(pl[i]) and l[i] < pl[i] and c[i] > pl[i] and dn_w[i] > 0.22:
            sweep_lo[i] = True
            sweep_lo_lvl[i] = l[i]
        if not np.isnan(ph[i]) and h[i] > ph[i] and c[i] < ph[i] and up_w[i] > 0.22:
            sweep_hi[i] = True
            sweep_hi_lvl[i] = h[i]

    # --- گام 3: تاییدیه ساختاری
    choch_b = np.zeros(n, bool)
    choch_s = np.zeros(n, bool)
    for e in st["events"]:
        if e["dir"] == "bull":
            choch_b[e["i"]] = True
        else:
            choch_s[e["i"]] = True

    # کندل جابجایی (Displacement) به عنوان تایید جایگزین
    body = np.abs(c - o)
    disp_b = (c > o) & (body > 0.62 * np.nan_to_num(rng, nan=1e9)) & (body > 0.75 * atr)
    disp_s = (c < o) & (body > 0.62 * np.nan_to_num(rng, nan=1e9)) & (body > 0.75 * atr)

    long_sig = np.zeros(n, bool)
    short_sig = np.zeros(n, bool)
    sl_long = np.full(n, np.nan)
    sl_short = np.full(n, np.nan)

    for i in range(max(sweep_win + right + 3, 55), n):
        w0 = max(0, i - confirm)

        # ---------- لانگ ----------
        sw_idx = [k for k in range(w0, i + 1) if sweep_lo[k]]
        if sw_idx:
            k = sw_idx[-1]
            lvl = sweep_lo_lvl[k]
            reclaim = c[i] > max(pl[k], c[k]) if not np.isnan(pl[k]) else c[i] > c[k]
            confirmed = choch_b[i] or disp_b[i] or (i > k and c[i] > h[k])
            flow_ok = cd_sl[i] > 0
            regime_ok = adx[i] > 11 and c[i] > ema50[i] * 0.975
            if reclaim and confirmed and flow_ok and regime_ok:
                long_sig[i] = True
                sl_long[i] = min(lvl, np.nanmin(l[w0:i + 1])) - 0.35 * atr[i]

        # ---------- شورت ----------
        sw_idx = [k for k in range(w0, i + 1) if sweep_hi[k]]
        if sw_idx:
            k = sw_idx[-1]
            lvl = sweep_hi_lvl[k]
            reclaim = c[i] < min(ph[k], c[k]) if not np.isnan(ph[k]) else c[i] < c[k]
            confirmed = choch_s[i] or disp_s[i] or (i > k and c[i] < l[k])
            flow_ok = cd_sl[i] < 0
            regime_ok = adx[i] > 11 and c[i] < ema50[i] * 1.025
            if reclaim and confirmed and flow_ok and regime_ok:
                short_sig[i] = True
                sl_short[i] = max(lvl, np.nanmax(h[w0:i + 1])) + 0.35 * atr[i]

    if htf_bias_series is not None and len(htf_bias_series) == n:
        long_sig &= (htf_bias_series >= 0)
        short_sig &= (htf_bias_series <= 0)

    # فیلتر رژیم بازار: معامله فقط هم جهت با روند بلندمدت (اصل ICT)
    if regime_filter:
        ema200 = pd.Series(c).ewm(span=200, adjust=False).mean().to_numpy()
        slope200 = pd.Series(ema200).diff(20).to_numpy()
        bull_regime = (c > ema200) & (slope200 > 0)
        bear_regime = (c < ema200) & (slope200 < 0)
        long_sig &= bull_regime
        short_sig &= bear_regime

    out["SMC_Long"] = long_sig.astype(int)
    out["SMC_Short"] = short_sig.astype(int)
    out["SMC_SL_Long"] = sl_long
    out["SMC_SL_Short"] = sl_short
    out["SMC_ATR"] = atr
    out["SMC_Bias"] = st["bias_series"]
    out["SMC_ADX"] = adx
    return out


# ============================================================================
# اجرای کامل تحلیل
# ============================================================================
def run_full_smc(df: pd.DataFrame, interval: str = "1d",
                 htf_df: Optional[pd.DataFrame] = None,
                 intraday: Optional[pd.DataFrame] = None,
                 with_coalition: bool = True) -> Dict:
    left = right = 3 if interval != "1d" else 4
    struct = market_structure(df, left, right)
    fvgs = find_fvg(df)
    obs = find_order_blocks(df)
    liq = liquidity_levels(df, left, right)
    vprof = volume_profile(df)
    sweeps = detect_sweeps(df, liq)
    blocks = block_order_scanner(df)
    flow = institutional_flow(df, intraday)
    hft = hft_footprint(df)
    fake = fake_trend_detector(df, struct, flow)

    htf_bias = None
    if htf_df is not None and len(htf_df) > 60:
        htf_bias = market_structure(htf_df, 4, 4)["bias"]

    coalition = bank_coalition(interval="1d", period="6mo", ref=df) if with_coalition \
        else dict(ok=False, members=[], score=0.0, agreement=0.0)

    signal = build_master_signal(df, struct, fvgs, obs, liq, vprof, sweeps,
                                 blocks, flow, fake, hft, coalition, htf_bias)

    return dict(struct=struct, fvgs=fvgs, obs=obs, liq=liq, vprof=vprof,
                sweeps=sweeps, blocks=blocks, flow=flow, hft=hft, fake=fake,
                coalition=coalition, htf_bias=htf_bias, signal=signal,
                interval=interval)
