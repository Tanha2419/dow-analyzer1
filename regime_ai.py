#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regime_ai.py — بخش ۲ (رژیم و واگرایی) + بخش ۴ (قابلیت های AI/ML)

  * Hurst Exponent + Kalman Filter برای تشخیص رژیم بازار
  * WaveTrend + واگرایی معمولی و مخفی (Hidden Divergence)
  * Isolation Forest برای تشخیص ناهنجاری حجمی (ردپای نهادی)
  * Gradient Boosting برای پیش بینی احتمال حرکت ۱۰ کندل آینده
  * SuperTrend و Chandelier Exit برای تریلینگ استاپ
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN = True
except Exception:
    SKLEARN = False


# ================================================================ کمکی
def _atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(n, min_periods=1).mean().to_numpy()


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
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.where(_atr(df, n) == 0, np.nan, _atr(df, n))
    pdi = 100 * pd.Series(pdm).ewm(alpha=1 / n, adjust=False).mean().to_numpy() / tr
    mdi = 100 * pd.Series(mdm).ewm(alpha=1 / n, adjust=False).mean().to_numpy() / tr
    dx = 100 * np.abs(pdi - mdi) / np.where((pdi + mdi) == 0, np.nan, pdi + mdi)
    return pd.Series(dx).ewm(alpha=1 / n, adjust=False).mean().fillna(0).to_numpy()


# ================================================================ Hurst
def hurst_exponent(series: np.ndarray, max_lag: int = 40) -> float:
    """
    نمای هرست با روش Rescaled Range ساده شده.
      H > 0.55  -> رونددار (persistent)
      H ~ 0.50  -> تصادفی
      H < 0.45  -> بازگشت به میانگین (mean-reverting)
    """
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < max_lag * 2:
        max_lag = max(8, len(s) // 3)
    if len(s) < 20:
        return 0.5
    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        d = s[lag:] - s[:-lag]
        tau.append(np.sqrt(np.std(d)) if len(d) else np.nan)
    tau = np.array(tau, dtype=float)
    ok = np.isfinite(tau) & (tau > 0)
    if ok.sum() < 5:
        return 0.5
    x = np.log(np.array(list(lags))[ok])
    y = np.log(tau[ok])
    h = float(np.polyfit(x, y, 1)[0] * 2.0)
    return float(np.clip(h, 0.0, 1.0))


def kalman_trend(series: np.ndarray, q: float = 1e-4,
                 r: float = 1e-2) -> Tuple[np.ndarray, np.ndarray]:
    """
    فیلتر کالمن یک بعدی (سطح + شیب) برای استخراج روند پنهان.
    خروجی: (سطح هموارشده، شیب تخمینی)
    """
    z = np.asarray(series, dtype=float)
    n = len(z)
    if n < 5:
        return z.copy(), np.zeros(n)

    x = np.array([z[0], 0.0])              # [سطح، شیب]
    P = np.eye(2)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[q, 0.0], [0.0, q * 0.1]])
    R = np.array([[r * max(1.0, np.var(np.diff(z)) if n > 2 else 1.0)]])

    lvl = np.zeros(n)
    slp = np.zeros(n)
    for i in range(n):
        x = F @ x
        P = F @ P @ F.T + Q
        y = z[i] - (H @ x)[0]
        S = (H @ P @ H.T + R)[0, 0]
        K = (P @ H.T / S).ravel()
        x = x + K * y
        P = (np.eye(2) - np.outer(K, H)) @ P
        lvl[i], slp[i] = x[0], x[1]
    return lvl, slp


def detect_regime(df: pd.DataFrame, lookback: int = 220) -> Dict:
    """تشخیص رژیم بازار: رونددار / رنج / پرنوسان."""
    d = df.tail(min(lookback, len(df)))
    c = d["Close"].to_numpy(float)
    if len(c) < 40:
        return dict(regime="نامشخص", label="نامشخص", hurst=0.5,
                    confidence=0, strategy="—")

    logp = np.log(c)
    H = hurst_exponent(logp)
    lvl, slp = kalman_trend(logp)
    k_slope = float(slp[-1] * 1e4)          # شیب در واحد پایه

    adx = _adx(d)[-1]
    atr = _atr(d)
    atr_pct = float(atr[-1] / c[-1] * 100)
    atr_ma = float(np.nanmean(atr[-60:] / c[-60:] * 100)) if len(c) >= 60 else atr_pct
    vol_ratio = atr_pct / max(atr_ma, 1e-9)

    ret = np.diff(logp)
    net = abs(logp[-1] - logp[-min(20, len(logp) - 1)])
    path = np.abs(ret[-min(20, len(ret)):]).sum()
    efficiency = float(net / max(path, 1e-9))

    # امتیازدهی رژیم
    trend_s = 0.0
    trend_s += np.clip((H - 0.5) / 0.15, -1, 1) * 35
    trend_s += np.clip((adx - 20) / 15, -1, 1) * 30
    trend_s += np.clip((efficiency - 0.25) / 0.25, -1, 1) * 35
    volatile = vol_ratio > 1.45

    if volatile and trend_s < 25:
        regime, label = "volatile", "پرنوسان (Volatile)"
        strategy = "کاهش حجم، فقط معاملات با تایید چندگانه"
    elif trend_s >= 28:
        regime, label = "trending", "رونددار (Trending)"
        strategy = "استراتژی شکست و ادامه روند (Breakout/Continuation)"
    elif trend_s <= -12:
        regime, label = "ranging", "رنج (Ranging)"
        strategy = "بازگشت به میانگین بین مرزهای ناحیه ارزش"
    else:
        regime, label = "transition", "در حال گذار (Transition)"
        strategy = "صبر تا تثبیت رژیم، حجم کم"

    direction = 0
    if regime == "trending":
        direction = 1 if k_slope > 0 else -1

    return dict(
        regime=regime, label=label, strategy=strategy,
        hurst=round(H, 4),
        hurst_note=("رونددار (پایدار)" if H > 0.55 else
                    ("بازگشت به میانگین" if H < 0.45 else "نزدیک تصادفی")),
        kalman_slope=round(k_slope, 4),
        kalman_level=float(np.exp(lvl[-1])),
        adx=round(float(adx), 2),
        atr_pct=round(atr_pct, 3), vol_ratio=round(vol_ratio, 3),
        efficiency=round(efficiency, 4),
        trend_score=round(float(trend_s), 2),
        direction=direction,
        confidence=int(np.clip(abs(trend_s), 0, 100)),
    )


# ================================================================ WaveTrend
def wavetrend(df: pd.DataFrame, n1: int = 10, n2: int = 21) -> Dict:
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    ap = (h + l + c) / 3
    esa = pd.Series(ap).ewm(span=n1, adjust=False).mean()
    de = pd.Series(np.abs(ap - esa)).ewm(span=n1, adjust=False).mean()
    ci = (ap - esa) / (0.015 * de.replace(0, np.nan))
    tci = ci.ewm(span=n2, adjust=False).mean()
    wt1 = tci.fillna(0).to_numpy()
    wt2 = pd.Series(wt1).rolling(4).mean().fillna(0).to_numpy()
    return dict(wt1=wt1, wt2=wt2,
                last=float(wt1[-1]), signal=float(wt2[-1]),
                cross_up=bool(wt1[-1] > wt2[-1] and wt1[-2] <= wt2[-2]),
                cross_dn=bool(wt1[-1] < wt2[-1] and wt1[-2] >= wt2[-2]),
                zone=("اشباع خرید" if wt1[-1] > 53 else
                      ("اشباع فروش" if wt1[-1] < -53 else "خنثی")))


def _pivots(x: np.ndarray, left: int = 4, right: int = 4):
    n = len(x)
    hi, lo = [], []
    for i in range(left, n - right):
        w = x[i - left:i + right + 1]
        if w.argmax() == left:
            hi.append(i)
        if w.argmin() == left:
            lo.append(i)
    return hi, lo


def find_divergences(df: pd.DataFrame, lookback: int = 90) -> Dict:
    """
    واگرایی معمولی (Regular) = هشدار بازگشت روند
    واگرایی مخفی (Hidden)    = تایید ادامه روند (ورود در جهت روند)
    """
    d = df.tail(min(lookback, len(df)))
    c = d["Close"].to_numpy(float)
    hi_p = d["High"].to_numpy(float)
    lo_p = d["Low"].to_numpy(float)
    if len(c) < 30:
        return dict(items=[], bias=0)

    rsi = _rsi(c)
    wt = wavetrend(d)["wt1"]
    ph, pl = _pivots(c, 4, 4)
    items: List[Dict] = []

    def add(kind, dirn, i1, i2, ind):
        items.append(dict(
            kind=kind, dir=dirn, indicator=ind,
            from_i=int(i1), to_i=int(i2),
            from_time=pd.Timestamp(d.index[i1]).isoformat(),
            to_time=pd.Timestamp(d.index[i2]).isoformat(),
            bars_ago=int(len(c) - 1 - i2),
        ))

    for osc, nm in ((rsi, "RSI"), (wt, "WaveTrend")):
        # --- سقف ها ---
        for a, b in zip(ph, ph[1:]):
            if b - a < 3 or (len(c) - 1 - b) > 25:
                continue
            if hi_p[b] > hi_p[a] and osc[b] < osc[a] - 1.5:
                add("regular", "bear", a, b, nm)      # سقف بالاتر، اندیکاتور پایین تر
            elif hi_p[b] < hi_p[a] and osc[b] > osc[a] + 1.5:
                add("hidden", "bear", a, b, nm)       # سقف پایین تر، اندیکاتور بالاتر
        # --- کف ها ---
        for a, b in zip(pl, pl[1:]):
            if b - a < 3 or (len(c) - 1 - b) > 25:
                continue
            if lo_p[b] < lo_p[a] and osc[b] > osc[a] + 1.5:
                add("regular", "bull", a, b, nm)      # کف پایین تر، اندیکاتور بالاتر
            elif lo_p[b] > lo_p[a] and osc[b] < osc[a] - 1.5:
                add("hidden", "bull", a, b, nm)       # کف بالاتر، اندیکاتور پایین تر

    items = sorted(items, key=lambda x: x["bars_ago"])[:8]
    bias = 0.0
    for it in items:
        w = 1.0 if it["kind"] == "hidden" else 0.8
        w *= max(0.3, 1 - it["bars_ago"] / 30)
        bias += w * (1 if it["dir"] == "bull" else -1)

    return dict(items=items, bias=float(np.clip(bias, -3, 3)),
                n_hidden=sum(1 for i in items if i["kind"] == "hidden"),
                n_regular=sum(1 for i in items if i["kind"] == "regular"))


# ================================================================ تریلینگ
def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> Dict:
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    atr = _atr(df, period)
    hl2 = (h + l) / 2
    ub = hl2 + mult * atr
    lb = hl2 - mult * atr
    n = len(c)
    fu, fl = ub.copy(), lb.copy()
    dirn = np.ones(n, dtype=int)
    for i in range(1, n):
        fu[i] = min(ub[i], fu[i - 1]) if c[i - 1] <= fu[i - 1] else ub[i]
        fl[i] = max(lb[i], fl[i - 1]) if c[i - 1] >= fl[i - 1] else lb[i]
        if c[i] > fu[i - 1]:
            dirn[i] = 1
        elif c[i] < fl[i - 1]:
            dirn[i] = -1
        else:
            dirn[i] = dirn[i - 1]
    line = np.where(dirn == 1, fl, fu)
    return dict(line=line, direction=dirn,
                last=float(line[-1]), dir=int(dirn[-1]),
                flipped=bool(dirn[-1] != dirn[-2]) if n > 1 else False)


def chandelier_exit(df: pd.DataFrame, period: int = 22,
                    mult: float = 3.0) -> Dict:
    h = pd.Series(df["High"].to_numpy(float)).rolling(period).max()
    l = pd.Series(df["Low"].to_numpy(float)).rolling(period).min()
    atr = _atr(df, period)
    long_stop = (h - mult * atr).bfill().to_numpy()
    short_stop = (l + mult * atr).bfill().to_numpy()
    return dict(long_stop=long_stop, short_stop=short_stop,
                long_last=float(long_stop[-1]),
                short_last=float(short_stop[-1]))


# ================================================================ Isolation Forest
def anomaly_detection(df: pd.DataFrame, lookback: int = 400,
                      contamination: float = 0.04) -> Dict:
    """
    تشخیص ناهنجاری چندبعدی با Isolation Forest.
    هدف: یافتن ردپای نهادی (حجم/دامنه/دلتای غیرعادی) پیش از حرکت اصلی.
    """
    if not SKLEARN:
        return dict(ok=False, error="scikit-learn نصب نیست", items=[])

    d = df.tail(min(lookback, len(df))).copy()
    if len(d) < 60:
        return dict(ok=False, error="داده کافی نیست", items=[])

    o = d["Open"].to_numpy(float)
    h = d["High"].to_numpy(float)
    l = d["Low"].to_numpy(float)
    c = d["Close"].to_numpy(float)
    v = d["Volume"].to_numpy(float)
    atr = _atr(d)

    rng = np.where((h - l) == 0, np.nan, h - l)
    clv = np.nan_to_num(((c - l) - (h - c)) / rng)
    body = np.abs(c - o) / np.nan_to_num(rng, nan=1.0)
    ret = np.concatenate([[0.0], np.diff(c) / c[:-1]]) * 100
    vma = pd.Series(v).rolling(20).mean().bfill().to_numpy()
    vrel = v / np.where(vma == 0, np.nan, vma)
    rrel = np.nan_to_num(rng, nan=0) / np.where(atr == 0, np.nan, atr)
    delta = clv * v
    drel = delta / np.where(vma == 0, np.nan, vma)

    X = np.column_stack([
        np.nan_to_num(vrel, nan=1.0),
        np.nan_to_num(rrel, nan=1.0),
        np.nan_to_num(ret, nan=0.0),
        np.nan_to_num(body, nan=0.5),
        np.nan_to_num(clv, nan=0.0),
        np.nan_to_num(drel, nan=0.0),
    ])
    Xs = StandardScaler().fit_transform(X)

    iso = IsolationForest(n_estimators=220, contamination=contamination,
                          random_state=42, n_jobs=1)
    pred = iso.fit_predict(Xs)
    score = iso.score_samples(Xs)          # هرچه منفی تر، ناهنجارتر
    thr = float(np.percentile(score, contamination * 100))

    items: List[Dict] = []
    for i in range(len(d)):
        if pred[i] == -1:
            side = "BUY" if clv[i] > 0.15 else ("SELL" if clv[i] < -0.15 else "NEUTRAL")
            items.append(dict(
                i=int(i), time=pd.Timestamp(d.index[i]).isoformat(),
                price=float(c[i]), vol=float(v[i]),
                vol_rel=float(np.nan_to_num(vrel[i], nan=1)),
                range_atr=float(np.nan_to_num(rrel[i], nan=1)),
                ret=float(ret[i]), clv=float(clv[i]), side=side,
                score=float(score[i]),
                severity=float(np.clip((thr - score[i]) / max(abs(thr), 1e-6), 0, 3)),
            ))

    items = sorted(items, key=lambda x: x["i"])
    recent = [x for x in items if x["i"] >= len(d) - 20]
    buy_n = sum(1 for x in recent if x["side"] == "BUY")
    sell_n = sum(1 for x in recent if x["side"] == "SELL")

    bias = 0.0
    for x in recent:
        w = (1 if x["side"] == "BUY" else (-1 if x["side"] == "SELL" else 0))
        w *= min(1.0, x["severity"]) * max(0.3, 1 - (len(d) - 1 - x["i"]) / 25)
        bias += w

    return dict(ok=True, n_total=len(items), n_recent=len(recent),
                buy=buy_n, sell=sell_n, bias=float(np.clip(bias, -3, 3)),
                items=items[-14:],
                note=("ردپای نهادی خرید غالب" if bias > 0.5 else
                      ("ردپای نهادی فروش غالب" if bias < -0.5 else
                       "ناهنجاری جهت دار مشخصی نیست")))


# ================================================================ پیش بینی ML
FEATURES = [
    "ret1", "ret5", "ret10", "rsi", "rsi_d", "adx", "atr_pct",
    "vol_rel", "clv", "delta_rel", "ema_dist", "ema_slope",
    "bb_pos", "hurst_w", "wt", "range_atr",
]


def _make_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    v = df["Volume"].to_numpy(float)
    atr = _atr(df)
    rsi = _rsi(c)
    adx = _adx(df)
    ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().to_numpy()
    sma20 = pd.Series(c).rolling(20).mean().to_numpy()
    std20 = pd.Series(c).rolling(20).std().to_numpy()
    vma = pd.Series(v).rolling(20).mean().to_numpy()
    rng = np.where((h - l) == 0, np.nan, h - l)
    clv = np.nan_to_num(((c - l) - (h - c)) / rng)
    wt = wavetrend(df)["wt1"]

    hw = pd.Series(np.log(c)).rolling(64).apply(
        lambda s: hurst_exponent(s.to_numpy(), 20), raw=False).to_numpy()

    f = pd.DataFrame(index=df.index)
    f["ret1"] = pd.Series(c).pct_change(1).to_numpy() * 100
    f["ret5"] = pd.Series(c).pct_change(5).to_numpy() * 100
    f["ret10"] = pd.Series(c).pct_change(10).to_numpy() * 100
    f["rsi"] = rsi
    f["rsi_d"] = pd.Series(rsi).diff(3).to_numpy()
    f["adx"] = adx
    f["atr_pct"] = atr / c * 100
    f["vol_rel"] = v / np.where(vma == 0, np.nan, vma)
    f["clv"] = clv
    f["delta_rel"] = clv * v / np.where(vma == 0, np.nan, vma)
    f["ema_dist"] = (c - ema20) / c * 100
    f["ema_slope"] = pd.Series(ema20).pct_change(5).to_numpy() * 100
    f["bb_pos"] = (c - sma20) / np.where(std20 == 0, np.nan, 2 * std20)
    f["hurst_w"] = hw
    f["wt"] = wt
    f["range_atr"] = np.nan_to_num(rng, nan=0) / np.where(atr == 0, np.nan, atr)
    return f[FEATURES]


def ml_forecast(df: pd.DataFrame, horizon: int = 10,
                min_train: int = 300) -> Dict:
    """
    پیش بینی احتمال صعود در N کندل آینده با Gradient Boosting.
    اعتبارسنجی Walk-Forward انجام می شود تا از بیش برازش جلوگیری شود.
    """
    if not SKLEARN:
        return dict(ok=False, error="scikit-learn نصب نیست")
    if len(df) < min_train + horizon + 60:
        return dict(ok=False, error="داده کافی برای آموزش نیست")

    X = _make_features(df)
    c = df["Close"].to_numpy(float)
    fwd = pd.Series(c).shift(-horizon).to_numpy() / c - 1.0
    y = (fwd > 0).astype(int)

    valid = X.notna().all(axis=1).to_numpy() & np.isfinite(fwd)
    Xv = X[valid].to_numpy(float)
    yv = y[valid]
    idx = np.where(valid)[0]
    if len(Xv) < min_train + 50:
        return dict(ok=False, error="نمونه معتبر کافی نیست")

    # ---------- اعتبارسنجی Walk-Forward ----------
    folds, accs = 5, []
    step = (len(Xv) - min_train) // folds
    oos_true, oos_pred = [], []
    if step > 10:
        for k in range(folds):
            tr_end = min_train + k * step
            te_end = min(tr_end + step, len(Xv))
            if te_end - tr_end < 8:
                continue
            sc = StandardScaler().fit(Xv[:tr_end])
            m = GradientBoostingClassifier(
                n_estimators=140, max_depth=3, learning_rate=0.05,
                subsample=0.85, random_state=42)
            m.fit(sc.transform(Xv[:tr_end]), yv[:tr_end])
            p = m.predict(sc.transform(Xv[tr_end:te_end]))
            accs.append(float((p == yv[tr_end:te_end]).mean()))
            oos_true.extend(yv[tr_end:te_end].tolist())
            oos_pred.extend(p.tolist())

    # ---------- مدل نهایی روی همه داده ----------
    sc = StandardScaler().fit(Xv)
    model = GradientBoostingClassifier(
        n_estimators=180, max_depth=3, learning_rate=0.05,
        subsample=0.85, random_state=42)
    model.fit(sc.transform(Xv), yv)

    last_row = X.iloc[[-1]]
    if last_row.isna().any(axis=1).iloc[0]:
        last_valid = X.dropna().iloc[[-1]]
        prob = float(model.predict_proba(sc.transform(last_valid.to_numpy(float)))[0, 1])
    else:
        prob = float(model.predict_proba(sc.transform(last_row.to_numpy(float)))[0, 1])

    imp = sorted(zip(FEATURES, model.feature_importances_),
                 key=lambda t: -t[1])[:6]

    oos_acc = float(np.mean(accs)) if accs else None
    base = float(yv.mean())
    edge = (oos_acc - max(base, 1 - base)) if oos_acc is not None else None

    if prob >= 0.60:
        label, direction = "احتمال صعود بالا", 1
    elif prob <= 0.40:
        label, direction = "احتمال نزول بالا", -1
    else:
        label, direction = "بدون لبه آماری مشخص", 0

    return dict(
        ok=True, horizon=horizon, prob_up=round(prob, 4),
        direction=direction, label=label,
        oos_accuracy=None if oos_acc is None else round(oos_acc, 4),
        folds=len(accs), base_rate=round(base, 4),
        edge=None if edge is None else round(edge, 4),
        trustworthy=bool(oos_acc is not None and oos_acc > 0.53),
        n_samples=int(len(Xv)),
        top_features=[dict(name=k, importance=round(float(w), 4)) for k, w in imp],
        note=("مدل لبه آماری معناداری نشان نمی دهد — با احتیاط استفاده شود"
              if (oos_acc is None or oos_acc <= 0.53)
              else "مدل در اعتبارسنجی خارج از نمونه عملکرد بهتر از تصادف دارد"),
    )


# ================================================================ تجمیع
def run_intelligence(df: pd.DataFrame, with_ml: bool = True) -> Dict:
    out: Dict = {}
    out["regime"] = detect_regime(df)
    out["wavetrend"] = {k: v for k, v in wavetrend(df).items()
                        if k not in ("wt1", "wt2")}
    out["divergence"] = find_divergences(df)
    st = supertrend(df)
    ce = chandelier_exit(df)
    out["trailing"] = dict(
        supertrend=dict(value=st["last"], dir=st["dir"], flipped=st["flipped"]),
        chandelier=dict(long_stop=ce["long_last"], short_stop=ce["short_last"]),
    )
    out["anomaly"] = anomaly_detection(df)
    if with_ml:
        out["ml"] = ml_forecast(df)
    return out
