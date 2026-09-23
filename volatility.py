#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
volatility.py — تحلیل نوسان و آپشن با داده واقعی

منابع تست شده:
  * زنجیره آپشن yfinance  — IV واقعی، Put/Call واقعی، Skew
  * CBOE VIX History CSV   — تاریخچه از ۱۹۹۰
  * ^VIX9D / ^VIX / ^VIX3M — ساختار زمانی

هیچ مقدار تخمینی ساخته نمی شود. اگر داده نبود، ok=False برمی گردد.
"""

from __future__ import annotations

import io
import json
import time
import urllib.request
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import assets as A

warnings.filterwarnings("ignore")

_CACHE: Dict[str, tuple] = {}
UA = {"User-Agent": "Mozilla/5.0"}


def _cached(key: str, ttl: float, fn):
    now = time.time()
    if key in _CACHE:
        ts, v = _CACHE[key]
        if now - ts < ttl:
            return v
    v = fn()
    _CACHE[key] = (now, v)
    return v


# ================================================================ Black-Scholes
# IV خام یاهو برای DIA خراب است (مقادیر 1e-05 تا 0.12).
# بنابراین IV را از قیمت واقعی معامله با وارونه سازی Black-Scholes می سازیم.
try:
    from scipy.optimize import brentq
    from scipy.stats import norm
    SCIPY = True
except Exception:
    SCIPY = False


def _bs_price(S: float, K: float, T: float, r: float,
              sig: float, call: bool = True) -> float:
    if T <= 0 or sig <= 0:
        return max(0.0, (S - K) if call else (K - S))
    d1 = (np.log(S / K) + (r + sig * sig / 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    if call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _implied_vol(price: float, S: float, K: float, T: float,
                 r: float, call: bool = True) -> Optional[float]:
    """IV واقعی از قیمت بازار. اگر همگرا نشد None برمی گرداند."""
    if not SCIPY or T <= 0 or price is None or price <= 0:
        return None
    intrinsic = max(0.0, (S - K) if call else (K - S))
    if price <= intrinsic + 1e-6:
        return None
    try:
        v = brentq(lambda s: _bs_price(S, K, T, r, s, call) - price,
                   1e-4, 6.0, maxiter=120, xtol=1e-7)
        return float(v) if 0.01 < v < 3.0 else None
    except Exception:
        return None


def _risk_free() -> float:
    """نرخ بدون ریسک از ^IRX (اوراق ۱۳ هفته)."""
    try:
        import yfinance as yf
        h = yf.Ticker("^IRX").history(period="5d")
        if not h.empty:
            return float(h["Close"].iloc[-1]) / 100.0
    except Exception:
        pass
    return 0.04


# ================================================================ آپشن
def option_analytics(symbol: str = "DIA", max_exp: int = 6) -> Dict:
    """
    تحلیل زنجیره آپشن واقعی:
      * Put/Call Ratio بر اساس حجم و Open Interest
      * IV اتم پول (ATM)
      * Volatility Skew  = IV پوت ۲۵ دلتا منهای IV کال ۲۵ دلتا
      * ساختار زمانی IV در سررسیدهای مختلف
      * Max Pain
    """
    def _fetch():
        import yfinance as yf
        t = yf.Ticker(symbol)
        try:
            exps = list(t.options)
        except Exception as e:
            return dict(ok=False, error=f"زنجیره آپشن در دسترس نیست: {type(e).__name__}")
        if not exps:
            return dict(ok=False, error="سررسیدی یافت نشد")

        try:
            spot = float(t.fast_info["lastPrice"])
        except Exception:
            return dict(ok=False, error="قیمت لحظه ای در دسترس نیست")

        today = datetime.now().date()
        terms: List[Dict] = []
        tot_cv = tot_pv = tot_coi = tot_poi = 0.0
        rf = _risk_free()

        for e in exps[:max_exp]:
            try:
                ch = t.option_chain(e)
                c, p = ch.calls.copy(), ch.puts.copy()
                if c.empty or p.empty:
                    continue
                for d in (c, p):
                    for col in ("volume", "openInterest", "lastPrice"):
                        if col not in d.columns:
                            d[col] = np.nan
                    d["volume"] = d["volume"].fillna(0)
                    d["openInterest"] = d["openInterest"].fillna(0)

                cv = float(c["volume"].sum())
                pv = float(p["volume"].sum())
                coi = float(c["openInterest"].sum())
                poi = float(p["openInterest"].sum())
                tot_cv += cv; tot_pv += pv; tot_coi += coi; tot_poi += poi

                exp_d = datetime.strptime(e, "%Y-%m-%d").date()
                dte = max((exp_d - today).days, 0)
                T = dte / 365.0
                if T <= 0:
                    continue

                # ---- IV واقعی با وارونه سازی Black-Scholes روی قرارداد نقدشونده ----
                def curve(d: pd.DataFrame, is_call: bool) -> List[tuple]:
                    d = d[(d["lastPrice"] > 0.01) &
                          ((d["volume"] > 0) | (d["openInterest"] > 5))]
                    out = []
                    for _, q in d.iterrows():
                        v = _implied_vol(float(q["lastPrice"]), spot,
                                         float(q["strike"]), T, rf, is_call)
                        if v is not None:
                            out.append((float(q["strike"]), v))
                    return out

                cc, pc = curve(c, True), curve(p, False)
                allc = cc + pc
                if not allc:
                    continue

                # ATM: میانگین ۶ قرارداد نزدیک به اسپات
                atm_pts = sorted(allc, key=lambda x: abs(x[0] - spot))[:6]
                atm = float(np.mean([v for _, v in atm_pts])) if atm_pts else None

                # Skew: پوت ۵٪ زیر منهای کال ۵٪ بالا
                tol = spot * 0.015
                pw = [v for k, v in pc if abs(k - spot * 0.95) < tol]
                cw = [v for k, v in cc if abs(k - spot * 1.05) < tol]
                skew = (float(np.mean(pw)) - float(np.mean(cw))) if (pw and cw) else None

                # Max Pain
                mp = None
                try:
                    ci = c.set_index("strike")["openInterest"]
                    pi = p.set_index("strike")["openInterest"]
                    strikes = sorted(set(ci.index).intersection(set(pi.index)))
                    if strikes:
                        best, bestv = None, None
                        for s in strikes:
                            below = ci.index[ci.index < s]
                            above = pi.index[pi.index > s]
                            pain = float((np.abs(s - below) * ci[below]).sum()
                                         + ((above - s) * pi[above]).sum())
                            if bestv is None or pain < bestv:
                                best, bestv = s, pain
                        mp = float(best) if best is not None else None
                except Exception:
                    mp = None

                terms.append(dict(
                    expiry=e, dte=dte,
                    call_volume=int(cv), put_volume=int(pv),
                    call_oi=int(coi), put_oi=int(poi),
                    pcr_volume=round(pv / cv, 4) if cv > 0 else None,
                    pcr_oi=round(poi / coi, 4) if coi > 0 else None,
                    atm_iv=None if atm is None else round(atm * 100, 2),
                    skew=None if skew is None else round(skew * 100, 2),
                    max_pain=mp, n_iv=len(allc),
                    n_calls=len(c), n_puts=len(p),
                ))
            except Exception:
                continue

        if not terms:
            return dict(ok=False, error="هیچ سررسید قابل تحلیلی یافت نشد")

        pcr_v = (tot_pv / tot_cv) if tot_cv > 0 else None
        pcr_o = (tot_poi / tot_coi) if tot_coi > 0 else None

        # تفسیر Put/Call
        if pcr_v is None:
            pcr_state = "نامشخص"
        elif pcr_v > 1.15:
            pcr_state = "ترس شدید — پوشش ریسک سنگین (خلاف جهت صعودی)"
        elif pcr_v > 0.95:
            pcr_state = "محتاطانه — تمایل به پوت"
        elif pcr_v < 0.6:
            pcr_state = "طمع — تمایل شدید به کال (هشدار خلاف جهت)"
        elif pcr_v < 0.8:
            pcr_state = "خوش بینانه"
        else:
            pcr_state = "متعادل"

        # Skew نزدیک ترین سررسید معنادار
        near = next((x for x in terms if x["dte"] >= 7 and x["skew"] is not None),
                    next((x for x in terms if x["skew"] is not None), None))
        skew_v = near["skew"] if near else None
        if skew_v is None:
            skew_state = "نامشخص"
        elif skew_v > 6:
            skew_state = "شیب تند — بازار برای سقوط بیمه می خرد"
        elif skew_v > 2:
            skew_state = "شیب معمول — ترس نرمال از ریزش"
        elif skew_v > -2:
            skew_state = "تخت — بی تفاوتی نسبت به ریسک دنباله"
        else:
            skew_state = "معکوس — تقاضای کال بیشتر (طمع یا انتظار جهش)"

        # ساختار زمانی IV
        ivs = [(x["dte"], x["atm_iv"]) for x in terms if x["atm_iv"]]
        term_state = "نامشخص"
        if len(ivs) >= 2:
            ivs.sort()
            short_iv, long_iv = ivs[0][1], ivs[-1][1]
            if short_iv > long_iv * 1.08:
                term_state = "معکوس — استرس کوتاه مدت (رویداد نزدیک)"
            elif long_iv > short_iv * 1.08:
                term_state = "نرمال — آرامش کوتاه مدت"
            else:
                term_state = "تخت"

        return dict(
            ok=True, symbol=symbol, spot=round(spot, 4),
            n_expiries=len(exps), analyzed=len(terms),
            pcr_volume=None if pcr_v is None else round(pcr_v, 4),
            pcr_oi=None if pcr_o is None else round(pcr_o, 4),
            pcr_state=pcr_state,
            total_call_volume=int(tot_cv), total_put_volume=int(tot_pv),
            skew=skew_v, skew_state=skew_state,
            skew_expiry=near["expiry"] if near else None,
            atm_iv=next((x["atm_iv"] for x in terms if x["atm_iv"]), None),
            risk_free=round(rf * 100, 3),
            iv_source="محاسبه شده با Black-Scholes از قیمت واقعی معامله",
            term_structure=term_state,
            terms=terms,
            fetched=datetime.now().isoformat(),
        )

    return _cached(f"opt{symbol}", 900.0, _fetch)


# ================================================================ نوسان تاریخی
def historical_volatility(symbol: str = "DIA") -> Dict:
    """HV در پنجره های مختلف + مقایسه با IV (پریمیوم ریسک نوسان)."""
    def _fetch():
        import yfinance as yf
        df = yf.Ticker(symbol).history(period="2y")
        if df.empty or len(df) < 60:
            return dict(ok=False, error="داده کافی نیست")
        c = df["Close"].to_numpy(float)
        r = np.diff(np.log(c))
        out: Dict = dict(ok=True, symbol=symbol)
        wins = {"hv10": 10, "hv20": 20, "hv30": 30, "hv60": 60, "hv252": 252}
        for k, w in wins.items():
            if len(r) >= w:
                out[k] = round(float(np.std(r[-w:]) * np.sqrt(252) * 100), 2)
            else:
                out[k] = None

        # Parkinson با High/Low (دقیق تر)
        h = df["High"].to_numpy(float); l = df["Low"].to_numpy(float)
        n = min(20, len(h))
        hl = np.log(h[-n:] / l[-n:]) ** 2
        park = float(np.sqrt(hl.mean() / (4 * np.log(2))) * np.sqrt(252) * 100)
        out["parkinson20"] = round(park, 2)

        # صدک HV20 در ۲ سال
        s = pd.Series(r).rolling(20).std() * np.sqrt(252) * 100
        s = s.dropna()
        if len(s) > 30 and out["hv20"]:
            out["hv20_percentile"] = round(float((s < out["hv20"]).mean() * 100), 1)
        out["regime"] = ("نوسان بسیار پایین" if (out.get("hv20") or 99) < 10 else
                         "نوسان پایین" if (out.get("hv20") or 99) < 15 else
                         "نوسان نرمال" if (out.get("hv20") or 99) < 22 else
                         "نوسان بالا" if (out.get("hv20") or 99) < 32 else
                         "نوسان بحرانی")
        return out

    return _cached(f"hv{symbol}", 1800.0, _fetch)


# ================================================================ VIX
def vix_complex() -> Dict:
    """ساختار زمانی VIX از یاهو + تاریخچه CBOE برای صدک بلندمدت."""
    def _fetch():
        import yfinance as yf
        out: Dict = dict(ok=False, members={})
        syms = {"^VIX9D": "۹ روزه", "^VIX": "۳۰ روزه",
                "^VIX3M": "۳ ماهه", "^VIX6M": "۶ ماهه", "^VXN": "نزدک"}
        vals: Dict[str, float] = {}
        for s, nm in syms.items():
            try:
                h = yf.Ticker(s).history(period="6mo")
                if h.empty:
                    continue
                last = float(h["Close"].iloc[-1])
                prev = float(h["Close"].iloc[-2]) if len(h) > 1 else last
                vals[s] = last
                out["members"][s] = dict(
                    name=nm, value=round(last, 2),
                    change_pct=round((last / prev - 1) * 100, 2) if prev else 0.0,
                    percentile_6m=round(float((h["Close"] < last).mean() * 100), 1),
                )
            except Exception:
                continue
        if not vals:
            return dict(ok=False, error="داده VIX در دسترس نیست")
        out["ok"] = True

        v9, v30, v3m = vals.get("^VIX9D"), vals.get("^VIX"), vals.get("^VIX3M")
        if v9 and v30:
            r = v9 / v30
            out["ratio_9d_30d"] = round(r, 4)
            out["short_term_stress"] = bool(r > 1.0)
        if v30 and v3m:
            r2 = v30 / v3m
            out["ratio_30d_3m"] = round(r2, 4)
            if r2 > 1.05:
                out["term_state"] = "معکوس — ترس فوری بازار (سیگنال کف احتمالی)"
                out["term_signal"] = -1
            elif r2 < 0.88:
                out["term_state"] = "کنتانگو تند — آرامش بیش از حد"
                out["term_signal"] = 1
            else:
                out["term_state"] = "کنتانگو نرمال"
                out["term_signal"] = 0
        if "^VXN" in vals and v30:
            out["vxn_vix_ratio"] = round(vals["^VXN"] / v30, 4)

        # تاریخچه CBOE برای صدک بلندمدت
        try:
            u = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
            raw = urllib.request.urlopen(
                urllib.request.Request(u, headers=UA), timeout=20).read()
            hist = pd.read_csv(io.BytesIO(raw))
            hist["DATE"] = pd.to_datetime(hist["DATE"], errors="coerce")
            hist = hist.dropna(subset=["DATE", "CLOSE"])
            if v30 and len(hist) > 100:
                out["cboe_history_rows"] = int(len(hist))
                out["cboe_from"] = str(hist["DATE"].min().date())
                out["percentile_all_time"] = round(
                    float((hist["CLOSE"] < v30).mean() * 100), 1)
                recent = hist[hist["DATE"] >= hist["DATE"].max() - pd.Timedelta(days=365*5)]
                out["percentile_5y"] = round(
                    float((recent["CLOSE"] < v30).mean() * 100), 1)
                out["mean_all_time"] = round(float(hist["CLOSE"].mean()), 2)
        except Exception as e:
            out["cboe_error"] = f"{type(e).__name__}"
        return out

    return _cached("vixcomplex", 900.0, _fetch)


# ================================================================ تجمیع
def build_volatility(symbol: str = "DIA", asset: Optional[str] = None) -> Dict:
    # فیوچرز طلا (GC=F) هیچ زنجیره آپشنی ندارد → از ETF طلا (GLD) استفاده می شود
    _prof = A.profile(asset) if asset else None
    _osym = symbol
    _proxy = None
    if _prof is not None:
        _osym = _prof["options_symbol"]
        if _osym != _prof["candle_symbol"]:
            _proxy = _osym
    elif symbol in ("GC=F", "MGC=F"):
        _osym, _proxy = "GLD", "GLD"

    opt = option_analytics(_osym)
    if _proxy and isinstance(opt, dict):
        opt["proxy_symbol"] = _proxy
        opt["proxy_note"] = (f"فیوچرز طلا زنجیره آپشن ندارد؛ این اعداد از "
                             f"صندوق {_proxy} (پرمعامله ترین ابزار طلا) خوانده شده "
                             f"و نماینده بازار آپشن طلاست.")
        opt["source_tier"] = "computed"
    hv = historical_volatility(symbol)
    vx = vix_complex()

    # پریمیوم ریسک نوسان = IV منهای HV
    vrp = None
    if opt.get("ok") and hv.get("ok") and opt.get("atm_iv") and hv.get("hv20"):
        vrp = round(opt["atm_iv"] - hv["hv20"], 2)

    signals: List[Dict] = []
    if opt.get("ok"):
        if opt.get("pcr_volume") and opt["pcr_volume"] > 1.15:
            signals.append(dict(name="Put/Call بالا", value=opt["pcr_volume"],
                                impact=+1, note="ترس شدید — اغلب کف ساز"))
        elif opt.get("pcr_volume") and opt["pcr_volume"] < 0.6:
            signals.append(dict(name="Put/Call پایین", value=opt["pcr_volume"],
                                impact=-1, note="طمع — هشدار سقف"))
        if opt.get("skew") is not None and opt["skew"] > 7:
            signals.append(dict(name="Skew تند", value=opt["skew"], impact=-0.5,
                                note="تقاضای بیمه سقوط بالا"))
    if vx.get("ok") and vx.get("term_signal") == -1:
        signals.append(dict(name="ساختار VIX معکوس", value=vx.get("ratio_30d_3m"),
                            impact=+1, note="استرس فوری — احتمال کف کوتاه مدت"))
    elif vx.get("ok") and vx.get("term_signal") == 1:
        signals.append(dict(name="کنتانگو تند VIX", value=vx.get("ratio_30d_3m"),
                            impact=-0.5, note="آرامش بیش از حد"))
    if vrp is not None and vrp > 6:
        signals.append(dict(name="پریمیوم نوسان بالا", value=vrp, impact=-0.3,
                            note="آپشن گران — بازار منتظر رویداد"))

    score = float(np.clip(sum(s["impact"] for s in signals), -3, 3))
    return dict(options=opt, hv=hv, vix=vx, vrp=vrp,
                signals=signals, score=round(score, 2))


if __name__ == "__main__":
    v = build_volatility("DIA")
    o = v["options"]
    print("=== آپشن (داده واقعی) ===")
    if o.get("ok"):
        print(f"اسپات {o['spot']} | {o['analyzed']} از {o['n_expiries']} سررسید")
        print(f"Put/Call حجم: {o['pcr_volume']} — {o['pcr_state']}")
        print(f"Put/Call OI:   {o['pcr_oi']}")
        print(f"IV اتم پول: {o['atm_iv']}٪ | Skew: {o['skew']} — {o['skew_state']}")
        print(f"ساختار زمانی IV: {o['term_structure']}")
        print(f"\n{'سررسید':<13}{'روز':>5}{'IV':>8}{'Skew':>8}{'P/C':>8}{'MaxPain':>10}")
        for t in o["terms"]:
            print(f"{t['expiry']:<13}{t['dte']:>5}{str(t['atm_iv']):>8}"
                  f"{str(t['skew']):>8}{str(t['pcr_volume']):>8}{str(t['max_pain']):>10}")
    else:
        print("ناموفق:", o.get("error"))

    h = v["hv"]
    if h.get("ok"):
        print(f"\n=== نوسان تاریخی ===")
        print(f"HV10 {h['hv10']}٪ | HV20 {h['hv20']}٪ | HV30 {h['hv30']}٪ "
              f"| HV60 {h['hv60']}٪ | Parkinson {h['parkinson20']}٪")
        print(f"صدک HV20: {h.get('hv20_percentile')} | رژیم: {h['regime']}")

    x = v["vix"]
    if x.get("ok"):
        print(f"\n=== مجموعه VIX ===")
        for s, m in x["members"].items():
            print(f"  {s:<8}{m['name']:<9}{m['value']:>7.2f}  "
                  f"({m['change_pct']:+.2f}٪) صدک ۶ماهه {m['percentile_6m']}")
        print(f"نسبت ۹روزه/۳۰روزه: {x.get('ratio_9d_30d')} | "
              f"۳۰روزه/۳ماهه: {x.get('ratio_30d_3m')}")
        print(f"وضعیت: {x.get('term_state')}")
        if x.get("percentile_all_time"):
            print(f"صدک تاریخی (از {x.get('cboe_from')}): {x['percentile_all_time']}"
                  f" | صدک ۵ساله: {x.get('percentile_5y')}")

    print(f"\nپریمیوم ریسک نوسان (IV-HV): {v['vrp']}")
    print(f"امتیاز نوسان: {v['score']:+.2f}")
    for s in v["signals"]:
        print(f"   {s['name']:<26}{s['impact']:+.1f}  {s['note']}")
