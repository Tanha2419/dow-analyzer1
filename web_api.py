#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
web_api.py — لایه تبدیل نتایج موتور پول هوشمند به JSON برای وب سایت
"""

from __future__ import annotations

import time
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import assets as A
import yfinance as yf

import smart_money as smc

warnings.filterwarnings("ignore")

SYMBOL = "DIA"

# ضریب تبدیل قیمت ETF به مقیاس شاخص داوجونز (US30 / DJ30 / ^DJI)
# DIA تقریبا یک صدم شاخص داوجونز معامله می شود.
US30_SCALE = 100.0

INTERVAL_PERIODS = {
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "1h": "730d",
    "1d": "5y",
}

# کش ساده در حافظه برای جلوگیری از فراخوانی مکرر یاهو
_CACHE: Dict[str, dict] = {}
_TTL = {"5m": 120, "15m": 180, "30m": 300, "1h": 600, "1d": 900}


def _clean(v):
    """تبدیل مقادیر numpy/pandas به انواع قابل سریال سازی JSON."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if not np.isfinite(f) else round(f, 6)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, float):
        return None if not np.isfinite(v) else round(v, 6)
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    if isinstance(v, np.ndarray):
        return [_clean(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


# کلیدهایی که مقدارشان «قیمت» است و باید در ضریب مقیاس ضرب شوند
_PRICE_KEYS = {
    "last", "high", "low", "open", "o", "h", "l", "c",
    "sma20", "sma50", "ema200", "atr", "vwap",
    "poc", "vah", "val", "near_hvn", "near_lvn",
    "top", "bottom", "level", "extreme", "price", "change",
    "entry", "stop", "tp1", "tp2", "tp3", "risk",
}
# کلیدهایی که لیستی از قیمت خام هستند
_PRICE_LISTS = {"hvn", "lvn"}
# کلیدهایی که هرگز نباید مقیاس بخورند (درصد، نسبت، امتیاز، حجم)
_SKIP_KEYS = {
    "dist", "dist_pct", "change_pct", "vwap_dev", "poc_dist", "risk_pct",
    "rr1", "rr2", "rr3", "atr_x", "disp_atr", "depth", "wick", "vol_z",
    "score", "confidence", "vol", "volume", "vol_ma20", "notional", "ret",
}


def scale_prices(obj, k: float = US30_SCALE):
    """همه مقادیر قیمتی را در ضریب مقیاس ضرب می کند (بازگشتی)."""
    if isinstance(obj, dict):
        out = {}
        for key, val in obj.items():
            if key in _SKIP_KEYS:
                out[key] = val
            elif key in _PRICE_LISTS and isinstance(val, list):
                out[key] = [None if v is None else round(v * k, 4) for v in val]
            elif key in _PRICE_KEYS and isinstance(val, (int, float)) \
                    and not isinstance(val, bool):
                out[key] = round(val * k, 4)
            elif key == "bins" and isinstance(val, list):
                out[key] = [dict(price=None if b.get("price") is None
                                 else round(b["price"] * k, 4), vol=b.get("vol"))
                            for b in val]
            elif key == "periodic" and isinstance(val, dict):
                out[key] = {pk: dict(high=round(pv["high"] * k, 4),
                                     low=round(pv["low"] * k, 4))
                            for pk, pv in val.items()}
            else:
                out[key] = scale_prices(val, k)
        return out
    if isinstance(obj, list):
        return [scale_prices(x, k) for x in obj]
    return obj


def fetch(interval: str = "1d", period: Optional[str] = None,
          symbol: Optional[str] = None) -> pd.DataFrame:
    period = period or INTERVAL_PERIODS.get(interval, "1y")
    df = yf.Ticker(symbol or SYMBOL).history(interval=interval, period=period)
    if df.empty:
        raise RuntimeError("داده ای از Yahoo Finance دریافت نشد")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def _candles(df: pd.DataFrame, bars: int) -> List[dict]:
    d = df.tail(bars)
    return [
        dict(t=pd.Timestamp(i).isoformat(),
             o=_clean(r.Open), h=_clean(r.High),
             l=_clean(r.Low), c=_clean(r.Close), v=_clean(r.Volume))
        for i, r in d.iterrows()
    ]


def build_payload(interval: str = "1d", bars: int = 160,
                  with_coalition: bool = True, scale: bool = True,
                  asset: Optional[str] = None) -> dict:
    t0 = time.time()
    prof = A.profile(asset)
    sym = prof["candle_symbol"]
    df = fetch(interval, symbol=sym)

    # ضریب نمایش: داوجونز ×۱۰۰ ، طلا تبدیل فیوچرز → اسپات واقعی
    bas: dict = {}
    if prof["basis_mode"] == "scale":
        kfac = float(prof["display_scale"])
    else:
        bas = A.basis(prof["key"], futures_price=float(df["Close"].iloc[-1]))
        kfac = float(bas.get("factor", 1.0)) if bas.get("ok") else 1.0

    htf_df = None
    intraday = None
    try:
        if interval == "1d":
            htf_df = fetch("1wk", "5y")
            intraday = yf.Ticker(SYMBOL).history(interval="30m", period="60d")
        else:
            htf_df = fetch("1d", "2y")
            intraday = df
    except Exception:
        pass

    res = smc.run_full_smc(df, interval, htf_df=htf_df, intraday=intraday,
                           with_coalition=with_coalition)

    n = len(df)
    off = n - min(bars, n)
    price = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if n > 1 else price
    chg = price - prev
    sig = res["signal"]
    vp = res["vprof"]

    # ---------- اندیکاتورهای کلاسیک برای نمایش ----------
    c = df["Close"]
    rsi = smc._rsi(c.to_numpy(float))
    atr = smc.atr_array(df)
    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()

    def last(x):
        try:
            v = float(x.iloc[-1]) if hasattr(x, "iloc") else float(x[-1])
            return _clean(v)
        except Exception:
            return None

    # ---------- FVG و Order Block قابل نمایش ----------
    fvgs = [g for g in res["fvgs"] if not g["filled"] and g["i"] >= off]
    fvgs = sorted(fvgs, key=lambda g: abs((g["top"] + g["bottom"]) / 2 - price))[:14]
    obs = [b for b in res["obs"] if not b["mitigated"] and b["i"] >= off][:10]
    sweeps = [s for s in res["sweeps"] if s["i"] >= off]
    blocks = [b for b in res["blocks"]["blocks"] if b["i"] >= off][:14]
    events = [e for e in res["struct"]["events"] if e["i"] >= off]

    def idx_of(i):
        return int(i - off)

    payload = dict(
        meta=dict(
            symbol=sym, interval=interval, bars=min(bars, n),
            asset=prof["key"], asset_name=prof["name"],
            asset_full=prof["name_full"], emoji=prof["emoji"],
            unit=prof["unit"], decimals=prof["decimals"],
            spot=(bas.get("spot") if bas.get("ok") else None),
            basis=(bas or None),
            scaled=bool(scale), scale_factor=kfac if scale else 1.0,
            display_symbol=(f"{prof['key']} / {prof['name']}" if scale else sym),
            total_candles=n,
            generated=pd.Timestamp.now().isoformat(),
            elapsed=round(time.time() - t0, 2),
            first=pd.Timestamp(df.index[off]).isoformat(),
            last=pd.Timestamp(df.index[-1]).isoformat(),
        ),
        price=dict(
            last=_clean(price), change=_clean(chg),
            change_pct=_clean(chg / prev * 100 if prev else 0),
            high=_clean(df["High"].iloc[-1]), low=_clean(df["Low"].iloc[-1]),
            open=_clean(df["Open"].iloc[-1]),
            volume=_clean(df["Volume"].iloc[-1]),
            vol_ma20=_clean(df["Volume"].rolling(20).mean().iloc[-1]),
        ),
        indicators=dict(
            rsi=last(rsi), atr=last(atr), sma20=last(sma20), sma50=last(sma50),
            ema200=last(ema200),
            trend="صعودی" if last(sma20) and last(sma50) and last(sma20) > last(sma50)
                  else "نزولی",
            vwap=_clean(res["flow"]["vwap"]),
            vwap_dev=_clean(res["flow"]["vwap_dev_pct"]),
        ),
        candles=_candles(df, bars),
        signal=dict(
            direction=int(sig["direction"]), label=sig["label"],
            grade=sig["grade"], score=_clean(sig["score"]),
            raw_score=_clean(sig["raw_score"]),
            confidence=_clean(sig["confidence"]),
            fake_penalty=_clean(sig["fake_penalty"]),
            hft_penalty=_clean(sig["hft_penalty"]),
            parts=[dict(name=p[0], weight=_clean(p[1]), detail=p[2])
                   for p in sig["parts"]],
            plan=_clean(sig["plan"]),
        ),
        # ماژول 1
        coalition=_clean(res["coalition"]),
        hft=_clean(res["hft"]),
        # ماژول 2
        structure=dict(
            bias=int(res["struct"]["bias"]),
            htf_bias=res.get("htf_bias"),
            events=[dict(x=idx_of(e["i"]), ref_x=idx_of(max(e["ref_i"], off)),
                         time=pd.Timestamp(e["time"]).isoformat(),
                         type=e["type"], dir=e["dir"],
                         level=_clean(e["level"]))
                    for e in events],
        ),
        fvgs=[dict(x=idx_of(g["i"]), time=pd.Timestamp(g["time"]).isoformat(),
                   dir=g["dir"], top=_clean(g["top"]), bottom=_clean(g["bottom"]),
                   atr_x=_clean(g["atr_x"]), mitigated=bool(g["mitigated"]),
                   dist=_clean(((g["top"] + g["bottom"]) / 2 - price) / price * 100))
              for g in fvgs],
        order_blocks=[dict(x=idx_of(b["i"]), time=pd.Timestamp(b["time"]).isoformat(),
                           dir=b["dir"], top=_clean(b["top"]),
                           bottom=_clean(b["bottom"]),
                           disp_atr=_clean(b["disp_atr"]), vol_z=_clean(b["vol_z"]),
                           dist=_clean(((b["top"] + b["bottom"]) / 2 - price) / price * 100))
                      for b in obs],
        # ماژول 3
        volume_profile=dict(
            poc=_clean(vp.get("poc")), vah=_clean(vp.get("vah")),
            val=_clean(vp.get("val")),
            in_value=bool(vp.get("in_value", False)),
            poc_dist=_clean(vp.get("poc_dist_pct")),
            near_hvn=_clean(vp.get("near_hvn")), near_lvn=_clean(vp.get("near_lvn")),
            hvn=_clean(vp.get("hvn", [])[:8]), lvn=_clean(vp.get("lvn", [])[:8]),
            bins=[dict(price=_clean(p), vol=_clean(v))
                  for p, v in zip(vp.get("centers", []), vp.get("profile", []))]
                 if vp else [],
        ) if vp else {},
        liquidity=dict(
            bsl=[_clean(x) for x in res["liq"].get("fresh_bsl", [])[:6]],
            ssl=[_clean(x) for x in res["liq"].get("fresh_ssl", [])[:6]],
            periodic=_clean(res["liq"].get("periodic", {})),
        ),
        # ماژول 4
        flow=dict(
            cd_slope=_clean(res["flow"]["cd_slope"]),
            obv_slope=_clean(res["flow"]["obv_slope"]),
            ad_slope=_clean(res["flow"]["ad_slope"]),
            cmf=_clean(res["flow"]["cmf_last"]),
            mfi=_clean(res["flow"]["mfi_last"]),
            divergence=res["flow"]["divergence"],
            smi_trend=_clean(res["flow"].get("smi_trend")),
            cum_delta=[_clean(x) for x in res["flow"]["cum_delta"][off:]],
            cmf_series=[_clean(x) for x in res["flow"]["cmf"][off:]],
        ),
        # ماژول 5
        fake=_clean(res["fake"]),
        # ماژول 6
        blocks=dict(
            n=int(res["blocks"]["n_blocks"]),
            imbalance=_clean(res["blocks"]["imbalance"]),
            buy_notional=_clean(res["blocks"]["buy_notional"]),
            sell_notional=_clean(res["blocks"]["sell_notional"]),
            items=[dict(x=idx_of(b["i"]),
                        time=pd.Timestamp(b["time"]).isoformat(),
                        side=b["side"], vol=_clean(b["vol"]),
                        vol_z=_clean(b["vol_z"]), notional=_clean(b["notional"]),
                        price=_clean(b["price"]), ret=_clean(b["ret_pct"]))
                   for b in blocks],
            absorption=[dict(time=pd.Timestamp(a["time"]).isoformat(),
                             price=_clean(a["price"]), side=a["side"])
                        for a in res["blocks"]["absorption"][:6]],
            icebergs=[dict(time=pd.Timestamp(i["time"]).isoformat(),
                           price=_clean(i["price"]), side=i["side"])
                      for i in res["blocks"]["icebergs"][:6]],
        ),
        # ماژول 7
        sweeps=[dict(x=idx_of(s["i"]), time=pd.Timestamp(s["time"]).isoformat(),
                     type=s["type"], dir=s["dir"], level=_clean(s["level"]),
                     extreme=_clean(s["extreme"]), depth=_clean(s["depth_atr"]),
                     wick=_clean(s["wick"]), vol_z=_clean(s["vol_z"]))
                for s in sweeps],
    )
    if scale:
        payload = scale_prices(payload, kfac)
    return payload


def cached_payload(interval: str = "1d", bars: int = 160,
                   with_coalition: bool = True, force: bool = False,
                   scale: bool = True, asset: Optional[str] = None) -> dict:
    akey = A.resolve(asset)
    key = f"{akey}:{interval}:{bars}:{int(with_coalition)}:{int(scale)}"
    now = time.time()
    hit = _CACHE.get(key)
    ttl = _TTL.get(interval, 600)
    if hit and not force and (now - hit["ts"]) < ttl:
        out = dict(hit["data"])
        out["meta"] = dict(out["meta"])
        out["meta"]["cached"] = True
        out["meta"]["age"] = int(now - hit["ts"])
        return out
    data = build_payload(interval, bars, with_coalition, scale, asset=akey)
    data["meta"]["cached"] = False
    data["meta"]["age"] = 0
    _CACHE[key] = dict(ts=now, data=data)
    return data


def backtest_payload(interval: str = "1d", cash: float = 100_000,
                     long_only: bool = False, regime: bool = True,
                     scale: bool = True, asset: Optional[str] = None) -> dict:
    """اجرای بکتست و بازگرداندن نتایج به صورت JSON.

    توجه: محاسبات بکتست همیشه روی قیمت خام DIA انجام می شود تا اندازه
    موقعیت و گرد کردن تعداد سهم دقیق بماند. فقط قیمت های نمایشی
    (ورود/خروج معاملات) در ضریب مقیاس ضرب می شوند.
    """
    # این دو فقط برای بکتست لازم اند و در استقرار وب سبک نصب
    # نمی شوند (کتابخانه backtesting به numba وابسته است و numba
    # روی پایتون جدید ساخته نمی شود). بکتست کار محلی است.
    try:
        from backtesting import Backtest
        from smc_backtest import SmartMoneyStrategy, SmartMoneyLongOnly
    except ImportError:
        return dict(
            ok=False,
            error="بکتست در نسخه وب فعال نیست — این قابلیت را "
                  "روی کامپیوتر خودتان اجرا کنید",
            reason="backtesting_not_installed")

    _prof = A.profile(asset)
    df = fetch(interval, symbol=_prof["candle_symbol"])
    sig = smc.causal_smc_signals(df, regime_filter=regime)
    cols = ["Open", "High", "Low", "Close", "Volume", "SMC_Long", "SMC_Short",
            "SMC_SL_Long", "SMC_SL_Short", "SMC_ATR"]
    b = sig[cols].copy()
    b[["SMC_SL_Long", "SMC_SL_Short"]] = b[["SMC_SL_Long", "SMC_SL_Short"]].fillna(0.0)
    b = b.dropna()
    if b.empty or (b["SMC_Long"].sum() + b["SMC_Short"].sum()) == 0:
        return dict(ok=False, error="سیگنالی برای بکتست یافت نشد")

    strat = SmartMoneyLongOnly if long_only else SmartMoneyStrategy
    bt = Backtest(b, strat, cash=cash, commission=0.001, margin=1.0,
                  trade_on_close=True, exclusive_orders=True, finalize_trades=True)
    stats = bt.run()
    eq = stats["_equity_curve"]
    tr = stats["_trades"]

    step = max(1, len(eq) // 400)
    eq_s = eq.iloc[::step]
    bh = cash * (b["Close"] / b["Close"].iloc[0])
    bh_s = bh.iloc[::step]

    trades = []
    if tr is not None and len(tr):
        for r in tr.itertuples():
            if _prof["basis_mode"] == "scale":
                k = _prof["display_scale"] if scale else 1.0
            else:
                _b = A.basis(_prof["key"])
                k = (float(_b.get("factor", 1.0))
                     if (scale and _b.get("ok")) else 1.0)
            trades.append(dict(
                entry_time=pd.Timestamp(r.EntryTime).isoformat(),
                exit_time=pd.Timestamp(r.ExitTime).isoformat(),
                side="long" if r.Size > 0 else "short",
                entry=_clean(r.EntryPrice * k), exit=_clean(r.ExitPrice * k),
                pnl=_clean(r.PnL), ret=_clean(r.ReturnPct * 100),
                bars=int(r.ExitBar - r.EntryBar)))

    sides = {}
    if trades:
        t = pd.DataFrame(trades)
        for side, g in t.groupby("side"):
            gp = g.loc[g["pnl"] > 0, "pnl"].sum()
            gl = abs(g.loc[g["pnl"] < 0, "pnl"].sum())
            sides[side] = dict(n=len(g),
                               win_rate=_clean((g["pnl"] > 0).mean() * 100),
                               pnl=_clean(g["pnl"].sum()),
                               avg_ret=_clean(g["ret"].mean()),
                               pf=_clean(gp / gl) if gl else None)

    def g(k):
        return _clean(stats.get(k))

    return dict(
        ok=True,
        config=dict(interval=interval, cash=cash, long_only=long_only,
                    regime=regime, scaled=bool(scale),
                    n_long=int(b["SMC_Long"].sum()),
                    n_short=int(b["SMC_Short"].sum()),
                    candles=len(b)),
        stats=dict(
            start=str(eq.index[0].date()), end=str(eq.index[-1].date()),
            equity_final=g("Equity Final [$]"), ret=g("Return [%]"),
            bh_ret=g("Buy & Hold Return [%]"), cagr=g("Return (Ann.) [%]"),
            vol=g("Volatility (Ann.) [%]"), max_dd=g("Max. Drawdown [%]"),
            avg_dd=g("Avg. Drawdown [%]"), n_trades=g("# Trades"),
            win_rate=g("Win Rate [%]"), avg_trade=g("Avg. Trade [%]"),
            best=g("Best Trade [%]"), worst=g("Worst Trade [%]"),
            pf=g("Profit Factor"), sharpe=g("Sharpe Ratio"),
            sortino=g("Sortino Ratio"), calmar=g("Calmar Ratio"),
            sqn=g("SQN"), exposure=g("Exposure Time [%]"),
        ),
        sides=sides,
        equity=[dict(t=pd.Timestamp(i).isoformat(), e=_clean(v))
                for i, v in eq_s["Equity"].items()],
        buyhold=[dict(t=pd.Timestamp(i).isoformat(), e=_clean(v))
                 for i, v in bh_s.items()],
        drawdown=[dict(t=pd.Timestamp(i).isoformat(),
                       d=_clean((v / eq_s["Equity"].cummax().loc[i] - 1) * 100))
                  for i, v in eq_s["Equity"].items()],
        trades=trades[-40:],
    )
