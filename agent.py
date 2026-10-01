#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent.py — ایجنت یکپارچه معاملاتی داوجونز

تجمیع همه لایه ها:
  بخش ۱ : حواس (MTF، بین بازاری، تقویم، احساسات)      -> market_context
  بخش ۲ : مغز (SMC، رژیم، Killzone، واگرایی)          -> smart_money + regime_ai
  بخش ۳ : سپر (سایزینگ پویا، مدیریت پوزیشن، فیلتر خبر)
  بخش ۴ : هوش مصنوعی (Isolation Forest، GBM)          -> regime_ai
  بخش ۵ : چرخه تکامل (ژورنال، تحلیل عملکرد)           -> journal
"""

from __future__ import annotations

import datetime as _dt
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

import institutional as inst
import macro_data as macro
import market_context as mc
import orderflow as ofl
import regime_ai as ra
import smart_money as smc
import volatility as volx
import assets as A
import sentiment_ext as sx
import econ_actual as ea
import journal as jr

warnings.filterwarnings("ignore")

SYMBOL = "DIA"
US30_SCALE = 100.0
JOURNAL = Path("output/journal.jsonl")


# ================================================================ داده
def load(interval: str = "1h", period: Optional[str] = None,
         symbol: Optional[str] = None) -> pd.DataFrame:
    per = period or {"5m": "60d", "15m": "60d", "30m": "60d",
                     "1h": "730d", "1d": "5y"}.get(interval, "1y")
    df = yf.Ticker(symbol or SYMBOL).history(interval=interval, period=per)
    if df.empty:
        raise RuntimeError("داده دریافت نشد")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # یاهو برای روز جاری کندل ناقص می دهد (حجم دارد، قیمت ندارد).
    # همان یک ردیف محاسبات پایین دست را با NaN می شکند.
    _o = [c for c in ("Open", "High", "Low", "Close") if c in df.columns]
    if _o:
        _n = len(df)
        df = df.dropna(subset=_o, how="any")
        if len(df) < _n:
            print(f"[ایجنت] {_n - len(df)} کندل ناقص حذف شد ({interval})")
    if df.empty:
        raise RuntimeError("پس از حذف کندل های ناقص داده ای نماند")
    return df


# ================================================================ بخش ۳: سپر دفاعی
def position_sizing(equity: float, entry: float, stop: float,
                    atr: float, regime: Dict, session_w: float,
                    confidence: float, base_risk: float = 0.01) -> Dict:
    """
    سایزینگ پویا: ریسک پایه با ضرایب نوسان، رژیم، جلسه و اطمینان تعدیل می شود.
    """
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0 or equity <= 0:
        return dict(units=0, risk_amount=0.0, note="فاصله حد ضرر نامعتبر")

    atr_pct = atr / entry * 100 if entry else 1.0
    # ضریب نوسان: بازار پرنوسان -> حجم کمتر
    k_vol = float(np.clip(1.0 / max(atr_pct / 0.9, 0.35), 0.35, 1.6))
    # ضریب رژیم
    k_reg = {"trending": 1.15, "ranging": 0.75,
             "volatile": 0.5, "transition": 0.65}.get(regime.get("regime"), 0.8)
    # ضریب جلسه معاملاتی
    k_ses = float(np.clip(session_w, 0.2, 1.0))
    # ضریب اطمینان سیگنال
    k_conf = float(np.clip(confidence / 70.0, 0.4, 1.3))

    eff = base_risk * k_vol * k_reg * k_ses * k_conf
    eff = float(np.clip(eff, 0.0015, 0.02))

    risk_amt = equity * eff
    units = int(risk_amt / risk_per_unit)
    max_units = int(equity * 0.95 / entry)
    units = max(0, min(units, max_units))

    return dict(
        units=units, risk_pct=round(eff * 100, 3),
        risk_amount=round(units * risk_per_unit, 2),
        notional=round(units * entry, 2),
        risk_per_unit=round(risk_per_unit, 4),
        factors=dict(volatility=round(k_vol, 3), regime=round(k_reg, 3),
                     session=round(k_ses, 3), confidence=round(k_conf, 3)),
        note=f"ریسک موثر {eff*100:.2f}% از سرمایه",
    )


def trade_management(entry: float, stop: float, direction: int,
                     atr: float, trailing: Dict) -> Dict:
    """طرح مدیریت پوزیشن: سربه سر، خروج پله ای، تریلینگ."""
    r = abs(entry - stop)
    if r <= 0:
        return {}
    sign = 1 if direction > 0 else -1
    st = trailing.get("supertrend", {}).get("value")
    ce = trailing.get("chandelier", {})
    trail = ce.get("long_stop") if direction > 0 else ce.get("short_stop")

    return dict(
        breakeven=dict(trigger=round(entry + sign * 1.0 * r, 2),
                       action="انتقال حد ضرر به نقطه ورود (1R)"),
        partial=dict(trigger=round(entry + sign * 2.0 * r, 2),
                     portion=0.5, action="بستن ۵۰٪ حجم در 2R"),
        runner=dict(action="مابقی با تریلینگ استاپ",
                    supertrend=None if st is None else round(st, 2),
                    chandelier=None if trail is None else round(float(trail), 2),
                    active=("Chandelier Exit" if trail is not None else "SuperTrend")),
        time_stop=dict(bars=40, action="خروج در صورت عدم پیشروی تا ۴۰ کندل"),
        r_value=round(r, 4),
    )


# ================================================================ تجمیع سیگنال
def _sig_from_parts(parts: List[tuple]) -> float:
    return float(sum(p[1] for p in parts))


# اعتبار سیگنال بر حسب ساعت — هم راستا با tradeplan.VALID_HOURS
_VALID_HOURS = {"5m": 2, "15m": 4, "30m": 6, "1h": 12, "1d": 72}
_TEHRAN = _dt.timezone(_dt.timedelta(hours=3, minutes=30))


# ── اصلاح ۲۰۲۶-۰۹-۲۵: مقیاس نشدن پنل چند تایم فریمی ──────────
# ctx خام از market_context می آید و مستقیم داخل خروجی گذاشته
# می شد، بدون اینکه ضریب نمایش k روی آن اعمال شود. نتیجه: پنل
# MTF برای داوجونز عدد دلاری دیا نشان می داد (۵۱۷ به جای ۵۱٬۷۴۷).
#
# فقط زیردرخت mtf اصلاح می شود. intermarket دست نخورده می ماند
# چون قیمت دارایی های دیگر (دلار، طلا، بازده) را دارد و ضرب آنها
# در ۱۰۰ عدد بی معنی می سازد.
_MTF_PRICE_KEYS = {"price", "atr", "level", "high", "low", "open", "close"}


def _scale_mtf(node, k: float):
    """ضریب نمایش را فقط روی کلیدهای قیمتی زیردرخت mtf اعمال می کند."""
    if k == 1.0:
        return node
    if isinstance(node, dict):
        out = {}
        for key, val in node.items():
            if key in _MTF_PRICE_KEYS and isinstance(val, (int, float)) \
                    and not isinstance(val, bool):
                out[key] = round(float(val) * k, 4)
            else:
                out[key] = _scale_mtf(val, k)
        return out
    if isinstance(node, list):
        return [_scale_mtf(x, k) for x in node]
    return node


def _scale_mtf_ctx(ctx: Dict, k: float) -> Dict:
    """کپی سطحی از ctx با زیردرخت mtf مقیاس خورده."""
    if not isinstance(ctx, dict) or k == 1.0 or "mtf" not in ctx:
        return ctx
    out = dict(ctx)
    out["mtf"] = _scale_mtf(ctx.get("mtf"), k)
    return out


def _signal_timing(interval: str, df) -> Dict:
    """زمان صدور و مهلت اعتبار سیگنال — به وقت تهران.

    قبلا فقط یک `generated` بدون منطقه زمانی وجود داشت و کاربر
    نمی دانست سیگنال کی صادر شده و تا کی معتبر است.
    """
    now = _dt.datetime.now(_TEHRAN)
    hours = _VALID_HOURS.get(interval, 12)
    until = now + _dt.timedelta(hours=hours)

    # زمان آخرین کندل بسته شده — مبنای واقعی تحلیل
    bar_time = None
    try:
        ts = df.index[-1]
        bar_time = str(ts)[:16]
    except Exception:
        pass

    return dict(
        issued_at=now.strftime("%Y-%m-%d %H:%M"),
        issued_time=now.strftime("%H:%M"),
        issued_tz="تهران",
        valid_hours=hours,
        valid_until=until.strftime("%Y-%m-%d %H:%M"),
        valid_until_time=until.strftime("%H:%M"),
        last_bar=bar_time,
        timing_note=(f"این سیگنال ساعت {now:%H:%M} به وقت تهران صادر شد "
                     f"و تا {until:%H:%M} روز {until:%Y-%m-%d} معتبر است "
                     f"({hours} ساعت). بعد از آن باید دوباره بررسی شود "
                     f"چون ساختار بازار عوض می شود."),
    )


def decide(interval: str = "1h", equity: float = 100_000,
           with_ml: bool = True, with_mtf: bool = True,
           with_coalition: bool = True, scale: bool = True,
           with_macro: bool = True, with_vol: bool = True,
           with_flow: bool = True, asset: Optional[str] = None) -> Dict:
    """اجرای کامل ایجنت و تولید تصمیم نهایی — برای هر دارایی."""
    prof = A.profile(asset)
    sym = prof["candle_symbol"]
    df = load(interval, symbol=sym)

    # ضریب نمایش: داوجونز ×۱۰۰ ، طلا تبدیل فیوچرز به اسپات واقعی
    bas: Dict = {}
    if not scale:
        k = 1.0
    elif prof["basis_mode"] == "scale":
        k = float(prof["display_scale"])
    else:
        bas = A.basis(prof["key"], futures_price=float(df["Close"].iloc[-1]))
        k = float(bas.get("factor", 1.0)) if bas.get("ok") else 1.0

    # ---------- لایه ها ----------
    asset_key = prof["key"]
    ctx = mc.build_context(sym, with_mtf=with_mtf, asset=prof["key"])
    intel = ra.run_intelligence(df, with_ml=with_ml)

    # --- سه منبع احساسات/پوزیشن بیرونی (COT، ترس و طمع، StockTwits) ---
    try:
        sent_ext = sx.build_sentiment_ext(prof["key"])
    except Exception as _e:
        sent_ext = dict(ok=False, error=str(_e), sources_ok=0)

    try:
        surp = ea.surprise_score()
    except Exception:
        surp = dict(ok=False, tone=0.0)

    try:
        crit = ea.critical_events(72)
    except Exception:
        crit = dict(ok=False, veto=False, events=[], imminent=[])

    mac = macro.build_macro(with_earnings=True) if with_macro else {}
    vol = volx.build_volatility(sym, asset=prof["key"]) if with_vol else {}
    flow = (ofl.build_orderflow(sym, interval, with_seasonality=True)
            if with_flow else {})

    htf = None
    try:
        htf = (load("1d", "2y", symbol=sym) if interval != "1d"
               else load("1wk", "5y", symbol=sym))
    except Exception:
        pass
    intraday = df if interval != "1d" else None
    if interval == "1d":
        try:
            intraday = yf.Ticker(sym).history(interval="30m", period="60d")
        except Exception:
            pass

    smc_res = smc.run_full_smc(df, interval, htf_df=htf, intraday=intraday,
                               with_coalition=with_coalition,
                               asset=prof["key"])

    price = float(df["Close"].iloc[-1])
    atr = float(smc.atr_array(df)[-1])
    base = smc_res["signal"]

    # ---------- پنج لایه نهادی + بردار ویژگی ----------
    daily_ref = htf if (htf is not None and interval != "1d") else df
    try:
        institutional = inst.build_institutional(
            price, atr, df, daily_ref, ctx, vol, flow, intel,
            liq=smc_res.get("liq"), vp=smc_res.get("vprof"),
            scale=k)
    except Exception as _e:
        institutional = dict(ok=False, error=str(_e),
                             veto=dict(long=False, short=False,
                                       reasons=[], warnings=[]))

    # ---------- امتیازدهی یکپارچه ----------
    parts: List[tuple] = []

    # پایه SMC (وزن اصلی)
    # ⚠ ممیزی ۲۰۲۶-۰۹-۱۹: این بزرگ ترین سهم امتیاز را دارد ولی با IC
    # سنجش پذیر نیست (خروجی مرکب چند زیرسیستم است، نه یک سری زمانی).
    # ضریب دست نخورده ماند چون کاهشش هم حدس می بود؛ به جایش دفترچه
    # journal.py اثر واقعی اش را ثبت می کند تا با داده تصمیم بگیریم.
    parts.append(("موتور پول هوشمند (SMC)", float(base["score"]) * 0.85,
                  f"{base['label']} | درجه {base['grade']}"))

    # چند تایم فریمی
    if with_mtf and ctx.get("mtf", {}).get("frames"):
        m = ctx["mtf"]
        w = float(np.clip(m["aggregate"] * 22, -22, 22))
        parts.append(("همراستایی چند تایم فریمی", w,
                      f"{m['verdict']} | هماهنگی {m['agreement']*100:.0f}%"))

    # بین بازاری
    im = ctx.get("intermarket", {})
    if im.get("ok"):
        w = float(np.clip(im["risk_score"] * 7, -18, 18))
        parts.append(("شرایط بین بازاری", w, im["verdict"]))
        for s in im.get("smt", [])[:2]:
            sw = 6 if s["type"].startswith("bullish") else -6
            parts.append(("واگرایی SMT", float(sw), s["text"][:70]))

    # رژیم بازار
    rg = intel["regime"]
    if rg["regime"] == "trending":
        w = 10.0 * rg["direction"]
        parts.append(("رژیم رونددار", w,
                      f"Hurst {rg['hurst']} | ADX {rg['adx']}"))
    elif rg["regime"] == "volatile":
        parts.append(("رژیم پرنوسان", 0.0,
                      "حجم کاهش می یابد، ورود محتاطانه"))
    elif rg["regime"] == "ranging":
        parts.append(("رژیم رنج", 0.0,
                      "اولویت با بازگشت به میانگین، نه شکست"))

    # واگرایی
    dv = intel["divergence"]
    if dv["items"]:
        # ممیزی ۲۰۲۶-۰۹-۱۹: RSI روی طلای ۱ساعته IC=+0.055 (ادامه‌دهنده)
        # ولی روی داوجونز ۱ساعته IC=-0.036 (بازگشتی). هر دو با بازه
        # اطمینان بوت‌استرپ معنادار. پس علامت باید وابسته به دارایی باشد.
        _msign = float(prof["rules"].get("momentum_sign", 1))
        w = float(np.clip(dv["bias"] * 4.5 * _msign, -12, 12))
        parts.append(("واگرایی مومنتوم", w,
                      f"{dv['n_hidden']} مخفی / {dv['n_regular']} معمولی"))

    # ناهنجاری نهادی
    an = intel.get("anomaly", {})
    if an.get("ok") and an.get("n_recent", 0) > 0:
        w = float(np.clip(an["bias"] * 4, -10, 10))
        parts.append(("ردپای نهادی (Isolation Forest)", w, an["note"]))

    # پیش بینی ML — فقط اگر لبه آماری معتبر داشته باشد
    ml = intel.get("ml", {})
    if ml.get("ok"):
        if ml.get("trustworthy"):
            w = float(np.clip((ml["prob_up"] - 0.5) * 36, -9, 9))
            parts.append(("پیش بینی مدل یادگیری ماشین", w,
                          f"احتمال صعود {ml['prob_up']*100:.0f}% | "
                          f"دقت OOS {ml['oos_accuracy']}"))
        else:
            parts.append(("پیش بینی مدل یادگیری ماشین", 0.0,
                          f"احتمال {ml['prob_up']*100:.0f}% ولی فاقد اعتبار "
                          f"آماری (OOS {ml.get('oos_accuracy')}) — وزن صفر"))

    # ---------- کلان اقتصادی ----------
    if mac.get("score", {}).get("ok"):
        ms = mac["score"]
        w = float(np.clip(ms["score"] * 6, -16, 16))
        parts.append(("شرایط کلان اقتصادی (FRED)", w, ms["label"]))
    cv = mac.get("curve", {})
    if cv.get("ok") and cv.get("state") == "معکوس":
        parts.append(("منحنی بازده معکوس", -7.0, cv["note"]))
    fw = mac.get("fed", {})
    if fw.get("ok") and fw.get("direction") is not None:
        if fw["direction"] < 0:
            parts.append(("انتظار کاهش نرخ بهره", 8.0,
                          f"{abs(fw.get('cuts_priced',0)):.2f} کاهش قیمت گذاری شده"))
        elif fw["direction"] > 0:
            parts.append(("انتظار افزایش نرخ بهره", -9.0,
                          f"{fw.get('diff_bp',0):+.0f} بیپ بالاتر از نرخ فعلی"))
    ec = mac.get("earnings", {})
    if ec.get("ok") and ec.get("heavy_days"):
        nd = ec["heavy_days"][0]
        parts.append(("تجمع گزارش درآمدی", -2.5,
                      f"{nd['count']} شرکت در {nd['date']}"))

    # ---------- نوسان و آپشن ----------
    if vol.get("signals"):
        for s in vol["signals"][:3]:
            parts.append((f"آپشن: {s['name']}", float(s["impact"]) * 5.5,
                          f"{s['note']} ({s['value']})"))

    # ---------- جریان سفارش ----------
    fd = flow.get("delta", {})
    if fd.get("ok"):
        w = float(np.clip(fd["pressure"] * 4.5, -11, 11))
        parts.append(("جریان سفارش CVD (تخمینی)", w, fd["pressure_label"]))
        if fd.get("divergence"):
            dv2 = fd["divergence"]
            parts.append(("واگرایی CVD",
                          6.0 if dv2["type"] == "bullish" else -6.0,
                          dv2["note"]))
    bl = flow.get("blocks", {})
    if bl.get("ok") and bl.get("recent", 0) > 0:
        # ممیزی: حجم غیرعادی IC=-0.004 (طلا) / +0.013 (داو) — نویز.
        # وزن از ۳.۵ به ۰.۸ کاهش یافت؛ فقط به عنوان زمینه.
        parts.append(("معاملات بلوکی", float(np.clip(bl["bias"] * 0.8, -2, 2)),
                      f"{bl['buy']} خرید در برابر {bl['sell']} فروش"))
    cpz = flow.get("candles", {})
    if cpz.get("ok") and cpz.get("items"):
        # ممیزی: IC=+0.008 (طلا ۱س) / -0.030 (داو ۱س) — علامت ناپایدار.
        parts.append(("الگوهای کندلی", float(np.clip(cpz["bias"] * 0.6, -2, 2)),
                      cpz["items"][0]["name"] + f" ({cpz['n']} الگو)"))
    chz = flow.get("chart", {})
    if chz.get("ok") and chz.get("items"):
        parts.append(("الگوهای نموداری", float(np.clip(chz["bias"] * 3, -8, 8)),
                      chz["items"][0]["name"]))
    sz2 = flow.get("seasonality", {})
    if sz2.get("ok") and sz2.get("current_weekday"):
        cwd = sz2["current_weekday"]
        if cwd.get("significant"):
            # ممیزی: IC=-0.018 (طلا) / -0.010 (داو) — ضعیف ترین سیگنال
            # با بزرگ ترین ضریب. از ۲۲ به ۴ کاهش یافت.
            parts.append(("فصلی بودن روز هفته",
                          float(np.clip(cwd["mean"] * 4, -1.5, 1.5)),
                          f"{cwd['name']}: میانگین {cwd['mean']:+.3f}٪ "
                          f"(معنادار آماری)"))
        else:
            parts.append(("فصلی بودن روز هفته", 0.0,
                          f"{cwd['name']}: {cwd['mean']:+.3f}٪ — "
                          f"فاقد معناداری آماری، وزن صفر"))

    # احساسات خبری
    nw = ctx.get("news", {})
    if nw.get("ok") and nw.get("n", 0) >= 3:
        w = float(np.clip(nw["score"] * 8, -8, 8))
        parts.append(("احساسات خبری (NLP)", w,
                      f"{nw['label']} | {nw['n']} تیتر"))

    # VIX
    sen = ctx.get("sentiment", {})
    if sen.get("ok"):
        v = sen["vix"]
        _haven = bool(prof["rules"].get("vix_safe_haven"))
        if v["spike"]:
            # طلا پناهگاه امن است: جهش ترس به جای جریمه، امتیاز مثبت می گیرد
            parts.append(("جهش VIX", (+8.0 if _haven else -12.0),
                          f"VIX {v['last']:.1f} ({v['chg_pct']:+.1f}%)"
                          + (" — تقاضای پناهگاه امن" if _haven else "")))
        elif v["last"] < 14:
            parts.append(("آرامش بیش از حد VIX",
                          (-2.0 if _haven else -3.0),
                          "ریسک غافلگیری بالا" if not _haven
                          else "تقاضای پناهگاه کم"))

    # --- COT: پوزیشن واقعی صندوق های بزرگ (فقط طلا) ---
    _ct = sent_ext.get("cot") or {}
    if _ct.get("ok"):
        _t = int(_ct.get("contrarian_tone") or 0)
        if _t:
            # ممیزی: ازدحام صدک۹۰ → بازده ۴هفته +۳.۰۰٪ در برابر +۱.۰۲٪
            # میانه (p=0.016). علامت در sentiment_ext اصلاح شد؛ وزن از
            # ۹ به ۴ کم شد چون افق COT هفتگی است نه ساعتی.
            parts.append(("پوزیشن صندوق ها (COT)", float(_t) * 4.0,
                          f"{_ct['state']} | خالص {_ct['mm_net']:+,} "
                          f"({_ct['mm_net_pct']}٪)"))

    # --- ترس و طمع CNN ---
    _fg = sent_ext.get("fear_greed") or {}
    if _fg.get("ok"):
        _t = int(_fg.get("gold_tone" if prof["key"] == "XAUUSD"
                         else "equity_tone") or 0)
        if _t:
            parts.append(("شاخص ترس و طمع (CNN)", float(_t) * 7.0,
                          f"{_fg['score']} — {_fg['state']}"))

    # --- StockTwits: احساسات خرد (معکوس) ---
    _st = sent_ext.get("stocktwits") or {}
    if _st.get("ok") and not _st.get("low_sample"):
        _t = int(_st.get("contrarian_tone") or 0)
        if _t:
            parts.append(("احساسات خرد (StockTwits)", float(_t) * 5.0,
                          f"{_st['state']} | {_st['bull_pct']}٪ صعودی "
                          f"از {_st['tagged']} پیام"))

    if surp.get("ok") and surp.get("n", 0) > 0 and abs(surp.get("tone", 0)) >= 0.34:
        _st = float(surp["tone"])
        # داده بهتر از انتظار ⟵ مثبت برای سهام، منفی برای طلا
        _sv = _st if asset != "XAUUSD" else -_st
        parts.append(("سورپرایز داده های اقتصادی", round(_sv * 6.0, 2),
                      surp.get("tone_fa", "")))

    if crit.get("veto") and crit.get("imminent"):
        _im = crit["imminent"][0]
        parts.append(("رویداد بسیار مهم نزدیک", 0.0,
                      "%s تا %s ساعت دیگر — احتیاط"
                      % (_im["name"], _im["hours_until"])))

    # --- بخش های کلان که در ممیزی معنادار بودند ---
    # بازده ۱۰ساله ← داوجونز IC=-0.048 [-0.084,-0.011] ✅
    # DXY ← داوجونز IC=-0.045 [-0.079,-0.010] ✅ / طلا IC=+0.038 [+0.001,+0.072] ✅
    try:
        _im = (ctx.get("intermarket") or {}).get("assets") or []
        for _a in _im:
            _sym = str(_a.get("symbol") or "")
            _chg = _a.get("chg_pct")
            if _chg is None:
                continue
            _chg = float(_chg)
            if _sym == "^TNX" and asset_key != "XAUUSD":
                # بازده بالاتر ← سهام پایین تر
                parts.append(("بازده اوراق ۱۰ساله", float(np.clip(-_chg * 0.9, -5, 5)),
                              "تغییر %.2f٪ — ممیزی: IC=-0.048 معنادار" % _chg))
            elif _sym in ("DX-Y.NYB", "DXY"):
                _dir = +1.0 if asset_key == "XAUUSD" else -1.0
                parts.append(("شاخص دلار (DXY)", float(np.clip(_chg * _dir * 0.8, -4, 4)),
                              "تغییر %.2f٪ — ممیزی: معنادار در هر دو دارایی" % _chg))
    except Exception:
        pass

    raw = _sig_from_parts(parts)

    # ---------- ضرایب تعدیل ----------
    fake_k = 1 - (smc_res["fake"]["score"] / 100) * 0.5
    hft_k = 1 - max(0.0, (smc_res["hft"]["hft_index"] - 55) / 100) * 0.3
    ses_k = float(np.clip(0.55 + ctx["killzone"]["weight"] * 0.45, 0.55, 1.0))
    reg_k = {"trending": 1.0, "ranging": 0.8,
             "volatile": 0.65, "transition": 0.85}.get(rg["regime"], 0.85)

    score = raw * fake_k * hft_k * ses_k * reg_k
    conf = float(np.clip(abs(score) / 68 * 100, 0, 97))

    gate = ctx["gate"]
    iv_veto = institutional.get("veto", {}) if institutional.get("ok") else {}
    veto_long = bool(iv_veto.get("long"))
    veto_short = bool(iv_veto.get("short"))
    veto_reasons = list(iv_veto.get("reasons", []))

    if not gate["allowed"]:
        direction, label = 0, "معامله ممنوع — " + gate["blocks"][0]
    elif veto_long and score > 0:
        direction, label = 0, "خرید وتو شد — " + (veto_reasons[0] if veto_reasons
                                                  else "شرایط بین بازاری")
    elif veto_short and score < 0:
        direction, label = 0, "فروش وتو شد — " + (veto_reasons[0] if veto_reasons
                                                  else "شرایط بین بازاری")
    elif score >= 26:
        direction, label = 1, "خرید (LONG)"
    elif score <= -26:
        direction, label = -1, "فروش (SHORT)"
    elif score >= 13:
        direction, label = 1, "خرید ضعیف — منتظر تایید"
    elif score <= -13:
        direction, label = -1, "فروش ضعیف — منتظر تایید"
    else:
        direction, label = 0, "خنثی — بدون موقعیت"

    grade = ("A+" if conf >= 78 else "A" if conf >= 64 else
             "B" if conf >= 50 else "C" if conf >= 34 else "D")

    # ---------- طرح معامله ----------
    plan: Dict = {}
    # اگر جهت خنثی باشد طرحی ساخته نمی شود. قبلا داشبورد در این
    # حالت چیزی نمی گفت و کاربر فکر می کرد طرح قبلی هنوز معتبر است.
    # حالا دلیلش صریح برگردانده می شود.
    if direction == 0:
        plan = dict(
            empty=True,
            reason_fa=label,
            note_fa=("در این لحظه طرحی پیشنهاد نمی شود. هر عددی که "
                     "از قبل روی صفحه مانده مربوط به محاسبه قبلی است "
                     "و دیگر معتبر نیست."),
        )
    elif not base.get("plan"):
        # جهت مشخص است ولی موتور ساختار بازار هیچ سطح ورودی نداد.
        # قبلا در این حالت plan خالی {} می ماند و کارت بی صدا خالی
        # می شد؛ کاربر فکر می کرد عدد قبلی هنوز معتبر است.
        plan = dict(
            empty=True,
            reason_fa=f"{label} — ولی سطح ورود پیدا نشد",
            note_fa=("جهت مشخص است، اما موتور ساختار بازار در این "
                     "تایم فریم سطح ورود معتبری نیافت، پس عددی ساخته "
                     "نمی شود. هر عددی که از قبل روی صفحه مانده مربوط "
                     "به محاسبه قبلی است و دیگر معتبر نیست."),
        )
    if direction != 0 and base.get("plan"):
        bp = base["plan"]
        entry = float(bp["entry"])
        stop = float(bp["stop"])
        smc_entry, smc_stop = entry, stop

        # اگر جهت ایجنت با جهت SMC مخالف است، از ساختار جاری بساز
        plan_source = "smc"
        overridden = (direction > 0) != (entry > stop)
        if overridden:
            plan_source = "agent_override"
            if direction > 0:
                entry, stop = price, price - 1.5 * atr
            else:
                entry, stop = price, price + 1.5 * atr
        sz = position_sizing(equity, entry, stop, atr, rg,
                             ctx["killzone"]["weight"], conf)
        tm = trade_management(entry, stop, direction, atr, intel["trailing"])
        r = abs(entry - stop)
        plan = dict(
            entry=entry * k, stop=stop * k,
            tp1=(entry + (1 if direction > 0 else -1) * 2.0 * r) * k,
            tp2=float(bp.get("tp2", entry + (1 if direction > 0 else -1) * 3.5 * r)) * k,
            tp3=float(bp.get("tp3", entry + (1 if direction > 0 else -1) * 5.0 * r)) * k,
            risk_points=r * k,
            rr1=2.0,
            rr2=abs(float(bp.get("tp2", 0)) - entry) / r if bp.get("tp2") else 3.5,
            sizing=sz, management={
                kk: (dict(vv, **{sk: (sv * k if isinstance(sv, (int, float))
                                      and sk in ("trigger", "supertrend", "chandelier")
                                      else sv) for sk, sv in vv.items()})
                     if isinstance(vv, dict) else vv)
                for kk, vv in tm.items()},
        )

        # --- چرا این طرح با طرح پول هوشمند فرق دارد؟ ---
        # دو موتور مستقل اند: SMC از ساختار بازار (سوییپ/بلوک) ورود
        # می سازد، ایجنت همه لایه ها را جمع می زند. وقتی جهتشان
        # مخالف شود، ایجنت طرح خودش را می سازد. تا امروز داشبورد
        # نمی گفت کدام را دنبال کنید.
        # ── نوع ورود و وضعیت زنده (اصلاح ۲۰۲۶-۰۹-۳۰) ────────────
        #
        # کاربر گزارش داد: «ورود ۵۱٬۴۵۷ زده در حالی که قیمت ۵۱٬۳۸۳
        # بوده» و «معلوم نیست الان وارد شوم یا منتظر بمانم».
        #
        # ریشه: وقتی طرح از موتور SMC می آید، entry یک *سطح
        # ساختاری* است (بلوک سفارش یا ناحیه سوییپ) نه قیمت بازار.
        # برای شورت همیشه بالای قیمت فعلی است و برای لانگ پایین آن،
        # چون سفارش لیمیت است. ولی داشبورد این را هیچ جا نمی گفت و
        # کاربر فکر می کرد سیگنال بازار است.
        _px = price * k
        _entry = plan["entry"]
        _gap = _entry - _px
        # فاصله بر حسب ATR تا برای طلا و داوجونز هر دو معنی بدهد
        _atr_disp = max(atr * k, 1e-9)
        _gap_atr = _gap / _atr_disp

        if plan_source == "agent_override":
            # ایجنت طرح خودش را ساخته و ورود را روی قیمت جاری گذاشته
            plan["entry_type"] = "market"
            plan["entry_type_fa"] = "ورود بازار — همین قیمت"
        else:
            plan["entry_type"] = "limit"
            plan["entry_type_fa"] = (
                "ورود در بازگشت قیمت به این سطح (سفارش لیمیت، نه بازار)")

        plan["current_price"] = round(_px, 4)
        plan["entry_gap"] = round(_gap, 4)
        plan["entry_gap_atr"] = round(_gap_atr, 3)
        plan["entry_above"] = bool(_gap > 0)

        # وضعیت: آیا باید منتظر ماند؟
        if plan["entry_type"] == "market":
            plan["entry_status"] = "now"
            plan["entry_status_fa"] = "✅ همین حالا — ورود روی قیمت جاری است"
        elif abs(_gap_atr) <= 0.10:
            plan["entry_status"] = "reached"
            plan["entry_status_fa"] = "✅ قیمت به ناحیه ورود رسیده است"
        elif abs(_gap_atr) <= 1.0:
            _dir_fa = "بالا" if _gap > 0 else "پایین"
            plan["entry_status"] = "waiting"
            plan["entry_status_fa"] = (
                "⏳ منتظر — قیمت باید %s برود تا به ورود برسد (%s %s)"
                % (_dir_fa, format(abs(_gap), ",.%df" % prof["decimals"]),
                   prof["unit"]))
        else:
            plan["entry_status"] = "far"
            plan["entry_status_fa"] = (
                "⚠️ خیلی دور — ورود %s برابر ATR فاصله دارد. تا قیمت "
                "نرسد این طرح اجرا نمی شود و ممکن است تا آن موقع "
                "باطل شده باشد." % format(abs(_gap_atr), ".1f"))

        # اعتبار زمانی: طرح تا کی معنی دارد
        plan["valid_note"] = (
            "این طرح بر پایه ساختار همین لحظه ساخته شده. اگر قیمت به "
            "ناحیه ورود نرسد و ساختار عوض شود، طرح باطل است — منتظر "
            "محاسبه بعدی بمانید.")

        plan["source"] = plan_source
        plan["is_primary"] = True
        if overridden:
            plan["source_fa"] = "🎯 طرح ایجنت (اصلی)"
            plan["why_differs"] = (
                f"موتور پول هوشمند جهت مخالف می دید (ورود "
                f"{smc_entry * k:,.2f} / حد ضرر {smc_stop * k:,.2f}) ولی "
                f"جمع بندی همه لایه ها جهت دیگری داد. چون ایجنت همه "
                f"شواهد را با هم می سنجد، طرح او مبناست و ورود روی "
                f"قیمت فعلی با حد ضرر ۱٫۵ برابر ATR گذاشته شد.")
            plan["smc_alternative"] = dict(
                entry=round(smc_entry * k, 4), stop=round(smc_stop * k, 4),
                note="طرح موتور پول هوشمند — جهت مخالف، دنبال نکنید")
        else:
            plan["source_fa"] = "🎯 طرح ایجنت (هم راستا با پول هوشمند)"
            plan["why_differs"] = (
                "هر دو موتور هم جهت اند. ورود و حد ضرر از ساختار "
                "بازار گرفته شده و هدف ها با نسبت ۲R تنظیم شده اند؛ "
                "ممکن است اعداد هدف با بخش پول هوشمند کمی فرق کند "
                "چون آنجا نسبت های خودش را می گذارد.")
        plan["follow_this"] = ("این طرح را دنبال کنید — خروجی نهایی "
                               "ایجنت است، نه یکی از لایه ها.")

    out = dict(
        meta=dict(symbol=sym, interval=interval,
                  asset=prof["key"], asset_name=prof["name"],
                  asset_full=prof["name_full"], emoji=prof["emoji"],
                  unit=prof["unit"], decimals=prof["decimals"],
                  display=(prof["key"] if scale else sym),
                  scale=k, price=price * k, raw_price=price,
                  spot=(bas.get("spot") if bas.get("ok") else None),
                  basis=(bas or None),
                  atr=atr * k, equity=equity,
                  generated=pd.Timestamp.now().isoformat(),
                  **_signal_timing(interval, df)),
        decision=dict(direction=direction, label=label, grade=grade,
                      score=round(score, 2), raw_score=round(raw, 2),
                      confidence=round(conf, 1),
                      modifiers=dict(fake=round(fake_k, 3), hft=round(hft_k, 3),
                                     session=round(ses_k, 3), regime=round(reg_k, 3))),
        parts=[dict(name=p[0], weight=round(p[1], 2), detail=p[2]) for p in parts],
        plan=plan,
        context=_scale_mtf_ctx(ctx, k),
        intelligence=intel,
        smc=dict(signal=base, fake=smc_res["fake"], hft=smc_res["hft"],
                 coalition=smc_res["coalition"]),
        macro=mac, volatility=vol, orderflow=flow,
        sentiment_ext=sent_ext,
        econ_surprise=surp,
        critical_events=crit,
        institutional=institutional,
    )

    # ثبت در دفترچه — برای کالیبراسیون آینده (بی صدا شکست می خورد)
    #
    # price اینجا عمدا خام است (برای داوجونز یعنی DIA ≈ ۵۱۲) چون
    # journal.evaluate همین نماد خام را دوباره می گیرد و اگر دو طرف
    # هم مقیاس نباشند R بی معنا می شود. ضریب نمایش جدا فرستاده
    # می شود تا رکورد هم خام را داشته باشد هم عددی که کاربر دیده.
    # ⚠ از k استفاده نمی کنیم: وقتی scale=False باشد (ورک فلو)،
    # k برابر ۱ است ولی عدد روی صفحه همچنان ۱۰۰ برابر است.
    try:
        if abs(float(out["decision"]["score"])) >= 1.0:
            if prof["basis_mode"] == "scale":
                disp_k = float(prof["display_scale"])
            else:
                disp_k = (float(bas.get("factor", 1.0))
                          if bas.get("ok") else 1.0)
            jr.record(out, prof["key"], interval, price=price, scale=disp_k)
    except Exception:
        pass

    return out


# ================================================================ بخش ۵: ژورنال
def journal_add(entry: Dict) -> bool:
    """افزودن رکورد به ژورنال توزیع ها — بدون تکرار.

    ⚠ از وقتی نتیجه ایجنت از انبار سرو می شود، هر چند دقیقه
    دقیقا همان رکورد دوباره می رسد (ts همان لحظه محاسبه است،
    نه لحظه درخواست). بدون این بررسی، توزیع روز هفته و برچسب ها
    با یک سیگنال تکراری پر می شد و آمار بی معنا می گشت.

    خروجی: True اگر واقعا نوشته شد.
    """
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    key = (str(entry.get("ts")), str(entry.get("interval")),
           str(entry.get("asset")))
    if not entry.get("ts"):
        return False
    try:
        if JOURNAL.exists():
            for ln in JOURNAL.read_text(
                    encoding="utf-8").splitlines()[-400:]:
                try:
                    o = json.loads(ln)
                except Exception:
                    continue
                if (str(o.get("ts")), str(o.get("interval")),
                        str(o.get("asset"))) == key:
                    return False
    except Exception:
        pass
    with JOURNAL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return True


def journal_from_decision(d: Dict) -> Dict:
    """ساخت رکورد ژورنال با برچسب های خودکار.

    ⚠ ورودی دو شکل دارد:
      ۱) خروجی کامل decide()
      ۲) نسخه سبک شده ای که از انبار (snapshot) می آید و کلید
         intelligence در آن حذف شده است.
    پس هیچ کلیدی با [] خوانده نمی شود. تجربه ۳۰ سپتامبر: یک
    دسترسی مستقیم d["plan"]["entry"] کل اجرای ایجنت را می شکست.
    """
    ctx = d.get("context") or {}
    intel = d.get("intelligence") or {}
    dec = d.get("decision") or {}
    meta = d.get("meta") or {}
    tags: List[str] = []

    kz = ctx.get("killzone") or {}
    if kz.get("active"):
        tags.append(f"سشن:{(kz['active'] or {}).get('name')}")
    reg = (intel.get("regime") or {}).get("label")
    if reg:
        tags.append(f"رژیم:{reg}")
    if ((d.get("smc") or {}).get("fake") or {}).get("score", 0) > 50:
        tags.append("ریسک روند فیک")
    sen = ctx.get("sentiment") or {}
    if sen.get("ok") and (sen.get("vix") or {}).get("spike"):
        tags.append("جهش VIX")
    ne = (ctx.get("calendar") or {}).get("next_event")
    if ne and ne.get("hours") is not None and ne["hours"] < 6:
        tags.append(f"نزدیک خبر:{ne.get('key')}")
    for it in ((intel.get("divergence") or {}).get("items") or [])[:2]:
        tags.append(f"واگرایی:{it.get('kind')}-{it.get('indicator')}")
    if (intel.get("anomaly") or {}).get("n_recent", 0) > 0:
        tags.append("ردپای نهادی")
    im = ctx.get("intermarket") or {}
    if im.get("ok") and im.get("headwinds"):
        tags.append(f"باد مخالف:{len(im['headwinds'])}")

    pl = d.get("plan") or {}
    sz = pl.get("sizing") or {}
    rec = dict(
        ts=meta.get("generated"), interval=meta.get("interval"),
        asset=meta.get("asset") or ctx.get("asset"),
        # ⚠ برخلاف journal.record، اینجا price به مقیاس *نمایش* است
        # (همان عددی که کاربر دیده). برای اینکه رکورد خودتوضیح باشد،
        # قیمت خام و ضریب هم کنارش می آید.
        price=meta.get("price"),
        raw_price=meta.get("raw_price"),
        scale=meta.get("scale"),
        direction=dec.get("direction"),
        label=dec.get("label"), grade=dec.get("grade"),
        score=dec.get("score"), confidence=dec.get("confidence"),
        gate=(ctx.get("gate") or {}).get("mode"), tags=tags,
        # فقط وقتی ورود واقعی هست؛ طرح خالی {"empty": True} ورود ندارد.
        plan=None if not pl.get("entry") else dict(
            entry=pl.get("entry"), stop=pl.get("stop"),
            tp1=pl.get("tp1"),
            units=sz.get("units"), risk_pct=sz.get("risk_pct")),
    )
    return rec


def journal_stats() -> Dict:
    """تحلیل عملکرد ژورنال: توزیع سیگنال بر اساس سشن، رژیم و روز هفته."""
    if not JOURNAL.exists():
        return dict(ok=False, error="ژورنالی ثبت نشده است", n=0)
    rows = []
    for line in JOURNAL.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    if not rows:
        return dict(ok=False, error="ژورنال خالی است", n=0)

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["weekday"] = df["ts"].dt.dayofweek.map(
        {0: "دوشنبه", 1: "سه شنبه", 2: "چهارشنبه", 3: "پنجشنبه",
         4: "جمعه", 5: "شنبه", 6: "یکشنبه"})
    df["hour"] = df["ts"].dt.hour

    def dist(col):
        return {str(k): int(v) for k, v in df[col].value_counts().head(8).items()}

    tag_counts: Dict[str, int] = {}
    for t in df.get("tags", pd.Series([[]] * len(df))):
        for x in (t or []):
            tag_counts[x] = tag_counts.get(x, 0) + 1

    return dict(
        ok=True, n=len(df),
        first=str(df["ts"].min()), last=str(df["ts"].max()),
        by_direction={str(k): int(v) for k, v in df["direction"].value_counts().items()},
        by_grade=dist("grade"), by_gate=dist("gate"),
        by_weekday=dist("weekday"),
        avg_score=round(float(df["score"].mean()), 2),
        avg_conf=round(float(df["confidence"].mean()), 2),
        top_tags=sorted(tag_counts.items(), key=lambda t: -t[1])[:10],
    )
