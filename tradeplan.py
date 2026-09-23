# -*- coding: utf-8 -*-
"""
================================================================
tradeplan.py — برگه طرح معامله
================================================================
هدف: کاربر نمی تواند هر لحظه داشبورد را چک کند. این ماژول یک
«برگه» می سازد که بدون باز بودن داشبورد هم قابل خواندن است و
می گوید:
    • الان معامله بکنم یا نه (و چرا)
    • اگر آری: ورود کجا، حد ضرر کجا، حد سود کجا
    • هر عدد از کدام سطح واقعی بازار آمده
    • چه اتفاقی طرح را باطل می کند
    • تا کی معتبر است

قانون طلایی این فایل: هیچ عددی ساخته نمی شود. هر سطح یا از
نقدینگی واقعی، بلوک سفارش، پروفایل حجم یا ATR می آید — و منبعش
کنارش نوشته می شود. اگر سطحی واقعی نباشد، به جایش «—» می آید.
================================================================
"""
from __future__ import annotations

import html
import json
import warnings
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np

import assets as A

warnings.filterwarnings("ignore")

TEHRAN = timezone(timedelta(hours=3, minutes=30))

# اعتبار طرح بر حسب ساعت — بر پایه طول کندل
VALID_HOURS = {"5m": 2, "15m": 4, "30m": 6, "1h": 12, "1d": 72}


def _now() -> datetime:
    return datetime.now(TEHRAN)


def _f(x) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _fmt(v: Optional[float], dec: int = 2) -> str:
    return "—" if v is None else f"{v:,.{dec}f}"


# ================================================================ سطوح
def _nearest_levels(liq: Dict, vprof: Dict, price: float,
                    atr: float) -> Dict:
    """
    نزدیک ترین سطوح واقعی بالا و پایین قیمت فعلی.

    منابع (همه واقعی، هیچ کدام حدسی نیست):
      • fresh_bsl / fresh_ssl : استخر نقدینگی دست نخورده
      • periodic              : سقف/کف روز و هفته قبل
      • vprof                 : POC / VAH / VAL از پروفایل حجم
    """
    ups: List[Dict] = []
    dns: List[Dict] = []

    for x in (liq.get("fresh_bsl") or []):
        p = _f(x.get("price"))
        if p and p > price:
            ups.append(dict(price=p, kind="نقدینگی خرید (BSL)",
                            note=f"{x.get('hits', 1)} بار لمس شده"
                                 + ("، سقف دوقلو" if x.get("eq") else ""),
                            src="liq"))
    for x in (liq.get("fresh_ssl") or []):
        p = _f(x.get("price"))
        if p and p < price:
            dns.append(dict(price=p, kind="نقدینگی فروش (SSL)",
                            note=f"{x.get('hits', 1)} بار لمس شده"
                                 + ("، کف دوقلو" if x.get("eq") else ""),
                            src="liq"))

    for lab, pv in (liq.get("periodic") or {}).items():
        hi, lo = _f(pv.get("high")), _f(pv.get("low"))
        if hi and hi > price:
            ups.append(dict(price=hi, kind=f"سقف {lab}", note="سطح مرجع دوره ای",
                            src="periodic"))
        if lo and lo < price:
            dns.append(dict(price=lo, kind=f"کف {lab}", note="سطح مرجع دوره ای",
                            src="periodic"))

    if isinstance(vprof, dict):
        for key, lab in (("poc", "بیشترین حجم (POC)"),
                         ("vah", "سقف ناحیه ارزش (VAH)"),
                         ("val", "کف ناحیه ارزش (VAL)")):
            p = _f(vprof.get(key))
            if not p:
                continue
            tgt = ups if p > price else dns
            tgt.append(dict(price=p, kind=lab,
                            note="جایی که بیشترین معامله انجام شده",
                            src="vprof"))

    ups.sort(key=lambda z: z["price"])
    dns.sort(key=lambda z: -z["price"])
    for z in ups + dns:
        z["dist"] = abs(z["price"] - price)
        z["dist_pct"] = (z["price"] - price) / price * 100
        z["dist_atr"] = (z["dist"] / atr) if atr else None
    return dict(up=ups[:6], down=dns[:6])


def _validate(plan: Dict, direction: int, price: float) -> List[str]:
    """
    بررسی سلامت منطقی طرح. اگر طرح با جهت سیگنال نخواند، به جای
    پنهان کردن، صریح گزارش می شود.
    """
    bad: List[str] = []
    e, s = _f(plan.get("entry")), _f(plan.get("stop"))
    t1 = _f(plan.get("tp1"))
    if e is None or s is None:
        return ["طرح ناقص است (ورود یا حد ضرر محاسبه نشد)"]
    if direction > 0 and s >= e:
        bad.append("حد ضرر بالای نقطه ورود است — با جهت خرید نمی خواند")
    if direction < 0 and s <= e:
        bad.append("حد ضرر پایین نقطه ورود است — با جهت فروش نمی خواند")
    if t1 is not None:
        if direction > 0 and t1 <= e:
            bad.append("هدف اول زیر نقطه ورود است")
        if direction < 0 and t1 >= e:
            bad.append("هدف اول بالای نقطه ورود است")
    r = abs(e - s)
    if r and (r / price * 100) > 5:
        bad.append(f"فاصله حد ضرر غیرعادی بزرگ است ({r / price * 100:.1f}٪)")
    return bad


# ================================================================ ساخت طرح
def build_plan(asset: str = "US30", interval: str = "1h",
               equity: float = 100_000, risk_pct: float = 1.0,
               decision: Optional[Dict] = None) -> Dict:
    """
    یک برگه طرح معامله کامل برای یک دارایی و یک تایم فریم.

    اگر `decision` داده شود، دوباره محاسبه نمی شود (برای سرعت).
    """
    import agent as agent_mod

    prof = A.profile(asset)
    dec = decision or agent_mod.decide(
        interval, equity=equity, asset=prof["key"],
        with_ml=False, with_mtf=True)

    meta = dec["meta"]
    d = dec["decision"]
    price = _f(meta["price"]) or 0.0
    atr = _f(meta.get("atr")) or 0.0
    dec_n = int(prof["decimals"])

    smc_sig = (dec.get("smc") or {}).get("signal") or {}
    base_plan = smc_sig.get("plan") or {}
    k = _f(meta.get("scale")) or 1.0

    ctx = dec.get("context") or {}
    gate = ctx.get("gate") or {}
    inst = dec.get("institutional") or {}
    veto = inst.get("veto") or {}

    # --- سطوح واقعی اطراف قیمت ---
    liq = (dec.get("smc") or {}).get("liq") or {}
    vprof = (dec.get("smc") or {}).get("vprof") or {}
    if not liq or not vprof:
        # ایجنت liq/vprof را برنمی گرداند؛ از نو محاسبه می کنیم
        try:
            import smart_money as smc
            raw = agent_mod.load(interval, symbol=prof["candle_symbol"])
            full = smc.run_full_smc(raw, interval)
            liq = full.get("liq") or {}
            vprof = full.get("vprof") or {}
            # تبدیل به مقیاس نمایشی
            liq = _scale_liq(liq, k)
            vprof = {kk: (vv * k if kk in ("poc", "vah", "val")
                          and isinstance(vv, (int, float)) else vv)
                     for kk, vv in vprof.items()}
        except Exception:
            liq, vprof = {}, {}

    levels = _nearest_levels(liq, vprof, price, atr)

    # --- وضعیت اصلی ---
    direction = int(d["direction"])
    score = _f(d["score"]) or 0.0
    conf = _f(d["confidence"]) or 0.0

    blocked = not gate.get("allowed", True)
    vetoed = (veto.get("long") and score > 0) or (veto.get("short") and score < 0)

    if blocked:
        status, status_fa = "no_trade", "معامله ممنوع"
    elif vetoed:
        status, status_fa = "vetoed", "سیگنال وتو شد"
    elif direction == 0:
        status, status_fa = "neutral", "خنثی — منتظر بمان"
    elif abs(score) < 26:
        status, status_fa = "weak", "سیگنال ضعیف — فقط با تایید"
    else:
        status, status_fa = "active", "سیگنال فعال"

    # --- طرح عددی (فقط اگر معنا داشته باشد) ---
    plan_out: Optional[Dict] = None
    issues: List[str] = []
    if base_plan:
        scaled = {kk: (_f(vv) * k if isinstance(vv, (int, float))
                       and kk in ("entry", "stop", "tp1", "tp2", "tp3", "risk")
                       else vv)
                  for kk, vv in base_plan.items()}
        pdir = 1 if (_f(scaled.get("entry")) or 0) > (_f(scaled.get("stop")) or 0) else -1
        issues = _validate(scaled, pdir, price)

        e, s = _f(scaled.get("entry")), _f(scaled.get("stop"))
        risk_unit = abs(e - s) if (e and s) else None
        risk_money = equity * risk_pct / 100.0
        size = (risk_money / risk_unit) if risk_unit else None

        plan_out = dict(
            side=("خرید (LONG)" if pdir > 0 else "فروش (SHORT)"),
            side_en=("long" if pdir > 0 else "short"),
            entry=e, stop=s,
            tp1=_f(scaled.get("tp1")), tp2=_f(scaled.get("tp2")),
            tp3=_f(scaled.get("tp3")),
            risk_unit=risk_unit,
            risk_pct_price=(risk_unit / price * 100) if (risk_unit and price) else None,
            rr1=_f(base_plan.get("rr1")), rr2=_f(base_plan.get("rr2")),
            rr3=_f(base_plan.get("rr3")),
            at_market=bool(base_plan.get("at_market")),
            size_units=size,
            risk_money=risk_money,
            matches_signal=(direction == 0 or pdir == direction),
            issues=issues,
        )

    # --- دلایل «چرا» ---
    why = _build_reasons(dec, plan_out, levels, price, atr, prof)

    # --- شرایط ابطال ---
    invalidation = _build_invalidation(plan_out, levels, price, atr,
                                       direction, dec_n)

    # --- سطوح هشدار (برای وقتی داشبورد باز نیست) ---
    watch = _build_watch(levels, price, dec_n, prof)

    valid_h = VALID_HOURS.get(interval, 12)
    now = _now()

    return dict(
        ok=True,
        asset=prof["key"], asset_name=prof["name"], emoji=prof["emoji"],
        unit=prof["unit"], decimals=dec_n,
        interval=interval,
        symbol=meta["symbol"],
        price=price,
        spot=_f(meta.get("spot")),
        basis=meta.get("basis"),
        atr=atr,
        generated=now.strftime("%Y-%m-%d %H:%M"),
        valid_until=(now + timedelta(hours=valid_h)).strftime("%Y-%m-%d %H:%M"),
        valid_hours=valid_h,
        status=status, status_fa=status_fa,
        score=score, confidence=conf, grade=d["grade"],
        label=d["label"],
        gate=dict(allowed=gate.get("allowed", True),
                  mode=gate.get("mode"),
                  blocks=gate.get("blocks", []),
                  warnings=gate.get("warnings", [])),
        veto=dict(long=bool(veto.get("long")), short=bool(veto.get("short")),
                  reasons=veto.get("reasons", [])[:4]),
        plan=plan_out,
        levels=levels,
        why=why,
        invalidation=invalidation,
        watch=watch,
        market=_market_line(ctx, prof),
    )


def _scale_liq(liq: Dict, k: float) -> Dict:
    if k == 1.0:
        return liq
    out = dict(liq)
    for key in ("fresh_bsl", "fresh_ssl", "bsl", "ssl"):
        if isinstance(liq.get(key), list):
            out[key] = [dict(x, price=_f(x.get("price")) * k)
                        for x in liq[key] if _f(x.get("price"))]
    if isinstance(liq.get("periodic"), dict):
        out["periodic"] = {lab: dict(high=_f(v.get("high")) * k,
                                     low=_f(v.get("low")) * k)
                           for lab, v in liq["periodic"].items()}
    return out


def _market_line(ctx: Dict, prof: Dict) -> Dict:
    kz = ctx.get("killzone") or {}
    return dict(open=bool(kz.get("market_open")),
                label=kz.get("phase_label") or "—",
                countdown=kz.get("countdown"),
                zone=(kz.get("active") or {}).get("name") if isinstance(kz.get("active"), dict) else None,
                weight=kz.get("weight"),
                nearly_24h=prof["hours"]["nearly_24h"])


# ================================================================ توضیح «چرا»
def _build_reasons(dec: Dict, plan: Optional[Dict], levels: Dict,
                   price: float, atr: float, prof: Dict) -> List[Dict]:
    """
    دلیل هر تصمیم — به زبان ساده، با عدد واقعی.
    هر مورد: عنوان، متن، و اثرش (مثبت/منفی/خنثی).
    """
    out: List[Dict] = []
    dec_n = prof["decimals"]

    # ۱) وزن دار ترین عوامل ایجنت
    parts = sorted(dec.get("parts", []),
                   key=lambda p: -abs(_f(p.get("weight")) or 0))[:5]
    for p in parts:
        w = _f(p.get("weight")) or 0
        if abs(w) < 1:
            continue
        out.append(dict(
            title=p["name"],
            text=f"{p.get('detail', '')} — وزن {w:+.1f} امتیاز",
            impact=("مثبت" if w > 0 else "منفی"),
            sign=(1 if w > 0 else -1),
        ))

    # ۲) توضیح ورود
    if plan and plan.get("entry") is not None:
        e = plan["entry"]
        dist = abs(e - price) / price * 100
        if plan.get("at_market"):
            txt = "قیمت همین الان داخل ناحیه ورود است."
        else:
            side = "بالاتر" if e > price else "پایین تر"
            txt = (f"ناحیه ورود {dist:.2f}٪ {side} از قیمت فعلی است — "
                   f"باید صبر کنید قیمت به آنجا برسد.")
        out.append(dict(
            title="چرا این نقطه ورود؟",
            text=(f"نقطه ورود {_fmt(e, dec_n)} میانگین نزدیک ترین بلوک سفارش "
                  f"یا شکاف قیمتی پرنشده است — جایی که پول بزرگ قبلا فعال "
                  f"بوده و احتمال واکنش دوباره وجود دارد. {txt}"),
            impact="خنثی", sign=0))

    # ۳) توضیح حد ضرر
    if plan and plan.get("stop") is not None:
        s = plan["stop"]
        r = plan.get("risk_unit")
        atr_mult = (r / atr) if (r and atr) else None
        out.append(dict(
            title="چرا این حد ضرر؟",
            text=(f"حد ضرر {_fmt(s, dec_n)} آن سوی آخرین سقف/کف ساختاری "
                  f"به اضافه نصف ATR گذاشته شده. یعنی اگر قیمت به آنجا برسد، "
                  f"فرض اولیه ما باطل شده است — نه اینکه فقط نوسان معمولی بوده. "
                  + (f"فاصله اش {atr_mult:.1f} برابر ATR است"
                     if atr_mult else "")
                  + (f" ({r / price * 100:.2f}٪ قیمت)." if r else ".")),
            impact="خنثی", sign=0))

    # ۴) توضیح اهداف
    if plan and plan.get("tp1") is not None:
        ups = levels.get("up") or []
        dns = levels.get("down") or []
        pool = ups if plan["side_en"] == "long" else dns
        src = pool[0]["kind"] if pool else "سطح محاسباتی"
        out.append(dict(
            title="چرا این اهداف؟",
            text=(f"هدف اول {_fmt(plan['tp1'], dec_n)} روی «{src}» قرار دارد — "
                  f"جایی که سفارش های زیادی جمع شده و قیمت معمولا به آن "
                  f"واکنش نشان می دهد. نسبت سود به زیان هدف اول "
                  f"{plan.get('rr1') or 0:.2f} است"
                  + (f" و هدف دوم {plan.get('rr2') or 0:.2f}."
                     if plan.get("rr2") else ".")),
            # ممیزی ۲۰۲۶-۰۹-۱۹: ادعای «۳R همیشه بهتر از ۲R» بازآزمایی شد.
            # بازه اطمینان ۹۵٪ انتظار در همه تنظیمات از خود اختلاف بزرگ تر
            # بود (مثلا طلا ۱ساعته ۳R=+0.134±0.212). یعنی تفاوت ۲R و ۳R
            # از نویز قابل تفکیک نیست. پس آستانه ۲ حفظ شد، ولی ۳+ به
            # عنوان «بهتر» علامت می خورد چون در هر چهار ترکیب یا بهترین
            # بود یا نزدیک بهترین.
            impact=("خیلی خوب" if (plan.get("rr1") or 0) >= 3
                    else ("مثبت" if (plan.get("rr1") or 0) >= 2 else "خنثی")),
            sign=(1 if (plan.get("rr1") or 0) >= 2 else 0)))

    # ۵) وتو و مسدودی
    inst = dec.get("institutional") or {}
    for r in (inst.get("veto", {}).get("reasons") or [])[:3]:
        out.append(dict(title="هشدار وتو", text=str(r),
                        impact="منفی", sign=-1))
    for b in ((dec.get("context") or {}).get("gate", {}).get("blocks") or [])[:2]:
        out.append(dict(title="مانع معاملاتی", text=str(b),
                        impact="منفی", sign=-1))

    return out


def _build_invalidation(plan: Optional[Dict], levels: Dict, price: float,
                        atr: float, direction: int, dec_n: int) -> List[str]:
    """چه چیزی طرح را باطل می کند."""
    out: List[str] = []
    if plan:
        s = plan.get("stop")
        if s is not None:
            out.append(f"بسته شدن یک کندل آن سوی {_fmt(s, dec_n)} "
                       f"(حد ضرر) — طرح تمام است، دوباره وارد نشوید.")
        e = plan.get("entry")
        if e is not None and not plan.get("at_market"):
            out.append(f"اگر قیمت بدون رسیدن به {_fmt(e, dec_n)} "
                       f"به سمت هدف رفت، این طرح دیگر اجرا نمی شود — "
                       f"دنبالش نروید.")
    if atr:
        out.append(f"حرکت ناگهانی بیش از {_fmt(2 * atr, dec_n)} "
                   f"(دو برابر ATR) در یک کندل — شرایط عوض شده، "
                   f"طرح را دوباره بسازید.")
    out.append("انتشار خبر مهم اقتصادی — تا ۱۵ دقیقه بعدش وارد نشوید.")
    return out


def _build_watch(levels: Dict, price: float, dec_n: int,
                 prof: Dict) -> List[Dict]:
    """سطوح هشدار: اگر قیمت به اینجا رسید، دوباره چک کن."""
    out: List[Dict] = []
    for z in (levels.get("up") or [])[:3]:
        out.append(dict(price=z["price"], dir="بالا", kind=z["kind"],
                        dist_pct=z["dist_pct"],
                        action=("اگر قیمت اینجا را رد کرد و کندل بالایش بسته شد، "
                                "روند صعودی تایید می شود.")))
    for z in (levels.get("down") or [])[:3]:
        out.append(dict(price=z["price"], dir="پایین", kind=z["kind"],
                        dist_pct=z["dist_pct"],
                        action=("اگر قیمت اینجا را شکست و کندل زیرش بسته شد، "
                                "فشار فروش جدی است.")))
    return out


# ================================================================ چند تایم فریم
def build_multi(asset: str = "US30", intervals: Optional[List[str]] = None,
                equity: float = 100_000, risk_pct: float = 1.0) -> Dict:
    """
    طرح را روی چند تایم فریم می سازد و هم جهتی آن ها را می سنجد.

    منطق هم جهتی: اگر تایم فریم بلند و کوتاه یک جهت را بگویند،
    اعتبار سیگنال بیشتر است. اگر مخالف باشند، معامله پرریسک است.
    """
    ivs = intervals or ["1h", "1d"]
    plans: Dict[str, Dict] = {}
    for iv in ivs:
        try:
            plans[iv] = build_plan(asset, iv, equity, risk_pct)
        except Exception as e:
            plans[iv] = dict(ok=False, interval=iv, error=str(e))

    ok = [p for p in plans.values() if p.get("ok")]
    dirs = []
    for p in ok:
        sc = p.get("score") or 0
        dirs.append(1 if sc >= 13 else (-1 if sc <= -13 else 0))

    if not dirs:
        align, align_fa, note = 0.0, "نامشخص", "هیچ تایم فریمی محاسبه نشد."
    elif all(x > 0 for x in dirs):
        align, align_fa = 1.0, "کاملا هم جهت (صعودی)"
        note = "همه تایم فریم ها صعودی هستند — قوی ترین حالت برای خرید."
    elif all(x < 0 for x in dirs):
        align, align_fa = 1.0, "کاملا هم جهت (نزولی)"
        note = "همه تایم فریم ها نزولی هستند — قوی ترین حالت برای فروش."
    elif all(x == 0 for x in dirs):
        align, align_fa = 0.0, "همه خنثی"
        note = "هیچ تایم فریمی جهت مشخصی ندارد — بازار بلاتکلیف است."
    elif any(x > 0 for x in dirs) and any(x < 0 for x in dirs):
        align, align_fa = -1.0, "متضاد"
        note = ("تایم فریم ها با هم مخالف اند — یکی صعودی یکی نزولی. "
                "این خطرناک ترین حالت است؛ بهتر است صبر کنید.")
    else:
        align, align_fa = 0.5, "نیمه هم جهت"
        note = ("فقط بخشی از تایم فریم ها جهت دارند — سیگنال هست ولی "
                "پشتوانه کامل ندارد. حجم کمتر بگیرید.")

    return dict(ok=True, asset=asset, plans=plans, intervals=ivs,
                alignment=align, alignment_fa=align_fa, alignment_note=note,
                generated=_now().strftime("%Y-%m-%d %H:%M"))


# ================================================================ متن ساده
def to_text(p: Dict) -> str:
    """نسخه متنی برگه — برای کپی کردن در تلگرام یا یادداشت."""
    if not p.get("ok"):
        return "خطا: " + str(p.get("error"))
    d = p["decimals"]
    L: List[str] = []
    L.append(f"{p['emoji']} برگه طرح معامله — {p['asset_name']} ({p['asset']})")
    L.append(f"تایم فریم {p['interval']} · ساخته شده {p['generated']} به وقت تهران")
    L.append(f"معتبر تا {p['valid_until']} ({p['valid_hours']} ساعت)")
    L.append("=" * 46)
    L.append(f"قیمت فعلی: {_fmt(p['price'], d)} {p['unit']}")
    L.append(f"وضعیت: {p['status_fa']} | امتیاز {p['score']:+.1f} | "
             f"اطمینان {p['confidence']:.0f}٪ | درجه {p['grade']}")
    L.append(f"بازار: {p['market']['label']}")
    L.append("")

    if p["status"] in ("no_trade", "vetoed", "neutral"):
        L.append("🚫 توصیه: الان وارد معامله نشوید.")
        L.append("دلیل:")
        for b in p["gate"]["blocks"]:
            L.append(f"  • {b}")
        for r in p["veto"]["reasons"]:
            L.append(f"  • {r}")
        if p["status"] == "neutral":
            L.append(f"  • امتیاز {p['score']:+.1f} از آستانه ±۱۳ عبور نکرده "
                     f"— بازار جهت مشخصی ندارد.")
        L.append("")

    pl = p.get("plan")
    if pl:
        head = ("📋 طرح معامله" if p["status"] == "active"
                else "📋 طرح شرطی (فقط در صورت تایید)")
        L.append(head)
        L.append(f"  جهت     : {pl['side']}")
        L.append(f"  ورود    : {_fmt(pl['entry'], d)}"
                 + ("  (قیمت همین الان در ناحیه است)" if pl["at_market"]
                    else "  (منتظر رسیدن قیمت)"))
        _sl = f"  حد ضرر  : {_fmt(pl['stop'], d)}"
        if pl.get("risk_unit") is not None:
            _sl += f"   ریسک {_fmt(pl['risk_unit'], d)}"
            if pl.get("risk_pct_price") is not None:
                _sl += f" ({pl['risk_pct_price']:.2f}٪)"
        L.append(_sl)
        L.append(f"  هدف ۱   : {_fmt(pl['tp1'], d)}   (نسبت {pl['rr1'] or 0:.2f})")
        L.append(f"  هدف ۲   : {_fmt(pl['tp2'], d)}   (نسبت {pl['rr2'] or 0:.2f})")
        L.append(f"  هدف ۳   : {_fmt(pl['tp3'], d)}   (نسبت {pl['rr3'] or 0:.2f})")
        if pl.get("size_units"):
            L.append(f"  حجم     : {pl['size_units']:.2f} واحد "
                     f"(ریسک {pl['risk_money']:,.0f} از سرمایه)")
        if not pl.get("matches_signal"):
            L.append("  ⚠️ این طرح با جهت سیگنال کلی هم خوان نیست.")
        for iss in pl.get("issues", []):
            L.append(f"  ⚠️ {iss}")
        L.append("")

    if p.get("why"):
        L.append("🔍 چرا؟")
        for w in p["why"]:
            mark = "✅" if w["sign"] > 0 else ("⚠️" if w["sign"] < 0 else "•")
            L.append(f"  {mark} {w['title']}: {w['text']}")
        L.append("")

    if p.get("watch"):
        L.append("👀 سطوح مهم (اگر قیمت به اینجا رسید دوباره چک کنید)")
        for w in p["watch"]:
            L.append(f"  {w['dir']:>4} {_fmt(w['price'], d)}  "
                     f"({w['dist_pct']:+.2f}٪)  {w['kind']}")
        L.append("")

    if p.get("invalidation"):
        L.append("❌ چه چیزی این طرح را باطل می کند")
        for iv in p["invalidation"]:
            L.append(f"  • {iv}")
        L.append("")

    L.append("─" * 46)
    L.append("این برگه ابزار کمکی است، نه توصیه سرمایه گذاری.")
    L.append("همه اعداد از داده واقعی بازار محاسبه شده اند.")
    return "\n".join(L)


# ================================================================ HTML
_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0b1020;color:#e8edff;font-family:Vazirmatn,Tahoma,system-ui,sans-serif;
  direction:rtl;padding:18px;line-height:1.85;font-size:14px}
.wrap{max-width:1000px;margin:0 auto}
.card{background:linear-gradient(150deg,#141a33,#101527);border:1px solid #ffffff14;
  border-radius:16px;padding:18px 20px;margin-bottom:14px}
h1{font-size:21px;margin-bottom:4px}
h2{font-size:16px;margin:0 0 11px;padding-bottom:8px;border-bottom:1px solid #ffffff14}
.sub{color:#8b9ac4;font-size:12px}
.hero{display:flex;flex-wrap:wrap;gap:18px;align-items:center;justify-content:space-between}
.px{font-size:30px;font-weight:800;direction:ltr;font-variant-numeric:tabular-nums}
.badge{display:inline-block;padding:5px 13px;border-radius:20px;font-weight:700;font-size:13px}
.b-no{background:#ff4d6a22;color:#ff6b82;border:1px solid #ff4d6a55}
.b-ok{background:#00d68f22;color:#00d68f;border:1px solid #00d68f55}
.b-wa{background:#ffb02022;color:#ffb020;border:1px solid #ffb02055}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:8px 9px;text-align:right;border-bottom:1px solid #ffffff0d}
th{color:#8b9ac4;font-weight:600;font-size:12px}
.num{direction:ltr;text-align:left;font-variant-numeric:tabular-nums;font-weight:700}
.up{color:#00d68f}.dn{color:#ff6b82}.dim{color:#8b9ac4}
.plan{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:6px}
.pb{background:#00000033;border:1px solid #ffffff12;border-radius:11px;padding:11px 13px}
.pb .l{font-size:11px;color:#8b9ac4;margin-bottom:4px}
.pb .v{font-size:18px;font-weight:800;direction:ltr;font-variant-numeric:tabular-nums}
.why{padding:9px 12px;margin:7px 0;border-radius:9px;background:#ffffff06;
  border-inline-start:3px solid #4a7fe0}
.why.pos{border-color:#00d68f}.why.neg{border-color:#ff6b82}
.why b{display:block;margin-bottom:2px}
ul{margin:4px 18px 0 0}li{margin-bottom:5px}
.warn{background:#ff4d6a12;border:1px solid #ff4d6a33;border-radius:11px;
  padding:12px 14px;margin-top:10px}
.foot{text-align:center;color:#66759c;font-size:11.5px;margin-top:16px;line-height:2}
.al{display:flex;gap:9px;flex-wrap:wrap;margin-top:8px}
.al div{flex:1;min-width:130px;background:#00000033;border:1px solid #ffffff12;
  border-radius:10px;padding:10px 12px;text-align:center}
"""


def _esc(s) -> str:
    return html.escape(str(s if s is not None else "—"))


def _plan_html(p: Dict) -> str:
    d = p["decimals"]
    st = p["status"]
    cls = ("b-ok" if st == "active" else
           "b-wa" if st == "weak" else "b-no")

    h = [f'<div class="card"><div class="hero"><div>',
         f'<h1>{_esc(p["emoji"])} {_esc(p["asset_name"])} '
         f'<span class="sub">({_esc(p["asset"])} · {_esc(p["interval"])})</span></h1>',
         f'<div class="sub">ساخته شده {_esc(p["generated"])} · '
         f'معتبر تا {_esc(p["valid_until"])}</div></div>',
         f'<div style="text-align:left"><div class="px">{_fmt(p["price"], d)}'
         f' <span class="sub">{_esc(p["unit"])}</span></div>',
         f'<div class="sub">امتیاز {p["score"]:+.1f} · اطمینان '
         f'{p["confidence"]:.0f}٪ · درجه {_esc(p["grade"])}</div></div>',
         f'<div><span class="badge {cls}">{_esc(p["status_fa"])}</span></div>',
         '</div></div>']

    # هشدار
    if st in ("no_trade", "vetoed", "neutral"):
        h.append('<div class="card"><h2>🚫 توصیه: الان وارد معامله نشوید</h2><ul>')
        for b in p["gate"]["blocks"]:
            h.append(f"<li>{_esc(b)}</li>")
        for r in p["veto"]["reasons"]:
            h.append(f"<li>{_esc(r)}</li>")
        if st == "neutral":
            h.append(f"<li>امتیاز {p['score']:+.1f} از آستانه ±۱۳ عبور نکرده "
                     f"— بازار جهت مشخصی ندارد.</li>")
        h.append('</ul></div>')

    # طرح
    pl = p.get("plan")
    if pl:
        title = ("📋 طرح معامله" if st == "active"
                 else "📋 طرح شرطی — فقط در صورت تایید")
        h.append(f'<div class="card"><h2>{title}</h2>')
        h.append(f'<div class="sub">جهت: <b>{_esc(pl["side"])}</b> · '
                 + ("قیمت همین الان در ناحیه ورود است"
                    if pl["at_market"] else "منتظر رسیدن قیمت به ناحیه ورود")
                 + '</div><div class="plan">')
        rows = [("ورود", pl["entry"], ""),
                ("حد ضرر", pl["stop"], "dn"),
                ("هدف ۱", pl["tp1"], "up"),
                ("هدف ۲", pl["tp2"], "up"),
                ("هدف ۳", pl["tp3"], "up")]
        for lab, val, c in rows:
            h.append(f'<div class="pb"><div class="l">{lab}</div>'
                     f'<div class="v {c}">{_fmt(val, d)}</div></div>')
        h.append('</div>')
        rr = (f'نسبت سود به زیان — هدف۱ <b>{pl["rr1"] or 0:.2f}</b> · '
              f'هدف۲ <b>{pl["rr2"] or 0:.2f}</b> · هدف۳ <b>{pl["rr3"] or 0:.2f}</b>')
        rk = (f'ریسک هر واحد <b>{_fmt(pl["risk_unit"], d)}</b>'
              + (f' ({pl["risk_pct_price"]:.2f}٪ قیمت)'
                 if pl.get("risk_pct_price") else ""))
        sz = (f' · حجم پیشنهادی <b>{pl["size_units"]:.2f}</b> واحد '
              f'(ریسک {pl["risk_money"]:,.0f})' if pl.get("size_units") else "")
        h.append(f'<div class="sub" style="margin-top:9px">{rr}<br>{rk}{sz}</div>')
        if not pl.get("matches_signal") or pl.get("issues"):
            h.append('<div class="warn">⚠️ <b>هشدار درباره همین طرح</b><ul>')
            if not pl.get("matches_signal"):
                h.append("<li>جهت این طرح با جهت سیگنال کلی هم خوان نیست — "
                         "موتور ساختاری و امتیاز نهایی اختلاف دارند.</li>")
            for iss in pl.get("issues", []):
                h.append(f"<li>{_esc(iss)}</li>")
            h.append('</ul></div>')
        h.append('</div>')

    # چرا
    if p.get("why"):
        h.append('<div class="card"><h2>🔍 چرا؟ دلیل هر تصمیم</h2>')
        for w in p["why"]:
            c = "pos" if w["sign"] > 0 else ("neg" if w["sign"] < 0 else "")
            h.append(f'<div class="why {c}"><b>{_esc(w["title"])}</b>'
                     f'{_esc(w["text"])}</div>')
        h.append('</div>')

    # سطوح
    if p.get("watch"):
        h.append('<div class="card"><h2>👀 سطوح مهم — اگر قیمت به اینجا رسید '
                 'دوباره بررسی کنید</h2><table>'
                 '<tr><th>جهت</th><th>قیمت</th><th>فاصله</th>'
                 '<th>نوع سطح</th><th>یعنی چه</th></tr>')
        for w in p["watch"]:
            c = "up" if w["dir"] == "بالا" else "dn"
            h.append(f'<tr><td class="{c}">{_esc(w["dir"])}</td>'
                     f'<td class="num">{_fmt(w["price"], d)}</td>'
                     f'<td class="num {c}">{w["dist_pct"]:+.2f}٪</td>'
                     f'<td>{_esc(w["kind"])}</td>'
                     f'<td class="dim">{_esc(w["action"])}</td></tr>')
        h.append('</table></div>')

    # ابطال
    if p.get("invalidation"):
        h.append('<div class="card"><h2>❌ چه چیزی این طرح را باطل می کند</h2><ul>')
        for iv in p["invalidation"]:
            h.append(f"<li>{_esc(iv)}</li>")
        h.append('</ul></div>')

    return "".join(h)


def to_html(data: Dict, multi: bool = False) -> str:
    """برگه کامل HTML — مستقل و بدون نیاز به سرور."""
    if multi:
        body = []
        a = data.get("plans", {})
        first = next((v for v in a.values() if v.get("ok")), {})
        name = first.get("asset_name", data.get("asset", ""))
        emoji = first.get("emoji", "")
        body.append(f'<div class="card"><h1>{_esc(emoji)} برگه طرح معامله — '
                    f'{_esc(name)}</h1>'
                    f'<div class="sub">مقایسه چند تایم فریم · '
                    f'{_esc(data.get("generated"))} به وقت تهران</div>')
        al = data.get("alignment", 0)
        cls = "b-ok" if al == 1.0 else ("b-no" if al < 0 else "b-wa")
        body.append(f'<div style="margin-top:10px">'
                    f'<span class="badge {cls}">هم جهتی: '
                    f'{_esc(data.get("alignment_fa"))}</span></div>')
        body.append(f'<div class="sub" style="margin-top:7px">'
                    f'{_esc(data.get("alignment_note"))}</div>')
        body.append('<div class="al">')
        for iv, p in a.items():
            if not p.get("ok"):
                continue
            sc = p.get("score", 0)
            c = "up" if sc >= 13 else ("dn" if sc <= -13 else "dim")
            body.append(f'<div><div class="sub">{_esc(iv)}</div>'
                        f'<div class="num {c}" style="font-size:17px">'
                        f'{sc:+.1f}</div>'
                        f'<div class="sub">{_esc(p.get("status_fa"))}</div></div>')
        body.append('</div></div>')
        for iv, p in a.items():
            if p.get("ok"):
                body.append(f'<div class="sub" style="margin:16px 0 6px;'
                            f'font-size:15px;font-weight:700">⏱ تایم فریم {_esc(iv)}</div>')
                body.append(_plan_html(p))
            else:
                body.append(f'<div class="card"><b>تایم فریم {_esc(iv)}</b> — '
                            f'خطا: {_esc(p.get("error"))}</div>')
        inner = "".join(body)
        title = f"طرح معامله — {name}"
    else:
        inner = _plan_html(data)
        title = f"طرح معامله — {data.get('asset_name', '')}"

    return (f'<!DOCTYPE html><html lang="fa" dir="rtl"><head>'
            f'<meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{_esc(title)}</title><style>{_CSS}</style></head>'
            f'<body><div class="wrap">{inner}'
            f'<div class="foot">این برگه ابزار کمکی تحلیل است، نه توصیه '
            f'سرمایه گذاری.<br>همه اعداد از داده واقعی بازار محاسبه شده اند — '
            f'هیچ عددی حدسی نیست.<br>مدیریت سرمایه و تصمیم نهایی با شماست.'
            f'</div></div></body></html>')


# ================================================================ اجرا
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ساخت برگه طرح معامله")
    ap.add_argument("--asset", default="XAUUSD", help="US30 یا XAUUSD")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--multi", action="store_true", help="چند تایم فریم")
    ap.add_argument("--intervals", default="1h,1d")
    ap.add_argument("--equity", type=float, default=100_000)
    ap.add_argument("--risk", type=float, default=1.0)
    ap.add_argument("--out", default="", help="مسیر فایل HTML")
    ap.add_argument("--json", default="", help="مسیر فایل JSON")
    a = ap.parse_args()

    if a.multi:
        data = build_multi(a.asset, a.intervals.split(","), a.equity, a.risk)
        for iv, p in data["plans"].items():
            if p.get("ok"):
                print(to_text(p))
                print()
        print("هم جهتی:", data["alignment_fa"], "—", data["alignment_note"])
        htm = to_html(data, multi=True)
    else:
        data = build_plan(a.asset, a.interval, a.equity, a.risk)
        print(to_text(data))
        htm = to_html(data)

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(htm)
        print(f"\n[+] فایل HTML ذخیره شد: {a.out}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1, default=str)
        print(f"[+] فایل JSON ذخیره شد: {a.json}")
