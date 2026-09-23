# -*- coding: utf-8 -*-
"""
پرسش و پاسخ قاعده محور — بدون هوش مصنوعی، بدون حدس.

چرا قاعده محور و نه هوش مصنوعی واقعی
────────────────────────────────────
مدل های زبانی یا کلید API پولی می خواهند یا چند گیگابایت حافظه که
رندر رایگان ندارد. مهم تر: مدل زبانی می تواند عدد از خودش بسازد،
و این با قانون «هیچ چیز حدسی» تناقض دارد.

این ماژول به جایش:
  ۱) سوال فارسی را با کلیدواژه به یک «موضوع» نگاشت می کند
  ۲) عدد را از همان API های واقعی داشبورد می خواند
  ۳) جواب را از داده می سازد

اگر داده ای نباشد، صریح می گوید «نمی دانم» — هرگز عدد نمی سازد.
هر جواب منبعش را هم برمی گرداند تا کاربر بتواند راستی آزمایی کند.
"""
from __future__ import annotations

import datetime as _dt
import re
from typing import Callable, Dict, List, Optional

TEHRAN = _dt.timezone(_dt.timedelta(hours=3, minutes=30))

# ————————————————————————————————— ابزار متن


def _norm(t: str) -> str:
    """یکسان سازی نویسه های فارسی/عربی و ارقام."""
    if not t:
        return ""
    t = t.strip()
    pairs = [("ي", "ی"), ("ك", "ک"), ("ؤ", "و"), ("إ", "ا"), ("أ", "ا"),
             ("ة", "ه"), ("\u200c", " "), ("ٔ", "")]
    for a, b in pairs:
        t = t.replace(a, b)
    for i, d in enumerate("۰۱۲۳۴۵۶۷۸۹"):
        t = t.replace(d, str(i))
    for i, d in enumerate("٠١٢٣٤٥٦٧٨٩"):
        t = t.replace(d, str(i))
    t = re.sub(r"[؟?!.،,:؛]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _has(t: str, *words: str) -> bool:
    return any(w in t for w in words)


def _num(x, d: int = 2) -> str:
    try:
        v = float(x)
        return f"{v:,.{d}f}"
    except Exception:
        return "—"


# ————————————————————————————————— دسترسی به داده


class _Ctx:
    """لود تنبل داده ها — فقط چیزی که سوال لازم دارد گرفته می شود."""

    def __init__(self, asset: str = "US30", interval: str = "1d"):
        self.asset = asset
        self.interval = interval
        self._c: Dict = {}

    def _get(self, key: str, fn: Callable):
        if key not in self._c:
            try:
                self._c[key] = fn()
            except Exception as e:
                self._c[key] = dict(_error=str(e)[:120])
        return self._c[key]

    def agent(self) -> Dict:
        def _f():
            import agent as A
            return A.decide(interval=self.interval, asset=self.asset)
        return self._get("agent", _f)

    def analysis(self) -> Dict:
        def _f():
            import web_api as W
            return W.cached_payload(self.interval, 160, True, False,
                                    True, self.asset)
        return self._get("analysis", _f)

    def extras(self) -> Dict:
        def _f():
            import macro_extras as M
            return M.build_extras(self.asset)
        return self._get("extras", _f)

    def quality(self) -> Dict:
        def _f():
            import signal_filter as S
            d = self.agent()
            sc = (d.get("decision") or {}).get("score", 0)
            return S.assess(self.asset, float(sc or 0), self.interval)
        return self._get("quality", _f)

    def cross(self) -> Dict:
        def _f():
            import cross_asset as C
            return C.correlation()
        return self._get("cross", _f)

    def cash(self) -> Dict:
        def _f():
            import dow_cash as D
            return D.cash_price()
        return self._get("cash", _f)


def _unknown(why: str) -> Dict:
    return dict(ok=True, answer=f"نمی‌دانم — {why}", confident=False,
                source="—")


def _ans(text: str, source: str, extra: Optional[Dict] = None) -> Dict:
    out = dict(ok=True, answer=text, confident=True, source=source)
    if extra:
        out.update(extra)
    return out


# ————————————————————————————————— پاسخ دهنده ها


def _a_signal(c: _Ctx) -> Dict:
    d = c.agent()
    if d.get("_error"):
        return _unknown("تحلیل ایجنت در دسترس نیست")
    dec = d.get("decision") or {}
    m = d.get("meta") or {}
    q = c.quality()

    lines = [f"سیگنال فعلی {m.get('asset_name', c.asset)} در تایم‌فریم "
             f"{c.interval}: **{dec.get('label', '—')}**",
             f"امتیاز {_num(dec.get('score'), 1)} · گرید "
             f"{dec.get('grade', '—')} · اطمینان "
             f"{_num(dec.get('confidence'), 0)}٪"]
    if q.get("ok"):
        lines.append(f"کیفیت: {q.get('tier_fa')} — {q.get('reason', '')}")
    if m.get("issued_at"):
        lines.append(f"صادر شده {m['issued_at']} به وقت تهران، معتبر تا "
                     f"{m.get('valid_until')} ({m.get('valid_hours')} ساعت).")
    return _ans("\n".join(lines), "ایجنت + فیلتر کیفیت")


def _a_why(c: _Ctx) -> Dict:
    d = c.agent()
    if d.get("_error"):
        return _unknown("تحلیل ایجنت در دسترس نیست")
    parts = d.get("parts") or []
    if not parts:
        return _unknown("اجزای امتیاز برنگشت")

    rows = []
    for p in parts:
        if isinstance(p, dict):
            nm, w = p.get("name"), p.get("weight")
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            nm, w = p[0], p[1]
        else:
            continue
        try:
            w = float(w)
        except Exception:
            continue
        if abs(w) >= 0.5:
            rows.append((nm, w))
    if not rows:
        return _unknown("هیچ عامل مؤثری ثبت نشده")

    rows.sort(key=lambda r: -abs(r[1]))
    dec = d.get("decision") or {}
    out = [f"تصمیم «{dec.get('label', '—')}» از جمع این عوامل آمده:"]
    for nm, w in rows[:8]:
        out.append(f"  {'🟢' if w > 0 else '🔴'} {nm}: {w:+.1f}")
    mo = dec.get("modifiers") or {}
    if mo:
        out.append(f"سپس در ضرایب ضرب شد — فیک {mo.get('fake')} · "
                   f"HFT {mo.get('hft')} · سشن {mo.get('session')} · "
                   f"رژیم {mo.get('regime')}")
    return _ans("\n".join(out), "اجزای امتیاز ایجنت")


def _a_plan(c: _Ctx) -> Dict:
    d = c.agent()
    if d.get("_error"):
        return _unknown("تحلیل ایجنت در دسترس نیست")
    p = d.get("plan") or {}
    if not p or not p.get("entry"):
        dec = (d.get("decision") or {}).get("label", "خنثی")
        return _ans(f"الان طرح معاملاتی فعالی وجود ندارد چون جهت "
                    f"«{dec}» است. وقتی جهت روشن شود، ورود و حد ضرر "
                    f"ساخته می‌شود.", "ایجنت")
    out = [p.get("source_fa", "طرح معاملاتی"),
           f"ورود: {_num(p.get('entry'))}",
           f"حد ضرر: {_num(p.get('stop'))}",
           f"هدف ۱: {_num(p.get('tp1'))} · هدف ۲: {_num(p.get('tp2'))}"]
    if p.get("why_differs"):
        out.append(p["why_differs"])
    if p.get("smc_alternative"):
        s = p["smc_alternative"]
        out.append(f"⚠️ طرح پول هوشمند ({_num(s.get('entry'))} / "
                   f"{_num(s.get('stop'))}) جهت مخالف دارد — دنبال نکنید.")
    return _ans("\n".join(out), "طرح اجرایی ایجنت")


def _a_stop(c: _Ctx) -> Dict:
    d = c.agent()
    p = (d or {}).get("plan") or {}
    if p.get("stop"):
        txt = [f"حد ضرر پیشنهادی: **{_num(p['stop'])}**",
               f"ورود {_num(p.get('entry'))} — فاصله "
               f"{_num(abs(float(p['entry']) - float(p['stop'])))} واحد."]
        sz = p.get("sizing") or {}
        if sz.get("units"):
            txt.append(f"حجم پیشنهادی {sz['units']} واحد "
                       f"(ریسک {sz.get('risk_pct')}٪).")
        txt.append("برای محاسبه دقیق حجم، از ماشین‌حساب نوار "
                   "«طلا ↔ داوجونز» استفاده کنید.")
        return _ans("\n".join(txt), "طرح ایجنت")
    # طرح فعالی نیست → از ATR واقعی یک فاصله متعارف بده.
    # ATR زیر indicators است، نه meta.
    an = c.analysis()
    atr = None
    if isinstance(an, dict):
        for box in (an.get("indicators"), an.get("meta"), an):
            if isinstance(box, dict) and box.get("atr"):
                atr = box["atr"]
                break
    last = None
    if isinstance(an, dict):
        last = (an.get("price") or {}).get("last")

    if atr:
        d15 = float(atr) * 1.5
        t = [f"الان طرح فعالی نیست (جهت خنثی)، ولی ATR فعلی "
             f"**{_num(atr)}** است.",
             f"حد ضرر متعارف ۱٫۵ برابر ATR یعنی حدود {_num(d15)} واحد "
             f"فاصله از نقطه ورود."]
        if last:
            t.append(f"با قیمت فعلی {_num(last)}: خرید ← حد ضرر حوالی "
                     f"{_num(float(last) - d15)} · فروش ← حوالی "
                     f"{_num(float(last) + d15)}")
        t.append("این یک فاصله متعارف است، نه سیگنال ورود.")
        return _ans("\n".join(t), "ATR تحلیل زنده")
    return _unknown("طرح فعالی وجود ندارد و ATR هم نیامد")


def _a_price(c: _Ctx) -> Dict:
    if c.asset == "US30":
        d = c.cash()
        if d.get("ok") and d.get("price"):
            t = [f"داوجونز: **{_num(d['price'], 0)}**"]
            if d.get("change") is not None:
                t.append(f"تغییر {d['change']:+,.0f} "
                         f"({d.get('change_pct'):+.2f}٪)")
            t.append(f"منبع: {d.get('source_fa')} — {d.get('accuracy_fa')}")
            if d.get("delay_fa"):
                t.append(f"⏱ {d['delay_fa']}")
            return _ans(" · ".join(t), "فید نقدی داوجونز")
    # طلا (و پشتیبان داوجونز): قیمت در شاخه price تحلیل است
    an = c.analysis()
    if not isinstance(an, dict):
        return _unknown("قیمت از هیچ منبعی نیامد")

    pr = an.get("price") or {}
    meta = an.get("meta") or {}
    px = pr.get("last") or meta.get("spot")
    if not px:
        return _unknown("قیمت از هیچ منبعی نیامد")

    dec = meta.get("decimals", 2)
    name = meta.get("asset_name") or c.asset
    t = [f"{name}: **{_num(px, dec)}**"]
    if pr.get("change") is not None:
        t.append(f"تغییر {pr['change']:+,.{dec}f} "
                 f"({(pr.get('change_pct') or 0):+.2f}٪)")
    if pr.get("high") and pr.get("low"):
        t.append(f"سقف {_num(pr['high'], dec)} / "
                 f"کف {_num(pr['low'], dec)}")
    src = "نرخ لحظه‌ای" if meta.get("spot") else "آخرین کندل"
    t.append(f"منبع: {src} ({meta.get('symbol', '—')})")
    return _ans(" · ".join(t), "تحلیل زنده")


def _a_session(c: _Ctx) -> Dict:
    e = c.extras()
    kz = (e or {}).get("killzone") or {}
    if not kz.get("ok"):
        return _unknown("اطلاعات جلسه معاملاتی نیامد")
    act = kz.get("active")
    now = kz.get("now_tehran", "")
    if act:
        t = (f"الان داخل کیل‌زون **{act.get('name')}** هستیم "
             f"(ساعت {now} تهران). "
             f"{act.get('ends_in_min', 0):.0f} دقیقه دیگر تمام می‌شود.")
        if act.get("note"):
            t += f" {act['note']}"
    else:
        nx = kz.get("next") or {}
        t = (f"الان خارج از کیل‌زون هستیم (ساعت {now} تهران). "
             f"جلسه بعدی: {nx.get('name', '—')}.")
    t += ("\nشما گفتید در جلسه لندن و نیویورک معامله می‌کنید — "
          "پرحجم‌ترین پنجره همپوشانی آن دو است.")
    return _ans(t, "تایمر کیل‌زون")


def _a_dollar(c: _Ctx) -> Dict:
    e = c.extras()
    dx = (e or {}).get("dxy") or {}
    if not dx.get("ok"):
        return _unknown("داده شاخص دلار نیامد")
    t = [f"شاخص دلار (DXY): **{_num(dx.get('dxy_last'))}**",
         f"همبستگی ۲۰ روزه با {c.asset}: {dx.get('corr_20')}",
         f"وضعیت: {dx.get('tone_fa')}"]
    if dx.get("broken"):
        t.append("⚠️ رابطه معمول شکسته — با احتیاط.")
    return _ans("\n".join(t), "همبستگی دلار")


def _a_fomc(c: _Ctx) -> Dict:
    e = c.extras()
    f = (e or {}).get("fomc") or {}
    if not f.get("ok"):
        return _unknown("تقویم فدرال رزرو نیامد")
    t = [f"نشست بعدی فدرال رزرو: **{f.get('date')}**",
         f"{f.get('days')} روز دیگر · ساعت {f.get('when_tehran')} تهران",
         f"سطح ریسک: {f.get('risk_fa', f.get('risk'))}"]
    if f.get("has_projections"):
        t.append("⚠️ این نشست گزارش چشم‌انداز اقتصادی دارد — نوسان بیشتر.")
    return _ans("\n".join(t), "تقویم رسمی فدرال رزرو")


def _a_hft(c: _Ctx) -> Dict:
    an = c.analysis()
    h = (an or {}).get("hft") or {}
    if not h:
        return _unknown("داده ردپای الگوریتمی نیامد")
    t = [f"شاخص فعالیت الگوریتمی: {_num(h.get('hft_index'), 0)}/۱۰۰",
         f"جذب نقدینگی {h.get('n_absorb', 0)} مورد · "
         f"شکار استاپ {h.get('n_hunt', 0)} مورد"]
    ev = (h.get("hunt_events") or [])[:3]
    if ev:
        t.append("نزدیک‌ترین شکار استاپ‌ها:")
        for x in ev:
            t.append(f"  • {_num(x.get('price'))} "
                     f"({x.get('dist_pct'):+.2f}٪ از قیمت فعلی)")
    ab = (h.get("absorb_events") or [])[:2]
    if ab:
        t.append("جذب نقدینگی در:")
        for x in ab:
            t.append(f"  • {_num(x.get('price'))} "
                     f"({x.get('dist_pct'):+.2f}٪)")
    return _ans("\n".join(t), "ردپای HFT")


def _a_fake(c: _Ctx) -> Dict:
    an = c.analysis()
    f = (an or {}).get("fake") or {}
    if not f:
        return _unknown("داده شکست جعلی نیامد")
    t = [f"ریسک روند فیک: {_num(f.get('score'), 0)}/۱۰۰ — "
         f"{f.get('verdict', '')}"]
    w = f.get("watch_levels") or []
    if w:
        t.append("سطوحی که باید مراقبشان بود:")
        for x in w:
            t.append(f"  • {x.get('side_fa')} {_num(x.get('level'))} "
                     f"({x.get('dist_pct'):+.2f}٪)")
        t.append("اگر قیمت رد شود ولی کندل داخل ببندد ← شکست جعلی.")
    return _ans("\n".join(t), "تشخیص روند فیک")


def _a_coalition(c: _Ctx) -> Dict:
    an = c.analysis()
    co = (an or {}).get("coalition") or {}
    if not co.get("ok"):
        return _unknown("داده ائتلاف نیامد")
    t = [f"{co.get('verdict')}",
         f"سبد سنجیده‌شده: {co.get('basket_fa', '—')}",
         f"هماهنگی {_num((co.get('agreement') or 0) * 100, 0)}٪"]
    for m in (co.get("members") or [])[:4]:
        z = m.get("zone") or {}
        t.append(f"  • {m.get('name')}: {m.get('action_fa', '—')}"
                 + (f" حول {_num(z.get('poc'))}" if z.get("poc") else ""))
    return _ans("\n".join(t), "ائتلاف بازیگران بزرگ")


def _a_corr(c: _Ctx) -> Dict:
    r = c.cross()
    if not r.get("ok"):
        return _unknown("همبستگی محاسبه نشد")
    return _ans(f"همبستگی طلا و داوجونز: ۲۰ روزه {r.get('corr_20')} · "
                f"۶۰ روزه {r.get('corr_60')} · ۱۲۰ روزه {r.get('corr_120')}\n"
                f"{r.get('tone_fa')} (بر پایه {r.get('days')} روز داده).",
                "همبستگی بین دارایی")


def _a_should_trade(c: _Ctx) -> Dict:
    q = c.quality()
    e = c.extras()
    kz = (e or {}).get("killzone") or {}
    t = []
    if q.get("ok"):
        t.append(f"کیفیت سیگنال: {q.get('tier_fa')}")
        t.append(q.get("reason", ""))
        t.append("✅ قابل معامله" if q.get("tradeable")
                 else "❌ فعلاً وارد نشوید")
    else:
        t.append("کیفیت سیگنال سنجیده نشد.")
    if kz.get("ok"):
        act = kz.get("active")
        t.append(f"جلسه: {act.get('name') if act else 'خارج از کیل‌زون'}")
    d = c.agent()
    gate = ((d.get("context") or {}).get("gate") or {}) if d else {}
    if gate.get("mode"):
        t.append(f"دروازه ورود: {gate['mode']}")
        for b in (gate.get("blocks") or [])[:3]:
            t.append(f"  ⛔ {b}")
    return _ans("\n".join(x for x in t if x), "کیفیت + کیل‌زون + دروازه")


def _a_backtest(c: _Ctx) -> Dict:
    try:
        import json
        import os
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "backtest_results.json")
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return _unknown("نتایج بک‌تست در دسترس نیست")
    t = ["نتیجه بک‌تست موتور روی گذشته:"]
    for k, r in (d.get("runs") or {}).items():
        if r.get("ok"):
            t.append(f"  • {k}: {r['n']} معامله · برد {r['win_rate']}٪ · "
                     f"میانگین {r['avg_r']:+.3f}R")
    oos = (d.get("threshold_study") or {}).get("out_of_sample") or {}
    if oos.get("verdict"):
        t.append(oos["verdict"])
    t.append("تایم‌فریم یک‌ساعته زیان‌ده بود؛ فقط روزانه اعتبارسنجی شده.")
    return _ans("\n".join(t), "بک‌تست ۸۸۲ سیگنال")


def _a_help(c: _Ctx) -> Dict:
    return _ans(
        "می‌توانید این‌ها را بپرسید:\n"
        "  • سیگنال الان چیست؟\n"
        "  • چرا این سیگنال صادر شد؟\n"
        "  • ورود و حد ضررم کجا باشد؟\n"
        "  • قیمت الان چند است؟\n"
        "  • الان وقت معامله هست؟\n"
        "  • وضعیت دلار چطور است؟\n"
        "  • نشست فدرال رزرو کی است؟\n"
        "  • شکار استاپ در چه قیمتی بوده؟\n"
        "  • شکست جعلی کجاست؟\n"
        "  • بانک‌ها چه می‌کنند؟\n"
        "  • همبستگی طلا و داوجونز چقدر است؟\n"
        "  • نتیجه بک‌تست چه بود؟",
        "راهنما")


# ————————————————————————————————— نگاشت موضوع
# ترتیب مهم است: خاص تر اول، عمومی تر آخر.

_RULES: List = [
    ("help", ("چه سوال", "چی بپرسم", "راهنما", "کمک", "چیکار میتون",
              "چه کار میکن")),
    ("backtest", ("بک تست", "بکتست", "backtest", "نرخ برد", "گذشته چطور",
                  "چقدر دقیق", "اعتبارسنجی")),
    ("hft", ("شکار استاپ", "جذب نقدینگی", "hft", "ربات", "الگوریتم",
             "ردپا")),
    ("fake", ("شکست جعلی", "روند فیک", "فیک", "تله", "جعلی")),
    ("coalition", ("ائتلاف", "بانک", "معدن", "انباشت", "توزیع",
                   "نهاد")),
    ("corr", ("همبستگی", "رابطه طلا", "طلا و داو", "correlation")),
    ("fomc", ("فدرال", "fomc", "نرخ بهره", "نشست", "پاول")),
    ("dollar", ("دلار", "dxy", "شاخص دلار")),
    ("session", ("کیل زون", "کیلزون", "جلسه", "سشن", "لندن", "نیویورک",
                 "ساعت معامله", "چه ساعتی")),
    ("should", ("وقت معامله", "الان وارد", "بخرم", "بفروشم", "وارد بشم",
                "معامله کنم", "ورود بزنم")),
    ("stop", ("حد ضرر", "استاپ لاس", "stop", "ریسک چقدر", "حجم")),
    ("plan", ("طرح", "ورود", "حد سود", "تارگت", "هدف", "tp", "برنامه")),
    ("price", ("قیمت", "چند شد", "چنده", "نرخ فعلی")),
    ("why", ("چرا", "دلیل", "علت", "بر چه اساس", "چطور تصمیم")),
    ("signal", ("سیگنال", "تحلیل", "نظرت", "وضعیت", "چه خبر")),
]

_HANDLERS: Dict[str, Callable] = {
    "help": _a_help, "backtest": _a_backtest, "hft": _a_hft,
    "fake": _a_fake, "coalition": _a_coalition, "corr": _a_corr,
    "fomc": _a_fomc, "dollar": _a_dollar, "session": _a_session,
    "should": _a_should_trade, "stop": _a_stop, "plan": _a_plan,
    "price": _a_price, "why": _a_why, "signal": _a_signal,
}


def classify(question: str) -> Optional[str]:
    t = _norm(question)
    if not t:
        return None
    for topic, kws in _RULES:
        if _has(t, *kws):
            return topic
    return None


_GOLD_WORDS = ("طلا", "انس", "gold", "xau", "طلای")
_DOW_WORDS = ("داو", "داوجونز", "dow", "us30", "شاخص داو")


def _asset_from_q(q: str, default: str) -> str:
    """اگر اسم دارایی در خود سوال آمده، آن را ترجیح بده."""
    n = _norm(q)
    gold = any(w in n for w in _GOLD_WORDS)
    dow = any(w in n for w in _DOW_WORDS)
    if gold and not dow:
        return "XAUUSD"
    if dow and not gold:
        return "US30"
    return default          # هیچ‌کدام یا هر دو → تب جاری


def ask(question: str, asset: str = "US30",
        interval: str = "1d") -> Dict:
    """پاسخ به یک سوال فارسی — فقط از داده واقعی."""
    q = (question or "").strip()
    if not q:
        return dict(ok=False, error="سوالی نوشته نشده")

    topic = classify(q)
    if topic is None:
        return dict(
            ok=True, confident=False, topic=None,
            answer=("این سوال را نفهمیدم. من یک دستیار قاعده‌محورم — "
                    "هوش مصنوعی نیستم و چیزی از خودم نمی‌سازم. "
                    "بنویسید «راهنما» تا فهرست سوال‌های قابل پاسخ "
                    "را ببینید."),
            source="—")

    # اگر خود سوال اسم دارایی را برده، همان بر تب جاری مقدم است.
    asset = _asset_from_q(q, asset)

    ctx = _Ctx(asset, interval)
    try:
        res = _HANDLERS[topic](ctx)
    except Exception as e:
        return dict(ok=True, confident=False, topic=topic,
                    answer=f"موقع خواندن داده خطا خوردم: {str(e)[:120]}",
                    source="—")

    res["topic"] = topic
    res["asset"] = asset
    res["interval"] = interval
    res["asked_at"] = _dt.datetime.now(TEHRAN).strftime("%H:%M")
    res["engine"] = "قاعده‌محور — بدون هوش مصنوعی، فقط داده واقعی"
    return res


def suggestions() -> List[str]:
    return ["سیگنال الان چیست؟", "چرا این سیگنال؟",
            "حد ضررم کجا باشد؟", "الان وقت معامله است؟",
            "شکار استاپ در چه قیمتی؟", "شکست جعلی کجاست؟",
            "وضعیت دلار چطور است؟", "نتیجه بک‌تست چه بود؟"]


if __name__ == "__main__":
    for q in ("سیگنال الان چیه؟", "چرا؟", "شکار استاپ کجا بوده",
              "حد ضررم کجا باشه", "الان وقت معامله هست؟", "بلابلا"):
        r = ask(q, "US30", "1d")
        print(f"\n▸ {q}\n  [{r.get('topic')}] "
              f"{r['answer'][:150]}")
