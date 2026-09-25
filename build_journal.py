"""ثبت و ارزیابی سیگنال ها روی گیت هاب اکشنز — با انبار دائمی.

مشکلی که حل می کند
──────────────────
دفترچه سیگنال در فایل journal.jsonl کنار برنامه ذخیره می شد. روی Render
رایگان دیسک موقتی است: هر ری استارت، هر دیپلوی و هر بار که سایت می خوابد
و بیدار می شود فایل پاک می شود. یعنی آمار هیچ وقت جمع نمی شد و هرگز به
۱۰۰ نمونه لازم برای معناداری آماری نمی رسید.

راه حل
──────
منبع حقیقت از Render به ریپوی گیت هاب منتقل می شود. ریپو دائمی است،
تاریخچه دارد و رایگان است.

چرخه هر اجرا:

    ۱) دفترچه فعلیِ سایت خوانده و با نسخه ریپو ادغام می شود
       (رکوردهایی که کاربر با باز کردن داشبورد ساخته از دست نرود)
    ۲) برای هر هدف یک تصمیم واقعی گرفته و ثبت می شود
    ۳) سیگنال های قدیمی با داده واقعی بازار ارزیابی می شوند
    ۴) نسخه کامل به سایت پوش می شود تا کارنامه روی داشبورد دیده شود
    ۵) ورک فلو فایل را در ریپو کامیت می کند  ← ماندگاری واقعی اینجاست

هیچ عدد حدسی ساخته نمی شود. اگر کندل کافی نباشد، سیگنال «در انتظار»
می ماند — دقیقا مثل قبل.

متغیرهای لازم:
    SITE_URL            آدرس سایت، مثل https://YOUR-SITE.onrender.com
    SNAPSHOT_KEY        همان کلید مشترک تنظیم شده روی Render

اختیاری:
    JOURNAL_ASSETS      پیش فرض "US30,XAUUSD"
    JOURNAL_INTERVALS   پیش فرض "1d"  (می توانید "1d,1h" بگذارید)

اجرای دستی برای تست:
    SITE_URL=... SNAPSHOT_KEY=... python build_journal.py
"""
import json
import os
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")

SITE = (os.environ.get("SITE_URL") or "").rstrip("/")
KEY = (os.environ.get("SNAPSHOT_KEY") or "").strip()

_A = (os.environ.get("JOURNAL_ASSETS") or "US30,XAUUSD").replace(" ", ",")
_I = (os.environ.get("JOURNAL_INTERVALS") or "1d").replace(" ", ",")
ASSETS = [a.strip().upper() for a in _A.split(",")
          if a.strip().upper() in ("US30", "XAUUSD")] or ["US30", "XAUUSD"]
INTERVALS = [i.strip() for i in _I.split(",")
             if i.strip() in ("5m", "15m", "30m", "1h", "1d")] or ["1d"]
TARGETS = [(a, i) for a in ASSETS for i in INTERVALS]


def _req(path, data=None, timeout=180):
    """درخواست به سایت با کلید مشترک. خروجی: دیکشنری پاسخ یا None."""
    import urllib.error
    import urllib.request
    body = None
    if data is not None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{SITE}{path}", data=body,
        method=("POST" if body else "GET"),
        headers={"Content-Type": "application/json",
                 "X-Snapshot-Key": KEY,
                 "User-Agent": "dow-journal-bot"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def pull_from_site():
    """رکوردهای ثبت شده روی سایت را بردار و با نسخه ریپو ادغام کن.

    چرا لازم است: وقتی کاربر داشبورد را باز می کند، agent.decide خودش
    یک رکورد روی سایت می سازد. آن رکورد با ری استارت بعدی نابود می شود
    مگر اینکه اینجا برش داریم.

    اولین درخواست ممکن است سایتِ خواب را بیدار کند ⇒ تلاش مجدد.
    """
    import journal as jr
    for attempt in range(3):
        try:
            out = _req("/api/journal?what=export")
            rows = out.get("rows") or []
            if not rows:
                print("  سایت رکوردی نداشت (طبیعی است اگر تازه ری استارت شده)")
                return 0
            m = jr.merge(rows)
            print(f"  از سایت {len(rows)} رکورد گرفته شد → "
                  f"{m.get('added', 0)} تازه، {m.get('updated', 0)} به روز")
            return len(rows)
        except Exception as e:
            print(f"  تلاش {attempt + 1}: {type(e).__name__} — {str(e)[:110]}")
            time.sleep(12 * (attempt + 1))
    print("  ⚠️ خواندن از سایت نشد — با نسخه ریپو ادامه می دهیم")
    return 0


def record_all():
    """برای هر هدف یک تصمیم واقعی بگیر. agent.decide خودش ثبت می کند."""
    import agent as agent_mod
    ok = 0
    for asset, iv in TARGETS:
        t0 = time.time()
        try:
            d = agent_mod.decide(interval=iv, asset=asset, equity=100000,
                                 with_ml=False, with_mtf=False,
                                 with_coalition=False, scale=False)
            sc = float((d.get("decision") or {}).get("score") or 0)
            lbl = (d.get("decision") or {}).get("label") or "—"
            # آستانه ۱٫۰ همان شرط داخل agent.decide است؛ زیر آن ثبت نمی شود
            mark = "ثبت شد" if abs(sc) >= 1.0 else "زیر آستانه — ثبت نشد"
            print(f"  {asset:<7} {iv:<3} امتیاز {sc:+7.1f}  {lbl:<12} "
                  f"{mark}  ({time.time() - t0:.0f}s)")
            ok += 1
        except Exception as e:
            print(f"  ❌ {asset} {iv} — {type(e).__name__}: {str(e)[:130]}")
            traceback.print_exc()
    return ok


def main() -> int:
    if not SITE or not KEY:
        print("❌ SITE_URL یا SNAPSHOT_KEY تنظیم نشده")
        return 1
    print(f"سایت هدف: تنظیم شد ({len(SITE)} کاراکتر) ✅")
    print(f"اهداف: {', '.join('%s %s' % t for t in TARGETS)}\n")

    import journal as jr

    print("▸ ۱/۴ گرفتن دفترچه از سایت")
    pull_from_site()
    n0 = len(jr.load_rows())
    print(f"  دفترچه بعد از ادغام: {n0} رکورد\n")

    print("▸ ۲/۴ ثبت سیگنال تازه")
    record_all()
    n1 = len(jr.load_rows())
    print(f"  دفترچه: {n0} → {n1} رکورد\n")

    print("▸ ۳/۴ ارزیابی سیگنال های سررسید شده")
    try:
        ev = jr.evaluate()
        new = int(ev.get("newly_checked") or 0)
        print(f"  {new} سیگنال تازه ارزیابی شد")
        if ev.get("n"):
            print(f"  کارنامه: {ev['n']} ارزیابی شده از {ev['total_recorded']} — "
                  f"نرخ برد {ev.get('win_rate')}٪ · میانگین R {ev.get('avg_r')}")
            for k, v in (ev.get("by_interval") or {}).items():
                print(f"    {k:<4} n={v['n']:<4} برد {v['win_rate']:>5}٪  "
                      f"R {v['avg_r']:+.3f}" + ("" if v["enough"] else "  (نمونه کم)"))
        else:
            print(f"  {ev.get('note')}")
    except Exception as e:
        print(f"  ❌ ارزیابی شکست خورد: {type(e).__name__}: {str(e)[:150]}")
        traceback.print_exc()

    print("\n▸ ۴/۴ پوش دفترچه کامل به سایت")
    rows = jr.load_rows()
    sent = False
    for attempt in range(3):
        try:
            out = _req("/api/snapshot",
                       {"items": {"journal:rows": rows,
                                  "journal:summary": jr.summary()},
                        "built_at": time.time()})
            print(f"  پاسخ سایت: {json.dumps(out, ensure_ascii=False)[:220]}")
            sent = bool(out.get("ok"))
            break
        except Exception as e:
            print(f"  تلاش {attempt + 1}: {type(e).__name__} — {str(e)[:110]}")
            time.sleep(12 * (attempt + 1))

    print(f"\n{'✅ پوش موفق' if sent else '⚠️ پوش ناموفق'} — "
          f"{len(rows)} رکورد در دفترچه")
    # حتی اگر پوش نشد، کامیت ریپو باید انجام شود: ماندگاری مهم تر از نمایش
    return 0


if __name__ == "__main__":
    sys.exit(main())
