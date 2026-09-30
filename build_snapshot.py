"""محاسبه نتیجه ها و پوش کردنشان به سایت زنده.

این اسکریپت را گیت هاب اکشنز هر ۱۵ دقیقه اجرا می کند. روی ماشین اکشنز
CPU کامل هست، پس محاسبه ای که روی Render رایگان ۱۴۰ ثانیه طول می کشد
اینجا چند ثانیه است. نتیجه با یک POST امن به سایت فرستاده می شود.

متغیرهای لازم:
    SITE_URL      آدرس سایت، مثل https://YOUR-SITE.onrender.com
    SNAPSHOT_KEY  همان کلید مشترکی که روی Render تنظیم شده

اجرای دستی برای تست:
    SITE_URL=... SNAPSHOT_KEY=... python build_snapshot.py
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

# چه چیزهایی از پیش محاسبه شوند. همان ترکیب هایی که داشبورد می خواهد.
#
# پیش فرض: هر دو دارایی (سایت ترکیبی).
# اگر سایت تک دارایی دارید، متغیر ASSETS را ست کنید تا وقت و دقیقه
# اکشنز بی خود مصرف نشود. مثال برای سایت طلا:
#     ASSETS=XAUUSD
# یا برای هر دو با فاصله یا کاما:
#     ASSETS="US30,XAUUSD"
_raw = (os.environ.get("ASSETS") or "US30,XAUUSD").replace(" ", ",")
_wanted = [a.strip().upper() for a in _raw.split(",") if a.strip()]
_INTERVAL = (os.environ.get("SNAPSHOT_INTERVAL") or "1d").strip()

TARGETS = [(a, _INTERVAL) for a in _wanted if a in ("US30", "XAUUSD")]
if not TARGETS:                       # ورودی غلط ⇒ برگرد به حالت امن
    TARGETS = [("US30", "1d"), ("XAUUSD", "1d")]


def build_one(asset: str, interval: str):
    """یک اجرای کامل ایجنت. خروجی: همان چیزی که روت /api/agent می دهد."""
    import agent as agent_mod
    import web_api

    d = agent_mod.decide(interval=interval, equity=100000, with_ml=True,
                         with_mtf=True, with_coalition=True, scale=True,
                         asset=asset)
    d.pop("intelligence", None)          # حجیم و غیرلازم برای نمایش
    return web_api._clean(d)


def push(items: dict, built_at: float) -> bool:
    """ارسال امن نتیجه ها به سایت."""
    import urllib.error
    import urllib.request

    body = json.dumps({"items": items, "built_at": built_at},
                      ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{SITE}/api/snapshot", data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "X-Snapshot-Key": KEY,
                 "User-Agent": "dow-snapshot-bot"})
    # اولین درخواست ممکن است سایت خواب را بیدار کند → صبر بلند + تلاش مجدد
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.loads(r.read().decode("utf-8"))
            print(f"  پاسخ سایت: {out}")
            return bool(out.get("ok"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")[:200]
            print(f"  تلاش {attempt + 1}: خطای HTTP {e.code} — {detail}")
            if e.code in (403, 503):
                return False             # کلید غلط یا تنظیم نشده → تکرار بی فایده
        except Exception as e:
            msg = str(e)[:120].replace(SITE, "[سایت]") if SITE else str(e)[:120]
            print(f"  تلاش {attempt + 1}: {type(e).__name__} — {msg}")
        time.sleep(10 * (attempt + 1))
    return False


def main() -> int:
    if not SITE or not KEY:
        print("❌ SITE_URL یا SNAPSHOT_KEY تنظیم نشده")
        return 1

    # آدرس را عمدا چاپ نمی کنیم: در ریپوی پابلیک لاگ اجراها عمومی است و
    # ماسک خودکار گیت هاب فقط روی مقدار دقیق Secret کار می کند — اگر کاربر
    # آدرس را با / انتها ذخیره کرده باشد، rstrip بالا ماسک را بی اثر می کند.
    print(f"سایت هدف: تنظیم شد ({len(SITE)} کاراکتر) ✅")
    items, ok_count = {}, 0
    t_all = time.time()

    for asset, interval in TARGETS:
        t0 = time.time()
        try:
            items[f"agent:{asset}:{interval}"] = build_one(asset, interval)
            ok_count += 1
            print(f"✅ {asset} {interval} — {time.time() - t0:.1f} ثانیه")
        except Exception as e:
            print(f"❌ {asset} {interval} — {type(e).__name__}: {str(e)[:150]}")
            traceback.print_exc()

    if not items:
        print("هیچ نتیجه ای ساخته نشد — چیزی پوش نمی شود")
        return 1

    print(f"\nمحاسبه {ok_count}/{len(TARGETS)} مورد در "
          f"{time.time() - t_all:.1f} ثانیه. در حال ارسال…")
    built = time.time()
    sent = push(items, built_at=built)
    print("✅ پوش موفق" if sent else "❌ پوش ناموفق")

    # ── انبار پشتیبان روی ریپو ───────────────────────────────────
    # چرا لازم است: /tmp روی Render رایگان موقتی است و با هر خواب
    # رفتن (۱۵ دقیقه بی کاری) پاک می شود. پوش مستقیم بالا فقط تا
    # اولین ری استارت دوام دارد. این فایل در ریپو کامیت می شود و
    # سایت وقتی حافظه اش خالی است از raw.githubusercontent می خواند،
    # پس با هر ری استارت خودش را ترمیم می کند.
    #
    # ساختار باید دقیقا همان چیزی باشد که snapshot._read_disk
    # انتظار دارد: {key: {payload, built_at, saved_at}}
    try:
        blob = {k: {"payload": v, "built_at": built, "saved_at": built}
                for k, v in items.items()}
        tmp = "snapshot_cache.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(blob, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, "snapshot_cache.json")
        print(f"✅ snapshot_cache.json نوشته شد "
              f"({os.path.getsize('snapshot_cache.json'):,} بایت)")
    except Exception as e:
        print(f"⚠️ نوشتن فایل پشتیبان نشد: {str(e)[:120]}")

    # حتی اگر پوش مستقیم شکست بخورد، فایل ریپو راه نجات است
    return 0 if sent else 1


if __name__ == "__main__":
    sys.exit(main())
