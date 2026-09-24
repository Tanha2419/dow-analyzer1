"""محاسبه نتیجه ها و پوش کردنشان به سایت زنده.

این اسکریپت را گیت هاب اکشنز هر ۱۵ دقیقه اجرا می کند. روی ماشین اکشنز
CPU کامل هست، پس محاسبه ای که روی Render رایگان ۱۴۰ ثانیه طول می کشد
اینجا چند ثانیه است. نتیجه با یک POST امن به سایت فرستاده می شود.

متغیرهای لازم:
    SITE_URL      آدرس سایت، مثل https://dow-analyzer1.onrender.com
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
TARGETS = [
    ("US30", "1d"),
    ("XAUUSD", "1d"),
]


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
            print(f"  تلاش {attempt + 1}: {type(e).__name__} — {str(e)[:120]}")
        time.sleep(10 * (attempt + 1))
    return False


def main() -> int:
    if not SITE or not KEY:
        print("❌ SITE_URL یا SNAPSHOT_KEY تنظیم نشده")
        return 1

    print(f"سایت هدف: {SITE}")
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
    sent = push(items, built_at=time.time())
    print("✅ پوش موفق" if sent else "❌ پوش ناموفق")
    return 0 if sent else 1


if __name__ == "__main__":
    sys.exit(main())
