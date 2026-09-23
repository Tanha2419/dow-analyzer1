# -*- coding: utf-8 -*-
"""نقطه ورود سرور تولیدی (gunicorn) — برای رندر.

چرا این فایل جداست:
    server.py وقتی مستقیم اجرا شود بلوک __main__ را می خواند و
    app.run() فلسک را بالا می آورد. آن سرور توسعه است و برای
    اینترنت مناسب نیست. gunicorn به جایش شیء app را از اینجا
    برمی دارد.

نکته مهم درباره رندر رایگان:
    فایل سیستم موقتی است — journal.jsonl با هر ری استارت پاک
    می شود. پس اینجا autolog را روشن نمی کنیم؛ ثبت دائمی کار
    گیت هاب اکشنز است که نتیجه را در مخزن commit می کند.
    این سرور فقط نمایش می دهد.
"""
from __future__ import annotations

import os

from server import app  # noqa: F401  — gunicorn همین را می خواهد


def _maybe_start_autolog() -> None:
    """ثبت داخل سرور، فقط اگر صراحتا خواسته شده باشد.

    پیش فرض خاموش است چون روی رندر رایگان داده اش از بین می رود
    و آمار ناقص بدتر از نبود آمار است.
    """
    if os.environ.get("ENABLE_AUTOLOG", "").strip() not in ("1", "true", "yes"):
        return
    try:
        import autolog
        autolog.start()
        print("[+] ثبت خودکار داخل سرور فعال شد (ENABLE_AUTOLOG)")
    except Exception as e:
        print("[!] ثبت خودکار فعال نشد:", e)


def _warm_feed() -> None:
    """گرم کردن موتور قیمت زنده هنگام بالا آمدن."""
    try:
        import live_feed
        live_feed.get_feed()
        print("[+] موتور رصد زنده فعال شد")
    except Exception as e:
        print("[!] موتور رصد زنده بالا نیامد:", e)


_warm_feed()
_maybe_start_autolog()
