# داشبورد پول هوشمند داوجونز

تحلیل زندهٔ شاخص داوجونز (US30) و طلا (XAUUSD) با روش‌های Smart Money و ICT.

## تنظیمات رندر

| کادر | مقدار |
|---|---|
| Language | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `python -m gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120` |
| Instance Type | Free |

پایتون روی نسخهٔ ۳.۱۲.۷ قفل شده (`runtime.txt`).

## آدرس‌های مهم

| آدرس | کاربرد |
|---|---|
| `/` | داشبورد |
| `/api/ping` | تست سلامت (برای UptimeRobot) |
| `/api/files` | تشخیص عیب |
| `/api/agent?interval=1h&asset=US30` | تحلیل کامل |
| `/api/plan?interval=1h&asset=XAUUSD` | برگهٔ معامله |

## نکات

- بکتست در نسخهٔ وب غیرفعال است (کتابخانه‌اش روی پایتون جدید ساخته نمی‌شود).
- ثبت خودکار روی رندر خاموش است چون فایل‌سیستم رایگان موقتی است.
- سرویس رایگان بعد از ۱۵ دقیقه بی‌کاری می‌خوابد. با UptimeRobot روی `/api/ping` بیدار نگهش دارید — هرگز روی `/`.
