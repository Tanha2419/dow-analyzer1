# -*- coding: utf-8 -*-
"""ثبت خودکار دوره ای سیگنال ها — تغذیه کننده دفترچه.

چرا لازم است: کارنامه فقط وقتی پر می شود که کسی داشبورد را باز کند.
با این زمان بند، سرور خودش هر کندل یک تصمیم می گیرد و ثبت می کند،
حتی وقتی هیچ کس آنلاین نیست.

قواعد محافظه کارانه:
  • فقط وقتی بازار آن دارایی باز است (بازار بسته = قیمت ثابت = نمونه بی معنا)
  • یک بار در هر کندل — journal.record خودش کلید کندلی دارد
  • با تاخیر تصادفی کوچک تا همه دارایی ها همزمان درخواست نزنند
  • ارزیابی خودکار هر ساعت تا نتیجه سیگنال های گذشته سنجیده شود
  • نخ daemon: با بسته شدن سرور می میرد، چیزی معلق نمی ماند
"""
from __future__ import annotations

import random
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional

# چه چیزی ثبت شود
TARGETS = [("XAUUSD", "1h"), ("US30", "1h"),
           ("XAUUSD", "1d"), ("US30", "1d")]

CHECK_EVERY = 300.0        # هر ۵ دقیقه بررسی کن که کندل تازه ای باز شده یا نه
EVAL_EVERY = 3600.0        # هر ساعت نتیجه سیگنال های قدیمی را بسنج

_state: Dict = dict(
    running=False, started=None, last_error=None,
    recorded=0, skipped_closed=0, skipped_same=0,
    evaluated=0, last_run=None, last_eval=None, log=[])
_lock = threading.Lock()
_thread: Optional[threading.Thread] = None


def _note(msg: str):
    with _lock:
        _state["log"].append("%s %s" % (
            datetime.now(timezone.utc).strftime("%m-%d %H:%M"), msg))
        _state["log"] = _state["log"][-40:]


def _market_open(asset: str) -> bool:
    """آیا بازار این دارایی باز است؟"""
    try:
        import assets as A
        st = A.market_state(asset)
        if isinstance(st, dict):
            for k in ("is_open", "open", "opened"):
                if k in st:
                    return bool(st[k])
        return True
    except Exception:
        return True


def _tick():
    """یک دور: برای هر هدف، اگر بازار باز است تصمیم بگیر و ثبت کن."""
    import agent
    import journal as jr

    for asset, iv in TARGETS:
        try:
            if not _market_open(asset):
                with _lock:
                    _state["skipped_closed"] += 1
                continue

            out = agent.decide(iv, asset=asset, with_ml=False, with_mtf=False)
            sc = float((out.get("decision") or {}).get("score") or 0)
            if abs(sc) < 1.0:
                continue

            # agent.decide خودش record را صدا می زند؛ اینجا فقط
            # نتیجه را می شماریم تا وضعیت شفاف باشد
            rid = "%s-%s" % (asset, iv)
            with _lock:
                _state["recorded"] += 1
            _note("ثبت %s امتیاز %+.1f" % (rid, sc))

            time.sleep(random.uniform(1.5, 4.0))
        except Exception as e:
            with _lock:
                _state["last_error"] = "%s: %s" % (asset, str(e)[:120])
            _note("خطا %s — %s" % (asset, str(e)[:70]))

    with _lock:
        _state["last_run"] = datetime.now(timezone.utc).isoformat()


def _eval():
    try:
        import journal as jr
        r = jr.evaluate()
        n = int(r.get("newly_checked") or 0)
        with _lock:
            _state["evaluated"] += n
            _state["last_eval"] = datetime.now(timezone.utc).isoformat()
        if n:
            _note("ارزیابی %d سیگنال" % n)
    except Exception as e:
        _note("خطای ارزیابی — %s" % str(e)[:70])


def _loop():
    _note("زمان بند شروع شد")
    last_eval = 0.0
    # کمی صبر تا سرور کامل بالا بیاید
    time.sleep(20)
    while _state.get("running"):
        try:
            _tick()
            now = time.time()
            if now - last_eval >= EVAL_EVERY:
                _eval()
                last_eval = now
        except Exception:
            with _lock:
                _state["last_error"] = traceback.format_exc()[-200:]
        for _ in range(int(CHECK_EVERY)):
            if not _state.get("running"):
                break
            time.sleep(1)
    _note("زمان بند متوقف شد")


def start() -> Dict:
    global _thread
    if _state.get("running"):
        return dict(ok=True, already=True, **status())
    _state["running"] = True
    _state["started"] = datetime.now(timezone.utc).isoformat()
    _thread = threading.Thread(target=_loop, daemon=True,
                               name="autolog")
    _thread.start()
    return dict(ok=True, started=True)


def stop() -> Dict:
    """توقف زمان بند. تا ۲ ثانیه صبر می کند تا نخ واقعا خارج شود،
    وگرنه وضعیت گزارش شده با واقعیت نمی خواند."""
    _state["running"] = False
    t = _thread
    if t and t.is_alive():
        t.join(timeout=2.5)
    return dict(ok=True, stopped=True)


def status() -> Dict:
    with _lock:
        s = dict(_state)
    # alive فقط وقتی درست است که هم قصد اجرا باشد هم نخ زنده
    s["alive"] = bool(_state.get("running") and _thread
                      and _thread.is_alive())
    s["targets"] = ["%s %s" % (a, i) for a, i in TARGETS]
    s["check_every_min"] = round(CHECK_EVERY / 60, 1)
    try:
        import journal as jr
        js = jr.summary()
        s["journal"] = dict(total=js.get("total_recorded", 0),
                            evaluated=js.get("n", 0),
                            win_rate=js.get("win_rate"),
                            avg_r=js.get("avg_r"))
    except Exception:
        pass
    return s


if __name__ == "__main__":
    import json
    start()
    print("زمان بند فعال شد — Ctrl+C برای خروج")
    try:
        while True:
            time.sleep(30)
            print(json.dumps(status(), ensure_ascii=False, indent=2)[:700])
    except KeyboardInterrupt:
        stop()
