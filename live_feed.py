#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_feed.py — رصد لحظه ای قیمت بدون کلید API

دو موتور با جایگزینی خودکار:
  1) WebSocket یاهو (yfinance.WebSocket) — جریان push، بدون کلید
  2) نظرسنجی fast_info هر N ثانیه — پشتیبان پایدار

اگر WebSocket قطع شود یا بیش از STALE_SEC ساکت بماند، موتور
نظرسنجی خودکار فعال می شود و به محض برقراری دوباره، کنترل برمی گردد.
"""

from __future__ import annotations

import json
import threading
import time
import warnings
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

TEHRAN = timezone(timedelta(hours=3, minutes=30))

import assets as A

SYMBOLS = ["DIA"]          # نماد اصلی رصد
US30_SCALE = 100.0
POLL_SEC = 5.0             # فاصله نظرسنجی پشتیبان
STALE_SEC = 25.0           # سکوت بیش از این -> وبسوکت مرده فرض می شود
WS_RETRY_SEC = 20.0        # فاصله تلاش مجدد وبسوکت
TICK_BUFFER = 600          # تعداد تیک نگهداری شده در حافظه


def _now() -> float:
    return time.time()


def _teh(ts: Optional[float] = None) -> str:
    return datetime.fromtimestamp(ts or _now(), TEHRAN).strftime("%H:%M:%S")


# ---------- منبع لحظه ای: Yahoo chart API با پشتیبانی پیش/پس بازار ----------
CHART_URL = ("https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
             "?interval=1m&range=1d&includePrePost=true")
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def fetch_quote(sym: str) -> Optional[Dict]:
    """آخرین قیمت واقعی شامل جلسه پیش گشایش و پس از بازار.

    fast_info و regularMarketPrice خارج از جلسه عادی «بیات» می شوند و
    قیمت بسته شدن روز قبل را برمی گردانند. این تابع آخرین کندل یک دقیقه ای
    را که شامل prepost است می خواند و تازه ترین عدد را انتخاب می کند.
    """
    import urllib.request
    try:
        req = urllib.request.Request(CHART_URL.format(sym=sym), headers=_UA)
        raw = urllib.request.urlopen(req, timeout=8).read()
        res = json.loads(raw)["chart"]["result"][0]
        meta = res.get("meta", {})
        reg_px = meta.get("regularMarketPrice")
        reg_ts = meta.get("regularMarketTime") or 0

        # آخرین کندل معتبر (شامل پیش گشایش و پس از بازار)
        stamps = res.get("timestamp") or []
        closes = (res.get("indicators", {}).get("quote") or [{}])[0].get("close") or []
        bar_px, bar_ts = None, 0
        for t, c in zip(stamps, closes):
            if c is not None:
                bar_px, bar_ts = float(c), int(t)

        # تازه ترین را انتخاب کن
        if bar_px is not None and bar_ts > reg_ts:
            px, ts_used, sess = bar_px, bar_ts, "extended"
        elif reg_px:
            px, ts_used, sess = float(reg_px), reg_ts, "regular"
        elif bar_px is not None:
            px, ts_used, sess = bar_px, bar_ts, "extended"
        else:
            return None

        prev = meta.get("previousClose") or meta.get("chartPreviousClose")
        return dict(price=px, quote_ts=ts_used, session=sess,
                    prev_close=float(prev) if prev else None,
                    day_high=meta.get("regularMarketDayHigh"),
                    day_low=meta.get("regularMarketDayLow"),
                    volume=meta.get("regularMarketVolume"),
                    stale_sec=max(0, int(_now() - ts_used)))
    except Exception:
        return None


class LiveFeed:
    """موتور رصد زنده با دو منبع و جایگزینی خودکار."""

    def __init__(self, symbols: Optional[List[str]] = None,
                 asset: Optional[str] = None):
        self.profile = A.profile(asset)
        self.asset = self.profile["key"]
        # دارایی هایی که فید اسپات مستقیم دارند (طلا) از یاهو استفاده نمی کنند
        self.spot_mode = bool(self.profile.get("spot_source"))
        self.symbols = symbols or [self.profile["candle_symbol"]]
        self._lock = threading.RLock()
        self._state: Dict[str, Dict] = {}
        self._ticks: Dict[str, deque] = {s: deque(maxlen=TICK_BUFFER)
                                         for s in self.symbols}
        self._last_msg = 0.0
        self._last_ws = 0.0
        self._source = "starting"
        self._ws = None
        self._ws_ok = False
        self._ws_attempts = 0
        self._ws_msgs = 0
        self._poll_msgs = 0
        self._started = False
        self._stop = threading.Event()
        self._subs: List[deque] = []       # صف مشترکین SSE

    # ------------------------------------------------ عمومی
    def start(self) -> None:
        if self._started:
            return
        self._started = True
        if not self.spot_mode:
            threading.Thread(target=self._ws_loop, daemon=True,
                             name="live-ws").start()
            threading.Thread(target=self._poll_loop, daemon=True,
                             name="live-poll").start()
        else:
            # طلا: مستقیم از فید اسپات واقعی (سوییس کوت) خوانده می شود
            threading.Thread(target=self._spot_loop, daemon=True,
                             name="live-spot").start()

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass

    def snapshot(self, scale: bool = True) -> Dict:
        if not scale:
            k = 1.0
        elif self.spot_mode:
            k = 1.0          # قیمت اسپات از قبل واقعی است
        else:
            k = float(self.profile.get("display_scale", US30_SCALE))
        _dec = int(self.profile.get("decimals", 0))
        with self._lock:
            out = {}
            for s, v in self._state.items():
                d = dict(v)
                _r = _dec if k == 1 else max(1, _dec)
                for f in ("price", "day_high", "day_low", "open", "prev_close", "change"):
                    if d.get(f) is not None:
                        d[f] = round(d[f] * k, _r)
                d["spark"] = [round(p * k, _r)
                              for _, p in list(self._ticks[s])[-90:]]
                out[s] = d
            age = _now() - self._last_msg if self._last_msg else None
            return dict(
                symbols=out,
                source=self._source,
                live=bool(age is not None and age < STALE_SEC),
                age_sec=None if age is None else round(age, 1),
                scale=k,
                asset=self.asset,
                asset_name=self.profile["name"],
                emoji=self.profile["emoji"],
                unit=self.profile["unit"],
                decimals=_dec,
                spot_mode=self.spot_mode,
                display=(self.profile["key"] if scale
                         else self.profile["candle_symbol"]),
                ws_connected=self._ws_ok,
                ws_messages=self._ws_msgs,
                poll_messages=self._poll_msgs,
                ws_attempts=self._ws_attempts,
                server_time=_teh(),
                market=self._market_phase(),
            )

    def subscribe(self) -> deque:
        q: deque = deque(maxlen=50)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: deque) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    # ------------------------------------------------ داخلی
    def _market_phase(self) -> Dict:
        """فاز بازار — از منبع واحد market_hours (با DST و تعطیلات)."""
        if self.profile["hours"]["mode"] == "metal":
            try:
                st = A.market_state(self.asset)
                return dict(phase="regular" if st.get("is_open") else "closed",
                            label=("بازار طلا باز" if st.get("is_open")
                                   else str(st.get("reason"))),
                            open=bool(st.get("is_open")),
                            countdown=st.get("reason"),
                            nearly_24h=True,
                            session=dict(note=self.profile["hours"]["note"]),
                            holiday=None, half_day=False)
            except Exception as e:
                return dict(phase="unknown", label=str(e), open=False)
        try:
            import market_hours as mh
            s = mh.market_status()
            return dict(phase=s["phase"], label=s["label"], open=s["is_open"],
                        countdown=s["countdown"],
                        minutes_to_open=s["minutes_to_open"],
                        minutes_to_close=s["minutes_to_close"],
                        session=s["session"], dst=s["dst"],
                        next_open=s["next_open_tehran"],
                        holiday=s.get("holiday_name"),
                        half_day=s.get("is_half_day"))
        except Exception as e:
            return dict(phase="unknown", label=f"نامشخص ({type(e).__name__})",
                        open=False)

    def _publish(self, sym: str, price: float, src: str,
                 extra: Optional[Dict] = None) -> None:
        ts = _now()
        with self._lock:
            prev = self._state.get(sym, {})
            st = dict(prev)
            st.update(dict(symbol=sym, price=float(price), source=src,
                           ts=ts, time=_teh(ts)))
            if extra:
                for kk, vv in extra.items():
                    if vv is not None:
                        st[kk] = vv
            pc = st.get("prev_close")
            if pc:
                st["change"] = float(price) - pc
                st["change_pct"] = (float(price) / pc - 1) * 100
            # جهت تیک برای رنگ
            p0 = prev.get("price")
            st["tick_dir"] = 0 if p0 is None else (1 if price > p0 else
                                                   (-1 if price < p0 else 0))
            self._state[sym] = st
            self._ticks[sym].append((ts, float(price)))
            self._last_msg = ts
            self._source = src
            payload = json.dumps(dict(type="tick", symbol=sym,
                                      price=float(price), source=src,
                                      time=st["time"],
                                      change_pct=st.get("change_pct"),
                                      tick_dir=st["tick_dir"]),
                                 ensure_ascii=False)
            for q in self._subs:
                q.append(payload)

    # ---------- موتور ۱: وبسوکت ----------
    def _ws_loop(self) -> None:
        import yfinance as yf
        while not self._stop.is_set():
            self._ws_attempts += 1
            try:
                self._ws = yf.WebSocket()

                def on_msg(m: Dict) -> None:
                    try:
                        sym = m.get("id")
                        px = m.get("price")
                        if sym in self._ticks and px:
                            self._ws_ok = True
                            self._last_ws = _now()
                            self._ws_msgs += 1
                            self._publish(sym, float(px), "websocket", dict(
                                day_high=m.get("day_high"),
                                day_low=m.get("day_low"),
                                open=m.get("open_price"),
                                volume=(int(m["day_volume"])
                                        if m.get("day_volume") else None),
                                change_pct_raw=m.get("change_percent"),
                            ))
                    except Exception:
                        pass

                self._ws.subscribe(self.symbols)
                self._ws.listen(on_msg)          # مسدودکننده
            except Exception:
                pass
            self._ws_ok = False
            try:
                if self._ws:
                    self._ws.close()
            except Exception:
                pass
            self._stop.wait(WS_RETRY_SEC)

    # ---------- موتور ۲: نظرسنجی پشتیبان ----------
    def _spot_loop(self) -> None:
        """
        حلقه قیمت اسپات واقعی (طلا).

        از فید عمومی سوییس کوت می خواند که bid/ask واقعی می دهد و هر
        ثانیه تیک می خورد. اگر جواب نداد، به gold-api سوییچ می شود.
        مرجع روزانه (بسته دیروز / سقف / کف) از فیوچرز GC=F گرفته می شود
        چون فید اسپات فقط قیمت لحظه ای می دهد.
        """
        sym = self.symbols[0]
        ref: Dict = {}
        ref_ts = 0.0

        def _refresh_ref() -> Dict:
            try:
                q = fetch_quote(sym)
                if q:
                    return q
            except Exception:
                pass
            return {}

        while not self._stop.is_set():
            try:
                sp = A.live_spot(self.asset, ttl=0.0)
                if sp.get("ok"):
                    price = float(sp["price"])

                    # مرجع روزانه هر ۳ دقیقه یک بار از فیوچرز تازه می شود
                    if (_now() - ref_ts) > 180:
                        r = _refresh_ref()
                        if r:
                            b = A.basis(self.asset,
                                        futures_price=r.get("price"))
                            f = float(b.get("factor", 1.0)) if b.get("ok") else 1.0
                            ref = {k2: (v2 * f if isinstance(v2, (int, float))
                                        and k2 in ("prev_close", "day_high",
                                                   "day_low", "open") else v2)
                                   for k2, v2 in r.items()}
                            ref_ts = _now()

                    extra = dict(
                        bid=sp.get("bid"), ask=sp.get("ask"),
                        spread=sp.get("spread"),
                        feed=sp.get("source"),
                        prev_close=ref.get("prev_close"),
                        day_high=ref.get("day_high"),
                        day_low=ref.get("day_low"),
                        volume=ref.get("volume"),
                        session="spot", quote_age=0,
                    )
                    self._poll_msgs += 1
                    self._publish(sym, price,
                                  "spot:" + str(sp.get("source", "")), extra)
                    self._source = "spot"
            except Exception:
                pass
            self._stop.wait(1.5)


    def _poll_loop(self) -> None:
        import yfinance as yf
        tk = {s: yf.Ticker(s) for s in self.symbols}
        # مقدار اولیه برای پر شدن فوری کارت
        for s in self.symbols:
            q = fetch_quote(s)
            if q:
                self._publish(s, q["price"], "chart", dict(
                    prev_close=q["prev_close"], day_high=q["day_high"],
                    day_low=q["day_low"], volume=q["volume"],
                    session=q["session"], quote_age=q["stale_sec"]))

        while not self._stop.is_set():
            self._stop.wait(POLL_SEC)
            if self._stop.is_set():
                break
            # فقط قدمت پیام های وبسوکت مهم است، نه پیام های خود نظرسنجی
            ws_age = _now() - self._last_ws if self._last_ws else 1e9
            if self._ws_ok and ws_age < STALE_SEC:
                continue
            if ws_age >= STALE_SEC:
                self._ws_ok = False
            for s in self.symbols:
                q = fetch_quote(s)
                if q:
                    self._poll_msgs += 1
                    self._publish(s, q["price"], "chart", dict(
                        prev_close=q["prev_close"], day_high=q["day_high"],
                        day_low=q["day_low"], volume=q["volume"],
                        session=q["session"], quote_age=q["stale_sec"]))
                    continue
                # آخرین سنگر: fast_info (ممکن است خارج از جلسه بیات باشد)
                try:
                    fi = tk[s].fast_info
                    px = fi["lastPrice"]
                    if px:
                        self._poll_msgs += 1
                        self._publish(s, float(px), "poll", dict(
                            prev_close=float(fi.get("previousClose") or 0) or None,
                            day_high=fi.get("dayHigh"), day_low=fi.get("dayLow"),
                            open=fi.get("open"), volume=fi.get("lastVolume"),
                        ))
                except Exception:
                    continue


FEED = LiveFeed()

# هر دارایی فید مستقل خودش را دارد (داوجونز از یاهو، طلا از فید اسپات)
_FEEDS: Dict[str, "LiveFeed"] = {FEED.asset: FEED}
_FEEDS_LOCK = threading.RLock()


def get_feed(asset: Optional[str] = None) -> LiveFeed:
    key = A.resolve(asset)
    with _FEEDS_LOCK:
        f = _FEEDS.get(key)
        if f is None:
            f = LiveFeed(asset=key)
            _FEEDS[key] = f
    if not f._started:
        f.start()
    return f


if __name__ == "__main__":
    f = get_feed()
    print("رصد زنده آغاز شد — ۳۰ ثانیه نمونه گیری\n")
    for i in range(10):
        time.sleep(3)
        s = f.snapshot(scale=True)
        d = s["symbols"].get("DIA", {})
        print(f"[{s['server_time']}] {s['display']} "
              f"{d.get('price','—')} | منبع {s['source']:<10} "
              f"زنده={s['live']} قدمت={s['age_sec']}s "
              f"| ws={s['ws_messages']} poll={s['poll_messages']} "
              f"| {s['market']['label']}")
    f.stop()
