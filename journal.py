# -*- coding: utf-8 -*-
"""دفترچه ثبت سیگنال — پیش نیاز هر کالیبراسیون واقعی.

مشکلی که حل می کند: ممیزی ۲۰۲۶-۰۹-۱۹ نشان داد هیچ کدام از ۳۲ وزن
موتور هرگز کالیبره نشده اند. علتش ساده است — هیچ جا ثبت نمی شد که
سیگنال داده شده بعدا درست از آب درآمد یا نه.

این ماژول هر تصمیم را با قیمت و زمان ذخیره می کند، و بعدا نتیجه اش
را با داده واقعی بازار می سنجد. هیچ عدد حدسی تولید نمی کند: اگر
کندل کافی نباشد، وضعیت «در انتظار» می ماند.

فایل: journal.jsonl (هر خط یک رکورد JSON)
"""
from __future__ import annotations

import io
import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

BASE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(BASE, "journal.jsonl")

# چند کندل جلوتر را برای ارزیابی نگاه کنیم
HORIZON = {"5m": 12, "15m": 8, "30m": 6, "1h": 8, "1d": 5}

# آستانه موفقیت بر حسب ATR
WIN_ATR = 0.75


def _now():
    return datetime.now(timezone.utc)


# دقیقه هر کندل — برای ساخت کلید یکتا
_BUCKET_MIN = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "1d": 1440}


def _bucket(ts: datetime, interval: str) -> str:
    """کندل متعلق به این زمان. دو تصمیم در یک کندل = یک رکورد."""
    m = _BUCKET_MIN.get(interval, 60)
    if m >= 1440:
        return ts.strftime("%Y-%m-%d")
    total = ts.hour * 60 + ts.minute
    slot = (total // m) * m
    return "%s %02d:%02d" % (ts.strftime("%Y-%m-%d"), slot // 60, slot % 60)


def record(decision: Dict, asset: str, interval: str,
           price: Optional[float] = None) -> Dict:
    """یک تصمیم را ثبت می کند.

    ⚠️ یک رکورد به ازای هر کندل، نه هر فراخوان. اگر کاربر ۲۰ بار
    رفرش کند، ۲۰ نمونه وابسته ثبت نمی شود — رکورد موجود فقط
    به روز می شود. بدون این، نرخ برد بی معنا می شد.
    """
    try:
        dec = decision.get("decision") or {}
        score = float(dec.get("score") or 0)
        label = str(dec.get("label") or "")
        if price is None:
            price = (decision.get("meta") or {}).get("price")
        if price is None:
            return dict(ok=False, error="قیمت در دسترس نیست")

        gate = (decision.get("institutional") or {}).get("gate") or {}
        now = _now()
        bkt = _bucket(now, interval)
        rid = "%s-%s-%s" % (asset, interval, bkt.replace(" ", "T"))
        rec = dict(
            id=rid, bucket=bkt, updates=1,
            ts=now.isoformat(),
            asset=asset, interval=interval,
            price=float(price), score=score, label=label,
            conf=float(dec.get("confidence") or 0),
            grade=str(dec.get("grade") or ""),
            allowed=bool(gate.get("allowed", True)),
            parts={p.get("name"): p.get("weight")
                   for p in (decision.get("parts") or [])
                   if isinstance(p, dict) and p.get("weight")},
            outcome=None, checked=False)

        rows = _load()
        for i, r in enumerate(rows):
            if r.get("id") == rid:
                if r.get("checked"):
                    return dict(ok=True, id=rid, skipped="قبلا ارزیابی شده")
                rec["updates"] = int(r.get("updates", 1)) + 1
                rec["ts"] = r["ts"]          # زمان اولین تصمیم می ماند
                rec["price"] = r["price"]    # قیمت ورود اولیه می ماند
                rows[i] = rec
                _save(rows)
                return dict(ok=True, id=rid, updated=True)
        with io.open(PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return dict(ok=True, id=rid, created=True)
    except Exception as e:
        return dict(ok=False, error=str(e)[:150])


def _load() -> List[Dict]:
    if not os.path.exists(PATH):
        return []
    out = []
    with io.open(PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _save(rows: List[Dict]):
    with io.open(PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def evaluate(limit: int = 500) -> Dict:
    """نتیجه سیگنال های ثبت شده را با داده واقعی بازار می سنجد."""
    rows = _load()
    if not rows:
        return dict(ok=True, n=0, note="هنوز سیگنالی ثبت نشده")

    pending = [r for r in rows if not r.get("checked")][-limit:]
    if not pending:
        return summary()

    try:
        import pandas as pd
        import yfinance as yf
        import assets as A
    except Exception as e:
        return dict(ok=False, error="کتابخانه در دسترس نیست: %s" % e)

    cache = {}
    done = 0
    for r in pending:
        try:
            prof = A.profile(r["asset"])
            sym = prof["candle_symbol"]
            iv = r["interval"]
            key = (sym, iv)
            if key not in cache:
                per = {"5m": "1mo", "15m": "2mo", "30m": "2mo",
                       "1h": "6mo", "1d": "2y"}.get(iv, "3mo")
                cache[key] = yf.Ticker(sym).history(period=per, interval=iv)
            df = cache[key]
            if df is None or df.empty:
                continue

            t0 = pd.Timestamp(r["ts"])
            idx = df.index
            if getattr(idx, "tz", None) is not None:
                t0 = t0.tz_convert(idx.tz)
            else:
                t0 = t0.tz_localize(None)
            after = df[idx > t0]
            k = HORIZON.get(iv, 8)
            if len(after) < k:
                continue      # هنوز زود است — بدون حدس رها می شود

            w = after.iloc[:k]
            entry = float(r["price"])
            hi = float(w["High"].max())
            lo = float(w["Low"].min())
            close = float(w["Close"].iloc[-1])

            tr = (df["High"] - df["Low"]).rolling(14).mean()
            atr = float(tr[idx <= t0].iloc[-1]) if (idx <= t0).any() else None
            if not atr or atr <= 0:
                continue

            side = 1 if r["score"] > 0 else (-1 if r["score"] < 0 else 0)
            if side == 0:
                r["checked"] = True
                r["outcome"] = dict(result="خنثی", r_mult=0.0)
                done += 1
                continue

            move = (close - entry) * side
            best = (hi - entry) if side > 0 else (entry - lo)
            r_mult = move / atr
            hit = best >= WIN_ATR * atr
            r["checked"] = True
            r["outcome"] = dict(
                result=("موفق" if r_mult > 0 else "ناموفق"),
                touched_target=bool(hit),
                r_mult=round(r_mult, 3),
                move_pct=round((close - entry) / entry * 100, 3),
                bars=k, atr=round(atr, 4),
                checked_at=_now().isoformat())
            done += 1
        except Exception:
            continue

    if done:
        byid = {r["id"]: r for r in pending}
        for i, r in enumerate(rows):
            if r["id"] in byid:
                rows[i] = byid[r["id"]]
        _save(rows)
    out = summary()
    out["newly_checked"] = done
    return out


def summary(asset: Optional[str] = None) -> Dict:
    """کارنامه: نرخ برد، میانگین R، و اثر هر بخش امتیازدهی."""
    rows = [r for r in _load() if r.get("checked") and r.get("outcome")]
    if asset:
        rows = [r for r in rows if r["asset"] == asset]
    allrows = _load()
    total = len(allrows)
    # سلامت نمونه: چند بار هر کندل به روز شده و چه نسبتی مستقل است
    ups = [int(r.get("updates") or 1) for r in allrows]
    avg_up = round(sum(ups) / len(ups), 2) if ups else None
    indep = round(len(ups) / sum(ups), 3) if sum(ups) else None
    if not rows:
        return dict(ok=True, n=0, total_recorded=total,
                    avg_updates=avg_up, independence=indep,
                    note="هنوز سیگنال ارزیابی شده ای نیست — "
                         "هر سیگنال چند کندل زمان لازم دارد")

    rs = [r["outcome"]["r_mult"] for r in rows]
    wins = [x for x in rs if x > 0]
    by_asset = {}
    for r in rows:
        a = r["asset"]
        by_asset.setdefault(a, []).append(r["outcome"]["r_mult"])

    # اثر هر بخش: میانگین R وقتی آن بخش فعال و هم جهت بوده
    contrib = {}
    for r in rows:
        rm = r["outcome"]["r_mult"]
        for name, w in (r.get("parts") or {}).items():
            try:
                w = float(w)
            except Exception:
                continue
            if abs(w) < 0.01:
                continue
            agree = (w > 0) == (r["score"] > 0)
            d = contrib.setdefault(name, dict(n=0, r_sum=0.0, agree=0))
            d["n"] += 1
            d["r_sum"] += rm
            d["agree"] += 1 if agree else 0
    parts_stat = sorted(
        [dict(name=k, n=v["n"], avg_r=round(v["r_sum"] / v["n"], 3),
              agree_pct=round(100.0 * v["agree"] / v["n"], 1))
         for k, v in contrib.items() if v["n"] >= 5],
        key=lambda x: -abs(x["avg_r"]))

    return dict(
        ok=True, n=len(rows), total_recorded=total,
        avg_updates=avg_up, independence=indep,
        win_rate=round(100.0 * len(wins) / len(rows), 1),
        avg_r=round(sum(rs) / len(rs), 3),
        median_r=round(sorted(rs)[len(rs) // 2], 3),
        best=round(max(rs), 3), worst=round(min(rs), 3),
        by_asset={k: dict(n=len(v), avg_r=round(sum(v) / len(v), 3),
                          win_rate=round(100.0 * len([x for x in v if x > 0]) / len(v), 1))
                  for k, v in by_asset.items()},
        parts=parts_stat[:15],
        source_tier="real",
        note="بر پایه %d سیگنال ارزیابی شده از %d ثبت شده" % (len(rows), total))


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "summary"
    if cmd == "eval":
        print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
    else:
        s = summary()
        print("=" * 58)
        print("کارنامه سیگنال ها")
        print("=" * 58)
        if not s.get("n"):
            print(" ", s.get("note"))
        else:
            print("  تعداد ارزیابی شده : %d از %d" % (s["n"], s["total_recorded"]))
            print("  نرخ برد           : %.1f٪" % s["win_rate"])
            print("  میانگین R         : %+.3f" % s["avg_r"])
            print("  بهترین / بدترین   : %+.2f / %+.2f" % (s["best"], s["worst"]))
            for a, v in s["by_asset"].items():
                print("   %-8s n=%-4d برد %.0f٪  R %+.3f"
                      % (a, v["n"], v["win_rate"], v["avg_r"]))
            if s["parts"]:
                print("\n  اثر هر بخش (حداقل ۵ نمونه):")
                for p in s["parts"]:
                    print("   %-32s n=%-4d R %+.3f  هم جهت %.0f٪"
                          % (p["name"][:32], p["n"], p["avg_r"], p["agree_pct"]))
