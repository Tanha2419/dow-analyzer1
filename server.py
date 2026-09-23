#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
server.py — وب سرور داشبورد پول هوشمند داوجونز
اجرا:  .venv/bin/python server.py  [--port 8080]
"""

from __future__ import annotations

import argparse
import base64
import os
import traceback
import warnings

import json as _json
import time as _time
from flask import Flask, Response, jsonify, request, send_from_directory

import web_api
import agent as agent_mod
import live_feed
import macro_data
import orderflow
import volatility as volx
import assets as assets_mod

warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder="static", static_url_path="")

# نسخهٔ جاسازی شدهٔ ظاهر برنامه (پشتیبان). اگر این فایل نبود، اشکالی ندارد.
try:
    import embedded_ui as _embedded
except Exception:
    _embedded = None

_MISSING_STATIC_HTML = """<!doctype html>
<html lang="fa" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>پوشه static آپلود نشده</title></head>
<body style="font-family:Tahoma,sans-serif;background:#0f1117;color:#e8e8ea;
max-width:680px;margin:40px auto;padding:0 20px;line-height:2">
<h1 style="color:#ffb020">&#9888;&#65039; پوشهٔ static آپلود نشده</h1>
<p>سرور سالم است و کار می کند &mdash; همهٔ روت های API پاسخ می دهند.
فقط دو فایل رابط کاربری روی سرور نیستند:</p>
<pre style="background:#1a1d27;padding:14px;border-radius:8px;direction:ltr;
text-align:left">static/index.html
static/edu.js</pre>
<h3>راه حل</h3>
<p>در گیت هاب، پوشهٔ <code>static</code> را جداگانه بسازید و دو فایل بالا
را داخلش آپلود کنید. جزئیات کامل در فایل
<code>رفع-static.md</code> داخل بستهٔ پروژه است.</p>
<p style="color:#8b8f9a">برای دیدن فهرست فایل های موجود روی سرور:
<a href="/api/files" style="color:#4ea1ff">/api/files</a></p>
</body></html>"""


@app.after_request
def headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Cache-Control"] = "no-store"
    # اجازه نمایش داخل iframe پیش نمایش
    resp.headers.pop("X-Frame-Options", None)
    resp.headers["Content-Security-Policy"] = "frame-ancestors *"
    return resp


def _asset() -> str:
    """کلید دارایی از پارامتر ?asset= (پیش فرض داوجونز)."""
    return assets_mod.resolve(request.args.get("asset"))


@app.route("/api/assets")
def api_assets():
    """فهرست دارایی های پشتیبانی شده برای سوییچ داشبورد."""
    try:
        out = []
        for a in assets_mod.list_assets():
            p = assets_mod.profile(a["key"])
            out.append(dict(
                a,
                candle_symbol=p["candle_symbol"],
                options_symbol=p["options_symbol"],
                nearly_24h=p["hours"]["nearly_24h"],
                hours_note=p["hours"]["note"],
                decimals=p["decimals"],
                intervals=p["intervals"],
            ))
        return jsonify(ok=True, data=dict(assets=out,
                                          default=assets_mod.DEFAULT_ASSET))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.route("/api/spot")
def api_spot():
    """قیمت اسپات زنده + بیسیس (فقط برای دارایی هایی که فید اسپات دارند)."""
    try:
        a = _asset()
        sp = assets_mod.live_spot(a)
        bs = assets_mod.basis(a)
        return jsonify(ok=True, data=dict(spot=sp, basis=bs, asset=a))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


def _serve_ui(name, mime):
    """فایل ظاهر برنامه را از سه جا می گردد، به این ترتیب:
      ۱. پوشهٔ static/  (حالت عادی پروژه)
      ۲. کنار خود server.py  (اگر کاربر بدون پوشه آپلود کرده باشد)
      ۳. نسخهٔ جاسازی شده داخل همین فایل  (تک فایلی)
    با این کار استقرار حتی اگر هیچ فایل جانبی آپلود نشود هم کار می کند."""
    for folder in ("static", ""):
        d = os.path.join(app.root_path, folder) if folder else app.root_path
        if os.path.exists(os.path.join(d, name)):
            return send_from_directory(d, name, mimetype=mime)
    if _embedded is not None:
        blob = _embedded.get(name)
        if blob:
            return Response(blob, mimetype=mime)
    return None


def _ui_source(name):
    """فقط برای گزارش تشخیصی: این فایل از کجا سرو می شود."""
    for folder in ("static", ""):
        d = os.path.join(app.root_path, folder) if folder else app.root_path
        if os.path.exists(os.path.join(d, name)):
            return "static/" if folder else "ریشه"
    if _embedded is not None and _embedded.get(name):
        return "جاسازی شده"
    return "یافت نشد"


@app.route("/")
def index():
    """صفحهٔ اصلی."""
    r = _serve_ui("index.html", "text/html")
    if r is None:
        return Response(_MISSING_STATIC_HTML, mimetype="text/html"), 503
    return r


@app.route("/edu.js")
def edu_js():
    r = _serve_ui("edu.js", "application/javascript")
    if r is None:
        return Response("// edu.js یافت نشد", mimetype="application/javascript"), 404
    return r


@app.route("/api/files")
def api_files():
    """تشخیصی: چه فایل هایی واقعاً روی سرور هستند."""
    root = app.root_path
    sdir = os.path.join(root, "static")
    try:
        top = sorted(os.listdir(root))
    except Exception as e:
        top = ["<error: %s>" % e]
    try:
        stat_files = sorted(os.listdir(sdir))
    except Exception:
        stat_files = []
    return jsonify(
        ok=True,
        root_path=root,
        cwd=os.getcwd(),
        static_dir=sdir,
        static_exists=os.path.isdir(sdir),
        static_files=stat_files,
        index_html=os.path.exists(os.path.join(sdir, "index.html")),
        edu_js=os.path.exists(os.path.join(sdir, "edu.js")),
        embedded_available=_embedded is not None,
        index_source=_ui_source("index.html"),
        edu_source=_ui_source("edu.js"),
        py_count=sum(1 for f in top if f.endswith(".py")),
        top_level=top,
    )


@app.route("/api/analysis")
def api_analysis():
    interval = request.args.get("interval", "1d")
    if interval not in web_api.INTERVAL_PERIODS:
        return jsonify(ok=False, error="تایم فریم نامعتبر"), 400
    bars = max(60, min(int(request.args.get("bars", 160)), 400))
    coalition = request.args.get("coalition", "1") != "0"
    force = request.args.get("force", "0") == "1"
    scale = request.args.get("scale", "1") != "0"
    try:
        data = web_api.cached_payload(interval, bars, coalition, force, scale,
                                      asset=_asset())
        return jsonify(ok=True, data=data)
    except Exception as e:
        traceback.print_exc()
        return jsonify(ok=False, error=str(e)), 500


@app.route("/api/backtest")
def api_backtest():
    interval = request.args.get("interval", "1d")
    if interval not in web_api.INTERVAL_PERIODS:
        return jsonify(ok=False, error="تایم فریم نامعتبر"), 400
    cash = float(request.args.get("cash", 100000))
    long_only = request.args.get("long_only", "0") == "1"
    regime = request.args.get("regime", "1") != "0"
    scale = request.args.get("scale", "1") != "0"
    try:
        return jsonify(ok=True, data=web_api.backtest_payload(
            interval, cash, long_only, regime, scale, asset=_asset()))
    except Exception as e:
        traceback.print_exc()
        return jsonify(ok=False, error=str(e)), 500


@app.route("/api/agent")
def api_agent():
    interval = request.args.get("interval", "1h")
    if interval not in web_api.INTERVAL_PERIODS:
        return jsonify(ok=False, error="تایم فریم نامعتبر"), 400
    equity = float(request.args.get("equity", 100000))
    scale = request.args.get("scale", "1") != "0"
    with_ml = request.args.get("ml", "1") != "0"
    with_mtf = request.args.get("mtf", "1") != "0"
    log = request.args.get("log", "0") == "1"
    try:
        d = agent_mod.decide(interval=interval, equity=equity, with_ml=with_ml,
                             with_mtf=with_mtf, with_coalition=True, scale=scale,
                             asset=_asset())
        if log:
            agent_mod.journal_add(agent_mod.journal_from_decision(d))
        d.pop("intelligence", None)
        return jsonify(ok=True, data=web_api._clean(d))
    except Exception as e:
        traceback.print_exc()
        return jsonify(ok=False, error=str(e)), 500


@app.route("/api/journal")
def api_journal():
    """کارنامه سیگنال ها.

    what=eval    نتیجه واقعی سیگنال های گذشته (journal.py)
    what=summary خلاصه بدون ارزیابی مجدد
    what=dist    توزیع سشن/رژیم/روز هفته (ژورنال قدیمی agent)
    """
    what = (request.args.get("what") or "eval").lower()
    asset = request.args.get("asset") or None
    try:
        import journal as jr
        if what == "dist":
            return jsonify(ok=True, data=agent_mod.journal_stats())
        if what == "summary":
            return jsonify(ok=True, data=jr.summary(asset))
        out = jr.evaluate()
        # توزیع ها را هم ضمیمه کن تا کارت یک جا همه را داشته باشد
        try:
            d = agent_mod.journal_stats()
            if d.get("ok"):
                for k in ("by_weekday", "by_gate", "by_grade",
                          "by_direction", "top_tags", "avg_conf",
                          "avg_score", "first", "last"):
                    if k in d:
                        out.setdefault(k, d[k])
        except Exception:
            pass
        return jsonify(ok=True, data=out)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.route("/api/layers")
def api_layers():
    """پنج لایه نهادی + بردار ویژگی."""
    interval = request.args.get("interval", "1h")
    scale = request.args.get("scale", "1") != "0"
    try:
        d = agent_mod.decide(interval=interval, with_ml=False, with_mtf=False,
                             with_coalition=False, scale=scale, asset=_asset())
        out = d.get("institutional", {})
        out["_decision"] = d["decision"]
        out["_meta"] = d["meta"]
        return jsonify(ok=True, data=web_api._clean(out))
    except Exception as e:
        traceback.print_exc()
        return jsonify(ok=False, error=str(e)), 500


@app.route("/api/macro")
def api_macro():
    try:
        return jsonify(ok=True, data=web_api._clean(macro_data.build_macro()))
    except Exception as e:
        traceback.print_exc()
        return jsonify(ok=False, error=str(e)), 500


@app.route("/api/volatility")
def api_vol():
    try:
        a = _asset()
        p = assets_mod.profile(a)
        return jsonify(ok=True, data=web_api._clean(
            volx.build_volatility(p["candle_symbol"], asset=a)))
    except Exception as e:
        traceback.print_exc()
        return jsonify(ok=False, error=str(e)), 500


@app.route("/api/orderflow")
def api_flow():
    iv = request.args.get("interval", "1h")
    try:
        p = assets_mod.profile(_asset())
        d = orderflow.build_orderflow(p["candle_symbol"], iv,
                                      with_seasonality=True)
        if d.get("delta", {}).get("series"):
            d["delta"]["series"] = {k: v[-90:] for k, v in d["delta"]["series"].items()}
        return jsonify(ok=True, data=web_api._clean(d))
    except Exception as e:
        traceback.print_exc()
        return jsonify(ok=False, error=str(e)), 500


@app.route("/api/live")
def api_live():
    """عکس فوری قیمت زنده."""
    scale = request.args.get("scale", "1") != "0"
    try:
        return jsonify(ok=True,
                       data=live_feed.get_feed(_asset()).snapshot(scale=scale))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.route("/api/live/stream")
def api_live_stream():
    """جریان Server-Sent Events — قیمت به محض تغییر push می شود."""
    scale = request.args.get("scale", "1") != "0"
    feed = live_feed.get_feed(_asset())

    def gen():
        q = feed.subscribe()
        try:
            snap = feed.snapshot(scale=scale)
            yield f"event: snapshot\ndata: {_json.dumps(snap, ensure_ascii=False)}\n\n"
            last_beat = _time.time()
            while True:
                sent = False
                while q:
                    q.popleft()
                    sent = True
                if sent:
                    snap = feed.snapshot(scale=scale)
                    yield f"event: tick\ndata: {_json.dumps(snap, ensure_ascii=False)}\n\n"
                    last_beat = _time.time()
                elif _time.time() - last_beat > 12:
                    snap = feed.snapshot(scale=scale)
                    yield f"event: heartbeat\ndata: {_json.dumps(snap, ensure_ascii=False)}\n\n"
                    last_beat = _time.time()
                _time.sleep(0.7)
        except GeneratorExit:
            pass
        finally:
            feed.unsubscribe(q)

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no",
                             "Connection": "keep-alive"})


@app.route("/api/real")
def api_real():
    """منابع داده واقعی — تقویم، خزانه داری، Put/Call، اخبار، صدک VIX."""
    try:
        import real_data
        what = request.args.get("what", "all")
        if what == "calendar":
            return jsonify(ok=True, data=real_data.economic_calendar(30))
        if what == "vix":
            import yfinance as yf
            v = yf.Ticker("^VIX").history(period="5d")["Close"].dropna()
            return jsonify(ok=True,
                           data=real_data.vix_percentile(float(v.iloc[-1])))
        if what == "treasury":
            return jsonify(ok=True, data=real_data.treasury_yields())
        if what == "putcall":
            _p = assets_mod.profile(_asset())
            return jsonify(ok=True,
                           data=real_data.put_call_real(_p["options_symbol"]))
        if what == "news":
            return jsonify(ok=True, data=real_data.finnhub_news(limit=40))
        if what == "earnings":
            return jsonify(ok=True, data=real_data.finnhub_earnings(45))
        _a = _asset()
        _p = assets_mod.profile(_a)
        return jsonify(ok=True, data=real_data.build_real_data(
            _p["candle_symbol"], asset=_a))
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 500


@app.route("/api/plan")
def api_plan():
    """برگه طرح معامله — ورود، حد ضرر، حد سود، و دلیل هر کدام."""
    try:
        import tradeplan
        a = _asset()
        iv = request.args.get("interval", "1h")
        if iv not in web_api.INTERVAL_PERIODS:
            return jsonify(ok=False, error="تایم فریم نامعتبر"), 400
        equity = float(request.args.get("equity", 100000))
        risk = float(request.args.get("risk", 1.0))
        multi = request.args.get("multi", "0") == "1"
        if multi:
            ivs = [x for x in request.args.get("intervals", "1h,1d").split(",")
                   if x in web_api.INTERVAL_PERIODS]
            d = tradeplan.build_multi(a, ivs or ["1h", "1d"], equity, risk)
        else:
            d = tradeplan.build_plan(a, iv, equity, risk)
        return jsonify(ok=True, data=web_api._clean(d))
    except Exception as e:
        traceback.print_exc()
        return jsonify(ok=False, error=str(e)), 500


@app.route("/api/plan.html")
def api_plan_html():
    """همان برگه به صورت فایل HTML مستقل و قابل ذخیره."""
    try:
        import tradeplan
        a = _asset()
        iv = request.args.get("interval", "1h")
        equity = float(request.args.get("equity", 100000))
        risk = float(request.args.get("risk", 1.0))
        multi = request.args.get("multi", "0") == "1"
        if multi:
            ivs = [x for x in request.args.get("intervals", "1h,1d").split(",")
                   if x in web_api.INTERVAL_PERIODS]
            d = tradeplan.build_multi(a, ivs or ["1h", "1d"], equity, risk)
            htm = tradeplan.to_html(d, multi=True)
        else:
            d = tradeplan.build_plan(a, iv, equity, risk)
            htm = tradeplan.to_html(d)
        dl = request.args.get("download", "0") == "1"
        hdrs = {"Content-Type": "text/html; charset=utf-8"}
        if dl:
            hdrs["Content-Disposition"] = (
                f'attachment; filename="plan_{a}_{iv}.html"')
        return Response(htm, headers=hdrs)
    except Exception as e:
        traceback.print_exc()
        return Response(f"<p dir=rtl>خطا: {e}</p>", status=500,
                        headers={"Content-Type": "text/html; charset=utf-8"})


@app.route("/api/plan.txt")
def api_plan_txt():
    """نسخه متنی برگه — برای کپی در تلگرام یا یادداشت."""
    try:
        import tradeplan
        a = _asset()
        iv = request.args.get("interval", "1h")
        equity = float(request.args.get("equity", 100000))
        risk = float(request.args.get("risk", 1.0))
        d = tradeplan.build_plan(a, iv, equity, risk)
        return Response(tradeplan.to_text(d),
                        headers={"Content-Type": "text/plain; charset=utf-8"})
    except Exception as e:
        return Response(f"خطا: {e}", status=500,
                        headers={"Content-Type": "text/plain; charset=utf-8"})


@app.route("/api/sentiment")
def api_sentiment():
    """سه منبع بیرونی: COT، ترس و طمع CNN، StockTwits."""
    try:
        import sentiment_ext
        what = request.args.get("what", "all")
        a = _asset()
        if what == "cot":
            return jsonify(ok=True, data=sentiment_ext.cot_report("XAUUSD"))
        if what == "fg":
            return jsonify(ok=True, data=sentiment_ext.fear_greed())
        if what == "stocktwits":
            return jsonify(ok=True, data=sentiment_ext.stocktwits(a))
        return jsonify(ok=True,
                       data=web_api._clean(sentiment_ext.build_sentiment_ext(a)))
    except Exception as e:
        traceback.print_exc()
        return jsonify(ok=False, error=str(e)), 500


@app.route("/api/econ")
def api_econ():
    """تقویم اقتصادی با اعداد واقعی: actual / forecast / previous."""
    import econ_actual as ea
    what = (request.args.get("what") or "all").lower()
    try:
        if what == "surprise":
            out = ea.surprise_score()
        elif what == "critical":
            out = ea.critical_events(
                float(request.args.get("hours") or 72))
        elif what == "ff":
            out = ea.forexfactory(request.args.get("week") or "thisweek")
        elif what == "enriched":
            import real_data
            out = ea.enrich_calendar(real_data.economic_calendar(14))
        else:
            out = ea.calendar_actuals()
            out["surprise_score"] = ea.surprise_score()
            out["critical"] = ea.critical_events(72)
        return jsonify(web_api._clean(out))
    except Exception as e:
        return jsonify(dict(ok=False, error=str(e)[:200])), 200


@app.route("/api/autolog")
def api_autolog():
    """وضعیت و کنترل ثبت خودکار دوره ای."""
    import autolog
    act = (request.args.get("action") or "status").lower()
    try:
        if act == "start":
            autolog.start()
        elif act == "stop":
            autolog.stop()
        return jsonify(ok=True, data=web_api._clean(autolog.status()))
    except Exception as e:
        return jsonify(ok=False, error=str(e)[:200]), 200


@app.route("/api/ping")
def api_ping():
    """نقطه بیدارباش برای سرویس های پایش (UptimeRobot و مانند آن).

    عمدا سبک است: هیچ داده بازاری نمی خواند، هیچ درخواست بیرونی
    نمی زند. فقط ثابت می کند پردازه زنده است. اگر مانیتور را به
    روتی وصل کنید که یاهو را صدا می زند، هر ۵ دقیقه یک درخواست
    بی دلیل می رود و ممکن است نرخ محدود شوید.
    """
    import time as _t
    return jsonify(ok=True, pong=True, ts=int(_t.time()))


@app.route("/api/health")
def health():
    return jsonify(ok=True, service="dow-smart-money", version="2.1",
                   assets=[a["key"] for a in assets_mod.list_assets()])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    live_feed.get_feed()
    print(f"[+] داشبورد پول هوشمند روی http://{a.host}:{a.port} اجرا شد")
    print("[+] موتور رصد زنده فعال شد (WebSocket + نظرسنجی پشتیبان)")
    try:
        import autolog
        autolog.start()
        print("[+] ثبت خودکار سیگنال فعال شد (هر ۵ دقیقه بررسی، "
              "یک نمونه در هر کندل)")
    except Exception as _e:
        print("[!] ثبت خودکار فعال نشد:", _e)
    app.run(host=a.host, port=a.port, debug=False, threaded=True)
