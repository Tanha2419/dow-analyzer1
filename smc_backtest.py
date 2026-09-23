#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smc_backtest.py — بکتست استراتژی پول هوشمند (دوطرفه: لانگ و شورت)
منطق: شکار نقدینگی -> بازپس گیری -> تایید ساختاری/جابجایی -> ورود با تایید دلتا
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from tabulate import tabulate

from smart_money import causal_smc_signals

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


class SmartMoneyStrategy(Strategy):
    """
    ورود : سیگنال علّی SMC (Sweep + Reclaim + تایید + دلتا)
    خروج : حد ضرر ساختاری (زیر/بالای نقطه شکار) و اهداف مضربی از ریسک
    مدیریت: خروج پله ای در TP1 و انتقال حد ضرر به سربه سر (Break-Even)
    """
    rr_tp1 = 2.0          # هدف اول به صورت مضرب ریسک (بهینه شده)
    rr_tp2 = 3.5          # هدف دوم (بهینه شده)
    partial = 0.5         # درصد خروج در هدف اول
    risk_pct = 0.02       # ریسک هر معامله از کل سرمایه
    allow_short = True
    max_bars = 60         # حداکثر مدت نگهداری (کندل، بهینه شده)
    use_be = True         # انتقال حد ضرر به سربه سر بعد از TP1

    def init(self):
        d = self.data.df
        self.long_sig = self.I(lambda v: v, d["SMC_Long"].values, name="Long", overlay=False)
        self.short_sig = self.I(lambda v: v, d["SMC_Short"].values, name="Short", overlay=False)
        self.sl_l = self.I(lambda v: v, d["SMC_SL_Long"].values, name="SLL", overlay=False)
        self.sl_s = self.I(lambda v: v, d["SMC_SL_Short"].values, name="SLS", overlay=False)
        self.atr = self.I(lambda v: v, d["SMC_ATR"].values, name="ATR", overlay=False)
        self._bar_in = None
        self._tp1_hit = False
        self._entry = None
        self._risk = None

    def next(self):
        price = float(self.data.Close[-1])
        i = len(self.data) - 1

        # ---------------- مدیریت موقعیت باز ----------------
        if self.position:
            if self._bar_in is not None and (i - self._bar_in) >= self.max_bars:
                self.position.close()
                self._reset()
                return
            if self._entry is not None and self._risk and self._risk > 0:
                if self.position.is_long:
                    prog = (price - self._entry) / self._risk
                    if not self._tp1_hit and prog >= self.rr_tp1:
                        self.position.close(portion=self.partial)
                        self._tp1_hit = True
                        if self.use_be:
                            for t in self.trades:
                                t.sl = max(t.sl or -np.inf, self._entry)
                    elif self._tp1_hit and prog >= self.rr_tp2:
                        self.position.close()
                        self._reset()
                        return
                else:
                    prog = (self._entry - price) / self._risk
                    if not self._tp1_hit and prog >= self.rr_tp1:
                        self.position.close(portion=self.partial)
                        self._tp1_hit = True
                        if self.use_be:
                            for t in self.trades:
                                t.sl = min(t.sl or np.inf, self._entry)
                    elif self._tp1_hit and prog >= self.rr_tp2:
                        self.position.close()
                        self._reset()
                        return
            return

        # ---------------- ورود جدید ----------------
        atr = float(self.atr[-1])
        if not np.isfinite(atr) or atr <= 0:
            return

        if self.long_sig[-1] == 1:
            sl = float(self.sl_l[-1])
            if not np.isfinite(sl) or sl >= price:
                sl = price - 1.2 * atr
            risk = price - sl
            if risk <= 0 or risk / price > 0.06:
                return
            size = self._size(risk, price)
            if size > 0:
                self.buy(size=size, sl=sl)
                self._open(i, price, risk)

        elif self.allow_short and self.short_sig[-1] == 1:
            sl = float(self.sl_s[-1])
            if not np.isfinite(sl) or sl <= price:
                sl = price + 1.2 * atr
            risk = sl - price
            if risk <= 0 or risk / price > 0.06:
                return
            size = self._size(risk, price)
            if size > 0:
                self.sell(size=size, sl=sl)
                self._open(i, price, risk)

    # ---------------- کمکی ----------------
    def _size(self, risk: float, price: float) -> int:
        cap = self.equity * self.risk_pct
        units = int(cap / risk)
        max_units = int(self.equity * 0.95 / price)
        return max(0, min(units, max_units))

    def _open(self, i, price, risk):
        self._bar_in = i
        self._entry = price
        self._risk = risk
        self._tp1_hit = False

    def _reset(self):
        self._bar_in = None
        self._entry = None
        self._risk = None
        self._tp1_hit = False


class SmartMoneyLongOnly(SmartMoneyStrategy):
    allow_short = False


# ============================================================================
def run_smc_backtest(df: pd.DataFrame, cash: float = 100_000,
                     commission: float = 0.001, interval: str = "1d",
                     long_only: bool = False, regime_filter: bool = True) -> dict:
    print("\n" + "=" * 74)
    print("   بکتست استراتژی پول هوشمند (Smart Money / ICT)")
    print("=" * 74)

    sig = causal_smc_signals(df, regime_filter=regime_filter)
    cols = ["Open", "High", "Low", "Close", "Volume", "SMC_Long", "SMC_Short",
            "SMC_SL_Long", "SMC_SL_Short", "SMC_ATR"]
    bt_df = sig[cols].copy()
    bt_df[["SMC_SL_Long", "SMC_SL_Short"]] = bt_df[["SMC_SL_Long", "SMC_SL_Short"]].fillna(0.0)
    bt_df = bt_df.dropna()

    nl = int(bt_df["SMC_Long"].sum())
    ns = int(bt_df["SMC_Short"].sum())
    mode = "فقط خرید (لانگ)" if long_only else "دوطرفه (لانگ + شورت)"
    reg = "فعال (فقط هم جهت با روند بلندمدت)" if regime_filter else "غیرفعال"
    print(f"   نماد: DIA  |  تایم فریم: {interval}  |  حالت: {mode}")
    print(f"   فیلتر رژیم بازار: {reg}")
    print(f"   تعداد کندل: {len(bt_df)}  |  سیگنال لانگ: {nl}  |  سیگنال شورت: {ns}")
    if nl + ns == 0:
        print("   [!] سیگنالی برای بکتست یافت نشد.")
        return {}

    strat = SmartMoneyLongOnly if long_only else SmartMoneyStrategy
    bt = Backtest(bt_df, strat, cash=cash, commission=commission,
                  margin=1.0, trade_on_close=True, exclusive_orders=True,
                  finalize_trades=True)
    stats = bt.run()
    eq = stats["_equity_curve"]
    tr = stats["_trades"]

    def f(v, kind="p"):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "N/A"
        if kind == "m":
            return f"${v:,.0f}"
        if kind == "p":
            return f"{v:+,.2f}%"
        if kind == "n":
            return f"{v:.2f}"
        return f"{int(v)}"

    rows = [
        ["بازه بکتست", f"{eq.index[0].date()}  تا  {eq.index[-1].date()}"],
        ["سرمایه اولیه", f(cash, "m")],
        ["سرمایه نهایی", f(stats.get("Equity Final [$]"), "m")],
        ["بازده کل", f(stats.get("Return [%]"))],
        ["بازده خرید و نگهداری", f(stats.get("Buy & Hold Return [%]"))],
        ["بازده سالانه (CAGR)", f(stats.get("Return (Ann.) [%]"))],
        ["نوسان سالانه", f(stats.get("Volatility (Ann.) [%]"))],
        ["حداکثر افت سرمایه", f(stats.get("Max. Drawdown [%]"))],
        ["میانگین افت سرمایه", f(stats.get("Avg. Drawdown [%]"))],
        ["تعداد معاملات", f(stats.get("# Trades"), "i")],
        ["نرخ برد", f(stats.get("Win Rate [%]"))],
        ["میانگین معامله", f(stats.get("Avg. Trade [%]"))],
        ["بهترین معامله", f(stats.get("Best Trade [%]"))],
        ["بدترین معامله", f(stats.get("Worst Trade [%]"))],
        ["فاکتور سود", f(stats.get("Profit Factor"), "n")],
        ["نسبت شارپ", f(stats.get("Sharpe Ratio"), "n")],
        ["نسبت سورتینو", f(stats.get("Sortino Ratio"), "n")],
        ["نسبت کالمار", f(stats.get("Calmar Ratio"), "n")],
        ["نسبت SQN", f(stats.get("SQN"), "n")],
        ["بیشترین زمان در بازار", f(stats.get("Exposure Time [%]"))],
    ]
    print("\n" + tabulate(rows, headers=["شاخص عملکرد", "مقدار"], tablefmt="fancy_grid"))

    # تفکیک لانگ / شورت
    if tr is not None and len(tr):
        t = tr.copy()
        t["side"] = np.where(t["Size"] > 0, "لانگ", "شورت")
        agg = []
        for side, g in t.groupby("side"):
            wins = (g["PnL"] > 0).sum()
            gp = g.loc[g["PnL"] > 0, "PnL"].sum()
            gl = abs(g.loc[g["PnL"] < 0, "PnL"].sum())
            agg.append([side, len(g), f"{wins / len(g) * 100:.1f}%",
                        f"${g['PnL'].sum():,.0f}",
                        f"{g['ReturnPct'].mean() * 100:+.2f}%",
                        f"{gp / gl:.2f}" if gl else "inf",
                        f"{g['Duration'].mean().days} روز"])
        print("\n" + tabulate(agg, headers=["سمت", "تعداد", "نرخ برد", "سود خالص",
                                            "میانگین بازده", "فاکتور سود", "میانگین مدت"],
                              tablefmt="fancy_grid"))

        last = t.tail(8)[["EntryTime", "ExitTime", "side", "EntryPrice",
                          "ExitPrice", "PnL", "ReturnPct"]].copy()
        rows2 = [[pd.Timestamp(r.EntryTime).strftime("%Y-%m-%d"),
                  pd.Timestamp(r.ExitTime).strftime("%Y-%m-%d"), r.side,
                  f"${r.EntryPrice:,.2f}", f"${r.ExitPrice:,.2f}",
                  f"${r.PnL:,.0f}", f"{r.ReturnPct * 100:+.2f}%"]
                 for r in last.itertuples()]
        print("\n   آخرین معاملات:")
        print(tabulate(rows2, headers=["ورود", "خروج", "سمت", "قیمت ورود",
                                       "قیمت خروج", "سود/زیان", "بازده"],
                       tablefmt="fancy_grid"))

    # ---------------- نمودار ----------------
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2.4, 1]})
    ax = axes[0]
    ax.plot(eq.index, eq["Equity"], color="#2962FF", lw=1.7,
            label="Smart Money Strategy")
    bh = cash * (bt_df["Close"] / bt_df["Close"].iloc[0])
    ax.plot(bh.index, bh.values, color="#9E9E9E", lw=1.2, ls="--",
            label="Buy & Hold")
    ax.axhline(cash, color="black", lw=0.7, alpha=0.4)
    if tr is not None and len(tr):
        lt = tr[tr["Size"] > 0]
        stt = tr[tr["Size"] < 0]
        ax.scatter(lt["EntryTime"], [cash * 0.98] * len(lt), marker="^", s=28,
                   c="#00C853", alpha=0.75, label="Long entries")
        ax.scatter(stt["EntryTime"], [cash * 0.98] * len(stt), marker="v", s=28,
                   c="#D50000", alpha=0.75, label="Short entries")
    ax.set_ylabel("Equity ($)")
    ax.set_title(f"Smart Money (ICT) Backtest — DIA {interval}   |   "
                 f"Return {stats.get('Return [%]', 0):+.2f}%   "
                 f"MaxDD {stats.get('Max. Drawdown [%]', 0):.2f}%   "
                 f"WinRate {stats.get('Win Rate [%]', 0):.1f}%",
                 fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)

    dd = (eq["Equity"] / eq["Equity"].cummax() - 1) * 100
    axes[1].fill_between(eq.index, dd, 0, color="#EF5350", alpha=0.55)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].grid(alpha=0.25)
    plt.tight_layout()
    p = OUTPUT_DIR / f"smc_backtest_{interval}.png"
    plt.savefig(p, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n[+] نمودار بکتست پول هوشمند: {p}")

    try:
        hp = OUTPUT_DIR / f"smc_backtest_{interval}.html"
        bt.plot(open_browser=False, filename=str(hp), resample=False)
        print(f"[+] گزارش تعاملی HTML: {hp}")
    except Exception as e:
        print(f"    (گزارش HTML ذخیره نشد: {e})")

    return dict(stats=stats, equity=eq, trades=tr)
