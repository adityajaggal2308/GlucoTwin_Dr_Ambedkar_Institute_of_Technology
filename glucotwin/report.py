"""Printable PDF patient report (A4, light theme) generated with matplotlib."""
from __future__ import annotations

import io
import textwrap
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from .branding import draw_logo
from .features import HIGH, HORIZON, LOW

INK, MUTED, TEAL, RED, ORANGE, GREEN, BLUE, GOLD = ("#0f172a", "#64748b", "#0f766e", "#dc2626", "#ea580c",
                                                    "#16a34a", "#2563eb", "#d97706")
FLAG_COLOR = {"high": RED, "watch": ORANGE, "ok": GREEN, "n/a": "#94a3b8"}
A4 = (8.27, 11.69)


def _header(fig, title, subtitle):
    ax = fig.add_axes([0, 0.93, 1, 0.07]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="#0b1220"))
    lx = fig.add_axes([0.035, 0.937, 0.05, 0.05 * A4[0] / A4[1] * 1.4]); lx.axis("off")
    lx.set_xlim(0, 1); lx.set_ylim(0, 1); draw_logo(lx, 0, 0, 1)
    ax.text(0.10, 0.62, "Gluco", color="white", fontsize=17, weight="bold", va="center")
    ax.text(0.181, 0.62, "Twin", color="#14b8a6", fontsize=17, weight="bold", va="center")
    ax.text(0.10, 0.25, title, color="#cbd5e1", fontsize=9.5, va="center")
    ax.text(0.97, 0.62, subtitle, color="#cbd5e1", fontsize=8.5, ha="right", va="center")
    ax.text(0.97, 0.25, f"Generated {datetime.now():%d %b %Y, %H:%M:%S}", color="#94a3b8", fontsize=8,
            ha="right", va="center")


def _footer(fig, page):
    fig.text(0.5, 0.015, f"GlucoTwin · research prototype, not a medical device · page {page}",
             ha="center", fontsize=7.5, color=MUTED)


def _box(fig, x, y, w, h, label, value, note, color):
    ax = fig.add_axes([x, y, w, h]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fc="#f8fafc", ec="#e2e8f0"))
    ax.add_patch(plt.Rectangle((0, 0.93), 1, 0.07, color=color))
    ax.text(0.07, 0.72, label.upper(), fontsize=6.8, color=MUTED, weight="bold")
    ax.text(0.07, 0.36, value, fontsize=15, color=color if color in (RED, ORANGE) else INK, weight="bold")
    ax.text(0.07, 0.1, note, fontsize=6.6, color=MUTED)


def patient_report_pdf(rec, prow: pd.DataFrame, now: int, summ: dict, rf: pd.DataFrame, params,
                       cohort: str, profiles: dict | None = None) -> bytes:
    d, s = rec.df, rec.static
    g = d.glucose.to_numpy(float)
    r = prow[prow.t == now].iloc[0]
    low_alert = (r.risk_low >= r.thr_low) and g[now] >= LOW
    high_alert = (r.risk_spike >= r.thr_spike) and g[now] < HIGH
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # ---------------------------------------------------------------- page 1
        fig = plt.figure(figsize=A4)
        _header(fig, "Patient report · glucose lows & highs 2 hours ahead", f"{cohort}")
        info = fig.add_axes([0.05, 0.84, 0.9, 0.08]); info.axis("off")
        sex = "Male" if s["sex_m"] else "Female"
        info.text(0, 0.78, f"{rec.pid}  ·  {s['age']:.0f} y {sex}  ·  {s['diabetes_status']}", fontsize=15,
                  weight="bold", color=INK)
        labs = [f"BMI {s['bmi']:.1f}"]
        labs += [f"HbA1c {s['hba1c_pct']:.1f}%"] if pd.notna(s.get("hba1c_pct")) else []
        labs += [f"Fasting glucose {s['fasting_glucose_mgdl']:.0f} mg/dL"] if pd.notna(s.get("fasting_glucose_mgdl")) else []
        labs += [f"eGFR {s['egfr']:.0f}"] if pd.notna(s.get("egfr")) else []
        info.text(0, 0.42, " · ".join(labs), fontsize=9, color=MUTED)
        info.text(0, 0.1, f"Medicines: {s.get('meds_text') or 'none'}   ·   Report time point: "
                  f"{rec.time(now):%a %d %b %Y, %H:%M}", fontsize=9, color=MUTED)

        last24 = g[max(now - 96, 0):now + 1]; last24 = last24[np.isfinite(last24)]
        tir = ((last24 >= LOW) & (last24 <= HIGH)).mean()
        boxes = [("Glucose now", f"{g[now]:.0f} mg/dL", "CGM reading", RED if g[now] < LOW else ORANGE if g[now] >= HIGH else BLUE),
                 ("Risk of LOW · 2 h", f"{r.risk_low:.0%}", "ALERT" if low_alert else "below alert level", RED if low_alert else GREEN),
                 ("Risk of HIGH · 2 h", f"{r.risk_spike:.0%}", "ALERT" if high_alert else "below alert level", ORANGE if high_alert else GREEN),
                 ("Time in range · 24 h", f"{tir:.0%}", f"{(last24 < LOW).mean():.0%} below 70", GREEN if tir >= .7 else ORANGE)]
        for i, (a, b, c, col) in enumerate(boxes):
            _box(fig, 0.05 + i * 0.23, 0.745, 0.21, 0.075, a, b, c, col)

        msg = []
        if low_alert:
            msg.append("HYPOGLYCAEMIA PREDICTED within 2 h: check capillary glucose, confirm the patient has eaten "
                       "after any insulin / sulfonylurea dose, consider a 15 g carbohydrate snack.")
        if high_alert:
            msg.append("GLUCOSE SPIKE PREDICTED above 180 mg/dL within 2 h: review the next meal portion, "
                       "lower-glycaemic swap, post-meal walk.")
        if not msg:
            msg.append("Stable: no low or high predicted in the next 2 hours.")
        fig.text(0.05, 0.715, "\n".join(textwrap.fill(m_, 115) for m_ in msg), fontsize=8.5, color=RED if low_alert else ORANGE if high_alert else GREEN,
                 va="top", wrap=True)

        lo, hi = max(now - 96, 0), min(now + HORIZON, len(d) - 1)
        x = [rec.time(t) for t in range(lo, hi + 1)]
        ax = fig.add_axes([0.08, 0.42, 0.84, 0.25])
        ax.axhspan(LOW, HIGH, color=GREEN, alpha=0.07)
        ax.axhline(LOW, color=RED, ls=":", lw=1); ax.axhline(HIGH, color=ORANGE, ls=":", lw=1)
        ax.plot(x[: now - lo + 1], g[lo:now + 1], color=BLUE, lw=1.8, label="CGM glucose")
        ax.axvline(rec.time(now), color=MUTED, ls="--", lw=1)
        mt = [t for t in range(lo, now + 1) if d.meal[t]]
        ax.scatter([rec.time(t) for t in mt], [g[t] if np.isfinite(g[t]) else 100 for t in mt], marker="*", s=90,
                   color=GOLD, zorder=5, label="meal")
        ax.set_ylabel("glucose mg/dL", fontsize=8); ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
        ax2 = ax.twinx()
        pr = prow[(prow.t >= lo) & (prow.t <= now)]
        ax2.plot([rec.time(t) for t in pr.t], pr.risk_low * 100, color=RED, lw=1, label="low risk %")
        ax2.plot([rec.time(t) for t in pr.t], pr.risk_spike * 100, color=ORANGE, lw=1, ls="--", label="high risk %")
        ax2.set_ylim(0, 100); ax2.set_ylabel("predicted risk %", fontsize=8); ax2.tick_params(labelsize=7)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left", ncol=4, frameon=False)
        ax.set_title("Last 24 hours of glucose and predicted risk", fontsize=9.5, color=INK, loc="left")

        meals = [(rec.time(t), d.carbs[t], (d.meal_text[t] or "")) for t in range(lo, now + 1) if d.meal[t]]
        tx = fig.add_axes([0.05, 0.06, 0.9, 0.31]); tx.axis("off")
        tx.text(0, 1, "Carbohydrate intake (last 24 h)", fontsize=10, weight="bold", color=INK, va="top")
        if meals:
            rows = [[m[0].strftime("%a %d %b  %H:%M:%S"), f"{m[1]:.0f} g", (m[2][:70] + ("…" if len(m[2]) > 70 else "")) or "logged meal"]
                    for m in meals[-10:]]
            tb = tx.table(cellText=rows, colLabels=["Intake time", "Carbs", "Food"], cellLoc="left", colLoc="left",
                          bbox=[0, max(0.0, 0.9 - 0.085 * (len(rows) + 1)), 1, 0.085 * (len(rows) + 1)],
                          colWidths=[0.27, 0.1, 0.63])
            tb.auto_set_font_size(False); tb.set_fontsize(7.5)
            for (i, j), c in tb.get_celld().items():
                c.set_edgecolor("#e2e8f0")
                if i == 0:
                    c.set_facecolor("#f1f5f9"); c.set_text_props(weight="bold")
        else:
            tx.text(0, 0.85, "No meals logged in the last 24 hours.", fontsize=9, color=MUTED, va="top")
        _footer(fig, 1); pdf.savefig(fig); plt.close(fig)

        # ---------------------------------------------------------------- page 2
        fig = plt.figure(figsize=A4)
        _header(fig, "Risk factors & monitoring summary", rec.pid)
        tx = fig.add_axes([0.05, 0.47, 0.9, 0.44]); tx.axis("off")
        tx.text(0, 1, "Risk factors vs healthy reference", fontsize=10, weight="bold", color=INK, va="top")
        cells = [[a, b, c, st.split(" ", 1)[1] if " " in st else st] for a, b, c, st in
                 zip(rf["Risk factor"], rf["This patient"], rf["Healthy reference"], rf["Status"])]
        tb = tx.table(cellText=cells, colLabels=["Risk factor", "This patient", "Healthy reference", "Status"],
                      cellLoc="left", colLoc="left", bbox=[0, 0, 1, 0.94], colWidths=[0.36, 0.22, 0.22, 0.2])
        tb.auto_set_font_size(False); tb.set_fontsize(8)
        for (i, j), c in tb.get_celld().items():
            c.set_edgecolor("#e2e8f0")
            if i == 0:
                c.set_facecolor("#f1f5f9"); c.set_text_props(weight="bold")
            elif j == 3:
                c.set_text_props(color=FLAG_COLOR[rf.flag.iloc[i - 1]], weight="bold")

        tx = fig.add_axes([0.05, 0.33, 0.42, 0.12]); tx.axis("off")
        tx.text(0, 1, "Monitoring summary", fontsize=10, weight="bold", color=INK, va="top")
        lines = [f"Days monitored: {len(d) / 96:.1f}",
                 f"Lows per day: {summ['Lows per day']:.2f}   ·   Highs per day: {summ['Highs per day']:.2f}",
                 f"Time below 70: {summ['Time below 70 (%)']:.1f} %   ·   above 180: {summ['Time above 180 (%)']:.0f} %",
                 f"Low-alert rate: {((prow.risk_low >= prow.thr_low) & (prow.g_now >= LOW)).mean():.0%} of monitored time",
                 f"High-alert rate: {((prow.risk_spike >= prow.thr_spike) & (prow.g_now < HIGH)).mean():.0%} of monitored time"]
        tx.text(0, 0.78, "\n".join(lines), fontsize=8.3, color=INK, va="top", linespacing=1.7)
        tx = fig.add_axes([0.53, 0.33, 0.42, 0.12]); tx.axis("off")
        tx.text(0, 1, "Digital-twin physiology", fontsize=10, weight="bold", color=INK, va="top")
        tx.text(0, 0.78, "\n".join([f"Overnight baseline glucose: {params.twin_basal:.0f} mg/dL",
                                     f"Peak rise per 10 g carbs: +{10 * params.twin_peak_per_g:.0f} mg/dL",
                                     f"Glucose clearance half-life: {np.log(2) / params.twin_k:.0f} min",
                                     f"Meals used to calibrate: {params.n_meals_fit}"]),
                fontsize=8.3, color=INK, va="top", linespacing=1.7)

        ax = fig.add_axes([0.1, 0.07, 0.82, 0.22])
        hours = np.arange(96) / 4
        if profiles:
            for grp, col in [("Healthy", GREEN), ("Type 2 diabetes", ORANGE)]:
                if grp in profiles:
                    p = profiles[grp]
                    ax.fill_between(hours, p.p25, p.p75, color=col, alpha=0.15, label=f"{grp} (middle 50 %)")
        n = len(g) // 96 * 96
        ax.plot(hours, np.nanmedian(g[:n].reshape(-1, 96), axis=0), color=BLUE, lw=2.2, label=f"{rec.pid} typical day")
        ax.axhline(LOW, color=RED, ls=":", lw=1); ax.axhline(HIGH, color=ORANGE, ls=":", lw=1)
        ax.set_xticks(range(0, 25, 3)); ax.set_xlabel("time of day (h)", fontsize=8); ax.set_ylabel("mg/dL", fontsize=8)
        ax.tick_params(labelsize=7); ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=3)
        ax.set_title("This patient's typical day vs healthy people", fontsize=9.5, color=INK, loc="left")
        _footer(fig, 2); pdf.savefig(fig); plt.close(fig)
    return buf.getvalue()
