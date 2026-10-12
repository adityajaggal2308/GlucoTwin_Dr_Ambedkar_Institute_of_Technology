"""GlucoTwin clinician dashboard.   Run:  streamlit run app.py"""
from __future__ import annotations

import html as html_lib
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from plotly.subplots import make_subplots

from glucotwin import twin
from glucotwin.branding import wordmark_html
from glucotwin.compare import GROUP_COLORS, GROUP_ORDER, group_of, patient_summary, risk_factors
from glucotwin.features import HIGH, HORIZON, LOW
from glucotwin.model import group_contributions
from glucotwin.report import patient_report_pdf

ROOT = Path(__file__).parent
_icon = ROOT / "docs" / "figures" / "logo.png"
st.set_page_config(page_title="GlucoTwin - Clinician View", page_icon=Image.open(_icon) if _icon.exists() else "🩺",
                   layout="wide", initial_sidebar_state="collapsed")
RED, ORANGE, BLUE, GREEN, GOLD, PURPLE, TEAL = "#ef4444", "#f97316", "#38bdf8", "#22c55e", "#facc15", "#a78bfa", "#14b8a6"
INK, MUTED, GRID, PANEL = "#e5e7eb", "#94a3b8", "#1e293b", "#0f172a"

st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1450px;}
  [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"],
  [data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"] {display: none !important;}
  header[data-testid="stHeader"] {display: none;}
  .stDeployButton, [data-testid="stAppDeployButton"] {display: none;}
  .stTabs [data-baseweb="tab-list"] {gap: 4px; border-bottom: 1px solid #1e293b;}
  .stTabs [data-baseweb="tab"] {padding: 8px 14px; font-weight: 600;}
  h4 {font-weight: 700; margin-top: .6rem;}
  .gt-top {display:flex; align-items:center; gap:14px; margin-bottom:10px;}
  .gt-top .logo {width:44px; height:44px; border-radius:12px; background:linear-gradient(135deg,#14b8a6,#3b82f6);
                 color:#fff; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:22px;}
  .gt-top .name {font-size:26px; font-weight:800; color:#f8fafc; line-height:1.05;}
  .gt-top .tag {font-size:13px; color:#94a3b8;}
  .gt-patient {background:linear-gradient(135deg,#131c2e,#0f1a2e); border:1px solid #1e293b; border-radius:14px;
               padding:16px 20px; margin:6px 0 14px;}
  .gt-patient .title {font-size:24px; font-weight:800; color:#f8fafc;}
  .gt-patient .sub {color:#94a3b8; font-size:14px; margin-top:3px;}
  .chip {display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:600; margin:8px 6px 0 0;}
  .chip.teal {background:rgba(20,184,166,.15); color:#5eead4;} .chip.blue {background:rgba(56,189,248,.15); color:#7dd3fc;}
  .chip.amber {background:rgba(250,204,21,.15); color:#fde047;} .chip.red {background:rgba(239,68,68,.15); color:#fca5a5;}
  .chip.grey {background:rgba(148,163,184,.15); color:#cbd5e1;} .chip.green {background:rgba(34,197,94,.15); color:#86efac;}
  .kpi {background:#131c2e; border:1px solid #1e293b; border-radius:14px; padding:14px 16px; min-height:116px;
        border-top:4px solid #334155;}
  .kpi .label {font-size:11.5px; text-transform:uppercase; letter-spacing:.05em; color:#94a3b8; font-weight:700;}
  .kpi .value {font-size:30px; font-weight:800; color:#f8fafc; margin-top:4px; line-height:1.1;}
  .kpi .note {font-size:12.5px; color:#94a3b8; margin-top:4px;}
  .kpi.danger {border-top-color:#ef4444;} .kpi.danger .value {color:#f87171;}
  .kpi.warn {border-top-color:#f97316;} .kpi.warn .value {color:#fb923c;}
  .kpi.ok {border-top-color:#22c55e;} .kpi.ok .value {color:#4ade80;}
  .kpi.info {border-top-color:#38bdf8;}
  .gt-alert {border-radius:12px; padding:12px 16px; margin:12px 0 4px; font-size:15px; display:flex; gap:12px;}
  .gt-alert .icon {font-size:22px; line-height:1;}
  .gt-alert.danger {background:rgba(239,68,68,.12); border:1px solid rgba(239,68,68,.45); color:#fecaca;}
  .gt-alert.warn {background:rgba(249,115,22,.12); border:1px solid rgba(249,115,22,.45); color:#fed7aa;}
  .gt-alert.ok {background:rgba(34,197,94,.10); border:1px solid rgba(34,197,94,.40); color:#bbf7d0;}
  table.rf {width:100%; border-collapse:collapse; font-size:14px;}
  table.rf th {text-align:left; color:#94a3b8; font-weight:600; padding:7px 10px; border-bottom:1px solid #1e293b;}
  table.rf td {padding:7px 10px; border-bottom:1px solid #1e293b; color:#e5e7eb;}
  table.rf tr.high td:first-child {border-left:3px solid #ef4444;} table.rf tr.watch td:first-child {border-left:3px solid #f97316;}
  table.rf tr.ok td:first-child {border-left:3px solid #22c55e;} table.rf tr.na td:first-child {border-left:3px solid #475569;}
  .gt-foot {color:#64748b; font-size:12px; margin-top:24px; text-align:center;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- helpers
def kpi(col, label, value, note="", tone=""):
    col.markdown(f'<div class="kpi {tone}"><div class="label">{label}</div><div class="value">{value}</div>'
                 f'<div class="note">{note}</div></div>', unsafe_allow_html=True)


def banner(tone, icon, html):
    st.markdown(f'<div class="gt-alert {tone}"><div class="icon">{icon}</div><div>{html}</div></div>',
                unsafe_allow_html=True)


def style_fig(fig, height, title=None):
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PANEL,
                      font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=INK),
                      margin=dict(l=10, r=10, t=44 if title else 20, b=10),
                      legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"))
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=14, color=MUTED), x=0, y=0.99))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def hexa(c, a):
    c = c.lstrip("#")
    return f"rgba({int(c[0:2], 16)},{int(c[2:4], 16)},{int(c[4:6], 16)},{a})"


SLOT_HOURS = np.arange(96) / 4

CLOCK_FONT = ('<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap" rel="stylesheet">')

CLOCK_CSS = """
<style>
 .gtc-wrap{background:#131c2e;border:1px solid #1e293b;border-radius:14px;padding:14px 16px;min-height:270px;color:#e5e7eb}
 .gtc-row{display:flex;gap:12px} .gtc-cell{flex:1;min-width:0}
 .gtc-lab{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;color:#94a3b8;font-weight:700}
 .gtc-clk{font-family:'Share Tech Mono',Consolas,monospace;font-size:34px;color:#5eead4;letter-spacing:2px;line-height:1.15}
 .gtc-clk.small{font-size:22px;color:#7dd3fc}
 .gtc-sub{font-size:12px;color:#94a3b8}
 .gtc-el{font-family:'Share Tech Mono',Consolas,monospace;color:#fde047;font-size:14px}
 .gtc-list{margin-top:10px;font-size:12.5px;max-height:96px;overflow:auto;border-top:1px solid #1e293b;padding-top:6px}
 .gtc-list div{display:flex;justify-content:space-between;gap:8px;padding:2px 0}
 .gtc-over{color:#fca5a5}
 .gtc-foot{display:flex;align-items:center;justify-content:center;gap:22px;background:#131c2e;border:1px solid #1e293b;
           border-radius:14px;padding:14px 20px}
 .gtc-big{font-family:'Share Tech Mono',Consolas,monospace;font-size:52px;color:#5eead4;letter-spacing:4px;
          text-shadow:0 0 14px rgba(20,184,166,.45)}
 .gtc-d{color:#cbd5e1;font-size:15px;font-weight:600} .gtc-z{color:#94a3b8;font-size:12px}
</style>
"""

CLOCK_JS_COMMON = """
 window.__gtc = window.__gtc || {};
 const p2 = n => String(n).padStart(2, "0");
 const hms = d => p2(d.getHours()) + ":" + p2(d.getMinutes()) + ":" + p2(d.getSeconds());
 const dur = ms => { const s = Math.max(0, Math.floor(ms / 1000));
   return Math.floor(s / 3600) + " h " + p2(Math.floor(s % 3600 / 60)) + " m " + p2(s % 60) + " s"; };
 const dfmt = (d, long) => d.toLocaleDateString(undefined, long ? {weekday: "long", day: "numeric", month: "long", year: "numeric"}
                                                 : {weekday: "short", day: "2-digit", month: "short", year: "numeric"});
 const $ = id => document.getElementById(id);
 const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls;
   if (text !== undefined) e.textContent = text; return e; };
"""

MEAL_CLOCK_HTML = CLOCK_FONT + CLOCK_CSS + """
<div class="gtc-wrap" id="gtc-meal" data-gt="__DATA__">
 <div class="gtc-row">
  <div class="gtc-cell"><div class="gtc-lab">Patient clock (replay)</div><div id="gtc-pc" class="gtc-clk">--:--:--</div><div id="gtc-pd" class="gtc-sub"></div></div>
  <div class="gtc-cell"><div class="gtc-lab">Your live clock</div><div id="gtc-lc" class="gtc-clk small">--:--:--</div><div id="gtc-ld" class="gtc-sub"></div></div>
 </div>
 <div class="gtc-row" style="margin-top:10px">
  <div class="gtc-cell"><div class="gtc-lab">Last meal</div><div id="gtc-lm" class="gtc-sub"></div><div id="gtc-le" class="gtc-el"></div></div>
  <div class="gtc-cell"><div class="gtc-lab">Next expected meal (twin routine)</div><div id="gtc-nm" class="gtc-sub"></div><div id="gtc-ne" class="gtc-el"></div></div>
 </div>
 <div id="gtc-list" class="gtc-list"></div>
</div>
"""

# One page-level script drives every clock (scripts inside columns are not executed by st.html).
FOOTER_CLOCK_HTML = CLOCK_FONT + CLOCK_CSS + """
<div class="gtc-foot"><div id="gtc-t" class="gtc-big">--:--:--</div>
 <div><div id="gtc-d" class="gtc-d"></div><div id="gtc-z" class="gtc-z"></div></div></div>
<script>
(() => {""" + CLOCK_JS_COMMON + """
 clearInterval(window.__gtc.foot);
 const st = window.__gtc.state = window.__gtc.state || {};
 function meal() {
   const box = $("gtc-meal"); if (!box) return;
   if (box.dataset.gt !== st.raw) { st.raw = box.dataset.gt; st.D = JSON.parse(st.raw); st.start = Date.now(); }
   const D = st.D, pt = new Date(new Date(D.now).getTime() + (Date.now() - st.start)), lt = new Date();
   $("gtc-pc").textContent = hms(pt); $("gtc-pd").textContent = dfmt(pt) + " · carbs on board ≈ " + D.cob + " g";
   $("gtc-lc").textContent = hms(lt); $("gtc-ld").textContent = dfmt(lt);
   if (D.meals.length) { const m = D.meals[D.meals.length - 1], mt = new Date(m.t);
     $("gtc-lm").textContent = hms(mt) + " · " + m.carbs + " g carbs"; $("gtc-le").textContent = "eaten " + dur(pt - mt) + " ago"; }
   else { $("gtc-lm").textContent = "no meal in the last 24 h"; $("gtc-le").textContent = ""; }
   const ne = $("gtc-ne");
   if (D.next) { const nt = new Date(D.next), diff = nt - pt;
     $("gtc-nm").textContent = "around " + hms(nt).slice(0, 5);
     ne.textContent = diff >= 0 ? "in " + dur(diff) : "overdue by " + dur(-diff);
     ne.className = diff >= 0 ? "gtc-el" : "gtc-el gtc-over"; }
   else { $("gtc-nm").textContent = "routine not learned yet"; ne.textContent = ""; }
   const list = $("gtc-list"); list.replaceChildren();
   if (!D.meals.length) list.append(el("div", "", "No carbohydrate intake recorded in the last 24 h."));
   D.meals.slice().reverse().forEach(m => { const mt = new Date(m.t), row = el("div");
     row.append(el("span", "", hms(mt) + " · " + m.carbs + " g · " + m.text.slice(0, 40)),
                el("span", "gtc-el", dur(pt - mt) + " ago"));
     list.append(row); });
 }
 function tick() {
   const n = new Date();
   if ($("gtc-t")) { $("gtc-t").textContent = hms(n); $("gtc-d").textContent = dfmt(n, true);
     $("gtc-z").textContent = "Live clock · " + Intl.DateTimeFormat().resolvedOptions().timeZone; }
   try { meal(); } catch (e) { console.error("GlucoTwin meal clock:", e); }
 }
 tick(); window.__gtc.foot = setInterval(tick, 1000);
})();
</script>
"""


@st.cache_resource
def load():
    path = ROOT / "artifacts" / "dashboard_bundle.joblib"
    if not path.exists():
        st.error("Model artefacts not found. Run `python run_pipeline.py` first.")
        st.stop()
    comp_path = ROOT / "artifacts" / "comparison.joblib"
    return joblib.load(path), (joblib.load(comp_path) if comp_path.exists() else None)


B, COMP = load()


@st.cache_data
def predicted_risk_by_patient():
    out = []
    for key in ["synthetic", "real"]:
        r = B[key]["rows"]
        recs = {x.pid: x for x in B[key]["records"]}
        agg = r[r.pid.isin(recs)].groupby("pid").agg(low=("risk_low", "mean"), high=("risk_spike", "mean")).reset_index()
        agg["group"] = [group_of(recs[p]) for p in agg.pid]
        out.append(agg)
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------- header & controls
st.markdown('<div style="margin-bottom:12px">' + wordmark_html(
    48, "Diabetes Digital Twin · predicts dangerous lows (&lt;70) and highs (&gt;180) 2 hours ahead · clinician view")
    + "</div>", unsafe_allow_html=True)

cc = st.columns([1.4, 1.1, 2.6])
cohort_name = cc[0].selectbox("Patient cohort", ["Real patients (ShanghaiT2DM)",
                                                 "Synthetic (healthy · prediabetes · diabetes)"])
REAL = cohort_name.startswith("Real")
C = B["real"] if REAL else B["synthetic"]
records = {r.pid: r for r in C["records"]}
rows = C["rows"]
rows = rows[rows.pid.isin(records)].assign(
    low_alert=lambda r: (r.risk_low >= r.thr_low) & (r.g_now >= LOW),
    high_alert=lambda r: (r.risk_spike >= r.thr_spike) & (r.g_now < HIGH))
groups_here = sorted({group_of(r) for r in records.values()}, key=GROUP_ORDER.index)
gfilter = cc[1].selectbox("Show", ["All groups"] + groups_here, disabled=REAL)
rows = rows.sort_values(["pid", "t"])
_new = lambda col: col & ~col.groupby(rows.pid).shift(fill_value=False)        # alert switches on = 1 episode
burden = pd.DataFrame({"lows": _new(rows.low_alert), "highs": _new(rows.high_alert), "pid": rows.pid}).groupby("pid").sum()
burden["group"] = [group_of(records[p]) for p in burden.index]
if gfilter != "All groups" and not REAL:
    burden = burden[burden.group == gfilter]
burden = burden.sort_values(["lows", "highs"], ascending=False)


def fmt(p):
    s_ = records[p].static
    return (f"{p} · {group_of(records[p])} · {s_['age']:.0f} y {'M' if s_['sex_m'] else 'F'} · "
            f"{burden.lows[p]} low / {burden.highs[p]} high alert episodes")


pid = cc[2].selectbox(f"Patient ({len(burden)} available · type to search · most alerts first)",
                      burden.index.tolist(), format_func=fmt)
rec = records[pid]
s = rec.static
d = rec.df
g = d.glucose.to_numpy(float)
prow = rows[rows.pid == pid].sort_values("t").reset_index(drop=True)
params = twin.calibrate(g, d.carbs.to_numpy(float), min(C["calib_steps"], int(len(d) * 0.4)))
summ = patient_summary(rec)

tabs = st.tabs(["👤 Patient twin", "🧬 Risk factors", "📊 Diabetes vs healthy", "📋 Cohort triage",
                "📈 Model performance", "📚 Data"])

# ---------------------------------------------------------- 1. patient twin
with tabs[0]:
    hit = prow.index[(prow.low_alert & (prow.label_low == 1))]
    if not len(hit):
        hit = prow.index[(prow.high_alert & (prow.label_spike == 1))]
    default_t = int(prow.t[hit[0]] if len(hit) else prow.t.iloc[len(prow) // 2])
    tc = st.columns([3, 1])
    now = tc[0].select_slider("Current time ('now'): drag to replay the monitoring period", options=prow.t.tolist(),
                              value=default_t, format_func=lambda t: rec.time(t).strftime("%a %d %b %H:%M"),
                              key=f"now_{pid}")
    reveal = tc[1].toggle("Reveal what happened next", value=True)
    row = prow[prow.t == now]
    r = row.iloc[0]

    group = group_of(rec)
    dx = (f"T2D for {s['years_since_dx']:.0f} y" if s.get("years_since_dx", 0) >= 1 else "T2D, newly diagnosed") \
        if "diabetes" in group.lower() else group
    labs = [f"BMI {s['bmi']:.1f}"]
    labs += [f"HbA1c {s['hba1c_pct']:.1f}%"] if pd.notna(s.get("hba1c_pct")) else []
    labs += [f"FPG {s['fasting_glucose_mgdl']:.0f} mg/dL"] if pd.notna(s.get("fasting_glucose_mgdl")) else []
    labs += [f"eGFR {s['egfr']:.0f}"] if pd.notna(s.get("egfr")) else []
    chips = [("teal", "Real patient" if REAL else "Synthetic patient"),
             ("green" if group == "Healthy" else "amber" if group == "Prediabetes" else "blue", dx)]
    chips += [("red", "Insulin")] if s.get("insulin") else []
    chips += [("amber", "Sulfonylurea")] if s.get("sulfonylurea") else []
    chips += [("grey", "Metformin")] if s.get("metformin") else []
    chips += [("grey", "Hypertension")] if s.get("hypertension") else []
    st.markdown(f'<div class="gt-patient"><div class="title">{pid} &nbsp;·&nbsp; {s["age"]:.0f} y '
                f'{"Male" if s["sex_m"] else "Female"} &nbsp;·&nbsp; <span style="color:#94a3b8;font-size:16px">'
                f'{rec.time(now):%a %d %b %Y, %H:%M}</span></div>'
                f'<div class="sub">{" · ".join(labs)} &nbsp;|&nbsp; Medicines: {s["meds_text"] or "none"}</div>'
                + "".join(f'<span class="chip {c}">{t}</span>' for c, t in chips) + "</div>", unsafe_allow_html=True)

    rf_now = risk_factors(s, summ)

    def _pdf():
        return io.BytesIO(patient_report_pdf(rec, prow, now, summ, rf_now, params, cohort_name,
                                             COMP["profiles"] if COMP else None))

    def _csv():
        out = d.assign(timestamp=[rec.time(t) for t in range(len(d))]).merge(
            prow[["t", "risk_low", "risk_spike"]], left_index=True, right_on="t", how="left").drop(columns="t")
        return out[["timestamp"] + [c_ for c_ in out.columns if c_ != "timestamp"]].to_csv(index=False)

    dl = st.columns([1.2, 1.2, 3])
    dl[0].download_button("📄 Download PDF report", data=_pdf, file_name=f"GlucoTwin_{pid}_{rec.time(now):%Y%m%d_%H%M}.pdf",
                          mime="application/pdf", on_click="ignore", width="stretch", type="primary")
    dl[1].download_button("⬇️ Download patient data (CSV)", data=_csv, file_name=f"GlucoTwin_{pid}_timeline.csv",
                          mime="text/csv", on_click="ignore", width="stretch")
    dl[2].caption("The PDF is a printable 2-page report: status, 24-h glucose & risk chart, meals, risk factors "
                  "and twin physiology. Open it and press Ctrl+P to print.")

    trend = g[now] - g[now - 1] if now > 0 and np.isfinite(g[now - 1]) else 0.0
    arrow = "↑↑" if trend > 15 else "↑" if trend > 5 else "↓↓" if trend < -15 else "↓" if trend < -5 else "→"
    last24 = g[max(now - 96, 0):now + 1]; last24 = last24[np.isfinite(last24)]
    c = st.columns(5)
    kpi(c[0], "Glucose now", f"{g[now]:.0f} <small style='font-size:15px'>mg/dL {arrow}</small>",
        f"{trend:+.0f} mg/dL in last 15 min", "danger" if g[now] < LOW else "warn" if g[now] >= HIGH else "info")
    kpi(c[1], "Risk of LOW · 2 h", f"{r.risk_low:.0%}", "⚠ alert" if r.low_alert else "below alert level",
        "danger" if r.low_alert else "ok")
    kpi(c[2], "Risk of HIGH · 2 h", f"{r.risk_spike:.0%}", "⚠ alert" if r.high_alert else "below alert level",
        "warn" if r.high_alert else "ok")
    tir = ((last24 >= LOW) & (last24 <= HIGH)).mean()
    kpi(c[3], "Time in range · 24 h", f"{tir:.0%}",
        f"{(last24 < LOW).mean():.0%} below 70 · {(last24 > HIGH).mean():.0%} above 180", "ok" if tir >= 0.7 else "warn")
    if REAL:
        u = d.insulin_bolus_u[max(now - 96, 0):now + 1].sum() + d.insulin_basal_u[max(now - 96, 0):now + 1].sum()
        kpi(c[4], "Insulin · 24 h", f"{u:.0f} <small style='font-size:15px'>IU</small>", "from treatment log", "info")
    else:
        kpi(c[4], "Sleep last night", f"{d.sleep_h_last[now]:.1f} <small style='font-size:15px'>h</small>",
            f"night HRV {d.hrv_night_last[now]:.0f} ms", "info")

    if g[now] < LOW:
        banner("danger", "🔻", f"<b>Hypoglycaemic now</b> ({g[now]:.0f} mg/dL).")
    elif r.low_alert:
        on_drug = s.get("insulin") or s.get("sulfonylurea")
        banner("danger", "⚠️", f"<b>Hypoglycaemia predicted within 2 hours</b> (risk {r.risk_low:.0%}).<br>"
               "Suggested for clinician review: check capillary glucose · "
               + ("confirm the patient has eaten after their insulin / sulfonylurea dose · " if on_drug
                  else "confirm the next meal is not delayed · ") + "consider a 15 g carbohydrate snack.")
    if g[now] >= HIGH:
        banner("warn", "🔺", f"<b>Hyperglycaemic now</b> ({g[now]:.0f} mg/dL).")
    elif r.high_alert:
        banner("warn", "⚠️", f"<b>Glucose spike predicted</b> above {HIGH} mg/dL within 2 hours (risk {r.risk_spike:.0%})."
               " Try the what-if simulator below: smaller portion, millet swap, post-meal walk.")
    if not (r.low_alert or r.high_alert) and LOW <= g[now] < HIGH:
        banner("ok", "✅", "<b>Stable.</b> No low or high predicted in the next 2 hours.")

    # ---- timeline
    lo, hi = max(now - 96, 0), min(now + HORIZON, len(d) - 1)
    ts = rec.time
    past = np.arange(lo, now + 1); fut = np.arange(now, hi + 1)
    wear = d.steps.notna().any()
    nrow = 4 if wear else 2
    fig = make_subplots(rows=nrow, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.55, 0.15, 0.15, 0.15] if wear else [0.78, 0.22],
                        specs=[[{"secondary_y": True}]] + [[{}]] * (nrow - 1))
    fig.add_trace(go.Scatter(x=[ts(t) for t in past], y=g[past], name="CGM glucose", line=dict(color=BLUE, width=2.5)), 1, 1)
    if reveal:
        fig.add_trace(go.Scatter(x=[ts(t) for t in fut], y=g[fut], name="What actually happened",
                                 line=dict(color=BLUE, dash="dot", width=2), opacity=0.7), 1, 1)
    pr = prow[(prow.t >= lo) & (prow.t <= now)]
    fig.add_trace(go.Scatter(x=[ts(t) for t in pr.t], y=pr.risk_low * 100, name="Low risk %",
                             line=dict(color=RED, width=1.6)), 1, 1, secondary_y=True)
    fig.add_trace(go.Scatter(x=[ts(t) for t in pr.t], y=pr.risk_spike * 100, name="High risk %",
                             line=dict(color=ORANGE, width=1.3, dash="dash")), 1, 1, secondary_y=True)
    mt = [t for t in range(lo, now + 1) if d.meal[t]]
    fig.add_trace(go.Scatter(x=[ts(t) for t in mt], y=[g[t] if np.isfinite(g[t]) else 100 for t in mt], mode="markers",
                             marker=dict(symbol="star", size=13, color=GOLD), name="Meal",
                             hovertext=[(d.meal_text[t] or "") + f" (~{d.carbs[t]:.0f} g carbs)" for t in mt],
                             hoverinfo="text"), 1, 1)
    fig.add_hrect(y0=LOW, y1=HIGH, fillcolor=GREEN, opacity=0.07, line_width=0, row=1, col=1, exclude_empty_subplots=False)
    fig.add_hline(y=LOW, line_color=RED, line_width=1, line_dash="dot", row=1, col=1, exclude_empty_subplots=False,
                  annotation_text="low (70)", annotation_position="bottom left", annotation_font_color=RED)
    fig.add_hline(y=HIGH, line_color=ORANGE, line_width=1, line_dash="dot", row=1, col=1, exclude_empty_subplots=False,
                  annotation_text="high (180)", annotation_position="top left", annotation_font_color=ORANGE)
    fig.add_vline(x=ts(now), line_dash="dash", line_color=MUTED)
    if wear:
        fig.add_trace(go.Bar(x=[ts(t) for t in past], y=d.steps[past], name="Steps / 15 min", marker_color=GREEN), 2, 1)
        fig.add_trace(go.Scatter(x=[ts(t) for t in past], y=d.hr[past], name="Heart rate", line=dict(color=PURPLE, width=1.2)), 3, 1)
        st_ = d.sleep_stage[past].to_numpy()
        fig.add_trace(go.Scatter(x=[ts(t) for t in past], y=np.where(st_ >= 1, st_, None), name="Sleep stage",
                                 line=dict(color=TEAL, shape="hv")), 4, 1)
        fig.update_yaxes(title_text="steps", row=2, col=1); fig.update_yaxes(title_text="bpm", row=3, col=1)
        fig.update_yaxes(title_text="sleep", tickvals=[1, 2, 3, 4], ticktext=["Wake", "Light", "Deep", "REM"], row=4, col=1)
    else:
        ins = d.insulin_bolus_u[past] + d.insulin_basal_u[past]
        fig.add_trace(go.Bar(x=[ts(t) for t in past], y=d.carbs[past], name="Carbs (g)", marker_color=GOLD), 2, 1)
        fig.add_trace(go.Bar(x=[ts(t) for t in past], y=ins, name="Insulin (IU)", marker_color=PURPLE), 2, 1)
        fig.update_yaxes(title_text="g / IU", row=2, col=1)
    gmax = np.nanmax(g[lo:hi + 1]) if np.isfinite(g[lo:hi + 1]).any() else 250
    fig.update_yaxes(title_text="glucose mg/dL", range=[40, max(260, gmax + 20)], row=1, col=1)
    fig.update_yaxes(title_text="risk %", range=[0, 100], tickvals=[0, 25, 50, 75, 100], secondary_y=True,
                     row=1, col=1, showgrid=False)
    style_fig(fig, 620 if wear else 480, "Last 24 hours  →  next 2 hours")
    fig.update_layout(legend=dict(y=1.1), bargap=0.1)
    st.plotly_chart(fig, width="stretch")

    # ---- meal intake timing with live clocks
    st.markdown("#### Carbohydrate intake & meal timing")
    cal_days = max(min(C["calib_steps"], int(len(d) * 0.4)) // 96, 1)
    eat = d.meal.to_numpy()[: cal_days * 96].astype(float)
    eat = np.pad(eat, (0, cal_days * 96 - len(eat))).reshape(cal_days, 96).mean(0)
    sm = np.convolve(np.r_[eat[-2:], eat, eat[:2]], np.ones(5), "valid")
    peaks = [i for i in range(96) if sm[i] >= 0.3 and sm[i] == sm[np.arange(i - 4, i + 5) % 96].max()]
    slot_now = now % 96
    nxt = next((p_ for p_ in peaks if p_ > slot_now), peaks[0] + 96 if peaks else None)
    next_meal = rec.time(now - slot_now + nxt).isoformat() if nxt is not None else None
    meals_24 = [{"t": rec.time(t).isoformat(), "carbs": round(float(d.carbs[t])), "text": (d.meal_text[t] or "logged meal")[:80]}
                for t in range(max(now - 96, 0), now + 1) if d.meal[t]]
    cob_now = float(twin.carbs_on_board(d.carbs.to_numpy(float))[now])
    mc1, mc2 = st.columns([1.15, 1])
    with mc1:
        cfig = go.Figure(go.Bar(x=[pd.Timestamp(m_["t"]) for m_ in meals_24], y=[m_["carbs"] for m_ in meals_24],
                                marker_color=GOLD, width=1000 * 60 * 40,
                                text=[f"<b>{pd.Timestamp(m_['t']):%H:%M}</b><br>{m_['carbs']} g" for m_ in meals_24],
                                textposition="outside", textfont=dict(size=13, color=INK), cliponaxis=False, constraintext="none",
                                hovertext=[m_["text"] for m_ in meals_24], hoverinfo="text+y"))
        cfig.add_vline(x=rec.time(now), line_dash="dash", line_color=MUTED)
        style_fig(cfig, 300, "Carbs eaten in the last 24 hours (bar = intake time)")
        cfig.update_layout(xaxis_range=[rec.time(max(now - 96, 0)), rec.time(now + 4)], yaxis_title="carbs (g)",
                           yaxis_range=[0, max([m_["carbs"] for m_ in meals_24] + [60]) * 1.35], showlegend=False)
        st.plotly_chart(cfig, width="stretch")
    with mc2:
        st.html(MEAL_CLOCK_HTML.replace("__DATA__", html_lib.escape(json.dumps({
            "now": rec.time(now).isoformat(), "meals": meals_24, "next": next_meal, "cob": round(cob_now)}), quote=True)))

    left, right = st.columns(2)
    with left:
        st.markdown("#### Why this risk? (explainable drivers)")
        which = st.radio("Explain", ["Low risk", "High risk"], horizontal=True, label_visibility="collapsed")
        target = "low" if which == "Low risk" else "spike"
        if REAL:
            mdl, thr = C["models"][target][int(prow.fold.iloc[0])]
            feats = C["features"]
        else:
            mdl, feats, thr = C["models"][target]
        contrib = group_contributions(mdl, row, C["reference"], feats)
        cf = go.Figure(go.Bar(x=contrib.values * 100, y=contrib.index, orientation="h",
                              marker_color=[RED if v > 0 else GREEN for v in contrib.values]))
        style_fig(cf, 290)
        cf.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="risk points vs a typical patient (red = raises risk)")
        st.plotly_chart(cf, width="stretch")
        st.markdown(f"#### Twin physiology (learned from the first {min(C['calib_steps'], int(len(d) * 0.4)) // 96} days)")
        st.dataframe(pd.DataFrame({
            "Parameter": ["Overnight baseline glucose", "Peak rise per 10 g carbs", "Glucose clearance half-life",
                          "Meals used to calibrate"],
            "This patient": [f"{params.twin_basal:.0f} mg/dL", f"+{10 * params.twin_peak_per_g:.0f} mg/dL",
                             f"{np.log(2) / params.twin_k:.0f} min", f"{params.n_meals_fit}"]}),
            hide_index=True, width="stretch")
    with right:
        st.markdown("#### What-if simulator")
        st.caption("Test lifestyle advice on this patient's twin before giving it. Medication changes are left to the clinician.")
        meal_c = st.slider("Planned meal (g carbohydrate)", 0, 150, 60, 5)
        meal_in = st.slider("Eaten in (minutes)", 0, 90, 15, 15)
        swap = st.checkbox("Swap white rice for millet / brown rice (−25 % carbs)")
        walk = st.radio("Post-meal walk", [0, 10, 15, 30], horizontal=True, format_func=lambda m: f"{m} min" if m else "none")
        recent = [(now - t, float(d.carbs[t])) for t in range(max(now - 20, 0), now + 1) if d.carbs[t] > 0]
        g0 = g[now] if np.isfinite(g[now]) else params.twin_basal
        planned = lambda c_: (meal_in // 15, c_) if c_ else None
        base = twin.forecast(params, g0, recent, planned(meal_c), 0)
        plan = twin.forecast(params, g0, recent, planned(meal_c * (0.75 if swap else 1)), walk)
        xf = [rec.time(now + k) for k in range(1, 13)]
        wf = go.Figure()
        wf.add_trace(go.Scatter(x=xf, y=base, name="Usual plan", line=dict(color=RED, width=2)))
        wf.add_trace(go.Scatter(x=xf, y=plan, name="With advice", line=dict(color=GREEN, width=3)))
        wf.add_hrect(y0=LOW, y1=HIGH, fillcolor=GREEN, opacity=0.07, line_width=0)
        style_fig(wf, 290)
        wf.update_layout(yaxis_title="mg/dL")
        st.plotly_chart(wf, width="stretch")
        st.markdown(f"Predicted peak: **{base.max():.0f} → {plan.max():.0f} mg/dL** · minutes above 180: "
                    f"**{(base > HIGH).sum() * 15} → {(plan > HIGH).sum() * 15}**")
    with st.expander("📄 Electronic Health Record"):
        ehr = dict(rec.static); ehr.update(rec.info)
        st.dataframe(pd.Series(ehr, name="value").astype(str), width="stretch")

# ---------------------------------------------------------- 2. risk factors
with tabs[1]:
    st.markdown(f"#### Risk profile of {pid} ({group_of(rec)}, {s['age']:.0f} y)")
    rf = risk_factors(s, summ)
    n_high, n_watch = int((rf.flag == "high").sum()), int((rf.flag == "watch").sum())
    k = st.columns(5)
    kpi(k[0], "Risk factors flagged", f"<span style='color:#f87171'>{n_high}</span><small style='font-size:14px;color:#94a3b8'> risk</small> "
        f"<span style='color:#fb923c'>{n_watch}</span><small style='font-size:14px;color:#94a3b8'> watch</small>", f"of {len(rf)} checked",
        "danger" if n_high >= 4 else "warn" if n_high else "ok")
    kpi(k[1], "Low-alert rate", f"{prow.low_alert.mean():.0%}", "of monitored time with a LOW alert",
        "danger" if prow.low_alert.mean() > 0.05 else "ok")
    kpi(k[2], "High-alert rate", f"{prow.high_alert.mean():.0%}", "of monitored time with a HIGH alert",
        "warn" if prow.high_alert.mean() > 0.15 else "ok")
    kpi(k[3], "Actual lows / day", f"{summ['Lows per day']:.2f}", f"time below 70: {summ['Time below 70 (%)']:.1f} %",
        "danger" if summ["Lows per day"] > 0.1 else "ok")
    kpi(k[4], "Actual highs / day", f"{summ['Highs per day']:.2f}", f"time above 180: {summ['Time above 180 (%)']:.0f} %",
        "warn" if summ["Highs per day"] > 1 else "ok")

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("#### Risk factors vs healthy reference")
        cls = {"high": "high", "watch": "watch", "ok": "ok", "n/a": "na"}
        html = ['<table class="rf"><tr><th>Risk factor</th><th>This patient</th><th>Healthy reference</th><th>Status</th></tr>']
        for _, x in rf.iterrows():
            html.append(f'<tr class="{cls[x.flag]}"><td>{x["Risk factor"]}</td><td><b>{x["This patient"]}</b></td>'
                        f'<td>{x["Healthy reference"]}</td><td>{x.Status}</td></tr>')
        st.markdown("".join(html) + "</table>", unsafe_allow_html=True)
    with right:
        st.markdown("#### This patient's typical day vs healthy people")
        pf = go.Figure()
        if COMP:
            for grp in ["Healthy", "Type 2 diabetes"]:
                p = COMP["profiles"][grp]; col = GROUP_COLORS[grp]
                pf.add_trace(go.Scatter(x=np.r_[SLOT_HOURS, SLOT_HOURS[::-1]], y=np.r_[p.p75, p.p25[::-1]], fill="toself",
                                        fillcolor=hexa(col, 0.18), line=dict(width=0), name=f"{grp} (middle 50 %)",
                                        hoverinfo="skip"))
                pf.add_trace(go.Scatter(x=SLOT_HOURS, y=p.p50, line=dict(color=col, width=1.5, dash="dot"),
                                        name=f"{grp} median"))
        n = len(g) // 96 * 96
        mine = np.nanmedian(g[:n].reshape(-1, 96), axis=0)
        pf.add_trace(go.Scatter(x=SLOT_HOURS, y=mine, line=dict(color=BLUE, width=3.5), name=f"{pid}"))
        pf.add_hline(y=LOW, line_color=RED, line_dash="dot", line_width=1)
        pf.add_hline(y=HIGH, line_color=ORANGE, line_dash="dot", line_width=1)
        style_fig(pf, 400)
        pf.update_layout(xaxis=dict(title="time of day (h)", tickvals=list(range(0, 25, 3))), yaxis_title="glucose mg/dL",
                         legend=dict(y=1.25))
        st.plotly_chart(pf, width="stretch")

    if COMP:
        st.markdown("#### Where this patient sits compared with each group")
        S = COMP["summary"]
        metrics = ["Time in range 70-180 (%)", "Time below 70 (%)", "Time above 180 (%)", "Glucose variability CV (%)",
                   "Mean glucose (mg/dL)", "HbA1c (%)"]
        bf = make_subplots(rows=1, cols=len(metrics), subplot_titles=[m_.replace(" (", "<br>(") for m_ in metrics])
        for i, m_ in enumerate(metrics, 1):
            med = S.groupby("group")[m_].median().reindex(GROUP_ORDER)
            bf.add_trace(go.Bar(x=[x.replace("Type 2 diabetes", "T2D").replace(" (real)", "<br>real") for x in med.index],
                                y=med.values, marker_color=[GROUP_COLORS[x] for x in med.index], showlegend=False,
                                opacity=0.75), 1, i)
            val = summ.get(m_, np.nan)
            if pd.notna(val):
                bf.add_hline(y=val, line_color=BLUE, line_width=3, row=1, col=i)
        style_fig(bf, 380)
        bf.update_layout(margin=dict(t=60))
        bf.update_annotations(font_size=11, font_color=MUTED)
        st.plotly_chart(bf, width="stretch")
        st.caption("Bars = group medians · blue line = this patient.")

    st.markdown("#### Predicted risk across the whole monitoring period")
    daily = prow.assign(day=(prow.t // 96)).groupby("day").agg(low=("risk_low", "max"), high=("risk_spike", "max")).reset_index()
    ev = []
    for day in daily.day:
        gg = g[day * 96:(day + 1) * 96]
        ev.append((int(((gg[1:] < LOW) & (gg[:-1] >= LOW)).sum()), int(((gg[1:] >= HIGH) & (gg[:-1] < HIGH)).sum())))
    daily["actual_lows"], daily["actual_highs"] = zip(*ev)
    xd = [rec.time(int(x) * 96).strftime("%d %b") for x in daily.day]
    dfig = make_subplots(specs=[[{"secondary_y": True}]])
    dfig.add_trace(go.Bar(x=xd, y=daily.actual_lows, name="Actual lows", marker_color=hexa(RED, 0.6)), secondary_y=True)
    dfig.add_trace(go.Bar(x=xd, y=daily.actual_highs, name="Actual highs", marker_color=hexa(ORANGE, 0.45)), secondary_y=True)
    dfig.add_trace(go.Scatter(x=xd, y=daily.low * 100, name="Peak LOW risk %", line=dict(color=RED, width=3), mode="lines+markers"))
    dfig.add_trace(go.Scatter(x=xd, y=daily.high * 100, name="Peak HIGH risk %", line=dict(color=ORANGE, width=3, dash="dash"),
                              mode="lines+markers"))
    style_fig(dfig, 340)
    dfig.update_yaxes(title_text="peak predicted risk %", range=[0, 100], secondary_y=False)
    dfig.update_yaxes(title_text="episodes per day", secondary_y=True, showgrid=False)
    st.plotly_chart(dfig, width="stretch")

# ---------------------------------------------------- 3. diabetes vs healthy
with tabs[2]:
    if not COMP:
        st.warning("Run `python make_comparison.py` to build the comparison data.")
    else:
        S = COMP["summary"]
        st.markdown("#### How glucose behaves in healthy people vs people with diabetes")
        st.caption(f"Synthetic: {(S.group == 'Healthy').sum()} healthy, {(S.group == 'Prediabetes').sum()} prediabetes, "
                   f"{(S.group == 'Type 2 diabetes').sum()} with T2D · Real: {(S.group == 'Type 2 diabetes (real)').sum()} "
                   "recordings of people with T2D (ShanghaiT2DM).")
        sel = st.multiselect("Groups to compare", GROUP_ORDER, default=GROUP_ORDER) or GROUP_ORDER
        n1, n2 = st.columns(2)
        with n1:
            f1 = go.Figure()
            for grp in sel:
                p = COMP["profiles"][grp]; col = GROUP_COLORS[grp]
                f1.add_trace(go.Scatter(x=np.r_[SLOT_HOURS, SLOT_HOURS[::-1]], y=np.r_[p.p75, p.p25[::-1]], fill="toself",
                                        fillcolor=hexa(col, 0.15), line=dict(width=0), showlegend=False, hoverinfo="skip"))
                f1.add_trace(go.Scatter(x=SLOT_HOURS, y=p.p50, name=grp, line=dict(color=col, width=3)))
            f1.add_hrect(y0=LOW, y1=HIGH, fillcolor=GREEN, opacity=0.05, line_width=0)
            style_fig(f1, 400, "Typical 24-hour glucose (median; shaded = middle 50 %)")
            f1.update_layout(xaxis=dict(title="time of day (h)", tickvals=list(range(0, 25, 3))), yaxis_title="mg/dL",
                             legend=dict(y=1.16))
            st.plotly_chart(f1, width="stretch")
        with n2:
            f2 = go.Figure()
            for grp in sel:
                f2.add_trace(go.Histogram(x=COMP["samples"][grp], name=grp, histnorm="percent", nbinsx=70, opacity=0.55,
                                          marker_color=GROUP_COLORS[grp]))
            f2.add_vline(x=LOW, line_color=RED, line_dash="dot"); f2.add_vline(x=HIGH, line_color=ORANGE, line_dash="dot")
            style_fig(f2, 400, "Distribution of all glucose readings")
            f2.update_layout(barmode="overlay", xaxis_title="glucose mg/dL", yaxis_title="% of readings",
                             xaxis_range=[40, 350], legend=dict(y=1.16))
            st.plotly_chart(f2, width="stretch")
        n3, n4 = st.columns(2)
        with n3:
            med = S.groupby("group")[["Time below 70 (%)", "Time in range 70-180 (%)", "Time above 180 (%)"]].mean().reindex(sel)
            f3 = go.Figure()
            for col_, name, color in [("Time below 70 (%)", "Below 70 (low)", RED),
                                      ("Time in range 70-180 (%)", "70-180 (in range)", GREEN),
                                      ("Time above 180 (%)", "Above 180 (high)", ORANGE)]:
                f3.add_trace(go.Bar(y=med.index, x=med[col_], name=name, orientation="h", marker_color=color,
                                    text=[f"{v:.0f}%" for v in med[col_]], textposition="inside"))
            style_fig(f3, 340, "Time in range (average % of the day)")
            f3.update_layout(barmode="stack", xaxis_title="% of time", yaxis=dict(autorange="reversed"), legend=dict(y=1.14))
            st.plotly_chart(f3, width="stretch")
        with n4:
            ev_ = S.groupby("group")[["Lows per day", "Highs per day"]].mean().reindex(sel)
            f4 = go.Figure([go.Bar(x=ev_.index, y=ev_["Highs per day"], name="High episodes / day", marker_color=ORANGE),
                            go.Bar(x=ev_.index, y=ev_["Lows per day"], name="Low episodes / day", marker_color=RED)])
            style_fig(f4, 340, "How often glucose goes out of range")
            f4.update_layout(barmode="group", yaxis_title="episodes per day", legend=dict(y=1.14))
            st.plotly_chart(f4, width="stretch")

        st.markdown("#### Compare any risk factor between groups")
        numeric = [c_ for c_ in S.columns if c_ not in ("pid", "group", "source") and S[c_].notna().sum() > 10]
        n5, n6 = st.columns(2)
        with n5:
            factor = st.selectbox("Risk factor", numeric, index=numeric.index("HbA1c (%)"))
            f5 = go.Figure()
            for grp in sel:
                v = S.loc[S.group == grp, factor].dropna()
                if len(v):
                    f5.add_trace(go.Box(y=v, name=grp, marker_color=GROUP_COLORS[grp], boxmean=True))
            if pd.notna(summ.get(factor, np.nan)):
                f5.add_hline(y=summ[factor], line_color=BLUE, line_width=2, line_dash="dash",
                             annotation_text=f"selected patient {pid}", annotation_font_color=BLUE)
            style_fig(f5, 400, f"{factor} by group")
            f5.update_layout(showlegend=False)
            st.plotly_chart(f5, width="stretch")
        with n6:
            sx = st.columns(2)
            xf_ = sx[0].selectbox("Scatter x-axis", numeric, index=numeric.index("HbA1c (%)"))
            yf_ = sx[1].selectbox("Scatter y-axis", numeric, index=numeric.index("Time in range 70-180 (%)"))
            f6 = go.Figure()
            for grp in sel:
                q = S[S.group == grp]
                f6.add_trace(go.Scatter(x=q[xf_], y=q[yf_], mode="markers", name=grp, text=q.pid,
                                        marker=dict(color=GROUP_COLORS[grp], size=7, opacity=0.7)))
            if pd.notna(summ.get(xf_, np.nan)) and pd.notna(summ.get(yf_, np.nan)):
                f6.add_trace(go.Scatter(x=[summ[xf_]], y=[summ[yf_]], mode="markers", name=f"selected: {pid}",
                                        marker=dict(color=BLUE, size=16, symbol="star", line=dict(color="white", width=1))))
            style_fig(f6, 330)
            f6.update_layout(xaxis_title=xf_, yaxis_title=yf_, legend=dict(y=1.2))
            st.plotly_chart(f6, width="stretch")

        st.markdown("#### Predicted risk by group (held-out patients only)")
        pr_ = predicted_risk_by_patient()
        n7, n8 = st.columns(2)
        for col_, key, title in [(n7, "low", "Average predicted LOW risk per patient"),
                                 (n8, "high", "Average predicted HIGH risk per patient")]:
            ff = go.Figure()
            for grp in sel:
                v = pr_.loc[pr_.group == grp, key] * 100
                if len(v):
                    ff.add_trace(go.Box(y=v, name=grp, marker_color=GROUP_COLORS[grp], boxpoints="all", jitter=0.4))
            style_fig(ff, 340, title)
            ff.update_layout(showlegend=False, yaxis_title="risk %")
            col_.plotly_chart(ff, width="stretch")

        if "Steps per day" in S:
            st.markdown("#### Lifestyle and wearable signals (synthetic patients with smartwatch)")
            wcols = ["Steps per day", "Sleep (h/night)", "Night heart rate (bpm)", "Night HRV (ms)"]
            wf_ = make_subplots(rows=1, cols=4, subplot_titles=wcols)
            for i, wc in enumerate(wcols, 1):
                for grp in [x for x in sel if x != "Type 2 diabetes (real)"]:
                    wf_.add_trace(go.Box(y=S.loc[S.group == grp, wc], name=grp.replace("Type 2 diabetes", "T2D"),
                                         marker_color=GROUP_COLORS[grp], showlegend=False), 1, i)
            style_fig(wf_, 350)
            wf_.update_layout(margin=dict(t=50))
            wf_.update_annotations(font_size=12, font_color=MUTED)
            st.plotly_chart(wf_, width="stretch")

        st.markdown("#### Summary table (group medians)")
        st.dataframe(S.groupby("group")[numeric].median().reindex(GROUP_ORDER).T.round(1), width="stretch")

# ---------------------------------------------------------- 4. cohort triage
with tabs[3]:
    if REAL:
        st.markdown("#### Remote-monitoring worklist · latest reading of each patient")
        snap = rows.sort_values("t").groupby("pid").tail(1).set_index("pid")
    else:
        st.markdown(f"#### Remote-monitoring worklist · {rec.time(now):%a %d %b, %H:%M} (same moment for all synthetic patients)")
        snap = rows[rows.t == now].set_index("pid")
    tri = pd.DataFrame({
        "Group": [group_of(records[p]) for p in snap.index], "Glucose": snap.g_now, "Low risk": snap.risk_low,
        "High risk": snap.risk_spike, "Time <70 (24 h)": snap.tbr_24h, "Time >180 (24 h)": snap.tar_24h,
        "HbA1c %": [records[p].static.get("hba1c_pct") for p in snap.index],
        "Medicines": [records[p].static.get("meds_text") for p in snap.index]})
    tri.insert(0, "Status", np.select([tri.Glucose < LOW, snap.low_alert, tri.Glucose >= HIGH, snap.high_alert],
                                      ["🔴 Low now", "🔴 Low likely", "🟠 High now", "🟠 High likely"], "🟢 Stable"))
    tri = tri.sort_values(["Low risk", "High risk"], ascending=False)
    st.dataframe(tri.style.format({"Low risk": "{:.0%}", "High risk": "{:.0%}", "Time <70 (24 h)": "{:.0%}",
                                   "Time >180 (24 h)": "{:.0%}", "Glucose": "{:.0f}", "HbA1c %": "{:.1f}"})
                 .background_gradient(subset=["Low risk"], cmap="Reds", vmin=0, vmax=1)
                 .background_gradient(subset=["High risk"], cmap="Oranges", vmin=0, vmax=1),
                 width="stretch", height=560)
    st.download_button("⬇️ Download worklist (CSV)", data=tri.to_csv().encode("utf-8"),
                       file_name="GlucoTwin_worklist.csv", mime="text/csv", on_click="ignore")

# ------------------------------------------------------ 5. model performance
with tabs[4]:
    m = B["metrics"]
    cols = ["auroc", "per_patient_auroc", "auprc", "events", "events_warned_pct", "median_lead_min",
            "warned_30min_plus_pct", "false_alerts_per_day", "sensitivity", "ppv"]
    nice = {"low": "LOW glucose (<70) within 2 h", "spike": "HIGH glucose (>180) within 2 h"}
    st.markdown(f"#### Validation on {m['cohort']['real_people']} real patients "
                f"({m['cohort']['cv_folds']}-fold cross-validation, grouped by patient)")
    for t in ["low", "spike"]:
        st.markdown(f"**{nice[t]}**")
        st.dataframe(pd.DataFrame({k_: v for k_, v in m["real"][t].items() if isinstance(v, dict)}).T[cols], width="stretch")
        st.caption(f"Dashboard risk for real patients comes from: {m['real'][t].get('dashboard_uses', '')}")
    st.markdown(f"#### Synthetic cohort: what does each data stream add? ({m['cohort']['split']['test']} held-out patients)")
    for t in ["low", "spike"]:
        st.markdown(f"**{nice[t]}**")
        st.dataframe(pd.DataFrame(m["synthetic"][t]).T[cols], width="stretch")
    for f in ["real_validation.png", "synthetic_roc.png", "feature_group_importance.png"]:
        p = ROOT / "docs" / "figures" / f
        if p.exists():
            st.image(str(p))

# ------------------------------------------------------------------ 6. data
with tabs[5]:
    st.markdown("""
**Real patients: ShanghaiT2DM.** 100 people with Type 2 diabetes (109 recordings, ~13 days each):
CGM every 15 min, meals (food and weight), insulin and oral drugs, plus clinical and lab summary.
Zhao Q. *et al.*, "Chinese diabetes datasets for data-driven machine learning", *Scientific Data* 10, 35 (2023),
licensed under CC BY 4.0. Only the English columns are used. Carbohydrates are estimated from food weights.
The dataset has no smartwatch data.

**Synthetic patients.** 400 virtual adults (healthy, prediabetes, T2D) with EHR, labs, a TCF7L2 genetic marker, and
14 days of CGM, steps, heart rate, HRV and sleep stages. They are generated by a physiological simulator with Indian
meal patterns, insulin and sulfonylurea effects, missed doses, skipped meals, sleep debt and stress. The patient
picker shows the 60 held-out test patients, whose risks come from a model that never saw them. The comparison
charts use all 400.

**How predictions on real patients stay honest.** Each real patient's risk comes from a model that never saw that
patient (patient-grouped cross-validation).
""")

st.markdown("---")
st.html(FOOTER_CLOCK_HTML, unsafe_allow_javascript=True)
st.markdown('<div class="gt-foot">GlucoTwin · research prototype, not a medical device · real data: ShanghaiT2DM '
            '(CC BY 4.0) · synthetic data generated by this project</div>', unsafe_allow_html=True)
