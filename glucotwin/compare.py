"""Population comparisons: healthy vs prediabetes vs Type 2 diabetes, and patient risk factors."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import HIGH, LOW
from .records import PatientRecord
from .twin import SPD

GROUP_NAMES = {"Normoglycaemic": "Healthy", "Prediabetes": "Prediabetes", "T2D": "Type 2 diabetes"}
GROUP_ORDER = ["Healthy", "Prediabetes", "Type 2 diabetes", "Type 2 diabetes (real)"]
GROUP_COLORS = {"Healthy": "#22c55e", "Prediabetes": "#eab308", "Type 2 diabetes": "#f97316",
                "Type 2 diabetes (real)": "#ef4444"}


def group_of(rec: PatientRecord) -> str:
    return "Type 2 diabetes (real)" if rec.source == "real" else GROUP_NAMES[rec.static["diabetes_status"]]


def patient_summary(rec: PatientRecord) -> dict:
    d = rec.df
    g = d.glucose.to_numpy(float)
    gv = g[np.isfinite(g)]
    days = max(np.isfinite(g).sum() / SPD, 1e-9)
    lows = ((g[1:] < LOW) & (g[:-1] >= LOW)).sum()
    highs = ((g[1:] >= HIGH) & (g[:-1] < HIGH)).sum()
    rises = []
    for t in np.flatnonzero(d.meal.to_numpy()):
        win = g[t + 1:t + 9]
        if np.isfinite(g[t]) and np.isfinite(win).sum() >= 4:
            rises.append(np.nanmax(win) - g[t])
    s = rec.static
    out = {
        "pid": rec.pid, "group": group_of(rec), "source": rec.source,
        "Mean glucose (mg/dL)": gv.mean(), "Glucose variability CV (%)": 100 * gv.std() / gv.mean(),
        "Time in range 70-180 (%)": 100 * ((gv >= LOW) & (gv <= HIGH)).mean(),
        "Time below 70 (%)": 100 * (gv < LOW).mean(), "Time above 180 (%)": 100 * (gv > HIGH).mean(),
        "Lows per day": lows / days, "Highs per day": highs / days,
        "Post-meal rise (mg/dL)": float(np.mean(rises)) if rises else np.nan,
        "HbA1c (%)": s.get("hba1c_pct", np.nan), "Fasting glucose (mg/dL)": s.get("fasting_glucose_mgdl", np.nan),
        "BMI": s.get("bmi", np.nan), "Age": s.get("age", np.nan), "eGFR": s.get("egfr", np.nan),
        "Triglycerides (mg/dL)": s.get("triglycerides_mgdl", np.nan),
    }
    if d.steps.notna().any():
        asleep = d.sleep_stage.to_numpy() >= 2
        out.update({"Steps per day": d.steps.sum() / days, "Sleep (h/night)": float(np.nanmean(d.sleep_h_last)),
                    "Night heart rate (bpm)": float(d.hr[asleep].mean()),
                    "Night HRV (ms)": float(np.nanmean(d.hrv_night_last))})
    return out


def daily_profile(recs: list[PatientRecord]) -> pd.DataFrame:
    """Glucose percentiles by time of day across a group (each patient's own median first)."""
    per_patient = []
    for r in recs:
        g = r.df.glucose.to_numpy(float)
        n = len(g) // SPD * SPD
        per_patient.append(np.nanmedian(g[:n].reshape(-1, SPD), axis=0))
    m = np.vstack(per_patient)
    q = np.nanpercentile(m, [10, 25, 50, 75, 90], axis=0)
    return pd.DataFrame(q.T, columns=["p10", "p25", "p50", "p75", "p90"])


def build(syn: list[PatientRecord], real: list[PatientRecord], seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    allrec = syn + real
    summary = pd.DataFrame([patient_summary(r) for r in allrec])
    groups = {k: [r for r in allrec if group_of(r) == k] for k in GROUP_ORDER}
    profiles = {k: daily_profile(v) for k, v in groups.items() if v}
    samples = {}
    for k, v in groups.items():
        vals = np.concatenate([r.df.glucose.dropna().to_numpy() for r in v])
        samples[k] = rng.choice(vals, min(len(vals), 30000), replace=False)
    return {"summary": summary, "profiles": profiles, "samples": samples}


# ------------------------------------------------------------------ risk factors
def _flag(v, green, amber):
    """green/amber are callables -> 'ok' / 'watch' / 'high'."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return "ok" if green(v) else "watch" if amber(v) else "high"


def risk_factors(static: dict, summ: dict) -> pd.DataFrame:
    """Each risk factor with its value, the healthy reference and a traffic-light flag."""
    s, m = static, summ
    rows = [
        ("HbA1c", s.get("hba1c_pct"), "%", "< 5.7 %", _flag(s.get("hba1c_pct"), lambda v: v < 5.7, lambda v: v <= 7)),
        ("Fasting glucose", s.get("fasting_glucose_mgdl"), "mg/dL", "< 100",
         _flag(s.get("fasting_glucose_mgdl"), lambda v: v < 100, lambda v: v < 126)),
        ("BMI (Asian cut-offs)", s.get("bmi"), "kg/m²", "< 23", _flag(s.get("bmi"), lambda v: v < 23, lambda v: v < 25)),
        ("Age", s.get("age"), "years", "< 45", _flag(s.get("age"), lambda v: v < 45, lambda v: v < 65)),
        ("Years with diabetes", s.get("years_since_dx"), "years", "0",
         _flag(s.get("years_since_dx"), lambda v: v < 1, lambda v: v <= 10)),
        ("Kidney function (eGFR)", s.get("egfr"), "mL/min", "≥ 90", _flag(s.get("egfr"), lambda v: v >= 90, lambda v: v >= 60)),
        ("Triglycerides", s.get("triglycerides_mgdl"), "mg/dL", "< 150",
         _flag(s.get("triglycerides_mgdl"), lambda v: v < 150, lambda v: v < 200)),
        ("Insulin therapy", s.get("insulin"), "", "no", _flag(s.get("insulin"), lambda v: v == 0, lambda v: False)),
        ("Sulfonylurea", s.get("sulfonylurea"), "", "no", _flag(s.get("sulfonylurea"), lambda v: v == 0, lambda v: False)),
        ("Hypertension", s.get("hypertension"), "", "no", _flag(s.get("hypertension"), lambda v: v == 0, lambda v: False)),
        ("Smoker", s.get("smoker"), "", "no", _flag(s.get("smoker"), lambda v: v == 0, lambda v: False)),
        ("Time in range 70-180", m["Time in range 70-180 (%)"], "%", "> 70 %",
         _flag(m["Time in range 70-180 (%)"], lambda v: v > 70, lambda v: v > 50)),
        ("Time below 70", m["Time below 70 (%)"], "%", "< 4 %", _flag(m["Time below 70 (%)"], lambda v: v < 1, lambda v: v < 4)),
        ("Time above 180", m["Time above 180 (%)"], "%", "< 25 %", _flag(m["Time above 180 (%)"], lambda v: v < 25, lambda v: v < 50)),
        ("Glucose variability (CV)", m["Glucose variability CV (%)"], "%", "≤ 36 %",
         _flag(m["Glucose variability CV (%)"], lambda v: v <= 36, lambda v: v <= 42)),
    ]
    if "tcf7l2_risk_alleles" in s and pd.notna(s.get("tcf7l2_risk_alleles")):
        rows.append(("TCF7L2 risk alleles (genetic)", s["tcf7l2_risk_alleles"], "copies", "0",
                     _flag(s["tcf7l2_risk_alleles"], lambda v: v == 0, lambda v: v == 1)))
    if "family_history_t2d" in s and pd.notna(s.get("family_history_t2d")):
        rows.append(("Family history of diabetes", s["family_history_t2d"], "", "no",
                     _flag(s["family_history_t2d"], lambda v: v == 0, lambda v: False)))
    df = pd.DataFrame(rows, columns=["Risk factor", "value", "unit", "Healthy reference", "flag"])

    def show(r):
        v = r.value
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        if r.unit == "" and r["Healthy reference"] == "no":
            return "yes" if v else "no"
        return f"{v:.1f} {r.unit}".strip() if isinstance(v, float) and v % 1 else f"{v:.0f} {r.unit}".strip()

    df["This patient"] = df.apply(show, axis=1)
    df["Status"] = df.flag.map({"ok": "🟢 healthy", "watch": "🟠 watch", "high": "🔴 risk", "n/a": "⚪ not recorded"})
    return df[["Risk factor", "This patient", "Healthy reference", "Status", "flag"]]
