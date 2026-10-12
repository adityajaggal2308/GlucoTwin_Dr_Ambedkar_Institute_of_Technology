"""Build docs/architecture.pdf and docs/presentation.pdf from the latest metrics.

    python docs/make_docs.py      (run after run_pipeline.py)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

DOCS = Path(__file__).parent
ROOT = DOCS.parent
TEAM = "<Team Name>"                      # TODO: fill in
COLLEGE = "<College / Incubator Name>"    # TODO: fill in
TEAL, RED, GOLD, GREEN, INK = "#1f6f8b", "#d1495b", "#edae49", "#66a182", "#22313f"


def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015",
                                fc=color + "22", ec=color, lw=1.6))
    ax.text(x + w / 2, y + h - 0.025, title, ha="center", va="top", fontsize=10.5, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.068, body, ha="center", va="top", fontsize=7.6, color=INK, linespacing=1.45)


def arrow(ax, a, b, label=""):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#555"))
    if label:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.015, label, ha="center", fontsize=7, color="#555")


def architecture(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.97, "GlucoTwin - Hybrid Digital Twin architecture", ha="center", fontsize=15, weight="bold", color=INK)
    box(ax, 0.02, 0.60, 0.21, 0.28, "Static stream: EHR",
        "Demographics, diagnoses\nHbA1c, FPG, lipids, eGFR\nInsulin / sulfonylurea / metformin\n"
        "TCF7L2 marker (synthetic)\n\nREAL: ShanghaiT2DM (100 pts)\nSYNTHETIC: 400 virtual pts", TEAL)
    box(ax, 0.02, 0.22, 0.21, 0.30, "Dynamic stream",
        "CGM glucose (15 min)\nMeals & insulin doses\nSmartwatch: steps, heart\nrate, HRV, sleep stages\n"
        "(synthetic patients)\n\nREAL: CGM + meals + insulin", GOLD)
    box(ax, 0.30, 0.22, 0.20, 0.66, "Ingestion & fusion",
        "Time alignment (15-min grid)\n\nWindowed features:\n- CGM trend 15/30/60 min\n- 24 h time-above-range\n"
        "- carbs-on-board\n- steps, HR last 30-60 min\n- last-night sleep, HRV\n- personal meal routine\n\n"
        "Static features joined\nper patient", "#7a7a7a")
    box(ax, 0.56, 0.60, 0.20, 0.28, "Mechanistic twin",
        "Calibrated per patient\n(first 5 days of wear):\nbaseline glucose\ncarb gain, clearance k\n"
        "-> what-if forward\nsimulation", GREEN)
    box(ax, 0.56, 0.22, 0.20, 0.30, "ML risk engine",
        "Gradient-boosted trees\n2 models, every 15 min:\nP(LOW <70 in 2 h)\nP(HIGH >180 in 2 h)\n"
        "calibrated + explainable\nvalidated on real patients", RED)
    box(ax, 0.81, 0.22, 0.17, 0.66, "Clinician dashboard",
        "Virtual patient view\nRisk timeline & alerts\nWhy-this-risk drivers\nWhat-if simulator\n"
        "(meal, millet swap,\npost-meal walk)\n\nCohort triage\nworklist\n\nModel performance", TEAL)
    arrow(ax, (0.23, 0.74), (0.30, 0.74)); arrow(ax, (0.23, 0.37), (0.30, 0.37))
    arrow(ax, (0.50, 0.74), (0.56, 0.74), "calibrate"); arrow(ax, (0.50, 0.37), (0.56, 0.37), "features")
    arrow(ax, (0.66, 0.60), (0.66, 0.52), "twin params")
    arrow(ax, (0.76, 0.74), (0.81, 0.74), "what-if"); arrow(ax, (0.76, 0.37), (0.81, 0.37), "risk + drivers")
    ax.text(0.5, 0.12, "Privacy by design: 100 % synthetic data · no PII · patient-level train/val/test split · "
            "DPDP Act / HIPAA sandbox compliant", ha="center", fontsize=9, color="#555")
    ax.text(0.5, 0.07, "Stack: Python · NumPy · pandas · scikit-learn · Streamlit · Plotly", ha="center",
            fontsize=9, color="#555")


def text_slide(pdf, title, bullets, sub=None):
    fig = plt.figure(figsize=(13.33, 7.5)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.88), 1, 0.12, color=TEAL))
    ax.text(0.04, 0.94, title, fontsize=26, weight="bold", color="white", va="center")
    if sub:
        ax.text(0.04, 0.82, sub, fontsize=14, color="#555", style="italic")
    y = 0.74
    for b in bullets:
        indent = b.startswith("  ")
        ax.text(0.07 if indent else 0.05, y, ("–  " if indent else "•  ") + b.strip(),
                fontsize=13 if indent else 16, color=INK, va="top", wrap=True)
        y -= 0.065 if indent else 0.08
    pdf.savefig(fig); plt.close(fig)


def image_slide(pdf, title, img, caption=""):
    fig = plt.figure(figsize=(13.33, 7.5)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.88), 1, 0.12, color=TEAL))
    ax.text(0.04, 0.94, title, fontsize=26, weight="bold", color="white", va="center")
    if Path(img).exists():
        iax = fig.add_axes([0.08, 0.1, 0.84, 0.74]); iax.imshow(mpimg.imread(img)); iax.axis("off")
    ax.text(0.5, 0.04, caption, ha="center", fontsize=12, color="#555")
    pdf.savefig(fig); plt.close(fig)


def table_slide(pdf, title, rows, cols, note=""):
    fig = plt.figure(figsize=(13.33, 7.5)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.88), 1, 0.12, color=TEAL))
    ax.text(0.04, 0.94, title, fontsize=24, weight="bold", color="white", va="center")
    tb = ax.table(cellText=rows, colLabels=cols, bbox=[0.03, 0.34, 0.94, 0.48], cellLoc="center",
                  colWidths=[0.24] + [0.76 / (len(cols) - 1)] * (len(cols) - 1))
    tb.auto_set_font_size(False); tb.set_fontsize(11)
    ax.text(0.04, 0.12, note, fontsize=13, color=INK, va="bottom")
    pdf.savefig(fig); plt.close(fig)


def rows_of(res):
    return [[k, f"{v['auroc']:.3f}", f"{v['per_patient_auroc']:.3f}", f"{v['auprc']:.3f}", v["events"],
             f"{v['events_warned_pct']}%", f"{v['median_lead_min']:.0f}", f"{v['false_alerts_per_day']}"]
            for k, v in res.items() if isinstance(v, dict)]


COLS = ["Model", "AUROC", "Per-patient\nAUROC", "AUPRC", "Events", "Events\nwarned", "Median\nlead (min)",
        "False\nalerts/day"]


def main():
    m = json.loads((ROOT / "artifacts" / "metrics.json").read_text())
    c, real, syn = m["cohort"], m["real"], m["synthetic"]
    rl = real["low"][real["low"]["dashboard_uses"]]
    rs = real["spike"][real["spike"]["dashboard_uses"]]

    fig = plt.figure(figsize=(13.33, 7.5)); architecture(fig.add_axes([0.02, 0.02, 0.96, 0.96]))
    fig.savefig(DOCS / "architecture.pdf"); fig.savefig(DOCS / "figures" / "architecture.png", dpi=140); plt.close(fig)

    with PdfPages(DOCS / "presentation.pdf") as pdf:
        fig = plt.figure(figsize=(13.33, 7.5)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=TEAL))
        ax.text(0.06, 0.62, "GlucoTwin", fontsize=54, weight="bold", color="white")
        ax.text(0.06, 0.50, "A hybrid Digital Twin that warns of dangerous glucose LOWS and HIGHS\n"
                "2 hours ahead, validated on 100 real patients with Type 2 Diabetes", fontsize=20, color="white")
        ax.text(0.06, 0.25, f"{TEAM}  ·  {COLLEGE}\nHappiest Health Digital Twin Challenge 2026",
                fontsize=15, color="white")
        pdf.savefig(fig); plt.close(fig)
        text_slide(pdf, "The problem", [
            "India has ~101 million people living with diabetes (ICMR-INDIAB, 2023)",
            "Dangerous LOWS (<70 mg/dL) from insulin & sulfonylureas cause falls, seizures, ER visits",
            "Repeated HIGHS (>180 mg/dL) after rice-heavy meals drive long-term complications",
            "Care is reactive: HbA1c every 3 months, occasional glucometer checks",
            "CGMs & smartwatches produce data every few minutes, but nobody turns it into foresight",
        ], "Reactive diabetes care misses the moments that matter")
        text_slide(pdf, "Our solution: a hybrid Digital Twin", [
            "Fuses the static EHR (labs, medicines, genetics) with live CGM, meal, insulin & smartwatch data",
            "Mechanistic twin: learns each patient's own baseline, carb response and clearance",
            "Two calibrated ML models re-score LOW and HIGH risk every 15 minutes",
            "Explainable: shows which data stream drives each alert",
            "What-if simulator tests lifestyle advice on the twin before it reaches the patient",
            "Validated on 100 REAL patients (ShanghaiT2DM), not just synthetic data",
        ], "Predict -> explain -> simulate -> act")
        image_slide(pdf, "Architecture", DOCS / "figures" / "architecture.png")
        text_slide(pdf, "Data: real + synthetic, privacy-safe", [
            f"REAL: ShanghaiT2DM, {c['real_people']} people / {c['real_recordings']} recordings "
            "(Scientific Data 2023, CC BY 4.0)",
            "  CGM every 15 min, meals with food weights, insulin & oral drugs, labs, complications",
            f"SYNTHETIC: {c['synthetic_patients']} virtual Indian adults x {c['synthetic_days']} days",
            "  adds smartwatch streams (steps, heart rate, HRV, sleep) the real data lacks",
            "  physiology simulator: Indian meals, insulin/sulfonylurea, skipped meals, sleep debt, stress",
            "No identifiable data; patient-level splits so no patient is in both training and testing",
        ])
        table_slide(pdf, f"{c['real_people']} REAL patients: LOW glucose (<70) within 2 h", rows_of(real["low"]), COLS,
                    f"The deployed model warns of {rl['events_warned_pct']}% of real lows, median {rl['median_lead_min']:.0f} min "
                    f"ahead ({rl['warned_30min_plus_pct']}% with 30+ min warning).\n"
                    f"{c['cv_folds']}-fold cross-validation grouped by patient: every prediction comes from a model "
                    "that never saw that patient.")
        table_slide(pdf, f"{c['real_people']} REAL patients: HIGH glucose (>180) within 2 h", rows_of(real["spike"]),
                    COLS,
                    f"The deployed model warns of {rs['events_warned_pct']}% of real spikes, median "
                    f"{rs['median_lead_min']:.0f} min ahead.\n"
                    "A model trained on synthetic data alone transfers poorly to real patients: "
                    "that is why we validate on real data.")
        image_slide(pdf, "Real-patient validation", DOCS / "figures" / "real_validation.png")
        table_slide(pdf, "Synthetic cohort: what does each data stream add? (LOW)", rows_of(syn["low"]), COLS,
                    "Per-patient AUROC shows whether a model knows WHEN an event comes, not just WHO is at risk.")
        image_slide(pdf, "Which data streams matter?", DOCS / "figures" / "feature_group_importance.png",
                    "Drop in AUROC when a feature group is randomly shuffled (synthetic test patients)")
        image_slide(pdf, "Clinician dashboard", DOCS / "figures" / "dashboard.png",
                    "Virtual patient · LOW & HIGH risk timeline · explainable drivers · what-if simulator · triage")
        text_slide(pdf, "Healthcare impact", [
            "Doctors: a worklist of who is about to go low or high, not 3-monthly snapshots",
            "Patients: just-in-time nudges ('eat a 15 g snack now', 'walk 15 min after lunch')",
            "Health systems: scalable remote monitoring for tier-2/3 India via low-cost CGMs",
            "Safety-first: medication changes are always left to the clinician",
        ])
        text_slide(pdf, "Limitations & roadmap", [
            "Real validation cohort is Chinese and mostly hospitalised: next, validate on Indian CGM cohorts",
            "Real data has no smartwatch stream: the wearable benefit is shown on synthetic data only",
            "Carbs estimated from food weights: next, use the Indian Food Composition Tables (IFCT)",
            "Next: FHIR / ABDM integration, on-device inference, a gestational-diabetes twin",
            "Clinical pilot with endocrinologists; CDSCO SaMD pathway",
        ])
    print("Wrote", DOCS / "architecture.pdf", "and", DOCS / "presentation.pdf")


if __name__ == "__main__":
    main()
