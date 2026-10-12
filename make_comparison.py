"""Build the healthy-vs-diabetes comparison data used by the dashboard (~2 min, no training).

    python make_comparison.py

Re-creates the same synthetic cohort as run_pipeline.py (same seed) plus the real
ShanghaiT2DM patients, and stores per-patient summaries, daily glucose profiles and
glucose samples per group in artifacts/comparison.joblib.
"""
import argparse
import warnings
from pathlib import Path

import joblib

from glucotwin.compare import build
from glucotwin.real_data import load_shanghai
from glucotwin.simulator import generate_cohort, to_records

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    syn = to_records(generate_cohort(a.n, 14, a.seed))
    real = load_shanghai(ROOT / "data" / "real_shanghai_raw")
    comp = build(syn, real)
    joblib.dump(comp, ROOT / "artifacts" / "comparison.joblib", compress=3)
    print(comp["summary"].groupby("group").size().to_string())
    print(comp["summary"].groupby("group")[["Mean glucose (mg/dL)", "Time in range 70-180 (%)",
                                             "Lows per day", "HbA1c (%)"]].median().round(1).to_string())
