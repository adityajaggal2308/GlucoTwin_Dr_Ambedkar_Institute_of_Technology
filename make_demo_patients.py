"""Add fresh, never-seen synthetic patients to the dashboard (no retraining).

    python make_demo_patients.py

The held-out synthetic test set is mostly people with diabetes (as in the training cohort),
so it contains very few healthy people. This script simulates a NEW cohort with a different
seed, picks a balanced set (healthy / prediabetes / T2D), scores it with the already-trained
synthetic models and appends it to artifacts/dashboard_bundle.joblib. These patients were never
used for training or threshold selection, so their risks are honest out-of-sample predictions.
"""
import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np

from glucotwin import features as F
from glucotwin.simulator import generate_cohort, to_records

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
BUNDLE = ROOT / "artifacts" / "dashboard_bundle.joblib"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=15)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    B = joblib.load(BUNDLE)
    syn = B["synthetic"]
    syn["records"] = [r for r in syn["records"] if not r.pid.startswith("N")]     # idempotent
    syn["rows"] = syn["rows"][~syn["rows"].pid.str.startswith("N")]

    cohort = generate_cohort(250, 14, a.seed)
    recs = to_records(cohort)
    picked = []
    for status in ["Normoglycaemic", "Prediabetes", "T2D"]:
        picked += [r for r in recs if r.static["diabetes_status"] == status][: a.per_group]
    for r in picked:
        r.pid = "N" + r.pid[1:]                     # N = new, never seen by any model

    rows = F.build_dataset(picked, syn["calib_steps"])
    for t in ["spike", "low"]:
        mdl, feats, thr = syn["models"][t]
        rows[f"risk_{t}"] = mdl.predict_proba(rows[feats])[:, 1]
        rows[f"thr_{t}"] = thr
    syn["records"] += picked
    syn["rows"] = (rows if syn["rows"].empty else
                   __import__("pandas").concat([syn["rows"], rows], ignore_index=True))
    joblib.dump(B, BUNDLE, compress=3)
    print(f"added {len(picked)} new patients:",
          {s: sum(r.static["diabetes_status"] == s for r in picked) for s in ["Normoglycaemic", "Prediabetes", "T2D"]})
