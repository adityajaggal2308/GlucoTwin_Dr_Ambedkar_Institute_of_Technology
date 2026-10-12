"""Shared patient format so synthetic and real patients flow through one pipeline.

Every patient becomes a PatientRecord:
  * static - the EHR snapshot (same keys for both sources; missing = NaN)
  * df     - a 15-minute grid starting at midnight with columns
             glucose, carbs, meal, meal_text, insulin_bolus_u, insulin_basal_u,
             and (wearables, synthetic only) steps, hr, hrv, sleep_stage, sleep_* summaries
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

WEARABLE_COLS = ["steps", "hr", "hrv", "sleep_stage", "sleep_h_last", "deep_frac_last",
                 "waso_min_last", "hrv_night_last", "rhr_night_last", "sleep_h_7d"]

STATIC_COMMON = ["age", "sex_m", "bmi", "years_since_dx", "hba1c_pct", "fasting_glucose_mgdl", "egfr",
                 "triglycerides_mgdl", "ldl_mgdl", "metformin", "sulfonylurea", "insulin",
                 "hypertension", "smoker"]
STATIC_SYNTHETIC_ONLY = ["waist_cm", "family_history_t2d", "tcf7l2_risk_alleles", "systolic_bp", "status_code"]


@dataclass
class PatientRecord:
    pid: str
    source: str                      # "synthetic" | "real"
    static: dict
    start: pd.Timestamp              # midnight of the first day
    df: pd.DataFrame
    info: dict = field(default_factory=dict)   # display-only details (diagnoses, drug names ...)

    def time(self, t: int) -> pd.Timestamp:
        return self.start + pd.Timedelta(minutes=15 * int(t))


def ehr_static_from_synthetic(row: pd.Series) -> dict:
    st = {k: float(row[k]) for k in STATIC_COMMON if k in row and k != "sex_m"}
    st["sex_m"] = float(row.sex == "M")
    for k in STATIC_SYNTHETIC_ONLY:
        st[k] = float({"Normoglycaemic": 0, "Prediabetes": 1, "T2D": 2}[row.diabetes_status]) \
            if k == "status_code" else float(row[k])
    st["diabetes_status"] = row.diabetes_status
    st["meds_text"] = ", ".join(m for m in ["metformin", "sulfonylurea", "insulin"] if row[m]) or "none"
    return st


def empty_grid(n: int) -> pd.DataFrame:
    df = pd.DataFrame({"glucose": np.full(n, np.nan), "carbs": 0.0, "meal": False, "meal_text": "",
                       "insulin_bolus_u": 0.0, "insulin_basal_u": 0.0})
    for c in WEARABLE_COLS:
        df[c] = np.nan
    return df
