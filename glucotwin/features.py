"""Fuse the static EHR and the dynamic stream of one PatientRecord into model-ready rows.

One row = one moment in time (every 15 min). Two labels:
  * label_spike - glucose crosses ABOVE 180 mg/dL within the next 2 h
  * label_low   - glucose falls BELOW 70 mg/dL within the next 2 h
Wearable features are NaN for real patients (the ShanghaiT2DM data has no smartwatch);
gradient-boosted trees handle missing values natively.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import twin
from .records import STATIC_COMMON, STATIC_SYNTHETIC_ONLY, PatientRecord

HORIZON = 8               # 2 h ahead (8 x 15 min)
HIGH, LOW = 180, 70       # ADA time-in-range thresholds (mg/dL)
TARGETS = {"spike": ("label_spike", "High (>180) within 2 h"), "low": ("label_low", "Low (<70) within 2 h")}

FEATURE_GROUPS = {
    "Glucose trend (CGM)": ["g_now", "g_d15", "g_d30", "g_d60", "g_mean_2h", "g_std_2h", "g_max_2h",
                            "g_min_2h", "g_mean_24h", "tar_24h", "tbr_24h", "g_min_24h"],
    "Meals": ["carbs_60m", "carbs_120m", "cob_g", "mins_since_meal", "carbs_last_meal"],
    "Insulin doses": ["bolus_u_4h", "basal_u_24h", "mins_since_bolus"],
    "Daily routine": ["tod_sin", "tod_cos", "meal_prior_2h"],
    "Twin physiology": ["twin_basal", "twin_gain", "twin_k", "twin_peak_per_g", "twin_expected_rise",
                        "g_minus_basal"],
    "Activity & heart (wearable)": ["steps_30m", "steps_60m", "steps_today", "hr_30m", "hr_rise_30m", "hrv_60m"],
    "Sleep & recovery (wearable)": ["sleep_h_last", "deep_frac_last", "waso_min_last", "hrv_night_last",
                                    "rhr_night_last", "sleep_h_7d"],
    "EHR profile": STATIC_COMMON + STATIC_SYNTHETIC_ONLY,
}
ALL_FEATURES = [f for fs in FEATURE_GROUPS.values() for f in fs]
WEARABLE = FEATURE_GROUPS["Activity & heart (wearable)"] + FEATURE_GROUPS["Sleep & recovery (wearable)"]
EHR_ALL = FEATURE_GROUPS["EHR profile"]
CGM_DYNAMIC = [f for g in ["Glucose trend (CGM)", "Meals", "Insulin doses", "Daily routine"] for f in FEATURE_GROUPS[g]]
TWIN = FEATURE_GROUPS["Twin physiology"]

# Ablations on the synthetic cohort: what does each data stream add?
FEATURE_SETS = {
    "EHR only (static)": EHR_ALL + ["tod_sin", "tod_cos"],
    "Wearables + CGM only (dynamic)": CGM_DYNAMIC + WEARABLE,
    "Fused EHR + dynamic": EHR_ALL + CGM_DYNAMIC + WEARABLE,
    "GlucoTwin (fused + twin)": ALL_FEATURES,
}
# Features that exist in BOTH synthetic and real data (used for real-patient validation)
REAL_FEATURES = CGM_DYNAMIC + TWIN + STATIC_COMMON


def _roll(x: np.ndarray, w: int, how: str) -> np.ndarray:
    return getattr(pd.Series(x).rolling(w, min_periods=1), how)().to_numpy()


def patient_features(rec: PatientRecord, calib_steps: int) -> tuple[pd.DataFrame, twin.TwinParams]:
    d = rec.df
    g = d.glucose.to_numpy(float); carbs = d.carbs.to_numpy(float); meal = d.meal.to_numpy(bool)
    bolus = d.insulin_bolus_u.to_numpy(float); basal_ins = d.insulin_basal_u.to_numpy(float)
    T = len(d)
    calib_steps = min(calib_steps, int(T * 0.4))
    params = twin.calibrate(g, carbs, calib_steps)
    prior = twin.meal_prior(meal, calib_steps)
    cob = twin.carbs_on_board(carbs)

    slot = np.arange(T) % twin.SPD
    idx = np.arange(T)
    last_meal = np.maximum.accumulate(np.where(meal, idx, -10 ** 6))
    last_bolus = np.maximum.accumulate(np.where(bolus > 0, idx, -10 ** 6))
    meal_size = pd.Series(np.where(meal, carbs, np.nan)).ffill().fillna(0).to_numpy()
    gs = pd.Series(g)
    lag = lambda k: g - gs.shift(k).to_numpy()

    # future labels (only where the next 2 h are observed)
    fut = np.lib.stride_tricks.sliding_window_view(np.append(g, [np.nan] * HORIZON), HORIZON + 1)[:, 1:]
    seen = np.isfinite(fut).sum(1) >= HORIZON - 2
    up = np.nanmax(np.where(np.isfinite(fut), fut, -1), axis=1) >= HIGH
    down = np.nanmin(np.where(np.isfinite(fut), fut, 999), axis=1) < LOW

    f = pd.DataFrame({
        "pid": rec.pid, "source": rec.source, "t": idx, "slot": slot,
        "g_now": g, "g_d15": lag(1), "g_d30": lag(2), "g_d60": lag(4),
        "g_mean_2h": _roll(g, 8, "mean"), "g_std_2h": _roll(g, 8, "std"), "g_max_2h": _roll(g, 8, "max"),
        "g_min_2h": _roll(g, 8, "min"), "g_mean_24h": _roll(g, 96, "mean"),
        "tar_24h": _roll((g > HIGH).astype(float), 96, "mean"), "tbr_24h": _roll((g < LOW).astype(float), 96, "mean"),
        "g_min_24h": _roll(g, 96, "min"),
        "carbs_60m": _roll(carbs, 4, "sum"), "carbs_120m": _roll(carbs, 8, "sum"), "cob_g": cob,
        "mins_since_meal": np.clip((idx - last_meal) * 15, 0, 720), "carbs_last_meal": meal_size,
        "bolus_u_4h": _roll(bolus, 16, "sum"), "basal_u_24h": _roll(basal_ins, 96, "sum"),
        "mins_since_bolus": np.clip((idx - last_bolus) * 15, 0, 1440),
        "tod_sin": np.sin(2 * np.pi * slot / twin.SPD), "tod_cos": np.cos(2 * np.pi * slot / twin.SPD),
        "meal_prior_2h": prior[slot],
        **{k: v for k, v in params.as_dict().items() if k != "n_meals_fit"},
        "twin_expected_rise": params.twin_peak_per_g * cob,
        "g_minus_basal": g - params.twin_basal,
    })
    if d.steps.notna().any():
        steps, hr, hrv = d.steps.to_numpy(float), d.hr.to_numpy(float), d.hrv.to_numpy(float)
        steps_today = pd.Series(steps).groupby(idx // twin.SPD).cumsum().to_numpy()
        f = f.assign(steps_30m=_roll(steps, 2, "sum"), steps_60m=_roll(steps, 4, "sum"), steps_today=steps_today,
                     hr_30m=_roll(hr, 2, "mean"), hr_rise_30m=_roll(hr, 2, "mean") - _roll(hr, 8, "mean"),
                     hrv_60m=_roll(hrv, 4, "mean"))
        for c in FEATURE_GROUPS["Sleep & recovery (wearable)"]:
            f[c] = d[c].to_numpy()
    else:
        for c in WEARABLE:
            f[c] = np.nan
    for k in STATIC_COMMON + STATIC_SYNTHETIC_ONLY:
        f[k] = rec.static.get(k, np.nan)

    f["label_spike"] = (up & (g < HIGH)).astype(int)
    f["label_low"] = (down & (g >= LOW)).astype(int)
    f["eligible"] = (idx >= calib_steps) & np.isfinite(g) & seen
    return f, params


def build_dataset(records: list[PatientRecord], calib_steps: int) -> pd.DataFrame:
    rows = [patient_features(r, calib_steps)[0] for r in records]
    df = pd.concat(rows, ignore_index=True)
    return df[df.eligible].reset_index(drop=True)
