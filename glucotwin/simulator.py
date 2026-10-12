"""Synthetic data generator for GlucoTwin.

Produces two fused data streams for an Indian adult cohort (normoglycaemic,
prediabetic and Type 2 diabetic):

1. Static / historical EHR  - demographics, diagnoses, medications, labs and a
   genetic marker (TCF7L2 rs7903146 risk-allele count).
2. Dynamic wearable / IoT   - 5-minute CGM glucose, heart rate, HRV (RMSSD),
   step count and sleep stage, plus meals logged in a phone app.

Every patient has hidden physiological parameters (insulin sensitivity,
beta-cell function, carb response). The EHR is *derived* from those hidden
parameters, and the glucose trace is produced by a compartment model driven
by meals, activity, sleep debt, stress, dawn phenomenon and medication
adherence. No real patient data is used anywhere (DPDP Act / HIPAA safe).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STEP_MIN = 5
SPD = 24 * 60 // STEP_MIN  # samples per day (288)

SLEEP_STAGES = {0: "Awake", 1: "Awake in bed", 2: "Light", 3: "Deep", 4: "REM"}


def gamma_kernel(shape: float, scale_min: float, length_min: int = 300) -> np.ndarray:
    """Carbohydrate appearance kernel (fraction of a meal absorbed per 5-min step)."""
    t = np.arange(0, length_min, STEP_MIN) + STEP_MIN / 2
    k = t ** (shape - 1) * np.exp(-t / scale_min)
    return k / k.sum()


K_FAST = gamma_kernel(2.5, 12.0)   # refined carbs: white rice, sweets, chai
K_SLOW = gamma_kernel(3.5, 18.0)   # mixed meals: dal, roti, vegetables
K_DRUG = gamma_kernel(2.5, 55.0, 480)  # glucose-lowering action of insulin / sulfonylurea


# --------------------------------------------------------------------------- EHR
def generate_ehr(n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ehr, hidden) dataframes. `hidden` is never shown to the model."""
    status = rng.choice(["T2D", "Prediabetes", "Normoglycaemic"], size=n, p=[0.65, 0.28, 0.07])
    t2d, pre = status == "T2D", status == "Prediabetes"

    age = np.where(t2d, rng.integers(35, 72, n), rng.integers(25, 65, n))
    sex = rng.choice(["M", "F"], n)
    bmi = np.clip(rng.normal(np.where(status == "Normoglycaemic", 23.0, 26.5), 3.2), 18, 40).round(1)
    waist = np.round(bmi * 3.4 + np.where(sex == "M", 4, 0) + rng.normal(0, 4, n))
    tcf7l2 = rng.binomial(2, np.where(t2d, 0.35, 0.25))
    family_hx = rng.random(n) < np.where(t2d, 0.6, np.where(pre, 0.45, 0.25))
    years_dx = np.where(t2d, rng.integers(0, 20, n), 0)
    metformin = t2d & (rng.random(n) < 0.85)
    sulfonylurea = t2d & (rng.random(n) < 0.35)
    insulin = t2d & (years_dx >= 6) & (rng.random(n) < 0.5)
    on_meds = metformin | sulfonylurea | insulin
    adherence = np.where(on_meds, rng.beta(8, 2, n), 1.0)
    hypertension = rng.random(n) < (0.15 + 0.006 * (age - 25) + 0.12 * t2d)
    smoker = rng.random(n) < np.where(sex == "M", 0.22, 0.04)

    # hidden physiology
    si = (np.exp(rng.normal(0, 0.22, n)) * np.select([t2d, pre], [0.62, 0.85], 1.15)
          * (1 - 0.025 * (bmi - 23)) * (1 - 0.07 * tcf7l2))
    beta = np.clip(np.select([t2d, pre], [0.78 - 0.02 * years_dx, 0.9], 1.0)
                   * np.exp(rng.normal(0, 0.12, n)), 0.25, 1.3)
    med = adherence * (0.15 * metformin + 0.10 * sulfonylurea + 0.30 * insulin)
    clearance = 0.022 * np.clip(si, 0.2, 2) * beta            # 1/min
    carb_gain = 1.25 / beta ** 0.9 * np.exp(rng.normal(0, 0.12, n))
    basal = np.maximum(84 + 55 * np.clip(1 - si * beta, 0, None) + rng.normal(0, 6, n), 84)
    dawn = np.where(t2d, rng.uniform(8, 22, n), rng.uniform(0, 6, n))
    rest_hr = rng.normal(70, 6, n) + 3 * t2d + 2 * smoker
    hrv_base = np.clip(48 - 0.45 * (age - 30) - 6 * t2d + rng.normal(0, 6, n), 12, None)

    ehr = pd.DataFrame({
        "patient_id": [f"P{i:04d}" for i in range(n)],
        "age": age, "sex": sex, "bmi": bmi, "waist_cm": waist,
        "diabetes_status": status, "years_since_dx": years_dx,
        "family_history_t2d": family_hx.astype(int), "tcf7l2_risk_alleles": tcf7l2,
        "metformin": metformin.astype(int), "sulfonylurea": sulfonylurea.astype(int),
        "insulin": insulin.astype(int), "hypertension": hypertension.astype(int),
        "smoker": smoker.astype(int),
        # labs are filled in after simulation (HbA1c reflects simulated glucose)
    })
    hidden = pd.DataFrame({
        "patient_id": ehr.patient_id, "si": si, "beta": beta, "med_effect": med,
        "adherence": adherence, "clearance": clearance, "carb_gain": carb_gain,
        "basal": basal, "dawn_amp": dawn, "rest_hr": rest_hr, "hrv_base": hrv_base,
        "log_prob": rng.uniform(0.55, 0.95, n),             # how diligently meals are logged
        "daily_steps": rng.uniform(2500, 10000, n),
        "morning_walk_p": rng.uniform(0, 0.7, n),
        "postmeal_walk_p": rng.uniform(0.05, 0.5, n),
        "sleep_habit_h": np.clip(rng.normal(6.8, 0.6, n), 5, 8.5),
        "stress_p": rng.uniform(0.05, 0.25, n),
        "carb_appetite": rng.uniform(0.8, 1.3, n),
        "meal_shift_h": rng.normal(0, 0.9, n),              # early/late eaters, shift workers
        "routine_jitter_h": rng.uniform(0.3, 1.2, n),       # how regular the routine is
    })
    return ehr, hidden


# --------------------------------------------------------------- wearable stream
def _meals_for_day(h: pd.Series, rng: np.random.Generator, feast: bool, weekend: bool):
    """Indian meal pattern: breakfast, lunch, evening chai-snack, dinner, late snack."""
    plan = [  # (start_h, end_h, carbs_lo, carbs_hi, prob, fast_frac)
        (7.5, 9.5, 35, 75, 0.95, 0.4),
        (12.5, 14.5, 60, 115, 0.98, 0.55),
        (16.5, 18.0, 15, 45, 0.6, 0.8),
        (20.0, 22.0, 50, 110, 0.97, 0.5),
        (22.5, 23.3, 10, 30, 0.12, 0.8),
    ]
    out = []
    for kind, (lo, hi, clo, chi, p, ff) in enumerate(plan):
        if rng.random() < p:
            c = rng.uniform(clo, chi) * h.carb_appetite * (1.4 if feast else 1.0) * (1.1 if weekend else 1.0)
            hour = (lo + hi) / 2 + h.meal_shift_h + 0.8 * weekend + rng.normal(0, h.routine_jitter_h)
            out.append((float(np.clip(hour, 5.5, 23.8)), c, np.clip(ff + rng.normal(0, 0.15), 0.1, 0.9), kind))
    return out


def simulate_streams(ehr: pd.DataFrame, hidden: pd.DataFrame, days: int,
                     rng: np.random.Generator) -> dict[str, np.ndarray]:
    n, T = len(ehr), days * SPD
    carbs_fast = np.zeros((n, T)); carbs_slow = np.zeros((n, T))
    carbs_true = np.zeros((n, T)); carbs_logged = np.zeros((n, T))
    steps = np.zeros((n, T)); stage = np.zeros((n, T), dtype=np.int8)
    stress = np.zeros((n, days)); sleep_h = np.zeros((n, days)); missed = np.zeros((n, days))
    drug_dose = np.zeros((n, T)); insulin_u = np.zeros((n, T))
    on_insulin = ehr.insulin.to_numpy().astype(bool)
    on_drug = on_insulin | ehr.sulfonylurea.to_numpy().astype(bool)
    # glucose-lowering strength per dose (mg/dL-equivalent), recorded insulin units scale with it
    strength = np.where(on_insulin, rng.uniform(130, 230, n), rng.uniform(80, 140, n)) * on_drug

    for i in range(n):
        h = hidden.iloc[i]
        for d in range(days):
            base = d * SPD
            # ---- sleep (night ending on morning of day d)
            dur = np.clip(rng.normal(h.sleep_habit_h, 0.7) - (1.6 if rng.random() < 0.12 else 0), 3.5, 9.5)
            sleep_h[i, d] = dur
            wake = rng.normal(6.6, 0.4)
            onset = wake - dur - 0.15
            s0, s1 = int(base + onset * 12), int(base + wake * 12)
            for t in range(max(s0, 0), s1):
                pos = ((t - s0) * 5) % 90 / 90           # position in 90-min cycle
                frac = (t - s0) / max(s1 - s0, 1)        # deep sleep front-loaded
                r = rng.random()
                if r < 0.06:
                    stage[i, t] = 1
                elif pos < 0.45 and r < 0.06 + 0.45 * (1 - frac):
                    stage[i, t] = 3
                elif pos > 0.7 and r < 0.06 + 0.25 + 0.35 * frac:
                    stage[i, t] = 4
                else:
                    stage[i, t] = 2
            # ---- day context
            stress[i, d] = rng.random() < h.stress_p
            missed[i, d] = rng.random() > h.adherence
            feast = rng.random() < 0.08
            # ---- meals
            for hour, c, ff, kind in _meals_for_day(h, rng, feast, weekend=d % 7 in (5, 6)):
                if on_drug[i] and kind in (0, 3) and not missed[i, d]:
                    # pre-meal dose; sometimes the meal is then skipped, small or delayed -> hypo risk
                    td = int(base + (hour - 0.25) * 12)
                    drug_dose[i, td] += strength[i] * rng.uniform(0.85, 1.15)
                    if on_insulin[i]:
                        insulin_u[i, td] += round(strength[i] / 12 * rng.uniform(0.9, 1.1))
                    r = rng.random()
                    if r < 0.06:
                        continue                                     # dosed but did not eat
                    if r < 0.14:
                        c *= 0.4                                     # ate much less than planned
                    elif r < 0.24:
                        hour = min(hour + rng.uniform(0.75, 1.5), 23.8)  # meal delayed after dose
                t = int(base + hour * 12)
                carbs_fast[i, t] += c * ff; carbs_slow[i, t] += c * (1 - ff); carbs_true[i, t] += c
                if rng.random() < h.log_prob:  # app logging is imperfect & noisy
                    carbs_logged[i, t] += round(c * np.exp(rng.normal(0, 0.25)))
                # post-meal walk
                if hour > 12 and rng.random() < h.postmeal_walk_p:
                    w0 = t + rng.integers(2, 8); wl = rng.integers(3, 6)
                    seg = steps[i, w0:w0 + wl]; seg += rng.normal(480, 60, len(seg))
            # ---- activity
            if rng.random() < h.morning_walk_p:
                w0 = int(base + rng.uniform(6.7, 7.3) * 12); wl = rng.integers(4, 9)
                seg = steps[i, w0:w0 + wl]; seg += rng.normal(520, 70, len(seg))
            awake = np.arange(s1, base + int(23 * 12))
            lam = h.daily_steps * 0.75 / max(len(awake), 1)
            burst = rng.random(len(awake)) < 0.15
            steps[i, awake] += rng.poisson(lam * np.where(burst, 4.5, 0.4))
    asleep = stage >= 1
    steps = np.where(asleep, 0, np.clip(steps, 0, None)).round()

    # ---- glucose compartment model, vectorised over patients
    ra = np.stack([np.convolve(carbs_fast[i], K_FAST)[:T] + np.convolve(carbs_slow[i], K_SLOW)[:T]
                   for i in range(n)])                                # g absorbed per step
    day_idx = np.arange(T) // SPD
    tod = (np.arange(T) % SPD) / 12.0
    sleep_debt = np.clip(7 - sleep_h, 0, None)
    day_factor = (1 - 0.07 * sleep_debt - 0.15 * stress) * (1 + hidden.med_effect.values[:, None] * (1 - missed))
    basal_day = hidden.basal.values[:, None] + 9 * stress + 14 * missed * (hidden.med_effect.values[:, None] > 0)
    dawn = hidden.dawn_amp.values[:, None] * np.exp(-0.5 * ((tod - 6.5) / 1.1) ** 2)

    k = hidden.clearance.values; g = hidden.carb_gain.values
    drug = np.stack([np.convolve(drug_dose[i], K_DRUG)[:T] for i in range(n)])
    floor = np.where(on_drug, -150.0, np.minimum(-5.0, 78 - hidden.basal.to_numpy()))
    E = np.zeros(n); act = np.zeros(n)
    glucose = np.zeros((n, T)); noise = np.zeros(n)
    for t in range(T):
        d = day_idx[t]
        act = 0.7 * act + 0.3 * steps[:, t] / 500
        k_t = k * np.clip(day_factor[:, d], 0.4, 2) * (1 + 0.9 * act)
        E = E + g * ra[:, t] - STEP_MIN * k_t * E - 1.2 * act - drug[:, t] * (1 + 0.6 * act)
        # counter-regulation (glucagon): without glucose-lowering drugs the body stops a fall early
        E = np.maximum(E, floor)
        noise = 0.85 * noise + rng.normal(0, 2.2, n)
        glucose[:, t] = basal_day[:, d] + dawn[:, t] + E + noise
    glucose = np.clip(glucose + rng.normal(0, 1.5, glucose.shape), 40, 400).round(0)

    # ---- heart rate & HRV
    stress_t = stress[:, day_idx]; debt_t = sleep_debt[:, day_idx]
    hr = (hidden.rest_hr.values[:, None] + 0.055 * steps + 6 * stress_t
          - 9 * asleep + 0.04 * np.clip(glucose - 140, 0, None)
          + 0.35 * np.clip(75 - glucose, 0, None)                    # adrenergic response to a low
          + rng.normal(0, 2.5, (n, T)))
    hrv = (hidden.hrv_base.values[:, None] * (1 - 0.25 * stress_t - 0.05 * debt_t)
           * (1 + 0.25 * (stage == 3) + 0.1 * asleep - 0.004 * steps / 10)
           * (1 - 0.006 * np.clip(75 - glucose, 0, None))
           * np.exp(rng.normal(0, 0.15, (n, T))))
    return {
        "glucose": glucose, "hr": hr.round(0), "hrv": np.clip(hrv, 5, 150).round(1),
        "steps": steps, "sleep_stage": stage, "carbs_logged": carbs_logged,
        "carbs_true": carbs_true, "stress_day": stress, "sleep_h": sleep_h, "insulin_u": insulin_u,
    }


def add_labs(ehr: pd.DataFrame, hidden: pd.DataFrame, streams: dict, rng) -> pd.DataFrame:
    """Historical labs consistent with the simulated physiology (with lab noise)."""
    g = streams["glucose"]
    mean_g = g.mean(axis=1)
    night = g.reshape(len(ehr), -1, SPD)[:, :, 36:48].mean(axis=(1, 2))   # 3-4 am
    ehr = ehr.copy()
    ehr["hba1c_pct"] = np.round((mean_g + 46.7) / 28.7 + rng.normal(0, 0.25, len(ehr)), 1)
    ehr["fasting_glucose_mgdl"] = np.round(night + rng.normal(0, 7, len(ehr)))
    ehr["triglycerides_mgdl"] = np.round(110 + 90 * (1 - np.clip(hidden.si, 0, 1.3)) + rng.normal(0, 25, len(ehr)))
    ehr["ldl_mgdl"] = np.round(rng.normal(112, 28, len(ehr)))
    ehr["egfr"] = np.round(np.clip(118 - 0.8 * ehr.age - 6 * (ehr.diabetes_status == "T2D")
                                   + rng.normal(0, 9, len(ehr)), 25, 130))
    ehr["systolic_bp"] = np.round(118 + 0.35 * (ehr.age - 30) + 12 * ehr.hypertension + rng.normal(0, 8, len(ehr)))
    return ehr


def generate_cohort(n: int = 400, days: int = 14, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    ehr, hidden = generate_ehr(n, rng)
    streams = simulate_streams(ehr, hidden, days, rng)
    ehr = add_labs(ehr, hidden, streams, rng)
    return {"ehr": ehr, "hidden": hidden, "streams": streams, "days": days}


def nightly_sleep(stage: np.ndarray, hr: np.ndarray, hrv: np.ndarray, days: int) -> pd.DataFrame:
    """Summaries of the night that ends on the morning of each day (5-min data)."""
    rows = []
    for d in range(days):
        lo, hi = max(d * SPD - 72, 0), d * SPD + 144          # 18:00 previous day .. 12:00
        st = stage[lo:hi]; asleep = st >= 2
        rows.append({
            "sleep_h": asleep.sum() * 5 / 60,
            "deep_frac": (st == 3).sum() / max(asleep.sum(), 1),
            "waso_min": (st == 1).sum() * 5,
            "hrv_night": float(hrv[lo:hi][asleep].mean()) if asleep.any() else np.nan,
            "rhr_night": float(hr[lo:hi][asleep].mean()) if asleep.any() else np.nan,
        })
    return pd.DataFrame(rows)


def to_records(cohort: dict, idx=None) -> list:
    """Convert simulated patients to the shared 15-minute PatientRecord format."""
    from .records import PatientRecord, ehr_static_from_synthetic
    s, ehr, days = cohort["streams"], cohort["ehr"], cohort["days"]
    idx = range(len(ehr)) if idx is None else idx
    start = pd.Timestamp("2026-09-01")
    out = []
    for i in idx:
        q = lambda x, f: getattr(x.reshape(-1, 3), f)(axis=1)        # 5-min -> 15-min
        sleep = nightly_sleep(s["sleep_stage"][i], s["hr"][i], s["hrv"][i], days)
        T15 = days * SPD // 3
        slot = np.arange(T15) % 96
        day = np.arange(T15) // 96
        night_day = np.clip(np.where(slot >= 40, day, day - 1), 0, None)   # known after 10:00
        df = pd.DataFrame({
            "glucose": s["glucose"][i][::3].astype(float),
            "carbs": q(s["carbs_logged"][i], "sum"),
            "insulin_bolus_u": q(s["insulin_u"][i], "sum"),
            "insulin_basal_u": 0.0,
            "steps": q(s["steps"][i], "sum"),
            "hr": q(s["hr"][i], "mean").round(1),
            "hrv": q(s["hrv"][i], "mean").round(1),
            "sleep_stage": q(s["sleep_stage"][i].astype(float), "max"),
            "sleep_h_last": sleep.sleep_h.to_numpy()[night_day],
            "deep_frac_last": sleep.deep_frac.to_numpy()[night_day],
            "waso_min_last": sleep.waso_min.to_numpy()[night_day],
            "hrv_night_last": sleep.hrv_night.to_numpy()[night_day],
            "rhr_night_last": sleep.rhr_night.to_numpy()[night_day],
            "sleep_h_7d": sleep.sleep_h.expanding().mean().to_numpy()[night_day],
        })
        df["meal"] = df.carbs > 0
        df["meal_text"] = ""
        out.append(PatientRecord(pid=ehr.patient_id[i], source="synthetic",
                                 static=ehr_static_from_synthetic(ehr.iloc[i]), start=start, df=df))
    return out
