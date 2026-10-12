"""Loader for the real ShanghaiT2DM dataset (Zhao et al., Scientific Data 10:35, 2023, CC BY 4.0).

100 real people with Type 2 diabetes (109 recordings): 15-min CGM, meals (English food
names + weights), insulin and oral drugs, plus clinical/lab summary. Converted here into the
shared PatientRecord format. Only the English columns are used.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .records import PatientRecord, WEARABLE_COLS, empty_grid

# Approximate available carbohydrate per gram of food (food-composition tables, rounded).
CARB_PER_G = [
    ("fried rice", 0.30), ("rice in soup", 0.12), ("porridge", 0.10), ("congee", 0.10),
    ("rice cake", 0.45), ("black rice", 0.28), ("rice", 0.28),
    ("steamed bread", 0.47), ("steamed bun", 0.45), ("bun", 0.45), ("bread", 0.48), ("pancake", 0.40),
    ("noodle", 0.25), ("dumpling", 0.25), ("wonton", 0.22), ("cereal", 0.55), ("oat", 0.55),
    ("coarse grain", 0.30), ("biscuit", 0.70), ("cracker", 0.70), ("cake", 0.50),
    ("meal replacement", 0.45), ("sweet potato leaves", 0.04), ("sweet potato", 0.20),
    ("potato", 0.17), ("taro", 0.18), ("yam", 0.20), ("corn", 0.19), ("pumpkin", 0.07),
    ("lotus root", 0.15), ("red date", 0.60), ("watermelon", 0.08), ("banana", 0.23), ("grape", 0.17),
    ("apple", 0.14), ("pear", 0.13), ("peach", 0.11), ("orange", 0.11), ("grapefruit", 0.09),
    ("kiwi", 0.14), ("fruit", 0.12), ("soybean milk", 0.03), ("yogurt", 0.10), ("milk", 0.05),
    ("beer", 0.04), ("rice wine", 0.04), ("broad bean", 0.15), ("peanut", 0.16), ("walnut", 0.10),
]
DEFAULT_CARB_PER_G = 0.05            # vegetables, meat, eggs, fish, tofu ...
UNKNOWN_MEAL_CARBS = 55.0            # meal time recorded but "data not available"

BASAL_INSULINS = ("degludec", "glargine", "detemir")
SULFONYLUREAS = ("gliclazide", "glimepiride", "glipizide", "gliquidone", "glibenclamide")


def meal_carbs(text: str) -> float:
    total, known = 0.0, False
    for line in str(text).lower().split("\n"):
        m = re.search(r"([\d.]+)\s*(g|ml)\b", line)
        if not m:
            continue
        known = True
        grams = float(m.group(1))
        frac = next((f for k, f in CARB_PER_G if k in line), DEFAULT_CARB_PER_G)
        total += grams * frac
    return round(total, 1) if known else UNKNOWN_MEAL_CARBS


def insulin_units(text: str) -> tuple[float, float]:
    """Split a free-text s.c. insulin entry into (bolus/premixed units, basal units)."""
    bolus = basal = 0.0
    for part in re.split(r"(?<=IU)\s*,?\s*", str(text)):
        m = re.search(r"([\d.]+)\s*IU", part)
        if not m:
            continue
        if any(b in part.lower() for b in BASAL_INSULINS):
            basal += float(m.group(1))
        else:
            bolus += float(m.group(1))
    return bolus, basal


def _col(df: pd.DataFrame, *keys: str):
    for c in df.columns:
        norm = c.replace(" ", "").lower()
        if all(k in norm for k in keys):
            return c
    return None


def _num(v) -> float:
    return float(pd.to_numeric(v, errors="coerce"))


def _static(row: pd.Series) -> tuple[dict, dict]:
    agents = str(row.get("Hypoglycemic Agents", "")).lower()
    hba1c_mmol = _num(row["HbA1c (mmol/mol)"])
    static = {
        "age": _num(row["Age (years)"]),
        "sex_m": float(row["Gender (Female=1, Male=2)"] == 2),
        "bmi": _num(row["BMI (kg/m2)"]),
        "years_since_dx": _num(row["Duration of diabetes (years)"]),
        "hba1c_pct": round(hba1c_mmol / 10.929 + 2.15, 1),
        "fasting_glucose_mgdl": _num(row["Fasting Plasma Glucose (mg/dl)"]),
        "egfr": _num(row[[c for c in row.index if c.startswith("Estimated Glomerular")][0]]),
        "triglycerides_mgdl": _num(row["Triglyceride (mmol/L)"]) * 88.57,
        "ldl_mgdl": _num(row["Low-Density Lipoprotein Cholesterol (mmol/L)"]) * 38.67,
        "metformin": float("metformin" in agents),
        "sulfonylurea": float(any(s in agents for s in SULFONYLUREAS)),
        "insulin": float(any(s in agents for s in ("insulin", "novolin", "humulin", "gansulin", "lantus",
                                                     "levemir", "toujeo", "novorapid", "humalog", "tresiba"))),
        "hypertension": float("hypertension" in str(row.get("Comorbidities", "")).lower()),
        "smoker": float(_num(row["Smoking History (pack year)"]) > 0),
        "diabetes_status": "T2D",
        "meds_text": str(row.get("Hypoglycemic Agents", "")),
    }
    info = {k: str(row.get(c, "")) for k, c in [
        ("Macrovascular complications", "Diabetic Macrovascular  Complications"),
        ("Microvascular complications", "Diabetic Microvascular Complications"),
        ("Comorbidities", "Comorbidities"), ("Other drugs", "Other Agents"),
        ("Fasting C-peptide (nmol/L)", "Fasting C-peptide (nmol/L)"),
        ("Glycated albumin (%)", "Glycated Albumin (%)")]}
    return static, info


def load_shanghai(root: str | Path) -> list[PatientRecord]:
    root = Path(root)
    summary = pd.read_excel(root / "Shanghai_T2DM_Summary.xlsx")
    summary["Patient Number"] = summary["Patient Number"].astype(str).str.strip()
    records = []
    for f in sorted((root / "Shanghai_T2DM").glob("*.xls*")):
        key = f.stem
        srow = summary[summary["Patient Number"] == key]
        if srow.empty:
            continue
        static, info = _static(srow.iloc[0])
        raw = pd.read_excel(f)
        ts = pd.to_datetime(raw[_col(raw, "date")])
        start = ts.min().normalize()
        step = ((ts - start).dt.total_seconds() // 900).astype(int).to_numpy()
        df = empty_grid(int(step.max()) + 1)
        glucose = pd.to_numeric(raw[_col(raw, "cgm")], errors="coerce").to_numpy()
        df.loc[step, "glucose"] = glucose
        df["glucose"] = df.glucose.interpolate(limit=2, limit_area="inside")   # bridge <=30-min gaps

        diet_col = _col(raw, "dietary")
        for s, txt in zip(step, raw[diet_col]):
            if isinstance(txt, str) and txt.strip():
                df.loc[s, "carbs"] += meal_carbs(txt)
                df.loc[s, "meal"] = True
                df.loc[s, "meal_text"] = txt.replace("\n", ", ")
        sc_col, csii_col = _col(raw, "insulindose-s.c"), _col(raw, "csii-bolus")
        for s, txt in zip(step, raw[sc_col]):
            if isinstance(txt, str) and "IU" in txt:
                b, ba = insulin_units(txt)
                df.loc[s, "insulin_bolus_u"] += b; df.loc[s, "insulin_basal_u"] += ba
        if csii_col is not None:
            for s, v in zip(step, pd.to_numeric(raw[csii_col], errors="coerce")):
                if np.isfinite(v):
                    df.loc[s, "insulin_bolus_u"] += v
        oral_col = _col(raw, "non-insulin")
        info["Oral drugs during monitoring"] = ", ".join(sorted({str(x).strip() for x in raw[oral_col].dropna()}))[:200]
        records.append(PatientRecord(pid=f"SH-{key.split('_')[0]}-{key.split('_')[1]}", source="real",
                                     static=static, start=start, df=df, info=info))
    return records
