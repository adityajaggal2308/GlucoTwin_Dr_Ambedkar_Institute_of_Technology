"""Risk models: gradient-boosted trees over fused EHR + CGM + wearable + twin features."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from .features import FEATURE_GROUPS, HIGH, HORIZON, LOW


def make_model(seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=25, random_state=seed)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))[:, None]


class Calibrated:
    """Model + Platt scaling fitted on held-out patients, so a displayed 30 % means ~30 %."""

    def __init__(self, model, p_holdout: np.ndarray, y_holdout: np.ndarray):
        self.model = model
        self.cal = LogisticRegression().fit(_logit(p_holdout), y_holdout)

    def predict_proba(self, X):
        return self.cal.predict_proba(_logit(self.model.predict_proba(X)[:, 1]))


def at_risk_rows(df: pd.DataFrame, target: str) -> pd.Series:
    """Rows where the prediction makes sense (not already high / already low)."""
    return df.g_now < HIGH if target == "spike" else df.g_now >= LOW


def label(df: pd.DataFrame, target: str) -> np.ndarray:
    return df[f"label_{target}"].to_numpy()


def best_f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(thr[np.nanargmax(f1[:-1])])


def per_patient_auroc(df: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> float:
    """Median AUROC computed *within* each patient: does the model know WHEN, not just WHO?"""
    scores = []
    for _, ix in df.groupby("pid").indices.items():
        if 0 < y[ix].sum() < len(ix):
            scores.append(roc_auc_score(y[ix], p[ix]))
    return round(float(np.median(scores)), 4) if scores else float("nan")


def event_metrics(df: pd.DataFrame, prob: np.ndarray, thr: float, target: str) -> dict:
    """Event-level view: did we warn before each episode, how early, how many false alarms?"""
    d = df[["pid", "t", "g_now"]].assign(p=prob).sort_values(["pid", "t"])
    leads, n_events, false_alerts, patient_days = [], 0, 0, 0.0
    for _, q in d.groupby("pid"):
        t, g, p = q.t.to_numpy(), q.g_now.to_numpy(), q.p.to_numpy()
        patient_days += len(t) / 96
        ok = g < HIGH if target == "spike" else g >= LOW
        alert = (p >= thr) & ok
        contiguous = np.r_[False, np.diff(t) == 1]
        if target == "spike":
            onset = np.flatnonzero(contiguous & (g >= HIGH) & np.r_[False, g[:-1] < HIGH])
        else:
            onset = np.flatnonzero(contiguous & (g < LOW) & np.r_[False, g[:-1] >= LOW])
        covered = np.zeros(len(t), bool)
        for o in onset:
            n_events += 1
            win = np.flatnonzero((t < t[o]) & (t >= t[o] - HORIZON))
            covered[win] = True
            hits = win[alert[win]]
            if len(hits):
                leads.append((t[o] - t[hits.min()]) * 15)
        starts = np.flatnonzero(alert & ~np.r_[False, alert[:-1]])
        false_alerts += int((~covered[starts]).sum())
    leads = np.array(leads)
    return {
        "events": n_events,
        "events_warned_pct": round(100 * len(leads) / max(n_events, 1), 1),
        "median_lead_min": float(np.median(leads)) if len(leads) else 0.0,
        "warned_30min_plus_pct": round(100 * (leads >= 30).sum() / max(n_events, 1), 1),
        "false_alerts_per_day": round(false_alerts / max(patient_days, 1), 2),
    }


def evaluate(df: pd.DataFrame, p_rows: np.ndarray, thr: float, target: str,
             df_all: pd.DataFrame | None = None, p_all: np.ndarray | None = None) -> dict:
    y = label(df, target)
    pred = p_rows >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    out = {"auroc": round(roc_auc_score(y, p_rows), 4), "auprc": round(average_precision_score(y, p_rows), 4),
           "per_patient_auroc": per_patient_auroc(df, y, p_rows), "threshold": round(thr, 3),
           "sensitivity": round(tp / max(tp + fn, 1), 3), "ppv": round(tp / max(tp + fp, 1), 3),
           "prevalence": round(float(y.mean()), 4), "rows": int(len(y))}
    if df_all is not None:
        out.update(event_metrics(df_all, p_all, thr, target))
    return out


def group_contributions(model, x_row: pd.DataFrame, reference: pd.Series, features: list[str]) -> pd.Series:
    """Local explanation: change in risk when each feature group is reset to the cohort median."""
    base = model.predict_proba(x_row[features])[0, 1]
    out = {}
    for group, cols in FEATURE_GROUPS.items():
        cols = [c for c in cols if c in features and pd.notna(x_row[c].iloc[0])]
        if not cols:
            continue
        x2 = x_row[features].copy()
        x2[cols] = reference[cols].values
        out[group] = base - model.predict_proba(x2)[0, 1]
    return pd.Series(out).sort_values(key=np.abs, ascending=False)


def group_permutation_importance(model, X: pd.DataFrame, y, features, rng, repeats=3) -> pd.Series:
    base = roc_auc_score(y, model.predict_proba(X[features])[:, 1])
    out = {}
    for group, cols in FEATURE_GROUPS.items():
        cols = [c for c in cols if c in features]
        if not cols:
            continue
        drops = []
        for _ in range(repeats):
            Xp = X[features].copy()
            Xp[cols] = Xp[cols].to_numpy()[rng.permutation(len(Xp))]
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        out[group] = float(np.mean(drops))
    return pd.Series(out).sort_values(ascending=False)
