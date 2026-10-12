"""End-to-end GlucoTwin pipeline.

    python run_pipeline.py            # full run (400 synthetic patients + 100 real patients)
    python run_pipeline.py --quick    # smoke test (100 synthetic patients, 2 CV folds)

1. Simulate a synthetic cohort (EHR + CGM + smartwatch) and load the real ShanghaiT2DM cohort.
2. Calibrate a twin for every patient and fuse features on a shared 15-min grid.
3. Synthetic cohort: train spike & low models, ablate data streams, evaluate on held-out patients.
4. Real cohort: patient-grouped 5-fold cross-validation of three strategies
   (synthetic-only transfer, real-only, synthetic + real).
5. Export metrics, figures and a bundle for the dashboard.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupKFold

from glucotwin import features as F
from glucotwin import model as M
from glucotwin.real_data import load_shanghai
from glucotwin.simulator import generate_cohort, to_records

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
DATA, ART, FIG = ROOT / "data", ROOT / "artifacts", ROOT / "docs" / "figures"
REAL_ROOT = DATA / "real_shanghai_raw"
SYN_CALIB, REAL_CALIB = 5 * 96, 3 * 96
TARGETS = ["spike", "low"]
REAL_WEIGHT = 3.0          # up-weight real rows when mixing with synthetic ones


def fit(df: pd.DataFrame, feats: list[str], target: str, seed: int, weight=None):
    m = df.index[M.at_risk_rows(df, target)]
    w = None if weight is None else np.asarray(weight)[df.index.get_indexer(m)]
    return M.make_model(seed).fit(df.loc[m, feats], M.label(df.loc[m], target), sample_weight=w)


def predict(model, df, feats):
    return model.predict_proba(df[feats])[:, 1]


def threshold_on(model, df, feats, target):
    m = M.at_risk_rows(df, target)
    return M.best_f1_threshold(M.label(df[m], target), predict(model, df[m], feats))


def evaluate_split(model, df, feats, thr, target, p=None):
    p = predict(model, df, feats) if p is None else p
    m = M.at_risk_rows(df, target).to_numpy()
    return M.evaluate(df[m], p[m], thr, target, df_all=df, p_all=p)


def main(n: int, folds: int, seed: int):
    for p in (DATA, ART, FIG):
        p.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ 1. data
    print(f"[1/5] Simulating {n} synthetic patients x 14 days; loading real ShanghaiT2DM ...")
    cohort = generate_cohort(n, 14, seed)
    cohort["ehr"].to_csv(DATA / "ehr_synthetic.csv", index=False)
    syn = to_records(cohort)
    sample = pd.concat([r.df.assign(patient_id=r.pid, timestamp=[r.time(t) for t in range(len(r.df))])
                        for r in syn[:5]])
    sample.drop(columns=["meal_text"]).to_csv(DATA / "wearables_sample_5_patients.csv", index=False)
    real = load_shanghai(REAL_ROOT)
    print(f"   synthetic: {len(syn)} patients | real: {len(real)} recordings from "
          f"{len({r.pid.split('-')[1] for r in real})} people")

    # ----------------------------------------------------------- 2. features
    print("[2/5] Calibrating twins & fusing features ...")
    ids = rng.permutation(len(syn))
    split = {"train": ids[: int(0.7 * n)], "val": ids[int(0.7 * n): int(0.85 * n)], "test": ids[int(0.85 * n):]}
    S = {k: F.build_dataset([syn[i] for i in sorted(v)], SYN_CALIB) for k, v in split.items()}
    R = F.build_dataset(real, REAL_CALIB)
    R["group"] = R.pid.str.split("-").str[1]
    print("   rows:", {k: len(v) for k, v in S.items()}, "| real:", len(R))

    metrics = {"cohort": {"synthetic_patients": n, "synthetic_days": 14, "real_recordings": len(real),
                          "real_people": int(R.group.nunique()), "horizon_min": F.HORIZON * 15,
                          "high_mgdl": F.HIGH, "low_mgdl": F.LOW, "cv_folds": folds,
                          "split": {k: len(v) for k, v in split.items()}},
               "synthetic": {t: {} for t in TARGETS}, "real": {t: {} for t in TARGETS}, "importance": {}}
    syn_models, curves = {}, {t: {} for t in TARGETS}

    # ---------------------------------------------------- 3. synthetic models
    print("[3/5] Synthetic cohort: training & ablations ...")
    for target in TARGETS:
        for name, feats in F.FEATURE_SETS.items():
            mdl = fit(S["train"], feats, target, seed)
            thr = threshold_on(mdl, S["val"], feats, target)
            p = predict(mdl, S["test"], feats)
            res = evaluate_split(mdl, S["test"], feats, thr, target, p)
            metrics["synthetic"][target][name] = res
            m = M.at_risk_rows(S["test"], target).to_numpy()
            curves[target][name] = roc_curve(M.label(S["test"][m], target), p[m])[:2]
            if name == "GlucoTwin (fused + twin)":
                syn_models[target] = (mdl, feats, thr, p)
            print(f"   {target:5s} {name:32s} AUROC={res['auroc']:.3f} perPatient={res['per_patient_auroc']:.3f} "
                  f"warned={res['events_warned_pct']}% lead={res['median_lead_min']:.0f}m FA/day={res['false_alerts_per_day']}")
        mdl, feats, _, _ = syn_models[target]
        m = M.at_risk_rows(S["test"], target)
        metrics["importance"][target] = M.group_permutation_importance(
            mdl, S["test"][m], M.label(S["test"][m], target), feats, rng).round(4).to_dict()

    # ------------------------------------------------- 4. real-patient validation
    print(f"[4/5] Real patients: {folds}-fold patient-grouped cross-validation ...")
    feats = F.REAL_FEATURES
    syn_train = pd.concat([S["train"], S["val"]], ignore_index=True)
    syn_train = syn_train.sample(min(len(syn_train), 120_000), random_state=seed)   # keep real data influential
    fold_of = {g: k for k, (_, te) in enumerate(GroupKFold(folds).split(R, groups=R.group))
               for g in R.group.iloc[te].unique()}
    R["fold"] = R.group.map(fold_of)
    real_models = {t: {} for t in TARGETS}
    for target in TARGETS:
        transfer = fit(S["train"], feats, target, seed)
        t_thr = threshold_on(transfer, S["val"], feats, target)
        names = ["Trained on synthetic only", "Trained on real patients", "Trained on synthetic + real"]
        probs = {k: np.zeros(len(R)) for k in names}
        scores = {k: np.zeros(len(R)) for k in names}          # probability shifted so 0.5 = alert
        probs[names[0]] = predict(transfer, R, feats)
        scores[names[0]] = probs[names[0]] - t_thr + 0.5
        for k in range(folds):
            tr, te = R[R.fold != k], R[R.fold == k]
            groups = tr.group.unique().copy(); rng.shuffle(groups)
            inner = tr.group.isin(groups[: int(0.8 * len(groups))]).to_numpy()
            for name in names[1:]:
                def train_on(frame: pd.DataFrame):
                    frame = frame.reset_index(drop=True)
                    if name == "Trained on real patients":
                        return fit(frame, feats, target, seed)
                    cols = list(dict.fromkeys(feats + ["g_now", f"label_{target}"]))
                    mix = pd.concat([syn_train[cols], frame[cols]], ignore_index=True)
                    w = np.r_[np.ones(len(syn_train)), np.full(len(frame), REAL_WEIGHT)]
                    return fit(mix, feats, target, seed, weight=w)
                # inner model on 80 % of training patients -> calibrate + pick threshold on the other 20 %
                hold = tr[~inner]; hold = hold[M.at_risk_rows(hold, target)]
                inner_mdl = train_on(tr[inner])
                p_hold, y_hold = predict(inner_mdl, hold, feats), M.label(hold, target)
                thr = M.best_f1_threshold(y_hold, predict(M.Calibrated(inner_mdl, p_hold, y_hold), hold, feats))
                mdl = M.Calibrated(train_on(tr), p_hold, y_hold)
                p = predict(mdl, te, feats)
                probs[name][te.index] = p
                scores[name][te.index] = p - thr + 0.5
                real_models[target].setdefault(name, [None] * folds)[k] = (mdl, thr)
        m = M.at_risk_rows(R, target).to_numpy()
        y = M.label(R[m], target)
        for name in names:
            res = M.evaluate(R[m], probs[name][m], 0.5, target, df_all=R, p_all=scores[name])
            a = scores[name][m] >= 0.5                       # alerts at each fold's own threshold
            res.update(sensitivity=round(float((a & (y == 1)).sum() / max(y.sum(), 1)), 3),
                       ppv=round(float((a & (y == 1)).sum() / max(a.sum(), 1)), 3), threshold="chosen per fold")
            metrics["real"][target][name] = res
            print(f"   {target:5s} {name:36s} AUROC={res['auroc']:.3f} perPatient={res['per_patient_auroc']:.3f} "
                  f"warned={res['events_warned_pct']}% lead={res['median_lead_min']:.0f}m FA/day={res['false_alerts_per_day']}")
        # the dashboard uses whichever cross-validated strategy scored best on real patients
        best = max(names[1:], key=lambda k: metrics["real"][target][k]["auprc"])
        metrics["real"][target]["dashboard_uses"] = best
        real_models[target] = real_models[target][best]
        R[f"risk_{target}"] = probs[best]
        R[f"thr_{target}"] = [real_models[target][k][1] for k in R.fold]
        print(f"   -> dashboard uses '{best}' for {target}")
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ------------------------------------------------------------ 5. export
    print("[5/5] Exporting dashboard bundle & figures ...")
    test_ids = sorted(split["test"])
    syn_rows = S["test"].copy()
    for t in TARGETS:
        syn_rows[f"risk_{t}"] = syn_models[t][3]
        syn_rows[f"thr_{t}"] = syn_models[t][2]
    joblib.dump({
        "synthetic": {"records": [syn[i] for i in test_ids], "rows": syn_rows,
                      "models": {t: syn_models[t][:3] for t in TARGETS},
                      "reference": S["train"][F.ALL_FEATURES].median(), "calib_steps": SYN_CALIB},
        "real": {"records": real, "rows": R, "models": real_models, "features": feats,
                 "reference": pd.concat([syn_train[feats], R[feats]]).median(), "calib_steps": REAL_CALIB},
        "metrics": metrics,
    }, ART / "dashboard_bundle.joblib", compress=3)
    make_figures(curves, metrics)
    print("Done ->", ART / "metrics.json")


def make_figures(curves, metrics):
    nice = {"spike": "High glucose (>180) within 2 h", "low": "Low glucose (<70) within 2 h"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for a, t in zip(ax, TARGETS):
        for name, (fpr, tpr) in curves[t].items():
            a.plot(fpr, tpr, label=f"{name} ({metrics['synthetic'][t][name]['auroc']:.2f})")
        a.plot([0, 1], [0, 1], ":", c="grey")
        a.set(title=f"Synthetic test patients - {nice[t]}", xlabel="False positive rate", ylabel="Sensitivity")
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "synthetic_roc.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for a, t in zip(ax, TARGETS):
        res = {k: v for k, v in metrics["real"][t].items() if isinstance(v, dict)}
        names = list(res)
        x = np.arange(len(names))
        a.bar(x - 0.2, [res[k]["auroc"] for k in names], 0.4, label="AUROC", color="#1f6f8b")
        a.bar(x + 0.2, [res[k]["per_patient_auroc"] for k in names], 0.4, label="Per-patient AUROC", color="#66a182")
        a.set_xticks(x, [k.replace(", ", ",\n").replace(": ", ":\n") for k in names], fontsize=8)
        a.set_ylim(0.5, 1); a.set_title(f"100 REAL patients - {nice[t]}"); a.legend(fontsize=8); a.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(FIG / "real_validation.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for a, t in zip(ax, TARGETS):
        pd.Series(metrics["importance"][t]).sort_values().plot.barh(ax=a, color="#d1495b" if t == "spike" else "#30638e")
        a.set(title=nice[t], xlabel="Drop in AUROC when the group is shuffled")
    fig.suptitle("Which data streams matter? (synthetic test patients)")
    fig.tight_layout(); fig.savefig(FIG / "feature_group_importance.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    main(n=100 if a.quick else 400, folds=2 if a.quick else 5, seed=a.seed)
