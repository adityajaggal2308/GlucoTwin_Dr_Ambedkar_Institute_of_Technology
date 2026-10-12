"""Redraw docs/figures/real_validation.png from artifacts/metrics.json (no retraining needed)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
m = json.loads((ROOT / "artifacts" / "metrics.json").read_text())
nice = {"spike": "High glucose (>180) within 2 h", "low": "Low glucose (<70) within 2 h"}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for a, t in zip(ax, ["spike", "low"]):
    res = {k: v for k, v in m["real"][t].items() if isinstance(v, dict)}
    names = list(res)
    x = np.arange(len(names))
    a.bar(x - 0.2, [res[k]["auroc"] for k in names], 0.4, label="AUROC", color="#1f6f8b")
    a.bar(x + 0.2, [res[k]["per_patient_auroc"] for k in names], 0.4, label="Per-patient AUROC", color="#66a182")
    a.set_xticks(x, [k.replace("Trained on ", "Trained on\n") for k in names], fontsize=9)
    a.set_ylim(0.5, 1); a.set_title(f"{m['cohort']['real_people']} REAL patients - {nice[t]}")
    a.legend(fontsize=8); a.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(ROOT / "docs" / "figures" / "real_validation.png", dpi=160)
print("saved")
