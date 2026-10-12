"""Mechanistic layer of the Digital Twin (works on the shared 15-minute grid).

During a calibration window (first days of wear) each patient's twin learns:
  * twin_basal      - personal overnight baseline glucose (mg/dL)
  * twin_gain       - glucose appearance per gram of logged carbohydrate
  * twin_k          - personal glucose clearance rate (1/min)
  * twin_peak_per_g - resulting post-meal peak rise per gram of carbs
  * meal_prior      - personal probability of eating in the next 2 h, per time of day

These parameters feed the ML risk models (hybrid twin) and power the
"what-if" simulator in the clinician dashboard.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

STEP_MIN = 15
SPD = 24 * 60 // STEP_MIN      # 96 steps per day
H_LEN = 20                     # 5 h impulse response
FIT_LEN = 12                   # fit the first 3 h after a meal
K_GRID = np.geomspace(0.003, 0.06, 28)
WALK_EFFECT = {0: 1.0, 10: 0.85, 15: 0.78, 30: 0.65}   # population prior on post-meal walking


def gamma_kernel(shape: float, scale_min: float, length_min: int = 300, step: int = STEP_MIN) -> np.ndarray:
    """Fraction of a meal absorbed in each time step (gamma-shaped gut absorption)."""
    t = np.arange(0, length_min, step) + step / 2
    k = t ** (shape - 1) * np.exp(-t / scale_min)
    return k / k.sum()


K_POP = gamma_kernel(3.0, 15.0)


def impulse_response(k: float, gain: float = 1.0, length: int = H_LEN) -> np.ndarray:
    """Glucose rise (mg/dL) after 1 g carbs: gamma absorption -> first-order clearance."""
    ra = np.zeros(length); n = min(len(K_POP), length); ra[:n] = K_POP[:n]
    e, out = 0.0, np.zeros(length)
    decay = np.exp(-k * STEP_MIN)
    for t in range(length):
        e = e * decay + gain * ra[t]
        out[t] = e
    return out


_BASE_IR = {k: impulse_response(k) for k in K_GRID}


@dataclass
class TwinParams:
    twin_basal: float
    twin_gain: float
    twin_k: float
    twin_peak_per_g: float
    n_meals_fit: int

    def as_dict(self):
        return asdict(self)


POP_DEFAULT = TwinParams(130.0, 2.0, 0.012, 1.4, 0)


def calibrate(glucose: np.ndarray, carbs: np.ndarray, calib_steps: int) -> TwinParams:
    """Fit personal parameters from the calibration window using isolated logged meals."""
    g, c = glucose[:calib_steps], carbs[:calib_steps]
    slot = np.arange(len(g)) % SPD
    night = g[(slot >= 8) & (slot < 16)]                        # 02:00-04:00
    basal = float(np.nanmedian(night)) if np.isfinite(night).any() else float(np.nanmedian(g))
    if not np.isfinite(basal):
        basal = POP_DEFAULT.twin_basal

    meals = np.flatnonzero(c > 0)
    curves, grams = [], []
    for m in meals:
        if m < 2 or m + FIT_LEN >= len(g):
            continue
        if ((meals != m) & (meals > m - 12) & (meals < m + FIT_LEN)).any():
            continue                                             # not an isolated meal
        win = g[m + 1:m + 1 + FIT_LEN]; pre = g[m - 1:m + 1]
        if np.isnan(win).any() or np.isnan(pre).any():
            continue
        curves.append(win - pre.mean()); grams.append(c[m])
    if len(curves) < 2:
        return TwinParams(basal, POP_DEFAULT.twin_gain, POP_DEFAULT.twin_k, POP_DEFAULT.twin_peak_per_g, len(curves))

    Y, C = np.array(curves), np.array(grams)[:, None]
    best = (np.inf, POP_DEFAULT.twin_k, POP_DEFAULT.twin_gain)
    for k, ir in _BASE_IR.items():
        X = C * ir[1:FIT_LEN + 1][None, :]
        gain = float(np.clip((X * Y).sum() / (X * X).sum(), 0.05, 12))
        sse = float(((Y - gain * X) ** 2).sum())
        if sse < best[0]:
            best = (sse, k, gain)
    _, k, gain = best
    return TwinParams(basal, gain, float(k), float(gain * _BASE_IR[k].max()), len(curves))


def meal_prior(meal: np.ndarray, calib_steps: int) -> np.ndarray:
    """P(a logged meal within the next 2 h) for each 15-min slot of the day."""
    days = max(calib_steps // SPD, 1)
    eat = np.zeros(SPD)
    m = meal[: days * SPD].astype(bool)
    if len(m) < days * SPD:
        m = np.pad(m, (0, days * SPD - len(m)))
    eat = m.reshape(days, SPD).mean(axis=0)
    ahead = np.array([eat[np.arange(s + 1, s + 9) % SPD].sum() for s in range(SPD)])
    return np.clip(ahead, 0, 1)


def carbs_on_board(carbs: np.ndarray) -> np.ndarray:
    """Logged carbohydrate (g) not yet absorbed at each step."""
    remaining = np.concatenate([[1.0], 1 - np.cumsum(K_POP)[:-1]])
    return np.convolve(carbs, remaining)[: len(carbs)]


def forecast(params: TwinParams, g_now: float, recent_meals: list[tuple[int, float]],
             planned_meal: tuple[int, float] | None = None, walk_min: int = 0,
             horizon: int = 12) -> np.ndarray:
    """What-if forward simulation of glucose for the next `horizon` 15-min steps.

    recent_meals: (steps_ago, grams) already eaten; planned_meal: (steps_from_now, grams).
    """
    ir = impulse_response(params.twin_k, params.twin_gain, H_LEN + horizon + 24)
    s = np.arange(1, horizon + 1)
    at = lambda idx: np.where((idx >= 0) & (idx < len(ir)), ir[np.clip(idx, 0, len(ir) - 1)], 0.0)
    pipeline_now = sum(c * at(np.array(a)) for a, c in recent_meals)
    future = sum((c * at(a + s) for a, c in recent_meals), np.zeros(horizon))
    excess = g_now - params.twin_basal - pipeline_now          # unexplained deviation decays
    traj = params.twin_basal + future + excess * np.exp(-params.twin_k * STEP_MIN * s)
    if planned_meal:
        start, grams = planned_meal
        traj = traj + grams * WALK_EFFECT.get(walk_min, 1.0) * at(s - start)
    return traj
