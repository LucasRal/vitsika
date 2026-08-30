"""Temperature-scaled linear probe: the deployed classifier (data/probe.pkl).

Wraps a fitted multinomial LogisticRegression and rescales its logits by a
single temperature T before the softmax. Ranking is unchanged by
construction (a scalar on the logits cannot reorder classes); only the
probabilities move. T is fitted in scripts/07_eval.py by minimising the
negative log-likelihood of out-of-fold train logits (5-fold CV), never on
the test split. Lives in api/ so the API process can unpickle it.
"""
from __future__ import annotations

import numpy as np


class TemperatureScaledProbe:
    def __init__(self, base, temperature: float):
        self.base = base
        self.temperature = float(temperature)
        self.classes_ = base.classes_

    def logits(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.base.decision_function(x), dtype=np.float64)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = self.logits(x) / self.temperature
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(x).argmax(axis=1)]

    def __repr__(self) -> str:
        return f"TemperatureScaledProbe(T={self.temperature:.3f}, base={type(self.base).__name__})"


def fit_temperature(logits: np.ndarray, y_idx: np.ndarray) -> float:
    """Scalar T minimising NLL of softmax(logits / T); golden-section search on
    log T in [1/50, 50] (the NLL is convex in 1/T)."""
    def nll(t: float) -> float:
        z = logits / t
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-logp[np.arange(len(y_idx)), y_idx].mean())
    lo, hi = np.log(1 / 50), np.log(50)
    phi = (np.sqrt(5) - 1) / 2
    a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fa, fb = nll(np.exp(a)), nll(np.exp(b))
    for _ in range(80):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - phi * (hi - lo); fa = nll(np.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + phi * (hi - lo); fb = nll(np.exp(b))
    return float(np.exp((lo + hi) / 2))
