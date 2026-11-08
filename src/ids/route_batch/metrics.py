"""One strict normal-only rank rule for all registered log-odds score interfaces."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc


def scores(value):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError("scores must be a finite one-dimensional array")
    return result


def threshold(normal_scores, alpha=.01):
    x = scores(normal_scores)
    a = Fraction(str(alpha))
    if not 0 < a < 1 or len(x) == 0:
        raise ValueError("normal calibration and 0 < alpha < 1 are required")
    order = (len(x) + 1) * (1 - a)
    rank = (order.numerator + order.denominator - 1) // order.denominator
    value = None if rank > len(x) else float(np.partition(x, rank - 1)[rank - 1])
    return {"alpha": float(a), "n_normal": len(x), "rank": rank, "threshold": value,
            "comparison": ">", "comparison_dtype": "float64",
            "calibration_fp": 0 if value is None else int((x > value).sum()),
            "scope": "development_normal_calibration"}


def decisions(x, registration):
    values = scores(x)
    t = registration["threshold"]
    return np.zeros(len(values), dtype=bool) if t is None else values > t


def evaluate(truth, values, registration, families=None):
    x, y = scores(values), np.asarray(truth)
    if y.shape != x.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("aligned binary labels required")
    prediction = decisions(x, registration)
    tp, fp = int(prediction[y == 1].sum()), int(prediction[y == 0].sum())
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    result = {"tp": tp, "fp": fp, "fn": n1-tp, "tn": n0-fp,
              "attack_rows": n1, "normal_rows": n0,
              "tpr": tp/n1 if n1 else None, "fpr": fp/n0 if n0 else None,
              "auc": None, "normalized_pauc_0_01": None}
    if n1 and n0:
        fpr, tpr, _ = roc_curve(y, x)
        cut = .01
        stop = int(np.searchsorted(fpr, cut, side="right"))
        clipped_fpr, clipped_tpr = fpr[:stop], tpr[:stop]
        if clipped_fpr[-1] < cut:
            clipped_fpr = np.r_[clipped_fpr, cut]
            clipped_tpr = np.r_[clipped_tpr, np.interp(cut, fpr[stop-1:stop+1], tpr[stop-1:stop+1])]
        result["auc"] = float(roc_auc_score(y, x))
        result["normalized_pauc_0_01"] = float(auc(clipped_fpr, clipped_tpr) / cut)
    if families is not None:
        family = np.asarray(families).astype(str)
        if family.shape != y.shape:
            raise ValueError("families must align with labels")
        result["families"] = {name: {"rows": int(((family == name) & (y == 1)).sum()),
            "tp": int((prediction & (family == name) & (y == 1)).sum())}
            for name in sorted(set(family[y == 1]))}
    return result


def paired(truth, old, new):
    y, left, right = np.asarray(truth), np.asarray(old, dtype=bool), np.asarray(new, dtype=bool)
    if not (y.shape == left.shape == right.shape):
        raise ValueError("paired decisions must align")
    return {"recovered_fn": int(((y == 1) & ~left & right).sum()),
            "new_fn": int(((y == 1) & left & ~right).sum()),
            "repaired_fp": int(((y == 0) & left & ~right).sum()),
            "new_fp": int(((y == 0) & ~left & right).sum())}
