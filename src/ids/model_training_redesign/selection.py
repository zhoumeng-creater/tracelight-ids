"""Pure ranking selection; validation never constructs a decision threshold."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


def _metric(scores, labels):
    if not len(labels) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Each ranking panel requires both labels")
    return {"normal": int(np.sum(labels == 0)), "positive": int(np.sum(labels == 1)),
            "pauc": float(roc_auc_score(labels, scores, max_fpr=0.01)),
            "nll": float(np.logaddexp(0.0, np.where(labels == 1, -scores, scores)).mean()),
            "strictly_separated": bool(scores[labels == 1].min() > scores[labels == 0].max())}


def selection_metrics(scores, frame):
    values = np.asarray(scores, dtype=np.float64)
    labels = frame.label.to_numpy()
    if values.shape != labels.shape or not np.isfinite(values).all():
        raise ValueError("Expected finite row-aligned validation scores")
    panels = {}
    for name, profile, protocol in (("A_TCP", "A", 6), ("B_UDP", "B", 17)):
        mask = (frame.profile.to_numpy() == profile) & (frame.ip_protocol.to_numpy() == protocol)
        panels[name] = _metric(values[mask], labels[mask])
    pooled = _metric(values, labels)
    cells = {str(cell): _metric(values[mask], labels[mask]) for cell in sorted(frame.source_id.unique())
             for mask in [frame.source_id.to_numpy() == cell]}
    return {"panels": panels, "pooled": pooled, "cells": cells,
            "strictly_separated": pooled["strictly_separated"] and all(
                p["strictly_separated"] for p in panels.values())}


def selection_key(metrics, step):
    a, b = (metrics["panels"][name]["pauc"] for name in ("A_TCP", "B_UDP"))
    return min(a, b), metrics["pooled"]["pauc"], (a + b) / 2, -int(step)


@dataclass
class Selector:
    patience: int = 16
    best_key: tuple | None = None
    best_step: int | None = None
    stale: int = 0
    first_separated_step: int | None = None

    def update(self, metrics, step):
        key = selection_key(metrics, step)
        saturated = bool(metrics["strictly_separated"])
        first_separation = saturated and self.first_separated_step is None
        improved = first_separation or self.best_key is None or key > self.best_key
        if improved:
            self.best_key, self.best_step, self.stale = key, int(step), 0
        else:
            self.stale += 1
        if first_separation:
            self.first_separated_step = int(step)
        reason = "strict_full_validation_separation" if saturated else (
            "ranking_patience" if self.stale >= self.patience else None)
        return {"improved": improved, "stop": reason is not None, "stop_reason": reason,
                "selected_step": self.best_step, "stale": self.stale, "key": list(key),
                "replacement_reason": "first_strict_full_separation" if first_separation else (
                    "ranking_improved" if improved else "ranking_not_improved")}
