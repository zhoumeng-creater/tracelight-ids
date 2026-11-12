"""Calibrate a common anchor and compare equal-budget additions on earlier normal data."""

import numpy as np
from ids.route_batch.metrics import decisions, scores, threshold

SINGLE = ("raw", "history", "service", "contact")
POLICIES = (*SINGLE, "anchor", "unrestricted_add", "contact_add")
GATE_COLUMNS = ("svc_prior_reply_fraction_60s", "contact_prior_reply_fraction_60s")
COMPARISONS = (
    ("history", "contact"),
    ("service", "contact"),
    ("raw", "history"),
    ("unrestricted_add", "contact_add"),
    ("raw", "contact_add"),
    ("history", "contact_add"),
    ("anchor", "contact_add"),
)


def contact_gate(frame):
    """A prior known zero-reply service flow and no observed reply to this endpoint."""
    service = frame[GATE_COLUMNS[0]].to_numpy()
    pair = frame[GATE_COLUMNS[1]].to_numpy()
    return (service >= 0) & (service < 1) & (pair <= 0)


def _calibrate_add(normal, eligible, allowance):
    pool = scores(normal)[eligible]
    fit = {"candidates": len(pool), "allowance": allowance, "threshold": None, "included_at_cut": False}
    if allowance <= 0 or not len(pool):
        return fit
    if len(pool) <= allowance:
        # Use the weakest actually observed eligible score as a floor even when room is left.
        fit.update(threshold=float(pool.min()), included_at_cut=True)
    else:
        fit["threshold"] = float(np.partition(pool, len(pool) - allowance - 1)[len(pool) - allowance - 1])
    return fit


def _addition(values, eligible, rule):
    values = scores(values)
    cut = rule["threshold"]
    if cut is None:
        return np.zeros(len(values), dtype=bool)
    return eligible & ((values >= cut) if rule["included_at_cut"] else (values > cut))


def register(cal_scores, gate):
    count = len(gate)
    if not count or any(len(scores(cal_scores[key])) != count for key in SINGLE):
        raise ValueError("Unaligned or empty calibration")
    reg = {key: threshold(cal_scores[key]) for key in SINGLE}
    reg["anchor"] = threshold(cal_scores["raw"], alpha=0.008)
    anchored = decisions(cal_scores["raw"], reg["anchor"])
    allowance = count // 100 - int(anchored.sum())
    for key, eligible in (("unrestricted_add", ~anchored), ("contact_add", ~anchored & gate)):
        reg[key] = _calibrate_add(cal_scores["contact"], eligible, allowance)
    output = apply(reg, cal_scores, gate)
    if any(((output >> i) & 1).sum() > count // 100 for i in range(len(POLICIES))):
        raise RuntimeError("Calibration budget violation")
    return reg


def apply(reg, values, gate):
    alerts = {key: decisions(values[key], reg[key]) for key in SINGLE}
    anchored = decisions(values["raw"], reg["anchor"])
    alerts["anchor"] = anchored
    for key, eligible in (("unrestricted_add", ~anchored), ("contact_add", ~anchored & gate)):
        alerts[key] = anchored | _addition(values["contact"], eligible, reg[key])
    flags = np.zeros(len(gate), dtype=np.uint8)
    for bit, key in enumerate(POLICIES):
        flags |= alerts[key].astype(np.uint8) << bit
    flags |= np.asarray(gate, dtype=np.uint8) << 7
    return flags
