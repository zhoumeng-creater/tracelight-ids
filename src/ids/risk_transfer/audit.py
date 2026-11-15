"""Audit six fixed rules using only the R26 return's scores and historical calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

from ids.route_batch.protocol import REPO

REGISTRATION = REPO / "configs/risk-transfer-audit-20261115/registration.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def order_rule(scores, *, alpha=.01, delta=None, preserve=None):
    """Fixed nested score thresholds; NP rank controls selection via an order statistic.

    For an immutable old decision B, Z=+inf if B else score. A finite
    threshold realizes B OR score>t. Infinite Z_(k) is explicitly infeasible.
    Strict > with ties is conservative; no per-threshold confidence selection.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Calibration scores must be a nonempty finite vector")
    if not 0 < alpha < 1 or (delta is not None and not 0 < delta < 1):
        raise ValueError("Invalid fixed risk/confidence level")
    base = np.zeros(len(scores), bool) if preserve is None else np.asarray(preserve, dtype=bool)
    if base.shape != scores.shape:
        raise ValueError("Preserved decisions must align with calibration rows")
    n = len(scores)
    if delta is None:
        k = int(np.ceil((n + 1) * (1 - alpha)))
        tail = None
    else:
        ranks = np.arange(1, n + 1)
        admissible = ranks[binom.sf(ranks - 1, n, 1 - alpha) <= delta]
        k = int(admissible[0]) if len(admissible) else n + 1
        tail = float(binom.sf(k - 1, n, 1 - alpha)) if k <= n else None
    z = np.where(base, np.inf, scores)
    cutoff = float(np.partition(z, k - 1)[k - 1]) if k <= n else float("inf")
    ready = bool(np.isfinite(cutoff))
    alerts = base | (scores > cutoff)
    return {"n_normal": n, "alpha": alpha, "delta": delta, "rank": k,
            "threshold": cutoff if ready else None, "ready": ready,
            "status": "ready" if ready else "calibration_infeasible",
            "binomial_tail": tail, "calibration_fp": int(alerts.sum()),
            "preserved_calibration_fp": int(base.sum()),
            "comparison": ">", "preserve_old": preserve is not None}


def decisions(score, baseline, rule):
    baseline = np.asarray(baseline, bool)
    if rule.get("baseline_only"):
        return baseline.copy()
    # An infeasible added rule leaves old alerts intact, but cannot qualify.
    added = np.asarray(score) > rule["threshold"] if rule["ready"] else np.zeros(len(baseline), bool)
    return (baseline | added) if rule["preserve_old"] else added


def metrics(frame, alerts, baseline, ready):
    normal = frame.family.eq("Benign").to_numpy()
    unresolved = frame.family.eq("Unresolved").to_numpy()
    attack = ~(normal | unresolved)
    n, fp = int(normal.sum()), int((normal & alerts).sum())
    result = {"normal_n": n, "fp": fp, "budget": n // 100,
              "fpr": fp / n if n else None, "ready": bool(ready),
              "qualified": bool(ready and n and fp <= n // 100),
              "attack_n": int(attack.sum()), "tp": int((attack & alerts).sum()),
              "gained": int((attack & ~baseline & alerts).sum()),
              "lost": int((attack & baseline & ~alerts).sum()),
              "retained": int((attack & baseline & alerts).sum()),
              "fp_gained": int((normal & ~baseline & alerts).sum()),
              "fp_lost": int((normal & baseline & ~alerts).sum()),
              "unresolved_n": int(unresolved.sum()), "unresolved_alerts": int((unresolved & alerts).sum())}
    for family in ["ExternalC2", "Scanning", "Download"]:
        mask = frame.family.eq(family).to_numpy()
        result[family + "_n"] = int(mask.sum())
        result[family + "_tp"] = int((mask & alerts).sum())
    return result


def run(reference, output):
    reference, output = Path(reference), Path(output)
    reg = read(REGISTRATION)
    manifest = read(reference / "archive_manifest.json")
    protocol = read(reference / "protocol.json")
    names = protocol["registration"]["roles"]["forward_development"] + protocol["registration"]["roles"]["transport_external"]
    relcal = "evidence/replay/calibration.parquet"
    checked = [relcal, "protocol.json", "evidence/calibration/mixed-raw8-hgb.json"]
    checked += [f"evidence/replay/{name}.parquet" for name in names]
    for relative in checked:
        if digest(reference / relative) != manifest[relative]["sha256"]:
            raise ValueError("Return hash mismatch: " + relative)
    candidate, baseline = reg["candidate"], reg["baseline"]
    cal = pd.read_parquet(reference / relcal)
    oldcal = read(reference / "evidence/calibration/mixed-raw8-hgb.json")
    if len(cal) != reg["calibration_normal_rows"] or not cal.family.eq("Benign").all():
        raise ValueError("Unexpected historical normal pool")
    if oldcal["identity_sha256"] != reg["calibration_identity_sha256"]:
        raise ValueError("Historical calibration identity differs")
    base = cal[baseline].to_numpy() > .5
    score = cal[candidate].to_numpy()
    original = {"ready": True, "threshold": oldcal["threshold"], "preserve_old": False}
    rules = {
        "old_strong": {"ready": True, "baseline_only": True, "preserve_old": True},
        "mixed_original": original,
        "old_or_mixed_original": dict(original, preserve_old=True),
        "old_or_mixed_empirical": order_rule(score, alpha=reg["alpha"], preserve=base),
        "mixed_np95": order_rule(score, alpha=reg["alpha"], delta=reg["delta"]),
        "old_or_mixed_np95": order_rule(score, alpha=reg["alpha"], delta=reg["delta"], preserve=base),
    }
    assert list(rules) == reg["rules"]
    binding = {"registration": reg, "registration_sha256": digest(REGISTRATION),
               "inputs": {name: manifest[name]["sha256"] for name in checked},
               "source_sha256": digest(__file__), "rules": rules,
               "calibration_decisions": {name: int(decisions(score, base, rule).sum()) for name, rule in rules.items()}}
    # Persist the historical-data-only rules before opening evaluation scores or labels.
    write(output / "calibration_receipt.json", binding)
    rows = []
    for name in names:
        frame = pd.read_parquet(reference / f"evidence/replay/{name}.parquet")
        old = frame[baseline].to_numpy() > .5
        days = pd.to_datetime(frame.flow_end_ms, unit="ms", utc=True).dt.strftime("%Y-%m-%d")
        scopes = [("cell", "all", np.ones(len(frame), bool))]
        scopes += [("utc_day_within_cell", day, days.eq(day).to_numpy()) for day in sorted(days.unique())]
        for rule_name, rule in rules.items():
            alerts = decisions(frame[candidate].to_numpy(), old, rule)
            if rule["preserve_old"] and np.any(old & ~alerts):
                raise AssertionError("Preserved alert was lost")
            for scope, window, mask in scopes:
                rows.append({"source_id": name, "profile": name[4], "scope": scope, "window": window,
                             "rule": rule_name, **metrics(frame.loc[mask], alerts[mask], old[mask], rule["ready"])})
    result = pd.DataFrame(rows)
    result.to_csv(output / "metrics_and_pairs.csv", index=False)
    whole = result[result.scope.eq("cell")]
    summary = whole.groupby(["rule", "profile"]).agg(cells=("source_id", "size"),
        qualified_cells=("qualified", "sum"), fp=("fp", "sum"), normal_n=("normal_n", "sum"),
        tp=("tp", "sum"), attack_n=("attack_n", "sum"), gained=("gained", "sum"), lost=("lost", "sum"))
    summary.to_csv(output / "summary.csv")
    write(output / "outcome.json", {"status": "complete", "new_fits": 0, "model_forward_calls": 0,
                                    "calibration_rules": len(rules), "cells": len(names),
                                    "research_status": reg["status"], "new_blind_confirmations": 0})
    print(summary.to_string())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.reference_run, args.output)


if __name__ == "__main__":
    main()
