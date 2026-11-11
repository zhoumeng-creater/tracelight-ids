"""Retrospective replay on the R20 calibration and previously opened E samples.

Run from the research checkout with "python -m ids.threshold_validation.cic_replay --help".
No raw dataset, live model fitting, full-network stream or new forward holdout is used here.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import json

import pandas as pd

from ids.common import sha256_file, write_json
from ids.route_batch.metrics import threshold
from .rules import POLICIES, audit_reference_mix, paired_operating_point, replay_thresholds


ARCHIVE_SHA = "c19bdf20f745801f3b860e0ef849025e73a748ada0da04659b7acffcb7aed63e"
MODELS = {
    "hgb48": "b1ba959fc0efab8bb1920b8f70f1b5f6f28a4ba34b6bfa777122fb1716404036",
    "hgb52_history": "2dfe99171dbe03ff9facaa151ae647c5ff5f4ec799bf6ce2318e4cdc875e51f3",
}
REPLAY_CONFIG = {
    "refresh_ms": 900_000,
    "window_ms": 3_600_000,
    "min_reference": 1000,
    "trusted_lag_ms": 900_000,
}


def load_capture(archive):
    """Bind immutable predictions, published decisions and timestamps by registered row identity."""
    if sha256_file(archive) != ARCHIVE_SHA:
        raise ValueError("Expected the original R20 result ZIP with pinned SHA-256")

    with ZipFile(archive) as z:

        def frame(name):
            return pd.read_parquet(BytesIO(z.read(name))).set_index("row_id", verify_integrity=True)

        def read(name):
            return json.loads(z.read(name))

        if read("last_invocation.json")["status"] != "complete":
            raise ValueError("Incomplete R20 run")
        reg = read("protocol.json")["registration"]
        if reg["release"] != "independent-capture-lab-v1":
            raise ValueError("Unregistered R20 release")
        rows = frame("evidence/inputs/registered_rows.parquet")
        cal_ids = rows.index[rows.role.eq("C_time")]
        evaluation = rows.loc[rows.role.eq("E_time"), ["label", "attack_family", "flow_end_ms"]].copy()
        if not rows.loc[cal_ids, "label"].eq(0).all() or len(cal_ids) != 10000 or len(evaluation) != 50000:
            raise ValueError("Unexpected R20 calibration or evaluation")
        if rows.loc[cal_ids, "flow_end_ms"].max() >= evaluation.flow_end_ms.min():
            raise ValueError("Calibration overlaps evaluation")
        calibrations = {}
        for arm, digest in MODELS.items():
            prefix = "evidence/forward/" + arm + "/"
            if (
                sha256(z.read(f"models/forward/{arm}.joblib")).hexdigest() != digest
                or read(prefix + "fit.json")["model_sha256"] != digest
            ):
                raise ValueError("Frozen model identity changed")
            cal = frame(prefix + "calibration.parquet")
            prediction = frame(prefix + "evaluation.parquet")
            if set(cal.index) != set(cal_ids) or set(prediction.index) != set(evaluation.index):
                raise ValueError("R20 row identity mismatch")
            prediction = prediction.loc[evaluation.index]
            if not cal.label.eq(0).all() or not prediction[["label", "attack_family"]].equals(
                evaluation[["label", "attack_family"]]
            ):
                raise ValueError("R20 label mismatch")
            calibrations[arm] = threshold(cal.score)
            fixed = calibrations[arm]["threshold"]
            saved = read(prefix + "metrics.json")["threshold"]
            if fixed != saved["threshold"] or saved["rank"] != calibrations[arm]["rank"]:
                raise ValueError("R20 threshold changed")
            if not prediction.score.gt(fixed).equals(prediction.alert):
                raise ValueError("Published R20 decisions changed")
            evaluation[arm] = prediction.score
    return evaluation.reset_index(), calibrations


def replay_observed(evaluation, calibrations, config):
    # Only observed E scores enter the update pool. The calibration table was sampled normal-only.
    time_sorted = evaluation.sort_values("flow_end_ms", kind="stable").reset_index(drop=True)
    start = int(time_sorted.flow_end_ms.iloc[0])
    end = int(time_sorted.flow_end_ms.iloc[-1]) + 1
    outcomes = time_sorted[["row_id", "flow_end_ms", "label", "attack_family"]].copy()
    confirmations = (time_sorted.flow_end_ms + config["trusted_lag_ms"]).astype("Int64")
    confirmations = confirmations.where(time_sorted.label.eq(0))
    audits, all_mix = [], []
    for arm in MODELS:
        stream = time_sorted[["row_id", "flow_end_ms", arm]].rename(columns={arm: "score"})
        decisions, updates = replay_thresholds(
            stream,
            calibrations[arm]["threshold"],
            start_ms=start,
            end_ms=end,
            trusted_normal_available_ms=confirmations,
            **config,
        )
        if not decisions.row_id.equals(outcomes.row_id):
            raise ValueError("Decisions and evaluation truth are not aligned")
        for policy in POLICIES:
            outcomes[arm + "/" + policy] = decisions[policy + "_alert"]
        updates.insert(0, "arm", arm)
        audits.append(updates)
        # Per-block empirical contamination is the same for both frozen scorers.
        if not all_mix:
            all_mix.append(audit_reference_mix(stream, updates, time_sorted.label))
    return outcomes, pd.concat(audits, ignore_index=True), all_mix[0]


def summary(outcomes, updates, config):
    arm_comparisons = {
        policy: paired_operating_point(outcomes, "hgb48/" + policy, "hgb52_history/" + policy)
        for policy in POLICIES
    }
    vs_fixed = {
        arm + "/" + policy: paired_operating_point(outcomes, arm + "/fixed", arm + "/" + policy)
        for arm in MODELS
        for policy in ("recent_normal", "unlabeled")
    }
    by_day = []
    days = pd.to_datetime(outcomes.flow_end_ms, unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    for date, rows in outcomes.groupby(days, sort=True):
        for policy in POLICIES:
            by_day.append(
                {
                    "date_utc": date,
                    "policy": policy,
                    **paired_operating_point(rows, "hgb48/" + policy, "hgb52_history/" + policy),
                }
            )
    coverage = updates.groupby(["arm", "policy", "status"], as_index=False).agg(
        blocks=("update_ms", "size"), decisions=("decision_rows", "sum")
    )
    return {
        "type": "retrospective_R20_previously_opened_E_only",
        "source_result_sha256": ARCHIVE_SHA,
        "model_sha256": MODELS,
        "rows": len(outcomes),
        "utc_first_flow_end_ms": int(outcomes.flow_end_ms.min()),
        "utc_last_flow_end_ms": int(outcomes.flow_end_ms.max()),
        "config": config,
        "alpha": 0.01,
        "population": "Observed R20 adjudicated E sample only. Normal-only C provides fixed thresholds and no update rows.",
        "feedback": "Conditional oracle: every registered normal is assumed confirmed after a 15-minute delay. No actual feedback timestamps in this source. Neither fixed nor unlabeled uses it.",
        "reference_rule": "At UTC refresh, unlabeled uses all previous one-hour E scores. Recent-normal's one-hour flow window ends 15 minutes earlier, with confirmation strictly before refresh. Insufficient pool falls back to fixed.",
        "between_models_same_policy": arm_comparisons,
        "adaptive_vs_fixed_same_model": vs_fixed,
        "per_UTC_day_same_policy": by_day,
        "policy_update_coverage": coverage.to_dict("records"),
    }


def run(archive, output):
    evaluation, calibration = load_capture(archive)
    outcomes, updates, mix = replay_observed(evaluation, calibration, REPLAY_CONFIG)
    results = summary(outcomes, updates, REPLAY_CONFIG)
    results["registered_calibration"] = calibration
    # Each window appears once; neighboring windows overlap. Do not sum repeated rows as unique traffic.
    results["reference_contamination"] = {
        "evaluated_blocks": len(mix),
        "blocks_with_known_attacks": int(mix.known_attack.gt(0).sum()),
        "max_fraction_of_labeled_flows_in_one_reference_window": float(mix.attack_fraction_of_labeled.max()),
    }
    output.mkdir(parents=True, exist_ok=True)
    updates.to_parquet(output / "threshold_updates.parquet", index=False)
    mix.to_csv(output / "reference_mix.csv", index=False)
    write_json(output / "replay_summary.json", results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Pinned R20 14,706,746-byte result ZIP")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.archive, args.output)
    print(f"Replayed {result['rows']} previously opened E rows; wrote {args.output / 'replay_summary.json'}")


if __name__ == "__main__":
    main()
