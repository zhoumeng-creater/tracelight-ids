"""复核第二十一轮归档。阈值和逐行判决从冻结分数独立重建，不加载训练/评分类代码。"""

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ARMS = ("raw", "history")
POLICIES = ("fixed", "unlabeled", "recent_normal")
FRACTION = 100
MINIMUM = 1000
UPDATE = 900_000
PAST = {"unlabeled": 4 * UPDATE, "recent_normal": 96 * UPDATE}
DELAY = 900_000
TESTS = {"CTU-IoT-Malware-Capture-34-1": 23145, "CTU-IoT-Malware-Capture-49-1": 5410561}
CONTROL = {"CTU-Honeypot-Capture-4-1": 452, "CTU-Honeypot-Capture-5-1": 1374}


def checked(condition, message):
    if not condition:
        raise ValueError(message)


def readj(z, name):
    return json.loads(z.read(name))


def readp(z, name):
    with BytesIO(z.read(name)) as stream:
        return pd.read_parquet(stream)


def rank_cut(x):
    rank = (99 * (len(x) + 1) + 99) // 100
    return rank, float(np.partition(x, rank - 1)[rank - 1])


def same_optional(observed, expected):
    return pd.isna(observed) if expected is None else observed == expected


def summaries(base, candidate, labels, group_values):
    result = {}
    for family, code in group_values.items():
        if family == "all_attacks":
            mask = (labels != group_values["Benign"]) & (
                labels != group_values.get("Unresolved", -1)
            )
        else:
            mask = labels == code
        x, y = base[mask], candidate[mask]
        result["normal" if family == "Benign" else family] = {
            "rows": int(mask.sum()),
            "baseline": int(x.sum()),
            "candidate": int(y.sum()),
            "added": int((~x & y).sum()),
            "lost": int((x & ~y).sum()),
            "net": int(y.sum()) - int(x.sum()),
        }
    return result


def calibrations(z):
    result = {}
    identities = None
    for arm in ARMS:
        cal = readp(z, f"evidence/{arm}/calibration.parquet")
        keys = cal[["source", "row_id"]]
        checked(
            not keys.duplicated().any()
            and cal.groupby("source").size().to_dict() == CONTROL,
            "calibration rows",
        )
        if identities is not None:
            checked(keys.equals(identities), "different calibration rows")
        identities = keys
        values = cal.score.to_numpy()
        checked(np.isfinite(values).all(), "nonfinite calibration score")
        rank, cut = rank_cut(values)
        above = int((values > cut).sum())
        registered = readj(z, f"evidence/{arm}/fit.json")["calibration"]
        for key, val in {
            "threshold": cut,
            "rank": rank,
            "n_normal": len(cal),
            "calibration_fp": above,
        }.items():
            checked(registered[key] == val, f"{arm}: calibration {key}")
        result[arm] = {"rank": rank, "cut": cut, "n": len(cal), "strictly_above": above}
    return result


def scenario(z, name, n, fixed):
    folder = "evidence/" + name
    frame = readp(z, folder + "/row-scores-and-alerts.parquet")
    times = frame.flow_end_ms.to_numpy()
    ids = frame.row_id.to_numpy()
    checked(
        len(frame) == n and ids[0] == 1 and ids[-1] == n and (np.diff(ids) == 1).all(),
        "invalid physical identities",
    )
    checked((times[1:] >= times[:-1]).all(), "invalid chronological watermark")
    kind = pd.Categorical(frame.attack_family)
    labels = kind.codes
    groups = {str(k): i for i, k in enumerate(kind.categories) if k != "Unresolved"}
    groups["all_attacks"] = -1
    normal = labels == groups["Benign"]
    unknown = labels == (
        kind.categories.get_loc("Unresolved") if "Unresolved" in kind.categories else -1
    )
    checked(
        (frame.label.to_numpy() == (~normal).astype("int8")).all(),
        "inconsistent binary labels",
    )
    block_vals = times // UPDATE * UPDATE
    starts = np.r_[0, np.flatnonzero(block_vals[1:] != block_vals[:-1]) + 1, n]
    del block_vals
    archived_cmp = readj(z, folder + "/comparison.json")
    checked(
        archived_cmp["n_observations"] == n
        and archived_cmp["n_adjudicated"] == int((~unknown).sum()),
        "scenario size",
    )
    budget = int(normal.sum()) // 100
    audit = {}
    for arm in ARMS:
        updates = readp(z, f"{folder}/{arm}-updates.parquet").set_index(
            ["policy", "update_ms"], verify_integrity=True
        )
        mix = readp(z, f"{folder}/{arm}-reference-mix.parquet").set_index(
            "update_ms", verify_integrity=True
        )
        checked(
            len(updates) == 3 * (len(starts) - 1) and len(mix) == len(starts) - 1,
            "missing updates",
        )
        score = frame[arm + "_score"].to_numpy()
        checked(np.isfinite(score).all(), "nonfinite evaluation score")
        audit[arm] = {}
        for policy in POLICIES:
            active_by_family = Counter()
            pool_attack_fraction = []
            active_periods = []
            all_refs = []
            statuses = Counter()
            n_accepted = 0
            relmask = np.zeros(n, dtype=bool)
            for start, stop in zip(starts[:-1], starts[1:]):
                block = int(times[start]) // UPDATE * UPDATE
                record = updates.loc[(policy, block)]
                baseline_cut = fixed[arm]["cut"]
                expected = {
                    "decision_rows": int(stop - start),
                    "status": "fixed",
                    "threshold": baseline_cut,
                }
                if policy != "fixed":
                    upper = block - (DELAY if policy == "recent_normal" else 0)
                    lower = upper - PAST[policy]
                    left, right = np.searchsorted(times, [lower, upper], side="left")
                    past = score[left:right]
                    when = times[left:right]
                    if policy == "recent_normal":
                        good = normal[left:right]
                        past = past[good]
                        when = when[good]
                    enough = len(past) >= MINIMUM
                    rank, cut = rank_cut(past) if enough else (None, baseline_cut)
                    max_observed = int(when[-1]) if len(when) else None
                    max_confirmation = (
                        max_observed + DELAY
                        if max_observed is not None and policy == "recent_normal"
                        else None
                    )
                    expected.update(
                        window_start_ms=lower,
                        window_end_exclusive_ms=upper,
                        n_reference=len(past),
                        rank=rank,
                        threshold=cut,
                        reference_strictly_above=int((past > cut).sum()),
                        max_flow_end_ms=max_observed,
                        max_confirmation_ms=max_confirmation,
                        status="updated" if enough else "insufficient_reference",
                    )
                    if max_confirmation is not None:
                        checked(max_confirmation < block, "future confirmation")
                    if max_observed is not None:
                        checked(max_observed < block, "future score")
                    all_refs.append(len(past))
                    if enough:
                        relmask[start:stop] = True
                        active_periods.append(block)
                        active_by_family.update(
                            {
                                str(k): int(v)
                                for k, v in zip(
                                    kind.categories,
                                    np.bincount(
                                        labels[start:stop].astype(np.int32),
                                        minlength=len(kind.categories),
                                    ),
                                )
                            }
                        )
                    if policy == "unlabeled":
                        mixrow = mix.loc[block]
                        window_normal = int(normal[left:right].sum())
                        window_unknown = int(unknown[left:right].sum())
                        window_attack = len(past) - window_normal - window_unknown
                        purity = (
                            window_attack / (len(past) - window_unknown)
                            if len(past) > window_unknown
                            else None
                        )
                        for col, want in {
                            "reference_rows": len(past),
                            "known_normal": window_normal,
                            "known_attack": window_attack,
                            "unadjudicated": window_unknown,
                            "attack_fraction_of_labeled": purity,
                        }.items():
                            checked(
                                same_optional(mixrow[col], want),
                                f"contamination {name} {arm} {block} {col}",
                            )
                        if enough and purity is not None:
                            pool_attack_fraction.append(purity)
                for col, want in expected.items():
                    checked(
                        same_optional(record[col], want),
                        f"update {name} {arm} {policy} {block} {col}",
                    )
                recorded = frame[arm + "_" + policy].to_numpy()[start:stop]
                got = score[start:stop] > expected["threshold"]
                checked(
                    np.array_equal(got, recorded),
                    f"row alert mismatch: {name} {arm} {policy} {block}",
                )
                n_accepted += int(got.sum())
                statuses[expected["status"]] += 1
            audit[arm][policy] = {
                "blocks": len(starts) - 1,
                "by_status": dict(statuses),
                "alerts": n_accepted,
                "rows_verified": n,
                "rows_on_updated_cut": int(relmask.sum()),
                "max_reference": max(all_refs, default=None),
                "families_on_updated_cut": dict(active_by_family),
                "first_active_ms": min(active_periods, default=None),
                "active_unlabeled_reference_attack_fraction_min_median_max": [
                    float(np.min(pool_attack_fraction)),
                    float(np.median(pool_attack_fraction)),
                    float(np.max(pool_attack_fraction)),
                ]
                if pool_attack_fraction
                else None,
            }
    pairs = [(f"raw_vs_history_{p}", "raw_" + p, "history_" + p) for p in POLICIES]
    pairs += [
        (f"{a}_fixed_vs_{p}", a + "_fixed", a + "_" + p)
        for p in POLICIES
        if p != "fixed"
        for a in ARMS
    ]
    paired = {}
    for key, raw_col, candidate_col in pairs:
        a, b = frame[raw_col].to_numpy(), frame[candidate_col].to_numpy()
        result = summaries(a, b, labels, groups)
        old = archived_cmp["comparisons"][key]
        for grp, vals in result.items():
            checked(
                all(old["groups"][grp][k] == val for k, val in vals.items()),
                f"paired {name} {key} {grp}",
            )
        a_ok, b_ok = (
            result["normal"]["baseline"] <= budget,
            result["normal"]["candidate"] <= budget,
        )
        checked(
            old["normal_false_alert_budget"] == budget
            and old["baseline_within_one_percent"] == a_ok
            and old["candidate_within_one_percent"] == b_ok,
            "FPR decision disagrees",
        )
        strict_result = (
            a_ok
            and b_ok
            and result["Scanning"]["net"] > 0
            and result["C&C"]["baseline"] > 0
            and result["C&C"]["lost"] == 0
        )
        checked(
            old["both_within_budget_more_scan_and_cc_preserved"] == strict_result,
            "joint comparison",
        )
        paired[key] = {
            "base_within_1pct": a_ok,
            "candidate_within_1pct": b_ok,
            "nonvacuous_joint_success": strict_result,
            "groups": result,
        }
    return {
        "rows": n,
        "normal_denominator": int(normal.sum()),
        "max_false_alerts": budget,
        "policy_and_contamination_checks": audit,
        "paired": paired,
        "score_decision_checks": n * len(ARMS) * len(POLICIES),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    with args.archive.open("rb") as f:
        digest = sha256()
        for block in iter(lambda: f.read(4 << 20), b""):
            digest.update(block)
    result = {
        "source": args.archive.name,
        "sha256": digest.hexdigest(),
        "cut_rule": "ceil(0.99*(n+1)), strict >",
    }
    with ZipFile(args.archive) as z:
        checked(
            readj(z, "last_invocation.json")["status"] == "complete",
            "not a finished execution",
        )
        paired = readj(z, "evidence/paired-outcome.json")
        checked(
            paired["primary"] == "raw_vs_history_unlabeled"
            and set(paired["scenarios"]) == set(TESTS),
            "missing final paired outcome",
        )
        registration = readj(z, "protocol.json")["registration"]
        settings = registration["thresholds"]
        checked(
            settings["alpha"] == 0.01
            and settings["update_every_minutes"] == 15
            and settings["unlabeled_reference_hours"] == 1
            and settings["conditional_trusted_normal_window_hours"] == 24
            and settings["simulated_feedback_lag_minutes"] == 15
            and settings["minimum_reference"] == 1000,
            "unexpected registered settings",
        )
        m = readj(z, "archive_manifest.json")
        for name, entry in m.items():
            h = sha256()
            with z.open(name) as f:
                for block in iter(lambda: f.read(4 << 20), b""):
                    h.update(block)
            checked(
                z.getinfo(name).file_size == entry["bytes"]
                and h.hexdigest() == entry["sha256"],
                name,
            )
        result["manifest_entries_verified"] = len(m)
        fixed = calibrations(z)
        result["fixed"] = fixed
        result["scenarios"] = {}
        for name, n in TESTS.items():
            report = scenario(z, name, n, fixed)
            for key, compared in report["paired"].items():
                summary = paired["scenarios"][name][key]
                checked(
                    compared["nonvacuous_joint_success"]
                    == summary["both_within_budget_more_scan_and_cc_preserved"]
                    and compared["groups"]["normal"]["candidate"]
                    == summary["fp"]["candidate"]
                    and compared["groups"]["Scanning"]["candidate"]
                    == summary["scans"]["candidate"]
                    and compared["groups"]["C&C"]["candidate"]
                    == summary["cc"]["candidate"],
                    "returned paired-outcome differs",
                )
            expected = next(
                s["families"] for s in registration["sources"] if s["name"] == name
            )
            groups = report["paired"]["raw_vs_history_fixed"]["groups"]
            for family, cnt in expected.items():
                checked(
                    groups["normal" if family == "Benign" else family]["rows"] == cnt,
                    "unregistered totals",
                )
            result["scenarios"][name] = report
    result["total_score_decision_checks"] = sum(
        d["score_decision_checks"] for d in result["scenarios"].values()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf8",
    )
    print(args.output, result["total_score_decision_checks"])


if __name__ == "__main__":
    main()
