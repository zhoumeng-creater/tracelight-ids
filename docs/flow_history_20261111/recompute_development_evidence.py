"""Read saved R17/R18 evidence; never open raw NetFlow or evaluate any new model."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
ROUTE = "20261109_第十七轮路线批次/route-results-20261109T034142631Z.zip"
PROJECTION = "20261111_第十八轮路线补验/复核/历史混淆组复核.json"
STORE = "TraceLight-证据库-2026-11-09.zip"


def csv_dicts(data):
    return csv.DictReader(io.StringIO(data.decode("utf-8-sig")))


def summarize(feature_csv, score_csv, threshold, signature):
    scores = {r["row_id"]: r for r in csv_dicts(score_csv)}
    seen = set()
    counts = Counter()
    collision = defaultdict(Counter)
    for r in csv_dicts(feature_csv):
        row_id = r["sample_id"]
        saved = scores[row_id]
        assert row_id not in seen
        seen.add(row_id)
        label = int(r["label"])
        assert label == int(saved["label"]) and r["attack_family"] == saved["attack_family"]
        alert = float(saved["score"]) > threshold
        assert alert == bool(int(saved["predicted_attack"]))
        counts["rows"] += 1
        counts["normal"] += 1 - label
        counts["tp"] += label * alert
        counts["fp"] += (1 - label) * alert
        if not all(r[name] and float(r[name]) == number for name, number in signature.items()):
            continue
        assert r["attack_family"] in ("Scanning", "Benign")
        counts["group_scan"] += label
        counts["group_normal"] += 1 - label
        counts["group_tp"] += label * alert
        counts["group_fp"] += (1 - label) * alert
        collision[int(float(r["L4_SRC_PORT"]))][label] += 1
    assert seen == set(scores)
    mixed = [tally for tally in collision.values() if tally[0] and tally[1]]
    fp_budget = counts["normal"] // 100
    add_tp = counts["group_scan"] - counts["group_tp"]
    add_fp = counts["group_normal"] - counts["group_fp"]
    new_fp = counts["fp"] + add_fp
    return {
        "evaluation_rows": counts["rows"],
        "normal_rows": counts["normal"],
        "full_kv_tp": counts["tp"],
        "full_kv_fp": counts["fp"],
        "full_kv_empirical_fpr": counts["fp"] / counts["normal"],
        "max_fp_at_empirical_1pct": fp_budget,
        "extra_fp_headroom": fp_budget - counts["fp"],
        "group_scanning": counts["group_scan"],
        "group_normal": counts["group_normal"],
        "full_kv_group_tp": counts["group_tp"],
        "full_kv_group_fp": counts["group_fp"],
        "alert_entire_group_with_other_decisions_fixed": {
            "additional_tp": add_tp,
            "additional_fp": add_fp,
            "tp": counts["tp"] + add_tp,
            "fp": new_fp,
            "fpr": new_fp / counts["normal"],
            "other_fp_repairs_needed_if_entire_group_alerts_and_empirical_fpr_at_most_1pct": max(
                0, new_fp - fp_budget
            ),
        },
        "opposite_labels_identical_all_48_fields_within_group": {
            "distinct_mixed_inputs": len(mixed),
            "scanning": sum(c[1] for c in mixed),
            "normal": sum(c[0] for c in mixed),
            "minimum_errors_for_a_deterministic_per_row_48_field_decision": sum(
                min(c[0], c[1]) for c in mixed
            ),
        },
    }


def recompute(root):
    r18_bytes = (root / PROJECTION).read_bytes()
    r18 = json.loads(r18_bytes)
    signature = r18["fixed_47_features"]
    assert len(signature) == 47 and "L4_SRC_PORT" not in signature
    prior_e_sha = r18["sources"]["context-input-v2/prepared/development_evaluation.csv"]
    with ZipFile(root / STORE) as store:
        old_e = store.read("objects/" + prior_e_sha)
    assert hashlib.sha256(old_e).hexdigest() == prior_e_sha
    with (root / ROUTE).open("rb") as source:
        route_sha = hashlib.file_digest(source, "sha256").hexdigest()
    assert route_sha == "477724966f028e636d545e5c04f7dd0139d057b8b67d0c0d714758fdf1692b31"
    with ZipFile(root / ROUTE) as run:
        old_path = "tasks/reference_full/attempt-001/"
        opened_path = "confirmation/models/reference_full/attempt-001/"
        dev = summarize(
            old_e,
            run.read(old_path + "evaluation.csv"),
            json.loads(run.read(old_path + "threshold.json"))["threshold"],
            signature,
        )
        opened = summarize(
            run.read("confirmation/prepared/confirmation_evaluation.csv"),
            run.read(opened_path + "evaluation.csv"),
            json.loads(run.read(opened_path + "threshold.json"))["threshold"],
            signature,
        )
        prep = json.loads(run.read("confirmation/prepared/preparation.json"))
    assert (dev["group_scanning"], dev["group_normal"]) == (r18["scanning"], r18["normal"])
    cf = r18["round_17_confirmation_projection"]
    assert (opened["group_scanning"], opened["group_normal"]) == (cf["group_scanning"], cf["group_normal"])
    lower = cf["both_labels_same_full_48_fields"]
    assert opened["opposite_labels_identical_all_48_fields_within_group"] == {
        "distinct_mixed_inputs": len(lower["source_port_values"]),
        "scanning": lower["scanning_rows"],
        "normal": lower["normal_rows"],
        "minimum_errors_for_a_deterministic_per_row_48_field_decision": lower[
            "minimum_row_errors_for_deterministic_48_field_classifier"
        ],
    }
    names = ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
    assert set(names) <= set(prep["scan"]["source_header"])
    return {
        "evidence": {
            "round17_result_zip_sha256": route_sha,
            "round18_projection_sha256": hashlib.sha256(r18_bytes).hexdigest(),
            "opened_development_E_member_verified_sha256": prior_e_sha,
        },
        "opened_development_E": dev,
        "previously_opened_round17_Eprime": opened,
        "source_clock_evidence": {
            "source_sha256": prep["source"]["sha256"],
            "completed_source_records": prep["scan"]["source"]["rows"],
            "csv_read_to_end_and_crc_verified": prep["source"]["member_complete_read_and_crc_verified"],
            "present_in_observed_header": names,
            "limit": "The archived R17 header demonstrates field availability. It does not establish that every "
            "row has valid times, an export/arrival timestamp, sufficient history, or a feasible forward split. "
            "R19 checks the event times and cohort counts on the whole official file.",
        },
        "interpretation": [
            "Counterfactual changes only alarms inside the fixed R16 group, never the saved threshold or other rows.",
            "One percent here is empirical FPR on E or E-prime. The original thresholds were registered on C or C-prime.",
            "The paired-label bound applies solely to these identical 48-field inputs under a deterministic per-flow decision. "
            "It neither bounds all scan misses nor establishes that any available history separates those labels.",
            "Both evaluations were already opened. These numbers are design constraints, never R19 temporal evidence.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=HERE.parents[3] / "实验归档")
    parser.add_argument("--output", type=Path, default=HERE / "development_evidence.json")
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(recompute(args.archive_root), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
