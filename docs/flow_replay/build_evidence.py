"""Export the archived row identities and paired decisions for an offline replay.

Usage: python docs/flow_replay/build_evidence.py R20_RESULT.zip
No model is loaded. Run from any directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "forward_threshold_20261111"
HISTORY = ["hist_flows_60s", "hist_dst_ips_60s", "hist_dst_ports_60s", "hist_no_reply_fraction_60s"]
POLICIES = ["fixed", "unlabeled", "recent_normal"]


def load_json(name):
    return json.loads((SOURCE / name).read_text(encoding="utf-8"))


def extract_rows(zip_path, tail):
    with zip_path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if digest != tail["evidence_sha256"]:
        raise ValueError("Expected the archived round-20 result with SHA-256 " + tail["evidence_sha256"])

    def frame(archive, member):
        return pd.read_parquet(BytesIO(archive.read(member))).set_index("row_id", verify_integrity=True)

    with ZipFile(zip_path) as archive:
        rows = frame(archive, "evidence/inputs/registered_rows.parquet")
        history = frame(archive, "evidence/inputs/registered_history.parquet")
        scores = frame(archive, "evidence/forward/hgb52_history/calibration.parquet")
    host = rows.loc[rows.role.eq("C_time") & rows.host_key.eq(tail["leading_C_host"])]
    host = host.join(history[HISTORY], validate="one_to_one").join(scores.score, validate="one_to_one")
    fixed = tail["fixed_thresholds"]["hist"]
    subset = tail["history_zero_ablation"]["subsets"]["C_major_host"]
    assert len(host) == subset["n"] and (host.score > fixed).sum() == subset["actual_fp"]
    assert subset["zero_history_fp"] == 0 and host.source_family.eq("Benign").all()
    assert host[["score", *HISTORY]].notna().all().all()
    assert tail["leading_C_host_all"]["L4_DST_PORT_top"] == {"135": len(host)}
    assert tail["leading_C_host_all"]["PROTOCOL_top"] == {"6": len(host)}
    above = tail["leading_C_host_above"]
    start, end = (
        int(datetime.fromisoformat(above[k]).timestamp() * 1000) for k in ["first_end_utc", "last_end_utc"]
    )
    segment = host[host.flow_end_ms.between(start, end)].sort_values(["flow_end_ms", "row_id"])
    assert (segment.score > fixed).sum() == subset["actual_fp"]
    compact = []
    for row in segment.itertuples():
        csv_digest, row_number = row.Index.rsplit(":", 1)
        assert csv_digest == tail["source_csv_sha256"]
        compact.append(
            {
                "row": int(row_number),
                "time": int(row.flow_end_ms),
                "score": row.score,
                "flows": int(row.hist_flows_60s),
                "ips": int(row.hist_dst_ips_60s),
                "ports": int(row.hist_dst_ports_60s),
                "noReply": row.hist_no_reply_fraction_60s,
            }
        )
    return compact


def groups(pair):
    result = {}
    for name, g in pair["groups"].items():
        if name == "all_attacks":
            continue
        assert g["candidate"] - g["baseline"] == g["added"] - g["lost"] == g["net"]
        result[name] = {k: g[k] for k in ("rows", "baseline", "candidate", "added", "lost")}
    return result


def make_evidence(zip_path):
    tail = load_json("cic-tail.json")
    old = load_json("cic-retrospective.json")
    new = load_json("verified-results.json")
    assert old["source_result_sha256"] == tail["evidence_sha256"]
    old_pairs = {"compare": {}, "update": {}}
    for policy in POLICIES:
        old_pairs["compare"][policy] = groups(old["between_models_same_policy"][policy])
        if policy != "fixed":
            old_pairs["update"][policy] = groups(
                old["adaptive_vs_fixed_same_model"]["hgb52_history/" + policy]
            )
    scenes = {"CIC2018": old_pairs}
    for full_name, scene in new["scenarios"].items():
        pairs = {"compare": {}, "update": {}}
        for policy in POLICIES:
            pairs["compare"][policy] = groups(scene["paired"]["raw_vs_history_" + policy])
            if policy != "fixed":
                pairs["update"][policy] = groups(scene["paired"]["history_fixed_vs_" + policy])
        scenes[full_name.rsplit("Capture-", 1)[-1]] = pairs
    stats = tail["history_zero_ablation"]["subsets"]
    return {
        "version": 1,
        "threshold": tail["fixed_thresholds"]["hist"],
        "sourceCSV": tail["source_csv_sha256"],
        "round20": tail["evidence_sha256"],
        "round21": new["sha256"],
        "sourceHost": tail["leading_C_host"],
        "hostCount": stats["C_major_host"]["n"],
        "calCount": stats["C"]["n"],
        "calFP": stats["C"]["actual_fp"],
        "hostFP": stats["C_major_host"]["actual_fp"],
        "ties": stats["C_major_host_at_boundary"]["n"],
        "zeroHostFP": stats["C_major_host"]["zero_history_fp"],
        "rows": extract_rows(zip_path, tail),
        "scenes": scenes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path)
    payload = make_evidence(parser.parse_args().zip_path)
    data = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    (HERE / "replay-data.js").write_text("window.replayEvidence = " + data + ";\n", encoding="utf-8")
