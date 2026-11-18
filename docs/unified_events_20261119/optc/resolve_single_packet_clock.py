"""Resolve only the offline observed-end clock of two one-packet observations.

python resolve_single_packet_clock.py
Reads only the existing small context export. No source rescan or download.
Native duration stays missing; online availability remains unknown.
"""
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from fetch_normal_sources import BASE, sha256
from assemble_normal_business import write_jsonl_gz

INPUT_SHA = "612ff8b20a3b794ccb35fe69ebf3d60f7bae2907081331340214ec14baddf18c"
EXPECTED_UIDS = {"CPdC8j3SCUkw9Gccff", "C4hCrq1WFLFdAFIKf9"}
REFERENCES = [
    {"url": "https://github.com/zeek/zeek/blob/v2.6.1/scripts/base/protocols/conn/main.bro#L18-L19",
     "meaning": "Conn::Info.ts is the first-packet time."},
    {"url": "https://github.com/zeek/zeek/blob/v2.6.1/scripts/base/protocols/conn/main.bro#L229-L245",
     "meaning": "duration is assigned only for positive c$duration; packet counts can be present without positive duration."},
    {"url": "https://github.com/zeek/zeek/blob/v2.6.1/scripts/base/protocols/conn/main.bro#L275-L282",
     "meaning": "The record is written on connection_state_remove; packet time is not the logging/timeout event time."},
]


def main():
    input_path = BASE / "context_inputs.jsonl.gz"
    source_sha = sha256(input_path)
    if source_sha != INPUT_SHA:
        raise ValueError("Input is not the reviewed context export")
    with gzip.open(input_path, "rt", encoding="utf-8") as f:
        original_rows = [json.loads(line) for line in f]
    revised_rows = []
    resolved = []
    unchanged = 0
    for original in original_rows:
        revised = dict(original)
        vector = original["raw8"]
        if len(vector) != 8:
            raise ValueError("Expected raw8")
        total_packets = None if vector[4] is None or vector[5] is None else vector[4] + vector[5]
        if total_packets == 1 and vector[3] is None and original.get("flow_end_at") is None:
            first = original.get("connection_start")
            if not first or datetime.fromisoformat(first).utcoffset() is None:
                raise ValueError("Missing explicit first-packet timestamp")
            revised["flow_end_at"] = first
            revised["observed_end_basis"] = {
                "rule": "single_observed_packet_first_equals_last",
                "orig_pkts": vector[4], "resp_pkts": vector[5], "total_observed_packets": 1,
                "inference": "The only observed packet is both the first and last observed packet, so observed packet end equals connection_start (the native ts).",
                "native_duration_changed": False,
                "clock_scope": "offline_observed_packet_end_only",
                "actual_log_emission_or_timeout_time_inferred": False,
                "reference_urls": [r["url"] for r in REFERENCES],
                "version_note": "v2.6.1 is an independently inspected official semantic reference; the exact OpTC executable build was not established by this step.",
            }
            changed_keys = {k for k in set(original) | set(revised) if original.get(k) != revised.get(k)}
            if changed_keys != {"flow_end_at", "observed_end_basis"}:
                raise ValueError("Unexpected mutation in resolved record")
            resolved.append({
                "flow_id": original["flow_id"], "uid": original["uid"],
                "source_path": original["source_path"], "source_line": original["source_line"],
                "raw8_unchanged": vector,
                "flow_end_at_before": None, "offline_observed_end_after": first,
                "actual_available_at": revised.get("available_at"),
            })
        else:
            if revised != original:
                raise ValueError("Unresolved record changed")
            unchanged += 1
        if revised["raw8"] != original["raw8"]:
            raise ValueError("Raw8 changed")
        if original.get("available_at") is not None or revised.get("available_at") is not None:
            raise ValueError("Actual availability must remain unknown")
        if revised.get("actual_available_at") is not None:
            raise ValueError("Actual availability was inferred")
        revised_rows.append(revised)
    if len(original_rows) != 45309 or len(resolved) != 2 or {r["uid"] for r in resolved} != EXPECTED_UIDS:
        raise ValueError("Resolution must affect exactly the two reviewed one-packet records")
    if any(r.get("flow_end_at") is None for r in revised_rows):
        raise ValueError("Unexpected unresolved observed-end clocks")
    if len({r["flow_id"] for r in revised_rows}) != 45309:
        raise ValueError("View IDs are not unique")
    output_path = BASE / "context_replay_inputs.jsonl.gz"
    write_jsonl_gz(output_path, revised_rows)
    if sha256(input_path) != source_sha:
        raise ValueError("Original context export changed")
    receipt = {
        "schema_version": "optc_single_packet_clock_resolution_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
        "input": {"path": str(input_path), "sha256_before": source_sha, "sha256_after": sha256(input_path), "rows": len(original_rows)},
        "output": {"path": str(output_path), "sha256": sha256(output_path), "bytes": output_path.stat().st_size, "rows": len(revised_rows)},
        "resolved_count": len(resolved), "unchanged_dictionary_count": unchanged,
        "all_raw8_unchanged": True,
        "native_duration_null_count_before": sum(r["raw8"][3] is None for r in original_rows),
        "native_duration_null_count_after": sum(r["raw8"][3] is None for r in revised_rows),
        "actual_available_at_null_count": sum(r.get("available_at") is None for r in revised_rows),
        "unresolved_observed_end_count": sum(r.get("flow_end_at") is None for r in revised_rows),
        "resolved_records": resolved,
        "inspected_primary_source": {
            "raw_url": "https://raw.githubusercontent.com/zeek/zeek/v2.6.1/scripts/base/protocols/conn/main.bro",
            "references": REFERENCES,
            "exact_OpTC_runtime_version_confirmed": False,
        },
        "reasoning": {
            "sufficient_basis": "Observed packet count equals one and native ts denotes the first observed packet. Therefore first and last observed packet times coincide.",
            "duration_semantics": "Positive-only optional duration explains why a one-packet record can omit duration. Missing duration by itself is NOT treated as zero.",
            "what_was_changed": "Only flow_end_at plus explicit observed_end_basis in the two reviewed records. Native raw8 duration and all actual availability fields remain unchanged/unknown.",
            "what_was_not_inferred": "UDP timeout time, connection-state removal time, actual log writing/ingestion time, or online availability.",
            "mode": "Explicit offline observed-end replay; not an online availability trace.",
        },
        "operations": "Read existing 2.7 MB context artifact only; no original conn rescan, no data download, no prefix/model execution.",
    }
    (BASE / "context_replay_inputs_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": receipt["output"], "resolved": len(resolved), "unchanged": unchanged,
                      "raw8_unchanged": True, "actual_time_still_null": receipt["actual_available_at_null_count"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
