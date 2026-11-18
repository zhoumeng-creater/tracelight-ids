"""Checksum/length audit of the 38 fixed request-active packet candidates.

Reads the existing PCAP and frozen mapping only. Does not alter packets, rerun
Zeek, select by model scores, or change any frozen mapping artifact.
"""
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import struct

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent.parent / "代码/tracelight-ids/.worktrees/ait-flow-contract-20261118"
DIRECTIONS = ("proxy_to_cloud", "cloud_to_proxy")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def folded_sum(data):
    if len(data) % 2:
        data += b"\0"
    value = sum(struct.unpack("!" + "H" * (len(data) // 2), data))
    while value >> 16:
        value = (value & 0xFFFF) + (value >> 16)
    return value


def ipv4(frame, linktype):
    if linktype != 1 or len(frame) < 14:
        return None
    offset, ether = 14, struct.unpack("!H", frame[12:14])[0]
    while ether in (0x8100, 0x88A8):
        if len(frame) < offset + 4:
            return None
        ether = struct.unpack("!H", frame[offset + 2:offset + 4])[0]
        offset += 4
    if ether != 0x0800:
        return None
    ip = frame[offset:]
    if len(ip) < 20 or ip[0] >> 4 != 4:
        return None
    ihl, total = (ip[0] & 15) * 4, struct.unpack("!H", ip[2:4])[0]
    if ip[9] != 6 or ihl < 20 or len(ip) < ihl + 20:
        return None
    tcp = ip[ihl:]
    sport, dport, seq, ack = struct.unpack("!HHII", tcp[:12])
    return ip, tcp, {
        "source_ip": socket.inet_ntoa(ip[12:16]), "destination_ip": socket.inet_ntoa(ip[16:20]),
        "source_port": sport, "destination_port": dport, "sequence": seq, "acknowledgment": ack,
        "flags": tcp[13], "ip_header_bytes": ihl, "ip_total_length": total,
        "captured_ip_bytes": len(ip), "tcp_header_bytes": (tcp[12] >> 4) * 4,
        "fragment_bits": struct.unpack("!H", ip[6:8])[0] & 0x3FFF,
    }


def checksum_info(ip, tcp, meta):
    ihl, total, tcp_header = meta["ip_header_bytes"], meta["ip_total_length"], meta["tcp_header_bytes"]
    info = {"ipv4_checksum": "correct" if folded_sum(ip[:ihl]) == 0xFFFF else "incorrect",
            "tcp_checksum": "unverifiable", "tcp_unverifiable_reason": None,
            "stored_tcp_checksum": struct.unpack("!H", tcp[16:18])[0],
            "tcp_checksum_matches_pseudoheader_seed": None,
            "ipv4_total_exceeds_captured": total > len(ip), "ipv4_total_zero": total == 0}
    tcp_length = total - ihl
    if meta["fragment_bits"]:
        reason = "fragmented_ipv4_not_reassembled"
    elif total < ihl or tcp_length < 20:
        reason = "invalid_or_zero_ip_total_length"
    elif tcp_header < 20 or tcp_header > tcp_length:
        reason = "invalid_tcp_header_length"
    elif total > len(ip):
        reason = "declared_tcp_segment_not_fully_captured"
    else:
        reason = None
    if 20 <= tcp_length <= 65535:
        pseudo = ip[12:20] + b"\0\x06" + struct.pack("!H", tcp_length)
        seed = folded_sum(pseudo)
        info["pseudoheader_folded_sum"] = seed
        info["tcp_checksum_matches_pseudoheader_seed"] = info["stored_tcp_checksum"] == seed
        if reason is None:
            full_tcp = tcp[:tcp_length]
            info["tcp_checksum"] = "correct" if folded_sum(pseudo + full_tcp) == 0xFFFF else "incorrect"
            zeroed = full_tcp[:16] + b"\0\0" + full_tcp[18:]
            info["recomputed_tcp_checksum"] = (~folded_sum(pseudo + zeroed)) & 0xFFFF
    info["tcp_unverifiable_reason"] = reason
    return info


def main():
    # RFC 1071's illustrative 16-bit checksum data.
    assert ((~folded_sum(bytes.fromhex("0001f203f4f5f6f7"))) & 0xFFFF) == 0x220D
    frozen_names = ["normal_event_connections.json", "event_flow_links.csv", "connections.jsonl", "referenced_connections.jsonl", "checksum_mapping_audit.json", "summary.json"]
    before = {name: sha(ROOT / name) for name in frozen_names}
    audit = json.loads((ROOT / "checksum_mapping_audit.json").read_text(encoding="utf-8"))
    source = json.loads((ROOT / "source/cloud-source-receipt.json").read_text(encoding="utf-8"))
    pcap = Path(source["pcap"])
    assert sha(pcap) == source["sha256"]
    spec = importlib.util.spec_from_file_location("ait_checksum_reader", REPO / "scripts/prepare_ait_packet_evidence.py")
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    candidates = {row["packet_connection_id"]: row for row in audit["packet_candidates"]}
    assert len(candidates) == 38
    by_key = defaultdict(list)
    stats, handshakes, samples = {}, {}, []
    for cid, row in candidates.items():
        p = row["packet_envelope"]
        key = tuple(sorted(((p["origin_ip"], p["origin_port"]), (p["responder_ip"], p["responder_port"]))))
        by_key[key].append(cid)
        stats[cid] = {direction: Counter() for direction in DIRECTIONS}
        handshakes[cid] = {"syn": None, "syn_ack": None, "final_ack": None, "max_client_acknowledged_server_sequence_bytes": 0}
    packet_assignments, duplicate_assignments = 0, 0
    for packet_number, (timestamp, linktype, frame, wire) in enumerate(reader.pcap_records(pcap), 1):
        parsed = ipv4(frame, linktype)
        if parsed is None:
            continue
        ip, tcp, meta = parsed
        key = tuple(sorted(((meta["source_ip"], meta["source_port"]), (meta["destination_ip"], meta["destination_port"])) ))
        matches = [cid for cid in by_key.get(key, []) if candidates[cid]["packet_envelope"]["first"] <= timestamp <= candidates[cid]["packet_envelope"]["last"]]
        if not matches:
            continue
        duplicate_assignments += len(matches) > 1
        check = checksum_info(ip, tcp, meta)
        direction = "proxy_to_cloud" if meta["source_ip"] == "172.19.130.68" else "cloud_to_proxy"
        event = {"packet_number": packet_number, "timestamp": timestamp, "direction": direction, **meta, **check}
        for cid in matches:
            packet_assignments += 1
            c = stats[cid][direction]
            c["packets"] += 1
            c["ipv4_declared_bytes_all"] += meta["ip_total_length"]
            if check["tcp_checksum"] == "correct":
                c["ipv4_declared_bytes_tcp_checksum_correct"] += meta["ip_total_length"]
            c["pcap_captured_less_than_original"] += len(frame) < wire
            c["ipv4_checksum_" + check["ipv4_checksum"]] += 1
            c["tcp_checksum_" + check["tcp_checksum"]] += 1
            c["tcp_checksum_verifiable"] += check["tcp_checksum"] != "unverifiable"
            if check["tcp_unverifiable_reason"]:
                c["unverifiable_" + check["tcp_unverifiable_reason"]] += 1
            c["ipv4_total_exceeds_captured"] += check["ipv4_total_exceeds_captured"]
            c["ipv4_total_zero"] += check["ipv4_total_zero"]
            c["tcp_checksum_matches_pseudoheader_seed"] += check["tcp_checksum_matches_pseudoheader_seed"] is True
            c["tcp_incorrect_matches_pseudoheader_seed"] += check["tcp_checksum"] == "incorrect" and check["tcp_checksum_matches_pseudoheader_seed"] is True
            if len(samples) < 20 and (check["tcp_checksum"] != "correct" or check["ipv4_checksum"] != "correct"):
                samples.append({"packet_connection_id": cid, **event})
            h = handshakes[cid]
            flags = meta["flags"]
            if direction == "proxy_to_cloud" and flags & 2 and not flags & 16 and h["syn"] is None:
                h["syn"] = event
            elif direction == "cloud_to_proxy" and flags & 2 and flags & 16 and h["syn"] and meta["acknowledgment"] == (h["syn"]["sequence"] + 1) % (2 ** 32) and h["syn_ack"] is None:
                h["syn_ack"] = event
            elif direction == "proxy_to_cloud" and flags & 16 and not flags & 2 and h["syn_ack"]:
                delta = (meta["acknowledgment"] - h["syn_ack"]["sequence"] - 1) % (2 ** 32)
                if delta < 2 ** 31:
                    h["max_client_acknowledged_server_sequence_bytes"] = max(h["max_client_acknowledged_server_sequence_bytes"], delta)
                if meta["acknowledgment"] == (h["syn_ack"]["sequence"] + 1) % (2 ** 32) and meta["sequence"] == (h["syn"]["sequence"] + 1) % (2 ** 32) and h["final_ack"] is None:
                    h["final_ack"] = event
    aggregate = {direction: Counter() for direction in DIRECTIONS}
    rows = []
    for cid, directions in stats.items():
        p = candidates[cid]["packet_envelope"]
        assert directions["proxy_to_cloud"]["packets"] == p["origin_packets"], cid
        assert directions["cloud_to_proxy"]["packets"] == p["responder_packets"], cid
        for direction, counters in directions.items():
            for key in ["ipv4_checksum_correct", "ipv4_checksum_incorrect", "ipv4_checksum_unverifiable", "tcp_checksum_correct", "tcp_checksum_incorrect", "tcp_checksum_unverifiable"]:
                counters.setdefault(key, 0)
            counters["ipv4_checksum_verifiable"] = counters["ipv4_checksum_correct"] + counters["ipv4_checksum_incorrect"]
            assert counters["packets"] == counters["tcp_checksum_correct"] + counters["tcp_checksum_incorrect"] + counters["tcp_checksum_unverifiable"]
            aggregate[direction].update(counters)
        h = handshakes[cid]
        h["three_way_sequence_numbers_match"] = all(h[k] is not None for k in ["syn", "syn_ack", "final_ack"])
        rows.append({"packet_connection_id": cid, "event_ids": candidates[cid]["event_ids_with_request_second_proxy_payload"], "directions": {k: dict(v) for k, v in directions.items()}, "handshake": h, "frozen_checksum_mapping": candidates[cid]})
    config_root = ROOT / "source/capture_config"
    facts_path = config_root / "gather/cloud_share/facts.json"
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    facts_selected = {
        "snapshot": facts.get("ansible_date_time"), "default_ipv4": facts.get("ansible_default_ipv4"),
        "ens3_features": facts["ansible_ens3"]["features"],
        "suricata_interface": facts.get("suricata_interface"), "suricata_listenmode": facts.get("suricata_listenmode"),
        "kernel": facts.get("ansible_kernel"),
    }
    needed_flow_ids = {f["flow_id"] for row in audit["packet_candidates"] for arm in row["arms"].values() for f in arm}
    zeek_counts = {mode: {direction: Counter() for direction in DIRECTIONS} for mode in ["strict", "ignore_checksums"]}
    zeek_states = {mode: Counter() for mode in zeek_counts}
    with (ROOT / "referenced_connections.jsonl").open(encoding="utf-8") as f:
        for line in f:
            flow = json.loads(line)
            if flow["flow_id"] not in needed_flow_ids:
                continue
            mode = flow["variant"]
            zeek_states[mode][str(flow["conn_state"])] += 1
            for side in ["orig", "resp"]:
                direction = "proxy_to_cloud" if (flow["source"] == "172.19.130.68") == (side == "orig") else "cloud_to_proxy"
                zeek_counts[mode][direction]["packets"] += flow[side + "_pkts"]
                zeek_counts[mode][direction]["ip_bytes"] += flow[side + "_ip_bytes"]
    conservation = {}
    for direction in DIRECTIONS:
        raw = aggregate[direction]
        conservation[direction] = {
            "strict_packets_equal_checksum_correct_packets": zeek_counts["strict"][direction]["packets"] == raw["tcp_checksum_correct"],
            "strict_ip_bytes_equal_checksum_correct_ip_bytes": zeek_counts["strict"][direction]["ip_bytes"] == raw["ipv4_declared_bytes_tcp_checksum_correct"],
            "ignore_packets_equal_all_captured_packets": zeek_counts["ignore_checksums"][direction]["packets"] == raw["packets"],
            "ignore_ip_bytes_equal_all_declared_ip_bytes": zeek_counts["ignore_checksums"][direction]["ip_bytes"] == raw["ipv4_declared_bytes_all"],
        }
        assert all(conservation[direction].values()), (direction, conservation[direction])
    assert facts_selected["ens3_features"]["tx_checksumming"] == "on"
    config_evidence = []
    for path in config_root.rglob("*"):
        if path.is_file() and path.suffix in [".yml", ".yaml", ".j2"]:
            matching = [{"line": i, "text": text} for i, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if any(word in text.lower() for word in ["checksum", "offload", "listenmode", "interface: ens3", "suricata_interface", "pcap-log:", "use-stream-depth"])]
            config_evidence.append({"path": str(path), "sha256": sha(path), "lines": matching})
    after = {name: sha(ROOT / name) for name in frozen_names}
    assert before == after
    result = {
        "pcap": source, "selected_packet_candidates": 38, "packet_assignments": packet_assignments,
        "duplicate_packet_assignments": duplicate_assignments,
        "aggregate_by_direction": {k: dict(v) for k, v in aggregate.items()},
        "three_way_handshakes_with_matching_sequence_numbers": sum(r["handshake"]["three_way_sequence_numbers_match"] for r in rows),
        "handshakes_with_incorrect_syn_ack_tcp_checksum": sum(r["handshake"]["syn_ack"] is not None and r["handshake"]["syn_ack"]["tcp_checksum"] == "incorrect" for r in rows),
        "connections_with_client_acknowledging_server_payload": sum(r["handshake"]["max_client_acknowledged_server_sequence_bytes"] > 0 for r in rows),
        "zeek_directional_totals": {mode: {direction: dict(c) for direction, c in arms.items()} for mode, arms in zeek_counts.items()},
        "zeek_connection_states": {mode: dict(c) for mode, c in zeek_states.items()},
        "packet_and_ip_byte_conservation": conservation,
        "capture_host_facts": facts_selected, "capture_facts_sha256": sha(facts_path), "capture_config_evidence": config_evidence,
        "method": {"ipv4_checksum": "One's-complement 16-bit sum over the entire captured IPv4 header, including stored checksum; correct iff 0xffff.", "tcp_checksum": "IPv4 pseudoheader plus full IPv4-declared TCP segment, including stored checksum; correct iff 0xffff. Odd byte padded with zero only for summation.", "unverifiable": "Fragments are not reassembled. Missing declared segment bytes and malformed lengths are reported as unverifiable, never as bad checksums.", "partial_seed_check": "Compare stored TCP checksum with folded pseudoheader sum separately; this is diagnostic evidence, not checksum validation.", "matching": "Fixed 38 packet candidates, exact bidirectional tuple and inclusive observed packet envelope; per-direction packet counts asserted against frozen catalog."},
        "packet_candidates": rows, "sample_bad_or_unverifiable_packets": samples,
        "frozen_inputs_sha256": before, "frozen_inputs_unchanged": before == after,
        "semantic_decision": {
            "canonical_variant_for_audited_candidates": "ignore_checksums",
            "candidate_ids": sorted(candidates),
            "criterion": "Capture-host TX checksum offload enabled, every outgoing bad checksum equals the partial pseudoheader seed, complete valid IPv4/segment lengths, independent sequence-number handshakes and ACKs, and exact strict/ignore packet-and-byte conservation.",
            "strict_role": "checksum-filtered diagnostic, not an alternative interchangeable model input",
            "merge_strict_and_ignore_rows": False,
            "frozen_mapping_modified": False,
            "scope": "These 38 audited cloud-side packet candidates; not automatic clearance of other hosts, captures, or the whole-capture length anomalies.",
            "model_fits": 0, "model_forward_calls": 0,
        },
        "script_sha256": sha(Path(__file__)),
        "official_sources": ["https://docs.zeek.org/en/v8.0.11/quickstart.html", "https://raw.githubusercontent.com/zeek/zeek/v8.0.11/scripts/base/init-bare.zeek", "https://www.rfc-editor.org/rfc/rfc9293.html#section-3.1", "https://raw.githubusercontent.com/torvalds/linux/v4.15/Documentation/networking/checksum-offloads.txt", "https://zenodo.org/records/19483937"],
    }
    (ROOT / "checksum_semantics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["selected_packet_candidates", "packet_assignments", "duplicate_packet_assignments", "aggregate_by_direction", "three_way_handshakes_with_matching_sequence_numbers", "handshakes_with_incorrect_syn_ack_tcp_checksum", "connections_with_client_acknowledging_server_payload", "frozen_inputs_unchanged"]}), flush=True)


if __name__ == "__main__":
    main()
