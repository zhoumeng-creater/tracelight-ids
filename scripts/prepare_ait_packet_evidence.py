"""Fetch one AIT PCAP member and audit packet/action candidates, never model features.

Standard library only. This entry point performs no fitting, model scoring, Zeek
conversion, FPR estimation, or automatic assignment of attack labels to connections.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import time
import urllib.request
from urllib.parse import urlsplit
import zlib

REPO = Path(__file__).resolve().parents[1]
URL = "https://zenodo.org/api/records/19483937/files/russellmitchell.zip/content"
MEMBER = "gather/intranet_server/logs/suricata/log.pcap.1642684613"
ARCHIVE_BYTES = 7_132_670_599
HEADER_OFFSET = 6_820_279_205
DATA_OFFSET = 6_820_279_319
COMPRESSED_BYTES = 291_883_232
PCAP_BYTES = 343_232_723
PCAP_CRC = 55_161_529
CHUNK_BYTES = 1_048_576
WORKERS = 2
ATTEMPTS = 3
SERVER = "10.143.2.4"  # Author servers.yaml: intranet_server.
START = datetime(2022, 1, 21, tzinfo=timezone.utc).timestamp()
STOP = datetime(2022, 1, 25, tzinfo=timezone.utc).timestamp()
CONNECTION_RULE = (
    "IPv4 TCP bidirectional endpoint tuple; a new initial SYN with a different "
    "sequence/origin starts a new candidate. No Zeek timeout or stream reassembly. "
    "Without an initial SYN, direction is inferred from the fixed server endpoint "
    "when present, otherwise first observed packet. Connections are evidence "
    "candidates, not Zeek flows or pure labelled attack/normal examples."
)


def dump_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest_file(path: Path) -> dict:
    crc, size, sha = 0, 0, hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(CHUNK_BYTES), b""):
            size += len(block)
            crc = zlib.crc32(block, crc)
            sha.update(block)
    return {"bytes": size, "crc32": crc, "sha256": sha.hexdigest()}


def verify_pcap(path: Path, expected_size: int = PCAP_BYTES, expected_crc: int = PCAP_CRC) -> dict:
    actual = digest_file(path)
    if (actual["bytes"], actual["crc32"]) != (expected_size, expected_crc):
        raise ValueError("PCAP is incomplete or differs from the pinned member size/CRC")
    return actual


def fetch_range(start: int, end: int) -> bytes:
    """Bound both blocking reads and wall time; reject non-exact server ranges."""
    began = time.monotonic()
    request = urllib.request.Request(
        f"{URL}?member=intranet-pcap&offset={start}",
        headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        expected = f"bytes {start}-{end}/{ARCHIVE_BYTES}"
        if response.status != 206 or response.headers.get("Content-Range") != expected:
            raise ValueError("Official endpoint did not honor the exact pinned byte range")
        blocks, total = [], 0
        while True:
            if time.monotonic() - began > 90:
                raise TimeoutError("Range exceeded the 90-second wall-time limit")
            block = response.read1(65_536)
            if not block:
                break
            blocks.append(block)
            total += len(block)
            if total > end - start + 1:
                raise ValueError("Range response exceeds requested length")
    if total != end - start + 1:
        raise ValueError("Incomplete range response")
    return b"".join(blocks)


def get_cached_chunk(cache: Path, index: int, start: int, end: int) -> Path:
    part = cache / f"{index:04d}.deflate"
    receipt = part.with_suffix(".json")
    if part.exists() and receipt.exists():
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        actual = digest_file(part)
        if (saved.get("start"), saved.get("end"), saved.get("sha256"), actual["bytes"]) == (
            start, end, actual["sha256"], end - start + 1
        ):
            return part
    error = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            payload = fetch_range(start, end)
            temporary = part.with_suffix(".partial")
            temporary.write_bytes(payload)
            temporary.replace(part)
            dump_json(receipt, {"start": start, "end": end,
                                "sha256": hashlib.sha256(payload).hexdigest()})
            return part
        except (OSError, ValueError) as exc:
            error = exc
            print(f"Chunk {index}: attempt {attempt}/{ATTEMPTS}: {type(exc).__name__}", flush=True)
    raise RuntimeError(f"Chunk {index} exhausted {ATTEMPTS} attempts; verified earlier chunks retained") from error


def extract_cached_member(parts: list[Path], destination: Path, *, expected_size: int = PCAP_BYTES,
                          expected_crc: int = PCAP_CRC,
                          expected_compressed: int = COMPRESSED_BYTES) -> dict:
    """Only promote a fully decoded, exact-size CRC-verified member to PCAP."""
    partial = destination.with_suffix(destination.suffix + ".partial")
    decompressor = zlib.decompressobj(-15)
    received = 0
    try:
        with partial.open("wb") as output:
            for part in parts:
                payload = part.read_bytes()
                received += len(payload)
                output.write(decompressor.decompress(payload))
            output.write(decompressor.flush())
        if received != expected_compressed or not decompressor.eof or decompressor.unused_data:
            raise ValueError("Compressed member is truncated or has extra bytes")
        actual = verify_pcap(partial, expected_size, expected_crc)
        partial.replace(destination)
        return actual
    except Exception:
        # Only our incomplete extraction is removed; resumable compressed chunks remain.
        partial.unlink(missing_ok=True)
        raise


def prepare_pcap(path: Path) -> dict:
    receipt_path = path.with_suffix(path.suffix + ".source.json")
    cache = path.parent / ".ait-intranet-member-cache"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        actual = verify_pcap(path)
        if receipt_path.exists():
            previous = json.loads(receipt_path.read_text(encoding="utf-8"))
            if previous["sha256"] != actual["sha256"]:
                raise ValueError("Existing PCAP SHA-256 differs from its previous receipt")
        reused = True
    else:
        cache.mkdir(exist_ok=True)
        # The pinned central-directory entry was also checked against this local header.
        header = fetch_range(HEADER_OFFSET, HEADER_OFFSET + 29)
        signature, _, flags, method, _, _, crc, compressed, size, nlen, xlen = struct.unpack(
            "<IHHHHHIIIHH", header
        )
        if (signature, flags, method, crc, compressed, size, HEADER_OFFSET + 30 + nlen + xlen) != (
            0x04034B50, 0, 8, PCAP_CRC, COMPRESSED_BYTES, PCAP_BYTES, DATA_OFFSET
        ):
            raise ValueError("ZIP local header differs from the pinned member")
        ranges = [(left, min(left + CHUNK_BYTES, DATA_OFFSET + COMPRESSED_BYTES) - 1)
                  for left in range(DATA_OFFSET, DATA_OFFSET + COMPRESSED_BYTES, CHUNK_BYTES)]
        parts = []
        # At most two requests are in flight; verified chunks persist across runs.
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            jobs = {i: pool.submit(get_cached_chunk, cache, i, *ranges[i])
                    for i in range(min(WORKERS, len(ranges)))}
            for i in range(len(ranges)):
                try:
                    parts.append(jobs.pop(i).result())
                except Exception:
                    for job in jobs.values():
                        job.cancel()
                    raise
                if i + WORKERS < len(ranges):
                    j = i + WORKERS
                    jobs[j] = pool.submit(get_cached_chunk, cache, j, *ranges[j])
                print(f"Verified member chunks: {i + 1}/{len(ranges)}", flush=True)
        actual = extract_cached_member(parts, path)
        reused = False
    # Remove only this script's fixed-member chunks after full PCAP verification.
    removed_bytes, removed_files = 0, 0
    if cache.exists():
        for file in cache.iterdir():
            if file.is_file() and file.stem.isdigit() and file.suffix in {".deflate", ".json", ".partial", ".tmp"}:
                removed_bytes += file.stat().st_size
                file.unlink()
                removed_files += 1
        if not any(cache.iterdir()):
            cache.rmdir()
    result = {**actual, "source_url": URL, "member": MEMBER, "archive_bytes": ARCHIVE_BYTES,
              "header_offset": HEADER_OFFSET, "data_offset": DATA_OFFSET,
              "compressed_bytes": COMPRESSED_BYTES, "reused_without_network": reused,
              "chunk_bytes": CHUNK_BYTES, "parallel_requests": WORKERS,
              "max_attempts_per_chunk": ATTEMPTS, "removed_cache_files": removed_files,
              "removed_cache_logical_bytes": removed_bytes, "full_member_verified": True}
    dump_json(receipt_path, result)
    return result


def pcap_records(path: Path):
    formats = {b"\xd4\xc3\xb2\xa1": ("<", 1e6), b"\xa1\xb2\xc3\xd4": (">", 1e6),
               b"\x4d\x3c\xb2\xa1": ("<", 1e9), b"\xa1\xb2\x3c\x4d": (">", 1e9)}
    with path.open("rb") as source:
        header = source.read(24)
        if len(header) != 24 or header[:4] not in formats:
            raise ValueError("Expected classic PCAP with a complete global header")
        endian, scale = formats[header[:4]]
        linktype = struct.unpack(endian + "I", header[20:24])[0]
        while True:
            record = source.read(16)
            if not record:
                break
            if len(record) != 16:
                raise ValueError("Truncated PCAP record header")
            seconds, fraction, captured, wire = struct.unpack(endian + "IIII", record)
            if captured > 16_777_216:
                raise ValueError("Unsupported PCAP record size")
            payload = source.read(captured)
            if len(payload) != captured:
                raise ValueError("Truncated PCAP packet")
            yield seconds + fraction / scale, linktype, payload, wire


def ipv4_tcp(payload: bytes, linktype: int, counts: Counter):
    if linktype == 1:  # Ethernet, including stacked 802.1Q/802.1ad tags.
        if len(payload) < 14:
            counts["short_link_header"] += 1
            return None
        ether, offset = struct.unpack("!H", payload[12:14])[0], 14
        while ether in {0x8100, 0x88A8}:
            if len(payload) < offset + 4:
                counts["short_vlan_header"] += 1
                return None
            ether = struct.unpack("!H", payload[offset + 2:offset + 4])[0]
            offset += 4
    elif linktype in {113, 276}:  # Linux cooked v1/v2.
        offset = 16 if linktype == 113 else 20
        if len(payload) < offset:
            counts["short_link_header"] += 1
            return None
        ether = struct.unpack("!H", payload[14:16] if linktype == 113 else payload[:2])[0]
    elif linktype == 101:  # Raw IP.
        offset, ether = 0, 0x0800 if payload and payload[0] >> 4 == 4 else 0x86DD
    else:
        counts[f"unsupported_linktype_{linktype}"] += 1
        return None
    if ether != 0x0800:
        counts["ipv6" if ether == 0x86DD else "non_ip"] += 1
        return None
    ip = payload[offset:]
    if len(ip) < 20 or ip[0] >> 4 != 4:
        counts["invalid_ipv4_header"] += 1
        return None
    ihl, total = (ip[0] & 15) * 4, struct.unpack("!H", ip[2:4])[0]
    if ihl < 20 or len(ip) < ihl or total < ihl:
        counts["invalid_ipv4_length"] += 1
        return None
    if struct.unpack("!H", ip[6:8])[0] & 0x3FFF:
        counts["ipv4_fragment_not_reassembled"] += 1
        return None
    if ip[9] != 6:
        counts[f"non_tcp_ipv4_protocol_{ip[9]}"] += 1
        return None
    tcp = ip[ihl:min(total, len(ip))]
    if len(tcp) < 20 or (tcp[12] >> 4) * 4 < 20 or len(tcp) < (tcp[12] >> 4) * 4:
        counts["short_tcp_header"] += 1
        return None
    src, dst = socket.inet_ntoa(ip[12:16]), socket.inet_ntoa(ip[16:20])
    sport, dport, sequence = struct.unpack("!HHI", tcp[:8])
    if len(ip) < total:
        counts["captured_ipv4_shorter_than_wire"] += 1
    return (src, sport), (dst, dport), sequence, tcp[13], len(tcp) - (tcp[12] >> 4) * 4


def parse_connections(path: Path, start: float = START, stop: float = STOP) -> tuple[list[dict], dict]:
    counts, connections, active = Counter(), [], {}
    first_packet, last_packet = None, None
    for timestamp, linktype, packet, _wire in pcap_records(path):
        first_packet = timestamp if first_packet is None else min(first_packet, timestamp)
        last_packet = timestamp if last_packet is None else max(last_packet, timestamp)
        counts["all_pcap_packets"] += 1
        counts["captured_shorter_than_wire_packets"] += len(packet) < _wire
        counts[f"linktype_{linktype}"] += 1
        if not start <= timestamp < stop:
            counts["outside_requested_interval"] += 1
            continue
        counts["in_interval_packets"] += 1
        parsed = ipv4_tcp(packet, linktype, counts)
        if parsed is None:
            continue
        source, target, sequence, flags, payload_bytes = parsed
        counts["parsed_ipv4_tcp_packets"] += 1
        key = tuple(sorted((source, target)))
        syn = bool(flags & 0x02 and not flags & 0x10)
        current = active.get(key)
        if current is None or (syn and (current["syn_sequence"] != sequence or current["origin"] != source)):
            origin, responder, basis = source, target, "initial_syn" if syn else "first_observed_packet"
            if not syn and source[0] == SERVER:
                origin, responder, basis = target, source, "fixed_server_endpoint_without_syn"
            elif not syn and target[0] == SERVER:
                basis = "fixed_server_endpoint_without_syn"
            current = {"connection_id": len(connections), "origin": origin, "responder": responder,
                       "protocol": "tcp", "first": timestamp, "last": timestamp,
                       "syn_seen": syn, "syn_sequence": sequence if syn else None,
                       "direction_basis": basis, "origin_packets": 0, "responder_packets": 0,
                       "origin_captured_tcp_payload_bytes": 0, "responder_captured_tcp_payload_bytes": 0,
                       "fin_packets": 0, "rst_packets": 0}
            connections.append(current)
            active[key] = current
        direction = "origin" if current["origin"] == source else "responder"
        current["first"], current["last"] = min(current["first"], timestamp), max(current["last"], timestamp)
        current[direction + "_packets"] += 1
        current[direction + "_captured_tcp_payload_bytes"] += payload_bytes
        current["fin_packets"] += bool(flags & 0x01)
        current["rst_packets"] += bool(flags & 0x04)
    for connection in connections:
        origin, responder = connection.pop("origin"), connection.pop("responder")
        connection.update(origin_ip=origin[0], origin_port=origin[1],
                          responder_ip=responder[0], responder_port=responder[1])
    return connections, {**counts, "first_pcap_packet_epoch": first_packet,
                         "last_pcap_packet_epoch": last_packet}


def as_epoch(value: str) -> float:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("Action timestamps must include an explicit UTC offset")
    return timestamp.timestamp()


def load_actions(evidence: dict) -> list[dict]:
    actions = []
    for role in evidence["controlled_normal_business"]:
        for action in role["actions"]:
            actions.append({"kind": "controlled_normal_action", "action_id": action["transition_id"],
                            "client_ip": role["apache_observed_ip"],
                            "start": float(action["intent_timestamp"]),
                            "end": float(action["success_timestamp"]),
                            "action_initiated_time": float(action["intent_timestamp"]),
                            "action_succeeded_time": float(action["success_timestamp"]),
                            "normal_confirmation_available_time": None,
                            "normal_confirmation_status": "original_operational_availability_not_reported",
                            "evidence_member": role["member"], "evidence_line": action["intent_line"],
                            "target_url": action["target_url"]})
    command_ips = list(evidence["apache"]["command_client_addresses"])
    for command in evidence["apache"]["commands"]:
        client = command.get("client") or command.get("client_ip")
        if not client:
            if len(command_ips) != 1:
                raise ValueError("Per-command client IP is absent and aggregate IP is not unique")
            client = command_ips[0]
        begin = as_epoch(command["time"])
        actions.append({"kind": "webshell_command_request", "action_id": f'{command["member"]}:{command["line"]}',
                        "client_ip": client, "start": begin, "end": begin + 1.0,
                        "action_initiated_time": None, "action_succeeded_time": None,
                        "request_time": begin, "normal_confirmation_available_time": None,
                        "normal_confirmation_status": "not_applicable_command_request",
                        "evidence_member": command["member"], "evidence_line": command["line"],
                        "target_url": "", "time_resolution": "Apache one-second bin"})
    return actions


def match_actions(connections: list[dict], actions: list[dict]) -> list[dict]:
    rows = []
    for action in actions:
        scheme = urlsplit(action.get("target_url", "")).scheme.lower()
        ports = {443} if scheme == "https" else {80} if scheme == "http" else {80, 443}
        candidates = []
        for connection in connections:
            forward = (connection["origin_ip"] == action["client_ip"]
                       and connection["responder_ip"] == SERVER
                       and connection["responder_port"] in ports)
            reverse_unknown = (not connection.get("syn_seen", False)
                               and connection["responder_ip"] == action["client_ip"]
                               and connection["origin_ip"] == SERVER
                               and connection["origin_port"] in ports)
            if ((forward or reverse_unknown) and connection["first"] < action["end"]
                    and connection["last"] >= action["start"]):
                candidates.append(connection)
        rows.append({**action, "candidate_count": len(candidates),
                     "candidate_allowed_service_ports": ";".join(map(str, sorted(ports))),
                     "match_class": "zero" if not candidates else "one" if len(candidates) == 1 else "multiple",
                     "candidate_connection_ids": ";".join(str(c["connection_id"]) for c in candidates),
                     "candidate_service_ports": ";".join(map(str, sorted({
                         c["origin_port"] if c["origin_ip"] == SERVER else c["responder_port"]
                         for c in candidates}))),
                     "candidate_first_times": ";".join(str(c["first"]) for c in candidates),
                     "candidate_last_times": ";".join(str(c["last"]) for c in candidates),
                     "candidate_direction_confirmed": ";".join(str(c.get("syn_seen", False)) for c in candidates),
                     "association_is_pure_flow_label": False})
    return rows


def download_candidates(connections: list[dict], evidence: dict) -> tuple[list[dict], dict]:
    actions = evidence.get("actor_evidence", {}).get("actions", [])
    candidates, windows = [], []
    for name in ("WPHashCrack", "password list"):
        begins = [a for a in actions if a["message"] == f"Downloading {name}"]
        ends = [a for a in actions if a["message"] == f"Downloaded {name}"]
        if len(begins) != 1 or len(ends) != 1:
            windows.append({"download_name": name, "status": "missing_explicit_download_interval"})
            continue
        begin, end = begins[0]["timestamp"], ends[0]["timestamp"]
        if begin > end:
            raise ValueError("Download action starts after its reported completion")
        overlap = [c for c in connections
                   if ((c.get("syn_seen", False) and c["origin_ip"] == SERVER)
                       or (not c.get("syn_seen", False) and SERVER in {c["origin_ip"], c["responder_ip"]}))
                   and c["first"] <= end and c["last"] >= begin]
        windows.append({"download_name": name, "status": "victim_time_overlap_candidates_only",
                        "start": begin, "end": end, "candidate_count": len(overlap),
                        "outbound_syn_confirmed_candidates": sum(bool(c.get("syn_seen", False)) for c in overlap),
                        "direction_unknown_candidates": sum(not c.get("syn_seen", False) for c in overlap),
                        "match_class": "zero" if not overlap else "one" if len(overlap) == 1 else "multiple"})
        candidates.extend({**c, "download_name": name, "action_start": begin, "action_end": end,
                           "direction_confirmed": bool(c.get("syn_seen", False)),
                           "download_direction_status": "outbound_syn_confirmed" if c.get("syn_seen", False)
                           else "unknown_without_initial_syn",
                           "association_is_download_truth": False} for c in overlap)
    return candidates, {"windows": windows, "endpoint_identity_verified": False,
                        "file_url_or_hash_from_packets_verified": False,
                        "missing_interval_policy": "No download interval is invented when either action endpoint is absent."}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["no_rows"]
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def service_distribution(connections: list[dict]) -> list[dict]:
    """Describe all parsed TCP candidates; no action overlap turns them normal."""
    groups = Counter()
    for connection in connections:
        origin_server = connection["origin_ip"] == SERVER
        responder_server = connection["responder_ip"] == SERVER
        if origin_server or responder_server:
            server_port = connection["origin_port"] if origin_server else connection["responder_port"]
            peer_port = connection["responder_port"] if origin_server else connection["origin_port"]
            direction = ("outbound_syn_confirmed" if origin_server else "inbound_syn_confirmed") \
                if connection["syn_seen"] else "unknown_without_initial_syn"
        else:
            server_port, peer_port, direction = None, None, "no_fixed_server_endpoint"
        groups[(server_port, peer_port, direction)] += 1
    return [{"protocol": "tcp", "server_endpoint_port": key[0], "peer_endpoint_port": key[1],
             "direction_status": key[2], "normal_or_attack_status": "unknown_not_assigned",
             "connection_candidates": value} for key, value in groups.items()]


def audit(pcap: Path, evidence_json: Path, output: Path) -> dict:
    started, cpu_started = time.perf_counter(), time.process_time()
    output.mkdir(parents=True, exist_ok=True)
    dump_json(output / "run-status.json", {"status": "started", "purpose": "packet source evidence only"})
    evidence = json.loads(evidence_json.read_text(encoding="utf-8-sig"))
    actions = load_actions(evidence)
    receipt = prepare_pcap(pcap)
    prepared, prepared_cpu = time.perf_counter(), time.process_time()
    connections, counts = parse_connections(pcap)
    parsed, parsed_cpu = time.perf_counter(), time.process_time()
    matches = match_actions(connections, actions)
    outbound, download = download_candidates(connections, evidence)
    write_csv(output / "tcp-connection-candidates.csv", connections)
    write_csv(output / "tcp-service-distribution.csv", service_distribution(connections))
    write_csv(output / "action-connection-candidates.csv", matches)
    write_csv(output / "victim-download-connection-candidates.csv", outbound)
    distributions = {kind: dict(Counter(m["match_class"] for m in matches if m["kind"] == kind))
                     for kind in sorted({m["kind"] for m in matches})}
    summary = {"status": "packet_source_audit_complete", "eligibility_established": False, "pcap": receipt,
               "evidence_json_sha256": digest_file(evidence_json)["sha256"],
               "start_utc": "2022-01-21T00:00:00Z", "end_exclusive_utc": "2022-01-25T00:00:00Z",
               "connection_rule": CONNECTION_RULE, "packet_counts": counts,
               "field_contract": {"compatible_with_zeek_eight_fields": False,
                                  "packet_counts": "captured IPv4 TCP packets within the stated interval; retransmissions included",
                                  "byte_counts": "captured TCP payload bytes, NOT Zeek orig_ip_bytes/resp_ip_bytes",
                                  "connection_times": "first/last observed packet in requested interval; not Zeek duration",
                                  "direction": "initial SYN confirmed or explicitly inferred; see each connection",
                                  "normal_confirmation_available_time": "null; original operational availability not reported"},
               "input_action_counts": dict(Counter(action["kind"] for action in actions)),
               "connection_candidates": len(connections), "action_candidate_distributions": distributions,
               "download_candidates": download, "model_features_generated": False,
               "models_scored": 0, "fpr_computed": False,
               "resources": {"logical_cpu_count": os.cpu_count(), "network_workers": WORKERS,
                             "parser_processes": 1, "gpu_jobs": 0,
                             "prepare_wall_seconds": prepared - started,
                             "prepare_process_cpu_seconds": prepared_cpu - cpu_started,
                             "parse_wall_seconds": parsed - prepared,
                             "parse_process_cpu_seconds": parsed_cpu - prepared_cpu,
                             "match_and_write_wall_seconds": time.perf_counter() - parsed,
                             "elapsed_seconds": time.perf_counter() - started,
                             "process_cpu_seconds": time.process_time() - cpu_started},
               "remaining_gaps": ["Time/IP overlap is candidate association, even when exactly one connection remains.",
                                  "VPN address is shared; no IP-only attack/normal connection labels.",
                                  "No Zeek extraction, TLS decryption, independent normal denominator, or FPR assessment.",
                                  "Unsupported IP/link/fragment packets, if present, are counted explicitly."]}
    dump_json(output / "packet-evidence-summary.json", summary)
    dump_json(output / "run-status.json", {"status": "packet_source_audit_complete",
                                         "eligibility_established": False,
                                         "model_effect_experiment_complete": False})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcap", type=Path, default=REPO / "data/raw/ait-lds/source-audit-20261115" / MEMBER)
    parser.add_argument("--output", type=Path, default=REPO / "data/derived/ait-packet-source-audit")
    parser.add_argument("--evidence-json", type=Path,
                        default=REPO / "docs/research_alignment_20261115/risk_transfer/source/ait-small-evidence.json")
    args = parser.parse_args()
    try:
        summary = audit(args.pcap, args.evidence_json, args.output)
    except Exception as exc:
        failure = {"status": "incomplete", "error_type": type(exc).__name__, "error": str(exc),
                   "full_source_audit_complete": False,
                   "next_action": "Rerun the same command to resume verified member chunks; no model jobs were started."}
        dump_json(args.output / "run-status.json", failure)
        print(json.dumps(failure, ensure_ascii=False), flush=True)
        return 1
    print(json.dumps({"status": summary["status"], "action_candidates": summary["action_candidate_distributions"],
                      "remaining_gaps": summary["remaining_gaps"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
