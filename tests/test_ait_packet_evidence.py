"""Small synthetic packet evidence and finite resumable-source checks; no network."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import socket
import struct
import zlib

import pytest

SPEC = importlib.util.spec_from_file_location(
    "ait_packet_evidence", Path(__file__).resolve().parents[1] / "scripts/prepare_ait_packet_evidence.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def tcp_packet(src, sport, dst, dport, seq=1, flags=2, data=b"", vlan=False):
    tcp = struct.pack("!HHIIBBHHH", sport, dport, seq, 0, 5 << 4, flags, 4096, 0, 0) + data
    ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 20 + len(tcp), 0, 0, 64, 6, 0,
                     socket.inet_aton(src), socket.inet_aton(dst)) + tcp
    return b"\x00" * 12 + (b"\x81\x00\x00\x01\x08\x00" if vlan else b"\x08\x00") + ip


def write_pcap(path, records, linktype=1, endian="<", nano=False):
    magic = 0xA1B23C4D if nano else 0xA1B2C3D4
    content = struct.pack(endian + "IHHIIII", magic, 2, 4, 0, 0, 65535, linktype)
    scale = 1_000_000_000 if nano else 1_000_000
    for timestamp, packet in records:
        seconds = int(timestamp)
        content += struct.pack(endian + "IIII", seconds, round((timestamp - seconds) * scale),
                               len(packet), len(packet)) + packet
    path.write_bytes(content)
    return content


def raw_deflate(content):
    compressor = zlib.compressobj(wbits=-15)
    return compressor.compress(content) + compressor.flush()


def test_tcp_direction_time_bounds_vlan_and_tuple_reuse(tmp_path):
    server, client = audit.SERVER, "10.143.2.91"
    path = tmp_path / "test.pcap"
    records = [
        (9, tcp_packet(client, 50000, server, 443)),
        (10.25, tcp_packet(client, 50000, server, 443, seq=100, vlan=True)),
        (10.5, tcp_packet(server, 443, client, 50000, seq=800, flags=18, data=b"ok")),
        (11, tcp_packet(client, 50000, server, 443, seq=101, flags=16, data=b"hello")),
        (13, tcp_packet(client, 50000, server, 443, seq=900)),
        (14, tcp_packet(server, 443, client, 50000, seq=1800, flags=18)),
        (20, tcp_packet(client, 50001, server, 443)),
    ]
    write_pcap(path, records)
    connections, counts = audit.parse_connections(path, start=10, stop=20)
    assert counts["all_pcap_packets"] == 7
    assert counts["outside_requested_interval"] == 2
    assert counts["first_pcap_packet_epoch"] == 9 and counts["last_pcap_packet_epoch"] == 20
    assert len(connections) == 2
    first = connections[0]
    assert (first["origin_ip"], first["origin_port"], first["responder_ip"], first["responder_port"]) == (
        client, 50000, server, 443
    )
    assert first["first"] == 10.25 and first["last"] == 11
    assert first["origin_packets"] == 2 and first["responder_packets"] == 1
    assert first["origin_captured_tcp_payload_bytes"] == 5
    assert first["responder_captured_tcp_payload_bytes"] == 2
    assert first["direction_basis"] == "initial_syn"


def test_missing_syn_direction_is_explicit_and_nanosecond_big_endian_reads(tmp_path):
    path = tmp_path / "test.pcap"
    write_pcap(path, [(10.125, tcp_packet(audit.SERVER, 443, "10.0.0.2", 60000, flags=16))],
               endian=">", nano=True)
    connections, _ = audit.parse_connections(path, 10, 11)
    assert connections[0]["first"] == 10.125
    assert connections[0]["origin_ip"] == "10.0.0.2"
    assert connections[0]["direction_basis"] == "fixed_server_endpoint_without_syn"
    assert not connections[0]["syn_seen"]


def test_unsupported_link_ipv6_fragment_and_truncated_pcap_are_not_silent(tmp_path):
    counts = Counter()
    assert audit.ipv4_tcp(b"anything", 999, counts) is None
    assert counts["unsupported_linktype_999"] == 1
    assert audit.ipv4_tcp(b"\0" * 12 + b"\x86\xdd" + b"\x60" * 40, 1, counts) is None
    assert counts["ipv6"] == 1
    fragmented = bytearray(tcp_packet("10.0.0.2", 50000, audit.SERVER, 443))
    fragmented[20:22] = b"\x20\x00"
    assert audit.ipv4_tcp(bytes(fragmented), 1, counts) is None
    assert counts["ipv4_fragment_not_reassembled"] == 1
    path = tmp_path / "bad.pcap"
    content = write_pcap(path, [(10, tcp_packet("10.0.0.2", 50000, audit.SERVER, 443))])
    path.write_bytes(content[:-1])
    with pytest.raises(ValueError, match="Truncated PCAP packet"):
        audit.parse_connections(path, 0, 20)


def test_action_overlap_reports_zero_one_multiple_without_assigning_flow_truth():
    connections = [{"connection_id": i, "origin_ip": "10.0.0.2", "responder_ip": audit.SERVER,
                    "responder_port": 443, "first": begin, "last": end}
                   for i, begin, end in [(0, 10, 15), (1, 12, 13)]]
    actions = [{"kind": "controlled_normal_action", "client_ip": "10.0.0.2", "start": t, "end": t + .5}
               for t in [1, 11, 12]]
    rows = audit.match_actions(connections, actions)
    assert [row["match_class"] for row in rows] == ["zero", "one", "multiple"]
    assert [row["candidate_count"] for row in rows] == [0, 1, 2]
    assert all(not row["association_is_pure_flow_label"] for row in rows)


@pytest.mark.parametrize("kind", ["truncated", "bad_crc"])
def test_incomplete_or_bad_crc_member_never_creates_verified_pcap(tmp_path, kind):
    content = b"synthetic packet source" * 100
    compressed = raw_deflate(content)
    part = tmp_path / "0000.deflate"
    part.write_bytes(compressed[:-1] if kind == "truncated" else compressed)
    destination = tmp_path / "complete.pcap"
    with pytest.raises(ValueError):
        audit.extract_cached_member([part], destination, expected_size=len(content),
                                    expected_crc=zlib.crc32(content) + (kind == "bad_crc"),
                                    expected_compressed=len(compressed))
    assert not destination.exists()
    assert not destination.with_suffix(".pcap.partial").exists()
    assert part.exists()  # This member's resumable chunks remain, not a false complete PCAP.


def test_cached_chunk_reuse_and_bounded_failure(tmp_path, monkeypatch):
    calls = []

    def fetch(start, end):
        calls.append((start, end))
        return b"abc"

    monkeypatch.setattr(audit, "fetch_range", fetch)
    first = audit.get_cached_chunk(tmp_path, 0, 100, 102)
    assert audit.get_cached_chunk(tmp_path, 0, 100, 102) == first
    assert len(calls) == 1
    first.write_bytes(b"bad")  # Equal size alone must not count as a reusable cache hit.
    audit.get_cached_chunk(tmp_path, 0, 100, 102)
    assert len(calls) == 2

    def fail(start, end):
        calls.append((start, end))
        raise TimeoutError("synthetic timeout")

    monkeypatch.setattr(audit, "fetch_range", fail)
    with pytest.raises(RuntimeError, match="exhausted 3 attempts"):
        audit.get_cached_chunk(tmp_path, 1, 103, 105)
    assert len(calls) == 5


def test_verified_existing_member_skips_network_and_cleans_only_member_chunks(tmp_path, monkeypatch):
    pcap = tmp_path / "source.pcap"
    content = write_pcap(pcap, [])
    # Pin the synthetic source for this test without weakening real constants.
    real_verify = audit.verify_pcap
    monkeypatch.setattr(audit, "verify_pcap", lambda p: real_verify(p, len(content), zlib.crc32(content)))
    monkeypatch.setattr(audit, "fetch_range", lambda *_: pytest.fail("Existing verified PCAP must not use network"))
    cache = tmp_path / ".ait-intranet-member-cache"
    cache.mkdir()
    (cache / "0000.deflate").write_bytes(b"cached")
    (cache / "keep-unrelated.txt").write_text("retained")
    result = audit.prepare_pcap(pcap)
    assert result["full_member_verified"] and result["reused_without_network"]
    assert result["removed_cache_files"] == 1
    assert (cache / "keep-unrelated.txt").exists()
    assert json.loads(pcap.with_suffix(".pcap.source.json").read_text())["sha256"] == audit.digest_file(pcap)["sha256"]


def test_actor_download_interval_is_candidate_only_and_missing_start_not_invented():
    connections = [{"connection_id": 0, "origin_ip": audit.SERVER, "responder_ip": "10.0.0.2",
                    "syn_seen": True, "first": 10, "last": 13}]
    evidence = {"actor_evidence": {"actions": [
        {"message": "Downloading WPHashCrack", "timestamp": 11},
        {"message": "Downloaded WPHashCrack", "timestamp": 12},
        {"message": "Downloaded password list", "timestamp": 20},
    ]}}
    candidates, result = audit.download_candidates(connections, evidence)
    assert len(candidates) == 1 and result["windows"][0]["end"] == 12
    assert not result["endpoint_identity_verified"]
    assert result["windows"][1]["status"] == "missing_explicit_download_interval"
    evidence["actor_evidence"]["actions"].append({"message": "Downloading password list", "timestamp": 12})
    candidates, result = audit.download_candidates(connections, evidence)
    assert len(candidates) == 2
    assert [w["match_class"] for w in result["windows"]] == ["one", "one"]
    assert all(not row["association_is_download_truth"] for row in candidates)
    assert all(w["status"] == "missing_explicit_download_interval"
               for w in audit.download_candidates(connections, {})[1]["windows"])


def test_https_action_uses_443_and_confirmation_time_is_not_action_success():
    evidence = {"controlled_normal_business": [{
        "apache_observed_ip": "10.0.0.2", "member": "normal/sm.log", "actions": [{
            "transition_id": "normal-one", "intent_timestamp": 10., "success_timestamp": 11.,
            "intent_line": 1, "target_url": "https://intranet.example/?p=5"}]}],
        "apache": {"command_client_addresses": {"10.0.0.2": 1}, "commands": []}}
    actions = audit.load_actions(evidence)
    connections = [{"connection_id": port, "origin_ip": "10.0.0.2", "origin_port": 50000 + port,
                    "responder_ip": audit.SERVER, "responder_port": port,
                    "first": 9., "last": 12., "syn_seen": True} for port in [80, 443]]
    matched = audit.match_actions(connections, actions)[0]
    assert matched["candidate_count"] == 1
    assert matched["candidate_service_ports"] == "443"
    assert matched["action_succeeded_time"] == 11.
    assert matched["normal_confirmation_available_time"] is None
    assert matched["normal_confirmation_status"] == "original_operational_availability_not_reported"


def test_download_retains_unknown_direction_and_service_summary_does_not_label_normal():
    connections = [
        {"connection_id": 0, "origin_ip": "10.0.0.2", "origin_port": 443,
         "responder_ip": audit.SERVER, "responder_port": 50000, "syn_seen": False, "first": 10, "last": 13},
        {"connection_id": 1, "origin_ip": "10.0.0.3", "origin_port": 50001,
         "responder_ip": audit.SERVER, "responder_port": 443, "syn_seen": True, "first": 10, "last": 13},
        {"connection_id": 2, "origin_ip": audit.SERVER, "origin_port": 50002,
         "responder_ip": "10.0.0.4", "responder_port": 443, "syn_seen": True, "first": 10, "last": 13},
    ]
    evidence = {"actor_evidence": {"actions": [
        {"message": "Downloading WPHashCrack", "timestamp": 11},
        {"message": "Downloaded WPHashCrack", "timestamp": 12}]}}
    candidates, result = audit.download_candidates(connections, evidence)
    assert [row["connection_id"] for row in candidates] == [0, 2]
    assert [row["direction_confirmed"] for row in candidates] == [False, True]
    assert result["windows"][0]["direction_unknown_candidates"] == 1
    services = audit.service_distribution(connections)
    assert sum(row["connection_candidates"] for row in services) == 3
    assert {row["direction_status"] for row in services} == {
        "unknown_without_initial_syn", "inbound_syn_confirmed", "outbound_syn_confirmed"}
    assert all(row["normal_or_attack_status"] == "unknown_not_assigned" for row in services)


def test_complete_audit_writes_all_evidence_without_scoring(tmp_path, monkeypatch):
    evidence_path = Path(__file__).resolve().parents[1] / (
        "docs/research_alignment_20261115/risk_transfer/source/ait-small-evidence.json")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    actions = audit.load_actions(evidence)
    normal = next(a for a in actions if a["kind"] == "controlled_normal_action")
    pcap, output = tmp_path / "synthetic.pcap", tmp_path / "output"
    write_pcap(pcap, [
        (normal["start"], tcp_packet(normal["client_ip"], 54321, audit.SERVER, 443)),
        (normal["end"], tcp_packet(audit.SERVER, 443, normal["client_ip"], 54321, flags=18)),
    ])
    monkeypatch.setattr(audit, "prepare_pcap", lambda path: {**audit.digest_file(path), "synthetic_test": True})
    result = audit.audit(pcap, evidence_path, output)
    assert result["input_action_counts"] == {"controlled_normal_action": 68, "webshell_command_request": 32}
    assert len(result["download_candidates"]["windows"]) == 2
    assert all(w["status"] != "missing_explicit_download_interval"
               for w in result["download_candidates"]["windows"])
    assert result["models_scored"] == 0 and not result["eligibility_established"]
    assert result["resources"]["gpu_jobs"] == 0
    assert result["packet_counts"]["all_pcap_packets"] == 2
    assert {p.name for p in output.iterdir()} == {
        "packet-evidence-summary.json", "run-status.json", "tcp-connection-candidates.csv",
        "tcp-service-distribution.csv", "action-connection-candidates.csv",
        "victim-download-connection-candidates.csv"}
