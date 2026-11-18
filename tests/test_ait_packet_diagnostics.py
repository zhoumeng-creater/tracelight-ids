"""Synthetic packet/header tests; no real traffic or downloads."""
import importlib.util
import json
from pathlib import Path
import socket
import struct

import pytest


SPEC = importlib.util.spec_from_file_location(
    "packet_diagnostics", Path(__file__).resolve().parents[1] / "scripts/ait_contract/packet_diagnostics.py")
diag = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diag)


def ipv4(protocol=6, declared_extra=0, vlan=0, fragment_offset=0):
    payload = struct.pack("!HH", 50000, 443) + b"private-content-not-for-output"
    header = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 20 + len(payload) + declared_extra,
                         0, fragment_offset, 64, protocol, 0,
                         socket.inet_aton("10.0.0.1"), socket.inet_aton("10.0.0.2"))
    link = b"\0" * 12 + (b"\x81\x00" if vlan else b"\x08\x00")
    for i in range(vlan):
        link += b"\x00\x01" + (b"\x81\x00" if i < vlan - 1 else b"\x08\x00")
    return link + header + payload


def pcap(path, records, linktype=1):
    data = struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, linktype)
    for timestamp, packet, wire_extra in records:
        seconds = int(timestamp)
        data += struct.pack("<IIII", seconds, round((timestamp - seconds) * 1e6),
                            len(packet), len(packet) + wire_extra) + packet
    path.write_bytes(data)


def test_length_definitions_and_half_open_filter_remain_separate(tmp_path):
    source = tmp_path / "source.pcap"
    pcap(source, [(diag.START - 1, ipv4(), 5), (diag.START, ipv4(declared_extra=20), 0),
                  (diag.STOP - 1, ipv4(declared_extra=7), 9), (diag.STOP, ipv4(), 0)])
    result = diag.diagnose(source, tmp_path / "out")
    whole, inside, outside = [result["scopes"][name] for name in (
        "whole_pcap", "r27_interval", "outside_r27_interval")]
    assert whole["counts"]["packets"] == 4
    assert whole["counts"]["pcap_captured_less_than_wire"] == 2
    assert whole["counts"]["ipv4_total_exceeds_captured_ip_bytes"] == 2
    assert inside["counts"]["packets"] == outside["counts"]["packets"] == 2
    assert inside["counts"]["pcap_captured_less_than_wire"] == 1
    assert inside["counts"]["ipv4_total_exceeds_captured_ip_bytes"] == 2
    assert outside["counts"]["ipv4_total_exceeds_captured_ip_bytes"] == 0
    assert whole["first_packet_epoch"] == diag.START - 1
    assert whole["last_packet_epoch"] == diag.STOP
    assert inside["last_packet_epoch"] == diag.STOP - 1
    stored = (tmp_path / "out/packet-diagnostics.json").read_text()
    assert "private-content-not-for-output" not in stored
    assert json.loads(stored) == result


def test_vlan_all_ipv4_protocols_and_fragment_ports_are_explicit(tmp_path):
    source = tmp_path / "source.pcap"
    pcap(source, [(diag.START, ipv4(protocol=p, declared_extra=5, vlan=2,
                                   fragment_offset=1 if p == 17 else 0), 0) for p in (1, 2, 6, 17)])
    result = diag.diagnose(source, tmp_path / "out")
    scope = result["scopes"]["r27_interval"]
    assert scope["ipv4_protocol_counts"] == {"1": 1, "2": 1, "6": 1, "17": 1}
    assert scope["link_layer_counts"] == [{"linktype": 1, "link_layer": "ethernet", "vlan_depth": 2, "packets": 4}]
    groups = {row["ip_protocol"]: row for row in scope["anomaly_groups"]}
    assert groups[6]["source_port"] == 50000 and groups[6]["destination_port"] == 443
    assert groups[17]["source_port"] is None  # Noninitial fragment is not a transport header.
    assert groups[2]["source_ip"] == "10.0.0.1"


def test_samples_are_bounded_but_counts_include_every_packet(tmp_path):
    source = tmp_path / "source.pcap"
    pcap(source, [(diag.START + i, ipv4(declared_extra=9), 0) for i in range(25)])
    result = diag.diagnose(source, tmp_path / "out")
    assert len(result["header_samples"]) == 20
    assert result["scopes"]["whole_pcap"]["counts"]["ipv4_total_exceeds_captured_ip_bytes"] == 25
    assert result["scopes"]["whole_pcap"]["anomaly_groups"][0]["packets"] == 25
    assert not result["packet_payload_retained"]


def test_unknown_link_type_does_not_hide_pcap_length_difference(tmp_path):
    source = tmp_path / "source.pcap"
    pcap(source, [(diag.START, b"unparsed", 5)], linktype=999)
    result = diag.diagnose(source, tmp_path / "out")
    counts = result["scopes"]["whole_pcap"]["counts"]
    assert counts["unsupported_linktype"] == 1
    assert counts["pcap_captured_less_than_wire"] == 1
    assert counts["ipv4_total_exceeds_captured_ip_bytes"] == 0
    assert result["header_samples"][0]["ip_protocol"] is None


def test_partial_record_is_not_written_as_complete_diagnostics(tmp_path):
    source = tmp_path / "source.pcap"
    pcap(source, [(diag.START, ipv4(), 0)])
    source.write_bytes(source.read_bytes()[:-1])
    with pytest.raises(ValueError, match="Truncated PCAP packet"):
        diag.diagnose(source, tmp_path / "out")
    assert not (tmp_path / "out/packet-diagnostics.json").exists()
