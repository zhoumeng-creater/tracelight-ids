"""Describe packet length observations; do not infer capture/offload causes."""
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import socket
import struct


_spec = importlib.util.spec_from_file_location(
    "_ait_packet_reader", Path(__file__).resolve().parents[1] / "prepare_ait_packet_evidence.py")
_reader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reader)
pcap_records = _reader.pcap_records
START, STOP = _reader.START, _reader.STOP


def _headers(packet: bytes, linktype: int) -> dict:
    """Return header metadata only; never retain packet content in a result."""
    meta = {"linktype": linktype, "vlan_depth": 0, "ip_protocol": None,
            "source_ip": None, "destination_ip": None, "source_port": None,
            "destination_port": None, "ipv4_total_length": None,
            "captured_ip_bytes": None, "ipv4_total_exceeds_captured": False}
    if linktype == 1:
        meta["link_layer"] = "ethernet"
        if len(packet) < 14:
            return {**meta, "parse_status": "short_link_header"}
        ether, offset = struct.unpack("!H", packet[12:14])[0], 14
        while ether in {0x8100, 0x88A8}:
            if len(packet) < offset + 4:
                return {**meta, "parse_status": "short_vlan_header"}
            meta["vlan_depth"] += 1
            ether = struct.unpack("!H", packet[offset + 2:offset + 4])[0]
            offset += 4
    elif linktype in {113, 276}:
        meta["link_layer"] = "linux_cooked_v1" if linktype == 113 else "linux_cooked_v2"
        offset = 16 if linktype == 113 else 20
        if len(packet) < offset:
            return {**meta, "parse_status": "short_link_header"}
        ether = struct.unpack("!H", packet[14:16] if linktype == 113 else packet[:2])[0]
    elif linktype == 101:
        meta["link_layer"], offset = "raw_ip", 0
        ether = 0x0800 if packet and packet[0] >> 4 == 4 else 0x86DD
    else:
        return {**meta, "link_layer": "unsupported", "parse_status": "unsupported_linktype"}
    if ether != 0x0800:
        return {**meta, "parse_status": "ipv6" if ether == 0x86DD else "non_ipv4"}
    ip = packet[offset:]
    if len(ip) < 20 or ip[0] >> 4 != 4:
        return {**meta, "parse_status": "short_or_invalid_ipv4_header"}
    ihl = (ip[0] & 15) * 4
    total = struct.unpack("!H", ip[2:4])[0]
    meta.update(ip_protocol=ip[9], source_ip=socket.inet_ntoa(ip[12:16]),
                destination_ip=socket.inet_ntoa(ip[16:20]), ipv4_total_length=total,
                captured_ip_bytes=len(ip), ipv4_total_exceeds_captured=total > len(ip))
    if ihl < 20 or total < ihl or len(ip) < ihl:
        return {**meta, "parse_status": "invalid_ipv4_header_length"}
    # Later fragments do not begin with transport ports.
    fragment_offset = struct.unpack("!H", ip[6:8])[0] & 0x1FFF
    if not fragment_offset and ip[9] in {6, 17} and min(total, len(ip)) >= ihl + 4:
        meta["source_port"], meta["destination_port"] = struct.unpack("!HH", ip[ihl:ihl + 4])
    return {**meta, "parse_status": "ipv4"}


def _scope():
    return {"counts": Counter({"packets": 0, "pcap_captured_less_than_wire": 0,
                                "ipv4_total_exceeds_captured_ip_bytes": 0}),
            "protocols": Counter(), "link_layers": Counter(),
            "anomalies": Counter(), "first": None, "last": None}


def _observe(scope, timestamp, headers, captured, wire):
    scope["first"] = timestamp if scope["first"] is None else min(scope["first"], timestamp)
    scope["last"] = timestamp if scope["last"] is None else max(scope["last"], timestamp)
    counts = scope["counts"]
    counts["packets"] += 1
    counts["pcap_captured_less_than_wire"] += captured < wire
    counts["ipv4_total_exceeds_captured_ip_bytes"] += headers["ipv4_total_exceeds_captured"]
    counts[headers["parse_status"]] += 1
    scope["link_layers"][(headers["linktype"], headers["link_layer"], headers["vlan_depth"])] += 1
    if headers["ip_protocol"] is not None:
        scope["protocols"][headers["ip_protocol"]] += 1
    kinds = (["pcap_captured_less_than_wire"] if captured < wire else [])
    if headers["ipv4_total_exceeds_captured"]:
        kinds.append("ipv4_total_exceeds_captured_ip_bytes")
    for kind in kinds:
        scope["anomalies"][(kind, headers["linktype"], headers["link_layer"], headers["vlan_depth"],
                            headers["ip_protocol"], headers["source_ip"], headers["destination_ip"],
                            headers["source_port"], headers["destination_port"])] += 1
    return kinds


def _finish(scope):
    def utc(value):
        return datetime.fromtimestamp(value, timezone.utc).isoformat() if value is not None else None

    group_fields = ("anomaly", "linktype", "link_layer", "vlan_depth", "ip_protocol",
                    "source_ip", "destination_ip", "source_port", "destination_port")
    return {"counts": dict(scope["counts"]),
            "first_packet_epoch": scope["first"], "last_packet_epoch": scope["last"],
            "first_packet_utc": utc(scope["first"]), "last_packet_utc": utc(scope["last"]),
            "ipv4_protocol_counts": {str(k): v for k, v in sorted(scope["protocols"].items())},
            "link_layer_counts": [dict(zip(("linktype", "link_layer", "vlan_depth"), key), packets=value)
                                  for key, value in sorted(scope["link_layers"].items())],
            "anomaly_groups": [dict(zip(group_fields, key), packets=value)
                               for key, value in sorted(scope["anomalies"].items(), key=lambda item: str(item[0]))]}


def diagnose(pcap: Path, output: Path) -> dict:
    """Stream PCAP once; write output/packet-diagnostics.json and return its data."""
    scopes = {name: _scope() for name in ("whole_pcap", "r27_interval", "outside_r27_interval")}
    samples = []
    for timestamp, linktype, packet, wire in pcap_records(Path(pcap)):
        in_interval = START <= timestamp < STOP
        headers = _headers(packet, linktype)
        kinds = _observe(scopes["whole_pcap"], timestamp, headers, len(packet), wire)
        _observe(scopes["r27_interval" if in_interval else "outside_r27_interval"],
                 timestamp, headers, len(packet), wire)
        if kinds and len(samples) < 20:
            samples.append({"timestamp_epoch": timestamp, "in_r27_interval": in_interval,
                            "captured_frame_bytes": len(packet), "wire_frame_bytes": wire,
                            "anomalies": kinds, **headers})
    result = {"schema_version": 1, "status": "complete", "interval_start_epoch": START,
              "interval_end_exclusive_epoch": STOP,
              "definitions": {
                  "pcap_captured_less_than_wire": "PCAP captured frame length < PCAP original frame length",
                  "ipv4_total_exceeds_captured_ip_bytes": "IPv4 total_length > captured bytes after the link/VLAN header",
                  "overlap": "An individual packet can contribute to both anomaly counters.",
                  "protocol": "Numeric IPv4 protocol field, including non-TCP traffic and fragments.",
                  "causal_inference": "None: these facts do not establish offload, packet loss, or corruption."},
              "scopes": {name: _finish(value) for name, value in scopes.items()},
              "header_samples": samples, "header_sample_limit": 20, "packet_payload_retained": False}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "packet-diagnostics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result
