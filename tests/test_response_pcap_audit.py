"""Verify the historical packet audit's checksum, framing and rejection rules."""

import importlib.util
from pathlib import Path
import struct

from pytest import raises

PATH = Path(__file__).resolve().parents[1] / "docs/response-evidence/audit_pcap.py"
SPEC = importlib.util.spec_from_file_location("response_pcap_audit", PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

ETH = bytes.fromhex("0000000000020000000000010800")
IP = bytes.fromhex("450000280000400040060000c0000201c6336402")
TCP = bytes.fromhex("c0000050000000010000000050022000e3590000")


def test_valid_checksum_and_invalid_checksum_have_same_direction():
    endpoints = ("192.0.2.1", 49152, "198.51.100.2", 80)
    assert audit.tcp_frame(ETH + IP + TCP) == (endpoints, True, 2)
    changed = bytearray(TCP)
    changed[16] ^= 1
    assert audit.tcp_frame(ETH + IP + changed) == (endpoints, False, 2)


def test_partial_fragment_and_truncated_segment_are_excluded():
    assert audit.tcp_frame(ETH + IP + TCP[:-1]) is None
    assert audit.tcp_frame(ETH + IP[:6] + b"\x20\x00" + IP[8:] + TCP) is None


def test_exact_pcap_boundaries_and_microseconds():
    frame = ETH + IP + TCP
    rec = struct.pack("<IIII", 1500000000, 7, len(frame), len(frame)) + frame
    assert list(audit.packet_records(rec + rec[:-1], 0, 65535)) == [(1500000000000007, 0, frame, len(frame))]
    assert audit.raw_ts_us(b"1500000000.000007") == 1500000000000007
    with raises(ValueError, match="Misaligned"):
        list(audit.packet_records(bytes(16), 0, 65535))
