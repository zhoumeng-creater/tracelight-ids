"""No-network tests for pinned export contracts; real Zeek self-test is explicit."""
import hashlib
import importlib.util
from pathlib import Path
import struct

import pytest

SPEC = importlib.util.spec_from_file_location(
    "ait_zeek_runtime", Path(__file__).resolve().parents[1] / "scripts/ait_contract/zeek_runtime.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


def test_synthetic_capture_has_valid_ip_and_one_bad_tcp_checksum(tmp_path):
    path = tmp_path / "test.pcap"
    runtime.synthetic_pcap(path)
    data = path.read_bytes()
    pos, good, bad = 24, 0, 0
    while pos < len(data):
        _, _, cap, wire = struct.unpack("<IIII", data[pos:pos+16])
        assert cap == wire
        packet = data[pos+16:pos+16+cap]
        ip, tcp = packet[14:34], packet[34:]
        assert runtime.checksum(ip) == 0
        pseudo = ip[12:20] + struct.pack("!BBH", 0, 6, len(tcp))
        if runtime.checksum(pseudo + tcp) == 0:
            good += 1
        else:
            bad += 1
        pos += 16 + cap
    assert (good, bad) == (9, 1)


def test_cached_package_is_hash_bound(tmp_path, monkeypatch):
    package = tmp_path / "example.deb"
    package.write_bytes(b"expected")
    spec = {"url": "https://example.invalid/example.deb", "bytes": 8,
            "sha256": hashlib.sha256(b"expected").hexdigest()}
    monkeypatch.setattr(runtime.urllib.request, "urlopen", lambda *_args, **_kw: pytest.fail("cached bytes should be reused"))
    assert runtime.fetch_package(spec, tmp_path) == package


def test_two_checksum_modes_do_not_change_pcap_or_script(tmp_path):
    config = {"kind": "native", "binary": "/zeek", "export_script": str(tmp_path / "export.zeek")}
    strict, _ = runtime.command_for(tmp_path / "input.pcap", tmp_path, "strict", config)
    ignored, _ = runtime.command_for(tmp_path / "input.pcap", tmp_path, "ignore_checksums", config)
    assert "-C" not in strict and "-C" in ignored
    assert [arg for arg in ignored if arg != "-C"] == strict
    with pytest.raises(ValueError, match="checksum"):
        runtime.command_for(tmp_path / "input.pcap", tmp_path, "choose_best", config)


def test_export_refuses_to_mix_existing_logs(tmp_path):
    (tmp_path / "conn.log").write_text("old evidence")
    with pytest.raises(FileExistsError):
        runtime.export_pcap(tmp_path / "anything.pcap", tmp_path, "strict", {})


def test_missing_resource_measurement_is_not_zero(tmp_path):
    result = runtime.resource_summary(tmp_path / "absent.txt")
    assert not result["measured"]
    assert "user_cpu_seconds" not in result


def test_gnu_time_metrics_are_the_child_process_measurement(tmp_path):
    path = tmp_path / "resources.txt"
    path.write_text("\tUser time (seconds): 1.25\n\tSystem time (seconds): 0.5\n"
                    "\tMaximum resident set size (kbytes): 123456\n\tExit status: 0\n")
    result = runtime.resource_summary(path)
    assert result["user_cpu_seconds"] == 1.25
    assert result["system_cpu_seconds"] == .5
    assert result["max_rss_kib"] == 123456
