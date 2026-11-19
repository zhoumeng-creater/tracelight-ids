"""Pinned Zeek 8.0.11 export, with private Ubuntu 24.04 WSL preparation.

No system install, service startup, source PCAP download, or model work. The
public functions prepare_runtime/export_pcap/self_test are used by the batch.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import time
import urllib.request

HERE = Path(__file__).resolve().parent
VERSION = "8.0.11"
MODES = {"strict": [], "ignore_checksums": ["-C"]}
LOGS = ("conn", "http", "ssl", "dns", "files", "weird")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def invoke(args, *, cwd=None, env=None, timeout=120):
    return subprocess.run(args, cwd=cwd, env=env, text=True, encoding="utf-8", errors="replace",
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=timeout)


def linux_path(path: Path, distro: str) -> str:
    return invoke(["wsl", "-d", distro, "--exec", "wslpath", "-a", str(path.resolve())]).stdout.strip()


def fetch_package(spec: dict, directory: Path) -> Path:
    path = directory / spec["url"].rsplit("/", 1)[1]
    if path.exists() and path.stat().st_size == spec["bytes"] and sha256(path) == spec["sha256"]:
        return path
    partial = path.with_suffix(path.suffix + ".partial")
    for attempt in range(1, 3):
        try:
            print(f"Runtime package {path.name}: attempt {attempt}/2", flush=True)
            with urllib.request.urlopen(spec["url"], timeout=45) as source, partial.open("wb") as target:
                while block := source.read(1 << 20):
                    target.write(block)
            if partial.stat().st_size != spec["bytes"] or sha256(partial) != spec["sha256"]:
                raise ValueError("Pinned runtime package differs: " + path.name)
            partial.replace(path)
            break
        except OSError as exc:
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise RuntimeError("Runtime package download failed: " + path.name) from exc
        except Exception:
            partial.unlink(missing_ok=True)
            raise
    return path


def prepare_runtime(repo: Path, runtime="auto", distro="Ubuntu") -> dict:
    """Use an exact native version or unpack fixed packages into repo/local.

    Ubuntu/WSL itself must already exist. We never install a distribution,
    modify apt sources, invoke sudo, or change global Python/model packages.
    """
    repo = Path(repo).resolve()
    if runtime not in {"auto", "native", "wsl"}:
        raise ValueError("Runtime must be auto, native, or wsl")
    binary = shutil.which("zeek")
    if binary and runtime in {"auto", "native"}:
        version = invoke([binary, "--version"]).stdout.strip()
        if version.endswith(" version " + VERSION):
            return {"kind": "native", "binary": binary, "version": VERSION, "version_output": version,
                    "export_script": str(HERE / "export.zeek"), "private_install": False,
                    "binary_sha256": sha256(Path(binary)), "gnu_time": shutil.which("time")}
        if runtime == "native":
            raise RuntimeError(f"Expected Zeek {VERSION}; found {version}")
    elif runtime == "native":
        raise RuntimeError(f"Native Zeek {VERSION} is not installed")
    if os.name != "nt" or not shutil.which("wsl"):
        raise RuntimeError(f"Install/use native Zeek {VERSION}, or use an existing Ubuntu 24.04 WSL on Windows")
    system = invoke(["wsl", "-d", distro, "--exec", "cat", "/etc/os-release"]).stdout
    arch = invoke(["wsl", "-d", distro, "--exec", "uname", "-m"]).stdout.strip()
    if 'ID=ubuntu' not in system or 'VERSION_ID="24.04"' not in system or arch != "x86_64":
        raise RuntimeError("Private fixed packages require existing Ubuntu 24.04 amd64 WSL")
    cache = repo / "local/ait-zeek-runtime"
    cache.mkdir(parents=True, exist_ok=True)
    rootfs = cache / "rootfs"
    rootfs.mkdir(exist_ok=True)
    manifest = json.loads((HERE / "runtime-packages.json").read_text(encoding="utf-8"))
    receipt_path = cache / "runtime-receipt.json"
    package_binding = sha256(HERE / "runtime-packages.json")
    old = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {}
    linux_root = linux_path(rootfs, distro)
    zeek = linux_root + "/opt/zeek/bin/zeek"
    if old.get("package_manifest_sha256") != package_binding:
        with ThreadPoolExecutor(max_workers=2) as pool:
            packages = list(pool.map(lambda p: fetch_package(p, cache), manifest["packages"]))
        for package in packages:
            invoke(["wsl", "-d", distro, "--exec", "dpkg-deb", "-x", linux_path(package, distro), linux_root])
    result = {"kind": "wsl", "distro": distro, "binary": zeek, "version": VERSION,
              "library_path": linux_root + "/usr/lib/x86_64-linux-gnu:" + linux_root + "/opt/zeek/lib",
              "zeek_path": ":".join(linux_root + "/opt/zeek/share/zeek/" + p for p in ("", "policy", "site", "builtin-plugins")),
              "export_script": str(HERE / "export.zeek"), "private_install": True,
              "package_manifest_sha256": package_binding, "rootfs": str(rootfs),
              "packages": manifest["packages"]}
    output = invoke(["wsl", "-d", distro, "--exec", "env", "LD_LIBRARY_PATH=" + result["library_path"],
                     zeek, "--version"])
    version = output.stdout.strip()
    if not version.endswith(" version " + VERSION):
        raise RuntimeError("Prepared binary version differs: " + version)
    result["version_output"] = version
    result["binary_sha256"] = sha256(rootfs / "opt/zeek/bin/zeek")
    probe = subprocess.run(["wsl", "-d", distro, "--exec", "test", "-x", "/usr/bin/time"])
    result["gnu_time"] = "/usr/bin/time" if probe.returncode == 0 else None
    save(receipt_path, result)
    return result


def command_for(pcap: Path, output: Path, mode: str, runtime: dict):
    if mode not in MODES:
        raise ValueError("Unknown checksum mode")
    script = Path(runtime["export_script"])
    timer = [runtime["gnu_time"], "-v", "-o", "zeek-resource.txt"] if runtime.get("gnu_time") else []
    if runtime["kind"] == "native":
        return [*timer, runtime["binary"], *MODES[mode], "-r", str(pcap.resolve()), str(script.resolve())], {**os.environ, "LC_ALL": "C"}
    distro = runtime["distro"]
    # All changing values are positional arguments; no paths are interpolated into shell code.
    shell = 'cd "$1" && shift && exec env "$@"'
    return (["wsl", "-d", distro, "--exec", "sh", "-c", shell, "ait-zeek",
             linux_path(output, distro), "LD_LIBRARY_PATH=" + runtime["library_path"],
             "ZEEKPATH=" + runtime["zeek_path"], "LC_ALL=C", *timer, runtime["binary"], *MODES[mode], "-r",
             linux_path(pcap, distro), linux_path(script, distro)], None)


def export_pcap(pcap: Path, output: Path, mode: str, runtime: dict) -> dict:
    pcap, output = Path(pcap).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("*.log")):
        raise FileExistsError("Zeek export directory contains logs; caller must validate/reuse or choose a new directory")
    args, env = command_for(pcap, output, mode, runtime)
    start = time.perf_counter()
    try:
        run = invoke(args, cwd=output, env=env, timeout=3600)
    except subprocess.CalledProcessError as exc:
        (output / "zeek-stderr.txt").write_text(exc.stderr or "", encoding="utf-8")
        raise
    (output / "zeek-stdout.txt").write_text(run.stdout, encoding="utf-8")
    (output / "zeek-stderr.txt").write_text(run.stderr, encoding="utf-8")
    logs = {}
    for name in LOGS:
        path = output / (name + ".log")
        logs[name] = {"exists": path.exists(), "rows": sum(1 for _ in path.open(encoding="utf-8")) if path.exists() else 0,
                      "sha256": sha256(path) if path.exists() else None}
    result = {"status": "complete", "mode": mode, "version": VERSION,
              "runtime_kind": runtime["kind"], "pcap_sha256": sha256(pcap),
              "export_script_sha256": sha256(Path(runtime["export_script"])),
              "wall_seconds": time.perf_counter() - start, "logs": logs,
              "command": args, "clock_contract": "Offline network_time at connection log generation; not detector ingestion. EOF/shutdown rows are marked.",
              "checksum_policy": "validate" if mode == "strict" else "explicitly ignore with -C",
              "model_forward_calls": 0, "gpu_jobs": 0,
              "process_resources": resource_summary(output / "zeek-resource.txt")}
    save(output / "export-receipt.json", result)
    return result


def resource_summary(path: Path) -> dict:
    if not path.exists():
        return {"measured": False, "reason": "GNU time is not available; parent wait CPU is not substituted"}
    fields = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ": " in line:
            key, value = line.strip().rsplit(": ", 1)
            fields[key] = value
    return {"measured": True, "source": "/usr/bin/time -v for this Zeek process",
            "user_cpu_seconds": float(fields["User time (seconds)"]),
            "system_cpu_seconds": float(fields["System time (seconds)"]),
            "max_rss_kib": int(fields["Maximum resident set size (kbytes)"]),
            "exit_status": int(fields["Exit status"])}


def checksum(data: bytes) -> int:
    if len(data) % 2:
        data += b"\0"
    value = sum(struct.unpack("!" + "H" * (len(data) // 2), data))
    while value >> 16:
        value = (value & 65535) + (value >> 16)
    return (~value) & 65535


def synthetic_pcap(path: Path) -> None:
    """HTTP close, unclosed TCP, and corrupt TCP checksum: no real traffic."""
    records = []
    client, server = "192.0.2.10", "192.0.2.20"

    def packet(t, src, sport, dst, dport, seq, ack, flags, data=b"", bad=False):
        raw_src, raw_dst = socket.inet_aton(src), socket.inet_aton(dst)
        tcp = struct.pack("!HHIIBBHHH", sport, dport, seq, ack, 80, flags, 65535, 0, 0) + data
        pseudo = raw_src + raw_dst + struct.pack("!BBH", 0, 6, len(tcp))
        cs = checksum(pseudo + tcp) ^ (1 if bad else 0)
        tcp = tcp[:16] + struct.pack("!H", cs) + tcp[18:]
        ip = struct.pack("!BBHHHBBH4s4s", 69, 0, 20 + len(tcp), 0, 0, 64, 6, 0, raw_src, raw_dst)
        ip = ip[:10] + struct.pack("!H", checksum(ip)) + ip[12:]
        body = b"\0" * 12 + b"\x08\x00" + ip + tcp
        records.append((t, body))

    request = b"GET /normal.txt HTTP/1.1\r\nHost: example.test\r\nConnection: close\r\n\r\n"
    response = b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\nbody"
    packet(1, client, 50000, server, 80, 100, 0, 2)
    packet(1.1, server, 80, client, 50000, 200, 101, 18)
    packet(1.2, client, 50000, server, 80, 101, 201, 16)
    packet(1.3, client, 50000, server, 80, 101, 201, 24, request)
    packet(1.4, server, 80, client, 50000, 201, 101 + len(request), 24, response)
    packet(1.5, server, 80, client, 50000, 201 + len(response), 101 + len(request), 17)
    packet(1.6, client, 50000, server, 80, 101 + len(request), 202 + len(response), 17)
    packet(1.7, server, 80, client, 50000, 202 + len(response), 102 + len(request), 16)
    packet(200, client, 50001, server, 443, 1000, 0, 2)
    packet(201, client, 50002, server, 443, 2000, 0, 2, bad=True)
    content = struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)
    for timestamp, body in records:
        seconds = int(timestamp)
        content += struct.pack("<IIII", seconds, round((timestamp - seconds) * 1e6), len(body), len(body)) + body
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def read_log(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def self_test(runtime: dict, output: Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pcap = output / "synthetic-only.pcap"
    synthetic_pcap(pcap)
    result = {mode: export_pcap(pcap, output / mode, mode, runtime) for mode in MODES}
    strict = read_log(output / "strict/conn.log")
    ignore = read_log(output / "ignore_checksums/conn.log")
    assert {r["id.orig_p"] for r in strict} == {50000, 50001, 50002}
    assert {r["id.orig_p"] for r in ignore} == {50000, 50001, 50002}
    # Zeek retains a rejected-packet connection record with zero accepted packets.
    strict_bad = next(r for r in strict if r["id.orig_p"] == 50002)
    ignore_bad = next(r for r in ignore if r["id.orig_p"] == 50002)
    assert strict_bad["orig_pkts"] == 0 and "C" in strict_bad["history"]
    assert ignore_bad["orig_pkts"] == 1 and "S" in ignore_bad["history"]
    assert any(r["name"] == "bad_TCP_checksum" for r in read_log(output / "strict/weird.log"))
    for rows in [strict, ignore]:
        closed = next(row for row in rows if row["id.orig_p"] == 50000)
        unclosed = next(row for row in rows if row["id.orig_p"] == 50001)
        assert closed["orig_ip_bytes"] > closed["orig_bytes"]
        assert closed["contract_export_network_time"] >= closed["ts"] + closed["duration"]
        assert not closed["contract_during_shutdown"]
        assert unclosed["contract_after_input_eof"] or unclosed["contract_during_shutdown"]
    assert read_log(output / "strict/http.log")[0]["uri"] == "/normal.txt"
    files = read_log(output / "strict/files.log")
    assert any(row.get("sha256") == hashlib.sha256(b"body").hexdigest() for row in files)
    verified = {"status": "passed", "real_zeek_executed": True, "source": "synthetic-only",
                "version": VERSION, "strict_connections": len(strict), "ignore_connections": len(ignore),
                "clock_eof_and_byte_semantics_checked": True, "http_and_file_sha_checked": True,
                "bad_checksum_zero_count_record_checked": True,
                "exports": result}
    save(output / "self-test.json", verified)
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=HERE.parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", choices=["auto", "native", "wsl"], default="auto")
    parser.add_argument("--distro", default="Ubuntu")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--pcap", type=Path)
    parser.add_argument("--mode", choices=list(MODES), default="strict")
    args = parser.parse_args()
    runtime = prepare_runtime(args.repo, args.runtime, args.distro)
    if args.self_test:
        result = self_test(runtime, args.output)
    elif args.pcap:
        result = export_pcap(args.pcap, args.output, args.mode, runtime)
    else:
        parser.error("Supply --self-test or --pcap")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
