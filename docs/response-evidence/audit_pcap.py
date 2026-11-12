"""Reproduce exact conn.log to PCAP matches on two already-opened R22 captures."""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import socket
import struct
import time
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd
import pyarrow.parquet as pq

from ids.contact_evidence.download import ssl_context
from ids.response_evidence.observations import QUALITY_INPUTS, classify_response, counter_gap_frame

HOST = "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios"
# 52 starts at the capture header. The second start is a packet-header offset
# found by time bisection around the archived first 60-1 ^c timestamp.
SOURCES = {
    "52-1": ("2019-03-08-13-24-30-192.168.1.197.pcap", 4904467819,
             "68796959779c4ec5b77bf0d40b220af2b4fa5550fc212725add1b48ae86f61a9"),
    "60-1": ("2019-09-20-02-40-32-192.168.1.195.pcap", 22028429274,
             "4e15902f6c39b1fad60605e667761b10c0c9866beb514aa6320b7baf108ea7c0"),
}
SAMPLE_BYTES = 4 << 20
# An additional block begins just before the median R22 newly alerted ^c row
# (UTC epoch 1569018378.305439). Each block starts at a classic PCAP header.
SEGMENTS = {
    "52-1": ("52-1", 0, "190afecf274c715526e39507cb621380cd873ee9527f3bf42c1f8cecc432676e"),
    "60-1": ("60-1", 20623104324, "0bbc786ef2c0e311e9edce85e9687959bde107a7f284c9da4532af7c5a508b90"),
    "60-1-added": ("60-1", 20775285438, "f6843f594cd3a575dc2271f1f6dd4706a130d45223a24cd8585c42fe8e47a4f5"),
}
HORIZON_US = 1_000_000
RECORD = struct.Struct("<IIII")
R22_PACKAGE_SHA = "1e62b5aaf768b477fee27bfd776e4d444211b03bbea69209de38f5c23996ddb1"


def get_range(url, start, size, total):
    end = start + size - 1
    span = f"{start}-{end}"
    request = Request(f"{url}?span={span}", headers={"Range": f"bytes={span}", "Accept-Encoding": "identity"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=90, context=ssl_context()) as response:
                if response.status != 206 or response.headers["Content-Range"] != f"bytes {span}/{total}":
                    raise ValueError(f"Server did not return requested bytes {span}")
                body = response.read(size + 1)
                if len(body) != size:
                    raise ValueError("Incomplete range")
                return body
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def packet_records(payload, skip, snaplen):
    pos = skip
    while pos + RECORD.size <= len(payload):
        sec, us, size, original_size = RECORD.unpack_from(payload, pos)
        if not (0 < size <= original_size and size <= snaplen and us < HORIZON_US):
            raise ValueError(f"Misaligned packet header at {pos}")
        start = pos + RECORD.size
        if start + size > len(payload):
            break
        yield sec * HORIZON_US + us, pos, payload[start:start + size], original_size
        pos = start + size


def network_type(packet):
    if len(packet) < 14:
        return 0, 0
    ip = 14
    ether = int.from_bytes(packet[12:14], "big")
    while ether in (0x8100, 0x88A8) and len(packet) >= ip + 4:
        ether = int.from_bytes(packet[ip + 2:ip + 4], "big")
        ip += 4
    return ether, ip


def tcp_frame(packet):
    """Fully captured, unfragmented Ethernet IPv4 TCP packets only."""
    ether, ip = network_type(packet)
    if ether != 0x0800 or len(packet) < ip + 20 or packet[ip] >> 4 != 4 or packet[ip + 9] != 6:
        return None
    header = (packet[ip] & 15) * 4
    total = int.from_bytes(packet[ip + 2:ip + 4], "big")
    fragments = int.from_bytes(packet[ip + 6:ip + 8], "big") & 0x3FFF
    if header < 20 or total < header + 20 or len(packet) < ip + total or fragments:
        return None
    hosts = packet[ip + 12:ip + 20]
    tcp = packet[ip + header:ip + total]
    if not 20 <= (tcp[12] >> 4) * 4 <= len(tcp):
        return None
    sp, dp = struct.unpack_from("!HH", tcp)
    endpoints = (socket.inet_ntoa(hosts[:4]), sp, socket.inet_ntoa(hosts[4:]), dp)
    data = hosts + struct.pack("!BBH", 0, 6, len(tcp)) + tcp
    if len(data) % 2:
        data += b"\0"
    summed = sum(struct.unpack(f"!{len(data) // 2}H", data))
    while summed >> 16:
        summed = (summed & 65535) + (summed >> 16)
    return endpoints, summed == 65535, tcp[13]


def raw_ts_us(field):
    sec, fract = field.split(b".")
    if len(fract) > 6:
        raise ValueError("Unexpected sub-microsecond conn.log timestamp")
    return int(sec) * HORIZON_US + int(fract.ljust(6, b"0"))


def group_key(capture, fields):
    details = fields[20].split()
    label, detail = (v.decode() for v in details[-2:])
    if capture == "60-1":
        if {fields[2], fields[4]} != {b"162.248.88.215", b"192.168.1.195"}:
            return None
        if (fields[2], fields[3], label, fields[15]) == (b"162.248.88.215", b"62336", "Benign", b"^c"):
            return "Benign-^c"
        if (fields[4], fields[5], detail, fields[15]) == (b"162.248.88.215", b"62336", "DDoS", b"C"):
            return "DDoS-C"
        return None
    if detail == "PartOfAHorizontalPortScan" and fields[15] == b"S" and fields[11] == b"S0":
        return "Scanning-S0-S"
    if detail in ("C&C", "C&C-FileDownload"):
        return detail
    return None


def frozen_alerts(package, name):
    if hashlib.sha256(package.read_bytes()).hexdigest() != R22_PACKAGE_SHA:
        raise ValueError("Unexpected R22 package")
    with ZipFile(package) as archive:
        policies = json.loads(archive.read("protocol.json"))["policies_in_bit_order"]
        raw_mask = 1 << policies.index("raw")
        final_mask = 1 << policies.index("contact_add")
        buf = BytesIO(archive.read(f"evidence/blind/{name}/hgb.parquet"))
        return pq.read_table(buf, columns=["flags"])["flags"].to_numpy(), raw_mask, final_mask


def audit(segment, directory, package):
    capture, offset, range_sha = SEGMENTS[segment]
    filename, total, expected = SOURCES[capture]
    name = f"CTU-IoT-Malware-Capture-{capture}"
    old = frozen_alerts(package, name) if package else None
    url = f"{HOST}/{name}/{filename}"
    body = get_range(url, offset, SAMPLE_BYTES, total)
    if hashlib.sha256(body).hexdigest() != range_sha:
        raise ValueError("PCAP byte range differs from the registered diagnostic sample")
    header = body[:24] if not offset else get_range(url, 0, 24, total)
    magic, major, minor, _, _, snaplen, link = struct.unpack("<IHHiiII", header)
    if (magic, major, minor, link) != (0xA1B2C3D4, 2, 4, 1):
        raise ValueError("Expected microsecond-resolution Ethernet classic PCAP")
    frames = defaultdict(list)
    times = defaultdict(list)
    frame_count = tcp_count = 0
    skipped = Counter()
    first = last = None
    previous = None
    out_of_order = 0
    for ts, position, packet, _ in packet_records(body, 0 if offset else 24, snaplen):
        frame_count += 1
        out_of_order += int(previous is not None and ts < previous)
        previous = ts
        if first is None:
            first = ts
        last = ts
        parsed = tcp_frame(packet)
        if parsed is not None:
            endpoints, good_checksum, flags = parsed
            times[endpoints].append(ts)
            frames[(ts, endpoints)].append((position + offset, good_checksum, flags))
            tcp_count += 1
        else:
            ether, ip = network_type(packet)
            protocol = packet[ip + 9] if ether == 0x0800 and len(packet) >= ip + 20 else None
            skipped[f"ethertype_0x{ether:04x}_ipproto_{protocol}"] += 1
    if first is None or last <= first:
        raise ValueError("Empty or invalid observation interval")
    for stamps in times.values():
        stamps.sort()
    low, high = str(first // HORIZON_US).encode(), str(last // HORIZON_US).encode()
    rows = defaultdict(Counter)
    states = defaultdict(list)
    examples = defaultdict(list)
    sha = hashlib.sha256()
    ordinal = -1
    with gzip.open(directory / f"{name}.conn.log.labeled.gz", "rb") as log:
        for line in log:
            sha.update(line)
            if line.startswith(b"#"):
                if line.startswith(b"#fields"):
                    columns = line.split()[1:]
                    quality_index = [columns.index(col.encode()) for col in QUALITY_INPUTS]
                continue
            if not line.strip():
                continue
            ordinal += 1
            if not low <= line[:10] <= high:
                continue
            fields = line.rstrip().split(b"\t")
            if fields[6] != b"tcp":
                continue
            key = group_key(capture, fields)
            if key is None:
                continue
            ts = raw_ts_us(fields[0])
            if not first <= ts <= last:
                continue
            counts = rows[key]
            counts["log_starts_in_range"] += 1
            direction = (fields[2].decode(), int(fields[3]), fields[4].decode(), int(fields[5]))
            reverse = direction[2:] + direction[:2]
            a, b = frames.get((ts, direction), []), frames.get((ts, reverse), [])
            if not a and not b:
                counts["no_matching_frame_at_start"] += 1
                continue
            if a and b:
                counts["both_directions_at_exact_start"] += 1
                continue
            observed = "log_originator" if a else "log_responder"
            matches = a or b
            states[key].append([fields[i].decode() for i in quality_index])
            if old is not None:
                flags, raw_mask, final_mask = old
                bits = int(flags[ordinal])
                raw_alert, final_alert = bool(bits & raw_mask), bool(bits & final_mask)
                counts["R22_raw_alert"] += raw_alert
                counts["R22_contact_add_alert"] += final_alert
                counts["R22_new_alert"] += final_alert and not raw_alert
                counts["R22_lost_alert"] += raw_alert and not final_alert
            counts["exact_start_" + observed] += 1
            if all(not m[1] for m in matches):
                counts["all_matching_start_checksums_invalid"] += 1
            if all(m[1] for m in matches):
                counts["all_matching_start_checksums_valid"] += 1
            for flags, name_flags in ((2, "SYN"), (255, "all_eight_bits")):
                if all(m[2] == flags for m in matches):
                    counts["start_tcp_flags_" + name_flags] += 1
            if fields[18] == b"0" and b:
                counts["logged_zero_responder_despite_responder_frame_at_start"] += 1
            if fields[16] == fields[18] == b"0":
                counts["both_counters_zero_despite_frame_at_start"] += 1
            if ts + HORIZON_US <= last:
                rt = times.get(reverse, [])
                responses = bisect_right(rt, ts + HORIZON_US) - bisect_left(rt, ts)
                counts["complete_next_second"] += 1
                counts["responder_frame_in_next_second" if responses else "no_responder_frame_in_next_second"] += 1
            if len(examples[key]) < 2:
                pos, good, flags = matches[0]
                examples[key].append(dict(uid=fields[1].decode(), physical_log_row_zero_based=ordinal,
                                          start_us=ts, logged_tuple=list(direction),
                                          start_frame_direction=observed, pcap_record_byte_offset=pos,
                                          tcp_flags=flags, valid_tcp_checksum=good, history=fields[15].decode(),
                                          orig_pkts=int(fields[16]), resp_pkts=int(fields[18])))
    if sha.hexdigest() != expected:
        raise ValueError("conn.log hash differs from R22 registration")
    for key, items in states.items():
        raw = pd.DataFrame(items, columns=QUALITY_INPUTS)
        codes = classify_response(raw)
        gaps = counter_gap_frame(raw)
        for column in gaps.columns:
            rows[key][column] = int(gaps[column].sum())
        for code in (-1, 0, 1):
            rows[key][f"response_evidence_code_{code}"] = int((codes == code).sum())
    if old is not None and ordinal + 1 != len(old[0]):
        raise ValueError("R22 flags and physical log row counts differ")
    return dict(url=url, pcap_total_bytes=total, range_start=offset, range_bytes=len(body),
                range_sha256=range_sha, conn_log_uncompressed_sha256=expected,
                frozen_r22_package_sha256=R22_PACKAGE_SHA if package else None,
                first_record_us=first, last_record_us=last, all_complete_frames=frame_count,
                fully_captured_tcp_ipv4_frames=tcp_count, other_frames=dict(skipped),
                adjacent_out_of_order_timestamps=out_of_order,
                groups=dict(rows), first_two_examples=dict(examples))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True, help="R22 verified conn.log.labeled.gz directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--r22-package", type=Path, help="Also read sealed R22 HGB alert bits")
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=len(SEGMENTS)) as pool:
        results = dict(zip(SEGMENTS, pool.map(lambda key: audit(key, args.logs, args.r22_package), SEGMENTS)))
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
