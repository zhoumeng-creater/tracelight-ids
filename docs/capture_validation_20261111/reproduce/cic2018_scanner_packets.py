"""Map native one-packet SYN flows to independently observed scanner-side PCAP packets.

A packet authenticates the observation. The interpretation of an in-window directed
packet uses the independently published DistriNet NMAP scenario schedule.
"""

import hashlib
import ipaddress
import json
import struct
import zlib
import numpy as np
import pandas as pd
from audit_paths import RESULTS as ROOT, PCAP as RAW, UQ_WORK as UQ


SYN = UQ / "cic2018_native_packet_probes.parquet"
KEY = ["IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT", "FLOW_START_MILLISECONDS"]
TARGETS_KEY = ["src", "sport", "dst", "dport"]


def ip(n):
    return str(ipaddress.IPv4Address(int(n)))


def packets(e):
    b = (RAW / (e["label"] + ".deflate")).read_bytes()
    raw = zlib.decompress(b, -15)
    assert len(raw) == e["bytes"] and zlib.crc32(raw) == e["crc32"]
    # These five specified pcap files all use classic microsecond little-endian Ethernet.
    assert raw[:4] == b"\xd4\xc3\xb2\xa1" and struct.unpack_from("<I", raw, 20)[0] == 1
    host = int(ipaddress.IPv4Address(e["host"]))
    syn = []
    other = []
    pkt = 0
    cur = 24
    lo = None
    hi = None
    while cur < len(raw):
        sec, us, captured, wire = struct.unpack_from("<IIII", raw, cur)
        assert us < 1000000 and 0 < captured <= wire and cur + 16 + captured <= len(raw)
        ts = sec * 1000000 + us
        lo = min(lo, ts) if lo is not None else ts
        hi = max(hi, ts) if hi is not None else ts
        s = cur + 16
        cur = s + captured
        pkt += 1
        if captured < 34:
            continue
        et = struct.unpack_from(">H", raw, s + 12)[0]
        link = 14
        while et in (0x8100, 0x88A8):
            et = struct.unpack_from(">H", raw, s + link + 2)[0]
            link += 4
        if et != 0x800:
            continue
        v = s + link
        if raw[v] >> 4 != 4 or raw[v + 9] != 6:
            continue
        ihl = (raw[v] & 15) * 4
        if ihl < 20 or v + ihl + 20 > cur:
            continue
        if struct.unpack_from(">H", raw, v + 6)[0] & 0x1FFF:
            continue
        src, dst = struct.unpack_from(">II", raw, v + 12)
        if src != host and dst != host:
            continue
        tcp = v + ihl
        sport, dport = struct.unpack_from(">HH", raw, tcp)
        flags = raw[tcp + 13]
        if flags == 2 and raw[tcp + 12] & 1 == 0:
            length = struct.unpack_from(">H", raw, v + 2)[0]
            ttl = raw[v + 8]
            win = struct.unpack_from(">H", raw, tcp + 14)[0]
            syn.append((src, sport, dst, dport, ts, length, ttl, win, pkt))
        # Directional timestamps provide a descriptive check for a later reverse packet.
        other.append((src, sport, dst, dport, ts))
    assert cur == len(raw)
    obs = pd.DataFrame.from_records(syn, columns=[*TARGETS_KEY, "start_us", "ip_len", "ttl", "win", "packet"])
    obs["member"] = e["label"]
    times = pd.DataFrame.from_records(other, columns=[*TARGETS_KEY, "start_us"])
    return (
        obs,
        times,
        {
            **e,
            "index_first_in_member_one_based": True,
            "packets": pkt,
            "compact_scanner_TCP_events": len(times),
            "scanner_TCP_pure_SYN_packets": len(obs),
            "first_us": lo,
            "last_us": hi,
            "pcap_sha256": hashlib.sha256(raw).hexdigest(),
            "deflate_sha256": hashlib.sha256(b).hexdigest(),
            "crc32_verified": True,
        },
    )


def run():
    syn = pd.read_parquet(SYN)
    flows = pd.read_parquet(UQ / "cic2018_native_endpoints.parquet")
    assert syn.one_based_id.is_unique
    # Duplicate native keys must be counted across all TCP flows in these two scanner hosts.
    flows = flows[flows.PROTOCOL.eq(6)]
    entries = json.loads((RAW / "pcap-members.json").read_bytes())
    measurements = {}
    positive = []
    control = []
    per_packet = []
    for day in sorted({e["day"] for e in entries}):
        host = next(e["host"] for e in entries if e["day"] == day)
        infos = []
        obs = []
        times = []
        for e in [e for e in entries if e["day"] == day]:
            a, b, source = packets(e)
            obs.append(a)
            times.append(b)
            infos.append(source)
        p = pd.concat(obs, ignore_index=True)
        thread = pd.concat(times, ignore_index=True)
        p["IPV4_SRC_ADDR"] = p.src.map({n: ip(n) for n in p.src.unique()})
        p["IPV4_DST_ADDR"] = p.dst.map({n: ip(n) for n in p.dst.unique()})
        p["FLOW_START_MILLISECONDS"] = p.start_us // 1000
        p["L4_SRC_PORT"] = p.sport
        p["L4_DST_PORT"] = p.dport
        p["pcap_multiplicity"] = p.groupby(KEY, sort=False).start_us.transform("size")
        d = syn[syn.day.eq(day) & syn.involves_scanner].copy()
        allnf = flows[flows.day.eq(day)].groupby(KEY, sort=False).size().rename("all_native_tcp_multiplicity")
        d = d.join(allnf, on=KEY)
        assert d.all_native_tcp_multiplicity.notna().all()
        d = d.merge(
            p[KEY + ["start_us", "ip_len", "ttl", "win", "packet", "member", "pcap_multiplicity"]],
            on=KEY,
            how="left",
        )
        d["unique"] = d.all_native_tcp_multiplicity.eq(1) & d.pcap_multiplicity.eq(1)
        d["fields"] = (
            d.IN_BYTES.eq(d.ip_len)
            & d.MAX_IP_PKT_LEN.eq(d.ip_len)
            & d.TCP_WIN_MAX_IN.eq(d.win)
            & d["FLOW_END_MILLISECONDS"].eq(d.FLOW_START_MILLISECONDS)
        )
        d["strict"] = d.unique & d.fields
        d["role"] = np.select(
            [d.match_forward_rule, d.raw_uq_normal_outside_rule],
            ["published_NMAP_window", "native_Benign_outside_directed_rule"],
            default="other",
        )
        # Restrict reverse-event dictionary to reverse tuples of observed strict SYNs.
        strong = d[d.strict].copy()
        strong["src"] = strong.IPV4_SRC_ADDR.map(lambda x: int(ipaddress.IPv4Address(x)))
        strong["dst"] = strong.IPV4_DST_ADDR.map(lambda x: int(ipaddress.IPv4Address(x)))
        lookup = (
            thread.sort_values("start_us")
            .groupby(TARGETS_KEY, sort=False)["start_us"]
            .apply(lambda a: a.to_numpy())
        )
        available = np.array([[a["first_us"], a["last_us"]] for a in infos])

        def reverse_info(r):
            at = int(r.start_us)
            key = (r.dst, r.L4_DST_PORT, r.src, r.L4_SRC_PORT)
            if key in lookup.index:
                rr = lookup.loc[key]
                ix = np.searchsorted(rr, at, side="left")
                if ix < len(rr):
                    return int(rr[ix] - at)
            return -1

        strong["next_reverse_delta_us"] = [reverse_info(row) for row in strong.itertuples()]
        strong["host_cap_extends_1s"] = [
            bool(((available[:, 0] <= x) & (available[:, 1] >= x + 1000000)).any()) for x in strong.start_us
        ]
        strong["host_cap_extends_60s"] = [
            bool(((available[:, 0] <= x) & (available[:, 1] >= x + 60000000)).any()) for x in strong.start_us
        ]
        counts = {}
        for name, g in d.groupby("role"):
            good = strong[strong.role.eq(name)]
            u = g.drop_duplicates("one_based_id")
            counts[name] = {
                "NF_one_packet_rows": len(u),
                "native_directed_tuple_and_ms_observed_in_PCAP": int(
                    g.loc[g.pcap_multiplicity.notna(), "one_based_id"].nunique()
                ),
                "both_unique": int(g.unique.sum()),
                "strict_packet_field_match": len(good),
                "round19_exact47": int(good.exact47_verified_direction_candidate.sum()),
                "native_class": {k: int(v) for k, v in good.Attack.value_counts().items()},
                "NF_TTL_minmax_both_zero": int((good.MIN_TTL.eq(0) & good.MAX_TTL.eq(0)).sum()),
                "actual_TTL_range": [int(good.ttl.min()), int(good.ttl.max())] if len(good) else None,
                "orig_source_is_scanner": int(good.IPV4_SRC_ADDR.eq(host).sum()),
                "destination_is_scanner": int(good.IPV4_DST_ADDR.eq(host).sum()),
                "next_reverse_packet_within_1s": int(good.next_reverse_delta_us.between(0, 1_000_000).sum()),
                "next_reverse_packet_within_60s": int(
                    good.next_reverse_delta_us.between(0, 60_000_000).sum()
                ),
                "inside_capture_time_extent_for_1s": int(good.host_cap_extends_1s.sum()),
                "inside_capture_time_extent_for_60s": int(good.host_cap_extends_60s.sum()),
                "examples": good[
                    [
                        "one_based_id",
                        "member",
                        "packet",
                        "start_us",
                        "IPV4_SRC_ADDR",
                        "L4_SRC_PORT",
                        "IPV4_DST_ADDR",
                        "L4_DST_PORT",
                        "ip_len",
                        "win",
                        "ttl",
                        "Attack",
                        "next_reverse_delta_us",
                    ]
                ]
                .head(3)
                .to_dict("records"),
            }
        positive.extend(strong.loc[strong.role.eq("published_NMAP_window"), "one_based_id"].tolist())
        control.extend(
            strong.loc[strong.role.eq("native_Benign_outside_directed_rule"), "one_based_id"].tolist()
        )
        per_packet.append(
            strong[
                [
                    "one_based_id",
                    "day",
                    "role",
                    "member",
                    "packet",
                    "start_us",
                    "ip_len",
                    "ttl",
                    "win",
                    "next_reverse_delta_us",
                    "host_cap_extends_1s",
                    "host_cap_extends_60s",
                    "exact47_verified_direction_candidate",
                    "Attack",
                    "IPV4_SRC_ADDR",
                    "L4_SRC_PORT",
                    "IPV4_DST_ADDR",
                    "L4_DST_PORT",
                ]
            ]
        )
        measurements[day] = {"source_host": host, "capture_files": infos, "comparison": counts}
        print(
            day,
            [(k, v["strict_packet_field_match"], v["round19_exact47"]) for k, v in counts.items()],
            flush=True,
        )
    ids = {}
    for prefix, values in [("nmap", positive), ("nominal_benign_control", control)]:
        x = np.array(sorted(values), dtype="<u4")
        assert len(set(values)) == len(x)
        path = ROOT / ("cic2018_packet_" + prefix + "_rows.npy")
        np.save(path, x, allow_pickle=False)
        ids[prefix] = {
            "rows": len(x),
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    pd.concat(per_packet, ignore_index=True).to_parquet(
        RAW / "strict_packet_lineage.parquet", compression="zstd", index=False
    )
    out = {
        "matching": "Both native TCP flows and raw pure SYN packets must be unique at the directed IPv4 address/port 5-tuple and first millisecond. Require NF 1 forward packet, zero reverse, pure SYN bits, start=end, NF IN_BYTES=MAX_IP_PKT_LEN=actual IPv4 total_length, and TCP_WIN_MAX_IN=actual TCP receive window. TTL is compared descriptively because the NF values are commonly both zero.",
        "interpretation": "NMAP labels combine packet observation with independent author-documented scanner direction, endpoints, no DHCP, and time window; different evidence source from the corrected-flow match. Controls are observed SYN packets with native Benign label outside this NMAP rule, not individually guaranteed benign. Reverse counts describe later TCP packets at the reverse tuple, not proof of causal replies or packet absence in the physical network.",
        "ids": ids,
        "days": measurements,
    }
    (ROOT / "cic2018_pcap_evidence.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )


if __name__ == "__main__":
    run()
