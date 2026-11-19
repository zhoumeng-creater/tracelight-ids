"""Extract TCP header facts from existing development captures; never inspect payload.

No labels, model scores, calibration sources, packet content, or fitted parameters
are inputs. PCAP record order is retained, including timestamp reversals.
"""
from __future__ import annotations
import argparse
import bisect
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import socket
import struct
import time


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def key(src, sp, dst, dp):
    return tuple(sorted(((src, int(sp)), (dst, int(dp)))))


class Intervals:
    def __init__(self, rows, lifecycle=False):
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            raw = row['raw8']
            if raw['ip_protocol'] != 6:
                continue
            lo = round(row['connection_start'] * 1e6)
            end = row.get('zeek_reported_end') or row['connection_start']
            if lifecycle:
                end = max(end, row.get('replay_export_time') or end)
            groups[key(row['source'], raw['orig_port'], row['destination'], raw['resp_port'])].append((lo - 10, round(end * 1e6) + 10, i))
        self.groups = {}
        for k, values in groups.items():
            values.sort()
            maximum = -1
            ends = []
            for start, end, index in values:
                maximum = max(maximum, end)
                ends.append(maximum)
            self.groups[k] = values, [v[0] for v in values], ends

    def query(self, k, us):
        if k not in self.groups:
            return []
        values, starts, ends = self.groups[k]
        pos = bisect.bisect_right(starts, us) - 1
        found = []
        while pos >= 0 and ends[pos] >= us:
            if values[pos][1] >= us:
                found.append(values[pos][2])
            pos -= 1
        return sorted(found)


def checksum_valid(data):
    value = sum(struct.unpack('!' + str(len(data) // 2) + 'H', data))
    while value >> 16:
        value = (value & 65535) + (value >> 16)
    return value == 65535


def iter_tcp_headers(path, counts, only_host=None):
    """Read Ethernet/IP/TCP headers and seek past every payload.

    TCP options are not read: TCP data offset and captured record size establish
    their availability. IPv6 is explicitly counted as outside this extractor.
    """
    formats = {b'\xd4\xc3\xb2\xa1': ('<', 1000000), b'\xa1\xb2\xc3\xd4': ('>', 1000000), b'\x4d\x3c\xb2\xa1': ('<', 1000000000), b'\xa1\xb2\x3c\x4d': ('>', 1000000000)}
    file_size = Path(path).stat().st_size
    with Path(path).open('rb') as stream:
        header = stream.read(24)
        if len(header) != 24 or header[:4] not in formats:
            raise ValueError('Classic PCAP required')
        endian, factor = formats[header[:4]]
        snaplen, link = struct.unpack_from(endian + 'II', header, 16)
        if link != 1:
            raise ValueError('Only Ethernet PCAP is supported')
        counts['pcap_snaplen'] = snaplen
        index, previous = 0, None
        while True:
            offset = stream.tell()
            record = stream.read(16)
            if not record:
                break
            if len(record) != 16:
                raise ValueError('Incomplete record header')
            seconds, fraction, caplen, original = struct.unpack(endian + 'IIII', record)
            index += 1
            start, end = stream.tell(), stream.tell() + caplen
            if end > file_size:
                raise ValueError('Incomplete packet; use existing complete-record derivative')
            counts['all_complete_records'] += 1
            us = seconds * 1000000 + fraction * 1000000 // factor
            ns = seconds * 1000000000 + fraction * 1000000000 // factor
            counts['adjacent_timestamp_reversals_all'] += previous is not None and ns < previous
            previous = ns
            counts['caplen_less_than_orig_len_all'] += caplen < original
            try:
                if caplen < 14:
                    counts['short_ethernet'] += 1
                    continue
                ethernet = stream.read(14)
                ethertype = struct.unpack_from('!H', ethernet, 12)[0]
                layer = 14
                while ethertype in (0x8100, 0x88a8, 0x9100):
                    if caplen < layer + 4:
                        break
                    vlan = stream.read(4)
                    layer += 4
                    ethertype = struct.unpack_from('!H', vlan, 2)[0]
                if ethertype != 0x0800:
                    counts['non_ipv4_records'] += 1
                    continue
                if caplen < layer + 20:
                    counts['short_ipv4_header'] += 1
                    continue
                ip = stream.read(20)
                ihl = (ip[0] & 15) * 4
                total = struct.unpack_from('!H', ip, 2)[0]
                if ip[0] >> 4 != 4 or ihl < 20 or total < ihl or caplen < layer + ihl:
                    counts['invalid_ipv4_header'] += 1
                    continue
                if ihl > 20:
                    ip += stream.read(ihl - 20)
                if ip[9] != 6:
                    counts['non_tcp_ipv4_records'] += 1
                    continue
                src, dst = socket.inet_ntoa(ip[12:16]), socket.inet_ntoa(ip[16:20])
                if only_host is not None and only_host not in (src, dst):
                    counts['tcp_outside_declared_host'] += 1
                    continue
                fragment = struct.unpack_from('!H', ip, 6)[0]
                if fragment & 0x1fff:
                    counts['nonfirst_ipv4_fragment'] += 1
                    continue
                if min(total - ihl, caplen - layer - ihl) < 20:
                    counts['incomplete_tcp_fixed_header'] += 1
                    continue
                tcp = stream.read(20)
                sp, dp, seq, ack = struct.unpack_from('!HHII', tcp, 0)
                thl = (tcp[12] >> 4) * 4
                if thl < 20 or min(total - ihl, caplen - layer - ihl) < thl:
                    counts['invalid_or_incomplete_tcp_header'] += 1
                    continue
                flags = ((tcp[12] & 1) << 8) | tcp[13]
                declared = total - ihl - thl
                captured = min(total, caplen - layer) - ihl - thl
                counts['extracted_tcp_headers'] += 1
                counts['tcp_declared_length_exceeds_capture'] += declared > captured
                counts['tcp_ip_total_gt_1500'] += total > 1500
                counts['tcp_ip_total_gt_9000'] += total > 9000
                counts['tcp_fragmented_first'] += bool(fragment & 0x2000)
                valid = checksum_valid(ip)
                counts['tcp_invalid_ipv4_checksum'] += not valid
                yield dict(record_index=index, pcap_offset=offset, timestamp_us=us,
                           timestamp_ns=ns, source=src, source_port=sp,
                           destination=dst, destination_port=dp, seq=seq, ack=ack,
                           flags=flags, ipv4_header_bytes=ihl, tcp_header_bytes=thl,
                           ip_total_bytes=total, caplen=caplen, original_record_length=original,
                           declared_tcp_payload_bytes=declared, captured_tcp_payload_bytes=captured,
                           ip_fragmented=bool(fragment & 0x3fff), ipv4_checksum_valid=valid,
                           timestamp_resolution_ns=1000000000 // factor)
            finally:
                stream.seek(end)


def source_specs(work):
    for scenario in ('santos', 'fox'):
        for host in ('intranet_server', 'cloud_share'):
            base = work / '拟合来源与划分落实_20261119' / 'ait_extracted' / scenario / host
            receipt = base / 'normalization/normalization_receipt.json'
            r = load(receipt)
            yield dict(name=f'ait_{scenario}_{host}', pcap=Path(r['source_path']),
                       source_sha256=r['source_sha256'], bytes=r['source_bytes'],
                       source_receipt=receipt, connections=base / 'connections.jsonl.gz',
                       role='F_base' if scenario == 'santos' else 'V', only_host=None)
    base = work / '覆盖与历史依据落实_20261119/attack_coverage/full_capture'
    receipt = base / 'extraction_receipt.json'
    r = load(receipt)
    d = r['complete_records_derivation']
    yield dict(name='unraveled_target', pcap=Path(d['processing_pcap']),
               source_sha256=d['processing_sha256'], bytes=d['preserved_source_interval_half_open'][1],
               original_pcap=Path(d['source_pcap']), original_sha256=d['source_sha256'],
               source_receipt=receipt, connections=base / 'base_connections.jsonl.gz',
               annotated_connections=base / 'connections.jsonl.gz',
               ledger=base / 'packet_membership.jsonl.gz',
               role='opened_source_development', only_host='10.1.3.8')


def extract(spec, output):
    begun = time.monotonic()
    if spec['pcap'].stat().st_size != spec['bytes']:
        raise ValueError('PCAP byte count differs from existing receipt')
    rows = list(read_rows(spec['connections']))
    if 'annotated_connections' in spec:
        annotated = list(read_rows(spec['annotated_connections']))
        assert [r['flow_id'] for r in rows] == [r['flow_id'] for r in annotated]
    observed, lifecycle = Intervals(rows), Intervals(rows, lifecycle=True)
    ledger_stream = read_rows(spec['ledger']) if 'ledger' in spec else None
    ledger_row = next(ledger_stream, None) if ledger_stream is not None else None
    counts = Counter()
    mapped = set()
    target = output / (spec['name'] + '.jsonl.gz')
    with gzip.open(target, 'wt', encoding='utf-8', compresslevel=1, newline='\n') as stream:
        for p in iter_tcp_headers(spec['pcap'], counts, spec['only_host']):
            k = key(p['source'], p['source_port'], p['destination'], p['destination_port'])
            oi = observed.query(k, p['timestamp_us'])
            if ledger_stream is not None:
                while ledger_row is not None and ledger_row[0] < p['record_index']:
                    ledger_row = next(ledger_stream, None)
                assert ledger_row is not None and ledger_row[0] == p['record_index']
                assert ledger_row[1:5] == [p['pcap_offset'], p['timestamp_us'], p['caplen'], p['ip_total_bytes']]
                indices = ledger_row[6]
                basis = 'existing_packet_membership_ledger'
            else:
                indices = lifecycle.query(k, p['timestamp_us'])
                basis = 'bidirectional_tuple_zeek_lifecycle_all_candidates'
            counts['packets_without_connection_candidate'] += not indices
            counts['packets_with_multiple_connection_candidates'] += len(indices) > 1
            counts['packets_with_observed_candidates'] += bool(oi)
            counts['packets_with_observed_multiple_candidates'] += len(oi) > 1
            counts['packets_with_lifecycle_only_candidates'] += bool(indices) and not oi
            mapped.update(indices)
            p['connection_indices'] = indices
            p['observed_interval_connection_indices'] = oi
            p['mapping_basis'] = basis
            stream.write(json.dumps(p, ensure_ascii=False, separators=(',', ':')) + '\n')
    connections_path = output / (spec['name'] + '.connections.jsonl.gz')
    with gzip.open(connections_path, 'wt', encoding='utf-8', compresslevel=1, newline='\n') as stream:
        for i, row in enumerate(rows):
            keep = {k: row.get(k) for k in ('uid','flow_id','stable_flow_id','source','destination','raw8','connection_start','zeek_reported_end','replay_export_time','after_input_eof','during_shutdown','missed_bytes','conn_state','history','raw8_missing_fields')}
            keep['connection_index'] = i
            stream.write(json.dumps(keep, ensure_ascii=False, separators=(',', ':')) + '\n')
    summary = dict(source_name=spec['name'], role=spec['role'], source_pcap=str(spec['pcap']),
                   source_bytes=spec['bytes'], source_sha256=spec['source_sha256'],
                   source_sha256_authority=str(spec['source_receipt']), source_size_rechecked=True,
                   whole_source_sha256_recomputed_in_this_step=False,
                   original_pcap=str(spec.get('original_pcap', spec['pcap'])),
                   original_sha256=spec.get('original_sha256', spec['source_sha256']),
                   header_file=str(target), header_sha256=sha(target), header_bytes=target.stat().st_size,
                   connection_file=str(connections_path), connection_file_sha256=sha(connections_path),
                   source_connections=str(spec['connections']),
                   packet_membership_ledger=str(spec.get('ledger')) if spec.get('ledger') else None,
                   source_connection_count=len(rows), mapped_connection_count=len(mapped),
                   scope='all IPv4 TCP records' if spec['only_host'] is None else 'all IPv4 TCP records involving '+spec['only_host'],
                   counts=dict(counts), seconds=round(time.monotonic()-begun, 3),
                   payload_content_read=False, training_or_scoring=False)
    (output / (spec['name'] + '.receipt.json')).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'complete': spec['name'], 'headers': counts['extracted_tcp_headers'], 'seconds':summary['seconds']}, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', action='append', choices=['ait_santos_intranet_server','ait_santos_cloud_share','ait_fox_intranet_server','ait_fox_cloud_share','unraveled_target'])
    args = parser.parse_args()
    args.work = args.work.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    specs = [s for s in source_specs(args.work) if not args.source or s['name'] in args.source]
    with ThreadPoolExecutor(max_workers=2) as pool:
        summaries = list(pool.map(lambda spec: extract(spec, args.output), specs))
    (args.output/'extraction_summary.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
