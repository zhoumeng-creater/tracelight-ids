"""Label-independent declared-IPv4-length representation of classic PCAP.

Only record orig_len may increase. Timestamp, caplen and captured bytes are never
changed. Declared datagram bytes/capture records are not physical wire traffic.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
from pathlib import Path
import socket
import struct

CONTRACT = 'declared-ip-truncation-v1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


@contextmanager
def gzip_lines(path):
    with Path(path).open('wb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding='utf-8', newline='\n') as text:
                yield text


def folded_sum(data):
    if len(data) % 2:
        data += b'\0'
    value = sum(struct.unpack('!'+str(len(data)//2)+'H', data))
    while value >> 16:
        value = (value & 65535) + (value >> 16)
    return value


class Capture:
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open('rb') as stream:
            self.header = stream.read(24)
        formats = {bytes.fromhex('d4c3b2a1'):('<', 1000000), bytes.fromhex('a1b2c3d4'):('>',1000000), bytes.fromhex('4d3cb2a1'):('<',1000000000), bytes.fromhex('a1b23c4d'):('>',1000000000)}
        if len(self.header) != 24 or self.header[:4] not in formats:
            raise ValueError('Expected classic PCAP, not PCAPNG or a partial header')
        self.endian, self.factor = formats[self.header[:4]]
        self.linktype = struct.unpack(self.endian+'I', self.header[20:24])[0]

    def records(self):
        with self.path.open('rb') as stream:
            stream.seek(24)
            index = 0
            while True:
                header = stream.read(16)
                if not header:
                    return
                if len(header) != 16:
                    raise ValueError('Partial PCAP record header')
                seconds, fraction, caplen, original = struct.unpack(self.endian+'IIII', header)
                data = stream.read(caplen)
                if len(data) != caplen:
                    raise ValueError('Partial captured record bytes')
                index += 1
                yield index, seconds+fraction/self.factor, header, data, original


def inspect_frame(frame, linktype):
    """Return intrinsic header facts only; no host, time interval or label input."""
    if linktype != 1:
        return {'status':'unsupported_linktype', 'eligible':False}
    if len(frame) < 14:
        return {'status':'incomplete_ethernet_header', 'eligible':False}
    offset, ethertype = 14, struct.unpack('!H', frame[12:14])[0]
    while ethertype in (0x8100,0x88a8,0x9100):
        if len(frame) < offset+4:
            return {'status':'incomplete_vlan_header', 'eligible':False}
        ethertype = struct.unpack('!H',frame[offset+2:offset+4])[0]
        offset += 4
    if ethertype != 0x0800:
        return {'status':'not_ipv4', 'eligible':False}
    ip = frame[offset:]
    if len(ip) < 20 or ip[0] >> 4 != 4:
        return {'status':'incomplete_or_invalid_ipv4_header', 'eligible':False}
    ihl = (ip[0] & 15)*4
    total = struct.unpack('!H',ip[2:4])[0]
    facts = dict(eligible=False, L2_offset=offset, ipv4_total_length=total,
                 captured_ip_bytes=len(ip), declared_frame_length=offset+total,
                 declared_missing_ip_bytes=max(0,total-len(ip)), ip_protocol=ip[9],
                 source=socket.inet_ntoa(ip[12:16]), destination=socket.inet_ntoa(ip[16:20]),
                 source_port=None, destination_port=None)
    if ihl < 20 or ihl > len(ip) or total < ihl:
        return {**facts,'status':'invalid_or_incomplete_ipv4_header_length'}
    if folded_sum(ip[:ihl]) != 65535:
        return {**facts,'status':'invalid_ipv4_header_checksum'}
    facts['ipv4_header_checksum_valid'] = True
    if struct.unpack('!H',ip[6:8])[0] & 0x3fff:
        return {**facts,'status':'fragmented_ipv4_unchanged'}
    transport, declared = ip[ihl:], total-ihl
    protocol = ip[9]
    if protocol == 6:
        if len(transport) < 20 or declared < 20:
            return {**facts,'status':'incomplete_tcp_header'}
        header_size = (transport[12] >> 4)*4
        facts['source_port'],facts['destination_port'] = struct.unpack('!HH',transport[:4])
        if header_size < 20 or header_size > len(transport) or header_size > declared:
            return {**facts,'status':'invalid_or_incomplete_tcp_header_length'}
    elif protocol == 17:
        header_size = 8
        if min(len(transport),declared) < header_size:
            return {**facts,'status':'incomplete_udp_header'}
        facts['source_port'],facts['destination_port'],udp_length = struct.unpack('!HHH',transport[:6])
        if not 8 <= udp_length <= declared:
            return {**facts,'status':'invalid_udp_length'}
    elif protocol == 1:
        if min(len(transport),declared) < 8:
            return {**facts,'status':'incomplete_icmp_header'}
        icmp_type = transport[0]
        header_size = {0:8,3:8,4:8,5:8,8:8,9:8,10:8,11:8,12:8,13:20,14:20,17:12,18:12}.get(icmp_type)
        if header_size is None:
            return {**facts,'status':'unsupported_icmp_type'}
        if min(len(transport),declared) < header_size:
            return {**facts,'status':'incomplete_icmp_message_header'}
        facts['icmp_type'],facts['icmp_code'] = icmp_type,transport[1]
    else:
        return {**facts,'status':'unsupported_ipv4_transport'}
    return {**facts,'eligible':True,'status':'eligible_complete_headers',
            'transport_header_bytes':header_size,'transport_header_complete':True,
            'ipv4_header_bytes':ihl}


def normalize_pcap(source, output_dir, expected_sha256=None):
    source, output_dir = Path(source).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True,exist_ok=True)
    source_hash = sha256(source)
    if expected_sha256 is not None and source_hash != expected_sha256:
        raise ValueError('Input SHA256 differs from the source receipt')
    receipt_path = output_dir/'normalization_receipt.json'
    if receipt_path.exists():
        saved = json.loads(receipt_path.read_text(encoding='utf-8'))
        if saved['source_sha256'] != source_hash or saved['normalizer_sha256'] != sha256(Path(__file__)):
            raise ValueError('Existing normalization belongs to different input/code; use a new output directory')
        for name in ('derived_path','change_journal','truncation_journal'):
            if sha256(Path(saved[name])) != saved[name+'_sha256']:
                raise ValueError('Cached normalization artifact changed: '+name)
        return saved
    capture = Capture(source)
    partial = output_dir/'declared-ip.pcap.partial'
    derived = output_dir/'declared-ip.pcap'
    if source in (partial,derived):
        raise ValueError('Original PCAP cannot be an output path')
    changes = output_dir/'record_changes.jsonl.gz'
    truncated = output_dir/'capture_truncation_records.jsonl.gz'
    counts, changed_by_protocol = Counter(), Counter()
    unchanged_hash = hashlib.sha256()
    with partial.open('wb') as target, gzip_lines(changes) as journal, gzip_lines(truncated) as trunc_journal:
        target.write(capture.header)
        for index,timestamp,header,frame,original in capture.records():
            counts['all_records'] += 1
            facts = inspect_frame(frame,capture.linktype)
            counts['status:'+facts['status']] += 1
            counts['original_caplen_less_than_orig_len'] += len(frame) < original
            new_original = original
            if original < len(frame):
                counts['invalid_record_orig_len_less_than_caplen'] += 1
            elif facts['eligible'] and original < facts['declared_frame_length']:
                new_original = facts['declared_frame_length']
                counts['orig_len_changed_records'] += 1
                changed_by_protocol[str(facts['ip_protocol'])] += 1
            missing = facts.get('declared_missing_ip_bytes',0)
            if missing:
                counts['ipv4_records_with_missing_captured_bytes'] += 1
                counts['ipv4_declared_missing_bytes_sum_not_unique_stream'] += missing
                if not facts['eligible']:
                    counts['truncated_records_not_eligible_for_normalization'] += 1
            audit = dict(source_record_index_1based=index,timestamp=timestamp,caplen=len(frame),
                         original_orig_len=original,declared_orig_len=new_original,
                         orig_len_changed=new_original!=original,**facts)
            if new_original != original:
                journal.write(json.dumps(audit,separators=(',',':'))+'\n')
            if missing:
                trunc_journal.write(json.dumps(audit,separators=(',',':'))+'\n')
            target.write(header[:12]+struct.pack(capture.endian+'I',new_original))
            target.write(frame)
            unchanged_hash.update(header[:12]);unchanged_hash.update(frame)
    # Independent verification of every original/derived record, not just the changed subset.
    other = Capture(partial)
    assert capture.header == other.header
    changed = 0
    for left,right in zip(capture.records(),other.records(),strict=True):
        assert left[:2] == right[:2] and left[2][:12] == right[2][:12] and left[3] == right[3]
        assert right[4] >= left[4]
        changed += right[4] != left[4]
    assert changed == counts['orig_len_changed_records']
    if changed:
        partial.replace(derived)
    else:
        assert sha256(partial) == source_hash
        partial.unlink()  # Only the disposable output created above; original is never removed.
        derived = source
    assert sha256(source) == source_hash
    for name in ('orig_len_changed_records','ipv4_records_with_missing_captured_bytes',
                 'ipv4_declared_missing_bytes_sum_not_unique_stream',
                 'truncated_records_not_eligible_for_normalization'):
        counts.setdefault(name,0)
    receipt = dict(contract=CONTRACT,source_path=str(source),source_sha256=source_hash,
                   source_bytes=source.stat().st_size,derived_path=str(derived),
                   derived_path_sha256=sha256(derived),change_journal=str(changes),
                   change_journal_sha256=sha256(changes),truncation_journal=str(truncated),
                   truncation_journal_sha256=sha256(truncated),counts=dict(counts),
                   changes_by_ip_protocol=dict(changed_by_protocol),source_unchanged=True,
                   only_record_orig_len_changed=True,timestamps_caplen_and_frame_bytes_sha256=unchanged_hash.hexdigest(),
                   normalizer_sha256=sha256(Path(__file__)),label_inputs_used=False,
                   packet_filters_used=False,payload_reconstructed=False,
                   length_semantics='IPv4-declared datagram bytes and capture records; not independently observed physical wire bytes/packets.',
                   unsupported_or_invalid_headers='Preserve original record unchanged and report reason; no zero-fill or inferred missing transport header.')
    write_json(receipt_path,receipt)
    return receipt


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pcap',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-sha256')
    args=parser.parse_args()
    receipt=normalize_pcap(args.pcap,args.output,args.expected_sha256)
    print(json.dumps({'contract':CONTRACT,'counts':receipt['counts'],'source_sha256':receipt['source_sha256'],'derived_sha256':receipt['derived_path_sha256']}),flush=True)
