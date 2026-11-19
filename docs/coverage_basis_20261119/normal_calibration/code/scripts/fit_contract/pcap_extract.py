"""Full-capture fixed Zeek extraction with explicit declared-IP measurement semantics."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

from pcap_normalize import CONTRACT, gzip_lines, normalize_pcap, sha256, write_json

RAW8=('ip_protocol','orig_port','resp_port','duration_seconds','orig_pkts','resp_pkts','orig_ip_bytes','resp_ip_bytes')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def endpoint_key(protocol,source,source_port,destination,destination_port):
    if source_port is None or destination_port is None:
        return None
    return (int(protocol),*sorted(((source,int(source_port)),(destination,int(destination_port)))))


def assign_packet_provenance(rows,journal):
    groups=defaultdict(list)
    for i,row in enumerate(rows):
        key=endpoint_key(row['ip_protocol'],row['source'],row['orig_port'],row['destination'],row['resp_port'])
        if key is not None:
            groups[key].append((row['connection_start'],row['zeek_reported_end'] if row['zeek_reported_end'] is not None else row['connection_start'],i))
        row['capture_provenance']={'orig_len_adjusted_record_candidates':0,'capture_truncated_record_candidates':0,'declared_missing_ip_bytes_candidate_sum':0,'ambiguous_packet_record_matches':0,'association_basis':'IPv4 protocol + bidirectional endpoints + packet timestamp inside Zeek observed interval; all matches retained','missing_byte_sum_is_unique_TCP_gap':False}
    index={}
    for key,items in groups.items():
        items.sort();maximum=float('-inf');ends=[]
        for start,end,i in items:
            maximum=max(maximum,end);ends.append(maximum)
        index[key]=(items,[v[0] for v in items],ends)
    counts=Counter()
    with gzip.open(journal,'rt',encoding='utf-8') as stream:
        for line in stream:
            p=json.loads(line);counts['truncated_capture_records']+=1
            key=endpoint_key(p.get('ip_protocol',0),p.get('source'),p.get('source_port'),p.get('destination'),p.get('destination_port'))
            hits=[];timestamp=p['timestamp'];epsilon=0.00001
            if key in index:
                items,starts,prefix_ends=index[key];position=bisect_right(starts,timestamp+epsilon)-1
                while position>=0 and prefix_ends[position]>=timestamp-epsilon:
                    start,end,i=items[position]
                    if start-epsilon<=timestamp<=end+epsilon:
                        hits.append(i)
                    position-=1
            counts['unassigned_truncated_records']+=not hits
            counts['uniquely_assigned_truncated_records']+=len(hits)==1
            counts['multiply_assigned_truncated_records']+=len(hits)>1
            for i in hits:
                row=rows[i]['capture_provenance']
                row['orig_len_adjusted_record_candidates']+=int(p['orig_len_changed'])
                row['capture_truncated_record_candidates']+=1
                row['declared_missing_ip_bytes_candidate_sum']+=p['declared_missing_ip_bytes']
                row['ambiguous_packet_record_matches']+=len(hits)>1
    for row in rows:
        p=row['capture_provenance'];p['association_unambiguous_for_matched_records']=p['ambiguous_packet_record_matches']==0
        row['payload_completeness']='captured_payload_truncation_present' if p['capture_truncated_record_candidates'] else 'not_established_from_raw8'
    return dict(counts)


def extract(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    final=output/'extraction_receipt.json'
    if final.exists():
        old=load(final)
        if old['extractor_sha256']!=sha256(Path(__file__)) or old['source_sha256']!=args.expected_sha256:
            raise ValueError('Existing extraction uses different input/code; use a new output directory')
        for name,info in old['outputs'].items():
            if sha256(output/name)!=info['sha256']:
                raise ValueError('Existing extraction output changed: '+name)
        return old
    normalized=normalize_pcap(args.pcap,output/'normalization',args.expected_sha256)
    runtime=load(args.runtime_receipt)
    if runtime['version']!='8.0.11':
        raise ValueError('The contract requires Zeek 8.0.11')
    sys.path.insert(0,str(args.ait_code.resolve()/'scripts/ait_contract'))
    import analyze,zeek_runtime
    log_dir=output/'zeek'
    reused=False
    if args.reuse_zeek_export is not None and normalized['source_sha256']==normalized['derived_path_sha256']:
        prior=args.reuse_zeek_export.resolve();export=load(prior/'export-receipt.json')
        if export['pcap_sha256']!=normalized['derived_path_sha256'] or export['mode']!='ignore_checksums' or export['version']!='8.0.11':
            raise ValueError('Reusable export is not the exact source, version and checksum mode')
        if export['export_script_sha256']!=sha256(Path(runtime['export_script'])):
            raise ValueError('Reusable export uses a different Zeek export policy')
        log_dir.mkdir(parents=True,exist_ok=True)
        for name,info in export['logs'].items():
            if info['exists']:
                source=prior/(name+'.log')
                if sha256(source)!=info['sha256']:
                    raise ValueError('Reusable Zeek log hash mismatch: '+name)
                shutil.copy2(source,log_dir/source.name)
        shutil.copy2(prior/'export-receipt.json',log_dir/'export-receipt.json')
        reused=True
    else:
        export=zeek_runtime.export_pcap(Path(normalized['derived_path']),log_dir,'ignore_checksums',runtime)
    raw_records=analyze.records(log_dir/'conn.log')
    assert len({r['uid'] for r in raw_records})==len(raw_records),'Duplicate UID inside one conn.log'
    source_scope=f"ait:{args.scenario}:{args.capture_host}:{normalized['source_sha256']}"
    rows=[]
    for raw in raw_records:
        row=analyze.normalize_connection(raw,(0,float('inf')))
        features={name:row[name] for name in RAW8}
        missing=[name for name,value in features.items() if value is None or value<0]
        key={'capture_sha256':normalized['source_sha256'],'native_protocol':row['native_protocol'],'source':row['source'],'orig_port':row['orig_port'],'destination':row['destination'],'resp_port':row['resp_port'],'connection_start':row['connection_start']}
        key_hash=hashlib.sha256(json.dumps(key,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        row.update(flow_id=f"ait:{args.scenario}:{args.capture_host}:{row['uid']}",source_scope=source_scope,
                   scenario=args.scenario,capture_host=args.capture_host,flow_key=key,flow_key_id=key_hash,
                   stable_flow_id=source_scope+':flowkey:'+key_hash,raw8=features,raw8_missing_fields=missing,
                   raw8_complete=not missing,source_address=row['source'],destination_address=row['destination'],
                   observed_end=row['zeek_reported_end'],actual_detector_received_time=None,
                   capture_sha256_original=normalized['source_sha256'],capture_sha256_declared=normalized['derived_path_sha256'],
                   measurement_contract=CONTRACT,raw8_byte_semantics='IPv4-declared datagram bytes',
                   raw8_packet_semantics='capture records, potentially aggregated by offload',
                   label=None,zeek_record=raw,orig_bytes=raw.get('orig_bytes'),resp_bytes=raw.get('resp_bytes'))
        rows.append(row)
    duplicate=Counter(row['flow_key_id'] for row in rows)
    for row in rows:
        row['flow_key_cardinality']=duplicate[row['flow_key_id']]
        row['flow_key_unique']=duplicate[row['flow_key_id']]==1
    association=assign_packet_provenance(rows,Path(normalized['truncation_journal']))
    rows.sort(key=lambda row:(row['connection_start'],row['source'],row['destination'],row['uid']))
    with gzip_lines(output/'connections.jsonl.gz') as stream, gzip_lines(output/'uid_flowkeys.jsonl.gz') as mapping:
        for row in rows:
            stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
            mapping.write(json.dumps({key:row[key] for key in ('flow_id','uid','stable_flow_id','flow_key_id','flow_key','flow_key_cardinality')},separators=(',',':'))+'\n')
    write_json(output/'schema_example.json',rows[0] if rows else {})
    summary={'connections':len(rows),'protocols':dict(Counter(row['native_protocol'] for row in rows)),
             'rows_with_missing_raw8':sum(not row['raw8_complete'] for row in rows),
             'rows_with_capture_truncation_candidates':sum(row['capture_provenance']['capture_truncated_record_candidates']>0 for row in rows),
             'nonunique_flow_keys':sum(n>1 for n in duplicate.values()),'packet_provenance_association':association,
             'normalization_counts':normalized['counts'],'model_calls':0,'labels_assigned':0,
             'new_zeek_exports':0 if reused else 1,'reused_exact_full_capture_export':reused,
             'scope':'All conn.log records from the full capture, with no label/action/endpoint/time filtering.'}
    write_json(output/'summary.json',summary)
    names=('connections.jsonl.gz','uid_flowkeys.jsonl.gz','schema_example.json','summary.json','normalization/normalization_receipt.json','zeek/export-receipt.json')
    receipt={'contract':CONTRACT,'source_scope':source_scope,'source_sha256':normalized['source_sha256'],
             'declared_capture_sha256':normalized['derived_path_sha256'],'source_path':str(args.pcap.resolve()),
             'scenario':args.scenario,'capture_host':args.capture_host,'runtime_receipt_path':str(args.runtime_receipt.resolve()),
             'runtime_receipt_sha256':sha256(args.runtime_receipt),'zeek_version':'8.0.11','zeek_mode':'ignore_checksums',
             'zeek_log_directory':str(log_dir),'reused_export_source':str(args.reuse_zeek_export) if reused else None,
             'normalizer_sha256':sha256(Path(__file__).with_name('pcap_normalize.py')),'extractor_sha256':sha256(Path(__file__)),
             'author_analyzer_sha256':sha256(args.ait_code/'scripts/ait_contract/analyze.py'),
             'summary':summary,'outputs':{name:{'bytes':(output/name).stat().st_size,'sha256':sha256(output/name)} for name in names},
             'training_admission_decided_here':False,'missing_payload_reconstructed':False}
    write_json(final,receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pcap',type=Path,required=True)
    parser.add_argument('--expected-sha256',required=True)
    parser.add_argument('--scenario',required=True)
    parser.add_argument('--capture-host',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runtime-receipt',type=Path,required=True)
    parser.add_argument('--ait-code',type=Path,required=True)
    parser.add_argument('--reuse-zeek-export',type=Path)
    args=parser.parse_args()
    result=extract(args)
    print(json.dumps({'source_scope':result['source_scope'],**result['summary']},ensure_ascii=False),flush=True)
