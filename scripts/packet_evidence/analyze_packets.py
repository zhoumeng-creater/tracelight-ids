"""Describe already-open real packet geometry; never fit, score, or calibrate."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from byte_geometry import analyze


def rows(path):
    with gzip.open(path, 'rt', encoding='utf8') as f:
        for line in f:
            yield json.loads(line)


def q(values):
    values = [v for v in values if v is not None]
    return dict(zip(('n','min','q25','median','q75','max'),
                    [len(values), *np.quantile(values,[0,.25,.5,.75,1]).tolist()])) if values else {'n':0}


def summarize(rr):
    metrics = ('packet_count','payload_packets','pure_ack_packets','duplicate_fraction',
               'first_observed_piece_count','ack_segment_count','ack_query_count',
               'direction_burst_count','ack_covered_pair_fraction','first20_unique_byte_fraction')
    return {'connections': len(rr), 'geometry_eligible': sum(r['geometry_eligible'] for r in rr),
            'with_ambiguous_membership':sum(r['ambiguous_packet_candidates']>0 for r in rr),
            'with_capture_truncation':sum(r['truncated_payload_packets']>0 for r in rr),
            'with_duplicate_bytes':sum(r['duplicate_payload_bytes']>0 for r in rr),
            'with_ip_over_1500':sum(r['large_ip_packets']>0 for r in rr),
            'ack_map_exactly_reconstructed_from_direction_bursts':sum(r['ack_map_exactly_reconstructed_from_direction_bursts'] for r in rr),
            'metrics':{k:q([r[k] for r in rr]) for k in metrics}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--headers',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    batch = repo/'docs/coverage_basis_20261119/batch/rows.jsonl.gz'
    meta = {r['flow_id']:r for r in rows(batch)
            if r['dataset_role'] not in ('C','C_background') and r['source_family'] in ('AIT','Unraveled')}
    results, source_reports, invariant_checks = [], [], []
    for path in sorted(args.headers.glob('*.connections.jsonl.gz')):
        name = path.name.removesuffix('.connections.jsonl.gz')
        assert not any(x in name.lower() for x in ('russell','harrison','meterpreter'))
        connections = {r['connection_index']:r for r in rows(path)
                       if r['flow_id'] in meta and r['raw8']['ip_protocol']==6}
        packets, ambiguous = defaultdict(list), Counter()
        for p in rows(args.headers/(name+'.jsonl.gz')):
            matches = p['connection_indices']
            if len(matches) != 1:
                for i in matches:
                    if i in connections:
                        ambiguous[i] += 1
                continue
            i = matches[0]
            if i not in connections:
                continue
            r = connections[i]
            d = 0 if (p['source'],p['source_port']) == (r['source'],r['raw8']['orig_port']) else 1
            packets[i].append((p['record_index'],p['timestamp_ns'],d,p['seq'],p['ack'],p['flags'],
                               p['declared_tcp_payload_bytes'],p['captured_tcp_payload_bytes'],p['ip_total_bytes']))
        before = len(results)
        for i,c in connections.items():
            m = meta[c['flow_id']]
            r = analyze(packets.get(i,[]))
            r.update(source=name,flow_id=c['flow_id'],stable_flow_id=c['stable_flow_id'],
                     label=m['target_label'],target_kind=m['target_kind'],role=m['dataset_role'],
                     raw8=c['raw8'],connection_start=c['connection_start'],
                     observed_end=c['zeek_reported_end'],replay_export_time=c['replay_export_time'],
                     ambiguous_packet_candidates=ambiguous[i])
            r['geometry_eligible'] &= ambiguous[i]==0 and not r['serial_span_ambiguous']
            results.append(r)
        # Representative real inputs, chosen only by source/label and ID or
        # maximum packet count. These are representation tests, not new samples.
        chosen = set()
        for kind in ('control','download','normal','unknown'):
            candidates = [i for i,c in connections.items() if meta[c['flow_id']]['target_kind']==kind and packets.get(i)]
            if candidates:
                chosen.add(min(candidates,key=lambda i:connections[i]['flow_id']))
                chosen.add(max(candidates,key=lambda i:len(packets[i])))
        for i in sorted(chosen):
            original=packets[i]
            refined=[]
            for p in original:
                rec,time,d,seq,ack,flags,length,cap,iplen=p
                if length>1:
                    left=length//2
                    refined.extend([(rec*2,time,d,seq,ack,flags&~1,left,min(cap,left),iplen),
                                    (rec*2+1,time,d,(seq+bool(flags&2)+left)%2**32,ack,flags&~2,
                                     length-left,max(0,cap-left),iplen)])
                else:
                    refined.append((rec*2,*p[1:]))
            repeated=[p for original_p in original for p in (original_p,original_p)]
            base=analyze(original)
            a,b=analyze(refined),analyze(repeated)
            fields=('unique_payload_bytes','segments','ack_covered_pair_fraction')
            check={'flow_id':connections[i]['flow_id'],'source':name,'packets':len(original),
                   'identical_ack_refinement_preserves':all(base[k]==a[k] for k in fields),
                   'duplicate_replay_preserves':all(base[k]==b[k] for k in fields)}
            assert check['identical_ack_refinement_preserves'] and check['duplicate_replay_preserves'],check
            invariant_checks.append(check)
        source_reports.append({'source':name,**summarize(results[before:])})
        print(json.dumps({'source':name,'connections':len(connections),'mapped_packets':sum(map(len,packets.values()))}),flush=True)
    groups = {}
    for name in sorted({r['source'] for r in results}):
        rr = [r for r in results if r['source']==name]
        for kind in ('control','download','normal','unknown','other_attack'):
            chosen = [r for r in rr if r['target_kind']==kind]
            groups[name+'/'+kind] = summarize(chosen)
        controls = [r for r in rr if r['target_kind']=='control' and r['raw8']['resp_port']==443]
        normal443 = [r for r in rr if r['target_kind']=='normal' and r['raw8']['resp_port']==443]
        groups[name+'/control443'] = summarize(controls)
        groups[name+'/normal443'] = summarize(normal443)
        if controls:
            lo,hi = min(r['connection_start'] for r in controls),max(r['observed_end'] for r in controls)
            concurrent = [r for r in normal443 if lo<=r['connection_start']<=hi]
            groups[name+'/normal443_during_control_start_end'] = {'interval':[lo,hi],**summarize(concurrent)}
    compact = []
    with gzip.open(args.output/'connection_geometry.jsonl.gz','wt',encoding='utf8') as f:
        for r in results:
            f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')
            if r['target_kind'] in ('control','download'):
                compact.append({k:v for k,v in r.items() if k not in ('segments','bursts')})
    report = {'protocol':'docs/packet_interaction_20261119/protocol.md',
              'connections':len(results),'sources':source_reports,'groups':groups,
              'positive_connections':compact,'model_calls':0,'thresholds_selected':0,
              'associations':'Only unique lifecycle/ledger packet membership admitted; ambiguous candidates counted per connection.',
              'coordinate':'First observed declared TCP byte union, with captured truncation recorded; complete-connection ACK coverage.',
              'batch_sha256':hashlib.sha256(batch.read_bytes()).hexdigest()}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    (args.output/'real_input_invariant_checks.json').write_text(json.dumps(invariant_checks,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
