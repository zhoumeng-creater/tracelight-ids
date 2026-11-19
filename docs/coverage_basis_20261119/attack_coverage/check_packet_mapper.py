"""Check new packet candidate mapping against already-verified historical slices."""
import json
from pathlib import Path
import assemble_full_capture as m
def main():
    authors,_,_=m.flow_csv(m.SRC/'flow_20210621_net1013x.csv')
    results=[]
    for scope,expected in [('normal_mail',16),('apt_c2',5)]:
        m.OUT=m.ROOT/'mapping_check'/scope;m.OUT.mkdir(parents=True,exist_ok=True)
        r=json.loads((m.SRC/'pcap_slices'/scope/'receipt.json').read_text(encoding='utf-8'))
        m.dump(m.OUT/'pcap_quality.json',{'max_packet_time':r['max_packet_time']})
        subset=[a for a in authors if int(a['bidirectional_first_seen_ms'])/1000<=r['max_packet_time'] and int(a['bidirectional_last_seen_ms'])/1000>=r['min_packet_time']]
        cs=m.readgz(m.ROOT/'network_mapping'/scope/'all_connections.jsonl.gz')
        payload,links=m.packet_mapping(subset,cs,Path(r['output_path']))
        old=[json.loads(x) for x in (m.ROOT/'network_mapping/connections.jsonl').read_text(encoding='utf-8').splitlines()]
        accepted={x['flow_id'] for x in old if x['source_scope']==scope and x['label'] is not None}
        found=[]
        for zi,c in enumerate(cs):
            if c['flow_id'] not in accepted:continue
            s=payload['zeek_stats'][zi];als={x['author_index'] for x in links if x['zeek_index']==zi};labels={m.truth(subset[a])[0] for a in als}
            assert len(labels)==1 and None not in labels,(scope,c['flow_id'],labels)
            assert s['uncovered_packets']==0
            assert all(c['raw8'][k]==s[k] for k in ('orig_pkts','resp_pkts','orig_ip_bytes','resp_ip_bytes'))
            found.append(c['flow_id'])
        assert len(found)==expected,(scope,len(found),expected)
        results.append({'scope':scope,'previously_verified_connections_recovered':len(found),'packet_identity_ledger':str(m.OUT/'packet_membership.jsonl.gz'),'source_annotations_subset':len(subset),'all_connections':len(cs),'links':len(links)})
    m.dump(m.ROOT/'mapping_check/validation.json',{'status':'passed','no_model_calls':True,'checks':results});print(json.dumps(results))
if __name__=='__main__':main()
