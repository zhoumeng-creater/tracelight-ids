"""Bind the actual context export to unified identities without model scoring."""
import gzip
import hashlib
import json
from pathlib import Path
from build_unified import write_rows, write_json

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:
        yield from (json.loads(s) for s in f)

def main():
    here=Path(__file__).resolve().parent; base=here.parent
    context=base/'split_method/context_rows.jsonl.gz'
    audit_path=base/'split_method/context_audit.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8'))['audit']
    views={};count=0;violations=0
    for r in rows(context):
        count+=1
        violations+=int(r['max_history_available_at'] is not None and r['max_history_available_at']>=r['effective_available_at'])
        assert r['physical_flow_id'] not in views, 'This binding requires one monitored-host view per physical flow.'
        views[r['physical_flow_id']]=r
    assert count==45309 and violations==0
    assert audit['blocked_observations']==0 and audit['all_supplied_rows_accounted_for']
    expected={r['flow_id'] for r in rows(here/'observations.jsonl.gz') if r['group_id'] in
              {'optc:20190919:declared_normal_business_subset','optc:empire_custom:20190924_continues_20190925'}}
    assert len(expected)==42173 and expected<=set(views)
    linked=[]
    for r in rows(here/'event_flow_links.jsonl.gz'):
        if r['flow_id'] in expected:
            v=views[r['flow_id']]
            linked.append(dict(event_id=r['event_id'],physical_flow_id=r['flow_id'],context_view_id=v['flow_id'],
                               group_id=v['group'],time_mode=v['time_mode'],effective_available_at=v['effective_available_at'],
                               relation='same_network_observation_not_causal_predecessor_supervision'))
    assert len(linked)==46682
    out=here/'context_links.jsonl.gz';write_rows(out,linked)
    receipt=dict(schema='context-binding-v1',actual_export_rows=count,not_strictly_prior_history_rows=violations,
                 linked_observations=len(expected),event_context_links=len(linked),
                 input_sha256=sha(base/'optc/context_replay_inputs.jsonl.gz'),
                 original_input_sha256=sha(base/'optc/context_inputs.jsonl.gz'),output_sha256=sha(context),
                 module_sha256=sha(base/'split_method/prefix_context.py'),audit_sha256=sha(audit_path),
                 binding_sha256=sha(out),binding_script_sha256=sha(Path(__file__)),
                 model_calls=0,training_calls=0,time_mode='offline_flow_end_replay',
                 actual_online_availability_claimed=False,held_out_Meterpreter_context_generated=False)
    write_json(base/'split_method/context_execution_receipt.json',receipt)
    write_json(here/'context_binding_receipt.json',receipt)
    print(json.dumps(receipt,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
