"""Check shared-proxy attribution against all original normal-actor logs.

Keeps the fixed 254 event targets; alternative actors are evidence, not new targets.
"""
import collections,json,math
from fetch_sources import ROOT,index,write_json

def main():
    params=json.loads((ROOT/'network_mapping_parameters.json').read_text(encoding='utf-8'))['events_parameters']
    receipt_path=ROOT/'normal_actor_context_receipt.json'
    receipts=json.loads(receipt_path.read_text(encoding='utf-8')) if receipt_path.exists() else []
    result={};summaries=[]
    for run in ('santos','fox'):
        expected=[x['path'] for x in index(run+'_no-pcaps.zip') if len(x['path'].split('/'))==4 and x['path'].split('/')[0]=='gather' and x['path'].split('/')[2:]==['logs','sm.log'] and x['path'].split('/')[1].startswith(('internal_employee_','remote_employee_','ext_user_'))]
        missing=[];downloads=[];rows_read=0
        for member in expected:
            p=ROOT/'source'/run/member
            if not p.exists():missing.append(member);continue
            transition_starts={}
            for n,s in enumerate(p.open(encoding='utf-8'),1):
                x=json.loads(s);rows_read+=1;tid=x.get('transition_id')
                if x.get('message')=='Starting menu action':transition_starts[tid]=(n,x)
                elif x.get('message')=='Downloaded file' and tid in transition_starts:
                    bn,b=transition_starts[tid]
                    downloads.append({'actor_role':member.split('/')[1],'member':member,'start_line':bn,'complete_line':n,'start':b['timestamp'],'completion':x['timestamp'],'transition_id':tid,'file':x['file'],'actor_category':'author_normal_actor'})
        for event in params:
            if event['run']!=run or event['kind']!='normal_owncloud_file_download':continue
            obj=event['file_object'];ts=event['service_request_time_interval']['start_inclusive']
            matches=[x for x in downloads if x['file']['path']==obj['path'] and x['file']['name']==obj['name'] and x['file']['file_type']==obj['file_type'] and math.floor(x['start'])<=ts<=x['completion']]
            result[event['episode_id']]={'all_normal_actor_logs_read':not missing,'expected_members':len(expected),'missing_members':missing,'actor_candidates':matches,'actor_attribution_cardinality':'none' if not matches else 'unique_among_all_normal_actors' if len(matches)==1 and not missing else 'multiple_or_incomplete','all_candidates_author_normal':bool(matches) and all(x['actor_category']=='author_normal_actor' for x in matches),'known_named_actor_retained':any(x['actor_role']==event['actor_endpoint']['role'] and x['transition_id'] in event['episode_id'] for x in matches),'scope':'All author normal-actor logs and exact object/time candidates; proxy endpoint itself does not identify actor. Does not add target events or independently identify a TCP transaction.'}
        summaries.append({'run':run,'expected_actor_logs':len(expected),'missing_actor_logs':len(missing),'rows_read':rows_read,'download_actions_in_all_logs':len(downloads),'fixed_cloud_events':sum(x['run']==run and x['kind']=='normal_owncloud_file_download' for x in params)})
    write_json(ROOT/'all_actor_attribution.json',{'summary':summaries,'fixed_target_event_count':254,'source_receipts':receipts,'events':result})
    print(json.dumps({'summary':summaries,'event_actor_cardinality':dict(collections.Counter(x['actor_attribution_cardinality'] for x in result.values()))},indent=2))

if __name__=='__main__':main()
