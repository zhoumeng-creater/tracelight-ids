"""Fetch remaining complete normal-actor logs only, without adding target events."""
import json
from fetch_sources import ROOT,index,member,write_json

if __name__=='__main__':
    selection=[]
    for run in ('santos','fox'):
        for x in index(run+'_no-pcaps.zip'):
            parts=x['path'].split('/')
            if len(parts)==4 and parts[0]=='gather' and parts[2:]==['logs','sm.log'] and parts[1].startswith(('internal_employee_','remote_employee_','ext_user_')):
                selection.append({'run':run,**x})
    write_json(ROOT/'normal_actor_context_selection.json',{'target_events_unchanged':254,'members':selection,'compressed_bytes':sum(x['compressed'] for x in selection)})
    receipts=[]
    for x in selection:
        receipts.append(member(x['run']+'_no-pcaps.zip',x['path']))
    write_json(ROOT/'normal_actor_context_receipt.json',receipts)
    print(json.dumps({'normal_actor_logs_complete':len(receipts),'compressed_bytes':sum(x['compressed'] for x in receipts),'target_events_unchanged':254}))
