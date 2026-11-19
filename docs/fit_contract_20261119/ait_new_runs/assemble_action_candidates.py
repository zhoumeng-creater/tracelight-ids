"""Build run-level action evidence without inventing per-flow labels."""
import base64, collections, datetime as dt, hashlib, json, math, re
from urllib.parse import urlsplit, unquote, parse_qs
import yaml
from fetch_sources import ROOT, index, write_json

UTC=dt.timezone.utc
def iso(t):return dt.datetime.fromtimestamp(t,UTC).isoformat()
def logs(path):
    with path.open(encoding='utf-8') as f:
        for n,s in enumerate(f,1):yield n,json.loads(s)
def ref(path,n,x):
    return {'member':path.as_posix(),'line':n,'timestamp':x['timestamp'],'utc':iso(x['timestamp']),'message':x['message']}
def apache(base,host,start,end):
    rows=[]; stats=collections.Counter()
    for p in sorted((base/'gather'/host/'logs/apache2').glob('*access.log*')):
        lp=base/'labels'/host/'logs/apache2'/p.name
        labels={n['line']:n['labels'] for _,n in logs(lp)} if lp.exists() else {}
        with p.open(encoding='utf-8') as f:
            for n,s in enumerate(f,1):
                m=re.match(r'(?:(\S+:\d+) )?(\S+) - \S+ \[(.*?)\] "(.*?)" (\d+) (\S+) "(.*?)" "(.*?)"',s)
                if not m:stats['unparsed']+=1;continue
                vhost,client,ts,req,status,size,referer,agent=m.groups()
                if ts.endswith('+0000'):
                    month=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'].index(ts[3:6])+1
                    t=dt.datetime(int(ts[7:11]),month,int(ts[:2]),int(ts[12:14]),int(ts[15:17]),int(ts[18:20]),tzinfo=UTC).timestamp()
                else:t=dt.datetime.strptime(ts,'%d/%b/%Y:%H:%M:%S %z').timestamp()
                if not start<=t<end:stats['outside_simulation']+=1;continue
                rows.append({'member':p.relative_to(base).as_posix(),'line':n,'timestamp':t,'utc':iso(t),'client':client,'request':req,'status':int(status),'logged_bytes':None if size=='-' else int(size),'referer':referer,'agent':agent,'labels':labels.get(n,[]),'vhost':vhost})
    stats['in_interval']=len(rows)
    return rows,dict(stats)

def run_one(run):
    base=ROOT/'source'/run
    meta=yaml.safe_load((base/'dataset.yaml').read_text())
    start=dt.datetime.fromisoformat(meta['start']).replace(tzinfo=UTC).timestamp();end=dt.datetime.fromisoformat(meta['end']).replace(tzinfo=UTC).timestamp()
    servers=yaml.safe_load((base/'processing/config/servers.yaml').read_text())
    ap,stats=apache(base,'intranet_server',start,end);cloud,cstats=apache(base,'cloud_share',start,end)
    actor=__import__('pathlib').Path('gather/attacker_0/logs/ait.aecid.attacker.wpdiscuz/sm.log')
    ar=[(n,x) for n,x in logs(base/actor) if start<=x['timestamp']<end]
    chain=f'ait:{run}:wpdiscuz_complete_chain'
    pending={};control=[];downloads=[];reverse=[]
    ap_by_path=collections.defaultdict(list)
    for row in ap:
        parts=row['request'].split()
        if len(parts)>=2:ap_by_path[parts[1].split('?')[0]].append(row)
    for n,x in ar:
        msg=x['message'];key=(x.get('transition_id'),json.dumps(x.get('cmd'),sort_keys=True))
        if msg=='Sending web shell command':pending[key]=(n,x)
        if msg=='Web shell command response':
            begin,b=pending.pop(key)
            path=urlsplit(b['web_shell']).path
            matches=[r for r in ap_by_path[path] if math.floor(b['timestamp'])<=r['timestamp']<=x['timestamp'] and r['status']==200 and all(parse_qs(urlsplit(r['request'].split()[1]).query).get(k)==[str(v)] for k,v in b.get('params',{}).items())]
            output=json.dumps(x.get('output'),ensure_ascii=False)
            command=b.get('cmd')
            if command is None:command=json.loads(base64.b64decode(b['params']['wp_meta']))
            control.append({'episode_id':f'{chain}:webshell:{begin}','chain_id':chain,'kind':'webshell_command','intent':ref(actor,begin,b),'response':ref(actor,n,x),'command':command,'command_basis':'actor cmd field' if 'cmd' in b else 'decoded actor wp_meta request parameter','url':b['web_shell'],'response_output_sha256':hashlib.sha256(output.encode()).hexdigest(),'response_output_characters':len(output),'apache_matches':matches,'network_match_status':'pending_pcap','label_level':'action_response_and_service_log'})
    for name,beginmsg,donemsg,urlkey in [('wphashcrack','Downloading WPHashCrack','Downloaded WPHashCrack','url'),('password_list','Downloading password list','Downloaded password list','wl_url')]:
        begins=[(n,x) for n,x in ar if x['message']==beginmsg];dones=[(n,x) for n,x in ar if x['message']==donemsg]
        for n,x in begins:
            matches=[(dn,d) for dn,d in dones if d['transition_id']==x['transition_id'] and d['timestamp']>=x['timestamp']]
            assert len(matches)==1
            dn,d=matches[0]
            downloads.append({'episode_id':f'{chain}:download:{name}','chain_id':chain,'kind':'attack_download','object':name,'url':x[urlkey],'intent':ref(actor,n,x),'completion':ref(actor,dn,d),'network_match_status':'pending_pcap','label_level':'actor_completion_pending_network_confirmation'})
    for n,x in ar:
        if x['message']=='Reverse shell connected':reverse.append({'episode_id':f'{chain}:reverse_shell','chain_id':chain,'kind':'reverse_shell_channel','connection_receipt':ref(actor,n,x),'listen_socket':x['listen_socket'],'remote_socket':x['remote_socket'],'network_match_status':'pending_pcap'})
    normal=[]
    for role in ('internal_employee_0','internal_employee_1'):
        member=__import__('pathlib').Path('gather')/role/'logs/sm.log'
        rs=[(n,x) for n,x in logs(base/member) if start<=x['timestamp']<end]
        transitions=collections.defaultdict(list)
        for n,x in rs:transitions[x.get('transition_id')].append((n,x))
        for n,x in rs:
            msg=x['message'];tr=transitions[x.get('transition_id')]
            if msg=='Opened wordpress post':
                begins=[(bn,b) for bn,b in tr if b['message']=='Opening wordpress post' and b['timestamp']<=x['timestamp']]
                if len(begins)!=1:continue
                bn,b=begins[0];u=urlsplit(x['wordpress_post']['link']);target=u.path+('?' + u.query if u.query else '')
                matches=[r for r in ap_by_path[u.path] if math.floor(b['timestamp'])<=r['timestamp']<=x['timestamp'] and r['request'].split()[1]==target and r['client']==servers[role]['default_ipv4_address'] and r['status']==200 and not r['labels']]
                normal.append({'episode_id':f'ait:{run}:normal:{role}:{x["transition_id"]}','run_id':run,'kind':'normal_intranet_page_retrieval','host_role':role,'host_ip':servers[role]['default_ipv4_address'],'intent':ref(member,bn,b),'completion':ref(member,n,x),'url':x['wordpress_post']['link'],'apache_matches':matches,'network_match_status':'pending_intranet_pcap','scope':'HTML page retrieval; not executable download'})
            elif msg=='Downloaded file':
                begins=[(bn,b) for bn,b in tr if b['message']=='Starting menu action' and b['timestamp']<=x['timestamp']]
                if len(begins)!=1:continue
                bn,b=begins[0];obj=x['file'];filename=obj['name'];filepath=obj['path'].rstrip('/')+'/'+filename
                matches=[]
                for r in cloud:
                    if not(math.floor(b['timestamp'])<=r['timestamp']<=x['timestamp'] and r['status']==200 and not r['labels']):continue
                    target=r['request'].split()[1];u=urlsplit(target);qs=parse_qs(u.query)
                    if (obj['file_type']=='file' and unquote(u.path)=='/remote.php/webdav'+filepath) or (obj['file_type']=='dir' and u.path=='/index.php/apps/files/ajax/download.php' and qs.get('files')==[filename] and qs.get('dir')==[obj['path']]):matches.append(r)
                normal.append({'episode_id':f'ait:{run}:normal:{role}:{x["transition_id"]}','run_id':run,'kind':'normal_owncloud_file_download','host_role':role,'host_ip':servers[role]['default_ipv4_address'],'intent':ref(member,bn,b),'completion':ref(member,n,x),'file':obj,'apache_matches':matches,'network_match_status':'cloud_pcap_not_selected','association':'action time + exact file path/name; observed NAT client preserved, not substituted with actor IP'})
    pcs=[x for x in index(f'{run}.zip') if x['path'].startswith('gather/intranet_server/logs/suricata/log.pcap.')]
    assert len(pcs)==1
    summary={'run':run,'chain_id':chain,'simulation_start':iso(start),'simulation_end_exclusive':iso(end),'timezone_basis':'servers.yaml timezone UTC; actor epoch timestamps; Apache explicit +0000','webshell_actions':len(control),'webshell_with_service_match':sum(bool(x['apache_matches']) for x in control),'download_completion_actions':len(downloads),'reverse_shell_connections':len(reverse),'normal_counts':dict(collections.Counter(x['kind'] for x in normal)),'normal_service_matched':dict(collections.Counter(x['kind'] for x in normal if x['apache_matches'])),'intranet_apache':stats,'cloud_apache':cstats,'intranet_pcap_candidate':pcs[0],'pcap_selection_status':'not_downloaded; capture-integrity issue under bounded source review','candidate_reason':'Only intranet member in full archive; victim control and outbound download plus normal HTTPS page retrieval. ownCloud business-file actions require cloud PCAP and are not inferred to occur in intranet capture.','fit_role':'candidate_unassigned','independence':'whole run/attack chain; same AIT attack implementation; actions are not independent attack chains'}
    write_json(ROOT/f'{run}_action_candidates.json',{'summary':summary,'control_actions':control,'download_actions':downloads,'reverse_shell':reverse,'normal_actions':normal})
    return summary

if __name__=='__main__':
    results=[run_one(run) for run in ('fox','santos')]
    write_json(ROOT/'action_candidate_summary.json',results)
    print(json.dumps(results,ensure_ascii=False,indent=2))
