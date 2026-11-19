"""Bounded first-1MiB ZIP probes. No attack-period completeness claim."""
import collections, datetime as dt, hashlib, json, struct, zlib
from concurrent.futures import ThreadPoolExecutor
from fetch_sources import ROOT, index, range_bytes, write_json

CHOICES={
 'fox':{'intranet_server':'1642159255','inet-firewall':'1642508536'},
 'santos':{'intranet_server':'1642084650','inet-firewall':'1642339189'},
}
def one(job):
    run,host,suffix=job;archive=run+'.zip'
    name=f'gather/{host}/logs/suricata/log.pcap.{suffix}'
    entry=next(x for x in index(archive) if x['path']==name)
    cache=ROOT/'capture_probes'/run/host
    h=range_bytes(archive,entry['offset'],entry['offset']+255,cache/'zip_header.bin')
    v=struct.unpack('<IHHHHHIIIHH',h[:30]);assert v[0]==0x04034b50 and v[3]==8
    assert h[30:30+v[-2]].decode()==name
    offset=entry['offset']+30+v[-2]+v[-1]
    packed=range_bytes(archive,offset,offset+1024*1024-1,cache/'first_1MiB.deflate')
    decoder=zlib.decompressobj(-15);b=decoder.decompress(packed)
    assert b[:4] in (b'\xd4\xc3\xb2\xa1',b'\xa1\xb2\xc3\xd4')
    endian='<' if b[:4]==b'\xd4\xc3\xb2\xa1' else '>'
    _,major,minor,zone,sig,snaplen,linktype=struct.unpack(endian+'IHHIIII',b[:24]);assert linktype==1
    counts=collections.Counter();examples=[];times=[];at=24
    while at+16<=len(b):
        sec,usec,caplen,origlen=struct.unpack(endian+'4I',b[at:at+16])
        if at+16+caplen>len(b):break
        raw=b[at+16:at+16+caplen];at+=16+caplen;counts['complete_pcap_records']+=1;times.append(sec+usec/1e6)
        if caplen<origlen:counts['caplen_below_origlen']+=1
        if len(raw)<14:counts['short_ethernet']+=1;continue
        etype=struct.unpack('!H',raw[12:14])[0];ipoffset=14
        while etype in (0x8100,0x88a8) and len(raw)>=ipoffset+4:
            etype=struct.unpack('!H',raw[ipoffset+2:ipoffset+4])[0];ipoffset+=4
        if etype!=0x0800:continue
        counts['ipv4_records']+=1
        if len(raw)<ipoffset+20:counts['short_ipv4_header']+=1;continue
        declared=struct.unpack('!H',raw[ipoffset+2:ipoffset+4])[0];available=len(raw)-ipoffset
        if available<declared:
            counts['ipv4_declared_exceeds_stored']+=1
            if len(examples)<5:examples.append({'packet_index':counts['complete_pcap_records'],'timestamp':sec+usec/1e6,'caplen':caplen,'origlen':origlen,'ipv4_declared':declared,'ipv4_stored':available,'protocol':raw[ipoffset+9]})
    result={'run':run,'host':host,'archive':archive,'member':name,'member_compressed_bytes':entry['compressed'],'probe_compressed_bytes':len(packed),'decoded_prefix_bytes':len(b),'probe_sha256':hashlib.sha256(packed).hexdigest(),'whole_member_crc_verified':False,'decoder_end_of_member':decoder.eof,'pcap_snaplen':snaplen,'pcap_linktype':linktype,'counts':dict(counts),'examples':examples,'first_record_time':dt.datetime.fromtimestamp(min(times),dt.timezone.utc).isoformat(),'last_complete_record_time':dt.datetime.fromtimestamp(max(times),dt.timezone.utc).isoformat(),'scope':'First compressed MiB only; last incomplete record excluded. Does not establish attack-window completeness or actual packet loss cause.'}
    write_json(cache/'probe_receipt.json',result)
    return result

if __name__=='__main__':
    jobs=[(run,host,suffix) for run,hosts in CHOICES.items() for host,suffix in hosts.items()]
    with ThreadPoolExecutor(max_workers=4) as p:results=list(p.map(one,jobs))
    write_json(ROOT/'capture_probe_summary.json',results)
    print(json.dumps(results,indent=2))
