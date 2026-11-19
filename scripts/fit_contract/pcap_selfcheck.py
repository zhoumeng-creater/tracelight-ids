"""Small normalization guard check; no Zeek, source data, labels or model use."""
import argparse
import json
from pathlib import Path
import struct

from pcap_normalize import Capture, folded_sum, inspect_frame, normalize_pcap, sha256, write_json


def frame(protocol,transport,declared_payload,fragment=0,bad_checksum=False):
    ip=bytearray(struct.pack('!BBHHHBBH4s4s',0x45,0,20+len(transport)+declared_payload,1,fragment,64,protocol,0,b'\xc0\x00\x02\x01',b'\xc0\x00\x02\x02'))
    struct.pack_into('!H',ip,10,(~folded_sum(bytes(ip)))&65535)
    if bad_checksum:ip[10]^=1
    return bytes(12)+bytes.fromhex('0800')+ip+transport+b'abcd'


def main(output):
    output.mkdir(parents=True,exist_ok=True)
    tcp=struct.pack('!HHIIBBHHH',12345,443,1,1,0x50,0x18,65535,0,0)
    option_header=bytearray(tcp);option_header[12]=0xf0
    udp=struct.pack('!HHHH',12345,53,108,0)
    icmp=struct.pack('!BBHHH',8,0,0,1,1)
    cases=[('valid_tcp_truncated',frame(6,tcp,100),True),
           ('valid_udp_truncated',frame(17,udp,100),True),
           ('valid_icmp_echo_truncated',frame(1,icmp,100),True),
           ('complete_tcp',frame(6,tcp,4),False),
           ('bad_ipv4_checksum',frame(6,tcp,100,bad_checksum=True),False),
           ('incomplete_tcp_options',frame(6,option_header,100),False),
           ('fragmented',frame(6,tcp,100,fragment=0x2000),False)]
    original=output/'synthetic.pcap'
    with original.open('wb') as f:
        f.write(struct.pack('<IHHIIII',0xa1b2c3d4,2,4,0,0,262144,1))
        for i,(name,data,change) in enumerate(cases):
            f.write(struct.pack('<IIII',1700000000+i,0,len(data),len(data)));f.write(data)
    receipt=normalize_pcap(original,output/'normalization',sha256(original))
    original_rows=list(Capture(original).records());derived_rows=list(Capture(receipt['derived_path']).records())
    results=[]
    for (name,data,expected),before,after in zip(cases,original_rows,derived_rows,strict=True):
        changed=before[4]!=after[4]
        assert changed==expected,(name,changed,expected)
        assert before[2][:12]==after[2][:12] and before[3]==after[3]
        results.append({'case':name,'changed':changed,'status':inspect_frame(data,1)['status'],'pass':True})
    result={'cases':results,'all_passed':True,'changed_records':receipt['counts']['orig_len_changed_records'],'source_bytes_unchanged':receipt['source_unchanged'],'models_called':0,'zeek_called':0}
    write_json(output/'selfcheck.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
