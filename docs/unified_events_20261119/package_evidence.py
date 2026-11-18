"""Copy compact evidence into the isolated research worktree, preserving bytes."""
from pathlib import Path
import hashlib
import json
import shutil
import argparse

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repository',type=Path,required=True)
    args=ap.parse_args()
    src=Path(__file__).resolve().parent
    repo=args.repository.resolve()
    dest=repo/'docs/unified_events_20261119'
    files={src/'README.md',src/'package_evidence.py'}
    for section in ['ait','ait_attack','optc','split_method','unified']:
        for f in (src/section).iterdir():
            if f.is_file() and f.suffix in {'.py','.json','.md','.gz'}: files.add(f)
    for name in ['referenced_connections.jsonl','event_flow_links.csv','cloud_packet_connections.csv']:
        files.add(src/'ait'/name)
    for section,receipt in [('ait','cloud-source-receipt.json'),('ait_attack','intranet-source-receipt.json')]:
        files.add(src/section/'source'/receipt)
        for f in (src/section/'source/capture_config').rglob('*'):
            if f.is_file() and f.stat().st_size<1_000_000:files.add(f)
    for f in (src/'ait_attack/contract').glob('*.json'):files.add(f)
    for mode in ['strict','ignore_checksums']:
        stage=next((src/'ait_attack/contract'/('analysis-'+mode)).glob('attempt-*'))
        for name in ['raw8.csv','action_connection_pairs.csv','evidence_summary.json']:
            files.add(stage/name)
    for f in (src/'ait_attack/review_unified').iterdir():
        if f.is_file() and f.suffix in {'.py','.json'}:files.add(f)
    manifest={}
    for f in sorted(files):
        target=dest/f.relative_to(src);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(f,target)
        digest=hashlib.sha256(f.read_bytes()).hexdigest()
        assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
        if target.suffix=='.md':
            target.write_text(target.read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
        manifest[f.relative_to(src).as_posix()]={'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    attrs='* -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n*.md text eol=lf\nait/source/capture_config/** -whitespace\nait_attack/source/capture_config/** -whitespace\n'
    (dest/'.gitattributes').write_text(attrs,encoding='utf-8',newline='\n')
    (dest/'package_receipt.json').write_text(json.dumps({'files':manifest,'file_count':len(manifest),'bytes':sum(v['bytes'] for v in manifest.values()),'raw_pcaps_and_complete_rotations_included':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    # Published README uses repository-local evidence links rather than workspace aliases.
    body=(src/'README.md').read_text(encoding='utf-8')
    body=body.replace('../独立来源与算法主线_20261118/algorithm_sources.md','../source_method_route_20261118/algorithm_sources.md')
    (dest/'README.md').write_text(body,encoding='utf-8',newline='\n')
    # Rebind the intentionally adjusted Markdown file; source JSON/scripts remain byte-exact.
    manifest['README.md']['sha256']=hashlib.sha256((dest/'README.md').read_bytes()).hexdigest()
    manifest['README.md']['bytes']=(dest/'README.md').stat().st_size
    (dest/'package_receipt.json').write_text(json.dumps({'files':manifest,'file_count':len(manifest),'bytes':sum(v['bytes'] for v in manifest.values()),'raw_pcaps_and_complete_rotations_included':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    summary='# 统一事件与方法落实：2026-11-19\n\n'
    summary+='现行入口为[完整数据、划分与方法决定](unified_events_20261119/README.md)。统一表已落地：46,831 条事件／对象、44,234 条连接观察、58,274 条关联；26 个正常业务评价单元。AIT 31 次正常下载已映射实际连接，两次攻击下载因抓包长度缺损不进入完整流训练。OpTC 已有 41,630 条声明正常连接及整分支预留的 Meterpreter 动作片段。\n\n'
    summary+='唯一方法候选为单流 HGB 与相同 HGB 加固定 19 维历史摘要；代码和真实背景导出已完成。新的 F/V/C 尚未登记，当前不启动训练、评分或阈值搜索。旧 R30／R31 注册及结果不改。\n'
    (repo/'docs/UNIFIED_EVENTS_20261119.md').write_text(summary,encoding='utf-8',newline='\n')
    print(json.dumps({'files':len(manifest),'bytes':sum(v['bytes'] for v in manifest.values()),'destination':str(dest)},ensure_ascii=False))

if __name__=='__main__':main()
