"""Read-only F-defined deterministic bin counts. No score/model/threshold operations."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT.parent.parent / '实验归档/20261117_第三十轮模型训练重设计/回传/inputs'
COLS = ['orig_pkts', 'resp_pkts', 'orig_ip_bytes', 'resp_ip_bytes']
MIN_N, MAX_DEPTH = 256, 4

def identity_hash(frame):
    canonical = frame[['source_id','row_id']].sort_values(['source_id','row_id'], kind='stable')
    rows = [[str(s), int(r)] for s,r in canonical.itertuples(index=False,name=None)]
    return hashlib.sha256(json.dumps(rows,ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()

def values(frame):
    arr = frame[COLS].to_numpy(dtype=np.float64)
    arr[np.isnan(arr)] = -1
    assert np.isfinite(arr).all() and ((arr >= 0) | (arr == -1)).all()
    return arr

def tree(x, idx, depth, key):
    node = {'key':key, 'depth':depth, 'n':int(len(idx))}
    if depth == MAX_DEPTH or len(idx) < 2 * MIN_N:
        return node
    for offset in range(4):
        f = (depth + offset) % 4
        v, c = np.unique(x[idx, f], return_counts=True)
        left = np.cumsum(c)[:-1]
        legal = np.flatnonzero((left >= MIN_N) & (len(idx) - left >= MIN_N))
        if not len(legal):
            continue
        j = min(legal, key=lambda k:(abs(2 * int(left[k]) - len(idx)), float(v[k])))
        cut = float(v[j])
        mask = x[idx, f] <= cut
        node.update(feature=COLS[f], cut_le=cut,
                    left=tree(x, idx[mask], depth+1, key+'0'),
                    right=tree(x, idx[~mask], depth+1, key+'1'))
        return node
    return node

def leaves(node):
    if 'left' not in node:
        return [node]
    return leaves(node['left']) + leaves(node['right'])

def route(row, node):
    while 'left' in node:
        node = node['left'] if row[COLS.index(node['feature'])] <= node['cut_le'] else node['right']
    return node['key']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=INPUT)
    parser.add_argument('--output', type=Path, default=ROOT/'f_group_counts.json')
    args = parser.parse_args()
    fpath, vpath = args.input_dir/'F.parquet', args.input_dir/'V.parquet'
    f = pd.read_parquet(fpath)
    normal = f[f.label.eq(0)].reset_index(drop=True)
    x = values(normal)
    raw_keys = ['duration_seconds'] + COLS
    raw_profile_counts = normal.groupby(raw_keys, dropna=False).profile.nunique()
    trees = {p:tree(x, np.flatnonzero(normal.profile.eq(p)), 0, p+'_') for p in ['A','B']}
    frouted = [route(x[i], trees[p]) for i,p in enumerate(normal.profile)]
    summary = [{'key':n['key'],'profile':p,'F_normal_n':n['n']} for p,t in trees.items() for n in leaves(t)]
    v = pd.read_parquet(vpath)
    vnormal = v[v.label.eq(0)].reset_index(drop=True)
    xv = values(vnormal)
    routed = [route(xv[i], trees[p]) if p in trees else 'UNSEEN_PROFILE' for i,p in enumerate(vnormal.profile)]
    vc = pd.Series(routed).value_counts().to_dict()
    for s in summary:
        s['V_normal_n'] = int(vc.get(s['key'],0))
        s['F_identity_sha256'] = identity_hash(normal.loc[np.asarray(frouted) == s['key']])
        s['V_identity_sha256'] = identity_hash(vnormal.loc[np.asarray(routed) == s['key']])
    result = {'purpose':'static F-only bin support counts, no candidate scores or model operations',
              'inputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [fpath,vpath]},
              'identity_sha256':{'F_all':identity_hash(f),'F_normal':identity_hash(normal),
                                 'V_all':identity_hash(v),'V_normal':identity_hash(vnormal)},
              'identity_hash_format':'UTF8 compact JSON sorted [source_id string,row_id integer] records; ensure_ascii=False',
              'fields':COLS,'min_leaf_n':MIN_N,'max_depth':MAX_DEPTH,
              'F_normal_n':len(normal),'V_normal_n':len(vnormal),'group_n':len(summary),
              'F_normal_exact_raw_input_keys_with_multiple_profiles':int((raw_profile_counts > 1).sum()),
              'groups':summary,'trees':trees,'V_unseen_profile_n':int(vc.get('UNSEEN_PROFILE',0)),
              'F_missing_rows':int((x == -1).any(axis=1).sum()),
              'V_missing_rows':int((xv == -1).any(axis=1).sum())}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'trees'},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
