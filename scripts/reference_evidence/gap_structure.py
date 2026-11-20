"""Real structure/lossless-pairing facts, no detector score or threshold."""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
import random
from itertools import product
from pathlib import Path
from paired_no_gap import pair_runs,unpack,full_bottleneck,reference_cost


def enumerate_full_paths(costs,i=0,j=0,bottleneck=0.):
    bottleneck=max(bottleneck,costs[i][j])
    n,m=len(costs),len(costs[0])
    if i==n-1 and j==m-1:return [bottleneck]
    scores=[]
    for ni,nj in ((i+1,j+1),(i+1,j),(i,j+1)):
        if ni<n and nj<m:
            scores.extend(enumerate_full_paths(costs,ni,nj,bottleneck))
    return scores


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--geometry',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with gzip.open(args.geometry,'rt',encoding='utf8') as f:
        rows=[json.loads(line) for line in f]
    assert not any(any(x in r['source'].lower() for x in ('russell','harrison','meterpreter')) for r in rows)
    paired={}
    for r in rows:
        if not r['bursts']:continue
        p=pair_runs(r['bursts'])
        assert unpack(p)==r['bursts']
        paired[r['flow_id']]=p
    bank=[r for r in rows if r['role'].startswith('F') and r['label']==0 and r['geometry_eligible']]
    assert len({r['stable_flow_id'] for r in bank})==len(bank)
    bankids={r['stable_flow_id'] for r in bank}
    bytype=Counter(paired[r['flow_id']].boundary_type for r in bank)
    bypattern=Counter(tuple(x[0] for x in r['bursts']) for r in bank)
    groups={}
    for source in sorted({r['source'] for r in rows}):
        for kind in ('normal','control','download','unknown','other_attack'):
            selected=[r for r in rows if r['source']==source and r['target_kind']==kind]
            if not selected:continue
            eligible=[r for r in selected if r['geometry_eligible']]
            checks=[]
            for r in eligible:
                p=paired[r['flow_id']]
                assert p.pairs
                excluded=int(r['stable_flow_id'] in bankids)
                strict=bypattern[tuple(x[0] for x in r['bursts'])]>excluded
                supported=bytype[p.boundary_type]>excluded
                checks.append((strict,supported,r))
            groups[source+'/'+kind]={
                'connections':len(selected),'eligible':len(eligible),
                'strict_direction_no_gap_reference_available':sum(v[0] for v in checks),
                'paired_no_gap_reference_available':sum(v[1] for v in checks),
                'pair_counts':dict(sorted(Counter(len(paired[r['flow_id']].pairs) for r in eligible).items())),
                'strict_unavailable_ids':[v[2]['flow_id'] for v in checks if not v[0]],
                'pair_unavailable_ids':[v[2]['flow_id'] for v in checks if not v[1]],
            }
    # Abstract matrices check the operator; they are not real traffic scores.
    q=pair_runs([[0,2],[1,3],[0,5],[1,7]])
    r=pair_runs([[0,2],[1,3]])
    cost,path=reference_cost(q,r,[[0.],[5.]])
    assert cost==5. and {i for i,j in path}=={0,1} and {j for i,j in path}=={0}
    repeated,path2=full_bottleneck([[0.],[0.]])
    assert repeated==0.
    exhaustive_cases=0
    for n in range(1,4):
        for m in range(1,4):
            for values in product((0.,1.),repeat=n*m):
                costs=[list(values[i*m:(i+1)*m]) for i in range(n)]
                found,foundpath=full_bottleneck(costs)
                assert found==min(enumerate_full_paths(costs))
                assert {i for i,j in foundpath}==set(range(n))
                assert {j for i,j in foundpath}==set(range(m))
                exhaustive_cases+=1
    rng=random.Random(20261119)
    brute_cases=0
    for n in range(1,5):
        for m in range(1,5):
            costs=[[rng.randrange(1,20) for _ in range(m)] for _ in range(n)]
            found,foundpath=full_bottleneck(costs)
            assert found==min(enumerate_full_paths(costs))
            assert {i for i,j in foundpath}==set(range(n))
            assert {j for i,j in foundpath}==set(range(m))
            brute_cases+=1
    tailq=pair_runs([[0,2],[1,3],[0,8]])
    tailr=pair_runs([[0,2],[1,3],[0,4]])
    tailscore,_=reference_cost(tailq,tailr,[[0.]],8.)
    assert tailscore==8.
    try:
        reference_cost(tailq,tailr,[[0.]])
    except ValueError:pass
    else:raise AssertionError('A real tail must not be silently omitted')
    try:
        reference_cost(tailq,r,[[0.]])
    except ValueError:pass
    else:raise AssertionError('Missing tail must not be replaced by null')
    # Real repetition equivalence is established from exact token identities,
    # without invoking a learned/fixed-distance detector on these connections.
    collapsed_groups=defaultdict(list)
    for row in rows:
        if not row['geometry_eligible']:continue
        p=paired[row['flow_id']]
        collapsed=[]
        for token in p.pairs:
            if not collapsed or collapsed[-1]!=token:collapsed.append(token)
        collapsed_groups[(p.first_direction,tuple(collapsed),p.tail)].append(row)
    collisions=[]
    for (direction,collapsed,tail),members in collapsed_groups.items():
        profiles=defaultdict(list)
        for row in members:profiles[tuple(map(tuple,row['bursts']))].append(row)
        if len(profiles)<2:continue
        profile_rows=[]
        for profile,observations in sorted(profiles.items()):
            representative=observations[0]
            profile_rows.append({'bursts':profile,'pair_count':len(profile)//2,
                'unique_payload_bytes':representative['unique_payload_bytes'],
                'connections':[{'flow_id':r['flow_id'],'source':r['source'],'role':r['role'],
                    'target_kind':r['target_kind']} for r in observations]})
        collisions.append({'first_direction':direction,'collapsed_pairs':collapsed,'tail':tail,
            'connections':len(members),'profile_count':len(profiles),
            'kind_counts':dict(Counter(r['target_kind'] for r in members)),
            'profiles':profile_rows})
    report={'source_sha256':hashlib.sha256(args.geometry.read_bytes()).hexdigest(),
        'F_normal_reference_count':len(bank),'F_boundary_types':{str(k):v for k,v in bytype.items()},
        'real_lossless_roundtrip_connections':len(paired),'sources':groups,
        'abstract_checks':{'unmatched_later_pair_cannot_disappear_cost':cost,
                           'repeated_identical_pairs_may_warp_at_zero_cost':repeated,
                           'exhaustive_binary_matrices_up_to_3x3':exhaustive_cases,
                           'small_matrix_bruteforce_and_full_coverage_cases':brute_cases,
                           'tail_cost_is_included':tailscore,
                           'missing_tail_cost_and_boundary_mismatch_rejected':True},
        'real_repetition_equivalence':{'groups':len(collisions),
            'connections':sum(c['connections'] for c in collisions),
            'distinct_original_profiles':sum(c['profile_count'] for c in collisions),
            'kind_counts':dict(sum((Counter(c['kind_counts']) for c in collisions),Counter())),
            'evidence':collisions},
        'exclusion':'same stable physical measurement identity, not all identical value profiles',
        'model_calls':0,'thresholds_selected':0,'structural_availability_not_numerical_similarity':True}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
    summary={k:v for k,v in report.items() if k not in ('sources','real_repetition_equivalence')}
    summary['real_repetition_equivalence']={k:v for k,v in report['real_repetition_equivalence'].items() if k!='evidence'}
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
