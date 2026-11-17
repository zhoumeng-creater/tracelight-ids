"""Independent audit of existing R30 scores and decisions; no model execution."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ID = ["source_id", "row_id"]
NUM = ["duration_seconds", "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes"]
COUNTS = NUM[1:]
SEEDS = (17, 29, 43)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def clean_frame(frame):
    assert not frame.duplicated(ID).any()
    return frame.sort_values(ID).reset_index(drop=True)


def metrics(frame):
    neg, pos = frame.label.eq(0), frame.label.eq(1)
    n, p = int(neg.sum()), int(pos.sum())
    fp, tp = int((neg & frame.alert).sum()), int((pos & frame.alert).sum())
    return {"rows": len(frame), "normal_n": n, "attack_n": p, "fp": fp, "tp": tp,
            "fn": p-tp, "fp_budget": n//100, "qualified": fp <= n//100,
            "fpr": fp/n if n else None,
            "pauc": float(roc_auc_score(frame.label, frame.score, max_fpr=.01))
            if n and p and "score" in frame and frame.score.notna().all() else None}


def basic(frame):
    result = {k: metrics(g) for k, g in frame.groupby("source_id", sort=True)}
    d = frame.loc[frame.source_id.str.startswith("TQH-D_")]
    b = frame.loc[frame.source_id.str.startswith("TQH-B_")]
    budget_summary = {}
    for role in ('B', 'D', 'all'):
        cells = [v for k,v in result.items() if role == 'all' or k.startswith('TQH-'+role+'_')]
        budget_summary[role] = {'sum_cell_budgets':sum(v['fp_budget'] for v in cells),
            'sum_positive_cell_excess':sum(max(v['fp']-v['fp_budget'],0) for v in cells),
            'aggregate_fp_minus_sum_cell_budgets':sum(v['fp']-v['fp_budget'] for v in cells),
            'failed_cells':sum(not v['qualified'] for v in cells),
            'excess_by_failed_cell':[v['fp']-v['fp_budget'] for v in cells if not v['qualified']]}
    return {"B": metrics(b), "D": metrics(d), "D_TCP": metrics(d.loc[d.ip_protocol.eq(6)]),
            "qualified_B_cells": sum(v["qualified"] for k,v in result.items() if k.startswith("TQH-B_")),
            "qualified_D_cells": sum(v["qualified"] for k,v in result.items() if k.startswith("TQH-D_")),
            "budget_summary":budget_summary, "cells": result}


def signature_set(frame, fields):
    return set(frame[fields].itertuples(index=False, name=None))


def support(frame, pool):
    exact = signature_set(pool, NUM)
    typed = signature_set(pool, ["ip_protocol", *NUM])
    counts = signature_set(pool, COUNTS)
    typed_counts = signature_set(pool, ["ip_protocol", *COUNTS])
    return {"five_numeric_exact":sum(x in exact for x in frame[NUM].itertuples(index=False,name=None)),
            "protocol_five_numeric_exact":sum(x in typed for x in frame[["ip_protocol",*NUM]].itertuples(index=False,name=None)),
            "four_counts_exact":sum(x in counts for x in frame[COUNTS].itertuples(index=False,name=None)),
            "protocol_four_counts_exact":sum(x in typed_counts for x in frame[["ip_protocol",*COUNTS]].itertuples(index=False,name=None))}


def duration(frame):
    if not len(frame):return None
    values=frame.duration_seconds
    return {"min":float(values.min()),"median":float(values.median()),"max":float(values.max()),
            "missing_minus_one":int(values.eq(-1).sum())}


def independent_pair(candidate, reference):
    assert candidate[ID].equals(reference[ID])
    result=[]
    for source in sorted(candidate.source_id.unique()):
        mask=candidate.source_id.eq(source)
        a,b=candidate.loc[mask],reference.loc[mask]
        pos=a.label.eq(1);neg=a.label.eq(0)
        ac,bc=metrics(a),metrics(b)
        result.append({"source_id":source,"candidate":ac,"reference":bc,
                       "added_tp":int((pos&a.alert&~b.alert).sum()),
                       "lost_tp":int((pos&~a.alert&b.alert).sum()),
                       "added_fp":int((neg&a.alert&~b.alert).sum()),
                       "removed_fp":int((neg&~a.alert&b.alert).sum()),
                       "no_tp_drop_in_reference_qualified_cell":not bc['qualified'] or ac['tp']>=bc['tp']})
    return {"cells":result,"added_tp":sum(v['added_tp'] for v in result),
            "lost_tp":sum(v['lost_tp'] for v in result),
            "added_fp":sum(v['added_fp'] for v in result),"removed_fp":sum(v['removed_fp'] for v in result),
            "no_tp_drop_in_reference_qualified_cells":all(v['no_tp_drop_in_reference_qualified_cell'] for v in result),
            "strict_D_cell_tp_gain":any(v['candidate']['tp']>v['reference']['tp'] for v in result if v['source_id'].startswith('TQH-D_'))}


def main(root,out):
    out.mkdir(parents=True,exist_ok=True)
    expected=read(root/'reports/comparison.json')
    inputs=clean_frame(pd.read_parquet(root/'inputs/ED.parquet'))
    f=pd.read_parquet(root/'inputs/F.parquet')
    frames={name:clean_frame(g) for name,g in pd.read_parquet(root/'inputs/reference_decisions.parquet').groupby('model_id')}
    checks=[];model_results={};units=[];identities=[];seed_verdicts={}
    for arm in ('blind','aware'):
      for seed in SEEDS:
        name=f'protocol_{arm}_s{seed}';d=clean_frame(pd.read_parquet(root/f'reports/{name}-decisions.parquet'))
        s=clean_frame(pd.read_parquet(root/f'evidence/{name}/ED-scores.parquet'))
        assert d[ID].equals(inputs[ID]) and s[ID].equals(inputs[ID])
        assert np.array_equal(d[NUM+['ip_protocol','label']],inputs[NUM+['ip_protocol','label']])
        assert np.array_equal(s.score,d.score)
        calibration=read(root/f'evidence/{name}/calibration.json')
        c=pd.read_parquet(root/f'evidence/{name}/C-scores.parquet')
        assert len(c)==5511 and calibration['rank']==5457
        assert float(np.sort(c.score.to_numpy())[5456])==calibration['threshold']
        assert int(c.score.gt(calibration['threshold']).sum())==calibration['calibration_fp']==54
        assert calibration['ready'] and d.ready.all()
        assert np.all(d.threshold==calibration['threshold'])
        assert np.array_equal(d.alert,d.score>calibration['threshold'])
        frames[name]=d;result=basic(d)
        for role,refrole in [('D','D_pooled'),('D_TCP','D_TCP')]:
          assert result[role]['tp']==expected['models'][name][refrole]['tp']
          assert result[role]['fp']==expected['models'][name][refrole]['fp']
          assert result[role]['pauc']==expected['models'][name][refrole]['pauc_0_01']
        savedcells={v['source_id']:v for v in expected['models'][name]['cells']}
        for source,cell in result['cells'].items():
          for key in ('normal_n','attack_n','tp','fp','fp_budget','qualified'):
            assert cell[key]==savedcells[source][key]
          units.append({'model_id':name,'source_id':source,**cell})
        fp=d.loc[d.source_id.str.startswith('TQH-D_')&d.label.eq(0)&d.alert]
        result['D_FP_by_protocol']={str(int(proto)):{'rows':len(g),'duration':duration(g),
            'ports':{str(int(k)):int(v) for k,v in g.resp_port.value_counts().items()},
            'counts_medians':{col:float(g[col].median()) for col in COUNTS}}
            for proto,g in fp.groupby('ip_protocol')}
        result['D_FP_support']={f'{profile}{label}':support(fp,f.loc[f.profile.eq(profile)&f.label.eq(label)])
                              for profile in ('A','B') for label in (0,1)}
        result['D_TCP_FP_support']={f'{profile}{label}':support(fp.loc[fp.ip_protocol.eq(6)],f.loc[f.profile.eq(profile)&f.label.eq(label)])
                              for profile in ('A','B') for label in (0,1)}
        result['D_TCP_FP_count_patterns']=[{**dict(zip(COUNTS,pattern)),'rows':int(n)} for pattern,n in
            fp.loc[fp.ip_protocol.eq(6)].groupby(COUNTS).size().sort_values(ascending=False).items()]
        result['D_FN_rows']=d.loc[d.source_id.str.startswith('TQH-D_')&d.label.eq(1)&~d.alert,
                                 [*ID,'ip_protocol','resp_port',*NUM,'score']].to_dict('records')
        result['C_threshold']=calibration['threshold'];model_results[name]=result
        checks.append(name+': full original-score/decision/threshold/cell/ranking reproduction')
    for name in frames:
      if name not in model_results:model_results[name]=basic(frames[name])
    pairs={};old_loss={}
    for seed in SEEDS:
      a=f'protocol_blind_s{seed}';b=f'protocol_aware_s{seed}';candidate=frames[a]
      refnames=[b,'legacy_hgb_or_rule',*['r26_target_mlp'+str(s) for s in SEEDS],f'r29_tabm_ple_s{seed}']
      pairs[str(seed)]={}
      for name in refnames:
        reference=frames[name];pair=independent_pair(candidate,reference);pairs[str(seed)][name]=pair
        if name!=f'r29_tabm_ple_s{seed}':
          reported=expected['seeds'][str(seed)]['comparisons'][name]
          for key in ('added_tp','lost_tp','no_tp_drop_in_reference_qualified_cells','strict_D_cell_tp_gain'):
            assert pair[key]==reported[key]
          for cell,saved in zip(pair['cells'],reported['cells']):
            assert cell['source_id']==saved['source_id']
            for key in ('added_tp','lost_tp','no_tp_drop_in_reference_qualified_cell'):assert cell[key]==saved[key]
            for event,flag in [('added_tp',candidate.label.eq(1)&candidate.alert&~reference.alert),('lost_tp',candidate.label.eq(1)&~candidate.alert&reference.alert)]:
              cids=set(candidate.loc[flag&candidate.source_id.eq(cell['source_id']),ID].itertuples(index=False,name=None))
              sids=set((x['source_id'],x['row_id']) for x in saved[event+'_identities'])
              assert cids==sids
        events={'added_tp':candidate.label.eq(1)&candidate.alert&~reference.alert,
                'lost_tp':candidate.label.eq(1)&~candidate.alert&reference.alert}
        if name==b:events.update(added_fp=candidate.label.eq(0)&candidate.alert&~reference.alert,
                                removed_fp=candidate.label.eq(0)&~candidate.alert&reference.alert)
        for event,mask in events.items():
          for row in candidate.loc[mask,[*ID,'ip_protocol','resp_port',*NUM]].to_dict('records'):
            identities.append({'model_id':a,'reference':name,'event':event,**row})
      dmask=candidate.source_id.str.startswith('TQH-D_')
      aware=frames[b]
      pairs[str(seed)]['D_FP_change_by_protocol']={str(proto):{
        'added_fp':int((dmask&candidate.ip_protocol.eq(proto)&candidate.label.eq(0)&candidate.alert&~aware.alert).sum()),
        'removed_fp':int((dmask&candidate.ip_protocol.eq(proto)&candidate.label.eq(0)&~candidate.alert&aware.alert).sum())}
        for proto in (1,6,17)}
      old=frames[f'r29_tabm_ple_s{seed}'];old_loss[str(seed)]={}
      for name in ['legacy_hgb_or_rule',*['r26_target_mlp'+str(s) for s in SEEDS]]:
        ref=frames[name];mask=dmask&ref.label.eq(1)&ref.alert&~old.alert
        old_loss[str(seed)][name]={'old_lost_n':int(mask.sum()),'duration':duration(ref.loc[mask]),
            'blind_recovered':int((mask&candidate.alert).sum()),'aware_recovered':int((mask&aware.alert).sum()),
            'blind_still_lost':int((mask&~candidate.alert).sum()),
            'aware_still_lost':int((mask&~aware.alert).sum())}
        saved=expected['seeds'][str(seed)]['historical_R29_loss_diagnostics'][name]
        assert saved['old_lost_n']==int(mask.sum()) and saved['recovered_n']==int((mask&candidate.alert).sum())
      result=model_results[a];control=model_results[b]
      mechanism=result['D']['pauc']>control['D']['pauc'] and result['D']['tp']>=control['D']['tp']
      effects={
        'all_15_cells_qualified':all(x['qualified'] for x in result['cells'].values()),
        'B_each_cell_24_TP_total_72':result['B']['tp']==72 and all(v['tp']==24 for k,v in result['cells'].items() if k.startswith('TQH-B_')),
        'no_TP_drop_in_qualified_historical_cells':all(pairs[str(seed)][n]['no_tp_drop_in_reference_qualified_cells'] for n in ['legacy_hgb_or_rule',*['r26_target_mlp'+str(s) for s in SEEDS]]),
        'strict_D_cell_TP_gain_over_legacy_union':pairs[str(seed)]['legacy_hgb_or_rule']['strict_D_cell_tp_gain'],
        'no_TP_drop_in_qualified_control_cells':pairs[str(seed)][b]['no_tp_drop_in_reference_qualified_cells'],
        'strict_D_total_TP_gain_over_control':result['D']['tp']>control['D']['tp']}
      assert effects==expected['seeds'][str(seed)]['effect_checks']
      assert mechanism==expected['seeds'][str(seed)]['mechanism_pass']
      seed_verdicts[str(seed)]={'mechanism_pass':bool(mechanism),'effect_checks':effects,
          'effect_pass':all(effects.values())}
    all_mechanism=all(v['mechanism_pass'] for v in seed_verdicts.values())
    all_effect=all(v['effect_pass'] for v in seed_verdicts.values())
    assert expected['all_seed_mechanism_pass']==all_mechanism and expected['adoption_pass']==all_effect
    pd.DataFrame(units).to_csv(out/'detection_cells.csv',index=False)
    pd.DataFrame(identities).to_csv(out/'detection_identity_changes.csv',index=False)
    report={'scope':'Existing frozen scores/decisions only; no model loads, forwards, fitting, new thresholds or candidate policies',
            'independent_reproduction_passed':True,'checks':checks,'models':model_results,'paired':pairs,'old_loss_recovery':old_loss,
            'source_bindings':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['reports/comparison.json','inputs/F.parquet','inputs/ED.parquet','inputs/reference_decisions.parquet']},
            'interpretation':'Input-strategy plus its registered selection process effect; selected optimization steps differ, not a fixed-step architecture contrast',
            'seed_verdicts':seed_verdicts,
            'all_three_mechanism_pass':all_mechanism,'all_three_effect_pass':all_effect,'adoption_pass':all_effect}
    (out/'detection_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('Independent detection reproduction passed; cached comparisons and identities agree')
    for name,m in model_results.items():print(name,'B',m['B']['tp'],m['B']['fp'],'D',m['D']['tp'],m['D']['fp'],'qualified',m['qualified_B_cells'],m['qualified_D_cells'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--return-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();main(a.return_dir,a.output_dir)
