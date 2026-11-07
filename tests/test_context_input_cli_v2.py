"""New v2 commands select explicit contracts; v1 entry remains separate."""

from ids.context_cache import __main__ as cli


def test_v2_prepare_selects_independent_profile(monkeypatch, tmp_path):
    from ids.context_cache import lab_data
    calls = []
    monkeypatch.setattr(lab_data, 'prepare_lab_data', lambda *a, **k: calls.append((a,k)) or {'ok':True})
    args=cli.build_parser().parse_args(['lab-prepare-v2','--raw-csv',str(tmp_path/'raw.csv'),
                                      '--output',str(tmp_path/'prepared')])
    assert cli.csv_command(args)=={'ok':True}
    assert calls[0][1] == {'sampling_profile':'group_disjoint_flow_v2'}
    args.command='lab-prepare'
    cli.csv_command(args)
    assert calls[1][1] == {}


def test_input_comparison_cli_returns_each_paired_comparison(monkeypatch, tmp_path):
    from ids.context_cache import csv_workflow
    payload={'scope':'synthetic','candidate_count':3,'evaluation':{},
             'paired':{'legacy_full_vs_full':{'changes':{'new_fn':1}},
                       'full_vs_qcoc':{'changes':{'new_fn':2}}}}
    monkeypatch.setattr(csv_workflow,'run_input_comparison',lambda *a:payload)
    args=cli.build_parser().parse_args(['lab-input-compare','--config',str(tmp_path/'p.json'),
                                      '--output',str(tmp_path/'result')])
    result=cli.csv_command(args)
    assert result['candidate_count']==3
    assert result['paired_changes']=={'legacy_full_vs_full':{'new_fn':1},'full_vs_qcoc':{'new_fn':2}}
