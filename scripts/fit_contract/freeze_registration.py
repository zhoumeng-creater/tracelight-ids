"""Freeze already-materialized identities and file hashes, without running models."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from materialize import load, rows, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delivery', type=Path, required=True)
    args = parser.parse_args()
    base = args.delivery.resolve()
    data = base / 'dataset'
    contract_path = base / 'fit_contract/fit_contract.json'
    contract = load(contract_path)
    model = list(rows(data / 'model_rows.jsonl.gz'))
    events = list(rows(data / 'event_metadata.jsonl.gz'))
    event_by_id = {r['event_id']: r for r in events}
    summary = load(data / 'summary.json')
    preflight = load(base / 'fit_contract/inputs_preflight/inputs_audit.json')
    assert preflight['mode'] == 'inputs_only'
    assert all(preflight[key] == 0 for key in ('model_constructions', 'fit_calls', 'score_calls', 'threshold_calculations'))
    assert preflight['metadata']['all_event_references_resolve']
    assert preflight['model_rows'] == len(model)
    for role in ['F', 'V', 'C']:
        selected = [r for r in model if r['dataset_role'] == role]
        contract[role]['network_row_ids'] = [r['flow_id'] for r in selected]
        contract[role]['stable_network_row_ids'] = [r['stable_flow_id'] for r in selected]
        contract[role]['label_counts'] = dict(Counter(str(r['label']) for r in selected))
        if role in ('F', 'V'):
            kinds = Counter(kind for r in selected if r['label'] == 1
                            for kind in {event_by_id[eid]['kind'] for eid in r['event_ids']})
            assert kinds['webshell_command'] > 0 and kinds['attack_download'] > 0, (role, kinds)
            contract[role]['positive_connections_by_event_kind'] = dict(kinds)
        else:
            assert len(selected) == 25 and all(r['label'] == 0 for r in selected)
            contract[role]['status'] = '25_whole_flow_ids_bound_to_new_full_capture_export'
    contract['status'] = 'identities_features_background_and_minimal_runner_materialized'
    contract['not_materialized_yet'] = ['actual laboratory comparison results and measured model costs']
    contract['measurement_contract']['name'] = 'declared-ip-truncation-v1'
    contract['measurement_contract']['actual_full_capture_rule'] = (
        'IPv4 header complete, checksum valid, unfragmented, complete supported TCP/UDP/ICMP header; '
        'raise orig_len only when less than L2 offset + declared IPv4 total length. '
        'All records are inspected without label/action filtering. No whole-background ACK proof is claimed.')
    contract['measurement_contract']['target_sequence_validation_scope'] = (
        'The original three russell download target connections have separate TCP sequence/ACK evidence; '
        'this is not a claim of individually validating every full-background sequence extent.')
    contract['background_summary'] = summary
    contract['same_clock_tie_order'] = 'stable capture SHA + protocol + ordered endpoints + start, not random Zeek UID'
    contract['real_training_calls'] = 0
    save(contract_path, contract)
    names = ['model_rows.jsonl.gz', 'inference_rows.jsonl.gz', 'event_metadata.jsonl.gz', 'normal_units.jsonl.gz']
    registration = {
        'schema': 'fit-contract-registration-v1', 'data_materialized': True,
        'comparison_ready_on_laboratory_workstation': True, 'real_training_launched': False,
        'runner_inputs': [{'name': name, 'sha256': sha(data / name), 'bytes': (data / name).stat().st_size}
                          for name in names],
        'fit_contract_sha256': sha(contract_path),
        'preflight_sha256': sha(base / 'fit_contract/inputs_preflight/inputs_audit.json'),
        'preflight_summary': {k: v for k, v in preflight.items() if k != 'metadata'},
        'parameters': {'max_iter': 200, 'max_leaf_nodes': 31, 'learning_rate': 0.1,
                       'early_stopping': False, 'random_state': 20261111},
        'fit_calls': 2, 'fit_role': 'F', 'refits': 0,
        'method_arms': ['raw8_log1p', 'raw8_log1p_plus_fixed_19_history'],
        'threshold_rule': 'max of 25 C normal scores per frozen model; strictly greater than alerts',
        'selection': 'none; report both arms; no seed, capacity, window, or threshold search',
        'scope': 'limited source-separated development comparison, not a new algorithm or blind deployment claim',
        'reserved_sources_scored': [],
    }
    save(base / 'fit_contract/registration.json', registration)
    print(json.dumps({'status': contract['status'], 'runner_inputs': registration['runner_inputs']}, indent=2))


if __name__ == '__main__':
    main()
