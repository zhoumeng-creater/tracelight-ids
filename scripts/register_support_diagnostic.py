"""Materialize chronological whole-cell roles from already verified publisher metadata."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = json.loads((ROOT / 'configs/response-evidence-20261112/tqh.json').read_text(encoding='utf-8'))
b = sorted((s for s in source['sources'] if s['profile'] == 'B'), key=lambda s: s['first_flow_end_ms'])
d = sorted((s for s in source['sources'] if s['profile'] == 'D'), key=lambda s: s['first_flow_end_ms'])
assert len(b) == len(d) == 12
assert all(a['last_flow_end_ms'] < z['first_flow_end_ms'] for a, z in zip(b, b[1:]))
assert max(s['last_flow_end_ms'] for s in b) < min(s['first_flow_end_ms'] for s in d)
groups = {'target_train': b[:6], 'normal_calibration': b[6:9], 'forward_development': b[9:],
          'transport_external': d}
config = {
    'version': 'support-diagnostic-r26-v1', 'registered_on': '2026-11-15',
    'status': 'already_opened_development_diagnostic',
    'label_availability': 'offline_retrospective_assumption: historical B labels and normal confirmation '
                          'are available by the end of the calibration cells; actual historical availability unknown',
    'roles': {role: [s['name'] for s in rows] for role, rows in groups.items()},
    'source_train': ['CTU-IoT-Malware-Capture-1-1', 'CTU-IoT-Malware-Capture-8-1'],
    'groups': ['hgb', 'mlp17', 'mlp29', 'mlp43'],
    'supports': ['source', 'target', 'mixed'], 'views': ['raw8', 'ports_zero8'],
    'alpha': 0.01, 'normal_calibration_rule': 'same B calibration identities for every arm, strict score > quantile',
    'weighting': {'source': 'original balanced 4096+4096, uniform', 'target': 'equal total mass by binary class',
                  'mixed': 'equal total mass per domain and binary class; same target identities as target-only'},
    'new_fits': 20, 'reused_models': 4, 'evaluation_cells': 15,
    'fixed_baselines': {
        'udp443_duration30': 'ip_protocol == 17 and (orig_port == 443 or resp_port == 443) and duration_seconds >= 30',
        'hgb_source_raw_or_rule': 'source/raw8/hgb with frozen B calibration OR udp443_duration30',
        'original_hgb': 'source/raw8/hgb with original frozen R23 calibration threshold',
        'original_hgb_or_rule': 'original_hgb OR udp443_duration30',
        'origin': 'already-opened R23 raw-workpoint diagnosis; no threshold search in this batch',
        'primary_reference': 'original_hgb_or_rule',
    },
    'not_claimed': ['new_attack_implementation', 'new_blind_confirmation', 'scanning_improvement',
                    'malicious_download_improvement', 'original_algorithm', 'online_label_availability'],
}
dest = ROOT / 'configs/support-diagnostic-20261115/registration.json'
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(dest)
