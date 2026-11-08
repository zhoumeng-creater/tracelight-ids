"""End-to-end frozen controls using only synthetic CSVs and random tiny weights."""
import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ids.common import read_json, sha256_file, write_json
from ids.context_cache.calibration import apply_threshold, register_threshold
from ids.context_cache.csv_workflow import V2_POLICIES
from ids.context_cache.preprocessing import CSVPreprocessor, CSVSchema, PROTECTED_IDS_PROFILE, read_csv
from ids.context_cache.scanning_inputs import ARMS
from ids.context_cache.scanning_protocol import frozen_template


@pytest.fixture
def official(monkeypatch):
    torch = pytest.importorskip('torch')
    source = Path(__file__).resolve().parents[1] / 'local/vendor/tabicl/src'
    if not (source / 'tabicl/_model/tabicl.py').is_file():
        pytest.skip('pinned official source absent; no download')
    monkeypatch.syspath_prepend(str(source))
    from ids.context_cache.upstream import require_pinned_tabicl
    require_pinned_tabicl()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield torch
    torch.set_num_threads(previous_threads)


@pytest.fixture
def tiny_protocol(tmp_path, official):
    from tabicl._model.tabicl import TabICL
    from ids.context_cache.scanning_workflow import _score_matrix
    from ids.context_cache.tabicl_adapter import CompactPredictor, FullCacheAdapter

    torch = official
    spec = frozen_template()
    spec.update(data_kind='synthetic', build_device='cpu', seed=847, query_batch_size=128)
    schema = CSVSchema.from_dict(spec['schema'])
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    frames = {}
    for index, (role, n) in enumerate((('support', 32), ('calibration', 99), ('evaluation', 24))):
        rng = np.random.default_rng(8150 + index)
        frame = pd.DataFrame(rng.normal(size=(n, 48)), columns=schema.features)
        frame['L4_SRC_PORT'] = np.arange(n) + 1024 + 500 * index
        frame['L4_DST_PORT'] = np.where(np.arange(n) % 2, 80, 443)
        frame['PROTOCOL'] = np.where(np.arange(n) % 3, 6, 17)
        frame['L7_PROTO'] = np.where(np.arange(n) % 2, 1, 91)
        frame['label'] = 0 if role == 'calibration' else np.arange(n) % 2
        frame['sample_id'] = [f'{role}-{i}' for i in range(n)]
        frame['communication_group'] = [f'{role}-group-{i}' for i in range(n)]
        frame['attack_family'] = np.where(frame['label'], 'Scanning', 'Benign')
        path = prepared / f'{role}.csv'
        frame.to_csv(path, index=False)
        spec['files'][role] = {'path': str(path), 'sha256': sha256_file(path), 'rows': n}
        frames[role] = read_csv(path, schema)
    prep = CSVPreprocessor(schema, profile=PROTECTED_IDS_PROFILE).fit(frames['support'])
    assert prep.n_features_out_ == 48
    prep_path = tmp_path / 'frozen_synthetic_preprocessor.joblib'
    prep.save(prep_path)
    spec['artifacts']['preprocessor'] = {'path': str(prep_path), 'sha256': sha256_file(prep_path)}
    spec.update(center=float(prep.protected_mean_[0]), scale=float(prep.protected_scale_[0]))
    model_config = {'max_classes': 10, 'embed_dim': 8, 'col_num_blocks': 1, 'col_nhead': 2,
                    'col_num_inds': 4, 'row_num_blocks': 1, 'row_nhead': 2, 'row_num_cls': 2,
                    'icl_num_blocks': 2, 'icl_nhead': 2, 'ff_factor': 2, 'dropout': 0., 'zero_init': False}
    with torch.random.fork_rng():
        torch.manual_seed(847)
        model = TabICL(**model_config).eval()
    checkpoint = tmp_path / 'random_tiny_NOT_PRETRAINED.ckpt'
    torch.save({'config': model_config, 'state_dict': model.state_dict()}, checkpoint)
    spec['checkpoint'] = {'path': str(checkpoint), 'sha256': sha256_file(checkpoint)}
    adapter = FullCacheAdapter.from_checkpoint(checkpoint, expected_sha256=sha256_file(checkpoint))
    adapter.build(torch.from_numpy(prep.transform(frames['support'])).unsqueeze(0),
                  torch.from_numpy(frames['support']['label'].to_numpy()).unsqueeze(0))
    baseline_model = tmp_path / 'old_full_model.pt'
    adapter.as_predictor().save(baseline_model)
    predictor = CompactPredictor.load(baseline_model)
    c_scores = _score_matrix(predictor, prep.transform(frames['calibration']), spec['query_batch_size'])
    # A query at the calibration maximum probes strict-threshold CSV boundaries.
    maximum_row = frames['calibration'].iloc[int(np.argmax(c_scores))]
    frames['evaluation'].loc[0, list(schema.features)] = maximum_row.loc[list(schema.features)]
    e_path = Path(spec['files']['evaluation']['path'])
    frames['evaluation'].to_csv(e_path, index=False)
    spec['files']['evaluation']['sha256'] = sha256_file(e_path)
    frames['evaluation'] = read_csv(e_path, schema)
    e_scores = _score_matrix(predictor, prep.transform(frames['evaluation']), spec['query_batch_size'])
    spec['baseline_model_sha256'] = sha256_file(baseline_model)
    registration = register_threshold(c_scores, .01, model_sha256=spec['baseline_model_sha256'],
                                      preprocessing_sha256=sha256_file(prep_path),
                                      calibration_sha256=spec['files']['calibration']['sha256'])
    assert abs(e_scores[0] - registration['threshold']) < 1e-5
    threshold = tmp_path / 'baseline_threshold.json'
    write_json(threshold, registration)
    spec['artifacts']['baseline_threshold'] = {'path': str(threshold), 'sha256': sha256_file(threshold)}
    saved_c = pd.DataFrame({'row_id': frames['calibration']['sample_id'], 'label': 0, 'score': c_scores,
                            'predicted_attack': apply_threshold(c_scores, registration)})
    saved_e = pd.DataFrame({'row_id': frames['evaluation']['sample_id'],
                            'label': frames['evaluation']['label'], 'full_score': e_scores,
                            'full_predicted': apply_threshold(e_scores, registration),
                            'legacy_full_score': e_scores + 1, 'legacy_full_predicted': True,
                            'attack_family': frames['evaluation']['attack_family']})
    for role, frame in [('baseline_calibration', saved_c), ('baseline_evaluation', saved_e)]:
        path = tmp_path / f'{role}.csv'
        frame.to_csv(path, index=False)
        spec['artifacts'][role] = {'path': str(path), 'sha256': sha256_file(path)}
    receipt = {'format': 'tracelight-nf-ton-v3-preparation-v2', 'data_kind': 'synthetic',
               'schema': spec['schema'], 'seed': spec['seed'], 'files': copy.deepcopy(spec['files']),
               **V2_POLICIES, 'source': {'official_release_verified': False}}
    path = prepared / 'preparation.json'
    write_json(path, receipt)
    spec['artifacts']['preparation'] = {'path': str(path), 'sha256': sha256_file(path)}
    config = tmp_path / 'protocol.json'
    write_json(config, spec)
    return config, spec, frames, prep, c_scores, e_scores, registration


def _forbidden(*args, **kwargs):
    raise AssertionError('No new preprocessing fit or artifact deserialization permitted')


def test_three_full_caches_frozen_preprocessing_and_cold_cli(tiny_protocol, monkeypatch, tmp_path):
    from ids.context_cache.scanning_workflow import run_scanning, verify_scanning_result
    from ids.context_cache.tabicl_adapter import FullCacheAdapter
    import joblib
    import torch

    config, spec, frames, prep, *_ = tiny_protocol
    built = []
    original_build = FullCacheAdapter.build

    def record_build(self, x, y):
        built.append((self.provenance['scanning_arm'], x.detach().cpu().numpy().copy(), y.clone()))
        return original_build(self, x, y)

    monkeypatch.setattr(FullCacheAdapter, 'build', record_build)
    monkeypatch.setattr(CSVPreprocessor, 'fit', _forbidden)
    output = tmp_path / 'comparison'
    report = run_scanning(config, output)
    assert [arm for arm, _, _ in built] == list(ARMS)
    assert report['workload'] == {'full_context_builds': 3, 'calibration_scored_rows': 297,
                                  'evaluation_scored_rows': 72, 'optimizer_updates': 0, 'qcoc_compilations': 0}
    baseline = prep.transform(frames['support'])
    assert built[0][1][0].tobytes() == baseline.tobytes()
    for _, x, y in built[1:]:
        assert x[0, :, 1:].tobytes() == baseline[:, 1:].tobytes()
        assert not np.array_equal(x[0, :, 0], baseline[:, 0])
        assert torch.equal(y, built[0][2])
    assert np.all(built[2][1][0, :, 0].view(np.uint32) == 0)
    scores = pd.read_csv(output / 'evaluation_scores.csv', float_precision='round_trip')
    assert read_json(output / 'baseline_replay.json')['passed']
    for arm in ARMS:
        registration = read_json(output / arm / 'threshold.json')
        calibration = pd.read_csv(output / f'{arm}_calibration.csv', float_precision='round_trip')
        # With 99 normal rows, the strict rule is the observed maximum and zero C alarms.
        assert registration['rank'] == 99
        assert registration['threshold'] == calibration['score'].max()
        assert not calibration['predicted_attack'].astype(bool).any()
        assert registration['normal_scores_sha256'] == register_threshold(
            calibration['score'], .01, **registration['binding'])['normal_scores_sha256']
        assert registration['binding']['preprocessing_sha256'] == read_json(
            output / arm / 'deployment.json')['input_identity_sha256']
    assert len({read_json(output / arm / 'threshold.json')['binding']['preprocessing_sha256']
                for arm in ARMS}) == 3
    with monkeypatch.context() as no_load:
        no_load.setattr(torch, 'load', _forbidden)
        no_load.setattr(joblib, 'load', _forbidden)
        no_load.setattr(CSVPreprocessor, 'load', _forbidden)
        assert verify_scanning_result(output)['status'] == 'verified'
    # Move only the deployment tree to a new working directory; no reference P/C labels.
    cold_root = tmp_path / 'cold'
    cold_root.mkdir()
    shutil.copytree(output / 'shared', cold_root / 'shared')
    feature_csv = cold_root / 'queries.csv'
    frames['evaluation'].loc[:, spec['schema']['features']].to_csv(feature_csv, index=False)
    repo = Path(__file__).resolve().parents[1]
    vendor = repo / 'local/vendor/tabicl'
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONPATH'] = str(repo / 'src')
    for arm in ARMS:
        shutil.copytree(output / arm, cold_root / arm)
        prediction = cold_root / f'{arm}_prediction.csv'
        command = [sys.executable, '-m', 'ids.context_cache', '--tabicl-source', str(vendor),
                   'scanning-predict', '--deployment', str(cold_root / arm), '--input', str(feature_csv),
                   '--output', str(prediction)]
        completed = subprocess.run(command, cwd=cold_root, env=env, capture_output=True,
                                   text=True, encoding='utf-8', timeout=60)
        assert completed.returncode == 0, completed.stderr[-4000:]
        cold = pd.read_csv(prediction, float_precision='round_trip')
        np.testing.assert_array_equal(cold['score'], scores[f'{arm}_score'])
        np.testing.assert_array_equal(cold['predicted_attack'].astype(bool), scores[f'{arm}_predicted'])


@pytest.mark.parametrize('mismatch', ['score', 'threshold', 'decision'])
def test_baseline_failure_stops_before_intervention_builds(tiny_protocol, monkeypatch, tmp_path, mismatch):
    import ids.context_cache.scanning_workflow as workflow
    from ids.context_cache.tabicl_adapter import FullCacheAdapter

    config, _, _, _, _, old_e, old_registration = tiny_protocol
    builds = []
    original_build = FullCacheAdapter.build

    def count_build(self, *args, **kwargs):
        builds.append(self.provenance['scanning_arm'])
        return original_build(self, *args, **kwargs)

    monkeypatch.setattr(FullCacheAdapter, 'build', count_build)
    monkeypatch.setattr(CSVPreprocessor, 'fit', _forbidden)
    if mismatch == 'threshold':
        register = workflow.register_threshold

        def wrong_registration(*args, **kwargs):
            result = register(*args, **kwargs)
            result['threshold'] += 1.
            return result

        monkeypatch.setattr(workflow, 'register_threshold', wrong_registration)
    else:
        score = workflow._score_matrix
        calls = []

        def changed_score(*args, **kwargs):
            result = score(*args, **kwargs)
            calls.append(len(result))
            if len(calls) == 2:
                if mismatch == 'score':
                    result[1] += 1.
                else:
                    threshold = old_registration['threshold']
                    old_alarm = old_e[0] > threshold
                    result[0] = np.nextafter(threshold, -np.inf if old_alarm else np.inf)
            return result

        monkeypatch.setattr(workflow, '_score_matrix', changed_score)
    output = tmp_path / 'failed-comparison'
    with pytest.raises(ValueError, match='B0 baseline replay failed'):
        workflow.run_scanning(config, output)
    assert builds == [ARMS[0]]
    assert not (output / ARMS[1]).exists() and not (output / ARMS[2]).exists()
    assert read_json(output / 'state.json')['status'] == 'failed'
    assert (output / 'failure.txt').is_file()
    replay = read_json(output / 'baseline_replay.json')
    assert not replay['passed']
    if mismatch == 'score':
        assert replay['roles']['evaluation']['outside_tolerance'] > 0
    elif mismatch == 'threshold':
        assert not replay['threshold_passed']
    else:
        assert replay['roles']['evaluation']['outside_tolerance'] == 0
        assert replay['roles']['evaluation']['decision_mismatches'] > 0
        assert replay['threshold_passed']


def test_changed_mapping_rejected_before_deserialization(tiny_protocol, monkeypatch, tmp_path):
    from ids.context_cache.scanning_workflow import ScanningDetector, run_scanning, verify_scanning_result
    import joblib
    import torch

    config, *_ = tiny_protocol
    output = tmp_path / 'comparison'
    run_scanning(config, output)
    mapping_path = output / 'shared/source_port_mapping.npy'
    mapping = np.load(mapping_path, allow_pickle=False)
    mapping[0], mapping[1] = mapping[1], mapping[0]
    np.save(mapping_path, mapping, allow_pickle=False)
    monkeypatch.setattr(torch, 'load', _forbidden)
    monkeypatch.setattr(joblib, 'load', _forbidden)
    with pytest.raises(ValueError, match='hash mismatch'):
        ScanningDetector.load(output / ARMS[1])
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_scanning_result(output)


def test_artifact_verifier_rejects_identity_workload_and_swapped_arm(tiny_protocol, monkeypatch, tmp_path):
    from ids.context_cache.scanning_workflow import run_scanning, verify_scanning_result
    import joblib
    import torch

    config, *_ = tiny_protocol
    output = tmp_path / 'comparison'
    run_scanning(config, output)
    monkeypatch.setattr(torch, 'load', _forbidden)
    monkeypatch.setattr(joblib, 'load', _forbidden)
    monkeypatch.setattr(CSVPreprocessor, 'load', _forbidden)
    paths = [output / name for name in ('report.json', 'evaluation_scores.csv', 'evaluation_cohorts.csv',
                                        f'{ARMS[1]}/deployment.json')]
    original_bytes = {path: path.read_bytes() for path in paths}
    for mutation, message in (
        ('identity', 'evaluation identity, labels or families differ from frozen reference'),
        ('workload', 'workload differs from three-arm protocol'),
        ('swapped_arm', 'deployment differs from frozen arm protocol'),
    ):
        try:
            if mutation == 'identity':
                # Keep mutable score/cohort files internally consistent; the frozen
                # reference must still detect identity drift independently of hashes.
                report = read_json(output / 'report.json')
                for name in ('evaluation_scores.csv', 'evaluation_cohorts.csv'):
                    path = output / name
                    frame = pd.read_csv(path, float_precision='round_trip')
                    frame.loc[0, 'row_id'] = 'incorrectly_reassigned_row'
                    frame.to_csv(path, index=False)
                    report['files'][name] = sha256_file(path)
                write_json(output / 'report.json', report)
            elif mutation == 'workload':
                report = read_json(output / 'report.json')
                report['workload']['full_context_builds'] = 4
                write_json(output / 'report.json', report)
            else:
                shutil.copyfile(output / ARMS[0] / 'deployment.json', output / ARMS[1] / 'deployment.json')
            with pytest.raises(ValueError, match=message):
                verify_scanning_result(output)
        finally:
            for path, content in original_bytes.items():
                path.write_bytes(content)
