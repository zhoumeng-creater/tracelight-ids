"""Evidence orchestration invariants, using only small synthetic files."""
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile
import zlib

import pytest

SPEC = importlib.util.spec_from_file_location('ait_contract_runner',
    Path(__file__).resolve().parents[1] / 'scripts/ait_contract/run.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_existing_source_identity_rejects_same_size_mutation(tmp_path):
    pcap = tmp_path / 'source.pcap'
    content = b'synthetic fixed source'
    pcap.write_bytes(content)
    expected = {'bytes': len(content), 'crc32': zlib.crc32(content),
                'sha256': hashlib.sha256(content).hexdigest()}
    assert runner.verify_pcap(pcap, expected)['sha256'] == expected['sha256']
    pcap.write_bytes(b'x' + content[1:])
    with pytest.raises(ValueError, match='differs from R27'):
        runner.verify_pcap(pcap, expected)


def test_stage_reuses_verified_outputs_and_keeps_prior_attempt_on_mutation(tmp_path):
    calls = []

    def operation(out):
        calls.append(out)
        (out / 'conn.log').write_text('{"uid":"example"}\n')
        return {'connections': 1}

    first = runner.stage(tmp_path, 'strict', 'binding', operation)
    second = runner.stage(tmp_path, 'strict', 'binding', operation)
    assert second['reused'] and len(calls) == 1
    (Path(first['attempt']) / 'conn.log').write_text('changed')
    third = runner.stage(tmp_path, 'strict', 'binding', operation)
    assert len(calls) == 2 and not third['reused']
    assert third['attempt'] != first['attempt']
    assert Path(first['attempt']).exists()


def test_failed_attempt_not_promoted_or_discarded(tmp_path):
    def fail(out):
        (out / 'stderr.log').write_text('synthetic error')
        raise RuntimeError('no extraction')

    with pytest.raises(RuntimeError):
        runner.stage(tmp_path, 'strict', 'binding', fail)
    assert not (tmp_path / 'strict/complete.json').exists()
    failures = list((tmp_path / 'strict').glob('*/failure.json'))
    assert len(failures) == 1
    assert json.loads(failures[0].read_text(encoding='utf-8'))['status'] == 'failed'


def test_return_package_excludes_pcaps_and_weights(tmp_path):
    out = tmp_path / 'result'
    out.mkdir()
    (out / 'status.json').write_text('{}')
    (out / 'synthetic.pcap').write_bytes(b'not returned')
    (out / 'model.pt').write_bytes(b'not returned')
    archive = runner.pack(out)
    with zipfile.ZipFile(archive) as z:
        assert z.namelist() == ['status.json']
    assert Path(str(archive) + '.sha256').read_text().strip() == runner.digest(archive)


def test_rebuilt_export_invalidates_downstream_analysis(tmp_path):
    uid = ['first']

    def export(out):
        (out / 'conn.log').write_text(uid[0])
        return {}

    first = runner.stage(tmp_path, 'export', 'same-input', export)
    calls = []

    def analyze(out):
        calls.append(1)
        (out / 'analysis.csv').write_text(uid[0])
        return {}

    runner.stage(tmp_path, 'analysis', runner.analysis_binding('same-input', first), analyze)
    (Path(first['attempt']) / 'conn.log').unlink()
    uid[0] = 'new-replay-uid'
    rebuilt = runner.stage(tmp_path, 'export', 'same-input', export)
    final = runner.stage(tmp_path, 'analysis', runner.analysis_binding('same-input', rebuilt), analyze)
    assert len(calls) == 2 and not final['reused']
    assert (Path(final['attempt']) / 'analysis.csv').read_text() == uid[0]
