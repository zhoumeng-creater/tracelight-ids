"""One existing-PCAP evidence batch: two Zeek exports, packet audit, no models."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import zipfile
import zlib

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / 'configs/ait-flow-contract-20261118'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def locate_pcap(explicit, member):
    if explicit:
        path = explicit.resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Existing PCAP not found: {path}')
        return path
    roots = [REPO]
    result = subprocess.run(['git', '-C', str(REPO), 'worktree', 'list', '--porcelain'],
                            capture_output=True, text=True, encoding='utf-8', check=True)
    roots.extend(Path(line[9:]) for line in result.stdout.splitlines() if line.startswith('worktree '))
    roots.extend([REPO.parent / 'tracelight-risk-audit', REPO.parent.parent / 'tracelight-risk-audit'])
    for root in dict.fromkeys(roots):
        path = root / 'data/raw/ait-lds/source-audit-20261115' / member
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError('R27 PCAP was not found in registered worktrees. Supply -Pcap / --pcap; no download is started.')


def verify_pcap(path, expected):
    sha, crc, total = hashlib.sha256(), 0, 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
            crc = zlib.crc32(block, crc)
            total += len(block)
    actual = {'path': str(path), 'bytes': total, 'crc32': crc, 'sha256': sha.hexdigest()}
    if any(actual[k] != expected[k] for k in ('bytes', 'crc32', 'sha256')):
        raise ValueError(f'Existing PCAP differs from R27 pinned member: {actual}')
    return actual


def stage(root, name, binding, operation):
    """Reuse only matching, hash-verified successful outputs; retain failed attempts."""
    base = root / name
    receipt = base / 'complete.json'
    if receipt.exists():
        saved = read(receipt)
        if saved['binding'] == binding and all((base / p).is_file() and digest(base / p) == h
                                               for p, h in saved['files'].items()):
            return {**saved, 'reused': True}
    attempt = base / datetime.now(timezone.utc).strftime('attempt-%Y%m%dT%H%M%S%fZ')
    attempt.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    try:
        result = operation(attempt)
    except Exception:
        write(attempt / 'failure.json', {'status': 'failed', 'traceback': traceback.format_exc(),
                                         'wall_seconds': time.perf_counter() - began})
        raise
    files = {str(p.relative_to(base)): digest(p) for p in attempt.rglob('*') if p.is_file()}
    record = {'binding': binding, 'files': files, 'attempt': str(attempt.resolve()),
              'result': result, 'wall_seconds': time.perf_counter() - began, 'reused': False}
    write(receipt, record)
    return record


def pack(output):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = output.parent / f'ait-flow-contract-results-{stamp}.zip'
    files = [p for p in output.rglob('*') if p.is_file() and p.suffix.lower() in
             {'.json', '.csv', '.log', '.txt', '.xml'}]
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files):
            z.write(path, path.relative_to(output).as_posix())
    Path(str(archive) + '.sha256').write_text(digest(archive) + '\n', encoding='ascii')
    print(f'Return package: {archive}', flush=True)
    return archive


def analysis_binding(binding, export_result):
    return binding + ':' + hashlib.sha256(json.dumps(export_result['files'], sort_keys=True).encode()).hexdigest()


def run(args):
    # Imports do not create model environments or load model weights.
    from analyze import analyze
    from packet_diagnostics import diagnose
    from compare_exports import compare_exports
    from zeek_runtime import prepare_runtime, export_pcap, self_test

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    write(root / 'status.json', {'status': 'started', 'effect_experiment': False})
    contract = read(CONFIG / 'contract.json')
    pcap = locate_pcap(args.pcap, contract['pcap']['member'])
    identity = verify_pcap(pcap, contract['pcap'])
    source_files = (list(Path(__file__).parent.glob('*.py')) + list(Path(__file__).parent.glob('*.zeek'))
                    + list(Path(__file__).parent.glob('*.json')))
    source_files += list(CONFIG.glob('*'))
    evidence = REPO / 'docs/research_alignment_20261115/risk_transfer/source/ait-small-evidence.json'
    source_files += [evidence, REPO / 'scripts/prepare_ait_packet_evidence.py']
    sources = {p.relative_to(REPO).as_posix(): digest(p) for p in source_files if p.is_file()}
    binding = hashlib.sha256(json.dumps({'sources': sources, 'pcap': identity['sha256']}, sort_keys=True).encode()).hexdigest()
    write(root / 'input-receipt.json', {'pcap': identity, 'sources': sources, 'binding': binding,
                                      'contract': contract, 'logical_cpus': os.cpu_count()})
    runtime = prepare_runtime(REPO, runtime=args.runtime, distro=args.distro)
    write(root / 'runtime.json', runtime)
    runtime_binding = hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest()
    binding += ':' + runtime_binding
    stage(root, 'runtime-self-test', binding, lambda out: self_test(runtime, out))
    tasks = {'packet-diagnostics': lambda out: diagnose(pcap, out)}
    for variant in contract['variants']:
        tasks['zeek-' + variant] = lambda out, variant=variant: export_pcap(pcap, out, variant, runtime)
    results, errors = {}, {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(stage, root, name, binding, fn): name for name, fn in tasks.items()}
        for future in as_completed(pending):
            name = pending[future]
            try:
                results[name] = future.result()
                print(f'{name}: complete; reused={results[name]["reused"]}', flush=True)
            except Exception as exc:
                errors[name] = f'{type(exc).__name__}: {exc}'
    # Analyze every successful extraction, even if the other arm failed.
    analysis = {}
    for variant in contract['variants']:
        key = 'zeek-' + variant
        if key not in results:
            continue
        try:
            analysis[variant] = stage(root, 'analysis-' + variant, analysis_binding(binding, results[key]),
                lambda out, variant=variant, key=key: analyze(Path(results[key]['attempt']), evidence, out,
                                                             variant, CONFIG / 'anchors.json'))
        except Exception as exc:
            errors['analysis-' + variant] = f'{type(exc).__name__}: {exc}'
    paired = None
    if all('zeek-' + v in results for v in contract['variants']):
        try:
            pair_binding = analysis_binding(analysis_binding(binding, results['zeek-strict']),
                                            results['zeek-ignore_checksums'])
            paired = stage(root, 'paired-export-comparison', pair_binding,
                lambda out: compare_exports(Path(results['zeek-strict']['attempt']),
                                            Path(results['zeek-ignore_checksums']['attempt']), out))
        except Exception as exc:
            errors['paired-export-comparison'] = f'{type(exc).__name__}: {exc}'
    write(root / 'comparison.json', {
        'status': 'incomplete' if errors else 'data_contract_batch_complete',
        'variants': {k: v['result'] for k, v in analysis.items()},
        'paired_export_comparison': paired['result'] if paired else None,
        'variant_selection': 'none', 'model_fits': 0, 'model_forward_calls': 0,
        'threshold_updates': 0, 'gpu_jobs': 0, 'max_parallel_jobs': 3,
        'errors': errors, 'elapsed_seconds': time.perf_counter() - started,
        'next_step': 'Review evidence, unknown cohorts and field/time contracts; no automatic effect experiment.'})
    write(root / 'status.json', {'status': 'incomplete' if errors else 'data_contract_batch_complete',
                               'errors': errors, 'effect_experiment': False})
    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pcap', type=Path)
    parser.add_argument('--output', type=Path, default=REPO / 'local/ait-flow-contract')
    parser.add_argument('--runtime', choices=['auto', 'wsl', 'native'], default='auto')
    parser.add_argument('--distro', default='Ubuntu')
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception:
        write(args.output / 'status.json', {'status': 'incomplete', 'traceback': traceback.format_exc(),
                                          'effect_experiment': False})
        result = 1
    pack(args.output.resolve())
    return result


if __name__ == '__main__':
    raise SystemExit(main())
