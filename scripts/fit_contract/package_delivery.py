"""Copy the reviewable data release; leave PCAPs and runtime binaries at source."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    selected = set()
    for name in ['README.md', 'dataset_review.json', 'review_dataset.py']:
        selected.add(source / name)
    for directory in ['fit_contract', 'calibration', 'dataset']:
        selected.update(p for p in (source / directory).rglob('*') if p.is_file() and
                        p.suffix in ('.json', '.gz', '.py', '.md', '.txt') and '__pycache__' not in p.parts)
    for directory in ['ait_new_runs', 'ait_recovery']:
        selected.update(p for p in (source / directory).iterdir() if p.is_file() and
                        p.suffix in ('.json', '.py', '.md'))
    for directory in ['ait_new_runs/metadata', 'ait_new_runs/network_mapping',
                      'ait_new_runs/pcap_receipts', 'ait_new_runs/actor_evidence',
                      'ait_recovery/declared_length_representation']:
        if (source / directory).exists():
            selected.update(p for p in (source / directory).rglob('*') if p.is_file() and
                            p.suffix in ('.json', '.jsonl', '.gz', '.md') and 'zeek-ignore_checksums' not in p.parts)
    for p in (source / 'ait_extracted').rglob('*'):
        if p.is_file() and p.name in ('EXTRACTION_CONTRACT.md', 'extraction_receipt.json',
              'normalization_receipt.json', 'summary.json', 'schema_example.json', 'export-receipt.json',
              'runtime_packaging_receipt.json', 'vendored_runtime_receipt.json', 'extraction_index.json',
              'capture_truncation_records.jsonl.gz', 'record_changes.jsonl.gz', 'pcap_selfcheck.json'):
            selected.add(p)
    receipt = []
    for path in sorted(selected):
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source_hash = sha(path)
        assert source_hash == sha(target)
        receipt.append({'path': relative.as_posix(), 'sha256': source_hash, 'bytes': path.stat().st_size})
    (destination / '.gitattributes').write_text(
        '* -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n',
        encoding='utf-8', newline='\n')
    output = {'source_directory': str(source), 'files': receipt,
              'file_count': len(receipt), 'total_bytes': sum(r['bytes'] for r in receipt),
              'raw_PCAPs_copied': False, 'model_artifacts_created': False,
              'historical_v1_directory': 'docs/unified_events_20261119'}
    (destination / 'delivery_receipt.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n',
                                                       encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in output.items() if k != 'files'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
