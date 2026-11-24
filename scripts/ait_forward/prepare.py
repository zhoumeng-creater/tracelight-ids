"""Package independently reviewed Harrison observations and evidence without scoring."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.ait_forward import run


INPUT_SCHEMA = "ait-harrison-reviewed-inputs-v1"
INPUT_NAMES = dict(zip(("rows", "normal_units", "events"), run.INPUTS, strict=True))


def _json_lines(stream):
    rows = [json.loads(line) for line in stream if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Inputs require one JSON object per line")
    return rows


def _read_bound(root, binding, *, rows=False):
    if not isinstance(binding, dict):
        raise ValueError("Each reviewed file binding must be a path and SHA-256 object")
    path, expected = binding.get("path"), binding.get("sha256")
    if not isinstance(path, str) or not path.strip() or Path(path).is_absolute() or Path(path).drive:
        raise ValueError("Each input and evidence path must be relative to the review manifest")
    if not isinstance(expected, str) or not run.SHA256.fullmatch(expected):
        raise ValueError(f"Each source must supply its reviewed SHA-256: {path}")
    with (root / path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected:
            raise ValueError(f"Reviewed source hash differs: {path}")
        receipt = {"path": path, "sha256": digest, "bytes": stream.tell()}
        if rows:
            stream.seek(0)
            if path.lower().endswith(".gz"):
                with gzip.GzipFile(fileobj=stream) as compressed:
                    return receipt, _json_lines(compressed)
            return receipt, _json_lines(stream)
    return receipt


def _bindings(manifest, spec):
    inputs = spec.get("inputs")
    if not isinstance(inputs, dict) or inputs.keys() != INPUT_NAMES.keys():
        raise ValueError("Bind the independently reviewed rows, normal_units and events inputs")
    evidence = spec.get("evidence_files")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Bind the original review and extraction evidence files used for admission")
    receipt, data = {}, {}
    for name in INPUT_NAMES:
        receipt[name], data[name] = _read_bound(manifest.parent, inputs[name], rows=True)
    evidence_receipts = [_read_bound(manifest.parent, item) for item in evidence]
    if len({item["path"] for item in evidence_receipts}) != len(evidence_receipts):
        raise ValueError("Evidence file bindings must be unique")
    return receipt, evidence_receipts, data


def _same_run_rows(rows):
    """Add the fixed dataset role if the reviewed exporter has not written it."""
    fields = {
        "allocation_group": run.GROUP, "source_run": "harrison",
        "dataset_role": "confirmation", "clock_basis": run.CLOCK,
    }
    projected = []
    for row in rows:
        if any(key in row and row[key] != value for key, value in fields.items()):
            raise ValueError("Reviewed observation declares a different source, role or clock")
        # The exporter supplies all eight values and the curator supplies every label.
        projected.append({**row, **fields})
    return projected


def _write_json(path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n",
    )


def _write_rows(path, rows):
    with path.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.write(line.encode("utf-8") + b"\n")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(manifest, output, *, diagnostic=False):
    """Publish one complete model-free package in a new directory."""
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Prepare requires a new output directory")
    payload = manifest.read_bytes()
    spec = json.loads(payload.decode("utf-8-sig"))
    if (
        spec.get("schema") != INPUT_SCHEMA or spec.get("source_group") != run.GROUP
        or spec.get("clock_basis") != run.CLOCK
    ):
        raise ValueError("Provide a reviewed whole-run Harrison input manifest with the offline export clock")

    # Validate admission metadata using the same receipt schema as the scoring entry point.
    qualification = {
        key: spec[key] for key in (
            "source_group", "clock_basis", "confirmation_window", "normal_evidence", "clock_evidence", "targets",
        )
    }
    qualification.update(schema=run.CONFIRMATION_SCHEMA, status="qualified")
    qualified = [kind for kind, item in qualification["targets"].items() if item.get("status") == "qualified"]
    if not qualified and not diagnostic:
        raise ValueError("No target qualifies; use --diagnostic to record reviewed input coverage only")
    bound_inputs, evidence, data = _bindings(manifest, spec)
    data["rows"] = _same_run_rows(data["rows"])

    lineage = {
        "review_manifest": {"name": manifest.name, "sha256": hashlib.sha256(payload).hexdigest()},
        "inputs": bound_inputs,
        "evidence_files": evidence,
        "prepare_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    coverage = {
        "rows": len(data["rows"]),
        "labels": dict(Counter(str(row.get("label")) for row in data["rows"])),
        "positive_flows_by_target": dict(Counter(
            str(row.get("target_kind")) for row in data["rows"] if row.get("label") == 1
        )),
        "normal_units": len(data["normal_units"]),
        "events_by_target": dict(Counter(str(event.get("kind")) for event in data["events"])),
        "model_calls": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".ait-forward-prepare-", dir=output.parent) as temporary:
        stage = Path(temporary) / "package"
        stage.mkdir()
        if diagnostic:
            result = {
                **qualification,
                "schema": "ait-harrison-preparation-diagnostic-v1",
                "status": "diagnostic_only",
                "coverage": coverage,
                "lineage": lineage,
            }
            name = "diagnostic.json"
        else:
            files = {
                filename: _write_rows(stage / filename, data[key])
                for key, filename in INPUT_NAMES.items()
            }
            result = {**qualification, "files": files, "coverage": coverage, "lineage": lineage}
            name = "qualification.json"
        _write_json(stage / name, result)

        if not diagnostic:
            accepted, _ = run.load_qualification(stage / name)
            run.validate_confirmation_rows(
                data["rows"], data["normal_units"], data["events"], accepted,
            )
        if output.exists():
            raise FileExistsError("Prepare output was created while packaging")
        stage.rename(output)
    return output / name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    print(prepare(args.manifest, args.output, diagnostic=args.diagnostic))


if __name__ == "__main__":
    main()
