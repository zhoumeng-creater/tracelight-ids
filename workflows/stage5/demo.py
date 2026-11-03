"""Build an offline Stage 4 validation replay, or score a new feature CSV on CPU.

Run from the repository with its installed IDS environment:
    python -m workflows.stage5.demo --help
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from ids.common import read_json, sha256_file  # noqa: E402
from workflows.stage5.core import _artifact, _score  # noqa: E402


SEAL = "59b3c7909f2a5242d947b852e32e292b93a23c58287548bf6ef878d8c8cba8fa"
PAIRS = {"unsw": (("05-mlp-s17", "小 MLP · 全字段"), ("00-rf-s17", "随机森林 · 全字段")),
         "cicids": (("04-mlp-s17", "小 MLP · MI24"), ("03-mlp-s17", "小 MLP · 全字段"))}
TEMPLATE = Path(__file__).with_name("demo.html")
MAX_ROWS = 10000
MAX_BYTES = 24 * 1024 * 1024


def artifacts(experiment, dataset):
    seal = read_json(_artifact(experiment, "STAGE4_COMPLETE.json", SEAL))["files"]

    def bound(rel):
        return _artifact(experiment, rel, seal[rel])

    manifest = read_json(bound(f"{dataset}/manifest.json"))
    study = read_json(bound(f"reports/{dataset}/study.json"))
    pairs = []
    for run_id, title in PAIRS[dataset]:
        group, run = next((g, r) for g in study["groups"] for r in g["runs"] if r["run_id"] == run_id)
        run_setup = [s for s in seal if s.startswith(f"{dataset}/runs/") and s.endswith(f"/{run_id}/run.json")]
        if len(run_setup) != 1 or seal[run_setup[0]] != run["run_manifest_sha256"] or run["seed"] != 17:
            raise ValueError("Development run binding differs")
        root = run_setup[0].rsplit("/", 1)[0]
        record = read_json(bound(run_setup[0]))
        threshold = read_json(bound(f"{root}/validation_threshold.json"))
        model_path = f"{root}/model.joblib"
        if (record["files"]["model.joblib"] != run["model_sha256"] or seal[model_path] != run["model_sha256"]
                or threshold["source_split"] != "validation" or threshold["threshold"] != run["threshold"]):
            raise ValueError("Threshold or model identity differs from the accepted run")
        pairs.append({"run_id": run_id, "title": title, "seed": 17, "model": group["spec"]["model"],
                      "selected_features": run["selected_features"], "model_path": model_path,
                      "model_sha256": run["model_sha256"], "threshold": float(threshold["threshold"]),
                      "directory": root, "bytes": run["metrics"]["model_file_bytes"]})
    return manifest, pairs, bound


def _decisions(scores, threshold):
    values = np.asarray(scores, dtype=np.float64)
    if (values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()
            or not np.isfinite(threshold)):
        raise ValueError("Invalid model scores or threshold")
    return (values >= threshold).astype(int)


def paired_predictions(paths, models):
    frames = [pd.read_parquet(p) for p in paths]
    ids = ["sample_id", "true_label", "attack_category"]
    if not frames[0][ids].equals(frames[1][ids]) or frames[0]["sample_id"].duplicated().any():
        raise ValueError("Validation pairs differ in identity or truth")
    evidence = frames[0][ids].copy()
    for i, (frame, model) in enumerate(zip(frames, models)):
        t = model["threshold"]
        if (not frame["split"].eq("validation").all() or not frame["run_id"].eq(model["run_id"]).all()
                or not frame["threshold"].eq(t).all()
                or not np.array_equal(_decisions(frame["score"], t), frame["predicted_label"])):
            raise ValueError("Validation row decisions do not match their saved threshold")
        evidence[f"s{i}"] = frame["score"].astype("float64")
        evidence[f"p{i}"] = frame["predicted_label"].astype(int)
    return evidence


def _base(dataset, manifest, models, mode):
    return {"dataset": dataset, "mode": mode, "seed": 17, "schema": manifest["features"],
            "kind": manifest["data_kind"], "models": models, "stage4_complete_sha256": SEAL}


def _features(row, columns):
    fields = {}
    for key in columns:
        value = row[key]
        fields[key] = None if pd.isna(value) else value.item() if isinstance(value, np.generic) else value
    return fields


def _summary(frame):
    result = []
    for a, b in ((1, 1), (1, 0), (0, 1), (0, 0)):
        subset = frame.loc[(frame["p0"] == a) & (frame["p1"] == b)]
        result.append({"key": f"{a}{b}", "n": len(subset),
                       "attacks": int(subset["true_label"].sum()) if "true_label" in subset else None})
    return result


def build_dataset(experiment, dataset, validation=None):
    manifest, models, bound = artifacts(experiment, dataset)
    full = paired_predictions([bound(m["directory"] + "/validation_predictions.parquet") for m in models], models)
    examples = read_json(bound(models[0]["directory"] + "/examples.json"))["events"]
    by_id = {e["sample_id"]: e["features"] for e in examples}
    # Fixed, inspectable selection: first two saved rows per truth/family/decision pair.
    selected = full.groupby(["true_label", "attack_category", "p0", "p1"], sort=False).head(2)
    payload = _base(dataset, manifest, models, "validation")
    raw = None
    if validation is not None:
        validation = Path(validation)
        if validation.name != "validation.parquet":
            raise ValueError("Only validation.parquet can supply fields for validation replay")
        columns = manifest["features"]
        raw = pd.read_parquet(validation, columns=["sample_id", "label", "attack_cat", *columns])
        if (not np.array_equal(raw["sample_id"], full["sample_id"])
                or not np.array_equal(raw["label"], full["true_label"])
                or not np.array_equal(raw["attack_cat"].astype(str), full["attack_category"].astype(str))):
            raise ValueError("Supplied validation fields do not match the archived prediction identities")
        raw = raw.set_index("sample_id")
        for sample_id, fields in by_id.items():
            if _features(raw.loc[sample_id], columns) != fields:
                raise ValueError("Supplied field values contradict sealed validation examples")
        payload["local_validation_sha256"] = sha256_file(validation)
    events = selected.to_dict("records")
    for row in events:
        row["fields"] = (_features(raw.loc[row["sample_id"]], manifest["features"])
                         if raw is not None else by_id.get(row["sample_id"]))
    if raw is None:
        known = {row["sample_id"] for row in events}
        # Expose archived raw examples even when sorted source IDs cover just one family.
        extra = full.loc[full["sample_id"].isin(by_id) & ~full["sample_id"].isin(known)].head(8)
        for row in extra.to_dict("records"):
            row["fields"] = by_id[row["sample_id"]]
            events.append(row)
    return {**payload, "count": len(full), "summary": _summary(full), "events": events}


def read_input(path, dataset, manifest):
    if path.suffix.lower() != ".csv" or path.stat().st_size > MAX_BYTES:
        raise ValueError("Use a UTF-8 feature CSV of at most 24 MiB")
    data = path.read_bytes()
    reader = csv.reader(io.StringIO(data.decode("utf-8-sig")), strict=True)
    columns = next(reader, [])
    expected = manifest["features"]
    if len(columns) != len(set(columns)) or set(columns) != set(expected):
        raise ValueError("CSV needs exactly the original feature columns, without IDs, labels or scores")
    rows = []
    for row in reader:
        if len(row) != len(columns) or len(rows) >= MAX_ROWS:
            raise ValueError("Malformed CSV row width or over 10,000 records")
        rows.append(row)
    if not rows:
        raise ValueError("CSV contains no traffic records")
    if dataset == "unsw":
        from ids.data import load_unsw_csv
        return load_unsw_csv(path, require_labels=False).loc[:, expected]
    frame = pd.DataFrame(rows, columns=columns).loc[:, expected].replace("", np.nan)
    for k in expected:
        nums = pd.to_numeric(frame[k], errors="raise").astype("float64")
        if np.isinf(nums).any():
            raise ValueError(f"Non-finite input: {k}")
        if k in manifest["categorical_features"]:
            valid = nums.dropna()
            if ((valid % 1 != 0) | (valid < (0 if k == "Protocol" else -1)) | (valid > 255)).any():
                raise ValueError(f"Invalid CIC categorical code: {k}")
            frame[k] = nums.map(lambda x: None if pd.isna(x) else str(int(x)))
        else:
            frame[k] = nums
    return frame


def predict_dataset(experiment, dataset, source):
    manifest, models, _ = artifacts(experiment, dataset)
    frame = read_input(source, dataset, manifest)
    evidence = pd.DataFrame({"sample_id": [f"CSV-{i + 1:05d}" for i in range(len(frame))]})
    for i, model in enumerate(models):
        scores = _score(experiment, model, frame, 1)
        evidence[f"s{i}"] = scores
        evidence[f"p{i}"] = _decisions(scores, model["threshold"])
    events = evidence.to_dict("records")
    for event, row in zip(events, frame.to_dict("records")):
        event["fields"] = _features(row, manifest["features"])
    return {**_base(dataset, manifest, models, "prediction"), "count": len(frame), "events": events,
            "summary": _summary(evidence), "input_sha256": sha256_file(source)}


def render(datasets, output, experiment):
    output = Path(output).resolve()
    if output.suffix.lower() != ".html" or output.is_relative_to(experiment.resolve()) or output.exists():
        raise ValueError("Choose a new .html file outside the sealed experiment")
    encoded = json.dumps(datasets, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    html = TEMPLATE.read_text(encoding="utf-8").replace("<!--DATA-->", encoded)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        try:
            f.write(html)
        except OSError:
            f.close()
            output.unlink(missing_ok=True)
            raise
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build", help="Create a standalone validation replay from sealed predictions")
    pred = sub.add_parser("predict", help="Score an unlabelled feature CSV with both real development models")
    for cmd in (build, pred):
        cmd.add_argument("--experiment", type=Path, required=True, help="Trusted, accepted Stage 4 experiment")
        cmd.add_argument("--output", type=Path, required=True, help="New standalone .html outside the experiment")
    for name in PAIRS:
        build.add_argument(f"--{name}-validation", type=Path, help="Optional corresponding validation.parquet for full fields")
    pred.add_argument("--dataset", choices=list(PAIRS), required=True)
    pred.add_argument("--input", type=Path, required=True, help="UTF-8 CSV with the exact raw feature schema")
    args = parser.parse_args(argv)
    root = args.experiment.resolve()
    try:
        if args.action == "build":
            results = [build_dataset(root, d, getattr(args, f"{d}_validation")) for d in PAIRS]
        else:
            results = [predict_dataset(root, args.dataset, args.input)]
        print(render(results, args.output, root))
    except (OSError, ValueError, KeyError, csv.Error) as e:
        parser.exit(2, f"error: {e}\n")


if __name__ == "__main__":
    main()
