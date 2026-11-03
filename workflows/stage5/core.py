"""Frozen confirmatory evaluation using the original Stage 4 inference engine.

All model files must be bound to the accepted Stage 4 seal before unpickling.
The final output is a separate directory. Thresholds for *all* candidates must
be durably committed before a test partition is opened.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import os
import tempfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ids.common import canonical_hash, code_fingerprint, environment_snapshot, read_json, sha256_file, utc_now, write_json
from ids.data import load_partition
from ids.evaluation import calibrate_threshold, evaluate_binary
from ids.models import load_bundle

HERE = Path(__file__).resolve().parent
DATASETS = ("unsw", "cicids")
PLAN = "STAGE5_PLAN.json"
BARRIER = "THRESHOLDS_LOCK.json"
START = "TEST_STARTED.json"
DONE = "EVALUATION_COMPLETE.json"
CHECKSUM = "EVALUATION_COMPLETE.sha256"
ROLES = {"unsw": {("mlp", "all"): "candidate", ("rf", "all"): "reference"},
         "cicids": {("mlp", "mi"): "candidate", ("mlp", "all"): "reference"}}


@contextmanager
def _exclusive(output):
    """OS-backed single writer. A killed process releases its lock automatically."""
    path = output / ".stage5.lock"
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another evaluation uses this output directory") from exc
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another evaluation uses this output directory") from exc
        yield


def _artifact(root, relative, expected):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"Artifact checksum/path mismatch: {relative}")
    return path


def _record(root, name, value):
    """Verify an existing immutable record or atomically publish it once under the lock."""
    path = root / name
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen record differs: {name}")
    else:
        write_json(path, value)
    return sha256_file(path)


def _hash_record(payload):
    return {**payload, "record_sha256": canonical_hash(payload)}


def _read_record(path):
    obj = read_json(path)
    digest = obj.pop("record_sha256", None)
    if canonical_hash(obj) != digest:
        raise ValueError(f"Invalid record checksum: {path.name}")
    return obj


def _evaluator_source():
    files = {name: sha256_file(HERE / name) for name in ("core.py", "run.py", "reporting.py")}
    return {"files": files, "sha256": canonical_hash(files)}


def _cpu_environment():
    env = environment_snapshot()
    env.pop("recorded_at", None)
    return env


def _deployment(group, member):
    seed = member["seed"]
    values = [b for b in group["cpu_benchmarks"] if seed in b["seeds"]]
    rows = [next(r["metrics"] for r in b["runs"] if r["seed"] == seed) for b in values]
    if not rows or any(b["spec"]["threads"] != 1 for b in values):
        raise ValueError("Missing comparable Stage 4 CPU measurements")
    return {"model_file_bytes": member["metrics"]["model_file_bytes"],
            "raw_fields": len(member["selected_features"]), "cpu_rounds": len(rows),
            "cpu_p95_ms": float(np.median([r["p95_ms"] for r in rows])),
            "cpu_rows_per_s": float(np.median([r["throughput_rows_per_second"] for r in rows])),
            "source": "stage4_validation_cpu"}


def _score(experiment, item, frame, threads):
    # Recheck this exact trusted byte identity immediately before deserialization.
    path = _artifact(experiment, item["model_path"], item["model_sha256"])
    bundle = load_bundle(path, device="cpu")
    if (bundle.model_name != item["model"] or bundle.seed != item["seed"]
            or bundle.selected_features != item["selected_features"]):
        raise ValueError("Serialized model is inconsistent with the frozen candidate")
    bundle.num_threads = threads
    if bundle.model_name == "rf":
        bundle.estimator.n_jobs = threads
    if bundle.model_name == "mlp":
        import torch
        torch.set_num_threads(threads)
    with threadpool_limits(limits=threads):
        return np.asarray(bundle.predict_scores(frame), dtype=np.float64)


def _threshold_file(item):
    return f"calibration/{item['dataset']}/{item['run_id']}.json"


def _test_dir(output, item):
    return output / "tests" / item["dataset"] / item["run_id"]


def _validate_calibration(record, binding, partitions):
    if any(record.get(key) != val for key, val in binding.items()):
        raise ValueError("Calibration record identity changed")
    n, fp = record["n_normal"], record["false_positives"]
    if (not 0 < n <= partitions["calibration"]["n_rows"] or not 0 <= fp <= int(record["max_fpr"] * n)
            or record["allowed_false_positives"] != int(record["max_fpr"] * n)
            or record["achieved_fpr"] != fp / n
            or record["threshold"] != np.nextafter(record["next_disallowed_normal_score"], np.inf)):
        raise ValueError("Invalid saved operating point")


def _calibrate(experiment, data, output, plan, plan_hash, thresholds_locked):
    thresholds = {}
    for dataset in DATASETS:
        frame = normal_ids = None
        for item in (r for r in plan["candidates"] if r["dataset"] == dataset):
            path = output / _threshold_file(item)
            normal_path = path.with_suffix(".normal.parquet")
            action = "verify" if path.exists() else "score normal records"
            print(f"Calibration | {dataset} | {item['run_id']} | {action}", flush=True)
            binding = {"dataset": dataset, "run_id": item["run_id"], "seed": item["seed"],
                       "source_split": "calibration", "plan_sha256": plan_hash,
                       "model_sha256": item["model_sha256"], "dataset_manifest_sha256": item["dataset_manifest_sha256"],
                       "calibration_partition_sha256": plan["partitions"][dataset]["calibration"]["sha256"],
                       "max_fpr": plan["max_fpr"]}
            if not path.exists():
                if thresholds_locked:
                    raise ValueError("A calibration record is missing after the thresholds were locked")
                if frame is None:
                    partition = load_partition(data[dataset], "calibration")
                    frame = partition.loc[partition["label"].eq(0)].reset_index(drop=True)
                    del partition
                    if frame.empty:
                        raise ValueError("Calibration requires normal records")
                scores = _score(experiment, item, frame, plan["threads"])
                normals = pd.DataFrame({"sample_id": frame["sample_id"].to_numpy(),
                                        "score": np.asarray(scores, dtype=np.float64)})
                path.parent.mkdir(parents=True, exist_ok=True)
                if normal_path.exists():
                    if not pd.read_parquet(normal_path).equals(normals):
                        raise ValueError("Uncommitted normal-score artifact differs from frozen inference")
                else:
                    tmp = normal_path.with_suffix(".parquet.tmp")
                    normals.to_parquet(tmp, index=False)
                    tmp.replace(normal_path)
                binding["normal_scores_sha256"] = sha256_file(normal_path)
                record = {**calibrate_threshold(frame["label"], scores, plan["max_fpr"]), **binding}
                write_json(path, _hash_record(record))
            elif not normal_path.is_file():
                raise ValueError("Missing committed normal calibration scores")
            else:
                binding["normal_scores_sha256"] = sha256_file(normal_path)
            record = _read_record(path)
            _validate_calibration(record, binding, plan["partitions"][dataset])
            normals = pd.read_parquet(normal_path)
            if (list(normals) != ["sample_id", "score"] or normals["score"].dtype != np.dtype("float64")
                    or normals["sample_id"].isna().any() or not normals["sample_id"].is_unique):
                raise ValueError("Invalid saved normal calibration scores")
            if normal_ids is None:
                normal_ids = normals["sample_id"].copy()
            elif not normal_ids.equals(normals["sample_id"]):
                raise ValueError("Candidates have different normal calibration sample identities")
            checked = calibrate_threshold(np.zeros(len(normals), dtype=np.uint8), normals["score"], plan["max_fpr"])
            if any(record.get(k) != v for k, v in checked.items()):
                raise ValueError("Threshold cannot be reproduced from saved normal scores")
            thresholds[_threshold_file(item)] = {"sha256": sha256_file(path), "threshold": record["threshold"],
                "normal_scores_file": normal_path.relative_to(output).as_posix(),
                "normal_scores_sha256": binding["normal_scores_sha256"]}
    return {"schema_version": 1, "plan_sha256": plan_hash, "n_candidates": len(thresholds), "thresholds": thresholds}


def _source_feature_hashes(data_dir, partitions):
    """The original UNSW training CSV is the union of three internal partitions."""
    groups = set()
    for split in ("train", "validation", "calibration"):
        record = partitions[split]
        if record["file"] != split + ".parquet":
            raise ValueError("Unexpected UNSW source partition file")
        path = _artifact(data_dir, record["file"], record["sha256"])
        frame = pd.read_parquet(path, columns=["feature_hash"])
        if len(frame) != record["n_rows"] or frame["feature_hash"].isna().any():
            raise ValueError("UNSW source identity column is incomplete")
        groups.update(frame["feature_hash"].unique())
    return groups


def _result_files():
    return ("metrics.json", "predictions.parquet")


def _verified_result(path, binding):
    record = _read_record(path / "completion.json")
    if any(record.get(k) != v for k, v in binding.items()) or set(record["files"]) != set(_result_files()):
        raise ValueError("Test result binding or file inventory changed")
    if {p.name for p in path.iterdir()} != {*_result_files(), "completion.json"}:
        raise ValueError("Unexpected files in completed test result")
    for name, digest in record["files"].items():
        _artifact(path, name, digest)
    return record


def _test_result(experiment, path, item, frame, threshold, nonoverlap, plan, binding):
    scores = _score(experiment, item, frame, plan["threads"])
    full = evaluate_binary(frame["label"], scores, threshold, frame["attack_cat"])
    subset = None
    if nonoverlap is not None and nonoverlap.any():
        subset = evaluate_binary(frame.loc[nonoverlap, "label"], scores[nonoverlap], threshold,
                                 frame.loc[nonoverlap, "attack_cat"])
    metrics = {**binding, "seed": item["seed"], "model": item["model"], "condition": item["condition"],
               "threshold": threshold, "split": "test", "threshold_source": "calibration",
               "target_fpr": plan["max_fpr"], "full": full, "nonoverlap": subset,
               "n_nonoverlap": int(nonoverlap.sum()) if nonoverlap is not None else None,
               "n_overlap": int((~nonoverlap).sum()) if nonoverlap is not None else None,
               "nonoverlap_status": ("not_applicable" if nonoverlap is None else
                                     "reported" if nonoverlap.any() else "empty_subset"),
               "deployment": item["deployment"]}
    columns = ["sample_id", "feature_hash", "label", "attack_cat"]
    if item["dataset"] == "cicids":
        columns.extend(["group_hash", "session_hash"])
    predictions = frame[columns].rename(columns={"label": "true_label", "attack_cat": "attack_category"}).copy()
    predictions["score"] = scores
    predictions["threshold"] = threshold
    predictions["predicted_label"] = (scores >= threshold).astype(np.uint8)
    predictions["split"] = "test"
    predictions["run_id"] = item["run_id"]
    if nonoverlap is not None:
        predictions["nonoverlap"] = nonoverlap
    # Commit a complete result directory on the same filesystem. Incomplete
    # writes never appear as published runs. A crash can leave scratch files.
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.parents[2] / ".scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch, prefix="result-") as tmp:
        staged = Path(tmp) / "result"
        staged.mkdir()
        predictions.to_parquet(staged / "predictions.parquet", index=False)
        write_json(staged / "metrics.json", metrics)
        files = {n: sha256_file(staged / n) for n in _result_files()}
        write_json(staged / "completion.json", _hash_record({**binding, "files": files}))
        staged.rename(path)
    return _verified_result(path, binding)


def _result_binding(item, plan, thresholds, barrier_hash, start_hash):
    dataset = item["dataset"]
    entry = thresholds["thresholds"][_threshold_file(item)]
    return {"dataset": dataset, "run_id": item["run_id"], "model_sha256": item["model_sha256"],
            "run_manifest_sha256": item["run_manifest_sha256"],
            "dataset_manifest_sha256": item["dataset_manifest_sha256"], "plan_sha256": thresholds["plan_sha256"],
            "thresholds_lock_sha256": barrier_hash, "test_started_sha256": start_hash,
            "threshold_record_sha256": entry["sha256"], "test_partition_sha256": plan["partitions"][dataset]["test"]["sha256"]}


def _run_tests(experiment, data, output, plan, thresholds, barrier_hash, start_hash):
    files, committed = {}, {}
    # Check every previously committed result before reopening any test partition.
    for item in plan["candidates"]:
        path = _test_dir(output, item)
        if path.exists():
            binding = _result_binding(item, plan, thresholds, barrier_hash, start_hash)
            committed[path] = _verified_result(path, binding)
    for dataset in DATASETS:
        frame = nonoverlap = None
        for item in (r for r in plan["candidates"] if r["dataset"] == dataset):
            path = _test_dir(output, item)
            entry = thresholds["thresholds"][_threshold_file(item)]
            binding = _result_binding(item, plan, thresholds, barrier_hash, start_hash)
            action = "reuse verified result" if path in committed else "score fixed test split"
            print(f"Frozen test | {dataset} | {item['run_id']} | {action}", flush=True)
            if path in committed:
                record = committed[path]
            else:
                if frame is None:
                    frame = load_partition(data[dataset], "test")
                    if frame["sample_id"].isna().any() or not frame["sample_id"].is_unique:
                        raise ValueError("Nonunique final test sample IDs")
                    if dataset == "unsw":
                        groups = _source_feature_hashes(data[dataset], plan["partitions"][dataset])
                        nonoverlap = ~frame["feature_hash"].isin(groups).to_numpy()
                        original = read_json(data[dataset] / "audit.json")["train_test_overlap"]
                        if (int((~nonoverlap).sum()) != original["test_rows"]
                                or frame.loc[~nonoverlap, "feature_hash"].nunique() != original["n_feature_groups"]):
                            raise ValueError("UNSW full-field overlap differs from prepared audit")
                record = _test_result(experiment, path, item, frame, entry["threshold"], nonoverlap, plan, binding)
            files.update({(path / name).relative_to(output).as_posix(): sha for name, sha in record["files"].items()})
            files[(path / "completion.json").relative_to(output).as_posix()] = sha256_file(path / "completion.json")
    return files


def _checksum(root):
    stamp = root / CHECKSUM
    actual = sha256_file(root / DONE)
    if stamp.is_file():
        if stamp.read_text(encoding="ascii").strip() != actual:
            raise ValueError("Final completion checksum mismatch")
    else:
        tmp = stamp.with_suffix(".sha256.tmp")
        tmp.write_text(actual + "\n", encoding="ascii")
        tmp.replace(stamp)


def verify_core(output, *, _pending_checksum=False):
    """Verify the frozen core inventory, including the complete expected test matrix."""
    root = Path(output).resolve()
    if not _pending_checksum and (root / CHECKSUM).read_text(encoding="ascii").strip() != sha256_file(root / DONE):
        raise ValueError("Final completion checksum mismatch")
    completion = _read_record(root / DONE)
    plan = read_json(root / PLAN)
    barrier = read_json(root / BARRIER)
    start = read_json(root / START)
    expected = {PLAN, BARRIER, START}
    for item in plan["candidates"]:
        expected.add(_threshold_file(item))
        expected.add(Path(_threshold_file(item)).with_suffix(".normal.parquet").as_posix())
        run = _test_dir(root, item)
        expected.update((run / n).relative_to(root).as_posix() for n in (*_result_files(), "completion.json"))
    if (completion.get("status") != "complete" or completion.get("phase") != "frozen_test_evaluation"
            or completion.get("plan_sha256") != sha256_file(root / PLAN)
            or completion.get("thresholds_lock_sha256") != sha256_file(root / BARRIER)
            or start["n_candidates"] != len(plan["candidates"]) or barrier["n_candidates"] != len(plan["candidates"])
            or set(completion["files"]) != expected or completion["n_candidates"] != len(plan["candidates"])
            or len(barrier["thresholds"]) != len(plan["candidates"])
            or set(barrier["thresholds"]) != {_threshold_file(item) for item in plan["candidates"]}
            or barrier["plan_sha256"] != sha256_file(root / PLAN)
            or start["plan_sha256"] != barrier["plan_sha256"]
            or start["thresholds_lock_sha256"] != sha256_file(root / BARRIER)):
        raise ValueError("Incomplete final evaluation matrix or broken threshold barrier")
    for name, digest in completion["files"].items():
        _artifact(root, name, digest)
    for item in plan["candidates"]:
        key = _threshold_file(item)
        entry = barrier["thresholds"][key]
        cal = _read_record(root / key)
        normal_file = Path(key).with_suffix(".normal.parquet").as_posix()
        if (entry["sha256"] != completion["files"][key] or entry["normal_scores_file"] != normal_file
                or entry["normal_scores_sha256"] != completion["files"][normal_file]
                or cal["normal_scores_sha256"] != entry["normal_scores_sha256"] or cal["threshold"] != entry["threshold"]):
            raise ValueError("Calibration evidence differs from matrix barrier")
        run = _test_dir(root, item)
        record = _read_record(run / "completion.json")
        binding = _result_binding(item, plan, barrier, completion["files"][BARRIER], completion["files"][START])
        if (any(record.get(k) != v for k, v in binding.items())
                or record["files"] != {n: completion["files"][(run / n).relative_to(root).as_posix()] for n in _result_files()}):
            raise ValueError("Committed test result differs from matrix barrier")
    return completion


def _collection_paths(output, candidates):
    known_cal = set(_threshold_file(r) for r in candidates)
    known_cal.update(Path(n).with_suffix(".normal.parquet").as_posix() for n in list(known_cal))
    known_runs = {(_test_dir(output, r)).relative_to(output).as_posix() for r in candidates}
    if (output / "calibration").is_dir():
        for path in (output / "calibration").rglob("*"):
            if path.is_file() and path.suffix != ".tmp" and path.relative_to(output).as_posix() not in known_cal:
                raise ValueError("Unplanned calibration record in final output")
    if (output / "tests").is_dir():
        for path in (output / "tests").glob("*/*"):
            if path.relative_to(output).as_posix() not in known_runs:
                raise ValueError("Unplanned test candidate in final output")


def run_evaluation(experiment: Path, data_dirs: dict, output: Path, config: dict, *, resume: bool = False) -> None:
    """Execute one frozen matrix, or verify and resume the same output directory.

    The accepted source experiment and prepared inputs are read-only. Existing
    results are reused only if bound to the same plan and complete threshold
    barrier. No thresholds or predictions from completed runs are overwritten.
    """
    experiment = Path(experiment).resolve()
    output = Path(output).resolve()
    if set(data_dirs) != set(DATASETS):
        raise ValueError("Both independent prepared datasets are required")
    data = {key: Path(value).resolve() for key, value in data_dirs.items()}
    for src in (experiment, *data.values()):
        if src.is_relative_to(output) or output.is_relative_to(src):
            raise ValueError("Final evaluation output must be separate from every input")
    if not output.exists():
        output.mkdir(parents=True)
    with _exclusive(output):
        existing = {p.name for p in output.iterdir()} - {".stage5.lock", PLAN + ".tmp"}
        if existing and not resume:
            raise FileExistsError("Final evaluation output exists; use resume for this fixed matrix")
        if existing and PLAN not in existing:
            raise ValueError("Unrecognized nonempty final evaluation output")
        # Bind inputs without loading any model or reading a reserved partition.
        plan = _build_plan(experiment, data, config)
        saved = read_json(output / PLAN) if (output / PLAN).is_file() else None
        plan["frozen_at"] = saved["frozen_at"] if saved is not None else utc_now()
        plan["evaluator_release_commit"] = saved["evaluator_release_commit"] if saved is not None else code_fingerprint().get("commit")
        plan_hash = _record(output, PLAN, plan)
        _collection_paths(output, plan["candidates"])
        if (output / DONE).is_file():
            if not (output / CHECKSUM).exists():
                verify_core(output, _pending_checksum=True)
                _checksum(output)
            verify_core(output)
            return
        if (output / CHECKSUM).exists():
            raise ValueError("Missing previously sealed final completion")
        started = (output / START).is_file()
        has_thresholds = (output / BARRIER).is_file()
        if (started and not has_thresholds) or ((output / "tests").exists() and not started):
            raise ValueError("Incomplete matrix barrier: would reopen test without fixed thresholds")
        barrier = _calibrate(experiment, data, output, plan, plan_hash, has_thresholds)
        barrier["locked_at"] = read_json(output / BARRIER)["locked_at"] if has_thresholds else utc_now()
        barrier_hash = _record(output, BARRIER, barrier)
        start = {"schema_version": 1, "plan_sha256": plan_hash, "thresholds_lock_sha256": barrier_hash,
                 "n_candidates": len(plan["candidates"]), "state": "test_opened_for_this_fixed_matrix",
                 "started_at": read_json(output / START)["started_at"] if started else utc_now()}
        start_hash = _record(output, START, start)
        files = {PLAN: plan_hash, BARRIER: barrier_hash, START: start_hash}
        files.update({name: entry["sha256"] for name, entry in barrier["thresholds"].items()})
        files.update({entry["normal_scores_file"]: entry["normal_scores_sha256"]
                      for entry in barrier["thresholds"].values()})
        files.update(_run_tests(experiment, data, output, plan, barrier, barrier_hash, start_hash))
        if _evaluator_source() != plan["evaluator_source"] or code_fingerprint()["source_sha256"] != plan["training_source_sha256"]:
            raise ValueError("Evaluation source changed during scoring; cannot seal results")
        _record(output, DONE, _hash_record({"schema_version": 1, "status": "complete",
                "phase": "frozen_test_evaluation", "n_candidates": len(plan["candidates"]),
                "plan_sha256": plan_hash, "thresholds_lock_sha256": barrier_hash,
                "finished_at": utc_now(), "files": files}))
        _checksum(output)
        verify_core(output)


def _bound_candidates(experiment, dataset, wanted, stage4, seal, manifest, manifest_sha):
    """Read only externally sealed metadata/bytes, never execute archived code."""
    from ids.stage4 import conditions, make_suite
    cfg = stage4["identity"]["config"]
    suite = make_suite(cfg, dataset)
    names = conditions(cfg, dataset)
    run_root = f"{dataset}/runs/{suite['suite_name']}"
    reports = f"reports/{dataset}/study.json"
    study = read_json(_artifact(experiment, reports, seal[reports]))
    if study["dataset_manifest_sha256"] != manifest_sha:
        raise ValueError("Study belongs to a different prepared dataset")
    specs = {f"{i:02d}-{m['model']}-s{seed}": (m, seed) for i, m in enumerate(suite["models"]) for seed in suite["seeds"]}
    study_runs = {r["run_id"]: (g, r) for g in study["groups"] for r in g["runs"]}
    items = []
    for condition, name in wanted:
        if name not in specs or name not in study_runs:
            raise ValueError(f"Missing fixed candidate {dataset}/{name}")
        model, seed = specs[name]
        if condition not in names or model["features"] != names[condition]:
            raise ValueError("Requested condition differs from Stage 4 candidate identity")
        group, member = study_runs[name]
        if member["seed"] != seed or group["spec"]["model"] != model["model"]:
            raise ValueError("Sealed study model or seed differs")
        selected = member["selected_features"]
        if len(selected) != group["spec"]["raw_feature_budget"] or not set(selected).issubset(manifest["features"]):
            raise ValueError("Invalid frozen feature budget")
        if group["spec"]["feature_config"] != model["features"]:
            raise ValueError("Sealed study feature choice differs from fixed condition")
        directory = f"{run_root}/{name}"
        rel = f"{directory}/run.json"
        digest = seal[rel]
        meta = read_json(_artifact(experiment, rel, digest))
        model_path = f"{directory}/model.joblib"
        if (meta["run_id"] != name or meta["status"] != "complete" or meta["phase"] != "development"
                or meta["model_name"] != model["model"] or digest != member["run_manifest_sha256"]
                or meta["dataset_manifest_sha256"] != manifest_sha or meta["data_kind"] != manifest["data_kind"]
                or member["source_sha256"] != stage4["identity"]["source_sha256"]
                or member["model_sha256"] != meta["files"]["model.joblib"]
                or member["model_sha256"] != seal[model_path]):
            raise ValueError("Frozen Stage 4 run identity differs from accepted inventory")
        for filename in ("config.json", "code.json", "dataset.json", "selected_features.json"):
            relative = f"{directory}/{filename}"
            if meta["files"][filename] != seal[relative]:
                raise ValueError("Run file digest differs from Stage 4 seal")
            _artifact(experiment, relative, seal[relative])
        conf = read_json(experiment / directory / "config.json")
        if (conf["seed"] != seed or conf["max_fpr"] != cfg["max_fpr"]
                or conf["model_config"] != group["spec"]["model_config"]
                or canonical_hash(conf) != meta["config_sha256"] or meta["files"]["dataset.json"] != manifest_sha
                or read_json(experiment / directory / "selected_features.json") != selected
                or read_json(experiment / directory / "code.json")["source_sha256"] != member["source_sha256"]):
            raise ValueError("Model config, code, dataset or selected feature identity changed")
        path = _artifact(experiment, model_path, member["model_sha256"])
        deployment = _deployment(group, member)
        if path.stat().st_size != deployment["model_file_bytes"] or deployment["cpu_rounds"] != cfg["benchmark_rounds"]:
            raise ValueError("Invalid deployment measurement identity")
        if (model["model"], condition) not in ROLES[dataset]:
            raise ValueError("Model/condition is outside the fixed Stage 5 comparison")
        items.append({"dataset": dataset, "run_id": name, "seed": seed, "model": model["model"],
                      "role": ROLES[dataset][model["model"], condition],
                      "condition": condition, "run_manifest_sha256": digest, "model_path": model_path,
                      "model_sha256": member["model_sha256"], "dataset_manifest_sha256": manifest_sha,
                      "config_sha256": meta["config_sha256"], "selected_features": selected, "deployment": deployment})
    expected = {(m, c, s) for m, c in ROLES[dataset] for s in cfg["seeds"]}
    observed = {(r["model"], r["condition"], r["seed"]) for r in items}
    if expected != observed or len(items) != len(expected):
        raise ValueError("Incomplete or duplicate fixed candidate/seed matrix")
    return items

def _build_plan(experiment, data, config):
    from ids.stage4 import verify_completion
    policy = _policy(config)
    seal_path = _artifact(experiment, "STAGE4_COMPLETE.json", policy["complete"])
    complete = read_json(seal_path)
    stage4_path = _artifact(experiment, "STAGE4_PLAN.json", policy["plan"])
    stage4 = read_json(stage4_path)
    if (stage4["identity"]["source_sha256"] != policy["source"]
            or code_fingerprint()["source_sha256"] != policy["source"]
            or complete["files"].get("STAGE4_PLAN.json") != policy["plan"]
            or stage4["identity"]["datasets"] != policy["datasets"]
            or complete["profile"] != ("smoke" if policy["smoke"] else "research")):
        raise ValueError("Accepted Stage 4 source, data, plan or profile changed")
    verify_completion(experiment)
    candidates, partitions, known = [], {}, {}
    for dataset in DATASETS:
        digest = policy["datasets"][dataset]
        if complete["files"][f"{dataset}/manifest.json"] != digest:
            raise ValueError("Stage 4 dataset is outside external trust anchors")
        manifest = read_json(_artifact(data[dataset], "manifest.json", digest))
        if manifest["data_kind"] == "synthetic" and not policy["smoke"]:
            raise ValueError("Synthetic inputs cannot be used for a research evaluation")
        if policy["smoke"] and manifest["data_kind"] != "synthetic":
            raise ValueError("Smoke mode is restricted to synthetic data")
        if (data[dataset] / "manifest.sha256").read_text(encoding="ascii").strip() != digest:
            raise ValueError("Prepared manifest sidecar does not match training")
        partitions[dataset] = manifest["partitions"]
        for split in ("train", "validation", "calibration", "test"):
            if partitions[dataset][split]["file"] != split + ".parquet":
                raise ValueError("Invalid canonical partition filename")
        diag_path = f"{dataset}/selection/diagnostics.json"
        diagnostics = read_json(_artifact(experiment, diag_path, complete["files"][diag_path]))
        if diagnostics["identity"]["dataset_manifest_sha256"] != digest:
            raise ValueError("Selection diagnostics belong to another dataset")
        known[dataset] = sorted(k for k in diagnostics["pool"]["source_family_counts"] if k != "Normal")
        candidates.extend(_bound_candidates(experiment, dataset, policy["runs"][dataset], stage4,
                                             complete["files"], manifest, digest))
    return {"schema_version": 1, "phase": "frozen_final_plan", "config": config, "config_sha256": canonical_hash(config),
            "stage4_plan_sha256": policy["plan"], "stage4_complete_sha256": policy["complete"],
            "training_source_sha256": policy["source"], "evaluator_source": _evaluator_source(),
            "environment": _cpu_environment(), "inference_device": "cpu", "threads": policy["threads"],
            "max_fpr": .01, "calibration_scoring": "normal_only_in_original_partition_order",
            "candidates": candidates, "n_candidates": len(candidates),
            "dataset_manifest_sha256": policy["datasets"], "partitions": partitions, "known_attack_families": known}

def _policy(config):
    if config.get("profile") != "smoke":
        return _research_policy(config)
    # Internal synthetic fixtures. Research entrypoints never expose this branch.
    keys = {"profile", "stage4_plan_sha256", "stage4_complete_sha256", "training_source_sha256",
            "dataset_manifest_sha256", "candidates", "threads"}
    if set(config) != keys or type(config["threads"]) is not int or not 1 <= config["threads"] <= 4:
        raise ValueError("Unexpected synthetic evaluation policy")
    runs = {d: [(r["condition"], r["run_id"]) for r in config["candidates"][d]] for d in DATASETS}
    if set(config["candidates"]) != set(DATASETS) or any(not rows for rows in runs.values()):
        raise ValueError("Missing fixed synthetic candidate list")
    return {"smoke": True, "complete": config["stage4_complete_sha256"], "plan": config["stage4_plan_sha256"],
            "source": config["training_source_sha256"], "datasets": config["dataset_manifest_sha256"],
            "runs": runs, "threads": config["threads"]}

def _research_policy(config):
    """Restrict research deserialization to the accepted Stage 4 artifact seal."""
    published = HERE / "protocol.json"
    if not published.is_file() or config != read_json(published):
        raise ValueError("Research evaluation requires the published fixed policy")
    trusted = {"complete": "59b3c7909f2a5242d947b852e32e292b93a23c58287548bf6ef878d8c8cba8fa",
            "plan": "d2ea51df70d3c295384096dc196231c6baa78742ead1bd0b81130e23c4811dc4",
            "source": "233644d9d52d2d785a4de3ebca007dc34ada11cbd6c323876af3848eb779aaa5",
            "datasets": {"unsw": "6348a936f68a67a11e241452eff9abdf334d4ea02f71472f805f38c532c002d4",
                         "cicids": "af4bb4e6724910ff0e70e7479c265addd73e4f4bbdb2ad8c140d92c18ea3db7d"}}
    pins = config["pins"]
    if (config["profile"] != "research" or config["seeds"] != [17, 29, 43] or config["max_fpr"] != .01
            or type(config["inference_threads"]) is not int or config["inference_threads"] != 1
            or pins["stage4_complete_sha256"] != trusted["complete"] or pins["stage4_plan_sha256"] != trusted["plan"]
            or pins["stage4_source_sha256"] != trusted["source"] or pins["dataset_manifest_sha256"] != trusted["datasets"]):
        raise ValueError("Published protocol disagrees with accepted model/data/source anchors")
    expected = {"unsw_mlp_all42": ("unsw", "mlp", "all", 42, "05-mlp"),
                "unsw_rf_all42": ("unsw", "rf", "all", 42, "00-rf"),
                "cicids_mlp_mi24": ("cicids", "mlp", "mi", 24, "04-mlp"),
                "cicids_mlp_all82": ("cicids", "mlp", "all", 82, "03-mlp")}
    groups = config["groups"]
    observed = {g["id"]: (g["dataset"], g["model"], g["selection"], g["n_features"], g["run_prefix"])
                for g in groups}
    if len(groups) != 4 or observed != expected:
        raise ValueError("Published candidate list differs from the fixed four groups")
    runs = {d: [(g["selection"], f"{g['run_prefix']}-s{s}") for g in groups if g["dataset"] == d
                for s in config["seeds"]] for d in DATASETS}
    return {"smoke": False, **trusted, "runs": runs, "threads": config["inference_threads"]}
