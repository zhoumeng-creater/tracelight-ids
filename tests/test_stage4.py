"""Real small-data stage-4 workflow, with only CPU repetition counts shortened."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd
import pytest
import yaml

from ids import benchmark, benchmark_workflow, data, runner, stage4
from ids.cicids import make_synthetic_cicids_dataset, prepare_cicids_dataset
from ids.common import code_fingerprint, environment_snapshot, read_json, sha256_file, write_json

REPO = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(f"stage4_test_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def prepared_pair(tmp_path_factory):
    root = tmp_path_factory.mktemp("stage4-inputs")
    raw = data.make_synthetic_dataset(root / "unsw-raw", n_train=600, n_test=120)
    unsw = root / "unsw"
    data.prepare_dataset(*raw, unsw, data_kind="synthetic", seed=17)
    raw_cic = make_synthetic_cicids_dataset(root / "cic-raw", n_rows=600)
    cic = root / "cicids"
    prepare_cicids_dataset(raw_cic, cic, data_kind="synthetic", seed=20261102)
    return {"unsw": unsw, "cicids": cic}


@pytest.fixture(scope="module")
def stage4_template(prepared_pair, tmp_path_factory):
    root = tmp_path_factory.mktemp("stage4-template")
    config = yaml.safe_load((REPO / "configs" / "stage4-smoke.yaml").read_text(encoding="utf-8"))
    fingerprint, environment = code_fingerprint(), environment_snapshot()
    load = data.load_partition
    cpu_predict = benchmark.benchmark_predict
    loaded, selection_loaded = [], []

    def development_load(path, split):
        assert split in {"train", "validation"}, "Stage 4 opened a reserved partition"
        loaded.append((Path(path).name, split))
        return load(path, split)

    def train_only(path, split):
        assert split == "train", "Selection opened data other than train"
        selection_loaded.append((Path(path).name, split))
        return load(path, split)

    def short_cpu(predict_fn, frame, threads=1, **_kwargs):
        return cpu_predict(predict_fn, frame, threads=threads, warmup=1,
                           repeats=2, single_rows=3, batch_size=16)

    with pytest.MonkeyPatch.context() as patch:
        for module in (stage4, runner, benchmark_workflow):
            patch.setattr(module, "code_fingerprint", lambda: fingerprint)
            patch.setattr(module, "environment_snapshot", lambda: copy.deepcopy(environment))
        patch.setattr(stage4, "load_partition", train_only)
        patch.setattr(runner, "load_partition", development_load)
        patch.setattr(data, "load_partition", development_load)
        patch.setattr(benchmark, "benchmark_predict", short_cpu)
        stage4.run_stage4(prepared_pair, root / "experiment", config, device="cpu")
    return {"root": root / "experiment", "data": prepared_pair, "config": config,
            "fingerprint": fingerprint, "environment": environment,
            "loaded": loaded, "selection_loaded": selection_loaded}


@pytest.fixture
def stage4_copy(stage4_template, tmp_path, monkeypatch):
    root = tmp_path / "experiment with spaces"
    shutil.copytree(stage4_template["root"], root)
    for module in (stage4, runner, benchmark_workflow):
        monkeypatch.setattr(module, "code_fingerprint", lambda: stage4_template["fingerprint"])
        monkeypatch.setattr(module, "environment_snapshot", lambda: copy.deepcopy(stage4_template["environment"]))
    return {**stage4_template, "root": root, "config": copy.deepcopy(stage4_template["config"])}


def test_formal_and_smoke_have_48_and_16_exactly_paired_runs():
    for filename, total, n_seeds in (("stage4.yaml", 48, 3), ("stage4-smoke.yaml", 16, 1)):
        config = yaml.safe_load((REPO / "configs" / filename).read_text(encoding="utf-8"))
        stage4.validate_config(config)
        assert sum(len(stage4.make_suite(config, name)["models"]) * len(config["seeds"])
                   for name in ("unsw", "cicids")) == total
        assert len(config["seeds"]) == n_seeds
        for dataset, expected in (("unsw", 5), ("cicids", 3)):
            conditions = stage4.conditions(config, dataset)
            assert len(conditions) == expected
            assert all(x["selection_seed"] == config["selection"]["selection_seed"] for x in conditions.values())
            for model in ("rf", "mlp"):
                entries = [m for m in stage4.make_suite(config, dataset)["models"] if m["model"] == model]
                assert len(entries) == expected
                assert all(x["hyperparameters"] == config["models"][model] for x in entries)


def test_group_pool_and_outer_sampling_preserve_groups_and_order():
    frame = pd.DataFrame({"sample_id": [f"s{i}" for i in range(12)],
                          "group_hash": np.repeat(["a", "b", "c"], 4),
                          "feature_hash": [f"f{i}" for i in range(12)],
                          "label": np.tile([0, 1], 6), "attack_cat": np.tile(["Normal", "Attack"], 6)})
    pool, manifest = stage4.sample_groups(frame, 22, max_rows=5)
    assert len(pool) == 8 and manifest["soft_max_rows"] == 5
    assert pool.index.is_monotonic_increasing
    assert (pool.groupby("group_hash").size() == 4).all()
    other, other_manifest = stage4.sample_groups(frame, 22, max_rows=5)
    pd.testing.assert_frame_equal(pool, other)
    assert manifest == other_manifest
    outer, outer_manifest = stage4.sample_groups(frame, 33, fraction=.8)
    assert len(outer) == 8 and outer_manifest["n_groups"] == 2
    assert set(outer.sample_id).issubset(frame.sample_id)


def test_prepare_selections_opens_only_train_bytes(prepared_pair, tmp_path, monkeypatch):
    config = yaml.safe_load((REPO / "configs" / "stage4-smoke.yaml").read_text(encoding="utf-8"))
    original = Path.open
    accesses = []

    def audited_open(path, *args, **kwargs):
        if path.suffix == ".parquet":
            assert path.name == "train.parquet", f"Selection opened forbidden file: {path.name}"
            accesses.append(path.name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", audited_open)
    for dataset, path in prepared_pair.items():
        results, diagnostic = stage4.prepare_selections(path, tmp_path / dataset, config, dataset, False)
        assert len(results) == (5 if dataset == "unsw" else 3)
        assert all(len(value["selections"]) == config["selection"]["outer_repeats"]
                   for value in diagnostic["methods"].values())
        assert len({r["metadata"]["pool_binding"]["identity_sha256"] for r in results.values()}) == 1
    assert accesses and set(accesses) == {"train.parquet"}


def test_real_workflow_uses_train_validation_and_has_all_artifacts(stage4_template):
    value = stage4_template
    assert set(split for _, split in value["loaded"]) == {"train", "validation"}
    assert set(value["selection_loaded"]) == {("unsw", "train"), ("cicids", "train")}
    completed = read_json(value["root"] / "STAGE4_COMPLETE.json")
    assert completed["n_runs"] == 16
    assert completed["final_evaluation_performed"] is False
    for dataset, expected in (("unsw", 10), ("cicids", 6)):
        study = read_json(value["root"] / "reports" / dataset / "study.json")
        assert study["n_runs"] == expected
        assert all(group["seeds"] == [17] for group in study["groups"])
        run_root = value["root"] / dataset / "runs" / stage4.make_suite(value["config"], dataset)["suite_name"]
        runs = sorted(run_root.glob("*-s17"))
        assert len(runs) == expected
        selections = {}
        for run in runs:
            runner.verify_run(run)
            selected = read_json(run / "selection.json")
            settings = selected["settings"]
            key = (settings["method"], settings["redundancy_threshold"])
            selections.setdefault(key, []).append(selected["result_sha256"])
            assert not (run / "evaluation_plan.json").exists()
            assert not (run / "test_metrics.json").exists()
        assert all(len(set(hashes)) == 1 for hashes in selections.values())


def test_complete_resume_verifies_without_refitting_or_rebenchmarking(stage4_copy, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Completed resume must not repeat fitting, selection, or CPU measurements")
    monkeypatch.setattr(runner, "train_model", forbidden)
    monkeypatch.setattr(stage4, "prepare_selections", forbidden)
    monkeypatch.setattr(benchmark_workflow, "benchmark_suite", forbidden)
    value = stage4_copy
    before = sha256_file(value["root"] / "STAGE4_COMPLETE.json")
    stage4.run_stage4(value["data"], value["root"], value["config"], device="cpu", resume=True)
    assert sha256_file(value["root"] / "STAGE4_COMPLETE.json") == before


@pytest.mark.parametrize("key,value", [("n_runs", 0), ("benchmark_rounds", 9),
                                      ("phase", "test_evaluation"), ("final_evaluation_performed", True)])
def test_complete_resume_rejects_broken_completion_claim(stage4_copy, key, value):
    item = stage4_copy
    path = item["root"] / "STAGE4_COMPLETE.json"
    completion = read_json(path)
    completion[key] = value
    write_json(path, completion)
    with pytest.raises(ValueError):
        stage4.run_stage4(item["data"], item["root"], item["config"], resume=True)


@pytest.mark.parametrize("mutation", ["checksum", "identity"])
def test_selector_cache_change_is_rejected(stage4_copy, mutation):
    item = stage4_copy
    target = item["root"] / "unsw" / "selection" / "mi.json"
    if mutation == "checksum":
        target.write_bytes(target.read_bytes() + b" ")
    else:
        content = read_json(target)
        content["identity"]["source_sha256"] = "0" * 64
        write_json(target, content)
        target.with_suffix(".json.sha256").write_text(sha256_file(target) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="selection artifact|cache"):
        stage4.prepare_selections(item["data"]["unsw"], target.parent, item["config"], "unsw", True)


@pytest.mark.parametrize("mutation", ["source", "config", "device", "environment"])
def test_resume_identity_changes_are_rejected(stage4_copy, monkeypatch, mutation):
    item = stage4_copy
    device = "cpu"
    if mutation == "source":
        monkeypatch.setattr(stage4, "code_fingerprint", lambda: {**item["fingerprint"], "source_sha256": "0" * 64})
    elif mutation == "config":
        item["config"]["models"]["rf"]["n_estimators"] += 1
    elif mutation == "device":
        device = "cuda"
    else:
        monkeypatch.setattr(stage4, "environment_snapshot", lambda: {**item["environment"], "python": "0.0.0"})
    with pytest.raises(ValueError, match="changed source/data/config/device/environment"):
        stage4.run_stage4(item["data"], item["root"], item["config"], device=device, resume=True)


def test_complete_resume_still_checks_train_and_validation_bytes(stage4_copy, tmp_path):
    item = stage4_copy
    copied = tmp_path / "changed-input"
    shutil.copytree(item["data"]["unsw"], copied)
    (copied / "validation.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Partition checksum mismatch: validation"):
        stage4.run_stage4({**item["data"], "unsw": copied}, item["root"], item["config"], resume=True)


def test_incomplete_resume_cannot_rebind_completed_models_to_regenerated_selection(stage4_copy, monkeypatch):
    from ids.selection import _hash
    item = stage4_copy
    (item["root"] / "STAGE4_COMPLETE.json").unlink()
    path = item["root"] / "unsw" / "selection" / "all.json"
    saved = read_json(path)
    # Represents an independently regenerated valid cache: identical fields and
    # settings, but a new construction record. Existing runs bind the old one.
    result = saved["result"]
    result["metadata"]["seconds"] += .1
    result["result_sha256"] = _hash({k: v for k, v in result.items() if k != "result_sha256"})
    write_json(path, saved)
    path.with_suffix(".json.sha256").write_text(sha256_file(path) + "\n", encoding="ascii")
    monkeypatch.setattr(runner, "train_model", lambda *_a, **_k: pytest.fail("Existing model was retrained"))
    with pytest.raises(ValueError, match="[Ss]election"):
        stage4.run_stage4(item["data"], item["root"], item["config"], resume=True)


def test_study_requires_expected_paired_comparison_coverage(stage4_template):
    root, config = stage4_template["root"], stage4_template["config"]
    for dataset in ("unsw", "cicids"):
        study = read_json(root / "reports" / dataset / "study.json")
        study["feature_budget_comparisons"] = []
        with pytest.raises(ValueError, match="[Pp]aired|[Cc]omparison|[Bb]aseline"):
            stage4.verify_study(study, stage4.make_suite(config, dataset))


def test_stage4_packaging_writes_verified_inventory(stage4_copy, tmp_path):
    root = stage4_copy["root"]
    output = tmp_path / "returned package.zip"
    package = script("package_run")
    assert package.package_run(root, output) == output
    assert output.with_suffix(".zip.sha256").read_text().strip() == sha256_file(output)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        lines = archive.read(f"{root.name}/MANIFEST.sha256").decode().splitlines()
        inventory = dict(line.split("  ", 1)[::-1] for line in lines)
        actual = {name.removeprefix(f"{root.name}/") for name in archive.namelist()}
        assert set(inventory) | {"MANIFEST.sha256"} == actual
        for filename, digest in inventory.items():
            assert hashlib.sha256(archive.read(f"{root.name}/{filename}")).hexdigest() == digest
    with pytest.raises(FileExistsError):
        package.package_run(root, output)


@pytest.mark.parametrize("mutation", ["extra", "modified", "missing", "incomplete_round"])
def test_package_rejects_unsealed_or_incomplete_outputs(stage4_copy, tmp_path, mutation):
    root = stage4_copy["root"]
    if mutation == "extra":
        (root / "unexpected.json").write_text("{}")
    elif mutation == "modified":
        report = root / "reports" / "index.html"
        report.write_bytes(report.read_bytes() + b"modified")
    elif mutation == "missing":
        next((root / "unsw" / "runs").rglob("model.joblib")).unlink()
    else:
        # Even resealing the modified report cannot invent a missing CPU round.
        path = root / "reports" / "unsw" / "study.json"
        study = read_json(path)
        study["groups"][0]["cpu_benchmarks"] = []
        write_json(path, study)
        completed = read_json(root / "STAGE4_COMPLETE.json")
        completed["files"][path.relative_to(root).as_posix()] = sha256_file(path)
        write_json(root / "STAGE4_COMPLETE.json", completed)
    with pytest.raises((ValueError, FileNotFoundError)):
        script("package_run").package_run(root, tmp_path / "should-not-exist.zip")
    assert not (tmp_path / "should-not-exist.zip").exists()


def test_chance_corrected_stability_is_not_raw_jaccard():
    result = stage4.stability_summary([["a", "b"], ["a", "c"]], ["a", "b", "c", "d"])
    assert result["mean_pairwise_jaccard"] == pytest.approx(1 / 3)
    assert result["chance_corrected_stability"] == pytest.approx(0)
    all_fields = stage4.stability_summary([["a", "b"], ["a", "b"]], ["a", "b"])
    assert all_fields["mean_pairwise_jaccard"] == 1
    assert all_fields["chance_corrected_stability"] is None


def test_stage5_reads_trained_stage4_artifacts_and_returns_reproducible_package(stage4_template, tmp_path):
    # Reuse the small, actually fitted Stage 4 fixture in this module. No research data is needed for CI.
    from ids.evaluation import calibrate_threshold, evaluate_binary
    from workflows.stage5 import core, reporting

    experiment, inputs = stage4_template["root"], stage4_template["data"]
    identity = read_json(experiment / "STAGE4_PLAN.json")["identity"]
    config = {"profile": "smoke", "stage4_plan_sha256": sha256_file(experiment / "STAGE4_PLAN.json"),
              "stage4_complete_sha256": sha256_file(experiment / "STAGE4_COMPLETE.json"),
              "training_source_sha256": identity["source_sha256"], "dataset_manifest_sha256": identity["datasets"],
              "threads": 1, "candidates": {
                  "unsw": [{"condition": "all", "run_id": "05-mlp-s17"}, {"condition": "all", "run_id": "00-rf-s17"}],
                  "cicids": [{"condition": "mi", "run_id": "04-mlp-s17"}, {"condition": "all", "run_id": "03-mlp-s17"}],
              }}
    output = tmp_path / "stage5"
    core.run_evaluation(experiment, inputs, output, config)
    archive = reporting.finalize(output, config)
    original = sha256_file(archive)
    with zipfile.ZipFile(archive) as z:
        summary = read_json(output / "reports" / "summary.json")
        assert len(summary["runs"]) == 4 and summary["profile"] == "smoke"
        assert len(summary["comparisons"]) == 3
        assert "合成数据" in z.read("reports/index.html").decode("utf-8")
        for run in summary["runs"]:
            directory = f"tests/{run['dataset']}/{run['run_id']}"
            with z.open(directory + "/predictions.parquet") as src:
                predictions = pd.read_parquet(src)
            with z.open(f"calibration/{run['dataset']}/{run['run_id']}.normal.parquet") as src:
                normals = pd.read_parquet(src)
            calibrated = calibrate_threshold(np.zeros(len(normals)), normals["score"], .01)
            assert calibrated["threshold"] == run["threshold"]
            assert evaluate_binary(predictions["true_label"], predictions["score"], run["threshold"],
                                   predictions["attack_category"]) == run["full"]
    core.run_evaluation(experiment, inputs, output, config, resume=True)
    assert sha256_file(reporting.finalize(output, config)) == original
