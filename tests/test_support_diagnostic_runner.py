from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import multiprocessing
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from ids.common import read_json, sha256_file, write_json
from ids.support_diagnostic import data, model, runner
from ids.threshold_validation.observations import RAW_FEATURES


class QuietResources:
    def __init__(self, root):
        self.activity = "synthetic"

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None


@pytest.fixture
def synthetic_batch(tmp_path, monkeypatch):
    reg = {"version": "synthetic-support", "status": "synthetic", "label_availability": "synthetic",
           "source_train": ["S"], "roles": {"target_train": ["T"], "normal_calibration": ["C"],
           "forward_development": ["B"], "transport_external": ["D"]},
           "groups": ["hgb"], "supports": ["source", "target", "mixed"],
           "views": ["raw8", "ports_zero8"], "alpha": 0.01}
    frames = {}
    for i, name in enumerate("STCBD"):
        n, attacks = 103, 3
        values = {feature: np.ones(n) for feature in RAW_FEATURES}
        values.update(ip_protocol=np.full(n, 17), resp_port=np.full(n, 443), orig_port=np.full(n, 50000),
                      duration_seconds=np.r_[np.full(n - attacks, 5.0), np.full(attacks, 60.0)])
        frames[name] = pd.DataFrame({**values, "source_id": name, "row_id": np.arange(n),
                                    "flow_end_ms": (i + 1) * 86400000 + np.arange(n),
                                    "family": ["Benign"] * (n - attacks) + ["ExternalC2"] * attacks})
    monkeypatch.setattr(data, "registration", lambda: reg)
    monkeypatch.setattr(data, "specs", lambda _: {name: {"name": name} for name in frames})
    monkeypatch.setattr(data, "frame", lambda root, name: pd.read_parquet(
        Path(root) / "local/prepared" / (name + ".parquet")))

    def prepare(root, raw_dir, spec):
        return [runner._write_frame(Path(root) / "local/prepared" / (spec["name"] + ".parquet"),
                                    frames[spec["name"]])]

    def training_sets(root, registered):
        outputs = []
        source = data.frame(root, "S").assign(domain="source")
        target = data.frame(root, "T").assign(domain="target")
        for support, rows in {"source": source, "target": target,
                              "mixed": pd.concat([source, target], ignore_index=True)}.items():
            rows = rows.assign(label=rows.family.ne("Benign").astype(int), weight=1.0)
            outputs.append(runner._write_frame(Path(root) / "local/training" / (support + ".parquet"), rows))
        return outputs

    monkeypatch.setattr(data, "prepare", prepare)
    monkeypatch.setattr(data, "training_sets", training_sets)
    monkeypatch.setattr(data, "calibration_frame", lambda root, registered: data.frame(root, "C").loc[
        lambda rows: rows.family.eq("Benign")].reset_index(drop=True))
    monkeypatch.setattr(runner, "resolve_inputs", lambda *_: {"synthetic": {"sha256": "fixed", "bytes": 1}})
    monkeypatch.setattr(runner, "code_snapshot", lambda: {"synthetic_code": "v1"})
    monkeypatch.setattr(runner, "environment_snapshot", lambda **_: {
        "python": "synthetic", "packages": {}, "torch": {"cuda_available": False}})
    monkeypatch.setattr(runner, "ResourceLog", QuietResources)
    reference = tmp_path / "reference"
    reference.mkdir()
    frozen = reference / "raw.joblib"
    write_json(frozen, {"arm": "source-raw8-hgb", "coefficient": 0.0})
    monkeypatch.setattr(runner, "inspect_reference", lambda *_: {"hgb": {
        "path": str(frozen), "relative": "raw.joblib", "sha256": sha256_file(frozen), "bytes": frozen.stat().st_size}})
    fits, forwards = [], []

    def fit(frame, view, path, **kwargs):
        fits.append(path.parent.name)
        write_json(path, {"arm": path.parent.name, "coefficient": 1.0})
        write_json(path.with_suffix(".fit.json"), {"synthetic_model_stub": True})
        return path

    def score(fitted, frame, view, group):
        assert "family" not in frame and "label" not in frame
        forwards.append((fitted["arm"], tuple(frame.source_id.unique())))
        return frame.duration_seconds.to_numpy() / 100 * fitted["coefficient"]

    monkeypatch.setattr(model, "fit", fit)
    monkeypatch.setattr(model, "load", lambda path, **_: read_json(path))
    monkeypatch.setattr(model, "score", score)
    options = dict(raw_dir=None, threads=3, cpu_workers=1, gpu_workers=1, prep_workers=2,
                   synthetic=True, pool_factory=ThreadPoolExecutor)
    return tmp_path / "run", reference, options, fits, forwards, reg


def test_complete_resume_reuses_all_fit_and_forward_stages_and_returns_replay(synthetic_batch):
    root, reference, options, fits, forwards, reg = synthetic_batch
    assert runner.run(root, reference, **options)
    assert len(fits) == 5  # One original raw HGB is copied; five new model-stub arms.
    assert read_json(root / "outcome.json")["completed_evaluations"] == 20
    first_forwards = len(forwards)
    assert runner.run(root, reference, **options)
    assert len(fits) == 5 and len(forwards) == first_forwards
    returned = sorted(root.glob("support-diagnostic-results-*.zip"))[-1]
    with ZipFile(returned) as bundle:
        names = set(bundle.namelist())
        assert "evidence/replay/B.parquet" in names
        assert "evidence/replay/D.parquet" in names
        assert "evidence/replay/calibration.parquet" in names
        assert not any(name.startswith("local/") for name in names)
    replay = pd.read_parquet(root / "evidence/replay/B.parquet")
    assert set(RAW_FEATURES).issubset(replay)
    assert "family" in replay and "source" not in replay and "destination" not in replay
    assert {task["id"] for task in runner.tasks(reg) + runner.reference_tasks()}.issubset(replay)
    pairs = pd.read_csv(root / "evidence/paired.csv.gz")
    strong = pairs.loc[pairs.left_arm.eq(runner.STRONG_ID)]
    assert strong.right_arm.nunique() == 6
    services = pd.read_csv(root / "evidence/services.csv.gz")
    assert services.has_same_service_normal_and_c2.all()


def test_failed_cell_is_packed_then_only_missing_forward_is_retried(synthetic_batch, monkeypatch):
    root, reference, options, fits, forwards, _ = synthetic_batch
    real_score = model.score
    once = [False]

    def fail_one(fitted, frame, view, group):
        if fitted["arm"] == "mixed-ports_zero8-hgb" and set(frame.source_id) == {"D"} and not once[0]:
            once[0] = True
            raise RuntimeError("synthetic cell failure")
        return real_score(fitted, frame, view, group)

    monkeypatch.setattr(model, "score", fail_one)
    assert not runner.run(root, reference, **options)
    assert read_json(root / "outcome.json")["status"] == "incomplete"
    assert list(root.glob("support-diagnostic-results-*.zip"))
    fit_count, forward_count = len(fits), len(forwards)
    assert runner.run(root, reference, **options)
    assert len(fits) == fit_count
    assert len(forwards) == forward_count + 1
    assert read_json(root / "outcome.json")["status"] == "complete"


def test_changed_cached_score_fails_resume_without_silent_refit(synthetic_batch):
    root, reference, options, fits, forwards, _ = synthetic_batch
    assert runner.run(root, reference, **options)
    path = runner.score_path(root, "target-raw8-hgb", "B")
    damaged = pd.read_parquet(path)
    damaged.loc[0, "score"] += 1
    damaged.to_parquet(path, index=False)
    counts = len(fits), len(forwards)
    assert not runner.run(root, reference, **options)
    assert (len(fits), len(forwards)) == counts
    assert "Completed output changed" in read_json(root / "last_attempt.json")["trace"]


def test_plan_never_loads_raw_data_or_models(monkeypatch, capsys):
    monkeypatch.setattr(runner, "inspect_reference", lambda *_: pytest.fail("plan loaded a model"))
    monkeypatch.setattr(runner, "resolve_inputs", lambda *_: pytest.fail("plan read raw data"))
    assert runner.main(["--plan"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["new_fits"] == 20
    assert report["reused_models"] == 4
    assert report["scoring_schemes"] == 28


def test_preflight_only_never_prepares_fits_or_scores(synthetic_batch):
    root, reference, options, fits, forwards, _ = synthetic_batch
    assert runner.run(root, reference, preflight_only=True, **options)
    assert not fits and not forwards
    assert not (root / "local").exists()
    assert read_json(root / "outcome.json")["status"] == "preflight_only"


def test_formal_execution_refuses_cpu_before_resolving_inputs(synthetic_batch, monkeypatch):
    root, reference, options, fits, forwards, _ = synthetic_batch
    options["synthetic"] = False
    monkeypatch.setattr(runner, "resolve_inputs", lambda *_: pytest.fail("CUDA failure should precede data IO"))
    assert not runner.run(root, reference, **options)
    assert not fits and not forwards
    assert "requires CUDA" in read_json(root / "outcome.json")["trace"]


def test_rule_is_exact_registered_observation_and_does_not_use_labels():
    frame = pd.DataFrame({"ip_protocol": [17, 17, 17, 17, 6, 6],
                          "orig_port": [50000, 50000, 443, 50000, 443, 50000],
                          "resp_port": [443, 443, 50000, 80, 50000, 443],
                          "duration_seconds": [29.999, 30, 60, 60, 60, 60]})
    assert runner._proxy(frame).tolist() == [False, True, True, False, False, False]


def test_rejected_schedule_and_preflight_preserve_existing_run_metadata(synthetic_batch):
    root, reference, options, _, _, _ = synthetic_batch
    assert runner.run(root, reference, **options)
    watched = [root / name for name in ("protocol.json", "environment.json", "plan.json", "outcome.json",
                                        "evidence/preflight.json")]
    before = {path: path.read_bytes() for path in watched}
    assert runner.run(root, reference, preflight_only=True, **options)
    assert all(path.read_bytes() == value for path, value in before.items())
    changed = {**options, "threads": 4}
    assert not runner.run(root, reference, **changed)
    assert all(path.read_bytes() == value for path, value in before.items())
    assert "Resume requires" in read_json(root / "last_attempt.json")["trace"]


def test_independent_external_parse_failure_keeps_B_work_and_resumes(synthetic_batch, monkeypatch):
    root, reference, options, fits, _, _ = synthetic_batch
    real_prepare = data.prepare
    once = [False]

    def fail_D(root, raw_dir, spec):
        if spec["name"] == "D" and not once[0]:
            once[0] = True
            raise RuntimeError("synthetic external parse failure")
        return real_prepare(root, raw_dir, spec)

    monkeypatch.setattr(data, "prepare", fail_D)
    assert not runner.run(root, reference, **options)
    assert len(fits) == 5
    outcome = read_json(root / "outcome.json")
    assert outcome["status"] == "incomplete" and outcome["completed_evaluations"] == 10
    assert (root / "evidence/replay/B.parquet").exists()
    assert runner.run(root, reference, **options)
    assert len(fits) == 5


def test_original_C_reference_keeps_original_threshold():
    frame = pd.DataFrame({"row_id": [0], "source_id": ["D"], "flow_end_ms": [0],
                          **{feature: [0.0] for feature in RAW_FEATURES}})
    old = read_json(runner.LEGACY_CALIBRATION)["threshold"]
    score = np.array([old + 0.1])
    strict_B = {"ready": True, "threshold": old + 1.0}
    refs = {task["id"]: task for task in runner.reference_tasks()}
    assert runner._reference_alerts(frame, refs[runner.LEGACY_ID], score, strict_B).tolist() == [True]
    assert runner._reference_alerts(frame, refs[runner.STRONG_ID], score, strict_B).tolist() == [True]
    assert runner._reference_alerts(frame, refs[runner.B_OR_ID], score, strict_B).tolist() == [False]


def test_frozen_runtime_guard_allows_supported_python_patch_and_rejects_package_change():
    packages = read_json(runner.REFERENCE)["runtime"]["packages"]
    runner.verify_runtime({"python": "3.12.99", "packages": packages})
    runner.verify_runtime({"python": "3.11.9", "packages": packages})
    with pytest.raises(ValueError, match="numpy"):
        runner.verify_runtime({"python": "3.12.99", "packages": {**packages, "numpy": "different"}})


def test_real_spawn_executes_two_synthetic_hgb_arms(tmp_path):
    """Actual spawned Python workers and real HGB fitting, using only generated rows."""
    root = tmp_path / "spawn"
    root.mkdir()
    reg = {"alpha": 0.01, "label_availability": "synthetic only",
           "roles": {"normal_calibration": ["C"], "forward_development": ["B"], "transport_external": []}}
    rng = np.random.default_rng(73)

    def fixture(name, n, include_attack):
        labels = np.arange(n) % 2 if include_attack else np.zeros(n, dtype=int)
        values = {feature: rng.uniform(0, 20, n) for feature in RAW_FEATURES}
        values.update(ip_protocol=np.full(n, 17), orig_port=np.full(n, 50000), resp_port=np.full(n, 443),
                      duration_seconds=rng.uniform(0, 1, n) + labels * 40)
        return pd.DataFrame({**values, "source_id": name, "row_id": np.arange(n),
                             "flow_end_ms": np.arange(n) + 1000, "label": labels,
                             "family": np.where(labels, "ExternalC2", "Benign"), "weight": np.ones(n)})

    training = fixture("T", 2048, True)
    runner._write_frame(root / "local/training/target.parquet", training)
    for name, n, attack in (("C", 200, False), ("B", 200, True)):
        frame = fixture(name, n, attack)
        obs = runner._write_frame(data.prepared(root, name), frame[["row_id", "flow_end_ms", *RAW_FEATURES]])
        truth_file = runner._write_frame(data.truth(root, name), frame[["row_id", "family"]])
        runner.stage(root, "prepare-" + name, "synthetic-spawn", lambda: [obs, truth_file])
    tasks = [{"id": "target-" + view + "-hgb", "support": "target", "view": view,
              "group": "hgb", "reuse": False} for view in ("raw8", "ports_zero8")]
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = [pool.submit(runner.arm_worker, root, reg, task, {}, 1, "cpu", "synthetic-spawn") for task in tasks]
        assert [job.result(timeout=90) for job in jobs] == [[], []]
    performance = [read_json(root / "evidence/performance" / (task["id"] + ".json")) for task in tasks]
    assert len({item["pid"] for item in performance}) == 2
    for task in tasks:
        report = read_json(root / "evidence/fits" / (task["id"] + ".json"))
        assert report["rows"] == 2048 and report["device"] == "cpu"
        assert (root / "evidence/evaluation" / task["id"] / "B.json").exists()
