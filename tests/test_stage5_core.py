"""Exercise final-evaluation ordering and crash recovery using only synthetic data."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ids.common import read_json, sha256_file, write_json

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("stage5_core_under_test", REPO / "workflows/stage5/core.py")
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    output, experiment = tmp_path / "final", tmp_path / "old"
    experiment.mkdir()
    data = {name: tmp_path / name for name in core.DATASETS}
    candidates, splits, calls = [], {}, []
    for ds, folder in data.items():
        folder.mkdir()
        splits[ds] = {}
        for split, hashes in {"train": ["train-hash", "dup"], "validation": ["val-hash", "other"],
                              "calibration": ["cal-hash", "c", "d", "e"],
                              "test": ["train-hash", "val-hash", "cal-hash", "new"]}.items():
            n = len(hashes)
            frame = pd.DataFrame({"sample_id": [f"{ds}-{split}-{i}" for i in range(n)],
                                  "feature_hash": hashes, "group_hash": hashes, "session_hash": hashes,
                                  "label": [0, 1] if n == 2 else [0, 0, 1, 1],
                                  "attack_cat": ["Normal", "Example"] if n == 2 else ["Normal", "Normal", "Example", "Rare"],
                                  "simulated_score": [.1, .7] if n == 2 else [.2, .2, .8, .9]})
            path = folder / (split + ".parquet")
            frame.to_parquet(path, index=False)
            splits[ds][split] = {"file": path.name, "sha256": sha256_file(path), "n_rows": n}
        write_json(folder / "audit.json", {"train_test_overlap": {"test_rows": 3, "n_feature_groups": 3}})
        for seed in (17, 29):
            candidates.append({"dataset": ds, "seed": seed, "model": "rf", "condition": "all",
                               "run_id": f"00-rf-s{seed}", "model_sha256": "model", "model_path": "unused",
                               "run_manifest_sha256": "run", "dataset_manifest_sha256": ds,
                               "deployment": {"raw_fields": 2}, "selected_features": ["simulated_score"]})
    def prepare(exp, inputs, cfg):
        return {"training_source_sha256": "training", "evaluator_source": {"sha256": "evaluator"},
                "threads": 1, "max_fpr": .01, "config": cfg, "candidates": copy.deepcopy(candidates),
                "partitions": splits}
    def load(directory, split):
        if split == "test":
            assert (output / core.START).is_file()
            barrier = read_json(output / core.BARRIER)
            assert len(barrier["thresholds"]) == len(candidates)
            assert all(sha256_file(output / rel) == row["sha256"] for rel, row in barrier["thresholds"].items())
        calls.append(("load", directory.name, split))
        return pd.read_parquet(directory / (split + ".parquet"))
    def score(exp, item, frame, threads):
        split = frame["sample_id"].iloc[0].split("-")[1]
        calls.append(("score", item["dataset"], item["run_id"], split))
        return frame["simulated_score"].to_numpy()
    monkeypatch.setattr(core, "_build_plan", prepare)
    monkeypatch.setattr(core, "_score", score)
    monkeypatch.setattr(core, "load_partition", load)
    monkeypatch.setattr(core, "_evaluator_source", lambda: {"sha256": "evaluator"})
    monkeypatch.setattr(core, "code_fingerprint", lambda: {"source_sha256": "training"})
    def run(resume=False, **kwargs):
        core.run_evaluation(experiment, data, output, kwargs or {"fixed": True}, resume=resume)
    return output, run, data, candidates, calls


def test_all_thresholds_sealed_before_any_test_and_overlap_scope(fixture):
    output, run, _, candidates, calls = fixture
    run()
    assert [x[2] for x in calls if x[0] == "load"] == ["calibration", "calibration", "test", "test"]
    assert core.verify_core(output)["n_candidates"] == 4
    for item in candidates:
        root = core._test_dir(output, item)
        row, scores = read_json(root / "metrics.json"), pd.read_parquet(root / "predictions.parquet")
        assert row["threshold"] == np.nextafter(.2, np.inf)
        assert scores["predicted_label"].tolist() == [0, 0, 1, 1]
        assert row["full"]["n_samples"] == 4
        if item["dataset"] == "unsw":
            assert scores["nonoverlap"].tolist() == [False, False, False, True]
            assert row["n_overlap"] == 3 and row["nonoverlap"]["n_samples"] == 1
        else:
            assert row["nonoverlap"] is None and "nonoverlap" not in scores
    with pytest.raises(FileExistsError):
        run()
    calls.clear()
    run(resume=True)
    assert calls == []
    with pytest.raises(ValueError, match="Frozen record differs"):
        run(resume=True, fixed=False)


def test_resume_after_interrupted_test_never_recalibrates_or_rescores_committed_run(fixture, monkeypatch):
    output, run, _, _, calls = fixture
    score = core._score
    def fail(exp, item, frame, threads):
        if item["run_id"].endswith("29") and "-test-" in frame["sample_id"].iat[0]:
            raise InterruptedError("worker interrupted")
        return score(exp, item, frame, threads)
    monkeypatch.setattr(core, "_score", fail)
    with pytest.raises(InterruptedError):
        run()
    assert (output / core.START).is_file() and not (output / core.DONE).exists()
    calls.clear()
    monkeypatch.setattr(core, "_score", score)
    run(resume=True)
    assert all("calibration" not in c for c in calls)
    assert ("score", "unsw", "00-rf-s17", "test") not in calls
    assert len([r for r in calls if r[0] == "score"]) == 3
    assert core.verify_core(output)["n_candidates"] == 4


@pytest.mark.parametrize("damage", ["threshold", "barrier", "start", "predictions"])
def test_incomplete_matrix_refuses_changed_or_missing_sealed_evidence(fixture, monkeypatch, damage):
    output, run, _, _, calls = fixture
    score = core._score
    def fail(exp, item, frame, threads):
        if item["dataset"] == "cicids" and "-test-" in frame["sample_id"].iat[0]:
            raise InterruptedError()
        return score(exp, item, frame, threads)
    monkeypatch.setattr(core, "_score", fail)
    with pytest.raises(InterruptedError):
        run()
    if damage == "threshold":
        path = output / "calibration/unsw/00-rf-s17.json"
        original = core._read_record(path)
        original["threshold"] = .5
        write_json(path, core._hash_record(original))
    elif damage in {"barrier", "start"}:
        (output / (core.BARRIER if damage == "barrier" else core.START)).unlink()
    else:
        path = output / "tests/unsw/00-rf-s17/predictions.parquet"
        path.write_bytes(b"changed")
    calls.clear()
    with pytest.raises(ValueError):
        run(resume=True)
    assert calls == []


def test_interrupted_checksum_publication_reverified_without_opening_data(fixture):
    output, run, _, _, calls = fixture
    run()
    (output / core.CHECKSUM).unlink()
    calls.clear()
    run(resume=True)
    assert calls == [] and (output / core.CHECKSUM).is_file()
    (output / core.CHECKSUM).write_text("wrong\n", encoding="ascii")
    with pytest.raises(ValueError, match="checksum"):
        run(resume=True)


def test_empty_sensitivity_subset_is_explicit(fixture, monkeypatch):
    output, run, data, _, _ = fixture
    original = core._source_feature_hashes
    monkeypatch.setattr(core, "_source_feature_hashes", lambda d, p: original(d, p) | {"new"})
    write_json(data["unsw"] / "audit.json", {"train_test_overlap": {"test_rows": 4, "n_feature_groups": 4}})
    run()
    metrics = read_json(output / "tests/unsw/00-rf-s17/metrics.json")
    assert metrics["nonoverlap"] is None and metrics["n_nonoverlap"] == 0
    assert metrics["nonoverlap_status"] == "empty_subset"

def test_interrupted_calibration_resumes_only_missing_models_and_only_scores_normals(fixture, monkeypatch):
    output, run, _, _, calls = fixture
    scorer = core._score
    blocked = True
    def may_interrupt(exp, item, frame, threads):
        split = frame["sample_id"].iat[0].split("-")[1]
        if split == "calibration":
            assert (frame["label"] == 0).all()
            if blocked and item["dataset"] == "cicids":
                raise InterruptedError()
        return scorer(exp, item, frame, threads)
    monkeypatch.setattr(core, "_score", may_interrupt)
    with pytest.raises(InterruptedError):
        run()
    assert all("test" not in x for x in calls)
    assert not (output / core.BARRIER).exists() and not (output / core.START).exists()
    blocked = False
    calls.clear()
    run(resume=True)
    assert not any(x[0] == "score" and x[1] == "unsw" and x[-1] == "calibration" for x in calls)
    assert len([x for x in calls if x[0] == "score" and x[-1] == "calibration"]) == 2


def test_normal_score_tamper_refused_before_any_additional_test_access(fixture, monkeypatch):
    output, run, _, _, calls = fixture
    scorer = core._score
    def stop(exp, item, frame, threads):
        if "-test-" in frame["sample_id"].iat[0]:
            raise InterruptedError()
        return scorer(exp, item, frame, threads)
    monkeypatch.setattr(core, "_score", stop)
    with pytest.raises(InterruptedError):
        run()
    path = output / "calibration/cicids/00-rf-s29.normal.parquet"
    normals = pd.read_parquet(path)
    normals["score"] = .99
    normals.to_parquet(path, index=False)
    calls.clear()
    with pytest.raises(ValueError, match="Calibration record identity"):
        run(resume=True)
    assert calls == []


def test_same_output_cannot_be_locked_concurrently(tmp_path):
    with core._exclusive(tmp_path):
        with pytest.raises(RuntimeError, match="Another evaluation"):
            with core._exclusive(tmp_path):
                pytest.fail("double entry")
    with core._exclusive(tmp_path):
        pass


def test_research_mode_rejects_arbitrary_configuration():
    with pytest.raises(ValueError, match="published fixed policy"):
        core._policy({"profile": "research", "candidate": "chosen later"})
