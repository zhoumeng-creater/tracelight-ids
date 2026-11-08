"""Lab label workflow exercised with synthetic CSVs and random tiny weights."""
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
from ids.context_cache.label_inputs import ARMS, attack_union_log_odds
from ids.context_cache.label_protocol import frozen_template
from ids.context_cache.preprocessing import CSVPreprocessor
from test_scanning_workflow import official, tiny_protocol  # noqa: F401


@pytest.fixture
def label_protocol(tiny_protocol, tmp_path):  # noqa: F811 - imported pytest fixture
    # Reuse the existing binary synthetic reference and refine P metadata only.
    _, binary_spec, frames, prep, *reference = tiny_protocol
    spec = frozen_template()
    for key in set(spec) & set(binary_spec):
        if key not in {"protocol", "arms"}:
            spec[key] = copy.deepcopy(binary_spec[key])
    support = frames["support"].copy()
    attack_indices = support.index[support["label"] == 1]
    support.loc[attack_indices[::2], "attack_family"] = "DDoS"
    support_path = Path(spec["files"]["support"]["path"])
    support.to_csv(support_path, index=False)
    spec["files"]["support"]["sha256"] = sha256_file(support_path)
    spec["sham_class_rows"] = int(support["attack_family"].eq("Scanning").sum())
    frames = {**frames, "support": support}
    receipt = read_json(spec["artifacts"]["preparation"]["path"])
    receipt["files"] = copy.deepcopy(spec["files"])
    write_json(spec["artifacts"]["preparation"]["path"], receipt)
    spec["artifacts"]["preparation"]["sha256"] = sha256_file(spec["artifacts"]["preparation"]["path"])
    config = tmp_path / "label-protocol.json"
    write_json(config, spec)
    return config, spec, frames, prep, reference


def _forbidden(*args, **kwargs):
    raise AssertionError("No new preprocessing fit or artifact deserialization permitted")


def test_three_labels_same_inputs_saved_logits_and_cold_process(label_protocol, monkeypatch, tmp_path):
    import joblib
    import torch
    from ids.context_cache.label_workflow import run_label, verify_label_result
    from ids.context_cache.tabicl_adapter import FullCacheAdapter

    config, spec, frames, prep, _ = label_protocol
    built = []
    original_build = FullCacheAdapter.build

    def record_build(self, x, y):
        built.append((self.provenance["label_arm"], x.detach().cpu().numpy().copy(), y.clone()))
        return original_build(self, x, y)

    monkeypatch.setattr(FullCacheAdapter, "build", record_build)
    monkeypatch.setattr(CSVPreprocessor, "fit", _forbidden)
    output = tmp_path / "comparison"
    report = run_label(config, output)
    assert [arm for arm, _, _ in built] == list(ARMS)
    assert report["workload"] == {"full_context_builds": 3, "calibration_scored_rows": 297,
                                   "evaluation_scored_rows": 72, "optimizer_updates": 0, "qcoc_compilations": 0}
    expected_x = prep.transform(frames["support"])
    for _, matrix, _ in built:
        assert matrix[0].tobytes() == expected_x.tobytes()
    assert torch.equal(built[0][2][0], torch.from_numpy(frames["support"]["label"].to_numpy()))
    assert (built[1][2][0].numpy() == 1).sum() == spec["sham_class_rows"]
    assert torch.equal(torch.bincount(built[1][2][0]), torch.bincount(built[2][2][0]))
    assert read_json(output / "baseline_replay.json")["passed"]
    scores = pd.read_csv(output / "evaluation_scores.csv", float_precision="round_trip")
    assert "candidate_gate" in report and "semantic_vs_sham_paired_changes" in report
    for arm in ARMS:
        manifest = read_json(output / arm / "deployment.json")
        registration = read_json(output / arm / "threshold.json")
        c = pd.read_csv(output / f"{arm}_calibration.csv", float_precision="round_trip")
        assert registration["rank"] == 99 and not c["predicted_attack"].astype(bool).any()
        z = scores[[f"{arm}_logit_{i}" for i in range(manifest["num_classes"])]].to_numpy(np.float32)
        np.testing.assert_array_equal(attack_union_log_odds(z).astype(np.float64), scores[f"{arm}_score"])
        assert manifest["score_contract"]["engine"] == "numpy-float32-logaddexp-v1"
    with monkeypatch.context() as no_load:
        no_load.setattr(torch, "load", _forbidden)
        no_load.setattr(joblib, "load", _forbidden)
        no_load.setattr(CSVPreprocessor, "load", _forbidden)
        assert verify_label_result(output)["model_loads"] == 0
    # Move only deployment dependencies; neither P nor query labels go with it.
    cold = tmp_path / "cold"
    (cold / "shared").mkdir(parents=True)
    shutil.copyfile(output / "shared/preprocessor.joblib", cold / "shared/preprocessor.joblib")
    feature_csv = cold / "features.csv"
    frames["evaluation"].loc[:, spec["schema"]["features"]].to_csv(feature_csv, index=False)
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src") + os.pathsep + str(repo / "local/vendor/tabicl/src")
    for arm in ARMS:
        shutil.copytree(output / arm, cold / arm)
        prediction = cold / f"{arm}.csv"
        command = [sys.executable, "-c", "from ids.context_cache.label_workflow import LabelDetector; "
                   "import sys; LabelDetector.load(sys.argv[1]).predict_csv(sys.argv[2],sys.argv[3])",
                   str(cold / arm), str(feature_csv), str(prediction)]
        completed = subprocess.run(command, cwd=cold, env=env, capture_output=True, text=True,
                                   encoding="utf-8", timeout=60)
        assert completed.returncode == 0, completed.stderr[-4000:]
        actual = pd.read_csv(prediction, float_precision="round_trip")
        np.testing.assert_array_equal(actual["score"], scores[f"{arm}_score"])
        np.testing.assert_array_equal(actual["predicted_attack"].astype(bool), scores[f"{arm}_predicted"])
    with pytest.raises(FileExistsError):
        run_label(config, output)


@pytest.mark.parametrize("mismatch", ["score", "threshold", "decision"])
def test_baseline_failure_blocks_both_refined_builds(label_protocol, monkeypatch, tmp_path, mismatch):
    import ids.context_cache.label_workflow as workflow
    from ids.context_cache.tabicl_adapter import FullCacheAdapter

    config, _, _, _, reference = label_protocol
    _, old_e, old_registration = reference
    builds = []
    original_build = FullCacheAdapter.build

    def record_build(self, *args, **kwargs):
        builds.append(self.provenance["label_arm"])
        return original_build(self, *args, **kwargs)

    monkeypatch.setattr(FullCacheAdapter, "build", record_build)
    if mismatch == "threshold":
        register = workflow.register_threshold

        def altered(*args, **kwargs):
            result = register(*args, **kwargs)
            result["threshold"] += 1
            return result

        monkeypatch.setattr(workflow, "register_threshold", altered)
    else:
        scorer = workflow._score_matrix
        calls = []

        def altered(*args, **kwargs):
            scores, logits = scorer(*args, **kwargs)
            calls.append(len(scores))
            if len(calls) == 2:
                if mismatch == "score":
                    scores[1] += 1
                else:
                    t = old_registration["threshold"]
                    scores[0] = np.nextafter(t, -np.inf if old_e[0] > t else np.inf)
            return scores, logits

        monkeypatch.setattr(workflow, "_score_matrix", altered)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="T0 baseline replay failed|registered threshold differs"):
        workflow.run_label(config, output)
    assert builds == ["baseline"]
    assert not (output / "semantic_family").exists() and not (output / "grouped_sham").exists()
    assert read_json(output / "state.json")["status"] == "failed"


def test_changed_support_label_artifact_rejected_without_deserialization(label_protocol, monkeypatch, tmp_path):
    import joblib
    import torch
    from ids.context_cache.label_workflow import run_label, verify_label_result

    config, *_ = label_protocol
    output = tmp_path / "comparison"
    run_label(config, output)
    monkeypatch.setattr(torch, "load", _forbidden)
    monkeypatch.setattr(joblib, "load", _forbidden)
    path = output / "support_labels.csv"
    frame = pd.read_csv(path)
    index = frame.index[frame["binary_label"].eq(1)][0]
    frame.loc[index, "grouped_sham"] = 3 - frame.loc[index, "grouped_sham"]
    frame.to_csv(path, index=False)
    report = read_json(output / "report.json")
    report["files"]["support_labels.csv"] = sha256_file(path)
    write_json(output / "report.json", report)
    with pytest.raises(ValueError, match="fixed whole-group construction"):
        verify_label_result(output)


@pytest.mark.parametrize("mutation", ["score_contract", "class_count", "wrong_arm"])
def test_deployment_contract_rejected_before_model_load(label_protocol, monkeypatch, tmp_path, mutation):
    import joblib
    import torch
    from ids.context_cache.label_workflow import LabelDetector, run_label

    config, *_ = label_protocol
    output = tmp_path / "comparison"
    run_label(config, output)
    path = output / "semantic_family/deployment.json"
    manifest = read_json(path)
    if mutation == "score_contract":
        manifest["score_contract"]["temperature"] = .9
    elif mutation == "class_count":
        manifest["num_classes"] = 2
    else:
        manifest["arm"] = "grouped_sham"
    write_json(path, manifest)
    monkeypatch.setattr(torch, "load", _forbidden)
    monkeypatch.setattr(joblib, "load", _forbidden)
    with pytest.raises(ValueError, match="contract|identity"):
        LabelDetector.load(path.parent)


def test_verifier_rejects_modified_logits_even_with_updated_digest(label_protocol, monkeypatch, tmp_path):
    from ids.context_cache.label_workflow import run_label, verify_label_result

    config, *_ = label_protocol
    output = tmp_path / "comparison"
    run_label(config, output)
    path = output / "evaluation_scores.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    frame.loc[0, "semantic_family_logit_0"] += 1
    frame.to_csv(path, index=False)
    report = read_json(output / "report.json")
    report["files"]["evaluation_scores.csv"] = sha256_file(path)
    write_json(output / "report.json", report)
    with pytest.raises(ValueError, match="saved scores differ"):
        verify_label_result(output)


def test_cold_loader_rejects_valid_but_unbound_batch_size(label_protocol, tmp_path):
    from ids.context_cache.label_workflow import LabelDetector, run_label

    config, *_ = label_protocol
    output = tmp_path / "comparison"
    run_label(config, output)
    path = output / "semantic_family/deployment.json"
    manifest = read_json(path)
    manifest["query_batch_size"] += 1
    write_json(path, manifest)
    # The value is a valid integer, but disagrees with hash-bound model provenance.
    with pytest.raises(ValueError, match="label cache input/class contract differs"):
        LabelDetector.load(path.parent)
