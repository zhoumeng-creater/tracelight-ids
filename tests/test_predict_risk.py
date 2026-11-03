"""Trusted-artifact mock replay and tiny CPU forward tests; no model training."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("predict_risk_script", REPO / "scripts/predict_risk.py")
predict = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(predict)


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf8")


class ToyPreprocessor:
    def transform(self, frame):
        return frame[["value"]].to_numpy(dtype=np.float32)


def bundle():
    model = torch.nn.Sequential(torch.nn.Linear(1, 1))
    with torch.no_grad():
        model[0].weight.fill_(1)
        model[0].bias.zero_()
    return SimpleNamespace(model_name="mlp", network_state=model.state_dict(), encoded_dim=1,
                           hidden_dims=(), batch_size=2, num_threads=1, preprocessor=ToyPreprocessor(),
                           feature_schema={"name": "toy", "features": ["value"], "categorical": [], "discrete": []},
                           predict_scores=lambda *_: pytest.fail("probability API must never be called"))


@pytest.fixture
def ready(tmp_path, monkeypatch):
    root, data = tmp_path / "research", tmp_path / "prepared"
    root.mkdir()
    data.mkdir()
    manifest = {"data_kind": "synthetic", "schema_version": "mock", "partitions": {"validation": {"n_rows": 4}}}
    put_json(data / "manifest.json", manifest)
    (data / "manifest.sha256").write_text(predict.sha256_file(data / "manifest.json") + "\n", encoding="ascii")
    put_json(root / "plan.json", {"config": {"conditions": ["hard"], "seeds": [17]}, "data": manifest})
    student = root / "students" / "hard-s17"
    threshold = float(np.nextafter(np.float64(1.0), np.inf))
    put_json(student / "run.json", {"condition": "hard", "seed": 17, "status": "diagnostic_no_eligible_checkpoint",
        "working_point": {"threshold": threshold, "decision_rule": "float64(raw_logit) >= frozen_raw_logit_threshold"}})
    frame = pd.DataFrame({"sample_id": ["c", "a", "d", "b"], "feature_hash": ["hc", "ha", "hd", "hb"],
                          "label": [1, 0, 1, 0], "attack_cat": ["attack", "normal", "attack", "normal"],
                          "value": [2., 0., 4., 1.]})
    identity = frame.set_index("sample_id", drop=False).loc[["b", "c", "a"]].drop(columns="value").reset_index(drop=True)
    identity.to_parquet(root / "validation-identity.parquet", index=False)
    calls = []

    def verify(path):
        calls.append(("verify", Path(path)))
        return {"status": "development_complete"}

    def load(data_path, split):
        calls.append(("load_partition", split))
        assert split == "validation"
        return frame.copy()

    def load_model(path, device):
        calls.append(("load_bundle", device))
        return bundle()

    monkeypatch.setattr(predict, "verify_risk_research", verify)
    monkeypatch.setattr(predict, "load_partition", load)
    monkeypatch.setattr(predict, "load_bundle", load_model)
    return root, data, frame, identity, calls


def args(ready, tmp_path):
    root, data, *_ = ready
    return {"research_dir": root, "condition": "hard", "seed": 17,
            "data_dir": data, "output": tmp_path / "out" / "prediction.parquet"}


def test_raw_float64_boundary_and_saved_cohort_order_no_probability_api(ready, tmp_path):
    options = args(ready, tmp_path)
    result = predict.predict_validation(**options)
    output = pd.read_parquet(options["output"])
    assert output["sample_id"].tolist() == ["b", "c", "a"]
    assert output["cpu_raw_logit"].dtype == np.float64
    assert output["cpu_raw_logit"].tolist() == [1., 2., 0.]
    assert output["predicted_label"].tolist() == [0, 1, 0]
    # Casting the frozen threshold to float32 incorrectly turns row b positive.
    assert bool(np.float32(1.) >= np.float32(result["threshold"]))
    assert not bool(np.float64(1.) >= result["threshold"])
    assert result["candidate_status"] == "diagnostic_no_eligible_checkpoint"
    assert result["data_kind"] == "synthetic" and result["reserved_partitions_read"] == []
    assert result["sha256"] == predict.sha256_file(options["output"])
    calls = ready[-1]
    assert [name for name, _ in calls] == ["verify", "load_partition", "load_bundle"]
    assert calls[-1][1] == "cpu"


@pytest.mark.parametrize("split", ["test", "calibration", "train"])
def test_any_nonvalidation_partition_rejected_before_artifact_or_data_read(ready, tmp_path, split):
    with pytest.raises(ValueError, match="validation only"):
        predict.predict_validation(**args(ready, tmp_path), split=split)
    assert not ready[-1]


def test_complete_directory_verification_precedes_model_loading(ready, tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "verify_risk_research", lambda *_: (_ for _ in ()).throw(ValueError("broken sealed artifact")))
    with pytest.raises(ValueError, match="broken sealed"):
        predict.predict_validation(**args(ready, tmp_path))
    assert not ready[-1]


def test_output_never_overwrites_or_modifies_sealed_inputs(ready, tmp_path):
    options = args(ready, tmp_path)
    options["output"].parent.mkdir()
    options["output"].write_bytes(b"keep existing")
    with pytest.raises(FileExistsError, match="exists"):
        predict.predict_validation(**options)
    assert options["output"].read_bytes() == b"keep existing"
    for parent in ready[:2]:
        with pytest.raises(ValueError, match="outside"):
            predict.predict_validation(**(options | {"output": parent / "new.parquet"}))
        assert not (parent / "new.parquet").exists()


@pytest.mark.parametrize("bad", ["condition", "seed", "threshold", "probability_contract", "manifest"])
def test_invalid_identity_or_threshold_or_data_binding_fail_closed(ready, tmp_path, bad):
    options = args(ready, tmp_path)
    root, data, *_ = ready
    if bad == "condition":
        options["condition"] = "../../hard"
    elif bad == "seed":
        options["seed"] = 29
    elif bad == "manifest":
        manifest = predict.read_json(data / "manifest.json") | {"data_kind": "different"}
        put_json(data / "manifest.json", manifest)
        (data / "manifest.sha256").write_text(predict.sha256_file(data / "manifest.json"), encoding="ascii")
    else:
        path = root / "students" / "hard-s17" / "run.json"
        record = predict.read_json(path)
        if bad == "threshold":
            record["working_point"]["threshold"] = float("nan")
        else:
            record["working_point"]["decision_rule"] = "probability >= threshold"
        put_json(path, record)
    with pytest.raises(ValueError):
        predict.predict_validation(**options)
    assert not options["output"].exists()
    assert not any(name == "load_bundle" for name, _ in ready[-1])


def test_cohort_identity_mismatch_and_duplicate_ids_rejected(ready):
    _, _, frame, identity, _ = ready
    changed = identity.copy()
    changed.loc[0, "feature_hash"] = "different"
    with pytest.raises(ValueError, match="differs"):
        predict._validation_cohort(frame, changed)
    duplicate = pd.concat([identity, identity.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        predict._validation_cohort(frame, duplicate)


def test_raw_forward_restores_threads_and_uses_unconverted_logits():
    saved_threads = torch.get_num_threads()
    frame = pd.DataFrame({"value": [-100., -1., 0., 1., 100.]})
    values = predict.raw_cpu_logits(bundle(), frame)
    np.testing.assert_array_equal(values, [-100., -1., 0., 1., 100.])
    assert torch.get_num_threads() == saved_threads


def test_nonmlp_model_refused():
    bad = copy.copy(bundle())
    bad.model_name = "hgb"
    with pytest.raises(ValueError, match="MLP"):
        predict.raw_cpu_logits(bad, pd.DataFrame({"value": [1.]}))


def test_direct_help_has_validation_only_without_probability_threshold_option(tmp_path):
    result = subprocess.run([sys.executable, str(REPO / "scripts/predict_risk.py"), "--help"],
                            cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--split {validation}" in result.stdout
    assert "--threshold" not in result.stdout and "--device" not in result.stdout
