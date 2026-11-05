"""Synthetic binding/dispatch checks; no archived model inference or training."""
import copy

import numpy as np
import pandas as pd
import pytest
import torch

import ids.unsw_diagnostic_experiment as impl
from ids.common import canonical_hash, read_json, sha256_file, write_json
from ids.data import CATEGORICAL, FEATURES, SCHEMA_VERSION
from ids.models import FeaturePreprocessor, ModelBundle, _network, save_bundle
from ids.risk_research import _state, _state_hash
from ids.unsw_kd import build_targets


@pytest.fixture
def bound(tmp_path, monkeypatch):
    monkeypatch.setattr(impl, "_check_native_dependency", lambda: "1.2.10")
    screen, previous, data = (tmp_path / k for k in ("screen", "round10", "data"))
    for p in (screen, previous, data):
        p.mkdir()
    schema = {"name": "synthetic-diagnostic-unit", "features": list(FEATURES),
              "categorical": list(CATEGORICAL), "discrete": list(CATEGORICAL)}
    frames = {}
    for role, n in (("fit", 24), ("reference", 16), ("constraint", 256)):
        y = np.arange(n) % 2
        features = {k: np.where(y, "a", "b") if k in CATEGORICAL else np.arange(n, dtype=float)
                    for k in FEATURES}
        frames[role] = pd.DataFrame({**features,
            "label": y, "attack_cat": np.where(y, "Attack", "Normal"),
            "sample_id": [f"{role}-{i}" for i in range(n)],
            "feature_hash": [f"{role}-hash-{i}" for i in range(n)]})
        frames[role][["sample_id", "feature_hash", "label", "attack_cat"]].to_parquet(
            previous / f"{role}-identity.parquet", index=False)
    train = pd.concat(frames.values(), ignore_index=True)
    train.to_parquet(data / "train.parquet", index=False)
    write_json(data / "audit.json", {})
    manifest = {"schema_version": SCHEMA_VERSION, "dataset_id": "synthetic-unit",
                "data_kind": "mirror", "features": schema["features"],
                "categorical_features": list(CATEGORICAL), "partitions": {
                    "train": {"file": "train.parquet", "sha256": sha256_file(data / "train.parquet"),
                              "n_rows": len(train)},
                    "validation": {"file": "NEVER_READ_VALIDATION.parquet"},
                    "calibration": {"file": "NEVER_READ_CALIBRATION.parquet"},
                    "test": {"file": "NEVER_READ_TEST.parquet"}},
                "audit": {"file": "audit.json", "sha256": sha256_file(data / "audit.json")}}
    write_json(data / "manifest.json", manifest)
    (data / "manifest.sha256").write_text(sha256_file(data / "manifest.json"))
    prep = FeaturePreprocessor.fit(frames["fit"], schema["features"], schema)
    net = _network(prep.encoded_dim, (4, 2))
    bundle = ModelBundle("mlp", prep, schema["features"], prep.encoded_dim, [],
                         hidden_dims=(4, 2), network_state=_state(net), feature_schema=schema)
    for p in (screen / "warmups/all-small-s17", previous / "arms/H", previous / "arms/K"):
        p.mkdir(parents=True)
        save_bundle(bundle, p / "model.joblib")
    for arm in ("H", "K"):
        write_json(previous / f"arms/{arm}/run.json", {"final_state_sha256": _state_hash(bundle.network_state),
                   "endpoint": {"working_point": {"threshold": .1}}})
        write_json(previous / f"arms/{arm}/history.json", [
            {"step": i + 1, "hard_bce": .7, "objective_bce": .7} for i in range(40)])
    for name in ("teachers/all-s17", "native-trees/catboost-reference-s17"):
        (screen / name).mkdir(parents=True)
        write_json(screen / name / "run.json", {"selection": {"working_point": {"threshold": .2}}})
    logits = np.linspace(-2, 2, len(frames["fit"]))
    np.savez(screen / "teachers/all-s17/training-scores.npz", fit=logits,
             test=np.asarray(["RESERVED_ARRAY_MUST_NOT_BE_DECODED"], dtype=object))
    _, _, targets = build_targets(frames["fit"], logits)
    np.savez(previous / "targets.npz", **targets)
    original = {"student": {"hidden": [4, 2], "input_features": prep.encoded_dim,
                "parameters": sum(p.numel() for p in net.parameters()),
                "anchor_state_sha256": _state_hash(bundle.network_state)},
                "data": {"preprocessing_semantic_sha256": canonical_hash(impl.preprocessing_state(prep))}}
    write_json(previous / "protocol.json", original)
    write_json(previous / "preflight.json", {"data_manifest": manifest})
    ids = frames["constraint"].sample_id.tolist()
    batches = {str(size): {"sample_ids": [ids[:size]] * 75} for size in (1, 256)}
    write_json(previous / "summary.json", {"cost": {"timing": {"batches": batches}}})
    write_json(previous / "UNSW_KD_COMPLETE.json", {"status": "unsw_kd_complete", "student_fits": 4,
        "files": {f"arms/{arm}/history.json": sha256_file(previous / f"arms/{arm}/history.json")
                  for arm in ("H", "K")}})
    files = []
    for alias, root in (("screen", screen), ("round10", previous)):
        for p in root.rglob("*"):
            if p.is_file() and p.name != "history.json":
                files.append({"root_alias": alias, "path": p.relative_to(root).as_posix(),
                              "sha256": sha256_file(p), "bytes": p.stat().st_size})
    binding = {"protocol_id": "synthetic-unit", "current_fit_budget": 0, "files": files,
               "prepared_contract": {"data_manifest": manifest},
               "data_roles": {"F": 24, "R": 16, "C": 256}}
    binding_file = tmp_path / "binding.json"
    write_json(binding_file, binding)
    protocol = read_json(impl.PROTOCOL)
    protocol["protocol_id"] = "synthetic-unit"
    protocol_file = tmp_path / "diagnostic-protocol.json"
    write_json(protocol_file, protocol)
    monkeypatch.setattr(impl, "BINDING", binding_file)
    monkeypatch.setattr(impl, "PROTOCOL", protocol_file)
    return screen, previous, data, binding


def test_preflight_uses_only_train_and_cached_F_without_network_forward(bound, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Preflight must never invoke a network, optimizer, or model prediction")
    monkeypatch.setattr(torch.nn.Linear, "forward", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    monkeypatch.setattr(ModelBundle, "predict_scores", forbidden)
    receipt = impl.preflight_diagnostic(*bound[:3])
    assert receipt["status"] == "preflight_complete"
    assert receipt["new_inference_rows"] == receipt["optimizer_updates"] == receipt["student_fits"] == 0
    assert receipt["reserved_partitions_read"] == []
    assert receipt["initial_anchor_losses"] == {"hard_bce": .7, "mixed_bce": .7}


@pytest.mark.parametrize("change", ["model", "history", "train", "manifest", "target"])
def test_preflight_rejects_changed_bound_inputs_before_any_inference(bound, change):
    screen, previous, data, _ = bound
    paths = {"model": screen / "warmups/all-small-s17/model.joblib",
             "history": previous / "arms/H/history.json", "train": data / "train.parquet",
             "manifest": data / "manifest.json", "target": previous / "targets.npz"}
    with paths[change].open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError):
        impl.preflight_diagnostic(screen, previous, data)


def test_external_binding_override_is_rejected(bound, tmp_path):
    alternate = tmp_path / "alternate.json"
    changed = copy.deepcopy(bound[3])
    changed["current_fit_budget"] = 2
    write_json(alternate, changed)
    with pytest.raises(ValueError, match="checked-in"):
        impl.preflight_diagnostic(*bound[:3], binding_path=alternate)


def test_existing_output_preserved_before_loading_or_cuda(bound, tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "retain.txt").write_text("retain")
    monkeypatch.setattr(impl, "_bound_inputs", lambda *a, **k: pytest.fail("loaded inputs"))
    with pytest.raises(FileExistsError):
        impl.run_diagnostic(*bound[:3], output)
    assert (output / "retain.txt").read_text() == "retain"


def test_production_does_not_allow_cpu_override(bound, tmp_path, monkeypatch):
    monkeypatch.setattr(impl, "_bound_inputs", lambda *a, **k: pytest.fail("loaded inputs"))
    with pytest.raises(ValueError, match="4090D"):
        impl.run_diagnostic(*bound[:3], tmp_path / "output", device="cpu")


@pytest.mark.parametrize("installed", [None, "1.2.9"])
def test_native_dependency_checked_without_importing_or_loading_model(monkeypatch, installed):
    def metadata(name):
        assert name == "catboost"
        if installed is None:
            raise impl.PackageNotFoundError(name)
        return installed
    monkeypatch.setattr(impl, "version", metadata)
    with pytest.raises(ValueError, match="catboost==1.2.10"):
        impl._check_native_dependency()


def _execution_context(bound):
    context = impl._bound_inputs(*bound[:3])
    context["protocol"]["budget"]["F_forward_rows"] = 3 * len(context["frames"]["fit"])
    context["protocol"]["budget"]["anchor_backward_calls"] = 4
    bundle = context["bundles"]["anchor"]
    net = _network(bundle.encoded_dim, bundle.hidden_dims)
    net.load_state_dict(bundle.network_state)
    with torch.no_grad():
        z = net(torch.from_numpy(bundle.preprocessor.transform(context["frames"]["fit"]))).numpy().ravel()
    z = z.astype(float)
    weights = context["targets"]["weights"]
    context["receipt"]["initial_anchor_losses"] = {
        name: float(np.sum(weights * (np.logaddexp(0, z) - target * z)))
        for name, target in (("hard_bce", context["targets"]["labels"]),
                             ("mixed_bce", context["targets"]["target_K"]))}
    return context


def _fake_cost(context):
    labels = list(context["model_specs"])
    batches = {size: {"sample_ids": item["sample_ids"], "execution_order": [labels] * 75,
                     **{name: {"durations_ms": [1.] * 75, "blocks": [{}, {}, {}]} for name in labels}}
               for size, item in context["timing_batches"].items()}
    return {"status": "complete", "batches": batches,
            "budget": {"formal_calls": 450, "formal_rows": 57825, "warmup_calls": 30, "warmup_rows": 3855,
                       "rss_processes": 3, "rss_forward_calls": 3, "rss_forward_rows": 768,
                       "new_fits": 0, "optimizer_updates": 0},
            "rss": {name: {"model": name, "pid": 100 + i, "threads": 4, "forward_calls": 1,
                           "forward_rows": 256, "sample_ids": batches["256"]["sample_ids"][0],
                           "rss_after_import_bytes": 100, "rss_after_load_bytes": 200,
                           "rss_after_probe_bytes": 300, "model_load_ms": 1.}
                    for i, name in enumerate(labels)}}


@pytest.mark.parametrize("invalid", [False, True])
def test_synthetic_execution_keeps_complete_invalid_result_and_zero_updates(bound, tmp_path, monkeypatch, invalid):
    import ids.unsw_diagnostic_cost as costs
    context = _execution_context(bound)
    if invalid:
        context["receipt"]["initial_anchor_losses"]["mixed_bce"] += .01
    cost = _fake_cost(context)
    def measure(specs, frame, timing, output):
        assert specs == context["model_specs"] and timing == context["timing_batches"]
        assert frame.equals(context["frames"]["constraint"])
        write_json(output / "cost-summary.json", cost)
        return cost
    monkeypatch.setattr(costs, "measure_cost", measure)
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *a, **k: pytest.fail("Optimizer update forbidden"))
    output = tmp_path / "execution"
    result = impl._execute(context, output, device="cpu", source={"files": {}})
    assert result["status"] == "unsw_diagnostic_complete"
    assert result["validity_passed"] is (not invalid)
    assert result["optimizer_updates"] == result["student_fits"] == result["teacher_fits"] == 0
    assert result["F"]["budget"]["forward_rows"] == 72
    done = read_json(output / impl.COMPLETE)
    assert done["validity_passed"] is (not invalid)
    for name, digest in done["files"].items():
        assert sha256_file(output / name) == digest
    assert (output / "REPORT.md").is_file()


@pytest.mark.parametrize("corruption", ["extra_call", "wrong_order", "nonfinite", "rss", "state"])
def test_acceptance_rejects_budget_and_integrity_errors(bound, tmp_path, corruption):
    from ids.unsw_diagnostic_f import diagnose_f
    context = _execution_context(bound)
    f = diagnose_f(context["bundles"], context["frames"]["fit"], context["targets"], tmp_path / "F", "cpu")
    cost = _fake_cost(context)
    if corruption == "extra_call":
        cost["budget"]["formal_calls"] += 1
    elif corruption == "wrong_order":
        cost["batches"]["1"]["execution_order"][0] = ["H40"] * 3
    elif corruption == "nonfinite":
        cost["batches"]["1"]["H40"]["durations_ms"][0] = float("nan")
    elif corruption == "rss":
        cost["rss"]["H40"]["forward_rows"] = 257
    else:
        f["models"]["K40"]["state_checks"]["unchanged"] = False
    assert not all(impl._acceptance(context, f, cost).values())


def test_technical_failure_retains_F_and_never_marks_complete(bound, tmp_path, monkeypatch):
    import ids.unsw_diagnostic_cost as costs
    context = _execution_context(bound)
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic CPU failure")
    monkeypatch.setattr(costs, "measure_cost", fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="synthetic CPU"):
        impl._execute(context, output, device="cpu", source={"files": {}})
    assert (output / "F/fit-scores.parquet").is_file()
    assert (output / "FAILURE.json").is_file()
    assert not (output / impl.COMPLETE).exists()
