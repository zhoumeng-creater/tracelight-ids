"""Runner phase and pairing contracts with synthetic receipts, never real fits."""
from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from ids.common import read_json, write_json
from ids.normal_subgroup_training import protocol, runner
from ids.normal_subgroup_training.selection import VALIDATION_STEPS


def pair(arm, seed=17):
    result = {name: "same" for name in (
        "original_artifact_sha256", "preflight_manifest_sha256", "preprocessing_sha256",
        "numeric_bins_sha256", "F_identity_sha256", "V_identity_sha256", "initial_state_sha256",
        "initial_dropout_rng_sha256", "initial_sampler_rng_sha256")}
    result.update(target_arm=arm, seed=seed, dropout_seed=seed, sampler_seeds={"A0": seed},
                  optimizer_initial_state_entries=0, cpu_score_batch_size=256,
                  validation_steps=list(VALIDATION_STEPS), normal_target_sha256=arm,
                  draw_chain_by_validation_step={str(s): str(s) for s in VALIDATION_STEPS},
                  dropout_rng_by_validation_step={str(s): str(s) for s in VALIDATION_STEPS},
                  selection_rng_unchanged_by_step={str(s): True for s in VALIDATION_STEPS})
    return result


def test_warmstart_pair_accepts_different_normal_targets_and_requires_complete_budget():
    left, right = pair("transport_target"), pair("identity_target")
    result = runner.check_pairing(left, right)
    assert result["passed"] and result["common_validation_steps"] == list(VALIDATION_STEPS)
    # No random-initialization seed is needed: same original state is the initialization.
    for field in ("draw_chain_by_validation_step", "dropout_rng_by_validation_step",
                  "selection_rng_unchanged_by_step"):
        altered = deepcopy(right)
        altered[field].pop(str(VALIDATION_STEPS[-1]))
        with pytest.raises(ValueError, match="sixteen-event|Every CPU"):
            runner.check_pairing(left, altered)


@pytest.mark.parametrize("field", ["V_identity_sha256", "preflight_manifest_sha256", "original_artifact_sha256"])
def test_pair_rejects_different_reference_contract(field):
    left, right = pair("transport_target"), pair("identity_target")
    right[field] = "different"
    with pytest.raises(ValueError, match="contract differs"):
        runner.check_pairing(left, right)


def test_pair_rejects_nonfresh_optimizer_even_if_both_sides_match():
    left, right = pair("transport_target"), pair("identity_target")
    left["optimizer_initial_state_entries"] = right["optimizer_initial_state_entries"] = 1
    with pytest.raises(ValueError, match="fresh optimizer"):
        runner.check_pairing(left, right)


def test_cross_host_compute_rejected_before_source_or_model_use(monkeypatch):
    monkeypatch.setattr(protocol, "scoring_host", lambda: {"hostname": "lab"})
    monkeypatch.setattr(protocol, "specifications", lambda: pytest.fail("Cross-host replay must fail first"))
    with pytest.raises(ValueError, match="scoring host differs"):
        protocol.verify_live_code({"scoring_host": {"hostname": "laptop"}})


def test_matching_host_still_requires_frozen_code(monkeypatch):
    registration, design, code = {"v": 1}, {"v": 2}, {"a.py": "hash"}
    monkeypatch.setattr(protocol, "scoring_host", lambda: {"hostname": "lab"})
    monkeypatch.setattr(protocol, "specifications", lambda: (registration, design))
    monkeypatch.setattr(protocol, "code_snapshot", lambda: code)
    value = {"scoring_host": {"hostname": "lab"}, "registration": registration, "design": design, "code": code}
    protocol.verify_live_code(value)
    with pytest.raises(ValueError, match="changed during"):
        protocol.verify_live_code(dict(value, code={"a.py": "changed"}))


def test_training_threads_must_match_single_thread_selector(tmp_path):
    with pytest.raises(ValueError, match="threads=1"):
        runner.initialize(tmp_path, tmp_path / "archive", 4, "cuda:0")


def test_code_freeze_includes_training_default_bindings():
    from ids.normal_subgroup_training.bindings import DEFAULT_BINDINGS
    assert DEFAULT_BINDINGS in protocol.source_paths()


def all_costs():
    return {f"{arm}_s{seed}": {"deployment_contract_pass": True}
            for seed in (17, 29, 43) for arm in ("transport_target", "identity_target")}


def test_selected_control_must_pass_all_its_deployment_checks():
    comparison = {"adoption_pass": False, "retained_development_arm": "identity_target"}
    costs = all_costs()
    assert runner.final_outcome(comparison, costs)["retained_development_arm"] == "identity_target"
    costs["identity_target_s29"]["deployment_contract_pass"] = False
    result = runner.final_outcome(comparison, costs)
    assert result["retained_development_arm"] == "R30"
    assert result["selected_arm_deployment_contract_pass"] is False
    assert not result["adoption_pass"]
    # Effect evidence is preserved; only the final retained arm changes.
    assert comparison["retained_development_arm"] == "identity_target"


def test_transport_still_requires_all_six_costs_but_control_uses_its_corresponding_three():
    costs = all_costs()
    transport = {"adoption_pass": True, "retained_development_arm": "transport_target"}
    assert runner.final_outcome(transport, costs)["adoption_pass"]
    costs["identity_target_s17"]["deployment_contract_pass"] = False
    result = runner.final_outcome(transport, costs)
    assert not result["adoption_pass"] and result["retained_development_arm"] == "R30"
    costs = all_costs()
    costs["transport_target_s17"]["deployment_contract_pass"] = False
    control = {"adoption_pass": False, "retained_development_arm": "identity_target"}
    result = runner.final_outcome(control, costs)
    assert result["retained_development_arm"] == "identity_target"
    assert result["selected_arm_deployment_contract_pass"] and not result["deployment_contract_pass"]


def test_deployment_costs_cannot_rescue_failed_effect_conditions_or_missing_reports():
    costs = all_costs()
    comparison = {"adoption_pass": False, "retained_development_arm": "R30"}
    result = runner.final_outcome(comparison, costs)
    assert result["retained_development_arm"] == "R30" and not result["adoption_pass"]
    costs.pop("identity_target_s43")
    with pytest.raises(ValueError, match="all six"):
        runner.final_outcome(comparison, costs)


def freeze_fixture(root, monkeypatch):
    tasks = [{"id": f"{arm}_s{seed}", "arm": arm, "seed": seed}
             for seed in (17, 29, 43) for arm in ("transport_target", "identity_target")]
    configuration = {"tasks": tasks, "registration": {"design_sha256": "design",
                                                      "input_bindings_sha256": "bindings"}}
    monkeypatch.setattr(runner, "verify_receipt", lambda *args, **kwargs: None)
    (root / "preflight").mkdir()
    (root / "preflight/output_manifest.json").write_text("{}", encoding="utf-8")
    for task in tasks:
        folder = root / "models" / task["id"] / "trajectory"
        folder.mkdir(parents=True)
        write_json(folder / "summary.json", {"status": "complete", "arm": task["arm"],
                   "seed": task["seed"], "fit_calls": 1, "final_refits": 0,
                   "optimizer_steps": 2224, "selected_step": 139, "completed_validation_events": 16})
        write_json(folder / "history.json", [{"step": s} for s in VALIDATION_STEPS])
        write_json(folder / "pairing.json", pair(task["arm"], task["seed"]))
        for name in ("best.pt", "model.pt"):
            (folder / name).write_bytes(b"synthetic frozen model")
        for name in ("selected_metrics.json", "baseline_metrics.json"):
            write_json(folder / name, {"synthetic": True})
    return configuration


def test_global_freeze_has_six_artifacts_and_three_passed_pairs(tmp_path, monkeypatch):
    configuration = freeze_fixture(tmp_path, monkeypatch)
    runner.freeze_models(tmp_path, configuration)
    frozen = runner.verify_freeze(tmp_path, configuration)
    assert len(frozen["models"]) == 6
    assert all(v["passed"] for v in frozen["paired_contracts"].values())
    assert frozen["calibration_and_evaluation_scoring_started"] is False
    frozen["paired_contracts"]["17"]["passed"] = False
    write_json(tmp_path / "freeze.json", frozen)
    with pytest.raises(ValueError, match="three complete paired"):
        runner.verify_freeze(tmp_path, configuration)


def test_incomplete_trajectory_cannot_create_ce_freeze(tmp_path, monkeypatch):
    configuration = freeze_fixture(tmp_path, monkeypatch)
    path = tmp_path / "models/identity_target_s43/trajectory/summary.json"
    value = read_json(path)
    value["optimizer_steps"] = 2223
    write_json(path, value)
    with pytest.raises(ValueError, match="trajectory budget"):
        runner.freeze_models(tmp_path, configuration)
    assert not (tmp_path / "freeze.json").exists()


def test_failed_root_cannot_invoke_another_training_or_cost_worker(tmp_path):
    write_json(tmp_path / "status.json", {"status": "failed"})
    with pytest.raises(ValueError, match="Failed attempts"):
        runner.train_task(tmp_path, "transport_target_s17")
    with pytest.raises(ValueError, match="Failed attempts"):
        runner.cost_worker(tmp_path, "transport_target_s17")


def test_completed_trajectory_reuses_receipt_without_fitting_again(tmp_path, monkeypatch):
    from ids.normal_subgroup_training import training
    task = {"id": "transport_target_s17", "arm": "transport_target", "seed": 17}
    configuration = {"tasks": [task], "registration": {"input_bindings_file": "bindings.json"},
                     "archive": str(tmp_path / "archive"), "threads": 1, "device": "cuda:0", "design": {}}
    write_json(tmp_path / "protocol.json", configuration)
    write_json(tmp_path / "bindings.json", {"models": {"protocol_blind_s17": {"path": "original.pt"}}})
    monkeypatch.setattr(runner, "REPO", tmp_path)
    monkeypatch.setattr(runner, "verify_live_code", lambda p: None)
    monkeypatch.setattr(runner, "verify_runtime", lambda *args: {})
    real_verify = runner.verify_receipt
    monkeypatch.setattr(runner, "verify_receipt", lambda root, receipt, binding=None:
                        None if receipt.name == "preflight.json" else real_verify(root, receipt, binding))
    monkeypatch.setattr(pd, "read_parquet", lambda p: pd.DataFrame({"synthetic": [1]}))
    fits = []

    def fit(frames, preflight, artifact, folder, **kwargs):
        fits.append(kwargs)
        folder.mkdir(parents=True)
        write_json(folder / "summary.json", {"status": "complete"})
        return {"status": "complete"}

    monkeypatch.setattr(training, "fit_trajectory", fit)
    runner.train_task(tmp_path, task["id"])
    runner.train_task(tmp_path, task["id"])
    assert len(fits) == 1
    write_json(tmp_path / "models" / task["id"] / "trajectory/summary.json", {"status": "changed"})
    with pytest.raises(ValueError, match="Completed stage artifact changed"):
        runner.train_task(tmp_path, task["id"])
    assert len(fits) == 1
