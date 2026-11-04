"""Deterministic guard termination and retained Phase-I artifact regressions."""
import copy

import pytest
import torch

from ids.common import read_json, sha256_file, write_json
from ids.models import load_bundle
from ids.upgrade_research import run_upgrade_research, verify_upgrade_research
from test_upgrade_research import config, source as _source_fixture


@pytest.fixture
def synthetic_source(tmp_path, monkeypatch):
    return _source_fixture.__wrapped__(tmp_path, monkeypatch)


def _configuration():
    value = config()
    value["max_fpr"] = .25
    value["conditions"] = ["guarded_hard"]
    value["optimization"].update(attack_loss_limit=1., macro_steps=9)
    return value


def _reseal(root):
    complete = read_json(root / "UPGRADE_COMPLETE.json")
    complete["files"] = {p.relative_to(root).as_posix(): sha256_file(p)
                         for p in root.rglob("*") if p.is_file() and p.name != "UPGRADE_COMPLETE.json"}
    write_json(root / "UPGRADE_COMPLETE.json", complete)


def test_full_rejection_attempted_once_keeps_step0_and_optimizer(synthetic_source, tmp_path, monkeypatch):
    import ids.upgrade_research as upgrade
    data, v3 = synthetic_source
    original_oracle, original_guard = upgrade._oracle, upgrade.guarded_adamw_step
    baseline = None
    optimizer_transactions = []

    def reject_moved_parameters(network, *args):
        nonlocal baseline
        state = upgrade._state(network)
        if baseline is None:
            baseline = state
        result = original_oracle(network, *args)
        unchanged = all(torch.equal(value, baseline[name]) for name, value in state.items())
        # This synthetic callback makes EVERY nonzero proposal inadmissible.
        # It exercises the transaction/termination contract, not IDS quality.
        result["feasible"] = unchanged
        result["metrics"]["synthetic_forced_rejection"] = not unchanged
        return result

    def tracked_guard(network, optimizer, **kwargs):
        before = copy.deepcopy(optimizer.state_dict())
        weights = upgrade._state(network)
        report = original_guard(network, optimizer, **kwargs)
        assert not report["accepted"]
        # This is the first optimizer transaction, so full rollback removes
        # its just-created moments/step as well as restoring parameter groups.
        assert before["state"] == {} and optimizer.state_dict()["state"] == {}
        assert optimizer.state_dict()["param_groups"] == before["param_groups"]
        assert all(torch.equal(value, weights[name]) for name, value in upgrade._state(network).items())
        optimizer_transactions.append(report)
        return report

    monkeypatch.setattr(upgrade, "_oracle", reject_moved_parameters)
    monkeypatch.setattr(upgrade, "guarded_adamw_step", tracked_guard)
    root = tmp_path / "stalled"
    result = run_upgrade_research(v3, data, _configuration(), root, device="cpu")
    assert result["artifact_schema_version"] == 2
    assert len(optimizer_transactions) == 1
    run = read_json(root / "students/guarded_hard-s17/run.json")
    history = read_json(root / "students/guarded_hard-s17/history.json")
    assert run["planned_updates"] == 9 and run["attempted_updates"] == 1
    assert run["termination_reason"] == "deterministic_guard_rejection"
    assert run["selected_step"] == 0 and run["selected_is_step0"]
    assert run["accepted_updates"] == 0 and run["accepted_updates_with_parameter_movement"] == 0
    assert run["common_initial_state_sha256"] == run["selected_state_sha256"]
    assert [step["step"] for step in history] == [0, 1]
    assert history[-1]["termination_reason"] == "deterministic_guard_rejection"
    saved = load_bundle(root / "students/guarded_hard-s17/model.joblib", device="cpu")
    assert all(torch.equal(value, baseline[name]) for name, value in saved.network_state.items())
    assert verify_upgrade_research(root) == result


def test_exhausted_budget_is_reported_without_guard_stop(synthetic_source, tmp_path):
    data, v3 = synthetic_source
    value = _configuration()
    value["conditions"] = ["hard"]
    value["optimization"]["macro_steps"] = 2
    root = tmp_path / "normal"
    run_upgrade_research(v3, data, value, root, device="cpu")
    run = read_json(root / "students/hard-s17/run.json")
    assert run["attempted_updates"] == run["planned_updates"] == 2
    assert run["termination_reason"] == "budget_exhausted"


def test_failed_phase1_checkpoint_saved_and_bound_even_without_students(synthetic_source, tmp_path, monkeypatch):
    import ids.upgrade_research as upgrade
    data, v3 = synthetic_source
    original = upgrade._oracle

    def infeasible(*args):
        result = original(*args)
        result["feasible"] = False
        return result

    monkeypatch.setattr(upgrade, "_oracle", infeasible)
    value = _configuration()
    value["optimization"]["phase1_steps"] = 1
    root = tmp_path / "failed-phase"
    result = run_upgrade_research(v3, data, value, root, device="cpu")
    assert result["artifact_schema_version"] == 2 and result["n_students"] == 0
    phase = read_json(root / "phase1-s17.json")
    assert phase["status"] == "no_feasible_start"
    assert phase["checkpoint_file"] == "phase1-s17.joblib"
    assert phase["checkpoint_sha256"] == sha256_file(root / phase["checkpoint_file"])
    model = load_bundle(root / phase["checkpoint_file"], device="cpu")
    assert model.history == [] and model.training_statistics["status"] == "no_feasible_start"
    assert not model.training_statistics["new_source_anchor_fit"]
    assert upgrade._state_hash(model.network_state) == phase["warm_state_sha256"]
    (root / phase["checkpoint_file"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact mismatch"):
        verify_upgrade_research(root)


def test_artifact_schema_v1_compatibility_v2_requirement_and_unknown_rejection(synthetic_source, tmp_path):
    data, v3 = synthetic_source
    value = _configuration()
    value["conditions"] = ["hard"]
    value["optimization"]["macro_steps"] = 1
    root = tmp_path / "schemas"
    run_upgrade_research(v3, data, value, root, device="cpu")
    original_complete = read_json(root / "UPGRADE_COMPLETE.json")

    unknown = copy.deepcopy(original_complete)
    unknown["artifact_schema_version"] = 3
    write_json(root / "UPGRADE_COMPLETE.json", unknown)
    with pytest.raises(ValueError, match="unsupported.*schema"):
        verify_upgrade_research(root)

    # Produce the original v1 artifact shape, with no Phase-I checkpoint.
    write_json(root / "UPGRADE_COMPLETE.json", original_complete)
    phase = read_json(root / "phase1-s17.json")
    (root / phase.pop("checkpoint_file")).unlink()
    phase.pop("checkpoint_sha256")
    write_json(root / "phase1-s17.json", phase)
    _reseal(root)
    missing_v2 = read_json(root / "UPGRADE_COMPLETE.json")
    with pytest.raises(ValueError, match="Phase-I checkpoint"):
        verify_upgrade_research(root)
    missing_v2.pop("artifact_schema_version")
    write_json(root / "UPGRADE_COMPLETE.json", missing_v2)
    legacy = verify_upgrade_research(root)
    assert "artifact_schema_version" not in legacy
