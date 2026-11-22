"""Generated-array checks for snapshots, diagnostics and exact continuation."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from scripts.comprehensive_batch import training
from scripts.comprehensive_batch.matrix import build_plan


class TinyHeads(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.dropout = torch.nn.Dropout(.1)

    def forward(self, numeric, categorical):
        return self.linear(self.dropout(numeric)).unsqueeze(-1)


def inputs():
    numeric = np.random.default_rng(4).normal(size=(12, 3)).astype(np.float32)
    categorical = np.zeros((12, 1), dtype=np.int64)
    labels = np.array([0] * 8 + [1] * 4, dtype=np.float32)
    groups = {"references": [{"id": "ref1", "indices": [0, 1, 2, 3]},
                              {"id": "ref2", "indices": [4, 5, 6, 7]}],
              "targets": [{"id": "t1", "indices": [0, 1, 4, 5]},
                          {"id": "t2", "indices": [2, 3, 6, 7]}],
              "attacks": [8, 9, 10, 11]}
    spec = dict(training.FIXED_TRAINING, batch_size=4, draws_per_class=2,
                score_batch_size=4, report_every=2, recovery_every=2)
    return numeric, categorical, labels, groups, spec


def job(arm):
    return {"job_id": "synthetic_" + arm, "arm": arm, "seed": 17,
            "learning_rate": .002, "auxiliary_weight": 1.,
            "optimizer_steps": 4, "checkpoints": [1, 2, 4]}


def deterministic_history(history):
    result = deepcopy(history)
    for record in result:
        record["cumulative_cost"] = {key: value for key, value in record["cumulative_cost"].items()
                                     if not key.endswith("_seconds")}
    return result


@pytest.mark.parametrize("arm", training.ARMS)
def test_exact_resume_matches_uninterrupted_for_every_objective(tmp_path, arm):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    base = TinyHeads()
    numeric, categorical, labels, groups, spec = inputs()
    one, two = deepcopy(base), deepcopy(base)
    args = dict(job=job(arm), device="cpu", training_spec=spec)
    expected = training.train_loop(one, numeric, categorical, labels, groups,
                                   output_dir=tmp_path / "one", **args)
    with pytest.raises(InterruptedError):
        training.train_loop(two, numeric, categorical, labels, groups,
                            output_dir=tmp_path / "two", interrupt_after_step=3, **args)
    # Construction and unrelated randomness between attempts cannot affect recovery.
    torch.rand(19)
    restored = TinyHeads()
    actual = training.train_loop(restored, numeric, categorical, labels, groups,
                                 output_dir=tmp_path / "two", resume=True, **args)
    assert training.state_digest(one.state_dict()) == training.state_digest(restored.state_dict())
    for key in ("sampling", "exposure", "initial_state_sha256", "initial_dropout_rng_sha256",
                "final_dropout_rng_sha256", "sampler_draw_sha256", "diagnostic_points"):
        assert actual[key] == expected[key]
    assert deterministic_history(actual["history"]) == deterministic_history(expected["history"])
    assert actual["diagnostic_points"] == [0, 1, 2, 4]
    assert [r["step"] for r in actual["snapshots"]] == [1, 2, 4]
    assert actual["cost"]["transfer_discovery_forward_rows"] == (0 if arm == "A_BCE" else 4 * len(labels))
    for a, b in zip(actual["snapshots"], expected["snapshots"]):
        assert a["state_sha256"] == b["state_sha256"]
        left = np.load(tmp_path / "two" / a["F_scores"])
        right = np.load(tmp_path / "one" / b["F_scores"])
        np.testing.assert_array_equal(left["scores"], right["scores"])
        np.testing.assert_array_equal(left["exposure"], right["exposure"])
        diag = json.loads((tmp_path / "two" / a["diagnostics"]).read_text())
        assert set(diag["objectives"]) == set(training.ARMS)
        assert "BCE_auxiliary_cosine" in diag["gradient_probe"]
        assert set(diag["gradient_probe"]["unweighted_components_at_same_parameters"]) == {"D", "H", "G"}
        assert set(diag["gradient_probe"]["component_cosines"]) == {"D_H", "D_G"}
        assert training.file_digest(tmp_path / "two" / a["F_scores"]) == a["F_scores_sha256"]
    assert json.loads((tmp_path / "two" / "snapshots.json").read_text()) == actual["snapshots"]


def test_probe_restores_gradient_mode_rng_and_never_updates_parameters():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = TinyHeads().train()
    numeric, categorical, labels, groups, _ = inputs()
    for p in model.parameters():
        p.grad = torch.full_like(p, 3.)
    before = training.state_digest(model.state_dict())
    rng = torch.get_rng_state().clone()
    discovered = training.score_full(model, numeric, categorical, device="cpu", batch_size=4)
    result = training.gradient_probe(model, numeric, categorical, labels, groups,
                                    job("R_reference_envelope"), device=torch.device("cpu"),
                                    batch_size=4, score_pass=discovered)
    assert model.training and torch.equal(rng, torch.get_rng_state())
    assert before == training.state_digest(model.state_dict())
    assert all(torch.equal(p.grad, torch.full_like(p, 3.)) for p in model.parameters())
    assert result["BCE_forward_rows"] == len(labels)


def test_resume_rejects_changed_recipe_and_retains_earlier_snapshots(tmp_path):
    numeric, categorical, labels, groups, spec = inputs()
    chosen = job("R_reference_envelope")
    with pytest.raises(InterruptedError):
        training.train_loop(TinyHeads(), numeric, categorical, labels, groups,
                            job=chosen, device="cpu", output_dir=tmp_path,
                            training_spec=spec, interrupt_after_step=2)
    before = (tmp_path / "recovery.pt").read_bytes()
    with pytest.raises(ValueError, match="changed"):
        training.train_loop(TinyHeads(), numeric, categorical, labels, groups,
                            job=dict(chosen, learning_rate=.0005), device="cpu", output_dir=tmp_path,
                            training_spec=spec, resume=True)
    assert before == (tmp_path / "recovery.pt").read_bytes()
    snapshots = json.loads((tmp_path / "snapshots.json").read_text())
    assert [i["step"] for i in snapshots] == [1, 2]


def test_production_rejects_modified_budget_before_constructing_model(tmp_path, monkeypatch):
    chosen = dict(build_plan()["tabm_jobs"][0], optimizer_steps=4, checkpoints=[2, 4])
    monkeypatch.setattr(training, "make_tabm_network", lambda *a, **k: pytest.fail("real model construction"))
    with pytest.raises(ValueError, match="registered"):
        training.fit([], None, tmp_path, job=chosen, device="cpu")


def test_initial_diagnostic_failure_has_recoverable_zero_step(tmp_path, monkeypatch):
    numeric, categorical, labels, groups, spec = inputs()
    torch.manual_seed(31)
    base = TinyHeads()
    args = dict(job=job("R_reference_envelope"), device="cpu", training_spec=spec)
    expected = training.train_loop(deepcopy(base), numeric, categorical, labels, groups,
                                   output_dir=tmp_path / "full", **args)
    original = training.gradient_probe
    monkeypatch.setattr(training, "gradient_probe", lambda *a, **k: (_ for _ in ()).throw(
        OSError("synthetic initialization interruption")))
    with pytest.raises(OSError):
        training.train_loop(deepcopy(base), numeric, categorical, labels, groups,
                            output_dir=tmp_path / "retry", **args)
    saved = torch.load(tmp_path / "retry" / "recovery.pt", weights_only=True)
    assert saved["completed_step"] == 0 and saved["diagnostic_points"] == []
    monkeypatch.setattr(training, "gradient_probe", original)
    actual = training.train_loop(TinyHeads(), numeric, categorical, labels, groups,
                                 output_dir=tmp_path / "retry", resume=True, **args)
    assert actual["state_sha256"] == expected["state_sha256"]
    assert deterministic_history(actual["history"]) == deterministic_history(expected["history"])


def test_initial_checkpoint_partial_write_can_resume_without_any_updates(tmp_path, monkeypatch):
    numeric, categorical, labels, groups, spec = inputs()
    torch.manual_seed(71)
    base = TinyHeads()
    args = dict(job=job("S_alignment_group"), device="cpu", training_spec=spec)
    expected = training.train_loop(deepcopy(base), numeric, categorical, labels, groups,
                                   output_dir=tmp_path / "full", **args)
    original = training._save

    def fail_initial_commit(path, value):
        assert path.name == "recovery.pt" and value["completed_step"] == 0
        path.with_name(path.name + ".part").write_bytes(b"interrupted initial checkpoint")
        raise OSError("synthetic first recovery write interruption")

    monkeypatch.setattr(training, "_save", fail_initial_commit)
    retry = tmp_path / "retry"
    with pytest.raises(OSError, match="first recovery"):
        training.train_loop(deepcopy(base), numeric, categorical, labels, groups,
                            output_dir=retry, **args)
    assert [p.name for p in retry.iterdir()] == ["recovery.pt.part"]
    # Production fit also permits its bound initialization metadata here.
    for name in ("job.json", "preprocessing.json", "groups.json"):
        (retry / name).write_text("{}", encoding="utf8")
    monkeypatch.setattr(training, "_save", original)
    actual = training.train_loop(deepcopy(base), numeric, categorical, labels, groups,
                                 output_dir=retry, resume=True, **args)
    assert actual["state_sha256"] == expected["state_sha256"]
    assert deterministic_history(actual["history"]) == deterministic_history(expected["history"])
    assert actual["sampling"] == expected["sampling"]
    assert actual["final_dropout_rng_sha256"] == expected["final_dropout_rng_sha256"]
    assert not (retry / "recovery.pt.part").exists()
    for one, two in zip(actual["snapshots"], expected["snapshots"]):
        np.testing.assert_array_equal(np.load(retry / one["F_scores"])["scores"],
                                      np.load(tmp_path / "full" / two["F_scores"])["scores"])


@pytest.mark.parametrize("existing", ["history.json", "snapshots/step000000/model.pt"])
def test_missing_recovery_with_training_artifacts_never_restarts(tmp_path, existing):
    numeric, categorical, labels, groups, spec = inputs()
    artifact = tmp_path / existing
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"preserve prior training")
    (tmp_path / "recovery.pt.part").write_bytes(b"partial")
    with pytest.raises(ValueError, match="initialization-only"):
        training.train_loop(TinyHeads(), numeric, categorical, labels, groups,
                            job=job("S_alignment_group"), device="cpu", output_dir=tmp_path,
                            training_spec=spec, resume=True)
    assert artifact.read_bytes() == b"preserve prior training"
    assert not (tmp_path / "recovery.pt").exists()


@pytest.mark.parametrize("failure", ["interrupted", "numerical"])
def test_budget_and_failure_costs_persist_without_return_or_recovery_file(tmp_path, monkeypatch, failure):
    numeric, categorical, labels, groups, spec = inputs()
    chosen = job("A_BCE")
    if failure == "numerical":
        original = training.tabm_head_loss
        calls = 0

        def fail_third_update(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise FloatingPointError("synthetic numerical failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(training, "tabm_head_loss", fail_third_update)
    error = FloatingPointError if failure == "numerical" else InterruptedError
    with pytest.raises(error):
        training.train_loop(TinyHeads(), numeric, categorical, labels, groups,
                            job=chosen, device="cpu", output_dir=tmp_path, training_spec=spec,
                            interrupt_after_step=3 if failure == "interrupted" else None)
    # Return packages omit optimizer recovery state, but retain report costs.
    persisted = json.loads((tmp_path / "cost.json").read_text())
    snapshots = json.loads((tmp_path / "snapshots.json").read_text())
    history = json.loads((tmp_path / "history.json").read_text())
    assert persisted["step"] == 2
    assert [item["step"] for item in snapshots] == [1, 2]
    assert persisted["cost"] == history[-1]["cumulative_cost"] == snapshots[-1]["cost"]
    assert persisted["cost"]["bce_forward_rows"] == 2 * spec["batch_size"]
    assert persisted["cost"]["active_optimizer_seconds"] > 0
    assert persisted["cost"]["diagnostic_seconds"] > 0
    for item in snapshots:
        cost_file = tmp_path / item["cost_record"]
        record = json.loads(cost_file.read_text())
        assert record == {"job_id": chosen["job_id"], "step": item["step"], "cost": item["cost"]}
        assert training.file_digest(cost_file) == item["cost_record_sha256"]
        assert item["cost"]["bce_forward_rows"] == item["step"] * spec["batch_size"]
        assert item["cost"]["diagnostic_score_rows"] == (item["step"] + 1) * len(labels)
        model = torch.load(tmp_path / item["path"], weights_only=True)
        assert "cost" not in model and "cumulative_cost" not in model
    assert snapshots[0]["cost"]["bce_forward_rows"] < snapshots[1]["cost"]["bce_forward_rows"]
