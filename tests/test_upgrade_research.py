"""Synthetic V4 integration and working-point/finite-event regression checks."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from ids.common import read_json, sha256_file
from ids.data import make_synthetic_dataset, prepare_dataset
from ids.risk_distillation import RiskConstraint, build_risk_reference
from ids.risk_research import _fixed_metrics, run_risk_research
from ids.upgrade_research import (_network_from, _oracle, empirical_constraints, run_upgrade_research,
                                  validate_upgrade_config, verify_upgrade_research)

REPO = Path(__file__).resolve().parents[1]


def config():
    return yaml.safe_load((REPO / "configs/upgrade-v4-smoke.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def source(tmp_path, monkeypatch):
    import ids.research as research_module
    import ids.risk_research as v3_module
    raw, data, v3 = tmp_path / "raw", tmp_path / "prepared", tmp_path / "v3"
    make_synthetic_dataset(raw, n_train=900, n_test=120)
    prepare_dataset(raw / "UNSW_NB15_training-set.csv", raw / "UNSW_NB15_testing-set.csv", data, data_kind="synthetic")
    vc = yaml.safe_load((REPO / "configs/risk-v3-smoke.yaml").read_text(encoding="utf-8"))
    vc.update(conditions=["hard"], max_fpr=.25)
    # Tiny synthetic families cannot allow even one lost TP at 1%. Exercise
    # the full pipeline under a declared loose TOY budget, not a real IDS claim.
    vc["optimization"]["attack_loss_limit"] = 1.
    def zero_output_anchor(*args, **kwargs):
        bundle = research_module.train_research_mlp(*args, **kwargs)
        # A deliberately tied synthetic incumbent: safe step0 even when no
        # upgraded model qualifies. Hidden features remain nonconstant.
        keys = list(bundle.network_state)
        bundle.network_state[keys[-2]].zero_()
        bundle.network_state[keys[-1]].zero_()
        return bundle

    monkeypatch.setattr(v3_module, "train_research_mlp", zero_output_anchor)
    run_risk_research(data, vc, v3, device="cpu")
    (data / "calibration.parquet").unlink()
    (data / "test.parquet").unlink()
    return data, v3


def test_teacher_diagnostic_rejects_float32_plateau_nextafter():
    raw = np.array([-13.815509796142578], dtype=np.float32)
    threshold = np.nextafter(np.float64(raw[0]), np.inf)
    frame = pd.DataFrame({"label": [0], "attack_cat": ["Normal"]})
    assert bool((raw >= threshold)[0])  # original NumPy weak-scalar cast bug
    metric = _fixed_metrics(frame, raw, threshold)
    assert metric["false_positives"] == 0 and metric["fpr"] == 0


def test_finite_guard_uses_integer_budget_and_original_tp_not_surrogate():
    anchor = torch.tensor([-1., -1., .1, .1, .1, .1], dtype=torch.float64)
    labels = torch.tensor([0, 0, 1, 1, 1, 1])
    reference = build_risk_reference(anchor, labels,
        [RiskConstraint("normal", "normal_fpr", .5), RiskConstraint("attack", "attack_loss", .25)],
        sample_ids=[f"i{i}" for i in range(6)])
    # One of four ORIGINAL TPs lost; normal FP one of two. Exactly allowed.
    report = empirical_constraints(np.array([1., -1., -1., .001, .001, .001]), 0., reference)
    assert report["feasible"]
    assert report["constraints"]["normal"]["allowed_count"] == 1
    assert report["constraints"]["attack"]["count"] == 1
    assert not empirical_constraints(np.array([1., -1., -1., -1., .001, .001]), 0., reference)["feasible"]
    with pytest.raises(ValueError, match="finite numeric"):
        empirical_constraints(np.zeros(6), float("nan"), reference)


def test_cpu_oracle_needs_no_validation_and_preserves_rng():
    from types import SimpleNamespace
    from ids.models import _network

    net = _network(1, (1,))
    with torch.no_grad():
        net[0].weight.fill_(1.)
        net[0].bias.zero_()
        net[2].weight.fill_(1.)
        net[2].bias.zero_()
    anchor = SimpleNamespace(encoded_dim=1, hidden_dims=(1,), network_state=net.state_dict())
    net = _network_from(anchor, "cpu")
    xs = {"fit": np.array([[0.]], dtype=np.float32), "reference": np.array([[-1.], [-1.]], dtype=np.float32),
          "constraint": np.array([[-1.], [1.]], dtype=np.float32)}
    frames = {"reference": pd.DataFrame({"label": [0, 0]})}  # deliberately has NO V
    reference = build_risk_reference(torch.tensor([-1., 1.]), torch.tensor([0, 1]),
        [RiskConstraint("normal", "normal_fpr", 0.), RiskConstraint("attack", "attack_loss", 0.)], sample_ids=["n", "a"])
    rng = torch.random.get_rng_state().clone()
    report = _oracle(net, xs, frames, 0., reference, .01, 2)
    assert report["feasible"]
    assert torch.equal(rng, torch.random.get_rng_state())


def test_all_eight_conditions_reuse_roles_and_raw_targets_without_reserved_reads(source, tmp_path, monkeypatch):
    import ids.upgrade_research as upgrade
    data, v3 = source
    original = upgrade.load_partition
    reads = []

    def load(path, split):
        reads.append(split)
        assert split in {"train", "validation"}
        return original(path, split)

    monkeypatch.setattr(upgrade, "load_partition", load)
    import ids.models as models
    monkeypatch.setattr(models, "train_model", lambda *a, **k: pytest.fail("v4 refitted a teacher"))
    before = {p.relative_to(v3).as_posix(): sha256_file(p) for p in v3.rglob("*") if p.is_file()}
    cfg = config()
    cfg["max_fpr"] = .25
    cfg["optimization"]["attack_loss_limit"] = 1.
    output = tmp_path / "v4"
    result = run_upgrade_research(v3, data, cfg, output, device="cpu")
    assert reads == ["train", "validation"]
    assert result["n_students_planned"] == 8
    assert result["n_students"] == 8  # fixture must exercise every real loss/guard path
    assert result["n_teacher_fits"] == 0 and result["n_anchor_fits"] == 0
    assert {p.relative_to(v3).as_posix(): sha256_file(p) for p in v3.rglob("*") if p.is_file()} == before
    runs = read_json(output / "summary.json")["runs"]
    assert len({r["common_initial_state_sha256"] for r in runs}) == 1
    assert all(r["working_point_regime"] == "recalibrated_reference" for r in runs)
    for r in runs:
        history = read_json(output / f"students/{r['condition']}-s17/history.json")
        assert history[0]["step"] == 0 and history[0]["losses"] is None
        assert "gradient_diagnostics" in history[1]["losses"]
        if r["condition"].startswith("guarded_"):
            assert all(step["constraints"]["feasible"] for step in history)
        if r["selected_step"] is not None:
            assert history[r["selected_step"]]["eligible"]
    assert run_upgrade_research(v3, data, cfg, output, device="cpu", resume=True) == result
    assert reads == ["train", "validation"]
    with pytest.raises(FileExistsError):
        run_upgrade_research(v3, data, cfg, output, device="cpu")
    changed = config()
    changed["max_fpr"] = .25
    changed["optimization"]["attack_loss_limit"] = 1.
    changed["optimization"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="identity differs"):
        run_upgrade_research(v3, data, changed, output, device="cpu", resume=True)
    (output / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact mismatch"):
        verify_upgrade_research(output)


def test_no_feasible_phase1_is_explicit_complete_diagnostic_not_success(source, tmp_path, monkeypatch):
    import ids.upgrade_research as upgrade
    data, v3 = source
    original = upgrade._oracle

    def infeasible(*args):
        result = original(*args)
        result["feasible"] = False
        return result

    monkeypatch.setattr(upgrade, "_oracle", infeasible)
    monkeypatch.setattr(upgrade, "_fit_condition", lambda *a: pytest.fail("infeasible start trained a student"))
    cfg = config()
    cfg["max_fpr"] = .25
    cfg["optimization"]["attack_loss_limit"] = 1.
    cfg["optimization"]["phase1_steps"] = 1
    root = tmp_path / "infeasible"
    result = run_upgrade_research(v3, data, cfg, root, device="cpu")
    assert result["n_students"] == 0 and result["n_skipped_no_feasible_start"] == 8
    assert {r["status"] for r in read_json(root / "summary.json")["runs"]} == {"no_feasible_start"}
    assert not list((root / "students").rglob("model.joblib"))


@pytest.mark.parametrize("change", [
    {"purpose": "final_test"}, {"primary_working_point": "validation"}, {"seeds": [17, 17]},
    {"conditions": ["invented"]}, {"max_fpr": True}, {"extra": 3},
])
def test_invalid_config_rejected(change):
    cfg = config()
    cfg.update(change)
    with pytest.raises(ValueError):
        validate_upgrade_config(cfg)


def test_cannot_relax_old_budgets_or_overwrite_source(source, tmp_path):
    data, v3 = source
    cfg = config()
    cfg["max_fpr"] = .25
    cfg["optimization"]["attack_loss_limit"] = .02
    with pytest.raises(ValueError, match="budgets"):
        run_upgrade_research(v3, data, cfg, tmp_path / "relaxed", device="cpu")
    with pytest.raises(ValueError, match="non-nested"):
        run_upgrade_research(v3, data, cfg, v3 / "v4", device="cpu")
