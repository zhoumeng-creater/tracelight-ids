"""Synthetic runner loss-contract checks, with mocked export and no data files.

The teacher is an explicit array; no teacher fit, prepared dataset, checkpoint
download or real-data experiment occurs. Full pipeline tests live elsewhere.
"""
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from ids import risk_research as runner  # noqa: E402
from ids.risk_distillation import freeze_margin_threshold  # noqa: E402


class _SyntheticPreprocessor:
    def transform(self, frame):
        return np.ascontiguousarray(frame.loc[:, ["toy0", "toy1"]].to_numpy(dtype=np.float32))


class _ExportOnlyBundle:
    """Only the export interface used by _student, not the deployed bundle API."""

    def __init__(self, model_name, preprocessor, selected_features, encoded_dim, history, *, network_state,
                 hidden_dims, **kwargs):
        self.preprocessor, self.encoded_dim, self.hidden_dims = preprocessor, encoded_dim, hidden_dims
        self.network_state = network_state

    def predict_scores(self, frame):
        network = runner._network(self.encoded_dim, self.hidden_dims)
        network.load_state_dict(self.network_state)
        network.eval()
        with torch.inference_mode():
            return torch.sigmoid(network(torch.from_numpy(self.preprocessor.transform(frame))).squeeze(1)).numpy()


@pytest.fixture
def synthetic_case(monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    rng = np.random.default_rng(271828)
    labels = {"fit": [0, 0, 0, 1, 1, 0, 1, 0], "constraint": [0, 1, 0, 1, 0, 1],
              "reference": [0, 0, 0, 0, 1, 1, 1, 1], "validation": [0, 1, 0, 1, 0, 1]}
    frames = {}
    for name, y in labels.items():
        x = rng.uniform(-.8, .9, size=(len(y), 2)).astype(np.float32)
        frames[name] = pd.DataFrame({"toy0": x[:, 0], "toy1": x[:, 1], "label": y,
                                     "attack_cat": ["toy_attack" if value else "toy_normal" for value in y],
                                     "sample_id": [f"synthetic-{name}-{i}" for i in range(len(y))],
                                     "feature_hash": [f"synthetic-feature-{name}-{i}" for i in range(len(y))]})
    # Known frozen anchor; no anchor training occurs in this loss-focused test.
    network = runner._network(2, (3,))
    with torch.no_grad():
        network[0].weight.copy_(torch.tensor([[.5, -.2], [-.1, .4], [.3, .2]]))
        network[0].bias.copy_(torch.tensor([.1, .2, .05]))
        network[2].weight.copy_(torch.tensor([[.4, -.3, .2]]))
        network[2].bias.fill_(.05)
    anchor = SimpleNamespace(preprocessor=_SyntheticPreprocessor(), selected_features=["toy0", "toy1"],
                             encoded_dim=2, hidden_dims=(3,), network_state=runner._state(network),
                             feature_schema={"name": "synthetic_loss_contract_only"})
    teacher_logits = np.array([-.7, .4, 1.1, -.2, .8, -.6, .2, 1.3], dtype=np.float64)
    # Both folds have unequal class composition; teacher working points differ.
    folds = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    grids = np.array([[.9, .2, -.3], [.6, -.1, -.5]], dtype=np.float64)
    teacher = (teacher_logits, folds, grids, {"scope": "explicit_synthetic_arrays"})
    config = {"max_fpr": .25, "folds": 2, "threads": 1, "optimization": {
        "macro_steps": 2, "chunk_size": 2, "learning_rate": .02, "weight_decay": .01,
        "temperature": 1.7, "kd_weight": .2, "fd_alpha": 1., "fd_beta": 5., "fd_temperature": 2.3,
        "ekd_temperature": .4, "margin_floor": .1, "dual_step_size": .2, "dual_maximum": 10.,
        "attack_loss_limit": .1, "gradient_clip": 1e8, "min_family_support": 1}}
    exports = {}

    def save_export(bundle, path):
        exports[path.parent.name] = {key: value.detach().cpu().clone()
                                     for key, value in bundle.network_state.items()}
        path.write_bytes(b"synthetic-export-interface-mock")

    monkeypatch.setattr(runner, "ModelBundle", _ExportOnlyBundle)
    monkeypatch.setattr(runner, "save_bundle", save_export)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, **kwargs: path.write_bytes(b"mock"))
    monkeypatch.setattr(runner, "calibrate_threshold", lambda *args: {"threshold": .5})
    monkeypatch.setattr(runner, "evaluate_binary", lambda *args: {"scope": "export_diagnostics_mock"})
    yield frames, anchor, teacher, config, exports
    torch.set_num_threads(previous_threads)


def _run(case, tmp_path, condition):
    frames, anchor, teacher, config, _ = case
    return runner._student(frames["fit"], frames["constraint"], frames["reference"], frames["validation"],
                           anchor, teacher, config, 17, condition, tmp_path / condition, "cpu")


def _capture_step(monkeypatch):
    captured = []

    class CaptureWithoutUpdating:
        def __init__(self, parameters, **kwargs):
            self.parameters = list(parameters)

        def zero_grad(self, set_to_none=True):
            for parameter in self.parameters:
                parameter.grad = None

        def step(self):
            captured.append([parameter.grad.detach().clone() for parameter in self.parameters])

    monkeypatch.setattr(torch.optim, "AdamW", CaptureWithoutUpdating)
    return captured


def _full_population_oracle(case, condition):
    """One unchunked equation, independent of _student's accumulation code."""
    frames, anchor, teacher, config, _ = case
    op, chunk = config["optimization"], config["optimization"]["chunk_size"]
    net = runner._network(anchor.encoded_dim, anchor.hidden_dims)
    net.load_state_dict(anchor.network_state)
    x = {role: anchor.preprocessor.transform(frame) for role, frame in frames.items()}
    y = {role: torch.tensor(frame.label.to_numpy(), dtype=torch.float32) for role, frame in frames.items()}
    # Calibration is a detached function of the frozen R anchor, never of F/C.
    reference_logits = runner._infer(net, x["reference"], "cpu", chunk)
    record = freeze_margin_threshold(torch.tensor(reference_logits), y["reference"], config["max_fpr"])
    threshold = record["threshold"]
    zf = net(torch.from_numpy(x["fit"])).squeeze(1)
    zc = net(torch.from_numpy(x["constraint"])).squeeze(1)
    all_y = torch.cat((y["fit"], y["constraint"]))
    positive = (all_y == 0).sum().float() / (all_y == 1).sum()
    hard = (F.binary_cross_entropy_with_logits(zf, y["fit"], pos_weight=positive, reduction="sum")
            + F.binary_cross_entropy_with_logits(zc, y["constraint"], pos_weight=positive, reduction="sum")) / len(all_y)
    teacher_logits, fold, grids, _ = teacher
    t = torch.tensor(teacher_logits)
    row_grid = torch.tensor(grids[fold])
    teacher_margin = t - row_grid[:, 1]
    beta, temp = op["kd_weight"], op["temperature"]
    if condition == "ekd":
        rates = [config["max_fpr"] / 2, config["max_fpr"], min(config["max_fpr"] * 2, .99)]
        student_grid = torch.tensor([freeze_margin_threshold(torch.tensor(reference_logits), y["reference"], rate)["threshold"]
                                     for rate in rates], dtype=torch.float64)
        tau = op["ekd_temperature"]
        rank_s = torch.sigmoid((zf.double()[:, None]-student_grid[None, :])/tau).sum(1)
        rank_t = torch.sigmoid((t[:, None]-row_grid)/tau).sum(1)
        squared = (rank_s-rank_t).square()
        soft = beta * (squared[y["fit"] == 0].mean() + squared[y["fit"] == 1].mean())
    else:
        target = y["fit"].double() if condition == "risk_matched" else torch.sigmoid(teacher_margin/temp)
        soft = beta * temp**2 * F.binary_cross_entropy_with_logits((zf.double()-threshold)/temp, target)
    loss = hard + soft
    gradients = torch.autograd.grad(loss, tuple(net.parameters()))
    return float(soft.detach()), gradients


def test_zero_soft_weight_all_nonrisk_students_save_exactly_the_same_state_as_hard(synthetic_case, tmp_path):
    _, _, _, config, exports = synthetic_case
    config["optimization"]["kd_weight"] = 0.
    conditions = ["hard", "boundary_kd", "pct", "dual_source_fd", "ekd"]
    records = {condition: _run(synthetic_case, tmp_path, condition) for condition in conditions}
    for condition in conditions:
        assert records[condition]["common_initial_state_sha256"] == records["hard"]["common_initial_state_sha256"]
        for key, value in exports["hard"].items():
            torch.testing.assert_close(exports[condition][key], value, rtol=0, atol=0)
        history = json.loads((tmp_path / condition / "history.json").read_text(encoding="utf-8"))
        assert all(row["soft"] == 0. for row in history)


@pytest.mark.parametrize("condition", ["boundary_kd", "risk_kd", "risk_matched", "normal_hinge_kd"])
def test_actual_runner_applies_nonunit_beta_to_boundary_soft_gradient_once(
        synthetic_case, tmp_path, monkeypatch, condition):
    config = synthetic_case[3]
    config["optimization"].update(macro_steps=1, chunk_size=2, kd_weight=.2)
    expected_soft, expected_gradients = _full_population_oracle(synthetic_case, condition)
    captured = _capture_step(monkeypatch)
    _run(synthetic_case, tmp_path, condition)
    assert len(captured) == 1
    for actual, expected in zip(captured[0], expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    history = json.loads((tmp_path / condition / "history.json").read_text(encoding="utf-8"))
    assert history[0]["soft"] == pytest.approx(expected_soft, rel=2e-7, abs=2e-8)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 32])
def test_actual_runner_ekd_fold_chunks_recover_global_class_means(
        synthetic_case, tmp_path, monkeypatch, chunk_size):
    config = synthetic_case[3]
    config["optimization"].update(macro_steps=1, chunk_size=chunk_size)
    expected_soft, expected_gradients = _full_population_oracle(synthetic_case, "ekd")
    captured = _capture_step(monkeypatch)
    _run(synthetic_case, tmp_path, "ekd")
    assert len(captured) == 1
    for actual, expected in zip(captured[0], expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    history = json.loads((tmp_path / "ekd" / "history.json").read_text(encoding="utf-8"))
    assert history[0]["soft"] == pytest.approx(expected_soft, rel=2e-7, abs=2e-8)
    # The fixture passes numpy teacher grids, as _teacher_targets actually does;
    # exercising the real runner also catches a missing .tolist() API conversion.
