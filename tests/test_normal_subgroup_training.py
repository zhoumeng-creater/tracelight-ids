"""Fresh synthetic official CPU/CUDA models; production budgets are never reduced."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ids.model_training_redesign import training as original
from ids.model_training_redesign.preprocessing import CONTINUOUS, Preprocessor
from ids.normal_subgroup_training import groups, preflight, runner, selection, training
from ids.normal_subgroup_training.runner import check_pairing


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class FallbackSelector(selection.Selector):
    """Force fallback only to exercise delivery; real metric/guard computation still runs."""

    def update(self, metrics, step):
        decision = super().update(metrics, step)
        self.best_step, self.best_H = 0, self.baseline["tail"]["H"]
        decision.update(improved=False, selected_step=0, selected_H=self.best_H, candidate_improved=False)
        return decision


class FirstCheckpointSelector(selection.Selector):
    """Inject a selection decision to verify actual selected-weight serialization."""

    def update(self, metrics, step):
        decision = super().update(metrics, step)
        if self.events == 1:
            self.first_H = metrics["tail"]["H"]
        self.best_step, self.best_H = selection.VALIDATION_STEPS[0], self.first_H
        decision.update(improved=self.events == 1, selected_step=self.best_step,
                        selected_H=self.best_H, candidate_improved=True)
        return decision


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    pytest.importorskip("tabm")
    pytest.importorskip("rtdl_num_embeddings")
    from ids.recent_baselines import _parameters, make_tabm_network

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(groups, "MIN_CHILD_ROWS", 2)
    monkeypatch.setattr(groups, "MAX_DEPTH", 1)
    monkeypatch.setattr(selection, "VALIDATION_STEPS", (1, 2))
    monkeypatch.setattr(runner, "VALIDATION_STEPS", (1, 2))
    monkeypatch.setattr(training, "STEPS", (1, 2))
    # The public production contract is separately tested. This test-only patch
    # reaches the real core with a freshly created small artifact and tiny budget.
    monkeypatch.setattr(training, "_validate_production", lambda *a, **kw: None)
    torch.manual_seed(17)
    rng = np.random.default_rng(17)
    rows = 16
    raw = rng.uniform(.1, 10., (2 * rows, 5))
    frames = {}
    for role, interval in (("F", 30), ("V", 300)):
        frame = pd.DataFrame(raw.copy(), columns=CONTINUOUS)
        frame["profile"] = ["A"] * rows + ["B"] * rows
        frame["source_id"] = [f"TQH-A_i{interval}_j0"] * rows + [f"TQH-B_i{interval}_j0"] * rows
        frame["row_id"] = list(range(rows)) * 2
        frame["ip_protocol"] = [6] * rows + [17] * rows
        frame["label"] = np.tile([0, 1], rows)
        frames[role] = frame
    repo = Path(__file__).resolve().parents[1]
    r30_design = read(repo / "configs/model-training-redesign-20261117/design.json")
    processor = Preprocessor.fit(frames["F"], arm="protocol_blind", design=r30_design)
    parameters = _parameters({"k": 32, "n_blocks": 1, "d_block": 8, "dropout": .1,
                              "embedding_dim": 4, "n_bins": 3, "batch_size": 256})
    bins = [[-2., 0., 2.]] * 5
    network = make_tabm_network(5, processor.cat_cardinalities, parameters, numeric_bins=bins)
    predictor = original.Predictor(network.eval(), processor, torch.device("cpu"), 256)
    scores = predictor.score(frames["F"])
    assert len(np.unique(scores)) == 2 * rows
    threshold = np.median(scores)
    for frame in frames.values():
        frame["label"] = (scores > threshold).astype(int)
    assert all(set(part.label) == {0, 1} for _, part in frames["F"].groupby("profile"))

    archive = tmp_path / "archive"
    archive.mkdir()
    models = {}
    for seed in (17, 29, 43):
        artifact = archive / f"original-{seed}.pt"
        state = {name: value.detach().clone() for name, value in network.state_dict().items()}
        torch.save({
            "format": original.FORMAT, "arm": "protocol_blind", "seed": seed,
            "final_refits": 0, "selected_step": 123, "score_batch_size": 256,
            "preprocessing": processor.to_dict(), "parameters": parameters,
            "numeric_bins": bins, "constant_bin_columns": [],
            "state_dict": state, "state_sha256": original._state_sha256(state),
            "preprocessing_sha256": original._hash_json(processor.base.to_dict()),
            "numeric_bins_sha256": original._hash_json(bins),
        }, artifact)
        models[f"protocol_blind_s{seed}"] = {"path": artifact.name}
    monkeypatch.setattr(preflight, "load_bound_inputs", lambda *a, **kw: (frames, {"models": models}, {}))
    monkeypatch.setattr(preflight, "_runtime", lambda: {"kind": "fresh_synthetic_CPU"})
    frozen = tmp_path / "preflight"
    result = preflight.run(archive, frozen)
    assert result["status"] == "zero_update_preflight_complete"

    design = read(training.DEFAULT_DESIGN)
    design["model"].update(k=32, n_blocks=1, d_block=8, dropout=.1)
    design["training"]["sampler"]["draws_per_group"] = 2
    yield {"frames": frames, "preflight": frozen, "artifact": archive / "original-17.pt",
           "root": tmp_path, "design": design}
    torch.set_num_threads(previous_threads)


def fit(fixture, arm="transport_target", suffix=None, *, device="cpu"):
    return training.fit_trajectory(
        fixture["frames"], fixture["preflight"], fixture["artifact"],
        fixture["root"] / (suffix or arm), arm=arm, seed=17,
        device=device, threads=1, design=fixture["design"])


def test_actual_pair_completes_budget_rng_pairing_and_original_fallback(synthetic_run, monkeypatch):
    monkeypatch.setattr(training, "Selector", FallbackSelector)
    before_rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("Training must not refit preprocessing or bins")

    monkeypatch.setattr(Preprocessor, "fit", forbidden)
    from ids.recent_baselines import _numeric_bins
    monkeypatch.setattr("ids.recent_baselines._numeric_bins", forbidden)
    assert callable(_numeric_bins)
    results = [fit(synthetic_run, arm) for arm in training.ARMS]
    assert torch.equal(torch.get_rng_state(), before_rng)
    pairs = []
    for result in results:
        assert result["status"] == "complete", result
        assert result["optimizer_steps"] == 2 and result["completed_validation_events"] == 2
        assert result["fit_calls"] == 1 and result["final_refits"] == 0
        assert result["selected_step"] == 0 and result["fallback"] and not result["candidate_improved"]
        assert Path(result["artifact_path"]).read_bytes() == synthetic_run["artifact"].read_bytes()
        assert result["selected_artifact_sha256"] == result["original_artifact_sha256"]
        assert result["model_artifact_bytes"] == synthetic_run["artifact"].stat().st_size
        assert result["selection_device"] == "cpu"
        history = read(result["history_path"])
        assert [r["step"] for r in history] == [1, 2]
        assert all(r["gradient"]["window_steps"] == 1 for r in history)
        pair = read(result["pairing_path"])
        assert pair["preprocessing_sha256"] == result["selected_preprocessing_sha256"]
        assert pair["numeric_bins_sha256"] == result["selected_numeric_bins_sha256"]
        assert set(pair["draw_chain_by_validation_step"]) == {"1", "2"}
        assert all(pair["selection_rng_unchanged_by_step"].values())
        assert pair["optimizer_initial_state_entries"] == 0
        pairs.append(pair)
        original_model = training.load_predictor(result["artifact_path"], threads=1, device="cpu")
        with np.load(result["selected_scores_path"]) as values:
            for role in ("F", "V"):
                np.testing.assert_array_equal(values[role], original_model.score(synthetic_run["frames"][role]))
        baseline = read(result["baseline_metrics_path"])
        assert baseline == read(result["selected_metrics_path"])
        assert {"numerator", "denominator", "value"} == set(baseline["tail"]["H"])
    checked = check_pairing(*pairs)
    assert checked["passed"]
    assert checked["common_validation_steps"] == [1, 2]
    assert pairs[0]["normal_target_sha256"] != pairs[1]["normal_target_sha256"]
    with pytest.raises(ValueError, match="incomplete fits are not retried"):
        fit(synthetic_run)


def test_selected_checkpoint_is_original_format_and_cpu_scores_cold_replay(synthetic_run, monkeypatch):
    monkeypatch.setattr(training, "Selector", FirstCheckpointSelector)
    result = fit(synthetic_run)
    assert result["status"] == "complete", result
    assert result["selected_step"] == 1 and result["optimizer_steps"] == 2
    assert not result["fallback"] and result["candidate_improved"]
    assert result["selected_artifact_sha256"] != result["original_artifact_sha256"]
    state = torch.load(result["artifact_path"], map_location="cpu", weights_only=True)
    assert state["arm"] == "protocol_blind" and state["target_arm"] == "transport_target"
    assert state["selected_step"] == state["rng_step"] == 1
    original_state = torch.load(synthetic_run["artifact"], map_location="cpu", weights_only=True)
    for key in ("preprocessing", "numeric_bins", "parameters", "preprocessing_sha256", "numeric_bins_sha256"):
        assert state[key] == original_state[key]
    model = training.load_predictor(result["artifact_path"], threads=1, device="cpu")
    with np.load(result["selected_scores_path"]) as scores:
        for role in ("F", "V"):
            np.testing.assert_array_equal(model.score(synthetic_run["frames"][role]), scores[role])
    pair = read(result["pairing_path"])
    assert original._state_sha256(state["dropout_rng_state"]) == pair["dropout_rng_by_validation_step"]["1"]
    assert state["sampling_state"]["audit"]["draw_sha256"] == pair["draw_chain_by_validation_step"]["1"]
    assert read(result["score_identities_path"]) == {
        r: f[["source_id", "row_id"]].values.tolist() for r, f in synthetic_run["frames"].items()}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; CPU coverage is separate")
def test_fresh_tiny_cuda_pair_keeps_training_and_moments_on_gpu_cpu_selection_is_rng_neutral(
        synthetic_run, monkeypatch):
    monkeypatch.setattr(training, "Selector", FallbackSelector)
    actual_step, actual_score = training.training_step, training._cpu_scores
    device_checks, selection_checks = [], []

    def checked_step(network, optimizer, numeric, categorical, y, targets):
        assert all(p.device.type == "cuda" for p in network.parameters())
        assert all(value.device.type == "cuda" for value in (numeric, categorical, y))
        result = actual_step(network, optimizer, numeric, categorical, y, targets)
        # AdamW's scalar step counters are intentionally CPU tensors when
        # capturable=False; the parameter-shaped moments must remain on CUDA.
        assert optimizer.state
        assert all(state["exp_avg"].device.type == state["exp_avg_sq"].device.type == "cuda"
                   for state in optimizer.state.values())
        device_checks.append(True)
        return result

    def checked_score(predictor, network, arrays, actual):
        assert actual.type == "cuda"
        assert all(p.device.type == "cuda" for p in network.parameters())
        assert all(p.device.type == "cpu" for p in predictor.network.parameters())
        before = original._state_sha256(original._rng_state(actual))
        result = actual_score(predictor, network, arrays, actual)
        assert before == original._state_sha256(original._rng_state(actual))
        assert all(p.device.type == "cuda" for p in network.parameters())
        selection_checks.append(True)
        return result

    monkeypatch.setattr(training, "training_step", checked_step)
    monkeypatch.setattr(training, "_cpu_scores", checked_score)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(0).clone()
    results = [fit(synthetic_run, arm, suffix="cuda_" + arm, device="cuda:0") for arm in training.ARMS]
    assert all(result["status"] == "complete" for result in results), results
    assert len(device_checks) == 4 and len(selection_checks) == 6  # Two step0 + four events.
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(0), cuda_rng)
    pairs = [read(result["pairing_path"]) for result in results]
    assert check_pairing(*pairs)["common_validation_steps"] == [1, 2]
    assert all(result["training_device"] == "cuda:0" and result["selection_device"] == "cpu"
               for result in results)
    assert all(result["optimizer_steps"] == 2 and result["final_refits"] == 0 for result in results)
    assert all(Path(result["artifact_path"]).read_bytes() == synthetic_run["artifact"].read_bytes()
               for result in results)


def test_preflight_payload_tampering_fails_before_original_model_load(synthetic_run, monkeypatch):
    target = synthetic_run["preflight"] / "protocol_blind_s17/target_metadata.json"
    target.write_text(target.read_text() + " ", encoding="utf-8")
    monkeypatch.setattr(original, "load_predictor", lambda *a, **kw: pytest.fail("must not load"))
    result = fit(synthetic_run)
    assert result["status"] == "failed" and "payload binding changed" in result["error"]
    assert result["fit_calls"] == result["optimizer_steps"] == 0
    assert read(synthetic_run["root"] / "transport_target/failure.json")["traceback"]
    with pytest.raises(ValueError, match="must be empty"):
        fit(synthetic_run)


def test_identity_mismatch_rejected_before_model_load(synthetic_run, monkeypatch):
    synthetic_run["frames"]["F"].loc[0, "row_id"] = 1000
    monkeypatch.setattr(original, "load_predictor", lambda *a, **kw: pytest.fail("must not load"))
    result = fit(synthetic_run)
    assert result["status"] == "failed" and "physical identity order changed" in result["error"]
    assert result["optimizer_steps"] == 0


def test_step0_mismatch_rejected_without_fit(synthetic_run, monkeypatch):
    actual = training._cpu_scores

    def changed(*args, **kwargs):
        value = actual(*args, **kwargs)
        value["F"][0] = np.nextafter(value["F"][0], np.inf)
        return value

    monkeypatch.setattr(training, "_cpu_scores", changed)
    result = fit(synthetic_run)
    assert result["status"] == "failed" and "Step0 CPU outputs differ" in result["error"]
    assert result["fit_calls"] == result["optimizer_steps"] == 0


def test_partial_fit_is_archived_without_retry_and_records_real_update_count(synthetic_run, monkeypatch):
    actual = training.training_step
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("fabricated second-step failure")
        return actual(*args, **kwargs)

    monkeypatch.setattr(training, "training_step", fail_second)
    result = fit(synthetic_run)
    assert result["status"] == "failed" and result["optimizer_steps"] == 1
    assert result["attempted_update"] == 2 and result["retry_allowed"] is False
    assert [r["step"] for r in read(synthetic_run["root"] / "transport_target/history.json")] == [1]
    assert not (synthetic_run["root"] / "transport_target/model.pt").exists()
    with pytest.raises(ValueError, match="must be empty"):
        fit(synthetic_run)


def test_production_entry_rejects_tiny_budget_before_inputs_or_weights(tmp_path):
    design = read(training.DEFAULT_DESIGN)
    design["training"]["optimizer_steps_per_trajectory"] = 2
    with pytest.raises(ValueError, match="frozen v2 contract"):
        training.fit_trajectory({"F": None, "V": None}, tmp_path / "absent", tmp_path / "absent.pt",
                                tmp_path / "output", arm="transport_target", seed=17,
                                device="cpu", design=design)
    assert not (tmp_path / "output").exists()


def test_production_entry_rejects_C_E_and_nonregistered_seed(tmp_path):
    for frames, seed, message in (({"F": None, "V": None, "C": None}, 17, "exactly F and V"),
                                  ({"F": None, "V": None}, 18, "registered target arm/seed")):
        with pytest.raises(ValueError, match=message):
            training.fit_trajectory(frames, tmp_path / "absent", tmp_path / "absent.pt",
                                    tmp_path / "output", arm="transport_target", seed=seed, device="cpu")
