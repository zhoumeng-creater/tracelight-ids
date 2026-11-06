"""Synthetic bound H/K execution; no archived model inference or training."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import ids.unsw_diagnostic_experiment as diagnostic
import ids.unsw_kd_minibatch_experiment as impl
from ids.common import canonical_hash, read_json, sha256_file, write_json
from ids.unsw_kd import array_sha256
from test_unsw_diagnostic_experiment import bound as diagnostic_bound  # noqa: F401


@pytest.fixture
def mini_bound(request, tmp_path, monkeypatch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(314)
        screen, previous, data, binding = request.getfixturevalue("diagnostic_bound")
    # The original synthetic fixture has 24 F, 16 R and 256 C rows. Bind old
    # cached scores with exactly five additional K detections; no model query.
    for arm in ("H", "K"):
        cached = {"reference": np.zeros(16), "constraint": np.zeros(256)}
        if arm == "K":
            cached["constraint"][np.arange(1, 10, 2)] = 1.
        path = previous / f"arms/{arm}/scores.npz"
        np.savez(path, **cached)
        binding["files"].append({"root_alias": "round10", "path": path.relative_to(previous).as_posix(),
                                 "sha256": sha256_file(path), "bytes": path.stat().st_size})
    write_json(diagnostic.BINDING, binding)
    protocol = read_json(impl.PROTOCOL)
    original = read_json(previous / "protocol.json")
    protocol["protocol_id"] = "synthetic-mini-integration"
    protocol["student"].update(original["student"])
    protocol["data"]["preprocessing_semantic_sha256"] = original["data"]["preprocessing_semantic_sha256"]
    protocol["data"]["roles"] = {
        role: {"rows": n, "normal": n//2, "attack": n//2}
        for role, n in (("fit", 24), ("reference", 16), ("constraint", 256))}
    protocol["training"].update(epochs=2, endpoint_epoch=2, batches_per_epoch=1, last_batch_rows=24)
    specification = protocol["permutations"]
    specification["shape"] = [2, 24]
    specification["numpy_version"] = np.__version__
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(specification["seed_sequence"])))
    orders = np.stack([rng.permutation(24) for _ in range(2)]).astype("<i8")
    specification["sha256"] = array_sha256(orders, "<i8")
    specification["epoch_sha256"] = [array_sha256(row, "<i8") for row in orders]
    protocol["budget"].update(gradient_forward_rows=96, forward_calls=4, backward_calls=4,
                              optimizer_updates=4, endpoint_R_C_scoring_rows=544,
                              endpoint_R_C_scoring_calls=4)
    qualification = {"independent_validity_passed": True, "checks": {"synthetic_integrity": True},
                     "source": {"commit": "a"*40}}
    qualification_path = tmp_path / "qualification.json"
    write_json(qualification_path, qualification)
    protocol["qualification"].update(path=str(qualification_path), semantic_sha256=canonical_hash(qualification),
                                     diagnostic_commit=qualification["source"]["commit"])
    protocol_path = tmp_path / "mini-protocol.json"
    write_json(protocol_path, protocol)
    monkeypatch.setattr(impl, "PROTOCOL", protocol_path)
    monkeypatch.setattr(impl, "PROTOCOL_SEMANTIC_SHA256", canonical_hash(protocol))
    return screen, previous, data, protocol


def test_bound_preflight_has_no_fit_inference_or_reserved_reads(mini_bound, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Preflight must not call a network, teacher, or optimizer")

    monkeypatch.setattr(torch.nn.Sequential, "forward", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    monkeypatch.setattr(impl, "fit_minibatch_arm", forbidden)
    receipt = impl.preflight_minibatch(*mini_bound[:3])
    assert receipt["status"] == "minibatch_preflight_complete"
    assert receipt["historical_full_C_net_TP"] == 5
    assert receipt["new_inference_rows"] == receipt["optimizer_updates"] == receipt["student_fits"] == 0
    assert receipt["teacher_fits"] == 0 and receipt["reserved_partitions_read"] == []
    assert receipt["permutations_sha256"] == mini_bound[3]["permutations"]["sha256"]
    assert len(receipt["epoch_permutation_sha256"]) == 2


def test_real_synthetic_training_and_endpoint_scoring_use_exact_budget_and_keep_negative_result(
        mini_bound, tmp_path, monkeypatch):
    context = impl._context(*mini_bound[:3])
    forwards, updates, fit_calls, scoring_calls = [], [], [], []
    sequential_forward = torch.nn.Sequential.forward
    adam_step = torch.optim.AdamW.step
    fitting = impl.fit_minibatch_arm
    scoring = impl._scores

    def counted_forward(net, batch):
        forwards.append((torch.is_grad_enabled(), len(batch)))
        return sequential_forward(net, batch)

    def counted_step(optimizer, *args, **kwargs):
        updates.append(1)
        return adam_step(optimizer, *args, **kwargs)

    def counted_fit(anchor, x, y, target, permutations, **kwargs):
        fit_calls.append(kwargs["arm"])
        assert permutations is context["permutations"] and not permutations.flags.writeable
        assert len(permutations) == 2 and len(x) == 24
        return fitting(anchor, x, y, target, permutations, **kwargs)

    def counted_scores(bundle, frames, chunk):
        scoring_calls.append((bundle.best_epoch, tuple(frames)))
        assert bundle.best_epoch == 2 and tuple(frames) == ("reference", "constraint")
        return scoring(bundle, frames, chunk)

    monkeypatch.setattr(torch.nn.Sequential, "forward", counted_forward)
    monkeypatch.setattr(torch.optim.AdamW, "step", counted_step)
    monkeypatch.setattr(impl, "fit_minibatch_arm", counted_fit)
    monkeypatch.setattr(impl, "_scores", counted_scores)
    output = tmp_path / "execution"
    result = impl._execute(context, output, device="cpu", source={"files": {}})
    assert fit_calls == ["H", "K"] and len(updates) == 4
    assert scoring_calls == [(2, ("reference", "constraint"))]*2
    assert forwards == [(True, 24), (True, 24), (False, 16), (False, 256)]*2
    assert result["budget"] == mini_bound[3]["budget"]
    assert result["validity_passed"] and all(result["checks"].values())
    assert result["status"] == "unsw_kd_minibatch_complete"
    assert result["quality_acceptance_passed"] is False and result["acceptance_passed"] is False
    assert result["evaluation"]["interaction"]["historical_full_net_TP"] == 5
    assert result["automatic_expansion"] is False and result["reserved_partitions_read"] == []
    assert result["independent_confirmation"] is False
    assert not (output / "FAILURE.json").exists()
    assert (output / "REPORT.md").is_file()
    done = read_json(output / impl.COMPLETE)
    assert done["validity_passed"] and done["acceptance_passed"] is False
    for name, digest in done["files"].items():
        assert sha256_file(output / name) == digest
    for arm in ("H", "K"):
        folder = output / "arms" / arm
        assert (folder / "model.joblib").is_file() and (folder / "scores.npz").is_file()
        history = read_json(folder / "history.json")
        streamed = [json.loads(row) for row in (folder / "history.jsonl").read_text().splitlines()]
        assert history == streamed and len(history) == 2
        assert read_json(folder / "run.json")["endpoint_epoch"] == 2


def test_second_arm_failure_keeps_first_arm_and_partial_epoch_log_without_retry(mini_bound, tmp_path, monkeypatch):
    context = impl._context(*mini_bound[:3])
    fit = impl.fit_minibatch_arm
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(kwargs["arm"])
        if kwargs["arm"] == "K":
            kwargs["on_epoch"]({"epoch": 1, "cumulative_optimizer_updates": 1, "partial": True})
            raise RuntimeError("synthetic second-arm failure")
        return fit(*args, **kwargs)

    monkeypatch.setattr(impl, "fit_minibatch_arm", fail_second)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="second-arm failure"):
        impl._execute(context, output, device="cpu", source={"files": {}})
    assert calls == ["H", "K"]
    assert (output / "arms/H/model.joblib").is_file()
    assert (output / "arms/H/scores.npz").is_file()
    assert json.loads((output / "arms/K/history.jsonl").read_text())["partial"]
    assert read_json(output / "FAILURE.json")["status"] == "technical_failure"
    assert not (output / impl.COMPLETE).exists() and not (output / "summary.json").exists()


def test_existing_output_rejected_before_context_or_cuda(mini_bound, tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "retain").write_bytes(b"old scientific evidence")
    monkeypatch.setattr(impl, "_context", lambda *a: pytest.fail("loaded inputs"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA checked"))
    with pytest.raises(FileExistsError, match="no overwrite"):
        impl.run_minibatch(*mini_bound[:3], output)
    assert (output / "retain").read_bytes() == b"old scientific evidence"


@pytest.mark.parametrize("device", ["cpu", "cuda:1", "mps"])
def test_public_run_rejects_cpu_and_nonzero_gpu_before_loading(mini_bound, tmp_path, monkeypatch, device):
    monkeypatch.setattr(impl, "_context", lambda *a: pytest.fail("loaded inputs"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA should not be checked"))
    output = tmp_path / "no-run"
    with pytest.raises(ValueError, match="RTX4090D"):
        impl.run_minibatch(*mini_bound[:3], output, device=device)
    assert not output.exists()


@pytest.mark.parametrize("change", ["protocol", "permutation", "epoch_hash", "numpy", "qualification",
                                     "qualification_failed", "qualification_commit"])
def test_configuration_or_qualification_changes_fail_before_input_load_or_fit(
        mini_bound, monkeypatch, change):
    protocol = read_json(impl.PROTOCOL)
    if change == "protocol":
        protocol["training"]["epochs"] += 1
    elif change == "permutation":
        protocol["permutations"]["sha256"] = "0"*64
    elif change == "epoch_hash":
        protocol["permutations"]["epoch_sha256"][0] = "0"*64
    elif change == "numpy":
        protocol["permutations"]["numpy_version"] = "invalid-version"
    else:
        qualification = read_json(protocol["qualification"]["path"])
        if change == "qualification":
            qualification["unexpected_change"] = True
        elif change == "qualification_failed":
            qualification["checks"]["synthetic_integrity"] = False
            protocol["qualification"]["semantic_sha256"] = canonical_hash(qualification)
        else:
            qualification["source"]["commit"] = "b"*40
            protocol["qualification"]["semantic_sha256"] = canonical_hash(qualification)
        write_json(protocol["qualification"]["path"], qualification)
    write_json(impl.PROTOCOL, protocol)
    # Exercise downstream hash/qualification checks independently of the outer
    # protocol digest. No production path supports this test monkeypatch.
    if change != "protocol":
        monkeypatch.setattr(impl, "PROTOCOL_SEMANTIC_SHA256", canonical_hash(protocol))
    monkeypatch.setattr(impl, "_bound_inputs", lambda *a: pytest.fail("loaded bound inputs"))
    monkeypatch.setattr(impl, "fit_minibatch_arm", lambda *a, **k: pytest.fail("started training"))
    with pytest.raises(ValueError, match="protocol|permutation|numpy|qualification"):
        impl.preflight_minibatch(*mini_bound[:3])


@pytest.mark.parametrize("change", ["model", "scores", "target", "train"])
def test_changed_frozen_files_fail_before_any_new_forward_or_fit(mini_bound, monkeypatch, change):
    screen, previous, data, _ = mini_bound
    path = {"model": screen / "warmups/all-small-s17/model.joblib",
            "scores": previous / "arms/K/scores.npz", "target": previous / "targets.npz",
            "train": data / "train.parquet"}[change]
    with path.open("ab") as stream:
        stream.write(b"changed")
    monkeypatch.setattr(torch.nn.Sequential, "forward", lambda *a: pytest.fail("new forward"))
    monkeypatch.setattr(impl, "fit_minibatch_arm", lambda *a, **k: pytest.fail("started training"))
    with pytest.raises(ValueError):
        impl.preflight_minibatch(*mini_bound[:3])


def test_historical_difference_is_recounted_and_must_match_frozen_five(mini_bound):
    context = impl._context(*mini_bound[:3])
    path = mini_bound[1] / "arms/K/scores.npz"
    with np.load(path, allow_pickle=False) as existing:
        arrays = {name: existing[name].copy() for name in existing.files}
    arrays["constraint"][1] = 0
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="frozen \\+5"):
        impl._historical(context)


def test_mutation_during_training_prevents_completion_and_retains_models(mini_bound, tmp_path, monkeypatch):
    context = impl._context(*mini_bound[:3])
    fit = impl.fit_minibatch_arm

    def mutate_after_fit(*args, **kwargs):
        result = fit(*args, **kwargs)
        if kwargs["arm"] == "K":
            with impl.PROTOCOL.open("a", encoding="utf-8") as stream:
                stream.write(" ")
        return result

    monkeypatch.setattr(impl, "fit_minibatch_arm", mutate_after_fit)
    output = tmp_path / "changed-mid-run"
    with pytest.raises(ValueError, match="changed during execution"):
        impl._execute(context, output, device="cpu", source={"files": {}})
    assert (output / "arms/H/model.joblib").is_file()
    assert (output / "arms/K/model.joblib").is_file()
    assert (output / "FAILURE.json").is_file() and not (output / impl.COMPLETE).exists()
