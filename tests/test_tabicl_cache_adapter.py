"""Actual pinned architecture, tiny random weights and synthetic tensors only."""

import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from ids.context_cache.cache import WeightedKVEntry  # noqa: E402
from ids.context_cache.tabicl_adapter import CompactPredictor, FullCacheAdapter  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def official_source_and_threads():
    if importlib.util.find_spec("tabicl") is None:
        pytest.skip("put the verified pinned TabICL source on PYTHONPATH; no auto-download")
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def fixture_model(norm_first=True):
    from tabicl._model.tabicl import TabICL
    config = dict(max_classes=10, embed_dim=16, col_num_blocks=1, col_nhead=2,
                  col_num_inds=4, row_num_blocks=1, row_nhead=2, row_num_cls=2,
                  icl_num_blocks=2, icl_nhead=4, ff_factor=2, dropout=0.0,
                  norm_first=norm_first, zero_init=False)
    torch.manual_seed(812)
    model = TabICL(**config).eval()
    # Nonlinear query modulation must be active, otherwise scale-order bugs hide.
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "query_mlp"):
                module.query_mlp[-1].weight.normal_(0, 0.08)
                module.query_mlp[-1].bias.fill_(0.03)
    generator = torch.Generator().manual_seed(716)
    support = torch.randn(2, 12, 3, generator=generator)
    labels = torch.tensor([[0, 1] * 6, [1, 0] * 6])
    queries = torch.randn(2, 5, 3, generator=generator)
    return model, config, support, labels, queries


@pytest.mark.parametrize("norm_first", [True, False])
def test_full_bridge_matches_official_and_cache_has_an_effect(norm_first):
    model, config, support, labels, queries = fixture_model(norm_first)
    adapter = FullCacheAdapter(model, model_config=config).build(support, labels)
    expected = adapter.predict_logits(queries)
    predictor = adapter.as_predictor()
    torch.testing.assert_close(predictor.predict_logits(queries), expected, rtol=3e-5, atol=3e-6)
    assert expected.shape == (2, 5, 2)  # max_classes=10 does not change binary mapping.
    assert model._cache is None
    for i, phi in adapter.support_queries.items():
        assert phi.shape == adapter.entries[i].key.shape
        assert phi.shape[2] == 12
    # Detect vacuous agreement from the official zero-output initialization.
    changed = {i: WeightedKVEntry(e.key, e.value * 0, e.log_mass, e.source_count)
               for i, e in predictor.entries.items()}
    rows = predictor.encode_queries(queries)
    with torch.no_grad():
        altered = predictor.stage3_logits(rows, entries=changed)
    assert (altered - expected).abs().max().item() > 1e-4


def test_compilation_keeps_original_length_and_cold_artifact(tmp_path):
    model, config, support, labels, queries = fixture_model()
    full = FullCacheAdapter(model, model_config=config, provenance={"data_kind": "synthetic"}).build(support, labels)
    compact, report = full.compile(slots=3, query_chunk_size=5)
    assert report["compiled"]["icl_key_value_tensor_bytes"] * 4 == report["full"]["icl_key_value_tensor_bytes"]
    called_lengths, hooks = [], []
    for block in model.icl_predictor.tf_icl.blocks:
        hooks.append(block.attn.ssmax_layer.register_forward_pre_hook(
            lambda module, args: called_lengths.append(args[1])
        ))
    try:
        expected = compact.predict_logits(queries)
    finally:
        for hook in hooks:
            hook.remove()
    assert called_lengths == [12, 12]
    assert all(e.num_slots == 3 and e.source_count == 12 for e in compact.entries.values())

    artifact = compact.save(tmp_path / "compact.pt")
    payload = torch.load(artifact, weights_only=True)
    assert set(payload) == {"format", "model_config", "model_state", "column_cache", "icl_cache",
                            "train_shape", "num_classes", "dtype", "cache_kind", "input_contract",
                            "upstream", "provenance"}
    assert all(v["key"].shape[-2] == 3 for v in payload["icl_cache"].values())
    loaded = CompactPredictor.load(artifact)
    torch.testing.assert_close(loaded.predict_logits(queries), expected, rtol=0, atol=0)
    assert loaded.model._cache is None

    # A fresh interpreter gets only this artifact and numeric queries, no full
    # adapter, training data, support phi, or original checkpoint file.
    query_path, output_path = tmp_path / "queries.pt", tmp_path / "predictions.pt"
    torch.save(queries, query_path)
    script = (
        "import sys,torch; from ids.context_cache.tabicl_adapter import CompactPredictor; "
        "torch.set_num_threads(1); p=CompactPredictor.load(sys.argv[1]); "
        "x=torch.load(sys.argv[2],weights_only=True); "
        "torch.save(p.predict_logits(x),sys.argv[3]); assert p.model._cache is None"
    )
    env = dict(os.environ)
    repo = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join([str(repo / "src"), *sys.path])
    result = subprocess.run([sys.executable, "-c", script, str(artifact), str(query_path), str(output_path)],
                            capture_output=True, text=True, env=env, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    torch.testing.assert_close(torch.load(output_path, weights_only=True), expected, rtol=0, atol=0)


def test_full_export_slice_and_column_storage_are_independent(tmp_path):
    model, config, support, labels, queries = fixture_model()
    adapter = FullCacheAdapter(model, model_config=config).build(support, labels)
    full = adapter.as_predictor()
    sliced = full.slice_batch(1, 2)
    expected = adapter.predict_logits(queries)[1:2]
    torch.testing.assert_close(sliced.predict_logits(queries[1:2]), expected, rtol=3e-5, atol=3e-6)
    path = sliced.save(tmp_path / "full-slice.pt")
    payload = torch.load(path, weights_only=True)
    for group in (payload["column_cache"], payload["icl_cache"]):
        for entry in group.values():
            for tensor in entry.values():
                if isinstance(tensor, torch.Tensor):
                    assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    reloaded = CompactPredictor.load(path).to("cpu")
    torch.testing.assert_close(reloaded.predict_logits(queries[1:2]), expected, rtol=3e-5, atol=3e-6)


def test_independent_cache_leaf_gradient_agrees_with_difference():
    model, config, support, labels, queries = fixture_model()
    full = FullCacheAdapter(model, model_config=config).build(support, labels)
    predictor = full.as_predictor()
    rows = predictor.encode_queries(queries)
    entry = predictor.entries[0]
    variable = entry.value.detach().clone().requires_grad_(True)
    entries = dict(predictor.entries)
    entries[0] = WeightedKVEntry(entry.key, variable, entry.log_mass, entry.source_count)
    output = predictor.stage3_logits(rows, entries=entries)
    score = output[0, 0, 1] - output[0, 0, 0]
    gradient = torch.autograd.grad(score, variable)[0]
    flat = gradient.abs().argmax()
    direction = torch.zeros_like(variable).flatten()
    direction[flat] = 1
    direction = direction.reshape_as(variable)
    epsilon = 1e-2
    scores = []
    with torch.no_grad():
        for sign in (-1, 1):
            entries[0] = WeightedKVEntry(entry.key, variable.detach() + sign * epsilon * direction,
                                         entry.log_mass, entry.source_count)
            output = predictor.stage3_logits(rows, entries=entries)
            scores.append(output[0, 0, 1] - output[0, 0, 0])
    finite = (scores[1] - scores[0]) / (2 * epsilon)
    torch.testing.assert_close(finite, gradient.flatten()[flat], rtol=0.02, atol=2e-5)
    assert gradient.abs().max() > 1e-5
    assert all(not p.requires_grad for p in model.parameters())


def test_local_checkpoint_hash_and_label_contract(tmp_path):
    model, config, support, labels, queries = fixture_model()
    path = tmp_path / "synthetic-not-pretrained.ckpt"
    torch.save({"config": config, "state_dict": model.state_dict()}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    adapter = FullCacheAdapter.from_checkpoint(path, expected_sha256=digest).build(support, labels)
    assert adapter.predict_logits(queries).shape[-1] == 2
    assert adapter.provenance["checkpoint_sha256"] == digest
    with pytest.raises(ValueError, match="SHA-256"):
        FullCacheAdapter.from_checkpoint(path, expected_sha256="0" * 64)
    with pytest.raises(FileNotFoundError, match="local checkpoint"):
        FullCacheAdapter.from_checkpoint(tmp_path / "absent.ckpt")
    with pytest.raises(ValueError, match="both prepared labels"):
        FullCacheAdapter(model, model_config=config).build(support, labels + 1)
    with pytest.raises(ValueError, match="feature count"):
        adapter.predict_logits(queries[:, :, :2])


def test_moving_one_predictor_does_not_mutate_the_other_control():
    model, config, support, labels, queries = fixture_model()
    adapter = FullCacheAdapter(model, model_config=config).build(support, labels)
    full = adapter.as_predictor()
    compact, _ = adapter.compile(slots=3, query_chunk_size=5)
    expected_full = full.predict_logits(queries)
    expected_compact = compact.predict_logits(queries)
    assert compact.model is full.model
    compact.to("cpu", dtype=torch.float64)
    assert compact.model is not full.model
    assert next(full.model.parameters()).dtype == torch.float32
    assert all(entry.key.dtype == torch.float32 for entry in full.entries.values())
    torch.testing.assert_close(full.predict_logits(queries), expected_full, rtol=0, atol=0)
    torch.testing.assert_close(compact.predict_logits(queries).float(), expected_compact, rtol=3e-5, atol=3e-6)
    with pytest.raises(ValueError, match="construction requires a float32"):
        FullCacheAdapter(compact.model, model_config=config).build(support, labels)
