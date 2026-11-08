"""Pinned tiny random models and synthetic arrays only; no pretrained weights."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from ids.context_cache.tabicl_adapter import (  # noqa: E402
    CompactPredictor,
    FullCacheAdapter,
    attack_union_log_odds,
)


@pytest.fixture(scope="module", autouse=True)
def official_source_and_threads():
    if importlib.util.find_spec("tabicl") is None:
        pytest.skip("put the verified pinned TabICL source on PYTHONPATH; no auto-download")
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def fixture(classes=3):
    from tabicl._model.tabicl import TabICL
    config = dict(max_classes=10, embed_dim=16, col_num_blocks=1, col_nhead=2,
                  col_num_inds=4, row_num_blocks=1, row_nhead=2, row_num_cls=2,
                  icl_num_blocks=2, icl_nhead=4, ff_factor=2, dropout=0.0,
                  norm_first=True, zero_init=False)
    torch.manual_seed(812)
    model = TabICL(**config).eval()
    generator = torch.Generator().manual_seed(716)
    support = torch.randn(2, 12, 3, generator=generator)
    labels = torch.stack([torch.arange(12) % classes, torch.arange(12).flip(0) % classes])
    query = torch.randn(2, 5, 3, generator=generator)
    return model, config, support, labels, query


@pytest.mark.parametrize("classes", [2, 3])
def test_full_native_logits_and_attack_union(classes):
    model, config, support, labels, query = fixture(classes)
    adapter = FullCacheAdapter(model, model_config=config, num_classes=classes).build(support, labels)
    predictor = adapter.as_predictor()
    logits = predictor.predict_logits(query)
    assert logits.shape == (2, 5, classes)
    assert predictor.num_classes == adapter.num_classes == classes
    assert predictor.class_contract["attack_classes"] == list(range(1, classes))
    torch.testing.assert_close(logits, adapter.predict_logits(query), atol=3e-6, rtol=3e-5)
    score = predictor.score(query)
    if classes == 2:
        assert torch.equal(score, logits[..., 1] - logits[..., 0])
    else:
        probability = torch.softmax(logits, dim=-1)[..., 1:].sum(dim=-1)
        torch.testing.assert_close(score.sigmoid(), probability, atol=2e-7, rtol=0)
    assert model._cache is None


def test_union_handles_extreme_logits_and_preserves_binary_bits():
    binary = torch.tensor([[1000., -1000.], [0., 0.], [-3.75, -9.5]])
    assert torch.equal(attack_union_log_odds(binary), binary[:, 1] - binary[:, 0])
    logits = torch.tensor([[3., 2.5, 2.5], [1000., 999., -1000.], [-1000., 1000., 1000.]])
    score = attack_union_log_odds(logits)
    assert bool(torch.isfinite(score).all())
    assert int(logits[0].argmax()) == 0 and score[0] > 0
    torch.testing.assert_close(score, attack_union_log_odds(logits[:, [0, 2, 1]]), rtol=0, atol=0)
    with pytest.raises(ValueError, match="finite"):
        attack_union_log_odds(torch.tensor([[float("nan"), 0., 1.]]))


def test_label_reassignment_rebuilds_both_levels_without_changing_layout():
    model, config, support, labels, query = fixture()
    first = FullCacheAdapter(model, model_config=config).build(support, labels)
    before = first.as_predictor()
    baseline = before.predict_logits(query)
    # All class counts remain fixed, but their support assignments change.
    changed = labels.roll(1, dims=1)
    second = FullCacheAdapter(model, model_config=config).build(support, changed)
    after = second.as_predictor()
    for left, right in ((before.column_cache.kv, after.column_cache.kv), (before.entries, after.entries)):
        assert left.keys() == right.keys()
        assert all(left[i].key.shape == right[i].key.shape and left[i].value.shape == right[i].value.shape
                   for i in left)
        assert any(not torch.equal(left[i].key, right[i].key) or not torch.equal(left[i].value, right[i].value)
                   for i in left)
    assert not torch.equal(baseline, after.predict_logits(query))
    # Changing another adapter must not overwrite the first adapter's cache.
    torch.testing.assert_close(before.predict_logits(query), baseline, rtol=0, atol=0)


def test_three_class_slice_serialization_and_fresh_process(tmp_path):
    model, config, support, labels, query = fixture()
    full = FullCacheAdapter(model, model_config=config).build(support, labels).as_predictor()
    sliced = full.slice_batch(1, 2)
    assert sliced.num_classes == 3
    expected = sliced.predict_logits(query[1:2])
    path = sliced.save(tmp_path / "three-class.pt")
    payload = torch.load(path, weights_only=True)
    assert payload["format"] == "tracelight-tabicl-numeric-cache-v2"
    assert payload["num_classes"] == 3
    assert payload["class_contract"] == full.class_contract
    loaded = CompactPredictor.load(path)
    torch.testing.assert_close(loaded.predict_logits(query[1:2]), expected, rtol=0, atol=0)
    queries, output = tmp_path / "queries.pt", tmp_path / "output.pt"
    torch.save(query[1:2], queries)
    script = (
        "import sys,torch; from ids.context_cache.tabicl_adapter import CompactPredictor; "
        "torch.set_num_threads(1); p=CompactPredictor.load(sys.argv[1]); "
        "x=torch.load(sys.argv[2],weights_only=True); "
        "torch.save({'logits':p.predict_logits(x),'score':p.score(x),'classes':p.num_classes},sys.argv[3]); "
        "assert p.model._cache is None"
    )
    env = dict(os.environ)
    repo = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join([str(repo / "src"), *sys.path])
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run([sys.executable, "-c", script, str(path), str(queries), str(output)],
                            capture_output=True, text=True, encoding="utf-8", env=env, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    cold = torch.load(output, weights_only=True)
    assert cold["classes"] == 3
    torch.testing.assert_close(cold["logits"], expected, rtol=0, atol=0)
    torch.testing.assert_close(cold["score"], sliced.score(query[1:2]), rtol=0, atol=0)


def test_binary_legacy_artifact_keeps_its_contract_and_keys(tmp_path):
    model, config, support, labels, query = fixture(2)
    predictor = FullCacheAdapter(model, model_config=config).build(support, labels).as_predictor()
    path = predictor.save(tmp_path / "binary-v1.pt")
    payload = torch.load(path, weights_only=True)
    assert payload["format"] == "tracelight-tabicl-numeric-cache-v1"
    assert payload["num_classes"] == 2 and "class_contract" not in payload
    loaded = CompactPredictor.load(path)
    assert loaded.class_contract == predictor.class_contract
    torch.testing.assert_close(loaded.score(query), predictor.score(query), rtol=0, atol=0)
    payload["num_classes"] = 3
    torch.save(payload, path)
    with pytest.raises(ValueError, match="legacy numeric cache"):
        CompactPredictor.load(path)


@pytest.mark.parametrize("field,value", [
    ("normal_class", 1), ("attack_classes", [1]),
    ("attack_classes", [2, 1]), ("temperature", 0.9),
    ("score", "scanning-only"),
])
def test_changed_multiclass_mapping_or_score_contract_rejected(tmp_path, field, value):
    model, config, support, labels, _ = fixture()
    predictor = FullCacheAdapter(model, model_config=config).build(support, labels).as_predictor()
    path = predictor.save(tmp_path / "three-class.pt")
    payload = torch.load(path, weights_only=True)
    payload["class_contract"][field] = value
    torch.save(payload, path)
    with pytest.raises(ValueError, match="class/score contract"):
        CompactPredictor.load(path)


@pytest.mark.parametrize("problem", ["missing_zero", "gap", "different_contexts", "fractional", "nonfinite", "capacity", "requested"])
def test_invalid_support_class_contract_fails_before_build(problem):
    model, config, support, labels, _ = fixture()
    requested = None
    if problem == "missing_zero":
        labels = labels + 1
    elif problem == "gap":
        labels = labels * 2
    elif problem == "different_contexts":
        labels[1, labels[1] == 2] = 1
    elif problem == "fractional":
        labels = labels.float() + 0.25
    elif problem == "nonfinite":
        labels = labels.float()
        labels[0, 0] = float("nan")
    elif problem == "capacity":
        labels = torch.arange(12).repeat(2, 1)
    else:
        requested = 2
    adapter = FullCacheAdapter(model, model_config=config, num_classes=requested)
    with pytest.raises(ValueError):
        adapter.build(support, labels)
    assert adapter.cache is None and adapter.num_classes is None


def test_compiled_multiclass_retains_mapping_and_score(tmp_path):
    model, config, support, labels, query = fixture()
    adapter = FullCacheAdapter(model, model_config=config).build(support, labels)
    compact, _ = adapter.compile(slots=3, query_chunk_size=5)
    assert compact.num_classes == 3
    assert compact.predict_logits(query).shape == (2, 5, 3)
    path = compact.save(tmp_path / "synthetic-compiled-three.pt")
    cold = CompactPredictor.load(path)
    torch.testing.assert_close(cold.score(query), compact.score(query), rtol=0, atol=0)
