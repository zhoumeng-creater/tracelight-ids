"""Synthetic semantic checks, including faults that identity tests can miss."""

import math

import pytest

torch = pytest.importorskip("torch")

from ids.context_cache.attention import prepare_query, weighted_attention  # noqa: E402
from ids.context_cache.cache import WeightedKVEntry  # noqa: E402


def fixture_entry(batch=2):
    gen = torch.Generator().manual_seed(42)
    key = torch.randn(batch, 2, 3, 4, generator=gen, dtype=torch.float64)
    value = torch.randn(batch, 2, 3, 4, generator=gen, dtype=torch.float64)
    counts = torch.tensor([1, 3, 2], dtype=torch.float64).view(1, 1, 1, 3).expand(batch, 2, 1, 3)
    return WeightedKVEntry(key, value, counts.log(), 6)


def test_mass_attention_equals_expanded_source_not_unweighted():
    cache = fixture_entry()
    counts = torch.tensor([1, 3, 2])
    full = WeightedKVEntry.full(cache.key.repeat_interleave(counts, dim=2),
                               cache.value.repeat_interleave(counts, dim=2))
    query = torch.randn(2, 2, 5, 4, dtype=torch.float64)
    phi = prepare_query(query, cache.source_count)
    actual = weighted_attention(phi, cache)
    torch.testing.assert_close(actual, weighted_attention(phi, full), atol=1e-12, rtol=1e-12)
    wrong = WeightedKVEntry.full(cache.key, cache.value)
    assert (actual - weighted_attention(phi, wrong)).abs().max() > 1e-3
    assert cache.tensor_bytes < full.tensor_bytes


def test_length_and_query_aware_scaling_order_is_observable():
    cache = fixture_entry()
    query = torch.ones(2, 2, 3, 4, dtype=torch.float64) * 0.7

    def nonlinear_scale(q, n):
        return q * math.log(n) * (1 + torch.tanh(q))

    phi = prepare_query(query, cache.source_count, nonlinear_scale)
    expected = nonlinear_scale(query, 6) / 2
    torch.testing.assert_close(phi, expected)
    wrong_order = nonlinear_scale(query / 2, 6)
    wrong_length = prepare_query(query, cache.num_slots, nonlinear_scale)
    for wrong in (wrong_order, wrong_length):
        assert (weighted_attention(phi, cache) - weighted_attention(wrong, cache)).abs().max() > 1e-3


def test_batch_slice_concat_cast_and_serialization_keep_mass_and_original_n(tmp_path):
    cache = fixture_entry()
    joined = WeightedKVEntry.concat([cache[0], cache[1]])
    torch.testing.assert_close(joined.key, cache.key)
    torch.testing.assert_close(joined.log_mass, cache.log_mass)
    assert joined.source_count == 6 and joined.num_slots == 3
    path = tmp_path / "cache.pt"
    joined.to("cpu", torch.float32).save(path)
    restored = WeightedKVEntry.load(path)
    assert restored.key.dtype == torch.float32
    assert restored.source_count == 6
    torch.testing.assert_close(restored.log_mass, cache.log_mass.float())
    assert set(restored.state_dict()) == {"key", "value", "log_mass", "source_count"}


def test_saved_slice_does_not_retain_omitted_batch_storage(tmp_path):
    sliced = fixture_entry(batch=4)[:1]
    assert sliced.key.untyped_storage().nbytes() > sliced.key.numel() * sliced.key.element_size()
    path = tmp_path / "slice.pt"
    sliced.save(path)
    loaded = WeightedKVEntry.load(path)
    for tensor in (loaded.key, loaded.value, loaded.log_mass):
        assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()


def test_full_attention_and_cache_gradient_match_explicit_operator():
    key = torch.randn(1, 2, 4, 3, dtype=torch.float64, requires_grad=True)
    value = torch.randn(1, 2, 4, 3, dtype=torch.float64, requires_grad=True)
    phi = torch.randn(1, 2, 2, 3, dtype=torch.float64)
    full = WeightedKVEntry.full(key, value)
    actual = weighted_attention(phi, full)
    expected = torch.softmax(phi @ key.transpose(-1, -2), dim=-1) @ value
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    actual_grad = torch.autograd.grad(actual.square().sum(), (key, value), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.square().sum(), (key, value))
    for left, right in zip(actual_grad, expected_grad):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)


def test_reject_inconsistent_mass_length_and_cross_context_concatenation():
    cache = fixture_entry()
    with pytest.raises(ValueError, match="sum to source_count"):
        WeightedKVEntry(cache.key, cache.value, cache.log_mass, 7)
    with pytest.raises(ValueError, match="positive integer"):
        WeightedKVEntry(cache.key, cache.value, cache.log_mass + 0.1, 6)
    with pytest.raises(ValueError, match="only on the batch"):
        WeightedKVEntry.concat([cache, cache], dim=2)
    with pytest.raises(ValueError, match="share head layout"):
        WeightedKVEntry.concat([cache, WeightedKVEntry.full(cache.key, cache.value)])
