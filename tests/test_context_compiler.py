"""Synthetic operator checks for the 2026 QCOC implementation, no checkpoint."""

import math

import pytest

torch = pytest.importorskip("torch")

from ids.context_cache.cache import WeightedKVEntry  # noqa: E402
from ids.context_cache.compiler import _joint_kv_prototypes, compile_entry  # noqa: E402


def full_fixture(*, batches=1, heads=1, n=17, width=4, q=19, dtype=torch.float64):
    generator = torch.Generator().manual_seed(416)
    key = torch.randn(batches, heads, n, width, generator=generator, dtype=dtype)
    value = torch.randn(batches, heads, n, width, generator=generator, dtype=dtype)
    phi = torch.randn(batches, heads, q, width, generator=generator, dtype=dtype) / math.sqrt(width)
    return WeightedKVEntry.full(key, value), phi


def attention(entry, phi):
    return (phi @ entry.key.transpose(-1, -2) + entry.log_mass).softmax(dim=-1) @ entry.value


def test_identity_is_exact_independent_and_uses_no_fit():
    full, phi = full_fixture(batches=2, heads=2)
    compiled, report = compile_entry(full, phi, slots=17)
    assert report["identity"] and report["target_attention_passes"] == 0
    assert torch.equal(attention(compiled, phi), attention(full, phi))
    for name in ("key", "value", "log_mass"):
        assert getattr(compiled, name).data_ptr() != getattr(full, name).data_ptr()
    compiled.value.zero_()
    assert torch.count_nonzero(full.value) > 0


def test_unequal_duplicate_masses_preserve_operator():
    key = torch.tensor([[[[1., 0.]] * 4 + [[-1., 0.]]]], dtype=torch.float64)
    value = torch.tensor([[[[2., -3.]] * 4 + [[7., 1.]]]], dtype=torch.float64)
    full = WeightedKVEntry.full(key, value)
    phi = torch.tensor([[[[-2., 1.], [0., 0.], [2., 0.]]]], dtype=torch.float64)
    compiled, report = compile_entry(full, phi, slots=2, query_chunk_size=1)
    torch.testing.assert_close(compiled.log_mass.exp().sort().values,
                               torch.tensor([[[[1., 4.]]]], dtype=compiled.log_mass.dtype))
    assert compiled.source_count == 5
    torch.testing.assert_close(attention(compiled, phi), attention(full, phi), rtol=1e-12, atol=1e-12)
    assert report["heads"][0]["min_mass"] == 1


def test_chunked_fit_matches_dense_regression_and_reduces_anchor_objective():
    full, phi = full_fixture(batches=2, heads=2)
    compiled, report = compile_entry(full, phi, slots=4, query_chunk_size=3)
    dense, _ = compile_entry(full, phi, slots=4, query_chunk_size=1000)
    torch.testing.assert_close(compiled.key, dense.key, rtol=0, atol=0)
    torch.testing.assert_close(compiled.log_mass, dense.log_mass, rtol=0, atol=0)
    torch.testing.assert_close(compiled.value, dense.value, rtol=1e-11, atol=1e-11)
    for b in range(2):
        for h in range(2):
            _, anchor, _, _ = _joint_kv_prototypes(full.key[b, h], full.value[b, h], 4, 15)
            p = (phi[b, h] @ compiled.key[b, h].T + compiled.log_mass[b, h]).softmax(dim=-1)
            target = (phi[b, h] @ full.key[b, h].T).softmax(dim=-1) @ full.value[b, h]
            coefficient = 1e-3 * torch.trace(p.T @ p) / 4
            stationarity = (p.T @ p + coefficient * torch.eye(4, dtype=p.dtype)) @ compiled.value[b, h]
            torch.testing.assert_close(stationarity, p.T @ target + coefficient * anchor,
                                       rtol=1e-11, atol=1e-11)
    assert compiled.key.shape == (2, 2, 4, 4)
    assert compiled.log_mass.shape == (2, 2, 1, 4)
    torch.testing.assert_close(compiled.log_mass.exp().sum(dim=-1),
                               torch.full((2, 2, 1), 17., dtype=compiled.log_mass.dtype))
    for info in report["heads"]:
        assert info["fitted_objective_float64"] <= info["anchor_sse"] + 1e-11
        assert info["stored_value_sse"] <= info["anchor_sse"] + 1e-11


@pytest.mark.parametrize("constant", [0., 4.])
def test_identical_states_keep_all_slots_nonempty_and_constant_output(constant):
    key = torch.full((1, 1, 9, 3), constant, dtype=torch.float64)
    value = torch.full((1, 1, 9, 3), constant, dtype=torch.float64)
    phi = torch.arange(21, dtype=torch.float64).reshape(1, 1, 7, 3)
    compiled, report = compile_entry(WeightedKVEntry.full(key, value), phi, slots=5)
    assert compiled.key.shape[-2] == 5
    assert bool((compiled.log_mass.exp() >= 1 - 1e-12).all())
    assert compiled.log_mass.exp().sum().item() == pytest.approx(9)
    torch.testing.assert_close(attention(compiled, phi), torch.full_like(phi, constant))
    info = report["heads"][0]
    assert info["zero_key_scale"] and info["zero_value_scale"]
    assert info["final_count_empty_repairs"] == 4


def test_one_slot_preserves_n_and_solves_constant_regression():
    full, phi = full_fixture()
    compiled, report = compile_entry(full, phi, slots=1)
    target = attention(full, phi)[0, 0]
    anchor = full.value[0, 0].mean(dim=0)
    # P is identically one, gamma=Q and lambda*gamma=0.001*Q.
    expected = (target.mean(dim=0) + .001 * anchor) / 1.001
    torch.testing.assert_close(compiled.value[0, 0, 0], expected, rtol=1e-12, atol=1e-12)
    assert compiled.source_count == full.source_count
    assert compiled.log_mass.exp().item() == pytest.approx(17)
    assert report["heads"][0]["gamma"] == 19


def test_float32_export_reports_actual_rounded_value_error():
    full, phi = full_fixture(dtype=torch.float32)
    compiled, report = compile_entry(full, phi, slots=3)
    p = (phi.double() @ compiled.key.double().transpose(-1, -2) + compiled.log_mass.double()).softmax(-1)
    y = (phi.double() @ full.key.double().transpose(-1, -2)).softmax(-1) @ full.value.double()
    sse = (p @ compiled.value.double() - y).square().sum().item()
    assert compiled.key.dtype == compiled.value.dtype == torch.float32
    assert report["heads"][0]["stored_value_sse"] == pytest.approx(sse, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("kwargs", [{"slots": 18}, {"slots": 0}, {"slots": 4, "iterations": 16},
                                    {"slots": 4, "query_chunk_size": 0}, {"slots": 4, "ridge": 0}])
def test_invalid_compilation_budget_rejected(kwargs):
    full, phi = full_fixture()
    with pytest.raises(ValueError):
        compile_entry(full, phi, **kwargs)


def test_recompression_and_wrong_query_shape_rejected():
    full, phi = full_fixture()
    compiled, _ = compile_entry(full, phi, slots=4)
    with pytest.raises(ValueError, match="full unit-mass"):
        compile_entry(compiled, phi, slots=2)
    with pytest.raises(ValueError, match="shape"):
        compile_entry(full, phi[:, :, :, :2], slots=2)
    with pytest.raises(ValueError, match="finite"):
        compile_entry(full, phi * float("nan"), slots=2)
