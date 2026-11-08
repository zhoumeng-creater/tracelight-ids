"""Pinned native TabICL multiclass/cache checks using tiny random CPU models only.

No checkpoint, CSV, joblib, real features or real labels are read. The only
output is a compact JSON report beside this script. This is not a lab runner.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
VENDOR = ROOT / "local" / "vendor" / "tabicl"
PIN = "d83091b845a8c5f7c087bd54964f755a9e560f45"
sys.path.insert(0, str(VENDOR / "src"))

import torch  # noqa: E402
from tabicl._model.inference_config import InferenceConfig  # noqa: E402
from tabicl._model.tabicl import TabICL  # noqa: E402


CONFIG = dict(
    max_classes=10, embed_dim=16, col_num_blocks=1, col_nhead=2,
    col_num_inds=4, row_num_blocks=1, row_nhead=2, row_num_cls=2,
    icl_num_blocks=2, icl_nhead=4, ff_factor=2, dropout=0.0,
    norm_first=True, zero_init=False,
)


def tiny_model():
    # Same fixture as tests/test_tabicl_cache_adapter.py. Never loads weights.
    torch.manual_seed(812)
    model = TabICL(**CONFIG).eval()
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "query_mlp"):
                module.query_mlp[-1].weight.normal_(0, 0.08)
                module.query_mlp[-1].bias.fill_(0.03)
    return model


def inference_config():
    stage = dict(device=torch.device("cpu"), use_amp=False, use_fa3=False,
                 offload=False, use_async=False)
    return InferenceConfig(COL_CONFIG=dict(stage), ROW_CONFIG=dict(stage),
                           ICL_CONFIG=dict(stage))


def error(a, b):
    return float((a - b).abs().max())


def cache_layout(cache):
    result = {}
    for name in ("col_cache", "icl_cache"):
        group = getattr(cache, name)
        result[name] = {
            str(index): {"key_shape": list(entry.key.shape),
                         "value_shape": list(entry.value.shape),
                         "key_value_numel": entry.key.numel() + entry.value.numel()}
            for index, entry in sorted(group.kv.items())
        }
    return result


def cache_delta(before, after):
    result = {}
    for name in ("col_cache", "icl_cache"):
        left, right = getattr(before, name), getattr(after, name)
        assert left.kv.keys() == right.kv.keys()
        layer_changes = {}
        for index in sorted(left.kv):
            a, b = left.kv[index], right.kv[index]
            assert a.key.shape == b.key.shape and a.value.shape == b.value.shape
            layer_changes[str(index)] = {
                "key_max_abs_change": error(a.key, b.key),
                "value_max_abs_change": error(a.value, b.value),
            }
        # Test that both cache levels actually react to label changes.
        assert any(v["key_max_abs_change"] > 1e-6 or
                   v["value_max_abs_change"] > 1e-6
                   for v in layer_changes.values())
        result[name] = layer_changes
    return result


@torch.inference_mode()
def check_case(support, labels, queries, classes):
    model = tiny_model()
    settings = inference_config()
    direct = model(torch.cat([support, queries], dim=1), labels,
                   return_logits=True, inference_config=settings)
    result = model.forward_with_cache(
        X_train=support, y_train=labels, store_cache=True, use_cache=False,
        cache_mode="kv", inference_config=settings,
    )
    assert result is None
    cache = copy.deepcopy(model._cache)
    assert cache.num_classes == classes
    model.clear_cache()
    assert model._cache is None

    # A separately initialized model with identical random weights receives
    # only numeric queries and an explicit external cache, never P or labels.
    cold = tiny_model()
    assert cold._cache is None
    cached = cold.forward_with_cache(
        X_test=queries, cache=cache, return_logits=True,
        inference_config=settings,
    )
    cold.clear_cache()
    assert cold._cache is None
    assert list(cached.shape) == [2, 5, classes]
    torch.testing.assert_close(cached, direct, rtol=3e-5, atol=3e-6)
    assert cached.abs().max() > 1e-4

    # Explicit temperature=1 fixes the mathematical convention for this check.
    # This does not choose the eventual production ensemble/temperature policy.
    probabilities = cold.forward_with_cache(
        X_test=queries, cache=cache, return_logits=False,
        softmax_temperature=1.0, inference_config=settings,
    )
    cold.clear_cache()
    softmax = cached.softmax(dim=-1)
    torch.testing.assert_close(probabilities, softmax, rtol=3e-5, atol=3e-6)
    union_log_odds = torch.logsumexp(cached[..., 1:], dim=-1) - cached[..., 0]
    merged_attack_probability = softmax[..., 1:].sum(dim=-1)
    torch.testing.assert_close(union_log_odds.sigmoid(), merged_attack_probability,
                               rtol=3e-5, atol=3e-6)
    old_binary_error = None
    if classes == 2:
        old = cached[..., 1] - cached[..., 0]
        torch.testing.assert_close(union_log_odds, old, rtol=0, atol=0)
        old_binary_error = error(union_log_odds, old)
    return cache, {
        "num_classes": classes,
        "output_shape": list(cached.shape),
        "labels_per_table": torch.bincount(labels[0], minlength=classes).tolist(),
        "direct_vs_external_cache_max_abs_error": error(direct, cached),
        "native_probabilities_vs_softmax_max_abs_error": error(probabilities, softmax),
        "sigmoid_union_vs_softmax_attack_sum_max_abs_error": error(
            union_log_odds.sigmoid(), merged_attack_probability),
        "binary_union_vs_old_logit_difference_max_abs_error": old_binary_error,
        "fresh_model_had_no_internal_cache": True,
        "cached_query_arguments": ["X_test", "cache", "return_logits", "inference_config"],
        "cached_query_received_support_or_labels": False,
        "cache_layout": cache_layout(cache),
    }


def main():
    commit = subprocess.check_output(
        ["git", "-C", str(VENDOR), "rev-parse", "HEAD"], text=True).strip()
    assert commit == PIN, (commit, PIN)
    tracked_changes = subprocess.check_output(
        ["git", "-C", str(VENDOR), "status", "--porcelain", "--untracked-files=no"],
        text=True).strip()
    assert not tracked_changes, tracked_changes
    assert Path(sys.modules["tabicl"].__file__).resolve().is_relative_to(VENDOR)
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(716)
    support = torch.randn(2, 12, 3, generator=generator)
    queries = torch.randn(2, 5, 3, generator=generator)
    binary = torch.tensor([[0, 1] * 6] * 2)
    semantic = torch.tensor([[0, 1, 0, 1, 0, 2, 0, 2, 0, 2, 0, 2]] * 2)
    changed = torch.tensor([[0, 2, 0, 2, 0, 2, 0, 2, 0, 1, 0, 1]] * 2)
    assert torch.equal(torch.bincount(semantic[0]), torch.bincount(changed[0]))
    cache2, result2 = check_case(support, binary, queries, 2)
    cache3, result3 = check_case(support, semantic, queries, 3)
    cache_changed, result_changed = check_case(support, changed, queries, 3)
    assert cache_layout(cache2) == cache_layout(cache3)
    assert cache_layout(cache3) == cache_layout(cache_changed)

    adapter_path = ROOT / "src" / "ids" / "context_cache" / "tabicl_adapter.py"
    report = {
        "status": "passed",
        "data_kind": "synthetic tensors only",
        "model_kind": "tiny random initialization, no pretrained checkpoint",
        "vendor_commit": commit,
        "vendor_tracked_worktree_clean": True,
        "torch_version": torch.__version__,
        "device": "cpu",
        "num_threads": 1,
        "model_seed": 812,
        "tensor_seed": 716,
        "model_config": CONFIG,
        "cases": {"binary": result2, "three_class": result3,
                  "three_class_same_counts_changed_labels": result_changed},
        "cache_changes_binary_to_three_class": cache_delta(cache2, cache3),
        "cache_changes_three_class_same_counts_reassignment": cache_delta(cache3, cache_changed),
        "cache_shapes_and_tensor_counts_equal_for_two_and_three_classes": True,
        "formal_adapter_source_sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
        "formal_adapter_verification": "covered by tests/test_tabicl_multiclass.py and tests/test_label_workflow.py, not this native-only check",
        "cold_check_scope": "fresh model object and explicit in-memory cache, not artifact serialization or a new-process runner",
        "score_check_scope": "single-model logits at temperature 1; native numerical identity check",
        "real_checkpoint_csv_joblib_reads": 0,
        "real_model_runs": 0,
        "real_preprocessing_fits": 0,
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Passed: native 2/3-class shapes, both cache levels respond to labels, "
          "fresh-model cache-only prediction, direct/cache parity and union scores.")
    print("Production adapter and runner validation is recorded separately in implementation_validation.json.")


if __name__ == "__main__":
    main()
