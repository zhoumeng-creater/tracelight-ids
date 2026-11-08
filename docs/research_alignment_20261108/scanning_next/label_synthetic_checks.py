"""Synthetic checks of diagnostic labels and score semantics, not IDS effects."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from label_reference import (
    attack_union_log_odds,
    grouped_sham_support_labels,
    semantic_support_labels,
)


def main():
    checks = []

    def check(name, passed, detail=None):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            raise AssertionError(name)

    def raises(name, function):
        try:
            function()
        except ValueError:
            check(name, True)
        else:
            check(name, False)

    # Repeated inputs and enough singleton groups to fill the fixed small class.
    pattern_ids = np.array([0] * 4 + [1] * 3 + [2] * 2 + list(range(3, 17)))
    x = np.zeros((len(pattern_ids), 48), dtype=np.float32)
    x[:, 0] = pattern_ids
    binary = (pattern_ids != 0).astype(np.int64)
    families = np.where(binary == 0, "Benign", np.where(pattern_ids <= 2, "Scanning", "DDoS"))
    original_x, original_y = x.tobytes(), binary.tobytes()
    semantic = semantic_support_labels(binary, families)
    check("semantic_partition_preserves_original_binary_target", np.array_equal(semantic > 0, binary > 0))
    check("semantic_counts_are_4_5_14", np.bincount(semantic).tolist() == [4, 5, 14])
    check("only_support_scanning_gets_semantic_class1", np.array_equal(semantic == 1, families == "Scanning"))
    sham, metadata = grouped_sham_support_labels(x, binary, 5)
    check("sham_exactly_matches_semantic_counts", np.array_equal(np.bincount(sham), np.bincount(semantic)))
    check("sham_preserves_normal_attack_partition", np.array_equal(sham > 0, binary > 0))
    check("sham_does_not_split_repeated_attack_input", all(np.unique(sham[pattern_ids == k]).size == 1 for k in range(1, 17)))
    check("caller_arrays_not_mutated", x.tobytes() == original_x and binary.tobytes() == original_y)
    again, again_metadata = grouped_sham_support_labels(x, binary, 5)
    check("fixed_hash_assignment_repeatable", np.array_equal(sham, again) and metadata == again_metadata)
    order = np.arange(len(x))[::-1]
    reordered, _ = grouped_sham_support_labels(x[order], binary[order], 5)
    check("assignment_independent_of_input_row_order", np.array_equal(reordered, sham[order]))
    signed_x = x.copy()
    signed_x[4, 1] = -0.0
    signed, _ = grouped_sham_support_labels(signed_x, binary, 5)
    check("signed_zero_does_not_create_pseudo_class_conflict", np.array_equal(signed, sham))

    impossible_x = np.zeros((9, 48), dtype=np.float32)
    impossible_x[1:5, 0] = 1
    impossible_x[5:, 0] = 2
    impossible_y = np.array([0] + [1] * 8)
    raises("unfillable_whole_group_quota_fails_without_seed_search", lambda: grouped_sham_support_labels(impossible_x, impossible_y, 3))
    raises("binary_family_disagreement_rejected", lambda: semantic_support_labels(binary, np.repeat("Benign", len(binary))))
    raises("missing_semantic_class_rejected", lambda: semantic_support_labels(binary, np.where(binary == 0, "Benign", "DDoS")))
    raises("non_FP32_support_rejected", lambda: grouped_sham_support_labels(x.astype(np.float64), binary, 5))

    z2 = np.array([[0, 0], [-2, 7], [1000, -1000], [-1000, 1000]], dtype=np.float32)
    score2 = attack_union_log_odds(z2)
    check("binary_score_byte_exact_to_existing_subtraction", score2.tobytes() == (z2[:, 1] - z2[:, 0]).tobytes())
    z3 = np.array([[0, 1, 2], [2, -1, 4], [-3, -4, -5], [3, 2.5, 2.5]], dtype=np.float64)
    p3 = np.exp(z3 - z3.max(axis=1, keepdims=True))
    p3 /= p3.sum(axis=1, keepdims=True)
    expected = np.log(p3[:, 1] + p3[:, 2]) - np.log(p3[:, 0])
    score3 = attack_union_log_odds(z3)
    check("three_class_score_is_all_attack_probability_log_odds", np.allclose(score3, expected, atol=1e-14, rtol=0))
    check("attack_output_column_swap_preserves_union", np.array_equal(score3, attack_union_log_odds(z3[:, [0, 2, 1]])))
    check("common_logit_shift_does_not_change_union", np.allclose(score3, attack_union_log_odds(z3 + 700), atol=1e-12, rtol=0))
    check("extreme_logits_stay_finite", np.isfinite(attack_union_log_odds(np.array([[1000., -1000., 999.], [-1000., 1000., 1000.]]))).all())
    check("binary_argmax_is_not_a_valid_union_replacement", z3[-1].argmax() == 0 and score3[-1] > 0,
          "Normal is largest single class, while the sum of both attacks exceeds normal.")
    split_z = np.column_stack([z2[:, 0].astype(float), z2[:, 1].astype(float) - np.log(2), z2[:, 1].astype(float) - np.log(2)])
    check("splitting_one_attack_logit_mass_in_two_is_score_invariant", np.allclose(attack_union_log_odds(split_z), score2, atol=1e-12, rtol=0))

    # At a fixed calibrated rank, positive global class reweighting only shifts
    # binary log odds; it cannot reorder targets against the normal tail.
    calibration = np.array([-4., -1., 0., 1., 2., 2., 4.])
    queries = np.array([-2., 1., 2., 2.0001, 4., 9.])
    threshold = np.sort(calibration)[4]
    shift = np.log(7.)
    check("global_binary_class_weight_preserves_calibrated_decisions", np.array_equal(queries > threshold, queries + shift > np.sort(calibration + shift)[4]))
    check("strict_threshold_keeps_all_equal_scores_together", np.array_equal(queries > threshold, [False, False, False, True, True, True]))

    root = Path(__file__).resolve().parent
    result = {
        "format": "tracelight-scanning-label-reference-synthetic-v1",
        "scope": "fabricated arrays only; no real model, support labels, preprocessing fit or sham assignment",
        "passed": len(checks), "failed": 0, "checks": checks,
        "real_pretrained_model_runs": 0, "real_support_sham_assignments": 0,
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ("label_reference.py", "label_synthetic_checks.py")},
    }
    path = root / "label_synthetic_checks.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": len(checks), "failed": 0, "output": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
