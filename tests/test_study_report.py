"""Renderer boundaries; browser interaction is verified separately on real artifacts."""
import copy
import json
import re

import pytest

from ids.study_report import render_study_report


def study_fixture():
    """Synthetic measurements for rendering tests, never research evidence."""
    def stats(values):
        mean = sum(values) / len(values)
        return {"mean": mean, "std": None if len(values) < 2 else
                (sum((x - mean) ** 2 for x in values) / (len(values) - 1)) ** .5,
                "min": min(values), "max": max(values), "n": len(values), "values": values}

    seeds = [17, 29, 43]
    groups = []
    for model, budget, size in [("lr", 42, 42000), ("rf", 8, 800000)]:
        group_id = f"{model}-{budget}"
        groups.append({
            "id": group_id, "spec": {"model": model, "feature_method": "all" if budget == 42 else
                                       "mutual_info", "raw_feature_budget": budget, "hyperparameters": {}},
            "seeds": seeds, "n_runs": 3,
            "runs": [{"run_key": f"{group_id}-{seed}", "seed": seed,
                      "selected_features": ["dur", "proto", "service"],
                      "environment": {"processor": "Synthetic CPU fixture", "torch": {
                          "gpus": [{"name": "Synthetic GPU fixture"}]}}} for seed in seeds],
            "metrics": {"attack_recall": stats([.60, .63, .66]), "normal_fpr": stats([.01, .01, .009]),
                        "average_precision": stats([.8, .81, .82]),
                        "model_file_bytes": stats([size, size + 100, size + 200])},
            "per_attack": [{"category": "Fuzzers", "support": 100, "recall": stats([.2, .3, .4])},
                           {"category": "Exploits", "support": 100, "recall": stats([.8, .85, .9])}],
            "features": {"mean_pairwise_jaccard": 1., "pairwise_jaccard": [1., 1., 1.]},
            "cpu_benchmarks": [],
        })
    return {"schema_version": 1, "phase": "validation_development", "data_kind": "synthetic",
            "dataset_manifest_sha256": "test-fixture-not-real-data", "n_validation_samples": 300,
            "generated_at": "2026-11-01T00:00:00Z", "groups": groups, "warnings": [],
            "comparisons": {"common_seeds": seeds, "candidate_group_ids": [g["id"] for g in groups],
                            "per_seed": [
                                {"seed": seed, "n_attacks": 200, "n_normal": 100,
                                 "or_alerts": {"attack_recall": .72, "normal_fpr": .02,
                                               "false_positives": 2},
                                 "unique_detections": [{"group_id": "lr-42", "count": 10},
                                                       {"group_id": "rf-8", "count": 15}],
                                 "pairwise": [{"left": "lr-42", "right": "rf-8",
                                               "left_only_detected": 10, "right_only_detected": 15,
                                               "both_missed": 30, "or_attack_recall": .72,
                                               "or_normal_fpr": .02, "decision_agreement": .9}]}
                                for seed in seeds]}}


def test_roundtrip_data_and_missing_benchmarks(tmp_path):
    study = study_fixture()
    before = copy.deepcopy(study)
    output = tmp_path / "nested" / "study.html"
    assert render_study_report(study, output) == output
    document = output.read_text(encoding="utf-8")
    payload = re.search(r'<script id="study-data" type="application/json">(.*?)</script>',
                        document, re.S).group(1)
    assert json.loads(payload) == before
    assert study == before
    assert json.loads(payload)["groups"][0]["cpu_benchmarks"] == []
    assert "未测量" in document
    assert "不是置信区间" in document
    assert "正常误报" in document
    assert "未展示最终 test 结果" in document
    assert "connect-src 'none'" in document
    assert "fetch(" not in document and "XMLHttpRequest" not in document
    assert ".innerHTML" not in document


def test_hostile_labels_are_inert_and_lossless(tmp_path):
    study = study_fixture()
    hostile = '</script><img src=x onerror="alert(1)">&\u2028\u2029'
    study["groups"][0]["per_attack"][0]["category"] = hostile
    study["warnings"] = [hostile]
    document = render_study_report(study, tmp_path / "safe.html").read_text(encoding="utf-8")
    assert hostile not in document
    assert r"\u003c/script\u003e" in document
    assert r"\u0026\u2028\u2029" in document
    payload = re.search(r'<script id="study-data" type="application/json">(.*?)</script>',
                        document, re.S).group(1)
    assert json.loads(payload)["warnings"] == [hostile]
    assert document.count("</script>") == 2


@pytest.mark.parametrize("invalid", [{}, {"groups": []}, {"groups": [None]}])
def test_empty_or_invalid_study_is_not_presented_as_results(tmp_path, invalid):
    with pytest.raises(ValueError, match="groups"):
        render_study_report(invalid, tmp_path / "invalid.html")
    assert not (tmp_path / "invalid.html").exists()


@pytest.mark.parametrize("phase", ["test", "calibration", None])
def test_renderer_cannot_relabel_final_test_as_validation(tmp_path, phase):
    study = study_fixture()
    study["phase"] = phase
    with pytest.raises(ValueError, match="validation_development"):
        render_study_report(study, tmp_path / "wrong-phase.html")


def test_nonfinite_metric_is_rejected_before_overwriting(tmp_path):
    study = study_fixture()
    output = tmp_path / "existing.html"
    output.write_text("previous artifact", encoding="utf-8")
    study["groups"][0]["metrics"]["attack_recall"]["mean"] = float("nan")
    with pytest.raises(ValueError):
        render_study_report(study, output)
    assert output.read_text(encoding="utf-8") == "previous artifact"


def test_distillation_and_repeated_timing_evidence_remain_explicit(tmp_path):
    study = study_fixture()
    for group, variant in zip(study["groups"], ["student_hard", "student_kd"]):
        group["spec"].update(model="mlp", variant=variant, raw_feature_budget=42)
    study["distillation_comparisons"] = [{
        "kind": "distillation_effect", "status": "paired", "candidate_group_id": "rf-8",
        "control_group_id": "lr-42", "common_seeds": [17], "n_pairs": 1,
        "complete_expected_seeds": False, "missing_expected_seeds": [29, 43],
        "per_seed": [], "per_attack": [], "cpu_comparisons": [], "metrics": {}}]
    study["groups"][1]["repeated_benchmarks"] = [{
        "spec": {"planned_rounds": 3, "threads": 1, "batch_size": 256}, "seeds": [],
        "observed_seeds": [17], "missing_seeds": [17, 29, 43], "status": "incomplete_rounds",
        "per_seed": [{"seed": 17, "rounds": [1], "missing_rounds": [2, 3]}], "metrics": {}}]
    document = render_study_report(study, tmp_path / "distillation.html").read_text(encoding="utf-8")
    payload = re.search(r'<script id="study-data" type="application/json">(.*?)</script>', document, re.S).group(1)
    assert json.loads(payload) == study
    assert 'id="distill-families"' in document and 'id="distill-seeds"' in document
    assert "新增漏检" in document and "恢复检出" in document
    assert "轮p95中位数再跨seed均值" in document
    assert "教师额外成本" in document
    assert "不可将本机结果写成实验室" not in document
