"""The overview must identify engineering data before showing any scores."""
from ids.stage4 import render_overview


def test_overview_labels_synthetic_scope_and_actual_seed_count(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    study = {"data_kind": "synthetic", "n_runs": 1, "groups": [{
        "seeds": [17], "spec": {"model": "rf", "feature_method": "all", "feature_config": {},
                                 "raw_feature_budget": 42},
        "metrics": {"attack_recall": {"mean": .5}, "model_file_bytes": {"mean": 100}}}]}
    render_overview(tmp_path, {"unsw": study}, {"unsw": {"methods": {}}},
                    {"unsw": {"comparisons": []}})
    output = (reports / "index.html").read_text(encoding="utf-8")
    assert "synthetic" in output and "合成工程预检" in output
    assert "训练种子：1" in output and "不用于论文性能结论" in output
