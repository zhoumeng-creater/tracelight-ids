import json

import pytest

from ids import reporting


def test_report_requires_same_dataset_and_escapes_html(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    for name, dataset in [("a", "one"), ("b", "two")]:
        run = root / name
        run.mkdir(parents=True)
        (run / "run.json").write_text(json.dumps({"run_id": name, "model_name": "lr", "data_kind": "synthetic",
                                                 "dataset_manifest_sha256": dataset}))
        for file in ["validation_metrics", "validation_threshold", "examples", "selected_features", "config"]:
            (run / f"{file}.json").write_text(json.dumps({"hostile": "</script><script>alert(1)</script>"}))
    monkeypatch.setattr(reporting, "verify_run", lambda p: json.loads((p / "run.json").read_text()))
    with pytest.raises(ValueError, match="identical"):
        reporting.build_report(root, tmp_path / "out.html")
    target = reporting.build_report(root / "a", tmp_path / "out.html")
    html = target.read_text(encoding="utf-8")
    assert "</script><script>alert(1)</script>" not in html
    assert r"\u003c/script\u003e" in html
    assert "connect-src 'none'" in html


def test_empty_report_is_not_fake_results(tmp_path):
    with pytest.raises(ValueError, match="No completed"):
        reporting.build_report(tmp_path, tmp_path / "report.html")
