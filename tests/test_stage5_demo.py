import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflows.stage5 import demo  # noqa: E402


def test_validation_pairs_use_saved_threshold_and_preserve_identity(tmp_path):
    threshold = np.nextafter(0.5, 1)
    frame = pd.DataFrame({"sample_id": ["a", "b", "c"], "true_label": [0, 1, 1],
                          "attack_category": ["Normal", "F", "F"], "score": [0.5, threshold, 0.1],
                          "threshold": threshold, "predicted_label": [0, 1, 0],
                          "split": "validation", "run_id": "r"})
    path = tmp_path / "pred.parquet"
    frame.to_parquet(path)
    model = {"run_id": "r", "threshold": threshold}
    pair = demo.paired_predictions([path, path], [model, model])
    assert demo._summary(pair) == [{"key": "11", "n": 1, "attacks": 1},
                                   {"key": "10", "n": 0, "attacks": 0}, {"key": "01", "n": 0, "attacks": 0},
                                   {"key": "00", "n": 2, "attacks": 1}]
    reverse = tmp_path / "reverse.parquet"
    frame.iloc[::-1].to_parquet(reverse)
    with pytest.raises(ValueError, match="identity"):
        demo.paired_predictions([path, reverse], [model, model])
    frame.loc[0, "predicted_label"] = 1
    frame.to_parquet(path)
    with pytest.raises(ValueError, match="threshold"):
        demo.paired_predictions([path, path], [model, model])


@pytest.mark.parametrize("contents", ["a,a\n1,2\n", "a,b,label\n1,2,0\n", "a,b\n1\n",
                                      "a,b\n", "a,b\ninf,2\n"])
def test_csv_schema_rejects_contamination_and_invalid_rows(tmp_path, contents):
    source = tmp_path / "data.csv"
    source.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        demo.read_input(source, "unsw", {"features": ["a", "b"], "categorical_features": []})


def test_cic_codes_and_unlabelled_inference_reach_static_html(tmp_path, monkeypatch):
    source = tmp_path / "features.csv"
    source.write_text("Flow Duration,Protocol\n4,6.0\n12,17\n", encoding="utf-8-sig")
    schema = {"features": ["Protocol", "Flow Duration"], "categorical_features": ["Protocol"],
              "data_kind": "fixture"}
    pair = [{"threshold": 0.5, "run_id": "one"}, {"threshold": 0.75, "run_id": "two"}]
    monkeypatch.setattr(demo, "artifacts", lambda *args: (schema, pair, None))

    def score(experiment, model, frame, threads):
        assert frame["Protocol"].tolist() == ["6", "17"] and threads == 1
        return np.array([0.6, 0.75])

    monkeypatch.setattr(demo, "_score", score)
    payload = demo.predict_dataset(tmp_path / "sealed", "cicids", source)
    assert payload["mode"] == "prediction" and payload["events"][0]["p0"] == 1 and payload["events"][0]["p1"] == 0
    assert all(item["attacks"] is None for item in payload["summary"])
    assert all("true_label" not in item for item in payload["events"])
    payload["events"][0]["sample_id"] = "</script><img onerror=x>"
    out = demo.render([payload], tmp_path / "demo.html", tmp_path / "sealed")
    raw = out.read_text(encoding="utf-8")
    assert "</script><img onerror=x>" not in raw
    body = re.search(r'<script type="application/json" id="records">(.*?)</script>', raw).group(1)
    assert json.loads(body)[0]["events"][0]["sample_id"] == "</script><img onerror=x>"
    with pytest.raises(ValueError, match="new"):
        demo.render([payload], out, tmp_path / "sealed")
    with pytest.raises(ValueError, match="outside"):
        demo.render([payload], tmp_path / "sealed" / "any.html", tmp_path / "sealed")


def test_nonaccepted_experiment_cannot_reach_model_load(tmp_path):
    (tmp_path / "STAGE4_COMPLETE.json").write_text('{"files":{}}')
    with pytest.raises(ValueError, match="checksum"):
        demo.artifacts(tmp_path.resolve(), "unsw")
