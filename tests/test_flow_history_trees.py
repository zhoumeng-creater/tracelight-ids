from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ids.common import read_json
from ids.flow_history import trees
from ids.route_batch.metrics import evaluate, threshold


def _frames():
    rng = np.random.default_rng(5678)
    x = rng.uniform(size=(400, 3))
    data = pd.DataFrame(x, columns=["L4_SRC_PORT", "IN_BYTES", "history_prior_flows"])
    data["label"] = (x[:, 1] + x[:, 2] > 1).astype(int)
    data["row_id"] = [f"synthetic:{i}" for i in range(len(x))]
    return data.iloc[:220], data[data.label == 0], data.iloc[220:]


def test_train_predict_and_replay(tmp_path, monkeypatch):
    p, c, e = _frames()
    names = ["L4_SRC_PORT", "IN_BYTES", "history_prior_flows"]
    monkeypatch.setitem(trees.PARAMETERS, "max_iter", 12)
    path = trees.fit_arm(p, names, tmp_path / "hgb.joblib", threads=2)
    assert read_json(path.with_suffix(".fit.json"))["iterations_fitted"] == 12
    assert read_json(path.with_suffix(".fit.json"))["positive_rows"] == int(p.label.sum())
    model = trees.load_arm(path, threads=2)
    ref_c, ref_e = [trees.score_arm(model, frame, names) for frame in (c, e)]
    with pytest.raises(ValueError, match="order"):
        trees.score_arm(model, e, names[::-1])
    with pytest.raises(ValueError, match="finite"):
        trees.score_arm(model, e.assign(IN_BYTES=np.nan), names)
    report = trees.bench_arm(path, c, e, names, 1, reference_c=ref_c,
                             reference_e=ref_e, output_dir=tmp_path / "cpu")
    assert report["kind"] == "hgb" and report["device"] == "cpu"
    assert report["replay"]["equivalent_to_reference"]
    assert report["model_artifact_bytes"] == Path(path).stat().st_size
    assert report["metrics"]["tp"] == evaluate(e.label, ref_e, threshold(ref_c))["tp"]
    assert report["benchmark"]["single"]["n_observations"] == 192


def test_no_identity_or_label_features(tmp_path):
    p, _, _ = _frames()
    with pytest.raises(ValueError, match="labels and identifiers"):
        trees.fit_arm(p, ["row_id"], tmp_path / "unused", threads=1)
