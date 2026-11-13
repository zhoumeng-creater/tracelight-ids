"""Cost checks use the same synthetic fitted HGB fixtures as the scorer."""

from shutil import copyfile

import pytest

from ids.common import read_json
from ids.raw_workpoint import cost, scoring
from test_workpoint_scoring import build_inputs


def test_measured_decisions_and_model_sizes_match_frozen_synthetic_stream(tmp_path):
    rows, scores, _ = build_inputs(tmp_path)
    rules, extra = scoring.register(tmp_path, "hgb")
    for name, values in scoring.make_decisions(scores, rules, extra, rows).items():
        rows[name] = values
    for view in scoring.VIEWS:
        rows["score_" + view] = scores[view]
    base = tmp_path / "local/controls"
    frozen = base / "models/frozen"
    frozen.mkdir(parents=True)
    for view in scoring.VIEWS:
        model = "hgb-" + view + ".joblib"
        copyfile(tmp_path / model, frozen / model)
    calibration = base / "evidence/calibration"
    calibration.mkdir(parents=True)
    for file in ("hgb.json", "hgb.parquet"):
        copyfile(tmp_path / file, calibration / file)
    dest = tmp_path / "evidence/hgb-examples.parquet"
    dest.parent.mkdir()
    rows.sample(frac=1, random_state=13).to_parquet(dest, index=False)
    out = read_json(cost.measure(tmp_path)[0])
    assert out["row_ids"] == list(range(1, 11)) and out["verified"]["exact_rows_per_column"] == 10
    for policy, report in out["policies"].items():
        models = report["models"]
        assert report["combined_model_artifact_bytes"] == sum(out["models"][k]["bytes"] for k in models)
        assert len(models) == (2 if policy in ("add_contact", "add_quality") else 1)
        assert report["timed_single_record_decisions"] == 10 * cost.REPEATS
        assert 0 < report["p50_ms"] <= report["p95_ms"]
    rows.loc[rows.index[0], "add_quic30"] = not rows["add_quic30"].iloc[0]
    rows.to_parquet(dest, index=False)
    with pytest.raises(ValueError, match="Sealed decision differs for add_quic30"):
        cost.measure(tmp_path)
