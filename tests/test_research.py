"""Scientific boundaries and controlled training for the new research version."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ids.common import read_json
from ids.data import make_synthetic_dataset, prepare_dataset
from ids.research import group_budget, run_research, train_research_mlp, validate_config, verify_research


def configuration():
    root = Path(__file__).resolve().parents[1]
    value = yaml.safe_load((root / "configs/research-v2.yaml").read_text(encoding="utf-8"))
    value.update(seeds=[17], folds=2, max_train_rows=100, max_validation_rows=40, cohort_policy="bounded_pilot")
    value["teacher"]["hyperparameters"].update(max_iter=3, min_samples_leaf=2)
    value["models"] = copy.deepcopy(value["models"][:2])
    for model in value["models"]:
        model["hyperparameters"].update(epochs=2, patience=2, batch_size=32)
    return value


@pytest.fixture
def prepared(tmp_path):
    raw, data = tmp_path / "raw", tmp_path / "prepared"
    make_synthetic_dataset(raw, n_train=300, n_test=80)
    prepare_dataset(raw / "UNSW_NB15_training-set.csv", raw / "UNSW_NB15_testing-set.csv", data,
                    data_kind="synthetic")
    return data


def test_whole_group_budget_keeps_groups_and_does_not_overflow():
    frame = pd.DataFrame({"sample_id": [str(i) for i in range(12)], "feature_hash": [f"f{i // 2}" for i in range(12)],
                          "label": [0, 0, 1, 1] * 3})
    result = group_budget(frame, 8, 19)
    assert len(result) <= 8
    assert set(result["feature_hash"]).issubset(set(frame["feature_hash"]))
    assert (result.groupby("feature_hash").size() == 2).all()
    pd.testing.assert_frame_equal(result, group_budget(frame, 8, 19))


def test_run_never_loads_or_hashes_reserved_partitions(prepared, tmp_path, monkeypatch):
    import ids.research as research

    original = research.load_partition
    accessed = []

    def tracked(path, split):
        accessed.append(split)
        assert split in {"train", "validation"}
        return original(path, split)

    monkeypatch.setattr(research, "load_partition", tracked)
    # A reserved-partition read would fail, even if made without the tracked API.
    (prepared / "test.parquet").unlink()
    (prepared / "calibration.parquet").unlink()
    output = tmp_path / "research"
    result = run_research(prepared, configuration(), output)
    assert accessed == ["train", "validation"]
    assert result["status"] == "development_complete"
    assert result["n_students"] == 2
    plan = read_json(output / "plan.json")
    assert plan["reserved_partitions_read"] == []
    records = read_json(output / "summary.json")["runs"]
    assert records[0]["statistics"]["initial_weights_sha256"] == records[1]["statistics"]["initial_weights_sha256"]
    assert records[0]["oof_targets_sha256"] == records[1]["oof_targets_sha256"]
    assert run_research(prepared, configuration(), output, resume=True) == result
    with pytest.raises(FileExistsError):
        run_research(prepared, configuration(), output)
    (output / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact mismatch"):
        verify_research(output)


def test_dry_run_does_not_load_any_partition(prepared, tmp_path, monkeypatch):
    import ids.research as research

    monkeypatch.setattr(research, "load_partition", lambda *args: pytest.fail("dry run loaded frame"))
    output = tmp_path / "dry"
    result = run_research(prepared, configuration(), output, dry_run=True)
    assert result["status"] == "plan_only" and not output.exists()


@pytest.mark.parametrize("change", [
    {"purpose": "final_test"}, {"seeds": [17, 17]}, {"max_fpr": True}, {"folds": 1},
    {"max_train_rows": -1}, {"unrecognized": 2},
])
def test_bad_configuration_rejected(change):
    config = configuration()
    config.update(change)
    with pytest.raises(ValueError):
        validate_config(config)


def test_validation_does_not_fit_preprocessor_or_change_initialization(prepared):
    from ids.data import load_partition
    train, val = load_partition(prepared, "train"), load_partition(prepared, "validation")
    config = configuration()["models"][0]
    first = train_research_mlp(train, val, config, 7, schema=None)
    changed = val.copy()
    changed["dur"] += 100000
    changed["proto"] = "validation-only-category"
    second = train_research_mlp(train, changed, config, 7, schema=None)
    np.testing.assert_array_equal(first.preprocessor.scaler.mean_, second.preprocessor.scaler.mean_)
    assert first.training_statistics["initial_weights_sha256"] == second.training_statistics["initial_weights_sha256"]
    assert "validation-only-category" not in second.preprocessor.encoder.categories_[0]
