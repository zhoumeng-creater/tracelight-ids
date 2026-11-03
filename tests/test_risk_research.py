"""Synthetic integration: real runner, all losses, no historical partitions."""
from pathlib import Path

import pytest
import yaml

from ids.common import read_json
from ids.data import make_synthetic_dataset, prepare_dataset
from ids.risk_research import run_risk_research, validate_risk_config, verify_risk_research


def configuration():
    return yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/risk-v3-smoke.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def prepared(tmp_path):
    raw, data = tmp_path / "raw", tmp_path / "prepared"
    make_synthetic_dataset(raw, n_train=650, n_test=100)
    prepare_dataset(raw / "UNSW_NB15_training-set.csv", raw / "UNSW_NB15_testing-set.csv", data, data_kind="synthetic")
    return data


def test_all_conditions_oof_roles_reserved_trap_and_sealed_resume(prepared, tmp_path, monkeypatch):
    import ids.risk_research as research

    original = research.load_partition
    accessed = []

    def tracked(path, split):
        accessed.append(split)
        assert split in {"train", "validation"}
        return original(path, split)

    monkeypatch.setattr(research, "load_partition", tracked)
    (prepared / "test.parquet").unlink()
    (prepared / "calibration.parquet").unlink()
    root = tmp_path / "v3"
    config = configuration()
    result = run_risk_research(prepared, config, root, device="cpu")
    assert result["n_students"] == 9 and result["n_teacher_fits"] == 4 and result["n_anchor_fits"] == 1
    assert accessed == ["train", "validation"]
    assert result["reserved_partitions_read"] == []
    plan = read_json(root / "plan.json")
    assert plan["reserved_partitions_read"] == []
    records = read_json(root / "summary.json")["runs"]
    assert len({r["common_initial_state_sha256"] for r in records}) == 1
    assert len({str(r["hard_labels"]) for r in records}) == 1
    assert len({r["positive_weight_F_plus_C"] for r in records}) == 1
    assert len({r["working_point"]["threshold"] for r in records}) == 1
    assert {r["macro_steps"] for r in records} == {config["optimization"]["macro_steps"]}
    for record in records:
        run = root / "students" / f"{record['condition']}-s17"
        history = read_json(run / "history.json")
        if record["condition"].startswith("risk_") or record["condition"] == "normal_hinge_kd":
            if record["selected_step"] is not None:
                assert history[record["selected_step"] - 1]["surrogate_feasible"]
        if record["status"] == "diagnostic_no_eligible_checkpoint":
            assert record["selected_step"] is None or not record["surrogate_feasible_cpu"] or record["validation_cpu_raw_logit"]["fpr"] > .01
        dual = read_json(run / "dual.json")
        assert set(dual) == {"selected_model", "final_trajectory_diagnostic"}
    assert run_risk_research(prepared, config, root, device="cpu", resume=True) == result
    assert accessed == ["train", "validation"]  # completed resume needs no frame reads
    with pytest.raises(FileExistsError):
        run_risk_research(prepared, config, root, device="cpu")
    changed = configuration()
    changed["optimization"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="identity differs"):
        run_risk_research(prepared, changed, root, device="cpu", resume=True)
    # Additional files and changed payloads are independently detected.
    extra = root / "untracked.txt"
    extra.write_text("added", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        verify_risk_research(root)
    extra.unlink()
    (root / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact mismatch"):
        verify_risk_research(root)


@pytest.mark.parametrize("change", [
    {"purpose": "final_test"}, {"seeds": [17, 17]}, {"max_fpr": True}, {"folds": 1},
    {"max_train_rows": 100}, {"conditions": ["fake"]}, {"unrecognized": 1},
    {"reference_fraction": .8, "constraint_fraction": .3},
])
def test_bad_configuration_rejected(change):
    config = configuration()
    config.update(change)
    with pytest.raises(ValueError):
        validate_risk_config(config)


def test_real_cpu_training_rejected_before_loading(prepared, tmp_path, monkeypatch):
    import ids.risk_research as research
    original = research.verify_development_inputs

    def real_manifest(path):
        result = original(path)
        result["data_kind"] = "real"
        return result

    monkeypatch.setattr(research, "verify_development_inputs", real_manifest)
    monkeypatch.setattr(research, "load_partition", lambda *a: pytest.fail("real local fit was attempted"))
    output = tmp_path / "real"
    with pytest.raises(ValueError, match="RTX4090D"):
        run_risk_research(prepared, configuration(), output, device="cpu")
    assert not output.exists()


def test_partial_output_is_preserved(prepared, tmp_path):
    output = tmp_path / "partial"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("partial", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        run_risk_research(prepared, configuration(), output, device="cpu", resume=True)
    assert marker.read_text(encoding="utf-8") == "partial"


def test_source_change_during_training_prevents_completion(prepared, tmp_path, monkeypatch):
    import ids.risk_research as research
    original = research.code_fingerprint
    calls = []

    def changed_source():
        value = original()
        calls.append(1)
        if len(calls) > 1:
            value["source_sha256"] = "f" * 64
        return value

    monkeypatch.setattr(research, "code_fingerprint", changed_source)
    config = configuration()
    config["conditions"] = ["hard"]
    output = tmp_path / "changed-source"
    with pytest.raises(ValueError, match="source changed"):
        run_risk_research(prepared, config, output, device="cpu")
    assert not (output / "RISK_RESEARCH_COMPLETE.json").exists()
