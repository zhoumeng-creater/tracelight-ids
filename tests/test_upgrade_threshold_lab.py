"""No-fit audit orchestration and threshold-selection provenance contracts."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from ids.common import canonical_hash, sha256_file
from ids.risk_distillation import RiskConstraint, build_risk_reference

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("threshold_audit_lab", REPO / "scripts/audit_upgrade_thresholds_lab.py")
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def hardware():
    return {"recorded_at": "mock", "torch": {"cuda_available": True, "cuda_runtime": "12.6", "gpus": [
        {"index": 0, "name": "NVIDIA GeForce RTX 4090 D", "total_memory_bytes": 24 * 1024 ** 3}]}}


def sources(tmp_path, name):
    data, v3, v4 = (tmp_path / f"{name}-{role}" for role in ("data", "v3", "v4"))
    manifest = {"data_kind": "synthetic", "name": name}
    put(data / "manifest.json", manifest)
    cfg3 = {"purpose": "risk_v3_development_only", "cohort_policy": "complete_development", "seeds": [17],
            "max_fpr": .01, "optimization": {"attack_loss_limit": .01, "min_family_support": 30}}
    p3 = {"config": cfg3, "data": manifest, "reserved_partitions_read": []}
    p3["plan_id"] = canonical_hash(p3)
    put(v3 / "plan.json", p3)
    put(v3 / "RISK_RESEARCH_COMPLETE.json", {"status": "development_complete", "reserved_partitions_read": [],
        "plan_id": p3["plan_id"], "files": {"plan.json": sha256_file(v3 / "plan.json"), "anchors/s17.joblib": "f" * 64}})
    b4 = {"config": {"purpose": "upgrade_v4_development_only", "seeds": [17], "max_fpr": .01,
                       "optimization": {"attack_loss_limit": .01, "min_family_support": 30}}, "data": manifest,
          "v3_complete_sha256": sha256_file(v3 / "RISK_RESEARCH_COMPLETE.json"), "v3_plan_id": p3["plan_id"]}
    p4 = {"binding": b4, "binding_sha256": canonical_hash(b4), "reserved_partitions_read": []}
    put(v4 / "plan.json", p4)
    put(v4 / "UPGRADE_COMPLETE.json", {"status": "development_complete", "reserved_partitions_read": [],
        "binding_sha256": p4["binding_sha256"], "files": {"plan.json": sha256_file(v4 / "plan.json")}})
    return v3, v4, data


@pytest.fixture
def ready(tmp_path, monkeypatch):
    pairs = {name: sources(tmp_path, name) for name in ("unsw", "cicids")}
    monkeypatch.setattr(lab, "check_environment", hardware)
    monkeypatch.setattr(lab, "_activate_cuda_zero", lambda: None)
    monkeypatch.setattr(lab, "_dataset_record", lambda name, path: {"name": name})
    monkeypatch.setattr(lab, "code_fingerprint", lambda: {"source_sha256": "a" * 64, "files": {}})
    kwargs = {f"{name}_{role}_dir": paths[i] for name, paths in pairs.items()
              for i, role in enumerate(("v3", "v4", "data"))}
    return kwargs, pairs


def test_dry_run_does_not_load_models_parquet_or_full_payloads(ready, tmp_path, monkeypatch):
    args, _ = ready
    for name in ("load_bundle", "_cohorts", "_full_source_check", "_audit_environment", "solve_threshold_interval"):
        monkeypatch.setattr(lab, name, lambda *a, **k: pytest.fail("dry-run accessed training/model payload"))
    result = lab.run_lab(work_dir=tmp_path / "dry", dry_run=True, **args)
    assert result["status"] == "threshold_audit_plan_only" and result["fits"] == 0
    assert not (tmp_path / "dry/threshold-audit-unsw").exists()
    assert not (tmp_path / "dry/threshold-audit-cicids").exists()


def test_missing_cicids_is_rejected_before_environment_or_output(ready, tmp_path, monkeypatch):
    args, _ = ready
    args["cicids_v4_dir"] = None
    monkeypatch.setattr(lab, "check_environment", lambda: pytest.fail("partial input touched environment"))
    with pytest.raises(ValueError, match="BOTH"):
        lab.run_lab(work_dir=tmp_path / "no-cicids", **args)
    assert not (tmp_path / "no-cicids").exists()


def test_new_prepared_or_v3_identity_is_rejected(ready, tmp_path):
    args, pairs = ready
    v3, v4, data = pairs["unsw"]
    original = lab.read_json(data / "manifest.json")
    put(data / "manifest.json", {**original, "new": "different data"})
    with pytest.raises(ValueError, match="identity mismatch"):
        lab.run_lab(work_dir=tmp_path / "mismatch", dry_run=True, **args)
    assert not (tmp_path / "mismatch").exists()
    put(data / "manifest.json", original)
    with (v3 / "RISK_RESEARCH_COMPLETE.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="identity mismatch"):
        lab._metadata_record(v3, v4, data)


def test_real_4050_hardware_rejected_before_source_reads(ready, tmp_path, monkeypatch):
    args, _ = ready
    env = hardware()
    env["torch"]["gpus"][0]["name"] = "NVIDIA GeForce RTX 4050 Laptop GPU"
    monkeypatch.setattr(lab, "check_environment", lambda: env)
    monkeypatch.setattr(lab, "_metadata_record", lambda *a: pytest.fail("nonlab GPU read source"))
    with pytest.raises(ValueError, match="4090"):
        lab.run_lab(work_dir=tmp_path / "wrong-gpu", dry_run=True, **args)


def test_solver_receives_only_R_C_and_validation_scored_after_selection(monkeypatch):
    def frame(ids, y):
        return pd.DataFrame({"sample_id": ids, "label": y,
            "attack_cat": ["Normal" if value == 0 else "Attack" for value in y]})
    frames = {"reference": frame(["r0", "r1", "r2"], [0, 0, 1]), "constraint": frame(["c0", "c1", "c2"], [0, 1, 1]),
              "validation": frame(["v0", "v1", "v2", "v3"], [0, 1, 0, 1])}
    reference = build_risk_reference(torch.tensor([-1., 1., 1.]), torch.tensor([0, 1, 1]),
        [RiskConstraint("normal", "normal_fpr", 0.), RiskConstraint("attack", "attack_loss", 0.)], sample_ids=["c0", "c1", "c2"])
    scores = {"reference": np.array([-2., -1., 2.]), "constraint": np.array([.5, 1., 2.])}
    monkeypatch.setattr(lab, "_score_model", lambda *a: copy.deepcopy(scores))
    real = lab.solve_threshold_interval
    selected = []

    def solve(r, labels, c, ref, **kwargs):
        assert np.array_equal(r, scores["reference"]) and np.array_equal(c, scores["constraint"])
        assert kwargs["reference_ids"] == ("r0", "r1", "r2")
        assert kwargs["constraint_ids"] == ("c0", "c1", "c2")
        selected.append(True)
        return real(r, labels, c, ref, **kwargs)

    def validation_logits(bundle, values):
        assert selected and values.sample_id.tolist() == ["v0", "v1", "v2", "v3"]
        return np.array([.4, 1.4, .2, 1.8])

    monkeypatch.setattr(lab, "solve_threshold_interval", solve)
    monkeypatch.setattr(lab, "raw_cpu_logits", validation_logits)
    result, saved = lab.audit_model({"name": "known"}, object(), frames, reference, 0., .5, 1)
    exact = result["regimes"]["exact_R_C_interval"]
    assert exact["solution"]["feasible"]
    assert exact["evaluation"]["constraint"]["feasible"]
    assert exact["evaluation"]["validation"]["tpr"] == 1.
    assert exact["evaluation"]["validation"]["fpr"] == 0.
    assert not result["regimes"]["R_only"]["constraint"]["feasible"]
    assert result["fits"] == 0 and set(saved) == {"reference", "constraint", "validation"}


def test_catalog_does_not_invent_failed_phase1_checkpoint(tmp_path):
    v3, v4, _ = sources(tmp_path, "toy")
    c3 = lab.read_json(v3 / "RISK_RESEARCH_COMPLETE.json")
    c3["files"]["students/hard-s17/model.joblib"] = "e" * 64
    put(v3 / "RISK_RESEARCH_COMPLETE.json", c3)
    put(v4 / "phase1-s17.json", {"status": "no_feasible_start", "updates": 30})
    put(v4 / "summary.json", {"runs": [{"condition": "raw_kd", "seed": 17, "status": "no_feasible_start"}]})
    catalog, missing = lab.model_catalog(v3, v4, 17)
    assert {x["source_kind"] for x in catalog} == {"v3_anchor", "v3_hard"}
    assert missing["status"] == "missing_phase1_snapshot"
    c4 = lab.read_json(v4 / "UPGRADE_COMPLETE.json")
    c4["files"]["phase1-s17.joblib"] = "d" * 64
    put(v4 / "UPGRADE_COMPLETE.json", c4)
    put(v4 / "phase1-s17.json", {"status": "no_feasible_start", "updates": 30,
        "checkpoint_file": "phase1-s17.joblib", "checkpoint_sha256": "d" * 64})
    catalog, missing = lab.model_catalog(v3, v4, 17)
    assert missing is None and catalog[-1]["source_kind"] == "v4_phase1_checkpoint"


def test_saved_reference_retains_original_gpu_events_not_cpu_scores(tmp_path):
    # No real GPU use: explicit known margins represent the archived baseline.
    reference = build_risk_reference(torch.tensor([-2., .2, .3], dtype=torch.float64), torch.tensor([0, 1, 1]),
        [RiskConstraint("normal", "normal_fpr", .01), RiskConstraint("attack", "attack_loss", .01)], sample_ids=["c0", "c1", "c2"])
    (tmp_path / "constraint-identity.parquet").write_bytes(b"identity-test-fixture")
    lab._save_reference(tmp_path, 17, reference)
    with np.load(tmp_path / "frozen-reference-s17.npz") as saved:
        assert np.array_equal(saved["anchor_margins_C"], reference.anchor_margins.numpy())
        assert np.array_equal(saved["labels_C"], reference.labels.numpy())
        assert np.array_equal(saved["member_1"], reference.members[1].numpy())
    metadata = lab.read_json(tmp_path / "frozen-reference-s17.json")
    assert metadata["signature"] == reference.signature
    assert metadata["constraint_identity_sha256"] == sha256_file(tmp_path / "constraint-identity.parquet")


def test_complete_resume_packages_and_tamper_rejection(ready, tmp_path, monkeypatch):
    args, _ = ready
    called = []
    monkeypatch.setattr(lab, "_full_source_check", lambda record: None)

    def sealed_mock(name, record, output, binding, threads):
        # The orchestration/ZIP test uses explicit mock score payloads. Actual
        # R/C selection and V timing are covered by audit_model above.
        called.append(name)
        output.mkdir(parents=True)
        put(output / "plan.json", {"binding": binding, "binding_sha256": canonical_hash(binding), "input": record,
                                   "reserved_partitions_read": [], "environment_name": name})
        for role in ("reference", "constraint", "validation"):
            (output / f"{role}-identity.parquet").write_bytes(b"mock identity payload")
        for seed in record["seeds"]:
            np.savez_compressed(output / f"frozen-reference-s{seed}.npz", anchor_margins_C=np.array([-1., 1.]))
            put(output / f"frozen-reference-s{seed}.json", {"scope": "mock frozen reference"})
            put(output / f"family-support-s{seed}.json", {})
        rows = []
        for kind in ("v3_anchor", "v3_hard"):
            model = {"name": kind + "-s17", "source_kind": kind, "seed": 17}
            row = {"model": model, "fits": 0}
            directory = output / "models" / model["name"]
            directory.mkdir(parents=True)
            put(directory / "audit.json", row)
            np.savez_compressed(directory / "scores.npz", reference=np.array([1.]))
            rows.append(row)
        put(output / "summary.json", {"results": rows, "fits": 0})
        put(output / lab.COMPLETE, {"status": "threshold_audit_complete", "fits": 0, "n_models": 2,
            "reserved_partitions_read": [], "binding_sha256": canonical_hash(binding),
            "files": {p.relative_to(output).as_posix(): sha256_file(p) for p in output.rglob("*") if p.is_file()}})
        return lab.verify_audit(output)

    monkeypatch.setattr(lab, "_audit_environment", sealed_mock)
    work = tmp_path / "audit"
    result = lab.run_lab(work_dir=work, **args)
    assert result["status"] == "threshold_audit_complete" and result["fits"] == 0
    assert called == ["unsw", "cicids"]
    for name in ("UNSW", "CICIDS"):
        archive = work / "packages" / f"THRESHOLD_AUDIT_{name}.zip"
        assert archive.is_file()
        assert archive.with_name(archive.name + ".sha256").read_text().strip() == sha256_file(archive)
    assert lab.run_lab(work_dir=work, resume=True, **args)["status"] == "threshold_audit_complete"
    assert called == ["unsw", "cicids"]  # complete resume does not score again
    with pytest.raises(FileExistsError):
        lab.run_lab(work_dir=work, **args)
    (work / "threshold-audit-unsw/summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact mismatch"):
        lab.run_lab(work_dir=work, resume=True, **args)


def test_partial_output_preserved(ready, tmp_path, monkeypatch):
    args, _ = ready
    work = tmp_path / "partial"
    lab.run_lab(work_dir=work, dry_run=True, **args)
    partial = work / "threshold-audit-unsw"
    partial.mkdir()
    marker = partial / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(lab, "_audit_environment", lambda *a: pytest.fail("partial output rescored"))
    with pytest.raises(FileExistsError, match="incomplete"):
        lab.run_lab(work_dir=work, resume=True, **args)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_actual_synthetic_artifacts_audit_and_package_without_fits_or_reserved_reads(tmp_path, monkeypatch):
    """Use real synthetic model bundles/solver/ZIP, not mock audit payloads."""
    import ids.models as models
    import ids.upgrade_research as upgrade

    spec = importlib.util.spec_from_file_location("upgrade_fixture_source", REPO / "tests/test_upgrade_research.py")
    fixture_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture_module)
    data, v3 = fixture_module.source.__wrapped__(tmp_path, monkeypatch)
    cfg = fixture_module.config()
    cfg.update(max_fpr=.25, conditions=["hard"])
    cfg["optimization"].update(attack_loss_limit=1., macro_steps=1)
    v4 = tmp_path / "actual-v4"
    upgrade.run_upgrade_research(v3, data, cfg, v4, device="cpu")
    assert not (data / "test.parquet").exists() and not (data / "calibration.parquet").exists()
    record = lab._metadata_record(v3, v4, data)
    binding = {"source": lab.code_fingerprint(), "purpose": "synthetic_no_fit_audit", "threads": 1}
    loaded = []
    original = upgrade.load_partition

    def track(path, split):
        assert split in {"train", "validation"}
        loaded.append(split)
        return original(path, split)

    monkeypatch.setattr(upgrade, "load_partition", track)
    monkeypatch.setattr(models, "train_model", lambda *a, **k: pytest.fail("audit refitted a teacher/model"))
    output = tmp_path / "actual-threshold-audit"
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        done = lab._audit_environment("synthetic", record, output, binding, threads=1)
    finally:
        torch.set_num_threads(previous)
    assert loaded == ["train", "validation"]
    assert done["fits"] == 0 and done["n_models"] == 4
    summary = lab.read_json(output / "summary.json")
    assert summary["missing_phase1_snapshots"] == []
    kinds = {r["model"]["source_kind"] for r in summary["results"]}
    assert kinds == {"v3_anchor", "v3_hard", "v4_selected", "v4_phase1_checkpoint"}
    for row in summary["results"]:
        assert set(row["regimes"]) == {"R_only", "frozen_anchor", "exact_R_C_interval"}
        assert row["threshold_selection_roles"] == ["reference", "constraint"]
        scores = output / "models" / row["model"]["name"] / "scores.npz"
        with np.load(scores) as values:
            for role in ("reference", "constraint", "validation"):
                identity = pd.read_parquet(output / f"{role}-identity.parquet")
                assert values[role].shape == (len(identity),)
    expected = lab.read_json(v3 / "students/hard-s17/run.json")["reference_signature"]
    assert lab.read_json(output / "frozen-reference-s17.json")["signature"] == expected
    assert lab.verify_audit(output) == done
    archive = tmp_path / "packages/actual-synthetic-audit.zip"
    packaged = lab.package_audit(output, archive)
    assert packaged["sha256"] == sha256_file(archive)
    assert lab.package_audit(output, archive, resume=True)["status"] == "verified_existing"


def test_help_works_outside_repository(tmp_path):
    result = subprocess.run([sys.executable, str(REPO / "scripts/audit_upgrade_thresholds_lab.py"), "--help"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0 and "--cicids-v4-dir" in result.stdout
