"""Synthetic v2 provenance, representation, three-arm and cold-deployment checks."""

import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ids.common import sha256_file, write_json
from ids.context_cache.calibration import compare_decisions, register_threshold
from ids.context_cache.csv_workflow import (
    PROTOCOL_V2, V2_POLICIES, load_protocol, preflight_protocol, register_protocol,
    run_comparison, run_input_comparison,
)


IDENTIFIERS = ["L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO"]
SCHEMA = {"features": IDENTIFIERS + ["signal", "duration"], "categorical": IDENTIFIERS,
          "label_column": "label", "id_column": "sample_id", "group_column": "session",
          "family_column": "family"}


@pytest.fixture
def v2_protocol(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    frames, files = {}, {}
    for role, count in (("support", 32), ("calibration", 99), ("evaluation", 24)):
        rng = np.random.default_rng(847 + count)
        truth = np.zeros(count, dtype=int) if role == "calibration" else np.arange(count) % 2
        frames[role] = pd.DataFrame({
            "L4_SRC_PORT": [str(100 + i % 4 if role == "support" else 1000 + i) for i in range(count)],
            "L4_DST_PORT": ["80" if i % 2 else "443" for i in range(count)],
            "PROTOCOL": ["6" if i % 2 else "17" for i in range(count)],
            "L7_PROTO": ["1" if i % 2 else "2" for i in range(count)],
            "signal": rng.normal(size=count), "duration": rng.lognormal(size=count),
            "label": truth, "sample_id": [f"{role}-{i}" for i in range(count)],
            "session": [f"{role}-group-{i // 3}" for i in range(count)],
            "family": ["Benign" if y == 0 else "synthetic_attack" for y in truth],
        })
    # Distinct communication identities may have identical measured input.
    frames["evaluation"].loc[0, SCHEMA["features"]] = frames["calibration"].loc[0, SCHEMA["features"]]
    # Two unseen ports must remain separate in the protected profile.
    frames["evaluation"].loc[1, SCHEMA["features"]] = frames["evaluation"].loc[0, SCHEMA["features"]]
    frames["evaluation"].loc[1, "L4_SRC_PORT"] = "60000"
    for role, frame in frames.items():
        path = prepared / f"{role}.csv"
        frame.to_csv(path, index=False)
        files[role] = {"path": str(path), "sha256": sha256_file(path), "rows": len(frame)}
    receipt = {
        "format": "tracelight-nf-ton-v3-preparation-v2", **V2_POLICIES,
        "data_kind": "synthetic", "schema": SCHEMA, "seed": 847,
        "files": copy.deepcopy(files), "source": {"official_release_verified": False},
        "budgets": {"support_per_class": 16, "calibration_rows": 99, "evaluation_rows": 24,
                    "slots": None, "alpha": .01},
    }
    receipt["scans"] = {role: {"source_rows_scanned": 10000, "identifier_validation": {
        "rows_validated": 10000, "fields": {name: {"minimum": 0, "maximum": maximum,
                                                  "accepted_minimum": 0, "accepted_maximum": maximum}
                                               for name, maximum in zip(IDENTIFIERS, [65535, 65535, 255, 65535])}}}
                        for role in files}
    preparation = prepared / "preparation.json"
    write_json(preparation, receipt)
    checkpoint = tmp_path / "random_tiny_NOT_PRETRAINED.ckpt"
    checkpoint.write_bytes(b"synthetic-placeholder-not-a-model")
    spec = {"protocol": PROTOCOL_V2, "data_kind": "synthetic", **V2_POLICIES,
            "preparation": {"path": str(preparation), "sha256": sha256_file(preparation)},
            "schema": SCHEMA, "files": files,
            "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
            "slots": 4, "alpha": .01, "build_device": "cpu", "query_batch_size": 17,
            "query_chunk_size": 7, "threads": 1, "seed": 847}
    manifest = tmp_path / "protocol.json"
    write_json(manifest, spec)
    return manifest, spec, frames, receipt, preparation, checkpoint


@pytest.fixture
def official(monkeypatch):
    torch = pytest.importorskip("torch")
    source = Path(__file__).resolve().parents[1] / "local" / "vendor" / "tabicl" / "src"
    if not (source / "tabicl" / "_model" / "tabicl.py").is_file():
        pytest.skip("pinned official source absent; no download")
    monkeypatch.syspath_prepend(str(source))
    from ids.context_cache.upstream import require_pinned_tabicl

    require_pinned_tabicl()
    return torch


def test_v2_reports_cross_role_inputs_and_binds_preparation_without_model_loading(v2_protocol, tmp_path):
    manifest, spec, _, _, preparation, checkpoint = v2_protocol
    output = tmp_path / "registered.json"
    result = register_protocol(
        manifest, **{f"{role}_csv": item["path"] for role, item in spec["files"].items()},
        checkpoint=checkpoint, checkpoint_sha256=sha256_file(checkpoint), output=output)
    loaded, _, _, _, audit = preflight_protocol(output, audit_representations=False)
    assert loaded["preparation"]["sha256"] == sha256_file(preparation)
    assert result["audit"] == audit
    assert audit["cross_role_pairs"]["calibration/evaluation"]["exact_feature_overlap"] == 1
    assert all(pair["explicit_id_overlap"] == pair["declared_group_overlap"] == 0
               for pair in audit["cross_role_pairs"].values())


@pytest.mark.parametrize("field,value", [("sampling_profile", "legacy_sequential_v1"),
                                        ("risk_unit", "group"), ("exact_feature_policy", "reject")])
def test_v2_rejects_implicit_legacy_policies(v2_protocol, field, value):
    manifest, spec, *_ = v2_protocol
    spec[field] = value
    write_json(manifest, spec)
    with pytest.raises(ValueError, match="v2 requires"):
        load_protocol(manifest)


@pytest.mark.parametrize("kind", ["bytes", "legacy_receipt", "csv_binding", "seed", "official_claim",
                                 "missing_domain_validation", "partial_scan", "wrong_l7_domain", "outside_domain"])
def test_v2_rejects_changed_or_mislabeled_preparation(v2_protocol, kind):
    manifest, spec, _, receipt, preparation, _ = v2_protocol
    if kind == "bytes":
        preparation.write_text(preparation.read_text("utf-8") + "\n", encoding="utf-8")
    else:
        if kind == "legacy_receipt":
            receipt["format"] = "tracelight-nf-ton-v3-preparation-v1"
        elif kind == "csv_binding":
            receipt["files"]["evaluation"]["sha256"] = "f" * 64
        elif kind == "seed":
            receipt["seed"] += 1
        elif kind == "official_claim":
            receipt["source"]["official_release_verified"] = True
        elif kind == "missing_domain_validation":
            receipt["scans"]["support"].pop("identifier_validation")
        elif kind == "partial_scan":
            receipt["scans"]["support"]["identifier_validation"]["rows_validated"] -= 1
        elif kind == "wrong_l7_domain":
            receipt["scans"]["support"]["identifier_validation"]["fields"]["L7_PROTO"]["accepted_maximum"] = 255
        else:
            receipt["scans"]["support"]["identifier_validation"]["fields"]["L7_PROTO"]["maximum"] = 65536
        write_json(preparation, receipt)
        spec["preparation"]["sha256"] = sha256_file(preparation)
        write_json(manifest, spec)
    with pytest.raises(ValueError, match="preparation"):
        preflight_protocol(manifest, audit_representations=False)


@pytest.mark.parametrize("column,kind", [("sample_id", "explicit_id_overlap"),
                                       ("session", "declared_group_overlap")])
def test_report_policy_still_rejects_row_or_group_leakage(v2_protocol, column, kind):
    manifest, spec, frames, receipt, preparation, _ = v2_protocol
    frames["evaluation"].loc[0, column] = frames["support"].loc[0, column]
    path = Path(spec["files"]["evaluation"]["path"])
    frames["evaluation"].to_csv(path, index=False)
    spec["files"]["evaluation"]["sha256"] = sha256_file(path)
    receipt["files"] = copy.deepcopy(spec["files"])
    write_json(preparation, receipt)
    spec["preparation"]["sha256"] = sha256_file(preparation)
    write_json(manifest, spec)
    with pytest.raises(ValueError, match=kind):
        preflight_protocol(manifest, audit_representations=False)


def test_v2_development_budget_and_schema_are_fixed_before_io(v2_protocol):
    from ids.context_cache.lab_data import SCHEMA as NF_SCHEMA

    manifest, spec, *_ = v2_protocol
    spec.update(data_kind="development", schema=NF_SCHEMA, seed=20261107, slots=512,
                build_device="cuda", query_batch_size=128, query_chunk_size=256)
    for role, count in (("support", 8192), ("calibration", 5000), ("evaluation", 20000)):
        spec["files"][role]["rows"] = count
    write_json(manifest, spec)
    load_protocol(manifest)
    spec["slots"] = 256
    write_json(manifest, spec)
    with pytest.raises(ValueError, match="frozen NF-v3"):
        load_protocol(manifest)


def test_preflight_reports_actual_final_inputs_of_both_profiles(v2_protocol, official):
    manifest, *_ = v2_protocol
    _, _, _, _, audit = preflight_protocol(manifest)
    reps = audit["representations"]
    assert set(reps) == {"legacy", "new"}
    old = reps["legacy"]["roles"]["evaluation"]["final_fp32"]
    new = reps["new"]["roles"]["evaluation"]["final_fp32"]
    assert old["mixed_label_patterns"] > new["mixed_label_patterns"]
    assert old["unique_input_patterns"] < new["unique_input_patterns"]


def test_input_profile_comparison_relaxes_only_preprocessing_binding():
    common = {"model_sha256": "a" * 64, "calibration_sha256": "c" * 64}
    left = register_threshold([0., 1., 2., 3.], .2, preprocessing_sha256="1" * 64, **common)
    right = register_threshold([0., 2., 4., 6.], .2, preprocessing_sha256="2" * 64, **common)
    args = ([0, 1], [0., 4.], left, [0., 7.], right)
    with pytest.raises(ValueError, match="preprocessing_sha256"):
        compare_decisions(*args)
    result = compare_decisions(*args, require_same_preprocessing=False)
    assert result["preprocessing_comparison"] == {"different_preprocessing_authorized": True,
                                                 "same_preprocessing": False}
    assert result["full_binding"] == left["binding"] and result["qcoc_binding"] == right["binding"]
    wrong_cohort = copy.deepcopy(right)
    wrong_cohort["binding"]["calibration_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="calibration_sha256"):
        compare_decisions([0, 1], [0., 4.], left, [0., 7.], wrong_cohort, require_same_preprocessing=False)


def test_three_fixed_random_model_arms_export_independent_cold_detectors(v2_protocol, official, tmp_path):
    torch = official
    from tabicl._model.tabicl import TabICL

    manifest, spec, frames, _, _, checkpoint = v2_protocol
    config = {"max_classes": 10, "embed_dim": 8, "col_num_blocks": 1, "col_nhead": 2,
              "col_num_inds": 4, "row_num_blocks": 1, "row_nhead": 2, "row_num_cls": 2,
              "icl_num_blocks": 2, "icl_nhead": 2, "ff_factor": 2, "dropout": 0., "zero_init": False}
    with torch.random.fork_rng():
        torch.manual_seed(847)
        model = TabICL(**config).eval()
    torch.save({"config": config, "state_dict": model.state_dict()}, checkpoint)
    spec["checkpoint"]["sha256"] = sha256_file(checkpoint)
    write_json(manifest, spec)
    output = tmp_path / "comparison"
    old_threads = torch.get_num_threads()
    try:
        with pytest.raises(ValueError, match="requires tracelight-context-csv-comparison-v1"):
            run_comparison(manifest, output)
        assert not output.exists()
        report = run_input_comparison(manifest, output)
        assert report["candidate_count"] == 3
        names = ("legacy_full", "full", "qcoc")
        assert set(report["evaluation"]) == set(names)
        assert set(report["paired"]) == {"legacy_full_vs_full", "full_vs_qcoc"}
        assert report["paired"]["legacy_full_vs_full"]["reference_name"] == "legacy_full"
        assert report["paired"]["legacy_full_vs_full"]["candidate_name"] == "full"
        assert report["paired"]["legacy_full_vs_full"]["preprocessing_comparison"]["different_preprocessing_authorized"]
        assert "preprocessing_comparison" not in report["paired"]["full_vs_qcoc"]
        assert report["arm_profiles"]["full"] == report["arm_profiles"]["qcoc"]
        scores = pd.read_csv(output / "evaluation_scores.csv", float_precision="round_trip")
        for left, right in (("legacy_full", "full"), ("full", "qcoc")):
            paired = pd.read_csv(output / f"paired_{left}_{right}.csv", float_precision="round_trip")
            for name in (left, right):
                np.testing.assert_array_equal(paired[f"{name}_score"], scores[f"{name}_score"])
        prep_hashes = {name: sha256_file(output / name / "preprocessor.joblib") for name in names}
        assert prep_hashes["full"] == prep_hashes["qcoc"] != prep_hashes["legacy_full"]
        feature_only = tmp_path / "queries.csv"
        frames["evaluation"].loc[:, SCHEMA["features"]].to_csv(feature_only, index=False)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(str(Path(p).resolve()) for p in sys.path if p and Path(p).is_dir())
        for name in names:
            registration = report["registrations"][name]
            calibration = pd.read_csv(output / f"{name}_calibration.csv", float_precision="round_trip")
            assert registration["threshold"] == np.sort(calibration["score"].to_numpy())[98]
            assert registration["binding"]["preprocessing_sha256"] == prep_hashes[name]
            cold = tmp_path / "cold" / name
            shutil.copytree(output / name, cold)
            assert {p.name for p in cold.iterdir()} == {"deployment.json", "threshold.json", "model.pt",
                                                      "preprocessor.joblib"}
            prediction_path = cold / "predictions.csv"
            command = [sys.executable, "-m", "ids.context_cache", "predict-csv", "--deployment", str(cold),
                       "--input", str(feature_only), "--output", str(prediction_path)]
            result = subprocess.run(command, cwd=cold, env=env, capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, result.stderr[-3000:]
            predicted = pd.read_csv(prediction_path, float_precision="round_trip")
            np.testing.assert_array_equal(predicted["score"], scores[f"{name}_score"])
            np.testing.assert_array_equal(predicted["predicted_attack"].astype(bool), scores[f"{name}_predicted"])
    finally:
        torch.set_num_threads(old_threads)
