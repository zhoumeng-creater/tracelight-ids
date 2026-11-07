"""Synthetic CSV role audit and two-candidate deployment round trip."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ids.common import sha256_file, write_json
from ids.context_cache.csv_workflow import (
    CSVDetector, PROTOCOL, load_protocol, preflight_protocol, register_protocol, run_comparison,
)


SCHEMA = {
    "features": ["signal", "duration", "protocol"],
    "numeric": ["signal", "duration"], "categorical": ["protocol"],
    "label_column": "label", "id_column": "sample_id", "group_column": "session",
    "family_column": "family",
}


@pytest.fixture
def csv_protocol(tmp_path):
    frames, paths = {}, {}
    for index, (role, n) in enumerate((("support", 32), ("calibration", 199), ("evaluation", 80))):
        generator = np.random.default_rng(7601 + index)
        truth = np.zeros(n, dtype=int) if role == "calibration" else np.arange(n) % 2
        frames[role] = pd.DataFrame({
            "signal": generator.normal(size=n), "duration": generator.lognormal(size=n),
            "protocol": ["tcp" if i % 3 else "udp" for i in range(n)],
            "label": truth, "sample_id": [f"{role}-row-{i:04d}" for i in range(n)],
            "session": [f"{role}-session-{i // 4}" for i in range(n)],
            "family": ["normal" if not y else f"attack-{i % 4}" for i, y in enumerate(truth)],
        })
        paths[role] = tmp_path / f"{role}.csv"
        frames[role].to_csv(paths[role], index=False)
    checkpoint = tmp_path / "random_tiny_NOT_PRETRAINED.ckpt"
    # Preflight must hash, but must not deserialize or execute, a checkpoint.
    checkpoint.write_bytes(b"synthetic-placeholder-not-a-model")
    spec = {
        "protocol": PROTOCOL, "data_kind": "synthetic", "schema": SCHEMA,
        "files": {role: {"path": path.name, "sha256": sha256_file(path).upper(), "rows": len(frames[role])}
                  for role, path in paths.items()},
        "checkpoint": {"path": checkpoint.name, "sha256": sha256_file(checkpoint).upper()},
        "slots": 4, "alpha": .01, "build_device": "cpu", "query_batch_size": 31,
        "query_chunk_size": 7, "threads": 1, "seed": 7601,
    }
    manifest = tmp_path / "protocol.json"
    write_json(manifest, spec)
    return manifest, spec, frames, paths, checkpoint


def rewrite_role(protocol, role):
    manifest, spec, frames, paths, _ = protocol
    frames[role].to_csv(paths[role], index=False)
    spec["files"][role]["sha256"] = sha256_file(paths[role]).upper()
    write_json(manifest, spec)


def test_registration_fills_paths_hashes_counts_without_model_loading(csv_protocol, tmp_path):
    manifest, _, _, paths, checkpoint = csv_protocol
    output = tmp_path / "lab" / "registered.json"
    result = register_protocol(manifest, support_csv=paths["support"], calibration_csv=paths["calibration"],
                               evaluation_csv=paths["evaluation"], checkpoint=checkpoint,
                               checkpoint_sha256=sha256_file(checkpoint).upper(), output=output)
    spec, _, resolved, _, audit = preflight_protocol(output)
    assert result["protocol_sha256"] == sha256_file(output)
    assert result["audit"] == audit
    for role, path in paths.items():
        assert resolved[role] == path.resolve()
        assert spec["files"][role]["sha256"] == sha256_file(path)
    with pytest.raises(FileExistsError):
        register_protocol(manifest, support_csv=paths["support"], calibration_csv=paths["calibration"],
                          evaluation_csv=paths["evaluation"], checkpoint=checkpoint,
                          checkpoint_sha256=sha256_file(checkpoint), output=output)


def test_registration_rejects_wrong_checkpoint_and_leaking_roles_before_writing(csv_protocol, tmp_path):
    manifest, _, _, paths, checkpoint = csv_protocol
    output = tmp_path / "registered.json"
    kwargs = dict(template_path=manifest, support_csv=paths["support"], calibration_csv=paths["calibration"],
                  evaluation_csv=paths["evaluation"], checkpoint=checkpoint, output=output)
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        register_protocol(**kwargs, checkpoint_sha256="f" * 64)
    assert not output.exists()
    kwargs["evaluation_csv"] = paths["support"]
    with pytest.raises(ValueError, match="separate CSV"):
        register_protocol(**kwargs, checkpoint_sha256=sha256_file(checkpoint))
    assert not output.exists()


def test_preflight_checks_three_roles_from_utf8_bom_manifest_without_loading_checkpoint(csv_protocol):
    manifest, spec, _, paths, _ = csv_protocol
    manifest.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8-sig")
    assert manifest.read_bytes().startswith(b"\xef\xbb\xbf")
    loaded, _, resolved, frames, audit = preflight_protocol(manifest)
    assert loaded["slots"] == 4
    assert {role: len(frame) for role, frame in frames.items()} == {
        "support": 32, "calibration": 199, "evaluation": 80}
    assert all(resolved[role] == path.resolve() for role, path in paths.items())
    assert all(not any(overlap.values()) for overlap in audit["cross_role_pairs"].values())
    assert audit["session_groups"] == "checked_declared_group"
    assert audit["roles"]["calibration"]["normal"] == spec["files"]["calibration"]["rows"]


def test_development_finite_threshold_row_boundary_is_checked_before_data_io(csv_protocol):
    manifest, spec, _, _, _ = csv_protocol
    spec["data_kind"] = "development"
    spec["alpha"] = .01
    for role in ("support", "calibration", "evaluation"):
        spec["files"][role]["path"] = f"not-present-{role}.csv"
    spec["checkpoint"]["path"] = "not-present-checkpoint.ckpt"
    spec["files"]["calibration"]["rows"] = 98
    write_json(manifest, spec)
    with pytest.raises(ValueError, match="insufficient for a finite threshold"):
        load_protocol(manifest)

    # ceil(100 * .99) == 99: the minimum finite rank passes configuration
    # validation even though none of the declared data/model files exists.
    spec["files"]["calibration"]["rows"] = 99
    write_json(manifest, spec)
    loaded, _, resolved = load_protocol(manifest)
    assert loaded["files"]["calibration"]["rows"] == 99
    assert all(not path.exists() for path in resolved.values())


@pytest.mark.parametrize("role", ["support", "calibration", "evaluation"])
def test_preflight_rejects_modified_csv_sha(csv_protocol, role):
    manifest, _, _, paths, _ = csv_protocol
    with paths[role].open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match=f"SHA-256 mismatch for {role}"):
        preflight_protocol(manifest)


@pytest.mark.parametrize("role", ["support", "calibration", "evaluation"])
def test_preflight_checks_registered_row_counts(csv_protocol, role):
    manifest, spec, _, _, _ = csv_protocol
    spec["files"][role]["rows"] += 1
    write_json(manifest, spec)
    with pytest.raises(ValueError, match=f"row count differs for {role}"):
        preflight_protocol(manifest)


@pytest.mark.parametrize("destination", ["calibration", "evaluation"])
def test_full_feature_overlap_cannot_be_hidden_by_changed_id_label_or_family(csv_protocol, destination):
    manifest, _, frames, _, _ = csv_protocol
    # Support row 1 is an attack; destination row 0 remains normal. Metadata
    # and labels deliberately differ while every model feature is identical.
    frames[destination].loc[0, SCHEMA["features"]] = frames["support"].loc[1, SCHEMA["features"]]
    assert frames[destination].loc[0, "label"] != frames["support"].loc[1, "label"]
    assert frames[destination].loc[0, "sample_id"] != frames["support"].loc[1, "sample_id"]
    rewrite_role(csv_protocol, destination)
    with pytest.raises(ValueError, match="exact_feature_overlap.*1"):
        preflight_protocol(manifest)


@pytest.mark.parametrize("column,kind", [("sample_id", "explicit_id_overlap"),
                                       ("session", "declared_group_overlap")])
def test_explicit_id_or_group_overlap_rejected_even_with_distinct_features(csv_protocol, column, kind):
    manifest, _, frames, _, _ = csv_protocol
    frames["evaluation"].loc[0, column] = frames["support"].loc[0, column]
    rewrite_role(csv_protocol, "evaluation")
    with pytest.raises(ValueError, match=f"{kind}.*1"):
        preflight_protocol(manifest)


def test_calibration_attack_row_is_rejected_not_filtered(csv_protocol):
    manifest, _, frames, _, _ = csv_protocol
    frames["calibration"].loc[0, "label"] = 1
    rewrite_role(csv_protocol, "calibration")
    with pytest.raises(ValueError, match="normal rows only"):
        preflight_protocol(manifest)


@pytest.mark.parametrize("change,expected", [({"data_kind": "final_test"}, "not final test"),
                                           ({"slots": 33}, "slots exceed"),
                                           ({"slots": [2, 4]}, "positive integer")])
def test_manifest_rejects_final_test_and_non_single_or_excess_budget(csv_protocol, change, expected):
    manifest, spec, _, _, _ = csv_protocol
    spec.update(change)
    write_json(manifest, spec)
    with pytest.raises(ValueError, match=expected):
        preflight_protocol(manifest)


def test_random_tiny_comparison_exports_two_bound_raw_csv_detectors(csv_protocol, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    checkout = Path(__file__).resolve().parents[1] / "local" / "vendor" / "tabicl"
    if not (checkout / "src" / "tabicl" / "_model" / "tabicl.py").is_file():
        pytest.skip("Pinned optional official source absent; no source or checkpoint download")
    monkeypatch.syspath_prepend(str(checkout / "src"))
    from ids.context_cache.upstream import require_pinned_tabicl

    require_pinned_tabicl()
    from tabicl._model.tabicl import TabICL

    manifest, spec, frames, paths, checkpoint = csv_protocol
    config = {
        "max_classes": 10, "embed_dim": 8, "col_num_blocks": 1, "col_nhead": 2,
        "col_num_inds": 4, "row_num_blocks": 1, "row_nhead": 2, "row_num_cls": 2,
        "icl_num_blocks": 2, "icl_nhead": 2, "ff_factor": 2, "dropout": 0.0,
        "zero_init": False, "col_feature_group": "same", "col_feature_group_size": 3,
        "col_target_aware": True, "col_ssmax": "qassmax-mlp-elementwise",
        "icl_ssmax": "qassmax-mlp-elementwise",
    }
    with torch.random.fork_rng():
        torch.manual_seed(7601)
        model = TabICL(**config).eval()
    torch.save({"config": config, "state_dict": model.state_dict()}, checkpoint)
    spec["checkpoint"]["sha256"] = sha256_file(checkpoint).upper()
    assert all(item["sha256"] == item["sha256"].upper() for item in spec["files"].values())
    write_json(manifest, spec)
    output = tmp_path / "comparison"
    old_threads = torch.get_num_threads()
    try:
        report = run_comparison(manifest, output)
        assert report["scope"] == "development_comparison_not_final_confirmation"
        assert report["candidate_count"] == 2 and report["slots"] == 4
        assert report["support_rows"] == 32
        assert set(report["evaluation"]) == {"full", "qcoc"}
        paired = pd.read_csv(output / "paired_evaluation.csv", float_precision="round_trip")
        prep_hash = sha256_file(output / "preprocessor.joblib")
        model_hashes = set()
        for name in ("full", "qcoc"):
            directory = output / name
            detector = CSVDetector.load(directory)
            registration = report["registrations"][name]
            model_hash = sha256_file(directory / "model.pt")
            model_hashes.add(model_hash)
            assert registration["binding"] == {
                "model_sha256": model_hash, "preprocessing_sha256": prep_hash,
                "calibration_sha256": sha256_file(paths["calibration"]),
            }
            own_calibration = pd.read_csv(output / f"{name}_calibration.csv", float_precision="round_trip")
            assert registration["n_normal"] == 199 and registration["rank"] == 198
            assert registration["threshold"] == np.sort(own_calibration["score"].to_numpy())[197]
            assert sha256_file(directory / "preprocessor.joblib") == prep_hash

            # A freshly loaded detector consumes the original raw evaluation
            # CSV. Removing labels/family must not change model inputs.
            unlabeled = tmp_path / f"{name}_raw_unlabeled.csv"
            frames["evaluation"].drop(columns=["label", "family"]).to_csv(unlabeled, index=False)
            prediction_path = tmp_path / f"{name}_reloaded_predictions.csv"
            result = detector.predict_csv(unlabeled, prediction_path)
            assert result["rows"] == 80
            predictions = pd.read_csv(prediction_path, float_precision="round_trip")
            assert predictions["row_id"].tolist() == paired["row_id"].tolist()
            np.testing.assert_array_equal(predictions["score"], paired[f"{name}_score"])
            np.testing.assert_array_equal(predictions["predicted_attack"].astype(bool),
                                          paired[f"{name}_predicted"])
        assert len(model_hashes) == 2

        # Updating a threshold-file hash does not authorize using the other
        # model's registered threshold: the internal model binding still fails.
        bad_threshold = json.loads((output / "full" / "threshold.json").read_text(encoding="utf-8"))
        write_json(output / "qcoc" / "threshold.json", bad_threshold)
        deployment_path = output / "qcoc" / "deployment.json"
        deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
        deployment["files"]["threshold.json"] = sha256_file(output / "qcoc" / "threshold.json")
        write_json(deployment_path, deployment)
        with pytest.raises(ValueError, match="threshold binding differs"):
            CSVDetector.load(output / "qcoc")
    finally:
        torch.set_num_threads(old_threads)
