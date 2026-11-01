"""OOF teacher isolation, identity, completion and tamper tests on small fixtures."""
import copy
import shutil

import numpy as np
import pandas as pd
import pytest

from ids import distillation
from ids.common import canonical_hash, read_json, sha256_file, write_json
from ids.data import FEATURES, load_partition, make_synthetic_dataset, prepare_dataset


def _fingerprint():
    files = {"src/ids/fixture.py": "a" * 64}
    return {"commit": None, "dirty": None, "files": files, "source_sha256": canonical_hash(files)}


TEACHER = {"model": "hgb", "features": {"method": "all"},
           "hyperparameters": {"max_iter": 3, "max_leaf_nodes": 7, "min_samples_leaf": 2,
                               "learning_rate": .1, "early_stopping": False}}


@pytest.fixture(autouse=True)
def stable_environment(monkeypatch):
    monkeypatch.setattr(distillation, "code_fingerprint", _fingerprint)
    monkeypatch.setattr(distillation, "environment_snapshot", lambda **_kwargs: {"purpose": "test_fixture"})


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    root = tmp_path_factory.mktemp("oof-template")
    train_source, test_source = make_synthetic_dataset(root / "raw", n_train=600, n_test=120, seed=8)
    raw = pd.read_csv(train_source)
    # Repeated predictors keep their original labels, deliberately including
    # possible label conflicts. The feature group must nevertheless stay whole.
    raw.loc[300:329, list(FEATURES)] = raw.loc[0:29, list(FEATURES)].to_numpy()
    raw.to_csv(train_source, index=False)
    data = root / "prepared"
    prepare_dataset(train_source, test_source, data, seed=17, data_kind="synthetic")
    target = root / "oof"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(distillation, "code_fingerprint", _fingerprint)
        patch.setattr(distillation, "environment_snapshot", lambda **_kwargs: {"purpose": "test_fixture"})
        distillation.build_oof_targets(data, TEACHER, target, seed=17, folds=3, fold_seed=19, threads=1)
    return {"data": data, "oof": target}


@pytest.fixture
def workspace(template, tmp_path):
    data, oof = tmp_path / "prepared", tmp_path / "oof"
    shutil.copytree(template["data"], data)
    shutil.copytree(template["oof"], oof)
    return {"data": data, "oof": oof}


def _reseal(path):
    manifest = read_json(path / "manifest.json")
    for name in manifest["files"]:
        manifest["files"][name] = sha256_file(path / name)
    write_json(path / "manifest.json", manifest)
    digest = sha256_file(path / "manifest.json")
    (path / "manifest.sha256").write_text(digest + "\n", encoding="ascii")
    completion = read_json(path / "completion.json")
    completion["manifest_sha256"] = digest
    write_json(path / "completion.json", completion)


def test_oof_complete_group_isolation_coverage_and_train_order(template):
    manifest = distillation.verify_oof_targets(template["oof"], template["data"])
    targets = pd.read_parquet(template["oof"] / "targets.parquet")
    train = load_partition(template["data"], "train")
    assert manifest["kind"] == "group_oof_teacher_targets"
    assert manifest["folds"] == 3
    assert manifest["n_training_rows"] == len(train)
    assert targets["sample_id"].is_unique
    assert targets["sample_id"].tolist() == train["sample_id"].tolist()
    assert targets["feature_hash"].duplicated().any()
    assert targets.groupby("feature_hash")["fold"].nunique().max() == 1
    assert set(targets["fold"]) == {0, 1, 2}
    assert np.isfinite(targets["teacher_score"]).all()
    assert targets["teacher_score"].between(0, 1).all()
    assert manifest["fold_fit_seconds_total"] > 0
    assert manifest["generation_seconds"] >= manifest["fold_fit_seconds_total"]
    assert manifest["train_identity_sha256"] == canonical_hash(
        train[["sample_id", "feature_hash", "label"]].to_dict(orient="records"))


def test_every_teacher_fit_preprocesses_only_other_groups_and_reads_only_train(template, tmp_path, monkeypatch):
    from ids import models
    loads, fitting, predictions = [], [], []
    original_fit = distillation.train_model
    original_predict = models.ModelBundle.predict_scores

    def guarded_load(data, split):
        assert split == "train", "OOF generation opened a reserved partition"
        loads.append(split)
        return load_partition(data, split)

    def fit(train, validation, config, seed, device):
        # Even the existing train_model diagnostic validation argument contains
        # fitting rows only: heldout labels never reach teacher fitting.
        assert train["sample_id"].tolist() == validation["sample_id"].tolist()
        fitting.append(set(train["sample_id"]))
        bundle = original_fit(train, validation, config, seed, device)
        assert bundle.estimator.early_stopping is False
        for field, categories in zip(bundle.preprocessor.categorical_columns, bundle.preprocessor.encoder.categories_):
            assert set(categories) <= set(train[field].fillna("<MISSING>").astype(str))
        for j, name in enumerate(bundle.preprocessor.numeric_columns):
            assert bundle.preprocessor.imputer.statistics_[j] == pytest.approx(train[name].median())
        return bundle

    def predict(bundle, frame):
        predictions.append(set(frame["sample_id"]))
        return original_predict(bundle, frame)

    monkeypatch.setattr(distillation, "load_partition", guarded_load)
    monkeypatch.setattr(distillation, "train_model", fit)
    monkeypatch.setattr(models.ModelBundle, "predict_scores", predict)
    result = distillation.build_oof_targets(template["data"], TEACHER, tmp_path / "fresh", 17, 3, 19, 1)
    targets = pd.read_parquet(result / "targets.parquet")
    assert loads and set(loads) == {"train"}
    assert len(fitting) == 3 and len(predictions) == 6
    for index in range(3):
        heldout = set(targets.loc[targets["fold"] == index, "sample_id"])
        assert fitting[index].isdisjoint(heldout)
        assert predictions[2 * index] == fitting[index]
        assert predictions[2 * index + 1] == heldout


def test_score_loader_aligns_by_ids_and_rejects_semantic_mismatch(template):
    train = load_partition(template["data"], "train")
    original = distillation.load_oof_scores(template["oof"], train)
    reverse = train.iloc[::-1]
    np.testing.assert_array_equal(distillation.load_oof_scores(template["oof"], reverse), original[::-1])
    bad = train.copy()
    bad.loc[bad.index[0], "label"] = 1 - bad.iloc[0]["label"]
    with pytest.raises(ValueError, match="misaligned"):
        distillation.load_oof_scores(template["oof"], bad)
    bad = train.copy()
    bad.loc[bad.index[0], "feature_hash"] = "wrong-group"
    with pytest.raises(ValueError, match="misaligned"):
        distillation.load_oof_scores(template["oof"], bad)
    with pytest.raises(ValueError, match="cover exactly"):
        distillation.load_oof_scores(template["oof"], train.iloc[:-1])
    with pytest.raises(ValueError, match="unique"):
        distillation.load_oof_scores(template["oof"], pd.concat([train.iloc[:1], train.iloc[:1]]))


def test_completed_oof_resume_verifies_without_refitting(template, monkeypatch):
    monkeypatch.setattr(distillation, "train_model", lambda *_a, **_kw: pytest.fail("completed OOF retrained"))
    before = sha256_file(template["oof"] / "completion.json")
    result = distillation.build_oof_targets(template["data"], TEACHER, template["oof"], 17, 3, 19, 1)
    assert result == template["oof"]
    assert sha256_file(template["oof"] / "completion.json") == before


def test_incomplete_oof_is_preserved_and_never_overwritten(workspace):
    (workspace["oof"] / "completion.json").unlink()
    before = {p.name: sha256_file(p) for p in workspace["oof"].iterdir()}
    with pytest.raises(FileExistsError, match="Incomplete"):
        distillation.build_oof_targets(workspace["data"], TEACHER, workspace["oof"], 17, 3, 19, 1)
    assert {p.name: sha256_file(p) for p in workspace["oof"].iterdir()} == before


@pytest.mark.parametrize("artifact", ["targets.parquet", "fold-00.joblib", "config.json", "folds.json"])
def test_changed_artifacts_are_rejected_without_deserializing_models(workspace, monkeypatch, artifact):
    import joblib
    monkeypatch.setattr(joblib, "load", lambda *_a, **_kw: pytest.fail("OOF verification loaded a model"))
    path = workspace["oof"] / artifact
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        distillation.verify_oof_targets(workspace["oof"])


def test_resealed_wrong_fold_assignments_are_rejected(workspace):
    path = workspace["oof"] / "targets.parquet"
    frame = pd.read_parquet(path)
    duplicate = frame["feature_hash"].duplicated(keep=False)
    index = frame.index[duplicate][0]
    frame.loc[index, "fold"] = (frame.loc[index, "fold"] + 1) % 3
    frame.to_parquet(path, index=False)
    _reseal(workspace["oof"])
    with pytest.raises(ValueError, match="fold assignments"):
        distillation.verify_oof_targets(workspace["oof"])


def test_resealed_fold_identity_metadata_is_rejected(workspace):
    path = workspace["oof"] / "folds.json"
    value = read_json(path)
    value["folds"][0]["holdout_sample_ids_sha256"] = "f" * 64
    write_json(path, value)
    _reseal(workspace["oof"])
    with pytest.raises(ValueError, match="fold train/holdout identity"):
        distillation.verify_oof_targets(workspace["oof"])


def test_resume_rejects_changed_source_or_teacher_configuration(template, monkeypatch):
    altered = copy.deepcopy(TEACHER)
    altered["hyperparameters"]["max_iter"] = 4
    with pytest.raises(ValueError, match="expected configuration"):
        distillation.build_oof_targets(template["data"], altered, template["oof"], 17, 3, 19, 1)
    changed = _fingerprint()
    changed["files"]["src/ids/fixture.py"] = "b" * 64
    changed["source_sha256"] = canonical_hash(changed["files"])
    monkeypatch.setattr(distillation, "code_fingerprint", lambda: changed)
    with pytest.raises(ValueError, match="expected configuration"):
        distillation.build_oof_targets(template["data"], TEACHER, template["oof"], 17, 3, 19, 1)


def test_changed_training_partition_is_rejected(workspace):
    with (workspace["data"] / "train.parquet").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Partition checksum"):
        distillation.verify_oof_targets(workspace["oof"], workspace["data"])


def test_verify_refuses_reserved_or_traversal_inventory_before_read(workspace):
    manifest = read_json(workspace["oof"] / "manifest.json")
    manifest["files"]["../test.parquet"] = "a" * 64
    write_json(workspace["oof"] / "manifest.json", manifest)
    digest = sha256_file(workspace["oof"] / "manifest.json")
    (workspace["oof"] / "manifest.sha256").write_text(digest + "\n")
    completion = read_json(workspace["oof"] / "completion.json")
    completion["manifest_sha256"] = digest
    write_json(workspace["oof"] / "completion.json", completion)
    with pytest.raises(ValueError, match="inventory"):
        distillation.verify_oof_targets(workspace["oof"])


@pytest.mark.parametrize("mutation", ["model", "selection", "early_stop", "threads"])
def test_invalid_teacher_is_rejected_before_training(template, tmp_path, mutation, monkeypatch):
    teacher = copy.deepcopy(TEACHER)
    if mutation == "model":
        teacher["model"] = "mlp"
    elif mutation == "selection":
        teacher["features"] = {"method": "mutual_info", "k": 8}
    elif mutation == "early_stop":
        teacher["hyperparameters"]["early_stopping"] = True
    else:
        teacher["hyperparameters"]["num_threads"] = 2
    monkeypatch.setattr(distillation, "load_partition", lambda *_a: pytest.fail("invalid teacher opened data"))
    with pytest.raises(ValueError, match="teacher|Teacher"):
        distillation.build_oof_targets(template["data"], teacher, tmp_path / "invalid", 17, 3, 19, 1)


def test_fold_support_failure_does_not_try_other_seeds():
    frame = pd.DataFrame({"sample_id": ["a", "b", "c", "d"],
                          "feature_hash": ["same", "same", "other", "other"], "label": [0, 1, 0, 1]})
    with pytest.raises(ValueError, match="enough feature groups"):
        distillation._fold_assignment(frame, folds=3, fold_seed=19)


def test_successful_verification_never_loads_teacher_models(template, monkeypatch):
    import joblib
    monkeypatch.setattr(joblib, "load", lambda *_a, **_kw: pytest.fail("verification deserialized a teacher"))
    assert distillation.verify_oof_targets(template["oof"], template["data"])["n_training_rows"] > 0
    with pytest.raises(ValueError, match="expected configuration"):
        distillation.verify_oof_targets(template["oof"], expected={"misspelled_binding": None})


def test_source_change_during_generation_never_marks_completion(template, tmp_path, monkeypatch):
    calls = []

    def fingerprint():
        value = _fingerprint()
        if calls:
            value["files"]["src/ids/fixture.py"] = "b" * 64
            value["source_sha256"] = canonical_hash(value["files"])
        calls.append(1)
        return value

    monkeypatch.setattr(distillation, "code_fingerprint", fingerprint)
    output = tmp_path / "changed-source"
    with pytest.raises(ValueError, match="Source changed"):
        distillation.build_oof_targets(template["data"], TEACHER, output, 17, 3, 19, 1)
    assert output.is_dir()
    assert not (output / "completion.json").exists()
    assert not (output / "manifest.json").exists()


def test_resealed_nonfinite_teacher_scores_are_rejected(workspace):
    path = workspace["oof"] / "targets.parquet"
    targets = pd.read_parquet(path)
    targets.loc[0, "teacher_score"] = np.nan
    targets.to_parquet(path, index=False)
    _reseal(workspace["oof"])
    with pytest.raises(ValueError, match="finite"):
        distillation.verify_oof_targets(workspace["oof"])
