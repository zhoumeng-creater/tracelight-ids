"""Budget-transfer lab orchestration: mocked GPU/data/runner, no real fitting."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import yaml

from ids.cicids import CIC_CATEGORICAL, CIC_DISCRETE, CIC_FEATURES, CIC_SCHEMA_VERSION
from ids.common import canonical_hash
from ids.data import FEATURES, SCHEMA_VERSION

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("budget_lab_script", REPO / "scripts/run_budget_transfer_lab.py")
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)
PROFILES = {profile: yaml.safe_load((REPO / "configs" / f"budget-transfer-{profile}.yaml").read_text(encoding="utf-8"))
            for profile in ("screen", "pilot", "full")}


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def hardware(name="NVIDIA GeForce RTX 4090 D"):
    return {"recorded_at": "fake-clock", "cpu_logical": 16, "packages": {"torch": "2.6.0+cu126"},
            "torch": {"cuda_available": True, "cuda_runtime": "12.6",
                      "gpus": [{"index": 0, "name": name, "total_memory_bytes": 24 * 1024 ** 3}]}}


def source():
    return {"commit": "e" * 40, "dirty": False, "source_sha256": "a" * 64, "files": {}}


def manifest(name):
    if name == "unsw":
        value = {"schema_version": SCHEMA_VERSION, "features": list(FEATURES), "data_kind": "mirror"}
    else:
        value = {"schema_version": CIC_SCHEMA_VERSION, "features": list(CIC_FEATURES),
                 "data_kind": "official_repaired", "feature_schema": {
                     "name": CIC_SCHEMA_VERSION, "features": list(CIC_FEATURES),
                     "categorical": list(CIC_CATEGORICAL), "discrete": list(CIC_DISCRETE)}}
    value["partitions"] = {split: {"file": split + ".parquet", "n_rows": 100, "sha256": "b" * 64}
                           for split in ("train", "validation", "calibration", "test")}
    return value


def save_manifest(path, value):
    put_json(path / "manifest.json", value)
    (path / "manifest.sha256").write_text(lab.sha256_file(path / "manifest.json") + "\n", encoding="ascii")


def old_v3(path, data):
    plan = {"config": {"purpose": "risk_v3_development_only", "cohort_policy": "complete_development",
                       "seeds": [17, 29, 43], "max_train_rows": None, "max_validation_rows": None},
            "data": data, "source": {"commit": "f" * 40, "source_sha256": "d" * 64},
            "role_manifest_hash": "c" * 64, "reserved_partitions_read": []}
    plan["plan_id"] = canonical_hash(plan)
    put_json(path / "plan.json", plan)
    # No teacher/role/parquet payload exists: dry-run cannot read any of them.
    put_json(path / lab.V3_COMPLETE, {"status": "development_complete", "reserved_partitions_read": [],
             "plan_id": plan["plan_id"], "files": {"plan.json": lab.sha256_file(path / "plan.json"),
             "teachers/s17/reference.joblib": "e" * 64}})


def seal(path, cfg, v3, data, *, declined=False):
    from ids.budget_transfer import METHODS
    path.mkdir(parents=True)
    put_json(path / "plan.json", {"config": cfg, "v3_dir": str(Path(v3).resolve()),
                                  "data_dir": str(Path(data).resolve())})
    (path / "result.txt").write_text("mock outputs; no model or data payload loaded", encoding="utf-8")
    runs, skipped = [], []
    for seed in cfg["seeds"]:
        for budget in ("all", "reduced"):
            for method in METHODS:
                r = {"seed": seed, "budget": budget, "method": method}
                if (budget == "all" and method == "direct_rank_matched") or (
                        declined and method not in {"hard", "direct_pauc"}):
                    skipped.append({**r, "reason": "exact alias or registered qualification decline"})
                else:
                    runs.append(r)
    n = len(cfg["seeds"])
    put_json(path / "summary.json", {"teacher_fits": 2*n, "warmup_fits": 4*n, "student_fits": len(runs),
             "runs": runs, "skipped": skipped, "independent_confirmation": False})
    put_json(path / lab.COMPLETE, {"schema_version": 1, "status": "development_complete", "reserved_partitions_read": [],
             "n_teacher_fits": 2*n, "n_warmup_fits": 4*n, "n_student_fits": len(runs),
             "files": {name: lab.sha256_file(path / name) for name in ("plan.json", "result.txt", "summary.json")}})


def fake_verify(path):
    path = Path(path)
    complete = lab.read_json(path / lab.COMPLETE)
    actual = {p.relative_to(path).as_posix(): lab.sha256_file(p) for p in path.rglob("*")
              if p.is_file() and p.name != lab.COMPLETE}
    if actual != complete["files"]:
        raise ValueError("mock completed artifact changed")
    return complete


@pytest.fixture
def ready(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    for profile, cfg in PROFILES.items():
        (repo / "configs" / f"budget-transfer-{profile}.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(lab, "REPO", repo)
    monkeypatch.setattr(lab, "check_environment", hardware)
    monkeypatch.setattr(lab, "_activate_cuda_zero", lambda: None)
    monkeypatch.setattr(lab, "code_fingerprint", source)
    monkeypatch.setattr(lab, "verify_transfer", fake_verify)
    inputs, calls = {}, []
    for name in ("unsw", "cicids"):
        data, v3 = tmp_path / ("data-" + name), tmp_path / ("v3-" + name)
        value = manifest(name)
        save_manifest(data, value)
        old_v3(v3, value)
        inputs[name] = {"data_dir": data, "v3_dir": v3}

    def runner(v3, data, cfg, output, **kwargs):
        calls.append((Path(v3), Path(data), copy.deepcopy(cfg), Path(output), kwargs))
        if kwargs["resume"]:
            if lab.read_json(Path(output) / "plan.json") != {
                    "config": cfg, "v3_dir": str(Path(v3).resolve()), "data_dir": str(Path(data).resolve())}:
                raise ValueError("mock runner input binding changed")
        else:
            seal(Path(output), cfg, v3, data)
        return fake_verify(output)

    monkeypatch.setattr(lab, "run_transfer_research", runner)
    return inputs, calls


def arguments(inputs, *names):
    return {f"{name}_{key}": value for name in names for key, value in inputs[name].items()}


@pytest.mark.parametrize("kwargs", [{}, {"unsw_v3_dir": "v3"}, {"cicids_data_dir": "data"},
                                    {"unsw_v3_dir": "v3", "cicids_data_dir": "data"}])
def test_environment_pairs_required_before_any_probe(tmp_path, monkeypatch, kwargs):
    monkeypatch.setattr(lab, "check_environment", lambda: pytest.fail("probe after invalid pairs"))
    with pytest.raises(ValueError, match="environment|Each"):
        lab.run_lab(profile="pilot", work_dir=tmp_path / "work", **kwargs)


def test_dry_run_metadata_only_has_no_fit_or_full_artifact_verifier(ready, tmp_path, monkeypatch):
    inputs, calls = ready
    monkeypatch.setattr(lab, "run_transfer_research", lambda *a, **kw: pytest.fail("dry-run fitted"))
    monkeypatch.setattr(lab, "verify_transfer", lambda *a: pytest.fail("dry-run loaded artifacts"))
    work = tmp_path / "check"
    result = lab.run_lab(profile="pilot", work_dir=work, dry_run=True, **arguments(inputs, "unsw", "cicids"))
    assert result["status"] == "budget_transfer_lab_plan_only" and calls == []
    assert result["expected_all_environments"]["total_fits"] == 46
    saved = lab.read_json(work / lab.PREFLIGHT)
    assert saved["binding"]["execution_mode"] == "dry_run"
    assert saved["binding"]["device"] == "cuda" and saved["binding"]["cuda_index"] == 0
    assert saved["binding"]["interpreter"]["executable"] == str(Path(sys.executable).absolute())
    assert saved["reserved_partitions_read"] == []
    assert [p.name for p in work.iterdir()] == [lab.PREFLIGHT]


def test_wrong_gpu_rejected_before_source_input_metadata(ready, tmp_path, monkeypatch):
    inputs, calls = ready
    monkeypatch.setattr(lab, "check_environment", lambda: hardware("RTX4050 Laptop GPU"))
    monkeypatch.setattr(lab, "_dataset_record", lambda *a: pytest.fail("data access after rejected GPU"))
    monkeypatch.setattr(lab, "_activate_cuda_zero", lambda: pytest.fail("bad GPU activated"))
    work = tmp_path / "work"
    with pytest.raises(ValueError, match="CUDA0"):
        lab.run_lab(profile="pilot", work_dir=work, dry_run=True, **arguments(inputs, "unsw"))
    assert not work.exists() and not calls


def test_locked_environment_rejection_precedes_inputs_and_outputs(ready, tmp_path, monkeypatch):
    inputs, calls = ready
    def fail():
        raise ValueError("Locked project dependency mismatch")
    monkeypatch.setattr(lab, "check_environment", fail)
    monkeypatch.setattr(lab, "_dataset_record", lambda *a: pytest.fail("read after dependency rejection"))
    with pytest.raises(ValueError, match="dependency mismatch"):
        lab.run_lab(profile="pilot", work_dir=tmp_path / "work", **arguments(inputs, "unsw"))
    assert not (tmp_path / "work").exists() and not calls


@pytest.mark.parametrize("mutation", [{"dirty": True}, {"dirty": None}, {"commit": None}, {"commit": "not-git"}])
def test_clean_committed_source_required_before_probe(ready, tmp_path, monkeypatch, mutation):
    inputs, calls = ready
    monkeypatch.setattr(lab, "code_fingerprint", lambda: {**source(), **mutation})
    monkeypatch.setattr(lab, "check_environment", lambda: pytest.fail("probe for uncommitted source"))
    with pytest.raises(ValueError, match="clean committed"):
        lab.run_lab(profile="pilot", work_dir=tmp_path / "work", dry_run=True, **arguments(inputs, "unsw"))
    assert not calls


@pytest.mark.parametrize("profile,multiplier", [("pilot", 1), ("full", 3)])
def test_serial_run_exact_six_packages_and_bytewise_receipt(ready, tmp_path, profile, multiplier):
    inputs, calls = ready
    work = tmp_path / "work"
    result = lab.run_lab(work_dir=work, profile=profile, **arguments(inputs, "unsw", "cicids"))
    assert result["status"] == "budget_transfer_lab_development_complete"
    assert result["expected_per_environment"]["total_fits"] == 23*multiplier
    assert [call[0] for call in calls] == [inputs["unsw"]["v3_dir"], inputs["cicids"]["v3_dir"]]
    assert all(c[-1] == {"device": "cuda", "resume": False} for c in calls)
    assert all(c[2]["macro_steps"] == 40 and c[2]["warmup_epochs"] == 40 for c in calls)
    expected_names = {"BUDGET_TRANSFER_UNSW.zip", "BUDGET_TRANSFER_UNSW.zip.sha256",
                      "BUDGET_TRANSFER_CICIDS.zip", "BUDGET_TRANSFER_CICIDS.zip.sha256",
                      lab.PREFLIGHT, lab.PREFLIGHT + ".sha256"}
    assert {p.name for p in (work / "packages").iterdir()} == expected_names
    assert (work / "packages" / lab.PREFLIGHT).read_bytes() == (work / lab.PREFLIGHT).read_bytes()
    for name in ("UNSW", "CICIDS"):
        package = work / "packages" / f"BUDGET_TRANSFER_{name}.zip"
        assert package.with_name(package.name + ".sha256").read_text().strip() == lab.sha256_file(package)
        with zipfile.ZipFile(package) as archive:
            listing = archive.read(lab.PACKAGE_MANIFEST).decode("utf-8")
            payload = dict((path, digest) for digest, path in (row.split("  ", 1) for row in listing.splitlines()))
            assert set(archive.namelist()) == set(payload) | {lab.PACKAGE_MANIFEST}
            for path, digest in payload.items():
                assert hashlib.sha256(archive.read(path)).hexdigest() == digest
            assert not any("teachers/s17/reference.joblib" in p for p in archive.namelist())


def test_qualification_cannot_skip_any_registered_kd_condition(ready, tmp_path, monkeypatch):
    inputs, calls = ready
    def declined(v3, data, cfg, output, **kwargs):
        calls.append(kwargs)
        seal(Path(output), cfg, v3, data, declined=True)
    monkeypatch.setattr(lab, "run_transfer_research", declined)
    work = tmp_path / "work"
    with pytest.raises(ValueError, match="planned fit counts"):
        lab.run_lab(profile="pilot", work_dir=work, **arguments(inputs, "unsw"))
    done = lab.read_json(work / "budget-transfer-unsw" / lab.COMPLETE)
    assert done["n_student_fits"] == 4 and len(calls) == 1
    assert not (work / "packages").exists()


def test_complete_only_resume_preserves_packages_and_original_receipt(ready, tmp_path):
    inputs, calls = ready
    work = tmp_path / "work"
    args = arguments(inputs, "unsw", "cicids")
    lab.run_lab(profile="pilot", work_dir=work, **args)
    before = {p.name: p.read_bytes() for p in (work / "packages").iterdir()}
    result = lab.run_lab(profile="pilot", work_dir=work, resume=True, **args)
    assert result["environments"]["unsw"]["package"]["status"] == "verified_existing"
    assert {p.name: p.read_bytes() for p in (work / "packages").iterdir()} == before
    assert len(calls) == 4 and all(c[-1]["resume"] for c in calls[2:])


def test_dry_run_work_cannot_be_reused_as_formal_run(ready, tmp_path):
    inputs, calls = ready
    work, args = tmp_path / "check", arguments(inputs, "unsw")
    lab.run_lab(profile="pilot", work_dir=work, dry_run=True, **args)
    with pytest.raises(ValueError, match="binding changed"):
        lab.run_lab(profile="pilot", work_dir=work, resume=True, **args)
    assert not calls


def test_partial_environment_is_preserved_and_never_dispatched(ready, tmp_path):
    inputs, calls = ready
    work, args = tmp_path / "work", arguments(inputs, "unsw")
    # Generate only a valid formal-mode preflight via an intentionally stopped runner.
    original = lab.run_transfer_research
    def stop(*a, **kw):
        raise RuntimeError("simulated interruption")
    lab.run_transfer_research = stop
    try:
        with pytest.raises(RuntimeError, match="interruption"):
            lab.run_lab(profile="pilot", work_dir=work, **args)
    finally:
        lab.run_transfer_research = original
    partial = work / "budget-transfer-unsw"
    partial.mkdir()
    sentinel = partial / "keep.txt"
    sentinel.write_text("keep partial", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Incomplete"):
        lab.run_lab(profile="pilot", work_dir=work, resume=True, **args)
    assert sentinel.read_text() == "keep partial" and not calls


@pytest.mark.parametrize("changed", ["threads", "source_hash", "commit", "interpreter", "hardware"])
def test_resume_binds_configuration_source_interpreter_hardware(ready, tmp_path, monkeypatch, changed):
    inputs, calls = ready
    work, args = tmp_path / "work", arguments(inputs, "unsw")
    lab.run_lab(profile="pilot", work_dir=work, **args)
    calls.clear()
    if changed == "threads":
        args["threads"] = 7
    elif changed in {"source_hash", "commit"}:
        updated = {**source(), "source_sha256" if changed == "source_hash" else "commit": "f" * (64 if changed == "source_hash" else 40)}
        monkeypatch.setattr(lab, "code_fingerprint", lambda: updated)
    elif changed == "interpreter":
        monkeypatch.setattr(lab.sys, "prefix", "different-prefix")
    else:
        monkeypatch.setattr(lab, "check_environment", lambda: {**hardware(), "cpu_logical": 8})
    with pytest.raises(ValueError, match="binding changed"):
        lab.run_lab(profile="pilot", work_dir=work, resume=True, **args)
    assert not calls


@pytest.mark.parametrize("mutation", ["plan_checksum", "reserved", "bounded", "missing_seed"])
def test_source_v3_metadata_binding_defects_rejected(ready, tmp_path, mutation):
    inputs, calls = ready
    root = inputs["unsw"]["v3_dir"]
    plan, done = lab.read_json(root / "plan.json"), lab.read_json(root / lab.V3_COMPLETE)
    if mutation == "reserved":
        done["reserved_partitions_read"] = ["test"]
    else:
        if mutation == "bounded":
            plan["config"]["cohort_policy"] = "bounded_pilot"
        elif mutation == "missing_seed":
            plan["config"]["seeds"] = [29, 43]
        else:
            plan["extra"] = True
        if mutation != "plan_checksum":
            plan["plan_id"] = canonical_hash({k: v for k, v in plan.items() if k != "plan_id"})
        put_json(root / "plan.json", plan)
        if mutation != "plan_checksum":
            done["plan_id"] = plan["plan_id"]
            done["files"]["plan.json"] = lab.sha256_file(root / "plan.json")
    put_json(root / lab.V3_COMPLETE, done)
    with pytest.raises(ValueError):
        lab.run_lab(profile="pilot", work_dir=tmp_path / "work", **arguments(inputs, "unsw"))
    assert not calls


def test_prepared_identity_must_match_old_v3(ready, tmp_path):
    inputs, calls = ready
    value = manifest("unsw")
    value["partitions"]["train"]["n_rows"] += 1
    save_manifest(inputs["unsw"]["data_dir"], value)
    with pytest.raises(ValueError, match="original V3 data identity"):
        lab.run_lab(profile="pilot", work_dir=tmp_path / "work", dry_run=True, **arguments(inputs, "unsw"))
    assert not calls


@pytest.mark.parametrize("mutation", ["synthetic", "wrong_schema"])
def test_only_real_canonical_prepared_schema_is_accepted(ready, tmp_path, mutation):
    inputs, calls = ready
    value = manifest("unsw")
    value["data_kind" if mutation == "synthetic" else "schema_version"] = "synthetic" if mutation == "synthetic" else "unknown"
    save_manifest(inputs["unsw"]["data_dir"], value)
    with pytest.raises(ValueError, match="exact schema"):
        lab.run_lab(profile="pilot", work_dir=tmp_path / "work", dry_run=True, **arguments(inputs, "unsw"))
    assert not calls


@pytest.mark.parametrize("input_key", ["data_dir", "v3_dir"])
@pytest.mark.parametrize("relation", ["work_below", "work_above"])
def test_work_and_immutable_inputs_cannot_nest(ready, input_key, relation):
    inputs, calls = ready
    old = inputs["unsw"][input_key]
    work = old / "new-work" if relation == "work_below" else old.parent
    with pytest.raises(ValueError, match="non-nested"):
        lab.run_lab(profile="pilot", work_dir=work, **arguments(inputs, "unsw"))
    assert not calls


@pytest.mark.parametrize("change", ["schema", "counts", "condition_inventory"])
def test_invalid_completion_is_not_packaged(ready, tmp_path, monkeypatch, change):
    inputs, calls = ready
    original = lab.run_transfer_research
    def changed(*args, **kwargs):
        original(*args, **kwargs)
        root = Path(args[3])
        done = lab.read_json(root / lab.COMPLETE)
        if change == "schema":
            done["schema_version"] = 2
        elif change == "counts":
            done["n_teacher_fits"] = 3
        else:
            summary = lab.read_json(root / "summary.json")
            summary["runs"][0]["method"] = "unregistered"
            put_json(root / "summary.json", summary)
            done["files"]["summary.json"] = lab.sha256_file(root / "summary.json")
        put_json(root / lab.COMPLETE, done)
    monkeypatch.setattr(lab, "run_transfer_research", changed)
    work = tmp_path / "work"
    with pytest.raises(ValueError):
        lab.run_lab(profile="pilot", work_dir=work, **arguments(inputs, "unsw"))
    assert len(calls) == 1 and not (work / "packages").exists()


@pytest.mark.parametrize("which", ["source", "v3", "data"])
def test_inputs_changing_during_run_block_package(ready, tmp_path, monkeypatch, which):
    inputs, calls = ready
    original = lab.run_transfer_research
    def changed(*args, **kwargs):
        original(*args, **kwargs)
        if which == "source":
            monkeypatch.setattr(lab, "code_fingerprint", lambda: {**source(), "source_sha256": "f" * 64})
        elif which == "v3":
            path = inputs["unsw"]["v3_dir"] / lab.V3_COMPLETE
            put_json(path, {**lab.read_json(path), "changed": True})
        else:
            value = manifest("unsw")
            value["partitions"]["train"]["n_rows"] += 1
            save_manifest(inputs["unsw"]["data_dir"], value)
    monkeypatch.setattr(lab, "run_transfer_research", changed)
    work = tmp_path / "work"
    with pytest.raises(ValueError, match="changed during"):
        lab.run_lab(profile="pilot", work_dir=work, **arguments(inputs, "unsw"))
    assert len(calls) == 1 and not (work / "packages").exists()


def test_corrupt_existing_zip_or_partial_sidecar_is_preserved(ready, tmp_path):
    inputs, _ = ready
    work, args = tmp_path / "work", arguments(inputs, "unsw")
    lab.run_lab(profile="pilot", work_dir=work, **args)
    package = work / "packages" / "BUDGET_TRANSFER_UNSW.zip"
    package.write_bytes(package.read_bytes() + b"tampered")
    before = package.read_bytes()
    with pytest.raises(ValueError, match="sidecar mismatch"):
        lab.run_lab(profile="pilot", work_dir=work, resume=True, **args)
    assert package.read_bytes() == before


@pytest.mark.parametrize("change", [{"seeds": [29]}, {"macro_steps": 20}, {"warmup_epochs": 20},
                                    {"purpose": "final_test"}, {"row_cap": 100}, {"kd_weight": .8}])
def test_lab_profiles_reject_changed_scope_or_different_training_budgets(ready, change):
    cfg = copy.deepcopy(PROFILES["pilot"])
    cfg.update(change)
    lab._profile_path("pilot").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError):
        lab.load_profile("pilot")


def test_help_available_outside_repo_without_loading_a_trainer(tmp_path):
    result = subprocess.run([sys.executable, str(REPO / "scripts/run_budget_transfer_lab.py"), "--help"],
                            cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0 and "--unsw-v3-dir" in result.stdout and "--cicids-data-dir" in result.stdout


def test_default_screen_preflight_stops_at_twenty_four_fits_without_loading_rows(ready, tmp_path, monkeypatch):
    inputs, calls = ready
    monkeypatch.setattr(lab, "require_screen_dependencies", lambda: None)
    monkeypatch.setattr(lab, "run_transfer_screen", lambda *a, **kw: pytest.fail("dry-run fitted"))
    result = lab.run_lab(work_dir=tmp_path / "screen-plan", dry_run=True, **arguments(inputs, "unsw", "cicids"))
    assert result["profile"] == "screen" and not calls
    assert result["expected_all_environments"] == {
        "teacher_fits": 4, "warmup_fits": 8, "student_fits": 8, "native_tree_fits": 4, "total_fits": 24}
    plan = lab.read_json(tmp_path / "screen-plan" / lab.PREFLIGHT)
    assert "no V loading" in plan["scope"]


def test_default_screen_dispatch_packages_and_resumes_only_its_stage(ready, tmp_path, monkeypatch):
    inputs, calls = ready
    monkeypatch.setattr(lab, "require_screen_dependencies", lambda: None)
    monkeypatch.setattr(lab, "run_transfer_research", lambda *a, **kw: pytest.fail("screen launched legacy matrix"))
    def verify_screen(folder):
        folder = Path(folder)
        done = lab.read_json(folder / lab.SCREEN_COMPLETE)
        files = {p.relative_to(folder).as_posix(): lab.sha256_file(p) for p in folder.rglob("*")
                 if p.is_file() and p.name != lab.SCREEN_COMPLETE}
        if files != done["files"]:
            raise ValueError("screen inventory changed")
        return done
    def runner(v3, data, cfg, output, **kwargs):
        calls.append(kwargs)
        folder = Path(output)
        if kwargs["resume"]:
            return verify_screen(folder)
        folder.mkdir(parents=True)
        rows = [{"seed": 17, "budget": b, "method": m} for b in ("all", "reduced") for m in ("hard", "direct_pauc")]
        put_json(folder / "summary.json", {"teacher_fits": 2, "warmup_fits": 4, "student_fits": 4,
            "native_tree_fits": 2, "runs": rows, "independent_confirmation": False, "validation_loaded": False,
            "automatic_expansion": False})
        put_json(folder / lab.SCREEN_COMPLETE, {"schema_version": 1, "status": "screen_complete",
            "reserved_partitions_read": [], "validation_loaded": False,
            "n_teacher_fits": 2, "n_warmup_fits": 4, "n_student_fits": 4, "n_native_tree_fits": 2,
            "files": {"summary.json": lab.sha256_file(folder / "summary.json")}})
        return verify_screen(folder)
    monkeypatch.setattr(lab, "run_transfer_screen", runner)
    monkeypatch.setattr(lab, "verify_transfer_screen", verify_screen)
    work = tmp_path / "screen-work"
    result = lab.run_lab(work_dir=work, **arguments(inputs, "unsw", "cicids"))
    assert result["status"] == "budget_transfer_lab_screen_complete"
    assert len(calls) == 2 and len(list((work / "packages").iterdir())) == 6
    result = lab.run_lab(work_dir=work, resume=True, **arguments(inputs, "unsw", "cicids"))
    assert len(calls) == 4 and all(c["resume"] for c in calls[2:])
    assert result["environments"]["unsw"]["package"]["status"] == "verified_existing"
    with pytest.raises(ValueError, match="binding changed"):
        lab.run_lab(work_dir=work, profile="pilot", resume=True, **arguments(inputs, "unsw", "cicids"))


def test_screen_requires_pinned_catboost_before_reading_data(ready, tmp_path, monkeypatch):
    inputs, _ = ready
    def missing():
        raise ImportError("catboost==1.2.10 required")
    monkeypatch.setattr(lab, "require_screen_dependencies", missing)
    monkeypatch.setattr(lab, "_dataset_record", lambda *a: pytest.fail("read data after missing dependency"))
    with pytest.raises(ImportError, match="catboost"):
        lab.run_lab(work_dir=tmp_path / "missing", **arguments(inputs, "unsw"))
    assert not (tmp_path / "missing").exists()
