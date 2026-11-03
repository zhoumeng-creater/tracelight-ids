"""Lab launcher contracts. No real prepared partitions are opened."""
import subprocess

import pytest

from workflows.stage5 import core, reporting, run


def test_protocol_pins_four_groups_and_signed_comparisons():
    policy = run.load_protocol()
    assert policy["profile"] == "research" and policy["seeds"] == [17, 29, 43]
    assert [(g["dataset"], g["run_prefix"], g["n_features"]) for g in policy["groups"]] == [
        ("unsw", "05-mlp", 42), ("unsw", "00-rf", 42),
        ("cicids", "04-mlp", 24), ("cicids", "03-mlp", 82),
    ]
    assert [(c["candidate"], c["baseline"]) for c in policy["comparisons"]] == [
        ("unsw_mlp_all42", "unsw_rf_all42"), ("cicids_mlp_mi24", "cicids_mlp_all82"),
    ]


def test_release_rejects_dirty_or_unpublished_evaluator(monkeypatch):
    state = {"status": "", "published": True}

    def git(command, **_):
        action = command[3]
        if action == "remote":
            return "https://github.com/zhoumeng-creater/tracelight-ids.git"
        if action == "rev-parse":
            return "a" * 40
        if action == "merge-base" and not state["published"]:
            raise subprocess.CalledProcessError(1, command)
        if action == "status":
            return state["status"]
        return ""

    monkeypatch.setattr(run.subprocess, "check_output", git)
    assert run.check_release(run.REPO) == "a" * 40
    state["status"] = " M workflows/stage5/protocol.json"
    with pytest.raises(ValueError, match="Local Stage 5"):
        run.check_release(run.REPO)
    state.update(status="", published=False)
    with pytest.raises(ValueError, match="Cannot verify"):
        run.check_release(run.REPO)


@pytest.fixture
def lab(tmp_path, monkeypatch):
    stage4, unsw, output = (tmp_path / name for name in ("stage4", "unsw", "stage5"))
    for record in (stage4 / "experiment/STAGE4_COMPLETE.json",
                   stage4 / "data/cicids-prepared/manifest.json", unsw / "manifest.json"):
        record.parent.mkdir(parents=True)
        record.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(run, "check_release", lambda repo: "a" * 40)
    monkeypatch.setattr(run, "check_dependencies", lambda repo: None)
    args = ["--stage4-work-dir", str(stage4), "--unsw-data-dir", str(unsw),
            "--output-dir", str(output)]
    return args, stage4, unsw, output


def test_launcher_routes_original_inputs_and_packages_only_after_completion(lab, monkeypatch):
    args, stage4, unsw, output = lab
    calls = []

    def evaluate(experiment, data_dirs, directory, policy, *, resume):
        assert experiment == stage4 / "experiment"
        assert data_dirs == {"unsw": unsw, "cicids": stage4 / "data/cicids-prepared"}
        assert directory == output and policy == run.load_protocol() and resume
        calls.append("evaluated")

    def finalize(directory, policy):
        assert calls == ["evaluated"] and directory == output and policy == run.load_protocol()
        calls.append("packaged")
        return directory / "TraceLight-Stage5.zip"

    monkeypatch.setattr(core, "run_evaluation", evaluate)
    monkeypatch.setattr(reporting, "finalize", finalize)
    run.main([*args, "--resume"])
    assert calls == ["evaluated", "packaged"]

    def fail(*_, **__):
        raise ValueError("incomplete threshold matrix")

    monkeypatch.setattr(core, "run_evaluation", fail)
    with pytest.raises(ValueError, match="incomplete threshold"):
        run.main(args)
    assert calls == ["evaluated", "packaged"]


def test_formal_launcher_rejects_external_config_and_stage4_output(lab):
    args, stage4, *_ = lab
    with pytest.raises(SystemExit):
        run.main([*args, "--config", "smoke.json"])
    with pytest.raises(ValueError, match="separate"):
        run.main([*args[:-1], str(stage4 / "experiment/extra")])
