"""Synthetic orchestration: six fits, no refit, global freeze and offline recovery."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ids.common import canonical_hash, read_json, write_json
from ids.model_training_redesign import runner
from ids.model_training_redesign.protocol import specifications


def rows(role, offset):
    frame = pd.DataFrame({name: [1.0, 2.0] for name in runner.data.PREDICTORS})
    frame["duration_seconds"] = [offset, offset + 0.5]
    frame["ip_protocol"] = [6, 17]
    frame["source_id"] = role
    frame["row_id"] = [1, 2]
    frame["label"] = [0, 0] if role == "C" else [0, 1]
    frame["profile"] = "A"
    return frame


def paired(seed, steps=(139,)):
    result = {key: f"same-{seed}" for key in (
        "preprocessing_sha256", "numeric_bins_sha256", "initial_state_sha256", "F_identity_sha256",
        "initialization_seed", "dropout_seed", "sampler_seeds",
        "initial_dropout_rng_sha256", "initial_sampler_rng_sha256")}
    for key in ("draw_chain_by_validation_step", "dropout_rng_by_validation_step"):
        result[key] = {str(step): f"same-{seed}-{step}" for step in steps}
    return result


class FakeBackend:
    def __init__(self, root, events):
        self.root, self.events = root, events
        self.fits, self.loads, self.fail_next = 0, 0, False

    def fit_trajectory(self, f, v, output, **kwargs):
        assert set(f.source_id) == {"F"} and set(v.source_id) == {"V"}
        output = Path(output)
        if output.exists() and any(output.iterdir()):
            raise ValueError("Nonempty unfinished fit cannot be retrained")
        output.mkdir(parents=True, exist_ok=True)
        self.fits += 1
        self.events.append(("fit", output.parent.name))
        write_json(output / "attempt.json", {"synthetic": True})
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("synthetic interrupted fit")
        payload = output.parent.name.encode()
        (output / "model.pt").write_bytes(payload)
        (output / "best.pt").write_bytes(payload)
        write_json(output / "pairing.json", paired(kwargs["seed"]))
        summary = {"status": "complete", "fit_calls": 1, "final_refits": 0,
                   "selected_step": 139, "optimizer_steps": 139}
        write_json(output / "summary.json", summary)
        return summary

    def load_predictor(self, artifact, *, threads, device):
        assert threads == 1 and device == "cpu"
        frozen = read_json(self.root / "freeze.json")
        assert len(frozen["artifacts"]) == self.fits == 6
        assert (self.root / "receipts/freeze.json").exists()
        self.loads += 1
        name = artifact.read_text()

        def score(frame):
            assert list(frame.columns) == runner.data.PREDICTORS
            assert "label" not in frame and "profile" not in frame
            self.events.append(("score", name))
            return frame.duration_seconds.to_numpy(dtype=np.float64)

        return SimpleNamespace(score=score)


@pytest.fixture
def case(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    registration, design = specifications()
    design["roles"]["C"]["normal"] = 2
    protocol = {"version": runner.VERSION, "registration": registration, "design": design,
                "tasks": runner.tasks(design), "code": {}, "threads": 1, "device": "cpu"}
    write_json(root / "protocol.json", protocol)

    def inputs():
        folder = root / "inputs"
        folder.mkdir()
        for role, offset in (("F", 1), ("V", 2), ("C", 3), ("ED", 4)):
            rows(role, offset).to_parquet(folder / f"{role}.parquet", index=False)
        return runner.files(folder)

    def sources():
        path = root / "source_snapshot/fake.py"
        path.parent.mkdir()
        path.write_text("# fake source\n")
        return [path]

    key = canonical_hash(protocol)
    runner.checked_stage(root, "inputs", key, inputs)
    runner.checked_stage(root, "sources", key, sources)
    events = []
    backend = FakeBackend(root, events)
    monkeypatch.setattr(runner, "verify_live_code", lambda protocol: None)
    monkeypatch.setattr(runner, "verify_runtime", lambda *args: {"synthetic": True})
    monkeypatch.setattr(runner, "initialize", lambda *args: protocol)
    monkeypatch.setattr(runner, "own_windows_workers", lambda: None)
    monkeypatch.setattr(runner, "ResourceLog", lambda root: nullcontext(SimpleNamespace(activity="")))

    def launch(root, mode, task_id):
        if mode == "worker":
            runner.train_task(root, task_id, backend)
        else:
            assert sum(x[0] == "score" for x in events) == 12
            write_json(root / f"cost/{task_id}.json", {"synthetic": True})
            events.append(("cost", task_id))

    original_score = runner.score_task
    monkeypatch.setattr(runner, "launch", launch)
    monkeypatch.setattr(runner, "score_task", lambda root, protocol, task:
                        original_score(root, protocol, task, backend))

    def report(root, protocol):
        assert backend.fits == 6 and backend.loads == 6
        path = root / "reports/comparison.json"
        write_json(path, {"synthetic": True, "adoption_pass": False})
        events.append(("report",))
        return [path]

    monkeypatch.setattr(runner, "report", report)
    original_read = pd.read_parquet

    def guarded(path, *args, **kwargs):
        if Path(path).name in {"C.parquet", "ED.parquet"}:
            assert backend.fits == 6 and (root / "receipts/freeze.json").exists()
        assert Path(path).name != "P.parquet"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded)
    return SimpleNamespace(root=root, protocol=protocol, backend=backend, events=events,
                           r29=tmp_path / "r29", raw=tmp_path / "raw")


def complete(c):
    return runner.run(c.root, r29=c.r29, raw_dir=c.raw, threads=1, device="cpu")


def test_six_fits_freeze_before_score_and_complete_resume_has_zero_new_work(case):
    assert complete(case)["status"] == "complete"
    assert case.backend.fits == case.backend.loads == 6
    assert [e[0] for e in case.events[:6]] == ["fit"] * 6
    assert sum(e[0] == "score" for e in case.events) == 12
    frozen = read_json(case.root / "freeze.json")
    assert frozen["actual_fit_calls"] == 6 and frozen["final_refits"] == 0
    assert len(frozen["paired_contracts"]) == 3
    before = list(case.events)
    assert complete(case)["status"] == "complete"
    assert case.events == before
    assert runner.verify(case.root)["model_forwards"] == 0
    assert Path(runner.pack(case.root)["archive"]).is_file()


def test_unfinished_fit_is_not_automatically_retried(case):
    case.backend.fail_next = True
    with pytest.raises(RuntimeError, match="interrupted"):
        complete(case)
    assert case.backend.fits == 1
    assert read_json(case.root / "failure_integrity.json")["completed_experiment_verified"] is False
    assert (case.root / f"{runner.VERSION}-failed-evidence.zip").is_file()
    with pytest.raises(ValueError, match="Nonempty"):
        complete(case)
    assert case.backend.fits == 1


def test_completed_receipt_tamper_is_rejected_before_any_additional_fit(case):
    runner.train_task(case.root, case.protocol["tasks"][0]["id"], case.backend)
    artifact = case.root / f"models/{case.protocol['tasks'][0]['id']}/trajectory/model.pt"
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        runner.train_task(case.root, case.protocol["tasks"][0]["id"], case.backend)
    assert case.backend.fits == 1


def test_freeze_rejects_replaced_selected_checkpoint(case):
    runner.train_all(case.root, case.protocol)
    folder = case.root / f"models/{case.protocol['tasks'][0]['id']}/trajectory"
    (folder / "best.pt").write_bytes(b"different")
    with pytest.raises(ValueError):
        runner.freeze(case.root, case.protocol)
    assert case.backend.loads == 0


def test_pairing_compares_shared_prefix_not_total_early_stopping_length():
    assert runner.check_pairing(paired(17, (139, 278)), paired(17))["common_validation_steps"] == [139]
    other = paired(17)
    other["draw_chain_by_validation_step"]["139"] = "changed"
    with pytest.raises(ValueError, match="random stream"):
        runner.check_pairing(paired(17), other)
    with pytest.raises(ValueError, match="initialization"):
        runner.check_pairing(paired(17), paired(29))


def test_registered_plan_has_exact_fixed_budget_and_preserves_design_only_document():
    registration, design = specifications()
    assert registration["execution_enabled"] and not design["execution_enabled"]
    assert len(runner.tasks(design)) == 6
    assert registration["final_refits"] == 0
    design["training"]["seeds"].append(99)
    with pytest.raises(ValueError, match="three paired"):
        runner.tasks(design)
