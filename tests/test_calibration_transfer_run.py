"""Synthetic phase barriers for the one-run lab package; no model construction."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load("transfer_run_under_test", REPO / "scripts/calibration_transfer/run.py")
evaluation_fixture = _load("transfer_evaluation_fixture", Path(__file__).with_name("test_calibration_transfer_evaluation.py"))


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


class SyntheticBatch:
    """Injected fitted artifacts are plain text; scores are predefined numbers."""
    def __init__(self, output):
        self.output = output
        self.rows, self.cohorts, self.events, self.units, self.scores, _ = evaluation_fixture.fixture()
        self.calls = []
        self.fit_failure_at = None
        self.pair_mismatch_at = None
        self.pair_mismatch_key = "sampler_draw_sha256"
        self.invalid_score_mode = None
        self.mutate_before_calibration = False
        self.mutate_before_development = False

    def fit(self, rows, preprocessor, directory, *, arm, seed, training_spec, device, threads):
        run = f"{arm}_s{seed}"
        assert not any(call[0] in {"calibration_load", "score", "evaluation_load"} for call in self.calls)
        assert rows == [row for row in self.rows if row["flow_id"] in self.cohorts["F_expanded"]]
        assert preprocessor is None
        assert training_spec == runner.TRAINING
        self.calls.append(("fit", run))
        if run == self.fit_failure_at:
            raise RuntimeError("synthetic fit failure")
        directory.mkdir(parents=True)
        (directory / "model.pt").write_text(f"synthetic artifact {run}", encoding="utf8")
        summary = {"status": "complete", "initial_state_sha256": f"synthetic_init_{seed}",
                   "sampler_draw_sha256": f"synthetic_sampler_{seed}",
                   "fit_seconds": 0., "synthetic_only": True}
        if run == self.pair_mismatch_at:
            summary[self.pair_mismatch_key] += "_changed"
        return summary

    def _assert_models_frozen(self):
        assert [call[1] for call in self.calls if call[0] == "fit"] == list(runner.RUNS)
        frozen = read(self.output / "models_frozen.json")
        assert set(frozen["runs"]) == set(runner.RUNS)
        assert frozen["new_fits"] == 9
        for run, item in frozen["runs"].items():
            artifact = self.output / item["path"]
            assert artifact.is_file()
            assert artifact.parent.name == run
            assert runner.data.file_digest(artifact) == item["sha256"]
        return frozen

    def calibration(self, repo):
        self._assert_models_frozen()
        assert not (self.output / "thresholds.json").exists()
        self.calls.append(("calibration_load", None))
        if self.mutate_before_calibration:
            (self.output / "fits" / runner.RUNS[0] / "model.pt").write_text("changed synthetic artifact", encoding="utf8")
        return [row for row in self.rows if row["flow_id"] in self.cohorts["C"]]

    def evaluation(self, repo):
        self._assert_models_frozen()
        thresholds = read(self.output / "thresholds.json")
        assert set(thresholds) == set(runner.RUNS)
        assert [call[1] for call in self.calls if call[0] == "score" and call[2] == "C"] == list(runner.RUNS)
        frozen = read(self.output / "thresholds_frozen.json")
        assert frozen["sha256"] == runner.data.file_digest(self.output / "thresholds.json")
        assert frozen["C_scores_sha256"] == runner.data.file_digest(self.output / "C_scores.json")
        assert frozen["threshold_calculations"] == 9
        self.calls.append(("evaluation_load", None))
        if self.mutate_before_development:
            (self.output / "fits" / runner.RUNS[0] / "model.pt").write_text("changed synthetic artifact", encoding="utf8")
        return self.rows, self.cohorts, self.events, self.units

    def predictor(self, artifact, *, device, threads):
        batch = self
        run = artifact.parent.name
        assert run in runner.RUNS

        class Predictor:
            def score(self, rows):
                ids = [row["flow_id"] for row in rows]
                phase = "C" if set(ids) == set(batch.cohorts["C"]) else "remaining"
                if phase == "C":
                    assert not any(call[0] == "evaluation_load" for call in batch.calls)
                else:
                    assert any(call[0] == "evaluation_load" for call in batch.calls)
                    assert set(read(batch.output / "thresholds.json")) == set(runner.RUNS)
                batch.calls.append(("score", run, phase))
                values = np.asarray([batch.scores[run][fid] for fid in ids], dtype=np.float64)
                if batch.invalid_score_mode == "short":
                    values = values[:-1]
                elif batch.invalid_score_mode == "long":
                    values = np.append(values, 0.)
                elif batch.invalid_score_mode == "nan":
                    values[0] = float("nan")
                return values

        return Predictor()

    def execute(self):
        fit_rows = [row for row in self.rows if row["flow_id"] in self.cohorts["F_expanded"]]
        return runner.execute(self.output, fit_rows, None,
                              {"training": runner.TRAINING, "selection": "terminal_step_15600_no_refit"},
                              fit_fn=self.fit, predictor_loader=self.predictor,
                              calibration_loader=self.calibration, evaluation_loader=self.evaluation)


def test_nine_model_then_nine_threshold_then_development_barriers(tmp_path):
    batch = SyntheticBatch(tmp_path)
    result = batch.execute()
    assert result["fit_calls"] == result["threshold_calculations"] == 9
    assert result["refits"] == result["threshold_searches"] == 0
    assert result["model_selection"] is False
    assert result["reserved_data_scored"] is False
    assert [call[0] for call in batch.calls] == ["fit"] * 9 + ["calibration_load"] + ["score"] * 9 + ["evaluation_load"] + ["score"] * 9
    assert all(call[2] == "C" for call in batch.calls if call[0] == "score" and batch.calls.index(call) < 19)
    assert (tmp_path / "evaluation/comparison.md").is_file()
    assert len(list((tmp_path / "evaluation").glob("*.scores.jsonl.gz"))) == 9
    assert set(read(tmp_path / "C_scores.json")) == set(runner.RUNS)
    runner.output_receipt(tmp_path)
    receipt = read(tmp_path / "output_receipt.json")
    assert "models_frozen.json" in receipt
    assert "evaluation/decision_summary.json" in receipt
    assert "output_receipt.json" not in receipt


def test_failed_fit_stops_before_any_C_material_or_score(tmp_path):
    batch = SyntheticBatch(tmp_path)
    batch.fit_failure_at = runner.RUNS[4]
    with pytest.raises(RuntimeError, match="synthetic fit failure"):
        batch.execute()
    assert all(call[0] == "fit" for call in batch.calls)
    assert len(batch.calls) == 5
    assert not (tmp_path / "models_frozen.json").exists()
    assert not (tmp_path / "thresholds.json").exists()


@pytest.mark.parametrize("key", ["initial_state_sha256", "sampler_draw_sha256"])
def test_paired_initialization_and_sampler_mismatch_blocks_C(tmp_path, key):
    batch = SyntheticBatch(tmp_path)
    batch.pair_mismatch_at = "C_real_s29"
    batch.pair_mismatch_key = key
    with pytest.raises(ValueError, match="Paired initialization or BCE sampling differs"):
        batch.execute()
    assert len(batch.calls) == 9
    assert all(call[0] == "fit" for call in batch.calls)
    assert not (tmp_path / "models_frozen.json").exists()


@pytest.mark.parametrize("mode", ["short", "long", "nan"])
def test_invalid_score_vector_stops_before_threshold_barrier(tmp_path, mode):
    batch = SyntheticBatch(tmp_path)
    batch.invalid_score_mode = mode
    with pytest.raises(ValueError):
        batch.execute()
    assert [call[0] for call in batch.calls] == ["fit"] * 9 + ["calibration_load", "score"]
    assert not (tmp_path / "thresholds.json").exists()
    assert not (tmp_path / "thresholds_frozen.json").exists()
    assert not (tmp_path / "evaluation").exists()


@pytest.mark.parametrize("phase", ["calibration", "development"])
def test_mutated_frozen_artifact_stops_scoring(tmp_path, phase):
    batch = SyntheticBatch(tmp_path)
    if phase == "calibration":
        batch.mutate_before_calibration = True
    else:
        batch.mutate_before_development = True
    with pytest.raises(ValueError, match=f"Frozen model changed before {phase}"):
        batch.execute()
    score_calls = [call for call in batch.calls if call[0] == "score"]
    assert len(score_calls) == (0 if phase == "calibration" else 9)
    assert not (tmp_path / "evaluation").exists()
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize("n, threshold, budget", [(1, 0., 0), (50, 49., 0), (99, 98., 0),
                                                  (100, 98., 1), (101, 99., 1), (200, 197., 2)])
def test_calibration_is_integer_order_statistic_not_interpolated_quantile(n, threshold, budget):
    values = np.arange(n, dtype=np.float64)
    result = runner.calibration_threshold(values)
    assert result["threshold"] == threshold
    assert result["integer_FP_budget"] == result["C_empirical_FP"] == budget
    assert result["comparison"] == ">"
    assert result["population_guarantee"] is False
    if n == 50:
        assert result["threshold"] != np.quantile(values, .99)


def test_equal_boundary_scores_never_split_ties():
    for values, threshold, fp in (([1.] * 200, 1., 0),
                                  ([0.] * 197 + [1.] * 3, 1., 0),
                                  ([0.] * 198 + [1.] * 2, 0., 2)):
        result = runner.calibration_threshold(values)
        assert result["threshold"] == threshold
        assert result["C_empirical_FP"] == fp
    for values in ([], [[0., 1.]], [float("nan")], [float("inf")]):
        with pytest.raises(ValueError):
            runner.calibration_threshold(values)
