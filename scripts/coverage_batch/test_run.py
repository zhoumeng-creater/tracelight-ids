"""Targeted synthetic checks for the fixed coverage x ports batch.

All rows are invented below. ToyEstimator records calls and never trains a
statistical model; no project dataset is loaded by this test module.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SPEC = importlib.util.spec_from_file_location("coverage_batch_under_test", Path(__file__).with_name("run.py"))
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)
ARTIFACTS = None


def fixture():
    def row(fid, marker, label, role, group):
        return {"flow_id": fid, "stable_flow_id": "stable:" + fid,
                "features8": [marker, 4.0, 5.0, 1.0, 2.0, 3.0, 6.0, 7.0],
                "label": label, "dataset_role": role, "allocation_group": group,
                "target_kind": "control" if label == 1 else None}
    data = [row("base-normal", 1.0, 0, "F_base", "synthetic-santos-complete"),
            row("base-control", 2.0, 1, "F_base", "synthetic-santos-complete"),
            row("added-normal", 3.0, 0, "F_expansion", "synthetic-unraveled-complete"),
            row("added-control", 4.0, 1, "F_expansion", "synthetic-unraveled-complete"),
            row("C0", 10.0, 0, "C", "synthetic-russell-complete"),
            row("C1", 11.0, 0, "C", "synthetic-russell-complete"),
            row("C2", 12.0, 0, "C", "synthetic-russell-complete"),
            row("V-normal", 20.0, 0, "V", "synthetic-fox-complete"),
            row("V-control", 21.0, 1, "V", "synthetic-fox-complete"),
            row("external-normal", 30.0, 0, "external_development", "synthetic-optc-complete"),
            row("external-control", 31.0, 1, "external_development", "synthetic-optc-complete")]
    cohorts = {"F_base": ["base-normal", "base-control"],
               "F_expanded": ["base-normal", "base-control", "added-normal", "added-control"],
               "C": ["C0", "C1", "C2"], "V": ["V-normal", "V-control"],
               "external_development": ["external-normal", "external-control"]}
    events = [{"event_id": "synthetic-action", "label": 1, "kind": "control", "role": "V",
               "candidate_flow_ids": ["V-normal", "V-control"], "association_mode": "one_of",
               "candidate_set_complete": True}]
    units = [{"unit_id": "synthetic-normal-V", "flow_ids": ["V-normal"]}]
    return data, cohorts, events, units


class ToyEstimator:
    calls = []
    instances = []
    total_fit_calls = 0
    total_score_calls = 0
    output = None
    calibration_markers = {10.0, 11.0, 12.0}

    def __init__(self, **parameters):
        self.parameters = parameters
        self.ordinal = len(self.instances)
        self.instances.append(self)
        self.calls.append({"call": "construct", "ordinal": self.ordinal, "parameters": parameters})

    def fit(self, x, y):
        type(self).total_fit_calls += 1
        self.fit_x = x.copy()
        self.fit_y = y.copy()
        self.calls.append({"call": "fit", "ordinal": self.ordinal,
                           "markers": x[:, 0].tolist(), "labels": y.tolist()})
        return self

    def decision_function(self, x):
        type(self).total_score_calls += 1
        phase = "C" if set(x[:, 0]) == self.calibration_markers else "remaining"
        fits = [r for r in self.calls if r["call"] == "fit"]
        assert len(fits) == 4, "A score was computed before all four fits"
        models = list(self.output.glob("*.joblib"))
        receipts = list(self.output.glob("*.fit_receipt.json"))
        assert len(models) == len(receipts) == 4, "A score preceded model persistence"
        if phase == "remaining":
            thresholds = self.output / "thresholds.json"
            assert thresholds.exists(), "Evaluation preceded saved C thresholds"
            saved = json.loads(thresholds.read_text(encoding="utf-8"))
            assert set(saved) == set(runner.ARMS), "Some fixed thresholds were not saved"
            assert sum(r["call"] == "score" and r["phase"] == "C" for r in self.calls) == 4
        self.calls.append({"call": "score", "ordinal": self.ordinal, "phase": phase,
                           "markers": x[:, 0].tolist(), "all_four_models_and_fit_receipts_exist": True,
                           "all_thresholds_saved": (self.output / "thresholds.json").exists()})
        return x[:, 0] / 100.0 + self.ordinal / 1000.0


class CoverageBatchChecks(unittest.TestCase):
    def test_01_inspection_constructs_no_model(self):
        with patch.object(runner, "HistGradientBoostingClassifier", side_effect=AssertionError("Real model construction")):
            result = runner.inspect(*fixture())
        self.assertEqual(result["fit_calls"], 0)
        self.assertEqual(result["score_calls"], 0)
        self.assertEqual(result["planned_fits"], 4)

    def test_02_fit_and_evaluation_identity_overlap_rejected(self):
        data, cohorts, events, units = fixture()
        cohorts["V"].append("base-normal")
        with self.assertRaisesRegex(ValueError, "overlap"):
            runner.inspect(data, cohorts, events, units)

    def test_03_complete_source_group_leak_rejected(self):
        data, cohorts, events, units = fixture()
        data[4]["allocation_group"] = data[0]["allocation_group"]
        with self.assertRaisesRegex(ValueError, "group crosses"):
            runner.inspect(data, cohorts, events, units)

    def test_04_F_base_must_be_a_true_subset(self):
        for expanded in (["base-normal", "base-control"], ["added-normal", "added-control"]):
            with self.subTest(expanded=expanded):
                data, cohorts, events, units = fixture()
                cohorts["F_expanded"] = expanded
                with self.assertRaisesRegex(ValueError, "strictly include"):
                    runner.inspect(data, cohorts, events, units)

    def test_05_reserved_and_unregistered_score_rows_rejected(self):
        for role, group in (("reserved_confirmation", "synthetic-held-out"),
                            ("V", "synthetic-fox-complete"),
                            ("F_expansion", "synthetic-unregistered-source")):
            with self.subTest(role=role, group=group):
                data, cohorts, events, units = fixture()
                extra = dict(data[0], flow_id="unregistered", stable_flow_id="stable:unregistered",
                             dataset_role=role, allocation_group=group, label=None)
                data.append(extra)
                with self.assertRaisesRegex(ValueError, "unregistered|reserved|outside"):
                    runner.inspect(data, cohorts, events, units)

    def test_06_unknown_background_in_registered_fit_group_is_not_fitted(self):
        data, cohorts, events, units = fixture()
        data.append(dict(data[0], flow_id="background", stable_flow_id="stable:background", label=None))
        result = runner.inspect(data, cohorts, events, units)
        self.assertEqual(result["unique_rows"], 12)
        self.assertEqual(result["cohorts"]["F_expanded"]["rows"], 4)

    def test_07_explicit_fit_classes_and_normal_only_C(self):
        for change in ("one_fit_class", "non_normal_C"):
            with self.subTest(change=change):
                data, cohorts, events, units = fixture()
                data[1 if change == "one_fit_class" else 4]["label"] = 0 if change == "one_fit_class" else 1
                with self.assertRaisesRegex(ValueError, "requires explicit|non-normal"):
                    runner.inspect(data, cohorts, events, units)

    def test_08_ports_only_and_missingness_are_preserved(self):
        data = [{"features8": [6.0, None, 8.0, None, 1.0, 2.0, 3.0, 4.0]},
                {"features8": [17.0, 9.0, None, 1.0, None, 2.0, 0.0, 3.0]}]
        original = copy.deepcopy(data)
        raw = runner.features(data, False)
        zero = runner.features(data, True)
        np.testing.assert_array_equal(np.isnan(raw), np.isnan(zero))
        np.testing.assert_allclose(raw[:, [0, 3, 4, 5, 6, 7]], zero[:, [0, 3, 4, 5, 6, 7]], equal_nan=True)
        self.assertEqual(zero[0, 2], 0.0)
        self.assertEqual(zero[1, 1], 0.0)
        self.assertTrue(np.isnan(zero[0, 1]) and np.isnan(zero[1, 2]))
        self.assertEqual(data, original)

    def test_09_integer_budget_small_C_and_strict_threshold(self):
        for n, expected_budget, expected_threshold in ((1, 0, 0), (99, 0, 98), (100, 1, 98),
                                                       (101, 1, 99), (199, 1, 197), (200, 2, 197)):
            with self.subTest(n=n):
                scores = np.arange(n, dtype=float)
                r = runner.calibration_threshold(scores)
                self.assertEqual(r["integer_FP_budget"], expected_budget)
                self.assertEqual(r["threshold"], expected_threshold)
                self.assertEqual(r["C_empirical_FP"], expected_budget)
                self.assertEqual(r["comparison"], ">")
                self.assertFalse(r["population_guarantee"])
                self.assertFalse(float(r["threshold"]) > r["threshold"])

    def test_10_ties_are_not_split_or_forced_to_spend_budget(self):
        for scores, threshold, fp in (([1.0] * 200, 1.0, 0),
                                      ([0.0] * 197 + [1.0] * 3, 1.0, 0),
                                      ([0.0] * 198 + [1.0] * 2, 0.0, 2),
                                      ([-4.0, -3.0, -2.0], -2.0, 0)):
            with self.subTest(threshold=threshold, fp=fp):
                result = runner.calibration_threshold(scores)
                self.assertEqual(result["threshold"], threshold)
                self.assertEqual(result["C_empirical_FP"], fp)
        for invalid in ([], [0, float("nan")], [float("inf")], [[0.0, 1.0]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                runner.calibration_threshold(invalid)

    def test_11_four_fits_persist_before_C_and_thresholds_before_evaluation(self):
        data, cohorts, events, units = fixture()
        data[0]["features8"][1] = None
        ToyEstimator.calls, ToyEstimator.instances = [], []
        with tempfile.TemporaryDirectory(prefix="coverage-batch-artificial-") as temp:
            output = (ARTIFACTS / "toy_run") if ARTIFACTS is not None else Path(temp) / "toy_run"
            ToyEstimator.output = output
            with patch.object(runner, "HistGradientBoostingClassifier", side_effect=AssertionError("Real HGB forbidden")):
                result = runner.run(data, cohorts, events, units, output, 1,
                                    {"synthetic_only": True, "real_data_rows": 0}, estimator_factory=ToyEstimator)
            self.assertEqual(result["fit_calls"], 4)
            calls = ToyEstimator.calls
            phases = [r["phase"] for r in calls if r["call"] == "score"]
            self.assertEqual(phases, ["C"] * 4 + ["remaining"] * 4)
            self.assertEqual(len(list(output.glob("*.joblib"))), 4)
            for ordinal, (arm, (cohort, zero_ports)) in enumerate(runner.ARMS.items()):
                estimator = ToyEstimator.instances[ordinal]
                selected = [r for r in data if r["flow_id"] in cohorts[cohort]]
                np.testing.assert_allclose(estimator.fit_x, runner.features(selected, zero_ports), equal_nan=True)
                self.assertEqual(estimator.parameters["class_weight"], "balanced")
                self.assertEqual(estimator.parameters["max_iter"], 200)
                self.assertIs(estimator.parameters["early_stopping"], False)
                self.assertNotIn("sample_weight", estimator.parameters)
                scores = runner.report.read_rows(output / f"{arm}.scores.jsonl.gz")
                self.assertEqual({r["flow_id"] for r in scores}, {r["flow_id"] for r in data})
                self.assertEqual(len(scores), len(data))
                self.assertTrue(all(r["prediction"] == int(r["score"] > r["threshold"]) for r in scores))
            for name in runner.COMPARISONS:
                paired = runner.report.read_rows(output / f"{name}.paired.jsonl.gz")
                self.assertEqual(len(paired), len(data))
            if ARTIFACTS is not None:
                runner.report.save_json(ARTIFACTS / "toy_calls.json", {"real_estimator_fit_calls": 0, "calls": calls})

    def test_12_pairwise_added_and_removed_conservation(self):
        labels, before, after = [1, 1, 0, 0, None, 0], [0, 1, 0, 1, 0, 0], [1, 0, 1, 0, 1, 0]
        def make(predictions):
            return {str(i): {"flow_id": str(i), "dataset_role": "V", "label": label,
                             "prediction": p, "score": float(p), "threshold": 0.5}
                    for i, (label, p) in enumerate(zip(labels, predictions))}
        a, b = make(before), make(after)
        changes, summary = runner.report.paired_results(a, b)
        self.assertEqual(len(changes), 6)
        self.assertEqual(summary, [{"role": "V", "rows": 6, "TP_added": 1, "TP_lost": 1,
                                    "FP_added": 1, "FP_removed": 1, "unknown_added_alerts": 1,
                                    "unknown_removed_alerts": 0}])
        before_metrics, after_metrics = runner.report.role_results(a)[0], runner.report.role_results(b)[0]
        self.assertEqual(after_metrics["TP"] - before_metrics["TP"], summary[0]["TP_added"] - summary[0]["TP_lost"])
        self.assertEqual(after_metrics["FP"] - before_metrics["FP"], summary[0]["FP_added"] - summary[0]["FP_removed"])
        with self.assertRaises(ValueError):
            runner.report.paired_results(a, {k: v for k, v in b.items() if k != "0"})

    def test_13_multiple_candidate_event_bounds_and_unresolved_cases(self):
        scores = {"yes": {"prediction": 1}, "no": {"prediction": 0}}
        cases = [("one_of", True, ["yes", "no", "yes"], False, True),
                 ("all_confirmed", True, ["yes", "no"], True, True),
                 ("one_of", True, ["no"], False, False),
                 ("one_of", True, ["yes"], True, True),
                 ("one_of", False, ["yes"], None, None),
                 ("one_of", True, [], None, None),
                 ("one_of", True, ["yes", "missing"], None, None)]
        for i, (mode, complete, candidates, lower, upper) in enumerate(cases):
            with self.subTest(case=i):
                event = {"event_id": str(i), "candidate_flow_ids": candidates,
                         "association_mode": mode, "candidate_set_complete": complete}
                r = runner.report.event_results([event], scores)[0]
                self.assertIs(r["lower_hit"], lower)
                self.assertIs(r["upper_hit"], upper)
                self.assertEqual(len(r["candidate_flow_ids"]), len(set(candidates)))
                self.assertEqual(r["status"], "unresolved" if lower is None else "resolved_bounds")

    def test_14_normal_unit_integer_budget_and_unknown_rejection(self):
        for n, fp, expected in ((99, 1, False), (100, 1, True), (200, 3, False)):
            with self.subTest(n=n):
                scores = {str(i): {"label": 0, "prediction": int(i < fp)} for i in range(n)}
                unit = {"unit_id": "normal", "flow_ids": list(scores)}
                r = runner.report.normal_unit_results([unit], scores)[0]
                self.assertEqual(r["normal_count"], n)
                self.assertEqual(r["FP"], fp)
                self.assertIs(r["empirical_budget_met"], expected)
                scores["0"]["label"] = None
                with self.assertRaises(ValueError):
                    runner.report.normal_unit_results([unit], scores)

    def test_15_C_background_is_scored_but_not_calibrated_or_counted_normal(self):
        data, cohorts, events, units = fixture()
        background = dict(data[4], flow_id="C-background-unknown", stable_flow_id="stable:C-background-unknown",
                          dataset_role="C_background", label=None, features8=[99.0, 4.0, 5.0, 1.0, 2.0, 3.0, 6.0, 7.0])
        data.append(background)
        events.append({"event_id": "synthetic-C-ambiguous", "label": 0, "kind": "normal_file",
                       "candidate_flow_ids": ["C0", background["flow_id"]],
                       "association_mode": "one_of", "candidate_set_complete": True})
        units.append({"unit_id": "synthetic-normal-C", "flow_ids": list(cohorts["C"])})
        inspection = runner.inspect(data, cohorts, events, units)
        self.assertEqual(inspection["cohorts"]["C"]["rows"], 3)
        bad_group = copy.deepcopy(data)
        bad_group[-1]["allocation_group"] = "synthetic-unregistered-C-source"
        with self.assertRaisesRegex(ValueError, "unregistered|reserved"):
            runner.inspect(bad_group, cohorts, events, units)
        bad_units = units + [{"unit_id": "unknown-is-not-normal", "flow_ids": [background["flow_id"]]}]
        with self.assertRaisesRegex(ValueError, "non-normal"):
            runner.inspect(data, cohorts, events, bad_units)
        ToyEstimator.calls, ToyEstimator.instances = [], []
        with tempfile.TemporaryDirectory(prefix="coverage-batch-C-background-") as temp:
            output = (ARTIFACTS / "toy_run_C_background") if ARTIFACTS is not None else Path(temp) / "toy_run"
            ToyEstimator.output = output
            with patch.object(runner, "HistGradientBoostingClassifier", side_effect=AssertionError("Real HGB forbidden")):
                result = runner.run(data, cohorts, events, units, output, 1,
                                    {"synthetic_only": True, "real_data_rows": 0}, estimator_factory=ToyEstimator)
            for arm in runner.ARMS:
                self.assertEqual(result["thresholds"][arm]["C_count"], 3)
                self.assertEqual(result["thresholds"][arm]["integer_FP_budget"], 0)
                scored = {r["flow_id"]: r for r in runner.report.read_rows(output / f"{arm}.scores.jsonl.gz")}
                unknown = scored[background["flow_id"]]
                self.assertIsNone(unknown["label"])
                self.assertEqual(unknown["prediction"], 1)
                reported_units = runner.report.read_rows(output / f"{arm}.normal_units.jsonl.gz")
                c_unit = next(r for r in reported_units if r["unit_id"] == "synthetic-normal-C")
                self.assertEqual(c_unit["normal_count"], 3)
                self.assertEqual(c_unit["FP"], 0)
                reported_events = runner.report.read_rows(output / f"{arm}.events.jsonl.gz")
                bounds = next(r for r in reported_events if r["event_id"] == "synthetic-C-ambiguous")
                self.assertEqual(bounds["candidate_flow_ids"], ["C0", background["flow_id"]])
                self.assertIs(bounds["lower_hit"], False)
                self.assertIs(bounds["upper_hit"], True)
            calibration_calls = [c for c in ToyEstimator.calls if c["call"] == "score" and c["phase"] == "C"]
            self.assertEqual(len(calibration_calls), 4)
            self.assertTrue(all(set(c["markers"]) == {10.0, 11.0, 12.0} for c in calibration_calls))
            if ARTIFACTS is not None:
                runner.report.save_json(ARTIFACTS / "toy_C_background_calls.json", {"real_estimator_fit_calls": 0, "calls": ToyEstimator.calls})

    def test_16_source_metadata_and_arm_specific_fit_markers_are_preserved(self):
        data, cohorts, events, units = fixture()
        mixed_arms = [arm for arm, (cohort, _) in runner.ARMS.items() if cohort == "F_expanded"]
        event = {"event_id": "synthetic-unraveled-action", "label": 1, "kind": "control",
                 "source": "Unraveled", "role": "F_expansion",
                 "candidate_flow_ids": ["added-control"], "association_mode": "all_confirmed",
                 "candidate_set_complete": True}
        unit = {"unit_id": "synthetic-unraveled-normal", "source": "Unraveled",
                "role": "F_expansion", "denominator_scope": "author_labeled_normal_whole_connections",
                "flow_ids": ["added-normal"], "in_sample_for_arms": mixed_arms}
        events.append(event)
        units.append(unit)
        ToyEstimator.calls, ToyEstimator.instances = [], []
        with tempfile.TemporaryDirectory(prefix="coverage-batch-source-metadata-") as temp:
            output = (ARTIFACTS / "toy_run_source_metadata") if ARTIFACTS is not None else Path(temp) / "toy_run"
            ToyEstimator.output = output
            with patch.object(runner, "HistGradientBoostingClassifier", side_effect=AssertionError("Real HGB forbidden")):
                runner.run(data, cohorts, events, units, output, 1,
                           {"synthetic_only": True, "real_data_rows": 0}, estimator_factory=ToyEstimator)
            for arm in runner.ARMS:
                with self.subTest(arm=arm):
                    reported_units = runner.report.read_rows(output / f"{arm}.normal_units.jsonl.gz")
                    reported_unit = next(r for r in reported_units if r["unit_id"] == unit["unit_id"])
                    for field in ("source", "role", "denominator_scope", "in_sample_for_arms"):
                        self.assertEqual(reported_unit[field], unit[field])
                    self.assertIs(reported_unit["in_sample_for_this_arm"], arm in mixed_arms)
                    self.assertEqual(reported_unit["normal_count"], 1)
                    self.assertEqual(reported_unit["FP"], 0)
                    reported_events = runner.report.read_rows(output / f"{arm}.events.jsonl.gz")
                    reported_event = next(r for r in reported_events if r["event_id"] == event["event_id"])
                    for field in ("source", "role", "candidate_flow_ids"):
                        self.assertEqual(reported_event[field], event[field])
                    self.assertIs(reported_event["source_used_for_fitting_this_arm"], arm in mixed_arms)
                    self.assertIs(reported_event["lower_hit"], False)
                    self.assertIs(reported_event["upper_hit"], False)
            if ARTIFACTS is not None:
                runner.report.save_json(ARTIFACTS / "toy_source_metadata_calls.json",
                                        {"real_estimator_fit_calls": 0, "calls": ToyEstimator.calls})


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outcomes = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.outcomes.append({"test": test.id(), "status": "passed"})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.outcomes.append({"test": test.id(), "status": "failed", "detail": self._exc_info_to_string(err, test)})

    def addError(self, test, err):
        super().addError(test, err)
        self.outcomes.append({"test": test.id(), "status": "error", "detail": self._exc_info_to_string(err, test)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path)
    args = parser.parse_args()
    if args.results_dir:
        ARTIFACTS = args.results_dir / datetime.now(timezone.utc).strftime("synthetic_%Y%m%dT%H%M%S%fZ")
        ARTIFACTS.mkdir(parents=True, exist_ok=False)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CoverageBatchChecks)
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordedResult).run(suite)
    if ARTIFACTS:
        summary = {"status": "passed" if result.wasSuccessful() else "failed", "tests_run": result.testsRun,
                   "failure_count": len(result.failures), "error_count": len(result.errors),
                   "outcomes": result.outcomes, "input_scope": "Only artificial rows declared in this module",
                   "real_data_rows_read": 0, "real_model_constructions": 0, "real_model_fits": 0,
                   "toy_fit_calls": ToyEstimator.total_fit_calls,
                   "toy_score_calls": ToyEstimator.total_score_calls,
                   "runner_sha256": runner.report.sha(Path(__file__).with_name("run.py")),
                   "report_helper_sha256": runner.report.sha(runner.LEGACY),
                   "test_sha256": runner.report.sha(Path(__file__)), "artifacts": str(ARTIFACTS.resolve())}
        runner.report.save_json(ARTIFACTS / "results.json", summary)
        runner.report.save_json(args.results_dir / "latest.json", summary)
        print(json.dumps({"results": str((ARTIFACTS / "results.json").resolve()), "status": summary["status"]}))
    sys.exit(not result.wasSuccessful())
