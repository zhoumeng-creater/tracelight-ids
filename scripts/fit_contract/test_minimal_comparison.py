"""Synthetic-only tests. Never read a project data table."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


SCRIPT = Path(__file__).with_name("run_minimal_comparison.py")
SPEC = importlib.util.spec_from_file_location("minimal_comparison", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def row(fid, role, label, value, missing=False):
    current = [value] * 8
    if missing:
        current[3] = None
    return {"flow_id": fid, "stable_flow_id": fid, "dataset_role": role,
            "label": label, "features8": current, "features27": current + [0.0] * 19,
            "clock_basis": "synthetic_export", "replay_export_time": 123.0}


def synthetic_rows():
    fit = [row(f"F{i}", "F", i % 2, float(i % 2), missing=i == 0) for i in range(64)]
    calibration = [row(f"C{i}", "C", 0, 0.0, missing=i == 0) for i in range(25)]
    validation = [row(f"V{i}", "V", i % 2, 100.0 + i) for i in range(6)]
    return fit + calibration + validation


class MinimalComparisonTests(unittest.TestCase):
    def test_only_F_fits_artifacts_precede_C_native_missing_and_ties(self):
        models = synthetic_rows()
        unknown = row("candidate", "V", None, 0.5, missing=True)
        order = []
        fitted = []
        real_fit = HistGradientBoostingClassifier.fit
        real_score = HistGradientBoostingClassifier.decision_function
        real_dump = joblib.dump

        def tracked_fit(estimator, x, y, *args, **kwargs):
            self.assertTrue((out / "source_receipt.json").is_file())
            order.append("fit")
            fitted.append((x.copy(), y.copy()))
            return real_fit(estimator, x, y, *args, **kwargs)

        def tracked_score(estimator, x):
            order.append("score")
            return real_score(estimator, x)

        def tracked_dump(value, filename, *args, **kwargs):
            order.append("save")
            return real_dump(value, filename, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "result"
            with patch.object(HistGradientBoostingClassifier, "fit", tracked_fit), \
                    patch.object(HistGradientBoostingClassifier, "decision_function", tracked_score), \
                    patch.object(runner.joblib, "dump", tracked_dump):
                result = runner.compare(models, [unknown], [],
                                        [{"unit_id": "C25", "flow_ids": [f"C{i}" for i in range(25)]}],
                                        out, threads=1, source_receipt={"synthetic_only": True})
            self.assertEqual(order[:4], ["fit", "save", "fit", "save"])
            self.assertEqual(order.count("fit"), 2)
            for x, y in fitted:
                self.assertEqual(len(x), 64)
                self.assertTrue(np.isnan(x).any())
                self.assertLessEqual(np.nanmax(x), 1.0)  # V values >=100 never fitted.
                self.assertEqual(set(y), {0, 1})
            self.assertEqual(result["unique_scored_flow_ids_per_arm"], 96)
            scores = runner.read_rows(out / "scores.jsonl.gz")
            for arm in runner.ARMS:
                loaded = joblib.load(out / f"{arm}.joblib")
                self.assertEqual(loaded["estimator"].n_iter_, 200)
                self.assertEqual(loaded["parameters"]["random_state"], 20261111)
                calibration = [r for r in scores if r["arm"] == arm and r["dataset_role"] == "C"]
                self.assertEqual(len(calibration), 25)
                self.assertEqual(sum(r["prediction"] for r in calibration), 0)
                self.assertEqual(max(r["score"] for r in calibration), calibration[0]["threshold"])
                self.assertEqual(len({r["flow_id"] for r in scores if r["arm"] == arm}), 96)
            units = runner.read_rows(out / "normal_unit_results.jsonl.gz")
            self.assertTrue(all(r["FP"] == r["integer_FP_budget"] == 0 for r in units))

    def test_candidate_bounds_and_incomplete_mapping(self):
        scores = {"a": {"prediction": 0}, "b": {"prediction": 1}}
        events = [
            {"event_id": "ambiguous", "candidate_flow_ids": ["a", "b"],
             "association_mode": "one_of", "candidate_set_complete": True},
            {"event_id": "confirmed", "candidate_flow_ids": ["a", "b"],
             "association_mode": "all_confirmed", "candidate_set_complete": True},
            {"event_id": "incomplete", "candidate_flow_ids": ["b"],
             "association_mode": "one_of", "candidate_set_complete": False},
            {"event_id": "missing", "candidate_flow_ids": ["b", "absent"],
             "association_mode": "one_of", "candidate_set_complete": True},
        ]
        result = runner.event_results(events, scores)
        self.assertEqual((result[0]["lower_hit"], result[0]["upper_hit"]), (False, True))
        self.assertEqual((result[1]["lower_hit"], result[1]["upper_hit"]), (True, True))
        self.assertIsNone(result[2]["lower_hit"])
        self.assertIsNone(result[3]["upper_hit"])
        self.assertEqual(result[3]["missing_score_flow_ids"], ["absent"])

    def test_reserved_and_conflicting_aliases_rejected(self):
        models = synthetic_rows()
        with self.assertRaisesRegex(ValueError, "reserved"):
            runner.prepare(models, [row("reserved", "reserved_historical_confirmation", None, 1.0)])
        with self.assertRaisesRegex(ValueError, "conflict"):
            runner.prepare(models, [row("F0", "V", 0, 0.0)])
        _, _, merged = runner.prepare(models, [dict(models[0])])
        self.assertEqual(len(merged), len(models))

    def test_normal_unit_requires_unique_known_normal_denominator(self):
        scores = {"a": {"prediction": 1, "label": 0}, "u": {"prediction": 0, "label": None}}
        with self.assertRaisesRegex(ValueError, "non-normal"):
            runner.normal_unit_results([{"unit_id": "bad", "flow_ids": ["u"]}], scores)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            runner.normal_unit_results([{"unit_id": "dup", "flow_ids": ["a", "a"]}], scores)
        result = runner.normal_unit_results([{"unit_id": "one", "flow_ids": ["a"]}], scores)[0]
        self.assertEqual(result["FP"], 1)
        self.assertFalse(result["empirical_budget_met"])

    def test_bad_denominator_fails_before_either_fit(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(HistGradientBoostingClassifier, "fit", side_effect=AssertionError("must not fit")):
                with self.assertRaisesRegex(ValueError, "non-normal"):
                    runner.compare(synthetic_rows(), [], [],
                                   [{"unit_id": "bad", "flow_ids": ["not_observed"]}],
                                   Path(temp) / "result", threads=1)

    def test_inputs_only_never_constructs_model_or_invents_scores(self):
        with patch.object(runner, "HistGradientBoostingClassifier", side_effect=AssertionError("no model")):
            result = runner.inspect_inputs(synthetic_rows(), [],
                                           [{"event_id": "unresolved", "candidate_flow_ids": ["not_observed"],
                                             "association_mode": "one_of", "candidate_set_complete": False}],
                                           [{"unit_id": "C25", "flow_ids": [f"C{i}" for i in range(25)]}])
        self.assertEqual(result["model_constructions"], 0)
        self.assertEqual(result["score_calls"], 0)
        self.assertEqual(result["unique_merged_flow_ids"], 95)
        self.assertFalse(result["metadata"]["all_event_references_resolve"])
        self.assertNotIn("thresholds", result)

    def test_paired_changes_preserve_TP_FP_and_unknown_identity(self):
        before, after = {}, {}
        for fid, label, pred8, pred27 in [("gain", 1, 0, 1), ("lost", 1, 1, 0),
                                         ("fp_add", 0, 0, 1), ("fp_remove", 0, 1, 0),
                                         ("unknown", None, 0, 1)]:
            base = {"flow_id": fid, "dataset_role": "V", "label": label,
                    "score": float(pred8), "prediction": pred8, "threshold": 0.5}
            before[fid] = base
            after[fid] = dict(base, score=float(pred27), prediction=pred27)
        pairs, metrics = runner.paired_results(before, after)
        self.assertEqual({p["flow_id"] for p in pairs}, set(before))
        for key in ["TP_added", "TP_lost", "FP_added", "FP_removed", "unknown_added_alerts"]:
            self.assertEqual(metrics[0][key], 1)
        self.assertEqual(metrics[0]["unknown_removed_alerts"], 0)


if __name__ == "__main__":
    unittest.main()
