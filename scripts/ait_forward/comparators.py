"""Load the 22 fixed historical comparators from the verified November 22 archive."""
from __future__ import annotations

from dataclasses import dataclass
import io
from importlib.metadata import version
import json
from pathlib import PurePosixPath

import numpy as np

from scripts.ait_forward import baseline as archive_baseline
from scripts.comprehensive_batch.matrix import ARMS, HGB_ARMS, PRIMARY_STEP, SEEDS, job_id, snapshot_id
from scripts.coverage_batch.materialize import RAW8_FIELDS


_RECEIPT = "output_receipt.json"
_RECEIPT_PIN = (1157194, "4d359b0919324bd0ca345eaea7d2d9fb8b671464a3a39f38f7473bd042a2af6e")
_IDENTITY_MEMBER = "score_identities.json"


def _fixed_model_paths():
    paths = {f"HGB_{arm}": f"fits/HGB_{arm}/model.joblib" for arm in HGB_ARMS}
    for seed in SEEDS:
        for arm in ARMS:
            job = job_id(arm, seed)
            model = snapshot_id(job, PRIMARY_STEP)
            paths[model] = f"fits/{job}/snapshots/step{PRIMARY_STEP:06d}/model.pt"
    return paths


def _selected_members(model_id, model_path):
    common = (
        model_path, f"scores/{model_id}.npz", f"evaluation/models/{model_id}/summary.json",
        f"benchmarks/{model_id}.json",
    )
    if model_id.startswith("HGB_"):
        return (*common, f"job_statuses/{model_id}.json")
    job = model_id.removesuffix(f"_step{PRIMARY_STEP}")
    diagnostic = PurePosixPath(model_path).parent / "F_diagnostics.json"
    return (*common, f"fits/{job}/summary.json", diagnostic.as_posix())


@dataclass(frozen=True)
class FrozenComparator:
    model_id: str
    family: str
    model_sha256: str
    model_bytes: bytes
    archive_member_sha256: dict[str, str]
    metadata: dict
    calibration_scores: dict[str, float]
    archived_fox_scores: dict[str, float]
    pooled_threshold: float
    original_summary: dict
    benchmark: dict
    training_history_summary: dict
    primary_training_diagnostics: dict | None

    def score(self, rows, *, device="cpu", threads=4):
        """Materialize and score one pinned model without fitting preprocessors."""
        if not rows or any(not isinstance(row.get("raw8"), dict) or set(row["raw8"]) != set(RAW8_FIELDS)
                           for row in rows):
            raise ValueError("Frozen comparators require canonical raw8 observations")
        if self.family == "hgb":
            from joblib import load
            from sklearn.ensemble import HistGradientBoostingClassifier

            if version("scikit-learn") != archive_baseline.SKLEARN_VERSION:
                raise RuntimeError(f"The frozen HGB requires scikit-learn {archive_baseline.SKLEARN_VERSION}")
            model = load(io.BytesIO(self.model_bytes))
            if type(model) is not HistGradientBoostingClassifier or model.n_features_in_ != 8:
                raise ValueError("Verified artifact does not contain the registered eight-field HGB")
            return archive_baseline.score_hgb(model, rows, zero_ports=self.metadata["ports_zero"], threads=threads)

        from scripts.comprehensive_batch.training import load_predictor

        predictor = load_predictor(io.BytesIO(self.model_bytes), device=device, threads=threads)
        scores = np.asarray(predictor.score(rows), dtype=np.float64)
        if scores.shape != (len(rows),) or not np.isfinite(scores).all():
            raise ValueError("Frozen TabM did not produce one finite scalar score per row")
        return scores

    def historical_fox_regression(self, rows, rules, *, device="cpu", threads=4):
        """Report numeric and strict-decision parity with the original opened scores."""
        if list(self.archived_fox_scores) != [row["flow_id"] for row in rows]:
            raise ValueError("Use the bound, in-order opened Fox regression rows")
        if float(rules["pooled"]["threshold"]) != self.pooled_threshold:
            raise ValueError("Pooled rule does not reproduce the original comparator threshold")
        saved = np.array(list(self.archived_fox_scores.values()), dtype=np.float64)
        actual = self.score(rows, device=device, threads=threads)
        exact = np.array_equal(saved, actual)
        if self.family == "hgb" and not exact:
            raise ValueError(f"Frozen HGB did not reproduce archive scores: {self.model_id}")
        normal = np.array([row["label"] == 0 for row in rows])
        known = np.array([row["label"] is not None for row in rows])
        targets = {
            kind: np.array([row["label"] == 1 and row["target_kind"] == kind for row in rows])
            for kind in ("control", "download")
        }
        decisions = {}
        for name, rule in rules.items():
            threshold = float(rule["threshold"])
            previous, current = saved > threshold, actual > threshold
            decisions[name] = {
                "threshold": threshold,
                "flipped_decisions": int((previous != current).sum()),
                "known_label_flips": int(((previous != current) & known).sum()),
                "actual_positive_alerts": int((current & (known & ~normal)).sum()),
                "actual_normal_alerts": int((current & normal).sum()),
                "targets": {
                    kind: {"total": int(mask.sum()), "tp": int((current & mask).sum())}
                    for kind, mask in targets.items()
                },
            }
        return {
            "model_id": self.model_id, "rows": len(rows),
            "device": "cpu" if self.family == "hgb" else device,
            "exact_scores": bool(exact),
            "max_absolute_score_difference": float(np.max(np.abs(actual - saved))),
            "rules": decisions,
        }


def load_comparators(archive, baseline):
    """Verify all model and evidence bytes before any comparator inference."""
    receipt_bytes = archive_baseline._verified_members(archive, {_RECEIPT: _RECEIPT_PIN})[_RECEIPT]
    receipt = json.loads(receipt_bytes)
    paths = _fixed_model_paths()
    selected = {key: _selected_members(key, path) for key, path in paths.items()}
    names = {name for members in selected.values() for name in members} | {_IDENTITY_MEMBER}
    pins = {name: (receipt[name]["bytes"], receipt[name]["sha256"]) for name in names}
    if pins[_IDENTITY_MEMBER][1] != baseline.archive_member_sha256[_IDENTITY_MEMBER]:
        raise ValueError("The comparator score order and frozen HGB identity order differ")
    members = archive_baseline._verified_members(archive, pins)
    order = json.loads(members[_IDENTITY_MEMBER])
    positions = {identity: index for index, identity in enumerate(order)}
    if len(positions) != len(order) or positions.keys() != baseline.known_flow_ids:
        raise ValueError("Verified comparators do not share the historical score universe")
    c_positions = [positions[fid] for fid in baseline.calibration_scores]
    fox_positions = [positions[fid] for fid in baseline.archived_fox_scores]

    comparators = {}
    for model_id, names_here in selected.items():
        model_path, score_path, summary_path, benchmark_path, fit_path, *diagnostic_path = names_here
        summary = json.loads(members[summary_path])
        metadata = summary["model"]
        family = "hgb" if model_id.startswith("HGB_") else "tabm"
        digest = pins[model_path][1]
        if (metadata["snapshot_id"] != model_id or metadata["family"] != family
                or metadata["path"] != model_path or metadata["sha256"] != digest):
            raise ValueError(f"Comparator model metadata differs: {model_id}")
        training = json.loads(members[fit_path])
        snapshots = [snapshot for snapshot in training["snapshots"] if snapshot["sha256"] == digest]
        if training["job"]["job_id"] != metadata["job_id"] or len(snapshots) != 1:
            raise ValueError(f"Comparator training summary does not bind its exact snapshot: {model_id}")
        diagnostic = None
        if diagnostic_path:
            (path,) = diagnostic_path
            if snapshots[0]["diagnostics_sha256"] != pins[path][1]:
                raise ValueError(f"Comparator primary training diagnostic hash differs: {model_id}")
            diagnostic = json.loads(members[path])
        with np.load(io.BytesIO(members[score_path]), allow_pickle=False) as saved:
            values = saved["scores"]
        if values.shape != (len(order),) or values.dtype != np.float64:
            raise ValueError(f"Comparator score array/order differs: {model_id}")
        c = dict(zip(baseline.calibration_scores, values[c_positions].tolist(), strict=True))
        fox = dict(zip(baseline.archived_fox_scores, values[fox_positions].tolist(), strict=True))
        if not all(np.isfinite(list(c.values()))) or not all(np.isfinite(list(fox.values()))):
            raise ValueError(f"Comparator C or Fox scores are not finite: {model_id}")
        pooled = sorted(c.values())[-(len(c) // 100 + 1)]
        threshold = summary["threshold"]
        if (threshold["threshold"] != pooled or threshold["comparison"] != ">"
                or threshold["C_count"] != len(c) or threshold["integer_FP_budget"] != len(c) // 100
                or sum(score > pooled for score in c.values()) != threshold["C_empirical_FP"]):
            raise ValueError(f"Comparator C rule does not reproduce its archived threshold: {model_id}")
        members_sha = {name: pins[name][1] for name in (*names_here, _IDENTITY_MEMBER)}
        members_sha[_RECEIPT] = _RECEIPT_PIN[1]
        comparators[model_id] = FrozenComparator(
            model_id, family, digest, members[model_path], members_sha, metadata, c, fox, pooled,
            summary, json.loads(members[benchmark_path]), training, diagnostic,
        )
    original = comparators[archive_baseline.MODEL_ID]
    if (original.model_sha256 != baseline.model_sha256 or original.calibration_scores != baseline.calibration_scores
            or original.archived_fox_scores != baseline.archived_fox_scores):
        raise ValueError("Primary comparator differs from the original frozen HGB evidence")
    return comparators
