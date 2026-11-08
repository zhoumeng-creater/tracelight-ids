"""Frozen source-port mechanism controls; real weights run only at the laboratory.

No fitting, selection, compression, optimizer or checkpoint download is performed.
The Python API accepts synthetic fixtures; the laboratory CLI requires development.
"""
from __future__ import annotations

import gc
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ids.common import canonical_hash, environment_snapshot, read_json, sha256_file, write_json
from .calibration import apply_threshold, register_threshold
from .csv_workflow import _code_snapshot, _row_ids
from .preprocessing import CSVPreprocessor, PROTECTED_IDS_PROFILE, labels, read_csv
from .scanning_inputs import (
    ARMS, apply_intervention, audit_input_collisions, audit_mapping,
    build_frequency_bijection, mapping_sha256,
)
from .scanning_metrics import analyze_arm, build_cohorts, empirical_percentiles
from .scanning_protocol import preflight_scanning
from .tabicl_adapter import CompactPredictor, FullCacheAdapter

DEPLOYMENT = "tracelight-scanning-mechanism-deployment-v1"
ROLES = ("support", "calibration", "evaluation")


def _score_matrix(predictor, matrix, batch_size):
    import torch

    chunks = []
    with torch.no_grad():
        for begin in range(0, len(matrix), batch_size):
            logits = predictor.predict_logits(torch.from_numpy(matrix[begin:begin + batch_size]).unsqueeze(0))
            chunks.append((logits[0, :, 1] - logits[0, :, 0]).cpu().numpy().astype(np.float64))
    result = np.concatenate(chunks)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite model scores")
    return result


def _input_identity(arm, prep_hash, mapping_hash, center, scale):
    return canonical_hash({"arm": arm, "preprocessor_sha256": prep_hash,
                           "mapping_sha256": mapping_hash if arm == ARMS[1] else None,
                           "center": center, "scale": scale, "width": 48})


def _deployment_files(directory, manifest):
    expected = {"model.pt", "threshold.json", "../shared/preprocessor.joblib"}
    if manifest["arm"] == ARMS[1]:
        expected.add("../shared/source_port_mapping.npy")
    if set(manifest["files"]) != expected:
        raise ValueError("scanning deployment file set differs")
    for name, digest in manifest["files"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"scanning deployment hash mismatch: {name}")
    registration = read_json(directory / "threshold.json")
    if registration["binding"] != {
        "model_sha256": manifest["files"]["model.pt"],
        "preprocessing_sha256": manifest["input_identity_sha256"],
        "calibration_sha256": manifest["calibration_sha256"],
    }:
        raise ValueError("scanning threshold input binding differs")
    apply_threshold([], registration)
    expected_input = _input_identity(
        manifest["arm"], manifest["files"]["../shared/preprocessor.joblib"],
        manifest["mapping_sha256"], manifest["center"], manifest["scale"])
    if expected_input != manifest["input_identity_sha256"]:
        raise ValueError("scanning intervention identity differs")
    return registration


class ScanningDetector:
    """Cold detector retains frozen preprocessing AND the arm-specific intervention."""

    @classmethod
    def load(cls, directory):
        import torch

        directory = Path(directory)
        manifest = read_json(directory / "deployment.json")
        if (manifest.get("format") != DEPLOYMENT or manifest.get("arm") not in ARMS
                or manifest.get("threads") != 1 or manifest.get("width") != 48):
            raise ValueError("unsupported scanning deployment")
        registration = _deployment_files(directory, manifest)
        torch.set_num_threads(manifest["threads"])
        obj = cls()
        obj.manifest, obj.registration = manifest, registration
        obj.preprocessor = CSVPreprocessor.load(
            directory / "../shared/preprocessor.joblib",
            expected_sha256=manifest["files"]["../shared/preprocessor.joblib"])
        obj.mapping = None
        if manifest["arm"] == ARMS[1]:
            obj.mapping = np.load(directory / "../shared/source_port_mapping.npy", allow_pickle=False)
            if mapping_sha256(obj.mapping) != manifest["mapping_sha256"]:
                raise ValueError("mapping payload hash differs")
        obj.predictor = CompactPredictor.load(directory / "model.pt", device="cpu")
        contract = obj.predictor.provenance.get("input_contract", {})
        if (obj.predictor.cache_kind != "full" or obj.predictor.train_shape[-1] != 48
                or contract.get("preprocessing_id") != manifest["input_identity_sha256"]
                or contract.get("feature_names") != obj.preprocessor.report()["output_features"]
                or obj.predictor.provenance.get("scanning_arm") != manifest["arm"]):
            raise ValueError("scanning cache input contract differs")
        return obj

    def predict_frame(self, frame):
        matrix = apply_intervention(
            self.manifest["arm"], self.preprocessor.transform(frame),
            frame["L4_SRC_PORT"].to_numpy(dtype=np.int64), self.mapping,
            self.manifest["center"], self.manifest["scale"])
        scores = _score_matrix(self.predictor, matrix, self.manifest["query_batch_size"])
        return pd.DataFrame({"row_id": _row_ids(frame, self.preprocessor.schema, "query"),
                             "score": scores,
                             "predicted_attack": apply_threshold(scores, self.registration).astype(np.uint8)})

    def predict_csv(self, input_path, output_path):
        output_path = Path(output_path)
        if output_path.exists():
            raise FileExistsError(output_path)
        frame = read_csv(input_path, self.preprocessor.schema, require_label=False)
        result = self.predict_frame(frame)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        return {"rows": len(result), "output": str(output_path.resolve()), "arm": self.manifest["arm"]}


def _bools(series):
    mapped = series.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if mapped.isna().any():
        raise ValueError("invalid saved binary decisions")
    return mapped.to_numpy(dtype=bool)


def baseline_replay(calibration_scores, evaluation_scores, registration, paths, frames, schema, spec):
    """Check scores, threshold and every decision before allowing B1/B2 builds."""
    old_c = pd.read_csv(paths["baseline_calibration"], float_precision="round_trip")
    old_e = pd.read_csv(paths["baseline_evaluation"], float_precision="round_trip")
    old_reg = read_json(paths["baseline_threshold"])
    result = {"passed": True, "atol": spec["score_atol"], "rtol": spec["score_rtol"], "roles": {}}
    for role, new, old, score_key, pred_key in (
        ("calibration", calibration_scores, old_c, "score", "predicted_attack"),
        ("evaluation", evaluation_scores, old_e, "full_score", "full_predicted"),
    ):
        if old["row_id"].astype(str).tolist() != _row_ids(frames[role], schema, role):
            raise ValueError(f"baseline {role} row order mismatch")
        expected = old[score_key].to_numpy(dtype=np.float64)
        close = np.abs(new - expected) <= spec["score_atol"] + spec["score_rtol"] * np.abs(expected)
        same = apply_threshold(new, registration) == _bools(old[pred_key])
        result["roles"][role] = {"rows": len(new), "outside_tolerance": int((~close).sum()),
                                 "decision_mismatches": int((~same).sum()),
                                 "max_absolute_score_difference": float(np.abs(new - expected).max())}
        result["passed"] &= bool(close.all() and same.all())
    same_rule = all(registration[k] == old_reg[k] for k in ("rank", "n_normal", "alpha_exact", "threshold_kind"))
    if registration["threshold"] is None or old_reg["threshold"] is None:
        threshold_close = registration["threshold"] is old_reg["threshold"]
    else:
        threshold_close = abs(registration["threshold"] - old_reg["threshold"]) <= (
            spec["score_atol"] + spec["score_rtol"] * abs(old_reg["threshold"]))
    result["threshold_passed"] = bool(same_rule and threshold_close)
    result["passed"] &= result["threshold_passed"]
    return result


def run_scanning(config_path, output_dir):
    import torch

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    spec, schema, paths, frames, audit = preflight_scanning(config_path)
    output.mkdir(parents=True)
    state = {"status": "running", "completed_arms": [], "stage": "frozen_preprocessing_load"}
    write_json(output / "state.json", state)
    write_json(output / "protocol.json", spec)
    write_json(output / "preflight.json", audit)
    write_json(output / "source_snapshot.json", _code_snapshot())
    write_json(output / "environment.json", environment_snapshot())
    try:
        torch.set_num_threads(spec["threads"])
        shared = output / "shared"
        shared.mkdir()
        shutil.copyfile(paths["preprocessor"], shared / "preprocessor.joblib")
        prep_hash = sha256_file(shared / "preprocessor.joblib")
        prep = CSVPreprocessor.load(shared / "preprocessor.joblib", expected_sha256=prep_hash)
        if (prep.profile != PROTECTED_IDS_PROFILE or prep.n_features_out_ != 48
                or list(prep.output_features_) != list(schema.features)
                or prep.protected_mean_[0] != spec["center"] or prep.protected_scale_[0] != spec["scale"]):
            raise ValueError("frozen preprocessing width/order/scaling mismatch")
        frozen = {role: prep.transform(frames[role]) for role in ROLES}
        ports = {role: frames[role]["L4_SRC_PORT"].to_numpy(dtype=np.int64) for role in ROLES}
        destinations = {role: frames[role]["L4_DST_PORT"].to_numpy(dtype=np.int64) for role in ROLES}
        # This is the only construction call; it receives P alone, before any model score.
        mapping, construction = build_frequency_bijection(ports["support"])
        np.save(shared / "source_port_mapping.npy", mapping, allow_pickle=False)
        map_hash = mapping_sha256(mapping)
        old_e = pd.read_csv(paths["baseline_evaluation"], float_precision="round_trip")
        cohorts = build_cohorts(frames, schema, old_e)
        cohort_frame = pd.DataFrame({"row_id": _row_ids(frames["evaluation"], schema, "evaluation"), **cohorts})
        cohort_frame.to_csv(output / "evaluation_cohorts.csv", index=False)
        mapping_audit = audit_mapping(
            mapping, ports["support"], ports,
            cohort_masks={name: {"role": "evaluation", "mask": mask} for name, mask in cohorts.items()},
            role_destination_ports=destinations)
        write_json(output / "mapping_audit.json", {"construction": construction, "audit": mapping_audit,
                                                   "mapping_sha256": map_hash})
        reference = output / "reference"
        reference.mkdir()
        for name in ("baseline_calibration", "baseline_evaluation", "baseline_threshold", "preparation"):
            shutil.copyfile(paths[name], reference / (name + Path(paths[name]).suffix))
        scores, calibrations, registrations, results, timings, collisions = {}, {}, {}, {}, {}, {}
        truth = labels(frames["evaluation"], schema)
        families = frames["evaluation"][schema.family_column].astype(str).to_numpy()
        evaluation_rows = pd.DataFrame({"row_id": _row_ids(frames["evaluation"], schema, "evaluation"),
                                        "label": truth, "attack_family": families})
        for arm in ARMS:
            state["stage"] = f"build_{arm}"
            write_json(output / "state.json", state)
            directory = output / arm
            directory.mkdir()
            matrices = {role: apply_intervention(arm, frozen[role], ports[role], mapping,
                                                 spec["center"], spec["scale"]) for role in ROLES}
            collisions[arm] = {role: audit_input_collisions(matrices[role], labels(frames[role], schema))
                               for role in ROLES}
            write_json(output / "input_collisions.json", collisions)
            identity = _input_identity(arm, prep_hash, map_hash, spec["center"], spec["scale"])
            torch.manual_seed(spec["seed"])
            adapter = FullCacheAdapter.from_checkpoint(
                paths["checkpoint"], device=spec["build_device"], expected_sha256=spec["checkpoint"]["sha256"])
            adapter.provenance.update({"scanning_arm": arm, "data_kind": spec["data_kind"],
                                      "protocol_sha256": canonical_hash(spec),
                                      "input_contract": {"preprocessing_id": identity,
                                                         "feature_names": list(schema.features),
                                                         "label_mapping": {"normal": 0, "attack": 1}}})
            start = time.perf_counter()
            adapter.build(torch.from_numpy(matrices["support"]).unsqueeze(0),
                          torch.from_numpy(labels(frames["support"], schema)).unsqueeze(0))
            if spec["build_device"] == "cuda":
                torch.cuda.synchronize()
            timings[f"{arm}_build_seconds"] = time.perf_counter() - start
            predictor = adapter.as_predictor()
            predictor.save(directory / "model.pt")
            del predictor, adapter
            gc.collect()
            if spec["build_device"] == "cuda":
                torch.cuda.empty_cache()
            predictor = CompactPredictor.load(directory / "model.pt", device="cpu")
            state["stage"] = f"score_{arm}"
            write_json(output / "state.json", state)
            start = time.perf_counter()
            c = _score_matrix(predictor, matrices["calibration"], spec["query_batch_size"])
            registration = register_threshold(c, spec["alpha"], model_sha256=sha256_file(directory / "model.pt"),
                                              preprocessing_sha256=identity,
                                              calibration_sha256=spec["files"]["calibration"]["sha256"])
            write_json(directory / "threshold.json", registration)
            e = _score_matrix(predictor, matrices["evaluation"], spec["query_batch_size"])
            timings[f"{arm}_C_E_score_seconds"] = time.perf_counter() - start
            del predictor, matrices
            scores[arm], calibrations[arm], registrations[arm] = e, c, registration
            pd.DataFrame({"row_id": _row_ids(frames["calibration"], schema, "calibration"), "label": 0,
                          "score": c, "predicted_attack": apply_threshold(c, registration).astype(np.uint8)}).to_csv(
                              output / f"{arm}_calibration.csv", index=False)
            evaluation_rows[f"{arm}_score"] = e
            evaluation_rows[f"{arm}_predicted"] = apply_threshold(e, registration)
            evaluation_rows[f"{arm}_C_percentile"] = empirical_percentiles(e, c)
            evaluation_rows.to_csv(output / "evaluation_scores.csv", index=False)
            files = {"model.pt": sha256_file(directory / "model.pt"),
                     "threshold.json": sha256_file(directory / "threshold.json"),
                     "../shared/preprocessor.joblib": prep_hash}
            if arm == ARMS[1]:
                files["../shared/source_port_mapping.npy"] = sha256_file(shared / "source_port_mapping.npy")
            deployment = {"format": DEPLOYMENT, "arm": arm, "threads": spec["threads"], "width": 48,
                          "query_batch_size": spec["query_batch_size"], "center": spec["center"],
                          "scale": spec["scale"], "mapping_sha256": map_hash,
                          "input_identity_sha256": identity, "data_kind": spec["data_kind"],
                          "calibration_sha256": spec["files"]["calibration"]["sha256"], "files": files}
            write_json(directory / "deployment.json", deployment)
            if arm == ARMS[0]:
                replay = baseline_replay(c, e, registration, paths, frames, schema, spec)
                write_json(output / "baseline_replay.json", replay)
                if not replay["passed"]:
                    raise ValueError("B0 baseline replay failed; B1/B2 were not run")
            results[arm] = analyze_arm(truth, families, e, c, registration, scores[ARMS[0]],
                                       registrations[ARMS[0]], cohorts,
                                       baseline_calibration_scores=calibrations[ARMS[0]])
            state["completed_arms"].append(arm)
            write_json(output / "state.json", state)
            write_json(output / "timings.json", timings)
        report = {"format": "tracelight-scanning-mechanism-results-v1", "data_kind": spec["data_kind"],
                  "scope": "opened development mechanism diagnosis", "arms": results,
                  "workload": {"full_context_builds": 3, "calibration_scored_rows": sum(map(len, calibrations.values())),
                               "evaluation_scored_rows": sum(map(len, scores.values())),
                               "optimizer_updates": 0, "qcoc_compilations": 0},
                  "cohort_counts": {name: int(mask.sum()) for name, mask in cohorts.items()},
                  "timing_scope": "one execution wall time; not p95 or RSS"}
        report["files"] = {name: sha256_file(output / name) for name in (
            "evaluation_scores.csv", "evaluation_cohorts.csv", "mapping_audit.json", "input_collisions.json",
            *(f"{arm}_calibration.csv" for arm in ARMS))}
        write_json(output / "report.json", report)
        state.update(status="complete", stage="complete")
        write_json(output / "state.json", state)
        verify_scanning_result(output)
        return report
    except Exception as exc:
        state.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        write_json(output / "state.json", state)
        (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def verify_scanning_result(output):
    """Artifact-only verification; does not deserialize a model or preprocessing."""
    output = Path(output)
    state = read_json(output / "state.json")
    if state["status"] != "complete" or state["completed_arms"] != list(ARMS):
        raise ValueError("three-arm result is incomplete")
    if not read_json(output / "baseline_replay.json")["passed"]:
        raise ValueError("baseline replay did not pass")
    spec = read_json(output / "protocol.json")
    e = pd.read_csv(output / "evaluation_scores.csv", float_precision="round_trip")
    cohort_rows = pd.read_csv(output / "evaluation_cohorts.csv", float_precision="round_trip")
    if e["row_id"].tolist() != cohort_rows["row_id"].tolist():
        raise ValueError("cohort row order differs")
    cohorts = {name: _bools(cohort_rows[name]) for name in cohort_rows if name != "row_id"}
    report = read_json(output / "report.json")
    for name, digest in report["files"].items():
        if sha256_file(output / name) != digest:
            raise ValueError(f"result file hash differs: {name}")
    for name, suffix in (("baseline_calibration", ".csv"), ("baseline_evaluation", ".csv"),
                         ("baseline_threshold", ".json"), ("preparation", ".json")):
        if sha256_file(output / "reference" / (name + suffix)) != spec["artifacts"][name]["sha256"]:
            raise ValueError(f"reference identity differs: {name}")
    reference_c = pd.read_csv(output / "reference/baseline_calibration.csv", float_precision="round_trip")
    reference_e = pd.read_csv(output / "reference/baseline_evaluation.csv", float_precision="round_trip")
    if any(e[column].tolist() != reference_e[column].tolist() for column in ("row_id", "label", "attack_family")):
        raise ValueError("evaluation identity, labels or families differ from frozen reference")
    if report["cohort_counts"] != {name: int(mask.sum()) for name, mask in cohorts.items()}:
        raise ValueError("cohort counts differ")
    expected_work = {"full_context_builds": 3, "calibration_scored_rows": 3 * len(reference_c),
                     "evaluation_scored_rows": 3 * len(reference_e), "optimizer_updates": 0,
                     "qcoc_compilations": 0}
    if report["workload"] != expected_work:
        raise ValueError("workload differs from three-arm protocol")
    registrations, calibrations = {}, {}
    for arm in ARMS:
        directory = output / arm
        manifest = read_json(directory / "deployment.json")
        if (manifest.get("format") != DEPLOYMENT or manifest.get("arm") != arm
                or manifest.get("width") != spec["width"] or manifest.get("threads") != spec["threads"]
                or manifest.get("query_batch_size") != spec["query_batch_size"]
                or manifest.get("center") != spec["center"] or manifest.get("scale") != spec["scale"]
                or manifest.get("calibration_sha256") != spec["files"]["calibration"]["sha256"]
                or manifest.get("files", {}).get("../shared/preprocessor.joblib")
                != spec["artifacts"]["preprocessor"]["sha256"]):
            raise ValueError("deployment differs from frozen arm protocol")
        registrations[arm] = _deployment_files(directory, manifest)
        c = pd.read_csv(output / f"{arm}_calibration.csv", float_precision="round_trip")
        if c["row_id"].tolist() != reference_c["row_id"].tolist() or not (c["label"] == 0).all():
            raise ValueError("calibration identity/order/normal labels differ")
        calibrations[arm] = c["score"].to_numpy()
        if len(c) != spec["files"]["calibration"]["rows"] or len(e) != spec["files"]["evaluation"]["rows"]:
            raise ValueError("score row counts differ")
        for saved_scores, saved_pred in ((c["score"], c["predicted_attack"]),
                                         (e[f"{arm}_score"], e[f"{arm}_predicted"])):
            if not np.array_equal(apply_threshold(saved_scores, registrations[arm]), _bools(saved_pred)):
                raise ValueError("saved decisions differ from registered threshold")
        recalculated = analyze_arm(e["label"].to_numpy(), e["attack_family"].to_numpy(),
                                  e[f"{arm}_score"].to_numpy(), calibrations[arm], registrations[arm],
                                  e[f"{ARMS[0]}_score"].to_numpy(), registrations[ARMS[0]], cohorts,
                                  baseline_calibration_scores=calibrations[ARMS[0]])
        # CSV round-trip may change a float64 final decimal; compare numeric leaves accordingly.
        _compare_report(report["arms"][arm], recalculated)
    return {"status": "verified", "arms": 3, "model_loads": 0, "preprocessing_loads": 0,
            "evaluation_rows": len(e)}


def _compare_report(expected, actual):
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            raise ValueError("report keys differ")
        for key in expected:
            _compare_report(expected[key], actual[key])
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            raise ValueError("report list length differs")
        for left, right in zip(expected, actual):
            _compare_report(left, right)
    elif isinstance(expected, float):
        if not np.isclose(expected, actual, atol=1e-12, rtol=1e-12):
            raise ValueError("recomputed report numeric value differs")
    elif expected != actual:
        raise ValueError("recomputed report value differs")
