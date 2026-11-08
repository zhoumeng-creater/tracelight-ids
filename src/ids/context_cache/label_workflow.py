"""Frozen support-label controls; real weights and data run in the laboratory.

The runner reuses frozen inputs and preprocessing. It never fits preprocessing,
selects support rows, optimizes weights, or supplies query labels to the model.
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
from .label_inputs import ARMS, attack_union_log_odds, build_support_label_views
from .label_metrics import analyze_label_results, score_diagnostics
from .label_protocol import preflight_label
from .preprocessing import CSVPreprocessor, CSVSchema, PROTECTED_IDS_PROFILE, labels, read_csv
from .scanning_inputs import audit_input_collisions
from .scanning_metrics import build_cohorts
from .scanning_workflow import _bools, _compare_report, baseline_replay
from .tabicl_adapter import CompactPredictor, FullCacheAdapter

DEPLOYMENT = "tracelight-scanning-label-deployment-v1"
RESULTS = "tracelight-scanning-label-results-v1"
ROLES = ("support", "calibration", "evaluation")
CLASS_NAMES = {
    "baseline": ["Normal", "Attack"],
    "semantic_family": ["Normal", "Scanning", "OtherAttack"],
    "grouped_sham": ["Normal", "AttackA", "AttackB"],
}
SCORE_CONTRACT = {"engine": "numpy-float32-logaddexp-v1", "temperature": 1,
                  "normal_class": 0, "formula": "logsumexp(attack_logits)-normal_logit",
                  "binary_operation": "float32(logit_1-logit_0)"}


def _score_matrix(predictor, matrix, batch_size):
    """Keep original FP32 logits and one identical scoring path for all callers."""
    import torch

    chunks = []
    with torch.no_grad():
        for begin in range(0, len(matrix), batch_size):
            logits = predictor.predict_logits(torch.from_numpy(matrix[begin:begin + batch_size]).unsqueeze(0))
            chunks.append(logits[0].cpu().numpy())
    logits = np.concatenate(chunks)
    if logits.dtype != np.float32:
        raise ValueError("label experiment requires FP32 model logits")
    scores = attack_union_log_odds(logits).astype(np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite model scores")
    return scores, logits


def _identity(arm, prep_hash, labels_hash):
    return canonical_hash({"arm": arm, "preprocessor_sha256": prep_hash,
                           "support_labels_sha256": labels_hash, "width": 48,
                           "classes": CLASS_NAMES[arm], "score_contract": SCORE_CONTRACT})


def _deployment_files(directory, manifest):
    arm = manifest.get("arm")
    if (manifest.get("format") != DEPLOYMENT or arm not in ARMS
            or manifest.get("num_classes") != len(CLASS_NAMES[arm])
            or manifest.get("classes") != CLASS_NAMES[arm]
            or manifest.get("score_contract") != SCORE_CONTRACT
            or manifest.get("threads") != 1 or manifest.get("width") != 48
            or type(manifest.get("query_batch_size")) is not int or manifest["query_batch_size"] < 1):
        raise ValueError("unsupported label deployment class/score contract")
    expected = {"model.pt", "threshold.json", "../shared/preprocessor.joblib"}
    if set(manifest["files"]) != expected:
        raise ValueError("label deployment file set differs")
    for name, digest in manifest["files"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"label deployment hash mismatch: {name}")
    identity = _identity(arm, manifest["files"]["../shared/preprocessor.joblib"],
                         manifest["support_labels_sha256"])
    if identity != manifest["input_identity_sha256"]:
        raise ValueError("label deployment input/class identity differs")
    registration = read_json(directory / "threshold.json")
    if registration["binding"] != {
        "model_sha256": manifest["files"]["model.pt"],
        "preprocessing_sha256": identity,
        "calibration_sha256": manifest["calibration_sha256"],
    }:
        raise ValueError("label threshold class/input binding differs")
    apply_threshold([], registration)
    return registration


class LabelDetector:
    """Cold all-attack detector; inference needs only raw feature columns."""

    @classmethod
    def load(cls, directory):
        import torch

        directory = Path(directory)
        manifest = read_json(directory / "deployment.json")
        registration = _deployment_files(directory, manifest)
        torch.set_num_threads(manifest["threads"])
        obj = cls()
        obj.manifest, obj.registration = manifest, registration
        obj.preprocessor = CSVPreprocessor.load(
            directory / "../shared/preprocessor.joblib",
            expected_sha256=manifest["files"]["../shared/preprocessor.joblib"])
        obj.predictor = CompactPredictor.load(directory / "model.pt", device="cpu")
        provenance = obj.predictor.provenance
        contract = provenance.get("input_contract", {})
        if (obj.predictor.cache_kind != "full" or obj.predictor.train_shape[-1] != 48
                or obj.predictor.num_classes != manifest["num_classes"]
                or contract.get("preprocessing_id") != manifest["input_identity_sha256"]
                or contract.get("feature_names") != obj.preprocessor.report()["output_features"]
                or contract.get("classes") != manifest["classes"]
                or contract.get("score_contract") != SCORE_CONTRACT
                or contract.get("query_batch_size") != manifest["query_batch_size"]
                or provenance.get("label_arm") != manifest["arm"]
                or provenance.get("support_labels_sha256") != manifest["support_labels_sha256"]):
            raise ValueError("label cache input/class contract differs")
        return obj

    def predict_frame(self, frame):
        scores, logits = _score_matrix(self.predictor, self.preprocessor.transform(frame),
                                       self.manifest["query_batch_size"])
        result = pd.DataFrame({"row_id": _row_ids(frame, self.preprocessor.schema, "query"),
                               "score": scores,
                               "predicted_attack": apply_threshold(scores, self.registration).astype(np.uint8)})
        for index in range(logits.shape[1]):
            result[f"logit_{index}"] = logits[:, index]
        return result

    def predict_csv(self, input_path, output_path):
        output_path = Path(output_path)
        if output_path.exists():
            raise FileExistsError(output_path)
        frame = read_csv(input_path, self.preprocessor.schema, require_label=False)
        result = self.predict_frame(frame)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        return {"rows": len(result), "output": str(output_path.resolve()), "arm": self.manifest["arm"]}


def _labels_digest(values):
    import hashlib

    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def run_label(config_path, output_dir):
    import torch

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    spec, schema, paths, frames, audit = preflight_label(config_path)
    output.mkdir(parents=True)
    state = {"status": "running", "completed_arms": [], "stage": "frozen_preprocessing_load"}
    for name, value in (("state", state), ("protocol", spec), ("preflight", audit),
                        ("source_snapshot", _code_snapshot()), ("environment", environment_snapshot())):
        write_json(output / f"{name}.json", value)
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
        support_binary = labels(frames["support"], schema)
        views, label_audit = build_support_label_views(
            frozen["support"], support_binary, frames["support"][schema.family_column].to_numpy(),
            small_class_rows=spec["sham_class_rows"])
        # Freeze all support-label interventions before the first model score.
        np.save(shared / "support_fp32.npy", frozen["support"], allow_pickle=False)
        label_rows = pd.DataFrame({"row_id": _row_ids(frames["support"], schema, "support"),
                                  "binary_label": support_binary,
                                  "attack_family": frames["support"][schema.family_column].to_numpy(),
                                  "input_group_sha256": label_audit["row_input_group_sha256"], **views})
        label_rows["communication_group"] = frames["support"][schema.group_column].astype(str).to_numpy()
        label_rows.to_csv(output / "support_labels.csv", index=False)
        write_json(output / "label_audit.json", label_audit)
        collisions = {role: audit_input_collisions(frozen[role], labels(frames[role], schema)) for role in ROLES}
        write_json(output / "input_collisions.json", collisions)
        old_e = pd.read_csv(paths["baseline_evaluation"], float_precision="round_trip")
        cohorts = build_cohorts(frames, schema, old_e)
        pd.DataFrame({"row_id": _row_ids(frames["evaluation"], schema, "evaluation"), **cohorts}).to_csv(
            output / "evaluation_cohorts.csv", index=False)
        reference = output / "reference"
        reference.mkdir()
        for name in ("baseline_calibration", "baseline_evaluation", "baseline_threshold", "preparation", "support"):
            shutil.copyfile(paths[name], reference / (name + Path(paths[name]).suffix))
        scores, calibrations, registrations, timings, sizes = {}, {}, {}, {}, {}
        truth = labels(frames["evaluation"], schema)
        families = frames["evaluation"][schema.family_column].astype(str).to_numpy()
        evaluation_rows = pd.DataFrame({"row_id": _row_ids(frames["evaluation"], schema, "evaluation"),
                                        "label": truth, "attack_family": families})
        evaluation_rows["communication_group"] = frames["evaluation"][schema.group_column].astype(str).to_numpy()
        for arm in ARMS:
            state["stage"] = f"build_{arm}"
            write_json(output / "state.json", state)
            directory = output / arm
            directory.mkdir()
            labels_hash = _labels_digest(views[arm])
            identity = _identity(arm, prep_hash, labels_hash)
            torch.manual_seed(spec["seed"])
            adapter = FullCacheAdapter.from_checkpoint(
                paths["checkpoint"], device=spec["build_device"], expected_sha256=spec["checkpoint"]["sha256"])
            adapter.provenance.update({"label_arm": arm, "data_kind": spec["data_kind"],
                                      "support_labels_sha256": labels_hash, "protocol_sha256": canonical_hash(spec),
                                      "input_contract": {"preprocessing_id": identity,
                                                         "feature_names": list(schema.features),
                                                         "classes": CLASS_NAMES[arm],
                                                         "query_batch_size": spec["query_batch_size"],
                                                         "score_contract": SCORE_CONTRACT}})
            start = time.perf_counter()
            adapter.build(torch.from_numpy(frozen["support"]).unsqueeze(0),
                          torch.from_numpy(views[arm]).unsqueeze(0))
            if spec["build_device"] == "cuda":
                torch.cuda.synchronize()
            timings[f"{arm}_build_seconds"] = time.perf_counter() - start
            predictor = adapter.as_predictor()
            if predictor.num_classes != len(CLASS_NAMES[arm]):
                raise ValueError("built cache class count differs")
            sizes[arm] = predictor.size_report()
            predictor.save(directory / "model.pt")
            sizes[arm]["model_artifact_bytes"] = (directory / "model.pt").stat().st_size
            del predictor, adapter
            gc.collect()
            if spec["build_device"] == "cuda":
                torch.cuda.empty_cache()
            predictor = CompactPredictor.load(directory / "model.pt", device="cpu")
            state["stage"] = f"score_{arm}"
            write_json(output / "state.json", state)
            start = time.perf_counter()
            c, c_logits = _score_matrix(predictor, frozen["calibration"], spec["query_batch_size"])
            registration = register_threshold(c, spec["alpha"], model_sha256=sha256_file(directory / "model.pt"),
                                              preprocessing_sha256=identity,
                                              calibration_sha256=spec["files"]["calibration"]["sha256"])
            write_json(directory / "threshold.json", registration)
            e, e_logits = _score_matrix(predictor, frozen["evaluation"], spec["query_batch_size"])
            timings[f"{arm}_C_E_score_seconds"] = time.perf_counter() - start
            del predictor
            scores[arm], calibrations[arm], registrations[arm] = e, c, registration
            c_rows = pd.DataFrame({"row_id": _row_ids(frames["calibration"], schema, "calibration"), "label": 0,
                                   "score": c, "predicted_attack": apply_threshold(c, registration).astype(np.uint8)})
            c_rows["communication_group"] = frames["calibration"][schema.group_column].astype(str).to_numpy()
            for index in range(c_logits.shape[1]):
                c_rows[f"logit_{index}"] = c_logits[:, index]
                evaluation_rows[f"{arm}_logit_{index}"] = e_logits[:, index]
            for name, values in score_diagnostics(c, c, registration).items():
                c_rows[name] = values
            c_rows.to_csv(output / f"{arm}_calibration.csv", index=False)
            evaluation_rows[f"{arm}_score"] = e
            evaluation_rows[f"{arm}_predicted"] = apply_threshold(e, registration)
            for name, values in score_diagnostics(e, c, registration).items():
                evaluation_rows[f"{arm}_{name}"] = values
            evaluation_rows.to_csv(output / "evaluation_scores.csv", index=False)
            files = {"model.pt": sha256_file(directory / "model.pt"),
                     "threshold.json": sha256_file(directory / "threshold.json"),
                     "../shared/preprocessor.joblib": prep_hash}
            write_json(directory / "deployment.json", {
                "format": DEPLOYMENT, "arm": arm, "threads": spec["threads"], "width": 48,
                "query_batch_size": spec["query_batch_size"], "num_classes": len(CLASS_NAMES[arm]),
                "classes": CLASS_NAMES[arm], "score_contract": SCORE_CONTRACT,
                "support_labels_sha256": labels_hash, "input_identity_sha256": identity,
                "data_kind": spec["data_kind"], "calibration_sha256": spec["files"]["calibration"]["sha256"],
                "files": files})
            sizes[arm]["cold_deployment_file_bytes"] = {
                name: (directory / name).stat().st_size for name in (*files, "deployment.json")}
            sizes[arm]["cold_deployment_total_bytes"] = sum(sizes[arm]["cold_deployment_file_bytes"].values())
            if arm == ARMS[0]:
                replay = baseline_replay(c, e, registration, paths, frames, schema, spec)
                write_json(output / "baseline_replay.json", replay)
                if not replay["passed"]:
                    raise ValueError("T0 baseline replay failed; TF/TP were not run")
            state["completed_arms"].append(arm)
            write_json(output / "state.json", state)
            write_json(output / "timings.json", timings)
        metrics = analyze_label_results(truth, families, scores, calibrations, registrations, cohorts)
        report = {"format": RESULTS, "data_kind": spec["data_kind"],
                  "scope": "opened development label-granularity diagnosis", **metrics,
                  "score_contract": SCORE_CONTRACT,
                  "workload": {"full_context_builds": 3, "calibration_scored_rows": sum(map(len, calibrations.values())),
                               "evaluation_scored_rows": sum(map(len, scores.values())),
                               "optimizer_updates": 0, "qcoc_compilations": 0},
                  "cohort_counts": {name: int(mask.sum()) for name, mask in cohorts.items()},
                  "metadata_scope": {"communication_groups": "retained from frozen CSV metadata",
                                     "source_hosts": "unavailable in frozen prepared CSVs; not inferred from IDs",
                                     "timestamps": "unavailable in frozen prepared CSVs; not inferred from IDs",
                                     "independent_host_or_forward_time_confirmation": False},
                  "resources": sizes, "timing_scope": "one execution wall time; not p95 or RSS"}
        report["files"] = {name: sha256_file(output / name) for name in (
            "evaluation_scores.csv", "evaluation_cohorts.csv", "support_labels.csv", "label_audit.json",
            "shared/support_fp32.npy", "input_collisions.json", *(f"{arm}_calibration.csv" for arm in ARMS))}
        write_json(output / "report.json", report)
        state.update(status="complete", stage="complete")
        write_json(output / "state.json", state)
        verify_label_result(output)
        return report
    except Exception as exc:
        state.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        write_json(output / "state.json", state)
        (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def _verify_scores(frame, prefix, classes, registration, calibration):
    scores = frame[prefix + "score"].to_numpy(dtype=np.float64)
    logits = frame[[prefix + f"logit_{index}" for index in range(classes)]].to_numpy(dtype=np.float32)
    recomputed = attack_union_log_odds(logits).astype(np.float64)
    if not np.array_equal(scores, recomputed):
        raise ValueError("saved scores differ from FP32 attack-union logits")
    if not np.array_equal(apply_threshold(scores, registration), _bools(frame[prefix + "predicted_attack"]
                          if not prefix else frame[prefix + "predicted"])):
        raise ValueError("saved decisions differ from registered threshold")
    for name, values in score_diagnostics(scores, calibration, registration).items():
        if not np.array_equal(frame[prefix + name].to_numpy(dtype=np.float64), values):
            raise ValueError("saved calibration percentile/margin differs")


def verify_label_result(output):
    """Recompute saved-array semantics without loading a model or preprocessor."""
    output = Path(output)
    state, spec = read_json(output / "state.json"), read_json(output / "protocol.json")
    if state["status"] != "complete" or state["completed_arms"] != list(ARMS):
        raise ValueError("three-arm label result is incomplete")
    report = read_json(output / "report.json")
    if report.get("format") != RESULTS or report.get("score_contract") != SCORE_CONTRACT:
        raise ValueError("unsupported label result format/score contract")
    for name, digest in report["files"].items():
        if sha256_file(output / name) != digest:
            raise ValueError(f"result file hash differs: {name}")
    for name, suffix in (("baseline_calibration", ".csv"), ("baseline_evaluation", ".csv"),
                         ("baseline_threshold", ".json"), ("preparation", ".json"), ("support", ".csv")):
        expected = spec["files"][name] if name == "support" else spec["artifacts"][name]
        if sha256_file(output / "reference" / (name + suffix)) != expected["sha256"]:
            raise ValueError(f"reference identity differs: {name}")
    schema = CSVSchema.from_dict(spec["schema"])
    support = read_csv(output / "reference/support.csv", schema)
    support_rows = pd.read_csv(output / "support_labels.csv", float_precision="round_trip")
    if (support_rows["row_id"].astype(str).tolist() != _row_ids(support, schema, "support")
            or not np.array_equal(support_rows["binary_label"], labels(support, schema))
            or support_rows["attack_family"].tolist() != support[schema.family_column].tolist()):
        raise ValueError("support label identity/order/family differs from frozen P")
    matrix = np.load(output / "shared/support_fp32.npy", allow_pickle=False)
    views, label_audit = build_support_label_views(matrix, support_rows["binary_label"].to_numpy(),
                                                  support_rows["attack_family"].to_numpy(),
                                                  small_class_rows=spec["sham_class_rows"])
    if matrix.shape != (spec["files"]["support"]["rows"], 48):
        raise ValueError("saved support matrix shape differs")
    for arm in ARMS:
        if not np.array_equal(support_rows[arm], views[arm]):
            raise ValueError("saved support labels differ from fixed whole-group construction")
    if support_rows["input_group_sha256"].tolist() != label_audit["row_input_group_sha256"]:
        raise ValueError("saved input group identities differ")
    _compare_report(read_json(output / "label_audit.json"), label_audit)
    e = pd.read_csv(output / "evaluation_scores.csv", float_precision="round_trip")
    cohort_rows = pd.read_csv(output / "evaluation_cohorts.csv", float_precision="round_trip")
    if e["row_id"].tolist() != cohort_rows["row_id"].tolist():
        raise ValueError("cohort row order differs")
    cohorts = {name: _bools(cohort_rows[name]) for name in cohort_rows if name != "row_id"}
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
    registrations, calibrations, scores = {}, {}, {}
    for arm in ARMS:
        directory = output / arm
        manifest = read_json(directory / "deployment.json")
        if (manifest.get("arm") != arm or manifest.get("width") != spec["width"]
                or manifest.get("threads") != spec["threads"]
                or manifest.get("query_batch_size") != spec["query_batch_size"]
                or manifest.get("support_labels_sha256") != _labels_digest(views[arm])
                or manifest.get("calibration_sha256") != spec["files"]["calibration"]["sha256"]
                or manifest.get("files", {}).get("../shared/preprocessor.joblib")
                != spec["artifacts"]["preprocessor"]["sha256"]):
            raise ValueError("deployment differs from frozen label protocol")
        registrations[arm] = _deployment_files(directory, manifest)
        c = pd.read_csv(output / f"{arm}_calibration.csv", float_precision="round_trip")
        if c["row_id"].tolist() != reference_c["row_id"].tolist() or not (c["label"] == 0).all():
            raise ValueError("calibration identity/order/normal labels differ")
        calibrations[arm], scores[arm] = c["score"].to_numpy(), e[f"{arm}_score"].to_numpy()
        if len(c) != spec["files"]["calibration"]["rows"] or len(e) != spec["files"]["evaluation"]["rows"]:
            raise ValueError("score row counts differ")
        _verify_scores(c, "", len(CLASS_NAMES[arm]), registrations[arm], calibrations[arm])
        _verify_scores(e, f"{arm}_", len(CLASS_NAMES[arm]), registrations[arm], calibrations[arm])
        if report["resources"][arm]["model_artifact_bytes"] != (directory / "model.pt").stat().st_size:
            raise ValueError("model artifact byte count differs")
        file_bytes = {name: (directory / name).stat().st_size for name in (*manifest["files"], "deployment.json")}
        if (report["resources"][arm]["cold_deployment_file_bytes"] != file_bytes
                or report["resources"][arm]["cold_deployment_total_bytes"] != sum(file_bytes.values())):
            raise ValueError("cold deployment byte count differs")
    recomputed = analyze_label_results(e["label"].to_numpy(), e["attack_family"].to_numpy(), scores,
                                      calibrations, registrations, cohorts)
    for name, value in recomputed.items():
        _compare_report(report[name], value)
    # Recheck replay from saved reference, not just the recorded pass flag.
    frames = {"calibration": pd.DataFrame({schema.id_column: reference_c["row_id"]}),
              "evaluation": pd.DataFrame({schema.id_column: reference_e["row_id"]})}
    paths = {name: output / "reference" / (name + suffix) for name, suffix in
             (("baseline_calibration", ".csv"), ("baseline_evaluation", ".csv"), ("baseline_threshold", ".json"))}
    replay = baseline_replay(calibrations[ARMS[0]], scores[ARMS[0]], registrations[ARMS[0]], paths, frames, schema, spec)
    if not replay["passed"]:
        raise ValueError("saved T0 baseline replay did not pass")
    _compare_report(read_json(output / "baseline_replay.json"), replay)
    return {"status": "verified", "arms": 3, "model_loads": 0, "preprocessing_loads": 0,
            "evaluation_rows": len(e), "support_label_rows": len(support_rows)}
