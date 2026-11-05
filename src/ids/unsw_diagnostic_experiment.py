"""Frozen zero-fit diagnostic: bound F evidence and three-model CPU costs."""
from __future__ import annotations

from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
import shutil
import traceback

import numpy as np

from .budget_transfer import _no_links, _state_hash, _training_frames
from .common import canonical_hash, code_fingerprint, environment_snapshot, read_json, sha256_file, utc_now, write_json
from .models import load_bundle
from .runner import verify_development_inputs
from .unsw_kd import build_targets
from .unsw_kd_experiment import preprocessing_state

REPO = Path(__file__).resolve().parents[2]
PROTOCOL = REPO / "configs/unsw-zero-fit-diagnostic-20261105.json"
BINDING = REPO / "docs/research_alignment_20261105/kd_failure_research/mechanism-audit-binding.json"
COMPLETE = "UNSW_DIAGNOSTIC_COMPLETE.json"


def _check_native_dependency():
    from .native_trees import CATBOOST_VERSION
    try:
        actual = version("catboost")
    except PackageNotFoundError as exc:
        raise ValueError(f"Diagnostic requires existing lab catboost=={CATBOOST_VERSION}") from exc
    if actual != CATBOOST_VERSION:
        raise ValueError(f"Diagnostic requires catboost=={CATBOOST_VERSION}; found {actual}")
    return actual


def _bound_file(root, relative, expected):
    path = _no_links(root / relative)
    if not path.is_relative_to(root) or not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"Frozen diagnostic artifact changed or missing: {relative}")
    return path


def _verify_files(binding, roots):
    result = {}
    for item in binding["files"]:
        path = _bound_file(roots[item["root_alias"]], item["path"], item["sha256"])
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"Frozen artifact byte count changed: {path}")
        result[f"{item['root_alias']}/{item['path']}"] = item["sha256"]
    return result


def _bound_inputs(screen_dir, round10_dir, data_dir, *, binding_path=None):
    selected = _no_links(BINDING if binding_path is None else binding_path)
    binding = read_json(selected)
    if binding != read_json(BINDING):
        raise ValueError("Only the checked-in diagnostic binding is executable")
    protocol = read_json(PROTOCOL)
    if protocol["protocol_id"] != binding["protocol_id"] or binding["current_fit_budget"] != 0:
        raise ValueError("Diagnostic protocol/binding identity mismatch")
    native_version = _check_native_dependency()
    roots = {"screen": _no_links(screen_dir), "round10": _no_links(round10_dir)}
    data = _no_links(data_dir)
    files = _verify_files(binding, roots)
    manifest = verify_development_inputs(data, splits=("train",))
    if manifest != binding["prepared_contract"]["data_manifest"] or manifest.get("data_kind") == "synthetic":
        raise ValueError("Original prepared physical data and manifest are required")
    previous = read_json(roots["round10"] / "preflight.json")
    if previous["data_manifest"] != manifest:
        raise ValueError("Prepared manifest differs from successful round10 preflight")
    done = read_json(roots["round10"] / "UNSW_KD_COMPLETE.json")
    if done.get("status") != "unsw_kd_complete" or done.get("student_fits") != 4:
        raise ValueError("Expected the completed four-arm experiment")
    # These scalar logs are already bound by the sealed completion file. No model query.
    initial = {}
    for arm in ("H", "K"):
        relative = f"arms/{arm}/history.json"
        digest = done["files"][relative]
        history = read_json(_bound_file(roots["round10"], relative, digest))
        if len(history) != 40 or history[0]["step"] != 1:
            raise ValueError("Original 40-step history is incomplete")
        initial[arm] = history[0]
        files[f"round10/{relative}"] = digest
    frames = _training_frames(roots["round10"], data)
    for role, short in (("fit", "F"), ("reference", "R"), ("constraint", "C")):
        if len(frames[role]) != binding["data_roles"][short]:
            raise ValueError(f"Frozen {short} row count changed")
    original = read_json(roots["round10"] / "protocol.json")
    paths = {"anchor": roots["screen"] / "warmups/all-small-s17/model.joblib",
             "H40": roots["round10"] / "arms/H/model.joblib",
             "K40": roots["round10"] / "arms/K/model.joblib"}
    bundles = {key: load_bundle(path, device="cpu") for key, path in paths.items()}
    for key, bundle in bundles.items():
        if (bundle.model_name != "mlp" or bundle.estimator is not None
                or list(bundle.hidden_dims) != original["student"]["hidden"]
                or bundle.encoded_dim != original["student"]["input_features"]
                or sum(v.numel() for v in bundle.network_state.values()) != original["student"]["parameters"]
                or canonical_hash(preprocessing_state(bundle.preprocessor))
                != original["data"]["preprocessing_semantic_sha256"]):
            raise ValueError(f"Frozen student architecture/preprocessing changed: {key}")
        expected = (original["student"]["anchor_state_sha256"] if key == "anchor" else
                    read_json(paths[key].parent / "run.json")["final_state_sha256"])
        if _state_hash(bundle.network_state) != expected:
            raise ValueError(f"Student state mismatch: {key}")
    with np.load(roots["round10"] / "targets.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    with np.load(roots["screen"] / "teachers/all-s17/training-scores.npz", allow_pickle=False) as saved:
        teacher_F = saved["fit"].copy()  # Do not decode any other cached role.
    _, target_audit, rebuilt = build_targets(frames["fit"], teacher_F, frozen_q=arrays["teacher_q"])
    if set(arrays) != set(rebuilt) or any(not np.array_equal(arrays[k], rebuilt[k]) for k in arrays):
        raise ValueError("Original F target/identity/weight alignment changed")
    summary = read_json(roots["round10"] / "summary.json")
    timing = summary["cost"]["timing"]["batches"]
    ids = set(frames["constraint"].sample_id.astype(str))
    for size in (1, 256):
        batches = timing[str(size)]["sample_ids"]
        if len(batches) != 75 or any(len(v) != size or not set(map(str, v)).issubset(ids) for v in batches):
            raise ValueError("Original CPU input batches do not match C identities")
    hgb = roots["screen"] / "teachers/all-s17"
    cb = roots["screen"] / "native-trees/catboost-reference-s17"
    specs = {
        "H40": {"kind": "mlp", "path": str(paths["H40"]),
                "threshold": read_json(paths["H40"].parent / "run.json")["endpoint"]["working_point"]["threshold"]},
        "HGB": {"kind": "hgb", "path": str(hgb / "model.joblib"),
                "threshold": read_json(hgb / "run.json")["selection"]["working_point"]["threshold"]},
        "CatBoost_reference": {"kind": "catboost", "path": str(cb),
                "threshold": read_json(cb / "run.json")["selection"]["working_point"]["threshold"]},
    }
    if any(not np.isfinite(v["threshold"]) for v in specs.values()):
        raise ValueError("Nonfinite original threshold")
    receipt = {
        "status": "preflight_complete", "protocol_id": binding["protocol_id"],
        "binding_sha256": sha256_file(selected), "protocol_sha256": sha256_file(PROTOCOL),
        "catboost_version": native_version,
        "files_sha256": files, "data_manifest": manifest, "target_audit": target_audit,
        "initial_anchor_losses": {"hard_bce": initial["H"]["hard_bce"],
                                  "mixed_bce": initial["K"]["objective_bce"]},
        "roots": {k: str(v) for k, v in roots.items()}, "data_dir": str(data),
        "student_fits": 0, "teacher_fits": 0, "optimizer_updates": 0,
        "new_inference_rows": 0, "reserved_partitions_read": [],
    }
    return {"binding": binding, "protocol": protocol, "receipt": receipt, "frames": frames,
            "bundles": bundles, "targets": arrays, "model_specs": specs, "timing_batches": timing,
            "roots": roots, "data": data, "binding_path": selected}


def preflight_diagnostic(screen_dir, round10_dir, data_dir, *, binding_path=None):
    return _bound_inputs(screen_dir, round10_dir, data_dir, binding_path=binding_path)["receipt"]


def _report(output, result):
    lines = ["# UNSW 零拟合诊断", "", f"有效性验收：{'通过' if result['validity_passed'] else '未通过'}。",
             "本轮没有训练，也不据此自动启动训练。", "", "## 有效性检查", ""]
    lines.extend(f"- {name}：{'通过' if value else '未通过'}" for name, value in result["checks"].items())
    lines += ["", "## F 目标", "", "| 模型 | hard BCE | teacher KL | mixed BCE |",
              "|---|---:|---:|---:|"]
    for name, info in result["F"]["models"].items():
        loss = info["losses"]
        lines.append(f"| {name} | {loss['hard_bce']:.9f} | {loss['teacher_kl']:.9f} | {loss['mixed_bce']:.9f} |")
    lines += ["", "anchor-gradients.npz 的梯度只对应共同起点，不代表 H40/K40 终点梯度。",
              "成本逐次时长、输入顺序和独立进程 RSS 见 cost/；F 逐行及分组残差见 F/。",
              "原四臂失败结论不变；本次有效性通过不等于新方法或独立泛化通过。"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _execute(context, output, *, device, source):
    from .unsw_diagnostic_f import diagnose_f
    from .unsw_diagnostic_cost import measure_cost

    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "protocol.json", context["protocol"])
    write_json(root / "binding.json", context["binding"])
    write_json(root / "preflight.json", context["receipt"])
    write_json(root / "plan.json", {"source": source, "environment": environment_snapshot(),
                                   "device": device, "budget": context["protocol"]["budget"],
                                   "training_branch_enabled": False, "automatic_retry": False})
    try:
        for name, digest in source.get("files", {}).items():
            origin = _bound_file(REPO, name, digest)
            target = root / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target)
        print("[diagnostic] F: three frozen models, anchor gradients, zero updates", flush=True)
        f = diagnose_f(context["bundles"], context["frames"]["fit"], context["targets"], root / "F", device=device)
        print("[diagnostic] CPU: frozen 450 timings, 30 warmups, three RSS probes", flush=True)
        cost = measure_cost(context["model_specs"], context["frames"]["constraint"],
                            context["timing_batches"], root / "cost")
        checks = _acceptance(context, f, cost)
        # Reverify immutable files/data after measurement, without further model calls.
        _verify_files(context["binding"], context["roots"])
        for key, digest in context["receipt"]["files_sha256"].items():
            alias, relative = key.split("/", 1)
            _bound_file(context["roots"][alias], relative, digest)
        if verify_development_inputs(context["data"], splits=("train",)) != context["receipt"]["data_manifest"]:
            raise ValueError("Prepared input changed during diagnosis")
        if source.get("files") and code_fingerprint() != source:
            raise ValueError("Source changed during diagnosis")
        if (sha256_file(context["binding_path"]) != context["receipt"]["binding_sha256"]
                or sha256_file(PROTOCOL) != context["receipt"]["protocol_sha256"]):
            raise ValueError("Diagnostic binding/protocol changed during measurement")
        checks["immutable_inputs_and_source"] = True
        result = {"status": "unsw_diagnostic_complete", "protocol_id": context["binding"]["protocol_id"],
                  "student_fits": 0, "teacher_fits": 0, "optimizer_updates": 0,
                  "F": f, "cost": cost, "checks": checks, "validity_passed": all(checks.values()),
                  "budget_limits": context["protocol"]["budget"],
                  "reserved_partitions_read": [], "automatic_expansion": False,
                  "independent_confirmation": False, "training_branch_enabled": False}
        write_json(root / "summary.json", result)
        _report(root, result)
        write_json(root / COMPLETE, {"status": result["status"], "created_at": utc_now(),
                    "student_fits": 0, "teacher_fits": 0, "optimizer_updates": 0,
                    "validity_passed": result["validity_passed"],
                    "files": {p.relative_to(root).as_posix(): sha256_file(p)
                              for p in root.rglob("*") if p.is_file()}})
        return result
    except BaseException as exc:
        write_json(root / "FAILURE.json", {"status": "technical_failure", "exception": type(exc).__name__,
                   "message": str(exc), "traceback": traceback.format_exc(),
                   "action": "preserve partial results; no automatic retry or training"})
        raise


def _acceptance(context, f, cost):
    """Combine actual module measurements against the frozen diagnostic contract."""
    expected = context["receipt"]["initial_anchor_losses"]
    losses = f["models"]["anchor"]["losses"]
    atol = context["protocol"]["acceptance"]["anchor_loss_atol"]
    checks = {"anchor_initial_losses": all(abs(losses[k] - v) <= atol for k, v in expected.items()),
              "gradient_identity": f["gradient_identity"]["valid"]}
    checks.update(_module_checks(context, f, cost))
    return checks


def _module_checks(context, f, cost):
    limits, fb, cb = context["protocol"]["budget"], f["budget"], cost["budget"]
    budget = {
        "student_fits": fb["new_fits"] + cb["new_fits"],
        "teacher_fits": fb["new_fits"] + cb["new_fits"],
        "optimizer_updates": fb["optimizer_updates"] + cb["optimizer_updates"],
        "teacher_F_queries": fb["new_teacher_queries"],
        "F_forward_rows": fb["forward_rows"], "anchor_backward_calls": fb["backward_calls"],
        "CPU_timed_calls": cb["formal_calls"], "CPU_timed_rows": cb["formal_rows"],
        "CPU_warmup_calls": cb["warmup_calls"], "CPU_warmup_rows": cb["warmup_rows"],
        "CPU_RSS_calls": cb["rss_forward_calls"], "CPU_RSS_rows": cb["rss_forward_rows"],
        "reserved_partition_score_rows": fb["V_calibration_test_rows"],
    }
    timings_valid = cost["status"] == "complete" and set(cost["batches"]) == {"1", "256"}
    for size in ("1", "256"):
        batch = cost["batches"][size]
        timings_valid &= batch["sample_ids"] == context["timing_batches"][size]["sample_ids"]
        timings_valid &= len(batch["execution_order"]) == 75
        timings_valid &= all(sorted(order) == sorted(context["model_specs"]) for order in batch["execution_order"])
        for label in context["model_specs"]:
            durations = np.asarray(batch[label]["durations_ms"], dtype=float)
            timings_valid &= (durations.shape == (75,) and bool(np.isfinite(durations).all())
                              and bool((durations > 0).all()) and len(batch[label]["blocks"]) == 3)
    rss_valid = set(cost["rss"]) == set(context["model_specs"]) and cb["rss_processes"] == 3
    rss_pids = []
    for name, record in cost["rss"].items():
        rss_pids.append(record["pid"])
        rss_valid &= (record["model"] == name and record["threads"] == 4
                      and record["forward_calls"] == 1 and record["forward_rows"] == 256
                      and record["sample_ids"] == context["timing_batches"]["256"]["sample_ids"][0]
                      and all(record[k] > 0 for k in ("rss_after_import_bytes", "rss_after_load_bytes",
                                                     "rss_after_probe_bytes", "model_load_ms")))
    # Fresh subprocesses may legally reuse a PID after exit, so PID uniqueness is not an acceptance rule.
    rss_valid &= all(isinstance(pid, int) and pid > 0 for pid in rss_pids)
    return {
        "exact_frozen_budget": budget == limits and fb["valid"],
        "finite_F_scores_losses": all(v["finite"] for v in f["models"].values()),
        "pattern_loss_bounds": all(v["bounds_valid"] for v in f["models"].values()),
        "frozen_network_states": all(v["state_checks"]["unchanged"] for v in f["models"].values()),
        "complete_CPU_timings_and_order": bool(timings_valid),
        "independent_RSS_probes": bool(rss_valid),
    }


def run_diagnostic(screen_dir, round10_dir, data_dir, output, *, device="cuda", binding_path=None):
    import torch

    root = _no_links(output)
    inputs = [_no_links(p) for p in (screen_dir, round10_dir, data_dir)]
    if root.exists():
        raise FileExistsError("Preserve previous diagnostic output; no overwrite or automatic resume")
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in [REPO, *inputs]):
        raise ValueError("Output must be separate from repository and immutable inputs")
    if (device not in {"cuda", "cuda:0"} or not torch.cuda.is_available()
            or "4090D" not in torch.cuda.get_device_name(0).replace(" ", "").upper()):
        raise ValueError("Real zero-fit diagnosis requires laboratory CUDA0 RTX4090D")
    source = code_fingerprint()
    if source.get("dirty") is not False or source.get("commit") is None:
        raise ValueError("Laboratory execution requires a clean committed checkout")
    context = _bound_inputs(*inputs, binding_path=binding_path)
    return _execute(context, root, device=device, source=source)
