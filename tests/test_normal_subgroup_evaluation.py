"""Synthetic bound archives and scores; no real C/E, models, or training are read."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from ids.model_training_redesign import evaluation as old_evaluation
from ids.normal_subgroup_training import evaluation as ev
from ids.normal_subgroup_training.bindings import sha256
from ids.normal_subgroup_training.selection import metrics_to_json, selection_metrics


def frame(source, normal=100, positive=1, profile="B"):
    labels = np.r_[np.zeros(normal, dtype=int), np.ones(positive, dtype=int)]
    n = len(labels)
    return pd.DataFrame({"source_id": source, "row_id": np.arange(n), "label": labels,
                         "family": np.where(labels, "Attack", "Benign"), "profile": profile,
                         "flow_end_ms": np.arange(n) + 1000, "ip_protocol": 17 if profile == "B" else 6,
                         "orig_port": 123, "resp_port": 443, "duration_seconds": 1.,
                         "orig_pkts": labels * 2, "resp_pkts": np.r_[np.ones(min(2, normal)),
                                                                           np.zeros(n - min(2, normal))],
                         "orig_ip_bytes": 100, "resp_ip_bytes": 200})


def inputs():
    roles, rows = {}, []
    for name, count, profile in (("E_B", 3, "B"), ("E_D", 12, "D")):
        cells = []
        for j in range(count):
            source = f"{profile}_{j:02}"
            rows.append(frame(source, profile=profile))
            cells.append({"source_id": source, "normal": 100, "attack": 1})
        roles[name] = {"cells": cells, "normal": count * 100, "attack": count, "rows": count * 101}
    data = pd.concat(rows, ignore_index=True)
    design = {"seeds": list(ev.SEEDS), "arms": list(ev.ARMS),
              "calibration": {"normal_rows": 5511, "one_based_sorted_rank": 5457,
                              "decision": "strict_greater", "alpha": .01, "times_per_model": 1},
              "adoption": {"each_seed_B_TP": 3, "each_seed_D_TP": 12}}
    return data, design, {"roles": roles}


def reference(data, name="protocol_blind_s17", false_positives=True):
    scores = data.orig_pkts.to_numpy(dtype=float)
    if false_positives:
        scores = scores + data.resp_pkts.to_numpy() * 2
    return old_evaluation.evaluate(data, scores, 0., model_id=name)[2]


def metrics(improved=False):
    f = pd.concat([frame("F_A", 2, 1, "A"), frame("F_B", 2, 1, "B")], ignore_index=True)
    v = pd.concat([frame("V_A", 10, 1, "A"), frame("V_B", 10, 1, "B")], ignore_index=True)
    normal = np.r_[np.arange(0, 20, 2), np.arange(1, 20, 2)] if improved else np.arange(20)
    scores = np.r_[normal[:10], 20, normal[10:], 21]
    return selection_metrics([0, 1, 20, 0, 1, 21], f, scores, v, ["A_0"] * 10 + ["B_0"] * 10,
                             ["A_0", "B_0"])


def evaluate(scores=None, historical=None):
    data, design, protocol = inputs()
    return ev.evaluate_scores(np.zeros(5511), data,
                              data.orig_pkts.to_numpy() if scores is None else scores,
                              reference(data), design=design, protocol_design=protocol,
                              model_id="transport_target_s17", historical_decisions=historical)


def pair_reports(*, main_X=0, control_X=1, fallback=False):
    reports, selections = {}, {}
    for seed in ev.SEEDS:
        for arm in ev.ARMS:
            name = f"{arm}_s{seed}"
            reports[name] = {"excess_X": main_X if arm == "transport_target" else control_X,
                             "development_pass": main_X == 0 if arm == "transport_target" else control_X == 0}
            selections[name] = {"metrics": metrics(arm == "transport_target" and not fallback),
                                "baseline_metrics": metrics(),
                                "selected_step": 139 if arm == "transport_target" and not fallback else 0}
    return reports, selections


def test_original_C_rank_and_strict_equality_and_identity_protection():
    result = evaluate()
    assert result["calibration"]["rank"] == 5457
    assert result["calibration"]["threshold"] == 0
    assert result["development_pass"] and result["excess_X"] == 0
    assert not result["decision_frame"].loc[result["decision_frame"].score.eq(0), "alert"].any()
    assert not result["identity_changes"]["tp_lost"]
    assert len(result["identity_changes"]["fp_removed"]) == 30


def test_one_lost_attack_identity_rejects_and_is_saved():
    data, _, _ = inputs()
    scores = data.orig_pkts.to_numpy(dtype=float)
    scores[100] = 0
    result = evaluate(scores)
    assert not result["development_pass"]
    assert result["identity_changes"]["tp_lost"][0]["source_id"] == "B_00"
    assert result["identity_changes"]["tp_lost"][0]["row_id"] == 100


def test_one_cell_over_budget_cannot_be_offset_by_other_cells():
    data, _, _ = inputs()
    scores = data.orig_pkts.to_numpy(dtype=float)
    scores[:2] = 1
    result = evaluate(scores)
    assert sum(c["fp"] for c in result["cells"]) < sum(c["fp_budget"] for c in result["cells"])
    assert result["excess_X"] == 1 and not result["development_pass"]


def test_complete_old_references_report_cell_and_identity_gains_without_recalibration():
    data, _, _ = inputs()
    previous = reference(data, "historical_strong", false_positives=False)
    previous.loc[100, "alert"] = False
    result = evaluate(historical={"historical_strong": previous})
    comparison = result["reference_comparisons"]["historical_strong"]
    assert len(comparison["cells"]) == 15
    assert comparison["identity_changes"]["tp_added"][0]["row_id"] == 100
    assert comparison["cells"][0]["net_tp"] == 1


@pytest.mark.parametrize("change", ["identity", "denominator", "truth"])
def test_mismatched_frozen_reference_or_original_denominators_fail(change):
    data, design, protocol = inputs()
    old = reference(data)
    if change == "identity":
        old.loc[0, "row_id"] = 99999
    elif change == "denominator":
        protocol["roles"]["E_B"]["cells"][0]["normal"] += 1
    else:
        old.loc[0, "duration_seconds"] = 99
    with pytest.raises(ValueError):
        ev.evaluate_scores(np.zeros(5511), data, data.orig_pkts, old, design=design,
                           protocol_design=protocol, model_id="transport_target_s17")


def test_mechanism_uses_exact_fraction_receipts_not_display_values():
    a, c, b = [metrics_to_json(m) for m in (metrics(True), metrics(), metrics())]
    for value in (a, c, b):
        value["tail"]["H"]["value"] = .123456
    result = ev.compare_mechanism(a, c, b, selected_step=139)
    assert result["passed"] and result["H_transport"] < result["H_control"]


def test_source_tail_vs_control_condition_is_independent_of_baseline_protection():
    a, c, b = metrics(True), metrics(), metrics()
    from fractions import Fraction
    for value in (a, b):
        value["source_tails"]["B"]["H_q"]["0.99"] = Fraction(1, 100)
    result = ev.compare_mechanism(a, c, b, selected_step=139)
    assert result["checks"]["transport_guards_pass"]
    assert not result["checks"]["each_source_H_q_not_above_control"]
    assert not result["passed"]


def test_complete_does_not_mean_adopt_and_zero_X_tie_can_retain_valid_control():
    reports, selections = pair_reports(main_X=0, control_X=0)
    result = ev.compare_models(reports, selections, implementation_contract_pass=True)
    assert result["completed"] and result["status"] == "complete"
    assert not result["adoption_pass"] and result["retained_development_arm"] == "identity_target"
    reports["identity_target_s43"]["development_pass"] = False
    assert ev.compare_models(reports, selections, implementation_contract_pass=True)["retained_development_arm"] == "R30"


def test_fallback_is_not_mechanism_success_and_one_seed_failure_is_not_averaged():
    reports, selections = pair_reports(fallback=True)
    assert not ev.compare_models(reports, selections, implementation_contract_pass=True)["adoption_pass"]
    reports, selections = pair_reports()
    assert ev.compare_models(reports, selections, implementation_contract_pass=True)["adoption_pass"]
    reports["transport_target_s43"]["development_pass"] = False
    assert not ev.compare_models(reports, selections, implementation_contract_pass=True)["adoption_pass"]
    del reports["identity_target_s43"]
    with pytest.raises(ValueError, match="six"):
        ev.compare_models(reports, selections, implementation_contract_pass=True)


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(metrics_to_json(value), allow_nan=False), encoding="utf-8")


def bound(path, relative=None):
    result = {"sha256": sha256(path), "bytes": Path(path).stat().st_size}
    if relative is not None:
        result["path"] = relative
    return result


def frozen_fixture(tmp_path):
    archive, preflight, models, output = (tmp_path / n for n in ("archive", "preflight", "models", "evaluation"))
    data, design, protocol = inputs()
    design_path, bindings_path, freeze_path = (tmp_path / n for n in ("design.json", "bindings.json", "freeze.json"))
    write(design_path, design)
    write(archive / "protocol.json", {"design": protocol})
    (archive / "inputs").mkdir()
    frame("C", 5511, 0).assign(resp_pkts=0).to_parquet(archive / "inputs/C.parquet", index=False)
    for role, prefix in (("E_B", "B_"), ("E_D", "D_")):
        data.loc[data.source_id.str.startswith(prefix)].to_parquet(archive / f"inputs/{role}.parquet", index=False)
    reference(data, "historical_strong").to_parquet(archive / "inputs/reference_decisions.parquet", index=False)
    (archive / "reports").mkdir()
    for seed in ev.SEEDS:
        reference(data, f"protocol_blind_s{seed}").to_parquet(
            archive / f"reports/protocol_blind_s{seed}-decisions.parquet", index=False)
    files = {p.relative_to(archive).as_posix(): bound(p) for p in archive.rglob("*") if p.is_file()}
    write(bindings_path, {"files": files})
    write(preflight / "payload.json", {"synthetic": True})
    write(preflight / "output_manifest.json", {"files": {"payload.json": bound(preflight / "payload.json")}})
    manifest_hash = sha256(preflight / "output_manifest.json")
    write(preflight / "status.json", {"status": "zero_update_preflight_complete",
                                       "output_manifest": {"sha256": manifest_hash}})
    frozen = {"status": "all_six_artifacts_frozen", "implementation_contract_pass": True,
              "design_sha256": sha256(design_path), "bindings_sha256": sha256(bindings_path),
              "preflight_manifest_sha256": manifest_hash,
              "paired_contracts": {str(s): {"passed": True} for s in ev.SEEDS}, "models": {}}
    for name in ev.MODEL_IDS:
        arm, seed = name.rsplit("_s", 1)
        folder = models / name
        write(folder / "selected_metrics.json", metrics(arm == "transport_target"))
        write(folder / "baseline_metrics.json", metrics())
        (folder / "model.pt").write_bytes(name.encode())
        frozen["models"][name] = {"arm": arm, "seed": int(seed), "selected_step": 139 if arm == "transport_target" else 0,
                                   **{key: bound(folder / filename, f"{name}/{filename}") for key, filename in
                                      (("artifact", "model.pt"), ("selected_metrics", "selected_metrics.json"),
                                       ("baseline_metrics", "baseline_metrics.json"))}}
    write(freeze_path, frozen)
    return {"archive": archive, "preflight_dir": preflight, "models_root": models,
            "freeze_path": freeze_path, "output_dir": output, "design_path": design_path,
            "bindings_path": bindings_path}


def test_incomplete_freeze_cannot_read_C_E_or_load_a_model(tmp_path, monkeypatch):
    freeze_path = tmp_path / "freeze.json"
    write(freeze_path, {"status": "started", "models": {}})
    def forbidden(*args, **kwargs):
        raise AssertionError("C/E or model access occurred before complete freeze")
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    with pytest.raises(ValueError, match="six"):
        ev.evaluate_frozen(tmp_path / "absent", tmp_path / "absent", tmp_path / "absent", freeze_path,
                           tmp_path / "output", design_path=tmp_path / "absent", bindings_path=tmp_path / "absent",
                           loader=forbidden)


def test_synthetic_frozen_end_to_end_scores_and_calibrates_each_artifact_once(tmp_path, monkeypatch):
    args = frozen_fixture(tmp_path)
    calls = {"loads": 0, "scores": 0, "calibration": 0}
    calibrate = old_evaluation.calibrate
    def tracked(scores, **kwargs):
        calls["calibration"] += 1
        return calibrate(scores, **kwargs)
    monkeypatch.setattr(old_evaluation, "calibrate", tracked)
    def loader(path):
        calls["loads"] += 1
        identity = "identity_target" in str(path)
        class Predictor:
            def score(self, f):
                calls["scores"] += 1
                assert "label" not in f and "source_id" not in f
                return f.orig_pkts.to_numpy(dtype=float) + (2 * f.resp_pkts.to_numpy() if identity else 0)
        return Predictor()
    result = ev.evaluate_frozen(**args, loader=loader)
    assert result["status"] == "complete" and result["adoption_pass"]
    assert calls == {"loads": 6, "scores": 12, "calibration": 6}
    assert (args["output_dir"] / "E.parquet").exists()
    for name in ev.MODEL_IDS:
        for filename in ("C-scores.parquet", "E-scores.parquet", "decisions.parquet", "calibration.json"):
            assert (args["output_dir"] / name / filename).exists()
    with pytest.raises(ValueError, match="recalibrate"):
        ev.evaluate_frozen(**args, loader=loader)
    assert calls["calibration"] == 6


@pytest.mark.parametrize("changed", ["artifact", "preflight", "bindings", "historical"])
def test_changed_frozen_payload_rejected_before_any_parquet_read(tmp_path, monkeypatch, changed):
    args = frozen_fixture(tmp_path)
    paths = {"artifact": args["models_root"] / "transport_target_s17/model.pt",
             "preflight": args["preflight_dir"] / "payload.json", "bindings": args["bindings_path"],
             "historical": args["archive"] / "inputs/reference_decisions.parquet"}
    paths[changed].write_bytes(paths[changed].read_bytes() + b"changed")
    def forbidden(*unused, **kwargs):
        raise AssertionError("Unverified C/E was read")
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    with pytest.raises(ValueError):
        ev.evaluate_frozen(**args, loader=forbidden)
