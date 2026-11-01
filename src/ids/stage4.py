"""Matched selection, independent selection diagnostics and two-data development.

The orchestration never evaluates calibration/test. Dataset preparation is a
separate operation that creates these reserved partitions before this workflow.
"""
from __future__ import annotations

import copy
import html
import itertools
import shutil
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .common import canonical_hash, code_fingerprint, environment_snapshot, read_json, sha256_file, utc_now, write_json
from .data import CATEGORICAL, FEATURES, load_partition
from .runner import run_suite, verify_development_inputs, verify_run

REPO = Path(__file__).resolve().parents[2]


def validate_config(config):
    if config.get("profile") not in {"research", "smoke"}:
        raise ValueError("Stage 4 needs a research or smoke profile")
    seeds = config.get("seeds")
    if not seeds or any(type(x) is not int for x in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Distinct integer training seeds required")
    if set(config["models"]) != {"rf", "mlp"}:
        raise ValueError("Stage 4 compares RF and MLP")
    if config["max_fpr"] != .01:
        raise ValueError("Stage 4 fixes max_fpr at 0.01")
    for key in ("threads", "benchmark_rounds"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Invalid {key}")
    selection = config["selection"]
    if selection.get("pool_policy") != "stratified_rows":
        raise ValueError("Stage 4 requires the fixed train-family-stratified selection pool")
    for key in ("k", "n_resamples", "pool_max_rows", "outer_repeats"):
        if type(selection[key]) is not int or selection[key] < 2:
            raise ValueError(f"Invalid selection.{key}")
    if not .5 <= selection["outer_fraction"] < 1 or not 0 < selection["redundancy_threshold"] < 1:
        raise ValueError("Invalid resampling/redundancy setting")
    if config["profile"] == "research":
        if seeds != [17, 29, 43] or config["benchmark_rounds"] != 3:
            raise ValueError("Research requires seeds 17/29/43 and three CPU rounds")
        if selection["k"] != 24 or selection["n_resamples"] != 50 or selection["outer_repeats"] != 5:
            raise ValueError("Research requires k=24, 50 inner and 5 outer repetitions")


def conditions(config, dataset):
    s = config["selection"]
    mi = {"method": "mutual_info", "k": s["k"], "selection_seed": s["selection_seed"],
          "pool_policy": s["pool_policy"]}
    stable = {**mi, "method": "stable_mutual_info", "n_resamples": s["n_resamples"]}
    result = {"all": {"method": "all", "selection_seed": s["selection_seed"], "pool_policy": s["pool_policy"]},
              "mi": mi, "stable": stable}
    if dataset == "unsw":
        result.update(mi_redundant={**mi, "redundancy_threshold": s["redundancy_threshold"]},
                      stable_redundant={**stable, "redundancy_threshold": s["redundancy_threshold"]})
    elif dataset != "cicids":
        raise ValueError("Unknown stage-4 dataset")
    return result


def make_suite(config, dataset):
    return {"suite_name": f"{config['stage_name']}-{dataset}",
            "purpose": "stage4_matched_selection_validation_only", "seeds": config["seeds"],
            "max_fpr": config["max_fpr"], "threads": config["threads"],
            "benchmark_rounds": config["benchmark_rounds"],
            "models": [{"model": model, "features": copy.deepcopy(setting),
                        "hyperparameters": copy.deepcopy(hp)}
                       for model, hp in config["models"].items()
                       for setting in conditions(config, dataset).values()]}


def feature_schema(manifest):
    return {"name": manifest["schema_version"], "features": manifest["features"],
            "categorical": manifest.get("categorical_features", list(CATEGORICAL)),
            "discrete": manifest.get("discrete_features", list(CATEGORICAL))}


def group_column(frame):
    return next((x for x in ("group_hash", "group_id", "feature_hash") if x in frame), None)


def sample_groups(frame, seed, *, max_rows=None, fraction=None):
    """Seeded whole-group sample, preserving input order; soft row ceiling.

    Take a prefix of a random group permutation. The final whole group may
    cross the row ceiling. Unequal group sizes affect inclusion probabilities;
    labels and validation outcomes never decide membership.
    """
    column = group_column(frame)
    if column is None or frame[column].isna().any():
        raise ValueError("Selection requires nonmissing group identities")
    counts = frame.groupby(column, sort=True).size()
    if len(counts) < 2:
        raise ValueError("At least two independent groups required")
    order = np.random.default_rng(seed).permutation(len(counts))
    if fraction is not None:
        take = max(2, int(np.floor(len(counts) * fraction)))
    elif max_rows is not None:
        cumulative = np.cumsum(counts.to_numpy()[order])
        take = min(len(counts), int(np.searchsorted(cumulative, max_rows, side="left")) + 1)
    else:
        raise ValueError("Provide a group fraction or soft row ceiling")
    chosen = counts.index[order[:take]]
    selected = frame.loc[frame[column].isin(chosen)].copy()
    if set(selected["label"].unique()) != {0, 1}:
        raise ValueError("Fixed selection sample lacks both classes; do not retry seeds")
    return selected, {"seed": seed, "group_column": column, "n_source_rows": len(frame),
                      "n_source_groups": len(counts), "n_rows": len(selected), "n_groups": len(chosen),
                      "soft_max_rows": max_rows, "fraction_of_groups": fraction,
                      "sample_ids_sha256": canonical_hash(selected["sample_id"].astype(str).tolist()),
                      "groups_sha256": canonical_hash(sorted(map(str, chosen))),
                      "label_counts": {str(k): int(v) for k, v in selected["label"].value_counts().items()},
                      "family_counts": {str(k): int(v) for k, v in selected["attack_cat"].value_counts().items()}}


def stratified_selection_pool(train, max_rows, seed):
    """Bounded train-only row pool with family coverage and original group IDs.

    Reserve one record per observed training family; allocate the remaining
    slots in proportion to remaining family counts, using largest remainders.
    Sample once without replacement within each family, then restore input row
    order. A later group subsample includes every *pool* row in each chosen
    group; the source train group may have additional rows outside this pool.
    """
    column = group_column(train)
    if column is None or train[column].isna().any() or train["attack_cat"].isna().any():
        raise ValueError("Selection pool needs nonmissing train group and family metadata")
    if type(max_rows) is not int or max_rows < 2 or not train["sample_id"].is_unique:
        raise ValueError("Invalid selection pool size or sample identity")
    counts = train["attack_cat"].value_counts().sort_index()
    target = min(max_rows, len(train))
    if target < len(counts):
        raise ValueError("Pool budget cannot cover observed training families")
    remaining_counts = counts.to_numpy(dtype=np.int64) - 1
    available = target - len(counts)
    ideal = (remaining_counts / remaining_counts.sum() * available
             if remaining_counts.sum() else np.zeros(len(counts)))
    quotas = np.ones(len(counts), dtype=np.int64) + np.floor(ideal).astype(np.int64)
    leftover = target - int(quotas.sum())
    order = np.lexsort((np.arange(len(counts)), -(ideal - np.floor(ideal))))
    quotas[order[:leftover]] += 1
    rng = np.random.default_rng(seed)
    families = train["attack_cat"].to_numpy()
    indexes = []
    for family, quota in zip(counts.index, quotas):
        positions = np.flatnonzero(families == family)
        indexes.extend(rng.choice(positions, size=int(quota), replace=False).tolist())
    pool = train.iloc[sorted(indexes)].copy()
    if set(pool["label"].unique()) != {0, 1}:
        raise ValueError("Selection pool lacks both classes")
    group_counts = train.groupby(column, sort=True).size()
    pool_counts = pool.groupby(column, sort=True).size()
    partial = pool_counts < group_counts.loc[pool_counts.index]
    record = {"policy": "stratified_rows", "seed": seed, "group_column": column,
              "n_source_rows": len(train), "n_source_groups": len(group_counts),
              "n_rows": len(pool), "n_groups": len(pool_counts), "max_rows": max_rows,
              "partial_source_groups": int(partial.sum()), "complete_source_groups": int((~partial).sum()),
              "all_source_group_rows_retained": not bool(partial.any()),
              "sample_ids_sha256": canonical_hash(pool["sample_id"].astype(str).tolist()),
              "groups_sha256": canonical_hash(sorted(map(str, pool_counts.index))),
              "label_counts": {str(k): int(v) for k, v in pool["label"].value_counts().items()},
              "family_counts": {str(k): int(v) for k, v in pool["attack_cat"].value_counts().items()},
              "source_family_counts": {str(k): int(v) for k, v in counts.items()},
              "family_group_counts": {str(k): int(v) for k, v in pool.groupby("attack_cat")[column].nunique().items()},
              "missing_families": sorted(set(counts.index) - set(pool["attack_cat"])),
              "scope": "train-family-stratified rows; original group IDs retained; resampling keeps pool groups intact"}
    return pool, record


def cached_json(path, identity, compute, resume):
    checksum = path.with_suffix(path.suffix + ".sha256")
    if path.exists():
        if not resume or not checksum.is_file() or checksum.read_text().strip() != sha256_file(path):
            raise ValueError(f"Unverified existing selection artifact: {path.name}")
        value = read_json(path)
        if value["identity"] != identity:
            raise ValueError("Selection cache data, source or settings changed")
        print(f"Verified selection: {path.name}", flush=True)
        return value["result"]
    result = compute()
    write_json(path, {"identity": identity, "result": result})
    checksum.write_text(sha256_file(path) + "\n", encoding="ascii")
    return result


def stability_summary(selections, names):
    sets = [set(s) for s in selections]
    pairs = [len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2)]
    z = np.asarray([[name in s for name in names] for s in sets], dtype=float)
    pbar = float(z.mean())
    # Nogueira's finite-repeat, chance-corrected estimator; not a significance test.
    corrected = None if pbar in (0, 1) or len(z) < 2 else float(1 - z.var(axis=0, ddof=1).mean() / (pbar * (1 - pbar)))
    return {"n_outer_repeats": len(sets), "pairwise_jaccard": pairs,
            "mean_pairwise_jaccard": float(np.mean(pairs)) if pairs else None,
            "chance_corrected_stability": corrected,
            "selection_frequency": {name: float(z[:, i].mean()) for i, name in enumerate(names)},
            "scope": "selection under train group perturbation; no performance robustness or significance claim"}


def prepare_selections(data_dir, root, config, dataset, resume):
    from .selection import select_features
    manifest = verify_development_inputs(data_dir, splits=("train",))
    schema = feature_schema(manifest)
    # Only train is opened here, never validation/calibration/test.
    train = load_partition(data_dir, "train")
    settings = config["selection"]
    pool, pool_record = stratified_selection_pool(train, settings["pool_max_rows"], settings["pool_seed"])
    identity = {"dataset_manifest_sha256": sha256_file(Path(data_dir) / "manifest.json"),
                "source_sha256": code_fingerprint()["source_sha256"], "selection_settings": settings,
                "schema": schema, "pool": pool_record}
    selections, outer_sets = {}, {}
    for name, setting in conditions(config, dataset).items():
        print(f"{dataset}: selector {name} on {len(pool)} rows ({pool_record['n_groups']} groups)", flush=True)
        result = cached_json(root / f"{name}.json", {**identity, "features": setting},
                             lambda setting=setting: select_features(train, setting, config["seeds"][0],
                                 feature_schema=schema, selection_frame=pool), resume)
        selections[canonical_hash(setting)] = result
        if name != "all":
            outer_sets[name] = []
    outer_records = []
    for index in range(settings["outer_repeats"]):
        outer_pool, outer_record = sample_groups(pool, settings["outer_seed"] + index,
                                                fraction=settings["outer_fraction"])
        outer_records.append({"outer": outer_record, "pool": pool_record})
        outer_record["missing_families"] = sorted(set(pool["attack_cat"]) - set(outer_pool["attack_cat"]))
        for name, setting in conditions(config, dataset).items():
            if name == "all":
                continue
            print(f"{dataset}: independent selection diagnostic {index+1}/{settings['outer_repeats']} {name}", flush=True)
            result = cached_json(root / "outer" / f"{index:02d}-{name}.json",
                                 {**identity, "features": setting, "outer": outer_records[-1]},
                                 lambda setting=setting: select_features(outer_pool, setting, config["seeds"][0],
                                     feature_schema=schema), resume)
            outer_sets[name].append(result["selected_features"])
    diagnostics = {"schema_version": 1, "identity": identity, "pool": pool_record,
                   "outer_samples": outer_records,
                   "methods": {name: {**stability_summary(values, schema["features"]), "selections": values}
                               for name, values in outer_sets.items()},
                   "scope": "fixed train selection-pool perturbations only; model seeds are not data perturbations"}
    write_json(root / "diagnostics.json", diagnostics)
    return selections, diagnostics


def seal_files(root, exclude=()):
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name not in exclude and p.suffix not in {".log", ".tmp"}}


def verify_files(root, files):
    for name, digest in files.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or sha256_file(path) != digest:
            raise ValueError(f"Sealed artifact changed: {name}")


def prepare_plan(root, data_dirs, config, device, resume):
    manifests = {name: verify_development_inputs(path) for name, path in data_dirs.items()}
    for manifest in manifests.values():
        synthetic = manifest["data_kind"] == "synthetic"
        if synthetic != (config["profile"] == "smoke"):
            raise ValueError("Research requires real data; smoke requires synthetic fixtures")
    if manifests["unsw"]["features"] != list(FEATURES):
        raise ValueError("First dataset must retain the UNSW field contract")
    if not manifests["cicids"]["schema_version"].startswith("cicids2017-"):
        raise ValueError("Second dataset must be CNS2022 CICIDS2017, not another UNSW representation")
    source, environment = code_fingerprint(), environment_snapshot()
    environment.pop("recorded_at", None)
    identity = {"config": config, "datasets": {name: sha256_file(path / "manifest.json") for name, path in data_dirs.items()},
                "source_sha256": source["source_sha256"], "environment_sha256": canonical_hash(environment), "device": device}
    path = root / "STAGE4_PLAN.json"
    if path.exists():
        if not resume or read_json(path)["identity"] != identity:
            raise ValueError("Cannot resume changed source/data/config/device/environment")
        verify_files(root, read_json(path)["snapshot_files"])
        if (root / "STAGE4_COMPLETE.json").exists():
            verify_completion(root)
            return False
        return True
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("New experiment directory must be empty")
    for name, digest in source["files"].items():
        target = root / "protocol" / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, target)
        if sha256_file(target) != digest:
            raise ValueError("Source changed while snapshotting")
    for name in ("STAGE4_PROTOCOL.md", "STAGE4_LAB.md", "STAGE4_VALIDATION.md", "DATA.md", "PROTOCOL.md"):
        shutil.copyfile(REPO / "docs" / name, root / "protocol" / name)
    write_json(path, {"schema_version": 1, "identity": identity, "created_at": utc_now(),
                      "source": source, "environment": environment,
                      "expected_runs": sum(len(make_suite(config, d)["models"]) * len(config["seeds"]) for d in data_dirs),
                      "snapshot_files": seal_files(root), "scope": "validation development; no final evaluation"})
    return True


def verify_study(study, suite):
    expected = len(suite["models"]) * len(suite["seeds"])
    if study["n_runs"] != expected or len(study["groups"]) != len(suite["models"]):
        raise ValueError("Incomplete stage-4 model coverage")
    expected_conditions = {(m["model"], canonical_hash(m["features"])) for m in suite["models"]}
    actual_conditions = {(g["spec"]["model"], canonical_hash(g["spec"]["feature_config"])) for g in study["groups"]}
    if actual_conditions != expected_conditions:
        raise ValueError("Study conditions differ from planned model/feature matrix")
    for group in study["groups"]:
        if group["seeds"] != sorted(suite["seeds"]):
            raise ValueError("Incomplete training seeds")
        if len(group["cpu_benchmarks"]) != suite["benchmark_rounds"]:
            raise ValueError("Incomplete CPU rounds")
        for measurement in group["cpu_benchmarks"]:
            if measurement["seeds"] != sorted(suite["seeds"]):
                raise ValueError("Incomplete CPU seed coverage")
        if {b["spec"].get("measurement_round", 1) for b in group["cpu_benchmarks"]} != set(range(1, suite["benchmark_rounds"] + 1)):
            raise ValueError("CPU round identities differ from plan")
        if suite["benchmark_rounds"] > 1 and not any(
                r["status"] == "complete" and r["seeds"] == sorted(suite["seeds"])
                for r in group["repeated_benchmarks"]):
            raise ValueError("CPU round settings/environments disagree")
    if len(study["feature_budget_comparisons"]) != len(suite["models"]) - 2:
        raise ValueError("Missing all-field comparison coverage")
    if any(c["status"] != "paired" or c["n_pairs"] != len(suite["seeds"])
           for c in study["feature_budget_comparisons"]):
        raise ValueError("Missing matched all-field baseline")


def verify_completion(root):
    root = Path(root)
    completion, plan = read_json(root / "STAGE4_COMPLETE.json"), read_json(root / "STAGE4_PLAN.json")
    config = plan["identity"]["config"]
    validate_config(config)
    expected = 16 * len(config["seeds"])
    if (completion.get("phase") != "validation_development" or completion.get("final_evaluation_performed") is not False
            or completion.get("profile") != config["profile"] or completion.get("n_runs") != expected
            or plan.get("expected_runs") != expected or completion.get("benchmark_rounds") != config["benchmark_rounds"]):
        raise ValueError("Stage-4 completion disagrees with frozen plan")
    required = {"STAGE4_PLAN.json", "reports/index.html"}
    for dataset in ("unsw", "cicids"):
        required.update({f"reports/{dataset}/study.json", f"reports/{dataset}/study.html",
                         f"reports/{dataset}/validation-records.html", f"{dataset}/selection/diagnostics.json",
                         f"reports/{dataset}/method-comparisons.json"})
    if not required.issubset(completion["files"]):
        raise ValueError("Stage-4 required artifacts are not sealed")
    verify_files(root, completion["files"])
    for dataset in ("unsw", "cicids"):
        study = read_json(root / "reports" / dataset / "study.json")
        if study["dataset_manifest_sha256"] != plan["identity"]["datasets"][dataset]:
            raise ValueError("Study dataset differs from stage-4 plan")
        verify_study(study, make_suite(config, dataset))
        if any(g["spec"]["source_sha256"] != plan["identity"]["source_sha256"] for g in study["groups"]):
            raise ValueError("Source changed between planned and executed studies")
        if read_json(root / "reports" / dataset / "method-comparisons.json") != paired_method_comparisons(study, config, dataset):
            raise ValueError("Method comparisons differ from verified study")
    return completion


def paired_method_comparisons(study, config, dataset):
    """Within-model and within-seed effects, without pooling different datasets."""
    lookup = {(g["spec"]["model"], canonical_hash(g["spec"]["feature_config"])): g for g in study["groups"]}
    settings = conditions(config, dataset)
    pairs = [("stable", "mi")]
    if dataset == "unsw":
        pairs += [("mi_redundant", "mi"), ("stable_redundant", "stable"), ("stable_redundant", "mi_redundant")]
    output = []
    for model in config["models"]:
        for candidate, baseline in pairs:
            left = lookup[model, canonical_hash(settings[candidate])]
            right = lookup[model, canonical_hash(settings[baseline])]
            by_seed = {r["seed"]: r for r in right["runs"]}
            rows = []
            for run in left["runs"]:
                reference = by_seed[run["seed"]]
                lm, rm = run["metrics"], reference["metrics"]
                families = {r["category"]: r for r in rm["attack_categories"]}
                family_rows = []
                for family in lm["attack_categories"]:
                    other = families[family["category"]]
                    if family["n_attacks"] != other["n_attacks"]:
                        raise ValueError("Paired methods have different family support")
                    family_rows.append({"category": family["category"], "n_attacks": family["n_attacks"],
                                        "detected_delta": family["detected"] - other["detected"],
                                        "recall_delta_pp": 100 * (family["recall"] - other["recall"])
                                        if family["recall"] is not None and other["recall"] is not None else None})
                rows.append({"seed": run["seed"], "attack_recall_delta_pp": 100 * (lm["attack_recall"] - rm["attack_recall"]),
                             "normal_fpr_delta_pp": 100 * (lm["normal_fpr"] - rm["normal_fpr"]),
                             "model_size_ratio": lm["model_file_bytes"] / rm["model_file_bytes"], "families": family_rows})
            values = [r["attack_recall_delta_pp"] for r in rows]
            output.append({"model": model, "candidate": candidate, "baseline": baseline, "per_seed": rows,
                           "mean_recall_delta_pp": float(np.mean(values)),
                           "std_recall_delta_pp": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                           "scope": "paired training seeds, fixed validation; descriptive, not independent test inference"})
    return {"dataset": dataset, "comparisons": output}


def render_overview(root, studies, diagnostics, comparisons):
    sections = []
    for dataset, study in studies.items():
        rows = []
        for g in study["groups"]:
            method = g["spec"]["feature_method"] + (" + redundancy" if g["spec"]["feature_config"].get("redundancy_threshold") else "")
            rows.append(f"<tr><td>{html.escape(g['spec']['model'])}</td><td>{html.escape(method)}</td>"
                        f"<td>{g['spec']['raw_feature_budget']}</td><td>{100*g['metrics']['attack_recall']['mean']:.3f}%</td>"
                        f"<td>{g['metrics']['model_file_bytes']['mean']:.0f}</td></tr>")
        drows = []
        for name, value in diagnostics[dataset]["methods"].items():
            corrected = value["chance_corrected_stability"]
            drows.append(f"<tr><td>{html.escape(name)}</td><td>{value['mean_pairwise_jaccard']:.4f}</td>"
                         f"<td>{corrected:.4f}</td></tr>" if corrected is not None else
                         f"<tr><td>{html.escape(name)}</td><td>{value['mean_pairwise_jaccard']:.4f}</td><td>undefined</td></tr>")
        paired_rows = [f"<tr><td>{html.escape(p['model'])}</td><td>{html.escape(p['candidate'])} − {html.escape(p['baseline'])}</td>"
                       f"<td>{p['mean_recall_delta_pp']:+.3f}</td><td>{', '.join(format(r['attack_recall_delta_pp'], '+.3f') for r in p['per_seed'])}</td></tr>"
                       for p in comparisons[dataset]["comparisons"]]
        n_seeds = len({s for g in study["groups"] for s in g["seeds"]})
        sections.append(f"<section><h2>{html.escape(dataset)}</h2><p><strong>数据来源：{html.escape(study['data_kind'])}；"
                        f"训练种子：{n_seeds}；主运行：{study['n_runs']}</strong></p><p><a href='{dataset}/study.html'>模型、家族、CPU计时与配对差异</a> · "
                        f"<a href='{dataset}/validation-records.html'>逐样本记录</a> · <a href='../{dataset}/selection/diagnostics.json'>完整选择诊断</a> · "
                        f"<a href='{dataset}/method-comparisons.json'>方法配对原始数值与家族差异</a></p>"
                        "<div class='scroll'><table><tr><th>模型</th><th>选择方法</th><th>字段</th><th>Recall@1%FPR</th><th>部署字节</th></tr>"
                        + "".join(rows) + "</table></div><h3>同模型、同种子的选择方法差异</h3><div class='scroll'><table>"
                        "<tr><th>模型</th><th>候选 − 参照</th><th>平均召回变化/pp</th><th>逐种子变化/pp</th></tr>"
                        + "".join(paired_rows) + "</table></div><h3>独立训练组扰动下的选集稳定性</h3><div class='scroll'><table>"
                        "<tr><th>方法</th><th>平均Jaccard</th><th>机会校正稳定性</th></tr>" + "".join(drows) + "</table></div></section>")
    doc = """<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>TraceLight Stage 4</title><style>*{box-sizing:border-box}body{margin:36px auto;max-width:1100px;padding:0 22px;background:#0b111a;color:#edf5f6;font:16px/1.7 system-ui}a{color:#6ee7df}section{border-top:1px solid #344453;padding:22px 0}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:9px;border-bottom:1px solid #344453;white-space:nowrap}p{overflow-wrap:anywhere}</style>
<h1>稳定选择与第二数据环境</h1><p>验证集开发实验。两数据集独立训练与比较，不合并为同一总体，也不代表直接跨域迁移。选集扰动稳定性不等于预测性能扰动稳定性；模型种子一致性与独立外层诊断分别呈现。</p>"""
    notice = ("<p style='border-left:4px solid #eab55b;padding-left:14px;color:#f5d599'><strong>合成工程预检（synthetic）："
              "不用于论文性能结论。</strong> 以下数字仅验证运行、计时、报告及打包链路。</p>"
              if any(s["data_kind"] == "synthetic" for s in studies.values()) else "")
    (root / "reports" / "index.html").write_text(doc + notice + "".join(sections) + "</html>", encoding="utf-8")


def run_stage4(data_dirs, root, config, device="cpu", resume=False, preflight=None):
    from .benchmark_workflow import benchmark_suite
    from .reporting import build_report
    from .study import build_study
    from .study_report import render_study_report
    validate_config(config)
    root = Path(root).resolve()
    data_dirs = {k: Path(v).resolve() for k, v in data_dirs.items()}
    if set(data_dirs) != {"unsw", "cicids"}:
        raise ValueError("Both independent datasets are required")
    if not prepare_plan(root, data_dirs, config, device, resume):
        print("Stage 4 complete: sealed outputs verified; no experiments relaunched.", flush=True)
        return root
    if preflight is not None:
        write_json(root / "preflight.json", preflight)
    studies, diagnostics, comparisons = {}, {}, {}
    for dataset, data_dir in data_dirs.items():
        suite = make_suite(config, dataset)
        with threadpool_limits(limits=config["threads"]):
            selections, diagnostics[dataset] = prepare_selections(data_dir, root / dataset / "selection", config, dataset, resume)
        runs = run_suite(data_dir, suite, root / dataset / "runs", device=device, resume=resume,
                         selection_results=selections)
        run_root = root / dataset / "runs" / suite["suite_name"]
        for measurement_round in range(1, config["benchmark_rounds"] + 1):
            benchmark_suite(run_root, data_dir, root / dataset / "benchmarks" / f"round-{measurement_round}",
                            threads=1, order_seed=20261100 + measurement_round, resume=resume,
                            measurement_round=measurement_round, planned_rounds=config["benchmark_rounds"])
        for run in runs:
            verify_run(run)
        study = build_study(run_root, root / dataset / "benchmarks")
        verify_study(study, suite)
        studies[dataset] = study
        comparisons[dataset] = paired_method_comparisons(study, config, dataset)
        output = root / "reports" / dataset
        write_json(output / "study.json", study)
        write_json(output / "method-comparisons.json", comparisons[dataset])
        render_study_report(study, output / "study.html")
        build_report(run_root, output / "validation-records.html")
        # Include source/split audit metadata, never reserved partition data.
        for name in ("manifest.json", "manifest.sha256", "audit.json"):
            shutil.copyfile(data_dir / name, root / dataset / name)
        manifest = read_json(data_dir / "manifest.json")
        for key in ("provenance", "conflicts"):
            if key in manifest:
                name = manifest[key]["file"]
                if Path(name).name != name or sha256_file(data_dir / name) != manifest[key]["sha256"]:
                    raise ValueError("Dataset provenance/conflict record changed")
                shutil.copyfile(data_dir / name, root / dataset / name)
    render_overview(root, studies, diagnostics, comparisons)
    plan = read_json(root / "STAGE4_PLAN.json")
    if code_fingerprint()["source_sha256"] != plan["identity"]["source_sha256"] or any(
            g["spec"]["source_sha256"] != plan["identity"]["source_sha256"]
            for study in studies.values() for g in study["groups"]):
        raise ValueError("Source changed during stage 4; do not combine these outputs")
    write_json(root / "STAGE4_COMPLETE.json", {"schema_version": 1, "phase": "validation_development",
               "profile": config["profile"], "finished_at": utc_now(), "final_evaluation_performed": False,
               "n_runs": sum(s["n_runs"] for s in studies.values()), "benchmark_rounds": config["benchmark_rounds"],
               "files": seal_files(root, exclude=("STAGE4_COMPLETE.json",))})
    return root
