"""Label-assisted score ceilings; diagnostic outputs never supply deployment cuts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_SCORES = ("score_raw", "score_contact", "score_quality")
HISTORY_SCORES = ("hist_dst_ports_60s", "hist_dst_ips_60s")
RANKING_SCORES = (*MODEL_SCORES, *HISTORY_SCORES)
RARE_FAMILIES = frozenset({"ExternalC2", "C&C", "PureCC", "C2", "Download", "FileDownload"})
NON_ATTACK = frozenset({"Benign", "Unresolved", "Mixed"})
LABEL_GROUPS = ("family", "native_group", "evidence_level", "exposure")
CONTEXT = ("source", "source_role", "group", "scope", "ranking_score", "diagnostic_only")
TARGET_STATS = (
    "target_score", "normal", "cap", "oracle_normal_threshold", "oracle_detected",
    "global_normal_gt", "global_normal_ge", "global_normal_eq", "minimum_fp_to_include",
    "can_include_within_budget", "budget_excess", "service_normal", "service_supported",
    "service_cap", "service_normal_gt", "service_normal_ge", "service_normal_eq",
    "service_minimum_fp_to_include", "service_can_include_within_budget", "service_budget_excess",
)
TARGET_BOOLS = ("oracle_detected", "can_include_within_budget", "service_supported",
                "service_can_include_within_budget")


def _numeric(series, name):
    if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
        raise ValueError(f"{name} must be numeric")
    if pd.api.types.is_complex_dtype(series.dtype) or series.isna().any():
        raise ValueError(f"{name} must contain only finite real values")
    dtype = getattr(series.dtype, "numpy_dtype", series.dtype)
    values = series.to_numpy(dtype=dtype)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite real values")
    return values


def _identity(features, scores, truth):
    for name, frame in (("features", features), ("scores", scores), ("truth", truth)):
        if frame.columns.duplicated().any() or "row_id" not in frame:
            raise ValueError(f"{name} has duplicate columns or missing row_id")
        if not pd.api.types.is_integer_dtype(frame.row_id.dtype):
            raise ValueError(f"{name}.row_id must be integer")
        if frame.row_id.isna().any() or frame.row_id.duplicated().any():
            raise ValueError(f"{name}.row_id must be unique and nonmissing")
        if not np.array_equal(frame.row_id.to_numpy(), features.row_id.to_numpy()):
            raise ValueError(f"{name} row_id differs from features")


def _tail_counts(sorted_normal, target):
    left = np.searchsorted(sorted_normal, target, side="left")
    right = np.searchsorted(sorted_normal, target, side="right")
    return len(sorted_normal) - right, len(sorted_normal) - left, right - left


def _rare_rows(features, scores, truth, positions):
    rows = truth.iloc[positions].reset_index(drop=True).copy()
    rows = rows.rename(columns={c: "truth__" + c for c in CONTEXT if c in rows})
    # UID and long author comments are retained as evidence, never used for ranking.
    if "uid" not in rows:
        rows["uid"] = features.iloc[positions].uid.to_numpy() if "uid" in features else ""
    for column in features.select_dtypes(include="number").columns:
        if column == "row_id":
            continue
        destination = "feature__" + column if column in rows else column
        rows[destination] = features.iloc[positions][column].to_numpy()
    for column in MODEL_SCORES:
        if column in rows:
            rows = rows.rename(columns={column: "truth__" + column})
        rows[column] = scores.iloc[positions][column].to_numpy()
    return rows


def rank_diagnostics(features: pd.DataFrame, scores: pd.DataFrame, truth: pd.DataFrame,
                     scopes: dict, source: dict, group: str, output: Path) -> list[Path]:
    """Rank each score separately within each truth.window and positional Boolean scope.

    All evaluation labels are read here, after predictions have been sealed. This is
    not calibration: no cut or selected score/model may feed the deployed policies.
    Nonfinite rank values fail instead of silently changing the normal denominator.
    """
    _identity(features, scores, truth)
    for key in ("name", "role"):
        if not isinstance(source.get(key), str) or not source[key].strip():
            raise ValueError(f"source.{key} must be a nonempty string")
    if not isinstance(group, str) or not group.strip():
        raise ValueError("group must be a nonempty string")
    for key in ("window", "family"):
        if key not in truth or truth[key].isna().any() or truth[key].astype(str).str.strip().eq("").any():
            raise ValueError(f"truth.{key} must be present and nonmissing")
    if not scopes:
        raise ValueError("At least one scope is required")
    masks = {}
    for name, mask in scopes.items():
        values = np.asarray(mask)
        if not isinstance(name, str) or not name.strip() or values.shape != (len(features),):
            raise ValueError("Scope names and mask lengths must match the source")
        if values.dtype != np.dtype(bool):
            raise ValueError("Scopes must contain Boolean masks without missing values")
        masks[name] = values
    required_features = ("flow_end_ms", "ip_protocol", "resp_port", *HISTORY_SCORES)
    missing = set(required_features) - set(features)
    missing_scores = set(MODEL_SCORES) - set(scores)
    if missing or missing_scores:
        raise ValueError(f"Missing rank inputs: {sorted(missing | missing_scores)}")
    values = {key: _numeric(scores[key], key) for key in MODEL_SCORES}
    values.update({key: _numeric(features[key], key) for key in HISTORY_SCORES})
    protocol = _numeric(features.ip_protocol, "ip_protocol")
    port = _numeric(features.resp_port, "resp_port")
    _numeric(features.flow_end_ms, "flow_end_ms")

    labels = truth.reset_index(drop=True).copy()
    labels["window"] = labels.window.astype(str)
    labels["family"] = labels.family.astype(str)
    if "native_group" not in labels:
        column = "native_type" if "native_type" in labels else "native_detail"
        labels["native_group"] = labels[column] if column in labels else ""
    for column, fallback in (("native_group", ""), ("event_id", ""),
                             ("evidence_level", "author_label"), ("exposure", source["role"])):
        if column not in labels:
            labels[column] = fallback
        labels[column] = labels[column].fillna("").astype(str)
    normal_mask = labels.family.eq("Benign").to_numpy()
    rare_mask = labels.family.isin(RARE_FAMILIES).to_numpy()
    thresholds, families, targets = [], [], []

    for window, window_positions in labels.groupby("window", sort=True).indices.items():
        for scope, scope_mask in masks.items():
            positions = window_positions[scope_mask[window_positions]]
            normal_positions = positions[normal_mask[positions]]
            rare_positions = positions[rare_mask[positions]]
            n = len(normal_positions)
            cap = n // 100 if n else None
            common = dict(source=source["name"], source_role=source["role"], group=group,
                          window=str(window), scope=scope, diagnostic_only=True)
            if len(positions):
                codes, label_keys = pd.factorize(
                    pd.MultiIndex.from_frame(labels.iloc[positions][list(LABEL_GROUPS)]), sort=False)
            else:
                codes, label_keys = np.empty(0, dtype=int), []
            row_counts = np.bincount(codes, minlength=len(label_keys))
            service_normals = pd.DataFrame({"protocol": protocol[normal_positions],
                                           "port": port[normal_positions]}).groupby(
                                               ["protocol", "port"], sort=False).indices
            service_targets = pd.DataFrame({"protocol": protocol[rare_positions],
                                           "port": port[rare_positions]}).groupby(
                                               ["protocol", "port"], sort=False).indices
            rare_base = _rare_rows(features, scores, labels, rare_positions)
            for name, score in values.items():
                sorted_normal = np.sort(score[normal_positions])
                cut = sorted_normal[n - cap - 1].item() if n else None
                hit = score[positions] > cut if n else None
                fp = int(n - np.searchsorted(sorted_normal, cut, side="right")) if n else None
                boundary = int(np.count_nonzero(sorted_normal == cut)) if n else None
                thresholds.append(dict(**common, ranking_score=name, rows=len(positions), normal=n,
                                       cap=cap, oracle_normal_threshold=cut, fp=fp,
                                       fpr=fp / n if n else None, within_budget=fp <= cap if n else None,
                                       boundary_normal_ties=boundary, unused_budget=cap - fp if n else None,
                                       status="supported" if n else "no_normals"))
                alert_counts = np.bincount(codes[hit], minlength=len(label_keys)) if n else None
                for i, keys in enumerate(label_keys):
                    family = keys[0]
                    alerts = int(alert_counts[i]) if n else None
                    families.append(dict(**common, ranking_score=name,
                                         **dict(zip(LABEL_GROUPS, keys, strict=True)),
                                         rows=int(row_counts[i]), oracle_alerts=alerts,
                                         tp=alerts if family not in NON_ATTACK else None,
                                         fp=alerts if family == "Benign" else None,
                                         normal=n, cap=cap, oracle_normal_threshold=cut,
                                         status="supported" if n else "no_normals"))
                if not len(rare_positions):
                    continue
                target = rare_base.copy()
                for key, value in common.items():
                    target[key] = value
                target["ranking_score"] = name
                target["target_score"] = score[rare_positions]
                target["normal"], target["cap"] = n, cap
                target["oracle_normal_threshold"] = cut
                target["oracle_detected"] = score[rare_positions] > cut if n else None
                gt, ge, eq = _tail_counts(sorted_normal, score[rare_positions])
                target["global_normal_gt"], target["global_normal_ge"], target["global_normal_eq"] = gt, ge, eq
                target["minimum_fp_to_include"] = ge if n else None
                target["can_include_within_budget"] = ge <= cap if n else None
                target["budget_excess"] = np.maximum(ge - cap, 0) if n else None
                # A service-only diagnostic has its own denominator, not a deployment promise.
                for column in TARGET_STATS[TARGET_STATS.index("service_normal"):]:
                    target[column] = pd.Series([None] * len(target), dtype=object)
                for service, indices in service_targets.items():
                    rel_normal = service_normals.get(service, np.empty(0, dtype=int))
                    normal_values = np.sort(score[normal_positions[rel_normal]])
                    sn = len(normal_values)
                    sc = sn // 100 if sn else None
                    sgt, sge, seq = _tail_counts(normal_values, score[rare_positions[indices]])
                    target.loc[indices, "service_normal"] = sn
                    target.loc[indices, "service_supported"] = bool(sn)
                    target.loc[indices, "service_cap"] = sc
                    target.loc[indices, "service_normal_gt"] = sgt
                    target.loc[indices, "service_normal_ge"] = sge
                    target.loc[indices, "service_normal_eq"] = seq
                    if sn:
                        target.loc[indices, "service_minimum_fp_to_include"] = sge
                        target.loc[indices, "service_can_include_within_budget"] = sge <= sc
                        target.loc[indices, "service_budget_excess"] = np.maximum(sge - sc, 0)
                # Keep unsupported values nullable, including when concatenating empty-normal scopes.
                for column in TARGET_STATS:
                    if column in TARGET_BOOLS:
                        target[column] = target[column].astype("boolean")
                    elif column == "oracle_normal_threshold":
                        target[column] = pd.to_numeric(target[column]).astype("Float64")
                    elif column != "target_score":
                        target[column] = pd.to_numeric(target[column]).astype("Int64")
                targets.append(target)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    empty_targets = _rare_rows(features, scores, labels, np.empty(0, dtype=int))
    for column in (*CONTEXT, *TARGET_STATS):
        if column not in empty_targets:
            empty_targets[column] = pd.Series(dtype=object)
    frames = {
        "ranking_thresholds": pd.DataFrame(thresholds),
        "ranking_families": pd.DataFrame(families),
        "ranking_targets": pd.concat(targets, ignore_index=True) if targets else empty_targets,
    }
    paths = []
    for name, frame in frames.items():
        path = output / (name + ".csv.gz")
        frame.to_csv(path, index=False, compression="gzip")
        paths.append(path)
    report = output / "ranking_summary.json"
    report.write_text(json.dumps(dict(
        diagnostic_only=True, source=source["name"], source_role=source["role"], group=group,
        scores=list(RANKING_SCORES), scopes=list(masks), source_rows=len(features),
        unique_rare_rows=int(rare_mask.sum()), rare_families=sorted(RARE_FAMILIES),
        threshold_formula="sorted_normal[n - floor(n/100) - 1]; alert iff score > threshold",
        target_inclusion_formula="normal score >= target score count <= floor(n/100)",
        label_use="Evaluation labels are used retrospectively; these thresholds are not deployable.",
        selection="No model/score selection or cross-scope/window/model pooling is performed.",
        service_interpretation="Same protocol and destination port; zero normals is unsupported. "
                               "Service-only budgets do not replace the global budget.",
        non_attack_families=sorted(NON_ATTACK),
        non_attack_counts="Benign alerts are FP; Unresolved/Mixed alerts are not TP. Their tp is null.",
        overlap="All/startup/after/original scopes can overlap; do not sum them.",
    ), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return paths + [report]
