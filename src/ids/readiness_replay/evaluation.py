"""Compare sealed original/repaired flags with reusable, label-only group indices."""

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd

from ids.common import write_json


META = {"row_id", "flow_end_ms", "phase"}
RARE_FAMILIES = frozenset({"C&C", "ExternalC2", "PureCC", "C2", "Download", "FileDownload"})
COMPARISONS = ("same_policy", "raw_old", "add_quality_old")
SCOPES = {
    "all": (0, 1, 2, 3),
    "startup": (0, 2),
    "after": (1, 3),
    "original_startup": (0, 1),
    "original_after": (2, 3),
}
PAIR_FIELDS = ("policy", "comparison", "baseline", "rows", "before", "after", "gained", "lost")
COMMON = ("source", "source_role", "group", "window", "phase")
FAMILY = ("family", "native_group", "evidence_level", "exposure")


@dataclass
class Axis:
    keys: tuple
    labels: list
    codes: np.ndarray
    positions: np.ndarray | None = None


@dataclass
class Context:
    features: pd.DataFrame
    original_truth: pd.DataFrame
    metadata: pd.DataFrame
    clock: np.ndarray
    axes: dict
    event_starts: dict
    sample_positions: list
    exposure_missing: bool
    prepared_seconds: float


def _check_rows(frame, size, description):
    if frame.columns.duplicated().any() or "row_id" not in frame:
        raise ValueError(f"Ambiguous or missing {description} row identity")
    if len(frame) != size or not np.array_equal(frame.row_id.to_numpy(), np.arange(1, size + 1)):
        raise ValueError(f"{description} row identity/order differs")


def _strings(frame, column, fallback=""):
    if column not in frame:
        return np.full(len(frame), fallback, dtype=object)
    return frame[column].fillna("").astype(str).to_numpy()


def _axis(keys, encoded, levels, positions=None):
    arrays = [encoded[key] if positions is None else encoded[key][positions] for key in keys]
    if not len(arrays[0]):
        return Axis(tuple(keys), [], np.array([], dtype=np.int32), positions)
    cardinality = 1
    for key in keys:
        cardinality *= len(levels[key])
    if cardinality <= np.iinfo(np.int64).max:
        packed = np.zeros(len(arrays[0]), dtype=np.int64)
        for key, values in zip(keys, arrays):
            packed *= len(levels[key])
            packed += values
        codes, unique = pd.factorize(packed, sort=True)
        digits = []
        for key in reversed(keys):
            digits.append(unique % len(levels[key]))
            unique = unique // len(levels[key])
        digits.reverse()
        labels = [
            tuple(levels[key][digits[j][i]] for j, key in enumerate(keys)) for i in range(len(digits[0]))
        ]
    else:
        # Only unusually large label vocabularies need the tuple representation.
        codes, unique = pd.factorize(pd.MultiIndex.from_arrays(arrays), sort=True)
        labels = [tuple(levels[key][item[j]] for j, key in enumerate(keys)) for item in unique]
    return Axis(tuple(keys), labels, codes.astype(np.int32), positions)


def prepare_context(features, truth):
    """Factor source metadata once; reuse this object for the five frozen learners."""
    started = time.perf_counter()
    size = len(features)
    if not size:
        raise ValueError("Cannot evaluate an empty source")
    _check_rows(features, size, "feature")
    _check_rows(truth, size, "truth")
    clock = features.flow_end_ms.to_numpy(dtype=np.int64)
    if (np.diff(clock) < 0).any():
        raise ValueError("Evaluation requires the registered nondecreasing log clock")
    if "uid" in features and "uid" in truth and not np.array_equal(features.uid, truth.uid):
        raise ValueError("Feature and truth UID identities differ")
    metadata = pd.DataFrame(
        {
            "row_id": features.row_id.to_numpy(),
            "flow_end_ms": clock,
            "uid": _strings(truth if "uid" in truth else features, "uid"),
            "window": _strings(truth, "window"),
            "family": _strings(truth, "family"),
            "native_group": _strings(truth, "native_type" if "native_type" in truth else "native_detail"),
            "event_id": _strings(truth, "event_id"),
            "evidence_level": _strings(truth, "evidence_level", "author_label"),
            "exposure": _strings(truth, "exposure"),
            "protocol": features.ip_protocol.to_numpy(dtype=np.int64),
            "port": features.resp_port.to_numpy(dtype=np.int64),
        }
    )
    if (metadata.window == "").any() or (metadata.family == "").any():
        raise ValueError("Truth requires an observed UTC window and evaluation family")
    encoded, levels = {}, {}
    for column in ("window", *FAMILY, "event_id", "protocol", "port"):
        encoded[column], levels[column] = pd.factorize(metadata[column], sort=True)
    valid = metadata.event_id.ne("").to_numpy() & ~metadata.family.isin(["Benign", "Unresolved"]).to_numpy()
    positions = np.flatnonzero(valid)
    axes = {
        "family": _axis(("window", *FAMILY), encoded, levels),
        "service": _axis(("window", "protocol", "port", "family"), encoded, levels),
        "event": _axis(("window", "event_id", "family", "evidence_level"), encoded, levels, positions),
    }
    earliest = np.full(len(levels["event_id"]), np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(earliest, encoded["event_id"][valid], clock[valid])
    event_starts = {
        str(event): int(value)
        for event, value in zip(levels["event_id"], earliest)
        if value != np.iinfo(np.int64).max
    }
    sample_axis = _axis(("window", "family", "native_group"), encoded, levels)
    order = sample_axis.codes.argsort(kind="stable")
    borders = np.r_[0, np.flatnonzero(np.diff(sample_axis.codes[order])) + 1, len(order)]
    sample_positions = [
        (sample_axis.labels[int(sample_axis.codes[order[lo]])], order[lo:hi])
        for lo, hi in zip(borders[:-1], borders[1:])
    ]
    return Context(
        features,
        truth,
        metadata,
        clock,
        axes,
        event_starts,
        sample_positions,
        "exposure" not in truth,
        time.perf_counter() - started,
    )


def evaluation_scopes(old_decisions, repaired_decisions):
    """startup/after describe repaired readiness; original_* preserve v1's partition."""
    if len(old_decisions) != len(repaired_decisions):
        raise ValueError("Original and repaired decision lengths differ")
    old, new = old_decisions.phase.to_numpy(), repaired_decisions.phase.to_numpy()
    if not np.isin(old, ("startup", "after")).all() or not np.isin(new, ("startup", "after")).all():
        raise ValueError("Unexpected decision phase")
    return {
        "all": np.ones(len(old), dtype=bool),
        "startup": new == "startup",
        "after": new == "after",
        "original_startup": old == "startup",
        "original_after": old == "after",
    }


def _sum_columns(codes, flags, size):
    out = np.empty((size, flags.shape[1]), dtype=np.int64)
    for col in range(flags.shape[1]):
        out[:, col] = np.bincount(codes, weights=flags[:, col], minlength=size).astype(np.int64)
    return out


def _aggregate(axis, joint, old, new, baseline_indices, clock):
    positions = axis.positions
    if positions is not None:
        joint, old, new, clock = joint[positions], old[positions], new[positions], clock[positions]
    size = len(axis.labels) * 4
    codes = axis.codes.astype(np.int64) * 4 + joint
    shape = (len(axis.labels), 4, old.shape[1])
    before, after = _sum_columns(codes, old, size), _sum_columns(codes, new, size)
    overlap = []
    for baseline in baseline_indices:
        base = old[:, baseline]
        overlap.append(_sum_columns(codes, base & new, size).reshape(shape))
    result = dict(
        rows=np.bincount(codes, minlength=size).reshape(-1, 4),
        before=before.reshape(shape),
        after=after.reshape(shape),
        overlap=overlap,
    )
    if axis.keys[1] == "event_id":
        for version, flags in (("before", old), ("after", new)):
            first = np.full((size, old.shape[1]), np.iinfo(np.int64).max, dtype=np.int64)
            for col in range(old.shape[1]):
                hit = flags[:, col]
                np.minimum.at(first[:, col], codes[hit], clock[hit])
            result[version + "_first"] = first.reshape(shape)
    return result


def _scoped(stats, phases):
    return {
        key: (
            [item[:, phases].sum(axis=1) for item in value]
            if key == "overlap"
            else value[:, phases].min(axis=1)
            if key.endswith("_first")
            else value[:, phases].sum(axis=1)
        )
        for key, value in stats.items()
    }


def _pairs(stats, index, policies, baseline_indices):
    for p, policy in enumerate(policies):
        for b, comparison in enumerate(COMPARISONS):
            baseline = int(baseline_indices[b][p])
            before, after = int(stats["before"][index, baseline]), int(stats["after"][index, p])
            both = int(stats["overlap"][b][index, p])
            yield (
                p,
                baseline,
                dict(
                    policy=policy,
                    comparison=comparison,
                    baseline=policies[baseline],
                    rows=int(stats["rows"][index]),
                    before=before,
                    after=after,
                    gained=after - both,
                    lost=before - both,
                ),
            )


def _sample_data(context, scores, old_decisions, repaired_decisions, indices, source, group):
    indices = np.asarray(indices, dtype=np.int64)
    data = context.metadata.iloc[indices].drop(columns=["protocol", "port"]).reset_index(drop=True)
    if context.exposure_missing:
        data["exposure"] = source["role"]
    data.insert(0, "group", group)
    data.insert(0, "source", source["name"])
    data["original_phase"] = old_decisions.phase.to_numpy()[indices]
    data["repaired_phase"] = repaired_decisions.phase.to_numpy()[indices]
    additions = {}
    for name in context.features:
        if name not in ("row_id", "flow_end_ms") and pd.api.types.is_numeric_dtype(context.features[name]):
            additions["feature__" + name] = context.features[name].to_numpy()[indices]
    for name in scores:
        if name != "row_id" and pd.api.types.is_numeric_dtype(scores[name]):
            additions["score__" + name] = scores[name].to_numpy()[indices]
    for version, frame in (("original", old_decisions), ("repaired", repaired_decisions)):
        for name in frame:
            if name not in META:
                additions[version + "__" + name] = frame[name].to_numpy()[indices]
    return pd.concat([data, pd.DataFrame(additions)], axis=1)


def sample_rows(context, scores, old_decisions, repaired_decisions, indices, *, source, group):
    """Shared wide value rows; ranking can reuse these instead of duplicating rare payloads."""
    return _sample_data(context, scores, old_decisions, repaired_decisions, indices, source, group)


def evaluate_frames(
    features, scores, truth, old_decisions, repaired_decisions, *, source, group, output, context=None
):
    started = time.perf_counter()
    context = prepare_context(features, truth) if context is None else context
    if context.features is not features or context.original_truth is not truth:
        raise ValueError("Prepared context belongs to different feature/truth frames")
    size = len(features)
    for frame, label in (
        (scores, "scores"),
        (old_decisions, "original decisions"),
        (repaired_decisions, "repaired decisions"),
    ):
        _check_rows(frame, size, label)
    policies = [c for c in old_decisions if c not in META]
    if set(policies) != set(repaired_decisions.columns) - META or not {"raw_old", "add_quality_old"}.issubset(
        policies
    ):
        raise ValueError("Original and repaired policy sets/baselines differ")
    for frame in (old_decisions, repaired_decisions):
        if any(frame[c].dtype != bool for c in policies):
            raise ValueError("Only sealed Boolean policy flags may enter evaluation")
        if not np.array_equal(frame.flow_end_ms.to_numpy(), context.clock):
            raise ValueError("Decision and feature clocks differ")
    masks = evaluation_scopes(old_decisions, repaired_decisions)
    joint = (~masks["original_startup"]).astype(np.int8) * 2 + (~masks["startup"]).astype(np.int8)
    old = old_decisions[policies].to_numpy(dtype=bool)
    new = repaired_decisions[policies].to_numpy(dtype=bool)
    baseline_indices = [
        np.arange(len(policies)),
        np.full(len(policies), policies.index("raw_old")),
        np.full(len(policies), policies.index("add_quality_old")),
    ]
    totals = {
        key: _aggregate(axis, joint, old, new, baseline_indices, context.clock)
        for key, axis in context.axes.items()
    }
    records = {
        name: [] for name in ("budget", "paired", "services", "events", "normal_service_fp", "examples")
    }
    base = dict(source=source["name"], source_role=source["role"], group=group)
    service_labels = context.axes["service"].labels
    service_normal = {label[:3]: i for i, label in enumerate(service_labels) if label[3] == "Benign"}
    for phase, joint_columns in SCOPES.items():
        family = _scoped(totals["family"], joint_columns)
        budgets = {}
        for i, label in enumerate(context.axes["family"].labels):
            if not family["rows"][i]:
                continue
            window, *values = label
            common = dict(base, window=str(window), phase=phase)
            extra = dict(zip(FAMILY, map(str, values)))
            if context.exposure_missing:
                extra["exposure"] = source["role"]
            for _, _, comparison in _pairs(family, i, policies, baseline_indices):
                records["paired"].append(dict(common, **extra, **comparison))
            if window not in budgets:
                budgets[window] = dict(
                    normal=0,
                    unresolved_rows=0,
                    normal_flags=np.zeros((2, len(policies)), dtype=np.int64),
                    unresolved_flags=np.zeros((2, len(policies)), dtype=np.int64),
                )
            if values[0] in ("Benign", "Unresolved"):
                target = budgets[window]
                key = "normal" if values[0] == "Benign" else "unresolved_rows"
                flag_key = "normal_flags" if values[0] == "Benign" else "unresolved_flags"
                target[key] += int(family["rows"][i])
                target[flag_key] += np.stack([family["before"][i], family["after"][i]])
        for window, item in budgets.items():
            normal = item["normal"]
            for version, index in (("original", 0), ("repaired", 1)):
                for p, policy in enumerate(policies):
                    fp = int(item["normal_flags"][index, p])
                    records["budget"].append(
                        dict(
                            base,
                            window=str(window),
                            phase=phase,
                            version=version,
                            policy=policy,
                            normal=normal,
                            fp=fp,
                            cap=normal // 100,
                            fpr=fp / normal if normal else None,
                            within_budget=bool(fp <= normal // 100) if normal else None,
                            unresolved_rows=item["unresolved_rows"],
                            unresolved_alerts=int(item["unresolved_flags"][index, p]),
                        )
                    )
        service = _scoped(totals["service"], joint_columns)
        for i, (window, proto, port, family_name) in enumerate(service_labels):
            if not service["rows"][i]:
                continue
            common = dict(base, window=str(window), phase=phase, protocol=int(proto), port=int(port))
            if family_name == "Benign":
                normal = int(service["rows"][i])
                for version, key in (("original", "before"), ("repaired", "after")):
                    for p, policy in enumerate(policies):
                        fp = int(service[key][i, p])
                        if fp:
                            records["normal_service_fp"].append(
                                dict(
                                    common,
                                    version=version,
                                    policy=policy,
                                    normal=normal,
                                    fp=fp,
                                    fpr=fp / normal,
                                )
                            )
            elif family_name != "Unresolved":
                partner = service_normal.get((window, proto, port))
                normal = int(service["rows"][partner]) if partner is not None else 0
                for p, baseline, comparison in _pairs(service, i, policies, baseline_indices):
                    fp = int(service["after"][partner, p]) if partner is not None else 0
                    before_fp = int(service["before"][partner, baseline]) if partner is not None else 0
                    records["services"].append(
                        dict(
                            common,
                            family=str(family_name),
                            normal=normal,
                            normal_fp=fp,
                            normal_fpr=fp / normal if normal else None,
                            before_normal_fp=before_fp,
                            before_normal_fpr=before_fp / normal if normal else None,
                            **comparison,
                        )
                    )
        event = _scoped(totals["event"], joint_columns)
        for i, (window, event_id, family_name, level) in enumerate(context.axes["event"].labels):
            if not event["rows"][i]:
                continue
            start = context.event_starts[str(event_id)]
            common = dict(
                base,
                window=str(window),
                phase=phase,
                event_id=str(event_id),
                family=str(family_name),
                evidence_level=str(level),
                first_record_ms=start,
            )
            for p, baseline, comparison in _pairs(event, i, policies, baseline_indices):
                new_first, old_first = (
                    int(event["after_first"][i, p]),
                    int(event["before_first"][i, baseline]),
                )
                new_first = new_first if new_first != np.iinfo(np.int64).max else None
                old_first = old_first if old_first != np.iinfo(np.int64).max else None
                records["events"].append(
                    dict(
                        common,
                        first_alert_ms=new_first,
                        observed_delay_ms=new_first - start if new_first is not None else None,
                        before_first_alert_ms=old_first,
                        before_observed_delay_ms=old_first - start if old_first is not None else None,
                        **comparison,
                    )
                )
    selected_values = set()
    for (window, family_name, native), positions in context.sample_positions:
        if family_name in RARE_FAMILIES:
            continue
        for p, policy in enumerate(policies):
            current = new[positions, p]
            for b, comparison in enumerate(COMPARISONS):
                baseline = int(baseline_indices[b][p])
                previous = old[positions, baseline]
                for change, chosen in (
                    ("gained", current & ~previous),
                    ("lost", ~current & previous),
                    ("both_negative", ~current & ~previous),
                ):
                    for index in positions[np.flatnonzero(chosen)[:3]]:
                        selected_values.add(int(index))
                        records["examples"].append(
                            dict(
                                base,
                                window=str(window),
                                family=str(family_name),
                                native_group=str(native),
                                row_id=int(index) + 1,
                                policy=policy,
                                comparison=comparison,
                                baseline=policies[baseline],
                                change=change,
                                before=bool(old[index, baseline]),
                                after=bool(new[index, p]),
                            )
                        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    schemas = {
        "budget": (
            *COMMON,
            "version",
            "policy",
            "normal",
            "fp",
            "cap",
            "fpr",
            "within_budget",
            "unresolved_rows",
            "unresolved_alerts",
        ),
        "paired": (*COMMON, *FAMILY, *PAIR_FIELDS),
        "services": (
            *COMMON,
            "protocol",
            "port",
            "family",
            "normal",
            "normal_fp",
            "normal_fpr",
            "before_normal_fp",
            "before_normal_fpr",
            *PAIR_FIELDS,
        ),
        "events": (
            *COMMON,
            "event_id",
            "family",
            "evidence_level",
            "first_record_ms",
            "first_alert_ms",
            "observed_delay_ms",
            "before_first_alert_ms",
            "before_observed_delay_ms",
            *PAIR_FIELDS,
        ),
        "normal_service_fp": (*COMMON, "protocol", "port", "version", "policy", "normal", "fp", "fpr"),
        "examples": (
            "source",
            "source_role",
            "group",
            "window",
            "family",
            "native_group",
            "row_id",
            "policy",
            "comparison",
            "baseline",
            "change",
            "before",
            "after",
        ),
    }
    outputs = []
    for name, data in records.items():
        path = output / (name + ".csv.gz")
        pd.DataFrame(data, columns=schemas[name]).to_csv(path, index=False, compression="gzip")
        outputs.append(path)
    rare = np.flatnonzero(context.metadata.family.isin(RARE_FAMILIES).to_numpy())
    for name, positions in (
        ("rare_rows", rare),
        ("example_values", np.array(sorted(selected_values), dtype=np.int64)),
    ):
        path = output / (name + ".parquet")
        sample_rows(
            context, scores, old_decisions, repaired_decisions, positions, source=source, group=group
        ).to_parquet(path, index=False, compression="zstd")
        outputs.append(path)
    report = output / "summary.json"
    write_json(
        report,
        dict(
            base,
            rows=size,
            policies=policies,
            scope_rows={k: int(v.sum()) for k, v in masks.items()},
            family_rows=context.metadata.family.value_counts().to_dict(),
            events_with_known_attack_labels=len(context.event_starts),
            rare_rows=len(rare),
            bounded_example_value_rows=len(selected_values),
            comparisons=list(COMPARISONS),
            prepare_context_seconds=context.prepared_seconds,
            evaluation_seconds=time.perf_counter() - started,
            comparison_versions={"before": "original", "after": "repaired"},
            interpretation="Startup/after use repaired readiness. Original_* preserve v1 phases. Event starts come from the full source; stage IDs do not imply independent campaigns.",
        ),
    )
    return outputs + [report]
