"""Replay previously registered thresholds and two causal, periodically updated reference pools.

The caller supplies the observation stream, including warmup rows. A score becomes observable at
flow_end_ms. The model must already be frozen and the fixed threshold available before start_ms.
True labels never enter the unlabeled rule. A timestamp for a trusted normal confirmation is a
separate input; archive class labels by themselves do not establish when this feedback was known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ids.route_batch.metrics import scores as checked_scores
from ids.route_batch.metrics import threshold


POLICIES = ("fixed", "recent_normal", "unlabeled")
ALPHA = 0.01
MIN_CALIBRATION = 99  # First n for which ceil(.99*(n+1)) <= n.


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _timestamps(series, name, nullable=False):
    # Datetime milliseconds are exactly representable on this interval in both float64 and int64.
    if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
        raise ValueError(f"{name} must contain integer UTC milliseconds")
    values = series.to_numpy(dtype=np.float64, na_value=np.nan)
    absent = np.isnan(values)
    present = values[~absent]
    if (absent.any() and not nullable) or not np.isfinite(present).all():
        raise ValueError(f"Missing or nonfinite {name}")
    if (present < 0).any() or (present >= 2**53).any() or (present != np.floor(present)).any():
        raise ValueError(f"Invalid integer milliseconds in {name}")
    return pd.Series(values, copy=False).fillna(-1).to_numpy(dtype=np.int64)


def _aligned(stream, values):
    if not isinstance(values, pd.Series) or not values.index.equals(stream.index):
        raise ValueError("Auxiliary series must align with the supplied stream index")


def replay_thresholds(
    stream,
    fixed_threshold,
    *,
    start_ms,
    end_ms,
    trusted_normal_available_ms=None,
    refresh_ms=900_000,
    window_ms=3_600_000,
    min_reference=1000,
    trusted_lag_ms=0,
    trusted_window_ms=None,
):
    """Return chronological decisions and one threshold receipt per policy and nonempty UTC block.

    Input columns: row_id (unique), flow_end_ms, score. Output interval is [start_ms, end_ms).
    The unlabeled pool at update u includes every supplied score in [u-window_ms, u).
    Trusted normal scores use the same event-time window shifted back by trusted_lag_ms and a
    confirmation strictly before u; supply one nullable timestamp per row, aligned by pandas index.
    Each adaptive rule uses the registered rank and strict >, or the fixed threshold on shortfall.
    With default parameters a pool needs at least 1,000 rows (about ten upper-tail rows).
    """
    start_ms = _integer(start_ms, "start_ms")
    end_ms = _integer(end_ms, "end_ms", start_ms + 1)
    refresh_ms = _integer(refresh_ms, "refresh_ms", 1)
    window_ms = _integer(window_ms, "window_ms", 1)
    trusted_lag_ms = _integer(trusted_lag_ms, "trusted_lag_ms")
    trusted_window_ms = (
        window_ms if trusted_window_ms is None else _integer(trusted_window_ms, "trusted_window_ms", 1)
    )
    min_reference = _integer(min_reference, "min_reference", MIN_CALIBRATION)
    if not np.isfinite(fixed_threshold):
        raise ValueError("The fixed threshold must be finite")
    if stream.row_id.isna().any() or not stream.row_id.is_unique:
        raise ValueError("Missing or duplicate observed row ID")
    times = _timestamps(stream.flow_end_ms, "flow_end_ms")
    order = np.argsort(times, kind="stable")
    times = times[order]
    scores = checked_scores(stream.score)[order]
    released = None
    if trusted_normal_available_ms is not None:
        _aligned(stream, trusted_normal_available_ms)
        released = _timestamps(trusted_normal_available_ms, "trusted_normal_available_ms", nullable=True)[
            order
        ]
        if np.any((released >= 0) & (released < times)):
            raise ValueError("Trusted normal confirmation predates observation")

    first, last = times.searchsorted([start_ms, end_ms])
    observed = stream.iloc[order[first:last]]
    decisions = observed[["row_id"]].reset_index(drop=True)
    decisions["flow_end_ms"] = times[first:last]
    starts = (times[first:last] // refresh_ms) * refresh_ms
    decisions["update_ms"] = starts
    blocks, offsets = np.unique(starts, return_index=True)
    cuts = {policy: np.full(len(decisions), float(fixed_threshold)) for policy in POLICIES}
    receipts = []
    for block, lo, hi in zip(blocks.tolist(), offsets.tolist(), [*offsets[1:], len(starts)]):
        receipts.append(
            {
                "update_ms": block,
                "policy": "fixed",
                "threshold": float(fixed_threshold),
                "status": "fixed",
                "decision_rows": int(hi - lo),
            }
        )
        for policy in ("recent_normal", "unlabeled"):
            cutoff = block - trusted_lag_ms if policy == "recent_normal" else block
            beginning = cutoff - (trusted_window_ms if policy == "recent_normal" else window_ms)
            left, right = times.searchsorted([beginning, cutoff])
            values = scores[left:right]
            event_times = times[left:right]
            max_available_ms = None
            status = "updated"
            if policy == "recent_normal":
                if released is None:
                    values = values[:0]
                    event_times = event_times[:0]
                    status = "feedback_unavailable"
                else:
                    release_window = released[left:right]
                    keep = (release_window >= 0) & (release_window < block)
                    values = values[keep]
                    event_times = event_times[keep]
                    if keep.any():
                        max_available_ms = int(release_window[keep].max())
            cut = float(fixed_threshold)
            rank = None
            if len(values) >= min_reference:
                registration = threshold(values, ALPHA)
                cut = registration["threshold"]
                rank = registration["rank"]
            elif status == "updated":
                status = "insufficient_reference"
            cuts[policy][lo:hi] = cut
            receipts.append(
                {
                    "update_ms": block,
                    "policy": policy,
                    "status": status,
                    "threshold": cut,
                    "window_start_ms": beginning,
                    "window_end_exclusive_ms": cutoff,
                    "n_reference": len(values),
                    "rank": rank,
                    "reference_strictly_above": int((values > cut).sum()),
                    "max_flow_end_ms": int(event_times[-1]) if len(event_times) else None,
                    "max_confirmation_ms": max_available_ms,
                    "decision_rows": int(hi - lo),
                }
            )
    for policy, values in cuts.items():
        decisions[policy + "_threshold"] = values
        decisions[policy + "_alert"] = scores[first:last] > values
    receipt_columns = [
        "update_ms",
        "policy",
        "status",
        "threshold",
        "window_start_ms",
        "window_end_exclusive_ms",
        "n_reference",
        "rank",
        "reference_strictly_above",
        "max_flow_end_ms",
        "max_confirmation_ms",
        "decision_rows",
    ]
    return decisions, pd.DataFrame.from_records(receipts, columns=receipt_columns)


def audit_reference_mix(stream, updates, labels):
    """Audit the unlabeled reference pools after replay. NA labels count as unadjudicated."""
    _aligned(stream, labels)
    if not labels.dropna().isin([0, 1]).all():
        raise ValueError("Reference labels must be 0, 1 or missing")
    times = _timestamps(stream.flow_end_ms, "flow_end_ms")
    order = np.argsort(times, kind="stable")
    times = times[order]
    truth = labels.to_numpy(dtype=np.float64, na_value=np.nan)[order]
    records = []
    for update in updates.loc[updates.policy.eq("unlabeled")].itertuples(index=False):
        lo, hi = times.searchsorted([update.window_start_ms, update.window_end_exclusive_ms])
        pool = truth[lo:hi]
        if len(pool) != update.n_reference:
            raise ValueError("Reference receipt and audit population do not align")
        positive = int((pool == 1).sum())
        negative = int((pool == 0).sum())
        records.append(
            {
                "update_ms": int(update.update_ms),
                "reference_rows": len(pool),
                "known_attack": positive,
                "known_normal": negative,
                "unadjudicated": len(pool) - positive - negative,
                "attack_fraction_of_labeled": positive / (positive + negative)
                if positive + negative
                else None,
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "update_ms",
            "reference_rows",
            "known_attack",
            "known_normal",
            "unadjudicated",
            "attack_fraction_of_labeled",
        ],
    )


def paired_operating_point(evaluated_rows, baseline_col, candidate_col):
    """Compare alert flags on identical rows. No-positive cohorts have zero counts and a null recall."""
    frame = evaluated_rows
    if frame.row_id.isna().any() or not frame.row_id.is_unique:
        raise ValueError("Evaluation row identities must be unique")
    if (
        frame.attack_family.isna().any()
        or not frame.label.isin([0, 1]).all()
        or frame.attack_family.eq("Benign").ne(frame.label.eq(0)).any()
    ):
        raise ValueError("Evaluation requires adjudicated binary and family labels")
    for col in (baseline_col, candidate_col):
        if not pd.api.types.is_bool_dtype(frame[col]) or frame[col].isna().any():
            raise ValueError("Alert columns must be Boolean without missing values")
    normal = frame.label.eq(0)
    attack = ~normal
    family_names = sorted(set(frame.loc[attack, "attack_family"]) | {"Scanning", "Bot"})
    groups = {"normal": normal, "all_attacks": attack}
    groups.update({family: frame.attack_family.eq(family) for family in family_names})
    counts = {}
    for name, mask in groups.items():
        baseline, candidate = frame.loc[mask, baseline_col], frame.loc[mask, candidate_col]
        old_count, new_count = int(baseline.sum()), int(candidate.sum())
        counts[name] = {
            "rows": len(baseline),
            "baseline": old_count,
            "candidate": new_count,
            "added": int((~baseline & candidate).sum()),
            "lost": int((baseline & ~candidate).sum()),
            "net": new_count - old_count,
            "baseline_rate": old_count / len(baseline) if len(baseline) else None,
            "candidate_rate": new_count / len(baseline) if len(baseline) else None,
        }
    budget = int(normal.sum()) // 100
    return {
        "groups": counts,
        "normal_false_alert_budget": budget,
        "baseline_within_one_percent": counts["normal"]["baseline"] <= budget if normal.any() else None,
        "candidate_within_one_percent": counts["normal"]["candidate"] <= budget if normal.any() else None,
    }
