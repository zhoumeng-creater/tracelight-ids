"""Retrospective score ceilings on each UTC day for the ordinary frozen HGB."""

from collections import defaultdict
from contextlib import ExitStack
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ids.common import read_json, write_json
from .evaluate import _aligned

VIEWS = ("raw", "contact", "quality")


def _top(previous, values, keep):
    joined = np.concatenate((previous, values))
    if len(joined) > keep:
        joined = np.partition(joined, len(joined) - keep)[-keep:]
    return joined


def _batches(pred, truth, blind, batch_rows):
    cols = (
        ["row_id", "raw", *["score_" + v for v in VIEWS]],
        ["row_id", "family", "native_detail"],
        ["row_id", "flow_end_ms"],
    )
    with ExitStack() as stack:
        files = [stack.enter_context(pq.ParquetFile(path)) for path in (pred, truth, blind)]
        for scored, labels, times in _aligned(files, cols, batch_rows):
            ids = scored["row_id"].to_numpy()
            if not all(np.array_equal(ids, b["row_id"].to_numpy()) for b in (labels, times)):
                raise ValueError("Unaligned rank diagnostic")
            days = times["flow_end_ms"].to_numpy() // 86400000
            raw = scored["raw"].to_numpy(zero_copy_only=False)
            yield days, raw, {v: scored["score_" + v].to_numpy() for v in VIEWS}, labels


def diagnose(pred, truth, blind, judgement, output, batch_rows):
    paired = read_json(judgement)
    bounds = {}
    epoch = date(1970, 1, 1)
    for day, summary in paired["utc_days"].items():
        d = (date.fromisoformat(day) - epoch).days
        base = summary["comparisons"]["raw"]
        n, fp, limit = base["normal_rows"], base["raw_fp"], base["allowed_fp"]
        if n:
            bounds[d] = {v: (limit if v == "raw" else limit - fp) for v in VIEWS}

    # k+1 largest normal scores suffice for the lowest observed strict cutoff with <= k false alerts.
    tops = defaultdict(lambda: np.empty(0))
    for days, raw, scores, labels in _batches(pred, truth, blind, batch_rows):
        benign = labels["family"].to_pandas().eq("Benign").to_numpy()
        for d in np.unique(days[benign]):
            normal = (days == d) & benign
            for v, k in bounds[d].items():
                if k >= 0:
                    mask = normal if v == "raw" else normal & ~raw
                    tops[d, v] = _top(tops[d, v], scores[v][mask], k + 1)
    cuts = {}
    for (d, v), values in tops.items():
        if len(values) != bounds[d][v] + 1:
            raise ValueError("Unexpected eligible normal denominator")
        cuts[d, v] = float(values.min())
    # Capture native categories separately; report gross additions and losses against raw.
    counts = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    for days, raw, scores, labels in _batches(pred, truth, blind, batch_rows):
        metadata = labels.select(["family", "native_detail"]).to_pandas()
        codes, keys = pd.factorize(pd.MultiIndex.from_frame(metadata), sort=False)
        for d in np.unique(days):
            selected = days == d
            before = raw[selected]
            for v in VIEWS:
                if (d, v) not in cuts:
                    continue
                after = scores[v][selected] > cuts[d, v]
                if v != "raw":
                    after |= before
                flags = before.astype("int8") + 2 * after
                hist = np.bincount(4 * codes[selected] + flags, minlength=4 * len(keys)).reshape(-1, 4)
                for key, row in zip(keys, hist, strict=True):
                    if row.any():
                        counts[d, v, *key] += row
    report = {}
    for d, limits in sorted(bounds.items()):
        day = (epoch + timedelta(days=int(d))).isoformat()
        report[day] = {}
        for v, k in limits.items():
            rows = [
                {
                    "family": family,
                    "native_detail": detail,
                    "rows": int(h.sum()),
                    "before": int(h[1] + h[3]),
                    "after": int(h[2] + h[3]),
                    "gained": int(h[2]),
                    "lost": int(h[1]),
                }
                for (date_key, view, family, detail), h in sorted(counts.items())
                if date_key == d and view == v
            ]
            report[day][v] = {
                "feasible": k >= 0,
                "allowed_fp_in_ranked_pool": k,
                "strict_cut": cuts.get((d, v)),
                "rows_by_label": rows,
            }
    write_json(
        output,
        {
            "diagnostic_only": True,
            "use": "Uses evaluation normals after all decisions are sealed",
            "days_without_normals": sorted(set(paired["utc_days"]) - set(report)),
            "utc_days": report,
        },
    )
    return [output]
