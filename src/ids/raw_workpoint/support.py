"""Post-seal service support for raw misses and port-only versus large-response additions."""

import gzip
import io
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ids.response_evidence.evaluation import endpoint
from .evaluate import ARMS, _aligned

TARGETS = {"PureCC", "Download", "Scanning", "Heartbeat"}
GROUP = ["utc_day", "ip_protocol", "resp_port", "endpoint", "native_detail"]
METRICS = [
    "rows",
    "raw_negative",
    "web64k_trigger",
    *[arm + "_gain" for arm in ARMS if arm != "raw"],
    "port80_gain",
]


def _list_counts(groups, gains, raw, web, port):
    groups["rows"] = 1
    groups["raw_negative"] = (~raw).astype(np.int64)
    groups["web64k_trigger"] = web.astype(np.int64)
    for arm, mask in gains.items():
        groups[arm + "_gain"] = (mask & ~raw).astype(np.int64)
    groups["port80_gain"] = (port & ~raw).astype(np.int64)
    return groups.groupby(GROUP, sort=False, observed=True)[METRICS].sum()


def _format(groups, total=False):
    # Normal rows on a common service are repeated per target detail. Comparisons across
    # services use the separate control rows, never a sum of these denominators.
    keys = ["utc_day", "ip_protocol", "resp_port"]
    normal = groups.loc[groups.endpoint.eq("Benign"), keys + METRICS]
    target = groups.loc[groups.endpoint.isin(TARGETS)]
    service = target.merge(normal, how="left", on=keys, suffixes=("", "_normal"), validate="many_to_one")
    service["record_type"] = "service"
    control = groups.groupby(["utc_day", "endpoint", "native_detail"], as_index=False, sort=False)[
        METRICS
    ].sum()
    control["record_type"] = "port80_control"
    # Counts in normal columns of control rows are whole-day counts, even if no targets occurred there.
    daily_normal = normal.groupby("utc_day", as_index=False)[METRICS].sum()
    control = control.merge(
        daily_normal, on="utc_day", how="left", suffixes=("", "_normal"), validate="many_to_one"
    )
    out = pd.concat([control, service], ignore_index=True)
    for col in METRICS:
        out[col + "_normal"] = out[col + "_normal"].fillna(0).astype(np.int64)
    out["scope"] = "capture" if total else "utc_day"
    if total:
        out["utc_day"] = "ALL"
    else:
        out["utc_day"] = pd.to_datetime(out.utc_day, unit="D", utc=True).dt.strftime("%Y-%m-%d")
    out["raw_alerts"] = out.rows - out.raw_negative
    out["raw_fp_normal"] = out.rows_normal - out.raw_negative_normal
    out["raw_or_port80"] = out.raw_alerts + out.port80_gain
    out["raw_or_port80_fp_normal"] = out.raw_fp_normal + out.port80_gain_normal
    out["raw_or_web64k"] = out.raw_alerts + out.add_web64k_gain
    out["raw_or_web64k_fp_normal"] = out.raw_fp_normal + out.add_web64k_gain_normal
    for col in ("ip_protocol", "resp_port"):
        out[col] = out[col].astype("Int64")
    return out.sort_values(
        ["record_type", "utc_day", "endpoint", "native_detail", "ip_protocol", "resp_port"], kind="stable"
    )


def summarize(pred: Path, truth: Path, blind: Path, output: Path, *, batch_rows: int = 131072):
    """Write an exhaustive gzipped CSV after scoring. Native downloads and normal supports remain separate."""
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    predicted = ["row_id", *ARMS]
    labels = ["row_id", "family", "native_detail"]
    inputs = ["row_id", "flow_end_ms", "ip_protocol", "resp_port", "resp_ip_bytes"]
    seen = 0
    last = None
    # Merge partial grouped tables in a binary tree; raw records never accumulate in memory.
    partial = []
    with pq.ParquetFile(pred) as p, pq.ParquetFile(truth) as t, pq.ParquetFile(blind) as b:
        sizes = [f.metadata.num_rows for f in (p, t, b)]
        if sizes[0] <= 0 or len(set(sizes)) != 1:
            raise ValueError("Three nonempty sealed streams must have identical lengths")
        if any(not pa.types.is_boolean(p.schema_arrow.field(a).type) for a in ARMS):
            raise ValueError("Expected sealed boolean decisions")
        for x, y, z in _aligned((p, t, b), (predicted, labels, inputs), batch_rows):
            ids = np.arange(seen + 1, seen + len(x) + 1)
            for batch in (x, y, z):
                if any(c.null_count for c in batch.columns) or not np.array_equal(
                    ids, batch["row_id"].to_numpy()
                ):
                    raise ValueError("Missing data or inconsistent physical IDs in service support")
            seen += len(x)
            clock = z["flow_end_ms"].to_numpy()
            if (last is not None and clock[0] < last) or np.any(clock[1:] < clock[:-1]):
                raise ValueError("Observation clock regresses")
            last = clock[-1]
            fam, det = y["family"].to_pandas(), y["native_detail"].to_pandas()
            codes, pairs = pd.factorize(pd.MultiIndex.from_arrays([fam, det]), sort=False)
            category = np.asarray([endpoint(a, d) for a, d in pairs], dtype=object)[codes]
            normal = category == "Benign"
            interested = normal | np.isin(category, list(TARGETS))
            proto, port, size = (z[c].to_numpy() for c in inputs[2:])
            port80 = (proto == 6) & (port == 80)
            web = port80 & (size >= 65536)
            raw = x["raw"].to_numpy(zero_copy_only=False)
            decisions = {a: x[a].to_numpy(zero_copy_only=False) for a in ARMS if a != "raw"}
            if not np.array_equal(decisions["add_web64k"], raw | web):
                raise ValueError("Sealed web64k alerts disagree with observed fields")
            if any(np.any(raw & ~v) for a, v in decisions.items() if a.startswith("add_")):
                raise ValueError("Additive policy removed a raw alert")
            # A date with no trusted normals or targets must still have a global control row.
            # Keep excluded classes in one diagnostic bucket, never in the normal denominator.
            category[~interested] = "Other"
            detail = np.where(normal | ~interested, "", det.to_numpy())
            groups = pd.DataFrame(
                {
                    "utc_day": clock // 86400000,
                    "ip_protocol": np.where(interested, proto, -1),
                    "resp_port": np.where(interested, port, -1),
                    "endpoint": category,
                    "native_detail": detail,
                }
            )
            block = _list_counts(groups, decisions, raw, web, port80)
            level = 0
            while level < len(partial) and partial[level] is not None:
                block = block.add(partial[level], fill_value=0).astype(np.int64)
                partial[level] = None
                level += 1
            if level == len(partial):
                partial.append(block)
            else:
                partial[level] = block
    if seen != sizes[0]:
        raise ValueError("Incomplete observation capture")
    table = (
        pd.concat([v for v in partial if v is not None]).groupby(level=GROUP, sort=False).sum().reset_index()
    )
    total = table.groupby(GROUP[1:], as_index=False, sort=False)[METRICS].sum()
    total["utc_day"] = 0
    frames = [_format(table), _format(total, total=True)]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as temp:
        pending = Path(temp.name)
    try:
        with (
            pending.open("wb") as stream,
            gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as gz,
            io.TextIOWrapper(gz, encoding="utf8", newline="") as text,
        ):
            for i, frame in enumerate(frames):
                fields = [
                    "record_type",
                    "scope",
                    "utc_day",
                    "endpoint",
                    "native_detail",
                    "ip_protocol",
                    "resp_port",
                ]
                frame.loc[:, fields + [c for c in frame if c not in fields]].to_csv(
                    text, index=False, header=i == 0
                )
        pending.replace(output)
    finally:
        pending.unlink(missing_ok=True)
    return [output]
