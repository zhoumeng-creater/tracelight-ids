"""Apply predeclared local-prefix policies to frozen continuous scores, without labels."""
from pathlib import Path

import numpy as np
import pandas as pd

from ids.common import sha256_file, write_json
from ids.raw_workpoint import controls
from . import policy

COUNT_COLUMNS = ("hist_dst_ports_60s", "hist_dst_ips_60s")
COMPONENTS = {
    "ports": (COUNT_COLUMNS[0],),
    "targets": (COUNT_COLUMNS[1],),
    "counts": COUNT_COLUMNS,
    "raw": ("score_raw",),
    "quality": ("score_quality",),
    "quality_counts": ("score_quality", *COUNT_COLUMNS),
    "raw_quality_counts": ("score_raw", "score_quality", *COUNT_COLUMNS),
}
HOUR_MS = 3_600_000


def calibration_frame(root, group):
    base = Path(root) / "local/controls/evidence/calibration"
    cal = pd.read_parquet(base / (group + ".parquet")).set_index(["capture", "row_id"])
    features = controls.CONFIG / "early-normal-features.parquet"
    ref = controls.CONFIG / "controls.json"
    from ids.common import read_json
    if sha256_file(features) != read_json(ref)["calibration_features_sha256"]:
        raise ValueError("Early-normal history features changed")
    counts = pd.read_parquet(features).set_index(["capture", "row_id"])[list(COUNT_COLUMNS)]
    if not cal.index.equals(counts.index):
        raise ValueError("Early-normal score and history identities differ")
    return pd.concat([cal.rename(columns={v: "score_" + v for v in ("raw", "contact", "quality")}),
                      counts], axis=1)


def decide(scored_file, features_file, proxy_file, calibration, output, report):
    scores = pd.read_parquet(scored_file)
    frame = pd.read_parquet(features_file, columns=["row_id", "flow_end_ms", *COUNT_COLUMNS])
    proxy = pd.read_parquet(proxy_file, columns=["row_id", "scan_proxy"])
    ids = np.arange(1, len(frame) + 1)
    for other in (scores, frame, proxy):
        if not len(other) or not np.array_equal(other.row_id, ids):
            raise ValueError("Blind score, history or proxy row order changed")
    clock = frame.flow_end_ms.to_numpy()
    if (np.diff(clock) < 0).any():
        raise ValueError("Prefix decisions require monotone log arrival order")
    cutoff = int(clock[0]) + HOUR_MS
    before = clock < cutoff
    blind = frame.assign(score_raw=scores.score_raw, score_quality=scores.score_quality)
    out = pd.DataFrame({"row_id": ids, "flow_end_ms": clock,
                        "phase": np.where(before, "startup", "after")})
    for view in ("raw", "contact", "quality", "add_quality"):
        out[view + "_old"] = scores[view].to_numpy(dtype=bool)
    registrations = {}
    for name, components in COMPONENTS.items():
        old_rules = policy.register(calibration, components)
        local = policy.prefix_rules(blind, old_rules, components, cutoff_ms=cutoff)
        fallback = policy.apply(blind, old_rules)
        after = policy.apply(blind, local["rules"])
        out[name + "_local"] = np.where(before, fallback, after)
        registrations[name] = {"old": old_rules, "local": local}
        if name in ("ports", "targets", "counts", "quality_counts"):
            out[name + "_old"] = fallback
    out["frozen_plus_ports_local"] = out.add_quality_old | out.ports_local
    out["frozen_plus_counts_local"] = out.add_quality_old | out.counts_local
    out["no_reply_scan_proxy"] = proxy.scan_proxy.to_numpy(dtype=bool)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False, compression="zstd")
    write_json(report, {"rows": len(frame), "prefix_start_ms": int(clock[0]), "cutoff_ms": cutoff,
                        "startup_rows": int(before.sum()), "after_rows": int((~before).sum()),
                        "history_reset": "Each registered day file/block is a cold-start deployment; one prefix per block",
                        "reference": "All first-hour rows; labels, event identities and native classes never read",
                        "registration": registrations,
                        "decisions_sha256": sha256_file(output),
                        "components": {k: list(v) for k, v in COMPONENTS.items()},
                        "allocation": "1% divided equally among components; strict rank with indivisible ties",
                        "scope": "Only the observed prefix alert fraction is constrained; future FPR is measured"})
    return [Path(output), Path(report)]
