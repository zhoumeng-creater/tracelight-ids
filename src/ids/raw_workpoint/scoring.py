"""Score held-out blind records with frozen R23 weights and past normal-only thresholds."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ids.common import read_json
from ids.contact_evidence.experiment import GROUPS, TREE_GROUPS, backend
from ids.response_evidence.experiment import ARMS
from ids.route_batch.metrics import decisions, scores, threshold

VIEWS = ("raw", "contact", "quality")
POLICIES = (*VIEWS, "add_contact", "add_quality", "add_quic30", "add_web64k")
SCORE_COLUMNS = tuple("score_" + view for view in VIEWS)


def residual_rule(helper, raw_alert, cap):
    """Lowest strict threshold admissible on the earlier normals missed by raw."""
    values = scores(helper)
    raw_alert = np.asarray(raw_alert)
    if raw_alert.dtype != bool or raw_alert.shape != values.shape or not 0 <= cap < len(values):
        raise ValueError("Invalid calibration mask or false-positive budget")
    hits = int(raw_alert.sum())
    remaining = cap - hits
    if remaining < 0:
        raise ValueError("The registered raw detector already exceeds the calibration budget")
    pool = values[~raw_alert]
    # With strict '>' the (remaining+1)-th highest excluded score keeps all ties together.
    index = len(pool) - remaining - 1
    cut = float(np.partition(pool, index)[index])
    added = int((pool > cut).sum())
    return {
        "threshold": cut,
        "comparison": ">",
        "normal_rows": len(values),
        "fp_cap": cap,
        "raw_fp": hits,
        "raw_negative_rows": len(pool),
        "remaining_fp": remaining,
        "added_calibration_fp": added,
        "total_calibration_fp": hits + added,
    }


def register(calibration_dir, group):
    """Read only the group's earlier C scores, never the evaluation observations."""
    calibration_dir = Path(calibration_dir)
    cal = pd.read_parquet(calibration_dir / (group + ".parquet"), columns=["capture", "row_id", *VIEWS])
    if (
        cal.empty
        or cal[["capture", "row_id"]].isna().any().any()
        or cal.duplicated(["capture", "row_id"]).any()
    ):
        raise ValueError("Missing or duplicate normal calibration identity")
    original = read_json(calibration_dir / (group + ".json"))
    rules = {k: threshold(cal[k], alpha=0.01) for k in VIEWS}
    for k in VIEWS:
        if rules[k] != original[k]:
            raise ValueError(f"Normal calibration scores differ from the frozen {group}/{k} cut")
    raw = decisions(cal["raw"], rules["raw"])
    cap = len(cal) // 100
    supplements = {view: residual_rule(cal[view], raw, cap) for view in ("contact", "quality")}
    return rules, supplements


def make_decisions(values, rules, supplements, frame):
    out = {k: decisions(values[k], rules[k]) for k in VIEWS}
    raw = out["raw"]
    for k in ("contact", "quality"):
        out["add_" + k] = raw | decisions(values[k], supplements[k])
    service = frame["orig_port"].eq(443) | frame["resp_port"].eq(443)
    quic = frame["ip_protocol"].eq(17) & service & frame["duration_seconds"].ge(30)
    out["add_quic30"] = raw | quic.to_numpy(dtype=bool)
    web = frame["ip_protocol"].eq(6) & frame["resp_port"].eq(80) & frame["resp_ip_bytes"].ge(65_536)
    out["add_web64k"] = raw | web.to_numpy(dtype=bool)
    return out


def score_group(
    features: Path,
    truth_free_raw: Path,
    frozen_dir: Path,
    calibration_dir: Path,
    group: str,
    out: Path,
    *,
    batch_rows: int,
    threads: int,
    device: str,
    neural_batch_rows: int | None = None,
):
    """Write aligned decisions and continuous scores. Return a JSON-serializable audit record."""
    if group not in GROUPS or batch_rows < 1 or threads < 1:
        raise ValueError("Unregistered learner or invalid scoring resource configuration")
    cpu_tree = group in TREE_GROUPS
    if (cpu_tree and device != "cpu") or (not cpu_tree and str(device).split(":")[0] != "cuda"):
        raise ValueError("HGB requires CPU; the registered MLP scoring device is CUDA")
    rules, supplements = register(calibration_dir, group)
    impl = backend(group)
    options = {"threads": threads} if cpu_tree else {"threads": threads, "device": device}
    suffix = ".joblib" if cpu_tree else ".pt"
    models = {k: impl.load_arm(Path(frozen_dir) / (group + "-" + k + suffix), **options) for k in VIEWS}
    if neural_batch_rows is not None:
        if cpu_tree or not isinstance(neural_batch_rows, int) or neural_batch_rows < 1:
            raise ValueError("An explicit neural batch size requires a positive integer and a neural model")
        for model in models.values():
            model.score_batch = neural_batch_rows
    columns = ["row_id", *ARMS["quality"]]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            pa.field("row_id", pa.int64()),
            *(pa.field(k, pa.bool_()) for k in POLICIES),
            *(pa.field(k, pa.float64()) for k in SCORE_COLUMNS),
        ]
    )
    with NamedTemporaryFile(dir=out.parent, prefix=out.stem + "-", suffix=".tmp", delete=False) as temp:
        pending = Path(temp.name)
    n = 0
    alerts = dict.fromkeys(POLICIES, 0)
    scoring_seconds = dict.fromkeys(VIEWS, 0.0)
    try:
        with (
            pq.ParquetFile(features) as src,
            pq.ParquetFile(truth_free_raw) as raw_source,
            pq.ParquetWriter(pending, schema, compression="zstd") as dest,
        ):
            if src.metadata.num_rows != raw_source.metadata.num_rows:
                raise ValueError("Blind input and enriched feature row counts differ")
            # Record IDs in both streams must be uninterrupted physical positions starting at 1.
            raw_rows = 0
            for batch in raw_source.iter_batches(batch_size=batch_rows, columns=["row_id"]):
                ids = batch.column("row_id").to_numpy()
                if not np.array_equal(ids, np.arange(raw_rows + 1, raw_rows + 1 + len(ids))):
                    raise ValueError("Blind input is missing or reordering physical rows")
                raw_rows += len(batch)
            for batch in src.iter_batches(batch_size=batch_rows, columns=columns):
                frame = batch.to_pandas()
                if not np.array_equal(frame.row_id, np.arange(n + 1, n + 1 + len(frame))):
                    raise ValueError("Feature input is missing or reordering physical rows")
                values = {}
                for k in VIEWS:
                    start = perf_counter()
                    values[k] = impl.score_arm(models[k], frame, ARMS[k])
                    scoring_seconds[k] += perf_counter() - start
                bits = make_decisions(values, rules, supplements, frame)
                data = {"row_id": frame.row_id, **bits, **{"score_" + k: values[k] for k in VIEWS}}
                dest.write_table(pa.table(data, schema=schema))
                for k, flagged in bits.items():
                    alerts[k] += int(flagged.sum())
                n += len(frame)
            if not n or n != raw_rows:
                raise ValueError("Empty or incomplete score output")
        pending.replace(out)
    finally:
        pending.unlink(missing_ok=True)
    return {
        "group": group,
        "rows": n,
        "device": device,
        "threads": threads,
        "batch_rows": batch_rows,
        "scoring_seconds": scoring_seconds,
        "model_score_batch": {k: getattr(models[k], "score_batch", None) for k in VIEWS},
        "frozen": rules,
        "supplements": supplements,
        "alerts": alerts,
    }
