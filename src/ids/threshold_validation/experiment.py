"""Fit two frozen Zeek HGBs before the held-out captures and replay identical threshold policies."""

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits, threadpool_info

from ids.common import read_json, write_json
from ids.flow_history.trees import PARAMETERS
from ids.route_batch.metrics import threshold
from .observations import RAW_FEATURES, HISTORY_FEATURES
from .rules import POLICIES, audit_reference_mix, paired_operating_point, replay_thresholds

ARMS = {"raw": tuple(RAW_FEATURES), "history": (*RAW_FEATURES, *HISTORY_FEATURES)}
DELAY = 15 * 60_000


def prepared(root, name):
    return root / "prepared" / (name + ".parquet")


def inputs(registration, role):
    return [s["name"] for s in registration["sources"] if s["role"] == role]


def draw_train(root, registration):
    chunks = []
    for name in inputs(registration, "P"):
        table = pd.read_parquet(prepared(root, name), columns=["row_id", "family"])
        table["source"] = name
        chunks.append(table.loc[table.family.ne("Unresolved")])
    candidates = pd.concat(chunks, ignore_index=True)
    rng = np.random.default_rng(registration["seed"])
    picked = []
    for is_normal in (True, False):
        indices = np.flatnonzero(candidates.family.eq("Benign").to_numpy() == is_normal)
        picked.extend(rng.choice(indices, size=registration["train_rows_per_class"], replace=False))
    sample = candidates.iloc[picked].sort_values(["source", "row_id"])
    if not {"Scanning", "C&C", "Benign"}.issubset(set(sample.family)):
        raise ValueError("Registered training sample lacks a target family")
    frames = []
    for name, group in sample.groupby("source", sort=False):
        # Project only the selected training rows from these earlier sources.
        frame = pd.read_parquet(prepared(root, name), filters=[("row_id", "in", group.row_id.tolist())])
        frame = frame.sort_values("row_id").reset_index(drop=True)
        if not np.array_equal(frame.row_id, group.row_id) or not np.array_equal(frame.family, group.family):
            raise ValueError(f"Training identity changed in {name}")
        frame["source"] = name
        frames.append(frame)
    train = pd.concat(frames, ignore_index=True)
    selected = root / "evidence" / "train-selected.parquet"
    train[["source", "row_id", "family"]].to_parquet(selected, index=False)
    matrix = root / "prepared/train.parquet"
    train.to_parquet(matrix, index=False)
    return [matrix, selected]


def train_arm(root, arm, threads, registration):
    columns = list(ARMS[arm])
    fit_data = pd.read_parquet(root / "prepared/train.parquet", columns=[*columns, "family"])
    detector = HistGradientBoostingClassifier(**PARAMETERS)
    with threadpool_limits(limits=threads):
        detector.fit(fit_data[columns], fit_data.family.ne("Benign"))
        cal_scores = []
        for name in inputs(registration, "C"):
            cal = pd.read_parquet(prepared(root, name), columns=[*columns, "row_id", "family"])
            if not cal.family.eq("Benign").all():
                raise ValueError(f"Designated benign control device has nonbenign records: {name}")
            cal_scores.append(
                pd.DataFrame(
                    {"source": name, "row_id": cal.row_id, "score": detector.decision_function(cal[columns])}
                )
            )
        pools = threadpool_info()
    cal = pd.concat(cal_scores, ignore_index=True)
    cut = threshold(cal.score)
    if cut["threshold"] is None:
        raise ValueError("Insufficient past normal controls")
    directory = root / "evidence" / arm
    directory.mkdir(parents=True, exist_ok=True)
    cal_path = directory / "calibration.parquet"
    cal.to_parquet(cal_path, index=False)
    dest = root / "models/frozen" / (arm + ".joblib")
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(detector, dest, compress=3)
    fit_path = directory / "fit.json"
    write_json(
        fit_path,
        {
            "features": columns,
            "class_counts": fit_data.family.value_counts().to_dict(),
            "calibration": cut,
            "hgb": PARAMETERS,
            "iterations": detector.n_iter_,
            "threads": threads,
            "threadpools": pools,
        },
    )
    return [cal_path, dest, fit_path]


def score_arm(root, name, arm, threads):
    """Score the entire observation stream without exposing any evaluation label to the estimator."""
    columns = list(ARMS[arm])
    model = joblib.load(root / "models/frozen" / (arm + ".joblib"))
    out = root / "scores" / arm / (name + ".parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    schema = pa.schema([("row_id", pa.int64()), ("flow_end_ms", pa.int64()), ("score", pa.float64())])
    with threadpool_limits(limits=threads), pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
        for batch in pq.ParquetFile(prepared(root, name)).iter_batches(
            batch_size=65_536, columns=["row_id", "flow_end_ms", *columns]
        ):
            frame = batch.to_pandas()
            result = model.decision_function(frame[columns])
            if not np.isfinite(result).all():
                raise ValueError("Nonfinite held-out score")
            writer.write_table(
                pa.Table.from_pydict(
                    {"row_id": frame.row_id, "flow_end_ms": frame.flow_end_ms, "score": result}, schema=schema
                )
            )
    tmp.replace(out)
    return [out]


def judge(root, name):
    """Open labels for conditional feedback and evaluation only after both scoring files are closed."""
    truth = pd.read_parquet(prepared(root, name), columns=["row_id", "flow_end_ms", "family"])
    truth["attack_family"] = truth.pop("family").astype("category")
    valid = truth.attack_family.ne("Unresolved")
    truth["label"] = truth.attack_family.ne("Benign").astype("int8")
    times = truth.flow_end_ms
    release = (times + DELAY).where(truth.attack_family.eq("Benign")).astype("Int64")
    binary = truth.label.where(valid)
    out = root / "evidence" / name
    out.mkdir(parents=True, exist_ok=True)
    produced = []
    alert_cols = []
    for arm in ARMS:
        stream = pd.read_parquet(root / "scores" / arm / (name + ".parquet"))
        if not stream[["row_id", "flow_end_ms"]].equals(truth[["row_id", "flow_end_ms"]].astype("int64")):
            raise ValueError("Unaligned event or score identities")
        fixed = read_json(root / "evidence" / arm / "fit.json")["calibration"]["threshold"]
        decisions, updates = replay_thresholds(
            stream,
            fixed,
            start_ms=int(times.min()),
            end_ms=int(times.max()) + 1,
            trusted_normal_available_ms=release,
            trusted_lag_ms=DELAY,
            trusted_window_ms=86_400_000,
        )
        if not np.array_equal(decisions.row_id, truth.row_id):
            raise ValueError("Replay changed physical observation order")
        mix = audit_reference_mix(stream, updates, binary)
        for label, table in (("updates", updates), ("reference-mix", mix)):
            path = out / f"{arm}-{label}.parquet"
            table.to_parquet(path, index=False)
            produced.append(path)
        truth[arm + "_score"] = stream.score
        for policy in POLICIES:
            col = arm + "_" + policy
            truth[col] = decisions[policy + "_alert"].to_numpy()
            alert_cols.append(col)
        del decisions, stream
    judged = truth.loc[valid, ["row_id", "attack_family", "label", *alert_cols]]
    comparisons = {}
    for policy in POLICIES:
        comparisons["raw_vs_history_" + policy] = paired_operating_point(
            judged, "raw_" + policy, "history_" + policy
        )
        if policy != "fixed":
            comparisons["history_fixed_vs_" + policy] = paired_operating_point(
                judged, "history_fixed", "history_" + policy
            )
            comparisons["raw_fixed_vs_" + policy] = paired_operating_point(
                judged, "raw_fixed", "raw_" + policy
            )
    for comparison in comparisons.values():
        by = comparison["groups"]
        # Bot is the CIC2018 native class. This dataset supplies C&C as a distinct endpoint.
        by.pop("Bot", None)
        scan, cc = by["Scanning"], by.get("C&C")
        ok = bool(by["normal"]["rows"] and scan["rows"] and cc and cc["rows"])
        comparison["primary_counts_evaluable"] = ok
        comparison["within_budget_more_scan_and_no_previously_detected_cc_lost"] = (
            bool(comparison["candidate_within_one_percent"] and scan["net"] > 0 and cc["lost"] == 0)
            if ok
            else None
        )
        comparison["both_within_budget_more_scan_and_cc_preserved"] = bool(
            ok
            and comparison["baseline_within_one_percent"]
            and comparison["candidate_within_one_percent"]
            and scan["net"] > 0
            and cc["baseline"] > 0
            and cc["lost"] == 0
        )
        comparison["any_cc_detected"] = bool(cc and cc["candidate"])
        comparison["cc_all_retained_nonvacuous"] = bool(cc and cc["baseline"] > 0 and cc["lost"] == 0)
    report = out / "comparison.json"
    write_json(
        report,
        {
            "scenario": name,
            "n_observations": len(truth),
            "n_adjudicated": int(valid.sum()),
            "families": truth.attack_family.value_counts().to_dict(),
            "comparisons": comparisons,
        },
    )
    compact = out / "row-scores-and-alerts.parquet"
    truth.to_parquet(compact, index=False, compression="zstd")
    return [*produced, report, compact]
