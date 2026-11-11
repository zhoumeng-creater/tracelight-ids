"""Score an external, completion-ordered NF-v3 stream with both original frozen R20 HGB models.

Score first; submit independent per-row truth only to the separate evaluate command. The feed must
supply a physically ordered complete capture where each flow is observable at its end time.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
from itertools import groupby
from pathlib import Path
from zipfile import ZipFile
import json

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq

from ids.common import read_json, sha256_file, write_json
from ids.context_cache.lab_data import FEATURES
from ids.flow_history import trees
from ids.flow_history.features import HISTORY_COLUMNS, WINDOW_MS, _RollingState, _ip_code
from ids.route_batch.metrics import threshold
from .cic_replay import ARCHIVE_SHA, MODELS
from .rules import paired_operating_point, replay_thresholds


LAST_R20_EXCLUSIVE = 1_520_035_200_000  # End of the frozen R20 forward interval (UTC).
INPUT = ["row_id", "observed_ms", "FLOW_END_MILLISECONDS", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", *FEATURES]
POLICIES = ("fixed", "unlabeled")
PREDICTIONS = "frozen_scores_and_alerts.parquet"
RECEIPT = "score_receipt.json"
RULE = {"refresh_ms": 900_000, "window_ms": 3_600_000, "min_reference": 1000}


def load_models(archive, threads):
    if sha256_file(archive) != ARCHIVE_SHA:
        raise ValueError("Only the original verified R20 result ZIP is supported")
    models, boundaries = {}, {}
    with ZipFile(archive) as saved:
        for name, digest in MODELS.items():
            prefix = f"evidence/forward/{name}/"
            blob = saved.read(f"models/forward/{name}.joblib")
            fit = json.loads(saved.read(prefix + "fit.json"))
            if sha256(blob).hexdigest() != digest or fit["model_sha256"] != digest:
                raise ValueError("Frozen model file digest mismatch")
            models[name] = trees.load_arm(BytesIO(blob), threads=threads)
            expected = tuple(FEATURES) + (tuple(HISTORY_COLUMNS) if name == "hgb52_history" else ())
            if tuple(models[name].features) != expected:
                raise ValueError("Frozen model schema mismatch")
            normal = pd.read_parquet(BytesIO(saved.read(prefix + "calibration.parquet")))
            if len(normal) != 10000 or not normal.label.eq(0).all() or not normal.row_id.is_unique:
                raise ValueError("R20 fixed calibration is not the registered normal population")
            boundaries[name] = threshold(normal.score)["threshold"]
            if boundaries[name] != json.loads(saved.read(prefix + "metrics.json"))["threshold"]["threshold"]:
                raise ValueError("Frozen calibration threshold mismatch")
    return models, boundaries


class CompletionHistory:
    """Keep all updates at the current timestamp pending, including across Parquet batch boundaries."""

    def __init__(self):
        self.state = _RollingState()
        self.instant = None
        self.pending = []

    def observe(self, flows):
        clocks = flows["observed_ms"]
        end = flows["FLOW_END_MILLISECONDS"]
        if not pd.api.types.is_integer_dtype(clocks.dtype) or not pd.api.types.is_integer_dtype(end.dtype):
            raise ValueError("Observed and completion times must be integer Unix milliseconds")
        if clocks.isna().any() or end.isna().any() or not np.array_equal(clocks, end):
            raise ValueError("This replay requires an actual observation at each flow completion")
        if not clocks.is_monotonic_increasing or (self.instant is not None and clocks.iloc[0] < self.instant):
            raise ValueError("Physical source is not ordered by observation time")
        src = flows.IPV4_SRC_ADDR.map(_ip_code).to_numpy()
        dst = flows.IPV4_DST_ADDR.map(_ip_code).to_numpy()
        numbers = flows[["PROTOCOL", "L4_DST_PORT", "OUT_PKTS"]].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(numbers).all()
            or (numbers < 0).any()
            or (np.floor(numbers) != numbers).any()
            or (numbers[:, 0] > 255).any()
            or (numbers[:, 1] > 65535).any()
        ):
            raise ValueError("Invalid protocol, destination port or return packet count")
        key = (src.astype(np.uint64) << 8) | numbers[:, 0].astype(np.uint64)
        stored = np.empty((len(flows), 4), dtype=np.float64)
        for tick, grouped in groupby(enumerate(clocks), key=lambda row: row[1]):
            if self.instant != tick:
                for source, destination, port, unreturned in self.pending:
                    self.state.update(self.instant, source, destination, port, unreturned)
                self.pending.clear()
                self.state.expire(tick)
                self.instant = tick
            for i, _ in grouped:
                source = int(key[i])
                stored[i] = self.state.features(source)
                self.pending.append((source, int(dst[i]), int(numbers[i, 1]), int(numbers[i, 2] == 0)))
        return stored


def score_observations(file, models, start, end):
    physical = pq.ParquetFile(file)
    if physical.metadata.num_rows == 0 or not set(INPUT).issubset(physical.schema_arrow.names):
        raise ValueError("External input must contain the full NF-v3 stream schema")
    first_seen, last_seen = None, None
    history = CompletionHistory()
    parts = []
    raw_before_end = 0
    warmup_ids = set()
    for batch in physical.iter_batches(batch_size=65_536, columns=INPUT):
        flows = batch.to_pandas()
        values = history.observe(flows)
        if first_seen is None:
            first_seen = int(flows.observed_ms.iloc[0])
            if first_seen < LAST_R20_EXCLUSIVE or first_seen > start - WINDOW_MS:
                raise ValueError(
                    "Provide a later external stream and at least 60 seconds before scoring starts"
                )
        last_seen = int(flows.observed_ms.iloc[-1])
        raw_before_end += int(flows.observed_ms.lt(end).sum())
        initial_ids = flows.loc[flows.observed_ms.lt(first_seen + WINDOW_MS), "row_id"]
        if initial_ids.isna().any() or not initial_ids.is_unique or not warmup_ids.isdisjoint(initial_ids):
            raise ValueError("External row IDs must be globally unique")
        warmup_ids.update(initial_ids)
        # The first minute populates history; truncated history must not enter threshold calibration.
        keep = flows.observed_ms.ge(first_seen + WINDOW_MS) & flows.observed_ms.lt(end)
        if not keep.any():
            continue
        flows = flows.loc[keep].copy()
        flows[list(HISTORY_COLUMNS)] = values[keep.to_numpy()]
        chunk = flows[["row_id", "observed_ms"]].rename(columns={"observed_ms": "flow_end_ms"})
        for arm, model in models.items():
            chunk[arm] = trees.score_arm(model, flows, model.features)
        parts.append(chunk)
    if last_seen is None or last_seen < end - 1 or not parts:
        raise ValueError("Physical capture does not span the complete prespecified interval")
    scores = pd.concat(parts, ignore_index=True)
    if scores.row_id.isna().any() or not scores.row_id.is_unique or not warmup_ids.isdisjoint(scores.row_id):
        raise ValueError("External row IDs must be globally unique")
    return scores, first_seen, last_seen, raw_before_end


def predict(archive, file, start, end, output, threads):
    if start < LAST_R20_EXCLUSIVE or end <= start or threads < 1:
        raise ValueError("Invalid later forward interval or CPU budget")
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / RECEIPT
    out = output / PREDICTIONS
    # An existing sealed prediction is never redefined using subsequently obtained truth.
    if receipt.exists():
        record = read_json(receipt)
        if (
            record["observations_sha256"] != sha256_file(file)
            or record["interval_ms"] != [start, end]
            or record["predictions_sha256"] != sha256_file(out)
            or sha256_file(archive) != ARCHIVE_SHA
            or record["updates_sha256"] != sha256_file(output / "frozen_threshold_updates.parquet")
        ):
            raise ValueError("Existing frozen prediction receipt has a different input, interval or artifact")
        return record
    models, thresholds = load_models(archive, threads)
    source_sha = sha256_file(file)
    scores, first, last, raw_count = score_observations(file, models, start, end)
    predictions = scores.loc[scores.flow_end_ms.ge(start)].reset_index(drop=True)
    if predictions.empty:
        raise ValueError("No rows in the later evaluation interval")
    audits = []
    for arm in MODELS:
        stream = scores[["row_id", "flow_end_ms", arm]].rename(columns={arm: "score"})
        results, updates = replay_thresholds(stream, thresholds[arm], start_ms=start, end_ms=end, **RULE)
        if not results.row_id.equals(predictions.row_id):
            raise ValueError("Sealed score identity mismatch")
        for policy in POLICIES:
            predictions[arm + "/" + policy] = results[policy + "_alert"]
        active = updates.loc[updates.policy.isin(POLICIES)].copy()
        active.insert(0, "arm", arm)
        audits.append(active)
    if sha256_file(file) != source_sha:
        raise ValueError("External observation file changed while reading")
    out_tmp = out.with_suffix(".parquet.part")
    predictions.to_parquet(out_tmp, index=False)
    out_tmp.replace(out)
    update_path = output / "frozen_threshold_updates.parquet"
    pd.concat(audits, ignore_index=True).to_parquet(update_path, index=False)
    record = {
        "schema": "r20_frozen_external_nfv3_v1",
        "interval_ms": [start, end],
        "R20_result_sha256": ARCHIVE_SHA,
        "frozen_model_sha256": MODELS,
        "observations_sha256": source_sha,
        "predictions_sha256": sha256_file(out),
        "updates_sha256": sha256_file(update_path),
        "rule": RULE,
        "fixed_thresholds": thresholds,
        "physical_rows_used_before_end": raw_count,
        "history_only_warmup_rows": raw_count - len(scores),
        "prior_scored_reference_rows": len(scores) - len(predictions),
        "first_observed_ms": first,
        "last_observed_ms": last,
        "evaluation_rows": len(predictions),
        "coverage": "Physical full NF-v3 source supplied by operator. Requires immediate completed-flow arrival; rows missing from capture and label provenance cannot be inferred from scores.",
        "truth_opened_by_scoring": False,
    }
    write_json(receipt, record)
    return record


def evaluate(truth_file, output):
    frozen = read_json(output / RECEIPT)
    saved = output / PREDICTIONS
    if (
        frozen["schema"] != "r20_frozen_external_nfv3_v1"
        or frozen["frozen_model_sha256"] != MODELS
        or frozen["predictions_sha256"] != sha256_file(saved)
    ):
        raise ValueError("Missing or changed frozen predictions")
    predictions = pd.read_parquet(saved)
    truth = pd.read_parquet(truth_file, columns=["row_id", "label", "attack_family"])
    if (
        truth.row_id.isna().any()
        or not truth.row_id.is_unique
        or set(truth.row_id) != set(predictions.row_id)
    ):
        raise ValueError("Truth must cover precisely all evaluation row IDs once")
    rows = predictions.merge(truth, on="row_id", validate="one_to_one")
    result = {
        "source_prediction_receipt_sha256": sha256_file(output / RECEIPT),
        "truth_sha256": sha256_file(truth_file),
        "same_policy_between_models": {
            k: paired_operating_point(rows, "hgb48/" + k, "hgb52_history/" + k) for k in POLICIES
        },
        "updated_vs_fixed_within_model": {
            arm: paired_operating_point(rows, arm + "/fixed", arm + "/unlabeled") for arm in MODELS
        },
        "has_scan_bot_and_normal": {"Scanning", "Bot", "Benign"}.issubset(set(rows.attack_family)),
    }
    write_json(output / "frozen_evaluation.json", result)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="cmd", required=True)
    score = commands.add_parser("score")
    score.add_argument("--archive", type=Path, required=True)
    score.add_argument("--observations", type=Path, required=True)
    score.add_argument("--start-ms", type=int, required=True)
    score.add_argument("--end-ms", type=int, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--threads", type=int, default=min(24, psutil.cpu_count(logical=False) or 1))
    check = commands.add_parser("evaluate")
    check.add_argument("--truth", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.cmd == "score":
        predict(args.archive, args.observations, args.start_ms, args.end_ms, args.output, args.threads)
    else:
        evaluate(args.truth, args.output)


if __name__ == "__main__":
    main()
