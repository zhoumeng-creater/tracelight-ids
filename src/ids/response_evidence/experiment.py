"""Paired fits, normal-only thresholds and sealed decisions from blind records."""

import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ids.common import read_json, write_json
from ids.contact_evidence import experiment as previous
from ids.contact_evidence.features import ARMS as LEGACY
from ids.route_batch.metrics import decisions, threshold
from . import data
from .history import QUALITY_HISTORY_COLUMNS
from .observations import GAP_COLUMNS, QUALITY_COLUMNS

CURRENT = tuple(QUALITY_COLUMNS) + tuple(GAP_COLUMNS)
HISTORY = tuple(QUALITY_HISTORY_COLUMNS)
EXTRA = CURRENT + HISTORY
FULL = LEGACY["contact"] + EXTRA


def masked_columns(active):
    return LEGACY["contact"] + tuple(k if k in active else "zero_" + k for k in EXTRA)


ARMS = {
    "raw": LEGACY["raw"],
    "contact": LEGACY["contact"],
    "zero": masked_columns(()),
    "current": masked_columns(CURRENT),
    "prior": masked_columns(HISTORY),
    "quality": FULL,
}
ARM_NAMES = (*ARMS, "counter_masked")
COMPARISONS = (
    ("contact", "quality"),
    ("raw", "quality"),
    ("zero", "quality"),
    ("current", "quality"),
    ("prior", "quality"),
    ("contact", "current"),
    ("contact", "prior"),
    ("raw", "contact"),
    ("contact", "zero"),
    ("quality", "counter_masked"),
    ("contact", "counter_masked"),
    ("raw", "counter_masked"),
)


def with_placeholders(frame):
    return frame.assign(**{"zero_" + k: np.int8(0) for k in EXTRA})


def training(root, reg):
    path, manifest = previous.training(root, reg, features=FULL)
    frame = pd.read_parquet(path)
    with_placeholders(frame).to_parquet(path, index=False)
    import hashlib

    reference_dir = previous.data.REGISTRY.parent
    selected = pd.read_parquet(manifest)[["capture", "row_id", "family"]]
    digest = hashlib.sha256(selected.to_csv(index=False, lineterminator="\n").encode()).hexdigest()
    if digest != read_json(reference_dir / "r21-reference.json")["train_selected_sha256_csv_lf"]:
        raise ValueError("Round 23 changed the frozen training identities")
    report = root / "evidence/train-support.json"
    write_json(
        report,
        {
            "rows": len(frame),
            "families": frame.train_family.value_counts().to_dict(),
            "r21_selection_sha256": digest,
            "by_observation": frame.groupby(["train_family", *CURRENT])
            .size()
            .reset_index(name="n")
            .to_dict("records"),
            "zero_or_missing_in_training": {
                k: {"zero": int(frame[k].eq(0).sum()), "missing": int(frame[k].eq(-1).sum())}
                for k in ("orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes")
            },
        },
    )
    return [path, manifest, report]


def fit(root, group, arm, threads):
    return previous.fit(root, group, arm, threads, arms=ARMS)


def models(root, group, threads):
    return previous.models(root, group, threads, arm_names=ARMS)


def predict(group, fitted, frame):
    view = with_placeholders(frame)
    impl = previous.backend(group)
    values = {arm: impl.score_arm(fitted[arm], view, columns) for arm, columns in ARMS.items()}
    masked = values["quality"].copy()
    bad = frame.loc[:, list(GAP_COLUMNS)].any(axis=1)
    if bad.any():
        changed = mask_uncounted(frame.loc[bad])
        masked[bad.to_numpy()] = impl.score_arm(fitted["quality"], changed, FULL)
    values["counter_masked"] = masked
    return values


def mask_uncounted(frame):
    changed = frame.copy()
    for side, flag in zip(("orig", "resp"), GAP_COLUMNS):
        for field in (side + "_pkts", side + "_ip_bytes"):
            changed.loc[frame[flag].eq(1) & frame[field].eq(0), field] = -1
    return changed


def register_scores(frame):
    return {arm: threshold(frame[arm], alpha=0.01) for arm in ARM_NAMES}


def seal(scores, rules, codes):
    # Low seven bits encode decisions. The next two bits encode the observable state.
    codes = np.asarray(codes)
    if codes.ndim != 1 or not np.isin(codes, [-1, 0, 1]).all():
        raise ValueError("Invalid response-evidence state")
    flags = (codes.astype(np.int16) + 1).astype(np.uint16) << len(ARM_NAMES)
    for index, arm in enumerate(ARM_NAMES):
        decision = decisions(scores[arm], rules[arm])
        if decision.shape != flags.shape:
            raise ValueError("Unaligned evidence and decisions")
        flags |= decision.astype(np.uint8) << index
    return flags


def calibrate(root, reg, group, threads):
    fitted = models(root, group, threads)
    rows = []
    for name in previous.sources(reg, "C"):
        frame = pd.read_parquet(
            previous.enriched(root, name), columns=["row_id", "response_evidence_code", *FULL]
        )
        labels = pd.read_parquet(data.truth(root, name), columns=["row_id", "family"])
        if not labels.family.eq("Benign").all() or not np.array_equal(frame.row_id, labels.row_id):
            raise ValueError("Calibration requires aligned normal-only controls")
        rows.append(
            pd.DataFrame(
                {
                    "capture": name,
                    "row_id": frame.row_id,
                    "code": frame.response_evidence_code,
                    **predict(group, fitted, frame),
                }
            )
        )
    cal = pd.concat(rows, ignore_index=True)
    rules = register_scores(cal)
    cal["flags"] = seal(cal, rules, cal.code)
    folder = root / "evidence/calibration"
    folder.mkdir(parents=True, exist_ok=True)
    path, spec = folder / (group + ".parquet"), folder / (group + ".json")
    cal.to_parquet(path, index=False)
    write_json(spec, rules)
    return [path, spec]


def blind_score(root, name, group, threads):
    fitted = models(root, group, threads)
    rules = read_json(root / "evidence/calibration" / (group + ".json"))
    folder = root / "evidence/blind" / name
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / (group + ".parquet")
    temp = output.with_suffix(".tmp")
    n = 0
    samples = []
    start = time.perf_counter()
    with (
        pq.ParquetFile(previous.enriched(root, name)) as source,
        pq.ParquetWriter(temp, pa.schema([("flags", pa.uint16())]), compression="zstd") as writer,
    ):
        for batch in source.iter_batches(
            batch_size=previous.ROWS, columns=["row_id", "response_evidence_code", *FULL]
        ):
            frame = batch.to_pandas()
            if not np.array_equal(frame.row_id, np.arange(n + 1, n + 1 + len(frame))):
                raise ValueError("Blind stream is missing physical row IDs")
            scores = predict(group, fitted, frame)
            flags = seal(scores, rules, frame.response_evidence_code)
            writer.write_table(pa.table({"flags": flags}))
            ix = np.flatnonzero((frame.row_id.to_numpy() - 1) % 4096 == 0)
            if len(ix):
                samples.append(
                    pd.DataFrame(
                        {
                            "row_id": frame.row_id.to_numpy()[ix],
                            "flags": flags[ix],
                            **{arm: x[ix] for arm, x in scores.items()},
                        }
                    )
                )
            n += len(frame)
    if not n:
        raise ValueError("Empty blind stream")
    temp.replace(output)
    sample, audit = folder / (group + "-sample.parquet"), folder / (group + ".json")
    pd.concat(samples, ignore_index=True).to_parquet(sample, index=False)
    sec = time.perf_counter() - start
    write_json(
        audit,
        {
            "rows": n,
            "seconds": sec,
            "rows_per_second": n / sec,
            "device": "cpu" if group in previous.TREE_GROUPS else "cuda:0",
            "input": "blind features only",
            "sample_stride": 4096,
        },
    )
    return [output, sample, audit]
