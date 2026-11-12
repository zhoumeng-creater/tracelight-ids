"""Fit on earlier captures, freeze normal-only rules, and write compact blind decisions."""

import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ids.common import read_json, write_json
from . import data
from .features import ARMS
from .policy import GATE_COLUMNS, SINGLE, apply, contact_gate, register


TREE_GROUPS = ("hgb", "hgb_family")
NEURAL_GROUPS = ("mlp17", "mlp29", "mlp43")
GROUPS = TREE_GROUPS + NEURAL_GROUPS
ROWS = 65_536


def sources(reg, role):
    return [s["name"] for s in reg["sources"] if s["role"] == role]


def enriched(root, name):
    return root / "features" / (name + ".parquet")


def training(root, reg, *, features=ARMS["contact"]):
    pools = []
    for name in sources(reg, "P"):
        labels = pd.read_parquet(data.truth(root, name), columns=["row_id", "family"])
        pools.append(labels.loc[labels.family.ne("Unresolved")].assign(capture=name))
    candidates = pd.concat(pools, ignore_index=True)
    rng = np.random.default_rng(20261111)
    indices = []
    for is_normal in (True, False):
        pool = np.flatnonzero(candidates.family.eq("Benign").to_numpy() == is_normal)
        indices.extend(rng.choice(pool, size=4096, replace=False))
    picked = candidates.iloc[indices].sort_values(["capture", "row_id"])
    frames = []
    for name, group in picked.groupby("capture", sort=False):
        frame = pd.read_parquet(
            enriched(root, name),
            columns=["row_id", *features],
            filters=[("row_id", "in", group.row_id.tolist())],
        ).sort_values("row_id")
        if not np.array_equal(group.row_id, frame.row_id):
            raise ValueError("Training selection does not match observation identities")
        frames.append(
            frame.assign(
                label=group.family.ne("Benign").to_numpy(dtype="int8"), train_family=group.family.to_numpy()
            )
        )
    matrix = root / "features/train.parquet"
    manifest = root / "evidence/train-selected.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(matrix, index=False)
    picked.to_parquet(manifest, index=False)
    return [matrix, manifest]


def artifact(root, group, arm):
    ext = ".joblib" if group in TREE_GROUPS else ".pt"
    return root / "models/frozen" / (group + "-" + arm + ext)


def backend(group):
    if group in TREE_GROUPS:
        from ids.flow_history import trees as impl
    else:
        from ids.flow_history import model as impl
    return impl


def family_weights(family):
    # Preserve the total mass of each binary class. Split attack mass equally among observed families.
    attack = family.ne("Benign")
    sizes = family[attack].value_counts()
    if not len(sizes) or not (~attack).any():
        raise ValueError("Both training classes are required for family balancing")
    per_family = {name: float(attack.sum() / (len(sizes) * n)) for name, n in sizes.items()}
    per_family["Benign"] = 1.0
    return family.map(per_family).to_numpy(dtype=np.float64), per_family


def fit(root, group, arm, threads, *, arms=ARMS):
    path = artifact(root, group, arm)
    columns = ["label", *arms[arm]]
    if group == "hgb_family":
        columns.append("train_family")
    frame = pd.read_parquet(root / "features/train.parquet", columns=columns)
    options = {}
    mass = {}
    if group == "hgb_family":
        options["sample_weight"], mass = family_weights(frame.pop("train_family"))
    elif group in NEURAL_GROUPS:
        options = {"device": "cuda:0", "seed": int(group[3:]), "base_width": len(arms["raw"])}
    backend(group).fit_arm(frame, arms[arm], path, threads=threads, **options)
    source = path.with_suffix(".fit.json")
    report = root / "evidence/fits" / (path.stem + ".json")
    write_json(report, read_json(source) | {"train_family_sample_weights": mass})
    return [path, report]


def models(root, group, threads, *, arm_names=SINGLE):
    impl = backend(group)
    kwargs = {"threads": threads}
    if group in NEURAL_GROUPS:
        kwargs["device"] = "cuda:0"
    return {arm: impl.load_arm(artifact(root, group, arm), **kwargs) for arm in arm_names}


def predict(group, fitted, frame):
    return {arm: backend(group).score_arm(fitted[arm], frame, ARMS[arm]) for arm in SINGLE}


def replay_previous_controls(root):
    """Report independent R21 control replay before seeing new capture labels."""
    import hashlib
    from ids.route_batch.metrics import threshold

    folder = data.REGISTRY.parent
    spec = read_json(folder / "r21-reference.json")
    sample = pd.read_parquet(root / "evidence/train-selected.parquet")
    payload = sample[["capture", "row_id", "family"]].to_csv(index=False, lineterminator="\n").encode()
    if hashlib.sha256(payload).hexdigest() != spec["train_selected_sha256_csv_lf"]:
        raise ValueError("R21 training identities changed")
    prior = pd.read_parquet(folder / "r21-normal-controls.parquet")
    actual = pd.read_parquet(root / "evidence/calibration/hgb.parquet")
    if not np.array_equal(prior[["capture", "row_id"]], actual[["capture", "row_id"]]):
        raise ValueError("R21 control identities changed")
    report = {"training_ids_match": True, "calibration_ids_match": True, "arms": {}}
    for arm in ("raw", "history"):
        a, b = prior[arm].to_numpy(), actual[arm].to_numpy()
        ca, cb = threshold(a)["threshold"], threshold(b)["threshold"]
        report["arms"][arm] = {
            "max_abs_score_delta": float(np.max(np.abs(a - b))),
            "changed_calibration_alerts": int(np.sum((a > ca) != (b > cb))),
            "r21_threshold": ca,
            "new_threshold": cb,
        }
    output = root / "evidence/r21-control-replay.json"
    write_json(output, report)
    return [output]


def calibrate(root, reg, group, threads):
    fitted = models(root, group, threads)
    frames = []
    for name in sources(reg, "C"):
        frame = pd.read_parquet(enriched(root, name), columns=["row_id", *ARMS["contact"]])
        labels = pd.read_parquet(data.truth(root, name), columns=["row_id", "family"])
        if not np.array_equal(labels.row_id, frame.row_id) or not labels.family.eq("Benign").all():
            raise ValueError("Normal calibration capture has wrong labels or row identities")
        scores = predict(group, fitted, frame)
        frames.append(
            pd.DataFrame({"capture": name, "row_id": frame.row_id, "gate": contact_gate(frame), **scores})
        )
    cal = pd.concat(frames, ignore_index=True)
    rules = register({arm: cal[arm] for arm in SINGLE}, cal.gate.to_numpy())
    raw_values = {arm: cal[arm].to_numpy() for arm in SINGLE}
    flags = apply(rules, raw_values, cal.gate.to_numpy())
    cal["flags"] = flags
    from .policy import POLICIES

    breakdown = {}
    for capture, rows in cal.groupby("capture", sort=False):
        bits = rows["flags"].to_numpy()
        alerts = {p: int(((bits >> i) & 1).sum()) for i, p in enumerate(POLICIES)}
        anchor = ((bits >> POLICIES.index("anchor")) & 1).astype(bool)
        breakdown[capture] = {
            "n": len(rows),
            "gate": int(rows.gate.sum()),
            "eligible_outside_anchor": int((rows.gate.to_numpy() & ~anchor).sum()),
            "alerts": alerts,
        }
    folder = root / "evidence/calibration"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (group + ".parquet")
    cal.to_parquet(path, index=False)
    spec = folder / (group + ".json")
    write_json(spec, rules)
    detail = folder / (group + "-support.json")
    write_json(detail, breakdown)
    return [path, spec, detail]


def blind_score(root, name, group, threads):
    fitted = models(root, group, threads)
    rules = read_json(root / "evidence/calibration" / (group + ".json"))
    folder = root / "evidence/blind" / name
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / (group + ".parquet")
    sample = folder / (group + "-sample.parquet")
    audit = folder / (group + ".json")
    n = 0
    samples = []
    start = time.perf_counter()
    with pq.ParquetWriter(
        dest.with_suffix(".tmp"), pa.schema([("flags", pa.uint8())]), compression="zstd"
    ) as writer:
        for batch in pq.ParquetFile(enriched(root, name)).iter_batches(
            batch_size=ROWS, columns=["row_id", *ARMS["contact"]]
        ):
            frame = batch.to_pandas()
            if not np.array_equal(frame.row_id, np.arange(n + 1, n + 1 + len(frame))):
                raise ValueError("Nonconsecutive source row IDs in blind stream")
            values = predict(group, fitted, frame)
            flags = apply(rules, values, contact_gate(frame))
            writer.write_table(pa.table({"flags": flags}))
            ix = np.flatnonzero((frame.row_id.to_numpy() - 1) % 4096 == 0)
            if len(ix):
                rows = frame.iloc[ix][["row_id", *GATE_COLUMNS]].copy()
                for arm, vector in values.items():
                    rows[arm] = vector[ix]
                rows["flags"] = flags[ix]
                samples.append(rows)
            n += len(frame)
    if n == 0:
        raise ValueError("Empty evaluation stream")
    dest.with_suffix(".tmp").replace(dest)
    pd.concat(samples, ignore_index=True).to_parquet(sample, index=False)
    sec = time.perf_counter() - start
    write_json(
        audit,
        {
            "rows": n,
            "first_row_id": 1,
            "last_row_id": n,
            "seconds": sec,
            "rows_per_second": n / sec,
            "score_device": "cpu" if group in TREE_GROUPS else "cuda:0",
            "sample_every": 4096,
            "input": "blind observations only",
        },
    )
    return [dest, sample, audit]
