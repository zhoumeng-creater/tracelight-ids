import numpy as np
import pandas as pd
import pytest

from ids.common import read_json
from ids.contact_evidence import experiment as previous
from ids.flow_history import trees
from ids.response_evidence import data, evaluation as ev, experiment as xp


def test_target_taxonomy():
    assert ev.endpoint("C&C", "C&C") == "PureCC"
    assert ev.endpoint("Mixed", "C&C-FileDownload") == "Download"
    assert ev.endpoint("OtherAttack", "FileDownload") == "Download"
    assert ev.endpoint("Mixed", "C&C-HeartBeat") == "Heartbeat"
    assert ev.endpoint("Mixed", "C&C-HeartBeat-FileDownload") == "Download"
    assert ev.endpoint("Unresolved", "C&C") == "Unresolved"


def test_same_width_ablation_uses_only_registered_features():
    assert len({len(xp.ARMS[k]) for k in ("zero", "current", "prior", "quality")}) == 1
    for k in ("zero", "current", "prior", "quality"):
        assert xp.ARMS[k][:16] == xp.ARMS["contact"]
        assert len(set(xp.ARMS[k])) == len(xp.ARMS[k])
    assert (
        len(set(xp.ARMS["quality"]) & {"row_id", "family", "source", "destination", "uid", "ts", "label"})
        == 0
    )


def test_rank_ties_and_evidence_round_trip():
    rules = xp.register_scores({k: np.arange(200) for k in xp.ARM_NAMES})
    cut = rules["raw"]["threshold"]
    values = {k: [cut, cut + 1, cut - 1] for k in xp.ARM_NAMES}
    flags = xp.seal(values, rules, [-1, 0, 1])
    assert (flags >> len(xp.ARM_NAMES)).tolist() == [0, 1, 2]
    assert (flags & 127).tolist() == [0, 127, 0]
    with pytest.raises(ValueError):
        xp.seal(values, rules, [-1, 0, 2])


def test_checksum_gap_masks_only_affected_endpoint_and_only_zero_values():
    f = pd.DataFrame(
        {
            "orig_pkts": [0, 0, 0],
            "orig_ip_bytes": [0, 91, 0],
            "resp_pkts": [0, 0, 0],
            "resp_ip_bytes": [0, 0, 0],
            xp.GAP_COLUMNS[0]: [1, 1, 0],
            xp.GAP_COLUMNS[1]: [0, 0, 1],
        }
    )
    x = xp.mask_uncounted(f)
    assert x.orig_pkts.tolist() == [-1, -1, 0]
    assert x.orig_ip_bytes.tolist() == [-1, 91, 0]
    assert x.resp_pkts.tolist() == [0, 0, -1]
    assert x.resp_ip_bytes.tolist() == [0, 0, -1]
    assert f.orig_pkts.tolist() == [0, 0, 0]


def test_joint_zero_prior_detection_ceiling_and_budget():
    def rows(n, a, b):
        return {"rows": n, "before": a, "after": b, "net": b - a}

    counts = {"Benign": rows(199, 0, 1), **{k: rows(1, 0, 1) for k in ev.TARGETS}}
    assert ev.criteria(counts)["joint"] == "pass"
    counts["Benign"] = rows(199, 0, 2)
    assert ev.criteria(counts)["joint"] == "fail"
    counts["Benign"] = rows(199, 0, 1)
    counts["Scanning"] = rows(1, 1, 1)
    assert ev.criteria(counts)["targets"]["Scanning"]["recoverable_misses"] == 0
    assert ev.criteria(counts)["joint"] == "fail"
    counts.pop("PureCC")
    assert ev.criteria(counts)["joint"] == "not_evaluable"
    assert ev.summarize_joint([]) == "not_evaluable"
    assert ev.summarize_joint(["pass", "fail"]) == "fail"


def test_paired_gain_and_loss_are_individually_visible():
    hist = np.zeros(len(ev.BINS), np.int64)
    old = 1 << xp.ARM_NAMES.index("contact")
    new = 1 << xp.ARM_NAMES.index("quality")
    hist[128 | old], hist[256 | new], hist[new | old] = 1, 2, 3
    result = ev.pair(hist, "contact", "quality")
    assert [result[k] for k in ("rows", "before", "after", "gained", "lost", "net")] == [6, 4, 5, 2, 1, 1]


def exercise_detector(root, group):
    rng = np.random.default_rng(17)
    features = root / "features"
    features.mkdir(parents=True)

    def make(n):
        frame = pd.DataFrame(rng.integers(0, 10, (n, len(xp.FULL))), columns=xp.FULL)
        code = rng.integers(-1, 2, n, dtype=np.int8)
        for state, col in zip((-1, 0, 1), xp.CURRENT):
            frame[col] = (code == state).astype(np.int8)
        for col in xp.GAP_COLUMNS:
            frame[col] = rng.integers(0, 2, n, dtype=np.int8)
        return frame.assign(row_id=np.arange(1, n + 1), response_evidence_code=code)

    p = make(256)
    xp.with_placeholders(p).assign(
        label=p.orig_port.gt(4).astype(int), train_family=np.where(p.orig_port.gt(4), "Scanning", "Benign")
    ).to_parquet(features / "train.parquet")
    for arm in xp.ARMS:
        xp.fit(root, group, arm, 2)
    for name in ("C1", "E1"):
        f = make(256)
        f.to_parquet(previous.enriched(root, name), index=False)
        labels = pd.DataFrame({"row_id": f.row_id, "family": "Benign", "native_detail": "-"})
        if name == "E1":
            labels["window"] = np.where(labels.row_id <= 128, "day1", "day2")
            for i, pair in enumerate(
                (("Scanning", "PartOfAHorizontalPortScan"), ("C&C", "C&C"), ("Mixed", "C&C-FileDownload"))
            ):
                labels.loc[240 + i, ["family", "native_detail"]] = pair
        dest = data.truth(root, name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        labels.to_parquet(dest, index=False)
    reg = {"sources": [{"role": "C", "name": "C1"}]}
    xp.calibrate(root, reg, group, 2)
    xp.blind_score(root, "E1", group, 2)
    spec = {"role": "E", "name": "E1"}
    (path,) = ev.judge(root, spec, groups=(group,))
    item = read_json(path)["groups"][group]["contact_to_quality"]
    assert item["endpoints"]["Download"]["rows"] == 1
    assert sum(c["rows"] for c in item["by_evidence_state"]["Benign"].values()) == 253
    if group == "hgb":
        (out,) = ev.overview(root, [spec])
        assert read_json(out)["new_forward_captures"] == ["E1"]
        assert len(read_json(out)["captures"]["E1"]["windows"]) == 2


def test_synthetic_hgb_arms_through_sealed_evaluation(tmp_path, monkeypatch):
    monkeypatch.setitem(trees.PARAMETERS, "max_iter", 4)
    exercise_detector(tmp_path, "hgb")


def test_synthetic_cuda_arms_through_sealed_evaluation(tmp_path, monkeypatch):
    import torch
    from ids.flow_history import model

    if not torch.cuda.is_available():
        pytest.skip("CUDA smoke requires a physical GPU")
    monkeypatch.setattr(model, "EPOCHS", 2)
    exercise_detector(tmp_path, "mlp17")
