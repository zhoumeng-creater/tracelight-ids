from copy import deepcopy
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from calibration_transfer.data import Preprocessor, build_episodes, identity_permutation, read_f_records


def rows():
    result = []
    for source in ("source_a", "source_b"):
        for day in range(1, 4):
            for offset in range(4):
                result.append({"flow_id": f"{source}:{day}:{offset}", "source_group": source,
                               "dataset_role": "F_base", "label": int(offset == 3),
                               "replay_export_time": datetime(2022, 1, day, tzinfo=timezone.utc).timestamp(),
                               "raw8": {"ip_protocol": 6, "orig_port": 1000 + offset, "resp_port": 443,
                                        "duration_seconds": offset, "orig_pkts": day, "resp_pkts": offset,
                                        "orig_ip_bytes": day * 100, "resp_ip_bytes": offset * 80}})
    return result


def test_role_filtered_before_parsing(tmp_path):
    path = tmp_path / "rows.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf8") as stream:
        stream.write('{"dataset_role":"C", intentionally malformed JSON}\n')
        stream.write('{"dataset_role":"V", intentionally malformed JSON}\n')
        stream.write(json.dumps({"dataset_role": "F_base", "label": 0}) + "\n")
        stream.write(json.dumps({"dataset_role": "F_expansion", "label": None}) + "\n")
    assert [r["label"] for r in read_f_records(path)] == [0, None]


def test_cumulative_episodes_and_sourcewise_bijection():
    rr = rows()
    actual = build_episodes(rr)
    random = build_episodes(rr, 17)
    mapping, details = identity_permutation(rr, 17)
    assert [(len(e["Q"]), len(e["E"])) for e in actual] == [(3, 3), (6, 3)] * 2
    assert np.array_equal(mapping, identity_permutation(rr, 17)[0])
    assert not np.array_equal(mapping, identity_permutation(rr, 29)[0])
    for info in details:
        assert sorted(info["positions"]) == sorted(info["identities"])
    for before, after in zip(actual, random):
        assert after["Q"] == mapping[before["Q"]].tolist()
        assert after["E"] == mapping[before["E"]].tolist()
        assert set(after["Q"]).isdisjoint(after["E"])
        assert all(rr[i]["label"] == 0 and rr[i]["source_group"] == before["source"] for i in after["Q"] + after["E"])
    assert all(mapping[i] == i for i, r in enumerate(rr) if r["label"] == 1)
    for pair in ((0, 1), (2, 3)):
        assert set(random[pair[0]]["E"]).issubset(random[pair[1]]["Q"])
    before = np.bincount([i for e in actual for i in e["Q"] + e["E"]], minlength=len(rr))
    after = np.bincount([i for e in random for i in e["Q"] + e["E"]], minlength=len(rr))
    assert np.array_equal(np.sort(before), np.sort(after))


def test_ports_zero_missingness_and_f_only_preprocessing():
    rr = rows()
    processor = Preprocessor.fit(rr)
    x, c = processor.transform(rr)
    changed = deepcopy(rr)
    for row in changed:
        row["raw8"]["orig_port"] = 65535
        row["raw8"]["resp_port"] = 0
        row["flow_id"] += "different_identity"
        row["source_group"] = "unseen_source"
    x2, c2 = processor.transform(changed)
    assert np.array_equal(x, x2) and np.array_equal(c, c2)
    changed[0]["raw8"]["orig_port"] = None
    assert processor.transform(changed)[1][0, -2] == 1
    assert c[0, -2] == 0
    assert x.dtype == np.float32 and c.dtype == np.int64
    assert processor.cat_cardinalities == [6, 2, 2, 2, 2, 2, 2, 2]
    restored = Preprocessor.from_dict(json.loads(json.dumps(processor.to_dict())))
    assert np.array_equal(restored.transform(rr)[0], x)
    changed[0]["dataset_role"] = "C"
    with pytest.raises(ValueError, match="F rows only"):
        Preprocessor.fit(changed)


def test_missing_constant_and_out_of_range_inputs():
    rr = rows()
    for row in rr:
        row["raw8"]["duration_seconds"] = None
        row["raw8"]["orig_pkts"] = 0
    processor = Preprocessor.fit(rr)
    numeric, categorical = processor.transform(rr)
    assert np.isfinite(numeric).all()
    assert np.all(numeric[:, :2] == 0)
    assert np.all(categorical[:, 1] == 1)
    assert processor.base.scale[1] == 1
    changed = deepcopy(rr)
    changed[0]["raw8"]["orig_ip_bytes"] = 10**12
    assert np.isfinite(processor.transform(changed)[0]).all()
    changed[0]["raw8"]["resp_port"] = float("inf")
    with pytest.raises(ValueError, match="Ports require"):
        processor.transform(changed)
