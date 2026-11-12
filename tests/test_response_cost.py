import numpy as np
import pandas as pd
import pytest

from ids.common import read_json, write_json
from ids.contact_evidence import experiment as previous
from ids.flow_history import trees
from ids.response_evidence.cost import cost
from ids.response_evidence.experiment import ARM_NAMES, ARMS


def test_cost_checks_independent_single_row_predictions_and_flag_bytes(tmp_path, monkeypatch):
    root, name = tmp_path, "capture"
    rng = np.random.default_rng(9917)
    rows = 8193
    frame = pd.DataFrame({col: rng.normal(size=rows) for col in ARMS["quality"]})
    frame.insert(0, "row_id", np.arange(1, rows + 1))
    output = previous.enriched(root, name)
    output.parent.mkdir(parents=True)
    frame.to_parquet(output, index=False, row_group_size=977)
    ix = np.arange(0, rows, 4096)
    saved = pd.DataFrame({"row_id": frame.row_id.iloc[ix].to_numpy()})
    flags = np.full(rows, 256, dtype=np.uint16)
    rules = {}
    monkeypatch.setitem(trees.PARAMETERS, "max_iter", 2)
    train = frame.iloc[:256].assign(label=(frame.iloc[:256][ARMS["quality"][0]] > 0).astype(int))
    for arm, cut in (("quality", 0.01), ("contact", None)):
        path = previous.artifact(root, "hgb", arm)
        trees.fit_arm(train, ARMS[arm], path, threads=1)
        scores = trees.score_arm(trees.load_arm(path, threads=1), frame, ARMS[arm])
        if cut is not None:
            flags |= (scores > cut).astype(np.uint16) << ARM_NAMES.index(arm)
        rules[arm] = {"threshold": cut}
        saved[arm] = scores[ix]
    saved["flags"] = flags[ix]
    write_json(root / "evidence/calibration/hgb.json", rules)
    folder = root / "evidence/blind" / name
    folder.mkdir(parents=True)
    pd.DataFrame({"flags": flags}).to_parquet(folder / "hgb.parquet", index=False, row_group_size=211)
    sample = folder / "hgb-sample.parquet"
    saved.to_parquet(sample, index=False)
    report = read_json(cost(root, name)[0])
    assert report["row_ids"] == [1, 4097, 8193]
    assert all(value["exact_sealed_scores"] == 3 for value in report["arms"].values())
    corrupt = saved.copy()
    corrupt.loc[0, "quality"] += 1
    corrupt.to_parquet(sample, index=False)
    with pytest.raises(ValueError, match="differs from the sealed batch"):
        cost(root, name)
    corrupt = saved.copy()
    corrupt.loc[0, "flags"] ^= 256
    corrupt.to_parquet(sample, index=False)
    with pytest.raises(ValueError, match="full decision stream"):
        cost(root, name)
