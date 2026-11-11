"""Only synthetic records, including reversed endpoints, boundaries and unavailable roles."""

import csv
import json
import zipfile

import pandas as pd
import pyarrow as pa
import pytest

from ids.common import read_json, sha256_file, write_json
from ids.context_cache import lab_data as lab
from ids.flow_history import cohort as c


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    source = tmp_path / "synthetic.csv"
    rows = []
    for lower, upper in c.INTERVALS.values():
        for i in range(300):
            end = (lower, lower + 59999, lower + 60000, upper - 1, upper)[i % 5]
            values = {k: "0" for k in lab.RAW_FEATURES}
            values.update(IPV4_SRC_ADDR="192.0.2.2", IPV4_DST_ADDR="192.0.2.1", L4_SRC_PORT=str(3000 + i),
                          L4_DST_PORT="443", PROTOCOL="6", FLOW_START_MILLISECONDS=str(end - 1),
                          FLOW_END_MILLISECONDS=str(end), IN_BYTES=str(i), Label=str(i % 2),
                          Attack="scanning" if i % 2 else "Benign")
            if i % 7 == 0:
                values.update(IPV4_SRC_ADDR="192.0.2.1", IPV4_DST_ADDR="192.0.2.2",
                              L4_SRC_PORT="443", L4_DST_PORT=str(3000 + i))
            rows.append(values)
    with source.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    source_sha = sha256_file(source)
    monkeypatch.setattr(c, "SOURCE_ROWS", len(rows))
    monkeypatch.setattr(c, "SOURCE_SHA", source_sha)
    monkeypatch.setattr(c, "QUOTAS", {"P_normal": 3, "P_attack": 4, "C_normal": 5, "E_all": 7})
    run = tmp_path / "run"
    prep = run / "confirmation" / "prepared" / "preparation.json"
    seen, files = [], {}
    opened_roles = [(name, role) for name in ["round13", "round14"] for role in lab.ROLES]
    opened_roles += [("round17", role) for role in ["calibration", "evaluation"]]
    for i, (name, role) in enumerate(opened_roles):
        idx = (i + 1) * 10
        row = rows[idx]
        group = lab._communication_group(row["IPV4_SRC_ADDR"], row["IPV4_DST_ADDR"], row["L4_SRC_PORT"],
                                         row["L4_DST_PORT"], row["PROTOCOL"])
        frame = pd.DataFrame([{**{f: row[f] for f in lab.FEATURES}, "label": int(row["Label"]),
                               "sample_id": f"{source_sha}:{idx + 1}", "communication_group": group,
                               "attack_family": lab.FAMILIES[row["Attack"].casefold()]}])
        path = tmp_path / f"{name}-{role}.csv"
        frame.to_csv(path, index=False)
        item = {"path": str(path), "rows": 1, "sha256": sha256_file(path)}
        if name == "round17":
            files[role] = item
        else:
            seen.append({**item, "role": role, "round": name})
    member = "data/NF-ToN-IoT-v3.csv"
    archive = tmp_path / "raw.zip"
    with zipfile.ZipFile(archive, "w") as out:
        out.write(source, member)
    monkeypatch.setattr(c, "ARCHIVE_SHA", sha256_file(archive))
    write_json(prep, {"source": {"member": member, "bytes": source.stat().st_size,
                                 "archive_path": str(archive)}, "seen_evidence": seen, "files": files})
    original_receipt = tmp_path / "original.json"
    write_json(original_receipt, {"source": {"path": str(source)}})
    write_json(run / "protocol.json", {"files": {s["role"]: s for s in seen if s["round"] == "round14"},
        "artifacts": {"preparation": {"path": str(original_receipt), "sha256": sha256_file(original_receipt)}}})
    monkeypatch.setattr(c, "PROTOCOL_SHA", sha256_file(run / "protocol.json"))
    monkeypatch.setattr(c, "PREPARATION_SHA", sha256_file(prep))
    return source, run, archive, rows


def test_registered_source_plain_zip_and_identity(tiny, tmp_path):
    source, run, archive, _ = tiny
    assert c.resolve_source(run) == source
    expanded = c.resolve_source(run, archive, work=tmp_path / "stage")
    assert expanded.read_bytes() == source.read_bytes()
    assert c.resolve_source(run, archive, work=tmp_path / "stage") == expanded
    expanded.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        c.resolve_source(run, archive, work=tmp_path / "stage")


def test_six_roles_are_disjoint_registered_and_resumable(tiny, tmp_path, monkeypatch):
    source, run, _, rows = tiny
    output = tmp_path / "cohorts"
    frames = c.prepare_cohorts(run, source, output, 2)
    assert set(frames) == {"P", "C", "E", *c.INTERVALS}
    assert frames["P_time"].label.value_counts().to_dict() == {1: 4, 0: 3}
    assert len(frames["C_time"]) == 5 and not frames["C_time"].label.any()
    assert len(frames["E_time"]) == 7
    _, seen = c._read_opened(run)
    for role, (left, right) in c.INTERVALS.items():
        frame = frames[role]
        ids = frame.row_id.map(lambda value: int(value.split(":")[1]))
        assert frame.row_id.is_unique and not set(frame.communication_group) & seen
        assert frame.flow_end_ms.between(left + 60000, right - 1).all()
        assert frame.IN_BYTES.tolist() == [float(rows[i - 1]["IN_BYTES"]) for i in ids]
        assert frame.label.tolist() == [int(rows[i - 1]["Label"]) for i in ids]
        seen.update(frame.communication_group)
    first = read_json(output / "cohort_manifest.json")
    assert first["forward"]["binary_evaluable"] is False  # <99 calibrators in the synthetic fixture
    monkeypatch.setattr(c, "_metadata", lambda *args: pytest.fail("already prepared"))
    again = c.prepare_cohorts(run, source, output, 1)
    for role in frames:
        pd.testing.assert_frame_equal(frames[role], again[role])
    with (output / "C_time.parquet").open("ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="SHA-256"):
        c.prepare_cohorts(run, source, output, 2)


def test_e_blind_to_labels_shortfall_and_split_boundary(tiny, tmp_path, monkeypatch):
    source, run, _, _ = tiny
    meta = c._metadata(source, 2)
    def rng(t):
        return c._sample_indices(t, set(), intervals=c.INTERVALS, quotas=c.QUOTAS, seed=c.SEED)
    indices, summary = rng(meta)
    edited = meta.set_column(meta.column_names.index("Label"), "Label", pa.array(1 - meta["Label"].to_numpy()))
    assert indices["E_all"] == rng(edited)[0]["E_all"]
    assert all(row[3] == c.INTERVALS["E_time"][0] + 60000 or row[3] == c.INTERVALS["E_time"][1] - 1
               for row in indices["E_all"])
    assert summary["P_attack"]["quota_met"]
    monkeypatch.setattr(c, "INTERVALS", {k: (0, 1) for k in c.INTERVALS})
    result = c.prepare_cohorts(run, source, tmp_path / "missing-period", 2)
    assert not result["P"].empty and all(result[k].empty for k in c.INTERVALS)
    assert all(result[k].columns.tolist() == c.FIELDS + ["flow_end_ms"] for k in c.INTERVALS)
    receipt = json.loads((tmp_path / "missing-period" / "cohort_manifest.json").read_text())
    assert not receipt["forward"]["scanning_evaluable"]
