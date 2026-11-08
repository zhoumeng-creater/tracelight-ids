"""Synthetic ZIP and role-isolation checks; never read the real NF-v3 archive."""

import csv
import hashlib
import io
import json
import zipfile

import pytest

from ids.context_cache import lab_data as lab
from ids.context_cache.preprocessing import CSVSchema
from ids.route_batch import confirmation as confirm
from ids.route_batch.data import load_inputs


def raw_row(number, label, signal=1):
    row = {name: str(signal) for name in lab.RAW_FEATURES}
    row.update(IPV4_SRC_ADDR=f"10.20.{number // 256}.{number % 256}", IPV4_DST_ADDR="192.0.2.1",
               L4_SRC_PORT="12345", L4_DST_PORT="443", PROTOCOL="6", L7_PROTO="91",
               FLOW_START_MILLISECONDS="1700000000000", FLOW_END_MILLISECONDS="1700000000001",
               Label=str(label), Attack="DDoS" if label else "Benign")
    return row


def group_for(row):
    return lab._communication_group(*(row[name] for name in (
        "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL")))


def encode_raw(rows):
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=[*lab.RAW_FEATURES, "Label", "Attack"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return text.getvalue().encode()


def write_zip(path, payload):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("synthetic-bag/data/NF-ToN-IoT-v3.csv", payload)
    return path


@pytest.fixture
def source_fixture(tmp_path, monkeypatch):
    rows = [raw_row(number + 1, label, number + 1)
            for number, label in enumerate([0, 1, 0, 0, 1, 0, 1, 0, 0, 1])]
    rows += [dict(rows[5]), dict(rows[6])]  # Unseen IDs belonging to round-13 seen groups.
    next_number = 100
    for role, labels in (("calibration", [0, 0, 0, 0, 1, 1]),
                         ("evaluation", [0, 1, 1, 1, 0, 1, 1, 1])):
        for label in labels:
            while True:
                row = raw_row(next_number, label, 1)  # Same input patterns across different groups.
                next_number += 1
                if confirm._confirmation_role(group_for(row), confirm.CONFIRMATION_SEED) == role:
                    break
            rows.append(row)
    payload = encode_raw(rows)
    source = {"sha1": hashlib.sha1(payload).hexdigest(), "sha256": hashlib.sha256(payload).hexdigest(),
              "bytes": len(payload), "dataset": lab.DATASET, "official_release_verified": True}
    monkeypatch.setattr(lab, "SOURCE_SHA1", source["sha1"])
    monkeypatch.setattr(lab, "SOURCE_BYTES", source["bytes"])
    archive = write_zip(tmp_path / "synthetic.zip", payload)
    schema = CSVSchema.from_dict(lab.SCHEMA)
    prepared = {}
    for round_name, roles in (("round14", {"support": [0, 1], "calibration": [2], "evaluation": [3, 4]}),
                              ("round13", {"support": [5, 6], "calibration": [7], "evaluation": [8, 9]})):
        root = tmp_path / round_name
        root.mkdir()
        files = {}
        for role, indices in roles.items():
            path = root / lab.FILENAMES[role]
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow([*lab.FEATURES, "label", "sample_id", "communication_group", "attack_family"])
                for index in indices:
                    row = rows[index]
                    values, _ = lab._features(list(row.values()), {name: position for position, name in enumerate(row)})
                    writer.writerow(values + [row["Label"], f"{source['sha256']}:{index + 1}",
                                               group_for(row), row["Attack"]])
            files[role] = {"path": str(path), "rows": len(indices), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        receipt = {"format": "tracelight-nf-ton-v3-preparation-v2" if round_name == "round14"
                            else "tracelight-nf-ton-v3-preparation-v1",
                   "data_kind": "development", "seed": lab.SEED, "source": source,
                   "schema": schema.to_dict(), "files": files}
        (root / "preparation.json").write_text(json.dumps(receipt), encoding="utf-8")
        prepared[round_name] = (root, files)
    current, files = prepared["round14"]
    config = {"schema": schema.to_dict(), "files": files,
              "source_zip": {"path": str(archive), "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}}
    config_path = tmp_path / "registered.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return {"archive": archive, "inputs": load_inputs(config_path), "previous": prepared["round13"][0],
            "payload": payload, "rows": rows}


def prepare(fixture, output, **kwargs):
    return confirm.prepare_confirmation(fixture["archive"], fixture["inputs"], output,
                                        previous_prepared_dir=fixture["previous"],
                                        calibration_rows=kwargs.pop("calibration_rows", 4),
                                        evaluation_rows=kwargs.pop("evaluation_rows", 8), **kwargs)


def test_excludes_both_rounds_and_entire_seen_groups_without_excluding_patterns(source_fixture, tmp_path):
    result = prepare(source_fixture, tmp_path / "out")
    assert result["status"] == "prepared" and result["seen_unique_ids"] == 10
    assert len(result["seen_evidence"]) == 6 and result["seen_unique_communication_groups"] == 10
    assert result["scan"]["source"] == {"rows": 26, "normal": 13, "attack": 13}
    assert result["scan"]["excluded_union"] == {"rows": 12, "normal": 7, "attack": 5}
    assert result["scan"]["excluded_by_seen_id"]["rows"] == 10
    assert result["scan"]["excluded_by_seen_group"]["rows"] == 12
    assert result["scan"]["calibration_attack_rows_not_sampled"] == 2
    assert result["selected_family_counts"]["evaluation"] == {"Benign": 2, "DDoS": 6}
    assert result["input_pattern_overlap"]["evaluation"]["rows_with_input_pattern_seen_before"] == 8
    assert result["source"]["member_complete_read_and_crc_verified"] is True
    assert result["source"]["registered_archive_sha256_verified"] is True
    assert result["model_runs"] == result["preprocessor_fits"] == result["threshold_registrations"] == 0


def test_calibration_quota_does_not_change_evaluation_sample(source_fixture, tmp_path):
    first = prepare(source_fixture, tmp_path / "first", calibration_rows=2, evaluation_rows=5)
    second = prepare(source_fixture, tmp_path / "second", calibration_rows=3, evaluation_rows=5)
    assert first["files"]["evaluation"]["sha256"] == second["files"]["evaluation"]["sha256"]
    assert first["sampling"]["evaluation_label_or_family_stratification"] is False


def test_evaluation_bottom_k_is_unchanged_when_only_labels_change(source_fixture):
    source_sha = "1" * 64
    first_rows = source_fixture["rows"][18:]
    changed = [{**row, "Label": "0", "Attack": "Benign"} for row in first_rows]
    def scan(rows):
        selected, _ = confirm._scan_rows(csv.reader(io.StringIO(encode_raw(rows).decode())),
                                         source_sha256=source_sha, seen_ids={}, seen_groups=set(),
                                         seed=confirm.CONFIRMATION_SEED, quotas={"calibration": 1, "evaluation": 4})
        return [row[-3] for row in selected["evaluation"]]
    assert scan(first_rows) == scan(changed)


def test_previous_round_receipt_binding_cannot_be_omitted_or_tampered(source_fixture, tmp_path):
    path = source_fixture["previous"] / "support.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="round13/support"):
        prepare(source_fixture, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_registered_archive_hash_is_checked(source_fixture, tmp_path):
    source_fixture["inputs"]["spec"]["source_zip"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="registered SHA-256"):
        prepare(source_fixture, tmp_path / "wrong-zip")


def test_complete_member_crc_is_checked(source_fixture, tmp_path):
    archive = source_fixture["archive"]
    data = bytearray(archive.read_bytes())
    start = data.find(source_fixture["payload"])
    assert start >= 0
    data[start + len(source_fixture["payload"]) - 3] ^= 1
    archive.write_bytes(data)
    source_fixture["inputs"]["spec"].pop("source_zip")
    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        prepare(source_fixture, tmp_path / "corrupt")
    assert not (tmp_path / "corrupt").exists()


def test_valid_zip_crc_does_not_replace_official_csv_digest(source_fixture, tmp_path):
    rows = [dict(row) for row in source_fixture["rows"]]
    rows[-1]["IN_BYTES"] = "2"
    replacement = encode_raw(rows)
    assert len(replacement) == len(source_fixture["payload"])
    write_zip(source_fixture["archive"], replacement)
    source_fixture["inputs"]["spec"].pop("source_zip")
    with pytest.raises(ValueError, match="Complete source CSV digest"):
        prepare(source_fixture, tmp_path / "different-source")
    assert not (tmp_path / "different-source").exists()


def test_insufficient_rows_do_not_reseed_or_write_partial_data(source_fixture, tmp_path):
    with pytest.raises(ValueError, match="seed unchanged"):
        prepare(source_fixture, tmp_path / "short", calibration_rows=5)
    assert not (tmp_path / "short").exists()
    with pytest.raises(ValueError, match="frozen"):
        prepare(source_fixture, tmp_path / "new-seed", seed=1)
