from collections import Counter
import gzip
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ids.threshold_validation.observations import RAW_FEATURES, _prepare_chunk

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/recover_r26_training_rows.py"
SPEC = importlib.util.spec_from_file_location("recover_r26_training_rows", SCRIPT)
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)

FIELDS = "ts id.orig_h id.resp_h proto duration label det_label id.orig_p id.resp_p orig_pkts resp_pkts orig_ip_bytes resp_ip_bytes"
ROWS = [
    "1 10.0.0.1 10.0.0.2 tcp - Benign - 123 443 1 - 60 -",
    "2 10.0.0.1 10.0.0.2 udp 0.5 Malicious C&C 432 53 2 4 120 200",
    "3 10.0.0.1 10.0.0.2 unknown 1 Benign - - - 0 0 0 0",
]


def fixture_log(tmp_path, fields=FIELDS):
    raw = ("#separator \\x09\n#fields " + fields + "\n#types ignored\n" + ROWS[0]
           + "\n#intermediate comment\n\n" + "\n".join(ROWS[1:]) + "\n#close end\n").encode()
    path = tmp_path / "source.gz"
    with gzip.open(path, "wb") as stream:
        stream.write(raw)
    return path, {"name": "source", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "rows": 3}


def test_physical_selection_and_original_feature_semantics(tmp_path):
    path, spec = fixture_log(tmp_path)
    result, receipt = recovery.selected_gzip(path, spec, [1, 3])
    raw = pd.DataFrame([ROWS[i].split() for i in (0, 2)],
                       columns=FIELDS.replace("det_label", "detailed-label").split())
    expected = _prepare_chunk(raw, 0, -1, -1, Counter(), Counter(), Counter())[0].to_pandas()
    assert result.row_id.tolist() == [1, 3]
    assert receipt["physical_rows"] == 3
    pd.testing.assert_frame_equal(result[list(RAW_FEATURES)], expected[list(RAW_FEATURES)])
    assert np.array_equal(result.family, expected.family)
    assert result.duration_seconds.tolist() == [-1.0, 1.0]
    assert result.ip_protocol.tolist() == [6, -1]
    assert "flow_end_ms" not in result


def test_missing_required_field_rejected(tmp_path):
    path, spec = fixture_log(tmp_path, FIELDS.replace("orig_pkts", "unknown_packets"))
    with pytest.raises(ValueError, match="Missing or ambiguous"):
        recovery.selected_gzip(path, spec, [1])


def test_source_hash_and_missing_identity_rejected(tmp_path):
    path, spec = fixture_log(tmp_path)
    with pytest.raises(ValueError, match="differs from protocol"):
        recovery.selected_gzip(path, dict(spec, sha256="0" * 64), [1])
    with pytest.raises(ValueError, match="selected identities differ"):
        recovery.selected_gzip(path, spec, [4])


def identity_table():
    singles = pd.DataFrame({"source_id": ["source", "target"], "row_id": [1, 1],
                            "family": ["Benign", "ExternalC2"], "domain": ["source", "target"],
                            "label": [0, 1], "weight": [1.0, 1.0], "support": ["source", "target"]})
    return pd.concat([singles, singles.assign(support="mixed")], ignore_index=True)


def test_identity_label_and_duplicate_mismatch():
    original = identity_table()
    mixed = recovery.registered_identities(original)
    assert len(mixed) == 2
    altered = original.copy()
    altered.loc[2, "family"] = "Scanning"
    with pytest.raises(ValueError, match="identities or labels differ"):
        recovery.registered_identities(altered)
    with pytest.raises(ValueError, match="Duplicated physical"):
        recovery.registered_identities(pd.concat([original, original.iloc[[2]]]))


def test_recovered_label_and_identity_mismatch():
    identities = recovery.registered_identities(identity_table())
    values = identities[["source_id", "row_id", "family"]].copy()
    for feature in RAW_FEATURES:
        values[feature] = 0.0
    assert recovery.align_inputs(values.iloc[::-1], identities).source_id.tolist() == ["source", "target"]
    altered = values.copy()
    altered.loc[0, "family"] = "Scanning"
    with pytest.raises(ValueError, match="author family differs"):
        recovery.align_inputs(altered, identities)
    altered = values.copy()
    altered.loc[0, "row_id"] = 9
    with pytest.raises(ValueError, match="physical identities differ"):
        recovery.align_inputs(altered, identities)
