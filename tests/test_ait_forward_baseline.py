from hashlib import sha256
from zipfile import ZipFile

import joblib
import numpy as np
import pytest

from scripts.ait_forward import baseline


def test_archive_member_bytes_must_match_trusted_pin_before_loading(tmp_path, monkeypatch):
    archive = tmp_path / "wrong.zip"
    data = b"untrusted"
    trusted = b"expected!"
    assert len(data) == len(trusted)
    monkeypatch.setattr(baseline, "_MEMBERS", {
        baseline.MODEL_MEMBER: (len(trusted), sha256(trusted).hexdigest()),
    })
    with ZipFile(archive, "w") as writer:
        writer.writestr(baseline.ARCHIVE_PREFIX + baseline.MODEL_MEMBER, data)

    def reject_load(_):
        pytest.fail("Deserializer ran before archive member verification")

    monkeypatch.setattr(joblib, "load", reject_load)
    with pytest.raises(ValueError, match="Frozen archive member hash differs"):
        baseline.load_baseline(archive)


def test_cpu_scorer_uses_frozen_field_order_and_keeps_port_missingness():
    class SpyModel:
        def decision_function(self, features):
            np.testing.assert_equal(features, [[6, 0, np.nan, 0.1, 5, 7, 500, 700]])
            return np.array([0.5])

    predictor = baseline.FrozenBaseline(
        model=SpyModel(),
        model_bytes=b"",
        pooled_threshold=0.0,
        calibration_scores={},
        calibration_units={},
        known_flow_ids=frozenset(),
        known_stable_flow_ids=frozenset(),
        last_calibration_time=0.0,
        archived_fox_scores={},
        archived_fox_identities=(),
        archived_fox_units={},
        archived_fox_summary={},
    )
    score = predictor.score([{
        "raw8": {
            "ip_protocol": 6,
            "orig_port": 23456,
            "resp_port": None,
            "duration_seconds": 0.1,
            "orig_pkts": 5,
            "resp_pkts": 7,
            "orig_ip_bytes": 500,
            "resp_ip_bytes": 700,
        },
    }])
    assert score.dtype == np.float64
    np.testing.assert_equal(score, [0.5])
