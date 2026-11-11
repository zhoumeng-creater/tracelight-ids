import hashlib

import numpy as np
import pandas as pd
import pytest

from ids.capture_validation import cohort as co


def _row(src, dst, sport, when, label, family, *, dport=443, proto=6, syn=False):
    r = dict.fromkeys(co.lab.FEATURES, 0)
    r.update(
        IPV4_SRC_ADDR=src,
        IPV4_DST_ADDR=dst,
        L4_SRC_PORT=sport,
        L4_DST_PORT=dport,
        PROTOCOL=proto,
        FLOW_START_MILLISECONDS=when,
        FLOW_END_MILLISECONDS=when + 1,
        Label=label,
        Attack=family,
        IN_PKTS=1,
        OUT_PKTS=0 if syn else 1,
        TCP_FLAGS=2 if syn else 18,
        CLIENT_TCP_FLAGS=2 if syn else 18,
    )
    return r


def _ports(src, dst, role, n):
    s = int(co.IPv4Address(src))
    d = int(co.IPv4Address(dst))
    seq = np.arange(1000, 65535, dtype=np.uint64)

    def const(value):
        return np.full(len(seq), value, dtype=np.uint64)

    bins = co._bucket(const(s), const(d), seq, const(443), const(6))
    m = {"P_time": bins < 50, "C_time": (bins >= 50) & (bins < 70), "E_time": bins >= 70}[role]
    return seq[m][:n].tolist()


@pytest.fixture
def native(tmp_path):
    rows = []
    for role, ts, host in [
        ("P_time", co.RULES[0][1] // 1000000 + 1000, "172.31.69.24"),
        ("E_time", co.RULES[1][1] // 1000000 + 1000, "172.31.69.13"),
    ]:
        ports = _ports(host, "172.31.69.1", role, 3)
        rows.append(_row(host, "172.31.69.1", ports[0], ts, 0, "Benign", syn=True))
        rows.append(_row(host, "172.31.69.1", ports[1], ts, 1, "Infilteration", syn=True))
        rows.append(_row("172.31.69.1", host, 443, ts, 0, "Benign", dport=ports[0], syn=True))
    confirmed = np.array([1, 4], dtype="<u4")
    for role, ts, n in [
        ("P_time", co.P_END - 200000, 8),
        ("C_time", co.P_END + co.WINDOW, 103),
        ("E_time", co.C_END + co.WINDOW, 12),
    ]:
        for i, port in enumerate(_ports("10.0.0.1", "10.0.0.2", role, n)):
            attack = role != "C_time" and i % 2 == 0
            rows.append(
                _row(
                    "10.0.0.1",
                    "10.0.0.2",
                    port,
                    ts,
                    int(attack),
                    "Bot" if attack else "Benign",
                    syn=not attack,
                )
            )
    # All native Infilteration outside the NMAP rule must be excluded as well.
    rows.append(
        _row(
            "10.0.0.1",
            "10.0.0.2",
            _ports("10.0.0.1", "10.0.0.2", "E_time", 1)[0],
            co.C_END + 90000,
            1,
            "Infilteration",
        )
    )
    p = tmp_path / "native.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p, rows, confirmed


def _prepare(path, rows, confirmed, **options):
    return co.prepare_targets(
        path,
        confirmed,
        expected_rows=len(rows),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        quotas={"P_attack": 5, "P_normal": 4, "C_time": 100, "E_time": 13},
        **options,
    )


def test_confirmed_only_chronology_and_group_isolation(native):
    path, rows, confirmed = native
    out, meta = _prepare(path, rows, confirmed, block_size=1500)
    assert meta["eligible_for_training"] and all(meta["coverage_checks"].values())
    assert meta["excluded"] == {
        "uncertain_scan_rows": 4,
        "remaining_native_infiltration": 1,
        "verified_original_native_benign": 2,
    }
    assert meta["subgroup_evaluable"] == {"one_unanswered_syn": True, "original_47_field_signature": False}
    assert meta["rows_by_role"]["E_time"]["scan_on_unseen_sources"] == 1
    assert meta["rows_by_role"]["E_time"]["normal_syn"] == 6
    assert sorted(out.loc[out["attack_family"].eq("Scanning"), "source_label"]) == [0, 0]
    assert set(out.role) == set(co.ROLES)
    assert out.row_id.is_unique and all(k in out for k in co.lab.FEATURES)
    assert not out[out.role.eq("C_time")].label.any() and "Infiltration" not in out.attack_family.unique()
    assert out.groupby("communication_group").role.nunique().max() == 1
    alt, other = _prepare(path, rows, confirmed, threads=2, block_size=5012)
    pd.testing.assert_frame_equal(out, alt)
    assert meta == other


def test_unconfirmed_boundary_and_reverse_are_quarantined(native):
    path, rows, confirmed = native
    rule = co.RULES[1]
    # First reported ms contains times before the source author's lower boundary.
    rows[3]["FLOW_START_MILLISECONDS"] = rule[1] // 1000000
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match="directed reference"):
        _prepare(path, rows, confirmed)
    out, meta = _prepare(path, rows, confirmed[:1])
    assert (
        not meta["eligible_for_training"]
        and meta["rows_by_role"]["E_time"]["families"].get("Scanning", 0) == 0
    )
    assert not (out["source_family"] == "Infilteration").any()
    for forbidden in [2, 3, 4, 5, 6, len(rows)]:
        assert forbidden not in {int(x.rsplit(":", 1)[1]) for x in out.row_id}


def test_source_integrity_and_identity_fail_closed(native):
    path, rows, verified = native
    with pytest.raises(ValueError, match="hash, row total"):
        co.prepare_targets(path, verified)
    for ids in (
        verified[::-1],
        np.array([1, 1], dtype="<u4"),
        np.array([0, 1], dtype="<u4"),
        np.array([1, 4], dtype="int64"),
    ):
        with pytest.raises(ValueError, match="ascending"):
            _prepare(path, rows, ids)
    with pytest.raises(ValueError, match="confirmed identity coverage"):
        _prepare(path, rows, np.array([len(rows) + 1], dtype="<u4"))


def test_evaluation_priority_not_conditioned_on_native_attack_family(native):
    path, rows, verified = native
    quotas = {"P_attack": 2, "P_normal": 2, "C_time": 100, "E_time": 6}

    def selected():
        f, _ = co.prepare_targets(
            path,
            verified,
            expected_rows=len(rows),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            quotas=quotas,
        )
        return {int(s.split(":")[-1]) for s in f.loc[f.role.eq("E_time"), "row_id"]}

    initial = selected()
    for row in rows:
        if row["FLOW_START_MILLISECONDS"] == co.C_END + co.WINDOW and row["Attack"] in {"Bot", "Benign"}:
            row["Attack"] = "Bot" if row["Label"] == 0 else "Benign"
            row["Label"] = 1 - row["Label"]
    pd.DataFrame(rows).to_csv(path, index=False)
    assert selected() == initial


def test_shape_overlap_distinguishes_pooled_from_concurrent_matches():
    keys = dict.fromkeys(co.SIGNATURE_FIELDS, 2)
    frame = pd.DataFrame(
        [
            {**keys, "attack_family": "Scanning", "end_day_utc": "2018-03-01"},
            {**keys, "attack_family": "Benign", "end_day_utc": "2018-03-02"},
            {**keys, "attack_family": "Bot", "end_day_utc": "2018-03-01"},
            {**keys, "L4_DST_PORT": 80, "attack_family": "Scanning", "end_day_utc": "2018-03-01"},
            {**keys, "L4_DST_PORT": 80, "attack_family": "Benign", "end_day_utc": "2018-03-01"},
        ]
    )
    overlap = co.signature_overlap(frame)
    assert overlap["all_days"]["matched_scan_rows"] == overlap["all_days"]["matched_normal_rows"] == 2
    assert overlap["within_day"]["matched_scan_rows"] == overlap["within_day"]["matched_normal_rows"] == 1
    assert (overlap["all_days"]["distinct_signatures"], overlap["within_day"]["matched_keys"]) == (2, 1)


def test_distinct_native_packet_reference_bound_to_verified_class(native):
    path, rows, confirmed = native
    packet = np.array([2, 5], dtype="<u4")
    f, manifest = _prepare(path, rows, confirmed, packet_rows=packet)
    assert (
        manifest["packet_confirmed_rows"] == 2
        and manifest["confirmed_rows"] == 2
        and manifest["all_confirmed_rows"] == 4
    )
    assert any(
        row["truth_source"] == "published_Nmap_event_packet_match"
        for row in manifest["source_truth_counts_before_sampling"]
    )
    assert f.loc[f.truth_source == "published_Nmap_event_packet_match", "attack_family"].eq("Scanning").all()
    with pytest.raises(ValueError, match="Overlapping"):
        _prepare(path, rows, confirmed, packet_rows=confirmed)
    rows[1]["OUT_PKTS"] = 2
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match="native single unanswered SYN"):
        _prepare(path, rows, confirmed, packet_rows=packet)


def test_role_identity_sorted_numerically_and_evidence_sensitive():
    f = pd.DataFrame(
        [
            {
                "role": "P_time",
                "row_id": "digest:10",
                "attack_family": "Scanning",
                "truth_source": "reference",
            },
            {"role": "P_time", "row_id": "digest:2", "attack_family": "Benign", "truth_source": "native"},
        ]
    )
    actual = co.role_identity_sha256(f)
    assert actual["P_time"] == hashlib.sha256(b"2\tBenign\tnative\n10\tScanning\treference\n").hexdigest()
    assert actual["C_time"] == hashlib.sha256(b"").hexdigest()
    f.loc[0, "truth_source"] = "different"
    assert co.role_identity_sha256(f)["P_time"] != actual["P_time"]
