"""Reporting must retain failed operating points and verify transmitted predictions."""
import copy
from pathlib import Path
import zipfile

import pytest

from ids.common import sha256_file
from ids.evaluation import evaluate_binary
from workflows.stage5 import reporting


def test_paired_keeps_actual_test_errors_and_signed_family_differences():
    labels = [0, 0, 1, 1, 1]
    families = ["Normal", "Normal", "Fuzzers", "Fuzzers", "Generic"]
    reference = evaluate_binary(labels, [.1, .2, .8, .4, .9], .5, families)
    candidate = evaluate_binary(labels, [.7, .3, .8, .7, .2], .5, families)
    delta = reporting._paired(candidate, reference)
    assert reporting._over_budget(candidate, .01) is True
    assert reporting._over_budget(reference, .01) is False
    assert reporting._over_budget(None, .01) is None
    assert delta["delta_tp"] == 0 and delta["delta_fp"] == 1
    assert delta["delta_pp"]["normal_fpr"] == 50 and delta["delta_pp"]["attack_recall"] == 0
    assert [f["delta_detected"] for f in delta["families"]] == [1, -1]
    assert reporting._paired(None, None) is None
    changed = copy.deepcopy(candidate)
    changed["attack_categories"][0]["n_attacks"] += 1
    with pytest.raises(ValueError, match="denominators"):
        reporting._paired(changed, reference)


def test_undefined_seed_does_not_turn_into_improvement():
    assert reporting._stats([.3, None, .4]) == {"n_defined": 2, "n_planned": 3,
                                               "mean": None, "sample_sd": None}


def test_package_resume_recovers_after_sidecar_gap_and_rejects_corruption(tmp_path):
    p = tmp_path / "scores"
    p.mkdir()
    (p / "sample").write_bytes(b"sealed predictions")
    inv = {"scores/sample": sha256_file(p / "sample")}
    output = reporting._archive(tmp_path, inv)
    sidecar = Path(str(output) + ".sha256")
    assert sidecar.read_text().strip() == sha256_file(output)
    with zipfile.ZipFile(output) as z:
        assert z.read("MANIFEST.sha256").decode() == inv["scores/sample"] + "  scores/sample\n"
    sidecar.unlink()
    reporting._archive(tmp_path, inv)
    assert sidecar.read_text().strip() == sha256_file(output)
    with zipfile.ZipFile(output, "a") as z:
        z.writestr("surprise", b"extra")
    with pytest.raises(ValueError, match="inventory"):
        reporting._archive(tmp_path, inv)


def test_report_resume_never_overwrites_modified_artifact(tmp_path):
    p = tmp_path / "reports" / "index.html"
    reporting._publish(p, b"first")
    reporting._publish(p, b"first")
    with pytest.raises(ValueError, match="differs"):
        reporting._publish(p, b"second")
    assert p.read_bytes() == b"first"
