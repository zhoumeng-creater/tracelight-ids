"""Independent cache-boundary checks; no model, network, or original data is used."""

from pathlib import Path
import shutil

import pandas as pd
import pytest

from ids.common import canonical_hash, sha256_file, write_json
from ids.readiness_replay import GROUPS
from ids.readiness_replay.inputs import distinct_roots, inspect_previous


def cached_run(tmp_path):
    root = tmp_path / "relocated-previous"
    protocol = {"release": "synthetic-r25-review", "fixed": True}
    binding = canonical_hash(protocol)
    write_json(root / "protocol.json", protocol)
    specs = [{"name": "IoT-SFC-synthetic", "provider": "sfc", "role": "synthetic", "source_id": "sfc-fixed"},
             {"name": "DEDALE-synthetic", "provider": "dedale", "role": "synthetic", "source_id": "dedale-fixed"}]
    registration = {"previous_protocol_sha256": binding, "sources": specs}
    write_json(root / "outcome.json", {"status": "complete", "failures": [],
                                       "successful_prepared_sources": 2, "evaluated_group_sources": 10})

    def save(member, content="synthetic unused content"):
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return member

    def receipt(stage, members):
        write_json(root / "receipts" / (stage + ".json"), {
            "stage": stage, "binding": binding,
            "outputs": {member: sha256_file(root / member) for member in members},
        })

    for spec in specs:
        name, provider = spec["name"], spec["provider"]
        feature = f"local/features/{name}.parquet"
        truth = f"local/providers/{provider}/truth/{name}.parquet"
        for member in (feature, truth):
            path = root / member
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"row_id": [1, 2, 3]}).to_parquet(path, index=False)
        audit = f"evidence/sources/{name}.json"
        write_json(root / audit, {"rows": 3})
        index = f"evidence/sources/{provider}-index.json"
        write_json(root / index, [{**spec, "status": "complete",
                                  "features": "Z:/obsolete-lab-root/no-such-feature.parquet",
                                  "truth": "Z:/obsolete-lab-root/no-such-truth.parquet",
                                  "audit": "Z:/obsolete-lab-root/no-such-audit.json"}])
        receipt("prepare-" + provider, [truth, audit, index])
        receipt("history-" + name, [feature])
        for group in GROUPS:
            receipt(f"score-{name}-{group}", [save(f"local/scores/{name}/{group}.parquet")])
            receipt(f"seal-{name}-{group}", [save(f"local/decisions/{name}/{group}.parquet"),
                                             save(f"evidence/thresholds/{name}/{group}.json")])
            receipt(f"judge-{name}-{group}", [save(f"evidence/evaluation/{name}/{group}/{table}.csv.gz")
                                              for table in ("budget", "paired")])
    return root, registration


def test_relocation_ignores_stale_absolute_paths_but_hash_mismatch_is_source_local(tmp_path):
    root, registration = cached_run(tmp_path)
    good, report = inspect_previous(root, registration=registration, hash_threads=2)
    assert len(good) == 2 and not report["failures"]
    assert all(not Path(source["features"]).is_absolute() for source in good)
    old = root / "local/scores/IoT-SFC-synthetic/hgb.parquet"
    old.write_bytes(b"changed frozen score bytes")
    good, report = inspect_previous(root, registration=registration, hash_threads=2)
    assert [source["name"] for source in good] == ["DEDALE-synthetic"]
    assert len(report["failures"]) == 1
    failure = report["failures"][0]
    assert failure["source"] == "IoT-SFC-synthetic" and failure["stage"] == "cached_inputs"
    assert "SHA-256 differs" in failure["problems"][0]["problem"]


def test_return_summary_without_local_cache_is_explicitly_unusable(tmp_path):
    original, registration = cached_run(tmp_path)
    returned = tmp_path / "returned-summary"
    for path in original.rglob("*"):
        relative = path.relative_to(original)
        if path.is_file() and relative.parts[0] != "local":
            target = returned / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    good, report = inspect_previous(returned, registration=registration)
    assert not good and len(report["failures"]) == 2
    assert all(item["stage"] == "cached_inputs" for item in report["failures"])
    assert all("summary ZIP is insufficient" in item["action"] for item in report["failures"])


@pytest.mark.parametrize("previous,output", [("same", "same"), ("old", "old/new"), ("new/old", "new")])
def test_readonly_previous_and_new_output_must_not_overlap(tmp_path, previous, output):
    with pytest.raises(ValueError, match="separate directory trees"):
        distinct_roots(tmp_path / previous, tmp_path / output)


def test_protocol_changes_reject_before_cache_work(tmp_path):
    root, registration = cached_run(tmp_path)
    write_json(root / "protocol.json", {"release": "different-source-run"})
    with pytest.raises(ValueError, match="registered completed R25 protocol"):
        inspect_previous(root, registration=registration)
