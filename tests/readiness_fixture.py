"""Small R25-format artifacts; no trained estimator or original dataset required."""
from pathlib import Path

import numpy as np
import pandas as pd

from ids.behavior_transfer.decisions import COMPONENTS, decide
from ids.behavior_transfer.evaluation import judge
from ids.common import canonical_hash, sha256_file, write_json
from ids.readiness_replay import GROUPS


def make_previous(directory, *, rows=650):
    root = Path(directory)
    root.mkdir(parents=True)
    protocol = {"release": "behavior-transfer-lab-v1", "synthetic": True, "components": COMPONENTS}
    binding = canonical_hash(protocol)
    write_json(root / "protocol.json", protocol)
    sources = []
    written = []

    def receipt(stage, files):
        write_json(root / "receipts" / (stage + ".json"), {
            "stage": stage, "binding": binding,
            "outputs": {str(p.relative_to(root).as_posix()): sha256_file(p) for p in files}})

    for number, provider in enumerate(("sfc", "dedale")):
        name = "synthetic-" + provider
        spec = dict(name=name, provider=provider, role="synthetic_opened", source_id=name)
        sources.append(spec)
        time = np.r_[0, 1_000, 3_600_000 + np.arange(rows - 2) * 1000] + number * 86_400_000
        frame = pd.DataFrame({"row_id": np.arange(1, rows + 1), "flow_end_ms": time,
                              "uid": [name + "-" + str(i) for i in range(rows)],
                              "ip_protocol": np.repeat(6, rows), "resp_port": np.repeat(443, rows),
                              "duration_seconds": np.ones(rows), "orig_ip_bytes": np.arange(rows) + 40,
                              "hist_dst_ports_60s": np.arange(rows) % 9,
                              "hist_dst_ips_60s": np.arange(rows) % 5})
        feature = root / "local/features" / (name + ".parquet")
        feature.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(feature, index=False)
        truth = pd.DataFrame({"row_id": frame.row_id, "uid": frame.uid,
                              "window": "2025-01-0" + str(number + 1),
                              "family": ["Benign"] * (rows - 5) + ["Download", "ExternalC2", "Scanning", "OtherAttack", "Unresolved"],
                              "native_detail": ["normal"] * (rows - 5) + ["file", "poll", "scan", "camera", "related"],
                              "event_id": [""] * (rows - 5) + ["download", "control", "scan", "other", "related"],
                              "evidence_level": "synthetic_label", "exposure": "synthetic_opened"})
        target = root / f"local/providers/{provider}/truth/{name}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        truth.to_parquet(target, index=False)
        audit = root / f"evidence/sources/{name}.json"
        write_json(audit, {"rows": rows})
        index = root / f"evidence/sources/{provider}-index.json"
        write_json(index, [{**spec, "status": "complete", "features": "Z:/old-computer/unused.parquet",
                            "truth": "Z:/old-computer/unused-truth.parquet"}])
        receipt("prepare-" + provider, [target, audit, index])
        receipt("history-" + name, [feature])
        proxy = root / "fixture-proxy" / (name + ".parquet")
        proxy.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"row_id": frame.row_id, "scan_proxy": np.zeros(rows, dtype=bool)}).to_parquet(proxy)
        calibration = pd.DataFrame({**{key: np.linspace(-3, 3, 600) for key in ("score_raw", "score_contact", "score_quality")},
                                    "hist_dst_ports_60s": np.zeros(600), "hist_dst_ips_60s": np.zeros(600)})
        for group in GROUPS:
            score = pd.DataFrame({"row_id": frame.row_id,
                                  **{v: np.arange(rows) % 17 > 12 for v in ("raw", "contact", "quality", "add_quality")},
                                  **{"score_" + v: np.arange(rows) % 17 / 2 - 3 for v in ("raw", "contact", "quality")}})
            score_path = root / f"local/scores/{name}/{group}.parquet"
            score_path.parent.mkdir(parents=True, exist_ok=True)
            score.to_parquet(score_path, index=False)
            receipt("score-" + name + "-" + group, [score_path])
            prediction, threshold = decide(score_path, feature, proxy, calibration,
                                            root / f"local/decisions/{name}/{group}.parquet",
                                            root / f"evidence/thresholds/{name}/{group}.json")
            receipt("seal-" + name + "-" + group, [prediction, threshold])
            files = judge(prediction, target, feature, root / f"evidence/evaluation/{name}/{group}",
                          source=spec, group=group, scores_file=score_path)
            receipt("judge-" + name + "-" + group, files)
        written.extend([feature, target])
    config = {"previous_protocol_sha256": binding, "sources": sources}
    write_json(root / "outcome.json", {"status": "complete", "failures": [],
                                       "successful_prepared_sources": len(sources),
                                       "evaluated_group_sources": len(sources) * len(GROUPS)})
    return root, config
