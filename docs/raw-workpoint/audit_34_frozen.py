"""Replay Round 23 frozen policies on opened Round 21 capture 34-1, including rank ceilings."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from ids.common import read_json, write_json
from ids.raw_workpoint.controls import unpack
from ids.raw_workpoint.evaluate import evaluate
from ids.raw_workpoint.ranking import diagnose
from ids.raw_workpoint.scoring import score_group
from ids.response_evidence.data import prepare_log
from ids.response_evidence.history import enrich_prepared
from ids.route_batch.protocol import REPO
from ids.threshold_validation.sources import verify_log


def run(log, output):
    name = "CTU-IoT-Malware-Capture-34-1"
    reg = read_json(REPO / "configs/threshold-validation-20261111/registration.json")
    spec = next(s for s in reg["sources"] if s["name"] == name)
    verify_log(log, spec)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as work:
        root = Path(work)
        unpack(root)
        base = root / "local/controls"
        blind, target, feat, pred, paired, bounds = [
            root / n for n in ("blind.pq", "truth.pq", "enriched.pq", "pred.pq", "pairs.json", "bounds.json")
        ]
        prepare_log(log, blind, target)
        enrich_prepared(blind, feat, workers=4)
        score_group(
            feat,
            blind,
            base / "models/frozen",
            base / "evidence/calibration",
            "hgb",
            pred,
            threads=4,
            device="cpu",
            batch_rows=65536,
        )
        result = evaluate(pred, target, blind, paired)
        diagnose(pred, target, blind, paired, bounds, 65536)
        write_json(
            output,
            {
                "source_sha256": spec["sha256"],
                "utc_days": result["utc_days"],
                "ranking_ceilings": read_json(bounds)["utc_days"],
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        default=REPO / "data/raw/iot23-forward-20261111/CTU-IoT-Malware-Capture-34-1.conn.log.labeled.gz",
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("r21-34-frozen-r23.json"))
    args = parser.parse_args()
    run(args.log, args.output)
