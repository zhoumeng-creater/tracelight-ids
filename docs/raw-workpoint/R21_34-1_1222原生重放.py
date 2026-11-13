"""Join published R21 row scores to the entire 34-1 source; diagnose the Dec 22 raw misses."""

from hashlib import sha256
from io import BytesIO
import gzip
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve()
# Allow both a repository docs checkout and the project archive layout.
if (SCRIPT.parents[2] / "src/ids").is_dir():
    REPO = SCRIPT.parents[2]
    default_archive = REPO.parents[1] / "实验归档/20261113_第二十三轮回包状态"
else:
    default_archive = SCRIPT.parents[2]
    REPO = default_archive.parents[1] / "代码/tracelight-ids"
REPO = Path(os.environ.get("TRACELIGHT_REPO", REPO))
ROOT = Path(os.environ.get("TRACELIGHT_R23_ARCHIVE", default_archive))
HERE = Path(os.environ.get("TRACELIGHT_AUDIT_OUTPUT", ROOT / "复核/八字段基线"))
HERE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "src"))
from ids.flow_history.trees import load_arm, score_arm  # noqa: E402
from ids.threshold_validation.observations import (  # noqa: E402
    prepare,
    RAW_FEATURES,
    HISTORY_FEATURES,
)
from ids.threshold_validation.sources import verify_log  # noqa: E402


def pair(df, mask):
    a = df.raw_fixed & mask
    b = df.history_fixed & mask
    return dict(
        rows=int(mask.sum()),
        raw=int(a.sum()),
        history=int(b.sum()),
        gained=int((b & ~a).sum()),
        lost=int((a & ~b).sum()),
        raw_OR_history=int((a | b).sum()),
    )


def overlap(attack, normal, fields):
    keys = pd.MultiIndex.from_frame(attack[list(fields)])
    controls = pd.MultiIndex.from_frame(normal[list(fields)])
    counts = controls.value_counts().reindex(keys, fill_value=0).to_numpy()
    return dict(
        attack_rows=int(len(attack)),
        matching_rows=int((counts > 0).sum()),
        smallest_match_count=int(counts.min()) if len(attack) else None,
        largest_match_count=int(counts.max()) if len(attack) else None,
        distinct_normal_matches=int(controls.isin(keys).sum()),
    )


def modes(frame, fields, size=5):
    return [
        {**dict(zip(fields, k)), "rows": int(n)}
        for k, n in frame[list(fields)].value_counts().head(size).items()
    ]


def main():
    archive = next((ROOT.parent / "20261111_第二十一轮前向阈值").glob("*.zip"))
    name = "CTU-IoT-Malware-Capture-34-1"
    with ZipFile(archive) as z, TemporaryDirectory(dir=HERE) as d:
        src = json.loads(z.read("protocol.json"))["registration"]["sources"]
        spec = next(s for s in src if s["name"] == name)
        source = verify_log(
            REPO / "data/raw/iot23-forward-20261111" / (name + ".conn.log.labeled.gz"),
            spec,
        )
        f = Path(d) / "rows"
        prepare(source, f)
        df = pd.read_parquet(f)
        with gzip.open(source, "rt", encoding="utf8") as stream:
            for line in stream:
                if line.startswith("#fields"):
                    cols = line.split()[1:]
                    break
            add = pd.read_csv(
                stream,
                names=cols,
                sep=r"\s+",
                comment="#",
                usecols=["id.orig_h", "id.resp_h", "history", "conn_state"],
                na_filter=False,
                dtype=str,
                quoting=3,
            )
        assert len(add) == len(df)
        df = df.join(
            add.rename(columns={"id.orig_h": "source", "id.resp_h": "destination"})
        )
        sealed = pd.read_parquet(
            BytesIO(z.read(f"evidence/{name}/row-scores-and-alerts.parquet"))
        )
        for original, predicted in (
            ("row_id", "row_id"),
            ("flow_end_ms", "flow_end_ms"),
            ("family", "attack_family"),
        ):
            assert np.array_equal(df[original].to_numpy(), sealed[predicted].to_numpy())
        for col in ("raw_fixed", "history_fixed", "raw_score", "history_score"):
            df[col] = sealed[col]
        df["day"] = pd.to_datetime(df.flow_end_ms, unit="ms", utc=True).dt.strftime(
            "%Y-%m-%d"
        )
        day = df[df.day == "2018-12-22"]
        normal = day[day.family == "Benign"]
        capture_normal = df[df.family == "Benign"]
        service = ("ip_protocol", "resp_port")
        direction = ("orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes")
        queries = {
            "exact_raw_8": RAW_FEATURES,
            "without_source_port": tuple(s for s in RAW_FEATURES if s != "orig_port"),
            "without_source_port_or_duration": tuple(
                s for s in RAW_FEATURES if s not in ("orig_port", "duration_seconds")
            ),
            "protocol_dest_port_both_packet_counts": (
                *service,
                "orig_pkts",
                "resp_pkts",
            ),
            "protocol_dest_port": service,
            "origin_destination_protocol_port": ("source", "destination", *service),
        }
        out = {
            "source_sha256": spec["sha256"],
            "package_sha256": sha256(archive.read_bytes()).hexdigest(),
            "whole_capture_rows": len(df),
            "all_native_support": df.native_detail.value_counts().to_dict(),
            "scope": "2018-12-22 UTC frozen R21 row order and thresholds, previously scored",
            "by_day": {
                day: {
                    family: pair(period, period.family == family)
                    for family in sorted(period.family.unique())
                }
                for day, period in df.groupby("day")
            },
            "missing_baseline": {},
        }
        for name_ in ("Scanning", "C&C"):
            missed = day[(day.family == name_) & ~day.raw_fixed]
            out["missing_baseline"][name_] = dict(
                same_day_normal={
                    k: overlap(missed, normal, v) for k, v in queries.items()
                },
                same_capture_normal={
                    k: overlap(missed, capture_normal, v) for k, v in queries.items()
                },
                common_services=modes(missed, service),
                common_directions=modes(missed, direction),
                common_source_destinations=modes(missed, ("source", "destination")),
                history_rescued=int(missed.history_fixed.sum()),
                example=missed[
                    [
                        "row_id",
                        "source",
                        "destination",
                        "native_detail",
                        "history",
                        "conn_state",
                        *RAW_FEATURES,
                        *HISTORY_FEATURES,
                        "raw_score",
                        "history_score",
                        "history_fixed",
                    ]
                ]
                .head(3)
                .to_dict("records"),
            )
        out["normal_common_services"] = modes(normal, service, 8)
        out["R21_raw_model_sha256"] = sha256(
            z.read("models/frozen/raw.joblib")
        ).hexdigest()
        with ZipFile(next(ROOT.glob("response-*.zip"))) as r23:
            out["R23_raw_model_sha256"] = sha256(
                r23.read("models/frozen/hgb-raw.joblib")
            ).hexdigest()
            model = Path(d) / "r23.joblib"
            model.write_bytes(r23.read("models/frozen/hgb-raw.joblib"))
            score = score_arm(load_arm(model, threads=2), df, RAW_FEATURES)
            cut = json.loads(r23.read("evidence/calibration/hgb.json"))["raw"][
                "threshold"
            ]
            out["R23_comparison_on_all_34_1"] = {
                "rows": len(df),
                "r23_cut": cut,
                "max_abs_score_delta": float(
                    np.max(np.abs(score - df.raw_score.to_numpy()))
                ),
                "changed_alerts": int(np.sum((score > cut) != df.raw_fixed.to_numpy())),
            }
        out["cc_response_status"] = (
            day[day.family == "C&C"]
            .groupby(["history", "conn_state"])
            .size()
            .reset_index(name="rows")
            .to_dict("records")
        )
        out["cc_responder_packets"] = dict(
            zero=int(((day.family == "C&C") & (day.resp_pkts == 0)).sum()),
            positive=int(((day.family == "C&C") & (day.resp_pkts > 0)).sum()),
        )
        (HERE / "R21_34-1_1222原生重放.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf8"
        )
        print(
            out["by_day"],
            out["all_native_support"],
            out["missing_baseline"]["Scanning"],
            out["missing_baseline"]["C&C"],
        )


if __name__ == "__main__":
    main()
