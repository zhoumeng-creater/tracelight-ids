"""Paths shared by the optional coverage and label provenance reproduction scripts."""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
WORK = (
    Path(os.environ.get("TRACELIGHT_LABEL_AUDIT_DIR", REPO / "local/capture-label-audit"))
    .expanduser()
    .resolve()
)
RESULTS = WORK / "results"
UQ_WORK = WORK / "uq-metadata"
CORRECTED = WORK / "corrected"
PCAP = WORK / "pcap"
UQ_FILENAME = "NF-CSE-CIC-IDS2018-v3-official.zip"
UQ_ZIP = (
    Path(os.environ.get("TRACELIGHT_CIC2018_UQ_ZIP", REPO / "data/raw/nf-cse-cic-ids2018-v3" / UQ_FILENAME))
    .expanduser()
    .resolve()
)
EXPECTED_UQ_ZIP_SHA256 = "7399bdef73d29875ab388a53e2ae7e592d1952c4ef07ae6de195ba133823d97f"

for directory in (RESULTS, UQ_WORK, CORRECTED, PCAP):
    directory.mkdir(exist_ok=True, parents=True)
